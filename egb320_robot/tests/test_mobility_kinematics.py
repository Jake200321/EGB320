"""Hardware-free cover for the DFR0592 driver's maths (runs on a laptop).

Everything that touches I2C lives in DFR0592Mobility; these two functions are the
part that can be wrong silently, so they get tested.
"""

import math

import pytest

from egb320.config import Config
from egb320.interfaces.messages import VelocityCommand
from egb320.mobility.dfr0592 import velocity_to_wheel_rpm, wheel_rpm_to_duty

CFG = Config()
TRACK = CFG.geometry.track_spacing_m
CIRC = math.pi * CFG.geometry.sprocket_pitch_diameter_m
MAX_RPM = 200.0


def rpm(v, w, max_rpm=MAX_RPM, turn_gain=1.0):
    return velocity_to_wheel_rpm(VelocityCommand(v, w), TRACK, CIRC, max_rpm, turn_gain)


def test_stop_is_zero():
    assert rpm(0.0, 0.0) == (0.0, 0.0)


def test_straight_ahead_drives_both_wheels_equally():
    left, right = rpm(0.15, 0.0)
    assert left == pytest.approx(right)
    # 0.15 m/s on a 24 mm pitch-diameter sprocket
    assert left == pytest.approx(0.15 / CIRC * 60.0)


def test_positive_yaw_turns_left_meaning_right_wheel_runs_faster():
    left, right = rpm(0.10, 0.5)
    assert right > left


def test_spin_on_the_spot_is_equal_and_opposite():
    left, right = rpm(0.0, 1.0)
    assert left == pytest.approx(-right)


def test_reverse_is_negative_on_both_wheels():
    left, right = rpm(-0.10, 0.0)
    assert left < 0 and right < 0


def test_saturation_scales_both_wheels_and_preserves_turn_radius():
    """Clipping wheels independently would change the turn radius, not just the speed."""
    v, w = 5.0, 2.0                     # deliberately far past what the drivetrain can do
    unsat_l, unsat_r = rpm(v, w, max_rpm=1e9)
    sat_l, sat_r = rpm(v, w)
    assert max(abs(sat_l), abs(sat_r)) == pytest.approx(MAX_RPM)
    assert sat_l / sat_r == pytest.approx(unsat_l / unsat_r)


def test_turn_gain_scales_only_the_angular_term():
    left_a, right_a = rpm(0.10, 0.5, turn_gain=1.0)
    left_b, right_b = rpm(0.10, 0.5, turn_gain=2.0)
    assert (left_a + right_a) == pytest.approx(left_b + right_b)   # same forward speed
    assert (right_b - left_b) == pytest.approx(2.0 * (right_a - left_a))


def test_duty_zero_below_the_deadband():
    assert wheel_rpm_to_duty(0.0, 25.0, 80.0, MAX_RPM) == 0.0
    assert wheel_rpm_to_duty(0.5, 25.0, 80.0, MAX_RPM) == 0.0     # < 1 % of full scale


def test_duty_starts_at_min_duty_not_at_zero():
    """A creeping command must still clear stiction rather than stall the geartrain."""
    duty = wheel_rpm_to_duty(4.0, 25.0, 80.0, MAX_RPM)
    assert 25.0 <= duty < 30.0


def test_duty_is_capped_at_max_duty():
    assert wheel_rpm_to_duty(MAX_RPM * 10, 25.0, 80.0, MAX_RPM) == pytest.approx(80.0)


def test_duty_sign_follows_rpm_sign():
    assert wheel_rpm_to_duty(-100.0, 25.0, 80.0, MAX_RPM) < 0


# --------------------------------------------------------------------------- driver, faked board
class FakeBoard:
    """Stands in for DFRobot_DC_Motor_IIC so the driver itself can be exercised off-Pi."""
    STA_OK, ALL, CW, CCW, M1, M2 = 0x00, 0xffffffff, 0x01, 0x02, 0x01, 0x02

    def __init__(self, bus_id, addr):
        self.bus_id, self.addr = bus_id, addr
        self.calls = []
        self.rpm = {1: 0.0, 2: 0.0}

    def begin(self):
        return self.STA_OK

    def set_encoder_enable(self, i): self.calls.append(("enc_on", i))
    def set_encoder_reduction_ratio(self, i, r): self.calls.append(("ratio", i, r))
    def set_moter_pwm_frequency(self, f): self.calls.append(("freq", f))
    def motor_stop(self, i): self.calls.append(("stop", i))
    def motor_movement(self, i, orient, speed): self.calls.append(("move", i, orient, speed))
    def get_encoder_speed(self, i): return [self.rpm[1], self.rpm[2]]


@pytest.fixture
def driver(monkeypatch):
    from egb320.mobility import dfr0592
    monkeypatch.setattr(dfr0592, "load_board_class", lambda: FakeBoard)
    cfg = Config()
    cfg.drive.watchdog_timeout_s = 0.0          # no background thread inside a test
    cfg.drive.max_wheel_rpm = MAX_RPM
    d = dfr0592.DFR0592Mobility(cfg)
    d.board.calls.clear()
    return d


def test_driver_configures_the_board_on_init(monkeypatch):
    from egb320.mobility import dfr0592
    monkeypatch.setattr(dfr0592, "load_board_class", lambda: FakeBoard)
    cfg = Config()
    cfg.drive.watchdog_timeout_s = 0.0
    d = dfr0592.DFR0592Mobility(cfg)
    kinds = [c[0] for c in d.board.calls]
    assert kinds == ["enc_on", "ratio", "freq", "stop"]
    assert ("ratio", FakeBoard.ALL, 50) in d.board.calls   # gear ratio pushed to the board


def test_forward_command_drives_both_channels_the_same_physical_way(driver):
    driver.set_velocity(VelocityCommand(0.10, 0.0))
    moves = {c[1]: c for c in driver.board.calls if c[0] == "move"}
    assert set(moves) == {1, 2}
    # right channel is inverted in config, so forward must come out as opposite orientations
    assert moves[1][2] != moves[2][2]
    assert moves[1][3] == pytest.approx(moves[2][3])       # equal duty -> straight line


def test_stop_command_stops_rather_than_driving_at_min_duty(driver):
    driver.set_velocity(VelocityCommand.STOP)
    assert [c for c in driver.board.calls if c[0] == "move"] == []
    assert len([c for c in driver.board.calls if c[0] == "stop"]) == 2


def test_stop_never_raises_even_when_the_bus_is_dead(driver):
    def boom(_):
        raise OSError("i2c bus went away")
    driver.board.motor_stop = boom
    driver.stop()                                          # must not propagate


def test_ticks_integrate_from_rpm_in_the_right_direction(driver):
    t = [0.0]
    driver._clock = lambda: t[0]
    driver.reset_encoders()
    driver.read_encoders()                                 # baseline sample
    driver.board.rpm = {1: 60.0, 2: 60.0}                  # 1 rev/s on both sides
    t[0] = 1.0
    s = driver.read_encoders()
    assert s.left_ticks == pytest.approx(CFG.geometry.counts_per_output_rev, rel=1e-6)
    assert s.right_ticks == pytest.approx(CFG.geometry.counts_per_output_rev, rel=1e-6)


def test_unsigned_encoder_mode_takes_direction_from_the_last_command(driver):
    """If the board only reports magnitude, reversing must still decrement ticks."""
    driver.cfg.drive.encoder_reports_signed = False
    t = [0.0]
    driver._clock = lambda: t[0]
    driver.reset_encoders()
    driver.read_encoders()
    driver.set_velocity(VelocityCommand(-0.10, 0.0))       # reverse
    driver.board.rpm = {1: 60.0, 2: 60.0}                  # magnitude only, no sign
    t[0] = 1.0
    s = driver.read_encoders()
    assert s.left_ticks < 0 and s.right_ticks < 0


def test_read_odometry_is_none_because_the_hat_integrates_nothing(driver):
    assert driver.read_odometry() is None


def test_wall_ranges_report_none_rather_than_a_made_up_distance(driver):
    r = driver.read_wall_ranges()
    assert (r.left, r.front, r.right) == (None, None, None)


def test_watchdog_stops_the_motors_when_the_control_loop_dies(monkeypatch):
    """A tracked robot whose nav loop has crashed must not keep driving."""
    import time as _time
    from egb320.mobility import dfr0592
    monkeypatch.setattr(dfr0592, "load_board_class", lambda: FakeBoard)
    cfg = Config()
    cfg.drive.watchdog_timeout_s = 0.2
    d = dfr0592.DFR0592Mobility(cfg)
    d.board.calls.clear()                     # __init__ legitimately stops ALL
    try:
        d.set_velocity(VelocityCommand(0.10, 0.0))
        _time.sleep(0.1)
        assert not any(c[0] == "stop" and c[1] == FakeBoard.ALL
                       for c in d.board.calls), "watchdog fired too early"
        _time.sleep(0.5)                      # now stop feeding it commands
        assert any(c[0] == "stop" and c[1] == FakeBoard.ALL for c in d.board.calls)
    finally:
        d.close()
