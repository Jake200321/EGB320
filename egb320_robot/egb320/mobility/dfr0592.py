"""DFRobot DFR0592 motor HAT -> MobilityInterface. Raspberry Pi only.

BOARD LIMITATION (confirmed against the DFRobot library source, 2026-09-09):
the DFR0592 exposes `get_encoder_speed()` -> RPM and nothing else. There is **no
cumulative tick register**. So `read_encoders()` integrates RPM in software: ticks
are a derived quantity and they drift, because every sample is a rectangle-rule
approximation of whatever the wheel did between polls. That settles the open
question in `mobility/README.md` -- pose has to be re-anchored against walls or
markers, not dead-reckoned across the whole maze.

Bench order (do not skip -- the duty mapping below is only as good as these):
    tools/motor_spin_test.py    do the motors turn, and which way is forward?
    tools/bench_motor_rpm.py    fills in drive.max_wheel_rpm, min_duty_percent, encoder scale

Watchdog: if `set_velocity()` isn't called for `drive.watchdog_timeout_s`, the motors
stop themselves. A tracked robot whose control loop has died should not keep driving.
Call `set_velocity()` every tick (the nav loop already does, at 20 Hz).
"""

from __future__ import annotations

import math
import shlex
import sys
import threading
import time
from pathlib import Path

from ..config import Config
from ..interfaces.messages import EncoderSample, Pose2D, VelocityCommand, WallRanges

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def load_board_class():
    """Import DFRobot_DC_Motor_IIC. The library is not on PyPI, so we look in a
    vendored clone first and fall back to whatever is on sys.path."""
    def _try():
        from DFRobot_RaspberryPi_DC_Motor import DFRobot_DC_Motor_IIC
        return DFRobot_DC_Motor_IIC

    try:
        return _try()
    except ImportError:
        pass

    # The clone puts DFRobot_RaspberryPi_DC_Motor.py at its root, but glob anyway so a
    # nested checkout still works.
    for found in VENDOR_DIR.rglob("DFRobot_RaspberryPi_DC_Motor.py"):
        parent = str(found.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        break

    try:
        return _try()
    except ImportError as exc:
        raise ImportError(
            "DFRobot motor HAT library not found. It is not on PyPI -- clone it:\n"
            "    git clone https://github.com/DFRobot/DFRobot_RaspberryPi_Motor "
            f"{shlex.quote(str(VENDOR_DIR))}\n"
            "(the space before the destination is meant to be there -- "
            "git clone takes <url> then <destination>)"
        ) from exc


# --------------------------------------------------------------------------- pure kinematics
# Kept at module level, free of hardware, so tests/test_mobility_kinematics.py can
# cover the maths on a laptop.

def velocity_to_wheel_rpm(cmd: VelocityCommand, track_m: float, circumference_m: float,
                          max_wheel_rpm: float, turn_gain: float = 1.0) -> tuple:
    """(v m/s, w rad/s) -> (left_rpm, right_rpm) at the output shaft, signed.

    Differential drive with +w = turn left, so the RIGHT wheel runs faster in a left
    turn. If either wheel saturates, BOTH are scaled by the same factor -- clipping
    them independently would quietly change the turn radius instead of just the speed.
    """
    half_track = track_m / 2.0
    w = cmd.w_rps * turn_gain
    v_left = cmd.v_mps - w * half_track
    v_right = cmd.v_mps + w * half_track

    rpm_left = v_left / circumference_m * 60.0
    rpm_right = v_right / circumference_m * 60.0

    peak = max(abs(rpm_left), abs(rpm_right))
    if peak > max_wheel_rpm > 0:
        scale = max_wheel_rpm / peak
        rpm_left *= scale
        rpm_right *= scale
    return rpm_left, rpm_right


def wheel_rpm_to_duty(rpm: float, min_duty: float, max_duty: float,
                      max_wheel_rpm: float) -> float:
    """Signed output-shaft RPM -> signed PWM duty %, with deadband compensation.

    Anything below ~1 % of full scale is treated as a stop request; everything else is
    mapped into [min_duty, max_duty], because duty below min_duty just stalls the
    geartrain and heats the motor rather than turning it.
    """
    if max_wheel_rpm <= 0:
        return 0.0
    frac = abs(rpm) / max_wheel_rpm
    if frac < 0.01:
        return 0.0
    frac = min(frac, 1.0)
    return math.copysign(min_duty + frac * (max_duty - min_duty), rpm)


# --------------------------------------------------------------------------- driver
class DFR0592Mobility:
    """MobilityInterface backed by the DFR0592 HAT.

    Ultrasonics are NOT handled here yet -- `read_wall_ranges()` returns all-None until
    `cfg.hardware.ultrasonic_pins` is filled in and a ranger is wired up. That is the
    honest answer for now rather than a fabricated distance.
    """

    def __init__(self, cfg: Config, clock=time.monotonic):
        self.cfg = cfg
        self._clock = clock
        self._geom = cfg.geometry
        self._drive = cfg.drive
        self._circumference_m = math.pi * self._geom.sprocket_pitch_diameter_m
        self._counts_per_rev = self._geom.counts_per_output_rev

        board_cls = load_board_class()
        self.board = board_cls(cfg.hardware.i2c_bus, cfg.hardware.motor_hat_i2c_addr)

        for attempt in range(5):
            status = self.board.begin()
            if status == self.board.STA_OK:
                break
            time.sleep(0.3)
        else:
            raise RuntimeError(
                f"DFR0592 not responding on i2c-{cfg.hardware.i2c_bus} at "
                f"0x{cfg.hardware.motor_hat_i2c_addr:02x} (status {status}). "
                "Check `sudo i2cdetect -y 1`, the HAT seating, and that the 7-12 V "
                "motor rail is connected -- the board needs it to enumerate reliably."
            )

        self.board.set_encoder_enable(self.board.ALL)
        self.board.set_encoder_reduction_ratio(self.board.ALL, int(self._geom.gear_ratio))
        self.board.set_moter_pwm_frequency(self._drive.pwm_frequency_hz)
        self.board.motor_stop(self.board.ALL)

        self._left_id = cfg.hardware.motor_left_id
        self._right_id = cfg.hardware.motor_right_id
        self._left_invert = cfg.hardware.motor_left_invert
        self._right_invert = cfg.hardware.motor_right_invert

        # odometry state (integrated from RPM -- see module docstring)
        self._ticks_left = 0.0
        self._ticks_right = 0.0
        self._last_enc_t = None
        self._last_dir = {self._left_id: 1, self._right_id: 1}

        self.last_cmd = VelocityCommand.STOP
        self._last_cmd_t = self._clock()
        self._closed = False
        self._lock = threading.Lock()

        self._watchdog = None
        if self._drive.watchdog_timeout_s > 0:
            self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog.start()

    # ----------------------------------------------------------------- commands
    def set_velocity(self, cmd: VelocityCommand) -> None:
        rpm_l, rpm_r = velocity_to_wheel_rpm(
            cmd, self._geom.track_spacing_m, self._circumference_m,
            self._drive.max_wheel_rpm, self._drive.turn_gain)
        duty_l = wheel_rpm_to_duty(rpm_l, self._drive.min_duty_percent,
                                   self._drive.max_duty_percent, self._drive.max_wheel_rpm)
        duty_r = wheel_rpm_to_duty(rpm_r, self._drive.min_duty_percent,
                                   self._drive.max_duty_percent, self._drive.max_wheel_rpm)

        with self._lock:
            self.last_cmd = cmd
            self._last_cmd_t = self._clock()
            self._drive_channel(self._left_id, duty_l, self._left_invert)
            self._drive_channel(self._right_id, duty_r, self._right_invert)

    def _drive_channel(self, motor_id: int, duty: float, invert: bool) -> None:
        if duty == 0.0:
            self.board.motor_stop(motor_id)
            self._last_dir[motor_id] = 0
            return
        forward = duty > 0
        self._last_dir[motor_id] = 1 if forward else -1
        if invert:
            forward = not forward
        orientation = self.board.CW if forward else self.board.CCW
        self.board.motor_movement(motor_id, orientation, abs(duty))

    def stop(self) -> None:
        """Never raises -- this is the last thing called on every failure path."""
        try:
            with self._lock:
                self.last_cmd = VelocityCommand.STOP
                self._last_cmd_t = self._clock()
                self._last_dir = {self._left_id: 0, self._right_id: 0}
                self.board.motor_stop(self.board.ALL)
        except Exception as exc:            # noqa: BLE001 - deliberate: stop() must not throw
            print(f"[mobility] stop() failed: {exc}")

    def close(self) -> None:
        self._closed = True
        self.stop()

    def _watchdog_loop(self) -> None:
        timeout = self._drive.watchdog_timeout_s
        tripped = False
        while not self._closed:
            time.sleep(timeout / 4.0)
            idle = self._clock() - self._last_cmd_t
            moving = any(self._last_dir.values())
            if idle > timeout and moving and not tripped:
                print(f"[mobility] WATCHDOG: no command for {idle:.2f}s -- stopping motors")
                self.stop()
                tripped = True
            elif idle <= timeout:
                tripped = False

    # ----------------------------------------------------------------- feedback
    def read_encoders(self) -> EncoderSample:
        """Cumulative signed ticks, integrated from the board's RPM readings.

        Derived, not measured -- the HAT has no tick register. Accuracy depends on how
        regularly this is polled, so call it once per control tick, not sporadically.
        """
        now = self._clock()
        try:
            speeds = self.board.get_encoder_speed(self.board.ALL)
        except Exception as exc:            # noqa: BLE001
            print(f"[mobility] encoder read failed: {exc}")
            return EncoderSample(now, int(self._ticks_left), int(self._ticks_right))

        if self._last_enc_t is None:        # first call establishes the baseline only
            self._last_enc_t = now
            return EncoderSample(now, int(self._ticks_left), int(self._ticks_right))

        dt = now - self._last_enc_t
        self._last_enc_t = now

        self._ticks_left += self._rpm_to_ticks(speeds, self._left_id, dt)
        self._ticks_right += self._rpm_to_ticks(speeds, self._right_id, dt)
        return EncoderSample(now, int(self._ticks_left), int(self._ticks_right))

    def _rpm_to_ticks(self, speeds, motor_id: int, dt: float) -> float:
        rpm = float(speeds[motor_id - 1]) * self._drive.encoder_rpm_scale
        if not self._drive.encoder_reports_signed:
            rpm = abs(rpm) * self._last_dir.get(motor_id, 0)
        return rpm / 60.0 * self._counts_per_rev * dt

    def reset_encoders(self) -> None:
        self._ticks_left = 0.0
        self._ticks_right = 0.0
        self._last_enc_t = None

    def read_odometry(self):
        """None -- the HAT integrates nothing, so nav owns pose (navigation/localisation.py)."""
        return None

    def read_wall_ranges(self) -> WallRanges:
        # TODO: ultrasonics. Blocked on cfg.hardware.ultrasonic_pins being filled in
        # once the HAT's passthrough pin layout is known.
        return WallRanges(self._clock(), None, None, None)
