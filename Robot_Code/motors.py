#!/usr/bin/env python3
"""Motor driver -- wraps the unit's Controller board (0x57 on i2c-8).

Built against the real controller.py (now in the repo root), not guessed at. The
things that matter, from its own source:

  set_raw_motor_speed(l, r)   -127..127, and it RAISES outside that range.
                              None for both = standby. This is open loop, the
                              board's PID is not involved.
  set_motor_speed(l, r)       closed loop, ticks per 1/100 s, regulated by the
                              board's PID. Better for driving straight -- not used
                              yet, see the note at the bottom.
  get_encoder_ticks()         "since the last time it was queried" -- a delta, and
                              it overflows int16 if you don't poll often enough.
  get_raw_encoder_ticks()     absolute uint16 with wraparound.
  new_relative() /            wraparound-safe deltas. This is what we accumulate,
  get_relative_encoder_ticks()  because it is unambiguous where the two above
                              disagree with how the unit's own example uses them.
  set_motor_shutdown_timeout(s)  the BOARD's own watchdog, 0.1-10 s. Used instead
                              of a Python thread -- it keeps working even if this
                              process dies, which is the case that matters.

Everything above I2C -- differential-drive kinematics, the speed mapping, odometry
accumulation -- lives here so nav never talks to the board directly.
"""

import math
import os
import sys
import time

# --- board / geometry -------------------------------------------------------
I2C_ADDR = 0x57
I2C_BUS = 8

TRACK_M = 0.123              # Pololu 30T sprocket centre-to-centre
SPROCKET_CIRCUM_M = math.pi * 0.024
MAX_WHEEL_RPM = 200.0        # MEASURE -- output-shaft RPM at MAX_SPEED_RAW

# Raw speed is -127..127 (the board rejects anything outside). The unit's own
# motor_control_test.py runs 100, so that's known to work on this hardware.
SPEED_LIMIT = 127
MAX_SPEED_RAW = 90           # what "full speed" maps to. 6 V motors -- keep headroom
MIN_SPEED_RAW = 30           # below this the geartrain won't break stiction. MEASURE

BOARD_WATCHDOG_S = 0.5       # board stops the motors if no command for this long


def _candidate_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    env = os.environ.get("EGB320_EXAMPLES")
    dirs = []
    if env:
        dirs += [env, os.path.join(env, "motor_controller")]
    dirs += [
        repo,                                        # where the examples were pushed
        here,                                        # Robot_Code/
        os.path.join(here, "motor_controller"),
        os.path.join(repo, "motor_controller"),
        os.path.expanduser("~/EGB320_Examples"),
        os.path.expanduser("~/EGB320_Examples/motor_controller"),
        "/home/egb320/EGB320_Examples/motor_controller",
        os.getcwd(),
    ]
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def load_controller_class():
    """Import Controller from wherever the unit's examples happen to live."""
    for d in _candidate_dirs():
        if os.path.exists(os.path.join(d, "controller.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            from controller import Controller
            return Controller
    raise ImportError(
        "controller.py not found. It ships with EGB320_Examples.\n"
        "Put it in the repo root or Robot_Code/, or set EGB320_EXAMPLES.\n"
        "Looked in:\n  " + "\n  ".join(_candidate_dirs())
    )


class MotorDriver:
    """Differential drive on the unit's controller board.

    set_velocity(v, w) is all nav calls: v forward m/s, w yaw rad/s, POSITIVE = LEFT.
    """

    def __init__(self, bus=I2C_BUS, addr=I2C_ADDR, max_speed=MAX_SPEED_RAW):
        Controller = load_controller_class()
        if addr != Controller.I2C_ADDR:
            Controller.I2C_ADDR = addr
        # Raises WhoAmIMismatch or FirmwareVersionMismatch -- both worth surfacing
        # as-is rather than swallowing; a wrong board should not look like no board.
        self.board = Controller(i2c_bus=bus)
        self.max_speed = min(abs(max_speed), SPEED_LIMIT)
        self.last = (0, 0)

        # The board watchdog beats a Python one: it still fires if this process is
        # killed, which is exactly when a robot running away matters most.
        try:
            self.board.set_motor_shutdown_timeout(BOARD_WATCHDOG_S)
        except Exception as exc:                     # noqa: BLE001
            print(f"[drive] could not set board watchdog: {exc}")

        # Odometry: accumulate wraparound-safe deltas into a real running total.
        self._relative = self.board.new_relative()
        self.ticks = [0, 0]

        print(f"[drive] unit controller 0x{addr:02x} on i2c-{bus}, "
              f"firmware {'.'.join(map(str, self.board.get_firmware_version()))}, "
              f"max raw speed {self.max_speed}")

    # -- commands ------------------------------------------------------------
    def set_velocity(self, v_mps, w_rps):
        left, right = wheel_speeds(v_mps, w_rps)
        self.set_raw(to_raw(left, self.max_speed), to_raw(right, self.max_speed))

    def set_raw(self, left, right):
        """Clamped to the board's -127..127 -- outside it, controller.py raises."""
        left = max(-SPEED_LIMIT, min(SPEED_LIMIT, int(left)))
        right = max(-SPEED_LIMIT, min(SPEED_LIMIT, int(right)))
        self.last = (left, right)
        self.board.set_raw_motor_speed(left, right)

    def stop(self):
        """Never raises -- called on every exit path."""
        self.last = (0, 0)
        try:
            self.board.set_raw_motor_speed(0, 0)
        except Exception as exc:                     # noqa: BLE001
            print(f"[drive] stop failed: {exc}")

    def close(self):
        self.stop()
        try:
            self.board.standby()
        except Exception:                            # noqa: BLE001
            pass

    # -- feedback ------------------------------------------------------------
    def read_encoders(self):
        """(left, right) cumulative ticks since this driver started, or None.

        Accumulated from get_relative_encoder_ticks() rather than read straight from
        get_encoder_ticks(): that one is documented as returning the count since the
        last query, so treating it as an absolute total would silently produce
        nonsense the moment anything else polled the board.
        """
        try:
            dl, dr = self.board.get_relative_encoder_ticks(self._relative)
        except Exception as exc:                     # noqa: BLE001
            print(f"[drive] encoder read failed: {exc}")
            return None
        self.ticks[0] += dl
        self.ticks[1] += dr
        return tuple(self.ticks)

    def status(self):
        try:
            return self.board.get_status()
        except Exception:                            # noqa: BLE001
            return None


# --- pure maths, testable without hardware ----------------------------------
def wheel_speeds(v_mps, w_rps, track_m=TRACK_M):
    """(v, w) -> (left, right) wheel speeds in m/s. +w = turn LEFT."""
    half = track_m / 2.0
    return v_mps - w_rps * half, v_mps + w_rps * half


def to_raw(wheel_mps, max_speed=MAX_SPEED_RAW, max_wheel_rpm=MAX_WHEEL_RPM,
           min_speed=MIN_SPEED_RAW):
    """Wheel speed in m/s -> the signed -127..127 the board wants.

    Deadband-compensated: under ~1% of full scale is a stop, everything else lands in
    [min_speed, max_speed], because a value below min_speed stalls the geartrain and
    heats the motor rather than turning it.
    """
    max_mps = max_wheel_rpm / 60.0 * SPROCKET_CIRCUM_M
    if max_mps <= 0:
        return 0
    frac = abs(wheel_mps) / max_mps
    if frac < 0.01:
        return 0
    frac = min(frac, 1.0)
    raw = min_speed + frac * (max_speed - min_speed)
    return int(math.copysign(min(raw, SPEED_LIMIT), wheel_mps))


# NOTE for later: the board also does closed-loop speed via set_motor_speed(), in
# ticks per 1/100 s, regulated by its own PID (tunable with set_pid_coefficients).
# That would hold a straight line far better than open-loop raw PWM, which drifts
# whenever the two motors differ. Worth moving to once MAX_WHEEL_RPM is measured and
# there's a tick-per-metre figure to convert against.
