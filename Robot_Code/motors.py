#!/usr/bin/env python3
"""Motor driver -- wraps the unit's Controller board (0x57 on i2c-8).

The board is the one EGB320_Examples/motor_controller/controller.py drives, NOT a
DFRobot HAT. It speaks its own protocol (a WHO_AM_I register, set_raw_motor_speed,
get_encoder_ticks), so this imports their Controller rather than reimplementing it --
their driver is known-good on this hardware and guessing at register commands would
be a slower way to get something worse.

controller.py is not in this repo (EGB320_Examples ships separately), so we look for
it in the usual places. Put a copy in Robot_Code/motor_controller/ to make this
self-contained for the whole team.

Everything above the I2C layer -- differential-drive kinematics, the duty cap, the
watchdog -- lives here so nav never talks to the board directly.
"""

import math
import os
import sys
import threading
import time

# --- board / geometry -------------------------------------------------------
I2C_ADDR = 0x57              # the unit's controller, confirmed working on the robot
I2C_BUS = 8                  # /dev/i2c-8

TRACK_M = 0.123              # Pololu 30T sprocket centre-to-centre
SPROCKET_CIRCUM_M = math.pi * 0.024
MAX_WHEEL_RPM = 200.0        # MEASURE -- output-shaft RPM at MAX_SPEED_RAW

# set_raw_motor_speed()'s units are NOT documented anywhere we have. This is the
# magnitude sent for "full speed", and everything scales against it. Start low --
# the N20s are 6 V motors -- and raise it once the bench test shows what actually
# moves. If the board takes 0-100 this is 40%; if it takes 0-255 it's 16%.
MAX_SPEED_RAW = 40
MIN_SPEED_RAW = 12           # below this the geartrain won't break stiction. MEASURE

WATCHDOG_TIMEOUT_S = 0.5     # no command for this long -> stop. 0 disables.

_SEARCH_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "motor_controller"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "motor_controller"),
    os.path.expanduser("~/EGB320_Examples/motor_controller"),
    "/home/egb320/EGB320_Examples/motor_controller",
]


def load_controller_class():
    """Import Controller from the unit's motor_controller package."""
    env = os.environ.get("EGB320_EXAMPLES")
    dirs = ([os.path.join(env, "motor_controller")] if env else []) + _SEARCH_DIRS

    for d in dirs:
        candidate = os.path.join(d, "controller.py")
        if os.path.exists(candidate):
            if d not in sys.path:
                sys.path.insert(0, d)
            from controller import Controller
            return Controller

    raise ImportError(
        "controller.py not found. It comes with EGB320_Examples and isn't in this\n"
        "repo. Either copy the folder in:\n"
        "    cp -r ~/EGB320_Examples/motor_controller Robot_Code/\n"
        "or point at it:\n"
        "    export EGB320_EXAMPLES=~/EGB320_Examples\n"
        f"Looked in: {', '.join(dirs)}"
    )


class MotorDriver:
    """Differential drive on top of the unit's Controller.

    set_velocity(v, w) is all nav calls: v forward m/s, w yaw rad/s, POSITIVE = LEFT.

    Watchdog: if set_velocity() isn't called for WATCHDOG_TIMEOUT_S the motors stop
    themselves. A tracked robot whose control loop has died should not keep driving.
    """

    def __init__(self, bus=I2C_BUS, addr=I2C_ADDR, max_speed=MAX_SPEED_RAW):
        Controller = load_controller_class()
        # Their Controller takes the bus and reads I2C_ADDR from its own class
        # attribute, so the address is only overridable by setting it on the class.
        if addr != getattr(Controller, "I2C_ADDR", addr):
            Controller.I2C_ADDR = addr
        self.board = Controller(i2c_bus=bus)      # raises if WHO_AM_I disagrees
        self.max_speed = max_speed
        self.last = (0.0, 0.0)
        self._last_cmd_t = time.monotonic()
        self._moving = False
        self._closed = False
        print(f"[drive] unit controller at 0x{addr:02x} on i2c-{bus}, "
              f"max raw speed {max_speed}")

        self._watchdog = None
        if WATCHDOG_TIMEOUT_S > 0:
            self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog.start()

    # -- commands ------------------------------------------------------------
    def set_velocity(self, v_mps, w_rps):
        left, right = wheel_speeds(v_mps, w_rps)
        self.set_raw(to_raw(left, self.max_speed), to_raw(right, self.max_speed))

    def set_raw(self, left, right):
        self.last = (left, right)
        self._last_cmd_t = time.monotonic()
        self._moving = bool(left or right)
        self.board.set_raw_motor_speed(int(left), int(right))

    def stop(self):
        """Never raises -- called on every exit path."""
        self.last = (0.0, 0.0)
        self._moving = False
        self._last_cmd_t = time.monotonic()
        try:
            self.board.set_raw_motor_speed(0, 0)
        except Exception as exc:                  # noqa: BLE001
            print(f"[drive] stop failed: {exc}")

    def close(self):
        self._closed = True
        self.stop()

    def _watchdog_loop(self):
        tripped = False
        while not self._closed:
            time.sleep(WATCHDOG_TIMEOUT_S / 4.0)
            idle = time.monotonic() - self._last_cmd_t
            if idle > WATCHDOG_TIMEOUT_S and self._moving and not tripped:
                print(f"[drive] WATCHDOG: no command for {idle:.2f}s -- stopping")
                self.stop()
                tripped = True
            elif idle <= WATCHDOG_TIMEOUT_S:
                tripped = False

    # -- feedback ------------------------------------------------------------
    def read_encoders(self):
        """(left, right) cumulative ticks, or None if the board won't answer.

        Real counts, not integrated from RPM -- this board has a tick register, which
        the DFRobot one did not.
        """
        try:
            return self.board.get_encoder_ticks()
        except Exception as exc:                  # noqa: BLE001
            print(f"[drive] encoder read failed: {exc}")
            return None


# --- pure maths, testable without hardware ----------------------------------
def wheel_speeds(v_mps, w_rps, track_m=TRACK_M):
    """(v, w) -> (left, right) wheel speeds in m/s. +w = turn LEFT."""
    half = track_m / 2.0
    return v_mps - w_rps * half, v_mps + w_rps * half


def to_raw(wheel_mps, max_speed=MAX_SPEED_RAW, max_wheel_rpm=MAX_WHEEL_RPM,
           min_speed=MIN_SPEED_RAW):
    """Wheel speed in m/s -> the signed number set_raw_motor_speed() wants.

    Deadband-compensated: anything under ~1% of full scale is a stop, everything else
    lands in [min_speed, max_speed], because a value below min_speed just stalls the
    geartrain and heats the motor instead of turning it.
    """
    max_mps = max_wheel_rpm / 60.0 * SPROCKET_CIRCUM_M
    if max_mps <= 0:
        return 0
    frac = abs(wheel_mps) / max_mps
    if frac < 0.01:
        return 0
    frac = min(frac, 1.0)
    return int(math.copysign(min_speed + frac * (max_speed - min_speed), wheel_mps))
