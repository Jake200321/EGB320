#!/usr/bin/env python3
"""Motor control for the EGB320 controller board (0x57 on i2c-8). Self-contained.

Speaks the board's I2C protocol directly -- nothing here imports from
EGB320_Examples, so the nav system has no dependency on files that ship separately
and could move or change underneath it. The register map and packing below were
taken from the unit's controller.py, which is the authority on this hardware; only
the commands nav actually needs are implemented.

    MotorController   the wire protocol: WHO_AM_I, firmware, raw speed, encoders,
                      the board's own shutdown watchdog
    MotorDriver       what nav uses: set_velocity(v, w) in m/s and rad/s, plus
                      accumulated odometry

Two things about this board worth knowing:

  * Raw motor speed is -127..127. 127 and -128 are reserved: -128 written to both
    motors means standby. Values are clamped here rather than sent blind.
  * The board has its own motor shutdown timeout. That's a better watchdog than
    anything in Python, because it still fires when this process dies -- which is
    exactly when a robot driving away matters.
"""

import math
import struct
import time


# --- board ------------------------------------------------------------------
I2C_ADDR = 0x57
I2C_BUS = 8
REQUIRED_FIRMWARE = (1, 3)          # major must match, minor must be >= this

# --- geometry / tuning ------------------------------------------------------
TRACK_M = 0.123                     # Pololu 30T sprocket centre-to-centre
SPROCKET_CIRCUM_M = math.pi * 0.024
# Output-shaft RPM at MAX_SPEED_RAW. This is the scale every m/s command is measured
# against, so setting it too HIGH makes the robot crawl: it thinks 0.10 m/s is a small
# fraction of what it can do and sends a correspondingly small raw speed. 200 was the
# 6 V no-load figure; under load on tracks it's nowhere near that. MEASURE.
MAX_WHEEL_RPM = 130.0

# ---- MOTOR DIRECTION -- change these if a track drives the wrong way ----------
# Which way a POSITIVE raw speed turns each track. The motors sit facing opposite
# directions on the chassis, and can be mounted backwards, so these fix it in
# software instead of rewiring.
#
#   both tracks drive backwards  -> set BOTH to -1
#   one track backwards          -> set just that one to -1
#   robot spins instead of driving straight -> the two disagree; flip one
#
# Find out with:  python3 motor_spin_test.py
# Encoder counts are flipped by the same sign, so forward always counts up.
LEFT_SIGN = 1
RIGHT_SIGN = -1

# Set True if the board's two channels are wired to the opposite tracks. Tell it
# apart from a sign problem by what the robot gets WRONG:
#
#   drives backwards, turns correctly     -> both signs are wrong
#   spins instead of driving straight     -> one sign is wrong
#   drives straight fine, turns MIRRORED  -> SWAP_MOTORS  (signs are fine)
#
# A swap leaves straight driving looking perfect, because both channels get the same
# command -- it only shows up the moment the robot tries to turn. It also inverts the
# encoder straight-line correction, which then steers further off instead of back.
SWAP_MOTORS = False

# ---- TRACK TRIM -- fix a robot that pulls to one side -------------------------
# Per-track multipliers on every command. Raise one, or lower the other, until it
# runs straight. 1.0 is untrimmed.
#
#   pulls RIGHT (left track weaker) -> raise LEFT_TRIM,  e.g. 1.08
#   pulls LEFT  (right track weaker) -> raise RIGHT_TRIM
#
# Prefer LOWERING the strong side over raising the weak one. At full speed the
# strong track is already at 127 and there is no headroom to add -- raising the
# weak side then does nothing at all, and only the trim on the fast track has any
# effect. Trimming down always works.
#
# This is for the open-loop stretches -- the blind run-in, and the approach once
# the heading is locked -- where nothing is correcting. While exploring, the
# encoder straight-line correction already compensates for a track imbalance, so
# a trim there just reduces how hard it has to work.
LEFT_TRIM = 1.3
RIGHT_TRIM = 1.0
# -------------------------------------------------------------------------------

SPEED_LIMIT = 127                   # the board's hard limit
MAX_SPEED_RAW = 127                 # full board range -- this is the aggression knob
MIN_SPEED_RAW = 45                  # below this the geartrain stalls. MEASURE

BOARD_WATCHDOG_S = 0.5              # board cuts the motors after this with no command


class WhoAmIMismatch(Exception):
    """Something answered at 0x57, but it isn't this board."""


class FirmwareVersionMismatch(Exception):
    """The board is running firmware this protocol doesn't match."""


def _to_i16(v):
    """Reinterpret a uint16 as int16 -- how encoder wraparound is made sane."""
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


class MotorController:
    """The board's I2C protocol. Only what nav needs."""

    CMD_FIRMWARE_VERSION = 0x08
    CMD_WHO_AM_I = 0x0F
    CMD_MOTOR_SHUTDOWN_TIMEOUT = 0x28
    CMD_RAW_MOTOR_SPEED = 0x30
    CMD_ENCODER_TICKS = 0x32
    CMD_RAW_ENCODER_TICKS = 0x33
    CMD_STATUS = 0x36

    STANDBY = -128                  # written to both motors = standby

    def __init__(self, bus=I2C_BUS, addr=I2C_ADDR, check=True):
        self.addr = addr
        self.bus_num = bus
        # python3-smbus on Pi OS; smbus2 is a drop-in and is what pip installs.
        try:
            import smbus
        except ImportError:
            try:
                import smbus2 as smbus
            except ImportError as exc:
                raise ImportError(
                    "no smbus module -- install one of:\n"
                    "    sudo apt install python3-smbus\n"
                    "    pip install smbus2"
                ) from exc
        self.i2c = smbus.SMBus(bus)
        if check:
            self.check_who_am_i()
            self.check_firmware_version()

    # -- wire ----------------------------------------------------------------
    def _read(self, command, n, spec):
        return struct.unpack(spec, bytes(self.i2c.read_i2c_block_data(self.addr, command, n)))

    def _write(self, command, data):
        self.i2c.write_i2c_block_data(self.addr, command, list(data))

    # -- identity ------------------------------------------------------------
    def who_am_i(self):
        return self._read(self.CMD_WHO_AM_I, 1, "B")[0]

    def check_who_am_i(self):
        w = self.who_am_i()
        if w != self.addr:
            raise WhoAmIMismatch(
                f"WHO_AM_I returned {w:#04x}, expected {self.addr:#04x} -- "
                f"something else is on i2c-{self.bus_num} at that address")

    def get_firmware_version(self):
        return self._read(self.CMD_FIRMWARE_VERSION, 3, "BBB")

    def check_firmware_version(self):
        major, minor, patch = self.get_firmware_version()
        need_major, need_minor = REQUIRED_FIRMWARE
        if major != need_major or minor < need_minor:
            raise FirmwareVersionMismatch(
                f"board runs firmware {major}.{minor}.{patch}, this expects "
                f"{need_major}.{need_minor}.*")

    # -- motors --------------------------------------------------------------
    def set_raw_motor_speed(self, left, right):
        """Open loop, -127..127 each. Clamped -- the board rejects out-of-range."""
        left = max(-SPEED_LIMIT, min(SPEED_LIMIT, int(left)))
        right = max(-SPEED_LIMIT, min(SPEED_LIMIT, int(right)))
        self._write(self.CMD_RAW_MOTOR_SPEED, struct.pack("bb", left, right))

    def standby(self):
        """Motors off, board idle."""
        self._write(self.CMD_RAW_MOTOR_SPEED,
                    struct.pack("bb", self.STANDBY, self.STANDBY))

    def set_motor_shutdown_timeout(self, seconds):
        """Board stops the motors if it hears nothing for this long. 0.1-10 s."""
        if not 0.1 <= seconds <= 10.0:
            raise ValueError("shutdown timeout must be 0.1-10 s")
        self._write(self.CMD_MOTOR_SHUTDOWN_TIMEOUT, [round(seconds * 10)])

    # -- encoders ------------------------------------------------------------
    def get_raw_encoder_ticks(self):
        """Absolute uint16 per side, wrapping. Deltas are taken against these."""
        return self._read(self.CMD_RAW_ENCODER_TICKS, 4, "HH")

    def get_status(self):
        (s,) = self._read(self.CMD_STATUS, 1, "B")
        return {"moving": bool(s & 1), "controlled": bool(s & 2)}


class MotorDriver:
    """Differential drive. set_velocity(v, w): v forward m/s, w yaw rad/s, +w = LEFT."""

    def __init__(self, bus=I2C_BUS, addr=I2C_ADDR, max_speed=MAX_SPEED_RAW,
                 controller=None):
        self.board = controller if controller is not None else MotorController(bus, addr)
        self.max_speed = min(abs(max_speed), SPEED_LIMIT)
        self.last = (0, 0)

        try:
            self.board.set_motor_shutdown_timeout(BOARD_WATCHDOG_S)
        except Exception as exc:                     # noqa: BLE001
            print(f"[drive] could not set the board watchdog: {exc}")

        self._prev = self._read_raw()
        self.ticks = [0, 0]

        try:
            fw = ".".join(map(str, self.board.get_firmware_version()))
        except Exception:                            # noqa: BLE001
            fw = "?"
        print(f"[drive] controller 0x{addr:02x} on i2c-{bus}, firmware {fw}, "
              f"max raw speed {self.max_speed}")

    # -- commands ------------------------------------------------------------
    def set_velocity(self, v_mps, w_rps):
        left, right = wheel_speeds(v_mps, w_rps)
        self.set_raw(to_raw(left, self.max_speed), to_raw(right, self.max_speed))

    def set_raw(self, left, right):
        """Applies trim, then LEFT_SIGN / RIGHT_SIGN, then SWAP_MOTORS, so callers
        always mean 'positive = forward' on the track they named.

        Trim goes first, while left/right still refer to physical tracks -- after the
        swap they're board channels, and trimming there would land on the wrong one.
        """
        left = int(round(left * LEFT_TRIM))
        right = int(round(right * RIGHT_TRIM))
        left, right = left * LEFT_SIGN, right * RIGHT_SIGN
        if SWAP_MOTORS:
            left, right = right, left
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
    def _read_raw(self):
        try:
            return self.board.get_raw_encoder_ticks()
        except Exception:                            # noqa: BLE001
            return None

    def read_encoders(self):
        """(left, right) cumulative ticks since this driver started, or None.

        Accumulated from wrapping uint16 counters: each poll takes the difference
        against the previous reading and reinterprets it as int16, so it stays correct
        across the 65535 -> 0 rollover and counts down when reversing. Poll often
        enough that a wheel can't turn more than half a counter between reads.
        """
        now = self._read_raw()
        if now is None:
            return None
        if self._prev is None:
            self._prev = now
            return tuple(self.ticks)
        # Same corrections as the motors: a track driving forward counts up, and the
        # channels line up with the tracks they actually drive. Without the swap here
        # the straight-line correction would steer away from straight.
        dl = _to_i16(now[0] - self._prev[0])
        dr = _to_i16(now[1] - self._prev[1])
        if SWAP_MOTORS:
            dl, dr = dr, dl
        self.ticks[0] += dl * LEFT_SIGN
        self.ticks[1] += dr * RIGHT_SIGN
        self._prev = now
        return tuple(self.ticks)

    def reset_encoders(self):
        self.ticks = [0, 0]
        self._prev = self._read_raw()

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
    raw = min_speed + min(frac, 1.0) * (max_speed - min_speed)
    return int(math.copysign(min(raw, SPEED_LIMIT), wheel_mps))


# NOTE for later: the board also does closed-loop speed (CMD_CONTROLLED_MOTOR_SPEED,
# 0x31, "<hh" in ticks per 1/100 s) regulated by its own PID. That holds a straight
# line far better than open-loop PWM, which drifts whenever the two motors differ.
# Worth adding once MAX_WHEEL_RPM is measured and there's a ticks-per-metre figure.
