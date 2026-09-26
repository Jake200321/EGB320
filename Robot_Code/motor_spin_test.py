#!/usr/bin/env python3
"""STEP ONE: make the motors turn, and check which way is forward.

Talks to the controller board (0x57 on i2c-8) through motors.py, which speaks the
board's I2C protocol directly -- no dependency on the EGB320_Examples files.

    ROBOT ON A BLOCK. Tracks off the ground.

    python3 motor_spin_test.py               # each track forward, then reverse
    python3 motor_spin_test.py --speed 60    # gentler
    python3 motor_spin_test.py --motor left  # one side only
    python3 motor_spin_test.py --spin        # spin left then right, via set_velocity
    python3 motor_spin_test.py --raw         # bypass motors.py's signs/swap/trim
    python3 motor_spin_test.py --whoami      # identify the board, then exit

By default every command goes through MotorDriver, so LEFT_SIGN, RIGHT_SIGN,
SWAP_MOTORS and the trims in motors.py are all applied -- exactly what the robot
will do. "LEFT, forward" should move the physical LEFT track forward.

WHAT YOU'RE CHECKING:
    1. the board answers WHO_AM_I     -> if not: sudo i2cdetect -y 8, expect 57
    2. each track turns on its own    -> if not: raise --speed, check leads/battery
    3. the track named is the one that moves    -> if not: toggle SWAP_MOTORS
    4. "forward" drives that track forward      -> if not: flip its LEFT_/RIGHT_SIGN
    5. encoder ticks count up going forward, down going back

--raw sends the board's channels unflipped, for working out the settings from
scratch. --speed is the board's raw motor speed, 1-127.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import motors  # noqa: E402
from motors import I2C_ADDR, I2C_BUS, MotorController, MotorDriver, _to_i16  # noqa: E402


def connect(bus, addr):
    try:
        board = MotorController(bus, addr)
    except Exception as exc:                       # noqa: BLE001
        raise SystemExit(
            f"board did not come up on i2c-{bus} at {addr:#04x} -- "
            f"{type(exc).__name__}: {exc}\n"
            f"  sudo i2cdetect -y {bus}     (expect {addr:02x} in the grid)\n"
            "  check the board is powered and seated")
    fw = ".".join(map(str, board.get_firmware_version()))
    print(f"board OK at {addr:#04x} on i2c-{bus}, firmware {fw}")
    return board


class RawOutput:
    """Board channels as-is: no signs, no swap, no trim."""

    def __init__(self, board):
        self.board = board

    def send(self, left, right):
        self.board.set_raw_motor_speed(left, right)

    def ticks(self):
        return self.board.get_raw_encoder_ticks()

    def moved(self, before, after):
        return tuple(_to_i16(a - b) for a, b in zip(after, before))


class DriverOutput:
    """Through MotorDriver, so motors.py's signs, swap and trims all apply."""

    def __init__(self, board):
        self.driver = MotorDriver(controller=board)

    def send(self, left, right):
        self.driver.set_raw(left, right)

    def ticks(self):
        return self.driver.read_encoders()

    def moved(self, before, after):
        return tuple(a - b for a, b in zip(after, before))


def run(out, left, right, seconds, label):
    print(f"\n>>> {label}: left={left:+d} right={right:+d} for {seconds:.1f}s")
    before = out.ticks()
    deadline = time.time() + seconds
    while time.time() < deadline:
        # Re-sent every loop: MotorDriver arms the board's watchdog, which cuts the
        # motors if it goes BOARD_WATCHDOG_S without a command.
        out.send(left, right)
        time.sleep(0.25)
        print(f"      encoder counters: {out.ticks()}")
    out.send(0, 0)
    time.sleep(0.4)

    moved = out.moved(before, out.ticks())
    print(f"      moved {moved} ticks  (left, right)")
    if moved == (0, 0):
        print("      !! nothing moved -- raise --speed, or check leads and battery")
    return moved


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--speed", type=int, default=100, help="raw speed, 1-127")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--motor", choices=["left", "right"], default=None)
    ap.add_argument("--bus", type=int, default=I2C_BUS)
    ap.add_argument("--addr", type=lambda v: int(v, 0), default=I2C_ADDR)
    ap.add_argument("--whoami", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="bypass motors.py's signs, swap and trims")
    ap.add_argument("--spin", action="store_true",
                    help="spin left then right, to check turn direction")
    args = ap.parse_args()

    if not 1 <= args.speed <= 127:
        raise SystemExit("--speed must be 1..127")

    board = connect(args.bus, args.addr)
    print(f"motors.py: LEFT_SIGN={motors.LEFT_SIGN}, RIGHT_SIGN={motors.RIGHT_SIGN}, "
          f"SWAP_MOTORS={motors.SWAP_MOTORS}, "
          f"LEFT_TRIM={motors.LEFT_TRIM}, RIGHT_TRIM={motors.RIGHT_TRIM}")
    if args.whoami:
        print(f"who_am_i: {board.who_am_i():#04x}")
        print(f"status:   {board.get_status()}")
        print(f"encoders: {board.get_raw_encoder_ticks()}")
        return

    if args.raw:
        print("--raw: sending board channels unflipped -- the settings above are IGNORED")
        out = RawOutput(board)
    else:
        print("sending through MotorDriver -- the settings above are APPLIED")
        out = DriverOutput(board)

    s = args.speed
    try:
        if args.spin:
            d = out.driver if isinstance(out, DriverOutput) else MotorDriver(controller=board)
            for label, w in (("LEFT (counter-clockwise)", 2.0),
                             ("RIGHT (clockwise)", -2.0)):
                print(f"\n>>> should spin {label}")
                deadline = time.time() + args.seconds
                while time.time() < deadline:
                    d.set_velocity(0.0, w)
                    time.sleep(0.1)
                d.stop()
                time.sleep(0.5)
            print("\n    -> did it spin the way each line said?")
            print("       opposite both times -> toggle SWAP_MOTORS in motors.py")
            print("       didn't spin at all  -> one sign is wrong; flip LEFT_SIGN")
            return

        if args.raw:
            fwd, back = "positive speed", "negative speed"
        else:
            fwd, back = "FORWARD", "BACKWARD"
        if args.motor in (None, "left"):
            run(out, s, 0, args.seconds, f"LEFT track, {fwd}")
            run(out, -s, 0, args.seconds, f"LEFT track, {back}")
        if args.motor in (None, "right"):
            run(out, 0, s, args.seconds, f"RIGHT track, {fwd}")
            run(out, 0, -s, args.seconds, f"RIGHT track, {back}")
        if args.motor is None:
            run(out, s, s, args.seconds, f"BOTH, {fwd} (straight, not a spin?)")

        if args.raw:
            print("\n    -> per physical track: if NEGATIVE speed drove it forward, "
                  "its SIGN is -1, else +1")
            print("    -> if the 'LEFT' lines moved the physical RIGHT track, "
                  "SWAP_MOTORS = True")
        else:
            print("\n    -> wrong track moved for its line  -> toggle SWAP_MOTORS")
            print("    -> right track, wrong direction    -> flip that track's SIGN")
        print("\nDone.")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            board.set_raw_motor_speed(0, 0)
            board.standby()
        except Exception:                          # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
