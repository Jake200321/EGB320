#!/usr/bin/env python3
"""STEP ONE: make the motors turn, and find out which way is forward.

Drives the unit's controller board (0x57 on i2c-8) through its own Controller class
from EGB320_Examples -- the same driver their motor_control_test.py uses, so if that
works, this works.

SETUP: controller.py isn't in this repo. Either copy it in --
    cp -r ~/EGB320_Examples/motor_controller Robot_Code/
or point at it:
    export EGB320_EXAMPLES=~/EGB320_Examples

RUN IT -- ROBOT ON A BLOCK, TRACKS OFF THE GROUND:
    python3 motor_spin_test.py                # both motors, forward then reverse
    python3 motor_spin_test.py --speed 60     # harder, if it doesn't break stiction
    python3 motor_spin_test.py --motor left   # one side only
    python3 motor_spin_test.py --whoami       # just check the board answers, then exit

WHAT YOU'RE CHECKING:
    1. board answers WHO_AM_I           -> if not: i2cdetect -y 8, expect 57
    2. each motor turns on its own      -> if not: raise --speed, check leads/battery
    3. which direction a POSITIVE speed drives each track  -> write it down, then set
       LEFT_SIGN / RIGHT_SIGN below so positive means forward on both
    4. encoder ticks move, and which way they count

set_raw_motor_speed()'s units aren't documented anywhere we have, so --speed is in
whatever the board wants. Start low: these are 6 V motors.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motors import I2C_ADDR, I2C_BUS, load_controller_class   # noqa: E402

# Flip these once you know which way each track actually drives. The motors face
# opposite ways on the chassis, so one side almost certainly needs -1.
LEFT_SIGN = 1
RIGHT_SIGN = 1


def connect(bus=I2C_BUS, addr=I2C_ADDR):
    Controller = load_controller_class()
    if addr != getattr(Controller, "I2C_ADDR", addr):
        Controller.I2C_ADDR = addr
    try:
        board = Controller(i2c_bus=bus)
    except Exception as exc:                       # noqa: BLE001
        raise SystemExit(
            f"controller did not come up on i2c-{bus} at 0x{addr:02x} -- "
            f"{type(exc).__name__}: {exc}\n"
            f"  sudo i2cdetect -y {bus}      (expect {addr:02x} in the grid)\n"
            "  check the board is powered and seated"
        )
    print(f"board OK at 0x{addr:02x} on i2c-{bus}")
    return board


def ticks(board):
    try:
        return board.get_encoder_ticks()
    except Exception:                              # noqa: BLE001
        return None


def run(board, left, right, seconds, label):
    print(f"\n>>> {label}: left={left:+d} right={right:+d} for {seconds:.1f}s")
    before = ticks(board)
    board.set_raw_motor_speed(int(left), int(right))
    deadline = time.time() + seconds
    while time.time() < deadline:
        time.sleep(0.25)
        t = ticks(board)
        if t is not None:
            print(f"      encoder ticks: {t}")
    board.set_raw_motor_speed(0, 0)
    time.sleep(0.4)

    after = ticks(board)
    if before is not None and after is not None:
        moved = tuple(a - b for a, b in zip(after, before))
        print(f"      moved {moved} ticks")
        if moved == (0, 0):
            print("      !! nothing moved -- raise --speed, or check leads and battery")
    return after


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--speed", type=int, default=40, help="raw speed magnitude")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--motor", choices=["left", "right"], default=None)
    ap.add_argument("--bus", type=int, default=I2C_BUS)
    ap.add_argument("--addr", type=lambda v: int(v, 0), default=I2C_ADDR)
    ap.add_argument("--whoami", action="store_true", help="check the board, then exit")
    args = ap.parse_args()

    board = connect(args.bus, args.addr)
    if args.whoami:
        print("ticks:", ticks(board))
        return

    s = args.speed
    try:
        if args.motor in (None, "left"):
            run(board, LEFT_SIGN * s, 0, args.seconds, "LEFT track, positive")
            run(board, -LEFT_SIGN * s, 0, args.seconds, "LEFT track, negative")
            print("    -> which drove the LEFT track FORWARD? set LEFT_SIGN accordingly.")
        if args.motor in (None, "right"):
            run(board, 0, RIGHT_SIGN * s, args.seconds, "RIGHT track, positive")
            run(board, 0, -RIGHT_SIGN * s, args.seconds, "RIGHT track, negative")
            print("    -> same for RIGHT_SIGN.")
        if args.motor is None:
            run(board, LEFT_SIGN * s, RIGHT_SIGN * s, args.seconds,
                "BOTH forward (should drive straight, not spin)")
        print("\nDone.")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            board.set_raw_motor_speed(0, 0)
        except Exception:                          # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
