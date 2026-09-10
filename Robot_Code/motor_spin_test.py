#!/usr/bin/env python3
"""STEP ONE: make the motors turn, and find out which way is forward.

Talks to the controller board (0x57 on i2c-8) through motors.py, which speaks the
board's I2C protocol directly -- no dependency on the EGB320_Examples files.

    ROBOT ON A BLOCK. Tracks off the ground.

    python3 motor_spin_test.py               # each track forward, then reverse
    python3 motor_spin_test.py --speed 60    # gentler
    python3 motor_spin_test.py --motor left  # one side only
    python3 motor_spin_test.py --whoami      # identify the board, then exit

WHAT YOU'RE CHECKING:
    1. the board answers WHO_AM_I     -> if not: sudo i2cdetect -y 8, expect 57
    2. each track turns on its own    -> if not: raise --speed, check leads/battery
    3. which way a POSITIVE speed drives each track -> write it down, then set
       LEFT_SIGN / RIGHT_SIGN in motors.py so positive means forward on both
    4. encoder ticks move, and count up when the track drives forward

--speed is the board's raw motor speed, 1-127.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motors import I2C_ADDR, I2C_BUS, LEFT_SIGN, RIGHT_SIGN, MotorController  # noqa: E402


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


def run(board, left, right, seconds, label):
    print(f"\n>>> {label}: left={left:+d} right={right:+d} for {seconds:.1f}s")
    before = board.get_raw_encoder_ticks()
    board.set_raw_motor_speed(left, right)
    deadline = time.time() + seconds
    while time.time() < deadline:
        time.sleep(0.25)
        print(f"      raw encoder counters: {board.get_raw_encoder_ticks()}")
    board.set_raw_motor_speed(0, 0)
    time.sleep(0.4)

    from motors import _to_i16
    after = board.get_raw_encoder_ticks()
    moved = tuple(_to_i16(a - b) for a, b in zip(after, before))
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
    ap.add_argument("--spin", action="store_true",
                    help="spin left then right, to check turn direction")
    args = ap.parse_args()

    if not 1 <= args.speed <= 127:
        raise SystemExit("--speed must be 1..127")

    board = connect(args.bus, args.addr)
    print(f"motors.py currently has LEFT_SIGN={LEFT_SIGN}, RIGHT_SIGN={RIGHT_SIGN} "
          "(this test sends raw, unflipped speeds so you can see the truth)")
    if args.whoami:
        print(f"who_am_i: {board.who_am_i():#04x}")
        print(f"status:   {board.get_status()}")
        print(f"encoders: {board.get_raw_encoder_ticks()}")
        return

    s = args.speed
    try:
        if args.spin:
            # Sent through MotorDriver, so this exercises the real signs and swap --
            # unlike the per-motor tests below, which are deliberately raw.
            from motors import MotorDriver
            d = MotorDriver(controller=board)
            for label, w in (("LEFT (counter-clockwise)", 2.0),
                             ("RIGHT (clockwise)", -2.0)):
                print(f"\n>>> should spin {label}")
                d.set_velocity(0.0, w)
                time.sleep(args.seconds)
                d.stop()
                time.sleep(0.5)
            print("\n    -> did it spin the way each line said?")
            print("       no  -> set SWAP_MOTORS = True in motors.py")
            return
        if args.motor in (None, "left"):
            run(board, s, 0, args.seconds, "LEFT track, positive speed")
            run(board, -s, 0, args.seconds, "LEFT track, negative speed")
            print("    -> if NEGATIVE speed drove the LEFT track forward, set "
                  "LEFT_SIGN = -1 in motors.py")
        if args.motor in (None, "right"):
            run(board, 0, s, args.seconds, "RIGHT track, positive speed")
            run(board, 0, -s, args.seconds, "RIGHT track, negative speed")
            print("    -> if NEGATIVE speed drove the RIGHT track forward, set "
                  "RIGHT_SIGN = -1 in motors.py")
        if args.motor is None:
            run(board, s, s, args.seconds, "BOTH positive (straight, not a spin?)")
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
