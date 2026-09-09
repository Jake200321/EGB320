#!/usr/bin/env python3
"""STEP ONE: make the motors turn, and find out which way is forward.

Self-contained -- no other files needed. Talks to the DFRobot DFR0592 HAT directly
over I2C, so if this fails the problem is wiring, power or I2C, not software.

SETUP (on the Pi, once):
    sudo raspi-config          # Interface Options -> I2C -> enable
    sudo i2cdetect -y 1        # expect the HAT to show up at 0x10
    git clone https://github.com/DFRobot/DFRobot_RaspberryPi_Motor vendor
      ^ run that from inside this folder. The space before "vendor" is meant to be
        there: git clone takes <url> then <where-to-put-it>. The library is not on
        PyPI, so pip can't fetch it.

RUN IT -- ROBOT ON A BLOCK, TRACKS OFF THE GROUND:
    python3 motor_spin_test.py               # full sequence, 30 % duty
    python3 motor_spin_test.py --duty 50     # harder, if it doesn't break stiction
    python3 motor_spin_test.py --motor 1     # one channel only
    python3 motor_spin_test.py --detect      # just scan the I2C bus and exit

WHAT YOU'RE CHECKING, IN ORDER:
    1. board answers on I2C       -> if not: i2cdetect, HAT seating, 7-12 V rail
    2. each motor turns alone     -> if not: motor leads in M1+/M1-, duty too low,
                                     or a flat battery
    3. which way CW actually goes -> write it down; that's the sign convention for
                                     everything you build on top
    4. encoder RPM is non-zero    -> if 0 while the shaft spins, the encoder JST is
                                     wrong: P2=GND, P3=B, P4=A, P5=VCC into E1/E2
    5. does RPM go NEGATIVE in reverse, or stay positive? -> tells you whether the
                                     board reports signed speed or just magnitude

SAFETY: the N20s are 6 V motors and the HAT needs a 7-12 V rail, so full duty puts
the whole pack across a 6 V motor. Keep --duty modest here and the runs short.
"""

import argparse
import shlex
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- hardware
I2C_BUS = 1              # /dev/i2c-1 on every Pi, including the Pi 5
HAT_ADDRESS = 0x10       # DFR0592 default
LEFT_MOTOR = 1           # M1 terminal
RIGHT_MOTOR = 2          # M2 terminal
GEAR_RATIO = 50          # N20 1:50 -- tells the HAT to report output-shaft RPM
PWM_FREQUENCY_HZ = 1000  # board accepts 100-12750

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def load_board_class():
    """Import DFRobot_DC_Motor_IIC from the vendored clone, or anywhere on sys.path."""
    def _try():
        from DFRobot_RaspberryPi_DC_Motor import DFRobot_DC_Motor_IIC
        return DFRobot_DC_Motor_IIC

    try:
        return _try()
    except ImportError:
        pass

    # The clone puts the .py at its root, but glob anyway so a nested checkout works.
    for found in VENDOR_DIR.rglob("DFRobot_RaspberryPi_DC_Motor.py"):
        if str(found.parent) not in sys.path:
            sys.path.insert(0, str(found.parent))
        break

    try:
        return _try()
    except ImportError:
        raise SystemExit(
            "DFRobot motor HAT library not found. It is not on PyPI -- clone it:\n"
            "    git clone https://github.com/DFRobot/DFRobot_RaspberryPi_Motor "
            f"{shlex.quote(str(VENDOR_DIR))}\n"
            "(the space before the destination is deliberate: git clone takes "
            "<url> then <destination>)"
        )


def connect(verbose=True):
    board = load_board_class()(I2C_BUS, HAT_ADDRESS)
    for attempt in range(1, 6):
        status = board.begin()
        if status == board.STA_OK:
            if verbose:
                print(f"board OK at 0x{HAT_ADDRESS:02x} on i2c-{I2C_BUS}")
            return board
        print(f"  begin() attempt {attempt}: status {status}")
        time.sleep(0.5)
    raise SystemExit(
        "Board did not come up. Check, in this order:\n"
        "  sudo i2cdetect -y 1        (expect 10 in the grid)\n"
        "  HAT fully seated on the 40-pin header\n"
        "  7-12 V on VIN/GND -- the board needs the motor rail, not just Pi 5 V\n"
        "  I2C enabled: sudo raspi-config -> Interface Options -> I2C"
    )


def spin(board, motor_id, orientation, duty, seconds, label):
    name = "CW " if orientation == board.CW else "CCW"
    print(f"\n>>> {label}: M{motor_id} {name} @ {duty:.0f}% for {seconds:.1f}s")
    board.motor_movement(motor_id, orientation, duty)
    deadline = time.time() + seconds
    peak = 0.0
    while time.time() < deadline:
        time.sleep(0.25)
        rpm = float(board.get_encoder_speed(board.ALL)[motor_id - 1])
        if abs(rpm) > abs(peak):
            peak = rpm
        print(f"      encoder M{motor_id}: {rpm:>7.1f} rpm")
    board.motor_stop(motor_id)
    time.sleep(0.4)

    if peak == 0.0:
        print("      !! encoder read 0 the whole time -- did the shaft actually turn?")
        print("         shaft turned + 0 rpm  => encoder wiring (E1A/E1B, encoder VCC/GND)")
        print("         shaft didn't turn     => raise --duty, check motor leads / battery")
    else:
        print(f"      peak {peak:.1f} rpm")
    return peak


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--duty", type=float, default=30.0, help="PWM duty %% (default 30)")
    ap.add_argument("--seconds", type=float, default=2.0, help="run time per step")
    ap.add_argument("--motor", type=int, choices=[1, 2], default=None,
                    help="test one channel only (default: both)")
    ap.add_argument("--detect", action="store_true", help="scan the I2C bus and exit")
    args = ap.parse_args()

    if args.detect:
        print("addresses found:", load_board_class()(I2C_BUS, HAT_ADDRESS).detecte())
        return

    if args.duty > 60:
        print(f"WARNING: --duty {args.duty:.0f} on a 6 V motor from a 7-12 V rail. "
              "Keep the run short.\n")

    board = connect()
    board.set_encoder_enable(board.ALL)
    board.set_encoder_reduction_ratio(board.ALL, GEAR_RATIO)
    board.set_moter_pwm_frequency(PWM_FREQUENCY_HZ)
    board.motor_stop(board.ALL)

    motors = [args.motor] if args.motor else [LEFT_MOTOR, RIGHT_MOTOR]
    side = {LEFT_MOTOR: "LEFT", RIGHT_MOTOR: "RIGHT"}

    try:
        for motor_id in motors:
            label = side.get(motor_id, f"M{motor_id}")
            fwd = spin(board, motor_id, board.CW, args.duty, args.seconds, f"{label} track")
            rev = spin(board, motor_id, board.CCW, args.duty, args.seconds, f"{label} track")

            print(f"\n    -> which of those drove the {label} track FORWARD? Write it down.")
            if fwd and rev:
                if (fwd > 0) == (rev > 0):
                    print("    -> encoder RPM stayed positive both ways: the board reports "
                          "MAGNITUDE ONLY, so direction has to come from what you commanded.")
                else:
                    print("    -> encoder RPM flipped sign with direction: the board reports "
                          "SIGNED speed, so you can read direction straight off it.")

        if not args.motor:
            print(f"\n>>> BOTH motors CW together for {args.seconds:.1f}s")
            print("    (if the tracks are mounted mirrored, this spins rather than "
                  "driving straight -- that's expected, and it's why one side gets "
                  "inverted in software later)")
            board.motor_movement(board.ALL, board.CW, args.duty)
            time.sleep(args.seconds)
            board.motor_stop(board.ALL)

        print("\nDone.")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            board.motor_stop(board.ALL)
        except Exception:
            pass


if __name__ == "__main__":
    main()
