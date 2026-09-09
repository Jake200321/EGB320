#!/usr/bin/env python3
"""STEP ONE: make the motors turn, and find out which way is forward.

Run this on the Pi before anything else. It talks to the DFR0592 HAT DIRECTLY --
no nav stack, no mobility driver -- so if it fails, the problem is wiring, power or
I2C, not software.

    PUT THE ROBOT ON A BLOCK. The tracks must be off the ground.

    python3 tools/motor_spin_test.py                 # full sequence, 30 % duty
    python3 tools/motor_spin_test.py --duty 50       # harder, if it doesn't break stiction
    python3 tools/motor_spin_test.py --motor 1       # one channel only
    python3 tools/motor_spin_test.py --detect        # just scan the I2C bus and exit

What you are checking, in order:
  1. board found on I2C            -> if not: `sudo i2cdetect -y 1`, check HAT seating + 7-12 V rail
  2. each motor turns on its own   -> if not: motor leads in M1+/M1-, duty too low, or flat battery
  3. which physical direction CW is -> write it down, then set motor_left_invert /
                                       motor_right_invert in egb320/config.py
  4. encoder RPM is non-zero       -> if 0 while the shaft spins: encoder JST wiring
                                       (P2=GND, P3=B, P4=A, P5=VCC) into E1/E2
  5. does RPM read NEGATIVE in reverse, or just positive magnitude?
     -> sets drive.encoder_reports_signed in config.py

SAFETY: the N20s are 6 V motors on a 7-12 V rail. Keep --duty modest (<= 60) and the
runs short here; set drive.max_duty_percent from your actual pack voltage before any
sustained driving.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egb320.config import Config
from egb320.mobility.dfr0592 import load_board_class


def connect(cfg, verbose=True):
    board = load_board_class()(cfg.hardware.i2c_bus, cfg.hardware.motor_hat_i2c_addr)
    for attempt in range(1, 6):
        status = board.begin()
        if status == board.STA_OK:
            if verbose:
                print(f"board OK at 0x{cfg.hardware.motor_hat_i2c_addr:02x} "
                      f"on i2c-{cfg.hardware.i2c_bus}")
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
    print(f"\n>>> {label}: motor M{motor_id} {name} @ {duty:.0f}% for {seconds:.1f}s")
    board.motor_movement(motor_id, orientation, duty)
    deadline = time.time() + seconds
    peak = 0.0
    while time.time() < deadline:
        time.sleep(0.25)
        speeds = board.get_encoder_speed(board.ALL)
        rpm = float(speeds[motor_id - 1])
        peak = rpm if abs(rpm) > abs(peak) else peak
        print(f"      encoder M{motor_id}: {rpm:>7.1f} rpm")
    board.motor_stop(motor_id)
    time.sleep(0.4)
    if peak == 0.0:
        print("      !! encoder read 0 the whole time -- did the shaft actually turn?")
        print("         shaft turning + 0 rpm  => encoder wiring (E1A/E1B or encoder VCC/GND)")
        print("         shaft not turning      => raise --duty, or check motor leads / battery")
    else:
        print(f"      peak {peak:.1f} rpm  ({'signed' if peak < 0 else 'positive'})")
    return peak


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--duty", type=float, default=30.0, help="PWM duty %% (default 30)")
    ap.add_argument("--seconds", type=float, default=2.0, help="run time per step")
    ap.add_argument("--motor", type=int, choices=[1, 2], default=None,
                    help="test one channel only (default: both)")
    ap.add_argument("--detect", action="store_true", help="scan the I2C bus and exit")
    args = ap.parse_args()

    if args.duty > 60:
        print(f"WARNING: --duty {args.duty:.0f} on a 6 V motor from a 7-12 V rail. "
              "Keep the run short.\n")

    cfg = Config()

    if args.detect:
        board = load_board_class()(cfg.hardware.i2c_bus, cfg.hardware.motor_hat_i2c_addr)
        print("addresses found:", board.detecte())
        return

    board = connect(cfg)
    board.set_encoder_enable(board.ALL)
    board.set_encoder_reduction_ratio(board.ALL, int(cfg.geometry.gear_ratio))
    board.set_moter_pwm_frequency(cfg.drive.pwm_frequency_hz)
    board.motor_stop(board.ALL)

    motors = [args.motor] if args.motor else [cfg.hardware.motor_left_id,
                                              cfg.hardware.motor_right_id]
    side = {cfg.hardware.motor_left_id: "LEFT", cfg.hardware.motor_right_id: "RIGHT"}

    try:
        for motor_id in motors:
            label = side.get(motor_id, f"M{motor_id}")
            fwd = spin(board, motor_id, board.CW, args.duty, args.seconds,
                       f"{label} track, CW")
            rev = spin(board, motor_id, board.CCW, args.duty, args.seconds,
                       f"{label} track, CCW")
            print(f"    -> which of these drove the {label} track FORWARD? "
                  "Set the matching *_invert in egb320/config.py.")
            if fwd and rev and (fwd > 0) == (rev > 0):
                print("    -> encoder RPM stayed positive in both directions: "
                      "set drive.encoder_reports_signed = False in config.py")
            elif fwd and rev:
                print("    -> encoder RPM flipped sign with direction: "
                      "drive.encoder_reports_signed = True is correct")

        if not args.motor:
            print("\n>>> BOTH motors CW together (watch for a straight line, not a spin)")
            board.motor_movement(board.ALL, board.CW, args.duty)
            time.sleep(args.seconds)
            board.motor_stop(board.ALL)

        print("\nDone. Next: tools/bench_motor_rpm.py to calibrate max_wheel_rpm.")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            board.motor_stop(board.ALL)
        except Exception:
            pass


if __name__ == "__main__":
    main()
