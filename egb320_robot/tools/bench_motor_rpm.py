#!/usr/bin/env python3
"""STEP TWO: measure the numbers config.py is currently guessing.

Closes three open items from `nav Software/README.md`:
  * real no-load output-shaft RPM at the actual pack voltage (datasheet only gives
    430 rpm @ 12 V, and we run 6 V motors)
  * the duty at which the geartrain actually starts moving -> drive.min_duty_percent
  * whether the board's reported RPM is truthful -> drive.encoder_rpm_scale

    PUT THE ROBOT ON A BLOCK. Tracks off the ground for --sweep.

    python3 tools/bench_motor_rpm.py --sweep            # duty -> RPM curve, both motors
    python3 tools/bench_motor_rpm.py --stiction         # find the lowest duty that moves
    python3 tools/bench_motor_rpm.py --verify --duty 40 # hand-count revs to check the board

Note on the x2/x4 quadrature question in config.py: it does NOT affect this script.
The DFR0592 does its own pulse->RPM conversion onboard, so `--verify` measures the
whole chain at once and `encoder_rpm_scale` absorbs any error, including a wrong
pulses-per-rev assumption in the board firmware. `quadrature_decode` only matters for
turning RPM into ticks in dfr0592.read_encoders(), and since those ticks are integrated
rather than counted, a constant scale error there is equivalent to a wheel-radius error.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egb320.config import Config
from tools.motor_spin_test import connect   # noqa: E402


def settle_rpm(board, motor_id, duty, orientation=None, settle_s=1.5, sample_s=1.5):
    """Run at `duty`, wait for the speed to settle, then average."""
    orientation = orientation if orientation is not None else board.CW
    board.motor_movement(motor_id, orientation, duty)
    time.sleep(settle_s)
    samples, deadline = [], time.time() + sample_s
    while time.time() < deadline:
        samples.append(abs(float(board.get_encoder_speed(board.ALL)[motor_id - 1])))
        time.sleep(0.1)
    board.motor_stop(motor_id)
    return sum(samples) / len(samples) if samples else 0.0


def sweep(board, cfg, motors, top_duty):
    print(f"\nduty -> output-shaft RPM (capped at {top_duty:.0f} %)")
    print(f"{'duty %':>7} | " + " | ".join(f"{'M'+str(m)+' rpm':>10}" for m in motors))
    print("-" * (9 + 13 * len(motors)))
    best = {m: 0.0 for m in motors}
    for duty in range(10, int(top_duty) + 1, 10):
        row = []
        for m in motors:
            rpm = settle_rpm(board, m, float(duty))
            best[m] = max(best[m], rpm)
            row.append(f"{rpm:>10.1f}")
            time.sleep(0.3)
        print(f"{duty:>7} | " + " | ".join(row))

    top = min(best.values())
    print(f"\n  slowest motor at {top_duty:.0f} % duty: {top:.1f} rpm")
    print(f"  -> set in egb320/config.py:  DriveConfig.max_wheel_rpm = {top:.0f}")
    print(f"     (use the SLOWER motor -- max_duty_percent must be reachable by both)")
    circ_m = 3.14159265 * cfg.geometry.sprocket_pitch_diameter_m
    print(f"  -> top speed on 24 mm sprockets: {top / 60.0 * circ_m:.3f} m/s "
          f"(nav.forward_speed_mps is currently {cfg.nav.forward_speed_mps})")


def stiction(board, motors):
    print("\nlowest duty that gets the shaft moving:")
    for m in motors:
        found = None
        for duty in range(5, 71, 5):
            rpm = settle_rpm(board, m, float(duty), settle_s=1.0, sample_s=0.8)
            print(f"  M{m} @ {duty:>3} % -> {rpm:>7.1f} rpm")
            if rpm > 1.0:
                found = duty
                break
            time.sleep(0.3)
        print(f"  M{m}: starts at {found} %" if found
              else f"  M{m}: never moved up to 70 % -- check power and mechanical binding")
    print("\n  -> set DriveConfig.min_duty_percent to the HIGHER of the two, plus ~5 % margin")


def verify(board, cfg, motor_id, duty, seconds):
    print(f"\nMark the sprocket. Running M{motor_id} at {duty:.0f} % for {seconds:.0f}s.")
    print("Count the OUTPUT SHAFT revolutions (or film it and count frames).")
    input("Press Enter to start... ")
    board.motor_movement(motor_id, board.CW, duty)
    samples, t0 = [], time.time()
    while time.time() - t0 < seconds:
        samples.append(abs(float(board.get_encoder_speed(board.ALL)[motor_id - 1])))
        time.sleep(0.1)
    board.motor_stop(motor_id)
    elapsed = time.time() - t0

    reported_rpm = sum(samples) / len(samples) if samples else 0.0
    reported_revs = reported_rpm / 60.0 * elapsed
    print(f"\nboard reported {reported_rpm:.1f} rpm over {elapsed:.1f}s "
          f"= {reported_revs:.1f} revolutions")
    try:
        counted = float(input("revolutions you actually counted: "))
    except ValueError:
        print("skipped")
        return
    if reported_revs <= 0:
        print("board reported nothing -- fix the encoder wiring first")
        return
    print(f"\n  -> set DriveConfig.encoder_rpm_scale = {counted / reported_revs:.4f}")
    if abs(counted / reported_revs - 2.0) < 0.15:
        print("     (~2.0 -- the board is counting x2 where we assumed x4, or vice versa)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--stiction", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--motor", type=int, choices=[1, 2], default=None)
    ap.add_argument("--duty", type=float, default=40.0, help="duty for --verify")
    ap.add_argument("--seconds", type=float, default=10.0, help="run time for --verify")
    args = ap.parse_args()

    if not (args.sweep or args.stiction or args.verify):
        ap.error("pick one of --sweep / --stiction / --verify")

    cfg = Config()
    board = connect(cfg)
    board.set_encoder_enable(board.ALL)
    board.set_encoder_reduction_ratio(board.ALL, int(cfg.geometry.gear_ratio))
    board.set_moter_pwm_frequency(cfg.drive.pwm_frequency_hz)
    board.motor_stop(board.ALL)

    motors = [args.motor] if args.motor else [cfg.hardware.motor_left_id,
                                              cfg.hardware.motor_right_id]
    try:
        if args.stiction:
            stiction(board, motors)
        if args.sweep:
            sweep(board, cfg, motors, cfg.drive.max_duty_percent)
        if args.verify:
            verify(board, cfg, motors[0], args.duty, args.seconds)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            board.motor_stop(board.ALL)
        except Exception:
            pass


if __name__ == "__main__":
    main()
