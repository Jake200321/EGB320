"""Does the robot drive straight? Motors and encoders only -- no sonar, camera or maze.

    python3 straight_test.py                # all three steps, ~2.5 s of driving each
    python3 straight_test.py --seconds 3    # longer runs
    python3 straight_test.py --only 2       # just one step (1, 2 or 3)

Give it a metre or more of clear floor ahead. It asks before every run.

  1. STEERING CHECK -- commands a gentle LEFT turn and checks the encoders agree the
     right track went further. If steering and encoders disagree about which track is
     which, every heading correction pushes the robot AWAY from straight (it veers
     harder the more it "corrects"). Prints the fix.
  2. OPEN LOOP -- drives with no correction at all and compares how far the two tracks
     went. Tells you which way it pulls and the trim to put in motors.py.
  3. HEADING HOLD -- the same drive with the encoder heading hold the maze nav uses.
     Reports how far the heading wandered. If this is worse than step 2, the controller
     is the problem rather than the motors.

Nothing here edits a file: it prints the lines to change.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import motors as MOT
from maze import Odometry, wrap

TICK_S = 0.05                  # 20 Hz, like the nav loop


def _clamp(v, limit):
    return max(-limit, min(limit, v))


def drive_run(drive, odo, seconds, v, steer, label, clock=time):
    """Drive for `seconds`, calling steer(odo) -> yaw rate each tick. Returns a summary.

    Ticks are the encoder counts over the run (left, right); heading is the odometry's,
    in degrees, + = left of where it started.
    """
    start = drive.read_encoders()
    odo.reset(0.0, 0.0, 0.0)
    odo.update(start)
    heading_peak = 0.0
    t_end = clock.monotonic() + seconds
    try:
        while clock.monotonic() < t_end:
            odo.update(drive.read_encoders())
            drive.set_velocity(v, steer(odo))
            heading_peak = max(heading_peak, abs(math.degrees(odo.theta)))
            clock.sleep(TICK_S)
    finally:
        drive.stop()
    clock.sleep(0.3)
    end = drive.read_encoders()
    odo.update(end)
    dl, dr = end[0] - start[0], end[1] - start[1]
    return {"label": label, "dl": dl, "dr": dr, "heading": math.degrees(odo.theta),
            "peak": heading_peak, "x": odo.x, "y": odo.y}


def step1_steering(drive, odo, seconds, ticks_per_m):
    print("\n1) STEERING CHECK -- a gentle LEFT turn while moving; the RIGHT track must go further")
    r = drive_run(drive, odo, 1.0, 0.10, lambda o: 0.8, "steer left")
    print(f"   left track {r['dl']} ticks, right track {r['dr']} ticks; "
          f"odometry heading {r['heading']:+.1f} deg (should be +, i.e. turned left)")
    if r["dl"] <= 0 or r["dr"] <= 0:
        print("   !! A track counted DOWN while driving forward. Fix the encoder signs "
              "(ENCODER_LEFT_SIGN / ENCODER_RIGHT_SIGN in motors.py) -- both must count up.")
        return False
    if r["dr"] > r["dl"] * 1.05 and r["heading"] > 0:
        print("   OK: steering and encoders agree.")
        return True
    if r["dl"] > r["dr"] * 1.05:
        print("   !! MIRRORED: a left-turn command made the LEFT track go further. The motor "
              "channels and the encoder channels\n      disagree about which track is which.\n"
              f"      Set  SWAP_MOTORS = {not MOT.SWAP_MOTORS}  in motors.py (it is "
              f"{MOT.SWAP_MOTORS} now), then run this again.\n"
              "      (If the robot then turns the wrong way in motor_spin_test.py, swap the "
              "sprocket wiring instead.)")
        return False
    print("   ?? Inconclusive -- the tracks went about the same distance, so the turn "
          "command barely did anything.\n      Try a longer run or a higher speed.")
    return False


def trim_advice(dl, dr):
    """Which way it pulls, and the trim that evens the tracks out (trims are cumulative:
    this run already had the current trims applied)."""
    ratio = abs(dr) / max(abs(dl), 1)
    if abs(ratio - 1.0) <= 0.02:
        return ratio, "the tracks went within 2% of each other: trims are fine."
    if ratio > 1.0:
        new = MOT.RIGHT_TRIM / ratio
        return ratio, (f"RIGHT track went {100 * (ratio - 1):.0f}% further, so it pulls LEFT.\n"
                       f"   Lower the strong side:  RIGHT_TRIM = {new:.3f}  (now {MOT.RIGHT_TRIM})")
    new = MOT.LEFT_TRIM * ratio
    return ratio, (f"LEFT track went {100 * (1 / ratio - 1):.0f}% further, so it pulls RIGHT.\n"
                   f"   Lower the strong side:  LEFT_TRIM = {new:.3f}  (now {MOT.LEFT_TRIM})")


def step2_open_loop(drive, odo, seconds, ticks_per_m, v):
    print(f"\n2) OPEN LOOP -- {seconds:.1f} s at {v:.2f} m/s with NO correction")
    r = drive_run(drive, odo, seconds, v, lambda o: 0.0, "open loop")
    _, advice = trim_advice(r["dl"], r["dr"])
    dist = (abs(r["dl"]) + abs(r["dr"])) / 2 / ticks_per_m
    print(f"   left {r['dl']} ticks, right {r['dr']} ticks  (~{dist * 100:.0f} cm by TICKS_PER_M)")
    print(f"   heading drifted {r['heading']:+.1f} deg;  odometry says it ended {r['y'] * 100:+.1f} cm "
          "sideways")
    print("   " + advice)
    return r


def step3_hold(drive, odo, seconds, ticks_per_m, v, gain, max_rate):
    print(f"\n3) HEADING HOLD -- {seconds:.1f} s at {v:.2f} m/s, encoder heading hold "
          f"(gain {gain}, cap {max_rate})")
    r = drive_run(drive, odo, seconds, v,
                  lambda o: _clamp(gain * wrap(0.0 - o.theta), max_rate), "hold")
    print(f"   left {r['dl']} ticks, right {r['dr']} ticks")
    print(f"   heading wandered up to {r['peak']:.1f} deg, ended {r['heading']:+.1f} deg;  "
          f"{r['y'] * 100:+.1f} cm sideways")
    if r["peak"] > 6.0:
        print("   !! The hold let it wander more than 6 deg. Either it's fighting (try a "
              "lower HEADING_HOLD_GAIN) or the\n      tracks differ so much that the weak "
              "side is already at full power (lower the strong side's trim).")
    else:
        print("   OK: it held its heading.")
    return r


def ask(prompt):
    try:
        return input(prompt)
    except EOFError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--only", type=int, choices=(1, 2, 3))
    ap.add_argument("--speed", type=float, default=None, help="m/s (default: the nav's SEARCH_SPEED)")
    args = ap.parse_args()

    import main as M
    v = args.speed if args.speed is not None else M.SEARCH_SPEED
    drive = M.Drive()
    if not drive.board:
        raise SystemExit("no motor board -- nothing to test")
    odo = Odometry(M.TICKS_PER_M, M.EFFECTIVE_TRACK_M)
    print(f"motors.py: SWAP_MOTORS={MOT.SWAP_MOTORS}  LEFT_SIGN={MOT.LEFT_SIGN} "
          f"RIGHT_SIGN={MOT.RIGHT_SIGN}  encoder signs {MOT.ENCODER_LEFT_SIGN}/"
          f"{MOT.ENCODER_RIGHT_SIGN}  LEFT_TRIM={MOT.LEFT_TRIM} RIGHT_TRIM={MOT.RIGHT_TRIM}")
    print(f"main.py:   TICKS_PER_M={M.TICKS_PER_M:.0f}  speed {v:.2f} m/s  "
          f"HEADING_HOLD_GAIN={M.HEADING_HOLD_GAIN}  MAX_STEER_RATE={M.MAX_STEER_RATE}")
    try:
        steps = [args.only] if args.only else [1, 2, 3]
        for n in steps:
            ask(f"\nStep {n}: robot on clear floor, facing a metre or more of space. Enter to go... ")
            if n == 1:
                ok = step1_steering(drive, odo, args.seconds, M.TICKS_PER_M)
                if not ok and not args.only:
                    print("\nFix that first -- steps 2 and 3 are meaningless with steering "
                          "and encoders disagreeing.")
                    break
            elif n == 2:
                step2_open_loop(drive, odo, args.seconds, M.TICKS_PER_M, v)
            else:
                step3_hold(drive, odo, args.seconds, M.TICKS_PER_M, v,
                           M.HEADING_HOLD_GAIN, M.MAX_STEER_RATE)
    finally:
        drive.stop()
        drive.close()


if __name__ == "__main__":
    main()
