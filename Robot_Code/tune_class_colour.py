#!/usr/bin/env python3
"""Retune one class's HSV band in profiles.pkl, without a full recalibration.

    python3 tune_class_colour.py --show
    python3 tune_class_colour.py --class rubble --hue 170 10 --wrap --sat 120 255 --val 40 150
    python3 tune_class_colour.py --restore          # put the last backup back

Every write backs up profiles.pkl first (profiles.pkl.bak), because this edits
Kushal's calibration output in place and a bad band is worse than none -- a class
that matches half the room will produce detections that suppress real ones.

HSV here is OpenCV's convention, NOT the 0-360 one:
    hue 0-179   (so degrees / 2:  red 0/179, yellow 25, green 60, cyan 90, blue 120)
    sat 0-255
    val 0-255   -- lower this ceiling to reject bright, washed-out surfaces

RED WRAPS. Red sits at both ends of the hue range, so a red band needs --wrap with
hue_low ABOVE hue_high (e.g. --hue 170 10 --wrap), which the detector reads as
"170..179 OR 0..10". Without --wrap that same pair is an empty band and the class
silently stops matching anything at all.

This is a bench tool, not a substitute for recalibrating from reference images --
it has no idea what the real object looks like, only what you tell it.
"""

import argparse
import os
import pickle
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VISION_DIR = os.path.join(_REPO_ROOT, "vision")
sys.path.insert(0, _VISION_DIR)

PROFILES = os.path.join(_VISION_DIR, "profiles.pkl")
BACKUP = PROFILES + ".bak"


def load():
    import class_profile_v2_0                       # noqa: F401 -- pickle needs it
    with open(PROFILES, "rb") as f:
        return pickle.load(f)


def describe(profiles):
    print(f"{'class':10} {'hue':>12} {'wrap':>5} {'sat':>12} {'val':>12}")
    for name, p in sorted(profiles.items()):
        if p.hue_low is None:
            print(f"{name:10} {'(no colour band)':>12}")
            continue
        print(f"{name:10} {p.hue_low:5.0f}-{p.hue_high:<6.0f} {str(p.hue_wraps):>5} "
              f"{p.sat_low:5.0f}-{p.sat_high:<6.0f} {p.val_low:5.0f}-{p.val_high:<6.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--show", action="store_true", help="print every band and exit")
    ap.add_argument("--class", dest="cls", help="class to retune")
    ap.add_argument("--hue", nargs=2, type=float, metavar=("LOW", "HIGH"))
    ap.add_argument("--sat", nargs=2, type=float, metavar=("LOW", "HIGH"))
    ap.add_argument("--val", nargs=2, type=float, metavar=("LOW", "HIGH"))
    ap.add_argument("--wrap", action="store_true", help="band crosses the 0/180 hue seam")
    ap.add_argument("--no-wrap", action="store_true")
    ap.add_argument("--restore", action="store_true", help="restore profiles.pkl.bak")
    args = ap.parse_args()

    if args.restore:
        if not os.path.exists(BACKUP):
            raise SystemExit(f"no backup at {BACKUP}")
        shutil.copy(BACKUP, PROFILES)
        print(f"restored {PROFILES} from backup")
        describe(load())
        return

    profiles = load()
    if args.show or not args.cls:
        describe(profiles)
        if not args.cls:
            print("\nnothing changed -- pass --class NAME with --hue/--sat/--val to edit")
        return

    if args.cls not in profiles:
        raise SystemExit(f"no class {args.cls!r} -- have {sorted(profiles)}")
    p = profiles[args.cls]

    print("before:")
    describe({args.cls: p})

    if args.hue:
        p.hue_low, p.hue_high = args.hue
    if args.sat:
        p.sat_low, p.sat_high = args.sat
    if args.val:
        p.val_low, p.val_high = args.val
    if args.wrap:
        p.hue_wraps = True
    if args.no_wrap:
        p.hue_wraps = False

    # The one mistake that silently disables a class rather than erroring.
    if p.hue_low > p.hue_high and not p.hue_wraps:
        raise SystemExit(
            f"hue_low ({p.hue_low:.0f}) is above hue_high ({p.hue_high:.0f}) with wrap off "
            "-- that band matches nothing. Add --wrap if you meant to cross the red seam.")

    shutil.copy(PROFILES, BACKUP)
    with open(PROFILES, "wb") as f:
        pickle.dump(profiles, f)
    print(f"\nafter (backup at {os.path.basename(BACKUP)}):")
    describe({args.cls: p})


if __name__ == "__main__":
    main()
