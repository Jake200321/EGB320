"""
Shape Detector
====================================================================
Version: v2 (door/obstacle check rebuilt for the real door: a panel with a
square cut out, outlined in RED TAPE -- the same red as the obstacle)

Resolves the one colour ambiguity colour_detector.py can't: obstacle and
door are both red, so a candidate comes back from the colour layer as
candidate_classes = ["door", "obstacle"] and it's this module's job to
pick one.

What the camera actually sees in the red mask:
  - door:     a THIN red upside-down U: two vertical tape strips joined by
              a strip across the top, open at the bottom (the floor). Most
              of the box is empty -- the opening shows carpet/background,
              not red.
  - obstacle: a SOLID red lump. The middle of its box is red.

So the question is simply "is the middle of this red shape empty?".

Why this replaces the old hole + corner-count check
  The old check needed the red to form a CLOSED loop (so findContours
  could see a hole inside it) and then needed the outline to simplify to
  <= 8 corners. An upside-down U is open at the bottom, so it never has an
  enclosed hole -- the old check could not return "door" for it at all,
  and every door was called an obstacle.

  This version never looks at contours. It COUNTS pixels, so it doesn't
  matter whether the tape is open at the bottom or has a small gap from
  glare:

  1. FILL    = red pixels / area of the convex hull around them.
               U-frame ~ 0.3-0.6 (mostly empty), solid lump ~ 0.8-1.0.
  2. CENTRE  = fraction of the middle 40% x 40% of the box that is red.
               U-frame ~ 0.0 (that is the opening), solid lump ~ 1.0.

  Both must say "hollow" to call it a door. Needing both is the safety:
  a jagged obstacle with a dent near the middle fails the FILL test, and
  a stray gap that happens to land in the centre fails nothing on its own.

Run standalone to read the live numbers (shown above every blue/red
ambiguous box) and check the two thresholds against your REAL door and
REAL obstacle:
    python3 detectors/shape_detector.py
"""

import cv2
import numpy as np

# ---------------- Constants ----------------
# Door if FILL is at or below this. A U of tape is mostly empty space.
# If your real door reads higher than this (thick tape, small opening),
# raise it -- but keep it below what your obstacle reads.
DOOR_MAX_FILL = 0.75

# The middle window of the candidate's box, as fractions of its width and
# height. 0.30..0.70 = the central 40% on each axis. It must sit INSIDE the
# door's opening, so don't widen it past about 0.25..0.75.
CENTRE_WINDOW = (0.30, 0.70)

# Door if at most this fraction of the centre window is red.
DOOR_MAX_CENTRE_FILL = 0.15
# --------------------------------------------


def door_metrics(mask):
    """mask: the candidate's own uint8 0/255 pixel crop from ColourDetector
    (candidate["mask"] -- NOT candidate["contour"], a convex hull, which has
    already filled in any opening).

    Returns (fill, centre_fill), or None if the mask is empty/degenerate."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull_area = cv2.contourArea(cv2.convexHull(pts))
    if hull_area <= 0:
        return None
    fill = len(xs) / hull_area

    x, y, w, h = cv2.boundingRect(pts)
    lo, hi = CENTRE_WINDOW
    centre = mask[y + int(h * lo): y + int(h * hi) + 1,
                  x + int(w * lo): x + int(w * hi) + 1]
    # a window with no pixels in it can't prove anything is hollow
    centre_fill = cv2.countNonZero(centre) / centre.size if centre.size else 1.0

    return fill, centre_fill


def classify_obstacle_or_door(mask):
    """Returns "door", "obstacle", or None if the mask has nothing usable."""
    metrics = door_metrics(mask)
    if metrics is None:
        return None

    fill, centre_fill = metrics
    if fill <= DOOR_MAX_FILL and centre_fill <= DOOR_MAX_CENTRE_FILL:
        return "door"
    return "obstacle"


def resolve_candidates(candidates):
    """Takes ColourDetector.detect()'s output and returns a NEW list where
    every ["door", "obstacle"] candidate has been collapsed down to a
    single resolved class. Candidates that were already unambiguous pass
    through untouched."""
    resolved = []
    for cand in candidates:
        classes = cand["candidate_classes"]
        if set(classes) == {"door", "obstacle"}:
            final_class = classify_obstacle_or_door(cand["mask"]) or "obstacle"
        else:
            final_class = classes[0]

        resolved.append({
            "class": final_class,
            "bbox": cand["bbox"],
        })
    return resolved


def run_live():
    """Standalone demo: colour layer -> shape layer -> draw the resolved
    single-class boxes. For every door/obstacle candidate it also prints
    the two measured numbers, so you can hold up the real door and the
    real obstacle and see where each one lands relative to the thresholds.

    Run with:  python3 detectors/shape_detector.py
    (works from the vision folder or from inside detectors/, same as
    colour_detector.py)"""
    import os
    import sys

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        from detectors.colour_detector import ColourDetector
    except ImportError:
        from colour_detector import ColourDetector
    from camera_capture_v2_0 import CameraCapture

    detector = ColourDetector()
    camera = CameraCapture()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                break

            candidates = detector.detect(frame)
            detections = resolve_candidates(candidates)

            for cand, d in zip(candidates, detections):
                x, y, w, h = d["bbox"]
                label = d["class"]
                if len(cand["candidate_classes"]) > 1:
                    m = door_metrics(cand["mask"])
                    if m is not None:
                        label += f"  fill={m[0]:.2f} centre={m[1]:.2f}"
                        print(label)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Colour + Shape (obstacle/door resolved)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live()