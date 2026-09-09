"""
Shape Detector
====================================================================
Version: v1 (first shape layer -- resolves the obstacle/door colour
ambiguity using a hole + straight-edge check; other classes have no
shape confirmation yet)

Resolves the one colour ambiguity colour_detector.py can't: obstacle
and door share a blue colour band, so a candidate comes back from the
colour layer as candidate_classes = ["door", "obstacle"] and it's this
module's job to pick one.

The physical distinction (confirmed against the real arena, not guessed):
  - door:     ONE continuous blue square frame with a real hole punched
              through the middle -- straight, manufactured edges.
  - obstacle: one jagged, irregular blue blob, no hole through it, no
              long straight edges.

Both of those are checkable directly from the pixel mask:
  1. HOLE CHECK -- cv2.findContours with RETR_CCOMP returns hierarchy, so
     child contours (holes) nested inside the outer blob are visible
     separately from the outer boundary. A door's mask has a hole; a solid
     obstacle blob (usually) doesn't.
  2. STRAIGHT-EDGE CHECK -- cv2.approxPolyDP simplifies a contour down to
     its dominant corners. A manufactured square frame collapses to a
     handful of vertices (~4, some slack for camera noise/perspective). A
     jagged rubble/obstacle silhouette does NOT collapse cleanly -- it
     keeps far more vertices even at the same simplification tolerance.

Requiring BOTH (not just the hole) is what keeps this safe even though
there's nothing else blue in the arena to confuse it with: the one
realistic false trigger is the CLOSE step accidentally sealing off a small
pocket inside an obstacle's own irregular outline, which would produce a
"hole" but NOT a low, door-like vertex count -- so it still reads as
obstacle.
"""

import cv2
import numpy as np

# ---------------- Constants ----------------
# approxPolyDP's simplification tolerance, as a fraction of the contour's
# own perimeter -- the standard way to scale this check regardless of the
# object's size/distance from the camera.
APPROX_EPSILON_RATIO = 0.02

# A door's square frame should collapse to about 4 vertices. This ceiling
# gives slack for camera noise, a slightly rounded manufactured corner, or
# a bit of perspective skew -- an obstacle's jagged outline will still
# blow well past this even with the same tolerance.
STRAIGHT_EDGE_MAX_VERTICES = 8

# A "hole" has to be a real fraction of the outer blob's area to count as
# a genuine opening, not a stray pixel-level gap left by thresholding
# noise or a small pocket the CLOSE step accidentally sealed off.
MIN_HOLE_AREA_RATIO = 0.08
# --------------------------------------------


def _largest_outer_contour(mask):
    """Returns (outer_contour, hole_contours) for the biggest top-level
    (parent == -1) contour in the mask, using RETR_CCOMP so holes show up
    as their own child contours instead of being invisible."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None, []

    hierarchy = hierarchy[0]  # cv2 wraps it in an extra dimension
    outer_indices = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not outer_indices:
        return None, []

    outer_idx = max(outer_indices, key=lambda i: cv2.contourArea(contours[i]))
    holes = [contours[i] for i, h in enumerate(hierarchy) if h[3] == outer_idx]
    return contours[outer_idx], holes


def classify_obstacle_or_door(mask):
    """mask: the candidate's own uint8 0/255 pixel crop, straight from
    ColourDetector (candidate["mask"] -- NOT candidate["contour"], which is
    a convex hull and has already had any hole filled in).

    Returns "door", "obstacle", or None if the mask has nothing usable in
    it (shouldn't normally happen -- the colour layer already filtered by
    MIN_CONTOUR_AREA before handing this candidate over)."""
    outer, holes = _largest_outer_contour(mask)
    if outer is None:
        return None

    outer_area = cv2.contourArea(outer)
    if outer_area <= 0:
        return "obstacle"

    perimeter = cv2.arcLength(outer, True)
    approx = cv2.approxPolyDP(outer, APPROX_EPSILON_RATIO * perimeter, True)
    straight_edged = len(approx) <= STRAIGHT_EDGE_MAX_VERTICES

    hole_area = max((cv2.contourArea(h) for h in holes), default=0.0)
    has_real_hole = (hole_area / outer_area) >= MIN_HOLE_AREA_RATIO

    if straight_edged and has_real_hole:
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
    single-class boxes, so obstacle/door can be checked without the rest
    of the pipeline in the way.

    Run with:  python3 detectors/shape_detector.py
    (works from the vision folder or from inside detectors/, same as
    colour_detector.py)"""
    import os
    import sys
    import cv2 as _cv2

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Try the package-style import first (this is how vision_system_v2_0.py
    # sees it -- detectors/ is a package, imported from the vision folder).
    # Fall back to a plain sibling import for when THIS file is run
    # directly as a script -- Python then only puts detectors/ itself on
    # sys.path, where colour_detector.py sits right next to it.
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
            if detections:
                print([d["class"] for d in detections])

            for d in detections:
                x, y, w, h = d["bbox"]
                _cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                _cv2.putText(frame, d["class"], (x, max(0, y - 8)),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            _cv2.imshow("Colour + Shape (obstacle/door resolved)", frame)
            if _cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        _cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live()