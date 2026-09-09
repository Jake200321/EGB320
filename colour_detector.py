"""
Colour Detector
====================================================================
Version: v3 (colour-only detection layer, rebuilt from scratch --
proximity-clustered fragments, ambiguous-band groups resolved by the
shape layer rather than guessed here)

Colour-only object classification layer. Loads profiles.pkl (produced by
calibrate_v2_1_1.py) and, for each frame, returns every colour-matched
candidate blob together with the class name(s) it could be.

This is COLOUR ONLY, on purpose. It does NOT try to resolve two classes
that share a colour band -- in this project that's obstacle and door,
painted the exact same blue. Forcing a colour-only decision between them
would just be guessing. Instead, a blob matching a shared band comes back
labelled with candidate_classes = ["obstacle", "door"] and its outline
attached, so the shape layer (built next) can make the actual call using
what's genuinely different between them: a door's outline has a real gap
down the middle between two straight-edged bars; an obstacle's outline is
one solid, irregular/jagged blob with no long straight edges running
through it.

Everything else (ramp, victim, rubble) is assumed colour-distinct for now
and gets a single-class candidate label. If two of those turn out to
overlap too, add them to AMBIGUOUS_COLOUR_GROUPS below rather than trying
to force a colour-only guess between them either.

Run standalone to sanity-check colour detection BEFORE the shape layer
exists (works from anywhere -- run from the vision folder or from inside
detectors/, both resolve correctly):
    python3 detectors/colour_detector.py
Press 'q' to quit. Each candidate is drawn with its label(s); a yellow box
with two class names means "colour says it's one of these -- shape layer
still needs to decide."
"""

import os
import sys
import cv2
import numpy as np
import pickle

# This file lives in detectors/, but class_profile_v2_0.py and
# camera_capture_v2_0.py live one level up, in the main vision folder.
# When this module is imported normally (from detectors.colour_detector
# import ColourDetector, run from the vision folder) that parent folder
# is already on sys.path automatically. But run directly as a script
# (python3 detectors/colour_detector.py) Python only puts THIS file's own
# folder (detectors/) on sys.path -- so the parent needs to be added by
# hand for the standalone demo below to still work either way.
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from class_profile_v2_0 import ClassProfile  # noqa: F401 -- required so pickle can unpickle
from camera_capture_v2_0 import CameraCapture

# ---------------- Constants ----------------
PROFILES_FILE = "profiles.pkl"
MIN_CONTOUR_AREA = 400   # ignore colour blobs smaller than this (noise)

# Fragments of ONE physical object often land as several disconnected
# blobs in the colour mask (a highlight splitting the middle, a shadow
# cutting a corner off). Two blobs whose bounding boxes are within this
# many pixels of each other are treated as pieces of the SAME object and
# merged; anything further apart is a separate, unrelated blob.
CLUSTER_GAP_PX = 60

# The 60px value above is tuned to bridge gaps INSIDE one fragmented
# object (a highlight/shadow splitting one victim in two) -- it assumes
# there's only one real object of that colour to reassemble. That
# assumption is wrong for an AMBIGUOUS group: obstacle and door share a
# band precisely because they're two genuinely separate physical objects,
# and either can appear anywhere, including right next to the other. The
# 17x17 CLOSE step already bridges real internal fragmentation on its own;
# stacking the wider 60px cluster gap on top of that for this pass risks
# fusing a real obstacle and a real door sitting near each other into one
# blob, silently losing one of them. Use a much tighter gap here instead.
AMBIGUOUS_GROUP_CLUSTER_GAP_PX = 15

# A detection whose bounding box covers this much of the frame's width OR
# height is background/lighting bleed, not a real object -- reject it.
# Ramp is the one legitimate exception (see _find_clusters), because it
# regularly fills most/all of the frame at close range.
MAX_BBOX_FRAME_FRACTION = 0.9

# Classes painted in the exact same colour, so colour alone genuinely
# cannot tell them apart. A blob matching one of these groups' shared band
# is returned labelled with EVERY class in the group -- not resolved here.
AMBIGUOUS_COLOUR_GROUPS = [
    {"obstacle", "door"},
]
# --------------------------------------------


class ColourDetector:
    def __init__(self, profiles_file=PROFILES_FILE):
        with open(profiles_file, "rb") as f:
            self.classes = pickle.load(f)
        print(f"Loaded {len(self.classes)} class profile(s): {list(self.classes.keys())}")

        # names already spoken for by an ambiguous group, so the per-class
        # pass below doesn't also detect them on their own
        self._grouped_names = set()
        for group in AMBIGUOUS_COLOUR_GROUPS:
            self._grouped_names |= group

    # ---------------- Colour mask ----------------

    def _colour_mask(self, hsv_frame, profile):
        """Same hue-wrap-aware mask every detector in this project uses."""
        if not profile.hue_wraps:
            lower = np.array([max(0, profile.hue_low), profile.sat_low, profile.val_low])
            upper = np.array([min(179, profile.hue_high), profile.sat_high, profile.val_high])
            return cv2.inRange(hsv_frame, lower, upper)
        lower1 = np.array([0, profile.sat_low, profile.val_low])
        upper1 = np.array([min(profile.hue_high, 179), profile.sat_high, profile.val_high])
        lower2 = np.array([max(profile.hue_low, 0) % 180, profile.sat_low, profile.val_low])
        upper2 = np.array([179, profile.sat_high, profile.val_high])
        return cv2.inRange(hsv_frame, lower1, upper1) | cv2.inRange(hsv_frame, lower2, upper2)

    # ---------------- Blob clustering ----------------

    @staticmethod
    def _spans_full_frame(bbox, frame_shape):
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        return (w >= frame_w * MAX_BBOX_FRAME_FRACTION
                or h >= frame_h * MAX_BBOX_FRAME_FRACTION)

    @staticmethod
    def _union_bbox(box_a, box_b):
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1, y1 = min(ax, bx), min(ay, by)
        x2, y2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
        return (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _box_gap(box_a, box_b):
        """Pixel gap between two boxes' nearest edges -- 0 if they already
        touch or overlap on both axes."""
        ax1, ay1, aw, ah = box_a
        bx1, by1, bw, bh = box_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
        gap_y = max(0, max(ay1, by1) - min(ay2, by2))
        return max(gap_x, gap_y)

    @classmethod
    def _cluster_contours(cls, contours, gap_px):
        """Greedily groups contours whose bounding boxes are within gap_px
        of each other -- fragments of one physical object end up together;
        an unrelated blob elsewhere in frame that happens to pass the same
        colour test stays in its own cluster."""
        clusters = []
        for c in contours:
            bbox = cv2.boundingRect(c)
            merged = False
            for cluster in clusters:
                if cls._box_gap(cluster["bbox"], bbox) <= gap_px:
                    cluster["contours"].append(c)
                    cluster["bbox"] = cls._union_bbox(cluster["bbox"], bbox)
                    merged = True
                    break
            if not merged:
                clusters.append({"bbox": bbox, "contours": [c]})
        return clusters

    def _morph(self, mask):
        """CLOSE first, with a large kernel, so gaps INSIDE one object (a
        highlight, a shadow) get bridged back together before anything
        gets a chance to erode them apart -- doing OPEN first deletes
        those thin bridges permanently and is what causes one object to
        fragment into several small detections. OPEN afterward cleans up
        small isolated noise elsewhere in the frame."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return mask

    def _cluster_mask(self, mask, frame_shape, gap_px=CLUSTER_GAP_PX, allow_full_frame=False):
        """Contour -> cluster pipeline over an ALREADY-morphed mask. Kept
        separate from _morph so a caller that needs the processed mask
        afterward (e.g. to crop out a candidate's exact pixel mask for
        hole-checking) still has access to it -- morphing internally and
        throwing it away, like the old _find_clusters did, would lose
        exactly the pixel data the shape layer needs for the door/obstacle
        hole check."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        if not big:
            return []

        clusters = self._cluster_contours(big, gap_px)
        if not allow_full_frame:
            clusters = [cl for cl in clusters if not self._spans_full_frame(cl["bbox"], frame_shape)]
        return clusters

    # ---------------- Full colour pass ----------------

    def detect(self, frame):
        """Returns a list of candidates:
            {"candidate_classes": [...], "bbox": (x, y, w, h),
             "mask": uint8 crop, "contour": hull}
        candidate_classes has ONE name for a colour-distinct class, or
        several names when the blob's colour is shared between classes.

        "mask" is the actual 0/255 pixel crop for this candidate's bbox,
        straight from the (already CLOSE->OPEN morphed) colour mask -- this
        is what the shape layer needs for anything that cares about a HOLE
        inside the blob (the door/obstacle check), because "contour" below
        is a convex hull, and a convex hull fills in any hole by
        definition. Use "mask" for hole/hierarchy analysis, "contour" only
        for outline-shape checks (matchShapes, ORB ROI) that don't care
        about holes."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        candidates = []

        # ---- ambiguous groups first: one shared mask per group ----
        for group in AMBIGUOUS_COLOUR_GROUPS:
            profiles = [self.classes[n] for n in group
                        if n in self.classes and self.classes[n].hue_low is not None]
            if not profiles:
                continue

            mask = None
            for p in profiles:
                m = self._colour_mask(hsv, p)
                mask = m if mask is None else (mask | m)
            mask = self._morph(mask)

            # ramp isn't in this group today, but if a future ambiguous
            # group ever includes it, allow_full_frame keeps that working
            allow_full = "ramp" in group
            for cluster in self._cluster_mask(mask, frame.shape, gap_px=AMBIGUOUS_GROUP_CLUSTER_GAP_PX,
                                               allow_full_frame=allow_full):
                x, y, w, h = cluster["bbox"]
                candidates.append({
                    "candidate_classes": sorted(group),
                    "bbox": cluster["bbox"],
                    "mask": mask[y:y + h, x:x + w].copy(),
                    "contour": cv2.convexHull(np.vstack(cluster["contours"])),
                })

        # ---- every other class: its own mask, single-class candidates ----
        for name, profile in self.classes.items():
            if name in self._grouped_names or profile.hue_low is None:
                continue

            mask = self._morph(self._colour_mask(hsv, profile))
            # Ramp is the one class that's supposed to be able to fill
            # most/all of the frame at close range (it regularly gets
            # clipped by the frame edge), so it's exempt from the
            # full-frame rejection that every other class still gets.
            for cluster in self._cluster_mask(mask, frame.shape, allow_full_frame=(name == "ramp")):
                x, y, w, h = cluster["bbox"]
                candidates.append({
                    "candidate_classes": [name],
                    "bbox": cluster["bbox"],
                    "mask": mask[y:y + h, x:x + w].copy(),
                    "contour": cv2.convexHull(np.vstack(cluster["contours"])),
                })

        return candidates

    @staticmethod
    def draw_candidates(frame, candidates):
        for cand in candidates:
            x, y, w, h = cand["bbox"]
            ambiguous = len(cand["candidate_classes"]) > 1
            label = "/".join(cand["candidate_classes"]) + ("  (shape TBD)" if ambiguous else "")
            colour = (0, 255, 255) if ambiguous else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, label, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        return frame


def run_live():
    detector = ColourDetector()
    camera = CameraCapture()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                break

            candidates = detector.detect(frame)
            if candidates:
                print([c["candidate_classes"] for c in candidates])
            frame = detector.draw_candidates(frame, candidates)

            cv2.imshow("Colour Detector (colour-only -- shape layer comes next)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live()