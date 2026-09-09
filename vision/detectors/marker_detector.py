"""
Wall-marker detector.

Finds rectangular black symbols on bright wall placards, flattens their
perspective, and classifies each against a library of reference templates
loaded from templates/<marker_class>/*.png -- one FOLDER per class, any
number of reference photos inside it. This mirrors the
reference_images/<class>/positive/ convention used for colour/shape
calibration elsewhere in this project, for the same reason: a single
reference photo isn't a representative sample of every angle/lighting a
live view will actually show. A live detection is compared against every
template in every class; whichever class scores the single best match
wins, the same "best-of-several-references" pattern the colour/shape
layer already uses.

No templates yet? Run capture_marker_templates.py to build the folders.

--- Edge-cleanup added ---
adaptiveThreshold alone tends to pick up small-scale noise (glare, shadow
texture, jpeg artefacting) along the marker's border, breaking what should
be one smooth rectangular edge into a jagged outline. A jagged outline has
a much longer perimeter than a clean rectangle of the same size, and
approxPolyDP's simplification tolerance scales with perimeter length --
so a noisy edge gets over-simplified and collapses to far fewer than 4
points instead of finding the 4 real corners. A Gaussian blur before
thresholding, plus a morphological "close" after, smooths the noise out
and reconnects small gaps so the contour comes out as a clean rectangle.

--- Disconnected-icon handling added ---
Some marker designs (e.g. a person icon with a separate plus-sign above
it) have real, deliberate white space between two black shapes rather
than a printed border tying them together. Thresholding sees that as two
SEPARATE blobs, not one -- so the merge kernel below is larger than the
noise-cleanup one, specifically to bridge that gap into a single
connected region. And since a merged multi-icon silhouette isn't
rectangular, this version no longer requires approxPolyDP to find exactly
4 corners (that check only made sense for a printed rectangular border).
Instead it uses cv2.minAreaRect, which finds the smallest ROTATED
rectangle that fully encloses whatever blob shape is there -- giving one
bounding box around the whole marker regardless of its internal shape.
"""

import cv2
import numpy as np
import os
import glob
from objects import DetectedObject
from config import MARKER_THRESHOLD

# Preprocessing constants
BLUR_KERNEL = (5, 5)
NOISE_CLOSE_KERNEL = np.ones((5, 5), np.uint8)     # smooths jagged edges
ICON_MERGE_KERNEL = np.ones((21, 21), np.uint8)    # bridges gaps between separate icon parts
# If your marker's icons are still coming out as separate blobs, increase
# ICON_MERGE_KERNEL's size (e.g. (31,31)); if it starts fusing DIFFERENT
# markers together when they're close in frame, decrease it instead.
MIN_FILL_RATIO = 0.15  # rejects extremely sparse/scattered blobs as noise

# Yaw tolerance: turning the marker around a VERTICAL axis (shawarma spit /
# ballet spin) is a genuinely different distortion to in-plane rotation --
# the card's face visually narrows and develops a trapezoid "keystone"
# shape as it turns away, because you're seeing less of its width but the
# same height. This synthesizes an approximate keystone warp: one vertical
# edge of the template gets pulled in toward the centre (top and bottom),
# same as what a camera actually sees when a flat card turns away on that
# side. It's a geometric approximation, not calibrated to your specific
# camera's field of view -- real reference photos at a turned angle will
# always match better than this if you have the time to add a couple, but
# this covers the gap cheaply from photos you already have.
YAW_AUGMENT_ANGLES = [-20, 20]


def _synthesize_yaw(img, yaw_deg):
    """Approximates how a flat template would look if the physical card
    were turned by yaw_deg around a vertical axis. The side turning away
    from the camera gets pulled inward (top and bottom) proportional to
    sin(yaw) -- this is the visual keystone/trapezoid effect of a flat
    surface rotating in 3D, which a plain 2D rotation cannot reproduce."""
    h, w = img.shape[:2]
    margin = int((w / 2) * abs(np.sin(np.radians(yaw_deg))))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if yaw_deg > 0:
        # right edge turning away -- pull it inward top and bottom
        dst = np.float32([[0, 0], [w, margin], [w, h - margin], [0, h]])
    else:
        # left edge turning away
        dst = np.float32([[0, margin], [w, 0], [w, h], [0, h - margin]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=255)


class MarkerDetector:
    def __init__(self, template_dir="templates"):
        self.template_dir = template_dir
        self.templates = {}   # class_name -> list of normalized grayscale template images
        self._load_templates()

    def _load_templates(self):
        """Loads every *.png/*.jpg under templates/<class_name>/, grouped by
        the folder name (the class), not the individual filename."""
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            print(f"[marker_detector] created empty '{self.template_dir}/' -- "
                  f"run capture_marker_templates.py to add reference symbols.")
            return

        class_folders = sorted(
            d for d in glob.glob(os.path.join(self.template_dir, "*"))
            if os.path.isdir(d)
        )

        if not class_folders:
            print(f"[marker_detector] no template classes found under "
                  f"'{self.template_dir}/' -- every marker will read as "
                  f"'unknown_marker' until you add some.")
            return

        for folder in class_folders:
            class_name = os.path.basename(folder)
            paths = (glob.glob(os.path.join(folder, "*.png"))
                     + glob.glob(os.path.join(folder, "*.jpg"))
                     + glob.glob(os.path.join(folder, "*.jpeg")))

            loaded = []
            for path in paths:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    base = cv2.resize(img, (128, 128))
                    # Equalize once here and store the result -- these
                    # templates never change after loading, so re-running
                    # equalizeHist on them every single frame (as before)
                    # was pure wasted work, and got noticeably worse once
                    # the yaw augmentation multiplied the template count.
                    loaded.append(cv2.equalizeHist(base))
                    loaded.extend(cv2.equalizeHist(_synthesize_yaw(base, a)) for a in YAW_AUGMENT_ANGLES)
                else:
                    print(f"[warning] could not read template '{path}'")

            if loaded:
                self.templates[class_name] = loaded
                n_source = len(paths)
                print(f"[marker_detector] loaded {n_source} source photo(s) -> "
                      f"{len(loaded)} template(s) (with rotation variants) for '{class_name}'")
            else:
                print(f"[warning] '{class_name}' folder has no readable images -- skipped")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Smooth small-scale noise (glare specks, jpeg artefacting) before
        # thresholding, so edges come through clean instead of jagged
        blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

        # Adaptive thresholding to isolate black shapes on bright white placards
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # First pass: fix jagged noise on individual edges
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, NOISE_CLOSE_KERNEL)
        # Second pass: bridge deliberate white space between separate icon
        # parts of the same marker (e.g. a figure and a plus sign above it)
        # into one connected blob
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ICON_MERGE_KERNEL)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 600:  # ignore small noise regions
                continue

            # Smallest ROTATED rectangle that fully encloses this blob --
            # works for a printed border (already rectangular) AND for a
            # merged multi-icon silhouette (not rectangular on its own,
            # but this always returns a clean 4-point bounding box for it)
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            rect_area = rw * rh
            if rect_area == 0 or (area / rect_area) < MIN_FILL_RATIO:
                # blob is too sparse/scattered relative to its bounding
                # rectangle to plausibly be one marker -- likely noise
                continue

            approx = cv2.boxPoints(rect).astype(np.float32).reshape(4, 1, 2)

            roi_warped = self._warp_perspective(gray, approx)
            if roi_warped is None:
                continue

            matched_type, score = self._classify_marker(roi_warped)
            if matched_type == "unknown_marker":
                continue

            obj = DetectedObject()
            obj.type = matched_type
            # Use the bounding box of the SAME region that was actually
            # classified (the tight minAreaRect box), not the raw contour
            # -- the raw contour can be inflated/offset if the merge-close
            # step above happened to bridge the marker to something nearby
            # (a shadow, an edge, wall texture), producing a box bigger
            # than and shifted from the real marker even when the
            # classification itself was correct.
            obj.bounding_box = cv2.boundingRect(approx.astype(np.int32))
            obj.confidence = score
            objects.append(obj)

        return objects

    def _warp_perspective(self, gray_frame, pts):
        """Flattens tilted wall placard perspective into a flat square."""
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        dst = np.array([
            [0, 0],
            [127, 0],
            [127, 127],
            [0, 127]
        ], dtype="float32")

        try:
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(gray_frame, M, (128, 128))
        except Exception:
            return None

    def _classify_marker(self, roi):
        """Compares the warped ROI against every template in every class,
        keeping the single best (highest) match across ALL exemplars of a
        class -- not just one reference image."""
        if not self.templates:
            return "unknown_marker", 0.0

        roi_norm = cv2.equalizeHist(roi)

        best_match = "unknown_marker"
        best_score = -1.0

        for class_name, templates in self.templates.items():
            for tmpl_norm in templates:
                # tmpl_norm is already equalized at load time -- no need to
                # redo it here every frame
                res = cv2.matchTemplate(roi_norm, tmpl_norm, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = max_val
                    best_match = class_name

        # Filter matches using MARKER_THRESHOLD defined in config.py
        if best_score < MARKER_THRESHOLD:
            return "unknown_marker", float(best_score)

        return best_match, float(best_score)