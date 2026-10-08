"""
Wall-marker detector.

Finds rectangular black symbols on bright wall placards, flattens their
perspective, and classifies each against a library of reference templates
loaded from templates/<marker_class>/*.png -- one FOLDER per class, any
number of reference photos inside it. A live detection is compared against
every template in every class; whichever class scores the single best match
wins.

No templates yet? Run capture_marker_templates.py to build the folders.

--- Black-pixel mask (calibrated) ---
Markers are BLACK on white. In HSV terms black means a very low Value (V),
and a grey carpet sits at a mid V -- so a fixed V ceiling separates them,
unlike adaptiveThreshold, which flags anything locally darker than its
neighbours (carpet texture, darkened frame corners, shadows). Hue is
meaningless for black/grey/white, so only V (and a Saturation ceiling, to
exclude coloured props) are used.

Run calibrate_marker_threshold.py to choose the values; it saves
marker_threshold.json in the vision folder. If that file doesn't exist the
detector falls back to the old adaptiveThreshold behaviour, so nothing
breaks before you've calibrated.

--- White-placard check (second gate) ---
Dark alone is not enough: shadows, dark carpet patches and robot parts are
dark too. A real marker is black ink SURROUNDED BY WHITE PLACARD. For each
dark blob the detector looks at a ring just outside its bounding box and
requires most of that ring to be placard-white (V >= w_min, set in
calibrate_lighting_all.py). If w_min isn't calibrated yet, this check is
skipped.

--- Upright crop (important) ---
Each candidate is cropped with its UPRIGHT bounding box, not a rotated
rectangle. Markers hang vertically on walls and the camera is level, so
they appear upright. cv2.minAreaRect is unstable for round/compact symbols
(a hazard symbol has no obvious "up", so the fitted rectangle flips between
0 and 45 degrees from frame to frame with sensor noise). That made the same
marker produce differently-rotated crops -- and differently-rotated
templates -- which then failed to match each other. If you ever need to
detect markers that are genuinely rotated, this is the place to revisit.

--- Edge-cleanup ---
A Gaussian blur before thresholding plus a morphological "close" after
smooths noise so contours come out clean.

--- Disconnected-icon handling ---
Some markers (e.g. a person icon with a separate plus sign) have real white
space between black shapes. The large merge kernel bridges that gap into a
single blob, and cv2.minAreaRect gives one bounding box around the whole
marker regardless of its internal shape.
"""

import cv2
import numpy as np
import os
import glob
import json
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

# Smallest blob (in pixels of area) that gets warped and template-matched.
# Was a hard-coded 600. Lower = detects markers from further away, but more
# noise specks reach the (comparatively expensive) matching step.
MIN_BLOB_AREA = 200

# White-placard ring check. The ring is the area between the blob's box and
# the same box scaled up by SURROUND_RING_SCALE. At least SURROUND_MIN_WHITE
# of the ring's pixels must look like white placard.
SURROUND_RING_SCALE = 1.3
SURROUND_MIN_WHITE = 0.5

# Calibrated black-pixel thresholds, written by calibrate_lighting_all.py
THRESHOLD_FILENAME = "marker_threshold.json"

# Yaw tolerance: turning the marker around a VERTICAL axis is a genuinely
# different distortion to in-plane rotation -- the card's face narrows and
# develops a trapezoid "keystone" shape as it turns away. This synthesizes
# an approximate keystone warp from a straight-on template. Real reference
# photos at a turned angle always match better than this.
YAW_AUGMENT_ANGLES = [-20, 20]


def _synthesize_yaw(img, yaw_deg):
    """Approximates how a flat template would look if the physical card
    were turned by yaw_deg around a vertical axis."""
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
        self.v_max = None     # calibrated black-pixel ceilings (None = not calibrated)
        self.s_max = None
        self.w_min = None     # calibrated "placard white" floor (None or 0 = skip ring check)
        self._tmpl_matrix = None     # (N, 16384) zero-mean, unit-length template rows
        self._tmpl_names = []        # class name of each row
        self._tmpl_images = []       # the template image of each row
        self._load_threshold()
        self._load_templates()
        self._build_matrix()

    # ------------------------------------------------------------------
    def _load_threshold(self):
        """Looks for marker_threshold.json next to this file's parent folder
        (vision/), next to this file, then in the current directory."""
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "..", THRESHOLD_FILENAME),
            os.path.join(here, THRESHOLD_FILENAME),
            os.path.join(os.getcwd(), THRESHOLD_FILENAME),
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    self.v_max = int(data["v_max"])
                    self.s_max = int(data["s_max"])
                    self.w_min = int(data["w_min"]) if "w_min" in data else None
                    if self.w_min:
                        white_note = f", placard-white check ON (V>={self.w_min})"
                    else:
                        white_note = ", placard-white check OFF (run calibrate_lighting_all.py to enable it)"
                    print(f"[marker_detector] black threshold: V<={self.v_max}, "
                          f"S<={self.s_max}{white_note} (from {os.path.abspath(path)})")
                    return
                except Exception as e:
                    print(f"[marker_detector] could not read '{path}': {e}")
        print("[marker_detector] no marker_threshold.json found -- using adaptiveThreshold "
              "fallback. Run calibrate_lighting_all.py for a cleaner mask.")

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
                    # Equalize once here and store the result -- templates
                    # never change after loading.
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

    def _build_matrix(self):
        """Pre-computes every template as a zero-mean, unit-length row.

        For two images of the SAME size, cv2.TM_CCOEFF_NORMED is exactly the
        Pearson correlation: subtract each image's mean, then take the dot
        product of the two divided by the product of their lengths. Doing the
        template half of that once here means matching a live crop against
        ALL templates becomes ONE matrix-vector multiply (microseconds)
        instead of one cv2.matchTemplate call per template (tens of
        milliseconds with ~70 templates). The scores are identical."""
        rows, names, images = [], [], []
        for class_name, templates in self.templates.items():
            for t in templates:
                v = t.astype(np.float32).ravel()
                v -= v.mean()
                n = float(np.linalg.norm(v))
                rows.append(v / n if n > 1e-6 else np.zeros_like(v))
                names.append(class_name)
                images.append(t)
        self._tmpl_matrix = np.vstack(rows) if rows else None
        self._tmpl_names = names
        self._tmpl_images = images

    def _all_scores(self, roi):
        """Correlation of a warped 128x128 crop against EVERY template at once
        (same values cv2.matchTemplate(..., TM_CCOEFF_NORMED) would give)."""
        r = cv2.equalizeHist(roi).astype(np.float32).ravel()
        r -= r.mean()
        n = float(np.linalg.norm(r))
        if n < 1e-6:
            return np.zeros(len(self._tmpl_names), dtype=np.float32)
        return self._tmpl_matrix @ (r / n)

    # ------------------------------------------------------------------
    def make_mask(self, frame):
        """Returns a binary image where white = 'dark enough to be marker
        ink'. Shared with the capture/diagnostic scripts so they all see
        exactly what detect() sees."""
        blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)

        if self.v_max is not None:
            # Calibrated path: black = low Value, low Saturation. Grey carpet
            # (mid V) and coloured props (high S) fall outside this band.
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            thresh = cv2.inRange(hsv, (0, 0, 0), (179, self.s_max, self.v_max))
        else:
            # Fallback: old behaviour (anything locally darker than neighbours)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

        # First pass: fix jagged noise on individual edges
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, NOISE_CLOSE_KERNEL)
        # Second pass: bridge deliberate white space between separate icon
        # parts of the same marker into one connected blob
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ICON_MERGE_KERNEL)
        return thresh

    def make_placard_mask(self, frame):
        """White = pixel looks like white placard (bright AND colourless)."""
        blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (0, 0, self.w_min), (179, self.s_max, 255))

    def _ring_white_fraction(self, placard, rect):
        """Fraction of the ring around `rect` (its box scaled up by
        SURROUND_RING_SCALE, minus the box itself) that is placard-white.
        Ring pixels that fall outside the frame are not counted."""
        (cx, cy), (rw, rh), ang = rect
        H, W = placard.shape
        outer = cv2.boxPoints(((cx, cy), (rw * SURROUND_RING_SCALE, rh * SURROUND_RING_SCALE), ang))
        inner = cv2.boxPoints(((cx, cy), (rw, rh), ang))

        x0 = int(max(0, np.floor(outer[:, 0].min())))
        x1 = int(min(W, np.ceil(outer[:, 0].max()) + 1))
        y0 = int(max(0, np.floor(outer[:, 1].min())))
        y1 = int(min(H, np.ceil(outer[:, 1].max()) + 1))
        if x1 <= x0 or y1 <= y0:
            return 0.0

        off = np.array([x0, y0], dtype=np.float32)
        ring = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(ring, [np.round(outer - off).astype(np.int32)], 255)
        cv2.fillPoly(ring, [np.round(inner - off).astype(np.int32)], 0)
        ring_px = cv2.countNonZero(ring)
        if ring_px == 0:
            return 0.0
        white = cv2.countNonZero(cv2.bitwise_and(ring, placard[y0:y1, x0:x1]))
        return white / ring_px

    def find_candidates(self, frame):
        """Dark blobs that could be a marker: big enough, not too sparse and
        (if w_min is calibrated) sitting on a white placard. Returns a list
        of (contour, minAreaRect). Shared with the capture/diagnostic
        scripts so they all agree on what counts as a candidate."""
        thresh = self.make_mask(frame)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        placard = self.make_placard_mask(frame) if (self.w_min is not None and self.w_min > 0) else None

        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_BLOB_AREA:  # ignore small noise regions
                continue

            # Smallest ROTATED rectangle that fully encloses this blob
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            rect_area = rw * rh
            if rect_area == 0 or (area / rect_area) < MIN_FILL_RATIO:
                # too sparse/scattered relative to its bounding rectangle
                continue

            if placard is not None and self._ring_white_fraction(placard, rect) < SURROUND_MIN_WHITE:
                continue  # dark patch NOT sitting on a white placard

            out.append((c, rect))
        return out

    def crop_points(self, contour):
        """Corners (TL, TR, BR, BL) of the blob's UPRIGHT bounding box, in the
        shape _warp_perspective expects. Stable from frame to frame, unlike
        a rotated minAreaRect (see module docstring)."""
        x, y, w, h = cv2.boundingRect(contour)
        return np.array([[x, y], [x + w - 1, y],
                         [x + w - 1, y + h - 1], [x, y + h - 1]],
                        dtype=np.float32).reshape(4, 1, 2)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = []

        for c, rect in self.find_candidates(frame):
            approx = self.crop_points(c)

            roi_warped = self._warp_perspective(gray, approx)
            if roi_warped is None:
                continue

            matched_type, score = self._classify_marker(roi_warped)
            if matched_type == "unknown_marker":
                continue

            obj = DetectedObject()
            obj.type = matched_type
            # Bounding box of the SAME region that was actually classified
            obj.bounding_box = cv2.boundingRect(c)
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

    def best_templates(self, roi):
        """For each class: (best score, the template image that scored it).
        Used by diagnose_markers.py to show WHICH template won, side by side
        with the live crop."""
        out = {}
        if self._tmpl_matrix is None:
            return out
        scores = self._all_scores(roi)
        for i, class_name in enumerate(self._tmpl_names):
            v = float(scores[i])
            if class_name not in out or v > out[class_name][0]:
                out[class_name] = (v, self._tmpl_images[i])
        return out

    def class_scores(self, roi):
        """Best template-match score PER CLASS for a warped 128x128 crop
        (e.g. {'victim': 0.71, 'hazard': 0.44}). For debugging: shows how
        close the runner-up class was, which _classify_marker hides."""
        return {k: v[0] for k, v in self.best_templates(roi).items()}

    def _classify_marker(self, roi):
        """Compares the warped ROI against every template in every class,
        keeping the single best (highest) match across ALL exemplars."""
        if self._tmpl_matrix is None:
            return "unknown_marker", 0.0

        scores = self._all_scores(roi)
        i = int(np.argmax(scores))
        best_score = float(scores[i])

        # Filter matches using MARKER_THRESHOLD defined in config.py
        if best_score < MARKER_THRESHOLD:
            return "unknown_marker", best_score

        return self._tmpl_names[i], best_score