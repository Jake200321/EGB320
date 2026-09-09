"""
Vision System
====================================================================
Version: v3 (rebuilt as three independent, modular layers -- colour,
shape, and marker detection -- combined only here. Filename kept as
vision_system_v2_0.py to match the existing run command; the version
number that matters is this comment, not the filename.)

Top-level orchestrator. Wires together the three independent detection
layers this project is now built from, keeping each one doing only its
own job -- this file's ONLY responsibility is combining their output, not
detecting anything itself:

    colour_detector.ColourDetector
        Finds coloured blobs. Classes with a distinct band become a
        single-class candidate; obstacle/door (same band) come back as an
        ambiguous two-class candidate, carrying its own pixel mask.

    shape_detector.resolve_candidates
        Resolves the one ambiguity colour can't: obstacle vs door, using
        the hole + straight-edge geometry check. Passes every other
        (already unambiguous) candidate straight through unchanged.

    detectors.marker_detector.MarkerDetector
        A completely separate modality -- grayscale template matching for
        wall placards. Has nothing to do with colour or the shape check
        above; its detections just get merged into the same output list.

None of these three modules import or know about each other. Swapping one
out (e.g. giving victim/rubble/ramp their own shape confirmation later)
or adding a fourth layer means editing THIS file's classify_frame, not the
layers themselves.

Run:
    python3 vision_system_v2_0.py
"""

import cv2
from detectors.colour_detector import ColourDetector
from detectors.shape_detector import resolve_candidates
from detectors.marker_detector import MarkerDetector
from camera_capture_v2_0 import (
    CameraCapture, pixel_x_to_angle, ground_distance_from_bbox_bottom,
    CAMERA_HEIGHT_M, CAMERA_TILT_DEG, VERTICAL_FOV_DEG,
)

# ---------------- Constants ----------------
# Confidence is assigned HERE, at the fusion level -- not inside either
# detection layer -- because it depends on how many layers actually had to
# agree. A class colour identified on its own (ramp, victim, rubble: each
# has a distinct band, nothing to resolve) is more certain than one colour
# could only narrow down to two options, with shape breaking the tie
# (obstacle vs door).
COLOUR_ONLY_CONFIDENCE = 0.8
SHAPE_RESOLVED_CONFIDENCE = 0.7

# Bounding boxes overlapping more than this, from ANY of the three layers,
# are treated as "the same object" -- keep only the higher-confidence one.
OVERLAP_IOU_THRESHOLD = 0.3

# Variance of the Laplacian is a standard cheap sharpness proxy: a crisp
# image has strong edges everywhere, so the second-derivative filter's
# output varies a lot pixel to pixel; a motion-blurred frame smears those
# edges out, so the variance collapses toward zero. A frame scoring below
# this is treated as unusable -- every downstream check (colour survives
# blur reasonably well, but ORB and template matching do not) gets worse
# on a blurred frame, so there's no point running any of them on one.
#
# This number is scene-dependent (a blank wall is "low variance" even
# perfectly in focus) -- set it by printing the score on frames you know
# are sharp vs. frames grabbed mid-pan at your actual driving speed, then
# picking a threshold that separates them.
BLUR_VARIANCE_THRESHOLD = 100.0
# --------------------------------------------


class VisionSystem:
    def __init__(self, profiles_file="profiles.pkl"):
        self.colour_detector = ColourDetector(profiles_file)
        self.marker_detector = MarkerDetector()

        self._ground_distance_configured = None not in (
            CAMERA_HEIGHT_M, CAMERA_TILT_DEG, VERTICAL_FOV_DEG
        )
        if not self._ground_distance_configured:
            print("[vision_system] CAMERA_HEIGHT_M / CAMERA_TILT_DEG / VERTICAL_FOV_DEG "
                  "not set in camera_capture_v2_0.py -- 'distance_m' will be None on "
                  "every detection until these are measured and configured.")

    # ---------------- Fusion ----------------

    @staticmethod
    def _blur_variance(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def classify_frame(self, frame):
        """Returns a list of detections, OR None if the frame was too
        motion-blurred to trust. None is deliberately NOT the same as an
        empty list: [] means "we looked carefully and found nothing
        there", None means "we didn't get usable information this cycle
        at all". Collapsing those into one signal would make a fast pan
        look identical to a genuinely empty scene to anything consuming
        this -- callers should hold their last known state on None, not
        treat it as confirmation something disappeared.

        Each detection carries "distance_m", the estimated ground-plane
        distance to the object -- EXCEPT markers, which get None on
        purpose (see ground_distance_from_bbox_bottom's docstring for why
        a wall-mounted marker can't use this method)."""
        if self._blur_variance(frame) < BLUR_VARIANCE_THRESHOLD:
            return None

        detections = []

        # ---- colour, then shape resolves what colour alone can't ----
        # These classes all rest on the floor, so ground-plane distance
        # is valid for them.
        candidates = self.colour_detector.detect(frame)
        for cand, resolved in zip(candidates, resolve_candidates(candidates)):
            was_ambiguous = len(cand["candidate_classes"]) > 1
            detections.append({
                "class": resolved["class"],
                "bbox": resolved["bbox"],
                "confidence": SHAPE_RESOLVED_CONFIDENCE if was_ambiguous else COLOUR_ONLY_CONFIDENCE,
                "is_ground_object": True,
            })

        # ---- wall markers -- independent modality, merged in as-is ----
        # NOT floor-level objects -- ground-plane distance does not apply.
        for marker in self.marker_detector.detect(frame):
            detections.append({
                "class": marker.type,
                "bbox": marker.bounding_box,
                "confidence": marker.confidence,
                "is_ground_object": False,
            })

        detections = self._suppress_overlaps(detections)
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        for d in detections:
            x, y, w, h = d["bbox"]
            centre_x = x + w / 2
            d["angle_deg"] = pixel_x_to_angle(centre_x, frame_width)

            if d["is_ground_object"] and self._ground_distance_configured:
                y_bottom = y + h
                d["distance_m"] = ground_distance_from_bbox_bottom(y_bottom, frame_height)
            else:
                d["distance_m"] = None
        return detections

    @staticmethod
    def _iou(box_a, box_b):
        """Intersection-over-union of two (x, y, w, h) boxes."""
        ax1, ay1, aw, ah = box_a
        bx1, by1, bw, bh = box_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter_area == 0:
            return 0.0

        union_area = aw * ah + bw * bh - inter_area
        return inter_area / union_area

    def _suppress_overlaps(self, detections):
        """Same rule regardless of which layer(s) produced the overlapping
        boxes: highest confidence first, discard anything that overlaps
        something already kept."""
        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept = []
        for d in detections:
            if not any(self._iou(d["bbox"], k["bbox"]) > OVERLAP_IOU_THRESHOLD for k in kept):
                kept.append(d)
        return kept

    # ---------------- Display ----------------

    @staticmethod
    def draw_detections(frame, detections):
        for d in detections:
            x, y, w, h = d["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dist_str = f', {d["distance_m"]:.2f}m' if d["distance_m"] is not None else ''
            label = f'{d["class"]} ({d["confidence"]:.0%}, {d["angle_deg"]:+.1f} deg{dist_str})'
            cv2.putText(frame, label, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if detections:
            found_classes = sorted(set(d["class"] for d in detections))
            status_text = f"True: {', '.join(found_classes)}"
            status_colour = (0, 255, 0)
        else:
            status_text = "False"
            status_colour = (0, 0, 255)

        (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        frame_h, frame_w = frame.shape[:2]
        x_pos = frame_w - text_w - 10
        y_pos = text_h + 15
        cv2.putText(frame, status_text, (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_colour, 2)

        return frame


def run_live(vision_system):
    import time

    camera = CameraCapture()
    last_detections = []  # most recent GENUINE result -- survives blurry frames

    DARK_RED = (0, 0, 139)  # BGR, not RGB -- OpenCV's colour order
    last_time = time.perf_counter()

    while True:
        frame = camera.read()
        if frame is None:
            break

        now = time.perf_counter()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        detections = vision_system.classify_frame(frame)
        if detections is None:
            # Blurred frame -- no new information this cycle, not
            # confirmation the scene is empty. Keep showing the last
            # genuine result instead of flashing to "nothing detected",
            # and log it distinctly so it's obvious which frames were
            # actually skipped versus genuinely empty.
            print("[blurry frame skipped]")
            display = vision_system.draw_detections(frame, last_detections)
            cv2.putText(display, "BLURRY -- showing last known state", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            last_detections = detections
            if detections:
                print([(d["class"], f'{d["confidence"]:.2f}', f'{d["angle_deg"]:+.1f} deg') for d in detections])
            display = vision_system.draw_detections(frame, detections)

        cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, DARK_RED, 2)

        cv2.imshow("Vision System (colour + shape + marker)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    vs = VisionSystem()
    run_live(vs)
