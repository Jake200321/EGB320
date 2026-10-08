"""
Marker template capture tool.

Run this ON THE PI, point the camera at a marker, and move the Pi around
it (closer, further, left, right, tilted, different lighting). Each saved
image is the marker as the DETECTOR sees it: found with the same
preprocessing as marker_detector.py, then perspective-flattened to
128x128 grayscale. That matters because live detections are flattened
the same way, so templates made this way compare like-for-like.

Saves to:  templates/<class_name>/<class_name>_001.png, _002.png, ...

Usage (with a screen / VNC attached):
    python capture_marker_templates.py person_plus

    Keys:  SPACE = save now (even if slightly blurry)    a = toggle auto-capture    q = quit

Usage (headless over SSH, no display):
    python capture_marker_templates.py person_plus --headless --count 40 --interval 1.5

    Saves automatically every 1.5 s while a marker is in view, and stops
    after 40 images. Move the Pi around slowly while it runs.
"""

import argparse
import os
import time

import cv2
import numpy as np

# Use the project's own camera wrapper so templates are captured with the
# SAME locked white balance, exposure, gain and colour order as live runs.
try:
    from camera_capture_v2_0 import CameraCapture
except ImportError:
    from camera_capture import CameraCapture   # rename to match your file

# Layout assumed:
#   vision/
#     capture_marker_templates.py   <- this script (run from inside vision/)
#     camera_capture_v2_0.py
#     objects.py, config.py
#     detectors/marker_detector.py
try:
    from detectors.marker_detector import (
        MarkerDetector,
        BLUR_KERNEL,
        NOISE_CLOSE_KERNEL,
        ICON_MERGE_KERNEL,
        MIN_FILL_RATIO,
    )
except ImportError:
    # fallback if marker_detector.py is in the same folder as this script
    from marker_detector import (
        MarkerDetector,
        BLUR_KERNEL,
        NOISE_CLOSE_KERNEL,
        ICON_MERGE_KERNEL,
        MIN_FILL_RATIO,
    )

# Read the blob-size cutoff from the detector itself so the two never drift
# apart (add MIN_BLOB_AREA to marker_detector.py, see instructions). Falls
# back to the old hard-coded 600 if the constant isn't there yet.
import sys
MIN_AREA = getattr(sys.modules[MarkerDetector.__module__], "MIN_BLOB_AREA", 600)
# Size/shape limits that stop the script grabbing the whole scene instead of
# the marker. The old code just took the LARGEST blob, so a big dark region
# (corner vignetting, a long shadow, a door frame, or the marker fused to the
# wall edge by the merge kernel) beat the real marker.
MAX_AREA_FRACTION = 0.40     # blob may cover at most 40% of the frame
EDGE_MARGIN = 4              # px; blob touching the frame edge is rejected
ASPECT_RANGE = (0.33, 3.0)   # allowed width/height of the bounding rectangle
MIN_SHARPNESS = 60.0    # Laplacian variance; below this the crop is too blurry.
                        # Lower it if good captures keep getting rejected,
                        # raise it if blurry ones are slipping through.


# ----------------------------------------------------------------------
# Marker finding: same pipeline as MarkerDetector.detect(), minus the
# classification step (we're making the references, so nothing to match yet)
# ----------------------------------------------------------------------
def find_marker(frame, detector):
    """Returns (warped_128x128_gray, box_points) for the best marker
    candidate in the frame (size-limited, not touching the edges, near the
    centre), or (None, None).

    Candidates come from the detector's own find_candidates() when it has
    one, so the capture script crops exactly what the live detector would
    (including the white-placard check)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    frame_area = H * W

    if hasattr(detector, "find_candidates"):
        cands = detector.find_candidates(frame)
    else:
        # Older detector: replicate its adaptive pipeline here
        blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, NOISE_CLOSE_KERNEL)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ICON_MERGE_KERNEL)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            if rw * rh == 0 or (cv2.contourArea(c) / (rw * rh)) < MIN_FILL_RATIO:
                continue
            cands.append((c, rect))

    best = None
    best_contour = None
    best_score = -1.0
    for c, rect in cands:
        area = cv2.contourArea(c)
        if area > MAX_AREA_FRACTION * frame_area:
            continue                              # far too big to be a marker

        x, y, w, h = cv2.boundingRect(c)
        if (x <= EDGE_MARGIN or y <= EDGE_MARGIN or
                x + w >= W - EDGE_MARGIN or y + h >= H - EDGE_MARGIN):
            continue                              # touches the frame edge: part of something bigger

        (cx, cy), (rw, rh), _ = rect
        aspect = rw / rh if rh else 0
        if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
            continue                              # too long and thin (shadow, edge, cable)

        # Prefer blobs near the middle of the frame (you're pointing the
        # camera AT the marker), weighted by size.
        half_diag = (W ** 2 + H ** 2) ** 0.5 / 2
        centre_dist = (((cx - W / 2) ** 2 + (cy - H / 2) ** 2) ** 0.5) / half_diag
        score = area * (1.0 - centre_dist)
        if score > best_score:
            best_score = score
            best = rect
            best_contour = c

    if best is None:
        return None, None

    if hasattr(detector, "crop_points"):
        box = detector.crop_points(best_contour)      # upright crop, same as the live detector
    else:
        box = cv2.boxPoints(best).astype(np.float32).reshape(4, 1, 2)
    warped = detector._warp_perspective(gray, box)
    return warped, box.astype(np.int32)


def sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def next_index(folder, class_name):
    """Continue numbering after any images already in the folder, so
    re-running the script never overwrites earlier captures."""
    existing = [f for f in os.listdir(folder) if f.startswith(class_name + "_")]
    nums = []
    for f in existing:
        stem = os.path.splitext(f)[0].rsplit("_", 1)[-1]
        if stem.isdigit():
            nums.append(int(stem))
    return (max(nums) + 1) if nums else 1


def save_template(warped, folder, class_name, idx):
    path = os.path.join(folder, f"{class_name}_{idx:03d}.png")
    cv2.imwrite(path, warped)
    return path


# ----------------------------------------------------------------------
def ask_class_name(template_dir):
    """Prompts for the marker class name, showing any classes that already
    exist so you can add more photos to one instead of mistyping it."""
    existing = []
    if os.path.isdir(template_dir):
        for d in sorted(os.listdir(template_dir)):
            full = os.path.join(template_dir, d)
            if os.path.isdir(full):
                count = len([f for f in os.listdir(full)
                             if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                existing.append((d, count))

    if existing:
        print("Existing marker classes:")
        for name, count in existing:
            print(f"  - {name} ({count} image(s))")
    else:
        print(f"No classes in '{template_dir}/' yet.")

    while True:
        name = input("Marker class name to capture (new or existing): ").strip()
        name = name.replace(" ", "_")   # spaces in folder names cause trouble later
        if name:
            return name
        print("Name can't be empty.")


def main():
    ap = argparse.ArgumentParser(description="Capture marker reference templates.")
    ap.add_argument("class_name", nargs="?", default=None,
                    help="folder/class name, e.g. person_plus "
                         "(if omitted, you'll be asked)")
    ap.add_argument("--template-dir", default="templates")
    ap.add_argument("--headless", action="store_true",
                    help="no window; auto-capture only (for SSH)")
    ap.add_argument("--interval", type=float, default=1.5,
                    help="seconds between auto-captures (default 1.5)")
    ap.add_argument("--count", type=int, default=40,
                    help="stop after this many saves (default 40)")
    args = ap.parse_args()

    if not args.class_name:
        args.class_name = ask_class_name(args.template_dir)

    folder = os.path.join(args.template_dir, args.class_name)
    os.makedirs(folder, exist_ok=True)
    idx = next_index(folder, args.class_name)

    # The detector is only used for its _warp_perspective helper; its
    # template loading is harmless here (it just prints what it finds).
    detector = MarkerDetector(template_dir=args.template_dir)
    cam = CameraCapture()

    auto = args.headless          # headless always auto-captures
    saved = 0
    last_save = 0.0

    print(f"[capture] saving to '{folder}/' starting at #{idx}")
    if args.headless:
        print(f"[capture] headless: auto-saving every {args.interval}s, "
              f"stopping after {args.count}. Ctrl+C to stop early.")
    else:
        print("[capture] SPACE = save, a = toggle auto-capture, q = quit")

    try:
        while saved < args.count:
            frame = cam.read()
            if frame is None:
                continue

            warped, box = find_marker(frame, detector)
            now = time.time()
            sharp = sharpness(warped) if warped is not None else 0.0
            usable = warped is not None and sharp >= MIN_SHARPNESS

            do_save = False
            if auto and usable and (now - last_save) >= args.interval:
                do_save = True

            if not args.headless:
                view = frame.copy()
                if box is not None:
                    color = (0, 255, 0) if usable else (0, 165, 255)
                    cv2.polylines(view, [box], True, color, 2)
                if warped is None:
                    state = "NO MARKER FOUND"
                elif not usable:
                    state = f"BLURRY ({sharp:.0f} < {MIN_SHARPNESS:.0f})"
                else:
                    state = f"READY ({sharp:.0f})"
                status = (f"{args.class_name}  saved {saved}/{args.count}  "
                          f"auto={'ON' if auto else 'off'}  {state}")
                cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
                cv2.imshow("capture", view)
                if warped is not None:
                    cv2.imshow("what gets saved", warped)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("a"):
                    auto = not auto
                elif key == ord(" "):
                    if warped is None:
                        print("[skipped] SPACE pressed but no marker was found "
                              "in this frame (check the box in the window)")
                    else:
                        if not usable:
                            print(f"[note] saving a soft/blurry crop "
                                  f"(sharpness {sharp:.0f} < {MIN_SHARPNESS:.0f}) "
                                  f"-- delete it later if it looks bad")
                        do_save = True

            if do_save:
                path = save_template(warped, folder, args.class_name, idx)
                print(f"[saved] {path}  (sharpness {sharp:.0f})")
                idx += 1
                saved += 1
                last_save = now

            if args.headless:
                time.sleep(0.05)   # don't spin the CPU while waiting
    except KeyboardInterrupt:
        print("\n[capture] stopped by user")
    finally:
        cam.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"[capture] done: {saved} new image(s) in '{folder}/'")
    print("Tip: open the folder and delete any bad ones before running the detector.")


if __name__ == "__main__":
    main()