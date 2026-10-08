"""
FPS profiler: finds out WHERE the time goes per frame.

Times each stage separately on live camera frames, then prints a table and a
plain-English verdict:

    camera.read()          capture + resize (+ colour conversion)
    marker mask            blur, threshold, morphology
    marker candidates      mask + contours + filtering
    marker detect (all)    candidates + warp + template matching
    full classify_frame    colour layer + markers + overlap removal
      -> colour layer      = classify_frame minus marker detect
    draw + imshow          what the display costs (skipped if no display)

Run from ~/vision, with the camera pointed at the markers (the cost of
template matching depends on how many blobs are in view):
    python3 profile_pipeline.py
"""

import time
import cv2
import numpy as np
from camera_capture_v2_0 import CameraCapture
from vision_system_v2_0 import VisionSystem

N = 40          # frames timed per stage
WARMUP = 5


def timed(fn, n=N):
    """Returns (mean_ms, worst_ms) of calling fn() n times."""
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(samples)), float(np.max(samples))


def main():
    print("Starting camera and vision system...")
    cam = CameraCapture()
    vs = VisionSystem()
    md = vs.marker_detector

    # a frame to run the processing stages on (grab a fresh one)
    frame = None
    for _ in range(10):
        frame = cam.read()
    if frame is None:
        print("[error] could not read a frame")
        return
    h, w = frame.shape[:2]
    print(f"Frame size reaching the pipeline: {w}x{h}\n")

    rows = []

    ms, worst = timed(cam.read)
    rows.append(("camera.read()", ms, worst))
    cam_ms = ms

    ms, worst = timed(lambda: md.make_mask(frame))
    rows.append(("marker: make_mask", ms, worst))

    ms, worst = timed(lambda: md.find_candidates(frame))
    rows.append(("marker: find_candidates", ms, worst))
    n_cands = len(md.find_candidates(frame))

    ms, worst = timed(lambda: md.detect(frame))
    rows.append(("marker: detect (all)", ms, worst))
    marker_ms = ms

    ms, worst = timed(lambda: vs.classify_frame(frame))
    rows.append(("classify_frame (total)", ms, worst))
    total_proc_ms = ms
    colour_ms = max(0.0, total_proc_ms - marker_ms)
    rows.append(("  -> colour layer (approx)", colour_ms, float("nan")))

    # display cost (only if a window can be opened)
    disp_ms = None
    try:
        dets = vs.classify_frame(frame)

        def show():
            f = frame.copy()
            vs.draw_detections(f, dets)
            cv2.imshow("profile", f)
            cv2.waitKey(1)

        disp_ms, worst = timed(show)
        rows.append(("draw + imshow", disp_ms, worst))
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"(skipping display timing: {e})")

    # whole loop as the vision system really runs it
    def one_loop():
        f = cam.read()
        d = vs.classify_frame(f)
        if disp_ms is not None:
            vs.draw_detections(f, d)
            cv2.imshow("profile", f)
            cv2.waitKey(1)

    loop_ms, loop_worst = timed(one_loop, n=60)
    cv2.destroyAllWindows()
    cam.release()

    print(f"{'stage':32s} {'mean ms':>9s} {'worst ms':>9s}")
    print("-" * 52)
    for name, ms, worst in rows:
        worst_s = "" if worst != worst else f"{worst:9.1f}"
        print(f"{name:32s} {ms:9.1f} {worst_s:>9s}")
    print("-" * 52)
    print(f"{'WHOLE LOOP':32s} {loop_ms:9.1f} {loop_worst:9.1f}   => {1000.0 / loop_ms:.1f} fps\n")
    print(f"marker candidates in this frame: {n_cands}   "
          f"template classes: { {k: len(v) for k, v in md.templates.items()} }\n")

    # ---- verdict ----
    parts = {"camera capture": cam_ms, "marker detector": marker_ms,
             "colour layer": colour_ms}
    if disp_ms is not None:
        parts["display"] = disp_ms
    slowest = max(parts, key=parts.get)
    print("VERDICT: the biggest cost is the", slowest.upper(),
          f"({parts[slowest]:.0f} ms of ~{loop_ms:.0f} ms per frame).")
    hints = {
        "camera capture": "Capture is the bottleneck. Lower CAPTURE_SIZE in camera_capture_v2_0.py "
                          "to (2304, 1296) -- still full field of view, far less data per frame -- "
                          "and raise TARGET_FPS to 30.",
        "marker detector": "Marker detection is the bottleneck. Fewer candidate blobs or fewer templates "
                           "helps: keep ~10 good templates per class, raise MIN_BLOB_AREA, and check "
                           "that the placard/ink calibration isn't letting lots of blobs through.",
        "colour layer": "The colour layer is the bottleneck (ORB matching against every reference "
                        "image for every blob). Colour-only classes skip it; ask me to add a cheap "
                        "pre-check or to run it every Nth frame.",
        "display": "Drawing to the screen is the bottleneck (common over VNC). Test with the window "
                   "turned off or shown only every few frames; it won't matter on the real robot "
                   "if it runs headless.",
    }
    print(hints[slowest])


if __name__ == "__main__":
    main()