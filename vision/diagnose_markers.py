"""
Marker detection diagnostic.

Shows a live preview; press 'd' to capture that frame and print exactly what
MarkerDetector does with it:

    mask  ->  blobs  ->  filtered candidates  ->  per-class template scores

For EVERY candidate blob it prints its size and the best score against each
marker class, so you can see:
  * how many separate pieces one marker breaks into,
  * which class each piece matches and how close the runner-up is,
  * whether a piece is only labelled 'victim' because 'victim' is the only
    class with templates.

Saves (in the folder you run it from):
    debug_mask.png      what the detector treats as marker ink (white)
    debug_overlay.png   every candidate numbered; GREEN = accepted as a
                        marker, RED = best score below MARKER_THRESHOLD
    debug_cand_N.png    the flattened 128x128 crop of candidate N
    debug_cand_N_compare.png
                        live crop | best-matching template of each class,
                        best first, with scores -- shows WHY a class won

Run from ~/vision:
    python3 diagnose_markers.py
Press 'd' to diagnose the current frame (repeatable), 'q' to quit.
"""

import sys
import cv2
import numpy as np
from detectors.marker_detector import MarkerDetector
from camera_capture_v2_0 import CameraCapture
from config import MARKER_THRESHOLD

_mod = sys.modules[MarkerDetector.__module__]
MIN_BLOB_AREA = getattr(_mod, "MIN_BLOB_AREA", 600)
MIN_FILL_RATIO = getattr(_mod, "MIN_FILL_RATIO", 0.15)


def diagnose(md, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = md.make_mask(frame)
    cv2.imwrite("debug_mask.png", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
    big = [a for a in areas if a >= MIN_BLOB_AREA]

    print("\n================ DIAGNOSIS ================")
    print(f"Mask blobs found:                   {len(contours)}")
    print(f"  of those >= MIN_BLOB_AREA ({MIN_BLOB_AREA}):    {len(big)}")
    if areas:
        print(f"  largest areas: {[int(a) for a in areas[:8]]}")

    cands = md.find_candidates(frame)
    print(f"Candidates after fill/placard checks: {len(cands)}")
    print(f"Template classes loaded: "
          f"{ {k: len(v) for k, v in md.templates.items()} }  (3 templates per source photo)")
    print(f"MARKER_THRESHOLD = {MARKER_THRESHOLD}\n")

    overlay = frame.copy()
    accepted = 0
    for i, (c, rect) in enumerate(cands, start=1):
        box = (md.crop_points(c) if hasattr(md, "crop_points")
               else cv2.boxPoints(rect).astype(np.float32).reshape(4, 1, 2))
        roi = md._warp_perspective(gray, box)
        if roi is None:
            print(f"#{i}: warp failed")
            continue
        cv2.imwrite(f"debug_cand_{i}.png", roi)

        # Side-by-side sheet: live crop | best template of each class (best first).
        # Everything shown is what matchTemplate actually compared (equalized).
        per_class = md.best_templates(roi)
        order = sorted(per_class.items(), key=lambda kv: kv[1][0], reverse=True)
        tiles = []
        live = cv2.equalizeHist(roi)
        tiles.append(("LIVE", live))
        for cname, (sc, timg) in order:
            if timg is not None:
                tiles.append((f"{cname} {sc:.2f}", timg))
        sheet = []
        for label, img in tiles:
            tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            band = np.full((22, tile.shape[1], 3), 40, np.uint8)
            cv2.putText(band, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            sheet.append(np.vstack([band, tile]))
        cv2.imwrite(f"debug_cand_{i}_compare.png", np.hstack(sheet))

        scores = md.class_scores(roi)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best = ranked[0] if ranked else ("(no templates)", 0.0)
        runner = f", runner-up {ranked[1][0]} {ranked[1][1]:.2f}" if len(ranked) > 1 else ""

        (cx, cy), (rw, rh), _ = rect
        ok = best >= MARKER_THRESHOLD
        accepted += ok
        verdict = "ACCEPTED" if ok else "rejected (below threshold)"
        print(f"#{i}: area={cv2.contourArea(c):.0f}  box={rw:.0f}x{rh:.0f} at ({cx:.0f},{cy:.0f})")
        print(f"     best = {best_name} {best:.2f}{runner}  -> {verdict}")

        colour = (0, 255, 0) if ok else (0, 0, 255)
        cv2.polylines(overlay, [box.astype(np.int32)], True, colour, 2)
        cv2.putText(overlay, f"#{i} {best_name} {best:.2f}",
                    (int(cx - rw / 2), max(12, int(cy - rh / 2) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    cv2.imwrite("debug_overlay.png", overlay)
    print(f"\n{accepted} of {len(cands)} candidates would be reported as markers.")
    print("Saved debug_mask.png, debug_overlay.png, debug_cand_N.png,")
    print("      debug_cand_N_compare.png  (live crop next to the best template of each class)\n")

    print("How to read this:")
    print("  Many ACCEPTED candidates, all with area in the low hundreds")
    print("      -> one marker is breaking into pieces. Raise MIN_BLOB_AREA, or raise")
    print("         ICON_MERGE_KERNEL so the pieces merge into one blob.")
    print("  ACCEPTED, but the label is wrong (e.g. hazard marker -> victim)")
    print("      -> that class has no/too few templates. Capture templates for every")
    print("         marker type; check the runner-up score to see how close it was.")
    print("  Candidates all rejected -> templates don't resemble the live crops;")
    print("      look at debug_cand_N.png next to a template and recapture.")


def main():
    md = MarkerDetector()
    camera = CameraCapture()
    print("\nLive preview open. Click the window, aim at the markers.")
    print("Press 'd' to diagnose this frame (repeatable), 'q' to quit.\n")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("[error] could not read a frame from the camera")
                return
            cv2.imshow("Live preview (d=diagnose, q=quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                diagnose(md, frame.copy())
                print("Press 'd' again for another frame, 'q' to quit.\n")
            elif key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()