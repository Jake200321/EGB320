"""Check the camera's wall detection on the real maze, and tune it.

    python3 wall_vision_test.py              # live: prints what it sees, ~4x a second
    python3 wall_vision_test.py --save       # also saves frame + mask + overlay to runs/
    python3 wall_vision_test.py --image x.jpg  # analyse a saved picture instead

Stand the robot dead centre in a cell with walls on both sides and look down the
corridor. You should see  L ~14  R ~14  (cm, half a cell) and F the distance to the
far wall. The yellow dots in the overlay should sit along the bottom edge of each wall.

If it's wrong, the saved *_mask.png shows what it took for wall (white):
  * wall has holes / floor is white  -> raise WALL_V_MIN or lower WALL_S_MAX
  * wall is dark / in shadow         -> lower WALL_V_MIN (in wallvision.py)
  * boundary reads too far           -> a shadow at the wall foot; lower WALL_V_MIN
  * L and R differ when centred      -> camera is off-centre or tilted (check
                                        CAMERA_TILT_DEG / CAMERA_HEIGHT_M)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "vision"))

import cv2
import numpy as np

import wallvision as W
from camera_capture_v2_0 import (CAMERA_HEIGHT_M, CAMERA_TILT_DEG, HORIZONTAL_FOV_DEG,
                                 VERTICAL_FOV_DEG)

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def describe(view):
    def cm(v):
        return "  --" if v is None else f"{v * 100:4.0f}"
    front = " CLOSE" if view.front_close else cm(view.front_m)
    slope = lambda s: "  --" if s is None else f"{s:+.2f}"
    return (f"L {cm(view.left_m)}  F {front}  R {cm(view.right_m)} cm   "
            f"slopes L {slope(view.left_slope)} R {slope(view.right_slope)}   "
            f"{view.columns} boundary pts")


def mask_of(frame):
    small = cv2.resize(frame, (W.WORK_WIDTH, int(frame.shape[0] * W.WORK_WIDTH / frame.shape[1])),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = (hsv[:, :, 2] >= W.WALL_V_MIN) & (hsv[:, :, 1] <= W.WALL_S_MAX)
    return cv2.resize((m * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def save(frame, view, wc, tag):
    os.makedirs(RUNS, exist_ok=True)
    base = os.path.join(RUNS, f"wall_{tag}")
    cv2.imwrite(base + "_frame.jpg", frame)
    cv2.imwrite(base + "_mask.png", mask_of(frame))
    cv2.imwrite(base + "_overlay.jpg", wc.annotate(frame, view))
    print(f"   saved {base}_frame.jpg / _mask.png / _overlay.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="analyse this picture instead of the camera")
    ap.add_argument("--save", action="store_true", help="save frame, mask and overlay")
    args = ap.parse_args()

    wc = W.WallCamera(CAMERA_HEIGHT_M, CAMERA_TILT_DEG, HORIZONTAL_FOV_DEG, VERTICAL_FOV_DEG)
    print(f"camera {CAMERA_HEIGHT_M * 100:.0f} cm up, tilt {CAMERA_TILT_DEG} deg, "
          f"{HORIZONTAL_FOV_DEG} x {VERTICAL_FOV_DEG} deg   "
          f"wall = V >= {W.WALL_V_MIN}, S <= {W.WALL_S_MAX}")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"can't read {args.image}")
        view = wc.look(frame)
        print(describe(view))
        save(frame, view, wc, "image")
        return

    from camera_capture_v2_0 import CameraCapture
    cam = CameraCapture()
    n, last = 0, 0.0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            view = wc.look(frame)
            if time.monotonic() - last >= 0.25:
                last = time.monotonic()
                print(describe(view))
                if args.save and n % 8 == 0:
                    save(frame, view, wc, f"{n:04d}")
                n += 1
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()


if __name__ == "__main__":
    main()
