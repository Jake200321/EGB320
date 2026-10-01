"""Tests wallvision against rendered corridors with known geometry.

Renders what the real camera would see (pinhole, level, 10 cm up, same fields of view
as camera_capture_v2_0.py) from a robot standing in a 28 cm-cell corridor, then checks
the distances the detector reads off it. It proves the GEOMETRY is right -- not that
the thresholds suit your floor and walls; that needs wall_vision_test.py on the field.

    python3 test_wallvision.py        (needs numpy + opencv)
"""

import math
import sys

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"  SKIP  needs numpy + opencv ({exc})")
    sys.exit(0)

from wallvision import WallCamera

CELL = 0.28
CAM_H, CAM_TILT, HFOV, VFOV = 0.10, 0.0, 66.0, 41.0
CAM_FORWARD = 0.06           # camera ahead of the turning centre
WALL_H = 0.20
FRAME_W, FRAME_H = 640, 480
FLOOR, WALL, SKY = (50, 70, 60), (235, 235, 235), (40, 40, 40)   # BGR

failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def render(pos, heading, walls):
    """Frame seen from camera at ground position `pos`, facing `heading` (rad, world
    x east / y north), with wall segments [(x1, y1, x2, y2), ...]."""
    W, H = FRAME_W, FRAME_H
    fx = (W / 2) / math.tan(math.radians(HFOV) / 2)
    fy = (H / 2) / math.tan(math.radians(VFOV) / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dx, dy = (u - W / 2) / fx, (v - H / 2) / fy
    down, fwd = dy, np.ones_like(dy)               # level camera
    fx_w, fy_w = math.cos(heading), math.sin(heading)
    rx_w, ry_w = math.sin(heading), -math.cos(heading)   # right of heading
    gx = fwd * fx_w + dx * rx_w                     # ground direction per unit "dz"
    gy = fwd * fy_w + dx * ry_w
    img = np.full((H, W, 3), SKY, np.uint8)
    t_best = np.full((H, W), np.inf)
    # floor
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(down > 1e-6, CAM_H / down, np.inf)
    img[t_floor < np.inf] = FLOOR
    t_best = t_floor.copy()
    for (x1, y1, x2, y2) in walls:
        ex, ey = x2 - x1, y2 - y1
        px, py = x1 - pos[0], y1 - pos[1]
        den = gx * ey - gy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (px * ey - py * ex) / den           # along the ray
            s = (px * gy - py * gx) / den           # along the segment
        z_hit = CAM_H - down * t
        hit = (den != 0) & (t > 0) & (s >= 0) & (s <= 1) & (z_hit >= 0) & (z_hit <= WALL_H)
        hit &= t < t_best
        img[hit] = WALL
        t_best = np.where(hit, t, t_best)
    return img


def corridor(length_cells, left=True, right=True, front=True):
    """Corridor running north from y=0, x in [0, CELL]."""
    L = length_cells * CELL
    walls = []
    if left:
        walls.append((0.0, 0.0, 0.0, L))
    if right:
        walls.append((CELL, 0.0, CELL, L))
    if front:
        walls.append((0.0, L, CELL, L))
    return walls


def view_from(x, heading_deg, walls, y=CELL / 2):
    """Pose is the robot's turning centre; the camera sits CAM_FORWARD ahead of it."""
    th = math.radians(90 + heading_deg)             # 0 deg = facing north, +ve = left
    cam = (x + CAM_FORWARD * math.cos(th), y + CAM_FORWARD * math.sin(th))
    wc = WallCamera(CAM_H, CAM_TILT, HFOV, VFOV)
    return wc.look(render(cam, th, walls)), wc


def near(a, b, tol):
    return a is not None and abs(a - b) <= tol


print("1) centred in a 3-cell corridor, wall at the far end")
v, _ = view_from(CELL / 2, 0, corridor(3))
check("left wall 14 cm", near(v.left_m, 0.14, 0.02), f"got {v.left_m}")
check("right wall 14 cm", near(v.right_m, 0.14, 0.02), f"got {v.right_m}")
far = 3 * CELL - (CELL / 2 + CAM_FORWARD)
check(f"front wall {far*100:.0f} cm from the camera", near(v.front_m, far, 0.05),
      f"got {v.front_m}")
check("front not 'close'", not v.front_close)
check("walls parallel to heading", near(v.left_slope, 0, 0.06) and near(v.right_slope, 0, 0.06),
      f"slopes {v.left_slope}, {v.right_slope}")

print("2) 3 cm right of centre: the two distances split by 3 cm each way")
v, _ = view_from(CELL / 2 + 0.03, 0, corridor(3))
check("left 17 cm", near(v.left_m, 0.17, 0.02), f"got {v.left_m}")
check("right 11 cm", near(v.right_m, 0.11, 0.02), f"got {v.right_m}")

print("3) pointing 10 deg left: wall slopes show it")
v, _ = view_from(CELL / 2, 10, corridor(3))
check("both walls still found", v.left_m is not None and v.right_m is not None,
      f"L {v.left_m} R {v.right_m}")
if v.left_slope is not None and v.right_slope is not None:
    # facing left of the corridor: left wall closes in ahead, right wall falls away
    check("left wall converges, right wall diverges",
          v.left_slope < -0.08 and v.right_slope > 0.08,
          f"slopes {v.left_slope:.2f}, {v.right_slope:.2f} (tan10 = 0.18)")

print("4) wall right in front (the cell's own front wall): too close to range")
v, _ = view_from(CELL / 2, 0, corridor(1), y=CELL / 2)
check("front_close", v.front_close, f"front_m {v.front_m}")

print("5) opening on the left: only the right wall is seen")
walls = corridor(3, left=False)
v, _ = view_from(CELL / 2, 0, walls)
check("no left wall", v.left_m is None, f"got {v.left_m}")
check("right wall 14 cm", near(v.right_m, 0.14, 0.02), f"got {v.right_m}")

print("6) open corridor, no front wall in range: no front reading")
v, _ = view_from(CELL / 2, 0, corridor(6, front=False))
check("no front wall, not close", v.front_m is None and not v.front_close,
      f"front {v.front_m} close {v.front_close}")

print("7) nothing bright in view: nothing reported")
v, _ = view_from(CELL / 2, 0, [])
check("no walls at all", v.left_m is None and v.right_m is None and v.front_m is None
      and not v.front_close)

print("8) front wall plus a side opening: the front wall must not pass for a side wall")
walls = [(CELL, 0.0, CELL, 3 * CELL), (0.0, 3 * CELL, CELL, 3 * CELL)]   # left open
v, _ = view_from(CELL / 2, 0, walls)
check("left reads open despite the end wall in its half", v.left_m is None, f"got {v.left_m}")

print("8b) a REAL photo from the maze (test_images/real_corridor.png)")
# White panels with dark posts, beige carpet, uneven light -- the case the synthetic
# corridors can't cover. Robot is roughly centred looking down a 3-cell corridor, so:
# both walls about half a cell off, the far wall most of a metre away.
import os
_photo = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "test_images", "real_corridor.png"))
if _photo is None:
    print("  SKIP  test_images/real_corridor.png missing")
else:
    v = WallCamera(CAM_H, CAM_TILT, HFOV, VFOV).look(_photo)
    check("left wall found at roughly half a cell", v.left_m is not None and 0.10 <= v.left_m <= 0.22,
          f"got {v.left_m}")
    check("right wall found at roughly half a cell", v.right_m is not None and 0.10 <= v.right_m <= 0.22,
          f"got {v.right_m}")
    check("far wall found 60-110 cm away", v.front_m is not None and 0.6 <= v.front_m <= 1.1,
          f"got {v.front_m}")
    check("not mistaken for a wall right in front", not v.front_close)

print("9) speed")
import time
frame = render((0.14, 0.2), math.pi / 2, corridor(3))
wc = WallCamera(CAM_H, CAM_TILT, HFOV, VFOV)
wc.look(frame)
t0 = time.perf_counter()
for _ in range(50):
    wc.look(frame)
ms = (time.perf_counter() - t0) / 50 * 1000
check(f"{ms:.1f} ms per 640x480 frame on this machine", ms < 15)

print("\nALL PASSED" if not failures else f"\n{len(failures)} failed")
sys.exit(1 if failures else 0)
