"""Wall detection from the camera -- the sonars' complement.

The sonars read one number each, close up, and miss anything that isn't square-on or
soft. The camera is the opposite: it can't see a wall's base nearer than the bottom
row of the frame allows (about 27 cm for a level camera 10 cm up with a 41 deg
vertical field), but it sees the whole corridor beyond that in one frame, 20 times a
second -- both walls at once, their distance, and which way the robot is pointing
against them.

HOW: the walls are neutral white panels, the floor is warm beige carpet -- so the test
is saturation (carpet ~35, walls ~7-25), not brightness, which shadow ruins. In each
image column the lowest wall-coloured run is the wall/floor boundary. That row is on the floor, so with the camera
height and tilt it fixes a ground point (forward Z, right X) exactly -- the same
geometry the victim ranging uses. Then:

  * FRONT: the boundary in the middle columns is the wall ahead. If the wall is
    closer than the bottom row can see, those columns are bright to the bottom edge
    and front_close is set instead of a distance.
  * SIDES: the boundary points on each half of the image all lie on that side wall,
    so a line fit X = a + b*Z gives its distance (a) and its angle to the robot's
    heading (b) -- from one frame, with no motion needed. Points from a front wall in
    the same half are thrown out by the line fit (RANSAC).

Pure numpy + OpenCV, no hardware, so it runs on a laptop. Everything that depends on
the real maze (floor and wall colours, shadows at the wall foot) is a constant below
-- tune with `python3 wall_vision_test.py` on the actual field.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# --- appearance: TUNE on the real maze ----------------------------------------
# Measured on a photo from the real maze (white panels, beige carpet, uneven light):
#     carpet   V 117-148   S 31-38      walls (lit or shaded)   V 140-189   S 7-19
# Brightness can't separate them -- a shaded wall (V~144) is as dark as the carpet --
# but saturation can: the carpet is warm, the walls are neutral. So the test is mostly
# "not colourful", with V_MIN only there to throw out the black posts and the gaps.
WALL_V_MIN = 90             # at least this bright (HSV value, 0-255): drops the posts
WALL_S_MAX = 28             # ...and at most this colourful (HSV saturation, 0-255).
                            # Carpet reads ~33-40, a warm-lit wall ~25. If carpet leaks
                            # into the mask lower this; if walls drop out raise it.
MIN_RUN_PX = 4              # a wall is this many bright rows in a row (kills specks)
POST_GAP_PX = 5             # dark posts every panel cut the wall into strips; gaps
                            # this narrow (at WORK_WIDTH) are bridged
WORK_WIDTH = 160            # frames are shrunk to this wide first: plenty for the
                            # geometry, and keeps it to a few ms on the Pi

# --- what the numbers mean -----------------------------------------------------
MAX_RANGE_M = 1.0           # boundary points further than this are too coarse to use
FRONT_HALF_DEG = 12.0       # middle columns that count as "straight ahead"
FRONT_CLOSE_FRACTION = 0.6  # of those, this many bright to the bottom = wall is close
FRONT_MIN_FRACTION = 0.25   # ...or this many with a boundary, to trust a distance
                            # (posts and shadows knock holes in a wall's boundary)
SIDE_MIN_DEG = 6.0          # side fits use columns at least this far off the nose
SIDE_INLIER_M = 0.015       # a boundary point this near the fitted line is on the wall
SIDE_MIN_POINTS = 6
SIDE_MIN_SPREAD_M = 0.06    # ...spread over at least this much forward distance
SIDE_MAX_SLOPE = 0.45       # a wall more than ~24 deg off the heading isn't "alongside"


@dataclass
class WallView:
    """What one frame says about the walls. None means "no wall found", which is not
    the same as "open" -- the camera only ever vouches for walls it sees."""
    t: float                         # time.monotonic() of the frame
    front_close: bool                # wall ahead closer than the camera can range
    front_m: Optional[float]         # ground distance, camera to wall ahead
    left_m: Optional[float]          # perpendicular distance, camera to left wall
    right_m: Optional[float]
    left_slope: Optional[float]      # d(distance)/d(forward): + = wall falling away
    right_slope: Optional[float]
    columns: int = 0                 # columns with a usable boundary (debug)

    def age(self, now=None):
        return (time.monotonic() if now is None else now) - self.t


class WallCamera:
    """Turns a BGR frame into a WallView."""

    def __init__(self, height_m, tilt_deg, hfov_deg, vfov_deg, width=WORK_WIDTH):
        self.h = height_m
        self.tilt = math.radians(tilt_deg)
        self.hfov, self.vfov = hfov_deg, vfov_deg
        self.width = width
        self._rng = np.random.default_rng(0)
        self._kernel = np.ones((MIN_RUN_PX, 1), np.uint8)
        self._open = np.ones((3, 3), np.uint8)
        self._bridge = np.ones((1, POST_GAP_PX), np.uint8)
        self._geom = None            # (W, H) the cached ray table was built for
        # for the debug overlay: boundary points in full-frame pixels
        self.last_boundary = None

    # -- geometry --------------------------------------------------------------
    def _rays(self, W, H):
        """Per-column ray terms, cached: lateral slope dx, and for any boundary row v
        the down/forward components follow from dy."""
        if self._geom != (W, H):
            fx = (W / 2.0) / math.tan(math.radians(self.hfov) / 2.0)
            fy = (H / 2.0) / math.tan(math.radians(self.vfov) / 2.0)
            self.fx, self.fy = fx, fy
            self.dx = (np.arange(W) + 0.5 - W / 2.0) / fx
            self._geom = (W, H)
        return self.dx

    def ground_point(self, dx, v, H):
        """Ground (forward Z, right X) for pixel column slope `dx` and pixel-edge row
        `v`, plus whether the ray actually reaches the floor."""
        dy = (np.asarray(v, dtype=float) - H / 2.0) / self.fy
        c, s = math.cos(self.tilt), math.sin(self.tilt)
        down = dy * c + s                       # camera looks down by `tilt`
        fwd = c - dy * s
        ok = down > 1e-3
        t = np.where(ok, self.h / np.where(ok, down, 1.0), 0.0)
        return t * fwd, t * dx, ok

    # -- the one call ----------------------------------------------------------
    def look(self, frame):
        h0, w0 = frame.shape[:2]
        W = self.width
        H = max(8, int(round(h0 * W / w0)))
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 2] >= WALL_V_MIN) & (hsv[:, :, 1] <= WALL_S_MAX))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, self._open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._bridge)   # across the posts
        # run[y, x] true when the MIN_RUN_PX rows ending at y are all wall
        run = cv2.erode(mask, self._kernel, anchor=(0, MIN_RUN_PX - 1)) > 0

        dx = self._rays(W, H)
        has_wall = run.any(axis=0)
        last = H - 1 - np.argmax(run[::-1, :], axis=0)   # lowest wall row per column
        reaches_bottom = run[H - 1, :]
        v_edge = last + 1.0                              # floor starts below that row
        Z, X, on_floor = self.ground_point(dx, v_edge, H)
        usable = has_wall & ~reaches_bottom & on_floor & (Z <= MAX_RANGE_M) & (Z > 0)

        bearing = np.degrees(np.arctan(dx))
        # ---- front ----
        centre = np.abs(bearing) <= FRONT_HALF_DEG
        n_centre = max(1, int(centre.sum()))
        close = bool((reaches_bottom & centre).sum() / n_centre >= FRONT_CLOSE_FRACTION)
        # ---- sides ----
        left = self._side(Z, -X, usable & (bearing <= -SIDE_MIN_DEG))
        right = self._side(Z, X, usable & (bearing >= SIDE_MIN_DEG))

        # ---- front distance, from the middle columns -- but not the far ends of the
        # side walls, which converge into them down a long corridor ----
        front = None
        if not close:
            ahead = usable & centre
            for side in (left, right):
                if side:
                    ahead[side[2]] = False
            z = Z[ahead]
            if len(z) / n_centre >= FRONT_MIN_FRACTION:
                front = float(np.median(z))

        sx = (np.arange(W) + 0.5) * (w0 / W)
        self.last_boundary = (sx[usable], v_edge[usable] * (h0 / H))
        return WallView(time.monotonic(), close, front,
                        left[0] if left else None, right[0] if right else None,
                        left[1] if left else None, right[1] if right else None,
                        int(usable.sum()))

    def _side(self, Z, dist, sel):
        """Fit |X| = a + b*Z through the boundary points on one side.

        Returns (perpendicular distance camera->wall, slope b, indices of the columns on
        the line) or None. RANSAC, because
        the points can include a front wall's, which sit on a different line.
        """
        idx = np.flatnonzero(sel)
        z, d = Z[sel], dist[sel]
        n = len(z)
        if n < SIDE_MIN_POINTS:
            return None
        best = None
        for _ in range(40):
            i, j = self._rng.choice(n, 2, replace=False)
            if abs(z[i] - z[j]) < 0.02:
                continue
            b = (d[j] - d[i]) / (z[j] - z[i])
            if abs(b) > SIDE_MAX_SLOPE:
                continue
            a = d[i] - b * z[i]
            inl = np.abs(d - (a + b * z)) <= SIDE_INLIER_M
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < SIDE_MIN_POINTS:
            return None
        zi, di = z[best], d[best]
        if zi.max() - zi.min() < SIDE_MIN_SPREAD_M:
            return None
        b, a = np.polyfit(zi, di, 1)
        if abs(b) > SIDE_MAX_SLOPE or a <= 0:
            return None
        return float(a / math.sqrt(1.0 + b * b)), float(b), idx[best]

    # -- debug -----------------------------------------------------------------
    def annotate(self, frame, view):
        """Draw the boundary it found and the numbers on a copy of the frame."""
        out = frame.copy()
        if self.last_boundary is not None:
            for x, y in zip(*self.last_boundary):
                cv2.circle(out, (int(x), int(y)), 2, (0, 255, 255), -1)

        def cm(v):
            return "--" if v is None else f"{v * 100:.0f}"
        front = "CLOSE" if view.front_close else cm(view.front_m)
        cv2.putText(out, f"wall cam  L {cm(view.left_m)}  F {front}  R {cm(view.right_m)} cm",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return out
