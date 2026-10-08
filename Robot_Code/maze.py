"""The maze map, the planners and odometry -- pure logic, no hardware.

Ported from the sim's nav/wall_map.py, where frontier exploration + flood fill was
the only strategy that completed the full mission dependably (see Nav Research/
Algorithm comparison). Nothing here touches GPIO, I2C or the camera, so all of it
runs and is tested on a laptop.

FRAMES
    Cells are (col, row). Row 0 is the NORTH edge of the maze, col 0 the WEST edge
    -- the same convention as the sim and EGB320_Milestone2_Mazes.py.

    World coordinates are metres, x EAST and y NORTH, origin at the maze's
    south-west corner. Heading is radians anticlockwise from EAST, so NORTH is
    +pi/2. That's the ordinary maths convention, which keeps the odometry sums
    textbook: x += ds*cos(theta), y += ds*sin(theta).

WALLS
    Each wall sits between two cells and is True (present), False (open) or None
    (not seen yet). Setting one side sets its neighbour's matching side too. The
    perimeter is seeded as present. A wall found the hard way -- a drive that hit
    something the sonar said was open -- is locked, so a later noisy reading can't
    reopen it.
"""

import math
from collections import deque

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
HEADINGS = (NORTH, EAST, SOUTH, WEST)
HEADING_NAME = {NORTH: "N", EAST: "E", SOUTH: "S", WEST: "W"}
HEADING_BY_NAME = {v: k for k, v in HEADING_NAME.items()}

# (dcol, drow) for one step. Row 0 is north, so NORTH decreases the row.
STEP = {NORTH: (0, -1), EAST: (1, 0), SOUTH: (0, 1), WEST: (-1, 0)}
HEADING_RAD = {EAST: 0.0, NORTH: math.pi / 2, WEST: math.pi, SOUTH: -math.pi / 2}

OPPOSITE = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST}
LEFT_OF = {NORTH: WEST, WEST: SOUTH, SOUTH: EAST, EAST: NORTH}
RIGHT_OF = {NORTH: EAST, EAST: SOUTH, SOUTH: WEST, WEST: NORTH}


def wrap(angle):
    """Angle into (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def nearest_heading(theta):
    """The cardinal heading closest to a world angle."""
    return min(HEADINGS, key=lambda h: abs(wrap(theta - HEADING_RAD[h])))


class Grid:
    """Maze geometry: which cell is where."""

    def __init__(self, cols, rows, cell_m):
        self.cols, self.rows, self.cell_m = cols, rows, cell_m

    def in_bounds(self, cell):
        return 0 <= cell[0] < self.cols and 0 <= cell[1] < self.rows

    def neighbour(self, cell, heading):
        dc, dr = STEP[heading]
        return (cell[0] + dc, cell[1] + dr)

    def heading_between(self, a, b):
        """Heading that steps from cell a to adjacent cell b, or None."""
        for h in HEADINGS:
            if self.neighbour(a, h) == b:
                return h
        return None

    def centre(self, cell):
        col, row = cell
        return ((col + 0.5) * self.cell_m, (self.rows - row - 0.5) * self.cell_m)

    def cell_at(self, x, y):
        """Cell containing a world point, clamped onto the grid."""
        col = int(math.floor(x / self.cell_m))
        row = self.rows - 1 - int(math.floor(y / self.cell_m))
        return (min(max(col, 0), self.cols - 1), min(max(row, 0), self.rows - 1))


class MazeMap(Grid):
    """What the robot knows about the maze: walls, where it's been, what's blocked."""

    def __init__(self, cols, rows, cell_m):
        super().__init__(cols, rows, cell_m)
        self._walls = {}           # (cell, heading) -> True / False
        self._locked = set()       # (cell, heading) proven by a failed drive
        self.visited = set()       # cells whose walls have been sensed
        self.blocked = set()       # cells that can't be entered at all
        for col in range(cols):                   # the outside is always there
            self.set_wall((col, 0), NORTH, True, lock=True)
            self.set_wall((col, rows - 1), SOUTH, True, lock=True)
        for row in range(rows):
            self.set_wall((0, row), WEST, True, lock=True)
            self.set_wall((cols - 1, row), EAST, True, lock=True)

    # -- storage -------------------------------------------------------------
    def set_wall(self, cell, heading, present, lock=False):
        keys = [(cell, heading)]
        other = self.neighbour(cell, heading)
        if self.in_bounds(other):
            keys.append((other, OPPOSITE[heading]))
        if not lock and any(k in self._locked for k in keys):
            return                                # proven by contact; sonar can't undo it
        for k in keys:
            self._walls[k] = present
            if lock:
                self._locked.add(k)

    def wall(self, cell, heading):
        """True / False / None (unknown)."""
        return self._walls.get((cell, heading))

    def known_walls(self):
        """Every known wall once, as (cell, heading) -- for drawing."""
        seen = set()
        for (cell, heading), present in self._walls.items():
            if not present:
                continue
            other = self.neighbour(cell, heading)
            key = frozenset(((cell, heading), (other, OPPOSITE[heading])))
            if key not in seen:
                seen.add(key)
                yield cell, heading

    # -- queries -------------------------------------------------------------
    def passable(self, cell, heading, unknown_is_open):
        other = self.neighbour(cell, heading)
        if not self.in_bounds(other) or other in self.blocked:
            return False
        wall = self.wall(cell, heading)
        if wall is None:
            return unknown_is_open
        return not wall

    def flood(self, goals, unknown_is_open):
        """Flood fill: BFS step counts outward from the goal cells.

        unknown_is_open=True is the optimistic micromouse flood -- walls we haven't
        seen are assumed absent, which is what lets it plan into unexplored space.
        False plans only through edges we've actually seen open.
        """
        dist, queue = {}, deque()
        for g in goals:
            if self.in_bounds(g) and g not in self.blocked:
                dist[g] = 0
                queue.append(g)
        while queue:
            cell = queue.popleft()
            for h in HEADINGS:
                if self.passable(cell, h, unknown_is_open):
                    other = self.neighbour(cell, h)
                    if other not in dist:
                        dist[other] = dist[cell] + 1
                        queue.append(other)
        return dist

    def next_heading(self, cell, goals, unknown_is_open, facing, prefer_right=True):
        """First step of a shortest route from cell into goals, or None if unreachable.

        Ties are broken to save turns: straight on first, then the preferred side,
        then the other side, then back. That's where "turn towards the side with no
        wall" comes from -- a wall rules its side out, and among the open sides the
        one leading to unexplored ground wins.
        """
        dist = self.flood(goals, unknown_is_open)
        if cell not in dist or dist[cell] == 0:
            return None
        first, second = (RIGHT_OF, LEFT_OF) if prefer_right else (LEFT_OF, RIGHT_OF)
        order = [facing, first[facing], second[facing], OPPOSITE[facing]]
        best = None
        for h in order:
            if not self.passable(cell, h, unknown_is_open):
                continue
            d = dist.get(self.neighbour(cell, h))
            if d is not None and d < dist[cell] and (best is None or d < best[0]):
                best = (d, h)
        return None if best is None else best[1]

    def path(self, cell, goals, unknown_is_open, facing, prefer_right=True):
        """The whole route next_heading would follow, start and end included."""
        route, seen = [cell], {cell}
        while True:
            h = self.next_heading(route[-1], goals, unknown_is_open, facing, prefer_right)
            if h is None:
                break
            nxt = self.neighbour(route[-1], h)
            if nxt in seen:
                break
            route.append(nxt)
            seen.add(nxt)
            facing = h
        return route if route[-1] in set(goals) else None

    def frontier_cells(self):
        """Unvisited cells reachable from a visited one through a non-wall edge --
        the boundary between explored and unexplored."""
        out = set()
        for cell in self.visited:
            for h in HEADINGS:
                other = self.neighbour(cell, h)
                if (self.in_bounds(other) and other not in self.visited
                        and other not in self.blocked and self.wall(cell, h) is not True):
                    out.add(other)
        return out


class Odometry:
    """Dead-reckoned pose from the two track encoders.

    Heading comes from the DIFFERENCE between the tracks, so holding heading is the
    same thing as keeping one track from falling behind the other. On a tracked
    chassis the tracks skid sideways to rotate, so the effective track width is
    wider than the sprocket spacing -- EFFECTIVE_TRACK_M is calibrated, not measured
    with a ruler.
    """

    def __init__(self, ticks_per_m, track_m, x=0.0, y=0.0, theta=0.0):
        self.ticks_per_m, self.track_m = ticks_per_m, track_m
        self.x, self.y, self.theta = x, y, theta
        self.left_m = self.right_m = 0.0      # total track travel, for the HUD
        self.turned = 0.0                     # total signed rotation it believes in
        self._prev = None

    def reset(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta

    def update(self, ticks):
        """Integrate one encoder reading (cumulative left, right). None is ignored."""
        if ticks is None:
            return
        if self._prev is None:
            self._prev = ticks
            return
        dl = (ticks[0] - self._prev[0]) / self.ticks_per_m
        dr = (ticks[1] - self._prev[1]) / self.ticks_per_m
        self._prev = ticks
        self.left_m += dl
        self.right_m += dr
        ds = (dl + dr) / 2.0
        dth = (dr - dl) / self.track_m
        mid = self.theta + dth / 2.0          # midpoint heading: exact for arcs
        self.x += ds * math.cos(mid)
        self.y += ds * math.sin(mid)
        self.theta = wrap(self.theta + dth)
        self.turned += dth


# ---------------------------------------------------------------------------
# Localisation: odometry corrected by the sonars against the map
# ---------------------------------------------------------------------------
def raycast(maze, x, y, angle, max_range=0.7, max_incidence_deg=25.0):
    """Distance from (x, y) along `angle` to the first wall that isn't known open.

    Returns (distance, kind) -- kind "known" for a wall on the map, "maybe" for an
    edge not seen yet (there may or may not be a wall) -- or None if the ray runs
    past max_range, or meets the wall too obliquely for a sonar to get an echo back.
    Walks the grid one cell boundary at a time (Amanatides & Woo).
    """
    c = maze.cell_m
    dx, dy = math.cos(angle), math.sin(angle)
    col = int(math.floor(x / c))
    yrow = int(math.floor(y / c))                  # rows counted up from the south here
    if not (0 <= col < maze.cols and 0 <= yrow < maze.rows):
        return None
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    t_x = (((col + (step_x > 0)) * c - x) / dx) if abs(dx) > 1e-12 else math.inf
    t_y = (((yrow + (step_y > 0)) * c - y) / dy) if abs(dy) > 1e-12 else math.inf
    dt_x = c / abs(dx) if abs(dx) > 1e-12 else math.inf
    dt_y = c / abs(dy) if abs(dy) > 1e-12 else math.inf
    cos_limit = math.cos(math.radians(max_incidence_deg))
    while True:
        if t_x < t_y:
            t, heading = t_x, (EAST if step_x > 0 else WEST)
            square_on = abs(dx)
        else:
            t, heading = t_y, (NORTH if step_y > 0 else SOUTH)
            square_on = abs(dy)
        if t > max_range:
            return None
        cell = (col, maze.rows - 1 - yrow)
        wall = maze.wall(cell, heading)
        if wall is not False:
            if square_on < cos_limit:
                return None                           # glancing: no echo to trust
            return t, ("known" if wall else "maybe")
        if heading in (EAST, WEST):
            col += step_x
            t_x += dt_x
        else:
            yrow += step_y
            t_y += dt_y


class Localiser(Odometry):
    """Odometry, corrected by every sonar reading of a wall -- an extended Kalman
    filter over (x, y, heading, turn scale).

    PREDICT from the encoders. Straight-line travel is trusted; rotation much less,
    because a tracked chassis skids to turn. The fourth state, the turn scale k, is
    how much the chassis REALLY rotates per unit the encoders say -- so a robot whose
    turns come up short learns that while it drives, instead of relying only on the
    calibration.

    CORRECT with each sonar reading: ray-cast from where the filter thinks the sensor
    is, along where it thinks it's pointing, to the first wall on the map. The gap
    between that prediction and the reading moves x, y AND heading at once, in
    proportion to how uncertain each is -- a side wall pins the robot sideways and,
    over a few readings, its heading; a wall ahead pins it along the corridor.
    Readings that don't fit any wall -- a victim, a wall end, an edge not seen yet
    that turns out to be open -- are rejected by the gate rather than believed.
    """

    def __init__(self, ticks_per_m, track_m, mounts, x=0.0, y=0.0, theta=0.0,
                 sonar_sigma=0.008, gate_known=0.05, gate_maybe=0.025):
        super().__init__(ticks_per_m, track_m, x, y, theta)
        self.mounts = mounts          # name -> (forward m, left m, direction rad)
        self.k = 1.0
        # Distance scale, learned by the nav from walls it can see (see Nav._front_creep):
        # how much of the encoder-counted track travel really happened. 1.0 = TICKS_PER_M is
        # right; < 1 = the tracks slip or TICKS_PER_M is too small, so counted distance
        # over-states the real one.
        self.dscale = 1.0
        self.path_m = 0.0             # distance driven by the odometry's own reckoning
        # Placed by hand: good to a couple of cm and a few degrees, no better.
        self.P = [[0.02 ** 2, 0, 0, 0], [0, 0.02 ** 2, 0, 0],
                  [0, 0, math.radians(4) ** 2, 0], [0, 0, 0, 0.003]]
        self.R = sonar_sigma ** 2
        self.gate_known, self.gate_maybe = gate_known, gate_maybe
        self.accepted = self.rejected = 0

    # -- predict ---------------------------------------------------------------
    def update(self, ticks):
        if ticks is None:
            return
        if self._prev is None:
            self._prev = ticks
            return
        dl = (ticks[0] - self._prev[0]) / self.ticks_per_m
        dr = (ticks[1] - self._prev[1]) / self.ticks_per_m
        self._prev = ticks
        self.left_m += dl
        self.right_m += dr
        ds = (dl + dr) / 2.0 * self.dscale
        self.path_m += abs(ds)
        dth_enc = (dr - dl) / self.track_m
        dth = self.k * dth_enc
        mid = self.theta + dth / 2.0
        c, s = math.cos(mid), math.sin(mid)
        self.x += ds * c
        self.y += ds * s
        self.theta = wrap(self.theta + dth)
        self.turned += dth

        # F = d(new state)/d(old state)
        F = [[1, 0, -ds * s, -ds * s * dth_enc / 2],
             [0, 1, ds * c, ds * c * dth_enc / 2],
             [0, 0, 1, dth_enc],
             [0, 0, 0, 1]]
        q_along = (0.02 * abs(ds)) ** 2 + 1e-8
        q_theta = (0.004 * abs(ds) + 0.03 * abs(dth_enc)) ** 2 + 1e-9
        Q = [[q_along * c * c, q_along * c * s, 0, 0],
             [q_along * c * s, q_along * s * s, 0, 0],
             [0, 0, q_theta, 0],
             [0, 0, 0, (1e-4 * abs(dth_enc)) ** 2]]
        self.P = _add(_mul(_mul(F, self.P), _t(F)), Q)

    # -- correct ---------------------------------------------------------------
    def predict_range(self, maze, name, x=None, y=None, theta=None):
        fwd, left, rel = self.mounts[name]
        x = self.x if x is None else x
        y = self.y if y is None else y
        theta = self.theta if theta is None else theta
        c, s = math.cos(theta), math.sin(theta)
        sx = x + fwd * c - left * s
        sy = y + fwd * s + left * c
        return raycast(maze, sx, sy, theta + rel)

    def correct(self, maze, name, reading):
        """Fold one sonar reading in. True if it was used, False if rejected."""
        if reading is None or name not in self.mounts:
            return False
        hit = self.predict_range(maze, name)
        if hit is None:
            return False
        h, kind = hit
        # Jacobian of the predicted range, numerically -- the ray model is piecewise.
        H = [0.0, 0.0, 0.0, 0.0]
        for i, eps in ((0, 1e-4), (1, 1e-4), (2, 1e-4)):
            args = [self.x, self.y, self.theta]
            args[i] += eps
            other = self.predict_range(maze, name, *args)
            if other is None or other[1] != kind:
                return False                           # right on an edge: skip it
            H[i] = (other[0] - h) / eps
        innovation = reading - h
        PH = [sum(self.P[r][j] * H[j] for j in range(4)) for r in range(4)]
        S = sum(H[r] * PH[r] for r in range(4)) + self.R
        # The gate throws out the big misses -- a victim, a wall end -- which are
        # several cm off. It has a fixed floor, so an overconfident filter can't
        # reject exactly the readings that show it's wrong; and for a wall on the map
        # it widens with the filter's own uncertainty, so after a long stretch with
        # nothing to see, the first wall it can see is allowed to pull it back.
        if kind == "known":
            gate = min(max(self.gate_known, 2.5 * math.sqrt(S)), 0.10)
        else:
            gate = self.gate_maybe                     # might not even be a wall
        if abs(innovation) > gate:
            self.rejected += 1
            return False
        K = [PH[r] / S for r in range(4)]
        self.x += K[0] * innovation
        self.y += K[1] * innovation
        self.theta = wrap(self.theta + K[2] * innovation)
        self.k = min(1.3, max(0.75, self.k + K[3] * innovation))
        # Joseph form keeps P symmetric and positive under rounding.
        IKH = [[(1.0 if r == c else 0.0) - K[r] * H[c] for c in range(4)] for r in range(4)]
        self.P = _add(_mul(_mul(IKH, self.P), _t(IKH)),
                      [[K[r] * K[c] * self.R for c in range(4)] for r in range(4)])
        self.accepted += 1
        return True


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]


def _t(a):
    return [list(r) for r in zip(*a)]


def _add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
