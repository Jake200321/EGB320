"""Maze nav checks: the map and planners, then whole missions in a simulated maze.

    python3 test_maze.py            # all checks
    python3 test_maze.py --png      # also save the final maps to runs/test_*.png

The simulation is kinematic, not physics: a robot pose driven by Nav's own (v, w)
commands, three sonars ray-cast against the walls of a Milestone 2 maze, and a
camera that sees a victim when it's in the field of view with nothing in between.
It deliberately does NOT match nav's assumptions exactly -- the real turn rate is
off from what the odometry believes, one track is weaker, and the sonars are noisy
-- so the checks show the centring and wall corrections actually doing their job.

What it does NOT model: sonar specular misses off angled walls, wall thickness,
bumps and ramps, wheel slip on the straight, or the camera's real detection range.
Passing here means the logic is right; it doesn't replace running it in the maze.

Nav's real code runs unmodified: Nav, Mover, Ultrasonics, Odometry, MazeMap.
Only time, the encoders, the sonar pings and the camera are faked.
"""

import math
import os
import random
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as M                                              # noqa: E402
from maze import (EAST, NORTH, SOUTH, WEST, MazeMap, Odometry,  # noqa: E402
                  nearest_heading, wrap)

fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)


# --------------------------------------------------------------------- mazes
# Wall segments between grid posts (post (i, j): x = i cells east, j cells down
# from the north edge) -- copied from EGB320_Milestone2_Mazes.py in the sim repo.
MAZE_1_WALLS = (
    ((1, 0), (1, 1)), ((2, 1), (3, 1)), ((3, 1), (4, 1)), ((6, 0), (6, 1)),
    ((5, 1), (6, 1)), ((2, 1), (2, 2)), ((2, 2), (3, 2)), ((3, 2), (4, 2)),
    ((1, 2), (1, 3)), ((0, 3), (1, 3)), ((2, 2), (2, 3)), ((1, 3), (2, 3)),
    ((3, 2), (3, 3)), ((3, 3), (4, 3)), ((5, 2), (5, 3)), ((6, 2), (6, 3)),
    ((5, 3), (6, 3)), ((3, 3), (3, 4)), ((2, 4), (3, 4)), ((3, 4), (4, 4)),
    ((5, 3), (5, 4)), ((6, 4), (7, 4)), ((1, 4), (1, 5)), ((2, 5), (3, 5)),
    ((3, 5), (4, 5)), ((5, 4), (5, 5)), ((6, 5), (7, 5)), ((1, 5), (1, 6)),
    ((2, 5), (2, 6)), ((1, 6), (2, 6)), ((3, 6), (4, 6)), ((5, 5), (5, 6)),
    ((6, 5), (6, 6)), ((5, 6), (6, 6)), ((1, 6), (1, 7)), ((3, 6), (3, 7)),
)
MAZE_2_WALLS = (
    ((0, 1), (1, 1)), ((1, 1), (2, 1)), ((3, 0), (3, 1)), ((4, 1), (5, 1)),
    ((6, 1), (7, 1)), ((0, 2), (1, 2)), ((2, 1), (2, 2)), ((3, 2), (4, 2)),
    ((5, 1), (5, 2)), ((4, 2), (5, 2)), ((6, 2), (7, 2)), ((1, 3), (2, 3)),
    ((3, 2), (3, 3)), ((4, 2), (4, 3)), ((1, 3), (1, 4)), ((0, 4), (1, 4)),
    ((1, 4), (2, 4)), ((3, 3), (3, 4)), ((2, 4), (3, 4)), ((5, 3), (5, 4)),
    ((4, 4), (5, 4)), ((6, 3), (6, 4)), ((5, 4), (6, 4)), ((1, 5), (2, 5)),
    ((2, 5), (3, 5)), ((5, 4), (5, 5)), ((5, 5), (6, 5)), ((1, 5), (1, 6)),
    ((1, 6), (2, 6)), ((2, 6), (3, 6)), ((4, 5), (4, 6)), ((5, 5), (5, 6)),
    ((4, 6), (5, 6)), ((5, 6), (6, 6)), ((1, 6), (1, 7)), ((6, 6), (6, 7)),
)
COLS = ROWS = 7


def true_map(walls, cell_m):
    """The maze as a fully-known MazeMap -- the answer key for shortest paths."""
    m = MazeMap(COLS, ROWS, cell_m)
    for (a, b) in walls:
        (i0, j0), (i1, j1) = sorted((a, b))
        if i0 == i1:                                      # vertical: west/east
            for j in range(j0, j1):
                if i0 > 0:
                    m.set_wall((i0 - 1, j), EAST, True)
                if i0 < COLS:
                    m.set_wall((i0, j), WEST, True)
        else:                                             # horizontal: north/south
            for i in range(i0, i1):
                if j0 > 0:
                    m.set_wall((i, j0 - 1), SOUTH, True)
                if j0 < ROWS:
                    m.set_wall((i, j0), NORTH, True)
    for c in range(COLS):
        for r in range(ROWS):
            for h in (NORTH, EAST, SOUTH, WEST):
                if m.wall((c, r), h) is None:
                    m.set_wall((c, r), h, False)
    return m


# --------------------------------------------------------------------- world
class Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


class World:
    """The true robot and maze. Advances the pose from Nav's (v, w) command."""

    def __init__(self, walls, victims, cell_m, base, start_heading,
                 turn_slip=1.0, left_weak=1.0, noise=0.004, seed=1, x_offset=0.0,
                 turn_jitter=0.0, locked_tracks=True, tick_scale=1.0):
        self.cell_m = cell_m
        self.segs = []
        size = COLS * cell_m
        for (i0, j0), (i1, j1) in list(walls) + [((0, 0), (7, 0)), ((7, 0), (7, 7)),
                                                  ((0, 7), (7, 7)), ((0, 0), (0, 7))]:
            self.segs.append(((i0 * cell_m, size - j0 * cell_m),
                              (i1 * cell_m, size - j1 * cell_m)))
        self.victims = [self.cell_centre(c) for c in victims]
        self.x, self.y = self.cell_centre(base)
        self.x += x_offset
        self.theta = {"N": math.pi / 2, "E": 0.0, "S": -math.pi / 2, "W": math.pi}[start_heading]
        self.turn_slip = turn_slip       # real yaw / yaw the tracks' speeds imply
        self.turn_jitter = turn_jitter   # ...varying turn to turn by this fraction
        self._slip_now, self._pivoting = turn_slip, False
        self.left_weak = left_weak       # left track delivers this fraction
        # The real chassis can't steer by a speed difference while driving: measured, both
        # tracks run at the same speed whatever the commands (straight_test.py step 5).
        # Only a pivot (tracks counter-rotating) turns it.
        self.locked_tracks = locked_tracks
        # The encoders count this many times the track travel that really happens (wheel
        # slip, or TICKS_PER_M set too low): the odometry then thinks it has gone further
        # than it has and stops short.
        self.tick_scale = tick_scale
        self.ticks = [0.0, 0.0]
        self.rng = random.Random(seed)
        self.noise = noise
        self.carried = set()
        self.min_wall_gap = 1.0
        self.min_victim_gap = 1.0
        self.path = []

    def cell_centre(self, cell):
        return ((cell[0] + 0.5) * self.cell_m, (ROWS - cell[1] - 0.5) * self.cell_m)

    def cell(self):
        return (int(self.x // self.cell_m), ROWS - 1 - int(self.y // self.cell_m))

    def advance(self, v, w, dt):
        # What motors.py would ask each track for...
        half = M_TRACK / 2.0
        vl, vr = v - w * half, v + w * half
        vl *= self.left_weak                             # ...and what they deliver
        if self.locked_tracks and abs(v) > 1e-9:
            vl = vr = (vl + vr) / 2.0                    # driving: the chassis equalises them
        # Encoders count track travel, whatever the chassis actually does.
        self.ticks[0] += vl * dt * M.TICKS_PER_M * self.tick_scale
        self.ticks[1] += vr * dt * M.TICKS_PER_M * self.tick_scale
        pivoting = abs(v) < 1e-9 and abs(w) > 0.5
        if pivoting and not self._pivoting:           # a new turn: new grip
            self._slip_now = self.turn_slip * (1 + self.rng.gauss(0, self.turn_jitter))
        self._pivoting = pivoting
        slip = self._slip_now if pivoting else self.turn_slip
        ds = (vl + vr) / 2.0 * dt
        dth = (vr - vl) / M.EFFECTIVE_TRACK_M * slip * dt
        self.x += ds * math.cos(self.theta + dth / 2)
        self.y += ds * math.sin(self.theta + dth / 2)
        self.theta = wrap(self.theta + dth)
        self.min_wall_gap = min(self.min_wall_gap, self.nearest_wall())
        for i, (vx, vy) in enumerate(self.victims):
            if i not in self.carried:
                self.min_victim_gap = min(self.min_victim_gap,
                                          math.hypot(vx - self.x, vy - self.y))
        self.path.append((self.x, self.y))

    def nearest_wall(self):
        best = 1.0
        for (ax, ay), (bx, by) in self.segs:
            dx, dy = bx - ax, by - ay
            t = max(0.0, min(1.0, ((self.x - ax) * dx + (self.y - ay) * dy) / (dx * dx + dy * dy)))
            best = min(best, math.hypot(self.x - ax - t * dx, self.y - ay - t * dy))
        return best

    def ray(self, ox, oy, angle, max_d=2.0, hit_victims=False):
        ux, uy = math.cos(angle), math.sin(angle)
        best = None
        for (ax, ay), (bx, by) in self.segs:
            ex, ey = bx - ax, by - ay
            den = ux * ey - uy * ex
            if abs(den) < 1e-12:
                continue
            t = ((ax - ox) * ey - (ay - oy) * ex) / den
            s = ((ax - ox) * uy - (ay - oy) * ux) / den
            if t > 0 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t < best):
                best = t
        if hit_victims:
            # A small object scatters sound back from anywhere in the sonar's cone
            # (~15 deg either side on an HC-SR04), where a flat wall only answers
            # square on -- so victims are seen across the cone, walls along the axis.
            for i, (vx, vy) in enumerate(self.victims):
                if i in self.carried:
                    continue
                fx, fy = vx - ox, vy - oy
                d = math.hypot(fx, fy) - 0.03
                off = abs(wrap(math.atan2(fy, fx) - angle))
                if off < math.radians(15) and (best is None or d < best):
                    best = d
        return best if best is not None and best <= max_d else None

    def sonar(self, name):
        """Reading of one sonar, mounted so a wall reads *_AT_CENTRE_M from a cell centre."""
        if name == "front":
            off, rel = self.cell_m / 2 - M.FRONT_WALL_AT_CENTRE_M, 0.0
        else:
            off = self.cell_m / 2 - M.SIDE_WALL_AT_CENTRE_M
            rel = math.pi / 2 if name == "left" else -math.pi / 2
        a = self.theta + rel
        ox, oy = self.x + off * math.cos(a), self.y + off * math.sin(a)
        d = self.ray(ox, oy, a, hit_victims=(name == "front"))
        return None if d is None else max(0.02, d + self.rng.uniform(-self.noise, self.noise))

    def camera(self):
        """The nearest victim in view: Victim(bearing +RIGHT, distance), or None."""
        best = None
        cx = self.x + CAMERA_FORWARD_M * math.cos(self.theta)
        cy = self.y + CAMERA_FORWARD_M * math.sin(self.theta)
        for i, (vx, vy) in enumerate(self.victims):
            if i in self.carried:
                continue
            dx, dy = vx - cx, vy - cy
            d = math.hypot(dx, dy)
            bearing = -math.degrees(wrap(math.atan2(dy, dx) - self.theta))
            # The victim's base has to be in frame -- the same model main.py uses.
            blind = M.camera_blind_range_m() or 0.27
            if not (blind < d < 1.2 and abs(bearing) < M.HORIZONTAL_FOV_DEG / 2):
                continue
            wall = self.ray(cx, cy, math.atan2(dy, dx))
            if wall is not None and wall < d:
                continue                                  # behind a wall
            if best is None or d < best[1]:
                best = (bearing, d)
        if best is None:
            return None
        return M.Victim(best[0], best[1], (300, 250, 40, 40), 1600)

    def collect(self):
        """Victim within reach of the robot's nose is picked up."""
        for i, (vx, vy) in enumerate(self.victims):
            if i not in self.carried and math.hypot(vx - self.x, vy - self.y) < 0.30:
                self.carried.add(i)
                return i
        return None


M_TRACK = 0.123     # motors.TRACK_M -- the geometric gauge motors.py uses
CAMERA_FORWARD_M = 0.06   # camera sits ahead of the turning centre


# --------------------------------------------------------------------- fakes
class SimDrive:
    def __init__(self, world):
        self.world = world
        self.last = (0.0, 0.0)
        self.board = True

    def set_velocity(self, v, w):
        self.last = (v, w)

    def stop(self):
        self.last = (0.0, 0.0)

    def read_encoders(self):
        return (int(self.world.ticks[0]), int(self.world.ticks[1]))


class SimPing:
    def __init__(self, world, name):
        self.world, self.name = world, name

    def ping(self):
        return self.world.sonar(self.name)


class SimVision:
    geometry_ok = True
    last_frame = None
    last_detections = []
    draw_detections = None

    def __init__(self, world):
        self.world = world

    def look(self):
        return self.world.camera()


class Leds:
    def __init__(self):
        self.g = self.y = self.r = False
        self.log = []

    def _s(self, k, on):
        if getattr(self, k) != on:
            setattr(self, k, on)
            self.log.append((k, on))

    def green(self, on): self._s("g", on)
    def yellow(self, on): self._s("y", on)
    def red(self, on): self._s("r", on)
    def all_off(self): self.g = self.y = self.r = False


def run_mission(walls, victims, base=(0, 6), heading="N", seconds=M.MISSION_TIME_S,
                camera=True, stop_when=None, **world_kw):
    clock = Clock()
    M.time = types.SimpleNamespace(monotonic=clock.monotonic, sleep=lambda s: None,
                                   strftime=lambda *a: "sim")
    M.STATUS.enabled = False
    M.STATUS.event = lambda msg: events.append(msg)
    events = []
    world = World(walls, victims, M.CELL_M, base, heading, **world_kw)
    sonar = M.Ultrasonics(enabled=False)
    sonar.sensors = {n: SimPing(world, n) for n in ("front", "left", "right")}
    leds = Leds()
    vision = SimVision(world) if camera else M.NullVision()
    nav = M.Nav(SimDrive(world), vision, leds, sonar, base_cell=base, start_heading=heading)
    # A pulse of motion inside one control tick (the heading trim sleeps through them):
    # move the robot and the clock for that long under whatever is commanded now.
    M.time.sleep = lambda s: (world.advance(*nav.drive.last, s), setattr(clock, "t", clock.t + s))

    # Stand-in collection: the "mechanism" picks up whatever's at the robot's nose.
    real_at_victim = nav._at_victim

    def at_victim():
        if nav.rescue.done() and not nav.carrying:
            world.collect()
        real_at_victim()
    nav._at_victim = at_victim

    dt = 1.0 / M.CONTROL_HZ
    states = []
    home_legs = []          # (cells walked home, shortest possible) per return trip
    leg_start = None
    while clock.t - 1000.0 < seconds and nav.state != M.DONE:
        nav.step()
        if not states or states[-1] != nav.state:
            states.append(nav.state)
            if nav.state == M.RETURN and nav.carrying:
                leg_start = (clock.t, [world.cell()])
            if nav.state == M.AT_BASE and leg_start is not None:
                home_legs.append(leg_start[1])
                leg_start = None
        if leg_start is not None and world.cell() != leg_start[1][-1]:
            leg_start[1].append(world.cell())
        world.advance(*nav.drive.last, dt)
        clock.t += dt
        if stop_when and stop_when(nav):
            break
    return nav, world, leds, states, events, home_legs, clock.t - 1000.0


def save_png(nav, name):
    try:
        import cv2
        from map_view import render_map
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, f"test_{name}.png"), render_map(nav))
    except ImportError:
        pass


# ===================================================================== tests
print("\n1) map bookkeeping")
m = MazeMap(7, 7, 0.28)
check("perimeter seeded", m.wall((0, 0), NORTH) is True and m.wall((6, 3), EAST) is True)
check("interior unknown to start", m.wall((3, 3), NORTH) is None)
m.set_wall((3, 3), NORTH, True)
check("setting a wall sets the neighbour's side too", m.wall((3, 2), SOUTH) is True)
m.set_wall((3, 3), EAST, True, lock=True)
m.set_wall((3, 3), EAST, False)
check("a wall proven by contact can't be reopened by the sonar", m.wall((3, 3), EAST) is True)
check("centre / cell_at round trip", m.cell_at(*m.centre((2, 5))) == (2, 5))
check("row 0 is north", m.centre((0, 0))[1] > m.centre((0, 6))[1])

print("\n2) flood fill and the choice at a junction")
m = MazeMap(7, 7, 0.28)
check("with nothing known, the way north is open (optimistic)",
      m.next_heading((0, 6), [(0, 0)], True, NORTH) == NORTH)
check("...but not through unknown walls when planning on known edges only",
      m.next_heading((0, 6), [(0, 0)], False, NORTH) is None)
# A T-junction: wall ahead, both sides open, goal equally far either way.
m.set_wall((3, 3), NORTH, True)
m.set_wall((3, 3), EAST, False)
m.set_wall((3, 3), WEST, False)
h = m.next_heading((3, 3), [(4, 3), (2, 3)], True, NORTH, prefer_right=True)
check("wall ahead -> turns to an open side, the preferred one on a tie", h == EAST)
h = m.next_heading((3, 3), [(4, 3), (2, 3)], True, NORTH, prefer_right=False)
check("...and the other side when preferring left", h == WEST)
m.set_wall((3, 3), EAST, True)
check("the side with a wall is never chosen",
      m.next_heading((3, 3), [(4, 3), (2, 3)], True, NORTH, prefer_right=True) == WEST)
check("straight on beats a turn when both are as short",
      MazeMap(7, 7, 0.28).next_heading((3, 3), [(3, 0), (6, 3)], True, NORTH) == NORTH)
tm = true_map(MAZE_1_WALLS, 0.28)
route = tm.path((0, 6), [(2, 2)], False, NORTH)
check("known-map route to maze 1's L1 is the documented 6 cells (N,N,N,E,E,N)",
      route is not None and len(route) - 1 == 6)

print("\n3) frontier")
m = MazeMap(7, 7, 0.28)
m.visited.add((0, 6))
m.set_wall((0, 6), NORTH, False)
m.set_wall((0, 6), EAST, True)
check("frontier is the unexplored cell through the open side",
      m.frontier_cells() == {(0, 5)})

print("\n4) odometry")
o = Odometry(10000.0, 0.16, 0.0, 0.0, 0.0)
o.update((0, 0))
o.update((10000, 10000))
check("equal ticks -> straight 1 m", abs(o.x - 1.0) < 1e-9 and abs(o.theta) < 1e-9)
o.update((10000 - 1257, 10000 + 1257))
check("opposite ticks -> spins on the spot (~90 deg left)",
      abs(o.theta - math.pi / 2) < 0.01 and abs(o.x - 1.0) < 1e-6)
check("nearest cardinal heading", nearest_heading(math.radians(80)) == NORTH)

# ------------------------------------------------------------ whole missions
def mission_checks(label, walls, victims, **kw):
    nav, world, leds, states, events, legs, t = run_mission(walls, victims, **kw)
    tm = true_map(walls, M.CELL_M)
    check(f"{label}: all {len(victims)} victims collected and brought home "
          f"({nav.rescued} in {t:.0f}s)", nav.rescued == len(victims))
    check(f"{label}: finished inside 7 minutes", t < M.MISSION_TIME_S)
    check(f"{label}: finished at base", world.cell() == nav.base_cell)
    check(f"{label}: never hit a wall (closest {world.min_wall_gap*100:.1f}cm from a wall "
          "centreline)", world.min_wall_gap > 0.07)
    check(f"{label}: never drove over a victim (closest "
          f"{world.min_victim_gap*100:.0f}cm centre to centre)", world.min_victim_gap > 0.15)
    bad = [h for h in nav.map.known_walls() if tm.wall(*h) is not True]
    check(f"{label}: every wall on the map is a real wall ({len(bad)} wrong)", not bad)
    ok_legs = True
    for leg in legs:
        shortest = len(tm.path(leg[0], [nav.base_cell], False, NORTH)) - 1
        walked = len(leg) - 1
        if walked > shortest:
            ok_legs = False
            print(f"       home leg from {leg[0]}: walked {walked}, shortest {shortest}")
    check(f"{label}: every trip home was a shortest route ({len(legs)} trips)",
          ok_legs and len(legs) == len(victims))
    want = [M.SEARCH, M.SEEK, M.CLOSING, M.AT_VICTIM, M.RETURN, M.AT_BASE]
    check(f"{label}: explore -> go to victim -> close in -> collect -> home, per victim",
          states[:6] == want and states.count(M.AT_BASE) == len(victims))
    check(f"{label}: never steered at a victim by eye across open space",
          M.APPROACH not in states)
    reds = [on for k, on in leds.log if k == "r"]
    check(f"{label}: red LED only on the way home, then off", reds[:2] == [True, False])
    if "--png" in sys.argv:
        save_png(nav, label.replace(" ", "_"))
    return nav, world


print("\n5) full mission, maze 1 (ideal robot)")
mission_checks("maze 1", MAZE_1_WALLS, [(2, 2), (2, 1), (5, 5)])

print("\n6) full mission, maze 2")
mission_checks("maze 2", MAZE_2_WALLS, [(4, 5), (5, 4), (0, 1)])

print("\n7) maze 1 with a real robot's faults: turns 12% short, left track 8% weak")
mission_checks("maze 1 slip", MAZE_1_WALLS, [(2, 2), (2, 1), (5, 5)],
               turn_slip=0.88, left_weak=0.92, noise=0.006, seed=7)

print("\n8) no camera: explores and maps the whole maze, then goes home")
nav, world, leds, states, events, legs, t = run_mission(MAZE_1_WALLS, [], camera=False)
tm = true_map(MAZE_1_WALLS, M.CELL_M)
check(f"all 49 cells visited ({len(nav.map.visited)}) in {t:.0f}s",
      len(nav.map.visited) == 49)
wrong = sum(1 for c in nav.map.visited for h in (NORTH, EAST, SOUTH, WEST)
            if nav.map.wall(c, h) is not None and nav.map.wall(c, h) != tm.wall(c, h))
check(f"every wall it mapped matches the real maze ({wrong} wrong)", wrong == 0)
check("back at base and stopped", world.cell() == (0, 6) and nav.state == M.DONE)
check("no red LED without a victim on board", not any(k == "r" and on for k, on in leds.log))
if "--png" in sys.argv:
    save_png(nav, "explore_only")

print("\n9) centring: starts 3 cm off centre, pulls back into the middle")
nav, world, *_ = run_mission(MAZE_1_WALLS, [], camera=False, x_offset=0.03,
                             stop_when=lambda n: n.cell == (0, 4) and not n.mover.busy)
world2 = World(MAZE_1_WALLS, [], M.CELL_M, (0, 6), "N")
check(f"ends the corridor within 2 cm of centre, from 3 cm off "
      f"(off by {abs(world.x - world2.x)*100:.1f} cm)", abs(world.x - world2.x) < 0.02)

print("\n10) learns its own turn slip: calibration 10% out either way still completes")
for slip in (0.90, 1.10):
    nav, world, leds, states, events, legs, t = run_mission(
        MAZE_2_WALLS, [(4, 5), (5, 4), (0, 1)], turn_slip=slip, left_weak=0.92,
        noise=0.006, seed=5)
    tm = true_map(MAZE_2_WALLS, M.CELL_M)
    wrong = sum(1 for h in nav.map.known_walls() if tm.wall(*h) is not True)
    check(f"turns {abs(slip - 1) * 100:.0f}% {'short' if slip < 1 else 'long'}: learned "
          f"k={nav.odo.k:.3f} (true {slip})", abs(nav.odo.k - slip) < 0.02)
    check(f"...collected all 3 ({nav.rescued} released by {t:.0f}s), no contact, map right",
          len(world.carried) == 3 and world.min_wall_gap > 0.07
          and world.min_victim_gap > 0.15 and wrong == 0)

print("\n11) encoders that over-count (wheel slip / TICKS_PER_M too small): stops short, then fixes itself")
tm = true_map(MAZE_1_WALLS, M.CELL_M)
for scale in (1.0, 1.15):
    nav, world, leds, states, events, legs, t = run_mission(MAZE_1_WALLS, [], camera=False,
                                                            tick_scale=scale)
    wrong = sum(1 for c in nav.map.visited for h in (NORTH, EAST, SOUTH, WEST)
                if nav.map.wall(c, h) is not None and nav.map.wall(c, h) != tm.wall(c, h))
    creeps = sum("creeping up" in e for e in events)
    check(f"encoders x{scale:.2f}: maps all 49 cells ({len(nav.map.visited)}), "
          f"{wrong} wrong walls, no contact ({world.min_wall_gap*100:.1f} cm)",
          len(nav.map.visited) == 49 and wrong == 0 and world.min_wall_gap > 0.05)
    if scale == 1.0:
        check(f"a correct robot never creeps or changes its scale ({creeps} creeps, "
              f"scale {nav.odo.dscale:.3f})", creeps == 0 and abs(nav.odo.dscale - 1.0) < 1e-9)
    else:
        check(f"...it learned the scale from the walls ({nav.odo.dscale:.3f}, true "
              f"{1 / scale:.3f})", abs(nav.odo.dscale - 1 / scale) < 0.05 and creeps >= 1)

print("\n12) status LEDs over a whole mission, checked on every tick")
samples = []
def sample(n):
    L = n.leds
    samples.append((n.state, n.carrying, L.g, L.y, L.r))
    return False
nav, world, leds, states, events, legs, t = run_mission(MAZE_1_WALLS, [(2, 2), (2, 1), (5, 5)],
                                                        stop_when=sample)
def colour(g, y, r):
    return "A" if g and y and r else "G" if g and not y and not r else \
           "Y" if y and not g and not r else "R" if r and not g and not y else \
           "-" if not (g or y or r) else "?"
seq = [colour(*s[2:]) for s in samples]
check(f"never a mixed state: one LED, none, or all three together ({seq.count('?')} odd ticks)",
      "?" not in seq)
check("yellow whenever it is searching",
      all(colour(*s[2:]) == "Y" for s in samples if s[0] == M.SEARCH))
check("green while it goes to / closes on a victim",
      all(colour(*s[2:]) == "G" for s in samples if s[0] in (M.SEEK, M.APPROACH, M.CLOSING)))
# (the first tick at a victim is still the solid green it arrived with; the flashing starts
# on the next, so each visit's first sample is left out)
in_victim = [colour(*s[2:]) for i, s in enumerate(samples)
             if s[0] == M.AT_VICTIM and i > 0 and samples[i - 1][0] == M.AT_VICTIM]
check("all three flash together while collecting (at the victim), and nothing else does",
      all(k in ("A", "-") for k in in_victim)
      and all(colour(*s[2:]) != "A" for s in samples if s[0] != M.AT_VICTIM))
at_v = [colour(*s[2:]) for s in samples if s[0] == M.AT_VICTIM]
check(f"...really flashing: {at_v.count('A')} lit ticks, {at_v.count('-')} dark ticks",
      at_v.count("A") > 20 and at_v.count("-") > 20)
check("red whenever it is taking a victim home, and ONLY then",
      all((colour(*s[2:]) == "R") == (s[0] in (M.RETURN, M.AT_BASE) and s[1]) for s in samples))
check("yellow when heading home with nothing aboard",
      all(colour(*s[2:]) == "Y" for s in samples if s[0] == M.RETURN and not s[1]))
folded = ["F" if k in ("A", "-") and s[0] == M.AT_VICTIM else k for k, s in zip(seq, samples)]
runs = [k for i, k in enumerate(folded) if i == 0 or k != folded[i - 1]]
check(f"each rescue shows yellow -> green -> flashing -> red -> yellow ({''.join(runs)[:20]}...)",
      "".join(runs).startswith("YGFRYGFRY"))
check("it's yellow from the very first tick", seq[0] == "Y")

print("\n13) --test-mode: ONE victim, flash all LEDs, home, done")
_mock0, _tm0 = M.MOCK_COLLECTION, M.TEST_MODE
M.MOCK_COLLECTION, M.TEST_MODE = False, True
try:
    nav, world, leds, states, events, legs, t = run_mission(MAZE_1_WALLS, [(2, 2), (2, 1), (5, 5)])
finally:
    M.MOCK_COLLECTION, M.TEST_MODE = _mock0, _tm0
check(f"collects exactly one victim ({len(world.carried)}) and takes it home ({nav.rescued} rescued)",
      len(world.carried) == 1 and nav.rescued == 1)
check(f"then it's finished, back at the base, well inside the time ({t:.0f}s)",
      nav.state == M.DONE and world.cell() == (0, 6) and t < M.MISSION_TIME_S - 30)
check("it said why it finished", any("FINISHED" in e and "TEST MODE" in e and "1 victim rescued" in e
                                     for e in events))
check("announced the mock collection once", sum("TEST MODE" in e and "mock collection" in e
                                               for e in events) == 1)
check("LEDs: yellow -> green -> all flashing -> red home -> off at the end",
      any(k == "r" and on for k, on in leds.log) and not (leds.g or leds.y or leds.r))

print(f"\n{'ALL PASSED' if not fails else f'{len(fails)} FAILED'}")
sys.exit(1 if fails else 0)
