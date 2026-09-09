"""Nav logic checks that need no camera, no motors, no sonar and no Pi.

    python3 test_nav.py

Fakes all four subsystems and drives Nav.step() through scripted detections, so the
parts that fail quietly get checked on a laptop: detection debounce, the bearing sign
flip between vision and drive, the 10 cm stop, when the sonar may and may not be
believed, and the refusals -- no ranging, no echo, wall-marker geometry.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as M


# --------------------------------------------------------------------- fakes
class FakeDrive:
    def __init__(self): self.last = (0.0, 0.0); self.board = None
    def set_velocity(self, v, w): self.last = (v, w)
    def stop(self): self.last = (0.0, 0.0)


class FakeLeds:
    def __init__(self): self.g = None; self.y = None; self.r = None; self.ylog = []
    def green(self, on): self.g = on
    def yellow(self, on): self.y = on; self.ylog.append(on)
    def red(self, on): self.r = on
    def all_off(self): self.g = self.y = self.r = False


class FakeVision:
    """Replays a scripted list of Victim|None, one per look()."""
    def __init__(self, script, geometry_ok=True):
        self.script = list(script); self.geometry_ok = geometry_ok
    def look(self):
        return self.script.pop(0) if self.script else None


class FakeSonar:
    """Front sonar only; .front is writable mid-test to simulate closing in."""
    def __init__(self, front=None, available=True):
        self.front = front; self.available = available
        self.sensors = {"front": 1} if available else {}
    def update(self): pass
    def get(self, name): return self.front if name == "front" else None
    def walls(self): return (None, None)


NO_SONAR = FakeSonar(available=False)


def V(dist=None, bearing=0.0):
    """A victim detection. dist is the CAMERA estimate; None = camera can't range it."""
    return M.Victim(bearing, dist, (100, 100, 50, 50), 2500)


fails = []
def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        fails.append(name)

def run(nav, ticks):
    for _ in range(ticks):
        nav.step()


# --------------------------------------------------------------------- tests
print("1) SEARCH spins on the spot with green off")
d, l = FakeDrive(), FakeLeds()
run(M.Nav(d, FakeVision([None] * 3), l, FakeSonar(front=1.0)), 3)
check("spinning, no forward motion", d.last == (0.0, M.SEARCH_TURN_RATE))
check("green off", l.g is False)

print("2) one frame doesn't commit; DETECTION_DEBOUNCE frames do")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V(), None, V(), V(), V()]), l, FakeSonar(front=0.5))
n.step()
check("still SEARCH after a single hit", n.state == M.SEARCH)
run(n, 4)
check("APPROACH once debounced", n.state == M.APPROACH)
check("green ON", l.g is True)

print("3) approach steers the right way (+bearing = right = negative yaw)")
# Bearing kept inside US_TRUST_BEARING_DEG so the sonar is believed and it closes;
# test 9 covers the off-axis case, where refusing to close is the correct answer.
d = FakeDrive()
run(M.Nav(d, FakeVision([V(bearing=8.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("driving forward", d.last[0] > 0)
check("yaw negative for a victim to the right", d.last[1] < 0)
d = FakeDrive()
run(M.Nav(d, FakeVision([V(bearing=-8.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("yaw positive for a victim to the left", d.last[1] > 0)

print("4) creeps when close")
d1 = FakeDrive()
run(M.Nav(d1, FakeVision([V()] * 4), FakeLeds(), FakeSonar(front=0.50)), 4)
d2 = FakeDrive()
run(M.Nav(d2, FakeVision([V()] * 4), FakeLeds(), FakeSonar(front=0.20)), 4)
check("slower inside CREEP_RANGE_M", d2.last[0] < d1.last[0])

print("5) stops 10 cm short and flashes yellow")
d, l, son = FakeDrive(), FakeLeds(), FakeSonar(front=0.50)
n = M.Nav(d, FakeVision([V()] * 50), l, son)
run(n, 4)
check("approaching", n.state == M.APPROACH)
son.front = 0.11
n.step()
check("AT_VICTIM at 11 cm", n.state == M.AT_VICTIM)
check("motors stopped", d.last == (0.0, 0.0))
for _ in range(40):
    n.step(); time.sleep(0.012)
check("yellow actually toggled", True in l.ylog and False in l.ylog)
check("green stays on", l.g is True)

print("6) losing the victim returns to SEARCH")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V()] * 3 + [None] * 12), l, FakeSonar(front=0.5))
run(n, 15)
check("back to SEARCH", n.state == M.SEARCH)
check("green off again", l.g is False)

print("7) no ranging at all: green on, but it will NOT drive at the victim")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V()] * 5, geometry_ok=False), l, NO_SONAR)
run(n, 5)
check("stays in SEARCH", n.state == M.SEARCH)
check("green still ON", l.g is True)
check("not driving", d.last == (0.0, 0.0))

print("8) the front sonar alone is enough -- no camera geometry needed")
d = FakeDrive()
n = M.Nav(d, FakeVision([V()] * 6, geometry_ok=False), FakeLeds(), FakeSonar(front=0.50))
run(n, 6)
check("reaches APPROACH on sonar alone", n.state == M.APPROACH)
check("driving forward", d.last[0] > 0)
check("range came from the sonar", n.victim_range()[1] == "sonar")

print("9) sonar is not believed when the victim is off to one side")
d = FakeDrive()
n = M.Nav(d, FakeVision([V(bearing=30.0)] * 6, geometry_ok=False), FakeLeds(),
          FakeSonar(front=0.50))
run(n, 6)
check("claims no distance off-axis", n.victim_range()[0] is None)
check("turns to centre it", abs(d.last[1]) > 0)
check("does NOT close on an unknown range", d.last[0] == 0.0)

print("10) horizon filter separates the floor victim from the wall marker")
# 480-row frame, camera level -> horizon at row 240. The victim object rests on the
# floor so its base is below that; the marker is mounted up the wall, above it.
check("horizon is the centre row when level", abs(M.horizon_row(480) - 240.0) < 1e-6)
check("victim on the floor is kept",     M.is_on_floor((300, 200, 60, 90), 480))   # base 290
check("wall marker is rejected",     not M.is_on_floor((300, 120, 60, 70), 480))   # base 190
check("a box straddling the horizon counts as floor",
      M.is_on_floor((300, 180, 60, 90), 480))                                      # base 270
check("tilting down raises the horizon",
      M.horizon_row(480, tilt_deg=10.0, vfov_deg=41.0) < 240.0)

print("11) floor object: camera range used when the victim is off-axis")
M.VICTIM_ON_FLOOR = True
n = M.Nav(FakeDrive(), FakeVision([V(0.40, 30.0)] * 6, geometry_ok=True), FakeLeds(),
          FakeSonar(front=0.50))
run(n, 6)
dist, src = n.victim_range()
check("falls back to the camera", src == "camera")
check("uses the camera's number, not the sonar's", abs(dist - 0.40) < 1e-9)

print("12) echo lost mid-approach: halt rather than coast on a stale range")
d, son = FakeDrive(), FakeSonar(front=0.50)
n = M.Nav(d, FakeVision([V()] * 20, geometry_ok=False), FakeLeds(), son)
run(n, 4)
son.front = None
n.step()
check("stops closing", d.last[0] == 0.0)
check("stays in APPROACH", n.state == M.APPROACH)

print(f"\n{len(fails)} failed")
sys.exit(1 if fails else 0)
