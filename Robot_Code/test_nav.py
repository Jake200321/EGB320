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
    """Records commands. `ticks` is writable so a test can simulate track slip."""
    def __init__(self, ticks=None):
        self.last = (0.0, 0.0); self.board = None; self.ticks = ticks
    def set_velocity(self, v, w): self.last = (v, w)
    def read_encoders(self): return self.ticks
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
print("1) EXPLORING drives straight ahead with green off")
d, l = FakeDrive(), FakeLeds()
run(M.Nav(d, FakeVision([None] * 3), l, FakeSonar(front=1.0)), 3)
check("driving forward", d.last[0] == M.SEARCH_SPEED)
check("dead straight with no encoder error", d.last[1] == 0.0)
check("green off", l.g is False)

print("1a) the encoders hold it straight")
d = FakeDrive(ticks=(0, 0))
n = M.Nav(d, FakeVision([None] * 20), FakeLeds(), FakeSonar(front=1.0))
n.step()                                   # first tick sets the reference
d.ticks = (500, 500)                       # both tracks equal
n.step()
check("tracks equal -> no correction", d.last[1] == 0.0)
d.ticks = (600, 500)                       # left ran ahead: veered RIGHT
n.step()
check("veering right -> turns LEFT to correct", d.last[1] > 0)
d.ticks = (500, 600)                       # right ran ahead: veered LEFT
n.step()
check("veering left -> turns RIGHT to correct", d.last[1] < 0)
d.ticks = (99999, 0)                       # absurd error
n.step()
check("correction is capped", abs(d.last[1]) <= M.MAX_STRAIGHT_CORRECTION)
check("still driving forward while correcting", d.last[0] == M.SEARCH_SPEED)

print("1b) no encoders -> open-loop straight, not a crash")
d = FakeDrive(ticks=None)
run(M.Nav(d, FakeVision([None] * 3), FakeLeds(), FakeSonar(front=1.0)), 3)
check("still drives forward", d.last == (M.SEARCH_SPEED, 0.0))

print("1c) a wall ahead turns on the spot, with hysteresis")
d, son = FakeDrive(ticks=(0, 0)), FakeSonar(front=1.0)
n = M.Nav(d, FakeVision([None] * 30), FakeLeds(), son)
n.step()
son.front = 0.08                           # wall
n.step()
check("stops driving and spins", d.last[0] == 0.0 and d.last[1] != 0.0)
son.front = 0.20                           # clearing, but not clear enough yet
n.step()
check("keeps turning below TURN_CLEAR_M", d.last[0] == 0.0)
son.front = 0.50                           # clear
n.step(); n.step()
check("resumes driving once clear", d.last[0] == M.SEARCH_SPEED)

print("2) one frame doesn't commit; DETECTION_DEBOUNCE frames do")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V(), None, V(), V(), V()]), l, FakeSonar(front=0.5))
n.step()
check("still SEARCH after a single hit", n.state == M.SEARCH)
run(n, 4)
check("APPROACH once debounced", n.state == M.APPROACH)
check("green ON", l.g is True)

print("2a) approach holds heading to within HEADING_TOLERANCE_DEG")
check("dead centre -> no correction", M.heading_correction(0.0) == 0.0)
check("inside tolerance -> no correction, no hunting",
      M.heading_correction(M.HEADING_TOLERANCE_DEG) == 0.0)
check("just outside tolerance -> it does correct",
      M.heading_correction(M.HEADING_TOLERANCE_DEG + 0.2) != 0.0)
check("a small error still commands a real turn, not one the tracks ignore",
      abs(M.heading_correction(2.0)) >= M.MIN_TURN_RATE)
check("victim to the right -> turn right", M.heading_correction(10.0) < 0)
check("victim to the left -> turn left", M.heading_correction(-10.0) > 0)
check("capped at MAX_TURN_RATE", abs(M.heading_correction(180.0)) <= M.MAX_TURN_RATE)
check("symmetric", M.heading_correction(7.0) == -M.heading_correction(-7.0))

print("2b) badly off heading, it turns on the spot rather than driving off course")
d = FakeDrive()
run(M.Nav(d, FakeVision([V(bearing=30.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("no forward motion while way off heading", d.last[0] == 0.0)
check("turning towards it", d.last[1] < 0)
d = FakeDrive()
run(M.Nav(d, FakeVision([V(bearing=0.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("lined up -> drives, no correction", d.last[0] > 0 and d.last[1] == 0.0)

print("3) approach steers the right way (+bearing = right = negative yaw)")
# Bearing inside HEADING_COARSE_DEG so it drives while correcting (2b covers the
# turn-on-the-spot case) and inside US_TRUST_BEARING_DEG so the sonar is believed.
d = FakeDrive()
run(M.Nav(d, FakeVision([V(bearing=3.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("driving forward", d.last[0] > 0)
check("yaw negative for a victim to the right", d.last[1] < 0)
d = FakeDrive()
run(M.Nav(d, FakeVision([V(bearing=-3.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
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

print("12a) losing sight of the victim close up commits to a sonar run-in")
# At 10 cm camera height the victim leaves the frame around 27 cm out. That must not
# read as "victim lost" -- it's the moment the approach is most nearly finished.
d, l, son = FakeDrive(), FakeLeds(), FakeSonar(front=0.25)
n = M.Nav(d, FakeVision([V()] * 3 + [None] * 20), l, son)
run(n, 4)
check("approaching first", n.state == M.APPROACH)
run(n, 12)                                     # victim disappears under the camera
check("hands over to CLOSING, does not go back to SEARCH", n.state == M.CLOSING)
check("still driving forward", d.last[0] > 0)
check("straight -- no steering with nothing to steer on", d.last[1] == 0.0)
check("green stays on", l.g is True)
son.front = 0.10
n.step()
check("sonar alone gets it to AT_VICTIM", n.state == M.AT_VICTIM)

print("12a-ii) but it only commits when lined up and close")
for front, bearing, why in [(0.80, 0.0, "too far to be the under-camera case"),
                            (0.25, 40.0, "off to one side, not lined up"),
                            (None, 0.0, "no sonar range at all")]:
    d2, l2 = FakeDrive(), FakeLeds()
    n2 = M.Nav(d2, FakeVision([V(bearing=bearing)] * 3 + [None] * 20), l2,
               FakeSonar(front=front))
    run(n2, 16)
    check(f"{why} -> back to SEARCH", n2.state == M.SEARCH)

print("12a-iii) the blind run-in gives up rather than driving forever")
d3 = FakeDrive()
n3 = M.Nav(d3, FakeVision([V()] * 3 + [None] * 40), FakeLeds(), FakeSonar(front=0.25))
run(n3, 16)
check("in CLOSING", n3.state == M.CLOSING)
n3.closing_since = time.monotonic() - (M.CLOSING_TIMEOUT_S + 0.1)
n3.step()
check("timed out back to SEARCH", n3.state == M.SEARCH)
check("stopped", d3.last == (0.0, 0.0))

print("12a-iv) no range mid-run-in: hold, don't drive on faith")
d4, son4 = FakeDrive(), FakeSonar(front=0.25)
n4 = M.Nav(d4, FakeVision([V()] * 3 + [None] * 40), FakeLeds(), son4)
run(n4, 16)
son4.front = None
n4.step()
check("stops when the echo drops out", d4.last == (0.0, 0.0))
check("stays in CLOSING", n4.state == M.CLOSING)

print("12a-v) the geometry the handover is based on")
_b = M.camera_blind_range_m()
check("blind range is derived, not hardcoded", _b is not None and 0.2 < _b < 0.35)
check("handover starts before the camera goes blind", M.CLOSING_TRIGGER_M > _b)

print("12b) rescue times out after RESCUE_TIMEOUT_S and stops with red")
d, l, son = FakeDrive(), FakeLeds(), FakeSonar(front=0.11)
n = M.Nav(d, FakeVision([V()] * 400), l, son)
run(n, 4)
check("reached AT_VICTIM", n.state == M.AT_VICTIM)
n.arrived_at = time.monotonic() - (M.RESCUE_TIMEOUT_S - 0.2)   # just short of the timeout
n.step()
check("still rescuing just before the timeout", n.state == M.AT_VICTIM)
n.arrived_at = time.monotonic() - (M.RESCUE_TIMEOUT_S + 0.1)   # just past it
n.step()
check("DONE once the timeout passes", n.state == M.DONE)
check("motors stopped", d.last == (0.0, 0.0))
check("red LED on", l.r is True)
check("green off", l.g is False)
check("yellow off", l.y is False)
run(n, 40)
check("stays DONE -- terminal, even with a victim in view", n.state == M.DONE)
check("red stays on", l.r is True)
check("motors stay stopped", d.last == (0.0, 0.0))

print("12c) motors.py maths: differential drive and raw-speed mapping")
import motors as MOT
_l, _r = MOT.wheel_speeds(0.10, 0.0)
check("straight: both wheels equal", _l == _r == 0.10)
_l, _r = MOT.wheel_speeds(0.0, 1.0)
check("spin: equal and opposite", abs(_l + _r) < 1e-9 and _r > 0)
_l, _r = MOT.wheel_speeds(0.10, 0.5)
check("+w turns left, so the right wheel runs faster", _r > _l)
check("stop maps to raw 0", MOT.to_raw(0.0) == 0)
check("a creep still clears stiction", MOT.to_raw(0.004) >= MOT.MIN_SPEED_RAW)
check("full speed is capped at MAX_SPEED_RAW",
      MOT.to_raw(99.0) == MOT.MAX_SPEED_RAW)
check("never exceeds the board's -127..127, which controller.py rejects",
      all(-127 <= MOT.to_raw(v) <= 127 for v in (-99, -1, -0.001, 0, 0.001, 1, 99)))
check("no dependency on the EGB320_Examples files",
      not any(w in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f)).read()
              for f in ("motors.py", "main.py", "motor_spin_test.py")
              for w in ("from controller import", "import controller\n")))
check("encoder wraparound: 65535 -> 0 counts as +1", MOT._to_i16(0 - 65535) == 1)
check("...and reversing counts down", MOT._to_i16(65535 - 0) == -1)
check("standby is the reserved -128", MOT.MotorController.STANDBY == -128)

# Motor direction: the sign flip must reach the board AND the odometry, or a
# reversed motor drives right and counts backwards.
class _FakeBoard:
    STANDBY = -128
    def __init__(s): s.sent = None; s.raw = (0, 0)
    def set_raw_motor_speed(s, l, r): s.sent = (l, r)
    def set_motor_shutdown_timeout(s, t): pass
    def get_raw_encoder_ticks(s): return s.raw
    def get_firmware_version(s): return (1, 3, 0)
    def standby(s): pass

_fb = _FakeBoard()
_old = (MOT.LEFT_SIGN, MOT.RIGHT_SIGN)
MOT.LEFT_SIGN = MOT.RIGHT_SIGN = -1
_d = MOT.MotorDriver(controller=_fb)
_d.set_raw(50, 50)
check("LEFT_SIGN/RIGHT_SIGN reach the board", _fb.sent == (-50, -50))
_fb.raw = (100, 100)                       # counters advanced
_t = _d.read_encoders()
check("reversed motors also count forward", _t == (-100, -100) or _t == (100, 100))
check("...specifically, the sign is applied to ticks too", _t == (-100, -100))
MOT.LEFT_SIGN, MOT.RIGHT_SIGN = _old
check("reverse is negative", MOT.to_raw(-0.10) < 0)
check("magnitude matches forward", abs(MOT.to_raw(-0.10)) == MOT.to_raw(0.10))
check("board address is the unit's, not the DFRobot one",
      (MOT.I2C_ADDR, MOT.I2C_BUS) == (0x57, 8))

print("13) a blurred frame is not the victim disappearing")
# classify_frame() returns None on a motion-blurred frame, which is NOT an empty
# result. Nav must hold what it knows rather than counting it as a miss -- otherwise
# a fast pan looks identical to the victim vanishing.
d, l, son = FakeDrive(), FakeLeds(), FakeSonar(front=0.50)
n = M.Nav(d, FakeVision([V()] * 3 + [M.UNUSABLE] * 60), l, son)
run(n, 4)
check("approaching before the blur", n.state == M.APPROACH)
run(n, 30)                                  # far more than LOST_GRACE_FRAMES
check("still APPROACH through a long blur", n.state == M.APPROACH)
check("green still on", l.g is True)
check("misses not counted", n.misses == 0)

print("14) a genuinely empty frame still loses the victim")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V()] * 3 + [None] * 12), l, FakeSonar(front=0.50))
run(n, 15)
check("empty frames DO drop to SEARCH", n.state == M.SEARCH)

print("15) every name main() reaches for actually exists")
# The rest of this file injects fakes, so it never touches the real Leds, Drive or
# Ultrasonics classes -- a refactor once deleted all three and every test still
# passed. This walks main() for the globals it calls and checks they're defined.
import ast
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")).read()
_tree = ast.parse(_src)
_main_fn = next(n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
_called = {n.func.id for n in ast.walk(_main_fn)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
_missing = sorted(c for c in _called if not hasattr(M, c) and not hasattr(__builtins__, c)
                  and c not in dir(__builtins__))
check(f"main() references nothing undefined{(' -- missing ' + ', '.join(_missing)) if _missing else ''}",
      not _missing)
for _name in ("Leds", "Drive", "Ultrasonics", "VictimVision", "Display", "Nav",
              "StatusLine", "render_hud", "status_text", "horizon_row", "is_on_floor"):
    check(f"{_name} is defined", hasattr(M, _name))

print("16) the real Leds and Ultrasonics construct off-Pi")
# Both must degrade instead of raising when there's no GPIO, or nothing can be
# tested on a laptop.
_leds = M.Leds()
_leds.green(True); _leds.yellow(True); _leds.red(True); _leds.all_off()
check("Leds falls back to console without gpiozero", _leds.real is False)
_son = M.Ultrasonics(enabled=False)
check("Ultrasonics(enabled=False) reports unavailable", _son.available is False)
check("...and returns no reading rather than raising", _son.front is None)
_son.update()
check("...and update() is a no-op", _son.get("front") is None)
_st = M.status_text(M.Nav(FakeDrive(), FakeVision([]), _leds, _son), _son, _leds, 20.0)
check("status_text renders", "EXPLORING" in _st and "sonar" in _st)

print("17) a frame that goes nowhere says so instead of looking empty")
# The blur guard in VisionSystem.classify_frame discards a whole frame, which used to
# produce no boxes and no message -- indistinguishable from "nothing is there". That
# cost real bench time, so the three outcomes must now be told apart in the summary.
try:
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "vision"))
    import camera_capture_v2_0 as _cc

    _frames = []
    class _FakeCam:
        def __init__(self, *a, **k): pass
        def read(self): return _frames.pop(0) if _frames else None
        def release(self): pass
    _cc.CameraCapture = _FakeCam

    _cwd = os.getcwd()
    _vision = M.VictimVision()
    os.chdir(_cwd)
    _rng = np.random.default_rng(2)

    _flat = np.full((480, 640, 3), 70, np.uint8)                  # no texture -> rejected
    _cyan = _rng.integers(45, 95, (480, 640, 3), dtype=np.uint8)
    _cyan[320:390, 250:390] = (255, 255, 0)                       # BGR cyan on the floor
    _empty = _rng.integers(45, 95, (480, 640, 3), dtype=np.uint8)

    _frames.append(_flat)
    check("blurred frame returns UNUSABLE", _vision.look() is M.UNUSABLE)
    check("...and is reported as a rejection", "BLUR-REJECT" in M.vision_summary(_vision))

    _frames.append(_cyan)
    _r = _vision.look()
    check("cyan object on the floor is a victim", isinstance(_r, M.Victim))
    check("...and the summary counts it", "victim 1/1" in M.vision_summary(_vision))

    _frames.append(_empty)
    check("textured but empty frame returns None", _vision.look() is None)
    _sum = M.vision_summary(_vision)
    check("...reported as empty, NOT as a rejection",
          "none" in _sum and "BLUR-REJECT" not in _sum)
except ImportError as _e:
    print(f"  SKIP  needs numpy + opencv + the vision package ({_e})")

print("18) the blur guard is OFF by default, so a dim frame still detects")
# This was a flag you had to remember to pass, and forgetting it meant no victim
# detection at all from main.py. It's now the default; this locks that in.
check("BLUR_THRESHOLD defaults to off", M.BLUR_THRESHOLD == 0.0)
try:
    import numpy as np
    import vision_system_v2_0 as _vsmod
    _vsmod.BLUR_VARIANCE_THRESHOLD = M.BLUR_THRESHOLD
    _frames.append(np.full((480, 640, 3), 70, np.uint8))
    _frames[-1][300:390, 250:390] = (255, 255, 0)     # turquoise, low-texture scene
    _v2 = M.VictimVision(); os.chdir(_cwd)
    _r2 = _v2.look()
    check("dim low-texture frame still yields a victim", isinstance(_r2, M.Victim))
except (ImportError, NameError) as _e:
    print(f"  SKIP  needs numpy + opencv ({_e})")

print(f"\n{len(fails)} failed")
sys.exit(1 if fails else 0)
