"""Nav logic checks that need no camera, no motors and no Pi.

    python3 test_nav.py

Fakes the three subsystems and drives Nav.step() through scripted detections, so the
parts that are easy to get quietly wrong -- debounce, the bearing sign flip between
vision and drive, the 10 cm stop, the refusal to approach without ranging -- are
checked on a laptop before anything touches hardware.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as M

class FakeDrive:
    def __init__(self): self.last=(0.0,0.0); self.stops=0; self.board=None
    def set_velocity(self,v,w): self.last=(v,w)
    def stop(self): self.last=(0.0,0.0); self.stops+=1

class FakeLeds:
    def __init__(self): self.g=None; self.y=None; self.ylog=[]
    def green(self,on): self.g=on
    def yellow(self,on): self.y=on; self.ylog.append(on)
    def all_off(self): self.g=self.y=False

class FakeVision:
    def __init__(self, script, geometry_ok=True):
        self.script=list(script); self.geometry_ok=geometry_ok
    def look(self):
        return self.script.pop(0) if self.script else None

def V(dist,bearing=0.0): return M.Victim(bearing,dist,(100,100,50,50),2500)

fails=[]
def check(name,cond):
    print(("  PASS  " if cond else "  FAIL  ")+name)
    if not cond: fails.append(name)

print("1) SEARCH spins and keeps green off")
d,l=FakeDrive(),FakeLeds()
n=M.Nav(d,FakeVision([None,None,None]),l)
for _ in range(3): n.step()
check("state SEARCH", n.state==M.SEARCH)
check("spinning, no forward", d.last==(0.0, M.SEARCH_TURN_RATE))
check("green off", l.g is False)

print("2) debounce: 1 frame does not trigger, 3 do")
d,l=FakeDrive(),FakeLeds()
n=M.Nav(d,FakeVision([V(0.5),None,V(0.5),V(0.5),V(0.5)]),l)
n.step(); check("still SEARCH after 1 hit", n.state==M.SEARCH)
for _ in range(4): n.step()
check("APPROACH after 3 consecutive", n.state==M.APPROACH)
check("green ON", l.g is True)

print("3) approach steers the right way (+bearing = right = negative yaw)")
d,l=FakeDrive(),FakeLeds()
n=M.Nav(d,FakeVision([V(0.5,20.0)]*5),l)
for _ in range(5): n.step()
v,w=d.last
check("driving forward", v>0)
check("yaw negative for a victim to the right", w<0)

print("4) stops at 10cm and flashes yellow")
d,l=FakeDrive(),FakeLeds()
n=M.Nav(d,FakeVision([V(0.5)]*3+[V(0.30),V(0.11)]+[V(0.11)]*40),l)
for _ in range(5): n.step()
check("AT_VICTIM at 0.11m", n.state==M.AT_VICTIM)
check("drive stopped", d.last==(0.0,0.0))
for _ in range(40): n.step(); time.sleep(0.012)
check("yellow actually toggled", True in l.ylog and False in l.ylog)
check("green stays on", l.g is True)

print("5) creep speed inside 25cm")
d,_=FakeDrive(),None
n=M.Nav(d,FakeVision([V(0.5)]*4),FakeLeds()); [n.step() for _ in range(4)]
far=d.last[0]
d2=FakeDrive(); n2=M.Nav(d2,FakeVision([V(0.20)]*4),FakeLeds()); [n2.step() for _ in range(4)]
check("slower when close", d2.last[0]<far)

print("6) losing the victim returns to SEARCH")
d,l=FakeDrive(),FakeLeds()
n=M.Nav(d,FakeVision([V(0.5)]*3+[None]*12),l)
for _ in range(15): n.step()
check("back to SEARCH", n.state==M.SEARCH)
check("green off again", l.g is False)

print("7) no camera geometry -> green on, but WILL NOT drive at it")
d,l=FakeDrive(),FakeLeds()
n=M.Nav(d,FakeVision([V(None)]*5, geometry_ok=False),l)
for _ in range(5): n.step()
check("stays in SEARCH", n.state==M.SEARCH)
check("green still ON", l.g is True)
check("not driving", d.last==(0.0,0.0))


# --------------------------------------------------------------- ultrasonics
class FakeSonar:
    """Front sonar only, scripted. Sides return nothing."""
    def __init__(self, front=None, available=True):
        self._front = front; self.available = available; self.sensors = {"front": 1}
    def update(self): pass
    @property
    def front(self): return self._front
    def get(self, name): return self._front if name == "front" else None
    def walls(self): return (None, None)


print("8) front sonar alone is enough to approach -- no camera geometry needed")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V(None)] * 6, geometry_ok=False), l, FakeSonar(front=0.50))
for _ in range(6): n.step()
check("reaches APPROACH on sonar alone", n.state == M.APPROACH)
check("actually driving forward", d.last[0] > 0)
check("range came from the sonar", n.victim_range()[1] == "sonar")

print("9) sonar not trusted when the victim is off to one side")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V(None, 30.0)] * 6, geometry_ok=False), l, FakeSonar(front=0.50))
for _ in range(6): n.step()
check("no distance claimed off-axis", n.victim_range()[0] is None)
check("turns to centre it", abs(d.last[1]) > 0)
check("does NOT close on an unknown range", d.last[0] == 0.0)

print("10) sonar stops the robot at 10 cm")
d, l = FakeDrive(), FakeLeds()
son = FakeSonar(front=0.50)
n = M.Nav(d, FakeVision([V(None)] * 30, geometry_ok=False), l, son)
for _ in range(4): n.step()
check("approaching", n.state == M.APPROACH)
son._front = 0.11                     # victim now 11 cm away
n.step()
check("AT_VICTIM at 11 cm", n.state == M.AT_VICTIM)
check("motors stopped", d.last == (0.0, 0.0))

print("11) camera is used when calibrated and the victim is off-axis")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V(0.40, 30.0)] * 6, geometry_ok=True), l, FakeSonar(front=0.50))
for _ in range(6): n.step()
dist, src = n.victim_range()
check("falls back to the camera", src == "camera")
check("uses the camera's number, not the sonar's", abs(dist - 0.40) < 1e-9)

print("12) sonar dropping out mid-approach halts rather than coasting")
d, l = FakeDrive(), FakeLeds()
son = FakeSonar(front=0.50)
n = M.Nav(d, FakeVision([V(None)] * 20, geometry_ok=False), l, son)
for _ in range(4): n.step()
son._front = None                     # echo lost
n.step()
check("stops closing", d.last[0] == 0.0)
check("stays in APPROACH", n.state == M.APPROACH)

print("\n%d failed" % len(fails)); sys.exit(1 if fails else 0)
