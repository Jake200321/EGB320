"""
rescue_arm.py — deployment rocker control for the EGB320 rescue collector.

Companion to rescue_spool.py. The arm swings the pouch between a stowed pose
(inside the 200 mm startup cube) and a deployed pose (mouth down over the
victim, forward of the chassis). DEC-05: single driven rocker, one revolute
joint, one arc.

Target hardware
---------------
    Raspberry Pi 5      ->  signal on GPIO 13 (hardware PWM channel 1)
    Positional servo    ->  MG996R class, 9-13 kg.cm, metal gear
    XC4514 buck         ->  6 V rail shared with the spool servo

Not the FS90MR/FS90R/FT90R. Those are continuous rotation: they take a SPEED
command and cannot hold an angle. The arm needs a positional servo.

Why this is not just "write an angle"
-------------------------------------
1. SLEW LIMITING. A servo commanded straight to a new angle goes at full
   speed and stops dead. On a ~165 mm arm carrying a victim that means the
   payload swings, the pouch can spill, and the mounting takes a shock load
   every cycle. This module steps through intermediate positions at a
   commanded degrees-per-second instead.

2. HOLDING. Unlike the spool, the arm must resist gravity at the deployed
   pose, so it keeps its pulse train running. Detaching mid-travel would let
   the arm fall.

3. RANGE IS NOT WHAT THE DATASHEET SAYS. MG996R listings variously claim 90
   and 180 degrees, and it varies unit to unit. The endpoints here are found
   empirically with `jog`, not assumed.

Startup hazard
--------------
A servo jumps to whatever it is first commanded. If the arm is physically
somewhere else, that first command is a slam. Before powering up, put the arm
near the stowed pose by hand, then use `park` which attaches gently.

WIRING
------
    brown  -> GND, common with Pi GND
    red    -> 6 V from the XC4514, NEVER the Pi 5 V rail
    orange -> GPIO 13

An MG996R-class servo stalls near 2.5 A. Motors must never run from the Pi's
power or GPIO pins [KB 6]. Watch for the spool servo twitching when the arm
starts moving -- that is the arm's inrush dragging a shared rail down, and the
signal to split the rails.

Usage
-----
    python3 rescue_arm.py jog 13            # find the endpoints by hand
    python3 rescue_arm.py show 13           # print calibration
    python3 rescue_arm.py park 13           # gentle attach at stow
    python3 rescue_arm.py deploy 13         # stow -> deployed
    python3 rescue_arm.py stow 13           # deployed -> stow
    python3 rescue_arm.py sweep 13          # both ways, timed  (TEST-04)
    python3 rescue_arm.py capture 13 12     # full arm + spool cycle
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from gpiozero import Servo
except ImportError:  # pragma: no cover
    print("gpiozero not found.  sudo apt install python3-gpiozero python3-lgpio")
    raise

CAL_FILE = Path(__file__).with_name("arm_cal.json")

# Wider than the spool's 1000-2000. Most metal-gear standard servos accept
# roughly 600-2400 us, and the extra span is often what gets you past 90
# degrees of travel. Widen only after checking the servo does not buzz at the
# extremes -- buzzing means it is fighting its own end stop.
MIN_US, MAX_US = 700, 2300
FRAME_S = 0.020                  # 50 Hz

# Typical for a 180 degree servo over a 1000 us span. Only used to report
# angles; measure yours with a protractor and update via `jog`.
DEFAULT_US_PER_DEG = 5.56


@dataclass
class ArmCal:
    """Per-unit calibration. Endpoints are pulse widths, found empirically."""
    gpio: int = 13
    stow_us: int = 1450          # arm upright, inside the startup cube
    deploy_us: int = 2000        # arm forward-down, pouch over the victim
    us_per_deg: float = DEFAULT_US_PER_DEG
    slew_deg_s: float = 90.0     # commanded travel rate
    hold_at_stow: bool = False   # detach at stow if a mechanical rest carries it
    settle_s: float = 0.35       # let the servo catch up before releasing
    note: str = ""

    def span_us(self) -> int:
        return abs(self.deploy_us - self.stow_us)

    def span_deg(self) -> float:
        return self.span_us() / self.us_per_deg

    def slew_us_s(self) -> float:
        return self.slew_deg_s * self.us_per_deg


def load_cal(gpio: int) -> ArmCal:
    if CAL_FILE.exists():
        data = json.loads(CAL_FILE.read_text())
        if str(gpio) in data:
            return ArmCal(**data[str(gpio)])
    return ArmCal(gpio=gpio)


def save_cal(cal: ArmCal) -> None:
    data = json.loads(CAL_FILE.read_text()) if CAL_FILE.exists() else {}
    data[str(cal.gpio)] = asdict(cal)
    CAL_FILE.write_text(json.dumps(data, indent=2))
    print(f"saved GPIO {cal.gpio} -> {CAL_FILE.name}")


class PositionalServo:
    """Pulse-width control with an explicit notion of current position."""

    def __init__(self, gpio: int, start_us: int | None = None):
        self.gpio = gpio
        self._servo = Servo(
            gpio,
            initial_value=None,          # attach silently, no jump
            min_pulse_width=MIN_US / 1e6,
            max_pulse_width=MAX_US / 1e6,
            frame_width=FRAME_S,
        )
        self.current_us: int | None = start_us

    def _write(self, us: float) -> None:
        us = max(MIN_US, min(MAX_US, us))
        mid = (MIN_US + MAX_US) / 2
        half = (MAX_US - MIN_US) / 2
        self._servo.value = (us - mid) / half
        self.current_us = int(us)

    def snap(self, us: int) -> None:
        """Command immediately. Only safe when already at or near `us`."""
        self._write(us)

    def slew(self, target_us: int, us_per_s: float, step_hz: float = 50.0) -> float:
        """Walk the pulse width to `target_us` at a limited rate.

        Returns elapsed seconds. If the current position is unknown the move
        is refused rather than guessed -- guessing is how arms get slammed.
        """
        if self.current_us is None:
            raise RuntimeError("position unknown; call park() first")
        start = self.current_us
        delta = target_us - start
        if delta == 0:
            return 0.0
        duration = abs(delta) / max(us_per_s, 1.0)
        dt = 1.0 / step_hz
        t0 = time.monotonic()
        while True:
            frac = (time.monotonic() - t0) / duration
            if frac >= 1.0:
                break
            self._write(start + delta * frac)
            time.sleep(dt)
        self._write(target_us)
        return time.monotonic() - t0

    def detach(self) -> None:
        self._servo.detach()

    def close(self) -> None:
        try:
            self._servo.detach()
        finally:
            self._servo.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Arm:
    """Deployment rocker.

    Two commanded poses only, per DEC-05. The arm holds at deploy against
    gravity; at stow it can optionally detach if a mechanical rest carries the
    load, which saves current and removes servo hunting.
    """

    def __init__(self, cal: ArmCal):
        self.cal = cal
        self.servo = PositionalServo(cal.gpio)
        self.at_stow: bool | None = None

    def park(self) -> None:
        """Attach at the stowed pose.

        Place the arm near stow BY HAND first. This commands stow directly,
        so a large physical discrepancy will produce a fast correction.
        """
        self.servo.snap(self.cal.stow_us)
        time.sleep(self.cal.settle_s)
        self.at_stow = True
        if not self.cal.hold_at_stow:
            self.servo.detach()

    def deploy(self) -> float:
        if self.servo.current_us is None:
            self.park()
        if not self.cal.hold_at_stow:
            self.servo.snap(self.cal.stow_us)   # re-assert before moving
            time.sleep(0.1)
        t = self.servo.slew(self.cal.deploy_us, self.cal.slew_us_s())
        time.sleep(self.cal.settle_s)
        self.at_stow = False
        return t                                # holds here: gravity load

    def stow(self) -> float:
        if self.servo.current_us is None:
            raise RuntimeError("position unknown; call park() first")
        t = self.servo.slew(self.cal.stow_us, self.cal.slew_us_s())
        time.sleep(self.cal.settle_s)
        self.at_stow = True
        if not self.cal.hold_at_stow:
            self.servo.detach()
        return t

    def close(self) -> None:
        self.servo.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_show(gpio: int) -> None:
    c = load_cal(gpio)
    print(f"\nGPIO {c.gpio}")
    print(f"  stow      {c.stow_us} us")
    print(f"  deploy    {c.deploy_us} us")
    print(f"  span      {c.span_us()} us  ~ {c.span_deg():.0f} deg"
          f"   (at {c.us_per_deg:.2f} us/deg)")
    print(f"  slew      {c.slew_deg_s:.0f} deg/s -> "
          f"{c.span_deg() / c.slew_deg_s:.2f} s per sweep")
    print(f"  hold@stow {c.hold_at_stow}")
    print(f"  note      {c.note or '-'}\n")
    if c.span_deg() < 80:
        print("  span is under 80 deg -- check the servo actually reaches")
        print("  its endpoints, some MG996R units only travel ~90 deg total\n")


def cmd_jog(gpio: int) -> None:
    """Find the endpoints by hand.

    Moves in small steps only, so you can stop before the arm hits a hard
    stop or the chassis. Never type a large jump here.
    """
    cal = load_cal(gpio)
    print("Jog with:  + / -  (10 us)   ++ / --  (50 us)")
    print("Set with:  s = stow here,  d = deploy here,  q = quit")
    print("Watch for buzzing -- that means the servo is fighting an end stop.\n")

    us = cal.stow_us
    with PositionalServo(gpio) as sv:
        sv.snap(us)
        while True:
            print(f"  {us} us", end="")
            if cal.us_per_deg:
                print(f"   ({(us - cal.stow_us) / cal.us_per_deg:+.0f} deg from stow)",
                      end="")
            ans = input("  > ").strip().lower()
            if ans == "q":
                break
            elif ans == "s":
                cal.stow_us = us
                print(f"  stow = {us}")
            elif ans == "d":
                cal.deploy_us = us
                print(f"  deploy = {us}")
            elif ans in ("+", "-", "++", "--"):
                step = 50 if len(ans) == 2 else 10
                us += step if ans[0] == "+" else -step
                us = max(MIN_US, min(MAX_US, us))
                sv.snap(us)
                time.sleep(0.15)
            else:
                print("  use + - ++ -- s d q")

    deg = input(f"\nmeasured swing in degrees (blank to keep "
                f"{cal.span_deg():.0f}): ").strip()
    if deg:
        cal.us_per_deg = cal.span_us() / float(deg)
        print(f"  us/deg = {cal.us_per_deg:.2f}")
    cal.note = input("note (servo model, date): ").strip()
    save_cal(cal)
    cmd_show(gpio)


def cmd_park(gpio: int) -> None:
    print("Place the arm near the stowed pose by hand, then press Enter.")
    input()
    with Arm(load_cal(gpio)) as a:
        a.park()
        print("parked at stow")


def cmd_deploy(gpio: int) -> None:
    with Arm(load_cal(gpio)) as a:
        a.park()
        t = a.deploy()
        print(f"deployed in {t:.2f} s -- holding")
        input("Enter to stow ")
        a.stow()


def cmd_stow(gpio: int) -> None:
    a = Arm(load_cal(gpio))
    a.servo.current_us = a.cal.deploy_us     # assume we are deployed
    try:
        print(f"stowed in {a.stow():.2f} s")
    finally:
        a.close()


def cmd_sweep(gpio: int) -> None:
    """Both directions, timed. Feeds the TEST-04 cycle budget."""
    with Arm(load_cal(gpio)) as a:
        a.park()
        time.sleep(0.3)
        td = a.deploy()
        time.sleep(0.6)
        ts = a.stow()
    print(f"\ndeploy {td:.2f} s | stow {ts:.2f} s | total {td + ts:.2f} s")


def cmd_capture(arm_gpio: int, spool_gpio: int) -> None:
    """Full capture cycle: deploy, cinch, stow, hold.

    The arm and spool act in SEQUENCE, never together (DEC-06). That is a
    mechanical necessity, not a convenience -- cinching mid-swing closes the
    bag before it reaches the victim.
    """
    try:
        from rescue_spool import Spool, load_cal as load_spool_cal
    except ImportError:
        print("rescue_spool.py not found alongside this file")
        return

    with Arm(load_cal(arm_gpio)) as arm, Spool(load_spool_cal(spool_gpio)) as spool:
        arm.park()
        t0 = time.monotonic()

        print("1. deploy")
        arm.deploy()
        t1 = time.monotonic()

        print("2. cinch")
        spool.cinch()
        t2 = time.monotonic()

        print("3. stow with payload")
        arm.stow()
        t3 = time.monotonic()

        print(f"\ndeploy {t1-t0:.2f} s | cinch {t2-t1:.2f} s | "
              f"stow {t3-t2:.2f} s | TOTAL {t3-t0:.2f} s")
        print("victim retained. release over the base zone with:")
        print(f"  python3 rescue_spool.py release {spool_gpio}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd, args = sys.argv[1], sys.argv[2:]
    table = {
        "jog":     lambda: cmd_jog(int(args[0]) if args else 13),
        "show":    lambda: cmd_show(int(args[0]) if args else 13),
        "park":    lambda: cmd_park(int(args[0]) if args else 13),
        "deploy":  lambda: cmd_deploy(int(args[0]) if args else 13),
        "stow":    lambda: cmd_stow(int(args[0]) if args else 13),
        "sweep":   lambda: cmd_sweep(int(args[0]) if args else 13),
        "capture": lambda: cmd_capture(int(args[0]) if args else 13,
                                       int(args[1]) if len(args) > 1 else 12),
    }
    if cmd in table:
        table[cmd]()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
