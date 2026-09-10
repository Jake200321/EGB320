"""
rescue_spool.py — drawstring spool control for the EGB320 rescue collector.

Hardware as built
-----------------
    Raspberry Pi 5  ->  signal on GPIO 12 (hardware PWM channel 0)
    FEETECH FS90MR  ->  continuous rotation, metal gear, 1.8 kg.cm @ 6 V
    XC4514 buck     ->  6.05 V servo rail, separate from the Pi rail
    7.4 V pack      ->  buck input, paralleled with the motor HAT feed

Bench-measured values (Agilent 33500B, 20 ms period, 3.3 V logic)
-----------------------------------------------------------------
    1.30 ms  winds cord IN   (mouth closes)
    1.50 ms  stopped
    1.70 ms  pays cord OUT   (mouth opens)

These are the starting defaults below. Re-run `calibrate` on the Pi anyway:
the true neutral shifts with supply voltage and between units, and a servo
sitting at a mis-set neutral creeps -- over a 7 minute run that silently
unwinds or over-tightens the drawstring.

Release is POWERED, not passive
-------------------------------
A geared servo cannot be back-driven, so the ribs can never unwind the spool
no matter how much energy they store. This closes DEC-06 option C. The ribs
only have to reopen the fabric once the cord is slack, which is a much lower
bar and good news for rib sizing -- but the servo must actively pay out.

Force, and why spool diameter matters
-------------------------------------
Cord force is F = tau / r, so HALVING the spool diameter DOUBLES the force
while costing only extra revolutions -- which are free on a continuous servo.
Bench testing showed the mouth only just closing at 20 mm, so a smaller spool
is the fix, not more current. A servo draws what it needs; you cannot push
more amps into it.

    20 mm spool -> 17.7 N, 2.1 rev
    10 mm spool -> 35.4 N, 4.1 rev

WIRING
------
    brown  -> GND, common with Pi GND
    red    -> 6 V from the XC4514, NEVER the Pi 5 V rail
    orange -> GPIO 12

Three of these stall around 600 mA. Motors must never run from the Pi's power
or GPIO pins [KB 6]. Tie the buck's output ground to a Pi ground pin so the
signal has a reference.

Usage
-----
    python3 rescue_spool.py identify           # step each servo, confirm wiring
    python3 rescue_spool.py calibrate 12       # find true neutral
    python3 rescue_spool.py takeup 12          # measure mm/s  UNDER LOAD
    python3 rescue_spool.py cinch 12           # calibrated close
    python3 rescue_spool.py release 12         # calibrated open
    python3 rescue_spool.py cycle 12           # close, pause, open
    python3 rescue_spool.py show 12            # print calibration + predictions
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

try:
    from gpiozero import Servo
except ImportError:  # pragma: no cover
    print("gpiozero not found.  sudo apt install python3-gpiozero python3-lgpio")
    raise

CAL_FILE = Path(__file__).with_name("servo_cal.json")

MIN_US, MAX_US = 1000, 2000
FRAME_S = 0.020                 # 50 Hz
NOMINAL_NEUTRAL_US = 1500

# Measured on the built pouch: cord pull from fully open to fully closed.
# Far below the 364 mm the perimeter geometry predicts, because the fabric
# gathers rather than the perimeter shrinking. Measured, not calculated.
TAKE_UP_MM = 130.0

SERVO_STALL_TORQUE_NM = 0.177   # FS90MR, 1.8 kg.cm at 6 V


@dataclass
class SpoolCal:
    """Per-unit calibration, cached so it survives between sessions."""
    gpio: int = 12
    neutral_us: int = 1500          # bench value; re-measure on the Pi
    wind_us: int = 1300             # cord IN  -> mouth closes
    release_us: int = 1700          # cord OUT -> mouth opens
    spool_dia_mm: float = 20.0
    takeup_mm_s: float = 0.0        # measured UNDER LOAD, not free-running
    release_mm_s: float = 0.0       # may differ: unloaded by the ribs
    note: str = ""

    # ---- derived, not stored ----
    def rev_for(self, mm: float) -> float:
        return mm / (math.pi * self.spool_dia_mm)

    def cord_force_N(self) -> float:
        return SERVO_STALL_TORQUE_NM / (self.spool_dia_mm / 2000.0)


def load_cal(gpio: int) -> SpoolCal:
    if CAL_FILE.exists():
        data = json.loads(CAL_FILE.read_text())
        if str(gpio) in data:
            return SpoolCal(**data[str(gpio)])
    return SpoolCal(gpio=gpio)


def save_cal(cal: SpoolCal) -> None:
    data = json.loads(CAL_FILE.read_text()) if CAL_FILE.exists() else {}
    data[str(cal.gpio)] = asdict(cal)
    CAL_FILE.write_text(json.dumps(data, indent=2))
    print(f"saved GPIO {cal.gpio} -> {CAL_FILE.name}")


class ContinuousServo:
    """Direct pulse-width control in microseconds.

    gpiozero maps value in [-1, 1] onto [min_pulse_width, max_pulse_width];
    working in microseconds keeps the calibration numbers physical.
    """

    def __init__(self, gpio: int, neutral_us: int = NOMINAL_NEUTRAL_US):
        self.gpio = gpio
        self.neutral_us = neutral_us
        self._servo = Servo(
            gpio,
            initial_value=None,      # start detached: no pulse, no motion
            min_pulse_width=MIN_US / 1e6,
            max_pulse_width=MAX_US / 1e6,
            frame_width=FRAME_S,
        )

    def set_us(self, us: float) -> None:
        us = max(MIN_US, min(MAX_US, us))
        mid = (MIN_US + MAX_US) / 2
        half = (MAX_US - MIN_US) / 2
        self._servo.value = (us - mid) / half

    def stop(self) -> None:
        """Neutral briefly, then detach.

        Detaching matters. A continuous servo held at a slightly-off neutral
        creeps, and creep on a drawstring is silent and cumulative.
        """
        self.set_us(self.neutral_us)
        time.sleep(0.15)
        self._servo.detach()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._servo.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Spool:
    """Drawstring spool.

    Motion is open-loop and timed: a continuous servo reports nothing about
    its own position. The capture-confirmed signal owed to Navigation cannot
    come from this class alone -- attach confirm_hook once a microswitch,
    current sense or encoder exists.
    """

    def __init__(self, cal: SpoolCal):
        self.cal = cal
        self.servo = ContinuousServo(cal.gpio, cal.neutral_us)
        self.confirm_hook = None

    def _run_for(self, us: int, seconds: float, poll=None) -> bool:
        confirmed = False
        self.servo.set_us(us)
        t_end = time.monotonic() + seconds
        try:
            while time.monotonic() < t_end:
                if poll is not None and poll():
                    confirmed = True
                    break
                time.sleep(0.005)
        finally:
            self.servo.stop()
        return confirmed

    def cinch(self, take_up_mm: float = TAKE_UP_MM,
              overtravel: float = 1.10, timeout_s: float = 10.0) -> bool:
        """Close the mouth by winding in `take_up_mm` of cord.

        Slight over-travel is deliberate and self-limiting: the ribs bottom
        out and the cord goes slack rather than the servo jamming, so
        over-running is safe while under-running leaves the mouth open.
        """
        if self.cal.takeup_mm_s <= 0:
            raise RuntimeError(
                "takeup_mm_s not calibrated -- run: rescue_spool.py takeup "
                f"{self.cal.gpio}"
            )
        seconds = min(take_up_mm * overtravel / self.cal.takeup_mm_s, timeout_s)
        print(f"cinch {take_up_mm:.0f} mm  ->  {seconds:.2f} s  "
              f"@ {self.cal.wind_us} us  ({self.cal.rev_for(take_up_mm):.1f} rev)")
        return self._run_for(self.cal.wind_us, seconds, self.confirm_hook)

    def release(self, take_up_mm: float = TAKE_UP_MM,
                overtravel: float = 1.20, timeout_s: float = 10.0) -> None:
        """Open the mouth by paying cord out.

        Powered, because a geared servo cannot be back-driven by the ribs.
        Rate may differ from cinch: the ribs assist here rather than resist,
        so release_mm_s is measured separately when available.
        """
        rate = self.cal.release_mm_s or self.cal.takeup_mm_s
        if rate <= 0:
            raise RuntimeError("no rate calibrated -- run takeup first")
        seconds = min(take_up_mm * overtravel / rate, timeout_s)
        print(f"release {take_up_mm:.0f} mm  ->  {seconds:.2f} s  "
              f"@ {self.cal.release_us} us")
        self._run_for(self.cal.release_us, seconds)

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
    print(f"\nGPIO {c.gpio}   spool {c.spool_dia_mm:.0f} mm")
    print(f"  neutral      {c.neutral_us} us")
    print(f"  wind (close) {c.wind_us} us")
    print(f"  release      {c.release_us} us")
    print(f"  take-up      {c.takeup_mm_s:.1f} mm/s"
          if c.takeup_mm_s else "  take-up      NOT CALIBRATED")
    print(f"  note         {c.note or '-'}")
    print(f"\nfor {TAKE_UP_MM:.0f} mm of cord:")
    print(f"  {c.rev_for(TAKE_UP_MM):.1f} revolutions")
    if c.takeup_mm_s:
        print(f"  {TAKE_UP_MM / c.takeup_mm_s:.2f} s")
    print(f"  cord force available: {c.cord_force_N():.1f} N")
    print("\nsmaller spool = more force, more revolutions (revolutions are cheap):")
    for d in (8, 10, 15, 20, 25):
        tmp = SpoolCal(spool_dia_mm=d)
        print(f"  {d:2d} mm -> {tmp.cord_force_N():5.1f} N, "
              f"{tmp.rev_for(TAKE_UP_MM):.1f} rev")
    print()


def cmd_identify(gpios) -> None:
    for g in gpios:
        print(f"\nGPIO {g}: wind 1 s, stop 1 s, release 1 s")
        with ContinuousServo(int(g)) as s:
            s.set_us(1300); time.sleep(1.0)
            s.stop();       time.sleep(1.0)
            s.set_us(1700); time.sleep(1.0)
        input("  Enter for next ")


def cmd_calibrate(gpio: int) -> None:
    """Bisect toward the true stop point by eye."""
    print("Watch the spool. Answer which way it turns; 's' when truly still.\n")
    lo, hi = 1440, 1560
    with ContinuousServo(gpio) as s:
        while hi - lo > 2:
            mid = (lo + hi) // 2
            s.set_us(mid)
            ans = input(f"  {mid} us -- (w)ind in / (r)elease / (s)topped ? ").strip().lower()
            s.stop()
            if ans.startswith("s"):
                lo = hi = mid
                break
            if ans.startswith("w"):
                lo = mid
            elif ans.startswith("r"):
                hi = mid
            else:
                print("    answer w, r or s")
    neutral = (lo + hi) // 2
    print(f"\nneutral = {neutral} us")
    cal = load_cal(gpio)
    cal.neutral_us = neutral
    save_cal(cal)


def cmd_takeup(gpio: int) -> None:
    """Measure cord rate in mm/s.

    Run this with the pouch and ribs FITTED. The servo slows appreciably under
    load, and a free-running figure makes every cinch undershoot.

    The result is also report evidence (TEST-03) -- log the date, cord type,
    spool diameter and rail voltage next to it.
    """
    cal = load_cal(gpio)
    if cal.neutral_us == NOMINAL_NEUTRAL_US:
        print("note: neutral is still the bench default; calibrate first\n")

    dia = input(f"spool diameter mm [{cal.spool_dia_mm:.0f}]: ").strip()
    if dia:
        cal.spool_dia_mm = float(dia)

    for direction, us_field, rate_field in (
        ("WIND IN (close)", "wind_us", "takeup_mm_s"),
        ("RELEASE (open)", "release_us", "release_mm_s"),
    ):
        print(f"\n--- {direction} ---")
        d = input(f"  pulse width us [{getattr(cal, us_field)}]: ").strip()
        if d:
            setattr(cal, us_field, int(d))
        secs = float(input("  run duration s [3]: ").strip() or 3)
        print("  Mark the cord at the casing exit. Enter to run.")
        input()
        with ContinuousServo(gpio, cal.neutral_us) as s:
            s.set_us(getattr(cal, us_field))
            time.sleep(secs)
            s.stop()
        mm = input("  measured cord movement mm (blank to skip): ").strip()
        if mm:
            setattr(cal, rate_field, float(mm) / secs)
            print(f"  -> {float(mm) / secs:.1f} mm/s")

    cal.note = input("\nnote (cord type, rail V, date): ").strip()
    save_cal(cal)
    cmd_show(gpio)


def cmd_cinch(gpio: int) -> None:
    cal = load_cal(gpio)
    mm = input(f"take-up mm [{TAKE_UP_MM:.0f}]: ").strip()
    with Spool(cal) as sp:
        ok = sp.cinch(float(mm) if mm else TAKE_UP_MM)
        print("confirmed" if ok else "completed (timed, unconfirmed)")


def cmd_release(gpio: int) -> None:
    with Spool(load_cal(gpio)) as sp:
        sp.release()


def cmd_cycle(gpio: int) -> None:
    """Full close-hold-open cycle. Feeds TEST-04 cycle timing."""
    cal = load_cal(gpio)
    with Spool(cal) as sp:
        t0 = time.monotonic()
        sp.cinch()
        t1 = time.monotonic()
        print("  holding 2 s")
        time.sleep(2.0)
        sp.release()
        t2 = time.monotonic()
    print(f"\ncinch {t1 - t0:.2f} s | release {t2 - t1 - 2:.2f} s "
          f"| cycle {t2 - t0:.2f} s")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd, args = sys.argv[1], sys.argv[2:]
    table = {
        "identify":  lambda: cmd_identify(args or [12, 13]),
        "calibrate": lambda: cmd_calibrate(int(args[0])),
        "takeup":    lambda: cmd_takeup(int(args[0])),
        "cinch":     lambda: cmd_cinch(int(args[0])),
        "release":   lambda: cmd_release(int(args[0])),
        "cycle":     lambda: cmd_cycle(int(args[0])),
        "show":      lambda: cmd_show(int(args[0]) if args else 12),
    }
    if cmd in table:
        table[cmd]()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
