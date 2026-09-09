#!/usr/bin/env python3
"""Ultrasonic bring-up: print the distance from each sensor, continuously.

Self-contained -- no camera, no motors, no I2C.

    python3 ultrasonic_test.py                # all three, live
    python3 ultrasonic_test.py --sensor front # just one
    python3 ultrasonic_test.py --once         # one reading each, then exit
    python3 ultrasonic_test.py --plain        # no bars, one line per reading (for logging)
    python3 ultrasonic_test.py --interval 0.2 # slower

Wave a hand in front of one sensor at a time and watch which column moves -- that's
how you confirm each one is where you think it is. A sensor reading '---' saw no echo
at all, which at close range means it's not working, and at long range just means
nothing is in front of it.

WIRING -- ECHO IS 5 V, PI GPIO IS 3.3 V ONLY.
Put a divider on every ECHO line: series resistor to the pin, second resistor from
the pin to ground, sized 1:2. 10k series + 20k to ground gives 3.33 V and is fine
(so does 1k + 2k -- same ratio, lower impedance, more current). TRIG is an input to
the sensor and connects direct. Wiring ECHO straight to a GPIO will damage the pin.

Readings are timed in Python, so expect a centimetre or two of jitter. That's normal
and is no worse than gpiozero's own DistanceSensor, which works the same way.
"""

import argparse
import time

# BCM numbering -- must match ULTRASONIC_PINS in main.py
SENSOR_PINS = {
    "left":  (5, 6),      # (TRIG, ECHO)
    "front": (23, 24),
    "right": (22, 27),
}
DISPLAY_ORDER = ["left", "front", "right"]

SPEED_OF_SOUND = 343.0        # m/s at ~20 C
MAX_RANGE_M = 2.0
MIN_TRIGGER_GAP_S = 0.06      # >60 ms between pings so echoes can't overlap
BAR_FULL_CM = 100.0           # a full-width bar means this far or further


class Ultrasonic:
    def __init__(self, name, trig_bcm, echo_bcm):
        from gpiozero import DigitalInputDevice, DigitalOutputDevice
        self.name = name
        self.trig = DigitalOutputDevice(trig_bcm)
        self.echo = DigitalInputDevice(echo_bcm)
        self._timeout_s = (2.0 * MAX_RANGE_M / SPEED_OF_SOUND) + 0.01

    def ping(self):
        """Distance in metres, or None if nothing echoed back in range."""
        self.trig.on()
        time.sleep(0.00001)                 # 10 us trigger
        self.trig.off()

        deadline = time.monotonic() + self._timeout_s
        while not self.echo.value:          # wait for echo to rise
            if time.monotonic() > deadline:
                return None
        rise = time.monotonic()

        deadline = rise + self._timeout_s
        while self.echo.value:              # ...and fall
            if time.monotonic() > deadline:
                return None

        d = (time.monotonic() - rise) * SPEED_OF_SOUND / 2.0
        return d if 0.0 < d <= MAX_RANGE_M else None


def bar(distance_m, width=24):
    if distance_m is None:
        return " " * width
    filled = int(min(distance_m * 100.0 / BAR_FULL_CM, 1.0) * width)
    return "#" * max(1, filled) + "." * (width - max(1, filled))


def fmt(distance_m):
    return "  --- " if distance_m is None else f"{distance_m*100:5.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--sensor", choices=sorted(SENSOR_PINS), help="test just one")
    ap.add_argument("--once", action="store_true", help="one reading each, then exit")
    ap.add_argument("--plain", action="store_true", help="one line per reading, no bars")
    ap.add_argument("--interval", type=float, default=MIN_TRIGGER_GAP_S,
                    help="seconds between pings (min 0.06 -- echoes overlap below that)")
    args = ap.parse_args()

    gap = max(args.interval, MIN_TRIGGER_GAP_S)
    names = [args.sensor] if args.sensor else DISPLAY_ORDER

    sensors = {}
    for name in names:
        trig, echo = SENSOR_PINS[name]
        try:
            sensors[name] = Ultrasonic(name, trig, echo)
        except Exception as exc:                    # noqa: BLE001
            print(f"[{name}] could not claim TRIG {trig} / ECHO {echo} -- "
                  f"{type(exc).__name__}: {exc}")
    if not sensors:
        raise SystemExit("no sensors came up -- check SENSOR_PINS and the wiring")

    print("  ".join(f"{n}=GPIO{SENSOR_PINS[n][0]}/{SENSOR_PINS[n][1]}" for n in sensors))
    print("distances in cm, '---' = no echo.  Ctrl-C to stop.\n")
    if not args.plain and not args.once:
        print("  " + "".join(f"{n:>32}" for n in sensors))

    try:
        while True:
            # One at a time, never overlapping, so no sensor hears another's echo.
            readings = {}
            for name, sensor in sensors.items():
                readings[name] = sensor.ping()
                time.sleep(gap)

            if args.plain or args.once:
                print("  ".join(f"{n}: {fmt(readings[n])} cm" for n in sensors))
            else:
                cells = [f"{fmt(readings[n])} |{bar(readings[n])}|" for n in sensors]
                print("  " + "  ".join(cells), end="\r", flush=True)

            if args.once:
                return
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
