#!/usr/bin/env python3
"""Navigation main loop -- demo behaviour: find a victim, stop 10 cm short, signal.

    green LED  ON     while a victim is being tracked
    yellow LED FLASH  once stopped within STOP_DISTANCE_M -- placeholder standing in
                      for Roger's collection mechanism

    python3 main.py                      # full run, with the live camera window
    python3 main.py --no-motors          # everything except driving (motor HAT is dead)
    python3 main.py --no-display         # headless / over SSH with no X
    python3 main.py --placeholder-vision # use the stand-in detector, NOT Kushal's
    python3 main.py --check              # wire everything up, report, exit
    python3 main.py --blur-threshold 100 # put Kushal's blur guard back on
    python3 main.py --sonar-test         # just the three ranges, live

While it runs you get three views of what it's doing:
  * a camera window with Kushal's detection overlay, the horizon line, the victim
    nav has locked onto, the live range and the sonar readout. q or ESC quits.
  * a status line rewriting in place: state, victim bearing and range, all three
    sonar readings, which LEDs are lit, and the measured loop rate.
  * event lines above it whenever the state changes or a victim is reached.

Runs with pieces missing on purpose. The motor HAT doesn't enumerate yet and the real
victim detector isn't in the repo, so each subsystem degrades to a named stub and says
so at startup rather than refusing to start. What it will NOT do is invent a distance:
if the camera geometry below isn't measured, APPROACH is disabled rather than driving
at a made-up number.
"""

import argparse
import math
import os
import sys
import time

# Kushal's detectors sit at the repo root but import camera_capture_v2_0 / 
# class_profile_v2_0 as top-level modules, which live in vision/. Put both on the
# path so they resolve either way.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VISION_DIR = os.path.join(_REPO_ROOT, "vision")
for _p in (_REPO_ROOT, _VISION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===========================================================================
# CONFIG -- the only numbers you should need to touch
# ===========================================================================

GREEN_LED_BCM = 16          # victim detected
YELLOW_LED_BCM = 20         # collection placeholder
RED_LED_BCM = 21            # returning to base -- wired, not yet driven by the FSM

# Ultrasonics: (TRIG_BCM, ECHO_BCM). Front is the primary range to the victim;
# left/right are for walls. All reachable through the HAT's passthrough pins.
#
# *** ECHO IS A 5 V OUTPUT AND PI GPIO IS 3.3 V ONLY ***
# Wire each ECHO through a divider (1k in series, 2k to ground) or you will
# eventually damage the pin. TRIG is an input to the sensor and is fine direct.
# Right was on (20, 21) until the LEDs claimed those; moved to (22, 27).
# Full pin map -- keep these disjoint:
#   I2C to the motor HAT   2, 3
#   LEDs                   16 green, 20 yellow, 21 red
#   ultrasonics            23/24 front, 5/6 left, 22/27 right
ULTRASONIC_PINS = {
    "front": (23, 24),
    "left":  (5, 6),
    "right": (22, 27),
}
US_MAX_RANGE_M = 2.0        # ignore anything past this -- beyond the maze anyway
US_MIN_TRIGGER_GAP_S = 0.06 # HC-SR04 wants >60 ms between pings; sensors are fired
                            # one at a time so their echoes can't be confused
US_STALE_AFTER_S = 0.5      # a reading older than this is discarded, not reused
US_TRUST_BEARING_DEG = 10.0 # only believe the front sonar is ranging the VICTIM
                            # when the victim is this close to centre -- the cone
                            # is ~15 deg and it reports whatever is nearest in it

# VisionSystem discards any frame whose Laplacian variance falls under this, which
# on the real camera -- 8 ms exposure, high gain, ordinary room light -- threw away
# effectively every frame and made victim detection impossible from main.py.
# 0 disables the guard entirely, which is what the robot actually runs with.
# --blur-threshold overrides it (100.0 is Kushal's original) if motion blur ever
# becomes the bigger problem.
BLUR_THRESHOLD = 0.0

VICTIM_CLASS_NAME = "victim"   # matches the profiles.pkl classes: obstacle/ramp/rubble/victim

# The demo target is the victim OBJECT on the floor. But the arena also carries the
# wall MARKER that depicts a victim, and Kushal's 'victim' colour profile is trained
# on marker images -- so it will happily match both, and nav would otherwise be free
# to drive at whichever happens to look bigger.
#
# They're separated geometrically. The camera is level at 10 cm, so the horizon is
# the image centre row: anything resting on the floor has its base BELOW that row,
# and a marker mounted partway up a wall sits at or above it. Filtering on that both
# discards the marker and guarantees the surviving detections are on the ground
# plane, which is exactly the assumption ground_distance_from_bbox_bottom needs --
# so the camera range estimator becomes valid at the same time.
VICTIM_ON_FLOOR = True
HORIZON_MARGIN_PX = 8       # slack for mounting tolerance and tilt error

STOP_DISTANCE_M = 0.10      # stop this far short of the victim (assessment: 10 cm)
DISTANCE_TOLERANCE_M = 0.02 # close enough -- stops hunting back and forth
COLLISION_STOP_M = 0.12     # front sonar closer than this while searching = wall ahead
WALL_NEAR_M = 0.20          # side sonar below this = wall alongside

# --- exploring: drive straight, held straight by the encoders -----------------
SEARCH_SPEED = 0.13         # m/s forward while exploring
SEARCH_TURN_RATE = 2.4      # rad/s, spin rate when turning away from an obstacle
TURN_CLEAR_M = 0.35         # keep turning until the front is at least this clear

# Correction per tick of left-minus-right difference. It acts on ACCUMULATED ticks,
# not instantaneous speed, so it drives the total distance error to zero -- which is
# what "straight" means. Matching speeds alone would hold a constant heading error
# forever. TUNE: too high oscillates, too low drifts.
ENCODER_STRAIGHT_GAIN = 0.004      # rad/s per tick
MAX_STRAIGHT_CORRECTION = 1.0      # rad/s cap, so a big error can't spin the robot

# --- approach: hold the victim within HEADING_TOLERANCE_DEG -------------------
HEADING_TOLERANCE_DEG = 1.0   # aim to keep the victim inside this
# Beyond this it pivots; inside it, it corrects while driving. Set wide deliberately:
# correcting while rolling is both easier (no static friction to break) and smoother.
# A pivot is effectively bang-bang once the friction floor kicks in -- at 2.2 rad/s
# and 20 Hz it swings ~6 deg per tick -- so it needs plenty of margin to settle into,
# or it would hunt either side of the threshold instead of converging.
HEADING_COARSE_DEG = 20.0

# Heading lock. Once the victim is inside HEADING_LOCK_DEG the heading is committed
# and steering stops -- from there it drives straight. Bearings get noisy close in
# (the victim fills the frame, the bbox centre wanders), and chasing that noise
# steers the robot off a line that was already good enough.
#
# It only unlocks if the bearing drifts past HEADING_UNLOCK_DEG, which is well
# outside the noise: locking and unlocking on the same threshold would chatter.
HEADING_LOCK_DEG = 5.0
HEADING_UNLOCK_DEG = 15.0
# Two different floors, because pivoting and correcting are different physics.
# Stationary, a tracked chassis has to SKID both tracks sideways to rotate, and
# static friction across the whole contact patch is what was stalling turns under
# ~18 degrees -- the command was real, it just wasn't enough force. Already rolling,
# the tracks are moving and a much gentler differential steers fine.
MIN_TURN_RATE_PIVOT = 2.2     # rad/s floor when turning on the spot
MIN_TURN_RATE_MOVING = 0.3    # rad/s floor while driving forward
APPROACH_SPEED = 0.16       # m/s, closing speed once a victim is being tracked
CREEP_SPEED = 0.09          # m/s, inside CREEP_RANGE_M -- slow enough to stop cleanly
CREEP_RANGE_M = 0.25
HEADING_GAIN = 0.07         # rad/s per degree of bearing error
MAX_TURN_RATE = 2.5         # rad/s

DETECTION_DEBOUNCE = 3      # consecutive frames before believing a detection
LOST_GRACE_FRAMES = 8       # frames a victim may vanish for before we call it lost
YELLOW_FLASH_HZ = 2.0
RESCUE_TIMEOUT_S = 10.0     # how long the collection placeholder runs before giving up

# --- the blind run-in ---------------------------------------------------------
# Close up, the victim drops out of the bottom of the frame: at CAMERA_HEIGHT_M with
# VERTICAL_FOV_DEG, an object on the floor stops being visible somewhere under
# ~camera_blind_range_m(). Losing sight of it there is expected, NOT a lost victim --
# so once we're lined up and this close, the approach commits and finishes on the
# front sonar alone.
CLOSING_TRIGGER_M = 0.30    # start trusting sonar alone below this
CLOSING_ALIGN_DEG = 12.0    # ...but only if the victim was this well centred
CLOSING_TIMEOUT_S = 5.0     # give up and go back to searching if it never arrives
CONTROL_HZ = 20

# --- camera ranging: measured on the robot 2026-09-09 -------------------------
# The front sonar is still the primary estimator; these make the camera a usable
# fallback for when the victim is off to one side, outside the sonar's cone.
CAMERA_HEIGHT_M = 0.10      # lens centre above the floor
CAMERA_TILT_DEG = 0.0       # mounted level
VERTICAL_FOV_DEG = 41.0     # Camera Module 3 standard lens

# Bearing FOV. NOTE: vision/camera_capture_v2_0.py defaults to 45.0, which does not
# match a 41 deg vertical -- the Camera Module 3 standard lens is 66 x 41, and 45 x 41
# is not a real 4:3 lens. A too-small FOV understates every bearing (a victim 15 deg
# off-axis reads as ~10), which both slows the steering correction and lets the
# US_TRUST_BEARING_DEG gate believe the sonar when the victim is further off-axis than
# it thinks. Nav passes this explicitly rather than taking that default; Kushal's file
# still needs fixing at source.
HORIZONTAL_FOV_DEG = 66.0

# Placeholder detector only (--placeholder-vision). NOT Kushal's calibrated bands.
# Taken from the victim entry in profiles.pkl (hue 78-102, green-cyan) so the
# stand-in at least looks for the right colour. Still not a vision system.
PLACEHOLDER_HSV_LOW = (78, 183, 135)
PLACEHOLDER_HSV_HIGH = (102, 255, 255)
PLACEHOLDER_MIN_AREA_PX = 400


# ===========================================================================
# Terminal status + live camera window
# ===========================================================================
WINDOW_NAME = "EGB320 nav"

# What each state is actually doing, in words, for the terminal and the HUD.
STATE_LABEL = {
    "SEARCH":    "EXPLORING",
    "APPROACH":  "APPROACHING VICTIM",
    "CLOSING":   "CLOSING IN (camera blind)",
    "AT_VICTIM": "ENGAGING RESCUE",
    "DONE":      "STOPPED (rescue timed out)",
}


class StatusLine:
    """One self-rewriting line of live telemetry, plus event lines above it.

    Events (state changes, warnings) have to clear the status line before printing or
    they land on top of it and leave fragments behind, so everything routes through
    here rather than calling print() directly.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._width = 0

    def event(self, message):
        if self.enabled and self._width:
            print("\r" + " " * self._width + "\r", end="")
            self._width = 0
        print(message, flush=True)

    def update(self, text):
        if not self.enabled:
            return
        pad = max(0, self._width - len(text))
        print("\r" + text + " " * pad, end="", flush=True)
        self._width = len(text)

    def close(self):
        if self.enabled and self._width:
            print()
            self._width = 0


STATUS = StatusLine()


def status_text(nav, sonar, leds, hz):
    """The live line: what state we're in, what we can see, what the sonar reads."""
    state = STATE_LABEL.get(nav.state, nav.state)

    v = nav.victim
    if v is None:
        target = "victim: none"
    else:
        distance, source = nav.victim_range()
        rng = "range unknown" if distance is None else f"{distance*100:5.1f}cm ({source})"
        lock = " LOCK" if getattr(nav, "heading_locked", False) else "     "
        target = f"victim: {v.bearing_deg:+6.1f}deg{lock} {rng}"

    def cm(name):
        d = sonar.get(name)
        return "  ---" if d is None else f"{d*100:5.1f}"
    sonar_txt = f"sonar L{cm('left')} F{cm('front')} R{cm('right')} cm"

    lit = "".join(c if getattr(leds, "_state", {}).get(n) else "-"
                  for n, c in (("green", "G"), ("yellow", "Y"), ("red", "R")))
    return (f"{state:<19}| {target:<38}| {vision_summary(getattr(nav, 'vision', None))}"
            f" | {sonar_txt} | {lit} | {hz:4.1f}Hz")


def vision_summary(vision):
    """What vision did with the last frame -- the thing that was invisible before.

    A frame rejected by the blur guard produced no boxes and no message, which looks
    exactly like "nothing is there". It now says so, with the number it was judged on.
    """
    if vision is None:
        return "vis: --"
    if getattr(vision, "blur_rejected", False):
        blur = getattr(vision, "last_blur", None)
        thr = getattr(vision, "blur_threshold", None)
        return (f"vis: BLUR-REJECT {blur:.0f}<{thr:.0f}" if blur is not None and thr
                else "vis: BLUR-REJECT")
    counts = getattr(vision, "class_counts", {}) or {}
    blobs = ("none" if not counts else
             " ".join(f"{k}x{v}" for k, v in sorted(counts.items())))
    raw = getattr(vision, "n_victim_raw", 0)
    kept = getattr(vision, "n_victim_kept", 0)
    blur = getattr(vision, "last_blur", None)
    blur_txt = "" if blur is None else f" blur{blur:.0f}"
    return f"vis:{blur_txt} [{blobs}] victim {kept}/{raw}"


class Display:
    """Live camera window with the vision overlays plus a nav HUD.

    Off automatically when there's no GUI to draw into -- opencv-python-headless has
    no imshow at all, and an SSH session without X forwarding has nowhere to put a
    window. Either case disables the window and says so, rather than killing the run.
    """

    def __init__(self, enabled=True):
        self.ok = False
        self.cv2 = None
        if not enabled:
            return
        try:
            import cv2
            if not hasattr(cv2, "imshow"):
                raise RuntimeError("this OpenCV build has no GUI (opencv-python-headless)")
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 800, 600)
            self.cv2 = cv2
            self.ok = True
            STATUS.event(f"[display] window '{WINDOW_NAME}' open -- q or ESC to quit")
        except Exception as exc:                       # noqa: BLE001
            STATUS.event(f"[display] no camera window -- {type(exc).__name__}: {exc}\n"
                         "          (headless OpenCV, or no X display over SSH). "
                         "Everything else still runs; --no-display silences this.")

    def show(self, frame, detections, nav, sonar, draw_detections=None):
        """Draw and display one frame. Returns False if the user asked to quit."""
        if not self.ok or frame is None:
            return True
        self.cv2.imshow(WINDOW_NAME, render_hud(frame, detections, nav, sonar,
                                                draw_detections))
        return self.cv2.waitKey(1) & 0xFF not in (ord("q"), 27)

    def close(self):
        if self.ok:
            try:
                self.cv2.destroyAllWindows()
            except Exception:                          # noqa: BLE001
                pass


def render_hud(frame, detections, nav, sonar, draw_detections=None):
    """Overlay the vision output and nav's own state onto a copy of the frame.

    Separate from Display so it can be rendered and checked without a GUI.

    Draw order matters: the state banner goes down first, then Kushal's detection
    overlay on top of it, so his own readout isn't hidden behind our bar.
    """
    import cv2
    img = frame.copy()
    h, w = img.shape[:2]

    # --- state banner ---
    state = STATE_LABEL.get(nav.state, nav.state)
    colour = {"SEARCH": (200, 200, 200), "APPROACH": (0, 220, 255),
              "CLOSING": (0, 170, 255), "AT_VICTIM": (0, 255, 0),
              "DONE": (0, 0, 255)}.get(nav.state, (255, 255, 255))
    cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(img, state, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    distance, source = nav.victim_range() if nav.victim is not None else (None, None)
    if distance is not None:
        txt = f"{distance*100:.1f} cm ({source})   stop at {STOP_DISTANCE_M*100:.0f}"
        cv2.putText(img, txt, (w - 330, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if distance > STOP_DISTANCE_M else (0, 0, 255), 2)

    # --- Kushal's overlay: boxes, class, confidence, angle, distance ---
    if draw_detections is not None and detections:
        try:
            draw_detections(img, detections)
        except Exception:                              # noqa: BLE001
            pass

    # --- horizon: below it is floor. This line is what separates the victim object
    # from its wall marker, so being able to see it is the point.
    hy = int(horizon_row(h))
    cv2.line(img, (0, hy), (w, hy), (80, 80, 255), 1)
    cv2.putText(img, "horizon", (6, max(12, hy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 255), 1)

    # --- the victim nav has actually locked onto, distinct from everything else ---
    if nav.victim is not None:
        x, y, bw, bh = nav.victim.bbox
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 220, 255), 3)
        cx = x + bw // 2
        cv2.line(img, (cx, y), (cx, y + bh), (0, 220, 255), 1)
        cv2.circle(img, (w // 2, h - 30), 4, (255, 255, 255), -1)
        cv2.line(img, (w // 2, h - 30), (cx, y + bh), (0, 220, 255), 1)

    # --- what vision found, second row ---
    vision = getattr(nav, "vision", None)
    if vision is not None:
        counts = getattr(vision, "class_counts", {}) or {}
        txt = ("blobs: none" if not counts else
               "blobs: " + "  ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
        txt += f"   victim kept {getattr(vision,'n_victim_kept',0)}" \
               f"/{getattr(vision,'n_victim_raw',0)}"
        if getattr(nav, "heading_locked", False):
            txt += "   HEADING LOCKED"
        cv2.putText(img, txt, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)

    # --- sonar readout along the bottom ---
    cv2.rectangle(img, (0, h - 26), (w, h), (0, 0, 0), -1)
    parts = []
    for name in ("left", "front", "right"):
        d = sonar.get(name)
        parts.append(f"{name[0].upper()} {'---' if d is None else f'{d*100:.0f}cm'}")
    cv2.putText(img, "sonar  " + "   ".join(parts), (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    return img


# ===========================================================================
# LEDs
# ===========================================================================
class Leds:
    """Green = victim tracked. Yellow = collection placeholder, flashed by the loop.

    Falls back to console printing when gpiozero isn't available or the pins can't be
    claimed, so the nav logic stays testable on a laptop.
    """

    def __init__(self, green_pin=GREEN_LED_BCM, yellow_pin=YELLOW_LED_BCM,
                 red_pin=RED_LED_BCM):
        self.real = False
        try:
            from gpiozero import LED
            self._green = LED(green_pin)
            self._yellow = LED(yellow_pin)
            self._red = LED(red_pin)
            self.real = True
            print(f"[leds] GPIO {green_pin} green, {yellow_pin} yellow, {red_pin} red")
        except Exception as exc:                       # noqa: BLE001
            print(f"[leds] CONSOLE ONLY -- {type(exc).__name__}: {exc}")
        self._state = {"green": None, "yellow": None, "red": None}

    def _set(self, name, obj, on):
        if self._state[name] == on:
            return                                     # only log real transitions
        self._state[name] = on
        if self.real:
            obj.on() if on else obj.off()
        else:
            print(f"[led] {name} {'ON' if on else 'off'}")

    def green(self, on):
        self._set("green", getattr(self, "_green", None), on)

    def yellow(self, on):
        self._set("yellow", getattr(self, "_yellow", None), on)

    def red(self, on):
        # Wired and driveable, but nothing sets it yet -- there's no return-to-base
        # state in the FSM. Left here so that behaviour has somewhere to land.
        self._set("red", getattr(self, "_red", None), on)

    def all_off(self):
        try:
            self.green(False)
            self.yellow(False)
            self.red(False)
        except Exception:                              # noqa: BLE001
            pass


# ===========================================================================
# Ultrasonics
# ===========================================================================
class Ultrasonic:
    """One HC-SR04 / SRF05 on a trigger + echo pair.

    Timed in Python, so expect a centimetre or two of jitter -- fine against a 10 cm
    stop threshold, and no worse than gpiozero's own DistanceSensor, which does the
    same thing. Pinging is explicit rather than free-running precisely so the three
    sensors can be sequenced and never hear each other's echoes.
    """

    SPEED_OF_SOUND = 343.0     # m/s at ~20 C

    def __init__(self, trig_bcm, echo_bcm, max_range_m=US_MAX_RANGE_M):
        from gpiozero import DigitalInputDevice, DigitalOutputDevice
        self.trig = DigitalOutputDevice(trig_bcm)
        self.echo = DigitalInputDevice(echo_bcm)
        self.max_range_m = max_range_m
        # Longest an echo can legitimately take, plus margin. Past this there was
        # no echo -- nothing in range -- which is a normal answer, not a fault.
        self._timeout_s = (2.0 * max_range_m / self.SPEED_OF_SOUND) + 0.01

    def ping(self):
        """Distance in metres, or None if nothing answered within range."""
        self.trig.on()
        time.sleep(0.00001)                    # 10 us trigger pulse
        self.trig.off()

        deadline = time.monotonic() + self._timeout_s
        while not self.echo.value:             # wait for the echo to go high
            if time.monotonic() > deadline:
                return None
        rise = time.monotonic()

        deadline = rise + self._timeout_s
        while self.echo.value:                 # ...and for it to fall again
            if time.monotonic() > deadline:
                return None
        width = time.monotonic() - rise

        distance = width * self.SPEED_OF_SOUND / 2.0
        return distance if 0.0 < distance <= self.max_range_m else None


class Ultrasonics:
    """All three sensors, fired one per tick so they never overlap.

    Front is polled twice as often as the sides: it's the input to the stop decision,
    where the sides only inform wall logic. Readings go stale rather than lingering --
    a 2-second-old range is worse than admitting you don't know.
    """

    # front, left, front, right -- front lands on half the ticks
    ORDER = ["front", "left", "front", "right"]

    def __init__(self, pins=None, enabled=True):
        self.sensors = {}
        self.readings = {}                     # name -> (distance_m|None, timestamp)
        self._i = 0
        self._last_ping = 0.0
        if not enabled:
            print("[sonar] disabled (--no-sonar)")
            return
        for name, (trig, echo) in (pins or ULTRASONIC_PINS).items():
            try:
                self.sensors[name] = Ultrasonic(trig, echo)
            except Exception as exc:           # noqa: BLE001
                print(f"[sonar] {name} unavailable on TRIG {trig}/ECHO {echo} -- "
                      f"{type(exc).__name__}: {exc}")
        if self.sensors:
            print(f"[sonar] {', '.join(sorted(self.sensors))}")

    def update(self):
        """Fire at most one sensor. Call once per control tick."""
        if not self.sensors:
            return
        now = time.monotonic()
        if now - self._last_ping < US_MIN_TRIGGER_GAP_S:
            return                             # too soon; last echo may still be alive
        name = self.ORDER[self._i % len(self.ORDER)]
        self._i += 1
        sensor = self.sensors.get(name)
        if sensor is None:
            return
        self._last_ping = now
        self.readings[name] = (sensor.ping(), now)

    def get(self, name):
        """Latest range in metres, or None if unknown, out of range, or stale."""
        entry = self.readings.get(name)
        if entry is None:
            return None
        distance, t = entry
        if time.monotonic() - t > US_STALE_AFTER_S:
            return None
        return distance

    @property
    def front(self):
        return self.get("front")

    @property
    def available(self):
        return "front" in self.sensors

    def walls(self):
        """(left, right) booleans -- is there a wall alongside? None = don't know."""
        out = []
        for side in ("left", "right"):
            d = self.get(side)
            out.append(None if d is None else d < WALL_NEAR_M)
        return tuple(out)


# ===========================================================================
# Drive
# ===========================================================================
class Drive:
    """Differential drive on the unit's controller board (0x57, i2c-8).

    Thin wrapper over motors.MotorDriver so nav has one call -- set_velocity(v, w),
    v forward m/s, w yaw rad/s with POSITIVE = LEFT. If the board isn't there this
    becomes a no-op that records commands, so the rest of the loop still runs.
    """

    def __init__(self, enabled=True):
        self.driver = None
        self.last = (0.0, 0.0)
        if not enabled:
            STATUS.event("[drive] disabled (--no-motors)")
            return
        try:
            from motors import MotorDriver
            self.driver = MotorDriver()
        except Exception as exc:                   # noqa: BLE001
            STATUS.event(f"[drive] NO MOTORS -- {type(exc).__name__}: {exc}")

    @property
    def board(self):
        """Truthy when motors are live -- what the status line reports on."""
        return self.driver

    def set_velocity(self, v_mps, w_rps):
        self.last = (v_mps, w_rps)
        if self.driver is not None:
            self.driver.set_velocity(v_mps, w_rps)

    def read_encoders(self):
        return None if self.driver is None else self.driver.read_encoders()

    def stop(self):
        """Never raises -- called on every exit path."""
        self.last = (0.0, 0.0)
        if self.driver is not None:
            self.driver.stop()

    def close(self):
        if self.driver is not None:
            self.driver.close()


def heading_correction(bearing_deg, pivoting=False):
    """Yaw rate to put the victim on the nose, targeting HEADING_TOLERANCE_DEG.

    Inside the tolerance the correction is exactly zero -- without that deadband a
    proportional controller hunts either side of centre forever and never settles.

    Outside it, the magnitude is floored at MIN_TURN_RATE. That matters because the
    deadband compensation in motors.to_raw() turns anything smaller into a stop: a
    2 degree error would otherwise command a yaw the tracks never actually execute,
    and the robot would sit there off-heading believing it was correcting.

    Vision reports +ve to the RIGHT, drive takes +ve as LEFT, so the sign flips here.
    This is the one place the two conventions meet.
    """
    if abs(bearing_deg) <= HEADING_TOLERANCE_DEG:
        return 0.0
    floor = MIN_TURN_RATE_PIVOT if pivoting else MIN_TURN_RATE_MOVING
    w = math.copysign(max(abs(HEADING_GAIN * bearing_deg), floor), -bearing_deg)
    return max(-MAX_TURN_RATE, min(MAX_TURN_RATE, w))


def camera_blind_range_m(height_m=None, tilt_deg=None, vfov_deg=None):
    """How close a floor object can get before its base leaves the bottom of frame.

    Below this the camera cannot range it -- or see it at all -- which is why the
    last stretch of the approach has to run on sonar. Returns None if the camera
    geometry isn't known.
    """
    h = CAMERA_HEIGHT_M if height_m is None else height_m
    tilt = CAMERA_TILT_DEG if tilt_deg is None else tilt_deg
    vfov = VERTICAL_FOV_DEG if vfov_deg is None else vfov_deg
    if None in (h, tilt, vfov):
        return None
    edge_deg = tilt + vfov / 2.0        # depression angle at the bottom row
    if edge_deg <= 0:
        return None
    return h / math.tan(math.radians(edge_deg))


def horizon_row(frame_h, tilt_deg=None, vfov_deg=None):
    """Image row where the ground plane meets the horizon.

    Level camera -> the centre row. Tilted down -> the horizon rises up the frame.
    Everything below this row is ground; everything above is wall, marker or sky.
    """
    tilt = CAMERA_TILT_DEG if tilt_deg is None else tilt_deg
    vfov = VERTICAL_FOV_DEG if vfov_deg is None else vfov_deg
    if tilt is None or vfov is None:
        return frame_h / 2.0                      # best guess: assume level
    half = frame_h / 2.0
    # inverse of pixel_y_to_depression_angle at a depression of -tilt
    offset = math.tan(math.radians(-tilt)) / math.tan(math.radians(vfov / 2.0))
    return half * (1.0 + offset)


def is_on_floor(bbox, frame_h, margin_px=None):
    """True if this bbox's base is below the horizon, i.e. it's standing on the floor.

    This is what tells the victim object apart from the wall marker of the same colour.
    """
    margin = HORIZON_MARGIN_PX if margin_px is None else margin_px
    _, y, _, h = bbox
    return (y + h) > horizon_row(frame_h) + margin



class Victim:
    """One victim detection, already converted out of pixel space."""

    def __init__(self, bearing_deg, distance_m, bbox, area_px):
        self.bearing_deg = bearing_deg     # +ve = RIGHT of centre (vision's convention)
        self.distance_m = distance_m       # None when camera geometry isn't measured
        self.bbox = bbox                   # (x, y, w, h)
        self.area_px = area_px

    def __repr__(self):
        d = "?" if self.distance_m is None else f"{self.distance_m:.3f}m"
        return f"<Victim {self.bearing_deg:+.1f}deg {d} area={self.area_px}>"


UNUSABLE = object()     # frame carried no usable information (blurred, or no frame)


class VictimVision:
    """Kushal's VisionSystem, narrowed to the one thing nav needs: the victim object.

    Two things about classify_frame() that nav has to respect:

    * It returns None when the frame is too motion-blurred to trust, which is
      deliberately NOT an empty list. Empty means "looked, nothing there"; None means
      "no information this cycle". Collapsing them would make a fast pan look exactly
      like the victim vanishing. look() passes that through as UNUSABLE.

    * Its is_ground_object flag means "a colour detection rather than a marker-template
      one" -- so the wall MARKER, which shares the victim's colour, comes back as
      is_ground_object=True too. Verified: a frame with the victim on the floor and the
      same image up a wall yields two detections, both class=victim, both ground=True.
      So we gate on geometry as well, via is_on_floor().
    """

    def __init__(self, use_placeholder=False):
        from camera_capture_v2_0 import (CAMERA_HEIGHT_M as CH, CAMERA_TILT_DEG as CT,
                                         VERTICAL_FOV_DEG as VF, CameraCapture)
        self.camera = CameraCapture()
        self.frame_w = self.frame_h = None
        self.geometry_ok = None not in (CH, CT, VF)
        self.system = None
        self._placeholder = use_placeholder
        # kept for the live window; nav itself only needs the Victim
        self.last_frame = None
        self.last_detections = []
        self.draw_detections = None
        # diagnostics, so a frame going nowhere is visible instead of silent
        self.last_blur = None
        self.blur_rejected = False
        self.blur_threshold = None
        self.class_counts = {}
        self.n_victim_raw = 0
        self.n_victim_kept = 0

        if use_placeholder:
            print("[vision] PLACEHOLDER detector -- not the real vision system")
            return

        from vision_system_v2_0 import VisionSystem
        profiles = os.path.join(_VISION_DIR, "profiles.pkl")
        if not os.path.exists(profiles):
            raise SystemExit(f"profiles.pkl not found at {profiles}")
        cwd = os.getcwd()
        try:
            # VisionSystem and MarkerDetector both open paths relative to the cwd
            # (profiles.pkl, templates/), so run their construction from vision/.
            os.chdir(_VISION_DIR)
            self.system = VisionSystem("profiles.pkl")
        finally:
            os.chdir(cwd)
        self.draw_detections = type(self.system).draw_detections
        import vision_system_v2_0 as _vs
        self.blur_threshold = _vs.BLUR_VARIANCE_THRESHOLD
        print(f"[vision] VisionSystem, filtering for {VICTIM_CLASS_NAME!r}"
              f"{' on the floor' if VICTIM_ON_FLOOR else ''}")

    def _is_target(self, det):
        """Is this detection the victim OBJECT, as opposed to its wall marker?"""
        if det.get("class") != VICTIM_CLASS_NAME:
            return False
        if not det.get("is_ground_object", False):
            return False                       # a marker-template hit
        if VICTIM_ON_FLOOR and not is_on_floor(det["bbox"], self.frame_h):
            return False                       # colour hit on the wall marker
        return True

    def look(self):
        """The nearest victim, None if there isn't one, UNUSABLE if we can't tell."""
        frame = self.camera.read()
        self.last_frame = frame
        if frame is None:
            self.last_detections = []
            return UNUSABLE
        self.frame_h, self.frame_w = frame.shape[:2]

        if self._placeholder:
            boxes = self._placeholder_boxes(frame)
            dets = [{"class": VICTIM_CLASS_NAME, "bbox": b, "is_ground_object": True,
                     "angle_deg": self._angle(b), "distance_m": None} for b in boxes]
        else:
            # Measure the same thing classify_frame gates on, so a rejection can be
            # reported with its number rather than silently dropping the frame.
            try:
                self.last_blur = float(self.system._blur_variance(frame))
            except Exception:                  # noqa: BLE001
                self.last_blur = None
            dets = self.system.classify_frame(frame)
            if dets is None:
                self.blur_rejected = True
                self.last_detections = []
                self.class_counts = {}
                self.n_victim_raw = self.n_victim_kept = 0
                return UNUSABLE                # too blurred to conclude anything
            self.blur_rejected = False

        self.last_detections = dets
        counts = {}
        for d in dets:
            counts[d.get("class", "?")] = counts.get(d.get("class", "?"), 0) + 1
        self.class_counts = counts

        raw = [d for d in dets if d.get("class") == VICTIM_CLASS_NAME]
        victims = [d for d in raw if self._is_target(d)]
        self.n_victim_raw, self.n_victim_kept = len(raw), len(victims)
        if not victims:
            return None

        # Largest box = nearest. Fine while one victim is in frame at a time.
        best = max(victims, key=lambda d: d["bbox"][2] * d["bbox"][3])
        x, y, w, h = best["bbox"]
        return Victim(best["angle_deg"], best["distance_m"], best["bbox"], w * h)

    def _angle(self, bbox):
        from camera_capture_v2_0 import pixel_x_to_angle
        x, _, w, _ = bbox
        return pixel_x_to_angle(x + w / 2.0, self.frame_w)

    def _placeholder_boxes(self, frame_bgr):
        """STAND-IN ONLY -- a crude HSV blob finder. Not a vision system."""
        import cv2
        import numpy as np
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(PLACEHOLDER_HSV_LOW), np.array(PLACEHOLDER_HSV_HIGH))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours
                if cv2.contourArea(c) >= PLACEHOLDER_MIN_AREA_PX]

    def close(self):
        try:
            self.camera.release()
        except Exception:                              # noqa: BLE001
            pass


# ===========================================================================
# Nav loop
# ===========================================================================
SEARCH, APPROACH, CLOSING, AT_VICTIM, DONE = (
    "SEARCH", "APPROACH", "CLOSING", "AT_VICTIM", "DONE")


class Nav:
    def __init__(self, drive, vision, leds, sonar=None):
        self.drive, self.vision, self.leds = drive, vision, leds
        self.sonar = sonar if sonar is not None else Ultrasonics(enabled=False)
        self.state = SEARCH
        self.hits = 0            # consecutive frames with a victim
        self.misses = 0          # consecutive frames without one
        self.victim = None
        self.arrived_at = None
        self.blocked = False
        self._announced = False
        self.aligned = False        # was the victim well centred when last seen?
        self.closing_since = None
        self.heading_locked = False # committed to a heading; stop steering
        self.straight_ref = None    # encoder reading when this straight run began
        self.ticks = None           # latest cumulative encoder ticks

    # -- transitions ------------------------------------------------------
    def _enter(self, state):
        if state == self.state:
            return
        print(f"[nav] {self.state} -> {state}")
        self.state = state
        if state == AT_VICTIM:
            self.arrived_at = time.monotonic()
            self.drive.stop()
        if state == CLOSING:
            self.closing_since = time.monotonic()
            # Drop the stale detection: it's behind/underneath us now, and leaving it
            # around would draw a box where the victim no longer is.
            self.victim = None
        if state == DONE:
            # Set the final state here rather than waiting for the next tick, so
            # there's no window where the robot has stopped but the red LED hasn't
            # caught up. _done() then just holds it.
            self.drive.stop()
            self.leds.green(False)
            self.leds.yellow(False)
            self.leds.red(True)
        if state == SEARCH:
            self._announced = False
            self.aligned = False
            self.heading_locked = False
            self.straight_ref = None
            self.leds.green(False)
            self.leds.yellow(False)
            self.leds.red(False)
        if state == APPROACH:
            self.heading_locked = False

    @property
    def can_range(self):
        """Either estimator will do. Without one, we refuse to approach."""
        return self.sonar.available or self.vision.geometry_ok

    def victim_range(self):
        """Best available distance to the tracked victim, and where it came from.

        The front sonar is the primary estimator, but it ranges whatever is nearest
        in its cone -- not specifically the victim. So it's only trusted while the
        victim is near centre; off to one side, the camera estimate (if calibrated)
        is the honest source, and if neither applies we return None rather than
        guessing. Approach steers the victim towards centre anyway, so the sonar
        becomes valid exactly when it matters most -- the last few centimetres.
        """
        if self.state == CLOSING:
            d = self.sonar.front
            return (d, "sonar") if d is not None else (None, None)

        v = self.victim
        if v is None:
            return None, None

        # A locked heading is an assertion that we are pointed at the victim, so the
        # front sonar is ranging it whatever the bearing currently reads. Without
        # this, a noisy bearing past US_TRUST_BEARING_DEG would drop the range and
        # send it pivoting -- exactly the behaviour the lock exists to stop.
        if self.heading_locked:
            d = self.sonar.front
            if d is not None:
                return d, "sonar"

        if abs(v.bearing_deg) <= US_TRUST_BEARING_DEG:
            d = self.sonar.front
            if d is not None:
                return d, "sonar"
        if v.distance_m is not None and VICTIM_ON_FLOOR:
            return v.distance_m, "camera"
        return None, None

    def step(self):
        self.sonar.update()                    # one ping per tick, before deciding
        reader = getattr(self.drive, "read_encoders", None)
        self.ticks = reader() if reader else None
        seen = self.vision.look()

        # Debounce both ways: a single frame should neither trigger an approach
        # nor abandon one. Cheap insurance against detector flicker.
        if seen is UNUSABLE:
            # Blurred frame or dropped capture -- no evidence either way. Hold the
            # counters and keep acting on what we already believe, rather than
            # letting a fast pan read as the victim disappearing.
            pass
        elif seen is not None:
            self.hits, self.misses = self.hits + 1, 0
            self.victim = seen
        else:
            self.misses, self.hits = self.misses + 1, 0

        confirmed = self.hits >= DETECTION_DEBOUNCE
        lost = self.misses > LOST_GRACE_FRAMES

        if self.state == SEARCH:
            self._search(confirmed)
        elif self.state == APPROACH:
            self._approach(lost)
        elif self.state == CLOSING:
            self._closing()
        elif self.state == AT_VICTIM:
            self._at_victim()
        elif self.state == DONE:
            self._done()

    def _search(self, confirmed):
        self.leds.green(False)
        if confirmed:
            self.leds.green(True)                      # green the moment we commit
            if not self._announced:
                STATUS.event("[nav] VICTIM DETECTED -- green LED on")
                self._announced = True
            if self.can_range:
                self._enter(APPROACH)
            else:
                # Refuse to drive at a victim we cannot range at all. Holding with
                # green lit is the honest behaviour -- fit the front sonar, or
                # measure the camera constants.
                self.drive.stop()
            return
        front = self.sonar.front

        # Something ahead: turn on the spot until it's clear, then start a fresh
        # straight run. Hysteresis (in at COLLISION_STOP_M, out at TURN_CLEAR_M)
        # stops it dithering on the threshold.
        if self.blocked:
            if front is None or front >= TURN_CLEAR_M:
                self.blocked = False
                self.straight_ref = None       # new heading, new straight run
            else:
                self.drive.set_velocity(0.0, SEARCH_TURN_RATE)
                return
        elif front is not None and front < COLLISION_STOP_M:
            STATUS.event(f"[nav] wall at {front*100:.0f}cm -- turning")
            self.blocked = True
            self.straight_ref = None
            self.drive.set_velocity(0.0, SEARCH_TURN_RATE)
            return

        self.drive.set_velocity(SEARCH_SPEED, self.straight_correction())

    def straight_correction(self):
        """Yaw correction that holds an explore run straight, from the encoders.

        Compares how far each track has travelled since this run began and steers to
        equalise them. With no encoders it returns 0.0 -- an open-loop straight line,
        still forward, just uncorrected.
        """
        if self.ticks is None:
            return 0.0
        if self.straight_ref is None:
            self.straight_ref = self.ticks
            return 0.0
        left = self.ticks[0] - self.straight_ref[0]
        right = self.ticks[1] - self.straight_ref[1]
        # Left ahead of right means it has veered RIGHT. +w turns left, which speeds
        # the right track up and closes the gap.
        w = ENCODER_STRAIGHT_GAIN * (left - right)
        return max(-MAX_STRAIGHT_CORRECTION, min(MAX_STRAIGHT_CORRECTION, w))

    def _approach(self, lost):
        if lost:
            # Losing sight of the victim this close is expected, not a failure: it has
            # gone under the camera. If we were lined up on it, commit and finish on
            # the sonar rather than turning away and hunting for it again.
            front = self.sonar.front
            if self.aligned and front is not None and front <= CLOSING_TRIGGER_M:
                STATUS.event(f"[nav] victim under the camera at {front*100:.0f}cm -- "
                             "closing on sonar")
                self._enter(CLOSING)
            else:
                self.victim = None
                self._enter(SEARCH)
            return

        v = self.victim
        self.leds.green(True)

        self.aligned = abs(v.bearing_deg) <= CLOSING_ALIGN_DEG
        self.update_heading_lock(v.bearing_deg)
        distance, source = self.victim_range()

        if distance is None:
            # Can't range it from here: rotate to centre it (which is what makes the
            # sonar trustworthy) but don't close on an unknown distance.
            self.drive.set_velocity(0.0, heading_correction(v.bearing_deg, pivoting=True))
            return

        if distance - STOP_DISTANCE_M <= DISTANCE_TOLERANCE_M:
            STATUS.event(f"[nav] VICTIM REACHED -- stopping at "
                         f"{distance*100:.1f} cm ({source})")
            self._enter(AT_VICTIM)
            return

        # Past HEADING_COARSE_DEG, turn on the spot: driving on while badly off
        # heading travels further off course than the turn recovers.
        if not self.heading_locked and abs(v.bearing_deg) > HEADING_COARSE_DEG:
            self.drive.set_velocity(0.0, heading_correction(v.bearing_deg, pivoting=True))
            return

        # Locked: drive dead straight and ignore the bearing entirely.
        w = 0.0 if self.heading_locked else heading_correction(v.bearing_deg)
        speed = CREEP_SPEED if distance < CREEP_RANGE_M else APPROACH_SPEED
        self.drive.set_velocity(speed, w)

    def update_heading_lock(self, bearing_deg):
        """Latch the heading once we're lined up; only let go if it really drifts."""
        if self.heading_locked:
            if abs(bearing_deg) > HEADING_UNLOCK_DEG:
                STATUS.event(f"[nav] heading unlocked -- victim drifted to "
                             f"{bearing_deg:+.1f} deg")
                self.heading_locked = False
        elif abs(bearing_deg) <= HEADING_LOCK_DEG:
            STATUS.event(f"[nav] heading LOCKED at {bearing_deg:+.1f} deg -- "
                         "driving straight from here")
            self.heading_locked = True

    def _closing(self):
        """Final run-in with the victim out of frame. Straight ahead, sonar only.

        No steering: there's nothing to steer on, and a blind correction would be a
        guess. It ran in aligned, so it drives straight and stops on range.
        """
        self.leds.green(True)
        elapsed = time.monotonic() - self.closing_since

        if elapsed > CLOSING_TIMEOUT_S:
            STATUS.event(f"[nav] blind run-in gave up after {elapsed:.1f}s -- searching")
            self.drive.stop()
            self._enter(SEARCH)
            return

        front = self.sonar.front
        if front is None:
            # No vision and no range is no information at all -- hold rather than
            # drive forward on faith.
            self.drive.stop()
            return

        if front - STOP_DISTANCE_M <= DISTANCE_TOLERANCE_M:
            STATUS.event(f"[nav] VICTIM REACHED -- stopping at {front*100:.1f} cm (sonar)")
            self._enter(AT_VICTIM)
            return

        self.drive.set_velocity(CREEP_SPEED, 0.0)

    def _at_victim(self):
        """Stopped 10 cm short. Flash yellow -- Roger's collection code goes here.

        Gives up after RESCUE_TIMEOUT_S. The real mechanism will report its own
        success or failure; until it exists, a fixed timeout stands in for both so
        the robot ends in a defined state instead of flashing forever.
        """
        self.drive.stop()
        elapsed = time.monotonic() - self.arrived_at
        if elapsed >= RESCUE_TIMEOUT_S:
            STATUS.event(f"[nav] rescue timed out after {elapsed:.1f}s -- stopping")
            self._enter(DONE)
            return
        self.leds.green(True)
        phase = elapsed * YELLOW_FLASH_HZ
        self.leds.yellow(int(phase * 2) % 2 == 0)
        # TODO: rescue.collect() -- replace the flash once the mechanism exists.

    def _done(self):
        """Terminal. Motors off, red LED solid, nothing else changes.

        Deliberately has no way out: the run is over and the robot should sit still
        rather than wander off looking for another victim. Restart main.py to go again.
        """
        self.drive.stop()
        self.leds.green(False)
        self.leds.yellow(False)
        self.leds.red(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--no-motors", action="store_true", help="run without driving")
    ap.add_argument("--no-sonar", action="store_true", help="run without ultrasonics")
    ap.add_argument("--sonar-test", action="store_true",
                    help="print all three ranges continuously and exit on Ctrl-C")
    ap.add_argument("--placeholder-vision", action="store_true",
                    help="stand-in HSV detector instead of the real one")
    ap.add_argument("--no-display", action="store_true",
                    help="don't open the camera window (headless / over SSH)")
    ap.add_argument("--blur-threshold", type=float, default=None,
                    help=f"override the blur guard (default {BLUR_THRESHOLD:g} = off). "
                         "Frames scoring below it are discarded whole")
    ap.add_argument("--vision-debug", action="store_true",
                    help="print a line per frame: blur, blobs by class, victims kept")
    ap.add_argument("--no-status", action="store_true",
                    help="don't print the live status line")
    ap.add_argument("--check", action="store_true", help="wire up, report, exit")
    args = ap.parse_args()

    # Bring-up aid: no camera, no motors, just the sensors. Use it to check wiring
    # and that each sensor is where you think it is -- wave a hand in front of one.
    if args.sonar_test:
        sonar = Ultrasonics()
        if not sonar.sensors:
            raise SystemExit("no ultrasonics came up -- check ULTRASONIC_PINS wiring")
        print("Ctrl-C to stop. Wave a hand in front of each sensor in turn.")
        try:
            while True:
                sonar.update()
                cells = []
                for name in ("left", "front", "right"):
                    d = sonar.get(name)
                    cells.append(f"{name}: {'--- ' if d is None else f'{d*100:5.1f}'}cm")
                print("  ".join(cells), end="\r", flush=True)
                time.sleep(1.0 / CONTROL_HZ)
        except KeyboardInterrupt:
            print("\nstopped")
        return

    STATUS.enabled = not args.no_status
    leds = Leds()
    drive = Drive(enabled=not args.no_motors)
    sonar = Ultrasonics(enabled=not args.no_sonar)
    blur = BLUR_THRESHOLD if args.blur_threshold is None else args.blur_threshold
    if not args.placeholder_vision:
        import vision_system_v2_0 as _vs
        if _vs.BLUR_VARIANCE_THRESHOLD != blur:
            STATUS.event(f"[vision] blur guard {_vs.BLUR_VARIANCE_THRESHOLD:g} -> {blur:g}"
                         f"{' (off)' if blur <= 0 else ''}")
        _vs.BLUR_VARIANCE_THRESHOLD = blur
    vision = VictimVision(use_placeholder=args.placeholder_vision)
    vision.blur_threshold = blur
    nav = Nav(drive, vision, leds, sonar)
    display = Display(enabled=not args.no_display)

    ranging = ("front sonar" if sonar.available else
               "camera only" if vision.geometry_ok else "NONE -- approach disabled")
    STATUS.event(f"[status] motors={'yes' if drive.board else 'NO'}  "
          f"leds={'GPIO' if leds.real else 'console'}  "
                 f"sonar={len(sonar.sensors)}/3  "
                 f"ranging={ranging}")

    if args.check:
        for _ in range(12):                    # a few ticks so each sensor reports
            sonar.update()
            time.sleep(US_MIN_TRIGGER_GAP_S)
        for name in ("left", "front", "right"):
            d = sonar.get(name)
            print(f"  sonar {name:>5}: {'no reading' if d is None else f'{d*100:.1f} cm'}")
        drive.stop()
        leds.all_off()
        vision.close()
        display.close()
        return

    period = 1.0 / CONTROL_HZ
    hz = float(CONTROL_HZ)
    try:
        while True:
            t0 = time.monotonic()
            nav.step()

            # The window draws the frame nav just decided on, so what you see is
            # exactly what it acted on -- not a fresh capture that may differ.
            if not display.show(vision.last_frame, vision.last_detections,
                                nav, sonar, vision.draw_detections):
                STATUS.event("[display] quit requested")
                break

            if args.vision_debug:
                STATUS.event(f"[vision] {vision_summary(vision)}")
            STATUS.update(status_text(nav, sonar, leds, hz))
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))
            # measured, not the target -- if vision is slow this is where it shows
            hz = 0.8 * hz + 0.2 / max(time.monotonic() - t0, 1e-6)
    except KeyboardInterrupt:
        STATUS.event("\ninterrupted")
    finally:
        STATUS.close()
        drive.stop()
        leds.all_off()
        vision.close()
        display.close()
        print("stopped")


if __name__ == "__main__":
    main()
