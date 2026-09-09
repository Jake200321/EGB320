#!/usr/bin/env python3
"""Navigation main loop -- demo behaviour: find a victim, stop 10 cm short, signal.

    green LED  ON     while a victim is being tracked
    yellow LED FLASH  once stopped within STOP_DISTANCE_M -- placeholder standing in
                      for Roger's collection mechanism

    python3 main.py                      # full run
    python3 main.py --no-motors          # everything except driving (motor HAT is dead)
    python3 main.py --placeholder-vision # use the stand-in detector, NOT Kushal's
    python3 main.py --check              # wire everything up, report, exit

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

SEARCH_TURN_RATE = 0.6      # rad/s, spin speed while looking for a victim
APPROACH_SPEED = 0.10       # m/s, closing speed once a victim is being tracked
CREEP_SPEED = 0.05          # m/s, inside CREEP_RANGE_M -- slow enough to stop cleanly
CREEP_RANGE_M = 0.25
HEADING_GAIN = 0.03         # rad/s per degree of bearing error
MAX_TURN_RATE = 1.2         # rad/s

DETECTION_DEBOUNCE = 3      # consecutive frames before believing a detection
LOST_GRACE_FRAMES = 8       # frames a victim may vanish for before we call it lost
YELLOW_FLASH_HZ = 2.0
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
    """Differential drive over the DFR0592 HAT.

    set_velocity(v, w) is the only thing nav calls: v forward m/s, w yaw rad/s with
    POSITIVE = LEFT. If the HAT isn't answering (which it currently isn't) this
    becomes a no-op that logs commands, so the rest of the loop still runs honestly.
    """

    TRACK_M = 0.123             # Pololu 30T sprocket centre-to-centre
    SPROCKET_CIRCUM_M = math.pi * 0.024
    GEAR_RATIO = 50
    MAX_WHEEL_RPM = 200.0       # MEASURE -- see motor_spin_test.py
    MIN_DUTY, MAX_DUTY = 25.0, 80.0   # MAX_DUTY: 6 V motors on a 7-12 V rail

    def __init__(self, enabled=True):
        self.board = None
        self.last = (0.0, 0.0)
        if not enabled:
            print("[drive] disabled (--no-motors)")
            return
        try:
            from motor_spin_test import load_board_class, I2C_BUS, HAT_ADDRESS
            board = load_board_class()(I2C_BUS, HAT_ADDRESS)
            for _ in range(3):
                if board.begin() == board.STA_OK:
                    break
                time.sleep(0.3)
            else:
                raise RuntimeError(f"no response at 0x{HAT_ADDRESS:02x} on i2c-{I2C_BUS}")
            board.set_encoder_enable(board.ALL)
            board.set_encoder_reduction_ratio(board.ALL, self.GEAR_RATIO)
            board.motor_stop(board.ALL)
            self.board = board
            print("[drive] DFR0592 online")
        except Exception as exc:                       # noqa: BLE001
            print(f"[drive] NO MOTORS -- {type(exc).__name__}: {exc}")

    def set_velocity(self, v_mps, w_rps):
        self.last = (v_mps, w_rps)
        if self.board is None:
            return
        half = self.TRACK_M / 2.0
        rpm_l = (v_mps - w_rps * half) / self.SPROCKET_CIRCUM_M * 60.0
        rpm_r = (v_mps + w_rps * half) / self.SPROCKET_CIRCUM_M * 60.0
        peak = max(abs(rpm_l), abs(rpm_r))
        if peak > self.MAX_WHEEL_RPM:                  # scale BOTH, or the turn radius changes
            rpm_l *= self.MAX_WHEEL_RPM / peak
            rpm_r *= self.MAX_WHEEL_RPM / peak
        self._channel(1, rpm_l, invert=False)
        self._channel(2, rpm_r, invert=True)           # motors face opposite ways

    def _channel(self, motor_id, rpm, invert):
        frac = min(abs(rpm) / self.MAX_WHEEL_RPM, 1.0)
        if frac < 0.01:
            self.board.motor_stop(motor_id)
            return
        duty = self.MIN_DUTY + frac * (self.MAX_DUTY - self.MIN_DUTY)
        forward = (rpm > 0) != invert
        self.board.motor_movement(motor_id,
                                  self.board.CW if forward else self.board.CCW, duty)

    def stop(self):
        """Never raises -- called on every exit path."""
        self.last = (0.0, 0.0)
        try:
            if self.board is not None:
                self.board.motor_stop(self.board.ALL)
        except Exception as exc:                       # noqa: BLE001
            print(f"[drive] stop failed: {exc}")


# ===========================================================================
# Vision
# ===========================================================================
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
        if frame is None:
            return UNUSABLE
        self.frame_h, self.frame_w = frame.shape[:2]

        if self._placeholder:
            boxes = self._placeholder_boxes(frame)
            dets = [{"class": VICTIM_CLASS_NAME, "bbox": b, "is_ground_object": True,
                     "angle_deg": self._angle(b), "distance_m": None} for b in boxes]
        else:
            dets = self.system.classify_frame(frame)
            if dets is None:
                return UNUSABLE                # too blurred to conclude anything

        victims = [d for d in dets if self._is_target(d)]
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
SEARCH, APPROACH, AT_VICTIM = "SEARCH", "APPROACH", "AT_VICTIM"


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

    # -- transitions ------------------------------------------------------
    def _enter(self, state):
        if state == self.state:
            return
        print(f"[nav] {self.state} -> {state}")
        self.state = state
        if state == AT_VICTIM:
            self.arrived_at = time.monotonic()
            self.drive.stop()
        if state == SEARCH:
            self.leds.green(False)
            self.leds.yellow(False)
            self.leds.red(False)

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
        v = self.victim
        if v is None:
            return None, None
        if abs(v.bearing_deg) <= US_TRUST_BEARING_DEG:
            d = self.sonar.front
            if d is not None:
                return d, "sonar"
        if v.distance_m is not None and VICTIM_ON_FLOOR:
            return v.distance_m, "camera"
        return None, None

    def step(self):
        self.sonar.update()                    # one ping per tick, before deciding
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
        elif self.state == AT_VICTIM:
            self._at_victim()

    def _search(self, confirmed):
        self.leds.green(False)
        if confirmed:
            self.leds.green(True)                      # green the moment we commit
            if self.can_range:
                self._enter(APPROACH)
            else:
                # Refuse to drive at a victim we cannot range at all. Holding with
                # green lit is the honest behaviour -- fit the front sonar, or
                # measure the camera constants.
                self.drive.stop()
            return
        # Spinning on the spot can't run into anything, but if the front sonar says
        # a wall is right there, don't let a later state drive into it unnoticed.
        front = self.sonar.front
        if front is not None and front < COLLISION_STOP_M:
            self.blocked = True
        else:
            self.blocked = False
        self.drive.set_velocity(0.0, SEARCH_TURN_RATE)  # spin on the spot and look

    def _approach(self, lost):
        if lost:
            self.victim = None
            self._enter(SEARCH)
            return

        v = self.victim
        self.leds.green(True)

        distance, source = self.victim_range()
        # Steer on bearing regardless -- turning towards the victim is what brings it
        # into the sonar cone. Vision reports +ve to the RIGHT; drive takes +ve as
        # LEFT, so the sign flips here. This is the one place the two conventions meet.
        w = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, -HEADING_GAIN * v.bearing_deg))

        if distance is None:
            # Can't range it from here: rotate to centre it (which is what makes the
            # sonar trustworthy) but don't close on an unknown distance.
            self.drive.set_velocity(0.0, w)
            return

        if distance - STOP_DISTANCE_M <= DISTANCE_TOLERANCE_M:
            print(f"[nav] stopping at {distance*100:.1f} cm ({source})")
            self._enter(AT_VICTIM)
            return

        speed = CREEP_SPEED if distance < CREEP_RANGE_M else APPROACH_SPEED
        self.drive.set_velocity(speed, w)

    def _at_victim(self):
        """Stopped 10 cm short. Flash yellow -- Roger's collection code goes here."""
        self.drive.stop()
        self.leds.green(True)
        phase = (time.monotonic() - self.arrived_at) * YELLOW_FLASH_HZ
        self.leds.yellow(int(phase * 2) % 2 == 0)
        # TODO: rescue.collect() -- replace the flash once the mechanism exists.


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

    leds = Leds()
    drive = Drive(enabled=not args.no_motors)
    sonar = Ultrasonics(enabled=not args.no_sonar)
    vision = VictimVision(use_placeholder=args.placeholder_vision)
    nav = Nav(drive, vision, leds, sonar)

    ranging = ("front sonar" if sonar.available else
               "camera only" if vision.geometry_ok else "NONE -- approach disabled")
    print(f"\n[status] motors={'yes' if drive.board else 'NO'}  "
          f"leds={'GPIO' if leds.real else 'console'}  "
          f"sonar={len(sonar.sensors)}/3  "
          f"ranging={ranging}\n")

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
        return

    period = 1.0 / CONTROL_HZ
    try:
        while True:
            t0 = time.monotonic()
            nav.step()
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        drive.stop()
        leds.all_off()
        vision.close()
        print("stopped")


if __name__ == "__main__":
    main()
