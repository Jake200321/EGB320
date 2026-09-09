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
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "vision")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===========================================================================
# CONFIG -- the only numbers you should need to touch
# ===========================================================================

GREEN_LED_BCM = 17          # victim detected
YELLOW_LED_BCM = 27         # collection placeholder
# NOTE: the DFR0592 HAT covers the 40-pin header and has no passthrough pins.
# You need a stacking header (or to tap these off the LED board) before any GPIO
# is physically reachable. See the note at the bottom of this file.

VICTIM_CLASS_NAME = "victim"   # must match the name in Kushal's profiles.pkl

STOP_DISTANCE_M = 0.10      # stop this far short of the victim (assessment: 10 cm)
DISTANCE_TOLERANCE_M = 0.02 # close enough -- stops hunting back and forth

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

# --- camera geometry: MEASURE THESE ON THE ROBOT --------------------------
# Feeds vision/vision_system_v2_0.ground_distance_from_bbox_bottom(). Without all
# three there is no distance estimate, so the 10 cm rule cannot be enforced and
# APPROACH stays disabled. Nothing here can be guessed -- a wrong number produces a
# confidently wrong distance, which is worse than none.
CAMERA_HEIGHT_M = None      # lens centre height above the floor, metres
CAMERA_TILT_DEG = None      # degrees below horizontal (0.0 if mounted level)
VERTICAL_FOV_DEG = None     # Camera Module 3: 66 deg (standard) / 102 deg (wide) diagonal
                            # -- use the spec's VERTICAL figure, not the diagonal

# Placeholder detector only (--placeholder-vision). NOT Kushal's calibrated bands.
PLACEHOLDER_HSV_LOW = (20, 120, 120)     # yellow-ish, as the sim's victim token is
PLACEHOLDER_HSV_HIGH = (35, 255, 255)
PLACEHOLDER_MIN_AREA_PX = 400


# ===========================================================================
# LEDs
# ===========================================================================
class Leds:
    """Green = victim tracked. Yellow = collection placeholder, flashed by the loop.

    Falls back to console printing when gpiozero isn't available or the pins can't be
    claimed, so the nav logic stays testable on a laptop.
    """

    def __init__(self, green_pin=GREEN_LED_BCM, yellow_pin=YELLOW_LED_BCM):
        self.real = False
        try:
            from gpiozero import LED
            self._green = LED(green_pin)
            self._yellow = LED(yellow_pin)
            self.real = True
            print(f"[leds] GPIO {green_pin} (green), {yellow_pin} (yellow)")
        except Exception as exc:                       # noqa: BLE001
            print(f"[leds] CONSOLE ONLY -- {type(exc).__name__}: {exc}")
        self._state = {"green": None, "yellow": None}

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

    def all_off(self):
        try:
            self.green(False)
            self.yellow(False)
        except Exception:                              # noqa: BLE001
            pass


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


class VictimVision:
    """Camera + victim detector.

    Uses Kushal's ColourDetector, keeping only candidates labelled VICTIM_CLASS_NAME.
    Nav does the rest itself -- bearing via pixel_x_to_angle, range via
    ground_distance_from_bbox_bottom -- so the detector needs no knowledge of nav.
    """

    def __init__(self, use_placeholder=False):
        from vision.vision_system_v2_0 import CameraCapture, pixel_x_to_angle
        self._pixel_x_to_angle = pixel_x_to_angle
        self.camera = CameraCapture()
        self.frame_w = self.frame_h = None

        self.geometry_ok = None not in (CAMERA_HEIGHT_M, CAMERA_TILT_DEG, VERTICAL_FOV_DEG)
        if self.geometry_ok:
            from vision.vision_system_v2_0 import ground_distance_from_bbox_bottom
            self._ground_distance = ground_distance_from_bbox_bottom
        else:
            print("[vision] camera geometry not measured -- NO DISTANCE ESTIMATES.\n"
                  "         Set CAMERA_HEIGHT_M, CAMERA_TILT_DEG and VERTICAL_FOV_DEG\n"
                  "         at the top of main.py. APPROACH is disabled until then.")

        self.detect_boxes = self._placeholder if use_placeholder else self._find_detector()

    def _find_detector(self):
        """Build the real ColourDetector, or explain precisely what's missing."""
        try:
            from colour_detector import ColourDetector
        except ImportError as exc:
            raise SystemExit(self._missing_msg(f"import failed: {exc}"))

        # ColourDetector opens profiles_file relative to the cwd, so give it a path
        # that works no matter where main.py was launched from.
        for candidate in (os.path.join(_REPO_ROOT, "profiles.pkl"),
                          os.path.join(_REPO_ROOT, "vision", "profiles.pkl"),
                          "profiles.pkl"):
            if os.path.exists(candidate):
                detector = ColourDetector(candidate)
                break
        else:
            raise SystemExit(self._missing_msg("profiles.pkl not found"))

        names = set(detector.classes)
        if VICTIM_CLASS_NAME not in names:
            raise SystemExit(
                f"profiles.pkl has no {VICTIM_CLASS_NAME!r} class -- it holds {sorted(names)}.\n"
                "Set VICTIM_CLASS_NAME at the top of main.py to whichever of those is the victim.")
        print(f"[vision] ColourDetector, filtering for {VICTIM_CLASS_NAME!r}")

        def detect_victim_boxes(frame):
            # A candidate can carry several class names when a colour is shared
            # between classes; keep it if victim is among them.
            return [c["bbox"] for c in detector.detect(frame)
                    if VICTIM_CLASS_NAME in c["candidate_classes"]]

        return detect_victim_boxes

    @staticmethod
    def _missing_msg(why):
        return (
            f"Victim detector unavailable -- {why}\n\n"
            "  colour_detector.py is in the repo, but two things it needs are not:\n"
            "    class_profile_v2_0.py   the ClassProfile class, needed to unpickle\n"
            "    profiles.pkl            the calibrated HSV bands themselves\n"
            "  Both are produced/held by Kushal's calibration step. Ask him to push them\n"
            "  (profiles.pkl is a build artefact, so it may be gitignored on his side).\n\n"
            "  Meanwhile: python3 main.py --placeholder-vision  exercises the nav loop."
        )

    def _placeholder(self, frame_bgr):
        """STAND-IN ONLY. A crude HSV blob finder so the loop can be demonstrated
        before the real detector lands. Do not present this as the vision system."""
        import cv2
        import numpy as np
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(PLACEHOLDER_HSV_LOW), np.array(PLACEHOLDER_HSV_HIGH))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours
                 if cv2.contourArea(c) >= PLACEHOLDER_MIN_AREA_PX]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def look(self):
        """One frame -> the closest/largest victim, or None."""
        frame = self.camera.read()
        if frame is None:
            return None
        self.frame_h, self.frame_w = frame.shape[:2]

        boxes = self.detect_boxes(frame)
        if not boxes:
            return None

        # Largest box = nearest victim. Good enough while only one is in frame;
        # revisit if the demo ever has two at once.
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        bearing = self._pixel_x_to_angle(x + w / 2.0, self.frame_w)

        distance = None
        if self.geometry_ok:
            distance = self._ground_distance(
                y + h, self.frame_h,
                camera_height_m=CAMERA_HEIGHT_M,
                camera_tilt_deg=CAMERA_TILT_DEG,
                vertical_fov_deg=VERTICAL_FOV_DEG)

        return Victim(bearing, distance, (x, y, w, h), w * h)

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
    def __init__(self, drive, vision, leds):
        self.drive, self.vision, self.leds = drive, vision, leds
        self.state = SEARCH
        self.hits = 0            # consecutive frames with a victim
        self.misses = 0          # consecutive frames without one
        self.victim = None
        self.arrived_at = None

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

    def step(self):
        seen = self.vision.look()

        # Debounce both ways: a single frame should neither trigger an approach
        # nor abandon one. Cheap insurance against detector flicker.
        if seen is not None:
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
            if self.vision.geometry_ok:
                self._enter(APPROACH)
            else:
                # Refuse to drive at a victim we cannot range. Holding position with
                # green lit is the honest behaviour; measure the camera constants.
                self.drive.stop()
            return
        self.drive.set_velocity(0.0, SEARCH_TURN_RATE)  # spin and look

    def _approach(self, lost):
        if lost:
            self.victim = None
            self._enter(SEARCH)
            return

        v = self.victim
        self.leds.green(True)
        if v.distance_m is None:                       # ranging dropped out mid-approach
            self.drive.stop()
            return

        remaining = v.distance_m - STOP_DISTANCE_M
        if remaining <= DISTANCE_TOLERANCE_M:
            self._enter(AT_VICTIM)
            return

        # Steer on bearing. Vision reports +ve to the RIGHT; drive takes +ve as LEFT,
        # so the sign flips here -- this is the one place the two conventions meet.
        w = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, -HEADING_GAIN * v.bearing_deg))
        speed = CREEP_SPEED if v.distance_m < CREEP_RANGE_M else APPROACH_SPEED
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
    ap.add_argument("--placeholder-vision", action="store_true",
                    help="stand-in HSV detector instead of the real one")
    ap.add_argument("--check", action="store_true", help="wire up, report, exit")
    args = ap.parse_args()

    leds = Leds()
    drive = Drive(enabled=not args.no_motors)
    vision = VictimVision(use_placeholder=args.placeholder_vision)
    nav = Nav(drive, vision, leds)

    print(f"\n[status] motors={'yes' if drive.board else 'NO'}  "
          f"leds={'GPIO' if leds.real else 'console'}  "
          f"ranging={'yes' if vision.geometry_ok else 'NO -- approach disabled'}\n")

    if args.check:
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
