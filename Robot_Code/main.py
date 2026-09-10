#!/usr/bin/env python3
"""Navigation main loop -- the final demo mission.

Explores the maze cell by cell, mapping walls with the three sonars (frontier
exploration: always head for the nearest unexplored cell). When vision confirms a
victim it drives at it, stops 10 cm short and collects it, then returns to base by
the shortest route through the map it has built (flood fill), releases it, and goes
back out for the next. Along every corridor it holds itself dead centre between
the walls, and holds heading from the encoders so neither track falls behind.

    yellow LED   exploring / navigating
    green LED    victim detected, approaching and collecting
    red LED      returning to base with a victim

    python3 main.py                      # full run: camera window + map window
    python3 main.py --no-camera          # explore and map only, no vision at all
    python3 main.py --base-cell 6,6 --start-heading W   # base in another corner
    python3 main.py --calibrate straight # measure TICKS_PER_M
    python3 main.py --calibrate turn     # measure EFFECTIVE_TRACK_M
    python3 main.py --calibrate sonar    # measure the wall readings at a cell centre
    python3 main.py --no-motors          # everything except driving
    python3 main.py --no-display         # headless / over SSH with no X
    python3 main.py --placeholder-vision # use the stand-in detector, NOT Kushal's
    python3 main.py --check              # wire everything up, report, exit
    python3 main.py --blur-threshold 100 # put Kushal's blur guard back on
    python3 main.py --sonar-test         # just the three ranges, live

While it runs you get four views of what it's doing:
  * a camera window with Kushal's detection overlay, the horizon line, the victim
    nav has locked onto, the live range and the sonar readout. q or ESC quits.
  * a map window: walls found, cells visited, the frontier, the planned route and
    the robot's pose. The final map is saved to runs/ when the program exits.
  * a status line rewriting in place: state, cell, victim bearing and range, all
    three sonar readings, which LEDs are lit, and the measured loop rate.
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

from maze import (HEADING_BY_NAME, HEADING_NAME, HEADING_RAD, LEFT_OF,  # noqa: E402
                  RIGHT_OF, Localiser, MazeMap, Odometry, nearest_heading, wrap)


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
# Only these are pinged. Pinging a sensor that isn't there costs a full echo timeout
# (~22 ms) every time its turn comes round, so take one out here if it's unplugged.
# The maze nav needs all three: front to stop at walls, sides to map and centre.
SONARS_FITTED = ("front", "left", "right")

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

# --- the maze -----------------------------------------------------------------
# The grid size, cell size and base corner are released before the demo -- CHECK
# these against the real field. Defaults are the 2026 sim scene: 7 x 7 cells of
# 280 mm on the 2 m table, base in the south-west corner facing north.
# --base-cell and --start-heading override the last two without editing.
MAZE_COLS = 7
MAZE_ROWS = 7
CELL_M = 0.28
BASE_CELL = (0, 6)          # (col, row); row 0 is the NORTH edge
START_HEADING = "N"         # which way the robot faces in the base cell
MISSION_TIME_S = 420.0      # the 7-minute demo -- motors stop when it's up
VICTIMS_TOTAL = 3           # stop once this many are home
# When two open sides are equally good, which to try first. Straight on always
# beats both, because a turn costs time and heading accuracy.
EXPLORE_PREFER_RIGHT = True

# --- odometry: MEASURE with  python3 main.py --calibrate straight / turn -------
# Encoder ticks per metre of track travel. 1400 counts per output rev (N20 1:50,
# x4 quadrature) over a 24 mm sprocket gives 18570 -- if the board counts x2 it's
# half that. The calibration drive tells you which.
TICKS_PER_M = 18570.0
# Track width the ODOMETRY uses for heading. Wider than the 123 mm sprocket spacing
# because the tracks skid sideways to rotate, and the encoders count the full track
# travel regardless. If 90 degree turns come out short or long, --calibrate turn.
EFFECTIVE_TRACK_M = 0.16

# --- sonar geometry: MEASURE with  python3 main.py --calibrate sonar -----------
# What each sonar reads with the robot centred in a cell and a wall right there.
FRONT_WALL_AT_CENTRE_M = 0.045
SIDE_WALL_AT_CENTRE_M = 0.050
# A wall counts as present when the reading is within this of those. Deliberately
# tight rather than half a cell: a victim or rubble standing in the NEXT cell reads
# ~15 cm, and calling that a wall would seal off the very cell the victim is in.
WALL_PRESENT_WINDOW_M = 0.08
# ...except towards a cell a victim has been placed in, where anything past this is
# taken to be the victim rather than a wall.
VICTIM_WALL_WINDOW_M = 0.05
SENSE_TIMEOUT_S = 0.6       # max wait at a cell centre for fresh L/F/R readings

# --- driving one cell ---------------------------------------------------------
SEARCH_SPEED = 0.13         # m/s along a corridor
ARRIVE_SLOW_M = 0.06        # slow down this far from the cell centre...
ARRIVE_SPEED = 0.08         # ...to this, so it stops where it means to
ARRIVE_TOLERANCE_M = 0.01
REVERSE_SPEED = 0.10        # backing out of a cell that turned out to be blocked
DRIVE_TIMEOUT_S = 6.0       # no arrival in this long (stuck on a bump) = blocked
# Something this much closer than the far side of the cell ahead is in the way (a
# victim, rubble, a wall the side sonar missed). Seen on this many fresh pings.
BLOCK_MARGIN_M = 0.05
BLOCK_CONFIRM = 3

# Heading hold: the encoders say which way it's pointing, so holding heading is
# the same thing as not letting either track fall behind the other.
HEADING_HOLD_GAIN = 3.0     # rad/s per rad of heading error
MAX_STEER_RATE = 0.8        # rad/s cap while driving
# Corridor centring: aim this many rad of heading back towards the middle per metre
# off centre, capped at MAX_CENTRE_ANGLE. 4.0 means 1 cm off -> 2.3 deg correction.
# TUNE: weaves -> lower CENTRE_GAIN; drifts into walls -> raise it.
CENTRE_GAIN = 4.0
MAX_CENTRE_ANGLE = 0.25     # rad, ~14 deg

# --- localisation: odometry corrected by the sonars against the map ------------
# Each sonar reading is compared with the range the map predicts from where the
# robot thinks it is, and the difference corrects position AND heading. These say
# how much to trust a reading and when to throw one out.
SONAR_SIGMA_M = 0.008       # HC-SR04 noise at these ranges
GATE_KNOWN_M = 0.05         # reading this far off a mapped wall = something else
GATE_MAYBE_M = 0.025        # ...tighter for an edge whose wall hasn't been seen yet
# Where the side sonars sit, forward of the turning centre. MEASURE. (Their sideways
# offset and the front sonar's forward offset come from the *_AT_CENTRE_M readings.)
SIDE_SONAR_FORWARD_M = 0.0

# --- turning on the spot ------------------------------------------------------
SEARCH_TURN_RATE = 2.4      # rad/s, the fastest it spins
TURN_GAIN = 3.0             # rad/s per rad left to turn, floored at MIN_TURN_RATE_PIVOT
TURN_TOLERANCE_RAD = 0.06   # ~3.4 deg: close enough, the drive's heading hold finishes it
TURN_TIMEOUT_S = 4.0

# --- rescue placeholders -- Roger's mechanism replaces these ------------------
SEEK_ATTEMPTS = 2           # tries at a victim cell whose sonar check fails
RESCUE_TIME_S = 3.0         # "collecting" at the victim, green LED
RELEASE_TIME_S = 2.0        # "releasing" at base, red LED

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

# --- the blind run-in ---------------------------------------------------------
# Close up, the victim drops out of the bottom of the frame: at CAMERA_HEIGHT_M with
# VERTICAL_FOV_DEG, an object on the floor stops being visible somewhere under
# ~camera_blind_range_m(). Losing sight of it there is expected, NOT a lost victim --
# so once we're lined up and this close, the approach commits and finishes on the
# front sonar alone.
CLOSING_TRIGGER_M = 0.30    # start trusting sonar alone below this
CLOSING_ALIGN_DEG = 12.0    # ...but only if the victim was this well centred
CLOSING_TIMEOUT_S = 5.0     # give up and go back to searching if it never arrives
CLOSING_JUMP_M = 0.05       # a front reading this much further than expected = lost it
CONTROL_HZ = 20

# --- camera ranging: measured on the robot 2026-09-09 -------------------------
# The front sonar is still the primary estimator; these make the camera a usable
# fallback for when the victim is off to one side, outside the sonar's cone.
CAMERA_HEIGHT_M = 0.10      # lens centre above the floor
CAMERA_TILT_DEG = 0.0       # mounted level
CAMERA_FORWARD_M = 0.06     # lens ahead of the turning centre -- MEASURE. Used to
                            # place a seen victim in the right cell of the map.
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
MAP_WINDOW_NAME = "EGB320 map"

# What each state is actually doing, in words, for the terminal and the HUD.
STATE_LABEL = {
    "SEARCH":    "EXPLORING",
    "SEEK":      "GOING TO VICTIM",
    "APPROACH":  "APPROACHING VICTIM",
    "CLOSING":   "CLOSING IN (camera blind)",
    "AT_VICTIM": "COLLECTING VICTIM",
    "RETURN":    "RETURNING TO BASE",
    "AT_BASE":   "RELEASING AT BASE",
    "DONE":      "FINISHED",
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
    cell = getattr(nav, "cell", None)
    where = "" if cell is None else f"{cell}{HEADING_NAME[nav.facing]} "
    return (f"{state:<19}| {where}{target:<38}| "
            f"{vision_summary(getattr(nav, 'vision', None))}"
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
    """Live camera window with the vision overlays plus a nav HUD, and the map window
    beside it showing what nav has explored.

    Off automatically when there's no GUI to draw into -- opencv-python-headless has
    no imshow at all, and an SSH session without X forwarding has nowhere to put a
    window. Either case disables the window and says so, rather than killing the run.
    """

    def __init__(self, enabled=True, camera=True, show_map=True):
        self.ok = False
        self.cv2 = None
        self.camera = camera
        self.show_map = show_map
        if not enabled:
            return
        try:
            import cv2
            if not hasattr(cv2, "imshow"):
                raise RuntimeError("this OpenCV build has no GUI (opencv-python-headless)")
            if camera:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 800, 600)
            if show_map:
                cv2.namedWindow(MAP_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self.cv2 = cv2
            self.ok = True
            STATUS.event("[display] windows open -- q or ESC to quit")
        except Exception as exc:                       # noqa: BLE001
            STATUS.event(f"[display] no windows -- {type(exc).__name__}: {exc}\n"
                         "          (headless OpenCV, or no X display over SSH). "
                         "Everything else still runs; --no-display silences this.")

    def show(self, frame, detections, nav, sonar, draw_detections=None):
        """Draw and display one tick. Returns False if the user asked to quit."""
        if not self.ok:
            return True
        if self.camera and frame is not None:
            self.cv2.imshow(WINDOW_NAME, render_hud(frame, detections, nav, sonar,
                                                    draw_detections))
        if self.show_map:
            from map_view import render_map
            self.cv2.imshow(MAP_WINDOW_NAME, render_map(nav))
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
              "CLOSING": (0, 170, 255), "AT_VICTIM": (0, 255, 0), "SEEK": (0, 255, 0),
              "RETURN": (0, 0, 255), "AT_BASE": (0, 0, 255),
              "DONE": (200, 200, 200)}.get(nav.state, (255, 255, 255))
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
    """The assessment's colours: yellow = exploring or navigating, green = victim
    detected and collection in progress, red = returning to base with a victim.

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

    # front, left, front, right -- front lands on half the ticks. Anything not in
    # SONARS_FITTED is skipped, so with only the front fitted it gets every tick.
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
            if name not in SONARS_FITTED:
                continue
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
        # Skip past anything not fitted rather than burning a tick on it.
        for _ in range(len(self.ORDER)):
            name = self.ORDER[self._i % len(self.ORDER)]
            self._i += 1
            if name in self.sensors:
                break
        else:
            return
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

    def since(self, name, t):
        """(True, distance) if `name` pinged after time t, else (False, None).

        A fresh None distance is real information -- no echo, nothing near -- which
        is why this exists alongside get(), where None also means "don't know".
        """
        entry = self.readings.get(name)
        if entry is None or entry[1] < t:
            return False, None
        return True, entry[0]

    def stamp(self, name):
        """When `name` last pinged, so callers can tell a new reading from a repeat."""
        entry = self.readings.get(name)
        return None if entry is None else entry[1]

    @property
    def front(self):
        return self.get("front")

    @property
    def available(self):
        return "front" in self.sensors


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


# ===========================================================================
# Grid motion: turn, drive one cell, back out
# ===========================================================================
def _clamp(v, limit):
    return max(-limit, min(limit, v))


class Mover:
    """Moves the robot between cell centres, one tick at a time.

    Start a move with move_to(), then call update() every tick until it returns an
    outcome -- "arrived", or "blocked" once it has backed out to where it started.
    A move turns on the spot to the heading first if it isn't already facing it.

    It steers on the Localiser's pose, which the sonars keep correcting against the
    walls on the map. So while driving along a corridor it:
      * holds heading -- heading comes from the difference between the tracks, so
        holding it is what stops one track falling behind the other;
      * steers back onto the corridor's centreline whenever it's off it -- dead
        centre, with the side walls telling the pose where that is;
      * stops at the cell centre, where a wall ahead pins down exactly how far;
      * backs out if something turns up in the way that the map said was open.
    """

    IDLE, TURN, DRIVE, BACKOUT = "IDLE", "TURN", "DRIVE", "BACKOUT"

    def __init__(self, drive, sonar, odo):
        self.drive, self.sonar, self.odo = drive, sonar, odo
        self.phase = self.IDLE
        self.lateral = None                         # m right of centre, for the HUD
        self.near_since = None

    @property
    def busy(self):
        return self.phase != self.IDLE

    def stop(self):
        self.phase = self.IDLE
        self.drive.stop()

    def move_to(self, x, y, heading, grid=True, through=False):
        """Face `heading` (world rad), then drive straight to (x, y).

        grid=True means a cardinal move between cell centres, which turns on blocked
        detection. A rejoin move after a victim approach is grid=False.

        through=True rolls over the target at full speed without stopping, for a
        known cell the route goes straight on from.
        """
        self.target, self.heading, self.grid = (x, y), heading, grid
        self.through = through
        self.near_since = None          # when it came within ARRIVE_SLOW_M
        self.block_count = 0
        self.block_info = None          # (front reading, distance to go) when blocked
        self._front_stamp = None
        now = time.monotonic()
        self.phase_started = now
        if abs(wrap(heading - self.odo.theta)) > TURN_TOLERANCE_RAD:
            self.phase, self.turn_sign = self.TURN, None
        else:
            self._begin_drive(now)

    def _begin_drive(self, now):
        self.phase = self.DRIVE
        self.phase_started = now
        self.start = (self.odo.x, self.odo.y)
        # Which way the target lies at the start. A forward move that overshoots has
        # arrived; only a move that STARTED behind the target reverses to it.
        ux, uy = math.cos(self.heading), math.sin(self.heading)
        self.forward = ((self.target[0] - self.odo.x) * ux
                        + (self.target[1] - self.odo.y) * uy) >= 0

    def update(self):
        if self.phase == self.TURN:
            return self._turn()
        if self.phase == self.DRIVE:
            return self._drive()
        if self.phase == self.BACKOUT:
            return self._backout()
        return None

    # -- phases ----------------------------------------------------------------
    def _turn(self):
        now = time.monotonic()
        err = wrap(self.heading - self.odo.theta)
        if self.turn_sign is None:
            self.turn_sign = 1.0 if err >= 0 else -1.0
        # Crossing the target counts as done: at the pivot floor it moves ~6 deg a
        # tick, so it can't always land inside the tolerance. The drive's heading
        # hold takes out what's left. (Only near the target -- at 180 deg the error
        # flips sign by wrapping, not by crossing.)
        crossed = err * self.turn_sign < 0 and abs(err) < math.pi / 2
        if (abs(err) <= TURN_TOLERANCE_RAD or crossed
                or now - self.phase_started > TURN_TIMEOUT_S):
            self.drive.stop()
            self._begin_drive(now)
            return None
        rate = min(SEARCH_TURN_RATE, max(MIN_TURN_RATE_PIVOT, TURN_GAIN * abs(err)))
        self.drive.set_velocity(0.0, math.copysign(rate, err))
        return None

    def _drive(self):
        now = time.monotonic()
        o = self.odo
        tx, ty = self.target
        ux, uy = math.cos(self.heading), math.sin(self.heading)
        rx, ry = uy, -ux                          # unit vector to the robot's right
        remaining = (tx - o.x) * ux + (ty - o.y) * uy

        if self.grid and self.forward:
            # Something much closer than the far side of the cell ahead is in the way.
            front = self.sonar.front
            stamp = self.sonar.stamp("front") if hasattr(self.sonar, "stamp") else None
            fresh = stamp is not None and stamp != self._front_stamp
            self._front_stamp = stamp
            if front is not None and front - FRONT_WALL_AT_CENTRE_M < remaining - BLOCK_MARGIN_M:
                if fresh:
                    self.block_count += 1
                if self.block_count >= BLOCK_CONFIRM:
                    STATUS.event(f"[nav] blocked: {front*100:.0f}cm ahead with "
                                 f"{remaining*100:.0f}cm to go -- backing out")
                    self.block_info = (front, remaining)
                    return self._begin_backout(now)
            elif fresh:
                self.block_count = 0

        if remaining <= ARRIVE_SLOW_M and self.near_since is None:
            self.near_since = now       # side walls read from here are this cell's
        if abs(remaining) <= ARRIVE_TOLERANCE_M or (remaining < 0 and self.forward):
            return self._arrive()
        if now - self.phase_started > DRIVE_TIMEOUT_S:
            STATUS.event("[nav] no arrival in time -- treating the cell as blocked")
            self.block_info = None
            return self._begin_backout(now)

        # Off the centreline to the right -> aim a little left of the corridor, and
        # vice versa. The pose already carries what the side walls say.
        self.lateral = (o.x - tx) * rx + (o.y - ty) * ry
        aim = self.heading + _clamp(CENTRE_GAIN * self.lateral, MAX_CENTRE_ANGLE)
        w = _clamp(HEADING_HOLD_GAIN * wrap(aim - o.theta), MAX_STEER_RATE)
        v = SEARCH_SPEED if remaining > ARRIVE_SLOW_M or self.through else ARRIVE_SPEED
        if remaining < 0:
            v = -ARRIVE_SPEED                    # target is behind: back straight up
        self.drive.set_velocity(v, w)
        return None

    def _arrive(self):
        if not self.through:
            self.drive.stop()
        self.phase = self.IDLE
        return "arrived"

    def _begin_backout(self, now):
        self.drive.stop()
        self.phase = self.BACKOUT
        self.phase_started = now
        return None

    def _backout(self):
        o = self.odo
        ux, uy = math.cos(self.heading), math.sin(self.heading)
        ahead = (o.x - self.start[0]) * ux + (o.y - self.start[1]) * uy
        if ahead <= ARRIVE_TOLERANCE_M or time.monotonic() - self.phase_started > DRIVE_TIMEOUT_S:
            self.drive.stop()
            self.phase = self.IDLE
            return "blocked"
        w = _clamp(HEADING_HOLD_GAIN * wrap(self.heading - o.theta), MAX_STEER_RATE)
        self.drive.set_velocity(-REVERSE_SPEED, w)
        return None


def sonar_mounts():
    """Where each sonar sits on the robot: (forward m, left m, direction rad).

    Derived from the calibrated centre readings, taking the walls as sitting on the
    cell boundaries: a sonar reading FRONT_WALL_AT_CENTRE_M to a wall half a cell
    away is mounted half a cell minus that ahead of the centre.
    """
    side = CELL_M / 2 - SIDE_WALL_AT_CENTRE_M
    return {"front": (CELL_M / 2 - FRONT_WALL_AT_CENTRE_M, 0.0, 0.0),
            "left": (SIDE_SONAR_FORWARD_M, side, math.pi / 2),
            "right": (SIDE_SONAR_FORWARD_M, -side, -math.pi / 2)}


class RescueStub:
    """Stands in for Roger's collection mechanism: 'succeeds' after a fixed time.

    Replace start()/done() with the real calls. done() returning True is what moves
    nav on -- to RETURN after collecting, back to exploring after releasing.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.started = None

    def start(self):
        self.started = time.monotonic()

    def done(self):
        return self.started is not None and time.monotonic() - self.started >= self.seconds


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


class NullVision:
    """--no-camera: never sees a victim. For testing the maze nav on its own."""

    geometry_ok = False
    last_frame = None
    last_detections = []
    draw_detections = None

    def look(self):
        return None

    def close(self):
        pass


# ===========================================================================
# Nav loop
# ===========================================================================
SEARCH, SEEK, APPROACH, CLOSING, AT_VICTIM, RETURN, AT_BASE, DONE = (
    "SEARCH", "SEEK", "APPROACH", "CLOSING", "AT_VICTIM", "RETURN", "AT_BASE", "DONE")


class Nav:
    """The mission: explore the maze, approach and collect a victim, carry it home.

        SEARCH     frontier exploration, one cell at a time, mapping as it goes
        SEEK       a victim is confirmed: place it in a cell, drive there along the
                   corridors, face it from the next cell
        CLOSING    the last few cm on the front sonar -- the camera can't see the
                   floor that close
        APPROACH   only when a victim can't be placed (no range): steer at it by eye
        AT_VICTIM  stopped 10 cm short, collecting
        RETURN     flood fill home over the explored map; red LED while carrying
        AT_BASE    releasing, then back to SEARCH for the next victim
        DONE       all victims home, maze exhausted, or time up

    The map survives the whole run, so every trip out resumes from the frontier it
    left, and every trip home takes the shortest route through cells it has seen.
    """

    def __init__(self, drive, vision, leds, sonar=None, base_cell=None,
                 start_heading=None):
        self.drive, self.vision, self.leds = drive, vision, leds
        self.sonar = sonar if sonar is not None else Ultrasonics(enabled=False)
        self.state = SEARCH
        self.hits = 0            # consecutive frames with a victim
        self.misses = 0          # consecutive frames without one
        self.victim = None
        self.arrived_at = None
        self._announced = False
        self.aligned = False        # was the victim well centred when last seen?
        self.closing_since = None
        self.heading_locked = False # committed to a heading; stop steering
        self.ticks = None           # latest cumulative encoder ticks

        # --- the maze ---
        self.base_cell = tuple(base_cell or BASE_CELL)
        start = HEADING_BY_NAME[(start_heading or START_HEADING).upper()]
        self.map = MazeMap(MAZE_COLS, MAZE_ROWS, CELL_M)
        self.odo = Localiser(TICKS_PER_M, EFFECTIVE_TRACK_M, sonar_mounts(),
                             sonar_sigma=SONAR_SIGMA_M, gate_known=GATE_KNOWN_M,
                             gate_maybe=GATE_MAYBE_M)
        self.odo.reset(*self.map.centre(self.base_cell), HEADING_RAD[start])
        self._loc_stamps = {}
        self.mover = Mover(drive, self.sonar, self.odo)
        self.cell = self.base_cell   # cell it's in (or leaving, mid-move)
        self.facing = start          # cardinal heading when last on the grid
        self.move_heading = None     # heading of the move in progress
        self.target_cell = None
        self.route = None            # planned cells, for the map window
        self.off_grid = False        # left the grid to chase a victim
        self.rejoin_stage = None     # "point" then "align" on the way back on
        self.sensing = True          # at a cell centre, reading the walls
        self.sense_from = None       # sensor -> readings newer than this count
        self.carrying = False
        self.victim_cell = None      # where the victim being fetched is
        self.seek_tries = {}         # victim cell -> failed sonar checks
        self.block_tries = {}        # (cell, heading) -> drives that were blocked
        self.closing_heading = None  # heading CLOSING holds
        self.rescued = 0
        self.victim_points = []      # where victims were picked up, for the map
        self.rescue = RescueStub(RESCUE_TIME_S)
        self.release = RescueStub(RELEASE_TIME_S)
        self.mission_start = None

    # -- transitions ------------------------------------------------------
    def _enter(self, state):
        if state == self.state:
            return
        STATUS.event(f"[nav] {self.state} -> {state}")
        self.state = state
        if state in (APPROACH, CLOSING, AT_VICTIM, DONE):
            self._leave_grid()
        if state == AT_VICTIM:
            self.arrived_at = time.monotonic()
            self.drive.stop()
            self.rescue.start()      # TODO: Roger's collect() goes here
        if state == CLOSING:
            self.closing_since = time.monotonic()
            self.closing_from = (self.odo.x, self.odo.y)
            self.closing_range = None
            if self.closing_heading is None:
                self.closing_heading = self.odo.theta   # straight on from here
            # Drop the stale detection: it's behind/underneath us now, and leaving it
            # around would draw a box where the victim no longer is.
            self.victim = None
        if state == AT_BASE:
            self.drive.stop()
            self.release.start()     # TODO: Roger's release() goes here
        if state == DONE:
            self.drive.stop()
        if state in (SEARCH, RETURN):
            self.closing_heading = None
        if state == SEARCH:
            self.victim_cell = None
            self._announced = False
            self.aligned = False
            self.heading_locked = False
            self.victim = None
            self.hits = 0
        if state == APPROACH:
            self.heading_locked = False
        self._set_leds()

    def _localise(self):
        """Correct the pose with every sonar reading that's new since last tick."""
        stamp = getattr(self.sonar, "stamp", None)
        if stamp is None:
            return
        for name in ("front", "left", "right"):
            t = stamp(name)
            if t is None or t == self._loc_stamps.get(name):
                continue
            self._loc_stamps[name] = t
            self.odo.correct(self.map, name, self.sonar.get(name))

    def _set_leds(self):
        """The assessment's colours, decided in one place from the state."""
        green = self.state in (SEEK, APPROACH, CLOSING, AT_VICTIM)
        red = self.state in (RETURN, AT_BASE) and self.carrying
        yellow = self.state in (SEARCH, RETURN) and not red
        self.leds.green(green)
        self.leds.yellow(yellow)
        self.leds.red(red)

    def _leave_grid(self):
        """Stop any grid move; the next grid state starts by getting back on."""
        if self.mover.busy:
            self.mover.stop()
        self.off_grid = True
        self.rejoin_stage = None
        self.route = None

    def elapsed(self):
        return None if self.mission_start is None else time.monotonic() - self.mission_start

    def state_label(self):
        label = STATE_LABEL.get(self.state, self.state)
        if self.state == RETURN and not self.carrying:
            label = "HEADING HOME (maze explored)"
        return label

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
        if self.mission_start is None:
            self.mission_start = time.monotonic()
            self._set_leds()
        self.sonar.update()                    # one ping per tick, before deciding
        reader = getattr(self.drive, "read_encoders", None)
        self.ticks = reader() if reader else None
        self.odo.update(self.ticks)
        self._localise()

        if self.state != DONE and self.elapsed() >= MISSION_TIME_S:
            STATUS.event(f"[nav] {MISSION_TIME_S:.0f}s up -- stopping")
            self._enter(DONE)

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

        confirmed = self.hits >= DETECTION_DEBOUNCE and self._wanted(self.victim)
        lost = self.misses > LOST_GRACE_FRAMES

        if self.state == SEARCH:
            self._search(confirmed)
        elif self.state == SEEK:
            self._seek()
        elif self.state == APPROACH:
            self._approach(lost)
        elif self.state == CLOSING:
            self._closing()
        elif self.state == AT_VICTIM:
            self._at_victim()
        elif self.state == RETURN:
            self._return()
        elif self.state == AT_BASE:
            self._at_base()
        elif self.state == DONE:
            self._done()

    def _wanted(self, v):
        """Is this a victim still to be rescued?

        Not while one is on board, not one lying in the base cell -- that's a victim
        already delivered, and chasing it would 'rescue' it again -- and not one in a
        cell that has already failed the sonar check SEEK_ATTEMPTS times.
        """
        if v is None or self.carrying:
            return False
        delivered = self.rescued > 0          # only then is there one lying in base
        spot = self._victim_spot(v)
        if spot is None:
            here = self.map.cell_at(self.odo.x, self.odo.y)
            return not (delivered and here == self.base_cell)
        cell = self.map.cell_at(*spot)
        if delivered and cell == self.base_cell:
            return False
        return self.seek_tries.get(cell, 0) < SEEK_ATTEMPTS

    def _victim_spot(self, v):
        """World (x, y) of a detection, or None if it can't be ranged."""
        distance = v.distance_m
        if distance is None and abs(v.bearing_deg) <= US_TRUST_BEARING_DEG:
            distance = self.sonar.front
        if distance is None:
            return None
        o = self.odo
        cx = o.x + CAMERA_FORWARD_M * math.cos(o.theta)
        cy = o.y + CAMERA_FORWARD_M * math.sin(o.theta)
        a = o.theta - math.radians(v.bearing_deg)          # bearing is +ve RIGHT
        return cx + distance * math.cos(a), cy + distance * math.sin(a)

    def _search(self, confirmed):
        if confirmed:
            self.leds.green(True)                      # green the moment we commit
            self.leds.yellow(False)
            if not self._announced:
                STATUS.event("[nav] VICTIM DETECTED -- green LED on")
                self._announced = True
            spot = self._victim_spot(self.victim)
            if spot is not None:
                self.victim_cell = self.map.cell_at(*spot)
                self.map.blocked.discard(self.victim_cell)
                STATUS.event(f"[nav] victim placed in cell {self.victim_cell}")
                self._enter(SEEK)
            elif self.can_range:
                self._enter(APPROACH)
            else:
                # Refuse to drive at a victim we cannot range at all. Holding with
                # green lit is the honest behaviour -- fit the front sonar, or
                # measure the camera constants.
                self._leave_grid()
                self.drive.stop()
            return
        self._set_leds()
        self._grid(self._plan_explore)

    # -- the grid ------------------------------------------------------------
    def _grid(self, plan):
        """One tick of cell-to-cell travel. `plan` picks the next heading at each
        cell centre, or returns None having dealt with there being nowhere to go."""
        if self.mover.busy:
            outcome = self.mover.update()
            if outcome == "arrived":
                self._arrived()
            elif outcome == "blocked":
                self._blocked(self.mover.block_info)
            return

        if self.off_grid:
            self._rejoin()
            return

        if self.sensing:
            now = time.monotonic()
            self.drive.stop()
            if self.sense_from is None:
                self.sense_from = {n: now for n in ("front", "left", "right")}
            fitted = [n for n in ("front", "left", "right")
                      if n in getattr(self.sonar, "sensors", {})]
            fresh = all(self.sonar.since(n, self.sense_from[n])[0] for n in fitted)
            if not fresh and now - self.sense_from["front"] < SENSE_TIMEOUT_S:
                return                             # wait for a full set of pings
            self._record_walls(fitted)
            self.sensing = False

        heading = plan()
        if heading is None:
            return
        self.move_heading = heading
        self.target_cell = self.map.neighbour(self.cell, heading)
        # Straight on through a cell it already knows? Then don't stop in it.
        r = self.route
        through = (r is not None and len(r) > 2 and r[1] == self.target_cell
                   and self.target_cell in self.map.visited
                   and self.map.heading_between(r[1], r[2]) == heading)
        self.mover.move_to(*self.map.centre(self.target_cell), HEADING_RAD[heading],
                           through=through)

    def _blocked(self, info):
        """A drive the map said was open wasn't. Work out what was in the way.

        Past the boundary, it's something standing IN the next cell -- rubble, or a
        victim the camera was too close to see -- so the cell is marked, not the
        wall. At the boundary it's a wall the sonar missed: locked in for good.
        """
        self.facing = self.move_heading
        edge = (self.cell, self.move_heading)
        self.block_tries[edge] = self.block_tries.get(edge, 0) + 1
        if self.target_cell in self.map.visited and self.block_tries[edge] < 2:
            # It's been in that cell, so the way is (or was) open -- more likely a
            # bad reading from off-centre. Re-read the walls here and try again.
            STATUS.event(f"[nav] blocked into {self.target_cell}, which it has been "
                         "in -- re-checking")
            self.sensing = True
            self.sense_from = None
            return
        if info is not None:
            front, along = info
            in_cell = front - FRONT_WALL_AT_CENTRE_M > along - CELL_M + 0.03
            if in_cell and self.state == SEEK:
                # Going to a victim and something is standing in the next cell: that's
                # it, closer than the camera can see. Face it and close in.
                STATUS.event(f"[nav] the victim is in {self.target_cell}")
                self.victim_cell = self.target_cell
                return
            if in_cell and self.target_cell not in self.map.visited:
                if self.target_cell != self.victim_cell:
                    self.map.blocked.add(self.target_cell)
                STATUS.event(f"[nav] something in {self.target_cell} -- marked blocked")
                return
        STATUS.event(f"[nav] {self.cell}->{HEADING_NAME[self.move_heading]} "
                     "is walled -- marked on the map")
        self.map.set_wall(self.cell, self.move_heading, True, lock=True)

    def _record_walls(self, fitted):
        """Map the three sides it can see from this cell centre."""
        for name, heading in (("front", self.facing), ("left", LEFT_OF[self.facing]),
                              ("right", RIGHT_OF[self.facing])):
            if name not in fitted:
                continue
            fresh, d = self.sonar.since(name, self.sense_from[name])
            if not fresh:
                continue
            at_centre = FRONT_WALL_AT_CENTRE_M if name == "front" else SIDE_WALL_AT_CENTRE_M
            window = WALL_PRESENT_WINDOW_M
            if self.map.neighbour(self.cell, heading) == self.victim_cell:
                # Towards the victim, a reading that's short but not wall-short is the
                # victim standing just beyond the boundary -- don't seal it off.
                window = VICTIM_WALL_WINDOW_M
            self.map.set_wall(self.cell, heading, d is not None and d < at_centre + window)
        self.map.visited.add(self.cell)

    def _arrived(self):
        if self.rejoin_stage == "point":
            # At the cell centre; now square up to the nearest cardinal heading so
            # the walls can be read.
            self.rejoin_stage = "align"
            self.facing = nearest_heading(self.odo.theta)
            self.mover.move_to(*self.map.centre(self.cell), HEADING_RAD[self.facing])
            return
        now = time.monotonic()
        if self.rejoin_stage == "align":
            self.rejoin_stage = None
            self.off_grid = False
            self.sense_from = None
        else:
            self.cell = self.target_cell
            self.facing = self.move_heading
            # The side walls were read on the way in; the front has to be read from
            # here. A cell already mapped isn't re-read at all.
            near = self.mover.near_since or now
            self.sense_from = {"front": now, "left": near, "right": near}
        self.sensing = self.cell not in self.map.visited

    def _rejoin(self):
        """Back onto the grid after chasing a victim: go to the centre of the cell it's
        in, then face the nearest cardinal heading."""
        self.cell = self.map.cell_at(self.odo.x, self.odo.y)
        cx, cy = self.map.centre(self.cell)
        dx, dy = cx - self.odo.x, cy - self.odo.y
        STATUS.event(f"[nav] rejoining the grid at {self.cell}")
        facing = nearest_heading(self.odo.theta)
        u = HEADING_RAD[facing]
        sideways = abs(dx * math.sin(u) - dy * math.cos(u))
        if math.hypot(dx, dy) < 0.03 or sideways < 0.04:
            # On (or near) the cell's centreline already -- square up and roll
            # straight forward or back onto the centre, no about-turn.
            self.rejoin_stage = "align"
            self.facing = facing
            self.mover.move_to(cx, cy, u)
        else:
            self.rejoin_stage = "point"
            self.mover.move_to(cx, cy, math.atan2(dy, dx), grid=False)

    def _seek(self):
        """Drive along the corridors to the victim's cell, then face it."""
        self._set_leds()
        if self.misses == 0 and self.victim is not None:
            spot = self._victim_spot(self.victim)     # seen again: refine the cell
            if spot is not None:
                self.victim_cell = self.map.cell_at(*spot)
        self._grid(self._plan_victim)

    def _plan_victim(self):
        vc = self.victim_cell
        if self.cell == vc:
            self._start_closing()                     # it's in this very cell
            return None
        h = self.map.heading_between(self.cell, vc)
        if h is not None and self.map.wall(self.cell, h) is not True:
            if self.facing != h:
                # Next door: turn to face it without moving.
                self.move_heading, self.target_cell = h, self.cell
                self.mover.move_to(*self.map.centre(self.cell), HEADING_RAD[h])
                return None
            self._start_closing()
            return None
        heading = self.map.next_heading(self.cell, [vc], True, self.facing,
                                        EXPLORE_PREFER_RIGHT)
        if heading is None:
            STATUS.event(f"[nav] no way to the victim at {vc} -- exploring on")
            self.seek_tries[vc] = SEEK_ATTEMPTS
            self._enter(SEARCH)
            return None
        self.route = self.map.path(self.cell, [vc], True, self.facing, EXPLORE_PREFER_RIGHT)
        return heading

    def _start_closing(self):
        """Facing the victim's cell from the centre of the next one. The sonar must
        agree there's something standing in it before closing in -- a wall, or
        nothing at all, means the victim was placed in the wrong cell."""
        front = self.sonar.front
        at = FRONT_WALL_AT_CENTRE_M
        if self.cell == self.victim_cell:
            ok = front is not None and front < at + CELL_M / 2
        else:
            ok = front is not None and at + VICTIM_WALL_WINDOW_M < front < at + CELL_M + 0.03
        if not ok:
            vc = self.victim_cell
            self.seek_tries[vc] = self.seek_tries.get(vc, 0) + 1
            STATUS.event(f"[nav] nothing in {vc} on the sonar "
                         f"({'no echo' if front is None else f'{front*100:.0f}cm'}) -- "
                         "exploring on")
            self._enter(SEARCH)
            return
        self.closing_heading = HEADING_RAD[self.facing]
        self._enter(CLOSING)

    def _plan_explore(self):
        """Nearest frontier, through walls we haven't seen assumed open."""
        frontier = self.map.frontier_cells()
        heading = None
        if frontier:
            heading = self.map.next_heading(self.cell, frontier, True, self.facing,
                                            EXPLORE_PREFER_RIGHT)
        if heading is None:
            STATUS.event(f"[nav] maze explored ({len(self.map.visited)} cells) -- "
                         "heading home")
            self._enter(RETURN)
            return None
        self.route = self.map.path(self.cell, frontier, True, self.facing,
                                   EXPLORE_PREFER_RIGHT)
        return heading

    def _plan_home(self):
        """Flood fill to base, through edges it has actually seen open. Falls back to
        assuming unseen walls are open if the known map doesn't connect -- e.g. after
        a victim chase left it in a cell it never mapped."""
        if self.cell == self.base_cell:
            self.route = None
            self._enter(AT_BASE if self.carrying else DONE)
            return None
        goal = [self.base_cell]
        for optimistic in (False, True):
            heading = self.map.next_heading(self.cell, goal, optimistic, self.facing,
                                            EXPLORE_PREFER_RIGHT)
            if heading is not None:
                self.route = self.map.path(self.cell, goal, optimistic, self.facing,
                                           EXPLORE_PREFER_RIGHT)
                return heading
        if self.map.blocked:
            # Last resort: something marked blocked is sealing it in. Forget the
            # blocked cells rather than give up -- a wrong one costs a detour, giving
            # up costs the rescue.
            STATUS.event("[nav] no route home -- clearing blocked cells and retrying")
            self.map.blocked.clear()
            return self._plan_home()
        STATUS.event("[nav] no route to base on the map -- stopping")
        self._enter(DONE)
        return None

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
        #
        # Close in, pivot until properly lined up too. The camera loses the victim a
        # few cm further on, and CLOSING only takes over from an ALIGNED approach --
        # rolling forward while still correcting would drop it under the camera
        # unaligned, give up, and go round again. That's what happens when a victim
        # is first seen from the next cell, turning into a junction.
        close = distance < CLOSING_TRIGGER_M + 0.05
        if not self.heading_locked and (abs(v.bearing_deg) > HEADING_COARSE_DEG
                                        or (close and not self.aligned)):
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
            if self.victim_cell is not None:
                self.seek_tries[self.victim_cell] = self.seek_tries.get(self.victim_cell, 0) + 1
            self._enter(SEARCH)
            return

        # Close in, a victim a few cm off the centreline drops out of the sonar's
        # cone, and the sonar reports the wall behind it instead -- which would drive
        # straight over it. So remember the range, and when the reading suddenly jumps
        # further away, finish on the odometry: range then, minus distance since.
        front = self.sonar.front
        travelled = math.hypot(self.odo.x - self.closing_from[0],
                               self.odo.y - self.closing_from[1])
        expected = None if self.closing_range is None else self.closing_range - travelled
        if front is not None and (expected is None or front <= expected + CLOSING_JUMP_M):
            self.closing_range = front + travelled          # still on it
            distance, source = front, "sonar"
        elif expected is not None and self.ticks is not None:
            distance, source = expected, "odometry"         # lost it: dead reckon
        else:
            # No vision and no range is no information at all -- hold rather than
            # drive forward on faith.
            self.drive.stop()
            return

        if distance - STOP_DISTANCE_M <= DISTANCE_TOLERANCE_M:
            STATUS.event(f"[nav] VICTIM REACHED -- stopping at {distance*100:.1f} cm ({source})")
            self._enter(AT_VICTIM)
            return

        # Straight on, held by the encoders -- still no steering on the victim itself.
        w = 0.0
        if self.closing_heading is not None and self.ticks is not None:
            w = _clamp(HEADING_HOLD_GAIN * wrap(self.closing_heading - self.odo.theta),
                       MAX_STEER_RATE)
        self.drive.set_velocity(CREEP_SPEED, w)

    def _at_victim(self):
        """Stopped 10 cm short, green LED on, collecting. Then home with it."""
        self.drive.stop()
        if not self.rescue.done():
            return
        # Remember where it was, for the map: just ahead of where we stopped.
        reach = STOP_DISTANCE_M + CELL_M / 2 - FRONT_WALL_AT_CENTRE_M
        self.victim_points.append((self.odo.x + reach * math.cos(self.odo.theta),
                                   self.odo.y + reach * math.sin(self.odo.theta)))
        self.carrying = True
        if self.victim_cell is not None:
            self.map.blocked.discard(self.victim_cell)   # it's on board now
            # It closed in on the victim through this edge, so it's open -- even if
            # the victim standing just beyond it once read as a wall.
            if self.map.heading_between(self.cell, self.victim_cell) == self.facing:
                self.map.set_wall(self.cell, self.facing, False)
        self.victim_cell = None
        STATUS.event(f"[nav] victim collected after "
                     f"{time.monotonic() - self.arrived_at:.1f}s -- returning to base")
        self._enter(RETURN)

    def _return(self):
        self._set_leds()
        self._grid(self._plan_home)

    def _at_base(self):
        """Home with a victim. Release it, then go back out for the next one."""
        self.drive.stop()
        if not self.release.done():
            return
        self.carrying = False
        self.rescued += 1
        STATUS.event(f"[nav] VICTIM RESCUED -- {self.rescued} home")
        if self.rescued >= VICTIMS_TOTAL:
            self._enter(DONE)
        else:
            self._enter(SEARCH)

    def _done(self):
        """Terminal. Motors off, LEDs off, nothing else changes.

        Deliberately has no way out: the run is over and the robot should sit still
        rather than wander off. Restart main.py to go again.
        """
        self.drive.stop()
        self.leds.green(False)
        self.leds.yellow(False)
        self.leds.red(False)


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
    ap.add_argument("--no-camera", action="store_true",
                    help="no vision at all -- explore and map the maze only")
    ap.add_argument("--no-map", action="store_true", help="don't open the map window")
    ap.add_argument("--base-cell", default=None,
                    help=f"base cell as col,row (default {BASE_CELL[0]},{BASE_CELL[1]})")
    ap.add_argument("--start-heading", choices=["N", "E", "S", "W"], default=None,
                    help=f"which way it faces at the start (default {START_HEADING})")
    ap.add_argument("--calibrate", choices=["straight", "turn", "sonar"], default=None,
                    help="measure TICKS_PER_M, EFFECTIVE_TRACK_M or the sonar geometry")
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

    if args.calibrate:
        calibrate(args.calibrate)
        return

    base_cell = None
    if args.base_cell:
        try:
            base_cell = tuple(int(v) for v in args.base_cell.split(","))
            assert len(base_cell) == 2
        except (ValueError, AssertionError):
            raise SystemExit("--base-cell wants col,row -- e.g. 0,6")

    STATUS.enabled = not args.no_status
    leds = Leds()
    drive = Drive(enabled=not args.no_motors)
    sonar = Ultrasonics(enabled=not args.no_sonar)
    blur = BLUR_THRESHOLD if args.blur_threshold is None else args.blur_threshold
    if args.no_camera:
        vision = NullVision()
        STATUS.event("[vision] --no-camera: exploring and mapping only")
    else:
        if not args.placeholder_vision:
            import vision_system_v2_0 as _vs
            if _vs.BLUR_VARIANCE_THRESHOLD != blur:
                STATUS.event(f"[vision] blur guard {_vs.BLUR_VARIANCE_THRESHOLD:g} -> "
                             f"{blur:g}{' (off)' if blur <= 0 else ''}")
            _vs.BLUR_VARIANCE_THRESHOLD = blur
        vision = VictimVision(use_placeholder=args.placeholder_vision)
        vision.blur_threshold = blur
    nav = Nav(drive, vision, leds, sonar, base_cell=base_cell,
              start_heading=args.start_heading)
    display = Display(enabled=not args.no_display, camera=not args.no_camera,
                      show_map=not args.no_map)
    STATUS.event(f"[nav] {MAZE_COLS}x{MAZE_ROWS} maze of {CELL_M*1000:.0f}mm cells, "
                 f"base {nav.base_cell} facing {HEADING_NAME[nav.facing]}")

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
        save_map(nav)
        print("stopped")


def save_map(nav):
    """Write the final map to Robot_Code/runs/ -- handy for the report."""
    try:
        import cv2
        from map_view import render_map
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("map_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, render_map(nav))
        print(f"[map] saved {path}")
    except Exception as exc:                           # noqa: BLE001
        print(f"[map] not saved -- {type(exc).__name__}: {exc}")


def calibrate(what):
    """Measure the numbers the maze nav depends on. Robot on the floor, in the maze."""
    period = 1.0 / CONTROL_HZ

    if what == "sonar":
        sonar = Ultrasonics()
        print("Put the robot dead centre in a cell with walls ahead, left and right.")
        input("Enter to start... ")
        samples = {n: [] for n in ("front", "left", "right")}
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            sonar.update()
            for n in samples:
                d = sonar.get(n)
                if d is not None:
                    samples[n].append(d)
            time.sleep(period)
        med = {n: (sorted(v)[len(v) // 2] if v else None) for n, v in samples.items()}
        for n, d in med.items():
            print(f"  {n:>5}: {'no reading' if d is None else f'{d*100:.1f} cm'}")
        if med["front"] is not None:
            print(f"\nFRONT_WALL_AT_CENTRE_M = {med['front']:.3f}")
        sides = [d for d in (med["left"], med["right"]) if d is not None]
        if sides:
            print(f"SIDE_WALL_AT_CENTRE_M = {sum(sides) / len(sides):.3f}")
        return

    drive = Drive()
    if drive.driver is None:
        raise SystemExit("no motors -- can't calibrate driving")
    odo = Odometry(TICKS_PER_M, EFFECTIVE_TRACK_M)
    odo.update(drive.read_encoders())
    start = drive.read_encoders()
    try:
        if what == "straight":
            print("Drives straight for 2.5 s. Mark where the robot starts.")
            input("Enter to start... ")
            t_end = time.monotonic() + 2.5
            while time.monotonic() < t_end:
                odo.update(drive.read_encoders())
                w = _clamp(HEADING_HOLD_GAIN * wrap(0.0 - odo.theta), MAX_STEER_RATE)
                drive.set_velocity(SEARCH_SPEED, w)
                time.sleep(period)
            drive.stop()
            time.sleep(0.3)
            end = drive.read_encoders()
            ticks = ((end[0] - start[0]) + (end[1] - start[1])) / 2.0
            print(f"  left {end[0]-start[0]} ticks, right {end[1]-start[1]} ticks")
            cm = float(input("How far did it actually go, in cm? "))
            print(f"\nTICKS_PER_M = {ticks / (cm / 100.0):.0f}")
        else:
            print("Spins one full turn to the LEFT, by the encoders' reckoning. "
                  "Note which way it faces first.")
            input("Enter to start... ")
            turned = 0.0
            last = odo.theta
            t_end = time.monotonic() + 15.0
            while turned < 2 * math.pi and time.monotonic() < t_end:
                odo.update(drive.read_encoders())
                turned += wrap(odo.theta - last)
                last = odo.theta
                drive.set_velocity(0.0, MIN_TURN_RATE_PIVOT)
                time.sleep(period)
            drive.stop()
            print("  the encoders say that was 360 deg")
            deg = float(input("How many degrees did it ACTUALLY turn? (e.g. 330, 380) "))
            print(f"\nEFFECTIVE_TRACK_M = {EFFECTIVE_TRACK_M * 360.0 / deg:.4f}")
    finally:
        drive.stop()
        drive.close()


if __name__ == "__main__":
    main()
