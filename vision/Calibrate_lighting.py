"""
One-stop lighting calibration: markers + every colour object.

Walks through a series of stages, one window each, using the same live
camera:

    Stage 1      MARKERS   black-ink threshold (V max, S max)
                           -> saved to marker_threshold.json (ink + placard-white)
    Stage 2..N   OBJECTS   one stage per colour class in profiles.pkl
                           (ramp, obstacle, victim, rubble, ...)
                           -> hue/sat/val band re-centred, saved to profiles.pkl

Every stage has the same layout:
    Left half  = live camera view
    Right half = the mask (WHITE = detected, BLACK = ignored)

Drag the sliders until the thing you're calibrating is solid WHITE on the
right and everything else is BLACK. Aim the camera at the item for that
stage (the marker, then each coloured object in turn), with some of the
carpet/background in view too.

TIP: left-click any pixel to print its H, S, V in the terminal.
Window too big/small? Change DISPLAY_SCALE near the top of this file.

Keys (click the window first so it has focus):
    t / ENTER / SPACE = confirm this stage and go to the next
    q                 = skip this stage (leaves it unchanged)
    x                 = finish now (saves whatever is already confirmed)

--- How the OBJECT stages work ---
They do NOT rebuild anything from reference photos (that's
calibrate_v2_1_1.py). They only slide each class's existing colour band so
it's centred on today's lighting. The band's WIDTH (the tolerance learned
from your negative images) is preserved exactly, and the ORB descriptors,
shape contours and thresholds are left untouched. So: run calibrate_v2_1_1.py
once to build profiles.pkl, then run THIS at the venue / whenever lighting
changes.

--- Red / wrap-around hues (v2 fix) ---
Hue is a CIRCLE (0..179, and 179 is right next to 0), and red sits on the
seam. profiles.pkl stores a red band as e.g. low=170, high=8 ("start at 170,
go up through 179, wrap round to 0, stop at 8"). The first version of this
tool took the middle of that as (170+8)/2 = 89 (green/cyan) and the width
as (8-170)/2 = -81, so the mask it built was empty at EVERY slider position
and then, if you confirmed, it saved a broken band. Hue maths now goes
round the circle (see _band_centre_halfwidth / _shift_hue_band).

Safety: profiles.pkl is backed up (profiles_backup_<time>.pkl) before it is
overwritten, and only if something actually changed.

Run from ~/vision:
    python3 calibrate_lighting_all.py
"""

import cv2
import json
import os
import pickle
import shutil
import time
import numpy as np
from class_profile_v2_0 import ClassProfile  # noqa: F401 -- required so pickle can unpickle
from camera_capture_v2_0 import CameraCapture

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES_FILE = os.path.join(HERE, "profiles.pkl")
MARKER_FILE = os.path.join(HERE, "marker_threshold.json")

DEFAULT_V_MAX = 80
DEFAULT_S_MAX = 90
DEFAULT_W_MIN = 160     # "placard white" floor (0 = OFF); grey floor should sit BELOW this

# How big the preview window is, as a fraction of the camera frame. The
# preview shows two views side by side, so at 1.0 it is 1280 px wide and the
# sliders get pushed off a small screen. Lower = smaller window (0.5 = very
# small, 0.6 = default, 0.8 = larger). Detection always runs at full
# resolution -- this only changes what is DRAWN.
DISPLAY_SCALE = 0.6

# A shift bigger than this is just worth a second look (mirrors
# recalibrate_lighting.py / the original calibration tolerances)
LARGE_HUE_DELTA_NOTE = 12
LARGE_SV_DELTA_NOTE = 60

# Return values from a stage
FINISH_NOW = "finish"


# ----------------------------------------------------------------------
# Mask helpers
# ----------------------------------------------------------------------
def band_mask(hsv, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
    """Hue-wrap-aware colour mask.

    NOTE: this is the CORRECT wrap handling. A band like hue -10..14 (red,
    straddling the 0/180 seam) means hues 170..179 OR 0..14. The older
    _colour_mask copies in vision_system_v2_0.py / colour_detector.py /
    object_detector.py build the second range as [max(low,0)%180 .. 179],
    which for a wrapped band ends up covering EVERY hue. Patch those too, or
    what you see in this tool won't match what the runtime detects."""
    sat_low, sat_high = max(0, sat_low), min(255, sat_high)
    val_low, val_high = max(0, val_low), min(255, val_high)
    wraps = hue_low < 0 or hue_high > 179

    if not wraps:
        return cv2.inRange(hsv,
                           (max(0, int(round(hue_low))), sat_low, val_low),
                           (min(179, int(round(hue_high))), sat_high, val_high))

    lo = int(round(hue_low)) % 180
    hi = int(round(hue_high)) % 180
    upper_part = cv2.inRange(hsv, (lo, sat_low, val_low), (179, sat_high, val_high))
    lower_part = cv2.inRange(hsv, (0, sat_low, val_low), (hi, sat_high, val_high))
    return upper_part | lower_part


def _hue_delta(live_h, original_mean_h):
    """Shortest signed distance from original_mean_h to live_h on the 0-180
    hue wheel, e.g. 179 -> 1 is a delta of +2, not -178."""
    return ((live_h - original_mean_h + 90) % 180) - 90


def _band_centre_halfwidth(low, high):
    """Centre and half-width of a hue band, measured round the 0-179 circle.

    A band can be stored two ways and both must work:
        38..62     normal               -> centre 50,  half-width 12
        170..8     wraps through 0      -> centre 179, half-width 9
        -10..14    negative low (older) -> centre 2,   half-width 12
    The width is "how far you walk, going UP from low, to reach high" --
    the % 180 is what makes 170 -> 8 come out as 18 instead of -162."""
    width = (high - low) % 180
    return (low + width / 2) % 180, width / 2


def _shift_hue_band(low, high, delta):
    """Slides a hue band round the circle by delta, keeping its width.
    Returns (new_low, new_high, wraps) with both ends kept in 0..179, so a
    band that crosses the seam is stored as low > high and wraps=True --
    the form colour_detector reads."""
    width = (high - low) % 180
    new_low = (low + delta) % 180
    new_high = (new_low + width) % 180
    return new_low, new_high, new_low > new_high


def _shift_band_preserving_width(low, high, delta, floor=0, ceiling=255):
    """Shifts [low, high] by delta, then -- if that pushes a bound out of
    [floor, ceiling] -- slides the WHOLE band back in range instead of
    clamping each bound independently (which would silently change the
    band's width)."""
    width = high - low
    new_low, new_high = low + delta, high + delta
    if new_low < floor:
        new_low, new_high = floor, floor + width
    if new_high > ceiling:
        new_high, new_low = ceiling, ceiling - width
    return new_low, new_high


# ----------------------------------------------------------------------
# Shared window plumbing
# ----------------------------------------------------------------------
class ClickProbe:
    """Remembers the last left-click so the loop can print/annotate its HSV."""
    def __init__(self):
        self.pt = None

    def __call__(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pt = (x, y)


def annotate_click(view, hsv, probe, small_w):
    """Prints the HSV under the last click and marks it on the (already
    shrunk) view. Clicks are in display coordinates, so they're converted
    back to full-resolution frame coordinates before reading the HSV."""
    if probe.pt is None:
        return
    dx, dy = probe.pt
    probe.pt = None  # print once per click
    if dx >= small_w:
        dx -= small_w          # a click on the mask half maps to the same pixel
    fx, fy = int(dx / DISPLAY_SCALE), int(dy / DISPLAY_SCALE)
    if 0 <= fy < hsv.shape[0] and 0 <= fx < hsv.shape[1]:
        h, s, v = (int(c) for c in hsv[fy, fx])
        print(f"   clicked ({fx},{fy}) -> H={h} S={s} V={v}")
        cv2.circle(view, (dx, dy), 5, (0, 255, 0), 2)
        cv2.putText(view, f"H={h} S={s} V={v}", (dx + 8, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


def build_preview(frame, mask, hsv, probe, text):
    """Side-by-side live view + mask, shrunk by DISPLAY_SCALE."""
    small_w = max(1, int(frame.shape[1] * DISPLAY_SCALE))
    small_h = max(1, int(frame.shape[0] * DISPLAY_SCALE))
    view = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mask_bgr = mask if mask.ndim == 3 else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_view = cv2.resize(mask_bgr, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    annotate_click(view, hsv, probe, small_w)
    preview = np.hstack([view, mask_view])
    cv2.putText(preview, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return preview


def key_action(key):
    """Maps a keypress to 'confirm', 'skip', 'finish', or None."""
    if key in (13, 32, ord("t")):
        return "confirm"
    if key == ord("q"):
        return "skip"
    if key == ord("x"):
        return "finish"
    return None


# ----------------------------------------------------------------------
# Stage 1: markers
# ----------------------------------------------------------------------
def load_marker_values():
    try:
        with open(MARKER_FILE) as f:
            d = json.load(f)
        return int(d["v_max"]), int(d["s_max"]), int(d.get("w_min", DEFAULT_W_MIN))
    except Exception:
        return DEFAULT_V_MAX, DEFAULT_S_MAX, DEFAULT_W_MIN


def marker_stage(camera, label):
    """Returns (v_max, s_max, w_min) if confirmed, None if skipped, or
    FINISH_NOW.

    The right half is colour-coded:
        WHITE = counted as marker INK       (dark:  V <= V max, S <= S max)
        GREEN = counted as white PLACARD    (light: V >= W min, S <= S max)
        BLACK = ignored (the grey floor should be here)
    Correct setup: the black symbols show WHITE, sitting on a GREEN placard,
    and the grey floor is BLACK. A detection needs both: ink ON a placard."""
    v0, s0, w0 = load_marker_values()
    window = f"{label}: MARKERS (t=confirm, q=skip, x=finish)"
    probe = ClickProbe()
    cv2.namedWindow(window)
    cv2.moveWindow(window, 0, 0)   # top-left, so the sliders stay on screen
    cv2.setMouseCallback(window, probe)
    cv2.createTrackbar("V max", window, v0, 255, lambda v: None)
    cv2.createTrackbar("S max", window, s0, 255, lambda v: None)
    cv2.createTrackbar("W min (0=off)", window, w0, 255, lambda v: None)

    print(f"\n--- {label}: MARKERS ---")
    print("Aim at a marker with some grey floor in view. You want:")
    print("   black symbol -> WHITE,  white placard -> GREEN,  grey floor -> BLACK")
    print("  Click the black ink, the grey floor and the white placard to read their V values.")
    print("  V max : just ABOVE the ink's V (and well BELOW the floor's V).")
    print("  W min : just BELOW the placard's V (and well ABOVE the floor's V).")
    print("  The floor must be black in the mask -- if it shows white or green, move the")
    print("  sliders until it doesn't.")

    result = None
    last_fracs = (0.0, 0.0)
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("[error] could not read a frame from the camera")
                return None
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            v_max = cv2.getTrackbarPos("V max", window)
            s_max = cv2.getTrackbarPos("S max", window)
            w_min = cv2.getTrackbarPos("W min (0=off)", window)

            ink = cv2.inRange(hsv, (0, 0, 0), (179, s_max, v_max))
            placard = cv2.inRange(hsv, (0, 0, w_min), (179, s_max, 255))
            combined = np.zeros(frame.shape, np.uint8)             # BGR
            if w_min > 0:
                combined[placard > 0] = (0, 110, 0)                # dark green
            combined[ink > 0] = (255, 255, 255)                    # white (wins if both)
            total = float(ink.size)
            last_fracs = (cv2.countNonZero(ink) / total,
                          cv2.countNonZero(placard) / total if w_min > 0 else 0.0)

            preview = build_preview(frame, combined, hsv, probe,
                                    f"MARKERS  ink V<={v_max}  S<={s_max}  placard V>={w_min}")
            cv2.imshow(window, preview)

            action = key_action(cv2.waitKey(20) & 0xFF)
            if action == "confirm":
                result = (v_max, s_max, w_min)
                break
            if action == "skip":
                break
            if action == "finish":
                return FINISH_NOW
    finally:
        cv2.destroyWindow(window)

    # Sanity warnings -- the usual signs the sliders are catching the floor
    ink_frac, placard_frac = last_fracs
    if result is not None:
        if ink_frac > 0.15:
            print(f"  [warning] {ink_frac:.0%} of the frame counts as INK -- the grey floor or "
                  f"other dark areas are probably included. Lower V max and rerun.")
        if placard_frac > 0.5:
            print(f"  [warning] {placard_frac:.0%} of the frame counts as PLACARD -- the floor is "
                  f"probably included. Raise W min and rerun.")
        if result[2] == 0:
            print("  [note] W min is 0, so the white-placard check is OFF.")
    return result


# ----------------------------------------------------------------------
# Stages 2..N: colour objects
# ----------------------------------------------------------------------
def object_stage(camera, label, class_name, profile):
    """Returns (delta_h, delta_s, delta_v) if confirmed, None if skipped,
    or FINISH_NOW."""
    # Hue is circular, so its centre / half-width come from the circle helper
    # (a plain (low+high)/2 is wrong for red, stored as e.g. 170..8).
    orig_mean_h, tol_h = _band_centre_halfwidth(profile.hue_low, profile.hue_high)
    orig_mean_s = (profile.sat_low + profile.sat_high) / 2
    orig_mean_v = (profile.val_low + profile.val_high) / 2
    # half-width of the ORIGINAL calibrated band -- preserved throughout
    tol_s = (profile.sat_high - profile.sat_low) / 2
    tol_v = (profile.val_high - profile.val_low) / 2

    window = f"{label}: {class_name.upper()} (t=confirm, q=skip, x=finish)"
    probe = ClickProbe()
    cv2.namedWindow(window)
    cv2.moveWindow(window, 0, 0)   # top-left, so the sliders stay on screen
    cv2.setMouseCallback(window, probe)
    cv2.createTrackbar("Hue", window, int(round(orig_mean_h)) % 180, 179, lambda v: None)
    cv2.createTrackbar("Sat", window, int(min(255, max(0, round(orig_mean_s)))), 255, lambda v: None)
    cv2.createTrackbar("Val", window, int(min(255, max(0, round(orig_mean_v)))), 255, lambda v: None)

    print(f"\n--- {label}: {class_name} ---")
    print(f"Hold the {class_name} in view. Drag Hue/Sat/Val until it is solid WHITE in the "
          f"mask and everything else is BLACK. (Band width is preserved; only the centre moves.)")

    result = None
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("[error] could not read a frame from the camera")
                return None
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h = cv2.getTrackbarPos("Hue", window)
            s = cv2.getTrackbarPos("Sat", window)
            v = cv2.getTrackbarPos("Val", window)

            mask = band_mask(hsv, h - tol_h, h + tol_h,
                             s - tol_s, s + tol_s, v - tol_v, v + tol_v)

            preview = build_preview(frame, mask, hsv, probe,
                                    f"{class_name}: H={h} S={s} V={v}")
            cv2.imshow(window, preview)

            action = key_action(cv2.waitKey(20) & 0xFF)
            if action == "confirm":
                result = (h, s, v)
                break
            if action == "skip":
                break
            if action == "finish":
                return FINISH_NOW
    finally:
        cv2.destroyWindow(window)

    if result is None:
        return None
    h, s, v = result
    return _hue_delta(h, orig_mean_h), s - orig_mean_s, v - orig_mean_v


# ----------------------------------------------------------------------
def main():
    profiles = None
    if os.path.isfile(PROFILES_FILE):
        with open(PROFILES_FILE, "rb") as f:
            profiles = pickle.load(f)
    else:
        print(f"[note] no profiles.pkl at {PROFILES_FILE} -- only markers will be "
              f"calibrated. Run calibrate_v2_1_1.py first for the colour objects.")

    object_names = []
    if profiles:
        for name, p in profiles.items():
            if p.hue_low is None:
                print(f"  skipping '{name}' -- never had a colour band calibrated")
            else:
                object_names.append(name)

    total = 1 + len(object_names)
    camera = CameraCapture()
    profiles_changed = False
    flagged = []
    summary = []

    try:
        # ---- stage 1: markers ----
        res = marker_stage(camera, f"Stage 1/{total}")
        if res == FINISH_NOW:
            summary.append("finished early at the marker stage")
            raise StopIteration
        if res is None:
            summary.append("markers: skipped (unchanged)")
        else:
            v_max, s_max, w_min = res
            with open(MARKER_FILE, "w") as f:
                json.dump({"v_max": v_max, "s_max": s_max, "w_min": w_min}, f)
            summary.append(f"markers: ink V<={v_max}, S<={s_max}, placard V>={w_min} saved")
            print(f"  saved marker thresholds: ink V<={v_max}, S<={s_max}, placard V>={w_min}")

        # ---- stages 2..N: objects ----
        for i, name in enumerate(object_names, start=2):
            profile = profiles[name]
            res = object_stage(camera, f"Stage {i}/{total}", name, profile)
            if res == FINISH_NOW:
                summary.append("finished early")
                break
            if res is None:
                summary.append(f"{name}: skipped (unchanged)")
                continue

            delta_h, delta_s, delta_v = res
            profile.hue_low, profile.hue_high, profile.hue_wraps = _shift_hue_band(
                profile.hue_low, profile.hue_high, delta_h)
            profile.sat_low, profile.sat_high = _shift_band_preserving_width(
                profile.sat_low, profile.sat_high, delta_s)
            profile.val_low, profile.val_high = _shift_band_preserving_width(
                profile.val_low, profile.val_high, delta_v)
            profiles_changed = True

            print(f"  '{name}': delta H/S/V={delta_h:+.0f}/{delta_s:+.0f}/{delta_v:+.0f}  "
                  f"new band hue={profile.hue_low:.0f}..{profile.hue_high:.0f} "
                  f"sat={profile.sat_low:.0f}..{profile.sat_high:.0f} "
                  f"val={profile.val_low:.0f}..{profile.val_high:.0f}")
            summary.append(f"{name}: shifted H{delta_h:+.0f} S{delta_s:+.0f} V{delta_v:+.0f}")

            if (abs(delta_h) > LARGE_HUE_DELTA_NOTE or abs(delta_s) > LARGE_SV_DELTA_NOTE
                    or abs(delta_v) > LARGE_SV_DELTA_NOTE):
                print(f"  [note] large shift from the original calibration -- "
                      f"double-check '{name}' detects correctly afterwards")
                flagged.append(name)
    except StopIteration:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if profiles_changed:
            backup = os.path.join(HERE, f"profiles_backup_{int(time.time())}.pkl")
            shutil.copy(PROFILES_FILE, backup)
            with open(PROFILES_FILE, "wb") as f:
                pickle.dump(profiles, f)
            print(f"\nSaved updated colour bands to {PROFILES_FILE}")
            print(f"(original preserved at {backup} -- restore it if this makes things worse)")

    print("\nSummary:")
    for line in summary:
        print(f"  - {line}")
    if flagged:
        print(f"[note] large shifts for: {', '.join(flagged)} -- worth double-checking these live")


if __name__ == "__main__":
    main()