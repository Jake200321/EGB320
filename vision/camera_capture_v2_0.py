"""
Camera capture wrapper.

picamera2 (the standard Raspberry Pi camera library) returns frames as
NumPy arrays in RGB channel order. OpenCV's entire colour pipeline --
cv2.cvtColor, cv2.inRange, the HSV masking this project relies on -- assumes
BGR order. Feed a raw picamera2 frame straight into that pipeline and every
hue calibrated in calibrate.py will be wrong: reds and blues swap, greens
shift, even though nothing else in the code is broken.

This file does the RGB -> BGR conversion once, in one place, so nothing
downstream (vision_system.py) has to know or care which camera is attached.

Falls back to a normal USB webcam via cv2.VideoCapture if picamera2 isn't
installed, so the vision pipeline can still be developed/tested on a laptop
before being deployed to the Pi.
"""

import cv2
import math

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# ---------------- Constants ----------------
FRAME_SIZE = (640, 480)
USB_CAMERA_INDEX = 0

# Camera's HORIZONTAL field of view -- update this if the lens/camera ever
# changes. Everything that computes a bearing angle to a detected object
# depends on this being accurate; a wrong FOV here silently miscalibrates
# every angle downstream even though the pixel math is otherwise correct.
HORIZONTAL_FOV_DEG = 45.0


def pixel_x_to_angle(x, frame_width, fov_deg=HORIZONTAL_FOV_DEG):
    """Converts a pixel's horizontal position into a bearing angle in
    degrees: 0 = straight ahead, negative = left of centre, positive =
    right of centre. Uses a pinhole-camera model (see the module-level
    reasoning above this function) rather than assuming pixel offset
    scales linearly with angle, which breaks down at wider FOVs."""
    half_width = frame_width / 2
    half_fov_rad = math.radians(fov_deg / 2)
    normalised_offset = (x - half_width) / half_width  # -1 .. +1 across the frame
    return math.degrees(math.atan(normalised_offset * math.tan(half_fov_rad)))
# --------------------------------------------

# On some Pi camera / picamera2 builds, requesting "RGB888" actually returns
# frames that are ALREADY in BGR order (a known library quirk) -- so applying
# an RGB->BGR conversion on top of that swaps the channels a second time and
# produces the wrong colours again. Confirmed by testing: hold a known pure
# colour in front of the camera; if it displays as the "opposite" colour
# (red<->blue swapped, yellow<->cyan swapped), the frames were already BGR
# and this should be True. If colours display correctly with no conversion
# at all, set this to False.
PICAMERA_FRAME_ALREADY_BGR = False

# ---- Motion-blur control -------------------------------------------------
# A moving camera smears fine detail (marker symbol edges, ORB keypoints)
# across pixels during the exposure window -- the fix is a SHORT exposure,
# not a sharper lens. But a shorter exposure gathers less light, so the
# image goes dark unless something else compensates -- AnalogueGain is
# raised to make up for it below. This trades image NOISE for sharpness,
# which is the right trade for detection (template matching and ORB
# tolerate noise far better than they tolerate blur).
#
# Start here and adjust by testing while actually moving the camera at
# your real driving speed -- too short and frames get too dark/noisy even
# after gain compensation; too long and blur creeps back in.
MAX_EXPOSURE_TIME_US = 8000       # 8ms ceiling on exposure time
MAX_ANALOGUE_GAIN = 8.0           # ceiling on how far gain compensates --
                                   # sensor noise gets ugly well before most
                                   # gain ranges max out, so this is a floor
                                   # under image quality, not just a number

# Frame duration limits (microseconds), min and max both set to the same
# value below -- this is what actually pins the frame rate near a target,
# rather than just requesting FRAME_SIZE and hoping the sensor picks a fast
# enough rate on its own.
TARGET_FPS = 60
FRAME_DURATION_LIMIT_US = int(1_000_000 / TARGET_FPS)

# IMPORTANT: changing exposure/gain changes overall brightness, which
# shifts the Value channel of everything the colour detector sees. Any
# time these constants change, re-run recalibrate_lighting.py (or a full
# recalibrate) afterward -- the HSV bands calibrated at the OLD exposure
# will be wrong at the new one.
# --------------------------------------------


class CameraCapture:
    def __init__(self, use_picamera=None, frame_size=FRAME_SIZE):
        # auto-detect unless explicitly told which camera to use
        self.use_picamera = PICAMERA_AVAILABLE if use_picamera is None else use_picamera

        if self.use_picamera:
            self._picam = Picamera2()
            # requesting RGB888 explicitly -- picamera2's capture_array() then
            # hands back an (H, W, 3) array in R, G, B channel order
            config = self._picam.create_preview_configuration(
                main={"format": "RGB888", "size": frame_size}
            )
            self._picam.configure(config)
            self._picam.start()

            # Lock auto white balance and exposure once, right after start,
            # instead of leaving them on "auto". If AWB is left free-running,
            # the camera's colour cast can drift between the session you
            # calibrated in and a later session (different lighting, or even
            # just re-settling on startup) -- silently invalidating every
            # HSV band calibrate.py learned. Locking it makes whatever colour
            # cast exists repeatable, which is what the colour layer actually
            # depends on -- not realistic-looking colour, just repeatable colour.
            import time
            time.sleep(1)  # let AWB/AEC settle on the current scene first
            settled = self._picam.capture_metadata()

            # Clamp exposure DOWN to the motion-blur ceiling, then raise
            # gain to compensate for the light that shorter exposure no
            # longer gathers -- keeps roughly the same overall brightness
            # (and so the same HSV Value range) while cutting how much the
            # frame smears during camera movement.
            settled_exposure = settled["ExposureTime"]
            exposure_time = min(settled_exposure, MAX_EXPOSURE_TIME_US)
            if exposure_time < settled_exposure:
                gain_scale = settled_exposure / exposure_time
                analogue_gain = min(settled["AnalogueGain"] * gain_scale, MAX_ANALOGUE_GAIN)
            else:
                analogue_gain = settled["AnalogueGain"]

            self._picam.set_controls({
                "AwbEnable": False,
                "ColourGains": settled["ColourGains"],
                "AeEnable": False,
                "ExposureTime": exposure_time,
                "AnalogueGain": analogue_gain,
                "FrameDurationLimits": (FRAME_DURATION_LIMIT_US, FRAME_DURATION_LIMIT_US),
            })
            print(f"[camera] using picamera2 -- PICAMERA_FRAME_ALREADY_BGR = {PICAMERA_FRAME_ALREADY_BGR}, "
                  f"exposure={exposure_time}us (was {settled_exposure}us), "
                  f"gain={analogue_gain:.2f} (was {settled['AnalogueGain']:.2f}), "
                  f"target {TARGET_FPS}fps -- re-run recalibrate_lighting.py after "
                  f"changing MAX_EXPOSURE_TIME_US")
        else:
            self._cap = cv2.VideoCapture(USB_CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
            self._cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            # Best-effort only -- UVC webcam exposure control via
            # cv2.VideoCapture is inconsistent across hardware/drivers,
            # unlike picamera2's reliable ExposureTime/AnalogueGain above.
            # 0.25 here means "manual mode" on most UVC-standard cameras,
            # but some ignore it entirely -- check with a printed frame
            # timestamp/blur-variance test on YOUR specific webcam rather
            # than trusting this blindly.
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # roughly "short", scale is driver-specific
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open USB camera index {USB_CAMERA_INDEX}")
            print("[camera] using USB webcam via cv2.VideoCapture -- exposure control is "
                  "best-effort here, verify it actually took effect on your hardware")

    def read(self):
        """Returns one frame in BGR order (what OpenCV expects), or None on failure."""
        if self.use_picamera:
            frame = self._picam.capture_array()
            if PICAMERA_FRAME_ALREADY_BGR:
                return frame  # already correct order -- converting again would swap it back to wrong
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ok, frame_bgr = self._cap.read()  # cv2.VideoCapture already returns BGR
            return frame_bgr if ok else None

    def release(self):
        if self.use_picamera:
            self._picam.stop()
        else:
            self._cap.release()