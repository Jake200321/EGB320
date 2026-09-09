"""
====================================================================
Global Vision Configuration
====================================================================
Only holds pipeline-wide constants that are NOT tied to lighting/camera
conditions -- i.e. deliberate strictness settings, not measurements.
Anything that depends on what the camera actually sees (HSV colour bands,
ORB descriptors, reference contours) lives in profiles.pkl instead,
produced by calibrate_v2_1_1.py and loaded per-class into a ClassProfile.
Recalibrating (or re-running recalibrate_lighting.py after a lighting
change) updates profiles.pkl -- nothing here needs to be hand-edited when
that happens.
"""

# Minimum contour area to ignore noise (in pixels)
MIN_CONTOUR_AREA = 500

# Minimum confidence threshold for marker template matching
MARKER_THRESHOLD = 0.5

# Max cv2.matchShapes distance to accept a colour blob as the right object.
# matchShapes returns 0 for identical shapes and grows the less alike they
# are, so this is a ceiling, not a floor -- a contour scoring ABOVE this
# gets rejected outright, not just given lower confidence.
SHAPE_MAX_DIST = 0.2

{
  "panel_width_cm": 27.0,
  "post_width_cm": 1.0,
  "gap_width_cm": 1.5,
  "min_opening_width_cm": 15.0,
  "camera_height_cm": 4.5,
  "camera_pitch_deg": -11.0,
  "camera_hfov_deg": 62.2,
  "wall_luminance_min": 160,
  "wall_saturation_max": 65,
  "texture_kernel_size": 3,
  "texture_carpet_min": 18,
  "morph_kernel_size": 5,
  "column_step": 2,
  "ransac_tolerance_px": 8,
  "seam_max_width_cm": 4.0,
  "opening_min_depth_diff_px": 25
}