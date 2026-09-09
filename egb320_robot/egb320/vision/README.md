# Vision (Kushal) — what navigation needs from this folder

Put your cv2 pipeline here. Nav polls ONE method and needs it fast and non-blocking:

```python
frame = vision.latest()          # -> DetectionFrame | None
for d in frame.detections:       # Detection(cls, range_m, bearing_rad, confidence)
    ...
```

| Field | Units / frame | Notes |
|---|---|---|
| `cls` | `MarkerClass` enum | BASE, VICTIM (L1+L2 placard), TRAPPED_VICTIM (L3), HAZARD, VICTIM_OBJECT (the token) |
| `range_m` | metres | `range = H_real * f / h_px` (known marker height, calibrated focal length) |
| `bearing_rad` | radians, relative to camera centreline, **positive = LEFT** | `bearing = ((cx - W/2) / W) * FOV` — **confirm the sign convention with Jake** |
| `confidence` | 0..1 | optional |

Requirements from the brief: ≥ 10 Hz on the Pi, labelled detections overlaid on the live stream.
Run your loop in its own thread/process; `latest()` just returns the newest frame. An empty
detection list is a normal answer. Mark stale frames via `is_healthy()` so nav ignores them.

Camera: Pi Camera Module 3 (confirm 75° vs 120° FOV variant — it changes the bearing formula).
Markers sit ≈120 mm above the maze floor: camera height/tilt is a joint nav + vision + mech decision.
`stub.py` returns empty frames so the stack runs before your code lands.
