"""Vision (Kushal) -> Navigation (Jake).

Nav never sees images, masks or bounding boxes. It receives a short structured
message: class + range (m) + bearing (rad), at >= 10 Hz. That is exactly the
sim's GetDetections() contract, mirrored for the real camera.
"""

from typing import Protocol, Optional

from .messages import DetectionFrame


class VisionInterface(Protocol):
    def latest(self) -> Optional[DetectionFrame]:
        """Most recent frame's detections, or None if no frame has arrived yet.
        Non-blocking: vision runs its own loop (thread/process) and nav polls this."""

    def start(self) -> None:
        """Warm the camera / start the detection loop."""

    def stop(self) -> None: ...

    def is_healthy(self) -> bool:
        """False if frames have gone stale (e.g. > 0.5 s old) — nav treats stale as 'no detections'."""
