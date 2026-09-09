"""Placeholder VisionInterface: always 'no detections'."""

import time

from ..interfaces.messages import DetectionFrame


class StubVision:
    def start(self) -> None: pass
    def stop(self) -> None: pass
    def is_healthy(self) -> bool: return True
    def latest(self):
        return DetectionFrame(time.monotonic(), [])
