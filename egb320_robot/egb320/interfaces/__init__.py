"""Interface contracts between subsystems.

These are the ARROWS on the system architecture diagram, written as code so they
cannot drift from the diagram. Each Protocol is what Navigation *calls*; each
dataclass is what travels along an arrow. Every field carries units + frame.

    Vision   -> Nav : DetectionFrame          (>= 10 Hz, streamed)
    Mobility -> Nav : EncoderSample / WallRanges (continuous)
    Nav -> Mobility : VelocityCommand         (continuous while driving)
    Nav -> Rescue   : collect / release / clear_rubble  (on state entry)
    Rescue -> Nav   : RescueStatus            (busy / done / failed + has_victim)
    Nav -> LEDs     : LedState                (on every state change)
"""

from .messages import (Detection, DetectionFrame, EncoderSample, LedState,
                       MarkerClass, Pose2D, RescueOutcome, RescueStatus,
                       VelocityCommand, WallRanges)
from .mobility import MobilityInterface
from .vision import VisionInterface
from .rescue import RescueInterface
from .leds import LedInterface

__all__ = [
    "Detection", "DetectionFrame", "EncoderSample", "LedState", "MarkerClass",
    "Pose2D", "RescueOutcome", "RescueStatus", "VelocityCommand", "WallRanges",
    "MobilityInterface", "VisionInterface", "RescueInterface", "LedInterface",
]
