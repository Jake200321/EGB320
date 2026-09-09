"""The messages that travel between subsystems. Units and frames are part of the contract.

Frame conventions (agree these with the team — flagged in the 2026-08-18 learning log):
  * Robot frame: x forward, y LEFT, heading counter-clockwise positive (matches the sim:
    positive turn velocity turns left).
  * Bearing: radians, relative to the camera centreline, POSITIVE = LEFT.  <-- CONFIRM with Kushal
  * Range: metres, from the camera to the object.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class MarkerClass(Enum):
    """The five things vision can report. Mirrors mazebot_lib.GetDetections() keys."""
    BASE = auto()            # base-zone placard
    VICTIM = auto()          # exposed-victim placard (L1 and L2 share this marker)
    TRAPPED_VICTIM = auto()  # trapped-victim placard (L3, behind rubble)
    HAZARD = auto()          # dead end — never enter
    VICTIM_OBJECT = auto()   # the physical victim token itself (yellow in the sim)


@dataclass(frozen=True)
class Detection:
    cls: MarkerClass
    range_m: float           # metres
    bearing_rad: float       # radians, +ve left of camera centreline
    confidence: float = 1.0  # 0..1, optional — vision may leave at 1.0


@dataclass
class DetectionFrame:
    """One camera frame's worth of detections. An EMPTY list is a normal answer, not an error."""
    t: float                             # seconds, vision's clock at capture
    detections: list = field(default_factory=list)   # list[Detection]

    def of(self, cls: MarkerClass):
        return [d for d in self.detections if d.cls is cls]


@dataclass
class EncoderSample:
    t: float
    left_ticks: int          # signed, cumulative since reset
    right_ticks: int


@dataclass
class WallRanges:
    """Distance to nearest surface per sensor (metres) or None if nothing in range.
    Names are relative to the ROBOT, not the maze."""
    t: float
    left: Optional[float]
    front: Optional[float]
    right: Optional[float]


@dataclass
class Pose2D:
    x: float                 # metres, odometry frame (origin = start pose)
    y: float
    heading: float           # radians, CCW positive, 0 = initial forward


@dataclass(frozen=True)
class VelocityCommand:
    v_mps: float             # forward, metres/second (negative = reverse)
    w_rps: float             # yaw rate, radians/second, +ve = turn left

    STOP = None              # set below


VelocityCommand.STOP = VelocityCommand(0.0, 0.0)


class RescueOutcome(Enum):
    IDLE = auto()
    BUSY = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class RescueStatus:
    outcome: RescueOutcome
    has_victim: bool         # is a victim currently contained?


class LedState(Enum):
    """Straight from the rules: yellow = searching, green = victim detected/collecting,
    red = returning to base with victim."""
    OFF = auto()
    YELLOW = auto()
    GREEN = auto()
    RED = auto()
