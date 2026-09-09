"""Motion primitives. These are the only functions that send VelocityCommands.

The algorithms decide WHICH cell; these decide HOW to get there, one cell at a time:
    turn_to(heading)        rotate in place with a P controller on heading error
    drive_one_cell()        carrot on the cell centreline + wall-centering trim; watchdogs for
                            'blocked' (front range) and 'stalled' (no odometry progress)
    approach(detection)     visual servo: turn until bearing ~ 0, creep until range <= 10 cm
    back_off()              reverse ~half a cell (RECOVER / blocked cell)

Every primitive is NON-BLOCKING: call tick() each nav loop iteration, check .done / .result.
That keeps the state machine in charge of the clock (time-low must be able to interrupt a drive).
"""

from ..config import NavConfig
from ..interfaces.messages import Detection, VelocityCommand
from ..interfaces.mobility import MobilityInterface
from .localisation import Odometry, GridLocaliser
from .wall_map import Heading


class MotionController:
    def __init__(self, mobility: MobilityInterface, odom: Odometry,
                 localiser: GridLocaliser, cfg: NavConfig):
        self.mobility = mobility
        self.odom = odom
        self.localiser = localiser
        self.cfg = cfg
        self._active = None      # current primitive (a small generator/object) or None
        self.result = None       # "arrived" | "blocked" | "stalled" | "timeout"

    @property
    def busy(self) -> bool:
        return self._active is not None

    # ------------------------------------------------------------------ TODO: primitives
    def turn_to(self, heading: Heading) -> None:
        raise NotImplementedError

    def drive_one_cell(self) -> None:
        raise NotImplementedError

    def approach(self, target: Detection) -> None:
        raise NotImplementedError

    def back_off(self) -> None:
        raise NotImplementedError

    def tick(self, walls, dt: float) -> None:
        """Advance the active primitive by one control step; sets self.result when it finishes. TODO"""
        raise NotImplementedError

    def stop(self) -> None:
        self._active = None
        self.mobility.stop()
