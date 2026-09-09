"""Where am I? Encoder ticks -> Pose2D, then corrected against the maze structure.

Differential-drive dead reckoning, per tick (all in metres / radians):
    d_left  = delta_left_ticks  * metres_per_count
    d_right = delta_right_ticks * metres_per_count
    d_centre = (d_left + d_right) / 2          -> how far the robot moved
    d_theta  = (d_right - d_left) / track_spacing  -> how much it turned (CCW +ve)
    x += d_centre * cos(heading + d_theta/2)
    y += d_centre * sin(heading + d_theta/2)
    heading += d_theta   (wrap to -pi..pi)

Known problem (measured in the sim, expected worse on tracks): skid-steer heading from
encoders drifts badly. Plan: IMU for heading if the team adds one; otherwise snap to the
corridor using left/right wall ranges and re-anchor position at cell centres.
"""

import math
from typing import Optional

from ..config import RobotGeometry
from ..interfaces.messages import EncoderSample, Pose2D, WallRanges


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Odometry:
    def __init__(self, geometry: RobotGeometry):
        self.geometry = geometry
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self._last: Optional[EncoderSample] = None

    def reset(self) -> None:
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self._last = None

    def update(self, sample: EncoderSample) -> Pose2D:
        """Integrate one encoder sample into self.pose using the equations above. TODO"""
        raise NotImplementedError

    def fuse_heading(self, imu_heading_rad: float) -> None:
        """Optional: replace/blend the drifting encoder heading with an IMU reading. TODO"""
        raise NotImplementedError


class GridLocaliser:
    """Turns a continuous Pose2D into (cell, cardinal heading) and uses wall ranges to
    correct lateral drift inside a corridor."""

    def __init__(self, cell_size_m: float, base_cell):
        self.cell_size_m = cell_size_m
        self.base_cell = base_cell

    def cell_centre(self, cell) -> tuple:
        """Odometry-frame (x, y) of a cell centre, given the robot starts at base facing NORTH. TODO"""
        raise NotImplementedError

    def cell_of(self, pose: Pose2D) -> tuple:
        """Inverse of cell_centre. TODO"""
        raise NotImplementedError

    def lateral_correction(self, walls: WallRanges) -> float:
        """(left - right) imbalance -> a small yaw-rate trim to stay centred. TODO"""
        raise NotImplementedError
