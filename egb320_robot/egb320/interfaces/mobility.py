"""Mobility (Dan) <-> Navigation (Jake).

Nav SENDS VelocityCommand; nav RECEIVES encoder ticks (and, on the real robot,
the ultrasonic wall ranges — they are physically Dan's chassis sensors but nav
reads them). Who integrates ticks -> Pose2D is an open team decision; this
interface offers both so either answer works.
"""

from typing import Protocol, Optional

from .messages import EncoderSample, Pose2D, VelocityCommand, WallRanges


class MobilityInterface(Protocol):
    def set_velocity(self, cmd: VelocityCommand) -> None:
        """Command robot linear + angular velocity. Must be safe to call at 20 Hz.
        Implementations clamp to the drivetrain's real limits."""

    def stop(self) -> None:
        """Hard stop. Must never raise — it's the last thing called in any failure path."""

    def read_encoders(self) -> EncoderSample:
        """Cumulative signed ticks per side since the last reset."""

    def reset_encoders(self) -> None: ...

    def read_odometry(self) -> Optional[Pose2D]:
        """Pose if mobility integrates it onboard; None means nav must integrate ticks itself
        (see navigation/localisation.py)."""

    def read_wall_ranges(self) -> WallRanges:
        """Left / front / right distances in metres (ultrasonics on the robot, ray sensors in the sim)."""
