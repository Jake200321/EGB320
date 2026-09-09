"""Placeholder MobilityInterface so the stack imports and runs before Dan's code lands."""

from ..interfaces.messages import EncoderSample, Pose2D, VelocityCommand, WallRanges


class StubMobility:
    def __init__(self):
        self.last_cmd = VelocityCommand.STOP
        self._t = 0.0

    def set_velocity(self, cmd: VelocityCommand) -> None:
        self.last_cmd = cmd

    def stop(self) -> None:
        self.last_cmd = VelocityCommand.STOP

    def read_encoders(self) -> EncoderSample:
        return EncoderSample(self._t, 0, 0)

    def reset_encoders(self) -> None:
        pass

    def read_odometry(self):
        return None

    def read_wall_ranges(self) -> WallRanges:
        return WallRanges(self._t, None, None, None)
