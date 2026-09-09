"""CoppeliaSim backend: wraps mazebot_lib.MazeBot into the four interfaces.

API mapping (from Sim Repo/EGB320_sim/README.md):
    SetTargetVelocities(v, w)        <- MobilityInterface.set_velocity   (+w = left, same convention)
    GetWheelEncoders()               <- read_encoders   {'time','left_ticks','right_ticks'}
    GetOdometry()                    <- read_odometry   {'time','x','y','heading'}
    GetWallDistances()               <- read_wall_ranges {'left','front','right'} (m or None)
    UpdateObjectPositions(); GetDetections()  <- VisionInterface.latest
         {'base','victim','rubble_victim','hazard','victim_object'} -> [[range, bearing], ...]
    CollectVictim() -> (success, label, distance); ReleaseVictim(); HasVictim()  <- RescueInterface

TODO: implement the three adapter classes below (pure plumbing — no nav logic here).
"""

import os
import sys

from . import Subsystems
from ..config import Config
from ..interfaces.messages import (Detection, DetectionFrame, EncoderSample, MarkerClass,
                                   Pose2D, RescueOutcome, RescueStatus, VelocityCommand,
                                   WallRanges)
from ..leds.status_leds import ConsoleLeds
from ..utils.clock import SimClock

DETECTION_KEYS = {
    "base": MarkerClass.BASE,
    "victim": MarkerClass.VICTIM,
    "rubble_victim": MarkerClass.TRAPPED_VICTIM,
    "hazard": MarkerClass.HAZARD,
    "victim_object": MarkerClass.VICTIM_OBJECT,
}


def _import_mazebot(cfg: Config):
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.normpath(os.path.join(here, cfg.sim_repo_path)))
    from mazebot_lib import MazeBot, RobotParameters   # noqa: E402
    return MazeBot, RobotParameters


class SimMobility:
    def __init__(self, robot): self.robot = robot
    def set_velocity(self, cmd: VelocityCommand) -> None: raise NotImplementedError
    def stop(self) -> None: self.robot.SetTargetVelocities(0.0, 0.0)
    def read_encoders(self) -> EncoderSample: raise NotImplementedError
    def reset_encoders(self) -> None: raise NotImplementedError
    def read_odometry(self) -> Pose2D: raise NotImplementedError
    def read_wall_ranges(self) -> WallRanges: raise NotImplementedError


class SimVision:
    def __init__(self, robot): self.robot = robot
    def start(self) -> None: pass
    def stop(self) -> None: pass
    def is_healthy(self) -> bool: return True
    def latest(self) -> DetectionFrame: raise NotImplementedError   # use DETECTION_KEYS


class SimRescue:
    def __init__(self, robot): self.robot = robot
    def collect(self) -> None: raise NotImplementedError
    def release(self) -> None: raise NotImplementedError
    def clear_rubble(self) -> None: pass
    def abort(self) -> None: pass
    def status(self) -> RescueStatus: raise NotImplementedError


def build_sim(cfg: Config) -> Subsystems:
    MazeBot, RobotParameters = _import_mazebot(cfg)
    params = RobotParameters()
    params.wheelBase = 0.22        # measured effective skid-steer wheelbase in the 2026 scene
    robot = MazeBot(params)
    robot.StartSimulator()
    return Subsystems(mobility=SimMobility(robot), vision=SimVision(robot),
                      rescue=SimRescue(robot), leds=ConsoleLeds(),
                      clock=SimClock(robot.sim), shutdown=robot.StopSimulator)
