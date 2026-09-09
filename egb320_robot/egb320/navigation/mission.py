"""The mission loop: the one place where subsystems, map, planner, motion and the
state machine meet. Runs at NavConfig.control_dt_s (20 Hz).

    while not finished:
        obs = gather()                 # read vision / walls / encoders / rescue status
        state, why = fsm.step(obs)     # the rules
        act(state)                     # command mobility / rescue / LEDs for that state
        log.tick(...)                  # black box
        clock.sleep(dt)
"""

from ..config import Config
from ..interfaces import (LedInterface, MobilityInterface, RescueInterface,
                          VisionInterface, VelocityCommand)
from ..utils.clock import Clock
from ..utils.mission_log import MissionLog
from .localisation import GridLocaliser, Odometry
from .motion import MotionController
from .state_machine import NavState, NavStateMachine, Observation
from .wall_map import WallMap


class Mission:
    def __init__(self, cfg: Config, mobility: MobilityInterface, vision: VisionInterface,
                 rescue: RescueInterface, leds: LedInterface, clock: Clock, log: MissionLog):
        self.cfg = cfg
        self.mobility, self.vision, self.rescue, self.leds = mobility, vision, rescue, leds
        self.clock, self.log = clock, log

        self.map = WallMap(cfg.maze.columns, cfg.maze.rows)
        self.odom = Odometry(cfg.geometry)
        self.localiser = GridLocaliser(cfg.maze.cell_size_m, cfg.maze.base_cell)
        self.motion = MotionController(mobility, self.odom, self.localiser, cfg.nav)
        self.fsm = NavStateMachine(time_low_s=cfg.maze.time_low_s)

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        self.vision.start()
        self.leds.set(self.fsm.led)
        try:
            while self.fsm.state is not NavState.FINISHED:
                if self.clock.now() >= self.cfg.maze.time_limit_s:
                    break                                    # hard 7-minute cap
                obs = self.gather()
                state, why = self.fsm.step(obs)
                if why:
                    self.leds.set(self.fsm.led)              # LED set in ONE place
                    self.log.transition(obs.t, state, why)
                self.act(state, obs)
                self.clock.sleep(self.cfg.nav.control_dt_s)
        finally:
            self.motion.stop()
            self.mobility.stop()                             # never drive after the loop ends
            self.vision.stop()
            self.leds.off()
            self.log.save()

    # ------------------------------------------------------------------ TODO
    def gather(self) -> Observation:
        """Read every interface once and build the Observation. TODO
        Remember: stale vision -> treat as no detections; rescue.status() is polled, not awaited."""
        raise NotImplementedError

    def act(self, state: NavState, obs: Observation) -> None:
        """Per-state commands. Entry actions fire when the state just changed; 'do' actions every tick.
        e.g. EXPLORE: if not motion.busy -> plan next frontier step; motion.tick()
             COLLECT: on entry mobility.stop(); rescue.collect()
             FINISHED: mobility.stop()  (provably nothing drives after 7:00)
        TODO"""
        raise NotImplementedError
