"""The nav state machine. The diagram IS the code — keep docs/state_machine.mmd in sync.

A state = a stretch of time where these three answers don't change:
    1. COMMANDING   what nav tells Mobility / Rescue to do
    2. WAITING FOR  which incoming signal would change behaviour
    3. LED          yellow / green / red

Transitions are labelled with INFORMATION arrows (vision, rescue status, clock).
Actions inside states are COMMAND arrows (velocity, collect, LED).
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from ..interfaces.messages import DetectionFrame, LedState, RescueStatus, WallRanges


class NavState(Enum):
    BOOT = auto()          # sensors up, camera warm, map zeroed
    WAIT_FOR_GO = auto()   # sitting in base — TODO decide the go signal (button? first frame? switch?)
    EXPLORE = auto()       # frontier exploration, LED yellow
    APPROACH = auto()      # visual servo on bearing until range <= 10 cm, LED green
    COLLECT = auto()       # stopped, rescue routine running, LED green
    RETURN = auto()        # flood-fill home over known map, LED red
    DELIVER = auto()       # in base, release victim
    CLEAR_RUBBLE = auto()  # Level 3 only — TODO where does it hook in?
    RECOVER = auto()       # stuck / lost — reverse, re-localise, replan
    FINISHED = auto()      # terminal: entry action is STOP. Nothing drives after this.


# LED truth table — one place, never wrong.  TODO: fill the blanks you decide on.
LED_FOR_STATE = {
    NavState.BOOT: LedState.OFF,
    NavState.WAIT_FOR_GO: LedState.OFF,      # TODO yellow or off while waiting?
    NavState.EXPLORE: LedState.YELLOW,
    NavState.APPROACH: LedState.GREEN,
    NavState.COLLECT: LedState.GREEN,
    NavState.RETURN: LedState.RED,
    NavState.DELIVER: LedState.RED,          # TODO still "returning with victim"? decide + defend
    NavState.CLEAR_RUBBLE: LedState.GREEN,   # TODO
    NavState.RECOVER: LedState.YELLOW,       # TODO depends on whether carrying a victim?
    NavState.FINISHED: LedState.OFF,         # TODO
}


@dataclass
class Observation:
    """Everything the state machine is allowed to look at on one tick.
    Built by mission.py from the subsystem interfaces — the machine itself never touches hardware."""
    t: float                                  # mission clock, seconds since GO
    detections: Optional[DetectionFrame]      # None = vision has nothing / is stale
    walls: WallRanges
    rescue: RescueStatus
    at_target_cell: bool = False              # motion primitive finished the current cell / path
    at_base: bool = False
    stuck: bool = False                       # localisation/motion watchdog tripped
    victims_remaining: int = 3
    go_signal: bool = False


class NavStateMachine:
    def __init__(self, time_low_s: float):
        self.state = NavState.BOOT
        self.time_low_s = time_low_s
        self.history = []                     # (t, old, new, why) — your black box on demo day

    @property
    def led(self) -> LedState:
        return LED_FOR_STATE[self.state]

    def step(self, obs: Observation) -> Tuple[NavState, Optional[str]]:
        """Called every nav tick. Returns (new_state, reason) — reason is None if nothing changed.

        Write ONE `if` per arrow on your diagram. Guards (e.g. debounce N frames, retry counts,
        'carrying a victim at time-low?') live here, not in new states.
        """
        old = self.state
        new, why = self._transition(old, obs)
        if new is not old:
            self.state = new
            self.history.append((round(obs.t, 2), old.name, new.name, why))
        return self.state, why

    # ------------------------------------------------------------------ TODO: the rules
    def _transition(self, s: NavState, obs: Observation) -> Tuple[NavState, Optional[str]]:
        # Global event first: time low -> superstate exit (worksheet Step 4, Q4).
        # TODO: if obs.t >= self.time_low_s and s in <driving states>: carrying? -> RETURN : ...

        if s is NavState.BOOT:
            return NavState.WAIT_FOR_GO, "sensors OK"          # TODO real health checks
        if s is NavState.WAIT_FOR_GO:
            if obs.go_signal:
                return NavState.EXPLORE, "go"
        if s is NavState.EXPLORE:
            # TODO victim marker seen for N consecutive frames -> APPROACH
            # TODO hazard seen -> map write only (NOT a transition)
            # TODO obs.stuck -> RECOVER
            pass
        if s is NavState.APPROACH:
            # TODO range <= found_range -> COLLECT ; marker lost N frames -> EXPLORE (debounce!)
            pass
        if s is NavState.COLLECT:
            # TODO DONE + has_victim -> RETURN ; FAILED -> retry (how many?) ; timeout -> treat as FAILED
            pass
        if s is NavState.RETURN:
            # TODO obs.at_base -> DELIVER ; obs.stuck -> RECOVER
            pass
        if s is NavState.DELIVER:
            # TODO release DONE: [victims remain] -> EXPLORE ; [all delivered or time low] -> FINISHED
            pass
        if s is NavState.RECOVER:
            # TODO after back-off + re-localise: return to the state you came from? or always EXPLORE?
            pass
        return s, None
