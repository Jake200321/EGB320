"""One clock abstraction so the SAME mission code runs on wall-clock (robot),
simulation time (CoppeliaSim free-runs faster than real time) or stepped mock time."""

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, dt: float) -> None: ...


class WallClock:
    def __init__(self):
        self._start = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._start

    def sleep(self, dt: float) -> None:
        time.sleep(dt)


class SimClock:
    """Paces the loop on CoppeliaSim simulation time (correct at any sim speed)."""

    def __init__(self, sim):
        self._sim = sim
        self._offset = sim.getSimulationTime()

    def now(self) -> float:
        return self._sim.getSimulationTime() - self._offset

    def sleep(self, dt: float) -> None:
        target = self.now() + dt
        stalled = time.monotonic()
        while self.now() < target:
            time.sleep(0.003)
            if time.monotonic() - stalled > 5.0:
                raise RuntimeError("Simulation time not advancing — is CoppeliaSim paused?")
