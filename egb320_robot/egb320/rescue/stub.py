"""Placeholder RescueInterface: pretends every routine succeeds after `latency_s`."""

import time

from ..interfaces.messages import RescueOutcome, RescueStatus


class StubRescue:
    def __init__(self, latency_s: float = 1.0):
        self.latency_s = latency_s
        self._started = None
        self._pending = None          # "collect" | "release" | "rubble"
        self._has_victim = False

    def collect(self) -> None:      self._start("collect")
    def release(self) -> None:      self._start("release")
    def clear_rubble(self) -> None: self._start("rubble")
    def abort(self) -> None:        self._pending = None

    def _start(self, what):
        self._pending, self._started = what, time.monotonic()

    def status(self) -> RescueStatus:
        if self._pending is None:
            return RescueStatus(RescueOutcome.IDLE, self._has_victim)
        if time.monotonic() - self._started < self.latency_s:
            return RescueStatus(RescueOutcome.BUSY, self._has_victim)
        if self._pending == "collect":  self._has_victim = True
        if self._pending == "release":  self._has_victim = False
        self._pending = None
        return RescueStatus(RescueOutcome.DONE, self._has_victim)
