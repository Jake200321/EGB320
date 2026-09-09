"""Navigation -> status LEDs. Set from ONE place (the state machine) so the LED can never lie."""

from typing import Protocol

from .messages import LedState


class LedInterface(Protocol):
    def set(self, state: LedState) -> None: ...
    def off(self) -> None: ...
