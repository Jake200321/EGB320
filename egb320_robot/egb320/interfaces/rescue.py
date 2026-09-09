"""Rescue collection (Roger) <-> Navigation (Jake).

Nav triggers a routine on state entry and then POLLS status(). The done/failed
+ has_victim reply is the signal that ends COLLECT / DELIVER — a timeout in nav
is only a fallback guard. (This is the arrow that was missing from architecture v1.)
"""

from typing import Protocol

from .messages import RescueStatus


class RescueInterface(Protocol):
    def collect(self) -> None:
        """Start the capture routine (robot is stopped within 10 cm of the victim)."""

    def release(self) -> None:
        """Deposit the contained victim (robot is in the base zone)."""

    def clear_rubble(self) -> None:
        """Level 3: lift / slide the rubble obstruction. May be a no-op until the mechanism exists."""

    def status(self) -> RescueStatus:
        """BUSY while a routine runs, then DONE or FAILED; has_victim reflects containment."""

    def abort(self) -> None:
        """Stop whatever is running and return to a safe pose."""
