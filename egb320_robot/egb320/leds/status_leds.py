"""GPIO driver for the Team 12 LED board (LED indicators/ KiCad project).

Only the state machine sets these (via Mission -> LedInterface). TODO once BCM pins are known:
use gpiozero.LED per colour; set() turns exactly one on.
"""

from ..interfaces.messages import LedState


class ConsoleLeds:
    """Prints the LED state — for sim/mock runs and for the bench before the board exists."""
    def set(self, state: LedState) -> None:
        print(f"[LED] {state.name}")

    def off(self) -> None:
        print("[LED] OFF")


class GpioLeds:
    def __init__(self, pins: dict):
        self.pins = pins            # {"yellow": BCM, "green": BCM, "red": BCM}
        # TODO: from gpiozero import LED; self._leds = {name: LED(pin) for ...}
        raise NotImplementedError("set led_pins in config.py, then implement")

    def set(self, state: LedState) -> None:
        raise NotImplementedError

    def off(self) -> None:
        raise NotImplementedError
