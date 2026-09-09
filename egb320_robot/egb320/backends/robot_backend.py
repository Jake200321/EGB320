"""Real-robot backend for the Raspberry Pi 5. Wires the teammates' modules together.

When Kushal / Roger's code lands, replace the stub imports below with their classes:
    from ..vision.<kushals_module> import <KushalsVisionClass>
    from ..rescue.<rogers_module> import <RogersRescueClass>

Mobility is now real: DFR0592Mobility drives the motor HAT. If the HAT isn't present
(running --check on a laptop, or the board didn't enumerate) we fall back to the stub
and say so loudly, rather than crashing the whole build or pretending it worked.
"""

from . import Subsystems
from ..config import Config
from ..leds.status_leds import ConsoleLeds   # -> GpioLeds once led_pins are set
from ..rescue.stub import StubRescue         # -> Roger's class
from ..utils.clock import WallClock
from ..vision.stub import StubVision         # -> Kushal's class


def build_mobility(cfg: Config, allow_stub: bool = True):
    from ..mobility.dfr0592 import DFR0592Mobility
    try:
        return DFR0592Mobility(cfg)
    except (ImportError, RuntimeError, OSError) as exc:
        if not allow_stub:
            raise
        from ..mobility.stub import StubMobility
        print("=" * 72)
        print("[robot_backend] MOTOR HAT UNAVAILABLE -- falling back to StubMobility.")
        print(f"[robot_backend] {type(exc).__name__}: {exc}")
        print("[robot_backend] THE ROBOT WILL NOT MOVE. Run tools/motor_spin_test.py.")
        print("=" * 72)
        return StubMobility()


def build_robot(cfg: Config) -> Subsystems:
    mobility = build_mobility(cfg)
    vision = StubVision()
    rescue = StubRescue()
    leds = ConsoleLeds()                     # GpioLeds(cfg.hardware.led_pins)

    def shutdown():
        closer = getattr(mobility, "close", None)
        (closer or mobility.stop)()

    return Subsystems(mobility=mobility, vision=vision, rescue=rescue,
                      leds=leds, clock=WallClock(), shutdown=shutdown)
