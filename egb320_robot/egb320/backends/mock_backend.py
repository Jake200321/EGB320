"""All-stub backend: no hardware, no simulator. Use it for unit tests and for stepping
the state machine on a laptop. TODO (later): port the headless maze replica from
Sim Repo/EGB320_sim/nav/mock_mazebot.py so exploration can be tested end-to-end here."""

from . import Subsystems
from ..config import Config
from ..leds.status_leds import ConsoleLeds
from ..mobility.stub import StubMobility
from ..rescue.stub import StubRescue
from ..utils.clock import WallClock
from ..vision.stub import StubVision


def build_mock(cfg: Config) -> Subsystems:
    return Subsystems(mobility=StubMobility(), vision=StubVision(),
                      rescue=StubRescue(), leds=ConsoleLeds(), clock=WallClock())
