"""Backends build the four subsystem interfaces + a clock for a given target.

    build("mock")  -> stubs only; runs anywhere, instant (unit-test the state machine here)
    build("sim")   -> CoppeliaSim via mazebot_lib (Sim Repo/EGB320_sim must be reachable)
    build("robot") -> Dan / Kushal / Roger's real modules on the Pi
"""

from dataclasses import dataclass

from ..config import Config


@dataclass
class Subsystems:
    mobility: object
    vision: object
    rescue: object
    leds: object
    clock: object
    start: callable = lambda: None    # e.g. robot.StartSimulator()
    shutdown: callable = lambda: None


def build(target: str, cfg: Config) -> Subsystems:
    if target == "mock":
        from .mock_backend import build_mock
        return build_mock(cfg)
    if target == "sim":
        from .sim_backend import build_sim
        return build_sim(cfg)
    if target == "robot":
        from .robot_backend import build_robot
        return build_robot(cfg)
    raise ValueError(f"unknown backend {target!r}")
