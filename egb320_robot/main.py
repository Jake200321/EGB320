#!/usr/bin/env python3
"""EGB320 Team 12 robot — entry point.

    python main.py --backend mock            # laptop, stubs only
    python main.py --backend sim             # CoppeliaSim scene must be open (port 23000)
    python main.py --backend robot           # on the Raspberry Pi
    python main.py --backend mock --check    # import + wiring smoke test, no mission loop
"""

import argparse

from egb320.backends import build
from egb320.config import Config
from egb320.navigation.mission import Mission
from egb320.utils.mission_log import MissionLog


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["mock", "sim", "robot"], default="mock")
    parser.add_argument("--check", action="store_true", help="build everything, then exit")
    parser.add_argument("--time-limit", type=float, default=None, help="override the 7-min cap (s)")
    args = parser.parse_args()

    cfg = Config()
    if args.time_limit is not None:
        cfg.maze.time_limit_s = args.time_limit

    subsystems = build(args.backend, cfg)
    log = MissionLog(name=args.backend)
    mission = Mission(cfg, subsystems.mobility, subsystems.vision, subsystems.rescue,
                      subsystems.leds, subsystems.clock, log)

    if args.check:
        print(f"OK: backend={args.backend} state={mission.fsm.state.name} "
              f"map={cfg.maze.columns}x{cfg.maze.rows} counts/rev={cfg.geometry.counts_per_output_rev:.0f}")
        subsystems.shutdown()
        return

    try:
        mission.run()
    finally:
        subsystems.shutdown()


if __name__ == "__main__":
    main()
