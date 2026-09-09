# Software architecture — Team 12 robot

```
                 ┌────────────┐   DetectionFrame (≥10 Hz)   ┌──────────────────────────┐
   Pi Camera ──▶ │  vision/   │ ───────────────────────────▶ │                          │
                 │  (Kushal)  │                              │      navigation/ (Jake)   │
                 └────────────┘                              │                          │
                 ┌────────────┐   EncoderSample / WallRanges │  mission.py  (the loop)  │
   HAT+encoders  │ mobility/  │ ───────────────────────────▶ │  state_machine.py (rules)│ ──▶ leds/
   ultrasonics ◀─│  (Dan)     │ ◀─────────────────────────── │  planner + wall_map      │      Y/G/R
                 └────────────┘   VelocityCommand (20 Hz)    │  localisation + motion   │
                 ┌────────────┐   collect/release/rubble     │                          │
   mechanism ◀── │  rescue/   │ ◀─────────────────────────── │                          │
                 │  (Roger)   │ ───────────────────────────▶ │                          │
                 └────────────┘   RescueStatus               └──────────────────────────┘
```

* Every arrow is a dataclass/Protocol in `egb320/interfaces/` — the diagram and the code cannot drift.
* `backends/` swaps the real subsystems for the CoppeliaSim robot (`sim`) or stubs (`mock`) without
  touching navigation code. The same `Mission` runs on all three — that is the "same software
  architecture transfers to the robot" argument for the Milestone 2 HIL allowance.
* Threading: vision runs its own loop and publishes `latest()`; nav polls at 20 Hz. Budget nav's
  loop so vision keeps ≥ 10 Hz on the Pi 5.
* Time: `utils/clock.py` — wall clock on the robot, sim time in CoppeliaSim (which free-runs).

## Run
```
pip install -r requirements.txt
python main.py --backend mock --check      # wiring smoke test
python -m pytest                           # unit tests (state machine / map / planner)
python main.py --backend sim               # with EGB320_search_and_rescue_2026.ttt open
```
