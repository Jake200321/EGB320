# EGB320 Team 12 — Search and Rescue Maze Robot

Software for the 2026 EGB320 Search and Rescue Maze Challenge. One repo, one folder per
assessed subsystem, joined by explicit interface contracts.

| Folder | Owner | What goes there |
|---|---|---|
| `egb320/navigation/` | Jake | state machine, map, planner, localisation, motion, mission loop |
| `egb320/mobility/` | Dan | motor driver HAT + encoders + ultrasonics (see its README) |
| `egb320/vision/` | Kushal | Pi camera + OpenCV detections (see its README) |
| `egb320/rescue/` | Roger | collection mechanism (see its README) |
| `egb320/leds/` | Jake | yellow / green / red status LEDs |
| `egb320/interfaces/` | team | **the contracts** — messages + Protocols every subsystem must honour |
| `egb320/backends/` | Jake | build the subsystems for `mock` / `sim` / `robot` |
| `tests/` | all | pytest; state machine + map + planner are hardware-free |
| `tools/` | all | bench scripts for open hardware questions (RPM @ 6 V, decode mode, ultrasonics, LEDs) |
| `docs/` | team | architecture, interface table, Mermaid state machine |

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # laptop
pip install -r requirements-pi.txt         # Raspberry Pi only
python main.py --backend mock --check      # everything imports and wires up
python -m pytest                           # tests
python main.py --backend sim               # CoppeliaSim: open EGB320_search_and_rescue_2026.ttt first
python main.py --backend robot             # on the Pi
```

## First run on the Pi — bring the motors up before anything else
```bash
sudo raspi-config                          # Interface Options -> I2C -> enable
sudo i2cdetect -y 1                        # expect the HAT at 0x10
git clone https://github.com/DFRobot/DFRobot_RaspberryPi_Motor egb320/mobility/vendor
pip install -r requirements.txt -r requirements-pi.txt
```
Then, **robot on a block, tracks off the ground**:
```bash
python3 tools/motor_spin_test.py                      # do they turn? which way is forward?
python3 tools/bench_motor_rpm.py --stiction --sweep    # fills in config.py's MEASURE values
```
Write the results into `egb320/config.py`: `motor_*_invert`, `max_duty_percent`
(from your pack voltage — 6 V motors on a 7–12 V rail), `min_duty_percent`,
`max_wheel_rpm`, `encoder_reports_signed`.

## Teammates: dropping your code in
1. Put your files in your subsystem folder.
2. Provide one class that implements the Protocol in `egb320/interfaces/<subsystem>.py`
   (your folder's README lists exactly which methods and units).
3. Point `egb320/backends/robot_backend.py` at your class instead of the stub.
4. `python main.py --backend robot --check` must still pass.

## Status
Skeleton only (2026-09-03): interfaces and wiring are real; every `NotImplementedError` /
`TODO` is a piece of navigation logic still to be written. Design decisions and the
algorithm comparison behind them live in `../../Nav system/` and `../../nav Software/`.
