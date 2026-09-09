# Mobility (Dan) — what navigation needs from this folder

Put your motor-control code here (any file names you like). Then expose ONE class that
implements `MobilityInterface` (`egb320/interfaces/mobility.py`):

| Method | Nav calls it | Contract |
|---|---|---|
| `set_velocity(cmd)` | every tick (20 Hz) | `cmd.v_mps` forward m/s (negative = reverse), `cmd.w_rps` yaw rad/s, **positive = turn left**. Clamp to real limits. |
| `stop()` | on every failure path | must never raise |
| `read_encoders()` | every tick | `EncoderSample(t, left_ticks, right_ticks)` — signed, cumulative since reset |
| `reset_encoders()` | at GO | |
| `read_odometry()` | every tick | `Pose2D` if you integrate onboard, else `None` (nav integrates) |
| `read_wall_ranges()` | every tick | `WallRanges(t, left, front, right)` in metres, `None` = nothing in range |

Hardware we've agreed on (see `nav Software/Hardware_Parts_Reference.md`):
- DFRobot DFR0592 HAT over I2C (addr `0x10`), lib `DFRobot_RaspberryPi_DC_Motor` (clone into `vendor/`).
- N20 6 V 1:50 encoder motors, 7 ppr/channel at motor shaft → 700 (×2) or 1400 (×4) counts per output rev. **Please confirm which decode the HAT uses** — `config.py` has `quadrature_decode`.
- Pololu 30T tracks: 123 mm sprocket spacing (wheelbase), 24 mm pitch diameter.
- 2× SRF05 + 1× HC-SR04 ultrasonics: left / front / right. Fire sequentially, ≥60 ms apart.

## Answered (2026-09-09)

**Does the HAT give cumulative ticks, or only RPM?** Only RPM. Checked against the
library source — `get_encoder_speed()` is the entire encoder API, there is no tick
register. So:

* **Nav integrates pose**, and `read_odometry()` returns `None`.
* `read_encoders()` synthesises cumulative ticks by integrating RPM over the poll
  interval. They are derived, not counted, so they drift with poll jitter on top of
  track slip. Re-anchor against walls/markers; don't dead-reckon across the maze.
* The ×2/×4 quadrature question is therefore mostly moot — the board does its own
  pulse→RPM conversion onboard, and `drive.encoder_rpm_scale` absorbs any error in it.
  `tools/bench_motor_rpm.py --verify` measures that scale directly.

## Working driver already here

`dfr0592.py` implements the full interface against the HAT, so the robot can drive
today. Dan — take it over, replace it, or build on it; it's a starting point, not a
claim on the subsystem. Deliberately not done:

* **Ultrasonics** — `read_wall_ranges()` returns all-`None` until
  `cfg.hardware.ultrasonic_pins` is filled in. Nothing is fabricated.
* **Closed-loop speed control** — duty mapping is open-loop off `drive.max_wheel_rpm`.
  Encoder RPM is read but not yet fed back.

Numbers in `config.py` marked MEASURE are guesses until the bench scripts run:
`max_wheel_rpm`, `min_duty_percent`, `turn_gain`, `encoder_rpm_scale`.

**Watch the duty cap.** The N20s are 6 V motors on the HAT's 7–12 V rail, so 100 %
duty overdrives them. Set `drive.max_duty_percent` from the real pack voltage
(≈ 6/V_pack): 2S → ~80 %, 3S → ~54 %.

A stub returning zeros still lives in `stub.py`, and `robot_backend.py` falls back to
it (loudly) when the HAT isn't present, so `--check` still passes on a laptop.
