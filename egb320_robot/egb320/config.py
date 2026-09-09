"""Every tunable number in one place.

Rule: no magic numbers inside subsystem code — put them here with units in the name
or a comment, so they can be tuned on the bench without hunting through modules.
Values marked TODO/CONFIRM are still open items (see nav Software/README.md).
"""

from dataclasses import dataclass, field
import math


# --------------------------------------------------------------------------- maze
@dataclass
class MazeConfig:
    columns: int = 7                 # CONFIRM final grid size from the unit website
    rows: int = 7
    cell_size_m: float = 0.280       # CONFIRM final cell dimension (2 m x 2 m field)
    base_cell: tuple = (0, 6)        # (column, row) — robot starts here
    victims_expected: int = 3
    time_limit_s: float = 420.0      # 7-minute demo window
    time_low_s: float = 390.0        # 6:30 — "wrap it up" global event (state machine worksheet, Step 4 Q4)


# --------------------------------------------------------------------------- robot geometry
@dataclass
class RobotGeometry:
    """Real Team 12 hardware (see nav Software/Hardware_Parts_Reference.md)."""
    track_spacing_m: float = 0.123          # Pololu 30T sprocket centre-to-centre = effective wheelbase
    sprocket_pitch_diameter_m: float = 0.024  # where the track engages -> effective rolling diameter
    gear_ratio: float = 50.0                # N20 1:50
    encoder_pulses_per_motor_rev: int = 7   # per channel, at the motor shaft
    quadrature_decode: int = 4              # TODO CONFIRM: DFR0592 HAT uses x2 or x4 decoding?

    @property
    def counts_per_output_rev(self) -> float:
        return self.encoder_pulses_per_motor_rev * self.quadrature_decode * self.gear_ratio  # 700 or 1400

    @property
    def metres_per_count(self) -> float:
        return math.pi * self.sprocket_pitch_diameter_m / self.counts_per_output_rev


# --------------------------------------------------------------------------- hardware pins / buses
@dataclass
class HardwareConfig:
    i2c_bus: int = 1                        # /dev/i2c-1 on every Pi incl. Pi 5
    motor_hat_i2c_addr: int = 0x10          # DFRobot DFR0592 default
    motor_left_id: int = 1                  # M1
    motor_right_id: int = 2                 # M2
    # The two motors face opposite ways on the chassis, so one channel's "CW" is the
    # robot's forward and the other's is backward. CONFIRM both on the bench with
    # tools/motor_spin_test.py and flip these until both tracks drive forward together.
    motor_left_invert: bool = False
    motor_right_invert: bool = True
    ultrasonic_pins: dict = field(default_factory=lambda: {
        # name: (TRIG_BCM, ECHO_BCM)  — TODO set once the HAT passthrough pins are known
        "left": (None, None),
        "front": (None, None),
        "right": (None, None),
    })
    led_pins: dict = field(default_factory=lambda: {
        "yellow": None, "green": None, "red": None,   # TODO BCM pins to the LED board (LED indicators/)
    })
    ultrasonic_min_period_s: float = 0.060  # HC-SR04 wants >60 ms between triggers; fire sequentially


# --------------------------------------------------------------------------- drive / motor calibration
@dataclass
class DriveConfig:
    """Motor electrical limits + the numbers that turn m/s into PWM duty.

    Anything marked MEASURE is a placeholder until tools/bench_motor_rpm.py has been
    run on the real chassis. Open-loop duty mapping is only as good as these.
    """
    pwm_frequency_hz: int = 1000            # DFR0592 accepts 100-12750 Hz

    # SAFETY — the N20s are rated 6 V but the DFR0592 needs a 7-12 V motor rail, so
    # 100 % duty puts the full pack voltage across a 6 V motor. Cap duty at roughly
    # 6 V / V_pack: 2S LiPo (7.4 V) -> ~80 %, 3S (11.1 V) -> ~54 %.
    # SET THIS FROM THE ACTUAL BATTERY before any sustained running.
    max_duty_percent: float = 80.0

    min_duty_percent: float = 25.0          # MEASURE: below this the geartrain won't break stiction
    max_wheel_rpm: float = 200.0            # MEASURE: output-shaft RPM at max_duty_percent
                                            # (datasheet only gives 430 rpm @ 12 V no-load)

    # Skid-steer correction. On tracks the effective turning width is wider than the
    # 123 mm sprocket spacing because the inside track skids, so a commanded yaw rate
    # under-rotates. MEASURE: command a 360 deg spin, divide commanded by achieved.
    turn_gain: float = 1.0

    # get_encoder_speed() is the ONLY encoder output the DFR0592 gives (no tick register).
    encoder_reports_signed: bool = True     # CONFIRM: does it read negative in reverse, or just magnitude?
    encoder_rpm_scale: float = 1.0          # MEASURE: true RPM / board-reported RPM

    watchdog_timeout_s: float = 0.5         # no set_velocity() for this long -> motors stop. 0 disables.


# --------------------------------------------------------------------------- navigation tuning
@dataclass
class NavConfig:
    control_dt_s: float = 0.05              # 20 Hz nav loop (vision must stay >= 10 Hz on its own)
    forward_speed_mps: float = 0.15
    approach_speed_mps: float = 0.07
    max_turn_rate_rps: float = 1.6
    min_turn_rate_rps: float = 0.22
    k_heading: float = 2.8                  # P gain on heading error
    turn_tolerance_rad: float = 0.03
    wall_near_m: float = 0.20               # sensor reading below this at a cell centre = wall present
    collision_stop_m: float = 0.05
    found_range_m: float = 0.10             # rules: "found" = within 10 cm + green LED
    detection_debounce_frames: int = 3      # consecutive frames before acting on a detection
    rescue_timeout_s: float = 15.0          # fallback guard only — Roger's done/failed is the real signal
    stall_timeout_s: float = 3.0            # no progress for this long -> RECOVER


@dataclass
class Config:
    maze: MazeConfig = field(default_factory=MazeConfig)
    geometry: RobotGeometry = field(default_factory=RobotGeometry)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    nav: NavConfig = field(default_factory=NavConfig)
    sim_repo_path: str = "../../Sim Repo/EGB320_sim"   # relative to this repo root; used by --backend sim
