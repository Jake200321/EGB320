#!/usr/bin/env python3

from dataclasses import dataclass
import math
import time

from controller import Controller
import robot_config as cfg


@dataclass
class MotionResult:
    requested_left_ticks: float
    requested_right_ticks: float
    measured_left_ticks: int
    measured_right_ticks: int
    elapsed_s: float

    @property
    def average_abs_ticks(self) -> float:
        return (abs(self.measured_left_ticks) + abs(self.measured_right_ticks)) / 2.0


class RobotDrive:
    def __init__(self):
        self.controller = Controller(i2c_bus=cfg.I2C_BUS)
        self.controller.set_motor_shutdown_timeout(cfg.MOTOR_SHUTDOWN_TIMEOUT_S)
        self.stop()

    @staticmethod
    def mm_to_ticks(distance_mm: float) -> float:
        return distance_mm * cfg.TICKS_PER_MM

    @staticmethod
    def ticks_to_mm(ticks: float) -> float:
        return ticks / cfg.TICKS_PER_MM

    @staticmethod
    def controlled_speed_to_mm_s(speed: int) -> float:
        # Firmware unit: ticks / 0.01 s = 100 * ticks/s
        return (speed * 100.0) / cfg.TICKS_PER_MM

    def _set_logical_speed(self, left: int, right: int) -> None:
        left_hw = int(left * cfg.LEFT_MOTOR_SIGN)
        right_hw = int(right * cfg.RIGHT_MOTOR_SIGN)
        self.controller.set_motor_speed(left_hw, right_hw)

    def _logical_encoder_delta(self, relative) -> tuple[int, int]:
        left, right = self.controller.get_relative_encoder_ticks(relative)
        return (
            int(left * cfg.LEFT_MOTOR_SIGN),
            int(right * cfg.RIGHT_MOTOR_SIGN),
        )

    def set_track_speeds(self, left: int, right: int) -> None:

        for name, value in (("left", left), ("right", right)):
            if not isinstance(value, int):
                raise TypeError(f"{name} speed must be an integer")
            if abs(value) > cfg.MAX_TEST_SPEED:
                raise ValueError(
                    f"{name} speed must be between "
                    f"-{cfg.MAX_TEST_SPEED} and +{cfg.MAX_TEST_SPEED}"
                )
        self._set_logical_speed(left, right)

    def stop(self) -> None:
        self.controller.standby()

    def diagnostics(self) -> dict:
        return {
            "who_am_i": hex(self.controller.who_am_i()),
            "firmware": self.controller.get_firmware_version(),
            "pid": self.controller.get_pid_coefficients(),
            "shutdown_timeout_s": self.controller.get_motor_shutdown_timeout(),
            "ticks_per_mm": cfg.TICKS_PER_MM,
            "track_centre_spacing_mm": cfg.TRACK_CENTRE_SPACING_MM,
            "estimated_speed_mm_s_at_default": self.controlled_speed_to_mm_s(
                cfg.DEFAULT_DRIVE_SPEED
            ),
        }

    @staticmethod
    def _profile_speed(remaining_ticks: float, max_speed: int) -> int:
        max_speed = max(1, min(abs(int(max_speed)), cfg.MAX_TEST_SPEED))

        slowdown_ticks = max(
            cfg.POSITION_TOLERANCE_TICKS * 4,
            cfg.SLOWDOWN_DISTANCE_MM * cfg.TICKS_PER_MM,
        )

        scale = min(1.0, max(0.0, remaining_ticks / slowdown_ticks))
        command = round(max_speed * scale)

        if remaining_ticks > cfg.POSITION_TOLERANCE_TICKS:
            command = max(min(cfg.MIN_CONTROLLED_SPEED, max_speed), command)

        return min(command, max_speed)

    def _move_relative_ticks(
        self,
        left_target: float,
        right_target: float,
        max_speed: int,
        timeout_s: float | None = None,
        verbose: bool = True,
    ) -> MotionResult:
        if max_speed <= 0:
            raise ValueError("max_speed must be > 0")
        if max_speed > cfg.MAX_TEST_SPEED:
            raise ValueError(
                f"max_speed must be <= {cfg.MAX_TEST_SPEED} for the test configuration"
            )

        left_dir = 0 if left_target == 0 else (1 if left_target > 0 else -1)
        right_dir = 0 if right_target == 0 else (1 if right_target > 0 else -1)

        left_goal = abs(float(left_target))
        right_goal = abs(float(right_target))

        relative = self.controller.new_relative()
        left_total = 0
        right_total = 0

        largest_goal = max(left_goal, right_goal)
        expected_s = largest_goal / max(max_speed * 100.0, 1.0)
        if timeout_s is None:
            timeout_s = max(3.0, expected_s * 4.0 + 2.0)

        started = time.monotonic()
        next_print = started

        try:
            while True:
                dl, dr = self._logical_encoder_delta(relative)
                left_total += dl
                right_total += dr

                left_progress = left_total * left_dir if left_dir else left_goal
                right_progress = right_total * right_dir if right_dir else right_goal

                left_remaining = max(0.0, left_goal - left_progress)
                right_remaining = max(0.0, right_goal - right_progress)

                left_done = (
                    left_dir == 0
                    or left_remaining <= cfg.POSITION_TOLERANCE_TICKS
                )
                right_done = (
                    right_dir == 0
                    or right_remaining <= cfg.POSITION_TOLERANCE_TICKS
                )

                if left_done and right_done:
                    break

                left_cmd = 0 if left_done else (
                    left_dir * self._profile_speed(left_remaining, max_speed)
                )
                right_cmd = 0 if right_done else (
                    right_dir * self._profile_speed(right_remaining, max_speed)
                )

                self._set_logical_speed(left_cmd, right_cmd)

                now = time.monotonic()
                if verbose and now >= next_print:
                    print(
                        f"\rL {left_total:6d}/{left_target:8.1f}  "
                        f"R {right_total:6d}/{right_target:8.1f}  "
                        f"cmd=({left_cmd:3d},{right_cmd:3d})",
                        end="",
                        flush=True,
                    )
                    next_print = now + 0.20

                if now - started > timeout_s:
                    raise TimeoutError(
                        f"Motion timeout after {timeout_s:.1f}s. "
                        f"Encoder totals: L={left_total}, R={right_total}"
                    )

                time.sleep(cfg.CONTROL_PERIOD_S)

        finally:
            self.stop()
            time.sleep(cfg.STOP_SETTLE_TIME_S)

            # Capture encoder motion that occurred while stopping/coasting.
            try:
                dl, dr = self._logical_encoder_delta(relative)
                left_total += dl
                right_total += dr
            except Exception:
                pass

            if verbose:
                print()

        return MotionResult(
            requested_left_ticks=left_target,
            requested_right_ticks=right_target,
            measured_left_ticks=left_total,
            measured_right_ticks=right_total,
            elapsed_s=time.monotonic() - started,
        )

    def drive_distance(
        self,
        distance_mm: float,
        speed: int = cfg.DEFAULT_DRIVE_SPEED,
        verbose: bool = True,
    ) -> MotionResult:
        target = self.mm_to_ticks(distance_mm)
        return self._move_relative_ticks(target, target, speed, verbose=verbose)

    def pivot(
        self,
        angle_deg: float,
        speed: int = cfg.DEFAULT_TURN_SPEED,
        verbose: bool = True,
    ) -> MotionResult:

        if angle_deg == 0:
            return MotionResult(0, 0, 0, 0, 0.0)

        track_distance_mm = (
            math.pi
            * cfg.TRACK_CENTRE_SPACING_MM
            * abs(angle_deg)
            / 360.0
            * cfg.TURN_CALIBRATION
        )
        ticks = self.mm_to_ticks(track_distance_mm)

        if angle_deg > 0:   # clockwise / right
            left_target = ticks
            right_target = -ticks
        else:               # counter-clockwise / left
            left_target = -ticks
            right_target = ticks

        return self._move_relative_ticks(
            left_target, right_target, speed, verbose=verbose
        )

    def turn_right(
        self,
        angle_deg: float = 90.0,
        speed: int = cfg.DEFAULT_TURN_SPEED,
        verbose: bool = True,
    ) -> MotionResult:
        return self.pivot(abs(angle_deg), speed, verbose)

    def turn_left(
        self,
        angle_deg: float = 90.0,
        speed: int = cfg.DEFAULT_TURN_SPEED,
        verbose: bool = True,
    ) -> MotionResult:
        return self.pivot(-abs(angle_deg), speed, verbose)

    def run_velocity(
        self,
        left_speed: int,
        right_speed: int,
        duration_s: float,
    ) -> None:
        if duration_s <= 0:
            return

        for value in (left_speed, right_speed):
            if abs(value) > cfg.MAX_TEST_SPEED:
                raise ValueError(
                    f"Test speeds are limited to +/-{cfg.MAX_TEST_SPEED}"
                )

        started = time.monotonic()
        try:
            while time.monotonic() - started < duration_s:
                self._set_logical_speed(left_speed, right_speed)
                time.sleep(cfg.CONTROL_PERIOD_S)
        finally:
            self.stop()


def print_result(result: MotionResult) -> None:
    print(
        f"Requested ticks: L={result.requested_left_ticks:.1f}, "
        f"R={result.requested_right_ticks:.1f}"
    )
    print(
        f"Measured ticks:  L={result.measured_left_ticks}, "
        f"R={result.measured_right_ticks}"
    )
    print(f"Elapsed: {result.elapsed_s:.2f} s")
