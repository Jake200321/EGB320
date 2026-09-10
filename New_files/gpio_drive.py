#!/usr/bin/env python3


import time
from gpiozero import DigitalInputDevice

from robot_drive import RobotDrive
import gpio_input_config as io


class GPIOInputs:
    def __init__(self):
        self.left = self._make(io.LEFT_INPUT_PIN)
        self.forward = self._make(io.FORWARD_INPUT_PIN)
        self.right = self._make(io.RIGHT_INPUT_PIN)

    @staticmethod
    def _make(pin):
        return DigitalInputDevice(
            pin=pin,
            pull_up=io.PULL_UP,
            active_state=io.ACTIVE_HIGH,
            bounce_time=io.BOUNCE_TIME_S,
        )

    def read(self):
        return (
            bool(self.left.is_active),
            bool(self.forward.is_active),
            bool(self.right.is_active),
        )

    def close(self):
        self.left.close()
        self.forward.close()
        self.right.close()


def command_mode(left, forward, right):
    if left and right:
        return 0, 0

    if forward and left:
        return io.ARC_INNER_SPEED, io.ARC_OUTER_SPEED

    if forward and right:
        return io.ARC_OUTER_SPEED, io.ARC_INNER_SPEED

    if forward:
        return io.FORWARD_SPEED, io.FORWARD_SPEED

    if left:
        # Pivot left / counter-clockwise.
        return -io.PIVOT_SPEED, io.PIVOT_SPEED

    if right:
        # Pivot right / clockwise.
        return io.PIVOT_SPEED, -io.PIVOT_SPEED

    return 0, 0


def obstacle_mode(left_obstacle, front_obstacle, right_obstacle):
    if left_obstacle and front_obstacle and right_obstacle:
        return 0, 0

    if front_obstacle:
        if left_obstacle and not right_obstacle:
            # Left and front blocked -> turn right.
            return io.OBSTACLE_TURN_SPEED, -io.OBSTACLE_TURN_SPEED
        if right_obstacle and not left_obstacle:
            # Right and front blocked -> turn left.
            return -io.OBSTACLE_TURN_SPEED, io.OBSTACLE_TURN_SPEED
        # Front only, or both side states equal: default right pivot.
        return io.OBSTACLE_TURN_SPEED, -io.OBSTACLE_TURN_SPEED

    if left_obstacle and not right_obstacle:
        # Steer away from left obstacle.
        return io.ARC_OUTER_SPEED, io.ARC_INNER_SPEED

    if right_obstacle and not left_obstacle:
        # Steer away from right obstacle.
        return io.ARC_INNER_SPEED, io.ARC_OUTER_SPEED

    if left_obstacle and right_obstacle:
        # Corridor: continue straight.
        return io.OBSTACLE_FORWARD_SPEED, io.OBSTACLE_FORWARD_SPEED

    return io.OBSTACLE_FORWARD_SPEED, io.OBSTACLE_FORWARD_SPEED


def main():
    robot = RobotDrive()
    inputs = GPIOInputs()

    print("GPIO robot drive controller")
    print(f"Mode: {io.INPUT_MODE}")
    print(
        f"BCM pins: LEFT={io.LEFT_INPUT_PIN}, "
        f"FORWARD={io.FORWARD_INPUT_PIN}, RIGHT={io.RIGHT_INPUT_PIN}"
    )
    print("Ctrl+C to stop.\n")

    last_state = None
    last_command = None

    try:
        while True:
            state = inputs.read()
            left, forward, right = state

            if io.INPUT_MODE == "command":
                command = command_mode(left, forward, right)
            elif io.INPUT_MODE == "obstacle":
                command = obstacle_mode(left, forward, right)
            else:
                raise ValueError(
                    "INPUT_MODE must be either 'command' or 'obstacle'"
                )

            # Repeat continuously so the HAT watchdog remains satisfied.
            robot.set_track_speeds(*command)

            if state != last_state or command != last_command:
                print(
                    f"inputs L/F/R={state}  ->  "
                    f"tracks L/R={command}"
                )
                last_state = state
                last_command = command

            time.sleep(io.LOOP_PERIOD_S)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        robot.stop()
        inputs.close()
        print("Robot stopped.")


if __name__ == "__main__":
    main()
