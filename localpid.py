#! /usr/bin/env python3
#
# Basic PID test

from controller import Controller
import time

k_p = 0.25
k_i = 0.004
k_d = 0

min_speed = 0
precision = 10


# distance and max_speed are in ticks, max_motor_speed
def advance(c, distance, max_speed):

    # Current positions
    pos_left = 0
    pos_right = 0

    # Error
    prev_err_left, prev_err_right = 0, 0
    acc_err_left, acc_err_right = 0, 0

    # Reset ticks
    c.get_encoder_ticks()

    # Go until distance is reached
    final_target = distance
    target = 0
    stable_for = 0
    ramp_up_speed = 0
    while True:
        # Ramp-up the speed on four second
        ramp_up_speed = min(ramp_up_speed + max_speed / 400, max_speed)
        acceptable_speed = round(ramp_up_speed)
        if final_target > 0:
            target = min(target + acceptable_speed, final_target)
        else:
            target = max(target - acceptable_speed, final_target)
        time.sleep(0.01)  # Every 1/100th of seconds
        enc_left, enc_right = c.get_encoder_ticks()
        pos_left, pos_right = pos_left + enc_left, pos_right + enc_right

        if (
            abs(pos_left - final_target) <= precision
            and abs(pos_right - final_target) <= precision
        ):
            stable_for += 1
            if stable_for >= 10:
                c.set_raw_motor_speed(None, None)
                break
        else:
            stable_for = 0

        err_left, err_right = target - pos_left, target - pos_right
        der_err_left, der_err_right = (
            err_left - prev_err_left,
            err_right - prev_err_right,
        )
        acc_err_left, acc_err_right = (
            acc_err_left + err_left,
            acc_err_right + err_right,
        )
        prev_err_left, prev_err_right = err_left, err_right
        command_left = command(err_left, acc_err_left, der_err_left)
        command_right = command(err_right, acc_err_right, der_err_right)
        c.set_raw_motor_speed(command_left, command_right)
        print(
            f"target = {target}, pos = {(pos_left, pos_right)},",
            f"speed = {(command_left, command_right)}",
        )


def command(err, acc, der):
    speed = round(err * k_p + acc * k_i + der * k_d)
    if speed < -100:
        speed = -100
    elif speed > -min_speed and speed < 0:
        speed = -min_speed
    elif speed > 0 and speed < min_speed:
        speed = min_speed
    elif speed > 100:
        speed = 100
    return speed


c = Controller()
assert c.who_am_i() == 0x57
try:
    advance(c, 20000, 40)
    # advance(c, -10000, 50)
except KeyboardInterrupt:
    print("Stopping engine")
    c.set_raw_motor_speed(None, None)
