#! /usr/bin/env python3

# Check that the microcontroller behaves as expected by exchanging
# information with the microcontroller.

import controller
import time


def check_firmware():
    c = controller.Controller()
    print("- Restart the board")
    c.reset()
    time.sleep(0.1)
    print("- Check that the encoders have been reset")
    assert c.get_encoder_ticks() == (0, 0)
    print("- Check that motors are in standby mode")
    assert c.get_raw_motor_speed() is None
    print("- Setting the motors shutdown timeout to 0.2 seconds")
    c.set_motor_shutdown_timeout(0.2)
    print("- Check that the motor speed can be set and retrieved")
    c.set_raw_motor_speed(30, -40)
    assert c.get_raw_motor_speed() == (30, -40)
    print("- Check that it does not stop after 0.1 second")
    time.sleep(0.1)
    assert c.get_raw_motor_speed() == (30, -40)
    print("- Check that it stops after 0.2 second without command")
    time.sleep(0.25)
    assert c.get_raw_motor_speed() is None
    print("- Check that encoders have moved in the expected direction")
    time.sleep(0.5)  # Let some time to stop
    (left, right) = c.get_encoder_ticks()
    assert left > 0 and right < 0
    print("- Check that encoders are reset after being read")
    assert c.get_encoder_ticks() == (0, 0)
    print("- Check that PID coefficients can be set and read")
    c.set_pid_coefficients(0, 0, 0)
    assert c.get_pid_coefficients() == ((0, 0), (0, 0), (0, 0))
    c.set_pid_coefficients(0.25, 0.125, -0.0625)
    assert c.get_pid_coefficients() == (
        (0.25, 0.25),
        (0.125, 0.125),
        (-0.0625, -0.0625),
    )
    print("- Check that the shutdown timeout triggers in controlled mode")
    c.set_motor_shutdown_timeout(0.2)
    c.set_pid_coefficients(4, 0, 0)
    c.set_motor_speed(30, -30)
    print("- Check that controlled motor speed can be read back")
    assert c.get_motor_speed() == (30, -30)
    print("- Check that raw motor speed can be read in controlled mode")
    time.sleep(0.1)
    (left, right) = c.get_raw_motor_speed()
    assert left > 0 and right < 0
    print("- Check that the shutdown timeout triggers in controlled mode")
    time.sleep(0.25)
    assert c.get_raw_motor_speed() is None
    print("- Check that a positive P goes into the right direction")
    c.set_motor_shutdown_timeout(1.0)
    c.get_encoder_ticks()  # Reset ticks
    c.set_motor_speed(30, -30)
    time.sleep(1.5)  # Let some time to stop
    (left, right) = c.get_encoder_ticks()
    assert left > 0 and right < 0
    print("- Check that a negative P coefficient reverses the direction")
    c.set_pid_coefficients(-2, 0, 0)
    c.set_motor_speed(30, -30)
    time.sleep(1.5)  # Let some time to stop
    (left, right) = c.get_encoder_ticks()
    print("Negative P encoder values:", left, right)
    assert left < 0 and right > 0
    print("- Check that a positive I goes into the right direction")
    c.set_pid_coefficients(0, 10 / 256, 0)
    c.set_motor_speed(30, -30)
    time.sleep(1.5)
    (left, right) = c.get_encoder_ticks()
    assert left > 0 and right < 0
    print("- Check that the I error accumulator is reset in non-controlled mode")
    assert c.get_pid_i_accumulators() == (0, 0)
    print("- Check that a positive I stays behind the target")
    c.set_motor_speed(30, 30)
    time.sleep(0.5)
    (left, right) = c.get_pid_i_accumulators()
    assert left > 0 and right > 0
    time.sleep(1)


if __name__ == "__main__":
    check_firmware()
