import time
from controller import Controller

motor_driver = Controller()

speed = -100  # Valid range: -127 to 127

try:
    # Read the encoder values before starting
    start_left, start_right = motor_driver.get_encoder_ticks()

    print("Starting motor test")
    print("Press Ctrl+C to stop")

    # Run both motors
    motor_driver.set_raw_motor_speed(speed, speed)

    for i in range(30):
        left, right = motor_driver.get_encoder_ticks()

        # Display the encoder movement since the test started
        print(
            "Left:",
            left - start_left,
            "Right:",
            right - start_right
        )

        time.sleep(0.1)

finally:
    # Always stop the motors when the program finishes
    motor_driver.set_raw_motor_speed(0, 0)
    print("Motors stopped")