

LEFT_INPUT_PIN = 17
FORWARD_INPUT_PIN = 27
RIGHT_INPUT_PIN = 22


ACTIVE_HIGH = False
PULL_UP = True

BOUNCE_TIME_S = 0.03

# HAT controlled-speed units: encoder ticks per 0.01 seconds
FORWARD_SPEED = 5
PIVOT_SPEED = 4
ARC_INNER_SPEED = 2
ARC_OUTER_SPEED = 5

# Main-loop refresh. 
LOOP_PERIOD_S = 0.05

# "command"  -> inputs request movement.
# "obstacle" -> inputs mean an obstacle is detected on that side.
INPUT_MODE = "command"

# Obstacle-mode defaults.
OBSTACLE_FORWARD_SPEED = 4
OBSTACLE_TURN_SPEED = 4
