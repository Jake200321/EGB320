"""EGB320 Team 12 — Search and Rescue Maze robot software.

Package layout (one folder per assessed subsystem + the glue that joins them):

    egb320/interfaces/   the contracts every subsystem talks through (dataclasses + Protocols)
    egb320/navigation/   Jake   — state machine, map, planner, localisation, motion, mission
    egb320/mobility/     Dan    — motor driver HAT + encoders  (drops in behind MobilityInterface)
    egb320/vision/       Kushal — Pi camera + OpenCV            (drops in behind VisionInterface)
    egb320/rescue/       Roger  — collection mechanism          (drops in behind RescueInterface)
    egb320/leds/         Jake   — yellow / green / red status LEDs
    egb320/backends/     adapters that build the four interfaces for: sim | mock | robot
    egb320/utils/        clock + mission black-box log
"""

__version__ = "0.1.0"
