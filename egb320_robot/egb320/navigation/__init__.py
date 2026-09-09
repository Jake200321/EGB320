"""Navigation subsystem — Jake.

Layers, bottom to top (each only talks to the one below it):

    localisation.py   ticks / odometry -> Pose2D, snapped to the grid using walls
    wall_map.py       what the maze looks like so far: walls, visited, blocked, hazards
    planner.py        flood fill (known map -> path) and frontier selection (where next?)
    motion.py         primitives: turn_to(heading), drive_one_cell(), approach(bearing)
    state_machine.py  the rules: which state, which LED, which transition
    mission.py        the loop: tick sensors -> step state machine -> command subsystems

Design decisions already made (Nav system/Algorithm comparison):
  * frontier exploration for the SEARCH phase
  * flood fill over the discovered map for every GOAL-DIRECTED leg (return home, back to a victim)
  * wall following survives only as a low-level fallback if the map degrades
"""
