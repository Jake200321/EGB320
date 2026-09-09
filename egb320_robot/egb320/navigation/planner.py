"""Planning on the map.

Two jobs, two functions:
  * flood_fill(map, goals, unknown_is_open) -> {cell: distance}   BFS outward from the goal set
  * path_to(map, start, goals, unknown_is_open) -> [cells] | None  walk the flood-fill gradient downhill

Frontier exploration = path_to(map, here, map.frontier_cells(), unknown_is_open=True)
Return to base       = path_to(map, here, [base_cell],           unknown_is_open=False)
Back to a victim     = path_to(map, here, [victim_cell],         unknown_is_open=False)
"""

from typing import Iterable, Optional

from .wall_map import WallMap


def flood_fill(wall_map: WallMap, goals: Iterable, unknown_is_open: bool) -> dict:
    """Classic micromouse flood fill: BFS distances from the goal set over passable edges. TODO"""
    raise NotImplementedError


def path_to(wall_map: WallMap, start, goals: Iterable, unknown_is_open: bool) -> Optional[list]:
    """Shortest cell path from start into the goal set (inclusive of both ends), or None. TODO
    Hint: from `start`, repeatedly step to any neighbour whose distance is exactly one less."""
    raise NotImplementedError


def choose_frontier(wall_map: WallMap, here, distances: Optional[dict] = None):
    """Policy: which frontier next? Nearest-first is the baseline. TODO
    (Two victim markers in one frame is a guard here too: level priority or nearest-first.)"""
    raise NotImplementedError
