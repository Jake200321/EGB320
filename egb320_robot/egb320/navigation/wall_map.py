"""The map store: what nav remembers about the maze.

Cells are (column, row); walls are shared between neighbouring cells and are one of
True (present) / False (absent) / None (unknown). The perimeter is pre-seeded as
present. Also tracks: visited cells, blocked cells (tried to enter, couldn't),
hazard cells (dead-end placard seen — never path there), and remembered victim cells.

Everything the rescue phase needs to "go back" lives here.
"""

from enum import IntEnum


class Heading(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def opposite(self): return Heading((self + 2) % 4)
    @property
    def left(self):     return Heading((self - 1) % 4)
    @property
    def right(self):    return Heading((self + 1) % 4)


# (dcol, drow) for one cell step. Row 0 is the NORTH edge, so NORTH decreases the row.
DIR_VECTORS = {Heading.NORTH: (0, -1), Heading.EAST: (1, 0),
               Heading.SOUTH: (0, 1), Heading.WEST: (-1, 0)}


class WallMap:
    def __init__(self, columns: int, rows: int):
        self.columns, self.rows = columns, rows
        self._walls = {}        # ((col, row), Heading) -> bool
        self.visited = set()
        self.blocked = set()
        self.hazards = set()
        self.victims = {}       # label/cell -> (col, row) remembered for the rescue phase
        self._seed_perimeter()

    # ------------------------------------------------------------------ helpers
    def in_bounds(self, cell) -> bool:
        c, r = cell
        return 0 <= c < self.columns and 0 <= r < self.rows

    def neighbour(self, cell, heading: Heading):
        dc, dr = DIR_VECTORS[heading]
        return (cell[0] + dc, cell[1] + dr)

    # ------------------------------------------------------------------ TODO: storage
    def _seed_perimeter(self) -> None:
        """Mark every outer edge as a wall (called from __init__). TODO"""
        pass  # TODO: for every edge cell, set_wall(cell, outward heading, True)

    def set_wall(self, cell, heading: Heading, present: bool) -> None:
        """Record a wall AND its mirror on the neighbouring cell (walls are shared). TODO"""
        raise NotImplementedError

    def wall(self, cell, heading: Heading):
        """True / False / None (unknown)."""
        return self._walls.get((cell, heading))

    # ------------------------------------------------------------------ TODO: queries
    def passable(self, cell, heading: Heading, unknown_is_open: bool) -> bool:
        """Can the robot step from `cell` in `heading`? Blocked and hazard cells are never passable.
        `unknown_is_open` = optimistic (exploring) vs pessimistic (returning over known map). TODO"""
        raise NotImplementedError

    def frontier_cells(self) -> set:
        """Unvisited, unblocked cells reachable from a visited cell through a non-wall edge —
        the boundary between explored and unexplored space. TODO"""
        raise NotImplementedError
