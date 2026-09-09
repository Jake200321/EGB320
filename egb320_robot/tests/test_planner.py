import pytest

from egb320.navigation.planner import flood_fill, path_to
from egb320.navigation.wall_map import Heading, WallMap


@pytest.mark.skip(reason="TODO: implement flood_fill / path_to")
def test_flood_fill_open_grid_is_manhattan_distance():
    m = WallMap(3, 3)
    dist = flood_fill(m, goals=[(0, 0)], unknown_is_open=True)
    assert dist[(2, 2)] == 4


@pytest.mark.skip(reason="TODO: implement flood_fill / path_to")
def test_path_routes_around_a_wall():
    m = WallMap(3, 1)
    m.set_wall((0, 0), Heading.EAST, True)          # no direct route 0->1
    assert path_to(m, (0, 0), [(2, 0)], unknown_is_open=False) is None
