"""Start here: these tests describe the behaviour before you write it (red -> green)."""

import pytest

from egb320.navigation.wall_map import Heading, WallMap


def test_headings_rotate():
    assert Heading.NORTH.right is Heading.EAST
    assert Heading.NORTH.left is Heading.WEST
    assert Heading.EAST.opposite is Heading.WEST


@pytest.mark.skip(reason="TODO: implement WallMap._seed_perimeter / set_wall")
def test_perimeter_is_seeded():
    m = WallMap(7, 7)
    assert m.wall((0, 0), Heading.NORTH) is True
    assert m.wall((6, 6), Heading.EAST) is True
    assert m.wall((3, 3), Heading.NORTH) is None


@pytest.mark.skip(reason="TODO: implement WallMap.set_wall")
def test_walls_are_shared_between_neighbours():
    m = WallMap(7, 7)
    m.set_wall((2, 2), Heading.EAST, True)
    assert m.wall((3, 2), Heading.WEST) is True
