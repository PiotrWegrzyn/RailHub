import numpy as np
import pytest

from builders import flat_map
from railroads.sim.grids import SquareGrid
from railroads.sim.tilemap import Relief, Terrain, TileMap
from railroads.ui.relief import HILL_BRIGHTNESS, relief_brightness


def make_map(relief: list[list[Relief]], terrain: Terrain = Terrain.LAND) -> TileMap:
    levels = np.array(relief, dtype=np.uint8)
    grid = SquareGrid(rows=levels.shape[0], cols=levels.shape[1], km_per_cell=2.0)
    return flat_map(grid=grid, terrain=np.full(grid.shape, terrain, dtype=np.uint8), relief=levels)


def test_hills_are_drawn_lighter_than_flat_land_and_mountains() -> None:
    tile_map = make_map([[Relief.FLAT, Relief.HILLS, Relief.MOUNTAINS]])

    brightness = relief_brightness(tile_map)

    assert brightness[0].tolist() == pytest.approx([1.0, HILL_BRIGHTNESS, 1.0])


def test_water_is_never_lightened() -> None:
    tile_map = make_map([[Relief.HILLS, Relief.HILLS]], Terrain.LAKE)

    brightness = relief_brightness(tile_map)

    assert brightness.tolist() == [[1.0, 1.0]]
