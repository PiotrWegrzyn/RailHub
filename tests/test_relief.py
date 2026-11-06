import numpy as np
import pytest

from railroads.sim.geo import Bounds
from railroads.sim.grids import SquareGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.ui.relief import HILL_BRIGHTNESS, relief_brightness


def make_map(relief: list[list[Relief]], terrain: Terrain = Terrain.LAND) -> TileMap:
    levels = np.array(relief, dtype=np.uint8)
    grid = SquareGrid(rows=levels.shape[0], cols=levels.shape[1], km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, terrain, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=levels,
    )


def test_hills_are_drawn_lighter_than_flat_land_and_mountains() -> None:
    tile_map = make_map([[Relief.FLAT, Relief.HILLS, Relief.MOUNTAINS]])

    brightness = relief_brightness(tile_map)

    assert brightness[0].tolist() == pytest.approx([1.0, HILL_BRIGHTNESS, 1.0])


def test_water_is_never_lightened() -> None:
    tile_map = make_map([[Relief.HILLS, Relief.HILLS]], Terrain.LAKE)

    brightness = relief_brightness(tile_map)

    assert brightness.tolist() == [[1.0, 1.0]]
