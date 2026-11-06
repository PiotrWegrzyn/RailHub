import numpy as np
import pytest

from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Temperature, Terrain, TileMap, Vegetation
from railroads.ui.relief import height_levels, hill_shade

FROM_WEST = (-1.0, 0.0)
FROM_EAST = (1.0, 0.0)


def make_map(height_m: np.ndarray, grid: Grid | None = None) -> TileMap:
    grid = grid or SquareGrid(rows=height_m.shape[0], cols=height_m.shape[1], km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=height_m.astype(np.int16),
    )


def rising_east(rows: int = 5, cols: int = 6) -> np.ndarray:
    return np.tile(np.arange(cols) * 100, (rows, 1))


def test_heights_fall_into_eleven_levels() -> None:
    heights = np.array([-5, 0, 49, 50, 99, 2999, 3000, 8848])

    levels = height_levels(heights)

    assert levels.tolist() == [0, 0, 0, 1, 1, 9, 10, 10]


def test_flat_ground_keeps_its_brightness() -> None:
    tile_map = make_map(np.full((4, 4), 250))

    shade = hill_shade(tile_map, FROM_WEST)

    assert shade == pytest.approx(np.ones((4, 4)))


@pytest.mark.parametrize(
    "grid",
    [SquareGrid(rows=5, cols=6, km_per_cell=2.0), HexGrid(rows=5, cols=6, km_per_cell=2.0)],
    ids=["square", "hex"],
)
def test_slopes_facing_the_light_are_brighter(grid: Grid) -> None:
    tile_map = make_map(rising_east(), grid)

    lit_from_west, lit_from_east = hill_shade(tile_map, FROM_WEST), hill_shade(tile_map, FROM_EAST)

    assert (lit_from_west > 1).all()
    assert (lit_from_east < 1).all()


def test_the_sea_counts_as_flat_ground_at_sea_level() -> None:
    tile_map = make_map(np.array([[-500, -100, 0, 0]] * 3))

    shade = hill_shade(tile_map, FROM_WEST)

    assert shade == pytest.approx(np.ones((3, 4)))
