import numpy as np
import pytest

from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Temperature, Terrain, TileMap, Vegetation
from railroads.ui.biomes import biome_colors
from railroads.ui.ground import PIXELS_PER_CELL, GroundLayout, cell_colors

LIGHT = (-1.0, -1.0)


def make_map(grid: Grid, terrain: Terrain = Terrain.LAND) -> TileMap:
    height = np.zeros(grid.shape, dtype=np.int16)
    height[:, grid.cols // 2 :] = 1200
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, terrain, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=height,
    )


def test_higher_land_is_drawn_lighter() -> None:
    tile_map = make_map(SquareGrid(rows=4, cols=8, km_per_cell=2.0))

    colors = cell_colors(tile_map, LIGHT)

    low, high = colors[1, 0, :3].astype(int), colors[1, -1, :3].astype(int)
    assert (high > low).all()


def test_water_keeps_its_own_colour_at_any_height() -> None:
    tile_map = make_map(SquareGrid(rows=4, cols=8, km_per_cell=2.0), Terrain.LAKE)

    colors = cell_colors(tile_map, LIGHT)

    assert np.array_equal(colors, biome_colors(tile_map))


@pytest.mark.parametrize(
    "grid",
    [SquareGrid(rows=3, cols=4, km_per_cell=2.0), HexGrid(rows=3, cols=4, km_per_cell=2.0)],
    ids=["square", "hex"],
)
def test_each_pixel_shows_the_colour_of_its_cell(grid: Grid) -> None:
    layout = GroundLayout.of(grid)
    colors = np.arange(grid.rows * grid.cols * 4, dtype=np.uint8).reshape(*grid.shape, 4)
    x, y = grid.center(1, 2)

    rgba = layout.paint(colors)

    pixel = rgba[int(y * PIXELS_PER_CELL), int(x * PIXELS_PER_CELL)]
    assert pixel.tolist() == colors[1, 2].tolist()


def test_pixels_off_the_map_stay_clear() -> None:
    layout = GroundLayout.of(HexGrid(rows=3, cols=4, km_per_cell=2.0))
    colors = np.full((3, 4, 4), 200, dtype=np.uint8)

    rgba = layout.paint(colors)

    assert rgba[0, 0].tolist() == [0, 0, 0, 0]


def test_baked_grid_lines_darken_cell_edges_only() -> None:
    grid = SquareGrid(rows=3, cols=4, km_per_cell=2.0)
    layout = GroundLayout.of(grid)
    colors = np.full((3, 4, 4), 200, dtype=np.uint8)
    middle = (int(1.5 * PIXELS_PER_CELL), int(1.5 * PIXELS_PER_CELL))
    edge = (int(1.5 * PIXELS_PER_CELL), PIXELS_PER_CELL)

    rgba = layout.paint(colors, with_lines=True)

    assert rgba[middle].tolist() == [200, 200, 200, 200]
    assert (rgba[edge][:3] < 200).all()
