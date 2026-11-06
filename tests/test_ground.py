import numpy as np
import pytest

from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.ui.biomes import biome_colors
from railroads.ui.ground import PIXELS_PER_CELL, GroundLayout, cell_colors

FROM_WEST = (-1.0, 0.0)


def make_map(grid: Grid, terrain: Terrain = Terrain.LAND) -> TileMap:
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    relief[:, grid.cols // 2 :] = Relief.HILLS
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, terrain, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=relief,
    )


def pixel(grid: Grid, x: float, y: float) -> tuple[int, int]:
    return int(y * PIXELS_PER_CELL), int(x * PIXELS_PER_CELL)


def test_hills_are_drawn_lighter_than_flat_land() -> None:
    tile_map = make_map(SquareGrid(rows=4, cols=8, km_per_cell=2.0))

    colors = cell_colors(tile_map)

    flat, hills = colors[1, 0, :3].astype(int), colors[1, -1, :3].astype(int)
    assert (hills > flat).all()


def test_water_keeps_its_own_colour() -> None:
    tile_map = make_map(SquareGrid(rows=4, cols=8, km_per_cell=2.0), Terrain.LAKE)

    colors = cell_colors(tile_map)

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

    assert rgba[pixel(grid, x, y)].tolist() == colors[1, 2].tolist()


def test_pixels_off_the_map_stay_clear() -> None:
    layout = GroundLayout.of(HexGrid(rows=3, cols=4, km_per_cell=2.0))
    colors = np.full((3, 4, 4), 200, dtype=np.uint8)

    rgba = layout.paint(colors)

    assert rgba[0, 0].tolist() == [0, 0, 0, 0]


def test_raised_edges_are_lit_on_the_side_facing_the_light() -> None:
    grid = SquareGrid(rows=3, cols=3, km_per_cell=2.0)
    layout = GroundLayout.of(grid)
    colors = np.full((3, 3, 4), 120, dtype=np.uint8)
    relief = np.zeros((3, 3), dtype=np.uint8)
    relief[1, 1] = Relief.HILLS

    rgba = layout.paint(colors, relief, FROM_WEST)

    west_edge, centre, east_edge = (rgba[pixel(grid, x, 1.5)][0] for x in (1.05, 1.5, 1.95))
    assert west_edge > centre > east_edge


def test_baked_grid_lines_darken_cell_edges_only() -> None:
    grid = SquareGrid(rows=3, cols=4, km_per_cell=2.0)
    layout = GroundLayout.of(grid)
    colors = np.full((3, 4, 4), 200, dtype=np.uint8)

    rgba = layout.paint(colors, with_lines=True)

    assert rgba[pixel(grid, 1.5, 1.5)].tolist() == [200, 200, 200, 200]
    assert (rgba[pixel(grid, 1.5, 1.01)][:3] < 200).all()
