import math

import numpy as np
import numpy.typing as npt
import pytest

from builders import flat_map
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Relief, Terrain, TileMap
from railroads.ui import ground
from railroads.ui.biomes import biome_colors
from railroads.ui.ground import (
    BAKED_LINE_SHADE,
    EDGE_STEPS_PER_PIXEL,
    PIXELS_PER_CELL,
    GroundLayout,
    GroundPictures,
    Pixels,
    cell_colors,
)
from railroads.ui.relief import BEVEL_PIXELS, BEVEL_STRENGTH

FROM_WEST = (-1.0, 0.0)
FROM_NORTH_EAST = (0.6, -0.8)
# Hex and square grids from a single cell up, two of them taller than one pass of rows.
GRIDS = [
    HexGrid(rows=1, cols=1, km_per_cell=2.0),
    SquareGrid(rows=1, cols=1, km_per_cell=2.0),
    HexGrid(rows=3, cols=5, km_per_cell=2.0),
    SquareGrid(rows=5, cols=2, km_per_cell=2.0),
    HexGrid(rows=23, cols=17, km_per_cell=2.0),
    SquareGrid(rows=19, cols=26, km_per_cell=2.0),
    HexGrid(rows=40, cols=9, km_per_cell=2.0),
    SquareGrid(rows=33, cols=12, km_per_cell=2.0),
]


def grid_id(grid: Grid) -> str:
    return f"{grid.kind} {grid.rows}x{grid.cols}"


def make_map(grid: Grid, terrain: Terrain = Terrain.LAND) -> TileMap:
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    relief[:, grid.cols // 2 :] = Relief.HILLS
    return flat_map(grid=grid, terrain=np.full(grid.shape, terrain, dtype=np.uint8), relief=relief)


def pixel(grid: Grid, x: float, y: float) -> tuple[int, int]:
    return int(y * PIXELS_PER_CELL), int(x * PIXELS_PER_CELL)


def flat_pictures(grid: Grid) -> GroundPictures:
    return GroundPictures(GroundLayout.of(grid), np.zeros(grid.shape, dtype=np.uint8))


def random_ground(
    grid: Grid,
) -> tuple[GroundLayout, npt.NDArray[np.uint8], Pixels, tuple[float, float]]:
    """A grid's layout, with relief of three levels, cell colours and a light all at random."""
    rng = np.random.default_rng(7)
    relief = rng.integers(0, 3, size=grid.shape, dtype=np.uint8)
    colors = rng.integers(0, 256, size=(*grid.shape, 4), dtype=np.uint8)
    light_x, light_y = rng.normal(size=2)
    return GroundLayout.of(grid), relief, colors, (float(light_x), float(light_y))


def layout_pixel_by_pixel(grid: Grid) -> GroundLayout:
    """The ground layout worked out for every pixel on its own, all at once."""
    per_cell = ground.PIXELS_PER_CELL
    width, height = grid.size()
    xs = (np.arange(math.ceil(width * per_cell)) + 0.5) / per_cell
    ys = (np.arange(math.ceil(height * per_cell)) + 0.5) / per_cell
    px, py = np.meshgrid(xs, ys)
    rows, cols = grid.cell_at(px, py)
    cx, cy = grid.center(rows, cols)
    normals = grid.edge_normals()
    reach = (px - cx)[..., None] * normals[:, 0] + (py - cy)[..., None] * normals[:, 1]
    nearest = reach.argmax(axis=-1)
    to_edge = (0.5 - reach.max(axis=-1)) * per_cell * EDGE_STEPS_PER_PIXEL
    other_rows, other_cols = grid.cell_at(cx + normals[nearest, 0], cy + normals[nearest, 1])
    return GroundLayout(
        cell_index(grid, rows, cols),
        cell_index(grid, other_rows, other_cols),
        nearest.astype(np.uint8),
        np.clip(np.rint(to_edge), 0, 255).astype(np.uint8),
        normals,
        (len(xs) / per_cell, len(ys) / per_cell),
    )


def cell_index(grid: Grid, rows: npt.NDArray[np.int64], cols: npt.NDArray[np.int64]) -> npt.NDArray:
    return np.where(grid.contains(rows, cols), rows * grid.cols + cols, -1).astype(np.int32)


def painted_pixel_by_pixel(
    layout: GroundLayout,
    colors: Pixels,
    relief: npt.NDArray[np.uint8],
    light_from: tuple[float, float],
    with_lines: bool,
) -> Pixels:
    """The ground picture painted from scratch, every step over the whole picture."""
    rgba = np.vstack([colors.reshape(-1, 4), np.zeros((1, 4), dtype=np.uint8)])[layout.cells]
    width = BEVEL_PIXELS * EDGE_STEPS_PER_PIXEL
    levels = relief.ravel()
    near = (layout.edge_distance < width) & (layout.cells >= 0) & (layout.across >= 0)
    raised = near.copy()
    raised[near] = levels[layout.cells[near]] > levels[layout.across[near]]
    light_x, light_y = np.array(light_from) / math.hypot(*light_from)
    normals = layout.normals[layout.edge_direction[raised]]
    facing = normals[:, 0] * light_x + normals[:, 1] * light_y
    shade = 1 + BEVEL_STRENGTH * facing * (1 - layout.edge_distance[raised] / width)
    rgba[raised, :3] = np.clip(rgba[raised, :3] * shade[:, None], 0, 255).astype(np.uint8)
    if with_lines:
        on_edge = np.clip(1 - layout.edge_distance / EDGE_STEPS_PER_PIXEL, 0, 1)
        keep = 1 - BAKED_LINE_SHADE * on_edge[..., None]
        rgba[..., :3] = (rgba[..., :3] * keep).astype(np.uint8)
    return rgba


def assert_same_layout(layout: GroundLayout, expected: GroundLayout) -> None:
    for field in ("cells", "across", "edge_direction", "edge_distance", "normals"):
        np.testing.assert_array_equal(
            getattr(layout, field), getattr(expected, field), err_msg=field, strict=True
        )
    assert layout.size == expected.size


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
    pictures = flat_pictures(grid)
    colors = np.arange(grid.rows * grid.cols * 4, dtype=np.uint8).reshape(*grid.shape, 4)
    x, y = grid.center(1, 2)

    rgba = pictures.picture(colors, FROM_WEST)

    assert rgba[pixel(grid, x, y)].tolist() == colors[1, 2].tolist()


def test_pixels_off_the_map_stay_clear() -> None:
    pictures = flat_pictures(HexGrid(rows=3, cols=4, km_per_cell=2.0))
    colors = np.full((3, 4, 4), 200, dtype=np.uint8)

    rgba = pictures.picture(colors, FROM_WEST)

    assert rgba[0, 0].tolist() == [0, 0, 0, 0]


def test_raised_edges_are_lit_on_the_side_facing_the_light() -> None:
    grid = SquareGrid(rows=3, cols=3, km_per_cell=2.0)
    colors = np.full((3, 3, 4), 120, dtype=np.uint8)
    relief = np.zeros((3, 3), dtype=np.uint8)
    relief[1, 1] = Relief.HILLS
    pictures = GroundPictures(GroundLayout.of(grid), relief)

    rgba = pictures.picture(colors, FROM_WEST)

    west_edge, centre, east_edge = (rgba[pixel(grid, x, 1.5)][0] for x in (1.05, 1.5, 1.95))
    assert west_edge > centre > east_edge


def test_baked_grid_lines_darken_cell_edges_only() -> None:
    grid = SquareGrid(rows=3, cols=4, km_per_cell=2.0)
    pictures = flat_pictures(grid)
    colors = np.full((3, 4, 4), 200, dtype=np.uint8)

    rgba = pictures.picture(colors, FROM_WEST, with_lines=True)

    assert rgba[pixel(grid, 1.5, 1.5)].tolist() == [200, 200, 200, 200]
    assert (rgba[pixel(grid, 1.5, 1.01)][:3] < 200).all()


@pytest.mark.parametrize("grid", GRIDS, ids=grid_id)
def test_layout_matches_working_out_every_pixel(grid: Grid) -> None:
    expected = layout_pixel_by_pixel(grid)

    layout = GroundLayout.of(grid)

    assert_same_layout(layout, expected)


@pytest.mark.parametrize(
    ("per_cell", "grid"),
    [
        (1, HexGrid(rows=12, cols=12, km_per_cell=2.0)),
        (6, SquareGrid(rows=4, cols=4, km_per_cell=2.0)),
    ],
    ids=["hex, 1 pixel a cell", "square, 6 pixels a cell"],
)
def test_layout_is_exact_for_cell_widths_it_does_not_repeat_at(
    per_cell: int, grid: Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ground, "PIXELS_PER_CELL", per_cell)
    expected = layout_pixel_by_pixel(grid)

    layout = GroundLayout.of(grid)

    assert_same_layout(layout, expected)


@pytest.mark.parametrize("with_lines", [False, True], ids=["plain", "lined"])
@pytest.mark.parametrize("grid", GRIDS, ids=grid_id)
def test_pictures_match_painting_every_pixel(grid: Grid, with_lines: bool) -> None:
    layout, relief, colors, light = random_ground(grid)
    expected = painted_pixel_by_pixel(layout, colors, relief, light, with_lines)

    rgba = GroundPictures(layout, relief).picture(colors, light, with_lines)

    np.testing.assert_array_equal(rgba, expected, strict=True)


def test_picture_without_lines_after_one_with_lines_has_no_lines() -> None:
    layout, relief, colors, _ = random_ground(HexGrid(rows=23, cols=17, km_per_cell=2.0))
    pictures = GroundPictures(layout, relief)
    pictures.picture(colors, FROM_WEST, with_lines=True)
    expected = painted_pixel_by_pixel(layout, colors, relief, FROM_WEST, with_lines=False)

    rgba = pictures.picture(colors, FROM_WEST)

    np.testing.assert_array_equal(rgba, expected)


def test_picture_with_lines_follows_colours_changed_in_place() -> None:
    layout, relief, colors, _ = random_ground(HexGrid(rows=23, cols=17, km_per_cell=2.0))
    pictures = GroundPictures(layout, relief)
    pictures.picture(colors, FROM_WEST)
    colors[5, 5] = (1, 2, 3, 255)
    expected = painted_pixel_by_pixel(layout, colors, relief, FROM_WEST, with_lines=True)

    rgba = pictures.picture(colors, FROM_WEST, with_lines=True)

    np.testing.assert_array_equal(rgba, expected)


def test_picture_with_lines_follows_a_new_light() -> None:
    layout, relief, colors, _ = random_ground(HexGrid(rows=23, cols=17, km_per_cell=2.0))
    pictures = GroundPictures(layout, relief)
    pictures.picture(colors, FROM_WEST)
    expected = painted_pixel_by_pixel(layout, colors, relief, FROM_NORTH_EAST, with_lines=True)

    rgba = pictures.picture(colors, FROM_NORTH_EAST, with_lines=True)

    np.testing.assert_array_equal(rgba, expected)
