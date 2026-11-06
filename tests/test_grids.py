import math

import numpy as np
import pytest

from railroads.sim.grids import HEX_RADIUS, Grid, HexGrid, SquareGrid

GRIDS = [SquareGrid(rows=6, cols=7, km_per_cell=2.0), HexGrid(rows=6, cols=7, km_per_cell=2.0)]
EACH_GRID = pytest.mark.parametrize("grid", GRIDS, ids=lambda grid: grid.kind)


@EACH_GRID
def test_every_cell_centre_lies_in_its_own_cell(grid: Grid) -> None:
    rows, cols = np.indices(grid.shape)
    x, y = grid.center(rows, cols)

    found_rows, found_cols = grid.cell_at(x, y)

    assert np.array_equal(found_rows, rows)
    assert np.array_equal(found_cols, cols)


@EACH_GRID
def test_sample_points_stay_inside_their_cell(grid: Grid) -> None:
    x, y = grid.center(2, 3)
    offsets = grid.sample_offsets()

    rows, cols = grid.cell_at(x + offsets[:, 0], y + offsets[:, 1])

    assert set(zip(rows.tolist(), cols.tolist(), strict=True)) == {(2, 3)}


@EACH_GRID
def test_neighbours_are_one_unit_away(grid: Grid) -> None:
    centre = grid.center(3, 3)

    distances = [math.dist(centre, grid.center(*cell)) for cell in grid.neighbours(3, 3)]

    assert distances == pytest.approx([1.0] * len(distances))


@EACH_GRID
def test_neighbours_are_mutual(grid: Grid) -> None:
    cells = [(row, col) for row in range(grid.rows) for col in range(grid.cols)]

    one_sided = [
        (cell, other)
        for cell in cells
        for other in grid.neighbours(*cell)
        if cell not in grid.neighbours(*other)
    ]

    assert one_sided == []


def test_hexagons_have_six_neighbours_and_squares_four() -> None:
    square, hexagon = GRIDS

    counts = (
        len(square.neighbours(3, 3)),
        len(hexagon.neighbours(3, 3)),
        len(hexagon.neighbours(2, 3)),
    )

    assert counts == (4, 6, 6)


def test_hexagon_corners_are_one_radius_from_the_centre() -> None:
    grid = HexGrid(rows=6, cols=7, km_per_cell=2.0)
    centre = grid.center(2, 4)

    distances = [math.dist(centre, corner) for corner in grid.corners(2, 4)]

    assert distances == pytest.approx([HEX_RADIUS] * 6)


def test_squares_touching_only_at_a_corner_get_sealed() -> None:
    grid = SquareGrid(rows=2, cols=2, km_per_cell=2.0)
    mask = np.array([[True, False], [False, True]])

    sealed = grid.seal_corner_gaps(mask)

    assert sealed.sum() == 3
    assert sealed[0, 0] and sealed[1, 1]


def test_hexagons_never_touch_only_at_a_corner() -> None:
    grid = HexGrid(rows=2, cols=2, km_per_cell=2.0)
    mask = np.array([[True, False], [False, True]])

    sealed = grid.seal_corner_gaps(mask)

    assert np.array_equal(sealed, mask)


@pytest.mark.parametrize("grid_type", [SquareGrid, HexGrid], ids=["square", "hex"])
def test_grid_covering_an_area_spans_it(grid_type: type[Grid]) -> None:
    grid = grid_type.covering(width_km=100, height_km=60, km_per_cell=2.0)

    width, height = grid.size()

    assert width * 2.0 == pytest.approx(100, abs=2.0)
    assert height * 2.0 == pytest.approx(60, abs=2.0)


@EACH_GRID
def test_cell_centre_is_half_a_unit_from_the_edge(grid: Grid) -> None:
    x, y = grid.center(2, 3)

    distance = grid.edge_distance(x, y)

    assert distance == pytest.approx(0.5)


@EACH_GRID
def test_corners_lie_on_the_edge(grid: Grid) -> None:
    corners = np.array(grid.corners(2, 3))

    distances = grid.edge_distance(corners[:, 0], corners[:, 1])

    assert distances == pytest.approx(np.zeros(len(corners)), abs=1e-9)


@EACH_GRID
def test_points_halfway_to_each_neighbour_lie_on_the_edge(grid: Grid) -> None:
    centre = np.array(grid.center(2, 3))
    halfway = np.array([(centre + grid.center(*cell)) / 2 for cell in grid.neighbours(2, 3)])

    distances = grid.edge_distance(halfway[:, 0], halfway[:, 1])

    assert distances == pytest.approx(np.zeros(len(halfway)), abs=1e-9)
