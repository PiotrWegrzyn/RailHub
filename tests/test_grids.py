import math

import numpy as np
import pytest

from railroads.sim.grids import HEX_RADIUS, Grid, HexGrid, SquareGrid, flat_within

GRIDS = [SquareGrid(rows=6, cols=7, km_per_cell=2.0), HexGrid(rows=6, cols=7, km_per_cell=2.0)]
EACH_GRID = pytest.mark.parametrize("grid", GRIDS, ids=lambda grid: grid.kind)


@EACH_GRID
@pytest.mark.parametrize("cell", [(3, 3), (2, 4), (0, 0), (5, 6)])
def test_neighbour_grids_list_the_same_neighbours(grid: Grid, cell: tuple[int, int]) -> None:
    rows, cols = grid.neighbour_index_grids()
    row, col = cell

    listed = {(int(r), int(c)) for r, c in zip(rows[:, row, col], cols[:, row, col], strict=True)}

    assert listed - {cell} == set(grid.neighbours(*cell))


@EACH_GRID
def test_edge_normals_lead_to_the_neighbours(grid: Grid) -> None:
    x, y = grid.center(3, 3)

    reached = {
        tuple(int(v) for v in grid.cell_at(x + dx, y + dy)) for dx, dy in grid.edge_normals()
    }

    assert reached == set(grid.neighbours(3, 3))


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


@EACH_GRID
def test_track_points_lie_on_the_edge(grid: Grid) -> None:
    x, y = grid.center(2, 3)
    points = grid.track_points()

    distances = grid.edge_distance(x + points[:, 0], y + points[:, 1])

    assert distances == pytest.approx(np.zeros(len(points)), abs=1e-9)


def test_hexagons_have_twelve_track_points_and_squares_eight() -> None:
    square, hexagon = GRIDS

    counts = len(square.track_points()), len(hexagon.track_points())

    assert counts == (8, 12)


@EACH_GRID
def test_track_runs_through_edges_to_neighbours_and_through_corners_beyond(grid: Grid) -> None:
    neighbours = set(grid.neighbours(2, 3))

    reached = [grid.track_exit((2, 3), point)[0] for point in range(len(grid.track_points()))]

    assert set(reached[0::2]) == neighbours
    assert not set(reached[1::2]) & (neighbours | {(2, 3)})
    assert len(set(reached)) == len(reached)


@EACH_GRID
def test_track_exits_lead_back_the_way_they_came(grid: Grid) -> None:
    points = range(len(grid.track_points()))

    returns = [grid.track_exit(*grid.track_exit((2, 3), point)) for point in points]

    assert returns == [((2, 3), point) for point in points]


@EACH_GRID
def test_track_through_a_corner_runs_along_the_edge_of_the_cells_beside_it(grid: Grid) -> None:
    points = grid.track_points()
    next_cell, entry = grid.track_exit((2, 3), 1)
    start = np.array(grid.center(2, 3)) + points[1]
    end = np.array(grid.center(*next_cell)) + points[entry]

    along = start + np.linspace(0, 1, 5)[:, None] * (end - start)
    flanks = grid.track_flanks((2, 3), 1)

    assert grid.edge_distance(along[:, 0], along[:, 1]) == pytest.approx(np.zeros(5), abs=1e-9)
    assert len(flanks) == 2
    assert set(flanks) <= set(grid.neighbours(2, 3)) & set(grid.neighbours(*next_cell))


@EACH_GRID
def test_track_through_the_middle_of_an_edge_runs_between_no_cells(grid: Grid) -> None:
    flanks = grid.track_flanks((2, 3), 0)

    assert flanks == []


@pytest.mark.parametrize(
    ("grid", "end", "turn"),
    [
        (GRIDS[1], 6, 0.0),
        (GRIDS[1], 5, 30.0),
        (GRIDS[1], 4, 60.0),
        (GRIDS[1], 3, 90.0),
        (GRIDS[0], 4, 0.0),
        (GRIDS[0], 3, 45.0),
        (GRIDS[0], 2, 90.0),
    ],
    ids=["hex-straight", "hex-30", "hex-60", "hex-90", "square-straight", "square-45", "square-90"],
)
def test_track_turns_more_the_further_its_ends_are_from_opposite(
    grid: Grid, end: int, turn: float
) -> None:
    turned = grid.track_turn(0, end)

    assert turned == pytest.approx(turn)


def test_hex_track_runs_across_every_edge_and_only_up_and_down_through_corners() -> None:
    grid = HexGrid(rows=5, cols=5, km_per_cell=2.0)
    steps = grid.track_steps()

    ends = grid.track_ends()

    corners = [point for point in ends if point % 2 == 1]
    assert set(range(0, 12, 2)) <= set(ends)
    assert len(ends) == 8
    assert all(steps[point][0] == pytest.approx(0.0) for point in corners)


def test_square_track_may_leave_by_every_point() -> None:
    grid = SquareGrid(rows=5, cols=5, km_per_cell=2.0)

    ends = grid.track_ends()

    assert ends == tuple(range(8))


@EACH_GRID
def test_cells_within_reach_come_as_flat_indices_in_the_order_within_finds_them(
    grid: Grid,
) -> None:
    reached = grid.within((3, 3), 2)

    within = flat_within(grid, (3, 3), 2)

    assert within.cells.tolist() == [row * grid.cols + col for row, col in reached]
    assert within.steps.tolist() == list(reached.values())


def test_grids_of_one_size_keep_their_own_cells_within_reach() -> None:
    square, hexagon = GRIDS

    cells = [flat_within(grid, (3, 3), 1).cells.size for grid in (square, hexagon)]

    assert cells == [5, 7]


def test_the_cells_kept_within_reach_cannot_be_changed() -> None:
    within = flat_within(GRIDS[1], (2, 2), 1)

    with pytest.raises(ValueError, match="read-only"):
        within.cells[0] = 0
