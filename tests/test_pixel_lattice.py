import numpy as np
import pytest

from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.ui.pixel.lattice import Lattice
from railroads.ui.pixel.projection import PixelProjection

LOOKS = {HexGrid: (0.0, 1.0), SquareGrid: (-45.0, float(np.sqrt(2)))}
GRIDS = [HexGrid(rows=20, cols=16, km_per_cell=2.0), SquareGrid(rows=20, cols=16, km_per_cell=2.0)]


def projection_of(grid: Grid) -> PixelProjection:
    return PixelProjection.of(grid, *LOOKS[type(grid)])


@pytest.mark.parametrize("grid", GRIDS, ids=["hex", "square"])
@pytest.mark.parametrize("k", [1, 2])
def test_points_of_a_part_are_those_worked_out_for_the_part_alone(grid: Grid, k: int) -> None:
    projection = projection_of(grid)
    lattice = Lattice(projection, -30, 17, 120, 90, k, grid)

    x, y = lattice.plane(-21, 40, 64, 33)

    py, px = np.mgrid[0 : 33 * k, 0 : 64 * k]
    alone_x, alone_y = projection.to_plane(-21 + (px + 0.5) / k, 40 + (py + 0.5) / k)
    assert np.array_equal(x, alone_x) and np.array_equal(y, alone_y)


@pytest.mark.parametrize("grid", GRIDS, ids=["hex", "square"])
def test_cells_of_a_part_are_held_to_the_grid_and_say_which_lie_on_it(grid: Grid) -> None:
    projection = projection_of(grid)
    px, py = projection.to_pixels(*grid.center(0, 0))
    left, top = int(px) - 60, int(py) - 50
    lattice = Lattice(projection, left - 20, top - 20, 200, 160, grid=grid)

    rows, cols, inside = lattice.cells(left, top, 150, 120)

    iy, ix = np.mgrid[top : top + 120, left : left + 150]
    r, c = grid.cell_at(*projection.to_plane(ix + 0.5, iy + 0.5))
    assert np.array_equal(inside, grid.contains(r, c))
    assert np.array_equal(rows, np.clip(r, 0, grid.rows - 1))
    assert np.array_equal(cols, np.clip(c, 0, grid.cols - 1))
    assert not inside.all() and inside.any()


def test_a_lattice_holds_only_the_pixels_it_covers() -> None:
    grid = GRIDS[0]
    lattice = Lattice(projection_of(grid), 0, 0, 10, 10, grid=grid)

    inside, beyond = lattice.holds(2, 3, 8, 7), lattice.holds(2, 3, 9, 7)

    assert inside
    assert not beyond


def test_a_lattice_refuses_points_of_pixels_it_does_not_cover() -> None:
    grid = GRIDS[0]
    lattice = Lattice(projection_of(grid), 0, 0, 10, 10, grid=grid)

    with pytest.raises(ValueError):
        lattice.plane(-1, 0, 5, 5)


def test_a_lattice_at_a_resolution_not_a_power_of_two_gives_blocks_from_its_corner() -> None:
    projection = projection_of(GRIDS[0])
    lattice = Lattice(projection, -7, 5, 12, 10, 3)

    x, y = lattice.plane(-7, 5, 9, 10)

    py, px = np.mgrid[0:30, 0:27]
    alone_x, alone_y = projection.to_plane(-7 + (px + 0.5) / 3, 5 + (py + 0.5) / 3)
    assert np.array_equal(x, alone_x) and np.array_equal(y, alone_y)


def test_a_lattice_at_a_resolution_not_a_power_of_two_refuses_blocks_from_elsewhere() -> None:
    lattice = Lattice(projection_of(GRIDS[0]), -7, 5, 12, 10, 3)

    with pytest.raises(ValueError):
        lattice.plane(-6, 5, 9, 10)
