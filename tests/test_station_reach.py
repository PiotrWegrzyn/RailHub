import math

from railroads.sim.grids import HexGrid
from railroads.ui.build_menu import _reach_border

GRID = HexGrid(rows=20, cols=20, km_per_cell=2.0)


def test_the_border_of_a_stations_reach_runs_round_the_outside_of_its_cells() -> None:
    station = (10, 10)

    border = _reach_border(GRID, station, 1)

    middle = GRID.center(*station)
    nearest = min(math.dist(middle, a) for a, _ in border)
    assert len(border) == 18
    assert nearest > 1.0


def test_the_border_of_a_stations_reach_is_worked_out_once() -> None:
    first = _reach_border(GRID, (4, 5), 2)

    again = _reach_border(GRID, (4, 5), 2)

    assert again is first
