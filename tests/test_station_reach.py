import math

import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.game import Game
from railroads.sim.grids import HexGrid
from railroads.ui.build_menu import _border, draw_station_reach, reach_border
from railroads.ui.map_view import MapView
from railroads.ui.pixel.relief import Elevation

GRID = HexGrid(rows=20, cols=20, km_per_cell=2.0)


class DrawList:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def add_convex_poly_filled(self, *_: object) -> None:
        self.calls.append("cell")

    def add_polyline(self, *_: object) -> None:
        self.calls.append("outline")

    def add_line(self, *_: object) -> None:
        self.calls.append("border")


def test_the_border_of_a_stations_reach_runs_round_the_outside_of_its_cells() -> None:
    station = (10, 10)

    border = reach_border(GRID, (station,), 1)

    middle = GRID.center(*station)
    nearest = min(math.dist(middle, a) for a, _ in border)
    assert len(border) == 18
    assert nearest > 1.0


def test_the_border_of_a_stations_reach_is_worked_out_once() -> None:
    first = reach_border(GRID, ((4, 5),), 2)

    again = reach_border(GRID, ((4, 5),), 2)

    assert again is first


def test_the_border_of_a_station_two_rows_further_is_the_same_border_moved_with_it() -> None:
    here, further = (6, 7), (8, 9)
    dx = GRID.center(*further)[0] - GRID.center(*here)[0]
    dy = GRID.center(*further)[1] - GRID.center(*here)[1]

    border = reach_border(GRID, (here,), 2)
    moved = reach_border(GRID, (further,), 2)

    expected = {
        (round(a[0] + dx, 9), round(a[1] + dy, 9), round(b[0] + dx, 9), round(b[1] + dy, 9))
        for a, b in border
    }
    found = {(round(a[0], 9), round(a[1], 9), round(b[0], 9), round(b[1], 9)) for a, b in moved}
    assert found == expected


def test_the_border_of_a_stations_reach_cut_by_the_maps_edge_runs_round_what_is_left() -> None:
    station = (0, 3)
    cells = set(GRID.within(station, 2))

    border = reach_border(GRID, (station,), 2)

    expected = {(round(x, 9), round(y, 9)) for edge in _border(GRID, cells) for x, y in edge}
    found = {(round(x, 9), round(y, 9)) for edge in border for x, y in edge}
    assert len(border) == len(_border(GRID, cells))
    assert found == expected


def test_a_stations_reach_is_lifted_onto_the_ground_in_one_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = Game(Constructions(flat_map(20, 24)))
    game.finances.cash = 1_000_000.0
    game.build_station((10, 8))
    view = MapView(game)
    view.camera.zoom_level = 4
    view.look_at((10, 8))
    camera = view._marks_camera()
    looked: list[int] = []
    plain = Elevation.lift

    def counted(elevation: Elevation, x: object, y: object) -> object:
        looked.append(np.size(x))
        return plain(elevation, x, y)

    monkeypatch.setattr(Elevation, "lift", counted)
    drawn = DrawList()

    draw_station_reach(
        drawn,  # type: ignore[arg-type]
        game.tile_map.grid,
        (10, 8),
        [],
        camera,
        (640.0, 400.0),
        game.railway.reach,
    )

    assert len(looked) == 1
    assert looked[0] == 6 * drawn.calls.count("cell") + 2 * drawn.calls.count("border")
