from collections.abc import Iterator

import pytest
from imgui_bundle import imgui  # noqa: TID251
from screens import gui_context

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.game import Game
from railroads.sim.goods import Good
from railroads.sim.railway import Railway
from railroads.ui import stations_window
from railroads.ui.stations_window import (
    BUILD_BAR_ROOM,
    GOOD_GAP,
    WINDOW_SIZE,
    StationsWindow,
    counted,
    places,
    window_place,
)


@pytest.fixture
def gui() -> Iterator[None]:
    with gui_context():
        yield


def many_stations() -> Railway:
    game = Game(Constructions(flat_map(rows=30, cols=30)))
    game.finances.cash = 1e9
    for row in range(2, 28, 3):
        for col in range(2, 28, 4):
            game.build_station((row, col))
    return game.railway


def draw_frames(window: StationsWindow, railway: Railway, frames: int) -> None:
    for _ in range(frames):
        imgui.get_io().delta_time = 1 / 60
        imgui.new_frame()
        window.draw(railway, lambda _: None, lambda _: None)
        imgui.render()


def test_rows_of_stations_out_of_sight_are_not_drawn(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    railway = many_stations()
    window = StationsWindow(open=True)
    draw_frames(window, railway, 3)
    drawn: list[object] = []
    row = StationsWindow._draw_row
    monkeypatch.setattr(
        StationsWindow, "_draw_row", staticmethod(lambda *args: drawn.append(1) or row(*args))
    )

    draw_frames(window, railway, 1)

    assert 0 < len(drawn) < len(railway.stations())


@pytest.mark.parametrize(("width", "height"), [(1024.0, 768.0), (1920.0, 1080.0)])
def test_the_window_first_opens_whole_on_the_screen_above_the_build_bar(
    width: float, height: float
) -> None:
    x, y = window_place(width, height)

    assert x >= 0
    assert x + WINDOW_SIZE[0] <= width
    assert y + WINDOW_SIZE[1] <= height - BUILD_BAR_ROOM


def test_goods_are_counted_in_whole_units_the_most_first_leaving_out_crumbs() -> None:
    goods = {Good.COAL: 12.4, Good.POTATOES: 107.6, Good.LOGS: 0.3}

    shown = counted(goods)

    assert shown == [(Good.POTATOES, 108), (Good.COAL, 12)]


def test_goods_that_fit_stay_on_one_line() -> None:
    widths = [30.0, 30.0]

    placed = places(widths, 100.0)

    assert placed == [(0.0, 0), (30.0 + GOOD_GAP, 0)]


def test_a_good_that_would_run_past_the_edge_begins_a_new_line() -> None:
    widths = [40.0, 40.0, 40.0]

    placed = places(widths, 100.0)

    assert placed == [(0.0, 0), (40.0 + GOOD_GAP, 0), (0.0, 1)]


def test_a_good_wider_than_the_column_still_shows_on_its_own_line() -> None:
    widths = [150.0, 150.0]

    placed = places(widths, 100.0)

    assert placed == [(0.0, 0), (0.0, 1)]


def test_a_station_draws_an_icon_for_each_good_it_holds_and_each_traveller_waiting(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    railway = many_stations()
    station = railway.stations()[0]
    railway.station_goods(station).stockpile.update(
        {Good.COAL: 40.0, Good.POTATOES: 120.0, Good.PASSENGERS: 12.0}
    )
    drawn: list[Good] = []
    monkeypatch.setattr(
        stations_window, "draw_good_icon", lambda _list, good, *_: drawn.append(good)
    )
    window = StationsWindow(open=True)

    draw_frames(window, railway, 1)

    assert drawn == [Good.POTATOES, Good.COAL, Good.PASSENGERS]
