from collections.abc import Iterator

import pytest
from imgui_bundle import imgui  # noqa: TID251
from screens import gui_context

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.game import Game
from railroads.sim.railway import Railway
from railroads.ui.stations_window import BUILD_BAR_ROOM, WINDOW_SIZE, StationsWindow, window_place


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
