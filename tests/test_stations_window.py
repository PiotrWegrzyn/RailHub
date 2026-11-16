from collections.abc import Iterator

import pytest
from imgui_bundle import imgui  # noqa: TID251

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.game import Game
from railroads.sim.railway import Railway
from railroads.ui.stations_window import StationsWindow


@pytest.fixture
def gui() -> Iterator[None]:
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    yield
    imgui.destroy_context(context)


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
