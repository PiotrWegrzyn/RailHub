from collections.abc import Iterator
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui, implot

from builders import flat_map
from railroads.saves import Saves
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.game import Game
from railroads.sim.goods import Good
from railroads.sim.grids import HexGrid
from railroads.sim.tilemap import Terrain, TileMap
from railroads.sim.trains import Stop
from railroads.ui import app as app_module
from railroads.ui.app import App, Shell
from railroads.ui.build_menu import Tool
from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS
from railroads.ui.main_menu import Page
from railroads.ui.map_view import MapView
from railroads.ui.workers import FrameWorker


class FakeTexture:
    def texture_id(self) -> int:
        return 1


def make_map() -> TileMap:
    grid = HexGrid(rows=20, cols=24, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    return flat_map(grid=grid, terrain=terrain)


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture())
    gui = imgui.create_context()
    plots = implot.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    yield
    implot.destroy_context(plots)
    imgui.destroy_context(gui)


def made_app() -> App:
    constructions = Constructions(make_map())
    game = Game(constructions)
    game.finances.cash = 1_000_000.0
    for _ in range(4):
        game.growth.build(Building((10, 10), BuildingKind.HOUSE))
    for cell in ((10, 6), (10, 16)):
        game.build_station(cell)
    game.lay_track((10, 6), (10, 16))
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop((10, 6)), Stop((10, 16))]
    app = App(MapView(game))
    app.help.open = app.tech.open = app.reports.open = True
    app.view.build.trains.open = True
    app.view.selected = (10, 10)
    app.view.station_window.station = (10, 6)
    app.view.build.choose(Tool.TRACK)
    app.view.build.track_from = (10, 6)
    return app


def frame(app: App) -> None:
    imgui.get_io().delta_time = 1 / 60
    imgui.new_frame()
    imgui.set_next_window_pos((0, 0))
    imgui.set_next_window_size((1280, 800))
    imgui.begin("Railroads")
    app.gui()
    imgui.end()
    imgui.render()


@pytest.mark.parametrize("zoom", range(len(ZOOM_CELL_WIDTHS)))
def test_every_window_draws_at_every_zoom(context: None, zoom: int) -> None:
    app = made_app()
    app.view.camera.zoom_level = zoom
    app.view.look_at((10, 10))

    for _ in range(3):
        frame(app)

    assert app.view.camera.zoom_level == zoom


@pytest.fixture
def refreshes(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    asked: list[None] = []
    monkeypatch.setattr(hello_imgui, "request_refresh", lambda: asked.append(None))
    return asked


def test_frames_stay_at_full_speed_while_a_train_runs(context: None, refreshes: list[None]) -> None:
    app = made_app()

    frame(app)

    assert app.game.railway.trains[0].path
    assert len(refreshes) == 1


def test_frames_may_slow_down_while_trains_wait_at_their_stations(
    context: None, refreshes: list[None]
) -> None:
    app = made_app()
    app.game.railway.trains[0].stops = [Stop((10, 6), [Good.COAL]), Stop((10, 16))]

    frame(app)

    assert not app.game.railway.trains[0].path
    assert refreshes == []


def test_frames_may_slow_down_once_the_game_is_over(context: None, refreshes: list[None]) -> None:
    app = made_app()
    app.clock.months += app.clock.months_left

    frame(app)

    assert app.game.railway.trains[0].path
    assert refreshes == []


def shell_frame(shell: Shell) -> None:
    imgui.get_io().delta_time = 1 / 60
    imgui.new_frame()
    shell.gui()
    imgui.render()


def test_the_game_autosaves_as_a_month_begins(context: None, tmp_path: Path) -> None:
    app = made_app()
    app.saves = Saves(tmp_path)
    app.clock.seconds = app.clock.seconds_per_month - 0.001

    frame(app)
    app.finish_autosave()

    assert app.saves.load("autosave 1").clock.date == app.clock.date
    assert app.notice == "Autosaved"


def frames_until_the_month_is_done(app: App) -> int:
    frames = 1
    frame(app)
    while app._turning is not None and frames < 20:
        frame(app)
        frames += 1
    return frames


def test_a_month_is_done_over_a_few_frames_and_saved_on_the_frame_after(
    context: None, tmp_path: Path
) -> None:
    app = made_app()
    app.saves = Saves(tmp_path)
    app.clock.seconds = app.clock.seconds_per_month - 0.001

    frames = frames_until_the_month_is_done(app)
    started_with_the_month = app._autosaving is not None
    frame(app)
    started_after = app._autosaving is not None
    app.finish_autosave()

    assert frames > 1
    assert not started_with_the_month
    assert started_after
    assert [info.name for info in app.saves.listing()] == ["autosave 1"]


def test_the_clock_and_trains_wait_while_a_month_is_half_done(context: None) -> None:
    app = made_app()
    app.clock.seconds = app.clock.seconds_per_month - 0.001
    frame(app)
    seconds = app.clock.seconds
    months = app.clock.months

    frame(app)

    assert app._turning is not None
    assert (app.clock.months, app.clock.seconds) == (months, seconds)


def test_no_autosave_is_written_while_the_month_goes_on(context: None, tmp_path: Path) -> None:
    app = made_app()
    app.saves = Saves(tmp_path)

    frame(app)
    app.finish_autosave()

    assert app.saves.listing() == []


def test_the_saves_window_draws_both_tabs(context: None, tmp_path: Path) -> None:
    app = made_app()
    app.saves = Saves(tmp_path)
    app.saves.save("Earlier", app.saved())
    app.saves_window.show(app.saves)

    for _ in range(3):
        frame(app)

    assert app.saves_window.open
    assert [info.name for info in app.saves_window.listing] == ["Earlier"]


@pytest.mark.parametrize("page", ["main", "new", "load"])
def test_every_page_of_the_main_menu_draws(context: None, tmp_path: Path, page: Page) -> None:
    saves = Saves(tmp_path)
    saves.save("Earlier", made_app().saved())
    shell = Shell(saves)
    shell.menu.go_to(page)

    for _ in range(3):
        shell_frame(shell)

    assert shell.app is None
    assert shell.menu.page == page


def test_a_game_loaded_from_the_menu_comes_back_as_it_was_saved(
    context: None, tmp_path: Path
) -> None:
    saves = Saves(tmp_path)
    app = made_app()
    app.view.camera.zoom_level = 2
    saves.save("Earlier", app.saved())
    shell = Shell(saves)

    shell.load("Earlier")
    assert shell._task is not None
    shell._task.result()
    shell_frame(shell)

    assert shell.app is not None
    loaded = shell.app
    assert loaded.game.finances.cash == app.game.finances.cash
    assert loaded.game.railway.stations() == app.game.railway.stations()
    assert loaded.clock == app.clock
    assert loaded.view.camera.zoom_level == 2
    assert loaded.saves is saves


def test_a_save_that_fails_to_load_stays_in_the_menu_with_why(
    context: None, tmp_path: Path
) -> None:
    shell = Shell(Saves(tmp_path))

    shell.load("Nothing")
    assert shell._task is not None
    shell._task.exception()
    shell_frame(shell)

    assert shell.app is None
    assert "no save called 'Nothing'" in shell.menu.error


def test_without_threads_an_autosave_is_written_on_the_frame_after_the_month(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "WRITER", FrameWorker())
    app = made_app()
    app.saves = Saves(tmp_path)
    app.clock.seconds = app.clock.seconds_per_month - 0.001

    frames_until_the_month_is_done(app)
    asked = app.saves.listing()
    frame(app)

    assert asked == []
    assert app.saves.load("autosave 1").clock.date == app.clock.date
    assert app.notice == "Autosaved"


def test_without_threads_leaving_a_game_writes_its_autosave_first(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "WRITER", FrameWorker())
    app = made_app()
    app.saves = Saves(tmp_path)
    app.clock.seconds = app.clock.seconds_per_month - 0.001
    frame(app)

    app.finish_autosave()

    assert [info.name for info in app.saves.listing()] == ["autosave 1"]


def test_without_threads_a_save_loads_once_the_waiting_screen_has_shown(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "worker", lambda _: FrameWorker())
    saves = Saves(tmp_path)
    saves.save("Earlier", made_app().saved())
    shell = Shell(saves)
    shell.load("Earlier")

    shell_frame(shell)
    waiting = shell.app
    shell_frame(shell)

    assert waiting is None
    assert shell.app is not None
    assert shell.app.saves is saves


def test_leaving_a_game_goes_back_to_the_main_menu(context: None, tmp_path: Path) -> None:
    shell = Shell(Saves(tmp_path))
    shell.app = made_app()
    shell.app.saves = shell.saves
    shell.app.menu_wanted = True

    shell_frame(shell)
    shell_frame(shell)

    assert shell.app is None
    assert shell.menu.page == "main"


def full_screen_frame(shell: Shell, click: tuple[float, float] | None = None) -> None:
    io = imgui.get_io()
    io.delta_time = 1 / 60
    if click is not None:
        io.add_mouse_pos_event(*click)
        io.add_mouse_button_event(0, True)
    imgui.new_frame()
    imgui.set_next_window_pos((0, 0))
    imgui.set_next_window_size((1280, 800))
    imgui.begin("Railroads", None, imgui.WindowFlags_.no_decoration.value)
    shell.gui()
    imgui.end()
    imgui.render()
    if click is not None:
        io.add_mouse_button_event(0, False)


def test_the_main_menu_answers_clicks_after_leaving_a_game(context: None, tmp_path: Path) -> None:
    shell = Shell(Saves(tmp_path))
    full_screen_frame(shell)
    shell.app = made_app()
    shell.app.saves = shell.saves
    for _ in range(3):
        full_screen_frame(shell, click=(640.0, 400.0))
    shell.app.menu_wanted = True
    full_screen_frame(shell)

    for y in range(0, 800, 8):
        full_screen_frame(shell, click=(640.0, float(y)))
        full_screen_frame(shell)
        if shell.menu.page != "main":
            break

    assert shell.app is None
    assert shell.menu.page == "new"


def test_the_waiting_screen_shows_how_far_a_new_game_has_grown(
    context: None, tmp_path: Path
) -> None:
    shell = Shell(Saves(tmp_path))
    shell._task = Future()
    shell.report("Growing towns")
    shell.advance(0.4)

    shell_frame(shell)

    assert shell.app is None
    assert shell._done == 0.4
