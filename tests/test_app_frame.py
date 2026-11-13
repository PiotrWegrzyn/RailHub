from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any

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
from railroads.ui.ground import GroundPictures
from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS
from railroads.ui.main_menu import Page
from railroads.ui.map_view import MapView
from railroads.ui.route_map import soften
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


def test_without_threads_an_autosave_is_written_a_little_each_frame_after_the_month(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "WRITER", FrameWorker())
    monkeypatch.setattr(app_module, "WRITING_SECONDS", 0.0)
    app = made_app()
    app.saves = Saves(tmp_path)
    app.clock.seconds = app.clock.seconds_per_month - 0.001
    frames_until_the_month_is_done(app)
    along = app.game.railway.trains[0].along

    frame(app)
    first = app.saves.listing()
    frames = 1
    while app._autosaving is not None and frames < 1000:
        frame(app)
        frames += 1

    loaded = app.saves.load("autosave 1")
    assert first == []
    assert frames > 2
    assert loaded.clock.date == app.clock.date
    assert loaded.game.railway.trains[0].along == along != app.game.railway.trains[0].along
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


def test_drawing_the_new_game_page_pulls_an_early_start_up_to_1750(
    context: None, tmp_path: Path
) -> None:
    shell = Shell(Saves(tmp_path))
    shell.menu.go_to("new")
    shell.menu.start_year = 1500

    shell_frame(shell)

    assert shell.menu.start_year == 1750


def colours_changed_by_a_month(app: App) -> None:
    """Lets a month pass that the view takes as changing every cell's mood, so the ground
    pictures take their colours again."""
    view = app.view
    view._satisfaction = np.full_like(view._satisfaction, 2)
    view.month_passed(np.random.default_rng(1))


def test_a_month_that_changes_colours_paints_the_overview_again_within_its_steps(
    context: None,
) -> None:
    app = made_app()
    frame(app)
    shown = set(app.view._textures)

    colours_changed_by_a_month(app)

    assert shown == {False}
    assert set(app.view._textures) == shown


def test_the_trains_window_map_is_softened_anew_a_step_a_frame_once_colours_change(
    context: None,
) -> None:
    app = made_app()
    app.view.prepare((1280.0, 800.0), 1, chunks=False)
    frame(app)
    colours_changed_by_a_month(app)
    version = app.view._ground_version

    frame(app)
    assert app.view._soft_texture is not None
    shown_meanwhile = app.view._soft_texture[0]
    frames = 1
    while app.view._soft is not None and app.view._soft[0] != version and frames < 50:
        frame(app)
        frames += 1
    frame(app)

    assert shown_meanwhile == version - 1
    assert frames > 2
    assert app.view._soft is not None
    assert np.array_equal(app.view._soft[1], soften(app.view._plain_picture()))
    assert app.view._soft_texture[0] == version


def test_a_month_paints_again_only_the_ground_pictures_the_last_frame_showed(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = made_app()
    app.view.build.trains.open = False
    app.view.camera.zoom_level = 2
    frame(app)
    app.view.camera.zoom_level = 4
    frame(app)
    kept = set(app.view._textures)
    painted: list[bool] = []
    picture = GroundPictures.picture

    def counted(
        pictures: GroundPictures,
        colors: np.ndarray,
        light_from: tuple[float, float],
        with_lines: bool = False,
    ) -> np.ndarray:
        painted.append(with_lines)
        return picture(pictures, colors, light_from, with_lines)

    monkeypatch.setattr(GroundPictures, "picture", counted)

    colours_changed_by_a_month(app)
    frame(app)
    painted_by_the_month = list(painted)
    app.view.camera.zoom_level = 2
    frame(app)

    assert kept == {False, True}
    assert painted_by_the_month == [False]
    assert painted == [False, True]


@pytest.fixture
def slow_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Autosaves written on the frames, so little of the game compressed on each that one takes
    many frames, as a large game's does in a browser."""
    monkeypatch.setattr(app_module, "WRITER", FrameWorker())
    monkeypatch.setattr(app_module, "WRITING_SECONDS", 0.0)
    stepped = Saves.autosave_steps
    monkeypatch.setattr(Saves, "autosave_steps", lambda saves, saved: stepped(saves, saved, 16384))


def test_a_month_that_ends_while_an_autosave_is_written_is_autosaved_after_it(
    context: None, tmp_path: Path, slow_writer: None
) -> None:
    app = made_app()
    app.saves = Saves(tmp_path)
    app.clock.seconds = app.clock.seconds_per_month - 0.001
    frames_until_the_month_is_done(app)
    frame(app)
    first = app.clock.date
    app.clock.seconds = app.clock.seconds_per_month - 0.001
    frames_until_the_month_is_done(app)
    second = app.clock.date
    written_meanwhile = app._autosaving is not None

    frames = 0
    while (app._autosaving is not None or app._autosave_due) and frames < 1000:
        frame(app)
        frames += 1

    assert written_meanwhile
    assert first != second
    assert app.saves.load("autosave 1").clock.date == second
    assert app.saves.load("autosave 2").clock.date == first


class HeldWriter(Executor):
    """Takes jobs without doing them, each finished when the test likes."""

    def __init__(self) -> None:
        self.jobs: list[Future[Any]] = []

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        job: Future[Any] = Future()
        self.jobs.append(job)
        return job


def test_an_autosave_due_while_another_month_is_half_done_waits_for_that_month(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = HeldWriter()
    monkeypatch.setattr(app_module, "WRITER", writer)
    app = made_app()
    app.saves = Saves(tmp_path)
    app.autosave()
    app.clock.seconds = app.clock.seconds_per_month - 0.001
    frames_until_the_month_is_done(app)
    app.clock.seconds = app.clock.seconds_per_month - 0.001
    frame(app)
    writer.jobs[0].set_result(None)

    frame(app)
    frame(app)
    turning = app._turning is not None
    started_while_turning = len(writer.jobs) > 1
    frames_until_the_month_is_done(app)
    frame(app)

    assert turning
    assert not started_while_turning
    assert len(writer.jobs) == 2


def test_without_threads_frames_stay_at_full_speed_while_an_autosave_is_written(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refreshes: list[None]
) -> None:
    monkeypatch.setattr(app_module, "WRITER", FrameWorker())
    monkeypatch.setattr(app_module, "WRITING_SECONDS", 0.0)
    app = made_app()
    app.game.railway.trains.clear()
    app.saves = Saves(tmp_path)
    app.autosave()

    frame(app)
    asked = len(refreshes)
    app.finish_autosave()
    refreshes.clear()
    frame(app)

    assert asked == 1
    assert refreshes == []
