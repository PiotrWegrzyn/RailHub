import gc
import pickle
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui, implot
from imgui_bundle import icons_fontawesome_4 as fa
from test_saves import played_game
from test_web_setup import SETTINGS, FakePage, grows_as

from builders import flat_map
from railroads.land import lot_middle
from railroads.newgame import MapChoice, Settings, make_game
from railroads.saves import SavedGame, Saves
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.game import Game
from railroads.sim.goods import Good
from railroads.sim.grids import HexGrid
from railroads.sim.happenings import Headline
from railroads.sim.lots import Lots
from railroads.sim.steps import Steps
from railroads.sim.tilemap import Terrain, TileMap
from railroads.sim.trains import Stop
from railroads.ui import app as app_module
from railroads.ui import top_bar, web_setup
from railroads.ui.app import OPENING_ZOOM, App, Shell
from railroads.ui.build_menu import Tool
from railroads.ui.ground import GroundPictures
from railroads.ui.history_pictures import opening_center
from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS
from railroads.ui.main_menu import MENU_WIDTH, PANEL_PAD, MainMenu, Page
from railroads.ui.map_view import STATION_CARD_DELAY, VISIT_ZOOM, Lens, MapView
from railroads.ui.route_map import soften
from railroads.ui.station_window import draw_station_card
from railroads.ui.waiting_screen import WORK
from railroads.ui.web_setup import PageSetup
from railroads.ui.workers import FrameWorker

PICTURED = replace(SETTINGS, show_map_growing=True)


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
    game.growth.build(Building((10, 11), BuildingKind.FARM))
    for cell in ((10, 6), (10, 16)):
        game.build_station(cell)
    game.lay_track((10, 6), (10, 16))
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop((10, 6)), Stop((10, 16))]
    app = App(MapView(game))
    app.help.open = app.tiles.open = app.reports.open = True
    app.view.build.trains.open = True
    app.view.selected = (10, 11)
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


@pytest.mark.parametrize("lens", list(Lens))
def test_the_map_draws_with_each_lens_on(context: None, lens: Lens) -> None:
    app = made_app()
    app.view.lens = lens
    app.view.look_at((10, 10))

    for _ in range(3):
        frame(app)

    assert app.view.lens == lens


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
    app.game.railway.trains[0].stops = [
        Stop((10, 6), [Good.COAL], wait_for_load=True),
        Stop((10, 16)),
    ]

    frame(app)

    assert not app.game.railway.trains[0].path
    assert refreshes == []


def shell_frame(shell: Shell) -> None:
    imgui.get_io().delta_time = 1 / 60
    imgui.new_frame()
    shell.gui()
    imgui.render()


def frames_until_the_month_is_done(app: App) -> int:
    frames = 1
    frame(app)
    while app._turning is not None and frames < 100:
        frame(app)
        frames += 1
    return frames


def test_a_month_is_done_over_a_few_frames_and_its_autosave_taken_in_its_last_ones(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "MONTH_FRAME_SECONDS", 0.0)
    app = made_app()
    app.saves = Saves(tmp_path)
    app.clock.seconds = app.clock.seconds_per_month - 0.001

    frames = frames_until_the_month_is_done(app)
    started_with_the_month = app._autosaving is not None
    app.finish_autosave()

    assert frames > len(app.game.big_parts())
    assert started_with_the_month
    assert not app._autosave_due
    assert [info.name for info in app.saves.listing()] == ["autosave 1"]
    assert app.saves.load("autosave 1").clock.date == app.clock.date
    assert app.notice == "Autosaved"


def test_the_clock_and_trains_wait_while_a_month_is_half_done(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "MONTH_FRAME_SECONDS", 0.0)
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


def test_the_saves_window_lists_the_saves_on_disk_once_shown(context: None, tmp_path: Path) -> None:
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


@pytest.mark.parametrize("page", ["main", "new", "load"])
def test_the_menu_draws_nothing_past_its_buttons(context: None, tmp_path: Path, page: Page) -> None:
    menu = MainMenu(Saves(tmp_path))
    menu.go_to(page)
    left = (1280 - MENU_WIDTH) / 2 - PANEL_PAD

    for _ in range(3):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((1280, 800))
        imgui.begin("Menu")
        menu.draw()
        imgui.end()
        imgui.render()

    clips = [cmd.clip_rect for drawn in imgui.get_draw_data().cmd_lists for cmd in drawn.cmd_buffer]
    assert (left, left + MENU_WIDTH + 2 * PANEL_PAD) in [(clip.x, clip.z) for clip in clips]


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


def test_a_game_loaded_is_set_aside_from_the_garbage_collector(
    context: None, tmp_path: Path
) -> None:
    saves = Saves(tmp_path)
    saves.save("Earlier", made_app().saved())
    shell = Shell(saves)
    shell.load("Earlier")
    assert shell._task is not None
    shell._task.result()

    shell_frame(shell)

    assert shell.app is not None
    assert gc.get_freeze_count() > 0
    gc.unfreeze()


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
    frames = 1
    while shell.app is None and frames < 10:
        shell_frame(shell)
        frames += 1

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


def test_drawing_the_new_game_page_pulls_an_early_start_up_to_1804(
    context: None, tmp_path: Path
) -> None:
    shell = Shell(Saves(tmp_path))
    shell.menu.go_to("new")
    shell.menu.start_year = 1500

    shell_frame(shell)

    assert shell.menu.start_year == 1804


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


def grown_on_the_frames(
    saved: SavedGame, years: tuple[int, ...] = ()
) -> Callable[..., Steps[SavedGame]]:
    def made(
        settings: Settings,
        report: Callable[[str], None],
        progress: Callable[[float], None] = lambda done: None,
        look: Callable[[Constructions, int], None] = lambda constructions, year: None,
    ) -> Steps[SavedGame]:
        for done in (0.0, 0.5):
            progress(done)
            yield
        for year in years:
            look(saved.game.constructions, year)
            yield
        progress(1.0)
        return saved

    return made


def never_grown_here(*args: Any) -> Steps[SavedGame]:
    raise AssertionError("the game was to grow in the page's worker")


def test_without_threads_a_new_game_grown_by_the_page_worker_fills_the_waiting_bar(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "worker", lambda _: FrameWorker())
    monkeypatch.setattr(app_module, "make_game_steps", never_grown_here)
    grown = played_game()
    monkeypatch.setattr(web_setup, "make_game", grows_as(grown))
    page = FakePage()
    monkeypatch.setattr(app_module, "page_setup", lambda: PageSetup(page))
    shell = Shell(Saves(tmp_path))
    shell.start(SETTINGS)
    shell_frame(shell)
    shell_frame(shell)
    page.work()

    shell_frame(shell)
    share = shell._done
    frames = 1
    while shell.app is None and frames < 20:
        shell_frame(shell)
        frames += 1

    assert share == 0.5
    assert shell.app is not None
    assert shell.app.game.finances.cash == grown.game.finances.cash


def test_the_menu_has_the_page_worker_grow_the_game_it_would_start_once_its_settings_rest(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "AHEAD_AFTER_S", 0.0)
    page = FakePage()
    setup = PageSetup(page)
    monkeypatch.setattr(app_module, "page_setup", lambda: setup)
    shell = Shell(Saves(tmp_path))

    shell_frame(shell)

    assert page.started == 1
    assert pickle.loads(page.settings or b"")[0] == shell.menu.settings()


def test_the_menu_grows_nothing_ahead_while_its_settings_keep_changing(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = FakePage()
    monkeypatch.setattr(app_module, "page_setup", lambda: PageSetup(page))
    shell = Shell(Saves(tmp_path))

    for seed in range(3):
        shell.menu.seed = seed
        shell_frame(shell)

    assert page.started == 0


def test_starting_the_game_grown_ahead_takes_it_up_without_growing_it_again(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "AHEAD_AFTER_S", 0.0)
    monkeypatch.setattr(app_module, "worker", lambda _: FrameWorker())
    monkeypatch.setattr(app_module, "make_game_steps", never_grown_here)
    grown = played_game()
    monkeypatch.setattr(web_setup, "make_game", grows_as(grown))
    page = FakePage()
    setup = PageSetup(page)
    monkeypatch.setattr(app_module, "page_setup", lambda: setup)
    shell = Shell(Saves(tmp_path))
    shell_frame(shell)
    page.work()

    shell.start(shell.menu.settings())
    frames = 0
    while shell.app is None and frames < 20:
        shell_frame(shell)
        frames += 1

    assert shell.app is not None
    assert shell.app.game.finances.cash == grown.game.finances.cash
    assert page.started == 1


def test_loading_a_game_stops_the_one_grown_ahead(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "AHEAD_AFTER_S", 0.0)
    page = FakePage()
    setup = PageSetup(page)
    monkeypatch.setattr(app_module, "page_setup", lambda: setup)
    shell = Shell(Saves(tmp_path))
    shell_frame(shell)

    shell.load("missing")

    assert page.stopped == 1
    assert setup.ahead is None


def test_without_threads_a_new_game_grows_on_the_frames_when_the_page_worker_fails(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "worker", lambda _: FrameWorker())
    monkeypatch.setattr(app_module, "make_game_steps", grown_on_the_frames(played_game()))
    page = FakePage(fails=True)
    monkeypatch.setattr(app_module, "page_setup", lambda: PageSetup(page))
    shell = Shell(Saves(tmp_path))
    shell.start(SETTINGS)
    shell_frame(shell)
    shell_frame(shell)
    page.work()

    frames = 0
    while shell.app is None and frames < 20:
        shell_frame(shell)
        frames += 1

    assert shell.app is not None
    assert shell._done == 1.0


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


def test_the_map_is_prepared_a_part_at_a_time_ending_as_prepared_at_once(context: None) -> None:
    stepped = made_app()
    at_once = made_app()

    steps = sum(1 for _ in stepped.view.prepare_steps((1280.0, 800.0), 1, chunks=False))
    at_once.view.prepare((1280.0, 800.0), 1, chunks=False)

    assert steps > 4
    assert stepped.view._soft is not None
    assert at_once.view._soft is not None
    assert np.array_equal(stepped.view._soft[1], at_once.view._soft[1])
    assert stepped.view._pixels is not None
    assert stepped.view._pixels.world.ground.biomes.tolist() == (
        at_once.view.pixel_layer().world.ground.biomes.tolist()
    )


def test_the_trains_window_map_is_softened_anew_a_step_a_frame_once_colours_change(
    context: None,
) -> None:
    app = made_app()
    app.view.prepare((1280.0, 800.0), 1, chunks=False)
    frame(app)
    colours_changed_by_a_month(app)
    version = app.view._ground_version

    frame(app)
    shown_meanwhile = app.view._soft_texture
    frames = 1
    while app.view._soft is not None and app.view._soft[0] != version and frames < 50:
        frame(app)
        frames += 1
    frame(app)

    assert shown_meanwhile is not None
    assert shown_meanwhile[0] == version - 1
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
    context: None, tmp_path: Path, slow_writer: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "AUTOSAVE_GAP_S", 0.0)
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
    monkeypatch.setattr(app_module, "AUTOSAVE_GAP_S", 0.0)
    monkeypatch.setattr(app_module, "MONTH_FRAME_SECONDS", 0.0)
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


def test_a_saves_window_opened_while_an_autosave_is_written_lists_it_once_it_is_written(
    context: None, tmp_path: Path, slow_writer: None
) -> None:
    app = made_app()
    app.saves = Saves(tmp_path)
    app.clock.seconds = app.clock.seconds_per_month - 0.001
    frames_until_the_month_is_done(app)
    frame(app)
    app.toggle_saves()
    listed_meanwhile = list(app.saves_window.listing)
    written_meanwhile = app._autosaving is not None

    frames = 0
    while app._autosaving is not None and frames < 1000:
        frame(app)
        frames += 1

    listed = [(info.name, info.date) for info in app.saves_window.listing]
    assert written_meanwhile
    assert listed_meanwhile == []
    assert listed == [("autosave 1", app.clock.date)]


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


def test_a_new_game_opens_on_the_middle_of_the_map_and_stays_there_as_the_waiting_screen_falls(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "worker", lambda _: FrameWorker())
    monkeypatch.setattr(app_module, "page_setup", lambda: None)
    monkeypatch.setattr(app_module, "make_game_steps", grown_on_the_frames(played_game()))
    shell = Shell(Saves(tmp_path))
    shell.start(PICTURED)
    frames = 0
    while shell.app is None and frames < 20:
        shell_frame(shell)
        frames += 1

    opened = shell._shatter is not None
    while shell._shatter is not None and frames < 400:
        shell_frame(shell)
        frames += 1

    assert opened
    assert shell._shatter is None
    assert shell.app is not None
    camera = shell.app.view.camera
    middle = opening_center(shell.app.view.tile_map.grid, (1280, 800))
    assert (camera.center_x, camera.center_y) == middle
    assert camera.zoom_level == OPENING_ZOOM


def test_a_new_game_without_pictures_shows_the_work_and_opens_on_its_biggest_town(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "worker", lambda _: FrameWorker())
    monkeypatch.setattr(app_module, "page_setup", lambda: None)
    monkeypatch.setattr(
        app_module, "make_game_steps", grown_on_the_frames(played_game(), (850, 950))
    )
    shell = Shell(Saves(tmp_path))
    shell.start(SETTINGS)
    frames = 0
    while shell.app is None and frames < 20:
        shell_frame(shell)
        frames += 1

    shown = [label.text for label in shell._labels]
    while shell._shatter is not None and frames < 400:
        shell_frame(shell)
        frames += 1

    assert shell._show.year is None
    assert shown[0] is not None
    assert shown[0].startswith(WORK[0])
    assert shell.app is not None
    view = shell.app.view
    assert view.camera.zoom_level >= VISIT_ZOOM
    assert (view.camera.center_x, view.camera.center_y) == view.tile_map.grid.center(8, 15)


def test_the_yolo_button_starts_a_random_place_with_the_other_settings_kept(
    tmp_path: Path,
) -> None:
    menu = MainMenu(Saves(tmp_path), seed=42, start_year=1900)

    settings = menu.random_settings()

    assert settings.map == MapChoice.WORLD
    assert (settings.seed, settings.start_year) == (42, 1900)


def test_months_ending_soon_after_an_autosave_are_saved_once_the_gap_has_passed(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr(app_module.time, "monotonic", lambda: now[0])
    app = made_app()
    app.saves = Saves(tmp_path)
    for _ in range(3):
        app.clock.seconds = app.clock.seconds_per_month - 0.001
        frames_until_the_month_is_done(app)
        frame(app)
        app._finish_writing()
    saved_soon = [info.name for info in app.saves.listing()]
    now[0] += app_module.AUTOSAVE_GAP_S

    frame(app)
    app._finish_writing()

    assert saved_soon == ["autosave 1"]
    assert app.saves.load("autosave 1").clock.date == app.clock.date


def test_a_new_game_shows_its_map_as_its_towns_grow_and_breaks_up_the_last_picture(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "worker", lambda _: FrameWorker())
    monkeypatch.setattr(app_module, "page_setup", lambda: None)
    monkeypatch.setattr(
        app_module, "make_game_steps", grown_on_the_frames(played_game(), (850, 900, 950))
    )
    shell = Shell(Saves(tmp_path))
    shell.start(PICTURED)
    shown: list[int] = []
    frames = 0
    while shell.app is None and frames < 40:
        shell_frame(shell)
        year = shell._show.year
        if year is not None and year not in shown:
            shown.append(year)
        frames += 1

    assert shown == [850, 950]
    assert shell._shatter is not None
    assert shell._shatter.pictures
    assert "The map in 950" in [label.text for label in shell._shatter.labels]


def test_a_game_the_command_line_asks_for_starts_on_the_first_frame_for_the_windows_size(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "worker", lambda _: FrameWorker())
    monkeypatch.setattr(app_module, "page_setup", lambda: None)
    monkeypatch.setattr(app_module, "make_game_steps", grown_on_the_frames(played_game()))
    imgui.get_io().display_size = imgui.ImVec2(1000, 700)
    shell = Shell(Saves(tmp_path), SETTINGS)
    started_before = shell._task is not None

    shell_frame(shell)

    assert not started_before
    assert shell._task is not None
    assert shell._window == (1000, 700)


def test_the_menu_starts_games_without_pictures_of_the_map_growing_unless_ticked(
    tmp_path: Path,
) -> None:
    plain = MainMenu(Saves(tmp_path))
    ticked = MainMenu(Saves(tmp_path), show_map_growing=True)

    settings = (plain.settings(), ticked.settings(), ticked.random_settings())

    assert [each.show_map_growing for each in settings] == [False, True, True]


@pytest.mark.parametrize("zoom", range(len(ZOOM_CELL_WIDTHS)))
def test_the_land_for_sale_around_the_map_draws_at_every_zoom(context: None, zoom: int) -> None:
    app = made_app()
    lots = Lots.around_middle(app.game.tile_map)
    lots.offers = lots.next_to()
    app.game.lots = lots
    app.view.land.asked = (-2, 0)
    app.view.camera.zoom_level = zoom
    app.view.look_at((0, 0))

    for _ in range(3):
        frame(app)

    assert app.view.land.asked == (-2, 0)


def test_land_confirmed_is_bought_behind_the_waiting_screen_and_the_view_opens_on_it(
    context: None, tmp_path: Path
) -> None:
    settings = Settings(seed=5, map=MapChoice.POLAND, cash=2_000_000.0)
    shell = Shell(Saves(tmp_path))
    shell.app = App.from_saved(make_game(settings, report=lambda _: None), shell.saves)
    shell.app.view.camera.zoom_level = 3
    shell.app.view.land.wanted = (-2, 0)

    shell_frame(shell)
    assert shell._task is not None
    shell._task.result()
    shell_frame(shell)

    app = shell.app
    lots = app.game.lots
    assert lots is not None
    assert (-2, 0) in lots.owned
    assert app.game.finances.cash == pytest.approx(2_000_000.0 - 377_000.0, abs=1.0)
    assert app.view.camera.zoom_level == 3
    x, y = app.game.tile_map.grid.center(*lot_middle(lots, (-2, 0)))
    assert (app.view.camera.center_x, app.view.camera.center_y) == pytest.approx((x, y))


def test_a_train_whose_stop_was_bulldozed_still_draws_runs_and_can_be_rerouted(
    context: None,
) -> None:
    app = made_app()
    game = app.view.game
    game.build_station((10, 22))
    game.lay_track((10, 16), (10, 22))
    train = game.railway.trains[0]
    train.stops.append(Stop((10, 22), [Good.PASSENGERS]))
    app.view.station_window.station = (10, 16)
    app.view.build.choose(Tool.NONE)
    frame(app)
    done = game.bulldoze((10, 16))
    app.view.buildings_changed(done.changed)

    for _ in range(3):
        frame(app)
    train.stops[0].add_car(Good.PASSENGERS)
    train.cars_changed(0)
    frame(app)
    frames_until_the_month_is_done(app)
    app.clock.advance(1.0)
    frames_until_the_month_is_done(app)

    assert done.ok
    assert all(stop.cell != (10, 16) for stop in train.stops)


def test_a_news_card_and_its_story_draw_beside_the_map(context: None) -> None:
    app = made_app()
    headline = Headline("May 1981", "The Pope is shot", "At 17:17 on 13 May...", ("For a year",))
    app.view.summaries.add(headline)
    app.view.summaries.open = headline

    for _ in range(3):
        frame(app)

    assert app.view.summaries.open is headline
    assert app.view.summaries.outlined() == ()


def test_holding_an_arrow_key_moves_the_map(context: None) -> None:
    app = made_app()
    app.view.build.choose(Tool.NONE)
    frame(app)
    before = app.view.camera.center_x, app.view.camera.center_y

    imgui.get_io().add_key_event(imgui.Key.right_arrow, True)
    frame(app)
    frame(app)
    imgui.get_io().add_key_event(imgui.Key.right_arrow, False)

    assert (app.view.camera.center_x, app.view.camera.center_y) != before


def test_the_pause_button_stays_put_whatever_the_month_and_the_cash(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = made_app()
    places: list[float] = []
    button = top_bar.toggle_button

    def watched(label: str, *args: object) -> bool:
        if label == fa.ICON_FA_PAUSE:
            places.append(imgui.get_cursor_screen_pos().x)
        return button(label, *args)

    monkeypatch.setattr(top_bar, "toggle_button", watched)

    for months, cash in ((4, 5.0), (4, 5.0), (8, 123_456_789.0)):
        app.clock.months = months
        app.game.finances.cash = cash
        frame(app)

    assert len(places) == 3
    assert places[1] == places[2]


def test_a_save_that_failed_to_load_is_no_longer_reported_on_another_page(
    tmp_path: Path,
) -> None:
    menu = MainMenu(Saves(tmp_path))
    menu.go_to("load")
    menu.error = "'autosave 1' was saved by a different version of the game and can't be loaded."

    menu.go_to("new")

    assert menu.error == ""


@pytest.mark.parametrize("paused", [True, False])
def test_a_banner_over_the_map_says_so_while_time_stands_still(
    context: None, monkeypatch: pytest.MonkeyPatch, paused: bool
) -> None:
    app = made_app()
    app.clock.paused = paused
    drawn: list[object] = []
    monkeypatch.setattr(app_module, "draw_pause_banner", lambda *corners: drawn.append(corners))

    frame(app)

    assert bool(drawn) == paused


def test_a_new_game_begins_with_time_standing_still() -> None:
    saved = make_game(Settings(seed=5, map=MapChoice.POLAND), report=lambda _: None)

    assert saved.clock.paused


def test_a_station_preview_is_kept_apart_from_the_track_preview_and_worked_out_once_per_cell(
    context: None,
) -> None:
    app = made_app()
    frame(app)
    app.view._preview = object()  # type: ignore[assignment]

    first = app.view._station_preview((10, 11))
    again = app.view._station_preview((10, 11))
    other = app.view._station_preview((10, 12))

    assert again is first
    assert other is not first


def test_the_station_window_shows_what_the_station_reaches_with_its_own_reach_once_a_month(
    context: None,
) -> None:
    app = made_app()
    frame(app)
    station = app.view.station_window.station
    assert station is not None
    delivery, collection = app.view.railway.reach(station)

    first = app.view._built_station_preview(station)
    again = app.view._built_station_preview(station)

    assert again is first
    assert first == app.view.growth.station_trade.preview(station, delivery, collection)


def test_an_autosave_that_cant_be_taken_is_told_and_the_game_plays_on(
    context: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "MONTH_FRAME_SECONDS", 0.0)
    app = made_app()
    app.saves = Saves(tmp_path)
    app.clock.seconds = app.clock.seconds_per_month - 0.001
    monkeypatch.setattr(app.game, "big_parts", lambda: [lambda: None])

    frames_until_the_month_is_done(app)
    frame(app)

    assert app._turning is None
    assert app.notice.startswith("Autosave failed: The game couldn't be saved")
    assert app.saves.listing() == []


def test_a_stations_card_shows_once_the_mouse_has_rested_on_it_a_second(context: None) -> None:
    app = made_app()
    view = app.view
    view.hovered = (10, 6)

    at_first = view.rested_station(5.0)
    soon = view.rested_station(5.0 + STATION_CARD_DELAY / 2)
    later = view.rested_station(5.0 + STATION_CARD_DELAY)
    view.hovered = (10, 7)
    off = view.rested_station(5.0 + STATION_CARD_DELAY * 2)
    view.hovered = (10, 6)
    back = view.rested_station(5.0 + STATION_CARD_DELAY * 2)

    assert (at_first, soon, later, off, back) == (None, None, (10, 6), None, None)


def test_a_stations_card_draws_its_stockpile_waiting_and_demand(context: None) -> None:
    app = made_app()
    imgui.new_frame()

    draw_station_card(app.view.railway, (10, 6), (0.0, 0.0))

    imgui.render()
    assert imgui.internal.find_window_by_name("##station card") is not None


def test_a_frame_that_comes_an_hour_late_lets_only_a_moment_of_play_pass(context: None) -> None:
    app = made_app()
    imgui.get_io().delta_time = 3600.0

    imgui.new_frame()
    app.gui()
    imgui.render()

    assert app.clock.months == 0
    assert app.clock.seconds == pytest.approx(app_module.MOST_SECONDS_A_FRAME * app.clock.speed)
