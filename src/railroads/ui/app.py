import argparse
import gc
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp

from railroads import maps
from railroads.land import buy_lot_steps, lot_middle
from railroads.newgame import (
    EARLIEST_START_YEAR,
    MapChoice,
    Settings,
    allowed_start_year,
    make_game_steps,
    place_map,
)
from railroads.saves import (
    ReusedMemory,
    SavedGame,
    SaveError,
    Saves,
    Snapshot,
    snapshot,
    snapshot_steps,
)
from railroads.sim.clock import (
    END_YEAR,
    HISTORY_START,
    MONTHS_PER_YEAR,
    START_YEAR,
    GameClock,
)
from railroads.sim.game import Game
from railroads.sim.growth import MONTHS_BEFORE_PLAY
from railroads.sim.lots import Lot
from railroads.sim.steps import Steps, budgeted, finish
from railroads.tutorial.game import make_tutorial
from railroads.tutorial.steps import Seen, Tutorial
from railroads.ui import spotlight
from railroads.ui.goal_view import draw_goal_news
from railroads.ui.help_window import HelpWindow
from railroads.ui.history_pictures import HistoryPictures, opening_center
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, ZOOM_CELL_WIDTHS
from railroads.ui.lively import Lively
from railroads.ui.main_menu import (
    LoadChoice,
    MainMenu,
    NewGameChoice,
    QuitChoice,
    TutorialChoice,
    draw_version,
    random_seed,
)
from railroads.ui.map_view import MapView
from railroads.ui.picture_show import PictureShow
from railroads.ui.pixel import ground_shader
from railroads.ui.pixel.ground_shader import free_textures
from railroads.ui.reports_window import ReportsWindow
from railroads.ui.save_files import IN_BROWSER
from railroads.ui.saves_window import SavesWindow
from railroads.ui.theme import BRASS, apply_theme, color_u32
from railroads.ui.tiles_window import TilesWindow
from railroads.ui.top_bar import draw_pause_banner, draw_top_bar
from railroads.ui.tutorial_panel import draw_tutorial
from railroads.ui.waiting_screen import Label, Shatter, tip_for, waiting_line
from railroads.ui.web_setup import go_back_to_landing, page_setup
from railroads.ui.workers import FRAME_SECONDS, FrameWorker, submit_steps, worker

# How long a notice such as "Autosaved" stays in the top bar.
NOTICE_SECONDS = 4.0
WAITING_WIDTH = 320.0
# The panel the waiting screen's lines sit on over a picture of the map: how far it reaches past
# them, and its colour.
PANEL_MARGIN = 12.0
PANEL_COLOUR = (0.11, 0.1, 0.09, 0.85)
# A new game opens on the middle of the map at this zoom, as the pictures of its history show it.
OPENING_ZOOM = ZOOM_CELL_WIDTHS.index(PIXEL_CELL_WIDTH)
# The window's size when the game opens.
WINDOW_SIZE = (1280.0, 800.0)
# Saves are written one at a time, away from the frames, so play doesn't stutter. Where threads
# cannot start, an autosave is written a step at a time from the frame after it was asked for,
# each frame spending about WRITING_SECONDS on it: steps until that much has passed, one at least.
WRITER = worker("saves")
WRITING_SECONDS = 0.004
# The least time between autosaves, in seconds of play: pickling the game stalls a frame, which at
# the highest speed would come every few seconds.
AUTOSAVE_GAP_S = 60.0
# About how long each frame spends on a month, and the autosave after it, while the trains wait:
# their steps are run several to a frame, so a month that costs little takes few frames and the
# trains wait as briefly.
MONTH_FRAME_SECONDS = 0.016
# The most play one frame lets pass, in seconds: a frame that comes late, after the browser tab
# was hidden or the computer slept, carries on from where the game stood instead of running
# through all the months missed.
MOST_SECONDS_A_FRAME = 0.25
# In the browser, how long the main menu's settings stay as they are before the page's setup
# worker starts growing the game they would start, so typing a seed doesn't start one a digit.
AHEAD_AFTER_S = 1.0
LIVELY = Lively()


class App:
    def __init__(
        self,
        view: MapView,
        title: str = "",
        rng: np.random.Generator | None = None,
        clock: GameClock | None = None,
        saves: Saves | None = None,
        settings: Settings | None = None,
        tutorial: Tutorial | None = None,
    ) -> None:
        self.view = view
        self.title = title
        self.clock = GameClock() if clock is None else clock
        self.rng = np.random.default_rng() if rng is None else rng
        self.saves = saves
        self.settings = settings
        # How far the player has got through the tutorial, in a tutorial game.
        self.tutorial = tutorial
        self._styled = False
        self.help = HelpWindow()
        self.tiles = TilesWindow()
        self.reports = ReportsWindow()
        self.saves_window = SavesWindow()
        self.notice = ""
        self._notice_until = 0.0
        self._autosaving: Future[object] | None = None
        self._autosave_due = False
        self._autosaved_at = -AUTOSAVE_GAP_S
        # Kept between autosaves, which are written one at a time.
        self._autosave_memory = ReusedMemory()
        # The month being done a step a frame, if one is, and whether it autosaved at its end.
        self._turning: Steps[bool] | None = None
        # The months turned that wait on the tiles picked for the year they begin.
        self._waiting: list[str] = []
        # What the player asked to leave this game for: a save to load, or the main menu.
        self.load_wanted: str | None = None
        self.menu_wanted = False

    @classmethod
    def from_saved(cls, saved: SavedGame, saves: Saves | None = None) -> App:
        view = MapView(saved.game)
        for summary, seen in reversed(saved.summaries):
            view.summaries.add(summary)
            view.summaries.notes[0].seen = seen
        if saved.camera is not None:
            x, y, zoom = saved.camera
            view.camera.center_x, view.camera.center_y, view.camera.zoom_level = x, y, zoom
        lots = saved.game.lots
        if lots is not None and saved.settings is not None:
            view.land.show_place(place_map(saved.settings), lots)
        return cls(view, saved.title, saved.rng, saved.clock, saves, saved.settings, saved.tutorial)

    @property
    def game(self) -> Game:
        return self.view.game

    def saved(self) -> SavedGame:
        camera = self.view.camera
        return SavedGame(
            self.game,
            self.clock,
            self.rng,
            self.title,
            self.settings,
            [(note.summary, note.seen) for note in self.view.summaries.notes],
            (camera.center_x, camera.center_y, camera.zoom_level),
            self.tutorial,
        )

    def tell(self, notice: str) -> None:
        self.notice = notice
        self._notice_until = time.monotonic() + NOTICE_SECONDS

    def gui(self) -> None:
        LIVELY.keep()
        if not self._styled:
            apply_theme()
            self._styled = True
        if self.notice and time.monotonic() > self._notice_until:
            self.notice = ""
        self._start_due_autosave()
        if isinstance(WRITER, FrameWorker):
            WRITER.work_once(WRITING_SECONDS)
        self._check_autosave()
        goal = self.game.goal
        if self._turning is not None:
            self._turn_once()
        elif self._waiting and not self.tiles.choosing:
            dates, self._waiting = self._waiting, []
            self._turning = budgeted(self._months(dates), MONTH_FRAME_SECONDS)
            self._turn_once()
        elif goal.bankrupt is None and not goal.news and not self.tiles.choosing:
            # Time waits while the player reads news of the target or bankruptcy, and picks
            # the tiles for a new year.
            goal.watch(self.game.finances.cash)
            self._let_time_pass(min(imgui.get_io().delta_time, MOST_SECONDS_A_FRAME))
        busy = self._turning is not None
        # While a month is half done nothing may change the game, so every control is off for
        # those few frames, without greying out.
        imgui.push_style_var(imgui.StyleVar_.disabled_alpha.value, 1.0)
        imgui.begin_disabled(busy)
        try:
            self._draw()
        finally:
            imgui.end_disabled()
            imgui.pop_style_var()
        # Frames come slowly while the player only watches; an autosave written a little each
        # frame is finished at full speed instead.
        writing = self._autosaving is not None and isinstance(WRITER, FrameWorker)
        if busy or writing:
            hello_imgui.request_refresh()

    def _let_time_pass(self, seconds: float) -> None:
        """Moves the trains and the clock on; a month that begins is done a step a frame from
        here, with the trains and the clock waiting, and saved on the frame after it ends."""
        self.view.time_passed(self.clock.months_in(seconds))
        self._keep_trains_smooth()
        first = self.clock.months
        turned = self.clock.advance(seconds)
        if turned:
            months = range(first + 1, first + 1 + turned)
            dates = [self.clock.date_of(month) for month in months]
            januaries = [month for month in months if month % MONTHS_PER_YEAR == 0]
            if januaries and self.tiles.must_choose(
                self.clock.first_year + januaries[0] // MONTHS_PER_YEAR
            ):
                self._waiting = dates
                return
            self._turning = budgeted(self._months(dates), MONTH_FRAME_SECONDS)
            self._turn_once()

    def _months(self, dates: list[str]) -> Steps[bool]:
        """The months' steps and, if an autosave can start once they end, its snapshot's steps
        while the game still waits; returns whether it started one."""
        for date in dates:
            yield from self.view.month_steps(self.rng, date)
        if not self._can_autosave() or time.monotonic() - self._autosaved_at < AUTOSAVE_GAP_S:
            return False
        yield
        self._autosaved_at = time.monotonic()
        try:
            taken = yield from snapshot_steps(self.saved(), self._autosave_memory)
        except SaveError as error:
            self._autosave_failed(error)
            return True
        self._write_autosave(taken)
        return True

    def _turn_once(self) -> None:
        assert self._turning is not None
        try:
            next(self._turning)
        except StopIteration as done:
            self._turning = None
            self._autosave_due = not done.value

    def _draw(self) -> None:
        self._point_the_way()
        draw_top_bar(self)
        self.view.gui()
        self.help.draw()
        self.tiles.draw(self.view.railway.tiles)
        if self.reports.open:
            growth = self.view.growth
            railway = self.view.railway
            self.reports.draw(
                growth.history,
                growth.year,
                len(railway.stations()),
                growth.population(),
                railway.first_year,
                railway.finances,
                railway.fuel,
                self.game.goal,
                railway.tiles,
            )
        self._draw_saves()
        draw_goal_news(self)
        if self.clock.paused and not self.tiles.choosing:
            draw_pause_banner(*self.view.map_rect)
        if self.tutorial is not None:
            goal = self.game.goal
            held_up = bool(self.tiles.open or self.tiles.choosing or goal.news or goal.bankrupt)
            draw_tutorial(self.tutorial, in_front=not held_up)
            if self._turning is None:
                self.tutorial.update(self.game, self.seen())
        typing = imgui.get_io().want_text_input
        if not typing and imgui.is_key_pressed(imgui.Key.h, repeat=False):
            self.help.open = not self.help.open
        if not typing and imgui.is_key_pressed(imgui.Key.y, repeat=False):
            self.tiles.open = not self.tiles.open
        if not typing and imgui.is_key_pressed(imgui.Key.g, repeat=False):
            self.reports.open = not self.reports.open
        if not typing and imgui.is_key_pressed(imgui.Key.f5, repeat=False):
            self.toggle_saves()

    def _point_the_way(self) -> None:
        """Frames what the tutorial's step wants clicked and marks its cells on the map, until
        it is done."""
        tutorial = self.tutorial
        step = None if tutorial is None or tutorial.done else tutorial.current
        spotlight.want(() if step is None else step.spots)
        self.view.spotlit = [] if step is None else step.cells(self.game)

    def seen(self) -> Seen:
        """What the player has open and turned on, for the tutorial."""
        view = self.view
        io = imgui.get_io()
        moods = io.key_shift and not io.want_text_input
        return Seen(
            lens="satisfaction" if moods else None if view.lens is None else view.lens.value,
            selected=view.selected,
            boards_open=self.tiles.open,
            buying_train=view.build.trains.buy.open,
        )

    def _keep_trains_smooth(self) -> None:
        """Has the next frame drawn at full speed while a train runs. A few seconds after the
        last input hello_imgui slows to 10 frames a second to save power, and running trains
        would move in jumps while the player only watches."""
        if any(train.path and not train.giving_way for train in self.view.railway.trains):
            hello_imgui.request_refresh()

    def toggle_saves(self) -> None:
        if self.saves is None:
            return
        if self.saves_window.open:
            self.saves_window.open = False
        else:
            self.saves_window.show(self.saves)

    def autosave(self) -> None:
        """Starts writing the game as the newest autosave, unless the last one is still being
        written."""
        if not self._can_autosave():
            return
        self._autosaved_at = time.monotonic()
        try:
            taken = snapshot(self.saved(), self._autosave_memory)
        except SaveError as error:
            self._autosave_failed(error)
            return
        self._write_autosave(taken)

    def _autosave_failed(self, error: Exception) -> None:
        """Says an autosave couldn't be taken, and plays on."""
        print(f"Autosave failed: {error}", file=sys.stderr)
        self.tell(f"Autosave failed: {error}")

    def _can_autosave(self) -> bool:
        return self.saves is not None and self._autosaving is None

    def _write_autosave(self, taken: Snapshot) -> None:
        assert self.saves is not None
        if isinstance(WRITER, FrameWorker):
            self._autosaving = WRITER.submit_steps(self.saves.autosave_steps(taken))
        else:
            self._autosaving = WRITER.submit(self.saves.autosave, taken)

    def _start_due_autosave(self, right_away: bool = False) -> None:
        """Starts the autosave a month asked for once the last one is written, no month is half
        done and, unless `right_away`, AUTOSAVE_GAP_S have passed since the last, so a month
        that ends while an autosave is still being written is saved after it rather than not
        at all."""
        if self._autosave_due and self._autosaving is None and self._turning is None:
            if time.monotonic() - self._autosaved_at < AUTOSAVE_GAP_S and not right_away:
                return
            self._autosave_due = False
            self.autosave()

    def finish_autosave(self) -> None:
        """Finishes a month being done, then waits for the autosave being written and for one
        that is due."""
        if self._turning is not None:
            autosaved = finish(self._turning)
            self._turning = None
            self._autosave_due = not autosaved
        self._finish_writing()
        self._start_due_autosave(right_away=True)
        self._finish_writing()

    def _finish_writing(self) -> None:
        """Waits for the autosave being written, if one is."""
        if self._autosaving is None:
            return
        if isinstance(WRITER, FrameWorker):
            WRITER.work_once()
        self._autosaving.exception()
        self._check_autosave()

    def _check_autosave(self) -> None:
        if self._autosaving is None or not self._autosaving.done():
            return
        error = self._autosaving.exception()
        self._autosaving = None
        # The autosaves have moved down a place, so an open saves window lists them anew: what
        # it lists is what loads, the newest autosave showing as soon as it is written.
        if self.saves is not None and self.saves_window.open:
            self.saves_window.refresh(self.saves)
        if error is None:
            self.tell("Autosaved")
        else:
            self._autosave_failed(error)

    def _draw_saves(self) -> None:
        if self.saves is None:
            return
        request = self.saves_window.draw(self.saves)
        if request is None:
            return
        if request.kind == "load":
            self.load_wanted = request.name
            return
        try:
            self.saves.save(request.name, self.saved())
        except SaveError as error:
            self.saves_window.message = str(error)
            return
        self.saves_window.message = f"Saved as '{request.name}'."
        self.saves_window.refresh(self.saves)
        self.tell(f"Saved as '{request.name}'")


class Shell:
    """The whole program: the main menu, a game being set up or loaded, or a game in play. All
    of them draw into the window it is called in, hello_imgui's full-screen one."""

    def __init__(self, saves: Saves, start: Settings | None = None) -> None:
        self.saves = saves
        self.menu = MainMenu(saves)
        self.app: App | None = None
        self._styled = False
        self._task: Future[App] | None = None
        self._doing = ""
        self._progress = ""
        # How much of the work is done, where it can tell.
        self._done: float | None = None
        self._executor = worker("setup")
        # Whether the work is a new game, and on a random place; when it began; what the waiting
        # screen last showed; and the screen breaking up over a new game just begun.
        self._new_game_on: MapChoice | None = None
        self._pictured = False
        self._began = 0.0
        self._labels: list[Label] = []
        self._tip = ""
        self._shatter: Shatter | None = None
        # The pictures of a new map's history, kept until the screen has broken up, and the size
        # in points of the window they are drawn for.
        self._show = PictureShow()
        self._window = (int(WINDOW_SIZE[0]), int(WINDOW_SIZE[1]))
        # Where the setup runs on the frames, it starts once the waiting screen has shown, and
        # each frame spends about this long on it.
        self._waiting_shown = False
        self._frame_seconds = FRAME_SECONDS
        # The game the command line asks for, started on the first frame, once the window's size
        # is known.
        self._wanted = start
        # The menu's settings as last seen and since when they have stayed so.
        self._menu_settings: Settings | None = None
        self._menu_since = 0.0

    def report(self, progress: str) -> None:
        print(progress)
        self._progress = progress

    def advance(self, done: float) -> None:
        self._done = done

    def start(self, settings: Settings) -> None:
        self._begin(
            "Setting up a new game",
            lambda: self._new_game(settings),
            settings.map,
            settings.show_map_growing,
        )

    def _new_game(self, settings: Settings) -> Steps[SavedGame]:
        """A new game, grown in the page's setup worker where there is one, so a browser doesn't
        freeze while its towns grow; grown here where there isn't, or if the worker fails, a few
        steps a frame where the frames grow it."""
        pictured = settings.show_map_growing
        size = self._window if pictured else None
        see = self._show.see
        over_ground = _over_ground()
        setup = page_setup()
        if setup is not None:
            # The frames only look for word from the worker and take up what it grew, a step each.
            self._frame_seconds = 0.0
            saved = yield from setup.grow(
                settings, self.report, self.advance, see, size, over_ground
            )
            self._frame_seconds = FRAME_SECONDS
            if saved is not None:
                return saved
        if not pictured:
            return (yield from make_game_steps(settings, self.report, self.advance))
        pictures = HistoryPictures(self._window, see, over_ground)
        return (yield from make_game_steps(settings, self.report, self.advance, pictures))

    def start_tutorial(self) -> None:
        """Starts the tutorial's premade game (railroads.tutorial), the same every time."""
        self._begin("Setting up the tutorial", lambda: _at_once(make_tutorial))

    def load(self, name: str) -> None:
        self._begin(f"Loading '{name}'", lambda: _at_once(lambda: self.saves.load(name)))

    def buy_land(self, app: App, lot: Lot) -> None:
        """Buys a lot next to the map (railroads.land), growing it while the waiting screen
        shows, as a new game's towns grow, then opens the game on the middle of it."""
        app.finish_autosave()
        saved = app.saved()
        zoom = app.view.camera.zoom_level
        self._begin(
            "Buying land",
            lambda: buy_lot_steps(saved, lot, self.report, self.advance),
            visit=(lot, zoom),
        )

    def _begin(
        self,
        doing: str,
        work: Callable[[], Steps[SavedGame]],
        new_game_on: MapChoice | None = None,
        pictured: bool = False,
        visit: tuple[Lot, int] | None = None,
    ) -> None:
        """Starts the work of setting up or loading a game; `new_game_on` is the map of a new
        game, None for a game loaded, and `pictured` whether its map shows as its towns grow."""
        self._doing = doing
        self._new_game_on = new_game_on
        self._pictured = pictured
        self._began = time.monotonic()
        self._shatter = None
        self._tip = tip_for(np.random.default_rng())
        self._progress = ""
        self._done = None
        self._frame_seconds = FRAME_SECONDS
        size, density = WINDOW_SIZE, 1
        if imgui.get_current_context() is not None:
            io = imgui.get_io()
            size = (io.display_size.x, io.display_size.y)
            density = max(1, round(io.display_framebuffer_scale.x))
        self._window = (int(size[0]), int(size[1]))
        self._show = PictureShow(density)
        # Where frames do the work, the chunks are left to be drawn a frame at a time instead.
        chunks = not isinstance(self._executor, FrameWorker)

        def prepared() -> Steps[App]:
            saved = yield from work()
            yield
            app = App.from_saved(saved, self.saves)
            if new_game_on is not None and pictured:
                camera = app.view.camera
                center = opening_center(app.view.tile_map.grid, size)
                camera.center_x, camera.center_y = center
                camera.zoom_level = OPENING_ZOOM
            elif new_game_on is not None:
                app.view.visit_biggest_town()
            lots = app.game.lots
            if visit is not None and lots is not None:
                lot, zoom = visit
                app.view.camera.zoom_level = zoom
                app.view.look_at(lot_middle(lots, lot))
            yield
            yield from app.view.prepare_steps(size, density, chunks)
            if not chunks:
                yield
                app.view.ready_shaders()
            return app

        setup = page_setup()
        if new_game_on is None and setup is not None:
            setup.stop_ahead()
        self._task = submit_steps(self._executor, prepared)
        self._waiting_shown = False

    def gui(self) -> None:
        LIVELY.keep()
        if not self._styled:
            apply_theme()
            self._styled = True
        if self._wanted is not None:
            self.start(self._wanted)
            self._wanted = None
        # Chunk textures of a game left behind are deleted here, as the menu or a game shown
        # zoomed out draws none that would delete them.
        free_textures()
        if self._task is not None:
            if self._waiting_shown and isinstance(self._executor, FrameWorker):
                # Steps for the frame's share of time, one at least, so the waiting screen shows
                # between them.
                self._executor.work_once(self._frame_seconds)
            self._draw_waiting()
            self._waiting_shown = True
            if self._task.done():
                self._finish(self._task)
        elif self.app is not None:
            self.app.gui()
            if self._shatter is not None and not self._shatter.draw(imgui.get_io().delta_time):
                self._shatter = None
                self._show = PictureShow()
            self._follow(self.app)
        else:
            self._choose(self.menu.draw())
            if self._task is None:
                self._grow_ahead()

    def _grow_ahead(self) -> None:
        """Where the page has a setup worker, has it grow the game the menu would start once
        the menu's settings have stayed the same for AHEAD_AFTER_S, so Start finds it grown or
        on the way; not a game whose map is pictured as it grows, which needs the waiting
        screen."""
        setup = page_setup()
        if setup is None:
            return
        settings = self.menu.settings()
        now = time.monotonic()
        if settings != self._menu_settings:
            self._menu_settings, self._menu_since = settings, now
        if settings.show_map_growing:
            setup.stop_ahead()
        elif now - self._menu_since >= AHEAD_AFTER_S:
            setup.grow_ahead(settings, _over_ground())

    def _finish(self, task: Future[App]) -> None:
        self._task = None
        try:
            app = task.result()
        except (SaveError, OSError) as error:
            self._failed(str(error))
            return
        if self.app is not None:
            self.app.finish_autosave()
        self.app = app
        freeze_garbage()
        if self._new_game_on is not None:
            size = imgui.get_io().display_size
            # The see-through background as it shows over the black behind every window.
            bg = imgui.get_style_color_vec4(imgui.Col_.window_bg.value)
            colour = imgui.get_color_u32(imgui.ImVec4(bg.x * bg.w, bg.y * bg.w, bg.z * bg.w, 1.0))
            self._shatter = Shatter(
                size.x, size.y, colour, self._labels, pictures=self._show.textures
            )

    def _failed(self, message: str) -> None:
        if self.app is not None:
            self.app.saves_window.message = message
        else:
            self.menu.error = message

    def _follow(self, app: App) -> None:
        if app.load_wanted is not None:
            name, app.load_wanted = app.load_wanted, None
            self.load(name)
        elif app.menu_wanted:
            app.finish_autosave()
            self.app = None
            self.menu.go_to("main")
        elif app.view.land.wanted is not None:
            lot, app.view.land.wanted = app.view.land.wanted, None
            self.buy_land(app, lot)

    def _choose(self, choice: object) -> None:
        match choice:
            case NewGameChoice(settings):
                self.start(settings)
            case TutorialChoice():
                self.start_tutorial()
            case LoadChoice(name):
                self.load(name)
            case QuitChoice() if IN_BROWSER:
                go_back_to_landing()
            case QuitChoice():
                hello_imgui.get_runner_params().app_shall_exit = True

    def _draw_waiting(self) -> None:
        width, height = imgui.get_window_size()
        draw_list = imgui.get_window_draw_list()
        self._show.draw(draw_list, imgui.get_window_pos(), (width, height))
        year = self._show.year
        draw_list.channels_split(2)
        draw_list.channels_set_current(1)
        imgui.set_cursor_pos((width / 2 - WAITING_WIDTH / 2, height / 2 - 24))
        imgui.begin_group()
        dots = "." * (1 + int(time.monotonic() * 2) % 3)
        self._labels = [
            self._text(f"{self._waiting_line()}{dots:<3}", imgui.ImVec4(*BRASS)),
            self._text(
                self._progress or " ", imgui.get_style_color_vec4(imgui.Col_.text_disabled.value)
            ),
        ]
        if year is not None:
            self._labels.append(self._text(f"The map in {year}", imgui.ImVec4(*BRASS)))
        if self._done is not None:
            corner = imgui.get_cursor_screen_pos()
            imgui.progress_bar(self._done, imgui.ImVec2(WAITING_WIDTH, 0))
            bar_height = imgui.get_item_rect_size().y
            self._labels += [
                Label(corner.x, corner.y, WAITING_WIDTH, bar_height,
                      imgui.get_color_u32(imgui.Col_.frame_bg.value)),
                Label(corner.x, corner.y, WAITING_WIDTH * self._done, bar_height,
                      imgui.get_color_u32(imgui.Col_.plot_histogram.value)),
            ]  # fmt: skip
        if self._tip:
            self._labels.append(
                self._text(
                    f"Tip: {self._tip}",
                    imgui.get_style_color_vec4(imgui.Col_.text_disabled.value),
                )
            )
        imgui.end_group()
        if year is not None:
            draw_list.channels_set_current(0)
            top_left, bottom_right = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            panel = Label(
                top_left.x - PANEL_MARGIN,
                top_left.y - PANEL_MARGIN,
                bottom_right.x - top_left.x + 2 * PANEL_MARGIN,
                bottom_right.y - top_left.y + 2 * PANEL_MARGIN,
                color_u32(PANEL_COLOUR),
            )
            draw_list.add_rect_filled(
                (panel.x, panel.y), (panel.x + panel.width, panel.y + panel.height), panel.colour
            )
            self._labels.insert(0, panel)
        draw_list.channels_merge()
        draw_version()

    def _waiting_line(self) -> str:
        """What the waiting screen is doing: make-believe work for a new game set up without
        pictures of its map growing."""
        if self._new_game_on is None or self._pictured:
            return self._doing
        random_place = self._new_game_on == MapChoice.WORLD
        return waiting_line(time.monotonic() - self._began, random_place)

    @staticmethod
    def _text(text: str, colour: imgui.ImVec4) -> Label:
        """Shows a line of the waiting screen, and where it is for the screen's breaking up."""
        corner = imgui.get_cursor_screen_pos()
        size = imgui.calc_text_size(text)
        imgui.text_colored(colour, text)
        return Label(corner.x, corner.y, size.x, size.y, imgui.get_color_u32(colour), text)


def _over_ground() -> bool:
    """Whether pictures of a new map growing leave the ground to the graphics card."""
    return ground_shader.GL is not None and hello_imgui.is_using_hello_imgui()


def _at_once[T](work: Callable[[], T]) -> Steps[T]:
    """Work done in one go, as steps: it starts after the first pause."""
    yield
    return work()


def settings_from(args: argparse.Namespace) -> Settings | None:
    """The settings the command line asks for, or None to open the main menu."""
    if (
        args.seed is None
        and not args.poland
        and args.map is None
        and args.years is None
        and args.simulate is None
    ):
        return None
    years = MONTHS_BEFORE_PLAY // MONTHS_PER_YEAR if args.years is None else args.years
    baked = maps.start_map("poland" if args.poland else args.map or "")
    return Settings(
        seed=random_seed() if args.seed is None else args.seed,
        map=MapChoice.WORLD if baked is None else MapChoice(baked.title),
        start_year=allowed_start_year(HISTORY_START + years),
        stand_in_years=0 if args.simulate is None else args.simulate,
    )


def freeze_garbage() -> None:
    """Sets the objects of a game just begun or loaded, and those of the menu and the program,
    aside from Python's garbage collector: they last as long as the game, and a full collection
    walking them all stalls a frame for tens of milliseconds. What was set aside for a game left
    is let back in first, so it can go."""
    gc.unfreeze()
    gc.collect()
    gc.freeze()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line's options, refusing a history that begins play before
    EARLIEST_START_YEAR."""
    parser = argparse.ArgumentParser(
        prog="railroads",
        description="A railroad tycoon game. Without options it opens the main menu; with any, "
        "it starts that game straight away.",
    )
    parser.add_argument(
        "--seed", type=int, help="replay the start location and buildings of this seed"
    )
    parser.add_argument("--poland", action="store_true", help="play on the baked map of Poland")
    parser.add_argument(
        "--map",
        choices=[start.key for start in maps.START_MAPS],
        help="play on this baked map of a real place",
    )
    parser.add_argument(
        "--years",
        type=int,
        help=f"how many years towns grow, from {HISTORY_START}, before play starts in the year "
        f"they reach: at least {EARLIEST_START_YEAR - HISTORY_START}, so play begins in "
        f"{EARLIEST_START_YEAR} or later ({START_YEAR - HISTORY_START} is suggested and the "
        "default)",
    )
    parser.add_argument(
        "--simulate",
        type=int,
        nargs="?",
        const=END_YEAR - START_YEAR,
        metavar="YEARS",
        help="let a stand-in player build stations and serve the towns for this many years "
        "(all of play if left out), then open the map where it stopped",
    )
    args = parser.parse_args(argv)
    if args.years is not None and HISTORY_START + args.years < EARLIEST_START_YEAR:
        parser.error(
            f"--years must be at least {EARLIEST_START_YEAR - HISTORY_START}, so play begins in "
            f"{EARLIEST_START_YEAR} or later; {START_YEAR - HISTORY_START} (play beginning in "
            f"{START_YEAR}) is suggested"
        )
    return args


def main() -> None:
    args = parse_args()
    shell = Shell(Saves(), settings_from(args))
    immapp.run(
        gui_function=shell.gui,
        window_title="RailHub",
        window_size=(int(WINDOW_SIZE[0]), int(WINDOW_SIZE[1])),
        with_implot=True,
    )
