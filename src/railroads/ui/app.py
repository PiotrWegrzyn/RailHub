import argparse
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp

from railroads import maps
from railroads.newgame import (
    EARLIEST_START_YEAR,
    MapChoice,
    Settings,
    allowed_start_year,
    make_game_steps,
)
from railroads.saves import ReusedMemory, SavedGame, SaveError, Saves, snapshot
from railroads.sim.clock import END_YEAR, HISTORY_START, START_YEAR, GameClock
from railroads.sim.game import Game
from railroads.sim.growth import MONTHS_BEFORE_PLAY
from railroads.sim.steps import Steps, finish
from railroads.ui.help_window import HelpWindow
from railroads.ui.main_menu import LoadChoice, MainMenu, NewGameChoice, QuitChoice, random_seed
from railroads.ui.map_view import MapView
from railroads.ui.pixel.ground_shader import free_textures
from railroads.ui.reports_window import ReportsWindow
from railroads.ui.saves_window import SavesWindow
from railroads.ui.tech_window import TechWindow
from railroads.ui.theme import BRASS, apply_theme
from railroads.ui.top_bar import draw_top_bar
from railroads.ui.web_setup import page_setup
from railroads.ui.workers import FRAME_SECONDS, FrameWorker, submit_steps, worker

# How long a notice such as "Autosaved" stays in the top bar.
NOTICE_SECONDS = 4.0
WAITING_WIDTH = 320.0
# The window's size when the game opens.
WINDOW_SIZE = (1280.0, 800.0)
# Saves are written one at a time, away from the frames, so play doesn't stutter. Where threads
# cannot start, an autosave is written a step at a time from the frame after it was asked for,
# each frame spending about WRITING_SECONDS on it: steps until that much has passed, one at least.
WRITER = worker("saves")
WRITING_SECONDS = 0.004


class App:
    def __init__(
        self,
        view: MapView,
        title: str = "",
        rng: np.random.Generator | None = None,
        clock: GameClock | None = None,
        saves: Saves | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.view = view
        self.title = title
        self.clock = GameClock() if clock is None else clock
        self.rng = np.random.default_rng() if rng is None else rng
        self.saves = saves
        self.settings = settings
        self._styled = False
        self.help = HelpWindow()
        self.tech = TechWindow()
        self.reports = ReportsWindow()
        self.saves_window = SavesWindow()
        self.notice = ""
        self._notice_until = 0.0
        self._autosaving: Future[object] | None = None
        self._autosave_due = False
        # Kept between autosaves, which are written one at a time.
        self._autosave_memory = ReusedMemory()
        # The month being done a step a frame, if one is.
        self._turning: Steps[None] | None = None
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
        return cls(view, saved.title, saved.rng, saved.clock, saves, saved.settings)

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
        )

    def tell(self, notice: str) -> None:
        self.notice = notice
        self._notice_until = time.monotonic() + NOTICE_SECONDS

    def gui(self) -> None:
        if not self._styled:
            apply_theme()
            self._styled = True
        if self.notice and time.monotonic() > self._notice_until:
            self.notice = ""
        self._start_due_autosave()
        if isinstance(WRITER, FrameWorker):
            WRITER.work_once(WRITING_SECONDS)
        self._check_autosave()
        if self._turning is None:
            self._let_time_pass(imgui.get_io().delta_time)
        else:
            self._turn_once()
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
            dates = [self.clock.date_of(month) for month in range(first + 1, first + 1 + turned)]
            self._turning = self._months(dates)
            self._turn_once()

    def _months(self, dates: list[str]) -> Steps[None]:
        for date in dates:
            yield from self.view.month_steps(self.rng, date)

    def _turn_once(self) -> None:
        assert self._turning is not None
        try:
            next(self._turning)
        except StopIteration:
            self._turning = None
            self._autosave_due = True

    def _draw(self) -> None:
        draw_top_bar(self)
        self.view.gui()
        self.help.draw()
        self.tech.draw(self.view.railway.research)
        if self.reports.open:
            growth = self.view.growth
            self.reports.draw(
                growth.history, growth.year, len(self.view.railway.stations()), growth.population()
            )
        self._draw_saves()
        typing = imgui.get_io().want_text_input
        if not typing and imgui.is_key_pressed(imgui.Key.h, repeat=False):
            self.help.open = not self.help.open
        if not typing and imgui.is_key_pressed(imgui.Key.r, repeat=False):
            self.tech.open = not self.tech.open
        if not typing and imgui.is_key_pressed(imgui.Key.g, repeat=False):
            self.reports.open = not self.reports.open
        if not typing and imgui.is_key_pressed(imgui.Key.f5, repeat=False):
            self.toggle_saves()

    def _keep_trains_smooth(self) -> None:
        """Has the next frame drawn at full speed while a train runs. A few seconds after the
        last input hello_imgui slows to 10 frames a second to save power, and running trains
        would move in jumps while the player only watches."""
        if self.clock.finished:
            return
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
        if self.saves is None or self._autosaving is not None:
            return
        taken = snapshot(self.saved(), self._autosave_memory)
        if isinstance(WRITER, FrameWorker):
            self._autosaving = WRITER.submit_steps(self.saves.autosave_steps(taken))
        else:
            self._autosaving = WRITER.submit(self.saves.autosave, taken)

    def _start_due_autosave(self) -> None:
        """Starts the autosave a month asked for once the last one is written and no month is
        half done, so a month that ends while an autosave is still being written is saved after
        it rather than not at all."""
        if self._autosave_due and self._autosaving is None and self._turning is None:
            self._autosave_due = False
            self.autosave()

    def finish_autosave(self) -> None:
        """Finishes a month being done, then waits for the autosave being written and for one
        that is due."""
        if self._turning is not None:
            finish(self._turning)
            self._turning = None
            self._autosave_due = True
        self._finish_writing()
        self._start_due_autosave()
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
            print(f"Autosave failed: {error}", file=sys.stderr)
            self.tell(f"Autosave failed: {error}")

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
        # Where the setup runs on the frames, it starts once the waiting screen has shown, and
        # each frame spends about this long on it.
        self._waiting_shown = False
        self._frame_seconds = FRAME_SECONDS
        if start is not None:
            self.start(start)

    def report(self, progress: str) -> None:
        print(progress)
        self._progress = progress

    def advance(self, done: float) -> None:
        self._done = done

    def start(self, settings: Settings) -> None:
        self._begin("Setting up a new game", lambda: self._new_game(settings))

    def _new_game(self, settings: Settings) -> Steps[SavedGame]:
        """A new game, grown in the page's setup worker where there is one, so a browser doesn't
        freeze while its towns grow; grown here where there isn't, or if the worker fails, a few
        steps a frame where the frames grow it."""
        setup = page_setup()
        if setup is not None:
            # The frames only look for word from the worker and take up what it grew, a step each.
            self._frame_seconds = 0.0
            saved = yield from setup.grow(settings, self.report, self.advance)
            self._frame_seconds = FRAME_SECONDS
            if saved is not None:
                return saved
        return (yield from make_game_steps(settings, self.report, self.advance))

    def load(self, name: str) -> None:
        self._begin(f"Loading '{name}'", lambda: _at_once(lambda: self.saves.load(name)))

    def _begin(self, doing: str, work: Callable[[], Steps[SavedGame]]) -> None:
        self._doing = doing
        self._progress = ""
        self._done = None
        self._frame_seconds = FRAME_SECONDS
        size, density = WINDOW_SIZE, 1
        if imgui.get_current_context() is not None:
            io = imgui.get_io()
            size = (io.display_size.x, io.display_size.y)
            density = max(1, round(io.display_framebuffer_scale.x))
        # Where frames do the work, the chunks are left to be drawn a frame at a time instead.
        chunks = not isinstance(self._executor, FrameWorker)

        def prepared() -> Steps[App]:
            saved = yield from work()
            yield
            app = App.from_saved(saved, self.saves)
            yield
            app.view.prepare(size, density, chunks)
            return app

        self._task = submit_steps(self._executor, prepared)
        self._waiting_shown = False

    def gui(self) -> None:
        if not self._styled:
            apply_theme()
            self._styled = True
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
            self._follow(self.app)
        else:
            self._choose(self.menu.draw())

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

    def _choose(self, choice: object) -> None:
        match choice:
            case NewGameChoice(settings):
                self.start(settings)
            case LoadChoice(name):
                self.load(name)
            case QuitChoice():
                hello_imgui.get_runner_params().app_shall_exit = True

    def _draw_waiting(self) -> None:
        width, height = imgui.get_window_size()
        imgui.set_cursor_pos((width / 2 - WAITING_WIDTH / 2, height / 2 - 24))
        imgui.begin_group()
        dots = "." * (1 + int(time.monotonic() * 2) % 3)
        imgui.text_colored(imgui.ImVec4(*BRASS), f"{self._doing}{dots:<3}")
        imgui.text_disabled(self._progress or " ")
        if self._done is not None:
            imgui.progress_bar(self._done, imgui.ImVec2(WAITING_WIDTH, 0))
        imgui.end_group()


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
    years = MONTHS_BEFORE_PLAY // 12 if args.years is None else args.years
    baked = maps.start_map("poland" if args.poland else args.map or "")
    return Settings(
        seed=random_seed() if args.seed is None else args.seed,
        map=MapChoice.WORLD if baked is None else MapChoice(baked.title),
        start_year=allowed_start_year(HISTORY_START + years),
        stand_in_years=0 if args.simulate is None else args.simulate,
    )


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
        window_title="Railroads",
        window_size=(int(WINDOW_SIZE[0]), int(WINDOW_SIZE[1])),
        with_implot=True,
    )
