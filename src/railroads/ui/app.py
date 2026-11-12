import argparse
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp

from railroads.newgame import MapChoice, Settings, make_game
from railroads.saves import SavedGame, SaveError, Saves, snapshot
from railroads.sim.clock import END_YEAR, HISTORY_START, START_YEAR, GameClock
from railroads.sim.game import Game
from railroads.sim.growth import MONTHS_BEFORE_PLAY
from railroads.ui.help_window import HelpWindow
from railroads.ui.main_menu import LoadChoice, MainMenu, NewGameChoice, QuitChoice, random_seed
from railroads.ui.map_view import MapView
from railroads.ui.reports_window import ReportsWindow
from railroads.ui.saves_window import SavesWindow
from railroads.ui.tech_window import TechWindow
from railroads.ui.theme import BRASS, apply_theme
from railroads.ui.top_bar import draw_top_bar
from railroads.ui.workers import FrameWorker, worker

# How long a notice such as "Autosaved" stays in the top bar.
NOTICE_SECONDS = 4.0
WAITING_WIDTH = 320.0
# Saves are written one at a time, away from the frames, so play doesn't stutter. Where threads
# cannot start, an autosave is written on the frame after it was asked for.
WRITER = worker("saves")


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
        if isinstance(WRITER, FrameWorker):
            WRITER.work_once()
        self._check_autosave()
        seconds = imgui.get_io().delta_time
        months = self.clock.months_in(seconds)
        self.view.time_passed(months)
        self._keep_trains_smooth()
        first = self.clock.months
        turned = self.clock.advance(seconds)
        for month in range(first + 1, first + 1 + turned):
            self.view.month_passed(self.rng, self.clock.date_of(month))
        if turned:
            self.autosave()
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
        self._autosaving = WRITER.submit(self.saves.autosave, snapshot(self.saved()))

    def finish_autosave(self) -> None:
        """Waits for an autosave being written."""
        if self._autosaving is not None:
            if isinstance(WRITER, FrameWorker):
                WRITER.work_once()
            self._autosaving.exception()
            self._check_autosave()

    def _check_autosave(self) -> None:
        if self._autosaving is None or not self._autosaving.done():
            return
        error = self._autosaving.exception()
        self._autosaving = None
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
        self._task: Future[SavedGame] | None = None
        self._doing = ""
        self._progress = ""
        self._executor = worker("setup")
        # Where the setup runs on the frames, it starts once the waiting screen has shown.
        self._waiting_shown = False
        if start is not None:
            self.start(start)

    def report(self, progress: str) -> None:
        print(progress)
        self._progress = progress

    def start(self, settings: Settings) -> None:
        self._begin("Setting up a new game", lambda: make_game(settings, self.report))

    def load(self, name: str) -> None:
        self._begin(f"Loading '{name}'", lambda: self.saves.load(name))

    def _begin(self, doing: str, work: Callable[[], SavedGame]) -> None:
        self._doing = doing
        self._progress = ""
        self._task = self._executor.submit(work)
        self._waiting_shown = False

    def gui(self) -> None:
        if not self._styled:
            apply_theme()
            self._styled = True
        if self._task is not None:
            if self._waiting_shown and isinstance(self._executor, FrameWorker):
                self._executor.work_once()
            self._draw_waiting()
            self._waiting_shown = True
            if self._task.done():
                self._finish(self._task)
        elif self.app is not None:
            self.app.gui()
            self._follow(self.app)
        else:
            self._choose(self.menu.draw())

    def _finish(self, task: Future[SavedGame]) -> None:
        self._task = None
        try:
            saved = task.result()
        except (SaveError, OSError) as error:
            self._failed(str(error))
            return
        if self.app is not None:
            self.app.finish_autosave()
        self.app = App.from_saved(saved, self.saves)

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
        imgui.end_group()


def settings_from(args: argparse.Namespace) -> Settings | None:
    """The settings the command line asks for, or None to open the main menu."""
    if args.seed is None and not args.poland and args.years is None and args.simulate is None:
        return None
    years = MONTHS_BEFORE_PLAY // 12 if args.years is None else args.years
    return Settings(
        seed=random_seed() if args.seed is None else args.seed,
        map=MapChoice.POLAND if args.poland else MapChoice.WORLD,
        start_year=HISTORY_START + min(years, END_YEAR - HISTORY_START),
        stand_in_years=0 if args.simulate is None else args.simulate,
    )


def main() -> None:
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
        "--years",
        type=int,
        help=f"how many years towns grow, from {HISTORY_START}, before play starts in the year "
        f"they reach (default {START_YEAR - HISTORY_START})",
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
    args = parser.parse_args()
    shell = Shell(Saves(), settings_from(args))
    immapp.run(
        gui_function=shell.gui,
        window_title="Railroads",
        window_size=(1280, 800),
        with_implot=True,
    )
