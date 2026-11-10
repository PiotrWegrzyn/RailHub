import argparse
import random

import numpy as np
from imgui_bundle import imgui, immapp

from railroads import maps
from railroads.mapgen.seeding import play_begins, seed_buildings
from railroads.newgame import start_new_game
from railroads.sim.clock import (
    END_YEAR,
    HISTORY_START,
    START_YEAR,
    GameClock,
)
from railroads.sim.game import Game, place_names
from railroads.sim.growth import MONTHS_BEFORE_PLAY
from railroads.sim.novelties import Novelties
from railroads.sim.stats import History
from railroads.ui.help_window import HelpWindow
from railroads.ui.map_view import MapView
from railroads.ui.reports_window import ReportsWindow
from railroads.ui.tech_window import TechWindow
from railroads.ui.theme import apply_theme
from railroads.ui.top_bar import draw_top_bar


class App:
    def __init__(
        self,
        view: MapView,
        title: str = "",
        rng: np.random.Generator | None = None,
        clock: GameClock | None = None,
    ) -> None:
        self.view = view
        self.title = title
        self.clock = GameClock() if clock is None else clock
        self.rng = np.random.default_rng() if rng is None else rng
        self._styled = False
        self.help = HelpWindow()
        self.tech = TechWindow()
        self.reports = ReportsWindow()

    def gui(self) -> None:
        if not self._styled:
            apply_theme()
            self._styled = True
        seconds = imgui.get_io().delta_time
        months = self.clock.months_in(seconds)
        self.view.time_passed(months)
        first = self.clock.months
        for month in range(first + 1, first + 1 + self.clock.advance(seconds)):
            self.view.month_passed(self.rng, self.clock.date_of(month))
        draw_top_bar(self)
        self.view.gui()
        self.help.draw()
        self.tech.draw(self.view.railway.research)
        growth = self.view.growth
        self.reports.draw(
            growth.history, growth.year, len(self.view.railway.stations()), growth.population()
        )
        typing = imgui.get_io().want_text_input
        if not typing and imgui.is_key_pressed(imgui.Key.h, repeat=False):
            self.help.open = not self.help.open
        if not typing and imgui.is_key_pressed(imgui.Key.r, repeat=False):
            self.tech.open = not self.tech.open
        if not typing and imgui.is_key_pressed(imgui.Key.g, repeat=False):
            self.reports.open = not self.reports.open


def main() -> None:
    parser = argparse.ArgumentParser(prog="railroads", description="A railroad tycoon game.")
    parser.add_argument(
        "--seed", type=int, help="replay the start location and buildings of this seed"
    )
    parser.add_argument("--poland", action="store_true", help="play on the baked map of Poland")
    parser.add_argument(
        "--years",
        type=int,
        default=MONTHS_BEFORE_PLAY // 12,
        help=f"how many years towns grow, from {HISTORY_START}, before play starts in the year "
        "they reach",
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

    seed = random.randrange(1_000_000) if args.seed is None else args.seed
    if args.poland:
        tile_map = maps.load("poland", "hex")
        title = f"Poland, seed {seed}"
    else:
        game = start_new_game(seed)
        tile_map = game.tile_map
        title = f"Seed {seed}, {game.place}"

    months = min(args.years, END_YEAR - HISTORY_START) * 12
    begins = play_begins(months)
    print(f"Growing towns for {months // 12} years, from {HISTORY_START} to {begins}")
    history = History()
    seeded = seed_buildings(tile_map, np.random.default_rng(seed), months=months, history=history)
    game = Game(seeded, start_year=begins, history=history, names=place_names(seeded.tile_map))
    game.growth.novelties = Novelties.pick(game.tile_map, np.random.default_rng(seed + 3))
    view = MapView(game)
    if args.simulate is not None:
        years = min(args.simulate, END_YEAR - begins)
        print(f"Playing {years} years with a stand-in player")
        view.simulate(years, np.random.default_rng(seed + 2))
        title += ", simulated"
    clock = GameClock(first_year=view.growth.year)
    app = App(view, title=title, rng=np.random.default_rng(seed + 1), clock=clock)
    immapp.run(
        gui_function=app.gui,
        window_title=f"Railroads: {title}",
        window_size=(1280, 800),
        with_implot=True,
    )
