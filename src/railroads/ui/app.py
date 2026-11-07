import argparse
import random

import numpy as np
from imgui_bundle import imgui, immapp

from railroads import maps
from railroads.mapgen.seeding import seed_buildings
from railroads.newgame import start_new_game
from railroads.sim.clock import GameClock
from railroads.sim.grids import GridKind
from railroads.sim.growth import MONTHS_BEFORE_PLAY
from railroads.sim.tilemap import TileMap
from railroads.ui.map_view import MapView

GRID_CHOICES: dict[GridKind, str] = {"hex": "Hexagons", "square": "Squares"}


class App:
    def __init__(
        self,
        views: dict[GridKind, MapView],
        shown: GridKind,
        title: str = "",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.views = views
        self.shown = shown
        self.title = title
        self.clock = GameClock()
        self.rng = np.random.default_rng() if rng is None else rng

    def gui(self) -> None:
        seconds = imgui.get_io().delta_time
        months = 0.0 if self.clock.finished else seconds / self.clock.seconds_per_month
        for view in self.views.values():
            view.time_passed(months)
        for _ in range(self.clock.advance(seconds)):
            for view in self.views.values():
                view.month_passed(self.rng)
        heading = f"{self.title}   {self.clock.date}" if self.title else self.clock.date
        imgui.text(f"{heading}   Cells:")
        for kind, label in GRID_CHOICES.items():
            imgui.same_line()
            if imgui.radio_button(label, self.shown == kind):
                self.shown = kind
        self.views[self.shown].gui()


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
        help="how many years towns grow before play starts",
    )
    args = parser.parse_args()

    seed = random.randrange(1_000_000) if args.seed is None else args.seed
    tile_maps: dict[GridKind, TileMap]
    if args.poland:
        tile_maps = {kind: maps.load("poland", kind) for kind in GRID_CHOICES}
        title = f"Poland, seed {seed}"
    else:
        game = start_new_game(seed)
        tile_maps = game.maps
        title = f"Seed {seed}, {game.place}"

    print(f"Growing towns for {args.years} years and placing industry")
    views = {
        kind: MapView(
            tile_maps[kind],
            seed_buildings(tile_maps[kind], np.random.default_rng(seed), months=args.years * 12),
        )
        for kind in GRID_CHOICES
    }
    app = App(views, shown="hex", title=title, rng=np.random.default_rng(seed + 1))
    immapp.run(gui_function=app.gui, window_title=f"Railroads: {title}", window_size=(1280, 800))
