import argparse
import random

import numpy as np
from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui, immapp

from railroads import maps
from railroads.mapgen.seeding import seed_buildings
from railroads.newgame import start_new_game
from railroads.sim.clock import MAX_PLAY_SPEED, SPEEDS, GameClock
from railroads.sim.grids import GridKind
from railroads.sim.growth import MONTHS_BEFORE_PLAY
from railroads.sim.tilemap import TileMap
from railroads.ui.map_view import MapView
from railroads.ui.theme import BRASS, apply_theme, toggle_button

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
        self._styled = False

    def gui(self) -> None:
        if not self._styled:
            apply_theme()
            self._styled = True
        seconds = imgui.get_io().delta_time
        months = self.clock.months_in(seconds)
        for view in self.views.values():
            view.time_passed(months)
        first = self.clock.months
        for month in range(first + 1, first + 1 + self.clock.advance(seconds)):
            for view in self.views.values():
                view.month_passed(self.rng, self.clock.date_of(month))
        self._draw_top_bar()
        self.views[self.shown].gui()

    def _draw_top_bar(self) -> None:
        imgui.align_text_to_frame_padding()
        imgui.text_colored(imgui.ImVec4(*BRASS), f"{fa.ICON_FA_CLOCK}  {self.clock.date}")
        imgui.same_line(0, 16)
        self.views[self.shown].draw_money()
        imgui.same_line(0, 16)
        for speed in SPEEDS:
            testing = speed > MAX_PLAY_SPEED
            tooltip = f"Run {speed} times as fast" + (" (for testing)" if testing else "")
            if testing:
                imgui.push_style_var(imgui.StyleVar_.alpha.value, 0.6)
            if toggle_button(f"x{speed}", self.clock.speed == speed, tooltip):
                self.clock.speed = speed
            if testing:
                imgui.pop_style_var()
            imgui.same_line(0, 2)
        imgui.same_line(0, 16)
        for kind, label in GRID_CHOICES.items():
            if toggle_button(label, self.shown == kind, f"Show the map in {label.lower()}"):
                self.shown = kind
            imgui.same_line(0, 2)
        imgui.same_line(0, 16)
        self.views[self.shown].draw_controls()
        if self.title:
            imgui.same_line(0, 16)
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(self.title)


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
    seeded = {
        kind: seed_buildings(tile_maps[kind], np.random.default_rng(seed), months=args.years * 12)
        for kind in GRID_CHOICES
    }
    views = {kind: MapView(seeded[kind].tile_map, seeded[kind]) for kind in GRID_CHOICES}
    app = App(views, shown="hex", title=title, rng=np.random.default_rng(seed + 1))
    immapp.run(gui_function=app.gui, window_title=f"Railroads: {title}", window_size=(1280, 800))
