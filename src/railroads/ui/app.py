import argparse
import random

from imgui_bundle import imgui, immapp

from railroads import maps
from railroads.newgame import start_new_game
from railroads.sim.grids import GridKind
from railroads.sim.tilemap import TileMap
from railroads.ui.map_view import MapView

GRID_CHOICES: dict[GridKind, str] = {"hex": "Hexagons", "square": "Squares"}


class App:
    def __init__(self, views: dict[GridKind, MapView], shown: GridKind, title: str = "") -> None:
        self.views = views
        self.shown = shown
        self.title = title

    def gui(self) -> None:
        imgui.text(f"{self.title}   Cells:" if self.title else "Cells:")
        for kind, label in GRID_CHOICES.items():
            imgui.same_line()
            if imgui.radio_button(label, self.shown == kind):
                self.shown = kind
        self.views[self.shown].gui()


def main() -> None:
    parser = argparse.ArgumentParser(prog="railroads", description="A railroad tycoon game.")
    parser.add_argument("--seed", type=int, help="replay the start location of this seed")
    parser.add_argument("--poland", action="store_true", help="play on the baked map of Poland")
    args = parser.parse_args()

    tile_maps: dict[GridKind, TileMap]
    if args.poland:
        tile_maps = {kind: maps.load("poland", kind) for kind in GRID_CHOICES}
        title = "Poland"
    else:
        seed = random.randrange(1_000_000) if args.seed is None else args.seed
        game = start_new_game(seed)
        tile_maps = game.maps
        title = f"Seed {seed}, {game.place}"

    views = {kind: MapView(tile_maps[kind]) for kind in GRID_CHOICES}
    app = App(views, shown="hex", title=title)
    immapp.run(gui_function=app.gui, window_title=f"Railroads: {title}", window_size=(1280, 800))
