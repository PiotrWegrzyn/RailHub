from imgui_bundle import imgui, immapp

from railroads import maps
from railroads.sim.grids import GridKind
from railroads.ui.map_view import MapView

GRID_CHOICES: dict[GridKind, str] = {"hex": "Hexagons", "square": "Squares"}


class App:
    def __init__(self, views: dict[GridKind, MapView], shown: GridKind) -> None:
        self.views = views
        self.shown = shown

    def gui(self) -> None:
        imgui.text("Cells:")
        for kind, label in GRID_CHOICES.items():
            imgui.same_line()
            if imgui.radio_button(label, self.shown == kind):
                self.shown = kind
        self.views[self.shown].gui()


def main() -> None:
    views = {kind: MapView(maps.load("poland", kind)) for kind in GRID_CHOICES}
    app = App(views, shown="hex")
    immapp.run(gui_function=app.gui, window_title="Railroads", window_size=(1280, 800))
