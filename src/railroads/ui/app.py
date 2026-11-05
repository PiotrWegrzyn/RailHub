import math

import numpy as np
from imgui_bundle import imgui, immapp, implot

from railroads.sim.world import Terrain, World

TERRAIN_COLORS: dict[Terrain, tuple[float, float, float, float]] = {
    Terrain.WATER: (0.22, 0.42, 0.74, 1.0),
    Terrain.PLAINS: (0.47, 0.68, 0.34, 1.0),
    Terrain.HILLS: (0.62, 0.56, 0.36, 1.0),
    Terrain.MOUNTAINS: (0.56, 0.54, 0.52, 1.0),
}


class MapView:
    def __init__(self, world: World) -> None:
        self._world = world
        # ImPlot draws heatmap row 0 at the top of the bounds, so flip rows to put tile y=0 at y=0.
        self._heatmap_rows = np.ascontiguousarray(np.flipud(world.terrain))
        self._colormap: implot.Colormap | None = None
        self._hovered_tile: tuple[int, int] | None = None
        self._fit_on_next_frame = True

    def gui(self) -> None:
        self._draw_tile_info()
        self._draw_map()

    def _draw_tile_info(self) -> None:
        if self._hovered_tile is None:
            imgui.text("Drag to pan, scroll to zoom, double-click to reset.")
            return
        x, y = self._hovered_tile
        imgui.text(f"Tile ({x}, {y}): {self._world.terrain_at(x, y).name.title()}")

    def _draw_map(self) -> None:
        world = self._world
        if self._colormap is None:
            colors = np.array(list(TERRAIN_COLORS.values()), dtype=np.float32)
            self._colormap = implot.add_colormap("Terrain", colors, qual=True)

        if self._fit_on_next_frame:
            implot.set_next_axes_to_fit()
            self._fit_on_next_frame = False
        flags = implot.Flags_.equal | implot.Flags_.no_legend | implot.Flags_.no_menus
        if not implot.begin_plot("##map", (-1, -1), flags):
            return
        implot.setup_axes("", "", 0, implot.AxisFlags_.invert)
        implot.push_colormap(self._colormap)
        implot.plot_heatmap(
            "terrain",
            self._heatmap_rows,
            scale_min=0,
            scale_max=len(Terrain) - 1,
            label_fmt="",
            bounds_min=implot.Point(0, 0),
            bounds_max=implot.Point(world.width, world.height),
        )
        implot.pop_colormap()
        self._hovered_tile = self._tile_under_mouse() if implot.is_plot_hovered() else None
        implot.end_plot()

    def _tile_under_mouse(self) -> tuple[int, int] | None:
        mouse = implot.get_plot_mouse_pos()
        x, y = math.floor(mouse.x), math.floor(mouse.y)
        if 0 <= x < self._world.width and 0 <= y < self._world.height:
            return x, y
        return None


def main() -> None:
    world = World.generate(width=256, height=256, seed=1)
    map_view = MapView(world)
    immapp.run(
        gui_function=map_view.gui,
        window_title="Railroads",
        window_size=(1280, 800),
        with_implot=True,
    )
