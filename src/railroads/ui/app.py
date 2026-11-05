import numpy as np
import numpy.typing as npt
from imgui_bundle import hello_imgui, imgui, immapp, implot

from railroads.sim.maps import poland
from railroads.sim.units import KM_PER_TILE
from railroads.sim.world import Terrain, World

TERRAIN_COLORS: dict[Terrain, tuple[int, int, int]] = {
    Terrain.WATER: (56, 107, 189),
    Terrain.PLAINS: (120, 173, 87),
    Terrain.HILLS: (158, 143, 92),
    Terrain.MOUNTAINS: (143, 138, 133),
}
GRID_COLOR = (0.0, 0.0, 0.0, 0.25)
MIN_TILE_PIXELS_FOR_GRID = 8
# The renderer samples textures with linear filtering; drawing each tile as a block of
# identical pixels keeps tile edges sharp when zoomed in.
PIXELS_PER_TILE = 8


def terrain_rgba(terrain: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    palette = np.zeros((len(Terrain), 4), dtype=np.uint8)
    for kind, (r, g, b) in TERRAIN_COLORS.items():
        palette[kind] = (r, g, b, 255)
    pixels = palette[terrain].repeat(PIXELS_PER_TILE, axis=0).repeat(PIXELS_PER_TILE, axis=1)
    return np.ascontiguousarray(pixels)


class MapView:
    def __init__(self, world: World) -> None:
        self._world = world
        self._texture: hello_imgui.TextureGpu | None = None
        self._hovered_tile: tuple[int, int] | None = None
        self._fit_on_next_frame = True

    def gui(self) -> None:
        self._draw_tile_info()
        self._draw_map()

    def _draw_tile_info(self) -> None:
        if self._hovered_tile is None:
            imgui.text("Drag to pan, scroll to zoom, double-click to fit the map.")
            return
        x, y = self._hovered_tile
        terrain = self._world.terrain_at(x, y).name.title()
        imgui.text(f"Tile ({x}, {y}) at {x * KM_PER_TILE:g} km, {y * KM_PER_TILE:g} km: {terrain}")

    def _draw_map(self) -> None:
        world = self._world
        if self._texture is None:
            self._texture = hello_imgui.create_texture_gpu_from_rgba_data(
                terrain_rgba(world.terrain)
            )
        if self._fit_on_next_frame:
            implot.set_next_axes_to_fit()
            self._fit_on_next_frame = False

        flags = implot.Flags_.equal | implot.Flags_.no_legend | implot.Flags_.no_menus
        if not implot.begin_plot("##map", (-1, -1), flags):
            return
        implot.setup_axes("", "", 0, implot.AxisFlags_.invert)
        implot.setup_axis_format(implot.ImAxis_.x1, "%g km")
        implot.setup_axis_format(implot.ImAxis_.y1, "%g km")
        # The y axis grows downwards (row 0 at the top), so flip the texture's v coordinate.
        implot.plot_image(
            "terrain",
            imgui.ImTextureRef(self._texture.texture_id()),
            bounds_min=implot.Point(0, 0),
            bounds_max=implot.Point(world.width_km, world.height_km),
            uv0=(0, 1),
            uv1=(1, 0),
        )
        self._draw_tile_grid()
        mouse = implot.get_plot_mouse_pos()
        hovered = implot.is_plot_hovered()
        self._hovered_tile = world.tile_at_km(mouse.x, mouse.y) if hovered else None
        implot.end_plot()

    def _draw_tile_grid(self) -> None:
        limits = implot.get_plot_limits()
        pixels_per_km = implot.get_plot_size().x / (limits.x.max - limits.x.min)
        if pixels_per_km * KM_PER_TILE < MIN_TILE_PIXELS_FOR_GRID:
            return
        world = self._world
        y_low, y_high = sorted((limits.y.min, limits.y.max))
        x0, x1 = _visible_lines(limits.x.min, limits.x.max, world.width)
        y0, y1 = _visible_lines(y_low, y_high, world.height)
        xs = np.arange(x0, x1 + 1, dtype=np.float64) * KM_PER_TILE
        ys = np.arange(y0, y1 + 1, dtype=np.float64) * KM_PER_TILE
        top, bottom = y0 * KM_PER_TILE, y1 * KM_PER_TILE
        left, right = x0 * KM_PER_TILE, x1 * KM_PER_TILE
        seg_x = np.concatenate([np.repeat(xs, 2), np.tile([left, right], len(ys))])
        seg_y = np.concatenate([np.tile([top, bottom], len(xs)), np.repeat(ys, 2)])
        spec = implot.Spec(line_color=GRID_COLOR)
        spec.flags = implot.LineFlags_.segments | implot.ItemFlags_.no_fit
        implot.plot_line("grid", seg_x, seg_y, spec=spec)


def _visible_lines(low_km: float, high_km: float, tile_count: int) -> tuple[int, int]:
    first = max(0, int(low_km / KM_PER_TILE))
    last = min(tile_count, int(high_km / KM_PER_TILE) + 1)
    return first, last


def main() -> None:
    world = poland.generate()
    map_view = MapView(world)
    immapp.run(
        gui_function=map_view.gui,
        window_title="Railroads",
        window_size=(1280, 800),
        with_implot=True,
    )
