import math

import numpy as np
import numpy.typing as npt
from imgui_bundle import hello_imgui, imgui, immapp

from railroads.sim.maps import poland
from railroads.sim.units import KM_PER_TILE
from railroads.sim.world import Terrain, World
from railroads.ui.iso_camera import IsoCamera, Vec2

TERRAIN_COLORS: dict[Terrain, tuple[int, int, int]] = {
    Terrain.WATER: (56, 107, 189),
    Terrain.PLAINS: (120, 173, 87),
    Terrain.HILLS: (158, 143, 92),
    Terrain.MOUNTAINS: (143, 138, 133),
}
BACKGROUND_COLOR = (0.08, 0.09, 0.11, 1.0)
GRID_COLOR = (0.0, 0.0, 0.0, 0.25)
HOVER_COLOR = (1.0, 1.0, 1.0, 0.9)
MIN_TILE_WIDTH_FOR_GRID = 16
# Camera rotation (quarter turns) that puts map north (row 0) towards the top-left of the screen.
NORTH_TOP_LEFT = 3
# The renderer samples textures with linear filtering; drawing each tile as a block of
# identical pixels keeps tile edges sharp when zoomed in.
PIXELS_PER_TILE = 8


def terrain_rgba(terrain: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    palette = np.zeros((len(Terrain), 4), dtype=np.uint8)
    for kind, (r, g, b) in TERRAIN_COLORS.items():
        palette[kind] = (r, g, b, 255)
    pixels = palette[terrain].repeat(PIXELS_PER_TILE, axis=0).repeat(PIXELS_PER_TILE, axis=1)
    return np.ascontiguousarray(pixels)


def _color(rgba: tuple[float, float, float, float]) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))


def _any_key_pressed(*keys: imgui.Key) -> bool:
    return any(imgui.is_key_pressed(key, repeat=False) for key in keys)


class MapView:
    def __init__(self, world: World) -> None:
        self._world = world
        self.camera = IsoCamera(
            center_x=world.width / 2, center_y=world.height / 2, rotation=NORTH_TOP_LEFT
        )
        self._texture: hello_imgui.TextureGpu | None = None
        self._hovered_tile: tuple[int, int] | None = None
        self._wheel = 0.0

    def gui(self) -> None:
        self._draw_tile_info()
        self._draw_map()

    def _draw_tile_info(self) -> None:
        if imgui.button("Turn left (Q)") or imgui.is_key_pressed(imgui.Key.q, repeat=False):
            self.camera.turn(-1)
        imgui.same_line()
        if imgui.button("Turn right (E)") or imgui.is_key_pressed(imgui.Key.e, repeat=False):
            self.camera.turn(1)
        imgui.same_line()
        if imgui.button(" - ") or _any_key_pressed(imgui.Key.minus, imgui.Key.keypad_subtract):
            self.camera.zoom(-1, (0, 0), (0, 0))
        imgui.same_line()
        if imgui.button(" + ") or _any_key_pressed(imgui.Key.equal, imgui.Key.keypad_add):
            self.camera.zoom(1, (0, 0), (0, 0))
        imgui.same_line()
        zoom = f"Zoom {self.camera.zoom_level + 1}/5"
        if self._hovered_tile is None:
            imgui.text(f"{zoom}. Drag to pan, scroll or -/+ to zoom.")
            return
        x, y = self._hovered_tile
        terrain = self._world.terrain_at(x, y).name.title()
        km = f"{x * KM_PER_TILE:g} km, {y * KM_PER_TILE:g} km"
        imgui.text(f"{zoom}. Tile ({x}, {y}) at {km}: {terrain}")

    def _draw_map(self) -> None:
        if self._texture is None:
            self._texture = hello_imgui.create_texture_gpu_from_rgba_data(
                terrain_rgba(self._world.terrain)
            )
        origin = imgui.get_cursor_screen_pos()
        size = imgui.get_content_region_avail()
        if size.x <= 0 or size.y <= 0:
            return
        imgui.invisible_button("map", size)
        center = (origin.x + size.x / 2, origin.y + size.y / 2)
        self._handle_input(center)

        draw_list = imgui.get_window_draw_list()
        top_left = (origin.x, origin.y)
        bottom_right = (origin.x + size.x, origin.y + size.y)
        draw_list.push_clip_rect(top_left, bottom_right, True)
        draw_list.add_rect_filled(top_left, bottom_right, _color(BACKGROUND_COLOR))
        self._draw_terrain(draw_list, center)
        if self.camera.tile_width >= MIN_TILE_WIDTH_FOR_GRID:
            self._draw_grid(draw_list, center, top_left, bottom_right)
        if self._hovered_tile is not None:
            self._draw_tile_outline(draw_list, center, self._hovered_tile)
        draw_list.pop_clip_rect()

    def _handle_input(self, center: Vec2) -> None:
        io = imgui.get_io()
        if imgui.is_item_active():
            self.camera.pan(io.mouse_delta.x, io.mouse_delta.y)
        hovered = imgui.is_item_hovered()
        if hovered:
            self._wheel += io.mouse_wheel
            steps = int(self._wheel)
            if steps:
                self._wheel -= steps
                self.camera.zoom(steps, (io.mouse_pos.x, io.mouse_pos.y), center)
        self._hovered_tile = None
        if hovered:
            tx, ty = self.camera.to_tile(io.mouse_pos.x, io.mouse_pos.y, center)
            x, y = math.floor(tx), math.floor(ty)
            if 0 <= x < self._world.width and 0 <= y < self._world.height:
                self._hovered_tile = (x, y)

    def _draw_terrain(self, draw_list: imgui.ImDrawList, center: Vec2) -> None:
        assert self._texture is not None
        w, h = self._world.width, self._world.height
        to_screen = self.camera.to_screen
        draw_list.add_image_quad(
            imgui.ImTextureRef(self._texture.texture_id()),
            to_screen(0, 0, center),
            to_screen(w, 0, center),
            to_screen(w, h, center),
            to_screen(0, h, center),
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
        )

    def _draw_grid(
        self, draw_list: imgui.ImDrawList, center: Vec2, top_left: Vec2, bottom_right: Vec2
    ) -> None:
        corners = [
            self.camera.to_tile(x, y, center)
            for x in (top_left[0], bottom_right[0])
            for y in (top_left[1], bottom_right[1])
        ]
        x0 = max(0, math.floor(min(c[0] for c in corners)))
        x1 = min(self._world.width, math.ceil(max(c[0] for c in corners)))
        y0 = max(0, math.floor(min(c[1] for c in corners)))
        y1 = min(self._world.height, math.ceil(max(c[1] for c in corners)))
        color = _color(GRID_COLOR)
        to_screen = self.camera.to_screen
        for x in range(x0, x1 + 1):
            draw_list.add_line(to_screen(x, y0, center), to_screen(x, y1, center), color)
        for y in range(y0, y1 + 1):
            draw_list.add_line(to_screen(x0, y, center), to_screen(x1, y, center), color)

    def _draw_tile_outline(
        self, draw_list: imgui.ImDrawList, center: Vec2, tile: tuple[int, int]
    ) -> None:
        x, y = tile
        to_screen = self.camera.to_screen
        draw_list.add_quad(
            to_screen(x, y, center),
            to_screen(x + 1, y, center),
            to_screen(x + 1, y + 1, center),
            to_screen(x, y + 1, center),
            _color(HOVER_COLOR),
            2.0,
        )


def main() -> None:
    world = poland.generate()
    map_view = MapView(world)
    immapp.run(
        gui_function=map_view.gui,
        window_title="Railroads",
        window_size=(1280, 800),
        with_implot=True,
    )
