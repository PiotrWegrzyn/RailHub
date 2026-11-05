"""Draws one TileMap: the ground as a single texture, with the cell grid, the hovered cell and a
compass on top."""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from imgui_bundle import hello_imgui, imgui

from railroads.sim.grids import HEX_RADIUS, HEX_ROW_STEP, Cell, GridKind
from railroads.sim.tilemap import TileMap
from railroads.ui.biomes import biome_colors, biome_of
from railroads.ui.iso_camera import IsoCamera, Vec2

BACKGROUND_COLOR = (0.08, 0.09, 0.11, 1.0)
GRID_COLOR = (0.0, 0.0, 0.0, 0.25)
HOVER_COLOR = (1.0, 1.0, 1.0, 0.9)
COMPASS_COLOR = (1.0, 1.0, 1.0, 0.85)
# The renderer samples textures with linear filtering; drawing each cell as a block of many
# identical pixels keeps cell edges sharp when zoomed in.
PIXELS_PER_CELL = 8
TEXTURE_ROWS_PER_PASS = 256
# Lower left, upper left, top and upper right corner: three edges per hexagon draw the whole grid.
_HEX_EDGE_ANGLES = np.radians([150, 210, 270, 330])


@dataclass(frozen=True)
class GridLook:
    """How a grid sits on screen. Squares start as diamonds with north at the top-left, like RT2;
    hexagons start with north up."""

    base_angle: float
    cell_units: float
    start_rotation: int
    min_cell_width_for_lines: int


LOOKS: dict[GridKind, GridLook] = {
    "square": GridLook(45.0, math.sqrt(2), start_rotation=3, min_cell_width_for_lines=16),
    "hex": GridLook(0.0, 1.0, start_rotation=0, min_cell_width_for_lines=32),
}


def ground_rgba(tile_map: TileMap) -> npt.NDArray[np.uint8]:
    grid = tile_map.grid
    colors = biome_colors(tile_map)
    width, height = grid.size()
    width_px = math.ceil(width * PIXELS_PER_CELL)
    height_px = math.ceil(height * PIXELS_PER_CELL)
    rgba = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    xs = (np.arange(width_px) + 0.5) / PIXELS_PER_CELL
    for top in range(0, height_px, TEXTURE_ROWS_PER_PASS):
        ys = (np.arange(top, min(top + TEXTURE_ROWS_PER_PASS, height_px)) + 0.5) / PIXELS_PER_CELL
        rows, cols = grid.cell_at(*np.meshgrid(xs, ys))
        on_map = grid.contains(rows, cols)
        rgba[top : top + len(ys)][on_map] = colors[rows[on_map], cols[on_map]]
    return rgba


def _color(rgba: tuple[float, float, float, float]) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))


def _key_pressed(*keys: imgui.Key) -> bool:
    return any(imgui.is_key_pressed(key, repeat=False) for key in keys)


class MapView:
    def __init__(self, tile_map: TileMap) -> None:
        self.tile_map = tile_map
        self._look = LOOKS[tile_map.grid.kind]
        width, height = tile_map.grid.size()
        self.camera = IsoCamera(
            center_x=width / 2,
            center_y=height / 2,
            turn_steps=tile_map.grid.symmetry,
            base_angle=self._look.base_angle,
            cell_units=self._look.cell_units,
            rotation=self._look.start_rotation,
        )
        self._texture: hello_imgui.TextureGpu | None = None
        self._texture_size = (0.0, 0.0)
        self._hovered: Cell | None = None
        self._wheel = 0.0

    def gui(self) -> None:
        self._draw_toolbar()
        self._draw_map()

    def _draw_toolbar(self) -> None:
        if imgui.button("Turn left (Q)") or _key_pressed(imgui.Key.q):
            self.camera.turn(-1)
        imgui.same_line()
        if imgui.button("Turn right (E)") or _key_pressed(imgui.Key.e):
            self.camera.turn(1)
        imgui.same_line()
        if imgui.button(" - ") or _key_pressed(imgui.Key.minus, imgui.Key.keypad_subtract):
            self.camera.zoom(-1, (0, 0), (0, 0))
        imgui.same_line()
        if imgui.button(" + ") or _key_pressed(imgui.Key.equal, imgui.Key.keypad_add):
            self.camera.zoom(1, (0, 0), (0, 0))
        imgui.same_line()
        imgui.text(self._status())

    def _status(self) -> str:
        zoom = f"Zoom {self.camera.zoom_level + 1}/5"
        if self._hovered is None:
            return f"{zoom}. Drag to pan, scroll or -/+ to zoom, Q/E to turn."
        tile = self.tile_map.tile(self._hovered)
        lon, lat = self.tile_map.lonlat(self._hovered)
        row, col = self._hovered
        properties = (
            f"{tile.terrain.name.lower()}, {tile.vegetation.name.lower()}, "
            f"{tile.temperature.name.lower()}, {tile.height_m} m"
        )
        place = f"Cell {row}, {col} at {lat:.2f}°N {lon:.2f}°E"
        return f"{zoom}. {place}: {biome_of(tile).name.title()} ({properties})"

    def _draw_map(self) -> None:
        if self._texture is None:
            rgba = ground_rgba(self.tile_map)
            self._texture = hello_imgui.create_texture_gpu_from_rgba_data(rgba)
            self._texture_size = (rgba.shape[1] / PIXELS_PER_CELL, rgba.shape[0] / PIXELS_PER_CELL)
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
        self._draw_ground(draw_list, center)
        if self.camera.cell_width >= self._look.min_cell_width_for_lines:
            self._draw_grid(draw_list, center, top_left, bottom_right)
        if self._hovered is not None:
            self._draw_outline(draw_list, center, self._hovered)
        self._draw_compass(draw_list, (bottom_right[0] - 36, top_left[1] + 36))
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
        self._hovered = None
        if hovered:
            x, y = self.camera.to_plane(io.mouse_pos.x, io.mouse_pos.y, center)
            row, col = self.tile_map.grid.cell_at(x, y)
            cell = int(row), int(col)
            if self.tile_map.grid.contains(*cell):
                self._hovered = cell

    def _draw_ground(self, draw_list: imgui.ImDrawList, center: Vec2) -> None:
        assert self._texture is not None
        width, height = self._texture_size
        to_screen = self.camera.to_screen
        draw_list.add_image_quad(
            imgui.ImTextureRef(self._texture.texture_id()),
            to_screen(0, 0, center),
            to_screen(width, 0, center),
            to_screen(width, height, center),
            to_screen(0, height, center),
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
        )

    def _visible_area(self, center: Vec2, top_left: Vec2, bottom_right: Vec2) -> tuple[float, ...]:
        corners = [
            self.camera.to_plane(x, y, center)
            for x in (top_left[0], bottom_right[0])
            for y in (top_left[1], bottom_right[1])
        ]
        xs, ys = [corner[0] for corner in corners], [corner[1] for corner in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def _draw_grid(
        self, draw_list: imgui.ImDrawList, center: Vec2, top_left: Vec2, bottom_right: Vec2
    ) -> None:
        x0, y0, x1, y1 = self._visible_area(center, top_left, bottom_right)
        if self.tile_map.grid.kind == "square":
            self._draw_square_lines(draw_list, center, (x0, y0, x1, y1))
        else:
            self._draw_hex_lines(draw_list, center, (x0, y0, x1, y1))

    def _draw_square_lines(
        self, draw_list: imgui.ImDrawList, center: Vec2, visible: tuple[float, ...]
    ) -> None:
        grid = self.tile_map.grid
        x0, y0, x1, y1 = visible
        c0, c1 = max(0, math.floor(x0)), min(grid.cols, math.ceil(x1))
        r0, r1 = max(0, math.floor(y0)), min(grid.rows, math.ceil(y1))
        if c0 >= c1 or r0 >= r1:
            return
        color = _color(GRID_COLOR)
        to_screen = self.camera.to_screen
        for x in range(c0, c1 + 1):
            draw_list.add_line(to_screen(x, r0, center), to_screen(x, r1, center), color)
        for y in range(r0, r1 + 1):
            draw_list.add_line(to_screen(c0, y, center), to_screen(c1, y, center), color)

    def _draw_hex_lines(
        self, draw_list: imgui.ImDrawList, center: Vec2, visible: tuple[float, ...]
    ) -> None:
        grid = self.tile_map.grid
        x0, y0, x1, y1 = visible
        r0, r1 = (
            max(0, math.floor(y0 / HEX_ROW_STEP) - 1),
            min(grid.rows, math.ceil(y1 / HEX_ROW_STEP) + 1),
        )
        c0, c1 = max(0, math.floor(x0) - 1), min(grid.cols, math.ceil(x1) + 1)
        if c0 >= c1 or r0 >= r1:
            return
        rows, cols = np.mgrid[r0:r1, c0:c1]
        cx, cy = grid.center(rows.ravel(), cols.ravel())
        px = cx[:, None] + HEX_RADIUS * np.cos(_HEX_EDGE_ANGLES)
        py = cy[:, None] + HEX_RADIUS * np.sin(_HEX_EDGE_ANGLES)
        sx, sy = self.camera.to_screen(px, py, center)
        color = _color(GRID_COLOR)
        for points in np.stack([sx, sy], axis=2).tolist():
            draw_list.add_polyline([imgui.ImVec2(x, y) for x, y in points], color, 1.0, 0)

    def _draw_outline(self, draw_list: imgui.ImDrawList, center: Vec2, cell: Cell) -> None:
        points = [
            imgui.ImVec2(*self.camera.to_screen(x, y, center))
            for x, y in self.tile_map.grid.corners(*cell)
        ]
        draw_list.add_polyline(points, _color(HOVER_COLOR), 2.0, imgui.ImDrawFlags_.closed.value)

    def _draw_compass(self, draw_list: imgui.ImDrawList, anchor: Vec2) -> None:
        here = self.camera.to_screen(self.camera.center_x, self.camera.center_y, (0, 0))
        north = self.camera.to_screen(self.camera.center_x, self.camera.center_y - 1, (0, 0))
        dx, dy = north[0] - here[0], north[1] - here[1]
        length = math.hypot(dx, dy)
        dx, dy = dx / length, dy / length
        color = _color(COMPASS_COLOR)
        tip = (anchor[0] + dx * 14, anchor[1] + dy * 14)
        draw_list.add_line((anchor[0] - dx * 14, anchor[1] - dy * 14), tip, color, 2.0)
        draw_list.add_triangle_filled(
            tip,
            (tip[0] - dx * 7 - dy * 5, tip[1] - dy * 7 + dx * 5),
            (tip[0] - dx * 7 + dy * 5, tip[1] - dy * 7 - dx * 5),
            color,
        )
        label = imgui.calc_text_size("N")
        position = (anchor[0] + dx * 25 - label.x / 2, anchor[1] + dy * 25 - label.y / 2)
        draw_list.add_text(position, color, "N")
