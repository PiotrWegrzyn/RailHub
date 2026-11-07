"""Draws one TileMap. Zoomed out, the ground is one smooth texture with the cell grid and
buildings as dots on top; close up, it is pixel art (see railroads.ui.pixel). The hovered cell, a
compass and the minimap go over both."""

import math
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads import maps
from railroads.mapgen.world import WorldLand
from railroads.sim.constructions import (
    Building,
    Construction,
    Constructions,
    Station,
    Track,
)
from railroads.sim.economy import Satisfaction
from railroads.sim.geo import describe_lonlat
from railroads.sim.grids import HEX_RADIUS, HEX_ROW_STEP, Cell, GridKind
from railroads.sim.growth import Growth
from railroads.sim.tilemap import Terrain, Tile, TileMap
from railroads.sim.track_rules import TrackBed, track_rules_for
from railroads.ui.biomes import biome_of
from railroads.ui.buildings import BuildingSpots, draw_building_dots
from railroads.ui.ground import GroundLayout, cell_colors
from railroads.ui.hex_details import describe, draw_hex_details
from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS, IsoCamera, Vec2
from railroads.ui.minimap import Minimap, MinimapLayout, view_corners
from railroads.ui.pixel.layer import PixelLayer
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.world import GLOW, PixelWorld, glow_strength

BACKGROUND_COLOR = (0.08, 0.09, 0.11, 1.0)
GRID_COLOR = (0.0, 0.0, 0.0, 0.25)
HOVER_COLOR = (1.0, 1.0, 1.0, 0.9)
SELECTED_COLOR = (1.0, 0.85, 0.2, 1.0)
# A press that moves the mouse less than this many pixels is a click, not a drag.
CLICK_SLOP = 4.0
DETAILS_POS = (12.0, 90.0)
DETAILS_SIZE = (560.0, 320.0)
COMPASS_COLOR = (1.0, 1.0, 1.0, 0.85)
# Slopes are lit from the top-left of the screen.
LIGHT_ON_SCREEN = (-1.0, -1.0)
# Population points a cell needs to show as a town on the minimap.
MINIMAP_TOWN_POINTS = 3
# Lower left, upper left, top and upper right corner: three edges per hexagon draw the whole grid.
_HEX_EDGE_ANGLES = np.radians([150, 210, 270, 330])


@dataclass(frozen=True)
class GridLook:
    """How a grid sits on screen. Squares are diamonds with north at the top-left, like RT2;
    hexagons have north up."""

    angle_deg: float
    cell_units: float
    # Cell widths in pixels from which the grid shows: first baked into the ground picture, while
    # there are too many cells to outline one by one, then as sharp lines drawn on top.
    baked_lines_from: int | None
    drawn_lines_from: int


LOOKS: dict[GridKind, GridLook] = {
    "square": GridLook(-45.0, math.sqrt(2), baked_lines_from=None, drawn_lines_from=16),
    "hex": GridLook(0.0, 1.0, baked_lines_from=8, drawn_lines_from=24),
}


def _color(rgba: tuple[float, float, float, float]) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))


TRACK_ON = {
    TrackBed.GROUND: "track",
    TrackBed.BRIDGE: "track on a bridge",
    TrackBed.TUNNEL: "track in a tunnel",
}


def _track_note(tile: Tile) -> str:
    for bed, track in TRACK_ON.items():
        rules = track_rules_for(tile, bed)
        if rules.can_build:
            return f"{track} costs x{rules.cost_factor:g}, trains at {rules.speed_factor:.0%} speed"
    return "no track"


def _construction_name(construction: Construction) -> str:
    match construction:
        case Building():
            return construction.kind.value
        case Station():
            return "station"
        case Track():
            bed = "" if construction.bed == TrackBed.GROUND else f" {construction.bed.name.lower()}"
            return f"{construction.width.name.lower()} track{bed}"


@cache
def _sprite_sheet() -> SpriteSheet:
    return SpriteSheet.load()


def _key_pressed(*keys: imgui.Key) -> bool:
    return any(imgui.is_key_pressed(key, repeat=False) for key in keys)


class MapView:
    def __init__(
        self,
        tile_map: TileMap,
        constructions: Constructions | None = None,
        world: WorldLand | None = None,
    ) -> None:
        self.tile_map = tile_map
        self.constructions = Constructions(tile_map) if constructions is None else constructions
        self.growth = Growth(self.constructions)
        self._satisfaction = self.growth.satisfaction()
        self.minimap = Minimap(
            tile_map, maps.world_land() if world is None else world, self._town_cells()
        )
        self._spots = BuildingSpots.of(self.constructions)
        self._look = LOOKS[tile_map.grid.kind]
        width, height = tile_map.grid.size()
        self.camera = IsoCamera(
            center_x=width / 2,
            center_y=height / 2,
            angle_deg=self._look.angle_deg,
            cell_units=self._look.cell_units,
        )
        self._layout: GroundLayout | None = None
        self._pixels: PixelLayer | None = None
        self._textures: dict[bool, hello_imgui.TextureGpu] = {}
        self._hovered: Cell | None = None
        # The cell whose details are open, picked by clicking without dragging.
        self.selected: Cell | None = None
        self._dragged = 0.0
        self._over_minimap = False
        # Which minimap the current drag started on, if any.
        self._pressed_on: Literal["world", "overview"] | None = None
        self._wheel = 0.0

    def gui(self) -> None:
        self._draw_toolbar()
        self._draw_map()
        self._draw_details()

    def _draw_details(self) -> None:
        if self.selected is None:
            return
        imgui.set_next_window_pos(DETAILS_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(DETAILS_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, still_open = imgui.begin("Hex details", True)
        if expanded:
            draw_hex_details(describe(self.growth, self.selected))
        imgui.end()
        if not still_open:
            self.selected = None

    def month_passed(self, rng: np.random.Generator) -> None:
        """Grows the towns by a month and shows what changed."""
        changed = self.growth.month(rng)
        satisfaction = self.growth.satisfaction()
        moods_changed = not np.array_equal(satisfaction, self._satisfaction)
        self._satisfaction = satisfaction
        if moods_changed:
            self._textures.clear()
            if self._pixels is not None:
                self._pixels.refresh(self._pixels.world.satisfaction_changed(satisfaction))
        if not changed:
            return
        self._spots = BuildingSpots.of(self.constructions)
        self.minimap.show_towns(self._town_cells())
        if self._pixels is not None:
            self._pixels.refresh(self._pixels.world.buildings_changed(changed))

    def _town_cells(self) -> list[Cell]:
        """Cells big enough to show as towns on the minimap."""
        town = self.growth.points() >= MINIMAP_TOWN_POINTS
        return [(int(row), int(col)) for row, col in np.argwhere(town)]

    def _draw_toolbar(self) -> None:
        if imgui.button(" - ") or _key_pressed(imgui.Key.minus, imgui.Key.keypad_subtract):
            self.camera.zoom(-1, (0, 0), (0, 0))
        imgui.same_line()
        if imgui.button(" + ") or _key_pressed(imgui.Key.equal, imgui.Key.keypad_add):
            self.camera.zoom(1, (0, 0), (0, 0))
        imgui.same_line()
        _, self.minimap.shown = imgui.checkbox("Minimap (M)", self.minimap.shown)
        if _key_pressed(imgui.Key.m):
            self.minimap.shown = not self.minimap.shown
        imgui.same_line()
        _, self.minimap.world_shown = imgui.checkbox("World (W)", self.minimap.world_shown)
        if _key_pressed(imgui.Key.w):
            self.minimap.world_shown = not self.minimap.world_shown
        imgui.same_line()
        imgui.text(self._status())

    def _status(self) -> str:
        zoom = f"Zoom {self.camera.zoom_level + 1}/{len(ZOOM_CELL_WIDTHS)}"
        if self._over_minimap:
            return f"{zoom}. Click or drag on the map overview to look there."
        if self._hovered is None:
            return f"{zoom}. Drag to pan, scroll or -/+ to zoom."
        tile = self.tile_map.tile(self._hovered)
        lon, lat = self.tile_map.lonlat(self._hovered)
        row, col = self._hovered
        properties = (
            f"{tile.terrain.name.lower()}, {tile.vegetation.name.lower()}, "
            f"{tile.temperature.name.lower()}, {tile.height_m} m"
        )
        place = f"Cell {row}, {col} at {describe_lonlat(lon, lat)}"
        land = f"{tile.relief.name.lower()}, {_track_note(tile)}"
        built = self._built_note(self._hovered)
        level = int(self._satisfaction[row, col])
        mood = f"; {Satisfaction(level).name.lower()}" if level >= 0 else ""
        return f"{zoom}. {place}: {biome_of(tile).name.title()} ({properties}); {land}{built}{mood}"

    def _built_note(self, cell: Cell) -> str:
        built = self.constructions.at(cell)
        if not built:
            return ""
        counts = Counter(_construction_name(construction) for construction in built)
        listed = ", ".join(name if n == 1 else f"{name} x{n}" for name, n in counts.items())
        left = self.constructions.space_left(cell)
        return f"; {listed}, " + (f"room for {left} more" if left else "full")

    def _ground_layout(self) -> GroundLayout:
        if self._layout is None:
            self._layout = GroundLayout.of(self.tile_map.grid)
        return self._layout

    def _ground_texture(self, with_lines: bool) -> hello_imgui.TextureGpu:
        """The smooth ground picture used when zoomed out, with or without grid lines."""
        if with_lines not in self._textures:
            light = self.camera.map_direction(*LIGHT_ON_SCREEN)
            colors = cell_colors(self.tile_map)
            strength = glow_strength(self._satisfaction)[..., None]
            land = (self.tile_map.terrain == Terrain.LAND)[..., None]
            tinted = colors[..., :3] * (1 - strength) + np.array(GLOW) * strength
            colors[..., :3] = np.where(land, tinted, colors[..., :3]).astype(np.uint8)
            rgba = self._ground_layout().paint(colors, self.tile_map.relief, light, with_lines)
            self._textures[with_lines] = hello_imgui.create_texture_gpu_from_rgba_data(rgba)
        return self._textures[with_lines]

    def _draw_map(self) -> None:
        origin = imgui.get_cursor_screen_pos()
        size = imgui.get_content_region_avail()
        if size.x <= 0 or size.y <= 0:
            return
        imgui.invisible_button("map", size)
        center = (origin.x + size.x / 2, origin.y + size.y / 2)
        top_left = (origin.x, origin.y)
        bottom_right = (origin.x + size.x, origin.y + size.y)
        map_size = self._ground_layout().size
        shown, world_shown = self.minimap.shown, self.minimap.world_shown
        minimap = (
            MinimapLayout.of(map_size, bottom_right, shown, world_shown)
            if shown or world_shown
            else None
        )
        self._handle_input(center, minimap)

        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(top_left, bottom_right, True)
        draw_list.add_rect_filled(top_left, bottom_right, _color(BACKGROUND_COLOR))
        if self.camera.pixel_scale:
            self._draw_ground(draw_list, center, self._ground_texture(False))
            self._pixel_layer().draw(draw_list, self.camera, center, (top_left, bottom_right))
        else:
            self._draw_smooth_map(draw_list, center, top_left, bottom_right)
        if self._hovered is not None:
            self._draw_outline(draw_list, center, self._hovered)
        if self.selected is not None:
            self._draw_outline(draw_list, center, self.selected, SELECTED_COLOR)
        self._draw_compass(draw_list, (bottom_right[0] - 36, top_left[1] + 36))
        if minimap is not None:
            corners = view_corners(self.camera, center, top_left, bottom_right)
            ground = imgui.ImTextureRef(self._ground_texture(False).texture_id())
            textures = (self.minimap.world_texture(), ground)
            self.minimap.draw(draw_list, minimap, textures, corners)
        draw_list.pop_clip_rect()

    def _draw_smooth_map(
        self, draw_list: imgui.ImDrawList, center: Vec2, top_left: Vec2, bottom_right: Vec2
    ) -> None:
        cell_width = self.camera.cell_width
        drawn_lines = cell_width >= self._look.drawn_lines_from
        baked_from = self._look.baked_lines_from
        baked_lines = not drawn_lines and baked_from is not None and cell_width >= baked_from
        self._draw_ground(draw_list, center, self._ground_texture(baked_lines))
        if drawn_lines:
            self._draw_grid(draw_list, center, top_left, bottom_right)
        draw_building_dots(draw_list, self.camera, center, (top_left, bottom_right), self._spots)

    def _pixel_layer(self) -> PixelLayer:
        if self._pixels is None:
            world = PixelWorld(
                self.tile_map,
                self.constructions,
                _sprite_sheet(),
                self._look.angle_deg,
                self._look.cell_units,
            )
            world.satisfaction = self._satisfaction.copy()
            self._pixels = PixelLayer(world)
        return self._pixels

    def _handle_input(self, center: Vec2, minimap: MinimapLayout | None) -> None:
        io = imgui.get_io()
        self._over_minimap = minimap is not None and minimap.contains(
            io.mouse_pos.x, io.mouse_pos.y
        )
        if imgui.is_item_activated():
            self._dragged = 0.0
            self._pressed_on = (
                None
                if minimap is None or not self._over_minimap
                else "overview"
                if minimap.overview is not None
                and minimap.overview.contains(io.mouse_pos.x, io.mouse_pos.y)
                else "world"
            )
        if imgui.is_item_active():
            self._dragged += abs(io.mouse_delta.x) + abs(io.mouse_delta.y)
            if self._pressed_on is None:
                self.camera.pan(io.mouse_delta.x, io.mouse_delta.y)
            elif self._pressed_on == "overview" and minimap is not None:
                x, y = minimap.overview_to_map(io.mouse_pos.x, io.mouse_pos.y)
                self.camera.center_x, self.camera.center_y = x, y
        hovered = imgui.is_item_hovered() and not self._over_minimap
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
        clicked = imgui.is_item_deactivated() and self._dragged < CLICK_SLOP
        if clicked and self._pressed_on is None and self._hovered is not None:
            self.selected = None if self.selected == self._hovered else self._hovered

    def _draw_ground(
        self, draw_list: imgui.ImDrawList, center: Vec2, texture: hello_imgui.TextureGpu
    ) -> None:
        width, height = self._ground_layout().size
        to_screen = self.camera.to_screen
        draw_list.add_image_quad(
            imgui.ImTextureRef(texture.texture_id()),
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

    def _draw_outline(
        self,
        draw_list: imgui.ImDrawList,
        center: Vec2,
        cell: Cell,
        color: tuple[float, float, float, float] = HOVER_COLOR,
    ) -> None:
        points = [
            imgui.ImVec2(*self.camera.to_screen(x, y, center))
            for x, y in self.tile_map.grid.corners(*cell)
        ]
        draw_list.add_polyline(points, _color(color), 2.0, imgui.ImDrawFlags_.closed.value)

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
