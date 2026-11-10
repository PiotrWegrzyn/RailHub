"""Two small maps in the corner of the view, each with its own toggle: the world, with the game's
map marked on it, and the
whole game map seen from above, with the part on screen outlined. Clicking or dragging on the
second one moves the view there, and the wheel over it zooms it in, through OVERVIEW_ZOOMS, to the
part of the map around the view."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.mapgen.world import WorldLand
from railroads.sim.geo import Bounds
from railroads.sim.grids import Cell
from railroads.sim.tilemap import TileMap
from railroads.ui.iso_camera import IsoCamera, Vec2

# Latitudes the world map shows, north to south; the poles are left out.
WORLD_NORTH = 84.0
WORLD_SOUTH = -60.0
WORLD_WIDTH = 300.0
OVERVIEW_SIDE = 180.0
MARGIN = 12.0
GAP = 8.0
# Smallest size of the map's box on the world map, in pixels, so a small map still shows.
MIN_BOX = 6.0
# How many times as close the overview can show the map.
OVERVIEW_ZOOMS = (1, 2, 4, 8)

SEA_RGB = (32, 58, 92)
LAND_RGB = (138, 150, 104)
FRAME_COLOR = (0.06, 0.07, 0.09, 0.85)
BORDER_COLOR = (1.0, 1.0, 1.0, 0.35)
MARK_COLOR = (1.0, 0.82, 0.2, 1.0)
CROSSHAIR_COLOR = (1.0, 0.82, 0.2, 0.35)
VIEW_COLOR = (1.0, 1.0, 1.0, 0.95)
TOWN_COLOR = (0.15, 0.13, 0.12, 0.9)


@dataclass(frozen=True, slots=True)
class Panel:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom


@dataclass(frozen=True, slots=True)
class MinimapLayout:
    """Where the maps that are shown sit on screen, side by side in the bottom-right corner of
    the view: the overview of the game map in the corner, the world to its left."""

    world: Panel | None
    overview: Panel | None
    map_size: Vec2
    # The part of the map the overview shows: its left, top, right and bottom in map units.
    window: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def of(
        cls,
        map_size: Vec2,
        bottom_right: Vec2,
        overview: bool = True,
        world: bool = True,
        zoom: int = 1,
        focus: Vec2 | None = None,
    ) -> MinimapLayout:
        """`zoom` times as close, the overview shows the part of the map around `focus`, kept
        on the map."""
        width, height = map_size
        scale = OVERVIEW_SIDE / max(width, height)
        right = bottom_right[0] - MARGIN
        overview_panel = None
        if overview:
            overview_panel = Panel(
                right - width * scale,
                bottom_right[1] - MARGIN - height * scale,
                width * scale,
                height * scale,
            )
            right = overview_panel.left - GAP
        world_panel = None
        if world:
            world_height = WORLD_WIDTH * (WORLD_NORTH - WORLD_SOUTH) / 360.0
            world_panel = Panel(
                right - WORLD_WIDTH,
                bottom_right[1] - MARGIN - world_height,
                WORLD_WIDTH,
                world_height,
            )
        return cls(world_panel, overview_panel, map_size, _window(map_size, zoom, focus))

    def contains(self, x: float, y: float) -> bool:
        return any(panel.contains(x, y) for panel in (self.world, self.overview) if panel)

    def world_point(self, lon: float, lat: float) -> Vec2:
        panel = self.world
        assert panel is not None
        return (
            panel.left + (lon + 180.0) / 360.0 * panel.width,
            panel.top + (WORLD_NORTH - lat) / (WORLD_NORTH - WORLD_SOUTH) * panel.height,
        )

    def map_box(self, bounds: Bounds) -> tuple[float, float, float, float]:
        """Left, top, right and bottom of the map's box on the world, made big enough to see."""
        left, top = self.world_point(bounds.west, bounds.north)
        right, bottom = self.world_point(bounds.east, bounds.south)
        grow_x = max(0.0, MIN_BOX - (right - left)) / 2
        grow_y = max(0.0, MIN_BOX - (bottom - top)) / 2
        return left - grow_x, top - grow_y, right + grow_x, bottom + grow_y

    def overview_point(self, x: float, y: float) -> Vec2:
        panel, (x0, y0, x1, y1) = self.overview, self.window
        assert panel is not None
        return (
            panel.left + (x - x0) / (x1 - x0) * panel.width,
            panel.top + (y - y0) / (y1 - y0) * panel.height,
        )

    def overview_to_map(self, screen_x: float, screen_y: float) -> Vec2:
        """The map point under a spot on the overview, kept on the part it shows."""
        panel, (x0, y0, x1, y1) = self.overview, self.window
        assert panel is not None
        u = min(max((screen_x - panel.left) / panel.width, 0.0), 1.0)
        v = min(max((screen_y - panel.top) / panel.height, 0.0), 1.0)
        return x0 + u * (x1 - x0), y0 + v * (y1 - y0)


def _window(map_size: Vec2, zoom: int, focus: Vec2 | None) -> tuple[float, float, float, float]:
    width, height = map_size
    span_x, span_y = width / zoom, height / zoom
    cx, cy = (width / 2, height / 2) if focus is None else focus
    x0 = min(max(cx - span_x / 2, 0.0), width - span_x)
    y0 = min(max(cy - span_y / 2, 0.0), height - span_y)
    return x0, y0, x0 + span_x, y0 + span_y


def world_rgba(world: WorldLand) -> np.ndarray:
    share = world.land[..., None].astype(np.float64) / 255
    rgb = np.array(SEA_RGB) * (1 - share) + np.array(LAND_RGB) * share
    alpha = np.full((*world.land.shape, 1), 255.0)
    return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=-1).astype(np.uint8))


def _color(rgba: tuple[float, float, float, float]) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))


class Minimap:
    def __init__(self, tile_map: TileMap, world: WorldLand, towns: Iterable[Cell] = ()) -> None:
        self.tile_map = tile_map
        self.world = world
        # The overview of the game map, and the world map.
        self.shown = True
        self.world_shown = False
        # How many times as close the overview shows the map, and the map point it is centred
        # on, which follows the view except while the overview is being dragged.
        self.zoom = OVERVIEW_ZOOMS[0]
        self.focus: Vec2 | None = None
        self._world_texture: hello_imgui.TextureGpu | None = None
        self.show_towns(towns)

    def zoom_by(self, steps: int) -> None:
        """Zooms the overview in (above 0) or out by so many of OVERVIEW_ZOOMS."""
        at = OVERVIEW_ZOOMS.index(self.zoom) + steps
        self.zoom = OVERVIEW_ZOOMS[min(max(at, 0), len(OVERVIEW_ZOOMS) - 1)]

    def show_towns(self, towns: Iterable[Cell]) -> None:
        self._towns = np.array(sorted(set(towns)), dtype=int).reshape(-1, 2)

    def world_texture(self) -> imgui.ImTextureRef:
        if self._world_texture is None:
            rgba = world_rgba(self.world)
            self._world_texture = hello_imgui.create_texture_gpu_from_rgba_data(rgba)
        return imgui.ImTextureRef(self._world_texture.texture_id())

    def draw(
        self,
        draw_list: imgui.ImDrawList,
        layout: MinimapLayout,
        textures: tuple[imgui.ImTextureRef, imgui.ImTextureRef],
        view_corners: list[Vec2],
    ) -> None:
        """Draws the maps the layout holds, given the world picture and the ground picture of the
        game map."""
        world, ground = textures
        if layout.world is not None:
            self._draw_world(draw_list, layout, world)
        if layout.overview is not None:
            self._draw_overview(draw_list, layout, ground, view_corners)

    def _draw_world(
        self, draw_list: imgui.ImDrawList, layout: MinimapLayout, texture: imgui.ImTextureRef
    ) -> None:
        panel = layout.world
        assert panel is not None
        _frame(draw_list, panel)
        draw_list.add_image(
            texture,
            (panel.left, panel.top),
            (panel.right, panel.bottom),
            (0.0, (90.0 - WORLD_NORTH) / 180.0),
            (1.0, (90.0 - WORLD_SOUTH) / 180.0),
        )
        left, top, right, bottom = layout.map_box(self.tile_map.bounds)
        mid_x, mid_y = (left + right) / 2, (top + bottom) / 2
        draw_list.push_clip_rect((panel.left, panel.top), (panel.right, panel.bottom), True)
        crosshair = _color(CROSSHAIR_COLOR)
        draw_list.add_line((panel.left, mid_y), (left, mid_y), crosshair)
        draw_list.add_line((right, mid_y), (panel.right, mid_y), crosshair)
        draw_list.add_line((mid_x, panel.top), (mid_x, top), crosshair)
        draw_list.add_line((mid_x, bottom), (mid_x, panel.bottom), crosshair)
        draw_list.add_rect((left, top), (right, bottom), _color(MARK_COLOR), 0.0, 2.0)
        draw_list.pop_clip_rect()

    def _draw_overview(
        self,
        draw_list: imgui.ImDrawList,
        layout: MinimapLayout,
        ground: imgui.ImTextureRef,
        view_corners: list[Vec2],
    ) -> None:
        panel = layout.overview
        assert panel is not None
        _frame(draw_list, panel)
        (x0, y0, x1, y1), (width, height) = layout.window, layout.map_size
        draw_list.add_image(
            ground,
            (panel.left, panel.top),
            (panel.right, panel.bottom),
            (x0 / width, y0 / height),
            (x1 / width, y1 / height),
        )
        draw_list.push_clip_rect((panel.left, panel.top), (panel.right, panel.bottom), True)
        town = _color(TOWN_COLOR)
        grid = self.tile_map.grid
        if len(self._towns):
            xs, ys = grid.center(self._towns[:, 0], self._towns[:, 1])
            for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
                sx, sy = layout.overview_point(x, y)
                draw_list.add_rect_filled((sx - 1, sy - 1), (sx + 1, sy + 1), town)
        outline = [imgui.ImVec2(*layout.overview_point(x, y)) for x, y in view_corners]
        draw_list.add_polyline(outline, _color(VIEW_COLOR), 1.5, imgui.ImDrawFlags_.closed.value)
        if self.zoom > 1:
            label = f"x{self.zoom}"
            size = imgui.calc_text_size(label)
            spot = (panel.right - size.x - 4, panel.top + 2)
            draw_list.add_text(spot, _color(VIEW_COLOR), label)
        draw_list.pop_clip_rect()


def view_corners(camera: IsoCamera, center: Vec2, top_left: Vec2, bottom_right: Vec2) -> list[Vec2]:
    """The map points at the corners of the screen, in order around it."""
    (left, top), (right, bottom) = top_left, bottom_right
    return [
        camera.to_plane(x, y, center)
        for x, y in ((left, top), (right, top), (right, bottom), (left, bottom))
    ]


def _frame(draw_list: imgui.ImDrawList, panel: Panel) -> None:
    pad = 3.0
    draw_list.add_rect_filled(
        (panel.left - pad, panel.top - pad),
        (panel.right + pad, panel.bottom + pad),
        _color(FRAME_COLOR),
        4.0,
    )
    draw_list.add_rect(
        (panel.left - pad, panel.top - pad),
        (panel.right + pad, panel.bottom + pad),
        _color(BORDER_COLOR),
        4.0,
    )
