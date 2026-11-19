"""The land around the map the player can buy (railroads.sim.lots, railroads.land). Every lot not
owned is drawn where it lies, past the map's edge or in a gap of it, with the ground of the
place's map there (railroads.newgame.place_map): its rivers, lakes, forests, hills and mountains,
without the buildings, so what lies there is no surprise; or as the world map shows it where the
place's map isn't known. It lies behind a chain with a padlock, and with its price if it is on
offer. A click on one asks to confirm buying it."""

import math
from dataclasses import dataclass, field

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.sim.finance import Finances, dollars
from railroads.sim.grids import HEX_RADIUS, HEX_ROW_STEP
from railroads.sim.lots import Lot, Lots
from railroads.sim.tilemap import TileMap
from railroads.ui.ground import cell_colors
from railroads.ui.iso_camera import IsoCamera, Vec2
from railroads.ui.theme import BRASS, BRASS_DARK, GONE, OVERLAY, rgba_u32

CONFIRM_POPUP = "Buy land?"
# How the world map is shown under a lot, and the shade over it, lighter on the one hovered.
LAND_TINT = (0.82, 0.82, 0.82, 1.0)
SHADE = (0.05, 0.04, 0.03, 0.42)
HOVERED_SHADE = (0.05, 0.04, 0.03, 0.18)
STEEL = (0.7, 0.72, 0.74, 1.0)
STEEL_DARK = (0.16, 0.16, 0.17, 0.9)
# A chain link's length, in screen points, and the padlock's size, from a share of the lot's
# shorter side on screen but kept between the two.
LINK = 11.0
LOCK_SHARE = 0.16
LOCK_SIZE = (18.0, 54.0)
# How far past a lot's outline its padlock and price tag may reach on screen, in points: a lot
# drawn smaller than its tag still shows it whole.
TAG_REACH = 160.0


def lot_corners(lots: Lots, lot: Lot) -> list[Vec2]:
    """A lot's six corners on the game's map, in cell units, clockwise."""
    return [lots.map_point(corner) for corner in lots.corners(lot)]


def lot_under(lots: Lots, x: float, y: float) -> Lot:
    """The lot a point on the game's map in cell units lies in, past the map's edges too."""
    return lots.lot_at_point(*lots.place_point((x, y)))


def world_uv(tile_map: TileMap, point: Vec2) -> Vec2:
    """Where a point on the map, in cell units, lies on the world map's picture
    (railroads.ui.minimap.world_rgba): from 180°W to 180°E across, the poles down."""
    km = tile_map.grid.km_per_cell
    lon, lat = tile_map.bounds.to_lonlat(point[0] * km, point[1] * km)
    return (float(lon) + 180.0) / 360.0, (90.0 - float(lat)) / 180.0


@dataclass(frozen=True)
class PlacePicture:
    """The ground of the place's map, a pixel a cell, with two lots' width of sea round it, and
    the place's cell its pixel (0, 0) shows."""

    pixels: np.ndarray
    top: int
    left: int

    @classmethod
    def of(cls, place: TileMap, lots: Lots) -> PlacePicture:
        margin = math.ceil(4 * lots.radius) // 2 * 2 + 2
        top, left = -margin, -margin
        rows, cols = place.grid.rows + 2 * margin, place.grid.cols + 2 * margin
        pixels = cell_colors(place.window(top, left, rows, cols))
        return cls(np.ascontiguousarray(pixels), top, left)

    def uv(self, point: Vec2) -> Vec2:
        """Where a point on the place's map's plane lies on the picture."""
        rows, cols = self.pixels.shape[:2]
        row = (point[1] - HEX_RADIUS) / HEX_ROW_STEP
        return (point[0] - self.left) / cols, (row + 0.5 - self.top) / rows

    def uvs(self, lots: Lots, lot: Lot) -> list[Vec2]:
        """Where a lot's corners lie on the picture, in the order of lot_corners."""
        return [self.uv(corner) for corner in lots.corners(lot)]


def shown_lots(lots: Lots) -> list[tuple[Lot, bool]]:
    """Every lot not owned to draw, with whether it is on offer: those on offer, and those with
    cells in the map's gaps."""
    offers = set(lots.offers)
    return sorted((lot, lot in offers) for lot in offers | set(lots.locked_lots()))


@dataclass
class LandOffers:
    """The lots shown around the map, which one the mouse is over, the one the player clicked
    and is asked to confirm, and the one confirmed, for the game to buy (railroads.ui.app)."""

    hovered: Lot | None = None
    asked: Lot | None = None
    wanted: Lot | None = None
    # The ground of the place's map, and its texture once made.
    place: PlacePicture | None = None
    _texture: hello_imgui.TextureGpu | None = None
    # The lots shown, with the lots' state they were found for.
    _shown: tuple[tuple[int, int, int], list[tuple[Lot, bool]]] | None = None
    # Each lot's corners on the game's map and on the place's picture (lot_corners,
    # PlacePicture.uvs), kept while the lots shown are.
    _outlines: dict[Lot, tuple[list[Vec2], list[Vec2] | None]] = field(default_factory=dict)

    def shown(self, lots: Lots) -> list[tuple[Lot, bool]]:
        """shown_lots, found again only once a lot is bought."""
        key = (id(lots), lots.bought, len(lots.offers))
        if self._shown is None or self._shown[0] != key:
            self._shown = (key, shown_lots(lots))
            self._outlines = {}
        return self._shown[1]

    def _outline(self, lots: Lots, lot: Lot) -> tuple[list[Vec2], list[Vec2] | None]:
        outline = self._outlines.get(lot)
        if outline is None:
            uvs = None if self.place is None else self.place.uvs(lots, lot)
            outline = self._outlines[lot] = (lot_corners(lots, lot), uvs)
        return outline

    def show_place(self, place: TileMap, lots: Lots) -> None:
        """Draws the lots with the ground of the place's map from now on."""
        self.place = PlacePicture.of(place, lots)
        self._texture = None
        self._outlines = {}

    def _place_texture(self) -> imgui.ImTextureRef | None:
        if self.place is None:
            return None
        if self._texture is None:
            self._texture = hello_imgui.create_texture_gpu_from_rgba_data(self.place.pixels)
        return imgui.ImTextureRef(self._texture.texture_id())

    def hover(self, lots: Lots | None, point: Vec2 | None) -> None:
        """Notes the lot shown under a point on the map, if any."""
        self.hovered = None
        if lots is None or point is None:
            return
        lot = lot_under(lots, *point)
        if any(lot == shown for shown, _ in self.shown(lots)):
            self.hovered = lot

    def click(self, lots: Lots | None) -> bool:
        """Asks to confirm buying the lot clicked, if it is on offer; whether a lot took it."""
        if lots is None or self.hovered is None:
            return False
        if self.hovered in lots.offers:
            self.asked = self.hovered
        return True

    def draw(
        self,
        draw_list: imgui.ImDrawList,
        camera: IsoCamera,
        center: Vec2,
        tile_map: TileMap,
        lots: Lots,
        world: imgui.ImTextureRef,
        cash: float,
    ) -> None:
        price = lots.price()
        ground = self._place_texture()
        low, high = draw_list.get_clip_rect_min(), draw_list.get_clip_rect_max()
        seen = (low.x - LINK, low.y - LINK, high.x + LINK, high.y + LINK)
        steel = (rgba_u32(STEEL_DARK), rgba_u32(STEEL))
        for lot, offered in self.shown(lots):
            plane, place_uvs = self._outline(lots, lot)
            screen = [camera.to_screen(x, y, center) for x, y in plane]
            if not reaches_box(screen, seen, TAG_REACH):
                continue
            if ground is None or place_uvs is None:
                picture, uvs = world, [world_uv(tile_map, point) for point in plane]
            else:
                picture, uvs = ground, place_uvs
            points = [imgui.ImVec2(*point) for point in screen]
            corners = [imgui.ImVec2(*uv) for uv in uvs]
            tint = rgba_u32(LAND_TINT)
            for a, b, c, d in ((0, 1, 2, 3), (3, 4, 5, 0)):
                draw_list.add_image_quad(
                    picture,
                    points[a],
                    points[b],
                    points[c],
                    points[d],
                    corners[a],
                    corners[b],
                    corners[c],
                    corners[d],
                    tint,
                )
            shade = HOVERED_SHADE if lot == self.hovered and offered else SHADE
            draw_list.add_convex_poly_filled(points, rgba_u32(shade))
            _draw_chain(draw_list, screen, seen, steel)
            middle = (
                sum(x for x, _ in screen) / len(screen),
                sum(y for _, y in screen) / len(screen),
            )
            radius = math.dist(middle, screen[0])
            size = min(max(radius * LOCK_SHARE * 2, LOCK_SIZE[0]), LOCK_SIZE[1])
            _draw_padlock(draw_list, middle, size)
            if offered:
                _draw_price(draw_list, middle, size, dollars(price), cash >= price)

    def confirm(self, lots: Lots | None, finances: Finances) -> None:
        """The window asking to confirm buying the lot clicked."""
        if self.asked is not None and not imgui.is_popup_open(CONFIRM_POPUP):
            imgui.open_popup(CONFIRM_POPUP)
        flags = imgui.WindowFlags_.always_auto_resize.value
        if not imgui.begin_popup_modal(CONFIRM_POPUP, None, flags)[0]:
            return
        if lots is None or self.asked is None:
            imgui.close_current_popup()
            imgui.end_popup()
            return
        price = lots.price()
        imgui.text(f"Do you confirm buying this land for {dollars(price)}?")
        imgui.text_disabled("It grows up to this year first, as the rest of the map did.")
        afford = finances.can_afford(price)
        if not afford:
            imgui.text_colored(imgui.ImVec4(*GONE), f"You have {dollars(finances.cash)}.")
        imgui.begin_disabled(not afford)
        if imgui.button("Buy"):
            self.wanted, self.asked = self.asked, None
            imgui.close_current_popup()
        imgui.end_disabled()
        imgui.same_line()
        if imgui.button("Cancel") or imgui.is_key_pressed(imgui.Key.escape):
            self.asked = None
            imgui.close_current_popup()
        imgui.end_popup()


def _draw_chain(
    draw_list: imgui.ImDrawList,
    corners: list[Vec2],
    seen: tuple[float, float, float, float],
    colours: tuple[int, int],
) -> None:
    """A chain round the lot: links along each side, every other one seen edge on. Only the links
    in the box `seen` (left, top, right, bottom) are drawn, each where it lies on its whole side,
    in a dark and a light colour."""
    dark, light = colours
    for start, end in zip(corners, corners[1:] + corners[:1], strict=True):
        length = math.dist(start, end)
        if length < LINK:
            continue
        part = seen_part(start, end, seen)
        if part is None:
            continue
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        links = int(length / (LINK * 0.8))
        first = max(0, math.ceil(part[0] * links - 0.5))
        last = min(links - 1, math.floor(part[1] * links - 0.5))
        for index in range(first, last + 1):
            share = (index + 0.5) / links
            at = imgui.ImVec2(
                start[0] + (end[0] - start[0]) * share, start[1] + (end[1] - start[1]) * share
            )
            flat = index % 2 == 1
            radius = imgui.ImVec2(LINK * 0.55, LINK * (0.1 if flat else 0.3))
            draw_list.add_ellipse(at, radius, dark, angle, 12, 3.2)
            draw_list.add_ellipse(at, radius, light, angle, 12, 1.6)


def reaches_box(points: list[Vec2], box: tuple[float, float, float, float], reach: float) -> bool:
    """Whether the box round the points, `reach` larger each way, overlaps the box (left, top,
    right, bottom)."""
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return (
        min(xs) - reach < box[2]
        and max(xs) + reach > box[0]
        and min(ys) - reach < box[3]
        and max(ys) + reach > box[1]
    )


def seen_part(
    start: Vec2, end: Vec2, box: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    """The shares of the way from `start` to `end` between which the line lies in the box
    (left, top, right, bottom); None if it misses it."""
    low, high = 0.0, 1.0
    for origin, step, near, far in (
        (start[0], end[0] - start[0], box[0], box[2]),
        (start[1], end[1] - start[1], box[1], box[3]),
    ):
        if step == 0.0:
            if not near <= origin <= far:
                return None
            continue
        a, b = (near - origin) / step, (far - origin) / step
        low, high = max(low, min(a, b)), min(high, max(a, b))
        if low > high:
            return None
    return low, high


def _draw_padlock(draw_list: imgui.ImDrawList, middle: Vec2, size: float) -> None:
    """A brass padlock with a steel shackle, `size` points tall, centred a little above
    `middle`."""
    x, y = middle
    top = y - size * 0.15
    shackle = size * 0.28
    draw_list.path_clear()
    draw_list.path_arc_to(imgui.ImVec2(x, top), shackle, math.pi, 2 * math.pi, 16)
    draw_list.path_stroke(rgba_u32(STEEL_DARK), size * 0.16)
    draw_list.path_arc_to(imgui.ImVec2(x, top), shackle, math.pi, 2 * math.pi, 16)
    draw_list.path_stroke(rgba_u32(STEEL), size * 0.09)
    body = (imgui.ImVec2(x - size * 0.42, top), imgui.ImVec2(x + size * 0.42, top + size * 0.62))
    draw_list.add_rect_filled(*body, rgba_u32(BRASS), size * 0.1)
    draw_list.add_rect(*body, rgba_u32(BRASS_DARK), size * 0.1, max(1.0, size * 0.05))
    hole = imgui.ImVec2(x, top + size * 0.24)
    draw_list.add_circle_filled(hole, size * 0.08, rgba_u32(STEEL_DARK))
    draw_list.add_triangle_filled(
        imgui.ImVec2(x, hole.y),
        imgui.ImVec2(x - size * 0.06, top + size * 0.46),
        imgui.ImVec2(x + size * 0.06, top + size * 0.46),
        rgba_u32(STEEL_DARK),
    )


def _draw_price(
    draw_list: imgui.ImDrawList, middle: Vec2, size: float, price: str, afford: bool
) -> None:
    """The price on a tag below the padlock: brass if there is cash for it, red if not."""
    text = imgui.calc_text_size(price)
    x, y = middle[0] - text.x / 2, middle[1] + size * 0.55
    pad = 5.0
    draw_list.add_rect_filled(
        imgui.ImVec2(x - pad, y - pad / 2),
        imgui.ImVec2(x + text.x + pad, y + text.y + pad / 2),
        rgba_u32(OVERLAY),
        4.0,
    )
    draw_list.add_text(imgui.ImVec2(x, y), rgba_u32(BRASS if afford else GONE), price)
