"""The land around the map the player can buy (railroads.sim.lots, railroads.land). Every lot not
owned is drawn where it lies, past the map's edge or in a gap of it, as the world map shows it,
so what lies there is no surprise, behind a chain with a padlock, and with its price if it is on
offer. A click on one asks to confirm buying it."""

import math
from dataclasses import dataclass

from imgui_bundle import imgui

from railroads.sim.finance import Finances, dollars
from railroads.sim.grids import Grid
from railroads.sim.lots import Lot, Lots
from railroads.sim.tilemap import TileMap
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


def row_step(grid: Grid) -> float:
    """How far apart rows of cells lie, in the map's cell units."""
    return float(grid.center(1, 0)[1] - grid.center(0, 0)[1])


def lot_corners(grid: Grid, lots: Lots, lot: Lot) -> list[Vec2]:
    """A lot's corners on the map, in cell units: top-left, top-right, bottom-right, bottom-left."""
    row, col = lots.cell_of(lot)
    step = row_step(grid)
    top, bottom = row * step, (row + lots.rows) * step
    left, right = float(col), float(col + lots.cols)
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def lot_under(grid: Grid, lots: Lots, x: float, y: float) -> Lot:
    """The lot a point on the map in cell units lies in, past the map's edges too."""
    return lots.lot_at(math.floor(y / row_step(grid)), math.floor(x))


def world_uv(tile_map: TileMap, point: Vec2) -> Vec2:
    """Where a point on the map, in cell units, lies on the world map's picture
    (railroads.ui.minimap.world_rgba): from 180°W to 180°E across, the poles down."""
    km = tile_map.grid.km_per_cell
    lon, lat = tile_map.bounds.to_lonlat(point[0] * km, point[1] * km)
    return (float(lon) + 180.0) / 360.0, (90.0 - float(lat)) / 180.0


def shown_lots(lots: Lots) -> list[tuple[Lot, bool]]:
    """Every lot not owned to draw, with whether it is on offer: those on offer, and those in the
    map's gaps."""
    top, left = lots.corner()
    tall, wide = lots.span()
    inside = {
        (row, col)
        for row in range(top, top + tall)
        for col in range(left, left + wide)
        if (row, col) not in lots.owned
    }
    offers = set(lots.offers)
    return sorted((lot, lot in offers) for lot in inside | offers)


@dataclass
class LandOffers:
    """The lots shown around the map, which one the mouse is over, the one the player clicked
    and is asked to confirm, and the one confirmed, for the game to buy (railroads.ui.app)."""

    hovered: Lot | None = None
    asked: Lot | None = None
    wanted: Lot | None = None

    def hover(self, lots: Lots | None, grid: Grid, point: Vec2 | None) -> None:
        """Notes the lot shown under a point on the map, if any."""
        self.hovered = None
        if lots is None or point is None:
            return
        lot = lot_under(grid, lots, *point)
        if any(lot == shown for shown, _ in shown_lots(lots)):
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
        for lot, offered in shown_lots(lots):
            plane = lot_corners(tile_map.grid, lots, lot)
            screen = [camera.to_screen(x, y, center) for x, y in plane]
            uvs = [world_uv(tile_map, point) for point in plane]
            points = [imgui.ImVec2(*point) for point in screen]
            draw_list.add_image_quad(world, *points, *map(imgui.ImVec2, uvs), rgba_u32(LAND_TINT))
            shade = HOVERED_SHADE if lot == self.hovered and offered else SHADE
            draw_list.add_quad_filled(*points, rgba_u32(shade))
            _draw_chain(draw_list, screen)
            middle = (
                sum(x for x, _ in screen) / 4,
                sum(y for _, y in screen) / 4,
            )
            side = min(math.dist(screen[0], screen[1]), math.dist(screen[1], screen[2]))
            size = min(max(side * LOCK_SHARE, LOCK_SIZE[0]), LOCK_SIZE[1])
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


def _draw_chain(draw_list: imgui.ImDrawList, corners: list[Vec2]) -> None:
    """A chain round the lot: links along each side, every other one seen edge on."""
    for start, end in zip(corners, corners[1:] + corners[:1], strict=True):
        length = math.dist(start, end)
        if length < LINK:
            continue
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        links = int(length / (LINK * 0.8))
        for index in range(links):
            share = (index + 0.5) / links
            at = imgui.ImVec2(
                start[0] + (end[0] - start[0]) * share, start[1] + (end[1] - start[1]) * share
            )
            flat = index % 2 == 1
            radius = imgui.ImVec2(LINK * 0.55, LINK * (0.1 if flat else 0.3))
            draw_list.add_ellipse(at, radius, rgba_u32(STEEL_DARK), angle, 12, 3.2)
            draw_list.add_ellipse(at, radius, rgba_u32(STEEL), angle, 12, 1.6)


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
