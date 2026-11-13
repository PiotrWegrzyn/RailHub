"""The look of every window: warm dark panels with brass highlights, and the small widgets the
windows share (toggle buttons, car chips, good swatches)."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

import numpy as np
from imgui_bundle import hello_imgui, imgui
from PIL import Image

from railroads.sim.goods import Good, Need
from railroads.sim.trains import CAR_CAPACITY
from railroads.ui.pixel.good_icons import ICON_PIXELS, ICONS_INDEX, ICONS_PNG

type Rgba = tuple[float, float, float, float]

TEXT = (0.93, 0.89, 0.82, 1.0)
TEXT_DIM = (0.62, 0.57, 0.5, 1.0)
BRASS = (0.86, 0.66, 0.32, 1.0)
BRASS_DARK = (0.62, 0.45, 0.2, 1.0)
PANEL = (0.11, 0.1, 0.09, 0.95)
PANEL_LIGHT = (0.16, 0.14, 0.12, 1.0)
FRAME = (0.21, 0.18, 0.15, 1.0)
FRAME_HOVER = (0.29, 0.24, 0.18, 1.0)
FRAME_ACTIVE = (0.36, 0.28, 0.19, 1.0)
BORDER = (0.38, 0.3, 0.2, 0.7)
DANGER = (0.62, 0.2, 0.16, 1.0)
DANGER_HOVER = (0.78, 0.27, 0.2, 1.0)
OVERLAY = (0.08, 0.07, 0.06, 0.72)
# How each satisfaction reads in lists of buildings.
MOODS: dict[str, tuple[float, float, float, float]] = {
    "thriving": (0.5, 0.84, 0.42, 1.0),
    "content": (0.95, 0.8, 0.38, 1.0),
    "unsatisfied": (0.9, 0.45, 0.38, 1.0),
}
GROWN = (0.56, 0.8, 0.4, 1.0)
GONE = (0.92, 0.46, 0.36, 1.0)

# The colour of each good, as cars, swatches and loads show it.
GOOD_COLORS: Mapping[Good, Rgba] = {
    Good.WHEAT: (0.86, 0.7, 0.36, 1.0),
    Good.APPLES: (0.84, 0.2, 0.22, 1.0),
    Good.BANANAS: (0.98, 0.9, 0.3, 1.0),
    Good.DATES: (0.48, 0.28, 0.16, 1.0),
    Good.ORANGES: (0.96, 0.56, 0.14, 1.0),
    Good.SUGAR: (0.96, 0.95, 0.92, 1.0),
    Good.HONEY: (0.92, 0.66, 0.16, 1.0),
    Good.FISH: (0.46, 0.68, 0.8, 1.0),
    Good.CANS: (0.96, 0.56, 0.2, 1.0),
    Good.BREAD: (0.8, 0.56, 0.3, 1.0),
    Good.IRON: (0.6, 0.36, 0.28, 1.0),
    Good.COPPER: (0.8, 0.5, 0.3, 1.0),
    Good.GOLD: (1.0, 0.82, 0.2, 1.0),
    Good.DIAMONDS: (0.72, 0.92, 1.0, 1.0),
    Good.OIL: (0.2, 0.18, 0.24, 1.0),
    Good.SALT: (0.88, 0.9, 0.94, 1.0),
    Good.WOOL: (0.92, 0.88, 0.78, 1.0),
    Good.COTTON: (0.98, 0.98, 1.0, 1.0),
    Good.GRAPES: (0.46, 0.24, 0.56, 1.0),
    Good.COFFEE: (0.36, 0.22, 0.14, 1.0),
    Good.RUBBER: (0.28, 0.3, 0.26, 1.0),
    Good.TOOLS: (0.5, 0.56, 0.66, 1.0),
    Good.FUEL: (0.82, 0.74, 0.26, 1.0),
    Good.GAS: (0.5, 0.72, 0.92, 1.0),
    Good.CHEMICALS: (0.56, 0.84, 0.4, 1.0),
    Good.FURNITURE: (0.6, 0.42, 0.24, 1.0),
    Good.CLOTHES: (0.86, 0.48, 0.7, 1.0),
    Good.BARRELS: (0.58, 0.38, 0.2, 1.0),
    Good.WINE: (0.56, 0.1, 0.26, 1.0),
    Good.WAX: (0.96, 0.9, 0.62, 1.0),
    Good.INCENSE: (0.76, 0.58, 0.82, 1.0),
    Good.ELECTRONICS: (0.3, 0.84, 0.86, 1.0),
    Good.CARS: (0.9, 0.3, 0.5, 1.0),
    Good.GUNS: (0.44, 0.48, 0.3, 1.0),
    Good.TANKS: (0.32, 0.4, 0.22, 1.0),
    Good.PLANES: (0.7, 0.78, 0.88, 1.0),
    Good.LUMBER: (0.74, 0.54, 0.32, 1.0),
    Good.COAL: (0.36, 0.36, 0.4, 1.0),
    Good.STEEL: (0.62, 0.68, 0.76, 1.0),
    Good.GOODS: (0.34, 0.5, 0.88, 1.0),
    Good.CLAY: (0.78, 0.46, 0.3, 1.0),
    Good.LIMESTONE: (0.86, 0.84, 0.76, 1.0),
    Good.SAND: (0.94, 0.86, 0.58, 1.0),
    Good.POTATOES: (0.74, 0.6, 0.38, 1.0),
    Good.CATTLE: (0.5, 0.32, 0.2, 1.0),
    Good.PIGS: (0.9, 0.64, 0.62, 1.0),
    Good.CHICKENS: (0.94, 0.92, 0.86, 1.0),
    Good.EGGS: (0.96, 0.9, 0.78, 1.0),
    Good.SHEEP: (0.9, 0.9, 0.86, 1.0),
    Good.MEAT: (0.8, 0.26, 0.26, 1.0),
    Good.FLAX: (0.56, 0.66, 0.9, 1.0),
    Good.HOPS: (0.56, 0.78, 0.36, 1.0),
    Good.MACHINERY: (0.44, 0.46, 0.52, 1.0),
    Good.BRICKS: (0.74, 0.32, 0.24, 1.0),
    Good.CEMENT: (0.66, 0.66, 0.64, 1.0),
    Good.GLASS: (0.6, 0.86, 0.92, 1.0),
    Good.BEER: (0.94, 0.7, 0.2, 1.0),
    Good.VODKA: (0.88, 0.92, 0.96, 1.0),
    Good.LEATHER: (0.58, 0.36, 0.2, 1.0),
    Good.SHOES: (0.36, 0.22, 0.16, 1.0),
    Good.PLASTICS: (0.92, 0.36, 0.42, 1.0),
    Good.APPLIANCES: (0.86, 0.88, 0.9, 1.0),
    Good.PAPER: (0.96, 0.94, 0.86, 1.0),
    Good.NEWSPAPERS: (0.7, 0.7, 0.66, 1.0),
    Good.REFINED_SUGAR: (1.0, 0.92, 0.94, 1.0),
    Good.PORCELAIN: (0.4, 0.56, 0.86, 1.0),
    Good.PASSENGERS: (0.36, 0.76, 0.6, 1.0),
    Good.MAIL: (0.68, 0.44, 0.86, 1.0),
    Good.COKE: (0.3, 0.3, 0.32, 1.0),
    Good.MILK: (0.95, 0.95, 0.93, 1.0),
    Good.DAIRY: (0.95, 0.85, 0.5, 1.0),
    Good.POTS: (0.7, 0.45, 0.3, 1.0),
    Good.STOVE_TILES: (0.35, 0.55, 0.75, 1.0),
    Good.CASTINGS: (0.4, 0.4, 0.44, 1.0),
    Good.SAUSAGES: (0.7, 0.3, 0.24, 1.0),
    Good.LUXURIES: (0.6, 0.3, 0.65, 1.0),
}
CAR_EMPTY = (0.17, 0.15, 0.13, 1.0)
CAR_LOADED = (0.62, 0.45, 0.2, 0.75)
CHIP_SIZE = (24.0, 28.0)
CHIP_ICON = 20.0
# On-screen size of a good's icon; the texture holds each icon this many times over, so that the
# renderer's smoothing does not blur its pixels.
ICON_SIZE = 20.0
ICON_TEXTURE_SCALE = 8
# Room one good and its figure take in a row of them.
GOOD_ITEM_WIDTH = 78.0
# Room the monthly summary cards take along the right edge of the map; windows open beside it.
CARDS_WIDTH = 278.0


def apply_theme() -> None:
    style = imgui.get_style()
    style.window_rounding = 8.0
    style.child_rounding = 6.0
    style.frame_rounding = 5.0
    style.popup_rounding = 6.0
    style.grab_rounding = 4.0
    style.scrollbar_rounding = 6.0
    style.tab_rounding = 5.0
    style.window_border_size = 1.0
    style.frame_border_size = 0.0
    style.window_padding = imgui.ImVec2(12, 10)
    style.frame_padding = imgui.ImVec2(8, 4)
    style.item_spacing = imgui.ImVec2(8, 6)
    style.item_inner_spacing = imgui.ImVec2(6, 4)
    style.cell_padding = imgui.ImVec2(6, 4)
    style.scrollbar_size = 12.0
    style.window_title_align = imgui.ImVec2(0.0, 0.5)
    colors = {
        imgui.Col_.text: TEXT,
        imgui.Col_.text_disabled: TEXT_DIM,
        imgui.Col_.window_bg: PANEL,
        imgui.Col_.child_bg: (0.0, 0.0, 0.0, 0.0),
        imgui.Col_.popup_bg: (0.13, 0.11, 0.1, 0.98),
        imgui.Col_.border: BORDER,
        imgui.Col_.frame_bg: FRAME,
        imgui.Col_.frame_bg_hovered: FRAME_HOVER,
        imgui.Col_.frame_bg_active: FRAME_ACTIVE,
        imgui.Col_.title_bg: (0.15, 0.12, 0.1, 1.0),
        imgui.Col_.title_bg_active: (0.24, 0.17, 0.1, 1.0),
        imgui.Col_.title_bg_collapsed: (0.15, 0.12, 0.1, 0.8),
        imgui.Col_.scrollbar_bg: (0.0, 0.0, 0.0, 0.0),
        imgui.Col_.scrollbar_grab: FRAME_HOVER,
        imgui.Col_.scrollbar_grab_hovered: FRAME_ACTIVE,
        imgui.Col_.scrollbar_grab_active: BRASS_DARK,
        imgui.Col_.check_mark: BRASS,
        imgui.Col_.slider_grab: BRASS_DARK,
        imgui.Col_.slider_grab_active: BRASS,
        imgui.Col_.button: FRAME,
        imgui.Col_.button_hovered: FRAME_HOVER,
        imgui.Col_.button_active: BRASS_DARK,
        imgui.Col_.header: (0.3, 0.23, 0.15, 1.0),
        imgui.Col_.header_hovered: (0.38, 0.29, 0.18, 1.0),
        imgui.Col_.header_active: BRASS_DARK,
        imgui.Col_.separator: BORDER,
        imgui.Col_.tab: FRAME,
        imgui.Col_.tab_hovered: FRAME_ACTIVE,
        imgui.Col_.tab_selected: BRASS_DARK,
        imgui.Col_.tab_selected_overline: BRASS,
        imgui.Col_.tab_dimmed: FRAME,
        imgui.Col_.tab_dimmed_selected: BRASS_DARK,
        imgui.Col_.tab_dimmed_selected_overline: BRASS,
        imgui.Col_.resize_grip: (0.0, 0.0, 0.0, 0.0),
        imgui.Col_.resize_grip_hovered: BRASS_DARK,
        imgui.Col_.resize_grip_active: BRASS,
        imgui.Col_.plot_histogram: BRASS,
        imgui.Col_.table_header_bg: PANEL_LIGHT,
        imgui.Col_.table_border_strong: BORDER,
        imgui.Col_.table_border_light: (0.3, 0.24, 0.17, 0.4),
        imgui.Col_.table_row_bg_alt: (1.0, 0.95, 0.85, 0.03),
        imgui.Col_.text_selected_bg: (0.86, 0.66, 0.32, 0.35),
        imgui.Col_.nav_cursor: BRASS,
    }
    for index, color in colors.items():
        style.set_color_(index, imgui.ImVec4(*color))


def color_u32(rgba: Rgba) -> int:
    return imgui.get_color_u32(imgui.ImVec4(*rgba))


def toggle_button(
    label: str, on: bool, tooltip: str = "", size: tuple[float, float] = (0, 0)
) -> bool:
    """A button lit in brass while `on`; returns whether it was clicked."""
    if on:
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*BRASS_DARK))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(*BRASS))
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.12, 0.09, 0.05, 1.0))
    clicked = imgui.button(label, size)
    if on:
        imgui.pop_style_color(3)
    if tooltip and imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    return clicked


def danger_button(label: str) -> bool:
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*DANGER))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(*DANGER_HOVER))
    clicked = imgui.button(label)
    imgui.pop_style_color(2)
    return clicked


@dataclass(frozen=True)
class IconAtlas:
    texture: hello_imgui.TextureGpu
    # Where each good's icon lies on the texture, as two opposite corners from 0 to 1.
    places: Mapping[str, tuple[tuple[float, float], tuple[float, float]]]


@cache
def _icon_atlas() -> IconAtlas:
    with Image.open(ICONS_PNG) as sheet:
        rgba = np.asarray(sheet.convert("RGBA"), dtype=np.uint8)
    big = np.repeat(np.repeat(rgba, ICON_TEXTURE_SCALE, axis=0), ICON_TEXTURE_SCALE, axis=1)
    height, width = rgba.shape[:2]
    places = {
        name: ((x / width, y / height), ((x + ICON_PIXELS) / width, (y + ICON_PIXELS) / height))
        for name, (x, y) in json.loads(ICONS_INDEX.read_text()).items()
    }
    return IconAtlas(
        hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(big)), places
    )


def draw_good_icon(
    draw_list: imgui.ImDrawList,
    good: Good | Need,
    top_left: tuple[float, float],
    size: float = ICON_SIZE,
) -> None:
    """A good's icon, drawn at a place on the screen."""
    atlas = _icon_atlas()
    low, high = atlas.places[good.value]
    bottom_right = (top_left[0] + size, top_left[1] + size)
    draw_list.add_image(
        imgui.ImTextureRef(atlas.texture.texture_id()), top_left, bottom_right, low, high
    )


def good_icon(good: Good | Need, *, framed: bool = False, size: float = ICON_SIZE) -> None:
    """A good's icon, inline and named on hover, centred on the line's text; `framed` is for
    lines lined up with buttons by align_text_to_frame_padding."""
    height = imgui.get_frame_height() if framed else imgui.get_text_line_height()
    imgui.dummy((size, height))
    low = imgui.get_item_rect_min()
    draw_good_icon(imgui.get_window_draw_list(), good, (low.x, low.y + (height - size) / 2), size)
    if imgui.is_item_hovered():
        imgui.set_tooltip(good.value)


def good_amount(good: Good | Need, amount: str, *, framed: bool = False) -> None:
    """A good's icon followed by how much of it, on the current line."""
    good_icon(good, framed=framed)
    imgui.same_line(0, 3)
    imgui.text(amount)


def car_chip(key: str, good: Good, load: float | None = None, tooltip: str = "") -> bool:
    """A car drawn as a small upright chip with its good's icon, filled from the bottom by its
    load when one is given; returns whether it was clicked."""
    clicked = imgui.invisible_button(key, CHIP_SIZE)
    low = imgui.get_item_rect_min()
    high = imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(low, high, color_u32(CAR_EMPTY), 3.0)
    if load is not None:
        share = max(0.0, min(1.0, load / CAR_CAPACITY))
        if share > 0:
            top = high.y - (high.y - low.y) * share
            draw_list.add_rect_filled((low.x, top), high, color_u32(CAR_LOADED), 3.0)
    icon_x = low.x + (CHIP_SIZE[0] - CHIP_ICON) / 2
    draw_good_icon(draw_list, good, (icon_x, low.y + (CHIP_SIZE[1] - CHIP_ICON) / 2), CHIP_ICON)
    hovered = imgui.is_item_hovered()
    edge = BRASS if hovered else (0.0, 0.0, 0.0, 0.6)
    draw_list.add_rect(low, high, color_u32(edge), 3.0, 1.5 if hovered else 1.0)
    if tooltip and hovered:
        imgui.set_tooltip(tooltip)
    return clicked


OFFER_SIZE = (72.0, 28.0)
EMPTY_SLOT = (0.13, 0.12, 0.1, 1.0)


def empty_car_slot() -> None:
    """Room for one more car, drawn as a faint outline the size of a car chip."""
    imgui.dummy(CHIP_SIZE)
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(low, high, color_u32(EMPTY_SLOT), 3.0)
    draw_list.add_rect(low, high, color_u32(BORDER), 3.0, 1.0)


def offer_chip(key: str, good: Good, amount: str, wanted: bool, tooltip: str) -> bool:
    """A good to load, as a wide chip with its icon and how much of it waits, outlined in brass
    when the next stop wants it; returns whether it was clicked."""
    clicked = imgui.invisible_button(key, OFFER_SIZE)
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    hovered = imgui.is_item_hovered()
    draw_list = imgui.get_window_draw_list()
    fill = FRAME_HOVER if hovered else FRAME
    draw_list.add_rect_filled(low, high, color_u32(fill), 4.0)
    icon_y = low.y + (OFFER_SIZE[1] - CHIP_ICON) / 2
    draw_good_icon(draw_list, good, (low.x + 4, icon_y), CHIP_ICON)
    text_size = imgui.calc_text_size(amount)
    text_color = TEXT if wanted else TEXT_DIM
    draw_list.add_text(
        (high.x - 6 - text_size.x, low.y + (OFFER_SIZE[1] - text_size.y) / 2),
        color_u32(text_color),
        amount,
    )
    if wanted:
        draw_list.add_rect(low, high, color_u32(BRASS), 4.0, 2.0)
    elif hovered:
        draw_list.add_rect(low, high, color_u32(BORDER), 4.0, 1.0)
    if hovered:
        imgui.set_tooltip(tooltip)
    return clicked


def rows_that_fit() -> tuple[float, int]:
    """Where a run of goods starts on the current line, and how many fit in a row from there."""
    return imgui.get_cursor_pos_x(), max(
        1, int(imgui.get_content_region_avail().x // GOOD_ITEM_WIDTH)
    )


def next_good_slot(index: int, left: float, per_row: int) -> None:
    """Places the next good of a run: beside the last, or at the start of a new row."""
    if index % per_row:
        imgui.same_line(0, 10)
    elif index:
        imgui.set_cursor_pos_x(left)


def goods_inline(goods: Mapping[Good, float], empty: str = "-") -> None:
    """Each good held, as an icon and an amount, in as many rows as it takes."""
    shown = [(good, amount) for good, amount in goods.items() if amount >= 0.5]
    if not shown:
        imgui.text_disabled(empty)
        return
    left, per_row = rows_that_fit()
    for index, (good, amount) in enumerate(shown):
        next_good_slot(index, left, per_row)
        imgui.align_text_to_frame_padding()
        good_amount(good, f"{amount:.0f}", framed=True)
