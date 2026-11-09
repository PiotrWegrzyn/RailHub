"""The look of every window: warm dark panels with brass highlights, and the small widgets the
windows share (toggle buttons, car chips, good swatches)."""

from collections.abc import Mapping

from imgui_bundle import imgui

from railroads.sim.economy import Good
from railroads.sim.railway import CAR_CAPACITY

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
GROWN = (0.56, 0.8, 0.4, 1.0)
GONE = (0.92, 0.46, 0.36, 1.0)

# The colour of each good, as cars, swatches and loads show it.
GOOD_COLORS: Mapping[Good, Rgba] = {
    Good.FOOD: (0.95, 0.8, 0.25, 1.0),
    Good.LUMBER: (0.62, 0.4, 0.2, 1.0),
    Good.COAL: (0.36, 0.36, 0.4, 1.0),
    Good.STEEL: (0.55, 0.68, 0.85, 1.0),
    Good.GOODS: (0.86, 0.32, 0.26, 1.0),
}
CAR_EMPTY = (0.17, 0.15, 0.13, 1.0)
CHIP_SIZE = (16.0, 20.0)
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


def car_chip(key: str, good: Good, load: float | None = None, tooltip: str = "") -> bool:
    """A car drawn as a small upright chip in its good's colour, filled from the bottom by its
    load when one is given; returns whether it was clicked."""
    clicked = imgui.invisible_button(key, CHIP_SIZE)
    low = imgui.get_item_rect_min()
    high = imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()
    color = GOOD_COLORS[good]
    if load is None:
        draw_list.add_rect_filled(low, high, color_u32(color), 3.0)
    else:
        draw_list.add_rect_filled(low, high, color_u32(CAR_EMPTY), 3.0)
        share = max(0.0, min(1.0, load / CAR_CAPACITY))
        if share > 0:
            top = high.y - (high.y - low.y) * share
            draw_list.add_rect_filled((low.x, top), high, color_u32(color), 3.0)
    hovered = imgui.is_item_hovered()
    edge = BRASS if hovered else (0.0, 0.0, 0.0, 0.6)
    draw_list.add_rect(low, high, color_u32(edge), 3.0, 1.5 if hovered else 1.0)
    if tooltip and hovered:
        imgui.set_tooltip(tooltip)
    return clicked


def good_swatch(good: Good) -> None:
    """A small square in the good's colour, inline with text."""
    size = imgui.get_text_line_height() * 0.7
    imgui.color_button(
        f"##{good.value}",
        imgui.ImVec4(*GOOD_COLORS[good]),
        imgui.ColorEditFlags_.no_tooltip.value | imgui.ColorEditFlags_.no_border.value,
        (size, size),
    )


def goods_inline(goods: Mapping[Good, float], empty: str = "-") -> None:
    """Each good held, as a swatch and an amount, on one line."""
    shown = [(good, amount) for good, amount in goods.items() if amount >= 0.5]
    if not shown:
        imgui.text_disabled(empty)
        return
    for index, (good, amount) in enumerate(shown):
        if index:
            imgui.same_line(0, 10)
        imgui.align_text_to_frame_padding()
        good_swatch(good)
        if imgui.is_item_hovered():
            imgui.set_tooltip(good.value)
        imgui.same_line(0, 4)
        imgui.text(f"{amount:.0f}")
