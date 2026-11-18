"""Ornate window furniture in black, brown and brass, drawn by hand on imgui's draw lists: windows
with a thin glowing brass edge and corner brackets under a centred, spaced-out title of their own;
section headings with a fading rule; cards; filled bars; round badges holding a good's icon; and
tiles for a figure and its label."""

from collections.abc import Iterator
from contextlib import contextmanager

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.goods import Good, Need
from railroads.ui.theme import BRASS, TEXT_DIM, Rgba, color_u32, draw_good_icon

DEEP = (0.065, 0.05, 0.04, 0.97)
CARD = (0.12, 0.093, 0.07, 1.0)
CARD_HEAD = (0.17, 0.125, 0.085, 1.0)
EDGE = (0.5, 0.36, 0.19, 0.85)
GLOW = (0.86, 0.6, 0.28)
TRACK = (0.05, 0.04, 0.03, 1.0)
# How far the glow round a window reaches, in rings, and how long its corner brackets are.
GLOW_RINGS = 4
BRACKET = 14.0
# The room between the letters of titles and section headings.
SPACING = 1.6
TITLE_SCALE = 1.2
# The room at least on each side of a title, for its rules and its close cross.
TITLE_ROOM = 48.0
BADGE = 26.0


def _fade(colour: tuple[float, float, float], alpha: float) -> int:
    return color_u32((*colour, alpha))


def spaced(text: str, scale: float = 1.0) -> float:
    """How wide text drawn by spaced_text is."""
    widths = [imgui.calc_text_size(ch).x * scale for ch in text]
    return sum(widths) + SPACING * max(len(text) - 1, 0)


def spaced_text(
    draw_list: imgui.ImDrawList,
    at: tuple[float, float],
    text: str,
    colour: Rgba,
    scale: float = 1.0,
) -> None:
    """Text drawn a letter at a time with room between them, as titles are."""
    size = imgui.get_font_size() * scale
    font = imgui.get_font()
    x, y = at
    for ch in text:
        draw_list.add_text(font, size, (x, y), color_u32(colour), ch)
        x += imgui.calc_text_size(ch).x * scale + SPACING


@contextmanager
def ornate_window(title: str, key: str, flags: int = 0) -> Iterator[tuple[bool, bool]]:
    """begin_ornate and end_ornate round what the window holds; yields whether it is expanded
    and whether it stays open."""
    shown = begin_ornate(f"{title}###{key}", flags)
    try:
        yield shown
    finally:
        end_ornate()


# The window flags that say how a window scrolls, which its body takes over.
SCROLL_FLAGS = (
    imgui.WindowFlags_.horizontal_scrollbar.value
    | imgui.WindowFlags_.no_scrollbar.value
    | imgui.WindowFlags_.always_vertical_scrollbar.value
    | imgui.WindowFlags_.no_scroll_with_mouse.value
)
# Whether each ornate window begun and not yet ended opened its body, innermost last.
_bodies: list[bool] = []


def begin_ornate(title: str, flags: int = 0) -> tuple[bool, bool]:
    """Begins a window without imgui's title bar, framed in glowing brass with corner brackets,
    its title (the part before any "###", which keys it) centred over it in spaced capitals and a
    close cross in its corner. Both stay put at the top: what the window holds goes in a body
    under them, which scrolls on its own. Returns whether it is expanded and whether it stays
    open; end_ornate ends it."""
    label, _, key = title.partition("###")
    key = key or label
    imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(*DEEP))
    imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(0, 0, 0, 0))
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(16, 12))
    own = (flags & ~SCROLL_FLAGS) | imgui.WindowFlags_.no_title_bar.value
    own |= imgui.WindowFlags_.no_scrollbar.value | imgui.WindowFlags_.no_scroll_with_mouse.value
    expanded, _ = imgui.begin(f"{label}###{key}", None, own)
    imgui.pop_style_var()
    imgui.pop_style_color(2)
    still_open = True
    if expanded:
        _frame()
        still_open = _title(label)
        # A window sizing itself to what it holds keeps the width it is given, and its body
        # as tall as what is in it.
        sizes_itself = flags & imgui.WindowFlags_.always_auto_resize.value
        body = imgui.ChildFlags_.auto_resize_y.value if sizes_itself else 0
        imgui.begin_child(f"##body{key}", (0, 0), body, flags & SCROLL_FLAGS)
    _bodies.append(bool(expanded))
    return bool(expanded), still_open


def end_ornate() -> None:
    """Ends the window begin_ornate began, and its body."""
    if _bodies.pop():
        imgui.end_child()
    imgui.end()


def _frame() -> None:
    """The window's glowing brass edge and its corner brackets."""
    low, size = imgui.get_window_pos(), imgui.get_window_size()
    high = imgui.ImVec2(low.x + size.x, low.y + size.y)
    frame_rect(imgui.get_window_draw_list(), low, high, imgui.get_style().window_rounding)


def frame_rect(
    draw_list: imgui.ImDrawList,
    low: imgui.ImVec2,
    high: imgui.ImVec2,
    rounding: float,
    bracket: float = BRACKET,
) -> None:
    """A glowing brass edge round a rectangle on the screen, with brackets at its corners."""
    draw_list.push_clip_rect_full_screen()
    for ring in range(GLOW_RINGS, 0, -1):
        grow = ring * 1.5
        draw_list.add_rect(
            (low.x - grow, low.y - grow),
            (high.x + grow, high.y + grow),
            _fade(GLOW, 0.07 * (GLOW_RINGS + 1 - ring) / GLOW_RINGS),
            rounding + grow,
            thickness=1.5,
        )
    draw_list.add_rect(low, high, _fade(GLOW, 0.75), rounding, thickness=1.0)
    bright = _fade(GLOW, 0.95)
    inset = 4.0
    for x, y, dx, dy in (
        (low.x + inset, low.y + inset, 1, 1),
        (high.x - inset, low.y + inset, -1, 1),
        (low.x + inset, high.y - inset, 1, -1),
        (high.x - inset, high.y - inset, -1, -1),
    ):
        draw_list.add_line((x, y), (x + dx * bracket, y), bright, 2.0)
        draw_list.add_line((x, y), (x, y + dy * bracket), bright, 2.0)
    draw_list.pop_clip_rect()


def _title(title: str) -> bool:
    """The window's title centred over it in spaced capitals between fading rules, with a close
    cross at the right; returns False once the cross is pressed."""
    draw_list = imgui.get_window_draw_list()
    text = title.upper()
    left = imgui.get_cursor_screen_pos()
    height = imgui.get_font_size() * TITLE_SCALE
    wide = spaced(text, TITLE_SCALE)
    width = max(imgui.get_content_region_avail().x, wide + 2 * TITLE_ROOM)
    start = left.x + (width - wide) / 2
    spaced_text(draw_list, (start, left.y), text, BRASS, TITLE_SCALE)
    middle = left.y + height / 2 + 1
    rule(draw_list, left.x + 18, start - 10, middle, fade_left=True)
    rule(draw_list, start + wide + 10, left.x + width - 26, middle, fade_left=False)
    imgui.dummy((width - 18, height))
    imgui.same_line(0, 0)
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0, 0, 0, 0))
    closed = imgui.small_button(fa.ICON_FA_TIMES)
    imgui.pop_style_color()
    imgui.spacing()
    return not closed


def rule(
    draw_list: imgui.ImDrawList, left: float, right: float, y: float, *, fade_left: bool
) -> None:
    """A thin brass rule fading out towards one end."""
    if right <= left:
        return
    solid, clear = _fade(GLOW, 0.8), _fade(GLOW, 0.0)
    ends = (clear, solid) if fade_left else (solid, clear)
    draw_list.add_rect_filled_multi_color(
        (left, y), (right, y + 1), ends[0], ends[1], ends[1], ends[0]
    )


def section(text: str) -> None:
    """A section heading: small spaced capitals and a rule fading out after them."""
    imgui.spacing()
    draw_list = imgui.get_window_draw_list()
    low = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    label = text.upper()
    wide = spaced(label, 0.9)
    spaced_text(draw_list, (low.x, low.y), label, BRASS, 0.9)
    middle = low.y + imgui.get_font_size() * 0.45 + 1
    rule(draw_list, low.x + wide + 8, low.x + width, middle, fade_left=False)
    imgui.dummy((width, imgui.get_font_size()))


@contextmanager
def card(key: str, highlight: bool = False) -> Iterator[None]:
    """A card: a brown panel with a brass edge, brighter for the one picked."""
    imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(*CARD))
    edge = (*GLOW, 0.95) if highlight else EDGE
    imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(*edge))
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(10, 8))
    flags = (
        imgui.ChildFlags_.borders.value
        | imgui.ChildFlags_.auto_resize_y.value
        | imgui.ChildFlags_.always_use_window_padding.value
    )
    imgui.begin_child(key, (0, 0), flags)
    try:
        yield
    finally:
        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_style_color(2)


def bar(fraction: float, colour: Rgba, width: float, height: float = 6.0) -> None:
    """A rounded bar filled `fraction` of the way, past 1 drawn full, on the current line."""
    imgui.dummy((width, imgui.get_text_line_height()))
    low = imgui.get_item_rect_min()
    top = low.y + (imgui.get_text_line_height() - height) / 2
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(
        (low.x, top), (low.x + width, top + height), color_u32(TRACK), height / 2
    )
    filled = width * max(0.0, min(1.0, fraction))
    if filled >= 1:
        draw_list.add_rect_filled(
            (low.x, top), (low.x + filled, top + height), color_u32(colour), height / 2
        )
        draw_list.add_line(
            (low.x + height / 2, top + 1),
            (low.x + filled - height / 2, top + 1),
            color_u32((1.0, 0.95, 0.8, 0.25)),
        )
    draw_list.add_rect((low.x, top), (low.x + width, top + height), _fade(GLOW, 0.35), height / 2)


def badge(good: Good | Need | None) -> None:
    """A round brass-edged badge holding a good's icon, or a blank one."""
    imgui.dummy((BADGE, BADGE))
    low = imgui.get_item_rect_min()
    centre = (low.x + BADGE / 2, low.y + BADGE / 2)
    draw_list = imgui.get_window_draw_list()
    draw_list.add_circle_filled(centre, BADGE / 2, color_u32(TRACK))
    draw_list.add_circle(centre, BADGE / 2 + 1.5, _fade(GLOW, 0.2), 0, 2.0)
    draw_list.add_circle(centre, BADGE / 2, _fade(GLOW, 0.85), 0, 1.2)
    if good is not None:
        icon = 16.0
        draw_good_icon(draw_list, good, (centre[0] - icon / 2, centre[1] - icon / 2), icon)


def tile(label: str, value: str, colour: Rgba, width: float) -> None:
    """A tile of a figure: its label small and dim over its value, in a framed brown box."""
    height = imgui.get_text_line_height() * 2 + 14
    imgui.dummy((width, height))
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(low, high, color_u32(CARD), 5.0)
    draw_list.add_rect_filled_multi_color(
        (low.x + 1, low.y + 1),
        (high.x - 1, low.y + 3),
        _fade(GLOW, 0.0),
        _fade(GLOW, 0.5),
        _fade(GLOW, 0.5),
        _fade(GLOW, 0.0),
    )
    draw_list.add_rect(low, high, color_u32(EDGE), 5.0)
    draw_list.add_text((low.x + 8, low.y + 6), color_u32(TEXT_DIM), label)
    font = imgui.get_font()
    size = imgui.get_font_size() * TITLE_SCALE
    draw_list.add_text(
        font, size, (low.x + 8, low.y + 6 + imgui.get_text_line_height()), color_u32(colour), value
    )


def frame_window() -> None:
    """Frames the current window in glowing brass with corner brackets."""
    _frame()


@contextmanager
def bar_buttons() -> Iterator[None]:
    """Buttons drawn as dark brown plates with a thin brass edge, as on the bars."""
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*CARD))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(*CARD_HEAD))
    imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(*EDGE))
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.0)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 4.0)
    try:
        yield
    finally:
        imgui.pop_style_var(2)
        imgui.pop_style_color(3)


def diamond(height: float | None = None) -> None:
    """A small brass diamond between faint upright rules, setting groups of buttons apart."""
    tall = imgui.get_frame_height() if height is None else height
    width = 14.0
    imgui.dummy((width, tall))
    low = imgui.get_item_rect_min()
    x, y = low.x + width / 2, low.y + tall / 2
    draw_list = imgui.get_window_draw_list()
    draw_list.add_line((x, low.y + 2), (x, y - 6), _fade(GLOW, 0.3))
    draw_list.add_line((x, y + 6), (x, low.y + tall - 2), _fade(GLOW, 0.3))
    size = 3.5
    draw_list.add_quad_filled(
        (x, y - size), (x + size, y), (x, y + size), (x - size, y), _fade(GLOW, 0.9)
    )


def band(draw_list: imgui.ImDrawList, low: imgui.ImVec2, high: imgui.ImVec2) -> None:
    """A dark band right across the screen behind a bar, edged below by a glowing rule with a
    diamond at its middle."""
    draw_list.add_rect_filled(low, high, color_u32(DEEP))
    for ring in range(GLOW_RINGS):
        draw_list.add_line(
            (low.x, high.y + ring), (high.x, high.y + ring), _fade(GLOW, 0.25 / (ring + 1))
        )
    draw_list.add_line((low.x, high.y), (high.x, high.y), _fade(GLOW, 0.8))
    x, y, size = (low.x + high.x) / 2, high.y, 4.0
    draw_list.add_quad_filled(
        (x, y - size), (x + size, y), (x, y + size), (x - size, y), _fade(GLOW, 0.95)
    )
