"""The bar along the top of the window: the date and the game speeds beside it, the money, the
company's worth against its target and any countdown to bankruptcy, the view's buttons (zoom, the
two minimaps and the search), the map's title and any notice such as an autosave, and the buttons
opening the reports, the supply and demand boards, the help, the saves and the main menu."""

from typing import TYPE_CHECKING

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.clock import MONTH_NAMES, SPEEDS
from railroads.sim.finance import Finances, dollars
from railroads.ui.goal_view import draw_worth
from railroads.ui.map_input import key_pressed
from railroads.ui.ornate import DEEP, band, bar_buttons, diamond, frame_rect, spaced, spaced_text
from railroads.ui.spotlight import spot
from railroads.ui.summaries_panel import CARDS_TOP
from railroads.ui.theme import BRASS, GONE, GROWN, color_u32, danger_button, tip, toggle_button

if TYPE_CHECKING:
    from railroads.ui.app import App
    from railroads.ui.map_view import MapView

WINDOW_BUTTON_WIDTH = 30.0
# Below this window width the bar leaves out the target and the view's buttons (the wheel, M, W
# and F still do their work), so the speeds and the window buttons fit.
COMPACT_WIDTH = 1100.0
LEAVE_POPUP = "Leave this game?"
# How much larger than the text around it the banner over a paused game is written, and the room
# round it.
PAUSE_SCALE = 1.4
PAUSE_PADDING = 14.0


def draw_top_bar(app: App) -> None:
    """The bar on a dark band edged in brass, its groups set apart by diamonds."""
    draw_list = imgui.get_window_draw_list()
    top = imgui.get_cursor_screen_pos().y
    draw_list.channels_split(2)
    draw_list.channels_set_current(1)
    with bar_buttons():
        _draw_top_bar(app)
    draw_list.channels_set_current(0)
    pad = imgui.get_style().window_padding.y
    low = imgui.get_window_pos()
    bottom = top + imgui.get_frame_height() + pad - 2
    band(draw_list, low, imgui.ImVec2(low.x + imgui.get_window_width(), bottom))
    draw_list.channels_merge()


def _draw_top_bar(app: App) -> None:
    clock = app.clock
    if not imgui.get_io().want_text_input:
        if key_pressed(imgui.Key.minus, imgui.Key.keypad_subtract):
            clock.faster(-1)
        if key_pressed(imgui.Key.equal, imgui.Key.keypad_add):
            clock.faster(1)
        if key_pressed(imgui.Key.space):
            clock.paused = not clock.paused
    imgui.align_text_to_frame_padding()
    start = imgui.get_cursor_pos_x()
    imgui.text_colored(imgui.ImVec4(*BRASS), f"{fa.ICON_FA_CLOCK}  {clock.date}")
    imgui.same_line(start + date_width() + 8)
    if toggle_button(fa.ICON_FA_PAUSE, clock.paused, "Stop time while you build (Space)"):
        clock.paused = not clock.paused
    imgui.same_line(0, 2)
    for speed in SPEEDS:
        if toggle_button(
            f"x{speed}",
            clock.speed == speed and not clock.paused,
            f"Run {speed} times as fast (- and = change speed)",
        ):
            clock.speed, clock.paused = speed, False
        imgui.same_line(0, 2)
    imgui.same_line(0, 6)
    diamond()
    imgui.same_line(0, 8)
    draw_money(app.view.railway.finances)
    imgui.same_line(0, 8)
    diamond()
    imgui.same_line(0, 8)
    compact = imgui.get_window_width() < COMPACT_WIDTH
    draw_worth(app.game, target=not compact)
    if not compact:
        imgui.same_line(0, 8)
        diamond()
        imgui.same_line(0, 8)
        draw_view_controls(app.view)
    padding = imgui.get_style().window_padding.x
    buttons = 5 if app.saves is not None else 3
    right = imgui.get_window_width() - padding - buttons * WINDOW_BUTTON_WIDTH - (buttons - 1) * 4
    title = app.title.upper()
    imgui.same_line(0, 8)
    room = right - imgui.get_cursor_pos_x() - 24
    if app.title and spaced(title) <= room:
        diamond()
        imgui.same_line(0, 8)
        imgui.dummy((spaced(title), imgui.get_frame_height()))
        at = imgui.get_item_rect_min()
        text_top = at.y + (imgui.get_frame_height() - imgui.get_font_size()) / 2
        spaced_text(imgui.get_window_draw_list(), (at.x, text_top), title, BRASS)
    if app.notice:
        imgui.same_line(0, 16)
        imgui.align_text_to_frame_padding()
        imgui.text_colored(imgui.ImVec4(*GROWN), app.notice)
    imgui.same_line(0, 8)
    imgui.same_line(max(right, imgui.get_cursor_pos_x()))
    size = (WINDOW_BUTTON_WIDTH, 0)
    if toggle_button(fa.ICON_FA_CHART_BAR, app.reports.open, "Reports (G)", size):
        app.reports.open = not app.reports.open
    imgui.same_line(0, 4)
    if toggle_button(fa.ICON_FA_BALANCE_SCALE, app.tiles.open, "Supply and demand (Y)", size):
        app.tiles.open = not app.tiles.open
    spot("boards")
    imgui.same_line(0, 4)
    if toggle_button(
        fa.ICON_FA_QUESTION_CIRCLE, app.help.open, "Buildings, goods and engines (H)", size
    ):
        app.help.open = not app.help.open
    if app.saves is not None:
        imgui.same_line(0, 4)
        if toggle_button(fa.ICON_FA_SAVE, app.saves_window.open, "Save and load (F5)", size):
            app.toggle_saves()
        imgui.same_line(0, 4)
        if toggle_button(fa.ICON_FA_HOME, False, "Main menu", size):
            imgui.open_popup(LEAVE_POPUP)
        draw_leave_popup(app)


def draw_pause_banner(top_left: tuple[float, float], bottom_right: tuple[float, float]) -> None:
    """A banner saying time stands still, centred over the map at the height of the news cards
    and in front of everything else."""
    draw_list = imgui.get_foreground_draw_list()
    size = imgui.get_font_size() * PAUSE_SCALE
    text = f"{fa.ICON_FA_PAUSE}  PAUSED  -  Space to go on"
    width = imgui.calc_text_size(text).x * PAUSE_SCALE
    middle = (top_left[0] + bottom_right[0]) / 2
    top = top_left[1] + CARDS_TOP
    low = (middle - width / 2 - PAUSE_PADDING, top)
    high = (middle + width / 2 + PAUSE_PADDING, top + size + 2 * PAUSE_PADDING * 0.6)
    draw_list.add_rect_filled(low, high, color_u32(DEEP), 6.0)
    frame_rect(draw_list, imgui.ImVec2(*low), imgui.ImVec2(*high), 6.0, bracket=8.0)
    draw_list.add_text(
        imgui.get_font(),
        size,
        (low[0] + PAUSE_PADDING, top + PAUSE_PADDING * 0.6),
        color_u32(BRASS),
        text,
    )


def date_width() -> float:
    """How wide the date is in its longest month, so what follows it stays put."""
    return max(imgui.calc_text_size(f"{fa.ICON_FA_CLOCK}  {month} 8888").x for month in MONTH_NAMES)


def draw_leave_popup(app: App) -> None:
    """Asks before leaving the game for the main menu."""
    if not imgui.begin_popup_modal(LEAVE_POPUP, None, imgui.WindowFlags_.always_auto_resize)[0]:
        return
    imgui.text("Anything since your last save or autosave will be lost.")
    if danger_button("Leave for the main menu"):
        app.menu_wanted = True
        imgui.close_current_popup()
    imgui.same_line()
    if imgui.button("Stay") or imgui.is_key_pressed(imgui.Key.escape):
        imgui.close_current_popup()
    imgui.end_popup()


def draw_money(finances: Finances) -> None:
    """The cash, on the current line, with this month's and last month's books on hover."""
    imgui.align_text_to_frame_padding()
    color = BRASS if finances.cash >= 0 else GONE
    imgui.text_colored(imgui.ImVec4(*color), f"{fa.ICON_FA_MONEY_BILL}  {dollars(finances.cash)}")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        for title, books in (
            ("This month", finances.month),
            ("Last month", finances.last_month),
        ):
            imgui.text(f"{title}: {dollars(sum(books.values()))}")
            for what, amount in sorted(books.items(), key=lambda item: -item[1]):
                imgui.text_colored(
                    imgui.ImVec4(*(GROWN if amount >= 0 else GONE)), f"   {dollars(amount)}"
                )
                imgui.same_line()
                imgui.text_disabled(what)
        imgui.end_tooltip()


def draw_view_controls(view: MapView) -> None:
    """The view's buttons, on the current line: zoom, the two minimaps and the search."""
    camera, minimap = view.camera, view.minimap
    if imgui.button(f"{fa.ICON_FA_SEARCH_MINUS}"):
        camera.zoom(-1, (0, 0), (0, 0))
    tip("Zoom out, or turn the wheel over the map")
    imgui.same_line(0, 4)
    if imgui.button(f"{fa.ICON_FA_SEARCH_PLUS}"):
        camera.zoom(1, (0, 0), (0, 0))
    tip("Zoom in, or turn the wheel over the map")
    imgui.same_line()
    if toggle_button(f"{fa.ICON_FA_MAP}", minimap.shown, "Map overview (M)"):
        minimap.shown = not minimap.shown
    imgui.same_line(0, 4)
    if toggle_button(f"{fa.ICON_FA_GLOBE}", minimap.world_shown, "World map (W)"):
        minimap.world_shown = not minimap.world_shown
    imgui.same_line()
    if toggle_button(f"{fa.ICON_FA_SEARCH}", view.search.open, "Find buildings (F)"):
        view.toggle_search()
