"""The bar along the top of the window: the date and the money, the game speeds, the view's
buttons (zoom, the two minimaps and the search), the map's title, and the buttons opening the
reports, the tech trees and the help."""

from typing import TYPE_CHECKING

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.clock import MAX_PLAY_SPEED, SPEEDS
from railroads.sim.finance import Finances, dollars
from railroads.ui.theme import BRASS, GONE, GROWN, toggle_button

if TYPE_CHECKING:
    from railroads.ui.app import App
    from railroads.ui.map_view import MapView

WINDOW_BUTTON_WIDTH = 30.0


def draw_top_bar(app: App) -> None:
    clock = app.clock
    imgui.align_text_to_frame_padding()
    imgui.text_colored(imgui.ImVec4(*BRASS), f"{fa.ICON_FA_CLOCK}  {clock.date}")
    imgui.same_line(0, 16)
    draw_money(app.view.railway.finances)
    imgui.same_line(0, 16)
    for speed in SPEEDS:
        testing = speed > MAX_PLAY_SPEED
        tooltip = f"Run {speed} times as fast" + (" (for testing)" if testing else "")
        if testing:
            imgui.push_style_var(imgui.StyleVar_.alpha.value, 0.6)
        if toggle_button(f"x{speed}", clock.speed == speed, tooltip):
            clock.speed = speed
        if testing:
            imgui.pop_style_var()
        imgui.same_line(0, 2)
    imgui.same_line(0, 16)
    draw_view_controls(app.view)
    if app.title:
        imgui.same_line(0, 16)
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(app.title)
    padding = imgui.get_style().window_padding.x
    imgui.same_line(imgui.get_window_width() - padding - 3 * WINDOW_BUTTON_WIDTH - 8)
    size = (WINDOW_BUTTON_WIDTH, 0)
    if toggle_button(fa.ICON_FA_CHART_BAR, app.reports.open, "Reports (G)", size):
        app.reports.open = not app.reports.open
    imgui.same_line(0, 4)
    if toggle_button(fa.ICON_FA_FLASK, app.tech.open, "Tech trees (R)", size):
        app.tech.open = not app.tech.open
    imgui.same_line(0, 4)
    if toggle_button(fa.ICON_FA_QUESTION_CIRCLE, app.help.open, "Buildings and goods (H)", size):
        app.help.open = not app.help.open


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
    if imgui.is_item_hovered():
        imgui.set_tooltip("Zoom out (-)")
    imgui.same_line(0, 4)
    if imgui.button(f"{fa.ICON_FA_SEARCH_PLUS}"):
        camera.zoom(1, (0, 0), (0, 0))
    if imgui.is_item_hovered():
        imgui.set_tooltip("Zoom in (+)")
    imgui.same_line()
    if toggle_button(f"{fa.ICON_FA_MAP}", minimap.shown, "Map overview (M)"):
        minimap.shown = not minimap.shown
    imgui.same_line(0, 4)
    if toggle_button(f"{fa.ICON_FA_GLOBE}", minimap.world_shown, "World map (W)"):
        minimap.world_shown = not minimap.world_shown
    imgui.same_line()
    if toggle_button(f"{fa.ICON_FA_SEARCH}", view.search.open, "Find buildings (F)"):
        view.toggle_search()
