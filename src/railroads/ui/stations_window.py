"""The stations window: every station's stockpile of spare goods waiting for trains, and the
goods trains brought waiting to be handed out."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.goods import Good
from railroads.sim.grids import Cell
from railroads.sim.railway import Railway
from railroads.sim.station_goods import StationGoods
from railroads.ui.demand_view import draw_demand
from railroads.ui.ornate import begin_ornate, end_ornate
from railroads.ui.theme import (
    BRASS_DARK,
    LIST_TABLE_FLAGS,
    SIDE_WINDOW_TOP,
    draw_good_icon,
    left_of_cards,
    tip,
)

WINDOW_SIZE = (860.0, 180.0)
ROW_HEIGHT = 28.0
# How much of the screen's bottom the window leaves free for the build bar.
BUILD_BAR_ROOM = 80.0
# A good's icon in the Goods and Waiting columns, as small as those of a schedule's cargo, the
# room between its icon and its amount, between two goods, and between two lines of them.
GOOD_ICON = 18.0
AMOUNT_GAP = 1.0
GOOD_GAP = 6.0
LINE_GAP = 2.0
# The least of a good the station holds for it to show.
SHOWN_AMOUNT = 0.5


def window_place(width: float, height: float) -> tuple[float, float]:
    """Where the window first opens on a screen this big: left of the news cards, but never
    past the screen's left edge or down over the build bar."""
    x = left_of_cards(width, WINDOW_SIZE[0])
    y = min(SIDE_WINDOW_TOP, height - WINDOW_SIZE[1] - BUILD_BAR_ROOM)
    return x, max(8.0, y)


def stock_bar(goods: StationGoods, size: tuple[float, float]) -> None:
    """A stockpile bar filled as far as the fullest kind of freight, against the most of each
    the station holds."""
    good, held = goods.fullest
    label = f"{held:.0f}/{goods.limit:.0f}" + (f" {good.value}" if good else "")
    imgui.push_style_color(imgui.Col_.plot_histogram, imgui.ImVec4(*BRASS_DARK))
    imgui.progress_bar(min(1.0, held / goods.limit), size, label)
    imgui.pop_style_color()


def counted(goods: Mapping[Good, float]) -> list[tuple[Good, int]]:
    """Each good the station holds enough of to show, in whole units, the most first."""
    shown = [(good, round(amount)) for good, amount in goods.items() if amount >= SHOWN_AMOUNT]
    return sorted(shown, key=lambda pair: (-pair[1], pair[0].value))


def places(widths: list[float], room: float) -> list[tuple[float, int]]:
    """Where each of a run of goods this wide goes in a column this wide: how far from its left
    edge and on which line, a new line begun when the next would run past the edge."""
    placed: list[tuple[float, int]] = []
    x, line = 0.0, 0
    for width in widths:
        if x > 0 and x + width > room:
            x, line = 0.0, line + 1
        placed.append((x, line))
        x += width + GOOD_GAP
    return placed


def draw_good_counts(goods: Mapping[Good, float], empty: str, hint: str) -> None:
    """Goods each as a small icon and how much of it, as a schedule's cargo shows, on as many
    lines as the column needs; `hint` says where the amounts are counted."""
    shown = counted(goods)
    if not shown:
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(empty)
        return
    left, top = imgui.get_cursor_pos()
    frame = imgui.get_frame_height()
    text = imgui.get_text_line_height()
    widths = [GOOD_ICON + AMOUNT_GAP + imgui.calc_text_size(f"{n}").x for _, n in shown]
    room = imgui.get_content_region_avail().x
    draw_list = imgui.get_window_draw_list()
    for (good, count), (x, line) in zip(shown, places(widths, room), strict=True):
        y = top + line * (max(frame, GOOD_ICON) + LINE_GAP)
        imgui.set_cursor_pos((left + x + GOOD_ICON + AMOUNT_GAP, y + (frame - text) / 2))
        imgui.text_disabled(f"{count}")
        imgui.set_cursor_pos((left + x, y + (frame - GOOD_ICON) / 2))
        imgui.dummy((GOOD_ICON, GOOD_ICON))
        draw_good_icon(draw_list, good, imgui.get_item_rect_min(), GOOD_ICON)
        tip(f"{good.value}: {count} {hint}")


def look_at_button(station: Cell, look_at: Callable[[Cell], None]) -> None:
    """A small crosshair button that turns the map to the station."""
    if imgui.small_button(fa.ICON_FA_CROSSHAIRS):
        look_at(station)
    tip("Look at this station")


@dataclass
class StationsWindow:
    open: bool = False
    # How tall each station's row came out when last drawn, so a row out of sight takes its room
    # without being drawn.
    _heights: dict[Cell, float] = field(default_factory=dict)

    def draw(
        self,
        railway: Railway,
        look_at: Callable[[Cell], None],
        open_station: Callable[[Cell], None],
    ) -> None:
        if not self.open:
            return
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(
            window_place(display.x, display.y), imgui.Cond_.first_use_ever.value
        )
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, self.open = begin_ornate(f"{fa.ICON_FA_BUILDING}  Stations###stations")
        if expanded:
            stations = railway.stations()
            if not stations:
                imgui.text_disabled(
                    "No stations yet. Pick Station in the build bar and click the map."
                )
            else:
                self._draw_table(railway, stations, look_at, open_station)
        end_ornate()

    def _draw_table(
        self,
        railway: Railway,
        stations: list[Cell],
        look_at: Callable[[Cell], None],
        open_station: Callable[[Cell], None],
    ) -> None:
        if not imgui.begin_table("stations", 6, LIST_TABLE_FLAGS):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_setup_column("Station")
        imgui.table_setup_column("Stockpile")
        imgui.table_setup_column("Goods", imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_setup_column("Waiting", imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_setup_column("Demand left", imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_setup_column("")
        imgui.table_headers_row()
        for station in stations:
            height = self._heights.get(station, ROW_HEIGHT)
            imgui.table_next_row(0, height)
            imgui.table_next_column()
            if imgui.is_rect_visible((1.0, height)):
                self._heights[station] = self._draw_row(railway, station, look_at, open_station)
        imgui.end_table()

    @staticmethod
    def _draw_row(
        railway: Railway,
        station: Cell,
        look_at: Callable[[Cell], None],
        open_station: Callable[[Cell], None],
    ) -> float:
        """Draws a station's row; returns how tall it came out."""
        top = imgui.get_cursor_screen_pos().y
        bottom = top + ROW_HEIGHT
        goods = railway.station_goods(station)
        imgui.push_id(str(station))
        imgui.align_text_to_frame_padding()
        if imgui.selectable(railway.name_of(station), False)[0]:
            open_station(station)
        tip("Open this station's details")
        imgui.table_next_column()
        stock_bar(goods, (110, 0))
        imgui.table_next_column()
        draw_good_counts(goods.freight(), "empty", "in the stockpile")
        bottom = max(bottom, imgui.get_item_rect_max().y)
        imgui.table_next_column()
        draw_good_counts(goods.waiting(), "nobody", "waiting for a train")
        bottom = max(bottom, imgui.get_item_rect_max().y)
        imgui.table_next_column()
        draw_demand(railway.station_demand(station))
        bottom = max(bottom, imgui.get_item_rect_max().y)
        imgui.table_next_column()
        look_at_button(station, look_at)
        imgui.pop_id()
        return bottom - top
