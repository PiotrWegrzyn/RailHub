"""The stations window: every station's stockpile of spare goods waiting for trains, and the
goods trains brought waiting to be handed out."""

from collections.abc import Callable
from dataclasses import dataclass, field

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.grids import Cell
from railroads.sim.railway import Railway
from railroads.sim.station_goods import StationGoods
from railroads.ui.demand_view import draw_demand
from railroads.ui.ornate import begin_ornate, end_ornate
from railroads.ui.theme import (
    BRASS_DARK,
    CARDS_WIDTH,
    LIST_TABLE_FLAGS,
    goods_inline,
    tip,
)
from railroads.ui.trains_window import WINDOW_SIZE as TRAINS_SIZE
from railroads.ui.trains_window import WINDOW_TOP as TRAINS_TOP

WINDOW_SIZE = (860.0, 180.0)
ROW_HEIGHT = 28.0
GAP = 12.0


def stock_bar(goods: StationGoods, size: tuple[float, float]) -> None:
    """A stockpile bar filled as far as the fullest kind of freight, against the most of each
    the station holds."""
    good, held = goods.fullest
    label = f"{held:.0f}/{goods.limit:.0f}" + (f" {good.value}" if good else "")
    imgui.push_style_color(imgui.Col_.plot_histogram, imgui.ImVec4(*BRASS_DARK))
    imgui.progress_bar(min(1.0, held / goods.limit), size, label)
    imgui.pop_style_color()


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
            (display.x - WINDOW_SIZE[0] - CARDS_WIDTH - 16, TRAINS_TOP + TRAINS_SIZE[1] + GAP),
            imgui.Cond_.first_use_ever.value,
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
        imgui.table_setup_column("Waiting for trains", imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_setup_column("Brought by trains", imgui.TableColumnFlags_.width_stretch.value)
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
        goods_inline(goods.stockpile, "empty")
        bottom = max(bottom, imgui.get_item_rect_max().y)
        imgui.table_next_column()
        goods_inline(goods.brought, "nothing")
        bottom = max(bottom, imgui.get_item_rect_max().y)
        imgui.table_next_column()
        draw_demand(railway.station_demand(station))
        bottom = max(bottom, imgui.get_item_rect_max().y)
        imgui.table_next_column()
        look_at_button(station, look_at)
        imgui.pop_id()
        return bottom - top
