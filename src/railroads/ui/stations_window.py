"""The stations window: every station's stockpile of spare goods waiting for trains, and the
goods trains brought waiting to be handed out."""

from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.demand import StationDemand, price_of
from railroads.sim.finance import dollars
from railroads.sim.goods import FITS, Need
from railroads.sim.grids import Cell
from railroads.sim.railway import Railway
from railroads.ui.theme import (
    BRASS,
    BRASS_DARK,
    CARDS_WIDTH,
    GONE,
    GROWN,
    good_icon,
    goods_inline,
    next_good_slot,
    rows_that_fit,
)
from railroads.ui.trains_window import WINDOW_SIZE as TRAINS_SIZE
from railroads.ui.trains_window import WINDOW_TOP as TRAINS_TOP

WINDOW_SIZE = (860.0, 180.0)
GAP = 12.0


@dataclass
class StationsWindow:
    open: bool = False

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
        expanded, self.open = imgui.begin(f"{fa.ICON_FA_BUILDING}  Stations###stations", self.open)
        if expanded:
            stations = railway.stations()
            if not stations:
                imgui.text_disabled(
                    "No stations yet. Pick Station in the build bar and click the map."
                )
            else:
                self._draw_table(railway, stations, look_at, open_station)
        imgui.end()

    @staticmethod
    def _draw_table(
        railway: Railway,
        stations: list[Cell],
        look_at: Callable[[Cell], None],
        open_station: Callable[[Cell], None],
    ) -> None:
        flags = (
            imgui.TableFlags_.row_bg.value
            | imgui.TableFlags_.borders_inner_h.value
            | imgui.TableFlags_.scroll_y.value
            | imgui.TableFlags_.sizing_fixed_fit.value
        )
        if not imgui.begin_table("stations", 6, flags):
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
            goods = railway.station_goods(station)
            imgui.push_id(str(station))
            imgui.table_next_row(0, 28)
            imgui.table_next_column()
            imgui.align_text_to_frame_padding()
            if imgui.selectable(railway.name_of(station), False)[0]:
                open_station(station)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Open this station's details")
            imgui.table_next_column()
            imgui.push_style_color(imgui.Col_.plot_histogram, imgui.ImVec4(*BRASS_DARK))
            imgui.progress_bar(
                goods.stocked / goods.limit, (110, 0), f"{goods.stocked:.0f}/{goods.limit:.0f}"
            )
            imgui.pop_style_color()
            imgui.table_next_column()
            goods_inline(goods.stockpile, "empty")
            imgui.table_next_column()
            goods_inline(goods.brought, "nothing")
            imgui.table_next_column()
            draw_demand(railway.station_demand(station))
            imgui.table_next_column()
            if imgui.small_button(fa.ICON_FA_CROSSHAIRS):
                look_at(station)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Look at this station")
            imgui.pop_id()
        imgui.end_table()


def draw_demand(demand: StationDemand) -> None:
    """Each need around the station, with the share of its demand left: the share of its price
    a unit delivered there for it earns."""
    wanted = [need for need in Need if demand.capacity.get(need, 0) > 0]
    if not wanted:
        imgui.text_disabled("none")
        return
    left, per_row = rows_that_fit()
    for index, need in enumerate(wanted):
        record = demand.record(need)
        next_good_slot(index, left, per_row)
        imgui.align_text_to_frame_padding()
        good_icon(need, framed=True)
        imgui.same_line(0, 3)
        color = GONE if record.share < 0.25 else BRASS if record.share < 0.75 else GROWN
        imgui.text_colored(imgui.ImVec4(*color), f"{record.share:.0%}")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text(f"{need.value}: {record.left:.1f} of {record.capacity:.1f} demand left")
            price = price_of(FITS[need][0]) * record.share
            imgui.text(f"A unit delivered for it now earns {dollars(price)}")
            imgui.text_disabled(f"Met by {', '.join(good.value for good in FITS[need])}")
            last = demand.last_month.get(need)
            if last is not None:
                imgui.text_disabled(
                    f"Last month ended with {last.left:.1f} of {last.capacity:.1f} left"
                )
            imgui.end_tooltip()
