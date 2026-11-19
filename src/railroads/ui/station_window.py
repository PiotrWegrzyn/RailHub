"""The station window, opened by right clicking a station: its level and reach, what it holds
and wants, which of the goods made around it or held in it are kept for trains rather than handed
out there, the trains that call there, and buttons to raise its level or add storage."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.facilities import (
    ENGINE_SHED_SPEED,
    PAYS_MORE,
    SAND_TOWER_PICKUP,
    WAREHOUSE_ROOM,
    Facility,
)
from railroads.sim.facilities import PRICES as FACILITY_PRICES
from railroads.sim.facilities import USES as FACILITY_USES
from railroads.sim.finance import STATION_PRICE, dollars
from railroads.sim.goods import TRAVELLERS, Good
from railroads.sim.grids import Cell
from railroads.sim.railway import MAX_LEVEL, Railway
from railroads.sim.station_goods import STORAGE_STEP, WAITING_STEP, StationGoods
from railroads.ui.demand_view import draw_demand
from railroads.ui.ornate import begin_ornate, end_ornate, keep_in_front
from railroads.ui.stations_window import look_at_button, stock_bar
from railroads.ui.theme import (
    BRASS,
    GONE,
    SIDE_WINDOW_POS,
    good_icon,
    goods_inline,
    next_good_slot,
    rgba_u32,
    rows_that_fit,
    tip,
    town_tint,
)

WINDOW_WIDTH = 460.0
UNJOINED = "Not joined by track to another station: it gathers and hands out nothing"
LABEL_WIDTH = 140.0


@dataclass
class StationWindow:
    # The station shown, if the window is open.
    station: Cell | None = None

    def draw(
        self,
        railway: Railway,
        look_at: Callable[[Cell], None],
        upgraded: Callable[[Cell], None],
        move: Callable[[Cell], None] = lambda _: None,
        supplies: Callable[[Cell], Mapping[Good, float]] = lambda _: {},
        add_platform: Callable[[Cell], None] = lambda _: None,
    ) -> None:
        """`supplies` says what the buildings around a station make (StationTrade.supplies);
        `add_platform` lets the player choose where a platform goes, which raising a station to
        a level that allows one does at once."""
        station = self.station
        if station is None:
            return
        if station not in railway.stations():
            self.station = None
            return
        imgui.set_next_window_pos(SIDE_WINDOW_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size_constraints((WINDOW_WIDTH, 0), (WINDOW_WIDTH, 2000))
        name = railway.name_of(station)
        flags = imgui.WindowFlags_.always_auto_resize.value
        expanded, still_open = begin_ornate(f"{fa.ICON_FA_BUILDING}  {name}###station", flags)
        keep_in_front()
        if expanded:

            def raised(cell: Cell) -> None:
                upgraded(cell)
                if railway.platforms_left(cell):
                    add_platform(cell)

            self._draw_body(railway, station, look_at, raised, supplies)
            left = railway.platforms_left(station)
            if left:
                label = f"{fa.ICON_FA_PLUS}  Add a platform beside it ({left} to add)"
                if imgui.button(label):
                    add_platform(station)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Its own way through lets lines come in from another side;\n"
                        "trains cross to it inside the station"
                    )
            label = (
                f"{fa.ICON_FA_ARROWS_ALT}  Move it, keeping its trains  {dollars(STATION_PRICE)}+"
            )
            if _priced_button(label, railway.finances.can_afford(STATION_PRICE)):
                move(station)
        end_ornate()
        if not still_open:
            self.station = None

    @staticmethod
    def _draw_body(
        railway: Railway,
        station: Cell,
        look_at: Callable[[Cell], None],
        upgraded: Callable[[Cell], None],
        supplies: Callable[[Cell], Mapping[Good, float]],
    ) -> None:
        level = railway.level(station)
        delivery, collection = railway.reach(station)
        goods = railway.station_goods(station)
        imgui.align_text_to_frame_padding()
        imgui.text_colored(imgui.ImVec4(*BRASS), f"Level {level} of {MAX_LEVEL}")
        imgui.same_line()
        imgui.text_disabled(f"hands out within {delivery} cells, gathers within {collection}")
        imgui.same_line()
        look_at_button(station, look_at)
        if warning := station_warning(railway, station):
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(imgui.ImVec4(*GONE), f"{fa.ICON_FA_EXCLAMATION_TRIANGLE}  {warning}")
            imgui.pop_text_wrap_pos()
        imgui.separator()
        imgui.text_disabled("Stockpile")
        imgui.same_line(LABEL_WIDTH)
        stock_bar(goods, (-1, 0))
        imgui.dummy((LABEL_WIDTH, 0))
        imgui.same_line(LABEL_WIDTH)
        goods_inline(goods.freight(), "empty")
        imgui.text_disabled("Waiting")
        imgui.same_line(LABEL_WIDTH)
        goods_inline(goods.waiting(), "nobody")
        imgui.text_disabled("Demand left")
        imgui.same_line(LABEL_WIDTH)
        draw_demand(railway.station_demand(station))
        imgui.text_disabled("Not for local use")
        imgui.same_line(LABEL_WIDTH)
        _draw_kept(goods, supplies(station))
        imgui.separator()
        calling = [
            train.name for train in railway.trains if any(s.cell == station for s in train.stops)
        ]
        imgui.text_disabled("Trains calling")
        imgui.same_line(LABEL_WIDTH)
        imgui.text(", ".join(calling) if calling else "none")
        _draw_town(railway, station)
        imgui.separator()
        price = railway.upgrade_price(station)
        if price is None:
            imgui.text_disabled("At the top level")
        else:
            label = f"{fa.ICON_FA_ARROW_UP}  Level {level + 1}  {dollars(price)}"
            detail = f"Hands out within {delivery + 1} cells, gathers within {collection + 1}"
            affordable = railway.finances.can_afford(price)
            if _priced_button(label, affordable, detail) and railway.upgrade(station):
                upgraded(station)
        storage_price = railway.storage_price(station)
        storage = f"{fa.ICON_FA_PLUS}  Storage  {dollars(storage_price)}"
        detail = f"+{STORAGE_STEP} of each good, +{WAITING_STEP} passengers and mail"
        if _priced_button(storage, railway.finances.can_afford(storage_price), detail):
            railway.add_storage(station)
        _draw_facilities(railway, station)


def station_warning(railway: Railway, station: Cell) -> str:
    """What the station window warns of first: that no track joins the station to another, so
    it does nothing; empty when it is joined."""
    return "" if railway.joined(station) else UNJOINED


FACILITY_EFFECTS = {
    Facility.ENGINE_SHED: f"trains leaving here run {ENGINE_SHED_SPEED - 1:.0%} faster",
    Facility.SAND_TOWER: f"trains leaving here pick up speed {1 / SAND_TOWER_PICKUP:.0f} times as fast",
    Facility.POST_OFFICE: f"mail delivered here pays {PAYS_MORE[Facility.POST_OFFICE][1] - 1:.0%} more",
    Facility.REFRESHMENT_ROOM: f"passengers arriving here pay {PAYS_MORE[Facility.REFRESHMENT_ROOM][1] - 1:.0%} more",
    Facility.MARKET_HALL: f"food delivered here pays {PAYS_MORE[Facility.MARKET_HALL][1] - 1:.0%} more",
    Facility.CATTLE_PENS: f"livestock delivered here pays {PAYS_MORE[Facility.CATTLE_PENS][1] - 1:.0%} more",
    Facility.WAREHOUSE: f"+{WAREHOUSE_ROOM:.0f} room for each good",
}


def _draw_facilities(railway: Railway, station: Cell) -> None:
    """A line for each facility: whether it works this month, or a button to put it up."""
    built, working = railway.facilities_at(station), railway.working_at(station)
    for facility, (good, amount) in FACILITY_USES.items():
        effect = FACILITY_EFFECTS[facility]
        if facility in built:
            state = "working" if facility in working else f"idle: needs {amount:.0f} {good.value}"
            imgui.text(f"{facility.value.capitalize()}: {state}")
            tip(f"Uses {amount:.0f} {good.value} a month; {effect}")
            continue
        price = FACILITY_PRICES[facility]
        label = f"{fa.ICON_FA_PLUS}  {facility.value.capitalize()}  {dollars(price)}"
        detail = f"Uses {amount:.0f} {good.value} a month; {effect}"
        if _priced_button(label, railway.finances.can_afford(price), detail):
            railway.build_facility(station, facility)


def _draw_kept(goods: StationGoods, made: Mapping[Good, float]) -> None:
    """A tick for each good made around the station or held in it, and each already kept, to
    keep it for trains instead of handing it to the buildings around the station."""
    held = {g for g in {*goods.stockpile, *goods.brought} if goods.held(g) > 0}
    shown = sorted(
        set(made) | held - set(TRAVELLERS) | goods.kept,
        key=lambda g: g.value,
    )
    if not shown:
        imgui.text_disabled("nothing is made around it or held here")
        return
    left, per_row = rows_that_fit()
    for index, good in enumerate(shown):
        next_good_slot(index, left, per_row)
        kept = good in goods.kept
        changed, kept = imgui.checkbox(f"##keep {good.value}", kept)
        if changed:
            goods.keep_for_trains(good, kept)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Keep {good.value} for trains: the station won't hand it to the buildings "
                "around it"
            )
        imgui.same_line(0, 3)
        good_icon(good, framed=True)


def _priced_button(label: str, affordable: bool, detail: str = "") -> bool:
    """A button as wide as the window naming what it buys and its price, with what it does in
    its tooltip, so it fits however narrow the window is."""
    imgui.begin_disabled(not affordable)
    clicked = imgui.button(label, (-1, 0))
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        lines = [detail] if detail else []
        if not affordable:
            lines.append("Not enough money")
        if lines:
            imgui.set_tooltip("\n".join(lines))
    return clicked


def _draw_town(railway: Railway, station: Cell) -> None:
    """The town the station belongs to, in its colour on the Towns lens, and the stations of the
    player's in the same town, which passengers and mail don't travel between."""
    imgui.text_disabled("Town")
    imgui.same_line(LABEL_WIDTH)
    town = railway.town_of(station)
    if not town:
        imgui.text("none: nobody here sends passengers")
        return
    low = imgui.get_cursor_screen_pos()
    side = imgui.get_text_line_height()
    imgui.get_window_draw_list().add_rect_filled(
        low, (low.x + side, low.y + side), rgba_u32(town_tint(town, 1.0)), 3.0
    )
    imgui.dummy((side, side))
    imgui.same_line()
    others = [
        railway.name_of(other)
        for other in railway.stations()
        if other != station and railway.town_of(other) == town
    ]
    imgui.text(f"shared with {', '.join(others)}" if others else "its own")
    tip(
        "Passengers and mail only board for a stop in another town, so two stations in one "
        "town can't carry them to each other"
    )
