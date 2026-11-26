"""The station window, opened by right clicking a station: its level and reach, what it holds
and wants, what it reaches, as placing it shows (railroads.ui.station_preview), the trains that
call there, buttons to raise its level or add storage, and the facilities beside it with what
each costs a month and buttons to put up the others."""

from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.facilities import (
    BY_SHED,
    DELIVERED_SHARE,
    PAYS_MORE,
    PLACED,
    SAND_TOWER_PICKUP,
    SHED_LEVELS,
    SHED_STALLS,
    WAREHOUSE_ROOM,
    Facility,
    shed_stalls,
)
from railroads.sim.facilities import LEVEL as FACILITY_LEVEL
from railroads.sim.facilities import REACH as FACILITY_REACH
from railroads.sim.facilities import USES as FACILITY_USES
from railroads.sim.finance import STATION_PRICE
from railroads.sim.grids import Cell
from railroads.sim.railway import MAX_LEVEL, PLATFORM_REACH, REACH_LEVEL, Railway
from railroads.sim.station_goods import STORAGE_STEP, WAITING_STEP
from railroads.sim.station_trade import StationPreview
from railroads.sim.trains import amount_of
from railroads.ui.demand_view import draw_demand
from railroads.ui.ornate import begin_ornate, end_ornate, hover_card, keep_in_front
from railroads.ui.station_preview import draw_preview_body
from railroads.ui.stations_window import look_at_button
from railroads.ui.theme import (
    BRASS,
    GONE,
    SIDE_WINDOW_POS,
    colored,
    goods_inline,
    rgba_u32,
    tip,
    town_tint,
)
from railroads.words import dollars

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
        preview: Callable[[Cell], StationPreview] | None = None,
        add_platform: Callable[[Cell], None] = lambda _: None,
        add_facility: Callable[[Cell, Facility], None] = lambda _, __: None,
        rebuilt: Callable[[set[Cell]], None] = lambda _: None,
    ) -> None:
        """`preview` says what a station reaches (StationTrade.preview);
        `add_platform` lets the player choose where a platform goes, which raising a station to
        a level that allows one does at once, `add_facility` where a facility goes, and `rebuilt`
        redraws the cells where what is built changed, as raising an engine shed does."""
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

            self._draw_body(railway, station, look_at, raised, preview)
            _draw_facilities(railway, station, add_facility, rebuilt)
            left = railway.platforms_left(station)
            if left:
                label = f"{fa.ICON_FA_PLUS}  Add a platform beside it ({left} to add)"
                if imgui.button(label):
                    add_platform(station)
                tip(
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
        preview: Callable[[Cell], StationPreview] | None,
    ) -> None:
        level = railway.level(station)
        delivery, collection = railway.reach(station)
        imgui.align_text_to_frame_padding()
        colored(BRASS, f"Level {level} of {MAX_LEVEL}")
        imgui.same_line()
        imgui.text_disabled(f"hands out within {delivery} cells, gathers within {collection}")
        imgui.same_line()
        look_at_button(station, look_at)
        if warning := station_warning(railway, station):
            imgui.push_text_wrap_pos(0.0)
            colored(GONE, f"{fa.ICON_FA_EXCLAMATION_TRIANGLE}  {warning}")
            imgui.pop_text_wrap_pos()
        imgui.separator()
        draw_station_summary(railway, station)
        if preview is not None:
            imgui.separator()
            draw_preview_body(preview(station), built=True)
        imgui.separator()
        titles = railway.titles()
        calling = [
            titles[train.name]
            for train in railway.trains
            if any(s.cell == station for s in train.stops)
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
            detail = (
                f"Hands out within {delivery + 1} cells, gathers within {collection + 1}"
                if level < REACH_LEVEL
                else f"A platform more, up to {PLATFORM_REACH} cells away, handing out and "
                f"gathering as far round it as the station does"
            )
            affordable = railway.finances.can_afford(price)
            if _priced_button(label, affordable, detail) and railway.upgrade(station):
                upgraded(station)
        storage_price = railway.storage_price(station)
        storage = f"{fa.ICON_FA_PLUS}  Storage  {dollars(storage_price)}"
        detail = f"+{STORAGE_STEP} of each good, +{WAITING_STEP} passengers and mail"
        if _priced_button(storage, railway.finances.can_afford(storage_price), detail):
            railway.add_storage(station)


def draw_station_summary(railway: Railway, station: Cell) -> None:
    """What the station holds and how much of it it can, who waits there and what its town
    still wants: in the station window, and on the map once the mouse rests on the station."""
    goods = railway.station_goods(station)
    imgui.text_disabled("Stockpile")
    imgui.same_line(LABEL_WIDTH)
    goods_inline(goods.freight(), "empty")
    imgui.dummy((LABEL_WIDTH, 0))
    imgui.same_line(LABEL_WIDTH)
    imgui.text_disabled(f"holds up to {goods.limit:,.0f} of each good")
    imgui.text_disabled("Waiting")
    imgui.same_line(LABEL_WIDTH)
    goods_inline(goods.waiting(), "nobody")
    imgui.dummy((LABEL_WIDTH, 0))
    imgui.same_line(LABEL_WIDTH)
    imgui.text_disabled(f"up to {goods.waiting_limit:,.0f} passengers and as much mail")
    imgui.text_disabled("Demand left")
    imgui.same_line(LABEL_WIDTH)
    draw_demand(railway.station_demand(station))


def draw_station_card(railway: Railway, station: Cell, top_left: tuple[float, float]) -> None:
    """The station's name and summary in a card below the map's status strip."""
    with hover_card("##station card", top_left, WINDOW_WIDTH) as shown:
        if shown:
            colored(BRASS, railway.name_of(station))
            imgui.separator()
            draw_station_summary(railway, station)


def station_warning(railway: Railway, station: Cell) -> str:
    """What the station window warns of first: that no track joins the station to another, so
    it does nothing; empty when it is joined."""
    return "" if railway.joined(station) else UNJOINED


FACILITY_EFFECTS = {
    Facility.ENGINE_SHED: (
        f"trains short of a full load wait in it, {SHED_STALLS} for each of its levels"
    ),
    Facility.SAND_TOWER: f"trains leaving here pick up speed {1 / SAND_TOWER_PICKUP:.0f} times as fast",
    Facility.POST_OFFICE: f"mail delivered here pays {PAYS_MORE[Facility.POST_OFFICE][1] - 1:.0%} more",
    Facility.REFRESHMENT_ROOM: f"passengers arriving here pay {PAYS_MORE[Facility.REFRESHMENT_ROOM][1] - 1:.0%} more",
    Facility.MARKET_HALL: f"food delivered here pays {PAYS_MORE[Facility.MARKET_HALL][1] - 1:.0%} more",
    Facility.CATTLE_PENS: f"livestock delivered here pays {PAYS_MORE[Facility.CATTLE_PENS][1] - 1:.0%} more",
    Facility.WAREHOUSE: f"+{WAREHOUSE_ROOM:.0f} room for each good",
}


def _draw_facilities(
    railway: Railway,
    station: Cell,
    add: Callable[[Cell, Facility], None],
    rebuilt: Callable[[set[Cell]], None] = lambda _: None,
) -> None:
    """A line for each facility at the station with what its next month costs, those in the
    station itself with a button to close them, and a button for each of the others: those
    standing beside the station let the player choose their cell on the map, the rest go up
    in the station at once."""
    imgui.separator()
    imgui.text_disabled("Facilities")
    tip(
        "None costs anything to put up, but every month each uses a good and pays the market "
        f"price for it: {DELIVERED_SHARE:.0%} of that for what the station holds, gathered or "
        f"brought. An engine shed fills a cell up to {FACILITY_REACH} cells from the station with "
        "track of its own; the "
        "others go in the station itself"
    )
    built = railway.facilities_at(station)
    level = railway.level(station)
    for facility, (good, amount) in FACILITY_USES.items():
        name = facility.value.capitalize()
        cost = railway.facility_cost(station, facility)
        uses = f"Uses {amount_of(good, amount)} a month; {FACILITY_EFFECTS[facility]}"
        if facility in built:
            imgui.align_text_to_frame_padding()
            imgui.text(f"{name}: {dollars(cost)} a month")
            tip(uses)
            if facility == Facility.ENGINE_SHED:
                _draw_shed_level(railway, station, rebuilt)
            if facility not in PLACED:
                imgui.same_line()
                if imgui.small_button(f"Close##{facility.name}"):
                    railway.close_facility(station, facility)
                    if facility in BY_SHED:
                        rebuilt({railway.constructions.shed_of(station) or station})
                tip("Closes it, so it costs nothing more")
            continue
        needed = FACILITY_LEVEL[facility]
        no_shed = facility in BY_SHED and Facility.ENGINE_SHED not in built
        label = f"{fa.ICON_FA_PLUS}  {name}  {dollars(cost)} a month"
        imgui.begin_disabled(level < needed or no_shed)
        clicked = imgui.button(label, (-1, 0))
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            lines = [uses]
            if facility in PLACED:
                lines.append(
                    f"Click, then the cell up to {FACILITY_REACH} cells from the station it goes in"
                )
            if level < needed:
                lines.append(f"Raise the station to level {needed} first")
            if no_shed:
                lines.append("Put up an engine shed first: it stands by its lead track")
            imgui.set_tooltip("\n".join(lines))
        if clicked:
            if facility in PLACED:
                add(station, facility)
            elif railway.put_up(station, facility) and facility in BY_SHED:
                rebuilt({railway.constructions.shed_of(station) or station})


def _draw_shed_level(railway: Railway, station: Cell, rebuilt: Callable[[set[Cell]], None]) -> None:
    """The station's engine shed's level and how many trains it holds, and a button raising it a
    level, adding stalls to its roundhouse, while it can go higher."""
    shed = railway.constructions.shed_of(station)
    if shed is None:
        return
    level = railway.shed_level(shed)
    imgui.same_line()
    imgui.text_disabled(f"level {level} of {SHED_LEVELS}, {shed_stalls(level)} trains")
    price = railway.shed_upgrade_price(shed)
    if price is None:
        return
    label = f"{fa.ICON_FA_ARROW_UP}  Engine shed level {level + 1}  {dollars(price)}"
    detail = f"Adds {SHED_STALLS} stalls to its roundhouse, for {shed_stalls(level + 1)} trains"
    if _priced_button(label, railway.finances.can_afford(price), detail) and railway.upgrade_shed(
        shed
    ):
        rebuilt({shed})


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
