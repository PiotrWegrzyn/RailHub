"""The station window, opened by right clicking a station: its level and reach, what it holds
and wants, which of the goods made around it are kept for trains rather than handed out there,
the trains that call there, and buttons to raise its level or add storage."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.finance import STATION_PRICE, STORAGE_PRICE, dollars
from railroads.sim.goods import TRAVELLERS, Good
from railroads.sim.grids import Cell
from railroads.sim.railway import MAX_LEVEL, Railway
from railroads.sim.station_goods import STORAGE_STEP, WAITING_STEP, StationGoods
from railroads.ui.demand_view import draw_demand
from railroads.ui.stations_window import stock_bar
from railroads.ui.theme import (
    BRASS,
    BRASS_DARK,
    good_icon,
    goods_inline,
    next_good_slot,
    rows_that_fit,
)

WINDOW_POS = (16.0, 96.0)
WINDOW_WIDTH = 460.0
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
    ) -> None:
        """`supplies` says what the buildings around a station make (StationTrade.supplies)."""
        station = self.station
        if station is None:
            return
        if station not in railway.stations():
            self.station = None
            return
        imgui.set_next_window_pos(WINDOW_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size_constraints((WINDOW_WIDTH, 0), (WINDOW_WIDTH, 2000))
        name = railway.name_of(station)
        flags = imgui.WindowFlags_.always_auto_resize.value
        expanded, still_open = imgui.begin(f"{fa.ICON_FA_BUILDING}  {name}###station", True, flags)
        if expanded:
            self._draw_body(railway, station, look_at, upgraded, supplies)
            label = (
                f"{fa.ICON_FA_ARROWS_ALT}  Move it, keeping its trains  {dollars(STATION_PRICE)}+"
            )
            if _priced_button(label, railway.finances.can_afford(STATION_PRICE)):
                move(station)
        imgui.end()
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
        if imgui.small_button(fa.ICON_FA_CROSSHAIRS):
            look_at(station)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Look at this station")
        imgui.separator()
        imgui.text_disabled("Stockpile")
        imgui.same_line(LABEL_WIDTH)
        imgui.push_style_color(imgui.Col_.plot_histogram, imgui.ImVec4(*BRASS_DARK))
        imgui.progress_bar(*stock_bar(goods, (-1, 0)))
        imgui.pop_style_color()
        freight = {g: a for g, a in goods.stockpile.items() if g not in TRAVELLERS}
        waiting = {g: a for g, a in goods.stockpile.items() if g in TRAVELLERS}
        for label, held, empty in (
            ("Freight", freight, "none"),
            ("Waiting", waiting, "nobody"),
            ("Brought by trains", goods.brought, "nothing"),
        ):
            imgui.text_disabled(label)
            imgui.same_line(LABEL_WIDTH)
            goods_inline(held, empty)
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
        imgui.separator()
        price = railway.upgrade_price(station)
        if price is None:
            imgui.text_disabled("At the top level")
        else:
            label = f"{fa.ICON_FA_ARROW_UP}  Level {level + 1}: hands out within {delivery + 1}, gathers within {collection + 1}  {dollars(price)}"
            if _priced_button(label, railway.finances.can_afford(price)) and railway.upgrade(
                station
            ):
                upgraded(station)
        storage = f"{fa.ICON_FA_PLUS}  Storage +{STORAGE_STEP} of each good, +{WAITING_STEP} passengers and mail  {dollars(STORAGE_PRICE)}"
        if _priced_button(storage, railway.finances.can_afford(STORAGE_PRICE)):
            railway.add_storage(station)


def _draw_kept(goods: StationGoods, made: Mapping[Good, float]) -> None:
    """A tick for each good made around the station, and each already kept, to keep it for
    trains instead of handing it to the buildings around the station."""
    shown = sorted(
        {g for g, amount in made.items() if amount > 0 and g not in TRAVELLERS} | goods.kept,
        key=lambda g: g.value,
    )
    if not shown:
        imgui.text_disabled("nothing is made around it")
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


def _priced_button(label: str, affordable: bool) -> bool:
    imgui.begin_disabled(not affordable)
    clicked = imgui.button(label, (-1, 0))
    imgui.end_disabled()
    if not affordable and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip("Not enough money")
    return clicked
