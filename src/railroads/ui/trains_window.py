"""The trains window: every train in a list on the left; the chosen one's cars and route on the
right, a card for each stop with the cars it pulls on to the next, the goods made or waiting there
to click into cars, those the next stop wants outlined, and what the next stop wants; and beside the
cards a map to add stops on, which marks where a hovered good is wanted (railroads.ui.route_map)."""

import time
from dataclasses import dataclass, field

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.demand import StationDemand
from railroads.sim.finance import TRAIN_RESALE, dollars
from railroads.sim.goods import Good
from railroads.sim.railway import Railway
from railroads.sim.trains import ENGINES, Stop, Train
from railroads.ui.demand_view import draw_demand
from railroads.ui.route_map import GroundPicture, RouteMap, add_stop, can_stop_at
from railroads.ui.station_offers import Offer, Supplies, goods_to_load
from railroads.ui.theme import (
    BRASS,
    CARDS_WIDTH,
    OFFER_SIZE,
    TEXT_DIM,
    car_chip,
    color_u32,
    danger_button,
    empty_car_slot,
    offer_chip,
)

WINDOW_SIZE = (980.0, 500.0)
# The window can't be made smaller than this, so the map beside the stop cards keeps its room.
SMALLEST_WINDOW = (900.0, 440.0)
WINDOW_TOP = 96.0
LIST_WIDTH = 190.0
# The share of the space beside the list of trains the map takes.
MAP_SHARE = 0.45
# Where a stop card's rows start, after their labels.
CARD_LABEL = 130.0
# How often, in seconds, what a train's cargo is worth is worked out again.
WORTH_EVERY = 0.5


@dataclass
class TrainsWindow:
    open: bool = False
    # The name of the train shown on the right.
    chosen: str | None = None
    # What each train's cars would earn at its destination, worked out at most every
    # WORTH_EVERY seconds, by train name: when, and per car.
    _worth: dict[str, tuple[float, list[float]]] = field(default_factory=dict)
    map: RouteMap = field(default_factory=RouteMap)

    def worth(self, railway: Railway, train: Train) -> list[float]:
        """What each of the train's cars would earn at its destination now, kept for a moment
        so the list of trains stays cheap to draw."""
        now = time.monotonic()
        kept = self._worth.get(train.name)
        if kept is None or now - kept[0] > WORTH_EVERY or len(kept[1]) != len(train.cars):
            kept = (now, railway.cargo_worth(train))
            self._worth[train.name] = kept
        return kept[1]

    def draw(self, railway: Railway, supplies: Supplies, ground: GroundPicture) -> None:
        if not self.open:
            return
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(
            (max(8.0, display.x - WINDOW_SIZE[0] - CARDS_WIDTH - 16), WINDOW_TOP),
            imgui.Cond_.first_use_ever.value,
        )
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size_constraints(SMALLEST_WINDOW, (1e6, 1e6))
        expanded, self.open = imgui.begin(f"{fa.ICON_FA_TRAIN}  Trains###trains", self.open)
        if expanded:
            train = self._draw_list(railway)
            imgui.same_line()
            imgui.begin_child("train", (0, 0))
            if train is None:
                _centered_hint("Buy a train, then give it two stations joined by track.")
            else:
                self._draw_train(railway, train, supplies, ground)
            imgui.end_child()
        imgui.end()

    def _draw_list(self, railway: Railway) -> Train | None:
        imgui.begin_child("list", (LIST_WIDTH, 0), imgui.ChildFlags_.borders.value)
        if imgui.button(f"{fa.ICON_FA_PLUS}  Buy a train", (-1, 0)):
            imgui.open_popup("buy train")
        if imgui.begin_popup("buy train"):
            shown = next((t for t in railway.trains if t.name == self.chosen), None)
            if shown is not None and shown.stops:
                imgui.text_disabled(f"Calls at the stops of {shown.name}, with no cars yet")
            imgui.text_disabled("Buy a train with")
            for name in railway.engines():
                engine = ENGINES[name]
                affordable = railway.finances.can_afford(engine.price)
                imgui.begin_disabled(not affordable)
                label = f"{engine.title}  {dollars(engine.price)}##{name}"
                if imgui.selectable(label, False)[0]:
                    bought = railway.buy(name, route_of=shown)
                    if bought is not None:
                        self.chosen = bought.name
                imgui.end_disabled()
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                    track = ", electrified track only" if engine.electric else ""
                    money = "" if affordable else " - not enough money"
                    imgui.set_tooltip(
                        f"{engine.speed:g}x speed, up to {engine.cars} cars{track}{money}"
                    )
            imgui.end_popup()
        imgui.spacing()
        chosen = None
        for train in railway.trains:
            picked = train.name == self.chosen
            if imgui.selectable(f"##{train.name}", picked, 0, (0, 40))[0]:
                self.chosen = train.name
            low = imgui.get_item_rect_min()
            draw_list = imgui.get_window_draw_list()
            draw_list.add_text(
                (low.x + 6, low.y + 3), imgui.get_color_u32(imgui.Col_.text), train.name
            )
            status_color = imgui.get_color_u32(imgui.ImVec4(*TEXT_DIM))
            draw_list.add_text((low.x + 6, low.y + 21), status_color, train.status)
            worth = sum(self.worth(railway, train))
            if worth >= 0.5:
                money = dollars(worth)
                right = imgui.get_item_rect_max().x - 6 - imgui.calc_text_size(money).x
                draw_list.add_text((right, low.y + 3), color_u32(BRASS), money)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Its cargo is worth about {money} where it is going")
            if picked:
                chosen = train
        imgui.end_child()
        if chosen is None and railway.trains and self.chosen is None:
            self.chosen = railway.trains[0].name
        return chosen

    def _draw_train(
        self, railway: Railway, train: Train, supplies: Supplies, ground: GroundPicture
    ) -> None:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*BRASS))
        imgui.text(train.name)
        imgui.pop_style_color()
        imgui.same_line()
        imgui.text_disabled(f"{ENGINES[train.engine].title} - {train.status}")
        width = imgui.get_content_region_avail().x
        imgui.same_line(max(0.0, imgui.get_cursor_pos_x() + width - 90))
        sold = danger_button(f"{fa.ICON_FA_TRASH}  Sell")
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Sell it for {dollars(ENGINES[train.engine].price * TRAIN_RESALE)}")
        if sold:
            railway.sell(train)
            self.chosen = None
            return
        imgui.separator()
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Pulling")
        if not train.cars:
            imgui.same_line()
            imgui.text_disabled("no cars")
        destination = railway.destination(train)
        onward = "" if destination is None else railway.name_of(destination)
        worths = self.worth(railway, train)
        for index, (car, worth) in enumerate(zip(train.cars, worths, strict=True)):
            imgui.same_line(0, 4 if index else 12)
            tooltip = f"{car.good.value}: {car.load:.0f}/{car.capacity:.0f}"
            if destination is not None and car.load >= 0.5:
                tooltip += f"\nWorth about {dollars(worth)} at {onward}"
            car_chip(f"pulled{index}", car.good, car.load, tooltip, car.capacity)
        if destination is not None and any(car.load >= 0.5 for car in train.cars):
            imgui.same_line(0, 14)
            imgui.text_colored(imgui.ImVec4(*BRASS), f"worth about {dollars(sum(worths))}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"What its cargo would earn if it were unloaded at {onward} now, at the "
                    "demand left there.\nFreight is paid as the buildings around take it, and "
                    f"cargo that came from {onward} earns nothing."
                )
            imgui.same_line()
            imgui.text_disabled(f"at {onward}")
        imgui.spacing()
        self.map.good = None
        map_width = imgui.get_content_region_avail().x * MAP_SHARE
        imgui.begin_child("route", (-map_width, 0))
        imgui.text_disabled("Route")
        self._draw_stops(railway, train, supplies)
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("map", (0, 0), imgui.ChildFlags_.borders.value)
        self.map.draw(railway, train, ground, supplies)
        imgui.end_child()

    def _draw_stops(self, railway: Railway, train: Train, supplies: Supplies) -> None:
        stops = len(train.stops)
        current = train.heading % stops if stops and train.path else (train.heading - 1)
        for index, stop in enumerate(list(train.stops)):
            imgui.push_id(index)
            here = stops > 1 and index == current % stops
            removed = self._draw_stop(railway, train, index, stop, here, supplies)
            imgui.pop_id()
            if removed:
                break
        choices = [s for s in railway.stations() if can_stop_at(train, s)]
        if not choices:
            imgui.text_disabled("Build stations to give it stops.")
            return
        imgui.set_next_item_width(220)
        if imgui.begin_combo("##add stop", f"{fa.ICON_FA_PLUS}  Add a stop"):
            for station in choices:
                if imgui.selectable(railway.name_of(station), False)[0]:
                    add_stop(train, station)
            imgui.end_combo()
        imgui.same_line()
        if len(train.stops) == 1:
            imgui.text_disabled("A train needs a second stop to run.")
        else:
            imgui.text_disabled("or click one on the map")

    def _draw_stop(
        self,
        railway: Railway,
        train: Train,
        index: int,
        stop: Stop,
        here: bool,
        supplies: Supplies,
    ) -> bool:
        """One stop as a card: where it is, the cars the train takes on from it, the goods it
        has to load, and what the next stop wants; returns whether it was removed."""
        flags = imgui.ChildFlags_.borders.value | imgui.ChildFlags_.auto_resize_y.value
        imgui.begin_child("stop", (0, 0), flags)
        removed = self._draw_stop_head(railway, train, index, stop, here)
        if not removed:
            following = train.stops[(index + 1) % len(train.stops)]
            if len(train.stops) > 1:
                demand = railway.station_demand(following.cell)
                self._draw_cars(railway, train, index, stop, railway.name_of(following.cell))
                self._draw_offers(railway, train, index, stop, demand, supplies)
                imgui.align_text_to_frame_padding()
                imgui.text_disabled(f"{railway.name_of(following.cell)} wants")
                imgui.same_line(CARD_LABEL)
                draw_demand(demand)
        imgui.end_child()
        return removed

    def _draw_stop_head(
        self, railway: Railway, train: Train, index: int, stop: Stop, here: bool
    ) -> bool:
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(f"{index + 1}")
        imgui.same_line(0, 10)
        name = railway.name_of(stop.cell)
        if here:
            imgui.text_colored(imgui.ImVec4(*BRASS), f"{fa.ICON_FA_TRAIN}  {name}")
            imgui.same_line()
            imgui.text_disabled("heading here" if train.path else "standing here")
        else:
            imgui.text(name)
        imgui.same_line(right - 190)
        _, stop.wait_for_load = imgui.checkbox("Wait for full load", stop.wait_for_load)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Wait here, up to a month, for every car to fill")
        imgui.same_line(right - 22)
        if imgui.small_button(fa.ICON_FA_TIMES):
            train.stops.pop(index)
            train.restart()
            return True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Remove this stop")
        return False

    def _draw_cars(
        self, railway: Railway, train: Train, index: int, stop: Stop, onward: str
    ) -> None:
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(f"Cars to {onward}")
        imgui.same_line(CARD_LABEL)
        most = railway.most_cars(train)
        for car_index, good in enumerate(list(stop.cars)):
            if car_index:
                imgui.same_line(0, 4)
            removed = car_chip(
                f"car{car_index}",
                good,
                tooltip=f"{good.value} car of {railway.car_capacity(good):.0f} - click to remove",
            )
            if imgui.is_item_hovered():
                self.map.good = good
            if removed:
                stop.cars.pop(car_index)
                train.cars_changed(index)
        for slot in range(len(stop.cars), most):
            if slot:
                imgui.same_line(0, 4)
            empty_car_slot()
        if stop.cars:
            imgui.same_line(0, 10)
            if imgui.small_button("Clear"):
                stop.cars.clear()
                train.cars_changed(index)

    def _draw_offers(
        self,
        railway: Railway,
        train: Train,
        index: int,
        stop: Stop,
        demand: StationDemand,
        supplies: Supplies,
    ) -> None:
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Load here")
        imgui.same_line(CARD_LABEL)
        offers = goods_to_load(railway, stop.cell, supplies(stop.cell))
        here = {offer.good for offer in offers}
        waiting = railway.station_goods(stop.cell)
        carried = [g for g in carried_in(train, index) if g not in here]
        offers += [Offer(good, waiting.held(good), 0.0) for good in carried]
        if not offers:
            imgui.text_disabled("nothing made or waiting here yet")
            return
        most = railway.most_cars(train)
        full = len(stop.cars) >= most
        offers.sort(key=lambda o: (-(demand.monthly_want(o.good) > 0), -o.waiting, -o.made))
        left = imgui.get_cursor_pos_x()
        per_row = max(1, int(imgui.get_content_region_avail().x // (OFFER_SIZE[0] + 4)))
        for number, offer in enumerate(offers):
            if number % per_row:
                imgui.same_line(0, 4)
            elif number:
                imgui.set_cursor_pos_x(left)
            wanted = demand.monthly_want(offer.good) > 0
            tip = (
                f"{offer.good.value}: {offer.waiting:.0f} waiting, about {offer.made:.0f} a month"
                + (" - comes in on this train" if offer.good in carried else "")
                + (" - wanted at the next stop" if wanted else "")
                + ("\nThe train is full: remove a car first" if full else "\nClick to add a car")
            )
            imgui.begin_disabled(full)
            if offer_chip(f"offer{number}", offer.good, f"{offer.waiting:.0f}", wanted, tip):
                stop.add_car(offer.good, most)
                train.cars_changed(index)
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                self.map.good = offer.good
            imgui.end_disabled()


def carried_in(train: Train, index: int) -> list[Good]:
    """The goods a train's cars bring into its stop at `index`: what it takes on at the stop
    before, so a train can carry them on past a station that has no use for them."""
    if not train.stops:
        return []
    before = train.stops[(index - 1) % len(train.stops)]
    return list(dict.fromkeys(before.cars))


def _centered_hint(text: str) -> None:
    width, height = imgui.get_content_region_avail()
    size = imgui.calc_text_size(text)
    imgui.set_cursor_pos(
        (
            imgui.get_cursor_pos_x() + max(0.0, (width - size.x) / 2),
            imgui.get_cursor_pos_y() + height / 2 - size.y,
        )
    )
    imgui.text_disabled(text)
