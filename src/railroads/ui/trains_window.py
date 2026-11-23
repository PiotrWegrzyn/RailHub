"""The trains window: every train in a list on the left; the chosen one's cars and route on the
right, a card for each stop with the cars it pulls on to the next, the goods made or waiting there
to click into cars, those the next stop wants outlined, and what the next stop wants; and beside the
cards a map to add stops on, which marks where a hovered good is wanted (railroads.ui.route_map).
The chosen train's engine can be upgraded to a newer one of its line, alone or with every train of
that line with an older engine."""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.demand import StationDemand
from railroads.sim.economy import SCARCE_WORTH
from railroads.sim.finance import TRAIN_RESALE, dollars
from railroads.sim.fuel_market import fuel_of
from railroads.sim.goods import GENERAL_GOODS, General, Good
from railroads.sim.grids import Cell
from railroads.sim.railway import NO_WAY_ON, Railway, Ride
from railroads.sim.trains import (
    ENGINES,
    EXPRESS_FARE,
    MAGLEV_FARE,
    SLOWDOWN,
    Engine,
    Kind,
    Stop,
    Train,
    amount_of,
    unit_name,
)
from railroads.ui.buy_train import (
    BuyScreen,
    EnginePicture,
    engine_range,
    fuel_use,
    tonnes_text,
)
from railroads.ui.demand_view import draw_demand
from railroads.ui.ornate import card, ornate_window, section, tile
from railroads.ui.route_map import GroundPicture, RouteMap, add_stop, can_stop_at
from railroads.ui.spotlight import spot
from railroads.ui.station_offers import Offer, Supplies, goods_to_load
from railroads.ui.theme import (
    BRASS,
    CARDS_WIDTH,
    GONE,
    OFFER_SIZE,
    TEXT_DIM,
    caption_room,
    caption_under,
    car_chip,
    color_u32,
    danger_button,
    dining_chip,
    draw_small_car,
    empty_car_slot,
    offer_chip,
)

# How a cargo's worth at a stop is worked out, as freight is paid there.
CARGO_WORTH = (
    "Goods earn their freight for what the buildings around would take at once, up to "
    f"{SCARCE_WORTH:g} times as much where they went short; the rest waits at the station. "
    "Passengers and mail pay by the demand left there."
)
WINDOW_SIZE = (980.0, 500.0)
# The window can't be made smaller than this, so the map beside the stop cards keeps its room.
SMALLEST_WINDOW = (900.0, 440.0)
WINDOW_TOP = 96.0
LIST_WIDTH = 190.0
# Where the cars start in a train's row in the list, under its name and status.
ROW_CARS_TOP = 39.0
# The size of a car in a train's row, and the gap between cars and between their lines.
ROW_CAR = 16.0
ROW_CAR_GAP = 2.0
# The most lines a train's cars take in its row; a train too long for them has its cars narrowed.
ROW_CAR_LINES = 3
# The share of the space beside the list of trains the map takes.
MAP_SHARE = 0.45
# How far from a stop card's right edge its top line's buttons start: the Wait for full load
# box, the arrows moving the stop and the one removing it.
HEAD_BUTTONS = 236.0
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
    _rides: dict[str, tuple[float, Ride | None]] = field(default_factory=dict)
    map: RouteMap = field(default_factory=RouteMap)
    # The screen trains are bought on, and the map on it for the route of a train just bought.
    buy: BuyScreen = field(default_factory=BuyScreen)
    buy_map: RouteMap = field(default_factory=RouteMap)
    # How tall each stop card was last drawn, by train, stop and width: a card out of sight
    # only takes up its room.
    _card_heights: dict[tuple[str, int, int], float] = field(default_factory=dict)
    # The map of the train drawn now, which shows where a good hovered on its stop cards is wanted.
    _shown_map: RouteMap = field(default_factory=RouteMap)
    # What each stop's station has to send on each month, worked out at most every WORTH_EVERY
    # seconds, by station: when, and the goods.
    _supplied: dict[Cell, tuple[float, Mapping[Good, float]]] = field(default_factory=dict)
    # The generation of cars added to each train's stops, by train name: 0 for the newest.
    _asked: dict[str, int] = field(default_factory=dict)
    # Stop cards the player folded away or opened, by train and stop; the rest are open but for
    # those with cars set that the train is not heading for or standing at.
    _folded: dict[tuple[str, int], bool] = field(default_factory=dict)

    def worth(self, railway: Railway, train: Train) -> list[float]:
        """What each of the train's cars would earn at its destination now, kept for a moment
        so the list of trains stays cheap to draw."""
        now = time.monotonic()
        kept = self._worth.get(train.name)
        if kept is None or now - kept[0] > WORTH_EVERY or len(kept[1]) != len(train.cars):
            kept = (now, railway.cargo_worth(train))
            self._worth[train.name] = kept
        return kept[1]

    def ride(self, railway: Railway, train: Train) -> Ride | None:
        """What the train's ride to its next stop earns and costs, kept like its cars' worth."""
        now = time.monotonic()
        kept = self._rides.get(train.name)
        if kept is None or now - kept[0] > WORTH_EVERY:
            kept = (now, railway.ride(train, self.worth(railway, train)))
            self._rides[train.name] = kept
        return kept[1]

    def supplied(self, supplies: Supplies, station: Cell) -> Mapping[Good, float]:
        """What a station has to send on each month, kept for a moment as it changes only
        monthly and the stop cards ask every frame."""
        now = time.monotonic()
        kept = self._supplied.get(station)
        if kept is None or now - kept[0] > WORTH_EVERY:
            kept = (now, supplies(station))
            self._supplied[station] = kept
        return kept[1]

    def draw(
        self,
        railway: Railway,
        supplies: Supplies,
        ground: GroundPicture,
        picture: EnginePicture,
    ) -> None:
        if not self.open:
            self.buy.open = False
        if not self.buy.open:
            railway.let_go()
        if not self.open:
            return
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(
            (max(8.0, display.x - WINDOW_SIZE[0] - CARDS_WIDTH - 16), WINDOW_TOP),
            imgui.Cond_.first_use_ever.value,
        )
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size_constraints(SMALLEST_WINDOW, (1e6, 1e6))
        with ornate_window("Trains", "trains") as (expanded, still_open):
            if expanded:
                train = self._draw_list(railway)
                imgui.same_line()
                imgui.begin_child("train", (0, 0))
                if train is None:
                    _centered_hint("Buy a train, then give it two stations joined by track.")
                else:
                    self._draw_train(railway, train, supplies, ground, self.map)
                imgui.end_child()
        self.open = still_open
        bought = self.buy.draw(
            railway,
            engines_by_kind(railway.engines()),
            engine_summary,
            picture,
            lambda shown: self._draw_train(railway, shown, supplies, ground, self.buy_map),
        )
        if bought is not None:
            self.chosen = bought

    def _draw_list(self, railway: Railway) -> Train | None:
        """The Buy a train button, always at the top, over the list of trains, which scrolls on
        its own; only the trains in sight are drawn."""
        imgui.begin_child("list", (LIST_WIDTH, 0), imgui.ChildFlags_.borders.value)
        if imgui.button(f"{fa.ICON_FA_PLUS}  Buy a train", (-1, 0)):
            self.buy.open_for(next((t for t in railway.trains if t.name == self.chosen), None))
        spot("buy_train")
        imgui.spacing()
        imgui.begin_child("rows", (0, 0))
        trains = railway.trains
        for train in trains:
            self._draw_row(railway, train)
        imgui.end_child()
        imgui.end_child()
        chosen = next((t for t in trains if t.name == self.chosen), None)
        if chosen is None and trains and self.chosen is None:
            self.chosen = trains[0].name
        return chosen

    def _draw_row(self, railway: Railway, train: Train) -> None:
        """A train's row: its name, status and cargo's worth over its cars, in up to
        ROW_CAR_LINES lines; a row out of sight only takes up its room."""
        width = imgui.get_content_region_avail().x
        car_width, per_line = row_cars(len(train.cars), width - 12)
        lines = -(-len(train.cars) // per_line)
        height = ROW_CARS_TOP + lines * (ROW_CAR + ROW_CAR_GAP) + 1
        if not imgui.is_rect_visible((width, height)):
            imgui.dummy((width, height))
            return
        picked = train.name == self.chosen
        if imgui.selectable(f"##{train.name}", picked, 0, (0, height))[0]:
            self.chosen = train.name
        low = imgui.get_item_rect_min()
        draw_list = imgui.get_window_draw_list()
        draw_list.add_text((low.x + 6, low.y + 3), imgui.get_color_u32(imgui.Col_.text), train.name)
        stuck = train.status in NO_WAY_ON or bool(railway.long_legs(train))
        status_color = color_u32(GONE) if stuck else imgui.get_color_u32(imgui.ImVec4(*TEXT_DIM))
        draw_list.add_text((low.x + 6, low.y + 21), status_color, train.status)
        ride = self.ride(railway, train)
        if ride is not None:
            money = dollars(ride.net)
            right = imgui.get_item_rect_max().x - 6 - imgui.calc_text_size(money).x
            colour = BRASS if ride.net >= 0 else GONE
            draw_list.add_text((right, low.y + 3), color_u32(colour), money)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"What this ride earns after its costs\n\n{ride_lines(ride)}")
        for index, car in enumerate(train.cars):
            line, place = divmod(index, per_line)
            left = low.x + 6 + place * (car_width + ROW_CAR_GAP)
            top = low.y + ROW_CARS_TOP + line * (ROW_CAR + ROW_CAR_GAP)
            good = None if car.dining else car.good
            draw_small_car(
                draw_list, good, car.load, car.capacity, (left, top), (car_width, ROW_CAR)
            )

    def _draw_train(
        self,
        railway: Railway,
        train: Train,
        supplies: Supplies,
        ground: GroundPicture,
        route_map: RouteMap,
    ) -> None:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*BRASS))
        imgui.text(train.name)
        imgui.pop_style_color()
        imgui.same_line()
        imgui.text_disabled(f"{ENGINES[train.engine].title} - {train.status}")
        width = imgui.get_content_region_avail().x
        imgui.same_line(max(0.0, imgui.get_cursor_pos_x() + width - 330))
        first_button(train)
        imgui.same_line()
        upgrade_button(railway, train)
        imgui.same_line()
        sold = danger_button(f"{fa.ICON_FA_TRASH}  Sell")
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Sell it for {dollars(ENGINES[train.engine].price * TRAIN_RESALE)}")
        if sold:
            railway.sell(train)
            self.chosen = None
            return
        upgrade_menu(railway, train)
        imgui.separator()
        destination = railway.destination(train)
        onward = "" if destination is None else railway.name_of(destination)
        if warning := train_warning(railway, train):
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(imgui.ImVec4(*GONE), f"{fa.ICON_FA_EXCLAMATION_TRIANGLE}  {warning}")
            imgui.pop_text_wrap_pos()
        worths = self.worth(railway, train)
        ride = railway.ride(train, worths)
        if ride is None:
            self._draw_pulling(railway, train, worths, destination, onward)
        else:
            _draw_tiles(railway, ride, onward)
            if imgui.tree_node_ex(f"The ride's bill and the cars pulled now###bill{train.name}"):
                self._draw_pulling(railway, train, worths, destination, onward)
                self._draw_ride(ride, onward)
                imgui.tree_pop()
        imgui.spacing()
        route_map.good = None
        self._shown_map = route_map
        map_width = imgui.get_content_region_avail().x * MAP_SHARE
        imgui.begin_child("route", (-map_width, 0))
        section("Route")
        self._draw_stops(railway, train, supplies)
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("map", (0, 0), imgui.ChildFlags_.borders.value)
        route_map.draw(railway, train, ground, supplies)
        imgui.end_child()

    @staticmethod
    def _draw_pulling(
        railway: Railway, train: Train, worths: list[float], destination: Cell | None, onward: str
    ) -> None:
        """The cars the train pulls now, each with its load, and what its cargo is worth."""
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Pulling")
        if not train.cars:
            imgui.same_line()
            imgui.text_disabled("no cars")
        for index, (car, worth) in enumerate(zip(train.cars, worths, strict=True)):
            imgui.same_line(0, 4 if index else 12)
            if car.dining:
                dining_chip(f"pulled{index}", "Dining car: carries nothing, costs its trips")
                caption_under(f"{tonnes_text(car.tonnes)} t")
                continue
            tooltip = (
                f"{car.good.value}: {car.load:.0f}/{car.capacity:.0f} "
                f"{unit_name(car.good, car.capacity)}, {tonnes_text(car.tonnes)} t"
            )
            if car.general is not None:
                tooltip = f"{car.general.value} car, now {tooltip}"
            if destination is not None and car.load >= 0.5:
                tooltip += f"\nWorth about {dollars(worth)} at {onward}"
            car_chip(f"pulled{index}", car.good, car.load, tooltip, car.capacity)
            caption_under(f"{tonnes_text(car.tonnes)} t")
        if destination is not None and any(car.load >= 0.5 for car in train.cars):
            imgui.same_line(0, 14)
            imgui.text_colored(imgui.ImVec4(*BRASS), f"worth about {dollars(sum(worths))}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"What its cargo would earn if it were unloaded at {onward} now.\n{CARGO_WORTH}"
                    f"\nCargo that came from {onward} only goes back into its stockpile."
                )
            imgui.same_line()
            imgui.text_disabled(f"at {onward}")
        if train.cars:
            imgui.same_line(0, 14)
            draw_tonnage(railway, train)
            caption_room()

    @staticmethod
    def _draw_ride(ride: Ride, onward: str) -> None:
        """The ride to the next stop as a bill: what the cargo earns there, less the fuel for
        the way and the trips of its loaded and empty cars."""
        imgui.text_disabled(f"Ride to {onward}, {ride.cells:.0f} cells")
        if imgui.begin_table("ride", 2, imgui.TableFlags_.sizing_fixed_fit.value):
            for label, amount, tip in ride_rows(ride):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(label)
                if tip and imgui.is_item_hovered():
                    imgui.set_tooltip(tip)
                imgui.table_next_column()
                colour = BRASS if amount >= 0 else GONE
                imgui.text_colored(imgui.ImVec4(*colour), signed(amount))
            imgui.end_table()

    def _draw_stops(self, railway: Railway, train: Train, supplies: Supplies) -> None:
        stops = len(train.stops)
        current = train.heading % stops if stops and train.path else (train.heading - 1)
        width = imgui.get_content_region_avail().x
        for index, stop in enumerate(list(train.stops)):
            key = (train.name, index, round(width))
            low = imgui.get_cursor_screen_pos()
            height = self._card_heights.get(key)
            if height is not None and not imgui.is_rect_visible(
                low, (low.x + width, low.y + height)
            ):
                imgui.dummy((width, height))
                continue
            imgui.push_id(index)
            here = stops > 1 and index == current % stops
            removed = self._draw_stop(railway, train, index, stop, here, supplies)
            imgui.pop_id()
            if removed:
                self._card_heights.clear()
                break
            self._card_heights[key] = imgui.get_item_rect_size().y
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
        else:
            spot("add_stop")
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
        has to load, and what the next stop wants; returns whether it was removed or moved."""
        with card("stop", highlight=here):
            key = (train.name, index)
            opened = self._folded.setdefault(key, not (stop.cars and not here))
            removed, toggled, moved = self._draw_stop_head(
                railway, train, index, stop, here, opened
            )
            if moved:
                return True
            leg = dict(railway.long_legs(train)).get(index)
            if leg is not None and not removed:
                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(
                    imgui.ImVec4(*GONE),
                    f"{fa.ICON_FA_EXCLAMATION_TRIANGLE}  {leg_warning(railway, train, index, leg)}",
                )
                imgui.pop_text_wrap_pos()
            if toggled:
                opened = not opened
                self._folded[key] = opened
            if removed:
                self._folded = {k: v for k, v in self._folded.items() if k[0] != train.name}
            elif opened and len(train.stops) > 1:
                following = train.stops[train.onward(index)]
                if self._draw_refuel(railway, stop, following):
                    return removed
                demand = railway.station_demand(following.cell)
                self._draw_cars(railway, train, index, stop, railway.name_of(following.cell))
                self._draw_offers(railway, train, index, stop, demand, supplies)
                imgui.align_text_to_frame_padding()
                imgui.text_disabled(f"{railway.name_of(following.cell)} wants")
                imgui.same_line(CARD_LABEL)
                draw_demand(demand)
        return removed

    def _draw_stop_head(
        self, railway: Railway, train: Train, index: int, stop: Stop, here: bool, opened: bool
    ) -> tuple[bool, bool, bool]:
        """A stop card's top line: a button folding the card away or opening it, the stop's
        number and name, when folded the cars it sends on, and buttons moving it up or down the
        route and removing it; returns whether the stop was removed, whether the card was folded
        or opened and whether the stop was moved."""
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        arrow = fa.ICON_FA_CARET_DOWN if opened else fa.ICON_FA_CARET_RIGHT
        toggled = imgui.small_button(f"{arrow}##fold")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Fold this stop away" if opened else "Show this stop's cars and goods"
            )
        imgui.same_line(0, 6)
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(f"{index + 1}")
        imgui.same_line(0, 10)
        name = railway.name_of(stop.cell)
        if here:
            imgui.text_colored(imgui.ImVec4(*BRASS), f"{fa.ICON_FA_TRAIN}  {name}")
            _note_before("heading here" if train.path else "standing here", right - HEAD_BUTTONS)
        else:
            imgui.text(name)
        if not opened and len(train.stops) > 1:
            following = railway.name_of(train.stops[train.onward(index)].cell)
            note = (
                "only refuels"
                if stop.refuel_only
                else f"{len(stop.cars)} car{'' if len(stop.cars) == 1 else 's'} to {following}"
            )
            _note_before(note, right - HEAD_BUTTONS)
        imgui.same_line(right - HEAD_BUTTONS)
        imgui.begin_disabled(stop.refuel_only)
        _, stop.wait_for_load = imgui.checkbox("Wait for full load", stop.wait_for_load)
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            imgui.set_tooltip(
                "It only refuels here, so it doesn't wait for a load"
                if stop.refuel_only
                else "Wait here, up to a month, for every car to fill"
            )
        moved = False
        for step, icon, left, tip_text in (
            (-1, fa.ICON_FA_ARROW_UP, 70, "Call here one stop earlier"),
            (1, fa.ICON_FA_ARROW_DOWN, 46, "Call here one stop later"),
        ):
            imgui.same_line(right - left)
            other = index + step
            imgui.begin_disabled(not 0 <= other < len(train.stops))
            if imgui.small_button(f"{icon}##move{step}"):
                self._move_stop(train, index, other)
                moved = True
            imgui.end_disabled()
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"{tip_text}, with its cars")
        imgui.same_line(right - 22)
        if imgui.small_button(fa.ICON_FA_TIMES):
            train.stops.pop(index)
            train.restart()
            return True, toggled, False
        if imgui.is_item_hovered():
            imgui.set_tooltip("Remove this stop")
        return False, toggled, moved

    def _move_stop(self, train: Train, index: int, other: int) -> None:
        """Swaps a stop with the one before or after it, cars, flags and folding and all, and
        sends the train back to its first stop."""
        stops = train.stops
        stops[index], stops[other] = stops[other], stops[index]
        here, there = (train.name, index), (train.name, other)
        folded_here, folded_there = self._folded.get(here), self._folded.get(there)
        for key, value in ((here, folded_there), (there, folded_here)):
            if value is None:
                self._folded.pop(key, None)
            else:
                self._folded[key] = value
        train.restart()

    @staticmethod
    def _draw_refuel(railway: Railway, stop: Stop, following: Stop) -> bool:
        """The checkbox making a stop one the train only refuels at, for a leg longer than its
        fuel lasts, and what that means; returns whether the stop is one."""
        _, stop.refuel_only = imgui.checkbox("Only refuel here", stop.refuel_only)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "For a way longer than the engine carries fuel for: the train only takes on fuel\n"
                "here and goes on with its cars, taking its passengers and mail through"
            )
        if stop.refuel_only:
            imgui.text_disabled(
                f"Unloads and loads nothing here: its cars ride on to "
                f"{railway.name_of(following.cell)}"
            )
        return stop.refuel_only

    def _draw_cars(
        self, railway: Railway, train: Train, index: int, stop: Stop, onward: str
    ) -> None:
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(f"Cars to {onward}")
        imgui.same_line(CARD_LABEL)
        most = railway.most_cars(train)
        for car_index, (cargo, asked) in enumerate(stop.asked()):
            if car_index:
                imgui.same_line(0, 4)
            good = GENERAL_GOODS[cargo][0] if isinstance(cargo, General) else cargo
            generation = railway.car_generation(good, train.engine, asked)
            if isinstance(cargo, General):
                tooltip = (
                    f"G{generation} {cargo.value} car: takes whichever of its goods pays best at"
                    " the next stop - click to remove"
                )
            else:
                size = railway.car_capacity(cargo, train.engine, asked=asked)
                tooltip = (
                    f"G{generation} {cargo.value} car of {amount_of(cargo, size)} - click to remove"
                )
            tonnes = railway.car_tonnes(cargo, train.engine, asked)
            tooltip += f"\nWeighs {tonnes_text(tonnes)} t full"
            removed = car_chip(f"car{car_index}", cargo, tooltip=tooltip)
            caption_under(f"{tonnes_text(tonnes)} t")
            if imgui.is_item_hovered() and isinstance(cargo, Good):
                self._shown_map.good = cargo
            if removed:
                stop.remove_car(car_index)
                train.cars_changed(index)
        for slot in range(len(stop.cars), most):
            if slot:
                imgui.same_line(0, 4)
            empty_car_slot()
        if stop.cars:
            imgui.same_line(0, 10)
            if imgui.small_button("Clear"):
                stop.clear_cars()
                train.cars_changed(index)
            imgui.same_line(0, 10)
            planned = railway.stop_tonnes(train, stop)
            imgui.text_disabled(
                f"{tonnes_text(planned)} / {tonnes_text(railway.max_pull(train.engine))} t full"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "What these cars weigh full, against the most its engine pulls;\n"
                    f"full, it runs at {speed_share(railway, train, planned):.0%} of its top speed"
                )
        caption_room()

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
        asked = self._draw_generation_picker(railway, train, index)
        imgui.same_line(CARD_LABEL)
        offers = [
            offer
            for offer in goods_to_load(railway, stop.cell, self.supplied(supplies, stop.cell))
            if railway.can_carry(train, offer.good)
        ]
        here = {offer.good for offer in offers}
        waiting_here = railway.station_goods(stop.cell)
        carried = [
            g for g in carried_in(train, index) if g not in here and railway.can_carry(train, g)
        ]
        offers += [Offer(good, waiting_here.held(good), 0.0) for good in carried]
        generals = [general for general in General if railway.can_carry(train, general)]
        if not offers and not generals:
            imgui.text_disabled("nothing made or waiting here yet")
            return
        most = railway.most_cars(train)
        full = len(stop.cars) >= most
        spare = railway.max_pull(train.engine) - railway.stop_tonnes(train, stop)

        def fits(cargo: Good | General) -> bool:
            return railway.car_tonnes(cargo, train.engine, asked) <= spare + 1e-9

        offers.sort(key=lambda o: (-(demand.monthly_want(o.good) > 0), -o.waiting, -o.gathered))
        left = imgui.get_cursor_pos_x()
        per_row = max(1, int(imgui.get_content_region_avail().x // (OFFER_SIZE[0] + 4)))
        for number, offer in enumerate(offers):
            if number % per_row:
                imgui.same_line(0, 4)
            elif number:
                imgui.set_cursor_pos_x(left)
            wanted = demand.monthly_want(offer.good) > 0
            blocked = full or not fits(offer.good)
            tip = (
                f"{amount_of(offer.good, offer.waiting)} waiting, "
                f"{offer.gathered:.0f} given to the station in the last quarter"
                + (" - comes in on this train" if offer.good in carried else "")
                + (" - wanted at the next stop" if wanted else "")
                + adding_tip(full, railway.car_tonnes(offer.good, train.engine, asked), spare)
            )
            imgui.begin_disabled(blocked)
            if offer_chip(f"offer{number}", offer.good, f"{offer.waiting:.0f}", wanted, tip):
                railway.add_car(train, stop, offer.good, asked)
                train.cars_changed(index)
            spot(f"cargo:{index}:{offer.good.value}")
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                self._shown_map.good = offer.good
            imgui.end_disabled()
        for number, general in enumerate(generals, start=len(offers)):
            if number % per_row:
                imgui.same_line(0, 4)
            elif number:
                imgui.set_cursor_pos_x(left)
            waiting = sum(waiting_here.held(good) for good in GENERAL_GOODS[general])
            tip = (
                f"{general.value}: a car that takes whichever of its goods pays best at the "
                f"next stop; {waiting:.0f} of them waiting here"
                + adding_tip(full, railway.car_tonnes(general, train.engine, asked), spare)
            )
            imgui.begin_disabled(full or not fits(general))
            if offer_chip(f"general{number}", general, f"{waiting:.0f}", waiting > 0, tip):
                railway.add_car(train, stop, general, asked)
                train.cars_changed(index)
            imgui.end_disabled()

    def _draw_generation_picker(self, railway: Railway, train: Train, index: int) -> int:
        """A small list beside 'Load here' of the generations of cars that can be added: the
        newest its engine pulls, the default, or an older, smaller and lighter one; returns the
        one picked, 0 for the newest."""
        newest = railway.newest_generation(picker_good(train), train.engine)
        asked = min(self._asked.get(train.name, 0), newest)
        imgui.same_line(0, 8)
        imgui.set_next_item_width(max(44.0, CARD_LABEL - imgui.get_cursor_pos_x() - 6))
        shown = f"G{asked or newest}"
        if imgui.begin_combo(f"##cars{index}", shown):
            for generation in range(newest, 0, -1):
                label = f"G{generation} (newest)" if generation == newest else f"G{generation}"
                picked = (asked or newest) == generation
                if imgui.selectable(label, picked)[0]:
                    asked = 0 if generation == newest else generation
                    self._asked[train.name] = asked
            imgui.end_combo()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "The generation of the cars a click below adds: the newest by default, or an\n"
                "older one that holds less and weighs less, for light engines and hills"
            )
        return asked


def speed_share(railway: Railway, train: Train, tonnes: float) -> float:
    """How much of its engine's top speed a train makes pulling `tonnes`."""
    most = railway.max_pull(train.engine)
    return 1.0 - SLOWDOWN[ENGINES[train.engine].kind] * min(1.0, tonnes / most)


def draw_tonnage(railway: Railway, train: Train) -> None:
    """What a train's cars weigh now against the most its engine pulls, how fast that lets it
    run and how much fuel it burns."""
    most = railway.max_pull(train.engine)
    imgui.text_disabled(f"{tonnes_text(train.tonnes)} / {tonnes_text(most)} t")
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            f"What its cars weigh now, against the most its engine pulls;\nit runs at "
            f"{railway.pull(train):.0%} of its top speed, "
            f"{speed_share(railway, train, most):.0%} pulling the most"
            f"\nIt burns {railway.burn(train):.2g} {fuel_of(ENGINES[train.engine]).value} a cell, "
            f"{railway.burn(train) / ENGINES[train.engine].burn:.0%} of what it burns half loaded"
        )


def picker_good(train: Train) -> Good:
    """The good whose cars the generation picker lists: coaches behind an express."""
    return Good.PASSENGERS if ENGINES[train.engine].express else Good.COAL


FIRST_TIP = (
    "Puts it first in every queue at the signals until it reaches its next stop: it gets the\n"
    "way on before any train waiting there, and takes back the way set for a train that has\n"
    "yet to come to it. It still never runs into a train standing on the track."
)


def first_button(train: Train) -> None:
    """The button that puts a running train first in every queue for the way on, or, once it
    is, says so."""
    imgui.begin_disabled(not train.path or train.first)
    label = "First in line" if train.first else "Go first"
    if imgui.button(f"{fa.ICON_FA_FAST_FORWARD}  {label}"):
        train.first = True
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip(FIRST_TIP)


def adding_tip(full: bool, tonnes: float, spare: float) -> str:
    """The last line of a good's tooltip on a stop card: whether a car of it can be added."""
    if full:
        return "\nThe train is full: remove a car first"
    if tonnes > spare + 1e-9:
        return (
            f"\nToo heavy: a full car weighs {tonnes_text(tonnes)} t and the engine pulls only "
            f"{tonnes_text(max(0.0, spare))} t more"
        )
    return f"\nClick to add a car, {tonnes_text(tonnes)} t full"


def train_warning(railway: Railway, train: Train) -> str:
    """What the train window warns of first: that no track the train can run on reaches its next
    stop, or that a way between two of its stops is longer than its fuel lasts, so it waits where
    it is; empty when it can go all the way round."""
    destination = railway.destination(train)
    if train.status in NO_WAY_ON and destination is not None:
        return (
            f"{train.status}, {railway.name_of(destination)}: it waits here until track joins them"
        )
    legs = railway.long_legs(train)
    if not legs:
        return ""
    index, leg = legs[0]
    return f"{leg_warning(railway, train, index, leg)}. Add a stop between them to refuel at."


def leg_warning(railway: Railway, train: Train, index: int, leg: float) -> str:
    """That the way from a train's stop to the next is longer than its engine carries fuel for."""
    there = train.stops[(index + 1) % len(train.stops)].cell
    tank = ENGINES[train.engine].tank
    return (
        f"From {railway.name_of(train.stops[index].cell)} to {railway.name_of(there)} is "
        f"{leg:.0f} cells, but it carries fuel for only {tank:.0f}"
    )


def upgrade_button(railway: Railway, train: Train) -> None:
    """A button opening the menu of engines the train's engine can be swapped for, greyed out
    when there are none."""
    options = railway.engine_upgrades(train)
    imgui.begin_disabled(not options)
    if imgui.button(f"{fa.ICON_FA_ARROW_UP}  Upgrade"):
        imgui.open_popup("upgrade engine")
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip(
            "Swap its engine for a newer one, keeping its stops and cars;\n"
            "the old engine goes towards the price at half what it cost"
            if options
            else "No newer engine of its line is on sale"
        )


def upgrade_menu(railway: Railway, train: Train) -> None:
    """The menu the upgrade button opens: each newer engine for this train, and for every train
    of its line with an older engine at once."""
    if not imgui.begin_popup("upgrade engine"):
        return
    for name in railway.engine_upgrades(train):
        engine = ENGINES[name]
        alike = railway.trains_to_upgrade(name)
        imgui.separator_text(engine.title)
        price = railway.swap_price(train, name)
        imgui.begin_disabled(not railway.finances.can_afford(price))
        if imgui.selectable(f"This train  {dollars(price)}##one{name}", False)[0]:
            railway.swap_engine(train, name)
        imgui.end_disabled()
        summary = "\n".join(
            [engine_summary(engine, railway.fuel_cost(name)), *wagons_note(railway, name)]
        )
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            imgui.set_tooltip(summary)
        if len(alike) > 1:
            prices = [railway.swap_price(t, name) for t in alike]
            total = sum(prices)
            imgui.begin_disabled(not railway.finances.can_afford(min(prices)))
            line = engine.line.value
            label = (
                f"All {len(alike)} {line} trains with older engines  {dollars(total)}##all{name}"
            )
            if imgui.selectable(label, False)[0]:
                railway.swap_all_engines(name)
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                imgui.set_tooltip(f"{summary}\nAs many as the cash allows, in the order bought")
    imgui.end_popup()


def wagons_note(railway: Railway, name: str) -> list[str]:
    """A line saying the engine's cars stay the size of today's wagons until newer ones come,
    when it pulls newer ones than there are yet; none otherwise."""
    era = railway.wagon_era()
    top = ENGINES[name].top_generation
    if top <= era:
        return []
    note = f"Its cars stay generation {era}, the newest there are, until a new era brings "
    return [f"{note}generation {era + 1}"]


def engines_by_kind(names: list[str]) -> list[tuple[Kind, list[str]]]:
    """Engines grouped by what they are for, in the order of Kind, each group newest first."""
    groups = [
        (
            kind,
            sorted((n for n in names if ENGINES[n].kind == kind), key=lambda n: -ENGINES[n].year),
        )
        for kind in Kind
    ]
    return [(kind, group) for kind, group in groups if group]


def engine_summary(engine: Engine, per_cell: float | None = None) -> str:
    """An engine's speed, cars, fuel and what it runs on, for its tooltip, with what its fuel for
    a cell costs now if given."""
    cars = f"{engine.cars} coaches and a dining car" if engine.express else f"{engine.cars} cars"
    lines = [
        f"{engine.year}, {engine.power.value}: {engine.kmh:.0f} km/h, pulls up to {cars}",
        f"Pulls at most {tonnes_text(engine.tonnes)} t, {SLOWDOWN[engine.kind]:.0%} slower at that",
        f"Up to speed in {engine.pickup:g} s",
        f"Burns {fuel_use(engine, per_cell)}",
        f"Range: {engine_range(engine)}",
    ]
    lines.append(f"Pulls cars up to generation {engine.top_generation}")
    if engine.express:
        fare = MAGLEV_FARE if engine.maglev else EXPRESS_FARE
        lines.append(f"Passengers only, paying {fare - 1:.0%} more")
    if engine.electric:
        lines.append("Runs on electrified track only")
    if engine.maglev:
        lines.append("Runs on maglev track only")
    return "\n".join(lines)


def row_cars(cars: int, width: float) -> tuple[float, int]:
    """How wide a train's cars are drawn in its row of the list, and how many go on a line of
    that width: ROW_CAR wide over up to ROW_CAR_LINES lines, or narrowed to fit in them."""
    per_line = max(1, int((width + ROW_CAR_GAP) // (ROW_CAR + ROW_CAR_GAP)))
    if cars <= per_line * ROW_CAR_LINES:
        return ROW_CAR, per_line
    per_line = -(-cars // ROW_CAR_LINES)
    return max(1.0, (width + ROW_CAR_GAP) / per_line - ROW_CAR_GAP), per_line


def carried_in(train: Train, index: int) -> list[Good]:
    """The goods a train's cars bring into its stop at `index`: what it takes on at the stop
    before, past any it only refuels at, so a train can carry them on past a station that has no
    use for them."""
    if not train.stops:
        return []
    before = train.stops[train.before(index)]
    return list(dict.fromkeys(cargo for cargo in before.cars if isinstance(cargo, Good)))


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


def signed(amount: float) -> str:
    """Money earned with a plus, money paid with a minus."""
    return f"+{dollars(amount)}" if amount >= 0.5 else dollars(amount)


def ride_rows(ride: Ride) -> list[tuple[str, float, str]]:
    """A ride's bill, a line at a time: what it is for, how much, earned or paid, and what it
    means."""
    return [
        (
            "Cargo",
            ride.cargo,
            f"What its cargo would earn if it were unloaded there now.\n{CARGO_WORTH}",
        ),
        (
            f"Fuel for {ride.cells:.0f} cells",
            -ride.fuel,
            "Bought as it sets off, at retail; fuel the station has in stock comes cheaper",
        ),
        (
            f"{ride.loaded} loaded cars",
            -ride.loaded_cars,
            "Every car costs its trip each time the train leaves a stop with it",
        ),
        (
            f"{ride.empty} empty cars",
            -ride.empty_cars,
            "Empty cars cost their trip too: take only the cars the stop can fill",
        ),
        ("Ride earns", ride.net, ""),
    ]


def ride_lines(ride: Ride) -> str:
    """A ride's bill as lines of text."""
    return "\n".join(f"{label}: {signed(amount)}" for label, amount, _ in ride_rows(ride))


def ride_tiles(ride: Ride, onward: str) -> list[tuple[str, float, str]]:
    """The three figures over a train's route: what its cargo is worth at the next stop, what
    the ride there costs and what it earns after that."""
    costs = ride.fuel + ride.loaded_cars + ride.empty_cars
    return [
        (f"Cargo worth at {onward}", ride.cargo, ride_rows(ride)[0][2]),
        (
            f"Ride costs, {ride.cells:.0f} cells",
            -costs,
            "Fuel for the way and the trip of every car, loaded or empty",
        ),
        ("This ride earns", ride.net, "What its cargo earns there, less what the ride costs"),
    ]


def _draw_tiles(railway: Railway, ride: Ride, onward: str) -> None:
    """The ride to the next stop in three tiles side by side, the network's upkeep on hover."""
    stations, track = railway.network_upkeep()
    upkeep = (
        f"\n\nNot counted in any one ride: the whole network's upkeep, stations "
        f"{dollars(stations)} and track {dollars(track)} a month"
    )
    gap = 8.0
    width = (imgui.get_content_region_avail().x - 2 * gap) / 3
    for number, (label, amount, tip) in enumerate(ride_tiles(ride, onward)):
        if number:
            imgui.same_line(0, gap)
        tile(label, signed(amount), BRASS if amount >= 0 else GONE, width)
        if imgui.is_item_hovered():
            imgui.set_tooltip(tip + upkeep)


def _note_before(note: str, limit: float) -> None:
    """A dim note on the current line, left out if it would run past `limit`."""
    imgui.same_line()
    if imgui.get_cursor_pos_x() + imgui.calc_text_size(note).x < limit - 8:
        imgui.text_disabled(note)
