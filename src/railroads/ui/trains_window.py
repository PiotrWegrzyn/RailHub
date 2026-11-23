"""The trains window: every train in a list on the left; the chosen one's cars and route on the
right, a card for each stop with the cars it pulls on to the next, the goods made or waiting there
to click into cars, those the next stop wants outlined, and what the next stop wants; and beside the
cards a map to add stops on, which marks where a hovered good is wanted (railroads.ui.route_map).
The chosen train's engine can be upgraded to a newer one of its line, alone or with every train of
that line with an older engine."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.kept import Fresh
from railroads.sim.amounts import EPSILON
from railroads.sim.demand import StationDemand
from railroads.sim.finance import TRAIN_RESALE
from railroads.sim.fuel_market import fuel_of
from railroads.sim.goods import GENERAL_GOODS, General, Good
from railroads.sim.grids import Cell
from railroads.sim.railway import Railway, Ride
from railroads.sim.trains import (
    ENGINES,
    SLOWDOWN,
    Stop,
    Train,
    amount_of,
    unit_name,
)
from railroads.ui.buy_train import BuyScreen, EnginePicture
from railroads.ui.demand_view import draw_wants
from railroads.ui.ornate import card, ornate_window, section, tile
from railroads.ui.route_map import GroundPicture, RouteMap, add_stop, can_stop_at
from railroads.ui.spotlight import spot
from railroads.ui.station_offers import Offer, Supplies, goods_to_load
from railroads.ui.theme import (
    BORDER,
    BRASS,
    GONE,
    OFFER_SIZE,
    WINDOW_TOP,
    caption_room,
    caption_under,
    car_chip,
    color_u32,
    colored,
    danger_button,
    dining_chip,
    empty_car_slot,
    next_good_slot,
    offer_chip,
    tip,
)
from railroads.ui.train_table import ICON_BOX, TrainTable, engine_icon, shown_status, train_rows
from railroads.ui.train_texts import (
    CARGO_WORTH,
    FIRST_TIP,
    adding_tip,
    carried_in,
    engine_summary,
    engines_by_kind,
    leg_warning,
    picker_good,
    ride_rows,
    ride_tiles,
    signed,
    train_warning,
    wagons_note,
)
from railroads.words import dollars, plural, share_of, tonnes_text

# The share of the window the table of trains takes at first, the least it can be dragged to and
# the least it leaves the chosen train beside it.
LIST_SHARE = 0.5
LEAST_LIST = 420.0
LEAST_TRAIN = 700.0
# How wide the bar between the table and the chosen train is, which drags the split.
SPLIT_WIDTH = 8.0
# How wide the Go first, Upgrade and Sell buttons are together, at the right of a train's top
# line, and the room its name takes besides its text, for its gaps and the rename button.
HEAD_BUTTONS_WIDTH = 330.0
NAME_ROOM = 60.0
# The share of the space beside the chosen train's stop cards the map takes.
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
    # The name of the train shown on the right, and whether it fills the window.
    chosen: str | None = None
    maximized: bool = False
    table: TrainTable = field(default_factory=TrainTable)
    # How wide the player dragged the table, if they did.
    list_width: float | None = None
    _drawn: bool = False
    # The name being typed for the chosen train.
    _naming: str = ""
    # What each train's cars would earn at its destination, and its ride to its next stop,
    # worked out at most every WORTH_EVERY seconds, by train name.
    _worth: Fresh[str, list[float]] = field(default_factory=lambda: Fresh(WORTH_EVERY))
    _rides: Fresh[str, Ride | None] = field(default_factory=lambda: Fresh(WORTH_EVERY))
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
    # seconds, by station.
    _supplied: Fresh[Cell, Mapping[Good, float]] = field(default_factory=lambda: Fresh(WORTH_EVERY))
    # The generation of cars added to each train's stops, by train name: 0 for the newest.
    _asked: dict[str, int] = field(default_factory=dict)
    # Stop cards the player folded away or opened, by train and stop; the rest are open but for
    # those with cars set that the train is not heading for or standing at.
    _folded: dict[tuple[str, int], bool] = field(default_factory=dict)

    def worth(self, railway: Railway, train: Train) -> list[float]:
        """What each of the train's cars would earn at its destination now, kept for a moment
        so the list of trains stays cheap to draw."""
        return self._worth.get(train.name, lambda: railway.cargo_worth(train), key=len(train.cars))

    def ride(self, railway: Railway, train: Train) -> Ride | None:
        """What the train's ride to its next stop earns and costs, kept like its cars' worth."""
        return self._rides.get(train.name, lambda: railway.ride(train, self.worth(railway, train)))

    def supplied(self, supplies: Supplies, station: Cell) -> Mapping[Good, float]:
        """What a station has to send on each month, kept for a moment as it changes only
        monthly and the stop cards ask every frame."""
        return self._supplied.get(station, lambda: supplies(station))

    def draw(
        self,
        railway: Railway,
        supplies: Supplies,
        ground: GroundPicture,
        picture: EnginePicture,
    ) -> None:
        if not self.open:
            self.buy.open = False
            self._drawn = False
        if not self.buy.open:
            railway.let_go()
        if not self.open:
            return
        for kept in (self._worth, self._rides, self._supplied):
            kept.forget_old()
        if not self._drawn:
            imgui.set_next_window_focus()
            self._drawn = True
        display = imgui.get_io().display_size
        imgui.set_next_window_pos((0.0, WINDOW_TOP))
        imgui.set_next_window_size((display.x, display.y - WINDOW_TOP))
        flags = (
            imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_collapse.value
            | imgui.WindowFlags_.no_saved_settings.value
        )
        with ornate_window("Trains", "trains", flags) as (expanded, still_open):
            if expanded:
                self._draw_panes(railway, supplies, ground, picture)
        self.open = still_open
        bought = self.buy.draw(
            railway,
            engines_by_kind(railway.engines()),
            engine_summary,
            picture,
            lambda shown: self._draw_train(railway, shown, supplies, ground, self.buy_map, picture),
        )
        if bought is not None:
            self.chosen = bought

    def _draw_panes(
        self,
        railway: Railway,
        supplies: Supplies,
        ground: GroundPicture,
        picture: EnginePicture,
    ) -> None:
        """The table of trains on the left and the chosen train on the right, with a bar
        between them to drag; or the chosen train alone, filling the window."""
        trains = railway.trains
        if self.chosen is None and trains:
            self.chosen = trains[0].name
        train = next((t for t in trains if t.name == self.chosen), None)
        if train is None:
            self.maximized = False
        if not self.maximized:
            room = imgui.get_content_region_avail().x
            width = list_width(room, self.list_width)
            imgui.begin_child("list", (width, 0))
            self._draw_list(railway, picture)
            imgui.end_child()
            imgui.same_line(0, 0)
            imgui.invisible_button("split", (SPLIT_WIDTH, imgui.get_content_region_avail().y))
            if imgui.is_item_hovered() or imgui.is_item_active():
                imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
            if imgui.is_item_active():
                self.list_width = width + imgui.get_io().mouse_delta.x
            low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            middle = (low.x + high.x) / 2
            imgui.get_window_draw_list().add_line(
                (middle, low.y), (middle, high.y), color_u32(BORDER), 1.0
            )
            imgui.same_line(0, 0)
        imgui.begin_child("train", (0, 0))
        if train is None:
            _centered_hint("Buy a train, then give it two stations joined by track.")
        else:
            self._draw_maximize()
            imgui.same_line(0, 10)
            self._draw_train(railway, train, supplies, ground, self.map, picture)
        imgui.end_child()

    def _draw_maximize(self) -> None:
        """The button letting the chosen train fill the window, or taking it back beside the
        table."""
        if self.maximized:
            if imgui.button(f"{fa.ICON_FA_ARROW_LEFT}  All trains"):
                self.maximized = False
            tip("Back to the table of trains")
            return
        if imgui.button(fa.ICON_FA_EXPAND):
            self.maximized = True
        tip("Fill the window with this train; a double click on its row does too")

    def _draw_list(self, railway: Railway, picture: EnginePicture) -> None:
        """The Buy a train button and the filters, always at the top, over the table of
        trains, which scrolls on its own."""
        if imgui.button(f"{fa.ICON_FA_PLUS}  Buy a train"):
            self.buy.open_for(next((t for t in railway.trains if t.name == self.chosen), None))
        spot("buy_train")
        imgui.same_line(0, 12)
        self.table.draw_filters(railway)
        imgui.spacing()
        titles = railway.titles()
        rows = train_rows(railway, titles, lambda train: self.net(railway, train))
        clicked, twice = self.table.draw(rows, self.chosen, picture)
        if clicked is not None:
            self.chosen = clicked
            self.maximized = twice

    def net(self, railway: Railway, train: Train) -> float | None:
        ride = self.ride(railway, train)
        return None if ride is None else ride.net

    def _rename_button(self, train: Train, title: str) -> None:
        """A small pencil opening a box to name the train, or to give it back its route's
        name."""
        if imgui.small_button(f"{fa.ICON_FA_EDIT}##rename"):
            self._naming = train.label or title
            imgui.open_popup("rename train")
        tip("Name this train")
        if not imgui.begin_popup("rename train"):
            return
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here()
        imgui.set_next_item_width(260)
        entered, self._naming = imgui.input_text(
            "##name", self._naming, imgui.InputTextFlags_.enter_returns_true.value
        )
        if entered:
            train.label = self._naming.strip()
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Name it"):
            train.label = self._naming.strip()
            imgui.close_current_popup()
        imgui.begin_disabled(not train.label)
        if imgui.button("Use its route's name"):
            train.label = ""
            imgui.close_current_popup()
        imgui.end_disabled()
        tip("Called after its first and last stations and its category, as it was", disabled=True)
        imgui.end_popup()

    def _draw_train(
        self,
        railway: Railway,
        train: Train,
        supplies: Supplies,
        ground: GroundPicture,
        route_map: RouteMap,
        picture: EnginePicture,
    ) -> None:
        titles = railway.titles()
        title = titles.get(train.name, train.name)
        left = imgui.get_cursor_pos_x()
        right = left + imgui.get_content_region_avail().x
        engine_icon(picture, train)
        imgui.same_line(0, 8)
        imgui.align_text_to_frame_padding()
        colored(BRASS, title)
        imgui.same_line(0, 4)
        self._rename_button(train, title)
        named = left + ICON_BOX[0] + imgui.calc_text_size(title).x + NAME_ROOM
        if named + HEAD_BUTTONS_WIDTH <= right:
            imgui.same_line(right - HEAD_BUTTONS_WIDTH)
        else:
            imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), right - HEAD_BUTTONS_WIDTH))
        first_button(train)
        imgui.same_line()
        upgrade_button(railway, train)
        imgui.same_line()
        sold = danger_button(f"{fa.ICON_FA_TRASH}  Sell")
        tip(f"Sell it for {dollars(ENGINES[train.engine].price * TRAIN_RESALE)}")
        if sold:
            railway.sell(train)
            self.chosen = None
            return
        upgrade_menu(railway, train)
        imgui.text_disabled(f"{ENGINES[train.engine].title} - {shown_status(train.status, titles)}")
        imgui.separator()
        destination = railway.destination(train)
        onward = "" if destination is None else railway.name_of(destination)
        if warning := train_warning(railway, train):
            imgui.push_text_wrap_pos(0.0)
            colored(GONE, f"{fa.ICON_FA_EXCLAMATION_TRIANGLE}  {warning}")
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
            colored(BRASS, f"worth about {dollars(sum(worths))}")
            tip(
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
            for label, amount, hint in ride_rows(ride):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(label)
                if hint:
                    tip(hint)
                imgui.table_next_column()
                colour = BRASS if amount >= 0 else GONE
                colored(colour, signed(amount))
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
                colored(
                    GONE,
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
                draw_wants(demand)
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
        tip("Fold this stop away" if opened else "Show this stop's cars and goods")
        imgui.same_line(0, 6)
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(f"{index + 1}")
        imgui.same_line(0, 10)
        name = railway.name_of(stop.cell)
        if here:
            colored(BRASS, f"{fa.ICON_FA_TRAIN}  {name}")
            _note_before("heading here" if train.path else "standing here", right - HEAD_BUTTONS)
        else:
            imgui.text(name)
        if not opened and len(train.stops) > 1:
            following = railway.name_of(train.stops[train.onward(index)].cell)
            note = (
                "only refuels"
                if stop.refuel_only
                else f"{len(stop.cars)} {plural('car', len(stop.cars))} to {following}"
            )
            _note_before(note, right - HEAD_BUTTONS)
        imgui.same_line(right - HEAD_BUTTONS)
        imgui.begin_disabled(stop.refuel_only)
        _, stop.wait_for_load = imgui.checkbox("Wait for full load", stop.wait_for_load)
        imgui.end_disabled()
        tip(
            "It only refuels here, so it doesn't wait for a load"
            if stop.refuel_only
            else "Wait here, up to a month, for every car to fill",
            disabled=True,
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
            tip(f"{tip_text}, with its cars")
        imgui.same_line(right - 22)
        if imgui.small_button(fa.ICON_FA_TIMES):
            train.stops.pop(index)
            train.restart()
            return True, toggled, False
        tip("Remove this stop")
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
        tip(
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
            tip(
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
            return railway.car_tonnes(cargo, train.engine, asked) <= spare + EPSILON

        offers.sort(key=lambda o: (-(demand.monthly_want(o.good) > 0), -o.waiting, -o.gathered))
        left = imgui.get_cursor_pos_x()
        per_row = max(1, int(imgui.get_content_region_avail().x // (OFFER_SIZE[0] + 4)))
        for number, offer in enumerate(offers):
            next_good_slot(number, left, per_row, gap=4.0)
            wanted = demand.monthly_want(offer.good) > 0
            blocked = full or not fits(offer.good)
            hint = (
                f"{amount_of(offer.good, offer.waiting)} waiting, "
                f"{offer.gathered:.0f} given to the station in the last quarter"
                + (" - comes in on this train" if offer.good in carried else "")
                + (" - wanted at the next stop" if wanted else "")
                + adding_tip(full, railway.car_tonnes(offer.good, train.engine, asked), spare)
            )
            imgui.begin_disabled(blocked)
            if offer_chip(f"offer{number}", offer.good, f"{offer.waiting:.0f}", wanted, hint):
                railway.add_car(train, stop, offer.good, asked)
                train.cars_changed(index)
            spot(f"cargo:{index}:{offer.good.value}")
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                self._shown_map.good = offer.good
            imgui.end_disabled()
        for number, general in enumerate(generals, start=len(offers)):
            next_good_slot(number, left, per_row, gap=4.0)
            waiting = sum(waiting_here.held(good) for good in GENERAL_GOODS[general])
            hint = (
                f"{general.value}: a car that takes whichever of its goods pays best at the "
                f"next stop; {waiting:.0f} of them waiting here"
                + adding_tip(full, railway.car_tonnes(general, train.engine, asked), spare)
            )
            imgui.begin_disabled(full or not fits(general))
            if offer_chip(f"general{number}", general, f"{waiting:.0f}", waiting > 0, hint):
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
        tip(
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
    tip(
        f"What its cars weigh now, against the most its engine pulls;\nit runs at "
        f"{railway.pull(train):.0%} of its top speed, "
        f"{speed_share(railway, train, most):.0%} pulling the most"
        f"\nIt burns {railway.burn(train):.2g} {fuel_of(ENGINES[train.engine]).value} a cell, "
        f"{railway.burn(train) / ENGINES[train.engine].burn:.0%} of what it burns half loaded"
    )


def first_button(train: Train) -> None:
    """The button that puts a running train first in every queue for the way on, or, once it
    is, says so."""
    imgui.begin_disabled(not train.path or train.first)
    label = "First in line" if train.first else "Go first"
    if imgui.button(f"{fa.ICON_FA_FAST_FORWARD}  {label}"):
        train.first = True
    imgui.end_disabled()
    tip(FIRST_TIP, disabled=True)


def upgrade_button(railway: Railway, train: Train) -> None:
    """A button opening the menu of engines the train's engine can be swapped for, greyed out
    when there are none."""
    options = railway.engine_upgrades(train)
    imgui.begin_disabled(not options)
    if imgui.button(f"{fa.ICON_FA_ARROW_UP}  Upgrade"):
        imgui.open_popup("upgrade engine")
    imgui.end_disabled()
    tip(
        "Swap its engine for a newer one, keeping its stops and cars;\n"
        f"the old engine goes towards the price at {share_of(TRAIN_RESALE)} what it cost"
        if options
        else "No newer engine of its line is on sale",
        disabled=True,
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
        tip(summary, disabled=True)
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
            tip(f"{summary}\nAs many as the cash allows, in the order bought", disabled=True)
    imgui.end_popup()


def list_width(room: float, dragged: float | None) -> float:
    """How wide the table of trains is in a window this wide: as the player dragged it, or
    LIST_SHARE of it, but never under LEAST_LIST nor leaving the chosen train under
    LEAST_TRAIN, unless the window is too narrow for both."""
    wanted = room * LIST_SHARE if dragged is None else dragged
    return max(min(wanted, room - LEAST_TRAIN - SPLIT_WIDTH), min(LEAST_LIST, room / 2))


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


def _draw_tiles(railway: Railway, ride: Ride, onward: str) -> None:
    """The ride to the next stop in three tiles side by side, the network's upkeep on hover."""
    stations, track = railway.network_upkeep()
    upkeep = (
        f"\n\nNot counted in any one ride: the whole network's upkeep, stations "
        f"{dollars(stations)} and track {dollars(track)} a month"
    )
    gap = 8.0
    width = (imgui.get_content_region_avail().x - 2 * gap) / 3
    for number, (label, amount, hint) in enumerate(ride_tiles(ride, onward)):
        if number:
            imgui.same_line(0, gap)
        tile(label, signed(amount), BRASS if amount >= 0 else GONE, width)
        tip(hint + upkeep)


def _note_before(note: str, limit: float) -> None:
    """A dim note on the current line, left out if it would run past `limit`."""
    imgui.same_line()
    if imgui.get_cursor_pos_x() + imgui.calc_text_size(note).x < limit - 8:
        imgui.text_disabled(note)
