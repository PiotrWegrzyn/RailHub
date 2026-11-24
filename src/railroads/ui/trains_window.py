"""The trains window, in two tabs. Trains: every train in a list on the left; the chosen one's
cars and its schedule's stops on the right, a card for each stop with the cars it pulls on to the
next, the goods made or waiting there to click into cars, those the next stop wants outlined, and
what the next stop wants; and beside the cards a map to add stops on, which marks where a hovered
good is wanted (railroads.ui.route_map). The chosen train's engine can be upgraded to a newer one
of its line, alone or with every train of that line with an older engine. Schedules: every
schedule in a list (railroads.sim.schedules); the chosen one's trains in a table, how its
refuelling stops are set, and its stops as cards and on the map, as for a train."""

import math
from collections import Counter
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
from railroads.sim.schedules import MOST_CARS, Schedule
from railroads.sim.trains import (
    ENGINES,
    SLOWDOWN,
    Cargo,
    Stop,
    Train,
    amount_of,
    unit_name,
)
from railroads.ui.buy_train import BuyScreen, EnginePicture
from railroads.ui.demand_view import draw_needs, needs_of
from railroads.ui.ornate import card, ornate_window, section, tile
from railroads.ui.route_map import GroundPicture, RouteMap, add_stop, can_stop_at
from railroads.ui.spotlight import spot
from railroads.ui.station_offers import Offer, Supplies, goods_to_load
from railroads.ui.theme import (
    BORDER,
    BRASS,
    CHIP_SIZE,
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
from railroads.ui.train_table import (
    ICON_BOX,
    ROW_HEIGHT,
    TrainTable,
    draw_cargo_counts,
    engine_icon,
    shown_status,
    train_rows,
)
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
TRAINS, SCHEDULES = "Trains", "Schedules"
# How many rows of trains a schedule's table shows before it scrolls.
SCHEDULE_ROWS = 6


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
    # Whether the goods on the stop cards come in the order the next stop wants them most, or
    # in the order the most of them waits at the stop.
    by_wanted: bool = True
    # What each train's cars would earn at its destination, and its ride to its next stop,
    # worked out at most every WORTH_EVERY seconds, by train name.
    _worth: Fresh[str, list[float]] = field(default_factory=lambda: Fresh(WORTH_EVERY))
    _rides: Fresh[str, Ride | None] = field(default_factory=lambda: Fresh(WORTH_EVERY))
    map: RouteMap = field(default_factory=RouteMap)
    # The screen trains are bought on.
    buy: BuyScreen = field(default_factory=BuyScreen)
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
    # Stop cards the player folded away or opened, by schedule and stop; the rest are open but
    # for those with cars set, or that the game put in.
    _folded: dict[tuple[str, int], bool] = field(default_factory=dict)
    # The tab shown, and the one to show next, picked by a button rather than its tab.
    tab: str = TRAINS
    _pick_tab: str | None = None
    # The schedule shown on the Schedules tab, how wide the player dragged its list, the table
    # of its trains and the map of its stops.
    chosen_schedule: str | None = None
    schedules_width: float | None = None
    schedule_table: TrainTable = field(default_factory=TrainTable)
    schedule_map: RouteMap = field(default_factory=RouteMap)
    # The schedule whose stops are drawn now, and the train they are drawn for, if any.
    _editing: Schedule | None = None
    _riders: list[Train] = field(default_factory=list)

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
                self._draw_tabs(railway, supplies, ground, picture)
        self.open = still_open
        bought = self.buy.draw(
            railway,
            engines_by_kind(railway.engines()),
            engine_summary,
            picture,
        )
        if bought is not None:
            self._bought(railway, bought)

    def _bought(self, railway: Railway, name: str) -> None:
        """Picks a train just bought, and shows its schedule when that has no stops yet."""
        self.chosen = name
        train = next(t for t in railway.trains if t.name == name)
        schedule = railway.schedules.of(train)
        if schedule is not None and len(schedule.stops) < 2:
            self.chosen_schedule = schedule.name
            self._pick_tab = SCHEDULES

    def _draw_tabs(
        self,
        railway: Railway,
        supplies: Supplies,
        ground: GroundPicture,
        picture: EnginePicture,
    ) -> None:
        if not imgui.begin_tab_bar("trains tabs"):
            return
        for tab in (TRAINS, SCHEDULES):
            flags = imgui.TabItemFlags_.set_selected.value if self._pick_tab == tab else 0
            if imgui.begin_tab_item(tab, None, flags)[0]:
                self.tab = tab
                if tab == TRAINS:
                    self._draw_panes(railway, supplies, ground, picture)
                else:
                    self._draw_schedule_panes(railway, supplies, ground, picture)
                imgui.end_tab_item()
        self._pick_tab = None
        imgui.end_tab_bar()

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
            dragged = _draw_split(width)
            if dragged is not None:
                self.list_width = dragged
        imgui.begin_child("train", (0, 0))
        if train is None:
            _centered_hint("Buy a train, then give its schedule two stations joined by track.")
        else:
            self._draw_maximize()
            imgui.same_line(0, 10)
            self._draw_train(railway, train, supplies, ground, self.map, picture)
        imgui.end_child()

    def _draw_schedule_panes(
        self,
        railway: Railway,
        supplies: Supplies,
        ground: GroundPicture,
        picture: EnginePicture,
    ) -> None:
        """The list of schedules on the left and the chosen one on the right."""
        schedules = railway.schedules
        schedule = None if self.chosen_schedule is None else schedules.named(self.chosen_schedule)
        if schedule is None and schedules.all:
            schedule = schedules.all[0]
            self.chosen_schedule = schedule.name
        width = list_width(imgui.get_content_region_avail().x, self.schedules_width)
        imgui.begin_child("schedules", (width, 0))
        self._draw_schedule_list(railway, schedule)
        imgui.end_child()
        dragged = _draw_split(width)
        if dragged is not None:
            self.schedules_width = dragged
        imgui.begin_child("schedule", (0, 0))
        if schedule is None:
            _centered_hint(
                "Buy a train: it runs a schedule of its own, which more trains can join."
            )
        else:
            self._draw_schedule(railway, schedule, supplies, ground, picture)
        imgui.end_child()

    def _draw_schedule_list(self, railway: Railway, chosen: Schedule | None) -> None:
        """The buttons making a schedule and buying a train for the one chosen, over a table of
        every schedule: its trains, its stops and what its trains' rides earn."""
        schedules = railway.schedules
        if imgui.button(f"{fa.ICON_FA_PLUS}  New schedule"):
            self.chosen_schedule = schedules.new().name
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_SHOPPING_CART}  Buy a train"):
            self.buy.open_for(chosen)
        tip("Buy a train for the schedule chosen")
        imgui.spacing()
        members: dict[str, list[Train]] = {}
        for train in railway.trains:
            members.setdefault(train.schedule, []).append(train)
        flags = (
            imgui.TableFlags_.row_bg.value
            | imgui.TableFlags_.scroll_y.value
            | imgui.TableFlags_.borders_outer_h.value
        )
        if not imgui.begin_table("schedules", 5, flags):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        stretch, fixed = (
            imgui.TableColumnFlags_.width_stretch.value,
            imgui.TableColumnFlags_.width_fixed.value,
        )
        imgui.table_setup_column("Schedule", stretch, 1.2)
        imgui.table_setup_column("Carries", stretch, 1.0)
        imgui.table_setup_column("Trains", fixed, 52.0)
        imgui.table_setup_column("Stops", fixed, 44.0)
        imgui.table_setup_column("Last rounds", fixed, 90.0)
        imgui.table_headers_row()
        for schedule in schedules.all:
            trains = members.get(schedule.name, [])
            imgui.table_next_row(0, ROW_HEIGHT)
            imgui.table_next_column()
            top = imgui.get_cursor_pos_y()
            title = schedules.title(schedule)
            flags = imgui.SelectableFlags_.span_all_columns.value
            if imgui.selectable(f"##{schedule.name}", schedule is chosen, flags, (0, ROW_HEIGHT))[
                0
            ]:
                self.chosen_schedule = schedule.name
            imgui.same_line(0, 0)
            imgui.set_cursor_pos_y(top + (ROW_HEIGHT - imgui.get_text_line_height()) / 2)
            if schedule.problem:
                colored(GONE, f"{fa.ICON_FA_EXCLAMATION_TRIANGLE}  {title}")
                tip(schedule.problem)
            elif schedule is chosen:
                colored(BRASS, title)
            else:
                imgui.text(title)
            imgui.table_next_column()
            draw_cargo_counts(carries(schedule), "taken on over its round")
            for number, text in ((2, f"{len(trains)}"), (3, f"{len(schedule.own_stops())}")):
                imgui.table_set_column_index(number)
                imgui.set_cursor_pos_y(top + (ROW_HEIGHT - imgui.get_text_line_height()) / 2)
                imgui.text(text)
            imgui.table_next_column()
            imgui.set_cursor_pos_y(top + (ROW_HEIGHT - imgui.get_text_line_height()) / 2)
            rounds = [t.last_round for t in trains if t.last_round is not None]
            if rounds:
                total = sum(rounds)
                colored(BRASS if total >= 0 else GONE, dollars(total))
                tip(
                    "What its trains' last whole rounds earned together: their cargo's worth\n"
                    "where it was unloaded, less their fuel and car trips"
                )
            else:
                imgui.text_disabled("-")
        imgui.end_table()

    def _draw_schedule(
        self,
        railway: Railway,
        schedule: Schedule,
        supplies: Supplies,
        ground: GroundPicture,
        picture: EnginePicture,
    ) -> None:
        """A schedule: its name and buttons, how its refuelling stops are set, the table of its
        trains, and its stops as cards beside the map."""
        schedules = railway.schedules
        trains = schedules.trains(schedule)
        title = schedules.title(schedule)
        imgui.align_text_to_frame_padding()
        colored(BRASS, title)
        imgui.same_line(0, 4)
        self._rename_schedule(schedule, title)
        imgui.same_line(0, 16)
        if imgui.button(f"{fa.ICON_FA_SHOPPING_CART}  Buy a train"):
            self.buy.open_for(schedule)
        tip("Buy a train that runs this schedule")
        imgui.same_line()
        self._draw_add_train(railway, schedule)
        imgui.same_line()
        imgui.begin_disabled(bool(trains))
        if danger_button(f"{fa.ICON_FA_TRASH}  Delete"):
            schedules.remove(schedule)
            self.chosen_schedule = None
            imgui.end_disabled()
            return
        imgui.end_disabled()
        tip("Move or sell its trains first" if trains else "Delete this schedule", disabled=True)
        self._draw_refuel_mode(railway, schedule)
        problem = schedule.problem if trains else self._none_on_sale(railway, schedule)
        if problem:
            imgui.push_text_wrap_pos(0.0)
            colored(GONE, f"{fa.ICON_FA_EXCLAMATION_TRIANGLE}  {problem}")
            imgui.pop_text_wrap_pos()
        section(f"{len(trains)} {plural('train', len(trains))}")
        rows = [
            row
            for row in train_rows(railway, railway.titles())
            if row.train.schedule == schedule.name
        ]
        height = (min(max(len(rows), 1), SCHEDULE_ROWS) + 1.8) * ROW_HEIGHT
        imgui.begin_child("schedule trains", (0, height))
        clicked, _ = self.schedule_table.draw(rows, None, picture, picking=False, schedules=False)
        imgui.end_child()
        if clicked is not None:
            self.chosen = clicked
            self.maximized = False
            self._pick_tab = TRAINS
        view = _view(schedule, schedules.biggest_engine(schedule))
        self.schedule_map.good = None
        self._shown_map = self.schedule_map
        map_width = imgui.get_content_region_avail().x * MAP_SHARE
        imgui.begin_child("route", (-map_width, 0))
        section("Stops")
        self._draw_sort_choice()
        self._draw_stops(railway, view, supplies, schedule, riders=trains)
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("map", (0, 0), imgui.ChildFlags_.borders.value)
        self.schedule_map.draw(railway, view, ground, supplies, trains)
        imgui.end_child()
        schedules.settle(schedule)

    def _none_on_sale(self, railway: Railway, schedule: Schedule) -> str:
        """That no engine on sale could run the schedule, and why the one carrying fuel the
        furthest couldn't; "" if one could."""
        engines = sorted(railway.engines(), key=lambda name: -ENGINES[name].tank)
        reasons = [self.buy.cannot_run(railway, schedule, name) for name in engines]
        if not reasons or not all(reasons):
            return ""
        return f"No engine on sale could run it: {reasons[0]}"

    @staticmethod
    def _draw_add_train(railway: Railway, schedule: Schedule) -> None:
        """A list of the trains running other schedules, one picked moving to this one."""
        schedules = railway.schedules
        others = [train for train in railway.trains if train.schedule != schedule.name]
        imgui.set_next_item_width(170)
        imgui.begin_disabled(not others)
        if imgui.begin_combo("##add train", f"{fa.ICON_FA_PLUS}  Add a train"):
            titles = railway.titles()
            for train in others:
                own = schedules.of(train)
                came = "" if own is None else f"  (from {schedules.title(own)})"
                if imgui.selectable(f"{titles.get(train.name, train.name)}{came}##{train.name}")[0]:
                    schedules.join(train, schedule)
            imgui.end_combo()
        imgui.end_disabled()
        tip(
            "Move a train here from its schedule: it finishes its round first, then runs to\n"
            "this one's first stop, off the map for as long as the way there takes",
            disabled=True,
        )

    @staticmethod
    def _draw_refuel_mode(railway: Railway, schedule: Schedule) -> None:
        """Who sets the refuelling stops, the game or the player, and a button giving them back
        to the game."""
        if schedule.manual:
            imgui.align_text_to_frame_padding()
            imgui.text_disabled("Refuelling stops set by hand")
            imgui.same_line()
            if imgui.small_button("Let the game set them"):
                railway.schedules.set_manual(schedule, False)
            tip(
                "Takes out every refuelling stop and puts in those its weakest train needs,\n"
                "planned again whenever its stops, its trains or the track change"
            )
            return
        if schedule.tank:
            imgui.text_disabled(
                f"Refuelling stops set by the game for trains carrying fuel for "
                f"{schedule.tank:.0f} cells"
            )
        else:
            imgui.text_disabled("Refuelling stops set by the game: none are needed now")
        tip(
            "The fewest stations on the way for its weakest train that can make the round,\n"
            "planned again whenever its stops, its trains or the track change.\n"
            "Changing one of them sets them all by hand from then on."
        )

    def _rename_schedule(self, schedule: Schedule, title: str) -> None:
        """A small pencil opening a box to name the schedule, or to give it back its stations'
        name."""
        if imgui.small_button(f"{fa.ICON_FA_EDIT}##rename schedule"):
            self._naming = schedule.label or title
            imgui.open_popup("rename schedule")
        tip("Name this schedule")
        if not imgui.begin_popup("rename schedule"):
            return
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here()
        imgui.set_next_item_width(260)
        entered, self._naming = imgui.input_text(
            "##name", self._naming, imgui.InputTextFlags_.enter_returns_true.value
        )
        imgui.same_line()
        if imgui.button("Name it") or entered:
            schedule.label = self._naming.strip()
            imgui.close_current_popup()
        imgui.begin_disabled(not schedule.label)
        if imgui.button("Use its stations' name"):
            schedule.label = ""
            imgui.close_current_popup()
        imgui.end_disabled()
        imgui.end_popup()

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
        """The Buy a train button, the schedule filter and what to do with the trains picked,
        always at the top, over the table of trains, which scrolls on its own."""
        if imgui.button(f"{fa.ICON_FA_PLUS}  Buy a train"):
            chosen = next((t for t in railway.trains if t.name == self.chosen), None)
            self.buy.open_for(None if chosen is None else railway.schedules.of(chosen))
        spot("buy_train")
        imgui.same_line(0, 12)
        self.table.draw_filter(railway)
        self._draw_picked(railway)
        imgui.spacing()
        rows = train_rows(railway, railway.titles())
        clicked, twice = self.table.draw(rows, self.chosen, picture)
        if clicked is not None:
            self.chosen = clicked
            self.maximized = twice

    def _draw_picked(self, railway: Railway) -> None:
        """Buttons picking every train shown or none, and upgrading or selling the trains
        picked all at once."""
        names = {train.name for train in railway.trains}
        self.table.picked &= names
        picked = [train for train in railway.trains if train.name in self.table.picked]
        if imgui.small_button("Pick all shown"):
            self.table.picked |= set(self.table.shown)
        imgui.same_line()
        imgui.begin_disabled(not picked)
        if imgui.small_button("Pick none"):
            self.table.picked.clear()
        imgui.same_line(0, 14)
        imgui.text_disabled(f"{len(picked)} picked")
        imgui.same_line(0, 14)
        upgrades = sorted(
            {name for train in picked for name in railway.engine_upgrades(train)},
            key=lambda name: (ENGINES[name].year, name),
        )
        imgui.begin_disabled(not upgrades)
        if imgui.small_button(f"{fa.ICON_FA_ARROW_UP}  Upgrade"):
            imgui.open_popup("upgrade picked")
        imgui.end_disabled()
        tip(
            "Swap the engines of the trains picked for a newer one of their line"
            if upgrades
            else "No newer engine is on sale for the trains picked",
            disabled=True,
        )
        imgui.same_line()
        resale = sum(ENGINES[train.engine].price * TRAIN_RESALE for train in picked)
        if danger_button(f"{fa.ICON_FA_TRASH}  Sell"):
            imgui.open_popup("sell picked")
        tip(f"Sell the trains picked for {dollars(resale)}", disabled=True)
        imgui.end_disabled()
        if imgui.begin_popup("upgrade picked"):
            for name in upgrades:
                trains = [train for train in picked if name in railway.engine_upgrades(train)]
                total = sum(railway.swap_price(train, name) for train in trains)
                label = (
                    f"{ENGINES[name].title}: {len(trains)} {plural('train', len(trains))}"
                    f"  {dollars(total)}##{name}"
                )
                if imgui.selectable(label, False)[0]:
                    for train in trains:
                        railway.swap_engine(train, name)
                tip(
                    "\n".join(
                        [
                            engine_summary(ENGINES[name], railway.fuel_cost(name)),
                            *wagons_note(railway, name),
                            "As many as the cash allows, in the order bought",
                        ]
                    )
                )
            imgui.end_popup()
        if imgui.begin_popup("sell picked"):
            imgui.text(f"Sell {len(picked)} {plural('train', len(picked))} for {dollars(resale)}?")
            if danger_button("Sell them"):
                for train in picked:
                    railway.sell(train)
                self.table.picked.clear()
                if self.chosen not in {train.name for train in railway.trains}:
                    self.chosen = None
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Keep them"):
                imgui.close_current_popup()
            imgui.end_popup()

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
        schedule = railway.schedules.of(train)
        if schedule is not None:
            self._draw_schedule_line(railway, train, schedule)
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
        map_width = imgui.get_content_region_avail().x * MAP_SHARE
        imgui.begin_child("route", (-map_width, 0))
        section("Its stops")
        route_map.good = _draw_calls(railway, train)
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("map", (0, 0), imgui.ChildFlags_.borders.value)
        route_map.draw(railway, train, ground, supplies, [train], adding=False)
        imgui.end_child()
        if schedule is not None:
            railway.schedules.settle(schedule)

    def _draw_schedule_line(self, railway: Railway, train: Train, schedule: Schedule) -> None:
        """The schedule the train runs, in a list moving it to another, and a button showing it."""
        schedules = railway.schedules
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Schedule")
        imgui.same_line(0, 8)
        imgui.set_next_item_width(280)
        if imgui.begin_combo("##schedule", schedules.title(schedule)):
            for other in schedules.all:
                picked = imgui.selectable(
                    f"{schedules.title(other)}##{other.name}", other is schedule
                )
                if picked[0] and other is not schedule:
                    schedules.join(train, other)
            imgui.end_combo()
        tip(
            "Move it to another schedule: it finishes its round first, then runs to that one's\n"
            "first stop, off the map for as long as the way there takes"
        )
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_LIST}  Open schedule"):
            self.chosen_schedule = schedule.name
            self._pick_tab = SCHEDULES
        if train.taken != (schedule.name, schedule.version) and train.away <= 0:
            imgui.same_line(0, 10)
            imgui.text_disabled("takes up the schedule's new stops after this round")

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

    def _draw_sort_choice(self) -> None:
        """Which order the goods to load at each stop and those the next stop wants come in,
        the same in both."""
        imgui.text_disabled("Goods by")
        imgui.same_line()
        if imgui.radio_button("what the next stop wants most", self.by_wanted):
            self.by_wanted = True
        imgui.same_line()
        if imgui.radio_button("what waits here most", not self.by_wanted):
            self.by_wanted = False

    def _draw_stops(
        self,
        railway: Railway,
        train: Train,
        supplies: Supplies,
        schedule: Schedule,
        riders: list[Train],
    ) -> None:
        """The stop cards of a schedule's stops through `train` standing in for it (_view), and
        a list adding one; under each stop's cars, how many of them `riders` take on."""
        self._editing, self._riders = schedule, riders
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
            removed = self._draw_stop(railway, train, index, stop, supplies)
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
            imgui.text_disabled("A schedule needs a second stop to run.")
        else:
            imgui.text_disabled("or click one on the map")

    def _draw_stop(
        self,
        railway: Railway,
        train: Train,
        index: int,
        stop: Stop,
        supplies: Supplies,
    ) -> bool:
        """One stop as a card: where it is, the cars the train takes on from it, the goods it
        has to load, and what the next stop wants; returns whether it was removed or moved."""
        with card("stop"):
            key = (train.name, index)
            opened = self._folded.setdefault(key, not (stop.cars or stop.auto))
            removed, toggled, moved = self._draw_stop_head(railway, train, index, stop, opened)
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
                self._draw_last_ride(index, railway.name_of(following.cell))
                self._draw_offers(railway, train, index, stop, demand, supplies)
                imgui.align_text_to_frame_padding()
                imgui.text_disabled(f"{railway.name_of(following.cell)} wants")
                imgui.same_line(CARD_LABEL)
                offers, _ = self._offers(railway, train, index, stop, supplies)
                goods = [offer.good for offer in sort_offers(offers, demand, self.by_wanted)]
                needs = needs_of(demand, goods)
                if self.by_wanted:
                    needs.sort(key=lambda need: -demand.record(need).left)
                draw_needs(demand, needs)
        return removed

    def _draw_stop_head(
        self, railway: Railway, train: Train, index: int, stop: Stop, opened: bool
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
        imgui.text(railway.name_of(stop.cell))
        if stop.auto:
            tip("The game put this refuelling stop in; changing it sets them all by hand")
        if not opened and len(train.stops) > 1:
            following = railway.name_of(train.stops[train.onward(index)].cell)
            note = (
                ("refuels, set by the game" if stop.auto else "only refuels")
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
                self._set_by_hand(railway, stop, train.stops[other])
                self._move_stop(train, index, other)
                moved = True
            imgui.end_disabled()
            tip(f"{tip_text}, with its cars")
        imgui.same_line(right - 22)
        if imgui.small_button(fa.ICON_FA_TIMES):
            self._set_by_hand(railway, stop)
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

    def _set_by_hand(self, railway: Railway, *stops: Stop) -> None:
        """Sets the refuelling stops of the schedule drawn now by hand, once the player changes
        one the game put in."""
        if self._editing is not None and any(stop.auto for stop in stops):
            railway.schedules.set_manual(self._editing, True)

    def _draw_refuel(self, railway: Railway, stop: Stop, following: Stop) -> bool:
        """The checkbox making a stop one the train only refuels at, for a leg longer than its
        fuel lasts, and what that means; returns whether the stop is one."""
        changed, refuel_only = imgui.checkbox("Only refuel here", stop.refuel_only)
        if changed:
            self._set_by_hand(railway, stop)
            stop.refuel_only = refuel_only
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
        most = self._most_cars(railway, train)
        left = imgui.get_cursor_pos_x()
        per_row = max(1, int((imgui.get_content_region_avail().x + 4) // (CHIP_SIZE[0] + 4)))

        def place(slot: int) -> None:
            if slot % per_row:
                imgui.same_line(0, 4)
            elif slot:
                caption_room()
                imgui.set_cursor_pos_x(left)

        for car_index, (cargo, asked) in enumerate(stop.asked()):
            place(car_index)
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
            place(slot)
            empty_car_slot()
        wrapped = most > per_row
        if wrapped:
            caption_room()
            imgui.set_cursor_pos_x(left)
        row_top = imgui.get_cursor_pos_y()
        if stop.cars:
            if not wrapped:
                imgui.same_line(0, 10)
            if imgui.small_button("Clear"):
                stop.clear_cars()
                train.cars_changed(index)
            imgui.same_line(0, 10)
            planned = railway.stop_tonnes(train, stop)
            if self._editing is not None:
                imgui.text_disabled(f"{tonnes_text(planned)} t full")
                tip(
                    "What these cars weigh full. Each train takes them from the left while its\n"
                    "engine has room for them and can pull them"
                )
            else:
                imgui.text_disabled(
                    f"{tonnes_text(planned)} / {tonnes_text(railway.max_pull(train.engine))} t full"
                )
                tip(
                    "What these cars weigh full, against the most its engine pulls;\n"
                    f"full, it runs at {speed_share(railway, train, planned):.0%} of its top speed"
                )
        if wrapped:
            imgui.set_cursor_pos_y(row_top + imgui.get_frame_height_with_spacing())
        caption_room()
        self._draw_taken(railway, stop)

    def _draw_last_ride(self, index: int, following: str) -> None:
        """What the last ride from a schedule's stop to the next earned, once one was made on its
        stops as they are."""
        schedule = self._editing
        if schedule is None or schedule.rides_version != schedule.version:
            return
        earned = schedule.rides.get(index)
        if earned is None:
            return
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Last ride")
        imgui.same_line(CARD_LABEL)
        colored(BRASS if earned >= 0 else GONE, dollars(earned))
        tip(
            f"What the last train's ride from here to {following} earned: its cargo's worth\n"
            "where it was unloaded, less its fuel and car trips"
        )

    def _most_cars(self, railway: Railway, train: Train) -> int:
        """How many cars a stop card takes: a schedule's as many as any engine pulls
        (MOST_CARS), a lone train's as many as its engine pulls."""
        return MOST_CARS if self._editing is not None else railway.most_cars(train)

    def _add_car(
        self, railway: Railway, train: Train, stop: Stop, cargo: Good | General, asked: int
    ) -> None:
        if self._editing is not None:
            railway.schedules.add_car(train, stop, cargo, asked)
        else:
            railway.add_car(train, stop, cargo, asked)

    def _draw_taken(self, railway: Railway, stop: Stop) -> None:
        """Under a schedule's stop's cars, how many of them its trains take on, by engine, when
        some take fewer than all."""
        if self._editing is None or not stop.cars or not self._riders:
            return
        taken: dict[str, int] = {}
        for rider in self._riders:
            if rider.engine not in taken:
                taken[rider.engine] = railway.schedules.taken(rider, stop)
        if all(count == len(stop.cars) for count in taken.values()):
            return
        note = "; ".join(
            f"trains behind a {ENGINES[engine].title} take the first {count}"
            for engine, count in sorted(taken.items(), key=lambda pair: pair[1])
        )
        note = note[0].upper() + note[1:]
        imgui.set_cursor_pos_x(CARD_LABEL)
        imgui.push_text_wrap_pos(0.0)
        colored(GONE, note)
        imgui.pop_text_wrap_pos()
        tip(
            "A train takes a stop's cars from the left while its engine has room for them and\n"
            "can pull them full; the rest are for bigger engines"
        )

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
        offers, carried = self._offers(railway, train, index, stop, supplies)
        waiting_here = railway.station_goods(stop.cell)
        generals = [general for general in General if railway.can_carry(train, general)]
        if not offers and not generals:
            imgui.text_disabled("nothing made or waiting here yet")
            return
        most = self._most_cars(railway, train)
        full = len(stop.cars) >= most
        spare = (
            math.inf
            if self._editing is not None
            else railway.max_pull(train.engine) - railway.stop_tonnes(train, stop)
        )

        def fits(cargo: Good | General) -> bool:
            return railway.car_tonnes(cargo, train.engine, asked) <= spare + EPSILON

        offers = sort_offers(offers, demand, self.by_wanted)
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
                self._add_car(railway, train, stop, offer.good, asked)
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
                self._add_car(railway, train, stop, general, asked)
                train.cars_changed(index)
            imgui.end_disabled()

    def _offers(
        self, railway: Railway, train: Train, index: int, stop: Stop, supplies: Supplies
    ) -> tuple[list[Offer], list[Good]]:
        """The goods the train can take on at a stop: those made or waiting there, then those
        its cars bring in from the stop before that are not; and those brought in alone."""
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
        return offers + [Offer(good, waiting_here.held(good), 0.0) for good in carried], carried

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


def sort_offers(offers: list[Offer], demand: StationDemand, by_wanted: bool) -> list[Offer]:
    """The goods to load at a stop in the order the next stop uses most of them a month, or in
    the order the most of them waits at the stop, each order breaking its ties by the other."""
    if by_wanted:
        return sorted(offers, key=lambda o: (-demand.monthly_want(o.good), -o.waiting, -o.gathered))
    return sorted(offers, key=lambda o: (-o.waiting, -o.gathered, -demand.monthly_want(o.good)))


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


def carries(schedule: Schedule, most: int = 4) -> list[tuple[Cargo, int]]:
    """The cargos a schedule takes on the most cars of over its round, most first, and how many
    cars of each."""
    counts = Counter(cargo for stop in schedule.own_stops() for cargo in stop.cars)
    return counts.most_common(most)


def _view(schedule: Schedule, engine: str) -> Train:
    """A train standing in for a schedule on the stop cards and the map: its stops are the
    schedule's, so what they change changes the schedule, and its engine sets how many cars
    and how much weight a stop can take."""
    return Train(schedule.name, stops=schedule.stops, engine=engine)


def _draw_calls(railway: Railway, train: Train) -> Good | None:
    """The stops the train calls at as it took them up from its schedule, to look at only: the
    one it heads for or stands at picked out, and the cars it takes on at each. Returns the good
    of a car hovered, to show on the map where it is wanted."""
    stops = train.stops
    if len(stops) < 2:
        imgui.text_disabled("Its schedule has no stops yet: open it to give it some.")
        return None
    hovered = None
    current = -1
    if train.away <= 0:
        current = (train.heading if train.path else train.heading - 1) % len(stops)
    flags = imgui.TableFlags_.row_bg.value | imgui.TableFlags_.borders_inner_h.value
    if not imgui.begin_table("calls", 3, flags):
        return None
    imgui.table_setup_column("#", imgui.TableColumnFlags_.width_fixed.value, 22)
    imgui.table_setup_column("Stop", imgui.TableColumnFlags_.width_stretch.value, 1.0)
    imgui.table_setup_column("Cars", imgui.TableColumnFlags_.width_stretch.value, 1.2)
    for index, stop in enumerate(stops):
        imgui.table_next_row(0, ROW_HEIGHT)
        middle = (ROW_HEIGHT - imgui.get_text_line_height()) / 2
        imgui.table_set_column_index(0)
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + middle)
        imgui.text_disabled(f"{index + 1}")
        imgui.table_set_column_index(1)
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + middle)
        name = railway.name_of(stop.cell)
        if index == current:
            colored(BRASS, f"{fa.ICON_FA_TRAIN}  {name}")
            imgui.same_line(0, 8)
            imgui.text_disabled("heading here" if train.path else "standing here")
        else:
            imgui.text(name)
        if stop.wait_for_load:
            imgui.same_line(0, 8)
            imgui.text_disabled("waits for a full load")
        imgui.table_set_column_index(2)
        following = railway.name_of(stops[train.onward(index)].cell)
        if stop.refuel_only or not stop.cars:
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + middle)
            imgui.text_disabled("only refuels" if stop.refuel_only else "no cars")
        else:
            cargo = draw_cargo_counts(list(Counter(stop.cars).items()), f"to {following}")
            if isinstance(cargo, Good):
                hovered = cargo
    imgui.end_table()
    return hovered


def _draw_split(width: float) -> float | None:
    """The bar between a list and what is chosen in it, on the line after the list; returns the
    list's width dragged to, if it is being dragged."""
    imgui.same_line(0, 0)
    imgui.invisible_button("split", (SPLIT_WIDTH, imgui.get_content_region_avail().y))
    if imgui.is_item_hovered() or imgui.is_item_active():
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
    dragged = width + imgui.get_io().mouse_delta.x if imgui.is_item_active() else None
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    middle = (low.x + high.x) / 2
    imgui.get_window_draw_list().add_line((middle, low.y), (middle, high.y), color_u32(BORDER), 1.0)
    imgui.same_line(0, 0)
    return dragged


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
