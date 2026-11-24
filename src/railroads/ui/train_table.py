"""The table of trains in the trains window, like a mail program's list of mail: a row a train
with its engine's picture framed by its category, its name, its schedule, first and last station, the cars its
route takes most of, the cars it pulls now and how full each is, its status and what its ride earns; filters over it by station, a stop on
the way or a kind of car, and a click on a column's head sorts by it."""

from collections.abc import Callable
from dataclasses import dataclass, field

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.goods import Good
from railroads.sim.grids import Cell
from railroads.sim.railway import NO_WAY_ON, Railway
from railroads.sim.train_names import (
    Category,
    cars_used_most,
    category,
    first_station,
    last_station,
)
from railroads.sim.trains import Car, Cargo, Train, unit_name
from railroads.ui.buy_train import PICTURE_BACK, EnginePicture
from railroads.ui.theme import (
    BRASS,
    GONE,
    LIST_TABLE_FLAGS,
    TEXT,
    TEXT_DIM,
    color_u32,
    colored,
    draw_good_icon,
    draw_small_car,
    tip,
)
from railroads.words import dollars, plural

# The frame round an engine's picture, by its train's category; normal trains have none.
CATEGORY_COLOURS = {
    Category.EXPRESS: (0.9, 0.32, 0.26, 1.0),
    Category.LOCAL: (0.38, 0.62, 0.9, 1.0),
    Category.CARGO: (0.78, 0.55, 0.25, 1.0),
}
# The box an engine's picture is fitted in, and how thick its category's frame is.
ICON_BOX = (40.0, 20.0)
FRAME_WIDTH = 2.0
ROW_HEIGHT = 26.0
# The size of a good's icon among the cars a route takes most of.
CAR_ICON = 18.0
# A car among those a train pulls now, filled by its load, and the gap between two.
WAGON = (11.0, 16.0)
WAGON_GAP = 1.0
# How a status naming the train ahead begins: the simulation names it as bought.
NAMING_STATUSES = ("Following ", "Waiting at a signal for ")
ANY = "Any"


@dataclass(frozen=True)
class Column:
    title: str
    # Its width, or for a column that stretches, its share of the room the others leave.
    width: float
    stretch: bool = False


COLUMNS = (
    Column("Train", 2.2, stretch=True),
    Column("Schedule", 1.4, stretch=True),
    Column("First stop", 96.0),
    Column("Last stop", 96.0),
    Column("Cars used most", 112.0),
    Column("Carrying", 150.0),
    Column("Status", 1.0, stretch=True),
    Column("This ride", 72.0),
)
TRAIN, SCHEDULE, FIRST, LAST, CARS, CARRYING, STATUS, RIDE = range(len(COLUMNS))
# The least of a good a train carries for it to show among what it carries, and how wide the
# table must be to show each train's first and last station: narrower, the columns are left out,
# a train's name holding them already.
SHOWN_LOAD = 0.5
ROOM_FOR_STOPS = 900.0


@dataclass
class TrainFilter:
    """Which trains the table shows: those starting or ending at a station, calling at one
    anywhere on the way, or taking on cars of a cargo; None lets every train through."""

    first: Cell | None = None
    last: Cell | None = None
    calls_at: Cell | None = None
    cargo: Cargo | None = None

    def keeps(self, train: Train) -> bool:
        return (
            (self.first is None or first_station(train) == self.first)
            and (self.last is None or last_station(train) == self.last)
            and (self.calls_at is None or any(s.cell == self.calls_at for s in train.stops))
            and (self.cargo is None or any(self.cargo in s.cars for s in train.stops))
        )

    @property
    def any(self) -> bool:
        return any(v is not None for v in (self.first, self.last, self.calls_at, self.cargo))


@dataclass(frozen=True)
class Row:
    train: Train
    title: str
    first: str
    last: str
    cars: list[tuple[Cargo, int]]
    status: str
    stuck: bool
    net: float | None
    carrying: tuple[tuple[Good, float], ...] = ()
    wagons: tuple[Car, ...] = ()
    schedule: str = ""


def carried(train: Train) -> tuple[tuple[Good, float], ...]:
    """What the train carries now, each good once with its load over all its cars, the most
    first."""
    loads: dict[Good, float] = {}
    for car in train.cars:
        if not car.dining:
            loads[car.good] = loads.get(car.good, 0.0) + car.load
    shown = [(good, load) for good, load in loads.items() if load >= SHOWN_LOAD]
    return tuple(sorted(shown, key=lambda pair: -pair[1]))


def shown_status(status: str, titles: dict[str, str]) -> str:
    """A train's status with the train it follows or waits for called by its title."""
    for lead in NAMING_STATUSES:
        if status.startswith(lead):
            name = status[len(lead) :]
            return lead + titles.get(name, name)
    return status


def train_rows(
    railway: Railway, titles: dict[str, str], net_of: Callable[[Train], float | None]
) -> list[Row]:
    rows = []
    schedules = railway.schedules
    for train in railway.trains:
        first, last = first_station(train), last_station(train)
        schedule = schedules.of(train)
        rows.append(
            Row(
                train,
                titles.get(train.name, train.name),
                "" if first is None else railway.name_of(first),
                "" if last is None else railway.name_of(last),
                cars_used_most(train),
                shown_status(train.status, titles),
                train.status in NO_WAY_ON or bool(railway.long_legs(train)),
                net_of(train),
                carried(train),
                tuple(train.cars),
                "" if schedule is None else schedules.title(schedule),
            )
        )
    return rows


def sort_rows(rows: list[Row], column: int, descending: bool) -> list[Row]:
    """Rows sorted by a column, names without regard to case, cars by the one used most, rides
    by what they earn with those not running last; ties keep the order trains were bought in."""
    keys: dict[int, Callable[[Row], object]] = {
        TRAIN: lambda r: r.title.casefold(),
        SCHEDULE: lambda r: r.schedule.casefold(),
        FIRST: lambda r: r.first.casefold(),
        LAST: lambda r: r.last.casefold(),
        CARS: lambda r: r.cars[0][0].value if r.cars else "~",
        CARRYING: lambda r: sum(load for _, load in r.carrying),
        STATUS: lambda r: r.status.casefold(),
    }
    if column == RIDE:
        running = [r for r in rows if r.net is not None]
        idle = [r for r in rows if r.net is None]
        return sorted(running, key=lambda r: r.net or 0.0, reverse=descending) + idle
    return sorted(rows, key=keys[column], reverse=descending)


def draw_engine_icon(
    draw_list: imgui.ImDrawList, picture: EnginePicture, train: Train, low: imgui.ImVec2
) -> None:
    """A train's engine as the map draws it, fitted in ICON_BOX on a dark ground, framed in its
    category's colour."""
    high = (low.x + ICON_BOX[0], low.y + ICON_BOX[1])
    draw_list.add_rect_filled(low, high, color_u32(PICTURE_BACK), 3.0)
    density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
    texture, (width, height) = picture(train.engine, density)
    room = (ICON_BOX[0] - 2 * FRAME_WIDTH - 2, ICON_BOX[1] - 2 * FRAME_WIDTH - 2)
    scale = min(room[0] / width, room[1] / height)
    left = low.x + (ICON_BOX[0] - width * scale) / 2
    top = low.y + (ICON_BOX[1] - height * scale) / 2
    draw_list.add_image(texture, (left, top), (left + width * scale, top + height * scale))
    kind = category(train)
    if kind is not None:
        draw_list.add_rect(low, high, color_u32(CATEGORY_COLOURS[kind]), 3.0, FRAME_WIDTH)


def engine_icon(picture: EnginePicture, train: Train) -> None:
    """draw_engine_icon in the line, named by its category on hover."""
    imgui.dummy(ICON_BOX)
    draw_engine_icon(imgui.get_window_draw_list(), picture, train, imgui.get_item_rect_min())
    kind = category(train)
    tip("A normal train" if kind is None else f"{kind.value} train")


@dataclass
class TrainTable:
    filter: TrainFilter = field(default_factory=TrainFilter)
    # The column sorted by and whether from the top down.
    sorted_by: int = TRAIN
    descending: bool = False

    def draw_filters(self, railway: Railway) -> None:
        """A list for each filter, showing only what some train has, and a button clearing
        them all."""
        trains = railway.trains
        firsts = {first_station(t) for t in trains} - {None}
        lasts = {last_station(t) for t in trains} - {None}
        calls = {s.cell for t in trains for s in t.stops}
        cargos = {c for t in trains for s in t.stops for c in s.cars}
        clear_width = imgui.calc_text_size(fa.ICON_FA_TIMES).x + 16
        labels = ("From", "To", "Calls at", "Cars")
        label_room = sum(imgui.calc_text_size(label).x + 14 for label in labels)
        width = max(70.0, (imgui.get_content_region_avail().x - label_room - clear_width) / 4)
        self.filter.first = _station_filter("From", self.filter.first, firsts, railway, width)
        imgui.same_line(0, 10)
        self.filter.last = _station_filter("To", self.filter.last, lasts, railway, width)
        imgui.same_line(0, 10)
        self.filter.calls_at = _station_filter(
            "Calls at", self.filter.calls_at, calls, railway, width
        )
        imgui.same_line(0, 10)
        self.filter.cargo = _cargo_filter(self.filter.cargo, cargos, width)
        imgui.same_line(0, 6)
        imgui.begin_disabled(not self.filter.any)
        if imgui.small_button(f"{fa.ICON_FA_TIMES}##clear filters"):
            self.filter = TrainFilter()
        imgui.end_disabled()
        tip("Show every train", disabled=True)

    def draw(
        self, rows: list[Row], chosen: str | None, picture: EnginePicture
    ) -> tuple[str | None, bool]:
        """The table of the rows the filters keep, sorted; returns the train clicked, if any,
        and whether it was clicked twice."""
        clicked, twice = None, False
        flags = (
            LIST_TABLE_FLAGS
            | imgui.TableFlags_.sortable.value
            | imgui.TableFlags_.resizable.value
            | imgui.TableFlags_.borders_outer_h.value
            | imgui.TableFlags_.hideable.value
        )
        wide = imgui.get_content_region_avail().x >= ROOM_FOR_STOPS
        if not imgui.begin_table("trains", len(COLUMNS), flags):
            return None, False
        imgui.table_setup_scroll_freeze(0, 1)
        for number, column in enumerate(COLUMNS):
            own = (
                imgui.TableColumnFlags_.width_stretch.value
                if column.stretch
                else imgui.TableColumnFlags_.width_fixed.value
            )
            if number == TRAIN:
                own |= imgui.TableColumnFlags_.default_sort.value
            imgui.table_setup_column(column.title, own, column.width)
        imgui.table_set_column_enabled(FIRST, wide)
        imgui.table_set_column_enabled(LAST, wide)
        imgui.table_headers_row()
        specs = imgui.table_get_sort_specs()
        if specs is not None and specs.specs_count:
            spec = specs.get_specs(0)
            self.sorted_by = spec.column_index
            self.descending = spec.get_sort_direction() == imgui.SortDirection.descending
        shown = sort_rows(
            [r for r in rows if self.filter.keeps(r.train)], self.sorted_by, self.descending
        )
        clipper = imgui.ListClipper()
        clipper.begin(len(shown), ROW_HEIGHT)
        while clipper.step():
            for row in shown[clipper.display_start : clipper.display_end]:
                if _draw_row(row, row.train.name == chosen, picture):
                    clicked = row.train.name
                    twice = imgui.is_mouse_double_clicked(imgui.MouseButton_.left)
        clipper.end()
        imgui.end_table()
        if not shown:
            imgui.text_disabled(
                "No train matches these filters." if rows else "You have no trains yet."
            )
        return clicked, twice


def _draw_row(row: Row, picked: bool, picture: EnginePicture) -> bool:
    """One train's row, a click anywhere on it choosing the train; returns whether it was
    clicked."""
    imgui.table_next_row(0, ROW_HEIGHT)
    imgui.table_next_column()
    flags = (
        imgui.SelectableFlags_.span_all_columns.value
        | imgui.SelectableFlags_.allow_double_click.value
    )
    clicked = imgui.selectable(f"##{row.train.name}", picked, flags, (0, ROW_HEIGHT))[0]
    low = imgui.get_item_rect_min()
    draw_list = imgui.get_window_draw_list()
    icon_top = low.y + (ROW_HEIGHT - ICON_BOX[1]) / 2
    draw_engine_icon(draw_list, picture, row.train, imgui.ImVec2(low.x + 2, icon_top))
    text_top = low.y + (ROW_HEIGHT - imgui.get_text_line_height()) / 2
    colour = BRASS if picked else TEXT
    draw_list.add_text((low.x + ICON_BOX[0] + 10, text_top), color_u32(colour), row.title)
    _cell(SCHEDULE, row.schedule or "-", TEXT_DIM)
    _cell(FIRST, row.first or "-", TEXT_DIM if not row.first else None)
    _cell(LAST, row.last or "-", TEXT_DIM if not row.last else None)
    imgui.table_set_column_index(CARS)
    _draw_cars(row.cars)
    imgui.table_set_column_index(CARRYING)
    _draw_wagons(row.wagons)
    _cell(STATUS, row.status, GONE if row.stuck else TEXT_DIM)
    if row.net is None:
        _cell(RIDE, "-", TEXT_DIM)
    else:
        _cell(RIDE, dollars(row.net), BRASS if row.net >= 0 else GONE)
        tip("What its ride to the next stop earns after its costs")
    return clicked


def _cell(column: int, text: str, colour: tuple[float, float, float, float] | None = None) -> None:
    imgui.table_set_column_index(column)
    imgui.set_cursor_pos_y(
        imgui.get_cursor_pos_y() + (ROW_HEIGHT - imgui.get_text_line_height()) / 2
    )
    if colour is None:
        imgui.text(text)
    else:
        colored(colour, text)


def _draw_cars(cars: list[tuple[Cargo, int]]) -> None:
    """The cargos a route takes most cars of, each an icon and how many cars."""
    if not cars:
        _cell(CARS, "no cars", TEXT_DIM)
        return
    top = imgui.get_cursor_pos_y() + (ROW_HEIGHT - CAR_ICON) / 2
    for number, (cargo, count) in enumerate(cars):
        if number:
            imgui.same_line(0, 6)
        imgui.set_cursor_pos_y(top)
        imgui.dummy((CAR_ICON, CAR_ICON))
        draw_good_icon(imgui.get_window_draw_list(), cargo, imgui.get_item_rect_min(), CAR_ICON)
        tip(f"{cargo.value}: {count} {plural('car', count)} over its route")
        imgui.same_line(0, 1)
        imgui.set_cursor_pos_y(top + (CAR_ICON - imgui.get_text_line_height()) / 2)
        imgui.text_disabled(f"{count}")


def _draw_wagons(wagons: tuple[Car, ...]) -> None:
    """The cars a train pulls now, each filled from the bottom by its share of what it holds."""
    if not wagons:
        _cell(CARRYING, "no cars", TEXT_DIM)
        return
    imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + (ROW_HEIGHT - WAGON[1]) / 2)
    draw_list = imgui.get_window_draw_list()
    for number, car in enumerate(wagons):
        if number:
            imgui.same_line(0, WAGON_GAP)
        imgui.dummy(WAGON)
        low = imgui.get_item_rect_min()
        good = None if car.dining else car.good
        draw_small_car(draw_list, good, car.load, car.capacity, (low.x, low.y), WAGON)
        if car.dining:
            tip("Dining car")
        else:
            held = f"{car.load:.0f} of {car.capacity:.0f} {unit_name(car.good, car.capacity)}"
            tip(f"{car.good.value}: {held}")


def _station_filter(
    label: str, picked: Cell | None, stations: set[Cell], railway: Railway, width: float
) -> Cell | None:
    imgui.align_text_to_frame_padding()
    imgui.text_disabled(label)
    imgui.same_line(0, 4)
    imgui.set_next_item_width(width)
    shown = ANY if picked is None else railway.name_of(picked)
    if imgui.begin_combo(f"##{label}", shown):
        if imgui.selectable(ANY, picked is None)[0]:
            picked = None
        for station in sorted(stations, key=lambda s: railway.name_of(s).casefold()):
            if imgui.selectable(railway.name_of(station), station == picked)[0]:
                picked = station
        imgui.end_combo()
    return picked


def _cargo_filter(picked: Cargo | None, cargos: set[Cargo], width: float) -> Cargo | None:
    imgui.align_text_to_frame_padding()
    imgui.text_disabled("Cars")
    imgui.same_line(0, 4)
    imgui.set_next_item_width(width)
    if imgui.begin_combo("##Cars", ANY if picked is None else picked.value):
        if imgui.selectable(ANY, picked is None)[0]:
            picked = None
        for cargo in sorted(cargos, key=lambda c: c.value):
            if imgui.selectable(cargo.value, cargo == picked)[0]:
                picked = cargo
        imgui.end_combo()
    return picked
