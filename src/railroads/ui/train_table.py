"""The table of trains in the trains window, like a mail program's list of mail: a row a train
with a box picking it for upgrading or selling with others, its engine's picture framed by its
category, its name, its schedule, where it is heading and the cars it pulls now with how full each
is; a filter over it by schedule, and a click on a column's head sorts by it."""

from collections.abc import Callable
from dataclasses import dataclass, field

from imgui_bundle import imgui

from railroads.sim.goods import Good
from railroads.sim.railway import NO_WAY_ON, Railway
from railroads.sim.train_names import Category, category
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
from railroads.words import plural

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
# The size of a good's icon among the cars a schedule takes on.
CAR_ICON = 18.0
# A car among those a train pulls now, filled by its load, and the gap between two.
WAGON = (11.0, 16.0)
WAGON_GAP = 1.0
# How a status naming the train ahead begins: the simulation names it as bought.
NAMING_STATUSES = ("Following ", "Waiting at a signal for ")
ANY = "Any schedule"


@dataclass(frozen=True)
class Column:
    title: str
    # Its width, or for a column that stretches, its share of the room the others leave.
    width: float
    stretch: bool = False


COLUMNS = (
    Column("", 22.0),
    Column("Train", 2.2, stretch=True),
    Column("Schedule", 1.6, stretch=True),
    Column("Heading to", 1.4, stretch=True),
    Column("Cars", 150.0),
)
PICK, TRAIN, SCHEDULE, HEADING, CARRYING = range(len(COLUMNS))
# The least of a good a train carries for it to count among what it carries.
SHOWN_LOAD = 0.5


@dataclass(frozen=True)
class Row:
    train: Train
    title: str
    schedule: str = ""
    # The stop it runs to now, or what it is doing while it stands.
    heading: str = ""
    stuck: bool = False
    carrying: tuple[tuple[Good, float], ...] = ()
    wagons: tuple[Car, ...] = ()


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


def heading_of(railway: Railway, train: Train, titles: dict[str, str]) -> str:
    """Where a running train is heading, the stop it runs to now; what it is doing while it
    stands or is off the map."""
    if train.path and train.stops and train.away <= 0:
        stop = train.stops[train.heading % len(train.stops)]
        name = railway.name_of(stop.cell)
        return f"{name} to refuel" if stop.refuel_only else name
    return shown_status(train.status, titles)


def train_rows(railway: Railway, titles: dict[str, str]) -> list[Row]:
    rows = []
    schedules = railway.schedules
    for train in railway.trains:
        schedule = schedules.of(train)
        rows.append(
            Row(
                train,
                titles.get(train.name, train.name),
                "" if schedule is None else schedules.title(schedule),
                heading_of(railway, train, titles),
                train.status in NO_WAY_ON or bool(railway.long_legs(train)),
                carried(train),
                tuple(train.cars),
            )
        )
    return rows


def sort_rows(rows: list[Row], column: int, descending: bool) -> list[Row]:
    """Rows sorted by a column, names without regard to case, cars by the whole load; ties keep
    the order trains were bought in."""
    keys: dict[int, Callable[[Row], object]] = {
        TRAIN: lambda r: r.title.casefold(),
        SCHEDULE: lambda r: r.schedule.casefold(),
        HEADING: lambda r: r.heading.casefold(),
        CARRYING: lambda r: sum(load for _, load in r.carrying),
    }
    return sorted(rows, key=keys.get(column, keys[TRAIN]), reverse=descending)


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
    # The schedule whose trains it shows, by name; None for every train.
    schedule: str | None = None
    # The column sorted by and whether from the top down.
    sorted_by: int = TRAIN
    descending: bool = False
    # The trains picked with their boxes, by name, and those the table showed last.
    picked: set[str] = field(default_factory=set)
    shown: list[str] = field(default_factory=list)

    def keeps(self, row: Row) -> bool:
        return self.schedule is None or row.train.schedule == self.schedule

    def draw_filter(self, railway: Railway) -> None:
        """A list showing only the trains of one schedule."""
        schedules = railway.schedules
        chosen = None if self.schedule is None else schedules.named(self.schedule)
        if chosen is None:
            self.schedule = None
        imgui.set_next_item_width(220)
        if imgui.begin_combo(
            "##schedule filter", ANY if chosen is None else schedules.title(chosen)
        ):
            if imgui.selectable(ANY, chosen is None)[0]:
                self.schedule = None
            for schedule in schedules.all:
                label = f"{schedules.title(schedule)}##{schedule.name}"
                if imgui.selectable(label, schedule is chosen)[0]:
                    self.schedule = schedule.name
            imgui.end_combo()
        tip("Show only the trains of one schedule")

    def draw(
        self, rows: list[Row], chosen: str | None, picture: EnginePicture, picking: bool = True
    ) -> tuple[str | None, bool]:
        """The table of the rows of the schedule shown, sorted, with boxes picking trains if
        `picking`; returns the train clicked, if any, and whether it was clicked twice."""
        clicked, twice = None, False
        flags = (
            LIST_TABLE_FLAGS
            | imgui.TableFlags_.sortable.value
            | imgui.TableFlags_.resizable.value
            | imgui.TableFlags_.borders_outer_h.value
            | imgui.TableFlags_.hideable.value
        )
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
            if number == PICK:
                own |= imgui.TableColumnFlags_.no_sort.value
            imgui.table_setup_column(column.title, own, column.width)
        imgui.table_set_column_enabled(PICK, picking)
        imgui.table_headers_row()
        specs = imgui.table_get_sort_specs()
        if specs is not None and specs.specs_count:
            spec = specs.get_specs(0)
            self.sorted_by = spec.column_index
            self.descending = spec.get_sort_direction() == imgui.SortDirection.descending
        kept = [row for row in rows if self.keeps(row)]
        shown = sort_rows(kept, self.sorted_by, self.descending)
        self.shown = [row.train.name for row in shown]
        clipper = imgui.ListClipper()
        clipper.begin(len(shown), ROW_HEIGHT)
        while clipper.step():
            for row in shown[clipper.display_start : clipper.display_end]:
                if self._draw_row(row, row.train.name == chosen, picture, picking):
                    clicked = row.train.name
                    twice = imgui.is_mouse_double_clicked(imgui.MouseButton_.left)
        clipper.end()
        imgui.end_table()
        if not shown:
            imgui.text_disabled(
                "This schedule has no trains." if rows else "You have no trains yet."
            )
        return clicked, twice

    def _draw_row(self, row: Row, chosen: bool, picture: EnginePicture, picking: bool) -> bool:
        """One train's row, a click anywhere on it but its box choosing the train; returns
        whether it was clicked."""
        imgui.table_next_row(0, ROW_HEIGHT)
        imgui.table_set_column_index(TRAIN)
        flags = (
            imgui.SelectableFlags_.span_all_columns.value
            | imgui.SelectableFlags_.allow_double_click.value
            | imgui.SelectableFlags_.allow_overlap.value
        )
        left = imgui.get_cursor_screen_pos().x
        clicked = imgui.selectable(f"##{row.train.name}", chosen, flags, (0, ROW_HEIGHT))[0]
        low = imgui.ImVec2(left, imgui.get_item_rect_min().y)
        draw_list = imgui.get_window_draw_list()
        icon_top = low.y + (ROW_HEIGHT - ICON_BOX[1]) / 2
        draw_engine_icon(draw_list, picture, row.train, imgui.ImVec2(low.x + 2, icon_top))
        text_top = low.y + (ROW_HEIGHT - imgui.get_text_line_height()) / 2
        colour = BRASS if chosen else TEXT
        draw_list.add_text((low.x + ICON_BOX[0] + 10, text_top), color_u32(colour), row.title)
        if picking:
            imgui.table_set_column_index(PICK)
            imgui.set_cursor_pos_y(
                imgui.get_cursor_pos_y() + (ROW_HEIGHT - imgui.get_frame_height()) / 2
            )
            name = row.train.name
            ticked, on = imgui.checkbox(f"##pick{name}", name in self.picked)
            if ticked:
                if on:
                    self.picked.add(name)
                else:
                    self.picked.discard(name)
            tip("Pick it to upgrade or sell with the other trains picked")
        _cell(SCHEDULE, row.schedule or "-", TEXT_DIM)
        _cell(HEADING, row.heading, GONE if row.stuck else TEXT_DIM)
        imgui.table_set_column_index(CARRYING)
        _draw_wagons(row.wagons)
        return clicked


def draw_cargo_counts(cars: list[tuple[Cargo, int]], hint: str) -> None:
    """Cargos each as an icon and how many cars of it, in a table cell; `hint` says where the
    cars are counted."""
    if not cars:
        imgui.text_disabled("no cars")
        return
    top = imgui.get_cursor_pos_y() + (ROW_HEIGHT - CAR_ICON) / 2
    for number, (cargo, count) in enumerate(cars):
        if number:
            imgui.same_line(0, 6)
        imgui.set_cursor_pos_y(top)
        imgui.dummy((CAR_ICON, CAR_ICON))
        draw_good_icon(imgui.get_window_draw_list(), cargo, imgui.get_item_rect_min(), CAR_ICON)
        tip(f"{cargo.value}: {count} {plural('car', count)} {hint}")
        imgui.same_line(0, 1)
        imgui.set_cursor_pos_y(top + (CAR_ICON - imgui.get_text_line_height()) / 2)
        imgui.text_disabled(f"{count}")


def _cell(column: int, text: str, colour: tuple[float, float, float, float] | None = None) -> None:
    imgui.table_set_column_index(column)
    imgui.set_cursor_pos_y(
        imgui.get_cursor_pos_y() + (ROW_HEIGHT - imgui.get_text_line_height()) / 2
    )
    if colour is None:
        imgui.text(text)
    else:
        colored(colour, text)


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
