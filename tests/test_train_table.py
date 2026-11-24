from builders import joined_stations
from railroads.sim.goods import Good
from railroads.sim.trains import Car, Stop, Train
from railroads.ui.train_table import (
    CARRYING,
    SCHEDULE,
    TRAIN,
    Row,
    TrainTable,
    carried,
    heading_of,
    shown_status,
    sort_rows,
    train_rows,
)
from railroads.ui.trains_window import LEAST_LIST, LEAST_TRAIN, SPLIT_WIDTH, list_width


def row(title: str, schedule: str = "", carrying: tuple[tuple[Good, float], ...] = ()) -> Row:
    return Row(Train(title), title, schedule, carrying=carrying)


def test_the_schedule_filter_keeps_only_that_schedules_trains() -> None:
    ours, theirs = Train("a", schedule="Schedule 1"), Train("b", schedule="Schedule 2")
    table = TrainTable(schedule="Schedule 1")

    kept = [table.keeps(Row(train, train.name)) for train in (ours, theirs)]

    assert kept == [True, False]


def test_sorting_by_name_ignores_case_and_can_run_from_the_bottom_up() -> None:
    rows = [row("b"), row("A"), row("c")]

    up, down = sort_rows(rows, TRAIN, False), sort_rows(rows, TRAIN, True)

    assert [r.title for r in up] == ["A", "b", "c"]
    assert [r.title for r in down] == ["c", "b", "A"]


def test_sorting_by_schedule_puts_the_same_schedule_together() -> None:
    rows = [row("a", "Warka"), row("b", "Grójec"), row("c", "Warka")]

    shown = sort_rows(rows, SCHEDULE, False)

    assert [r.title for r in shown] == ["b", "a", "c"]


def test_what_a_train_carries_sums_each_good_over_its_cars_the_most_first() -> None:
    train = Train("a")
    train.cars = [
        Car(Good.COAL, load=10.0),
        Car(Good.CATTLE, load=20.0),
        Car(Good.COAL, load=15.0),
        Car(Good.WHEAT),
        Car(Good.PASSENGERS, dining=True),
    ]

    carrying = carried(train)

    assert carrying == ((Good.COAL, 25.0), (Good.CATTLE, 20.0))


def test_sorting_by_what_trains_carry_orders_them_by_their_whole_load() -> None:
    empty = row("empty")
    light = row("light", carrying=((Good.COAL, 5.0),))
    heavy = row("heavy", carrying=((Good.COAL, 5.0), (Good.WHEAT, 30.0)))

    shown = sort_rows([heavy, empty, light], CARRYING, True)

    assert [r.title for r in shown] == ["heavy", "light", "empty"]


def test_a_status_names_the_train_ahead_by_its_title() -> None:
    titles = {"Steam engine 3": "Warka – Magnuszew Cargo"}

    shown = (
        shown_status("Waiting at a signal for Steam engine 3", titles),
        shown_status("Running", titles),
    )

    assert shown == ("Waiting at a signal for Warka – Magnuszew Cargo", "Running")


def test_a_running_train_is_heading_to_its_next_stop_and_says_when_only_to_refuel() -> None:
    _, railway = joined_stations()
    train = Train("a", [Stop((6, 3)), Stop((6, 20), refuel_only=True)], path=[(0.0, 0.0)])
    standing = Train("b", [Stop((6, 3)), Stop((6, 20))], status="Loading at Warka")

    headings = (heading_of(railway, train, {}), heading_of(railway, standing, {}))

    assert headings == (f"{railway.name_of((6, 20))} to refuel", "Loading at Warka")


def test_the_table_is_dragged_no_narrower_than_its_least_nor_crowding_the_train_beside() -> None:
    room = 1600.0

    narrow, wide, untouched = (
        list_width(room, 10.0),
        list_width(room, 5000.0),
        list_width(room, None),
    )

    assert narrow == LEAST_LIST
    assert wide == room - LEAST_TRAIN - SPLIT_WIDTH
    assert untouched == room / 2


def test_a_row_shows_every_car_the_train_pulls_now() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.cars = [Car(Good.COAL, 10.0, capacity=144.0), Car(Good.WHEAT)]

    rows = train_rows(railway, {})

    assert rows[0].wagons == tuple(train.cars)
