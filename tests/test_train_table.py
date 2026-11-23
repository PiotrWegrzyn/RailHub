from railroads.sim.goods import Good
from railroads.sim.trains import Stop, Train
from railroads.ui.train_table import (
    FIRST,
    RIDE,
    TRAIN,
    Row,
    TrainFilter,
    shown_status,
    sort_rows,
)
from railroads.ui.trains_window import LEAST_LIST, LEAST_TRAIN, SPLIT_WIDTH, list_width


def route(*stops: Stop) -> Train:
    return Train("a", list(stops))


def row(title: str, first: str = "", net: float | None = None) -> Row:
    return Row(Train(title), title, first, "", [], "Running", False, net)


def test_the_first_and_last_station_filters_keep_only_trains_starting_and_ending_there() -> None:
    trains = [
        route(Stop((0, 0)), Stop((5, 5)), Stop((9, 9))),
        route(Stop((9, 9)), Stop((5, 5)), Stop((0, 0))),
    ]
    picked = TrainFilter(first=(0, 0), last=(9, 9))

    kept = [picked.keeps(each) for each in trains]

    assert kept == [True, False]


def test_the_stop_filter_keeps_trains_calling_there_anywhere_on_the_way() -> None:
    trains = [route(Stop((0, 0)), Stop((5, 5)), Stop((9, 9))), route(Stop((0, 0)), Stop((9, 9)))]
    picked = TrainFilter(calls_at=(5, 5))

    kept = [picked.keeps(each) for each in trains]

    assert kept == [True, False]


def test_the_car_filter_keeps_trains_taking_on_cars_of_that_good_at_any_stop() -> None:
    trains = [
        route(Stop((0, 0)), Stop((9, 9), [Good.COAL])),
        route(Stop((0, 0), [Good.MAIL]), Stop((9, 9))),
    ]
    picked = TrainFilter(cargo=Good.COAL)

    kept = [picked.keeps(each) for each in trains]

    assert kept == [True, False]


def test_sorting_by_name_ignores_case_and_can_run_from_the_bottom_up() -> None:
    rows = [row("b"), row("A"), row("c")]

    up, down = sort_rows(rows, TRAIN, False), sort_rows(rows, TRAIN, True)

    assert [r.title for r in up] == ["A", "b", "c"]
    assert [r.title for r in down] == ["c", "b", "A"]


def test_sorting_by_first_station_puts_the_same_station_together() -> None:
    rows = [row("a", "Warka"), row("b", "Grójec"), row("c", "Warka")]

    shown = sort_rows(rows, FIRST, False)

    assert [r.title for r in shown] == ["b", "a", "c"]


def test_trains_not_running_come_last_whichever_way_rides_are_sorted() -> None:
    rows = [row("idle"), row("poor", net=-50.0), row("rich", net=900.0)]

    up, down = sort_rows(rows, RIDE, False), sort_rows(rows, RIDE, True)

    assert [r.title for r in up] == ["poor", "rich", "idle"]
    assert [r.title for r in down] == ["rich", "poor", "idle"]


def test_a_status_names_the_train_ahead_by_its_title() -> None:
    titles = {"Steam engine 3": "Warka – Magnuszew Cargo"}

    shown = (
        shown_status("Waiting at a signal for Steam engine 3", titles),
        shown_status("Running", titles),
    )

    assert shown == ("Waiting at a signal for Warka – Magnuszew Cargo", "Running")


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
