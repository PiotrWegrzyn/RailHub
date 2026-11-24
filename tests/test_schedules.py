import pytest

from builders import flat_map, joined_stations
from railroads.sim.constructions import Constructions
from railroads.sim.goods import Good
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.schedules import Schedule
from railroads.sim.trains import ENGINES, Stop, Train

FIRST, MIDDLE, LAST = (6, 3), (6, 14), (6, 26)
ALONE = (20, 14)
# Carries fuel for 13 cells: enough from one station of the line to the next, not for the whole way.
SHORT_RANGE = "tank"


def line_of_three() -> Railway:
    constructions = Constructions(flat_map(rows=24, cols=32))
    growth = Growth(constructions, GrowthRule(trades=False))
    for station in (FIRST, MIDDLE, LAST, ALONE):
        build_station(growth, station)
    for start, end in ((FIRST, MIDDLE), (MIDDLE, LAST)):
        route = plan_route(constructions, start, end)
        assert route is not None
        lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    railway.year = 1850
    railway.finances.earn(10_000_000.0, "test")
    return railway


def schedule_of(railway: Railway, *stops: Stop) -> Schedule:
    schedule = railway.schedules.new()
    schedule.stops = list(stops)
    railway.schedules.settle(schedule)
    return schedule


def bought(railway: Railway, schedule: Schedule, engine: str = SHORT_RANGE) -> Train:
    train = railway.buy(engine, schedule=schedule)
    assert train is not None
    return train


def run_until(railway: Railway, done: object, most_months: float = 24.0) -> float:
    months = 0.0
    while not done() and months < most_months:  # type: ignore[operator]
        railway.run(0.01)
        months += 0.01
    return months


def test_a_train_bought_on_its_own_runs_a_new_schedule_of_its_own() -> None:
    _, railway = joined_stations()

    train = railway.buy()

    assert train is not None
    assert railway.schedules.of(train) is not None
    assert railway.schedules.trains(railway.schedules.of(train)) == [train]


def test_a_train_bought_for_a_schedule_calls_at_its_stops_with_their_cars_and_flags() -> None:
    _, railway = joined_stations()
    railway.finances.earn(1_000_000.0, "test")
    schedule = schedule_of(
        railway,
        Stop((6, 3), [Good.WHEAT], wait_for_load=True),
        Stop((6, 12), refuel_only=True),
        Stop((6, 20), [Good.COAL]),
    )

    train = bought(railway, schedule, "planet")

    assert [stop.cell for stop in train.stops] == [(6, 3), (6, 12), (6, 20)]
    assert [stop.cars for stop in train.stops] == [[Good.WHEAT], [], [Good.COAL]]
    assert [stop.wait_for_load for stop in train.stops] == [True, False, False]
    assert [stop.refuel_only for stop in train.stops] == [False, True, False]
    assert train.stops[0] is not schedule.stops[0]


def test_a_train_takes_a_stops_cars_from_the_left_as_far_as_its_engine_takes_them() -> None:
    _, railway = joined_stations()
    railway.finances.earn(1_000_000.0, "test")
    schedule = schedule_of(
        railway, Stop((6, 3), [Good.WHEAT] * 3 + [Good.COAL] * 27), Stop((6, 20))
    )

    train = bought(railway, schedule, "planet")

    taken = train.stops[0].cars
    assert 0 < len(taken) <= railway.most_cars(train)
    assert taken == schedule.stops[0].cars[: len(taken)]
    assert len(schedule.stops[0].cars) == 30


def test_the_game_puts_in_the_refuelling_stops_the_weakest_trains_fuel_needs() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(LAST))

    bought(railway, schedule)

    assert [(stop.cell, stop.auto) for stop in schedule.stops] == [
        (FIRST, False),
        (MIDDLE, True),
        (LAST, False),
        (MIDDLE, True),
    ]
    assert schedule.tank == ENGINES[SHORT_RANGE].tank
    assert schedule.problem == ""


def test_no_refuelling_stop_goes_in_where_the_fuel_lasts() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(MIDDLE))

    bought(railway, schedule)

    assert [stop.cell for stop in schedule.stops] == [FIRST, MIDDLE]


def test_a_schedule_with_no_track_between_two_stops_says_so() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(ALONE))

    bought(railway, schedule)

    assert schedule.problem.startswith("No track from")


def test_a_schedule_says_when_no_station_is_within_its_weakest_trains_reach() -> None:
    _, railway = joined_stations()
    railway.year = 1850
    railway.finances.earn(1_000_000.0, "test")
    schedule = schedule_of(railway, Stop((6, 3)), Stop((6, 20)))

    bought(railway, schedule)

    assert "no station is within the 13 cells" in schedule.problem


@pytest.mark.parametrize(("engine", "stands"), [(SHORT_RANGE, True), ("planet", False)])
def test_an_engine_that_cannot_reach_a_schedules_next_stop_is_said_to_stand_on_it(
    engine: str, stands: bool
) -> None:
    _, railway = joined_stations()
    railway.year = 1850
    railway.finances.earn(1_000_000.0, "test")
    schedule = schedule_of(railway, Stop((6, 3)), Stop((6, 20)))
    bought(railway, schedule, "planet")

    why = railway.schedules.cannot_run(schedule, engine)

    assert bool(why) == stands


def test_an_engine_runs_a_line_where_refuelling_can_be_planned_for_it() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(LAST))
    bought(railway, schedule, "planet")

    why = railway.schedules.cannot_run(schedule, SHORT_RANGE)

    assert why == ""


def test_the_weakest_train_that_can_make_the_round_sets_the_refuelling_stops() -> None:
    _, railway = joined_stations()
    railway.year = 1850
    railway.finances.earn(1_000_000.0, "test")
    schedule = schedule_of(railway, Stop((6, 3)), Stop((6, 20)))
    bought(railway, schedule)

    bought(railway, schedule, "planet")

    assert schedule.tank == ENGINES["planet"].tank
    assert schedule.problem.startswith("Trains carrying fuel for less than 21 cells wait")
    assert [stop.cell for stop in schedule.stops] == [(6, 3), (6, 20)]


def test_refuelling_stops_are_planned_again_when_the_weakest_train_is_sold() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(MIDDLE), Stop(LAST))
    weak = bought(railway, schedule)
    bought(railway, schedule, "planet")
    railway.schedules.set_manual(schedule, False)

    railway.sell(weak)

    assert schedule.tank == ENGINES["planet"].tank


def test_changing_a_refuelling_stop_the_game_put_in_sets_them_all_by_hand() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(LAST))
    bought(railway, schedule)

    railway.schedules.set_manual(schedule, True)
    schedule.stops.pop(1)
    railway.schedules.settle(schedule)

    assert schedule.manual
    assert [(stop.cell, stop.auto) for stop in schedule.stops] == [
        (FIRST, False),
        (LAST, False),
        (MIDDLE, False),
    ]


def test_giving_the_refuelling_stops_back_to_the_game_plans_them_anew() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(LAST))
    bought(railway, schedule)
    railway.schedules.set_manual(schedule, True)
    schedule.stops.pop(1)
    railway.schedules.settle(schedule)

    railway.schedules.set_manual(schedule, False)

    assert [stop.cell for stop in schedule.stops] == [FIRST, MIDDLE, LAST, MIDDLE]
    assert all(stop.auto for stop in schedule.stops if stop.refuel_only)


def test_new_cars_reach_a_schedules_trains_at_once() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(MIDDLE))
    train = bought(railway, schedule)
    version = schedule.version

    schedule.stops[1].add_car(Good.COAL)
    railway.schedules.settle(schedule)

    assert train.stops[1].cars == [Good.COAL]
    assert schedule.version == version


def test_a_train_on_its_round_keeps_its_stops_until_it_is_back_at_its_first() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(MIDDLE))
    train = bought(railway, schedule)
    run_until(railway, lambda: train.arrivals >= 1)

    schedule.stops.append(Stop(LAST))
    railway.schedules.settle(schedule)
    kept = [stop.cell for stop in train.stops]
    run_until(railway, lambda: train.taken == (schedule.name, schedule.version))

    assert kept == [FIRST, MIDDLE]
    assert [stop.cell for stop in train.stops] == [FIRST, MIDDLE, LAST, MIDDLE]
    assert train.stops[(train.heading - 1) % len(train.stops)].cell == FIRST


def test_a_train_moved_to_a_schedule_starting_elsewhere_is_off_the_map_as_long_as_the_way_takes() -> (
    None
):
    railway = line_of_three()
    train = bought(railway, schedule_of(railway, Stop(FIRST), Stop(MIDDLE)))
    other = schedule_of(railway, Stop(LAST), Stop(MIDDLE))
    railway.run(0.0)

    railway.schedules.join(train, other)
    railway.run(0.0)
    away = train.away
    months = run_until(railway, lambda: train.away <= 0)

    assert away == pytest.approx(railway.way_months(train, FIRST, LAST))
    assert months == pytest.approx(away, abs=0.02)
    assert train.stops[0].cell == LAST


def test_a_train_off_the_map_says_where_it_is_going_and_holds_no_way() -> None:
    railway = line_of_three()
    train = bought(railway, schedule_of(railway, Stop(FIRST), Stop(MIDDLE)))
    other = schedule_of(railway, Stop(LAST), Stop(MIDDLE))

    railway.schedules.join(train, other)
    railway.run(0.01)

    assert train.status == f"On its way to {railway.name_of(LAST)}"
    assert train.path == []


def test_an_engine_swap_plans_the_refuelling_stops_again() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(LAST))
    train = bought(railway, schedule, "planet")
    stops_before = len(schedule.stops)
    railway.year = 1860

    railway.swap_engine(train, "american")

    assert stops_before == 4
    assert [stop.cell for stop in schedule.stops] == [FIRST, LAST]


def test_a_schedule_is_named_after_the_first_and_last_of_the_players_stops() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(LAST))
    bought(railway, schedule)

    title = railway.schedules.title(schedule)

    assert title == f"{railway.name_of(FIRST)} – {railway.name_of(LAST)}"


def test_a_schedule_some_train_runs_cannot_be_removed() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST), Stop(MIDDLE))
    bought(railway, schedule)
    empty = railway.schedules.new()

    kept = railway.schedules.remove(schedule)
    removed = railway.schedules.remove(empty)

    assert not kept and removed
    assert railway.schedules.all == [schedule]


def test_a_train_is_named_after_its_schedules_ends_not_the_refuelling_stop_on_the_way_back() -> (
    None
):
    railway = line_of_three()
    train = bought(railway, schedule_of(railway, Stop(FIRST), Stop(LAST)))

    title = railway.titles()[train.name]

    assert title.startswith(f"{railway.name_of(FIRST)} – {railway.name_of(LAST)}")


def test_a_train_takes_a_stops_cars_from_the_left_only_while_its_engine_pulls_them_full() -> None:
    railway = line_of_three()
    schedule = schedule_of(railway, Stop(FIRST, [Good.WHEAT] * 3), Stop(MIDDLE))
    train = bought(railway, schedule)

    taken = railway.schedules.taken(train, schedule.stops[0])

    assert taken == len(train.stops[0].cars) < 3
