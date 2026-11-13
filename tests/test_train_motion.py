import pytest

from builders import flat_map
from railroads.sim.clock import SECONDS_PER_MONTH
from railroads.sim.constructions import Constructions, TrackWidth
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import CELLS_PER_MONTH, Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.trains import ENGINES, Stop, Train

WEST, EAST = (6, 3), (6, 20)
FRAME = 1 / 60 / SECONDS_PER_MONTH


def line(width: TrackWidth = TrackWidth.SINGLE) -> Railway:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, WEST)
    build_station(growth, EAST)
    route = plan_route(constructions, WEST, EAST, width)
    assert route is not None
    lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    return railway


def running(railway: Railway, name: str, start, end, engine: str = "planet") -> Train:
    train = Train(name, stops=[Stop(start), Stop(end)], engine=engine)
    railway.trains.append(train)
    return train


def run_until(railway: Railway, done, seconds: float = 600.0) -> None:
    for _ in range(int(seconds * 60)):
        railway.run(FRAME)
        if done():
            return


def test_the_planet_takes_ten_seconds_to_full_speed_and_later_engines_less() -> None:
    planet = ENGINES["planet"]

    later = [engine.pickup for engine in ENGINES.values() if engine.year > planet.year]
    earlier = [engine.pickup for engine in ENGINES.values() if engine.year < planet.year]

    assert planet.pickup == 10.0
    assert all(pickup <= planet.pickup for pickup in later)
    assert all(pickup > planet.pickup for pickup in earlier)


def test_a_train_sets_off_slowly_and_reaches_full_speed_after_its_pickup_time() -> None:
    railway = line()
    train = running(railway, "Slow starter", WEST, EAST)
    pickup = ENGINES["planet"].pickup / SECONDS_PER_MONTH

    for _ in range(50):
        railway.run(pickup / 100)
    halfway = train.pace
    for _ in range(50):
        railway.run(pickup / 100)

    assert halfway == pytest.approx(0.5)
    assert train.pace == pytest.approx(1.0)
    assert train.along == pytest.approx(0.5 * pickup * CELLS_PER_MONTH, rel=1e-3)


def test_how_far_a_train_gets_does_not_depend_on_the_frame_rate() -> None:
    smooth, choppy = line(), line()
    fast = running(smooth, "Smooth", WEST, EAST)
    slow = running(choppy, "Choppy", WEST, EAST)
    pickup = ENGINES["planet"].pickup / SECONDS_PER_MONTH

    for _ in range(120):
        smooth.run(pickup / 100)
    for _ in range(12):
        choppy.run(pickup / 10)

    assert fast.along == pytest.approx(slow.along)


def test_a_better_engine_picks_up_speed_sooner() -> None:
    steam = Train("Steam", engine="planet")
    electric = Train("Electric", engine="express")
    months = 2 / SECONDS_PER_MONTH

    steam.speed_up(months, ENGINES["planet"].pickup / SECONDS_PER_MONTH)
    electric.speed_up(months, ENGINES["express"].pickup / SECONDS_PER_MONTH)

    assert electric.pace > steam.pace


def test_a_train_reaching_a_stop_sets_off_again_from_a_stand() -> None:
    railway = line()
    train = running(railway, "Shuttle", WEST, EAST)

    run_until(railway, lambda: train.arrivals == 1)

    assert train.pace == 0.0


def test_trains_meeting_head_on_one_stops_and_lets_the_other_by() -> None:
    railway = line()
    first = running(railway, "First", WEST, EAST)
    second = running(railway, "Second", EAST, WEST)

    run_until(railway, lambda: bool(second.giving_way))
    stood = second.along
    railway.run(10 * FRAME)

    assert second.giving_way == "First"
    assert first.giving_way == ""
    assert second.along == stood
    assert second.status == "Giving way to First"


def test_a_train_that_let_another_by_sets_off_again_from_a_stand_and_both_arrive() -> None:
    railway = line()
    first = running(railway, "First", WEST, EAST)
    second = running(railway, "Second", EAST, WEST)
    run_until(railway, lambda: bool(second.giving_way))

    run_until(railway, lambda: not second.giving_way)
    pace = second.pace
    run_until(railway, lambda: first.arrivals >= 1 and second.arrivals >= 1)

    assert pace == 0.0
    assert first.arrivals >= 1 and second.arrivals >= 1


def test_the_better_engine_goes_first() -> None:
    railway = line()
    steam = running(railway, "Steam", WEST, EAST)
    express = running(railway, "Express", EAST, WEST, engine="pacific")

    run_until(railway, lambda: bool(steam.giving_way or express.giving_way))

    assert steam.giving_way == "Express"
    assert express.giving_way == ""


def test_a_train_that_has_stood_longer_letting_others_by_goes_first() -> None:
    railway = line()
    first = running(railway, "First", WEST, EAST)
    second = running(railway, "Second", EAST, WEST)
    second.waited = 0.01

    run_until(railway, lambda: bool(first.giving_way or second.giving_way))

    assert first.giving_way == "Second"


def test_trains_going_opposite_ways_on_double_track_pass_without_stopping() -> None:
    railway = line(TrackWidth.DOUBLE)
    first = running(railway, "First", WEST, EAST)
    second = running(railway, "Second", EAST, WEST)
    stopped = []

    for _ in range(60 * 60):
        railway.run(FRAME)
        stopped += [train.name for train in railway.trains if train.giving_way]
        if first.arrivals and second.arrivals:
            break

    assert stopped == []
    assert first.arrivals == 1 and second.arrivals == 1


def test_many_trains_on_one_line_all_keep_getting_through() -> None:
    railway = line()
    for number in range(8):
        start, end = (WEST, EAST) if number % 2 else (EAST, WEST)
        running(railway, f"Train {number}", start, end)

    for _ in range(int(3 / FRAME / 10)):
        railway.run(10 * FRAME)

    assert all(train.arrivals >= 2 for train in railway.trains)


def test_a_train_saved_before_trains_picked_up_speed_still_runs() -> None:
    railway = line()
    train = running(railway, "Old", WEST, EAST)
    for name in ("pace", "giving_way", "waited"):
        train.__dict__.pop(name)

    run_until(railway, lambda: train.arrivals == 1)

    assert train.arrivals == 1
