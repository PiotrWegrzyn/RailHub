import itertools
import math

import pytest

from builders import flat_map
from railroads.sim.clock import SECONDS_PER_MONTH
from railroads.sim.constructions import Constructions, TrackWidth
from railroads.sim.goods import Good
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import CELLS_PER_MONTH, Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.signals import passes
from railroads.sim.trains import ENGINES, Kind, Stop, Train

WEST, EAST = (6, 3), (6, 20)
FRAME = 1 / 60 / SECONDS_PER_MONTH


def line(width: TrackWidth = TrackWidth.SINGLE, *between: tuple[int, int]) -> Railway:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, WEST)
    build_station(growth, EAST)
    route = plan_route(constructions, WEST, EAST, width)
    assert route is not None
    lay_route(growth, route)
    for cell in between:
        assert build_station(growth, cell) is not None
    railway = Railway(constructions)
    growth.railway = railway
    return railway


def running(railway: Railway, name: str, start, end, engine: str = "planet") -> Train:
    train = Train(name, stops=[Stop(start), Stop(end)], engine=engine)
    railway.trains.append(train)
    return train


def along_to(railway: Railway, start, end, cell) -> float:
    path, _ = railway.route(start, end)
    ends = [0.0, *itertools.accumulate(math.dist(a, b) for a, b in itertools.pairwise(path))]
    return next(
        at for point, at in zip(path, ends, strict=True) if railway.network.cell_at(point) == cell
    )


def run_until(railway: Railway, done, seconds: float = 600.0) -> None:
    for _ in range(int(seconds * 60)):
        railway.run(FRAME)
        if done():
            return


def test_light_engines_get_going_quickest_and_pioneers_and_heavy_freight_ones_slowest() -> None:
    planet = ENGINES["planet"]

    pickups = {
        kind: [engine.pickup for engine in ENGINES.values() if engine.kind == kind] for kind in Kind
    }

    assert planet.pickup == 12.0
    assert max(pickups[Kind.LIGHT]) <= 3.0
    assert max(pickups[Kind.EXPRESS]) <= planet.pickup
    assert min(pickups[Kind.PIONEER]) > 2 * planet.pickup
    assert min(pickups[Kind.HEAVY]) > 2 * max(pickups[Kind.GENERAL])


def test_a_train_sets_off_slowly_and_reaches_full_speed_after_its_pickup_time() -> None:
    railway = line()
    train = running(railway, "Slow starter", WEST, EAST)
    pickup = ENGINES["planet"].pickup / SECONDS_PER_MONTH
    speed = ENGINES["planet"].speed

    for _ in range(50):
        railway.run(pickup / 100)
    halfway = train.pace
    for _ in range(50):
        railway.run(pickup / 100)

    assert halfway == pytest.approx(0.5)
    assert train.pace == pytest.approx(1.0)
    assert train.along == pytest.approx(0.5 * pickup * CELLS_PER_MONTH * speed, rel=1e-3)


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


def test_a_train_waits_at_its_station_while_another_comes_the_other_way_on_single_line() -> None:
    railway = line()
    first = running(railway, "First", WEST, EAST)
    second = running(railway, "Second", EAST, WEST)

    run_until(railway, lambda: first.along > 3.0)
    railway.run(10 * FRAME)

    assert second.giving_way == "First"
    assert first.giving_way == ""
    assert railway.network.passing(railway.network.spot(second, second.along)[2]) == (True, False)
    assert second.status == "Waiting at a signal for First"


def test_a_train_that_let_another_by_sets_off_again_from_a_stand_and_both_arrive() -> None:
    railway = line()
    first = running(railway, "First", WEST, EAST)
    second = running(railway, "Second", EAST, WEST)
    run_until(railway, lambda: bool(second.giving_way))

    run_until(railway, lambda: not second.giving_way)
    pace = second.pace
    run_until(railway, lambda: first.arrivals >= 1 and second.arrivals >= 1)

    assert pace == pytest.approx(FRAME / (ENGINES["planet"].pickup / SECONDS_PER_MONTH))
    assert first.arrivals >= 1 and second.arrivals >= 1


def test_the_train_that_waited_first_at_a_signal_gets_the_line_first() -> None:
    railway = line()
    first = running(railway, "First", WEST, EAST)
    waiting = running(railway, "Waiting", EAST, WEST)
    run_until(railway, lambda: first.along > 3.0)
    later = running(railway, "Later", WEST, EAST, engine="pacific")
    run_until(railway, lambda: bool(later.giving_way))

    run_until(railway, lambda: first.arrivals >= 1)
    run_until(railway, lambda: not waiting.giving_way or not later.giving_way)

    assert waiting.giving_way == ""
    assert later.giving_way == "Waiting"


def test_trains_going_opposite_ways_on_double_track_pass_without_stopping() -> None:
    railway = line(TrackWidth.DOUBLE)
    first = running(railway, "First", WEST, EAST)
    second = running(railway, "Second", EAST, WEST)
    stopped = []

    for _ in range(240 * 60):
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

    for _ in range(int(12 / FRAME / 10)):
        railway.run(10 * FRAME)

    arrivals = [train.arrivals for train in railway.trains]
    assert min(arrivals) >= 2
    assert max(arrivals) - min(arrivals) <= 1


def test_a_fast_train_behind_a_slow_one_on_double_track_follows_it_and_never_runs_into_it() -> None:
    railway = line(TrackWidth.DOUBLE)
    slow = running(railway, "Slow", WEST, EAST)
    run_until(railway, lambda: slow.along > 3.0)
    fast = running(railway, "Fast", WEST, EAST, engine="pacific")
    statuses = set()
    gaps = []

    while not slow.arrivals:
        railway.run(FRAME)
        statuses.add(fast.status)
        if fast.along > 0:
            gaps.append(slow.along - fast.along)

    assert "Following Slow" in statuses
    assert min(gaps) > slow.length


def test_trains_going_the_same_way_follow_each_other_onto_single_line() -> None:
    railway = line()
    first = running(railway, "First", WEST, EAST)
    run_until(railway, lambda: first.along > 3.0)
    second = running(railway, "Second", WEST, EAST)

    run_until(railway, lambda: second.along > 3.0 or first.arrivals > 0)

    assert first.arrivals == 0
    assert second.along > 3.0


def test_trains_following_each_other_on_single_line_never_stand_still_on_it() -> None:
    railway = line(TrackWidth.SINGLE, (6, 11))
    coal = [Good.COAL] * 3
    for number in range(6):
        start, end = (WEST, EAST) if number % 3 else (EAST, WEST)
        cars = coal if number % 2 else []
        railway.trains.append(Train(f"Train {number}", stops=[Stop(start, cars), Stop(end, cars)]))
    stood_on_line = []

    for _ in range(int(12 / FRAME / 10)):
        railway.run(10 * FRAME)
        for train in railway.trains:
            if train.giving_way and train.along > 0:
                tail = train.along - train.length
                stood_on_line += [
                    crossing.cell
                    for crossing in passes(railway.network, train)
                    if crossing.end > tail and crossing.start < train.along
                ]

    assert stood_on_line == []
    assert min(train.arrivals for train in railway.trains) >= 3


def test_a_train_waits_for_another_crossing_its_line_and_both_get_through() -> None:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    lines = (((6, 6), (6, 16)), ((1, 11), (11, 11)))
    for start, end in lines:
        build_station(growth, start)
        build_station(growth, end)
        route = plan_route(constructions, start, end, TrackWidth.SINGLE)
        assert route is not None
        lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    ahead = along_to(railway, *lines[0], (6, 12)) - along_to(railway, *lines[1], (6, 12))
    across = running(railway, "Across", *lines[0])
    run_until(railway, lambda: across.along > ahead)
    down = running(railway, "Down", *lines[1])
    statuses = set()

    for _ in range(int(240 / FRAME / 60)):
        railway.run(FRAME)
        statuses.add(down.status)
        if across.arrivals and down.arrivals:
            break

    assert "Waiting at a signal for Across" in statuses
    assert across.arrivals == 1 and down.arrivals == 1


def test_a_train_too_long_for_a_station_on_the_way_never_meets_another_there() -> None:
    railway = line(TrackWidth.SINGLE, (6, 11))
    coal = [Good.COAL] * 3
    long = Train("Long", stops=[Stop(WEST, cars=coal), Stop(EAST, cars=coal)])
    short = Train("Short", stops=[Stop(EAST), Stop(WEST)])
    railway.trains.extend([long, short])

    run_until(railway, lambda: long.arrivals >= 2 and short.arrivals >= 2)

    assert long.length > 1.5
    assert long.arrivals >= 2 and short.arrivals >= 2


def test_a_train_put_first_gets_the_line_before_a_train_that_waited_longer() -> None:
    railway = line()
    ahead = running(railway, "Ahead", WEST, EAST)
    waiting = running(railway, "Waiting", EAST, WEST)
    run_until(railway, lambda: ahead.along > 3.0)
    put_first = running(railway, "First", WEST, EAST)
    run_until(railway, lambda: bool(put_first.giving_way))
    put_first.first = True

    run_until(railway, lambda: ahead.arrivals >= 1)
    run_until(railway, lambda: not waiting.giving_way or not put_first.giving_way)

    assert put_first.giving_way == ""
    assert waiting.giving_way == "First"


def test_a_train_put_first_takes_the_way_from_a_train_yet_to_set_off_onto_it() -> None:
    railway = line()
    set_off = running(railway, "Set off", WEST, EAST)
    railway.run(FRAME)
    cleared = set_off.cleared
    put_first = running(railway, "First", EAST, WEST)
    put_first.first = True

    run_until(railway, lambda: put_first.along > 3.0)

    assert cleared > 1.0
    assert set_off.along < 0.5
    assert set_off.giving_way == "First"


def test_a_train_is_first_only_until_it_reaches_its_next_stop() -> None:
    railway = line()
    train = running(railway, "First", WEST, EAST)
    train.first = True

    run_until(railway, lambda: train.arrivals >= 1)

    assert not train.first
