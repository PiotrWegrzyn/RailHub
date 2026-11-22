import itertools
import math

import pytest
from test_railway import joined_stations

from railroads.sim.constructions import Track
from railroads.sim.railway import Railway, Stop
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.track_network import TURN_ROUND_CELLS
from railroads.sim.trains import Train


def branch_curving_one_way() -> Railway:
    """Two stations joined, and a third whose branch curves onto that line towards the first
    only: a train from the second must turn round to take it."""
    growth, railway = joined_stations()
    build_station(growth, (1, 10))
    route = plan_route(railway.constructions, (1, 10), (6, 3))
    assert route is not None
    lay_route(growth, route)
    return railway


def test_a_way_once_found_is_shared_by_every_train_taking_it() -> None:
    _, railway = joined_stations()

    first = railway.route((6, 3), (6, 20))
    second = railway.route((6, 3), (6, 20))

    assert first is not None
    assert second is first


def test_a_way_is_found_anew_once_its_track_is_pulled_up() -> None:
    _, railway = joined_stations()
    assert railway.route((6, 3), (6, 20)) is not None
    piece = next(p for p in railway.constructions.track() if isinstance(p, Track))

    railway.constructions.remove(piece)

    assert railway.route((6, 3), (6, 20)) is None


def test_a_train_is_placed_by_how_far_along_its_path_it_has_run() -> None:
    _, railway = joined_stations()
    train = Train("Test", path=[(0.0, 0.0), (2.0, 0.0), (2.0, 4.0)], speeds=[1.0, 0.5], along=3.0)

    placed = railway.position(train, back=0.5)

    assert placed == ((2.0, 0.5), (0.0, 4.0))
    assert railway.network.speed_at(train) == 0.5
    assert railway.network.length(train) == 6.0


def test_a_trains_body_runs_from_its_front_back_to_its_tail() -> None:
    _, railway = joined_stations()
    train = Train("Test", path=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)], along=5.0)
    step = 0.25

    body = railway.network.body(train, step)

    points = [at for at, _, _ in body]
    count = int(train.length / step) + 1
    assert len(points) == count
    assert points[0] == pytest.approx((4.0, 1.0))
    assert points[-1] == pytest.approx(railway.position(train, back=step * (count - 1))[0])


def test_where_track_leads_out_of_every_cell_is_found_at_once_as_one_by_one() -> None:
    _, railway = joined_stations()
    grid = railway.constructions.tile_map.grid

    railway.route((6, 3), (6, 20))

    found = railway.network._exits_to
    assert found
    assert all(grid.track_exit(cell, point) == exit for (cell, point), exit in found.items())


def test_a_routes_length_is_the_length_of_its_way_and_is_kept_until_track_changes() -> None:
    _, railway = joined_stations()
    network = railway.network
    way = network.route((6, 3), (6, 20))
    assert way is not None
    points = way[0]

    length = network.route_length((6, 3), (6, 20))
    points.append((points[-1][0] + 5.0, points[-1][1]))
    kept = network.route_length((6, 3), (6, 20))

    assert length == pytest.approx(sum(math.dist(a, b) for a, b in itertools.pairwise(points[:-1])))
    assert kept == length


def test_a_way_turns_round_on_plain_track_past_a_branch_curving_the_other_way() -> None:
    railway = branch_curving_one_way()
    network = railway.network
    way = network.route((6, 20), (1, 10))
    assert way is not None

    train = Train("Test", path=way[0], speeds=way[1])

    turns = network.turns(train)

    assert len(turns) == 1
    assert network.cell_at(way[0][network.ends(train).index(turns[0])]) == (6, 7)
    assert network.route_length((6, 20), (1, 10)) == pytest.approx(19.0)


def test_a_way_turns_round_on_plain_track_only_when_going_on_round_is_longer_still(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    railway = branch_curving_one_way()
    monkeypatch.setattr("railroads.sim.track_network.TURN_ROUND_CELLS", 1e9)

    length = railway.network.route_length((6, 20), (1, 10))

    assert length == pytest.approx(27.0)
    assert length - 19.0 > TURN_ROUND_CELLS


def test_a_way_that_needs_no_turning_round_has_no_turns() -> None:
    _, railway = joined_stations()
    way = railway.route((6, 3), (6, 20))
    assert way is not None

    turns = railway.network.turns(Train("Test", path=way[0], speeds=way[1]))

    assert turns == []


def test_a_train_stops_where_it_turns_round_on_plain_track_and_picks_up_speed_again() -> None:
    railway = branch_curving_one_way()
    railway.finances.cash = 1e9
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 20)), Stop((1, 10))]
    railway.run(0.01)
    turn = railway.network.turns(train)[0]
    train.along = turn - 0.01
    train.pace = 1.0

    railway.run(0.01)
    stopped = (train.along, train.pace)
    railway.run(0.01)

    assert stopped == (turn, 0.0)
    assert train.along > turn
    assert 0.0 < train.pace < 1.0
