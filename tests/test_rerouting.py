import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions, TrackWidth
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import REROUTE_MARKS, REROUTE_WAIT, Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.signals import Signals, passes
from railroads.sim.trains import Stop, Train

WEST, EAST = (6, 3), (6, 20)
# Cells of the straight line between the stations, which the way round through row 2 misses.
STRAIGHT = {(6, col) for col in range(8, 16)}


class Holding:
    """Signals as far as finding a way round goes: the cells other trains hold, blocking every
    way across them."""

    def __init__(self, held: set[tuple[int, int]]) -> None:
        self.held = held

    def held_cells(self, train: Train) -> set[tuple[int, int]]:
        return set(self.held)

    def blocked(self, train: Train, cell: tuple[int, int], way: object) -> bool:
        return cell in self.held


def two_ways() -> Railway:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, WEST)
    build_station(growth, EAST)
    for start, end in ((WEST, EAST), (WEST, (2, 8)), ((2, 8), (2, 15)), ((2, 15), EAST)):
        route = plan_route(constructions, start, end)
        assert route is not None
        lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    return railway


def set_off(railway: Railway) -> Train:
    train = Train("Test", stops=[Stop(WEST), Stop(EAST)])
    way = railway.route(WEST, EAST)
    assert way is not None
    train.path, train.speeds = list(way[0]), list(way[1])
    railway.trains.append(train)
    return train


def rows(railway: Railway, train: Train) -> set[int]:
    return {railway.network.cell_at(point)[0] for point in train.path}


def test_a_train_takes_a_way_round_track_other_trains_hold() -> None:
    railway = two_ways()
    train = set_off(railway)
    assert rows(railway, train) == {6}

    railway._reroute(train, Holding(STRAIGHT))  # type: ignore[arg-type]

    assert 2 in rows(railway, train)
    assert train.path[0] == railway.route(WEST, EAST)[0][0]  # type: ignore[index]
    assert len(train.speeds) == len(train.path) - 1


def test_a_train_keeps_its_way_when_no_way_round_is_clearly_quicker() -> None:
    railway = two_ways()
    train = set_off(railway)
    path = list(train.path)

    railway._reroute(train, Holding({(6, 9)}))  # type: ignore[arg-type]

    assert train.path == path


@pytest.mark.parametrize(
    ("left", "looks"), [(0.9, 0), (0.5, 1), (0.25, 2), (0.1, 3), (0.01, REROUTE_MARKS)]
)
def test_a_train_looks_for_a_quicker_way_as_half_a_quarter_and_an_eighth_are_left(
    monkeypatch: pytest.MonkeyPatch, left: float, looks: int
) -> None:
    railway = two_ways()
    train = set_off(railway)
    looked: list[float] = []
    monkeypatch.setattr(railway, "_reroute", lambda train, signals: looked.append(train.along))
    length = railway.network.length(train)
    signals = Signals(railway.network, railway.trains)

    for share in (1.0, 0.9, 0.5, 0.25, 0.1, 0.01):
        if share < left:
            break
        train.along = length * (1 - share)
        railway._maybe_reroute(train, signals)

    assert train.reroute_marks == looks
    assert len(looked) == looks


def test_a_train_waiting_at_a_signal_looks_again_after_each_spell_of_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    railway = two_ways()
    train = set_off(railway)
    looked: list[float] = []
    monkeypatch.setattr(railway, "_reroute", lambda train, signals: looked.append(train.waited))
    signals = Signals(railway.network, railway.trains)
    train.giving_way = "Other"

    for waited in (0.01, 0.04, 0.06, 0.08, 0.11):
        train.waited = waited
        railway._maybe_reroute(train, signals)

    assert looked == [0.06, 0.11]
    assert train.reroute_waits == int(0.11 / REROUTE_WAIT)


def test_trains_going_the_other_way_on_double_track_dont_block_a_train() -> None:
    railway = line_of(TrackWidth.DOUBLE)
    going = Train("Going", stops=[Stop(WEST), Stop(EAST)])
    coming = Train("Coming", stops=[Stop(EAST), Stop(WEST)])
    railway.trains += [going, coming]
    for train, (start, end) in ((going, (WEST, EAST)), (coming, (EAST, WEST))):
        way = railway.route(start, end)
        assert way is not None
        train.path, train.speeds = list(way[0]), list(way[1])
    coming.along = railway.network.length(coming) / 2
    signals = Signals(railway.network, railway.trains)
    signals.let_on(coming, 1, coming.along + 3.0)
    held = next(
        crossing for crossing in passes(railway.network, coming) if crossing.end > coming.along
    )
    mine = next(c for c in passes(railway.network, going) if c.cell == held.cell)

    blocked = signals.blocked(going, mine.cell, (mine.entry, mine.exit))

    assert not blocked
    assert signals.held_by_others(going, [mine.cell]) == 1


def line_of(width: TrackWidth) -> Railway:
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
