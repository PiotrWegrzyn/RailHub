import pytest

from builders import flat_map
from railroads.sim.clock import SECONDS_PER_MONTH
from railroads.sim.constructions import Constructions, TrackWidth
from railroads.sim.game import Game
from railroads.sim.signals import blocks_of, cleared_signals, passes
from railroads.sim.trains import Stop, Train

WEST, EAST, NORTH, SOUTH = (8, 3), (8, 20), (1, 11), (15, 12)
STEP = 1 / 6 / SECONDS_PER_MONTH


def game_with(*lines: tuple[tuple[int, int], tuple[int, int], TrackWidth]) -> Game:
    game = Game(Constructions(flat_map(rows=16, cols=24)))
    game.finances.cash = 1e9
    for station in (WEST, EAST, NORTH, SOUTH):
        game.build_station(station)
    for start, end, width in lines:
        assert game.lay_track(start, end, width).ok
    return game


def shuttle(game: Game, start: tuple[int, int], end: tuple[int, int]) -> Train:
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop(start, wait_for_load=False), Stop(end, wait_for_load=False)]
    return train


def run_months(game: Game, months: float) -> None:
    for _ in range(int(months / STEP)):
        game.railway.run(STEP)


def test_trains_through_the_junctions_of_a_double_track_main_line_never_lock_each_other_up() -> (
    None
):
    game = game_with(
        (WEST, EAST, TrackWidth.DOUBLE),
        (NORTH, EAST, TrackWidth.SINGLE),
        (SOUTH, WEST, TrackWidth.SINGLE),
        (NORTH, WEST, TrackWidth.SINGLE),
    )
    pairs = [(WEST, EAST), (EAST, WEST), (NORTH, EAST), (EAST, NORTH), (NORTH, WEST)]
    pairs += [(WEST, NORTH), (SOUTH, WEST), (WEST, SOUTH)]
    trains = [shuttle(game, start, end) for start, end in pairs]

    run_months(game, 24)

    assert min(train.arrivals for train in trains) >= 3


def test_a_plain_line_has_signals_only_where_it_leaves_each_station() -> None:
    game = game_with((WEST, EAST, TrackWidth.SINGLE))

    spots = blocks_of(game.constructions).spots

    assert len(spots) == 2


def test_signals_stand_before_a_junction_on_every_way_into_it() -> None:
    game = game_with((WEST, EAST, TrackWidth.DOUBLE), (NORTH, EAST, TrackWidth.SINGLE))
    blocks = blocks_of(game.constructions)
    junction = {cell for cell, block in blocks.block.items() if block[0] == "junction"}

    into = {spot.cell for spot in blocks.spots if spot.cell in junction}

    assert junction
    assert into == junction


def test_a_signal_shows_clear_once_the_way_past_it_is_set_for_a_train_coming_to_it() -> None:
    game = game_with((WEST, EAST, TrackWidth.SINGLE))
    train = shuttle(game, WEST, EAST)
    before = cleared_signals(game.railway.trains)

    game.railway.run(STEP)

    first = passes(game.railway.network, train)[0]
    spots = {(spot.cell, spot.at) for spot in blocks_of(game.constructions).spots}
    assert (first.cell, first.entry) in spots
    assert (first.cell, first.entry) not in before
    assert (first.cell, first.entry) in cleared_signals(game.railway.trains)


@pytest.mark.parametrize("width", [TrackWidth.SINGLE, TrackWidth.DOUBLE])
def test_a_train_never_holds_track_another_holds_unless_they_pass_on_double_track(
    width: TrackWidth,
) -> None:
    game = game_with((WEST, EAST, width))
    trains = [shuttle(game, WEST, EAST), shuttle(game, EAST, WEST), shuttle(game, WEST, EAST)]
    shared = []

    for _ in range(int(6 / STEP)):
        game.railway.run(STEP)
        held = [
            {(cell, entry, exit_) for cell, entry, exit_, *_ in train.claims} for train in trains
        ]
        for one, other in ((0, 1), (1, 2)):
            cells = {c for c, *_ in held[one]} & {c for c, *_ in held[other]}
            shared += [(one, other, cell) for cell in cells]

    assert bool(shared) is (width == TrackWidth.DOUBLE)
