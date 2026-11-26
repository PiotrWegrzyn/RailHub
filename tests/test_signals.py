import pytest

from builders import flat_map
from railroads.sim.clock import SECONDS_PER_MONTH
from railroads.sim.constructions import Constructions, TrackWidth
from railroads.sim.game import Game
from railroads.sim.goods import Good
from railroads.sim.railway import QUADRUPLE_ERA, Railway
from railroads.sim.signals import (
    FAST_LANE,
    SLOW_LANE,
    blocks_of,
    cleared_signals,
    passes,
)
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


def test_a_train_coming_down_a_single_line_leaves_the_junction_at_its_end_to_others_until_near() -> (
    None
):
    junction = (8, 15)
    game = game_with(
        (WEST, junction, TrackWidth.SINGLE),
        (junction, EAST, TrackWidth.DOUBLE),
        (NORTH, junction, TrackWidth.SINGLE),
    )
    coming = shuttle(game, NORTH, EAST)
    run_months(game, 0.01)
    leaving = shuttle(game, EAST, WEST)
    at_junction = next(p.start for p in passes(game.railway.network, coming) if p.cell == junction)
    coming_was = None

    for _ in range(int(1 / STEP)):
        game.railway.run(STEP)
        if junction in {claim[0] for claim in leaving.claims}:
            coming_was = coming.along
            break

    assert coming_was is not None
    assert coming_was < at_junction


def test_a_train_on_a_single_line_holds_only_its_line_while_a_train_from_a_branch_goes_ahead() -> (
    None
):
    junction = (8, 8)
    game = game_with(
        (EAST, junction, TrackWidth.SINGLE),
        (junction, WEST, TrackWidth.SINGLE),
        (SOUTH, junction, TrackWidth.SINGLE),
    )
    far = shuttle(game, EAST, WEST)
    run_months(game, 0.01)
    near = shuttle(game, SOUTH, WEST)
    at_junction = next(p.start for p in passes(game.railway.network, far) if p.cell == junction)
    far_was = None

    for _ in range(int(1 / STEP)):
        game.railway.run(STEP)
        if junction in {claim[0] for claim in near.claims}:
            far_was = far.along
            break

    assert far_was is not None
    assert far_was < at_junction


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


@pytest.mark.parametrize(("width", "kept"), [(TrackWidth.DOUBLE, True), (TrackWidth.SINGLE, False)])
def test_a_short_double_line_between_junctions_keeps_its_own_signals_and_a_single_one_does_not(
    width: TrackWidth, kept: bool
) -> None:
    game = game_with((WEST, EAST, width))
    assert game.lay_track((8, 9), NORTH, TrackWidth.SINGLE).ok
    assert game.lay_track((8, 12), SOUTH, TrackWidth.SINGLE).ok
    between = {(8, 10), (8, 11)}

    blocks = blocks_of(game.constructions)

    signalled = {spot.cell for spot in blocks.spots} & between
    assert (signalled == between) == kept
    assert (blocks.block[(8, 10)][0] == "junction") != kept


def test_plain_line_beside_where_junction_track_runs_through_a_corner_belongs_to_the_junction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Railway, "quadruple_open", property(lambda _: True))
    game = Game(Constructions(flat_map(rows=14, cols=18)))
    game.finances.cash = 1e9
    one, two, three, four, bend = (5, 7), (9, 7), (5, 15), (4, 10), (8, 12)
    for station in (one, two, three, four):
        game.build_station(station)
    for start, end, width in (
        (one, two, TrackWidth.SINGLE),
        (one, three, TrackWidth.SINGLE),
        (three, four, TrackWidth.SINGLE),
        (two, four, TrackWidth.SINGLE),
        (two, bend, TrackWidth.QUADRUPLE),
        (bend, one, TrackWidth.QUADRUPLE),
    ):
        assert game.lay_track(start, end, width).ok

    blocks = blocks_of(game.constructions)

    junction = blocks.block[(7, 7)]
    assert blocks.block[(6, 8)] == junction
    assert blocks.block[(8, 8)] == junction


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


def quadruple_game(monkeypatch: pytest.MonkeyPatch) -> Game:
    monkeypatch.setattr(Railway, "quadruple_open", property(lambda _: True))
    game = Game(Constructions(flat_map(rows=16, cols=24)))
    game.finances.cash = 1e9
    for station in (WEST, EAST):
        game.build_station(station)
    for start, end, width in (
        (WEST, (8, 6), TrackWidth.DOUBLE),
        ((8, 6), (8, 17), TrackWidth.QUADRUPLE),
        ((8, 17), EAST, TrackWidth.DOUBLE),
    ):
        assert game.lay_track(start, end, width).ok
    return game


def test_an_express_takes_the_fast_track_and_a_goods_train_the_slow_one_side_by_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = quadruple_game(monkeypatch)
    express, goods = shuttle(game, WEST, EAST), shuttle(game, WEST, EAST)
    express.engine = "single"
    side_by_side = []

    for _ in range(int(2 / STEP)):
        game.railway.run(STEP)
        side_by_side += alongside(game, express, goods)

    assert (FAST_LANE, SLOW_LANE) in side_by_side


def alongside(game: Game, one: Train, other: Train) -> list[tuple[int, int]]:
    lanes = [{cell: lane for cell, *_, lane in train.claims} for train in (one, other)]
    return [
        (lanes[0][cell], lanes[1][cell])
        for cell in lanes[0].keys() & lanes[1].keys()
        if cell in blocks_of(game.constructions).four_lanes
    ]


def test_a_train_coming_onto_quadruple_track_behind_another_takes_its_other_track_that_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = quadruple_game(monkeypatch)
    ahead = shuttle(game, WEST, EAST)
    ahead.engine = "goods"
    run_months(game, 0.6)
    behind = shuttle(game, WEST, EAST)
    behind.engine = "pacific"
    side_by_side = []

    for _ in range(int(1 / STEP)):
        game.railway.run(STEP)
        side_by_side += alongside(game, ahead, behind)

    assert (SLOW_LANE, FAST_LANE) in side_by_side


def test_a_quicker_train_passes_a_slower_one_ahead_on_the_other_track_of_quadruple_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = quadruple_game(monkeypatch)
    ahead = shuttle(game, WEST, EAST)
    ahead.engine = "goods"
    run_months(game, 0.5)
    behind = shuttle(game, WEST, EAST)
    behind.engine = "streamliner"

    for _ in range(int(6 / STEP)):
        game.railway.run(STEP)
        if ahead.arrivals or behind.arrivals:
            break

    assert behind.arrivals == 1
    assert ahead.arrivals == 0


def test_signals_stand_where_the_two_tracks_each_way_of_quadruple_track_come_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = quadruple_game(monkeypatch)
    blocks = blocks_of(game.constructions)

    ends = {spot.cell for spot in blocks.spots if spot.cell in blocks.four_lanes}

    assert {blocks.block[cell][0] for cell in blocks.four_lanes} == {"lanes"}
    assert len(ends) == 2


def test_quadruple_track_opens_with_its_era() -> None:
    game = Game(Constructions(flat_map(rows=16, cols=24)))
    game.finances.cash = 1e9
    game.build_station(WEST)
    game.build_station(EAST)

    done = game.lay_track(WEST, EAST, TrackWidth.QUADRUPLE)

    assert not done.ok
    assert str(QUADRUPLE_ERA) in done.message


def test_trains_set_off_together_and_brought_level_behind_another_never_wait_for_each_other() -> (
    None
):
    game = Game(Constructions(flat_map(rows=14, cols=18)))
    game.finances.cash = 1e9
    north, east, west, bend = (4, 15), (7, 16), (5, 6), (5, 12)
    for station in (north, east, west):
        game.build_station(station)
    for start, end, width in (
        (north, east, TrackWidth.SINGLE),
        (west, north, TrackWidth.SINGLE),
        (east, north, TrackWidth.DOUBLE),
        (west, bend, TrackWidth.DOUBLE),
        (bend, north, TrackWidth.SINGLE),
        (east, bend, TrackWidth.SINGLE),
        (bend, west, TrackWidth.SINGLE),
    ):
        assert game.lay_track(start, end, width).ok
    rounds = [
        ("tank", [north, west, east]),
        ("planet", [east, north]),
        ("tank", [east, north]),
        ("planet", [west, east]),
        ("planet", [west, north, east]),
        ("planet", [north, east]),
        ("planet", [north, west, east]),
        ("tank", [east, west, north]),
    ]
    for engine, stops in rounds:
        train = game.railway.buy(engine)
        assert train is not None
        train.stops = [Stop(cell) for cell in stops]

    for _ in range(3 * 240):
        game.railway.run(1 / 240)

    following = {
        (train.name, train.status.removeprefix("Following "))
        for train in game.railway.trains
        if train.status.startswith("Following ")
    }
    assert not any((other, name) in following for name, other in following)


def test_a_quicker_train_let_on_behind_another_in_a_station_never_passes_it_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Railway, "quadruple_open", property(lambda _: True))
    game = Game(Constructions(flat_map(rows=14, cols=18)))
    game.finances.cash = 1e9
    home, east, north, south = (4, 6), (4, 11), (1, 12), (12, 4)
    for station in (home, north, east, south):
        game.build_station(station)
    for start, end, width in (
        (south, north, TrackWidth.DOUBLE),
        (home, north, TrackWidth.DOUBLE),
        (home, east, TrackWidth.QUADRUPLE),
        (east, north, TrackWidth.DOUBLE),
    ):
        assert game.lay_track(start, end, width).ok
    rounds = [
        ("tank", [north, south, home, east]),
        ("planet", [home, east]),
        ("planet", [home, east, south, north]),
        ("planet", [east, south, home, north]),
        ("tank", [east, south, north]),
        ("tank", [home, north]),
        ("tank", [home, east, south, north]),
        ("planet", [south, north, home]),
    ]
    trains = []
    for engine, stops in rounds:
        train = game.railway.buy(engine)
        assert train is not None
        train.stops = [Stop(cell) for cell in stops]
        trains.append(train)

    for _ in range(3 * 240):
        game.railway.run(1 / 240)

    assert trains[1].arrivals and trains[5].arrivals


def test_a_train_following_one_that_turns_back_in_a_short_station_leaves_it_the_way_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Railway, "quadruple_open", property(lambda _: True))
    game = Game(Constructions(flat_map(rows=14, cols=18)))
    game.finances.cash = 1e9
    west, north, south = (3, 8), (3, 16), (11, 16)
    for station in (west, north, south):
        game.build_station(station)
    for start, end, width in (
        (west, (4, 4), TrackWidth.QUADRUPLE),
        ((4, 4), north, TrackWidth.DOUBLE),
        (north, south, TrackWidth.DOUBLE),
        (south, west, TrackWidth.DOUBLE),
        (south, (10, 13), TrackWidth.SINGLE),
        ((10, 13), west, TrackWidth.QUADRUPLE),
    ):
        assert game.lay_track(start, end, width).ok
    rounds = [
        ("planet", [south, north, west]),
        ("planet", [south, north]),
        ("tank", [south, north, west]),
        ("planet", [south, west]),
        ("planet", [west, north, south]),
        ("tank", [north, south, west]),
        ("planet", [north, west]),
    ]
    trains = []
    for engine, stops in rounds:
        train = game.railway.buy(engine)
        assert train is not None
        for cell in stops:
            stop = Stop(cell)
            train.stops.append(stop)
            game.railway.add_car(train, stop, Good.PASSENGERS)
        trains.append(train)
    planets = [train for train in trains if train.engine == "planet"]
    for _ in range(8 * 240):
        game.railway.run(1 / 240)
    before = [train.arrivals for train in planets]

    for _ in range(3 * 240):
        game.railway.run(1 / 240)

    assert all(train.arrivals > old for train, old in zip(planets, before, strict=True))


def test_a_train_stands_at_the_end_of_a_single_line_only_while_the_way_on_is_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Railway, "quadruple_open", property(lambda _: True))
    game = Game(Constructions(flat_map(rows=14, cols=18)))
    game.finances.cash = 1e9
    east, south, middle, west = (1, 14), (8, 8), (7, 11), (1, 7)
    for station in (east, south, middle, west):
        game.build_station(station)
    for start, end, width in (
        (south, (7, 13), TrackWidth.DOUBLE),
        ((7, 13), middle, TrackWidth.QUADRUPLE),
        (south, east, TrackWidth.DOUBLE),
        (west, (3, 9), TrackWidth.DOUBLE),
        ((3, 9), middle, TrackWidth.SINGLE),
        (south, west, TrackWidth.QUADRUPLE),
        (south, (12, 13), TrackWidth.DOUBLE),
        ((12, 13), east, TrackWidth.DOUBLE),
        (east, middle, TrackWidth.QUADRUPLE),
        (middle, (9, 7), TrackWidth.SINGLE),
        ((9, 7), east, TrackWidth.DOUBLE),
    ):
        assert game.lay_track(start, end, width).ok
    rounds = [
        ("planet", [south, middle, east, west]),
        ("planet", [east, west, south]),
        ("planet", [south, middle, east, west]),
        ("planet", [east, south, west]),
        ("planet", [west, middle, east]),
        ("tank", [middle, east]),
        ("planet", [middle, west]),
        ("planet", [west, middle]),
        ("tank", [middle, east, south]),
        ("planet", [south, east, middle, west]),
    ]
    planets = []
    for engine, stops in rounds:
        train = game.railway.buy(engine)
        assert train is not None
        for cell in stops:
            stop = Stop(cell)
            train.stops.append(stop)
            game.railway.add_car(train, stop, Good.PASSENGERS)
        if engine == "planet":
            planets.append(train)
    for _ in range(4 * 240):
        game.railway.run(1 / 240)
    before = [train.arrivals for train in planets]

    for _ in range(3 * 240):
        game.railway.run(1 / 240)

    assert all(train.arrivals > old for train, old in zip(planets, before, strict=True))


def fleet(game: Game, rounds: list[tuple[str, list[tuple[int, int]]]]) -> list[Train]:
    trains = []
    for engine, stops in rounds:
        train = game.railway.buy(engine)
        assert train is not None
        for cell in stops:
            stop = Stop(cell)
            train.stops.append(stop)
            game.railway.add_car(train, stop, Good.PASSENGERS)
        trains.append(train)
    return trains


def test_no_train_stands_beside_track_through_a_corner_where_it_would_block_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Railway, "quadruple_open", property(lambda _: True))
    game = Game(Constructions(flat_map(rows=14, cols=18)))
    game.finances.cash = 1e9
    a, b, c, d, e, f = (4, 9), (12, 9), (10, 5), (1, 16), (9, 1), (4, 5)
    for station in (a, b, c, d, e, f):
        game.build_station(station)
    for start, end, width in (
        (e, b, TrackWidth.QUADRUPLE),
        (c, (2, 4), TrackWidth.DOUBLE),
        ((2, 4), f, TrackWidth.QUADRUPLE),
        (c, f, TrackWidth.QUADRUPLE),
        (f, d, TrackWidth.SINGLE),
        (f, (11, 12), TrackWidth.SINGLE),
        ((11, 12), b, TrackWidth.QUADRUPLE),
        (f, a, TrackWidth.SINGLE),
        (e, f, TrackWidth.SINGLE),
    ):
        assert game.lay_track(start, end, width).ok
    trains = fleet(
        game,
        [
            ("planet", [b, f, a, c]),
            ("tank", [b, c]),
            ("planet", [e, f, a]),
            ("tank", [f, e, b, a]),
            ("tank", [c, a, b, e]),
            ("planet", [f, d, a, e]),
            ("planet", [e, d, b, c]),
            ("planet", [f, c, b, a]),
            ("tank", [f, b, a]),
            ("tank", [f, c]),
        ],
    )
    for _ in range(4 * 240):
        game.railway.run(1 / 240)
    earlier = [train.arrivals for train in trains]
    for _ in range(3 * 240):
        game.railway.run(1 / 240)
    before = [train.arrivals for train in trains]

    for _ in range(3 * 240):
        game.railway.run(1 / 240)

    running = [
        (train, old)
        for train, old, first in zip(trains, before, earlier, strict=True)
        if old > first
    ]
    assert running
    assert all(train.arrivals > old for train, old in running)


def test_no_train_stands_on_the_way_on_kept_by_a_train_at_the_end_of_a_single_line() -> None:
    game = Game(Constructions(flat_map(rows=14, cols=18)))
    game.finances.cash = 1e9
    a, b, c, d = (8, 5), (10, 12), (1, 9), (4, 8)
    for station in (a, b, c, d):
        game.build_station(station)
    for start, end, width in (
        (c, a, TrackWidth.SINGLE),
        (a, d, TrackWidth.DOUBLE),
        (b, c, TrackWidth.DOUBLE),
        (b, d, TrackWidth.SINGLE),
        (a, c, TrackWidth.DOUBLE),
        (a, (4, 5), TrackWidth.DOUBLE),
        ((4, 5), c, TrackWidth.SINGLE),
    ):
        assert game.lay_track(start, end, width).ok
    trains = fleet(
        game,
        [
            ("tank", [d, c]),
            ("tank", [c, b]),
            ("tank", [b, a, d, c]),
            ("planet", [c, a]),
            ("planet", [c, d, a]),
            ("planet", [a, d, b]),
            ("tank", [b, d, a]),
            ("planet", [d, c, b]),
            ("tank", [c, a, d]),
            ("tank", [b, a]),
        ],
    )
    for _ in range(5 * 240):
        game.railway.run(1 / 240)
    before = [train.arrivals for train in trains]

    for _ in range(6 * 240):
        game.railway.run(1 / 240)

    assert all(train.arrivals > old for train, old in zip(trains, before, strict=True))


def test_two_trains_coming_together_at_a_junction_of_quadruple_track_go_through_side_by_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = quadruple_game(monkeypatch)
    game.build_station(NORTH)
    assert game.lay_track((8, 14), (4, 15), TrackWidth.QUADRUPLE).ok
    assert game.lay_track((4, 15), NORTH, TrackWidth.DOUBLE).ok
    branch = shuttle(game, NORTH, WEST)
    for _ in range(120):
        game.railway.run(STEP)
    main = shuttle(game, EAST, WEST)
    together = waited = 0

    for _ in range(900):
        game.railway.run(STEP)
        held = [{claim[0] for claim in train.claims} for train in (main, branch)]
        together += (8, 14) in held[0] and (8, 14) in held[1]
        waited += any(train.status.startswith("Waiting") for train in (main, branch))

    assert together > 0
    assert waited == 0
