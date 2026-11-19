from dataclasses import replace

import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import (
    LOCKED,
    Building,
    BuildingKind,
    Constructions,
    Station,
    Track,
)
from railroads.sim.game import Game
from railroads.sim.growth import Growth
from railroads.sim.lots import START_LOTS, Lots, fibonacci
from railroads.sim.tilemap import TileMap


def start_lots() -> Lots:
    return Lots.around_middle(flat_map(80, 70))


def test_a_game_starts_owning_a_honeycomb_of_seven_lots_holding_a_quarter_of_the_place() -> None:
    lots = start_lots()

    locked = lots.locked()
    top, _, rows, cols = lots.window()

    assert lots.owned == set(START_LOTS)
    assert locked is not None
    assert int((~locked).sum()) == pytest.approx(80 * 70 / 4, rel=0.03)
    assert top % 2 == 0
    assert locked[0, 0] and locked[-1, -1]
    assert not locked[rows // 2, cols // 2]


def test_each_lot_is_a_seventh_of_the_starting_land() -> None:
    lots = start_lots()
    window = lots.window()

    sizes = [int(lots.in_lots(window, {lot}).sum()) for lot in sorted(START_LOTS)]

    assert max(sizes) - min(sizes) <= 0.08 * max(sizes)
    assert sum(sizes) == int((~lots.locked()).sum())


def test_twelve_lots_share_a_side_with_the_starting_honeycomb() -> None:
    lots = start_lots()

    around = lots.next_to()

    assert len(around) == 12
    assert all(max(abs(q), abs(r), abs(q + r)) == 2 for q, r in around)


def test_only_lots_with_land_are_offered() -> None:
    lots = start_lots()
    land = np.zeros((80, 70), dtype=bool)
    land[:, :35] = True

    offered = lots.offered(land)

    assert (-2, 0) in offered
    assert (2, 0) not in offered


def test_a_lot_past_the_places_map_has_no_land() -> None:
    lots = start_lots()
    land = np.ones((80, 70), dtype=bool)

    share = lots.land_share((9, 0), land)

    assert share == 0.0


@pytest.mark.parametrize(("n", "number"), [(1, 1), (2, 1), (10, 55), (14, 377), (15, 610)])
def test_fibonacci_numbers_count_from_one_one(n: int, number: int) -> None:
    assert fibonacci(n) == number


def test_each_lot_costs_the_next_fibonacci_number_of_thousands() -> None:
    lots = start_lots()
    prices = [lots.price()]

    for lot in ((-2, 0), (-2, 1)):
        lots.buy(lot)
        prices.append(lots.price())

    assert prices == [377_000, 610_000, 987_000]


def test_a_lot_apart_from_the_players_land_cant_be_bought() -> None:
    lots = start_lots()

    with pytest.raises(ValueError, match="share a side"):
        lots.buy((5, 5))


def test_a_lot_bought_to_the_left_widens_the_map_leftwards_from_an_even_row() -> None:
    lots = start_lots()
    before = lots.window()

    lots.buy((-2, 1))
    after = lots.window()

    assert after[1] < before[1]
    assert after[1] + after[3] == before[1] + before[3]
    assert after[0] % 2 == 0
    assert (-2, 1) not in lots.locked_lots()
    assert lots.lot_at(*lots.middle_cell((-2, 1))) == (-2, 1)


def test_the_middle_cell_of_the_map_lies_in_the_middle_lot() -> None:
    lots = start_lots()
    _, _, rows, cols = lots.window()

    lot = lots.lot_at(*lots.middle_cell((0, 0)))

    assert lot == (0, 0)
    assert lots.lot_at(rows // 2, cols // 2) == (0, 0)


def half_locked_map() -> TileMap:
    tile_map = flat_map(12, 20)
    locked = np.zeros(tile_map.grid.shape, dtype=bool)
    locked[:, 10:] = True
    return replace(tile_map, locked=locked)


def test_nothing_can_be_built_on_locked_land() -> None:
    constructions = Constructions(half_locked_map())

    problems = (
        constructions.why_not(Station((4, 14))),
        constructions.why_not(Building((4, 14), BuildingKind.HOUSE)),
        constructions.why_not(Station((4, 4))),
    )

    assert problems == (LOCKED, LOCKED, None)


def test_track_cant_run_into_locked_land() -> None:
    constructions = Constructions(half_locked_map())
    grid = constructions.tile_map.grid
    east = next(
        point for point in grid.track_ends() if grid.track_exit((4, 9), point)[0] == (4, 10)
    )
    west = next(point for point in grid.track_ends() if grid.track_exit((4, 9), point)[0] == (4, 8))

    problem = constructions.why_not(Track((4, 9), tuple(sorted((west, east)))))

    assert problem == LOCKED


def test_nothing_grows_on_locked_land() -> None:
    constructions = Constructions(half_locked_map())
    growth = Growth(constructions, start_year=1850)

    growth.run(120, np.random.default_rng(1), step=12)

    cells = {building.cell for building in constructions if isinstance(building, Building)}
    assert cells
    assert all(col < 10 for _, col in cells)


def test_a_station_on_locked_land_is_refused_without_harm() -> None:
    game = Game(Constructions(half_locked_map()))
    game.finances.cash = 1_000_000.0

    done = game.build_station((4, 14))

    assert not done.ok
    assert game.railway.stations() == []


def test_track_is_never_planned_through_locked_land() -> None:
    game = Game(Constructions(half_locked_map()))
    game.finances.cash = 1_000_000.0
    game.build_station((4, 2))

    done = game.lay_track((4, 2), (4, 16))

    assert not done.ok
    assert not any(col >= 10 for _, col in game.constructions._track)
