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
from railroads.sim.growth import Growth
from railroads.sim.lots import Lots, fibonacci
from railroads.sim.tilemap import TileMap


def start_lots() -> Lots:
    return Lots.of_square(top=8, left=6, rows=8, cols=6)


def test_a_game_starts_owning_the_four_lots_of_its_square() -> None:
    lots = start_lots()

    window = lots.window()

    assert (lots.rows, lots.cols) == (4, 3)
    assert lots.owned == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert window == (8, 6, 8, 6)
    assert lots.locked() is None


def test_a_square_that_doesnt_split_into_even_lots_is_refused() -> None:
    with pytest.raises(ValueError, match="four even lots"):
        Lots.of_square(top=0, left=0, rows=6, cols=6)


def test_eight_lots_share_a_side_with_the_starting_square() -> None:
    lots = start_lots()

    around = lots.next_to()

    assert around == [(-1, 0), (-1, 1), (0, -1), (0, 2), (1, -1), (1, 2), (2, 0), (2, 1)]


def test_only_lots_with_land_are_offered() -> None:
    lots = start_lots()
    land = np.zeros((24, 18), dtype=bool)
    land[:, :12] = True

    offered = lots.offered(land)

    assert (0, 2) not in offered
    assert (1, 2) not in offered
    assert (0, -1) in offered
    assert (-1, 0) in offered


def test_a_lot_past_the_places_map_has_no_land() -> None:
    lots = start_lots()
    land = np.ones((16, 12), dtype=bool)

    share = lots.land_share((-3, 0), land)

    assert share == 0.0


@pytest.mark.parametrize(("n", "number"), [(1, 1), (2, 1), (10, 55), (14, 377), (15, 610)])
def test_fibonacci_numbers_count_from_one_one(n: int, number: int) -> None:
    assert fibonacci(n) == number


def test_each_lot_costs_the_next_fibonacci_number_of_thousands() -> None:
    lots = start_lots()
    prices = [lots.price()]

    for lot in ((0, -1), (-1, 0)):
        lots.buy(lot)
        prices.append(lots.price())

    assert prices == [377_000, 610_000, 987_000]


def test_a_lot_apart_from_the_players_land_cant_be_bought() -> None:
    lots = start_lots()

    with pytest.raises(ValueError, match="share a side"):
        lots.buy((5, 5))


def test_a_lot_bought_to_the_top_left_moves_the_maps_corner_and_locks_the_lot_beside_it() -> None:
    lots = start_lots()

    lots.buy((0, -1))
    lots.buy((-1, -1))
    locked = lots.locked()

    assert lots.corner() == (-1, -1)
    assert lots.window() == (4, 3, 12, 9)
    assert lots.cell_of((0, 0)) == (4, 3)
    assert locked is not None
    assert locked.shape == (12, 9)
    assert locked[0, 3:].all()
    assert not locked[0, :3].any()
    assert not locked[4:8].any()
    assert locked[8:, :3].all()


def test_a_map_cell_lies_in_the_lot_counted_from_the_maps_corner() -> None:
    lots = start_lots()
    lots.buy((0, -1))

    lot = lots.lot_at(5, 2)

    assert lot == (1, -1)


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
