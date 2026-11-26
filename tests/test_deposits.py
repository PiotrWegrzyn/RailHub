import numpy as np
import pytest

from builders import flat_map
from railroads.mapgen.deposits import allowed_land, place_deposits
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.dying import dies_after
from railroads.sim.events import want_factor
from railroads.sim.goods import NEEDS, Need
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import JEWELLERS, RESOURCES, WORKS, GrowthRule
from railroads.sim.tilemap import Deposit, Relief, TileMap
from railroads.sim.yearly import needs_of

GOLD_MINES = next(ladder for ladder in RESOURCES if ladder.kinds[0] == BuildingKind.GOLD_MINE)
TOOL_WORKS = next(ladder for ladder in WORKS if ladder.kinds[0] == BuildingKind.TOOL_WORKS)
PRINTING_WORKS = next(ladder for ladder in WORKS if ladder.kinds[0] == BuildingKind.PRINTING_WORKS)


def make_map(
    size: int = 400, mountains: bool = True, deposits: np.ndarray | None = None
) -> TileMap:
    grid = HexGrid(rows=size, cols=size, km_per_cell=2.0)
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    if mountains:
        relief[20:40, 20:40] = Relief.MOUNTAINS
    return flat_map(grid=grid, relief=relief, deposits=deposits)


def test_deposits_lie_only_on_land_that_allows_them() -> None:
    tile_map = make_map()

    layer = place_deposits(tile_map, np.random.default_rng(1))
    misplaced = [
        deposit
        for deposit, mask in allowed_land(tile_map).items()
        if ((layer == deposit) & ~mask).any()
    ]

    assert (layer == Deposit.GOLD).any()
    assert misplaced == []


def test_the_same_seed_gives_the_same_deposits() -> None:
    tile_map = make_map()

    first = place_deposits(tile_map, np.random.default_rng(5))
    second = place_deposits(tile_map, np.random.default_rng(5))

    assert np.array_equal(first, second)


def test_a_map_without_the_land_for_a_resource_simply_has_none() -> None:
    tile_map = make_map(mountains=False)

    layer = place_deposits(tile_map, np.random.default_rng(1))

    assert not np.isin(layer, [Deposit.GOLD, Deposit.IRON, Deposit.DIAMONDS]).any()
    assert (layer == Deposit.OIL).any()


def test_where_little_land_allows_a_resource_its_clusters_still_fit() -> None:
    tile_map = make_map()

    layer = place_deposits(tile_map, np.random.default_rng(2))

    assert (layer == Deposit.IRON).sum() >= 6


def test_a_mine_opens_only_on_its_deposit() -> None:
    deposits = np.zeros((12, 12), dtype=np.uint8)
    deposits[5, 5] = Deposit.GOLD

    weights = GOLD_MINES.land.weights(make_map(12, False, deposits))

    assert weights[5, 5] > 0
    assert weights[5, 6] == 0


def test_a_map_only_holds_its_few_jewellers() -> None:
    constructions = Constructions(make_map(200, False))
    rule = GrowthRule(trades=False, ladders=(JEWELLERS,), forests=None)
    growth = Growth(constructions, rule)
    cap = 40_000 // JEWELLERS.land_cells_each

    built = [
        growth._turn((100, 20 + 2 * i), JEWELLERS, np.random.default_rng(0)) for i in range(cap + 3)
    ]

    assert sum(1 for cells in built if cells) == cap
    assert growth.count(BuildingKind.JEWELLER).sum() == cap


def test_buildings_of_a_later_era_do_not_open_before_their_year() -> None:
    tile_map = make_map(40, False)
    constructions = Constructions(tile_map)
    busy = GrowthRule(trades=False, forests=None, ladders=(TOOL_WORKS,))
    growth = Growth(constructions, busy, start_year=1860)
    growth._places[0].chances[:] = 1.0

    growth.month(np.random.default_rng(0))

    assert growth.count(BuildingKind.TOOL_WORKS).sum() == 0
    assert growth.year == 1860


def test_needs_start_in_the_era_of_their_goods() -> None:
    before = needs_of(BuildingKind.FLATS, 1842)
    after = needs_of(BuildingKind.FLATS, 1843)

    assert Need.FUEL not in before
    assert after[Need.FUEL] == pytest.approx(
        NEEDS[BuildingKind.FLATS][Need.FUEL] * want_factor(BuildingKind.FLATS, Need.FUEL, 1843)
    )


def test_works_with_nothing_to_work_from_close_after_exactly_thirty_years_starving() -> None:
    constructions = Constructions(make_map(40, False))
    constructions.build(Building((20, 20), BuildingKind.SAWMILL))
    growth = Growth(constructions, GrowthRule(ladders=(), forests=None))
    standing = []

    for months in (dies_after(BuildingKind.SAWMILL) - 2, 1):
        for _ in range(months):
            growth.trade_goods()
        standing.append(int(growth.count(BuildingKind.SAWMILL).sum()))

    assert growth.died_out == 1
    assert standing == [1, 0]


@pytest.mark.parametrize(("gap", "opens"), [(6, False), (12, True)])
def test_jewellers_keep_far_apart(gap: int, opens: bool) -> None:
    constructions = Constructions(make_map(mountains=False))
    rule = GrowthRule(trades=False, ladders=(JEWELLERS,), forests=None)
    growth = Growth(constructions, rule)
    growth._turn((200, 200), JEWELLERS, np.random.default_rng(0))

    built = growth._turn((200, 200 + gap), JEWELLERS, np.random.default_rng(0))

    assert bool(built) == opens
