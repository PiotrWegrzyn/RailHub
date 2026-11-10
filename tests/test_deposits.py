import numpy as np
import pytest

from builders import flat_map
from railroads.mapgen.deposits import allowed_land, place_deposits
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.demand import StationDemand, price_of
from railroads.sim.goods import MAKES, Good, Need, needs_of
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import JEWELLERS, RESOURCES, WORKS, GrowthRule
from railroads.sim.tilemap import Deposit, Relief, TileMap

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

    allowed = allowed_land(tile_map)
    assert (layer == Deposit.GOLD).any()
    for deposit, mask in allowed.items():
        assert not ((layer == deposit) & ~mask).any()


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


def test_diamonds_cars_and_gold_are_worth_a_lot_but_only_where_someone_wants_them() -> None:
    wanted = StationDemand(capacity={Need.DIAMONDS: 2.0})
    elsewhere = StationDemand(capacity={Need.FOOD: 20.0})

    earned = wanted.pay(Need.DIAMONDS, Good.DIAMONDS, 1.0)

    assert earned == pytest.approx(price_of(Good.DIAMONDS))
    assert elsewhere.deliver(Good.DIAMONDS, 1.0) == 0.0
    assert price_of(Good.CARS) == 5000


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


def test_needs_start_in_their_year() -> None:
    before = needs_of(BuildingKind.FLATS, 1909)
    after = needs_of(BuildingKind.FLATS, 1910)

    assert Need.FUEL not in before
    assert after[Need.FUEL] == 5


def test_works_with_nothing_to_work_from_stay_open() -> None:
    constructions = Constructions(make_map(40, False))
    constructions.build(Building((20, 20), BuildingKind.TEXTILE_MILL))
    growth = Growth(constructions, GrowthRule(ladders=(), forests=None))

    for _ in range(240):
        growth.trade_goods()

    assert growth.count(BuildingKind.TEXTILE_MILL).sum() == 1


def test_flats_want_a_car_from_1930() -> None:
    before = needs_of(BuildingKind.FLATS, 1929)
    after = needs_of(BuildingKind.FLATS, 1930)

    assert Need.CARS not in before
    assert after[Need.CARS] == 1


@pytest.mark.parametrize(("gap", "opens"), [(6, False), (12, True)])
def test_jewellers_keep_far_apart(gap: int, opens: bool) -> None:
    constructions = Constructions(make_map(mountains=False))
    rule = GrowthRule(trades=False, ladders=(JEWELLERS,), forests=None)
    growth = Growth(constructions, rule)
    growth._turn((200, 200), JEWELLERS, np.random.default_rng(0))

    built = growth._turn((200, 200 + gap), JEWELLERS, np.random.default_rng(0))

    assert bool(built) == opens


def test_a_map_holds_only_a_few_printing_works_each_printing_thousands_of_newspapers() -> None:
    constructions = Constructions(make_map(mountains=False))
    rule = GrowthRule(trades=False, ladders=(PRINTING_WORKS,), forests=None)
    growth = Growth(constructions, rule)
    cap = 160_000 // PRINTING_WORKS.land_cells_each

    for i in range(cap + 3):
        growth._turn((200, 20 + 10 * i), PRINTING_WORKS, np.random.default_rng(0))

    assert growth.count(BuildingKind.PRINTING_WORKS).sum() == cap == 4
    assert MAKES[BuildingKind.PRINTING_WORKS][Good.NEWSPAPERS] == 3000
