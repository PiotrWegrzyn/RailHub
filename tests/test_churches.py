import numpy as np
import pytest

from builders import flat_map
from railroads.mapgen.deposits import place_deposits
from railroads.sim.constructions import BUILDING_RULES, Building, BuildingKind, Constructions
from railroads.sim.economy import Economy, StationOrder
from railroads.sim.goods import NEEDS, Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import CHURCHES, FARMS, ORCHARDS, PASSENGERS_PER_POINT, POINTS, Points
from railroads.sim.tilemap import Deposit, Relief, TileMap

RANKS = (
    BuildingKind.CHURCH,
    BuildingKind.BASILICA,
    BuildingKind.CATHEDRAL,
    BuildingKind.ST_PAULS,
)


def make_map(size: int = 40) -> TileMap:
    return flat_map(size, size)


def settle(constructions: Constructions, cell: tuple[int, int], points: int) -> None:
    """Puts blocks of flats, and tenements and houses for the rest, round a cell."""
    flats, rest = divmod(points, 18)
    around = constructions.tile_map.grid.neighbours(*cell)
    for index in range(flats):
        constructions.build(Building(around[index // 6], BuildingKind.FLATS))
    others = [BuildingKind.TENEMENT] * (rest // 6) + [BuildingKind.HOUSE] * (rest % 6)
    for index, kind in enumerate(others):
        constructions.build(Building(around[4 + index // 6], kind))


def count(constructions: Constructions, kind: BuildingKind) -> int:
    return sum(1 for b in constructions if isinstance(b, Building) and b.kind == kind)


@pytest.mark.parametrize(("people", "opens"), [(9, False), (10, True)])
def test_a_church_needs_ten_population_nearby(people: int, opens: bool) -> None:
    constructions = Constructions(make_map())
    settle(constructions, (10, 10), people)
    growth = Growth(constructions)

    chance = growth.chances(CHURCHES)[10, 10]

    assert bool(chance > 0) is opens


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize(("steps", "opens"), [(4, False), (5, True)])
def test_no_church_opens_within_four_hexes_of_another(
    rank: BuildingKind, steps: int, opens: bool
) -> None:
    constructions = Constructions(make_map())
    grid = constructions.tile_map.grid
    cell = next(c for c, away in grid.within((20, 20), steps).items() if away == steps)
    constructions.build(Building((20, 20), rank))
    growth = Growth(constructions)

    growth._turn(cell, CHURCHES, np.random.default_rng(0))

    assert (Building(cell, BuildingKind.CHURCH) in constructions.at(cell)) is opens


@pytest.mark.parametrize(
    ("small", "big", "people", "rises"),
    [
        (BuildingKind.CHURCH, BuildingKind.BASILICA, 119, False),
        (BuildingKind.CHURCH, BuildingKind.BASILICA, 120, True),
        (BuildingKind.BASILICA, BuildingKind.CATHEDRAL, 239, False),
        (BuildingKind.BASILICA, BuildingKind.CATHEDRAL, 240, True),
        (BuildingKind.CATHEDRAL, BuildingKind.ST_PAULS, 399, False),
        (BuildingKind.CATHEDRAL, BuildingKind.ST_PAULS, 400, True),
    ],
)
def test_churches_rise_with_the_people_around_them(
    small: BuildingKind, big: BuildingKind, people: int, rises: bool
) -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((20, 20), small))
    settle(constructions, (20, 20), people)

    Growth(constructions)

    assert count(constructions, big) == int(rises)
    assert count(constructions, small) == int(not rises)


def test_only_the_first_cathedral_to_reach_four_hundred_becomes_st_pauls() -> None:
    constructions = Constructions(make_map())
    for cell, people in (((10, 10), 400), ((30, 30), 420)):
        constructions.build(Building(cell, BuildingKind.CATHEDRAL))
        settle(constructions, cell, people)
    growth = Growth(constructions)

    growth.trade_goods()

    assert [b.cell for b in constructions if isinstance(b, Building) and b.kind == RANKS[3]] == [
        (30, 30)
    ]
    assert count(constructions, BuildingKind.CATHEDRAL) == 1


@pytest.mark.parametrize(
    ("kind", "people"),
    [
        (BuildingKind.BASILICA, 20),
        (BuildingKind.CATHEDRAL, 50),
        (BuildingKind.ST_PAULS, 2137),
    ],
)
def test_churches_send_and_want_their_congregation(kind: BuildingKind, people: int) -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), kind))
    growth = Growth(constructions)
    flat = np.array([np.ravel_multi_index((5, 5), constructions.tile_map.grid.shape)])

    travellers = growth.station_trade.travellers_in(flat)

    assert travellers[Good.PASSENGERS] == people


@pytest.mark.parametrize("need", [Need.WAX, Need.INCENSE])
def test_each_rank_of_church_wants_more_wax_and_incense(need: Need) -> None:
    wanted = [NEEDS[rank][need] for rank in RANKS]

    assert wanted == sorted(set(wanted))


@pytest.mark.parametrize(
    ("grapes", "barrels", "wine"), [(30.0, 0.0, 30.0), (0.0, 30.0, 0.0), (30.0, 30.0, 60.0)]
)
def test_wine_is_made_of_grapes_and_barrels_make_up_to_twice_as_much(
    grapes: float, barrels: float, wine: float
) -> None:
    tile_map = make_map(12)
    economy = Economy(tile_map)
    flat = int(np.ravel_multi_index((5, 5), tile_map.grid.shape))
    economy.stock[(BuildingKind.WINERY, Need.GRAPES)][flat] = grapes
    economy.stock[(BuildingKind.WINERY, Need.BARRELS)][flat] = barrels
    wineries = np.zeros(tile_map.grid.shape, dtype=np.int64)
    wineries[5, 5] = 1

    trade = economy.trade({BuildingKind.WINERY: wineries}, {(5, 6): StationOrder({}, 100.0)})

    made, _ = trade.goods_gave[(BuildingKind.WINERY, Good.WINE)]
    assert made.sum() == wine


def test_honey_feeds_houses_and_wax_lights_churches() -> None:
    tile_map = make_map(12)
    economy = Economy(tile_map)
    neighbours = tile_map.grid.neighbours(5, 5)
    houses = np.zeros(tile_map.grid.shape, dtype=np.int64)
    houses[neighbours[0]] = 4
    churches = np.zeros(tile_map.grid.shape, dtype=np.int64)
    churches[neighbours[1]] = 1
    honey_and_wax = {Good.HONEY: 2.0, Good.WAX: 1.0}

    trade = economy.trade(
        {BuildingKind.HOUSE: houses, BuildingKind.CHURCH: churches},
        {(5, 5): StationOrder(honey_and_wax, 0.0)},
    )

    assert trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)][0].sum() == 2.0
    assert trade.goods_got[(BuildingKind.CHURCH, Need.WAX)][0].sum() == 1.0


@pytest.mark.parametrize("seed", range(6))
def test_incense_lies_in_only_a_handful_of_cells(seed: int) -> None:
    layer = place_deposits(make_map(400), np.random.default_rng(seed))

    assert int((layer == Deposit.INCENSE).sum()) <= 12


def test_a_house_is_one_population_and_one_traveller_and_wants_food_heat_and_small_comforts() -> (
    None
):
    assert POINTS[BuildingKind.HOUSE] == (Points.POPULATION, 1)
    assert PASSENGERS_PER_POINT[Points.POPULATION] == 1
    assert NEEDS[BuildingKind.HOUSE] == {
        Need.FOOD: 10,
        Need.HEATING: 1,
        Need.BEER: 1,
        Need.NEWSPAPERS: 1,
        Need.CLOTHES: 1,
        Need.POTS: 0.5,
        Need.FURNITURE: 0.5,
        Need.VODKA: 0.5,
        Need.REFINED_SUGAR: 0.5,
        Need.BRICKS: 0.5,
        Need.SHOES: 0.5,
        Need.GLASS: 0.2,
        Need.CASTINGS: 0.2,
    }


@pytest.mark.parametrize(
    ("kind", "space"),
    [
        (BuildingKind.HOUSE, 1),
        (BuildingKind.TENEMENT, 1),
        (BuildingKind.FLATS, 1),
        (BuildingKind.FARM, 6),
        (BuildingKind.ORCHARD, 6),
        (BuildingKind.SUGAR, 6),
        (BuildingKind.VINEYARD, 6),
        (BuildingKind.FACTORY, 3),
        (BuildingKind.BIG_FACTORY, 3),
        (BuildingKind.MEGA_FACTORY, 6),
        (BuildingKind.FOOD_PLANT, 3),
        (BuildingKind.STEEL_MILL, 3),
        (BuildingKind.LUMBER_MILL, 3),
        (BuildingKind.TEXTILE_MILL, 3),
    ],
)
def test_buildings_take_up_their_share_of_a_cell(kind: BuildingKind, space: int) -> None:
    assert BUILDING_RULES[kind].space == space


def test_hills_favour_orchards_over_farms() -> None:
    assert ORCHARDS.land.relief[Relief.HILLS] == 1.5
    assert ORCHARDS.land.relief[Relief.FLAT] == 0.4
    assert FARMS.land.relief[Relief.HILLS] == 0.25
