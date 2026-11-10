import numpy as np
import pytest

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.geo import Bounds
from railroads.sim.goods import Good, Need, price_of
from railroads.sim.grids import HexGrid
from railroads.sim.growth import OWNER_SHARE, TRAIN_LOAD, Growth
from railroads.sim.ladders import TOWNS, GrowthRule
from railroads.sim.railway import ENGINES, Railway, Stop
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.tech import (
    FIRST_LEVEL,
    GOOD_TREES,
    NODES,
    TREES,
    Effect,
    Research,
    Tree,
    tree_of,
    xp_for,
)
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation


def make_map() -> TileMap:
    grid = HexGrid(rows=12, cols=30, km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )


def joined() -> Growth:
    constructions = Constructions(make_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 3))
    build_station(growth, (6, 20))
    route = plan_route(constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    growth.railway = Railway(constructions)
    return growth


def learned(tree: Tree, *keys: str) -> Research:
    research = Research()
    research.earn(tree, xp_for(len(keys)))
    for key in keys:
        assert research.take(key)
    return research


def test_every_tree_has_five_to_twenty_nodes_and_every_good_a_tree() -> None:
    sizes = [len(nodes) for nodes in TREES.values()]

    assert all(5 <= size <= 20 for size in sizes)
    assert set(GOOD_TREES) == set(Good)


@pytest.mark.parametrize(
    ("xp", "level"), [(0, 0), (FIRST_LEVEL - 1, 0), (FIRST_LEVEL, 1), (xp_for(5), 5)]
)
def test_levels_come_with_experience_each_dearer_than_the_last(xp: float, level: int) -> None:
    research = Research()

    research.earn(Tree.FOOD, xp)

    assert research.level(Tree.FOOD) == level
    assert research.points(Tree.FOOD) == level


def test_a_node_needs_a_point_and_the_nodes_it_builds_on() -> None:
    research = Research()
    research.earn(Tree.FOOD, xp_for(1))

    early = research.take("rotation")
    first = research.take("seed")
    broke = research.take("rotation")

    assert (early, first, broke) == (False, True, False)
    assert research.taken == {"seed"}


def test_bonuses_add_up_in_their_own_tree_and_from_nodes_that_reach_everywhere() -> None:
    research = learned(Tree.FOOD, "seed", "rotation")
    research.earn(Tree.COMMERCE, xp_for(1))
    research.take("investors")

    food = research.bonus(Effect.PRODUCTION, Tree.FOOD)
    metals = research.bonus(Effect.SHARE, Tree.METALS)

    assert food == pytest.approx(0.25)
    assert metals == pytest.approx(0.05)


def test_an_owned_farm_with_better_seed_sells_more_and_earns_food_experience() -> None:
    growth = joined()
    growth.build(Building((6, 22), BuildingKind.FARM))
    growth.own((6, 22), BuildingKind.FARM)
    growth.railway.research = learned(Tree.FOOD, "seed")
    before = growth.railway.research.xp[Tree.FOOD]

    growth.trade_goods()

    earned = 5 * 1.1 * price_of(Good.WHEAT) * OWNER_SHARE
    assert growth.owner_income[BuildingKind.FARM] == pytest.approx(earned)
    assert growth.railway.research.xp[Tree.FOOD] - before == pytest.approx(earned)


def test_market_stalls_raise_what_food_cargo_earns() -> None:
    growth = joined()
    railway = growth.railway
    railway.research = learned(Tree.FOOD, "seed", "stalls")
    railway.station_demand((6, 20)).capacity = {Need.FOOD: 100.0}
    plain = Railway(growth.constructions)
    plain.station_demand((6, 20)).capacity = {Need.FOOD: 100.0}

    railway.pay_for((6, 20), Need.FOOD, Good.WHEAT, 10.0)
    plain.pay_for((6, 20), Need.FOOD, Good.WHEAT, 10.0)

    assert railway.finances.month["cargo"] == pytest.approx(1.05 * plain.finances.month["cargo"])


def test_only_the_first_engine_can_be_bought_until_its_tech_and_year_have_come() -> None:
    railway = joined().railway
    railway.year = 1879
    railway.research = learned(Tree.METALS, "boilers")
    early = railway.engines()

    railway.year = 1880

    assert early == ["planet"]
    assert railway.engines() == ["planet", "mogul"]
    assert railway.buy("pacific") is None


def test_a_mogul_pulls_more_cars_and_steel_couplings_one_more() -> None:
    railway = joined().railway
    railway.year = 1880
    railway.research = learned(Tree.METALS, "boilers", "bessemer", "couplings")
    railway.finances.earn(100_000.0, "test")

    train = railway.buy("mogul")

    assert train is not None
    assert railway.most_cars(train) == ENGINES["mogul"].cars + 1
    stop = Stop((6, 3))
    assert all(stop.add_car(Good.COAL, railway.most_cars(train)) for _ in range(9))
    assert not stop.add_car(Good.COAL, railway.most_cars(train))


def test_bogie_coaches_carry_more_passengers_but_freight_cars_stay_the_same() -> None:
    railway = joined().railway
    railway.research = learned(Tree.PASSENGERS, "coaches")

    coach = railway.car_capacity(Good.PASSENGERS)
    wagon = railway.car_capacity(Good.COAL)

    assert coach == pytest.approx(60.0)
    assert wagon == pytest.approx(50.0)


def test_every_building_that_makes_something_has_a_tree_and_shops_are_commerce() -> None:
    trees = {kind: tree_of(kind) for kind in BuildingKind}

    assert trees[BuildingKind.BAKERY] == Tree.FOOD
    assert trees[BuildingKind.STEEL_MILL] == Tree.METALS
    assert trees[BuildingKind.HOTEL] == Tree.COMMERCE
    assert trees[BuildingKind.HOUSE] is None
    assert all(NODES[key][0] in Tree for key in NODES)


def test_a_tenement_rising_by_a_station_gives_two_trains_worth_of_experience() -> None:
    growth = joined()
    grid = growth.constructions.tile_map.grid
    centre = (5, 20)
    spots = [centre, *(cell for cell in grid.neighbours(*centre) if cell != (6, 20))]
    for i in range(16):
        growth.build(Building(spots[i % len(spots)], BuildingKind.HOUSE))
    growth.trade_goods()
    before = sum(growth.railway.research.xp.values())

    growth._turn(centre, TOWNS, np.random.default_rng(0))

    gained = sum(growth.railway.research.xp.values()) - before
    _, kind, xp = growth.growth_rewards[-1]
    assert kind == BuildingKind.TENEMENT
    assert gained == pytest.approx(xp) and xp > 0
    worth = growth._load_worth(BuildingKind.TENEMENT)
    assert xp == pytest.approx(2 * TRAIN_LOAD * sum(worth.values()))


def test_nothing_rising_out_of_reach_of_the_stations_gives_experience() -> None:
    growth = joined()
    grid = growth.constructions.tile_map.grid
    spots = [(1, 12), *grid.neighbours(1, 12)]
    for i in range(16):
        growth.build(Building(spots[i % len(spots)], BuildingKind.HOUSE))
    growth.trade_goods()

    growth._turn((1, 12), TOWNS, np.random.default_rng(0))

    assert growth.count(BuildingKind.TENEMENT).sum() == 1
    assert growth.growth_rewards == []


def test_electric_engines_find_a_way_only_along_electrified_track() -> None:
    constructions = Constructions(make_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    for cell in ((6, 3), (6, 20), (2, 3), (2, 20)):
        build_station(growth, cell)
    for start, end, electric in (((6, 3), (6, 20), False), ((2, 3), (2, 20), True)):
        route = plan_route(constructions, start, end, electric=electric)
        assert route is not None
        lay_route(growth, route)
    railway = Railway(constructions)

    plain = railway.route((6, 3), (6, 20), electric=True)
    wired = railway.route((2, 3), (2, 20), electric=True)

    assert plain is None
    assert wired is not None
    assert railway.route((6, 3), (6, 20)) is not None


def test_electrified_track_costs_more_upkeep_than_plain_track() -> None:
    plain, wired = Constructions(make_map()), Constructions(make_map())
    for constructions, electric in ((plain, False), (wired, True)):
        growth = Growth(constructions, GrowthRule(trades=False))
        route = plan_route(constructions, (6, 3), (6, 20), electric=electric)
        assert route is not None
        lay_route(growth, route)
    railways = [Railway(plain), Railway(wired)]

    for railway in railways:
        railway.turn_month({})

    paid = [-railway.finances.month["track upkeep"] for railway in railways]
    assert paid[1] == pytest.approx(2.5 * paid[0])
