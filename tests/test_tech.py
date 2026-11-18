import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import Trade
from railroads.sim.finance import ELECTRIC_TRACK_UPKEEP, TRACK_UPKEEP
from railroads.sim.goods import Good, Need, price_of
from railroads.sim.growth import Growth
from railroads.sim.ladders import TOWNS, GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.station_trade import TRAIN_LOAD
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
from railroads.sim.trains import ENGINES, Stop
from railroads.ui.tech_window import describe, explain, layout, links


def joined() -> Growth:
    constructions = Constructions(flat_map())
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
    metals = research.bonus(Effect.DISCOUNT, Tree.METALS)

    assert food == pytest.approx(0.5)
    assert metals == pytest.approx(0.05)


def test_an_owned_farm_with_better_seed_brings_a_bigger_share_into_the_station() -> None:
    growth = joined()
    growth.build(Building((6, 22), BuildingKind.FARM))
    growth.owners.own((6, 22), BuildingKind.FARM)
    growth.railway.research = learned(Tree.FOOD, "seed")

    growth.trade_goods()

    assert growth.trade.owned_brought[(6, 20)][Good.WHEAT] == pytest.approx(10 * 1.25)


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


@pytest.mark.parametrize(("good", "times"), [(Good.CATTLE, 2.0), (Good.PIGS, 1.0)])
def test_cattle_pay_twice_their_price_when_a_train_delivers_them(good: Good, times: float) -> None:
    railway = joined().railway
    railway.station_demand((6, 20)).capacity = {Need.ANIMALS: 1000.0}

    railway.pay_for((6, 20), Need.ANIMALS, good, 1.0)

    assert railway.finances.month["cargo"] == pytest.approx(times * price_of(good))


def test_the_mogul_can_be_bought_only_once_its_tech_and_year_have_come() -> None:
    railway = joined().railway
    railway.year = 1864
    railway.research = learned(Tree.METALS, "boilers")
    early = railway.engines()

    railway.year = 1865

    assert "mogul" not in early
    assert "mogul" in railway.engines()
    assert railway.buy("pacific") is None


def test_a_mogul_pulls_more_cars_and_steel_couplings_one_more() -> None:
    railway = joined().railway
    railway.year = 1865
    railway.research = learned(Tree.METALS, "boilers", "bessemer", "couplings")
    railway.finances.earn(100_000.0, "test")

    train = railway.buy("mogul")

    assert train is not None
    assert railway.most_cars(train) == ENGINES["mogul"].cars + 1
    stop = Stop((6, 3))
    assert all(stop.add_car(Good.COAL, railway.most_cars(train)) for _ in range(7))
    assert not stop.add_car(Good.COAL, railway.most_cars(train))


def test_a_diesel_engine_and_bogie_wagons_each_put_a_car_one_fibonacci_size_up() -> None:
    railway, upgraded = joined().railway, joined().railway
    upgraded.research = learned(Tree.METALS, "bessemer", "couplings", "drawbars", "bogie_wagons")

    sizes = [
        each.car_capacity(Good.OIL, engine)
        for each in (railway, upgraded)
        for engine in ("planet", "mixed_diesel")
    ]

    assert sizes == [34.0, 55.0, 55.0, 89.0]


def test_grain_hoppers_put_only_food_cars_one_size_up() -> None:
    railway = joined().railway
    railway.research = learned(Tree.FOOD, "silos", "hoppers")

    sizes = railway.car_capacity(Good.WHEAT), railway.car_capacity(Good.COAL)

    assert sizes == (144.0, 89.0)


def test_bogie_coaches_carry_more_passengers_but_freight_cars_stay_the_same() -> None:
    railway = joined().railway
    railway.research = learned(Tree.PASSENGERS, "coaches")

    coach = railway.car_capacity(Good.PASSENGERS)
    wagon = railway.car_capacity(Good.OIL)

    assert coach == pytest.approx(66.0)
    assert wagon == pytest.approx(34.0)


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
    _, kind, xp = growth.station_trade.rewards[-1]
    assert kind == BuildingKind.TENEMENT
    assert gained == pytest.approx(xp) and xp > 0
    worth = growth.station_trade._load_worth(BuildingKind.TENEMENT)
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
    assert growth.station_trade.rewards == []


def test_electric_engines_find_a_way_only_along_electrified_track() -> None:
    constructions = Constructions(flat_map())
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


def test_electrified_track_costs_its_extra_upkeep_a_cell() -> None:
    plain, wired = Constructions(flat_map()), Constructions(flat_map())
    for constructions, electric in ((plain, False), (wired, True)):
        growth = Growth(constructions, GrowthRule(trades=False))
        route = plan_route(constructions, (6, 3), (6, 20), electric=electric)
        assert route is not None
        lay_route(growth, route)
    railways = [Railway(plain), Railway(wired)]

    for railway in railways:
        railway.turn_month({})

    paid = [-railway.finances.month["track upkeep"] for railway in railways]
    assert paid[1] == pytest.approx(paid[0] * (TRACK_UPKEEP + ELECTRIC_TRACK_UPKEEP) / TRACK_UPKEEP)


def test_fine_living_makes_each_bonus_good_worth_ten_points_more_satisfaction() -> None:
    growth = joined()
    growth.build(Building((6, 22), BuildingKind.HOUSE))
    growth.railway.research = learned(Tree.COMMERCE, "investors", "advertising", "fine_living")

    growth.trade_goods()

    assert growth.economy.bonus_per_good == pytest.approx(0.4)
    assert growth.trade is not None
    assert growth.trade.bonus_per_good == pytest.approx(0.4)


@pytest.mark.parametrize("tree", list(Tree))
def test_the_tech_window_draws_every_node_below_the_nodes_it_needs_each_on_its_own_spot(
    tree: Tree,
) -> None:
    spots = layout(tree)

    assert len(set(spots.values())) == len(spots) == len(TREES[tree])
    assert all(spots[need][1] < spots[key][1] for need, key in links(tree))


@pytest.mark.parametrize("tree", list(Tree))
def test_the_tech_window_links_every_node_to_each_it_needs_and_hangs_it_under_the_first(
    tree: Tree,
) -> None:
    spots = layout(tree)

    drawn = links(tree)

    assert len(drawn) == sum(len(node.requires) for node in TREES[tree])
    for node in TREES[tree]:
        hanging = [n.key for n in TREES[tree] if n.requires[:1] == (node.key,)]
        if hanging:
            middle = sum(spots[key][0] for key in hanging) / len(hanging)
            assert spots[node.key][0] == pytest.approx(middle)


def test_freight_taken_out_of_a_station_earns_experience_in_its_goods_tree() -> None:
    railway = joined().railway
    railway.station_demand((6, 20)).capacity = {Need.FOOD: 100.0}

    railway.pay_for((6, 20), Need.FOOD, Good.WHEAT, 10.0)

    earned = railway.finances.month["cargo"]
    assert earned > 0
    assert railway.research.xp[Tree.FOOD] == pytest.approx(earned)
    assert railway.research.xp[Tree.METALS] == 0.0


def test_commerce_experience_counts_only_what_shops_got_out_of_the_stations() -> None:
    growth = joined()
    assert growth.railway is not None
    cells = np.array([0, 1])
    trade = Trade(
        cells,
        {},
        {},
        {},
        {},
        goods_got={(BuildingKind.TAVERN, Need.BEER): (np.array([10.0, 10.0]), np.full(2, 60.0))},
        local={Need.BEER: np.array([10.0, 4.0])},
    )

    growth.owners.count_commerce(trade)

    assert growth.railway.research.xp[Tree.COMMERCE] == pytest.approx(6 * price_of(Good.BEER))


def test_production_nodes_reach_owned_buildings_and_discounts_every_share() -> None:
    seed, investors = NODES["seed"][1], NODES["investors"][1]

    food = explain(Tree.FOOD, seed)
    everywhere = explain(Tree.COMMERCE, investors)

    assert food == "Every food and drink building you own makes and takes in 25% more."
    assert everywhere == "Shares of every building cost 5% less."


def test_a_node_adding_cars_names_them_once_and_every_unlock_its_first_year() -> None:
    couplings, boilers = NODES["couplings"][1], NODES["boilers"][1]

    cars = describe(couplings)
    engines = describe(boilers)

    assert cars == "cars per train +1"
    assert engines == "opens heavy freight engines (from 1853)"


def test_liming_and_farm_machinery_make_owned_farms_take_limestone_and_machinery() -> None:
    research = learned(Tree.FOOD, "seed", "rotation", "liming", "fertiliser", "tractors")
    before = research.takes(BuildingKind.FARM)
    research.earn(Tree.FOOD, xp_for(6))
    research.take("farm_machinery")

    after = research.takes(BuildingKind.FARM)

    assert before == {Need.LIMESTONE}
    assert after == {Need.LIMESTONE, Need.MACHINERY}
