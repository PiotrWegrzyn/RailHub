from dataclasses import replace

import numpy as np
import pytest
from growing import NO_TRADE, build, make_map, only

from builders import flat_map, layer
from railroads.sim.breeding import BREEDING_TIMES, CATEGORY_SHARE, OWN_KIND_TIMES, breeding_of
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.dying import UNDYING
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth, _allowed
from railroads.sim.ladder_types import Merge
from railroads.sim.ladders import (
    BAKERIES,
    CHANCE_UNIT,
    FACTORIES,
    FARMS,
    FOOD_PLANTS,
    FOOD_TOWN,
    LADDERS,
    LOGGING_CAMPS,
    MINES,
    OPEN_FIELD,
    POINTS,
    PORTS,
    SHIPYARDS,
    STEEL_MILLS,
    TOWN_LAND,
    TOWNS,
    GrowthRule,
)
from railroads.sim.points import Points
from railroads.sim.room import SPILL_STEPS
from railroads.sim.surroundings import Land
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.sim.trade import Trade

PLAIN_FARMS = replace(FARMS, crowded_at=None, land=replace(FARMS.land, open_field=1.0))


def test_an_empty_map_has_the_base_chance_on_flat_grass_away_from_water() -> None:
    growth = Growth(Constructions(make_map()), NO_TRADE)

    chances = growth.chances(TOWNS)

    assert chances[5, 5] == pytest.approx(CHANCE_UNIT * TOWNS.base)
    assert chances[5, 0] == 0


@pytest.mark.parametrize(
    ("cell", "weight"),
    [
        ((5, -1), TOWN_LAND.relief[Relief.MOUNTAINS]),
        ((0, 6), TOWN_LAND.relief[Relief.HILLS]),
        ((1, 6), TOWN_LAND.vegetation[Vegetation.FOREST]),
        ((2, 6), TOWN_LAND.vegetation[Vegetation.BARREN]),
        ((3, 6), TOWN_LAND.temperature[Temperature.COLD]),
        ((5, 1), 1.0),
    ],
)
def test_the_land_multiplies_the_chance(cell: tuple[int, int], weight: float) -> None:
    growth = Growth(Constructions(make_map()), NO_TRADE)

    chances = growth.chances(TOWNS)

    assert chances[cell] == pytest.approx(CHANCE_UNIT * TOWNS.base * weight)


def test_land_weights_of_one_cell_multiply_and_the_best_water_counts() -> None:
    tile_map = make_map()
    land = Land(relief={Relief.FLAT: 0.8}, water={Terrain.SEA: 1.7, Terrain.LAND: 0.5})

    weights = land.weights(tile_map)

    assert weights[5, 1] == pytest.approx(0.8 * 1.7)
    assert weights[5, 5] == pytest.approx(0.8 * 0.5)
    assert weights[5, 0] == 0


@pytest.mark.parametrize(
    ("ladder", "cell", "grows"),
    [
        (FARMS, (5, 5), True),
        (FARMS, (2, 6), True),
        (FARMS, (5, -1), False),
        (LOGGING_CAMPS, (1, 6), False),
        (LOGGING_CAMPS, (5, 5), False),
        (MINES, (5, -1), True),
        (MINES, (5, 5), True),
        (PORTS, (5, 1), True),
        (PORTS, (5, 5), False),
        (SHIPYARDS, (5, 5), False),
    ],
)
def test_each_ladder_grows_only_on_land_that_suits_it(
    ladder: object, cell: tuple[int, int], grows: bool
) -> None:
    land = ladder.land  # type: ignore[attr-defined]

    weight = land.weights(make_map())[cell]

    assert (weight > 0) == grows


def test_mines_are_likelier_in_mountains_than_on_the_flat() -> None:
    weights = MINES.land.weights(make_map())

    assert weights[5, -1] > 50 * weights[5, 5]


def test_houses_nobody_serves_leave_the_base_chance_raised_by_the_people_around() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.HOUSE, 2)
    build(constructions, (8, 3), BuildingKind.FLATS, 1)
    neighbour = tile_map.grid.neighbours(5, 5)[0]

    chances = Growth(constructions, NO_TRADE).chances(TOWNS)

    for cell, people in (((5, 5), 2), (neighbour, 2), ((5, 8), 0), ((8, 3), 18)):
        drawn = CHANCE_UNIT * TOWNS.base * (1 + np.log2(1 + 6 * people))
        assert chances[cell] == pytest.approx(drawn)


def test_chances_follow_buildings_as_they_come_and_go() -> None:
    constructions = Constructions(make_map())
    growth = Growth(constructions, NO_TRADE)
    build(constructions, (5, 5), BuildingKind.HOUSE, 1)
    growth.add((5, 5), BuildingKind.HOUSE, 1)

    after = growth.chances(TOWNS)

    fresh = Growth(constructions, NO_TRADE).chances(TOWNS)
    np.testing.assert_allclose(after, fresh)


def test_a_ladder_has_no_chance_until_the_neighbourhood_holds_what_it_needs() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    without_food = Growth(constructions, NO_TRADE).chances(FOOD_PLANTS)
    build(constructions, (5, 4), BuildingKind.FARM, 1)

    with_food = Growth(constructions, NO_TRADE).chances(FOOD_PLANTS)

    assert without_food[5, 5] == 0
    assert with_food[5, 5] > 0


def test_a_sure_month_puts_up_a_house_for_every_land_cell_and_none_on_the_sea() -> None:
    constructions = Constructions(make_map())
    sure = replace(TOWNS, base=50.0 / CHANCE_UNIT, merges={})
    growth = Growth(constructions, only(sure))

    growth.month(np.random.default_rng(1))

    houses = growth.count(BuildingKind.HOUSE)
    assert houses.sum() == 12 * 11
    assert houses[:, 0].max() == 0


def test_small_chances_are_drawn_as_often_as_they_should_be() -> None:
    constructions = Constructions(make_map(40))
    rare = replace(TOWNS, base=0.01 / CHANCE_UNIT, per_doubling=0.0, merges={}, land=Land())
    growth = Growth(constructions, only(rare))
    rng = np.random.default_rng(4)

    turns = sum(len(growth._hits(growth._places[0], rng)) for _ in range(200))

    expected = 0.01 * 40 * 39 * 200
    assert turns == pytest.approx(expected, rel=0.05)


def test_farms_coming_up_in_a_cell_go_up_in_it_or_next_door() -> None:
    tile_map = make_map()
    growth = Growth(Constructions(tile_map), only(PLAIN_FARMS))
    nearby = {(5, 5), *tile_map.grid.neighbours(5, 5)}

    first, second = [growth._turn((5, 5), PLAIN_FARMS, np.random.default_rng(0)) for _ in range(2)]

    assert first | second <= nearby
    assert growth.count(BuildingKind.FARM).sum() == 2


def test_houses_coming_up_in_one_cell_spread_over_it_and_its_neighbours() -> None:
    tile_map = make_map()
    plain = replace(TOWNS, merges={})
    growth = Growth(Constructions(tile_map), only(plain))
    rng = np.random.default_rng(0)
    nearby = {(5, 5), *tile_map.grid.neighbours(5, 5)}

    changed = set().union(*[growth._turn((5, 5), plain, rng) for _ in range(12)])

    assert changed <= nearby
    assert len(changed) > 3


@pytest.mark.parametrize(("houses", "pulled_down"), [(4, False), (5, True)])
def test_five_people_nearby_crowd_out_a_farm(houses: int, pulled_down: bool) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FARM, 1)
    build(constructions, (9, 9), BuildingKind.FARM, 1)
    first, second = tile_map.grid.neighbours(5, 5)[:2]
    build(constructions, first, BuildingKind.HOUSE, 3)
    build(constructions, second, BuildingKind.HOUSE, houses - 3)
    growth = Growth(constructions, NO_TRADE)

    changed = growth._crowd_out()

    assert changed == ({(5, 5)} if pulled_down else set())
    assert growth.count(BuildingKind.FARM)[5, 5] == (0 if pulled_down else 1)
    assert growth.count(BuildingKind.FARM)[9, 9] == 1


def test_a_town_crowds_out_potato_farms_as_well() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.POTATO_FARM, 1)
    first, second = tile_map.grid.neighbours(5, 5)[:2]
    build(constructions, first, BuildingKind.HOUSE, 4)
    build(constructions, second, BuildingKind.HOUSE, 4)
    growth = Growth(constructions, NO_TRADE)

    changed = growth._crowd_out()

    assert changed == {(5, 5)}
    assert growth.count(BuildingKind.POTATO_FARM)[5, 5] == 0


@pytest.mark.parametrize(("houses", "grows"), [(4, True), (5, False)])
def test_farms_never_grow_where_five_people_live_nearby(houses: int, grows: bool) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    first, second = tile_map.grid.neighbours(5, 5)[:2]
    build(constructions, first, BuildingKind.HOUSE, 3)
    build(constructions, second, BuildingKind.HOUSE, houses - 3)
    growth = Growth(constructions, NO_TRADE)

    chance = growth.chances(FARMS)[5, 5]

    assert bool(chance > 0) is grows


def test_farms_are_open_field_times_likelier_on_flat_grassland() -> None:
    tile_map = make_map()

    weights = FARMS.land.weights(tile_map)

    plain = PLAIN_FARMS.land.weights(tile_map)
    assert weights[5, 5] == pytest.approx(OPEN_FIELD * plain[5, 5])
    assert weights[0, 6] == pytest.approx(plain[0, 6])


def test_only_one_farm_turn_in_three_goes_ahead_where_something_stands_already() -> None:
    constructions = Constructions(make_map())
    build(constructions, (5, 5), BuildingKind.ORCHARD, 1)
    growth = Growth(constructions, NO_TRADE)
    rng = np.random.default_rng(0)

    fell_through = [growth._not_open(FARMS, (5, 5), rng) for _ in range(3000)]

    assert np.mean(fell_through) == pytest.approx(1 - 1 / OPEN_FIELD, abs=0.03)
    assert not growth._not_open(FARMS, (8, 8), rng)


def test_sixteen_houses_nearby_let_six_make_way_for_a_tenement_the_cells_own_first() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbours = tile_map.grid.neighbours(5, 5)
    build(constructions, (5, 5), BuildingKind.HOUSE, 1)
    for near in neighbours[:2]:
        build(constructions, near, BuildingKind.HOUSE, 6)
    build(constructions, neighbours[2], BuildingKind.HOUSE, 3)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    houses = growth.count(BuildingKind.HOUSE)
    assert changed == {(5, 5), neighbours[0]}
    assert [int(houses[near]) for near in [(5, 5), *neighbours[:3]]] == [0, 1, 6, 3]
    assert growth.count(BuildingKind.TENEMENT)[5, 5] == 1
    assert growth.points().sum() == 10 + 6
    assert growth.nearby()[5, 5] == 16


def test_five_houses_nearby_are_not_enough_to_merge() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbours = tile_map.grid.neighbours(5, 5)
    build(constructions, (5, 5), BuildingKind.HOUSE, 2)
    build(constructions, neighbours[0], BuildingKind.HOUSE, 2)
    build(constructions, neighbours[1], BuildingKind.HOUSE, 1)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    assert changed <= {(5, 5), *neighbours}
    assert growth.count(BuildingKind.HOUSE).sum() == 6
    assert growth.count(BuildingKind.TENEMENT).sum() == 0


def test_six_houses_in_a_cell_and_its_neighbours_merge_into_a_tenement() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbours = tile_map.grid.neighbours(5, 5)
    build(constructions, (5, 5), BuildingKind.HOUSE, 2)
    for near in neighbours[:4]:
        build(constructions, near, BuildingKind.HOUSE, 1)
    growth = Growth(constructions)

    growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    assert growth.count(BuildingKind.HOUSE).sum() == 0
    assert growth.count(BuildingKind.TENEMENT)[5, 5] == 1


def test_five_tenements_nearby_make_flats_before_houses_merge() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    first, *others = tile_map.grid.neighbours(5, 5)[:5]
    second = others[0]
    build(constructions, (5, 5), BuildingKind.TENEMENT, 3)
    build(constructions, first, BuildingKind.TENEMENT, 2)
    for near in others:
        build(constructions, near, BuildingKind.HOUSE, 6)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    assert changed == {(5, 5)}
    assert growth.count(BuildingKind.FLATS)[5, 5] == 1
    assert growth.count(BuildingKind.TENEMENT)[first] == 2
    assert growth.count(BuildingKind.HOUSE)[second] == 6
    assert growth.points().sum() == 18 + 2 * 6 + 24


def test_a_merge_that_would_not_fit_leaves_the_houses_standing_and_a_house_spills() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FACTORY, 2)
    neighbours = tile_map.grid.neighbours(5, 5)
    for near in neighbours[:4]:
        build(constructions, near, BuildingKind.HOUSE, 6)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    assert len(changed) == 1
    assert changed <= set(neighbours[4:])
    assert growth.count(BuildingKind.HOUSE).sum() == 25
    assert growth.count(BuildingKind.TENEMENT).sum() == 0


def industry_trade(tile_map: TileMap, cell: tuple[int, int]) -> Trade:
    cells = np.array([np.ravel_multi_index(cell, tile_map.grid.shape)])
    factory = BuildingKind.FACTORY
    nothing = {factory: np.zeros(1)}
    one = {factory: np.ones(1)}
    return Trade(cells, one, one, nothing, nothing)


@pytest.mark.parametrize(("people", "industry"), [(0, 0), (6, 0), (0, 1)])
def test_a_served_factory_breeds_by_the_factories_around_it_and_a_tenth_as_much_by_industry(
    people: int, industry: int
) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FACTORY, 1)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    if people:
        build(constructions, neighbour, BuildingKind.HOUSE, people)
    if industry:
        build(constructions, neighbour, BuildingKind.STEEL_MILL, industry)
    growth = Growth(constructions, NO_TRADE)
    breeding = breeding_of(FACTORIES, growth.rule.points)
    points, value = growth.rule.points[BuildingKind.FACTORY]

    growth._count_served(industry_trade(tile_map, (5, 5)))

    steel = industry * growth.rule.points[BuildingKind.STEEL_MILL][1]
    drawing = people + 1.5 * (OWN_KIND_TIMES * value + CATEGORY_SHARE * (value + steel))
    served = (OWN_KIND_TIMES + CATEGORY_SHARE) * value
    raised = (breeding.spawn + breeding.per_doubling * served) * np.log2(1 + drawing)
    assert points == Points.INDUSTRY
    assert growth.chances(FACTORIES)[5, 5] == pytest.approx(
        (breeding.spawn + raised) * FACTORIES.land.weights(tile_map)[5, 5]
    )


def test_three_factories_merge_into_a_big_factory_on_a_factory_turn() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    build(constructions, (5, 5), BuildingKind.FACTORY, 1)
    build(constructions, neighbour, BuildingKind.FACTORY, 2)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), FACTORIES, np.random.default_rng(0))

    assert changed == {(5, 5), neighbour}
    assert growth.count(BuildingKind.BIG_FACTORY)[5, 5] == 1
    assert growth.count(BuildingKind.FACTORY).sum() == 0
    assert growth.points(Points.INDUSTRY).sum() == 3


SURE_SKYSCRAPERS = GrowthRule(skyscraper_people=60, skyscraper_chance=1.0)


def flats_around(constructions: Constructions, blocks: tuple[int, int]) -> list[tuple[int, int]]:
    neighbours = constructions.tile_map.grid.neighbours(5, 5)
    for near, n in zip(neighbours, blocks, strict=False):
        build(constructions, near, BuildingKind.FLATS, n)
    return neighbours


def test_a_big_enough_neighbourhood_trades_three_blocks_of_flats_for_a_skyscraper() -> None:
    constructions = Constructions(make_map())
    neighbours = flats_around(constructions, (2, 2))
    growth = Growth(constructions, SURE_SKYSCRAPERS)

    changed = growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    assert {(5, 5), *neighbours[:2]} <= changed
    assert growth.count(BuildingKind.SKYSCRAPER)[5, 5] == 1
    assert growth.count(BuildingKind.FLATS).sum() == 1
    assert growth.points(Points.CORPORATE).sum() == 10


def test_the_people_of_the_pulled_down_flats_move_into_ten_houses_nearby() -> None:
    constructions = Constructions(make_map())
    flats_around(constructions, (2, 2))
    growth = Growth(constructions, SURE_SKYSCRAPERS)

    changed = growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    houses = growth.count(BuildingKind.HOUSE)
    housed = {(int(row), int(col)) for row, col in np.argwhere(houses)}
    assert houses.sum() == 10
    assert housed <= changed
    assert housed <= set(constructions.tile_map.grid.within((5, 5), 2))


def test_a_skyscraper_needs_three_blocks_of_flats_however_many_people_live_near() -> None:
    constructions = Constructions(make_map())
    neighbours = flats_around(constructions, (2,))
    build(constructions, neighbours[1], BuildingKind.TENEMENT, 6)
    growth = Growth(constructions, SURE_SKYSCRAPERS)

    growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    assert growth.count(BuildingKind.SKYSCRAPER).sum() == 0


def test_too_small_a_neighbourhood_gets_no_skyscraper() -> None:
    constructions = Constructions(make_map())
    flats_around(constructions, (2, 2))
    growth = Growth(constructions)

    growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    assert growth.count(BuildingKind.SKYSCRAPER).sum() == 0
    assert growth.count(BuildingKind.FLATS).sum() == 4


def test_farms_never_merge_or_raise_skyscrapers() -> None:
    constructions = Constructions(make_map())
    flats_around(constructions, (2, 2))
    growth = Growth(constructions, GrowthRule(skyscraper_people=1, skyscraper_chance=1.0))

    changed = growth._turn((5, 5), PLAIN_FARMS, np.random.default_rng(0))

    assert changed <= {(5, 5), *constructions.tile_map.grid.neighbours(5, 5)}
    assert growth.count(BuildingKind.FARM).sum() == 1
    assert growth.count(BuildingKind.SKYSCRAPER).sum() == 0


def test_a_long_run_keeps_its_counts_and_chances_in_step_with_what_is_built() -> None:
    constructions = Constructions(make_map(16))
    busy = tuple(replace(ladder, base=ladder.base + 0.002 / CHANCE_UNIT) for ladder in LADDERS)
    growth = Growth(constructions, GrowthRule(ladders=busy, trades=False))

    growth.run(240, np.random.default_rng(2))
    fresh = Growth(constructions, growth.rule)
    miscounted = [
        kind
        for kind in growth.rule.points
        if not np.array_equal(growth.count(kind), fresh.count(kind))
    ]
    off_chances = [
        ladder
        for ladder in busy
        if not np.allclose(growth.chances(ladder), fresh.chances(ladder), rtol=1e-7, atol=1e-12)
    ]

    assert growth.count(BuildingKind.FARM).sum() > 0
    assert miscounted == []
    assert off_chances == []


def changes_between(
    before: dict[BuildingKind, np.ndarray], growth: Growth
) -> dict[tuple[Cell, BuildingKind], int]:
    return {
        ((row, col), kind): int(count[row, col] - before[kind][row, col])
        for kind, count in growth.counts.items()
        for row, col in np.argwhere(count != before[kind]).tolist()
    }


def test_the_log_holds_every_change_months_make_to_the_counts() -> None:
    vegetation = layer(20, 20, Vegetation.GRASS)
    vegetation[:, 15:] = Vegetation.FOREST
    constructions = Constructions(flat_map(20, 20, vegetation=vegetation))
    for row in range(6, 13):
        for col in range(6, 13):
            build(constructions, (row, col), BuildingKind.HOUSE, 3)
    for cell in ((5, 9), (13, 9)):
        build(constructions, cell, BuildingKind.FARM, 1)
    for cell in ((3, 16), (15, 17)):
        build(constructions, cell, BuildingKind.LOGGING_CAMP, 1)
    busy = tuple(replace(ladder, base=ladder.base + 0.002 / CHANCE_UNIT) for ladder in LADDERS)
    growth = Growth(constructions, GrowthRule(ladders=busy))
    rng = np.random.default_rng(2)
    logged, counted = [], []

    for _ in range(36):
        before = {kind: count.copy() for kind, count in growth.counts.items()}
        with growth.log_changes() as changes:
            growth.month(rng)
        logged.append({change: n for change, n in changes.items() if n})
        counted.append(changes_between(before, growth))

    assert logged == counted
    assert any(n > 0 for month in logged for n in month.values())
    assert any(n < 0 for month in logged for n in month.values())


def test_growth_with_the_same_seed_builds_the_same_map() -> None:
    first, second = Constructions(make_map(20)), Constructions(make_map(20))
    rule = only(replace(TOWNS, base=0.001 / CHANCE_UNIT), replace(FARMS, base=0.001 / CHANCE_UNIT))

    Growth(first, rule).run(240, np.random.default_rng(3))
    Growth(second, rule).run(240, np.random.default_rng(3))

    assert list(first) == list(second)
    assert len(list(first)) > 0


def test_merges_need_more_nearby_than_they_pull_down() -> None:
    merges = [merge for ladder in LADDERS for merge in ladder.merges.values()]

    assert all(isinstance(merge, Merge) and merge.needed >= merge.pulled_down for merge in merges)


def test_a_house_with_no_room_in_its_cell_spills_into_a_cell_next_to_it() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.HOUSE, 6)
    unmerged = replace(TOWNS, merges={})
    growth = Growth(constructions, only(unmerged))

    changed = growth._turn((5, 5), unmerged, np.random.default_rng(0))

    assert len(changed) == 1
    assert changed <= set(tile_map.grid.neighbours(5, 5))
    assert growth.count(BuildingKind.HOUSE)[5, 5] == 6
    assert growth.count(BuildingKind.HOUSE).sum() == 7


def test_with_no_room_within_three_cells_nothing_goes_up() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    for near in tile_map.grid.within((8, 4), SPILL_STEPS):
        build(constructions, near, BuildingKind.FARM, 1)
    growth = Growth(constructions, only(TOWNS))

    changed = growth._turn((8, 4), TOWNS, np.random.default_rng(0))

    assert changed == set()
    assert growth.count(BuildingKind.HOUSE).sum() == 0


def test_a_cell_at_its_ladders_cap_takes_no_more_and_sends_none_elsewhere() -> None:
    constructions = Constructions(make_map())
    build(constructions, (5, 5), BuildingKind.MINE, 2)
    growth = Growth(constructions, only(MINES))

    changed = growth._turn((5, 5), MINES, np.random.default_rng(0))

    assert changed == set()
    assert growth.count(BuildingKind.MINE).sum() == 2


def houses_trade(tile_map: TileMap, cell: tuple[int, int], got: float, needed: float) -> Trade:
    cells = np.array([np.ravel_multi_index(cell, tile_map.grid.shape)])
    house = BuildingKind.HOUSE
    nothing = {house: np.zeros(1)}
    return Trade(cells, {house: np.array([got])}, {house: np.array([needed])}, nothing, nothing)


@pytest.mark.parametrize(("got", "multiplier"), [(0.0, 0), (0.5, 0.25), (1.0, 0.5), (2.0, 1.0)])
def test_served_buildings_add_their_points_by_the_share_they_are_served_to_their_own_cell(
    got: float, multiplier: float
) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.HOUSE, 4)
    growth = Growth(constructions, NO_TRADE)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    neighbour_before = growth.chances(TOWNS)[neighbour]

    growth._count_served(houses_trade(tile_map, (5, 5), got, 2.0))

    chances = growth.chances(TOWNS)
    base = CHANCE_UNIT * TOWNS.base
    raised = (base + CHANCE_UNIT * TOWNS.per_doubling * multiplier * 4) * np.log2(1 + 6 * 4)
    assert chances[5, 5] == pytest.approx(base + raised)
    assert chances[neighbour] == pytest.approx(neighbour_before)


def test_the_served_bonus_goes_once_a_cell_is_no_longer_served() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.HOUSE, 4)
    growth = Growth(constructions, NO_TRADE)
    unserved = growth.chances(TOWNS)[5, 5]
    growth._count_served(houses_trade(tile_map, (5, 5), 2.0, 2.0))

    growth._count_served(houses_trade(tile_map, (5, 5), 0.0, 2.0))

    assert growth.chances(TOWNS)[5, 5] == pytest.approx(unserved)


@pytest.mark.parametrize("around", [0, 3])
def test_a_served_house_breeds_faster_the_more_people_live_around_it(around: int) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.HOUSE, 1)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    if around:
        build(constructions, neighbour, BuildingKind.HOUSE, around)
    growth = Growth(constructions, NO_TRADE)

    growth._count_served(houses_trade(tile_map, (5, 5), 2.0, 2.0))

    base = CHANCE_UNIT * TOWNS.base
    raised = (base + CHANCE_UNIT * TOWNS.per_doubling) * np.log2(1 + 6 * (1 + around))
    assert growth.chances(TOWNS)[5, 5] == pytest.approx(base + raised)


def test_a_cell_closed_and_opened_again_is_allowed_only_once() -> None:
    cells = np.array([2, 5, 9])
    land = np.array([0.0, 0, 1, 0, 0, 1, 1, 0, 0, 1])

    allowed = _allowed(cells, np.array([5, 6]), land)

    assert allowed.tolist() == [2, 5, 6, 9]


def test_cells_opened_to_a_ladder_that_had_none_are_all_allowed() -> None:
    land = np.array([0.0, 1, 0, 1])

    allowed = _allowed(np.array([], dtype=np.intp), np.array([1, 3]), land)

    assert allowed.tolist() == [1, 3]


def test_over_many_months_a_cell_takes_its_chance_times_the_months_of_turns() -> None:
    growth = Growth(Constructions(make_map()), NO_TRADE)
    place = growth._places[0]
    place.chances[:] = 0.0
    place.chances[place.cells] = 0.01
    place.highest = 0.01

    hits = growth._hits(place, np.random.default_rng(0), months=300)

    turns = np.bincount(hits, minlength=place.chances.size)[place.cells]
    assert turns.mean() == pytest.approx(3.0, rel=0.1)
    assert turns.max() > 1


def test_growing_years_at_a_step_still_closes_every_year() -> None:
    growth = Growth(Constructions(make_map()), NO_TRADE, start_year=850)

    growth.run(10 * 12, np.random.default_rng(0), step=60)

    assert sorted(growth.history.population) == list(range(850, 860))


@pytest.mark.parametrize("houses", [0, 1, 3, 6])
def test_a_bakery_is_likelier_by_every_doubling_of_the_people_around(houses: int) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    if houses:
        build(constructions, (5, 5), BuildingKind.HOUSE, houses)
    growth = Growth(constructions, NO_TRADE)
    land = BAKERIES.land.weights(tile_map)[5, 5]

    chance = growth.chances(BAKERIES)[5, 5]

    drawn = 1 + np.log2(1 + houses)
    assert chance == pytest.approx(CHANCE_UNIT * BAKERIES.base * drawn * land)


def test_brewers_distillers_and_potters_need_a_town_around_them() -> None:
    kinds = (BuildingKind.BREWERY, BuildingKind.DISTILLERY, BuildingKind.POTTER)

    needs = [ladder.needs for ladder in LADDERS if ladder.kinds[0] in kinds]

    assert needs == [{Points.POPULATION: FOOD_TOWN}] * 3


def test_soldiers_never_starve_to_death() -> None:
    soldiers = {BuildingKind.OUTPOST, BuildingKind.BASE, BuildingKind.HEADQUARTERS}

    undying = soldiers & UNDYING

    assert undying == soldiers


@pytest.mark.parametrize("houses", [1, 4])
def test_a_steel_mill_is_drawn_by_people_but_as_many_times_less_likely_as_one_plus_them(
    houses: int,
) -> None:
    tile_map = make_map()
    alone = Growth(Constructions(tile_map), NO_TRADE).chances(STEEL_MILLS)[5, 5]
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.HOUSE, houses)

    crowded = Growth(constructions, NO_TRADE).chances(STEEL_MILLS)[5, 5]

    assert crowded == pytest.approx(alone * (1 + np.log2(1 + houses)) / (1 + houses))


def test_gasworks_are_drawn_by_industry_and_dairies_by_people_twice() -> None:
    (gasworks,) = [ladder for ladder in LADDERS if ladder.kinds[0] == BuildingKind.GASWORKS]
    (dairies,) = [ladder for ladder in LADDERS if ladder.kinds[0] == BuildingKind.DAIRY]

    gas = breeding_of(gasworks, POINTS)
    milk = breeding_of(dairies, POINTS)

    assert gas.drawn_by == {
        Points.INDUSTRY: CATEGORY_SHARE,
        BuildingKind.GASWORKS: OWN_KIND_TIMES,
    }
    assert gas.shunned_by == {Points.POPULATION: 1.0}
    assert milk.drawn_by == {
        Points.POPULATION: 2.0,
        Points.INDUSTRY: CATEGORY_SHARE,
        BuildingKind.DAIRY: OWN_KIND_TIMES,
    }


def test_bakeries_give_food_points_but_come_up_and_breed_like_works() -> None:
    (sausages,) = [ladder for ladder in LADDERS if ladder.kinds[0] == BuildingKind.SAUSAGE_MAKER]

    bread = breeding_of(BAKERIES, POINTS)
    sausage = breeding_of(sausages, POINTS)

    assert (bread.own, sausage.own) == (Points.FOOD, Points.FOOD)
    assert bread.spawn == pytest.approx(CHANCE_UNIT * BAKERIES.base)
    assert bread.per_doubling == pytest.approx(BREEDING_TIMES * bread.spawn)
    assert sausage.per_doubling == pytest.approx(BREEDING_TIMES * sausage.spawn)


def test_homes_are_pulled_to_towns_by_their_own_chance_per_doubling_not_their_base_chance() -> None:
    towns = breeding_of(TOWNS, POINTS)

    pull = towns.per_doubling

    assert pull == pytest.approx(CHANCE_UNIT * TOWNS.per_doubling)
    assert pull != pytest.approx(BREEDING_TIMES * CHANCE_UNIT * TOWNS.base)


def test_a_served_sawmill_breeds_sawmills_far_more_than_brickworks() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.SAWMILL, 1)
    growth = Growth(constructions, NO_TRADE)
    sawmills = next(ladder for ladder in LADDERS if ladder.kinds[0] == BuildingKind.SAWMILL)
    bricks = next(ladder for ladder in LADDERS if ladder.kinds[0] == BuildingKind.BRICKWORKS)
    cells = np.array([np.ravel_multi_index((5, 5), tile_map.grid.shape)])
    one = {BuildingKind.SAWMILL: np.ones(1)}
    nothing = {BuildingKind.SAWMILL: np.zeros(1)}

    growth._count_served(Trade(cells, one, one, nothing, nothing))

    sawmill_raise = growth.chances(sawmills)[5, 5] / sawmills.land.weights(tile_map)[5, 5]
    brick_raise = growth.chances(bricks)[5, 5] / bricks.land.weights(tile_map)[5, 5]
    assert sawmill_raise / sawmills.base > 50 * brick_raise / bricks.base


def test_a_big_furniture_maker_counts_as_furniture_makers_around_it() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.BIG_FURNITURE_MAKER, 1)

    growth = Growth(constructions, NO_TRADE)

    big = growth.rule.points[BuildingKind.BIG_FURNITURE_MAKER][1]
    assert growth.nearby(BuildingKind.FURNITURE_MAKER)[5, 5] == big
