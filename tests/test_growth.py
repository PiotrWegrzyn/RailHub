from dataclasses import replace

import numpy as np
import pytest

from builders import flat_map, layer
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import Trade
from railroads.sim.grids import Cell, HexGrid
from railroads.sim.growth import Growth, _allowed
from railroads.sim.ladders import (
    FACTORIES,
    FARMS,
    FOOD_PLANTS,
    LADDERS,
    LUMBER_MILLS,
    MINES,
    OPEN_FIELD,
    PORTS,
    SHIPYARDS,
    TOWN_LAND,
    TOWNS,
    GrowthRule,
    Land,
    Merge,
    Points,
)
from railroads.sim.room import SPILL_STEPS
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

NO_TRADE = GrowthRule(trades=False)
PLAIN_FARMS = replace(FARMS, crowded_at=None, land=replace(FARMS.land, open_field=1.0))


def make_map(size: int = 12) -> TileMap:
    grid = HexGrid(rows=size, cols=size, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    relief[:, -1] = Relief.MOUNTAINS
    relief[0, 6] = Relief.HILLS
    vegetation = np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
    vegetation[1, 6] = Vegetation.FOREST
    vegetation[2, 6] = Vegetation.BARREN
    temperature = np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8)
    temperature[3, 6] = Temperature.COLD
    return flat_map(
        grid=grid, terrain=terrain, vegetation=vegetation, temperature=temperature, relief=relief
    )


def build(constructions: Constructions, cell: tuple[int, int], kind: BuildingKind, n: int) -> None:
    for _ in range(n):
        constructions.build(Building(cell, kind))


def only(*ladders: object) -> GrowthRule:
    return GrowthRule(ladders=tuple(ladders), trades=False)  # type: ignore[arg-type]


def test_an_empty_map_has_the_base_chance_on_flat_grass_away_from_water() -> None:
    growth = Growth(Constructions(make_map()), NO_TRADE)

    chances = growth.chances(TOWNS)

    assert chances[5, 5] == pytest.approx(TOWNS.base_chance)
    assert chances[5, 0] == 0


@pytest.mark.parametrize(
    ("cell", "weight"),
    [
        ((5, -1), TOWN_LAND.relief[Relief.MOUNTAINS]),
        ((0, 6), TOWN_LAND.relief[Relief.HILLS]),
        ((1, 6), TOWN_LAND.vegetation[Vegetation.FOREST]),
        ((2, 6), TOWN_LAND.vegetation[Vegetation.BARREN]),
        ((3, 6), TOWN_LAND.temperature[Temperature.COLD]),
        ((5, 1), TOWN_LAND.water[Terrain.SEA]),
    ],
)
def test_the_land_multiplies_the_chance(cell: tuple[int, int], weight: float) -> None:
    growth = Growth(Constructions(make_map()), NO_TRADE)

    chances = growth.chances(TOWNS)

    assert chances[cell] == pytest.approx(TOWNS.base_chance * weight)


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
        (LUMBER_MILLS, (1, 6), False),
        (LUMBER_MILLS, (5, 5), False),
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


def test_houses_nobody_serves_leave_every_cell_at_the_base_chance_of_a_house() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.HOUSE, 2)
    build(constructions, (8, 3), BuildingKind.FLATS, 1)
    neighbour = tile_map.grid.neighbours(5, 5)[0]

    chances = Growth(constructions, NO_TRADE).chances(TOWNS)

    for cell in ((5, 5), neighbour, (5, 8), (8, 3)):
        assert chances[cell] == pytest.approx(TOWNS.base_chance)


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


def test_a_sure_month_puts_a_house_on_every_land_cell() -> None:
    constructions = Constructions(make_map())
    sure = replace(TOWNS, base_chance=50.0, merges={})
    growth = Growth(constructions, only(sure))

    changed = growth.month(np.random.default_rng(1))

    houses = growth.count(BuildingKind.HOUSE)
    assert houses[:, 1:].min() == 1
    assert houses[:, 0].max() == 0
    assert len(changed) == 12 * 11


def test_small_chances_are_drawn_as_often_as_they_should_be() -> None:
    constructions = Constructions(make_map(40))
    rare = replace(TOWNS, base_chance=0.01, chance_per_doubling={}, merges={}, land=Land())
    growth = Growth(constructions, only(rare))
    rng = np.random.default_rng(4)

    turns = sum(len(growth._hits(growth._places[0], rng)) for _ in range(200))

    expected = 0.01 * 40 * 39 * 200
    assert turns == pytest.approx(expected, rel=0.05)


def test_a_farm_fills_its_cell_and_the_next_one_spills_next_door() -> None:
    tile_map = make_map()
    growth = Growth(Constructions(tile_map), only(PLAIN_FARMS))

    first, second = [growth._turn((5, 5), PLAIN_FARMS, np.random.default_rng(0)) for _ in range(2)]

    assert first == {(5, 5)}
    assert second <= set(tile_map.grid.neighbours(5, 5))
    assert growth.count(BuildingKind.FARM).sum() == 2


@pytest.mark.parametrize(("houses", "pulled_down"), [(7, False), (8, True)])
def test_eight_people_nearby_crowd_out_a_farm(houses: int, pulled_down: bool) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FARM, 1)
    build(constructions, (9, 9), BuildingKind.FARM, 1)
    first, second = tile_map.grid.neighbours(5, 5)[:2]
    build(constructions, first, BuildingKind.HOUSE, 4)
    build(constructions, second, BuildingKind.HOUSE, houses - 4)
    growth = Growth(constructions, NO_TRADE)

    changed = growth._crowd_out_farms()

    assert changed == ({(5, 5)} if pulled_down else set())
    assert growth.count(BuildingKind.FARM)[5, 5] == (0 if pulled_down else 1)
    assert growth.count(BuildingKind.FARM)[9, 9] == 1


@pytest.mark.parametrize(("houses", "grows"), [(7, True), (8, False)])
def test_farms_never_grow_where_eight_people_live_nearby(houses: int, grows: bool) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    first, second = tile_map.grid.neighbours(5, 5)[:2]
    build(constructions, first, BuildingKind.HOUSE, 4)
    build(constructions, second, BuildingKind.HOUSE, houses - 4)
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


def test_fifteen_houses_nearby_are_not_enough_to_merge() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbours = tile_map.grid.neighbours(5, 5)
    build(constructions, (5, 5), BuildingKind.HOUSE, 2)
    for near in neighbours[:2]:
        build(constructions, near, BuildingKind.HOUSE, 6)
    build(constructions, neighbours[2], BuildingKind.HOUSE, 1)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TOWNS, np.random.default_rng(0))

    assert changed == {(5, 5)}
    assert growth.count(BuildingKind.HOUSE).sum() == 16
    assert growth.count(BuildingKind.TENEMENT).sum() == 0


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


@pytest.mark.parametrize(
    ("factory", "industry", "drawn"),
    [(BuildingKind.BIG_FACTORY, 3, False), (BuildingKind.MEGA_FACTORY, 9, True)],
)
def test_factories_are_drawn_by_industry_only_from_five_points_nearby(
    factory: BuildingKind, industry: int, drawn: bool
) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FLATS, 1)
    build(constructions, tile_map.grid.neighbours(5, 5)[0], factory, 1)
    per = FACTORIES.chance_per_doubling

    chances = Growth(constructions, NO_TRADE).chances(FACTORIES)

    pull = np.log2(1 + industry) * per[Points.INDUSTRY] if drawn else 0.0
    nearby = np.log2(19) * per[Points.POPULATION] + pull
    assert chances[5, 5] == pytest.approx(FACTORIES.base_chance + nearby)
    assert chances[5, 8] == pytest.approx(FACTORIES.base_chance)


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

    assert changed == {(5, 5)}
    assert growth.count(BuildingKind.SKYSCRAPER).sum() == 0


def test_a_long_run_keeps_its_counts_and_chances_in_step_with_what_is_built() -> None:
    constructions = Constructions(make_map(30))
    busy = tuple(replace(ladder, base_chance=ladder.base_chance + 0.002) for ladder in LADDERS)
    growth = Growth(constructions, GrowthRule(ladders=busy, trades=False))

    growth.run(600, np.random.default_rng(2))

    fresh = Growth(constructions, growth.rule)
    assert growth.count(BuildingKind.FARM).sum() > 0
    for kind in growth.rule.points:
        np.testing.assert_array_equal(growth.count(kind), fresh.count(kind))
    for ladder in busy:
        np.testing.assert_allclose(growth.chances(ladder), fresh.chances(ladder), atol=1e-12)


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
        build(constructions, cell, BuildingKind.LUMBER_MILL, 1)
    busy = tuple(replace(ladder, base_chance=ladder.base_chance + 0.002) for ladder in LADDERS)
    growth = Growth(constructions, GrowthRule(ladders=busy))
    rng = np.random.default_rng(2)
    logged, counted = [], []

    for _ in range(120):
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
    rule = only(replace(TOWNS, base_chance=0.001), replace(FARMS, base_chance=0.001))

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
    growth = Growth(constructions, only(TOWNS))

    changed = growth._turn((5, 5), TOWNS, np.random.default_rng(0))

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


@pytest.mark.parametrize(("got", "multiplier"), [(0.0, 0), (0.5, 2.5), (1.0, 5), (2.0, 10)])
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
    per_doubling = TOWNS.chance_per_doubling[Points.POPULATION]
    bred = multiplier * 4 * per_doubling * np.log2(5)
    assert chances[5, 5] == pytest.approx(TOWNS.base_chance + bred)
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

    per_doubling = TOWNS.chance_per_doubling[Points.POPULATION]
    bred = 10 * per_doubling * np.log2(2 + around)
    assert growth.chances(TOWNS)[5, 5] == pytest.approx(TOWNS.base_chance + bred)


def test_a_cell_closed_and_opened_again_is_allowed_only_once() -> None:
    cells = np.array([2, 5, 9])
    land = np.array([0.0, 0, 1, 0, 0, 1, 1, 0, 0, 1])

    allowed = _allowed(cells, np.array([5, 6]), land)

    assert allowed.tolist() == [2, 5, 6, 9]
