from dataclasses import replace

import numpy as np
import pytest

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.growth import (
    FACTORIES,
    FARMS,
    FOOD_PLANTS,
    LADDERS,
    LUMBER_MILLS,
    MINES,
    PORTS,
    SHIPYARDS,
    TOWN_LAND,
    TOWNS,
    Growth,
    GrowthRule,
    Land,
    Merge,
    Points,
)
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

NO_TRADE = GrowthRule(trades=False)


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
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=terrain,
        vegetation=vegetation,
        temperature=temperature,
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=relief,
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
        ((5, 1), 1.0),
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
        (LUMBER_MILLS, (1, 6), True),
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


def test_buildings_raise_the_chance_across_their_neighbourhood_by_their_points() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.HOUSE, 2)
    build(constructions, (8, 3), BuildingKind.FLATS, 1)
    build(constructions, (8, 8), BuildingKind.TENEMENT, 1)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    base = TOWNS.base_chance
    per_point = TOWNS.chance_per_point[Points.POPULATION]

    chances = Growth(constructions, NO_TRADE).chances(TOWNS)

    assert chances[5, 5] == pytest.approx(base + 2 * per_point)
    assert chances[neighbour] == pytest.approx(base + 2 * per_point)
    assert chances[5, 8] == pytest.approx(base)
    assert chances[8, 3] == pytest.approx(base + 5 * per_point)
    assert chances[8, 8] == pytest.approx(base + 2 * per_point)


def test_chances_follow_buildings_as_they_come_and_go() -> None:
    constructions = Constructions(make_map())
    growth = Growth(constructions, NO_TRADE)
    build(constructions, (5, 5), BuildingKind.HOUSE, 1)
    growth._add((5, 5), BuildingKind.HOUSE, 1)

    after = growth.chances(TOWNS)

    fresh = Growth(constructions, NO_TRADE).chances(TOWNS)
    np.testing.assert_allclose(after, fresh)


def test_a_ladder_has_no_chance_until_the_neighbourhood_holds_what_it_needs() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 4), BuildingKind.FARM, 1)
    build(constructions, (5, 6), BuildingKind.HOUSE, 5)
    without_people = Growth(constructions, NO_TRADE).chances(FOOD_PLANTS)
    build(constructions, (5, 6), BuildingKind.HOUSE, 1)

    with_people = Growth(constructions, NO_TRADE).chances(FOOD_PLANTS)

    assert without_people[5, 5] == 0
    assert with_people[5, 5] > 0


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
    rare = replace(TOWNS, base_chance=0.01, chance_per_point={}, merges={}, land=Land())
    growth = Growth(constructions, only(rare))
    rng = np.random.default_rng(4)

    turns = sum(len(growth._hits(growth._places[0], rng)) for _ in range(200))

    expected = 0.01 * 40 * 39 * 200
    assert turns == pytest.approx(expected, rel=0.05)


def test_farms_can_share_a_cell() -> None:
    constructions = Constructions(make_map())
    growth = Growth(constructions, only(FARMS))

    turns = [growth._turn((5, 5), FARMS) for _ in range(3)]

    assert turns == [{(5, 5)}] * 3
    assert growth.count(BuildingKind.FARM)[5, 5] == 3


def test_a_town_crowds_out_the_farms_beside_it() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FARM, 2)
    build(constructions, (5, 7), BuildingKind.FARM, 1)
    build(constructions, tile_map.grid.neighbours(5, 5)[0], BuildingKind.FLATS, 2)
    build(constructions, tile_map.grid.neighbours(5, 5)[1], BuildingKind.HOUSE, 1)
    growth = Growth(constructions, NO_TRADE)

    changed = growth._crowd_out_farms()

    assert changed == {(5, 5)}
    assert growth.count(BuildingKind.FARM)[5, 5] == 0
    assert growth.count(BuildingKind.FARM)[5, 7] == 1


def test_six_houses_nearby_let_three_make_way_for_a_tenement_the_cells_own_first() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    build(constructions, (5, 5), BuildingKind.HOUSE, 2)
    build(constructions, neighbour, BuildingKind.HOUSE, 4)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TOWNS)

    houses = growth.count(BuildingKind.HOUSE)
    assert changed == {(5, 5), neighbour}
    assert (houses[5, 5], houses[neighbour]) == (0, 3)
    assert growth.count(BuildingKind.TENEMENT)[5, 5] == 1
    assert growth.points().sum() == 3 + 2
    assert growth.nearby()[5, 5] == 5


def test_five_houses_nearby_are_not_enough_to_merge() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.HOUSE, 2)
    build(constructions, tile_map.grid.neighbours(5, 5)[0], BuildingKind.HOUSE, 3)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TOWNS)

    assert changed == {(5, 5)}
    assert growth.count(BuildingKind.HOUSE).sum() == 6
    assert growth.count(BuildingKind.TENEMENT).sum() == 0


def test_six_tenements_nearby_make_flats_before_houses_merge() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    first, second = tile_map.grid.neighbours(5, 5)[:2]
    build(constructions, (5, 5), BuildingKind.TENEMENT, 3)
    build(constructions, first, BuildingKind.TENEMENT, 3)
    build(constructions, second, BuildingKind.HOUSE, 6)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TOWNS)

    assert changed == {(5, 5)}
    assert growth.count(BuildingKind.FLATS)[5, 5] == 1
    assert growth.count(BuildingKind.TENEMENT)[first] == 3
    assert growth.count(BuildingKind.HOUSE)[second] == 6
    assert growth.points().sum() == 5 + 3 * 2 + 6


def test_a_merge_that_would_not_fit_leaves_the_houses_standing() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FARM, 6)
    build(constructions, tile_map.grid.neighbours(5, 5)[0], BuildingKind.HOUSE, 6)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TOWNS)

    assert changed == set()
    assert growth.count(BuildingKind.HOUSE).sum() == 6
    assert growth.count(BuildingKind.TENEMENT).sum() == 0


def test_people_and_factories_nearby_raise_the_chance_of_a_factory() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FLATS, 1)
    build(constructions, tile_map.grid.neighbours(5, 5)[0], BuildingKind.BIG_FACTORY, 1)
    per = FACTORIES.chance_per_point

    chances = Growth(constructions, NO_TRADE).chances(FACTORIES)

    expected = 5 * per[Points.POPULATION] + 3 * per[Points.INDUSTRY]
    assert chances[5, 5] == pytest.approx(expected)
    assert chances[5, 8] == 0


def test_three_factories_merge_into_a_big_factory_on_a_factory_turn() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    build(constructions, (5, 5), BuildingKind.FACTORY, 1)
    build(constructions, neighbour, BuildingKind.FACTORY, 2)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), FACTORIES)

    assert changed == {(5, 5), neighbour}
    assert growth.count(BuildingKind.BIG_FACTORY)[5, 5] == 1
    assert growth.count(BuildingKind.FACTORY).sum() == 0
    assert growth.points(Points.INDUSTRY).sum() == 3


def test_a_big_enough_neighbourhood_trades_flats_and_factories_for_a_skyscraper() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbours = tile_map.grid.neighbours(5, 5)
    build(constructions, (5, 5), BuildingKind.FACTORY, 3)
    for near in neighbours[:2]:
        build(constructions, near, BuildingKind.FLATS, 2)
    rule = GrowthRule(skyscraper_people=15, skyscraper_industry=3)
    growth = Growth(constructions, rule)

    changed = growth._turn((5, 5), TOWNS)

    assert changed == {(5, 5), *neighbours[:2]}
    assert growth.count(BuildingKind.SKYSCRAPER)[5, 5] == 1
    assert growth.count(BuildingKind.FLATS).sum() == 1
    assert growth.count(BuildingKind.FACTORY).sum() == 0
    assert growth.points(Points.CORPORATE).sum() == 1
    assert growth.nearby(Points.POPULATION)[5, 5] == 5


def test_too_small_a_neighbourhood_gets_no_skyscraper() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FACTORY, 3)
    for near in tile_map.grid.neighbours(5, 5)[:2]:
        build(constructions, near, BuildingKind.FLATS, 2)
    growth = Growth(constructions)

    growth._turn((5, 5), TOWNS)

    assert growth.count(BuildingKind.SKYSCRAPER).sum() == 0
    assert growth.count(BuildingKind.FLATS).sum() == 4


def test_farms_never_merge_or_raise_skyscrapers() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    build(constructions, (5, 5), BuildingKind.FACTORY, 3)
    for near in tile_map.grid.neighbours(5, 5)[:2]:
        build(constructions, near, BuildingKind.FLATS, 2)
    growth = Growth(constructions, GrowthRule(skyscraper_people=1, skyscraper_industry=1))

    changed = growth._turn((5, 5), FARMS)

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
