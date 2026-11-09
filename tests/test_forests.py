import numpy as np

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import Neighbourhoods
from railroads.sim.forests import FORESTS, ForestRule, forest_turn, forests_nearby
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.growth import LUMBER_MILLS, TOWNS, Growth, GrowthRule
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

STILL = ForestRule(clearing_per_point=0.0, clearing_per_mill=0.0, spread_per_forest=0.0)


def make_map(size: int = 12, hill: tuple[int, int] | None = None) -> TileMap:
    grid = HexGrid(rows=size, cols=size, km_per_cell=2.0)
    vegetation = np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
    vegetation[2:9, 2:9] = Vegetation.FOREST
    vegetation[0, 11] = Vegetation.FOREST
    vegetation[11, :] = Vegetation.BARREN
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    if hill is not None:
        relief[hill] = Relief.HILLS
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=vegetation,
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=relief,
    )


def near(tile_map: TileMap) -> np.ndarray:
    return Neighbourhoods(tile_map.grid).index


def forest_cells(tile_map: TileMap) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(tile_map.vegetation == Vegetation.FOREST)}


def test_lumber_mills_need_four_forests_nearby_and_open_flat_land_beside() -> None:
    weights = LUMBER_MILLS.land.weights(make_map())

    assert weights[5, 2] > 0
    assert weights[5, 5] == 0
    assert weights[0, 11] == 0
    assert weights[5, 1] == 0


def test_people_and_industry_clear_the_forest_around_them() -> None:
    tile_map = make_map()
    people = np.zeros(tile_map.grid.rows * tile_map.grid.cols, dtype=np.int64)
    people[np.ravel_multi_index((5, 5), tile_map.grid.shape)] = 1
    rule = ForestRule(clearing_per_point=1.0, spread_per_forest=0.0)

    cleared, grown = forest_turn(
        tile_map, near(tile_map), people, np.zeros_like(people), rule, 1, np.random.default_rng(0)
    )

    assert cleared.tolist() == [np.ravel_multi_index((5, 5), tile_map.grid.shape)]
    assert grown.size == 0


def test_a_lumber_mill_clears_forest_a_hundred_times_faster_than_a_house() -> None:
    tile_map = make_map(40)
    tile_map.set_vegetation(np.arange(40 * 40), Vegetation.FOREST)
    ones = np.ones(40 * 40, dtype=np.int64)
    rng = np.random.default_rng(1)

    by_people, _ = forest_turn(tile_map, near(tile_map), ones, ones * 0, FORESTS, 100, rng)
    by_mills, _ = forest_turn(tile_map, near(tile_map), ones * 0, ones, FORESTS, 100, rng)

    assert by_people.size < 15
    assert 200 < by_mills.size < 400


def test_forest_spreads_onto_grass_beside_it_but_not_onto_barren_land() -> None:
    tile_map = make_map()
    tile_map.set_vegetation(np.arange(120, 132), Vegetation.FOREST)
    zeros = np.zeros(tile_map.grid.rows * tile_map.grid.cols, dtype=np.int64)
    rule = ForestRule(spread_per_forest=1.0)

    _, grown = forest_turn(
        tile_map, near(tile_map), zeros, zeros, rule, 1, np.random.default_rng(0)
    )

    rows = {divmod(int(index), tile_map.grid.cols)[0] for index in grown}
    assert 9 in rows
    assert 11 not in rows
    assert all(tile_map.vegetation.flat[index] == Vegetation.GRASS for index in grown)


def test_a_lumber_mill_left_with_too_little_forest_closes() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    constructions.build(Building((5, 2), BuildingKind.LUMBER_MILL))
    rule = GrowthRule(trades=False, ladders=(), forests=ForestRule(clearing_per_mill=1.0))
    growth = Growth(constructions, rule)

    changed = growth.month(np.random.default_rng(0))

    assert growth.count(BuildingKind.LUMBER_MILL)[5, 2] == 0
    assert (5, 2) in changed
    assert tile_map.vegetation[5, 2] == Vegetation.GRASS
    assert (5, 2) in growth.forests_changed


def test_forest_never_spreads_onto_built_cells() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    constructions.build(Building((1, 5), BuildingKind.HOUSE))
    rule = GrowthRule(trades=False, ladders=(), forests=ForestRule(spread_per_forest=1.0))
    growth = Growth(constructions, rule)

    growth.month(np.random.default_rng(0))

    assert tile_map.vegetation[1, 5] == Vegetation.GRASS
    assert tile_map.vegetation[1, 6] == Vegetation.FOREST


def test_cleared_forest_weighs_again_for_towns_and_for_land_food() -> None:
    tile_map = make_map(hill=(5, 5))
    constructions = Constructions(tile_map)
    constructions.build(Building((5, 5), BuildingKind.HOUSE))
    growth = Growth(constructions, GrowthRule(trades=False, ladders=(TOWNS,), forests=STILL))
    chance_before = growth.chances(TOWNS)[5, 5]
    food_before = growth.economy.free_food[np.ravel_multi_index((5, 5), tile_map.grid.shape)]
    clearing = ForestRule(clearing_per_point=1.0, spread_per_forest=0.0)

    growth._change_forests(clearing, 1, np.random.default_rng(0))

    index = np.ravel_multi_index((5, 5), tile_map.grid.shape)
    assert growth.chances(TOWNS)[5, 5] > 3 * chance_before
    assert food_before == 1.0
    assert growth.economy.free_food[index] == 0.0


def test_forests_nearby_counts_the_cell_and_the_cells_next_to_it() -> None:
    tile_map = make_map()

    counts = forests_nearby(tile_map, near(tile_map)).reshape(tile_map.grid.shape)

    assert counts[5, 5] == 7
    assert counts[0, 11] == 1
    assert counts[10, 0] == 0
