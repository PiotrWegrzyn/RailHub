import numpy as np

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.forests import FORESTS, ForestRule, change_forests, forest_turn, forests_nearby
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import LOGGING_CAMPS, TOWNS, GrowthRule
from railroads.sim.points import Points
from railroads.sim.reach import Neighbourhoods
from railroads.sim.tilemap import Relief, TileMap, Vegetation

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
    return flat_map(grid=grid, vegetation=vegetation, relief=relief)


def satisfy(growth: Growth, cell: tuple[int, int]) -> None:
    """Counts the logging camp in a cell as having sold all its logs last month."""
    growth.served(Points.TIMBER)[cell[0] * growth.grid.cols + cell[1]] = 1.0


def near(tile_map: TileMap) -> np.ndarray:
    return Neighbourhoods(tile_map.grid).index


def forest_cells(tile_map: TileMap) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(tile_map.vegetation == Vegetation.FOREST)}


def test_logging_camps_need_four_forests_nearby_and_open_flat_land_beside() -> None:
    weights = LOGGING_CAMPS.land.weights(make_map())

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


def test_a_satisfied_logging_camp_clears_forest_far_faster_than_a_house() -> None:
    tile_map = make_map(40)
    tile_map.set_vegetation(np.arange(40 * 40), Vegetation.FOREST)
    ones = np.ones(40 * 40, dtype=np.int64)
    rng = np.random.default_rng(1)

    by_people, _ = forest_turn(tile_map, near(tile_map), ones, ones * 0, FORESTS, 100, rng)
    by_mills, _ = forest_turn(tile_map, near(tile_map), ones * 0, ones, FORESTS, 100, rng)

    assert by_mills.size > 10 * max(by_people.size, 1)


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


def test_a_logging_camp_left_with_too_little_forest_closes() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    constructions.build(Building((5, 2), BuildingKind.LOGGING_CAMP))
    rule = GrowthRule(trades=False, ladders=(), forests=ForestRule(clearing_per_mill=1.0))
    growth = Growth(constructions, rule)
    satisfy(growth, (5, 2))

    changed = growth.month(np.random.default_rng(0))

    assert growth.count(BuildingKind.LOGGING_CAMP)[5, 2] == 0
    assert (5, 2) in changed
    assert tile_map.vegetation[5, 2] == Vegetation.GRASS
    assert (5, 2) in growth.forests_changed


def test_a_logging_camp_that_sells_nothing_clears_no_forest() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    constructions.build(Building((5, 2), BuildingKind.LOGGING_CAMP))
    rule = GrowthRule(trades=False, ladders=(), forests=STILL)
    growth = Growth(constructions, rule)
    before = forest_cells(tile_map)
    clearing = ForestRule(clearing_per_point=0.0, clearing_per_mill=1.0, spread_per_forest=0.0)

    change_forests(growth, clearing, 1, np.random.default_rng(0))

    assert forest_cells(tile_map) == before
    assert growth.count(BuildingKind.LOGGING_CAMP)[5, 2] == 1


def test_a_closed_logging_camp_comes_up_again_in_the_nearest_cell_that_suits_one() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    constructions.build(Building((5, 2), BuildingKind.LOGGING_CAMP))
    growth = Growth(
        constructions, GrowthRule(trades=False, ladders=(LOGGING_CAMPS,), forests=STILL)
    )
    clearing = ForestRule(clearing_per_mill=1.0, spread_per_forest=0.0)
    satisfy(growth, (5, 2))

    changed = change_forests(growth, clearing, 1, np.random.default_rng(0))

    camps = [tuple(cell) for cell in np.argwhere(growth.count(BuildingKind.LOGGING_CAMP) > 0)]
    assert len(camps) == 1
    camp = (int(camps[0][0]), int(camps[0][1]))
    assert camp != (5, 2)
    assert camp in changed
    assert tile_map.vegetation[camp] == Vegetation.FOREST
    assert growth.grid.within((5, 2), 3).get(camp) is not None


def test_logging_camps_are_likelier_the_more_forest_is_around() -> None:
    tile_map = make_map()
    growth = Growth(Constructions(tile_map), GrowthRule(trades=False, ladders=(LOGGING_CAMPS,)))

    land = growth.land_weights(LOGGING_CAMPS).reshape(tile_map.grid.shape)

    assert land[5, 2] > 0
    assert land[2, 4] > 0
    assert land[5, 2] > land[2, 2]


def test_forest_never_spreads_onto_built_cells() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    constructions.build(Building((1, 5), BuildingKind.HOUSE))
    rule = GrowthRule(trades=False, ladders=(), forests=ForestRule(spread_per_forest=1.0))
    growth = Growth(constructions, rule)

    growth.month(np.random.default_rng(0))

    assert tile_map.vegetation[1, 5] == Vegetation.GRASS
    assert tile_map.vegetation[1, 6] == Vegetation.FOREST


def test_cleared_forest_weighs_again_for_towns() -> None:
    tile_map = make_map(hill=(5, 5))
    constructions = Constructions(tile_map)
    constructions.build(Building((5, 5), BuildingKind.HOUSE))
    growth = Growth(constructions, GrowthRule(trades=False, ladders=(TOWNS,), forests=STILL))
    chance_before = growth.chances(TOWNS)[5, 5]
    clearing = ForestRule(clearing_per_point=1.0, spread_per_forest=0.0)

    change_forests(growth, clearing, 1, np.random.default_rng(0))

    assert growth.chances(TOWNS)[5, 5] > 3 * chance_before


def test_forests_nearby_counts_the_cell_and_the_cells_next_to_it() -> None:
    tile_map = make_map()

    counts = forests_nearby(tile_map, near(tile_map)).reshape(tile_map.grid.shape)

    assert counts[5, 5] == 7
    assert counts[0, 11] == 1
    assert counts[10, 0] == 0
