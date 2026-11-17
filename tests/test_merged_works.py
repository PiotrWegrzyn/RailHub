import numpy as np
import pytest
from test_growth import build, make_map

from railroads.sim.constructions import MERGED, BuildingKind, Constructions
from railroads.sim.goods import FOOD_MERGING, MERGE_BONUS, MERGING, NEEDS, RECIPES
from railroads.sim.growth import Growth
from railroads.sim.ladders import BAKERIES, LADDERS, POINTS, WORKS, Points

TEXTILE_MILLS = next(ladder for ladder in WORKS if ladder.kinds[0] == BuildingKind.TEXTILE_MILL)
GLASSWORKS = next(ladder for ladder in WORKS if ladder.kinds[0] == BuildingKind.GLASSWORKS)


def test_two_textile_mills_in_a_neighbourhood_merge_into_a_big_one() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    build(constructions, (5, 5), BuildingKind.TEXTILE_MILL, 1)
    build(constructions, neighbour, BuildingKind.TEXTILE_MILL, 1)
    growth = Growth(constructions)

    changed = growth._turn((5, 5), TEXTILE_MILLS, np.random.default_rng(0))

    assert changed == {(5, 5), neighbour}
    assert growth.count(BuildingKind.BIG_TEXTILE_MILL)[5, 5] == 1
    assert growth.count(BuildingKind.TEXTILE_MILL).sum() == 0


def test_two_big_textile_mills_merge_into_a_mega_one() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    build(constructions, (5, 5), BuildingKind.BIG_TEXTILE_MILL, 1)
    build(constructions, neighbour, BuildingKind.BIG_TEXTILE_MILL, 1)
    growth = Growth(constructions)

    growth._turn((5, 5), TEXTILE_MILLS, np.random.default_rng(0))

    assert growth.count(BuildingKind.MEGA_TEXTILE_MILL)[5, 5] == 1
    assert growth.count(BuildingKind.BIG_TEXTILE_MILL).sum() == 0


def test_works_merging_into_a_big_one_move_the_houses_it_needs_room_from_next_door() -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    neighbour = tile_map.grid.neighbours(5, 5)[0]
    build(constructions, (5, 5), BuildingKind.GLASSWORKS, 1)
    build(constructions, (5, 5), BuildingKind.HOUSE, 5)
    build(constructions, neighbour, BuildingKind.GLASSWORKS, 1)
    growth = Growth(constructions)

    growth._turn((5, 5), GLASSWORKS, np.random.default_rng(0))

    assert growth.count(BuildingKind.BIG_GLASSWORKS)[5, 5] == 1
    assert growth.count(BuildingKind.HOUSE)[5, 5] == 3
    assert growth.count(BuildingKind.HOUSE).sum() == 5


@pytest.mark.parametrize(("bakeries", "merged"), [(2, 0), (3, 1)])
def test_it_takes_three_bakeries_to_make_a_big_bakery(bakeries: int, merged: int) -> None:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    for cell in tile_map.grid.neighbours(5, 5)[:bakeries]:
        build(constructions, cell, BuildingKind.BAKERY, 1)
    growth = Growth(constructions)

    growth._turn((5, 5), BAKERIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.BIG_BAKERY).sum() == merged


def test_a_big_works_makes_more_than_the_works_merged_into_it_and_needs_inputs_to_match() -> None:
    mill, bakery = RECIPES[BuildingKind.TEXTILE_MILL], RECIPES[BuildingKind.BAKERY]

    big_mill, mega_bakery = (
        RECIPES[BuildingKind.BIG_TEXTILE_MILL],
        RECIPES[BuildingKind.MEGA_BAKERY],
    )

    assert big_mill.most == pytest.approx(mill.most * MERGING * MERGE_BONUS)
    assert mega_bakery.most == pytest.approx(bakery.most * (FOOD_MERGING * MERGE_BONUS) ** 2)
    assert big_mill.per_unit == mill.per_unit
    assert sum(NEEDS[BuildingKind.BIG_TEXTILE_MILL].values()) == pytest.approx(
        big_mill.most * sum(mill.per_unit.values())
    )


def test_a_mega_works_gives_the_points_of_all_the_works_merged_into_it() -> None:
    _, small = POINTS[BuildingKind.BAKERY]

    points, mega = POINTS[BuildingKind.MEGA_BAKERY]

    assert points == Points.INDUSTRY
    assert mega == small * FOOD_MERGING**2


def test_every_merging_works_grows_on_one_ladder_with_both_its_sizes() -> None:
    kinds = {ladder.kinds[0]: ladder.kinds for ladder in LADDERS}

    assert all(kinds[kind] == (kind, *sizes) for kind, sizes in MERGED.items())
