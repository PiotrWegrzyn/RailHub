import numpy as np

from builders import flat_map, layer
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.forests import ForestRule, change_forests
from railroads.sim.goods import MAKES, Good
from railroads.sim.growth import UNDYING, Growth
from railroads.sim.ladders import WOODCUTTERS, GrowthRule
from railroads.sim.tilemap import Vegetation

FELLING = ForestRule(clearing_per_point=0.0, clearing_per_mill=1.0, spread_per_forest=0.0)


def woodland(*placed: tuple[tuple[int, int], BuildingKind]) -> Growth:
    tile_map = flat_map(12, 12, vegetation=layer(12, 12, Vegetation.FOREST))
    constructions = Constructions(tile_map)
    for cell, kind in placed:
        constructions.build(Building(cell, kind))
    return Growth(constructions, GrowthRule(trades=False, ladders=(WOODCUTTERS,), forests=None))


def forest_cells(growth: Growth) -> int:
    return int((growth.constructions.tile_map.vegetation == Vegetation.FOREST).sum())


def test_a_woodcutter_makes_five_lumber_and_never_starves_to_death() -> None:
    assert MAKES[BuildingKind.WOODCUTTER] == {Good.LUMBER: 5}
    assert BuildingKind.WOODCUTTER in UNDYING


def test_woodcutters_fell_no_forest() -> None:
    growth = woodland(((5, 5), BuildingKind.WOODCUTTER), ((5, 6), BuildingKind.WOODCUTTER))
    before = forest_cells(growth)

    change_forests(growth, FELLING, 1, np.random.default_rng(0))

    assert forest_cells(growth) == before


def test_a_woodcutter_goes_when_a_lumber_mill_fells_its_forest() -> None:
    growth = woodland(((5, 5), BuildingKind.LUMBER_MILL), ((5, 6), BuildingKind.WOODCUTTER))

    change_forests(growth, FELLING, 1, np.random.default_rng(0))

    assert growth.counts[BuildingKind.WOODCUTTER].sum() == 0


def test_three_woodcutters_nearby_join_into_a_lumber_mill() -> None:
    growth = woodland(
        ((5, 5), BuildingKind.WOODCUTTER),
        ((5, 6), BuildingKind.WOODCUTTER),
        ((4, 5), BuildingKind.WOODCUTTER),
    )

    growth._turn((5, 5), WOODCUTTERS, np.random.default_rng(0))

    assert growth.counts[BuildingKind.WOODCUTTER].sum() == 0
    assert growth.counts[BuildingKind.LUMBER_MILL][5, 5] == 1
