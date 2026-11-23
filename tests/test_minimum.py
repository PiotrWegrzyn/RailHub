import numpy as np

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.growth import KEY_LEAST, Growth
from railroads.sim.ladders import FILLING_STATIONS, RESOURCES, TOWNS, WORKS, GrowthRule
from railroads.sim.tilemap import REFERENCE_LAND

(OIL_WELLS,) = [ladder for ladder in RESOURCES if ladder.kinds[0] == BuildingKind.OIL_WELL]
(REFINERIES,) = [ladder for ladder in WORKS if ladder.kinds[0] == BuildingKind.REFINERY]


def grown(year: int, *ladders, houses: int = 0) -> Growth:
    constructions = Constructions(flat_map(20, 20))
    constructions.build(Building((2, 2), BuildingKind.REFINERY))
    for _ in range(houses):
        constructions.build(Building((10, 10), BuildingKind.HOUSE))
    rule = GrowthRule(trades=False, ladders=ladders, forests=None)
    return Growth(constructions, rule, start_year=year)


def standing(growth: Growth, kind: BuildingKind) -> int:
    return int(growth.count(kind).sum())


def test_a_building_that_opens_in_some_year_goes_up_once_that_year_comes() -> None:
    growth = grown(1919, FILLING_STATIONS, houses=6)

    growth.run(12, np.random.default_rng(1))

    assert standing(growth, BuildingKind.FILLING_STATION) == 1


def test_a_building_does_not_go_up_before_the_year_it_opens() -> None:
    growth = grown(1900, FILLING_STATIONS, houses=6)

    growth.run(12, np.random.default_rng(1))

    assert standing(growth, BuildingKind.FILLING_STATION) == 0


def test_buildings_that_were_always_there_are_not_topped_up() -> None:
    growth = grown(1900, TOWNS)

    least = growth.least_of(TOWNS)

    assert least == 0


def test_key_buildings_keep_three_on_a_map_of_poland_and_more_on_a_bigger_one() -> None:
    growth = grown(1900, REFINERIES)

    least = growth.least_of(REFINERIES)
    growth._land_cells = REFERENCE_LAND
    poland = growth.least_of(REFINERIES)
    growth._land_cells = 2 * REFERENCE_LAND
    bigger = growth.least_of(REFINERIES)

    assert (least, poland, bigger) == (1, KEY_LEAST, 2 * KEY_LEAST)


def test_an_oil_well_goes_up_even_on_a_map_with_no_oil() -> None:
    growth = grown(1854, OIL_WELLS)

    growth.run(24, np.random.default_rng(1))

    assert standing(growth, BuildingKind.OIL_WELL) == 1
