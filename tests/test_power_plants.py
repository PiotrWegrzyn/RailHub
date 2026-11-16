import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.power_plants import (
    PLACE_STEPS,
    PLANT_APART,
    build_power_plants,
    electricity_wanted,
)

QUIET = GrowthRule(ladders=(), forests=None)


def town(year: int, *makers: tuple[tuple[int, int], BuildingKind]) -> Growth:
    constructions = Constructions(flat_map(40, 40))
    for _ in range(6):
        constructions.build(Building((20, 20), BuildingKind.HOUSE))
    for cell, kind in makers:
        constructions.build(Building(cell, kind))
    return Growth(constructions, QUIET, start_year=year)


def built(growth: Growth) -> dict[BuildingKind, list[tuple[int, int]]]:
    found = {}
    for kind in (
        BuildingKind.POWER_PLANT,
        BuildingKind.DIESEL_POWER_PLANT,
        BuildingKind.GAS_POWER_PLANT,
    ):
        cells = [tuple(cell) for cell in np.argwhere(growth.count(kind) > 0).tolist()]
        if cells:
            found[kind] = cells
    return found


def test_no_power_plant_goes_up_before_the_first_one_in_1882() -> None:
    growth = town(1870)

    changed = build_power_plants(growth)

    assert changed == set()
    assert built(growth) == {}


def test_a_coal_power_plant_goes_up_by_the_town_when_electricity_is_wanted() -> None:
    growth = town(2030)

    changed = build_power_plants(growth)

    plants_built = built(growth)
    assert list(plants_built) == [BuildingKind.POWER_PLANT]
    (cell,) = plants_built[BuildingKind.POWER_PLANT]
    assert changed == {cell}
    assert cell in growth.grid.within((20, 20), PLANT_APART)


def two_towns(year: int, maker: BuildingKind, *standing: BuildingKind) -> Growth:
    constructions = Constructions(flat_map(40, 40))
    for home in ((10, 10), (30, 30)):
        for _ in range(6):
            constructions.build(Building(home, BuildingKind.HOUSE))
    constructions.build(Building((33, 33), maker))
    for col, kind in enumerate(standing):
        constructions.build(Building((1, 30 + 4 * col), kind))
    return Growth(constructions, QUIET, start_year=year)


@pytest.mark.parametrize(
    ("year", "maker", "plant", "standing"),
    [
        (1890, BuildingKind.MINE, BuildingKind.POWER_PLANT, ()),
        (1910, BuildingKind.REFINERY, BuildingKind.DIESEL_POWER_PLANT, (BuildingKind.POWER_PLANT,)),
        (
            2030,
            BuildingKind.GASWORKS,
            BuildingKind.GAS_POWER_PLANT,
            (BuildingKind.POWER_PLANT, BuildingKind.DIESEL_POWER_PLANT),
        ),
    ],
)
def test_the_first_plant_of_a_kind_goes_up_by_the_town_making_most_of_its_fuel(
    year: int, maker: BuildingKind, plant: BuildingKind, standing: tuple[BuildingKind, ...]
) -> None:
    growth = two_towns(year, maker, *standing)

    build_power_plants(growth)

    ((row, col),) = built(growth)[plant]
    assert abs(row - 30) <= PLACE_STEPS and abs(col - 30) <= PLACE_STEPS


def test_a_plant_of_every_open_kind_goes_up_one_a_year_whatever_the_map_wants() -> None:
    growth = town(2030)

    for _ in range(10):
        build_power_plants(growth)

    assert {kind: len(cells) for kind, cells in built(growth).items()} == {
        BuildingKind.POWER_PLANT: 1,
        BuildingKind.DIESEL_POWER_PLANT: 1,
        BuildingKind.GAS_POWER_PLANT: 1,
    }


def test_homes_and_factories_want_no_electricity_before_electrification_begins() -> None:
    growth = town(1870)

    wanted = electricity_wanted(growth)

    assert wanted == 0.0
