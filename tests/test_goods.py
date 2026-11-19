import pytest

from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import (
    FITS,
    RECIPES,
    Need,
    bonus_of,
    demands_of,
    makes_of,
    needs_of,
)
from railroads.sim.ladders import LADDERS


def test_a_processor_takes_no_bonus_goods_and_demands_its_special_resource() -> None:
    bonus = bonus_of(BuildingKind.STEEL_MILL, 1900)

    assert bonus == {}
    assert Need.IRON in demands_of(BuildingKind.STEEL_MILL, 1900)


def test_a_jeweller_keeps_gold_and_diamonds_as_demands_since_they_are_all_it_wants() -> None:
    demands = demands_of(BuildingKind.JEWELLER, 1900)

    assert set(demands) == {Need.GOLD, Need.DIAMONDS}
    assert bonus_of(BuildingKind.JEWELLER, 1900) == {}


def test_nothing_that_can_open_needs_what_nothing_open_makes_yet() -> None:
    opens: dict[BuildingKind, int] = {}
    for ladder in LADDERS:
        for kind in ladder.kinds:
            opens[kind] = min(opens.get(kind, ladder.from_year), ladder.from_year)

    unmet = []
    for year in range(850, 2051, 5):
        made = {good for kind, y in opens.items() if y <= year for good in makes_of(kind, year)}
        unmet += [
            (kind, need, year)
            for kind, y in opens.items()
            if y <= year
            for need in needs_of(kind, year)
            if not any(good in made for good in FITS[need])
        ]

    assert unmet == []


@pytest.mark.parametrize("kind", [BuildingKind.TENEMENT, BuildingKind.FLATS])
def test_tenements_and_flats_must_have_food_common_goods_and_heating_and_take_the_rest_as_bonus(
    kind: BuildingKind,
) -> None:
    demands = demands_of(kind, 2000)

    assert set(demands) == {Need.FOOD, Need.COMMON, Need.HEATING}
    assert set(bonus_of(kind, 2000)) == set(needs_of(kind, 2000)) - set(demands)


def test_a_house_must_have_only_food_and_heating_and_takes_common_goods_as_a_bonus() -> None:
    demands = demands_of(BuildingKind.HOUSE, 2000)

    assert set(demands) == {Need.FOOD, Need.HEATING}
    assert Need.COMMON in bonus_of(BuildingKind.HOUSE, 2000)
    assert set(bonus_of(BuildingKind.HOUSE, 2000)) == set(needs_of(BuildingKind.HOUSE, 2000)) - set(
        demands
    )


@pytest.mark.parametrize(
    "kind",
    [
        kind
        for kind, recipe in RECIPES.items()
        if recipe.makes in FITS[Need.ALCOHOL] and not kind.name.startswith(("BIG_", "MEGA_"))
    ],
)
def test_alcohol_is_no_food_so_no_maker_of_it_makes_more_than_it_takes_in(
    kind: BuildingKind,
) -> None:
    recipe = RECIPES[kind]

    per_unit = recipe.per_unit

    assert sum(per_unit.values()) >= 1.0
