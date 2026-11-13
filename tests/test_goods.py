from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import FITS, Need, bonus_of, demands_of, makes_of, needs_of
from railroads.sim.ladders import LADDERS


def test_a_house_demands_food_and_heating_and_takes_beer_and_newspapers_as_bonus() -> None:
    demands, bonus = demands_of(BuildingKind.HOUSE, 1900), bonus_of(BuildingKind.HOUSE, 1900)

    assert set(demands) == {Need.FOOD, Need.HEATING}
    assert set(bonus) == {Need.BEER, Need.NEWSPAPERS}


def test_a_processor_demands_every_input_even_a_special_resource() -> None:
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
