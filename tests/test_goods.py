from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import Need, bonus_of, demands_of


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
