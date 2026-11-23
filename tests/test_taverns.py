import pytest

from builders import joined_stations, served_at
from railroads.sim.breeding import LIVELY_GROWTH
from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.goods import Good

STATION, HOUSE, TAVERN = (6, 20), (6, 21), (5, 20)


def served_house(tavern: bool) -> float:
    growth, railway = joined_stations()
    growth.build(Building(HOUSE, BuildingKind.HOUSE))
    if tavern:
        growth.build(Building(TAVERN, BuildingKind.TAVERN))
    railway.station_goods(STATION).brought.update(
        {Good.BREAD: 100.0, Good.COAL: 100.0, Good.POTS: 100.0, Good.BEER: 500.0, Good.VODKA: 500.0}
    )
    growth.trade_goods()
    return served_at(growth, HOUSE)


def test_a_tavern_with_all_its_drink_makes_the_homes_around_it_grow_half_as_fast_again() -> None:
    alone = served_house(tavern=False)

    lively = served_house(tavern=True)

    assert alone > 0
    assert lively == pytest.approx(alone * (1 + LIVELY_GROWTH))
