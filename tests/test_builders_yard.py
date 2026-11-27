import pytest
from trading import STATION, counts, holding, position
from trading import make_map as trade_map

from builders import joined_stations, served_at
from railroads.sim.breeding import BUILT_UP_GROWTH
from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.economy import Economy
from railroads.sim.goods import Good
from railroads.sim.growth import inputs_of
from railroads.sim.trade import BONUS_PER_GOOD

YARD_STATION, HOUSE, YARD = (6, 20), (6, 21), (5, 20)


def served_house(yard: bool) -> float:
    growth, railway = joined_stations()
    growth.build(Building(HOUSE, BuildingKind.HOUSE))
    if yard:
        growth.build(Building(YARD, BuildingKind.BUILDERS_YARD))
    railway.station_goods(YARD_STATION).brought.update(
        {
            Good.BREAD: 100.0,
            Good.COAL: 100.0,
            Good.BRICKS: 100.0,
            Good.CEMENT: 100.0,
            Good.GLASS: 100.0,
            Good.PLANKS: 100.0,
        }
    )
    growth.trade_goods()
    return served_at(growth, HOUSE)


def test_a_builders_yard_with_all_it_must_have_makes_the_homes_around_it_come_up_twice_as_fast() -> (
    None
):
    alone = served_house(yard=False)

    built_up = served_house(yard=True)

    assert alone > 0
    assert built_up == pytest.approx(alone * (1 + BUILT_UP_GROWTH))


def test_a_builders_yard_opens_only_once_the_map_makes_bricks_cement_and_glass() -> None:
    makers = inputs_of(BuildingKind.BUILDERS_YARD, 1900)

    assert [BuildingKind.BRICKWORKS in kinds for kinds in makers].count(True) == 1
    assert [BuildingKind.CEMENT_WORKS in kinds for kinds in makers].count(True) == 1
    assert [BuildingKind.GLASSWORKS in kinds for kinds in makers].count(True) == 1
    assert len(makers) == 3


def test_bricks_make_a_factory_more_satisfied_but_not_make_more_goods() -> None:
    tile_map = trade_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.FACTORY, 1))
    without = economy.trade(placed, holding(100.0, PLANKS=100.0, STEEL=100.0))

    with_bricks = economy.trade(placed, holding(100.0, PLANKS=100.0, STEEL=100.0, BRICKS=100.0))

    here = position(with_bricks, economy, (8, 8))
    made = without.gathered[STATION][Good.GOODS]
    assert with_bricks.gathered[STATION][Good.GOODS] == pytest.approx(made)
    assert with_bricks.share((BuildingKind.FACTORY,))[here] == pytest.approx(
        without.share((BuildingKind.FACTORY,))[here] * (1 + BONUS_PER_GOOD)
    )
