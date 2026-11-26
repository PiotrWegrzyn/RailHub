import pytest
from trading import STATION, counts, holding, make_map, position

from builders import joined_stations, served_at
from railroads.sim.breeding import LIVELY_GROWTH
from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.economy import Economy
from railroads.sim.events import make_factor
from railroads.sim.goods import DIESEL_BOOST, PIPELINE, Good, Need
from railroads.sim.tiles import Tiles, activate

MINE = (8, 9)


def mined(year: int, **held: float) -> float:
    economy = Economy(make_map())
    economy.year = year
    placed = counts(economy.tile_map, (MINE, BuildingKind.IRON_MINE, 1))

    trade = economy.trade(placed, holding(1_000_000.0, **held))

    return trade.gathered[STATION][Good.IRON]


def test_an_iron_mine_with_its_diesel_digs_half_as_much_again_once_the_oil_tile_opens_it() -> None:
    activate(Tiles.until(1950))
    without = mined(1950)

    fuelled = mined(1950, FUEL=1000.0)
    too_early = mined(1915, FUEL=1000.0)

    assert fuelled == pytest.approx(without * (1 + DIESEL_BOOST))
    assert too_early == pytest.approx(mined(1915))


def test_diesel_adds_its_half_to_a_farm_that_takes_no_limestone_or_machinery_unless_owned() -> None:
    economy = Economy(make_map())
    economy.year = 1950
    placed = counts(economy.tile_map, (MINE, BuildingKind.FARM, 1))

    trade = economy.trade(placed, holding(1000.0, FUEL=10.0, LIMESTONE=10.0, MACHINERY=10.0))

    assert trade.gathered[STATION][Good.WHEAT] == pytest.approx(
        10 * (1 + DIESEL_BOOST) * make_factor(Good.WHEAT, 1950)
    )


def test_diesel_fuel_heats_a_home() -> None:
    economy = Economy(make_map())
    economy.year = 1950
    placed = counts(economy.tile_map, (MINE, BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(FUEL=10.0))

    heated, needed = trade.goods_got[(BuildingKind.HOUSE, Need.HEATING)]
    at = position(trade, economy, MINE)
    assert heated[at] == needed[at] > 0


def test_diesel_is_shared_out_before_what_it_helps_make() -> None:
    fuel = PIPELINE.index(Need.FUEL)

    later = [PIPELINE.index(need) for need in (Need.PLANKS, Need.COAL, Need.IRON, Need.WHEAT)]

    assert all(fuel < step for step in later)


def served_house(filling_station: bool) -> float:
    growth, railway = joined_stations()
    growth.start_year = 1950
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    if filling_station:
        growth.build(Building((5, 20), BuildingKind.FILLING_STATION))
    railway.station_goods((6, 20)).brought.update(
        {Good.BREAD: 100.0, Good.COAL: 100.0, Good.POTS: 100.0, Good.FUEL: 500.0}
    )
    growth.trade_goods()
    return served_at(growth, (6, 21))


def test_a_filling_station_with_all_its_fuel_makes_the_homes_around_it_grow_faster() -> None:
    alone = served_house(filling_station=False)

    lively = served_house(filling_station=True)

    assert alone > 0
    assert lively == pytest.approx(alone * (1 + LIVELY_GROWTH))
