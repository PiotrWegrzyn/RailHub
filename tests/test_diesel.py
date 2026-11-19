import pytest
from test_economy import STATION, counts, holding, make_map, position
from test_railway import joined_stations
from test_taverns import served_at

from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.economy import Economy
from railroads.sim.goods import DIESEL_BOOST, PIPELINE, Good, Need
from railroads.sim.growth import LIVELY_GROWTH, breeding_of
from railroads.sim.ladders import FILLING_STATIONS, POINTS, WORKS, Points

MINE = (8, 9)


def mined(year: int, **held: float) -> float:
    economy = Economy(make_map())
    economy.year = year
    placed = counts(economy.tile_map, (MINE, BuildingKind.IRON_MINE, 1))

    trade = economy.trade(placed, holding(1000.0, **held))

    return trade.gathered[STATION][Good.IRON]


def test_an_iron_mine_with_its_diesel_digs_half_as_much_again_from_1920() -> None:
    without = mined(1950)

    fuelled = mined(1950, FUEL=10.0)
    too_early = mined(1919, FUEL=10.0)

    assert fuelled == pytest.approx(without * (1 + DIESEL_BOOST))
    assert too_early == pytest.approx(without)


def test_diesel_adds_its_half_to_a_farm_that_takes_no_limestone_or_machinery_unless_owned() -> None:
    economy = Economy(make_map())
    economy.year = 1950
    placed = counts(economy.tile_map, (MINE, BuildingKind.FARM, 1))

    trade = economy.trade(placed, holding(1000.0, FUEL=10.0, LIMESTONE=10.0, MACHINERY=10.0))

    assert trade.gathered[STATION][Good.WHEAT] == pytest.approx(10 * (1 + DIESEL_BOOST))


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

    later = [PIPELINE.index(need) for need in (Need.LUMBER, Need.COAL, Need.IRON, Need.WHEAT)]

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


def test_refineries_come_up_near_ore_and_ports_and_filling_stations_from_1920() -> None:
    (refineries,) = [ladder for ladder in WORKS if ladder.kinds[0] == BuildingKind.REFINERY]
    breeding = breeding_of(refineries, POINTS)

    near = breeding.drawn_by

    assert near == {Points.ORE: 1.0, Points.TRADE: 2.0}
    assert FILLING_STATIONS.from_year == 1920
