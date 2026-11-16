import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.economy import Trade
from railroads.sim.fuel_market import (
    BASE_PRICE,
    ELECTRICITY_SHORT,
    OWN_POWER_SHARE,
    POWER_PLANT_OUTPUT,
    RETAIL_MARKUP,
    Fuel,
    FuelMarket,
    Signals,
    inflation,
    signals_of,
)
from railroads.sim.goods import Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.tech import Research
from railroads.sim.trains import ENGINES, Stop

WEST, EAST = (6, 3), (6, 20)


def line(year: int = 1850) -> Railway:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, WEST)
    build_station(growth, EAST)
    route = plan_route(constructions, WEST, EAST)
    assert route is not None
    lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    railway.finances.earn(10_000_000.0, "test")
    railway.year = year
    return railway


def set_off(railway: Railway, engine: str = "planet") -> float:
    train = railway.buy(engine)
    assert train is not None
    train.stops = [Stop(WEST, wait_for_load=False), Stop(EAST, wait_for_load=False)]
    railway.run(0.001)
    return -railway.finances.month["coal"]


def test_fuel_prices_rise_every_year_compounded() -> None:
    years = [inflation(Fuel.COAL, year) for year in (1840, 1841, 1842)]

    assert years == [1.0, pytest.approx(1.02), pytest.approx(1.02**2)]


def test_a_train_setting_off_buys_the_coal_for_its_way_at_retail_without_coal_at_the_station() -> (
    None
):
    railway = line()

    paid = set_off(railway)

    leg = 17.0
    price = BASE_PRICE[Fuel.COAL] * inflation(Fuel.COAL, 1850) * (1 + RETAIL_MARKUP)
    assert paid == pytest.approx(ENGINES["planet"].burn * leg * price, rel=0.05)


def test_coal_out_of_the_stations_stockpile_costs_only_the_market_price() -> None:
    retail, stocked = line(), line()
    stocked.station_goods(WEST).stockpile[Good.COAL] = 100.0

    paid = set_off(retail), set_off(stocked)

    assert paid[1] == pytest.approx(paid[0] / (1 + RETAIL_MARKUP))
    assert stocked.station_goods(WEST).stockpile[Good.COAL] < 100.0


def test_the_regular_customer_node_cuts_the_retail_markup_to_a_quarter() -> None:
    plain, regular = line(), line()
    regular.research = Research(taken={"books", "regular_customer"})

    paid = set_off(plain), set_off(regular)

    assert paid[1] == pytest.approx(paid[0] * 1.25 / 1.5)


def test_a_train_cannot_set_off_for_a_stop_further_than_its_fuel_takes_it() -> None:
    railway = line(1804)
    train = railway.buy("penydarren")
    assert train is not None
    train.stops = [Stop(WEST, wait_for_load=False), Stop(EAST, wait_for_load=False)]

    railway.run(0.001)

    assert train.path == []
    assert train.status == "Next stop 17 cells away; it carries fuel for 4"
    assert railway.finances.month["coal"] == 0.0


def test_more_demand_against_supply_than_in_the_first_month_raises_the_price_bit_by_bit() -> None:
    market = FuelMarket()
    market.month(1850, {Fuel.COAL: Signals(buildings=1000.0, natural=1000.0)})
    first = market.price(Fuel.COAL, 1850)

    for _ in range(3):
        market.month(1850, {Fuel.COAL: Signals(buildings=4000.0, natural=1000.0)})

    prices = [month.price for month in market.history[Fuel.COAL]]
    assert prices == sorted(prices)
    assert first < prices[-1] < 2 * first


def test_coal_handed_out_by_the_players_stations_brings_the_price_down() -> None:
    market = FuelMarket()
    market.month(1850, {Fuel.COAL: Signals(buildings=1000.0, natural=1000.0)})

    for _ in range(3):
        market.month(1850, {Fuel.COAL: Signals(buildings=1000.0, network=3000.0, natural=1000.0)})

    assert market.factor[Fuel.COAL] < 1.0


def test_the_coal_trains_buy_counts_towards_next_months_demand() -> None:
    market = FuelMarket()
    market.buy(Fuel.COAL, 250.0, 1850)

    market.month(1850, {Fuel.COAL: Signals(buildings=1000.0, natural=1000.0)})

    assert market.history[Fuel.COAL][-1].trains == 250.0
    assert market.last_bought[Fuel.COAL] == 250.0


def test_electricity_is_dear_until_the_maps_power_plants_get_their_fuel() -> None:
    starved, fed = FuelMarket(), FuelMarket()

    for _ in range(40):
        starved.month(1950, {Fuel.ELECTRICITY: Signals(plants_fed=0.0)})
        fed.month(1950, {Fuel.ELECTRICITY: Signals(plants_fed=1.0)})

    assert starved.factor[Fuel.ELECTRICITY] == pytest.approx(ELECTRICITY_SHORT, rel=1e-3)
    assert fed.factor[Fuel.ELECTRICITY] == pytest.approx(1.0)


def test_electricity_from_the_players_own_power_plants_costs_a_fifth() -> None:
    market = FuelMarket()
    market.month(1950, {Fuel.ELECTRICITY: Signals(plants_fed=1.0, own_power=100.0)})
    price = market.price(Fuel.ELECTRICITY, 1950)

    cost = market.buy(Fuel.ELECTRICITY, 150.0, 1950)

    assert cost == pytest.approx(100.0 * price * OWN_POWER_SHARE + 50.0 * price)
    assert market.own_power == 0.0


def test_a_fed_power_plant_the_player_owns_makes_electricity_and_others_do_not() -> None:
    cells = np.array([0, 1])
    got = {
        (BuildingKind.POWER_PLANT, Need.BOILER_FUEL): (
            np.array([200.0, 200.0]),
            np.array([200.0, 200.0]),
        )
    }
    trade = Trade(cells, {}, {}, {}, {}, goods_got=got)
    owned = np.array([[1, 0]])

    signals = signals_of(trade, owned)

    assert signals[Fuel.ELECTRICITY].own_power == pytest.approx(POWER_PLANT_OUTPUT)
    assert signals[Fuel.ELECTRICITY].plants_fed == pytest.approx(1.0)


def test_a_power_plant_can_be_bought_for_two_million() -> None:
    growth = Growth(Constructions(flat_map()), GrowthRule(trades=False))
    growth.counts[BuildingKind.POWER_PLANT] = np.zeros(growth.grid.shape, dtype=np.int64)

    price = growth.owners.price_to_buy(BuildingKind.POWER_PLANT)

    assert price == 2_000_000.0
