import numpy as np
import pytest

from builders import flat_map, layer
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.economy import Trade
from railroads.sim.events import electricity_use
from railroads.sim.fuel_market import (
    BASE_PRICE,
    ELECTRICITY_LEAST,
    ELECTRICITY_MOST,
    ELECTRICITY_PER_POINT,
    OWN_POWER_SHARE,
    POWER_PLANT_OUTPUT,
    RETAIL_MARKUP,
    Fuel,
    FuelMarket,
    Signals,
    electrified,
    fuel_of,
    signals_of,
)
from railroads.sim.goods import Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.tech import Research
from railroads.sim.tilemap import Relief
from railroads.sim.trains import BULK_CAR, BURN_ALONE, ENGINES, Car, Stop, Train

WEST, EAST = (6, 3), (6, 20)


def line(year: int = 1850, relief: Relief = Relief.FLAT) -> Railway:
    constructions = Constructions(flat_map(relief=layer(12, 30, relief)))
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


def set_off(railway: Railway, engine: str = "planet", fuel: Fuel = Fuel.COAL) -> float:
    train = railway.buy(engine)
    assert train is not None
    train.stops = [Stop(WEST, wait_for_load=False), Stop(EAST, wait_for_load=False)]
    railway.run(0.001)
    return -railway.finances.month.get(fuel.value, 0.0)


def test_fuel_prices_stay_put_over_the_years_without_market_pressure() -> None:
    market = FuelMarket()

    for year in (1850, 1900, 1950):
        market.month(year, {Fuel.COAL: Signals(buildings=1000.0, natural=1000.0)})

    assert {month.price for month in market.history[Fuel.COAL]} == {BASE_PRICE[Fuel.COAL]}


def test_a_train_setting_off_buys_the_coal_for_its_way_at_retail_without_coal_at_the_station() -> (
    None
):
    railway = line()

    paid = set_off(railway)

    leg = 17.0
    price = BASE_PRICE[Fuel.COAL] * (1 + RETAIL_MARKUP)
    assert paid == pytest.approx(ENGINES["planet"].burn * BURN_ALONE * leg * price, rel=0.05)


def test_coal_out_of_the_stations_stockpile_costs_only_the_market_price() -> None:
    retail, stocked = line(), line()
    stocked.station_goods(WEST).stockpile[Good.COAL] = 100.0

    paid = set_off(retail), set_off(stocked)

    assert paid[1] == pytest.approx(paid[0] / (1 + RETAIL_MARKUP))
    assert stocked.station_goods(WEST).stockpile[Good.COAL] < 100.0


def test_the_american_burns_wood_taking_logs_out_of_the_stations_stockpile() -> None:
    railway = line(1860)
    railway.station_goods(WEST).stockpile[Good.LOGS] = 100.0

    wood = set_off(railway, "american", Fuel.WOOD)

    assert wood > 0
    assert railway.finances.month.get(Fuel.COAL.value, 0.0) == 0
    assert railway.station_goods(WEST).stockpile[Good.LOGS] < 100.0


@pytest.mark.parametrize(
    ("engine", "fuel"),
    [("planet", Fuel.COAL), ("american", Fuel.WOOD), ("heavy_diesel", Fuel.DIESEL)],
)
def test_each_engine_burns_its_own_fuel(engine: str, fuel: Fuel) -> None:
    burnt = ENGINES[engine]

    found = fuel_of(burnt)

    assert found == fuel


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


def test_a_train_sets_off_for_a_stop_as_far_as_its_fuel_takes_it_rounded_to_whole_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    railway = line()
    train = railway.buy("tank")
    assert train is not None
    train.stops = [Stop(WEST, wait_for_load=False), Stop(EAST, wait_for_load=False)]
    monkeypatch.setattr(railway.network, "route_length", lambda *_: ENGINES["tank"].tank + 0.4)

    railway.run(0.001)

    assert train.status == "Running"


def test_more_demand_against_supply_than_in_the_first_month_raises_the_price_bit_by_bit() -> None:
    market = FuelMarket()
    market.month(1850, {Fuel.COAL: Signals(buildings=1000.0, natural=1000.0)})
    first = market.price(Fuel.COAL)

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
    market.buy(Fuel.COAL, 250.0)

    market.month(1850, {Fuel.COAL: Signals(buildings=1000.0, natural=1000.0)})

    assert market.history[Fuel.COAL][-1].trains == 250.0
    assert market.last_bought[Fuel.COAL] == 250.0


def test_electricity_rests_at_its_base_price_with_no_trains_or_plants() -> None:
    market = FuelMarket()

    for _ in range(40):
        market.month(1950, {Fuel.ELECTRICITY: Signals(buildings=5000.0, natural=5000.0)})

    assert market.price(Fuel.ELECTRICITY) == pytest.approx(BASE_PRICE[Fuel.ELECTRICITY])


def test_electric_trains_raise_the_price_and_power_plants_lower_it_but_only_a_little() -> None:
    busy, powered = FuelMarket(), FuelMarket()

    for _ in range(60):
        busy.buy(Fuel.ELECTRICITY, 50_000.0)
        busy.month(1950, {Fuel.ELECTRICITY: Signals(buildings=5000.0, natural=5000.0)})
        powered.month(
            1950, {Fuel.ELECTRICITY: Signals(buildings=5000.0, network=50_000.0, natural=5000.0)}
        )

    base = BASE_PRICE[Fuel.ELECTRICITY]
    assert busy.price(Fuel.ELECTRICITY) == pytest.approx(ELECTRICITY_MOST * base, rel=1e-3)
    assert powered.price(Fuel.ELECTRICITY) == pytest.approx(ELECTRICITY_LEAST * base, rel=1e-3)


@pytest.mark.parametrize(("year", "share"), [(1850, 0.0), (1880, 0.0), (1955, 0.5), (2030, 1.0)])
def test_homes_and_factories_are_electrified_bit_by_bit_and_all_by_2030(
    year: int, share: float
) -> None:
    electrified_now = electrified(year)

    assert electrified_now == pytest.approx(share)


def test_homes_and_factories_want_electricity_by_their_points_and_how_far_electrified() -> None:
    trade = Trade(np.array([0]), {}, {}, {}, {})

    signals = signals_of(trade, None, 1000.0, 1955)

    assert signals[Fuel.ELECTRICITY].buildings == pytest.approx(
        ELECTRICITY_PER_POINT * 1000 * 0.5 * electricity_use(1955)
    )
    assert signals[Fuel.ELECTRICITY].natural == 0.0


def test_electricity_from_the_players_own_power_plants_costs_a_fifth() -> None:
    market = FuelMarket()
    market.month(1950, {Fuel.ELECTRICITY: Signals(plants_fed=1.0, own_power=100.0)})
    price = market.price(Fuel.ELECTRICITY)

    cost = market.buy(Fuel.ELECTRICITY, 150.0)

    assert cost == pytest.approx(100.0 * price * OWN_POWER_SHARE + 50.0 * price)
    assert market.own_power == 0.0


def test_a_fed_power_plant_the_player_owns_makes_electricity_and_others_do_not() -> None:
    cells = np.array([0, 1])
    got = {
        (BuildingKind.POWER_PLANT, Need.COAL): (
            np.array([200.0, 200.0]),
            np.array([200.0, 200.0]),
        )
    }
    trade = Trade(cells, {}, {}, {}, {}, goods_got=got)
    owned = {BuildingKind.POWER_PLANT: np.array([[1, 0]])}

    signals = signals_of(trade, owned)

    assert signals[Fuel.ELECTRICITY].own_power == pytest.approx(POWER_PLANT_OUTPUT)
    assert signals[Fuel.ELECTRICITY].network == pytest.approx(2 * POWER_PLANT_OUTPUT)
    assert signals[Fuel.ELECTRICITY].plants_fed == pytest.approx(1.0)


def test_a_power_plant_can_be_bought_for_two_million() -> None:
    growth = Growth(Constructions(flat_map()), GrowthRule(trades=False))
    growth.counts[BuildingKind.POWER_PLANT] = np.zeros(growth.grid.shape, dtype=np.int64)

    price = growth.owners.price_to_buy(BuildingKind.POWER_PLANT)

    assert price == 2_000_000.0


def test_a_heavier_train_burns_more_fuel_a_cell() -> None:
    railway = line()
    empty = Train("Empty", cars=[Car(Good.COAL)])
    full = Train("Full", cars=[Car(Good.COAL, BULK_CAR)])

    burns = railway.burn(empty), railway.burn(full)

    assert burns[0] < burns[1]


def test_a_train_burns_half_its_engines_fuel_alone_and_half_as_much_again_at_its_most() -> None:
    railway = line()
    engine = ENGINES["planet"]
    alone = Train("Alone")
    heaviest = Train("Heaviest", cars=[Car(Good.COAL, BULK_CAR) for _ in range(engine.cars)])

    burns = railway.burn(alone), railway.burn(heaviest)

    assert burns == (pytest.approx(0.5 * engine.burn), pytest.approx(1.5 * engine.burn))


def wheat_train(wheat_waiting: float) -> Railway:
    railway = line()
    railway.station_goods(WEST).stockpile[Good.WHEAT] = wheat_waiting
    train = railway.buy()
    assert train is not None
    train.stops = [Stop(WEST, [Good.WHEAT] * 3, wait_for_load=False), Stop(EAST)]
    return railway


def test_a_train_setting_off_full_buys_more_coal_than_one_setting_off_empty() -> None:
    empty, full = wheat_train(0.0), wheat_train(1_000.0)

    empty.run(0.001)
    full.run(0.001)

    paid = [-railway.finances.month.get(Fuel.COAL.value, 0.0) for railway in (empty, full)]
    assert 0 < paid[0] < paid[1]


def test_a_leg_through_hills_burns_half_as_much_fuel_again_as_its_length() -> None:
    railway = line(relief=Relief.HILLS)

    length = railway.network.route_length(WEST, EAST)
    burnt = railway.network.fuel_length(WEST, EAST)

    assert length is not None and burnt is not None
    assert burnt == pytest.approx(1.5 * length, rel=0.1)


def test_a_leg_over_flat_land_burns_its_length_in_fuel() -> None:
    railway = line()

    length = railway.network.route_length(WEST, EAST)
    burnt = railway.network.fuel_length(WEST, EAST)

    assert burnt == pytest.approx(length)


def test_a_train_through_hills_buys_more_fuel_than_one_over_flat_land() -> None:
    flat, hills = line(), line(relief=Relief.HILLS)

    paid = set_off(flat), set_off(hills)

    assert paid[0] < paid[1]
