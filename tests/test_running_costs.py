import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.facilities import Facility
from railroads.sim.finance import Finances
from railroads.sim.goods import Good
from railroads.sim.railway import Railway
from railroads.sim.running_costs import (
    MOST_DELIVERED_CUT,
    MOST_OWNED_CUT,
    MOST_POWER_STATION_CUT,
    Business,
    fuel_price,
    running_cost,
)
from railroads.sim.trains import ENGINES, MAGLEV_FARE, Power, Train
from railroads.ui.reports_window import books_by_year


def test_coal_gets_dear_fast_and_diesel_and_electricity_only_slowly() -> None:
    years = 1900, 2000

    steam = [fuel_price(Power.STEAM, year) for year in years]
    diesel = [fuel_price(Power.DIESEL, year) for year in years]
    electric = [fuel_price(Power.ELECTRIC, year) for year in years]

    assert steam[0] > 3.0 and steam[1] > 7.0
    assert diesel[1] < 1.7
    assert electric[1] < diesel[1]


def test_by_1950_a_pacific_costs_more_to_run_than_an_electric_twice_as_fast() -> None:
    steam, electric = ENGINES["pacific"], ENGINES["boxcab"]

    costs = running_cost(steam, 1950), running_cost(electric, 1950)

    assert costs[0] > 5 * costs[1]


def test_owned_buildings_and_delivered_freight_cut_running_costs_a_little_up_to_a_limit() -> None:
    planet = ENGINES["planet"]

    some = running_cost(planet, 1850, Business(owned=10, delivered=500))
    most = running_cost(planet, 1850, Business(owned=10_000, delivered=10_000_000))

    full = running_cost(planet, 1850)
    assert some == pytest.approx(full * 0.95 * 0.95)
    assert most == pytest.approx(full * (1 - MOST_OWNED_CUT) * (1 - MOST_DELIVERED_CUT))


def test_power_stations_cut_only_electric_and_maglev_running_costs() -> None:
    stations = Business(power_stations=10)

    cuts = {power: stations.cut(power) for power in Power}

    assert cuts[Power.ELECTRIC] == cuts[Power.MAGLEV] == pytest.approx(MOST_POWER_STATION_CUT)
    assert cuts[Power.STEAM] == cuts[Power.DIESEL] == 0.0


def test_a_working_power_station_makes_every_electric_engine_cheaper_to_run() -> None:
    railway = Railway(Constructions(flat_map()))
    railway.year = 1950
    before = railway.running_cost("boxcab")

    railway.supplied = {(6, 3): {Facility.POWER_STATION}}

    assert railway.running_cost("boxcab") == pytest.approx(0.85 * before)


def test_cars_behind_a_diesel_hold_half_as_much_again() -> None:
    railway = Railway(Constructions(flat_map()))

    sizes = [railway.car_capacity(Good.COAL, engine) for engine in ("big_boy", "heavy_diesel")]

    assert sizes[1] == pytest.approx(1.5 * sizes[0])


def test_passengers_on_a_maglev_pay_eleven_times_as_much() -> None:
    fares = [Railway.fare_factor(Train("T", engine=name)) for name in ("maglev", "bullet")]

    assert fares == [MAGLEV_FARE, pytest.approx(1.2)]
    assert MAGLEV_FARE == 11.0


def test_the_books_keep_each_years_money_by_what_it_was_for() -> None:
    finances = Finances()
    finances.spend(100.0, "running trains")
    finances.close_month(1850)
    finances.spend(50.0, "running trains")
    finances.close_month(1851)
    finances.spend(30.0, "track upkeep")
    finances.earn(70.0, "cargo")

    rows = books_by_year(finances, 1851)

    assert rows == [
        ("running trains", -150.0, 0.0),
        ("track upkeep", 0.0, -30.0),
        ("cargo", 0.0, 70.0),
    ]
