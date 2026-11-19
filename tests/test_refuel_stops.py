import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.goods import Good, Need, freight_of
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.trains import Stop, Train

FIRST, MIDDLE, LAST = (6, 3), (6, 14), (6, 26)
# Carries fuel for 13 cells: enough for each leg, not for the whole way.
SHORT_RANGE = "tank"


def line_of_three() -> Railway:
    constructions = Constructions(flat_map(cols=32))
    growth = Growth(constructions, GrowthRule(trades=False))
    for station in (FIRST, MIDDLE, LAST):
        build_station(growth, station)
    for start, end in ((FIRST, MIDDLE), (MIDDLE, LAST)):
        route = plan_route(constructions, start, end)
        assert route is not None
        lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    railway.year = 1850
    return railway


def passenger_train(railway: Railway, refuel: bool = True) -> Train:
    railway.station_goods(FIRST).stockpile[Good.PASSENGERS] = 30.0
    railway.station_demand(LAST).capacity = {Need.PASSENGERS: 200.0}
    train = railway.buy(SHORT_RANGE)
    assert train is not None
    stops = [Stop(FIRST, [Good.PASSENGERS])]
    if refuel:
        stops.append(Stop(MIDDLE, [Good.COAL], refuel_only=True))
    train.stops = [*stops, Stop(LAST)]
    return train


def run_until(railway: Railway, train: Train, reached: int, most_months: float = 12.0) -> None:
    months = 0.0
    while train.arrivals < reached and months < most_months:
        railway.run(0.01)
        months += 0.01


def test_passengers_ride_through_a_refuelling_stop_and_pay_at_the_far_station() -> None:
    railway = line_of_three()
    train = passenger_train(railway)

    run_until(railway, train, reached=2)

    assert railway.finances.month["passengers"] > 0.95 * 30 * freight_of(Good.PASSENGERS)


def test_nothing_unloads_or_loads_at_a_stop_the_train_only_refuels_at() -> None:
    railway = line_of_three()
    train = passenger_train(railway)
    railway.station_goods(MIDDLE).stockpile[Good.COAL] = 50.0

    run_until(railway, train, reached=1)

    middle = railway.station_goods(MIDDLE)
    assert [(car.good, round(car.load)) for car in train.cars] == [(Good.PASSENGERS, 30)]
    assert not middle.brought.get(Good.PASSENGERS)
    assert middle.stockpile[Good.COAL] == 50.0
    assert "passengers" not in railway.finances.month


def test_the_train_sets_off_from_a_refuelling_stop_for_the_far_station() -> None:
    railway = line_of_three()
    train = passenger_train(railway)
    run_until(railway, train, reached=1)

    railway.run(0.01)

    assert train.stops[train.heading % len(train.stops)].cell == LAST
    assert train.status == "Running"


def test_without_a_refuelling_stop_the_far_station_is_beyond_the_engines_range() -> None:
    railway = line_of_three()
    train = passenger_train(railway, refuel=False)

    railway.run(0.01)

    assert train.status.startswith("Next stop")


def test_cars_bound_past_a_refuelling_stop_are_worth_what_the_far_station_pays() -> None:
    railway = line_of_three()
    train = passenger_train(railway)
    railway.run(0.02)

    destination = railway.destination(train)

    assert destination == LAST
    assert railway.calls.next_cell(train, train.stops[0]) == LAST
    assert sum(railway.cargo_worth(train)) == pytest.approx(
        30 * freight_of(Good.PASSENGERS) * railway.price_factor(Good.PASSENGERS), rel=0.05
    )
