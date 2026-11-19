import pytest
from test_railway import joined_stations

from railroads.sim.goods import General, Good, Need
from railroads.sim.railway import Railway
from railroads.sim.tech import GENERAL_CARS, NODES
from railroads.sim.trains import MEDIUM_CAR, Stop, Train

HERE, THERE = (6, 3), (6, 20)


def general_train(railway: Railway, cargo: General = General.FOOD) -> Train:
    railway.research.taken.add(GENERAL_CARS[cargo])
    train = railway.buy()
    assert train is not None
    train.stops = [Stop(HERE, [cargo]), Stop(THERE)]
    return train


def test_a_general_car_needs_its_tech() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None

    before = railway.can_carry(train, General.FOOD)
    railway.research.taken.add(GENERAL_CARS[General.FOOD])

    assert (before, railway.can_carry(train, General.FOOD)) == (False, True)


def test_a_general_food_car_takes_the_food_that_fills_it_most_when_none_pays_more() -> None:
    _, railway = joined_stations()
    train = general_train(railway)
    railway.station_goods(HERE).stockpile.update({Good.FISH: 100.0, Good.HONEY: 100.0})
    railway.station_demand(THERE).capacity = {Need.FOOD: 1000.0}

    railway.calls.arrive(train, train.stops[0])

    (car,) = train.cars
    assert (car.general, car.good) == (General.FOOD, Good.FISH)
    assert car.load == pytest.approx(car.capacity)


def test_with_nothing_wanted_ahead_a_general_car_takes_what_there_is_most_of() -> None:
    _, railway = joined_stations()
    train = general_train(railway, General.COMMON)
    railway.station_goods(HERE).stockpile.update({Good.SHOES: 5.0, Good.POTS: 40.0})

    railway.calls.arrive(train, train.stops[0])

    (car,) = train.cars
    assert (car.good, car.load) == (Good.POTS, pytest.approx(min(40.0, car.capacity)))


def test_a_general_car_unloads_its_food_where_it_is_wanted_and_stays_on_for_the_next_leg() -> None:
    _, railway = joined_stations()
    train = general_train(railway)
    train.stops[1].cars = [General.FOOD]
    railway.station_goods(HERE).stockpile[Good.APPLES] = 30.0
    railway.station_demand(THERE).capacity = {Need.FOOD: 1000.0}
    railway.calls.arrive(train, train.stops[0])
    train.heading = 0

    railway.calls.arrive(train, train.stops[1])

    (car,) = train.cars
    assert car.general == General.FOOD
    assert railway.station_goods(THERE).held(Good.APPLES) == pytest.approx(30.0)


def test_a_general_processed_food_car_carries_less_and_takes_no_grown_food() -> None:
    _, railway = joined_stations()
    railway.research.taken.add(GENERAL_CARS[General.FOOD])
    train = general_train(railway, General.PROCESSED_FOOD)
    railway.station_goods(HERE).stockpile.update({Good.APPLES: 100.0, Good.BREAD: 100.0})

    railway.calls.arrive(train, train.stops[0])

    (car,) = train.cars
    assert car.good == Good.BREAD
    assert car.capacity == pytest.approx(MEDIUM_CAR)
    assert car.capacity < railway.car_capacity(Good.BREAD, train.engine)


def test_the_processed_food_car_is_the_next_tech_after_the_food_car() -> None:
    _, processed = NODES[GENERAL_CARS[General.PROCESSED_FOOD]]

    assert processed.requires == (GENERAL_CARS[General.FOOD],)
