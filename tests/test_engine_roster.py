import itertools

import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.goods import Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.tech import NODES, Effect, Research
from railroads.sim.trains import (
    ENGINES,
    EXPRESS_FARE,
    FULL_LOAD,
    MAX_CARS,
    Car,
    Kind,
    Stop,
    Train,
)
from railroads.ui.tech_window import describe
from railroads.ui.trains_window import engine_summary, engines_by_kind

WEST, EAST = (6, 3), (6, 20)
EVERY_NODE = frozenset(NODES)


def line() -> Railway:
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
    return railway


def knowing(railway: Railway, *keys: str) -> Railway:
    railway.research = Research(taken=set(keys))
    return railway


def by_year(kind: Kind) -> list[str]:
    return sorted((n for n, e in ENGINES.items() if e.kind == kind), key=lambda n: ENGINES[n].year)


def full_train(engine: str, cars: int) -> Train:
    train = Train("Test", engine=engine)
    train.cars = [Car(Good.COAL, Car(Good.COAL).capacity) for _ in range(cars)]
    return train


def test_there_are_about_thirty_engines() -> None:
    count = len(ENGINES)

    assert 28 <= count <= 32


def test_the_first_engine_on_rails_is_the_only_one_for_sale_in_1804() -> None:
    railway = line()
    railway.year = 1804

    engines = railway.engines()

    assert engines == ["penydarren"]
    assert ENGINES["penydarren"].cars == 1


def test_in_1850_only_a_cheap_small_engine_and_the_planet_are_for_sale() -> None:
    railway = line()
    railway.year = 1850

    engines = railway.engines()

    assert sorted(engines) == ["planet", "tank"]
    tank, planet = ENGINES["tank"], ENGINES["planet"]
    assert (tank.cars, planet.cars) == (3, MAX_CARS)
    assert tank.price < planet.price
    assert tank.speed < planet.speed == 1.0


def test_every_engine_before_the_planet_is_worse_and_cheaper() -> None:
    planet = ENGINES["planet"]

    pioneers = [ENGINES[name] for name in by_year(Kind.PIONEER)]

    assert all(e.year < planet.year for e in pioneers)
    assert all(e.cars < planet.cars and e.speed < planet.speed for e in pioneers)
    assert all(e.price < planet.price for e in pioneers)
    assert [e.cars for e in pioneers] == sorted(e.cars for e in pioneers)


@pytest.mark.parametrize("kind", [Kind.LIGHT, Kind.GENERAL, Kind.HEAVY, Kind.EXPRESS])
def test_each_class_only_gets_better_over_time(kind: Kind) -> None:
    engines = [ENGINES[name] for name in by_year(kind)]

    pairs = list(itertools.pairwise(engines))

    assert len(engines) >= 4
    assert all(new.speed >= old.speed and new.cars >= old.cars for old, new in pairs)
    assert all(new.speed > old.speed or new.cars > old.cars for old, new in pairs)


def test_the_first_heavy_freight_engine_pulls_two_more_cars_than_the_planet_more_slowly() -> None:
    heavy = ENGINES[by_year(Kind.HEAVY)[0]]

    planet = ENGINES["planet"]

    assert heavy.cars == planet.cars + 2
    assert heavy.speed < planet.speed


def test_the_specialised_classes_come_soon_after_1850_and_need_their_tech() -> None:
    railway = knowing(line(), "boilers", "express_trains")
    railway.year = 1865
    without = line()
    without.year = 1865

    engines = railway.engines()

    assert {"goods", "single"} <= set(engines)
    assert not {"goods", "single"} & set(without.engines())


def test_the_game_ends_with_a_500_kmh_electric_and_a_diesel_pulling_forty_cars() -> None:
    railway = knowing(line(), *EVERY_NODE)
    railway.year = 2040

    engines = {name: ENGINES[name] for name in railway.engines()}

    assert any(e.electric and e.kmh == pytest.approx(500.0) for e in engines.values())
    assert any(e.power.value == "diesel" and e.cars == 40 for e in engines.values())


def test_a_newer_engine_of_the_same_line_takes_the_older_ones_place_on_sale() -> None:
    railway = line()
    railway.year = 1870

    engines = railway.engines()

    assert "american" in engines
    assert "planet" not in engines


def test_an_electric_does_not_end_the_sale_of_the_engine_it_would_replace() -> None:
    railway = knowing(line(), "superheaters", "traction")
    railway.year = 1925

    engines = railway.engines()

    assert {"pacific", "boxcab"} <= set(engines)


def test_every_tech_an_engine_needs_is_a_node_that_opens_engines() -> None:
    needed = {need for engine in ENGINES.values() for need in engine.needs}

    opening = {key for key, (_, node) in NODES.items() if node.effect == Effect.UNLOCK}

    assert needed == opening


def test_an_express_carries_only_passengers() -> None:
    railway = knowing(line(), "express_trains")
    railway.year = 1865
    train = railway.buy("single")
    assert train is not None

    allowed = [good for good in Good if railway.can_carry(train, good)]

    assert allowed == [Good.PASSENGERS]


def test_an_express_leads_its_coaches_with_an_empty_dining_car() -> None:
    railway = knowing(line(), "express_trains")
    railway.year = 1865
    train = railway.buy("single")
    assert train is not None
    train.stops = [Stop(WEST, [Good.PASSENGERS, Good.COAL]), Stop(EAST)]
    railway.station_goods(WEST).stockpile[Good.PASSENGERS] = 1_000.0

    railway.calls.arrive(train, train.stops[0])

    dining, *coaches = train.cars
    assert dining.dining and dining.load == 0.0
    assert [car.good for car in coaches] == [Good.PASSENGERS]
    assert coaches[0].load > 0
    assert train.weight == pytest.approx(1.0 + 1.0 + FULL_LOAD)


def test_an_express_pays_a_car_trip_for_its_dining_car() -> None:
    railway = knowing(line(), "express_trains")
    railway.year = 1865
    train = railway.buy("single")
    assert train is not None
    train.stops = [Stop(WEST, [Good.PASSENGERS]), Stop(EAST)]
    train.stops[0].wait_for_load = False

    railway.run(0.001)

    assert railway.finances.month["car trips"] == pytest.approx(-2 * 20.0)


def test_passengers_on_an_express_pay_more() -> None:
    plain_railway, express_railway = line(), knowing(line(), "express_trains")
    express_railway.year = plain_railway.year = 1865
    plain = plain_railway.buy("planet")
    express = express_railway.buy("single")
    assert plain is not None and express is not None
    for railway, train in ((plain_railway, plain), (express_railway, express)):
        train.stops = [Stop(WEST, [Good.PASSENGERS]), Stop(EAST)]
        train.loaded_at = WEST
        train.cars = [Car(Good.PASSENGERS, 40.0, {WEST: 40.0})]
        railway.station_demand(EAST).capacity = {Need.PASSENGERS: 1_000.0}

    plain_railway.calls.arrive(plain, plain.stops[1])
    express_railway.calls.arrive(express, express.stops[1])

    paid = express_railway.finances.month[Good.PASSENGERS.value]
    assert paid == pytest.approx(EXPRESS_FARE * plain_railway.finances.month[Good.PASSENGERS.value])
    assert paid > 0


def test_a_full_train_as_long_as_its_engine_is_built_for_slows_it_as_much_as_a_full_planet() -> (
    None
):
    planet = full_train("planet", MAX_CARS)

    giant = full_train("super_diesel", ENGINES["super_diesel"].cars)

    assert giant.pull == pytest.approx(planet.pull)


def test_trains_are_named_by_what_drives_them() -> None:
    railway = knowing(line(), "diesel")
    railway.year = 1935

    train = railway.buy("railbus")

    assert train is not None
    assert train.name == "Diesel 1"


def test_the_buy_menu_groups_engines_by_class_newest_first() -> None:
    names = ["planet", "tank", "american", "light_tank", "goods"]

    groups = engines_by_kind(names)

    assert groups == [
        (Kind.LIGHT, ["light_tank", "tank"]),
        (Kind.GENERAL, ["american", "planet"]),
        (Kind.HEAVY, ["goods"]),
    ]


def test_an_express_tells_it_takes_only_passengers_and_has_a_dining_car() -> None:
    summary = engine_summary(ENGINES["bullet"])

    assert "210 km/h" in summary
    assert "7 coaches and a dining car" in summary
    assert "Passengers only, paying 20% more" in summary
    assert "electrified track only" in summary


def test_a_node_that_opens_engines_says_which_and_from_when() -> None:
    _, node = NODES["express_trains"]

    text = describe(node)

    assert text == "opens express engines (from 1865)"
