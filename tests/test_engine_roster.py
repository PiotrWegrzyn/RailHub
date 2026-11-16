import itertools

import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.finance import CAR_TRIP_PRICE
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
    Engine,
    Kind,
    Stop,
    Train,
    car_size,
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


def of_its_day(engine: Engine, kind: Kind) -> Engine:
    return ENGINES[[name for name in by_year(kind) if ENGINES[name].year <= engine.year][-1]]


@pytest.mark.parametrize("name", by_year(Kind.HEAVY))
def test_a_heavy_freight_engine_pulls_twice_the_cars_of_a_general_one_of_its_day_slower(
    name: str,
) -> None:
    heavy = ENGINES[name]

    general = of_its_day(heavy, Kind.GENERAL)

    assert heavy.cars >= 2 * general.cars
    assert heavy.speed < general.speed
    assert heavy.pickup > 2 * general.pickup


@pytest.mark.parametrize("name", by_year(Kind.LIGHT))
def test_a_light_engine_costs_far_less_to_run_and_is_quicker_off_the_mark_than_a_general_one(
    name: str,
) -> None:
    light = ENGINES[name]

    general = of_its_day(light, Kind.GENERAL)

    assert light.running < general.running / 2
    assert light.pickup < general.pickup / 2
    assert light.cars < general.cars


@pytest.mark.parametrize("name", by_year(Kind.EXPRESS))
def test_an_express_runs_far_faster_than_a_general_engine_but_takes_longer_to_get_going(
    name: str,
) -> None:
    express = ENGINES[name]

    general = of_its_day(express, Kind.GENERAL)

    assert express.speed >= 1.4 * general.speed
    assert express.pickup > general.pickup
    assert express.price > general.price


def test_a_better_engine_than_the_planet_comes_within_six_years_of_an_1850_start() -> None:
    railway = line()
    railway.year = 1856

    engines = railway.engines()

    assert "american" in engines
    assert ENGINES["american"].speed > ENGINES["planet"].speed


def test_the_specialised_classes_come_soon_after_1850_and_need_their_tech() -> None:
    railway = knowing(line(), "boilers", "express_trains")
    railway.year = 1858
    without = line()
    without.year = 1858

    engines = railway.engines()

    assert {"goods", "single"} <= set(engines)
    assert not {"goods", "single"} & set(without.engines())


def test_the_game_ends_with_a_500_kmh_maglev_and_a_diesel_pulling_forty_cars() -> None:
    railway = knowing(line(), *EVERY_NODE)
    railway.year = 2040

    engines = {name: ENGINES[name] for name in railway.engines()}

    assert any(e.maglev and e.kmh == pytest.approx(500.0) for e in engines.values())
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
    railway.year = 1858
    train = railway.buy("single")
    assert train is not None

    allowed = [good for good in Good if railway.can_carry(train, good)]

    assert allowed == [Good.PASSENGERS]


def test_an_express_leads_its_coaches_with_an_empty_dining_car() -> None:
    railway = knowing(line(), "express_trains")
    railway.year = 1858
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
    railway.year = 1858
    train = railway.buy("single")
    assert train is not None
    train.stops = [Stop(WEST, [Good.PASSENGERS]), Stop(EAST)]
    train.stops[0].wait_for_load = False

    railway.run(0.001)

    assert railway.finances.month["car trips"] == pytest.approx(-2 * CAR_TRIP_PRICE)


def test_passengers_on_an_express_pay_more() -> None:
    plain_railway, express_railway = line(), knowing(line(), "express_trains")
    express_railway.year = plain_railway.year = 1858
    plain = plain_railway.buy("american")
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


def test_a_full_train_as_long_as_its_engine_is_built_for_slows_a_light_one_most_and_a_heavy_least() -> (
    None
):
    light = full_train("tank", ENGINES["tank"].cars)
    planet = full_train("planet", MAX_CARS)

    heavy = full_train("goods", ENGINES["goods"].cars)

    assert light.pull < planet.pull < heavy.pull


def test_a_full_bulk_car_weighs_three_times_as_much_as_a_full_car_of_anything_else() -> None:
    bulk = full_train("planet", 1)
    other = Train("Test", cars=[Car(Good.MAIL, Car(Good.MAIL).capacity)])

    weights = bulk.weight - 1, other.weight - 1

    assert weights == (pytest.approx(3 * FULL_LOAD), pytest.approx(FULL_LOAD))


@pytest.mark.parametrize("good", [Good.WHEAT, Good.IRON, Good.APPLES, Good.COAL, Good.LUMBER])
def test_grain_ore_apples_and_other_bulk_goods_go_three_times_as_much_to_a_car(good: Good) -> None:
    size = car_size(good)

    assert size == 3 * car_size(Good.MAIL)


def test_every_engine_costs_its_running_cost_each_month() -> None:
    railway = line()
    railway.buy("planet")
    railway.buy("tank")

    railway.turn_month({})

    running = ENGINES["planet"].running + ENGINES["tank"].running
    assert railway.finances.month["running trains"] == pytest.approx(-running)
    assert ENGINES["planet"].running >= ENGINES["planet"].price / 50


def test_steam_costs_the_most_to_run_for_its_price_and_electric_the_least() -> None:
    steam, diesel, electric = (ENGINES[name] for name in ("pacific", "mixed_diesel", "boxcab"))

    shares = [engine.running / engine.price for engine in (steam, diesel, electric)]

    assert shares == sorted(shares, reverse=True)
    assert len(set(shares)) == 3


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

    assert "240 km/h" in summary
    assert "$5,625 a month to run" in summary
    assert "7 coaches and a dining car" in summary
    assert "Passengers only, paying 20% more" in summary
    assert "electrified track only" in summary


def test_a_node_that_opens_engines_says_which_and_from_when() -> None:
    _, node = NODES["express_trains"]

    text = describe(node)

    assert text == "opens express engines (from 1858)"
