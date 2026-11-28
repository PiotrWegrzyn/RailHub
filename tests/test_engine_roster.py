import itertools

import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.finance import CAR_TRIP_PRICE, TRAIN_RESALE, car_trip_price
from railroads.sim.goods import GENERAL_GOODS, General, Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.tiles import Tiles, activate, of_year
from railroads.sim.trains import (
    BASE_GENERATION,
    BULK_CAR,
    CAR_SIZES,
    ENGINES,
    FREIGHT_TARE,
    GENERATIONS,
    OPENED_BY,
    SLOWDOWN,
    Car,
    Engine,
    Kind,
    Stop,
    Train,
    amount_of,
    bigger,
    car_size,
    full_tonnes,
    generation_in,
    tare_of,
    unit_tonnes,
)
from railroads.ui.train_texts import (
    adding_tip,
    engine_summary,
    engines_by_kind,
    picker_good,
    wagons_note,
)
from railroads.words import tonnes_text

WEST, EAST = (6, 3), (6, 20)


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


def by_year(kind: Kind) -> list[str]:
    return sorted((n for n, e in ENGINES.items() if e.kind == kind), key=lambda n: ENGINES[n].year)


def full_train(engine: str, cars: int) -> Train:
    train = Train("Test", engine=engine)
    newest = ENGINES[engine].top_generation
    size, tare = car_size(Good.COAL, newest), tare_of(Good.COAL, newest)
    train.cars = [Car(Good.COAL, size, capacity=size, tare=tare) for _ in range(cars)]
    return train


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
    tank, planet = ENGINES["tank"], ENGINES["planet"]

    assert sorted(engines) == ["planet", "tank"]
    assert (tank.cars, planet.cars) == (2, 3)
    assert tank.price < planet.price == 10_000.0
    assert tank.speed < planet.speed
    assert planet.kmh == pytest.approx(30.0)


def test_every_engine_before_the_planet_is_worse_and_cheaper() -> None:
    planet = ENGINES["planet"]

    pioneers = [ENGINES[name] for name in by_year(Kind.PIONEER)]

    assert all(e.year < planet.year for e in pioneers)
    assert all(e.cars < planet.cars and e.speed < planet.speed for e in pioneers)
    assert all(e.price < planet.price for e in pioneers)
    assert [e.cars for e in pioneers] == sorted(e.cars for e in pioneers)


@pytest.mark.parametrize("kind", [Kind.LIGHT, Kind.GENERAL, Kind.HEAVY, Kind.EXPRESS])
def test_each_class_only_gets_better_over_time_on_the_same_power(kind: Kind) -> None:
    engines = [ENGINES[name] for name in by_year(kind)]

    pairs = [
        (old, new) for old, new in itertools.combinations(engines, 2) if old.power == new.power
    ]

    assert len(engines) >= 4
    assert all(new.speed >= old.speed and new.cars >= old.cars for old, new in pairs)
    assert all(new.speed > old.speed or new.cars > old.cars for old, new in pairs)


def test_an_engine_pulls_the_cars_of_the_generation_after_its_years() -> None:
    steam, diesel, electric = ENGINES["planet"], ENGINES["mixed_diesel"], ENGINES["express"]

    generations = steam.top_generation, diesel.top_generation, electric.top_generation

    assert electric.speed > diesel.speed
    assert generations == (2, 6, 7)


def of_its_day(engine: Engine, kind: Kind) -> Engine:
    names = by_year(kind)
    older = [name for name in names if ENGINES[name].year <= engine.year]
    return ENGINES[older[-1] if older else names[0]]


@pytest.mark.parametrize("name", by_year(Kind.HEAVY))
def test_a_heavy_freight_engine_pulls_more_cars_than_a_general_one_of_its_day_slower(
    name: str,
) -> None:
    heavy = ENGINES[name]

    general = of_its_day(heavy, Kind.GENERAL)

    assert heavy.cars > general.cars
    assert heavy.speed < general.speed
    assert heavy.pickup > 2 * general.pickup


@pytest.mark.parametrize("name", by_year(Kind.LIGHT))
def test_a_light_engine_costs_far_less_to_run_and_is_quicker_off_the_mark_than_a_general_one(
    name: str,
) -> None:
    light = ENGINES[name]

    general = of_its_day(light, Kind.GENERAL)

    assert light.burn < general.burn
    assert light.pickup < general.pickup / 2
    assert light.cars < general.cars


@pytest.mark.parametrize("name", by_year(Kind.EXPRESS))
def test_an_express_runs_far_faster_than_a_general_engine_and_gets_going_about_as_quickly(
    name: str,
) -> None:
    express = ENGINES[name]

    general = of_its_day(express, Kind.GENERAL)

    assert express.speed >= 1.4 * general.speed
    assert express.pickup <= 1.5 * general.pickup
    assert express.price > general.price


def test_the_specialised_classes_come_within_fifteen_years_of_1850() -> None:
    early, later = line_at(1850), line_at(1865)

    engines = set(early.engines()), set(later.engines())

    assert not {"goods", "single"} & engines[0]
    assert {"goods", "single"} <= engines[1]


def test_the_game_ends_with_a_500_kmh_maglev_and_a_diesel_pulling_twelve_of_the_biggest_cars() -> (
    None
):
    railway = line()
    railway.year = 2040

    engines = {name: ENGINES[name] for name in railway.engines()}

    assert any(e.maglev and e.kmh == pytest.approx(500.0) for e in engines.values())
    assert any(
        e.power.value == "diesel" and e.cars == 12 and e.top_generation == GENERATIONS
        for e in engines.values()
    )


def test_a_newer_engine_of_the_same_line_takes_the_older_ones_place_on_sale() -> None:
    railway = line()
    railway.year = 1870

    engines = railway.engines()

    assert "american" in engines
    assert "planet" not in engines


def test_an_electric_does_not_end_the_sale_of_the_engine_it_would_replace() -> None:
    railway = line()
    railway.year = opened_in("boxcab")

    engines = railway.engines()

    assert {"pacific", "boxcab"} <= set(engines)


def test_every_engine_is_opened_by_a_demand_tile_level() -> None:
    opened = set(OPENED_BY)

    engines = set(ENGINES)

    assert opened == engines


def test_an_engine_goes_on_sale_once_its_demand_tile_reaches_its_level() -> None:
    railway = line()
    tile, level = OPENED_BY["mogul"]
    railway.year = next(
        year for year in range(1804, 2050) if of_year(year).demand.get(tile, 0) >= level
    )

    sale = "mogul" in line_at(railway.year - 1).engines(), "mogul" in railway.engines()

    assert sale == (False, True)


def test_an_express_carries_only_passengers_and_mail() -> None:
    railway = line()
    railway.year = opened_in("single")
    train = railway.buy("single")
    assert train is not None

    allowed = [good for good in Good if railway.can_carry(train, good)]

    assert allowed == [Good.PASSENGERS, Good.MAIL]


def test_an_express_leads_its_coaches_with_an_empty_dining_car() -> None:
    railway = line()
    railway.year = opened_in("single")
    train = railway.buy("single")
    assert train is not None
    train.stops = [Stop(WEST, [Good.PASSENGERS, Good.COAL]), Stop(EAST)]
    railway.station_goods(WEST).stockpile[Good.PASSENGERS] = 1_000.0

    railway.calls.arrive(train, train.stops[0])

    dining, *coaches = train.cars
    assert dining.dining and dining.load == 0.0
    assert [car.good for car in coaches] == [Good.PASSENGERS]
    assert coaches[0].load > 0
    assert dining.tonnes == pytest.approx(railway.car_tare(Good.PASSENGERS, "single"))


def test_an_express_pays_a_car_trip_for_its_dining_car() -> None:
    railway = line()
    railway.year = opened_in("single")
    train = railway.buy("single")
    assert train is not None
    train.stops = [Stop(WEST, [Good.PASSENGERS]), Stop(EAST)]
    train.stops[0].wait_for_load = False

    railway.run(0.001)

    coach = car_trip_price(railway.car_capacity(Good.PASSENGERS, "single"))
    assert railway.finances.month["car trips"] == pytest.approx(-(CAR_TRIP_PRICE + coach))


def test_passengers_on_an_express_pay_more() -> None:
    plain_railway, express_railway = line(), line()
    express_railway.year = plain_railway.year = max(opened_in("american"), opened_in("single"))
    plain = plain_railway.buy("american")
    express = express_railway.buy("single")
    assert plain is not None
    assert express is not None
    for railway, train in ((plain_railway, plain), (express_railway, express)):
        train.stops = [Stop(WEST, [Good.PASSENGERS]), Stop(EAST)]
        train.loaded_at = WEST
        train.cars = [Car(Good.PASSENGERS, 40.0, {WEST: 40.0})]
        railway.station_demand(EAST).capacity = {Need.PASSENGERS: 1_000.0}

    plain_railway.calls.arrive(plain, plain.stops[1])
    express_railway.calls.arrive(express, express.stops[1])

    paid = express_railway.finances.month[Good.PASSENGERS.value]
    plain_paid = plain_railway.finances.month[Good.PASSENGERS.value]
    assert paid == pytest.approx(ENGINES["single"].fare * plain_paid)
    assert paid > 0


def test_mail_on_an_express_pays_as_on_any_train() -> None:
    plain_railway, express_railway = line(), line()
    express_railway.year = plain_railway.year = max(opened_in("american"), opened_in("single"))
    plain = plain_railway.buy("american")
    express = express_railway.buy("single")
    assert plain is not None
    assert express is not None
    for railway, train in ((plain_railway, plain), (express_railway, express)):
        train.stops = [Stop(WEST, [Good.MAIL]), Stop(EAST)]
        train.loaded_at = WEST
        train.cars = [Car(Good.MAIL, 40.0, {WEST: 40.0})]
        railway.station_demand(EAST).capacity = {Need.MAIL: 1_000.0}

    plain_railway.calls.arrive(plain, plain.stops[1])
    express_railway.calls.arrive(express, express.stops[1])

    paid = express_railway.finances.month[Good.MAIL.value]
    assert paid == pytest.approx(plain_railway.finances.month[Good.MAIL.value])
    assert paid > 0


def test_a_train_at_its_most_slows_a_light_engine_most_and_a_heavy_one_least() -> None:
    railway = line()
    light = full_train("tank", ENGINES["tank"].cars)
    planet = full_train("planet", ENGINES["planet"].cars)
    heavy = full_train("goods", ENGINES["goods"].cars)

    pulls = [railway.pull(train) for train in (light, planet, heavy)]

    assert pulls[0] < pulls[1] < pulls[2]


def test_a_general_engine_at_its_most_runs_at_half_speed_and_a_heavy_one_at_seventy_percent() -> (
    None
):
    railway = line()
    planet = full_train("planet", ENGINES["planet"].cars)
    heavy = full_train("goods", ENGINES["goods"].cars)

    pulls = railway.pull(planet), railway.pull(heavy)

    assert pulls == (pytest.approx(0.5, abs=0.02), pytest.approx(0.7, abs=0.02))


def test_a_train_slows_by_what_its_cars_weigh_against_the_most_its_engine_pulls() -> None:
    railway = line()
    train = Train("Test", cars=[Car(Good.WHEAT, BULK_CAR)])
    share = train.tonnes / railway.max_pull(train.engine)

    pull = railway.pull(train)

    assert pull == pytest.approx(1.0 - SLOWDOWN[Kind.GENERAL] * share)


def test_full_cars_of_light_or_bulky_goods_weigh_less_than_one_of_coal() -> None:
    light = [Good.FISH, Good.GOLD, Good.APPLES, Good.CATTLE, Good.CLOTHES, Good.FURNITURE]

    weights = [full_tonnes(good, car_size(good)) for good in light]

    assert max(weights) < full_tonnes(Good.COAL, BULK_CAR)


def test_freight_cars_weigh_the_same_empty_whatever_they_carry_and_a_coach_more() -> None:
    goods = [Good.GOLD, Good.STEEL, Good.FISH, Good.COAL]

    tares = [Car(good).tonnes for good in goods]

    assert tares == [pytest.approx(FREIGHT_TARE)] * len(goods)
    assert Car(Good.PASSENGERS).tonnes > FREIGHT_TARE


@pytest.mark.parametrize("good", [Good.COAL, Good.STEEL, Good.MAIL, Good.PASSENGERS])
def test_a_car_weighs_its_tare_and_what_it_carries(good: Good) -> None:
    car = Car(good, 10.0)

    weight = car.tonnes

    assert weight == pytest.approx(tare_of(good) + 10.0 * unit_tonnes(good))


@pytest.mark.parametrize(
    ("generation", "coal"), [(1, 55.0), (2, 89.0), (3, 144.0), (5, 377.0), (7, 987.0)]
)
def test_each_generation_of_cars_holds_one_fibonacci_size_more(
    generation: int, coal: float
) -> None:
    size = car_size(Good.COAL, generation)

    assert size == coal


def test_cars_ride_from_the_fourth_generation_and_tanks_and_planes_from_the_fifth() -> None:
    goods = [Good.CARS, Good.TANKS, Good.PLANES]

    sizes = [[car_size(good, generation) for generation in (3, 4, 5, 6, 7)] for good in goods]

    assert sizes == [
        [0.0, 3.0, 5.0, 8.0, 13.0],
        [0.0, 0.0, 1.0, 1.0, 2.0],
        [0.0, 0.0, 1.0, 1.0, 2.0],
    ]


def test_pioneers_pull_the_first_generation_and_the_planet_the_second_without_tech() -> None:
    railway = line()

    generations = (
        railway.car_generation(Good.COAL, "rocket"),
        railway.car_generation(Good.COAL, "planet"),
    )

    assert generations == (1, BASE_GENERATION)


@pytest.mark.parametrize(
    ("year", "generation"), [(1842, 1), (1843, 2), (1873, 3), (1931, 5), (1972, 7)]
)
def test_a_new_generation_of_cars_comes_in_its_year(year: int, generation: int) -> None:
    newest = generation_in(year)

    assert newest == generation


def test_cars_are_of_the_eras_generation_up_to_the_newest_an_engine_pulls() -> None:
    railway = line()
    railway.year = 1980

    generations = [
        railway.car_generation(Good.COAL, engine) for engine in ("planet", "mogul", "heavy_diesel")
    ]

    assert generations == [2, 3, GENERATIONS]


def test_before_the_second_era_even_the_planet_pulls_the_first_generation() -> None:
    railway = line()
    activate(Tiles())

    generation = railway.car_generation(Good.COAL, "planet")

    assert generation == 1


def test_a_pioneer_pulls_a_full_car_of_55_quintals_of_coal() -> None:
    railway = line()
    railway.year = opened_in("penydarren")
    train = railway.buy("penydarren")
    assert train is not None
    stop = Stop(WEST)

    added = railway.add_car(train, stop, Good.COAL)

    assert added
    assert railway.car_capacity(Good.COAL, "penydarren") == 55.0


@pytest.mark.parametrize("engine", ["american", "mogul"])
def test_a_freight_engine_pulls_a_full_train_of_coal_at_its_slowest(engine: str) -> None:
    railway = line()
    train = full_train(engine, ENGINES[engine].cars)

    share = railway.load_share(train)

    assert share == pytest.approx(1.0, abs=0.01)


def test_an_express_slows_about_as_much_empty_as_full() -> None:
    railway = line()
    railway.year = 1858
    coaches = ENGINES["single"].cars
    empty = Train("Empty", engine="single", cars=[Car(Good.PASSENGERS, dining=True)])
    empty.cars += [Car(Good.PASSENGERS) for _ in range(coaches)]
    full = Train("Full", engine="single", cars=[Car(Good.PASSENGERS, dining=True)])
    full.cars += [Car(Good.PASSENGERS, car_size(Good.PASSENGERS)) for _ in range(coaches)]

    pulls = railway.pull(empty), railway.pull(full)

    assert pulls[1] < pulls[0] <= pulls[1] * 1.15


@pytest.mark.parametrize(
    ("engine", "expected"), [("rocket", [True, False]), ("goods", [True, True, True, True])]
)
def test_an_engine_pulls_only_as_many_full_coal_cars_as_its_strength_allows(
    engine: str, expected: list[bool]
) -> None:
    railway = line()
    railway.year = opened_in(engine)
    train = railway.buy(engine)
    assert train is not None
    stop = Stop(WEST)

    added = [railway.add_car(train, stop, Good.COAL) for _ in expected]

    assert added == expected


def test_a_railbus_pulls_its_coaches_full_but_not_as_many_coal_cars_of_its_newest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    railway = line()
    railway.year = opened_in("railbus")
    monkeypatch.setattr(railway, "wagon_era", lambda: ENGINES["railbus"].top_generation)
    train = railway.buy("railbus")
    assert train is not None
    coaches, coal = Stop(WEST), Stop(EAST)

    coaches_added = [railway.add_car(train, coaches, Good.PASSENGERS) for _ in range(3)]
    coal_added = [railway.add_car(train, coal, Good.COAL) for _ in range(3)]

    assert coaches_added == [True, True, True]
    assert coal_added == [True, True, False]


def test_a_general_car_weighs_as_much_as_the_heaviest_of_its_goods_would() -> None:
    railway = line()
    heaviest = max(
        full_tonnes(good, railway.car_capacity(good, "planet", General.FOOD))
        for good in GENERAL_GOODS[General.FOOD]
    )

    weight = railway.car_tonnes(General.FOOD, "planet")

    assert weight == pytest.approx(heaviest)


def test_an_engine_pulls_the_same_whatever_era_its_cars_come_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    railway = line()
    monkeypatch.setattr(railway, "wagon_era", lambda: 2)
    early = railway.max_pull("light_tank")
    monkeypatch.setattr(railway, "wagon_era", lambda: 3)

    late = railway.max_pull("light_tank")

    assert late == early == ENGINES["light_tank"].pull


def test_cars_load_no_more_than_their_engine_pulls() -> None:
    railway = line()
    railway.year = opened_in("rocket")
    train = railway.buy("rocket")
    assert train is not None
    stop = Stop(WEST, [Good.COAL, Good.COAL])
    size, tare = railway.car_capacity(Good.COAL, "rocket"), railway.car_tare(Good.COAL, "rocket")
    train.cars = [Car(Good.COAL, capacity=size, tare=tare) for _ in range(2)]
    railway.station_goods(WEST).stockpile[Good.COAL] = 1_000.0

    railway.calls.load(train, stop)

    assert train.tonnes == pytest.approx(railway.max_pull(train.engine))
    assert 0 < train.cars[0].load < size


def test_each_car_size_holds_as_much_as_the_two_below_it_together() -> None:
    sums = [small + middle for small, middle in itertools.pairwise(CAR_SIZES)]

    assert sums[:-1] == list(CAR_SIZES[2:])


def test_a_car_already_the_biggest_size_grows_no_more() -> None:
    size = bigger(CAR_SIZES[-2], 3)

    assert size == CAR_SIZES[-1]


def test_electric_engines_burn_less_a_car_than_steam_and_light_engines_half_as_much() -> None:
    planet, boxcab, tank = (ENGINES[name] for name in ("planet", "boxcab", "tank"))

    per_car = [engine.burn / engine.hauls for engine in (planet, boxcab, tank)]

    assert per_car[1] < per_car[0]
    assert per_car[2] == pytest.approx(per_car[0] / 2)


def test_trains_are_named_by_what_drives_them() -> None:
    railway = line()
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
    assert "Burns 1.1 electricity a cell" in summary
    assert "As far as the wires go" in summary
    assert "6 coaches and a dining car" in summary
    assert "Passengers and mail only, passengers paying 2 times the fare" in summary
    assert "electrified track only" in summary


def planets_on_the_line(railway: Railway, count: int) -> list[Train]:
    railway.year = 1850
    trains = [railway.buy("planet") for _ in range(count)]
    for train in trains:
        assert train is not None
        train.stops = [Stop(WEST, [Good.COAL, Good.MAIL]), Stop(EAST, [Good.WHEAT])]
    railway.year = opened_in("american")
    return [train for train in trains if train is not None]


def test_a_train_can_swap_its_engine_only_for_a_newer_one_of_its_line_on_sale() -> None:
    railway = line()
    (train,) = planets_on_the_line(railway, 1)

    options = railway.engine_upgrades(train)

    assert options == ["american"]


def test_an_engine_swap_costs_the_new_engine_less_half_the_old_and_keeps_the_route() -> None:
    railway = line()
    (train,) = planets_on_the_line(railway, 1)
    stops = [(stop.cell, list(stop.cars)) for stop in train.stops]
    cash = railway.finances.cash
    price = ENGINES["american"].price - ENGINES["planet"].price * TRAIN_RESALE

    swapped = railway.swap_engine(train, "american")

    assert swapped
    assert train.engine == "american"
    assert train.name == "Steam engine 1"
    assert [
        (stop.cell, stop.cars[: len(cars)])
        for stop, (_, cars) in zip(train.stops, stops, strict=True)
    ] == stops
    assert railway.finances.cash == pytest.approx(cash - price)


def test_an_engine_swap_adds_more_of_each_stops_last_car_while_the_new_engine_takes_them() -> None:
    railway = line()
    (train,) = planets_on_the_line(railway, 1)
    before = [list(stop.cars) for stop in train.stops]

    railway.swap_engine(train, "american")

    after = [list(stop.cars) for stop in train.stops]
    assert all(cars[: len(old)] == old for cars, old in zip(after, before, strict=True))
    assert all(set(cars[len(old) :]) <= {old[-1]} for cars, old in zip(after, before, strict=True))
    assert sum(map(len, after)) > sum(map(len, before))
    for stop in train.stops:
        assert not railway.add_car(train, stop, stop.cars[-1])


def test_no_engine_swap_without_the_cash_or_for_an_engine_not_on_offer() -> None:
    railway = line()
    (train,) = planets_on_the_line(railway, 1)
    railway.finances.cash = railway.swap_price(train, "american") - 1

    poor = railway.swap_engine(train, "american")
    railway.finances.cash = 10_000_000.0
    older = railway.swap_engine(train, "tank")

    assert not poor and not older
    assert train.engine == "planet"


def test_a_mass_upgrade_swaps_every_train_of_the_category_with_an_older_engine_only() -> None:
    railway = line()
    railway.year = opened_in("lion")
    lion = railway.buy("lion")
    planets = planets_on_the_line(railway, 2)
    railway.year = opened_in("tank")
    tank = railway.buy("tank")
    railway.year = opened_in("american")

    swapped = railway.swap_all_engines("american")

    assert swapped == 3
    assert lion is not None and lion.engine == "american"
    assert [train.engine for train in planets] == ["american", "american"]
    assert tank is not None and tank.engine == "tank"


def test_a_mass_upgrade_goes_as_far_as_the_cash_in_the_order_the_trains_were_bought() -> None:
    railway = line()
    planets = planets_on_the_line(railway, 3)
    railway.finances.cash = 2.5 * railway.swap_price(planets[0], "american")

    swapped = railway.swap_all_engines("american")

    assert swapped == 2
    assert [train.engine for train in planets] == ["american", "american", "planet"]


def test_an_engines_summary_says_the_most_it_pulls_in_tonnes() -> None:
    engine = ENGINES["planet"]

    summary = engine_summary(engine)

    assert f"Pulls at most {tonnes_text(engine.tonnes)} t" in summary


def test_a_good_too_heavy_for_the_engine_says_so_instead_of_offering_a_car() -> None:
    tip = adding_tip(False, 98.0, 52.0)

    assert "Too heavy" in tip
    assert "Click to add" not in tip


@pytest.mark.parametrize(
    ("good", "amount", "told"),
    [
        (Good.COAL, 89.0, "89 sacks of coal"),
        (Good.CATTLE, 1.0, "1 head of cattle"),
        (Good.PASSENGERS, 34.0, "34 passengers"),
        (Good.CLOTHES, 21.0, "21 crates of clothes"),
    ],
)
def test_an_amount_of_a_good_is_told_in_its_units(good: Good, amount: float, told: str) -> None:
    text = amount_of(good, amount)

    assert text == told


def test_a_stop_can_ask_for_an_older_smaller_and_lighter_car() -> None:
    railway = line()
    railway.year = 1850
    train = railway.buy()
    assert train is not None
    train.stops = [Stop(WEST), Stop(EAST)]
    railway.add_car(train, train.stops[0], Good.COAL, 1)

    railway.calls.arrive(train, train.stops[0])

    (car,) = train.cars
    assert (car.capacity, car.tare, car.generation) == (
        car_size(Good.COAL, 1),
        pytest.approx(1.5),
        1,
    )


def test_older_cars_weigh_less_full_than_the_newest() -> None:
    railway = line()
    railway.year = 1850
    train = railway.buy()
    assert train is not None
    newest, older = Stop(WEST), Stop(EAST)
    railway.add_car(train, newest, Good.COAL)
    railway.add_car(train, older, Good.COAL, 1)

    weights = railway.stop_tonnes(train, newest), railway.stop_tonnes(train, older)

    assert weights[1] < weights[0]


def test_a_stop_cannot_ask_for_vehicles_in_a_generation_before_theirs() -> None:
    railway = line()
    railway.year = opened_in("heavy_diesel")
    railway.finances.earn(10_000_000.0, "test")
    train = railway.buy("heavy_diesel")
    assert train is not None
    stop = Stop(WEST)

    added = railway.add_car(train, stop, Good.CARS, 3), railway.add_car(train, stop, Good.CARS)

    assert added == (False, True)


def test_removing_a_car_from_a_stop_removes_the_generation_asked_for_it() -> None:
    stop = Stop(WEST)
    stop.add_car(Good.COAL, generation=1)
    stop.add_car(Good.WHEAT)

    stop.remove_car(0)

    assert stop.asked() == [(Good.WHEAT, 0)]


def test_a_train_swaps_a_newer_car_for_the_older_one_its_stop_now_asks_for() -> None:
    railway = line()
    railway.year = 1850
    train = railway.buy()
    assert train is not None
    train.stops = [Stop(WEST, [Good.COAL]), Stop(EAST)]
    railway.calls.arrive(train, train.stops[0])
    train.stops[0].clear_cars()
    train.stops[0].add_car(Good.COAL, generation=1)

    railway.calls.arrive(train, train.stops[0])

    assert [car.capacity for car in train.cars] == [car_size(Good.COAL, 1)]


def test_the_generation_picker_lists_coaches_behind_an_express_and_coal_cars_otherwise() -> None:
    express, freight = Train("Express", engine="single"), Train("Freight", engine="goods")

    goods = picker_good(express), picker_good(freight)

    assert goods == (Good.PASSENGERS, Good.COAL)


def line_at(year: int) -> Railway:
    railway = line()
    railway.year = year
    return railway


def opened_in(name: str) -> int:
    tile, level = OPENED_BY[name]
    return next(year for year in range(1804, 2051) if of_year(year).demand.get(tile, 0) >= level)


def test_a_train_s_cars_grow_at_its_next_stop_once_newer_wagons_come(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    railway = line()
    (train,) = planets_on_the_line(railway, 1)
    railway.swap_engine(train, "american")
    train.stops = [Stop(WEST, [Good.COAL]), Stop(EAST, [Good.COAL])]
    for _ in range(20_000):
        railway.run(0.001)
        if train.arrivals:
            break
    before = {car.good: car.capacity for car in train.cars}
    monkeypatch.setattr(railway, "wagon_era", lambda: 3)
    arrivals = train.arrivals

    for _ in range(20_000):
        railway.run(0.001)
        if train.arrivals > arrivals:
            break

    after = {car.good: car.capacity for car in train.cars}
    assert ENGINES["american"].top_generation >= 3
    assert after == {good: car_size(good, 3) for good in after}
    assert after[Good.COAL] > before[Good.COAL]


def test_an_upgrade_tells_its_cars_stay_today_s_size_until_newer_wagons_come(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    railway = line()
    monkeypatch.setattr(railway, "wagon_era", lambda: 2)

    waiting = wagons_note(railway, "light_tank")
    monkeypatch.setattr(railway, "wagon_era", lambda: 3)
    come = wagons_note(railway, "light_tank")

    assert ENGINES["light_tank"].top_generation == 3
    assert waiting == [
        "Its cars stay generation 2, the newest there are, until a new era brings generation 3"
    ]
    assert come == []
