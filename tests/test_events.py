import pytest

from railroads.maps import START_MAPS
from railroads.sim.constructions import BuildingKind
from railroads.sim.events import (
    EVENTS,
    WAR_GROWTH,
    in_eastern_bloc,
    military_growth,
    price_factor,
    rise_of,
    travel_factor,
    want_factor,
)
from railroads.sim.fuel_market import Fuel, history_factor
from railroads.sim.goods import Good, Need, boosts_of, makes_of, needs_of
from railroads.sim.happenings import changes_of

BOUNDS = {start.key: start.bounds for start in START_MAPS}


def test_people_want_a_tenth_more_goods_and_three_hundredths_more_food_each_decade() -> None:
    goods = rise_of(BuildingKind.HOUSE, Need.COMMON, 1879)
    food = rise_of(BuildingKind.HOUSE, Need.FOOD, 1879)

    assert goods == pytest.approx(1.1**2)
    assert food == pytest.approx(1.03**2)


@pytest.mark.parametrize(
    "kind", [BuildingKind.STEEL_MILL, BuildingKind.POWER_PLANT, BuildingKind.FARM]
)
def test_what_processors_power_plants_and_makers_take_does_not_rise(kind: BuildingKind) -> None:
    risen = [rise_of(kind, need, 2040) for need in Need]

    assert set(risen) == {1.0}


def test_the_telegraph_doubles_what_homes_want_of_newspapers() -> None:
    before = needs_of(BuildingKind.HOUSE, 1853)[Need.NEWSPAPERS]

    after = needs_of(BuildingKind.HOUSE, 1855)[Need.NEWSPAPERS]

    assert after == pytest.approx(2 * before)


def test_electric_light_leaves_wax_to_the_churches() -> None:
    house_before = needs_of(BuildingKind.HOUSE, 1881)

    house_after = needs_of(BuildingKind.HOUSE, 1882)
    church_after = needs_of(BuildingKind.CHURCH, 1882)

    assert Need.WAX in house_before
    assert Need.WAX not in house_after
    assert Need.WAX in church_after


def test_after_smartphones_nobody_wants_newspapers() -> None:
    wanting = [kind for kind in BuildingKind if Need.NEWSPAPERS in needs_of(kind, 2010)]

    assert wanting == []


def test_the_great_depression_cuts_beer_but_not_food() -> None:
    beer = want_factor(BuildingKind.HOUSE, Need.BEER, 1930)
    food = want_factor(BuildingKind.HOUSE, Need.FOOD, 1930)

    assert beer == pytest.approx(0.7 * rise_of(BuildingKind.HOUSE, Need.BEER, 1930))
    assert food == pytest.approx(rise_of(BuildingKind.HOUSE, Need.FOOD, 1930))


def test_the_financial_crisis_halves_food_in_homes_only() -> None:
    home = want_factor(BuildingKind.HOUSE, Need.FOOD, 2008)
    hotel = want_factor(BuildingKind.HOTEL, Need.FOOD, 2008)

    assert home == pytest.approx(0.5 * rise_of(BuildingKind.HOUSE, Need.FOOD, 2008))
    assert hotel == pytest.approx(rise_of(BuildingKind.HOTEL, Need.FOOD, 2008))


def test_the_market_opens_only_on_maps_of_the_old_eastern_bloc() -> None:
    east = want_factor(BuildingKind.FLATS, Need.CARS, 1990, east=True)
    west = want_factor(BuildingKind.FLATS, Need.CARS, 1990, east=False)

    assert east == pytest.approx(1.3 * west)
    assert in_eastern_bloc(BOUNDS["poland"])
    assert in_eastern_bloc(BOUNDS["moscow"])
    assert not in_eastern_bloc(BOUNDS["germany"])
    assert not in_eastern_bloc(BOUNDS["constantinople"])


def test_ports_take_guns_only_in_a_war_and_make_more_with_them() -> None:
    at_war = boosts_of(BuildingKind.PORT, 1916)
    at_peace = needs_of(BuildingKind.PORT, 1920)

    assert Need.GUNS in at_war
    assert Need.GUNS not in at_peace


def test_the_military_grows_fast_only_in_a_war() -> None:
    growths = [military_growth(year) for year in (1913, 1914, 1918, 1919, 1939, 1945, 1946)]

    assert growths == [1, WAR_GROWTH, WAR_GROWTH, 1, WAR_GROWTH, WAR_GROWTH, 1]


def test_the_oil_crisis_halves_the_oil_wells_and_doubles_oil_and_diesel_prices() -> None:
    before = makes_of(BuildingKind.OIL_WELL, 1972)[Good.OIL]

    during = makes_of(BuildingKind.OIL_WELL, 1974)[Good.OIL]

    assert during == pytest.approx(before / 2)
    assert price_factor(Good.OIL, 1974) == 2.0
    assert history_factor(Fuel.DIESEL, 1974) == 2.0
    assert history_factor(Fuel.DIESEL, 1975) == 1.0


def test_television_makes_electricity_dearer() -> None:
    before = history_factor(Fuel.ELECTRICITY, 1954)

    after = history_factor(Fuel.ELECTRICITY, 1955)

    assert (before, after) == (1.0, 1.25)


def test_in_the_pandemic_few_travel_and_more_mail_is_sent() -> None:
    passengers = travel_factor(Good.PASSENGERS, 2021)
    mail = travel_factor(Good.MAIL, 2021)

    assert (passengers, mail) == (pytest.approx(0.2), 1.5)


def test_a_war_brings_half_again_as_many_passengers() -> None:
    passengers = travel_factor(Good.PASSENGERS, 1940)

    assert passengers == 1.5


def test_the_news_of_an_event_says_what_it_changes() -> None:
    telegraph = next(event for event in EVENTS if event.year == 1854)

    lines = changes_of(telegraph)

    assert lines == ("Buildings want +100% newspapers",)


def test_events_come_in_the_order_of_their_dates() -> None:
    dates = [(event.year, event.month) for event in EVENTS]

    assert dates == sorted(dates)
