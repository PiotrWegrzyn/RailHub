import numpy as np
import pytest

from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Trade
from railroads.sim.events import MAKE_RISE, RISE, SLOW_MAKE_RISE, SLOW_RISE, make_factor
from railroads.sim.goods import Good, Need
from railroads.sim.goods_market import (
    ELASTICITY,
    FIRST_MONTHS,
    FOLLOW,
    LEAST_FACTOR,
    MOST_FACTOR,
    GoodsMarket,
    NeedMonth,
    signals_of,
)


def _trade(wanted: float, made: float) -> Trade:
    return Trade(
        np.array([0]),
        {},
        {},
        {},
        {},
        goods_got={(BuildingKind.HOUSE, Need.FOOD): (np.array([0.0]), np.array([wanted]))},
        goods_gave={(BuildingKind.FARM, Good.WHEAT): (np.array([0.0]), np.array([made]))},
    )


def test_a_need_is_wanted_by_the_maps_buildings_and_supplied_by_the_makers_of_its_goods() -> None:
    trade = _trade(300.0, 500.0)

    signals = signals_of(trade)

    assert signals[Need.FOOD] == (300.0, 500.0)


def test_goods_earn_their_price_in_the_first_year_of_play_whatever_the_market() -> None:
    market = GoodsMarket()

    for wanted in range(FIRST_MONTHS):
        market.month(1850, _trade(100.0 * (wanted + 1), 500.0))

    assert market.factor_of(Good.WHEAT) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("wanted", "made", "target"),
    [
        (300.0, 2000.0, 0.25**ELASTICITY),
        (1200.0, 500.0, 4.0**ELASTICITY),
        (300.0, 100_000.0, LEAST_FACTOR),
        (60_000.0, 500.0, MOST_FACTOR),
    ],
)
def test_a_good_heads_for_its_price_times_how_much_harder_demand_pulls_than_at_first(
    wanted: float, made: float, target: float
) -> None:
    market = GoodsMarket()
    for _ in range(FIRST_MONTHS):
        market.month(1850, _trade(300.0, 500.0))

    market.month(1851, _trade(wanted, made))

    assert market.factor_of(Good.WHEAT) == pytest.approx(1.0 + FOLLOW * (target - 1.0))


def test_a_good_no_one_wants_earns_its_price() -> None:
    market = GoodsMarket()
    market.month(1850, _trade(300.0, 500.0))

    factor = market.factor_of(Good.STEEL)

    assert factor == 1.0


def test_passengers_have_no_market() -> None:
    market = GoodsMarket()
    for _ in range(FIRST_MONTHS):
        market.month(1850, _trade(300.0, 500.0))
    market.month(1851, _trade(300.0, 100_000.0))

    factor = market.factor_of(Good.PASSENGERS)

    assert factor == 1.0


@pytest.mark.parametrize(
    ("good", "rise"),
    [(Good.IRON, MAKE_RISE), (Good.WHEAT, SLOW_MAKE_RISE), (Good.COAL, SLOW_MAKE_RISE)],
)
def test_makers_make_more_every_decade_and_less_more_of_food_and_heating(
    good: Good, rise: float
) -> None:
    year = 1873

    factor = make_factor(good, year)

    assert factor == pytest.approx((1.0 + rise) ** 2)


@pytest.mark.parametrize(
    ("good", "rise"), [(Good.STEEL, RISE), (Good.WHEAT, SLOW_RISE), (Good.COAL, SLOW_RISE)]
)
def test_a_unit_earns_as_much_less_every_decade_as_people_want_more(
    good: Good, rise: float
) -> None:
    market = GoodsMarket()

    market.month(1873, _trade(300.0, 500.0))

    assert market.factor_of(good) == pytest.approx((1.0 + rise) ** -2)


def test_passengers_and_mail_earn_as_much_whatever_the_decade() -> None:
    market = GoodsMarket()

    market.month(2040, _trade(300.0, 500.0))

    assert market.factor_of(Good.MAIL) == 1.0


def test_a_months_price_is_its_market_factor_as_much_less_as_its_decade_makes_it() -> None:
    month = NeedMonth(Need.COMMON, 1871, 100.0, 50.0, 0.8)

    price = month.price

    assert price == pytest.approx(0.8 / (1.0 + RISE) ** 2)


def test_a_need_the_map_makes_none_of_keeps_its_price_until_it_does_and_is_measured_from_then() -> (
    None
):
    market = GoodsMarket()
    for _ in range(FIRST_MONTHS):
        market.month(1850, _trade(300.0, 0.0))
    for _ in range(FIRST_MONTHS):
        market.month(1851, _trade(300.0, 500.0))

    market.month(1854, _trade(300.0, 500.0))

    assert market.factor_of(Good.WHEAT) == pytest.approx(1.0)
