import pytest

from railroads.sim.demand import PASSED_ALONG_PAY, StationDemand, capacity_of, price_of
from railroads.sim.goods import Good, Need


def test_demand_capacity_is_twice_what_the_buildings_need() -> None:
    capacity = capacity_of({Need.FOOD: 10.0, Need.GOODS: 0.0})

    assert capacity == {Need.FOOD: 20.0}


def test_the_first_unit_to_a_town_nobody_feeds_earns_the_full_price() -> None:
    demand = StationDemand(capacity={Need.FOOD: 20.0})

    earned = demand.deliver(Good.WHEAT, 1.0)

    assert earned == pytest.approx(price_of(Good.WHEAT))


def test_goods_passed_along_earn_a_share_of_a_delivery_and_use_up_as_much_demand() -> None:
    delivered = StationDemand(capacity={Need.FOOD: 20.0})
    passed = StationDemand(capacity={Need.FOOD: 20.0})

    full = delivered.pay(Need.FOOD, Good.WHEAT, 5.0)
    share = passed.pay(Need.FOOD, Good.WHEAT, 5.0, PASSED_ALONG_PAY)

    assert share == pytest.approx(PASSED_ALONG_PAY * full)
    assert passed.record(Need.FOOD).left == delivered.record(Need.FOOD).left


def test_each_unloaded_unit_lowers_the_price_of_the_next() -> None:
    demand = StationDemand(capacity={Need.FOOD: 20.0})

    first = demand.deliver(Good.WHEAT, 10.0)
    second = demand.deliver(Good.WHEAT, 10.0)

    assert 0.9 * 10 * price_of(Good.WHEAT) < first < 10 * price_of(Good.WHEAT)
    assert second < first


def test_a_flooded_station_recovers_its_demand_over_three_months() -> None:
    demand = StationDemand(capacity={Need.FOOD: 20.0})
    demand.deliver(Good.WHEAT, 60.0)
    shares = [demand.record(Need.FOOD).share]

    for _ in range(3):
        demand.turn_month({Need.FOOD: 20.0})
        shares.append(demand.record(Need.FOOD).share)

    assert shares[0] == 0.0
    assert shares[1] == pytest.approx(0.0)
    assert shares[2] == pytest.approx(0.0)
    assert shares[3] == pytest.approx(1.0)
    assert demand.last_month[Need.FOOD].left == pytest.approx(0.0)


def test_goods_nobody_around_wants_earn_nothing() -> None:
    demand = StationDemand(capacity={Need.FOOD: 20.0})

    earned = demand.deliver(Good.COAL, 10.0)

    assert earned == 0.0


def test_apples_sell_as_food_or_fruit_whichever_is_wanted_more() -> None:
    demand = StationDemand(capacity={Need.FOOD: 20.0, Need.FRUIT: 4.0})

    demand.deliver(Good.APPLES, 2.0)

    assert demand.used[Need.FOOD] == [1.0]
    assert demand.used[Need.FRUIT] == [1.0]


def test_wheat_never_counts_as_fruit() -> None:
    demand = StationDemand(capacity={Need.FRUIT: 20.0})

    earned = demand.deliver(Good.WHEAT, 5.0)

    assert earned == 0.0
