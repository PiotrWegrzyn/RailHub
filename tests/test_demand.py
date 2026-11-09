import pytest

from railroads.sim.demand import CARGO_PRICE, StationDemand, capacity_of
from railroads.sim.economy import Good


def test_demand_capacity_is_twice_what_the_buildings_need() -> None:
    capacity = capacity_of({Good.FOOD: 10.0, Good.GOODS: 0.0})

    assert capacity == {Good.FOOD: 20.0}


def test_the_first_unit_to_a_town_nobody_feeds_earns_the_full_price() -> None:
    demand = StationDemand(capacity={Good.FOOD: 20.0})

    earned = demand.deliver(Good.FOOD, 1.0)

    assert earned == pytest.approx(CARGO_PRICE)


def test_a_town_that_feeds_itself_pays_half() -> None:
    demand = StationDemand()
    for _ in range(3):
        demand.turn_month({Good.FOOD: 20.0}, {Good.FOOD: 10.0})

    earned = demand.deliver(Good.FOOD, 1.0)

    assert demand.record(Good.FOOD).capacity == 20.0
    assert earned == pytest.approx(CARGO_PRICE / 2)


def test_each_unloaded_unit_lowers_the_price_of_the_next() -> None:
    demand = StationDemand(capacity={Good.FOOD: 20.0})

    first = demand.deliver(Good.FOOD, 10.0)
    second = demand.deliver(Good.FOOD, 10.0)

    assert 0.9 * 10 * CARGO_PRICE < first < 10 * CARGO_PRICE
    assert second < first


def test_a_flooded_station_recovers_its_demand_over_three_months() -> None:
    demand = StationDemand(capacity={Good.FOOD: 20.0})
    demand.deliver(Good.FOOD, 60.0)
    shares = [demand.record(Good.FOOD).share]

    for _ in range(3):
        demand.turn_month({Good.FOOD: 20.0}, {})
        shares.append(demand.record(Good.FOOD).share)

    assert shares[0] == 0.0
    assert shares[1] == pytest.approx(0.0)
    assert shares[2] == pytest.approx(0.0)
    assert shares[3] == pytest.approx(1.0)
    assert demand.last_month[Good.FOOD].left == pytest.approx(0.0)


def test_goods_nobody_around_wants_earn_nothing() -> None:
    demand = StationDemand(capacity={Good.FOOD: 20.0})

    earned = demand.deliver(Good.COAL, 10.0)

    assert earned == 0.0
