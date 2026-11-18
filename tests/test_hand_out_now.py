import numpy as np
import pytest
from test_railway import joined_stations

from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.economy import Satisfaction
from railroads.sim.goods import Good, Need, needs_of

STATION = (6, 20)


def bring(railway, good: Good, amount: float) -> None:  # type: ignore[no-untyped-def]
    railway.station_goods(STATION).bring(good, {(6, 3): amount})


def test_houses_by_a_station_eat_food_the_moment_it_arrives_up_to_a_months_need() -> None:
    growth, railway = joined_stations()
    for _ in range(2):
        growth.build(Building((6, 21), BuildingKind.HOUSE))
    growth.station_trade.measure_demand(STATION)
    bring(railway, Good.BREAD, 50.0)

    growth.station_trade.hand_out_now(STATION, Good.BREAD)

    assert railway.station_goods(STATION).brought[Good.BREAD] == pytest.approx(30.0)
    assert railway.finances.month["cargo"] > 0


def test_food_taken_at_once_is_not_handed_out_again_and_counts_for_the_month() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    bring(railway, Good.BREAD, 50.0)
    growth.station_trade.hand_out_now(STATION, Good.BREAD)

    growth.trade_goods()

    assert growth.trade is not None
    eaten, _ = growth.trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]
    assert float(eaten.sum()) == pytest.approx(10.0)
    assert railway.station_goods(STATION).brought[Good.BREAD] == pytest.approx(40.0)


def test_freight_a_station_gathered_is_never_handed_out_at_once() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    goods = railway.station_goods(STATION)
    goods.stockpile[Good.WHEAT] = 30.0

    growth.station_trade.hand_out_now(STATION, Good.WHEAT)

    assert goods.stockpile[Good.WHEAT] == 30.0
    assert railway.finances.month["cargo"] == 0.0


def test_goods_trains_brought_but_kept_for_trains_are_not_handed_out_either() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    bring(railway, Good.BREAD, 50.0)
    railway.station_goods(STATION).keep_for_trains(Good.BREAD)

    growth.station_trade.hand_out_now(STATION, Good.BREAD)
    growth.trade_goods()

    assert railway.station_goods(STATION).brought[Good.BREAD] == pytest.approx(50.0)


def test_a_station_is_short_only_of_what_its_buildings_still_lack_this_month() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    growth.trade_goods()
    hungry = growth.station_trade.short_of(STATION)
    bring(railway, Good.BREAD, 50.0)

    growth.station_trade.hand_out_now(STATION, Good.BREAD)

    assert hungry[Need.FOOD] == pytest.approx(10.0)
    assert Need.FOOD not in growth.station_trade.short_of(STATION)


def test_a_train_can_leave_diamonds_nobody_around_wants_for_another_train_to_take() -> None:
    growth, railway = joined_stations()
    bring(railway, Good.DIAMONDS, 10.0)

    growth.station_trade.hand_out_now(STATION, Good.DIAMONDS)
    growth.trade_goods()

    assert railway.station_goods(STATION).brought[Good.DIAMONDS] == pytest.approx(10.0)


def test_goods_a_train_brings_go_to_the_nearest_building_first_however_well_fed() -> None:
    growth, railway = joined_stations()
    railway.levels[STATION] = 2
    shape = growth.constructions.tile_map.grid.shape
    near, far = (6, 21), (6, 22)
    for cell in (near, far):
        growth.build(Building(cell, BuildingKind.HOUSE))
    growth.economy.levels[np.ravel_multi_index(near, shape)] = Satisfaction.THRIVING
    growth.economy.levels[np.ravel_multi_index(far, shape)] = Satisfaction.UNSATISFIED
    house_food = needs_of(BuildingKind.HOUSE, growth.year)[Need.FOOD]
    bring(railway, Good.BREAD, house_food)

    growth.station_trade.hand_out_now(STATION, Good.BREAD)

    assert growth.economy.taken is not None
    taken = growth.economy.taken[(BuildingKind.HOUSE, Need.FOOD)]
    assert taken == {np.ravel_multi_index(near, shape): pytest.approx(house_food)}
