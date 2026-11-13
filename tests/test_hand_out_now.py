import pytest
from test_railway import joined_stations

from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.goods import Good, Need

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


def test_goods_kept_for_trains_stay_in_the_stockpile() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    goods = railway.station_goods(STATION)
    goods.stockpile[Good.WHEAT] = 30.0
    goods.keep_for_trains(Good.WHEAT)

    growth.station_trade.hand_out_now(STATION, Good.WHEAT)

    assert goods.stockpile[Good.WHEAT] == 30.0
