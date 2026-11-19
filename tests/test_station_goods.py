from railroads.sim.goods import Good
from railroads.sim.station_goods import StationGoods


def test_a_station_shows_one_stockpile_of_what_it_gathered_and_what_trains_brought() -> None:
    goods = StationGoods()
    goods.stockpile.update({Good.WHEAT: 30.0, Good.PASSENGERS: 12.0})
    goods.bring(Good.WHEAT, {(1, 1): 20.0})
    goods.bring(Good.COAL, {(1, 1): 5.0})

    shown = goods.freight(), goods.waiting()

    assert shown == ({Good.WHEAT: 50.0, Good.COAL: 5.0}, {Good.PASSENGERS: 12.0})
