import pytest

from builders import joined_stations
from railroads.sim.clock import QUARTER
from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.goods import TRAVELLERS, Good
from railroads.sim.station_goods import StationGoods
from railroads.sim.yearly import makes_of
from railroads.ui.station_offers import goods_to_load


def test_a_station_supplies_what_buildings_in_its_reach_make() -> None:
    growth, _ = joined_stations()
    growth.build(Building((6, 22), BuildingKind.MINE))
    growth.build(Building((6, 12), BuildingKind.LOGGING_CAMP))

    made = growth.station_trade.supplies((6, 20))

    assert set(makes_of(BuildingKind.MINE, growth.year)) <= set(made)
    assert Good.PLANKS not in made


def test_a_station_offers_what_a_processor_in_reach_makes_before_it_has_gathered_any() -> None:
    growth, _ = joined_stations()
    growth.build(Building((6, 21), BuildingKind.STEEL_MILL))

    made = growth.station_trade.supplies((6, 20))

    assert made == {Good.STEEL: 0.0} | {g: 0.0 for g in made if g in TRAVELLERS}


def test_a_station_supplies_only_what_it_was_given_over_the_last_quarter() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 22), BuildingKind.MINE))
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    for _ in range(QUARTER + 1):
        growth.trade_goods()

    made = growth.station_trade.supplies((6, 20))

    assert growth.last_trade is not None
    month = growth.last_trade.gathered[(6, 20)][Good.COAL]
    assert made[Good.COAL] == pytest.approx(QUARTER * month)
    assert 0 < month < makes_of(BuildingKind.MINE, growth.year)[Good.COAL]
    assert made[Good.PASSENGERS] == pytest.approx(
        QUARTER * railway.station_goods((6, 20)).stockpile[Good.PASSENGERS]
    )


def test_a_station_forgets_what_it_was_given_over_a_quarter_ago() -> None:
    goods = StationGoods()
    for amount in (100.0, 1.0, 2.0, 3.0):
        goods.count_month({Good.COAL: amount})

    quarter = goods.last_quarter()

    assert quarter == {Good.COAL: pytest.approx(6.0)}


def test_a_train_can_load_what_is_made_around_or_waits_at_a_station_and_nothing_else() -> None:
    _, railway = joined_stations()
    goods = railway.station_goods((6, 20))
    goods.stockpile[Good.COAL] = 30.0
    goods.bring(Good.WHEAT, {(6, 3): 12.0})

    offers = goods_to_load(railway, (6, 20), {Good.IRON: 50.0, Good.COAL: 40.0})

    assert {(o.good, o.waiting, o.gathered) for o in offers} == {
        (Good.COAL, 30.0, 40.0),
        (Good.IRON, 0.0, 50.0),
        (Good.WHEAT, 12.0, 0.0),
    }


def test_goods_not_yet_invented_are_not_offered() -> None:
    _, railway = joined_stations()
    railway.year = 1800

    offers = goods_to_load(railway, (6, 20), {Good.ELECTRONICS: 10.0, Good.COAL: 5.0})

    assert [o.good for o in offers] == [Good.COAL]
