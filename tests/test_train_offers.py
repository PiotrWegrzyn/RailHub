from test_railway import joined_stations

from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.goods import RECIPES, Good, makes_of
from railroads.ui.trains_window import goods_to_load


def test_a_station_supplies_what_buildings_in_its_reach_make() -> None:
    growth, _ = joined_stations()
    growth.build(Building((6, 22), BuildingKind.MINE))
    growth.build(Building((6, 12), BuildingKind.LUMBER_MILL))

    made = growth.station_trade.supplies((6, 20))

    assert set(makes_of(BuildingKind.MINE, growth.year)) <= set(made)
    assert Good.LUMBER not in made


def test_a_station_supplies_the_most_a_processor_in_reach_makes() -> None:
    growth, _ = joined_stations()
    growth.build(Building((6, 21), BuildingKind.STEEL_MILL))

    made = growth.station_trade.supplies((6, 20))

    assert made[Good.STEEL] == RECIPES[BuildingKind.STEEL_MILL].most


def test_a_train_can_load_what_is_made_around_or_waits_at_a_station_and_nothing_else() -> None:
    _, railway = joined_stations()
    goods = railway.station_goods((6, 20))
    goods.stockpile[Good.COAL] = 30.0
    goods.bring(Good.WHEAT, {(6, 3): 12.0})

    offers = goods_to_load(railway, (6, 20), {Good.IRON: 50.0, Good.COAL: 40.0})

    assert {(o.good, o.waiting, o.made) for o in offers} == {
        (Good.COAL, 30.0, 40.0),
        (Good.IRON, 0.0, 50.0),
        (Good.WHEAT, 12.0, 0.0),
    }


def test_goods_not_yet_invented_are_not_offered() -> None:
    _, railway = joined_stations()
    railway.year = 1800

    offers = goods_to_load(railway, (6, 20), {Good.ELECTRONICS: 10.0, Good.COAL: 5.0})

    assert [o.good for o in offers] == [Good.COAL]
