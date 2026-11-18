import pytest
from test_railway import joined_stations

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.demand import DEMAND_PER_NEED
from railroads.sim.goods import Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station


def close_stations() -> tuple[Growth, Railway]:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 10))
    build_station(growth, (6, 13))
    railway = Railway(constructions)
    railway.levels = {(6, 10): 2, (6, 13): 2}
    growth.railway = railway
    return growth, railway


def put_up(growth: Growth, cell: tuple[int, int], houses: int) -> None:
    for _ in range(houses):
        growth.build(Building(cell, BuildingKind.HOUSE))


def test_stations_whose_rings_overlap_each_count_every_building_in_their_own_ring() -> None:
    growth, railway = close_stations()
    put_up(growth, (6, 11), 2)
    put_up(growth, (6, 8), 1)
    put_up(growth, (6, 15), 3)

    growth.trade_goods()

    passengers = {
        station: railway.station_demand(station).capacity[Need.PASSENGERS]
        for station in railway.stations()
    }
    assert passengers == {(6, 10): DEMAND_PER_NEED * 3, (6, 13): DEMAND_PER_NEED * 5}


def test_a_raised_station_counts_the_buildings_in_its_wider_ring() -> None:
    growth, railway = close_stations()
    put_up(growth, (6, 7), 4)
    growth.trade_goods()
    before = railway.station_demand((6, 10)).capacity.get(Need.PASSENGERS)
    railway.levels[(6, 10)] = 3

    growth.trade_goods()

    assert before is None
    assert railway.station_demand((6, 10)).capacity[Need.PASSENGERS] == DEMAND_PER_NEED * 4


def test_the_cells_in_reach_are_every_cell_any_station_hands_goods_out_to() -> None:
    growth, _ = close_stations()
    grid = growth.grid

    growth.trade_goods()

    reached = set(grid.within((6, 10), 2)) | set(grid.within((6, 13), 2))
    assert growth.station_trade.in_reach == reached


def test_goods_buildings_hand_out_on_their_own_by_a_station_earn_the_player_nothing() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    growth.build(Building((6, 22), BuildingKind.FARM))

    growth.trade_goods()

    assert railway.finances.month["cargo"] == 0.0
    assert railway.station_goods((6, 20)).stockpile.get(Good.WHEAT, 0.0) == pytest.approx(
        10 * 8 / 21
    )


def food_wanted_with_a_farm_at(farm: tuple[int, int] | None) -> float:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    if farm is not None:
        growth.build(Building(farm, BuildingKind.FARM))
    growth.trade_goods()
    return railway.station_demand((6, 20)).capacity.get(Need.FOOD, 0.0)


def test_a_station_wants_more_food_the_less_the_farms_around_could_hand_its_houses() -> None:
    farms = [(6, 22), (6, 25), None]

    wanted = [food_wanted_with_a_farm_at(farm) for farm in farms]

    assert wanted[0] < wanted[1] < wanted[2]
