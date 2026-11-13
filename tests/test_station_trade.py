from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.demand import DEMAND_PER_NEED
from railroads.sim.goods import Need
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
    assert passengers == {(6, 10): DEMAND_PER_NEED * 6, (6, 13): DEMAND_PER_NEED * 10}


def test_a_raised_station_counts_the_buildings_in_its_wider_ring() -> None:
    growth, railway = close_stations()
    put_up(growth, (6, 7), 4)
    growth.trade_goods()
    before = railway.station_demand((6, 10)).capacity.get(Need.PASSENGERS)
    railway.levels[(6, 10)] = 2

    growth.trade_goods()

    assert before is None
    assert railway.station_demand((6, 10)).capacity[Need.PASSENGERS] == DEMAND_PER_NEED * 8


def test_the_cells_in_reach_are_every_cell_any_station_hands_goods_out_to() -> None:
    growth, _ = close_stations()
    grid = growth.grid

    growth.trade_goods()

    reached = set(grid.within((6, 10), 2)) | set(grid.within((6, 13), 2))
    assert growth.station_trade.in_reach == reached
