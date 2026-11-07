import numpy as np
import pytest

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import Economy, Good, StationOrder
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth, GrowthRule
from railroads.sim.railway import CAR_CAPACITY, CELLS_PER_MONTH, MAX_CARS, STOCKPILE, Car, Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation


def make_map() -> TileMap:
    grid = HexGrid(rows=12, cols=30, km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )


def joined_stations() -> tuple[Growth, Railway]:
    constructions = Constructions(make_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 3))
    build_station(growth, (6, 20))
    route = plan_route(constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    return growth, railway


def test_trains_find_their_way_along_track_between_stations() -> None:
    _, railway = joined_stations()

    way = railway.route((6, 3), (6, 20))

    assert way is not None
    points, speeds = way
    assert len(speeds) == len(points) - 1
    assert railway.route((6, 3), (2, 3)) is None


def test_a_train_runs_to_its_next_stop_and_unloads_there() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [(6, 3), (6, 20)]
    train.cars = [Car(Good.FOOD, 10.0)]
    train.heading = 1

    railway.run(0.1)
    halfway = railway.position(train)
    railway.run(25 / CELLS_PER_MONTH)

    assert halfway is not None
    assert train.heading == 0
    assert railway.station_goods((6, 20)).brought[Good.FOOD] == 10.0
    assert train.load == {}
    assert train.cars == [Car(Good.FOOD, 0.0)]


def test_each_car_fills_with_its_own_good_from_the_stockpile() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [(6, 20), (6, 3)]
    train.cars = [Car(Good.LUMBER), Car(Good.LUMBER), Car(Good.FOOD)]
    railway.station_goods((6, 3)).stockpile.update({Good.LUMBER: 14.0, Good.COAL: 30.0})
    train.heading = 1

    railway.run(30 / CELLS_PER_MONTH)

    assert [car.load for car in train.cars] == [CAR_CAPACITY, 4.0, 0.0]
    assert railway.station_goods((6, 3)).stockpile == {Good.LUMBER: 0.0, Good.COAL: 30.0}


def test_a_train_pulls_at_most_six_cars() -> None:
    _, railway = joined_stations()
    train = railway.buy()

    added = [train.add_car(Good.COAL) for _ in range(MAX_CARS + 1)]

    assert added == [True] * MAX_CARS + [False]


def test_without_track_a_train_waits() -> None:
    constructions = Constructions(make_map())
    railway = Railway(constructions)
    train = railway.buy()
    train.stops = [(6, 3), (6, 20)]

    railway.run(1.0)

    assert train.status == "No track to its next stop"


def test_stations_hand_goods_to_the_hungriest_buildings_and_gather_spare_ones() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    counts = {kind: np.zeros(tile_map.grid.shape, dtype=np.int64) for kind in BuildingKind}
    counts[BuildingKind.FLATS][6, 10] = 6
    counts[BuildingKind.LUMBER_MILL][6, 25] = 1
    orders = {
        (6, 11): StationOrder({Good.FOOD: 20.0}, 0.0),
        (6, 24): StationOrder({}, 50.0),
    }

    trade = economy.trade(counts, orders)

    index = int(np.searchsorted(trade.cells, np.ravel_multi_index((6, 25), tile_map.grid.shape)))
    assert trade.handed[(6, 11)][Good.FOOD] == pytest.approx(20.0)
    assert trade.gathered[(6, 24)][Good.LUMBER] == pytest.approx(2.0)
    assert trade.gave[BuildingKind.LUMBER_MILL][index] == pytest.approx(2.0)
    assert sum(trade.gathered[(6, 24)].values()) <= 50.0


def test_growth_hands_station_goods_out_and_stores_what_was_gathered() -> None:
    growth, railway = joined_stations()
    growth.constructions.build(Building((6, 21), BuildingKind.LUMBER_MILL))
    growth._counts[BuildingKind.LUMBER_MILL][6, 21] += 1

    growth.trade_goods()

    goods = railway.station_goods((6, 20))
    assert goods.stockpile[Good.LUMBER] == pytest.approx(2.0)
    assert goods.stocked <= STOCKPILE


def test_stations_never_take_the_food_the_land_makes() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    counts = {kind: np.zeros(tile_map.grid.shape, dtype=np.int64) for kind in BuildingKind}
    counts[BuildingKind.HOUSE][6, 10] = 1
    counts[BuildingKind.FARM][6, 14] = 1

    trade = economy.trade(counts, {(6, 12): StationOrder({}, 100.0)})

    assert trade.gathered[(6, 12)][Good.FOOD] == pytest.approx(6.0)


def test_cars_slow_a_train_and_full_cars_slow_it_more() -> None:
    _, railway = joined_stations()
    light, empty, full = railway.buy(), railway.buy(), railway.buy()
    empty.cars = [Car(Good.COAL) for _ in range(MAX_CARS)]
    full.cars = [Car(Good.COAL, CAR_CAPACITY) for _ in range(MAX_CARS)]

    pulls = [train.pull for train in (light, empty, full)]

    assert pulls[0] == 1.0
    assert pulls[0] > pulls[1] > pulls[2] >= 0.5
