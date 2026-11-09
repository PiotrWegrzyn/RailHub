import numpy as np
import pytest

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import Economy, Good, StationOrder
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth, GrowthRule
from railroads.sim.railway import (
    CAR_CAPACITY,
    CELLS_PER_MONTH,
    MAX_CARS,
    STOCKPILE,
    Car,
    Railway,
    Stop,
)
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


def test_a_new_train_leaves_its_first_stop_with_that_stops_cars_full() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.FOOD] = 15.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.FOOD, Good.FOOD], wait_for_load=False), Stop((6, 20))]

    railway.run(0.1)

    assert train.heading == 1
    assert [car.load for car in train.cars] == [CAR_CAPACITY, 5.0]
    assert railway.position(train) is not None


def test_a_train_waits_at_its_stop_until_new_goods_fill_its_cars() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.FOOD]), Stop((6, 20))]
    railway.run(0.1)
    waiting = train.status

    railway.station_goods((6, 3)).stockpile[Good.FOOD] = 12.0
    railway.run(0.1)

    assert waiting == "Loading at 6, 3"
    assert [car.load for car in train.cars] == [CAR_CAPACITY]
    assert train.status == "Running"


def test_a_train_gives_up_waiting_after_a_month() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.FOOD]), Stop((6, 20))]

    for _ in range(12):
        railway.run(0.1)

    assert train.status == "Running"
    assert train.cars[0].load == 0.0


def test_at_the_next_stop_it_unloads_and_takes_that_stops_cars() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.FOOD] = 10.0
    railway.station_goods((6, 20)).stockpile[Good.COAL] = 30.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.FOOD]), Stop((6, 20), [Good.COAL, Good.COAL])]

    railway.run(0.1)
    railway.run(25 / CELLS_PER_MONTH)

    assert train.heading == 0
    assert railway.station_goods((6, 20)).brought[Good.FOOD] == 10.0
    assert [(car.good, car.load) for car in train.cars] == [(Good.COAL, 10.0), (Good.COAL, 10.0)]


def test_each_car_fills_with_its_own_good_from_the_stockpile() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.LUMBER, Good.LUMBER, Good.FOOD]), Stop((6, 20))]
    railway.station_goods((6, 3)).stockpile.update({Good.LUMBER: 14.0, Good.COAL: 30.0})

    railway.run(0.01)

    assert [car.load for car in train.cars] == [CAR_CAPACITY, 4.0, 0.0]
    assert railway.station_goods((6, 3)).stockpile == {Good.LUMBER: 0.0, Good.COAL: 30.0}


def test_a_stop_sends_at_most_six_cars() -> None:
    stop = Stop((6, 3))

    added = [stop.add_car(Good.COAL) for _ in range(MAX_CARS + 1)]

    assert added == [True] * MAX_CARS + [False]


def test_without_track_a_train_waits() -> None:
    constructions = Constructions(make_map())
    railway = Railway(constructions)
    train = railway.buy()
    train.stops = [Stop((6, 3)), Stop((6, 20))]

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


def test_trains_run_through_a_station_on_the_way() -> None:
    growth, railway = joined_stations()
    build_station(growth, (6, 27))
    route = plan_route(growth.constructions, (6, 20), (6, 27))
    assert route is not None
    lay_route(growth, route)

    way = railway.route((6, 3), (6, 27))

    assert way is not None
    points, _ = way
    assert any(
        np.allclose(point, growth.constructions.tile_map.grid.center(6, 20)) for point in points
    )


def test_cars_set_after_the_stops_still_load_on_the_first_trip() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.FOOD] = 10.0
    train = railway.buy()
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    railway.run(0.05)

    train.stops[0].add_car(Good.FOOD)
    train.cars_changed()
    railway.run(0.05)

    assert [(car.good, car.load) for car in train.cars] == [(Good.FOOD, 10.0)]


def test_later_car_changes_wait_for_the_next_call() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    railway.run(1.0)
    heading = train.heading

    train.stops[0].add_car(Good.COAL)
    train.cars_changed()

    assert train.arrivals >= 1
    assert train.heading == heading


def test_removing_one_loaded_car_keeps_the_others_full_and_returns_its_goods() -> None:
    _, railway = joined_stations()
    stockpile = railway.station_goods((6, 3)).stockpile
    stockpile[Good.FOOD] = 80.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.FOOD] * MAX_CARS, wait_for_load=False), Stop((6, 20))]
    railway.run(0.001)

    train.stops[0].cars.pop()
    train.cars_changed()
    railway.run(0.001)

    assert [car.load for car in train.cars] == [CAR_CAPACITY] * (MAX_CARS - 1)
    assert stockpile[Good.FOOD] == pytest.approx(30.0)
    assert railway.station_goods((6, 3)).brought == {}


def test_a_train_called_back_on_its_first_trip_keeps_what_it_loaded() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.FOOD] = 60.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.FOOD] * MAX_CARS), Stop((6, 20))]
    railway.run(0.2)
    running = bool(train.path)

    train.stops[0].cars.pop()
    train.cars_changed()
    railway.run(0.001)

    assert running
    assert [car.load for car in train.cars] == [CAR_CAPACITY] * (MAX_CARS - 1)
    assert railway.station_goods((6, 3)).stockpile[Good.FOOD] == pytest.approx(10.0)
