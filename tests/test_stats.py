import numpy as np

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.geo import Bounds
from railroads.sim.goods import Good
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.stats import History
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.sim.trains import Car, Stop

QUIET = GrowthRule(trades=False, ladders=(), forests=None)


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


def test_each_year_closes_with_its_stations_and_population() -> None:
    growth = Growth(Constructions(make_map()), QUIET, start_year=1900)
    growth.build(Building((5, 5), BuildingKind.HOUSE))
    growth.build(Building((5, 6), BuildingKind.TENEMENT))
    growth.railway = Railway(growth.constructions)
    build_station(growth, (6, 3))

    growth.run(24, np.random.default_rng(1))

    assert growth.history.population == {1900: 7, 1901: 7}
    assert growth.history.stations == {1900: 1, 1901: 1}


def test_a_car_counts_as_delivered_where_it_leaves_its_load() -> None:
    growth = Growth(Constructions(make_map()), QUIET)
    for cell in ((6, 3), (6, 20)):
        build_station(growth, cell)
    route = plan_route(growth.constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    railway = Railway(growth.constructions)
    railway.year = 1860
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    train.loaded_at = (6, 3)
    train.cars = [Car(Good.COAL), Car(Good.COAL), Car(Good.WOOL)]
    for car in train.cars[:2]:
        car.fill({(6, 3): 10.0})

    railway.calls.arrive(train, Stop((6, 20)))

    assert railway.history.delivered == {1860: {Good.COAL: 2}}


def test_history_counts_cars_by_year_and_good() -> None:
    history = History()

    history.deliver(1900, Good.COAL)
    history.deliver(1900, Good.COAL, 2)
    history.deliver(1901, Good.WOOL)

    assert history.delivered == {1900: {Good.COAL: 3}, 1901: {Good.WOOL: 1}}
