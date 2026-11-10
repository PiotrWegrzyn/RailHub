import numpy as np
import pytest

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.demolition import (
    STATION_REMOVAL,
    bulldoze,
    bulldoze_price,
    demolition_price,
    station_site_price,
)
from railroads.sim.geo import Bounds
from railroads.sim.goods import Good
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway, Stop
from railroads.sim.routes import build_station, lay_route, move_station, plan_route, route_price
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

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


def grown(*placed: tuple[tuple[int, int], BuildingKind]) -> Growth:
    constructions = Constructions(make_map())
    growth = Growth(constructions, QUIET)
    for cell, kind in placed:
        growth.build(Building(cell, kind))
    return growth


@pytest.mark.parametrize(
    ("kind", "price"),
    [
        (BuildingKind.HOUSE, 500),
        (BuildingKind.FARM, 1_000),
        (BuildingKind.MINE, 20_000),
        (BuildingKind.FACTORY, 15_000),
        (BuildingKind.MEGA_FACTORY, 135_000),
        (BuildingKind.SKYSCRAPER, 1_000_000),
        (BuildingKind.SPACE_CENTER, None),
        (BuildingKind.ST_PAULS, None),
    ],
)
def test_homes_and_fields_are_cheap_to_take_down_works_dear_and_one_offs_impossible(
    kind: BuildingKind, price: float | None
) -> None:
    worth = demolition_price(kind)

    assert worth == price


def test_a_station_pays_for_every_building_it_clears_and_not_where_a_one_off_stands() -> None:
    growth = grown(
        ((5, 5), BuildingKind.HOUSE),
        ((5, 5), BuildingKind.HOUSE),
        ((5, 9), BuildingKind.ROYAL_PALACE),
    )

    houses = station_site_price(growth.constructions, (5, 5))
    palace = station_site_price(growth.constructions, (5, 9))

    assert houses == 1_000
    assert palace is None
    assert build_station(growth, (5, 9)) is None


def test_track_through_buildings_costs_what_clearing_them_does() -> None:
    growth = grown()
    route = plan_route(growth.constructions, (6, 3), (6, 9))
    assert route is not None
    plain = route_price(growth.constructions, route)
    middle = route.cells[1:-1]
    for cell in middle:
        for _ in range(6):
            growth.build(Building(cell, BuildingKind.HOUSE))

    price = route_price(growth.constructions, route)

    assert price == pytest.approx(plain + 500 * len(middle))


def test_the_bulldozer_clears_buildings_and_track_for_their_price_and_leaves_one_offs() -> None:
    growth = grown(((5, 5), BuildingKind.MINE), ((5, 5), BuildingKind.HOUSE))
    growth.build(Building((5, 9), BuildingKind.ROYAL_PALACE))
    build_station(growth, (5, 12))

    price = bulldoze_price(growth.constructions, (5, 5))
    bulldoze(growth, (5, 5))
    bulldoze(growth, (5, 9))

    assert price == 20_500
    assert growth.constructions.at((5, 5)) == ()
    assert growth.count(BuildingKind.MINE).sum() == 0
    assert growth.count(BuildingKind.ROYAL_PALACE).sum() == 1
    assert bulldoze_price(growth.constructions, (5, 12)) == STATION_REMOVAL


def test_a_moved_station_keeps_its_trains_name_and_goods_but_starts_its_upgrades_again() -> None:
    growth = grown()
    for cell in ((6, 3), (6, 20)):
        build_station(growth, cell)
    route = plan_route(growth.constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    railway = Railway(growth.constructions)
    name = railway.name_of((6, 20))
    railway.levels[(6, 20)] = 2
    railway.station_goods((6, 20)).stockpile[Good.COAL] = 40.0
    railway.station_goods((6, 20)).storage = 1
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3)), Stop((6, 20))]

    changed = move_station(growth, railway, (6, 20), (8, 22))

    assert changed is not None
    assert railway.stations() == [(6, 3), (8, 22)]
    assert [stop.cell for stop in train.stops] == [(6, 3), (8, 22)]
    assert railway.name_of((8, 22)) == name
    assert railway.level((8, 22)) == 1
    assert railway.station_goods((8, 22)).stockpile[Good.COAL] == 40.0
    assert railway.station_goods((8, 22)).storage == 0
