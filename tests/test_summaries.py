import numpy as np

from railroads.sim.clock import GameClock
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import COLLECTION_RADIUS
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.routes import build_station
from railroads.sim.summaries import Reach, snapshot, summarize
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation


def make_growth() -> Growth:
    grid = HexGrid(rows=20, cols=30, km_per_cell=2.0)
    tile_map = TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )
    growth = Growth(Constructions(tile_map), GrowthRule(trades=False, ladders=(), forests=None))
    build_station(growth, (10, 10))
    return growth


def put_up(growth: Growth, cell: tuple[int, int], kind: BuildingKind) -> None:
    growth.constructions.build(Building(cell, kind))
    growth.add(cell, kind, 1)


def test_a_summary_lists_what_was_built_within_reach_of_a_station_and_where() -> None:
    growth = make_growth()
    reach = Reach(growth.constructions.tile_map.grid).of([(10, 10)], lambda _: COLLECTION_RADIUS)
    before = snapshot(growth, reach)
    put_up(growth, (10, 12), BuildingKind.HOUSE)
    put_up(growth, (10, 12), BuildingKind.HOUSE)
    put_up(growth, (10, 25), BuildingKind.FARM)

    summary = summarize("March 1850", growth, reach, before)

    assert summary is not None
    assert summary.date == "March 1850"
    (station,) = summary.stations
    assert station.station == (10, 10)
    assert station.changes == {BuildingKind.HOUSE: 2}
    assert station.cells == ((10, 12),)
    assert summary.built == 2


def test_buildings_pulled_down_count_against_their_kind() -> None:
    growth = make_growth()
    put_up(growth, (11, 10), BuildingKind.HOUSE)
    reach = Reach(growth.constructions.tile_map.grid).of([(10, 10)], lambda _: COLLECTION_RADIUS)
    before = snapshot(growth, reach)
    growth.constructions.remove(Building((11, 10), BuildingKind.HOUSE))
    growth.add((11, 10), BuildingKind.HOUSE, -1)
    put_up(growth, (11, 10), BuildingKind.TENEMENT)

    summary = summarize("April 1850", growth, reach, before)

    assert summary is not None
    assert summary.stations[0].changes == {BuildingKind.HOUSE: -1, BuildingKind.TENEMENT: 1}
    assert (summary.built, summary.pulled_down) == (1, 1)


def test_a_month_with_nothing_changed_near_a_station_has_no_summary() -> None:
    growth = make_growth()
    reach = Reach(growth.constructions.tile_map.grid).of([(10, 10)], lambda _: COLLECTION_RADIUS)
    before = snapshot(growth, reach)
    put_up(growth, (2, 28), BuildingKind.HOUSE)

    summary = summarize("May 1850", growth, reach, before)

    assert summary is None


def test_the_clock_names_any_month_since_the_start() -> None:
    clock = GameClock()

    dates = [clock.date_of(months) for months in (0, 2, 13)]

    assert dates == ["January 1850", "March 1850", "February 1851"]
