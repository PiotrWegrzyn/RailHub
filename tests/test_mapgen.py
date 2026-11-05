from collections import deque
from collections.abc import Mapping

import numpy as np
import pytest

from railroads.mapgen.area import MapArea, clip_ring, densify
from railroads.mapgen.generate import generate_map
from railroads.mapgen.sources import ElevationGrid, Shape
from railroads.mapgen.strategies import (
    HeightFromElevation,
    Layer,
    TemperatureFromClimate,
    VegetationPlaceholder,
    WaterFromShapes,
    sea_from_height,
)
from railroads.sim.geo import Bounds
from railroads.sim.grids import Cell, Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Temperature, Terrain, Vegetation

BOUNDS = Bounds(west=20.0, east=21.0, south=50.0, north=50.6)


@pytest.fixture(params=[SquareGrid, HexGrid], ids=["square", "hex"])
def area(request: pytest.FixtureRequest) -> MapArea:
    grid = request.param.covering(BOUNDS.width_km, BOUNDS.height_km, km_per_cell=2.0)
    return MapArea.of(BOUNDS, grid)


def lonlat_points(area: MapArea, points_in_cells: list[tuple[float, float]]) -> np.ndarray:
    points = np.asarray(points_in_cells, dtype=float)
    lon, lat = area.to_lonlat(points[:, 0], points[:, 1])
    return np.column_stack([lon, lat])


def polygon(area: MapArea, *rings: list[tuple[float, float]]) -> Shape:
    return Shape("polygon", tuple(lonlat_points(area, ring) for ring in rings))


def line(area: MapArea, points: list[tuple[float, float]], name: str) -> Shape:
    return Shape("line", (lonlat_points(area, points),), {"name": name})


def cell_at(area: MapArea, x: float, y: float) -> Cell:
    row, col = area.grid.cell_at(x, y)
    return int(row), int(col)


def connected_parts(grid: Grid, cells: set[Cell]) -> int:
    unseen, parts = set(cells), 0
    while unseen:
        parts += 1
        queue = deque([unseen.pop()])
        while queue:
            for neighbour in grid.neighbours(*queue.popleft()):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    return parts


def water_only(**kwargs: object) -> WaterFromShapes:
    return WaterFromShapes(**{"lakes": [], "rivers": [], "sea": [], **kwargs})


def test_cover_measures_how_much_of_each_cell_a_polygon_covers(area: MapArea) -> None:
    width, height = area.grid.size()
    half = width / 2
    left_half = polygon(area, [(-1, -1), (half, -1), (half, height + 1), (-1, height + 1)])

    cover = area.cover([left_half])

    assert cover.mean() == pytest.approx(0.5, abs=0.03)
    assert cover[:, 0].min() == 1.0
    assert cover[:, -1].max() == 0.0


def test_cover_leaves_a_hole_uncovered(area: MapArea) -> None:
    outer = [(2, 2), (20, 2), (20, 20), (2, 20)]
    hole = [(8, 8), (14, 8), (14, 14), (8, 14)]

    cover = area.cover([polygon(area, outer, hole)])

    assert cover[cell_at(area, 11, 11)] == 0.0
    assert cover[cell_at(area, 5, 5)] == 1.0


def test_clip_ring_cuts_a_huge_square_down_to_the_window() -> None:
    ring = np.array([(-100.0, -100.0), (100.0, -100.0), (100.0, 100.0), (-100.0, 100.0)])

    clipped = clip_ring(ring, 0, 0, 10, 5)

    assert clipped.min(axis=0).tolist() == [0, 0]
    assert clipped.max(axis=0).tolist() == [10, 5]


def test_densify_keeps_points_close_together() -> None:
    points = densify(np.array([(0.0, 0.0), (3.0, 4.0)]), step=0.1)

    gaps = np.hypot(*np.diff(points, axis=0).T)

    assert gaps.max() <= 0.1 + 1e-9
    assert points[-1].tolist() == [3.0, 4.0]


def test_traced_river_is_one_unbroken_chain(area: MapArea) -> None:
    river = line(area, [(1.3, 2.1), (30.7, 27.9)], "Diagonal")

    terrain = water_only(rivers=[river], min_river_km=0)(area, {})

    cells = {(int(r), int(c)) for r, c in np.argwhere(terrain == Terrain.RIVER)}
    assert connected_parts(area.grid, cells) == 1


def test_rivers_shorter_than_the_minimum_are_dropped(area: MapArea) -> None:
    short = line(area, [(2, 5.3), (10, 5.3)], "Short")
    long = line(area, [(2, 15.3), (30, 15.3)], "Long")

    terrain = water_only(rivers=[short, long], min_river_km=30)(area, {})

    assert terrain[cell_at(area, 6, 5.3)] == Terrain.LAND
    assert terrain[cell_at(area, 16, 15.3)] == Terrain.RIVER


def test_big_lakes_stay_and_small_ones_disappear(area: MapArea) -> None:
    big = polygon(area, [(2, 2), (12, 2), (12, 12), (2, 12)])
    small = polygon(area, [(20, 20), (20.4, 20), (20.4, 20.4), (20, 20.4)])

    terrain = water_only(lakes=[big, small])(area, {})

    assert terrain[cell_at(area, 7, 7)] == Terrain.LAKE
    assert terrain[cell_at(area, 20.2, 20.2)] == Terrain.LAND


def test_sea_polygons_make_sea_cells(area: MapArea) -> None:
    width, _ = area.grid.size()
    north_strip = polygon(area, [(-1, -1), (width + 1, -1), (width + 1, 5), (-1, 5)])

    terrain = water_only(sea=[north_strip])(area, {})

    assert (terrain[0] == Terrain.SEA).all()
    assert (terrain[-1] == Terrain.LAND).all()


def test_sea_from_height_is_low_ground_linked_to_the_map_edge() -> None:
    height = np.array([[0, 0, 5, 5, 5], [0, 5, 5, -2, 5], [5, 5, 5, 5, 5]])

    sea = sea_from_height(height)

    assert sea.tolist() == [
        [True, True, False, False, False],
        [True, False, False, False, False],
        [False, False, False, False, False],
    ]


def test_height_is_the_average_over_each_cell(area: MapArea) -> None:
    rising_east = ElevationGrid(
        lons=np.array([19.0, 22.0]),
        lats=np.array([49.0, 51.0]),
        heights=np.array([[0.0, 300.0], [0.0, 300.0]]),
    )

    heights = HeightFromElevation(rising_east)(area, {})

    lon, _ = area.center_lonlat()
    assert heights == pytest.approx((lon - 19.0) * 100, abs=1)


def test_higher_cells_are_colder(area: MapArea) -> None:
    height = np.zeros(area.grid.shape, dtype=np.int16)
    height[:, : area.grid.cols // 2] = 2000

    temperature = TemperatureFromClimate()(area, {"height_m": height})

    assert (temperature[:, 0] == Temperature.COLD).all()
    assert (temperature[:, -1] == Temperature.NORMAL).all()


def test_a_warm_climate_makes_cells_hot(area: MapArea) -> None:
    height = np.zeros(area.grid.shape, dtype=np.int16)

    temperature = TemperatureFromClimate(sea_level_c=25.0)(area, {"height_m": height})

    assert (temperature == Temperature.HOT).all()


def test_nothing_grows_on_water_or_high_ground(area: MapArea) -> None:
    terrain = np.full(area.grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[0] = Terrain.SEA
    height = np.zeros(area.grid.shape, dtype=np.int16)
    height[-1] = 2000

    vegetation = VegetationPlaceholder()(area, {"terrain": terrain, "height_m": height})

    assert (vegetation[0] == Vegetation.BARREN).all()
    assert (vegetation[-1] == Vegetation.BARREN).all()


def test_forest_covers_the_chosen_share_of_land(area: MapArea) -> None:
    layers = {
        "terrain": np.full(area.grid.shape, Terrain.LAND, dtype=np.uint8),
        "height_m": np.zeros(area.grid.shape, dtype=np.int16),
    }

    vegetation = VegetationPlaceholder(forest_share=0.3)(area, layers)

    assert (vegetation == Vegetation.FOREST).mean() == pytest.approx(0.3, abs=0.02)


def test_later_strategies_can_read_earlier_layers(area: MapArea) -> None:
    def height(area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        return np.full(area.grid.shape, 100, dtype=np.int16)

    def terrain(area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        return np.where(layers["height_m"] > 50, Terrain.LAND, Terrain.SEA).astype(np.uint8)

    def constant(area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        return np.ones(area.grid.shape, dtype=np.uint8)

    strategies = {
        "height_m": height,
        "terrain": terrain,
        "temperature": constant,
        "vegetation": constant,
    }

    tile_map = generate_map("Test", BOUNDS, area.grid, strategies)

    assert (tile_map.terrain == Terrain.LAND).all()
