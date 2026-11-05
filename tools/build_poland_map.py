"""Bake the Poland map from public-domain data into src/railroads/sim/maps/poland.npz.

Inputs in data/raw/ (download commands in README.md):
- etopo_poland.csv: NOAA ETOPO1 elevation, 1 arc-minute grid.
- ne_10m_*.geojson: Natural Earth lakes and rivers.

Run with: uv run python tools/build_poland_map.py
"""

import itertools
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from railroads.sim.geo import Projection, polygon_mask
from railroads.sim.maps.poland import PROJECTION, TERRAIN_FILE
from railroads.sim.world import Terrain

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
LAKE_FILES = ("ne_10m_lakes.geojson", "ne_10m_lakes_europe.geojson")
RIVER_FILES = ("ne_10m_rivers_lake_centerlines.geojson", "ne_10m_rivers_europe.geojson")
HILLS_FROM_M = 250
MOUNTAINS_FROM_M = 600
RIVER_SAMPLES_PER_TILE = 4

Ring = list[tuple[float, float]]


def load_elevation(projection: Projection) -> npt.NDArray[np.int16]:
    rows = np.genfromtxt(RAW / "etopo_poland.csv", delimiter=",", skip_header=2)
    lats, lons = np.unique(rows[:, 0]), np.unique(rows[:, 1])
    grid = rows[:, 2].reshape(len(lats), len(lons))
    xs, ys = projection.tile_centers()
    lon = projection.west + xs / projection.width_tiles * (projection.east - projection.west)
    lat = projection.north - ys / projection.height_tiles * (projection.north - projection.south)
    lat_index = np.clip(np.searchsorted(lats, lat), 0, len(lats) - 1)
    lon_index = np.clip(np.searchsorted(lons, lon), 0, len(lons) - 1)
    return grid[lat_index, lon_index].astype(np.int16)


def sea_mask(elevation: npt.NDArray[np.int16]) -> npt.NDArray[np.bool_]:
    """Cells at or below sea level that connect to the map edge, so inland depressions stay land."""
    low = elevation <= 0
    sea = np.zeros_like(low)
    sea[0, :], sea[-1, :], sea[:, 0], sea[:, -1] = low[0, :], low[-1, :], low[:, 0], low[:, -1]
    while True:
        grown = sea.copy()
        grown[1:, :] |= sea[:-1, :]
        grown[:-1, :] |= sea[1:, :]
        grown[:, 1:] |= sea[:, :-1]
        grown[:, :-1] |= sea[:, 1:]
        grown &= low
        if np.array_equal(grown, sea):
            return sea
        sea = grown


def _geometries(file_name: str) -> list[dict]:
    features = json.loads((RAW / file_name).read_text())["features"]
    return [f["geometry"] for f in features if f["geometry"]]


def _polygon_rings(geometry: dict) -> list[Ring]:
    if geometry["type"] == "Polygon":
        return [ring for ring in geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return [ring for polygon in geometry["coordinates"] for ring in polygon]
    return []


def _lines(geometry: dict) -> list[Ring]:
    if geometry["type"] == "LineString":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiLineString":
        return list(geometry["coordinates"])
    return []


def lake_mask(projection: Projection) -> npt.NDArray[np.bool_]:
    xs, ys = projection.tile_centers()
    lakes = np.zeros(xs.shape, dtype=bool)
    for file_name in LAKE_FILES:
        for geometry in _geometries(file_name):
            for ring in _polygon_rings(geometry):
                points = np.array([projection.to_tile(lon, lat) for lon, lat in ring])
                x0, y0 = np.floor(points.min(axis=0)).astype(int)
                x1, y1 = np.ceil(points.max(axis=0)).astype(int)
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, projection.width_tiles), min(y1, projection.height_tiles)
                if x0 >= x1 or y0 >= y1:
                    continue
                window = (slice(y0, y1), slice(x0, x1))
                lon_lat_ring = [(lon, lat) for lon, lat in ring]
                lakes[window] ^= polygon_mask(xs[window], ys[window], projection, lon_lat_ring)
    return lakes


def river_mask(projection: Projection) -> npt.NDArray[np.bool_]:
    rivers = np.zeros((projection.height_tiles, projection.width_tiles), dtype=bool)
    for file_name in RIVER_FILES:
        for geometry in _geometries(file_name):
            for line in _lines(geometry):
                points = np.array([projection.to_tile(lon, lat) for lon, lat in line])
                for start, end in itertools.pairwise(points):
                    samples = max(2, int(np.hypot(*(end - start)) * RIVER_SAMPLES_PER_TILE))
                    for x, y in np.linspace(start, end, samples).astype(int):
                        if 0 <= x < projection.width_tiles and 0 <= y < projection.height_tiles:
                            rivers[y, x] = True
    return rivers


def main() -> None:
    elevation = load_elevation(PROJECTION)
    terrain = np.full(elevation.shape, Terrain.PLAINS, dtype=np.uint8)
    terrain[elevation >= HILLS_FROM_M] = Terrain.HILLS
    terrain[elevation >= MOUNTAINS_FROM_M] = Terrain.MOUNTAINS
    water = sea_mask(elevation) | lake_mask(PROJECTION) | river_mask(PROJECTION)
    terrain[water] = Terrain.WATER
    np.savez_compressed(TERRAIN_FILE, terrain=terrain, elevation=elevation)
    counts = {kind.name: int((terrain == kind).sum()) for kind in Terrain}
    print(f"wrote {TERRAIN_FILE} {terrain.shape[1]}x{terrain.shape[0]} tiles: {counts}")


if __name__ == "__main__":
    main()
