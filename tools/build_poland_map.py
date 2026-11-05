"""Bake the Poland maps, one with square cells and one with hexagons, into src/railroads/maps/.

Inputs in data/raw/ (download commands in README.md):
- etopo_poland.csv: NOAA ETOPO1 elevation, 1 arc-minute grid.
- ne_10m_*.geojson: Natural Earth lakes and rivers, and the ocean when it has been downloaded.

Run with: uv run python tools/build_poland_map.py
"""

from collections import Counter
from pathlib import Path

from railroads import maps
from railroads.mapgen.generate import generate_map
from railroads.mapgen.sources import ElevationGrid, Shape, read_geojson
from railroads.mapgen.strategies import (
    HeightFromElevation,
    Strategy,
    TemperatureFromClimate,
    VegetationPlaceholder,
    WaterFromShapes,
)
from railroads.sim.geo import Bounds
from railroads.sim.grids import GRIDS
from railroads.sim.tilemap import LAYERS, Temperature, Terrain, TileMap, Vegetation

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BOUNDS = Bounds(west=13.6, east=24.4, south=48.9, north=55.0)
KM_PER_CELL = 2.0
MIN_RIVER_KM = 200.0
LAKE_FILES = ("ne_10m_lakes.geojson", "ne_10m_lakes_europe.geojson")
RIVER_FILES = ("ne_10m_rivers_lake_centerlines.geojson", "ne_10m_rivers_europe.geojson")
OCEAN_FILE = RAW / "ne_10m_ocean.geojson"
SOURCES = (
    "Elevation: NOAA ETOPO1, 1 arc-minute, public domain",
    "Sea, lakes and rivers: Natural Earth 1:10m, public domain",
)


def strategies() -> dict[str, Strategy]:
    return {
        "height_m": HeightFromElevation(ElevationGrid.from_erddap_csv(RAW / "etopo_poland.csv")),
        "terrain": WaterFromShapes(
            lakes=_shapes(LAKE_FILES),
            rivers=_shapes(RIVER_FILES),
            sea=read_geojson(OCEAN_FILE) if OCEAN_FILE.exists() else None,
            min_river_km=MIN_RIVER_KM,
        ),
        "temperature": TemperatureFromClimate(),
        "vegetation": VegetationPlaceholder(seed=7),
    }


def _shapes(file_names: tuple[str, ...]) -> list[Shape]:
    return [shape for name in file_names for shape in read_geojson(RAW / name)]


def _summary(tile_map: TileMap) -> str:
    kinds = {"terrain": Terrain, "vegetation": Vegetation, "temperature": Temperature}
    parts = []
    for layer in LAYERS:
        if layer in kinds:
            counts = Counter(getattr(tile_map, layer).ravel().tolist())
            parts.append(
                ", ".join(f"{kinds[layer](k).name.lower()} {n}" for k, n in sorted(counts.items()))
            )
    return "; ".join(parts)


def main() -> None:
    chosen = strategies()
    print("sea from", "Natural Earth ocean" if OCEAN_FILE.exists() else "height (no ocean file)")
    for kind, grid_type in GRIDS.items():
        grid = grid_type.covering(BOUNDS.width_km, BOUNDS.height_km, KM_PER_CELL)
        tile_map = generate_map("Poland", BOUNDS, grid, chosen, SOURCES)
        path = maps.map_file("poland", kind)
        tile_map.save(path)
        print(f"wrote {path.name}: {grid.rows} x {grid.cols} {kind} cells; {_summary(tile_map)}")


if __name__ == "__main__":
    main()
