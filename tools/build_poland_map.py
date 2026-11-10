"""Bake the Poland map, in hexagons, into src/railroads/maps/.

Inputs in data/raw/ (download commands in README.md):
- etopo_poland.csv: NOAA ETOPO1 elevation, 1 arc-minute grid.
- ne_10m_*.geojson: Natural Earth sea, lakes, rivers and geography regions (for deserts).

Run with: uv run python tools/build_poland_map.py
"""

from collections import Counter
from pathlib import Path

from railroads import maps
from railroads.mapgen.generate import generate_map
from railroads.mapgen.presets import (
    ETOPO_SOURCE,
    NATURAL_EARTH_SOURCE,
    NaturalEarth,
    real_world_strategies,
)
from railroads.mapgen.sources import ElevationGrid
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.tilemap import LAYERS, Temperature, Terrain, TileMap, Vegetation

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BOUNDS = Bounds(west=14.08, east=24.19, south=48.97, north=54.87)
KM_PER_CELL = 2.0
SEED = 7


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
    elevation = ElevationGrid.from_erddap_csv(RAW / "etopo_poland.csv")
    strategies = real_world_strategies(NaturalEarth.load(RAW), elevation, SEED)
    grid = HexGrid.covering(BOUNDS.width_km, BOUNDS.height_km, KM_PER_CELL)
    sources = (ETOPO_SOURCE, NATURAL_EARTH_SOURCE)
    tile_map = generate_map("Poland", BOUNDS, grid, strategies, sources)
    path = maps.map_file("poland", "hex")
    tile_map.save(path)
    print(f"wrote {path.name}: {grid.rows} x {grid.cols} hex cells; {_summary(tile_map)}")


if __name__ == "__main__":
    main()
