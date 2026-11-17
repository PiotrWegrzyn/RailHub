"""Bake the Poland map, in hexagons, into src/railroads/maps/.

Inputs in data/raw/ (download commands in README.md):
- etopo_poland.csv: NOAA ETOPO1 elevation, 1 arc-minute grid.
- ne_10m_*.geojson: Natural Earth sea, lakes, rivers and geography regions (for deserts).

Run with: uv run python tools/build_poland_map.py
"""

from collections import Counter

from railroads import maps
from railroads.mapgen.generate import generate_map
from railroads.mapgen.presets import (
    ETOPO_SOURCE,
    NATURAL_EARTH_SOURCE,
    NaturalEarth,
    real_world_strategies,
)
from railroads.mapgen.sources import Elevation, ElevationGrid
from railroads.newgame import DATA_DIR
from railroads.sim.grids import HexGrid
from railroads.sim.tilemap import LAYERS, Temperature, Terrain, TileMap, Vegetation

RAW = DATA_DIR / "raw"
POLAND = maps.start_map("poland")
KM_PER_CELL = 2.0
SEED = 7


def summary(tile_map: TileMap) -> str:
    kinds = {"terrain": Terrain, "vegetation": Vegetation, "temperature": Temperature}
    parts = []
    for layer in LAYERS:
        if layer in kinds:
            counts = Counter(getattr(tile_map, layer).ravel().tolist())
            parts.append(
                ", ".join(f"{kinds[layer](k).name.lower()} {n}" for k, n in sorted(counts.items()))
            )
    return "; ".join(parts)


def bake(
    start: maps.StartMap, world: NaturalEarth, elevation: Elevation, elevation_source: str
) -> None:
    """Bakes a start map from the elevation and Natural Earth data and says what it holds."""
    bounds = start.bounds
    strategies = real_world_strategies(world, elevation, SEED)
    grid = HexGrid.covering(bounds.width_km, bounds.height_km, KM_PER_CELL)
    sources = (elevation_source, NATURAL_EARTH_SOURCE)
    tile_map = generate_map(start.title, bounds, grid, strategies, sources)
    path = maps.map_file(start.key, "hex")
    tile_map.save(path)
    print(
        f"wrote {path.name} ({path.stat().st_size / 1000:.0f} KB): "
        f"{grid.cols} x {grid.rows} hex cells; {summary(tile_map)}"
    )


def main() -> None:
    assert POLAND is not None
    elevation = ElevationGrid.from_erddap_csv(RAW / "etopo_poland.csv")
    bake(POLAND, NaturalEarth.load(RAW), elevation, ETOPO_SOURCE)


if __name__ == "__main__":
    main()
