"""Bake the start maps of real places (railroads.maps.START_MAPS), in hexagons, into
src/railroads/maps/, so they play without the elevation data, as in the browser.

Inputs: data/elevation.sqlite (tools/build_elevation_db.py), the Natural Earth files in
data/raw/ (download commands in README.md) and the minor rivers from world/streams/
(tools/build_world_streams.py). The made-up map of every land is baked by
tools/build_lands_map.py instead.

Run with: uv run python tools/build_start_maps.py [key ...]   (all without keys)
"""

import sys
from collections import Counter

from railroads import maps
from railroads.mapgen import world_files
from railroads.mapgen.generate import generate_map
from railroads.mapgen.presets import NATURAL_EARTH_SOURCE, NaturalEarth, real_world_strategies
from railroads.mapgen.sources import Elevation
from railroads.mapgen.streams import HYDRORIVERS_SOURCE, StreamLines
from railroads.mapgen.terrain_tiles import SOURCE as TERRAIN_TILES_SOURCE
from railroads.mapgen.terrain_tiles import ElevationDatabase
from railroads.newgame import DATA_DIR, ELEVATION_DB
from railroads.sim.geo import Bounds
from railroads.sim.grids import KM_PER_CELL, HexGrid
from railroads.sim.tilemap import LAYERS, Temperature, Terrain, TileMap, Vegetation

RAW = DATA_DIR / "raw"
SEED = 7


def minor_rivers(bounds: Bounds) -> StreamLines:
    """The minor rivers around the bounds, from world/streams/ (tools/build_world_streams.py)."""
    world = world_files.world()
    if world is None:
        raise SystemExit("no world/ folder to read the minor rivers from")
    return world.streams(bounds)


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


def bake(start: maps.StartMap, world: NaturalEarth, elevation: Elevation) -> None:
    """Bakes a start map from the elevation and Natural Earth data and says what it holds."""
    bounds = start.bounds
    strategies = real_world_strategies(world, elevation, SEED, minor_rivers(bounds))
    grid = HexGrid.covering(bounds.width_km, bounds.height_km, KM_PER_CELL)
    sources = (TERRAIN_TILES_SOURCE, NATURAL_EARTH_SOURCE, HYDRORIVERS_SOURCE)
    tile_map = generate_map(start.title, bounds, grid, strategies, sources)
    path = maps.map_file(start.key, "hex")
    tile_map.save(path)
    print(
        f"wrote {path.name} ({path.stat().st_size / 1000:.0f} KB): "
        f"{grid.cols} x {grid.rows} hex cells; {summary(tile_map)}"
    )


def main(keys: list[str]) -> None:
    wanted = [m for m in maps.START_MAPS if m.key != "lands" and (not keys or m.key in keys)]
    unknown = set(keys) - {m.key for m in wanted}
    if unknown:
        raise SystemExit(f"no start map {', '.join(sorted(unknown))} to bake here")
    world = NaturalEarth.load(RAW)
    with ElevationDatabase(ELEVATION_DB) as database:
        for start in wanted:
            bake(start, world, database.elevation_for(start.bounds))


if __name__ == "__main__":
    main(sys.argv[1:])
