"""Bake the start maps of real places (railroads.maps.START_MAPS), in hexagons, into
src/railroads/maps/, so they play without the elevation data, as in the browser.

Inputs: data/elevation.sqlite (tools/build_elevation_db.py) and the Natural Earth files in
data/raw/ (download commands in README.md). Poland is baked from its own ETOPO1 grid by
tools/build_poland_map.py instead.

Run with: uv run python tools/build_start_maps.py [key ...]   (all but Poland without keys)
"""

import sys

from build_poland_map import KM_PER_CELL, RAW, SEED, summary

from railroads import maps
from railroads.mapgen.generate import generate_map
from railroads.mapgen.presets import NATURAL_EARTH_SOURCE, NaturalEarth, real_world_strategies
from railroads.mapgen.terrain_tiles import SOURCE as TERRAIN_TILES_SOURCE
from railroads.mapgen.terrain_tiles import ElevationDatabase
from railroads.newgame import ELEVATION_DB
from railroads.sim.grids import HexGrid


def bake(start: maps.StartMap, world: NaturalEarth, database: ElevationDatabase) -> None:
    bounds = start.bounds
    strategies = real_world_strategies(world, database.elevation_for(bounds), SEED)
    grid = HexGrid.covering(bounds.width_km, bounds.height_km, KM_PER_CELL)
    sources = (TERRAIN_TILES_SOURCE, NATURAL_EARTH_SOURCE)
    tile_map = generate_map(start.title, bounds, grid, strategies, sources)
    path = maps.map_file(start.key, "hex")
    tile_map.save(path)
    print(
        f"wrote {path.name} ({path.stat().st_size / 1000:.0f} KB): "
        f"{grid.cols} x {grid.rows} hex cells; {summary(tile_map)}"
    )


def main(keys: list[str]) -> None:
    wanted = [m for m in maps.START_MAPS if m.key != "poland" and (not keys or m.key in keys)]
    unknown = set(keys) - {m.key for m in wanted}
    if unknown:
        raise SystemExit(f"no start map {', '.join(sorted(unknown))} to bake here")
    world = NaturalEarth.load(RAW)
    database = ElevationDatabase(ELEVATION_DB)
    try:
        for start in wanted:
            bake(start, world, database)
    finally:
        database.close()


if __name__ == "__main__":
    main(sys.argv[1:])
