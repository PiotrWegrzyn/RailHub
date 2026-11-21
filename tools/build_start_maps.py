"""Bake the start maps of real places (railroads.maps.START_MAPS), in hexagons, into
src/railroads/maps/, so they play without the elevation data, as in the browser.

Inputs: data/elevation.sqlite (tools/build_elevation_db.py) and the Natural Earth files in
data/raw/ (download commands in README.md). Poland is baked from its own ETOPO1 grid by
tools/build_poland_map.py instead, and the made-up map of every land by
tools/build_lands_map.py.

Run with: uv run python tools/build_start_maps.py [key ...]   (all but Poland without keys)
"""

import sys

from build_poland_map import RAW, bake

from railroads import maps
from railroads.mapgen.presets import NaturalEarth
from railroads.mapgen.terrain_tiles import SOURCE as TERRAIN_TILES_SOURCE
from railroads.mapgen.terrain_tiles import ElevationDatabase
from railroads.newgame import ELEVATION_DB


def main(keys: list[str]) -> None:
    wanted = [
        m
        for m in maps.START_MAPS
        if m.key not in ("poland", "lands") and (not keys or m.key in keys)
    ]
    unknown = set(keys) - {m.key for m in wanted}
    if unknown:
        raise SystemExit(f"no start map {', '.join(sorted(unknown))} to bake here")
    world = NaturalEarth.load(RAW)
    with ElevationDatabase(ELEVATION_DB) as database:
        for start in wanted:
            bake(start, world, database.elevation_for(start.bounds), TERRAIN_TILES_SOURCE)


if __name__ == "__main__":
    main(sys.argv[1:])
