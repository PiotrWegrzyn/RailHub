"""Start a new game on a map of a random, mostly-land spot in the world."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from railroads.mapgen.generate import generate_map
from railroads.mapgen.presets import NATURAL_EARTH_SOURCE, NaturalEarth, real_world_strategies
from railroads.mapgen.start import StartRegion, pick_start_bounds
from railroads.mapgen.terrain_tiles import SOURCE as TERRAIN_TILES_SOURCE
from railroads.mapgen.terrain_tiles import ElevationDatabase
from railroads.sim.geo import Bounds, describe_lonlat
from railroads.sim.grids import HexGrid
from railroads.sim.tilemap import TileMap

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ELEVATION_DB = DATA_DIR / "elevation.sqlite"
START_REGION = StartRegion()
MAP_WIDTH_KM = 537.0
MAP_HEIGHT_KM = 537.0
KM_PER_CELL = 2.0


@dataclass(frozen=True)
class NewGame:
    seed: int
    bounds: Bounds
    tile_map: TileMap

    @property
    def place(self) -> str:
        return describe_lonlat(*self.bounds.center)


def start_new_game(
    seed: int, region: StartRegion = START_REGION, report: Callable[[str], None] = print
) -> NewGame:
    if not ELEVATION_DB.exists():
        raise FileNotFoundError(
            f"No elevation database at {ELEVATION_DB}. "
            "Build it once with: uv run python tools/build_elevation_db.py"
        )
    rng = np.random.default_rng(seed)
    report("Loading world data")
    world = NaturalEarth.load(DATA_DIR / "raw")
    bounds = pick_start_bounds(rng, region, MAP_WIDTH_KM, MAP_HEIGHT_KM, world.ocean + world.lakes)
    report(f"Starting at {describe_lonlat(*bounds.center)}")
    database = ElevationDatabase(ELEVATION_DB)
    try:
        elevation = database.elevation_for(bounds)
    finally:
        database.close()
    strategies = real_world_strategies(world, elevation, seed)
    report("Building the map")
    tile_map = generate_map(
        f"Seed {seed}",
        bounds,
        HexGrid.covering(bounds.width_km, bounds.height_km, KM_PER_CELL),
        strategies,
        (TERRAIN_TILES_SOURCE, NATURAL_EARTH_SOURCE),
    )
    return NewGame(seed, bounds, tile_map)
