"""Start a new game: on the baked map of Poland or on a map of a random, mostly-land spot in the
world, with the towns grown from HISTORY_START to the year play begins."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from railroads import maps
from railroads.mapgen.generate import generate_map
from railroads.mapgen.presets import NATURAL_EARTH_SOURCE, NaturalEarth, real_world_strategies
from railroads.mapgen.seeding import play_begins, seed_buildings
from railroads.mapgen.start import StartRegion, pick_start_bounds
from railroads.mapgen.terrain_tiles import SOURCE as TERRAIN_TILES_SOURCE
from railroads.mapgen.terrain_tiles import ElevationDatabase
from railroads.saves import SavedGame
from railroads.sim.clock import END_YEAR, HISTORY_START, START_YEAR, GameClock
from railroads.sim.finance import START_CASH
from railroads.sim.game import Game, place_names
from railroads.sim.geo import Bounds, describe_lonlat
from railroads.sim.grids import HexGrid
from railroads.sim.novelties import Novelties
from railroads.sim.stats import History
from railroads.sim.tilemap import TileMap

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ELEVATION_DB = DATA_DIR / "elevation.sqlite"
START_REGION = StartRegion()
MAP_WIDTH_KM = 537.0
MAP_HEIGHT_KM = 537.0
KM_PER_CELL = 2.0


class MapChoice(StrEnum):
    POLAND = "Poland"
    WORLD = "A random place in the world"


@dataclass(frozen=True)
class Settings:
    """What a new game is set up with."""

    seed: int
    map: MapChoice = MapChoice.POLAND
    # Towns grow from HISTORY_START until this year, when play begins.
    start_year: int = START_YEAR
    cash: float = START_CASH
    # Years a stand-in player builds and runs the railway before the player takes over.
    stand_in_years: int = 0


def world_maps_available() -> bool:
    """Whether the data for maps of random places in the world is here."""
    return ELEVATION_DB.exists()


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


def make_game(settings: Settings, report: Callable[[str], None] = print) -> SavedGame:
    """A new game with these settings, its towns grown and ready to play."""
    seed = settings.seed
    if settings.map == MapChoice.POLAND:
        tile_map = maps.load("poland", "hex")
        title = f"Poland, seed {seed}"
    else:
        new = start_new_game(seed, report=report)
        tile_map = new.tile_map
        title = f"Seed {seed}, {new.place}"
    years = min(max(settings.start_year, HISTORY_START), END_YEAR - 1) - HISTORY_START
    months = years * 12
    begins = play_begins(months)
    report(f"Growing towns for {years} years, from {HISTORY_START} to {begins}")
    history = History()
    seeded = seed_buildings(tile_map, np.random.default_rng(seed), months=months, history=history)
    game = Game(seeded, start_year=begins, history=history, names=place_names(seeded.tile_map))
    game.finances.cash = settings.cash
    game.growth.novelties = Novelties.pick(game.tile_map, np.random.default_rng(seed + 3))
    if settings.stand_in_years > 0:
        played = min(settings.stand_in_years, END_YEAR - begins)
        report(f"Playing {played} years with a stand-in player")
        game.simulate(played, np.random.default_rng(seed + 2))
        title += ", simulated"
    clock = GameClock(first_year=game.growth.year)
    return SavedGame(game, clock, np.random.default_rng(seed + 1), title, settings)
