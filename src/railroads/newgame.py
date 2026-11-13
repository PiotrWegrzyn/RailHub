"""Start a new game: on one of the baked maps of real places (railroads.maps.START_MAPS) or on a
map of a random, mostly-land spot in the world, with the towns grown from HISTORY_START to the
year play begins."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import numpy as np

from railroads import maps
from railroads.mapgen import world_files
from railroads.mapgen.generate import generate_map
from railroads.mapgen.presets import NATURAL_EARTH_SOURCE, real_world_strategies
from railroads.mapgen.seeding import play_begins, seed_steps
from railroads.mapgen.start import StartRegion, pick_start_bounds
from railroads.mapgen.terrain_tiles import SOURCE as TERRAIN_TILES_SOURCE
from railroads.saves import SavedGame
from railroads.sim.clock import END_YEAR, HISTORY_START, START_YEAR, GameClock
from railroads.sim.finance import START_CASH
from railroads.sim.game import Game, place_names
from railroads.sim.geo import Bounds, describe_lonlat
from railroads.sim.grids import HexGrid
from railroads.sim.novelties import Novelties
from railroads.sim.stats import History
from railroads.sim.steps import Steps, finish
from railroads.sim.tilemap import TileMap

# The source data the tools build the world's files from (railroads.mapgen.world_files); the game
# itself reads only those.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ELEVATION_DB = DATA_DIR / "elevation.sqlite"
# Play can't begin before this year; START_YEAR is the start suggested.
# The year of the first steam engine on rails (railroads.sim.trains).
EARLIEST_START_YEAR = 1804
START_REGION = StartRegion()
MAP_WIDTH_KM = 537.0
MAP_HEIGHT_KM = 537.0
KM_PER_CELL = 2.0


class MapChoice(StrEnum):
    """Each baked map by its title in railroads.maps.START_MAPS, and a random place. Saves keep
    the value, so a title once used must not change."""

    POLAND = "Poland"
    GERMANY = "Germany"
    FRANCE = "France: Paris to Lyon"
    BRITAIN = "Great Britain"
    ITALY = "Italy: Venice to Rome"
    MOSCOW = "Moscow"
    CONSTANTINOPLE = "Constantinople"
    ALPS = "The Alps"
    NORWAY = "Norwegian fjords"
    US_NORTHEAST = "US Northeast"
    CALIFORNIA = "California"
    COLORADO = "Colorado Rockies"
    CHILE = "Central Chile"
    JAPAN = "Japan: Hiroshima to Tokyo"
    EAST_CHINA = "China: the Yangtze delta"
    SOUTHEAST_AUSTRALIA = "Southeast Australia"
    WORLD = "A random place in the world"

    @property
    def start_map(self) -> maps.StartMap | None:
        """The baked map this choice plays on, or None for a random place."""
        return maps.start_map(self.value)


@dataclass(frozen=True)
class Settings:
    """What a new game is set up with."""

    seed: int
    map: MapChoice = MapChoice.POLAND
    # Towns grow from HISTORY_START until this year, when play begins (see allowed_start_year).
    start_year: int = START_YEAR
    cash: float = START_CASH
    # Years a stand-in player builds and runs the railway before the player takes over.
    stand_in_years: int = 0


def allowed_start_year(year: int) -> int:
    """The nearest year play can begin in: from EARLIEST_START_YEAR to the year before
    END_YEAR."""
    return min(max(year, EARLIEST_START_YEAR), END_YEAR - 1)


def world_maps_available() -> bool:
    """Whether the data for maps of random places in the world is here, or can be downloaded."""
    return world_files.world() is not None


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
    world = world_files.world()
    if world is None:
        raise FileNotFoundError(
            f"No world data at {world_files.WORLD_DIR}. "
            "Build it with: uv run python tools/build_world_files.py"
        )
    rng = np.random.default_rng(seed)
    bounds = pick_start_bounds(
        rng, region, MAP_WIDTH_KM, MAP_HEIGHT_KM, maps.world_land().land_share
    )
    report(f"Starting at {describe_lonlat(*bounds.center)}")
    report("Loading the land there")
    shapes = world.natural_earth(bounds)
    elevation = world.elevation_for(bounds)
    strategies = real_world_strategies(shapes, elevation, seed)
    report("Building the map")
    tile_map = generate_map(
        f"Seed {seed}",
        bounds,
        HexGrid.covering(bounds.width_km, bounds.height_km, KM_PER_CELL),
        strategies,
        (TERRAIN_TILES_SOURCE, NATURAL_EARTH_SOURCE),
    )
    return NewGame(seed, bounds, tile_map)


def make_game(
    settings: Settings,
    report: Callable[[str], None] = print,
    progress: Callable[[float], None] = lambda done: None,
) -> SavedGame:
    """A new game with these settings, its towns grown and ready to play, beginning in the
    nearest year play can begin in; `progress` hears the share of the towns' history grown."""
    return finish(make_game_steps(settings, report, progress))


def make_game_steps(
    settings: Settings,
    report: Callable[[str], None] = print,
    progress: Callable[[float], None] = lambda done: None,
) -> Steps[SavedGame]:
    """make_game in steps, telling `progress` the share of the towns' history grown so far."""
    settings = replace(settings, start_year=allowed_start_year(settings.start_year))
    seed = settings.seed
    baked = settings.map.start_map
    if baked is not None:
        tile_map = maps.load(baked.key, "hex")
        title = f"{baked.title}, seed {seed}"
    else:
        new = start_new_game(seed, report=report)
        tile_map = new.tile_map
        title = f"Seed {seed}, {new.place}"
    years = settings.start_year - HISTORY_START
    months = years * 12
    begins = play_begins(months)
    report(f"Generating map from year {HISTORY_START} to {begins}")
    history = History()
    seeded = yield from seed_steps(
        tile_map, np.random.default_rng(seed), months=months, history=history, progress=progress
    )
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
