"""Maps that ship with the game, baked by the scripts in tools/."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from railroads.mapgen.world import WorldLand
from railroads.sim.geo import Bounds
from railroads.sim.grids import GridKind
from railroads.sim.tilemap import TileMap

MAPS_DIR = Path(__file__).parent
WORLD_LAND_FILE = MAPS_DIR / "world-land.npz"


@dataclass(frozen=True)
class StartMap:
    """A real place a new game can start on, baked once into MAPS_DIR. Its lon/lat box can be
    any shape, so a long country gets a long map."""

    key: str
    title: str
    bounds: Bounds
    about: str


START_MAPS = (
    StartMap(
        "poland",
        "Poland",
        Bounds(west=14.08, east=24.19, south=48.97, north=54.87),
        "The whole country, from the Baltic coast to the Tatra mountains.",
    ),
    StartMap(
        "germany",
        "Germany",
        Bounds(west=5.8, east=15.1, south=47.3, north=55.1),
        "From the North Sea and the Baltic down the Rhine to the Alps.",
    ),
    StartMap(
        "france",
        "France: Paris to Lyon",
        Bounds(west=-1.8, east=7.6, south=44.6, north=50.9),
        "Northern and central France: Paris, Normandy, the Loire, Alsace and the Rhône at Lyon.",
    ),
    StartMap(
        "britain",
        "Great Britain",
        Bounds(west=-6.4, east=2.0, south=49.9, north=58.7),
        "England, Wales and Scotland up to the Highlands.",
    ),
    StartMap(
        "italy",
        "Italy: Venice to Rome",
        Bounds(west=8.6, east=15.6, south=40.9, north=46.6),
        "The Po plain and Venice's lagoon, across the Apennines to Florence and Rome.",
    ),
    StartMap(
        "moscow",
        "Moscow",
        Bounds(west=33.4, east=41.8, south=53.5, north=58.0),
        "The flat, forested land around Moscow, with the upper Volga and the Oka.",
    ),
    StartMap(
        "constantinople",
        "Constantinople",
        Bounds(west=25.8, east=32.2, south=38.8, north=42.6),
        "The Bosporus and the Sea of Marmara, between Europe and Asia.",
    ),
    StartMap(
        "alps",
        "The Alps",
        Bounds(west=5.6, east=14.0, south=45.0, north=48.2),
        "Switzerland and Austria: high mountains with a few passes through.",
    ),
    StartMap(
        "norway",
        "Norwegian fjords",
        Bounds(west=4.6, east=11.4, south=59.0, north=63.4),
        "Bergen, Oslo and the fjords cutting deep between the mountains.",
    ),
    StartMap(
        "us_northeast",
        "US Northeast",
        Bounds(west=-78.6, east=-69.8, south=38.6, north=43.2),
        "From Washington through Philadelphia and New York to Boston.",
    ),
    StartMap(
        "california",
        "California",
        Bounds(west=-123.6, east=-117.0, south=33.6, north=40.4),
        "San Francisco Bay, the Central Valley, the Sierra Nevada and Los Angeles.",
    ),
    StartMap(
        "colorado",
        "Colorado Rockies",
        Bounds(west=-108.6, east=-102.4, south=37.0, north=41.0),
        "The Great Plains running up against the Front Range at Denver.",
    ),
    StartMap(
        "chile",
        "Central Chile",
        Bounds(west=-73.8, east=-69.6, south=-38.2, north=-30.0),
        "A thin strip of land between the Pacific and the Andes, with Santiago.",
    ),
    StartMap(
        "japan",
        "Japan: Hiroshima to Tokyo",
        Bounds(west=131.8, east=141.2, south=33.2, north=37.2),
        "Honshu's Pacific coast from Hiroshima past Osaka and Nagoya to Tokyo.",
    ),
    StartMap(
        "east_china",
        "China: the Yangtze delta",
        Bounds(west=116.0, east=122.4, south=28.6, north=33.4),
        "Shanghai, Nanjing and Hangzhou on the delta, with hills to the south.",
    ),
    StartMap(
        "southeast_australia",
        "Southeast Australia",
        Bounds(west=144.0, east=153.4, south=-38.6, north=-32.6),
        "Melbourne, Canberra and Sydney, along the Great Dividing Range.",
    ),
)


def map_file(name: str, grid: GridKind) -> Path:
    return MAPS_DIR / f"{name}-{grid}.npz"


def load(name: str, grid: GridKind) -> TileMap:
    return TileMap.load(map_file(name, grid))


def start_map(key_or_title: str) -> StartMap | None:
    """The start map with this key or menu title, if there is one."""
    return next((m for m in START_MAPS if key_or_title in (m.key, m.title)), None)


@cache
def world_land() -> WorldLand:
    return WorldLand.load(WORLD_LAND_FILE)
