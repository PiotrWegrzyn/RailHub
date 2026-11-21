"""Maps that ship with the game, baked by the scripts in tools/."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from railroads.mapgen import lands
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


# The made-up map of every land (railroads.mapgen.lands), baked by tools/build_lands_map.py.
LANDS_TITLE = "Every land (test map)"
START_MAPS = (
    StartMap(
        "poland",
        "Poland",
        Bounds(west=14.08, east=24.19, south=48.97, north=54.87),
        "Central Poland: Warsaw, Łódź, Poznań and Wrocław on the plain of the Vistula and the Oder.",
    ),
    StartMap(
        "germany",
        "Germany",
        Bounds(west=5.8, east=15.1, south=47.3, north=55.1),
        "Central Germany: Frankfurt, Hanover, Leipzig and Magdeburg, with the Harz between them.",
    ),
    StartMap(
        "france",
        "France: Paris and the Loire",
        Bounds(west=-1.8, east=7.6, south=44.6, north=50.9),
        "Paris and its basin, east to Reims and south to the Loire at Orléans and Tours.",
    ),
    StartMap(
        "britain",
        "Great Britain",
        Bounds(west=-6.4, east=2.0, south=49.9, north=58.7),
        "Northern England and southern Scotland: Manchester, Leeds, Glasgow and Edinburgh.",
    ),
    StartMap(
        "italy",
        "Italy: Bologna and Florence",
        Bounds(west=8.6, east=15.6, south=40.9, north=46.6),
        "Bologna on the Po plain, across the Apennines to Florence and the Adriatic at Rimini.",
    ),
    StartMap(
        "moscow",
        "Moscow",
        Bounds(west=33.4, east=41.8, south=53.5, north=58.0),
        "Moscow and the flat, forested land around it, with the upper Volga at Tver.",
    ),
    StartMap(
        "constantinople",
        "Constantinople",
        Bounds(west=25.8, east=32.2, south=38.8, north=42.6),
        "The Bosporus and the Sea of Marmara, between Europe and Asia, with Bursa to the south.",
    ),
    StartMap(
        "alps",
        "The Alps",
        Bounds(west=5.6, east=14.0, south=45.0, north=48.2),
        "From Zürich over the high Alps to Lake Como, Bergamo and Trento: a few passes through.",
    ),
    StartMap(
        "norway",
        "Norwegian fjords",
        Bounds(west=4.6, east=11.4, south=59.0, north=63.4),
        "The Sognefjord cutting deep into the mountains, with small villages along the water.",
    ),
    StartMap(
        "us_northeast",
        "US Northeast",
        Bounds(west=-78.6, east=-69.8, south=38.6, north=43.2),
        "New York and Philadelphia, with New Jersey between them and the Hudson valley to the north.",
    ),
    StartMap(
        "california",
        "California",
        Bounds(west=-123.6, east=-117.0, south=33.6, north=40.4),
        "San Jose at the foot of the Bay, across the Central Valley to the Sierra Nevada.",
    ),
    StartMap(
        "colorado",
        "Colorado Rockies",
        Bounds(west=-108.6, east=-102.4, south=37.0, north=41.0),
        "Denver and Colorado Springs, where the Great Plains run up against the Front Range.",
    ),
    StartMap(
        "chile",
        "Central Chile",
        Bounds(west=-73.8, east=-69.6, south=-38.2, north=-30.0),
        "Santiago and Valparaíso, between the Pacific and the Andes.",
    ),
    StartMap(
        "japan",
        "Japan: Osaka to Nagoya",
        Bounds(west=131.8, east=141.2, south=33.2, north=37.2),
        "Osaka, Kobe and Kyoto by Lake Biwa, east to Nagoya, with the Kii mountains to the south.",
    ),
    StartMap(
        "east_china",
        "China: the Yangtze delta",
        Bounds(west=116.0, east=122.4, south=28.6, north=33.4),
        "Nanjing, Suzhou and Hangzhou around Lake Tai on the delta, with hills to the south.",
    ),
    StartMap(
        "lands",
        LANDS_TITLE,
        lands.BOUNDS,
        "A made-up map for testing: ice, tundra, taiga, steppe, farmland, woods, desert, savanna "
        "and jungle side by side, each with plains, hills, mountains, a lake, a bay and rivers.",
    ),
    StartMap(
        "southeast_australia",
        "Southeast Australia",
        Bounds(west=144.0, east=153.4, south=-38.6, north=-32.6),
        "Canberra and the Snowy Mountains on the Great Dividing Range, down to the coast.",
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
