"""The map as pixel art, drawn a square chunk at a time.

The view never turns, so every point of the map has a fixed place in one big picture, at one art
pixel per screen pixel when PIXEL_CELL_WIDTH pixels make a cell. That picture is far too big to
hold at once, so it is drawn in CHUNK x CHUNK pieces as the view reaches them: first the ground,
every pixel coloured from its cell with a little texture, then everything standing on it (trees,
rocks, buildings) from back to front. Textures come from a hash of the pixel's place, so chunks
meet without seams and look the same each time they are drawn.
"""

import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from railroads.sim.constructions import BuildingKind, Constructions, Track, TrackWidth
from railroads.sim.economy import Satisfaction
from railroads.sim.grids import Cell, Grid
from railroads.sim.tilemap import Deposit, Relief, Terrain, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.ui.biomes import Biome, biome_grid
from railroads.ui.buildings import BuildingSpots, Plot, plots_of
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, TILT
from railroads.ui.pixel.art import HILL, MOUNTAIN, TINT_GREYS
from railroads.ui.pixel.sheet import FINE, SpriteSheet

CHUNK = 256
# How far in art pixels a sprite can reach from where it stands.
SPRITE_REACH = 48
# Room left above the map's top corner for the tallest sprites, in art pixels.
TOP_MARGIN = 48
# Ground is worked out this far past a chunk's edges, so shores and bevels meet across chunks.
APRON = 2

type RGB = tuple[int, int, int]
type Point = tuple[float, float]
type Path = list[Point]
type ToArt = Callable[[Path], Path]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Rgba = npt.NDArray[np.uint8]

# Four shades per biome, dark to light; the texture picks between them pixel by pixel.
RAMPS: dict[Biome, list[RGB]] = {
    Biome.SEA: [(26, 56, 110), (34, 72, 136), (44, 90, 160), (110, 150, 200)],
    Biome.LAKE: [(30, 70, 130), (40, 88, 156), (52, 106, 178), (120, 164, 214)],
    Biome.RIVER: [(30, 70, 130), (40, 88, 156), (52, 106, 178), (120, 164, 214)],
    Biome.SNOW: [(176, 188, 204), (206, 214, 226), (228, 234, 242), (250, 252, 255)],
    Biome.TUNDRA: [(130, 132, 100), (150, 152, 116), (168, 170, 132), (188, 190, 152)],
    Biome.TAIGA: [(36, 64, 44), (46, 78, 54), (58, 92, 64), (76, 110, 80)],
    Biome.BADLANDS: [(150, 126, 88), (170, 146, 104), (188, 164, 120), (206, 184, 140)],
    Biome.GRASSLAND: [(70, 116, 42), (88, 138, 50), (106, 156, 58), (134, 178, 74)],
    Biome.FOREST: [(36, 70, 30), (46, 86, 36), (58, 102, 42), (76, 120, 50)],
    Biome.DESERT: [(186, 156, 98), (206, 176, 114), (222, 194, 132), (238, 214, 156)],
    Biome.SAVANNA: [(150, 144, 70), (170, 162, 82), (188, 180, 96), (206, 198, 116)],
    Biome.JUNGLE: [(22, 70, 32), (30, 86, 40), (40, 102, 48), (56, 120, 60)],
    Biome.MOUNTAIN: [(86, 84, 82), (110, 106, 100), (136, 132, 124), (172, 168, 158)],
}
WATER = (Biome.SEA, Biome.LAKE, Biome.RIVER)
SAND: RGB = (200, 184, 128)
FOAM: RGB = (110, 150, 200)
# Field colours for farms, in pairs of furrow and ridge.
FIELDS: list[tuple[RGB, RGB]] = [
    ((150, 108, 64), (120, 84, 48)),
    ((214, 186, 90), (186, 156, 70)),
    ((140, 164, 70), (112, 138, 56)),
]
FIELD_STRIPES = 9


# Groves of trees in rows over the ground of the cell: the ground, the trees, the colours of the
# fruit some of them bear, how many rows a cell each way and how big each crown is.
@dataclass(frozen=True)
class Grove:
    ground: RGB
    tree: RGB
    fruit: tuple[RGB, ...]
    rows: int
    crown: float


GROVES: dict[BuildingKind, Grove] = {
    BuildingKind.ORCHARD: Grove(
        (146, 178, 84), (58, 104, 46), ((204, 60, 46), (226, 150, 52)), 6, 0.32
    ),
    BuildingKind.DATE_PALMS: Grove((214, 192, 134), (78, 122, 56), ((110, 62, 36),), 5, 0.3),
    BuildingKind.BANANAS: Grove((76, 122, 50), (44, 100, 38), ((236, 206, 64),), 7, 0.4),
    BuildingKind.ORANGE_FARM: Grove((150, 176, 84), (50, 104, 44), ((244, 142, 36),), 6, 0.36),
    BuildingKind.VINEYARD: Grove((150, 150, 90), (70, 110, 50), ((104, 50, 124),), 8, 0.3),
    BuildingKind.COFFEE_PLANTATION: Grove((96, 120, 62), (40, 86, 44), ((190, 52, 44),), 7, 0.36),
    BuildingKind.RUBBER_PLANTATION: Grove((70, 104, 56), (36, 72, 40), ((226, 224, 210),), 5, 0.38),
    BuildingKind.SHEEP_FARM: Grove((140, 182, 86), (238, 234, 222), ((60, 56, 54),), 4, 0.2),
    BuildingKind.CATTLE_RANCH: Grove((132, 176, 80), (120, 78, 48), ((236, 232, 224),), 4, 0.24),
    BuildingKind.HOP_GARDEN: Grove((150, 140, 96), (72, 132, 50), ((170, 214, 110),), 7, 0.34),
}
# Crops in rows, like a farm's furrows.
CROP_ROWS: dict[BuildingKind, tuple[RGB, RGB]] = {
    BuildingKind.SUGAR: ((156, 200, 100), (118, 162, 74)),
    BuildingKind.COTTON_FIELD: ((238, 236, 226), (150, 172, 100)),
    BuildingKind.POTATO_FARM: ((150, 120, 80), (96, 140, 64)),
    BuildingKind.FLAX_FIELD: ((140, 166, 226), (130, 160, 90)),
}
FIELD_KINDS = (BuildingKind.FARM, *CROP_ROWS, *GROVES)
# The specks showing each deposit on the ground, in the order of Deposit.
DEPOSIT_SPECKS = np.array(
    [
        (0, 0, 0),
        (172, 96, 70),
        (206, 130, 76),
        (255, 214, 60),
        (196, 240, 255),
        (24, 22, 28),
        (240, 240, 244),
        (120, 176, 210),
        (236, 230, 212),
        (116, 58, 136),
        (248, 248, 244),
        (92, 58, 38),
        (58, 62, 54),
        (0, 0, 0),
        (226, 196, 128),
    ],
    dtype=np.float64,
)
DEPOSIT_SPECK_SHARE = 0.08
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 - 0.5
HILL_LIFT = 0.3
# Slopes are lit from the top left of the screen, from the land's real heights. Heights are
# smoothed over a raster this many steps per cell wide, so slopes run across cell edges.
RELIEF_STEPS_PER_CELL = 4
RELIEF_SMOOTHING_CELLS = 0.75
# Shades of brightening per unit of grade facing the light (0.05 is a 5% grade), and the most
# a slope can brighten or darken.
SLOPE_LIGHT = 16.0
MAX_SLOPE_SHADES = 1.5
EDGE_SHADE = 0.94
# Track, in cells: how wide the gravel bed and a bridge deck are, the gap between the rails, how
# long and wide a sleeper is and how far apart sleepers lie, and a tunnel portal's width. The two
# tracks of double track lie DOUBLE_SPACING apart, and every piece rounds its turn over the last
# BEND either side of the middle of its cell.
TRACK_BED = 0.3
BRIDGE_WIDTH = 0.32
GAUGE = 0.15
SLEEPER_LENGTH = 0.25
SLEEPER_WIDTH = 0.045
SLEEPER_GAP = 0.125
PORTAL = 0.32
DOUBLE_SPACING = 0.28
BEND = 0.2
# Overhead wires on electric track: how far beside the track their masts stand and how far
# apart, in cells, and how tall the masts are and how high the wires hang, in art pixels.
MAST_SIDE = 0.16
MAST_GAP = 1.0
MAST_HEIGHT = 7
WIRE_HEIGHT = 5
BALLAST: RGB = (118, 110, 98)
SLEEPER: RGB = (84, 60, 40)
RAIL: RGB = (178, 176, 172)
BRIDGE_DECK: RGB = (126, 86, 50)
BRIDGE_SHADOW: RGB = (62, 42, 26)
RAILING: RGB = (70, 48, 30)
PORTAL_STONE: RGB = (118, 112, 104)
PORTAL_DARK: RGB = (20, 18, 18)
MAST: RGB = (96, 96, 104)
WIRE: RGB = (40, 40, 46)
# Land whose buildings are content or thriving is washed with a warm glow: how strongly, by
# their satisfaction.
GLOW: RGB = (255, 214, 110)
SERVED_GLOW = {
    Satisfaction.UNSATISFIED: 0.0,
    Satisfaction.CONTENT: 0.12,
    Satisfaction.THRIVING: 0.24,
}
BEVEL_LIGHT = 1.18
BEVEL_SHADE = 0.78

# What grows on each biome: how many trees per cell (a fraction is a chance of one) and which
# kinds, by sprite name prefix.
TREES: dict[Biome, tuple[float, tuple[str, ...]]] = {
    Biome.FOREST: (5, ("pine", "leafy", "leafy")),
    Biome.TAIGA: (5, ("pine",)),
    Biome.JUNGLE: (6, ("leafy",)),
    Biome.GRASSLAND: (0.06, ("leafy",)),
    Biome.SAVANNA: (0.04, ("leafy",)),
    Biome.TUNDRA: (0.03, ("pine",)),
}
ROCKS_ON = (Biome.MOUNTAIN, Biome.SNOW)
# Sprites raised on hill and mountain cells: which kind, and the chance of each of a cell's
# places holding one.
RELIEF_SPRITES = {
    Relief.HILLS: (HILL, (0.55, 0.35)),
    Relief.MOUNTAINS: (MOUNTAIN, (0.95, 0.6)),
}
RELIEF_SPREAD = 0.3
TINT_OUTLINE = 0.72
TINT_HIGHLIGHT = 1.12
ROCK_CHANCE = 0.4
TREE_SPREAD = 0.4
# Buildings drawn in one of several looks, picked by sprite name prefix; others by one name.
VARIED_SPRITES = {
    BuildingKind.HOUSE: "house",
    BuildingKind.TENEMENT: "tenement",
    BuildingKind.FLATS: "flats",
}
BUILDING_SPRITES = {
    BuildingKind.FARM: "farm",
    BuildingKind.MINE: "mine",
    BuildingKind.FACTORY: "factory",
    BuildingKind.BIG_FACTORY: "big_factory",
    BuildingKind.MEGA_FACTORY: "mega_factory",
    BuildingKind.SKYSCRAPER: "skyscraper",
    BuildingKind.ORCHARD: "orchard",
    BuildingKind.IRON_MINE: "iron_mine",
    BuildingKind.COPPER_MINE: "copper_mine",
    BuildingKind.GOLD_MINE: "gold_mine",
    BuildingKind.DIAMOND_MINE: "diamond_mine",
    BuildingKind.OIL_WELL: "oil_well",
    BuildingKind.SALT_WORKS: "salt_works",
    BuildingKind.FISHERY: "fishery",
    BuildingKind.SHEEP_FARM: "sheep_farm",
    BuildingKind.COTTON_FIELD: "cotton_field",
    BuildingKind.VINEYARD: "vineyard",
    BuildingKind.COFFEE_PLANTATION: "coffee_plantation",
    BuildingKind.RUBBER_PLANTATION: "rubber_plantation",
    BuildingKind.TEXTILE_MILL: "textile_mill",
    BuildingKind.FURNITURE_MAKER: "furniture_maker",
    BuildingKind.TOOL_WORKS: "tool_works",
    BuildingKind.WINERY: "winery",
    BuildingKind.REFINERY: "refinery",
    BuildingKind.CHEMICAL_WORKS: "chemical_works",
    BuildingKind.ELECTRONICS_PLANT: "electronics_plant",
    BuildingKind.CAR_FACTORY: "car_factory",
    BuildingKind.CAR_DEALER: "car_dealer",
    BuildingKind.JEWELLER: "jeweller",
    BuildingKind.ORANGE_FARM: "orange_farm",
    BuildingKind.OUTPOST: "outpost",
    BuildingKind.BASE: "base",
    BuildingKind.HEADQUARTERS: "headquarters",
    BuildingKind.GUN_FACTORY: "gun_factory",
    BuildingKind.TANK_FACTORY: "tank_factory",
    BuildingKind.PLANE_FACTORY: "plane_factory",
    BuildingKind.AIRPORT: "airport",
    BuildingKind.BARREL_MAKER: "barrel_maker",
    BuildingKind.BEEKEEPER: "beekeeper",
    BuildingKind.INCENSE_GROVE: "incense_grove",
    BuildingKind.CHURCH: "church",
    BuildingKind.BASILICA: "basilica",
    BuildingKind.CATHEDRAL: "cathedral",
    BuildingKind.ST_PAULS: "st_pauls",
    BuildingKind.DATE_PALMS: "date_palms",
    BuildingKind.BANANAS: "bananas",
    BuildingKind.SUGAR: "sugar",
    BuildingKind.FOOD_PLANT: "food_plant",
    BuildingKind.BAKERY: "bakery",
    BuildingKind.STEEL_MILL: "steel_mill",
    BuildingKind.SHIPYARD: "shipyard",
    BuildingKind.LUMBER_MILL: "lumber_mill",
    BuildingKind.PORT: "port",
    BuildingKind.CLAY_PIT: "clay_pit",
    BuildingKind.LIMESTONE_QUARRY: "limestone_quarry",
    BuildingKind.SAND_PIT: "sand_pit",
    BuildingKind.POTATO_FARM: "potato_farm",
    BuildingKind.CATTLE_RANCH: "cattle_ranch",
    BuildingKind.FLAX_FIELD: "flax_field",
    BuildingKind.HOP_GARDEN: "hop_garden",
    BuildingKind.MACHINE_WORKS: "machine_works",
    BuildingKind.BRICKWORKS: "brickworks",
    BuildingKind.CEMENT_WORKS: "cement_works",
    BuildingKind.GLASSWORKS: "glassworks",
    BuildingKind.BREWERY: "brewery",
    BuildingKind.DISTILLERY: "distillery",
    BuildingKind.TANNERY: "tannery",
    BuildingKind.SHOE_FACTORY: "shoe_factory",
    BuildingKind.PLASTICS_WORKS: "plastics_works",
    BuildingKind.APPLIANCE_FACTORY: "appliance_factory",
    BuildingKind.PAPER_MILL: "paper_mill",
    BuildingKind.PRINTING_WORKS: "printing_works",
    BuildingKind.SUGAR_REFINERY: "sugar_refinery",
    BuildingKind.PORCELAIN_WORKS: "porcelain_works",
    BuildingKind.MEAT_FACTORY: "meat_factory",
    BuildingKind.POWER_PLANT: "power_plant",
    BuildingKind.SPACE_CENTER: "space_center",
    BuildingKind.PEAR_HQ: "pear_hq",
    BuildingKind.ROYAL_PALACE: "royal_palace",
    BuildingKind.SALT_CATHEDRAL: "salt_cathedral",
    BuildingKind.SPA_RESORT: "spa_resort",
    BuildingKind.WORLDS_FAIR: "worlds_fair",
    BuildingKind.FILM_STUDIO: "film_studio",
    BuildingKind.AMUSEMENT_PARK: "amusement_park",
    BuildingKind.OLYMPIC_STADIUM: "olympic_stadium",
    BuildingKind.TAVERN: "tavern",
    BuildingKind.DEPARTMENT_STORE: "department_store",
    BuildingKind.HOTEL: "hotel",
}


def station_sprite(level: int) -> str:
    """The sprite of a station raised to a level."""
    return "station" if level <= 1 else f"station_{level}"


def hash01(*keys: npt.ArrayLike) -> FloatArray:
    """A repeatable pseudo-random number from 0 to 1 for each combination of integer keys."""
    h = np.uint64(0x9E3779B97F4A7C15)
    for key in keys:
        h = h ^ np.asarray(key).astype(np.uint64)
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        h = h ^ (h >> np.uint64(31))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


@dataclass(frozen=True)
class PixelProjection:
    """Where map points land in the big picture, in art pixels from its top-left corner."""

    matrix: FloatArray
    origin: tuple[float, float]
    size: tuple[int, int]

    @classmethod
    def of(cls, grid: Grid, angle_deg: float, cell_units: float) -> Self:
        scale = PIXEL_CELL_WIDTH / cell_units
        cos, sin = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        matrix = np.array([[cos, -sin], [sin * TILT, cos * TILT]]) * scale
        width, height = grid.size()
        corners = matrix @ np.array([[0, width, width, 0], [0, 0, height, height]])
        left, top = corners.min(axis=1) - (0, TOP_MARGIN)
        right, bottom = corners.max(axis=1)
        origin = (math.floor(left), math.floor(top))
        return cls(matrix, origin, (math.ceil(right) - origin[0], math.ceil(bottom) - origin[1]))

    def to_pixels(self, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        (a, b), (c, d) = self.matrix
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return a * x + b * y - self.origin[0], c * x + d * y - self.origin[1]

    def to_plane(self, px: npt.ArrayLike, py: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        (a, b), (c, d) = np.linalg.inv(self.matrix)
        px = np.asarray(px, dtype=np.float64) + self.origin[0]
        py = np.asarray(py, dtype=np.float64) + self.origin[1]
        return a * px + b * py, c * px + d * py

    @property
    def chunks(self) -> tuple[int, int]:
        """How many chunks across and down the picture takes."""
        return math.ceil(self.size[0] / CHUNK), math.ceil(self.size[1] / CHUNK)


def glow_strength(levels: npt.NDArray[np.int8]) -> FloatArray:
    """How strongly each cell glows for its Satisfaction; -1 (nothing trading) does not."""
    table = np.array([0.0, *(SERVED_GLOW[level] for level in Satisfaction)])
    return table[levels.astype(np.int64) + 1]


@dataclass(frozen=True)
class Scenery:
    """Everything standing on the ground, sorted back to front: where each stands in art pixels,
    which sprite it is, and for tinted sprites the biome whose shades colour it (else -1)."""

    x: IntArray
    y: IntArray
    sprite: IntArray
    tint: IntArray

    @classmethod
    def of(
        cls,
        tile_map: TileMap,
        constructions: Constructions,
        projection: PixelProjection,
        sheet: SpriteSheet,
        station_levels: Mapping[Cell, int] | None = None,
    ) -> Self:
        grid = tile_map.grid
        biomes = biome_grid(tile_map)
        built = np.zeros(grid.shape, dtype=bool)
        for construction in constructions:
            built[construction.cell] = True
        xs: list[FloatArray] = []
        ys: list[FloatArray] = []
        sprites: list[IntArray] = []
        tints: list[IntArray] = []
        for biome, (per_cell, kinds) in TREES.items():
            rows, cols = np.nonzero((biomes == biome) & ~built)
            choices = np.concatenate([sheet.variants(kind) for kind in kinds])
            weights = np.concatenate([np.full(len(sheet.variants(k)), 1.0) for k in kinds])
            weights /= weights.sum()
            for k in range(math.ceil(per_cell)):
                chance = min(1.0, per_cell - k)
                keep = hash01(rows, cols, k, 1) < chance
                r, c = rows[keep], cols[keep]
                x, y = _jitter(grid, r, c, k)
                pick = np.searchsorted(np.cumsum(weights), hash01(r, c, k, 2), side="right")
                xs.append(x)
                ys.append(y)
                sprites.append(choices[np.minimum(pick, len(choices) - 1)])
                tints.append(np.full(len(x), -1))
        for relief, (kind, chances) in RELIEF_SPRITES.items():
            raised = (tile_map.relief == relief) & (tile_map.terrain == Terrain.LAND) & ~built
            rows, cols = np.nonzero(raised)
            shapes = np.array(sheet.variants(kind))
            for k, chance in enumerate(chances):
                keep = hash01(rows, cols, k, 30 + relief) < chance
                r, c = rows[keep], cols[keep]
                x, y = _jitter(grid, r, c, 31 + 3 * relief + k, RELIEF_SPREAD)
                xs.append(x)
                ys.append(y)
                pick = (hash01(r, c, k, 32 + relief) * len(shapes)).astype(np.int64)
                sprites.append(shapes[np.minimum(pick, len(shapes) - 1)])
                tints.append(biomes[r, c].astype(np.int64))
        rocky = np.isin(biomes, ROCKS_ON) & (tile_map.relief != Relief.MOUNTAINS)
        rows, cols = np.nonzero(rocky & ~built)
        keep = hash01(rows, cols, 3) < ROCK_CHANCE
        x, y = _jitter(grid, rows[keep], cols[keep], 9)
        xs.append(x)
        ys.append(y)
        sprites.append(np.full(len(x), sheet.index("rocks")))
        tints.append(np.full(len(x), -1))
        stations = constructions.stations()
        if stations:
            rows, cols = np.array(stations).T
            x, y = grid.center(rows, cols)
            xs.append(np.asarray(x, dtype=np.float64))
            ys.append(np.asarray(y, dtype=np.float64))
            levels = station_levels or {}
            sprites.append(
                np.array([sheet.index(station_sprite(levels.get(cell, 1))) for cell in stations])
            )
            tints.append(np.full(len(stations), -1))
        spots = BuildingSpots.of(constructions)
        looks = {kind: sheet.variants(name) for kind, name in VARIED_SPRITES.items()}
        xs.append(np.asarray(spots.x, dtype=np.float64))
        ys.append(np.asarray(spots.y, dtype=np.float64))
        sprites.append(
            np.array(
                [
                    looks[kind][variant % len(looks[kind])]
                    if kind in looks
                    else sheet.index(BUILDING_SPRITES[kind])
                    for kind, variant in zip(spots.kinds, spots.variants, strict=True)
                ],
                dtype=np.int64,
            )
        )
        tints.append(np.full(len(spots.kinds), -1))
        px, py = projection.to_pixels(np.concatenate(xs), np.concatenate(ys))
        px, py = np.rint(px).astype(np.int64), np.rint(py).astype(np.int64)
        sprite = np.concatenate(sprites).astype(np.int64)
        tint = np.concatenate(tints).astype(np.int64)
        order = np.lexsort((px, py))
        return cls(px[order], py[order], sprite[order], tint[order])

    def __len__(self) -> int:
        return len(self.x)


def _jitter(
    grid: Grid, rows: IntArray, cols: IntArray, k: int, spread: float = TREE_SPREAD
) -> tuple[FloatArray, FloatArray]:
    cx, cy = grid.center(rows, cols)
    dx = (hash01(rows, cols, k, 4) - 0.5) * 2 * spread
    dy = (hash01(rows, cols, k, 5) - 0.5) * 2 * spread
    return np.asarray(cx + dx, dtype=np.float64), np.asarray(cy + dy, dtype=np.float64)


class PixelWorld:
    """Draws chunks of one map as pixel art."""

    def __init__(
        self,
        tile_map: TileMap,
        constructions: Constructions,
        sheet: SpriteSheet,
        angle_deg: float,
        cell_units: float,
        fine: SpriteSheet | None = None,
    ) -> None:
        self.tile_map = tile_map
        self.constructions = constructions
        self.sheet = sheet
        self.projection = PixelProjection.of(tile_map.grid, angle_deg, cell_units)
        # Each station's level (railroads.sim.railway), which picks how grand it looks.
        self.station_levels: Mapping[Cell, int] = {}
        self.scenery = Scenery.of(tile_map, constructions, self.projection, sheet)
        self._biomes = biome_grid(tile_map)
        land = tile_map.terrain == Terrain.LAND
        self._levels = np.where(land, tile_map.relief.astype(np.int64), -1)
        # Each cell's Satisfaction, or -1 where nothing trades.
        self.satisfaction = np.full(tile_map.grid.shape, -1, dtype=np.int8)
        self._hills = land & (tile_map.relief == Relief.HILLS)
        self._relief_light = self._light_slopes(angle_deg)
        self._ramps = np.zeros((len(Biome), 4, 3))
        for biome, ramp in RAMPS.items():
            self._ramps[biome] = ramp
        self._plots = plots_of(constructions, FIELD_KINDS)
        # The sprites at each resolution chunks can be drawn at: the sheet, and the same sprites
        # drawn finer if there are any.
        self._sprites = {1: _Sprites.of(sheet)}
        if fine is not None:
            if fine.names != sheet.names:
                raise ValueError("The fine sprite sheet must hold the same sprites in order")
            self._sprites[FINE] = _Sprites.of(fine)
        self._tints: dict[tuple[int, int, int], npt.NDArray[np.float32]] = {}

    @property
    def resolutions(self) -> tuple[int, ...]:
        """The resolutions chunks can be drawn at, in pixels per art pixel."""
        return tuple(self._sprites)

    def buildings_changed(self, cells: Iterable[Cell]) -> set[tuple[int, int]]:
        """Puts the ground and scenery up again after buildings or forests changed in `cells`;
        returns the chunks that look different now."""
        self._biomes = biome_grid(self.tile_map)
        self.scenery = Scenery.of(
            self.tile_map, self.constructions, self.projection, self.sheet, self.station_levels
        )
        self._plots = plots_of(self.constructions, FIELD_KINDS)
        return self._chunks_around(cells)

    def satisfaction_changed(self, levels: npt.NDArray[np.int8]) -> set[tuple[int, int]]:
        """Takes each cell's new satisfaction; returns the chunks whose glow changed."""
        changed = np.argwhere(levels != self.satisfaction)
        self.satisfaction = levels.copy()
        return self._chunks_around([(int(r), int(c)) for r, c in changed])

    def _chunks_around(self, cells: Iterable[Cell]) -> set[tuple[int, int]]:
        cells = list(cells)
        if not cells:
            return set()
        rows, cols = np.array(cells).T
        px, py = self.projection.to_pixels(*self.tile_map.grid.center(rows, cols))
        reach = PIXEL_CELL_WIDTH + SPRITE_REACH
        chunks: set[tuple[int, int]] = set()
        for x, y in zip(px.tolist(), py.tolist(), strict=True):
            for cx in range(int((x - reach) // CHUNK), int((x + reach) // CHUNK) + 1):
                for cy in range(int((y - reach) // CHUNK), int((y + reach) // CHUNK) + 1):
                    chunks.add((cx, cy))
        return chunks

    def render(self, chunk_x: int, chunk_y: int, resolution: int = 1) -> Rgba:
        """One chunk as RGBA pixels, `resolution` of them for every art pixel each way; off the
        map it is clear. Above 1 the ground is blown up and the track and everything standing
        on it are drawn that much finer."""
        left, top = chunk_x * CHUNK, chunk_y * CHUNK
        rgb, alpha = self._ground(left, top)
        canvas = np.concatenate([rgb / 255, alpha[..., None]], axis=-1)
        if resolution > 1:
            canvas = canvas.repeat(resolution, axis=0).repeat(resolution, axis=1)
        self._lay_track(canvas, left, top, resolution)
        self._stand_scenery(canvas, left, top, resolution)
        alpha = canvas[..., 3:]
        canvas[..., :3] /= np.maximum(alpha, 1e-6)
        return np.ascontiguousarray(np.rint(np.clip(canvas, 0, 1) * 255).astype(np.uint8))

    def _ground(self, left: int, top: int) -> tuple[FloatArray, FloatArray]:
        grid = self.tile_map.grid
        size = CHUNK + 2 * APRON
        iy, ix = np.mgrid[top - APRON : top - APRON + size, left - APRON : left - APRON + size]
        x, y = self.projection.to_plane(ix + 0.5, iy + 0.5)
        rows, cols = grid.cell_at(x, y)
        inside = grid.contains(rows, cols)
        rows = np.clip(rows, 0, grid.rows - 1)
        cols = np.clip(cols, 0, grid.cols - 1)
        biomes = self._biomes[rows, cols]
        water = np.isin(biomes, WATER)

        grain = hash01(ix, iy, 11)
        cell_tone = hash01(rows, cols, 12)
        value = 1.4 + 0.9 * (grain - 0.5) + 0.5 * (cell_tone - 0.5) + 0.9 * BAYER[iy % 4, ix % 4]
        value += HILL_LIFT * self._hills[rows, cols]
        value += _sample(self._relief_light, x * RELIEF_STEPS_PER_CELL, y * RELIEF_STEPS_PER_CELL)
        ripple = hash01(ix // 3, iy, 13) > 0.94
        value = np.where(water, np.where(ripple, 3, 1.3 + 0.6 * (grain - 0.5)), value)
        shade = np.clip(np.rint(value), 0, 3).astype(np.int64)
        rgb = self._ramps[biomes, shade]

        near_water = _near(water, [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2)])
        near_land = _near(~water, [(0, 1), (0, -1), (1, 0), (-1, 0)])
        rgb[~water & near_water & inside] = SAND
        rgb[water & near_land & inside] = FOAM

        cell_id = rows * grid.cols + cols
        edge = (cell_id != np.roll(cell_id, -1, axis=1)) | (cell_id != np.roll(cell_id, -1, axis=0))
        rgb[edge & ~water] *= EDGE_SHADE
        levels = self._levels[rows, cols]
        lit = (levels > np.roll(levels, (1, 1), axis=(0, 1))) | (
            levels > np.roll(levels, (2, 2), axis=(0, 1))
        )
        shaded = (levels > np.roll(levels, (-1, -1), axis=(0, 1))) | (
            levels > np.roll(levels, (-2, -2), axis=(0, 1))
        )
        rgb[lit & ~water] *= BEVEL_LIGHT
        rgb[shaded & ~water] *= BEVEL_SHADE
        strength = glow_strength(self.satisfaction)[rows, cols][..., None] * ~water[..., None]
        rgb = rgb * (1 - strength) + np.array(GLOW) * strength
        if self.tile_map.deposits is not None:
            deposit = self.tile_map.deposits[rows, cols]
            shown = (deposit > 0) & (deposit != Deposit.GARRISON)
            speck = shown & (hash01(ix, iy, 31) < DEPOSIT_SPECK_SHARE)
            rgb[speck] = DEPOSIT_SPECKS[deposit[speck]]
        self._plough(rgb, x, y, rows * grid.cols + cols)
        self._plant_orchards(rgb, x, y, rows * grid.cols + cols, ix, iy)

        inner = (slice(APRON, APRON + CHUNK), slice(APRON, APRON + CHUNK))
        alpha = inside[inner].astype(np.float64)
        return np.clip(rgb[inner], 0, 255) * alpha[..., None], alpha

    def _light_slopes(self, angle_deg: float) -> FloatArray:
        """How many shades lighter (or darker, below 0) the ground is where it faces towards (or
        away from) the light, on a raster RELIEF_STEPS_PER_CELL steps per cell. Water is level."""
        grid = self.tile_map.grid
        width, height = grid.size()
        steps = RELIEF_STEPS_PER_CELL
        ys, xs = np.mgrid[0 : math.ceil(height * steps), 0 : math.ceil(width * steps)]
        rows, cols = grid.cell_at((xs + 0.5) / steps, (ys + 0.5) / steps)
        rows = np.clip(rows, 0, grid.rows - 1)
        cols = np.clip(cols, 0, grid.cols - 1)
        heights = self.tile_map.height_m[rows, cols].astype(np.float64)
        water = self.tile_map.terrain[rows, cols] != Terrain.LAND
        heights = _smooth(heights, round(RELIEF_SMOOTHING_CELLS * steps))
        km_per_step = grid.km_per_cell / steps
        rise_y, rise_x = np.gradient(heights, km_per_step * 1000)
        light_x, light_y = self._towards_light(angle_deg)
        facing = -(rise_x * light_x + rise_y * light_y)
        light = np.clip(SLOPE_LIGHT * facing, -MAX_SLOPE_SHADES, MAX_SLOPE_SHADES)
        return np.where(water, 0.0, light)

    def _towards_light(self, angle_deg: float) -> tuple[float, float]:
        """The direction on the map, as a unit vector, towards the top left of the screen."""
        x, y = np.linalg.inv(self.projection.matrix) @ np.array([-1.0, -1.0])
        length = math.hypot(x, y)
        return x / length, y / length

    def _plough(self, rgb: FloatArray, x: FloatArray, y: FloatArray, cells: IntArray) -> None:
        """Lays striped fields over the half of a cell each farm or crop in rows takes up."""
        grid = self.tile_map.grid
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        fields: list[tuple[Plot, BuildingKind | None]] = [
            (plot, None) for plot in self._plots[BuildingKind.FARM]
        ]
        fields += [(plot, kind) for kind in CROP_ROWS for plot in self._plots[kind]]
        for plot, crop in fields:
            row, col = plot.cell
            fx, fy = grid.center(row, col)
            if not (x0 - 1 < fx < x1 + 1 and y0 - 1 < fy < y1 + 1):
                continue
            field = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
            if not field.any():
                continue
            pick = row * 7 + col * 13
            across = x - fx if pick % 2 else y - fy
            stripe = np.floor(across * FIELD_STRIPES) % 2 == 0
            furrow, ridge = CROP_ROWS[crop] if crop is not None else FIELDS[pick % len(FIELDS)]
            rgb[field] = np.where(stripe[field][:, None], furrow, ridge)

    def _plant_orchards(
        self,
        rgb: FloatArray,
        x: FloatArray,
        y: FloatArray,
        cells: IntArray,
        ix: IntArray,
        iy: IntArray,
    ) -> None:
        """Lays groves of trees in rows over the half of a cell each orchard, date palms or
        plantation takes up."""
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        for kind, grove in GROVES.items():
            self._plant(rgb, x, y, cells, ix, iy, grove, self._plots[kind], (x0, x1, y0, y1))

    def _plant(
        self,
        rgb: FloatArray,
        x: FloatArray,
        y: FloatArray,
        cells: IntArray,
        ix: IntArray,
        iy: IntArray,
        grove: Grove,
        planted: list[Plot],
        box: tuple[float, float, float, float],
    ) -> None:
        grid = self.tile_map.grid
        x0, x1, y0, y1 = box
        for plot in planted:
            row, col = plot.cell
            fx, fy = grid.center(row, col)
            if not (x0 - 1 < fx < x1 + 1 and y0 - 1 < fy < y1 + 1):
                continue
            ground = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
            if not ground.any():
                continue
            u = (x[ground] - fx) * grove.rows + 0.5
            v = (y[ground] - fy) * grove.rows + 0.5
            near = np.hypot(u - np.round(u), v - np.round(v))
            colors = np.broadcast_to(np.array(grove.ground, dtype=np.float64), (u.size, 3)).copy()
            tree = near < grove.crown
            colors[tree] = grove.tree
            fruit = tree & (hash01(ix[ground], iy[ground], 21) > 0.82)
            pick = (hash01(ix[ground], iy[ground], 22) * len(grove.fruit)).astype(np.int64)
            for index, color in enumerate(grove.fruit):
                colors[fruit & (pick == index)] = color
            rgb[ground] = colors

    def _lay_track(self, canvas: FloatArray, left: int, top: int, resolution: int) -> None:
        """Draws the track crossing the chunk over its ground (track_layer)."""
        pieces = self.constructions.track()
        if not pieces:
            return
        track = self.track_layer(pieces, left, top, CHUNK, CHUNK, resolution)
        alpha = track[..., 3:] * (canvas[..., 3:] > 0)
        canvas[..., :3] = canvas[..., :3] * (1 - alpha) + track[..., :3] * alpha

    def track_layer(
        self,
        pieces: list[Track],
        left: int,
        top: int,
        width: int,
        height: int,
        resolution: int = 1,
    ) -> FloatArray:
        """These pieces of track as RGBA pixels from 0 to 1, over the art pixels from `left`,
        `top` across `width` by `height`, `resolution` pixels for each art pixel: a gravel bed with wooden sleepers and two steel rails
        on the ground, a timber deck with railings over water, and dark portals where it goes
        into mountains. It is drawn layer by layer, beds, then sleepers, then rails, so the
        pieces meeting in a cell make one junction or crossing. Double track is two tracks side
        by side that close up where they run into single track, and electric track has masts
        beside it holding wires over each track."""
        grid = self.tile_map.grid
        steps = grid.track_steps() / 2
        reach = PIXEL_CELL_WIDTH * 2
        k = resolution
        size = ((width + 2 * reach) * k, (height + 2 * reach) * k)
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        paint = ImageDraw.Draw(layer)
        projection = self.projection
        shift = (left - reach, top - reach)

        def art(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
            xs, ys = projection.to_pixels([p[0] for p in points], [p[1] for p in points])
            return [
                ((float(x) - shift[0]) * k, (float(y) - shift[1]) * k)
                for x, y in zip(xs, ys, strict=True)
            ]

        laid: list[_Laid] = []
        for piece in pieces:
            cx, cy = grid.center(*piece.cell)
            px, py = projection.to_pixels(cx, cy)
            if not (
                left - reach < px < left + width + reach and top - reach < py < top + height + reach
            ):
                continue
            a, b = (steps[point] for point in piece.ends)
            middle = (float(cx), float(cy))
            line = _bend((cx + a[0], cy + a[1]), middle, (cx + b[0], cy + b[1]))
            laid.append(_Laid(piece, middle, line, self._lanes(piece, line)))
        self._lay_bridges(paint, art, [each for each in laid if each.piece.bed == TrackBed.BRIDGE])
        for each in laid:
            if each.piece.bed == TrackBed.GROUND:
                for lane in each.lanes:
                    for segment in itertools.pairwise(lane):
                        paint.polygon(art(_band(*segment, TRACK_BED / 2)), fill=(*BALLAST, 255))
        for each in laid:
            if each.piece.bed == TrackBed.TUNNEL:
                self._lay_portals(paint, art, each)
        laid = [each for each in laid if each.piece.bed != TrackBed.TUNNEL]
        for each in laid:
            for lane in each.lanes:
                for sleeper in _sleepers(lane, _lengths(art(lane))[-1] / k):
                    paint.polygon(art(sleeper), fill=(*SLEEPER, 255))
        for each in laid:
            for lane in each.lanes:
                for side in (-GAUGE / 2, GAUGE / 2):
                    paint.line(art(_offset(lane, side)), fill=(*RAIL, 255), width=1)
        for each in laid:
            if each.piece.electric:
                self._hang_wires(paint, art, each, k)
        pixels = np.asarray(layer, dtype=np.float32) / 255
        return pixels[reach * k : (reach + height) * k, reach * k : (reach + width) * k]

    def _lanes(self, piece: Track, line: Path) -> list[Path]:
        """The middle of each track of the piece. Double track has one either side of its
        centre line, closing up to it at an end that runs into single track. Single track keeps
        to its centre line, except where it branches off double track in the same cell: there
        it runs on along the outer track on the side it turns to, and closes in to its centre
        line by its other end."""
        lengths = _lengths(line)
        half_length = lengths[-1] / 2
        if piece.width == TrackWidth.SINGLE:
            shifts = [self._branch_shift(piece, point) for point in piece.ends]
            if not any(shifts):
                return [line]
            widths = [
                shifts[0] * min(1.0, (lengths[-1] - along) / half_length)
                + shifts[1] * min(1.0, along / half_length)
                for along in lengths
            ]
            return [_offset(line, widths)]
        wide_start, wide_end = (self._double_beyond(piece, point) for point in piece.ends)
        widths = []
        for along in lengths:
            share = 1.0
            if not wide_start:
                share = min(share, along / half_length)
            if not wide_end:
                share = min(share, (lengths[-1] - along) / half_length)
            widths.append(DOUBLE_SPACING / 2 * share)
        return [_offset(line, [-width for width in widths]), _offset(line, widths)]

    def _branch_shift(self, piece: Track, point: int) -> float:
        """How far to the left of its centre line single track starts at this end, to leave
        from the outer track of full double track it branches off there: half the spacing to
        whichever side it turns away to, or nothing."""
        steps = self.tile_map.grid.track_steps()
        mine = piece.ends[1] if piece.ends[0] == point else piece.ends[0]
        for other in self.constructions.track_at(piece.cell):
            if other.width != TrackWidth.DOUBLE or point not in other.ends:
                continue
            if not self._double_beyond(other, point):
                continue
            theirs = other.ends[1] if other.ends[0] == point else other.ends[0]
            ux, uy = -steps[point]
            turn = sum(
                sign * (ux * steps[end][1] - uy * steps[end][0])
                for sign, end in ((1, mine), (-1, theirs))
            )
            side = math.copysign(DOUBLE_SPACING / 2, turn)
            return side if point == piece.ends[0] else -side
        return 0.0

    def _double_beyond(self, piece: Track, point: int) -> bool:
        """Whether track leaving the piece by this point runs on as double track, or into a
        station or nowhere, rather than into single track."""
        grid = self.tile_map.grid
        beyond, entry = grid.track_exit(piece.cell, point)
        if not grid.contains(*beyond):
            return True
        joined = [other for other in self.constructions.track_at(beyond) if entry in other.ends]
        return not joined or any(other.width == TrackWidth.DOUBLE for other in joined)

    def _lay_bridges(self, paint: ImageDraw.ImageDraw, art: ToArt, laid: list[_Laid]) -> None:
        for half, color in (
            (BRIDGE_WIDTH / 2 + 0.03, BRIDGE_SHADOW),
            (BRIDGE_WIDTH / 2, BRIDGE_DECK),
        ):
            for each in laid:
                for lane in each.lanes:
                    for segment in itertools.pairwise(lane):
                        paint.polygon(art(_band(*segment, half)), fill=(*color, 255))
        for each in laid:
            outer = each.lanes[0], each.lanes[-1]
            for lane, side in zip(outer, (-BRIDGE_WIDTH / 2, BRIDGE_WIDTH / 2), strict=True):
                paint.line(art(_offset(lane, side)), fill=(*RAILING, 255), width=1)

    def _lay_portals(self, paint: ImageDraw.ImageDraw, art: ToArt, laid: _Laid) -> None:
        """A dark arch where the tunnel meets track out in the open."""
        grid = self.tile_map.grid
        piece, middle = laid.piece, laid.middle
        half = PORTAL / 2 + (DOUBLE_SPACING / 2 if piece.width == TrackWidth.DOUBLE else 0.0)
        for point, end in zip(piece.ends, (laid.line[0], laid.line[-1]), strict=True):
            beyond, _ = grid.track_exit(piece.cell, point)
            if not grid.contains(*beyond) or self.tile_map.relief[beyond] == Relief.MOUNTAINS:
                continue
            inner = (
                middle[0] + (end[0] - middle[0]) * 0.6,
                middle[1] + (end[1] - middle[1]) * 0.6,
            )
            paint.polygon(art(_band(inner, end, half + 0.03)), fill=(*PORTAL_STONE, 255))
            paint.polygon(art(_band(inner, end, half)), fill=(*PORTAL_DARK, 255))

    def _hang_wires(
        self, paint: ImageDraw.ImageDraw, art: ToArt, laid: _Laid, resolution: int = 1
    ) -> None:
        """A wire over each track, and masts at even steps along the piece on the far side of
        it, each with an arm reaching over the track to hold the wires."""
        (_, _), (c, d) = self.projection.matrix
        line, lanes = laid.line, laid.lanes
        lengths = _lengths(line)
        count = max(1, round(lengths[-1] / MAST_GAP))
        for lane in lanes:
            wire = [(x, y - WIRE_HEIGHT * resolution) for x, y in art(lane)]
            paint.line(wire, fill=(*WIRE, 255), width=1)
        for k in range(count):
            along = (k + 0.5) * lengths[-1] / count
            _, (ux, uy) = _point_at(line, lengths, along)
            far = (
                (-uy, ux)
                if c * -uy + d * ux < 0 or (c * -uy + d * ux == 0 and ux < 0)
                else (uy, -ux)
            )
            overs = []
            for lane in lanes:
                lane_lengths = _lengths(lane)
                over, _ = _point_at(lane, lane_lengths, along * lane_lengths[-1] / lengths[-1])
                overs.append(over)
            overs.sort(key=lambda over: over[0] * far[0] + over[1] * far[1])
            near, back = overs[0], overs[-1]
            base = (back[0] + far[0] * MAST_SIDE, back[1] + far[1] * MAST_SIDE)
            (bx, by), (wx, wy) = art([base, near])
            top = (bx, by - MAST_HEIGHT * resolution)
            paint.line([(bx, by), top], fill=(*MAST, 255), width=1)
            paint.line([top, (wx, wy - WIRE_HEIGHT * resolution)], fill=(*MAST, 255), width=1)

    def _stand_scenery(self, canvas: FloatArray, left: int, top: int, resolution: int) -> None:
        scenery, drawn, k = self.scenery, self._sprites[resolution], resolution
        x0 = (scenery.x - left) * k - drawn.anchors[scenery.sprite, 0]
        y0 = (scenery.y - top) * k - drawn.anchors[scenery.sprite, 1]
        w, h = drawn.sizes[scenery.sprite, 0], drawn.sizes[scenery.sprite, 1]
        span = CHUNK * k
        overlapping = (x0 < span) & (x0 + w > 0) & (y0 < span) & (y0 + h > 0)
        for i in np.flatnonzero(overlapping).tolist():
            sprite, tint = int(scenery.sprite[i]), int(scenery.tint[i])
            pixels = (
                drawn.premultiplied[sprite] if tint < 0 else self._tinted(sprite, tint, resolution)
            )
            _blend(canvas, pixels, int(x0[i]), int(y0[i]))

    def _tinted(self, sprite: int, biome: int, resolution: int = 1) -> npt.NDArray[np.float32]:
        """A sprite with its pure greys coloured with a biome's shades, premultiplied."""
        key = (sprite, biome, resolution)
        if key not in self._tints:
            ramp = self._ramps[biome]
            colors = np.vstack([ramp[0] * TINT_OUTLINE, ramp, ramp[3] * TINT_HIGHLIGHT]) / 255
            pixels = self._sprites[resolution].pixels[sprite].copy()
            grey = (pixels[..., 3] == 1) & (pixels[..., 0] == pixels[..., 1])
            grey &= pixels[..., 1] == pixels[..., 2]
            levels = np.rint(pixels[..., 0] * 255).astype(np.int64)
            level = np.abs(levels[..., None] - np.array(TINT_GREYS)).argmin(axis=-1)
            pixels[grey, :3] = np.clip(colors[level[grey]], 0, 1)
            self._tints[key] = np.concatenate(
                [pixels[..., :3] * pixels[..., 3:], pixels[..., 3:]], axis=-1
            )
        return self._tints[key]


@dataclass(frozen=True)
class _Sprites:
    """A sheet's sprites as the renderer uses them: each one's pixels, premultiplied too, its
    width and height, and its anchor."""

    pixels: list[npt.NDArray[np.float32]]
    premultiplied: list[npt.NDArray[np.float32]]
    sizes: IntArray
    anchors: IntArray

    @classmethod
    def of(cls, sheet: SpriteSheet) -> Self:
        premultiplied = [
            np.concatenate([p[..., :3] * p[..., 3:], p[..., 3:]], axis=-1) for p in sheet.pixels
        ]
        return cls(sheet.pixels, premultiplied, sheet.sizes, sheet.anchors)


def _smooth(values: FloatArray, radius: int) -> FloatArray:
    """Two passes of a box blur `radius` steps either side, along both axes."""
    if radius < 1:
        return values
    for axis in (0, 1, 0, 1):
        padded = np.pad(
            values, [(radius + 1, radius) if a == axis else (0, 0) for a in (0, 1)], mode="edge"
        )
        sums = np.cumsum(padded, axis=axis)
        upper = np.take(sums, np.arange(2 * radius + 1, sums.shape[axis]), axis=axis)
        lower = np.take(sums, np.arange(0, sums.shape[axis] - 2 * radius - 1), axis=axis)
        values = (upper - lower) / (2 * radius + 1)
    return values


def _sample(raster: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
    """The raster at fractional raster positions, blended from its four nearest steps."""
    fx, fy = np.clip(x - 0.5, 0, raster.shape[1] - 1), np.clip(y - 0.5, 0, raster.shape[0] - 1)
    x0, y0 = (
        np.minimum(fx.astype(np.int64), raster.shape[1] - 2),
        np.minimum(fy.astype(np.int64), raster.shape[0] - 2),
    )
    tx, ty = fx - x0, fy - y0
    top = raster[y0, x0] * (1 - tx) + raster[y0, x0 + 1] * tx
    bottom = raster[y0 + 1, x0] * (1 - tx) + raster[y0 + 1, x0 + 1] * tx
    return top * (1 - ty) + bottom * ty


def _near(mask: npt.NDArray[np.bool_], shifts: list[tuple[int, int]]) -> npt.NDArray[np.bool_]:
    near = np.zeros_like(mask)
    for shift in shifts:
        near |= np.roll(mask, shift, axis=(0, 1))
    return near


def _band(
    start: tuple[float, float], end: tuple[float, float], half_width: float, overlap: bool = True
) -> list[tuple[float, float]]:
    """The corners of a strip from `start` to `end`, `half_width` either side; with `overlap`, a
    little longer at both ends so strips meet without gaps."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy * half_width, ux * half_width
    grow = half_width * 0.5 if overlap else 0.0
    sx, sy = start[0] - ux * grow, start[1] - uy * grow
    ex, ey = end[0] + ux * grow, end[1] + uy * grow
    return [(sx + nx, sy + ny), (ex + nx, ey + ny), (ex - nx, ey - ny), (sx - nx, sy - ny)]


def _along(
    middle: tuple[float, float], start: tuple[float, float], end: tuple[float, float], half: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """A short stretch of the line from `start` to `end`, `half` either side of `middle`, turned
    across it: the line a sleeper lies along."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    return (middle[0] - ux * half, middle[1] - uy * half), (
        middle[0] + ux * half,
        middle[1] + uy * half,
    )


def _offset(path: Path, side: float | list[float]) -> Path:
    """The path moved `side` to its left, by one amount or by one for each of its points,
    joined at each bend where the moved lines meet."""
    sides = side if isinstance(side, list) else [side] * len(path)
    normals = []
    for (x0, y0), (x1, y1) in itertools.pairwise(path):
        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        normals.append((-(y1 - y0) / length, (x1 - x0) / length))
    moved = []
    for index, ((x, y), amount) in enumerate(zip(path, sides, strict=True)):
        before = normals[max(0, index - 1)]
        after = normals[min(len(normals) - 1, index)]
        mx, my = before[0] + after[0], before[1] + after[1]
        dot = (mx * before[0] + my * before[1]) or 1.0
        moved.append((x + mx / dot * amount, y + my / dot * amount))
    return moved


def _bend(start: Point, middle: Point, end: Point) -> Path:
    """The line from `start` through `middle` to `end`, rounded off over the last BEND before
    and after the middle instead of turning sharply there."""
    (ax, ay), (cx, cy), (bx, by) = start, middle, end
    la, lb = math.hypot(ax - cx, ay - cy), math.hypot(bx - cx, by - cy)
    ua, ub = ((ax - cx) / la, (ay - cy) / la), ((bx - cx) / lb, (by - cy) / lb)
    if ua[0] * ub[0] + ua[1] * ub[1] < -0.999:
        return [start, middle, end]
    p = (cx + ua[0] * BEND, cy + ua[1] * BEND)
    q = (cx + ub[0] * BEND, cy + ub[1] * BEND)
    curve = [
        (
            (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * cx + t**2 * q[0],
            (1 - t) ** 2 * p[1] + 2 * (1 - t) * t * cy + t**2 * q[1],
        )
        for t in (0.25, 0.5, 0.75)
    ]
    return [start, p, *curve, q, end]


def _lengths(path: Path) -> list[float]:
    """How far along the path each of its points lies."""
    return [0.0, *itertools.accumulate(math.dist(a, b) for a, b in itertools.pairwise(path))]


def _point_at(path: Path, lengths: list[float], along: float) -> tuple[Point, Point]:
    """The point `along` the path and the way the path runs there, as a unit vector."""
    for (a, b), (start, stop) in zip(
        itertools.pairwise(path), itertools.pairwise(lengths), strict=True
    ):
        if along <= stop or b == path[-1]:
            span = (stop - start) or 1.0
            t = min(1.0, max(0.0, (along - start) / span))
            point = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            return point, ((b[0] - a[0]) / span, (b[1] - a[1]) / span)
    raise ValueError("A path needs two points")


def _sleepers(lane: Path, art_length: float) -> list[Path]:
    """The sleepers under one track, evenly along it, each lying across the way it runs. They
    are spaced by the track's length in the picture, `art_length` pixels, so they stay apart
    where the view foreshortens it."""
    lengths = _lengths(lane)
    count = max(1, round(art_length / (SLEEPER_GAP * PIXEL_CELL_WIDTH)))
    sleepers = []
    for k in range(count):
        (x, y), (ux, uy) = _point_at(lane, lengths, (k + 0.5) * lengths[-1] / count)
        half = SLEEPER_WIDTH / 2
        start, end = (x - ux * half, y - uy * half), (x + ux * half, y + uy * half)
        sleepers.append(_band(start, end, SLEEPER_LENGTH / 2, overlap=False))
    return sleepers


@dataclass(frozen=True)
class _Laid:
    """A piece of track as it is drawn: the middle of its cell, its centre line through the
    cell, and the middle of each of its tracks."""

    piece: Track
    middle: Point
    line: Path
    lanes: list[Path]


def _blend(canvas: FloatArray, sprite: npt.NDArray[np.float32], x: int, y: int) -> None:
    """Lays a premultiplied RGBA sprite over the canvas with its top-left corner at (x, y)."""
    h, w = sprite.shape[:2]
    cx0, cy0 = max(x, 0), max(y, 0)
    cx1, cy1 = min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
    if cx0 >= cx1 or cy0 >= cy1:
        return
    part = sprite[cy0 - y : cy1 - y, cx0 - x : cx1 - x]
    under = canvas[cy0:cy1, cx0:cx1]
    under *= 1 - part[..., 3:]
    under += part


@dataclass(frozen=True)
class ChunkPlacement:
    """Where the big picture sits on screen at a pixel scale: the screen position of its top-left
    corner, rounded to a whole pixel so art pixels line up with screen pixels, and the chunks
    that show."""

    origin: tuple[int, int]
    scale: int
    chunks: list[tuple[int, int]]

    @classmethod
    def of(
        cls,
        projection: PixelProjection,
        camera_pixel: tuple[float, float],
        scale: int,
        center: tuple[float, float],
        clip: tuple[tuple[float, float], tuple[float, float]],
    ) -> Self:
        """`camera_pixel` is the art pixel at the middle of the view, which sits at `center`."""
        origin = (
            round(center[0] - camera_pixel[0] * scale),
            round(center[1] - camera_pixel[1] * scale),
        )
        (left, top), (right, bottom) = clip
        across, down = projection.chunks
        span = CHUNK * scale
        first_x = max(0, math.floor((left - origin[0]) / span))
        last_x = min(across - 1, math.floor((right - origin[0]) / span))
        first_y = max(0, math.floor((top - origin[1]) / span))
        last_y = min(down - 1, math.floor((bottom - origin[1]) / span))
        chunks = [(x, y) for y in range(first_y, last_y + 1) for x in range(first_x, last_x + 1)]
        return cls(origin, scale, chunks)

    def screen_rect(self, chunk: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        span = CHUNK * self.scale
        left = self.origin[0] + chunk[0] * span
        top = self.origin[1] + chunk[1] * span
        return (left, top), (left + span, top + span)
