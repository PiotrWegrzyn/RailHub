"""Everything standing on the pixel map's ground, back to front: trees, rocks, and every building,
station and farmstead, each with its sprite, lifted with the ground under it where hills and
mountains rise (railroads.ui.pixel.relief)."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np

from railroads.sim.constructions import GREW_FROM, BuildingKind, Constructions
from railroads.sim.grids import Cell, FloatArray, Grid, IntArray
from railroads.sim.tilemap import Relief, TileMap
from railroads.ui.biomes import Biome, biome_grid
from railroads.ui.building_spots import BuildingSpots
from railroads.ui.pixel.ground import FIELD_KINDS
from railroads.ui.pixel.projection import PixelProjection, hash01
from railroads.ui.pixel.relief import Elevation
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.track_art import station_house, station_line

# What grows on each biome as single trees: how many per cell (a fraction is a chance of one)
# and which kinds, by sprite name prefix. Woods are drawn into the ground as a canopy instead
# (railroads.ui.pixel.ground).
TREES: dict[Biome, tuple[float, tuple[str, ...]]] = {
    Biome.GRASSLAND: (0.06, ("leafy",)),
    Biome.SAVANNA: (0.04, ("leafy",)),
    Biome.TUNDRA: (0.03, ("pine",)),
}


ROCKS_ON = (Biome.MOUNTAIN, Biome.SNOW)


ROCK_CHANCE = 0.4


TREE_SPREAD = 0.4


# How far a building's foot may reach past the edge of its cell, in pixels, and how far along
# the way from where it stands to the cell's middle it is moved in turn until it fits.
CELL_SLACK = 1.0
FIT_STEPS = np.linspace(0.0, 1.0, 6)
# How opaque a sprite's pixel must be to count as part of what it shows.
FOOT_ALPHA = 0.1
# How far above its anchor, in pixels, a sprite's column of pixels may end and still stand on the
# ground.
GROUND_REACH = 3


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
    BuildingKind.GASWORKS: "gasworks",
    BuildingKind.COKE_PLANT: "coke_plant",
    BuildingKind.POTTER: "potter",
    BuildingKind.TILE_WORKS: "tile_works",
    BuildingKind.FOUNDRY: "foundry",
    BuildingKind.DAIRY: "dairy",
    BuildingKind.SAUSAGE_MAKER: "sausage_maker",
    BuildingKind.CARPET_WEAVER: "carpet_weaver",
    BuildingKind.CLOCK_MAKER: "clock_maker",
    BuildingKind.MIRROR_MAKER: "mirror_maker",
    BuildingKind.MANOR: "manor",
    BuildingKind.THEATRE: "theatre",
    BuildingKind.HOSPITAL: "hospital",
    BuildingKind.RAILWAY_WORKS: "railway_works",
    BuildingKind.STEEL_MILL: "steel_mill",
    BuildingKind.SHIPYARD: "shipyard",
    BuildingKind.LUMBER_MILL: "lumber_mill",
    BuildingKind.PORT: "port",
    BuildingKind.CLAY_PIT: "clay_pit",
    BuildingKind.LIMESTONE_QUARRY: "limestone_quarry",
    BuildingKind.SAND_PIT: "sand_pit",
    BuildingKind.POTATO_FARM: "potato_farm",
    BuildingKind.CATTLE_RANCH: "cattle_ranch",
    BuildingKind.PIG_FARM: "pig_farm",
    BuildingKind.CHICKEN_FARM: "chicken_farm",
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
    BuildingKind.DIESEL_POWER_PLANT: "diesel_power_plant",
    BuildingKind.GAS_POWER_PLANT: "gas_power_plant",
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
    BuildingKind.CLUB: "tavern",
    BuildingKind.FILLING_STATION: "car_dealer",
    BuildingKind.DEPARTMENT_STORE: "department_store",
    BuildingKind.HOTEL: "hotel",
    BuildingKind.JUICE_WORKS: "winery",
}

# Big and mega works have sprites of their own (railroads.ui.pixel.art.GROWN_SCALE).
for _grown, _kind in GREW_FROM.items():
    BUILDING_SPRITES[_grown] = f"{_grown.name.split('_')[0].lower()}_{BUILDING_SPRITES[_kind]}"


def platform_sprite(line: str | None = None) -> str:
    """The sprite of a platform added to a station, turned like station_sprite."""
    return "platform" if line is None else f"platform_{line}"


def station_sprite(level: int, line: str | None = None) -> str:
    """The sprite of a station raised to a level, turned along track running `line` on the
    screen (railroads.ui.pixel.track_art.station_line) if it is given."""
    name = "station" if level <= 1 else f"station_{level}"
    return name if line is None else f"{name}_{line}"


@dataclass(frozen=True)
class Scenery:
    """Everything standing on the ground, sorted back to front: where each stands in art pixels,
    which sprite it is, and how many art pixels the ground under it is lifted."""

    x: IntArray
    y: IntArray
    sprite: IntArray
    lift: IntArray

    @classmethod
    def of(
        cls,
        tile_map: TileMap,
        constructions: Constructions,
        projection: PixelProjection,
        sheet: SpriteSheet,
        station_levels: Mapping[Cell, int] | None = None,
        elevation: Elevation | None = None,
    ) -> Self:
        grid = tile_map.grid
        biomes = biome_grid(tile_map)
        built = np.zeros(grid.shape, dtype=bool)
        for construction in constructions:
            built[construction.cell] = True
        xs: list[FloatArray] = []
        ys: list[FloatArray] = []
        sprites: list[IntArray] = []
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
        rocky = np.isin(biomes, ROCKS_ON) & (tile_map.relief != Relief.MOUNTAINS)
        rows, cols = np.nonzero(rocky & ~built)
        keep = hash01(rows, cols, 3) < ROCK_CHANCE
        x, y = _jitter(grid, rows[keep], cols[keep], 9)
        xs.append(x)
        ys.append(y)
        sprites.append(np.full(len(x), sheet.index("rocks")))
        stations = constructions.every_station_cell()
        if stations:
            x, y = _station_houses(constructions, projection, stations)
            xs.append(x)
            ys.append(y)
            levels = station_levels or {}
            turned = grid.kind == "hex"
            sprites.append(
                np.array(
                    [
                        sheet.index(
                            station_sprite(levels.get(cell, 1), line)
                            if constructions.main_station(cell) == cell
                            else platform_sprite(line)
                        )
                        for cell in stations
                        for line in [station_line(constructions, cell) if turned else None]
                    ]
                )
            )
        spots = BuildingSpots.of(constructions, frozenset(FIELD_KINDS))
        looks = {kind: sheet.variants(name) for kind, name in VARIED_SPRITES.items()}
        built_looks = np.array(
            [
                looks[kind][variant % len(looks[kind])]
                if kind in looks
                else sheet.index(BUILDING_SPRITES[kind])
                for kind, variant in zip(spots.kinds, spots.variants, strict=True)
            ],
            dtype=np.int64,
        )
        before = sum(len(x) for x in xs)
        xs.append(np.asarray(spots.x, dtype=np.float64))
        ys.append(np.asarray(spots.y, dtype=np.float64))
        sprites.append(built_looks)
        px, py = projection.to_pixels(np.concatenate(xs), np.concatenate(ys))
        fitted = np.flatnonzero(~np.isin(spots.kinds, list(FIELD_KINDS)))
        if fitted.size:
            px[before + fitted], py[before + fitted] = _inside_cells(
                grid,
                projection,
                sheet,
                spots.x[fitted],
                spots.y[fitted],
                built_looks[fitted],
            )
        px, py = np.rint(px).astype(np.int64), np.rint(py).astype(np.int64)
        sprite = np.concatenate(sprites).astype(np.int64)
        lift = np.zeros(px.shape, dtype=np.int64)
        if elevation is not None and px.size:
            x, y = projection.to_plane(px, py)
            lift = np.rint(elevation.lift(x, y)).astype(np.int64)
        order = np.lexsort((px, py))
        return cls(px[order], py[order], sprite[order], lift[order])

    def __len__(self) -> int:
        return len(self.x)


def _inside_cells(
    grid: Grid,
    projection: PixelProjection,
    sheet: SpriteSheet,
    x: FloatArray,
    y: FloatArray,
    sprite: IntArray,
) -> tuple[FloatArray, FloatArray]:
    """Where in the picture buildings standing at (x, y) on the map are drawn so the foot of each
    stays inside its cell (_feet): moved no further than it must towards the middle of its cell,
    and all the way if even there it does not fit."""
    rows, cols = grid.cell_at(x, y)
    px, py = projection.to_pixels(x, y)
    cx, cy = projection.to_pixels(*grid.center(rows, cols))
    middle = np.array(projection.to_pixels(*grid.center(0, 0)))
    corners = np.array([projection.to_pixels(*c) for c in grid.corners(0, 0)]) - middle
    outward = np.roll(corners, -1, axis=0) - corners
    outward = np.stack([outward[:, 1], -outward[:, 0]], axis=1)
    outward /= np.linalg.norm(outward, axis=1, keepdims=True)
    outward *= np.sign(np.einsum("ij,ij->i", outward, corners))[:, None]
    reach = np.einsum("ij,ij->i", outward, corners) + CELL_SLACK
    unique, which = np.unique(sprite, return_inverse=True)
    feet = _feet(sheet, unique)[which]
    away = np.stack([px - cx, py - cy], axis=1)
    kept = np.zeros(len(px))
    for share in FIT_STEPS:
        points = feet + (away * share)[:, None, :]
        fits = (np.einsum("npj,ej->npe", points, outward) <= reach).all(axis=(1, 2))
        kept[fits] = share
    return cx + away[:, 0] * kept, cy + away[:, 1] * kept


def _feet(sheet: SpriteSheet, sprites: IntArray) -> FloatArray:
    """Where each sprite meets the ground at its left and right ends and at its lowest, in
    pixels from its anchor: of the columns of pixels reaching down to within GROUND_REACH of the
    anchor, so smoke and chimneys do not count."""
    feet = np.zeros((len(sprites), 3, 2))
    for i, index in enumerate(sprites.tolist()):
        painted = sheet.pixels[index][..., 3] > FOOT_ALPHA
        lowest = painted.shape[0] - 1 - np.argmax(painted[::-1], axis=0)
        reaching = painted.any(axis=0) & (lowest >= sheet.anchors[index][1] - GROUND_REACH)
        columns = np.flatnonzero(reaching)
        if not columns.size:
            continue
        bottom = columns[np.argmax(lowest[columns])]
        feet[i] = [
            (column, lowest[column]) for column in (columns[0], columns[-1], bottom)
        ] - sheet.anchors[index]
    return feet


def _station_houses(
    constructions: Constructions, projection: PixelProjection, stations: list[Cell]
) -> tuple[FloatArray, FloatArray]:
    """Where each station's house stands: set back from the track through it, away from the
    viewer, so the track runs past in front of it (railroads.ui.pixel.track_art)."""
    grid = constructions.tile_map.grid
    xs, ys = [], []
    for cell in stations:
        back = np.linalg.solve(projection.matrix, station_house(constructions, projection, cell))
        cx, cy = grid.center(*cell)
        xs.append(float(cx) + float(back[0]))
        ys.append(float(cy) + float(back[1]))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _jitter(
    grid: Grid, rows: IntArray, cols: IntArray, k: int, spread: float = TREE_SPREAD
) -> tuple[FloatArray, FloatArray]:
    cx, cy = grid.center(rows, cols)
    dx = (hash01(rows, cols, k, 4) - 0.5) * 2 * spread
    dy = (hash01(rows, cols, k, 5) - 0.5) * 2 * spread
    return np.asarray(cx + dx, dtype=np.float64), np.asarray(cy + dy, dtype=np.float64)
