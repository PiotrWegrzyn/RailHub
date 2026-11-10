"""Everything standing on the pixel map's ground, back to front: trees, rocks, hills and mountains, and every building, station and farmstead, each with its sprite."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np

from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.grids import Cell, Grid
from railroads.sim.tilemap import Relief, Terrain, TileMap
from railroads.ui.biomes import Biome, biome_grid
from railroads.ui.buildings import BuildingSpots
from railroads.ui.pixel.art import HILL, MOUNTAIN
from railroads.ui.pixel.projection import FloatArray, IntArray, PixelProjection, hash01
from railroads.ui.pixel.sheet import SpriteSheet

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
