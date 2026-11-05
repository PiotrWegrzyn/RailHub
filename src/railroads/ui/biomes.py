"""How the map looks: every cell is drawn as a biome picked from its properties.

Water is always drawn as its own kind of water. On land, temperature and vegetation decide
together, so hot and barren is desert while hot and forest is jungle.
"""

from enum import IntEnum

import numpy as np
import numpy.typing as npt

from railroads.sim.tilemap import Temperature, Terrain, Tile, TileMap, Vegetation


class Biome(IntEnum):
    SEA = 0
    LAKE = 1
    RIVER = 2
    SNOW = 3
    TUNDRA = 4
    TAIGA = 5
    ROCK = 6
    GRASSLAND = 7
    FOREST = 8
    DESERT = 9
    SAVANNA = 10
    JUNGLE = 11


WATER_BIOMES = {Terrain.SEA: Biome.SEA, Terrain.LAKE: Biome.LAKE, Terrain.RIVER: Biome.RIVER}
LAND_BIOMES = {
    (Temperature.COLD, Vegetation.BARREN): Biome.SNOW,
    (Temperature.COLD, Vegetation.GRASS): Biome.TUNDRA,
    (Temperature.COLD, Vegetation.FOREST): Biome.TAIGA,
    (Temperature.NORMAL, Vegetation.BARREN): Biome.ROCK,
    (Temperature.NORMAL, Vegetation.GRASS): Biome.GRASSLAND,
    (Temperature.NORMAL, Vegetation.FOREST): Biome.FOREST,
    (Temperature.HOT, Vegetation.BARREN): Biome.DESERT,
    (Temperature.HOT, Vegetation.GRASS): Biome.SAVANNA,
    (Temperature.HOT, Vegetation.FOREST): Biome.JUNGLE,
}
BIOME_COLORS: dict[Biome, tuple[int, int, int]] = {
    Biome.SEA: (38, 84, 158),
    Biome.LAKE: (58, 118, 196),
    Biome.RIVER: (74, 140, 214),
    Biome.SNOW: (232, 235, 240),
    Biome.TUNDRA: (164, 168, 132),
    Biome.TAIGA: (54, 96, 78),
    Biome.ROCK: (146, 136, 120),
    Biome.GRASSLAND: (124, 174, 86),
    Biome.FOREST: (50, 112, 56),
    Biome.DESERT: (220, 194, 128),
    Biome.SAVANNA: (186, 178, 92),
    Biome.JUNGLE: (28, 92, 42),
}


def biome_of(tile: Tile) -> Biome:
    if tile.terrain in WATER_BIOMES:
        return WATER_BIOMES[tile.terrain]
    return LAND_BIOMES[tile.temperature, tile.vegetation]


def biome_grid(tile_map: TileMap) -> npt.NDArray[np.uint8]:
    table = np.zeros((len(Terrain), len(Temperature), len(Vegetation)), dtype=np.uint8)
    for terrain in Terrain:
        for temperature in Temperature:
            for vegetation in Vegetation:
                tile = Tile(terrain, vegetation, temperature, height_m=0)
                table[terrain, temperature, vegetation] = biome_of(tile)
    return table[tile_map.terrain, tile_map.temperature, tile_map.vegetation]


def biome_colors(tile_map: TileMap) -> npt.NDArray[np.uint8]:
    palette = np.zeros((len(Biome), 4), dtype=np.uint8)
    for biome, (r, g, b) in BIOME_COLORS.items():
        palette[biome] = (r, g, b, 255)
    return palette[biome_grid(tile_map)]
