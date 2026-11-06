import itertools

import numpy as np
import pytest

from railroads.sim.geo import Bounds
from railroads.sim.grids import SquareGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, Tile, TileMap, Vegetation
from railroads.ui.biomes import BIOME_COLORS, Biome, biome_grid, biome_of


@pytest.mark.parametrize(
    ("relief", "temperature", "vegetation"),
    list(itertools.product(Relief, Temperature, Vegetation)),
)
def test_a_river_always_looks_like_a_river(
    relief: Relief, temperature: Temperature, vegetation: Vegetation
) -> None:
    tile = Tile(Terrain.RIVER, vegetation, temperature, height_m=0, relief=relief)

    biome = biome_of(tile)

    assert biome is Biome.RIVER


@pytest.mark.parametrize(
    ("temperature", "vegetation", "expected"),
    [
        (Temperature.HOT, Vegetation.BARREN, Biome.DESERT),
        (Temperature.HOT, Vegetation.FOREST, Biome.JUNGLE),
        (Temperature.COLD, Vegetation.FOREST, Biome.TAIGA),
        (Temperature.COLD, Vegetation.BARREN, Biome.TUNDRA),
        (Temperature.NORMAL, Vegetation.GRASS, Biome.GRASSLAND),
    ],
)
def test_land_biome_comes_from_temperature_and_vegetation(
    temperature: Temperature, vegetation: Vegetation, expected: Biome
) -> None:
    tile = Tile(Terrain.LAND, vegetation, temperature, height_m=0, relief=Relief.HILLS)

    biome = biome_of(tile)

    assert biome is expected


@pytest.mark.parametrize("vegetation", list(Vegetation))
@pytest.mark.parametrize(
    ("temperature", "expected"),
    [
        (Temperature.COLD, Biome.SNOW),
        (Temperature.NORMAL, Biome.MOUNTAIN),
        (Temperature.HOT, Biome.MOUNTAIN),
    ],
)
def test_mountains_are_grey_rock_or_snow_whatever_grows_there(
    vegetation: Vegetation, temperature: Temperature, expected: Biome
) -> None:
    tile = Tile(Terrain.LAND, vegetation, temperature, height_m=0, relief=Relief.MOUNTAINS)

    biome = biome_of(tile)

    assert biome is expected


def test_only_mountains_are_drawn_grey_or_snowy() -> None:
    flat_and_hill_biomes = {
        biome_of(Tile(Terrain.LAND, vegetation, temperature, height_m=0, relief=relief))
        for relief in (Relief.FLAT, Relief.HILLS)
        for temperature in Temperature
        for vegetation in Vegetation
    }

    assert not flat_and_hill_biomes & {Biome.MOUNTAIN, Biome.SNOW}


def test_every_biome_has_a_colour() -> None:
    colored = set(BIOME_COLORS)

    assert colored == set(Biome)


def test_biome_grid_matches_the_biome_of_every_cell() -> None:
    combinations = list(itertools.product(Terrain, Relief, Temperature, Vegetation))
    grid = SquareGrid(rows=9, cols=12, km_per_cell=2.0)
    layers = np.array(combinations, dtype=np.uint8).T.reshape(4, *grid.shape)
    tile_map = TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=layers[0],
        relief=layers[1],
        temperature=layers[2],
        vegetation=layers[3],
        height_m=np.zeros(grid.shape, dtype=np.int16),
    )

    biomes = biome_grid(tile_map)

    expected = [
        [biome_of(tile_map.tile((r, c))) for c in range(grid.cols)] for r in range(grid.rows)
    ]
    assert biomes.tolist() == expected
