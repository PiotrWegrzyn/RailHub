import itertools

import numpy as np
import pytest

from railroads.sim.geo import Bounds
from railroads.sim.grids import SquareGrid
from railroads.sim.tilemap import Temperature, Terrain, Tile, TileMap, Vegetation
from railroads.ui.biomes import BIOME_COLORS, Biome, biome_grid, biome_of


@pytest.mark.parametrize(
    ("temperature", "vegetation"), list(itertools.product(Temperature, Vegetation))
)
def test_a_river_always_looks_like_a_river(
    temperature: Temperature, vegetation: Vegetation
) -> None:
    tile = Tile(Terrain.RIVER, vegetation, temperature, height_m=0)

    biome = biome_of(tile)

    assert biome is Biome.RIVER


@pytest.mark.parametrize(
    ("temperature", "vegetation", "expected"),
    [
        (Temperature.HOT, Vegetation.BARREN, Biome.DESERT),
        (Temperature.HOT, Vegetation.FOREST, Biome.JUNGLE),
        (Temperature.COLD, Vegetation.FOREST, Biome.TAIGA),
        (Temperature.NORMAL, Vegetation.GRASS, Biome.GRASSLAND),
    ],
)
def test_land_biome_comes_from_temperature_and_vegetation(
    temperature: Temperature, vegetation: Vegetation, expected: Biome
) -> None:
    tile = Tile(Terrain.LAND, vegetation, temperature, height_m=0)

    biome = biome_of(tile)

    assert biome is expected


def test_every_biome_has_a_colour() -> None:
    colored = set(BIOME_COLORS)

    assert colored == set(Biome)


def test_biome_grid_matches_the_biome_of_every_cell() -> None:
    grid = SquareGrid(rows=4, cols=9, km_per_cell=2.0)
    cells = list(itertools.product(Terrain, Temperature, Vegetation))[: grid.rows * grid.cols]
    layers = np.array(cells, dtype=np.uint8).T.reshape(3, *grid.shape)
    tile_map = TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=layers[0],
        temperature=layers[1],
        vegetation=layers[2],
        height_m=np.zeros(grid.shape, dtype=np.int16),
    )

    biomes = biome_grid(tile_map)

    expected = [
        [biome_of(tile_map.tile((r, c))) for c in range(grid.cols)] for r in range(grid.rows)
    ]
    assert biomes.tolist() == expected
