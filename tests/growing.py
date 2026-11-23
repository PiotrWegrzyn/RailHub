"""What tests of growth stand on: a map to grow on and buildings put up by hand."""

import numpy as np

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.grids import HexGrid
from railroads.sim.ladders import (
    GrowthRule,
)
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

NO_TRADE = GrowthRule(trades=False)


def make_map(size: int = 12) -> TileMap:
    grid = HexGrid(rows=size, cols=size, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    relief[:, -1] = Relief.MOUNTAINS
    relief[0, 6] = Relief.HILLS
    vegetation = np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
    vegetation[1, 6] = Vegetation.FOREST
    vegetation[2, 6] = Vegetation.BARREN
    temperature = np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8)
    temperature[3, 6] = Temperature.COLD
    return flat_map(
        grid=grid, terrain=terrain, vegetation=vegetation, temperature=temperature, relief=relief
    )


def build(constructions: Constructions, cell: tuple[int, int], kind: BuildingKind, n: int) -> None:
    for _ in range(n):
        constructions.build(Building(cell, kind))


def only(*ladders: object) -> GrowthRule:
    return GrowthRule(ladders=tuple(ladders), trades=False)  # type: ignore[arg-type]
