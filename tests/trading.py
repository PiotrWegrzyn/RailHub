"""What tests of trade stand on: a map of farms and towns, goods held at a station, and counts
of what was traded."""

import numpy as np

from builders import flat_map
from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import (
    COLLECTION_RADIUS,
    Economy,
    StationOrder,
)
from railroads.sim.goods import Good
from railroads.sim.grids import HexGrid
from railroads.sim.tilemap import Relief, Terrain, TileMap, Vegetation

STATION = (8, 10)


def make_map(size: int = 12) -> TileMap:
    grid = HexGrid(rows=size, cols=size, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    relief[:, -3:] = Relief.HILLS
    relief[:, -1] = Relief.MOUNTAINS
    vegetation = np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
    vegetation[:6, 4:8] = Vegetation.FOREST
    return flat_map(grid=grid, terrain=terrain, vegetation=vegetation, relief=relief)


def counts(tile_map: TileMap, *placed: tuple[tuple[int, int], BuildingKind, int]) -> dict:
    grids = {kind: np.zeros(tile_map.grid.shape, dtype=np.int64) for kind in BuildingKind}
    for cell, kind, n in placed:
        grids[kind][cell] += n
    return grids


def flat(economy: Economy, cell: tuple[int, int]) -> int:
    return int(np.ravel_multi_index(cell, economy.tile_map.grid.shape))


def position(trade: object, economy: Economy, cell: tuple[int, int]) -> int:
    return int(np.searchsorted(trade.cells, flat(economy, cell)))  # type: ignore[attr-defined]


def holding(room: float = 0.0, **goods: float) -> dict:
    brought = {Good[name]: amount for name, amount in goods.items()}
    return {STATION: StationOrder(brought, room, delivery=COLLECTION_RADIUS)}
