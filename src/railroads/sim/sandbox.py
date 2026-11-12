"""A small world to balance the economy in: a piece of a start map that grows only houses, farms,
orchards and fishing grounds (FOOD_ONLY), so food alone decides how towns rise and die out."""

from dataclasses import replace

import numpy as np

from railroads.sim.constructions import BuildingKind
from railroads.sim.geo import Bounds
from railroads.sim.ladders import FARMS, ORCHARDS, RESOURCES, TOWNS, GrowthRule
from railroads.sim.tilemap import Terrain, TileMap

SANDBOX_SIZE = 50

HOUSES_ONLY = replace(TOWNS, kinds=(BuildingKind.HOUSE,), merges={}, rises={}, skyscrapers=False)
FISHERIES = next(ladder for ladder in RESOURCES if ladder.kinds == (BuildingKind.FISHERY,))
FOOD_ONLY = GrowthRule(ladders=(HOUSES_ONLY, FARMS, ORCHARDS, FISHERIES))


def small_map(tile_map: TileMap, size: int = SANDBOX_SIZE) -> TileMap:
    """The size by size piece of a map with the most river and land in it, rows starting even so
    the hexes keep their shape."""
    land = tile_map.terrain == Terrain.LAND
    river = tile_map.terrain == Terrain.RIVER
    score = land.astype(np.int64) + 20 * river
    summed = np.pad(score.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    windows = (
        summed[size:, size:]
        - summed[:-size, size:]
        - summed[size:, :-size]
        + summed[:-size, :-size]
    )
    windows[1::2] = -1
    top, left = np.unravel_index(int(np.argmax(windows)), windows.shape)
    rows, cols = slice(top, top + size), slice(left, left + size)
    west, north = tile_map.lonlat((top, left))
    east, south = tile_map.lonlat((top + size - 1, left + size - 1))
    return replace(
        tile_map,
        name=f"{tile_map.name}, {size}x{size}",
        grid=replace(tile_map.grid, rows=size, cols=size),
        bounds=Bounds(west=west, east=east, south=south, north=north),
        terrain=tile_map.terrain[rows, cols].copy(),
        vegetation=tile_map.vegetation[rows, cols].copy(),
        temperature=tile_map.temperature[rows, cols].copy(),
        height_m=tile_map.height_m[rows, cols].copy(),
        relief=tile_map.relief[rows, cols].copy(),
        deposits=None,
    )
