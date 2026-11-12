"""Rare resources: each new map gets its deposits from its own geography and its seed.

Every resource has the land it can lie in (allowed_land) and a number of clusters for the map's
land (DEPOSITS), or for the land it can lie in: fishing grounds come with the length of rivers,
lakes and coast. Cluster centres are drawn at random on that land, kept apart so they spread over
the map, and each grows into a small blob. Where little land allows a resource the clusters sit
closer together; clusters that no longer fit are left out, and a map with no such land has none
of that resource.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.tilemap import Deposit, Relief, Temperature, Terrain, TileMap, Vegetation

type Mask = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class DepositRule:
    # One cluster for this many land cells, of between so many cells.
    land_cells_per_cluster: int
    size: tuple[int, int]
    # Whether the land counted is only the land the resource can lie in.
    on_its_land: bool = False


# Rarest first, so they get the land they want before commoner ones spread over it.
DEPOSITS: dict[Deposit, DepositRule] = {
    Deposit.DIAMONDS: DepositRule(130_000, (2, 4)),
    Deposit.INCENSE: DepositRule(65_000, (2, 4)),
    Deposit.GOLD: DepositRule(65_000, (3, 8)),
    Deposit.RUBBER: DepositRule(40_000, (5, 12)),
    Deposit.COFFEE: DepositRule(40_000, (5, 12)),
    Deposit.COTTON: DepositRule(30_000, (6, 14)),
    Deposit.OIL: DepositRule(45_000, (5, 12)),
    Deposit.COPPER: DepositRule(45_000, (4, 10)),
    Deposit.SALT: DepositRule(40_000, (5, 10)),
    Deposit.GRAPES: DepositRule(30_000, (5, 12)),
    Deposit.IRON: DepositRule(30_000, (6, 14)),
    Deposit.WOOL: DepositRule(12_000, (6, 12)),
    Deposit.FISH: DepositRule(300, (6, 14), on_its_land=True),
}
# Military ground, whose few clusters hold every outpost on the map: how many, and of how many cells.
GARRISONS = (1, 3)
GARRISON_SIZE = (10, 18)
# How far apart clusters are kept, as a share of the even spacing over all the map's land.
SPREAD = 0.6


def allowed_land(tile_map: TileMap) -> dict[Deposit, Mask]:
    """The cells each resource can lie in."""
    rows, cols = tile_map.grid.neighbour_index_grids()
    land = tile_map.terrain == Terrain.LAND
    relief, vegetation, temperature = tile_map.relief, tile_map.vegetation, tile_map.temperature
    mountains = land & (relief == Relief.MOUNTAINS)
    hills = land & (relief == Relief.HILLS)
    rocky = mountains | (hills & mountains[rows, cols].any(axis=0))
    by_water = (~land)[rows, cols].any(axis=0)
    plains = land & (relief != Relief.MOUNTAINS) & ~by_water
    hot = temperature == Temperature.HOT
    open_land = vegetation != Vegetation.FOREST
    return {
        Deposit.IRON: rocky,
        Deposit.COPPER: rocky,
        Deposit.GOLD: rocky,
        Deposit.DIAMONDS: rocky,
        Deposit.OIL: plains,
        Deposit.SALT: plains,
        Deposit.FISH: land & by_water & (relief != Relief.MOUNTAINS),
        Deposit.WOOL: hills & (vegetation == Vegetation.GRASS),
        Deposit.GRAPES: hills & (temperature == Temperature.NORMAL) & open_land,
        Deposit.COTTON: land & hot & (relief == Relief.FLAT) & open_land,
        Deposit.COFFEE: hills & hot,
        Deposit.RUBBER: land
        & hot
        & (vegetation == Vegetation.FOREST)
        & (relief != Relief.MOUNTAINS),
        Deposit.GARRISON: land & (relief != Relief.MOUNTAINS),
        Deposit.INCENSE: land & (relief != Relief.MOUNTAINS),
    }


def place_deposits(tile_map: TileMap, rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Every cell's Deposit for a new map."""
    grid = tile_map.grid
    layer = np.full(grid.shape, Deposit.NONE, dtype=np.uint8)
    land_cells = int((tile_map.terrain == Terrain.LAND).sum())
    allowed = allowed_land(tile_map)
    garrisons = int(rng.integers(GARRISONS[0], GARRISONS[1] + 1))
    _place_clusters(layer, allowed, Deposit.GARRISON, garrisons, GARRISON_SIZE, tile_map, rng)
    for deposit, rule in DEPOSITS.items():
        counted = int(allowed[deposit].sum()) if rule.on_its_land else land_cells
        wanted = round(counted / rule.land_cells_per_cluster)
        _place_clusters(layer, allowed, deposit, wanted, rule.size, tile_map, rng)
    return layer


def _place_clusters(
    layer: npt.NDArray[np.uint8],
    allowed: dict[Deposit, Mask],
    deposit: Deposit,
    wanted: int,
    sizes: tuple[int, int],
    tile_map: TileMap,
    rng: np.random.Generator,
) -> None:
    """Grows up to `wanted` clusters of a deposit, spread apart over the land it allows."""
    grid = tile_map.grid
    land_cells = int((tile_map.terrain == Terrain.LAND).sum())
    rows, cols = np.nonzero(allowed[deposit] & (layer == Deposit.NONE))
    if not wanted or not rows.size:
        return
    xs, ys = grid.center(rows, cols)
    spacing = min(SPREAD * math.sqrt(land_cells / wanted), math.sqrt(rows.size / wanted))
    free = np.ones(rows.size, dtype=bool)
    for _ in range(wanted):
        left = np.flatnonzero(free)
        if not left.size:
            break
        pick = int(left[rng.integers(left.size)])
        size = int(rng.integers(sizes[0], sizes[1] + 1))
        center = (int(rows[pick]), int(cols[pick]))
        _grow(layer, allowed[deposit], center, deposit, size, tile_map, rng)
        free &= np.hypot(xs - xs[pick], ys - ys[pick]) >= spacing


def _grow(
    layer: npt.NDArray[np.uint8],
    allowed: Mask,
    center: tuple[int, int],
    deposit: Deposit,
    size: int,
    tile_map: TileMap,
    rng: np.random.Generator,
) -> None:
    """Grows a cluster out from its centre over land that allows it."""
    layer[center] = deposit
    edge = [center]
    placed = 1
    while placed < size and edge:
        cell = edge[int(rng.integers(len(edge)))]
        options = [
            near
            for near in tile_map.grid.neighbours(*cell)
            if allowed[near] and layer[near] == Deposit.NONE
        ]
        if not options:
            edge.remove(cell)
            continue
        near = options[int(rng.integers(len(options)))]
        layer[near] = deposit
        edge.append(near)
        placed += 1
