"""Seeds a new map with towns and industry: houses, farms, mines, factories, lumber mills and
ports. Track and stations are left for the player.

Each kind of building has a SeedRule saying which kinds of cell suit it: weights for the cell's
relief, vegetation and temperature, for the water next to it and for whether it lies in a town.
A cell's suitability is the product of its weights, from 0 (never) to 1 (best). The more suitable
land a map has, the more of a building it gets, drawn at random with the most suitable cells the
likeliest, and spaced apart. Houses come in towns, which start on suitable cells and grow
outwards along the most suitable land, densest in the middle.
"""

import heapq
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import CELL_SPACE, Building, BuildingKind, Constructions
from railroads.sim.grids import BoolGrid, Cell, Grid
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

FloatGrid = npt.NDArray[np.float64]

ANY_VEGETATION: Mapping[Vegetation, float] = MappingProxyType(dict.fromkeys(Vegetation, 1.0))
ANY_TEMPERATURE: Mapping[Temperature, float] = MappingProxyType(dict.fromkeys(Temperature, 1.0))
ANY_WATER: Mapping[Terrain, float] = MappingProxyType(dict.fromkeys(Terrain, 1.0))
# How far a town's edge wanders from a neat circle, in cells.
TOWN_ROUGHNESS = 1.5


@dataclass(frozen=True, slots=True)
class SeedRule:
    """Which cells suit a kind of building, from weights between 0 and 1; a property value left
    out has weight 0."""

    relief: Mapping[Relief, float]
    vegetation: Mapping[Vegetation, float]
    temperature: Mapping[Temperature, float]
    # How many to seed for every 10,000 km² of fully suitable land, or None for as many as the
    # spacing allows. For houses, this counts towns.
    per_10k_km2: float | None
    spacing_km: float
    # Weight by the water next to the cell, LAND standing for none; with several kinds of water
    # next to it, the best counts.
    water: Mapping[Terrain, float] = ANY_WATER
    # Weights for a cell away from towns and for one in or next to a town.
    town: tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True, slots=True)
class TownSizes:
    """Houses in the biggest town; the n-th biggest gets 1/n of that, but at least `smallest`."""

    largest: int = 240
    smallest: int = 6

    def houses(self, rank: int) -> int:
        return max(self.smallest, round(self.largest / rank))


TOWN_SIZES = TownSizes()


# Seeded in this order, towns first so that what belongs in towns can find them.
SEED_RULES: dict[BuildingKind, SeedRule] = {
    BuildingKind.HOUSE: SeedRule(
        relief={Relief.FLAT: 1.0, Relief.HILLS: 0.4},
        vegetation={Vegetation.GRASS: 1.0, Vegetation.FOREST: 0.4, Vegetation.BARREN: 0.3},
        temperature={Temperature.NORMAL: 1.0, Temperature.HOT: 0.6, Temperature.COLD: 0.3},
        water={Terrain.LAND: 0.3, Terrain.RIVER: 1.0, Terrain.LAKE: 1.0, Terrain.SEA: 1.0},
        per_10k_km2=3.0,
        spacing_km=30.0,
    ),
    BuildingKind.FACTORY: SeedRule(
        relief={Relief.FLAT: 1.0, Relief.HILLS: 0.4},
        vegetation={Vegetation.GRASS: 1.0, Vegetation.BARREN: 1.0, Vegetation.FOREST: 0.3},
        temperature=ANY_TEMPERATURE,
        town=(0.0, 1.0),
        per_10k_km2=60.0,
        spacing_km=12.0,
    ),
    BuildingKind.PORT: SeedRule(
        relief={Relief.FLAT: 1.0, Relief.HILLS: 0.5},
        vegetation=ANY_VEGETATION,
        temperature=ANY_TEMPERATURE,
        water={Terrain.SEA: 1.0, Terrain.LAKE: 0.3},
        town=(0.25, 1.0),
        per_10k_km2=100.0,
        spacing_km=50.0,
    ),
    BuildingKind.FARM: SeedRule(
        relief={Relief.FLAT: 1.0, Relief.HILLS: 0.5},
        vegetation={Vegetation.GRASS: 1.0},
        temperature={Temperature.NORMAL: 1.0, Temperature.HOT: 0.5},
        per_10k_km2=3.0,
        spacing_km=8.0,
    ),
    BuildingKind.LUMBER_MILL: SeedRule(
        relief={Relief.FLAT: 1.0, Relief.HILLS: 1.0},
        vegetation={Vegetation.FOREST: 1.0},
        temperature={Temperature.COLD: 1.0, Temperature.NORMAL: 1.0, Temperature.HOT: 0.6},
        per_10k_km2=1.5,
        spacing_km=20.0,
    ),
    BuildingKind.MINE: SeedRule(
        relief={Relief.HILLS: 0.6, Relief.MOUNTAINS: 1.0},
        vegetation=ANY_VEGETATION,
        temperature=ANY_TEMPERATURE,
        per_10k_km2=1.5,
        spacing_km=25.0,
    ),
}


def seed_buildings(
    tile_map: TileMap,
    rng: np.random.Generator,
    rules: Mapping[BuildingKind, SeedRule] = SEED_RULES,
    towns: TownSizes = TOWN_SIZES,
) -> Constructions:
    constructions = Constructions(tile_map)
    for kind, rule in rules.items():
        scores = suitability(tile_map, rule, town_cells(constructions))
        sites = _pick_sites(
            constructions,
            kind,
            scores,
            _how_many(tile_map.grid, scores, rule),
            rule.spacing_km,
            rng,
        )
        if kind == BuildingKind.HOUSE:
            sites.sort(key=lambda cell: -scores[cell])
            for rank, centre in enumerate(sites, start=1):
                _grow_town(constructions, centre, towns.houses(rank), scores, rng)
        else:
            for cell in sites:
                constructions.build(Building(cell, kind))
    return constructions


def suitability(tile_map: TileMap, rule: SeedRule, in_town: BoolGrid | None = None) -> FloatGrid:
    """How well each cell suits the building, from 0 to 1; without `in_town`, every cell counts
    as away from towns."""
    score = (
        _lookup(rule.relief, Relief)[tile_map.relief]
        * _lookup(rule.vegetation, Vegetation)[tile_map.vegetation]
        * _lookup(rule.temperature, Temperature)[tile_map.temperature]
        * _water_weights(tile_map, rule.water)
    )
    away, near = rule.town
    score = score * (away if in_town is None else np.where(in_town, near, away))
    return np.where(tile_map.terrain == Terrain.LAND, score, 0.0)


def town_cells(constructions: Constructions) -> BoolGrid:
    """Cells with houses on them or next to them."""
    grid = constructions.tile_map.grid
    houses = np.zeros(grid.shape, dtype=bool)
    for built in constructions:
        if isinstance(built, Building) and built.kind == BuildingKind.HOUSE:
            houses[built.cell] = True
    rows, cols = grid.neighbour_index_grids()
    return houses | houses[rows, cols].any(axis=0)


def _lookup[E: IntEnum](weights: Mapping[E, float], values: type[E]) -> FloatGrid:
    return np.array([weights.get(value, 0.0) for value in values])


def _water_weights(tile_map: TileMap, weights: Mapping[Terrain, float]) -> FloatGrid:
    table = _lookup(weights, Terrain)
    rows, cols = tile_map.grid.neighbour_index_grids()
    around = tile_map.terrain[rows, cols]
    is_water = around != Terrain.LAND
    best = np.where(is_water, table[around], 0.0).max(axis=0)
    return np.where(is_water.any(axis=0), best, table[Terrain.LAND])


def _how_many(grid: Grid, scores: FloatGrid, rule: SeedRule) -> int | None:
    if rule.per_10k_km2 is None:
        return None
    suitable_km2 = float(scores.sum()) * grid.cell_area * grid.km_per_cell**2
    return round(rule.per_10k_km2 * suitable_km2 / 10_000)


def _pick_sites(
    constructions: Constructions,
    kind: BuildingKind,
    scores: FloatGrid,
    count: int | None,
    spacing_km: float,
    rng: np.random.Generator,
) -> list[Cell]:
    """Cells for `count` buildings, or as many as fit, drawn at random with the more suitable
    cells likelier, and each at least `spacing_km` from the others."""
    grid = constructions.tile_map.grid
    candidates = np.flatnonzero(scores > 0)
    keys = np.log(rng.random(len(candidates))) / scores.ravel()[candidates]
    x, y = grid.centers()
    blocked = np.zeros(grid.shape, dtype=bool)
    radius = spacing_km / grid.km_per_cell
    sites: list[Cell] = []
    for index in candidates[np.argsort(-keys)].tolist():
        if count is not None and len(sites) >= count:
            break
        cell = divmod(index, grid.cols)
        if blocked[cell] or constructions.why_not(Building(cell, kind)) is not None:
            continue
        sites.append(cell)
        rows = slice(
            max(0, cell[0] - math.ceil(radius * 1.2) - 1), cell[0] + math.ceil(radius * 1.2) + 2
        )
        cols = slice(max(0, cell[1] - math.ceil(radius) - 1), cell[1] + math.ceil(radius) + 2)
        blocked[rows, cols] |= np.hypot(x[rows, cols] - x[cell], y[rows, cols] - y[cell]) < radius
    return sites


def _grow_town(
    constructions: Constructions,
    centre: Cell,
    houses: int,
    scores: FloatGrid,
    rng: np.random.Generator,
) -> None:
    """Fills the centre cell with houses and spreads to the nearest, most suitable cells, with
    fewer houses per cell the further out it gets."""
    grid = constructions.tile_map.grid
    centre_x, centre_y = grid.center(*centre)
    queue: list[tuple[float, Cell]] = [(0.0, centre)]
    queued = {centre}
    left = houses
    while left > 0 and queue:
        _, cell = heapq.heappop(queue)
        wanted = min(left, 1 + round((CELL_SPACE - 1) * left / houses))
        built = 0
        while built < wanted and constructions.why_not(Building(cell, BuildingKind.HOUSE)) is None:
            constructions.build(Building(cell, BuildingKind.HOUSE))
            built += 1
        left -= built
        if not built:
            continue
        for neighbour in grid.neighbours(*cell):
            if neighbour in queued or scores[neighbour] <= 0:
                continue
            queued.add(neighbour)
            x, y = grid.center(*neighbour)
            reach = math.hypot(x - centre_x, y - centre_y) / math.sqrt(scores[neighbour])
            heapq.heappush(queue, (reach + TOWN_ROUGHNESS * rng.random(), neighbour))
