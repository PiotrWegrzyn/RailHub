"""Forests come and go with the people around them (railroads.sim.growth runs them).

Each month a forest cell may be cleared, by chance: CLEARING_PER_POINT for every population or
industry point in its neighbourhood (the cell and the cells next to it), and CLEARING_PER_MILL,
110 times as much, for every logging camp there, so a camp among four to seven forest cells clears
one every five to ten years. Open grassland may turn to forest, by SPREAD_PER_FOREST for every
forest cell next to it, where nothing is built. A logging camp with fewer than MILLS_CLOSE_BELOW
forest cells in its neighbourhood closes, and another comes up in the nearest cell one could go
up in (Growth.fits), picked at random among those as near. POPULATION_PER_CLEARING_POINT
population points clear a forest as fast as one point of industry.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.grids import Cell, FlatIndices
from railroads.sim.points import Points
from railroads.sim.tilemap import Terrain, TileMap, Vegetation

if TYPE_CHECKING:
    from railroads.sim.growth import Growth

CLEARING_PER_POINT = 0.00002
CLEARING_PER_MILL = 110 * CLEARING_PER_POINT
SPREAD_PER_FOREST = 0.00004
MILLS_CLOSE_BELOW = 3
# How far from a closed logging camp the next one is looked for, in cells.
NEXT_CAMP_STEPS = 12
POPULATION_PER_CLEARING_POINT = 4


@dataclass(frozen=True, slots=True)
class ForestRule:
    clearing_per_point: float = CLEARING_PER_POINT
    clearing_per_mill: float = CLEARING_PER_MILL
    spread_per_forest: float = SPREAD_PER_FOREST
    mills_close_below: int = MILLS_CLOSE_BELOW


FORESTS = ForestRule()


def forests_nearby(
    tile_map: TileMap, near: FlatIndices, cells: FlatIndices | None = None
) -> npt.NDArray[np.int64]:
    """How many forest cells each cell's neighbourhood holds, as a flat array, or each of the
    given flat cells alone. `near` holds every cell's neighbours as flat indices, one past the
    last cell standing for off the map."""
    forest = (tile_map.vegetation == Vegetation.FOREST) & (tile_map.terrain == Terrain.LAND)
    padded = np.append(forest.ravel(), False)
    if cells is None:
        return forest.ravel().astype(np.int64) + padded[near].sum(axis=0)
    return forest.ravel()[cells].astype(np.int64) + padded[near[:, cells]].sum(axis=0)


def forest_turn(
    tile_map: TileMap,
    near: FlatIndices,
    people: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    mills: npt.NDArray[np.int64],
    rule: ForestRule,
    months: int,
    rng: np.random.Generator,
) -> tuple[FlatIndices, FlatIndices]:
    """Which forest cells are cleared and which grassland cells turn to forest over `months`,
    from the population and industry points and the logging camps in each neighbourhood. Says
    nothing of what stands in the cells; the vegetation is left as it is."""
    land = tile_map.terrain.ravel() == Terrain.LAND
    vegetation = tile_map.vegetation.ravel()
    is_forest = land & (vegetation == Vegetation.FOREST)
    forest = np.flatnonzero(is_forest & ((people > 0) | (mills > 0)))
    clearing = rule.clearing_per_point * people[forest] + rule.clearing_per_mill * mills[forest]
    cleared = forest[rng.random(forest.size) < _over(clearing, months)]
    grass = np.flatnonzero(land & (vegetation == Vegetation.GRASS))
    beside = np.bincount(near[:, np.flatnonzero(is_forest)].ravel(), minlength=near.shape[1] + 1)
    beside = beside[grass]
    open_ = grass[beside > 0]
    spread = rule.spread_per_forest * beside[beside > 0]
    grown = open_[rng.random(open_.size) < _over(spread, months)]
    return cleared, grown


def _over(chance: npt.NDArray[np.float64], months: int) -> npt.NDArray[np.float64]:
    """The chance of it happening at least once in `months` with this chance each month."""
    if months == 1:
        return chance
    return 1.0 - (1.0 - np.minimum(chance, 1.0)) ** months


def change_forests(
    growth: Growth, rule: ForestRule, months: int, rng: np.random.Generator
) -> set[Cell]:
    """Clears and spreads forests over `months`, closes the logging camps left without
    forest, and has growth weigh the land again around them; returns the cells that changed."""
    constructions = growth.constructions
    tile_map, grid, near = constructions.tile_map, growth.grid, growth.near
    people = (
        growth.nearby(Points.POPULATION) / POPULATION_PER_CLEARING_POINT
        + growth.nearby(Points.INDUSTRY)
    ).ravel()
    mills = growth.nearby(Points.TIMBER).ravel()
    cleared, grown = forest_turn(tile_map, near, people, mills, rule, months, rng)
    cols = grid.cols
    grown = np.array(
        [index for index in grown.tolist() if not constructions.at(divmod(index, cols))],
        dtype=np.intp,
    )
    if not cleared.size and not grown.size:
        return set()
    tile_map.set_vegetation(cleared, Vegetation.GRASS)
    tile_map.set_vegetation(grown, Vegetation.FOREST)
    changed_cells = np.concatenate([cleared, grown])
    growth.forests_changed = {divmod(index, cols) for index in changed_cells.tolist()}
    changed = set(growth.forests_changed)
    if BuildingKind.LOGGING_CAMP in growth.counts:
        count = growth.counts[BuildingKind.LOGGING_CAMP]
        mills = np.flatnonzero(count)
        forests = forests_nearby(tile_map, near, mills)
        closing = mills[forests < rule.mills_close_below]
        closed = []
        for row, col in (divmod(int(index), cols) for index in closing):
            for _ in range(int(count[row, col])):
                mill = Building((row, col), BuildingKind.LOGGING_CAMP)
                growth.take_down(mill)
                closed.append((row, col))
            changed.add((row, col))
        growth.reweigh_land(changed_cells)
        for cell in closed:
            changed |= _next_camp(growth, cell, rng)
        return changed
    growth.reweigh_land(changed_cells)
    return changed


def _next_camp(growth: Growth, closed: Cell, rng: np.random.Generator) -> set[Cell]:
    """Puts up a logging camp in place of one closed in a cell: in the nearest cell one could go
    up in, picked at random among those as near; returns the cell, if it found one."""
    ladder = next(
        (ladder for ladder in growth.rule.ladders if ladder.kinds[0] == BuildingKind.LOGGING_CAMP),
        None,
    )
    if ladder is None:
        return set()
    weights = growth.land_weights(ladder)
    cols = growth.grid.cols
    by_steps: dict[int, list[Cell]] = {}
    for cell, steps in growth.grid.within(closed, NEXT_CAMP_STEPS).items():
        if steps and weights[cell[0] * cols + cell[1]] > 0:
            by_steps.setdefault(steps, []).append(cell)
    for steps in sorted(by_steps):
        cells = [cell for cell in by_steps[steps] if growth.fits(ladder, cell)]
        if cells:
            cell = cells[int(rng.integers(len(cells)))]
            growth.build(Building(cell, BuildingKind.LOGGING_CAMP))
            return {cell}
    return set()
