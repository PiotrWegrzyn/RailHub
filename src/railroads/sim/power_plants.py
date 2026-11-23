"""Power plants (railroads.sim.growth puts them up once a year): coal ones from the third era,
diesel ones from the fifth and gas ones from the sixth (railroads.sim.tiles.KIND_ERAS), each burning its fuel and making electricity for the map
(railroads.sim.fuel_market). Whenever the plants standing would make less than the map's
electrified homes and factories want, one more goes up, at most one a year, the first of each kind
as soon as it opens: by whichever of the
PLANT_TOWNS biggest towns with none within PLANT_APART cells has the most fuel made within FUEL_REACH
cells of it, burning that fuel; by the biggest of them, burning coal, where none has any."""

from typing import TYPE_CHECKING

import numpy as np

from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.fuel_market import (
    ELECTRICITY_PER_POINT,
    PLANT_FUEL,
    POWER_PLANT_OUTPUT,
    electrified,
)
from railroads.sim.goods import FITS, MAKES
from railroads.sim.grids import Cell, flat_within
from railroads.sim.points import Points
from railroads.sim.tiles import era_come, kind_era

if TYPE_CHECKING:
    from railroads.sim.growth import Growth

PLANT_APART = 6
FUEL_REACH = 10
# How far from its town a new plant looks for a free cell, and how many of the biggest towns
# without one it chooses among.
PLACE_STEPS = 8
PLANT_TOWNS = 12


def powered_points(growth: Growth) -> float:
    """The points of every home and factory on the map, which want electricity."""
    total = 0.0
    for kind, count in growth.counts.items():
        points, value = growth.rule.points[kind]
        if points in (Points.POPULATION, Points.INDUSTRY):
            total += float(count.sum()) * value
    return total


def electricity_wanted(growth: Growth) -> float:
    """What the map's homes and factories want of electricity a month this year."""
    return ELECTRICITY_PER_POINT * powered_points(growth) * electrified(growth.year)


def plants(growth: Growth) -> np.ndarray:
    """How many power plants of any kind stand in each cell."""
    counts = [growth.counts[kind] for kind in PLANT_FUEL if kind in growth.counts]
    return sum(counts, start=np.zeros(growth.grid.shape, dtype=np.int64))


def build_power_plants(growth: Growth) -> set[Cell]:
    """Puts up one more power plant if those standing would make less than the map wants;
    returns the cells that changed."""
    opened = [kind for kind in PLANT_FUEL if era_come(kind_era(kind), growth.year)]
    if not opened:
        return set()
    standing = plants(growth)
    missing = [kind for kind in opened if not growth.counts[kind].any()]
    if missing:
        return _first_of(growth, missing[0], standing)
    if float(standing.sum()) * POWER_PLANT_OUTPUT >= electricity_wanted(growth):
        return set()
    towns = _towns_without_plant(growth, standing)
    if not towns:
        return set()
    best = max(
        (
            (_fuel_near(growth, town, kind), -rank, town, kind)
            for rank, town in enumerate(towns)
            for kind in opened
        ),
        key=lambda found: found[:2],
    )
    made, _, town, kind = best
    if made <= 0:
        town, kind = towns[0], BuildingKind.POWER_PLANT
    cell = growth.constructions.nearest_free(kind, town, PLACE_STEPS)
    if cell is None:
        return set()
    growth.build(Building(cell, kind))
    return {cell}


def _first_of(growth: Growth, kind: BuildingKind, standing: np.ndarray) -> set[Cell]:
    """Puts up the map's first plant of a kind by whichever of the biggest towns has the most of
    its fuel made near it; returns the cells that changed."""
    towns = _towns_without_plant(growth, standing) or _towns_without_plant(
        growth, np.zeros_like(standing)
    )
    if not towns:
        return set()
    town = max(towns, key=lambda found: _fuel_near(growth, found, kind))
    cell = growth.constructions.nearest_free(kind, town, PLACE_STEPS)
    if cell is None:
        return set()
    growth.build(Building(cell, kind))
    return {cell}


def _towns_without_plant(growth: Growth, standing: np.ndarray) -> list[Cell]:
    """The PLANT_TOWNS cells with the most people around them, biggest first, with no power plant
    near them, apart from each other."""
    near = growth.nearby(Points.POPULATION).astype(np.float64)
    blocked = np.zeros(near.shape, dtype=bool)
    for row, col in np.argwhere(standing > 0).tolist():
        for cell in growth.grid.within((row, col), PLANT_APART):
            blocked[cell] = True
    near[blocked] = -1.0
    towns: list[Cell] = []
    for index in np.argsort(near, axis=None)[::-1].tolist():
        if len(towns) >= PLANT_TOWNS or near.flat[index] <= 0:
            break
        cell = divmod(index, near.shape[1])
        if all(cell not in growth.grid.within(town, PLANT_APART) for town in towns):
            towns.append(cell)
    return towns


def _fuel_near(growth: Growth, town: Cell, kind: BuildingKind) -> float:
    """How much of a plant kind's fuel the buildings within FUEL_REACH of a town make a month."""
    reach = flat_within(growth.grid, town, FUEL_REACH).cells
    goods = FITS[PLANT_FUEL[kind]]
    return sum(
        float(count.ravel()[reach].sum()) * MAKES[maker].get(good, 0.0)
        for maker, count in growth.counts.items()
        if maker in MAKES
        for good in goods
    )
