"""How likely every ladder's first kind of building is to come up within a year, in each cell and
on the whole map, and why not where it can't, as railroads.sim.growth decides it month by month.

A cell's chance in a month is its chance of a turn on the ladder (Growth.chances: the land, what is
around, the needs and the player's buildings), none where the ladder's caps leave no room for one
more, cut by the land's open field where something stands on open grassland, and none anywhere
while the ladder has not opened, the map holds as many as it may or nothing on it makes the main
inputs. A year is twelve such months, and the map's count a year the sum of every cell's chances:
the most to expect, as a turn may merge or join buildings instead, or find no room."""

from dataclasses import dataclass, fields, replace
from itertools import pairwise

import numpy as np

from railroads.sim.clock import MONTHS_PER_YEAR
from railroads.sim.constructions import BuildingKind
from railroads.sim.events import military_growth
from railroads.sim.grids import Cell, FloatArray
from railroads.sim.growth import Growth, inputs_of, ladder_open
from railroads.sim.ladder_types import Ladder
from railroads.sim.ladders import MILITARY
from railroads.sim.points import Points
from railroads.sim.surroundings import Land, Surroundings
from railroads.sim.tilemap import Relief, Temperature, Terrain, Vegetation
from railroads.sim.tiles import era_come, kind_era
from railroads.words import either, times_over

WATERS = {Terrain.SEA: "the sea", Terrain.LAKE: "a lake", Terrain.RIVER: "a river"}
# How many kinds of building the map-wide reason names as makers of a missing input.
MOST_MAKERS_NAMED = 3


@dataclass(frozen=True, slots=True)
class LadderOdds:
    """A ladder on the whole map: how many of its buildings stand, how many new ones a year at
    most, the likeliest cell, why none can come up if so, and the other ways its buildings come."""

    ladder: Ladder
    standing: int
    per_year: float
    best: Cell | None
    why_none: str | None
    other_ways: tuple[str, ...]


def monthly_chances(growth: Growth, ladder: Ladder) -> FloatArray:
    """Each flat cell's chance of one more of the ladder's first kind coming up there this month."""
    size = growth.grid.rows * growth.grid.cols
    if _map_reasons(growth, ladder):
        return np.zeros(size)
    chances = growth.chances(ladder).ravel().copy()
    if ladder is MILITARY:
        chances *= military_growth(growth.year)
    chances[_capped(growth, ladder)] = 0.0
    if ladder.land.open_field != 1.0:
        chances[_built_on_open_field(growth)] /= ladder.land.open_field
    return np.minimum(chances, 1.0)


def yearly(monthly: FloatArray) -> FloatArray:
    """The chance of at least one in a year, of cells with these chances a month."""
    return 1.0 - (1.0 - monthly) ** MONTHS_PER_YEAR


def ladder_odds(growth: Growth, ladder: Ladder, monthly: FloatArray) -> LadderOdds:
    """A ladder on the whole map, with its cells' chances this month (monthly_chances)."""
    per_year = float(monthly.sum()) * MONTHS_PER_YEAR
    best = None
    if per_year > 0:
        best = divmod(int(np.argmax(monthly)), growth.grid.cols)
    standing = sum(int(growth.count(kind).sum()) for kind in ladder.kinds)
    return LadderOdds(
        ladder=ladder,
        standing=standing,
        per_year=per_year,
        best=best,
        why_none=None if per_year > 0 else _why_none(growth, ladder),
        other_ways=other_ways(growth, ladder),
    )


def why_not_here(growth: Growth, ladder: Ladder, cell: Cell) -> list[str]:
    """Every reason one more of the ladder's first kind can't come up in a cell; none if it can."""
    reasons = _map_reasons(growth, ladder)
    flat = cell[0] * growth.grid.cols + cell[1]
    if growth.land_weights(ladder)[flat] <= 0:
        reasons += land_reasons(growth, ladder.land, flat)
    for points, needed in ladder.needs.items():
        have = int(growth.nearby(points)[cell])
        if have < needed:
            reasons.append(f"Needs {needed} {points.value} points nearby; there are {have}.")
    people = int(growth.nearby(Points.POPULATION)[cell])
    if ladder.crowded_at is not None and people >= ladder.crowded_at:
        reasons.append(
            f"Too many people nearby: {people} population points, it grows only under"
            f" {ladder.crowded_at}."
        )
    reasons += _cap_reasons(growth, ladder, cell)
    return reasons


def turn_instead(growth: Growth, ladder: Ladder, cell: Cell) -> str | None:
    """What a turn of the ladder in a cell does instead of putting up one more, if anything."""
    neighbourhood = [cell, *growth.grid.neighbours(*cell)]

    def around(kind: BuildingKind) -> int:
        return sum(int(growth.count(kind)[near]) for near in neighbourhood)

    for small, big in reversed(list(pairwise(ladder.kinds))):
        merge = ladder.merges.get(small)
        if merge is not None and around(small) >= merge.needed:
            gone = times_over(small.value, merge.pulled_down)
            return f"A turn here merges {gone} into {_a(big.value)}."
    join = ladder.joins
    if join is not None and around(ladder.kinds[0]) >= join.needed:
        joined = times_over(ladder.kinds[0].value, join.needed)
        return f"A turn here joins {joined} into {_a(join.into.value)}."
    return None


def other_ways(growth: Growth, ladder: Ladder) -> tuple[str, ...]:
    """The other ways the ladder's buildings come up: joins into its first kind, the map's
    least, merges and rises up it."""
    first = ladder.kinds[0]
    ways = [
        f"{times_over(other.kinds[0].value, other.joins.needed)} in a neighbourhood join into one,"
        " once a year as well."
        for other in growth.rule.ladders
        if other.joins is not None and other.joins.into == first
    ]
    least = growth.least_of(ladder)
    tops_up = growth.rule.top_ups is None or first in growth.rule.top_ups
    if least and tops_up:
        ways.append(f"The map keeps at least {least}: one more a year where likeliest.")
    for small, big in pairwise(ladder.kinds):
        merge = ladder.merges.get(small)
        if merge is not None:
            ways.append(
                f"{times_over(small.value, merge.needed)} in a neighbourhood merge into"
                f" {_a(big.value)}."
            )
        rise = ladder.rises.get(small)
        if rise is not None:
            only = ", the first to get there only" if rise.only_one else ""
            ways.append(
                f"{_a(small.value).capitalize()} becomes {_a(big.value)} with {rise.people}"
                " population points"
                f" nearby{only}."
            )
    if ladder.joins is not None:
        join = ladder.joins
        ways.append(
            f"{times_over(ladder.kinds[0].value, join.needed)} in a neighbourhood join into"
            f" {_a(join.into.value)}."
        )
    return tuple(ways)


def land_reasons(growth: Growth, land: Land, flat: int) -> list[str]:
    """Why a land weighs nothing in a flat cell: each of its rules that alone rules the cell out."""
    tile_map = growth.constructions.tile_map
    if tile_map.locked is not None and tile_map.locked.ravel()[flat]:
        return ["This land is not bought yet."]
    around = Surroundings(tile_map, np.array([flat]), growth.near)
    if not around.land[0]:
        return ["Nothing grows on the water."]
    bare = Land()
    reasons = []
    for rule in fields(Land):
        if rule.name == "open_field" or getattr(land, rule.name) == getattr(bare, rule.name):
            continue
        alone = replace(bare, **{rule.name: getattr(land, rule.name)})
        if alone.weights_in(around)[0] <= 0:
            reasons.append(_land_reason(rule.name, land, around))
    return reasons or ["The land here does not suit it."]


def _land_reason(rule: str, land: Land, around: Surroundings) -> str:
    match rule:
        case "vegetation":
            return f"Not on {Vegetation(int(around.vegetation[0])).name.lower()}."
        case "relief":
            relief = Relief(int(around.relief[0]))
            return (
                "Not on flat land." if relief == Relief.FLAT else f"Not in {relief.name.lower()}."
            )
        case "temperature":
            return f"Not in a {Temperature(int(around.temperature[0])).name.lower()} climate."
        case "water":
            waters = [WATERS[t] for t, w in land.water.items() if w > 0 and t in WATERS]
            return f"Only beside {either(waters)}." if waters else "Not beside this water."
        case "beside_mountains":
            return "Not beside mountains."
        case "forest_edge":
            return "Only where a forest meets open flat land."
        case "least_forest_nearby":
            return (
                f"Needs {land.least_forest_nearby} forest cells around; there are"
                f" {int(around.forests[0])}."
            )
        case "deposit":
            assert land.deposit is not None
            return f"Only over {land.deposit.name.lower()} deposits."
        case "most_latitude":
            return f"Only within {land.most_latitude:g}° of the equator."
        case _:
            return "Only in its own part of the world."


def _map_reasons(growth: Growth, ladder: Ladder) -> list[str]:
    """Why none of the ladder's first kind can come up anywhere on the map, whatever the land."""
    first = ladder.kinds[0]
    year = growth.year
    reasons = []
    if ladder.from_year > year:
        reasons.append(f"Grows from {ladder.from_year}.")
    era = kind_era(first)
    if not era_come(era, year):
        reasons.append(f"Comes with era {era}.")
    if not ladder_open(ladder, year):
        return reasons
    for makers in inputs_of(first, year):
        if not any(int(growth.counts[maker].sum()) for maker in makers if maker in growth.counts):
            named = sorted(maker.value for maker in makers)[:MOST_MAKERS_NAMED]
            reasons.append(f"Needs {_a(either(named))} somewhere on the map.")
    each = ladder.land_cells_each
    if each is not None and growth.as_many_as_first(ladder) >= growth.land_cells // each:
        reasons.append(f"The map holds one for every {each:,} land cells at most, and has as many.")
    return reasons


def _why_none(growth: Growth, ladder: Ladder) -> str:
    """The first reason none of the ladder's first kind can come up anywhere."""
    reasons = _map_reasons(growth, ladder)
    if reasons:
        return reasons[0]
    suits = growth.land_weights(ladder) > 0
    if not suits.any():
        return "No land on this map suits it."
    met = suits.copy()
    for points, needed in ladder.needs.items():
        has = growth.nearby(points).ravel() >= needed
        if not (suits & has).any():
            return (
                f"Needs {needed} {points.value} points nearby, and no land that suits it has them."
            )
        met &= has
    if not met.any():
        return "No land that suits it has all it needs nearby."
    if ladder.crowded_at is not None:
        uncrowded = growth.nearby(Points.POPULATION).ravel() < ladder.crowded_at
        if not (met & uncrowded).any():
            return "Too many people live everywhere it could grow."
    return "Everywhere it could grow is full, or too close to another."


def _capped(growth: Growth, ladder: Ladder) -> np.ndarray:
    """Which flat cells the ladder's caps leave no room in for one more of its first kind."""
    first = growth.count(ladder.kinds[0]).ravel()
    capped = np.zeros(first.size, dtype=bool)
    if ladder.most_per_cell is not None:
        held = (
            sum(growth.count(kind).ravel() for kind in ladder.kinds)
            if ladder.one_per_cell
            else first
        )
        capped |= held >= ladder.most_per_cell
    if ladder.most_nearby is not None:
        padded = np.append(first, 0)
        capped |= first + padded[growth.near].sum(axis=0) >= ladder.most_nearby
    if ladder.apart is not None:
        standing = sum(growth.count(kind) for kind in ladder.kinds)
        cols = growth.grid.cols
        for row, col in np.argwhere(standing > 0).tolist():
            for near in growth.grid.within((row, col), ladder.apart):
                capped[near[0] * cols + near[1]] = True
    return capped


def _cap_reasons(growth: Growth, ladder: Ladder, cell: Cell) -> list[str]:
    first = ladder.kinds[0]
    reasons = []
    kinds = ladder.kinds if ladder.one_per_cell else (first,)
    here = sum(int(growth.count(kind)[cell]) for kind in kinds)
    if ladder.most_per_cell is not None and here >= ladder.most_per_cell:
        reasons.append(f"{here} stand here already, the most a cell holds.")
    if ladder.most_nearby is not None:
        nearby = sum(
            int(growth.count(first)[near]) for near in [cell, *growth.grid.neighbours(*cell)]
        )
        if nearby >= ladder.most_nearby:
            reasons.append(f"{nearby} stand nearby already, the most a neighbourhood holds.")
    if ladder.apart is not None:
        reached = growth.grid.within(cell, ladder.apart)
        if any(int(growth.count(kind)[near]) for kind in ladder.kinds for near in reached):
            reasons.append(f"Another stands within {ladder.apart} cells.")
    return reasons


def _built_on_open_field(growth: Growth) -> np.ndarray:
    """Which flat cells are open flat grassland with something on them, where only one turn in a
    land's open field goes ahead."""
    tile_map = growth.constructions.tile_map
    cols = growth.grid.cols
    built = np.zeros(growth.grid.rows * cols, dtype=bool)
    for (row, col), here in growth.constructions.cells():
        if here:
            built[row * cols + col] = True
    flat_grass = (tile_map.relief == Relief.FLAT) & (tile_map.vegetation == Vegetation.GRASS)
    return built & flat_grass.ravel()


def _a(name: str) -> str:
    """A name with "a" or "an" before it."""
    return f"an {name}" if name[:1] in "aeiou" else f"a {name}"
