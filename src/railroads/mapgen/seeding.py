"""Seeds a new map with its towns and industry by letting it grow (railroads.sim.growth) for the
years before play, from HISTORY_START on: houses, farms, orchards, mines, lumber mills, ports and factories all come up
where chance, the land and their neighbours take them. Stations and track are left for the player.
"""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from railroads.mapgen.deposits import place_deposits
from railroads.sim.clock import HISTORY_START
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.grids import Cell, Grid
from railroads.sim.growth import MONTHS_BEFORE_PLAY, Growth
from railroads.sim.ladders import GROWTH, MILITARY, GrowthRule
from railroads.sim.stats import History
from railroads.sim.steps import Steps, finish
from railroads.sim.tilemap import Deposit, TileMap

# Before play, the map grows this many years at a step: goods are shared out, forests change and
# buildings come up once a step, at the chances of every month of it, to seed quickly.
HISTORY_STEP_YEARS = 5
HISTORY_STEP = HISTORY_STEP_YEARS * 12

# The outposts each patch of military ground starts play with beside its one base.
STARTING_OUTPOSTS = 2


def seed_buildings(
    tile_map: TileMap,
    rng: np.random.Generator,
    growth: GrowthRule = GROWTH,
    months: int = MONTHS_BEFORE_PLAY,
    history: History | None = None,
) -> Constructions:
    """Grows the map's buildings for `months` on a copy of the map, whose forests change as
    they grow; the copy is the constructions' tile_map. Each year's population goes into
    `history` if given."""
    return finish(seed_steps(tile_map, rng, growth, months, history))


def seed_steps(
    tile_map: TileMap,
    rng: np.random.Generator,
    growth: GrowthRule = GROWTH,
    months: int = MONTHS_BEFORE_PLAY,
    history: History | None = None,
    progress: Callable[[float], None] = lambda done: None,
    look: Callable[[Constructions, int], None] = lambda constructions, year: None,
) -> Steps[Constructions]:
    """seed_buildings in steps, a pause after each HISTORY_STEP, telling `progress` the share of
    the months grown so far and showing `look` the buildings and the year they have grown to."""
    tile_map = replace(
        tile_map, vegetation=tile_map.vegetation.copy(), deposits=place_deposits(tile_map, rng)
    )
    constructions = Constructions(tile_map)
    rule = replace(growth, trade_every_months=HISTORY_STEP, forests_every_months=HISTORY_STEP)
    grown = Growth(constructions, rule, start_year=HISTORY_START, history=history)
    for done in range(0, months, HISTORY_STEP):
        progress(done / months if months else 1.0)
        look(constructions, grown.year)
        yield
        yield from grown.month_steps(rng, min(HISTORY_STEP, months - done))
    muster_garrisons(constructions, play_begins(months), rng)
    look(constructions, grown.year)
    progress(1.0)
    return constructions


def muster_garrisons(constructions: Constructions, year: int, rng: np.random.Generator) -> None:
    """Gives every patch of military ground one base and STARTING_OUTPOSTS outposts around it in
    place of whatever military grew there, when play begins once the military has come
    (MILITARY.from_year); they grow from there in play."""
    if year < MILITARY.from_year or constructions.tile_map.deposits is None:
        return
    for building in list(constructions):
        if isinstance(building, Building) and building.kind in MILITARY.kinds:
            constructions.remove(building)
    garrison = constructions.tile_map.deposits == Deposit.GARRISON
    wanted = (BuildingKind.BASE,) + (BuildingKind.OUTPOST,) * STARTING_OUTPOSTS
    for patch in _patches(garrison, constructions.tile_map.grid):
        free = iter(rng.permutation(len(patch)))
        for kind in wanted:
            for index in free:
                building = Building(patch[index], kind)
                if constructions.why_not(building) is None:
                    constructions.build(building)
                    break


def _patches(ground: npt.NDArray[np.bool_], grid: Grid) -> list[list[Cell]]:
    """The cells of `ground`, in patches of neighbouring cells."""
    seen: set[Cell] = set()
    patches = []
    for start in map(tuple, np.argwhere(ground)):
        cell = (int(start[0]), int(start[1]))
        if cell in seen:
            continue
        seen.add(cell)
        patch, edge = [], [cell]
        while edge:
            here = edge.pop()
            patch.append(here)
            for near in grid.neighbours(*here):
                if near not in seen and ground[near]:
                    seen.add(near)
                    edge.append(near)
        patches.append(sorted(patch))
    return patches


def play_begins(months: int = MONTHS_BEFORE_PLAY) -> int:
    """The year play begins after `months` of history."""
    return HISTORY_START + months // 12
