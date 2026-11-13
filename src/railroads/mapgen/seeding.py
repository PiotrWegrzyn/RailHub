"""Seeds a new map with its towns and industry by letting it grow (railroads.sim.growth) for the
years before play, from HISTORY_START on: houses, farms, orchards, mines, lumber mills, ports and factories all come up
where chance, the land and their neighbours take them. Stations and track are left for the player.
"""

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from railroads.mapgen.deposits import place_deposits
from railroads.sim.clock import HISTORY_START
from railroads.sim.constructions import Constructions
from railroads.sim.growth import MONTHS_BEFORE_PLAY, Growth
from railroads.sim.ladders import GROWTH, GrowthRule
from railroads.sim.stats import History
from railroads.sim.steps import Steps, finish
from railroads.sim.tilemap import TileMap

# Before play, the map grows this many years at a step: goods are shared out, forests change and
# buildings come up once a step, at the chances of every month of it, to seed quickly.
HISTORY_STEP_YEARS = 5
HISTORY_STEP = HISTORY_STEP_YEARS * 12


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
) -> Steps[Constructions]:
    """seed_buildings in steps, a pause after each HISTORY_STEP, telling `progress` the share of
    the months grown so far."""
    tile_map = replace(
        tile_map, vegetation=tile_map.vegetation.copy(), deposits=place_deposits(tile_map, rng)
    )
    constructions = Constructions(tile_map)
    rule = replace(growth, trade_every_months=HISTORY_STEP, forests_every_months=HISTORY_STEP)
    grown = Growth(constructions, rule, start_year=HISTORY_START, history=history)
    for done in range(0, months, HISTORY_STEP):
        progress(done / months if months else 1.0)
        yield
        yield from grown.month_steps(rng, min(HISTORY_STEP, months - done))
    progress(1.0)
    return constructions


def play_begins(months: int = MONTHS_BEFORE_PLAY) -> int:
    """The year play begins after `months` of history."""
    return HISTORY_START + months // 12
