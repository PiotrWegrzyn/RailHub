"""Seeds a new map with its towns and industry by letting it grow (railroads.sim.growth) for the
years before play, from HISTORY_START on: houses, farms, orchards, mines, lumber mills, ports and factories all come up
where chance, the land and their neighbours take them. Stations and track are left for the player.
"""

from dataclasses import replace

import numpy as np

from railroads.mapgen.deposits import place_deposits
from railroads.sim.clock import HISTORY_START
from railroads.sim.constructions import Constructions
from railroads.sim.growth import MONTHS_BEFORE_PLAY, Growth
from railroads.sim.ladders import GROWTH, GrowthRule
from railroads.sim.stats import History
from railroads.sim.tilemap import TileMap

# Before play, goods are shared out every two years and forests change once a year, to
# seed quickly.
TRADE_EVERY_MONTHS = 24
FORESTS_EVERY_MONTHS = 12


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
    tile_map = replace(
        tile_map, vegetation=tile_map.vegetation.copy(), deposits=place_deposits(tile_map, rng)
    )
    constructions = Constructions(tile_map)
    rule = replace(
        growth,
        trade_every_months=TRADE_EVERY_MONTHS,
        forests_every_months=FORESTS_EVERY_MONTHS,
    )
    Growth(constructions, rule, start_year=HISTORY_START, history=history).run(months, rng)
    return constructions


def play_begins(months: int = MONTHS_BEFORE_PLAY) -> int:
    """The year play begins after `months` of history."""
    return HISTORY_START + months // 12
