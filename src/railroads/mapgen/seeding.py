"""Seeds a new map with its towns and industry by letting it grow (railroads.sim.growth) for the
years before play: houses, farms, orchards, mines, lumber mills, ports and factories all come up
where chance, the land and their neighbours take them. Track and stations are left for the
player.
"""

import numpy as np

from railroads.sim.constructions import Constructions
from railroads.sim.growth import GROWTH, MONTHS_BEFORE_PLAY, Growth, GrowthRule
from railroads.sim.tilemap import TileMap


def seed_buildings(
    tile_map: TileMap,
    rng: np.random.Generator,
    growth: GrowthRule = GROWTH,
    months: int = MONTHS_BEFORE_PLAY,
) -> Constructions:
    constructions = Constructions(tile_map)
    Growth(constructions, growth).run(months, rng)
    return constructions
