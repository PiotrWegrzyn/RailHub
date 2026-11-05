"""Poland at 2 km per tile, baked from public-domain data by tools/build_poland_map.py.

Elevation: NOAA ETOPO1. Lakes and rivers: Natural Earth.
"""

from pathlib import Path

import numpy as np

from railroads.sim.geo import Projection
from railroads.sim.world import World

PROJECTION = Projection(west=13.6, east=24.4, south=48.9, north=55.0)
TERRAIN_FILE = Path(__file__).with_name("poland.npz")


def generate() -> World:
    with np.load(TERRAIN_FILE) as data:
        terrain = data["terrain"]
    height, width = terrain.shape
    return World(width=width, height=height, terrain=terrain)
