"""How the shape of the land shows on the map. Hills are drawn lighter than flat land and
mountains in their own grey or snow (see biomes). Wherever the land steps up from one level to
the next, the edge of the higher cell gets a bevel: lit on the side facing the light, shaded on
the side facing away, so raised ground stands out even close up."""

import numpy as np
import numpy.typing as npt

from railroads.sim.tilemap import Relief, Terrain, TileMap

HILL_BRIGHTNESS = 1.15
# The bevel runs this far into the higher cell, in pixels of the ground picture (8 per cell).
BEVEL_PIXELS = 2.0
BEVEL_STRENGTH = 0.6


def relief_brightness(tile_map: TileMap) -> npt.NDArray[np.float64]:
    hills = (tile_map.relief == Relief.HILLS) & (tile_map.terrain == Terrain.LAND)
    return np.where(hills, HILL_BRIGHTNESS, 1.0)
