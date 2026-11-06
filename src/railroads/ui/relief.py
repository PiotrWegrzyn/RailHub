"""How height shows on the map. Every cell sits on one of eleven height levels, and higher levels
are drawn lighter. On top, slopes are shaded as if lit from one side, which makes hills and
valleys stand out."""

import math

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import HEX_ROW_STEP
from railroads.sim.tilemap import TileMap

FloatGrid = npt.NDArray[np.float64]

# Level n is for heights under the n-th limit in metres; level 10 is 3000 m and up.
LEVEL_LIMITS_M = (50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000)
LEVEL_BRIGHTNESS = np.linspace(0.85, 1.25, len(LEVEL_LIMITS_M) + 1)
# Slopes are drawn this many times steeper than they are, so hills show at 2 km per cell.
RELIEF_EXAGGERATION = 6.0
SUN_ELEVATION_DEG = 45.0
SHADE_STRENGTH = 0.6


def height_levels(height_m: npt.NDArray[np.integer]) -> npt.NDArray[np.uint8]:
    return np.digitize(height_m, LEVEL_LIMITS_M).astype(np.uint8)


def hill_shade(tile_map: TileMap, light_from: tuple[float, float]) -> FloatGrid:
    """Brightness of each cell from the slope it is on: 1 on flat ground, more on slopes facing
    the light, less on slopes facing away. `light_from` points towards the light on the map,
    with x east and y south. The sea counts as flat ground at 0 m."""
    grid = tile_map.grid
    metres_per_unit = grid.km_per_cell * 1000
    height = np.maximum(tile_map.height_m, 0) * RELIEF_EXAGGERATION / metres_per_unit
    row_spacing = HEX_ROW_STEP if grid.kind == "hex" else 1.0
    slope_y, slope_x = np.gradient(height, row_spacing, 1.0)
    light_x, light_y = light_from
    length = math.hypot(light_x, light_y)
    sun = math.radians(SUN_ELEVATION_DEG)
    across = math.cos(sun) / length
    lit = (-slope_x * light_x * across - slope_y * light_y * across + math.sin(sun)) / np.sqrt(
        slope_x**2 + slope_y**2 + 1
    )
    return 1 + SHADE_STRENGTH * (lit / math.sin(sun) - 1)
