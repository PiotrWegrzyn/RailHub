"""Minor rivers on the pixel map (railroads.mapgen.streams). The map keeps each cell's links to
the cells beside it its stream runs to or comes from, six bits, one for each neighbour, and a
stream is drawn as thin lines from its cell's middle towards theirs, by the CPU ground and the
shader alike."""

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import HEX_ROW_STEP, FloatArray, HexGrid, IntArray
from railroads.sim.tilemap import TileMap

# From a hexagon's middle to each of its neighbours' in the plane, in the order of the bits of a
# cell's links; each is opposite the one three on.
STEPS = ((1.0, 0.0), (0.5, HEX_ROW_STEP), (-0.5, HEX_ROW_STEP), (-1.0, 0.0), (-0.5, -HEX_ROW_STEP),
         (0.5, -HEX_ROW_STEP))  # fmt: skip
# How far from its line a stream's water reaches, and its banks past that, in cells.
HALF_WIDTH = 0.035
BANK_WIDTH = 0.07
# Farther than any stream, for points with none near.
NONE_NEAR = 9.0
# Streams bend this much, in cells, so they don't run quite straight from middle to middle.
MEANDER = 0.06
STREAM_WATER = (56, 112, 168)
STREAM_BANK = 0.8


def stream_links(tile_map: TileMap) -> npt.NDArray[np.uint8]:
    """Each cell's links to the cells beside it that its stream runs to or comes from, a bit for
    each of STEPS, as the map was made with them (railroads.mapgen.streams); 0 where no stream
    flows, or on a grid that isn't hexagons."""
    if not isinstance(tile_map.grid, HexGrid):
        return np.zeros(tile_map.grid.shape, dtype=np.uint8)
    return tile_map.stream


def stream_distance(
    links: npt.NDArray[np.uint8], grid: HexGrid, x: FloatArray, y: FloatArray, r: IntArray,
    c: IntArray,
) -> FloatArray:  # fmt: skip
    """How far each plane point `x`, `y` in cell `r`, `c` lies from the stream through that cell,
    in cells, or NONE_NEAR where none flows through it."""
    mask = links[r, c].astype(np.int64)
    distance = np.full(x.shape, NONE_NEAR)
    flowing = mask > 0
    if not flowing.any():
        return distance
    px, py = x[flowing], y[flowing]
    cx, cy = grid.center(r[flowing], c[flowing])
    px, py = px + MEANDER * np.sin(py * 7.3 + px * 2.1), py + MEANDER * np.sin(px * 6.1 - py * 1.7)
    nearest = np.full(px.shape, NONE_NEAR)
    for step, (dx, dy) in enumerate(STEPS):
        has = (mask[flowing] >> step) & 1 == 1
        along = np.clip(((px - cx) * dx + (py - cy) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        apart = np.hypot(px - cx - along * dx, py - cy - along * dy)
        nearest = np.where(has, np.minimum(nearest, apart), nearest)
    distance[flowing] = nearest
    return distance
