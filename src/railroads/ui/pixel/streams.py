"""Minor rivers on the pixel map (railroads.mapgen.streams). Each cell's stream runs on to the
lowest cell beside it that has a stream or water, so they join up and flow down into the rivers,
lakes and sea, and is drawn as a thin line from its cell's middle to that cell's: the links of
every cell are kept as six bits, one for each neighbour, for the CPU ground and the shader alike."""

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import HEX_ROW_STEP, FloatArray, HexGrid, IntArray
from railroads.sim.tilemap import Terrain, TileMap

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
    each of STEPS; 0 where no stream flows, or on a grid that isn't hexagons."""
    grid = tile_map.grid
    links = np.zeros(grid.shape, dtype=np.uint8)
    if not isinstance(grid, HexGrid):
        return links
    rows, cols = np.nonzero(tile_map.stream)
    if not len(rows):
        return links
    x, y = grid.center(rows, cols)
    wet = tile_map.terrain != Terrain.LAND
    flowing = tile_map.stream.astype(bool) | wet
    # Lower is better: water first, then the lowest ground, then the first cell, so no stream
    # runs round in a ring.
    height = tile_map.height_m.astype(np.int64) + 2**15
    rank = np.where(wet, np.int64(0), np.int64(2**40)) + height * np.int64(2**20)
    rank = rank + np.arange(rank.size, dtype=np.int64).reshape(rank.shape)
    best = np.full(len(rows), np.iinfo(np.int64).max, dtype=np.int64)
    towards = np.full(len(rows), -1, dtype=np.int64)
    to_rows, to_cols = np.zeros(len(rows), dtype=np.int64), np.zeros(len(rows), dtype=np.int64)
    for step, (dx, dy) in enumerate(STEPS):
        r, c = grid.cell_at(x + dx, y + dy)
        on = grid.contains(r, c)
        r, c = np.where(on, r, 0), np.where(on, c, 0)
        better = on & flowing[r, c] & (rank[r, c] < best)
        best = np.where(better, rank[r, c], best)
        towards = np.where(better, step, towards)
        to_rows, to_cols = np.where(better, r, to_rows), np.where(better, c, to_cols)
    linked = towards >= 0
    np.bitwise_or.at(
        links, (rows[linked], cols[linked]), (np.int64(1) << towards[linked]).astype(np.uint8)
    )
    back = linked & ~wet[to_rows, to_cols]
    opposite = ((towards[back] + 3) % 6).astype(np.uint8)
    np.bitwise_or.at(links, (to_rows[back], to_cols[back]), (1 << opposite).astype(np.uint8))
    return links


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
