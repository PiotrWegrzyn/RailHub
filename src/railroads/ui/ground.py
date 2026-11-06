"""The ground picture: every cell in its biome colour, lightened by its height level and shaded
by its slope, laid out as the pixels of one texture."""

import math
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import Grid
from railroads.sim.tilemap import Terrain, TileMap
from railroads.ui.biomes import biome_colors
from railroads.ui.relief import LEVEL_BRIGHTNESS, height_levels, hill_shade

Pixels = npt.NDArray[np.uint8]

# The renderer samples textures with linear filtering; drawing each cell as a block of many
# identical pixels keeps cell edges sharp when zoomed in.
PIXELS_PER_CELL = 8
ROWS_PER_PASS = 256
# How much grid lines baked into the ground darken it, close to the look of the drawn lines.
BAKED_LINE_SHADE = 0.3


def cell_colors(tile_map: TileMap, light_from: tuple[float, float]) -> Pixels:
    """Colour of every cell: water as it is, land lightened by height and shaded by slope."""
    base = biome_colors(tile_map)
    land = tile_map.terrain == Terrain.LAND
    relief = LEVEL_BRIGHTNESS[height_levels(tile_map.height_m)] * hill_shade(tile_map, light_from)
    brightness = np.where(land, relief, 1.0)
    colors = base.copy()
    colors[..., :3] = np.clip(base[..., :3] * brightness[..., None], 0, 255).astype(np.uint8)
    return colors


@dataclass(frozen=True)
class GroundLayout:
    """Which cell every pixel of the ground picture shows, and how much a grid line darkens it.
    Worked out once per map, so repainting the picture in new colours is quick."""

    cells: npt.NDArray[np.int32]
    line_keep: Pixels
    size: tuple[float, float]

    @classmethod
    def of(cls, grid: Grid) -> Self:
        width, height = grid.size()
        width_px = math.ceil(width * PIXELS_PER_CELL)
        height_px = math.ceil(height * PIXELS_PER_CELL)
        cells = np.empty((height_px, width_px), dtype=np.int32)
        line_keep = np.empty((height_px, width_px), dtype=np.uint8)
        xs = (np.arange(width_px) + 0.5) / PIXELS_PER_CELL
        for top in range(0, height_px, ROWS_PER_PASS):
            band = slice(top, min(top + ROWS_PER_PASS, height_px))
            ys = (np.arange(band.start, band.stop) + 0.5) / PIXELS_PER_CELL
            px, py = np.meshgrid(xs, ys)
            rows, cols = grid.cell_at(px, py)
            cells[band] = np.where(grid.contains(rows, cols), rows * grid.cols + cols, -1)
            on_edge = np.clip(1 - grid.edge_distance(px, py) * PIXELS_PER_CELL, 0, 1)
            line_keep[band] = np.rint(255 * (1 - BAKED_LINE_SHADE * on_edge)).astype(np.uint8)
        return cls(cells, line_keep, (width_px / PIXELS_PER_CELL, height_px / PIXELS_PER_CELL))

    def paint(self, colors: Pixels, with_lines: bool = False) -> Pixels:
        """The picture's RGBA pixels for these cell colours; pixels off the map stay clear."""
        palette = np.vstack([colors.reshape(-1, 4), np.zeros((1, 4), dtype=np.uint8)])
        rgba = palette[self.cells]
        if with_lines:
            darker = rgba[..., :3].astype(np.uint16) * self.line_keep[..., None] // 255
            rgba[..., :3] = darker.astype(np.uint8)
        return rgba
