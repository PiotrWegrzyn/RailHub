"""The ground picture: every cell in its biome colour, hills lighter, and bevels where the land
steps up, laid out as the pixels of one texture."""

import math
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import Grid
from railroads.sim.tilemap import TileMap
from railroads.ui.biomes import biome_colors
from railroads.ui.relief import BEVEL_PIXELS, BEVEL_STRENGTH, relief_brightness

Pixels = npt.NDArray[np.uint8]

# The renderer samples textures with linear filtering; drawing each cell as a block of many
# identical pixels keeps cell edges sharp when zoomed in.
PIXELS_PER_CELL = 8
ROWS_PER_PASS = 256
# How much grid lines baked into the ground darken it, close to the look of the drawn lines.
BAKED_LINE_SHADE = 0.3
# Edge distances are stored in sixteenths of a pixel.
EDGE_STEPS_PER_PIXEL = 16


def cell_colors(tile_map: TileMap) -> Pixels:
    """Colour of every cell: its biome colour, lighter on hills."""
    base = biome_colors(tile_map)
    colors = base.copy()
    lit = base[..., :3] * relief_brightness(tile_map)[..., None]
    colors[..., :3] = np.clip(lit, 0, 255).astype(np.uint8)
    return colors


@dataclass(frozen=True)
class GroundLayout:
    """For every pixel of the ground picture: which cell it shows, how far it is from that cell's
    nearest edge, and which neighbour lies across that edge. Worked out once per map, so
    repainting the picture in new colours or light is quick."""

    cells: npt.NDArray[np.int32]
    across: npt.NDArray[np.int32]
    edge_direction: Pixels
    edge_distance: Pixels
    normals: npt.NDArray[np.float64]
    size: tuple[float, float]

    @classmethod
    def of(cls, grid: Grid) -> Self:
        width, height = grid.size()
        width_px = math.ceil(width * PIXELS_PER_CELL)
        height_px = math.ceil(height * PIXELS_PER_CELL)
        shape = (height_px, width_px)
        cells = np.empty(shape, dtype=np.int32)
        across = np.empty(shape, dtype=np.int32)
        edge_direction = np.empty(shape, dtype=np.uint8)
        edge_distance = np.empty(shape, dtype=np.uint8)
        normals = grid.edge_normals()
        xs = (np.arange(width_px) + 0.5) / PIXELS_PER_CELL
        for top in range(0, height_px, ROWS_PER_PASS):
            band = slice(top, min(top + ROWS_PER_PASS, height_px))
            ys = (np.arange(band.start, band.stop) + 0.5) / PIXELS_PER_CELL
            px, py = np.meshgrid(xs, ys)
            rows, cols = grid.cell_at(px, py)
            cx, cy = grid.center(rows, cols)
            reach = (px - cx)[..., None] * normals[:, 0] + (py - cy)[..., None] * normals[:, 1]
            nearest = reach.argmax(axis=-1)
            to_edge = (0.5 - reach.max(axis=-1)) * PIXELS_PER_CELL * EDGE_STEPS_PER_PIXEL
            other_rows, other_cols = grid.cell_at(
                cx + normals[nearest, 0], cy + normals[nearest, 1]
            )
            cells[band] = np.where(grid.contains(rows, cols), rows * grid.cols + cols, -1)
            across[band] = np.where(
                grid.contains(other_rows, other_cols), other_rows * grid.cols + other_cols, -1
            )
            edge_direction[band] = nearest
            edge_distance[band] = np.clip(np.rint(to_edge), 0, 255)
        size = (width_px / PIXELS_PER_CELL, height_px / PIXELS_PER_CELL)
        return cls(cells, across, edge_direction, edge_distance, normals, size)

    def paint(
        self,
        colors: Pixels,
        relief: npt.NDArray[np.uint8] | None = None,
        light_from: tuple[float, float] = (-1.0, -1.0),
        with_lines: bool = False,
    ) -> Pixels:
        """The picture's RGBA pixels for these cell colours; pixels off the map stay clear.
        With `relief`, raised edges get bevels lit from `light_from` (a direction on the map)."""
        palette = np.vstack([colors.reshape(-1, 4), np.zeros((1, 4), dtype=np.uint8)])
        rgba = palette[self.cells]
        if relief is not None:
            self._bevel(rgba, relief.ravel(), light_from)
        if with_lines:
            on_edge = np.clip(1 - self.edge_distance / EDGE_STEPS_PER_PIXEL, 0, 1)
            keep = 1 - BAKED_LINE_SHADE * on_edge[..., None]
            rgba[..., :3] = (rgba[..., :3] * keep).astype(np.uint8)
        return rgba

    def _bevel(
        self, rgba: Pixels, levels: npt.NDArray[np.uint8], light: tuple[float, float]
    ) -> None:
        width = BEVEL_PIXELS * EDGE_STEPS_PER_PIXEL
        near = (self.edge_distance < width) & (self.cells >= 0) & (self.across >= 0)
        cells, across = self.cells[near], self.across[near]
        raised = levels[cells] > levels[across]
        if not raised.any():
            return
        light_x, light_y = np.array(light) / math.hypot(*light)
        normals = self.normals[self.edge_direction[near][raised]]
        facing = normals[:, 0] * light_x + normals[:, 1] * light_y
        fade = 1 - self.edge_distance[near][raised] / width
        shade = 1 + BEVEL_STRENGTH * facing * fade
        pixels = rgba[near]
        lifted = pixels[raised]
        lifted[:, :3] = np.clip(lifted[:, :3] * shade[:, None], 0, 255).astype(np.uint8)
        pixels[raised] = lifted
        rgba[near] = pixels
