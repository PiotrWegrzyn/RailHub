"""The ground picture: every cell in its biome colour, hills lighter, and bevels where the land
steps up, laid out as the pixels of one texture.

A cell is PIXELS_PER_CELL pixels wide, so the layout of the picture repeats every cell across: it
is worked out for one cell-wide strip of pixel columns and copied east, each copy one column of
cells further on. The picture is painted again whenever cells change colour, so what depends only
on the map is worked out once: which pixels the bevels light or shade, and which pixels baked grid
lines darken. Painting then gathers each pixel's colour as one 32-bit number, shades only those
pixels, and starts the picture with grid lines from a copy of the one without."""

import math
from dataclasses import dataclass
from functools import cached_property
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
# Most cells whose colour changed for the kept picture to be touched up where they lie instead of
# being painted again whole.
TOUCH_UP_CELLS = 2000


def cell_colors(tile_map: TileMap) -> Pixels:
    """Colour of every cell: its biome colour, lighter on hills."""
    base = biome_colors(tile_map)
    colors = base.copy()
    lit = base[..., :3] * relief_brightness(tile_map)[..., None]
    colors[..., :3] = np.clip(lit, 0, 255).astype(np.uint8)
    return colors


def _line_shades() -> Pixels:
    """Every channel value as a baked grid line darkens it: a row of all 256 values for each
    distance from the edge under one pixel, the rows one after another."""
    distance = np.arange(EDGE_STEPS_PER_PIXEL, dtype=np.uint8)
    on_edge = np.clip(1 - distance / EDGE_STEPS_PER_PIXEL, 0, 1)
    keep = 1 - BAKED_LINE_SHADE * on_edge[:, None]
    return (np.arange(256, dtype=np.uint8) * keep).astype(np.uint8).ravel()


LINE_SHADES = _line_shades()


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
        """Works the layout out a band of pixel rows at a time: for one cell-wide strip of pixel
        columns, repeated east, where the layout repeats every cell, and for every pixel where it
        doesn't."""
        width, height = grid.size()
        width_px = math.ceil(width * PIXELS_PER_CELL)
        height_px = math.ceil(height * PIXELS_PER_CELL)
        shape = (height_px, width_px)
        cells = np.empty(shape, dtype=np.int32)
        across = np.empty(shape, dtype=np.int32)
        edge_direction = np.empty(shape, dtype=np.uint8)
        edge_distance = np.empty(shape, dtype=np.uint8)
        normals = grid.edge_normals()
        strip = PIXELS_PER_CELL if _repeats_every_cell(grid) else width_px
        xs = (np.arange(strip) + 0.5) / PIXELS_PER_CELL
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
            cells[band] = _spread_east(grid, rows, cols, width_px)
            across[band] = _spread_east(grid, other_rows, other_cols, width_px)
            edge_direction[band] = _repeat_east(nearest.astype(np.uint8), width_px)
            distance = np.clip(np.rint(to_edge), 0, 255).astype(np.uint8)
            edge_distance[band] = _repeat_east(distance, width_px)
        size = (width_px / PIXELS_PER_CELL, height_px / PIXELS_PER_CELL)
        return cls(cells, across, edge_direction, edge_distance, normals, size)


def _repeats_every_cell(grid: Grid) -> bool:
    """Whether the layout repeats every PIXELS_PER_CELL pixel columns, one column of cells further
    east each time. It does when cell columns are one unit apart and a cell is a power of two
    pixels wide, two or more: then pixel centres a cell apart are exactly one unit apart, with
    nothing lost to rounding, and none lies on the edge between two cells of a row, where rounding
    could tip it either way."""
    column_step = grid.center(0, 1)[0] - grid.center(0, 0)[0]
    power_of_two = PIXELS_PER_CELL & (PIXELS_PER_CELL - 1) == 0
    return column_step == 1 and PIXELS_PER_CELL >= 2 and power_of_two


def _spread_east(
    grid: Grid, rows: npt.NDArray[np.int64], cols: npt.NDArray[np.int64], width_px: int
) -> npt.NDArray[np.int32]:
    """The flat index of the cell under every pixel of a band, -1 off the map, from the cells
    under its first strip of pixel columns: each further strip shows the cells one column east."""
    repeats = math.ceil(width_px / cols.shape[1])
    rows = rows.astype(np.int32)[:, None, :]
    cols = cols.astype(np.int32)[:, None, :] + np.arange(repeats, dtype=np.int32)[:, None]
    index = np.where(grid.contains(rows, cols), rows * grid.cols + cols, -1)
    return index.reshape(len(index), -1)[:, :width_px]


def _repeat_east(strip: Pixels, width_px: int) -> Pixels:
    """A band of the layout from its first strip of pixel columns, repeated east."""
    repeats = math.ceil(width_px / strip.shape[1])
    return np.tile(strip, (1, repeats))[:, :width_px]


@dataclass(frozen=True)
class _Painted:
    """A ground picture without grid lines, and the colours and light it was painted in."""

    colors: Pixels
    light_from: tuple[float, float]
    rgba: Pixels

    def shows(self, colors: Pixels, light_from: tuple[float, float]) -> bool:
        return light_from == self.light_from and bool(np.array_equal(colors, self.colors))


class GroundPictures:
    """One map's ground picture in its cells' colours, with or without baked grid lines.

    The relief never changes, so the pixels on raised edges, which the bevels light or shade, are
    found once; the pixels near an edge, which grid lines darken, are found the first time lines
    are wanted. The last picture without lines is kept with the colours and light it was painted
    in: while those stay the same it is handed out again, and the picture with lines starts from a
    copy of it. Given the grid, a picture whose colours changed in a few cells is touched up in
    place where those cells lie."""

    def __init__(
        self, layout: GroundLayout, relief: npt.NDArray[np.uint8], grid: Grid | None = None
    ) -> None:
        self.layout = layout
        self._grid = grid
        width = BEVEL_PIXELS * EDGE_STEPS_PER_PIXEL
        near = (layout.edge_distance < width) & (layout.cells >= 0) & (layout.across >= 0)
        at = np.flatnonzero(near)
        levels = relief.ravel()
        self._raised = at[levels[layout.cells.ravel()[at]] > levels[layout.across.ravel()[at]]]
        self._raised_direction = layout.edge_direction.ravel()[self._raised]
        self._raised_fade = 1 - layout.edge_distance.ravel()[self._raised] / width
        self._plain: _Painted | None = None

    def picture(
        self, colors: Pixels, light_from: tuple[float, float], with_lines: bool = False
    ) -> Pixels:
        """The picture's RGBA pixels for these cell colours, raised edges lit from `light_from`
        (a direction on the map); pixels off the map stay clear. A picture without lines is the
        one kept, so it must not be changed."""
        plain = self._plain
        if plain is None or not plain.shows(colors, light_from):
            changed = self._changed_cells(plain, colors, light_from)
            if changed is None:
                self._plain = _Painted(colors.copy(), light_from, self._paint(colors, light_from))
            else:
                self._touch_up(plain.rgba, changed, colors, light_from)
                plain.colors[...] = colors
        if not with_lines:
            return self._plain.rgba
        lined = self._plain.rgba.copy()
        self._add_lines(lined)
        return lined

    def _paint(self, colors: Pixels, light_from: tuple[float, float]) -> Pixels:
        """Every pixel in its cell's colour, gathered as one 32-bit number, and the raised edges
        shaded."""
        palette = np.vstack([colors.reshape(-1, 4), np.zeros((1, 4), dtype=np.uint8)])
        packed = palette.view(np.uint32)[:, 0][self.layout.cells]
        rgba = packed.view(np.uint8).reshape(*packed.shape, 4)
        flat = rgba.reshape(-1, 4)
        lifted = flat[self._raised, :3] * self._shade(light_from)[:, None]
        flat[self._raised, :3] = np.clip(lifted, 0, 255).astype(np.uint8)
        return rgba

    def _shade(
        self, light_from: tuple[float, float], which: npt.NDArray[np.intp] | None = None
    ) -> npt.NDArray[np.float64]:
        """How much brighter (over 1) or darker (under 1) each pixel on a raised edge is, or
        those of them `which` picks."""
        light_x, light_y = np.array(light_from) / math.hypot(*light_from)
        normals = self.layout.normals
        facing = normals[:, 0] * light_x + normals[:, 1] * light_y
        direction, fade = self._raised_direction, self._raised_fade
        if which is not None:
            direction, fade = direction[which], fade[which]
        return 1 + BEVEL_STRENGTH * facing[direction] * fade

    def _changed_cells(
        self, plain: _Painted | None, colors: Pixels, light_from: tuple[float, float]
    ) -> npt.NDArray[np.intp] | None:
        """The cells, by their index in the flattened grid, whose colour changed since the kept
        picture was painted, if it can be touched up there; otherwise None."""
        if self._grid is None or plain is None or plain.light_from != light_from:
            return None
        if plain.colors.shape != colors.shape:
            return None
        changed = np.flatnonzero((plain.colors != colors).any(axis=-1).ravel())
        return changed if len(changed) <= TOUCH_UP_CELLS else None

    def _touch_up(
        self,
        rgba: Pixels,
        changed: npt.NDArray[np.intp],
        colors: Pixels,
        light_from: tuple[float, float],
    ) -> None:
        """Paints again the pixels of the changed cells, looking only in a box round each."""
        assert self._grid is not None
        cells = self.layout.cells
        height, width = cells.shape
        flat_colors = colors.reshape(-1, 4)
        rows, cols = np.divmod(changed, colors.shape[1])
        centers_x, centers_y = self._grid.center(rows, cols)
        flat = rgba.reshape(-1, 4)
        shaded: list[npt.NDArray[np.intp]] = []
        for cell, cx, cy in zip(
            changed.tolist(), np.ravel(centers_x), np.ravel(centers_y), strict=True
        ):
            left = max(0, math.floor((cx - 1) * PIXELS_PER_CELL))
            right = min(width, math.ceil((cx + 1) * PIXELS_PER_CELL))
            top = max(0, math.floor((cy - 1) * PIXELS_PER_CELL))
            bottom = min(height, math.ceil((cy + 1) * PIXELS_PER_CELL))
            if left >= right or top >= bottom:
                continue
            window = rgba[top:bottom, left:right]
            window[cells[top:bottom, left:right] == cell] = flat_colors[cell]
            starts = np.arange(top, bottom) * width
            low = np.searchsorted(self._raised, starts + left)
            high = np.searchsorted(self._raised, starts + right)
            near = np.concatenate([np.arange(a, b) for a, b in zip(low, high, strict=True)])
            if len(near):
                shaded.append(near[cells.ravel()[self._raised[near]] == cell])
        if not shaded:
            return
        which = np.concatenate(shaded)
        at = self._raised[which]
        base = flat_colors[cells.ravel()[at], :3]
        lifted = base * self._shade(light_from, which)[:, None]
        flat[at, :3] = np.clip(lifted, 0, 255).astype(np.uint8)

    def _add_lines(self, rgba: Pixels) -> None:
        """Darkens the pixels near an edge as baked grid lines do, each channel looked up in
        LINE_SHADES."""
        at, shade_rows = self._line_pixels
        packed = rgba.reshape(-1).view(np.uint32)
        near = packed[at].view(np.uint8).reshape(-1, 4)
        for channel in range(3):
            near[:, channel] = LINE_SHADES[shade_rows + near[:, channel]]
        packed[at] = near.reshape(-1).view(np.uint32)

    @cached_property
    def _line_pixels(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint16]]:
        """The pixels less than a pixel from an edge, which grid lines darken, and where the row
        of LINE_SHADES for each one's distance starts."""
        distance = self.layout.edge_distance.ravel()
        at = np.flatnonzero(distance < EDGE_STEPS_PER_PIXEL).astype(np.int32)
        return at, distance[at].astype(np.uint16) * 256
