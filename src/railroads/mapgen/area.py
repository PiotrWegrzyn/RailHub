"""The part of the world being made into a map, and how shapes are measured against its cells."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.mapgen.sources import Shape
from railroads.sim.geo import Bounds, Coord
from railroads.sim.grids import BoolGrid, FloatArray, Grid

# Lines are walked in steps this small (in cells), so a traced river never skips a cell.
LINE_STEP = 0.1


@dataclass(frozen=True)
class CellSamples:
    """Points spread over every cell, to measure how much of a cell something covers.

    Points are stored cell by cell, `per_cell` at a time. They are also indexed by y, so a polygon
    can be scanned one row of points at a time.
    """

    x: FloatArray
    y: FloatArray
    per_cell: int
    shape: tuple[int, int]
    order: npt.NDArray[np.int64]
    row_y: FloatArray
    row_starts: npt.NDArray[np.int64]

    @classmethod
    def of(cls, grid: Grid) -> Self:
        cx, cy = grid.centers()
        offsets = grid.sample_offsets()
        x = (cx[..., None] + offsets[:, 0]).ravel()
        y = (cy[..., None] + offsets[:, 1]).ravel()
        order = np.argsort(y, kind="stable")
        row_y, starts = np.unique(y[order], return_index=True)
        return cls(x, y, len(offsets), grid.shape, order, row_y, np.append(starts, len(y)))

    def per_cell_mean(self, values: npt.NDArray[np.generic]) -> FloatArray:
        return values.reshape(*self.shape, self.per_cell).mean(axis=2)

    def inside(self, rings: list[FloatArray]) -> npt.NDArray[np.bool_]:
        """Which points lie inside the polygon these rings outline. Rings combine by the even-odd
        rule, so a ring inside another one is a hole."""
        x0 = np.concatenate([ring[:, 0] for ring in rings])
        y0 = np.concatenate([ring[:, 1] for ring in rings])
        x1 = np.concatenate([np.roll(ring[:, 0], -1) for ring in rings])
        y1 = np.concatenate([np.roll(ring[:, 1], -1) for ring in rings])
        result = np.zeros(len(self.x), dtype=bool)
        first = int(np.searchsorted(self.row_y, y0.min()))
        last = int(np.searchsorted(self.row_y, y0.max(), side="right"))
        for k in range(first, last):
            y = self.row_y[k]
            crossing = (y0 <= y) != (y1 <= y)
            if not crossing.any():
                continue
            cx0, cy0 = x0[crossing], y0[crossing]
            xs = np.sort(cx0 + (y - cy0) * (x1[crossing] - cx0) / (y1[crossing] - cy0))
            members = self.order[self.row_starts[k] : self.row_starts[k + 1]]
            result[members] = np.searchsorted(xs, self.x[members]) % 2 == 1
        return result


@dataclass(frozen=True)
class MapArea:
    """A lon/lat box with a grid laid over it, in the grid's cell units."""

    bounds: Bounds
    grid: Grid
    samples: CellSamples

    @classmethod
    def of(cls, bounds: Bounds, grid: Grid) -> Self:
        return cls(bounds, grid, CellSamples.of(grid))

    def to_plane(self, lon: Coord, lat: Coord) -> tuple[Coord, Coord]:
        x_km, y_km = self.bounds.to_km(lon, lat)
        return x_km / self.grid.km_per_cell, y_km / self.grid.km_per_cell

    def to_lonlat(self, x: Coord, y: Coord) -> tuple[Coord, Coord]:
        return self.bounds.to_lonlat(x * self.grid.km_per_cell, y * self.grid.km_per_cell)

    def center_lonlat(self) -> tuple[Coord, Coord]:
        return self.to_lonlat(*self.grid.centers())

    def sample_lonlat(self) -> tuple[Coord, Coord]:
        return self.to_lonlat(self.samples.x, self.samples.y)

    def cover(self, shapes: Iterable[Shape]) -> FloatArray:
        """How much of each cell, from 0 to 1, lies inside any of the polygons."""
        width, height = self.grid.size()
        inside = np.zeros(len(self.samples.x), dtype=bool)
        for shape in shapes:
            if shape.kind != "polygon":
                continue
            rings = [
                clip_ring(self._project(part), -1, -1, width + 1, height + 1)
                for part in shape.parts
            ]
            rings = [ring for ring in rings if len(ring) >= 3]
            if rings:
                inside |= self.samples.inside(rings)
        return self.samples.per_cell_mean(inside)

    def trace(self, shapes: Iterable[Shape]) -> BoolGrid:
        """Cells that any of the lines pass through."""
        mask = np.zeros(self.grid.shape, dtype=bool)
        for shape in shapes:
            if shape.kind != "line":
                continue
            for part in shape.parts:
                points = densify(self._project(part), LINE_STEP)
                rows, cols = self.grid.cell_at(points[:, 0], points[:, 1])
                on_map = self.grid.contains(rows, cols)
                mask[rows[on_map], cols[on_map]] = True
        return mask

    def length_km(self, shapes: Iterable[Shape]) -> float:
        """How far the lines run inside the map."""
        width, height = self.grid.size()
        total = 0.0
        for shape in shapes:
            if shape.kind != "line":
                continue
            for part in shape.parts:
                points = self._project(part)
                on_map = (points >= 0).all(axis=1) & (points <= (width, height)).all(axis=1)
                both_ends = on_map[:-1] & on_map[1:]
                steps = np.hypot(*np.diff(points, axis=0).T)
                total += float(steps[both_ends].sum())
        return total * self.grid.km_per_cell

    def _project(self, lonlat: FloatArray) -> FloatArray:
        x, y = self.to_plane(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])


def clip_ring(
    ring: FloatArray, x_min: float, y_min: float, x_max: float, y_max: float
) -> FloatArray:
    """Cut a polygon ring down to a rectangle (Sutherland-Hodgman), so huge shapes like an ocean
    shrink to the few edges that matter."""
    low, high = ring.min(axis=0), ring.max(axis=0)
    if high[0] < x_min or low[0] > x_max or high[1] < y_min or low[1] > y_max:
        return ring[:0]
    if low[0] >= x_min and high[0] <= x_max and low[1] >= y_min and high[1] <= y_max:
        return ring
    for axis, bound, keep_above in (
        (0, x_min, True),
        (0, x_max, False),
        (1, y_min, True),
        (1, y_max, False),
    ):
        if len(ring) == 0:
            break
        ring = _clip_half_plane(ring, axis, bound, keep_above)
    return ring


def _clip_half_plane(ring: FloatArray, axis: int, bound: float, keep_above: bool) -> FloatArray:
    previous = np.roll(ring, 1, axis=0)
    value, previous_value = ring[:, axis], previous[:, axis]
    inside = value >= bound if keep_above else value <= bound
    previous_inside = np.roll(inside, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (bound - previous_value) / (value - previous_value)
        crossing = previous + t[:, None] * (ring - previous)
    points = np.stack([crossing, ring], axis=1)
    keep = np.stack([inside != previous_inside, inside], axis=1)
    return points[keep]


def densify(line: FloatArray, step: float) -> FloatArray:
    """Points along a polyline, no further apart than `step`."""
    segments = np.diff(line, axis=0)
    counts = np.maximum(1, np.ceil(np.hypot(segments[:, 0], segments[:, 1]) / step).astype(np.intp))
    starts = np.repeat(line[:-1], counts, axis=0)
    steps = np.repeat(segments / counts[:, None], counts, axis=0)
    index = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.vstack([starts + steps * index[:, None], line[-1:]])
