"""Minor rivers: the HydroRIVERS reaches too small for Natural Earth, which flow through land
cells without making them water. A cell with one stays land that anything may be built on, but
counts as fresh water by a river for what grows (railroads.sim.ladders.Surroundings)."""

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.mapgen.area import LINE_STEP, MapArea, densify
from railroads.sim.grids import BoolGrid, FloatArray, Grid
from railroads.sim.tilemap import Terrain

HYDRORIVERS_SOURCE = "Minor rivers: HydroRIVERS v1.0 (Lehner and Grill 2013), www.hydrosheds.org"
# A reach becomes a minor river when it drains at least this much land.
MIN_STREAM_KM2 = 1500.0
# Reaches draining this much are big rivers, which Natural Earth may have drawn already: where one
# runs this close to a river cell, in cells, it is taken to be that river.
BIG_STREAM_KM2 = 5000.0
BIG_STREAM_GAP = 2
# The points of a square's reaches are kept in steps of this many degrees from its middle.
POINT_STEP_DEG = 2e-4


@dataclass(frozen=True)
class StreamLines:
    """Reaches of minor rivers: each a line of lon, lat points, how far `ends` says, and how much
    land it drains."""

    points: FloatArray
    ends: npt.NDArray[np.int64]
    upland_km2: FloatArray

    @classmethod
    def empty(cls) -> Self:
        return cls(np.zeros((0, 2)), np.zeros(0, dtype=np.int64), np.zeros(0))

    @classmethod
    def joined(cls, many: Sequence[StreamLines]) -> Self:
        if not many:
            return cls.empty()
        offsets = np.cumsum([0] + [len(lines.points) for lines in many[:-1]])
        return cls(
            np.concatenate([lines.points for lines in many]),
            np.concatenate(
                [lines.ends + offset for lines, offset in zip(many, offsets, strict=True)]
            ),
            np.concatenate([lines.upland_km2 for lines in many]),
        )

    def __len__(self) -> int:
        return len(self.ends)

    def lines(self) -> list[FloatArray]:
        return np.split(self.points, self.ends[:-1]) if len(self.ends) else []

    def draining(self, least_km2: float) -> Self:
        """Only the reaches draining at least this much land."""
        keep = self.upland_km2 >= least_km2
        lines = [line for line, kept in zip(self.lines(), keep, strict=True) if kept]
        return type(self)(
            np.concatenate(lines) if lines else np.zeros((0, 2)),
            np.cumsum([len(line) for line in lines], dtype=np.int64),
            self.upland_km2[keep],
        )


def write_streams(lines: StreamLines, middle_lon: float, middle_lat: float) -> bytes:
    """A square's reaches packed small: points as steps from the square's middle, each reach's
    first point whole and the rest as differences from the point before."""
    steps = np.rint((lines.points - (middle_lon, middle_lat)) / POINT_STEP_DEG).astype(np.int32)
    starts = np.concatenate([[0], lines.ends[:-1]]).astype(np.int64)
    moves = np.diff(steps, axis=0, prepend=steps[:1])
    moves[starts] = steps[starts]
    out = io.BytesIO()
    np.savez_compressed(
        out,
        middle=np.array([middle_lon, middle_lat]),
        moves=moves.astype(np.int16 if np.abs(moves).max(initial=0) < 2**15 else np.int32),
        counts=np.diff(lines.ends, prepend=0).astype(np.uint16),
        upland_km2=lines.upland_km2.astype(np.float32),
    )
    return out.getvalue()


def read_streams(data: bytes) -> StreamLines:
    with np.load(io.BytesIO(data)) as arrays:
        middle = arrays["middle"]
        moves = arrays["moves"].astype(np.int64)
        ends = np.cumsum(arrays["counts"].astype(np.int64))
        upland = arrays["upland_km2"].astype(np.float64)
    starts = np.concatenate([[0], ends[:-1]])
    reach = np.repeat(np.arange(len(ends)), np.diff(ends, prepend=0))
    walked = np.cumsum(moves, axis=0)
    steps = walked - (walked[starts] - moves[starts])[reach]
    return StreamLines(steps * POINT_STEP_DEG + middle, ends, upland)


@dataclass(frozen=True)
class StreamsFromLines:
    """Where minor rivers flow, as each land cell's links to the cells beside it that its stream
    runs to or comes from, a bit for each of the grid's edge_normals: along the reaches draining
    at least `least_km2`, through land only, leaving out where a big reach only follows a river
    drawn already, and pieces that mostly run along a river, which are that river's wiggles."""

    lines: StreamLines
    least_km2: float = MIN_STREAM_KM2

    def __call__(self, area: MapArea, layers: dict) -> npt.NDArray[np.uint8]:
        terrain = layers["terrain"]
        grid = area.grid
        small = np.zeros(grid.shape, dtype=np.uint8)
        big = np.zeros(grid.shape, dtype=np.uint8)
        for line, upland in zip(self.lines.lines(), self.lines.upland_km2, strict=True):
            if upland >= self.least_km2:
                _trace(area, line, big if upland >= BIG_STREAM_KM2 else small)
        river = terrain == Terrain.RIVER
        big[_near(river, area, BIG_STREAM_GAP)] = 0
        links = small | big
        links[terrain != Terrain.LAND] = 0
        links = _tied(links, terrain, grid)
        links[_along_rivers(links > 0, river, area)] = 0
        return _tied(links, terrain, grid)


def _trace(area: MapArea, line: FloatArray, links: npt.NDArray[np.uint8]) -> None:
    """Sets the links between each cell the line passes and the next one."""
    grid = area.grid
    x, y = area.to_plane(line[:, 0], line[:, 1])
    points = densify(np.column_stack([x, y]), LINE_STEP)
    rows, cols = grid.cell_at(points[:, 0], points[:, 1])
    moved = np.flatnonzero((np.diff(rows) != 0) | (np.diff(cols) != 0))
    if not len(moved):
        return
    rows_from, cols_from, rows_to, cols_to = (
        rows[moved],
        cols[moved],
        rows[moved + 1],
        cols[moved + 1],
    )
    normals = grid.edge_normals()
    from_x, from_y = grid.center(rows_from, cols_from)
    to_x, to_y = grid.center(rows_to, cols_to)
    step = np.stack([np.asarray(to_x) - from_x, np.asarray(to_y) - from_y], axis=-1)
    towards = np.argmax(step @ normals.T, axis=1)
    back = (towards + len(normals) // 2) % len(normals)
    for r, c, bits in ((rows_from, cols_from, towards), (rows_to, cols_to, back)):
        on_map = grid.contains(r, c)
        np.bitwise_or.at(links, (r[on_map], c[on_map]), (1 << bits[on_map]).astype(np.uint8))


def _tied(
    links: npt.NDArray[np.uint8], terrain: npt.NDArray[np.uint8], grid: Grid
) -> npt.NDArray[np.uint8]:
    """The links, without those to a cell with neither a stream nor water."""
    rows, cols = grid.neighbour_index_grids()
    reaches = (links > 0) | (terrain != Terrain.LAND)
    tied = links.copy()
    for bit in range(len(rows)):
        beyond = reaches[rows[bit], cols[bit]] & (
            (rows[bit], cols[bit]) != np.indices(grid.shape)
        ).any(axis=0)
        tied &= np.where(beyond, 0xFF, 0xFF ^ (1 << bit)).astype(np.uint8)
    return tied


def _along_rivers(streams: BoolGrid, river: BoolGrid, area: MapArea) -> BoolGrid:
    """The cells of the pieces of streams that mostly lie beside a river cell."""
    grid = area.grid
    beside = streams & _near(river, area, 1)
    rows, cols = grid.neighbour_index_grids()
    label = np.where(streams, np.arange(streams.size).reshape(streams.shape), -1)
    while True:
        around = np.where(streams[rows, cols], label[rows, cols], -1).max(axis=0)
        grown = np.where(streams, np.maximum(label, around), -1)
        if np.array_equal(grown, label):
            break
        label = grown
    pieces = label[streams]
    cells = np.bincount(pieces, minlength=streams.size)
    by_river = np.bincount(label[beside], minlength=streams.size)
    along = np.zeros(streams.size, dtype=bool)
    along[cells > 0] = by_river[cells > 0] * 2 >= cells[cells > 0]
    return streams & along[np.where(streams, label, 0)]


def _near(cells: BoolGrid, area: MapArea, steps: int) -> BoolGrid:
    rows, cols = area.grid.neighbour_index_grids()
    near = cells.copy()
    for _ in range(steps):
        near = near | near[rows, cols].any(axis=0)
    return near
