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
from railroads.sim.grids import BoolGrid, FloatArray
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
    """Which land cells a minor river flows through: those the reaches draining at least
    `least_km2` pass, but no water cell, and no cell where a big reach only follows a river that
    is drawn already."""

    lines: StreamLines
    least_km2: float = MIN_STREAM_KM2

    def __call__(self, area: MapArea, layers: dict) -> npt.NDArray[np.uint8]:
        terrain = layers["terrain"]
        grid = area.grid
        small = np.zeros(grid.shape, dtype=bool)
        big = np.zeros(grid.shape, dtype=bool)
        for line, upland in zip(self.lines.lines(), self.lines.upland_km2, strict=True):
            if upland >= self.least_km2:
                _trace(area, line, big if upland >= BIG_STREAM_KM2 else small)
        big &= ~_near(terrain == Terrain.RIVER, area, BIG_STREAM_GAP)
        return ((small | big) & (terrain == Terrain.LAND)).astype(np.uint8)


def _trace(area: MapArea, line: FloatArray, into: BoolGrid) -> None:
    x, y = area.to_plane(line[:, 0], line[:, 1])
    points = densify(np.column_stack([x, y]), LINE_STEP)
    rows, cols = area.grid.cell_at(points[:, 0], points[:, 1])
    on_map = area.grid.contains(rows, cols)
    into[rows[on_map], cols[on_map]] = True


def _near(cells: BoolGrid, area: MapArea, steps: int) -> BoolGrid:
    rows, cols = area.grid.neighbour_index_grids()
    near = cells.copy()
    for _ in range(steps):
        near = near | near[rows, cols].any(axis=0)
    return near
