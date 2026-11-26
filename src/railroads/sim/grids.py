"""Ways to divide a map into cells: squares, or hexagons with a point at the top.

Cells are addressed as (row, col) with row 0 along the north edge, and every property of a map is
a numpy array of shape (rows, cols). Positions on the map are measured in cell units: centres of
neighbouring cells are one unit apart, so units times km_per_cell gives kilometres.

The cells within some steps of a cell are also kept as flat indices once worked out (flat_within):
a grid never changes, and every month the simulation asks again for the cells around each station.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import ClassVar, Literal, Self

import numpy as np
import numpy.typing as npt

from railroads.sim.geo import Coord

type Cell = tuple[int, int]
type GridKind = Literal["square", "hex"]
type Index = int | npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolGrid = npt.NDArray[np.bool_]
# Cells as their flat indices into a map's layers, row by row.
FlatIndices = npt.NDArray[np.intp]
# How many kilometres across a cell of every map is.
KM_PER_CELL = 2.0

HEX_RADIUS = 1 / math.sqrt(3)
HEX_ROW_STEP = 1.5 * HEX_RADIUS
_HEX_CORNER_ANGLES = tuple(math.radians(60 * i - 30) for i in range(6))
# (col, row) steps to the six neighbours, for even rows and for odd rows (shifted half a cell east).
_HEX_STEPS = (
    ((1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)),
    ((1, 0), (1, -1), (0, -1), (-1, 0), (0, 1), (1, 1)),
)


@dataclass(frozen=True, slots=True)
class Grid(ABC):
    rows: int
    cols: int
    km_per_cell: float

    kind: ClassVar[GridKind]
    # How far out a cell's corner is, as a share of the way to the cell beyond that corner.
    corner_share: ClassVar[float]

    @classmethod
    @abstractmethod
    def covering(cls, width_km: float, height_km: float, km_per_cell: float) -> Self: ...

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def contains(self, row: Index, col: Index) -> bool | BoolGrid:
        return (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)

    def centers(self) -> tuple[FloatArray, FloatArray]:
        rows, cols = np.indices(self.shape)
        x, y = self.center(rows, cols)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    @abstractmethod
    def size(self) -> tuple[float, float]:
        """Width and height of the whole grid, in cell units."""

    @abstractmethod
    def center(self, row: Index, col: Index) -> tuple[Coord, Coord]: ...

    @abstractmethod
    def cell_at(self, x: Coord, y: Coord) -> tuple[Index, Index]:
        """The cell under each point; it can lie outside the grid."""

    def cell_on_map(self, x: Coord, y: Coord) -> tuple[Index, Index]:
        """The cell under each point, kept on the grid: a point beyond an edge takes the row or
        column of that edge."""
        rows, cols = self.cell_at(x, y)
        return np.clip(rows, 0, self.rows - 1), np.clip(cols, 0, self.cols - 1)

    def raster_cells(self, steps: int) -> tuple[Index, Index]:
        """The cell under the middle of each point of a raster `steps` points to a cell unit
        over the whole grid (cell_on_map)."""
        width, height = self.size()
        ys, xs = np.mgrid[0 : math.ceil(height * steps), 0 : math.ceil(width * steps)]
        return self.cell_on_map((xs + 0.5) / steps, (ys + 0.5) / steps)

    @abstractmethod
    def neighbours(self, row: int, col: int) -> list[Cell]:
        """Cells that share an edge with this one."""

    def within(self, cell: Cell, steps: int) -> dict[Cell, int]:
        """Every cell at most `steps` steps from this one, cell to neighbouring cell, and how
        many steps away it is."""
        reached = {cell: 0}
        edge = [cell]
        for step in range(1, steps + 1):
            nxt = []
            for here in edge:
                for near in self.neighbours(*here):
                    if near not in reached:
                        reached[near] = step
                        nxt.append(near)
            edge = nxt
        return reached

    @abstractmethod
    def corners(self, row: int, col: int) -> list[tuple[float, float]]: ...

    @abstractmethod
    def sample_offsets(self) -> FloatArray:
        """Points spread evenly inside a cell, relative to its centre, as an (n, 2) array."""

    @abstractmethod
    def seal_corner_gaps(self, mask: BoolGrid) -> BoolGrid:
        """Add cells to the mask wherever two of its cells touch only at a corner, so that what
        the mask outlines, like a river, cannot be crossed diagonally."""

    @abstractmethod
    def edge_distance(self, x: Coord, y: Coord) -> Coord:
        """How far each point is from the nearest edge of the cell it lies in, in cell units."""

    @abstractmethod
    def edge_normals(self) -> FloatArray:
        """One unit vector per edge, pointing from a cell's centre straight across that edge to
        the neighbour's centre, as a (directions, 2) array."""

    def neighbour_index_grids(self) -> tuple[IntArray, IntArray]:
        """Row and column of every cell's neighbour in each direction of `edge_normals`, as
        arrays of shape (directions, rows, cols). Off the map, a cell is its own neighbour."""
        x, y = self.centers()
        rows, cols = np.indices(self.shape)
        neighbour_rows, neighbour_cols = [], []
        for dx, dy in self.edge_normals():
            row, col = self.cell_at(x + dx, y + dy)
            inside = self.contains(row, col)
            neighbour_rows.append(np.where(inside, row, rows))
            neighbour_cols.append(np.where(inside, col, cols))
        return np.stack(neighbour_rows), np.stack(neighbour_cols)

    def track_steps(self) -> FloatArray:
        """From a cell's centre to the centre of the cell that track leaving through each track
        point runs into, as an (n, 2) array. The points go round the cell, alternating the middle
        of an edge, which leads to the neighbour across it, and a corner, which leads along the
        edge between the two cells beside it to the cell beyond."""
        return _track_steps(self).copy()

    def _find_track_steps(self) -> FloatArray:
        normals = self.edge_normals()
        beyond_corners = normals + np.roll(normals, -1, axis=0)
        return np.stack([normals, beyond_corners], axis=1).reshape(-1, 2)

    def track_points(self) -> FloatArray:
        """Where track can cross into or out of a cell, relative to its centre, in the order of
        `track_steps`: twelve on a hexagon, eight on a square."""
        points = self.track_steps()
        points[0::2] *= 0.5
        points[1::2] *= self.corner_share
        return points

    def track_ends(self) -> tuple[int, ...]:
        """The track points track may leave a cell by."""
        return _track_ends(self)

    def _find_track_ends(self) -> tuple[int, ...]:
        """Every track point."""
        return tuple(range(len(self.track_steps())))

    def track_exit(self, cell: Cell, point: int) -> tuple[Cell, int]:
        """The cell that track leaving through this point runs into, and the point it enters by;
        worked out once for each cell and point, as the railway asks thousands of times each
        time it changes."""
        exits = _EXITS.get(self)
        if exits is None:
            exits = _EXITS[self] = {}
        key = (cell, point)
        found = exits.get(key)
        if found is None:
            steps = _track_steps(self)
            x, y = self.center(*cell)
            row, col = self.cell_at(x + steps[point, 0], y + steps[point, 1])
            found = exits[key] = ((int(row), int(col)), (point + len(steps) // 2) % len(steps))
        return found

    def track_flanks(self, cell: Cell, point: int) -> list[Cell]:
        """The two cells that track through a corner runs between; none at the middle of an
        edge."""
        if point % 2 == 0:
            return []
        count = len(_track_steps(self))
        return [self.track_exit(cell, side % count)[0] for side in (point - 1, point + 1)]

    def track_turn(self, start: int, end: int) -> float:
        """How many degrees track bends going into a cell at one track point and out at
        another."""
        return _track_turn(self, start, end)

    def _find_track_turn(self, start: int, end: int) -> float:
        points = self.track_points()
        heading_in = -points[start] / np.linalg.norm(points[start])
        heading_out = points[end] / np.linalg.norm(points[end])
        return math.degrees(math.acos(float(np.clip(heading_in @ heading_out, -1.0, 1.0))))


@dataclass(frozen=True, slots=True)
class SquareGrid(Grid):
    kind: ClassVar[GridKind] = "square"
    corner_share: ClassVar[float] = 1 / 2

    @classmethod
    def covering(cls, width_km: float, height_km: float, km_per_cell: float) -> Self:
        cols = max(1, round(width_km / km_per_cell))
        rows = max(1, round(height_km / km_per_cell))
        return cls(rows=rows, cols=cols, km_per_cell=km_per_cell)

    def size(self) -> tuple[float, float]:
        return float(self.cols), float(self.rows)

    def center(self, row: Index, col: Index) -> tuple[Coord, Coord]:
        return col + 0.5, row + 0.5

    def cell_at(self, x: Coord, y: Coord) -> tuple[Index, Index]:
        return np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)

    def neighbours(self, row: int, col: int) -> list[Cell]:
        candidates = [(row, col + 1), (row - 1, col), (row, col - 1), (row + 1, col)]
        return [cell for cell in candidates if self.contains(*cell)]

    def corners(self, row: int, col: int) -> list[tuple[float, float]]:
        return [(col, row), (col + 1, row), (col + 1, row + 1), (col, row + 1)]

    def sample_offsets(self) -> FloatArray:
        steps = (np.arange(4) + 0.5) / 4 - 0.5
        dx, dy = np.meshgrid(steps, steps)
        return np.column_stack([dx.ravel(), dy.ravel()])

    def edge_distance(self, x: Coord, y: Coord) -> Coord:
        dx = np.abs(np.asarray(x, dtype=float) % 1 - 0.5)
        dy = np.abs(np.asarray(y, dtype=float) % 1 - 0.5)
        return 0.5 - np.maximum(dx, dy)

    def edge_normals(self) -> FloatArray:
        return np.array([(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)])

    def seal_corner_gaps(self, mask: BoolGrid) -> BoolGrid:
        sealed = mask.copy()
        while True:
            top_left, top_right = sealed[:-1, :-1], sealed[:-1, 1:]
            bottom_left, bottom_right = sealed[1:, :-1], sealed[1:, 1:]
            falling = top_left & bottom_right & ~top_right & ~bottom_left
            rising = top_right & bottom_left & ~top_left & ~bottom_right
            if not (falling.any() or rising.any()):
                return sealed
            sealed[:-1, 1:] |= falling
            sealed[:-1, :-1] |= rising


@dataclass(frozen=True, slots=True)
class HexGrid(Grid):
    """Hexagons with a point at the top, laid in rows; odd rows sit half a cell further east."""

    kind: ClassVar[GridKind] = "hex"
    corner_share: ClassVar[float] = 1 / 3

    @classmethod
    def covering(cls, width_km: float, height_km: float, km_per_cell: float) -> Self:
        width, height = width_km / km_per_cell, height_km / km_per_cell
        cols = max(1, round(width - 0.5))
        rows = max(1, round((height - 2 * HEX_RADIUS) / HEX_ROW_STEP) + 1)
        return cls(rows=rows, cols=cols, km_per_cell=km_per_cell)

    def size(self) -> tuple[float, float]:
        return self.cols + 0.5, 2 * HEX_RADIUS + (self.rows - 1) * HEX_ROW_STEP

    def center(self, row: Index, col: Index) -> tuple[Coord, Coord]:
        return col + 0.5 + 0.5 * (row % 2), HEX_RADIUS + row * HEX_ROW_STEP

    def cell_at(self, x: Coord, y: Coord) -> tuple[Index, Index]:
        px = np.asarray(x, dtype=float) - 0.5
        py = np.asarray(y, dtype=float) - HEX_RADIUS
        q = px - py / math.sqrt(3)
        r = py * 2 / math.sqrt(3)
        s = -q - r
        round_q, round_r, round_s = np.rint(q), np.rint(r), np.rint(s)
        error_q, error_r, error_s = abs(round_q - q), abs(round_r - r), abs(round_s - s)
        fix_q = (error_q > error_r) & (error_q > error_s)
        fix_r = ~fix_q & (error_r > error_s)
        round_q = np.where(fix_q, -round_r - round_s, round_q)
        round_r = np.where(fix_r, -round_q - round_s, round_r)
        row = round_r.astype(np.int64)
        col = round_q.astype(np.int64) + (row - (row & 1)) // 2
        return row, col

    def neighbours(self, row: int, col: int) -> list[Cell]:
        candidates = [(row + d_row, col + d_col) for d_col, d_row in _HEX_STEPS[row % 2]]
        return [cell for cell in candidates if self.contains(*cell)]

    def corners(self, row: int, col: int) -> list[tuple[float, float]]:
        x, y = self.center(row, col)
        return [
            (x + HEX_RADIUS * math.cos(angle), y + HEX_RADIUS * math.sin(angle))
            for angle in _HEX_CORNER_ANGLES
        ]

    def sample_offsets(self) -> FloatArray:
        steps = (np.arange(5) + 0.5) / 5 - 0.5
        dx, dy = np.meshgrid(steps, steps * 2 * HEX_RADIUS)
        inside = np.abs(dy) < HEX_RADIUS - np.abs(dx) / math.sqrt(3) - 1e-9
        return np.column_stack([dx[inside], dy[inside]])

    def edge_distance(self, x: Coord, y: Coord) -> Coord:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        cx, cy = self.center(*self.cell_at(x, y))
        dx, dy = x - cx, y - cy
        slant = math.sqrt(3) / 2
        across = np.maximum(np.abs(dx / 2 + slant * dy), np.abs(dx / 2 - slant * dy))
        return 0.5 - np.maximum(np.abs(dx), across)

    def edge_normals(self) -> FloatArray:
        angles = np.radians(np.arange(0, 360, 60))
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def _find_track_ends(self) -> tuple[int, ...]:
        """The middle of every edge, and the top and bottom corners, through which track runs
        straight up and down along the edge between two cells: eight directions in all."""
        steps = self.track_steps()
        return tuple(
            point for point, (dx, _) in enumerate(steps) if point % 2 == 0 or abs(float(dx)) < 1e-9
        )

    def seal_corner_gaps(self, mask: BoolGrid) -> BoolGrid:
        return mask.copy()


GRIDS: dict[GridKind, type[Grid]] = {"square": SquareGrid, "hex": HexGrid}

# How many cells' surroundings flat_within keeps: far more than a big railway's stations at
# every reach they have.
WITHIN_KEPT = 1024


@dataclass(frozen=True, slots=True, eq=False)
class Within:
    """The cells at most some steps from one cell, as flat indices in the order Grid.within finds
    them, nearest first, and how many steps away each is."""

    cells: npt.NDArray[np.intp]
    steps: FloatArray


@lru_cache(maxsize=WITHIN_KEPT)
def flat_within(grid: Grid, cell: Cell, steps: int) -> Within:
    """Grid.within as flat indices, worked out once for each grid, cell and number of steps. Its
    arrays can't be written to, so no caller can spoil what is kept."""
    reached = grid.within(cell, steps)
    rows, cols = zip(*reached, strict=True)
    cells = np.ravel_multi_index((rows, cols), grid.shape)
    away = np.array(list(reached.values()), dtype=np.float64)
    cells.setflags(write=False)
    away.setflags(write=False)
    return Within(cells, away)


@lru_cache(maxsize=WITHIN_KEPT)
def flat_within_any(grid: Grid, cells: tuple[Cell, ...], steps: int) -> Within:
    """The cells at most some steps from any of these cells, as flat_within, each as many steps
    away as from the nearest of them: the reach of a station with more platforms."""
    if len(cells) == 1:
        return flat_within(grid, cells[0], steps)
    parts = [flat_within(grid, cell, steps) for cell in cells]
    found = np.concatenate([part.cells for part in parts])
    away = np.concatenate([part.steps for part in parts])
    order = np.lexsort((away, found))
    found, away = found[order], away[order]
    first = np.concatenate(([True], found[1:] != found[:-1]))
    found, away = found[first], away[first]
    nearest = np.argsort(away, kind="stable")
    found, away = found[nearest], away[nearest]
    found.setflags(write=False)
    away.setflags(write=False)
    return Within(found, away)


# Where track leaving each cell by each point runs into, by grid (Grid.track_exit).
_EXITS: dict[Grid, dict[tuple[Cell, int], tuple[Cell, int]]] = {}
# Grid.track_steps of each kind of grid, which are the same for every grid of the kind.
_STEPS: dict[type[Grid], FloatArray] = {}


def _track_steps(grid: Grid) -> FloatArray:
    """Grid.track_steps, worked out once for each kind of grid: track asks for them many
    thousands of times when the railway changes. Never changed in place: Grid.track_steps
    hands out copies."""
    steps = _STEPS.get(type(grid))
    if steps is None:
        steps = _STEPS[type(grid)] = grid._find_track_steps()
        steps.flags.writeable = False
    return steps


@lru_cache(maxsize=64)
def _track_ends(grid: Grid) -> tuple[int, ...]:
    return grid._find_track_ends()


@lru_cache(maxsize=4096)
def _track_turn(grid: Grid, start: int, end: int) -> float:
    return grid._find_track_turn(start, end)


type Offsets = tuple[tuple[int, int, int], ...]


@lru_cache(maxsize=64)
def parity_steps(grid: Grid, steps: int) -> tuple[Offsets, Offsets]:
    """Every cell at most `steps` steps from a cell on an even row, and from one on an odd row,
    as its row and column offsets and how many steps away it is, nearest first: on a hex grid
    the offsets depend on the row. Worked out on a grid of the same kind big enough that none
    is cut off by its edge."""
    middle = 2 * steps + 2
    roomy = replace(grid, rows=2 * middle + 2, cols=2 * middle + 2)
    even, odd = (
        tuple(
            (r - row, c - middle, away)
            for (r, c), away in sorted(
                roomy.within((row, middle), steps).items(), key=lambda item: item[1]
            )
        )
        for row in (middle, middle + 1)
    )
    return even, odd
