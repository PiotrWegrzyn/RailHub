"""Ways to divide a map into cells: squares, or hexagons with a point at the top.

Cells are addressed as (row, col) with row 0 along the north edge, and every property of a map is
a numpy array of shape (rows, cols). Positions on the map are measured in cell units: centres of
neighbouring cells are one unit apart, so units times km_per_cell gives kilometres.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal, Self

import numpy as np
import numpy.typing as npt

type Cell = tuple[int, int]
type GridKind = Literal["square", "hex"]
type Coord = float | npt.NDArray[np.float64]
type Index = int | npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolGrid = npt.NDArray[np.bool_]

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
    # How many equal turns of the view bring the grid back onto itself.
    symmetry: ClassVar[int]

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

    @abstractmethod
    def neighbours(self, row: int, col: int) -> list[Cell]:
        """Cells that share an edge with this one."""

    @abstractmethod
    def corners(self, row: int, col: int) -> list[tuple[float, float]]: ...

    @abstractmethod
    def sample_offsets(self) -> FloatArray:
        """Points spread evenly inside a cell, relative to its centre, as an (n, 2) array."""

    @abstractmethod
    def seal_corner_gaps(self, mask: BoolGrid) -> BoolGrid:
        """Add cells to the mask wherever two of its cells touch only at a corner, so that what
        the mask outlines, like a river, cannot be crossed diagonally."""


@dataclass(frozen=True, slots=True)
class SquareGrid(Grid):
    kind: ClassVar[GridKind] = "square"
    symmetry: ClassVar[int] = 4

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
    symmetry: ClassVar[int] = 6

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

    def seal_corner_gaps(self, mask: BoolGrid) -> BoolGrid:
        return mask.copy()


GRIDS: dict[GridKind, type[Grid]] = {"square": SquareGrid, "hex": HexGrid}
