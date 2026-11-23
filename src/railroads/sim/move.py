"""Where the cells of a game's map land when the game moves onto a bigger map, as when the player
buys land next to it (railroads.sim.regrid): every cell moves down and right by the rows and
columns added above and to the left. Everything the game keeps by cell moves by a Move."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import HEX_ROW_STEP, Cell, Grid, HexGrid


@dataclass(frozen=True)
class Move:
    """Where the cells of the `old` grid lie on the `new`: `rows` further down and `cols`
    further right."""

    old: Grid
    new: Grid
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows % 2 and isinstance(self.old, HexGrid):
            raise ValueError("Hex rows move in twos, so every hex keeps its neighbours")
        if (
            self.rows < 0
            or self.cols < 0
            or self.rows + self.old.rows > self.new.rows
            or self.cols + self.old.cols > self.new.cols
        ):
            raise ValueError(f"{self.old.shape} moved by {self.rows, self.cols} is off the new map")

    def cell(self, cell: Cell) -> Cell:
        return cell[0] + self.rows, cell[1] + self.cols

    def cells(self, cells: Iterable[Cell]) -> set[Cell]:
        return {self.cell(cell) for cell in cells}

    def keys[V](self, by_cell: Mapping[Cell, V]) -> dict[Cell, V]:
        return {self.cell(cell): value for cell, value in by_cell.items()}

    def parcels[V](self, parcels: Mapping[Cell | None, V]) -> dict[Cell | None, V]:
        return {None if cell is None else self.cell(cell): value for cell, value in parcels.items()}

    def flat(self, index: int) -> int:
        row, col = divmod(index, self.old.cols)
        return (row + self.rows) * self.new.cols + col + self.cols

    def flats(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        rows, cols = np.divmod(np.asarray(indices, dtype=np.intp), self.old.cols)
        return (rows + self.rows) * self.new.cols + cols + self.cols

    def grid[T: np.generic](self, array: npt.NDArray[T], fill: Any = 0) -> npt.NDArray[T]:
        """An array over the old grid's rows and columns, the first two axes, on the new grid."""
        moved = np.full((*self.new.shape, *array.shape[2:]), fill, dtype=array.dtype)
        moved[self.rows : self.rows + self.old.rows, self.cols : self.cols + self.old.cols] = array
        return moved

    def flat_cells[T: np.generic](self, array: npt.NDArray[T], fill: Any = 0) -> npt.NDArray[T]:
        """An array over the old grid's flat cells, its last axis, on the new grid's."""
        lead = array.shape[:-1]
        on_grid = np.moveaxis(array.reshape(*lead, *self.old.shape), (-2, -1), (0, 1))
        moved = np.moveaxis(self.grid(on_grid, fill), (0, 1), (-2, -1))
        return np.ascontiguousarray(moved.reshape(*lead, self.new.rows * self.new.cols))

    def point(self, point: tuple[float, float]) -> tuple[float, float]:
        """A point on the map in cell units (railroads.sim.grids.Grid.center)."""
        step = HEX_ROW_STEP if isinstance(self.old, HexGrid) else 1.0
        return point[0] + self.cols, point[1] + self.rows * step
