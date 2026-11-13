"""The plane points at the middle of the pixels of a block of the big picture, worked out once for
every layer of a chunk drawn over them. The ground, the raised ground, the fields and the woods
each look at their own part of the same block, so a chunk turns its pixels into plane points and
finds the cells under them once instead of once a layer.

A point is the same however large the block it was worked out in: pixel middles are whole numbers
and halves (quarters at two pixels to an art pixel), which add up exactly, so the plane point of a
pixel comes out the same to the last bit whichever block it is part of."""

from functools import cached_property

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import Grid
from railroads.ui.pixel.projection import FloatArray, IntArray, PixelProjection

type BoolArray = npt.NDArray[np.bool_]
type Block = tuple[slice, slice]


class Lattice:
    """The middle of every pixel over `width` by `rows` art pixels from (`left`, `top`), `k`
    pixels to an art pixel each way, as plane points and the cells they lie in on `grid`."""

    def __init__(
        self,
        projection: PixelProjection,
        left: int,
        top: int,
        width: int,
        rows: int,
        k: int = 1,
        grid: Grid | None = None,
    ) -> None:
        if k & (k - 1):
            raise ValueError("Pixel middles add up exactly only at a power of two to an art pixel")
        self.projection, self.grid = projection, grid
        self.left, self.top, self.width, self.rows, self.k = left, top, width, rows, k

    def holds(self, left: int, top: int, width: int, rows: int) -> bool:
        """Whether the lattice covers these art pixels."""
        return (
            self.left <= left
            and left + width <= self.left + self.width
            and self.top <= top
            and top + rows <= self.top + self.rows
        )

    def part(self, left: int, top: int, width: int, rows: int) -> Block:
        """Where these art pixels lie among the lattice's pixels."""
        if not self.holds(left, top, width, rows):
            raise ValueError("The lattice does not cover these art pixels")
        k = self.k
        down, across = (top - self.top) * k, (left - self.left) * k
        return slice(down, down + rows * k), slice(across, across + width * k)

    @cached_property
    def _plane(self) -> tuple[FloatArray, FloatArray]:
        k = self.k
        py, px = np.mgrid[0 : self.rows * k, 0 : self.width * k]
        return self.projection.to_plane(self.left + (px + 0.5) / k, self.top + (py + 0.5) / k)

    @cached_property
    def _cells(self) -> tuple[IntArray, IntArray, BoolArray]:
        if self.grid is None:
            raise ValueError("A lattice finds cells only on a grid")
        grid = self.grid
        r, c = grid.cell_at(*self._plane)
        inside = grid.contains(r, c)
        return np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1), inside

    def plane(self, left: int, top: int, width: int, rows: int) -> tuple[FloatArray, FloatArray]:
        """The plane points of these art pixels."""
        block = self.part(left, top, width, rows)
        x, y = self._plane
        return x[block], y[block]

    def cells(
        self, left: int, top: int, width: int, rows: int
    ) -> tuple[IntArray, IntArray, BoolArray]:
        """The cells under these art pixels, as rows and columns held to the grid, and which of
        them lie on it."""
        block = self.part(left, top, width, rows)
        r, c, inside = self._cells
        return r[block], c[block], inside[block]
