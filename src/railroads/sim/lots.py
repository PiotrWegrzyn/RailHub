"""Land the player can buy next to the map, a lot at a time.

A game starts on the square in the middle of its place (railroads.sim.tilemap.TileMap.square_middle),
which splits into two by two lots; every lot is a quarter of it, half as wide and half as tall.
Lots are numbered by their row and column in steps of a lot, (0, 0) being the top-left quarter of
the starting square, so lots to its left or above have negative numbers. Any lot sharing a side
with the player's land can be bought, the first for FIRST_PRICE and each after it for the next
Fibonacci number of thousands.

The game's map covers the smallest rectangle of lots around the player's land; the lots in it the
player has not bought are locked (railroads.sim.tilemap.TileMap.locked)."""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

type Lot = tuple[int, int]

# The first lot costs the FIRST_FIBONACCI-th Fibonacci number of PRICE_UNIT dollars, $377,000, and
# every lot bought after it the next one: $610,000, $987,000 and so on.
FIRST_FIBONACCI = 14
PRICE_UNIT = 1000
# The least share of a lot's cells that must be land for it to be offered.
LEAST_LAND = 0.02


def fibonacci(n: int) -> int:
    """The n-th Fibonacci number, counting 1, 1, 2, 3, 5 from n = 1."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@dataclass
class Lots:
    """The lots of one game: their size in cells, where lot (0, 0) starts on the place's map,
    which the player owns and how many were bought."""

    rows: int
    cols: int
    # The place's map cell (railroads.newgame.place_map) lot (0, 0)'s top-left cell is.
    top: int
    left: int
    owned: set[Lot] = field(default_factory=lambda: {(0, 0), (0, 1), (1, 0), (1, 1)})
    bought: int = 0

    @classmethod
    def of_square(cls, top: int, left: int, rows: int, cols: int) -> Self:
        """The lots of a game starting on the square of `rows` by `cols` cells at row `top` and
        column `left` of its place's map."""
        if rows % 4 or cols % 2:
            raise ValueError(f"A {rows} by {cols} square doesn't split into four even lots")
        return cls(rows=rows // 2, cols=cols // 2, top=top, left=left)

    def corner(self) -> Lot:
        """The lot at the top-left of the map: the topmost row and leftmost column of owned lots."""
        return min(row for row, _ in self.owned), min(col for _, col in self.owned)

    def span(self) -> tuple[int, int]:
        """How many lots tall and wide the map is."""
        top, left = self.corner()
        bottom = max(row for row, _ in self.owned)
        right = max(col for _, col in self.owned)
        return bottom - top + 1, right - left + 1

    def window(self) -> tuple[int, int, int, int]:
        """Where the map lies on its place's map: top row, left column, rows and columns."""
        top, left = self.corner()
        tall, wide = self.span()
        return (
            self.top + top * self.rows,
            self.left + left * self.cols,
            tall * self.rows,
            wide * self.cols,
        )

    def cell_of(self, lot: Lot) -> tuple[int, int]:
        """The map cell a lot's top-left cell is, which lies off the map for a lot beyond it."""
        top, left = self.corner()
        return (lot[0] - top) * self.rows, (lot[1] - left) * self.cols

    def place_cell_of(self, lot: Lot) -> tuple[int, int]:
        """The place's map cell a lot's top-left cell is."""
        return self.top + lot[0] * self.rows, self.left + lot[1] * self.cols

    def lot_at(self, row: int, col: int) -> Lot:
        """The lot a map cell lies in; past the map's edges too."""
        top, left = self.corner()
        return top + row // self.rows, left + col // self.cols

    def next_to(self) -> list[Lot]:
        """Every lot not owned that shares a side with one owned, in order."""
        around = {
            (row + d_row, col + d_col)
            for row, col in self.owned
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1))
        }
        return sorted(around - self.owned)

    def offered(self, land: npt.NDArray[np.bool_]) -> list[Lot]:
        """The lots next to the player's land with at least LEAST_LAND of land, as `land` tells
        it: True for each land cell of the place's map."""
        return [lot for lot in self.next_to() if self.land_share(lot, land) >= LEAST_LAND]

    def land_share(self, lot: Lot, land: npt.NDArray[np.bool_]) -> float:
        """The share of a lot's cells that are land on the place's map, none past its edges."""
        top, left = self.place_cell_of(lot)
        rows = slice(max(top, 0), max(min(top + self.rows, land.shape[0]), 0))
        cols = slice(max(left, 0), max(min(left + self.cols, land.shape[1]), 0))
        return float(land[rows, cols].sum()) / (self.rows * self.cols)

    def price(self) -> int:
        """What the next lot costs, in dollars."""
        return fibonacci(FIRST_FIBONACCI + self.bought) * PRICE_UNIT

    def buy(self, lot: Lot) -> None:
        if lot not in self.next_to():
            raise ValueError(f"Lot {lot} doesn't share a side with the player's land")
        self.owned.add(lot)
        self.bought += 1

    def locked(self) -> npt.NDArray[np.bool_] | None:
        """Each map cell in a lot not owned, or None where every lot of the map is owned."""
        tall, wide = self.span()
        top, left = self.corner()
        owned = np.zeros((tall, wide), dtype=bool)
        for row, col in self.owned:
            owned[row - top, col - left] = True
        if owned.all():
            return None
        return np.repeat(np.repeat(~owned, self.rows, axis=0), self.cols, axis=1)
