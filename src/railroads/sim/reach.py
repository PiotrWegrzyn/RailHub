"""How far goods go on their own: LOCAL_REACH cells, losing a share of the amount with each one
(REACH_SHARES), and the rings of cells around every cell they go out in."""

from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import Grid, parity_steps

# The most of a seller's month a cell can take, by how many cells away from it it is: all of it in
# the seller's own cell, then falling by the Fibonacci numbers, 13, 8, 5, 3, 2 and 1 twenty-firsts,
# and nothing from LOCAL_REACH + 1 cells away.
REACH_SHARES = (21 / 21, 13 / 21, 8 / 21, 5 / 21, 3 / 21, 2 / 21, 1 / 21)


LOCAL_REACH = len(REACH_SHARES) - 1


# The cells of -1 around the map that let MonthOfTrade._pairs look LOCAL_REACH cells out from any cell
# without checking for the edge.
RING_PAD = LOCAL_REACH


def reach_share(away: int) -> float:
    """The most of a seller's month a cell `away` cells from it can take."""
    return REACH_SHARES[away] if 0 <= away <= LOCAL_REACH else 0.0


def ring_starts(away: npt.NDArray[np.intp | np.float64]) -> tuple[int, ...]:
    """Where the pairs of each distance start among pairs in order of how far apart they are, and
    where the last ends."""
    return (0, *(np.flatnonzero(np.diff(away)) + 1).tolist(), away.size)


class Neighbourhoods:
    """Every cell's neighbours, as flat indices."""

    def __init__(self, grid: Grid) -> None:
        rows, cols = grid.neighbour_index_grids()
        own = np.indices(grid.shape)
        real = ((rows != own[0]) | (cols != own[1])).reshape(len(rows), -1)
        index = np.ravel_multi_index((rows, cols), grid.shape).reshape(len(rows), -1)
        # Off the map, a neighbour points at one extra cell past the last.
        self.index = np.where(real, index, grid.rows * grid.cols)


@dataclass(frozen=True, slots=True, eq=False)
class Rings:
    """The steps to every cell up to LOCAL_REACH cells away, nearest first, as row and column
    steps for even rows and for odd rows, and how far each step goes."""

    rows: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    cols: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    away: npt.NDArray[np.intp]


@cache
def rings_of(grid: Grid) -> Rings:
    even, odd = parity_steps(grid, LOCAL_REACH)
    rows, cols = (
        tuple(np.array([offset[part] for offset in steps], dtype=np.intp) for steps in (even, odd))
        for part in (0, 1)
    )
    away = np.array([offset[2] for offset in odd], dtype=np.intp)
    return Rings((rows[0], rows[1]), (cols[0], cols[1]), away)
