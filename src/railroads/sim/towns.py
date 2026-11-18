"""Towns as colours on the map: every cell whose buildings send passengers (railroads.sim.ladders
PASSENGERS_PER_POINT) belongs to a town, numbered from 1, 0 for none.

Whenever a building sending passengers goes up, every sending cell next to it (TOWN_REACH), its own
included, takes the town that sends the most passengers there, or a new town if none does, and
every other town with a cell there joins it whole. So towns that grow into each other merge, and a
building on its own starts one. A cell that empties leaves its town. A station belongs to the
town that sends the most passengers from the cells it hands out to, worked out when it is built and
every year (railroads.sim.railway). Passengers and mail never board for a stop of their own
station's town, so two stations in one town can't carry them to each other for pay.
"""

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import Cell, Grid

TOWN_REACH = 1


class Towns:
    def __init__(self, grid: Grid) -> None:
        self._grid = grid
        # Each cell's town, 0 for none, and the passengers its buildings send a month.
        self.town = np.zeros(grid.shape, dtype=np.int32)
        self.sending = np.zeros(grid.shape, dtype=np.float64)
        self._next = 1

    def add(self, cell: Cell, passengers: float) -> None:
        """Counts buildings put up (or pulled down, below 0) in a cell that send this many
        passengers a month more; one put up paints the cells around it one town."""
        self.sending[cell] = max(0.0, self.sending[cell] + passengers)
        if self.sending[cell] <= 0:
            self.town[cell] = 0
        elif passengers > 0:
            self._paint_around(cell)

    def _paint_around(self, cell: Cell) -> None:
        around = [near for near in self._grid.within(cell, TOWN_REACH) if self.sending[near] > 0]
        votes: dict[int, float] = {}
        for near in around:
            town = int(self.town[near])
            if town:
                votes[town] = votes.get(town, 0.0) + float(self.sending[near])
        if votes:
            town = max(votes, key=lambda town: (votes[town], -town))
        else:
            town = self._next
            self._next += 1
        for near in around:
            self.town[near] = town
        merged = [other for other in votes if other != town]
        if merged:
            self.town[np.isin(self.town, merged)] = town

    def settle(self, sending: npt.NDArray[np.float64]) -> None:
        """Gives every cell sending passengers a town, the busiest first, as if each went up
        in that order: for the buildings standing when play or a map begins."""
        self.town[:] = 0
        self.sending[:] = 0.0
        self._next = 1
        flat = np.flatnonzero(sending.ravel() > 0)
        order = flat[np.argsort(-sending.ravel()[flat], kind="stable")]
        for index in order.tolist():
            cell = divmod(index, self._grid.cols)
            self.add(cell, float(sending[cell]))

    def most_in(self, cells: npt.NDArray[np.intp]) -> int:
        """The town sending the most passengers from these cells, as flat indices; 0 for none."""
        towns = self.town.ravel()[cells]
        sending = self.sending.ravel()[cells]
        held = (towns > 0) & (sending > 0)
        if not held.any():
            return 0
        totals = np.bincount(towns[held], weights=sending[held])
        return int(np.argmax(totals))
