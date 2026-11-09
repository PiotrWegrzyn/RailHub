"""Monthly summaries: what was built and pulled down within reach of each station
(COLLECTION_RADIUS) when a month turned over."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import COLLECTION_RADIUS
from railroads.sim.grids import Cell, Grid
from railroads.sim.growth import Growth

type Counts = Mapping[BuildingKind, npt.NDArray[np.int64]]


@dataclass(frozen=True, slots=True)
class StationSummary:
    station: Cell
    # How many more (or, below 0, fewer) of each kind stand within the station's reach.
    changes: Mapping[BuildingKind, int]
    # The cells within reach whose buildings changed.
    cells: tuple[Cell, ...]


@dataclass(frozen=True, slots=True)
class MonthSummary:
    date: str
    stations: tuple[StationSummary, ...]

    @property
    def built(self) -> int:
        return sum(n for s in self.stations for n in s.changes.values() if n > 0)

    @property
    def pulled_down(self) -> int:
        return -sum(n for s in self.stations for n in s.changes.values() if n < 0)


class Reach:
    """The cells within reach of each station, worked out once per set of stations."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self._stations: tuple[Cell, ...] = ()
        self._cells: dict[Cell, tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] = {}

    def of(self, stations: list[Cell]) -> dict[Cell, tuple[npt.NDArray[np.intp], ...]]:
        if tuple(stations) != self._stations:
            self._stations = tuple(stations)
            self._cells = {}
            for station in stations:
                rows, cols = zip(*self.grid.within(station, COLLECTION_RADIUS), strict=True)
                self._cells[station] = (np.array(rows), np.array(cols))
        return self._cells  # type: ignore[return-value]


def snapshot(growth: Growth, reach: Mapping[Cell, tuple[npt.NDArray[np.intp], ...]]) -> dict:
    """The count of every kind in every cell within reach, before a month grows."""
    return {
        station: {kind: growth.count(kind)[rows, cols].copy() for kind in growth.rule.points}
        for station, (rows, cols) in reach.items()
    }


def summarize(
    date: str,
    growth: Growth,
    reach: Mapping[Cell, tuple[npt.NDArray[np.intp], ...]],
    before: dict,
) -> MonthSummary | None:
    """What changed within each station's reach since `before`; None when nothing did."""
    stations = []
    for station, (rows, cols) in reach.items():
        if station not in before:
            continue
        changes: dict[BuildingKind, int] = {}
        moved = np.zeros(rows.size, dtype=bool)
        for kind, counts in before[station].items():
            delta = growth.count(kind)[rows, cols] - counts
            moved |= delta != 0
            if total := int(delta.sum()):
                changes[kind] = total
        cells = tuple((int(r), int(c)) for r, c in zip(rows[moved], cols[moved], strict=True))
        if changes:
            stations.append(StationSummary(station, changes, cells))
    if not stations:
        return None
    return MonthSummary(date, tuple(stations))
