"""Monthly summaries: what was built and pulled down within reach of each station (the cells it
gathers goods from) when a month turned over, worked out from the changes growth logged while the
month ran (railroads.sim.growth), so a month costs what changed rather than the stations' reach."""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind
from railroads.sim.grids import Cell, Grid, flat_within


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
    """The cells within reach of each station, worked out once per set of stations and reaches."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self._stations: tuple[tuple[Cell, int], ...] = ()
        self._cells: dict[Cell, tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] = {}

    def of(
        self, stations: list[Cell], radius: Callable[[Cell], int]
    ) -> dict[Cell, tuple[npt.NDArray[np.intp], ...]]:
        """Each station's cells within `radius` of it."""
        key = tuple((station, radius(station)) for station in stations)
        if key != self._stations:
            self._stations = key
            self._cells = {}
            for station, steps in key:
                flat = flat_within(self.grid, station, steps).cells
                self._cells[station] = np.divmod(flat, self.grid.cols)
        return self._cells  # type: ignore[return-value]


def summarize(
    date: str,
    kinds: Collection[BuildingKind],
    reach: Mapping[Cell, tuple[npt.NDArray[np.intp], ...]],
    logged: Mapping[tuple[Cell, BuildingKind], int],
) -> MonthSummary | None:
    """What changed within each station's reach, from the net change in each cell's count of
    each kind logged over the month (Growth.log_changes), the kinds in the order of `kinds`; None
    when nothing did."""
    moved: dict[Cell, dict[BuildingKind, int]] = {}
    for (cell, kind), change in logged.items():
        if change:
            moved.setdefault(cell, {})[kind] = change
    if not moved:
        return None
    stations = []
    for station, (rows, cols) in reach.items():
        reached = zip(rows.tolist(), cols.tolist(), strict=True)
        cells = tuple(cell for cell in reached if cell in moved)
        totals: dict[BuildingKind, int] = {}
        for cell in cells:
            for kind, change in moved[cell].items():
                totals[kind] = totals.get(kind, 0) + change
        changes = {kind: totals[kind] for kind in kinds if totals.get(kind)}
        if changes:
            stations.append(StationSummary(station, changes, cells))
    if not stations:
        return None
    return MonthSummary(date, tuple(stations))
