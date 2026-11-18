"""Making room on the map (railroads.sim.growth keeps one Room): clearing a cell for track, a
station or a merged works, putting up a building that found no room in its own cell nearby, and
housing people a skyscraper pulled their flats down on. A building cleared away comes down and
never moves to another cell, the player's too, with their shares in it.

A building that comes up in a cell goes up in it or a cell next to it, picked at random by how
well their land suits it (scatter). One that finds no room there goes up in the nearest cell
within SPILL_STEPS that has the room and land for it and that its ladder's caps allow, chosen
among cells equally near at random by how well their land suits it. People rehoused go into the nearest cells within
HOMES_STEPS with room.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.grids import Cell
from railroads.sim.ladders import POINTS, TOWN_LAND, Ladder, Points

if TYPE_CHECKING:
    from railroads.sim.growth import Growth

HOMES_STEPS = 6
SPILL_STEPS = 3


def leaving_order(
    buildings: list[Building], points: Mapping[BuildingKind, tuple[Points, int]] = POINTS
) -> list[Building]:
    """The order buildings leave a cell to make room: the smallest first."""
    return sorted(
        buildings, key=lambda b: (points.get(b.kind, (Points.POPULATION, 0))[1], b.kind.value)
    )


class Room:
    def __init__(self, growth: Growth) -> None:
        self._growth = growth

    def clear(self, cell: Cell, units: int) -> set[Cell]:
        """Makes `units` of room in the cell for track, a station or a merged works, the smallest
        buildings first (railroads.sim.demolition prices them in the same order): each comes
        down, the player's shares in it with it; a building never moves to another cell. Returns
        the cells whose buildings changed."""
        growth = self._growth
        constructions = growth.constructions
        changed: set[Cell] = set()
        while constructions.space_left(cell) < units:
            here = constructions.buildings_at(cell)
            if not here:
                break
            growth.take_down(leaving_order(here, growth.rule.points)[0])
            changed.add(cell)
        return changed

    def scatter(self, cell: Cell, ladder: Ladder, rng: np.random.Generator) -> set[Cell]:
        """Puts up the ladder's first kind that came up in this cell in it or a cell next to
        it, picked at random by how well their land suits it, or, where none of them will take
        it, in the nearest cell further off that will (spill); returns the cell it went up in, if
        it found one."""
        return self._put_near(cell, ladder, rng, 1, nearest_first=False) or self.spill(
            cell, ladder, rng
        )

    def spill(self, cell: Cell, ladder: Ladder, rng: np.random.Generator) -> set[Cell]:
        """Puts up the ladder's first kind in the nearest cell but this one that will take it,
        picked among cells equally near at random by how well their land suits it; returns the
        cell it went up in, if it found one."""
        return self._put_near(cell, ladder, rng, SPILL_STEPS, nearest_first=True)

    def _put_near(
        self, cell: Cell, ladder: Ladder, rng: np.random.Generator, most: int, nearest_first: bool
    ) -> set[Cell]:
        growth = self._growth
        constructions, grid = growth.constructions, growth.grid
        reached = [(near, steps) for near, steps in grid.within(cell, most).items()]
        steps = np.array([step for _, step in reached])
        rows, cols = zip(*(near for near, _ in reached), strict=True)
        flat = np.ravel_multi_index((rows, cols), grid.shape)
        land = growth.land(ladder.land, flat)
        first = ladder.kinds[0]
        suits = rng.random(steps.size) ** (1.0 / np.maximum(land, 1e-12))
        order = np.lexsort((-suits, steps)) if nearest_first else np.argsort(-suits)
        for index in order.tolist():
            near = reached[index][0]
            new = Building(near, first)
            if (
                (near != cell or not nearest_first)
                and land[index] > 0
                and constructions.why_not(new) is None
                and growth.ladder_allows(ladder, near)
            ):
                constructions.build(new)
                growth.add(near, first, 1)
                return {near}
        return set()

    def house(self, cell: Cell, homes: int) -> set[Cell]:
        """Puts up as many as `homes` houses in the nearest cells around this one with room and
        land for them; returns the cells they went up in."""
        growth = self._growth
        constructions = growth.constructions
        changed: set[Cell] = set()
        reached = sorted(growth.grid.within(cell, HOMES_STEPS).items(), key=lambda item: item[1])
        for near, _ in reached:
            house = Building(near, BuildingKind.HOUSE)
            if growth.land(TOWN_LAND, self._flat([near]))[0] <= 0:
                continue
            while homes and constructions.why_not(house) is None:
                constructions.build(house)
                growth.add(near, BuildingKind.HOUSE, 1)
                changed.add(near)
                homes -= 1
            if not homes:
                break
        return changed

    def _flat(self, cells: list[Cell]) -> npt.NDArray[np.intp]:
        rows, cols = zip(*cells, strict=True)
        return np.ravel_multi_index((rows, cols), self._growth.grid.shape)
