"""Making room on the map (railroads.sim.growth keeps one Room): clearing a cell for track or a
station, putting up a building that found no room in its own cell nearby, housing people a
skyscraper pulled their flats down on, and putting up again the player's buildings that growth
pulled down.

A building that finds no room in its cell goes up in the nearest cell within SPILL_STEPS that has
the room and land for it and that its ladder's caps allow, chosen among cells equally near at
random by how well their land suits it. People rehoused go into the nearest cells within
HOMES_STEPS with room, and the player's buildings go up again within RELOCATE_STEPS.
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
RELOCATE_STEPS = 12
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

    def clear(self, cell: Cell, units: int, avoid: frozenset[Cell] = frozenset()) -> set[Cell]:
        """Makes `units` of room in the cell for track or a station, the smallest buildings first
        (railroads.sim.demolition prices them in the same order): each moves to a cell next to it
        with room for it (never into `avoid`) if it can, and otherwise comes down. Returns the
        cells whose buildings changed."""
        growth = self._growth
        constructions = growth.constructions
        changed: set[Cell] = set()
        while constructions.space_left(cell) < units:
            here = [b for b in constructions.at(cell) if isinstance(b, Building)]
            if not here:
                break
            victim = leaving_order(here, growth.rule.points)[0]
            target = self._beside(victim, avoid | {cell})
            owned = growth.take_down(victim)
            changed.add(cell)
            if target is not None:
                constructions.build(Building(target, victim.kind))
                if victim.kind in growth.counts:
                    growth.add(target, victim.kind, 1)
                if owned:
                    growth.owners.own(target, victim.kind)
                changed.add(target)
            elif owned:
                changed |= self.relocate(victim)
        return changed

    def _beside(self, building: Building, avoid: frozenset[Cell]) -> Cell | None:
        """The cell next to the building's with the most room that would take it, if any."""
        constructions = self._growth.constructions
        options = [
            near
            for near in self._growth.grid.neighbours(*building.cell)
            if near not in avoid and constructions.why_not(Building(near, building.kind)) is None
        ]
        return max(options, key=constructions.space_left, default=None)

    def relocate(self, building: Building) -> set[Cell]:
        """Puts up again, as the player's, a building of theirs that growth pulled down: in the
        nearest cell with room and the land its ladder grows on."""
        growth = self._growth
        constructions, grid = growth.constructions, growth.grid
        ladder = next((ld for ld in growth.rule.ladders if building.kind in ld.kinds), None)
        reached = grid.within(building.cell, RELOCATE_STEPS)
        for near, _ in sorted(reached.items(), key=lambda item: item[1]):
            if near == building.cell:
                continue
            moved = Building(near, building.kind)
            if constructions.why_not(moved) is not None:
                continue
            if ladder is not None:
                flat = np.array([np.ravel_multi_index(near, grid.shape)])
                land = ladder.land.weights(constructions.tile_map, flat, growth.near)
                if land[0] <= 0:
                    continue
            growth.build(moved)
            growth.owners.own(near, building.kind)
            return {near}
        return set()

    def spill(self, cell: Cell, ladder: Ladder, rng: np.random.Generator) -> set[Cell]:
        """Puts up the ladder's first kind in the nearest cell that will take it, picked among
        cells equally near at random by how well their land suits it; returns the cell it went up
        in, if it found one."""
        growth = self._growth
        constructions, grid = growth.constructions, growth.grid
        reached = [(near, steps) for near, steps in grid.within(cell, SPILL_STEPS).items()]
        steps = np.array([step for _, step in reached])
        rows, cols = zip(*(near for near, _ in reached), strict=True)
        flat = np.ravel_multi_index((rows, cols), grid.shape)
        land = ladder.land.weights(constructions.tile_map, flat, growth.near)
        first = ladder.kinds[0]
        suits = rng.random(steps.size) ** (1.0 / np.maximum(land, 1e-12))
        for index in np.lexsort((-suits, steps)).tolist():
            near = reached[index][0]
            new = Building(near, first)
            if (
                near != cell
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
            if TOWN_LAND.weights(constructions.tile_map, self._flat([near]), growth.near)[0] <= 0:
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
