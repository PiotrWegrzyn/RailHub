"""Where buildings stand on a map.

A cell's room is split into CELL_SPACE slots in a ring round its centre, and each building stands
in the middle of the slots it takes, in the order it was built: six houses make a ring of roofs,
a farm fills half the cell. Zoomed out, buildings show as coloured dots there
(railroads.ui.buildings); close up they are sprites in the pixel art (see railroads.ui.pixel), which
is also painted where imgui is not loaded, as in web workers, so nothing here needs it.
"""

import math
import weakref
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import (
    CELL_SPACE,
    Building,
    BuildingKind,
    Construction,
    Constructions,
)
from railroads.sim.grids import Cell

FloatArray = npt.NDArray[np.float64]

SLOT_RADIUS = 0.3
# How far from its cell's middle a farmstead stands at the back of the field filling its cell,
# in cells, and how far either side of straight back it may stand, in radians.
FARMSTEAD_OUT = 0.42
FARMSTEAD_SWING = 0.7
# How far from the middle of a track through its cell a building stands at least, in cells, and
# how much further for double track.
TRACK_CLEARANCE = 0.25
DOUBLE_CLEARANCE = 0.11


@dataclass(frozen=True)
class BuildingSpots:
    """Where each building stands, in cell units, with a number that picks its look."""

    x: FloatArray
    y: FloatArray
    kinds: list[BuildingKind]
    variants: list[int]

    @classmethod
    def of(
        cls, constructions: Constructions, at_back: frozenset[BuildingKind] = frozenset()
    ) -> Self:
        """Where each building stands: buildings of the kinds `at_back`, the farmsteads of
        fields filling their cell, stand at the back edge of their field, so it lies in front
        of them. A cell whose constructions are the same as last time keeps its spots."""
        known = _known(constructions, ("spots", at_back))
        x: list[float] = []
        y: list[float] = []
        kinds: list[BuildingKind] = []
        variants: list[int] = []
        for cell, built in constructions.cells():
            entry = known.get(cell)
            if entry is None or entry[0] != built:
                entry = (list(built), _cell_spots(constructions, cell, built, at_back))
                known[cell] = entry
            cell_x, cell_y, cell_kinds, cell_variants = entry[1]
            x += cell_x
            y += cell_y
            kinds += cell_kinds
            variants += cell_variants
        return cls(np.array(x), np.array(y), kinds, variants)


def _cell_spots(
    constructions: Constructions,
    cell: Cell,
    built: list[Construction],
    at_back: frozenset[BuildingKind],
) -> tuple[list[float], list[float], list[BuildingKind], list[int]]:
    grid = constructions.tile_map.grid
    row, col = cell
    center_x, center_y = grid.center(row, col)
    x, y, kinds, variants = [], [], [], []
    first = 0
    for one in built:
        start, first = first, first + one.space
        if not isinstance(one, Building):
            continue
        if one.kind in at_back:
            turn = -math.pi / 2 + FARMSTEAD_SWING * math.sin(_cell_turn(row, col) * 37)
            x.append(center_x + FARMSTEAD_OUT * math.cos(turn))
            y.append(center_y + FARMSTEAD_OUT * math.sin(turn))
        else:
            slots = np.arange(start, start + one.space) + 0.5
            angles = _cell_turn(row, col) + 2 * math.pi * slots / CELL_SPACE
            spot = (
                center_x + SLOT_RADIUS * float(np.cos(angles).mean()),
                center_y + SLOT_RADIUS * float(np.sin(angles).mean()),
            )
            spot = _clear_of_track(constructions, cell, spot)
            x.append(spot[0])
            y.append(spot[1])
        kinds.append(one.kind)
        variants.append((row * 7 + col * 13 + start) % 16)
    return x, y, kinds, variants


# What was worked out per cell, for each Constructions and question, with a copy of the cell's
# constructions it was worked out from. The copy must stay a list: compared with the cell's own
# list, a tuple would never be equal and nothing would be kept.
_KNOWN: weakref.WeakKeyDictionary[Constructions, dict[object, dict[Cell, tuple]]] = (
    weakref.WeakKeyDictionary()
)


def _known(constructions: Constructions, question: object) -> dict[Cell, tuple]:
    return _KNOWN.setdefault(constructions, {}).setdefault(question, {})


@dataclass(frozen=True)
class Plot:
    """The wedge of a cell a field covers, as wide as its share of the cell's space: `facing`
    points from the cell's centre into the middle of it, and `spread` is the cosine of the angle
    from there to either edge, so a field of half the space covers half the cell."""

    cell: Cell
    facing: tuple[float, float]
    spread: float = 0.0

    def covers(self, across: FloatArray, along: FloatArray) -> npt.NDArray[np.bool_]:
        """Which points, given by their offsets from the cell's centre, lie in the plot."""
        toward = self.facing[0] * across + self.facing[1] * along
        return toward >= self.spread * np.hypot(across, along)


def plots_of(
    constructions: Constructions, kinds: tuple[BuildingKind, ...]
) -> dict[BuildingKind, list[Plot]]:
    """Each field of these kinds, on the side of its cell that its slots take up. A cell whose
    constructions are the same as last time keeps its plots."""
    known = _known(constructions, ("plots", kinds))
    plots: dict[BuildingKind, list[Plot]] = {kind: [] for kind in kinds}
    for cell, built in constructions.cells():
        entry = known.get(cell)
        if entry is None or entry[0] != built:
            entry = (list(built), _cell_plots(cell, built, plots))
            known[cell] = entry
        for kind, plot in entry[1]:
            plots[kind].append(plot)
    return plots


def _cell_plots(
    cell: Cell, built: list[Construction], kinds: dict[BuildingKind, list[Plot]]
) -> list[tuple[BuildingKind, Plot]]:
    found = []
    first = 0
    for one in built:
        start, first = first, first + one.space
        if isinstance(one, Building) and one.kind in kinds:
            middle = (start + one.space / 2) / CELL_SPACE
            angle = _cell_turn(*cell) + 2 * math.pi * middle
            spread = math.cos(math.pi * one.space / CELL_SPACE)
            facing = (math.cos(angle), math.sin(angle))
            found.append((one.kind, Plot(cell, facing, spread)))
    return found


def _clear_of_track(
    constructions: Constructions, cell: Cell, spot: tuple[float, float]
) -> tuple[float, float]:
    """The spot moved, if it must be, straight away from any track through the cell until it
    stands TRACK_CLEARANCE from the track's middle, so the building stands beside the line
    rather than on it."""
    pieces = constructions.track_at(cell)
    if not pieces:
        return spot
    grid = constructions.tile_map.grid
    steps = grid.track_steps() / 2
    cx, cy = (float(v) for v in grid.center(*cell))
    x, y = spot
    for _ in range(3):
        nearest, away = math.inf, (0.0, 0.0)
        for piece in pieces:
            clearance = TRACK_CLEARANCE + (DOUBLE_CLEARANCE if int(piece.width) > 1 else 0.0)
            for point in piece.ends:
                ex, ey = cx + float(steps[point][0]), cy + float(steps[point][1])
                dx, dy = ex - cx, ey - cy
                length = math.hypot(dx, dy) or 1.0
                t = max(0.0, min(1.0, ((x - cx) * dx + (y - cy) * dy) / length**2))
                px, py = cx + dx * t, cy + dy * t
                gap = math.hypot(x - px, y - py)
                if gap - clearance < nearest:
                    nearest = gap - clearance
                    if gap > 1e-6:
                        away = ((x - px) / gap, (y - py) / gap)
                    else:
                        away = (-dy / length, dx / length)
        if nearest >= 0:
            break
        x, y = x - away[0] * nearest, y - away[1] * nearest
    return x, y


def _cell_turn(row: int, col: int) -> float:
    """A turn of up to one slot that differs from cell to cell, so towns do not look ruled."""
    return (row * 0.618034 + col * 0.414214) % 1.0 * 2 * math.pi / CELL_SPACE
