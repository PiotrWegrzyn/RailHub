"""Where buildings stand on a map, and how they show when zoomed out.

A cell's room is split into CELL_SPACE slots in a ring round its centre, and each building stands
in the middle of the slots it takes, in the order it was built: six houses make a ring of roofs,
a farm fills half the cell. Zoomed out, buildings are coloured dots; close up they are sprites in
the pixel art (see railroads.ui.pixel).
"""

import math
import weakref
from dataclasses import dataclass
from functools import cache
from typing import Self

import numpy as np
import numpy.typing as npt
from imgui_bundle import imgui

from railroads.sim.constructions import (
    CELL_SPACE,
    Building,
    BuildingKind,
    Construction,
    Constructions,
)
from railroads.sim.grids import Cell
from railroads.ui.iso_camera import IsoCamera, Vec2

type Color = tuple[int, int, int]
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
DOTS: dict[BuildingKind, tuple[Color, float]] = {
    BuildingKind.HOUSE: ((176, 66, 50), 0.09),
    BuildingKind.TENEMENT: ((206, 150, 96), 0.12),
    BuildingKind.FLATS: ((92, 86, 96), 0.14),
    BuildingKind.FARM: ((222, 196, 96), 0.12),
    BuildingKind.MINE: ((48, 46, 50), 0.14),
    BuildingKind.FACTORY: ((235, 120, 40), 0.12),
    BuildingKind.BIG_FACTORY: ((222, 96, 30), 0.16),
    BuildingKind.MEGA_FACTORY: ((200, 70, 20), 0.2),
    BuildingKind.SKYSCRAPER: ((110, 170, 220), 0.16),
    BuildingKind.ORCHARD: ((150, 196, 70), 0.12),
    BuildingKind.DATE_PALMS: ((196, 170, 100), 0.12),
    BuildingKind.BANANAS: ((236, 206, 64), 0.12),
    BuildingKind.SUGAR: ((168, 210, 120), 0.12),
    BuildingKind.FOOD_PLANT: ((232, 220, 170), 0.14),
    BuildingKind.BAKERY: ((214, 160, 92), 0.1),
    BuildingKind.GASWORKS: ((110, 118, 112), 0.14),
    BuildingKind.RAILWAY_WORKS: ((150, 70, 52), 0.16),
    BuildingKind.STEEL_MILL: ((90, 88, 96), 0.16),
    BuildingKind.SHIPYARD: ((80, 140, 200), 0.16),
    BuildingKind.LUMBER_MILL: ((128, 88, 52), 0.14),
    BuildingKind.PORT: ((60, 110, 190), 0.16),
    BuildingKind.FISHERY: ((110, 170, 200), 0.12),
    BuildingKind.SHEEP_FARM: ((236, 232, 216), 0.12),
    BuildingKind.COTTON_FIELD: ((246, 246, 240), 0.12),
    BuildingKind.VINEYARD: ((120, 60, 140), 0.12),
    BuildingKind.COFFEE_PLANTATION: ((110, 70, 44), 0.12),
    BuildingKind.RUBBER_PLANTATION: ((66, 90, 60), 0.12),
    BuildingKind.IRON_MINE: ((150, 90, 70), 0.14),
    BuildingKind.COPPER_MINE: ((200, 126, 76), 0.14),
    BuildingKind.GOLD_MINE: ((255, 210, 50), 0.16),
    BuildingKind.DIAMOND_MINE: ((180, 234, 255), 0.16),
    BuildingKind.OIL_WELL: ((30, 28, 36), 0.14),
    BuildingKind.SALT_WORKS: ((226, 230, 238), 0.14),
    BuildingKind.TEXTILE_MILL: ((220, 122, 178), 0.14),
    BuildingKind.FURNITURE_MAKER: ((152, 106, 60), 0.14),
    BuildingKind.TOOL_WORKS: ((128, 142, 168), 0.14),
    BuildingKind.WINERY: ((142, 26, 66), 0.14),
    BuildingKind.REFINERY: ((208, 188, 66), 0.16),
    BuildingKind.CHEMICAL_WORKS: ((142, 214, 102), 0.16),
    BuildingKind.ELECTRONICS_PLANT: ((76, 214, 220), 0.16),
    BuildingKind.CAR_FACTORY: ((230, 76, 128), 0.18),
    BuildingKind.CAR_DEALER: ((250, 120, 170), 0.16),
    BuildingKind.JEWELLER: ((255, 236, 130), 0.16),
    BuildingKind.ORANGE_FARM: ((244, 142, 36), 0.12),
    BuildingKind.OUTPOST: ((116, 124, 74), 0.12),
    BuildingKind.BASE: ((96, 106, 60), 0.16),
    BuildingKind.HEADQUARTERS: ((74, 86, 44), 0.2),
    BuildingKind.GUN_FACTORY: ((110, 96, 70), 0.16),
    BuildingKind.TANK_FACTORY: ((90, 100, 62), 0.18),
    BuildingKind.PLANE_FACTORY: ((170, 186, 204), 0.18),
    BuildingKind.AIRPORT: ((200, 210, 224), 0.18),
    BuildingKind.BARREL_MAKER: ((150, 100, 56), 0.14),
    BuildingKind.BEEKEEPER: ((240, 190, 50), 0.12),
    BuildingKind.INCENSE_GROVE: ((200, 160, 210), 0.16),
    BuildingKind.CHURCH: ((232, 226, 206), 0.14),
    BuildingKind.BASILICA: ((240, 228, 190), 0.18),
    BuildingKind.CATHEDRAL: ((250, 232, 170), 0.22),
    BuildingKind.ST_PAULS: ((255, 220, 90), 0.3),
    BuildingKind.CLAY_PIT: ((190, 112, 74), 0.12),
    BuildingKind.LIMESTONE_QUARRY: ((214, 210, 192), 0.12),
    BuildingKind.SAND_PIT: ((236, 214, 140), 0.12),
    BuildingKind.POTATO_FARM: ((176, 146, 90), 0.12),
    BuildingKind.CATTLE_RANCH: ((140, 96, 60), 0.12),
    BuildingKind.PIG_FARM: ((226, 160, 156), 0.12),
    BuildingKind.CHICKEN_FARM: ((236, 224, 196), 0.12),
    BuildingKind.FLAX_FIELD: ((140, 166, 226), 0.12),
    BuildingKind.HOP_GARDEN: ((120, 186, 80), 0.12),
    BuildingKind.MACHINE_WORKS: ((104, 108, 124), 0.16),
    BuildingKind.BRICKWORKS: ((186, 80, 56), 0.14),
    BuildingKind.CEMENT_WORKS: ((170, 170, 166), 0.16),
    BuildingKind.GLASSWORKS: ((150, 220, 234), 0.14),
    BuildingKind.BREWERY: ((226, 170, 60), 0.14),
    BuildingKind.DISTILLERY: ((220, 226, 236), 0.14),
    BuildingKind.TANNERY: ((150, 96, 56), 0.14),
    BuildingKind.SHOE_FACTORY: ((96, 62, 44), 0.14),
    BuildingKind.PLASTICS_WORKS: ((234, 96, 108), 0.16),
    BuildingKind.APPLIANCE_FACTORY: ((214, 222, 232), 0.16),
    BuildingKind.PAPER_MILL: ((236, 230, 206), 0.14),
    BuildingKind.PRINTING_WORKS: ((160, 160, 154), 0.14),
    BuildingKind.SUGAR_REFINERY: ((246, 228, 232), 0.14),
    BuildingKind.PORCELAIN_WORKS: ((96, 136, 214), 0.14),
    BuildingKind.MEAT_FACTORY: ((176, 70, 64), 0.14),
    BuildingKind.POWER_PLANT: ((120, 120, 132), 0.18),
    BuildingKind.SPACE_CENTER: ((236, 238, 244), 0.26),
    BuildingKind.PEAR_HQ: ((200, 220, 230), 0.28),
    BuildingKind.ROYAL_PALACE: ((240, 214, 150), 0.28),
    BuildingKind.SALT_CATHEDRAL: ((244, 244, 248), 0.24),
    BuildingKind.SPA_RESORT: ((170, 214, 220), 0.24),
    BuildingKind.WORLDS_FAIR: ((190, 220, 236), 0.28),
    BuildingKind.FILM_STUDIO: ((150, 150, 170), 0.24),
    BuildingKind.AMUSEMENT_PARK: ((240, 120, 160), 0.24),
    BuildingKind.OLYMPIC_STADIUM: ((220, 90, 70), 0.3),
    BuildingKind.TAVERN: ((196, 130, 56), 0.1),
    BuildingKind.DEPARTMENT_STORE: ((232, 200, 110), 0.16),
    BuildingKind.HOTEL: ((200, 70, 90), 0.16),
}


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


def draw_building_dots(
    draw_list: imgui.ImDrawList,
    camera: IsoCamera,
    center: Vec2,
    clip: tuple[Vec2, Vec2],
    spots: BuildingSpots,
) -> None:
    """Every building on screen as a dot in its kind's colour."""
    if not spots.kinds:
        return
    unit = camera.cell_width / camera.cell_units
    screen_x, screen_y = camera.to_screen(spots.x, spots.y, center)
    (left, top), (right, bottom) = clip
    shown = np.flatnonzero(
        (screen_x > left - unit)
        & (screen_x < right + unit)
        & (screen_y > top - unit)
        & (screen_y < bottom + unit)
    )
    xs, ys = np.asarray(screen_x).tolist(), np.asarray(screen_y).tolist()
    for i in shown.tolist():
        color, radius = DOTS[spots.kinds[i]]
        half = max(1.0, unit * radius)
        draw_list.add_rect_filled(
            (xs[i] - half, ys[i] - half), (xs[i] + half, ys[i] + half), _color(color)
        )


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


@cache
def _color(rgb: Color, alpha: float = 1.0) -> int:
    r, g, b = rgb
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(r / 255, g / 255, b / 255, alpha))
