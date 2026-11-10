"""Where buildings stand on a map, and how they show when zoomed out.

A cell's room is split into CELL_SPACE slots in a ring round its centre, and each building stands
in the middle of the slots it takes, in the order it was built: six houses make a ring of roofs,
a farm fills half the cell. Zoomed out, buildings are coloured dots; close up they are sprites in
the pixel art (see railroads.ui.pixel).
"""

import math
from dataclasses import dataclass
from functools import cache
from typing import Self

import numpy as np
import numpy.typing as npt
from imgui_bundle import imgui

from railroads.sim.constructions import CELL_SPACE, Building, BuildingKind, Constructions
from railroads.sim.grids import Cell
from railroads.ui.iso_camera import IsoCamera, Vec2

type Color = tuple[int, int, int]
FloatArray = npt.NDArray[np.float64]

SLOT_RADIUS = 0.3
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
    def of(cls, constructions: Constructions) -> Self:
        grid = constructions.tile_map.grid
        used: dict[Cell, int] = {}
        x, y, kinds, variants = [], [], [], []
        for built in constructions:
            first = used.get(built.cell, 0)
            used[built.cell] = first + built.space
            if not isinstance(built, Building):
                continue
            row, col = built.cell
            slots = np.arange(first, first + built.space) + 0.5
            angles = _cell_turn(row, col) + 2 * math.pi * slots / CELL_SPACE
            center_x, center_y = grid.center(row, col)
            x.append(center_x + SLOT_RADIUS * float(np.cos(angles).mean()))
            y.append(center_y + SLOT_RADIUS * float(np.sin(angles).mean()))
            kinds.append(built.kind)
            variants.append((row * 7 + col * 13 + first) % 16)
        return cls(np.array(x), np.array(y), kinds, variants)


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
    """Each field of these kinds, on the side of its cell that its slots take up."""
    plots: dict[BuildingKind, list[Plot]] = {kind: [] for kind in kinds}
    used: dict[Cell, int] = {}
    for built in constructions:
        first = used.get(built.cell, 0)
        used[built.cell] = first + built.space
        if isinstance(built, Building) and built.kind in plots:
            middle = (first + built.space / 2) / CELL_SPACE
            angle = _cell_turn(*built.cell) + 2 * math.pi * middle
            spread = math.cos(math.pi * built.space / CELL_SPACE)
            facing = (math.cos(angle), math.sin(angle))
            plots[built.kind].append(Plot(built.cell, facing, spread))
    return plots


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


def _cell_turn(row: int, col: int) -> float:
    """A turn of up to one slot that differs from cell to cell, so towns do not look ruled."""
    return (row * 0.618034 + col * 0.414214) % 1.0 * 2 * math.pi / CELL_SPACE


@cache
def _color(rgb: Color, alpha: float = 1.0) -> int:
    r, g, b = rgb
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(r / 255, g / 255, b / 255, alpha))
