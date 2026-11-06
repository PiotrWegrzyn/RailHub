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
    BuildingKind.FARM: ((222, 196, 96), 0.12),
    BuildingKind.MINE: ((48, 46, 50), 0.14),
    BuildingKind.FACTORY: ((235, 120, 40), 0.16),
    BuildingKind.LUMBER_MILL: ((128, 88, 52), 0.14),
    BuildingKind.PORT: ((60, 110, 190), 0.16),
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
