"""Draws the buildings on a map.

A cell's room is split into CELL_SPACE slots in a ring round its centre, and each building stands
in the middle of the slots it takes, in the order it was built: six houses make a ring of roofs,
a farm fills half the cell. Close up, buildings are little pictures standing on the ground, with
fields and quays laid flat on it; further out they are coloured dots.
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
# Cell width in pixels from which buildings are drawn as pictures rather than dots.
PICTURES_FROM = 24
# Picture widths, in cell units.
SIZES = {
    BuildingKind.HOUSE: 0.28,
    BuildingKind.FARM: 0.26,
    BuildingKind.MINE: 0.56,
    BuildingKind.FACTORY: 0.62,
    BuildingKind.LUMBER_MILL: 0.56,
    BuildingKind.PORT: 0.6,
}
DOTS: dict[BuildingKind, tuple[Color, float]] = {
    BuildingKind.HOUSE: ((176, 66, 50), 0.09),
    BuildingKind.FARM: ((222, 196, 96), 0.12),
    BuildingKind.MINE: ((48, 46, 50), 0.14),
    BuildingKind.FACTORY: ((235, 120, 40), 0.16),
    BuildingKind.LUMBER_MILL: ((128, 88, 52), 0.14),
    BuildingKind.PORT: ((60, 110, 190), 0.16),
}
WALLS: list[Color] = [(232, 222, 200), (245, 240, 230), (210, 196, 170), (220, 206, 186)]
ROOFS: list[Color] = [(168, 64, 48), (140, 52, 40), (96, 92, 100), (180, 96, 60)]
CROPS: list[Color] = [(222, 196, 96), (186, 146, 90), (232, 214, 132)]
OUTLINE: Color = (50, 40, 35)
SHADOW = ((0, 0, 0), 0.22)


@dataclass(frozen=True)
class BuildingSpots:
    """Where each building stands, in cell units, with how its cell is turned and a number
    that picks its colours."""

    x: FloatArray
    y: FloatArray
    kinds: list[BuildingKind]
    turns: list[float]
    variants: list[int]

    @classmethod
    def of(cls, constructions: Constructions) -> Self:
        grid = constructions.tile_map.grid
        used: dict[Cell, int] = {}
        x, y, kinds, turns, variants = [], [], [], [], []
        for built in constructions:
            first = used.get(built.cell, 0)
            used[built.cell] = first + built.space
            if not isinstance(built, Building):
                continue
            row, col = built.cell
            turn = _cell_turn(row, col)
            slots = np.arange(first, first + built.space) + 0.5
            angles = turn + 2 * math.pi * slots / CELL_SPACE
            center_x, center_y = grid.center(row, col)
            x.append(center_x + SLOT_RADIUS * float(np.cos(angles).mean()))
            y.append(center_y + SLOT_RADIUS * float(np.sin(angles).mean()))
            kinds.append(built.kind)
            turns.append(turn)
            variants.append((row * 7 + col * 13 + first) % 16)
        return cls(np.array(x), np.array(y), kinds, turns, variants)


def draw_buildings(
    draw_list: imgui.ImDrawList,
    camera: IsoCamera,
    center: Vec2,
    clip: tuple[Vec2, Vec2],
    spots: BuildingSpots,
) -> None:
    if not spots.kinds:
        return
    unit = camera.cell_width / camera.cell_units
    screen_x, screen_y = camera.to_screen(spots.x, spots.y, center)
    (left, top), (right, bottom) = clip
    shown = np.flatnonzero(
        (screen_x > left - unit)
        & (screen_x < right + unit)
        & (screen_y > top - unit)
        & (screen_y < bottom + 2 * unit)
    )
    xs, ys = np.asarray(screen_x).tolist(), np.asarray(screen_y).tolist()
    if camera.cell_width < PICTURES_FROM:
        for i in shown.tolist():
            color, radius = DOTS[spots.kinds[i]]
            half = max(1.0, unit * radius)
            draw_list.add_rect_filled(
                (xs[i] - half, ys[i] - half), (xs[i] + half, ys[i] + half), _color(color)
            )
        return
    order = shown[np.argsort(np.asarray(screen_y)[shown], kind="stable")].tolist()
    for i in order:
        kind = spots.kinds[i]
        if kind == BuildingKind.FARM:
            _draw_field(draw_list, camera, center, spots, i)
        elif kind == BuildingKind.PORT:
            _draw_quay(draw_list, camera, center, spots, i)
    for i in order:
        kind, size, variant = spots.kinds[i], SIZES[spots.kinds[i]] * unit, spots.variants[i]
        x, y = xs[i], ys[i]
        match kind:
            case BuildingKind.HOUSE:
                wall, roof = WALLS[variant % len(WALLS)], ROOFS[variant // 4 % len(ROOFS)]
                _draw_house(draw_list, x, y, size, wall, roof)
            case BuildingKind.FARM:
                _draw_house(draw_list, x - size * 0.4, y, size, (160, 58, 44), (90, 84, 80))
                _draw_silo(draw_list, x + size * 0.45, y, size)
            case BuildingKind.MINE:
                _draw_mine(draw_list, x, y, size)
            case BuildingKind.FACTORY:
                _draw_factory(draw_list, x, y, size)
            case BuildingKind.LUMBER_MILL:
                _draw_lumber_mill(draw_list, x, y, size)
            case BuildingKind.PORT:
                _draw_port(draw_list, x, y, size)


def _cell_turn(row: int, col: int) -> float:
    """A turn of up to one slot that differs from cell to cell, so towns do not look ruled."""
    return (row * 0.618034 + col * 0.414214) % 1.0 * 2 * math.pi / CELL_SPACE


@cache
def _color(rgb: Color, alpha: float = 1.0) -> int:
    r, g, b = rgb
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(r / 255, g / 255, b / 255, alpha))


def _shade(rgb: Color, factor: float) -> Color:
    r, g, b = rgb
    return min(255, int(r * factor)), min(255, int(g * factor)), min(255, int(b * factor))


def _ground_patch(
    camera: IsoCamera, center: Vec2, spots: BuildingSpots, i: int, half_x: float, half_y: float
) -> list[tuple[float, float]]:
    """The corners of a rectangle lying on the ground round a building, turned with its cell."""
    cos, sin = math.cos(spots.turns[i]), math.sin(spots.turns[i])
    x, y = float(spots.x[i]), float(spots.y[i])
    corners = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
    points = []
    for dx, dy in corners:
        sx, sy = camera.to_screen(x + dx * cos - dy * sin, y + dx * sin + dy * cos, center)
        points.append((float(sx), float(sy)))
    return points


def _draw_field(
    draw_list: imgui.ImDrawList, camera: IsoCamera, center: Vec2, spots: BuildingSpots, i: int
) -> None:
    crop = CROPS[spots.variants[i] % len(CROPS)]
    a, b, c, d = _ground_patch(camera, center, spots, i, 0.3, 0.3)
    draw_list.add_quad_filled(a, b, c, d, _color(crop))
    furrow = _color(_shade(crop, 0.82))
    for step in (0.2, 0.4, 0.6, 0.8):
        start = (a[0] + (d[0] - a[0]) * step, a[1] + (d[1] - a[1]) * step)
        end = (b[0] + (c[0] - b[0]) * step, b[1] + (c[1] - b[1]) * step)
        draw_list.add_line(start, end, furrow, 1.0)


def _draw_quay(
    draw_list: imgui.ImDrawList, camera: IsoCamera, center: Vec2, spots: BuildingSpots, i: int
) -> None:
    a, b, c, d = _ground_patch(camera, center, spots, i, 0.3, 0.16)
    draw_list.add_quad_filled(a, b, c, d, _color((150, 148, 140)))
    draw_list.add_quad(a, b, c, d, _color((96, 94, 90)), 1.0)


def _draw_shadow(draw_list: imgui.ImDrawList, x: float, y: float, size: float) -> None:
    draw_list.add_ellipse_filled(
        (x + size * 0.15, y + size * 0.04), (size * 0.6, size * 0.16), _color(*SHADOW)
    )


def _draw_house(
    draw_list: imgui.ImDrawList, x: float, y: float, size: float, wall: Color, roof: Color
) -> None:
    outline = _color(OUTLINE)
    half, height, depth = size / 2, size * 0.62, size * 0.28
    draw_list.add_quad_filled(
        (x - half, y),
        (x + half + depth, y - depth * 0.5),
        (x + half + depth * 2.2, y + depth * 0.4),
        (x - half + depth * 1.2, y + depth * 0.9),
        _color(*SHADOW),
    )
    draw_list.add_quad_filled(
        (x + half, y),
        (x + half + depth, y - depth * 0.5),
        (x + half + depth, y - height - depth * 0.5),
        (x + half, y - height),
        _color(_shade(wall, 0.72)),
    )
    draw_list.add_rect_filled((x - half, y - height), (x + half, y), _color(wall))
    draw_list.add_rect((x - half, y - height), (x + half, y), outline)
    peak = (x, y - height - size * 0.45)
    draw_list.add_quad_filled(
        (x + half, y - height),
        (x + half + depth, y - height - depth * 0.5),
        (peak[0] + depth, peak[1] - depth * 0.5),
        peak,
        _color(_shade(roof, 0.7)),
    )
    eaves = ((x - half - size * 0.08, y - height), (x + half + size * 0.08, y - height), peak)
    draw_list.add_triangle_filled(*eaves, _color(roof))
    draw_list.add_triangle(*eaves, outline)
    window = size * 0.16
    draw_list.add_rect_filled(
        (x - window / 2, y - height * 0.62),
        (x + window / 2, y - height * 0.62 + window),
        _color((80, 110, 140)),
    )


def _draw_silo(draw_list: imgui.ImDrawList, x: float, y: float, size: float) -> None:
    half, height = size * 0.18, size * 1.05
    draw_list.add_rect_filled((x - half, y - height), (x + half, y), _color((206, 204, 196)))
    draw_list.add_rect_filled((x, y - height), (x + half, y), _color((172, 170, 164)))
    draw_list.add_circle_filled((x, y - height), half, _color((150, 148, 144)))
    draw_list.add_rect((x - half, y - height), (x + half, y), _color(OUTLINE))


def _draw_mine(draw_list: imgui.ImDrawList, x: float, y: float, size: float) -> None:
    _draw_shadow(draw_list, x, y, size)
    peak = (x - size * 0.15, y - size * 0.42)
    draw_list.add_triangle_filled(
        (x - size * 0.55, y), peak, (x + size * 0.2, y), _color((78, 72, 70))
    )
    draw_list.add_triangle_filled(
        (x - size * 0.55, y), peak, (x - size * 0.2, y), _color((110, 102, 96))
    )
    draw_list.add_rect_filled(
        (x + size * 0.1, y - size * 0.28), (x + size * 0.5, y), _color((130, 124, 116))
    )
    draw_list.add_rect((x + size * 0.1, y - size * 0.28), (x + size * 0.5, y), _color(OUTLINE))
    top = (x + size * 0.3, y - size * 0.9)
    timber, thickness = _color((70, 50, 38)), max(1.0, size * 0.06)
    draw_list.add_line((x + size * 0.05, y - size * 0.28), top, timber, thickness)
    draw_list.add_line((x + size * 0.55, y - size * 0.28), top, timber, thickness)
    draw_list.add_line(
        (x + size * 0.15, y - size * 0.55), (x + size * 0.45, y - size * 0.55), timber, thickness
    )
    draw_list.add_circle(top, size * 0.1, _color((40, 36, 34)), 0, thickness)


def _draw_factory(draw_list: imgui.ImDrawList, x: float, y: float, size: float) -> None:
    outline = _color((40, 30, 28))
    brick, roof = (150, 70, 55), (80, 80, 86)
    half, height, depth = size / 2, size * 0.45, size * 0.2
    draw_list.add_quad_filled(
        (x - half, y),
        (x + half + depth, y - depth * 0.5),
        (x + half + depth * 2.4, y + depth * 0.5),
        (x - half + depth * 1.4, y + depth),
        _color(*SHADOW),
    )
    draw_list.add_quad_filled(
        (x + half, y),
        (x + half + depth, y - depth * 0.5),
        (x + half + depth, y - height - depth * 0.5),
        (x + half, y - height),
        _color(_shade(brick, 0.7)),
    )
    draw_list.add_rect_filled((x - half, y - height), (x + half, y), _color(brick))
    for tooth in range(3):
        tooth_left = x - half + tooth * size / 3
        draw_list.add_triangle_filled(
            (tooth_left, y - height),
            (tooth_left + size / 3, y - height),
            (tooth_left + size / 3, y - height - size * 0.22),
            _color(roof),
        )
    draw_list.add_rect((x - half, y - height), (x + half, y), outline)
    for window in range(4):
        left = x - half + size * (0.12 + window * 0.22)
        draw_list.add_rect_filled(
            (left, y - height * 0.6), (left + size * 0.1, y - height * 0.3), _color((60, 52, 50))
        )
    chimney = x + size * 0.3
    draw_list.add_rect_filled(
        (chimney - size * 0.055, y - size * 1.12),
        (chimney + size * 0.055, y - height),
        _color((92, 60, 52)),
    )
    for puff, age in enumerate((0.15, 0.45, 0.75)):
        rise = age * size * 0.9
        draw_list.add_circle_filled(
            (chimney + rise * 0.45 + puff, y - size * 1.18 - rise),
            size * 0.07 + age * size * 0.13,
            _color((130, 130, 136), 0.6 * (1 - age)),
        )


def _draw_lumber_mill(draw_list: imgui.ImDrawList, x: float, y: float, size: float) -> None:
    _draw_shadow(draw_list, x, y, size)
    wall, roof = (156, 112, 70), (86, 96, 70)
    left, right, height, depth = x - size * 0.5, x + size * 0.1, size * 0.32, size * 0.2
    draw_list.add_quad_filled(
        (right, y),
        (right + depth, y - depth * 0.5),
        (right + depth, y - height - depth * 0.5),
        (right, y - height),
        _color(_shade(wall, 0.72)),
    )
    draw_list.add_rect_filled((left, y - height), (right, y), _color(wall))
    draw_list.add_quad_filled(
        (left - size * 0.04, y - height),
        (right + size * 0.04, y - height),
        (right + depth, y - height - depth * 0.7),
        (left + depth * 0.6, y - height - depth * 0.7),
        _color(roof),
    )
    draw_list.add_rect((left, y - height), (right, y), _color(OUTLINE))
    radius = size * 0.07
    for row, count in enumerate((3, 2, 1)):
        for log in range(count):
            spot = (x + size * (0.24 + 0.14 * log + 0.07 * row), y - radius - row * radius * 1.7)
            draw_list.add_circle_filled(spot, radius, _color((190, 146, 92)))
            draw_list.add_circle(spot, radius, _color((118, 80, 48)))


def _draw_port(draw_list: imgui.ImDrawList, x: float, y: float, size: float) -> None:
    wall, roof = (176, 164, 140), (74, 92, 120)
    left, right, height, depth = x - size * 0.45, x + size * 0.05, size * 0.3, size * 0.2
    draw_list.add_quad_filled(
        (right, y),
        (right + depth, y - depth * 0.5),
        (right + depth, y - height - depth * 0.5),
        (right, y - height),
        _color(_shade(wall, 0.72)),
    )
    draw_list.add_rect_filled((left, y - height), (right, y), _color(wall))
    draw_list.add_quad_filled(
        (left, y - height),
        (right, y - height),
        (right + depth, y - height - depth * 0.5),
        (left + depth, y - height - depth * 0.5),
        _color(roof),
    )
    draw_list.add_rect((left, y - height), (right, y), _color(OUTLINE))
    crane, thickness = _color((226, 168, 48)), max(1.0, size * 0.06)
    mast = x + size * 0.3
    draw_list.add_line((mast, y), (mast, y - size * 0.95), crane, thickness)
    draw_list.add_line(
        (x + size * 0.1, y - size * 0.9), (x + size * 0.68, y - size * 0.9), crane, thickness
    )
    hook = x + size * 0.62
    draw_list.add_line((hook, y - size * 0.9), (hook, y - size * 0.55), _color(OUTLINE), 1.0)
    draw_list.add_rect_filled(
        (hook - size * 0.07, y - size * 0.55),
        (hook + size * 0.07, y - size * 0.45),
        _color((60, 120, 170)),
    )
