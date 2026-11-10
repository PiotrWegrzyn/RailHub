"""Trains as pixel art. Each engine and car is a small model of boxes and cylinders, drawn for
HEADINGS directions with the map's own projection: faces shaded by how much they face the light
from the top left, painted back to front, then outlined in a dark line like the rest of the art.

Trains are drawn at DETAIL pixels for every map art pixel, so they keep their windows, wheels and
trim even though a car is less than a cell long.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cache

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from railroads.sim.goods import Good
from railroads.ui.iso_camera import TILT
from railroads.ui.pixel.art import Sprite

type RGB = tuple[int, int, int]
type Vec3 = tuple[float, float, float]

HEADINGS = 24
DETAIL = 2
# How many map art pixels tall a cell-wide height is, for a view from 30 degrees above.
HEIGHT_PER_WIDTH = math.sqrt(1 - TILT**2)
OUTLINE: RGB = (22, 18, 18)
# Shades a face can take, from facing away from the light to facing it.
SHADES = (0.7, 0.8, 0.9, 1.0, 1.08)
CYLINDER_SIDES = 10

BLACK_IRON: RGB = (48, 48, 54)
SMOKEBOX: RGB = (72, 72, 80)
CAB_RED: RGB = (168, 42, 34)
ROOF_RED: RGB = (104, 28, 24)
BRASS: RGB = (206, 162, 70)
WHEEL_RED: RGB = (150, 36, 30)
FRAME: RGB = (34, 32, 34)
WINDOW: RGB = (44, 52, 68)
COAL: RGB = (28, 28, 30)
LOG: RGB = (148, 98, 56)
LOG_END: RGB = (200, 160, 110)
STEEL: RGB = (150, 166, 186)
DECK: RGB = (104, 74, 48)
BOXCAR: RGB = (138, 56, 42)
REEFER: RGB = (214, 194, 140)
ROOF_GREY: RGB = (110, 108, 112)
HOPPER: RGB = (70, 70, 76)
COACH: RGB = (46, 92, 64)
COACH_BAND: RGB = (214, 200, 160)
LIT_WINDOW: RGB = (236, 206, 120)
MAIL_VAN: RGB = (92, 54, 112)
GRAIN: RGB = (214, 176, 92)
GRAIN_HOPPER: RGB = (150, 112, 64)
APPLE: RGB = (196, 40, 40)
APPLE_LIGHT: RGB = (226, 92, 60)
BANANA_VAN: RGB = (228, 204, 72)
BREAD_VAN: RGB = (176, 120, 70)
DATE: RGB = (110, 62, 36)
SUGAR_HOPPER: RGB = (226, 224, 214)
CANS_STRIPE: RGB = (236, 140, 50)
# Closed vans, by good: body and stripe.
VANS: dict[Good, tuple[RGB, RGB]] = {
    Good.CLOTHES: ((72, 90, 132), (232, 232, 232)),
    Good.FURNITURE: ((128, 92, 58), (200, 160, 110)),
    Good.TOOLS: ((86, 92, 100), (220, 180, 70)),
    Good.WINE: ((110, 40, 70), (220, 190, 120)),
    Good.ELECTRONICS: ((48, 60, 84), (90, 200, 220)),
    Good.COFFEE: ((84, 56, 40), (200, 160, 110)),
    Good.CHEMICALS: ((196, 192, 118), (90, 90, 60)),
    Good.FISH: ((214, 194, 140), (90, 130, 170)),
    Good.GOLD: ((96, 98, 106), (232, 190, 70)),
    Good.DIAMONDS: ((96, 98, 106), (190, 238, 255)),
    Good.GUNS: ((84, 92, 64), (190, 56, 48)),
    Good.HONEY: ((200, 150, 50), (250, 220, 120)),
    Good.WAX: ((220, 208, 150), (150, 120, 70)),
    Good.INCENSE: ((96, 70, 110), (230, 200, 130)),
    Good.BEER: ((150, 96, 40), (240, 190, 70)),
    Good.VODKA: ((210, 214, 220), (60, 90, 170)),
    Good.LEATHER: ((120, 80, 48), (190, 140, 90)),
    Good.SHOES: ((82, 52, 38), (200, 160, 110)),
    Good.CATTLE: ((132, 88, 56), (220, 200, 170)),
    Good.SHEEP: ((120, 96, 70), (238, 234, 222)),
    Good.MEAT: ((214, 194, 140), (190, 60, 56)),
    Good.GLASS: ((96, 130, 150), (190, 236, 244)),
    Good.PLASTICS: ((220, 220, 224), (220, 70, 80)),
    Good.APPLIANCES: ((226, 228, 232), (90, 120, 170)),
    Good.PAPER: ((214, 204, 176), (120, 110, 90)),
    Good.NEWSPAPERS: ((100, 100, 104), (230, 228, 220)),
    Good.REFINED_SUGAR: ((236, 230, 232), (220, 120, 150)),
    Good.PORCELAIN: ((230, 232, 240), (60, 90, 190)),
}
# Hoppers, by good: body and load.
HOPPERS: dict[Good, tuple[RGB, RGB]] = {
    Good.IRON: ((96, 72, 62), (146, 88, 68)),
    Good.COPPER: ((96, 84, 66), (194, 122, 72)),
    Good.SALT: ((176, 176, 172), (238, 238, 232)),
    Good.RUBBER: ((74, 74, 72), (42, 42, 44)),
    Good.CLAY: ((112, 80, 62), (194, 116, 76)),
    Good.LIMESTONE: ((120, 118, 112), (226, 222, 204)),
    Good.SAND: ((124, 110, 80), (240, 220, 150)),
    Good.CEMENT: ((150, 150, 148), (196, 196, 192)),
}
# Low wagons heaped with a load, and tank cars.
HEAPS: dict[Good, RGB] = {
    Good.WOOL: (235, 232, 220),
    Good.COTTON: (246, 246, 240),
    Good.GRAPES: (92, 48, 110),
    Good.BARRELS: (128, 84, 48),
    Good.POTATOES: (180, 146, 94),
    Good.FLAX: (196, 180, 120),
    Good.HOPS: (130, 190, 84),
    Good.BRICKS: (180, 82, 60),
    Good.MACHINERY: (96, 100, 112),
}
TANKS: dict[Good, RGB] = {
    Good.OIL: (34, 32, 38),
    Good.FUEL: (190, 170, 60),
    Good.GAS: (200, 206, 214),
}
MOGUL_GREEN: RGB = (38, 86, 56)
MOGUL_ROOF: RGB = (24, 54, 36)
MAROON: RGB = (118, 30, 40)
MAROON_ROOF: RGB = (70, 20, 26)
BOXCAB_BROWN: RGB = (122, 74, 42)
EXPRESS_BLUE: RGB = (36, 78, 168)
EXPRESS_WHITE: RGB = (230, 232, 236)
PANTOGRAPH: RGB = (60, 60, 66)
AUTO_BODY: RGB = (176, 44, 60)
TANK_HULL: RGB = (88, 100, 60)
PLANE_BODY: RGB = (188, 196, 206)
ORANGE: RGB = (240, 140, 36)

# Sizes in cells: how far apart the engine and cars sit, and their half lengths.
ENGINE_HALF = 0.36
TENDER_HALF = 0.22
CAR_HALF = 0.24
MOGUL_HALF = 0.42
MOGUL_TENDER_HALF = 0.26
PACIFIC_HALF = 0.46
PACIFIC_TENDER_HALF = 0.3
BOXCAB_HALF = 0.36
EXPRESS_HALF = 0.42
COUPLING = 0.04


@dataclass(frozen=True)
class Face:
    corners: list[Vec3]
    color: RGB
    normal: Vec3


class Model:
    """A model built of faces, in cells: u along the track (forward positive), v across it (to
    the left), z up from the rails."""

    def __init__(self) -> None:
        self.faces: list[Face] = []

    def box(
        self, u: tuple[float, float], v: tuple[float, float], z: tuple[float, float], color: RGB
    ) -> None:
        (u0, u1), (v0, v1), (z0, z1) = u, v, z
        self.faces += [
            Face([(u0, v0, z1), (u1, v0, z1), (u1, v1, z1), (u0, v1, z1)], color, (0, 0, 1)),
            Face([(u1, v0, z0), (u1, v1, z0), (u1, v1, z1), (u1, v0, z1)], color, (1, 0, 0)),
            Face([(u0, v0, z0), (u0, v0, z1), (u0, v1, z1), (u0, v1, z0)], color, (-1, 0, 0)),
            Face([(u0, v1, z0), (u0, v1, z1), (u1, v1, z1), (u1, v1, z0)], color, (0, 1, 0)),
            Face([(u0, v0, z0), (u1, v0, z0), (u1, v0, z1), (u0, v0, z1)], color, (0, -1, 0)),
        ]

    def panel(self, u: tuple[float, float], z: tuple[float, float], v: float, color: RGB) -> None:
        """A flat patch on both sides of the model, `v` out from its middle: windows, doors."""
        (u0, u1), (z0, z1) = u, z
        self.faces.append(
            Face([(u0, v, z0), (u1, v, z0), (u1, v, z1), (u0, v, z1)], color, (0, 1, 0))
        )
        self.faces.append(
            Face([(u0, -v, z0), (u0, -v, z1), (u1, -v, z1), (u1, -v, z0)], color, (0, -1, 0))
        )

    def cylinder_along(
        self, u: tuple[float, float], radius: float, z: float, color: RGB, cap: RGB | None = None
    ) -> None:
        """A cylinder lying along the track, its middle `z` above the rails."""
        u0, u1 = u
        ring = [
            (
                radius * math.cos(2 * math.pi * k / CYLINDER_SIDES),
                radius * math.sin(2 * math.pi * k / CYLINDER_SIDES),
            )
            for k in range(CYLINDER_SIDES)
        ]
        for (a, b), (c, d) in zip(ring, ring[1:] + ring[:1], strict=True):
            mid = ((a + c) / 2, (b + d) / 2)
            length = math.hypot(*mid) or 1
            self.faces.append(
                Face(
                    [(u0, a, z + b), (u1, a, z + b), (u1, c, z + d), (u0, c, z + d)],
                    color,
                    (0, mid[0] / length, mid[1] / length),
                )
            )
        end = cap or color
        self.faces.append(Face([(u1, a, z + b) for a, b in ring], end, (1, 0, 0)))
        self.faces.append(Face([(u0, a, z + b) for a, b in reversed(ring)], end, (-1, 0, 0)))

    def cylinder_up(self, u: float, radius: float, z: tuple[float, float], color: RGB) -> None:
        """A cylinder standing up, like a chimney."""
        z0, z1 = z
        ring = [
            (
                u + radius * math.cos(2 * math.pi * k / CYLINDER_SIDES),
                radius * math.sin(2 * math.pi * k / CYLINDER_SIDES),
            )
            for k in range(CYLINDER_SIDES)
        ]
        for (a, b), (c, d) in zip(ring, ring[1:] + ring[:1], strict=True):
            mid = ((a + c) / 2 - u, (b + d) / 2)
            length = math.hypot(*mid) or 1
            self.faces.append(
                Face(
                    [(a, b, z0), (c, d, z0), (c, d, z1), (a, b, z1)],
                    color,
                    (mid[0] / length, mid[1] / length, 0),
                )
            )
        self.faces.append(Face([(a, b, z1) for a, b in ring], color, (0, 0, 1)))

    def slope(
        self, a: tuple[float, float], b: tuple[float, float], v: tuple[float, float], color: RGB
    ) -> None:
        """A flat plate tilted along the track from `a` to `b`, each a (u, z) point, spanning `v`
        across, like a pantograph arm or a windscreen. It shows from either side."""
        (au, az), (bu, bz), (v0, v1) = a, b, v
        du, dz = bu - au, bz - az
        length = math.hypot(du, dz) or 1
        normal = (-dz / length, 0.0, du / length)
        corners = [(au, v0, az), (bu, v0, bz), (bu, v1, bz), (au, v1, az)]
        self.faces.append(Face(corners, color, normal))
        self.faces.append(Face(corners[::-1], color, (-normal[0], 0.0, -normal[2])))

    def nose(
        self, u: tuple[float, float], z: tuple[float, float, float], v: float, color: RGB
    ) -> None:
        """A streamlined end running from `u[0]` out to its tip at `u[1]`, `v` either side of the
        middle: its floor at `z[0]` and its top sloping down from `z[2]` to `z[1]` at the tip."""
        (u0, u1), (z0, low, high) = u, z
        out = 1.0 if u1 > u0 else -1.0
        du, dz = abs(u1 - u0), high - low
        length = math.hypot(du, dz)
        self.faces += [
            Face(
                [(u0, -v, high), (u1, -v, low), (u1, v, low), (u0, v, high)],
                color,
                (out * dz / length, 0.0, du / length),
            ),
            Face([(u1, -v, z0), (u1, v, z0), (u1, v, low), (u1, -v, low)], color, (out, 0.0, 0.0)),
            Face([(u0, v, z0), (u0, v, high), (u1, v, low), (u1, v, z0)], color, (0.0, 1.0, 0.0)),
            Face(
                [(u0, -v, z0), (u1, -v, z0), (u1, -v, low), (u0, -v, high)],
                color,
                (0.0, -1.0, 0.0),
            ),
        ]

    def wheels(self, positions: list[float], radius: float, color: RGB, v: float) -> None:
        for u in positions:
            self.box((u - radius, u + radius), (v - 0.012, v + 0.012), (0.0, 2 * radius), color)
            self.box((u - radius, u + radius), (-v - 0.012, -v + 0.012), (0.0, 2 * radius), color)


def engine() -> Model:
    m = Model()
    h = ENGINE_HALF
    m.box((-h, h - 0.04), (-0.07, 0.07), (0.035, 0.065), FRAME)
    m.wheels([0.2, 0.05, -0.1], 0.04, WHEEL_RED, 0.075)
    m.wheels([-0.26], 0.03, FRAME, 0.075)
    m.cylinder_along((-0.08, h - 0.06), 0.075, 0.14, BLACK_IRON, SMOKEBOX)
    m.box((h - 0.06, h - 0.04), (-0.06, 0.06), (0.07, 0.2), SMOKEBOX)
    m.box((h - 0.04, h), (-0.075, 0.075), (0.0, 0.05), CAB_RED)
    m.cylinder_up(h - 0.11, 0.024, (0.2, 0.29), BLACK_IRON)
    m.box((h - 0.135, h - 0.085), (-0.03, 0.03), (0.29, 0.305), BLACK_IRON)
    m.cylinder_up(0.08, 0.028, (0.2, 0.25), BRASS)
    m.cylinder_up(-0.0, 0.02, (0.2, 0.235), BRASS)
    m.box((h - 0.05, h - 0.03), (-0.015, 0.015), (0.2, 0.225), BRASS)
    m.box((-h, -0.06), (-0.085, 0.085), (0.06, 0.26), CAB_RED)
    m.panel((-0.24, -0.12), (0.16, 0.23), 0.0855, WINDOW)
    m.box((-h - 0.02, -0.04), (-0.095, 0.095), (0.26, 0.285), ROOF_RED)
    m.panel((-h + 0.01, -0.07), (0.065, 0.08), 0.0856, BRASS)
    return m


def tender() -> Model:
    m = Model()
    h = TENDER_HALF
    m.box((-h, h), (-0.07, 0.07), (0.035, 0.06), FRAME)
    m.wheels([0.13, -0.13], 0.035, FRAME, 0.075)
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.2), CAB_RED)
    m.panel((-h + 0.02, h - 0.02), (0.08, 0.095), 0.0805, BRASS)
    m.box((-h + 0.03, h - 0.03), (-0.065, 0.065), (0.2, 0.23), COAL)
    return m


def _bands(m: Model, positions: tuple[float, ...], radius: float, z: float, color: RGB) -> None:
    """Thin rings round a boiler."""
    for u in positions:
        m.cylinder_along((u - 0.006, u + 0.006), radius + 0.003, z, color)


def _side_rods(m: Model, u: tuple[float, float], z: float) -> None:
    for v in (-0.09, 0.087):
        m.box(u, (v, v + 0.003), (z - 0.005, z + 0.005), STEEL)


def _pantograph(m: Model, u: float, roof: float) -> None:
    """A diamond-shaped current collector folded up on the roof at `u`."""
    top = roof + 0.08
    m.box((u - 0.08, u + 0.08), (-0.04, 0.04), (roof, roof + 0.01), PANTOGRAPH)
    m.slope((u - 0.07, roof + 0.01), (u, top), (-0.03, 0.03), PANTOGRAPH)
    m.slope((u + 0.07, roof + 0.01), (u, top), (-0.03, 0.03), PANTOGRAPH)
    m.box((u - 0.012, u + 0.012), (-0.07, 0.07), (top, top + 0.01), PANTOGRAPH)


def mogul() -> Model:
    """A heavy freight engine of the 1880s: a long dark green boiler banded in brass over three
    pairs of driving wheels, with a pony truck in front."""
    m = Model()
    h = MOGUL_HALF
    m.box((-h, h - 0.04), (-0.07, 0.07), (0.035, 0.07), FRAME)
    m.wheels([h - 0.09], 0.03, FRAME, 0.075)
    m.wheels([0.17, 0.05, -0.07], 0.045, WHEEL_RED, 0.075)
    _side_rods(m, (-0.07, 0.17), 0.045)
    m.cylinder_along((-0.14, h - 0.07), 0.08, 0.155, MOGUL_GREEN, SMOKEBOX)
    m.cylinder_along((h - 0.15, h - 0.07), 0.083, 0.155, SMOKEBOX, SMOKEBOX)
    _bands(m, (-0.05, 0.08, 0.2), 0.08, 0.155, BRASS)
    m.box((h - 0.07, h - 0.04), (-0.065, 0.065), (0.07, 0.22), SMOKEBOX)
    m.box((h - 0.04, h), (-0.08, 0.08), (0.0, 0.055), CAB_RED)
    m.cylinder_up(h - 0.11, 0.026, (0.22, 0.31), BLACK_IRON)
    m.box((h - 0.14, h - 0.08), (-0.032, 0.032), (0.31, 0.325), BLACK_IRON)
    m.cylinder_up(0.06, 0.03, (0.22, 0.27), BRASS)
    m.cylinder_up(-0.06, 0.018, (0.22, 0.25), BRASS)
    m.box((-h, -0.12), (-0.088, 0.088), (0.07, 0.28), MOGUL_GREEN)
    m.panel((-0.3, -0.17), (0.17, 0.24), 0.0885, WINDOW)
    m.box((-h - 0.02, -0.1), (-0.098, 0.098), (0.28, 0.3), MOGUL_ROOF)
    m.panel((-h + 0.01, -0.13), (0.08, 0.095), 0.0886, BRASS)
    return m


def mogul_tender() -> Model:
    m = Model()
    h = MOGUL_TENDER_HALF
    m.box((-h, h), (-0.07, 0.07), (0.035, 0.06), FRAME)
    m.wheels([0.17, 0.0, -0.17], 0.035, FRAME, 0.075)
    m.box((-h, h), (-0.085, 0.085), (0.06, 0.22), MOGUL_GREEN)
    m.panel((-h + 0.02, h - 0.02), (0.09, 0.105), 0.0855, BRASS)
    m.box((-h + 0.03, h - 0.03), (-0.07, 0.07), (0.22, 0.25), COAL)
    m.box((-0.12, 0.12), (-0.05, 0.05), (0.25, 0.27), COAL)
    return m


def pacific() -> Model:
    """A fast express engine of the 1910s: a long maroon boiler lined in gold over three pairs of
    tall driving wheels, with smoke deflectors beside its smokebox and a bogie in front."""
    m = Model()
    h = PACIFIC_HALF
    m.box((-h, h - 0.04), (-0.07, 0.07), (0.035, 0.075), FRAME)
    m.wheels([h - 0.08, h - 0.16], 0.03, FRAME, 0.075)
    m.wheels([0.15, 0.01, -0.13], 0.058, MAROON, 0.075)
    m.wheels([-0.31], 0.032, FRAME, 0.075)
    _side_rods(m, (-0.13, 0.15), 0.058)
    m.box((-0.2, h - 0.05), (-0.095, 0.095), (0.075, 0.085), FRAME)
    m.panel((-0.2, h - 0.06), (0.076, 0.084), 0.0955, BRASS)
    m.cylinder_along((-0.16, h - 0.07), 0.088, 0.17, MAROON, SMOKEBOX)
    m.cylinder_along((h - 0.17, h - 0.07), 0.09, 0.17, SMOKEBOX, SMOKEBOX)
    _bands(m, (-0.06, 0.07, 0.2), 0.088, 0.17, BRASS)
    for v in ((0.092, 0.1), (-0.1, -0.092)):
        m.box((h - 0.2, h - 0.08), v, (0.1, 0.25), SMOKEBOX)
    m.box((h - 0.07, h - 0.04), (-0.07, 0.07), (0.075, 0.24), SMOKEBOX)
    m.box((h - 0.04, h), (-0.085, 0.085), (0.0, 0.06), CAB_RED)
    m.cylinder_up(h - 0.12, 0.028, (0.24, 0.3), BLACK_IRON)
    m.cylinder_up(0.05, 0.03, (0.24, 0.28), MAROON)
    m.box((-h, -0.14), (-0.09, 0.09), (0.075, 0.3), MAROON)
    m.panel((-0.38, -0.22), (0.19, 0.26), 0.0905, WINDOW)
    m.panel((-h + 0.01, -0.15), (0.1, 0.115), 0.0906, BRASS)
    m.box((-h - 0.02, -0.12), (-0.1, 0.1), (0.3, 0.32), MAROON_ROOF)
    return m


def pacific_tender() -> Model:
    m = Model()
    h = PACIFIC_TENDER_HALF
    m.box((-h, h), (-0.07, 0.07), (0.035, 0.065), FRAME)
    m.wheels([0.2, 0.0, -0.2], 0.038, FRAME, 0.075)
    m.box((-h, h), (-0.09, 0.09), (0.065, 0.24), MAROON)
    m.panel((-h + 0.02, h - 0.02), (0.1, 0.115), 0.0905, BRASS)
    m.panel((-h + 0.02, h - 0.02), (0.21, 0.222), 0.0905, BRASS)
    m.box((-h + 0.03, h - 0.08), (-0.075, 0.075), (0.24, 0.27), COAL)
    m.box((-0.14, 0.1), (-0.05, 0.05), (0.27, 0.29), COAL)
    return m


def boxcab() -> Model:
    """An electric engine of the 1920s: a plain brown box cab on two bogies, with windows at both
    ends and a pantograph on its roof."""
    m = Model()
    h = BOXCAB_HALF
    for u in (-0.2, 0.2):
        m.box((u - 0.12, u + 0.12), (-0.065, 0.065), (0.03, 0.07), FRAME)
        m.wheels([u - 0.07, u + 0.07], 0.035, FRAME, 0.075)
    m.box((-h, h), (-0.075, 0.075), (0.07, 0.09), FRAME)
    m.box((-h + 0.01, h - 0.01), (-0.085, 0.085), (0.09, 0.27), BOXCAB_BROWN)
    m.box((-h, h), (-0.09, 0.09), (0.27, 0.285), ROOF_GREY)
    for end in (h - 0.01, -h + 0.01):
        for v in ((-0.065, -0.012), (0.012, 0.065)):
            m.box((end - 0.004, end + 0.004), v, (0.18, 0.24), WINDOW)
    for u in (h - 0.1, -h + 0.03):
        m.panel((u, u + 0.07), (0.18, 0.24), 0.0855, WINDOW)
    for u in (h - 0.15, -h + 0.12):
        m.panel((u, u + 0.03), (0.1, 0.25), 0.0855, shade(BOXCAB_BROWN, 0.72))
    for u in (-0.12, -0.04, 0.04):
        m.panel((u, u + 0.06), (0.15, 0.22), 0.0855, shade(BOXCAB_BROWN, 0.82))
    _pantograph(m, 0.0, 0.285)
    return m


def express() -> Model:
    """An express electric engine of the 1960s: a long blue-and-white body with a streamlined nose
    at each end and two pantographs."""
    m = Model()
    h = EXPRESS_HALF
    body = h - 0.1
    for u in (-0.26, 0.26):
        m.box((u - 0.1, u + 0.1), (-0.065, 0.065), (0.03, 0.07), FRAME)
        m.wheels([u - 0.055, u + 0.055], 0.035, FRAME, 0.075)
    m.box((-body, body), (-0.085, 0.085), (0.07, 0.27), EXPRESS_BLUE)
    m.panel((-body, body), (0.17, 0.25), 0.0855, EXPRESS_WHITE)
    for u in (-0.2, -0.1, 0.0, 0.1, 0.2):
        m.panel((u - 0.03, u + 0.03), (0.19, 0.23), 0.0856, WINDOW)
    m.box((-body, body), (-0.08, 0.08), (0.27, 0.285), ROOF_GREY)
    for tip in (h, -h):
        root = body if tip > 0 else -body
        m.nose((root, tip), (0.07, 0.13, 0.27), 0.085, EXPRESS_BLUE)
        out = 0.004 if tip > 0 else -0.004
        a = (root + out, 0.27 + 0.004)
        b = (root + (tip - root) * 0.6 + out, 0.27 + (0.13 - 0.27) * 0.6 + 0.004)
        m.slope(a, b, (-0.065, 0.065), WINDOW)
    for u in (-0.18, 0.18):
        _pantograph(m, u, 0.285)
    return m


def car(good: Good, full: bool) -> Model:
    """A car for a good on its frame and wheels, loaded or empty, its body by what it carries."""
    m = Model()
    h = CAR_HALF
    m.box((-h, h), (-0.07, 0.07), (0.035, 0.06), FRAME)
    m.wheels([h - 0.07, -h + 0.07], 0.035, FRAME, 0.075)
    _body_of(good)(m, good, full)
    return m


type CarBody = Callable[[Model, Good, bool], None]


def _body_of(good: Good) -> CarBody:
    if good in VANS:
        return _van
    if good in HOPPERS:
        return _hopper
    if good in HEAPS:
        return _heap
    if good in TANKS:
        return _tank
    return CAR_BODIES.get(good, _bare)


def _bare(m: Model, good: Good, full: bool) -> None:
    """Nothing on the frame."""


def _van(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    body, stripe = VANS[good]
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.24), body)
    m.box((-h - 0.01, h + 0.01), (-0.088, 0.088), (0.24, 0.26), ROOF_GREY)
    m.panel((-0.06, 0.06), (0.07, 0.22), 0.0805, shade(body, 0.7) if full else WINDOW)
    m.panel((-h + 0.01, h - 0.01), (0.2, 0.215), 0.0806, stripe)


def _hopper(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    body, load = HOPPERS[good]
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.19), body)
    m.panel((-h + 0.02, h - 0.02), (0.08, 0.17), 0.0805, shade(body, 0.85))
    if full:
        m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.19, 0.22), load)
        m.box((-0.1, 0.1), (-0.05, 0.05), (0.22, 0.24), load)


def _heap(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.13), DECK)
    m.panel((-h + 0.02, h - 0.02), (0.075, 0.12), 0.0805, shade(DECK, 0.8))
    if full:
        m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.13, 0.17), HEAPS[good])
        m.box((-0.13, 0.13), (-0.05, 0.05), (0.17, 0.2), HEAPS[good])


def _tank(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.07, 0.07), (0.06, 0.085), DECK)
    m.cylinder_along((-h + 0.01, h - 0.01), 0.08, 0.165, TANKS[good])
    m.cylinder_up(0.0, 0.025, (0.24, 0.27), TANKS[good])


def _deck(m: Model) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.085), DECK)


def _car_carrier(m: Model, good: Good, full: bool) -> None:
    _deck(m)
    if full:
        for u in (-0.12, 0.12):
            m.box((u - 0.09, u + 0.09), (-0.05, 0.05), (0.095, 0.14), AUTO_BODY)
            m.box((u - 0.05, u + 0.05), (-0.045, 0.045), (0.14, 0.18), WINDOW)


def _tank_carrier(m: Model, good: Good, full: bool) -> None:
    _deck(m)
    if full:
        m.box((-0.2, 0.2), (-0.07, 0.07), (0.085, 0.15), TANK_HULL)
        m.box((-0.08, 0.08), (-0.05, 0.05), (0.15, 0.2), shade(TANK_HULL, 0.85))
        m.box((0.08, 0.26), (-0.012, 0.012), (0.165, 0.185), shade(TANK_HULL, 0.6))


def _plane_carrier(m: Model, good: Good, full: bool) -> None:
    _deck(m)
    if full:
        m.box((-0.24, 0.24), (-0.03, 0.03), (0.09, 0.15), PLANE_BODY)
        m.box((-0.05, 0.05), (-0.085, 0.085), (0.12, 0.135), shade(PLANE_BODY, 0.85))
        m.box((-0.24, -0.17), (-0.012, 0.012), (0.15, 0.2), shade(PLANE_BODY, 0.75))


def _boxcar(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    body, roof = {
        Good.GOODS: (BOXCAR, ROOF_GREY),
        Good.CANS: (BOXCAR, ROOF_GREY),
        Good.BANANAS: (BANANA_VAN, (240, 236, 220)),
        Good.BREAD: (BREAD_VAN, (236, 226, 204)),
    }[good]
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.24), body)
    m.box((-h - 0.01, h + 0.01), (-0.088, 0.088), (0.24, 0.26), roof)
    door = shade(body, 0.7) if full else WINDOW
    m.panel((-0.06, 0.06), (0.07, 0.22), 0.0805, door)
    m.panel((-h + 0.01, h - 0.01), (0.07, 0.085), 0.0806, shade(body, 0.75))
    if good == Good.CANS:
        m.panel((-h + 0.01, h - 0.01), (0.2, 0.215), 0.0806, CANS_STRIPE)


def _grain_hopper(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    body = GRAIN_HOPPER if good == Good.WHEAT else SUGAR_HOPPER
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.2), body)
    m.panel((-h + 0.02, h - 0.02), (0.08, 0.18), 0.0805, shade(body, 0.85))
    if good == Good.SUGAR:
        m.box((-h + 0.01, h - 0.01), (-0.07, 0.07), (0.2, 0.235), shade(body, 0.95))
    elif full:
        m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.2, 0.22), GRAIN)
        m.box((-0.12, 0.12), (-0.05, 0.05), (0.22, 0.24), GRAIN)


def _fruit_wagon(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.13), DECK)
    m.panel((-h + 0.02, h - 0.02), (0.075, 0.12), 0.0805, shade(DECK, 0.8))
    if full:
        fruit = {Good.APPLES: APPLE, Good.DATES: DATE, Good.ORANGES: ORANGE}[good]
        m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.13, 0.16), fruit)
        m.box((-0.13, 0.13), (-0.05, 0.05), (0.16, 0.18), fruit)
        if good == Good.APPLES:
            for u in (-0.15, -0.05, 0.06, 0.14):
                m.box((u - 0.012, u + 0.012), (-0.04, -0.02), (0.16, 0.175), APPLE_LIGHT)


def _coal_hopper(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.19), HOPPER)
    m.panel((-h + 0.02, h - 0.02), (0.08, 0.17), 0.0805, shade(HOPPER, 0.85))
    if full:
        m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.19, 0.22), COAL)
        m.box((-0.1, 0.1), (-0.05, 0.05), (0.22, 0.24), COAL)


def _log_wagon(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.085), DECK)
    for u in (-h + 0.005, h - 0.025):
        m.box((u, u + 0.02), (-0.08, 0.08), (0.085, 0.2), FRAME)
    if full:
        for row, z in enumerate((0.115, 0.17)):
            for v in (-0.045, 0.0, 0.045)[: 3 - row]:
                m.cylinder_along((-h + 0.03, h - 0.03), 0.026, z, LOG, LOG_END)
                m.faces[-CYLINDER_SIDES - 2 :] = [
                    Face([(c[0], c[1] + v, c[2]) for c in f.corners], f.color, f.normal)
                    for f in m.faces[-CYLINDER_SIDES - 2 :]
                ]


def _coach(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.23), COACH)
    m.box((-h - 0.01, h + 0.01), (-0.086, 0.086), (0.23, 0.25), ROOF_GREY)
    m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.25, 0.265), ROOF_GREY)
    m.panel((-h + 0.01, h - 0.01), (0.105, 0.122), 0.0805, COACH_BAND)
    window = LIT_WINDOW if full else WINDOW
    for u in (-0.17, -0.085, 0.0, 0.085, 0.17):
        m.panel((u - 0.03, u + 0.03), (0.135, 0.195), 0.0806, window)
    m.panel((-h + 0.01, h - 0.01), (0.07, 0.085), 0.0806, shade(COACH, 0.75))


def _mail_van(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.24), MAIL_VAN)
    m.box((-h - 0.01, h + 0.01), (-0.088, 0.088), (0.24, 0.26), ROOF_GREY)
    m.panel((-h + 0.01, h - 0.01), (0.2, 0.215), 0.0805, BRASS)
    door = shade(MAIL_VAN, 0.7) if full else WINDOW
    m.panel((-0.05, 0.05), (0.07, 0.19), 0.0806, door)
    m.panel((-h + 0.03, -h + 0.08), (0.15, 0.19), 0.0806, WINDOW)
    m.panel((h - 0.08, h - 0.03), (0.15, 0.19), 0.0806, WINDOW)


def _steel_wagon(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    _deck(m)
    if full:
        for v in (-0.04, 0.04):
            m.box((-h + 0.03, h - 0.03), (v - 0.03, v + 0.03), (0.085, 0.12), STEEL)
        m.box((-h + 0.05, h - 0.05), (-0.03, 0.03), (0.12, 0.155), STEEL)


# The body of each car that is not a van, hopper, heap or tank (VANS, HOPPERS, HEAPS, TANKS).
CAR_BODIES: dict[Good, CarBody] = {
    Good.CARS: _car_carrier,
    Good.TANKS: _tank_carrier,
    Good.PLANES: _plane_carrier,
    Good.GOODS: _boxcar,
    Good.CANS: _boxcar,
    Good.BANANAS: _boxcar,
    Good.BREAD: _boxcar,
    Good.WHEAT: _grain_hopper,
    Good.SUGAR: _grain_hopper,
    Good.APPLES: _fruit_wagon,
    Good.DATES: _fruit_wagon,
    Good.ORANGES: _fruit_wagon,
    Good.COAL: _coal_hopper,
    Good.LUMBER: _log_wagon,
    Good.PASSENGERS: _coach,
    Good.MAIL: _mail_van,
    Good.STEEL: _steel_wagon,
}


def shade(color: RGB, factor: float) -> RGB:
    r, g, b = color
    return (min(255, round(r * factor)), min(255, round(g * factor)), min(255, round(b * factor)))


class Look(Enum):
    ENGINE = "engine"
    TENDER = "tender"


@dataclass(frozen=True)
class EngineModel:
    """One engine of the catalogue: how to build it and its tender if it has one, their half
    lengths in cells, and where its chimney is (cells ahead of its middle and above the rails) if
    it makes smoke."""

    engine: Callable[[], Model]
    half: float
    tender: Callable[[], Model] | None = None
    tender_half: float = 0.0
    chimney: tuple[float, float] | None = None


ENGINES: dict[str, EngineModel] = {
    "planet": EngineModel(engine, ENGINE_HALF, tender, TENDER_HALF, (ENGINE_HALF - 0.11, 0.31)),
    "mogul": EngineModel(
        mogul, MOGUL_HALF, mogul_tender, MOGUL_TENDER_HALF, (MOGUL_HALF - 0.11, 0.33)
    ),
    "pacific": EngineModel(
        pacific, PACIFIC_HALF, pacific_tender, PACIFIC_TENDER_HALF, (PACIFIC_HALF - 0.12, 0.3)
    ),
    "boxcab": EngineModel(boxcab, BOXCAB_HALF),
    "express": EngineModel(express, EXPRESS_HALF),
}


MODELS: dict[object, Callable[[], Model]] = {
    Look.ENGINE: engine,
    Look.TENDER: tender,
    **{(Look.ENGINE, name): model.engine for name, model in ENGINES.items()},
    **{(Look.TENDER, name): model.tender for name, model in ENGINES.items() if model.tender},
    **{(good, full): (lambda g=good, f=full: car(g, f)) for good in Good for full in (True, False)},
}


def render(model: Model, heading: float, matrix: npt.NDArray[np.float64]) -> Sprite:
    """The model seen running `heading` radians round from the map's x axis, with the map's
    plane-to-art-pixel matrix, at DETAIL pixels per art pixel."""
    scale = matrix * DETAIL
    cos, sin = math.cos(heading), math.sin(heading)
    up = float(np.hypot(*scale[:, 0])) * HEIGHT_PER_WIDTH
    # The way to the viewer in the plane: what moves straight down the screen.
    towards = np.linalg.solve(matrix, np.array([0.0, 1.0]))
    towards = towards / np.hypot(*towards)
    view = np.array([towards[0] * TILT, towards[1] * TILT, HEIGHT_PER_WIDTH])
    light_plane = np.linalg.solve(matrix, np.array([-1.0, -1.0]))
    light_plane = light_plane / np.hypot(*light_plane)
    light = np.array([*light_plane * 0.6, 0.8])

    def world(p: Vec3) -> tuple[float, float, float]:
        u, v, z = p
        return u * cos - v * sin, u * sin + v * cos, z

    def to_screen(p: tuple[float, float, float]) -> tuple[float, float]:
        x, y, z = p
        sx, sy = scale @ np.array([x, y])
        return float(sx), float(sy - z * up)

    drawn = []
    for face in model.faces:
        normal = np.array(world(face.normal))
        if normal @ view <= 1e-6:
            continue
        points = [world(c) for c in face.corners]
        depth = float(np.mean([np.array([x, y, z]) @ view for x, y, z in points]))
        lit = float(np.clip(normal @ light, 0, 1))
        level = SHADES[min(len(SHADES) - 1, int(lit * len(SHADES)))]
        drawn.append((depth, [to_screen(p) for p in points], shade(face.color, level)))
    drawn.sort(key=lambda item: item[0])
    xs = [x for _, pts, _ in drawn for x, _ in pts]
    ys = [y for _, pts, _ in drawn for _, y in pts]
    left, top = math.floor(min(xs)) - 2, math.floor(min(ys)) - 2
    width, height = math.ceil(max(xs)) - left + 3, math.ceil(max(ys)) - top + 3
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    paint = ImageDraw.Draw(image)
    for _, points, color in drawn:
        paint.polygon([(x - left, y - top) for x, y in points], fill=(*color, 255))
    _outline(image)
    return Sprite(image, (-left, -top))


def _outline(image: Image.Image) -> None:
    pixels = np.asarray(image).copy()
    solid = pixels[..., 3] > 0
    around = np.zeros_like(solid)
    around[1:] |= solid[:-1]
    around[:-1] |= solid[1:]
    around[:, 1:] |= solid[:, :-1]
    around[:, :-1] |= solid[:, 1:]
    edge = around & ~solid
    pixels[edge] = (*OUTLINE, 255)
    image.paste(Image.fromarray(pixels))


@dataclass(frozen=True)
class TrainSprites:
    """Every look of every engine and car, packed into one picture: where each sits in it."""

    image: npt.NDArray[np.uint8]
    places: dict[tuple[object, int], tuple[int, int, int, int, int, int]]

    def place(self, look: object, heading: float) -> tuple[int, int, int, int, int, int]:
        """x, y, width, height and anchor of the look running closest to `heading` radians."""
        index = round(heading / (2 * math.pi) * HEADINGS) % HEADINGS
        return self.places[(look, index)]


@cache
def train_sprites(matrix_key: tuple[float, float, float, float]) -> TrainSprites:
    """All the train looks for a map projection, given as its matrix's four numbers."""
    matrix = np.array(matrix_key, dtype=np.float64).reshape(2, 2)
    sprites = {
        (look, index): render(make(), 2 * math.pi * index / HEADINGS, matrix)
        for look, make in MODELS.items()
        for index in range(HEADINGS)
    }
    width = 1024
    x = y = row = 0
    places = {}
    for key, sprite in sprites.items():
        w, h = sprite.image.size
        if x + w > width:
            x, y, row = 0, y + row + 1, 0
        places[key] = (x, y, w, h, *sprite.anchor)
        x += w + 1
        row = max(row, h)
    sheet = Image.new("RGBA", (width, y + row), (0, 0, 0, 0))
    for key, sprite in sprites.items():
        sheet.paste(sprite.image, places[key][:2])
    return TrainSprites(np.asarray(sheet).copy(), places)
