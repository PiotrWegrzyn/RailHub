"""Trains as pixel art. Each engine and car is a small model of boxes and cylinders
(railroads.ui.pixel.models), drawn for HEADINGS directions with the map's own projection.

Trains are drawn at DETAIL pixels for every map art pixel, so they keep their windows, wheels and
trim even though a car is less than a cell long.

All the looks go on one sheet. models.draw_model works out a picture one face at a time, which is
slow for thousands of pictures, so the sheet does its maths over all of a model's faces at once
instead: the same sums in the same order, so the pictures come out the same to the pixel, and a
change to draw_model's maths needs the same change here.
"""

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache
from typing import Self

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from railroads.sim.goods import Good
from railroads.sim.grids import FloatArray, IntArray
from railroads.ui.iso_camera import TILT
from railroads.ui.pixel import train_parts as parts
from railroads.ui.pixel.art import Sprite
from railroads.ui.pixel.models import (
    HEIGHT_PER_WIDTH,
    RGB,
    SHADES,
    Face,
    Model,
    _outline,
    decals,
    fine_shades,
    halfway,
    keeping_fine,
    shade,
    without_fine,
)

type Points = tuple[FloatArray, FloatArray, FloatArray]

HEADINGS = 24
# How many times as fine as the sheet close-up pictures must be drawn to show the finest details:
# valve gear, pipes, grab irons and the like, which only clutter a smaller picture.
FINEST_FROM = 6
DETAIL = 2
# Trains are modelled in cells and drawn this much smaller, to sit on the narrow track.
SCALE = 0.8

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
PLANK: RGB = (206, 168, 112)
PLANK_END: RGB = (232, 204, 150)
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
    Good.FERTILISER: ((206, 184, 140), (120, 160, 80)),
    Good.INCENSE: ((96, 70, 110), (230, 200, 130)),
    Good.BEER: ((150, 96, 40), (240, 190, 70)),
    Good.VODKA: ((210, 214, 220), (60, 90, 170)),
    Good.SAKE: ((200, 170, 120), (190, 40, 40)),
    Good.SOY_SAUCE: ((92, 56, 40), (196, 46, 40)),
    Good.TOFU: ((226, 226, 220), (120, 160, 196)),
    Good.MISO: ((170, 120, 70), (230, 210, 160)),
    Good.TEA: ((96, 130, 70), (210, 220, 160)),
    Good.DRIED_FISH: ((150, 140, 116), (200, 206, 210)),
    Good.LEATHER: ((120, 80, 48), (190, 140, 90)),
    Good.SHOES: ((82, 52, 38), (200, 160, 110)),
    Good.CATTLE: ((132, 88, 56), (220, 200, 170)),
    Good.PIGS: ((132, 88, 56), (230, 166, 160)),
    Good.CHICKENS: ((150, 120, 80), (244, 240, 230)),
    Good.EGGS: ((214, 194, 140), (244, 232, 206)),
    Good.SHEEP: ((120, 96, 70), (238, 234, 222)),
    Good.MEAT: ((214, 194, 140), (190, 60, 56)),
    Good.GLASS: ((96, 130, 150), (190, 236, 244)),
    Good.PLASTICS: ((220, 220, 224), (220, 70, 80)),
    Good.APPLIANCES: ((226, 228, 232), (90, 120, 170)),
    Good.PAPER: ((214, 204, 176), (120, 110, 90)),
    Good.NEWSPAPERS: ((100, 100, 104), (230, 228, 220)),
    Good.REFINED_SUGAR: ((236, 230, 232), (220, 120, 150)),
    Good.PORCELAIN: ((230, 232, 240), (60, 90, 190)),
    Good.DAIRY: ((236, 232, 218), (230, 196, 90)),
    Good.SAUSAGES: ((214, 194, 140), (160, 70, 50)),
    Good.POTS: ((150, 100, 70), (210, 150, 100)),
    Good.STOVE_TILES: ((214, 206, 190), (60, 110, 160)),
    Good.CASTINGS: ((70, 72, 78), (150, 150, 156)),
    Good.LUXURIES: ((90, 40, 90), (232, 190, 70)),
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
    Good.WOOD_CHIPS: ((112, 88, 62), (214, 180, 122)),
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
    Good.MILK: (240, 240, 236),
    Good.JUICE: (236, 150, 40),
}
# Vans for animals, with slatted sides they show through, and refrigerated vans with ice hatches
# on the roof.
LIVESTOCK = frozenset({Good.CATTLE, Good.PIGS, Good.SHEEP, Good.CHICKENS})
REEFERS = frozenset({Good.MEAT, Good.FISH, Good.DAIRY, Good.SAUSAGES, Good.EGGS})
MOGUL_GREEN: RGB = (38, 86, 56)
MOGUL_ROOF: RGB = (24, 54, 36)
MAROON: RGB = (118, 30, 40)
MAROON_ROOF: RGB = (70, 20, 26)
BOXCAB_BROWN: RGB = (122, 74, 42)
EXPRESS_BLUE: RGB = (36, 78, 168)
EXPRESS_WHITE: RGB = (230, 232, 236)
TEE_RED: RGB = (128, 30, 42)
TEE_BEIGE: RGB = (214, 198, 156)
PANTOGRAPH: RGB = (60, 60, 66)
AUTO_BODY: RGB = (176, 44, 60)
TANK_HULL: RGB = (88, 100, 60)
PLANE_BODY: RGB = (188, 196, 206)
ORANGE: RGB = (240, 140, 36)
DINER: RGB = (98, 58, 40)
DINER_CREAM: RGB = (232, 218, 178)
WOOD: RGB = (128, 88, 54)
COPPER: RGB = (184, 108, 58)
LAGGING: RGB = (142, 90, 50)
TENDER_GREEN: RGB = (50, 84, 58)
APPLE_GREEN: RGB = (82, 136, 62)
LINING: RGB = (222, 222, 212)
GRAPHITE: RGB = (196, 198, 204)
BRUNSWICK: RGB = (34, 52, 42)
OCHRE: RGB = (176, 136, 62)
OAK: RGB = (178, 98, 42)
ROCKET_YELLOW: RGB = (236, 196, 32)

# Sizes in cells: how far apart the engine and cars sit, and their half lengths.
CAR_HALF = 0.24
BOXCAB_HALF = 0.36
EXPRESS_HALF = 0.42
COUPLING = 0.04


PLANK_SHADES = (1.0, 0.8, 0.93, 0.74, 0.97, 0.85, 0.9, 0.78)


def _planks(m: Model, u: tuple[float, float], radius: float, z: float, color: RGB) -> None:
    """Up close, the boards lagging a boiler: strips along it round its upper part, each a shade
    of its own, with a dark seam between each two."""
    if not keeping_fine():
        return
    u0, u1 = u
    count = 14
    first, last = -0.15 * math.pi, 1.15 * math.pi
    step = (last - first) / count
    seam = shade(color, 0.5)
    with m.fine():
        for k in range(count):
            a, b = first + step * k, first + step * (k + 1)
            middle = (a + b) / 2
            normal = (0.0, math.cos(middle), math.sin(middle))
            out = radius + 0.0012
            corners = [
                (u0, out * math.cos(a), z + out * math.sin(a)),
                (u0, out * math.cos(b), z + out * math.sin(b)),
                (u1, out * math.cos(b), z + out * math.sin(b)),
                (u1, out * math.cos(a), z + out * math.sin(a)),
            ]
            m.faces.append(Face(corners[::-1], shade(color, PLANK_SHADES[k % 8]), normal))
            line = radius + 0.0018
            gap = step * 0.12
            corners = [
                (u0, line * math.cos(b - gap), z + line * math.sin(b - gap)),
                (u0, line * math.cos(b + gap), z + line * math.sin(b + gap)),
                (u1, line * math.cos(b + gap), z + line * math.sin(b + gap)),
                (u1, line * math.cos(b - gap), z + line * math.sin(b - gap)),
            ]
            m.faces.append(Face(corners[::-1], seam, (0.0, math.cos(b), math.sin(b))))


def _bands(m: Model, positions: tuple[float, ...], radius: float, z: float, color: RGB) -> None:
    """Thin rings round a boiler."""
    for u in positions:
        m.cylinder_along((u - 0.006, u + 0.006), radius + 0.003, z, color)


def _roof(
    m: Model,
    u: tuple[float, float],
    half: float,
    z: tuple[float, float],
    color: RGB,
    crown: float | None = None,
) -> None:
    """A roof over `u`, `half` either side of the middle: a flat slab from `z[0]` to `z[1]` far
    out, arched up to `crown` (a little over the slab unless given) up close."""
    with m.coarse():
        m.box(u, (-half, half), z, color)
    with m.fine():
        m.vault(u, half, (z[0] + 0.006, crown or z[1] + 0.008), color)


def _dome(m: Model, u: float, radius: float, base: float, height: float, color: RGB) -> None:
    """A dome standing on a boiler: a short drum with a rounded top."""
    drum = base + height * 0.45
    with m.coarse():
        m.cylinder_up(u, radius, (base, base + height * 0.8), color)
    with m.fine():
        m.cylinder_up(u, radius * 1.15, (base, base + 0.01), color)
        m.cylinder_up(u, radius, (base, drum), color)
        m.cap(u, radius, (drum, base + height), color)


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


def boxcab() -> Model:
    """An electric engine of the 1920s: a plain brown box cab on two bogies, with windows at both
    ends and a pantograph on its roof."""
    m = Model()
    h = BOXCAB_HALF
    for u in (-0.2, 0.2):
        m.box((u - 0.12, u + 0.12), (-0.065, 0.065), (0.03, 0.07), FRAME)
        m.wheels([u - 0.07, u + 0.07], 0.035, FRAME, 0.075)
    m.box((-h, h), (-0.075, 0.075), (0.07, 0.09), FRAME)
    m.body((-h + 0.01, h - 0.01), (-0.085, 0.085), (0.09, 0.27), BOXCAB_BROWN)
    _roof(m, (-h, h), 0.09, (0.27, 0.285), ROOF_GREY)
    for end in (h - 0.01, -h + 0.01):
        for v in ((-0.065, -0.012), (0.012, 0.065)):
            m.box((end - 0.004, end + 0.004), v, (0.18, 0.24), WINDOW)
    for u in (h - 0.1, -h + 0.03):
        m.panel((u, u + 0.07), (0.18, 0.24), 0.0855, WINDOW)
    for u in (h - 0.15, -h + 0.12):
        m.panel((u, u + 0.03), (0.1, 0.25), 0.0855, shade(BOXCAB_BROWN, 0.72))
    for u in (-0.12, -0.04, 0.04):
        m.panel((u, u + 0.06), (0.15, 0.22), 0.0855, shade(BOXCAB_BROWN, 0.82))
    with m.coarse():
        _pantograph(m, 0.0, 0.285)
    parts.pantograph(m, 0.0, 0.285, PANTOGRAPH)
    _hand_electric(m, h, (-0.2, 0.2), 0.07, 0.285)
    parts.louvres(m, (-0.12, 0.1), (0.1, 0.14), 0.0857, BOXCAB_BROWN, 5)
    parts.rivets(m, (-h + 0.01, h - 0.01), 0.262, 0.0857, shade(BOXCAB_BROWN, 0.7), 14)
    for end in (h, -h):
        parts.buffers(m, end, 0.08, beam=FRAME)
    return m


def _hand_electric(
    m: Model, half: float, bogies: tuple[float, ...], span: float, roof: float
) -> None:
    """The fine details of the hand-built electrics: bogies with axle boxes and solid wheels,
    lamps at both ends and a cable along the roof."""
    for c in bogies:
        for u in (c - span, c + span):
            parts.disc_wheel(m, u, 0.035, 0.075, FRAME)
        parts.bogie_frame(m, c, span + 0.05, 0.045, 0.088)
    for end in (half, -half):
        for v in (-0.05, 0.05):
            parts.lamp(m, end + (0.0006 if end > 0 else -0.0006), 0.12, v, parts.LAMP, 0.01)
    with m.fine():
        m.box((-half + 0.05, half - 0.05), (0.03, 0.036), (roof, roof + 0.006), parts.BRIGHT)


def express() -> Model:
    """An express electric engine of the 1960s: a long blue-and-white body with a streamlined nose
    at each end and two pantographs."""
    m = Model()
    h = EXPRESS_HALF
    body = h - 0.1
    for u in (-0.26, 0.26):
        m.box((u - 0.1, u + 0.1), (-0.065, 0.065), (0.03, 0.07), FRAME)
        m.wheels([u - 0.055, u + 0.055], 0.035, FRAME, 0.075)
    m.body((-body, body), (-0.085, 0.085), (0.07, 0.27), TEE_RED)
    m.panel((-body, body), (0.1, 0.15), 0.0855, TEE_BEIGE)
    for u in (-0.2, -0.1, 0.0, 0.1, 0.2):
        m.panel((u - 0.03, u + 0.03), (0.19, 0.23), 0.0856, WINDOW)
    _roof(m, (-body, body), 0.085, (0.27, 0.285), ROOF_GREY)
    for tip in (h, -h):
        root = body if tip > 0 else -body
        with m.coarse():
            m.nose((root, tip), (0.07, 0.13, 0.27), 0.085, TEE_RED)
        with m.fine():
            m.snout((root, tip), (0.07, 0.13, 0.27), 0.085, TEE_RED)
        out = 0.004 if tip > 0 else -0.004
        a = (root + out, 0.27 + 0.004)
        b = (root + (tip - root) * 0.6 + out, 0.27 + (0.13 - 0.27) * 0.6 + 0.004)
        m.slope(a, b, (-0.065, 0.065), WINDOW)
    for u in (-0.18, 0.18):
        with m.coarse():
            _pantograph(m, u, 0.285)
        parts.pantograph(m, u, 0.285, PANTOGRAPH, raised=u > 0)
    _hand_electric(m, h, (-0.26, 0.26), 0.055, 0.285)
    with m.fine():
        for v in (0.0857, -0.0857):
            parts.plate(m, (-body, body), (0.09, 0.098), v, shade(TEE_BEIGE, 0.8))
    for end in (h, -h):
        parts.coupler(m, end, 0.05)
    return m


def _disc(
    m: Model, u: float, z: float, radius: float, v: float, color: RGB, teeth: bool = False
) -> None:
    """A flat round plate on the side of the model `v` out from its middle, facing out: a
    flywheel, or with `teeth` a cog."""
    points = 16 if teeth else 8
    corners = []
    for k in range(points):
        angle = 2 * math.pi * k / points
        reach = radius * (0.8 if teeth and k % 2 else 1.0)
        corners.append((u + reach * math.cos(angle), v, z + reach * math.sin(angle)))
    m.faces.append(Face(corners, color, (0, 1 if v > 0 else -1, 0)))


def _slant(
    m: Model, a: tuple[float, float], b: tuple[float, float], thick: float, v: float, color: RGB
) -> None:
    """A bar `thick` tall running at a slant from `a` to `b`, each a (u, z) point, on both sides
    of the model `v` out from its middle."""
    (au, az), (bu, bz) = a, b
    half = thick / 2
    for side in (v, -v):
        corners = [(au, side, az - half), (bu, side, bz - half), (bu, side, bz + half)]
        corners.append((au, side, az + half))
        m.faces.append(Face(corners, color, (0, 1 if side > 0 else -1, 0)))


@dataclass(frozen=True)
class Pioneer:
    """A steam engine of the first years: a small boiler on a frame with no cab, a tall thin
    chimney at the front and wheels given as (u, radius). It may have upright cylinders on the
    boiler with crossheads over them, beams along both sides worked by side cylinders, a flywheel
    or a cog for a rack rail on one side, cylinders slanting down to the front wheels, and a
    firebox at the back, square or with a domed top. Its front end may be painted apart from the
    boiler, and its chimney crowned with a ring of petals."""

    half: float
    boiler: RGB
    frame: RGB
    wheels: tuple[tuple[float, float], ...]
    wheel: RGB = FRAME
    chimney_color: RGB = BLACK_IRON
    chimney_top: float = 0.42
    radius: float = 0.065
    rods: bool = False
    uprights: tuple[float, ...] = ()
    crossheads: bool = False
    beams: bool = False
    flywheel: bool = False
    rack: bool = False
    slanted: bool = False
    firebox: RGB | None = None
    haystack: bool = False
    front: RGB | None = None
    hoops: RGB | None = None
    crown: bool = False
    tyre: RGB = parts.TYRE
    flue: float = 0.017
    hoop_count: int = 3
    upright: RGB = BRASS
    column: RGB | None = None
    back_end: RGB | None = None
    spokes: int = 6
    tubes: RGB | None = None

    @property
    def chimney(self) -> tuple[float, float]:
        return self.half - 0.06, self.chimney_top

    def model(self) -> Model:
        m = Model()
        h, r = self.half, self.radius
        z = 0.075 + r
        top = z + r
        m.box((-h, h), (-0.065, 0.065), (0.045, 0.075), self.frame)
        for u, size in self.wheels:
            m.wheels([u], size, self.wheel, 0.075)
        if self.rods:
            us = [u for u, _ in self.wheels]
            _side_rods(m, (min(us), max(us)), self.wheels[0][1])
        start = -h + 0.07
        if self.firebox is not None:
            high = top + 0.01 if self.haystack else top - 0.015
            m.box((start, start + 0.08), (-r, r), (0.075, high), self.firebox)
            if self.haystack:
                m.spire(start + 0.04, r, (top + 0.01, top + 0.07), self.firebox)
            start += 0.08
        m.cylinder_along((start, h - 0.02), r, z, self.boiler, shade(self.boiler, 0.8))
        if self.tubes is not None:
            self._tubes(m, (start + 0.01, h - 0.1), z, self.tubes)
        if self.front is not None:
            m.cylinder_along((h - 0.1, h - 0.015), r + 0.004, z, self.front, self.front)
            m.box((h - 0.1, h - 0.03), (-r * 0.7, r * 0.7), (0.06, z), self.front)
        u = self.chimney[0]
        m.cylinder_up(u, self.flue, (top - 0.01, self.chimney_top - 0.02), self.chimney_color)
        rim = self.flue + 0.009
        m.box(
            (u - rim, u + rim),
            (-rim, rim),
            (self.chimney_top - 0.025, self.chimney_top),
            self.chimney_color,
        )
        if self.back_end is not None:
            m.cylinder_along((start, start + 0.03), r + 0.004, z, self.back_end, self.back_end)
        if self.column is not None and self.uprights:
            middle = sum(self.uprights) / len(self.uprights)
            m.box(
                (middle - 0.03, middle + 0.03), (-0.03, 0.03), (top - 0.01, top + 0.1), self.column
            )
            m.box(
                (middle - 0.014, middle + 0.014),
                (-0.014, 0.014),
                (top + 0.1, self.chimney_top - 0.02),
                self.column,
            )
        for u in self.uprights:
            m.cylinder_up(u, 0.024, (top - 0.01, top + 0.06), self.upright)
            if self.crossheads:
                m.box(
                    (u - 0.008, u + 0.008), (-0.095, 0.095), (top + 0.06, top + 0.072), BLACK_IRON
                )
                for v in (-0.095, 0.087):
                    m.box((u - 0.005, u + 0.005), (v, v + 0.008), (0.05, top + 0.06), STEEL)
        if self.beams:
            self._beams(m, z, top)
        if self.flywheel:
            for v in (0.096, -0.096):
                _disc(m, -0.03, z + 0.02, 0.09, v, BLACK_IRON)
                _disc(m, -0.03, z + 0.02, 0.025, v * 1.01, BRASS)
        if self.rack:
            for v in (0.096, -0.096):
                _disc(m, 0.0, 0.065, 0.065, v, STEEL, teeth=True)
        if self.slanted:
            (front, size), *_ = self.wheels
            _slant(m, (-h + 0.12, z + 0.04), (front, size), 0.03, 0.09, self.front or self.boiler)
        self._fine(m, z, top)
        return m

    def _fine(self, m: Model, z: float, top: float) -> None:
        """Spoked wheels, hoops round the boiler, the chimney's dark mouth, the firehole door
        at the back and the grain of a wooden frame."""
        h, r = self.half, self.radius
        for u, size in self.wheels:
            parts.spoked_wheel(
                m,
                u,
                size,
                0.075,
                self.wheel,
                spokes=12 if size > 0.06 else self.spokes,
                tyre=self.tyre,
            )
        start = -h + 0.07 + (0.08 if self.firebox is not None else 0.0)
        end = h - (0.1 if self.front is not None else 0.02)
        hoops = self.hoops or shade(self.boiler, 0.7)
        with m.fine():
            for k in range(1, self.hoop_count + 1):
                at = start + (end - start) * k / (self.hoop_count + 1)
                m.cylinder_along((at - 0.004, at + 0.004), r + 0.003, z, hoops)
            u = self.chimney[0]
            if self.crown:
                for k in range(8):
                    angle = 2 * math.pi * k / 8
                    begin = len(m.faces)
                    m.spire(
                        u, 0.006, (self.chimney_top, self.chimney_top + 0.03), self.chimney_color
                    )
                    reach = self.flue + 0.005
                    m.move(begin, reach * math.cos(angle), reach * math.sin(angle))
            mouth = self.flue + 0.001
            parts.roof_plate(
                m, (u - mouth, u + mouth), (-mouth, mouth), self.chimney_top + 0.0006, COAL
            )
            parts.front_ring(m, -h + 0.07 - 0.0006, z - r * 0.3, r * 0.35, BLACK_IRON, 0.0, 10)
        if self.frame in (WOOD, OAK):
            parts.lines(m, (-h, h), (0.045, 0.075), 0.0656, shade(self.frame, 0.7), 1)
        if self.boiler in (LAGGING, OAK):
            _planks(m, (start, end), r, z, self.boiler)

    def _tubes(self, m: Model, u: tuple[float, float], z: float, color: RGB) -> None:
        """The Rocket's boiler as the museum shows it, cut open along the top: black inside, with
        the copper fire tubes running through it."""
        r = self.radius
        m.box(u, (-r * 0.8, r * 0.8), (z + r * 0.5, z + r + 0.008), BLACK_IRON)
        with m.fine():
            for k in range(5):
                v = r * 0.6 * (k - 2) / 2
                m.box(u, (v - 0.003, v + 0.003), (z + r + 0.008, z + r + 0.012), color)

    def _beams(self, m: Model, z: float, top: float) -> None:
        """Puffing Billy's cylinders standing at the boiler's sides, working beams pivoted at the
        back that drive the wheels through rods hanging from their front ends."""
        h = self.half
        high = top + 0.08
        for v0, v1 in ((0.07, 0.092), (-0.092, -0.07)):
            m.box((0.0, 0.045), (v0, v1), (z - 0.03, high - 0.01), BLACK_IRON)
            m.box((-h + 0.07, 0.17), (v0 + 0.004, v1 - 0.004), (high, high + 0.016), FRAME)
            m.box((-h + 0.07, -h + 0.085), (v0 + 0.004, v1 - 0.004), (z, high), FRAME)
            m.box((0.155, 0.165), (v0 + 0.006, v1 - 0.006), (0.05, high), STEEL)


@dataclass(frozen=True)
class Steam:
    """A steam engine with a cab: driving wheels in one group, or two each with its own
    cylinders for an articulated engine, small wheels ahead and behind, and a boiler with a
    smokebox, chimney and domes. A tank engine carries its water in tanks along the boiler and its
    coal in a bunker behind the cab instead of a tender; a streamlined one is cased in a smooth
    shroud with stripes along it. The first engines stand open behind the boiler, with only a
    footplate and a rail where later ones have a cab."""

    half: float
    body: RGB
    roof: RGB
    drivers: tuple[tuple[float, ...], ...]
    driver: float
    wheel: RGB
    leading: tuple[float, ...] = ()
    trailing: tuple[float, ...] = ()
    radius: float = 0.08
    cab: float = 0.22
    cab_color: RGB | None = None
    trim: RGB = BRASS
    smokebox: RGB = SMOKEBOX
    domes: tuple[float, ...] = (0.0,)
    stack: float = 0.09
    balloon: bool = False
    cowcatcher: RGB | None = None
    headlight: RGB | None = None
    deflectors: bool = False
    tanks: bool = False
    shroud: RGB | None = None
    open_cab: bool = False
    beam: RGB = CAB_RED
    lead: float = 0.03
    dome: tuple[float, float] = (0.03, 0.055)
    dome_color: RGB | None = None
    bands: int = 1
    smokebox_length: float = 0.08
    cylinder: RGB | None = None
    firebox: RGB | None = None
    sand: bool = True

    @property
    def middle(self) -> float:
        """How high the middle of the boiler is."""
        return 0.08 + self.radius

    @property
    def chimney(self) -> tuple[float, float]:
        top = self.middle + self.radius
        front = self.half - 0.04
        if self.shroud is not None:
            return front - 0.145, top + 0.025
        return front - 0.07 - (0.03 if self.headlight is not None else 0.0), top + self.stack

    def model(self) -> Model:
        m = Model()
        h, r, z = self.half, self.radius, self.middle
        front, top = h - 0.04, z + r
        cab_front = -h + self.cab
        m.box((-h, front), (-0.07, 0.07), (0.035, 0.08), FRAME)
        m.wheels([*self.leading, *self.trailing], self.lead, FRAME, 0.075)
        for group in self.drivers:
            m.wheels(list(group), self.driver, self.wheel, 0.075)
            if len(group) > 1:
                _side_rods(m, (min(group), max(group)), self.driver)
            c = max(group) + self.driver + 0.045
            d = self.driver
            with m.coarse():
                m.box((c - 0.035, c + 0.035), (-0.1, 0.1), (d - 0.02, d + 0.03), self.smokebox)
            self._cylinders(m, c, d)
        m.box((cab_front, front), (-0.095, 0.095), (0.08, 0.09), FRAME)
        if self.shroud is not None:
            self._shroud(m, front, top, self.shroud)
        else:
            self._boiler(m, front, top)
        if self.cowcatcher is not None:
            m.nose((front, h + 0.03), (0.0, 0.012, 0.075), 0.075, self.cowcatcher)
        else:
            with m.coarse():
                m.box((front, h), (-0.085, 0.085), (0.01, 0.075), self.beam)
            with m.fine():
                m.box((front, h), (-0.095, 0.095), (0.045, 0.085), self.beam)
                m.box((front, front + 0.012), (-0.06, 0.06), (0.02, 0.045), FRAME)
        if self.tanks:
            self._tanks(m, cab_front, front, top)
        elif self.open_cab:
            self._footplate(m, -h, cab_front)
        else:
            self._cab(m, -h, cab_front, top + (0.04 if self.shroud else 0.05))
        self._fine(m, front, top)
        if keeping_fine():
            with m.finest():
                self._finest(m, front, top)
        return m

    def _finest(self, m: Model, front: float, top: float) -> None:
        """What shows only at the closest zooms: the valve gear and brake blocks by the driving
        wheels, sand pipes down to the rails, steam pipes from the smokebox to the cylinders, a
        feed pipe and clack valves on the boiler, the smokebox door's dart and hinges, a whistle,
        grab irons by the cab and the reversing rod along the running plate."""
        h, r, z = self.half, self.radius, self.middle
        cab_front = -h + self.cab
        d = self.driver
        dark = shade(FRAME, 1.4)
        for group in self.drivers:
            c = max(group) + d + 0.045
            link = c - 0.13
            for v in (0.104, -0.104):
                parts.bar(m, (link, d), (link, d + 0.05), 0.009, v, dark)
                parts.bar(m, (link, d + 0.04), (c - 0.04, d + 0.035), 0.005, v * 1.01, STEEL)
                parts.bar(m, (c - 0.045, d - 0.012), (c - 0.045, d + 0.04), 0.005, v * 1.02, STEEL)
                crank = group[len(group) // 2]
                parts.bar(m, (crank + d * 0.3, d + d * 0.2), (link, d + 0.004), 0.005, v, STEEL)
                if self.cab and not self.tanks:
                    parts.bar(m, (cab_front + 0.01, 0.097), (link, d + 0.05), 0.004, v, STEEL)
            for u in group:
                for v in (0.076, -0.076):
                    m.box(
                        (u - d - 0.009, u - d + 0.003),
                        (v - 0.008, v + 0.008),
                        (d * 0.6, d * 1.4),
                        dark,
                    )
            lead = max(group)
            for v in (0.092, -0.092):
                parts.bar(m, (lead + d * 0.8, 0.08), (lead + d + 0.004, 0.012), 0.004, v, dark)
        if self.shroud is not None:
            return
        for side in (1.0, -1.0):
            c = max(self.drivers[0]) + d + 0.045
            parts.bar(
                m,
                (front - 0.07, z - r * 0.3),
                (c + 0.01, d + 0.05),
                0.016,
                side * (r + 0.008),
                self.smokebox,
            )
            begin = len(m.faces)
            m.cylinder_along((cab_front + 0.02, front - 0.13), 0.0045, z - r * 0.35, COPPER)
            m.move(begin, 0.0, side * (r * 0.94 + 0.004))
            m.box(
                (front - 0.16, front - 0.142),
                tuple(sorted((side * r * 0.9, side * (r + 0.008)))),
                (z + r * 0.05, z + r * 0.3),
                BRASS,
            )
        parts.front_plate(m, front + 0.0018, (-0.03, 0.03), (z - 0.003, z + 0.003), STEEL)
        parts.front_plate(m, front + 0.0018, (-0.003, 0.003), (z - 0.03, z + 0.03), STEEL)
        for at in (z - r * 0.4, z + r * 0.4):
            parts.front_plate(
                m,
                front + 0.0016,
                (-r * 0.75, -r * 0.05),
                (at - 0.004, at + 0.004),
                shade(self.smokebox, 0.7),
            )
        if self.open_cab or self.tanks:
            return
        m.cylinder_up(cab_front + 0.012, 0.005, (top - 0.004, top + 0.045), BRASS)
        m.cylinder_up(cab_front + 0.012, 0.009, (top + 0.04, top + 0.052), BRASS)
        high = top + 0.05
        for v in (0.0916, -0.0916):
            parts.bar(
                m,
                (cab_front - 0.012, 0.1),
                (cab_front - 0.012, high - 0.03),
                0.004,
                v,
                parts.BRIGHT,
            )
            m.box((cab_front + 0.02, cab_front + 0.06), (v * 0.8, v * 0.88), (0.09, 0.105), dark)

    def _fine(self, m: Model, front: float, top: float) -> None:
        """Spoked wheels, rods and valve gear, the smokebox door, lamps and buffers, rails and
        pipes along the boiler and the cab's windows and steps."""
        h, r, z = self.half, self.radius, self.middle
        cab_front = -h + self.cab
        for group in self.drivers:
            for u in group:
                parts.spoked_wheel(m, u, self.driver, 0.075, self.wheel, weight=True)
            c = max(group) + self.driver + 0.045
            d = self.driver
            crank = (group[len(group) // 2] + d * 0.45, d - d * 0.25)
            with m.fine():
                for v in (0.101, -0.101):
                    parts.bar(
                        m, (c - 0.035, d + 0.005), (c - 0.035 - 0.07, d + 0.005), 0.006, v, STEEL
                    )
                    parts.bar(m, (c - 0.08, d + 0.004), crank, 0.012, v * 1.04, STEEL)
                    parts.bar(m, (c - 0.02, d + 0.03), (c - 0.09, d - 0.02), 0.005, v * 1.06, BRASS)
        for u in (*self.leading, *self.trailing):
            parts.spoked_wheel(m, u, self.lead, 0.075, FRAME, spokes=6 if self.lead < 0.04 else 10)
        if self.shroud is None:
            with m.fine():
                parts.front_ring(
                    m, front + 0.0006, z, min(r * 0.62, 0.05), shade(self.smokebox, 1.25)
                )
                parts.front_ring(m, front + 0.0012, z, 0.008, BRASS, 0.0, 8)
                parts.front_plate(m, front + 0.0012, (-0.02, 0.02), (z - 0.045, z - 0.03), BRASS)
            parts.handrail(m, (cab_front + 0.02, front - 0.06), z + r * 0.55, r + 0.008, STEEL)
            parts.rivets(m, (cab_front, front - 0.04), 0.087, 0.0956, shade(FRAME, 1.8), 10)
            with m.fine():
                for v in (0.096, -0.096):
                    parts.plate(m, (cab_front, front), (0.081, 0.087), v * 1.005, self.trim)
                u = cab_front + (0.11 if self.firebox else 0.04)
                m.cylinder_up(u, 0.008, (top - 0.006, top + 0.03), BRASS)
                m.cylinder_up(u + 0.025, 0.008, (top - 0.006, top + 0.03), BRASS)
                m.cylinder_up(u - 0.015, 0.005, (top + 0.04, top + 0.075), BRASS)
                m.box((u - 0.022, u - 0.008), (-0.004, 0.004), (top - 0.005, top + 0.04), BRASS)
                u, high = self.chimney
                parts.roof_plate(m, (u - 0.02, u + 0.02), (-0.02, 0.02), high + 0.0006, COAL)
                if self.bands == 1:
                    for at in (cab_front + 0.12, (cab_front + front) / 2 - 0.02):
                        m.cylinder_along((at - 0.005, at + 0.005), r + 0.004, z, self.trim)
        else:
            with m.fine():
                parts.front_ring(m, h + 0.0206, 0.05, 0.016, parts.LAMP, 0.0, 10)
        if self.cowcatcher is None:
            parts.buffers(m, h, 0.045, beam=None)
            parts.lamp(m, h + 0.003, 0.09, 0.045)
            parts.lamp(m, h + 0.003, 0.09, -0.045)
        else:
            with m.fine():
                for k in range(5):
                    v = -0.06 + 0.03 * k
                    m.slope(
                        (front + 0.004, 0.07),
                        (h + 0.028, 0.012),
                        (v - 0.004, v + 0.004),
                        shade(self.cowcatcher, 0.75),
                    )
        parts.buffers(m, -h, 0.05, beam=FRAME)

    def _cylinders(self, m: Model, c: float, d: float) -> None:
        """Up close, a round steam cylinder on each side ahead of the driving wheels, a steam
        chest on top of it, and a saddle across between them."""
        color = self.cylinder or self.smokebox
        with m.fine():
            for v in (0.088, -0.088):
                start = len(m.faces)
                m.cylinder_along((c - 0.04, c + 0.04), 0.03, d + 0.005, color, shade(color, 1.35))
                m.move(start, 0.0, v)
                m.box((c - 0.032, c + 0.032), (v - 0.018, v + 0.018), (d + 0.03, d + 0.052), color)
            m.box((c - 0.03, c + 0.03), (-0.06, 0.06), (d - 0.01, 0.08), self.smokebox)

    def _boiler(self, m: Model, front: float, top: float) -> None:
        r, z = self.radius, self.middle
        cab_front = -self.half + self.cab
        back = front - 0.03 - self.smokebox_length
        start = cab_front - 0.01
        if self.firebox is not None:
            start = cab_front + 0.07
            m.cylinder_along((cab_front - 0.01, start), r + 0.006, z + 0.006, self.firebox)
        m.cylinder_along((start, back), r, z, self.body)
        if self.body == LAGGING:
            _planks(m, (start, back), r, z, self.body)
        m.cylinder_along((back, front - 0.03), r + 0.003, z, self.smokebox, self.smokebox)
        with m.coarse():
            m.box((front - 0.03, front), (-0.065, 0.065), (0.08, z + 0.05), self.smokebox)
        with m.fine():
            m.cylinder_along(
                (front - 0.035, front), r + 0.003, z, self.smokebox, shade(self.smokebox, 0.85)
            )
            m.box((back, front - 0.02), (-0.05, 0.05), (0.08, z - r * 0.6), self.smokebox)
        if self.bands == 1:
            _bands(m, (start + 0.06,), r, z, self.trim)
        else:
            step = (back - start - 0.04) / (self.bands - 1)
            _bands(m, tuple(start + 0.02 + step * k for k in range(self.bands)), r, z, self.trim)
        radius, height = self.dome
        for u in self.domes:
            _dome(m, u, radius, top - 0.01, height, self.dome_color or self.trim)
        u, high = self.chimney
        if self.sand and self.domes and u - self.domes[0] > 0.15:
            with m.fine():
                _dome(m, (u + self.domes[0]) / 2 - 0.01, 0.022, top - 0.01, 0.04, self.body)
        if self.balloon:
            m.cylinder_up(u, 0.022, (top - 0.01, high - 0.04), BLACK_IRON)
            with m.coarse():
                m.box(
                    (u - 0.035, u + 0.035), (-0.035, 0.035), (high - 0.04, high - 0.015), BLACK_IRON
                )
                m.box((u - 0.045, u + 0.045), (-0.045, 0.045), (high - 0.015, high), BLACK_IRON)
            with m.fine():
                for k in range(5):
                    m.cylinder_up(
                        u,
                        0.026 + 0.0045 * k,
                        (high - 0.04 + 0.006 * k, high - 0.034 + 0.006 * k),
                        BLACK_IRON,
                    )
                m.cylinder_up(u, 0.047, (high - 0.01, high), BLACK_IRON)
        else:
            m.cylinder_up(u, 0.026, (top - 0.01, high - 0.015), BLACK_IRON)
            with m.coarse():
                m.box((u - 0.032, u + 0.032), (-0.032, 0.032), (high - 0.015, high), BLACK_IRON)
            with m.fine():
                m.cylinder_up(u, 0.036, (top - 0.012, top + 0.006), BLACK_IRON)
                m.cylinder_up(u, 0.03, (high - 0.015, high - 0.006), BLACK_IRON)
                m.cylinder_up(u, 0.035, (high - 0.006, high), BLACK_IRON)
        if self.headlight is not None:
            m.box(
                (front - 0.045, front - 0.005),
                (-0.025, 0.025),
                (top - 0.005, top + 0.045),
                self.headlight,
            )
            m.box((front - 0.005, front), (-0.016, 0.016), (top + 0.005, top + 0.035), LIT_WINDOW)
        if self.deflectors:
            for v in ((r + 0.004, r + 0.012), (-r - 0.012, -r - 0.004)):
                m.box((front - 0.16, front - 0.04), v, (0.1, top + 0.0), self.smokebox)

    def _shroud(self, m: Model, front: float, top: float, stripe: RGB) -> None:
        h, z = self.half, self.middle
        cab_front = -h + self.cab
        nose = front - 0.12
        m.box((cab_front, nose), (-0.092, 0.092), (0.07, top + 0.015), self.body)
        with m.coarse():
            m.nose((nose, h + 0.02), (0.02, 0.09, top + 0.015), 0.092, self.body)
        with m.fine():
            m.snout((nose, h + 0.02), (0.07, 0.09, top + 0.015), 0.092, self.body)
        for low in (z + 0.012, top - 0.035):
            m.panel((cab_front, nose), (low, low + 0.012), 0.0925, stripe)
        u, high = self.chimney
        m.box((u - 0.015, u + 0.015), (-0.02, 0.02), (top + 0.015, high - 0.003), BLACK_IRON)

    def _tanks(self, m: Model, cab_front: float, front: float, top: float) -> None:
        h, z = self.half, self.middle
        bunker = -h + 0.07
        m.body((-h, bunker), (-0.088, 0.088), (0.08, top - 0.03), self.body)
        m.box((-h + 0.01, bunker - 0.005), (-0.075, 0.075), (top - 0.03, top - 0.01), COAL)
        self._cab(m, bunker, cab_front, top + 0.05)
        m.body((cab_front, front - 0.12), (-0.092, 0.092), (0.08, z + 0.035), self.body)
        m.panel(
            (cab_front + 0.015, front - 0.135),
            (z - 0.035, z + 0.02),
            0.0925,
            shade(self.body, 0.85),
        )
        with m.fine():
            for v in (0.0926, -0.0926):
                parts.plate(
                    m, (cab_front + 0.02, front - 0.14), (z - 0.03, z - 0.024), v, self.trim
                )
                parts.plate(
                    m, (cab_front + 0.02, front - 0.14), (z + 0.009, z + 0.015), v, self.trim
                )
            m.cylinder_up(front - 0.15, 0.012, (z + 0.035, z + 0.045), BRASS)
            parts.lumps(m, (-h + 0.015, bunker - 0.01), (-0.07, 0.07), top - 0.01, COAL)

    def _footplate(self, m: Model, back: float, front: float) -> None:
        """An open footplate behind the boiler: a floor, and up close a rail round it on posts."""
        m.box((back, front), (-0.095, 0.095), (0.08, 0.09), FRAME)
        with m.fine():
            rail = 0.17
            for v in (0.088, -0.088):
                for u in (back + 0.008, (back + front) / 2):
                    m.box((u - 0.004, u + 0.004), (v - 0.004, v + 0.004), (0.09, rail), BLACK_IRON)
                m.box(
                    (back + 0.004, front - 0.01),
                    (v - 0.003, v + 0.003),
                    (rail, rail + 0.006),
                    BLACK_IRON,
                )
            m.box((back + 0.004, back + 0.012), (-0.088, 0.088), (rail, rail + 0.006), BLACK_IRON)

    def _cab(self, m: Model, back: float, front: float, high: float) -> None:
        color = self.cab_color or self.body
        m.body((back, front), (-0.09, 0.09), (0.08, high), color)
        m.panel((back + 0.03, front - 0.03), (high - 0.1, high - 0.035), 0.0905, WINDOW)
        with m.coarse():
            m.box((back - 0.02, front + 0.02), (-0.1, 0.1), (high, high + 0.02), self.roof)
        with m.fine():
            m.vault((back - 0.025, front + 0.02), 0.104, (high + 0.006, high + 0.03), self.roof)
            m.box((back, front), (-0.09, 0.09), (high, high + 0.006), shade(self.roof, 0.8))
        window = (back + 0.03, front - 0.03)
        parts.window_frames(m, window, (high - 0.1, high - 0.035), 0.0906, self.trim, WINDOW, 2)
        with m.fine():
            for v in (0.0906, -0.0906):
                parts.plate(m, (back + 0.01, front - 0.01), (0.1, 0.106), v, self.trim)
                parts.plate(m, (back + 0.01, back + 0.016), (0.1, high - 0.01), v, self.trim)
                parts.plate(m, (front - 0.016, front - 0.01), (0.1, high - 0.01), v, self.trim)
                parts.plate(
                    m, (back + 0.03, front - 0.03), (0.13, 0.15), v * 1.0003, shade(color, 0.85)
                )
            for v in (-0.055, 0.055):
                parts.front_ring(m, front + 0.0006, high - 0.05, 0.02, self.trim, v, 12)
                parts.front_ring(m, front + 0.0012, high - 0.05, 0.015, WINDOW, v, 12)
            for v in ((0.09, 0.104), (-0.104, -0.09)):
                m.box((back + 0.02, back + 0.05), v, (0.04, 0.046), FRAME)


@dataclass(frozen=True)
class Tender:
    """A tender behind a steam engine on wheels at `wheels`: its sides `top` high, lined in
    `trim`, heaped with its load, or for the first engines a low open wagon with a water barrel
    on it, or for a streamlined engine cased to match it."""

    half: float
    body: RGB
    wheels: tuple[float, ...]
    top: float = 0.22
    trim: RGB | None = BRASS
    trim_z: tuple[float, float] | None = None
    load: RGB = COAL
    barrel: RGB | None = None
    shroud: bool = False
    width: float = 0.085
    wheel: RGB = FRAME
    tyre: RGB = parts.TYRE
    tank: RGB | None = None
    slats: bool = False

    def model(self) -> Model:
        m = Model()
        h, w, top = self.half, self.width, self.top
        m.box((-h, h), (-0.07, 0.07), (0.035, 0.06), FRAME)
        m.wheels(list(self.wheels), 0.035, self.wheel, 0.075)
        if self.slats:
            self._slatted(m)
        else:
            m.body((-h, h), (-w, w), (0.06, top), self.body)
        if self.trim is not None:
            low, high = self.trim_z or (top - 0.035, top - 0.022)
            m.panel((-h + 0.02, h - 0.02), (low, high), w + 0.0005, self.trim)
        if self.shroud:
            m.box(
                (-h + 0.01, h - 0.01),
                (-w + 0.01, w - 0.01),
                (top, top + 0.02),
                shade(self.body, 0.85),
            )
        elif self.tank is not None:
            m.box((-h + 0.02, 0.06), (-w + 0.01, w - 0.01), (top, top + 0.09), self.tank)
            m.box((0.08, h - 0.02), (-w + 0.015, w - 0.015), (top, top + 0.03), self.load)
        elif self.barrel is not None:
            m.cylinder_along(
                (-h + 0.02, 0.05), 0.048, top + 0.048, self.barrel, shade(self.barrel, 0.75)
            )
            if not self.slats:
                m.box((0.07, h - 0.02), (-w + 0.015, w - 0.015), (top, top + 0.035), self.load)
        else:
            with m.coarse():
                m.box((-h + 0.03, h - 0.03), (-w + 0.015, w - 0.015), (top, top + 0.03), self.load)
                m.box((-0.5 * h, 0.5 * h), (-0.05, 0.05), (top + 0.03, top + 0.05), self.load)
        self._fine(m)
        if keeping_fine() and not self.shroud:
            with m.finest():
                self._finest(m)
        return m

    def _finest(self, m: Model) -> None:
        """What shows only at the closest zooms: grab irons at the front corners, a handbrake
        column and a tool box up front, and the brake blocks at the wheels."""
        h, w, top = self.half, self.width, self.top
        for v in (w + 0.0012, -w - 0.0012):
            parts.bar(m, (h - 0.012, 0.08), (h - 0.012, top - 0.01), 0.004, v, parts.BRIGHT)
        if self.barrel is None and self.tank is None and not self.slats:
            m.cylinder_up(h - 0.025, 0.006, (top - 0.03, top + 0.03), BLACK_IRON)
            m.cylinder_up(h - 0.025, 0.012, (top + 0.03, top + 0.034), BLACK_IRON)
            m.box(
                (h - 0.05, h - 0.03),
                (-w + 0.01, -w + 0.05),
                (top, top + 0.016),
                shade(self.body, 0.8),
            )
        for u in self.wheels:
            for v in (0.076, -0.076):
                m.box(
                    (u - 0.044, u - 0.034), (v - 0.007, v + 0.007), (0.02, 0.05), shade(FRAME, 1.4)
                )

    def _slatted(self, m: Model) -> None:
        """An open wagon of the first years: a flat bed with low sides of posts and rails, a
        solid box far out."""
        h, w, top = self.half, self.width, self.top
        with m.coarse():
            m.box((-h, h), (-w, w), (0.06, top), self.body)
        with m.fine():
            m.box((-h, h), (-w, w), (0.06, 0.08), self.body)
            m.box((-h + 0.01, h - 0.01), (-w + 0.01, w - 0.01), (0.08, top - 0.014), FRAME)
            for v in (w - 0.008, -w):
                m.box((-h, h), (v, v + 0.008), (top - 0.012, top), self.body)
                m.box((-h, h), (v, v + 0.008), (0.08, 0.088), self.body)
                posts = round(2 * h / 0.06)
                for k in range(posts + 1):
                    u = -h + 0.006 + (2 * h - 0.012) * k / posts
                    m.box((u - 0.006, u + 0.006), (v, v + 0.008), (0.08, top), self.body)
            for u in (h - 0.008, -h):
                m.box((u, u + 0.008), (-w, w), (top - 0.012, top), self.body)
                m.box((u, u + 0.008), (-w, w), (0.08, 0.088), self.body)

    def _fine(self, m: Model) -> None:
        """Spoked wheels behind axle boxes, rivets and lining, steps, the lumps of the load, a
        water hatch, and buffers, a lamp and a ladder at the back."""
        h, w, top = self.half, self.width, self.top
        for u in self.wheels:
            parts.spoked_wheel(m, u, 0.035, 0.075, self.wheel, spokes=8, tyre=self.tyre)
            parts.axle_box(m, u, 0.035, 0.088)
        parts.rivets(m, (-h, h), top - 0.008, w + 0.0006, shade(self.body, 0.7), 12)
        with m.fine():
            for v in ((w, w + 0.012), (-w - 0.012, -w)):
                m.box((h - 0.03, h - 0.005), v, (0.03, 0.036), FRAME)
                m.box((h - 0.03, h - 0.005), v, (0.045, 0.051), FRAME)
        if self.shroud:
            return
        if self.tank is not None:
            parts.rivets(m, (-h + 0.02, 0.06), top + 0.08, w - 0.0094, shade(self.tank, 1.6), 6)
            return
        if self.barrel is not None:
            with m.fine():
                for u in (-h + 0.05, -h + 0.12, 0.02):
                    m.cylinder_along((u - 0.004, u + 0.004), 0.05, top + 0.048, BLACK_IRON)
            return
        load = (-h + 0.03, h - 0.03), (-w + 0.015, w - 0.015)
        with m.fine():
            m.box(*load, (top, top + 0.012), shade(self.load, 0.8))
            _coal_rails(m, (-h + 0.06, h), w, top, self.body)
        if self.load == LOG:
            _logs(m, (-h + 0.035, h - 0.035), top + 0.012, w - 0.015, 0.011)
        else:
            parts.lumps(m, *load, top + 0.012, self.load, 0.02, 0.055)
        with m.fine():
            m.box((-h + 0.035, -h + 0.075), (-0.03, 0.03), (top, top + 0.02), shade(self.body, 0.8))
            m.cylinder_up(-h + 0.055, 0.016, (top + 0.02, top + 0.028), BLACK_IRON)
        parts.buffers(m, -h, 0.047)
        parts.lamp(m, -h - 0.013, top - 0.04, 0.05, parts.TAIL_LAMP, 0.01)
        parts.end_ladder(m, -h, (0.07, top - 0.01))


def _coal_rails(m: Model, u: tuple[float, float], half: float, top: float, color: RGB) -> None:
    """Plates flaring out along both sides of a tender's top, holding the coal in."""
    if m.skipping:
        return
    (u0, u1), rise, out = u, 0.028, 0.014
    for side in (1.0, -1.0):
        low, high = side * half, side * (half + out)
        slant = math.hypot(rise, out)
        normal = (0.0, side * rise / slant, out / slant)
        corners = [(u0, low, top), (u1, low, top), (u1, high, top + rise), (u0, high, top + rise)]
        m.faces.append(Face(corners, color, normal))
        m.faces.append(Face(corners[::-1], shade(color, 0.7), (0.0, -normal[1], -normal[2])))
    m.faces.append(
        Face(
            [
                (u0, -half, top),
                (u0, half, top),
                (u0, half + out, top + rise),
                (u0, -half - out, top + rise),
            ],
            color,
            (-1.0, 0.0, 0.0),
        )
    )


def _logs(m: Model, u: tuple[float, float], z: float, half: float, radius: float) -> None:
    """Logs piled lying along the track from height `z`, rows narrowing up a pyramid."""
    with m.fine():
        count = max(1, round(half / radius))
        for row in range(count):
            many = count - row
            for k in range(many):
                v = (k - (many - 1) / 2) * 2 * radius
                start = len(m.faces)
                m.cylinder_along(u, radius, z + radius * (1 + 1.7 * row), LOG, LOG_END)
                m.move(start, 0.0, v)


def _running_gear(m: Model, half: float, bogies: tuple[float, ...], axles: int) -> None:
    """An underframe the length of the body, on bogies of `axles` axles each at `bogies`."""
    m.box((-half + 0.01, half - 0.01), (-0.075, 0.075), (0.07, 0.09), FRAME)
    span = 0.0375 * (axles - 1)
    for c in bogies:
        m.box((c - span - 0.05, c + span + 0.05), (-0.065, 0.065), (0.03, 0.07), FRAME)
        m.wheels([c - span + 0.075 * k for k in range(axles)], 0.032, FRAME, 0.075)


@dataclass(frozen=True)
class Unit:
    """A diesel or electric engine. Most have a body the full width of the frame with a cab at
    the front, and at the back too if `both`: flat with windows, or a sloped nose `nose` long
    coming down to `tip` at its end, curving for a long one; with passenger windows along it for
    a railcar or a high-speed train. A hood unit instead has a narrow hood with walkways round it
    and its cab near the front. Electrics carry pantographs; a maglev floats over its guideway on
    a skirt instead of wheels."""

    half: float
    body: RGB
    stripe: RGB
    bogies: tuple[float, ...] = ()
    axles: int = 2
    top: float = 0.27
    hood: bool = False
    nose: float = 0.0
    tip: float = 0.2
    both: bool = True
    ends: RGB | None = None
    band: RGB | None = None
    windows: int = 0
    vents: int = 0
    glass: RGB = WINDOW
    pantographs: tuple[float, ...] = ()
    floats: bool = False
    roof: RGB = ROOF_GREY

    @property
    def chimney(self) -> None:
        return None

    def model(self) -> Model:
        m = Model()
        h = self.half
        if self.floats:
            m.box((-h + 0.02, h - self.nose), (-0.07, 0.07), (0.0, 0.03), FRAME)
        else:
            _running_gear(m, h, self.bogies, self.axles)
        roof = self._hood(m) if self.hood else self._shell(m)
        for u in self.pantographs:
            with m.coarse():
                _pantograph(m, u, roof)
            parts.pantograph(m, u, roof, PANTOGRAPH)
        for k in range(0 if self.hood else self.vents):
            u = (k - (self.vents - 1) / 2) * 0.12
            m.box((u - 0.04, u + 0.04), (-0.05, 0.05), (roof, roof + 0.012), shade(self.roof, 0.7))
            parts.roof_fan(m, u, roof + 0.012, 0.03, shade(self.roof, 0.9))
        self._fine(m, roof)
        if keeping_fine() and not self.floats:
            with m.finest():
                self._finest(m, roof)
        return m

    def _finest(self, m: Model, roof: float) -> None:
        """What shows only at the closest zooms: brake blocks and coil springs on the bogies,
        sand boxes ahead of the wheels, grab rails at the ends and horns on the roof."""
        h = self.half
        span = 0.0375 * (self.axles - 1)
        dark = shade(FRAME, 1.4)
        for c in self.bogies:
            for k in range(self.axles):
                u = c - span + 0.075 * k
                for v in (0.076, -0.076):
                    m.box((u + 0.031, u + 0.04), (v - 0.007, v + 0.007), (0.016, 0.05), dark)
            for u in (c - span - 0.035, c + span + 0.035):
                for v in (0.09, -0.09):
                    for k in range(4):
                        low = 0.05 + 0.006 * k
                        m.box(
                            (u - 0.008, u + 0.008),
                            (v - 0.003, v + 0.003),
                            (low, low + 0.003),
                            STEEL,
                        )
        for end in (h, -h) if self.both or self.hood else (h,):
            out = 1.0 if end > 0 else -1.0
            for v in (0.07, -0.07):
                parts.front_plate(
                    m, end + out * 0.0024, (v - 0.003, v + 0.003), (0.1, 0.16), parts.BRIGHT
                )
        u = h - (0.16 if self.hood else self.nose + 0.06)
        for k, v in enumerate((-0.012, 0.0, 0.012)):
            begin = len(m.faces)
            m.cylinder_along((u - 0.02 - 0.006 * k, u + 0.005), 0.004, roof + 0.012, parts.BRIGHT)
            m.move(begin, 0.0, v)

    @property
    def diesel(self) -> bool:
        return not self.pantographs and not self.floats

    def _fine(self, m: Model, roof: float) -> None:
        """Bogies with axle boxes and solid wheels, a fuel tank under a diesel, buffers and
        lamps at the ends, handrails and steps by the cab doors, grilles along the sides, and
        roof gear: insulators and cable on an electric, exhaust stacks on a diesel."""
        h, top = self.half, self.top
        if self.floats:
            self._floating(m)
            return
        span = 0.0375 * (self.axles - 1)
        for c in self.bogies:
            for k in range(self.axles):
                parts.disc_wheel(m, c - span + 0.075 * k, 0.032, 0.075, FRAME)
            parts.bogie_frame(m, c, span + 0.045, 0.045, 0.088)
        if self.diesel and len(self.bogies) > 1:
            inner = min(self.bogies) + span + 0.06, max(self.bogies) - span - 0.06
            with m.fine():
                m.box(inner, (-0.06, 0.06), (0.04, 0.072), shade(FRAME, 1.3))
                for v in (0.0606, -0.0606):
                    parts.plate(
                        m, (inner[0] + 0.02, inner[0] + 0.035), (0.05, 0.062), v, shade(FRAME, 2.2)
                    )
        ends = (h, -h) if self.both or self.hood else (h,)
        for end in ends:
            out = 1.0 if end > 0 else -1.0
            if self.nose < 0.08:
                parts.buffers(m, end, 0.072, beam=shade(FRAME, 1.4) if not self.hood else None)
            parts.coupler(m, end, 0.06 if self.nose >= 0.08 else 0.04)
            tip = self.tip if self.nose else top - 0.09
            for v in (-0.05, 0.05):
                parts.lamp(
                    m, end + out * 0.0006, max(0.11, min(tip, 0.14)) - 0.01, v, parts.LAMP, 0.01
                )
        if not self.both and not self.hood:
            for v in (-0.05, 0.05):
                parts.lamp(m, -h - 0.012, 0.13, v, parts.TAIL_LAMP, 0.009)
        if self.hood:
            self._hood_fine(m)
            return
        front = h - self.nose
        back = -front if self.both else -h
        floor = 0.09
        for root in [front, back][: 2 if self.both else 1]:
            out = 1.0 if root > 0 else -1.0
            door = root - out * 0.11
            with m.fine():
                for v in (0.0857, -0.0857):
                    parts.plate(
                        m,
                        (door - 0.022, door + 0.022),
                        (floor + 0.045, top - 0.015),
                        v,
                        shade(self.body, 0.75),
                    )
                    parts.plate(
                        m,
                        (door - 0.016, door + 0.016),
                        (floor + 0.045, top - 0.03),
                        v * 1.007,
                        self.body,
                    )
                    parts.plate(
                        m,
                        (door - 0.012, door + 0.012),
                        (top - 0.08, top - 0.04),
                        v * 1.014,
                        self.glass,
                    )
                    parts.plate(
                        m, (door + 0.026, door + 0.03), (floor + 0.05, top - 0.04), v, parts.BRIGHT
                    )
                    m.box(
                        (door - 0.02, door + 0.02),
                        tuple(sorted((v, v * 1.15))),
                        (0.03, 0.036),
                        FRAME,
                    )
        if self.diesel and not self.windows:
            middle = (back + front) / 2
            parts.louvres(
                m, (middle - 0.1, middle + 0.06), (top - 0.12, top - 0.04), 0.0858, self.body, 6
            )
            if self.vents:
                parts.stack(m, middle + 0.12, (roof, roof + 0.02), 0.012, shade(self.roof, 0.6))
        elif not self.windows:
            parts.louvres(
                m,
                (back + 0.18, front - 0.18),
                (top - 0.09, top - 0.045),
                0.0858,
                self.body,
                8,
                upright=True,
            )
        if self.pantographs:
            with m.fine():
                low, high = min(self.pantographs), max(self.pantographs)
                m.box((low - 0.06, high + 0.06), (0.03, 0.036), (roof, roof + 0.006), parts.BRIGHT)
                for u in (
                    (low + 0.1, high - 0.1) if high - low > 0.25 else ((low + high) / 2 + 0.12,)
                ):
                    m.box(
                        (u - 0.04, u + 0.04),
                        (-0.05, 0.05),
                        (roof, roof + 0.022),
                        shade(self.roof, 0.85),
                    )
                    parts.lines(
                        m,
                        (u - 0.04, u + 0.04),
                        (roof, roof + 0.022),
                        0.0506,
                        shade(self.roof, 0.6),
                        2,
                    )
        else:
            with m.fine():
                for u in (front - 0.05,):
                    m.box((u - 0.008, u + 0.008), (-0.03, 0.03), (roof, roof + 0.012), parts.BRIGHT)

    def _hood_fine(self, m: Model) -> None:
        """A hood unit's railings along its walkways, steps at the corners, louvres and grilles
        down the hood and exhaust stacks on it."""
        h, top = self.half, self.top
        cab = (h - 0.22, h - 0.1)
        parts.handrail(m, (-h + 0.01, h - 0.01), 0.15, 0.084, self.stripe, posts=6)
        with m.fine():
            for u in (h - 0.025, -h + 0.025):
                for v in ((0.09, 0.1), (-0.1, -0.09)):
                    m.box((u - 0.02, u + 0.02), v, (0.04, 0.046), FRAME)
                    m.box((u - 0.02, u + 0.02), v, (0.065, 0.071), FRAME)
        parts.louvres(
            m, (-h + 0.03, -h + 0.13), (0.12, top - 0.035), 0.0556, self.body, 6, upright=True
        )
        parts.louvres(m, (-h + 0.18, cab[0] - 0.04), (0.16, top - 0.04), 0.0556, self.body, 4)
        with m.fine():
            parts.front_plate(
                m, -h + 0.0206, (-0.04, 0.04), (0.11, top - 0.04), shade(self.body, 0.4)
            )
            for k in range(5):
                v = -0.032 + 0.016 * k
                parts.front_plate(
                    m, -h + 0.0212, (v - 0.003, v + 0.003), (0.11, top - 0.04), self.body
                )
        for u in (cab[0] - 0.08, cab[0] - 0.15):
            parts.stack(m, u, (top - 0.025, top), 0.01, shade(FRAME, 1.2))
        with m.fine():
            m.box(
                (cab[0] + 0.04, cab[0] + 0.07),
                (-0.008, 0.008),
                (top + 0.012, top + 0.03),
                parts.BRIGHT,
            )
            parts.front_plate(
                m, cab[1] + 0.0046, (-0.05, -0.01), (top - 0.065, top - 0.025), WINDOW
            )
            parts.front_plate(m, cab[1] + 0.0046, (0.01, 0.05), (top - 0.065, top - 0.025), WINDOW)

    def _floating(self, m: Model) -> None:
        """A maglev's lit band of windows, its doors, a light strip down the skirt and lights in
        its nose."""
        h, top = self.half, self.top
        front = h - self.nose
        with m.fine():
            for v in (0.0857, -0.0857):
                parts.plate(m, (-h + 0.04, front), (0.045, 0.052), v, (120, 230, 240))
                for u in (-h + 0.18, front - 0.12):
                    parts.plate(
                        m, (u - 0.02, u + 0.02), (0.06, top - 0.02), v, shade(self.body, 0.82)
                    )
                    parts.plate(
                        m, (u - 0.014, u + 0.014), (top - 0.08, top - 0.035), v * 1.007, self.glass
                    )
            for v in (-0.04, 0.04):
                parts.front_ring(m, h + 0.0006, self.tip * 0.7, 0.008, parts.LAMP, v, 8)
            m.box(
                (-h + 0.06, front - 0.06),
                (-0.03, 0.03),
                (top + 0.015, top + 0.022),
                shade(self.roof, 0.85),
            )

    def _shell(self, m: Model) -> float:
        h, top = self.half, self.top
        floor = 0.02 if self.floats else 0.09
        front = h - self.nose
        back = -front if self.both else -h
        skirt = floor + 0.04
        m.box((back, front), (-0.085, 0.085), (floor, skirt), self.stripe)
        m.body((back, front), (-0.085, 0.085), (skirt, top), self.body)
        _roof(m, (back, front), 0.085, (top, top + 0.015), self.roof)
        if self.band is not None:
            m.panel((back, front), (top - 0.09, top - 0.03), 0.0855, self.band)
        for k in range(self.windows):
            u = back + 0.1 + (front - back - 0.2) * (k + 0.5) / self.windows
            m.panel((u - 0.025, u + 0.025), (top - 0.08, top - 0.035), 0.0856, self.glass)
        for root, end in [(front, h), (back, -h)][: 2 if self.both else 1]:
            self._end(m, root, end, floor)
        if not self.both:
            m.box((-h - 0.012, -h), (-0.045, 0.045), (floor + 0.02, top - 0.02), FRAME)
        return top + 0.015

    def _end(self, m: Model, root: float, end: float, floor: float) -> None:
        """One end of the body: a flat cab front with windows, or a sloped nose with a
        windscreen, and cab windows on its sides."""
        top, color = self.top, self.ends or self.body
        out = 1.0 if end > 0 else -1.0
        side = root - out * 0.08
        m.panel(
            (min(root, side) + 0.015, max(root, side) - 0.015),
            (top - 0.085, top - 0.035),
            0.0857,
            WINDOW,
        )
        if not self.nose:
            m.box((end - 0.004, end + 0.004), (-0.065, 0.065), (top - 0.08, top - 0.025), WINDOW)
            if self.ends is not None:
                m.box(
                    (end - out * 0.01 - 0.004, end - out * 0.01 + 0.004),
                    (-0.085, 0.085),
                    (floor, top - 0.09),
                    self.ends,
                )
            return
        if self.nose > 0.1:
            bend = root + (end - root) * 0.6
            high = top - (top - self.tip) * 0.35
            with m.coarse():
                m.nose((root, bend), (floor, high, top), 0.085, color)
                m.nose((bend, end), (floor, self.tip, high), 0.085, color)
            with m.fine():
                m.snout((root, end), (floor, self.tip, top), 0.085, color, WINDOW)
            reach = 0.08
            fall = (top - high) * reach / abs(bend - root)
        else:
            m.nose((root, end), (floor, self.tip, top), 0.085, color)
            reach = abs(end - root) * 0.6
            fall = (top - self.tip) * 0.6
        a = (root + out * 0.004, top + 0.004)
        b = (root + out * (reach + 0.004), top - fall + 0.004)
        if self.nose > 0.1:
            with m.coarse():
                m.slope(a, b, (-0.065, 0.065), WINDOW)
        else:
            m.slope(a, b, (-0.065, 0.065), WINDOW)

    def _hood(self, m: Model) -> float:
        h, top = self.half, self.top
        cab = (h - 0.22, h - 0.1)
        m.box((-h, h), (-0.09, 0.09), (0.09, 0.1), FRAME)
        m.body((h - 0.1, h - 0.01), (-0.05, 0.05), (0.1, top - 0.08), self.body)
        m.body(cab, (-0.085, 0.085), (0.1, top), self.body)
        _roof(m, (cab[0] - 0.01, cab[1] + 0.01), 0.09, (top, top + 0.012), self.roof)
        m.box((cab[1] - 0.004, cab[1] + 0.004), (-0.07, 0.07), (top - 0.07, top - 0.02), WINDOW)
        m.panel((cab[0] + 0.02, cab[1] - 0.02), (top - 0.07, top - 0.02), 0.0855, WINDOW)
        m.body((-h + 0.02, cab[0]), (-0.055, 0.055), (0.1, top - 0.025), self.body)
        m.panel((-h + 0.02, cab[0]), (top - 0.08, top - 0.055), 0.0555, self.stripe)
        m.panel((h - 0.1, h - 0.01), (0.15, 0.17), 0.0505, self.stripe)
        for k in range(self.vents):
            u = -h + 0.07 + k * 0.09
            m.box(
                (u - 0.03, u + 0.03),
                (-0.04, 0.04),
                (top - 0.025, top - 0.012),
                shade(self.body, 0.6),
            )
        return top + 0.012


@dataclass(frozen=True)
class Turned:
    """An engine turned to face backwards, like the second unit of a pair coupled back to back."""

    unit: Unit

    @property
    def half(self) -> float:
        return self.unit.half

    def model(self) -> Model:
        m = Model()
        m.faces = [
            Face(
                [(-u, -v, z) for u, v, z in face.corners],
                face.color,
                (-face.normal[0], -face.normal[1], face.normal[2]),
            )
            for face in self.unit.model().faces
        ]
        return m


def car(good: Good, full: bool) -> Model:
    """A car for a good on its frame and wheels, loaded or empty, its body by what it carries."""
    m = Model()
    h = CAR_HALF
    m.box((-h, h), (-0.07, 0.07), (0.035, 0.06), FRAME)
    m.wheels([h - 0.07, -h + 0.07], 0.035, FRAME, 0.075)
    _body_of(good)(m, good, full)
    _underframe(m)
    if keeping_fine():
        with m.finest():
            _car_finest(m)
    return m


def _car_finest(m: Model) -> None:
    """What shows on a car only at the closest zooms: brake blocks at the wheels, and grab irons
    and steps at its corners."""
    h = CAR_HALF
    dark = shade(FRAME, 1.4)
    for u in (h - 0.07, -h + 0.07):
        for v in (0.076, -0.076):
            m.box((u + 0.034, u + 0.043), (v - 0.007, v + 0.007), (0.016, 0.05), dark)
    for end in (h, -h):
        out = 1.0 if end > 0 else -1.0
        for v in (0.0815, -0.0815):
            parts.bar(
                m, (end - out * 0.012, 0.075), (end - out * 0.012, 0.13), 0.004, v, parts.BRIGHT
            )
            m.box(
                tuple(sorted((end - out * 0.03, end - out * 0.005))),
                (v * 0.95, v * 1.12),
                (0.03, 0.035),
                dark,
            )


def _underframe(m: Model) -> None:
    """A car's solid wheels behind axle boxes and springs, and buffers at both ends."""
    h = CAR_HALF
    for u in (h - 0.07, -h + 0.07):
        parts.disc_wheel(m, u, 0.035, 0.075, FRAME)
        parts.axle_box(m, u, 0.035, 0.088)
    for end in (h, -h):
        parts.buffers(m, end, 0.048, beam=None)


type CarBody = Callable[[Model, Good, bool], None]


def _body_of(good: Good) -> CarBody:
    if good in LIVESTOCK:
        return _stock_car
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
    m.body((-h, h), (-0.08, 0.08), (0.06, 0.24), body)
    _roof(m, (-h - 0.01, h + 0.01), 0.088, (0.24, 0.26), ROOF_GREY)
    with m.coarse():
        m.panel((-0.06, 0.06), (0.07, 0.22), 0.0805, shade(body, 0.7) if full else WINDOW)
    m.panel((-h + 0.01, h - 0.01), (0.2, 0.215), 0.0806, stripe)
    _van_fine(m, body, stripe, full)
    if good in REEFERS:
        with m.fine():
            for u in (-h + 0.04, h - 0.04):
                for v in (-0.045, 0.045):
                    m.box(
                        (u - 0.022, u + 0.022),
                        (v - 0.022, v + 0.022),
                        (0.26, 0.268),
                        shade(body, 0.9),
                    )


def _van_fine(m: Model, body: RGB, stripe: RGB, full: bool) -> None:
    """A van's planked sides, its sliding door on rails, open on its load of crates in the
    good's colours when full and shut when empty, a ladder and a brake wheel at the ends and
    ribs over the roof."""
    h = CAR_HALF
    parts.lines(m, (-h + 0.01, h - 0.01), (0.07, 0.195), 0.0805, shade(body, 0.8), 9, upright=True)
    with m.fine():
        for v in (0.0806, -0.0806):
            parts.plate(m, (-0.08, 0.1), (0.222, 0.228), v, FRAME)
            parts.plate(m, (-0.08, 0.1), (0.064, 0.07), v, FRAME)
            if full:
                parts.plate(m, (-0.06, 0.04), (0.072, 0.218), v, shade(body, 0.3))
                for k, (a, b) in enumerate(((-0.055, -0.02), (-0.015, 0.035))):
                    for z0, z1, tint in ((0.075, 0.13, stripe), (0.135, 0.18, shade(stripe, 0.8))):
                        out = v * (1.004 + 0.002 * k)
                        parts.plate(m, (a, b), (z0, z1), out, tint if k else shade(tint, 0.9))
                door = (0.04, 0.1)
            else:
                door = (-0.06, 0.06)
            out = v * 1.012
            parts.plate(m, door, (0.072, 0.218), out, shade(body, 0.85))
            parts.plate(m, (door[0] + 0.004, door[1] - 0.004), (0.14, 0.15), out * 1.003, FRAME)
            parts.plate(m, (-h + 0.02, -h + 0.07), (0.15, 0.19), v * 1.004, stripe)
            parts.plate(m, (-h + 0.028, -h + 0.062), (0.158, 0.182), v * 1.008, shade(stripe, 0.6))
    parts.end_ladder(m, h, (0.07, 0.235))
    parts.brake_wheel(m, -h, 0.2)
    with m.fine():
        for k in range(5):
            u = -h + 0.02 + k * (2 * h - 0.04) / 4
            m.box((u - 0.004, u + 0.004), (-0.088, 0.088), (0.26, 0.264), shade(ROOF_GREY, 0.85))


def _stock_car(m: Model, good: Good, full: bool) -> None:
    """A van for animals with slatted sides: the animals show through the gaps."""
    h = CAR_HALF
    body, animal = VANS[good]
    m.body((-h, h), (-0.08, 0.08), (0.06, 0.24), body)
    _roof(m, (-h - 0.01, h + 0.01), 0.088, (0.24, 0.26), ROOF_GREY)
    with m.coarse():
        m.panel((-h + 0.02, h - 0.02), (0.14, 0.2), 0.0805, animal if full else WINDOW)
    with m.fine():
        for v in (0.0806, -0.0806):
            parts.plate(m, (-h + 0.012, h - 0.012), (0.07, 0.228), v, shade(body, 0.3))
            if full:
                for k in range(5):
                    u = -h + 0.04 + k * (2 * h - 0.08) / 4
                    high = 0.15 if good == Good.CHICKENS else 0.19
                    parts.plate(m, (u - 0.025, u + 0.025), (0.09, high), v * 1.003, animal)
                    parts.plate(
                        m,
                        (u + 0.012, u + 0.03),
                        (high - 0.03, high + 0.012),
                        v * 1.006,
                        shade(animal, 0.85),
                    )
            for k in range(6):
                z = 0.08 + k * 0.026
                parts.plate(m, (-h + 0.012, h - 0.012), (z, z + 0.013), v * 1.01, body)
            for u in (-h + 0.012, -0.004, h - 0.022):
                parts.plate(m, (u, u + 0.012), (0.07, 0.228), v * 1.012, shade(body, 0.8))
    parts.brake_wheel(m, -h, 0.2)


def _hopper(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    body, load = HOPPERS[good]
    m.body((-h, h), (-0.08, 0.08), (0.06, 0.19), body)
    m.panel((-h + 0.02, h - 0.02), (0.08, 0.17), 0.0805, shade(body, 0.85))
    if full:
        with m.coarse():
            m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.19, 0.22), load)
            m.box((-0.1, 0.1), (-0.05, 0.05), (0.22, 0.24), load)
    _open_top(m, body, load if full else None, 0.19)


def _open_top(m: Model, body: RGB, load: RGB | None, top: float) -> None:
    """An open wagon's ribbed sides, its rim, a ladder at one end and a brake wheel at the
    other, and its load heaped in lumps when it carries one."""
    h = CAR_HALF
    parts.ribs(m, (-h + 0.01, h - 0.01), (0.062, top), 0.0805, shade(body, 1.15), 6)
    with m.fine():
        for v in (0.0806, -0.0806):
            parts.plate(m, (-h, h), (top - 0.012, top), v, shade(body, 1.2))
    parts.end_ladder(m, h, (0.07, top))
    parts.brake_wheel(m, -h, top - 0.02)
    if load is not None:
        with m.fine():
            m.box((-h + 0.012, h - 0.012), (-0.072, 0.072), (top - 0.03, top), shade(load, 0.75))
        parts.lumps(m, (-h + 0.015, h - 0.015), (-0.07, 0.07), top, load, 0.024, 0.04)


def _heap(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.13), DECK)
    m.panel((-h + 0.02, h - 0.02), (0.075, 0.12), 0.0805, shade(DECK, 0.8))
    if full:
        with m.coarse():
            m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.13, 0.17), HEAPS[good])
            m.box((-0.13, 0.13), (-0.05, 0.05), (0.17, 0.2), HEAPS[good])
    parts.lines(m, (-h + 0.02, h - 0.02), (0.075, 0.12), 0.0806, shade(DECK, 0.6), 2)
    parts.ribs(m, (-h + 0.01, h - 0.01), (0.062, 0.13), 0.0805, shade(DECK, 0.7), 4)
    parts.brake_wheel(m, -h, 0.12)
    if full:
        HEAP_LOADS.get(good, _lumped)(m, HEAPS[good])


def _lumped(m: Model, color: RGB) -> None:
    """A load heaped in lumps, such as potatoes."""
    h = CAR_HALF
    parts.lumps(m, (-h + 0.02, h - 0.02), (-0.07, 0.07), 0.13, color, 0.024, 0.04)


def _bales(m: Model, color: RGB) -> None:
    """Bales tied with bands, two layers: wool, cotton, flax, hops."""
    with m.fine():
        for row, z in enumerate((0.13, 0.172)):
            us = (-0.15, -0.05, 0.05, 0.15) if row == 0 else (-0.1, 0.0, 0.1)
            for u in us:
                for v in ((-0.068, -0.003), (0.003, 0.068)) if row == 0 else ((-0.035, 0.035),):
                    m.box((u - 0.045, u + 0.045), v, (z, z + 0.04), shade(color, 1.0 - 0.06 * row))
                    m.box(
                        (u - 0.006, u + 0.006),
                        (v[0] - 0.001, v[1] + 0.001),
                        (z - 0.001, z + 0.041),
                        shade(color, 0.6),
                    )


def _barrels(m: Model, color: RGB) -> None:
    """Barrels lying across the wagon, hooped in iron."""
    with m.fine():
        for u in (-0.17, -0.085, 0.0, 0.085, 0.17):
            for v in (-0.035, 0.035):
                start = len(m.faces)
                m.cylinder_up(0.0, 0.032, (0.13, 0.205), color)
                m.cylinder_up(0.0, 0.034, (0.145, 0.152), BLACK_IRON)
                m.cylinder_up(0.0, 0.034, (0.183, 0.19), BLACK_IRON)
                m.move(start, u, v)


def _bricks(m: Model, color: RGB) -> None:
    """Bricks stacked on pallets, the courses showing."""
    with m.fine():
        for u in (-0.12, 0.12):
            m.box((u - 0.1, u + 0.1), (-0.07, 0.07), (0.13, 0.14), DECK)
            m.box((u - 0.09, u + 0.09), (-0.065, 0.065), (0.14, 0.21), color)
            for k in range(1, 4):
                z = 0.14 + 0.07 * k / 4
                for v in (0.0656, -0.0656):
                    parts.plate(
                        m, (u - 0.09, u + 0.09), (z - 0.002, z + 0.002), v, shade(color, 0.7)
                    )


def _crated(m: Model, color: RGB) -> None:
    """Machines lashed down under sheets, dark straps over them."""
    with m.fine():
        m.box((-0.18, 0.02), (-0.065, 0.065), (0.13, 0.2), color)
        m.box((-0.12, -0.02), (-0.04, 0.04), (0.2, 0.23), shade(color, 0.85))
        m.box((0.06, 0.2), (-0.055, 0.055), (0.13, 0.18), shade(color, 1.15))
        m.cylinder_along((0.07, 0.19), 0.02, 0.2, STEEL)
        for u in (-0.15, -0.05, 0.12):
            m.box(
                (u - 0.005, u + 0.005),
                (-0.068, 0.068),
                (0.13, 0.233 if u < 0 else 0.183),
                BLACK_IRON,
            )


def _tubs(m: Model, color: RGB) -> None:
    """Grapes heaped in tubs."""
    with m.fine():
        for u in (-0.16, -0.055, 0.055, 0.16):
            for v in (-0.04, 0.04):
                start = len(m.faces)
                m.cylinder_up(0.0, 0.036, (0.13, 0.18), WOOD)
                m.cylinder_up(0.0, 0.03, (0.18, 0.195), color)
                m.move(start, u, v)


def _tank(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    color = TANKS[good]
    m.box((-h, h), (-0.07, 0.07), (0.06, 0.085), DECK)
    m.cylinder_along((-h + 0.01, h - 0.01), 0.08, 0.165, color)
    m.cylinder_up(0.0, 0.025, (0.24, 0.27), color)
    with m.fine():
        for u in (-h + 0.06, h - 0.06):
            m.cylinder_along((u - 0.005, u + 0.005), 0.083, 0.165, shade(color, 0.7))
        m.box((-0.06, 0.06), (-0.04, 0.04), (0.242, 0.246), STEEL)
        m.cylinder_up(0.0, 0.01, (0.27, 0.285), shade(color, 0.6))
        for v in (0.084, -0.084):
            parts.plate(m, (-0.11, -0.05), (0.13, 0.17), v, (240, 240, 236))
            parts.plate(m, (-0.1, -0.06), (0.137, 0.163), v * 1.005, shade(color, 0.8))
        for v in (-0.09, 0.09):
            m.box((-h + 0.01, h - 0.01), (v - 0.004, v + 0.004), (0.085, 0.09), STEEL)
    parts.ladder(m, 0.06, (0.085, 0.24), 0.083)
    parts.brake_wheel(m, -h, 0.12)


def _deck(m: Model) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.085), DECK)


def _car_carrier(m: Model, good: Good, full: bool) -> None:
    _deck(m)
    _stakes(m)
    if full:
        for u in (-0.12, 0.12):
            m.box((u - 0.09, u + 0.09), (-0.05, 0.05), (0.095, 0.14), AUTO_BODY)
            m.box((u - 0.05, u + 0.05), (-0.045, 0.045), (0.14, 0.18), WINDOW)
            with m.fine():
                m.box((u - 0.05, u + 0.05), (-0.046, 0.046), (0.176, 0.182), AUTO_BODY)
                for at in (u - 0.055, u + 0.055):
                    for v in (0.05, -0.05):
                        parts.ring(m, at, 0.1, 0.018, v * 1.01, BLACK_IRON, 10)
                        parts.ring(m, at, 0.1, 0.008, v * 1.02, STEEL, 8)
                parts.front_plate(m, u + 0.0906, (-0.04, -0.025), (0.12, 0.13), parts.LAMP)
                parts.front_plate(m, u + 0.0906, (0.025, 0.04), (0.12, 0.13), parts.LAMP)


def _stakes(m: Model) -> None:
    """Stakes standing up along a flat wagon's sides."""
    h = CAR_HALF
    with m.fine():
        for u in (-h + 0.02, 0.0, h - 0.02):
            for v in ((0.074, 0.082), (-0.082, -0.074)):
                m.box((u - 0.004, u + 0.004), v, (0.085, 0.13), FRAME)


def _tank_carrier(m: Model, good: Good, full: bool) -> None:
    _deck(m)
    if full:
        m.box((-0.2, 0.2), (-0.07, 0.07), (0.085, 0.15), TANK_HULL)
        m.box((-0.08, 0.08), (-0.05, 0.05), (0.15, 0.2), shade(TANK_HULL, 0.85))
        m.box((0.08, 0.26), (-0.012, 0.012), (0.165, 0.185), shade(TANK_HULL, 0.6))
        with m.fine():
            for v in (0.0706, -0.0706):
                parts.plate(m, (-0.2, 0.2), (0.085, 0.125), v, shade(TANK_HULL, 0.45))
                for u in (-0.15, -0.075, 0.0, 0.075, 0.15):
                    parts.ring(m, u, 0.104, 0.018, v * 1.01, shade(TANK_HULL, 0.8), 10)
            m.cylinder_up(-0.04, 0.012, (0.2, 0.215), shade(TANK_HULL, 0.7))


def _plane_carrier(m: Model, good: Good, full: bool) -> None:
    _deck(m)
    if full:
        m.box((-0.24, 0.24), (-0.03, 0.03), (0.09, 0.15), PLANE_BODY)
        m.box((-0.05, 0.05), (-0.085, 0.085), (0.12, 0.135), shade(PLANE_BODY, 0.85))
        m.box((-0.24, -0.17), (-0.012, 0.012), (0.15, 0.2), shade(PLANE_BODY, 0.75))
        with m.fine():
            for v in (0.0306, -0.0306):
                for k in range(6):
                    u = -0.14 + k * 0.045
                    parts.plate(m, (u - 0.008, u + 0.008), (0.125, 0.138), v, WINDOW)
                parts.plate(m, (0.18, 0.225), (0.125, 0.142), v, WINDOW)
                parts.plate(m, (-0.24, 0.24), (0.105, 0.11), v * 1.003, EXPRESS_BLUE)
            for v in (-0.06, 0.06):
                start = len(m.faces)
                m.cylinder_along((-0.03, 0.03), 0.014, 0.112, shade(PLANE_BODY, 0.7))
                m.move(start, 0.0, v)


def _boxcar(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    body, roof = {
        Good.GOODS: (BOXCAR, ROOF_GREY),
        Good.CANS: (BOXCAR, ROOF_GREY),
        Good.BANANAS: (BANANA_VAN, (240, 236, 220)),
        Good.BREAD: (BREAD_VAN, (236, 226, 204)),
    }[good]
    m.body((-h, h), (-0.08, 0.08), (0.06, 0.24), body)
    _roof(m, (-h - 0.01, h + 0.01), 0.088, (0.24, 0.26), roof)
    door = shade(body, 0.7) if full else WINDOW
    with m.coarse():
        m.panel((-0.06, 0.06), (0.07, 0.22), 0.0805, door)
    m.panel((-h + 0.01, h - 0.01), (0.07, 0.085), 0.0806, shade(body, 0.75))
    stripe = {Good.CANS: CANS_STRIPE, Good.BANANAS: (110, 150, 50), Good.BREAD: (236, 200, 140)}
    if good == Good.CANS:
        m.panel((-h + 0.01, h - 0.01), (0.2, 0.215), 0.0806, CANS_STRIPE)
    _van_fine(m, body, stripe.get(good, (190, 150, 100)), full)


def _grain_hopper(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    body = GRAIN_HOPPER if good in (Good.WHEAT, Good.RICE, Good.SOYBEANS) else SUGAR_HOPPER
    m.body((-h, h), (-0.08, 0.08), (0.06, 0.2), body)
    m.panel((-h + 0.02, h - 0.02), (0.08, 0.18), 0.0805, shade(body, 0.85))
    if good == Good.SUGAR:
        with m.coarse():
            m.box((-h + 0.01, h - 0.01), (-0.07, 0.07), (0.2, 0.235), shade(body, 0.95))
        with m.fine():
            m.vault((-h + 0.005, h - 0.005), 0.08, (0.2, 0.237), shade(body, 0.95))
            for u in (-0.15, -0.05, 0.05, 0.15):
                m.box((u - 0.025, u + 0.025), (-0.025, 0.025), (0.235, 0.243), shade(body, 0.75))
        parts.ribs(m, (-h + 0.01, h - 0.01), (0.062, 0.2), 0.0805, shade(body, 0.9), 6)
        parts.end_ladder(m, h, (0.07, 0.235))
        return
    if full:
        with m.coarse():
            m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.2, 0.22), GRAIN)
            m.box((-0.12, 0.12), (-0.05, 0.05), (0.22, 0.24), GRAIN)
    _open_top(m, body, GRAIN if full else None, 0.2)


def _fruit_wagon(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.13), DECK)
    m.panel((-h + 0.02, h - 0.02), (0.075, 0.12), 0.0805, shade(DECK, 0.8))
    parts.lines(m, (-h + 0.02, h - 0.02), (0.075, 0.12), 0.0806, shade(DECK, 0.6), 2)
    parts.brake_wheel(m, -h, 0.12)
    if full:
        fruit = {Good.APPLES: APPLE, Good.DATES: DATE, Good.ORANGES: ORANGE}[good]
        with m.coarse():
            m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.13, 0.16), fruit)
            m.box((-0.13, 0.13), (-0.05, 0.05), (0.16, 0.18), fruit)
            if good == Good.APPLES:
                for u in (-0.15, -0.05, 0.06, 0.14):
                    m.box((u - 0.012, u + 0.012), (-0.04, -0.02), (0.16, 0.175), APPLE_LIGHT)
        with m.fine():
            for row, z in enumerate((0.13, 0.172)):
                for u in (-0.165, -0.055, 0.055, 0.165) if row == 0 else (-0.11, 0.0, 0.11):
                    for v in ((-0.068, -0.003), (0.003, 0.068)):
                        m.box((u - 0.05, u + 0.05), v, (z, z + 0.04), shade(WOOD, 1.1 - 0.1 * row))
                        m.box(
                            (u - 0.044, u + 0.044),
                            (v[0] + 0.006, v[1] - 0.006),
                            (z + 0.04, z + 0.046),
                            fruit,
                        )
                        m.box(
                            (u - 0.03, u + 0.0),
                            (v[0] + 0.012, v[1] - 0.03),
                            (z + 0.046, z + 0.05),
                            shade(fruit, 1.3),
                        )


def _coal_hopper(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.body((-h, h), (-0.08, 0.08), (0.06, 0.19), HOPPER)
    m.panel((-h + 0.02, h - 0.02), (0.08, 0.17), 0.0805, shade(HOPPER, 0.85))
    if full:
        with m.coarse():
            m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.19, 0.22), COAL)
            m.box((-0.1, 0.1), (-0.05, 0.05), (0.22, 0.24), COAL)
    _open_top(m, HOPPER, (52, 52, 56) if full else None, 0.19)


def _log_wagon(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.box((-h, h), (-0.08, 0.08), (0.06, 0.085), DECK)
    for u in (-h + 0.005, h - 0.025):
        m.box((u, u + 0.02), (-0.08, 0.08), (0.085, 0.2), FRAME)
    if full:
        with m.coarse():
            for row, z in enumerate((0.115, 0.17)):
                for v in (-0.045, 0.0, 0.045)[: 3 - row]:
                    start = len(m.faces)
                    m.cylinder_along((-h + 0.03, h - 0.03), 0.026, z, LOG, LOG_END)
                    m.move(start, 0.0, v)
        _logs(m, (-h + 0.03, h - 0.03), 0.085, 0.07, 0.0175)
        with m.fine():
            for u in (-0.08, 0.08):
                m.cylinder_along((u - 0.004, u + 0.004), 0.004, 0.2, BLACK_IRON)
                for v in (0.074, -0.074):
                    parts.plate(m, (u - 0.003, u + 0.003), (0.09, 0.2), v, BLACK_IRON)


def _coach(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.body((-h, h), (-0.08, 0.08), (0.06, 0.23), COACH)
    _roof(m, (-h - 0.01, h + 0.01), 0.086, (0.23, 0.25), ROOF_GREY, 0.275)
    with m.coarse():
        m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.25, 0.265), ROOF_GREY)
    m.panel((-h + 0.01, h - 0.01), (0.105, 0.122), 0.0805, COACH_BAND)
    window = LIT_WINDOW if full else WINDOW
    with m.coarse():
        for u in (-0.17, -0.085, 0.0, 0.085, 0.17):
            m.panel((u - 0.03, u + 0.03), (0.135, 0.195), 0.0806, window)
    m.panel((-h + 0.01, h - 0.01), (0.07, 0.085), 0.0806, shade(COACH, 0.75))
    for u in (-0.12, -0.04, 0.04, 0.12):
        parts.window_frames(m, (u - 0.032, u + 0.032), (0.133, 0.197), 0.0806, COACH_BAND, window)
    _coach_fine(m, COACH, COACH_BAND, window)


def _coach_fine(m: Model, body: RGB, band: RGB, window: RGB) -> None:
    """A coach's doors at both ends with drop lights, handles and steps under them, lining,
    vents along the roof and gangways at the ends."""
    h = CAR_HALF
    with m.fine():
        for v in (0.0806, -0.0806):
            for u in (-h + 0.035, h - 0.035):
                parts.plate(m, (u - 0.022, u + 0.022), (0.066, 0.222), v, shade(body, 0.7))
                parts.plate(m, (u - 0.018, u + 0.018), (0.07, 0.218), v * 1.004, body)
                parts.plate(m, (u - 0.012, u + 0.012), (0.15, 0.198), v * 1.008, window)
                parts.plate(m, (u + 0.01, u + 0.016), (0.12, 0.13), v * 1.008, BRASS)
                m.box((u - 0.018, u + 0.018), tuple(sorted((v, v * 1.15))), (0.042, 0.048), FRAME)
            parts.plate(m, (-h + 0.06, h - 0.06), (0.204, 0.208), v, band)
        for u in (-0.12, 0.0, 0.12):
            m.cylinder_up(u, 0.012, (0.265, 0.278), shade(ROOF_GREY, 0.75))
        for end in (h, -h):
            out = 1.0 if end > 0 else -1.0
            m.box(
                tuple(sorted((end, end + out * 0.012))),
                (-0.045, 0.045),
                (0.08, 0.22),
                shade(FRAME, 1.6),
            )


def dining() -> Model:
    """An express's dining car: umber below and cream above, its saloon windows always lit,
    with a kitchen at the back end behind small windows and a stovepipe through the roof."""
    m = Model()
    h = CAR_HALF
    m.box((-h, h), (-0.07, 0.07), (0.035, 0.06), FRAME)
    m.wheels([h - 0.07, -h + 0.07], 0.035, FRAME, 0.075)
    m.body((-h, h), (-0.08, 0.08), (0.06, 0.23), DINER)
    m.panel((-h + 0.01, h - 0.01), (0.125, 0.215), 0.0805, DINER_CREAM)
    _roof(m, (-h - 0.01, h + 0.01), 0.086, (0.23, 0.25), ROOF_GREY)
    m.box((-h + 0.08, h - 0.02), (-0.045, 0.045), (0.25, 0.27), shade(ROOF_GREY, 0.8))
    m.box((-h + 0.025, -h + 0.055), (-0.015, 0.015), (0.25, 0.3), BLACK_IRON)
    for u in (-0.03, 0.06, 0.15):
        m.panel((u - 0.035, u + 0.035), (0.14, 0.2), 0.0806, LIT_WINDOW)
        parts.window_frames(m, (u - 0.035, u + 0.035), (0.14, 0.2), 0.0806, DINER, LIT_WINDOW, 2)
    for u in (-0.2, -0.15):
        m.panel((u - 0.015, u + 0.015), (0.17, 0.2), 0.0806, WINDOW)
    m.panel((-h + 0.01, h - 0.01), (0.07, 0.085), 0.0806, shade(DINER, 0.75))
    with m.fine():
        for v in (0.0806, -0.0806):
            parts.plate(m, (-h + 0.01, h - 0.01), (0.125, 0.129), v, BRASS)
            parts.plate(m, (-h + 0.01, h - 0.01), (0.211, 0.215), v, BRASS)
    _underframe(m)
    return m


def _mail_van(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    m.body((-h, h), (-0.08, 0.08), (0.06, 0.24), MAIL_VAN)
    _roof(m, (-h - 0.01, h + 0.01), 0.088, (0.24, 0.26), ROOF_GREY)
    m.panel((-h + 0.01, h - 0.01), (0.2, 0.215), 0.0805, BRASS)
    door = shade(MAIL_VAN, 0.7) if full else WINDOW
    m.panel((-0.05, 0.05), (0.07, 0.19), 0.0806, door)
    m.panel((-h + 0.03, -h + 0.08), (0.15, 0.19), 0.0806, WINDOW)
    m.panel((h - 0.08, h - 0.03), (0.15, 0.19), 0.0806, WINDOW)
    with m.fine():
        for v in (0.0806, -0.0806):
            for u in (-h + 0.055, h - 0.055):
                parts.plate(m, (u - 0.02, u + 0.02), (0.165, 0.169), v * 1.004, MAIL_VAN)
            parts.plate(m, (0.065, 0.135), (0.1, 0.15), v, (236, 228, 200))
            parts.bar(m, (0.065, 0.15), (0.1, 0.122), 0.005, v * 1.004, shade(MAIL_VAN, 0.8))
            parts.bar(m, (0.135, 0.15), (0.1, 0.122), 0.005, v * 1.004, shade(MAIL_VAN, 0.8))
            parts.plate(m, (-0.046, 0.046), (0.125, 0.13), v * 1.004, BRASS)


def _steel_wagon(m: Model, good: Good, full: bool) -> None:
    h = CAR_HALF
    _deck(m)
    _stakes(m)
    if full:
        with m.fine():
            for u in (-0.15, 0.0, 0.15):
                m.box((u - 0.012, u + 0.012), (-0.07, 0.07), (0.085, 0.095), WOOD)
        for v in (-0.04, 0.04):
            m.box((-h + 0.03, h - 0.03), (v - 0.03, v + 0.03), (0.095, 0.13), STEEL)
            parts.lines(
                m, (-h + 0.03, h - 0.03), (0.095, 0.13), abs(v) + 0.0306, shade(STEEL, 0.75), 1
            )
        m.box((-h + 0.05, h - 0.05), (-0.03, 0.03), (0.13, 0.165), STEEL)


def _plank_wagon(m: Model, good: Good, full: bool) -> None:
    """A stake wagon with two stacks of sawn boards, their ends showing pale, and the gaps
    between the layers."""
    h = CAR_HALF
    _deck(m)
    _stakes(m)
    if full:
        for v in (-0.04, 0.04):
            m.box((-h + 0.03, h - 0.03), (v - 0.032, v + 0.032), (0.085, 0.16), PLANK)
            parts.lines(
                m, (-h + 0.03, h - 0.03), (0.085, 0.16), abs(v) + 0.0326, shade(PLANK, 0.7), 3
            )
        with m.fine():
            for u in (-h + 0.03, h - 0.03):
                m.box((u - 0.004, u + 0.004), (-0.07, 0.07), (0.085, 0.16), PLANK_END)


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
    Good.RICE: _grain_hopper,
    Good.SOYBEANS: _grain_hopper,
    Good.APPLES: _fruit_wagon,
    Good.DATES: _fruit_wagon,
    Good.ORANGES: _fruit_wagon,
    Good.COAL: _coal_hopper,
    Good.LOGS: _log_wagon,
    Good.PLANKS: _plank_wagon,
    Good.PASSENGERS: _coach,
    Good.MAIL: _mail_van,
    Good.STEEL: _steel_wagon,
}
# The load of each heap wagon (HEAPS) when it is drawn fine, lumps where none is given.
HEAP_LOADS: dict[Good, Callable[[Model, RGB], None]] = {
    Good.WOOL: _bales,
    Good.COTTON: _bales,
    Good.FLAX: _bales,
    Good.HOPS: _bales,
    Good.BARRELS: _barrels,
    Good.BRICKS: _bricks,
    Good.MACHINERY: _crated,
    Good.GRAPES: _tubs,
}


class Look(Enum):
    ENGINE = "engine"
    TENDER = "tender"
    DINING = "dining"


@dataclass(frozen=True)
class EngineModel:
    """One engine of the catalogue: how to build it and its tender if it has one, their half
    lengths in model cells, and where its chimney is (model cells ahead of its middle and above
    the rails) if it makes smoke. On the map each model cell is SCALE cells."""

    engine: Callable[[], Model]
    half: float
    tender: Callable[[], Model] | None = None
    tender_half: float = 0.0
    chimney: tuple[float, float] | None = None


def _entry(engine: Pioneer | Steam | Unit, tender: Tender | Turned | None = None) -> EngineModel:
    if tender is None:
        return EngineModel(engine.model, engine.half, chimney=engine.chimney)
    return EngineModel(engine.model, engine.half, tender.model, tender.half, engine.chimney)


def _twin(unit: Unit) -> EngineModel:
    """A pair of units coupled back to back, the second drawn as the tender."""
    return _entry(unit, Turned(unit))


# Each engine of railroads.sim.trains.ENGINES, its tender if it has one, and where its chimney is.
ENGINES: dict[str, EngineModel] = {
    "penydarren": _entry(
        Pioneer(0.24, (70, 70, 76), WOOD, ((0.12, 0.045), (-0.1, 0.045)), flywheel=True)
    ),
    "catch_me": _entry(
        Pioneer(
            0.25,
            (44, 62, 98),
            FRAME,
            ((0.12, 0.05), (-0.1, 0.05)),
            WHEEL_RED,
            uprights=(-0.08,),
            crossheads=True,
        )
    ),
    "salamanca": _entry(
        Pioneer(
            0.27,
            OAK,
            OAK,
            ((0.14, 0.045), (-0.14, 0.045)),
            STEEL,
            uprights=(-0.04, 0.08),
            crossheads=True,
            rack=True,
            hoops=BLACK_IRON,
            hoop_count=6,
            upright=OAK,
            column=OAK,
            back_end=BLACK_IRON,
        ),
    ),
    "puffing_billy": _entry(
        Pioneer(
            0.3,
            LAGGING,
            WOOD,
            ((0.16, 0.05), (-0.12, 0.05)),
            BLACK_IRON,
            chimney_top=0.5,
            beams=True,
            hoops=BLACK_IRON,
            hoop_count=5,
            back_end=BLACK_IRON,
            spokes=8,
        ),
        Tender(0.17, WOOD, (0.09, -0.09), top=0.11, trim=None, tank=BLACK_IRON, wheel=BLACK_IRON),
    ),
    "locomotion": _entry(
        Pioneer(
            0.3,
            BLACK_IRON,
            FRAME,
            ((0.16, 0.05), (-0.04, 0.05), (-0.2, 0.05)),
            BRASS,
            rods=True,
            uprights=(-0.1, 0.1),
            crossheads=True,
        ),
        Tender(0.17, WOOD, (0.09, -0.09), top=0.11, trim=None, barrel=(150, 104, 62)),
    ),
    "rocket": _entry(
        Pioneer(
            0.3,
            ROCKET_YELLOW,
            FRAME,
            ((0.14, 0.075), (-0.18, 0.042)),
            ROCKET_YELLOW,
            chimney_color=(250, 250, 246),
            chimney_top=0.56,
            flue=0.026,
            radius=0.06,
            slanted=True,
            firebox=BLACK_IRON,
            front=BLACK_IRON,
            hoops=BLACK_IRON,
            crown=True,
            tubes=COPPER,
            tyre=(46, 44, 44),
        ),
        Tender(
            0.16,
            ROCKET_YELLOW,
            (0.08, -0.08),
            top=0.11,
            trim=None,
            barrel=ROCKET_YELLOW,
            wheel=ROCKET_YELLOW,
            slats=True,
            tyre=(46, 44, 44),
        ),
    ),
    "lion": _entry(
        Pioneer(
            0.32,
            (52, 112, 64),
            CAB_RED,
            ((0.17, 0.055), (0.02, 0.055), (-0.22, 0.035)),
            rods=True,
            firebox=COPPER,
            haystack=True,
        ),
        Tender(0.2, (52, 112, 64), (0.11, -0.11), top=0.17),
    ),
    "planet": _entry(
        Steam(
            0.34,
            LAGGING,
            BLACK_IRON,
            ((-0.16,),),
            0.075,
            BLACK_IRON,
            leading=(0.17,),
            radius=0.07,
            cab=0.13,
            trim=BRASS,
            smokebox=BLACK_IRON,
            domes=(-0.17,),
            dome=(0.036, 0.085),
            stack=0.23,
            open_cab=True,
            beam=BLACK_IRON,
            lead=0.048,
            bands=5,
            firebox=BLACK_IRON,
            sand=False,
        ),
        Tender(0.2, TENDER_GREEN, (0.11, -0.11), top=0.16, trim=None, width=0.08),
    ),
    "tank": _entry(
        Steam(
            0.3,
            OCHRE,
            (104, 78, 40),
            ((0.11, -0.03),),
            0.045,
            OCHRE,
            radius=0.068,
            cab=0.2,
            domes=(0.02,),
            tanks=True,
        )
    ),
    "goods": _entry(
        Steam(
            0.38,
            (56, 56, 62),
            (36, 36, 40),
            ((0.17, 0.05, -0.07),),
            0.046,
            (80, 80, 88),
            radius=0.078,
            trim=CAB_RED,
        ),
        Tender(0.24, (56, 56, 62), (0.15, 0.0, -0.15), top=0.21, trim=CAB_RED),
    ),
    "single": _entry(
        Steam(
            0.4,
            (92, 142, 72),
            (52, 84, 42),
            ((0.0,),),
            0.088,
            (92, 142, 72),
            leading=(0.29, 0.2),
            trailing=(-0.22,),
            radius=0.072,
            cab=0.2,
            domes=(),
            stack=0.1,
        ),
        Tender(0.24, (92, 142, 72), (0.15, 0.0, -0.15), top=0.2),
    ),
    "american": _entry(
        Steam(
            0.4,
            BLACK_IRON,
            ROOF_RED,
            ((0.04, -0.11),),
            0.06,
            WHEEL_RED,
            leading=(0.27, 0.19),
            radius=0.068,
            cab=0.2,
            cab_color=CAB_RED,
            smokebox=BLACK_IRON,
            domes=(0.08, -0.06),
            stack=0.14,
            balloon=True,
            cowcatcher=CAB_RED,
            headlight=CAB_RED,
            beam=CAB_RED,
            bands=4,
            cylinder=BRASS,
        ),
        Tender(0.24, CAB_RED, (0.17, 0.09, -0.09, -0.17), top=0.19, load=LOG),
    ),
    "mogul": _entry(
        Steam(
            0.42,
            MOGUL_GREEN,
            MOGUL_ROOF,
            ((0.17, 0.05, -0.07),),
            0.045,
            MOGUL_GREEN,
            leading=(0.33,),
            radius=0.08,
            cab=0.3,
            domes=(0.06, -0.06),
        ),
        Tender(0.26, MOGUL_GREEN, (0.17, 0.0, -0.17), top=0.22),
    ),
    "light_tank": _entry(
        Steam(
            0.4,
            (40, 56, 92),
            (26, 36, 60),
            ((0.17, 0.06, -0.05),),
            0.048,
            (40, 56, 92),
            trailing=(-0.29,),
            radius=0.078,
            cab=0.24,
            tanks=True,
        )
    ),
    "atlantic": _entry(
        Steam(
            0.44,
            BRUNSWICK,
            BRUNSWICK,
            ((0.08, -0.08),),
            0.066,
            BRUNSWICK,
            leading=(0.35, 0.26),
            trailing=(-0.28,),
            radius=0.086,
            trim=(150, 128, 70),
            smokebox=BLACK_IRON,
            domes=(0.02,),
            dome_color=BRUNSWICK,
            beam=BLACK_IRON,
            headlight=BLACK_IRON,
        ),
        Tender(0.28, BRUNSWICK, (0.19, 0.0, -0.19), top=0.23, trim=None),
    ),
    "mallet": _entry(
        Steam(
            0.52,
            BLACK_IRON,
            BLACK_IRON,
            ((0.27, 0.18, 0.09), (-0.07, -0.16, -0.25)),
            0.042,
            BLACK_IRON,
            leading=(0.43,),
            trailing=(-0.37,),
            radius=0.086,
            trim=(70, 70, 76),
            smokebox=(84, 84, 92),
            domes=(0.0, -0.14),
            dome_color=BLACK_IRON,
            stack=0.07,
            beam=BLACK_IRON,
            headlight=BLACK_IRON,
        ),
        Tender(0.28, BLACK_IRON, (0.2, 0.12, -0.12, -0.2), top=0.24, trim=None),
    ),
    "pacific": _entry(
        Steam(
            0.46,
            APPLE_GREEN,
            BLACK_IRON,
            ((0.15, 0.01, -0.13),),
            0.058,
            APPLE_GREEN,
            leading=(0.38, 0.3),
            trailing=(-0.31,),
            radius=0.088,
            cab=0.32,
            trim=LINING,
            smokebox=BLACK_IRON,
            domes=(0.05,),
            dome_color=APPLE_GREEN,
            deflectors=True,
        ),
        Tender(0.3, APPLE_GREEN, (0.2, 0.0, -0.2), top=0.24, trim=LINING, width=0.09),
    ),
    "boxcab": EngineModel(boxcab, BOXCAB_HALF),
    "railbus": _entry(
        Unit(
            0.3,
            (176, 48, 40),
            (176, 48, 40),
            (0.17, -0.17),
            axles=1,
            top=0.25,
            nose=0.04,
            tip=0.2,
            band=DINER_CREAM,
            windows=4,
        )
    ),
    "streamliner": _entry(
        Steam(
            0.5,
            (40, 60, 96),
            (26, 38, 62),
            ((0.12, -0.01, -0.14),),
            0.062,
            (176, 40, 40),
            leading=(0.38, 0.29),
            trailing=(-0.3,),
            radius=0.088,
            cab=0.2,
            shroud=(200, 206, 216),
        ),
        Tender(
            0.32,
            (40, 60, 96),
            (0.22, 0.13, -0.13, -0.22),
            top=0.25,
            trim=(200, 206, 216),
            trim_z=(0.18, 0.192),
            shroud=True,
        ),
    ),
    "big_boy": _entry(
        Steam(
            0.6,
            BLACK_IRON,
            BLACK_IRON,
            ((0.3, 0.21, 0.12, 0.03), (-0.145, -0.235, -0.325, -0.415)),
            0.045,
            BLACK_IRON,
            leading=(0.53, 0.45),
            trailing=(-0.5,),
            radius=0.098,
            smokebox=GRAPHITE,
            smokebox_length=0.26,
            trim=(64, 64, 70),
            domes=(0.08, -0.08),
            dome=(0.04, 0.045),
            dome_color=BLACK_IRON,
            stack=0.06,
            cowcatcher=BLACK_IRON,
            headlight=BLACK_IRON,
            cylinder=BLACK_IRON,
            beam=BLACK_IRON,
        ),
        Tender(
            0.36,
            BLACK_IRON,
            (0.28, 0.19, 0.1, -0.1, -0.19, -0.28),
            top=0.26,
            trim=None,
            width=0.092,
        ),
    ),
    "mixed_diesel": _entry(
        Unit(
            0.42,
            (186, 190, 198),
            (176, 40, 40),
            (0.25, -0.25),
            top=0.28,
            nose=0.1,
            tip=0.17,
            both=False,
            ends=(176, 40, 40),
            vents=3,
        )
    ),
    "heavy_diesel": _entry(
        Unit(0.48, (214, 170, 52), (150, 36, 36), (0.29, -0.29), axles=3, hood=True, vents=4)
    ),
    "bullet": _entry(
        Unit(
            0.5,
            (238, 238, 232),
            (36, 64, 156),
            (0.3, -0.3),
            nose=0.18,
            tip=0.11,
            both=False,
            band=(36, 64, 156),
            windows=5,
            pantographs=(-0.12,),
        )
    ),
    "express": EngineModel(express, EXPRESS_HALF),
    "sprinter": _entry(
        Unit(
            0.42,
            (224, 226, 230),
            (40, 84, 170),
            (0.27, -0.27),
            nose=0.05,
            tip=0.21,
            ends=(236, 200, 40),
            windows=6,
        )
    ),
    "high_speed": _entry(
        Unit(
            0.54,
            (236, 236, 238),
            (200, 30, 40),
            (0.32, -0.3),
            nose=0.22,
            tip=0.1,
            both=False,
            pantographs=(-0.2,),
        )
    ),
    "twin_diesel": _twin(
        Unit(0.5, (40, 62, 120), (236, 196, 40), (0.3, -0.3), axles=3, hood=True, vents=4)
    ),
    "modern_electric": _entry(
        Unit(
            0.46,
            (196, 32, 40),
            (236, 236, 236),
            (0.25, -0.25),
            top=0.28,
            nose=0.05,
            tip=0.21,
            pantographs=(-0.2, 0.2),
        )
    ),
    "super_diesel": _entry(
        Unit(
            0.66,
            (52, 56, 64),
            ORANGE,
            (0.42, -0.42),
            axles=3,
            top=0.3,
            nose=0.07,
            tip=0.22,
            ends=(236, 200, 40),
            vents=5,
        )
    ),
    "maglev": _entry(
        Unit(
            0.58,
            (226, 232, 238),
            (0, 150, 160),
            top=0.22,
            nose=0.3,
            tip=0.05,
            both=False,
            band=(36, 44, 58),
            windows=6,
            glass=LIT_WINDOW,
            floats=True,
        )
    ),
}


MODELS: dict[object, Callable[[], Model]] = {
    **{(Look.ENGINE, name): model.engine for name, model in ENGINES.items()},
    **{(Look.TENDER, name): model.tender for name, model in ENGINES.items() if model.tender},
    Look.DINING: dining,
    **{(good, full): (lambda g=good, f=full: car(g, f)) for good in Good for full in (True, False)},
}


@dataclass(frozen=True)
class _Sight:
    """How models.draw_model sees models on one map projection, worked out the way it does: the
    plane-to-picture matrix, how many picture pixels tall a cell of height is, the ways to the
    viewer and to the light, and the empty margins it leaves round a model."""

    scale: FloatArray
    up: float
    view: FloatArray
    light: FloatArray
    shine: FloatArray
    pad: int
    edge: int

    @classmethod
    def of(cls, matrix: FloatArray, pixels: float) -> Self:
        scale = matrix * pixels
        towards = np.linalg.solve(matrix, np.array([0.0, 1.0]))
        towards = towards / np.hypot(*towards)
        light_plane = np.linalg.solve(matrix, np.array([-1.0, -1.0]))
        light_plane = light_plane / np.hypot(*light_plane)
        pad = max(2, round(2 * pixels))
        view = np.array([towards[0] * TILT, towards[1] * TILT, HEIGHT_PER_WIDTH])
        light = np.array([*light_plane * 0.6, 0.8])
        return cls(
            scale=scale,
            up=float(np.hypot(*scale[:, 0])) * HEIGHT_PER_WIDTH,
            view=view,
            light=light,
            shine=halfway(light, view),
            pad=pad,
            edge=pad + max(1, round(pixels)),
        )

    def on_picture(self, points: Points) -> tuple[FloatArray, FloatArray]:
        """Where turned points land in the picture, in pixels from the model's origin."""
        x, y, z = points
        (a, b), (c, d) = self.scale
        return a * x + b * y, c * x + d * y - z * self.up

    def frame(self, xs: FloatArray, ys: FloatArray) -> tuple[int, int, int, int]:
        """Left, top, width and height of the picture round points, margins and all."""
        left, top = math.floor(xs.min()) - self.pad, math.floor(ys.min()) - self.pad
        width = math.ceil(xs.max()) - left + self.edge
        height = math.ceil(ys.max()) - top + self.edge
        return left, top, width, height


@dataclass(frozen=True)
class _Faces:
    """A model's faces as arrays: their normals, the corners of all of them one face after
    another, where each face's corners start and end, each face's colour in every shade, and the
    faces grouped by how many corners they have."""

    normals: FloatArray
    corners: FloatArray
    counts: IntArray
    starts: list[int]
    shades: list[list[RGB]]
    groups: list[tuple[IntArray, IntArray]]
    decals: list[tuple[int, int]]
    # Drawn fine: each face's colour and how high its middle is, for models.fine_shades.
    colors: FloatArray | None = None
    lows: FloatArray | None = None

    @classmethod
    def of(cls, model: Model, fine: bool = False, finest: bool = False) -> Self:
        faces = [face for face in model.faces if face.drawn(fine, finest)]
        counts = np.array([len(face.corners) for face in faces])
        starts = np.concatenate([[0], np.cumsum(counts)])
        groups = []
        for k in np.unique(counts).tolist():
            alike = np.flatnonzero(counts == k)
            groups.append((alike, starts[alike, None] + np.arange(k)))
        shaded = {
            color: [shade(color, level) for level in SHADES]
            for color in {face.color for face in faces}
        }
        corners = np.array([c for face in faces for c in face.corners], dtype=np.float64)
        made = cls(
            normals=np.array([face.normal for face in faces], dtype=np.float64),
            corners=corners,
            counts=counts,
            starts=starts.tolist(),
            shades=[shaded[face.color] for face in faces],
            groups=groups,
            decals=decals(faces) if fine else [],
        )
        if not fine:
            return made
        return replace(
            made,
            colors=np.array([face.color for face in faces], dtype=np.float64),
            lows=made.means(corners[:, 2]),
        )

    def means(self, values: FloatArray) -> FloatArray:
        """Each face's mean of values given at every corner, taken as np.mean takes one face's."""
        means = np.empty(len(self.counts))
        for faces, corners in self.groups:
            means[faces] = values[corners].mean(axis=1)
        return means


def _pictures(
    model: Model, sight: _Sight, fine: bool = False, indices: Iterable[int] = range(HEADINGS)
) -> list[Sprite]:
    """The model running each of the HEADINGS headings, or those of them given, drawn as
    models.draw_model draws it but without the outline."""
    faces = _Faces.of(model, fine)
    return [_picture(faces, sight, 2 * math.pi * index / HEADINGS) for index in indices]


def _picture(faces: _Faces, sight: _Sight, heading: float) -> Sprite:
    """One heading of _pictures: every face turned, culled, shaded and placed at once, with the
    same sums in the same order as draw_model works them out one face at a time."""
    cos, sin = math.cos(heading), math.sin(heading)
    normals, corners = _turned(faces.normals, cos, sin), _turned(faces.corners, cos, sin)
    shown = _along(normals, sight.view) > 1e-6
    lit = np.clip(_along(normals, sight.light), 0, 1)
    levels = np.minimum(len(SHADES) - 1, (lit * len(SHADES)).astype(np.int64)).tolist()
    depth = faces.means(_along(corners, sight.view))
    for index, under in faces.decals:
        depth[index] = depth[under] + 1e-6
    order = np.flatnonzero(shown)[np.argsort(depth[shown], kind="stable")].tolist()
    xs, ys = sight.on_picture(corners)
    on = np.repeat(shown, faces.counts)
    left, top, width, height = sight.frame(xs[on], ys[on])
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    paint = ImageDraw.Draw(image)
    px, py = (xs - left).tolist(), (ys - top).tolist()
    if faces.colors is not None and faces.lows is not None:
        shine = _along(normals, sight.shine)
        rgbs = fine_shades(faces.colors[order], lit[order], shine[order], faces.lows[order])
        colors = [(r, g, b, 255) for r, g, b in rgbs]
    else:
        colors = [(*faces.shades[f][levels[f]], 255) for f in order]
    for f, color in zip(order, colors, strict=True):
        span = slice(faces.starts[f], faces.starts[f + 1])
        paint.polygon(list(zip(px[span], py[span], strict=True)), fill=color)
    return Sprite(image, (-left, -top))


def _turned(points: FloatArray, cos: float, sin: float) -> Points:
    """Points given along a model, across it and up, turned round to the map's x and y and up."""
    u, v, z = points.T
    return u * cos - v * sin, u * sin + v * cos, z


def _along(points: Points, way: FloatArray) -> FloatArray:
    """How far each point lies along `way`, summed in the order numpy sums `point @ way`."""
    x, y, z = points
    return x * way[0] + y * way[1] + z * way[2]


@dataclass(frozen=True)
class TrainSprites:
    """Every look of every engine and car, packed into one picture: where each sits in it."""

    image: npt.NDArray[np.uint8]
    places: dict[tuple[object, int], tuple[int, int, int, int, int, int]]

    def place(self, look: object, heading: float) -> tuple[int, int, int, int, int, int]:
        """x, y, width, height and anchor of the look running closest to `heading` radians."""
        return self.places[(look, turned(heading))]


class FinePainter:
    """Draws a look's pictures `times` as many pixels each way as on the sheet, with its fine
    details and an outline about half as wide as the sheet's blown up, one heading at a time: for
    the close zooms, whose pictures would otherwise be the sheet's blown up. Its model and faces
    are made once, when it is made."""

    def __init__(
        self, matrix_key: tuple[float, float, float, float], look: object, times: int
    ) -> None:
        matrix = np.array(matrix_key, dtype=np.float64).reshape(2, 2)
        self._sight = _Sight.of(matrix, DETAIL * SCALE * times)
        self._faces = _Faces.of(MODELS[look](), fine=True, finest=times >= FINEST_FROM)
        self._outline = max(1, times // 2)

    def picture(self, index: int) -> Sprite:
        """The look running heading `index` of HEADINGS."""
        picture = _picture(self._faces, self._sight, 2 * math.pi * index / HEADINGS)
        _outline(picture.image, self._outline)
        return picture


def fine_pictures(
    matrix_key: tuple[float, float, float, float],
    look: object,
    times: int,
    indices: Iterable[int] = range(HEADINGS),
) -> list[Sprite]:
    """A look every way round, or running the headings given, drawn by a FinePainter."""
    painter = FinePainter(matrix_key, look, times)
    return [painter.picture(index) for index in indices]


def turned(heading: float) -> int:
    """Which of the HEADINGS pictures of a look runs closest to `heading` radians."""
    return round(heading / (2 * math.pi) * HEADINGS) % HEADINGS


@cache
def train_sprites(matrix_key: tuple[float, float, float, float]) -> TrainSprites:
    """All the train looks for a map projection, given as its matrix's four numbers, packed
    on a roughly square sheet so it stays within MAX_TEXTURE when blown up for the closest zoom.
    The looks go on without their outline, which is drawn round the whole sheet at once: each
    keeps an empty margin round its model, so every outline pixel lands where draw_model puts it."""
    matrix = np.array(matrix_key, dtype=np.float64).reshape(2, 2)
    sight = _Sight.of(matrix, DETAIL * SCALE)
    with without_fine():
        models = {look: make() for look, make in MODELS.items()}
    sprites = {
        (look, index): sprite
        for look, model in models.items()
        for index, sprite in enumerate(_pictures(model, sight))
    }
    area = sum(sprite.image.size[0] * (sprite.image.size[1] + 1) for sprite in sprites.values())
    width = math.ceil(math.sqrt(area * 1.15) / 64) * 64
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
    _outline(sheet)
    return TrainSprites(np.asarray(sheet).copy(), places)


@cache
def _far_sight(matrix_key: tuple[float, float, float, float]) -> _Sight:
    return _Sight.of(np.array(matrix_key, dtype=np.float64).reshape(2, 2), DETAIL * SCALE)


def far_pictures(matrix_key: tuple[float, float, float, float], look: object) -> list[Sprite]:
    """A look every way round as train_sprites puts it on its sheet, each with its own outline,
    which is the sheet's: the margin round each model keeps outlines apart."""
    with without_fine():
        model = MODELS[look]()
    pictures = _pictures(model, _far_sight(matrix_key))
    for picture in pictures:
        _outline(picture.image)
    return pictures


# How wide the sheet of looks drawn as they are wanted is, in pixels: as wide as train_sprites'
# sheet of every look or so, which keeps it within MAX_TEXTURE tall.
GROWN_WIDTH = 2432


class GrownTrainSprites:
    """Looks of engines and cars packed into one picture as train_sprites packs them, each drawn
    only once it is first wanted or, a few at a time, ahead of being wanted: drawing every look
    at once would hold up loading a game."""

    def __init__(
        self, matrix_key: tuple[float, float, float, float], looks: Iterable[object] = ()
    ) -> None:
        self._matrix_key = matrix_key
        self.image = np.zeros((0, GROWN_WIDTH, 4), dtype=np.uint8)
        self.places: dict[tuple[object, int], tuple[int, int, int, int, int, int]] = {}
        # Where the next picture goes, and how tall the row it goes in is so far.
        self._x = self._y = self._row = 0
        self._ahead = iter(MODELS)
        for look in looks:
            self.draw(look)

    @property
    def height(self) -> int:
        """How many rows of the picture hold looks."""
        return self._y + self._row

    def has(self, look: object) -> bool:
        return (look, 0) in self.places

    def place(self, look: object, heading: float) -> tuple[int, int, int, int, int, int]:
        """x, y, width, height and anchor of the look running closest to `heading` radians,
        drawing the look first if it is not drawn yet."""
        self.draw(look)
        return self.places[(look, turned(heading))]

    def draw(self, look: object) -> None:
        """Draws a look on the picture unless it is there already."""
        if self.has(look):
            return
        for index, picture in enumerate(far_pictures(self._matrix_key, look)):
            w, h = picture.image.size
            if self._x + w > GROWN_WIDTH:
                self._x, self._y, self._row = 0, self._y + self._row + 1, 0
            if self._y + h > len(self.image):
                grown = np.zeros(
                    (max(len(self.image) + 512, self._y + h), GROWN_WIDTH, 4), np.uint8
                )
                grown[: len(self.image)] = self.image
                self.image = grown
            self.image[self._y : self._y + h, self._x : self._x + w] = np.asarray(picture.image)
            self.places[(look, index)] = (self._x, self._y, w, h, *picture.anchor)
            self._x += w + 1
            self._row = max(self._row, h)

    def draw_ahead(self, until: float) -> None:
        """Draws looks not wanted yet, one after another while time.perf_counter() is before
        `until`."""
        while time.perf_counter() < until:
            look = next(self._ahead, None)
            if look is None:
                return
            self.draw(look)
