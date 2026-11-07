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

from railroads.sim.economy import Good
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

# Sizes in cells: how far apart the engine and cars sit, and their half lengths.
ENGINE_HALF = 0.36
TENDER_HALF = 0.22
CAR_HALF = 0.24
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


def car(good: Good, full: bool) -> Model:
    m = Model()
    h = CAR_HALF
    m.box((-h, h), (-0.07, 0.07), (0.035, 0.06), FRAME)
    m.wheels([h - 0.07, -h + 0.07], 0.035, FRAME, 0.075)
    match good:
        case Good.FOOD | Good.GOODS:
            body, roof = (REEFER, (232, 228, 220)) if good == Good.FOOD else (BOXCAR, ROOF_GREY)
            m.box((-h, h), (-0.08, 0.08), (0.06, 0.24), body)
            m.box((-h - 0.01, h + 0.01), (-0.088, 0.088), (0.24, 0.26), roof)
            door = shade(body, 0.7) if full else WINDOW
            m.panel((-0.06, 0.06), (0.07, 0.22), 0.0805, door)
            m.panel((-h + 0.01, h - 0.01), (0.07, 0.085), 0.0806, shade(body, 0.75))
        case Good.COAL:
            m.box((-h, h), (-0.08, 0.08), (0.06, 0.19), HOPPER)
            m.panel((-h + 0.02, h - 0.02), (0.08, 0.17), 0.0805, shade(HOPPER, 0.85))
            if full:
                m.box((-h + 0.02, h - 0.02), (-0.07, 0.07), (0.19, 0.22), COAL)
                m.box((-0.1, 0.1), (-0.05, 0.05), (0.22, 0.24), COAL)
        case Good.LUMBER:
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
        case Good.STEEL:
            m.box((-h, h), (-0.08, 0.08), (0.06, 0.085), DECK)
            if full:
                for v in (-0.04, 0.04):
                    m.box((-h + 0.03, h - 0.03), (v - 0.03, v + 0.03), (0.085, 0.12), STEEL)
                m.box((-h + 0.05, h - 0.05), (-0.03, 0.03), (0.12, 0.155), STEEL)
    return m


def shade(color: RGB, factor: float) -> RGB:
    r, g, b = color
    return (min(255, round(r * factor)), min(255, round(g * factor)), min(255, round(b * factor)))


class Look(Enum):
    ENGINE = "engine"
    TENDER = "tender"


MODELS: dict[object, Callable[[], Model]] = {
    Look.ENGINE: engine,
    Look.TENDER: tender,
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
