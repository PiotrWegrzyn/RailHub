"""The game's sprites, drawn pixel by pixel in code and packed into one sheet.

`tools/build_sprites.py` saves the sheet as a PNG with a JSON index next to it, so any sprite can
be repainted by hand afterwards; the game only ever reads the PNG. Sprites are drawn for a cell 32
pixels wide, seen from the fixed view angle, with light from the top left. Each one stands on an
anchor: the point that sits on the ground where the thing is.

Hills and mountains are drawn in TINT_GREYS and coloured with the shades of the ground they stand
on when the map is drawn, so one hill fits grass, forest, desert or snow alike. Only pure greys
are recoloured.
"""

import itertools
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

type RGB = tuple[int, int, int]
type Point = tuple[float, float]

# Greys of tinted sprites, standing for: outline, the ground's four shades dark to light, and a
# highlight above the lightest shade.
TINT_GREYS = (0, 51, 102, 153, 204, 255)
HILL = "hill"
MOUNTAIN = "mountain"
# Width and height in pixels of each hill and mountain sprite.
HILL_SIZES = [(20, 5), (17, 4), (14, 4), (12, 3)]
MOUNTAIN_SIZES = [(32, 13), (28, 12), (26, 10), (22, 9)]
SNOW_LIT: RGB = (238, 242, 250)
SNOW_SHADE: RGB = (184, 196, 218)

SHADOW = (0, 0, 0, 72)
TRUNK: RGB = (88, 60, 38)
PINES: list[tuple[RGB, RGB, RGB]] = [
    ((22, 54, 34), (34, 78, 46), (56, 104, 60)),
    ((26, 60, 30), (40, 86, 40), (66, 114, 56)),
    ((20, 50, 42), (30, 72, 58), (50, 98, 78)),
]
LEAVES: list[tuple[RGB, RGB, RGB]] = [
    ((34, 78, 32), (54, 108, 44), (86, 140, 60)),
    ((44, 88, 28), (70, 120, 40), (106, 154, 58)),
    ((66, 92, 28), (96, 124, 40), (136, 158, 62)),
]
WALLS: list[RGB] = [(226, 214, 190), (210, 176, 146), (232, 228, 218)]
ROOFS: list[RGB] = [(176, 70, 52), (106, 102, 112), (150, 92, 56), (72, 94, 132)]
TENEMENT_WALLS: list[RGB] = [(232, 196, 120), (220, 160, 150), (170, 204, 176), (214, 210, 196)]
TENEMENT_ROOFS: list[RGB] = [(164, 74, 50), (130, 70, 56)]
FLAT_WALLS: list[RGB] = [(196, 190, 178), (214, 200, 160), (176, 184, 190)]
WINDOW: RGB = (58, 70, 92)
BRICK: RGB = (168, 86, 60)
METAL: RGB = (150, 152, 160)
TIMBER: RGB = (150, 112, 74)
STONE: RGB = (154, 150, 140)
ROCK: RGB = (124, 120, 114)
GLASS: RGB = (96, 136, 168)
FRUIT: list[RGB] = [(212, 52, 44), (236, 196, 60), (226, 112, 40)]
MILITARY_YARD: RGB = (150, 142, 100)
MILITARY_ROOF: RGB = (96, 104, 66)
MILITARY_FLAG: RGB = (190, 50, 46)
CHURCH_WALL: RGB = (216, 208, 190)
CHURCH_ROOF: RGB = (112, 84, 78)
CROSS: RGB = (244, 214, 110)
LEAD: RGB = (104, 124, 140)
BARREL: RGB = (150, 100, 56)
HIVE: RGB = (226, 180, 78)
CLOTH: list[RGB] = [(232, 220, 196), (110, 140, 200), (214, 120, 160)]
CAR_COLOURS: list[RGB] = [
    (222, 64, 92),
    (60, 110, 200),
    (240, 200, 70),
    (236, 236, 230),
    (60, 160, 100),
]
COPPER: RGB = (204, 124, 66)
THATCH: RGB = (186, 154, 88)
WHITEWASH: RGB = (238, 236, 228)
STATION_WALL: RGB = (238, 226, 196)
STATION_ROOF: RGB = (132, 44, 38)
PLATFORM: RGB = (184, 180, 170)
PLATFORM_EDGE: RGB = (248, 248, 244)
NAME_BOARD: RGB = (40, 74, 156)
STAINED: RGB = (62, 78, 132)
DOORWAY: RGB = (54, 46, 48)
SLATE: RGB = (78, 84, 98)
VERDIGRIS: RGB = (98, 150, 132)


def shade(color: RGB, factor: float) -> RGB:
    r, g, b = color
    return (min(255, round(r * factor)), min(255, round(g * factor)), min(255, round(b * factor)))


@dataclass(frozen=True)
class Sprite:
    image: Image.Image
    anchor: tuple[int, int]


# How many pixels the Painter draws for every unit sprites are laid out in (drawn_at).
_fineness = 1.0


@contextmanager
def drawn_at(fineness: float) -> Iterator[None]:
    """Sprites drawn inside take `fineness` pixels for every unit of their layout, so they come
    out that much bigger with outlines and lines still one pixel wide: finer, not blown up."""
    global _fineness
    before, _fineness = _fineness, fineness
    try:
        yield
    finally:
        _fineness = before


class Painter:
    """Draws on one sprite's canvas, with the anchor at (0, 0), at the fineness of drawn_at."""

    def __init__(self, width: int, height: int, anchor: tuple[int, int]) -> None:
        k = self._k = _fineness
        self.image = Image.new("RGBA", (round(width * k), round(height * k)), (0, 0, 0, 0))
        self.anchor = (round(anchor[0] * k), round(anchor[1] * k))
        self._draw = ImageDraw.Draw(self.image)

    def _at(self, points: list[Point]) -> list[Point]:
        ax, ay = self.anchor
        k = self._k
        return [(ax + x * k, ay + y * k) for x, y in points]

    def polygon(self, points: list[Point], fill: RGB, outline: RGB | None = None) -> None:
        self._draw.polygon(self._at(points), fill=fill, outline=outline)

    def line(self, points: list[Point], fill: RGB) -> None:
        self._draw.line(self._at(points), fill=fill, width=max(1, round(self._k / 2)))

    def rect(self, left: float, top: float, right: float, bottom: float, fill: RGB) -> None:
        (x0, y0), (x1, y1) = self._at([(left, top), (right, bottom)])
        self._draw.rectangle((x0, y0, x1, y1), fill=fill, outline=shade(fill, 0.6))

    def ellipse(self, left: float, top: float, right: float, bottom: float, fill: RGB) -> None:
        (x0, y0), (x1, y1) = self._at([(left, top), (right, bottom)])
        self._draw.ellipse((x0, y0, x1, y1), fill=fill, outline=shade(fill, 0.65))

    def blob(self, left: float, top: float, right: float, bottom: float, fill: RGB) -> None:
        (x0, y0), (x1, y1) = self._at([(left, top), (right, bottom)])
        self._draw.ellipse((x0, y0, x1, y1), fill=fill)

    def point(self, x: float, y: float, fill: RGB) -> None:
        if self._k == 1:
            self._draw.point(self._at([(x, y)]), fill=fill)
            return
        (x0, y0), (x1, y1) = self._at([(x, y), (x + 1, y + 1)])
        self._draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), fill=fill)

    def shadow(self, half_width: float, half_height: float, reach: float = 3) -> None:
        overlay = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
        (x0, y0), (x1, y1) = self._at(
            [(-half_width + 1, -half_height), (half_width + reach, half_height + 1)]
        )
        ImageDraw.Draw(overlay).ellipse((x0, y0, x1, y1), fill=SHADOW)
        self.image.alpha_composite(overlay)

    def box(
        self, x: float, y: float, half_width: float, height: float, wall: RGB, roof: RGB | None
    ) -> tuple[Point, Point, Point, Point]:
        """A block standing on a 2:1 diamond centred at (x, y): walls lit on the left, shaded on
        the right, and a flat top in the roof colour if one is given. Returns the top corners:
        left, front, right and back."""
        s = half_width
        return self._prism(
            (x - s, y), (x, y + s / 2), (x + s, y), (x, y - s / 2), height, wall, roof
        )

    def hall(
        self,
        x: float,
        y: float,
        length: float,
        depth: float,
        height: float,
        wall: RGB,
        roof: RGB | None,
    ) -> tuple[Point, Point, Point, Point]:
        """Like `box`, but on an oblong footprint centred at (x, y): its lit wall spans `length`
        pixels across and its shaded wall `depth`."""
        front = (x + (length - depth) / 2, y + (length + depth) / 4)
        left = (front[0] - length, front[1] - length / 2)
        right = (front[0] + depth, front[1] - depth / 2)
        back = (left[0] + depth, left[1] - depth / 2)
        return self._prism(left, front, right, back, height, wall, roof)

    def _prism(
        self,
        left: Point,
        front: Point,
        right: Point,
        back: Point,
        h: float,
        wall: RGB,
        roof: RGB | None,
    ) -> tuple[Point, Point, Point, Point]:
        def up(p: Point) -> Point:
            return p[0], p[1] - h

        self.polygon([up(left), up(front), front, left], wall, shade(wall, 0.55))
        dark = shade(wall, 0.78)
        self.polygon([up(front), up(right), right, front], dark, shade(wall, 0.5))
        if roof is not None:
            self.polygon([up(left), up(front), up(right), up(back)], roof, shade(roof, 0.6))
        return up(left), up(front), up(right), up(back)

    def gable(
        self, top: tuple[Point, Point, Point, Point], rise: float, roof: RGB, wall: RGB
    ) -> None:
        """A pitched roof on a box's top, its ridge running from back-left to front-right."""
        left, front, right, back = top
        ridge_back = ((left[0] + back[0]) / 2, (left[1] + back[1]) / 2 - rise)
        ridge_front = ((front[0] + right[0]) / 2, (front[1] + right[1]) / 2 - rise)
        self.polygon([back, right, ridge_front, ridge_back], shade(roof, 0.7), shade(roof, 0.45))
        self.polygon([left, front, ridge_front, ridge_back], roof, shade(roof, 0.55))
        self.polygon([front, ridge_front, right], shade(wall, 0.78), shade(wall, 0.5))
        self.line([ridge_back, ridge_front], shade(roof, 1.25))

    def sprite(self) -> Sprite:
        return Sprite(self.image, self.anchor)


def pine(colors: tuple[RGB, RGB, RGB]) -> Sprite:
    dark, mid, light = colors
    p = Painter(11, 16, (5, 14))
    p.shadow(3, 1.5)
    p.line([(0, 0), (0, -2)], TRUNK)
    for i, half in enumerate((4, 3, 2)):
        base = -2 - 3.5 * i
        p.polygon([(-half, base), (half, base), (0, base - 5)], mid, dark)
        p.line([(-half + 1, base - 1), (-1, base - 4)], light)
    return p.sprite()


def leafy(colors: tuple[RGB, RGB, RGB]) -> Sprite:
    dark, mid, light = colors
    p = Painter(11, 14, (5, 12))
    p.shadow(3, 1.5)
    p.line([(0, 0), (0, -3)], TRUNK)
    p.ellipse(-4, -10, 4, -3, mid)
    p.blob(-3, -9, 0, -6, light)
    p.point(2, -5, dark)
    p.point(-1, -4, dark)
    return p.sprite()


def rocks() -> Sprite:
    p = Painter(14, 10, (7, 7))
    p.shadow(5, 1.5)
    p.polygon([(-6, 1), (-4, -4), (0, -6), (3, -2), (2, 1)], ROCK, shade(ROCK, 0.55))
    p.polygon([(-4, -4), (0, -6), (-1, -2)], shade(ROCK, 1.25))
    p.polygon([(1, 1), (3, -3), (6, -1), (5, 2)], shade(ROCK, 0.9), shade(ROCK, 0.5))
    return p.sprite()


def house(wall: RGB, roof: RGB) -> Sprite:
    p = Painter(15, 16, (7, 12))
    p.shadow(6, 2)
    top = p.box(0, 0, 6, 4, wall, None)
    p.gable(top, 4, roof, wall)
    p.point(-4, -2, WINDOW)
    p.point(-2, -1, WINDOW)
    p.point(3, -2, shade(WINDOW, 0.8))
    return p.sprite()


def tenement(wall: RGB, roof: RGB) -> Sprite:
    """A tenement (kamienica): three floors of painted front under a steep tiled roof."""
    p = Painter(17, 26, (8, 20))
    p.shadow(7, 2.5)
    s, floors = 6, 3
    top = p.box(0, 0, s, 3 * floors + 1, wall, None)
    p.gable(top, 5, roof, wall)
    for floor in range(floors):
        rise = 3 * floor + 2.5
        for t in (0.3, 0.7):
            p.point(-s + t * s, t * s / 2 - rise, WINDOW)
            p.point(t * s, s / 2 - t * s / 2 - rise, shade(WINDOW, 0.75))
    p.line([(-s, -3 * floors - 1), (0, s / 2 - 3 * floors - 1)], shade(wall, 1.15))
    return p.sprite()


def flats(wall: RGB) -> Sprite:
    """A block of flats: five floors under a flat roof, with rows of windows."""
    p = Painter(19, 32, (9, 26))
    p.shadow(8, 3, reach=5)
    s, floors = 7, 5
    p.box(0, 0, s, 3 * floors + 1, wall, shade(wall, 0.62))
    for floor in range(floors):
        rise = 3 * floor + 2.5
        for t in (0.2, 0.45, 0.7):
            p.point(-s + t * s * 1.2, t * s * 0.6 - rise, WINDOW)
            p.point(t * s * 1.2, s / 2 - t * s * 0.6 - rise, shade(WINDOW, 0.75))
    p.rect(-1, -3 * floors - 3, 1, -3 * floors - 1, shade(wall, 0.7))
    return p.sprite()


def farm() -> Sprite:
    p = Painter(22, 22, (9, 17))
    p.shadow(8, 3)
    barn = (172, 62, 46)
    top = p.box(-1, 0, 7, 6, barn, None)
    p.gable(top, 4, (112, 98, 84), barn)
    p.rect(-3, -4, -1, -1, shade(barn, 0.6))
    p.rect(7, -12, 10, 1, (206, 206, 200))
    p.ellipse(7, -14, 10, -11, METAL)
    return p.sprite()


def factory() -> Sprite:
    """A small brick works with one chimney."""
    p = Painter(20, 28, (9, 22))
    p.shadow(7, 2.5)
    p.box(0, 0, 7, 6, BRICK, (96, 94, 100))
    for k in range(2):
        x, y = -5 + 5 * k, -6 + 1.2 * k
        p.polygon([(x, y), (x + 5, y - 2.5), (x + 5, y + 1)], METAL, shade(METAL, 0.55))
    p.rect(3, -16, 5, -6, shade(BRICK, 0.8))
    p.line([(3, -14), (5, -14)], (220, 220, 214))
    for i, (x, y, r) in enumerate([(4, -18, 1.5), (6, -20, 2)]):
        p.blob(x - r, y - r, x + r, y + r, (226, 226, 230) if i % 2 else (204, 204, 210))
    return p.sprite()


def skyscraper() -> Sprite:
    """A glass tower with a grid of windows and an aerial on top."""
    p = Painter(17, 60, (8, 54))
    p.shadow(7, 2.5, reach=8)
    s, height = 6, 44
    p.box(0, 0, s, height, GLASS, shade(GLASS, 0.8))
    lit, dark = shade(GLASS, 1.35), shade(GLASS, 0.62)
    for floor in range(2, height - 1, 3):
        p.line([(-s, -floor), (0, s / 2 - floor)], lit)
        p.line([(0, s / 2 - floor), (s, -floor)], dark)
    p.line([(-s / 2, s / 4), (-s / 2, s / 4 - height)], shade(GLASS, 0.85))
    p.line([(0, -height - 1), (0, -height - 6)], METAL)
    return p.sprite()


def big_factory() -> Sprite:
    p = Painter(30, 40, (14, 33))
    p.shadow(12, 4)
    p.box(0, 0, 12, 8, BRICK, (96, 94, 100))
    for k in range(3):
        x = -9 + 6 * k
        y = -8 + 1.5 * k
        p.polygon([(x, y), (x + 6, y - 3), (x + 6, y + 1.5)], METAL, shade(METAL, 0.55))
        p.polygon([(x, y), (x, y + 1.5 - 3), (x + 6, y - 3)], shade(METAL, 1.15))
    for x in (-6, 4):
        p.rect(x, -24, x + 2, -9, shade(BRICK, 0.8))
        p.line([(x, -22), (x + 2, -22)], (220, 220, 214))
    for i, (x, y, r) in enumerate(
        [(-5, -27, 2), (-3, -30, 2.5), (0, -32, 3), (5, -27, 2), (7, -30, 2.5)]
    ):
        p.blob(x - r, y - r, x + r, y + r, (226, 226, 230) if i % 2 else (204, 204, 210))
    return p.sprite()


def mine() -> Sprite:
    p = Painter(28, 30, (12, 24))
    p.shadow(11, 3)
    heap = (52, 48, 50)
    p.polygon([(-11, 2), (-6, -5), (-1, -3), (1, 3)], heap, shade(heap, 0.6))
    p.polygon([(-6, -5), (-4, -4), (-8, 0)], shade(heap, 1.4))
    top = p.box(6, 1, 5, 4, (150, 132, 112), None)
    p.gable(top, 2, (92, 72, 62), (150, 132, 112))
    frame = (74, 64, 58)
    p.line([(0, -1), (3, -20), (6, -1)], frame)
    p.line([(1, -8), (5, -8)], frame)
    p.line([(1, -14), (5, -14)], frame)
    p.ellipse(1, -23, 5, -19, (96, 86, 78))
    return p.sprite()


def lumber_mill() -> Sprite:
    p = Painter(28, 24, (13, 19))
    p.shadow(11, 3)
    top = p.box(-3, 0, 7, 6, TIMBER, None)
    p.gable(top, 4, (96, 84, 70), TIMBER)
    log = (138, 96, 58)
    for k in range(3):
        y = 2 - 2 * k
        x = 5 + k
        p.rect(x, y - 2, x + 7, y, log)
        p.point(x + 7, y - 1, (214, 176, 122))
    p.point(6, -6, (60, 44, 30))
    return p.sprite()


def port() -> Sprite:
    p = Painter(30, 30, (14, 22))
    p.polygon([(-13, 0), (0, 6.5), (13, 0), (0, -6.5)], STONE, shade(STONE, 0.6))
    p.line([(-13, 1), (0, 7.5), (13, 1)], shade(STONE, 0.5))
    p.shadow(7, 2)
    top = p.box(-4, 0, 6, 6, (196, 186, 166), None)
    p.gable(top, 3, (64, 92, 124), (196, 186, 166))
    crane = (214, 168, 40)
    p.line([(7, 1), (7, -18)], crane)
    p.line([(6, 1), (6, -18)], shade(crane, 0.7))
    p.line([(4, -18), (14, -16)], crane)
    p.line([(13, -16), (13, -9)], (70, 64, 58))
    p.rect(12, -9, 14, -7, (120, 70, 50))
    return p.sprite()


def _station_ground(p: Painter, half: float = 15) -> None:
    """The paved platform of a station filling its cell, with a white edge along the track."""
    p.polygon([(-half, 1), (0, half / 2 + 1), (half, 1), (0, -half / 2 + 1)], shade(PLATFORM, 0.62))
    p.polygon(
        [(-half, 0), (0, half / 2), (half, 0), (0, -half / 2)], PLATFORM, shade(PLATFORM, 0.8)
    )
    p.line([(-half + 2, 0), (-1, half / 2 - 1.5)], PLATFORM_EDGE)


def _name_board(p: Painter, x: float, y: float, half: float = 3) -> None:
    """A blue station name board on two posts, its foot at (x, y)."""
    post = (52, 50, 54)
    for dx in (-half + 1, half - 1):
        p.line([(x + dx, y), (x + dx, y - 4)], post)
    p.rect(x - half, y - 7, x + half, y - 4, NAME_BOARD)
    for dx in range(-round(half) + 1, round(half), 1):
        if dx % 2 == 0 or abs(dx) < half - 1:
            p.point(x + dx, y - 5.5, (236, 240, 250))


def _lamp(p: Painter, x: float, y: float, height: float = 7) -> None:
    p.line([(x, y), (x, y - height)], (40, 40, 46))
    p.rect(x - 1, y - height - 2, x + 1, y - height, (255, 226, 120))


def _clock(p: Painter, x: float, y: float, radius: float = 1.5) -> None:
    p.ellipse(x - radius, y - radius, x + radius, y + radius, (246, 244, 236))
    p.line([(x, y), (x, y - radius + 0.5)], (30, 30, 34))
    p.point(x + 1, y, (30, 30, 34))


def _canopy(
    p: Painter, x: float, y: float, length: float, depth: float, height: float
) -> tuple[Point, Point, Point, Point]:
    """A platform canopy on posts: a flat dark red roof `height` up with a cream valance."""
    top = p.hall(x, y - height, length, depth, 1.5, STATION_WALL, STATION_ROOF)
    p.line([_on_wall(top[0], top[1], 0, 1.5), _on_wall(top[0], top[1], 1, 1.5)], STATION_ROOF)
    for t in np.linspace(0.08, 0.92, max(2, round(length / 4))):
        fx, fy = _on_wall(top[0], top[1], t, height + 1.5)
        p.line([(fx, fy), (fx, fy - height)], (60, 58, 62))
    return top


def station() -> Sprite:
    """A small country station: a cream station house with a dark red roof and brick base, a
    canopy over the platform, a blue name board and a lamp by the white platform edge."""
    p = Painter(34, 38, (16, 27))
    _station_ground(p)
    house = p.hall(*_iso(0, 0, 1, 3), 9, 5, 6, STATION_WALL, None)
    left, front, *_ = house
    _patch(p, left, front, (0, 1), (4.5, 6), BRICK)
    _windows(p, left, front, (0.2, 0.75), (1.5,), 2)
    _patch(p, left, front, (0.42, 0.58), (2, 6), (88, 60, 44))
    p.gable(house, 4, STATION_ROOF, STATION_WALL)
    chimney = _on_wall(house[0], house[3], 0.5, -5)
    p.rect(chimney[0] - 1, chimney[1] - 3, chimney[0] + 1, chimney[1], BRICK)
    _canopy(p, *_iso(0, 0, 0, -1.5), 10, 3, 5)
    _name_board(p, *_iso(0, 0, -5, -4.5))
    _lamp(p, *_iso(0, 0, 5.5, -5))
    return p.sprite()


def station_2() -> Sprite:
    """A town station: a two-storey cream building under a dark red roof with a clock in its
    central gable, and a long canopy over the platform with name boards and lamps."""
    p = Painter(34, 42, (16, 30))
    _station_ground(p)
    house = p.hall(*_iso(0, 0, 0, 4), 13, 5, 10, STATION_WALL, None)
    left, front, right, _ = house
    _patch(p, left, front, (0, 1), (9, 10), BRICK)
    _windows(p, left, front, (0.1, 0.25, 0.4, 0.6, 0.75, 0.9), (1.5, 5.5), 2)
    _patch(p, left, front, (0.44, 0.56), (6.5, 10), (88, 60, 44))
    _windows(p, front, right, (0.3, 0.7), (1.5, 5.5), 2, shade(WINDOW, 0.75))
    p.gable(house, 4, STATION_ROOF, STATION_WALL)
    gx, gy = _iso(0, 0, 0, 1.5)
    gable = p.hall(gx, gy, 5, 1, 10, STATION_WALL, None)
    _gable_across(p, gable, 4, STATION_ROOF, STATION_WALL)
    _clock(p, *_on_wall(gable[0], gable[1], 0.5, -1.5))
    _canopy(p, *_iso(0, 0, 0, -2.5), 14, 3, 6)
    for u in (-5.5, 2):
        _name_board(p, *_iso(0, 0, u, -5))
    for u in (-1.5, 6):
        _lamp(p, *_iso(0, 0, u, -5.5))
    return p.sprite()


def station_3() -> Sprite:
    """A grand terminal: a cream head building with a clock tower under a dark red spire, before
    an arched train shed of iron and glass over the platforms, with name boards and lamps."""
    p = Painter(38, 58, (18, 45))
    _station_ground(p)
    house = p.hall(*_iso(0, 0, 0, 5.5), 15, 4, 11, STATION_WALL, None)
    left, front, *_ = house
    _patch(p, left, front, (0, 1), (10, 11), BRICK)
    _windows(p, left, front, tuple(k / 10 for k in range(1, 10)), (1.5, 5.5), 2)
    p.gable(house, 3, STATION_ROOF, STATION_WALL)
    tx, ty = _iso(0, 0, -5, 5)
    tower = p.box(tx, ty, 2.5, 22, STATION_WALL, None)
    for a, b in ((tower[0], tower[1]), (tower[1], tower[2])):
        p.line([_on_wall(a, b, 0, 1), _on_wall(a, b, 1, 1)], shade(STATION_WALL, 1.08))
        _windows(p, a, b, (0.5,), (8, 13), 2)
    _clock(p, *_on_wall(tower[0], tower[1], 0.5, 4), 1.6)
    apex = (tx, ty - 31)
    p.polygon([tower[0], tower[1], apex], STATION_ROOF, shade(STATION_ROOF, 0.55))
    p.polygon([tower[1], tower[2], apex], shade(STATION_ROOF, 0.7), shade(STATION_ROOF, 0.45))
    _flag(p, apex[0], apex[1], 4, NAME_BOARD)
    glass, iron = (186, 220, 236), (70, 82, 100)
    shed = p.hall(*_iso(0, 0, 0, -1.5), 15, 8, 5, STATION_WALL, None)
    sl, sf, sr, _ = shed
    for t in np.linspace(0.07, 0.93, 7):
        p.line([_on_wall(sl, sf, t, 0.5), _on_wall(sl, sf, t, 5)], iron)
    _patch(p, sl, sf, (0, 1), (0, 1), STATION_ROOF)
    _vault(p, shed, 7, glass, glass, tuple(k / 8 for k in range(1, 8)), iron)
    _patch(p, sf, sr, (0.25, 0.75), (1.5, 5), (52, 56, 66))
    board = _on_wall(sf, sr, 0.5, 0.5)
    p.rect(board[0] - 3, board[1] - 2, board[0] + 3, board[1] + 1, NAME_BOARD)
    for dx in (-2, 0, 2):
        p.point(board[0] + dx, board[1] - 0.5, (236, 240, 250))
    _name_board(p, *_iso(0, 0, -4, -6.5))
    for u in (-0.5, 5):
        _lamp(p, *_iso(0, 0, u, -6.5))
    return p.sprite()


def orchard(fruit: list[RGB] = FRUIT) -> Sprite:
    """A fruit grower's cottage, with crates of the fruit stacked beside it; the trees themselves
    are planted on the ground (railroads.ui.pixel.world)."""
    p = Painter(22, 20, (9, 15))
    p.shadow(8, 3)
    wall, crate = (226, 208, 166), (156, 108, 62)
    top = p.box(-1, 0, 7, 5, wall, None)
    p.gable(top, 4, (124, 82, 54), wall)
    p.rect(-4, -3, -3, -1, shade(wall, 0.55))
    p.point(-6, -2, WINDOW)
    p.point(3, -2, shade(WINDOW, 0.8))
    p.rect(6, -2, 10, 1, crate)
    p.rect(7, -4, 9, -2, crate)
    for x, colour in ((7, 0), (9, 1), (8, 2)):
        p.point(x, -3 if x == 8 else -2, fruit[colour % len(fruit)])
    return p.sprite()


def date_palms() -> Sprite:
    """Three date palms on a patch of sand, their fronds hung with dates."""
    p = Painter(32, 26, (16, 20))
    p.polygon([(-14, 0), (0, 7), (14, 0), (0, -7)], (214, 192, 134), (186, 164, 108))
    for x, y in ((-7, 1), (2, -2), (8, 2)):
        p.line([(x, y), (x + 1, y - 6), (x + 1, y - 11)], (120, 86, 54))
        top = (x + 1, y - 12)
        for dx, dy in ((-5, 2), (5, 2), (-4, -2), (4, -2), (0, -4)):
            p.line([top, (top[0] + dx, top[1] + dy)], (64, 128, 56))
        p.point(top[0] - 1, top[1] + 1, (110, 62, 36))
        p.point(top[0] + 1, top[1] + 1, (110, 62, 36))
    return p.sprite()


def bananas() -> Sprite:
    """Rows of broad-leaved banana plants with yellow bunches."""
    p = Painter(32, 22, (16, 14))
    p.polygon([(-14, 0), (0, 7), (14, 0), (0, -7)], (74, 120, 48), (60, 100, 40))
    for row in range(2):
        for col in range(3):
            x = -8 + 7 * col + 3 * row - 2
            y = -1 + 3.5 * row - 1.5 * col + 1
            p.line([(x, y), (x, y - 4)], (96, 120, 52))
            p.blob(x - 4, y - 8, x, y - 4, (52, 116, 44))
            p.blob(x, y - 9, x + 4, y - 5, (66, 140, 54))
            p.point(x + 1, y - 3, (236, 206, 64))
    return p.sprite()


def sugar() -> Sprite:
    """A field of tall sugar cane in rows."""
    p = Painter(30, 18, (15, 11))
    p.polygon([(-13, 0), (0, 6), (13, 0), (0, -6)], (132, 112, 72), (112, 94, 60))
    for row in range(4):
        for col in range(5):
            x = -9 + 4 * col + 2 * row - 3
            y = -3 + 2 * row - 1 * col + 2
            p.line([(x, y), (x, y - 4)], (118, 176, 82))
            p.point(x - 1, y - 4, (166, 206, 104))
    return p.sprite()


def shaft_mine(heap: RGB, glint: RGB) -> Sprite:
    """A pithead over a heap of what it digs."""
    p = Painter(28, 30, (12, 24))
    p.shadow(11, 3)
    p.polygon([(-11, 2), (-6, -5), (-1, -3), (1, 3)], heap, shade(heap, 0.6))
    for x, y in ((-7, -2), (-4, -3), (-5, 0), (-2, 0)):
        p.point(x, y, glint)
    top = p.box(6, 1, 5, 4, (150, 132, 112), None)
    p.gable(top, 2, (92, 72, 62), (150, 132, 112))
    frame = (74, 64, 58)
    p.line([(0, -1), (3, -20), (6, -1)], frame)
    p.line([(1, -8), (5, -8)], frame)
    p.line([(1, -14), (5, -14)], frame)
    p.ellipse(1, -23, 5, -19, glint)
    return p.sprite()


def oil_well() -> Sprite:
    """A timber derrick beside a storage tank."""
    p = Painter(28, 36, (12, 30))
    p.shadow(10, 3)
    frame = (70, 58, 48)
    p.line([(-7, 1), (-2, -26), (3, 1)], frame)
    for y in (-6, -12, -18):
        half = 4 + y / 6
        p.line([(-2 - half, y), (-2 + half, y)], frame)
    p.ellipse(5, -6, 13, -1, (40, 38, 44))
    p.rect(5, -4, 13, 2, (52, 50, 58))
    p.ellipse(5, -1, 13, 3, (40, 38, 44))
    return p.sprite()


def salt_works() -> Sprite:
    """White heaps of salt beside a low shed."""
    p = Painter(30, 22, (15, 16))
    p.shadow(12, 3)
    white = (232, 234, 238)
    p.polygon([(-13, 3), (-8, -6), (-3, 3)], white, shade(white, 0.7))
    p.polygon([(-5, 4), (-1, -3), (3, 4)], white, shade(white, 0.7))
    top = p.box(8, 1, 5, 4, (170, 152, 124), None)
    p.gable(top, 2, (112, 92, 74), (170, 152, 124))
    return p.sprite()


def fishery() -> Sprite:
    """A hut with racks of drying fish and a small boat."""
    p = Painter(30, 22, (15, 16))
    p.shadow(12, 3)
    top = p.box(-5, 0, 6, 5, (132, 112, 88), None)
    p.gable(top, 3, (84, 106, 130), (132, 112, 88))
    p.polygon([(3, 3), (13, 3), (11, 6), (5, 6)], (150, 62, 48), (92, 40, 34))
    p.line([(8, 3), (8, -6)], (88, 64, 44))
    for y in (-3, -1):
        p.line([(3, y), (6, y)], (120, 170, 196))
    return p.sprite()


def pasture() -> Sprite:
    """A shed on hilly grazing with sheep around it."""
    p = Painter(30, 22, (15, 16))
    top = p.box(-5, 0, 5, 4, (152, 124, 92), None)
    p.gable(top, 3, (110, 80, 60), (152, 124, 92))
    for x, y in ((3, -2), (8, 1), (4, 4), (11, -3), (-9, 5)):
        p.blob(x - 2, y - 1, x + 2, y + 1, (240, 236, 224))
        p.point(x + 2, y - 1, (50, 46, 44))
    return p.sprite()


def plantation(wall: RGB, roof: RGB) -> Sprite:
    """The small barn that stands on a field or grove."""
    p = Painter(18, 16, (9, 11))
    p.shadow(6, 2)
    top = p.box(0, 0, 5, 4, wall, None)
    p.gable(top, 3, roof, wall)
    return p.sprite()


def works(wall: RGB, roof: RGB, sign: RGB, stacks: int) -> Sprite:
    """A works: a long hall in its own colours with chimneys and a sign."""
    p = Painter(32, 40, (15, 32))
    p.shadow(13, 4)
    p.box(0, 0, 12, 8, wall, roof)
    p.rect(-9, -5, -4, -2, sign)
    p.point(-3, -3, WINDOW)
    p.point(5, -2, WINDOW)
    for k in range(stacks):
        x = 2 + 4 * k
        p.rect(x, -22, x + 2, -9, shade(BRICK, 0.8))
        p.blob(x - 1, -26, x + 3, -22, (150, 148, 150))
    return p.sprite()


def shop(wall: RGB, awning: RGB, goods: RGB) -> Sprite:
    """A smart city shop with a bright awning and its wares in the window."""
    p = Painter(24, 26, (11, 20))
    p.shadow(9, 3)
    p.box(0, 0, 8, 12, wall, shade(wall, 0.7))
    p.polygon([(-8, -6), (0, -2), (0, -5), (-8, -9)], awning, shade(awning, 0.6))
    p.rect(-6, -5, -2, -1, WINDOW)
    p.point(-4, -3, goods)
    p.point(-3, -2, goods)
    return p.sprite()


def _flag(p: Painter, x: float, y: float, height: float, colour: RGB = MILITARY_FLAG) -> None:
    p.line([(x, y), (x, y - height)], (70, 66, 62))
    p.polygon(
        [(x + 1, y - height), (x + 7, y - height + 2), (x + 1, y - height + 4)],
        colour,
    )


def outpost() -> Sprite:
    """A hut behind an earth bank, under a flag."""
    p = Painter(26, 30, (13, 23))
    p.shadow(10, 3)
    p.box(0, 0, 10, 2, (122, 94, 62), MILITARY_YARD)
    p.box(0, -1, 5, 6, MILITARY_YARD, MILITARY_ROOF)
    _flag(p, 7, -2, 16)
    return p.sprite()


def base() -> Sprite:
    """Barracks in a walled yard with a watchtower."""
    p = Painter(34, 34, (17, 26))
    p.shadow(14, 4)
    p.box(0, 0, 15, 3, STONE, MILITARY_YARD)
    p.box(-6, -2, 4, 6, MILITARY_YARD, MILITARY_ROOF)
    p.box(5, 0, 4, 6, MILITARY_YARD, MILITARY_ROOF)
    p.rect(9, -15, 12, -2, shade(STONE, 0.9))
    p.rect(8, -18, 13, -15, MILITARY_ROOF)
    _flag(p, -11, -2, 20)
    return p.sprite()


def headquarters() -> Sprite:
    """A great stone block with rows of windows, in a walled yard, under a tall flag."""
    p = Painter(40, 46, (19, 36))
    p.shadow(17, 5)
    p.box(0, 0, 18, 3, STONE, MILITARY_YARD)
    p.box(0, -2, 11, 15, (178, 172, 146), MILITARY_ROOF)
    for row in range(3):
        for col in range(3):
            p.point(-9 + 3 * col, -13 + 4 * row + 1.5 * col, WINDOW)
    _flag(p, 0, -22, 14)
    return p.sprite()


def airport() -> Sprite:
    """A runway, a hangar and a control tower."""
    p = Painter(42, 34, (21, 24))
    p.polygon([(-20, 1), (-1, 9), (20, 0), (1, -8)], (112, 114, 108), (88, 90, 86))
    p.line([(-15, 2), (-1, 8)], (232, 232, 224))
    p.box(-5, -2, 7, 7, (208, 210, 212), (146, 150, 158))
    p.rect(8, -18, 11, -1, (192, 194, 198))
    p.rect(6, -22, 13, -18, GLASS)
    return p.sprite()


def _barrel(p: Painter, x: float, y: float) -> None:
    p.ellipse(x - 2, y - 6, x + 2, y, BARREL)
    p.line([(x - 2, y - 2), (x + 2, y - 2)], shade(BARREL, 0.55))
    p.line([(x - 2, y - 4), (x + 2, y - 4)], shade(BARREL, 0.55))


def _on_wall(a: Point, b: Point, t: float, below: float) -> Point:
    """The point `t` of the way along a wall's top edge from corner `a` to corner `b`, `below`
    pixels down from that edge."""
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t + below


def _patch(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: tuple[float, float], fill: RGB
) -> None:
    """A flat patch on the wall under the edge from `a` to `b`, such as a door or a band of glass:
    from `span[0]` to `span[1]` of the way along it, and `drop[0]` to `drop[1]` pixels down."""
    (t0, t1), (d0, d1) = span, drop
    p.polygon(
        [
            _on_wall(a, b, t0, d0),
            _on_wall(a, b, t1, d0),
            _on_wall(a, b, t1, d1),
            _on_wall(a, b, t0, d1),
        ],
        fill,
    )


def _windows(
    p: Painter,
    a: Point,
    b: Point,
    spans: tuple[float, ...],
    drops: tuple[float, ...],
    tall: float = 1,
    colour: RGB = WINDOW,
    arch: RGB | None = None,
) -> None:
    """Windows `tall` pixels high on the wall under the edge from `a` to `b`, at each of `spans`
    along it and `drops` down it, with a light arch over each if `arch` is given."""
    for drop in drops:
        for t in spans:
            p.line([_on_wall(a, b, t, drop), _on_wall(a, b, t, drop + tall - 1)], colour)
            if arch is not None:
                p.point(*_on_wall(a, b, t, drop - 1), arch)


def _sawtooth(
    p: Painter,
    top: tuple[Point, Point, Point, Point],
    teeth: int,
    rise: float,
    roof: RGB,
    glass: RGB,
    wall: RGB,
) -> None:
    """A sawtooth roof over a box's top: `teeth` ridges across it, each sloping down towards the
    front from a strip of glazing along its crest."""
    left, front, right, back = top
    for k in range(teeth):
        t0, t1 = k / teeth, (k + 1) / teeth
        near_up, far_up = _on_wall(left, front, t0, -rise), _on_wall(back, right, t0, -rise)
        near, far = _on_wall(left, front, t1, 0), _on_wall(back, right, t1, 0)
        p.polygon([_on_wall(left, front, t0, 0), near_up, near], wall, shade(wall, 0.55))
        p.polygon([near_up, far_up, far, near], roof, shade(roof, 0.6))
        p.line([near_up, far_up], glass)


def _vault(
    p: Painter,
    top: tuple[Point, Point, Point, Point],
    rise: float,
    roof: RGB,
    wall: RGB,
    ribs: tuple[float, ...] = (0.25, 0.5, 0.75),
    rib: RGB | None = None,
) -> None:
    """An arched roof over a box's top, running along its lit wall, with ribs across it at
    `ribs` of the way along, in `rib` or a light shade of the roof."""
    left, front, right, back = top

    def arch(a: Point, b: Point, t: float) -> Point:
        return _on_wall(a, b, t, -rise * math.sin(math.pi * t))

    steps = 6
    for i in reversed(range(steps)):
        t0, t1 = i / steps, (i + 1) / steps
        tone = shade(roof, 1.25 - 0.55 * (t0 + t1) / 2)
        p.polygon(
            [
                arch(left, back, t0),
                arch(left, back, t1),
                arch(front, right, t1),
                arch(front, right, t0),
            ],
            tone,
        )
    for t in ribs:
        near, far = _on_wall(left, front, t, 0), _on_wall(back, right, t, 0)
        p.line([arch(near, far, i / 8) for i in range(9)], rib or shade(roof, 1.4))
    p.polygon([arch(front, right, i / 12) for i in range(13)], shade(wall, 0.78), shade(wall, 0.5))


def _turned(
    p: Painter,
    x: float,
    y: float,
    profile: list[tuple[float, float]],
    colour: RGB,
    cap: RGB | None = None,
) -> None:
    """A round thing standing on an upright axis at (x, y), like a tank, a kiln or a dome:
    `profile` gives its half-width at each height from the foot up. It is lit on the left, and
    closed with an oval top in `cap` if given."""
    split = math.acos(1 / 3)
    foot = profile[0][1]
    lit_arc = [
        (x + foot * math.cos(a), y + foot / 2 * math.sin(a)) for a in np.linspace(split, math.pi, 6)
    ]
    dark_arc = [
        (x + foot * math.cos(a), y + foot / 2 * math.sin(a)) for a in np.linspace(0, split, 4)
    ]
    left = [(x - w, y - h) for h, w in profile]
    middle = [(x + w / 3, y - h) for h, w in profile]
    right = [(x + w, y - h) for h, w in profile]
    p.polygon(lit_arc + left + middle[::-1], colour)
    p.polygon(dark_arc + middle + right[::-1], shade(colour, 0.78))
    edge = shade(colour, 0.55)
    p.line(left, edge)
    p.line(right, edge)
    p.line(dark_arc + lit_arc, edge)
    height, width = profile[-1]
    if cap is not None and width > 0:
        p.ellipse(x - width, y - height - width / 2, x + width, y - height + width / 2, cap)


def _chimney(p: Painter, x: float, foot: float, top: float, colour: RGB = BRICK) -> None:
    """A round brick chimney narrowing to a dark cap."""
    p.polygon(
        [(x - 2, foot), (x - 1, top), (x + 1, top), (x + 2, foot)], colour, shade(colour, 0.6)
    )
    p.line([(x - 1, foot - 1), (x - 1, top + 1)], shade(colour, 1.2))
    p.rect(x - 2, top - 1, x + 2, top, shade(colour, 0.7))


def _smoke(p: Painter, x: float, y: float, colour: RGB = (150, 148, 150), puffs: int = 3) -> None:
    for i in range(puffs):
        r = 1.5 + 0.6 * i
        cx, cy = x + 1.5 * i, y - 2.5 * i
        p.blob(cx - r, cy - r, cx + r, cy + r, colour if i % 2 else shade(colour, 0.86))


def _heap(p: Painter, x: float, y: float, half: float, height: float, colour: RGB) -> None:
    """A conical heap of loose stuff, lit on the left."""
    foot = (x + half * 0.3, y + half * 0.3)
    p.polygon([(x - half, y), (x - 1, y - height), foot], colour, shade(colour, 0.62))
    p.polygon(
        [(x - 1, y - height), (x + 1, y - height), (x + half, y), foot],
        shade(colour, 0.8),
        shade(colour, 0.55),
    )


def _gear(p: Painter, x: float, y: float, radius: float, colour: RGB) -> None:
    for k in range(8):
        a = k * math.pi / 4
        tx, ty = x + (radius + 0.8) * math.cos(a), y + (radius + 0.8) * math.sin(a)
        p.blob(tx - 1, ty - 1, tx + 1, ty + 1, shade(colour, 0.8))
    p.ellipse(x - radius, y - radius, x + radius, y + radius, colour)
    p.blob(x - 1, y - 1, x + 1, y + 1, shade(colour, 0.45))


def _bale(p: Painter, x: float, y: float, colour: RGB) -> None:
    p.box(x, y, 2, 2, colour, shade(colour, 1.15))


def barrel_maker() -> Sprite:
    """A cooper's shed with finished barrels stood outside."""
    p = Painter(28, 22, (13, 16))
    p.shadow(11, 3)
    top = p.box(-3, 0, 7, 5, TIMBER, None)
    p.gable(top, 3, (118, 84, 54), TIMBER)
    for x, y in ((7, -1), (11, 1), (6, 3)):
        _barrel(p, x, y)
    return p.sprite()


def beekeeper() -> Sprite:
    """Straw hives in rows on a patch of flowers."""
    p = Painter(28, 18, (14, 11))
    p.polygon([(-12, 0), (0, 6), (12, 0), (0, -6)], (128, 168, 84), (104, 142, 70))
    for x, y in ((-6, 1), (0, 4), (0, -2), (6, 1)):
        p.ellipse(x - 2.5, y - 6, x + 2.5, y, HIVE)
        p.point(x, y - 1, shade(HIVE, 0.4))
    for x, y in ((-9, 0), (-3, -3), (4, -5), (9, -2), (3, 4)):
        p.point(x, y, (240, 226, 120))
    return p.sprite()


def incense_grove() -> Sprite:
    """Gnarled resin trees on pale ground."""
    p = Painter(28, 20, (14, 13))
    p.polygon([(-12, 0), (0, 6), (12, 0), (0, -6)], (190, 170, 130), (160, 142, 106))
    for x, y in ((-6, 0), (0, 3), (6, -1)):
        p.line([(x, y), (x - 1, y - 3), (x + 1, y - 5)], TRUNK)
        p.blob(x - 3, y - 9, x + 3, y - 5, (108, 124, 76))
        p.point(x - 1, y - 2, (200, 160, 210))
    return p.sprite()


def _iso(x: float, y: float, u: float, v: float) -> Point:
    """The point `u` pixels across along a lit wall and `v` along a shaded one from (x, y)."""
    return x + u + v, y + u / 2 - v / 2


def _gable_across(
    p: Painter, top: tuple[Point, Point, Point, Point], rise: float, roof: RGB, wall: RGB
) -> None:
    """A pitched roof on a box's top whose ridge runs back from the middle of its lit wall, so
    that wall ends in a gable: the front of a church."""
    left, front, right, back = top
    near, far = _on_wall(left, front, 0.5, -rise), _on_wall(back, right, 0.5, -rise)
    p.polygon([left, back, far, near], roof, shade(roof, 0.55))
    p.polygon([front, right, far, near], shade(roof, 0.72), shade(roof, 0.45))
    p.polygon([left, front, near], wall, shade(wall, 0.55))
    p.line([near, far], shade(roof, 1.25))


def _cross(p: Painter, x: float, y: float, height: int = 6) -> None:
    """A gold cross standing on (x, y), two pixels thick so it keeps its shape when shrunk."""
    x, y = round(x), round(y)
    bar = y - round(height * 0.72)
    p.polygon([(x, y), (x, y - height), (x + 1, y - height), (x + 1, y)], CROSS)
    p.polygon([(x - 2, bar), (x + 3, bar), (x + 3, bar + 1), (x - 2, bar + 1)], CROSS)
    p.line([(x + 1, y), (x + 1, bar + 2)], shade(CROSS, 0.78))


def _rose(p: Painter, x: float, y: float, radius: float) -> None:
    """A round rose window of stained glass."""
    p.ellipse(x - radius, y - radius, x + radius, y + radius, STAINED)
    p.blob(x - radius / 2, y - radius / 2, x + radius / 2, y + radius / 2, shade(STAINED, 1.7))
    p.point(x, y, CROSS)


def _portal(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float, tall: float
) -> None:
    """A dark arched doorway on the wall under the edge from `a` to `b`, its foot `drop` down."""
    _patch(p, a, b, span, (drop - tall + 1, drop), DOORWAY)
    p.line(
        [
            _on_wall(a, b, span[0] + 0.25 * (span[1] - span[0]), drop - tall),
            _on_wall(a, b, span[1] - 0.25 * (span[1] - span[0]), drop - tall),
        ],
        DOORWAY,
    )


def _tower(
    p: Painter,
    x: float,
    y: float,
    half: float,
    height: float,
    spire: float,
    roof: RGB,
    wall: RGB = CHURCH_WALL,
) -> tuple[Point, Point, Point, Point]:
    """A square church tower with arched belfry openings under a pyramid spire and a gold cross.
    Returns the corners of its top."""
    top = p.box(x, y, half, height, wall, None)
    left, front, right, _ = top
    openings = (0.5,) if half < 3 else (0.33, 0.67)
    _windows(p, left, front, openings, (2,), 3, DOORWAY)
    _windows(p, front, right, openings, (2,), 3, shade(DOORWAY, 0.8))
    p.line([left, front, right], shade(wall, 1.12))
    apex = (x, y - height - spire)
    p.polygon([left, front, apex], roof, shade(roof, 0.55))
    p.polygon([front, right, apex], shade(roof, 0.7), shade(roof, 0.45))
    p.line([front, apex], shade(roof, 1.3))
    _cross(p, apex[0], apex[1] + 1)
    return top


def church() -> Sprite:
    """A whitewashed village church: a nave under a steep tiled roof with tall arched windows,
    and a bell tower with a round-arched door and a slate spire topped by a gold cross."""
    p = Painter(30, 46, (14, 36))
    p.shadow(12, 3.5)
    top = p.hall(-3, -1, 12, 7, 7, WHITEWASH, None)
    left, front, *_ = top
    _windows(p, left, front, (0.18, 0.42, 0.66), (1.5,), 4, STAINED)
    p.gable(top, 6, CHURCH_ROOF, WHITEWASH)
    tower = _tower(p, 4, 2.5, 3.5, 14, 11, SLATE, WHITEWASH)
    _portal(p, tower[0], tower[1], (0.3, 0.7), 14, 4)
    _rose(p, *_on_wall(tower[0], tower[1], 0.5, 7), 1)
    return p.sprite()


def basilica() -> Sprite:
    """A Romanesque basilica: twin bell towers with crosses flank a front with a rose window over
    a great arched door, before a tall nave lit by clerestory windows under a copper-green dome."""
    p = Painter(40, 52, (19, 40))
    p.shadow(16, 5)
    cx, cy = 1, -2
    depth, nave, aisle = 16, 7, 4
    side = (nave + aisle) / 2
    p.hall(*_iso(cx, cy, -side, 0), aisle, depth, 8, CHURCH_WALL, shade(CHURCH_ROOF, 0.9))
    _tower(p, *_iso(cx, cy, -5.5, -5.5), 2.5, 20, 4, CHURCH_ROOF)
    top = p.hall(cx, cy, nave, depth, 14, CHURCH_WALL, None)
    left, front, right, _ = top
    _portal(p, left, front, (0.25, 0.75), 14, 6)
    _windows(p, front, right, (0.45, 0.6, 0.75, 0.9), (2,), 3, shade(STAINED, 0.8))
    _gable_across(p, top, 4, CHURCH_ROOF, CHURCH_WALL)
    _rose(p, *_on_wall(left, front, 0.5, 2.5), 2)
    dome_x, dome_y = _iso(cx, cy, 0, 4.5)
    _turned(p, dome_x, dome_y - 16, [(0, 3.5), (4, 3.5)], CHURCH_WALL)
    _windows(p, (dome_x - 3, dome_y - 20), (dome_x + 3, dome_y - 20), (0.2, 0.5), (1,), 2, STAINED)
    _turned(p, dome_x, dome_y - 20, [(0, 4), (2, 3.6), (3.5, 2.6), (4.5, 1.2), (5, 0)], VERDIGRIS)
    _cross(p, dome_x, dome_y - 25)
    right_aisle = p.hall(
        *_iso(cx, cy, side, 0), aisle, depth, 8, CHURCH_WALL, shade(CHURCH_ROOF, 0.9)
    )
    _windows(p, right_aisle[1], right_aisle[2], (0.45, 0.65, 0.85), (2,), 3, shade(STAINED, 0.8))
    _tower(p, *_iso(cx, cy, 5.5, -5.5), 2.5, 20, 4, CHURCH_ROOF)
    return p.sprite()


def cathedral() -> Sprite:
    """A Gothic cathedral: two tall spires with crosses flank a great rose window over a pointed
    door, before a long nave held up by flying buttresses and lit by tall pointed windows."""
    p = Painter(50, 66, (24, 53))
    p.shadow(20, 6)
    cx, cy = 0, -3
    depth, nave, aisle = 22, 8, 4
    side = (nave + aisle) / 2
    roof = SLATE
    p.hall(*_iso(cx, cy, -side, 0), aisle, depth, 9, CHURCH_WALL, shade(roof, 0.9))
    _tower(p, *_iso(cx, cy, -5.5, -8.5), 3, 27, 15, roof)
    top = p.hall(cx, cy, nave, depth, 17, CHURCH_WALL, None)
    left, front, right, _ = top
    _portal(p, left, front, (0.3, 0.7), 17, 6)
    _windows(p, front, right, (0.4, 0.52, 0.64, 0.76, 0.88), (2,), 5, shade(STAINED, 0.8))
    _gable_across(p, top, 7, roof, CHURCH_WALL)
    _rose(p, *_on_wall(left, front, 0.5, 4.5), 2.5)
    right_aisle = p.hall(*_iso(cx, cy, side, 0), aisle, depth, 9, CHURCH_WALL, shade(roof, 0.9))
    _windows(p, right_aisle[1], right_aisle[2], (0.46, 0.64, 0.82), (2,), 5, shade(STAINED, 0.8))
    buttress = shade(CHURCH_WALL, 0.82)
    for v in (-2, 2, 6, 10):
        foot_x, foot_y = _iso(cx, cy, nave / 2 + aisle + 0.5, v)
        wall_x, wall_y = _iso(cx, cy, nave / 2, v)
        p.rect(foot_x - 1, foot_y - 12, foot_x, foot_y, buttress)
        p.line([(foot_x, foot_y - 11), (wall_x + 1, wall_y - 15)], buttress)
        p.line([(foot_x - 1, foot_y - 12), (wall_x, wall_y - 16)], shade(buttress, 1.15))
        p.line([(foot_x - 1, foot_y - 12), (foot_x - 1, foot_y - 14)], shade(buttress, 1.15))
    _tower(p, *_iso(cx, cy, 5.5, -8.5), 3, 27, 15, roof)
    return p.sprite()


def st_pauls() -> Sprite:
    """A Baroque cathedral like St Paul's: a great lead dome on a columned drum, crowned by a
    lantern and a gold cross, over a long body fronted by a columned portico between two west
    towers."""
    p = Painter(60, 74, (29, 61))
    p.shadow(25, 7)
    cx, cy = 3, -3
    stone = CHURCH_WALL

    def west_tower(u: float) -> None:
        tx, ty = _iso(cx, cy, u, -11)
        tower = p.box(tx, ty, 3, 23, stone, None)
        _windows(p, tower[0], tower[1], (0.33, 0.67), (3,), 3, DOORWAY)
        _windows(p, tower[1], tower[2], (0.33, 0.67), (3,), 3, shade(DOORWAY, 0.8))
        _turned(p, tx, ty - 23, [(0, 2.5), (3, 2.5)], stone)
        _turned(p, tx, ty - 26, [(0, 2.8), (2, 2), (3, 1), (3.5, 0)], LEAD)
        _cross(p, tx, ty - 28)

    west_tower(-9.5)
    top = p.hall(cx, cy, 14, 28, 13, stone, None)
    _, front, right, _ = top
    for t in range(1, 10):
        p.line(
            [_on_wall(front, right, t / 10, 1), _on_wall(front, right, t / 10, 13)],
            shade(stone, 0.68),
        )
    _windows(p, front, right, (0.15, 0.35, 0.55, 0.75, 0.95), (3,), 4, shade(DOORWAY, 0.8))
    _gable_across(p, top, 3, LEAD, stone)
    dome_x, dome_y = _iso(cx, cy, 0, 5)
    drum = dome_y - 14
    _turned(p, dome_x, drum, [(0, 9), (8, 9)], stone)
    for a in np.linspace(0.25, math.pi - 0.25, 9):
        col_x, col_y = dome_x + 9 * math.cos(a), drum + 4.5 * math.sin(a)
        p.line([(col_x, col_y - 1), (col_x, col_y - 7)], shade(stone, 0.62))
    _turned(p, dome_x, drum - 8, [(0, 10), (3, 9.6), (6, 8.3), (9, 5.8), (11, 2.8), (12, 0)], LEAD)
    p.blob(dome_x - 7, drum - 17, dome_x - 2, drum - 11, shade(LEAD, 1.3))
    _turned(p, dome_x, drum - 19, [(0, 1.8), (4, 1.8)], stone)
    _turned(p, dome_x, drum - 23, [(0, 2), (1.5, 1), (2, 0)], LEAD)
    _cross(p, dome_x, drum - 24, 7)
    portico = p.hall(*_iso(cx, cy, 0, -15.5), 10, 3, 9, stone, None)
    for t in range(1, 10, 2):
        p.line(
            [
                _on_wall(portico[0], portico[1], t / 10, 1),
                _on_wall(portico[0], portico[1], t / 10, 9),
            ],
            DOORWAY,
        )
    _gable_across(p, portico, 3, LEAD, stone)
    west_tower(9.5)
    return p.sprite()


def bakery() -> Sprite:
    """A baker's house with a brick oven chimney and a loaf hung over the door."""
    p = Painter(18, 22, (8, 17))
    p.shadow(7, 2.5)
    wall = (232, 214, 176)
    top = p.box(0, 0, 6, 5, wall, None)
    p.gable(top, 4, (150, 72, 52), wall)
    p.rect(2, -13, 4, -6, BRICK)
    for x, y, r in ((3, -15, 1.2), (4, -17, 1.6)):
        p.blob(x - r, y - r, x + r, y + r, (224, 222, 226))
    p.rect(-4, -3, -3, -1, shade(wall, 0.55))
    p.ellipse(-6, -6, -3, -4, (196, 128, 58))
    p.point(3, -2, WINDOW)
    return p.sprite()


def food_plant() -> Sprite:
    """A cannery: a pale hall with a green roof and two tall silos."""
    p = Painter(26, 32, (12, 25))
    p.shadow(10, 3)
    wall = (222, 214, 192)
    p.box(-2, 0, 8, 6, wall, (86, 128, 84))
    for x in (6, 10):
        p.rect(x - 2, -18, x + 1, 1, shade(METAL, 1.1))
        p.ellipse(x - 2, -20, x + 1, -16, shade(METAL, 1.25))
    p.point(-6, -2, WINDOW)
    p.point(-3, -1, WINDOW)
    return p.sprite()


def steel_mill() -> Sprite:
    """A dark mill with a blast furnace glowing at its foot and a plume of smoke."""
    p = Painter(32, 46, (14, 38))
    p.shadow(12, 4)
    iron = (82, 80, 84)
    p.box(0, 0, 11, 9, iron, (60, 58, 62))
    p.rect(4, -26, 9, -6, shade(iron, 0.85))
    p.polygon([(4, -26), (6.5, -30), (9, -26)], shade(iron, 0.7))
    p.rect(-9, -12, -6, -2, (238, 132, 40))
    p.point(-8, -10, (255, 214, 120))
    p.rect(-3, -30, -1, -9, shade(BRICK, 0.7))
    for i, (x, y, r) in enumerate([(-2, -33, 2), (0, -36, 3), (3, -39, 3.5)]):
        p.blob(x - r, y - r, x + r, y + r, (150, 148, 150) if i % 2 else (124, 122, 126))
    return p.sprite()


def textile_mill() -> Sprite:
    """A red-brick cotton mill of Łódź: a long block of storeys with rows of small arched windows,
    a sawtooth weaving shed at its end, a tall round chimney and bales of cloth by the door."""
    p = Painter(37, 55, (17, 44))
    p.shadow(15, 5, reach=4)
    wall, roof = (178, 78, 58), (80, 76, 84)
    _chimney(p, 11, 0, -32, shade(wall, 0.92))
    _smoke(p, 12, -35)
    left, front, right, _ = p.hall(-4, -1, 14, 8, 15, wall, roof)
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], shade(wall, 1.2))
    lintel = shade(wall, 1.3)
    _windows(p, left, front, (0.1, 0.3, 0.5, 0.7, 0.9), (3.5, 7, 10.5), 2, arch=lintel)
    _windows(p, left, front, (0.1, 0.3, 0.5), (14,), 1, arch=lintel)
    _windows(p, front, right, (0.3, 0.7), (3.5, 7), 2, shade(WINDOW, 0.75), lintel)
    _patch(p, left, front, (0.68, 0.84), (12, 15), (60, 44, 36))
    shed = p.box(7, 4.5, 8, 5, wall, None)
    _sawtooth(p, shed, 3, 4, (104, 100, 108), (160, 196, 218), wall)
    _windows(p, shed[0], shed[1], (0.2, 0.5, 0.8), (2.5,), 1, arch=lintel)
    for x, y, colour in ((-7, 6, CLOTH[0]), (-3.5, 7.5, CLOTH[1]), (-7, 4, CLOTH[2])):
        _bale(p, x, y, colour)
    return p.sprite()


def furniture_maker() -> Sprite:
    """A joiner's plank-walled workshop with sawn boards stacked in its yard and a chair standing
    on its roof for a sign."""
    p = Painter(30, 32, (14, 23))
    p.shadow(12, 3.5)
    wall, board = (176, 134, 88), (226, 190, 134)
    for x, y, h in ((9, -3, 6), (7, 2, 4)):
        stack = p.hall(x, y, 8, 3, h, board, shade(board, 1.12))
        gap = shade(board, 0.5)
        for drop in range(2, h, 2):
            p.line(
                [_on_wall(stack[0], stack[1], 0, drop), _on_wall(stack[0], stack[1], 1, drop)], gap
            )
            p.line(
                [_on_wall(stack[1], stack[2], 0, drop), _on_wall(stack[1], stack[2], 1, drop)], gap
            )
    top = p.box(-4, 0, 7, 6, wall, None)
    left, front, *_ = top
    for drop in (2, 4):
        p.line([_on_wall(left, front, 0, drop), _on_wall(left, front, 1, drop)], shade(wall, 0.86))
    _patch(p, left, front, (0.45, 0.82), (1.5, 6), (56, 40, 30))
    p.point(*_on_wall(left, front, 0.22, 2.5), WINDOW)
    p.gable(top, 4, (98, 74, 58), wall)
    chair = (156, 62, 44)
    x0, y0 = -6, -10
    p.polygon(
        [
            (x0 + dx, y0 + dy)
            for dx, dy in (
                (0, 0),
                (0, -7),
                (1, -7),
                (1, -3),
                (5, -3),
                (5, 0),
                (4, 0),
                (4, -2),
                (1, -2),
                (1, 0),
            )
        ],
        chair,
    )
    return p.sprite()


def tool_works() -> Sprite:
    """A dark forge hall with its door open on a glowing hearth, an anvil on a board over it and a
    short smoking stack."""
    p = Painter(30, 38, (14, 29))
    p.shadow(12, 4)
    wall = (92, 84, 82)
    top = p.hall(-1, 0, 13, 9, 7, wall, None)
    left, front, right, _ = top
    _patch(p, left, front, (0.52, 0.86), (2.5, 7), (238, 132, 40))
    _patch(p, left, front, (0.6, 0.78), (4.5, 7), (255, 208, 112))
    _patch(p, left, front, (0.03, 0.47), (0.5, 6), (214, 202, 176))
    cx, cy = _on_wall(left, front, 0.25, 3.5)
    p.polygon(
        [
            (cx + dx, cy + dy)
            for dx, dy in (
                (-3, -1),
                (2, -1),
                (2, 0),
                (1, 0),
                (1, 1),
                (2, 2),
                (-2, 2),
                (-1, 1),
                (-1, 0),
                (-2, 0),
            )
        ],
        (40, 40, 46),
    )
    _windows(p, front, right, (0.3, 0.7), (3,), 1, (240, 150, 60))
    p.gable(top, 4, (60, 60, 68), wall)
    p.rect(-1, -17, 1, -10, (74, 58, 52))
    p.point(0, -17, (255, 170, 60))
    _smoke(p, 1, -20, (96, 94, 98))
    return p.sprite()


def winery() -> Sprite:
    """A low cream winery under red tiles, with a cellar door dug into a grassy bank, barrels
    outside and vines trained on a trellis."""
    p = Painter(34, 30, (17, 19))
    p.shadow(13, 3.5)
    wall = (238, 224, 192)
    top = p.hall(0, -2, 12, 7, 5, wall, None)
    left, front, *_ = top
    p.gable(top, 3, (178, 86, 58), wall)
    _windows(p, left, front, (0.2, 0.45, 0.8), (2,), 1)
    for t in (0.2, 0.45, 0.8):
        p.point(*_on_wall(left, front, t - 0.07, 2), (90, 120, 70))
    _patch(p, left, front, (0.56, 0.68), (2, 5), (110, 74, 50))
    vine, grape, post = (78, 130, 50), (112, 50, 124), (110, 84, 58)
    for start in ((-16, 1), (-13, 5)):
        ends = [(start[0] + 3 * k, start[1] + 1.5 * k) for k in range(4)]
        for x, y in ends:
            p.line([(x, y), (x, y - 5)], post)
        p.line([(x, y - 4) for x, y in ends], vine)
        p.line([(x, y - 5) for x, y in ends], shade(vine, 1.25))
        for x, y in ends[:-1]:
            p.point(x + 1.5, y - 2.5, grape)
    cellar = p.box(10, 3, 4, 3, (176, 168, 150), (104, 140, 70))
    _patch(p, cellar[0], cellar[1], (0.3, 0.75), (1, 3), (42, 34, 30))
    p.point(*_on_wall(cellar[0], cellar[1], 0.5, 0.5), shade(STONE, 1.3))
    for x, y in ((3, 6), (6, 7.5)):
        _barrel(p, x, y)
    return p.sprite()


def chemical_works() -> Sprite:
    """A ball tank on legs, pale tanks with green tops and a pipe rack beside a small control
    house, with green fumes rising from a vent stack."""
    p = Painter(32, 47, (15, 37))
    p.shadow(13, 4)
    body, green = (208, 214, 198), (104, 176, 84)
    p.rect(10, -26, 11, -2, METAL)
    for i, (x, y, r) in enumerate(((11, -28, 1.5), (12, -31, 2), (14, -34, 2.5))):
        p.blob(x - r, y - r, x + r, y + r, (178, 222, 120) if i % 2 else (146, 204, 96))
    _turned(p, 4, -4, [(0, 3), (13, 3)], body, cap=green)
    for x in (-11, -9, -6):
        p.line([(x, 0), (x, -6)], shade(METAL, 0.7))
    p.ellipse(-13, -15, -4, -6, body)
    p.blob(-11, -13, -8, -10, shade(body, 1.15))
    p.line([(-13, -10), (-4, -10)], green)
    p.line([(-4, -9), (2, -9)], shade(METAL, 0.8))
    p.line([(-4, -7), (8, -7)], green)
    _turned(p, 8, 1, [(0, 2.5), (9, 2.5)], body, cap=green)
    left, front, *_ = p.box(-3, 4, 5, 4, (196, 200, 186), (96, 116, 84))
    _windows(p, left, front, (0.3, 0.7), (1.5,), 1, (110, 200, 120))
    return p.sprite()


def electronics_plant() -> Sprite:
    """A modern box of white panels and bands of cyan glass, with a dish and an aerial on its flat
    roof."""
    p = Painter(32, 38, (15, 28))
    p.shadow(13, 4)
    wall, glass, accent = (226, 232, 236), (84, 176, 206), (80, 216, 222)
    left, front, right, _ = p.hall(0, 0, 15, 9, 8, wall, (172, 180, 190))
    for d0, d1 in ((1.5, 3), (4.5, 6.5)):
        _patch(p, left, front, (0.04, 0.96), (d0, d1), glass)
        _patch(p, front, right, (0.06, 0.94), (d0, d1), shade(glass, 0.72))
    p.line([_on_wall(left, front, 0.04, 4.5), _on_wall(left, front, 0.96, 4.5)], accent)
    _patch(p, left, front, (0.42, 0.58), (6.5, 8), (52, 70, 84))
    p.line([(-4, -8), (-4, -11)], METAL)
    p.ellipse(-8, -16, -1, -11, (236, 238, 240))
    p.blob(-6, -15, -3, -13, (176, 186, 196))
    p.point(-4, -14, accent)
    p.line([(5, -7), (5, -18)], METAL)
    p.line([(3, -15), (7, -15)], METAL)
    p.point(5, -19, (230, 60, 60))
    p.rect(7, -11, 9, -9, (190, 198, 206))
    return p.sprite()


def _pit(p: Painter, x: float, y: float, half: float, ground: RGB, floor: RGB) -> None:
    """A hole dug into the ground at (x, y), ringed with spoil, showing its far walls and floor."""
    w, h = half, half / 2
    p.polygon(
        [(x - w - 3, y), (x, y + h + 1.5), (x + w + 3, y), (x, y - h - 1.5)], shade(ground, 1.12)
    )
    low = (x - w + 3, y + 1.5), (x, y - h + 3), (x + w - 3, y + 1.5)
    p.polygon([(x - w, y), (x, y - h), low[1], low[0]], shade(ground, 0.72))
    p.polygon([(x, y - h), (x + w, y), low[2], low[1]], ground)
    p.polygon([low[0], low[1], low[2], (x, y + h)], floor)


def clay_pit() -> Sprite:
    """An open pit dug into orange clay, with a heap of it, a tool shed and a loaded wheelbarrow."""
    p = Painter(32, 26, (16, 16))
    clay = (190, 108, 64)
    _heap(p, -9, -3, 5, 6, shade(clay, 1.05))
    _pit(p, -2, 2, 8, clay, shade(clay, 0.55))
    top = p.box(10, -1, 4, 4, (164, 150, 128), None)
    p.gable(top, 3, (96, 84, 76), (164, 150, 128))
    _patch(p, top[0], top[1], (0.5, 0.8), (1, 4), (70, 56, 46))
    barrow = (92, 118, 150)
    x, y = 6, 8
    p.line([(x - 3, y - 2), (x, y - 3)], TRUNK)
    p.polygon(
        [(x - 1, y - 4), (x + 5, y - 4), (x + 4, y - 1), (x, y - 1)], barrow, shade(barrow, 0.55)
    )
    p.line([(x, y - 4), (x + 4, y - 4)], shade(clay, 1.1))
    p.point(x + 2, y - 5, clay)
    p.rect(x + 4, y - 1, x + 5, y, (40, 38, 40))
    return p.sprite()


def limestone_quarry() -> Sprite:
    """A pale limestone face cut in steps into a grassy hill, with sawn blocks waiting below."""
    p = Painter(34, 34, (17, 24))
    p.shadow(13, 4)
    stone = (222, 218, 202)
    terrace = shade(stone, 0.9)
    for (x, y, h), top in (
        ((3, -6, 11), (118, 148, 80)),
        ((-1, -4, 7), terrace),
        ((-5, -2, 3), terrace),
    ):
        left, front, *_ = p.hall(x, y, 16, 5, h, stone, top)
        for drop in range(3, h, 3):
            p.line(
                [_on_wall(left, front, 0.05, drop), _on_wall(left, front, 0.95, drop)],
                shade(stone, 0.8),
            )
    for x, y, h in ((6, 5, 3), (10, 3, 3), (10, 0, 2.5)):
        p.box(x, y, 2.5, h, stone, shade(stone, 1.08))
    p.line([(8, -13), (8, -23)], TRUNK)
    p.line([(8, -22), (0, -17)], TRUNK)
    p.line([(0, -17), (0, -12)], (60, 56, 54))
    p.rect(-1, -12, 1, -10, stone)
    return p.sprite()


def sand_pit() -> Sprite:
    """Pale yellow heaps of sand beside a pit with water in its bottom, fed by a belt conveyor."""
    p = Painter(34, 28, (17, 18))
    sand = (230, 206, 140)
    _pit(p, -5, 2, 8, sand, (92, 148, 196))
    p.line([(-7, 3), (-3, 3)], (150, 196, 226))
    _heap(p, 9, -1, 5, 8, sand)
    _heap(p, 6, 6, 4, 5, shade(sand, 1.04))
    belt = (70, 70, 78)
    p.line([(-1, -1), (7, -9)], belt)
    p.line([(-1, 0), (7, -8)], shade(belt, 1.4))
    for x, y in ((1, -2), (4, -5)):
        p.line([(x, y), (x, y + 4)], shade(belt, 1.2))
        p.point(x + 1, y - 1, sand)
    return p.sprite()


def potato_farm() -> Sprite:
    """A low fieldstone cottage under thatch, with sacks of potatoes stood by its door."""
    p = Painter(26, 24, (12, 16))
    p.shadow(10, 3)
    wall = (172, 166, 150)
    top = p.hall(-2, -1, 11, 6, 4, wall, None)
    left, front, *_ = top
    for t, drop in ((0.15, 1), (0.4, 3), (0.85, 2), (0.6, 0.5)):
        p.point(*_on_wall(left, front, t, drop), shade(wall, 0.75))
    _windows(p, left, front, (0.25,), (2,), 1)
    _patch(p, left, front, (0.55, 0.72), (1, 4), (80, 62, 48))
    p.gable(top, 4, THATCH, wall)
    sack = (192, 162, 112)
    for x, y in ((8, 0), (5, 3), (9, 3.5)):
        p.ellipse(x - 1.5, y - 4, x + 1.5, y, sack)
        p.point(x, y - 4, shade(sack, 0.5))
    for x, y in ((3, 5), (4, 6), (6, 6)):
        p.point(x, y, (150, 104, 58))
    return p.sprite()


def _cow(p: Painter, x: float, y: float, hide: RGB, patch: RGB) -> None:
    p.line([(x - 2, y - 1), (x - 2, y)], shade(hide, 0.5))
    p.line([(x + 2, y - 1), (x + 2, y)], shade(hide, 0.5))
    p.ellipse(x - 3, y - 4, x + 3, y - 1, hide)
    p.point(x + 1, y - 3, patch)
    p.rect(x - 5, y - 5, x - 3, y - 3, hide)
    p.point(x - 5, y - 6, (232, 222, 196))


def _paddock(x: float, y: float, half: float) -> tuple[Point, Point, Point, Point]:
    """The left, back, right and front corners of a fenced patch of ground centred at (x, y)."""
    return (x - half, y), (x, y - half / 2), (x + half, y), (x, y + half / 2)


def _fence(p: Painter, a: Point, b: Point) -> None:
    """A post-and-rail fence along the ground from `a` to `b`."""
    for k in range(4):
        px, py = _on_wall(a, b, k / 3, 0)
        p.line([(px, py), (px, py - 3)], (122, 90, 60))
    p.line([_on_wall(a, b, 0, -2), _on_wall(a, b, 1, -2)], (206, 176, 126))


def cattle_ranch() -> Sprite:
    """A wooden cattle barn beside a fenced paddock with brown and black-and-white cows in it."""
    p = Painter(34, 26, (16, 18))
    p.shadow(12, 3.5)
    wood = (136, 86, 56)
    top = p.box(-8, -2, 6, 6, wood, None)
    left, front, *_ = top
    _patch(p, left, front, (0.45, 0.8), (1.5, 6), (236, 230, 214))
    _patch(p, left, front, (0.52, 0.73), (2.5, 6), shade(wood, 0.7))
    p.gable(top, 4, (86, 74, 70), wood)
    left, back, right, front = _paddock(6, 2, 9)
    _fence(p, left, back)
    _fence(p, back, right)
    _cow(p, 6, 0, (138, 84, 50), (96, 58, 36))
    _cow(p, 10, 4, (240, 238, 232), (40, 38, 42))
    _cow(p, 3, 4, (40, 38, 42), (240, 238, 232))
    _fence(p, left, front)
    _fence(p, front, right)
    return p.sprite()


def flax_field() -> Sprite:
    """A small whitewashed cottage with sheaves of flax stood up in stooks to dry beside it."""
    p = Painter(28, 24, (13, 16))
    p.shadow(10, 3)
    top = p.box(-6, -1, 5, 4, WHITEWASH, None)
    _windows(p, top[0], top[1], (0.3,), (2,), 1)
    _patch(p, top[0], top[1], (0.6, 0.8), (1, 4), (92, 70, 52))
    p.gable(top, 3, (128, 96, 70), WHITEWASH)
    straw = (204, 188, 120)
    for row in range(2):
        for col in range(3):
            x, y = 1 + 4 * col + 2 * row, -1 + 4 * row - col
            p.polygon([(x - 2, y), (x, y - 6), (x, y + 1)], straw, shade(straw, 0.6))
            p.polygon([(x, y - 6), (x + 2, y), (x, y + 1)], shade(straw, 0.78), shade(straw, 0.55))
            p.line([(x - 1, y - 3), (x + 1, y - 3)], (128, 104, 60))
    for x, y in ((-1, 5), (2, 6), (11, 6)):
        p.point(x, y, (110, 140, 230))
    return p.sprite()


def hop_garden() -> Sprite:
    """Tall poles strung with green hop bines, and a round oast house with a white cowl."""
    p = Painter(34, 40, (17, 30))
    pole, bine = (122, 96, 66), (88, 152, 58)
    for r in range(2):
        poles = [(-11 + 4 * k - 4 * r, -4 + 2 * k + 2 * r) for k in range(3)]
        tops = [(x, y - 16) for x, y in poles]
        for (x, y), (tx, ty) in zip(poles, tops, strict=True):
            p.line([(x, y), (tx, ty)], pole)
            p.line([(x - 1, y - 2), (x - 1, y - 12)], bine)
            p.line([(x + 1, y - 1), (x + 1, y - 14)], shade(bine, 0.85))
            for k in range(4):
                p.point(x + (k % 2) * 2 - 1, y - 3 - 3 * k, shade(bine, 1.4))
        p.line(tops, (70, 64, 58))
        for (x0, y0), (x1, y1) in itertools.pairwise(tops):
            p.line([(x0 + 1, y0 + 2), ((x0 + x1) / 2, (y0 + y1) / 2 + 3), (x1 - 1, y1 + 2)], bine)
    oast = (176, 92, 64)
    _turned(p, 10, 0, [(0, 4), (9, 4)], oast)
    _turned(p, 10, -9, [(0, 4.5), (8, 0.5)], (110, 76, 64))
    cowl = (244, 242, 236)
    p.polygon([(9, -17), (9, -20), (11, -22), (12, -19), (11, -17)], cowl, shade(cowl, 0.6))
    p.line([(12, -21), (14, -22)], cowl)
    stow = p.box(4, 4, 4, 4, (228, 222, 206), None)
    p.gable(stow, 3, (112, 82, 66), (228, 222, 206))
    return p.sprite()


def machine_works() -> Sprite:
    """A big brick engine hall with tall arched windows and a great gear on its end wall, beside a
    yard with a gantry crane lifting a casting."""
    p = Painter(38, 42, (18, 32))
    p.shadow(16, 5)
    top = p.hall(-2, -2, 16, 9, 10, BRICK, None)
    left, front, right, _ = top
    lintel = shade(BRICK, 1.3)
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2.5,), 5, arch=lintel)
    _patch(p, left, front, (0.8, 0.94), (5, 10), (58, 48, 44))
    p.gable(top, 5, (82, 84, 96), BRICK)
    _gear(p, *_on_wall(front, right, 0.5, 5), 2.5, (222, 180, 74))
    crane = (214, 168, 40)
    feet, height = ((11, 6), (17, 3)), 15
    for x, y in feet:
        p.line([(x, y), (x, y - height)], crane)
        p.line([(x - 1, y), (x, y - 3)], shade(crane, 0.7))
    p.line([(feet[0][0] - 1, feet[0][1] - height), (feet[1][0] + 1, feet[1][1] - height)], crane)
    p.line([(14, -11), (14, -4)], (70, 64, 58))
    p.rect(12, -4, 16, -1, (88, 90, 98))
    return p.sprite()


def brickworks() -> Sprite:
    """A round Hoffmann kiln ringed with arched firing doors around its tall central chimney, and
    stacks of new red bricks."""
    p = Painter(34, 54, (16, 44))
    p.shadow(14, 5)
    kiln = (152, 98, 76)
    x, y, r = -2, -1, 10
    _turned(p, x, y, [(0, r), (6, r)], kiln)
    for a in (0.35, 0.95, 1.55, 2.15, 2.75):
        ax, ay = x + r * math.cos(a), y + r / 2 * math.sin(a)
        p.rect(ax - 1, ay - 4, ax + 1, ay, (52, 36, 32))
    _turned(p, x, y - 6, [(0, r + 1), (3, r - 4)], (96, 86, 86), cap=(118, 106, 104))
    _chimney(p, x, y - 8, y - 35, shade(BRICK, 0.9))
    _smoke(p, x + 1, y - 38, puffs=2)
    for bx, by in ((13, 2), (8.5, 5.5)):
        stack = p.hall(bx, by, 4, 3, 3, BRICK, shade(BRICK, 1.25))
        for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
            p.line([_on_wall(a, b, 0, 1.5), _on_wall(a, b, 1, 1.5)], shade(BRICK, 1.3))
    return p.sprite()


def cement_works() -> Sprite:
    """Dusty grey silos and a preheater tower, with a long rotary kiln tube on piers before them."""
    p = Painter(40, 50, (20, 40))
    p.shadow(16, 5)
    grey, dust = (190, 188, 182), (214, 212, 206)
    tower = p.box(9, -6, 3.5, 22, shade(grey, 0.9), shade(grey, 0.75))
    for drop in (4, 9, 14, 19):
        p.line(
            [_on_wall(tower[0], tower[1], 0, drop), _on_wall(tower[0], tower[1], 1, drop)],
            shade(grey, 0.6),
        )
    p.rect(8, -32, 10, -28, shade(grey, 0.7))
    _smoke(p, 9, -35, dust, 2)
    for sx, sy in ((-6, -7), (-10, -5), (-14, -3)):
        _turned(p, sx, sy, [(0, 2.5), (16, 2.5)], grey, cap=shade(grey, 1.1))
    tube = (134, 128, 120)
    a, b = (-12, 5), (6, -4)
    for t in (0.15, 0.5, 0.85):
        px, py = _on_wall(a, b, t, 0)
        p.rect(px - 1, py - 2 - 2 * t, px + 1, py + 1, shade(grey, 0.8))
    lift = 4
    p.polygon(
        [
            (a[0], a[1] - lift - 3),
            (b[0], b[1] - lift - 5),
            (b[0], b[1] - lift),
            (a[0], a[1] - lift),
        ],
        tube,
        shade(tube, 0.55),
    )
    p.line([(a[0], a[1] - lift - 3), (b[0], b[1] - lift - 5)], shade(tube, 1.3))
    for t in (0.3, 0.6):
        px, py = _on_wall(a, b, t, -lift)
        p.line([(px, py - 3 - t), (px, py)], shade(tube, 0.6))
    p.ellipse(a[0] - 2, a[1] - lift - 3.5, a[0] + 1, a[1] - lift + 0.5, shade(tube, 0.8))
    for x, y, r in ((-14, 4, 2), (-10, 7, 1.5), (-16, 1, 1.5)):
        p.blob(x - r, y - r, x + r, y + r, dust)
    return p.sprite()


def glassworks() -> Sprite:
    """A tall brick glass cone with its furnace mouth glowing at the foot, beside sheets of new
    glass leant against a shed."""
    p = Painter(32, 50, (15, 40))
    p.shadow(13, 4)
    cone = (162, 88, 64)
    x = -3
    _turned(p, x, 0, [(0, 9), (7, 7.5), (20, 4), (28, 3)], cone, cap=(54, 40, 36))
    _smoke(p, x + 1, -31)
    p.ellipse(x - 3, -6, x + 1, 0, (240, 132, 40))
    p.rect(x - 3, -3, x + 1, 2, (240, 132, 40))
    p.blob(x - 2, -3, x, 1, (255, 214, 120))
    for gx, gy in ((x - 4, 4), (x - 1, 4), (x - 3, 5)):
        p.point(gx, gy, (236, 150, 60))
    top = p.box(10, 2, 4, 4, (196, 180, 156), None)
    p.gable(top, 2, (96, 84, 80), (196, 180, 156))
    pane = (176, 222, 234)
    for k in range(3):
        px, py = 5 + 2 * k, 7 - k
        p.polygon(
            [(px, py), (px + 1, py - 5), (px + 3, py - 6), (px + 2, py - 1)], pane, shade(pane, 0.6)
        )
        p.point(px + 1, py - 3, (246, 252, 255))
    return p.sprite()


def brewery() -> Sprite:
    """A brick brewhouse with a copper-domed tower, barrels at its door and a hop cone for a sign."""
    p = Painter(32, 44, (15, 34))
    p.shadow(13, 4)
    top = p.hall(-4, -1, 12, 8, 8, BRICK, None)
    left, front, *_ = top
    _windows(p, left, front, (0.45, 0.65, 0.85), (2,), 2, arch=shade(BRICK, 1.3))
    _patch(p, left, front, (0.04, 0.34), (1, 5), (222, 210, 180))
    hop = (110, 170, 70)
    hx, hy = _on_wall(left, front, 0.19, 3)
    p.polygon(
        [(hx, hy - 1.5), (hx + 1.5, hy), (hx, hy + 1.5), (hx - 1.5, hy)], hop, shade(hop, 0.6)
    )
    p.gable(top, 4, (84, 76, 80), BRICK)
    tower = shade(BRICK, 1.08)
    p.box(5, 3, 4, 15, tower, None)
    p.line([(1, -12), (5, -10), (9, -12)], shade(tower, 1.25))
    _windows(p, (1, -12), (5, -10), (0.5,), (3, 8), 2, arch=shade(BRICK, 1.35))
    _turned(p, 5, -12, [(0, 4.5), (2, 4.2), (4, 3), (5.5, 1.5), (6, 0)], COPPER)
    p.blob(2, -16, 4, -14, shade(COPPER, 1.3))
    p.line([(5, -18), (5, -20)], (90, 80, 70))
    for x, y in ((-7, 6), (-3, 8)):
        _barrel(p, x, y)
    return p.sprite()


def distillery() -> Sprite:
    """A whitewashed still house with a copper pot still at its side and a tall thin stack."""
    p = Painter(32, 50, (15, 41))
    p.shadow(13, 4)
    p.rect(3, -33, 4, -4, (150, 80, 60))
    p.line([(3, -31), (4, -31)], (230, 228, 222))
    _smoke(p, 4, -36, (190, 190, 194), 2)
    top = p.hall(-4, -1, 12, 7, 6, WHITEWASH, None)
    left, front, *_ = top
    _windows(p, left, front, (0.15, 0.35, 0.8), (2,), 2, (44, 46, 52))
    _patch(p, left, front, (0.5, 0.66), (2.5, 6), (60, 48, 42))
    p.gable(top, 4, (68, 72, 82), WHITEWASH)
    _turned(p, 8, 5, [(0, 3), (2, 4), (4, 3.5), (6, 1.5), (7, 1)], COPPER)
    p.blob(5, -1, 7, 2, shade(COPPER, 1.3))
    p.line([(8, -2), (8, -4), (12, -2)], shade(COPPER, 0.8))
    _barrel(p, 12, 4)
    return p.sprite()


def tannery() -> Sprite:
    """Low tanning sheds, with hides stretched to dry on wooden frames and a pit of tan liquor."""
    p = Painter(34, 30, (17, 18))
    p.shadow(12, 3.5)
    wall = (184, 160, 124)
    for x, y in ((-2, -6), (-7, -2)):
        top = p.hall(x, y, 10, 5, 4, wall, None)
        _patch(p, top[0], top[1], (0.3, 0.7), (1, 4), (70, 56, 44))
        p.gable(top, 3, (96, 82, 74), wall)
    p.ellipse(-6, 4, 0, 7, (138, 116, 86))
    p.blob(-5, 4.5, -1, 6.5, (96, 66, 40))
    frame = (104, 76, 50)
    for k, hide in enumerate(((198, 150, 100), (162, 108, 68), (214, 180, 130))):
        x, y = 2 + 4 * k, 6 - 2 * k
        p.line([(x, y), (x, y - 7)], frame)
        p.line([(x + 4, y - 1), (x + 4, y - 8)], frame)
        p.line([(x, y - 7), (x + 4, y - 8)], frame)
        p.polygon(
            [(x + 1, y - 6), (x + 3, y - 7), (x + 3.5, y - 2), (x + 2, y - 1.5), (x + 0.5, y - 1)],
            hide,
            shade(hide, 0.65),
        )
    return p.sprite()


def shoe_factory() -> Sprite:
    """A narrow workshop of four storeys with a big boot hung out from its wall as a sign."""
    p = Painter(24, 38, (12, 29))
    p.shadow(8, 2.5)
    wall = (204, 170, 116)
    s, floors = 6, 4
    top = p.box(1, 0, s, 4 * floors, wall, None)
    left, front, right, _ = top
    _windows(p, left, front, (0.3, 0.7), tuple(1.5 + 4 * f for f in range(floors - 1)), 2)
    _windows(
        p,
        front,
        right,
        (0.3, 0.7),
        tuple(1.5 + 4 * f for f in range(floors)),
        2,
        shade(WINDOW, 0.75),
    )
    _patch(p, left, front, (0.55, 0.8), (12.5, 16), (84, 60, 46))
    p.gable(top, 4, (124, 66, 52), wall)
    p.line([(-5, -12), (-8, -12)], (60, 56, 54))
    boot = (140, 54, 40)
    p.polygon(
        [(-9, -11), (-5, -11), (-5, -3), (-11, -3), (-11, -5), (-10, -6), (-9, -6)],
        boot,
        shade(boot, 0.5),
    )
    p.line([(-8, -10), (-8, -6)], shade(boot, 1.4))
    p.line([(-11, -3), (-5, -3)], (40, 32, 30))
    return p.sprite()


def plastics_works() -> Sprite:
    """White tanks with red and blue tops and a white hall under a blue roof, tied together by
    coloured pipes, under a red-banded stack."""
    p = Painter(32, 40, (15, 31))
    p.shadow(13, 4)
    white, red, blue = (232, 234, 236), (214, 58, 54), (64, 108, 196)
    p.rect(-2, -29, 0, -4, white)
    for y in (-27, -24):
        p.line([(-2, y), (0, y)], red)
    _turned(p, 6, -3, [(0, 3), (12, 3)], white, cap=red)
    _turned(p, 11, 2, [(0, 2.5), (8, 2.5)], white, cap=blue)
    left, front, *_ = p.box(-5, 2, 7, 5, white, blue)
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], red)
    _windows(p, left, front, (0.25, 0.5, 0.75), (2.5,), 1, shade(blue, 0.8))
    p.line([(1, -5), (3, -6), (3, -10)], blue)
    p.line([(2, -2), (8, -2), (9, -4)], red)
    return p.sprite()


def appliance_factory() -> Sprite:
    """A long modern factory with a big roller door, and its fridges and washing machines stacked
    outside in white."""
    p = Painter(36, 35, (17, 23))
    p.shadow(15, 4)
    wall, stripe = (216, 220, 224), (64, 116, 196)
    left, front, right, _ = p.hall(1, -3, 18, 8, 7, wall, (150, 160, 174))
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], stripe)
    p.line([_on_wall(front, right, 0, 1), _on_wall(front, right, 1, 1)], shade(stripe, 0.8))
    door = (170, 174, 180)
    _patch(p, left, front, (0.55, 0.88), (2.5, 7), door)
    for drop in (3.5, 5, 6.5):
        p.line(
            [_on_wall(left, front, 0.55, drop), _on_wall(left, front, 0.88, drop)], shade(door, 0.7)
        )
    _windows(p, left, front, (0.1, 0.2, 0.3, 0.4), (3,), 1, (110, 160, 200))
    white = (244, 246, 248)
    for x, y, h, washer in (
        (15, 1, 6, False),
        (12, 4, 4, True),
        (12, 0, 4, True),
        (9, 7, 4, True),
    ):
        crate = p.box(x, y, 2.5, h, white, shade(white, 0.9))
        if washer:
            cx, cy = _on_wall(crate[0], crate[1], 0.5, 2.5)
            p.blob(cx - 1, cy - 1, cx + 1, cy + 1, (64, 88, 118))
        else:
            p.line(
                [_on_wall(crate[0], crate[1], 0, 2), _on_wall(crate[0], crate[1], 1, 2)],
                shade(white, 0.55),
            )
    return p.sprite()


def paper_mill() -> Sprite:
    """A riverside paper mill turning a waterwheel in its stream, with a tall stack and rolls of
    paper stacked in the yard."""
    p = Painter(38, 51, (19, 41))
    water = (84, 140, 196)
    p.polygon([(-17, -3), (-1, 5), (-4, 7.5), (-17, 1)], water)
    p.line([(-14, -0.5), (-10, 1.5)], (150, 196, 226))
    p.shadow(11, 4)
    _chimney(p, 8, -4, -30)
    _smoke(p, 9, -33)
    wall = (192, 168, 136)
    top = p.hall(-1, -3, 12, 8, 9, wall, None)
    left, front, *_ = top
    _windows(p, left, front, (0.55, 0.75, 0.92), (2, 5.5), 2)
    p.gable(top, 4, (100, 86, 80), wall)
    cx, cy = _on_wall(left, front, 0.25, 7)
    wheel = (98, 72, 50)
    p.ellipse(cx - 3, cy - 5, cx + 3, cy + 4, wheel)
    p.ellipse(cx - 2, cy - 4, cx + 2, cy + 3, shade(wheel, 1.3))
    p.line([(cx, cy - 4), (cx, cy + 3)], wheel)
    p.line([(cx - 2, cy - 0.5), (cx + 2, cy - 0.5)], wheel)
    p.polygon([(cx - 4, cy + 2), (cx + 4, cy + 5), (cx + 4, cy + 6), (cx - 4, cy + 3)], water)
    roll = (240, 240, 234)
    for x, y in ((6, 6), (9, 4.5), (12, 3), (7.5, 2.5), (10.5, 1), (9, -1)):
        p.ellipse(x - 1.5, y - 3, x + 1.5, y, roll)
        p.point(x, y - 1.5, shade(roll, 0.55))
    return p.sprite()


def printing_works() -> Sprite:
    """A town printing house with tall windows and a newspaper held up on its roof as a sign."""
    p = Painter(26, 36, (12, 27))
    p.shadow(9, 3)
    wall = (220, 204, 172)
    left, front, right, _ = p.box(0, 0, 8, 12, wall, (110, 106, 116))
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], shade(wall, 1.12))
    _windows(p, left, front, (0.2, 0.45, 0.7, 0.9), (2.5,), 4)
    _windows(p, left, front, (0.2, 0.45, 0.9), (8,), 3)
    _patch(p, left, front, (0.58, 0.8), (8, 12), (70, 56, 48))
    _windows(p, front, right, (0.25, 0.5, 0.75), (2.5, 8), 3, shade(WINDOW, 0.75))
    for x in (-3, 3):
        p.line([(x, -14), (x, -17)], (70, 66, 62))
    page = (242, 240, 230)
    p.rect(-7, -25, 7, -17, page)
    p.line([(-5, -23), (5, -23)], (40, 40, 48))
    p.rect(-5, -21, -2, -19, (150, 150, 160))
    for y in (-21, -19):
        p.line([(0, y), (5, y)], (150, 150, 160))
    return p.sprite()


def sugar_refinery() -> Sprite:
    """A tall brick refinery with a big square chimney, and white sacks and sugar loaves in blue
    paper at its foot."""
    p = Painter(34, 61, (15, 51))
    p.shadow(13, 4, reach=5)
    wall = (184, 110, 74)
    stack = p.box(8, -3, 2, 34, shade(wall, 0.9), (60, 52, 50))
    for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
        p.line([_on_wall(a, b, 0, 2), _on_wall(a, b, 1, 2)], shade(wall, 1.2))
    _smoke(p, 9, -40, puffs=4)
    left, front, right, _ = p.box(-3, 0, 8, 18, wall, (92, 86, 94))
    _windows(p, left, front, (0.2, 0.45, 0.7, 0.92), (2, 5.5, 9, 12.5), 2)
    _windows(p, front, right, (0.3, 0.7), (2, 5.5, 9, 12.5, 16), 2, shade(WINDOW, 0.75))
    _patch(p, left, front, (0.55, 0.75), (15, 18), (70, 50, 40))
    white, blue = (240, 240, 236), (70, 100, 180)
    for x, y in ((9, 4), (11, 2.5), (10, 1)):
        p.ellipse(x - 2, y - 3, x + 2, y, white)
    for x, y in ((3, 7), (6, 7.5)):
        p.polygon([(x - 1.5, y), (x, y - 6), (x + 1.5, y)], white, shade(white, 0.7))
        p.polygon([(x - 1.5, y), (x - 1, y - 2), (x + 1, y - 2), (x + 1.5, y)], blue)
    return p.sprite()


def porcelain_works() -> Sprite:
    """Two small brick bottle ovens beside a workshop, with blue-and-white ware set out on shelves."""
    p = Painter(34, 36, (17, 27))
    p.shadow(13, 4)
    oven = (172, 116, 86)
    bottle = [(0, 4.5), (4, 5), (8, 4), (11, 2), (13, 1.8)]
    for x, y in ((-4, -6), (-10, -3)):
        _turned(p, x, y, bottle, oven, cap=(60, 46, 42))
    wall = (230, 218, 192)
    top = p.hall(5, -2, 10, 6, 5, wall, None)
    _windows(p, top[0], top[1], (0.25, 0.5, 0.75), (2,), 1)
    p.gable(top, 3, (122, 88, 70), wall)
    frame, china, blue = (120, 88, 58), (246, 246, 250), (54, 86, 176)
    for x in (-8, 2):
        p.line([(x, 6 + (x + 8) * 0.1), (x, -1)], frame)
    for y in (5, 2):
        p.line([(-8, y), (2, y + 1)], frame)
        for k in range(5):
            x = -7 + 2 * k
            p.line([(x, y - 1 + k * 0.2), (x, y - 2 + k * 0.2)], china if k % 2 else blue)
    return p.sprite()


def tavern() -> Sprite:
    """A half-timbered inn with a sign hung from a bracket and a beer barrel by the door."""
    p = Painter(25, 26, (12, 20))
    p.shadow(9, 3)
    wall, beam = (234, 222, 192), (92, 64, 44)
    top = p.box(0, 0, 7, 7, wall, None)
    left, front, right, _ = top
    for a, b in ((left, front), (front, right)):
        for t in (0, 0.5, 1):
            p.line([_on_wall(a, b, t, 0), _on_wall(a, b, t, 7)], beam)
        p.line([_on_wall(a, b, 0, 3.5), _on_wall(a, b, 1, 3.5)], beam)
        p.line([_on_wall(a, b, 0, 0.5), _on_wall(a, b, 0.5, 3.5)], beam)
    _windows(p, left, front, (0.25,), (5,), 1)
    _patch(p, left, front, (0.62, 0.84), (4, 7), (70, 48, 34))
    p.gable(top, 5, (126, 74, 54), wall)
    p.line([(-7, -5), (-10, -5)], (52, 48, 46))
    p.rect(-11, -4, -8, -1, (170, 46, 42))
    p.point(-10, -3, (240, 200, 90))
    _barrel(p, -5, 4)
    return p.sprite()


def department_store() -> Sprite:
    """A wide, ornate store of three storeys with great display windows, a corner dome and a flag."""
    p = Painter(28, 42, (13, 32))
    p.shadow(11, 3.5)
    wall = (234, 216, 184)
    left, front, right, _ = p.box(0, 0, 10, 13, wall, (128, 118, 124))
    display = (150, 196, 222)
    _patch(p, left, front, (0.05, 0.92), (9, 12), display)
    _patch(p, front, right, (0.08, 0.95), (9, 12), shade(display, 0.75))
    for a, b in ((left, front), (front, right)):
        p.line([_on_wall(a, b, 0, 1), _on_wall(a, b, 1, 1)], shade(wall, 1.12))
        p.line([_on_wall(a, b, 0, 8), _on_wall(a, b, 1, 8)], shade(wall, 0.7))
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2.5, 5.5), 2)
    _windows(p, front, right, (0.34, 0.52, 0.7, 0.88), (2.5, 5.5), 2, shade(WINDOW, 0.75))
    _turned(p, 0, -8, [(0, 3), (3, 3)], wall)
    _turned(p, 0, -11, [(0, 3.4), (2, 2.8), (3.5, 1.6), (4.5, 0)], (98, 150, 132))
    _flag(p, 0, -15, 8)
    return p.sprite()


def hotel() -> Sprite:
    """A tall, elegant hotel with rows of windows under a mansard roof, a canopy over its door and
    its name lit up down a sign at the corner."""
    p = Painter(27, 46, (12, 37))
    p.shadow(8, 3, reach=6)
    wall, slate = (226, 198, 176), (84, 88, 104)
    s, height = 7, 22
    left, front, right, _ = p.box(0, 0, s, height, wall, None)
    _windows(p, left, front, (0.2, 0.45, 0.7, 0.92), tuple(2 + 3 * f for f in range(6)), 2)
    _windows(
        p,
        front,
        right,
        (0.25, 0.5, 0.75),
        tuple(2 + 3 * f for f in range(7)),
        2,
        shade(WINDOW, 0.75),
    )
    p.box(0, -height, s - 1, 3, slate, shade(slate, 0.8))
    for x in (-3, 0, 3):
        p.point(x, -height - 1 + abs(x) * 0.5, (236, 214, 140))
    _patch(p, left, front, (0.42, 0.62), (19, 22), (70, 52, 44))
    awning = (170, 44, 48)
    a, b = _on_wall(left, front, 0.36, 18), _on_wall(left, front, 0.68, 18)
    p.polygon([a, b, (b[0] - 3, b[1] + 2), (a[0] - 3, a[1] + 2)], awning, shade(awning, 0.6))
    p.line([(a[0] - 3, a[1] + 2), (a[0] - 3, a[1] + 5)], (60, 56, 54))
    sign = (40, 48, 92)
    p.rect(-9, -19, -7, -8, sign)
    for y in (-17, -15, -13, -11):
        p.point(-8, y, (250, 214, 96))
    return p.sprite()


def meat_factory() -> Sprite:
    """A brick slaughterhouse with a ham hung out as its sign, a ramp up from a cattle pen, and a
    white cold store with a refrigeration unit and vent on its roof."""
    p = Painter(36, 36, (17, 25))
    p.shadow(15, 4.5)
    top = p.hall(0, -3, 14, 8, 8, BRICK, None)
    left, front, *_ = top
    _windows(p, left, front, (0.6, 0.75, 0.9), (2,), 2, arch=shade(BRICK, 1.3))
    _patch(p, left, front, (0.3, 0.45), (3, 6), (56, 44, 40))
    p.gable(top, 4, (86, 80, 88), BRICK)
    bracket = _on_wall(left, front, 0.08, 2)
    p.line([bracket, (bracket[0] - 3, bracket[1] + 1)], (60, 56, 54))
    ham = (196, 76, 66)
    hx, hy = bracket[0] - 3, bracket[1] + 4
    p.ellipse(hx - 3, hy - 2, hx + 1, hy + 2, ham)
    p.point(hx - 2, hy - 1, shade(ham, 1.3))
    p.line([(hx + 1, hy - 1), (hx + 2, hy - 2)], (244, 238, 224))
    p.point(hx + 3, hy - 3, (244, 238, 224))
    ramp = [_on_wall(left, front, t, 6) for t in (0.3, 0.45)]
    p.polygon(
        [ramp[0], ramp[1], (ramp[1][0] - 4, ramp[1][1] + 4), (ramp[0][0] - 4, ramp[0][1] + 4)],
        TIMBER,
        shade(TIMBER, 0.6),
    )
    pen_left, pen_back, pen_right, pen_front = _paddock(-10, 3, 5)
    _fence(p, pen_left, pen_back)
    _fence(p, pen_back, pen_right)
    _cow(p, -8, 4, (138, 84, 50), (96, 58, 36))
    _fence(p, pen_left, pen_front)
    _fence(p, pen_front, pen_right)
    cold = (234, 236, 238)
    store = p.box(11, 3, 5, 6, cold, shade(cold, 0.88))
    for t in (0.25, 0.5, 0.75):
        p.line(
            [_on_wall(store[0], store[1], t, 0.5), _on_wall(store[0], store[1], t, 6)],
            shade(cold, 0.85),
        )
    _patch(p, store[1], store[2], (0.3, 0.7), (2, 6), (150, 170, 186))
    p.box(12, -3.5, 2, 2, METAL, shade(METAL, 1.1))
    p.blob(11, -7, 13, -5, (60, 62, 70))
    p.line([(8, -4), (8, -9)], METAL)
    p.rect(7, -10, 9, -9, shade(METAL, 0.8))
    return p.sprite()


DIGITS = {
    1: ("010", "110", "010", "010", "111"),
    2: ("111", "001", "111", "100", "111"),
    3: ("111", "001", "111", "001", "111"),
}


def _plate(p: Painter, fill: RGB, half: float = 21, sides: int = 4) -> None:
    """The paved or planted ground of a site filling its whole cell: a slab with a dark lip, a
    diamond when it has four `sides` and nearly an oval when it has many."""
    rim = [
        (half * math.cos(a), half / 2 * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, sides + 1)[:-1]
    ]
    p.polygon([(x, y + 1) for x, y in rim], shade(fill, 0.68))
    p.polygon(rim, fill, shade(fill, 0.85))


def _oval(
    x: float, y: float, radius: float, rise: float, start: float, end: float, steps: int = 17
) -> list[Point]:
    """Points on the ground circle of `radius` about (x, y), raised by `rise`, from angle `start`
    to `end`: angle 0 is to the right and a quarter turn is to the front."""
    return [
        (x + radius * math.cos(a), y + radius / 2 * math.sin(a) - rise)
        for a in np.linspace(start, end, steps)
    ]


def _tree(p: Painter, x: float, y: float, size: float = 3) -> None:
    _, mid, light = LEAVES[0]
    p.line([(x, y), (x, y - size)], TRUNK)
    p.ellipse(x - size, y - 3 * size, x + size, y - size, mid)
    p.blob(x - size + 1, y - 3 * size + 1, x, y - 2 * size, light)


def _car(p: Painter, x: float, y: float, colour: RGB) -> None:
    p.box(x, y, 2, 1.5, colour, shade(colour, 1.2))
    p.point(x, y - 2, (172, 214, 234))


def _plane(p: Painter, x: float, y: float, size: float = 1) -> None:
    """An airliner standing at (x, y), its nose pointing the way the lit walls run; `size` sets its
    length and wingspan but not how thick it is."""
    body, stripe = (240, 242, 246), (70, 130, 210)
    edge = shade(body, 0.55)
    half = 8 * size

    def at(u: float, v: float, h: float) -> Point:
        px, py = _iso(x, y, u, v)
        return px, py - h

    p.polygon([at(-1.5, -half, 2), at(2, -half, 2), at(2.5, half, 2), at(-1, half, 2)], body, edge)
    tail = -half
    p.polygon(
        [
            at(tail - 0.5, -3, 3.5),
            at(tail + 1.5, -3, 3.5),
            at(tail + 1.5, 3, 3.5),
            at(tail - 0.5, 3, 3.5),
        ],
        body,
        edge,
    )
    p.polygon(
        [
            at(tail, -1.5, 1.5),
            at(half, -1.5, 1.5),
            at(half + 1.5, 0, 2.5),
            at(half, -1.5, 3.5),
            at(tail, -1.5, 3.5),
        ],
        shade(body, 0.9),
        edge,
    )
    p.polygon(
        [
            at(tail, -1.5, 3.5),
            at(half, -1.5, 3.5),
            at(half + 1.5, 0, 2.5),
            at(half, 1.5, 3.5),
            at(tail, 1.5, 3.5),
        ],
        body,
        edge,
    )
    p.line([at(tail + 1, -1.5, 2.5), at(half - 1, -1.5, 2.5)], stripe)
    p.polygon([at(tail, 0, 3.5), at(tail + 3, 0, 3.5), at(tail, 0, 9)], stripe, shade(stripe, 0.6))


def _tank_vehicle(p: Painter, x: float, y: float, olive: RGB) -> None:
    track = (50, 50, 54)
    base = p.hall(x, y, 9, 5, 2, track, None)
    for t in (0.15, 0.38, 0.62, 0.85):
        p.point(*_on_wall(base[0], base[1], t, 1), (130, 130, 136))
    hull = (150, 158, 108)
    p.hall(x, y - 2, 8, 4.5, 2, hull, shade(hull, 1.12))
    tx, ty = x - 1, y - 4
    p.box(tx, ty, 2.5, 2, shade(hull, 0.92), shade(hull, 1.18))
    p.line([(tx + 1, ty - 2), (tx + 8, ty + 1.5)], shade(olive, 0.5))
    p.line([(tx + 1, ty - 3), (tx + 8, ty + 0.5)], shade(hull, 0.85))


def _digit(p: Painter, x: float, y: float, digit: int, colour: RGB) -> None:
    """A digit three pixels wide and five tall with its top left at (x, y)."""
    for row, bits in enumerate(DIGITS[digit]):
        for col, bit in enumerate(bits):
            if bit == "1":
                p.point(x + col, y + row, colour)


def _water_tower(p: Painter, x: float, y: float, height: float = 14) -> None:
    legs, tank = (96, 88, 80), (176, 150, 118)
    for dx in (-3, 3):
        p.line([(x + dx, y), (x + dx * 0.7, y - height)], legs)
    p.line([(x, y + 1.5), (x, y - height)], shade(legs, 0.8))
    p.line([(x - 2.5, y - height / 2), (x + 2.5, y - height / 2)], legs)
    _turned(p, x, y - height, [(0, 3.5), (5, 3.5)], tank)
    _turned(p, x, y - height - 5, [(0, 4), (2.5, 0)], shade(tank, 0.7))


def _pylon(p: Painter, x: float, y: float, away: float) -> None:
    """A lattice pylon standing on (x, y), its cables running `away` pixels off sideways."""
    steel, cable = (110, 112, 120), (52, 52, 58)
    p.line([(x - 3, y), (x, y - 19)], steel)
    p.line([(x + 3, y), (x, y - 19)], steel)
    for dy, half in ((-15, 4), (-11, 3)):
        p.line([(x - half, y + dy), (x + half, y + dy)], steel)
    p.line([(x - 2, y - 4), (x + 2, y - 8)], steel)
    p.line([(x + 2, y - 4), (x - 2, y - 8)], steel)
    side = 1 if away > 0 else -1
    for dy, half in ((-15, 4), (-11, 3)):
        tip = (x + side * half, y + dy)
        p.line([tip, (tip[0] + away / 2, tip[1] + 1.5), (tip[0] + away, tip[1] - 1)], cable)


def _cooling_tower(p: Painter, x: float, y: float) -> None:
    concrete = (198, 196, 188)
    _turned(
        p, x, y, [(0, 7.5), (6, 5.8), (11, 4.6), (15, 4.9), (17, 5.4)], concrete, cap=(70, 70, 74)
    )
    for i, (dx, dy, r) in enumerate(((0, -19, 3.5), (3, -23, 4), (7, -26, 3.5))):
        p.blob(
            x + dx - r,
            y + dy - r,
            x + dx + r,
            y + dy + r,
            (240, 240, 242) if i % 2 else (224, 226, 230),
        )


def car_factory() -> Sprite:
    """A car works filling its cell: a long assembly hall under an arched roof beside a
    sawtoothed press shop, with new cars parked in rows on the lot out front."""
    p = Painter(50, 40, (24, 26))
    _plate(p, (166, 166, 162))
    wall = (156, 160, 170)
    press_wall = shade(wall, 0.92)
    press = p.hall(*_iso(0, 0, 3, 8), 14, 5, 6, press_wall, None)
    _sawtooth(p, press, 4, 3, (104, 110, 126), (160, 196, 218), press_wall)
    main = p.hall(*_iso(0, 0, 0, 0.5), 20, 7, 7, wall, None)
    left, front, right, _ = main
    _patch(p, left, front, (0.03, 0.97), (1.5, 3), (96, 136, 168))
    _patch(p, front, right, (0.2, 0.8), (1.5, 7), (64, 66, 74))
    _vault(p, main, 5, (98, 108, 130), wall)
    for row, v in enumerate((-5.5, -8.5)):
        for k in range(6):
            colour = CAR_COLOURS[(k + 2 * row) % len(CAR_COLOURS)]
            _car(p, *_iso(0, 0, -9 + 3.6 * k + row, v), colour)
    return p.sprite()


def plane_factory() -> Sprite:
    """An aircraft works filling its cell: a great hangar under an arched roof with its doors open,
    a long assembly shed behind, and a new airliner standing on the dark apron in front."""
    p = Painter(50, 40, (24, 26))
    _plate(p, (176, 176, 170))
    apron = (112, 114, 118)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-10.5, -10.5), (10.5, -10.5), (10.5, -1), (-10.5, -1))],
        apron,
    )
    p.line([_iso(0, 0, -10, -2.5), _iso(0, 0, 10, -2.5)], (226, 196, 70))
    wall = (200, 206, 214)
    shed = p.hall(*_iso(0, 0, 4, 9), 12, 3.5, 6, shade(wall, 0.9), (140, 150, 166))
    _windows(p, shed[0], shed[1], tuple(k / 7 for k in range(1, 7)), (2,), 1, (96, 136, 168))
    hangar = p.hall(*_iso(0, 0, -4, 3.5), 12, 9, 7, wall, None)
    left, front, right, _ = hangar
    _windows(p, left, front, tuple(k / 8 for k in range(1, 8)), (2,), 2, (96, 136, 168))
    _patch(p, front, right, (0.08, 0.92), (0.5, 7), (44, 48, 58))
    tail = [_on_wall(front, right, t, d) for t, d in ((0.55, 7), (0.62, 2), (0.68, 7))]
    p.polygon(tail, (70, 130, 210))
    _vault(p, hangar, 7, (150, 160, 176), wall)
    _plane(p, *_iso(0, 0, 4.5, -3.5), 0.75)
    return p.sprite()


def tank_factory() -> Sprite:
    """A tank works filling its cell: low armoured assembly halls in olive drab with slit windows
    and riveted bands, and a new tank standing on its churned-up proving ground."""
    p = Painter(50, 36, (24, 22))
    _plate(p, (150, 142, 112))
    olive = (104, 112, 80)
    roof = shade(olive, 0.72)
    mud = (112, 92, 66)
    p.polygon([_iso(0, 0, 0, -2), _iso(0, 0, 10, -2), _iso(0, 0, 10, -10), _iso(0, 0, 0, -10)], mud)
    for v in (-5, -7.5):
        p.line([_iso(0, 0, 1, v), _iso(0, 0, 9, v)], shade(mud, 0.7))
    for (u, v), length, depth, height, tone in (
        ((0, 7), 20, 6, 7, 1.0),
        ((-5.5, -2), 9, 8, 6, 0.9),
    ):
        top = p.hall(*_iso(0, 0, u, v), length, depth, height, shade(olive, tone), roof)
        for a, b in ((top[0], top[1]), (top[1], top[2])):
            p.line([_on_wall(a, b, 0, 1), _on_wall(a, b, 1, 1)], shade(olive, 1.3))
            p.line(
                [_on_wall(a, b, 0, height - 1), _on_wall(a, b, 1, height - 1)], shade(olive, 0.7)
            )
            for k in range(1, 6):
                t = k / 6
                p.line([_on_wall(a, b, t - 0.04, 2.5), _on_wall(a, b, t + 0.04, 2.5)], (34, 36, 30))
        _patch(p, top[0], top[1], (0.6, 0.85), (3.5, height - 1), shade(olive, 0.55))
    _tank_vehicle(p, *_iso(0, 0, 4.5, -6), olive)
    return p.sprite()


def mega_factory() -> Sprite:
    """A great works filling its cell: three long brick halls under sawtooth roofs, a row of tall
    chimneys behind them and a round gasholder."""
    p = Painter(50, 63, (24, 51))
    _plate(p, (150, 146, 140))
    for u in (-9, -4, 1, 6):
        x, y = _iso(0, 0, u, 10)
        _chimney(p, x, y - 4, y - 30, shade(BRICK, 0.8))
        _smoke(p, x + 1, y - 33, (214, 214, 218))
    for v, height, tone in ((6, 8, 0.9), (0, 9, 1.0), (-6, 7, 1.08)):
        wall = shade(BRICK, tone)
        top = p.hall(*_iso(0, 0, -1, v), 18, 5, height, wall, None)
        _windows(
            p, top[0], top[1], tuple(k / 8 for k in range(1, 8)), (2.5,), 2, arch=shade(BRICK, 1.3)
        )
        _sawtooth(p, top, 5, 3, METAL, (170, 200, 220), wall)
    gx, gy = _iso(0, 0, 9.5, 2)
    _turned(p, gx, gy, [(0, 4.5), (8, 4.5)], (118, 126, 118), cap=(138, 146, 138))
    for a in np.linspace(0.3, math.pi - 0.3, 5):
        fx, fy = gx + 5 * math.cos(a), gy + 2.5 * math.sin(a)
        p.line([(fx, fy), (fx, fy - 10)], (86, 92, 86))
    return p.sprite()


def refinery() -> Sprite:
    """An oil refinery filling its cell: tall silver distillation columns ringed with walkways,
    squat white storage tanks, pipe racks and a flare burning on a tall stack."""
    p = Painter(50, 58, (24, 46))
    _plate(p, (168, 164, 156))
    metal, white = (176, 178, 184), (224, 226, 228)
    fx, fy = _iso(0, 0, 9, 9)
    p.line([(fx, fy), (fx, fy - 36)], (90, 90, 96))
    p.line([(fx + 1, fy), (fx + 1, fy - 36)], (124, 124, 130))
    p.blob(fx - 2, fy - 43, fx + 3, fy - 36, (246, 150, 40))
    p.blob(fx - 0.5, fy - 40, fx + 1.5, fy - 37, (255, 220, 120))
    for u, v in ((-9, 8), (-3, 9)):
        _turned(p, *_iso(0, 0, u, v), [(0, 4.5), (4, 4.5)], white, cap=shade(white, 1.08))
    for u, v, r, h in ((-7, 2, 2, 18), (-2, 2, 2.5, 30), (3, 3, 3, 24)):
        x, y = _iso(0, 0, u, v)
        _turned(p, x, y, [(0, r), (h, r), (h + 1.5, r * 0.6)], metal)
        for ring in range(6, h, 6):
            p.line([(x - r - 1, y - ring), (x + r + 1, y - ring)], shade(metal, 0.6))
    rack = (96, 96, 104)
    for u in (-9, -3, 3, 9):
        x, y = _iso(0, 0, u, -2)
        p.line([(x, y), (x, y - 6)], rack)
    for h, colour in ((6, rack), (4.5, (150, 120, 60))):
        a, b = _iso(0, 0, -9, -2), _iso(0, 0, 9, -2)
        p.line([(a[0], a[1] - h), (b[0], b[1] - h)], colour)
    for u, v in ((-8, -6), (-1, -7.5), (6, -7)):
        _turned(p, *_iso(0, 0, u, v), [(0, 3.5), (3.5, 3.5)], white, cap=shade(white, 1.08))
    return p.sprite()


def power_plant() -> Sprite:
    """A coal power station filling its cell: a big brick turbine hall with tall windows, two
    concrete cooling towers letting off steam, a tall chimney, a coal heap fed by a conveyor and a
    pylon carrying power lines away."""
    p = Painter(50, 70, (24, 58))
    _plate(p, (160, 156, 148))
    cx, cy = _iso(0, 0, -9, 9)
    _chimney(p, cx, cy - 4, cy - 38, shade(BRICK, 0.9))
    _smoke(p, cx + 1, cy - 41, (124, 122, 126))
    _cooling_tower(p, *_iso(0, 0, -2, 9.5))
    _pylon(p, *_iso(0, 0, -8, -8), -4)
    top = p.hall(*_iso(0, 0, -3, -0.5), 14, 7, 12, BRICK, None)
    left, front, right, _ = top
    window = (88, 112, 140)
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2,), 7, window, shade(BRICK, 1.3))
    _patch(p, left, front, (0.8, 0.94), (7, 12), (58, 48, 44))
    _windows(p, front, right, (0.3, 0.7), (2,), 7, shade(window, 0.8), shade(BRICK, 1.2))
    p.gable(top, 4, (78, 78, 88), BRICK)
    _cooling_tower(p, *_iso(0, 0, 8.5, 4))
    hx, hy = _iso(0, 0, 5, -7)
    p.line([(hx - 1, hy - 5), _on_wall(left, front, 0.9, 4)], (70, 70, 78))
    p.line([(hx - 1, hy - 4), _on_wall(left, front, 0.9, 5)], (110, 110, 118))
    _heap(p, hx, hy, 5, 5, (48, 46, 50))
    for dx, dy in ((-2, -2), (0, -3), (2, -1), (-1, 0)):
        p.point(hx + dx, hy + dy, (104, 104, 116))
    return p.sprite()


def shipyard() -> Sprite:
    """A shipyard filling its cell: a big ship's hull on a slipway running down into the water,
    a gantry crane straddling it and a fabrication hall behind."""
    p = Painter(50, 46, (24, 32))
    _plate(p, STONE)
    water = (72, 128, 184)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((7.5, -10.5), (10.5, -10.5), (10.5, 10.5), (7.5, 10.5))],
        water,
    )
    for v in (-7, 5):
        p.line([_iso(0, 0, 8.5, v), _iso(0, 0, 9.5, v + 2)], (150, 196, 226))
    hall_wall = (176, 166, 152)
    hall = p.hall(*_iso(0, 0, -3, 7), 14, 6, 9, hall_wall, None)
    _windows(p, hall[0], hall[1], (0.15, 0.35, 0.55, 0.75), (2,), 3)
    _patch(p, hall[1], hall[2], (0.2, 0.8), (3, 9), shade(hall_wall, 0.5))
    p.gable(hall, 4, (92, 96, 110), hall_wall)
    slip = shade(STONE, 0.78)
    p.polygon([_iso(0, 0, u, v) for u, v in ((-10, -6.5), (9, -6.5), (9, 0.5), (-10, 0.5))], slip)
    crane = (214, 168, 40)
    for u in (-2,):
        far = _iso(0, 0, u, 1.5)
        p.line([far, (far[0], far[1] - 20)], crane)
    hull, band, deck = (150, 58, 46), (40, 40, 46), (196, 190, 178)
    ship = p.hall(*_iso(0, 0, -2, -3), 14, 5, 6, hull, deck)
    for a, b in ((ship[0], ship[1]), (ship[1], ship[2])):
        _patch(p, a, b, (0, 1), (0, 2), band)
    front, right = ship[1], ship[2]
    tip = _iso(*_on_wall(front, right, 0.5, 0), 4, 0)
    p.polygon([front, tip, (tip[0], tip[1] + 5), (front[0], front[1] + 6)], hull, shade(hull, 0.55))
    p.polygon([front, tip, (tip[0], tip[1] + 2), (front[0], front[1] + 2)], band)
    p.polygon([front, right, tip], deck, shade(deck, 0.6))
    bx, by = _iso(0, 0, -7, -3)
    bridge = p.box(bx, by - 6, 2.5, 4, (238, 236, 230), shade(STONE, 1.15))
    _windows(p, bridge[0], bridge[1], (0.3, 0.7), (1,), 1)
    p.rect(bx - 1, by - 14, bx + 1, by - 10, (196, 60, 48))
    near = _iso(0, 0, -2, -7.5)
    far = _iso(0, 0, -2, 1.5)
    p.line([near, (near[0], near[1] - 20)], crane)
    p.line([(near[0], near[1] - 20), (far[0], far[1] - 20)], crane)
    p.line([(near[0] + 1, near[1] - 19), (far[0] + 1, far[1] - 19)], shade(crane, 0.7))
    hook = _iso(0, 0, -2, -3)
    p.line([(hook[0], hook[1] - 19), (hook[0], hook[1] - 12)], (70, 64, 58))
    p.rect(hook[0] - 1, hook[1] - 12, hook[0] + 1, hook[1] - 10, (88, 90, 98))
    return p.sprite()


def space_center() -> Sprite:
    """A space centre: a tall white rocket on its launch pad beside a red gantry tower, a huge
    vehicle assembly hall and a low mission-control building with a dish."""
    p = Painter(50, 52, (24, 40))
    _plate(p, (184, 182, 176))
    p.line([_iso(0, 0, -2, 4), _iso(0, 0, 5, 4)], (150, 148, 142))
    vab = p.hall(*_iso(0, 0, -6, 4), 8, 8, 22, (224, 226, 230), (178, 182, 190))
    _patch(p, vab[0], vab[1], (0.3, 0.7), (3, 22), (70, 74, 84))
    for a, b in ((vab[0], vab[1]), (vab[1], vab[2])):
        p.line([_on_wall(a, b, 0, 1.5), _on_wall(a, b, 1, 1.5)], (60, 90, 170))
    px, py = _iso(0, 0, 6, 4)
    p.box(px, py, 5, 2, (150, 148, 144), (170, 168, 162))
    steel, white = (196, 84, 52), (242, 242, 238)
    for x in (px + 3, px + 5):
        p.line([(x, py - 3), (x, py - 33)], steel)
    for y in range(-6, -33, -4):
        p.line([(px + 3, py + y), (px + 5, py + y - 2)], steel)
    p.line([(px + 1, py - 25), (px + 3, py - 25)], steel)
    for dx in (-2.5, 2.5):
        _turned(p, px + dx, py - 2, [(0, 1.2), (12, 1.2), (14, 0)], white)
    _turned(p, px, py - 2, [(0, 2.5), (22, 2.5), (26, 1.6), (29, 0)], white)
    p.line([(px - 2, py - 14), (px + 2, py - 14)], (40, 40, 46))
    p.line([(px - 2, py - 22), (px + 2, py - 22)], (40, 40, 46))
    control = p.hall(*_iso(0, 0, 0, -6), 13, 5, 4, (204, 208, 214), (150, 156, 166))
    _patch(p, control[0], control[1], (0.05, 0.95), (1, 2.5), (90, 150, 196))
    dx, dy = _iso(0, 0, -4, -6)
    p.line([(dx, dy - 4), (dx, dy - 7)], METAL)
    p.ellipse(dx - 3, dy - 11, dx + 2, dy - 7, (236, 238, 240))
    return p.sprite()


def pear_hq() -> Sprite:
    """A modern campus: a great ring of glass offices around a wooded park, with trees all round
    it."""
    p = Painter(50, 38, (24, 24))
    _plate(p, (112, 160, 80), 20, 24)
    for x, y in ((-16, -2), (-9, -7), (0, -9), (9, -7), (16, -2)):
        _tree(p, x, y)
    big, small, h = 16, 10, 5
    roof, glass = (226, 230, 232), (120, 176, 204)
    p.polygon(_oval(0, 0, big, h, 0, 2 * math.pi, 33), roof, shade(roof, 0.7))
    p.polygon(_oval(0, 0, small, h, 0, 2 * math.pi, 25), shade(glass, 0.72))
    p.polygon(
        _oval(0, 0, small, 0, math.pi, 2 * math.pi) + _oval(0, 0, small, h, 0, math.pi),
        (98, 152, 70),
    )
    for x, y in ((-5, -2), (1, -3), (5, -1), (-1, -0.5)):
        _tree(p, x, y, 2)
    p.polygon(
        _oval(0, 0, big, h, 0, math.pi) + _oval(0, 0, small, h, math.pi, 0), roof, shade(roof, 0.75)
    )
    split = math.acos(1 / 3)
    p.polygon(_oval(0, 0, big, 0, split, math.pi) + _oval(0, 0, big, h, math.pi, split), glass)
    p.polygon(
        _oval(0, 0, big, 0, 0, split, 9) + _oval(0, 0, big, h, split, 0, 9), shade(glass, 0.78)
    )
    for a in np.linspace(0.2, math.pi - 0.2, 12):
        bx, by = big * math.cos(a), big / 2 * math.sin(a)
        p.line([(bx, by - 0.5), (bx, by - h + 0.5)], shade(glass, 1.3))
    p.line(_oval(0, 0, big, 0, 0, math.pi), shade(glass, 0.55))
    for x, y in ((-18, 3), (-12, 7), (12, 7), (18, 3), (0, 10)):
        _tree(p, x, y)
    return p.sprite()


def royal_palace() -> Sprite:
    """A baroque royal palace: long symmetrical wings with rows of tall windows, end pavilions and
    a central pavilion with a pediment under a green dome, behind a formal garden with a
    fountain."""
    p = Painter(50, 40, (24, 27))
    _plate(p, (206, 198, 166))
    hedge, bed = (70, 120, 56), (98, 154, 72)
    for u in (-6, 6):
        corners = [
            _iso(0, 0, u + du, -5 + dv)
            for du, dv in ((-3.5, -3.5), (3.5, -3.5), (3.5, 3.5), (-3.5, 3.5))
        ]
        p.polygon(corners, bed, hedge)
        p.line([_iso(0, 0, u - 3.5, -5), _iso(0, 0, u + 3.5, -5)], hedge)
    fx, fy = _iso(0, 0, 0, -5)
    p.ellipse(fx - 3, fy - 1.5, fx + 3, fy + 1.5, (96, 160, 206))
    p.line([(fx, fy), (fx, fy - 4)], (226, 240, 250))
    wall, roof = (236, 214, 160), (96, 106, 126)
    wing = p.hall(*_iso(0, 0, 0, 6), 21, 5, 7, wall, None)
    _windows(p, wing[0], wing[1], tuple(k / 17 for k in range(1, 17)), (1.5, 4.5), 2)
    p.gable(wing, 3, roof, wall)
    dome_x, dome_y = _iso(0, 0, 0, 5)
    _turned(p, dome_x, dome_y - 9, [(0, 3.5), (3, 3.5)], wall)
    _turned(p, dome_x, dome_y - 12, [(0, 4), (2, 3.4), (3.5, 2.2), (5, 0)], VERDIGRIS)
    _flag(p, dome_x, dome_y - 17, 6)
    for u in (-9, 9):
        pavilion = p.hall(*_iso(0, 0, u, 4.5), 3.5, 7, 9, wall, None)
        _windows(p, pavilion[0], pavilion[1], (0.5,), (1.5, 4.5), 2)
        _windows(p, pavilion[1], pavilion[2], (0.3, 0.7), (1.5, 4.5), 2, shade(WINDOW, 0.75))
        p.gable(pavilion, 3, roof, wall)
    centre = p.hall(*_iso(0, 0, 0, 2.5), 6, 3, 9, shade(wall, 1.04), None)
    _windows(p, centre[0], centre[1], (0.25, 0.5, 0.75), (1.5, 4.5), 2)
    _portal(p, centre[0], centre[1], (0.38, 0.62), 9, 2)
    _gable_across(p, centre, 3, roof, shade(wall, 1.04))
    return p.sprite()


def salt_cathedral() -> Sprite:
    """A salt mine's head: a timber shaft house with a tall tower over the shaft and a headframe
    wheel, white heaps of salt, and a little white chapel with a cross marking the cathedral
    carved in the salt below."""
    p = Painter(50, 46, (24, 34))
    _plate(p, (178, 172, 154))
    timber, shingle = (138, 98, 62), (92, 70, 54)
    wx, wy = _iso(0, 0, 4, 8)
    frame = (74, 64, 58)
    p.line([(wx - 3, wy), (wx, wy - 22), (wx + 3, wy)], frame)
    p.line([(wx - 2, wy - 8), (wx + 2, wy - 8)], frame)
    p.ellipse(wx - 3, wy - 26, wx + 3, wy - 20, frame)
    p.blob(wx - 1.5, wy - 24.5, wx + 1.5, wy - 21.5, (178, 172, 154))
    house = p.hall(*_iso(0, 0, -3, 3), 12, 7, 6, timber, None)
    _windows(p, house[0], house[1], (0.15, 0.35, 0.8), (2,), 2, (240, 220, 150))
    _portal(p, house[0], house[1], (0.5, 0.66), 6, 3)
    p.gable(house, 5, shingle, timber)
    p.line([(wx, wy - 23), _on_wall(house[2], house[3], 0.5, -6)], (52, 52, 58))
    tx, ty = _iso(0, 0, -3, 3)
    tower = p.box(tx, ty - 6, 3, 10, shade(timber, 1.1), None)
    _windows(p, tower[0], tower[1], (0.5,), (2,), 3, (54, 46, 48))
    apex = (tx, ty - 22)
    p.polygon([tower[0], tower[1], apex], shingle, shade(shingle, 0.55))
    p.polygon([tower[1], tower[2], apex], shade(shingle, 0.7), shade(shingle, 0.45))
    white = (240, 242, 246)
    for u, v, half, height in ((7, -2, 4, 6), (3, -7, 3.5, 5), (8, -7, 3, 4)):
        _heap(p, *_iso(0, 0, u, v), half, height, white)
    chapel = p.box(*_iso(0, 0, -7, -6), 3, 4, WHITEWASH, None)
    p.gable(chapel, 3, CHURCH_ROOF, WHITEWASH)
    _portal(p, chapel[0], chapel[1], (0.35, 0.65), 4, 2)
    ridge = _on_wall(chapel[1], chapel[2], 0.5, -3)
    _cross(p, ridge[0], ridge[1])
    return p.sprite()


def spa_resort() -> Sprite:
    """A spa resort: an elegant white spa house under a green roof and cupola, a long colonnade
    for taking the waters, a pool with a fountain and a small park of trees."""
    p = Painter(50, 40, (24, 27))
    _plate(p, (114, 162, 82))
    gravel = (214, 206, 176)
    p.polygon([_iso(0, 0, u, v) for u, v in ((-8, -9.5), (8, -9.5), (8, 0), (-8, 0))], gravel)
    for u, v in ((-9.5, 9), (9.5, 9), (-9.5, 3)):
        _tree(p, *_iso(0, 0, u, v))
    cream = (242, 234, 216)
    house = p.hall(*_iso(0, 0, 0, 6.5), 16, 6, 8, cream, None)
    _windows(
        p, house[0], house[1], tuple(k / 9 for k in range(1, 9)), (2,), 3, arch=shade(cream, 0.8)
    )
    p.gable(house, 4, VERDIGRIS, cream)
    cx, cy = _iso(0, 0, 0, 6.5)
    _turned(p, cx, cy - 11, [(0, 2), (2, 2)], cream)
    _turned(p, cx, cy - 13, [(0, 2.4), (1.5, 1.6), (2.5, 0)], VERDIGRIS)
    walk = p.hall(*_iso(0, 0, 0, 1.5), 18, 2.5, 4, (88, 80, 76), VERDIGRIS)
    for k in range(10):
        t = (k + 0.5) / 10
        p.line([_on_wall(walk[0], walk[1], t, 0.5), _on_wall(walk[0], walk[1], t, 4)], cream)
    pool = [_iso(0, 0, u, v) for u, v in ((-6, -8), (6, -8), (6, -3), (-6, -3))]
    p.polygon(pool, (92, 172, 214), (240, 240, 236))
    px, py = _iso(0, 0, 0, -5.5)
    p.ellipse(px - 2, py - 1, px + 2, py + 1, (240, 240, 236))
    p.line([(px, py), (px, py - 5)], (226, 242, 252))
    for dx in (-2, 2):
        p.point(px + dx, py - 3, (226, 242, 252))
    for u, v in ((-9.5, -6), (9.5, -6), (9.5, -1)):
        _tree(p, *_iso(0, 0, u, v))
    return p.sprite()


def worlds_fair() -> Sprite:
    """A world's fair: a vast exhibition palace of iron and glass under barrel-vaulted glass roofs,
    its taller middle crowned with flags."""
    p = Painter(50, 46, (24, 32))
    _plate(p, (206, 200, 186))
    wall, roof, iron = (184, 220, 234), (214, 236, 244), (86, 98, 118)
    main = p.hall(0, 0, 21, 9, 8, wall, None)
    for k in range(1, 7):
        p.line([_on_wall(main[0], main[1], k / 7, 0.5), _on_wall(main[0], main[1], k / 7, 8)], iron)
    for k in range(1, 3):
        p.line(
            [_on_wall(main[1], main[2], k / 3, 0.5), _on_wall(main[1], main[2], k / 3, 8)],
            shade(iron, 0.8),
        )
    for a, b in ((main[0], main[1]), (main[1], main[2])):
        p.line([_on_wall(a, b, 0, 4), _on_wall(a, b, 1, 4)], iron)
    _vault(p, main, 5, roof, wall, (0.15, 0.3, 0.7, 0.85), iron)
    centre = p.hall(0, 0, 7, 12, 13, wall, None)
    for k in range(1, 4):
        p.line(
            [_on_wall(centre[0], centre[1], k / 4, 0.5), _on_wall(centre[0], centre[1], k / 4, 13)],
            iron,
        )
    p.line([_on_wall(centre[0], centre[1], 0, 6), _on_wall(centre[0], centre[1], 1, 6)], iron)
    _vault(p, centre, 6, roof, wall, (0.25, 0.5, 0.75), iron)
    _portal(p, centre[0], centre[1], (0.3, 0.7), 13, 5)
    poles = (
        _on_wall(centre[0], centre[3], 0.5, -6),
        _on_wall(main[0], main[3], 0.5, -5),
        _on_wall(main[1], main[2], 0.5, -5),
    )
    for (x, y), colour in zip(poles, ((214, 58, 54), (60, 110, 200), (240, 200, 70)), strict=True):
        _flag(p, x, y, 7, colour)
    return p.sprite()


def film_studio() -> Sprite:
    """A film studio: big sound-stage sheds with their numbers painted on, a water tower on legs,
    and the studio gate with its sign."""
    p = Painter(50, 44, (24, 31))
    _plate(p, (150, 150, 146))
    stage, paint = (218, 206, 180), (60, 56, 60)
    for (u, v), length, depth, height, number in (((6.5, 6), 8, 8, 9, 2), ((-4, 5), 12, 9, 11, 1)):
        top = p.hall(*_iso(0, 0, u, v), length, depth, height, stage, None)
        p.gable(top, 3, (120, 116, 112), stage)
        x, y = _on_wall(top[0], top[1], 0.5, 2.5)
        p.rect(x - 3, y - 1, x + 3, y + 6, (240, 236, 226))
        _digit(p, x - 1, y, number, paint)
        _patch(p, top[1], top[2], (0.25, 0.75), (3, height), shade(stage, 0.55))
    _water_tower(p, *_iso(0, 0, 7, -4))
    gx, gy = _iso(0, 0, -5, -8)
    pillar = (230, 220, 196)
    for dx in (-5, 5):
        p.rect(gx + dx - 1, gy + dx / 2 - 8, gx + dx + 1, gy + dx / 2, pillar)
    p.polygon(
        [(gx - 6, gy - 11.5), (gx + 6, gy - 5.5), (gx + 6, gy - 8.5), (gx - 6, gy - 14.5)],
        (200, 64, 56),
        shade((200, 64, 56), 0.6),
    )
    p.line([(gx - 4, gy - 11), (gx + 4, gy - 7)], (250, 214, 96))
    p.box(*_iso(0, 0, 0, -9), 2, 3, (240, 236, 226), shade(STONE, 1.1))
    return p.sprite()


def amusement_park() -> Sprite:
    """An amusement park: a big Ferris wheel with coloured gondolas, a roller coaster looping the
    loop, a carousel and striped stalls along its paths."""
    p = Painter(50, 50, (24, 37))
    _plate(p, (116, 164, 84))
    gravel = (214, 206, 176)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-10.5, -2), (10.5, -2), (10.5, -4), (-10.5, -4))], gravel
    )
    rail, post = (214, 56, 56), (236, 236, 232)
    track = [(-1, 9, 2), (2, 9, 10), (4, 9, 3), (6, 8, 3), (8, 6, 7), (10, 3, 2), (9, 0, 2)]
    points = []
    for u, v, h in track:
        x, y = _iso(0, 0, u, v)
        p.line([(x, y), (x, y - h)], post)
        points.append((x, y - h))
    p.line(points, rail)
    lx, ly = _iso(0, 0, 6, 8)
    loop = [
        (lx + 4 * math.cos(a), ly - 7 + 5 * math.sin(a))
        for a in np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, 25)
    ]
    p.line(loop, rail)
    wx, wy = _iso(0, 0, -5, 4)
    hub = (wx, wy - 14)
    frame = (120, 120, 132)
    p.line([(wx - 5, wy), hub, (wx + 5, wy)], frame)
    rim = [
        (hub[0] + 11 * math.cos(a), hub[1] + 11 * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, 33)
    ]
    for a in np.linspace(0, math.pi, 5)[:-1]:
        p.line(
            [
                (hub[0] - 11 * math.cos(a), hub[1] - 11 * math.sin(a)),
                (hub[0] + 11 * math.cos(a), hub[1] + 11 * math.sin(a)),
            ],
            frame,
        )
    p.line(rim, (236, 236, 240))
    for k, a in enumerate(np.linspace(0, 2 * math.pi, 9)[:-1]):
        gx, gy = hub[0] + 11 * math.cos(a), hub[1] + 11 * math.sin(a)
        p.rect(gx - 1, gy, gx + 1, gy + 2, CAR_COLOURS[k % len(CAR_COLOURS)])
    for (u, v), awning in (
        ((-8, -7), (214, 56, 56)),
        ((-3, -8.5), (60, 110, 200)),
        ((9, -6), (240, 200, 70)),
    ):
        stall = p.box(*_iso(0, 0, u, v), 2.5, 3, (236, 230, 214), None)
        p.polygon(
            [
                _on_wall(stall[0], stall[3], 0, -1.5),
                stall[0],
                stall[1],
                stall[2],
                _on_wall(stall[3], stall[2], 1, -1.5),
                (stall[3][0], stall[3][1] - 1.5),
            ],
            awning,
            shade(awning, 0.6),
        )
        p.line([stall[0], stall[1], stall[2]], (250, 250, 248))
    cx, cy = _iso(0, 0, 3, -7)
    _turned(p, cx, cy, [(0, 4), (1, 4)], (230, 220, 196))
    for a in np.linspace(0.4, math.pi - 0.4, 4):
        px = cx + 3.5 * math.cos(a)
        p.line([(px, cy + 1.7 * math.sin(a)), (px, cy + 1.7 * math.sin(a) - 4)], (240, 200, 70))
    _turned(p, cx, cy - 5, [(0, 4.5), (3.5, 0)], (214, 56, 56))
    _flag(p, cx, cy - 8.5, 3, (240, 200, 70))
    return p.sprite()


def olympic_stadium() -> Sprite:
    """An Olympic stadium: an oval bowl of stands around a red running track and a green field,
    with floodlight masts round its rim and the flame burning in a cauldron on a tall stand."""
    p = Painter(52, 44, (25, 30))
    big, small, h = 20, 12, 7
    stand, seats = (196, 196, 202), ((198, 64, 60), (64, 96, 176))
    p.polygon(_oval(0, 0, big, h, 0, 2 * math.pi, 41), stand, shade(stand, 0.7))
    for k in range(1, 6):
        f = k / 6
        p.line(
            _oval(0, 0, big + (small - big) * f, h * (1 - f), math.pi, 2 * math.pi, 25),
            seats[k % 2],
        )

    def mast(angle: float) -> None:
        mx, my = big * math.cos(angle), big / 2 * math.sin(angle) - h
        p.line([(mx, my), (mx, my - 13)], (120, 120, 128))
        p.rect(mx - 1.5, my - 16, mx + 1.5, my - 13, (250, 246, 210))

    for angle in (math.pi + 0.7, 2 * math.pi - 0.7):
        mast(angle)
    p.polygon(_oval(0, 0, small, 0, 0, 2 * math.pi, 33), (192, 82, 62))
    p.line(_oval(0, 0, small - 1.5, 0, 0, 2 * math.pi, 33), (236, 200, 190))
    p.polygon(_oval(0, 0, small - 3, 0, 0, 2 * math.pi, 33), (92, 156, 68))
    p.line([(0, -3.5), (0, 3.5)], (220, 240, 210))
    split = math.acos(1 / 3)
    p.polygon(
        _oval(0, 0, big, 0, split, math.pi) + _oval(0, 0, big, h, math.pi, split),
        shade(stand, 0.92),
    )
    p.polygon(
        _oval(0, 0, big, 0, 0, split, 9) + _oval(0, 0, big, h, split, 0, 9), shade(stand, 0.74)
    )
    for a in np.linspace(0.25, math.pi - 0.25, 10):
        ax, ay = big * math.cos(a), big / 2 * math.sin(a)
        p.line([(ax, ay - 1), (ax, ay - 3)], (70, 70, 80))
    p.line(_oval(0, 0, big, h, 0, math.pi, 25), shade(stand, 1.15))
    for angle in (0.7, math.pi - 0.7):
        mast(angle)
    cx, cy = -big - 1, -h
    p.rect(cx - 1, cy - 12, cx + 1, cy + 5, (150, 150, 158))
    p.ellipse(cx - 3, cy - 14, cx + 3, cy - 11, CROSS)
    p.blob(cx - 2.5, cy - 20, cx + 2.5, cy - 13, (246, 140, 40))
    p.blob(cx - 1, cy - 18, cx + 1, cy - 14, (255, 220, 110))
    return p.sprite()


def relief(width: int, height: int, seed: int, *, peaked: bool) -> Sprite:
    """A hill, or a peaked mountain when `peaked`, as a height field seen from the view angle:
    every point of its ground is raised by its height and shaded by how much it faces the light
    from the top left, dithered between TINT_GREYS. The tallest mountains get snow on their tops,
    in colours that are not grey so they keep them when tinted."""
    rng = np.random.default_rng(seed)
    a = width / 2
    gx = np.arange(-a, a) + 0.5
    gy = np.arange(-a, a, 0.25) + 0.125
    x, y = np.meshgrid(gx / a, gy / a)
    envelope = np.clip(1 - x**2 - y**2, 0, 1)
    if peaked:
        lift = np.zeros_like(x)
        for _ in range(3):
            cx, cy = rng.uniform(-0.35, 0.35, 2)
            reach = rng.uniform(0.45, 0.7)
            cone = np.clip(1 - np.hypot(x - cx, (y - cy) * 1.3) / reach, 0, 1)
            lift = np.maximum(lift, rng.uniform(0.6, 1.0) * cone**1.2)
        ridges = 1 - np.abs(2 * _noise(rng, x.shape, 7) - 1)
        lift = lift * (0.8 + 0.35 * ridges) * np.sqrt(envelope)
    else:
        lift = np.zeros_like(x)
        for _ in range(2):
            cx, cy = rng.uniform(-0.3, 0.3, 2)
            reach = rng.uniform(0.45, 0.65)
            lift += rng.uniform(0.6, 1.0) * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / reach**2))
        lift = lift * (0.9 + 0.2 * _noise(rng, x.shape, 4)) * envelope
    rise = lift / lift.max() * height
    slope_y, slope_x = np.gradient(rise, 0.25, 1.0)
    towards_light = np.array([-0.62, 0.1, 0.78]) / np.hypot(np.hypot(-0.62, 0.1), 0.78)
    facing = (towards_light[2] - towards_light[0] * slope_x - towards_light[1] * slope_y) / np.sqrt(
        1 + slope_x**2 + slope_y**2
    )
    grain = _noise(rng, x.shape, 12) - 0.5
    value = (
        2.6 + (4 if peaked else 3) * (facing - towards_light[2]) + (0.8 if peaked else 0.3) * grain
    )
    snowy = peaked and height >= 11
    snow = rise > 0.72 * height + 1.2 * grain

    anchor_y = int(np.ceil(height + a / 2)) + 1
    w, h = width + 2, anchor_y + int(np.ceil(a / 2)) + 1
    image = np.zeros((h, w, 4), dtype=np.uint8)
    rows = np.arange(h)[:, None]
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 - 0.5
    dither = bayer[rows % 4, np.arange(w) % 4]
    for i in range(len(gy)):
        top = anchor_y + gy[i] / 2 - rise[i]
        base = anchor_y + gy[i] / 2
        raised = rise[i] > 0.4
        cover = raised & (rows >= np.floor(top)) & (rows <= np.floor(base))
        cover = np.pad(cover, ((0, 0), (1, 1)))
        level = np.clip(np.rint(np.pad(value[i], 1)[None, :] + dither), 0, 5).astype(int)
        rgb = np.array(TINT_GREYS)[level][..., None].repeat(3, axis=-1)
        if snowy:
            lit = np.pad(value[i], 1)[None, :] + dither > 2.2
            white = np.where(lit[..., None], SNOW_LIT, SNOW_SHADE)
            rgb = np.where(np.pad(snow[i], 1)[None, :, None], white, rgb)
        image[cover, :3] = rgb[cover]
        image[cover, 3] = 255
    return Sprite(Image.fromarray(image), (w // 2, anchor_y))


def _noise(rng: np.random.Generator, shape: tuple[int, ...], cells: int) -> np.ndarray:
    """Smooth random values from 0 to 1 over `shape`, varying over about `cells` steps across."""
    coarse = rng.random((cells + 1, cells + 1))
    v = np.linspace(0, cells, shape[0])
    u = np.linspace(0, cells, shape[1])
    v0, u0 = np.minimum(v.astype(int), cells - 1), np.minimum(u.astype(int), cells - 1)
    fv, fu = (v - v0)[:, None], (u - u0)[None, :]
    fv, fu = fv * fv * (3 - 2 * fv), fu * fu * (3 - 2 * fu)
    top = coarse[v0][:, u0] * (1 - fu) + coarse[v0][:, u0 + 1] * fu
    bottom = coarse[v0 + 1][:, u0] * (1 - fu) + coarse[v0 + 1][:, u0 + 1] * fu
    return top * (1 - fv) + bottom * fv


BUILDING_SCALE = 0.7
HOMES = ("house", "tenement", "flats")
HOME_SCALE = 0.59
# A farm's barn, an orchard's cottage and a bakery stand a touch bigger than a house.
FARMSTEADS = ("farm", "orchard", "orange_farm", "bakery")
FARMSTEAD_SCALE = 0.585
BUILDINGS = (
    "house",
    "tenement",
    "flats",
    "farm",
    "orchard",
    "date_palms",
    "bananas",
    "sugar",
    "iron_mine",
    "copper_mine",
    "gold_mine",
    "diamond_mine",
    "oil_well",
    "salt_works",
    "fishery",
    "sheep_farm",
    "cotton_field",
    "vineyard",
    "coffee_plantation",
    "rubber_plantation",
    "textile_mill",
    "furniture_maker",
    "tool_works",
    "winery",
    "refinery",
    "chemical_works",
    "electronics_plant",
    "car_factory",
    "car_dealer",
    "jeweller",
    "orange_farm",
    "outpost",
    "base",
    "headquarters",
    "gun_factory",
    "tank_factory",
    "plane_factory",
    "airport",
    "barrel_maker",
    "beekeeper",
    "incense_grove",
    "church",
    "basilica",
    "cathedral",
    "st_pauls",
    "factory",
    "big_factory",
    "mega_factory",
    "skyscraper",
    "food_plant",
    "bakery",
    "steel_mill",
    "shipyard",
    "mine",
    "lumber_mill",
    "port",
    "clay_pit",
    "limestone_quarry",
    "sand_pit",
    "potato_farm",
    "cattle_ranch",
    "flax_field",
    "hop_garden",
    "machine_works",
    "brickworks",
    "cement_works",
    "glassworks",
    "brewery",
    "distillery",
    "tannery",
    "shoe_factory",
    "plastics_works",
    "appliance_factory",
    "paper_mill",
    "printing_works",
    "sugar_refinery",
    "porcelain_works",
    "tavern",
    "department_store",
    "hotel",
    "meat_factory",
    "power_plant",
    "space_center",
    "pear_hq",
    "royal_palace",
    "salt_cathedral",
    "spa_resort",
    "worlds_fair",
    "film_studio",
    "amusement_park",
    "olympic_stadium",
)

SPRITES: dict[str, Callable[[], Sprite]] = {
    **{f"pine_{i}": (lambda c=c: pine(c)) for i, c in enumerate(PINES)},
    **{f"leafy_{i}": (lambda c=c: leafy(c)) for i, c in enumerate(LEAVES)},
    "rocks": rocks,
    **{
        f"{HILL}_{i}": (lambda w=w, h=h, i=i: relief(w, h, i, peaked=False))
        for i, (w, h) in enumerate(HILL_SIZES)
    },
    **{
        f"{MOUNTAIN}_{i}": (lambda w=w, h=h, i=i: relief(w, h, 100 + i, peaked=True))
        for i, (w, h) in enumerate(MOUNTAIN_SIZES)
    },
    **{f"house_{i}": (lambda w=WALLS[i % 3], r=ROOFS[i % 4]: house(w, r)) for i in range(12)},
    **{
        f"tenement_{i}": (lambda w=w, r=TENEMENT_ROOFS[i % 2]: tenement(w, r))
        for i, w in enumerate(TENEMENT_WALLS)
    },
    **{f"flats_{i}": (lambda w=w: flats(w)) for i, w in enumerate(FLAT_WALLS)},
    "farm": farm,
    "factory": factory,
    "big_factory": big_factory,
    "mega_factory": mega_factory,
    "skyscraper": skyscraper,
    "mine": mine,
    "lumber_mill": lumber_mill,
    "port": port,
    "orchard": orchard,
    "iron_mine": lambda: shaft_mine((104, 72, 62), (176, 98, 70)),
    "copper_mine": lambda: shaft_mine((96, 84, 66), (206, 130, 76)),
    "gold_mine": lambda: shaft_mine((88, 78, 62), (255, 214, 60)),
    "diamond_mine": lambda: shaft_mine((58, 60, 72), (190, 238, 255)),
    "oil_well": oil_well,
    "salt_works": salt_works,
    "fishery": fishery,
    "sheep_farm": pasture,
    "cotton_field": lambda: plantation((196, 176, 136), (130, 92, 62)),
    "vineyard": lambda: plantation((220, 204, 170), (160, 84, 56)),
    "coffee_plantation": lambda: plantation((186, 150, 108), (96, 66, 44)),
    "rubber_plantation": lambda: plantation((150, 136, 104), (72, 92, 62)),
    "textile_mill": textile_mill,
    "furniture_maker": furniture_maker,
    "tool_works": tool_works,
    "winery": winery,
    "refinery": refinery,
    "chemical_works": chemical_works,
    "electronics_plant": electronics_plant,
    "car_factory": car_factory,
    "car_dealer": lambda: shop((226, 224, 218), (236, 80, 132), (60, 90, 170)),
    "jeweller": lambda: shop((70, 62, 84), (232, 190, 70), (190, 238, 255)),
    "orange_farm": lambda: orchard([(244, 142, 36)]),
    "outpost": outpost,
    "base": base,
    "headquarters": headquarters,
    "airport": airport,
    "barrel_maker": barrel_maker,
    "beekeeper": beekeeper,
    "incense_grove": incense_grove,
    "church": church,
    "basilica": basilica,
    "cathedral": cathedral,
    "st_pauls": st_pauls,
    "gun_factory": lambda: works((120, 104, 78), (84, 74, 58), (190, 56, 48), 1),
    "tank_factory": tank_factory,
    "plane_factory": plane_factory,
    "date_palms": date_palms,
    "bananas": bananas,
    "sugar": sugar,
    "station": station,
    "station_2": station_2,
    "station_3": station_3,
    "food_plant": food_plant,
    "bakery": bakery,
    "steel_mill": steel_mill,
    "shipyard": shipyard,
    "clay_pit": clay_pit,
    "limestone_quarry": limestone_quarry,
    "sand_pit": sand_pit,
    "potato_farm": potato_farm,
    "cattle_ranch": cattle_ranch,
    "flax_field": flax_field,
    "hop_garden": hop_garden,
    "machine_works": machine_works,
    "brickworks": brickworks,
    "cement_works": cement_works,
    "glassworks": glassworks,
    "brewery": brewery,
    "distillery": distillery,
    "tannery": tannery,
    "shoe_factory": shoe_factory,
    "plastics_works": plastics_works,
    "appliance_factory": appliance_factory,
    "paper_mill": paper_mill,
    "printing_works": printing_works,
    "sugar_refinery": sugar_refinery,
    "porcelain_works": porcelain_works,
    "tavern": tavern,
    "department_store": department_store,
    "hotel": hotel,
    "meat_factory": meat_factory,
    "power_plant": power_plant,
    "space_center": space_center,
    "pear_hq": pear_hq,
    "royal_palace": royal_palace,
    "salt_cathedral": salt_cathedral,
    "spa_resort": spa_resort,
    "worlds_fair": worlds_fair,
    "film_studio": film_studio,
    "amusement_park": amusement_park,
    "olympic_stadium": olympic_stadium,
}


def shrink(sprite: Sprite, factor: float) -> Sprite:
    """A smaller copy of a sprite: each new pixel takes the colour of the old pixel under its
    middle, so edges stay hard and colours stay the sprite's own."""
    w, h = sprite.image.size
    small = sprite.image.resize(
        (max(1, round(w * factor)), max(1, round(h * factor))), Image.Resampling.NEAREST
    )
    ax, ay = sprite.anchor
    return Sprite(small, (round(ax * factor), round(ay * factor)))


def _scale_of(name: str) -> float:
    if name.startswith(HOMES):
        return HOME_SCALE
    if name in FARMSTEADS:
        return FARMSTEAD_SCALE
    return BUILDING_SCALE if name.startswith(BUILDINGS) else 1.0


def fine_sprite(name: str, resolution: int) -> Sprite:
    """A sprite drawn at `resolution` pixels for every art pixel of the map, at its size on the
    map. Hills and mountains are not drawn with a Painter, so they are blown up instead."""
    if name.startswith((HILL, MOUNTAIN)):
        sprite = SPRITES[name]()
        w, h = sprite.image.size
        big = sprite.image.resize((w * resolution, h * resolution), Image.Resampling.NEAREST)
        return Sprite(big, (sprite.anchor[0] * resolution, sprite.anchor[1] * resolution))
    with drawn_at(_scale_of(name) * resolution):
        return SPRITES[name]()


def draw_sheet(padding: int = 1, resolution: int = 1) -> tuple[Image.Image, dict[str, list[int]]]:
    """Every sprite packed in rows on one sheet, and where each sits: x, y, width, height and its
    anchor inside it. Buildings other than stations are drawn at BUILDING_SCALE, houses,
    tenements and flats at HOME_SCALE, and farms and orchards at FARMSTEAD_SCALE. At a
    resolution above 1 each sprite is drawn that many times finer (fine_sprite) rather than
    shrunk from its full size."""
    if resolution == 1:
        sprites = {name: shrink(draw(), _scale_of(name)) for name, draw in SPRITES.items()}
    else:
        sprites = {name: fine_sprite(name, resolution) for name in SPRITES}
    width = 128 * resolution
    x = y = row_height = 0
    places: dict[str, list[int]] = {}
    for name, sprite in sprites.items():
        w, h = sprite.image.size
        if x + w > width:
            x, y, row_height = 0, y + row_height + padding, 0
        places[name] = [x, y, w, h, *sprite.anchor]
        x += w + padding
        row_height = max(row_height, h)
    sheet = Image.new("RGBA", (width, y + row_height), (0, 0, 0, 0))
    for name, sprite in sprites.items():
        sheet.paste(sprite.image, tuple(places[name][:2]))
    return sheet, places
