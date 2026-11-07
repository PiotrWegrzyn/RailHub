"""The game's sprites, drawn pixel by pixel in code and packed into one sheet.

`tools/build_sprites.py` saves the sheet as a PNG with a JSON index next to it, so any sprite can
be repainted by hand afterwards; the game only ever reads the PNG. Sprites are drawn for a cell 32
pixels wide, seen from the fixed view angle, with light from the top left. Each one stands on an
anchor: the point that sits on the ground where the thing is.

Hills and mountains are drawn in TINT_GREYS and coloured with the shades of the ground they stand
on when the map is drawn, so one hill fits grass, forest, desert or snow alike. Only pure greys
are recoloured.
"""

from collections.abc import Callable
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


def shade(color: RGB, factor: float) -> RGB:
    r, g, b = color
    return (min(255, round(r * factor)), min(255, round(g * factor)), min(255, round(b * factor)))


@dataclass(frozen=True)
class Sprite:
    image: Image.Image
    anchor: tuple[int, int]


class Painter:
    """Draws on one sprite's canvas, with the anchor at (0, 0)."""

    def __init__(self, width: int, height: int, anchor: tuple[int, int]) -> None:
        self.image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self.anchor = anchor
        self._draw = ImageDraw.Draw(self.image)

    def _at(self, points: list[Point]) -> list[Point]:
        ax, ay = self.anchor
        return [(ax + x, ay + y) for x, y in points]

    def polygon(self, points: list[Point], fill: RGB, outline: RGB | None = None) -> None:
        self._draw.polygon(self._at(points), fill=fill, outline=outline)

    def line(self, points: list[Point], fill: RGB) -> None:
        self._draw.line(self._at(points), fill=fill)

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
        self._draw.point(self._at([(x, y)]), fill=fill)

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
        s, h = half_width, height
        left, front, right, back = (x - s, y), (x, y + s / 2), (x + s, y), (x, y - s / 2)

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


def mega_factory() -> Sprite:
    """Two long halls with a row of chimneys, filling a whole cell."""
    p = Painter(40, 50, (18, 41))
    p.shadow(16, 5, reach=4)
    p.box(-4, -2, 11, 9, shade(BRICK, 0.9), (88, 86, 92))
    p.box(5, 3, 12, 7, BRICK, (96, 94, 100))
    for k in range(4):
        x, y = -6 + 5.5 * k, -4 + 1.4 * k
        p.polygon([(x, y), (x + 5.5, y - 2.8), (x + 5.5, y + 1.4)], METAL, shade(METAL, 0.55))
        p.polygon([(x, y), (x, y - 1.4), (x + 5.5, y - 2.8)], shade(METAL, 1.15))
    for x in (-12, -6, 0):
        p.rect(x, -30, x + 2, -10, shade(BRICK, 0.75))
        p.line([(x, -28), (x + 2, -28)], (220, 220, 214))
        for i, (dx, dy, r) in enumerate([(1, -33, 2), (3, -36, 2.5)]):
            p.blob(
                x + dx - r, dy - r, x + dx + r, dy + r, (226, 226, 230) if i else (204, 204, 210)
            )
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


def station() -> Sprite:
    """A brick station hall with a long platform canopy and a clock tower."""
    p = Painter(34, 34, (16, 24))
    p.polygon([(-15, 0), (0, 7.5), (15, 0), (0, -7.5)], STONE, shade(STONE, 0.6))
    p.shadow(10, 3)
    top = p.box(-3, 0, 8, 7, BRICK, None)
    p.gable(top, 4, (70, 76, 92), BRICK)
    p.polygon([(4, 3), (14, -2), (14, -6), (4, -1)], (150, 156, 168), shade(METAL, 0.55))
    p.line([(13, -2), (13, 1)], shade(METAL, 0.6))
    p.line([(6, 3), (6, 6)], shade(METAL, 0.6))
    p.rect(-6, -20, -2, -8, shade(BRICK, 1.1))
    p.ellipse(-5.5, -18.5, -2.5, -15.5, (236, 232, 220))
    p.polygon([(-6, -20), (-4, -24), (-2, -20)], (70, 76, 92))
    for x in (-9, -6, 0):
        p.point(x, -2 + (x + 9) * 0.05, WINDOW)
    return p.sprite()


def orchard() -> Sprite:
    """Rows of small fruit trees on a patch of mown grass."""
    p = Painter(32, 22, (16, 13))
    p.polygon([(-14, 0), (0, 7), (14, 0), (0, -7)], (116, 158, 66), (92, 128, 54))
    leaves = LEAVES[1]
    for row in range(3):
        for col in range(3):
            x = -8 + 5 * col + 3 * row - 3
            y = -3 + 2.5 * row - 1.2 * col + 1
            p.line([(x, y), (x, y - 2)], TRUNK)
            p.blob(x - 2.5, y - 6, x + 2.5, y - 1.5, leaves[1])
            p.point(x - 1, y - 5, leaves[2])
            p.point(x + 1, y - 3, FRUIT[(row + col) % len(FRUIT)])
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


def shipyard() -> Sprite:
    """A ship's hull on a slipway under a gantry crane."""
    p = Painter(34, 34, (16, 24))
    p.polygon([(-15, 0), (0, 7.5), (15, 0), (0, -7.5)], STONE, shade(STONE, 0.6))
    hull = (120, 60, 50)
    p.polygon([(-10, -2), (6, 6), (9, 3), (-8, -5)], hull, shade(hull, 0.55))
    p.polygon([(-8, -5), (9, 3), (8, 0), (-7, -7)], (176, 172, 160), shade(STONE, 0.5))
    p.rect(-1, -8, 3, -3, (210, 206, 196))
    crane = (214, 168, 40)
    for x in (-12, 12):
        p.line([(x, 0), (x, -20)], crane)
    p.line([(-12, -20), (12, -20)], crane)
    p.line([(2, -20), (2, -11)], (70, 64, 58))
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
BUILDINGS = (
    "house",
    "tenement",
    "flats",
    "farm",
    "orchard",
    "factory",
    "big_factory",
    "mega_factory",
    "skyscraper",
    "food_plant",
    "steel_mill",
    "shipyard",
    "mine",
    "lumber_mill",
    "port",
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
    "station": station,
    "food_plant": food_plant,
    "steel_mill": steel_mill,
    "shipyard": shipyard,
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


def draw_sheet(padding: int = 1) -> tuple[Image.Image, dict[str, list[int]]]:
    """Every sprite packed in rows on one sheet, and where each sits: x, y, width, height and its
    anchor inside it. Buildings other than stations are drawn at BUILDING_SCALE."""
    sprites = {
        name: shrink(draw(), BUILDING_SCALE) if name.startswith(BUILDINGS) else draw()
        for name, draw in SPRITES.items()
    }
    width = 128
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
