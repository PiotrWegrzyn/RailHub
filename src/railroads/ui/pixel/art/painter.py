"""The Painter every sprite is drawn with, the Sprite it makes, and the colours they share."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from PIL import Image, ImageDraw

type RGB = tuple[int, int, int]

type Point = tuple[float, float]

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


def fineness() -> float:
    """How many pixels sprites drawn now take for every unit of their layout (drawn_at)."""
    return _fineness


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


DIGITS = {
    1: ("010", "110", "010", "010", "111"),
    2: ("111", "001", "111", "100", "111"),
    3: ("111", "001", "111", "001", "111"),
}
