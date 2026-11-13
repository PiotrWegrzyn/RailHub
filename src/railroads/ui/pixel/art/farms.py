"""Farms, orchards, plantations, pastures and the other fields."""

import itertools
import math

from railroads.ui.pixel.art.painter import (
    BRICK,
    FRUIT,
    HIVE,
    METAL,
    RGB,
    STONE,
    THATCH,
    TRUNK,
    WHITEWASH,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import (
    _cow,
    _fence,
    _heap,
    _on_wall,
    _paddock,
    _patch,
    _turned,
)

DOOR: RGB = (104, 66, 44)
SILL: RGB = (240, 236, 226)
CRATE: RGB = (156, 108, 62)
SACK: RGB = (192, 162, 112)
WICKER: RGB = (176, 132, 76)
WOOL: RGB = (240, 236, 224)
WATER: RGB = (92, 148, 196)
EARTH: RGB = (146, 124, 84)
WORN: RGB = (120, 128, 70)
MUD: RGB = (96, 76, 54)
FOOTING: RGB = (62, 70, 40)

type Top = tuple[Point, Point, Point, Point]

_SCATTER: tuple[Point, ...] = (
    (-0.9, 0.2),
    (0.7, -0.4),
    (-0.2, 0.7),
    (1.0, 0.5),
    (-0.6, -0.6),
    (0.3, 0.1),
    (-1.1, -0.2),
)


def _footing(p: Painter, top: Top, height: float, colour: RGB = FOOTING) -> None:
    """A dark damp strip on the ground where both walls of a block `height` high stand in it."""
    left, front, right, _ = top
    (lx, ly), (fx, fy), (rx, ry) = ((x, y + height) for x, y in (left, front, right))
    p.polygon(
        [
            (lx - 0.8, ly),
            (fx, fy),
            (rx + 0.8, ry),
            (rx + 0.8, ry + 0.6),
            (fx, fy + 1.1),
            (lx - 0.8, ly + 0.6),
        ],
        colour,
    )


def _worn(p: Painter, x: float, y: float, half: float, colour: RGB = EARTH) -> None:
    """A patch of bare trodden earth on the ground at (x, y), `half` across, its rim worn
    into the grass."""
    p.blob(x - half, y - half / 2, x + half, y + half / 2, WORN)
    p.blob(x - half * 0.68, y - half * 0.3, x + half * 0.68, y + half * 0.3, colour)


def _track(p: Painter, points: list[Point], wide: float = 1, colour: RGB = EARTH) -> None:
    """A trodden track `wide` across along the ground through `points`."""
    for rim, w in ((WORN, wide), (colour, wide * 0.55)):
        for (x0, y0), (x1, y1) in itertools.pairwise(points):
            steps = max(1, round(math.hypot(x1 - x0, y1 - y0) / 0.8))
            for i in range(steps + 1):
                x, y = x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps
                p.blob(x - w, y - w / 2, x + w, y + w / 2, rim)


def _ruts(p: Painter, points: list[Point], colour: RGB = EARTH) -> None:
    """A cart track along the ground through `points`: worn earth with two darker ruts."""
    _track(p, points, 1.4, colour)
    for side in (-0.7, 0.7):
        p.line([(x + side, y + side / 3) for x, y in points], shade(colour, 0.72))


def _mud(p: Painter, x: float, y: float, half: float) -> None:
    """Trampled wet mud on the ground round (x, y), `half` across, with a glint of a puddle."""
    p.blob(x - half, y - half / 2, x + half, y + half / 2, shade(MUD, 1.25))
    p.blob(x - half * 0.75, y - half * 0.32, x + half * 0.75, y + half * 0.36, MUD)
    p.line([(x - half * 0.4, y + half * 0.1), (x - half * 0.1, y + half * 0.1)], (132, 150, 156))


def _strewn(p: Painter, x: float, y: float, half: float, colour: RGB, count: int = 5) -> None:
    """`count` bits of `colour` dropped on the ground round (x, y), up to `half` away."""
    for k, (u, v) in enumerate(_SCATTER[:count]):
        p.point(x + u * half, y + v * half / 2, shade(colour, 0.85 if k % 2 else 1))


def _ridge(top: Top, rise: float) -> tuple[Point, Point]:
    """The back and front ends of the ridge of a gable `rise` high on a box's top (Painter.gable)."""
    left, front, right, back = top
    return (
        ((left[0] + back[0]) / 2, (left[1] + back[1]) / 2 - rise),
        ((front[0] + right[0]) / 2, (front[1] + right[1]) / 2 - rise),
    )


def _tiles(p: Painter, top: Top, rise: float, roof: RGB, rows: tuple[float, ...] = (0.34, 0.68)):
    """Rows of tiles, slates or thatch across the lit side of a gable roof."""
    left, front, _, _ = top
    ridge_back, ridge_front = _ridge(top, rise)
    for f in rows:
        a = (left[0] + (ridge_back[0] - left[0]) * f, left[1] + (ridge_back[1] - left[1]) * f)
        b = (front[0] + (ridge_front[0] - front[0]) * f, front[1] + (ridge_front[1] - front[1]) * f)
        p.line([a, b], shade(roof, 0.8))


def _chimney(
    p: Painter, top: Top, rise: float, along: float, tall: float = 2, colour: RGB = BRICK
) -> Point:
    """A chimney with a dark cap standing on the ridge, `along` of the way to its front. Returns
    the top of it."""
    ridge_back, ridge_front = _ridge(top, rise)
    x = ridge_back[0] + (ridge_front[0] - ridge_back[0]) * along
    y = ridge_back[1] + (ridge_front[1] - ridge_back[1]) * along
    p.rect(x - 1, y - tall, x, y + 1, colour)
    p.line([(x - 1, y - tall - 1), (x, y - tall - 1)], shade(colour, 0.55))
    return x, y - tall - 1


def _window(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float, fill: RGB = WINDOW
) -> None:
    """A window with a light sill under it on the wall under the edge from `a` to `b`."""
    _patch(p, a, b, span, (drop, drop + 1.2), fill)
    p.line([_on_wall(a, b, span[0], drop + 1.5), _on_wall(a, b, span[1], drop + 1.5)], SILL)


def _door(
    p: Painter,
    a: Point,
    b: Point,
    span: tuple[float, float],
    drop: float,
    foot: float,
    fill: RGB = DOOR,
) -> None:
    """A door from `drop` down to `foot` on the wall under the edge from `a` to `b`, under a
    light lintel."""
    _patch(p, a, b, span, (drop, foot), fill)
    p.line([_on_wall(a, b, span[0], drop - 0.4), _on_wall(a, b, span[1], drop - 0.4)], SILL)


def _plinth(p: Painter, top: Top, wall: RGB, height: float, band: float = 0.9) -> None:
    """A darker band `band` high along the foot of both walls of a block `height` high."""
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (height - band, height), shade(wall, 0.7))
    _patch(p, front, right, (0, 1), (height - band, height), shade(wall, 0.5))


def _boards(p: Painter, top: Top, wall: RGB, drops: tuple[float, ...]) -> None:
    """The seams between the boards of a timber wall, `drops` down both walls."""
    left, front, right, _ = top
    for drop in drops:
        p.line([_on_wall(left, front, 0, drop), _on_wall(left, front, 1, drop)], shade(wall, 0.84))
        p.line(
            [_on_wall(front, right, 0, drop), _on_wall(front, right, 1, drop)], shade(wall, 0.66)
        )


def _crate(p: Painter, x: float, y: float, load: RGB, half: float = 1.6) -> None:
    """A slatted wooden crate standing on (x, y), filled to its brim with `load`."""
    left, front, _, _ = p.box(x, y, half, half, CRATE, load)
    p.line(
        [_on_wall(left, front, 0, half / 2), _on_wall(left, front, 1, half / 2)], shade(CRATE, 0.7)
    )
    p.point(x - half / 2, y - half - 0.5, shade(load, 1.3))


def _sack(
    p: Painter, x: float, y: float, fill: RGB = SACK, tall: float = 4, half: float = 1.5
) -> None:
    """A full sack standing on (x, y), tied off at its neck."""
    p.ellipse(x - half, y - tall, x + half, y, fill)
    p.line([(x - 0.6, y - tall + 1), (x + 0.6, y - tall + 1)], shade(fill, 0.55))
    p.point(x, y - tall, shade(fill, 0.5))


def _basket(p: Painter, x: float, y: float, load: RGB) -> None:
    """A wicker basket standing on (x, y), heaped with `load`."""
    p.blob(x - 2, y - 3.5, x + 2, y - 1.5, load)
    p.polygon([(x - 2, y - 2.5), (x + 2, y - 2.5), (x + 1.5, y), (x - 1.5, y)], WICKER)
    p.line([(x - 1.8, y - 1.2), (x + 1.8, y - 1.2)], shade(WICKER, 0.7))
    p.point(x - 1, y - 3.5, shade(load, 1.3))


def _sheep(p: Painter, x: float, y: float, facing: int = 1) -> None:
    face = (50, 46, 44)
    p.blob(x - 1.6, y + 0.4, x + 2.6, y + 1.8, (70, 98, 44))
    p.point(x - 1, y + 0.5, face)
    p.point(x + 1, y + 0.5, face)
    p.blob(x - 2, y - 1, x + 2, y + 1, WOOL)
    p.blob(x - 1.5, y - 1.2, x + 0.5, y, shade(WOOL, 1.06))
    head = x + 2 if facing > 0 else x - 3
    p.point(head, y - 1, face)
    p.point(head, y, face)


def _arc(p: Painter, x: float, y: float, half: float, colour: RGB) -> None:
    """The near half of a ring `half` wide round an upright axis at (x, y)."""
    p.line(
        [
            (x + half * math.cos(a / 6 * math.pi), y + half / 2 * math.sin(a / 6 * math.pi))
            for a in range(7)
        ],
        colour,
    )


def farm() -> Sprite:
    """A farm: a red barn with cross-braced doors and a hayloft door under its roof, a round silo
    with hoops and a ladder, a haystack shedding straw and a bit of fence, with the yard trodden
    bare before the barn doors and a cart track running off from it."""
    p = Painter(22, 22, (9, 17))
    barn = (172, 62, 46)
    hay = (220, 186, 96)
    _ruts(p, [(-4.6, 3.4), (-4.2, 4.8), (-3.4, 6)])
    _worn(p, -4.8, 3.6, 3.4)
    _strewn(p, -8.2, 3.6, 2.4, hay, 6)
    p.shadow(8, 3)
    _heap(p, -8, 3, 2, 3, hay)
    top = p.box(-1, 0, 7, 6, barn, None)
    left, front, right, _ = top
    _footing(p, top, 6)
    _patch(p, left, front, (0, 1), (5.2, 6), shade(barn, 0.65))
    _patch(p, left, front, (0.3, 0.72), (1.6, 6), shade(barn, 0.6))
    trim = (232, 226, 210)
    door = [_on_wall(left, front, t, d) for t, d in ((0.3, 1.6), (0.72, 1.6), (0.72, 6), (0.3, 6))]
    p.line([door[0], door[1], door[2], door[3], door[0]], trim)
    p.line([door[0], door[2]], trim)
    p.line([door[1], door[3]], trim)
    _patch(p, front, right, (0.2, 0.4), (2, 3.4), shade(barn, 0.45))
    p.gable(top, 4, (112, 98, 84), barn)
    gx, gy = (front[0] + right[0]) / 2, (front[1] + right[1]) / 2 - 1.5
    p.rect(gx - 1, gy - 1, gx, gy, (60, 36, 28))
    silo = (210, 208, 200)
    p.blob(6.2, -0.2, 11, 2, FOOTING)
    _turned(p, 8.5, 1, [(0, 1.8), (12, 1.8)], silo)
    for h in (3.5, 7, 10.5):
        _arc(p, 8.5, 1 - h, 1.8, shade(silo, 0.7))
    _turned(p, 8.5, -11, [(0, 1.8), (1.2, 1.5), (2.2, 0.8), (2.7, 0)], METAL)
    p.line([(9.8, 1), (9.8, -10)], shade(METAL, 0.62))
    for y in (-2, -5, -8):
        p.point(10.3, y, shade(METAL, 0.62))
    _fence(p, (1, 6), (8, 2.5))
    return p.sprite()


def orchard(fruit: list[RGB] = FRUIT) -> Sprite:
    """A fruit grower's cottage on a plinth, with a door, windows with sills, a tiled roof and a
    chimney, a picking ladder leant on its wall, crates of the fruit stacked on trodden earth
    beside it with windfalls about them and a basket of it by the door, where a path runs off;
    the trees themselves are planted on the ground (railroads.ui.pixel.world)."""
    p = Painter(22, 20, (9, 15))
    _track(p, [(-4.4, 2.6), (-4.6, 4.4), (-3.8, 6)])
    _worn(p, -4.4, 2.8, 2.4)
    _worn(p, 9.4, 2.2, 3.4)
    p.shadow(8, 3)
    wall, roof = (226, 208, 166), (124, 82, 54)
    top = p.box(-1, 0, 7, 5, wall, None)
    left, front, right, _ = top
    _footing(p, top, 5)
    _plinth(p, top, wall, 5)
    _window(p, left, front, (0.12, 0.3), 1)
    _door(p, left, front, (0.48, 0.64), 1.3, 5)
    _window(p, left, front, (0.78, 0.94), 1)
    _window(p, front, right, (0.35, 0.62), 1, shade(WINDOW, 0.8))
    p.gable(top, 4, roof, wall)
    _tiles(p, top, 4, roof)
    _chimney(p, top, 4, 0.3)
    ladder = (200, 164, 108)
    foot, head = _on_wall(front, right, 0.12, 5.8), _on_wall(front, right, 0.2, 0.6)
    for side in (0, 1.3):
        p.line([(foot[0] + side, foot[1] - side / 2), (head[0] + side, head[1] - side / 2)], ladder)
    for k in (0.3, 0.6):
        x, y = foot[0] + (head[0] - foot[0]) * k, foot[1] + (head[1] - foot[1]) * k
        p.line([(x, y), (x + 1.3, y - 0.65)], shade(ladder, 0.8))
    _crate(p, 10.5, 1, fruit[1 % len(fruit)])
    _crate(p, 8.5, 2.6, fruit[0])
    _crate(p, 8.5, 1, fruit[2 % len(fruit)])
    for k, (x, y) in enumerate(((11.5, 3.2), (6.4, 3.6), (12.4, 1.8), (-2.6, 4.4))):
        p.point(x, y, fruit[k % len(fruit)])
    _basket(p, -6.5, 4.4, fruit[0])
    return p.sprite()


def _frond(p: Painter, x: float, y: float, reach: Point, arch: float, tone: float) -> None:
    """A feather-shaped palm frond springing from a crown at (x, y) to its tip `reach` away,
    bowed up `arch` in its middle, with a lit midrib and a fringe of leaflets hanging under it."""
    lit, leaf = shade((136, 186, 82), tone), shade((56, 112, 48), tone)
    dx, dy = reach

    def at(t: float) -> Point:
        return x + dx * t, y + dy * t - arch * math.sin(math.pi * t**0.6)

    steps = 7
    rib = [at(i / steps) for i in range(steps + 1)]
    hang = [
        (px - 0.25 * dx / abs(dx or 1), py + 1.5 * math.sin(math.pi * (0.15 + 0.85 * i / steps)))
        for i, (px, py) in enumerate(rib)
    ]
    p.polygon(rib + hang[::-1], shade(leaf, 0.85))
    for i in range(1, steps, 2):
        p.line([rib[i], (hang[i + 1][0], hang[i + 1][1] + 0.4)], leaf)
    p.line(rib, lit)


def _palm(p: Painter, x: float, y: float, tall: float, lean: float, date: RGB) -> None:
    """A date palm rooted at (x, y): a curving ringed trunk `tall` high leaning `lean` across at
    its crown, a crown of arching fronds and bunches of dates hung under it."""
    trunk = (128, 96, 62)

    def spine(t: float) -> Point:
        return x + lean * t**2, y - tall * t

    steps = 8
    core = [spine(i / steps) for i in range(steps + 1)]
    width = [1.1 - 0.35 * i / steps for i in range(steps + 1)]
    lit = [(cx - w, cy) for (cx, cy), w in zip(core, width, strict=True)]
    mid = [(cx + 0.2, cy) for cx, cy in core]
    dark = [(cx + w, cy) for (cx, cy), w in zip(core, width, strict=True)]
    p.polygon(lit + mid[::-1], trunk)
    p.polygon(mid + dark[::-1], shade(trunk, 0.72))
    for i in range(1, steps * 2):
        cx, cy = spine(i / (steps * 2))
        w = 1.1 - 0.35 * i / (steps * 2)
        p.line([(cx - w, cy + 0.4), (cx + w, cy - 0.3)], shade(trunk, 0.58 if i % 2 else 1.22))
    tx, ty = spine(1)
    for reach, arch in (((-4.5, -3), 1.8), ((4.5, -3.5), 1.8), ((0.5, -5.5), 0.6)):
        _frond(p, tx, ty, reach, arch, 0.78)
    stalk = (206, 164, 82)
    for side in (-1, 1):
        p.line([(tx, ty), (tx + 1.4 * side, ty + 1.6)], stalk)
        bx, by = tx + 1.6 * side, ty + 2
        p.blob(bx - 1.1, by, bx + 1.1, by + 3, shade(date, 0.8 if side > 0 else 1))
        p.point(bx - 0.6, by + 0.6, shade(date, 1.25))
        p.point(bx, by + 2.4, shade(date, 0.62))
    p.blob(tx - 1.1, ty - 0.8, tx + 1.1, ty + 0.8, shade(trunk, 0.8))
    for reach, arch in (
        ((-6.5, 3), 2.8),
        ((6.5, 2.5), 2.8),
        ((-6, -1), 2.6),
        ((6, -1.5), 2.6),
        ((-2.5, 4.5), 1.8),
        ((3, 4.5), 1.8),
    ):
        _frond(p, tx, ty, reach, arch, 1.0)


def date_palms() -> Sprite:
    """Three date palms on a patch of sand watered by a ditch, each in its own damp basin: tall
    curving ringed trunks under crowns of long arching fronds hung with bunches of dates, a
    fallen frond and a basket of picked dates."""
    p = Painter(32, 26, (16, 20))
    sand = (214, 192, 134)
    damp = shade(sand, 0.82)
    p.polygon([(-14, 0), (0, 7), (14, 0), (0, -7)], sand, shade(sand, 0.86))
    for x, y in ((-6, 4), (5, -3), (11, 1), (-11, -0.5)):
        p.point(x, y, shade(sand, 1.08))
    p.polygon([(-12, 0.2), (-1, 5.7), (0.5, 5.2), (-10.5, -0.3)], damp)
    p.line([(-11.2, 0.2), (-0.6, 5.5)], WATER)
    for x, y in ((1, -3), (-5, 1), (7.5, 1.5)):
        p.blob(x - 2.6, y - 1.1, x + 2.6, y + 1.3, damp)
        p.blob(x - 1.6, y - 0.6, x + 1.6, y + 0.7, shade(sand, 0.74))
    date = (190, 104, 40)
    _palm(p, 1, -3, 15, 0.8, date)
    _palm(p, -5, 1, 11.5, -2.4, date)
    _palm(p, 7.5, 1.5, 12.5, 2.2, shade(date, 0.92))
    fallen = (150, 140, 76)
    p.line([(2, 4.5), (7, 4)], fallen)
    for k in range(4):
        p.point(2.8 + 1.2 * k, 5 - 0.1 * k, shade(fallen, 0.8))
    _basket(p, -2, 5.5, date)
    return p.sprite()


def _leaf(p: Painter, x: float, y: float, reach: Point, arch: float, colour: RGB) -> None:
    """A broad banana leaf from (x, y) to its tip `reach` away, bowed up `arch` and drooping at
    its end, lit along its upper half with a pale midrib."""
    dx, dy = reach

    def at(t: float) -> Point:
        return x + dx * t, y + dy * t - arch * math.sin(math.pi * t**0.6)

    rib = [at(i / 5) for i in range(6)]
    width = [1.3 * math.sin(math.pi * (0.1 + 0.9 * i / 5)) ** 0.7 for i in range(6)]
    upper = [(px, py - w) for (px, py), w in zip(rib, width, strict=True)]
    lower = [(px, py + w) for (px, py), w in zip(rib, width, strict=True)]
    p.polygon(upper + rib[::-1], colour)
    p.polygon(rib + lower[::-1], shade(colour, 0.72))
    p.line(rib, shade(colour, 1.4))


def bananas() -> Sprite:
    """Rows of banana plants on earthed-up mounds, their broad leaves arching from thick stems
    over yellow bunches and red flower buds, a trodden path between the rows and a crate of cut
    bunches on it."""
    p = Painter(32, 22, (16, 14))
    p.polygon([(-14, 0), (0, 7), (14, 0), (0, -7)], (74, 120, 48), (60, 100, 40))
    _track(p, [(-12, 2.2), (-3, 3.6), (4, 3.4), (9.5, 4.6)], 0.9)
    stem, leaf = (110, 128, 60), (66, 140, 54)
    yellow = (236, 206, 64)
    soil = (92, 82, 52)
    plants = [
        (-10 + 7 * col + 3 * row, 3.5 * row - 1.5 * col) for row in range(2) for col in range(3)
    ]
    for x, y in sorted(plants, key=lambda q: q[1]):
        p.blob(x - 2, y - 0.6, x + 2.6, y + 1.2, soil)
        p.polygon([(x - 0.6, y), (x - 0.4, y - 5), (x + 0.9, y - 5), (x + 1.1, y)], stem)
        p.line([(x + 0.6, y), (x + 0.6, y - 5)], shade(stem, 0.7))
        cx, cy = x + 0.3, y - 5
        _leaf(p, cx, cy, (-1, -4.5), 0.8, shade(leaf, 0.9))
        _leaf(p, cx, cy, (-4.5, 0.5), 1.6, leaf)
        p.polygon([(x + 1, y - 4), (x + 2.4, y - 4), (x + 2.2, y - 2), (x + 1.2, y - 2)], yellow)
        p.point(x + 1, y - 3, shade(yellow, 0.75))
        p.point(x + 1.6, y - 1.4, (170, 52, 70))
        _leaf(p, cx, cy, (4.5, 1), 1.6, shade(leaf, 0.9))
        _leaf(p, cx, cy, (2.2, -4), 1, shade(leaf, 1.12))
    _worn(p, 5.6, 5.6, 2.6)
    _crate(p, 5, 5.2, yellow, 1.8)
    p.point(4, 2.6, (96, 120, 52))
    return p.sprite()


def sugar() -> Sprite:
    """A field of tall jointed sugar cane in rows along dark furrows, leaves splaying from its
    tops, and a bundle of cut cane tied up on the trodden headland before it, with cut stalks
    left lying about."""
    p = Painter(30, 18, (15, 11))
    ground = (132, 112, 72)
    p.polygon([(-13, 0), (0, 6), (13, 0), (0, -6)], ground, shade(ground, 0.85))
    _track(p, [(-8, 3.6), (0, 5.2), (11, 3)], 1.2, shade(EARTH, 1.06))
    cane = (118, 176, 82)
    for row in range(4):
        x0, y0 = -12 + 2 * row, -1 + 2 * row
        base = [(x0 - 1 + 4 * col, y0 - col + 0.4) for col in range(5)]
        crown = [(x + dx, y - 5.4 - (0.8 if dx else 0)) for x, y in base for dx in (-1, 0, 1)]
        p.polygon(base[::-1] + crown, shade(cane, 0.62 + 0.04 * row))
        p.line([(x0 - 1.6, y0 + 0.9), (x0 + 16.6, y0 - 3.3)], shade(ground, 0.68))
        for x, y in base:
            p.line([(x + 1, y), (x + 1, y - 5)], cane)
            p.point(x + 1, y - 2.2, shade(cane, 0.55))
            p.point(x + 1, y - 4, shade(cane, 0.55))
            p.line([(x + 1, y - 5), (x + 0.2, y - 6.6)], (166, 206, 104))
            p.line([(x + 1, y - 5), (x + 2, y - 6.4)], shade(cane, 1.15))
    cut = (176, 182, 104)
    for k in range(3):
        p.line([(5 + 0.5 * k, 5 - 0.6 * k), (10 + 0.5 * k, 2.5 - 0.6 * k)], shade(cut, 1 - 0.1 * k))
    p.line([(7.2, 4.6), (7.8, 2.8)], (128, 96, 58))
    p.line([(-5, 4.4), (-2.2, 4.8)], shade(cut, 0.85))
    p.line([(1, 5.6), (3, 4.8)], shade(cut, 0.9))
    return p.sprite()


def fishery() -> Sprite:
    """A boarded fisherman's hut with a door, windows and a chimney, a rack of fish hung up to
    dry, a basket of the catch, and a small boat with its sail furled drawn up on a shingle beach
    at the water's edge, with a trodden path down to it from the hut door."""
    p = Painter(30, 22, (15, 16))
    shingle = (198, 184, 146)
    p.polygon([(-2, 7), (7, 1.5), (15, -1.5), (15, 7)], shingle)
    p.polygon([(6, 7), (14, 0.5), (15, 0.5), (15, 7)], shade(shingle, 0.8))
    p.polygon([(8, 7), (15, 1), (15, 7)], WATER)
    p.line([(8.5, 6.6), (15, 0.9)], (226, 236, 238))
    p.line([(12, 5.4), (14.5, 3.4)], shade(WATER, 1.25))
    for x, y in ((1, 5.6), (4, 3.8), (8.5, 1.4), (12.5, 0)):
        p.point(x, y, shade(shingle, 0.7))
    _track(p, [(-7, 2.6), (-5, 4.4), (-1, 5.2), (2, 5)])
    _worn(p, -7.2, 3, 2.2)
    p.shadow(12, 3)
    wall, roof = (132, 112, 88), (84, 106, 130)
    fish = (120, 170, 196)
    pole = (88, 64, 44)
    for x, y in ((2, 1), (6, -1)):
        p.blob(x - 1.2, y - 0.4, x + 1.6, y + 0.8, FOOTING)
    p.line([(2, 1), (2, -6)], pole)
    p.line([(6, -1), (6, -8)], pole)
    p.line([(2, -6), (6, -8)], shade(pole, 1.2))
    for t in (0.2, 0.45, 0.7, 0.92):
        fx, fy = 2 + 4 * t, -6 - 2 * t
        p.line([(fx, fy + 1), (fx, fy + 3)], fish)
        p.point(fx, fy + 3, shade(fish, 0.7))
    top = p.box(-5, 0, 6, 5, wall, None)
    left, front, right, _ = top
    _footing(p, top, 5)
    _boards(p, top, wall, (1.6, 3.2))
    _plinth(p, top, wall, 5)
    _window(p, left, front, (0.14, 0.34), 1)
    _door(p, left, front, (0.56, 0.76), 1.5, 5)
    _window(p, front, right, (0.4, 0.66), 1, shade(WINDOW, 0.8))
    p.gable(top, 3, roof, wall)
    _tiles(p, top, 3, roof, (0.5,))
    _chimney(p, top, 3, 0.25, 1.2, STONE)
    hull = (150, 62, 48)
    p.blob(4, 5, 13, 7.4, shade(shingle, 0.62))
    p.line([(8, 3), (8, -6)], pole)
    p.line([(9, -5), (9, 1)], (226, 214, 190))
    p.line([(8, -5), (11, -4)], pole)
    p.polygon([(3, 3), (13, 3), (11, 6), (5, 6)], hull, (92, 40, 34))
    p.line([(3, 3), (13, 3)], shade(hull, 1.35))
    p.line([(4.5, 4.5), (11.5, 4.5)], (226, 214, 190))
    _basket(p, -12, 4, (196, 204, 210))
    p.point(-11, 0.5, fish)
    return p.sprite()


def pasture() -> Sprite:
    """A sheep shed with a door and a window under a tiled roof, on grazing with a water trough
    standing in trampled mud, a sheep track worn from the shed door to it and sheep around."""
    p = Painter(30, 22, (15, 16))
    _track(p, [(-6.8, 2.2), (-4, 3.6), (-1, 4)], 0.9)
    _worn(p, -7, 2.6, 2.2)
    _mud(p, 1.2, 4, 3.6)
    wall, roof = (152, 124, 92), (110, 80, 60)
    top = p.box(-5, 0, 5, 4, wall, None)
    left, front, right, _ = top
    _footing(p, top, 4)
    _boards(p, top, wall, (1.5,))
    _plinth(p, top, wall, 4, 0.8)
    _door(p, left, front, (0.5, 0.8), 1.2, 4, shade(DOOR, 0.9))
    p.line([_on_wall(left, front, 0.5, 1.2), _on_wall(left, front, 0.8, 4)], shade(DOOR, 1.4))
    _window(p, front, right, (0.35, 0.65), 1, shade(WINDOW, 0.8))
    p.gable(top, 3, roof, wall)
    _tiles(p, top, 3, roof, (0.5,))
    p.hall(1, 3.5, 3.5, 1.2, 1, (122, 90, 60), WATER)
    for x, y, facing in ((11, -3, 1), (3, -2, -1), (8, 1, 1), (5, 4.5, -1), (-9, 5, -1)):
        _sheep(p, x, y, facing)
    return p.sprite()


def plantation(wall: RGB, roof: RGB) -> Sprite:
    """The small barn that stands on a field or grove: a plinth, a cross-braced door, a window
    with a sill and a tiled roof, with the ground trodden bare before its door and a path off,
    and sacks of the crop and a crate stood out on worn earth."""
    p = Painter(18, 16, (9, 11))
    _track(p, [(-2.6, 2.4), (-3.4, 4), (-3, 5.5)], 0.9)
    _worn(p, -2.8, 2.4, 2.4)
    _worn(p, 6, 3.8, 2.6)
    p.shadow(6, 2)
    top = p.box(0, 0, 5, 4, wall, None)
    left, front, right, _ = top
    _footing(p, top, 4)
    _plinth(p, top, wall, 4, 0.8)
    _door(p, left, front, (0.3, 0.7), 1.3, 4)
    brace = shade(DOOR, 1.5)
    p.line([_on_wall(left, front, 0.3, 1.3), _on_wall(left, front, 0.7, 4)], brace)
    p.line([_on_wall(left, front, 0.7, 1.3), _on_wall(left, front, 0.3, 4)], brace)
    _window(p, front, right, (0.35, 0.65), 1, shade(WINDOW, 0.8))
    p.gable(top, 3, roof, wall)
    _tiles(p, top, 3, roof, (0.5,))
    _crate(p, -6.5, 2.5, shade(wall, 0.9), 1.4)
    _sack(p, 7, 3, SACK, 2.8, 1.2)
    _sack(p, 5, 4.6, shade(SACK, 0.92), 2.8, 1.2)
    return p.sprite()


def beekeeper() -> Sprite:
    """Coiled straw hives on wooden stands in a meadow of flowers, the grass worn bare under
    each and a path trodden in among them, with bees about them and jars of honey on a plank."""
    p = Painter(28, 18, (14, 11))
    p.polygon([(-12, 0), (0, 6), (12, 0), (0, -6)], (128, 168, 84), (104, 142, 70))
    _track(p, [(-9, 4.6), (-4, 3.2), (-1.4, 1.2)], 0.9)
    flowers = ((240, 226, 120), (238, 238, 232), (214, 120, 160))
    for k, (x, y) in enumerate(((-9, 0), (-3, -3), (4, -5), (9, -2), (3, 4), (-8, 2), (8, 2.5))):
        p.point(x, y, flowers[k % 3])
    stand = (122, 90, 60)
    hives = ((0, -2), (-6, 1), (6, 1), (0, 4))
    for x, y in hives:
        _worn(p, x + 0.2, y + 0.4, 3.2)
    for x, y in hives:
        p.point(x - 2, y, stand)
        p.point(x + 2, y, stand)
        p.line([(x - 3, y - 0.5), (x + 3, y - 0.5)], shade(stand, 1.2))
        p.ellipse(x - 2.5, y - 6.5, x + 2.5, y - 0.5, HIVE)
        coil = shade(HIVE, 0.72)
        p.line([(x - 2, y - 2.5), (x + 2, y - 2.5)], coil)
        p.line([(x - 2, y - 4.3), (x + 2, y - 4.3)], coil)
        p.point(x - 1, y - 5.5, shade(HIVE, 1.15))
        p.point(x, y - 1.5, shade(HIVE, 0.4))
    for x, y in ((-3, -7), (4, -9), (9, -4), (-9, -3)):
        p.point(x, y, (54, 44, 30))
    honey = (214, 150, 40)
    p.hall(9.8, 4.4, 5, 1.4, 0.8, shade(stand, 1.2), shade(stand, 1.4))
    for x, y in ((8.6, 4.4), (11, 3.2)):
        p.rect(x - 1, y - 2, x + 1, y, honey)
        p.point(x - 0.5, y - 1.5, shade(honey, 1.3))
        p.line([(x - 1, y - 2.5), (x + 1, y - 2.5)], SILL)
    return p.sprite()


def incense_grove() -> Sprite:
    """Gnarled resin trees on stony pale ground, each rooted in a darker hollow, their bark cut
    and weeping tears of resin that drip on the ground, and a basket of gathered resin at the
    end of a trodden path."""
    p = Painter(28, 20, (14, 13))
    ground = (190, 170, 130)
    p.polygon([(-12, 0), (0, 6), (12, 0), (0, -6)], ground, shade(ground, 0.84))
    _track(p, [(-6, 1.4), (0, 4.6), (6.4, 4.4)], 0.9, shade(ground, 1.08))
    for x, y in ((-10, 0), (9, -2), (-3, 4), (3, -4)):
        p.point(x, y, shade(STONE, 0.9))
        p.point(x + 1, y, shade(STONE, 0.66))
    leaf = (108, 124, 76)
    resin = (236, 222, 170)
    for x, y in ((6, -1), (-6, 0), (0, 3)):
        p.blob(x - 2.4, y - 0.8, x + 3, y + 1.2, shade(ground, 0.76))
        p.line([(x - 1.4, y + 0.4), (x + 0.4, y)], shade(TRUNK, 0.8))
        p.line([(x + 1, y), (x + 2.4, y + 0.6)], shade(TRUNK, 0.6))
        p.line([(x + 1, y), (x, y - 3), (x + 2, y - 5)], shade(TRUNK, 0.7))
        p.line([(x, y), (x - 1, y - 3), (x + 1, y - 5)], TRUNK)
        p.line([(x - 1, y - 3), (x - 3, y - 6)], TRUNK)
        p.line([(x + 1, y - 5), (x + 3, y - 7)], TRUNK)
        p.blob(x - 5, y - 8, x - 1, y - 5, shade(leaf, 0.8))
        p.blob(x - 3, y - 9, x + 3, y - 5, leaf)
        p.blob(x - 1, y - 10, x + 4, y - 7, shade(leaf, 1.2))
        p.point(x - 2, y - 8, (200, 160, 210))
        p.point(x, y - 2, resin)
        p.point(x - 1, y - 4, resin)
        p.point(x - 1.6, y + 0.8, resin)
    _basket(p, 8, 4.5, resin)
    return p.sprite()


def potato_farm() -> Sprite:
    """A low fieldstone cottage on a plinth, with a window, a door and a stone chimney under
    ridged thatch, rows of potato plants before it and sacks and a crate of potatoes by its
    door."""
    p = Painter(26, 24, (12, 16))
    _track(p, [(-3.4, 1.8), (-1.8, 4.4), (0.4, 8)])
    _worn(p, -3.4, 2.2, 2.4)
    _worn(p, 8, 2.6, 4)
    p.shadow(10, 3)
    wall = (172, 166, 150)
    tuber = (150, 104, 58)
    top = p.hall(-2, -1, 11, 6, 4, wall, None)
    left, front, right, _ = top
    _footing(p, top, 4)
    for t, drop in ((0.15, 1), (0.4, 3), (0.85, 2), (0.6, 0.5)):
        p.point(*_on_wall(left, front, t, drop), shade(wall, 0.75))
    _plinth(p, top, wall, 4, 0.8)
    _window(p, left, front, (0.2, 0.32), 1)
    _door(p, left, front, (0.55, 0.72), 1.2, 4, (80, 62, 48))
    _window(p, front, right, (0.4, 0.62), 1, shade(WINDOW, 0.8))
    p.gable(top, 4, THATCH, wall)
    _tiles(p, top, 4, THATCH, (0.25, 0.5, 0.75))
    _chimney(p, top, 4, 0.7, 2, STONE)
    soil = (112, 86, 58)
    p.polygon([(-10, 1.5), (-3, 5), (-6, 6.5), (-13, 3)], soil, shade(soil, 0.7))
    for row in range(2):
        x0, y0 = -11 - 1.2 * row, 2.5 + 0.6 * row
        p.line([(x0 - 0.4, y0 - 0.2), (x0 + 7.4, y0 + 3.7)], shade(soil, 1.3))
        p.line([(x0 + 0.4, y0 + 0.9), (x0 + 7.6, y0 + 4.5)], shade(soil, 0.68))
        for k in range(4):
            p.blob(x0 + 1.8 * k, y0 + 0.9 * k - 1, x0 + 1.8 * k + 1.6, y0 + 0.9 * k, (84, 132, 56))
    sack = SACK
    for x, y in ((8, 0), (5, 3), (9, 3.5)):
        _sack(p, x, y, sack)
    _crate(p, 12, 3, tuber, 1.4)
    for x, y in ((3, 5), (4, 6), (6, 6), (-7.6, 6), (-4.4, 5.4)):
        p.point(x, y, tuber)
    return p.sprite()


def cattle_ranch() -> Sprite:
    """A boarded cattle barn on a plinth with cross-braced doors, a hayloft hatch and a tiled
    roof, a haystack, and a fenced paddock with a water trough and brown and black-and-white cows
    in it."""
    p = Painter(34, 26, (16, 18))
    p.blob(-2, -0.4, 14, 5.6, WORN)
    p.blob(1, 1, 10, 4.6, shade(EARTH, 0.92))
    _mud(p, 10.4, 0.6, 3.6)
    _track(p, [(-10.2, 1.2), (-6.6, 3), (-2, 3.4)])
    _worn(p, -10.4, 1.4, 2.8)
    _strewn(p, -12.5, 4, 3.6, (220, 186, 96), 7)
    p.shadow(12, 3.5)
    wood = (136, 86, 56)
    top = p.box(-8, -2, 6, 6, wood, None)
    left, front, right, _ = top
    _footing(p, top, 6)
    _boards(p, top, wood, (1.5, 3, 4.5))
    _plinth(p, top, wood, 6)
    trim = (236, 230, 214)
    _patch(p, left, front, (0.45, 0.8), (1.5, 6), trim)
    _patch(p, left, front, (0.52, 0.73), (2.5, 6), shade(wood, 0.7))
    p.line([_on_wall(left, front, 0.52, 2.5), _on_wall(left, front, 0.73, 6)], trim)
    p.line([_on_wall(left, front, 0.73, 2.5), _on_wall(left, front, 0.52, 6)], trim)
    _patch(p, front, right, (0.35, 0.65), (1.4, 3), (60, 40, 30))
    p.point(*_on_wall(front, right, 0.45, 2), (220, 186, 96))
    p.gable(top, 4, (86, 74, 70), wood)
    _tiles(p, top, 4, (86, 74, 70))
    _heap(p, -12.5, 3.5, 2.6, 3.5, (220, 186, 96))
    left, back, right, front = _paddock(6, 2, 9)
    _fence(p, left, back)
    _fence(p, back, right)
    p.hall(10.5, 0, 4, 1.4, 1.2, (122, 90, 60), WATER)
    _cow(p, 6, 0, (138, 84, 50), (96, 58, 36))
    _cow(p, 10, 4, (240, 238, 232), (40, 38, 42))
    _cow(p, 3, 4, (40, 38, 42), (240, 238, 232))
    _fence(p, left, front)
    _fence(p, front, right)
    return p.sprite()


def _pig(p: Painter, x: float, y: float) -> None:
    skin: RGB = (232, 166, 160)
    p.line([(x - 2, y - 1), (x - 2, y)], shade(skin, 0.6))
    p.line([(x + 1, y - 1), (x + 1, y)], shade(skin, 0.6))
    p.ellipse(x - 3, y - 3.5, x + 2.5, y - 0.5, skin)
    p.rect(x - 4.5, y - 3, x - 3, y - 1.5, shade(skin, 0.9))
    p.point(x - 4.5, y - 2, shade(skin, 0.6))
    p.point(x + 2.5, y - 3, shade(skin, 0.75))


def _hen(p: Painter, x: float, y: float, feathers: RGB) -> None:
    p.ellipse(x - 1.5, y - 2, x + 1, y, feathers)
    p.point(x - 1.5, y - 2.5, feathers)
    p.point(x - 2, y - 2, (222, 160, 60))
    p.point(x - 1.5, y - 3, (200, 50, 44))


def pig_farm() -> Sprite:
    """A low boarded sty with a tiled roof over a plinth, a heap of straw, and a fenced pen of
    trampled mud with a feeding trough and pink pigs in it."""
    p = Painter(34, 24, (16, 17))
    p.blob(-2, -0.4, 14, 5.6, WORN)
    _mud(p, 7, 1.4, 6)
    _track(p, [(-10.2, 1.2), (-6.6, 3), (-2, 3.4)])
    _worn(p, -10.4, 1.4, 2.8)
    _strewn(p, -12.5, 4, 3.6, (220, 186, 96), 7)
    p.shadow(12, 3)
    wood = (150, 104, 72)
    top = p.box(-8, -2, 6, 4.4, wood, None)
    left, front, right, _ = top
    _footing(p, top, 4.4)
    _boards(p, top, wood, (1.5, 3))
    _plinth(p, top, wood, 4.4)
    _patch(p, left, front, (0.5, 0.75), (1.4, 4.4), (60, 40, 30))
    p.gable(top, 3, (150, 80, 60), wood)
    _tiles(p, top, 3, (150, 80, 60))
    _heap(p, -12.5, 3.5, 2.4, 3, (220, 186, 96))
    left, back, right, front = _paddock(6, 2, 9)
    _fence(p, left, back)
    _fence(p, back, right)
    p.hall(9, 0.2, 4, 1.4, 1.2, (122, 90, 60), (150, 120, 70))
    _pig(p, 4, 1)
    _pig(p, 9, 4)
    _pig(p, 2, 4.5)
    _fence(p, left, front)
    _fence(p, front, right)
    return p.sprite()


def chicken_farm() -> Sprite:
    """A whitewashed henhouse on stilts with a ramp up to its hatch and a tiled roof, nesting
    boxes along its side, and a fenced yard of scratched earth with white and brown hens and a
    rooster pecking about, grain strewn among them."""
    p = Painter(32, 24, (15, 17))
    p.blob(-2, -0.4, 13, 5.6, WORN)
    p.blob(0, 0.6, 12, 4.8, shade(EARTH, 0.95))
    _track(p, [(-10.2, 1.2), (-6.6, 3), (-2, 3.4)])
    _strewn(p, 6, 2.5, 4, (226, 196, 110), 9)
    p.shadow(10, 3)
    top = p.box(-7, -1, 5, 4, WHITEWASH, None)
    left, front, right, _ = top
    _footing(p, top, 4)
    _boards(p, top, WHITEWASH, (2,))
    _patch(p, left, front, (0.55, 0.75), (2, 4), (70, 50, 36))
    p.line([_on_wall(left, front, 0.65, 4), (-1.6, 4.6)], (150, 110, 70))
    _patch(p, front, right, (0.2, 0.8), (2.6, 3.4), (150, 110, 70))
    p.gable(top, 3, (150, 70, 52), WHITEWASH)
    _tiles(p, top, 3, (150, 70, 52))
    left, back, right, front = _paddock(6, 2, 8)
    _fence(p, left, back)
    _fence(p, back, right)
    for x, y, feathers in (
        (3, 1, (244, 240, 230)),
        (7, 3, (176, 108, 60)),
        (9, 0.5, (244, 240, 230)),
        (5, 4.5, (244, 240, 230)),
        (10, 3.6, (120, 70, 40)),
    ):
        _hen(p, x, y, feathers)
    _fence(p, left, front)
    _fence(p, front, right)
    return p.sprite()


def flax_field() -> Sprite:
    """A small whitewashed cottage on a plinth with a door, windows with sills, a tiled roof and
    a chimney, a strip of blue-flowered flax before it, and sheaves of flax stood up in stooks to
    dry beside it."""
    p = Painter(28, 24, (13, 16))
    stubble = (178, 168, 106)
    p.polygon([(-2.5, -0.5), (8, -5.5), (15.5, -1.5), (5, 4)], stubble, shade(stubble, 0.85))
    for k in range(1, 4):
        u, v = 7.5 * k / 4, 3.75 * k / 4
        p.line([(-2.5 + u, -0.5 + v), (8 + u, -5.5 + v)], shade(stubble, 0.8))
    _track(p, [(-7.4, 1.4), (-5.6, 3), (-2, 4.6), (1, 8)])
    _worn(p, -7.6, 1.8, 2.2)
    p.shadow(10, 3)
    roof = (128, 96, 70)
    top = p.box(-6, -1, 5, 4, WHITEWASH, None)
    left, front, right, _ = top
    _footing(p, top, 4)
    _plinth(p, top, WHITEWASH, 4, 0.8)
    _window(p, left, front, (0.18, 0.4), 1)
    _door(p, left, front, (0.6, 0.8), 1.2, 4, (92, 70, 52))
    _window(p, front, right, (0.4, 0.65), 1, shade(WINDOW, 0.8))
    p.gable(top, 3, roof, WHITEWASH)
    _tiles(p, top, 3, roof, (0.5,))
    _chimney(p, top, 3, 0.75, 1.2)
    blue = (110, 140, 230)
    p.polygon([(-11, 3), (-3, 7), (-5, 8), (-13, 4)], (88, 128, 84), (66, 100, 60))
    for k in range(5):
        p.point(-11.5 + 1.8 * k, 3.2 + 0.9 * k, blue)
        p.point(-12.2 + 1.8 * k, 4.6 + 0.9 * k, shade(blue, 1.15))
    straw = (204, 188, 120)
    stooks = [(1 + 4 * col + 2 * row, -1 + 4 * row - col) for row in range(2) for col in range(3)]
    for x, y in sorted(stooks, key=lambda q: q[1]):
        p.blob(x - 1.6, y - 0.2, x + 2.8, y + 1.6, shade(stubble, 0.62))
        p.polygon([(x - 2, y), (x, y - 6), (x, y + 1)], straw, shade(straw, 0.6))
        p.polygon([(x, y - 6), (x + 2, y), (x, y + 1)], shade(straw, 0.78), shade(straw, 0.55))
        p.line([(x - 1, y - 3), (x + 1, y - 3)], (128, 104, 60))
        p.point(x - 1, y - 6, shade(straw, 1.15))
    for x, y in ((2, 6), (11, 6)):
        p.point(x, y, blue)
    return p.sprite()


def hop_garden() -> Sprite:
    """Tall poles strung with wires and green hop bines hung with cones, a round brick oast
    house with a tiled kiln roof, a door and a white cowl, a boarded store beside it, and hop
    pockets and a basket of picked hops."""
    p = Painter(34, 40, (17, 30))
    pole, bine = (122, 96, 66), (88, 152, 58)
    cone = (180, 214, 112)
    for r in range(2):
        poles = [(-11 + 4 * k - 4 * r, -4 + 2 * k + 2 * r) for k in range(3)]
        tops = [(x, y - 16) for x, y in poles]
        for (x, y), (tx, ty) in zip(poles, tops, strict=True):
            p.blob(x - 2, y - 0.6, x + 2, y + 1.2, (98, 80, 54))
            p.blob(x - 1.2, y - 0.6, x + 1, y + 0.4, (122, 100, 66))
            p.line([(x, y), (tx, ty)], pole)
            p.line([(x - 1, y - 2), (x - 1, y - 12)], bine)
            p.line([(x + 1, y - 1), (x + 1, y - 14)], shade(bine, 0.85))
            for k in range(4):
                p.point(x + (k % 2) * 2 - 1, y - 3 - 3 * k, cone)
        p.line(tops, (70, 64, 58))
        for (x0, y0), (x1, y1) in itertools.pairwise(tops):
            p.line([(x0 + 1, y0 + 2), ((x0 + x1) / 2, (y0 + y1) / 2 + 3), (x1 - 1, y1 + 2)], bine)
            p.point((x0 + x1) / 2, (y0 + y1) / 2 + 4, cone)
    _worn(p, 6.6, 3.6, 5)
    _track(p, [(2, 5.4), (1.2, 7.6), (0.6, 10.5)])
    _worn(p, 13.4, 5.6, 2.6)
    oast = (176, 92, 64)
    p.blob(5.2, -1.6, 15.2, 2.6, FOOTING)
    _turned(p, 10, 0, [(0, 4), (9, 4)], oast)
    for h in (3, 6):
        _arc(p, 10, -h, 4, shade(oast, 0.8))
    p.rect(8, -3, 9, 1.6, (60, 40, 34))
    kiln = (110, 76, 64)
    _turned(p, 10, -9, [(0, 4.5), (8, 0.5)], kiln)
    for h in (2.5, 5):
        _arc(p, 10, -9 - h, 4.5 - 4 * h / 8, shade(kiln, 0.75))
    cowl = (244, 242, 236)
    p.polygon([(9, -17), (9, -20), (11, -22), (12, -19), (11, -17)], cowl, shade(cowl, 0.6))
    p.line([(12, -21), (14, -22)], cowl)
    p.point(11, -19, shade(cowl, 0.45))
    store = (228, 222, 206)
    stow = p.box(4, 4, 4, 4, store, None)
    left, front, right, _ = stow
    _footing(p, stow, 4)
    _plinth(p, stow, store, 4, 0.8)
    _door(p, left, front, (0.3, 0.6), 1.3, 4)
    _window(p, front, right, (0.35, 0.65), 1, shade(WINDOW, 0.8))
    p.gable(stow, 3, (112, 82, 66), store)
    _tiles(p, stow, 3, (112, 82, 66), (0.5,))
    pocket = (214, 200, 168)
    for x, y in ((12, 5.5), (14.5, 4.5)):
        p.ellipse(x - 1.2, y - 5, x + 1.2, y, pocket)
        p.line([(x - 1, y - 2.5), (x + 1, y - 2.5)], shade(pocket, 0.65))
    _basket(p, -4, 6, cone)
    return p.sprite()
