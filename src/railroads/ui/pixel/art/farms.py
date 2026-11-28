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
    Top,
    shade,
)
from railroads.ui.pixel.art.parts import (
    BEAST,
    DOOR,
    METRE,
    SILL,
    _band,
    _bands,
    _cow,
    _fence,
    _heap,
    _hen,
    _legs,
    _on_wall,
    _paddock,
    _patch,
    _pig,
    _ridge,
    _tiles,
    _turned,
    _window,
)

CRATE: RGB = (156, 108, 62)
SACK: RGB = (192, 162, 112)
WICKER: RGB = (176, 132, 76)
WOOL: RGB = (240, 236, 224)
WATER: RGB = (92, 148, 196)
EARTH: RGB = (146, 124, 84)
WORN: RGB = (120, 128, 70)
MUD: RGB = (96, 76, 54)
FOOTING: RGB = (62, 70, 40)

_SCATTER: tuple[Point, ...] = (
    (-0.9, 0.2),
    (0.7, -0.4),
    (-0.2, 0.7),
    (1.0, 0.5),
    (-0.6, -0.6),
    (0.3, 0.1),
    (-1.1, -0.2),
)


def _footing(p: Painter, top: Top, height: float, colour: RGB = FOOTING, reach: float = 1) -> None:
    """A dark damp strip on the ground where both walls of a block `height` high stand in it,
    `reach` times as wide as round a big block."""
    left, front, right, _ = top
    (lx, ly), (fx, fy), (rx, ry) = ((x, y + height) for x, y in (left, front, right))
    out, low, deep = 0.8 * reach, 0.6 * reach, 1.1 * reach
    p.polygon(
        [
            (lx - out, ly),
            (fx, fy),
            (rx + out, ry),
            (rx + out, ry + low),
            (fx, fy + deep),
            (lx - out, ly + low),
        ],
        colour,
    )


def _stack(p: Painter, x: float, y: float, half: float, tall: float, colour: RGB) -> None:
    """A rounded haystack `half` across and `tall` high standing on (x, y), lit on the left."""
    p.blob(x - half, y - tall, x + half, y + half * 0.3, shade(colour, 0.78))
    p.blob(x - half, y - tall, x + half * 0.35, y, colour)
    p.blob(x - half * 0.7, y - tall, x, y - tall * 0.6, shade(colour, 1.12))


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


def _strewn(p: Painter, x: float, y: float, half: float, colour: RGB, count: int = 5) -> None:
    """`count` bits of `colour` dropped on the ground round (x, y), up to `half` away."""
    for k, (u, v) in enumerate(_SCATTER[:count]):
        p.point(x + u * half, y + v * half / 2, shade(colour, 0.85 if k % 2 else 1))


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
        _bands(p, left, front, right, drop, shade(wall, 0.84), shade(wall, 0.66))


def _crate(p: Painter, x: float, y: float, load: RGB, half: float = 0.6 * METRE) -> None:
    """A slatted wooden crate standing on (x, y), filled to its brim with `load`."""
    left, front, _, _ = p.box(x, y, half, half, CRATE, load)
    _band(p, left, front, half / 2, shade(CRATE, 0.7))
    p.point(x - half / 2, y - half - 0.2, shade(load, 1.3))


def _sack(
    p: Painter,
    x: float,
    y: float,
    fill: RGB = SACK,
    tall: float = 1.2 * METRE,
    half: float = 0.4 * METRE,
) -> None:
    """A full sack standing on (x, y), tied off at its neck."""
    p.ellipse(x - half, y - tall, x + half, y, fill)
    p.line(
        [(x - half * 0.4, y - tall * 0.75), (x + half * 0.4, y - tall * 0.75)], shade(fill, 0.55)
    )
    p.point(x, y - tall, shade(fill, 0.5))


def _basket(p: Painter, x: float, y: float, load: RGB) -> None:
    """A wicker basket a metre across standing on (x, y), heaped with `load`."""
    h = 0.5 * METRE
    p.blob(x - h, y - 1.75 * h, x + h, y - 0.75 * h, load)
    p.polygon(
        [(x - h, y - 1.25 * h), (x + h, y - 1.25 * h), (x + 0.75 * h, y), (x - 0.75 * h, y)], WICKER
    )
    p.line([(x - 0.9 * h, y - 0.6 * h), (x + 0.9 * h, y - 0.6 * h)], shade(WICKER, 0.7))
    p.point(x - 0.5 * h, y - 1.75 * h, shade(load, 1.3))


def _sheep(p: Painter, x: float, y: float, facing: int = 1, metre: float = METRE) -> None:
    """A sheep grazing at (x, y), 1.3 m long: a thick fleece, a dark face and dark legs."""
    m, f = metre, facing
    face: RGB = (54, 48, 46)
    half, high, belly = 0.6 * m, 0.85 * m, 0.35 * m
    p.blob(x - 1.2 * half, y - 0.18 * m, x + 1.2 * half, y + 0.22 * m, (70, 98, 44))
    _legs(p, x, y, half * 0.85, belly, m, face)
    p.blob(x - half, y - high, x + half, y - belly + 0.08 * m, WOOL)
    p.blob(
        x - 0.8 * half, y - high + 0.02 * m, x + 0.4 * half, y - high + 0.3 * m, shade(WOOL, 1.06)
    )
    p.blob(
        x - 0.85 * half,
        y - belly - 0.2 * m,
        x + 0.85 * half,
        y - belly + 0.08 * m,
        shade(WOOL, 0.85),
    )
    hx = x + f * (half + 0.08 * m)
    p.blob(hx - 0.18 * m, y - high + 0.22 * m, hx + 0.18 * m, y - belly + 0.05 * m, face)
    p.blob(
        hx - f * 0.12 * m - 0.08 * m,
        y - high + 0.24 * m,
        hx - f * 0.12 * m + 0.08 * m,
        y - high + 0.34 * m,
        face,
    )


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
    top = p.box(-1, 0, 7, 4.5, barn, None)
    left, front, right, _ = top
    _footing(p, top, 4.5)
    _patch(p, left, front, (0, 1), (3.8, 4.5), shade(barn, 0.65))
    _patch(p, left, front, (0.3, 0.72), (1, 4.5), shade(barn, 0.6))
    trim = (232, 226, 210)
    door = [_on_wall(left, front, t, d) for t, d in ((0.3, 1), (0.72, 1), (0.72, 4.5), (0.3, 4.5))]
    p.line([door[0], door[1], door[2], door[3], door[0]], trim)
    p.line([door[0], door[2]], trim)
    p.line([door[1], door[3]], trim)
    _patch(p, front, right, (0.2, 0.4), (1.4, 2.6), shade(barn, 0.45))
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
    _crate(p, 5, 5.2, yellow)
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


def _boat(p: Painter, x: float, y: float, long: float, wide: float, hull: RGB, stripe: RGB) -> None:
    """A rowing boat `long` by `wide` metres drawn up at (x, y), its bow to the front right: a
    painted hull with a stripe under its gunwale, a dark planked inside, a thwart and two oars."""
    m = METRE
    h, w, d = long * m / 2, wide * m / 2, 0.7 * m

    def at(u: float, v: float) -> Point:
        return x + h * u + w * v, y + h * u / 2 - w * v / 2

    near = [at(-0.92, -0.55), at(-0.5, -0.95), at(0.2, -0.9), at(0.7, -0.5), at(1, 0)]
    far = [at(1, 0), at(0.7, 0.5), at(0.2, 0.9), at(-0.5, 0.95), at(-0.92, 0.55)]
    p.blob(x - h - 0.4, y - 0.2, x + h + 0.6, y + h / 2 + 0.6, (122, 110, 86))
    p.polygon(near + [(px, py + d) for px, py in reversed(near)], hull, shade(hull, 0.6))
    p.line(near, stripe)
    p.line([(px, py + 0.25 * m) for px, py in near], stripe)
    p.polygon(near + far, (110, 78, 52), shade(hull, 0.7))
    p.line([at(-0.1, -0.85), at(-0.1, 0.85)], (150, 112, 74))
    oar = (196, 170, 120)
    p.line([at(-0.6, -0.4), at(0.6, 0.3)], oar)
    p.line([at(-0.5, 0.4), at(0.7, -0.2)], oar)


def _fish(p: Painter, x: float, y: float, long: float) -> None:
    """A silver fish `long` units long hung by its tail from (x, y)."""
    silver = (204, 214, 220)
    p.ellipse(x - long * 0.18, y + long * 0.15, x + long * 0.18, y + long, silver)
    p.line([(x, y + long * 0.2), (x, y + long * 0.95)], shade(silver, 0.75))
    p.polygon([(x - long * 0.15, y), (x + long * 0.15, y), (x, y + long * 0.2)], shade(silver, 0.7))


def fishery() -> Sprite:
    """A fisherman's place on the shore: a boarded hut 8 by 6 m in weathered sea blue with white
    trim under a slate roof, a silver fish turning on its ridge, nets hung to dry between poles, a
    rack of silver fish drying, baskets of the catch, and a red-and-white rowing boat with its oars
    drawn up on the shingle by a little jetty running out into the water."""
    p = Painter(34, 26, (16, 17))
    m = METRE
    shingle = (198, 184, 146)
    p.polygon([(-2, 8), (8, 2.5), (17, -1.5), (17, 8)], shingle)
    p.polygon([(6, 8), (15, 1.2), (17, 0.6), (17, 8)], shade(shingle, 0.82))
    p.polygon([(9, 8), (17, 1.6), (17, 8)], WATER)
    p.line([(9.5, 7.6), (17, 1.5)], (226, 236, 238))
    p.line([(12.4, 6.4), (16, 3.6)], shade(WATER, 1.25))
    plank = (150, 118, 80)
    for k in range(4):
        jx, jy = 11.4 + 1.1 * k, 3.6 + 0.55 * k
        p.polygon(
            [(jx, jy), (jx + 1.0, jy + 0.5), (jx + 2.2, jy - 0.1), (jx + 1.2, jy - 0.6)],
            plank,
            shade(plank, 0.7),
        )
    for jx, jy in ((12.0, 3.4), (15.2, 5.0)):
        p.line([(jx, jy), (jx, jy + 1.4)], shade(plank, 0.6))
    _track(p, [(-8, 3.0), (-5.5, 4.6), (-1.5, 5.4), (2, 5)])
    _worn(p, -8, 3.2, 2.4)
    p.shadow(10, 3)
    pole, net = (92, 70, 48), (78, 70, 60)
    for x0, y0 in ((-15.2, 1.4),):
        a, b = (x0, y0), (x0 + 5.2, y0 - 2.6)
        for px, py in (a, b):
            p.line([(px, py), (px, py - 2.6 * m)], pole)
        top_a, top_b = (a[0], a[1] - 2.4 * m), (b[0], b[1] - 2.4 * m)
        p.line([top_a, top_b], shade(pole, 1.2))
        for k in range(9):
            t = (k + 0.5) / 9
            tx, ty = _along(top_a, top_b, t)
            sag = 1.3 * m * (1 - (2 * t - 1) ** 2) + 0.4 * m
            p.line([(tx, ty), (tx - 0.3, ty + sag)], net)
        for drop in (0.5 * m, 1.0 * m, 1.5 * m):
            pts = []
            for k in range(10):
                t = k / 9
                tx, ty = _along(top_a, top_b, t)
                sag = min(drop, 1.3 * m * (1 - (2 * t - 1) ** 2) + 0.4 * m)
                pts.append((tx - 0.3 * sag / (1.7 * m), ty + sag))
            p.line(pts, net)
        for k in range(5):
            cx, cy = _along(top_a, top_b, (k + 0.5) / 5)
            p.ellipse(cx - 0.2 * m, cy - 0.15 * m, cx + 0.2 * m, cy + 0.25 * m, (214, 170, 90))
    wall, trim, roof = (88, 126, 146), (236, 234, 226), (70, 76, 88)
    long, deep, high, rise = 8 * m, 6 * m, 3 * m, 2.6 * m
    top = p.hall(-4, -1, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.6)
    _boards(p, top, wall, (high / 3, high * 2 / 3))
    _plinth(p, top, wall, high, 0.4 * m)
    _patch(p, left, front, (0, 1), (0, 0.25 * m), trim)
    _patch(p, front, right, (0, 1), (0, 0.25 * m), shade(trim, 0.8))
    _window(p, left, front, (0.14, 0.32), 0.7 * m)
    _door(p, left, front, (0.56, 0.74), high - 2 * m, high, trim)
    _window(p, front, right, (0.4, 0.64), 0.7 * m, shade(WINDOW, 0.8))
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof, (0.5,))
    _chimney(p, top, rise, 0.25, 1.2, STONE)
    ridge_back, ridge_front = _ridge(top, rise)
    vx, vy = _along(ridge_back, ridge_front, 0.7)
    p.line([(vx, vy), (vx, vy - 1.2 * m)], (60, 58, 56))
    silver = (204, 214, 220)
    fx, fy = vx, vy - 1.3 * m
    p.ellipse(fx - 0.8 * m, fy - 0.25 * m, fx + 0.5 * m, fy + 0.25 * m, silver)
    p.polygon(
        [(fx + 0.4 * m, fy), (fx + 0.9 * m, fy - 0.35 * m), (fx + 0.9 * m, fy + 0.35 * m)], silver
    )
    rx, ry = 3.2, 0.6
    rack_end = (rx + 4.4, ry - 2.2)
    for px, py in ((rx, ry), rack_end):
        p.line([(px, py), (px, py - 2.4 * m)], pole)
    top_a, top_b = (rx, ry - 2.2 * m), (rack_end[0], rack_end[1] - 2.2 * m)
    p.line([top_a, top_b], shade(pole, 1.2))
    for k in range(7):
        hx, hy = _along(top_a, top_b, (k + 0.5) / 7)
        _fish(p, hx, hy, 1.6 * m)
    _boat(p, 8.2, 4.0, 7, 2.2, (240, 236, 226), (186, 52, 44))
    for bx, by in ((-11.6, 4.4), (-10.2, 5.2), (-10.8, 3.6)):
        _crate(p, bx, by, silver)
    return p.sprite()


def pasture() -> Sprite:
    """A sheep farm: a shepherd's long fieldstone house 12 by 6 m under thatch, his home with a
    chimney at one end and the sheep's shed with its wide boarded door at the other, bales of
    wool by the door, and beside it a fold walled in dry stone round a hay rack and a trough,
    where white sheep with dark faces stand, drawn twice their size (BEAST); the rest of the
    flock grazes the pasture round it (railroads.ui.pixel.ground_palette.GROVES)."""
    p = Painter(40, 30, (20, 21))
    m, b = METRE, BEAST
    x, y, long, deep, high, rise = 5.0, -2.5, 12 * m, 6 * m, 3 * m, 3.2 * m
    wall = (178, 172, 156)
    p.shadow(11, 3)
    top = p.hall(x, y, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.5)
    for k in range(18):
        t, drop = (0.5 + k * 0.618) % 1, 0.3 * m + (0.5 + k * 0.382) % 1 * (high - 0.6 * m)
        p.point(*_on_wall(left, front, t, drop), shade(wall, 0.78 if k % 2 else 1.12))
    _plinth(p, top, wall, high, 0.4 * m)
    _patch(p, left, front, (0.08, 0.4), (high - 2.4 * m, high), (122, 92, 64))
    p.line(
        [_on_wall(left, front, 0.24, high - 2.4 * m), _on_wall(left, front, 0.24, high)],
        (96, 72, 50),
    )
    _window(p, left, front, (0.56, 0.66), 0.8 * m)
    _door(p, left, front, (0.74, 0.84), high - 2 * m, high, shade(DOOR, 0.9))
    _window(p, left, front, (0.88, 0.96), 0.8 * m)
    _window(p, front, right, (0.35, 0.6), 0.8 * m, shade(WINDOW, 0.8))
    p.gable(top, rise, THATCH, wall)
    _tiles(p, top, rise, THATCH, (0.3, 0.6))
    _chimney(p, top, rise, 0.85, 1.6, STONE)
    for wx, wy in (
        (right[0] + 1.3, right[1] + high + 0.9),
        (right[0] + 2.5, right[1] + high + 0.3),
    ):
        p.ellipse(wx - 0.7 * m, wy - 1.1 * m, wx + 0.7 * m, wy, WOOL)
        p.line([(wx - 0.5 * m, wy - 0.55 * m), (wx + 0.5 * m, wy - 0.55 * m)], shade(WOOL, 0.7))
    fold = _paddock(-8.5, 3.2, 8.5)
    fl, fb, fr, fn = fold
    stone, low = (150, 146, 136), 0.8 * m
    _ground(p, fold, (128, 158, 78))
    p.soft([(-12, 1.5, -5, 5.5)], (140, 132, 96), 0.45, 1.2)
    _low_wall(p, fl, fb, low, stone)
    _low_wall(p, fb, fr, low, stone)
    hay = (214, 184, 100)
    sheep = (
        (-11.6, 2.6, 1),
        (-7.2, 1.6, -1),
        (-5.0, 3.2, -1),
        (-9.4, 4.0, 1),
        (-12.6, 4.4, 1),
        (-6.8, 4.9, -1),
        (-9.6, 6.0, 1),
    )
    things = [(sy, k) for k, (_, sy, _) in enumerate(sheep)] + [(0.9, -1), (4.6, -2)]
    for _, key in sorted(things):
        if key == -1:
            p.hall(-9.4, 0.9, 2.4 * m, 1 * m, 1.2 * m, (122, 90, 60), hay)
            p.blob(-10.6, -0.7, -8.2, 0.1, hay)
        elif key == -2:
            _trough(p, -3.6, 4.6, 2, WATER)
        else:
            sx, sy, facing = sheep[key]
            _sheep(p, sx, sy, facing, b)
    _low_wall(p, fl, fn, low, stone)
    _low_wall(p, fn, fr, low, stone)
    return p.sprite()


def _ground_of(corner: Point, high: float) -> Point:
    """Where the top corner of a wall `high` tall (Painter.hall) stands on the ground."""
    return corner[0], corner[1] + high


def _arches(
    p: Painter, a: Point, b: Point, spans: tuple[tuple[float, float], ...], top: float, high: float
) -> None:
    """Round-headed openings in the wall under the edge from `a` to `b`, one over each of `spans`,
    from `top` below the eaves to the ground `high` below them."""
    for t0, t1 in spans:
        _patch(p, a, b, (t0, t1), (top + 0.3 * METRE, high), (62, 44, 36))
        x, y = _on_wall(a, b, (t0 + t1) / 2, top + 0.4 * METRE)
        half = abs(_on_wall(a, b, t1, 0)[0] - _on_wall(a, b, t0, 0)[0]) / 2
        p.ellipse(x - half, y - 0.45 * METRE, x + half, y + 0.35 * METRE, (62, 44, 36))


def _bale(p: Painter, x: float, y: float, cloth: RGB, band: RGB) -> None:
    """A pressed bale 1.6 by 1 m wrapped in `cloth` and bound with `band`, standing at (x, y)."""
    m = METRE
    left, front, _, _ = p.hall(x, y, 1.6 * m, 1 * m, 1 * m, cloth, shade(cloth, 1.08))
    for t in (0.25, 0.75):
        p.line([_on_wall(left, front, t, 0), _on_wall(left, front, t, 1 * m)], band)


def _cask(p: Painter, x: float, y: float) -> None:
    """A wine cask 1.2 m long lying on its side at (x, y), its head to the front."""
    m = METRE
    wood, hoop = (138, 88, 52), (70, 60, 56)
    p.polygon(
        [(x - 0.8 * m, y - 1.2 * m), (x, y - 0.8 * m), (x, y), (x - 0.8 * m, y - 0.4 * m)],
        shade(wood, 0.8),
    )
    p.ellipse(x - 0.4 * m, y - 0.95 * m, x + 0.4 * m, y + 0.05 * m, wood)
    p.ellipse(x - 0.25 * m, y - 0.75 * m, x + 0.25 * m, y - 0.15 * m, shade(wood, 1.15))
    p.line([(x - 0.55 * m, y - 1.05 * m), (x - 0.55 * m, y - 0.3 * m)], hoop)


def cotton_field() -> Sprite:
    """A cotton gin house 12 by 7 m of whitewashed boards under a grey shingle roof, its wide
    door open, a loading beam over the loft door in its gable, pressed bales of cotton in burlap
    stacked by the door and a wagon heaped high with picked cotton; the cotton itself grows over
    the cell round it."""
    p = Painter(30, 24, (14, 16))
    m = METRE
    p.shadow(9, 3)
    wall, roof = (238, 234, 222), (138, 134, 128)
    long, deep, high, rise = 12 * m, 7 * m, 4.2 * m, 3 * m
    top = p.hall(1, -1.5, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.6)
    _boards(p, top, wall, (high / 3, high * 2 / 3))
    _plinth(p, top, wall, high, 0.4 * m)
    _patch(p, left, front, (0.36, 0.64), (high - 3.2 * m, high), (70, 56, 46))
    _patch(p, left, front, (0.36, 0.64), (high - 3.2 * m, high - 2.9 * m), (210, 206, 196))
    _window(p, left, front, (0.08, 0.2), 0.8 * m)
    _window(p, left, front, (0.8, 0.92), 0.8 * m)
    gx, gy = _along(front, right, 0.5)
    _patch(p, front, right, (0.38, 0.62), (0.3 * m, 1.6 * m), (70, 56, 46))
    p.line([(gx, gy - 0.4 * m), (gx + 1.2 * m, gy - 1.0 * m)], (96, 74, 52))
    p.line([(gx + 1.2 * m, gy - 1.0 * m), (gx + 1.2 * m, gy + 0.6 * m)], (60, 56, 52))
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof, (0.33, 0.66))
    burlap, band = (204, 182, 140), (110, 86, 60)
    bx, by = _ground_of(front, high)
    for dx, dy in ((0.8, 0.9), (2.2, 1.6), (0.9, 2.4), (2.4, 3.1)):
        _bale(p, bx + dx, by + dy, burlap, band)
    cotton = (248, 248, 244)
    wx, wy = -8.6, 3.2
    wagon = (128, 90, 58)
    p.hall(wx, wy, 3.6 * m, 1.8 * m, 0.7 * m, wagon, None)
    p.blob(wx - 2.2, wy - 2.6, wx + 1.6, wy - 0.5, cotton)
    p.blob(wx - 1.4, wy - 3.2, wx + 0.8, wy - 1.8, shade(cotton, 1.05))
    p.blob(wx - 2.0, wy - 1.4, wx - 0.6, wy - 0.6, shade(cotton, 0.9))
    for cx in (wx - 1.6, wx + 1.0):
        p.ellipse(cx - 0.5 * m, wy - 0.2, cx + 0.5 * m, wy + 0.6 * m, (70, 52, 38))
    for x, y in ((-5, 5.2), (-4, 4.4), (6.5, 5.4), (-11.5, 5.4)):
        p.point(x, y, cotton)
    return p.sprite()


def vineyard() -> Sprite:
    """A winegrower's house 11 by 7 m of pale stone under a wine-red roof, wine-red shutters at its
    windows, a vine trained over a pergola before its door hung with dark grapes, an arched
    cellar door in its gable, casks of wine lying by it and a wooden basket press; the vines
    themselves grow in rows over the cell round it."""
    p = Painter(30, 24, (14, 16))
    m = METRE
    p.shadow(9, 3)
    wall, roof = (234, 218, 190), (124, 40, 62)
    long, deep, high, rise = 11 * m, 7 * m, 4.6 * m, 2.6 * m
    top = p.hall(1, -1.5, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.6)
    for k in range(16):
        t, drop = (0.5 + k * 0.618) % 1, 0.4 * m + (0.5 + k * 0.382) % 1 * (high - 0.8 * m)
        p.point(*_on_wall(left, front, t, drop), shade(wall, 0.86))
    _plinth(p, top, wall, high, 0.4 * m)
    shutter = (122, 42, 62)
    for t in (0.12, 0.3, 0.7, 0.88):
        _window(p, left, front, (t - 0.04, t + 0.04), 0.8 * m)
        _patch(p, left, front, (t - 0.07, t - 0.045), (0.8 * m, 2.4 * m), shutter)
        _patch(p, left, front, (t + 0.045, t + 0.07), (0.8 * m, 2.4 * m), shutter)
    _door(p, left, front, (0.46, 0.56), high - 2.2 * m, high, (96, 64, 44))
    _arches(p, front, right, ((0.36, 0.64),), high - 2.4 * m, high)
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof, (0.3, 0.6))
    _chimney(p, top, rise, 0.25, 1.2, shade(wall, 0.9))
    gx, gy = _ground_of(_along(left, front, 0.51), high)
    post, leaf, grape = (110, 82, 56), (82, 134, 56), (88, 40, 92)
    for dx in (-1.6, 1.4):
        p.line([(gx + dx, gy + 1.6), (gx + dx, gy - 1.2)], post)
    p.blob(gx - 2.6, gy - 2.0, gx + 2.4, gy - 0.8, leaf)
    p.blob(gx - 2.0, gy - 2.4, gx + 1.2, gy - 1.6, shade(leaf, 1.15))
    for dx in (-1.6, -0.4, 0.8, 1.8):
        p.blob(gx + dx - 0.3, gy - 1.0, gx + dx + 0.3, gy - 0.2, grape)
    cx, cy = _ground_of(right, high)
    for dx, dy in ((0.8, 1.2), (2.0, 1.8), (1.4, 0.4)):
        _cask(p, cx + dx, cy + dy)
    px, py = -9.0, 3.4
    _turned(p, px, py, [(0, 0.8 * m), (1.1 * m, 0.8 * m)], (150, 106, 66))
    p.line([(px, py - 1.1 * m), (px, py - 2.2 * m)], (70, 66, 62))
    p.line([(px - 0.6 * m, py - 2.0 * m), (px + 0.6 * m, py - 2.0 * m)], (70, 66, 62))
    _basket(p, -6.6, 5.0, grape)
    return p.sprite()


def coffee_plantation() -> Sprite:
    """A coffee planter's house 12 by 7 m, washed the colour of milky coffee with dark coffee-brown
    window frames under a cherry-red tile
    roof, a shady veranda along its front, and before it a paved drying terrace where the red
    coffee cherries and brown beans lie spread in rows to dry, a rake on them, and sacks of coffee
    by the door; the coffee bushes grow over the cell round it."""
    p = Painter(32, 24, (16, 16))
    m = METRE
    terrace = (208, 198, 180)
    corners = _paddock(-8, 2.6, 8)
    _ground(p, corners, terrace)
    tl, tb, tr, tn = corners
    for k in range(7):
        t0, t1 = (k + 0.15) / 7, (k + 0.85) / 7
        a0, a1 = _along(tl, tb, t0), _along(tl, tb, t1)
        b0, b1 = _along(tn, tr, t0), _along(tn, tr, t1)
        colour = ((156, 50, 40), (112, 70, 46), (170, 70, 46))[k % 3]
        p.polygon([a0, a1, b1, b0], colour, shade(colour, 0.85))
    rx, ry = -6.4, 3.0
    p.line([(rx, ry), (rx + 1.8, ry - 0.9)], (150, 112, 70))
    p.line([(rx - 0.4, ry + 0.2), (rx + 0.4, ry - 0.2)], (110, 84, 56))
    p.shadow(9, 3)
    wall, roof, frame = (228, 202, 166), (168, 70, 50), (96, 60, 40)
    long, deep, high, rise = 12 * m, 7 * m, 3.8 * m, 2.6 * m
    top = p.hall(4, -2.6, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.6)
    _plinth(p, top, wall, high, 0.5 * m)
    for t in (0.12, 0.3, 0.7, 0.88):
        _patch(p, left, front, (t - 0.05, t + 0.05), (0.7 * m, 2.6 * m), frame)
        _patch(p, left, front, (t - 0.035, t + 0.035), (0.9 * m, 2.5 * m), shade(WINDOW, 0.9))
    _door(p, left, front, (0.46, 0.56), high - 2.4 * m, high, frame)
    _window(p, front, right, (0.35, 0.65), 0.8 * m, shade(WINDOW, 0.8))
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof, (0.35, 0.7))
    a, b = _ground_of(left, high), _ground_of(front, high)
    eave = 2.6 * m
    out = 1.8 * m
    shade_roof = shade(roof, 0.9)
    p.polygon(
        [
            (a[0], a[1] - eave),
            (b[0], b[1] - eave),
            (b[0] - out, b[1] - eave + out / 2),
            (a[0] - out, a[1] - eave + out / 2),
        ],
        shade_roof,
        shade(roof, 0.6),
    )
    for t in (0.02, 0.35, 0.65, 0.98):
        x, y = _along(a, b, t)
        p.line([(x - out, y + out / 2), (x - out, y + out / 2 - eave)], (240, 236, 226))
    sx, sy = _ground_of(right, high)
    for dx, dy in ((0.8, 1.0), (1.8, 1.6), (0.9, 2.2)):
        _sack(p, sx + dx, sy + dy, (176, 146, 104))
    return p.sprite()


def rubber_plantation() -> Sprite:
    """A rubber planter's bungalow 10 by 8 m raised on posts, of pale timber with wide eaves under
    a rusty iron roof, a ladder up to its door, and beside it an open drying shed under a palm
    thatch where cream sheets of rubber hang on poles to cure, with latex pails stood by it; the
    rubber trees grow in rows over the cell round it."""
    p = Painter(32, 26, (16, 18))
    m = METRE
    p.shadow(9, 3)
    stilts, wall, roof = 1.4 * m, (206, 176, 128), (150, 92, 62)
    long, deep, high, rise = 10 * m, 8 * m, 3.0 * m, 2.6 * m
    x, y = 4.5, -2.4
    top = p.hall(x, y - stilts, long, deep, high, wall, None)
    left, front, right, _ = top
    for corner in (left, front, right):
        gx, gy = _ground_of(corner, high + stilts)
        p.line([(gx, gy), (gx, gy - stilts)], (92, 70, 50))
    gx, gy = _ground_of(_along(left, front, 0.5), high + stilts)
    p.line([(gx, gy), (gx, gy - stilts)], (92, 70, 50))
    _boards(p, top, wall, (high / 2,))
    for t in (0.15, 0.32, 0.68, 0.85):
        _patch(p, left, front, (t - 0.05, t + 0.05), (0.6 * m, 1.9 * m), (74, 56, 40))
    _door(p, left, front, (0.46, 0.56), high - 2.2 * m, high, (90, 62, 42))
    lx, ly = _ground_of(_along(left, front, 0.51), high)
    p.line([(lx, ly), (lx - 1.0 * m, ly + stilts + 0.2)], (150, 112, 70))
    p.line([(lx + 0.7 * m, ly - 0.3 * m), (lx - 0.3 * m, ly + stilts - 0.1)], (150, 112, 70))
    _window(p, front, right, (0.3, 0.7), 0.6 * m, shade(WINDOW, 0.8))
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof, (0.3, 0.6))
    shed_x, shed_y = -8.6, 2.6
    thatch = (176, 150, 92)
    sheet = (238, 224, 180)
    reach, tall = 4.6, 3 * m
    for px, py in (
        (shed_x - reach, shed_y),
        (shed_x + reach, shed_y),
        (shed_x, shed_y + reach / 2),
    ):
        p.line([(px, py), (px, py - tall)], (96, 74, 50))
    for row in range(2):
        for k in range(4):
            hx = shed_x - 3.0 + 2.0 * k - 1.0 * row
            hy = shed_y - 0.6 + 1.0 * k * 0.5 + 0.9 * row
            p.line([(hx - 0.9, hy - tall + 0.6), (hx + 0.9, hy - tall + 0.1)], (110, 84, 56))
            p.polygon(
                [
                    (hx - 0.8, hy - tall + 0.6),
                    (hx + 0.8, hy - tall + 0.2),
                    (hx + 0.8, hy - 1.0),
                    (hx - 0.8, hy - 0.6),
                ],
                shade(sheet, 1 - 0.04 * k - 0.08 * row),
                shade(sheet, 0.78),
            )
    rx, ry = shed_x, shed_y - tall
    p.polygon(
        [
            (rx - reach - 0.6, ry),
            (rx, ry - reach / 2 - 0.3),
            (rx + reach + 0.6, ry),
            (rx, ry + reach / 2 + 0.3),
        ],
        thatch,
        shade(thatch, 0.7),
    )
    p.line([(rx - reach - 0.6, ry), (rx + reach + 0.6, ry)], shade(thatch, 1.15))
    for dx in (-1.2, -0.2):
        _turned(
            p,
            shed_x + reach + 1.6 + dx,
            shed_y + 2.2,
            [(0, 0.3 * m), (0.6 * m, 0.35 * m)],
            (176, 180, 184),
        )
    return p.sprite()


def orange_farm() -> Sprite:
    """A white finca 12 by 7 m under an orange tile roof, a loggia of three arches along its front
    and a little bell gable on its ridge, crates of oranges stacked before it, a picking ladder
    and a basket of oranges by the door; the orange trees grow in rows over the cell round it."""
    p = Painter(30, 24, (14, 16))
    m = METRE
    p.shadow(9, 3)
    wall, roof = (246, 236, 214), (214, 112, 52)
    orange = (244, 142, 36)
    long, deep, high, rise = 12 * m, 7 * m, 4.2 * m, 2.4 * m
    top = p.hall(1, -1.5, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.6)
    _plinth(
        p,
        top,
        wall,
        high,
        0.5 * m,
    )
    _arches(p, left, front, ((0.12, 0.28), (0.42, 0.58), (0.72, 0.88)), high - 3.0 * m, high)
    _window(p, front, right, (0.35, 0.65), 0.8 * m, shade(WINDOW, 0.8))
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof, (0.35, 0.7))
    ridge_back, ridge_front = _ridge(top, rise)
    bx, by = _along(ridge_back, ridge_front, 0.8)
    p.polygon(
        [
            (bx - 0.6 * m, by),
            (bx + 0.6 * m, by),
            (bx + 0.6 * m, by - 1.4 * m),
            (bx, by - 2.0 * m),
            (bx - 0.6 * m, by - 1.4 * m),
        ],
        wall,
        shade(wall, 0.7),
    )
    p.ellipse(bx - 0.25 * m, by - 1.3 * m, bx + 0.25 * m, by - 0.7 * m, (176, 136, 60))
    cx, cy = _ground_of(front, high)
    for dx, dy in ((1.0, 1.0), (2.3, 1.6), (1.1, 2.3), (2.4, 2.9)):
        _crate(p, cx + dx, cy + dy, orange)
    _crate(p, cx + 1.6, cy + 0.8, orange)
    ladder = (200, 164, 108)
    lx, ly = _ground_of(_along(front, right, 0.2), high)
    for side in (0, 0.6):
        p.line([(lx + side, ly - side / 2), (lx + side + 0.4, ly - side / 2 - 3.2 * m)], ladder)
    _basket(p, -6.0, 4.4, orange)
    for x, y in ((-4.4, 5.2), (-3.2, 4.6), (7.6, 5.2)):
        p.point(x, y, orange)
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
    """A low fieldstone cottage 9 by 6 m on a plinth, with a window, a door and a stone chimney
    under ridged thatch, rows of potato plants before it and sacks and a crate of potatoes by its
    door."""
    p = Painter(22, 18, (11, 12))
    m = METRE
    _track(p, [(-2.6, 1.6), (-1.4, 3.6), (0.4, 6.5)])
    _worn(p, -2.6, 1.8, 2)
    _worn(p, 5, 2.2, 3)
    p.shadow(7, 2.5)
    wall = (172, 166, 150)
    tuber = (150, 104, 58)
    long, deep, high, rise = 9 * m, 6 * m, 3.4 * m, 3.2 * m
    top = p.hall(-2, -1, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high)
    for t, drop in ((0.15, 0.7), (0.4, 2), (0.85, 1.4), (0.6, 0.4)):
        p.point(*_on_wall(left, front, t, drop), shade(wall, 0.75))
    _plinth(p, top, wall, high, 0.5 * m)
    _window(p, left, front, (0.18, 0.32), 0.7 * m)
    _door(p, left, front, (0.55, 0.72), high - 2 * m, high, (80, 62, 48))
    _window(p, front, right, (0.4, 0.62), 0.7 * m, shade(WINDOW, 0.8))
    p.gable(top, rise, THATCH, wall)
    _tiles(p, top, rise, THATCH, (0.25, 0.5, 0.75))
    _chimney(p, top, rise, 0.7, 1.4, STONE)
    soil = (112, 86, 58)
    p.polygon([(-9, 1.5), (-3.5, 4.2), (-5.8, 5.4), (-11.3, 2.7)], soil, shade(soil, 0.7))
    for row in range(2):
        x0, y0 = -10 - 1.0 * row, 2.4 + 0.5 * row
        p.line([(x0 - 0.3, y0 - 0.2), (x0 + 5.8, y0 + 2.9)], shade(soil, 1.3))
        p.line([(x0 + 0.3, y0 + 0.7), (x0 + 6, y0 + 3.5)], shade(soil, 0.68))
        for k in range(4):
            px, py = x0 + 1.4 * k, y0 + 0.7 * k
            p.blob(px, py - 0.8, px + 1.2, py, (84, 132, 56))
    for x, y in ((4.6, 0.6), (3.2, 2.4), (5.6, 2.6)):
        _sack(p, x, y, SACK)
    _crate(p, 7.6, 2.6, tuber)
    for x, y in ((2, 4), (2.8, 4.6), (4, 4.4), (-6.6, 5.4), (-4, 4.8)):
        p.point(x, y, tuber)
    return p.sprite()


def rice_paddy() -> Sprite:
    """A farmhouse of the rice lands 10 by 6 m: white plaster framed in dark timber on a stone
    plinth, a low wooden veranda along its front, a steep, thick thatched roof with a dark ridge,
    and sheaves of rice hung to dry on a bamboo rack beside it, the paddies themselves flooded
    over the cell round it (railroads.ui.pixel.world)."""
    p = Painter(28, 22, (13, 15))
    m = METRE
    p.shadow(7, 2.5)
    wall, plaster = (92, 66, 46), (232, 222, 196)
    long, deep, high, rise = 10 * m, 6 * m, 2.8 * m, 3.8 * m
    top = p.hall(0, -1, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.6)
    for t0, t1 in ((0.06, 0.3), (0.62, 0.94)):
        _patch(p, left, front, (t0, t1), (0.3 * m, high - 0.5 * m), plaster)
    _patch(p, left, front, (0.38, 0.54), (0.5 * m, high), (58, 42, 32))
    _patch(p, front, right, (0.12, 0.88), (0.3 * m, high - 0.5 * m), shade(plaster, 0.8))
    deck = (150, 112, 72)
    a, b = (left[0], left[1] + high), (front[0], front[1] + high)
    out = 0.9 * m
    p.polygon(
        [a, b, (b[0] - out, b[1] + out / 2), (a[0] - out, a[1] + out / 2)], deck, shade(deck, 0.6)
    )
    p.gable(top, rise, THATCH, wall)
    _tiles(p, top, rise, THATCH, (0.3, 0.6))
    ridge_back, ridge_front = _ridge(top, rise)
    p.line([ridge_back, ridge_front], (70, 58, 44))
    p.line(
        [(ridge_back[0], ridge_back[1] + 0.3), (ridge_front[0], ridge_front[1] + 0.3)], (70, 58, 44)
    )
    pole, straw = (150, 134, 84), (214, 186, 104)
    gx, gy = right[0] + 1.4, right[1] + high + 1.6
    ends = ((gx, gy), (gx + 4 * m, gy - 2 * m))
    for x, y in ends:
        p.line([(x, y), (x, y - 2 * m)], pole)
    p.line([(ends[0][0], ends[0][1] - 1.8 * m), (ends[1][0], ends[1][1] - 1.8 * m)], pole)
    for k in range(7):
        x, y = _along(
            (ends[0][0], ends[0][1] - 1.8 * m), (ends[1][0], ends[1][1] - 1.8 * m), (k + 0.5) / 7
        )
        p.line([(x, y), (x, y + 1.2 * m)], straw if k % 2 else shade(straw, 0.82))
    for x, y in ((-6.4, 3.4), (-5.2, 4.2)):
        p.ellipse(x - 0.6 * m, y - 0.8 * m, x + 0.6 * m, y, WICKER)
        p.blob(x - 0.4 * m, y - 1.0 * m, x + 0.4 * m, y - 0.6 * m, straw)
    return p.sprite()


def _farmhouse(p: Painter, x: float, y: float) -> Top:
    """A small dark-timbered farmhouse of East Asia 9 by 6 m standing on (x, y), with plastered
    panels and a door under a steep thatched roof."""
    m = METRE
    wall, high, rise = (112, 82, 58), 2.8 * m, 3.4 * m
    top = p.hall(x, y, 9 * m, 6 * m, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high)
    plaster = (226, 214, 186)
    _patch(p, left, front, (0.1, 0.4), (0.4 * m, high - 0.6 * m), plaster)
    _patch(p, left, front, (0.52, 0.72), (high - 2 * m, high), (58, 42, 32))
    _patch(p, front, right, (0.3, 0.7), (0.4 * m, high - 0.8 * m), shade(plaster, 0.78))
    p.gable(top, rise, THATCH, wall)
    _tiles(p, top, rise, THATCH, (0.35, 0.7))
    return top


def soy_field() -> Sprite:
    """Rows of bushy soybean plants with pale pods behind a small thatched farmhouse, and sacks
    of beans at its door."""
    p = Painter(30, 24, (14, 16))
    soil = (126, 100, 66)
    p.polygon([(-13, 1), (-2, 6.5), (4, 3.5), (-7, -2)], soil, shade(soil, 0.7))
    leaf = (92, 150, 58)
    for row in range(4):
        x0, y0 = -11.5 + 1.6 * row, 1 - 0.8 * row
        for k in range(5):
            x, y = x0 + 2 * k, y0 + k
            p.blob(x - 1, y - 1.6, x + 1, y, leaf if (k + row) % 2 else shade(leaf, 1.15))
            p.point(x, y - 0.6, (214, 206, 140))
    p.shadow(7, 2.5)
    _farmhouse(p, 5, -2)
    for x, y in ((8, 4), (10.5, 3)):
        _sack(p, x, y, (210, 190, 130))
    return p.sprite()


def tea_plantation() -> Sprite:
    """Rounded rows of clipped tea bushes on a slope, a picker's basket among them and a small
    thatched drying house above."""
    p = Painter(30, 24, (14, 16))
    p.shadow(6, 2)
    _farmhouse(p, 6, -5)
    bush = (62, 128, 56)
    for row in range(3):
        x0, y0 = -13 + 2 * row, 0 + 1.6 * row
        for k in range(4):
            x, y = x0 + 3.2 * k, y0 + 1.6 * k - 1.2 * row
            p.blob(x - 1.8, y - 2.2, x + 1.8, y + 0.4, bush)
            p.blob(x - 1.2, y - 2.2, x + 0.6, y - 1.2, shade(bush, 1.3))
    _basket(p, 3, 6, (120, 170, 80))
    return p.sprite()


def _along(a: Point, b: Point, t: float) -> Point:
    """The point `t` of the way from `a` to `b`."""
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _low_wall(p: Painter, a: Point, b: Point, high: float, colour: RGB) -> None:
    """A low wall `high` tall standing on the ground from `a` to `b`, with a lighter coping."""
    p.polygon([a, b, (b[0], b[1] - high), (a[0], a[1] - high)], colour, shade(colour, 0.6))
    p.line([(a[0], a[1] - high), (b[0], b[1] - high)], shade(colour, 1.12))


def _ground(p: Painter, corners: Top, colour: RGB) -> None:
    """The ground of a yard between its fences, `colour` with a darker rim."""
    p.polygon(list(corners), colour, shade(colour, 0.85))


def _cross_doors(
    p: Painter, a: Point, b: Point, spans: tuple[tuple[float, float], ...], top: float, high: float
) -> None:
    """White cross-braced barn doors in the wall under the edge from `a` to `b`, one over each of
    `spans`, from `top` below the eaves down to the ground `high` below them."""
    trim = (240, 236, 226)
    for t0, t1 in spans:
        _patch(p, a, b, (t0, t1), (top, high), trim)
        _patch(p, a, b, (t0 + 0.02, t1 - 0.02), (top + 0.2 * METRE, high), (96, 44, 36))
        p.line([_on_wall(a, b, t0 + 0.02, top), _on_wall(a, b, t1 - 0.02, high)], trim)
        p.line([_on_wall(a, b, t1 - 0.02, top), _on_wall(a, b, t0 + 0.02, high)], trim)


def _bales(p: Painter, spots: tuple[Point, ...]) -> None:
    """Round bales of hay standing on their ends at `spots`."""
    hay = (220, 186, 96)
    m = METRE
    for x, y in spots:
        p.ellipse(x - 0.9 * m, y - 1.4 * m, x + 0.9 * m, y, hay)
        p.blob(x - 0.5 * m, y - 1.2 * m, x + 0.2 * m, y - 0.5 * m, shade(hay, 1.12))
        p.line([(x - 0.6 * m, y - 0.4 * m), (x + 0.6 * m, y - 0.4 * m)], shade(hay, 0.75))


def _trough(p: Painter, x: float, y: float, long: float, fill: RGB) -> None:
    """A wooden trough `long` metres long full of `fill`."""
    m = METRE
    p.hall(x, y, long * m, 0.7 * m, 0.6 * m, (122, 90, 60), fill)


def cattle_ranch() -> Sprite:
    """A cattle farm: a dark red boarded barn 16 by 9 m on a plinth, with white cross-braced
    doors, a white hayloft door in its gable and a white cupola on its ridge, round bales of hay by
    its gable, and beside it a fenced yard of trodden grass with a hay rack and a water trough,
    where black-and-white and brown cows stand, drawn twice their size (BEAST)."""
    p = Painter(40, 30, (20, 21))
    m, b = METRE, BEAST
    x, y, long, deep, high, rise = 5.0, -2.5, 16 * m, 9 * m, 4 * m, 3.4 * m
    wood, trim, roof = (150, 62, 48), (240, 236, 226), (70, 66, 68)
    p.shadow(12, 3)
    top = p.hall(x, y, long, deep, high, wood, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.6)
    _boards(p, top, wood, (high / 3, high * 2 / 3))
    _plinth(p, top, wood, high, 0.5 * m)
    _cross_doors(p, left, front, ((0.18, 0.4), (0.6, 0.82)), high - 3 * m, high)
    _patch(p, front, right, (0.3, 0.7), (high - 2.8 * m, high), shade(wood, 0.5))
    for t in (0.06, 0.5, 0.92):
        _patch(p, left, front, (t - 0.025, t + 0.025), (0.7 * m, 1.5 * m), shade(WINDOW, 0.8))
    p.gable(top, rise, roof, wood)
    _tiles(p, top, rise, roof, (0.35, 0.7))
    gx, gy = (front[0] + right[0]) / 2, (front[1] + right[1]) / 2
    p.polygon(
        [
            (gx - 0.6 * m, gy - 0.3 * m),
            (gx + 0.6 * m, gy - 0.9 * m),
            (gx + 0.6 * m, gy - 2.1 * m),
            (gx - 0.6 * m, gy - 1.5 * m),
        ],
        trim,
    )
    ridge_back, ridge_front = _ridge(top, rise)
    cx, cy = _along(ridge_back, ridge_front, 0.5)
    cupola = p.box(cx, cy + 0.3 * m, 0.8 * m, 1.1 * m, trim, None)
    p.gable(cupola, 0.7 * m, roof, trim)
    _bales(p, ((right[0] + 1.4, right[1] + high + 1.0), (right[0] + 2.9, right[1] + high + 0.3)))
    yard = _paddock(-8.5, 3, 8.5)
    yl, yb, yr, yn = yard
    _ground(p, yard, (122, 146, 72))
    p.soft([(-13, 1.5, -5, 5.5), (-10, 3.5, -3, 6.5)], (138, 118, 82), 0.5, 1.2)
    _fence(p, yl, yb, m)
    _fence(p, yb, yr, m)
    hay = (220, 186, 96)
    black, white, brown = (40, 38, 42), (240, 238, 232), (138, 84, 50)
    things: list[tuple[float, int]] = [(0.5, -1), (5.2, -2)]
    cows = (
        (-11.5, 2.4, white, black, 1),
        (-6.2, 2.0, brown, white, -1),
        (-9.0, 4.6, black, white, 1),
        (-4.6, 4.8, white, black, -1),
    )
    things += [(cy, k) for k, (_, cy, *_) in enumerate(cows)]
    for _, key in sorted(things):
        if key == -1:
            p.hall(-8.5, 0.6, 3 * m, 1.2 * m, 1.3 * m, (122, 90, 60), hay)
            p.blob(-10, -1.3, -7, -0.4, hay)
        elif key == -2:
            _trough(p, -12.6, 5.2, 2.5, WATER)
        else:
            cx, cy, hide, patch, facing = cows[key]
            _cow(p, cx, cy, hide, patch, b, facing)
    _fence(p, yl, yn, m)
    _fence(p, yn, yr, m)
    return p.sprite()


def pig_farm() -> Sprite:
    """A pig farm: a long low sty 13 by 6 m of pink-washed walls under a terracotta roof, with a
    low door into each pen along it and vents on its ridge, a stack of straw by its gable, and
    before it a run walled in brick and churned into soft brown mud round a wallow and a feeding
    trough, where pink pigs root and wallow, drawn twice their size (BEAST)."""
    p = Painter(40, 30, (20, 21))
    m, b = METRE, BEAST
    x, y, long, deep, high, rise = 5.0, -3.0, 13 * m, 6 * m, 2.4 * m, 2.2 * m
    wall, roof = (236, 190, 182), (176, 82, 58)
    p.shadow(11, 3)
    top = p.hall(x, y, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.5)
    _plinth(p, top, wall, high, 0.4 * m)
    for t in (0.14, 0.38, 0.62, 0.86):
        _patch(p, left, front, (t - 0.04, t + 0.04), (high - 1.2 * m, high), DOOR)
        _patch(p, left, front, (t + 0.07, t + 0.1), (0.4 * m, 0.9 * m), shade(WINDOW, 0.8))
    _patch(p, front, right, (0.35, 0.65), (high - 2 * m, high), shade(DOOR, 1.2))
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof, (0.35, 0.7))
    ridge_back, ridge_front = _ridge(top, rise)
    for t in (0.3, 0.7):
        vx, vy = _along(ridge_back, ridge_front, t)
        vent = p.box(vx, vy + 0.2 * m, 0.5 * m, 0.6 * m, shade(wall, 0.9), None)
        p.gable(vent, 0.35 * m, shade(roof, 0.85), shade(wall, 0.9))
    straw = (226, 194, 112)
    _stack(p, right[0] + 2.2, right[1] + 2.4, 1.6 * m, 2 * m, straw)
    run = _paddock(-7.5, 3, 9)
    rl, rb, rr, rn = run
    brick, low = (164, 92, 70), 0.9 * m
    _ground(p, run, (126, 98, 70))
    p.soft([(-14, 1, -1, 6), (-11, 2.5, -4, 6.5)], (104, 78, 54), 0.7, 1.0)
    p.soft([(-12.6, 3.2, -9.2, 4.9)], (88, 74, 60), 0.9, 0.5)
    p.blob(-11.9, 3.6, -9.9, 4.5, (112, 104, 88))
    _low_wall(p, rl, rb, low, brick)
    _low_wall(p, rb, rr, low, brick)
    pigs = ((-8.0, 1.4, 1), (-4.4, 2.2, -1), (-6.5, 3.4, 1), (-11.0, 4.2, -1), (-6.8, 5.6, -1))
    things = [(cy, k) for k, (_, cy, _) in enumerate(pigs)] + [(2.6, -1)]
    for _, key in sorted(things):
        if key == -1:
            _trough(p, -2.6, 2.6, 2.4, (176, 150, 96))
        else:
            px, py, facing = pigs[key]
            _pig(p, px, py, b, facing)
    _low_wall(p, rl, rn, low, brick)
    _low_wall(p, rn, rr, low, brick)
    return p.sprite()


def chicken_farm() -> Sprite:
    """A poultry farm: a whitewashed wooden henhouse 10 by 4.5 m under a red roof, raised on a
    plinth, with three pop-holes and their ramps, nest boxes along its back and a golden cockerel
    turning on its ridge, a feed bin by its gable, and before it a run fenced with wire over
    scratched bare earth, round a feeder and a drinker, where white, brown and black hens peck and
    a red cockerel struts, drawn twice their size (BEAST)."""
    p = Painter(40, 30, (20, 21))
    m, b = METRE, BEAST
    x, y, long, deep, high, rise = 5.5, -3.0, 10 * m, 4.5 * m, 2.2 * m, 1.6 * m
    wall, roof = (244, 240, 230), (184, 58, 44)
    p.shadow(10, 3)
    top = p.hall(x, y, long, deep, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.5)
    _boards(p, top, wall, (high / 3, high * 2 / 3))
    _plinth(p, top, wall, high, 0.5 * m)
    for t in (0.2, 0.5, 0.8):
        hole = _on_wall(left, front, t, high - 0.5 * m)
        _patch(p, left, front, (t - 0.03, t + 0.03), (high - 1.1 * m, high - 0.5 * m), DOOR)
        p.line([hole, (hole[0] - 0.6 * m, hole[1] + 0.9 * m)], (150, 112, 70))
        _patch(p, left, front, (t + 0.07, t + 0.12), (0.4 * m, 0.9 * m), shade(WINDOW, 0.85))
    _patch(p, front, right, (0.3, 0.7), (high - 1.9 * m, high), (176, 132, 84))
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof, (0.4, 0.75))
    ridge_back, ridge_front = _ridge(top, rise)
    vx, vy = _along(ridge_back, ridge_front, 0.5)
    gold = (226, 176, 60)
    p.line([(vx, vy), (vx, vy - 1.2 * m)], (80, 76, 70))
    p.blob(vx - 0.3 * m, vy - 1.8 * m, vx + 0.3 * m, vy - 1.2 * m, gold)
    p.polygon(
        [(vx - 0.3 * m, vy - 1.6 * m), (vx - 0.7 * m, vy - 2.1 * m), (vx - 0.2 * m, vy - 1.3 * m)],
        gold,
    )
    p.point(vx + 0.2 * m, vy - 1.9 * m, (206, 40, 36))
    bin_ = (150, 156, 162)
    _turned(
        p, right[0] + 1.6, right[1] + 2.0, [(0, 0.7 * m), (1.8 * m, 0.7 * m), (2.4 * m, 0)], bin_
    )
    run = _paddock(-7.5, 3, 9)
    rl, rb, rr, rn = run
    _ground(p, run, (150, 150, 84))
    p.soft([(-14, 1, -1, 6), (-10, 2.5, -3, 6.5)], (170, 146, 102), 0.6, 1.2)
    wire = (190, 196, 200)
    for a, c in ((rl, rb), (rb, rr)):
        _fence(p, a, c, m)
        for drop in (0.25 * m, 1.5 * m):
            p.line([(a[0], a[1] - drop), (c[0], c[1] - drop)], wire)
    feeder, drinker = (-7.5, 2.6), (-3.2, 4.2)
    white, brown, black = (246, 244, 238), (176, 108, 60), (52, 48, 50)
    hens = []
    for k in range(16):
        u = (0.5 + k * 0.754878) % 1 * 1.6 - 0.8
        v = (0.5 + k * 0.569840) % 1 * 1.6 - 0.8
        hx, hy = -7.5 + 9 * (u + v) / 2, 3 + 9 * (u - v) / 4
        colour = (white, white, brown, white, black)[k % 5]
        hens.append((hx, hy, colour, 1 if k % 3 else -1))
    things = [(hy, k) for k, (_, hy, _, _) in enumerate(hens)]
    things += [(feeder[1], -1), (drinker[1], -2)]
    for _, key in sorted(things):
        if key == -1:
            _turned(
                p,
                *feeder,
                [(0, 0.5 * m), (0.4 * m, 0.5 * m), (0.6 * m, 0.2 * m), (0.9 * m, 0)],
                (196, 170, 110),
            )
        elif key == -2:
            _turned(p, *drinker, [(0, 0.4 * m), (0.7 * m, 0.4 * m), (0.9 * m, 0)], (170, 176, 184))
        else:
            hx, hy, colour, facing = hens[key]
            _hen(p, hx, hy, colour, b, facing)
    _hen(p, -10.2, 5.2, (150, 48, 34), 1.3 * b, 1)
    for a, c in ((rl, rn), (rn, rr)):
        _fence(p, a, c, m)
        for drop in (0.25 * m, 1.5 * m):
            p.line([(a[0], a[1] - drop), (c[0], c[1] - drop)], wire)
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
    """Hop poles 6 m tall strung with wires and green bines hung with cones, a round brick oast
    house 5 m across with a tiled kiln roof, a door and a white cowl, a boarded store beside it,
    and hop pockets and a basket of picked hops."""
    p = Painter(30, 26, (15, 19))
    m = METRE
    pole, bine = (122, 96, 66), (88, 152, 58)
    cone = (180, 214, 112)
    tall = 6 * m
    for r in range(2):
        poles = [(-11 + 4 * k - 4 * r, -3 + 2 * k + 2 * r) for k in range(3)]
        tops = [(x, y - tall) for x, y in poles]
        for (x, y), (tx, ty) in zip(poles, tops, strict=True):
            p.blob(x - 1.4, y - 0.4, x + 1.4, y + 0.8, (98, 80, 54))
            p.line([(x, y), (tx, ty)], pole)
            p.line([(x - 0.6, y - 0.6), (x - 0.6, y - 0.8 * tall)], bine)
            p.line([(x + 0.6, y - 0.4), (x + 0.6, y - 0.9 * tall)], shade(bine, 0.85))
            for k in range(3):
                p.point(x + (k % 2) * 1.2 - 0.6, y - 1.2 - 1.1 * k, cone)
        p.line(tops, (70, 64, 58))
        for (x0, y0), (x1, y1) in itertools.pairwise(tops):
            middle = (x0 + x1) / 2, (y0 + y1) / 2
            p.line([(x0 + 0.6, y0 + 1), (middle[0], middle[1] + 1.8), (x1 - 0.6, y1 + 1)], bine)
            p.point(middle[0], middle[1] + 2.2, cone)
    _worn(p, 5, 3, 4)
    _track(p, [(1.6, 4.4), (1, 6.4), (0.6, 8.5)])
    oast, kiln, cowl = (176, 92, 64), (110, 76, 64), (244, 242, 236)
    ox, oy, radius, wall, roof = 8.0, 0.0, 2.5 * m, 6 * m, 4.5 * m
    p.blob(ox - radius - 1, oy - 1.2, ox + radius + 1, oy + 1.8, FOOTING)
    _turned(p, ox, oy, [(0, radius), (wall, radius)], oast)
    for h in (wall / 3, 2 * wall / 3):
        _arc(p, ox, oy - h, radius, shade(oast, 0.8))
    p.rect(ox - 1.2, oy - 0.9, ox - 0.3, oy + 0.9, (60, 40, 34))
    _turned(p, ox, oy - wall, [(0, radius + 0.2), (roof, 0.3)], kiln)
    for h in (roof / 3, 2 * roof / 3):
        _arc(p, ox, oy - wall - h, radius + 0.2 - (radius - 0.1) * h / roof, shade(kiln, 0.75))
    cx, cy = ox, oy - wall - roof
    p.polygon(
        [
            (cx - 0.6, cy + 0.2),
            (cx - 0.6, cy - 1.2),
            (cx + 0.5, cy - 1.9),
            (cx + 1, cy - 0.9),
            (cx + 0.5, cy + 0.2),
        ],
        cowl,
        shade(cowl, 0.6),
    )
    p.line([(cx + 1, cy - 1.4), (cx + 2, cy - 1.9)], cowl)
    p.point(cx + 0.4, cy - 0.9, shade(cowl, 0.45))
    store = (228, 222, 206)
    high = 3.2 * m
    stow = p.box(3.4, 3.8, 3 * m, high, store, None)
    left, front, right, _ = stow
    _footing(p, stow, high)
    _plinth(p, stow, store, high, 0.4 * m)
    _door(p, left, front, (0.3, 0.6), high - 2 * m, high)
    _window(p, front, right, (0.35, 0.65), 0.7 * m, shade(WINDOW, 0.8))
    p.gable(stow, 2.2 * m, (112, 82, 66), store)
    _tiles(p, stow, 2.2 * m, (112, 82, 66), (0.5,))
    pocket = (214, 200, 168)
    for x, y in ((10.8, 4.2), (12.4, 3.5)):
        p.ellipse(x - 0.45 * m, y - 2 * m, x + 0.45 * m, y, pocket)
        p.line([(x - 0.35 * m, y - m), (x + 0.35 * m, y - m)], shade(pocket, 0.65))
    _basket(p, -4, 5.5, cone)
    return p.sprite()
