"""Factories and works of every kind."""

import math

import numpy as np

from railroads.ui.pixel.art.painter import (
    BARREL,
    BRICK,
    CAR_COLOURS,
    CLOTH,
    COPPER,
    DOORWAY,
    GLASS,
    METAL,
    RGB,
    STONE,
    TIMBER,
    WHITEWASH,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import (
    _barrel,
    _cow,
    _fence,
    _flag,
    _heap,
    _iso,
    _on_wall,
    _paddock,
    _patch,
    _plate,
    _turned,
    _vault,
    _windows,
)

DOOR: RGB = (104, 66, 44)

SILL: RGB = (240, 236, 226)

GLAZING: RGB = (160, 196, 218)

EARTH: RGB = (150, 128, 90)

SOIL: RGB = (78, 74, 52)

OIL: RGB = (62, 60, 62)

PUDDLE: RGB = (98, 126, 146)


def factory() -> Sprite:
    """A small brick works with one chimney: a sawtooth roof glazed along its teeth, a row of
    arched windows over a dark plinth, a loading door with a cart track worn out from it and
    crates by it."""
    p = Painter(20, 28, (9, 22))
    foot = _footing(0, 0, 7, 7)
    _bed(p, foot)
    p.shadow(7, 2.5)
    _track(p, foot[1], foot[2], (0.2, 0.5), 5)
    top = p.box(0, 0, 7, 6, BRICK, (96, 94, 100))
    left, front, right, _ = top
    _plinth(p, top, 6, BRICK)
    _stains(p, left, front, 6, (0.1, 0.5, 0.9), shade(BRICK, 0.62))
    _windows(p, left, front, (0.15, 0.38, 0.62, 0.85), (2,), 2, WINDOW, shade(BRICK, 1.35))
    _door(p, front, right, (0.2, 0.5), 2.5, 6, DOORWAY, shade(BRICK, 1.3))
    _window(p, front, right, (0.7, 0.85), 2.2, shade(WINDOW, 0.8))
    _sawtooth(p, top, 2, 2.5, METAL, (170, 200, 214), BRICK)
    _stack(p, 3, -6, -16, shade(BRICK, 0.8))
    _smoke(p, 4.5, -18, (214, 214, 218), 2)
    for x, y in ((7.5, 1.2), (9, 0.4)):
        _crate(p, x, y)
    return p.sprite()


def big_factory() -> Sprite:
    """A big brick works: sheds under a glazed sawtooth roof, rows of arched windows over a dark
    plinth, a loading door with cart ruts worn out from it past a coal heap, and two banded
    chimneys."""
    p = Painter(30, 40, (14, 33))
    foot = _footing(0, 0, 12, 12)
    _bed(p, foot)
    _spill(p, -9, 5, 2.5, (48, 46, 50))
    p.shadow(12, 4)
    _track(p, foot[1], foot[2], (0.15, 0.45), 3)
    top = p.box(0, 0, 12, 8, BRICK, (96, 94, 100))
    left, front, right, _ = top
    _plinth(p, top, 8, BRICK)
    _stains(p, left, front, 8, (0.04, 0.3, 0.55, 0.8), shade(BRICK, 0.62))
    _stains(p, front, right, 8, (0.6, 0.9), shade(BRICK, 0.5))
    spans = tuple(0.08 + k * 0.12 for k in range(8))
    _windows(p, left, front, spans, (2,), 2, WINDOW, shade(BRICK, 1.35))
    _windows(p, left, front, spans, (5,), 1, WINDOW)
    _windows(p, front, right, (0.6, 0.75, 0.9), (2,), 2, shade(WINDOW, 0.8))
    _door(p, front, right, (0.15, 0.45), 3.5, 8, DOORWAY, shade(BRICK, 1.3))
    _sawtooth(p, top, 3, 3, METAL, (170, 200, 214), BRICK)
    for x in (-6, 4):
        _stack(p, x, -9, -24, shade(BRICK, 0.8))
    for i, (x, y, r) in enumerate(
        [(-5, -27, 2), (-3, -30, 2.5), (0, -32, 3), (5, -27, 2), (7, -30, 2.5)]
    ):
        p.blob(x - r, y - r, x + r, y + r, (226, 226, 230) if i % 2 else (204, 204, 210))
    _heap(p, -9, 5, 2.5, 2, (48, 46, 50))
    _oil(p, 8.5, 5.4, 1.2)
    return p.sprite()


def works(wall: RGB, roof: RGB, sign: RGB, stacks: int) -> Sprite:
    """A works: a long hall in its own colours on a dark plinth, its flat roof behind a parapet
    with roof lights and a vent, rows of windows with sills, a loading door under a lintel with a
    cart track worn out from it, a framed sign and round banded chimneys."""
    p = Painter(32, 40, (15, 32))
    foot = _footing(0, 0, 12, 12)
    _bed(p, foot)
    p.shadow(13, 4)
    _track(p, foot[1], foot[2], (0.12, 0.4), 3.5)
    top = p.box(0, 0, 12, 8, wall, roof)
    left, front, right, _ = top
    _plinth(p, top, 8, wall)
    _stains(p, left, front, 8, (0.06, 0.3, 0.62, 0.88), shade(wall, 0.6))
    _stains(p, front, right, 8, (0.55, 0.85), shade(wall, 0.48))
    _band(p, left, front, 1, shade(wall, 1.15))
    _band(p, front, right, 1, shade(wall, 0.9))
    spans = tuple(0.5 + k * 0.1 for k in range(5))
    _windows(p, left, front, spans, (2.5,), 2, WINDOW, shade(wall, 1.25))
    _windows(p, left, front, spans, (4.8,), 1, shade(wall, 1.15))
    _windows(p, front, right, (0.55, 0.7, 0.85), (2.5,), 2, shade(WINDOW, 0.8), shade(wall, 1.1))
    _door(p, front, right, (0.12, 0.4), 3.5, 8, DOORWAY, shade(wall, 1.25))
    p.line([left, front, right], shade(roof, 1.3))
    for x, y in ((-5, -8), (1, -5)):
        p.hall(x, y, 6, 2, 0.8, shade(roof, 0.8), GLAZING)
    p.box(6, -9, 1.2, 1.5, shade(roof, 0.8), shade(roof, 1.1))
    p.rect(-9.5, -5.5, -3.5, -1.5, shade(sign, 0.6))
    p.rect(-9, -5, -4, -2, sign)
    p.line([(-8, -4), (-5, -3.5)], (240, 240, 236))
    for k in range(stacks):
        x = 3 + 4 * k
        _chimney(p, x, -9, -23, shade(BRICK, 0.8))
        _smoke(p, x + 1, -26, (150, 148, 150), 2)
    for x, y in ((9, 6.5), (11, 5.5)):
        _crate(p, x, y)
    _oil(p, 4.5, 5.5, 1.2)
    return p.sprite()


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


def _chimney(p: Painter, x: float, foot: float, top: float, colour: RGB = BRICK) -> None:
    """A round brick chimney narrowing to a dark cap, with a light band under the cap and a dark
    one halfway down."""
    p.polygon(
        [(x - 2, foot), (x - 1, top), (x + 1, top), (x + 2, foot)], colour, shade(colour, 0.6)
    )
    p.line([(x - 1, foot - 1), (x - 1, top + 1)], shade(colour, 1.2))
    for y, tone in ((top + 3, (220, 220, 214)), ((top + foot) / 2, shade(colour, 0.65))):
        half = 1 + (y - top) / (foot - top)
        p.line([(x - half, y), (x + half, y)], tone)
    p.rect(x - 2, top - 1, x + 2, top, shade(colour, 0.7))


def _smoke(p: Painter, x: float, y: float, colour: RGB = (150, 148, 150), puffs: int = 3) -> None:
    for i in range(puffs):
        r = 1.5 + 0.6 * i
        cx, cy = x + 1.5 * i, y - 2.5 * i
        p.blob(cx - r, cy - r, cx + r, cy + r, colour if i % 2 else shade(colour, 0.86))


def _gear(p: Painter, x: float, y: float, radius: float, colour: RGB) -> None:
    for k in range(8):
        a = k * math.pi / 4
        tx, ty = x + (radius + 0.8) * math.cos(a), y + (radius + 0.8) * math.sin(a)
        p.blob(tx - 1, ty - 1, tx + 1, ty + 1, shade(colour, 0.8))
    p.ellipse(x - radius, y - radius, x + radius, y + radius, colour)
    p.blob(x - 1, y - 1, x + 1, y + 1, shade(colour, 0.45))


def _bale(p: Painter, x: float, y: float, colour: RGB) -> None:
    p.box(x, y, 2, 2, colour, shade(colour, 1.15))


def _plinth(
    p: Painter, top: tuple[Point, Point, Point, Point], height: float, wall: RGB, tall: float = 0.8
) -> None:
    """A darker band along the foot of a block's lit and shaded walls, `height` below its top."""
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (height - tall, height), shade(wall, 0.7))
    _patch(p, front, right, (0, 1), (height - tall, height), shade(wall, 0.55))


def _band(p: Painter, a: Point, b: Point, drop: float, colour: RGB) -> None:
    """A line right along the wall under the edge from `a` to `b`, `drop` down it."""
    p.line([_on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop)], colour)


def _window(
    p: Painter,
    a: Point,
    b: Point,
    span: tuple[float, float],
    drop: float,
    fill: RGB = WINDOW,
    tall: float = 1.2,
) -> None:
    """A window `tall` high with a light sill under it, on the wall under the edge from `a` to
    `b`."""
    _patch(p, a, b, span, (drop, drop + tall), fill)
    sill = drop + tall + 0.4
    p.line([_on_wall(a, b, span[0], sill), _on_wall(a, b, span[1], sill)], SILL)


def _door(
    p: Painter,
    a: Point,
    b: Point,
    span: tuple[float, float],
    drop: float,
    foot: float,
    fill: RGB = DOORWAY,
    lintel: RGB | None = None,
) -> None:
    """A door from `drop` down to `foot` on the wall under the edge from `a` to `b`, with a light
    lintel over it if `lintel` is given."""
    _patch(p, a, b, span, (drop, foot), fill)
    if lintel is not None:
        p.line([_on_wall(a, b, span[0], drop - 0.6), _on_wall(a, b, span[1], drop - 0.6)], lintel)


def _ridge(top: tuple[Point, Point, Point, Point], rise: float) -> tuple[Point, Point]:
    """The back and front ends of the ridge of a gable `rise` high on a box's top (Painter.gable)."""
    left, front, right, back = top
    return (
        ((left[0] + back[0]) / 2, (left[1] + back[1]) / 2 - rise),
        ((front[0] + right[0]) / 2, (front[1] + right[1]) / 2 - rise),
    )


def _on_roof(top: tuple[Point, Point, Point, Point], rise: float, along: float, up: float) -> Point:
    """The point on the lit side of a gable roof `along` of the way from its back to its front and
    `up` of the way from its eave to its ridge."""
    left, front, _, _ = top
    ridge_back, ridge_front = _ridge(top, rise)
    ex, ey = _on_wall(left, front, along, 0)
    rx, ry = _on_wall(ridge_back, ridge_front, along, 0)
    return ex + (rx - ex) * up, ey + (ry - ey) * up


def _roof_rows(
    p: Painter,
    top: tuple[Point, Point, Point, Point],
    rise: float,
    roof: RGB,
    rows: tuple[float, ...] = (0.34, 0.68),
) -> None:
    """Rows of tiles across the lit side of a gable roof."""
    for up in rows:
        p.line([_on_roof(top, rise, 0, up), _on_roof(top, rise, 1, up)], shade(roof, 0.8))


def _skylight(
    p: Painter,
    top: tuple[Point, Point, Point, Point],
    rise: float,
    span: tuple[float, float] = (0.1, 0.9),
) -> None:
    """A strip of glazing along the lit side of a gable roof just under its ridge."""
    t0, t1 = span
    p.polygon(
        [
            _on_roof(top, rise, t0, 0.55),
            _on_roof(top, rise, t1, 0.55),
            _on_roof(top, rise, t1, 0.85),
            _on_roof(top, rise, t0, 0.85),
        ],
        GLAZING,
        shade(GLAZING, 0.6),
    )


def _pipe(p: Painter, x: float, y: float, tall: float, colour: RGB = (70, 66, 64)) -> None:
    """A thin stove pipe standing on (x, y) with a dark cap."""
    p.rect(x - 0.5, y - tall, x + 0.5, y, colour)
    p.line([(x - 1, y - tall), (x + 1, y - tall)], shade(colour, 0.6))


def _stack(p: Painter, x: float, foot: float, top: float, colour: RGB) -> None:
    """A square stack two wide from `foot` up to `top`, with a light band under a dark cap."""
    p.rect(x, top, x + 2, foot, colour)
    p.line([(x, top + 2), (x + 2, top + 2)], (220, 220, 214))
    p.line([(x, (top + foot) / 2), (x + 2, (top + foot) / 2)], shade(colour, 0.65))
    p.rect(x - 0.5, top - 1, x + 2.5, top, shade(colour, 0.6))


def _crate(p: Painter, x: float, y: float, colour: RGB = (150, 112, 74)) -> None:
    """A wooden crate standing on (x, y)."""
    p.box(x, y, 1.8, 1.8, colour, shade(colour, 1.2))


def _sack(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A full sack lying on (x, y), tied at its top."""
    p.ellipse(x - 1.5, y - 2.5, x + 1.5, y, colour)
    p.point(x - 0.5, y - 2, shade(colour, 1.15))


def _drum(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A steel drum standing on (x, y), with a band round its middle."""
    p.rect(x - 1, y - 3, x + 1, y, colour)
    p.line([(x - 1, y - 1.5), (x + 1, y - 1.5)], shade(colour, 0.65))


def _ring(p: Painter, x: float, y: float, half: float, colour: RGB) -> None:
    """A band round the front of an upright round thing whose axis passes through (x, y)."""
    p.line([(x - half, y), (x, y + half / 2), (x + half, y)], colour)


def _lengths(p: Painter, x: float, y: float, colour: RGB, rows: int = 2) -> None:
    """A stack of pipes or bars `rows` high lying along the lit walls on (x, y), their dark ends
    showing on its shaded side."""
    stack = p.hall(x, y, 6, 3, 1.2 * rows, colour, shade(colour, 1.2))
    left, front, right, _ = stack
    for row in range(rows):
        drop = 1.2 * row + 0.6
        if row:
            _band(p, left, front, 1.2 * row, shade(colour, 0.7))
        for t in (0.25, 0.75):
            p.point(*_on_wall(front, right, t, drop - 0.5), shade(colour, 0.45))


def _footing(x: float, y: float, length: float, depth: float) -> tuple[Point, Point, Point, Point]:
    """The left, front, right and back corners on the ground of a block that Painter.hall raises at
    (x, y) with the same `length` and `depth`; a box of half width s has both s."""
    front = (x + (length - depth) / 2, y + (length + depth) / 4)
    left = (front[0] - length, front[1] - length / 2)
    right = (front[0] + depth, front[1] - depth / 2)
    return left, front, right, (left[0] + depth, left[1] - depth / 2)


def _bed(
    p: Painter, foot: tuple[Point, Point, Point, Point], colour: RGB = SOIL, margin: float = 1.2
) -> None:
    """Darker ground hugging a block's foot `margin` out past its walls, so that it stands in the
    ground rather than on it; drawn before the block."""
    left, front, right, back = foot
    p.polygon(
        [
            (left[0] - 2 * margin, left[1]),
            (back[0], back[1] - margin),
            (right[0] + 2 * margin, right[1]),
            (front[0], front[1] + margin),
        ],
        colour,
    )


def _track(
    p: Painter,
    a: Point,
    b: Point,
    span: tuple[float, float],
    length: float,
    colour: RGB = EARTH,
    ruts: bool = True,
) -> None:
    """A track of trodden earth leading `length` out across the ground from a door at `span` along
    the foot of the wall from ground corner `a` to `b`, widening as it goes, with two cart ruts
    worn into it if `ruts`."""
    out = (-1 if b[1] > a[1] else 1, 0.5)
    t0, t1 = span
    flare = 0.12 * length / max(1.0, math.dist(a, b))

    def away(t: float, k: float) -> Point:
        x, y = _on_wall(a, b, t, 0)
        return x + out[0] * k, y + out[1] * k

    p.polygon(
        [away(t0, 0), away(t1, 0), away(t1 + flare, length), away(t0 - flare, length)], colour
    )
    if ruts:
        for t in (t0 + 0.3 * (t1 - t0), t1 - 0.3 * (t1 - t0)):
            p.line([away(t, 0.5), away(t, length - 0.5)], shade(colour, 0.76))


def _stains(
    p: Painter,
    a: Point,
    b: Point,
    height: float,
    spans: tuple[float, ...],
    wall: RGB,
    rise: float = 1.6,
) -> None:
    """Streaks of damp and soot creeping up from the foot of the wall under the edge from `a` to
    `b`, `height` below it, at each of `spans` along it."""
    for k, t in enumerate(spans):
        reach = rise * (0.7 + 0.3 * (k % 2))
        p.line([_on_wall(a, b, t, height - 0.5), _on_wall(a, b, t, height - reach)], wall)


def _spill(p: Painter, x: float, y: float, half: float, colour: RGB) -> None:
    """Loose stuff spread flat over the ground round the foot of a heap at (x, y), with a few
    lumps rolled off it; drawn before the heap."""
    p.blob(x - half * 1.4, y - half * 0.4, x + half * 1.3, y + half * 0.6, shade(colour, 0.82))
    for dx, dy in ((-1.5, 0.9), (0.8, 1.1), (1.3, 0.7)):
        p.point(x + dx * half, y + dy * half * 0.6, shade(colour, 0.7))


def _oil(p: Painter, x: float, y: float, r: float = 1.5) -> None:
    """A dark stain of spilt oil on the ground."""
    p.blob(x - r, y - r / 2, x + r, y + r / 2, OIL)


def _puddle(p: Painter, x: float, y: float, r: float = 1.5) -> None:
    """A shallow puddle on the ground with the sky glinting in it."""
    p.blob(x - r, y - r / 2, x + r, y + r / 2, PUDDLE)
    p.point(x - r / 2, y - 0.5, shade(PUDDLE, 1.4))


def barrel_maker() -> Sprite:
    """A cooper's shed with a wide door, a window and a smoking stove pipe, a footpath scattered
    with shavings worn out from its door to a chopping block, its finished barrels stood outside
    beside a pile of staves."""
    p = Painter(28, 22, (13, 16))
    foot = _footing(-3, 0, 7, 7)
    _bed(p, foot)
    p.shadow(11, 3)
    _track(p, foot[0], foot[1], (0.46, 0.84), 3, ruts=False)
    roof = (118, 84, 54)
    top = p.box(-3, 0, 7, 5, TIMBER, None)
    left, front, right, _ = top
    for drop in (2, 3.5):
        _band(p, front, right, drop, shade(TIMBER, 0.66))
    _plinth(p, top, 5, TIMBER)
    _stains(p, left, front, 5, (0.05, 0.3, 0.95), shade(TIMBER, 0.6), 1.2)
    _window(p, left, front, (0.1, 0.3), 1.6)
    _door(p, left, front, (0.46, 0.84), 1.8, 5, lintel=shade(TIMBER, 1.3))
    p.line([_on_wall(left, front, 0.65, 1.9), _on_wall(left, front, 0.65, 5)], shade(TIMBER, 0.55))
    _window(p, front, right, (0.35, 0.62), 1.6, shade(WINDOW, 0.8))
    p.gable(top, 3, roof, TIMBER)
    _roof_rows(p, top, 3, roof, (0.5,))
    x, y = _on_roof(top, 3, 0.3, 0.8)
    _pipe(p, x, y, 2.5)
    _smoke(p, x + 0.5, y - 4, puffs=2)
    staves = (206, 168, 116)
    for sx, sy in ((-6.5, 4.5), (-4.5, 5), (-8, 3.6), (-3, 4.2)):
        p.point(sx, sy, (226, 200, 150))
    pile = p.hall(-9.5, 4, 5, 2, 1.6, staves, shade(staves, 1.12))
    _band(p, pile[0], pile[1], 0.8, shade(staves, 0.6))
    p.ellipse(-2, 3.5, 1, 5, (128, 94, 62))
    p.line([(-0.5, 4), (0.5, 2.5)], (90, 88, 92))
    for x, y in ((7, -1), (11, 1), (6, 3)):
        _barrel(p, x, y)
    return p.sprite()


def bakery() -> Sprite:
    """A baker's house with a shop window of loaves under a striped red awning, a brick oven
    chimney, a loaf hung over the door, a swept path worn to its door and a sack of flour by its
    wall with flour spilt round it."""
    p = Painter(18, 22, (8, 17))
    foot = _footing(0, 0, 6, 6)
    _bed(p, foot, margin=1)
    p.shadow(7, 2.5)
    _track(p, foot[0], foot[1], (0.64, 0.9), 2, (176, 160, 128), ruts=False)
    wall, roof, loaf = (232, 214, 176), (150, 72, 52), (196, 128, 58)
    top = p.box(0, 0, 6, 5, wall, None)
    left, front, right, _ = top
    _plinth(p, top, 5, wall)
    _stains(p, front, right, 5, (0.2, 0.8), shade(wall, 0.6), 1.2)
    _patch(p, left, front, (0.08, 0.5), (2.2, 3.8), (238, 204, 132))
    for t in (0.16, 0.29, 0.42):
        p.point(*_on_wall(left, front, t, 3), loaf)
    _patch(p, left, front, (0.04, 0.54), (1, 2), (178, 52, 46))
    for t in (0.12, 0.29, 0.46):
        p.line([_on_wall(left, front, t, 1), _on_wall(left, front, t, 1.8)], (240, 228, 214))
    _door(p, left, front, (0.66, 0.88), 1.6, 5, DOOR)
    _window(p, front, right, (0.35, 0.65), 1.6, shade(WINDOW, 0.8))
    p.gable(top, 4, roof, wall)
    _roof_rows(p, top, 4, roof)
    p.rect(2, -13, 4, -6, BRICK)
    p.line([(2, -11), (4, -11)], shade(BRICK, 1.3))
    p.rect(1.5, -14, 4.5, -13, shade(BRICK, 0.6))
    for x, y, r in ((3, -15, 1.2), (4, -16.5, 1.4)):
        p.blob(x - r, y - r, x + r, y + r, (224, 222, 226))
    p.line([(-4.5, -7), (-4.5, -6)], (60, 56, 54))
    p.ellipse(-6, -6, -3, -4, loaf)
    p.line([(-5, -5), (-4, -5.5)], shade(loaf, 1.3))
    p.blob(3, 3, 7, 4.6, (226, 220, 204))
    _sack(p, 4.5, 3.5, (220, 206, 176))
    return p.sprite()


def food_plant() -> Sprite:
    """A cannery: a pale hall under a green roof with a row of windows, a loading door with cart
    ruts worn out from it, a red label for a sign and crates of tins, beside two round banded
    silos joined by a catwalk, with a puddle at their feet."""
    p = Painter(26, 32, (12, 25))
    foot = _footing(-2, 0, 8, 8)
    _bed(p, foot)
    p.blob(2, -1, 13, 2.5, SOIL)
    p.shadow(10, 3)
    _track(p, foot[0], foot[1], (0.56, 0.86), 3)
    wall = (222, 214, 192)
    top = p.box(-2, 0, 8, 6, wall, (86, 128, 84))
    left, front, right, _ = top
    _plinth(p, top, 6, wall)
    _stains(p, left, front, 6, (0.04, 0.24, 0.46), shade(wall, 0.62))
    _stains(p, front, right, 6, (0.6, 0.9), shade(wall, 0.5))
    p.line([left, front, right], shade((86, 128, 84), 1.3))
    _patch(p, left, front, (0.06, 0.42), (0.8, 2), (196, 58, 48))
    p.line([_on_wall(left, front, 0.12, 1.4), _on_wall(left, front, 0.36, 1.4)], (240, 230, 210))
    for span in ((0.08, 0.2), (0.28, 0.4)):
        _window(p, left, front, span, 2.8)
    _door(p, left, front, (0.56, 0.86), 2.6, 6, lintel=shade(wall, 1.12))
    _window(p, front, right, (0.2, 0.4), 2.4, shade(WINDOW, 0.8))
    p.line([(4, -14), (0, -14), (0, -8)], shade(METAL, 0.75))
    silo = shade(METAL, 1.1)
    for x in (6, 10):
        _turned(p, x - 0.5, 0.5, [(0, 1.8), (17, 1.8), (18.5, 1)], silo, cap=shade(METAL, 1.25))
        for h in (6, 12):
            _ring(p, x - 0.5, 0.5 - h, 1.8, shade(silo, 0.7))
    p.line([(5.5, -18.5), (9.5, -18.5)], shade(METAL, 0.6))
    p.line([(5.5, -19.5), (9.5, -19.5)], shade(METAL, 0.75))
    _puddle(p, 8, 3.5, 1.4)
    for x, y in ((-11, 3), (-1.5, 5.8)):
        _crate(p, x, y)
    return p.sprite()


def steel_mill() -> Sprite:
    """A dark mill on blackened ground with a blast furnace glowing at its foot and a plume of
    smoke, a row of glowing windows, a round banded furnace tower fed by an inclined hoist from a
    heap of ore spread over the ground, a cinder track out of its furnace door and ingots stacked
    by the wall."""
    p = Painter(32, 46, (14, 38))
    foot = _footing(0, 0, 11, 11)
    cinder = (64, 60, 60)
    _bed(p, foot, cinder, 1.6)
    _spill(p, 13, 5, 3.5, (150, 76, 54))
    p.shadow(12, 4)
    _track(p, foot[0], foot[1], (0.15, 0.45), 3, (92, 84, 80))
    iron = (82, 80, 84)
    top = p.box(0, 0, 11, 9, iron, (60, 58, 62))
    left, front, right, _ = top
    _plinth(p, top, 9, iron)
    _stains(p, front, right, 9, (0.15, 0.45, 0.8), shade(iron, 0.55), 2)
    glow = (186, 104, 48)
    _windows(p, left, front, (0.52, 0.66, 0.8, 0.94), (2.5,), 3, glow, shade(iron, 1.6))
    _windows(p, front, right, (0.3, 0.55), (3,), 2, shade(glow, 0.8), shade(iron, 1.4))
    p.line([left, front, right], shade(iron, 1.35))
    furnace = shade(iron, 0.85)
    _turned(p, 6.5, -6, [(0, 2.6), (13, 2.6), (17, 1.8), (20, 1.4)], furnace, (40, 36, 36))
    for h in (4, 8, 12):
        _ring(p, 6.5, -6 - h, 2.6, shade(furnace, 0.6))
    p.point(6.5, -26.5, (255, 170, 60))
    _heap(p, 13, 5, 3.5, 3, (150, 76, 54))
    p.line([(13, 2), (8.5, -22)], (126, 96, 70))
    p.line([(13.8, 2), (9.3, -22)], shade((126, 96, 70), 0.7))
    p.rect(10.5, -15, 12.5, -13, (110, 106, 110))
    p.rect(-9, -12, -6, -2, (238, 132, 40))
    p.point(-8, -10, (255, 214, 120))
    _stack(p, -3, -9, -30, shade(BRICK, 0.7))
    for i, (x, y, r) in enumerate([(-2, -33, 2), (0, -36, 3), (3, -39, 3.5)]):
        p.blob(x - r, y - r, x + r, y + r, (150, 148, 150) if i % 2 else (124, 122, 126))
    for gx, gy in ((-9.5, 3.4), (-8, 4.4)):
        p.point(gx, gy, (226, 120, 44))
    ingot = (128, 128, 136)
    for lift in (0, 1.2):
        p.hall(-3, 7 - lift, 4, 2, 1.2, ingot, shade(ingot, 1.2))
    return p.sprite()


def textile_mill() -> Sprite:
    """A red-brick cotton mill of Łódź: a long block of storeys with rows of small arched windows
    over a dark plinth stained by the damp, a drainpipe and a cornice, a sawtooth weaving shed at
    its end, a tall banded round chimney, cart tracks worn out of both doors and bales of cloth
    by the door."""
    p = Painter(37, 55, (17, 44))
    wall, roof = (178, 78, 58), (80, 76, 84)
    mill_foot, shed_foot = _footing(-4, -1, 14, 8), _footing(7, 4.5, 8, 8)
    _bed(p, mill_foot)
    _bed(p, shed_foot)
    p.shadow(15, 5, reach=4)
    _track(p, mill_foot[0], mill_foot[1], (0.68, 0.84), 3.5)
    _track(p, shed_foot[1], shed_foot[2], (0.4, 0.75), 3)
    _chimney(p, 11, 0, -32, shade(wall, 0.92))
    _smoke(p, 12, -35)
    top = p.hall(-4, -1, 14, 8, 15, wall, roof)
    left, front, right, _ = top
    _plinth(p, top, 15, wall)
    _stains(p, left, front, 15, (0.05, 0.24, 0.44, 0.94), shade(wall, 0.6), 2)
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], shade(wall, 1.2))
    p.line([_on_wall(front, right, 0, 1), _on_wall(front, right, 1, 1)], shade(wall, 0.95))
    p.line([left, front, right], shade(roof, 1.3))
    lintel = shade(wall, 1.3)
    _windows(p, left, front, (0.1, 0.3, 0.5, 0.7, 0.9), (3.5, 7, 10.5), 2, arch=lintel)
    _windows(p, left, front, (0.1, 0.3, 0.5), (14,), 1, arch=lintel)
    _windows(p, front, right, (0.3, 0.7), (3.5, 7), 2, shade(WINDOW, 0.75), lintel)
    p.line([_on_wall(left, front, 0.6, 1.5), _on_wall(left, front, 0.6, 14.6)], (70, 64, 66))
    _patch(p, left, front, (0.68, 0.84), (12, 15), (60, 44, 36))
    p.line([_on_wall(left, front, 0.66, 11.4), _on_wall(left, front, 0.86, 11.4)], lintel)
    shed = p.box(7, 4.5, 8, 5, wall, None)
    _plinth(p, shed, 5, wall)
    _stains(p, shed[1], shed[2], 5, (0.15, 0.9), shade(wall, 0.5), 1.2)
    _door(p, shed[1], shed[2], (0.4, 0.75), 2, 5, (60, 44, 36), shade(wall, 1.1))
    _sawtooth(p, shed, 3, 4, (104, 100, 108), (160, 196, 218), wall)
    _windows(p, shed[0], shed[1], (0.2, 0.5, 0.8), (2.5,), 1, arch=lintel)
    for x, y, colour in ((-10, 6, CLOTH[0]), (-6.5, 7.5, CLOTH[1]), (-10, 4, CLOTH[2])):
        _bale(p, x, y, colour)
    return p.sprite()


def furniture_maker() -> Sprite:
    """A joiner's plank-walled workshop with a window and a wide door, sawn boards stacked on
    bearers in its yard, a plank on a sawhorse over a heap of sawdust, a cart track worn out of its
    door and a chair standing on its roof for a sign."""
    p = Painter(30, 32, (14, 23))
    foot = _footing(-4, 0, 7, 7)
    _bed(p, foot)
    p.blob(1, -2, 15, 4, SOIL)
    p.shadow(12, 3.5)
    _track(p, foot[0], foot[1], (0.45, 0.82), 3)
    wall, board, roof = (176, 134, 88), (226, 190, 134), (98, 74, 58)
    for x, y, h in ((9, -3, 6), (7, 2, 4)):
        for t in (0.15, 0.85):
            bx, by = _on_wall(*_footing(x, y, 8, 3)[:2], t, 0)
            p.rect(bx - 0.5, by - 1, bx + 0.5, by + 0.5, (110, 80, 56))
        stack = p.hall(x, y - 0.8, 8, 3, h, board, shade(board, 1.12))
        gap = shade(board, 0.5)
        for drop in range(2, h, 2):
            p.line(
                [_on_wall(stack[0], stack[1], 0, drop), _on_wall(stack[0], stack[1], 1, drop)], gap
            )
            p.line(
                [_on_wall(stack[1], stack[2], 0, drop), _on_wall(stack[1], stack[2], 1, drop)], gap
            )
        p.point(*_on_wall(stack[1], stack[2], 0.5, h - 1), shade(board, 0.7))
    top = p.box(-4, 0, 7, 6, wall, None)
    left, front, right, _ = top
    for drop in (2, 4):
        _band(p, left, front, drop, shade(wall, 0.86))
        _band(p, front, right, drop, shade(wall, 0.68))
    _plinth(p, top, 6, wall)
    _stains(p, front, right, 6, (0.2, 0.55, 0.9), shade(wall, 0.55), 1.4)
    _window(p, left, front, (0.1, 0.32), 1.8)
    _door(p, left, front, (0.45, 0.82), 1.8, 6, (56, 40, 30), shade(wall, 1.3))
    p.line([_on_wall(left, front, 0.635, 1.9), _on_wall(left, front, 0.635, 6)], (88, 64, 44))
    _window(p, front, right, (0.3, 0.6), 1.8, shade(WINDOW, 0.8))
    p.gable(top, 4, roof, wall)
    _roof_rows(p, top, 4, roof)
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
    p.line([(x0 + 1, y0 - 3), (x0 + 5, y0 - 3)], shade(chair, 1.35))
    sawdust = (222, 196, 140)
    p.blob(-12.5, 6.5, -6.5, 8.5, sawdust)
    p.point(-11, 7, shade(sawdust, 1.1))
    leg = (110, 80, 56)
    for x in (-12, -8):
        p.line([(x - 1, 8), (x, 5), (x + 1, 8)], leg)
    p.line([(-13, 4), (-6, 6.5)], board)
    p.line([(-13, 5), (-6, 7.5)], shade(board, 0.7))
    return p.sprite()


def tool_works() -> Sprite:
    """A dark forge hall on sooty ground with its door open on a glowing hearth, an anvil on a
    board over it and a short banded smoking stack, a cinder track worn out of its door, with a
    heap of coal spilling over the ground, a quenching tub splashed round and a pile of iron bars
    outside."""
    p = Painter(30, 38, (14, 29))
    foot = _footing(-1, 0, 13, 9)
    _bed(p, foot, (62, 58, 56), 1.5)
    _spill(p, -9, 5, 3, (48, 46, 50))
    p.shadow(12, 4)
    _track(p, foot[0], foot[1], (0.52, 0.86), 3, (96, 88, 82))
    wall, roof = (92, 84, 82), (60, 60, 68)
    top = p.hall(-1, 0, 13, 9, 7, wall, None)
    left, front, right, _ = top
    _plinth(p, top, 7, wall)
    _stains(p, front, right, 7, (0.15, 0.5, 0.85), shade(wall, 0.55), 1.8)
    _patch(p, left, front, (0.52, 0.86), (2.5, 7), (238, 132, 40))
    _patch(p, left, front, (0.6, 0.78), (4.5, 7), (255, 208, 112))
    p.line([_on_wall(left, front, 0.5, 1.9), _on_wall(left, front, 0.88, 1.9)], shade(wall, 1.5))
    for t in (0.5, 0.88):
        p.line([_on_wall(left, front, t, 1.9), _on_wall(left, front, t, 7)], shade(wall, 0.6))
    _patch(p, left, front, (0.03, 0.47), (0.5, 6), (214, 202, 176))
    _patch(p, left, front, (0.03, 0.47), (5.2, 6), shade((214, 202, 176), 0.8))
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
    p.line([(cx - 2, cy - 1), (cx + 1, cy - 1)], (110, 110, 120))
    _windows(p, front, right, (0.3, 0.7), (3,), 1, (240, 150, 60), shade(wall, 1.5))
    p.gable(top, 4, roof, wall)
    _roof_rows(p, top, 4, roof)
    p.rect(-1, -17, 1, -10, (74, 58, 52))
    p.line([(-1, -15), (1, -15)], (200, 190, 180))
    p.rect(-1.5, -18, 1.5, -17, (50, 40, 36))
    p.point(0, -18, (255, 170, 60))
    _smoke(p, 1, -21, (96, 94, 98))
    _heap(p, -9, 5, 3, 2.5, (48, 46, 50))
    p.point(-10, 3.5, (104, 104, 116))
    _puddle(p, 4, 4.6, 2.2)
    p.ellipse(2, 2.5, 6, 3.5, (70, 96, 120))
    p.line([(2.5, 3.6), (5.5, 3.6)], (88, 64, 44))
    _barrel(p, 4, 8.5)
    _lengths(p, 9, 3, (120, 122, 130), 2)
    p.point(7, 5.5, (150, 92, 60))
    return p.sprite()


def winery() -> Sprite:
    """A low cream winery under red tiles, with shuttered windows, an arched door and a little
    chimney, a footpath worn to its door, a cellar door dug into a grassy bank, barrels outside on
    ground stained with spilt wine, and vines trained on trellises over rows of dug earth."""
    p = Painter(34, 30, (17, 19))
    foot = _footing(0, -2, 12, 7)
    _bed(p, foot)
    soil = (118, 96, 66)
    for start in ((-16, 1), (-13, 5)):
        x, y = start
        p.polygon([(x - 1, y), (x + 9, y + 4.5), (x + 9, y + 6), (x - 1, y + 1.5)], soil)
    p.blob(-3, 7.5, 5, 10.5, (112, 92, 70))
    p.shadow(13, 3.5)
    _track(p, foot[0], foot[1], (0.54, 0.7), 3.5, ruts=False)
    wall, roof = (238, 224, 192), (178, 86, 58)
    top = p.hall(0, -2, 12, 7, 5, wall, None)
    left, front, right, _ = top
    _plinth(p, top, 5, wall)
    _stains(p, left, front, 5, (0.05, 0.33, 0.9), shade(wall, 0.66), 1.2)
    p.gable(top, 3, roof, wall)
    _roof_rows(p, top, 3, roof, (0.5,))
    x, y = _on_roof(top, 3, 0.7, 0.9)
    p.rect(x - 0.5, y - 2, x + 0.5, y, shade(wall, 0.9))
    for t in (0.2, 0.45, 0.8):
        _window(p, left, front, (t - 0.03, t + 0.03), 1.6, tall=1.6)
        for side in (-0.07, 0.07):
            p.point(*_on_wall(left, front, t + side, 1.8), (90, 120, 70))
    _door(p, left, front, (0.56, 0.68), 2, 5, (110, 74, 50), shade(wall, 0.8))
    _window(p, front, right, (0.4, 0.6), 1.6)
    vine, grape, post = (78, 130, 50), (112, 50, 124), (110, 84, 58)
    for start in ((-16, 1), (-13, 5)):
        ends = [(start[0] + 3 * k, start[1] + 1.5 * k) for k in range(4)]
        for x, y in ends:
            p.line([(x, y), (x, y - 5)], post)
        p.line([(x, y - 4) for x, y in ends], vine)
        p.line([(x, y - 5) for x, y in ends], shade(vine, 1.25))
        for x, y in ends[:-1]:
            p.point(x + 1.5, y - 2.5, grape)
            p.point(x + 1, y - 3.5, shade(vine, 0.8))
    cellar = p.box(10, 3, 4, 3, (176, 168, 150), (104, 140, 70))
    _patch(p, cellar[0], cellar[1], (0.3, 0.75), (1, 3), (42, 34, 30))
    p.line(
        [_on_wall(cellar[0], cellar[1], 0.25, 0.5), _on_wall(cellar[0], cellar[1], 0.8, 0.5)],
        shade(STONE, 1.3),
    )
    p.point(*_on_wall(cellar[0], cellar[3], 0.5, -0.5), (134, 170, 90))
    for x, y in ((3, 6), (6, 7.5)):
        _barrel(p, x, y)
    p.ellipse(-2, 7, 3, 10, BARREL)
    for x in (-0.5, 1.5):
        p.line([(x, 7), (x, 10)], shade(BARREL, 0.55))
    p.blob(-2.5, 9.2, 0, 10.4, (110, 52, 76))
    return p.sprite()


def chemical_works() -> Sprite:
    """A ball tank on legs, banded pale tanks with green tops and a ladder, and a pipe rack with a
    valve beside a small control house, all on a concrete pad stained with spills, a path worn to
    its door, drums of chemicals outside and green fumes rising from a red-banded vent stack."""
    p = Painter(32, 47, (15, 37))
    pad = (156, 156, 148)
    p.polygon([(-15, -4), (2, -12), (15, -3), (-1, 6)], pad, shade(pad, 0.8))
    foot = _footing(-3, 4, 5, 5)
    _bed(p, foot, margin=1)
    p.blob(-6, -4, -1, -1.5, (142, 168, 92))
    p.blob(9, 2, 13, 4, (120, 144, 98))
    p.shadow(13, 4)
    _track(p, foot[0], foot[1], (0.58, 0.86), 2.5, ruts=False)
    body, green = (208, 214, 198), (104, 176, 84)
    p.rect(10, -26, 11, -2, METAL)
    for y in (-25, -22):
        p.line([(10, y), (11, y)], (214, 58, 54))
    for i, (x, y, r) in enumerate(((11, -28, 1.5), (12, -31, 2), (14, -34, 2.5))):
        p.blob(x - r, y - r, x + r, y + r, (178, 222, 120) if i % 2 else (146, 204, 96))
    _turned(p, 4, -4, [(0, 3), (13, 3)], body, cap=green)
    for h in (4, 9):
        _ring(p, 4, -4 - h, 3, shade(body, 0.75))
    p.line([(6, -5), (6, -17)], shade(METAL, 0.7))
    for y in (-7, -10, -13, -16):
        p.line([(5.5, y), (6.5, y)], shade(METAL, 0.7))
    for x in (-11, -9, -6):
        p.line([(x, 0), (x, -6)], shade(METAL, 0.7))
        p.point(x, 0, shade(METAL, 0.5))
    p.line([(-11, -3), (-6, -5)], shade(METAL, 0.6))
    p.ellipse(-13, -15, -4, -6, body)
    p.blob(-11, -13, -8, -10, shade(body, 1.15))
    p.line([(-13, -10), (-4, -10)], green)
    p.line([(-4, -9), (2, -9)], shade(METAL, 0.8))
    p.line([(-4, -7), (8, -7)], green)
    p.ellipse(-2, -10.5, 0, -8.5, (196, 60, 52))
    for x in (-3, 1):
        p.line([(x, -7), (x, -3)], shade(METAL, 0.7))
    _turned(p, 8, 1, [(0, 2.5), (9, 2.5)], body, cap=green)
    _ring(p, 8, -4, 2.5, shade(body, 0.75))
    house = (196, 200, 186)
    top = p.box(-3, 4, 5, 4, house, (96, 116, 84))
    left, front, right, _ = top
    _plinth(p, top, 4, house)
    _stains(p, front, right, 4, (0.25, 0.75), shade(house, 0.55), 1.2)
    _patch(p, left, front, (0.15, 0.4), (1.2, 2.4), (110, 200, 120))
    _door(p, left, front, (0.6, 0.85), 1.2, 4, DOORWAY)
    p.point(*_on_wall(front, right, 0.5, 1.5), (236, 196, 60))
    p.blob(-13, 7.5, -9, 9.5, (130, 150, 90))
    for x, y, colour in (
        (-11, 6, (64, 108, 196)),
        (-8.5, 7.5, (236, 196, 60)),
        (-12, 8.5, (64, 108, 196)),
    ):
        _drum(p, x, y, colour)
    return p.sprite()


def electronics_plant() -> Sprite:
    """A modern box of white panels and bands of cyan glass standing on a paved apron, with a lit
    sign and a canopy over its door, a paved walk out to flags flying before it on a trimmed
    lawn, and a dish, an aerial and plant boxes on its flat roof."""
    p = Painter(32, 38, (15, 28))
    foot = _footing(0, 0, 15, 9)
    paving = (186, 188, 184)
    _bed(p, foot, paving, 1.6)
    p.shadow(13, 4)
    _track(p, foot[0], foot[1], (0.4, 0.6), 4.5, paving, ruts=False)
    p.line([_on_wall(*foot[:2], 0.4, 0), (-8.2, 4.4)], shade(paving, 0.8))
    p.blob(-14, 3, -7, 7, (86, 150, 64))
    wall, glass, accent = (226, 232, 236), (84, 176, 206), (80, 216, 222)
    top = p.hall(0, 0, 15, 9, 8, wall, (172, 180, 190))
    left, front, right, _ = top
    _plinth(p, top, 8, wall)
    _band(p, left, front, 7.9, (120, 124, 128))
    _band(p, front, right, 7.9, (104, 108, 112))
    for d0, d1 in ((1.5, 3), (4.5, 6.5)):
        _patch(p, left, front, (0.04, 0.96), (d0, d1), glass)
        _patch(p, front, right, (0.06, 0.94), (d0, d1), shade(glass, 0.72))
        for t in (0.28, 0.72):
            p.line([_on_wall(left, front, t, d0), _on_wall(left, front, t, d1)], shade(glass, 1.3))
        p.line([_on_wall(front, right, 0.5, d0), _on_wall(front, right, 0.5, d1)], glass)
        p.line([_on_wall(left, front, 0.1, d0), _on_wall(left, front, 0.2, d0)], (200, 240, 250))
    p.line([_on_wall(left, front, 0.04, 4.5), _on_wall(left, front, 0.96, 4.5)], accent)
    _patch(p, left, front, (0.42, 0.58), (6.5, 8), (52, 70, 84))
    p.line([_on_wall(left, front, 0.38, 6.2), _on_wall(left, front, 0.62, 6.2)], (120, 130, 140))
    _patch(p, left, front, (0.4, 0.6), (3.3, 4.2), accent)
    p.line([left, front, right], shade(wall, 1.1))
    p.line([(-4, -8), (-4, -11)], METAL)
    p.ellipse(-8, -16, -1, -11, (236, 238, 240))
    p.blob(-6, -15, -3, -13, (176, 186, 196))
    p.point(-4, -14, accent)
    p.line([(5, -7), (5, -18)], METAL)
    p.line([(3, -15), (7, -15)], METAL)
    p.point(5, -19, (230, 60, 60))
    p.rect(7, -11, 9, -9, (190, 198, 206))
    p.rect(10, -9, 12, -7, (190, 198, 206))
    p.point(8, -10, (120, 130, 140))
    for x, y in ((-12, 4), (-9, 5.5)):
        _flag(p, x, y, 9, accent if x < -10 else (64, 108, 196))
    return p.sprite()


def machine_works() -> Sprite:
    """A big brick engine hall over a dark plinth stained by the damp, with tall arched windows, a
    glazed roof light, a drainpipe and a great gear on its end wall, a cart track worn out of its
    door, beside a gravelled yard where a gantry crane runs on rails lifting a casting over oil
    stains, and pipes piled up."""
    p = Painter(38, 42, (18, 32))
    foot = _footing(-2, -2, 16, 9)
    _bed(p, foot)
    gravel = (140, 134, 120)
    p.polygon([(9, 6.5), (16, 3), (19, 4.5), (12, 8)], gravel)
    p.shadow(16, 5)
    _track(p, foot[0], foot[1], (0.78, 0.96), 4)
    for dx, dy in ((0, 0), (5.6, -2.8)):
        p.line([(10.4 + dx, 6.9 + dy), (13 + dx, 8.2 + dy)], (96, 92, 90))
    _oil(p, 14, 5.4, 1.2)
    top = p.hall(-2, -2, 16, 9, 10, BRICK, None)
    left, front, right, _ = top
    lintel = shade(BRICK, 1.3)
    _plinth(p, top, 10, BRICK)
    _stains(p, left, front, 10, (0.04, 0.21, 0.39, 0.57, 0.73), shade(BRICK, 0.62), 2)
    _stains(p, front, right, 10, (0.2, 0.8), shade(BRICK, 0.5), 2)
    _band(p, left, front, 1, lintel)
    _band(p, front, right, 1, shade(BRICK, 1.05))
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2.5,), 5, arch=lintel)
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (8,), 1, shade(BRICK, 1.15))
    p.line([_on_wall(left, front, 0.76, 1.4), _on_wall(left, front, 0.76, 9.6)], (74, 64, 62))
    _door(p, left, front, (0.8, 0.94), 5, 10, (58, 48, 44), lintel)
    p.line([_on_wall(left, front, 0.87, 5), _on_wall(left, front, 0.87, 10)], (44, 36, 34))
    p.gable(top, 5, (82, 84, 96), BRICK)
    _skylight(p, top, 5, (0.08, 0.92))
    _gear(p, *_on_wall(front, right, 0.5, 5), 2.5, (222, 180, 74))
    _lengths(p, -14, 3, (96, 100, 112))
    crane = (214, 168, 40)
    feet, height = ((11, 6), (17, 3)), 15
    for x, y in feet:
        p.line([(x, y), (x, y - height)], crane)
        p.line([(x - 1, y), (x, y - 3)], shade(crane, 0.7))
        p.rect(x - 1, y - 0.5, x + 1, y + 0.5, (70, 70, 76))
    p.line([(feet[0][0] - 1, feet[0][1] - height), (feet[1][0] + 1, feet[1][1] - height)], crane)
    p.line(
        [(feet[0][0] - 1, feet[0][1] - height + 1), (feet[1][0] + 1, feet[1][1] - height + 1)],
        shade(crane, 0.7),
    )
    p.rect(13, -9.5, 15, -8, (60, 64, 74))
    p.line([(14, -8), (14, -4)], (70, 64, 58))
    p.rect(12, -4, 16, -1, (88, 90, 98))
    p.line([(12.5, -3), (15.5, -3)], shade((88, 90, 98), 1.3))
    return p.sprite()


def brickworks() -> Sprite:
    """A round Hoffmann kiln ringed with arched firing doors sooted above, around its tall banded
    chimney, feed holes on its roof, a barrow track worn from its front door to a heap of clay
    spreading over the ground, and stacks of new red bricks on ground red with brick dust."""
    p = Painter(34, 54, (16, 44))
    kiln = (152, 98, 76)
    x, y, r = -2, -1, 10
    p.blob(x - r - 2, y - r / 2 - 1, x + r + 2, y + r / 2 + 1.5, SOIL)
    p.blob(6, 1, 16, 7.5, (146, 104, 82))
    _spill(p, -12, 7, 3, (176, 122, 86))
    p.shadow(14, 5)
    _track(p, (-5, 2.5), (1, 5.5), (0.3, 0.6), 4)
    _turned(p, x, y, [(0, r), (6, r)], kiln)
    _ring(p, x, y - 0.8, r, shade(kiln, 0.7))
    for a in (0.35, 0.95, 1.55, 2.15, 2.75):
        ax, ay = x + r * math.cos(a), y + r / 2 * math.sin(a)
        p.rect(ax - 1, ay - 4, ax + 1, ay, (52, 36, 32))
        p.point(ax - 0.5, ay - 5, shade(kiln, 1.35))
        p.line([(ax, ay - 5.5), (ax, ay - 6)], shade(kiln, 0.62))
    _turned(p, x, y - 6, [(0, r + 1), (3, r - 4)], (96, 86, 86), cap=(118, 106, 104))
    for a in (0.5, 1.2, 1.9, 2.6):
        p.point(x + (r - 1) * math.cos(a), y - 7.5 + (r - 1) / 2 * math.sin(a), (60, 54, 54))
    _chimney(p, x, y - 8, y - 35, shade(BRICK, 0.9))
    _smoke(p, x + 1, y - 38, puffs=2)
    _heap(p, -12, 7, 3, 2.5, (176, 122, 86))
    for bx, by in ((13, 2), (8.5, 5.5)):
        stack = p.hall(bx, by, 4, 3, 3, BRICK, shade(BRICK, 1.25))
        for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
            p.line([_on_wall(a, b, 0, 1.5), _on_wall(a, b, 1, 1.5)], shade(BRICK, 1.3))
            p.line([_on_wall(a, b, 0.5, 0), _on_wall(a, b, 0.5, 1.5)], shade(BRICK, 0.7))
            p.line([_on_wall(a, b, 0.25, 1.5), _on_wall(a, b, 0.25, 3)], shade(BRICK, 0.7))
            p.line([_on_wall(a, b, 0.75, 1.5), _on_wall(a, b, 0.75, 3)], shade(BRICK, 0.7))
    for px, py in ((5, 8), (10, 8.5)):
        p.point(px, py, BRICK)
    return p.sprite()


def cement_works() -> Sprite:
    """Dusty grey banded silos joined to a windowed preheater tower by a conveyor, with a long
    rotary kiln tube on piers before them glowing at its firing end and bags of cement by it, all
    on ground whitened by the dust, with lorry ruts worn through it."""
    p = Painter(40, 50, (20, 40))
    grey, dust = (190, 188, 182), (214, 212, 206)
    ground = (176, 174, 164)
    p.polygon([(-19, 0), (-6, -9), (15, -7), (17, 2), (2, 9), (-14, 8)], ground)
    p.shadow(16, 5)
    _track(p, (4, 5), (13, 0.5), (0.25, 0.7), 4, shade(ground, 0.84))
    tower = p.box(9, -6, 3.5, 22, shade(grey, 0.9), shade(grey, 0.75))
    for drop in (4, 9, 14, 19):
        p.line(
            [_on_wall(tower[0], tower[1], 0, drop), _on_wall(tower[0], tower[1], 1, drop)],
            shade(grey, 0.6),
        )
        p.point(*_on_wall(tower[0], tower[1], 0.5, drop - 2.5), WINDOW)
        p.point(*_on_wall(tower[1], tower[2], 0.5, drop - 2.5), shade(WINDOW, 0.8))
    _plinth(p, tower, 22, shade(grey, 0.9))
    _patch(p, tower[1], tower[2], (0.3, 0.7), (19.5, 22), DOORWAY)
    p.rect(8, -32, 10, -28, shade(grey, 0.7))
    _smoke(p, 9, -35, dust, 2)
    p.line([(-5, -22), (5.5, -22)], shade(grey, 0.6))
    p.line([(-5, -21), (5.5, -21)], shade(grey, 0.8))
    for k in range(4):
        p.point(-3.5 + 2.5 * k, -21.5, shade(grey, 0.5))
    for sx, sy in ((-6, -7), (-10, -5), (-14, -3)):
        _turned(p, sx, sy, [(0, 2.5), (16, 2.5)], grey, cap=shade(grey, 1.1))
        for h in (5, 11):
            _ring(p, sx, sy - h, 2.5, shade(grey, 0.75))
        _ring(p, sx, sy - 0.8, 2.5, shade(grey, 0.62))
    p.line([(-14, -19.5), (-6, -23.5)], shade(grey, 0.55))
    tube = (134, 128, 120)
    a, b = (-12, 5), (6, -4)
    for t in (0.15, 0.5, 0.85):
        px, py = _on_wall(a, b, t, 0)
        p.blob(px - 2, py - 0.5, px + 2, py + 1.5, shade(ground, 0.8))
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
    p.blob(a[0] - 1.5, a[1] - lift - 2.5, a[0] + 0.5, a[1] - lift - 0.5, (240, 132, 40))
    for x, y, r in ((-14, 4, 2), (-10, 7, 1.5), (-16, 1, 1.5)):
        p.blob(x - r, y - r, x + r, y + r, dust)
    for x, y in ((-12.5, 5.5), (-15, 2.5)):
        p.point(x, y, shade(dust, 1.08))
    for x, y in ((8, 5), (11, 4), (9.5, 2.8)):
        _sack(p, x, y, (200, 198, 190))
    return p.sprite()


def glassworks() -> Sprite:
    """A tall banded brick glass cone sooted at its top, its furnace mouth glowing under an arch at
    the foot with embers on a trodden ash path out of it, beside a shed with a door and window, a
    path worn from its door, sheets of new glass leant against it, a crate of green bottles and a
    heap of broken glass glinting where it spills over the ground."""
    p = Painter(32, 50, (15, 40))
    cone = (162, 88, 64)
    x = -3
    p.blob(x - 11, -5.5, x + 11, 5.5, SOIL)
    shed = (196, 180, 156)
    shed_foot = _footing(10, 2, 4, 4)
    _bed(p, shed_foot, margin=1)
    _spill(p, -12, 6, 2.5, (170, 210, 196))
    p.shadow(13, 4)
    _track(p, (-7, 1.5), (-1, 4.5), (0.15, 0.6), 3, (112, 100, 92), ruts=False)
    _track(p, shed_foot[1], shed_foot[2], (0.3, 0.7), 2.5, ruts=False)
    _turned(p, x, 0, [(0, 9), (7, 7.5), (20, 4), (28, 3)], cone, cap=(54, 40, 36))
    for h, half in ((9, 7), (16, 5.1), (24, 3.5)):
        _ring(p, x, -h, half, shade(cone, 0.7))
    for h in (26, 27):
        _ring(p, x, -h, 3.2, shade(cone, 0.5))
    _smoke(p, x + 1, -31)
    p.line([(x - 4, -6), (x - 1, -7.5), (x + 2, -6)], shade(cone, 1.35))
    p.ellipse(x - 3, -6, x + 1, 0, (240, 132, 40))
    p.rect(x - 3, -3, x + 1, 2, (240, 132, 40))
    p.blob(x - 2, -3, x, 1, (255, 214, 120))
    for gx, gy in ((x - 4, 4), (x - 1, 4), (x - 3, 5), (x - 5, 5.5)):
        p.point(gx, gy, (236, 150, 60))
    top = p.box(10, 2, 4, 4, shed, None)
    _plinth(p, top, 4, shed)
    _stains(p, top[1], top[2], 4, (0.15, 0.85), shade(shed, 0.55), 1.2)
    _window(p, top[0], top[1], (0.3, 0.7), 1.2, tall=1.2)
    _door(p, top[1], top[2], (0.3, 0.7), 1.2, 4, DOORWAY, shade(shed, 1.15))
    p.gable(top, 2, (96, 84, 80), shed)
    pane = (176, 222, 234)
    for k in range(3):
        px, py = 5 + 2 * k, 7 - k
        p.polygon(
            [(px, py), (px + 1, py - 5), (px + 3, py - 6), (px + 2, py - 1)], pane, shade(pane, 0.6)
        )
        p.point(px + 1, py - 3, (246, 252, 255))
    crate = (150, 112, 74)
    _crate(p, 14, 7, crate)
    for dx in (-1, 0.5):
        p.line([(14 + dx, 4.2), (14 + dx, 2.6)], (70, 150, 90))
    _heap(p, -12, 6, 2.5, 2, (170, 210, 196))
    for gx, gy in ((-13, 5), (-11, 4.5), (-15, 6.5)):
        p.point(gx, gy, (236, 252, 250))
    return p.sprite()


def brewery() -> Sprite:
    """A brick brewhouse over a dark plinth with arched windows and door, a copper-domed tower with
    a loading door and a dray track worn out of it, barrels at its door on a path wet with spilt
    beer, a heap of spent grain and a hop cone for a sign."""
    p = Painter(32, 44, (15, 34))
    roof = (84, 76, 80)
    lintel = shade(BRICK, 1.3)
    foot = _footing(-4, -1, 12, 8)
    tower_foot = _footing(5, 3, 4, 4)
    _bed(p, foot)
    _bed(p, tower_foot)
    p.shadow(13, 4)
    _track(p, foot[0], foot[1], (0.58, 0.73), 3.5, ruts=False)
    _track(p, tower_foot[1], tower_foot[2], (0.25, 0.75), 3.5)
    _puddle(p, -5, 7.5, 1.6)
    top = p.hall(-4, -1, 12, 8, 8, BRICK, None)
    left, front, *_ = top
    _plinth(p, top, 8, BRICK)
    _stains(p, left, front, 8, (0.04, 0.38, 0.9), shade(BRICK, 0.62), 1.8)
    _windows(p, left, front, (0.45, 0.65, 0.85), (2,), 2, arch=lintel)
    _windows(p, left, front, (0.45, 0.85), (5,), 2, arch=lintel)
    _door(p, left, front, (0.6, 0.71), 5, 8, DOORWAY, lintel)
    _patch(p, left, front, (0.04, 0.34), (1, 5), (222, 210, 180))
    _patch(p, left, front, (0.04, 0.34), (1, 1.5), shade(BRICK, 0.8))
    for drop in (3.6, 4.2):
        p.line(
            [_on_wall(left, front, 0.08, drop), _on_wall(left, front, 0.3, drop)], (150, 130, 96)
        )
    hop = (110, 170, 70)
    hx, hy = _on_wall(left, front, 0.19, 2.6)
    p.polygon(
        [(hx, hy - 1.5), (hx + 1.5, hy), (hx, hy + 1.5), (hx - 1.5, hy)], hop, shade(hop, 0.6)
    )
    p.gable(top, 4, roof, BRICK)
    _roof_rows(p, top, 4, roof)
    x, y = _on_roof(top, 4, 0.25, 0.85)
    _stack(p, x - 1, y, y - 4, BRICK)
    tower = shade(BRICK, 1.08)
    tower_top = p.box(5, 3, 4, 15, tower, None)
    _plinth(p, tower_top, 15, tower)
    _stains(p, tower_top[1], tower_top[2], 15, (0.1, 0.9), shade(tower, 0.5), 1.6)
    p.line([(1, -12), (5, -10), (9, -12)], shade(tower, 1.25))
    _windows(p, (1, -12), (5, -10), (0.5,), (3, 8), 2, arch=shade(BRICK, 1.35))
    _windows(p, (5, -10), (9, -12), (0.5,), (3,), 2, shade(WINDOW, 0.8), shade(BRICK, 1.1))
    _door(p, tower_top[1], tower_top[2], (0.25, 0.75), 11, 15, (70, 50, 40), shade(BRICK, 1.1))
    _turned(p, 5, -12, [(0, 4.5), (2, 4.2), (4, 3), (5.5, 1.5), (6, 0)], COPPER)
    _ring(p, 5, -13.5, 4.3, shade(COPPER, 0.7))
    p.blob(2, -16, 4, -14, shade(COPPER, 1.3))
    p.line([(5, -18), (5, -20)], (90, 80, 70))
    _heap(p, 13, 6.5, 2.2, 1.6, (176, 150, 92))
    for x, y in ((-7, 6), (-3, 8)):
        _barrel(p, x, y)
    return p.sprite()


def distillery() -> Sprite:
    """A whitewashed still house on a stone plinth, its foot greyed by the damp, with a
    pagoda-roofed kiln vent, a cart track worn out of its door past a stack of peat, a copper pot
    still at its side over a wet patch, casks outside and a tall thin banded stack."""
    p = Painter(32, 50, (15, 41))
    foot = _footing(-4, -1, 12, 7)
    _bed(p, foot)
    peat = (112, 84, 60)
    _spill(p, -11, 6, 2.5, peat)
    p.shadow(13, 4)
    _track(p, foot[0], foot[1], (0.48, 0.68), 3.5)
    _puddle(p, 9, 7.5, 1.6)
    slate = (68, 72, 82)
    p.rect(3, -33, 4, -4, (150, 80, 60))
    for y in (-31, -20):
        p.line([(3, y), (4, y)], (230, 228, 222))
    p.rect(2.5, -34, 4.5, -33, (90, 52, 42))
    _smoke(p, 4, -36, (190, 190, 194), 2)
    top = p.hall(-4, -1, 12, 7, 6, WHITEWASH, None)
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (5, 6), STONE)
    _patch(p, front, right, (0, 1), (5, 6), shade(STONE, 0.78))
    for t in (0.1, 0.3, 0.55, 0.75, 0.95):
        p.point(*_on_wall(left, front, t, 5.4), shade(STONE, 0.75))
    _stains(p, left, front, 5, (0.05, 0.25, 0.45, 0.72, 0.92), (204, 204, 194), 1.4)
    _stains(p, front, right, 5, (0.2, 0.75), shade(WHITEWASH, 0.66), 1.4)
    _windows(p, left, front, (0.15, 0.35, 0.8), (2,), 2, (44, 46, 52))
    for t in (0.15, 0.35, 0.8):
        p.point(*_on_wall(left, front, t, 4), shade(STONE, 0.9))
    _door(p, left, front, (0.5, 0.66), 2.5, 6, (60, 48, 42), shade(WHITEWASH, 0.8))
    _windows(p, front, right, (0.5,), (2,), 2, (44, 46, 52))
    p.gable(top, 4, slate, WHITEWASH)
    _roof_rows(p, top, 4, slate, (0.5,))
    vx, vy = _on_roof(top, 4, 0.35, 1)
    p.rect(vx - 1.5, vy - 2, vx + 1.5, vy + 0.5, WHITEWASH)
    p.line([(vx - 1, vy - 1), (vx + 1, vy - 1)], shade(WHITEWASH, 0.6))
    p.polygon(
        [(vx - 3, vy - 1.5), (vx - 1, vy - 3), (vx, vy - 5), (vx + 1, vy - 3), (vx + 3, vy - 1.5)],
        slate,
        shade(slate, 0.6),
    )
    p.point(vx, vy - 6, (60, 60, 64))
    _turned(p, 8, 5, [(0, 3), (2, 4), (4, 3.5), (6, 1.5), (7, 1)], COPPER)
    p.blob(5, -1, 7, 2, shade(COPPER, 1.3))
    _ring(p, 8, 2, 3.6, shade(COPPER, 0.7))
    p.line([(8, -2), (8, -4), (12, -2)], shade(COPPER, 0.8))
    p.line([(12, -2), (12, 0)], shade(COPPER, 0.8))
    _heap(p, -11, 6, 2.5, 2, peat)
    _barrel(p, 12, 4)
    p.ellipse(9, 6, 13, 8.5, BARREL)
    p.line([(11, 6), (11, 8.5)], shade(BARREL, 0.55))
    return p.sprite()


def tannery() -> Sprite:
    """Low tanning sheds with doors, windows, slatted drying lofts and dark plinths, hides
    stretched to dry on wooden frames over drip-stained ground, a pit of tan liquor sunk in a
    sodden patch, a heap of oak bark spreading over the ground and a track worn out of the near
    shed's door."""
    p = Painter(34, 30, (17, 18))
    near = _footing(-7, -2, 10, 5)
    _bed(p, _footing(-2, -6, 10, 5))
    _bed(p, near)
    p.blob(-8, 3, 2, 8, (92, 84, 56))
    _spill(p, -12, 7, 3, (140, 92, 58))
    p.shadow(12, 3.5)
    _track(p, near[0], near[1], (0.3, 0.7), 2.5)
    wall, roof = (184, 160, 124), (96, 82, 74)
    for x, y in ((-2, -6), (-7, -2)):
        top = p.hall(x, y, 10, 5, 4, wall, None)
        _plinth(p, top, 4, wall, 0.7)
        _stains(p, top[1], top[2], 4, (0.2, 0.8), shade(wall, 0.52), 1)
        for t in (0.1, 0.18):
            p.line(
                [_on_wall(top[0], top[1], t, 0.5), _on_wall(top[0], top[1], t, 2.6)], (110, 92, 70)
            )
        _patch(p, top[0], top[1], (0.3, 0.7), (1, 4), (70, 56, 44))
        p.line([_on_wall(top[0], top[1], 0.5, 1), _on_wall(top[0], top[1], 0.5, 4)], (54, 42, 34))
        _window(p, top[0], top[1], (0.78, 0.92), 1.2)
        _window(p, top[1], top[2], (0.35, 0.65), 1.2, shade(WINDOW, 0.8))
        p.gable(top, 3, roof, wall)
        _roof_rows(p, top, 3, roof, (0.5,))
        ridge_back, ridge_front = _ridge(top, 3)
        p.line(
            [
                _on_wall(ridge_back, ridge_front, 0.3, -0.6),
                _on_wall(ridge_back, ridge_front, 0.7, -0.6),
            ],
            shade(roof, 0.7),
        )
    p.ellipse(-6, 4, 0, 7, (138, 116, 86))
    p.blob(-5, 4.5, -1, 6.5, (96, 66, 40))
    p.point(-4, 5, (140, 100, 62))
    _heap(p, -12, 7, 3, 2.5, (140, 92, 58))
    frame = (104, 76, 50)
    for k, hide in enumerate(((198, 150, 100), (162, 108, 68), (214, 180, 130))):
        x, y = 2 + 4 * k, 6 - 2 * k
        p.blob(x + 0.5, y - 0.5, x + 4, y + 1, (100, 108, 60))
        p.line([(x, y), (x, y - 7)], frame)
        p.line([(x + 4, y - 1), (x + 4, y - 8)], frame)
        p.line([(x, y - 7), (x + 4, y - 8)], frame)
        p.polygon(
            [(x + 1, y - 6), (x + 3, y - 7), (x + 3.5, y - 2), (x + 2, y - 1.5), (x + 0.5, y - 1)],
            hide,
            shade(hide, 0.65),
        )
        p.line([(x + 1, y - 6), (x + 3, y - 7)], frame)
    return p.sprite()


def shoe_factory() -> Sprite:
    """A narrow workshop of four storeys with cornices between its floors, windows with sills, a
    shop window and door on the ground floor, a chimney, a pavement of flags along its front with
    a worn path to the door, boxes of shoes at its foot and a big boot hung out from its wall as
    a sign."""
    p = Painter(24, 38, (12, 29))
    foot = _footing(1, 0, 6, 6)
    paving = (176, 170, 158)
    _bed(p, foot, paving, 2.2)
    _bed(p, foot, SOIL, 0.8)
    for t in (0.3, 0.6):
        p.line(
            [_on_wall(foot[0], foot[1], t, 0.8), _on_wall(foot[0], foot[1], t, 2)],
            shade(paving, 0.8),
        )
    p.shadow(8, 2.5)
    _track(p, foot[0], foot[1], (0.55, 0.8), 2.4, shade(paving, 0.9), ruts=False)
    wall, roof = (204, 170, 116), (124, 66, 52)
    s, floors = 6, 4
    top = p.box(1, 0, s, 4 * floors, wall, None)
    left, front, right, _ = top
    _plinth(p, top, 4 * floors, wall)
    _stains(p, front, right, 4 * floors, (0.15, 0.5, 0.85), shade(wall, 0.58), 1.5)
    for f in range(1, floors):
        _band(p, left, front, 4 * f - 0.5, shade(wall, 1.15))
        _band(p, front, right, 4 * f - 0.5, shade(wall, 0.85))
    for f in range(floors - 1):
        for t in (0.3, 0.7):
            _window(p, left, front, (t - 0.07, t + 0.07), 1.2 + 4 * f, tall=1.6)
    for f in range(floors):
        for t in (0.3, 0.7):
            _window(p, front, right, (t - 0.07, t + 0.07), 1.2 + 4 * f, shade(WINDOW, 0.75), 1.6)
    _patch(p, left, front, (0.1, 0.42), (12.8, 14.8), GLASS)
    p.line([_on_wall(left, front, 0.15, 13.4), _on_wall(left, front, 0.2, 13.2)], (220, 236, 244))
    _patch(p, left, front, (0.08, 0.44), (12.2, 12.8), (124, 66, 52))
    _door(p, left, front, (0.55, 0.8), 12.5, 16, (84, 60, 46))
    p.point(*_on_wall(left, front, 0.75, 14.2), (226, 196, 120))
    p.gable(top, 4, roof, wall)
    _roof_rows(p, top, 4, roof)
    x, y = _on_roof(top, 4, 0.7, 0.9)
    p.rect(x - 1, y - 2.5, x, y, BRICK)
    p.line([(x - 1.5, y - 3), (x + 0.5, y - 3)], shade(BRICK, 0.55))
    p.line([(-5, -12), (-8, -12)], (60, 56, 54))
    boot = (140, 54, 40)
    p.polygon(
        [(-9, -11), (-5, -11), (-5, -3), (-11, -3), (-11, -5), (-10, -6), (-9, -6)],
        boot,
        shade(boot, 0.5),
    )
    p.line([(-8, -10), (-8, -6)], shade(boot, 1.4))
    p.line([(-11, -3), (-5, -3)], (40, 32, 30))
    for x, y in ((6, 5), (8.5, 4)):
        _crate(p, x, y, (196, 160, 112))
    return p.sprite()


def plastics_works() -> Sprite:
    """White banded tanks with red and blue tops and a ladder standing on a concrete pad, and a
    white hall on a grey plinth under a blue roof with a door and tyre ruts worn out of it, tied
    together by coloured pipes, under a red-banded stack, with drums on a stained patch and sacks
    of pellets outside, a few pellets spilt round them."""
    p = Painter(32, 40, (15, 31))
    pad = (170, 170, 164)
    p.polygon([(-1, -2), (8, -7), (16, 2), (7, 7)], pad, shade(pad, 0.8))
    foot = _footing(-5, 2, 7, 7)
    _bed(p, foot)
    p.blob(-13, 6.5, -6.5, 8.6, (110, 116, 84))
    p.shadow(13, 4)
    _track(p, foot[1], foot[2], (0.3, 0.65), 3)
    white, red, blue = (232, 234, 236), (214, 58, 54), (64, 108, 196)
    p.rect(-2, -29, 0, -4, white)
    for y in (-27, -24):
        p.line([(-2, y), (0, y)], red)
    p.rect(-2.5, -30, 0.5, -29, (70, 70, 76))
    _turned(p, 6, -3, [(0, 3), (12, 3)], white, cap=red)
    _ring(p, 6, -9, 3, shade(white, 0.75))
    _ring(p, 6, -3.6, 3, shade(white, 0.62))
    p.line([(8, -4), (8, -15)], shade(METAL, 0.7))
    for y in (-6, -9, -12):
        p.line([(7.5, y), (8.5, y)], shade(METAL, 0.7))
    _turned(p, 11, 2, [(0, 2.5), (8, 2.5)], white, cap=blue)
    _ring(p, 11, -2, 2.5, shade(white, 0.75))
    _ring(p, 11, 1.4, 2.5, shade(white, 0.62))
    top = p.box(-5, 2, 7, 5, white, blue)
    left, front, right, _ = top
    _plinth(p, top, 5, white)
    _stains(p, front, right, 5, (0.15, 0.85), shade(white, 0.6), 1.2)
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], red)
    _windows(p, left, front, (0.25, 0.5, 0.75), (2.5,), 1, shade(blue, 0.8))
    _door(p, front, right, (0.3, 0.65), 2, 5, (70, 74, 84))
    for drop in (3, 4):
        p.line(
            [_on_wall(front, right, 0.3, drop), _on_wall(front, right, 0.65, drop)], (90, 94, 104)
        )
    p.line([left, front, right], shade(blue, 1.3))
    p.line([(1, -5), (3, -6), (3, -10)], blue)
    p.line([(2, -2), (8, -2), (9, -4)], red)
    p.point(5, -2, shade(red, 0.6))
    for x, y, colour in ((-11, 7, blue), (-8.5, 8, red)):
        _drum(p, x, y, colour)
    for x, y in ((5, 7.5), (8, 6.5)):
        _sack(p, x, y, (236, 236, 240))
    for x, y, colour in ((6.5, 8.4, white), (3.5, 8, blue), (9.5, 7.6, red)):
        p.point(x, y, colour)
    return p.sprite()


def appliance_factory() -> Sprite:
    """A long modern factory on a grey plinth standing on a concrete apron, with a big roller door
    and tyre marks run out of it, a row of windows, a blue sign and plant boxes on its roof, and
    its fridges and washing machines stacked outside in white."""
    p = Painter(36, 35, (17, 23))
    foot = _footing(1, -3, 18, 8)
    apron = (180, 180, 174)
    _bed(p, foot, apron, 2)
    _bed(p, foot, (120, 120, 116), 0.7)
    p.shadow(15, 4)
    _track(p, foot[0], foot[1], (0.55, 0.88), 5, apron)
    _oil(p, -4, 4, 1.2)
    wall, stripe = (216, 220, 224), (64, 116, 196)
    top = p.hall(1, -3, 18, 8, 7, wall, (150, 160, 174))
    left, front, right, back = top
    _plinth(p, top, 7, wall)
    _stains(p, front, right, 7, (0.2, 0.5, 0.8), shade(wall, 0.62), 1.2)
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], stripe)
    p.line([_on_wall(front, right, 0, 1), _on_wall(front, right, 1, 1)], shade(stripe, 0.8))
    p.line([left, front, right], shade(wall, 1.12))
    door = (170, 174, 180)
    _patch(p, left, front, (0.55, 0.88), (2.5, 7), door)
    for drop in (3.5, 5, 6.5):
        p.line(
            [_on_wall(left, front, 0.55, drop), _on_wall(left, front, 0.88, drop)], shade(door, 0.7)
        )
    for t in (0.53, 0.9):
        p.line([_on_wall(left, front, t, 5.5), _on_wall(left, front, t, 7)], (232, 196, 60))
    _patch(p, left, front, (0.52, 0.91), (2, 2.5), shade(wall, 0.7))
    _windows(p, left, front, (0.1, 0.2, 0.3, 0.4), (3,), 2, (110, 160, 200))
    _windows(p, left, front, (0.1, 0.2, 0.3, 0.4), (5.4,), 1, shade(wall, 1.1))
    _windows(p, front, right, (0.3, 0.6), (3,), 2, shade((110, 160, 200), 0.8))
    for t in (0.3, 0.7):
        x, y = _on_wall(_on_wall(left, back, 0.5, 0), _on_wall(front, right, 0.5, 0), t, 0)
        p.box(x, y, 1.5, 1.5, (190, 196, 204), (214, 218, 224))
        p.point(x, y - 2, (120, 130, 140))
    sx, sy = _on_wall(left, front, 0.25, 0)
    p.polygon([(sx - 4, sy - 2), (sx + 4, sy + 2), (sx + 4, sy - 1), (sx - 4, sy - 5)], stripe)
    p.line([(sx - 3, sy - 3), (sx + 3, sy)], (244, 246, 248))
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
            p.point(cx - 0.5, cy - 0.5, (150, 176, 200))
        else:
            p.line(
                [_on_wall(crate[0], crate[1], 0, 2), _on_wall(crate[0], crate[1], 1, 2)],
                shade(white, 0.55),
            )
            p.point(*_on_wall(crate[0], crate[1], 0.85, 3.5), shade(white, 0.5))
    return p.sprite()


def paper_mill() -> Sprite:
    """A riverside paper mill on a dark plinth turning a waterwheel in its stream between muddy
    banks, foam where the wheel churns it, with arched windows, a door with cart ruts worn out of
    it, a tall banded stack, rolls of paper stacked in the yard and a pile of pulpwood logs among
    scattered bark."""
    p = Painter(38, 51, (19, 41))
    water, bank = (84, 140, 196), (96, 88, 62)
    p.polygon([(-18, -3.8), (0.5, 5.2), (-4, 8.7), (-18, 2)], bank)
    p.polygon([(-17, -3), (-1, 5), (-4, 7.5), (-17, 1)], water)
    p.line([(-14, -0.5), (-10, 1.5)], (150, 196, 226))
    foot = _footing(-1, -3, 12, 8)
    _bed(p, foot)
    p.blob(12, 4, 18, 8, (120, 96, 66))
    p.shadow(11, 4)
    _track(p, foot[1], foot[2], (0.4, 0.65), 4)
    _chimney(p, 8, -4, -30)
    _smoke(p, 9, -33)
    wall, roof = (192, 168, 136), (100, 86, 80)
    top = p.hall(-1, -3, 12, 8, 9, wall, None)
    left, front, right, _ = top
    _plinth(p, top, 9, wall)
    _stains(p, left, front, 9, (0.45, 0.65, 0.85), shade(wall, 0.6), 2.2)
    _stains(p, front, right, 9, (0.15, 0.85), shade(wall, 0.5), 1.6)
    lintel = shade(wall, 1.2)
    _windows(p, left, front, (0.55, 0.75, 0.92), (2, 5.5), 2, arch=lintel)
    _windows(p, front, right, (0.3, 0.75), (2,), 2, shade(WINDOW, 0.8), lintel)
    _door(p, front, right, (0.4, 0.65), 5, 9, (70, 56, 48), lintel)
    p.gable(top, 4, roof, wall)
    _roof_rows(p, top, 4, roof)
    cx, cy = _on_wall(left, front, 0.25, 7)
    wheel = (98, 72, 50)
    p.ellipse(cx - 3, cy - 5, cx + 3, cy + 4, wheel)
    p.ellipse(cx - 2, cy - 4, cx + 2, cy + 3, shade(wheel, 1.3))
    p.line([(cx, cy - 4), (cx, cy + 3)], wheel)
    p.line([(cx - 2, cy - 0.5), (cx + 2, cy - 0.5)], wheel)
    for dx, dy in ((-3, -1), (3, 0), (-2, 3), (2, -4)):
        p.point(cx + dx, cy + dy, shade(wheel, 0.6))
    p.polygon([(cx - 4, cy + 2), (cx + 4, cy + 5), (cx + 4, cy + 6), (cx - 4, cy + 3)], water)
    for dx, dy in ((-3, 3), (1, 4.5), (3, 5.5)):
        p.point(cx + dx, cy + dy, (226, 238, 244))
    roll = (240, 240, 234)
    for x, y in ((6, 6), (9, 4.5), (12, 3), (7.5, 2.5), (10.5, 1), (9, -1)):
        p.ellipse(x - 1.5, y - 3, x + 1.5, y, roll)
        p.point(x, y - 1.5, shade(roll, 0.55))
    for x, y in ((13, 7.5), (17, 6.5), (16, 7.8)):
        p.point(x, y, (150, 112, 74))
    for x, y in ((14, 7), (16, 6), (15, 5)):
        p.ellipse(x - 1, y - 2, x + 1, y, (196, 160, 112))
        p.point(x, y - 1, (150, 112, 74))
    return p.sprite()


def printing_works() -> Sprite:
    """A town printing house with tall arched windows over a dark plinth, a door under a lintel
    with a path worn to it across a pavement of flags, bundles of papers at its foot by a stain of
    spilt ink and a newspaper held up on its roof as a sign."""
    p = Painter(26, 36, (12, 27))
    foot = _footing(0, 0, 8, 8)
    paving = (176, 170, 158)
    _bed(p, foot, paving, 2.2)
    _bed(p, foot, SOIL, 0.8)
    for t in (0.25, 0.5, 0.75):
        p.line(
            [_on_wall(foot[0], foot[1], t, 0.8), _on_wall(foot[0], foot[1], t, 2)],
            shade(paving, 0.8),
        )
    p.shadow(9, 3)
    _track(p, foot[0], foot[1], (0.58, 0.8), 2.4, shade(paving, 0.9), ruts=False)
    _oil(p, 9.5, 7, 1.3)
    wall = (220, 204, 172)
    top = p.box(0, 0, 8, 12, wall, (110, 106, 116))
    left, front, right, _ = top
    lintel = shade(wall, 1.12)
    _plinth(p, top, 12, wall)
    _stains(p, front, right, 12, (0.12, 0.38, 0.62, 0.88), shade(wall, 0.56), 1.5)
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], lintel)
    p.line([left, front, right], shade((110, 106, 116), 1.3))
    _band(p, left, front, 7, shade(wall, 0.85))
    _band(p, front, right, 7, shade(wall, 0.68))
    _windows(p, left, front, (0.2, 0.45, 0.7, 0.9), (2.5,), 4, arch=shade(wall, 0.75))
    _windows(p, left, front, (0.2, 0.45, 0.9), (8,), 3)
    _door(p, left, front, (0.58, 0.8), 8.5, 12, (70, 56, 48), shade(wall, 0.75))
    p.point(*_on_wall(left, front, 0.75, 10.4), (226, 196, 120))
    _windows(p, front, right, (0.25, 0.5, 0.75), (2.5, 8), 3, shade(WINDOW, 0.75))
    for x in (-3, 3):
        p.line([(x, -14), (x, -17)], (70, 66, 62))
    page = (242, 240, 230)
    p.rect(-7, -25, 7, -17, page)
    p.line([(-5, -23), (5, -23)], (40, 40, 48))
    p.rect(-5, -21, -2, -19, (150, 150, 160))
    for y in (-21, -19):
        p.line([(0, y), (5, y)], (150, 150, 160))
    paper = (236, 234, 224)
    for x, y, lift in ((6.5, 6, 0), (6.5, 6, 1.6), (10, 4.5, 0)):
        bundle = p.box(x, y - lift, 2, 1.6, paper, shade(paper, 1.04))
        p.line([bundle[1], (bundle[1][0], bundle[1][1] + 1.6)], (150, 110, 80))
    return p.sprite()


def sugar_refinery() -> Sprite:
    """A tall brick refinery over a dark plinth stained by the damp, with rows of arched windows
    and a cornice, a big banded square chimney, a cart track worn out of its door past a heap of
    sugar beet spreading over the ground, and white sacks and sugar loaves in blue paper at its
    foot."""
    p = Painter(34, 61, (15, 51))
    foot = _footing(-3, 0, 8, 8)
    _bed(p, foot)
    _bed(p, _footing(8, -3, 2, 2), margin=1)
    beet = (200, 170, 128)
    _spill(p, -12, 6, 2.5, beet)
    p.shadow(13, 4, reach=5)
    _track(p, foot[0], foot[1], (0.53, 0.77), 3.5)
    wall = (184, 110, 74)
    lintel = shade(wall, 1.25)
    stack = p.box(8, -3, 2, 34, shade(wall, 0.9), (60, 52, 50))
    for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
        p.line([_on_wall(a, b, 0, 2), _on_wall(a, b, 1, 2)], (220, 220, 214))
        p.line([_on_wall(a, b, 0, 14), _on_wall(a, b, 1, 14)], shade(wall, 0.65))
        p.line([_on_wall(a, b, 0, 33), _on_wall(a, b, 1, 33)], shade(wall, 0.6))
    _smoke(p, 9, -40, puffs=4)
    top = p.box(-3, 0, 8, 18, wall, (92, 86, 94))
    left, front, right, _ = top
    _plinth(p, top, 18, wall)
    _stains(p, left, front, 18, (0.08, 0.33, 0.85), shade(wall, 0.6), 2)
    _stains(p, front, right, 18, (0.15, 0.5, 0.85), shade(wall, 0.5), 2)
    _band(p, left, front, 0.8, lintel)
    _band(p, front, right, 0.8, shade(wall, 0.95))
    p.line([left, front, right], shade((92, 86, 94), 1.3))
    _windows(p, left, front, (0.2, 0.45, 0.7, 0.92), (2.5, 6, 9.5, 13), 2, arch=lintel)
    _windows(p, left, front, (0.2, 0.92), (15.5,), 1)
    _windows(p, front, right, (0.3, 0.7), (2.5, 6, 9.5, 13, 15.5), 2, shade(WINDOW, 0.75))
    _door(p, left, front, (0.55, 0.75), 15, 18, (70, 50, 40), lintel)
    _heap(p, -12, 6, 2.5, 2, beet)
    for x, y in ((-13, 5.5), (-11.5, 4.6)):
        p.point(x, y, (226, 206, 170))
    white, blue = (240, 240, 236), (70, 100, 180)
    for x, y in ((9, 4), (11, 2.5), (10, 1)):
        p.ellipse(x - 2, y - 3, x + 2, y, white)
        p.point(x, y - 2.5, shade(white, 0.7))
    for x, y in ((3, 7), (6, 7.5)):
        p.polygon([(x - 1.5, y), (x, y - 6), (x + 1.5, y)], white, shade(white, 0.7))
        p.polygon([(x - 1.5, y), (x - 1, y - 2), (x + 1, y - 2), (x + 1.5, y)], blue)
    return p.sprite()


def porcelain_works() -> Sprite:
    """Two small brick bottle ovens bound with iron hoops on sooty ground, one with its firing
    mouth glowing, beside a workshop with windows, a door with a track worn out of it and a
    chimney, with blue-and-white ware set out on shelves, a heap of broken shards and crates
    packed for shipping."""
    p = Painter(34, 36, (17, 27))
    oven = (172, 116, 86)
    for x, y in ((-4, -6), (-10, -3)):
        p.blob(x - 6, y - 3, x + 6, y + 3, SOIL)
    foot = _footing(5, -2, 10, 6)
    _bed(p, foot)
    _spill(p, -13, 6, 2, (220, 220, 226))
    p.shadow(13, 4)
    _track(p, foot[1], foot[2], (0.35, 0.7), 3)
    bottle = [(0, 4.5), (4, 5), (8, 4), (11, 2), (13, 1.8)]
    for x, y in ((-4, -6), (-10, -3)):
        _turned(p, x, y, bottle, oven, cap=(60, 46, 42))
        for h, half in ((3, 4.9), (6.5, 4.6)):
            _ring(p, x, y - h, half, (70, 60, 58))
        p.line([(x - 1.5, y - 12.5), (x + 1.5, y - 12.5)], shade(oven, 0.55))
    p.rect(-12, -6, -10, -3, (60, 40, 34))
    p.point(-11, -4, (240, 132, 40))
    wall, roof = (230, 218, 192), (122, 88, 70)
    top = p.hall(5, -2, 10, 6, 5, wall, None)
    left, front, right, _ = top
    _plinth(p, top, 5, wall)
    _stains(p, front, right, 5, (0.15, 0.85), shade(wall, 0.58), 1.2)
    for t in (0.25, 0.5, 0.75):
        _window(p, left, front, (t - 0.05, t + 0.05), 1.4, tall=1.6)
    _door(p, front, right, (0.35, 0.7), 1.6, 5, DOOR, shade(wall, 1.1))
    p.gable(top, 3, roof, wall)
    _roof_rows(p, top, 3, roof, (0.5,))
    x, y = _on_roof(top, 3, 0.75, 0.9)
    p.rect(x - 0.5, y - 2.5, x + 0.5, y, BRICK)
    frame, china, blue = (120, 88, 58), (246, 246, 250), (54, 86, 176)
    for x in (-8, 2):
        p.line([(x, 6 + (x + 8) * 0.1), (x, -1)], frame)
    for y in (5, 2):
        p.line([(-8, y), (2, y + 1)], frame)
        for k in range(5):
            x = -7 + 2 * k
            p.line([(x, y - 1 + k * 0.2), (x, y - 2 + k * 0.2)], china if k % 2 else blue)
    _heap(p, -13, 6, 2, 1.5, (220, 220, 226))
    for x, y in ((-14, 5.4), (-12, 6.4), (-15.5, 6.6)):
        p.point(x, y, blue)
    for x, y in ((10, 6.5), (13, 5)):
        _crate(p, x, y)
        p.point(x - 1, y - 2.5, (226, 196, 120))
    return p.sprite()


def meat_factory() -> Sprite:
    """A brick slaughterhouse over a dark plinth with two rows of arched windows, a louvred vent on
    its ridge and a ham hung out as its sign, a ramp up from a cattle pen trampled to mud, and a
    white cold store with an insulated door, a track worn out of it past a puddle of melt water,
    and a refrigeration unit and vent on its roof."""
    p = Painter(36, 36, (17, 25))
    foot = _footing(0, -3, 14, 8)
    store_foot = _footing(11, 3, 5, 5)
    _bed(p, foot)
    _bed(p, store_foot, margin=1)
    mud = (112, 94, 66)
    p.polygon(list(_paddock(-10, 3, 4.6)), mud)
    for x, y in ((-12, 3), (-8, 5), (-10, 1.6)):
        p.point(x, y, shade(mud, 0.75))
    p.shadow(15, 4.5)
    _track(p, store_foot[1], store_foot[2], (0.3, 0.7), 2.5)
    _puddle(p, 16, 7, 1.3)
    roof = (86, 80, 88)
    lintel = shade(BRICK, 1.3)
    top = p.hall(0, -3, 14, 8, 8, BRICK, None)
    left, front, *_ = top
    _plinth(p, top, 8, BRICK)
    _stains(p, left, front, 8, (0.06, 0.2, 0.55, 0.85), shade(BRICK, 0.62), 1.8)
    _windows(p, left, front, (0.6, 0.75, 0.9), (2,), 2, arch=lintel)
    _windows(p, left, front, (0.6, 0.75, 0.9), (5,), 1, arch=lintel)
    _patch(p, left, front, (0.3, 0.45), (3, 6), (56, 44, 40))
    p.line([_on_wall(left, front, 0.28, 2.4), _on_wall(left, front, 0.47, 2.4)], lintel)
    p.gable(top, 4, roof, BRICK)
    _roof_rows(p, top, 4, roof)
    ridge_back, ridge_front = _ridge(top, 4)
    a, b = _on_wall(ridge_back, ridge_front, 0.4, 0), _on_wall(ridge_back, ridge_front, 0.7, 0)
    p.polygon([(a[0], a[1] - 1.5), (b[0], b[1] - 1.5), b, a], (176, 168, 160), shade(roof, 0.8))
    p.line([(a[0] - 0.5, a[1] - 2), (b[0] + 0.5, b[1] - 2)], shade(roof, 0.8))
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
    for k in (1, 2, 3):
        p.line(
            [(ramp[0][0] - k, ramp[0][1] + k), (ramp[1][0] - k, ramp[1][1] + k)],
            shade(TIMBER, 0.75),
        )
    pen_left, pen_back, pen_right, pen_front = _paddock(-10, 3, 5)
    _fence(p, pen_left, pen_back)
    _fence(p, pen_back, pen_right)
    _cow(p, -8, 4, (138, 84, 50), (96, 58, 36))
    _fence(p, pen_left, pen_front)
    _fence(p, pen_front, pen_right)
    cold = (234, 236, 238)
    store = p.box(11, 3, 5, 6, cold, shade(cold, 0.88))
    _plinth(p, store, 6, cold, 0.6)
    _stains(p, store[1], store[2], 6, (0.15, 0.85), shade(cold, 0.66), 1.2)
    for t in (0.25, 0.5, 0.75):
        p.line(
            [_on_wall(store[0], store[1], t, 0.5), _on_wall(store[0], store[1], t, 5.4)],
            shade(cold, 0.85),
        )
    _patch(p, store[1], store[2], (0.3, 0.7), (2, 6), (150, 170, 186))
    p.line(
        [_on_wall(store[1], store[2], 0.5, 2), _on_wall(store[1], store[2], 0.5, 6)],
        shade(cold, 0.6),
    )
    p.point(*_on_wall(store[1], store[2], 0.62, 4), (60, 62, 70))
    p.box(12, -3.5, 2, 2, METAL, shade(METAL, 1.1))
    p.blob(11, -7, 13, -5, (60, 62, 70))
    p.line([(8, -4), (8, -9)], METAL)
    p.rect(7, -10, 9, -9, shade(METAL, 0.8))
    return p.sprite()


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
    """A car works filling its cell: a long assembly hall on a dark plinth under an arched roof,
    with a band of mullioned glass and a red sign on its end, beside a sawtoothed press shop, with new cars
    parked in rows on the lot out front."""
    p = Painter(50, 40, (24, 26))
    lot = (166, 166, 162)
    _plate(p, lot)
    under = shade(lot, 0.76)
    _bed(p, _footing(*_iso(0, 0, 3, 8), 14, 5), under, 0.8)
    _bed(p, _footing(*_iso(0, 0, 0, 0.5), 20, 7), under, 0.8)
    for u in (-10.2, -6.6, -3, 0.6, 4.2, 7.8):
        p.line([_iso(0, 0, u, -4.4), _iso(0, 0, u, -9.8)], shade(lot, 1.1))
    for u, v in ((-4.8, -7), (2.4, -5.6), (9, -9)):
        _oil(p, *_iso(0, 0, u, v), 1.1)
    wall = (156, 160, 170)
    press_wall = shade(wall, 0.92)
    press = p.hall(*_iso(0, 0, 3, 8), 14, 5, 6, press_wall, None)
    _plinth(p, press, 6, press_wall)
    _windows(p, press[1], press[2], (0.3, 0.7), (2,), 2, shade(GLASS, 0.8))
    _sawtooth(p, press, 4, 3, (104, 110, 126), (160, 196, 218), press_wall)
    main = p.hall(*_iso(0, 0, 0, 0.5), 20, 7, 7, wall, None)
    left, front, right, _ = main
    _plinth(p, main, 7, wall)
    _patch(p, left, front, (0.03, 0.97), (1.5, 3), (96, 136, 168))
    for k in range(1, 8):
        t = 0.03 + 0.94 * k / 8
        p.line([_on_wall(left, front, t, 1.5), _on_wall(left, front, t, 3)], (150, 184, 206))
    _patch(p, front, right, (0.2, 0.8), (1.5, 7), (64, 66, 74))
    p.line([_on_wall(front, right, 0.5, 1.5), _on_wall(front, right, 0.5, 7)], (84, 86, 96))
    _vault(p, main, 5, (98, 108, 130), wall)
    sx, sy = _on_wall(front, right, 0.5, -2.2)
    p.rect(sx - 2.5, sy - 1, sx + 2.5, sy + 1, (196, 58, 54))
    p.line([(sx - 1.5, sy), (sx + 1.5, sy)], (244, 240, 230))
    p.line([_iso(0, 0, -10, -3.6), _iso(0, 0, 9, -3.6)], shade(lot, 1.15))
    for row, v in enumerate((-5.5, -8.5)):
        for k in range(6):
            colour = CAR_COLOURS[(k + 2 * row) % len(CAR_COLOURS)]
            _car(p, *_iso(0, 0, -9 + 3.6 * k + row, v), colour)
    return p.sprite()


def plane_factory() -> Sprite:
    """An aircraft works filling its cell: a great hangar on a dark plinth under an arched roof
    with its sliding doors pushed open and a blue sign, a long assembly shed behind, and a new
    airliner standing on the dark apron in front by a windsock and a tug."""
    p = Painter(50, 40, (24, 26))
    _plate(p, (176, 176, 170))
    apron = (112, 114, 118)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-10.5, -10.5), (10.5, -10.5), (10.5, -1), (-10.5, -1))],
        apron,
    )
    p.line([_iso(0, 0, -10, -2.5), _iso(0, 0, 10, -2.5)], (226, 196, 70))
    under = shade((176, 176, 170), 0.76)
    _bed(p, _footing(*_iso(0, 0, 4, 9), 12, 3.5), under, 0.8)
    _bed(p, _footing(*_iso(0, 0, -4, 3.5), 12, 9), under, 0.8)
    for dv in (0, 1.6):
        p.line(
            [_iso(0, 0, 2.2, 2 + dv), _iso(0, 0, 3.4, 0 + dv), _iso(0, 0, 3.8, -2 + dv)],
            shade(apron, 0.82),
        )
    for u, v in ((-7, -5), (8, -8.5), (-2, -9)):
        _oil(p, *_iso(0, 0, u, v), 1.2)
    wall = (200, 206, 214)
    shed = p.hall(*_iso(0, 0, 4, 9), 12, 3.5, 6, shade(wall, 0.9), (140, 150, 166))
    _plinth(p, shed, 6, shade(wall, 0.9))
    _windows(p, shed[0], shed[1], tuple(k / 7 for k in range(1, 7)), (2,), 1, (96, 136, 168))
    hangar = p.hall(*_iso(0, 0, -4, 3.5), 12, 9, 7, wall, None)
    left, front, right, _ = hangar
    _plinth(p, hangar, 7, wall)
    _windows(p, left, front, tuple(k / 8 for k in range(1, 8)), (2,), 2, (96, 136, 168))
    _patch(p, left, front, (0.3, 0.7), (4.3, 5.8), (70, 130, 210))
    p.line([_on_wall(left, front, 0.36, 5), _on_wall(left, front, 0.64, 5)], (240, 242, 246))
    _patch(p, front, right, (0.08, 0.92), (0.5, 7), (44, 48, 58))
    for span in ((0, 0.12), (0.88, 1)):
        _patch(p, front, right, span, (0.5, 7), shade(wall, 0.7))
    for t in (0.06, 0.94):
        p.line([_on_wall(front, right, t, 0.5), _on_wall(front, right, t, 7)], shade(wall, 0.55))
    tail = [_on_wall(front, right, t, d) for t, d in ((0.55, 7), (0.62, 2), (0.68, 7))]
    p.polygon(tail, (70, 130, 210))
    _vault(p, hangar, 7, (150, 160, 176), wall)
    _plane(p, *_iso(0, 0, 4.5, -3.5), 0.75)
    wx, wy = _iso(0, 0, -9.5, -9.5)
    p.line([(wx, wy), (wx, wy - 7)], (90, 90, 96))
    p.polygon([(wx, wy - 7), (wx + 4, wy - 6), (wx, wy - 5)], (236, 120, 40))
    p.line([(wx + 2, wy - 6.5), (wx + 2, wy - 5.5)], (244, 244, 240))
    tug = (236, 196, 60)
    p.box(*_iso(0, 0, -6, -7), 1.5, 1.5, tug, shade(tug, 1.15))
    return p.sprite()


def tank_factory() -> Sprite:
    """A tank works filling its cell: low armoured assembly halls in olive drab with slit windows,
    riveted bands and vents on their roofs, a banded stack behind, and a new tank standing on its
    churned-up proving ground beside stacked ammunition crates and fuel drums."""
    p = Painter(50, 36, (24, 22))
    _plate(p, (150, 142, 112))
    olive = (104, 112, 80)
    roof = shade(olive, 0.72)
    mud = (112, 92, 66)
    cx, cy = _iso(0, 0, 8, 11)
    _chimney(p, cx, cy, cy - 19, shade(BRICK, 0.8))
    p.polygon([_iso(0, 0, 0, -2), _iso(0, 0, 10, -2), _iso(0, 0, 10, -10), _iso(0, 0, 0, -10)], mud)
    for v in (-5, -7.5):
        p.line([_iso(0, 0, 1, v), _iso(0, 0, 9, v)], shade(mud, 0.7))
    under = shade((150, 142, 112), 0.76)
    _bed(p, _footing(*_iso(0, 0, 0, 7), 20, 6), under, 0.8)
    _bed(p, _footing(*_iso(0, 0, -5.5, -2), 9, 8), under, 0.8)
    _puddle(p, *_iso(0, 0, 2.5, -3.5), 1.4)
    _puddle(p, *_iso(0, 0, 8.5, -9), 1.2)
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
        _plinth(p, top, height, shade(olive, tone), 0.6)
        left, front, right, back = top
        middle = _on_wall(left, back, 0.5, 0), _on_wall(front, right, 0.5, 0)
        for t in (0.3, 0.7):
            x, y = _on_wall(*middle, t, 0)
            p.box(x, y, 1.2, 1.2, shade(olive, 0.8), shade(olive, 1.05))
    _tank_vehicle(p, *_iso(0, 0, 4.5, -6), olive)
    crate = (140, 122, 80)
    for x, y, lift in ((-11.5, 3.5, 0), (-8.5, 5, 0), (-10, 4.2, 1.8)):
        p.box(x, y - lift, 1.5, 1.6, crate, shade(crate, 1.2))
    for x, y in ((-14.5, 2.5), (-13, 3.5)):
        _drum(p, x, y, (70, 76, 60))
    return p.sprite()


def mega_factory() -> Sprite:
    """A great works filling its cell: three long brick halls on dark plinths under sawtooth roofs,
    with loading doors in their ends, a row of tall banded chimneys behind them and a round
    gasholder in its guide frame."""
    p = Painter(50, 63, (24, 51))
    _plate(p, (150, 146, 140))
    under = shade((150, 146, 140), 0.76)
    for v in (6, 0, -6):
        foot = _footing(*_iso(0, 0, -1, v), 18, 5)
        _bed(p, foot, under, 0.8)
        _track(p, foot[1], foot[2], (0.25, 0.75), 2.2, shade((150, 146, 140), 1.1))
    _spill(p, *_iso(0, 0, 7, -9), 1.8, (48, 46, 50))
    _oil(p, *_iso(0, 0, -6, -7), 1.2)
    _puddle(p, *_iso(0, 0, 3, -9), 1.3)
    for u in (-9, -4, 1, 6):
        x, y = _iso(0, 0, u, 10)
        _chimney(p, x, y - 4, y - 30, shade(BRICK, 0.8))
        _smoke(p, x + 1, y - 33, (214, 214, 218))
    lintel = shade(BRICK, 1.3)
    for v, height, tone in ((6, 8, 0.9), (0, 9, 1.0), (-6, 7, 1.08)):
        wall = shade(BRICK, tone)
        top = p.hall(*_iso(0, 0, -1, v), 18, 5, height, wall, None)
        _plinth(p, top, height, wall)
        _windows(p, top[0], top[1], tuple(k / 8 for k in range(1, 8)), (2.5,), 2, arch=lintel)
        _door(p, top[1], top[2], (0.25, 0.75), 3, height, DOORWAY, lintel)
        _sawtooth(p, top, 5, 3, METAL, (170, 200, 220), wall)
    gx, gy = _iso(0, 0, 9.5, 2)
    holder = (118, 126, 118)
    _turned(p, gx, gy, [(0, 4.5), (8, 4.5)], holder, cap=(138, 146, 138))
    for h in (3, 6):
        _ring(p, gx, gy - h, 4.5, shade(holder, 0.75))
    frame = (86, 92, 86)
    tops = []
    for a in np.linspace(0.3, math.pi - 0.3, 5):
        fx, fy = gx + 5 * math.cos(a), gy + 2.5 * math.sin(a)
        p.line([(fx, fy), (fx, fy - 10)], frame)
        tops.append((fx, fy - 10))
    p.line(tops, frame)
    for u, v in ((-9, -9), (-6.5, -9.5)):
        _crate(p, *_iso(0, 0, u, v))
    _barrel(p, *_iso(0, 0, -3.5, -9.5))
    _heap(p, *_iso(0, 0, 7, -9), 1.8, 1.5, (48, 46, 50))
    return p.sprite()


def refinery() -> Sprite:
    """An oil refinery filling its cell: tall silver distillation columns ringed with walkways and
    climbed by ladders, squat white storage tanks with stairs up their sides, pipe racks with red
    valves and a flare burning on a tall red-and-white stack."""
    p = Painter(50, 58, (24, 46))
    plate = (168, 164, 156)
    _plate(p, plate)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-10.2, -10.2), (9.8, -10.2), (9.8, -3.6), (-10.2, -3.6))],
        shade(plate, 0.92),
        shade(plate, 0.6),
    )
    for u, v in ((-7, 2), (-2, 2), (3, 3)):
        x, y = _iso(0, 0, u, v)
        p.blob(x - 4, y - 1.6, x + 4, y + 2, shade(plate, 0.74))
    for u, v in ((-4.5, -6.5), (2.5, -8.5), (5, 0.5)):
        _oil(p, *_iso(0, 0, u, v), 1.2)
    metal, white = (176, 178, 184), (224, 226, 228)
    fx, fy = _iso(0, 0, 9, 9)
    p.line([(fx, fy), (fx, fy - 36)], (90, 90, 96))
    p.line([(fx + 1, fy), (fx + 1, fy - 36)], (124, 124, 130))
    for y in (-34, -30, -26):
        p.line([(fx, fy + y), (fx + 1, fy + y)], (206, 60, 50))
    p.blob(fx - 2, fy - 43, fx + 3, fy - 36, (246, 150, 40))
    p.blob(fx - 0.5, fy - 40, fx + 1.5, fy - 37, (255, 220, 120))
    for u, v in ((-9, 8), (-3, 9)):
        x, y = _iso(0, 0, u, v)
        _turned(p, x, y, [(0, 4.5), (4, 4.5)], white, cap=shade(white, 1.08))
        p.line([(x - 4, y - 0.5), (x - 1, y - 4.5)], shade(white, 0.65))
    for u, v, r, h in ((-7, 2, 2, 18), (-2, 2, 2.5, 30), (3, 3, 3, 24)):
        x, y = _iso(0, 0, u, v)
        _turned(p, x, y, [(0, r), (h, r), (h + 1.5, r * 0.6)], metal)
        for ring in range(6, h, 6):
            p.line([(x - r - 1, y - ring), (x + r + 1, y - ring)], shade(metal, 0.6))
        p.line([(x + r - 0.5, y - 1), (x + r - 0.5, y - h)], shade(metal, 0.62))
    rack = (96, 96, 104)
    for u in (-9, -3, 3, 9):
        x, y = _iso(0, 0, u, -2)
        p.line([(x, y), (x, y - 6)], rack)
    for h, colour in ((6, rack), (4.5, (150, 120, 60))):
        a, b = _iso(0, 0, -9, -2), _iso(0, 0, 9, -2)
        p.line([(a[0], a[1] - h), (b[0], b[1] - h)], colour)
    for u in (-6, 0, 6):
        x, y = _iso(0, 0, u, -2)
        p.point(x, y - 5.5, (206, 60, 50))
    for u, v in ((-8, -6), (-1, -7.5), (6, -7)):
        x, y = _iso(0, 0, u, v)
        _turned(p, x, y, [(0, 3.5), (3.5, 3.5)], white, cap=shade(white, 1.08))
        p.line([(x - 3, y - 0.5), (x - 0.5, y - 3.5)], shade(white, 0.65))
    return p.sprite()


def power_plant() -> Sprite:
    """A coal power station filling its cell: a big brick turbine hall on a dark plinth with tall
    windows, a cornice and a glazed roof light, two concrete cooling towers letting off steam, a
    tall banded chimney, a coal heap fed by a conveyor on trestles, a transformer and a pylon
    carrying power lines away."""
    p = Painter(50, 70, (24, 58))
    plate = (160, 156, 148)
    _plate(p, plate)
    hall = _footing(*_iso(0, 0, -3, -0.5), 14, 7)
    _bed(p, hall, shade(plate, 0.74), 0.8)
    _track(p, hall[0], hall[1], (0.78, 0.96), 3, shade(plate, 0.88))
    cx, cy = _iso(0, 0, 5, -7)
    p.blob(cx - 8, cy - 3, cx + 6, cy + 3.5, (110, 108, 106))
    _spill(p, cx, cy, 5, (48, 46, 50))
    _puddle(p, *_iso(0, 0, -8.5, -2), 1.3)
    cx, cy = _iso(0, 0, -9, 9)
    _chimney(p, cx, cy - 4, cy - 38, shade(BRICK, 0.9))
    _smoke(p, cx + 1, cy - 41, (124, 122, 126))
    _cooling_tower(p, *_iso(0, 0, -2, 9.5))
    _pylon(p, *_iso(0, 0, -8, -8), -4)
    tx, ty = _iso(0, 0, -4.5, -6)
    grey = (120, 126, 120)
    transformer = p.box(tx, ty, 2, 2.5, grey, shade(grey, 1.15))
    for t in (0.3, 0.7):
        p.line(
            [
                _on_wall(transformer[0], transformer[1], t, 0.5),
                _on_wall(transformer[0], transformer[1], t, 2.5),
            ],
            shade(grey, 0.7),
        )
    for dx in (-1, 1):
        p.line([(tx + dx, ty - 3.5), (tx + dx, ty - 5)], (210, 200, 170))
    top = p.hall(*_iso(0, 0, -3, -0.5), 14, 7, 12, BRICK, None)
    left, front, right, _ = top
    window = (88, 112, 140)
    _plinth(p, top, 12, BRICK)
    _band(p, left, front, 1, shade(BRICK, 1.3))
    _band(p, front, right, 1, shade(BRICK, 1.05))
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2.5,), 7, window, shade(BRICK, 1.3))
    _door(p, left, front, (0.8, 0.94), 7, 12, (58, 48, 44), shade(BRICK, 1.3))
    _windows(p, front, right, (0.3, 0.7), (2.5,), 7, shade(window, 0.8), shade(BRICK, 1.2))
    p.gable(top, 4, (78, 78, 88), BRICK)
    _skylight(p, top, 4, (0.08, 0.92))
    _cooling_tower(p, *_iso(0, 0, 8.5, 4))
    hx, hy = _iso(0, 0, 5, -7)
    end = _on_wall(left, front, 0.9, 4)
    for t in (0.35, 0.7):
        lx = hx - 1 + (end[0] - hx + 1) * t
        ly = hy - 5 + (end[1] - hy + 5) * t
        p.line([(lx, ly), (lx, ly + 5 + 2 * t)], (90, 90, 98))
    p.line([(hx - 1, hy - 5), end], (70, 70, 78))
    p.line([(hx - 1, hy - 4), _on_wall(left, front, 0.9, 5)], (110, 110, 118))
    _heap(p, hx, hy, 5, 5, (48, 46, 50))
    for dx, dy in ((-2, -2), (0, -3), (2, -1), (-1, 0)):
        p.point(hx + dx, hy + dy, (104, 104, 116))
    return p.sprite()


def shipyard() -> Sprite:
    """A shipyard filling its cell: a big ship's hull with a row of portholes on a slipway running
    down into the water past bollards on the quay, a gantry crane with its cab straddling it, a
    fabrication hall with a tiled roof behind and a stack of steel plates."""
    p = Painter(50, 46, (24, 32))
    _plate(p, STONE)
    hall_foot = _footing(*_iso(0, 0, -3, 7), 14, 6)
    _bed(p, hall_foot, shade(STONE, 0.74), 0.8)
    _track(p, hall_foot[1], hall_foot[2], (0.2, 0.8), 2.2, shade(STONE, 1.1))
    p.blob(*_iso(0, 0, 3.5, 2.5), *_iso(0, 0, 7.5, 2.5), (150, 104, 74))
    _puddle(p, *_iso(0, 0, -7, -9), 1.3)
    _oil(p, *_iso(0, 0, 4, -9.5), 1.1)
    water = (72, 128, 184)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((7.5, -10.5), (10.5, -10.5), (10.5, 10.5), (7.5, 10.5))],
        water,
    )
    for v in (-7, 5):
        p.line([_iso(0, 0, 8.5, v), _iso(0, 0, 9.5, v + 2)], (150, 196, 226))
    for v in (-9, 3, 8):
        x, y = _iso(0, 0, 7, v)
        p.rect(x - 0.5, y - 1.5, x + 0.5, y, (50, 50, 56))
    hall_wall, hall_roof = (176, 166, 152), (92, 96, 110)
    hall = p.hall(*_iso(0, 0, -3, 7), 14, 6, 9, hall_wall, None)
    _plinth(p, hall, 9, hall_wall)
    _windows(p, hall[0], hall[1], (0.15, 0.35, 0.55, 0.75), (2,), 3, arch=shade(hall_wall, 1.2))
    _patch(p, hall[1], hall[2], (0.2, 0.8), (3, 9), shade(hall_wall, 0.5))
    p.line(
        [_on_wall(hall[1], hall[2], 0.5, 3), _on_wall(hall[1], hall[2], 0.5, 9)],
        shade(hall_wall, 0.62),
    )
    p.gable(hall, 4, hall_roof, hall_wall)
    _roof_rows(p, hall, 4, hall_roof)
    plates = (110, 116, 126)
    px, py = _iso(0, 0, 5.5, 2.5)
    for lift in (0, 1, 2):
        p.hall(px + 0.3 * lift, py - lift, 4, 2.5, 1, plates, shade(plates, 1.25))
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
    for k in range(6):
        p.point(*_on_wall(ship[0], ship[1], 0.12 + 0.15 * k, 3.2), (226, 222, 210))
    front, right = ship[1], ship[2]
    tip = _iso(*_on_wall(front, right, 0.5, 0), 4, 0)
    p.polygon([front, tip, (tip[0], tip[1] + 5), (front[0], front[1] + 6)], hull, shade(hull, 0.55))
    p.polygon([front, tip, (tip[0], tip[1] + 2), (front[0], front[1] + 2)], band)
    p.polygon([front, right, tip], deck, shade(deck, 0.6))
    bx, by = _iso(0, 0, -7, -3)
    bridge = p.box(bx, by - 6, 2.5, 4, (238, 236, 230), shade(STONE, 1.15))
    _windows(p, bridge[0], bridge[1], (0.3, 0.7), (1,), 1)
    p.rect(bx - 1, by - 14, bx + 1, by - 10, (196, 60, 48))
    p.line([(bx - 1, by - 13), (bx + 1, by - 13)], (40, 40, 46))
    near = _iso(0, 0, -2, -7.5)
    far = _iso(0, 0, -2, 1.5)
    p.line([near, (near[0], near[1] - 20)], crane)
    p.line([(near[0], near[1] - 20), (far[0], far[1] - 20)], crane)
    p.line([(near[0] + 1, near[1] - 19), (far[0] + 1, far[1] - 19)], shade(crane, 0.7))
    p.rect(near[0] + 1, near[1] - 18, near[0] + 3, near[1] - 16, (70, 76, 90))
    hook = _iso(0, 0, -2, -3)
    p.line([(hook[0], hook[1] - 19), (hook[0], hook[1] - 12)], (70, 64, 58))
    p.rect(hook[0] - 1, hook[1] - 12, hook[0] + 1, hook[1] - 10, (88, 90, 98))
    return p.sprite()


def gasworks() -> Sprite:
    """A round gasholder in a lattice frame of iron columns, its bell riding high, beside a brick
    retort house with a tall chimney and a heap of coal by its door."""
    p = Painter(30, 30, (14, 22))
    p.shadow(12, 3.5)
    tank = (110, 118, 112)
    p.ellipse(-2, -10, 12, -3, shade(tank, 0.8))
    p.rect(-2, -7, 12, 3, tank)
    p.ellipse(-2, 0, 12, 6, shade(tank, 0.7))
    p.ellipse(-2, -10, 12, -4, shade(tank, 1.15))
    for x in (-2, 2.5, 7.5, 12):
        p.line([(x, -14), (x, 4)], shade(METAL, 0.8))
    p.line([(-2, -14), (12, -14)], shade(METAL, 0.9))
    p.line([(-2, -9), (12, -9)], shade(METAL, 0.7))
    top = p.box(-8, 1, 5, 6, BRICK, None)
    p.gable(top, 3, (90, 76, 70), BRICK)
    left, front, _, _ = top
    _patch(p, left, front, (0.4, 0.62), (1.5, 6), DOORWAY)
    p.rect(-12, -16, -10.6, -2, shade(BRICK, 0.9))
    p.line([(-12, -17), (-10.6, -17)], shade(BRICK, 0.5))
    _heap(p, -12.5, 6, 2.6, 2.6, (54, 52, 56))
    return p.sprite()


def railway_works() -> Sprite:
    """A long brick erecting shop with roof lights and three great doors, a tall chimney, and
    tracks running out past a locomotive being finished and a row of new wagons."""
    p = Painter(40, 32, (19, 22))
    p.shadow(16, 4)
    for k in range(3):
        p.line([(-16 + 2 * k, 6 + k), (16 + 2 * k, -10 + k)], METAL)
    top = p.hall(-1, -1, 22, 9, 7, BRICK, None)
    left, front, _, _ = top
    p.gable(top, 4, (88, 80, 78), BRICK)
    for f in (0.2, 0.45, 0.7):
        p.line([_on_wall(left, front, f, -1.6), _on_wall(left, front, f + 0.12, -1.6)], GLASS)
    for f in (0.15, 0.5, 0.85):
        _patch(p, left, front, (f - 0.07, f + 0.07), (2, 7), DOORWAY)
    p.rect(12, -22, 13.6, -6, shade(BRICK, 0.9))
    p.line([(12, -23), (13.6, -23)], shade(BRICK, 0.5))
    engine = (40, 44, 48)
    p.rect(-14, 2, -8, 5, engine)
    p.rect(-9, 0, -7, 5, engine)
    p.rect(-14, 0.6, -13, 2, engine)
    p.point(-12, 5.5, (150, 40, 36))
    p.point(-9.5, 5.5, (150, 40, 36))
    for x in (2, 7):
        p.rect(x, 4, x + 4, 6.5, (120, 64, 44))
    return p.sprite()
