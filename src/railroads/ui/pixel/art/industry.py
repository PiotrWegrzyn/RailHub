"""Factories and works of every kind."""

import math

import numpy as np

from railroads.ui.pixel.art.commerce import _glass_rack, _planks
from railroads.ui.pixel.art.painter import (
    BARREL,
    BRICK,
    CAR_COLOURS,
    CHIMNEY_SHARE,
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
    Top,
    chimney_top,
    shade,
)
from railroads.ui.pixel.art.parts import (
    BEAST,
    DOOR,
    METRE,
    SILL,
    _band,
    _bands,
    _bar,
    _barrel,
    _cow,
    _fence,
    _flag,
    _heap,
    _iso,
    _on_wall,
    _patch,
    _plate,
    _plinth,
    _ridge,
    _turned,
    _vault,
    _windows,
)

GLAZING: RGB = (160, 196, 218)

EARTH: RGB = (150, 128, 90)

SOIL: RGB = (78, 74, 52)
SMOKE: RGB = (150, 148, 150)
# The most puffs of smoke rising from anything.
MOST_PUFFS = 2
FRAME: RGB = (104, 76, 50)
JOINER_WALL: RGB = (176, 134, 88)
CHAIR: RGB = (156, 62, 44)
BUNDLE: RGB = (236, 234, 224)
ANVIL: RGB = (40, 40, 46)
STILL_SLATE: RGB = (68, 72, 82)

OIL: RGB = (62, 60, 62)

PUDDLE: RGB = (98, 126, 146)


MILL_BRICK: RGB = (158, 74, 52)
DRESSING: RGB = (230, 206, 158)
MILL_SLATE: RGB = (80, 86, 100)
NORTH_LIGHT: RGB = (178, 212, 228)
MILL_GLASS: RGB = (66, 84, 106)
CINDERS: RGB = (108, 100, 90)
GOODS_CRATE: RGB = (184, 142, 90)
GIRDER: RGB = (96, 102, 112)


def _round_head(
    a: Point, b: Point, length: float, t: float, wide: float, head: float, foot: float
) -> list[Point]:
    """The outline of an opening with a round head on the wall under the edge from `a` to `b`,
    which is `length` units long: `wide` across about `t` of the way along it, from its crown
    `head` down the wall to its foot `foot` down."""
    half, r = wide / 2 / length, wide / 2
    arc = [
        _on_wall(a, b, t + half * math.cos(turn), head + r - r * math.sin(turn))
        for turn in np.linspace(0, math.pi, 9)
    ]
    return [_on_wall(a, b, t - half, foot), _on_wall(a, b, t + half, foot), *arc]


def _mill_window(
    p: Painter,
    a: Point,
    b: Point,
    length: float,
    t: float,
    wide: float,
    span: tuple[float, float],
    glass: RGB = MILL_GLASS,
) -> None:
    """A tall iron-framed window of many small panes with a round head under an arch of cream
    brick and a stone sill, `wide` across about `t` along the wall under the edge from `a` to
    `b`, which is `length` units long, from `span[0]` down the wall to `span[1]`."""
    m = METRE
    head, foot = span
    half = wide / 2 / length
    p.polygon(
        _round_head(a, b, length, t, wide + 0.7 * m, head - 0.35 * m, head + wide / 2), DRESSING
    )
    p.polygon(_round_head(a, b, length, t, wide, head, foot), glass)
    bar = shade(glass, 1.55)
    _bar(p, _on_wall(a, b, t, head), _on_wall(a, b, t, foot), 0.07 * m, bar)
    for drop in np.arange(head + wide / 2, foot - 0.3 * m, 0.75 * m):
        _bar(p, _on_wall(a, b, t - half, drop), _on_wall(a, b, t + half, drop), 0.06 * m, bar)
    p.polygon(
        [
            _on_wall(a, b, t - half * 1.3, foot),
            _on_wall(a, b, t + half * 1.3, foot),
            _on_wall(a, b, t + half * 1.3, foot + 0.25 * m),
            _on_wall(a, b, t - half * 1.3, foot + 0.25 * m),
        ],
        DRESSING,
    )


def _north_lights(p: Painter, top: Top, teeth: int, rise: float, roof: RGB, wall: RGB) -> None:
    """A north-light roof over a block's top: `teeth` ridges across it, each a slope of slates
    rising from the back to a wall of glazing that faces the shaded side, its panes framed in
    light bars, and capped along its crest."""
    m = METRE
    left, front, right, back = top
    for k in range(teeth):
        t0, t1 = k / teeth, (k + 1) / teeth
        low_near, low_far = _on_wall(left, front, t0, 0), _on_wall(back, right, t0, 0)
        foot_near, foot_far = _on_wall(left, front, t1, 0), _on_wall(back, right, t1, 0)
        high_near, high_far = _on_wall(left, front, t1, -rise), _on_wall(back, right, t1, -rise)
        p.polygon([low_near, high_near, high_far, low_far], roof, shade(roof, 0.6))
        for up in (0.35, 0.7):
            _bar(
                p,
                _on_wall(low_near, high_near, up, 0),
                _on_wall(low_far, high_far, up, 0),
                0.06 * m,
                shade(roof, 0.8),
            )
        p.polygon(
            [foot_near, high_near, high_far, foot_far],
            shade(NORTH_LIGHT, 0.82),
            shade(NORTH_LIGHT, 0.5),
        )
        for s in (0.2, 0.4, 0.6, 0.8):
            _bar(
                p,
                _on_wall(foot_near, foot_far, s, 0),
                _on_wall(foot_near, foot_far, s, -rise),
                0.08 * m,
                shade(NORTH_LIGHT, 0.55),
            )
        _bar(p, high_near, high_far, 0.18 * m, shade(roof, 1.35))
        p.polygon([low_near, high_near, foot_near], wall, shade(wall, 0.55))


def _crates(p: Painter, x: float, y: float, rows: tuple[int, ...]) -> None:
    """Crates of goods a metre and a quarter a side stacked on (x, y): a row of `rows[0]` of them
    running along the lit walls, `rows[1]` on top of those and so on, slatted and lit on top."""
    m = METRE
    side = 1.25 * m
    for level, count in enumerate(rows):
        for k in range(count):
            cx, cy = _iso(x, y, (k + level * 0.5) * side, 0)
            left, front, right, _ = p.hall(
                cx, cy - level * side, side, side, side, GOODS_CRATE, shade(GOODS_CRATE, 1.15)
            )
            for a, b, tone in ((left, front, 0.72), (front, right, 0.55)):
                _bar(
                    p,
                    _on_wall(a, b, 0, side / 2),
                    _on_wall(a, b, 1, side / 2),
                    0.08 * m,
                    shade(GOODS_CRATE, tone),
                )
                _bar(
                    p,
                    _on_wall(a, b, 0.5, 0),
                    _on_wall(a, b, 0.5, side),
                    0.08 * m,
                    shade(GOODS_CRATE, tone),
                )


def _girders(p: Painter, x: float, y: float, long: float, layers: int) -> None:
    """Steel girders `long` metres long stacked `layers` deep on timber bearers at (x, y), their
    dark I-shaped ends showing on the shaded side."""
    m = METRE
    for along in (-0.3, 0.3):
        p.hall(
            x + along * long * m,
            y + along * long * m / 2,
            0.3 * m,
            2.2 * m,
            0.2 * m,
            shade(TIMBER, 0.6),
            None,
        )
    high = 0.35 * m * layers
    left, front, right, _ = p.hall(
        x, y - 0.2 * m, long * m, 1.8 * m, high, GIRDER, shade(GIRDER, 1.3)
    )
    for k in range(1, layers):
        _bar(
            p,
            _on_wall(left, front, 0, high * k / layers),
            _on_wall(left, front, 1, high * k / layers),
            0.06 * m,
            shade(GIRDER, 0.6),
        )
    for k in range(layers):
        for t in (0.25, 0.75):
            ex, ey = _on_wall(front, right, t, high * (k + 0.5) / layers)
            p.blob(ex - 0.2 * m, ey - 0.15 * m, ex + 0.2 * m, ey + 0.15 * m, shade(GIRDER, 0.4))


def _tall_chimney(p: Painter, x: float, y: float, tall: float) -> None:
    """A round brick mill chimney drawn for `tall` units and lowered (CHIMNEY_SHARE): a square
    plinth with a cream band, a tapering shaft ringed with iron bands, a cream band and a
    corbelled black crown at its top, smoking."""
    m = METRE
    tall *= CHIMNEY_SHARE
    base = 3.2 * m
    plinth = p.box(x, y, base / 2 * 1.4, base, MILL_BRICK, shade(MILL_BRICK, 1.1))
    _bands(p, plinth[0], plinth[1], plinth[2], 0.4 * m, DRESSING, shade(DRESSING, 0.78))
    foot = y - base
    shaft = tall - base
    r0, r1 = 1.5 * m, 0.95 * m
    _turned(p, x, foot, [(0, r0), (shaft, r1)], MILL_BRICK)

    def ring(up: float, wide: float, colour: RGB) -> None:
        half = r0 + (r1 - r0) * up / shaft
        y = foot - up
        for a, b in (((x - half, y), (x, y + half / 2)), ((x, y + half / 2), (x + half, y))):
            _bar(p, a, b, wide, colour)

    for up in (0.3, 0.55, 0.78):
        ring(shaft * up, 0.15 * m, (60, 52, 52))
    ring(shaft - 1.6 * m, 0.6 * m, DRESSING)
    _turned(
        p,
        x,
        foot - shaft,
        [(0, r1), (0.5 * m, r1 * 1.25), (1.3 * m, r1 * 1.25)],
        (72, 50, 44),
        (30, 28, 30),
    )
    _smoke(p, x + 1, foot - shaft - 4, (206, 206, 212), 2)


def factory() -> Sprite:
    """A Victorian brick factory, 28 by 13 m and two storeys high: rows of tall iron-framed
    windows with round heads under cream brick arches between pilasters, a north-light roof of
    seven glazed teeth, a stair tower at its end with a clock under a slate pyramid, an engine
    house with a tall round chimney smoking behind it, a big arched loading door with a hoist
    over it and a cart track worn out of it, and a cinder yard of crated goods, sawn planks and
    steel girders."""
    m = METRE
    p = Painter(48, 46, (24, 31))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 16.5) * m, (v - 6.5) * m)

    length, depth, high = 28 * m, 13 * m, 9.5 * m
    p.polygon([at(26, -4), at(36.5, -4), at(36.5, 13), at(26, 13)], CINDERS)
    foot = _footing(*at(14, 6.5), length, depth)
    _bed(p, foot)
    _bed(p, _footing(*at(-2.5, 2.5), 5 * m, 5 * m), margin=0.5)
    p.shadow(17, 5, 4)
    _track(p, foot[0], foot[1], (0.6, 0.72), 6, (140, 122, 92))
    _tall_chimney(p, *at(28, 17), 45 * m)
    tower = p.hall(*at(-2.5, 2.5), 5 * m, 5 * m, 17 * m, MILL_BRICK, None)
    t_left, t_front, t_right, _ = tower
    _plinth(p, tower, 17 * m, MILL_BRICK)
    for a, b in ((t_left, t_front), (t_front, t_right)):
        for drop in (0.3 * m, 4.9 * m):
            _bar(p, _on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop), 0.3 * m, DRESSING)
        for t in (0.03, 0.97):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, 16.3 * m), 0.35 * m, DRESSING)
        for head in (6 * m, 10 * m, 13.6 * m):
            _mill_window(p, a, b, 5 * m, 0.5, 1 * m, (head, head + 1.9 * m))
    cx, cy = _on_wall(t_left, t_front, 0.5, 2.6 * m)
    p.ellipse(cx - 1.5 * m, cy - 1.5 * m, cx + 1.5 * m, cy + 1.5 * m, DRESSING)
    p.ellipse(cx - 1.2 * m, cy - 1.2 * m, cx + 1.2 * m, cy + 1.2 * m, (246, 242, 230))
    _bar(p, (cx, cy), (cx, cy - 0.95 * m), 0.14 * m, (30, 30, 34))
    _bar(p, (cx, cy), (cx + 0.65 * m, cy + 0.2 * m), 0.14 * m, (30, 30, 34))
    apex = ((t_left[0] + t_right[0]) / 2, (t_left[1] + t_right[1]) / 2 - 4 * m)
    p.polygon([t_left, t_front, apex], MILL_SLATE, shade(MILL_SLATE, 0.6))
    p.polygon([t_front, t_right, apex], shade(MILL_SLATE, 0.72), shade(MILL_SLATE, 0.5))
    _bar(p, apex, (apex[0], apex[1] - 2 * m), 0.12 * m, (50, 50, 56))
    p.polygon(
        [
            (apex[0], apex[1] - 2 * m),
            (apex[0] + 1.2 * m, apex[1] - 1.7 * m),
            (apex[0], apex[1] - 1.3 * m),
        ],
        (214, 170, 70),
    )
    top = p.hall(*at(14, 6.5), length, depth, high, MILL_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, MILL_BRICK, 0.7 * m)
    _stains(p, left, front, high, (0.05, 0.31, 0.55, 0.93), shade(MILL_BRICK, 0.66), 1.6 * m)
    bays = 8
    for a, b in ((left, front), (front, right)):
        for drop, wide in ((0, 0.5 * m), (4.6 * m, 0.25 * m)):
            _bar(
                p,
                _on_wall(a, b, 0, drop + wide / 2),
                _on_wall(a, b, 1, drop + wide / 2),
                wide,
                DRESSING,
            )
    for k in range(bays + 1):
        t = k / bays
        _bar(
            p,
            _on_wall(left, front, t, 0.5 * m),
            _on_wall(left, front, t, high),
            0.5 * m,
            shade(MILL_BRICK, 0.86),
        )
    door = 5
    for k in range(bays):
        t = (k + 0.5) / bays
        if k == door:
            continue
        _mill_window(p, left, front, length, t, 1.5 * m, (1.3 * m, 3.9 * m))
        _mill_window(p, left, front, length, t, 1.7 * m, (5.4 * m, 8.4 * m))
    t = (door + 0.5) / bays
    p.polygon(_round_head(left, front, length, t, 3.6 * m, 4.9 * m, high), DRESSING)
    p.polygon(_round_head(left, front, length, t, 3 * m, 5.2 * m, high), DOORWAY)
    _patch(p, left, front, (t - 0.054, t - 0.002), (6.6 * m, high), DOOR)
    for s in (t - 0.04, t - 0.02):
        _bar(
            p,
            _on_wall(left, front, s, 6.6 * m),
            _on_wall(left, front, s, high),
            0.06 * m,
            shade(DOOR, 0.7),
        )
    _patch(p, left, front, (t - 0.03, t + 0.03), (1.3 * m, 3.9 * m), DOOR)
    beam = _on_wall(left, front, t, 0.9 * m)
    _bar(p, beam, (beam[0] - 1.6 * m, beam[1] + 0.8 * m), 0.2 * m, (54, 50, 50))
    hook = (beam[0] - 1.5 * m, beam[1] + 0.8 * m)
    _bar(p, hook, (hook[0], hook[1] + 2.4 * m), 0.05 * m, (40, 38, 38))
    for k in range(3):
        t = (k + 0.5) / 3
        _mill_window(
            p, front, right, depth, t, 1.5 * m, (1.3 * m, 3.9 * m), shade(MILL_GLASS, 0.85)
        )
        _mill_window(
            p, front, right, depth, t, 1.7 * m, (5.4 * m, 8.4 * m), shade(MILL_GLASS, 0.85)
        )
    _north_lights(p, top, 7, 2.8 * m, MILL_SLATE, MILL_BRICK)
    engine = p.hall(*at(31, 9.5), 6 * m, 7 * m, 6 * m, MILL_BRICK, None)
    e_left, e_front, e_right, _ = engine
    _plinth(p, engine, 6 * m, MILL_BRICK, 0.6 * m)
    _bands(p, e_left, e_front, e_right, 0.3 * m, DRESSING, shade(DRESSING, 0.78))
    _mill_window(p, e_left, e_front, 6 * m, 0.5, 2 * m, (1.2 * m, 4.6 * m))
    _patch(p, e_front, e_right, (0.3, 0.7), (2.4 * m, 6 * m), DOORWAY)
    p.gable(engine, 2.2 * m, MILL_SLATE, MILL_BRICK)
    _roof_rows(p, engine, 2.2 * m, MILL_SLATE)
    _girders(p, *at(33, 2), 6, 3)
    _planks(p, *at(28.5, -1.5), 5, 1)
    _crates(p, *at(34, -2.5), (3, 2, 1))
    _crates(p, *at(18.5, -3.5), (2, 1))
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
        cap = _stack(p, x, -9, -24, shade(BRICK, 0.8))
    for i, (x, y, r) in enumerate([(-5, -27, 2), (-3, -30, 2.5), (5, -27, 2), (7, -30, 2.5)]):
        y += cap + 24
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
    _bands(p, left, front, right, 1, shade(wall, 1.15), shade(wall, 0.9))
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
        _smoking_chimney(p, x, -9, -23, shade(BRICK, 0.8), puffs=2)
    for x, y in ((9, 6.5), (11, 5.5)):
        _crate(p, x, y)
    _oil(p, 4.5, 5.5, 1.2)
    return p.sprite()


def _sawtooth(
    p: Painter,
    top: Top,
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


def _chimney(p: Painter, x: float, foot: float, top: float, colour: RGB = BRICK) -> float:
    """A round brick chimney narrowing to a dark cap, with a light band under the cap and a dark
    one halfway down, lowered (chimney_top); returns where its cap is."""
    top = chimney_top(foot, top)
    p.polygon(
        [(x - 2, foot), (x - 1, top), (x + 1, top), (x + 2, foot)], colour, shade(colour, 0.6)
    )
    p.line([(x - 1, foot - 1), (x - 1, top + 1)], shade(colour, 1.2))
    for y, tone in ((top + 3, (220, 220, 214)), ((top + foot) / 2, shade(colour, 0.65))):
        half = 1 + (y - top) / (foot - top)
        p.line([(x - half, y), (x + half, y)], tone)
    p.rect(x - 2, top - 1, x + 2, top, shade(colour, 0.7))
    return top


def _smoke(p: Painter, x: float, y: float, colour: RGB = SMOKE, puffs: int = 3) -> None:
    """Puffs of smoke rising to the right from (x, y), at most MOST_PUFFS of them."""
    for i in range(min(puffs, MOST_PUFFS)):
        r = 1.5 + 0.6 * i
        cx, cy = x + 1.5 * i, y - 2.5 * i
        p.blob(cx - r, cy - r, cx + r, cy + r, colour if i % 2 else shade(colour, 0.86))


def _smoking_chimney(
    p: Painter,
    x: float,
    foot: float,
    top: float,
    colour: RGB = BRICK,
    smoke: RGB = SMOKE,
    puffs: int = 3,
    close: bool = False,
) -> None:
    """A chimney (_chimney) with smoke rising from its cap, puffing from right over it if `close`
    and a little higher and further right otherwise."""
    top = _chimney(p, x, foot, top, colour)
    if close:
        _smoke(p, x + 0.5, top - 2, smoke, puffs)
    else:
        _smoke(p, x + 1, top - 3, smoke, puffs)


def _gear(p: Painter, x: float, y: float, radius: float, colour: RGB) -> None:
    for k in range(8):
        a = k * math.pi / 4
        tx, ty = x + (radius + 0.8) * math.cos(a), y + (radius + 0.8) * math.sin(a)
        p.blob(tx - 1, ty - 1, tx + 1, ty + 1, shade(colour, 0.8))
    p.ellipse(x - radius, y - radius, x + radius, y + radius, colour)
    p.blob(x - 1, y - 1, x + 1, y + 1, shade(colour, 0.45))


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


def _on_roof(top: Top, rise: float, along: float, up: float) -> Point:
    """The point on the lit side of a gable roof `along` of the way from its back to its front and
    `up` of the way from its eave to its ridge."""
    left, front, _, _ = top
    ridge_back, ridge_front = _ridge(top, rise)
    ex, ey = _on_wall(left, front, along, 0)
    rx, ry = _on_wall(ridge_back, ridge_front, along, 0)
    return ex + (rx - ex) * up, ey + (ry - ey) * up


def _roof_rows(
    p: Painter,
    top: Top,
    rise: float,
    roof: RGB,
    rows: tuple[float, ...] = (0.34, 0.68),
) -> None:
    """Rows of tiles across the lit side of a gable roof."""
    for up in rows:
        p.line([_on_roof(top, rise, 0, up), _on_roof(top, rise, 1, up)], shade(roof, 0.8))


def _skylight(
    p: Painter,
    top: Top,
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


def _stack(p: Painter, x: float, foot: float, top: float, colour: RGB) -> float:
    """A square stack two wide from `foot` up to `top`, with a light band under a dark cap,
    lowered (chimney_top); returns where its cap is."""
    top = chimney_top(foot, top)
    p.rect(x, top, x + 2, foot, colour)
    p.line([(x, top + 2), (x + 2, top + 2)], (220, 220, 214))
    p.line([(x, (top + foot) / 2), (x + 2, (top + foot) / 2)], shade(colour, 0.65))
    p.rect(x - 0.5, top - 1, x + 2.5, top, shade(colour, 0.6))
    return top


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


def _footing(x: float, y: float, length: float, depth: float) -> Top:
    """The left, front, right and back corners on the ground of a block that Painter.hall raises at
    (x, y) with the same `length` and `depth`; a box of half width s has both s."""
    front = (x + (length - depth) / 2, y + (length + depth) / 4)
    left = (front[0] - length, front[1] - length / 2)
    right = (front[0] + depth, front[1] - depth / 2)
    return left, front, right, (left[0] + depth, left[1] - depth / 2)


def _bed(p: Painter, foot: Top, colour: RGB = SOIL, margin: float = 1.2) -> None:
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
    cap = _stack(p, -3, -9, -30, shade(BRICK, 0.7))
    for i, (x, y, r) in enumerate([(-2, -33, 2), (0, -36, 3)]):
        y += cap + 30
        p.blob(x - r, y - r, x + r, y + r, (150, 148, 150) if i % 2 else (124, 122, 126))
    for gx, gy in ((-9.5, 3.4), (-8, 4.4)):
        p.point(gx, gy, (226, 120, 44))
    ingot = (128, 128, 136)
    for lift in (0, 1.2):
        p.hall(-3, 7 - lift, 4, 2, 1.2, ingot, shade(ingot, 1.2))
    return p.sprite()


def textile_mill() -> Sprite:
    """A yellow-brick cotton mill of Łódź: a long block of storeys with rows of small arched windows
    over a dark plinth stained by the damp, a drainpipe and a cornice, a sawtooth weaving shed at
    its end, a tall banded round chimney, cart tracks worn out of both doors and bales of cloth
    by the door."""
    p = Painter(41, 55, (21, 44))
    wall, roof = (206, 164, 92), (80, 76, 84)
    mill_foot, shed_foot = _footing(-4, -1, 14, 8), _footing(7, 4.5, 8, 8)
    _bed(p, mill_foot)
    _bed(p, shed_foot)
    p.shadow(15, 5, reach=4)
    _track(p, mill_foot[0], mill_foot[1], (0.68, 0.84), 3.5)
    _track(p, shed_foot[1], shed_foot[2], (0.4, 0.75), 3)
    _smoking_chimney(p, 11, 0, -32, shade(wall, 0.92))
    top = p.hall(-4, -1, 14, 8, 15, wall, roof)
    left, front, right, _ = top
    _plinth(p, top, 15, wall)
    _stains(p, left, front, 15, (0.05, 0.24, 0.44, 0.94), shade(wall, 0.6), 2)
    _bands(p, left, front, right, 1, shade(wall, 1.2), shade(wall, 0.95))
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
    _spool(p, *_on_wall(left, front, 0.36, 3))
    _drying_cloth(p, (-20, 1.5), (-6, 8.5), 8)
    return p.sprite()


DYES: list[RGB] = [(206, 52, 50), (56, 98, 200), (238, 196, 56), (64, 158, 90), (150, 70, 170)]


def _spool(p: Painter, x: float, y: float) -> None:
    """A big cotton reel hung on a wall at (x, y) as a mill's sign, wound with red thread."""
    p.line([(x, y - 1), (x, y + 1)], (60, 56, 54))
    p.rect(x - 2.2, y + 1, x + 2.2, y + 1.8, (226, 200, 150))
    p.rect(x - 1.5, y + 1.8, x + 1.5, y + 4.8, DYES[0])
    p.line([(x - 1.5, y + 2.8), (x + 1.5, y + 2.8)], shade(DYES[0], 1.3))
    p.rect(x - 2.2, y + 4.8, x + 2.2, y + 5.6, (226, 200, 150))


def _drying_cloth(p: Painter, a: Point, b: Point, height: float) -> None:
    """A rack of dyed cloth hung out to dry: a pole on two posts from `a` to `b` on the ground,
    `height` up, with lengths of cloth in every dye hanging from it."""
    post = (110, 84, 60)
    for x, y in (a, b):
        p.line([(x, y), (x, y - height)], post)
    p.line([(a[0], a[1] - height), (b[0], b[1] - height)], shade(post, 1.2))
    pieces = 5
    for k in range(pieces):
        t0, t1 = (k + 0.1) / pieces, (k + 0.9) / pieces
        top0, top1 = _on_wall(a, b, t0, -height + 0.4), _on_wall(a, b, t1, -height + 0.4)
        drop = height - 1.6 - 0.8 * (k % 2)
        colour = DYES[k % len(DYES)]
        p.polygon(
            [top0, top1, (top1[0], top1[1] + drop), (top0[0], top0[1] + drop)],
            colour,
            shade(colour, 0.7),
        )
        p.line([top0, (top0[0], top0[1] + drop)], shade(colour, 1.25))


def _hide_frame(p: Painter, x: float, y: float, hide: RGB) -> None:
    """A hide stretched to dry on a wooden frame standing on (x, y), dripping on the ground."""
    p.blob(x + 0.5, y - 0.5, x + 4, y + 1, (100, 108, 60))
    p.line([(x, y), (x, y - 7)], FRAME)
    p.line([(x + 4, y - 1), (x + 4, y - 8)], FRAME)
    p.line([(x, y - 7), (x + 4, y - 8)], FRAME)
    p.polygon(
        [(x + 1, y - 6), (x + 3, y - 7), (x + 3.5, y - 2), (x + 2, y - 1.5), (x + 0.5, y - 1)],
        hide,
        shade(hide, 0.65),
    )
    p.line([(x + 1, y - 6), (x + 3, y - 7)], FRAME)


def _chair(p: Painter, x0: float, y0: float, s: float, colour: RGB = CHAIR) -> None:
    """A chair seen side on, standing on (x0, y0), `s` times the size of the small sign."""
    pts = ((0, 0), (0, -7), (1, -7), (1, -3), (5, -3), (5, 0), (4, 0), (4, -2), (1, -2), (1, 0))
    p.polygon([(x0 + dx * s, y0 + dy * s) for dx, dy in pts], colour)
    p.line([(x0 + 1 * s, y0 - 3 * s), (x0 + 5 * s, y0 - 3 * s)], shade(colour, 1.35))


def _plank_walls(p: Painter, top: Top, height: float) -> None:
    left, front, right, _ = top
    for drop in range(2, int(height), 2):
        _bands(p, left, front, right, drop, shade(JOINER_WALL, 0.86), shade(JOINER_WALL, 0.68))
    _plinth(p, top, height, JOINER_WALL)
    _stains(p, front, right, height, (0.2, 0.55, 0.9), shade(JOINER_WALL, 0.55), 1.4)


def _anvil(p: Painter, cx: float, cy: float, s: float = 1) -> None:
    pts = ((-3, -1), (2, -1), (2, 0), (1, 0), (1, 1), (2, 2), (-2, 2), (-1, 1), (-1, 0), (-2, 0))
    p.polygon([(cx + dx * s, cy + dy * s) for dx, dy in pts], ANVIL)
    p.line([(cx - 2 * s, cy - s), (cx + 1 * s, cy - s)], (110, 110, 120))


def _forge_stack(p: Painter, x: float, foot: float, top: float) -> float:
    """A dark forge stack glowing at its cap, lowered (chimney_top); returns where its cap is."""
    top = chimney_top(foot, top)
    p.rect(x - 1, top, x + 1, foot, (74, 58, 52))
    p.line([(x - 1, top + 2), (x + 1, top + 2)], (200, 190, 180))
    p.rect(x - 1.5, top - 1, x + 1.5, top, (50, 40, 36))
    p.point(x, top - 1, (255, 170, 60))
    return top


def _pagoda(p: Painter, x: float, y: float, size: float = 1) -> None:
    """A whitewashed kiln vent on a roof at (x, y) under a flared slate pagoda cap."""
    s = size
    p.rect(x - 1.5 * s, y - 2 * s, x + 1.5 * s, y + 0.5, WHITEWASH)
    p.line([(x - s, y - s), (x + s, y - s)], shade(WHITEWASH, 0.6))
    p.polygon(
        [
            (x - 3 * s, y - 1.5 * s),
            (x - s, y - 3 * s),
            (x, y - 5 * s),
            (x + s, y - 3 * s),
            (x + 3 * s, y - 1.5 * s),
        ],
        STILL_SLATE,
        shade(STILL_SLATE, 0.6),
    )
    p.point(x, y - 6 * s, (60, 60, 64))


def _bundle(p: Painter, x: float, y: float) -> None:
    """A bundle of papers tied with string, standing on (x, y)."""
    bundle = p.box(x, y, 2, 1.6, BUNDLE, shade(BUNDLE, 1.04))
    p.line([bundle[1], (bundle[1][0], bundle[1][1] + 1.6)], (150, 110, 80))


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
    wall, board, roof = JOINER_WALL, (226, 190, 134), (98, 74, 58)
    for x, y, h in ((9, -3, 6), (7, 2, 4)):
        for t in (0.15, 0.85):
            bx, by = _on_wall(*_footing(x, y, 8, 3)[:2], t, 0)
            p.rect(bx - 0.5, by - 1, bx + 0.5, by + 0.5, (110, 80, 56))
        stack = p.hall(x, y - 0.8, 8, 3, h, board, shade(board, 1.12))
        gap = shade(board, 0.5)
        for drop in range(2, h, 2):
            _band(p, stack[0], stack[1], drop, gap)
            _band(p, stack[1], stack[2], drop, gap)
        p.point(*_on_wall(stack[1], stack[2], 0.5, h - 1), shade(board, 0.7))
    top = p.box(-4, 0, 7, 6, wall, None)
    left, front, right, _ = top
    _plank_walls(p, top, 6)
    _window(p, left, front, (0.1, 0.32), 1.8)
    _door(p, left, front, (0.45, 0.82), 1.8, 6, (56, 40, 30), shade(wall, 1.3))
    p.line([_on_wall(left, front, 0.635, 1.9), _on_wall(left, front, 0.635, 6)], (88, 64, 44))
    _window(p, front, right, (0.3, 0.6), 1.8, shade(WINDOW, 0.8))
    p.gable(top, 4, roof, wall)
    _roof_rows(p, top, 4, roof)
    _chair(p, -6, -10, 1)
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
    _anvil(p, *_on_wall(left, front, 0.25, 3.5))
    _windows(p, front, right, (0.3, 0.7), (3,), 1, (240, 150, 60), shade(wall, 1.5))
    p.gable(top, 4, roof, wall)
    _roof_rows(p, top, 4, roof)
    cap = _forge_stack(p, 0, -10, -17)
    _smoke(p, 1, cap - 4, (96, 94, 98))
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
    wall, roof = (238, 224, 192), (118, 40, 66)
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
    _bands(p, left, front, right, 7.9, (120, 124, 128), (104, 108, 112))
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
    steel = (104, 118, 136)
    top = p.hall(-2, -2, 16, 9, 10, steel, None)
    left, front, right, _ = top
    lintel = shade(steel, 1.3)
    _plinth(p, top, 10, steel)
    _stains(p, left, front, 10, (0.04, 0.21, 0.39, 0.57, 0.73), shade(steel, 0.62), 2)
    _stains(p, front, right, 10, (0.2, 0.8), shade(steel, 0.5), 2)
    _bands(p, left, front, right, 1, lintel, shade(steel, 1.05))
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2.5,), 5, arch=lintel)
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (8,), 1, shade(steel, 1.15))
    p.line([_on_wall(left, front, 0.76, 1.4), _on_wall(left, front, 0.76, 9.6)], (74, 64, 62))
    _door(p, left, front, (0.8, 0.94), 5, 10, (58, 48, 44), lintel)
    p.line([_on_wall(left, front, 0.87, 5), _on_wall(left, front, 0.87, 10)], (44, 36, 34))
    p.gable(top, 5, (82, 84, 96), steel)
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
    _smoking_chimney(p, x, y - 8, y - 35, shade(BRICK, 0.9), puffs=2)
    _heap(p, -12, 7, 3, 2.5, (176, 122, 86))
    for bx, by in ((13, 2), (8.5, 5.5)):
        stack = p.hall(bx, by, 4, 3, 3, BRICK, shade(BRICK, 1.25))
        for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
            _band(p, a, b, 1.5, shade(BRICK, 1.3))
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
        _band(p, tower[0], tower[1], drop, shade(grey, 0.6))
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


GLASS_BRICK: RGB = (178, 94, 64)
GLASS_SOOT: RGB = (64, 50, 46)
GLASS_GLOW: RGB = (246, 140, 44)
GLASS_HOT: RGB = (255, 218, 128)
GLASS_AQUA: RGB = (150, 214, 212)
GLASS_GREEN: RGB = (62, 148, 92)
GLASS_AMBER: RGB = (206, 130, 42)
GLASS_SAND: RGB = (228, 210, 160)
GLASS_SODA: RGB = (242, 242, 238)
GLASS_YARD: RGB = (150, 140, 126)
GLASS_SLATE: RGB = (78, 82, 94)


def _glass_cone(p: Painter, x: float, y: float, r: float, tall: float) -> None:
    """An English glass cone of brick `r` across its foot and `tall` high standing on (x, y):
    its walls curving in as they rise, bound with iron hoops, sooted black towards its open top
    and smoking, the furnace mouth glowing in a deep arch at its foot and its light spilling
    over the ground before it."""
    m = METRE
    shape = ((0, 1.0), (0.1, 0.97), (0.3, 0.82), (0.55, 0.6), (0.8, 0.42), (1.0, 0.32))

    def ring(h: float, w: float, a0: float, a1: float) -> list[Point]:
        return [(x + w * math.cos(a), y - h + w / 2 * math.sin(a)) for a in np.linspace(a0, a1, 9)]

    split = math.acos(1 / 3)
    left = [(x - r * w, y - tall * h) for h, w in shape]
    middle = [(x + r * w / 3, y - tall * h + r * w / 2 * math.sin(split)) for h, w in shape]
    right = [(x + r * w, y - tall * h) for h, w in shape]
    p.polygon([*ring(0, r, split, math.pi), *left, *middle[::-1]], GLASS_BRICK)
    p.polygon([*ring(0, r, 0, split), *middle, *right[::-1]], shade(GLASS_BRICK, 0.74))
    soot = [(h, w) for h, w in shape if h >= 0.8]
    s_left = [(x - r * w, y - tall * h) for h, w in soot]
    s_mid = [(x + r * w / 3, y - tall * h + r * w / 2 * math.sin(split)) for h, w in soot]
    s_right = [(x + r * w, y - tall * h) for h, w in soot]
    p.polygon([*s_left, *s_mid[::-1]], GLASS_SOOT)
    p.polygon([*s_mid, *s_right[::-1]], shade(GLASS_SOOT, 0.8))
    _bar(
        p,
        (x - r * 0.72, y - tall * 0.08),
        (x - r * 0.45, y - tall * 0.7),
        0.35 * m,
        shade(GLASS_BRICK, 1.1),
    )
    for h in (0.25, 0.45, 0.65):
        w = r * np.interp(h, [a for a, _ in shape], [b for _, b in shape])
        hoop = ring(tall * h, w, 0, math.pi)
        for k in range(len(hoop) - 1):
            _bar(p, hoop[k], hoop[k + 1], 0.14 * m, (58, 48, 46))
    top = r * shape[-1][1]
    p.ellipse(x - top, y - tall - top / 2, x + top, y - tall + top / 2, (36, 30, 30))
    lip = ring(tall, top, 0, math.pi)
    for k in range(len(lip) - 1):
        _bar(p, lip[k], lip[k + 1], 0.25 * m, shade(GLASS_SOOT, 1.3))
    _smoke(p, x + 0.5, y - tall - 2.5, (190, 188, 190), 2)
    mx, my = x + r * math.cos(math.radians(118)), y + r / 2 * math.sin(math.radians(118))
    p.soft([(mx - 3 * m, my - 0.6 * m, mx + 2.4 * m, my + 2.2 * m)], GLASS_GLOW, 0.45, 1)
    arch = [(mx - 1.6 * m, my), (mx + 1.4 * m, my - 0.2 * m), (mx + 1.4 * m, my - 2.8 * m)]
    arch += [
        (mx - 0.1 * m + 1.5 * m * math.cos(a), my - 2.9 * m - 1.3 * m * math.sin(a))
        for a in np.linspace(0, math.pi, 9)
    ]
    p.polygon(arch, shade(GLASS_BRICK, 1.2))
    inner = [
        (mx - 1.1 * m, my - 0.1 * m),
        (mx + 0.9 * m, my - 0.25 * m),
        (mx + 0.9 * m, my - 2.6 * m),
    ]
    inner += [
        (mx - 0.1 * m + 1 * m * math.cos(a), my - 2.7 * m - 1 * m * math.sin(a))
        for a in np.linspace(0, math.pi, 9)
    ]
    p.polygon(inner, GLASS_GLOW)
    p.blob(mx - 0.7 * m, my - 2.4 * m, mx + 0.5 * m, my - 0.3 * m, GLASS_HOT)
    for dx, dy in ((-2, 0.9), (-0.6, 1.4), (-2.8, 1.8)):
        p.blob(
            mx + (dx - 0.12) * m,
            my + (dy - 0.08) * m,
            mx + (dx + 0.12) * m,
            my + (dy + 0.08) * m,
            GLASS_GLOW,
        )


def _glass_house(
    p: Painter,
    x: float,
    y: float,
    size: tuple[float, float, float],
    glazed: bool = False,
) -> Top:
    """A brick glasshouse where the glass is blown, `size` (length, depth and height to its eaves)
    on (x, y): tall arched windows glowing orange from the furnaces inside, cream dressings, and
    a slate roof, or one glazed in aqua panes if `glazed`, with a louvred vent along its ridge
    letting out the heat in a glow."""
    m = METRE
    length, depth, high = size
    top = p.hall(x, y, length, depth, high, GLASS_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, GLASS_BRICK, 0.5 * m)
    _bands(p, left, front, right, 0.3 * m, DRESSING, shade(DRESSING, 0.78))
    for a, b, span, tone in ((left, front, length, 1.0), (front, right, depth, 0.85)):
        bays = max(2, round(span / m / 3))
        for k in range(bays):
            _mill_window(
                p,
                a,
                b,
                span,
                (k + 0.5) / bays,
                1.3 * m,
                (1 * m, high - 1.3 * m),
                shade(GLASS_GLOW, 0.9 * tone),
            )
    roof = shade(GLASS_AQUA, 0.86) if glazed else GLASS_SLATE
    p.gable(top, 2.2 * m, roof, GLASS_BRICK)
    if glazed:
        for k in range(1, 12):
            t = k / 12
            _bar(
                p,
                _on_roof(top, 2.2 * m, t, 0),
                _on_roof(top, 2.2 * m, t, 1),
                0.08 * m,
                shade(GLASS_AQUA, 0.6),
            )
        _bar(
            p,
            _on_roof(top, 2.2 * m, 0.05, 0.6),
            _on_roof(top, 2.2 * m, 0.45, 0.6),
            0.12 * m,
            (240, 252, 250),
        )
    else:
        _roof_rows(p, top, 2.2 * m, GLASS_SLATE, (0.5,))
    ridge_back, ridge_front = _ridge(top, 2.2 * m)
    vx, vy = _on_wall(ridge_back, ridge_front, 0.5, 0)
    vent = p.hall(vx, vy + 0.4 * m, length * 0.6, 1.4 * m, 1 * m, shade(GLASS_GLOW, 0.8), None)
    for k in range(1, 4):
        drop = 1 * m * k / 4
        _bar(
            p,
            _on_wall(vent[0], vent[1], 0, drop),
            _on_wall(vent[0], vent[1], 1, drop),
            0.12 * m,
            GLASS_SOOT,
        )
    p.gable(vent, 0.6 * m, GLASS_SLATE, shade(GLASS_GLOW, 0.8))
    return top


def _glass_lehr(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """A long low lehr shed `length` by `depth` on (x, y) where new glass cools slowly: brick
    walls with small windows glowing amber, a slate roof and a row of short flues along its
    ridge."""
    m = METRE
    high = 3.2 * m
    top = p.hall(x, y, length, depth, high, shade(GLASS_BRICK, 1.08), None)
    left, front, right, _ = top
    _plinth(p, top, high, GLASS_BRICK, 0.4 * m)
    bays = max(3, round(length / m / 2.5))
    for k in range(bays):
        t = (k + 0.5) / bays
        _patch(
            p,
            left,
            front,
            (t - 0.25 / bays, t + 0.25 / bays),
            (0.9 * m, 1.9 * m),
            shade(GLASS_AMBER, 1.15),
        )
    _patch(p, front, right, (0.3, 0.7), (1 * m, high), DOORWAY)
    p.gable(top, 1.4 * m, GLASS_SLATE, shade(GLASS_BRICK, 1.08))
    _roof_rows(p, top, 1.4 * m, GLASS_SLATE, (0.5,))
    ridge_back, ridge_front = _ridge(top, 1.4 * m)
    for k in range(max(2, bays // 2)):
        fx, fy = _on_wall(ridge_back, ridge_front, (k + 0.5) / max(2, bays // 2), 0)
        p.hall(fx, fy + 0.3 * m, 0.7 * m, 0.7 * m, 1.3 * m, shade(GLASS_BRICK, 0.85), GLASS_SOOT)


def _glass_packing(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """A boarded packing house `length` by `depth` on (x, y), its wide door open on straw and
    crates, under a felted roof."""
    m = METRE
    wall, high = (150, 118, 84), 4 * m
    top = p.hall(x, y, length, depth, high, wall, None)
    left, front, right, _ = top
    _plinth(p, top, high, wall, 0.4 * m)
    for a, b, tone in ((left, front, 0.8), (front, right, 0.62)):
        for t in np.linspace(0.04, 0.96, 12):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, high), 0.04 * m, shade(wall, tone))
    _patch(p, left, front, (0.3, 0.7), (1.2 * m, high), (52, 44, 40))
    _patch(p, left, front, (0.32, 0.68), (high - 0.7 * m, high), (226, 196, 120))
    _patch(p, left, front, (0.36, 0.5), (high - 1.5 * m, high - 0.5 * m), GOODS_CRATE)
    _patch(p, front, right, (0.35, 0.65), (1 * m, 2 * m), shade(WINDOW, 0.8))
    p.gable(top, 1.6 * m, (70, 66, 64), wall)
    _roof_rows(p, top, 1.6 * m, (70, 66, 64), (0.5,))


def _glass_crates(p: Painter, x: float, y: float, bottles: tuple[RGB, ...]) -> None:
    """A row of open crates of new bottles along the lit walls from (x, y), one crate for each
    of `bottles`, its colour of glass showing in rows of necks, each glinting."""
    m = METRE
    for k, colour in enumerate(bottles):
        cx, cy = _iso(x, y, k * 1.3 * m, 0)
        left, front, right, back = p.hall(
            cx, cy, 1.2 * m, 0.9 * m, 0.55 * m, GOODS_CRATE, shade(colour, 0.6)
        )
        _bar(
            p,
            _on_wall(left, front, 0, 0.28 * m),
            _on_wall(left, front, 1, 0.28 * m),
            0.06 * m,
            shade(GOODS_CRATE, 0.7),
        )
        for row in (0.3, 0.7):
            for col in (0.2, 0.5, 0.8):
                bx, by = _on_wall(
                    _on_wall(left, back, row, 0), _on_wall(front, right, row, 0), col, 0
                )
                p.blob(bx - 0.13 * m, by - 0.3 * m, bx + 0.13 * m, by + 0.05 * m, colour)
        gx, gy = _on_wall(_on_wall(left, back, 0.3, 0), _on_wall(front, right, 0.3, 0), 0.2, 0)
        _bar(
            p,
            (gx - 0.05 * m, gy - 0.25 * m),
            (gx - 0.05 * m, gy - 0.1 * m),
            0.06 * m,
            (240, 252, 246),
        )


def _glass_heap(p: Painter, x: float, y: float, r: float, tall: float, colour: RGB) -> None:
    """A cone of loose sand or soda `r` across its foot and `tall` high on (x, y), lit on its
    left."""
    split = math.radians(65)
    apex = (x + r * 0.05, y - tall)
    lit = [(x + r * math.cos(a), y + r / 2 * math.sin(a)) for a in np.linspace(math.pi, split, 9)]
    dark = [(x + r * math.cos(a), y + r / 2 * math.sin(a)) for a in np.linspace(split, 0, 5)]
    p.polygon([*lit, apex], colour)
    p.polygon([*dark, apex], shade(colour, 0.8))


def _glass_yard(p: Painter, spots: list[tuple[Point, float]]) -> None:
    """The trodden yard of a glassworks spread round each of `spots`, an oval `half` metres
    across, solid in its middle and soft at its rim."""
    m = METRE
    p.soft(
        [
            (x - half * m, y - half * m / 2, x + half * m, y + half * m / 2)
            for (x, y), half in spots
        ],
        GLASS_YARD,
        1,
        0.5,
    )


def glassworks() -> Sprite:
    """A glassworks round an English glass cone of brick 24 m high and 14 m across, hooped with
    iron and sooted at its smoking top, its furnace mouth glowing in a deep arch and lighting the
    ground; joined to it a brick glasshouse whose tall arched windows and ridge vent glow with
    the furnaces, then a long low lehr shed with amber windows and flues; a boarded packing
    house; and a yard of crates of bottles in green and amber glass, racks of window panes
    glinting aqua, and heaps of sand and soda."""
    m = METRE
    p = Painter(44, 45, (21, 32))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 16) * m, (v - 6) * m)

    _glass_yard(p, [(at(14, 5), 13), (at(5, -1), 7), (at(26, 2), 8), (at(31, 7), 6)])
    for u, v, length, depth in ((19, 8.5, 10, 9), (30, 8, 12, 5), (31, -1, 7, 5)):
        _bed(p, _footing(*at(u, v), length * m, depth * m), margin=0.8)
    _glass_cone(p, *at(8, 8), 7 * m, 24 * m)
    _glass_house(p, *at(19, 8.5), (10 * m, 9 * m, 6.5 * m))
    _glass_lehr(p, *at(30, 8), 12 * m, 5 * m)
    _glass_packing(p, *at(31, -1), 7 * m, 5 * m)
    _glass_heap(p, *at(-1, 2), 2.8 * m, 2.4 * m, GLASS_SAND)
    _glass_heap(p, *at(1.5, -1.5), 1.8 * m, 1.6 * m, GLASS_SODA)
    _glass_rack(p, *at(11, -3.5))
    _glass_rack(p, *at(15.5, -4))
    _glass_crates(p, *at(21, -2), (GLASS_GREEN, GLASS_AMBER, GLASS_GREEN))
    _glass_crates(p, *at(21.6, -3.6), (GLASS_AMBER, GLASS_GREEN))
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
    slate = STILL_SLATE
    cap = chimney_top(-4, -20)
    p.rect(3, cap, 4, -4, (150, 80, 60))
    for y in (cap + 2, (cap - 4) / 2):
        p.line([(3, y), (4, y)], (230, 228, 222))
    p.rect(2.5, cap - 1, 4.5, cap, (90, 52, 42))
    _smoke(p, 4, cap - 3, (190, 190, 194), 2)
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
    _pagoda(p, *_on_roof(top, 4, 0.35, 1))
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


def _straw_cask(p: Painter, x: float, y: float) -> None:
    """A cask of sake wrapped in straw matting, roped round and marked with a red brand."""
    straw = (222, 204, 150)
    p.ellipse(x - 2, y - 5, x + 2, y, straw)
    p.ellipse(x - 2, y - 5.6, x + 2, y - 4, shade(straw, 1.12))
    for drop in (1.2, 3.4):
        p.line([(x - 2, y - drop), (x + 2, y - drop)], shade(straw, 0.6))
    p.rect(x - 0.8, y - 3, x + 0.6, y - 2, (180, 40, 36))


def sake_brewery() -> Sprite:
    """A long white-plastered storehouse on a dark tiled skirt, under a heavy black tiled roof
    with a squat chimney, a ball of cedar twigs hung by its door for the new brew, straw-wrapped
    casks stacked outside and sacks of rice by a cart track worn out of it."""
    p = Painter(32, 40, (15, 31))
    foot = _footing(-4, -1, 12, 7)
    _bed(p, foot)
    p.shadow(13, 4)
    _track(p, foot[0], foot[1], (0.52, 0.7), 3.5)
    plaster = (238, 234, 222)
    skirt = (70, 72, 80)
    top = p.hall(-4, -1, 12, 7, 7, plaster, None)
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (4.4, 7), skirt)
    _patch(p, front, right, (0, 1), (4.4, 7), shade(skirt, 0.78))
    for a, b in ((left, front), (front, right)):
        for t in (0.12, 0.3, 0.48, 0.66, 0.84):
            p.point(*_on_wall(a, b, t, 5.6), shade(skirt, 1.6))
        _band(p, a, b, 4.4, shade(plaster, 0.7))
    _windows(p, left, front, (0.2, 0.82), (1.6,), 2, (44, 46, 52))
    _door(p, left, front, (0.52, 0.7), 2.6, 7, (60, 44, 34), shade(plaster, 0.8))
    _windows(p, front, right, (0.5,), (1.6,), 2, (44, 46, 52))
    roof = (52, 54, 62)
    p.gable(top, 5, roof, plaster)
    _roof_rows(p, top, 5, roof, (0.3, 0.6, 0.9))
    x, y = _on_roof(top, 5, 0.3, 0.7)
    p.rect(x - 1, y - 3, x + 1, y, shade(plaster, 0.85))
    _smoke(p, x, y - 4, (200, 200, 204), 2)
    cedar = (96, 112, 52)
    bx, by = _on_wall(left, front, 0.46, 0.6)
    p.line([(bx, by - 0.6), (bx, by + 0.6)], (60, 48, 40))
    p.blob(bx - 1.6, by + 0.4, bx + 1.6, by + 3.4, cedar)
    p.point(bx - 0.6, by + 1.2, shade(cedar, 1.3))
    for cx, cy in ((-12, 5), (-9, 6.5), (-10.5, 2)):
        _straw_cask(p, cx, cy)
    for sx, sy in ((10, 6), (12.5, 5)):
        p.ellipse(sx - 1.5, sy - 3, sx + 1.5, sy, (226, 216, 190))
        p.line([(sx - 0.6, sy - 3), (sx + 0.6, sy - 3)], shade((226, 216, 190), 0.6))
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
    for k, hide in enumerate(((198, 150, 100), (162, 108, 68), (214, 180, 130))):
        _hide_frame(p, 2 + 4 * k, 6 - 2 * k, hide)
    return p.sprite()


def shoe_factory() -> Sprite:
    """A narrow workshop of two storeys in leather brown with a cornice between its floors, windows with sills, a
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
    wall, roof = (170, 112, 70), (92, 52, 40)
    s, floors = 6, 2
    top = p.box(1, 0, s, 4 * floors, wall, None)
    left, front, right, _ = top
    _plinth(p, top, 4 * floors, wall)
    _stains(p, front, right, 4 * floors, (0.15, 0.5, 0.85), shade(wall, 0.58), 1.5)
    for f in range(1, floors):
        _bands(p, left, front, right, 4 * f - 0.5, shade(wall, 1.15), shade(wall, 0.85))
    for f in range(floors - 1):
        for t in (0.3, 0.7):
            _window(p, left, front, (t - 0.07, t + 0.07), 1.2 + 4 * f, tall=1.6)
    for f in range(floors):
        for t in (0.3, 0.7):
            _window(p, front, right, (t - 0.07, t + 0.07), 1.2 + 4 * f, shade(WINDOW, 0.75), 1.6)
    ground = 4 * (floors - 1)
    _patch(p, left, front, (0.1, 0.42), (ground + 0.8, ground + 2.8), GLASS)
    p.line(
        [_on_wall(left, front, 0.15, ground + 1.4), _on_wall(left, front, 0.2, ground + 1.2)],
        (220, 236, 244),
    )
    _patch(p, left, front, (0.08, 0.44), (ground + 0.2, ground + 0.8), (210, 186, 140))
    _door(p, left, front, (0.55, 0.8), ground + 0.5, ground + 4, (60, 40, 30))
    p.point(*_on_wall(left, front, 0.75, ground + 2.2), (226, 196, 120))
    p.gable(top, 4, roof, wall)
    _roof_rows(p, top, 4, roof)
    x, y = _on_roof(top, 4, 0.7, 0.9)
    p.rect(x - 1, y - 2.5, x, y, BRICK)
    p.line([(x - 1.5, y - 3), (x + 0.5, y - 3)], shade(BRICK, 0.55))
    p.line([(-5, -8.5), (-7.5, -8.5)], (60, 56, 54))
    boot = (140, 54, 40)
    p.polygon(
        [(-8.5, -8), (-6, -8), (-6, -3.5), (-9.5, -3.5), (-9.5, -4.6), (-9, -5.2), (-8.5, -5.2)],
        boot,
        shade(boot, 0.5),
    )
    p.line([(-7.6, -7.4), (-7.6, -5.4)], shade(boot, 1.4))
    p.line([(-9.5, -3.5), (-6, -3.5)], (40, 32, 30))
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
    _band(p, left, front, 1, red)
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
    _bands(p, left, front, right, 1, stripe, shade(stripe, 0.8))
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
            _band(p, crate[0], crate[1], 2, shade(white, 0.55))
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
    _smoking_chimney(p, 8, -4, -30)
    wall, roof = (236, 232, 220), (120, 124, 130)
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
    _band(p, left, front, 1, lintel)
    p.line([left, front, right], shade((110, 106, 116), 1.3))
    _bands(p, left, front, right, 7, shade(wall, 0.85), shade(wall, 0.68))
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
    for x, y, lift in ((6.5, 6, 0), (6.5, 6, 1.6), (10, 4.5, 0)):
        _bundle(p, x, y - lift)
    return p.sprite()


def sugar_refinery() -> Sprite:
    """A tall refinery rendered sugar white over a dark plinth stained by the damp, with rows of arched windows
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
    wall = (232, 226, 210)
    lintel = shade(wall, 1.1)
    tall = 34 * CHIMNEY_SHARE
    stack = p.box(8, -3, 2, tall, shade(BRICK, 0.95), (60, 52, 50))
    for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
        p.line([_on_wall(a, b, 0, 2), _on_wall(a, b, 1, 2)], (220, 220, 214))
        _band(p, a, b, tall * 0.4, shade(wall, 0.65))
        _band(p, a, b, tall - 1, shade(wall, 0.6))
    _smoke(p, 9, -9 - tall, puffs=4)
    top = p.box(-3, 0, 8, 18, wall, (92, 86, 94))
    left, front, right, _ = top
    _plinth(p, top, 18, wall)
    _stains(p, left, front, 18, (0.08, 0.33, 0.85), shade(wall, 0.6), 2)
    _stains(p, front, right, 18, (0.15, 0.5, 0.85), shade(wall, 0.5), 2)
    _bands(p, left, front, right, 0.8, lintel, shade(wall, 0.95))
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


COLD_TILE: RGB = (238, 240, 240)
COLD_DOOR: RGB = (190, 204, 214)
OXBLOOD: RGB = (112, 30, 36)
GILT: RGB = (226, 180, 70)
STEER: RGB = (138, 84, 50)
STEER_PATCH: RGB = (96, 58, 36)
MUD: RGB = (112, 94, 66)
STRAW: RGB = (222, 192, 112)
WAGON: RGB = (242, 240, 232)


def _ox_head(p: Painter, x: float, y: float, size: float) -> None:
    """A white ox head seen from the front, its forehead at (x, y) and `size` units from the
    top of its forehead to its muzzle: a long face greyed on its right, short ears, dark eyes, a
    pink muzzle and two curved horns with dark tips."""
    horn, face = (236, 226, 200), (246, 244, 238)

    def at(points: list[tuple[float, float]]) -> list[Point]:
        return [(x + u * size, y + v * size) for u, v in points]

    for side in (-1, 1):
        p.polygon(
            at(
                [
                    (side * 0.28, -0.3),
                    (side * 0.6, -0.42),
                    (side * 0.84, -0.72),
                    (side * 0.8, -0.48),
                    (side * 0.58, -0.26),
                    (side * 0.3, -0.16),
                ]
            ),
            horn,
        )
        p.polygon(
            at([(side * 0.84, -0.72), (side * 0.78, -0.6), (side * 0.82, -0.56)]), (70, 60, 52)
        )
        ex, ey = x + side * 0.48 * size, y - 0.08 * size
        p.blob(
            ex - 0.16 * size,
            ey - 0.08 * size,
            ex + 0.16 * size,
            ey + 0.09 * size,
            shade(face, 0.86),
        )
    outline = [
        (-0.36, -0.36),
        (0.36, -0.36),
        (0.38, -0.1),
        (0.26, 0.3),
        (0.2, 0.56),
        (-0.2, 0.56),
        (-0.26, 0.3),
        (-0.38, -0.1),
    ]
    p.polygon(at(outline), face)
    p.polygon(
        at([(0.06, -0.36), (0.36, -0.36), (0.38, -0.1), (0.26, 0.3), (0.2, 0.56), (0.06, 0.56)]),
        shade(face, 0.88),
    )
    p.polygon(at([(-0.16, -0.36), (0.16, -0.36), (0, -0.2)]), shade(face, 0.8))
    for side in (-1, 1):
        ex, ey = x + side * 0.17 * size, y - 0.04 * size
        p.blob(ex - 0.06 * size, ey - 0.05 * size, ex + 0.06 * size, ey + 0.05 * size, (40, 34, 34))
    p.blob(x - 0.24 * size, y + 0.38 * size, x + 0.24 * size, y + 0.62 * size, (214, 170, 160))
    for side in (-1, 1):
        nx = x + side * 0.09 * size
        p.blob(nx - 0.04 * size, y + 0.47 * size, nx + 0.04 * size, y + 0.53 * size, (110, 70, 66))


def _butcher_sign(p: Painter, x: float, y: float) -> None:
    """A butcher's sign hung from a wrought-iron bracket fixed to a wall at (x, y), reaching out
    towards the lit side: an oxblood board in a gilt frame painted with a white ox head."""
    m = METRE
    end = (x - 2.8 * m, y + 1.4 * m)
    _bar(p, (x, y), end, 0.2 * m, (44, 40, 40))
    _bar(p, (x, y + 1.2 * m), (x - 1.4 * m, y + 0.7 * m), 0.12 * m, (44, 40, 40))
    cx = end[0] + 0.9 * m
    top = end[1] + 0.5 * m
    for dx in (-1.3, 1.3):
        _bar(p, (cx + dx * m, top), (cx + dx * 0.4 * m, end[1]), 0.06 * m, (44, 40, 40))
    p.rect(cx - 1.8 * m, top, cx + 1.8 * m, top + 3 * m, GILT)
    p.rect(cx - 1.6 * m, top + 0.2 * m, cx + 1.6 * m, top + 2.8 * m, OXBLOOD)
    _ox_head(p, cx, top + 1.3 * m, 1.75 * m)


def _spoked_wheel(p: Painter, x: float, y: float, r: float) -> None:
    """A cart wheel `r` units across its middle at (x, y): an iron tyre, six spokes and a hub."""
    p.ellipse(x - r, y - r, x + r, y + r, (54, 44, 38))
    p.ellipse(x - 0.78 * r, y - 0.78 * r, x + 0.78 * r, y + 0.78 * r, (64, 52, 44))
    p.blob(x - 0.7 * r, y - 0.7 * r, x + 0.7 * r, y + 0.7 * r, (122, 86, 54))
    for k in range(3):
        a = k * math.pi / 3
        dx, dy = 0.72 * r * math.cos(a), 0.72 * r * math.sin(a)
        _bar(p, (x - dx, y - dy), (x + dx, y + dy), 0.16 * r, (70, 52, 38))
    p.blob(x - 0.2 * r, y - 0.2 * r, x + 0.2 * r, y + 0.2 * r, (44, 38, 34))


def _meat_wagon(p: Painter, x: float, y: float) -> None:
    """A closed meat wagon 3.6 m long standing on (x, y) along the lit walls: a white box body
    with a red band lettered in white and a door, a dark roof, two big spoked wheels and its
    shafts down on the ground ahead of it."""
    m = METRE
    p.blob(x - 2.6 * m, y - 1 * m, x + 3 * m, y + 1.4 * m, (60, 56, 52), 110)
    shafts = [_iso(x, y, 1.8 * m, v * m) for v in (-0.4, 0.4)]
    for sx, sy in shafts:
        _bar(p, (sx, sy - 1 * m), _iso(sx, sy, 2.6 * m, 0), 0.14 * m, shade(TIMBER, 0.7))
    _bar(
        p,
        _iso(x, y - 1 * m, -1.8 * m, 0.9 * m),
        _iso(x, y - 1 * m, 1.8 * m, 0.9 * m),
        0.2 * m,
        (50, 44, 40),
    )
    body = p.hall(x, y - 1 * m, 3.6 * m, 1.8 * m, 1.9 * m, WAGON, (70, 66, 70))
    left, front, right, _ = body
    _patch(p, left, front, (0, 1), (0.55 * m, 1.15 * m), (196, 44, 48))
    for t0, t1 in ((0.1, 0.24), (0.28, 0.36), (0.4, 0.56)):
        _bar(
            p,
            _on_wall(left, front, t0, 0.85 * m),
            _on_wall(left, front, t1, 0.85 * m),
            0.22 * m,
            WAGON,
        )
    _patch(p, left, front, (0.66, 0.9), (0.3 * m, 1.8 * m), shade(WAGON, 0.86))
    _bar(
        p,
        _on_wall(left, front, 0.78, 0.3 * m),
        _on_wall(left, front, 0.78, 1.8 * m),
        0.05 * m,
        shade(WAGON, 0.6),
    )
    _patch(p, front, right, (0.15, 0.85), (0.3 * m, 0.9 * m), (60, 56, 60))
    for t in (0.16, 0.8):
        _spoked_wheel(p, *_on_wall(left, front, t, 1.9 * m + 0.45 * m), 0.6 * m)


def _cold_door(
    p: Painter, a: Point, b: Point, span: tuple[float, float], top: float, foot: float
) -> None:
    """A thick insulated door of a cold store on the wall under the edge from `a` to `b`, from
    `span[0]` to `span[1]` along it and `top` to `foot` down it: a pale padded leaf in a deep
    frame, held by three heavy strap hinges and shut with a lever latch."""
    m = METRE
    t0, t1 = span
    _patch(p, a, b, (t0 - 0.03, t1 + 0.03), (top - 0.2 * m, foot), shade(COLD_DOOR, 0.7))
    _patch(p, a, b, span, (top, foot), COLD_DOOR)
    _patch(
        p, a, b, (t0 + 0.02, t1 - 0.02), (top + 0.15 * m, foot - 0.15 * m), shade(COLD_DOOR, 1.06)
    )
    for k in range(3):
        drop = top + (foot - top) * (0.15 + 0.35 * k)
        _bar(
            p,
            _on_wall(a, b, t0, drop),
            _on_wall(a, b, t0 + (t1 - t0) * 0.45, drop),
            0.14 * m,
            (60, 62, 70),
        )
    latch = t1 - (t1 - t0) * 0.15
    _bar(
        p,
        _on_wall(a, b, latch, top + (foot - top) * 0.35),
        _on_wall(a, b, latch, top + (foot - top) * 0.6),
        0.12 * m,
        (60, 62, 70),
    )


def meat_factory() -> Sprite:
    """A red-brick slaughterhouse 15 by 8 m with walls 8 m high: two tidy rows of round-headed
    windows under cream arches between pilasters, a slate roof with a louvred vent along its
    ridge, a timber door into it from a fenced cattle pen of trampled mud and straw where brown
    steers wait by a water trough, a butcher's sign of a white ox head on an oxblood board hung
    from a bracket, a loading platform before its loading door with a white meat wagon drawn up
    at it, a white-tiled cold store with insulated doors and a refrigerating plant on its roof,
    and a modest brick chimney smoking behind."""
    m = METRE
    p = Painter(38, 30, (19, 22))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10) * m, (v + 1) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    length, depth, high = 15 * m, 8 * m, 8 * m
    p.polygon([at(8.5, -6), at(24, -6), at(24, 0), at(8.5, 0)], (150, 140, 126))
    pen = (at(-2, -10.5), at(-2, -1), at(8.5, -1), at(8.5, -10.5))
    p.polygon(list(pen), MUD)
    for u, v, r in ((1, -3, 1.2), (5.5, -8.5, 1), (-0.5, -7, 0.8), (6.5, -4, 0.7)):
        x, y = at(u, v)
        p.blob(x - r * m, y - r * m / 2, x + r * m, y + r * m / 2, STRAW, 170)
    foot = _footing(*at(7.5, 4), length, depth)
    _bed(p, foot)
    _bed(p, _footing(*at(18.25, 2.75), 5.5 * m, 5.5 * m), margin=1)
    p.shadow(14, 4, 3)
    sx, sy = at(13, 10)
    stack = p.hall(sx, sy, 1.4 * m, 1.4 * m, 22 * m * CHIMNEY_SHARE, BRICK, (52, 46, 46))
    _bands(p, stack[0], stack[1], stack[2], 1 * m, DRESSING, shade(DRESSING, 0.78))
    _smoke(p, sx + 1, stack[1][1] - 3, (206, 206, 212), 2)
    top = p.hall(*at(7.5, 4), length, depth, high, BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, BRICK, 0.6 * m)
    _stains(p, left, front, high, (0.04, 0.37, 0.63, 0.97), shade(BRICK, 0.66), 1.2 * m)
    for a, b in ((left, front), (front, right)):
        for drop, wide in ((0, 0.45 * m), (3.9 * m, 0.25 * m)):
            _bar(
                p,
                _on_wall(a, b, 0, drop + wide / 2),
                _on_wall(a, b, 1, drop + wide / 2),
                wide,
                DRESSING,
            )
    for k in range(6):
        t = k / 5
        _bar(
            p,
            _on_wall(left, front, t, 0.45 * m),
            _on_wall(left, front, t, high - 0.6 * m),
            0.45 * m,
            shade(BRICK, 0.86),
        )
    for k in range(5):
        t = (k + 0.5) / 5
        _mill_window(p, left, front, length, t, 1.2 * m, (1.1 * m, 3.3 * m))
        if k in (0, 2, 4):
            _mill_window(p, left, front, length, t, 1.3 * m, (4.7 * m, 6.9 * m))
    for t, foot_at in ((0.3, high), (0.7, high - 1.1 * m)):
        p.polygon(_round_head(left, front, length, t, 2.6 * m, 4.4 * m, foot_at), DRESSING)
        p.polygon(_round_head(left, front, length, t, 2.2 * m, 4.6 * m, foot_at), DOORWAY)
        _patch(p, left, front, (t - 0.073, t - 0.005), (5.4 * m, foot_at), DOOR)
        _bar(
            p,
            _on_wall(left, front, t - 0.04, 5.4 * m),
            _on_wall(left, front, t - 0.04, foot_at),
            0.06 * m,
            shade(DOOR, 0.7),
        )
    for t in (0.3, 0.7):
        _mill_window(
            p, front, right, depth, t, 1.2 * m, (1.1 * m, 3.3 * m), shade(MILL_GLASS, 0.85)
        )
    p.gable(top, 3 * m, MILL_SLATE, BRICK)
    _roof_rows(p, top, 3 * m, MILL_SLATE)
    ridge_back, ridge_front = _ridge(top, 3 * m)
    vx, vy = _on_wall(ridge_back, ridge_front, 0.5, 0)
    vent = p.hall(vx, vy + 0.4 * m, 6 * m, 1.4 * m, 1.3 * m, (178, 172, 164), None)
    for k in range(1, 4):
        drop = 1.3 * m * k / 4
        _bar(
            p,
            _on_wall(vent[0], vent[1], 0, drop),
            _on_wall(vent[0], vent[1], 1, drop),
            0.1 * m,
            (100, 96, 94),
        )
    p.gable(vent, 0.7 * m, MILL_SLATE, (178, 172, 164))
    store = p.hall(*at(18.25, 2.75), 5.5 * m, 5.5 * m, 6 * m, COLD_TILE, shade(COLD_TILE, 0.86))
    s_left, s_front, s_right, _ = store
    for a, b, tone in ((s_left, s_front, 0.9), (s_front, s_right, 0.72)):
        for drop in np.arange(1.1 * m, 5.6 * m, 0.45 * m):
            _bar(
                p,
                _on_wall(a, b, 0, drop),
                _on_wall(a, b, 1, drop),
                0.03 * m,
                shade(COLD_TILE, tone),
            )
        for t in np.linspace(0.08, 0.92, 9):
            _bar(
                p,
                _on_wall(a, b, t, 1.1 * m),
                _on_wall(a, b, t, 5.6 * m),
                0.03 * m,
                shade(COLD_TILE, tone),
            )
        _patch(p, a, b, (0, 1), (0, 0.7 * m), shade(BRICK, 1.0 if tone > 0.8 else 0.8))
        _patch(p, a, b, (0, 1), (5.6 * m, 6 * m), shade(BRICK, 0.9 if tone > 0.8 else 0.72))
    _cold_door(p, s_left, s_front, (0.36, 0.64), 2.2 * m, 6 * m - 1.1 * m)
    _cold_door(p, s_front, s_right, (0.3, 0.62), 2.2 * m, 6 * m)
    ux, uy = up(19, 4, 6)
    plant = p.hall(ux, uy, 2.4 * m, 1.6 * m, 1.3 * m, METAL, shade(METAL, 1.1))
    fx, fy = _on_wall(plant[3], plant[1], 0.5, 0)
    p.ellipse(fx - 0.7 * m, fy - 0.35 * m, fx + 0.7 * m, fy + 0.35 * m, (60, 62, 70))
    _bar(p, up(16.4, 1.5, 6), up(16.4, 1.5, 8.2), 0.25 * m, METAL)
    dock = (shade(BRICK, 0.9), shade(TIMBER, 1.1))
    deck = p.hall(*at(15.75, -1.1), 14.5 * m, 2.2 * m, 1.1 * m, dock[0], dock[1])
    for t in np.linspace(0.05, 0.95, 12):
        _bar(
            p,
            _on_wall(deck[0], deck[3], t, 0),
            _on_wall(deck[1], deck[2], t, 0),
            0.05 * m,
            shade(TIMBER, 0.8),
        )
    p.polygon(
        [up(23, -2.2, 1.1), up(23, 0, 1.1), up(25.5, 0, 0), up(25.5, -2.2, 0)],
        TIMBER,
        shade(TIMBER, 0.6),
    )
    _puddle(p, *at(22, -4.5), 1)
    _meat_wagon(p, *at(12.6, -3.6))
    _fence(p, pen[0], pen[1], 1.3 * METRE)
    _fence(p, pen[1], at(3, -1), 1.3 * METRE)
    _fence(p, at(6, -1), pen[2], 1.3 * METRE)
    tx, ty = at(-0.4, -2.8)
    trough = p.hall(tx, ty, 0.9 * m, 3 * m, 0.7 * m, shade(TIMBER, 0.85), PUDDLE)
    _bar(
        p,
        _on_wall(trough[3], trough[2], 0.2, 0),
        _on_wall(trough[3], trough[2], 0.7, 0),
        0.08 * m,
        shade(PUDDLE, 1.4),
    )
    for u, v, facing in ((4.5, -3.5, 1), (1.5, -6, -1), (6, -7.8, -1)):
        _cow(p, *at(u, v), STEER, STEER_PATCH, BEAST, facing)
    _fence(p, pen[0], pen[3], 1.3 * METRE)
    _fence(p, pen[3], pen[2], 1.3 * METRE)
    _butcher_sign(p, *_on_wall(left, front, 0.04, 1.6 * m))
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
            _band(p, a, b, 1, shade(olive, 1.3))
            _band(p, a, b, height - 1, shade(olive, 0.7))
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
        _smoking_chimney(p, x, y - 4, y - 30, shade(BRICK, 0.8), smoke=(214, 214, 218))
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


OIL_YELLOW: RGB = (232, 184, 40)
TANK_WHITE: RGB = (226, 226, 222)


def _storage_tank(p: Painter, x: float, y: float, r: float, h: float) -> None:
    """A big round oil storage tank on the ground at (x, y), `r` across and `h` tall: a low earth
    bund round it, white steel walls with a yellow band, a shallow domed roof, and a stair
    climbing its lit side."""
    p.blob(x - r - 1.6, y - r / 2 - 0.8, x + r + 1.6, y + r / 2 + 0.8, (150, 136, 108))
    p.blob(x - r - 0.8, y - r / 2 - 0.4, x + r + 0.8, y + r / 2 + 0.4, (176, 168, 150))
    profile = [(0, r), (h, r), (h + 0.5, r * 0.9), (h + 0.9, r * 0.62), (h + 1.1, r * 0.2)]
    _turned(p, x, y, profile, TANK_WHITE)
    for drop in (h * 0.6, h * 0.6 + 0.8):
        _ring(p, x, y - drop, r, OIL_YELLOW)
    _ring(p, x, y - h, r, shade(TANK_WHITE, 0.62))
    stair = shade(TANK_WHITE, 0.45)
    p.line([(x - r + 0.3, y - 0.2), (x - r * 0.55, y + r * 0.18 - h)], stair)
    p.line([(x - r + 0.9, y + 0.2), (x - r * 0.55 + 0.6, y + r * 0.22 - h)], stair)


def refinery() -> Sprite:
    """An oil refinery filling its cell, after the great refineries of the oil age: two tall brick
    chimneys smoking at the back, tall silver distillation columns ringed with yellow walkways, a
    pipe rack carrying yellow pipes, a flare burning on a red-and-white stack, and at the front
    big round white storage tanks banded in oil yellow, each in its earth bund with a stair up
    its side."""
    p = Painter(50, 64, (24, 52))
    plate = (168, 164, 156)
    _plate(p, plate)
    for x, y in ((-3, -6), (9, -3), (-12, 1)):
        _oil(p, x, y, 1.2)
    for x, y in ((-13, -3.5), (-9.5, -5.5)):
        _smoking_chimney(p, x, y, y - 34, shade(BRICK, 0.95), smoke=(150, 148, 150))
    fx, fy = 12, -4
    p.line([(fx, fy), (fx, fy - 36)], (90, 90, 96))
    p.line([(fx + 1, fy), (fx + 1, fy - 36)], (124, 124, 130))
    for y in (-34, -30, -26):
        p.line([(fx, fy + y), (fx + 1, fy + y)], (206, 60, 50))
    p.blob(fx - 2, fy - 43, fx + 3, fy - 36, (246, 150, 40))
    p.blob(fx - 0.5, fy - 40, fx + 1.5, fy - 37, (255, 220, 120))
    metal = (176, 178, 184)
    for x, y, r, h in ((1, -5, 2, 22), (5.5, -3.5, 2.5, 30), (-3, -2.5, 2, 18)):
        _turned(p, x, y, [(0, r), (h, r), (h + 1.5, r * 0.6)], metal)
        for ring in range(6, h, 6):
            p.line([(x - r - 1, y - ring), (x + r + 1, y - ring)], OIL_YELLOW)
            p.line(
                [(x - r - 1, y - ring + 0.5), (x + r + 1, y - ring + 0.5)], shade(OIL_YELLOW, 0.6)
            )
        p.line([(x + r - 0.5, y - 1), (x + r - 0.5, y - h)], shade(metal, 0.62))
    rack = (96, 96, 104)
    for x in (-15, -8, -1, 6, 13):
        p.line([(x, 0), (x, -6)], rack)
    for h, colour in (
        (6, rack),
        (5, OIL_YELLOW),
        (4, shade(OIL_YELLOW, 0.75)),
        (3, (120, 120, 128)),
    ):
        p.line([(-15, -h), (13, -h)], colour)
    for x in (-11, -4, 3, 10):
        p.point(x, -5.5, (206, 60, 50))
    for x, y, r, h in ((12.5, 1.5, 3.5, 3.5), (-8.5, 3.5, 5, 4.5), (3, 5.5, 5, 4.5)):
        _storage_tank(p, x, y, r, h)
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
    _smoking_chimney(p, cx, cy - 4, cy - 38, shade(BRICK, 0.9), smoke=(124, 122, 126))
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
    _bands(p, left, front, right, 1, shade(BRICK, 1.3), shade(BRICK, 1.05))
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


def diesel_power_plant() -> Sprite:
    """A diesel power station filling its cell: a long low engine hall in pale render with a row
    of short exhaust stacks along its roof, white fuel tanks with stairs up their sides on a pad
    behind, a transformer and a pylon carrying power lines away."""
    p = Painter(50, 58, (24, 46))
    plate = (164, 160, 152)
    _plate(p, plate)
    white = (224, 222, 216)
    for u, v in ((-8, 7), (-2, 8.5), (4, 9)):
        x, y = _iso(0, 0, u, v)
        _turned(p, x, y, [(0, 3.2), (5, 3.2)], white, cap=shade(white, 1.08))
        p.line([(x - 3, y - 0.5), (x - 0.5, y - 5)], shade(white, 0.62))
    _pylon(p, *_iso(0, 0, -8, -8), -4)
    wall = (206, 196, 176)
    top = p.hall(*_iso(0, 0, -1, -1), 16, 6, 8, wall, None)
    left, front, right, _ = top
    _plinth(p, top, 8, shade(wall, 0.7))
    _windows(
        p, left, front, (0.15, 0.32, 0.5, 0.68, 0.85), (2.5,), 4, (88, 112, 140), shade(wall, 1.1)
    )
    _windows(p, front, right, (0.5,), (2.5,), 4, (70, 90, 112), shade(wall, 1.0))
    p.gable(top, 2.5, (96, 92, 96), wall)
    stack = (86, 84, 90)
    for t in (0.2, 0.4, 0.6, 0.8):
        x, y = _on_wall(left, front, t, -2)
        p.line([(x + 2, y), (x + 2, y - 7)], stack)
        p.line([(x + 3, y), (x + 3, y - 7)], shade(stack, 1.3))
    _smoke(p, *_on_wall(left, front, 0.4, -10), (120, 118, 122), 2)
    return p.sprite()


def gas_power_plant() -> Sprite:
    """A gas power station filling its cell: a modern turbine hall clad in grey-green panels, two
    slim steel stacks, a round gas holder banded in green, a transformer yard and a pylon carrying
    power lines away."""
    p = Painter(50, 64, (24, 52))
    plate = (166, 168, 162)
    _plate(p, plate)
    holder = (110, 140, 120)
    x, y = _iso(0, 0, 6, 8)
    _turned(p, x, y, [(0, 5.5), (12, 5.5), (13.5, 4.2)], holder)
    for ring in (3, 6, 9):
        p.line([(x - 6, y - ring), (x + 6, y - ring)], shade(holder, 0.7))
    steel = (176, 180, 186)
    for u in (-9, -6):
        sx, sy = _iso(0, 0, u, 8)
        p.line([(sx, sy), (sx, sy - 34)], shade(steel, 0.75))
        p.line([(sx + 1, sy), (sx + 1, sy - 34)], steel)
        p.line([(sx, sy - 31), (sx + 1, sy - 31)], (206, 60, 50))
    _pylon(p, *_iso(0, 0, -8, -8), -4)
    tx, ty = _iso(0, 0, 5, -7)
    grey = (120, 126, 120)
    p.box(tx, ty, 2, 2.5, grey, shade(grey, 1.15))
    clad = (150, 170, 160)
    top = p.hall(*_iso(0, 0, -2, -0.5), 13, 7, 11, clad, None)
    left, front, right, _ = top
    _plinth(p, top, 11, shade(clad, 0.7))
    _bands(p, left, front, right, 7, shade(clad, 1.2), shade(clad, 1.05))
    _windows(p, left, front, (0.2, 0.4, 0.6, 0.8), (2.5,), 3, (88, 112, 140), shade(clad, 1.2))
    p.gable(top, 2, (100, 112, 108), clad)
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
    _locomotive(p, -12, 6)
    for x in (3, 8):
        p.rect(x, 3.5, x + 4, 6.5, (120, 64, 44))
        p.point(x + 1, 7, (40, 40, 44))
        p.point(x + 3, 7, (40, 40, 44))
    return p.sprite()


def _locomotive(p: Painter, x: float, y: float) -> None:
    """A green steam locomotive with red wheels and a tall black funnel, standing on (x, y)."""
    green, black, red = (48, 120, 72), (36, 38, 42), (190, 48, 40)
    p.rect(x - 4, y - 5, x + 3, y - 2, green)
    p.line([(x - 4, y - 4.4), (x + 3, y - 4.4)], shade(green, 1.4))
    p.rect(x + 2, y - 7.5, x + 5.5, y - 2, shade(green, 0.85))
    p.rect(x + 2.6, y - 6.8, x + 4.8, y - 5.4, WINDOW)
    p.rect(x - 3.6, y - 8, x - 2.2, y - 5, black)
    p.rect(x - 4.4, y - 8.8, x - 1.4, y - 8, black)
    p.rect(x - 5, y - 2, x + 5.5, y - 1.2, black)
    p.line([(x - 5.4, y - 2.6), (x - 5.4, y - 1)], red)
    for wx in (x - 2.5, x, x + 3.2):
        p.ellipse(wx - 1.1, y - 1.8, wx + 1.1, y + 0.4, red)


COLA_RED: RGB = (176, 28, 34)
COLA_BROWN: RGB = (74, 42, 30)
COLA_WALL: RGB = (232, 226, 212)
CONCRETE: RGB = (176, 172, 164)


def _bottle_crates(p: Painter, x: float, y: float, high: int) -> None:
    """A pallet a metre square stacked `high` red crates high with dark bottles, their caps
    showing in the top crate."""
    m = METRE
    p.hall(x, y, m, m, 0.15 * m, shade(TIMBER, 0.85), None)
    crate = 0.32 * m
    for k in range(high):
        top = p.hall(x, y - 0.15 * m - k * crate, m, m, crate, COLA_RED, COLA_BROWN)
    left, front, _, back = top
    for u in (0.25, 0.5, 0.75):
        for v in (0.3, 0.7):
            bx = left[0] + (front[0] - left[0]) * u + (back[0] - left[0]) * v
            by = left[1] + (front[1] - left[1]) * u + (back[1] - left[1]) * v
            p.blob(bx - 0.1 * m, by - 0.06 * m, bx + 0.1 * m, by + 0.06 * m, (214, 190, 120))


def _sign_bottle(p: Painter, a: Point, b: Point, t: float, drop: float, tall: float) -> None:
    """A white bottle drawn on a sign on the wall under the edge from `a` to `b`, `t` along it
    with its foot `drop` down and `tall` high."""
    fx, fy = _on_wall(a, b, t, drop)
    w = tall * 0.18
    p.polygon(
        [
            (fx - w, fy),
            (fx - w, fy - tall * 0.55),
            (fx - w * 0.35, fy - tall * 0.8),
            (fx - w * 0.35, fy - tall),
            (fx + w * 0.35, fy - tall),
            (fx + w * 0.35, fy - tall * 0.8),
            (fx + w, fy - tall * 0.55),
            (fx + w, fy),
        ],
        (246, 242, 236),
    )


def _van(p: Painter, x: float, y: float) -> None:
    """A red delivery van backed up to a dock, its box body on (x, y) and its white cab further
    out, with dark wheels."""
    m = METRE
    body, cab_length = 4.4 * m, 1.8 * m
    for u in (-0.3, 0.35):
        wx, wy = _iso(x, y, u * body, 1.1 * m)
        p.ellipse(wx - 0.4 * m, wy - 0.45 * m, wx + 0.4 * m, wy + 0.45 * m, (40, 38, 38))
    left, front, right, _ = p.hall(x, y - 0.5 * m, body, 2.2 * m, 2.6 * m, COLA_RED, COLA_RED)
    _band(p, left, front, 1.5 * m, (246, 242, 236))
    _band(p, front, right, 1.5 * m, (230, 226, 220))
    cx, cy = _iso(x, y, (body + cab_length) / 2, 0)
    cab = p.hall(cx, cy - 0.5 * m, cab_length, 2.2 * m, 1.9 * m, (236, 234, 228), (236, 234, 228))
    _patch(p, cab[1], cab[2], (0.1, 0.9), (0.2 * m, 0.9 * m), (130, 170, 200))
    _patch(p, cab[0], cab[1], (0.45, 0.95), (0.2 * m, 0.8 * m), (130, 170, 200))


def _syrup_tank(p: Painter, x: float, y: float) -> None:
    """A dark brown syrup tank 2.6 m across and 8 m tall standing on (x, y), with a red band round
    it, a lighter domed top and a ladder up its side."""
    m = METRE
    r, tall = 1.3 * m, 8 * m
    p.ellipse(x - r, y - r / 2, x + r, y + r / 2, shade(COLA_BROWN, 0.7))
    p.rect(x - r, y - tall, x + r, y, COLA_BROWN)
    p.rect(x - r, y - tall * 0.62, x + r, y - tall * 0.52, COLA_RED)
    p.ellipse(x - r, y - tall - r / 2, x + r, y - tall + r / 2, shade(COLA_BROWN, 1.35))
    _bar(p, (x + 0.4 * r, y), (x + 0.4 * r, y - tall), 0.12 * m, (150, 146, 140))
    _bar(p, (x - 0.6 * r, y - 0.2 * m), (x - 0.6 * r, y - tall), 0.18 * m, shade(COLA_BROWN, 1.2))


def cola_bottler() -> Sprite:
    """A cola bottling plant: a whitewashed bottling hall 22 by 14 m on a brown plinth under a
    dark brown roof with vents, a broad red band round its top and a red sign with a white
    bottle on it, windows showing bottles going by on the line inside, a syrup tank beside it,
    a loading dock with a red delivery van backed up to it, and red crates of dark bottles
    stacked on pallets by its door, on a concrete yard, all at their true size beside the
    houses."""
    p = Painter(30, 28, (15, 19))
    m = METRE
    length, depth, high = 22 * m, 14 * m, 7.5 * m
    x, y = -1.5, -1
    foot = _footing(x, y, length, depth)
    _bed(p, foot, CONCRETE)
    p.shadow(8, 2.5, reach=2)
    _track(p, foot[0], foot[1], (0.6, 0.78), 3, shade(CONCRETE, 0.9), ruts=False)
    top = p.hall(x, y, length, depth, high, COLA_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, high, COLA_BROWN, 0.9 * m, 1.0, 0.8)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (0, 1.3 * m), shade(COLA_RED, tone))
    glass = shade(GLAZING, 0.82)
    for t0 in (0.06, 0.2, 0.34):
        _patch(p, left, front, (t0, t0 + 0.11), (2.4 * m, 4.4 * m), glass)
        for k in range(3):
            bx, by = _on_wall(left, front, t0 + 0.025 + 0.03 * k, 4.4 * m)
            p.rect(bx - 0.1 * m, by - 0.8 * m, bx + 0.1 * m, by - 0.1 * m, COLA_BROWN)
    _patch(p, left, front, (0.52, 0.96), (1.9 * m, 4 * m), COLA_RED)
    for drop in (2.7 * m, 3.3 * m):
        _band_on(p, left, front, (0.64, 0.92), drop)
    _sign_bottle(p, left, front, 0.57, 3.7 * m, 1.6 * m)
    _door(p, left, front, (0.66, 0.8), 4.8 * m, high, DOORWAY)
    _patch(p, front, right, (0.1, 0.36), (2.4 * m, 4.2 * m), shade(glass, 0.8))
    _door(p, front, right, (0.5, 0.86), 3.4 * m, high - 1.2 * m, (96, 92, 88))
    rise = 2 * m
    p.gable(top, rise, COLA_BROWN, COLA_WALL)
    _roof_rows(p, top, rise, COLA_BROWN, (0.5,))
    for along in (0.3, 0.6):
        vx, vy = _on_roof(top, rise, along, 0.55)
        p.box(vx, vy, 0.8 * m, 1 * m, (150, 146, 140), shade((150, 146, 140), 1.15))
    dock = [
        _on_wall(front, right, 0.46, high - 1.2 * m),
        _on_wall(front, right, 0.92, high - 1.2 * m),
    ]
    ox, oy = 2 * m, m
    edge = [(dock[1][0] + ox, dock[1][1] + oy), (dock[0][0] + ox, dock[0][1] + oy)]
    p.polygon([dock[0], dock[1], *edge], CONCRETE, shade(CONCRETE, 0.7))
    p.polygon(
        [*edge, (edge[0][0], edge[0][1] + 1.2 * m), (edge[1][0], edge[1][1] + 1.2 * m)][::1],
        shade(CONCRETE, 0.75),
    )
    for t in (0.52, 0.64):
        bx, by = _on_wall(front, right, t, high - 1.2 * m)
        _bottle_crates(p, bx + 0.9 * m, by + 0.7 * m, 3)
    mx, my = _on_wall(front, right, 0.72, high)
    _van(p, *_iso(mx, my, 2 * m + 2.4 * m, 0))
    lx, ly = foot[0]
    _syrup_tank(p, lx - 1.2, ly + 1.4)
    for k, (u, crates) in enumerate(((0.36, 4), (0.46, 3), (0.56, 4))):
        cx, cy = _on_wall(foot[0], foot[1], u, 0)
        _bottle_crates(p, cx - 1.5 * m - 0.3 * k * m, cy + 0.8 * m + 0.15 * k * m, crates)
    return p.sprite()


def _band_on(p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float) -> None:
    """A white stripe of lettering on a sign from `span[0]` to `span[1]` along the wall under the
    edge from `a` to `b`, `drop` down it."""
    _bar(
        p,
        _on_wall(a, b, span[0], drop),
        _on_wall(a, b, span[1], drop),
        0.3 * METRE,
        (246, 242, 236),
    )
