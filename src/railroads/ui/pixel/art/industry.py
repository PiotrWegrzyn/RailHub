"""Factories and works of every kind."""

import itertools
import math
from collections.abc import Callable

import numpy as np

from railroads.ui.pixel.art.commerce import _planks
from railroads.ui.pixel.art.painter import (
    BRICK,
    CAR_COLOURS,
    CHIMNEY_SHARE,
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
    _gable_across,
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


def _tall_chimney(p: Painter, x: float, y: float, tall: float, girth: float = 1) -> None:
    """A round brick mill chimney drawn for `tall` units and lowered (CHIMNEY_SHARE), `girth`
    times as stout as a small works' one: a square plinth with a cream band, a tapering shaft
    ringed with iron bands, a cream band and a corbelled black crown at its top, smoking."""
    m = METRE
    tall *= CHIMNEY_SHARE
    base = 3.2 * m * girth
    plinth = p.box(x, y, base / 2 * 1.4, base, MILL_BRICK, shade(MILL_BRICK, 1.1))
    _bands(p, plinth[0], plinth[1], plinth[2], 0.4 * m, DRESSING, shade(DRESSING, 0.78))
    foot = y - base
    shaft = tall - base
    r0, r1 = 1.5 * m * girth, 0.95 * m * girth
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


WORKS_IRON: RGB = (70, 78, 80)
WORKS_STONE: RGB = (178, 170, 156)
WORKS_BALLAST: RGB = (128, 122, 114)
WORKS_SLEEPER: RGB = (86, 66, 50)
WORKS_RAIL: RGB = (150, 152, 158)
WORKS_VAN: RGB = (132, 58, 44)


def _works_hall(
    p: Painter,
    x: float,
    y: float,
    size: tuple[float, float, float],
    teeth: int,
    door: int | None = None,
) -> Top:
    """A red-brick works hall `size` (length, depth and wall height) standing on (x, y): two rows
    of tall iron-framed windows with round heads under cream arches between pilasters, cream
    bands at its eaves and between its floors, a big arched loading door with a hoist in bay
    `door` of its lit side, and a north-light roof of `teeth` glazed teeth."""
    m = METRE
    length, depth, high = size
    top = p.hall(x, y, length, depth, high, MILL_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, MILL_BRICK, 0.7 * m)
    _stains(p, left, front, high, (0.05, 0.31, 0.55, 0.93), shade(MILL_BRICK, 0.66), 1.6 * m)
    for a, b in ((left, front), (front, right)):
        for drop, wide in ((0, 0.5 * m), (high * 0.48, 0.25 * m)):
            _bar(
                p,
                _on_wall(a, b, 0, drop + wide / 2),
                _on_wall(a, b, 1, drop + wide / 2),
                wide,
                DRESSING,
            )
    upper, lower = (high * 0.137, high * 0.41), (high * 0.57, high * 0.884)
    bays = max(2, round(length / (3.5 * m)))
    for k in range(bays + 1):
        t = k / bays
        _bar(
            p,
            _on_wall(left, front, t, 0.5 * m),
            _on_wall(left, front, t, high),
            0.5 * m,
            shade(MILL_BRICK, 0.86),
        )
    for k in range(bays):
        t = (k + 0.5) / bays
        if k == door:
            p.polygon(_round_head(left, front, length, t, 3.6 * m, high * 0.52, high), DRESSING)
            p.polygon(_round_head(left, front, length, t, 3 * m, high * 0.55, high), DOORWAY)
            _patch(
                p,
                left,
                front,
                (t - 1.5 * m / length, t - 0.05 * m / length),
                (high * 0.7, high),
                DOOR,
            )
            _patch(p, left, front, (t - 0.8 * m / length, t + 0.8 * m / length), upper, DOOR)
            beam = _on_wall(left, front, t, 0.9 * m)
            _bar(p, beam, (beam[0] - 1.6 * m, beam[1] + 0.8 * m), 0.2 * m, (54, 50, 50))
            _bar(
                p,
                (beam[0] - 1.5 * m, beam[1] + 0.8 * m),
                (beam[0] - 1.5 * m, beam[1] + 3.2 * m),
                0.05 * m,
                (40, 38, 38),
            )
            continue
        _mill_window(p, left, front, length, t, 1.5 * m, upper)
        _mill_window(p, left, front, length, t, 1.7 * m, lower)
    side_bays = max(2, round(depth / (4 * m)))
    for k in range(side_bays):
        t = (k + 0.5) / side_bays
        _mill_window(p, front, right, depth, t, 1.5 * m, upper, shade(MILL_GLASS, 0.85))
        _mill_window(p, front, right, depth, t, 1.7 * m, lower, shade(MILL_GLASS, 0.85))
    _north_lights(p, top, teeth, 2.8 * m, MILL_SLATE, MILL_BRICK)
    return top


def _works_clock(p: Painter, x: float, y: float, r: float) -> None:
    """A white clock face `r` across its middle at (x, y) in a cream rim, its hands at ten past
    twelve."""
    m = METRE
    p.ellipse(x - r * 1.25, y - r * 1.25, x + r * 1.25, y + r * 1.25, DRESSING)
    p.ellipse(x - r, y - r, x + r, y + r, (246, 242, 230))
    _bar(p, (x, y), (x, y - r * 0.8), 0.14 * m, (30, 30, 34))
    _bar(p, (x, y), (x + r * 0.55, y + r * 0.15), 0.14 * m, (30, 30, 34))


def _works_pyramid(p: Painter, top: Top, rise: float, vane: bool = True) -> Point:
    """A slate pyramid roof `rise` high on a square top, with a gilt weather vane on its point if
    `vane`. Returns its point."""
    m = METRE
    left, front, right, _ = top
    apex = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2 - rise)
    p.polygon([left, front, apex], MILL_SLATE, shade(MILL_SLATE, 0.6))
    p.polygon([front, right, apex], shade(MILL_SLATE, 0.72), shade(MILL_SLATE, 0.5))
    if vane:
        _bar(p, apex, (apex[0], apex[1] - 2 * m), 0.12 * m, (50, 50, 56))
        p.polygon(
            [
                (apex[0], apex[1] - 2 * m),
                (apex[0] + 1.2 * m, apex[1] - 1.7 * m),
                (apex[0], apex[1] - 1.3 * m),
            ],
            (214, 170, 70),
        )
    return apex


def _works_tower(p: Painter, x: float, y: float, side: float, tall: float) -> None:
    """A red-brick clock and water tower `side` square and `tall` high standing on (x, y): cream
    quoins and bands, tiers of round-headed windows, a clock face on each side under an iron
    water tank riveted in bands, and a slate pyramid with a gilt vane on top."""
    m = METRE
    tower = p.hall(x, y, side, side, tall, MILL_BRICK, None)
    left, front, right, _ = tower
    _plinth(p, tower, tall, MILL_BRICK)
    for a, b in ((left, front), (front, right)):
        for drop in (0.3 * m, 5.4 * m):
            _bar(p, _on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop), 0.3 * m, DRESSING)
        for t in (0.03, 0.97):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, tall - 0.7 * m), 0.35 * m, DRESSING)
        for head in np.arange(7 * m, tall - 3 * m, 4 * m):
            _mill_window(p, a, b, side, 0.5, 1.1 * m, (head, head + 2 * m))
    for a, b in ((left, front), (front, right)):
        _works_clock(p, *_on_wall(a, b, 0.5, 2.8 * m), 1.3 * m)
    tank = p.hall(x, y - tall, side + 1 * m, side + 1 * m, 3.4 * m, WORKS_IRON, None)
    t_left, t_front, t_right, _ = tank
    for drop in (0.8 * m, 1.7 * m, 2.6 * m):
        _bands(p, t_left, t_front, t_right, drop, shade(WORKS_IRON, 1.3), shade(WORKS_IRON, 1.1))
    _works_pyramid(p, tank, 3 * m)


def _works_engine_house(p: Painter, x: float, y: float, size: tuple[float, float, float]) -> None:
    """A red-brick engine house `size` (length, depth and wall height) standing on (x, y): cream
    bands, a tall round-headed window and a dark doorway, under a slate roof."""
    m = METRE
    length, depth, high = size
    engine = p.hall(x, y, length, depth, high, MILL_BRICK, None)
    left, front, right, _ = engine
    _plinth(p, engine, high, MILL_BRICK, 0.6 * m)
    _bands(p, left, front, right, 0.3 * m, DRESSING, shade(DRESSING, 0.78))
    for t in (0.3, 0.7):
        _mill_window(p, left, front, length, t, 1.8 * m, (1.2 * m, high - 1.4 * m))
    _patch(p, front, right, (0.3, 0.7), (high - 3.6 * m, high), DOORWAY)
    p.gable(engine, 2.2 * m, MILL_SLATE, MILL_BRICK)
    _roof_rows(p, engine, 2.2 * m, MILL_SLATE)


def _works_siding(p: Painter, a: Point, b: Point) -> None:
    """A standard-gauge siding along the ground from `a` to `b`, which run the way lit walls do:
    a bed of grey ballast, timber sleepers across it and two steel rails."""
    m = METRE
    across = (m, -m / 2)
    edge = [(a[0] + across[0] * s, a[1] + across[1] * s) for s in (-1.5, 1.5)]
    far = [(b[0] + across[0] * s, b[1] + across[1] * s) for s in (1.5, -1.5)]
    p.polygon([edge[0], edge[1], far[0], far[1]], WORKS_BALLAST)
    length = math.hypot(b[0] - a[0], 2 * (b[1] - a[1])) / m
    count = round(length / 0.75)
    for k in range(count):
        t = (k + 0.5) / count
        cx, cy = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        _bar(
            p,
            (cx - across[0] * 1.25, cy - across[1] * 1.25),
            (cx + across[0] * 1.25, cy + across[1] * 1.25),
            0.25 * m,
            WORKS_SLEEPER,
        )
    for s in (-0.72, 0.72):
        _bar(
            p,
            (a[0] + across[0] * s, a[1] + across[1] * s),
            (b[0] + across[0] * s, b[1] + across[1] * s),
            0.1 * m,
            WORKS_RAIL,
        )


def _works_wagon(p: Painter, x: float, y: float, covered: bool) -> None:
    """A four-wheeled goods wagon 7 m long standing on a siding at (x, y) along the lit walls: a
    red-brown van with a pale arched roof and a sliding door if `covered`, or a grey open wagon
    piled with crated goods, on spoked wheels with buffers at its ends."""
    m = METRE
    for t in (0.22, 0.78):
        wx, wy = _iso(x, y, (t - 0.5) * 7 * m, 0.7 * m)
        p.ellipse(wx - 0.5 * m, wy - 1 * m, wx + 0.5 * m, wy, (40, 38, 40))
    p.hall(x, y - 0.6 * m, 7 * m, 2.2 * m, 0.4 * m, (44, 42, 44), None)
    for t in (0.22, 0.78):
        wx, wy = _iso(x, y, (t - 0.5) * 7 * m, -0.7 * m)
        p.ellipse(wx - 0.5 * m, wy - 1 * m, wx + 0.5 * m, wy, (36, 34, 36))
        p.blob(wx - 0.15 * m, wy - 0.65 * m, wx + 0.15 * m, wy - 0.35 * m, (120, 118, 120))
    for u in (-3.7, 3.7):
        bx, by = _iso(x, y, u * m, 0)
        _bar(
            p,
            (bx, by - 1.1 * m),
            _iso(bx, by - 1.1 * m, 0.3 * m if u > 0 else -0.3 * m, 0),
            0.25 * m,
            (40, 38, 40),
        )
    if covered:
        body = p.hall(x, y - 1 * m, 7 * m, 2.6 * m, 2.3 * m, WORKS_VAN, None)
        left, front, right, _ = body
        for t in np.linspace(0.06, 0.94, 12):
            _bar(
                p,
                _on_wall(left, front, t, 0),
                _on_wall(left, front, t, 2.3 * m),
                0.04 * m,
                shade(WORKS_VAN, 0.82),
            )
        _patch(p, left, front, (0.38, 0.62), (0.3 * m, 2.2 * m), shade(WORKS_VAN, 0.86))
        _bar(
            p,
            _on_wall(left, front, 0.38, 0.3 * m),
            _on_wall(left, front, 0.62, 2.2 * m),
            0.08 * m,
            shade(WORKS_VAN, 0.65),
        )
        _bar(
            p,
            _on_wall(left, front, 0.62, 0.3 * m),
            _on_wall(left, front, 0.38, 2.2 * m),
            0.08 * m,
            shade(WORKS_VAN, 0.65),
        )
        p.gable(body, 0.5 * m, (200, 198, 192), WORKS_VAN)
        return
    grey = (104, 106, 110)
    body = p.hall(x, y - 1 * m, 7 * m, 2.6 * m, 1.1 * m, grey, shade(grey, 0.6))
    left, front, right, back = body
    for t in np.linspace(0.12, 0.88, 6):
        _bar(
            p,
            _on_wall(left, front, t, 0),
            _on_wall(left, front, t, 1.1 * m),
            0.06 * m,
            shade(grey, 0.75),
        )
    for k in range(4):
        cx, cy = _on_wall(
            _on_wall(left, back, 0.5, 0), _on_wall(front, right, 0.5, 0), (k + 0.5) / 4, 0
        )
        p.hall(
            cx,
            cy + 0.3 * m,
            1.3 * m,
            1.3 * m,
            0.9 * m + 0.3 * m * (k % 2),
            GOODS_CRATE,
            shade(GOODS_CRATE, 1.15),
        )


def _works_water_tower(p: Painter, x: float, y: float) -> None:
    """A water tower standing on (x, y): a red-brick tower 5 m square and 13 m high with cream
    bands and round-headed openings, carrying a round iron tank riveted in bands under a low
    cone of a roof."""
    m = METRE
    tall = 13 * m
    tower = p.hall(x, y, 5 * m, 5 * m, tall, MILL_BRICK, None)
    left, front, right, _ = tower
    _plinth(p, tower, tall, MILL_BRICK)
    for a, b in ((left, front), (front, right)):
        _bar(p, _on_wall(a, b, 0, 0.3 * m), _on_wall(a, b, 1, 0.3 * m), 0.4 * m, DRESSING)
        for head in (3 * m, 7.5 * m):
            _mill_window(p, a, b, 5 * m, 0.5, 1.2 * m, (head, head + 2.4 * m))
    r, high = 3.4 * m, 4.2 * m
    _turned(p, x, y - tall, [(0, r), (high, r)], WORKS_IRON)
    for f in (0.25, 0.5, 0.75):
        hoop = [
            (x + r * math.cos(a), y - tall - high * f + r / 2 * math.sin(a))
            for a in np.linspace(0, math.pi, 9)
        ]
        for k in range(len(hoop) - 1):
            _bar(p, hoop[k], hoop[k + 1], 0.1 * m, shade(WORKS_IRON, 1.35))
    _turned(p, x, y - tall - high, [(0, r * 1.05), (1.4 * m, 0)], MILL_SLATE)


def big_factory() -> Sprite:
    """A big Victorian brick works round a yard: two long red-brick halls under north-light roofs
    of glazed teeth set at a right angle, rows of round-headed iron-framed windows under cream
    arches, arched loading doors with hoists, a clock tower at the corner carrying an iron water
    tank, an engine house and a taller round chimney smoking behind, a siding along the front
    with a red van and an open wagon of crated goods, and crates stacked in the yard."""
    m = METRE
    p = Painter(60, 58, (30, 40))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 20) * m, (v - 7) * m)

    p.polygon([at(2, -9), at(46, -9), at(46, 0), at(2, 0)], CINDERS)
    for (u, v), length, depth in (
        ((15, 6), 30, 12),
        ((39.5, 8.5), 11, 17),
        ((-3.25, 3.25), 6.5, 6.5),
    ):
        _bed(p, _footing(*at(u, v), length * m, depth * m), margin=0.6)
    p.shadow(22, 6, 3)
    _tall_chimney(p, *at(29, 24), 60 * m, 1.3)
    _works_engine_house(p, *at(22, 20), (10 * m, 7 * m, 7 * m))
    _works_tower(p, *at(-3.25, 3.25), 6.5 * m, 21 * m)
    _works_hall(p, *at(15, 6), (30 * m, 12 * m, 10 * m), 8, door=5)
    _works_hall(p, *at(39.5, 8.5), (11 * m, 17 * m, 10 * m), 3, door=1)
    _works_siding(p, at(0, -6), at(46, -6))
    _works_wagon(p, *at(9, -6), True)
    _works_wagon(p, *at(18.5, -6), False)
    _crates(p, *at(29, -2.5), (3, 2, 1))
    _crates(p, *at(33.5, -2.2), (2, 1))
    _girders(p, *at(40, -2.5), 6, 3)
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


OAK: RGB = (170, 114, 62)
CASK_HEAD: RGB = (204, 160, 104)
CASK_HOOP: RGB = (62, 58, 58)

type Way = tuple[float, float, float]

# The ways a cask can lie (_cask), each as the way along its axis from its hidden head to the one
# in sight and two ways across it: on its side with its head to the lit side ("v"), on its side
# with its head to the shaded side ("u") or standing on end ("up"). Ways are in metres along the
# lit walls, along the shaded walls and up.
CASK_AXES: dict[str, tuple[Way, Way, Way]] = {
    "v": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "u": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "up": ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
}


def _cask_at(x: float, y: float, u: float, v: float, z: float = 0) -> Point:
    """The point `u` metres along the lit walls, `v` along the shaded walls and `z` up from
    (x, y)."""
    m = METRE
    return x + (u + v) * m, y + (u - v) * m / 2 - z * m


STEEL_JACKET: RGB = (104, 104, 110)
STEEL_SLAG: RGB = (84, 80, 78)
STEEL_SHELL: RGB = (152, 158, 168)
STEEL_STOVE: RGB = (176, 180, 188)
STEEL_FRAME: RGB = (78, 84, 96)
STEEL_PIPE: RGB = (128, 134, 146)
STEEL_CLAD: RGB = (168, 170, 172)
STEEL_ROOF: RGB = (160, 80, 54)
STEEL_BRICK: RGB = (150, 70, 52)
STEEL_STOCK: RGB = (122, 142, 168)
STEEL_ORE: RGB = (158, 74, 50)
STEEL_COKE: RGB = (72, 72, 78)
STEEL_LIME: RGB = (204, 200, 188)
STEEL_GROUND: RGB = (116, 108, 100)
STEEL_BIN: RGB = (158, 152, 142)
STEEL_WAGON: RGB = (96, 92, 90)
STEEL_BALLAST: RGB = (138, 130, 118)
STEEL_SLEEPER: RGB = (80, 64, 50)
STEEL_FUME: RGB = (196, 178, 164)
HOT_WHITE: RGB = (255, 246, 214)
HOT_YELLOW: RGB = (255, 206, 92)
HOT_ORANGE: RGB = (240, 122, 34)
RED_HOT: RGB = (186, 58, 30)


def _steel_hoop(p: Painter, x: float, y: float, r: float, wide: float, colour: RGB) -> None:
    """A hoop `wide` deep round the front of an upright round thing `r` wide whose axis passes
    through (x, y), shaded on its right third as the thing is (_turned)."""
    split = math.acos(1 / 3)
    for a0, a1, tone in ((split, math.pi, 1.0), (0, split, 0.78)):
        arc = [(x + r * math.cos(a), y + r / 2 * math.sin(a)) for a in np.linspace(a0, a1, 7)]
        p.polygon(arc + [(ax, ay + wide) for ax, ay in arc[::-1]], shade(colour, tone))


def _steel_round(
    p: Painter,
    x: float,
    y: float,
    profile: list[tuple[float, float]],
    colour: RGB,
    hoops: tuple[float, ...] = (),
    cap: RGB | None = None,
) -> None:
    """An upright round thing of riveted steel plate (_turned) with a light streak down its lit
    side, so that it reads as round, and a dark hoop round it at each of `hoops` heights."""
    _turned(p, x, y, profile, colour, cap)
    heights, widths = [h for h, _ in profile], [w for _, w in profile]
    outer = [(x - 0.64 * w, y - h + 0.38 * w) for h, w in profile]
    inner = [(x - 0.4 * w, y - h + 0.46 * w) for h, w in profile]
    p.polygon(outer + inner[::-1], shade(colour, 1.14))
    for h in hoops:
        w = float(np.interp(h, heights, widths))
        _steel_hoop(p, x, y - h, w, 0.2 * METRE, shade(colour, 0.7))


def _steel_pipe(p: Painter, points: list[Point], wide: float, colour: RGB) -> None:
    """A steel pipe `wide` thick bending through `points`, with rounded bends and a light streak
    along its upper side."""
    for a, b in itertools.pairwise(points):
        _bar(p, a, b, wide, colour)
    for x, y in points[1:-1]:
        p.blob(x - wide / 2, y - wide / 2, x + wide / 2, y + wide / 2, colour)
    lift = wide * 0.25
    for (ax, ay), (bx, by) in itertools.pairwise(points):
        _bar(p, (ax - lift, ay - lift), (bx - lift, by - lift), wide * 0.25, shade(colour, 1.3))


def _steel_furnace(p: Painter, x: float, y: float, tall: float, r: float, down: Point) -> Point:
    """A blast furnace standing on (x, y): a hearth and bosh of dark iron jacketed in cooling
    plates, `r` wide at the bosh and fed through goosenecks from the bustle main round its waist,
    with tuyeres glowing at its foot, under a riveted stack of light steel tapering up `tall` to
    a deck railed round its bell; on the deck a lattice headgear with its sheave, and two gas
    uptakes arching over to a bleeder with a flame at its tip, a downcomer falling from them to
    `down`. Returns where its skip incline arrives."""
    m = METRE
    bosh = 6.5 * m
    _steel_round(
        p,
        x,
        y,
        [(0, 0.78 * r), (2.6 * m, 0.78 * r), (bosh, r)],
        STEEL_JACKET,
        (1 * m, 1.9 * m, 3.6 * m, 4.6 * m, 5.6 * m),
    )
    _steel_round(
        p,
        x,
        y - bosh,
        [(0, r), (tall - bosh - 0.8 * m, 0.64 * r), (tall - bosh, 0.62 * r)],
        STEEL_SHELL,
        tuple(np.arange(1.6 * m, tall - bosh - 0.5 * m, 2.6 * m)),
    )
    ring = r + 1 * m
    main = y - bosh - 1.4 * m
    for a in np.linspace(0.18 * math.pi, 0.82 * math.pi, 5):
        tx, ty = x + 0.8 * r * math.cos(a), y - 2.2 * m + 0.4 * r * math.sin(a)
        gx, gy = x + ring * math.cos(a), main + ring / 2 * math.sin(a)
        _steel_pipe(p, [(gx, gy), (gx, ty - 1.2 * m), (tx, ty)], 0.45 * m, STEEL_PIPE)
        p.blob(tx - 0.22 * m, ty - 0.18 * m, tx + 0.22 * m, ty + 0.18 * m, HOT_YELLOW)
    _steel_hoop(p, x, main - 0.5 * m, ring, 1.1 * m, STEEL_PIPE)
    _steel_hoop(p, x, main - 0.5 * m, ring, 0.25 * m, shade(STEEL_PIPE, 1.3))
    top = y - tall
    deck = 0.62 * r + 1.4 * m
    p.ellipse(x - deck, top - deck / 2 + 0.4 * m, x + deck, top + deck / 2 + 0.4 * m, STEEL_FRAME)
    p.ellipse(x - deck, top - deck / 2, x + deck, top + deck / 2, shade(STEEL_SHELL, 1.1))
    apex = (x - 0.4 * deck, top - 8.5 * m)
    feet = ((x - 0.85 * deck, top + 0.1 * m), (x + 0.05 * deck, top - 0.4 * m))
    for fx, fy in feet:
        _bar(p, (fx, fy), apex, 0.4 * m, STEEL_FRAME)
    for k in range(1, 4):
        t = k / 4
        a = (feet[0][0] + (apex[0] - feet[0][0]) * t, feet[0][1] + (apex[1] - feet[0][1]) * t)
        b = (feet[1][0] + (apex[0] - feet[1][0]) * t, feet[1][1] + (apex[1] - feet[1][1]) * t)
        _bar(p, a, b, 0.2 * m, STEEL_FRAME)
        if k < 3:
            c = (
                feet[1][0] + (apex[0] - feet[1][0]) * (t + 0.25),
                feet[1][1] + (apex[1] - feet[1][1]) * (t + 0.25),
            )
            _bar(p, a, c, 0.14 * m, STEEL_FRAME)
    sx, sy = apex[0], apex[1] - 0.3 * m
    p.ellipse(sx - 1 * m, sy - 1 * m, sx + 1 * m, sy + 1 * m, STEEL_FRAME)
    p.blob(sx - 0.65 * m, sy - 0.65 * m, sx + 0.65 * m, sy + 0.65 * m, shade(STEEL_SHELL, 1.15))
    p.blob(sx - 0.22 * m, sy - 0.22 * m, sx + 0.22 * m, sy + 0.22 * m, STEEL_FRAME)
    _steel_round(
        p,
        x,
        top,
        [(0, 0.52 * r), (1.6 * m, 0.32 * r), (2.8 * m, 0.28 * r)],
        STEEL_PIPE,
        cap=shade(STEEL_PIPE, 1.2),
    )
    shoulder, crown = top - 4 * m, top - 6.2 * m
    for side in (-1, 1):
        ux = x + side * 0.55 * deck
        _steel_pipe(
            p,
            [(ux, top + 0.3 * m), (ux, shoulder), (x + side * 0.25 * deck, crown)],
            1 * m,
            STEEL_PIPE,
        )
    _steel_pipe(p, [(x - 0.25 * deck, crown), (x + 0.25 * deck, crown)], 1 * m, STEEL_PIPE)
    _steel_pipe(p, [(x, crown), (x, crown - 2.4 * m)], 0.6 * m, STEEL_PIPE)
    p.rect(x - 0.55 * m, crown - 2.8 * m, x + 0.55 * m, crown - 2.4 * m, STEEL_FRAME)
    for scale, tone in ((1, HOT_ORANGE), (0.55, HOT_YELLOW)):
        p.polygon(
            [
                (x - 0.35 * m * scale, crown - 2.85 * m),
                (x + 0.35 * m * scale, crown - 2.85 * m),
                (x + 0.15 * m, crown - 2.85 * m - 1.3 * m * scale),
            ],
            tone,
        )
    ox = x + 0.55 * deck
    _steel_pipe(p, [(ox, shoulder), (ox + 1.6 * m, shoulder + 1.2 * m), down], 1.3 * m, STEEL_PIPE)
    rail = 1.1 * m
    arc = [
        (x + deck * math.cos(a), top + deck / 2 * math.sin(a)) for a in np.linspace(0, math.pi, 11)
    ]
    for a, b in itertools.pairwise(arc):
        _bar(p, (a[0], a[1] - rail), (b[0], b[1] - rail), 0.14 * m, STEEL_FRAME)
    for px, py in arc[1:-1:2]:
        _bar(p, (px, py), (px, py - rail), 0.12 * m, STEEL_FRAME)
    return apex[0], apex[1] + 2.5 * m


def _steel_dust_catcher(p: Painter, x: float, y: float) -> Point:
    """A dust catcher standing on (x, y): a squat steel drum on a cone over its hopper, standing on
    legs; returns the top of it, where the downcomer comes in."""
    m = METRE
    for dx in (-1.6, 0.4, 1.6):
        _bar(p, (x + dx * m, y), (x + dx * m * 0.7, y - 3 * m), 0.3 * m, STEEL_FRAME)
    _steel_round(
        p,
        x,
        y - 1.5 * m,
        [(0, 0.5 * m), (2 * m, 2.2 * m), (8 * m, 2.2 * m), (9.6 * m, 0.8 * m)],
        STEEL_SHELL,
        (3.5 * m, 6 * m),
    )
    return x, y - 11 * m


def _steel_stove(p: Painter, x: float, y: float, tall: float, r: float) -> None:
    """A Cowper hot-blast stove standing on (x, y): a tall drum of light riveted steel `r` wide,
    banded, with a domed top glinting where the light catches it and a valve on its crown."""
    m = METRE
    dome = [
        (tall + 0.87 * r * math.sin(a), r * math.cos(a)) for a in np.linspace(0, math.pi / 2, 7)
    ]
    _steel_round(p, x, y, [(0, r), *dome], STEEL_STOVE, tuple(np.arange(1.5 * m, tall, 2.4 * m)))
    _steel_hoop(p, x, y - tall, r, 0.35 * m, shade(STEEL_STOVE, 0.62))
    p.blob(
        x - 0.62 * r,
        y - tall - 0.62 * r,
        x - 0.16 * r,
        y - tall - 0.28 * r,
        shade(STEEL_STOVE, 1.25),
    )
    crown = y - tall - 0.87 * r
    _turned(
        p, x, crown + 0.3 * m, [(0, 0.5 * m), (1 * m, 0.5 * m)], STEEL_PIPE, shade(STEEL_PIPE, 1.2)
    )


def _steel_stack(p: Painter, x: float, y: float, tall: float) -> None:
    """A slim riveted steel stack standing on (x, y), drawn for `tall` and lowered
    (CHIMNEY_SHARE), flaring at its foot and sooted at its crown, smoking."""
    m = METRE
    tall *= CHIMNEY_SHARE
    profile = [(0, 1.8 * m), (2 * m, 1.25 * m), (tall - 1 * m, 1.05 * m), (tall, 1.2 * m)]
    _steel_round(p, x, y, profile, STEEL_PIPE, tuple(np.arange(4 * m, tall - 2 * m, 3.5 * m)))
    _steel_hoop(p, x, y - tall, 1.2 * m, 1.6 * m, (64, 62, 64))
    p.ellipse(x - 1.2 * m, y - tall - 0.6 * m, x + 1.2 * m, y - tall + 0.6 * m, (40, 38, 40))
    _smoke(p, x + 1, y - tall - 3, (212, 212, 216), 2)


def _steel_incline(p: Painter, foot: Point, under: Point, rise: float, side: Point) -> None:
    """A skip incline: two lattice girders `side` apart rising from `foot` on the ground to `rise`
    over `under` at a furnace's top, standing on trestles, with sleepers across them and a skip
    of ore on its way up."""
    m = METRE
    head = (under[0], under[1] - rise)

    def at(a: Point, b: Point, t: float) -> Point:
        return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t

    def girder(off: float, colour: RGB) -> list[Point]:
        a = (foot[0] + off * side[0], foot[1] + off * side[1])
        b = (head[0] + off * side[0], head[1] + off * side[1])
        g = (under[0] + off * side[0], under[1] + off * side[1])
        deep = 1.4 * m

        def low(t: float) -> Point:
            x, y = at(a, b, t)
            return x, y + deep * min(1.0, t * 5)

        for t in (0.3, 0.55, 0.8):
            _bar(p, low(t), at(a, g, t), 0.3 * m, shade(colour, 0.85))
        _bar(p, low(0), low(1), 0.22 * m, colour)
        panels = 10
        for k in range(panels):
            t0, t1 = k / panels, (k + 1) / panels
            _bar(p, at(a, b, t0), low(t1), 0.13 * m, colour)
            _bar(p, low(t1), at(a, b, t1), 0.13 * m, colour)
        _bar(p, a, b, 0.32 * m, colour)
        return [at(a, b, k / panels) for k in range(panels + 1)]

    far = girder(-1, shade(STEEL_FRAME, 0.8))
    near_rail = [at(foot, head, k / 10) for k in range(11)]
    for (fx, fy), (nx, ny) in zip(far, near_rail, strict=True):
        _bar(p, (fx, fy), (nx, ny), 0.18 * m, shade(STEEL_FRAME, 0.9))
    t0, t1 = 0.6, 0.65
    a, b = at(foot, head, t0), at(foot, head, t1)
    lift = 0.9 * m
    p.polygon([a, b, (b[0], b[1] - lift), (a[0], a[1] - lift)], shade(STEEL_FRAME, 1.2))
    p.polygon(
        [
            (a[0], a[1] - lift),
            (b[0], b[1] - lift),
            (b[0] - side[0], b[1] - side[1] - lift),
            (a[0] - side[0], a[1] - side[1] - lift),
        ],
        STEEL_ORE,
    )
    girder(0, STEEL_FRAME)
    _bar(p, foot, head, 0.1 * m, shade(STEEL_SHELL, 1.1))


def _steel_pile(
    p: Painter, x: float, y: float, half: float, high: float, colour: RGB, seed: int = 0
) -> None:
    """A heap of lumpy stuff `half` wide either side of (x, y) and `high` tall, lit on its left
    and strewn with lumps lighter and darker than it."""
    m = METRE
    apex = (x - 0.12 * half, y - high)
    split = 1.25

    def arc(a0: float, a1: float) -> list[Point]:
        return [
            (x + half * math.cos(a), y + half / 2 * math.sin(a)) for a in np.linspace(a0, a1, 8)
        ]

    p.polygon([*arc(split, math.pi), apex], colour)
    p.polygon([*arc(0, split), apex], shade(colour, 0.74))
    rng = np.random.default_rng(seed)
    for _ in range(int(6 + half / m * 4)):
        a = rng.uniform(0.15, math.pi - 0.15)
        s = rng.uniform(0.05, 0.8)
        bx, by = x + half * math.cos(a), y + half / 2 * math.sin(a)
        lx, ly = bx + (apex[0] - bx) * s, by + (apex[1] - by) * s
        tone = rng.uniform(1.15, 1.4) if math.cos(a) < 0.33 else rng.uniform(0.55, 0.7)
        r = rng.uniform(0.16, 0.3) * m
        p.blob(lx - r, ly - r * 0.7, lx + r, ly + r * 0.7, shade(colour, tone))


def _steel_bins(p: Painter, x: float, y: float, rows: tuple[tuple[RGB, ...], ...]) -> None:
    """A grid of open stock bins whose back left corner is on (x, y): low concrete walls round
    bins 4 m square, a row of them behind each of `rows`, every bin
    heaped over its walls with its load, such as ore, coke or limestone."""
    m = METRE
    cell, deep, wall, high = 4.0, 4.0, 0.4, 1.6
    long = cell * max(len(row) for row in rows)

    def at(u: float, v: float) -> Point:
        return _iso(x, y, u * m, -v * m)

    def wall_on(u0: float, v0: float, length: float, depth: float) -> None:
        cx, cy = at(u0 + length / 2, v0 + depth / 2)
        p.hall(cx, cy, length * m, depth * m, high * m, STEEL_BIN, shade(STEEL_BIN, 1.18))

    wall_on(0, 0, long + wall, wall)
    for j, row in enumerate(rows):
        for k, load in enumerate(row):
            wall_on(k * cell, j * deep, wall, deep)
            _steel_pile(
                p, *at((k + 0.55) * cell, (j + 0.5) * deep), 2.1 * m, 2.9 * m, load, 7 * j + k
            )
        wall_on(len(row) * cell, j * deep, wall, deep)
        wall_on(0, (j + 1) * deep - wall, long + wall, wall)


def _steel_siding(p: Painter, x: float, y: float, length: float) -> None:
    """A siding on the ground from (x, y) `length` metres along the lit walls: a bed of ballast,
    sleepers across it and two rails catching the light."""
    m = METRE

    def at(u: float, v: float) -> Point:
        return _iso(x, y, u * m, v * m)

    p.polygon([at(0, -1.7), at(length, -1.7), at(length, 1.7), at(0, 1.7)], STEEL_BALLAST)
    for u in np.arange(0.4, length, 0.8):
        _bar(p, at(u, -1.2), at(u, 1.2), 0.3 * m, STEEL_SLEEPER)
    for v in (-0.72, 0.72):
        _bar(p, at(0, v), at(length, v), 0.2 * m, shade(STEEL_PIPE, 0.6))
        (ax, ay), (bx, by) = at(0, v), at(length, v)
        _bar(p, (ax, ay - 0.12 * m), (bx, by - 0.12 * m), 0.1 * m, shade(STEEL_SHELL, 1.15))


def _steel_hopper(p: Painter, x: float, y: float, load: RGB) -> None:
    """A four-wheeled hopper wagon 6 m long standing along the lit walls with its middle on
    (x, y) of a siding: a ribbed body with a lit rim, its chutes hanging between its wheels,
    heaped with `load`."""
    m = METRE
    body = STEEL_WAGON
    rail = _iso(x, y, 0, -0.72 * m)
    for du in (-1.9, 1.9):
        wx, wy = _iso(*rail, du * m, 0)
        p.ellipse(wx - 0.5 * m, wy - 1 * m, wx + 0.5 * m, wy, (42, 40, 42))
        p.blob(wx - 0.15 * m, wy - 0.65 * m, wx + 0.15 * m, wy - 0.35 * m, (120, 118, 118))
    high, lift = 2 * m, 1.2 * m
    top = p.hall(x, y - lift, 6 * m, 2.7 * m, high, body, None)
    left, front, right, _ = top
    base_l, base_f = (left[0], left[1] + high), (front[0], front[1] + high)
    p.polygon(
        [
            _on_wall(base_l, base_f, 0.3, 0),
            _on_wall(base_l, base_f, 0.7, 0),
            _on_wall(base_l, base_f, 0.58, 0.75 * m),
            _on_wall(base_l, base_f, 0.42, 0.75 * m),
        ],
        shade(body, 0.7),
    )
    _bar(p, base_l, base_f, 0.3 * m, (50, 48, 50))
    for t in (0.04, 0.27, 0.5, 0.73, 0.96):
        _bar(
            p,
            _on_wall(left, front, t, 0),
            _on_wall(left, front, t, high),
            0.14 * m,
            shade(body, 0.72),
        )
    _bar(p, left, front, 0.2 * m, shade(body, 1.6))
    _bar(p, front, right, 0.2 * m, shade(body, 1.3))
    p.gable(top, 0.9 * m, load, load)
    ridge_back, ridge_front = _ridge(top, 0.9 * m)
    for t, up in ((0.2, 0.4), (0.45, 0.7), (0.7, 0.35), (0.85, 0.6)):
        lx, ly = _on_wall(
            _on_wall(left, front, t, 0), _on_wall(ridge_back, ridge_front, t, 0), up, 0
        )
        p.blob(lx - 0.2 * m, ly - 0.15 * m, lx + 0.2 * m, ly + 0.15 * m, shade(load, 1.3))


def _steel_stock(p: Painter, x: float, y: float, long: float, wide: float, layers: int) -> None:
    """Finished steel, rails and beams `long` metres long, stacked `layers` deep and `wide`
    across on timber bearers at (x, y) along the lit walls: blue-grey, lit along the top so each
    length shows, their ends dark on the shaded side."""
    m = METRE
    for along in (-0.32, 0.32):
        bx, by = _iso(x, y, along * long * m, 0)
        p.hall(
            bx, by, 0.35 * m, wide * m + 0.6 * m, 0.3 * m, shade(TIMBER, 0.62), shade(TIMBER, 0.8)
        )
    high = 0.32 * m * layers
    top = p.hall(x, y - 0.3 * m, long * m, wide * m, high, STEEL_STOCK, shade(STEEL_STOCK, 1.32))
    left, front, right, back = top
    count = round(wide / 0.45)
    for k in range(1, count):
        s = k / count
        _bar(
            p,
            _on_wall(left, back, s, 0),
            _on_wall(front, right, s, 0),
            0.06 * m,
            shade(STEEL_STOCK, 0.92),
        )
    for k in range(1, layers):
        _band(p, left, front, high * k / layers, shade(STEEL_STOCK, 0.7))
    for k in range(layers):
        for s in np.linspace(0.12, 0.88, count):
            ex, ey = _on_wall(front, right, s, high * (k + 0.5) / layers)
            p.blob(
                ex - 0.13 * m, ey - 0.1 * m, ex + 0.13 * m, ey + 0.1 * m, shade(STEEL_STOCK, 0.42)
            )


def _steel_corrugated(p: Painter, top: Top, rise: float, roof: RGB) -> None:
    """Corrugations running down the lit side of a gable roof `rise` high on a block's top."""
    m = METRE
    left, front, _, _ = top
    ridge_back, ridge_front = _ridge(top, rise)
    count = max(2, round(math.dist(left, front) / (0.7 * m)))
    for k in range(1, count):
        t = k / count
        _bar(
            p,
            _on_wall(left, front, t, 0),
            _on_wall(ridge_back, ridge_front, t, 0),
            0.1 * m,
            shade(roof, 0.84 if k % 2 else 1.08),
        )


def _steel_glow(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: tuple[float, float]
) -> None:
    """An open doorway on the wall under the edge from `a` to `b` with hot metal glowing inside,
    darkest at its head and brightest along the floor."""
    (t0, t1), (d0, d1) = span, drop
    tones = ((40, 28, 26), (96, 36, 24), RED_HOT, HOT_ORANGE, HOT_YELLOW)
    for k, tone in enumerate(tones):
        inset = (t1 - t0) * 0.07 * k
        top = d0 + (d1 - d0) * (0.0 if k == 0 else 0.3 + 0.15 * k)
        _patch(p, a, b, (t0 + inset, t1 - inset), (top, d1), tone)


def _steel_ingots(p: Painter, x: float, y: float, count: int) -> None:
    """A bogie of `count` red-hot ingots standing upright on (x, y) along the lit walls, glowing
    yellow at their tops."""
    m = METRE
    long = count * 1.3 * m + 0.5 * m
    rail = _iso(x, y, 0, -0.72 * m)
    for du in (-0.32, 0.32):
        wx, wy = _iso(*rail, du * long, 0)
        p.ellipse(wx - 0.4 * m, wy - 0.8 * m, wx + 0.4 * m, wy, (42, 40, 42))
    p.hall(x, y - 0.7 * m, long, 1.8 * m, 0.35 * m, (58, 56, 58), (88, 86, 90))
    for k in range(count):
        ix, iy = _iso(x, y - 1.05 * m, (k - (count - 1) / 2) * 1.3 * m, 0)
        p.hall(ix, iy, 0.75 * m, 0.75 * m, 1.9 * m, HOT_ORANGE, HOT_YELLOW)


def _steel_sparks(
    p: Painter, x: float, y: float, reach: float, count: int, seed: int, lean: float = 0
) -> None:
    """`count` sparks thrown from (x, y) up to `reach` away and falling, leaning right by
    `lean` radians, bright where they start and reddening as they fall."""
    m = METRE
    rng = np.random.default_rng(seed)
    for _ in range(count):
        angle = rng.uniform(-math.pi * 0.85, -math.pi * 0.15) + lean
        speed = rng.uniform(0.45, 1) * reach
        for s, tone, r in ((0.35, HOT_WHITE, 0.24), (0.7, HOT_YELLOW, 0.2), (1, HOT_ORANGE, 0.16)):
            sx = x + math.cos(angle) * speed * s
            sy = y + math.sin(angle) * speed * s + 0.6 * reach * s * s
            p.blob(sx - r * m, sy - r * m, sx + r * m, sy + r * m, tone)


def _steel_converter(p: Painter, x: float, y: float, size: float, pouring: bool = False) -> None:
    """A Bessemer converter standing on (x, y), `size` across: a pear-shaped iron vessel on its
    trunnion stand, rimmed by the glow; tipped to the left and blowing a roaring flame from its
    mouth, or if `pouring` tipped to the right and pouring a thin bright stream of steel into a
    ladle on the floor."""
    iron = (54, 46, 46)
    s = size
    p.polygon(
        [
            (x - 0.55 * s, y),
            (x - 0.3 * s, y - 0.9 * s),
            (x + 0.3 * s, y - 0.9 * s),
            (x + 0.55 * s, y),
        ],
        iron,
    )
    cx, cy = x, y - 1.05 * s
    p.blob(cx - 0.56 * s, cy - 0.56 * s, cx + 0.44 * s, cy + 0.56 * s, HOT_ORANGE)
    p.ellipse(cx - 0.48 * s, cy - 0.55 * s, cx + 0.5 * s, cy + 0.55 * s, iron)
    if pouring:
        p.polygon(
            [
                (cx + 0.3 * s, cy - 0.42 * s),
                (cx + 0.46 * s, cy + 0.2 * s),
                (cx + 1.0 * s, cy - 0.02 * s),
                (cx + 0.84 * s, cy - 0.5 * s),
            ],
            iron,
        )
        lx, ly = x + 1.05 * s, y
        lip = (cx + 0.97 * s, cy - 0.22 * s)
        _turned(p, lx, ly, [(0, 0.32 * s), (0.75 * s, 0.42 * s)], iron, HOT_ORANGE)
        p.blob(lx - 0.25 * s, ly - 0.83 * s, lx + 0.25 * s, ly - 0.67 * s, HOT_YELLOW)
        _bar(p, lip, (lip[0] + 0.05 * s, ly - 0.75 * s), 0.13 * s, HOT_YELLOW)
        _bar(p, lip, (lip[0] + 0.05 * s, ly - 0.75 * s), 0.05 * s, HOT_WHITE)
        p.blob(cx - 0.08 * s, cy - 0.12 * s, cx + 0.12 * s, cy + 0.08 * s, (90, 80, 78))
        return
    nose = [
        (cx - 0.44 * s, cy - 0.25 * s),
        (cx + 0.22 * s, cy - 0.5 * s),
        (cx - 0.1 * s, cy - 1.05 * s),
        (cx - 0.66 * s, cy - 0.8 * s),
    ]
    p.polygon(nose, iron)
    p.blob(cx - 0.08 * s, cy - 0.12 * s, cx + 0.12 * s, cy + 0.08 * s, (90, 80, 78))
    mx, my = cx - 0.38 * s, cy - 0.93 * s
    dx, dy = -0.36, -1.0
    nx, ny = 1.0, -0.36
    for scale, tone in ((1.0, HOT_ORANGE), (0.72, HOT_YELLOW), (0.42, HOT_WHITE)):
        w, long = 0.3 * s * scale, 2.0 * s * scale
        p.polygon(
            [
                (mx + nx * w, my + ny * w),
                (mx + dx * long * 0.45 + nx * w * 1.5, my + dy * long * 0.45 + ny * w * 1.5),
                (mx + dx * long, my + dy * long),
                (mx + dx * long * 0.45 - nx * w * 1.5, my + dy * long * 0.45 - ny * w * 1.5),
                (mx - nx * w, my - ny * w),
            ],
            tone,
        )


def _steel_shed(
    p: Painter,
    x: float,
    y: float,
    length: float,
    depth: float,
    high: float,
    rise: float,
    door: float,
) -> Top:
    """A long rolling mill: walls of grey corrugated iron on a red brick plinth between steel
    stanchions, a strip of glazing under the eaves, a roof of rust-red corrugated iron with a
    glazed clerestory along its ridge, and a tall open door `door` along its lit wall glowing
    from the hot steel inside."""
    m = METRE
    top = p.hall(x, y, length, depth, high, STEEL_CLAD, None)
    left, front, right, _ = top
    for a, b, size, tone in ((left, front, length, 0.86), (front, right, depth, 0.68)):
        step = 0.6 * m / size
        for t in np.arange(step, 1, step):
            _bar(
                p,
                _on_wall(a, b, t, 1.9 * m),
                _on_wall(a, b, t, high),
                0.08 * m,
                shade(STEEL_CLAD, tone),
            )
        _patch(p, a, b, (0, 1), (0.5 * m, 1.7 * m), shade(GLAZING, 0.92 if tone > 0.8 else 0.74))
        _patch(
            p, a, b, (0, 1), (high - 1.4 * m, high), shade(STEEL_BRICK, 1.0 if tone > 0.8 else 0.78)
        )
        _band(p, a, b, high - 1.4 * m, shade(DRESSING, 0.9))
        bays = max(1, round(size / (4.5 * m)))
        for k in range(bays + 1):
            t = k / bays
            _bar(
                p,
                _on_wall(a, b, t, 0),
                _on_wall(a, b, t, high - 1.4 * m),
                0.35 * m,
                shade(STEEL_FRAME, 1.1 if tone > 0.8 else 0.9),
            )
    half = 2.2 * m / length
    _patch(p, left, front, (door - half - 0.01, door + half + 0.01), (2.3 * m, high), STEEL_FRAME)
    _steel_glow(p, left, front, (door - half, door + half), (2.6 * m, high))
    p.gable(top, rise, STEEL_ROOF, STEEL_CLAD)
    _steel_corrugated(p, top, rise, STEEL_ROOF)
    ridge_back, ridge_front = _ridge(top, rise)
    _bar(p, ridge_back, ridge_front, 0.3 * m, shade(STEEL_ROOF, 1.3))
    wide = min(3 * m, depth * 0.3)
    sink = rise * wide / depth
    mx, my = _on_wall(ridge_back, ridge_front, 0.5, sink)
    lantern = p.hall(mx, my, length * 0.78, wide, 1.8 * m + sink, shade(GLAZING, 0.9), None)
    l_left, l_front, l_right, _ = lantern
    for a, b, size in ((l_left, l_front, length * 0.78), (l_front, l_right, wide)):
        for t in np.arange(0, 1.001, 1.2 * m / size):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, 1.8 * m), 0.12 * m, STEEL_FRAME)
    p.gable(lantern, 0.8 * m, STEEL_ROOF, STEEL_CLAD)
    _bar(p, *_ridge(lantern, 0.8 * m), 0.2 * m, shade(STEEL_ROOF, 1.3))
    return top


def _steel_bessemer(
    p: Painter,
    x: float,
    y: float,
    length: float,
    depth: float,
    high: float,
    seed: int,
    pouring: bool = False,
) -> Top:
    """A Bessemer shed of red brick under a slate roof, its ridge running back from its lit
    gable end, with a louvred lantern along it breathing a brown fume: the gable end stands open
    on the glow inside, where a converter tipped on its trunnions blows a flame from its mouth
    and sparks shower out, or if `pouring` pours its steel into a ladle (_steel_converter)."""
    m = METRE
    rise = 3.5 * m
    top = p.hall(x, y, length, depth, high, STEEL_BRICK, None)
    left, front, right, back = top
    _plinth(p, top, high, STEEL_BRICK, 0.7 * m)
    _stains(p, front, right, high, (0.1, 0.45, 0.9), shade(STEEL_BRICK, 0.55), 1.4 * m)
    _bar(
        p,
        _on_wall(front, right, 0, 0.25 * m),
        _on_wall(front, right, 1, 0.25 * m),
        0.5 * m,
        shade(DRESSING, 0.8),
    )
    for t in (0.28, 0.72):
        _mill_window(p, front, right, depth, t, 1.4 * m, (2 * m, 6.5 * m), shade(MILL_GLASS, 0.85))
    _gable_across(p, top, rise, MILL_SLATE, STEEL_BRICK)
    near, far = _on_wall(left, front, 0.5, -rise), _on_wall(back, right, 0.5, -rise)
    _bar(p, near, far, 0.3 * m, shade(MILL_SLATE, 1.35))
    wide = length * 0.64
    p.polygon(_round_head(left, front, length, 0.5, wide + 0.7 * m, -1.4 * m, high), DRESSING)
    p.polygon(_round_head(left, front, length, 0.5, wide, -1.05 * m, high), (40, 28, 26))
    for k, tone in enumerate(((96, 36, 24), RED_HOT, HOT_ORANGE)):
        p.polygon(
            _round_head(
                left, front, length, 0.5, wide * (0.84 - 0.18 * k), (1.6 + 1.9 * k) * m, high
            ),
            tone,
        )
    cx, cy = _on_wall(left, front, 0.5, high - 0.2 * m)
    if pouring:
        _steel_converter(p, cx - 0.9 * m, cy, 2 * m, pouring=True)
    else:
        _steel_converter(p, cx + 0.3 * m, cy, 2.3 * m)
    sink = rise * 1.6 / (length / m)
    lx, ly = _on_wall(near, far, 0.5, sink)
    louvre = (150, 142, 132)
    vent = p.hall(lx, ly, 1.6 * m, depth * 0.55, 1.2 * m + sink, louvre, None)
    for k in range(1, 4):
        drop = (1.2 * m + sink) * k / 4
        _band(p, vent[0], vent[1], drop, (92, 84, 80))
        _band(p, vent[1], vent[2], drop, (74, 68, 66))
    _gable_across(p, vent, 0.7 * m, MILL_SLATE, louvre)
    vx, vy = _on_wall(vent[3], vent[2], 0.5, -1.8 * m)
    p.soft(
        [
            (vx - 1.3 * m, vy - 1 * m, vx + 1.3 * m, vy + 1 * m),
            (vx + 0.4 * m, vy - 3.6 * m, vx + 3.6 * m, vy - 1 * m),
        ],
        STEEL_FUME,
        0.6,
        0.6,
    )
    sx, sy = _on_wall(left, front, 0.42, high - 5.2 * m)
    if not pouring:
        _steel_sparks(p, sx, sy, 4.5 * m, 14, seed, -0.35)
    return top


def _steel_hot_blast(p: Painter, points: list[Point]) -> None:
    """The hot-blast main through `points`: a fat steel pipe lagged in bands, carried from the
    stoves to a furnace's bustle main."""
    m = METRE
    _steel_pipe(p, points, 1.3 * m, STEEL_PIPE)
    for a, b in itertools.pairwise(points):
        steps = max(1, round(math.dist(a, b) / (2.5 * m)))
        for k in range(1, steps):
            cx, cy = a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps
            _bar(p, (cx, cy - 0.65 * m), (cx, cy + 0.65 * m), 0.25 * m, shade(STEEL_PIPE, 0.7))


def _steel_teeming(p: Painter, x: float, y: float) -> None:
    """Ingots teemed in the open on the siding at (x, y): a bogie of three tall iron moulds along
    the lit walls, and over the first a ladle hung from the jib of a crane, pouring a thin
    bright stream of steel into it."""
    m = METRE
    iron = (62, 58, 60)
    rail = _iso(x, y, 0, -0.72 * m)
    for du in (-1.6, 1.6):
        wx, wy = _iso(*rail, du * m, 0)
        p.ellipse(wx - 0.45 * m, wy - 0.9 * m, wx + 0.45 * m, wy, (42, 40, 42))
    p.hall(x, y - 0.8 * m, 5 * m, 2 * m, 0.4 * m, (58, 56, 58), (88, 86, 90))
    post = _iso(x, y, -1.2 * m, 3.4 * m)
    jib_top = (post[0], post[1] - 9 * m)
    first = _iso(x, y - 1.2 * m, -1.6 * m, 0)
    hook = (first[0], first[1] - 6.6 * m)
    _bar(p, post, jib_top, 0.5 * m, STEEL_FRAME)
    _bar(
        p,
        (jib_top[0], jib_top[1] + 1.4 * m),
        (hook[0] - 0.2 * m, jib_top[1] + 0.6 * m),
        0.35 * m,
        STEEL_FRAME,
    )
    _bar(p, jib_top, (hook[0] - 0.2 * m, jib_top[1] + 0.6 * m), 0.2 * m, STEEL_FRAME)
    _bar(p, (hook[0], jib_top[1] + 0.6 * m), hook, 0.08 * m, (40, 38, 40))
    for k in range(3):
        mx, my = _iso(x, y - 1.2 * m, (k - 1) * 1.6 * m, 0)
        top = p.hall(mx, my, 0.9 * m, 0.9 * m, 2 * m, iron, shade(iron, 1.3))
        if k == 0:
            cx, cy = (top[0][0] + top[2][0]) / 2, (top[0][1] + top[2][1]) / 2
            p.blob(cx - 0.3 * m, cy - 0.15 * m, cx + 0.3 * m, cy + 0.15 * m, HOT_YELLOW)
    _bar(p, (hook[0] - 0.9 * m, hook[1] + 0.9 * m), hook, 0.1 * m, (40, 38, 40))
    _bar(p, (hook[0] + 0.9 * m, hook[1] + 0.9 * m), hook, 0.1 * m, (40, 38, 40))
    _turned(p, hook[0], hook[1] + 2.9 * m, [(0, 0.7 * m), (2 * m, 1 * m)], iron, shade(iron, 0.8))
    p.blob(hook[0] - 0.7 * m, hook[1] + 0.75 * m, hook[0] + 0.7 * m, hook[1] + 1.15 * m, HOT_ORANGE)
    nozzle = (hook[0], hook[1] + 3.1 * m)
    mould = (first[0], first[1] - 2 * m)
    _bar(p, nozzle, mould, 0.2 * m, HOT_YELLOW)
    _bar(p, nozzle, mould, 0.08 * m, HOT_WHITE)
    _steel_sparks(p, mould[0], mould[1], 1.4 * m, 5, 11)


def _steel_trestle(p: Painter, x: float, y: float, length: float, high: float) -> Point:
    """A high-line trestle from (x, y) `length` metres along the lit walls and `high` tall, its
    steel bents braced crosswise every 3 m under a deck of sleepers and rails; returns where its
    middle is on the deck."""
    m = METRE

    def at(u: float, v: float, z: float = 0) -> Point:
        px, py = _iso(x, y, u * m, v * m)
        return px, py - z * m

    for u in np.arange(0, length + 0.01, 3):
        for v in (1.3, -1.3):
            tone = STEEL_FRAME if v < 0 else shade(STEEL_FRAME, 0.8)
            _bar(p, at(u, v), at(u, v * 0.75, high), 0.3 * m, tone)
        _bar(p, at(u, -1.3), at(u, 0.97, high), 0.14 * m, shade(STEEL_FRAME, 0.9))
        if u + 3 <= length + 0.01:
            _bar(p, at(u, -1.3, high * 0.15), at(u + 3, -1.1, high * 0.9), 0.12 * m, STEEL_FRAME)
    deck = [at(0, -1.7, high), at(length, -1.7, high), at(length, 1.7, high), at(0, 1.7, high)]
    p.polygon(deck, shade(STEEL_SLEEPER, 1.2))
    p.polygon(
        [deck[0], deck[1], (deck[1][0], deck[1][1] + 0.7 * m), (deck[0][0], deck[0][1] + 0.7 * m)],
        STEEL_FRAME,
    )
    for v in (-0.72, 0.72):
        _bar(p, at(0, v, high + 0.1), at(length, v, high + 0.1), 0.12 * m, shade(STEEL_SHELL, 1.1))
    return at(length / 2, 0, high)


def _steel_slag(p: Painter, x: float, y: float, half: float, high: float) -> None:
    """A tip of slag at (x, y): a long dark lumpy bank `half` wide either side and `high` tall,
    with a slag pot on a car at its crest tipping a glowing run of molten slag down its lit face
    and steam drifting off it."""
    m = METRE
    _steel_pile(p, x, y, half, high, STEEL_SLAG, 23)
    crest = (x - 0.12 * half, y - high)
    foot = (x - 0.5 * half, y + 0.3 * half)
    run = [
        (crest[0] - 0.2 * m, crest[1] + 0.3 * m),
        (crest[0] + 0.4 * m, crest[1] + 0.4 * m),
        (foot[0] + 0.7 * m, foot[1] - 0.2 * m),
        (foot[0] - 0.8 * m, foot[1]),
    ]
    p.polygon(run, RED_HOT)
    mid = ((run[0][0] + run[3][0]) / 2 + 0.3 * m, (run[0][1] + run[3][1]) / 2)
    _bar(p, run[0], mid, 0.35 * m, HOT_ORANGE)
    p.blob(foot[0] - 1.4 * m, foot[1] - 0.45 * m, foot[0] + 1 * m, foot[1] + 0.5 * m, RED_HOT)
    p.blob(foot[0] - 0.8 * m, foot[1] - 0.25 * m, foot[0] + 0.4 * m, foot[1] + 0.3 * m, HOT_ORANGE)
    pot = (crest[0] + 1.2 * m, crest[1] - 0.6 * m)
    _bar(
        p,
        (pot[0] - 1.6 * m, pot[1] + 0.5 * m),
        (pot[0] + 1.6 * m, pot[1] + 1.3 * m),
        0.4 * m,
        (44, 42, 44),
    )
    p.polygon(
        [
            (pot[0] - 1.4 * m, pot[1] - 1.6 * m),
            (pot[0] + 0.8 * m, pot[1] - 0.6 * m),
            (pot[0] + 0.4 * m, pot[1] + 0.7 * m),
            (pot[0] - 1.2 * m, pot[1] + 0.3 * m),
        ],
        (66, 60, 60),
    )
    p.polygon(
        [(pot[0] - 1.4 * m, pot[1] - 1.6 * m), (pot[0] - 0.9 * m, pot[1] - 1.2 * m), crest],
        HOT_YELLOW,
    )
    p.soft(
        [(x - 0.6 * half, y - high - 4 * m, x + 0.1 * half, y - high + 0.5 * m)],
        (226, 224, 222),
        0.7,
        0.8,
    )


def _steel_flat(p: Painter, x: float, y: float) -> None:
    """A flat wagon 7 m long standing along the lit walls with its middle on (x, y) of a siding,
    loaded with blue-grey rails chained down on bolsters."""
    m = METRE
    rail = _iso(x, y, 0, -0.72 * m)
    for du in (-2.3, 2.3):
        wx, wy = _iso(*rail, du * m, 0)
        p.ellipse(wx - 0.5 * m, wy - 1 * m, wx + 0.5 * m, wy, (42, 40, 42))
    p.hall(x, y - 1 * m, 7 * m, 2.6 * m, 0.45 * m, STEEL_WAGON, shade(STEEL_WAGON, 1.3))
    _steel_stock(p, x, y - 1.45 * m, 6.4, 1.8, 2)


def _steel_hot_car(p: Painter, x: float, y: float) -> None:
    """A hot-metal car with its middle on (x, y) of a siding: a squat iron ladle lined with
    firebrick on a four-wheeled frame, its molten iron glowing under a skin of crust."""
    m = METRE
    iron = (66, 60, 60)
    rail = _iso(x, y, 0, -0.72 * m)
    for du in (-1.5, 1.5):
        wx, wy = _iso(*rail, du * m, 0)
        p.ellipse(wx - 0.45 * m, wy - 0.9 * m, wx + 0.45 * m, wy, (42, 40, 42))
    p.hall(x, y - 0.9 * m, 4.6 * m, 2.4 * m, 0.4 * m, (52, 50, 52), (80, 78, 82))
    _turned(p, x, y - 1.2 * m, [(0, 1.2 * m), (2.2 * m, 1.6 * m)], iron, HOT_ORANGE)
    _steel_hoop(p, x, y - 2.2 * m, 1.45 * m, 0.3 * m, shade(iron, 0.7))
    top = y - 3.4 * m
    p.blob(x - 0.9 * m, top - 0.35 * m, x + 0.7 * m, top + 0.35 * m, HOT_YELLOW)
    p.blob(x - 0.2 * m, top - 0.5 * m, x + 1.1 * m, top + 0.1 * m, RED_HOT)


def _steel_cooling_bed(p: Painter, x: float, y: float, long: float, count: int) -> None:
    """A cooling bed at (x, y) of `count` fresh rails `long` metres long lying side by side along
    the lit walls on steel skids: the one furthest back, just out of the mill, still glowing
    yellow, and the rest reddening and cooling to blue-grey the further out they lie."""
    m = METRE
    gap = 0.45

    def at(u: float, v: float, z: float = 0) -> Point:
        px, py = _iso(x, y, u * m, v * m)
        return px, py - z * m

    wide = gap * (count + 1)
    for u in np.arange(-long / 2, long / 2 + 0.01, long / 6):
        _bar(p, at(u, -wide / 2, 0.3), at(u, wide / 2, 0.3), 0.3 * m, STEEL_FRAME)
        _bar(p, at(u, -wide / 2), at(u, -wide / 2, 0.3), 0.2 * m, shade(STEEL_FRAME, 0.8))
    tones = (HOT_YELLOW, HOT_ORANGE, RED_HOT, (128, 52, 38), (104, 84, 86))
    for k in range(count):
        v = wide / 2 - gap * (k + 1)
        tone = tones[k] if k < len(tones) else STEEL_STOCK
        _bar(p, at(-long / 2, v, 0.5), at(long / 2, v, 0.5), 0.2 * m, shade(tone, 0.8))
        _bar(p, at(-long / 2, v, 0.58), at(long / 2, v, 0.58), 0.08 * m, shade(tone, 1.2))


def steel_mill() -> Sprite:
    """An ironworks of the 1890s on cinder ground: a blast furnace with a dark iron hearth fed
    through goosenecks and a stack of light riveted steel, a lattice headgear and arching gas
    uptakes on its top, fed by a skip incline from a row of stock bins heaped with rust-red ore,
    coke and white limestone, a downcomer falling to its dust catcher, three domed hot-blast
    stoves on the hot-blast main and two steel stacks smoking behind it, a red brick Bessemer
    shed open at its gable on a converter blowing fire and sparks, a long rolling mill of grey
    corrugated iron under a rust-red roof with a glazed clerestory, its door glowing over a
    bogie of red-hot ingots, sidings of hopper wagons and a stack of blue-grey rails and
    beams."""
    m = METRE
    p = Painter(48, 56, (24, 37))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 19) * m, (v - 8) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    ground = [(-1, -1), (6, -7), (44, -7), (42, 10), (30, 23), (6, 23), (-5, 9)]
    p.polygon([at(u, v) for u, v in ground], STEEL_GROUND)
    p.soft([(*at(0, 6), *at(16, -2))], shade(STEEL_ORE, 0.9), 0.35, 1.5)
    p.shadow(18, 5, 2)
    for u, v in ((18, 22.5), (25, 22.5)):
        _steel_stack(p, *at(u, v), 46 * m)
    dust = _steel_dust_catcher(p, *at(13.5, 20.5))
    fx, fy = at(7.5, 17)
    head = _steel_furnace(p, fx, fy, 19 * m, 4.6 * m, dust)
    for u in (21, 26, 31):
        _steel_stove(p, *at(u, 17.5), 15 * m, 2.8 * m)
    _steel_hot_blast(p, [up(31, 14.2, 7), up(13, 14.2, 7), up(11.5, 14.2, 8.2)])
    _steel_siding(p, *at(-4, 11), 21)
    for u, load in ((2, STEEL_ORE), (9, STEEL_COKE)):
        _steel_hopper(p, *at(u, 11), load)
    under = (head[0], fy - 1.5 * m)
    _steel_incline(p, at(7, 8.6), under, under[1] - head[1], _iso(0, 0, 1.7 * m, 0))
    _steel_bins(
        p,
        *at(0.5, 8.6),
        ((STEEL_ORE, STEEL_COKE, STEEL_ORE), (STEEL_COKE, STEEL_LIME, STEEL_ORE)),
    )
    _steel_bessemer(p, *at(21.5, 5), 7 * m, 10 * m, 10 * m, 3)
    _steel_shed(p, *at(33.5, 5), 15 * m, 10 * m, 8 * m, 3 * m, 0.7)
    _steel_stock(p, *at(29.5, -1.7), 8, 2.2, 3)
    _steel_siding(p, *at(5, -5), 39)
    _steel_ingots(p, *at(36, -5), 3)
    return p.sprite()


def _cask_light(way: Way) -> float:
    """How bright a round face turned `way` is drawn: as a building's lit wall when it faces the
    lit side, as its shaded wall when it faces the shaded side and brightest facing up."""
    along_lit, along_shaded, up = way
    return 0.86 - 0.08 * along_lit - 0.14 * along_shaded + 0.18 * up


def _cask(p: Painter, x: float, y: float, axis: str = "v", wood: RGB = OAK) -> None:
    """An oak cask 1.2 m long and 0.84 m round its belly resting on (x, y): on its side with its
    head to the lit side (`axis` "v"), on its side with its head to the shaded side ("u") or
    standing on end ("up"). Its bellied staves are lit like a building's walls and bound by two
    dark iron hoops, and its head is paler end grain ringed by the dark chime, with a bung."""
    m = METRE
    half, head, belly = 0.6, 0.34, 0.42
    along, across, up = CASK_AXES[axis]
    middle = (0.0, 0.0, belly if axis != "up" else half)

    def way(turn: float) -> Way:
        c, s = math.cos(turn), math.sin(turn)
        return (c * across[0] + s * up[0], c * across[1] + s * up[1], c * across[2] + s * up[2])

    def at(s: float, r: float, turn: float) -> Point:
        w = way(turn)
        return _cask_at(x, y, *(middle[i] + s * along[i] + r * w[i] for i in range(3)))

    def radius(s: float) -> float:
        return head + (belly - head) * math.cos(math.pi * s / (2 * half))

    def seen(turn: float) -> bool:
        u, v, z = way(turn)
        return u - v + z > 0

    turns = np.linspace(0, 2 * math.pi, 13)
    for s in np.linspace(half, -half, 9):
        r = radius(s)
        centre = at(s, 0, 0)
        for t0, t1 in itertools.pairwise(turns):
            arc = [at(s, r, t) for t in np.linspace(t0, t1, 3)]
            p.polygon([centre, *arc], shade(wood, _cask_light(way((t0 + t1) / 2))))
    rim = np.linspace(0, 2 * math.pi, 37)
    for s in (0.34, -0.34):
        for t0, t1 in itertools.pairwise(rim):
            if seen(t0) and seen(t1):
                tone = shade(CASK_HOOP, _cask_light(way(t0)) * 1.1)
                _bar(p, at(s, radius(s) + 0.01, t0), at(s, radius(s) + 0.01, t1), 0.07 * m, tone)
    face = _cask_light((-along[0], -along[1], -along[2]))
    p.polygon([at(-half, head, t) for t in rim], shade(wood, face * 0.8))
    p.polygon([at(-half, head * 0.84, t) for t in rim], shade(CASK_HEAD, face))
    bx, by = at(-half, head * 0.45, -math.pi / 2)
    p.blob(bx - 0.06 * m, by - 0.05 * m, bx + 0.06 * m, by + 0.05 * m, shade(CASK_HEAD, 0.62))


def _cask_stack(
    p: Painter,
    x: float,
    y: float,
    rows: tuple[int, ...],
    axis: str = "v",
    wood: RGB = OAK,
    shadow: bool = True,
) -> None:
    """Casks on their sides (_cask) in a neat pyramid from (x, y): `rows[0]` of them side by side
    on the ground, `rows[1]` nestled in the hollows on top of those and so on, along the lit walls
    with their heads to the lit side (`axis` "v") or along the shaded walls with their heads to
    the shaded side ("u"), casting a soft shadow if `shadow`."""
    m = METRE
    gap, rise = 0.86, 0.74

    def spot(d: float, z: float) -> Point:
        return _cask_at(x, y, *((d, 0.0) if axis == "v" else (0.0, -d)), z)

    if shadow:
        feet = [spot(k * gap, 0) for k in range(rows[0])]
        p.soft(
            [(fx - 0.4 * m, fy - 0.35 * m, fx + 1.2 * m, fy + 0.4 * m) for fx, fy in feet],
            (34, 30, 24),
            0.45,
            0.15,
        )
    for level, count in enumerate(rows):
        for k in range(count):
            _cask(p, *spot((k + level * 0.5) * gap, level * rise), axis, wood)


COOPER_WALL: RGB = (140, 98, 60)
COOPER_ROOF: RGB = (90, 94, 102)
STAVE: RGB = (206, 164, 110)
SHAVINGS: RGB = (232, 208, 160)
YARD: RGB = (166, 146, 112)
IRON: RGB = (78, 80, 86)


def _cask_staves(p: Painter, x: float, y: float, layers: int) -> None:
    """A crib of fresh oak staves 1.2 m long stacked on (x, y) to season, `layers` deep, each
    layer of six laid across the one under it, their ends and sides showing on its faces."""
    m = METRE
    side, thick = 1.2 * m, 0.11 * m
    for k in range(layers):
        z = k * thick
        top = p.hall(x, y - z, side, side, thick, STAVE, shade(STAVE, 1.12))
        left, front, right, back = top
        ends, long = ((left, front), (front, right)) if k % 2 else ((front, right), (left, front))
        for j in range(1, 6):
            t = j / 6
            _bar(p, _on_wall(*ends, t, 0), _on_wall(*ends, t, thick), 0.05 * m, shade(STAVE, 0.55))
        _bar(p, _on_wall(*long, 0, thick), _on_wall(*long, 1, thick), 0.03 * m, shade(STAVE, 0.6))
        if k == layers - 1:
            a, b = (left, back) if k % 2 else (left, front)
            c, d = (front, right) if k % 2 else (back, right)
            for j in range(1, 6):
                t = j / 6
                _bar(p, _on_wall(a, b, t, 0), _on_wall(c, d, t, 0), 0.04 * m, shade(STAVE, 0.8))


def _cask_raised(p: Painter, x: float, y: float, fire: bool = False) -> None:
    """A cask being raised on (x, y): its fresh oak staves stood in a ring 1.2 m high, gathered
    at the top in an iron raising hoop with another below it and splayed wide apart at the foot. Over
    a fire, with `fire`, to bend them: flames leaping from its open top, a glow on the ground
    round it and smoke rising."""
    m = METRE
    profile = [(0, 0.66), (0.45, 0.5), (1.0, 0.37), (1.2, 0.35)]
    if fire:
        p.soft([(x - 1.3 * m, y - 0.5 * m, x + 1.3 * m, y + 0.6 * m)], (250, 150, 60), 0.55, 0.3)
    inside = (250, 168, 64) if fire else (96, 70, 48)
    _turned(p, x, y, [(h * m, w * m) for h, w in profile], STAVE, inside)

    def side_at(a: float, h: float) -> Point:
        w = float(np.interp(h, [h for h, _ in profile], [w for _, w in profile]))
        return x + w * m * math.cos(a), y - h * m + w * m / 2 * math.sin(a)

    for a in np.linspace(0.45, math.pi - 0.45, 4):
        _bar(p, side_at(a, 0), side_at(a, 0.4), 0.05 * m, shade(STAVE, 0.62))
    arc = np.linspace(0, math.pi, 10)
    for h in (0.78, 1.1):
        for a0, a1 in itertools.pairwise(arc):
            _bar(p, side_at(a0, h), side_at(a1, h), 0.06 * m, IRON)
    if fire:
        top = y - 1.2 * m
        for dx, tall, tone in ((-0.15, 0.9, (240, 120, 40)), (0.12, 0.65, (250, 196, 80))):
            fx = x + dx * m
            p.polygon(
                [(fx - 0.16 * m, top), (fx + 0.16 * m, top), (fx + 0.04 * m, top - tall * m)],
                tone,
            )
        p.soft(
            [
                (x - 0.3 * m, top - 2.4 * m, x + 0.7 * m, top - 1.4 * m),
                (x + 0.3 * m, top - 3.6 * m, x + 1.5 * m, top - 2.5 * m),
            ],
            (178, 176, 178),
            0.65,
            0.3,
        )


def _cask_hoops(p: Painter, x: float, y: float, sizes: tuple[float, ...]) -> None:
    """Iron hoops `sizes` metres round leant against the wall at whose foot (x, y) lies, one
    beside the other along it, their feet on the ground a little out from it."""
    m = METRE
    u = 0.0
    for r in sizes:
        u += r
        ring = [
            _cask_at(
                x,
                y,
                u + r * math.cos(a),
                -0.12 - 0.3 * (1 - (1 + math.sin(a)) / 2),
                r + r * math.sin(a),
            )
            for a in np.linspace(0, 2 * math.pi, 25)
        ]
        for k, (a, b) in enumerate(itertools.pairwise(ring)):
            lit = 6 <= k <= 13
            _bar(p, a, b, 0.07 * m, shade(IRON, 1.45 if lit else 1.0))
        u += r + 0.15


def _cooper_shavings(p: Painter, spots: list[Point]) -> None:
    """Curls of pale oak shavings scattered on the ground at `spots`."""
    m = METRE
    for k, (x, y) in enumerate(spots):
        r = (0.18 + 0.06 * (k % 3)) * m
        p.blob(x - r, y - r / 2, x + r, y + r / 2, shade(SHAVINGS, 1.0 - 0.06 * (k % 2)))


def barrel_maker() -> Sprite:
    """A cooperage: a weatherboarded oak workshop 12 by 7 m under an iron-grey roof, its big
    doors open on a cask standing in the dark inside, a smoking stove pipe, iron hoops leant
    against its wall and an open lean-to of seasoning staves at its end; in the yard of packed
    earth strewn with shavings, cribs of fresh staves, casks being raised in their hoops, one
    over a fire to bend its staves, a chopping block with an adze, and the finished oak casks
    stacked in a neat pyramid and stood on end."""
    m = METRE
    p = Painter(34, 28, (15, 18))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 6) * m, (v - 1) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon([at(-4, -5), at(15, -5), at(15, 2), at(-4, 2)], YARD)
    _cooper_shavings(
        p,
        [
            at(u, v)
            for u, v in (
                (5.5, 0.4),
                (6.6, -0.6),
                (4.2, -0.2),
                (3.4, -3.4),
                (5.2, -3.8),
                (0.2, -1.8),
                (7.8, 0.8),
            )
        ],
    )
    _bed(p, _footing(*at(6, 5.5), 12 * m, 7 * m), margin=0.6)
    p.shadow(9, 3, 4)
    length, depth, high = 12 * m, 7 * m, 4.6 * m
    top = p.hall(*at(6, 5.5), length, depth, high, COOPER_WALL, None)
    left, front, right, _ = top
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        for k in range(1, 10):
            _bar(
                p,
                _on_wall(a, b, 0, high * k / 10),
                _on_wall(a, b, 1, high * k / 10),
                0.05 * m,
                shade(COOPER_WALL, 0.78 * tone),
            )
        _patch(p, a, b, (0, 1), (high - 0.4 * m, high), shade(STONE, 0.8 * tone))
        for t in (0.0, 1.0):
            _bar(
                p,
                _on_wall(a, b, t, 0),
                _on_wall(a, b, t, high),
                0.2 * m,
                shade(COOPER_WALL, 0.6 * tone),
            )
    _patch(p, left, front, (0.4, 0.7), (1.0 * m, high - 0.05 * m), (36, 30, 28))
    for t0, t1 in ((0.33, 0.4), (0.7, 0.77)):
        _patch(p, left, front, (t0, t1), (1.0 * m, high - 0.05 * m), shade(COOPER_WALL, 1.12))
        _bar(
            p,
            _on_wall(left, front, t0, 1.0 * m),
            _on_wall(left, front, t1, high),
            0.08 * m,
            shade(COOPER_WALL, 0.7),
        )
    _bar(
        p,
        _on_wall(left, front, 0.38, 0.85 * m),
        _on_wall(left, front, 0.72, 0.85 * m),
        0.2 * m,
        shade(COOPER_WALL, 0.6),
    )
    _cask(p, *_on_wall(left, front, 0.6, high - 0.3 * m), "up")
    for t in (0.86,):
        _patch(p, left, front, (t - 0.05, t + 0.05), (1.4 * m, 2.6 * m), (54, 58, 70))
        _bar(
            p,
            _on_wall(left, front, t, 1.4 * m),
            _on_wall(left, front, t, 2.6 * m),
            0.06 * m,
            shade(COOPER_WALL, 1.2),
        )
        _bar(
            p,
            _on_wall(left, front, t - 0.05, 2.0 * m),
            _on_wall(left, front, t + 0.05, 2.0 * m),
            0.06 * m,
            shade(COOPER_WALL, 1.2),
        )
    _patch(p, front, right, (0.35, 0.65), (1.4 * m, 2.6 * m), shade((54, 58, 70), 0.85))
    p.gable(top, 2.6 * m, COOPER_ROOF, COOPER_WALL)
    _roof_rows(p, top, 2.6 * m, COOPER_ROOF, (0.5,))
    px, py = _on_roof(top, 2.6 * m, 0.25, 0.6)
    _bar(p, (px, py), (px, py - 2.2 * m), 0.3 * m, IRON)
    _bar(
        p, (px - 0.35 * m, py - 2.2 * m), (px + 0.35 * m, py - 2.2 * m), 0.15 * m, shade(IRON, 0.7)
    )
    p.soft(
        [
            (px - 0.3 * m, py - 3.8 * m, px + 0.9 * m, py - 2.6 * m),
            (px + 0.4 * m, py - 5.2 * m, px + 1.8 * m, py - 3.9 * m),
        ],
        (190, 188, 190),
        0.6,
        0.3,
    )

    roof_low, roof_high = 2.4, 3.6
    for u, v in ((16, 3.6), (16, 7.6)):
        _bar(p, at(u, v), up(u, v, roof_low), 0.16 * m, shade(TIMBER, 0.7))
    _cask_staves(p, *at(14.2, 5.6), 9)
    _cask_staves(p, *at(14.2, 4.0), 6)
    p.polygon(
        [
            up(12, 3.3, roof_high),
            up(16.4, 3.3, roof_low),
            up(16.4, 8, roof_low),
            up(12, 8, roof_high),
        ],
        shade(COOPER_ROOF, 1.1),
        shade(COOPER_ROOF, 0.7),
    )
    _cask_hoops(p, *at(0.6, 2), (0.5, 0.45, 0.42, 0.5))
    _cask_staves(p, *at(-2.4, 0.8), 7)
    _cask_staves(p, *at(-1.2, -1.2), 5)
    _cask_raised(p, *at(2.6, -1.6))
    _cask_raised(p, *at(4.2, -2.8))
    _cask_raised(p, *at(7.2, -3.0), fire=True)
    sx, sy = at(0.6, -3.6)
    _turned(p, sx, sy, [(0, 0.36 * m), (0.55 * m, 0.34 * m)], shade(OAK, 0.85), shade(STAVE, 1.05))
    _bar(
        p, (sx - 0.1 * m, sy - 0.55 * m), (sx + 0.4 * m, sy - 1.1 * m), 0.07 * m, shade(TIMBER, 0.7)
    )
    _bar(p, (sx + 0.3 * m, sy - 1.15 * m), (sx + 0.55 * m, sy - 0.95 * m), 0.14 * m, IRON)
    _cask_stack(p, *at(9.4, -1.0), (3, 2, 1))
    for u, v in ((12.4, 0.4), (13.2, -0.6)):
        _cask(p, *at(u, v), "up")
    return p.sprite()


FOOD_RENDER: RGB = (238, 226, 198)
FOOD_BRICK: RGB = (178, 82, 56)
FOOD_IRON: RGB = (172, 178, 188)
FOOD_GREEN: RGB = (34, 86, 62)
FOOD_CREAM: RGB = (242, 232, 204)
FOOD_TIN: RGB = (214, 218, 226)
FOOD_LABELS: tuple[RGB, ...] = ((214, 44, 38), (244, 198, 40), (62, 152, 60))
FOOD_CABBAGE: RGB = (110, 160, 72)
FOOD_CARROT: RGB = (238, 122, 30)
FOOD_TOMATO: RGB = (220, 40, 32)
FOOD_WICKER: RGB = (180, 138, 82)
FOOD_CASE: RGB = (208, 174, 124)
FOOD_SETTS: RGB = (150, 140, 124)
FOOD_BAY: RGB = (130, 78, 44)
FOOD_MANE: RGB = (46, 36, 30)
FOOD_FEATHER: RGB = (236, 228, 210)
FOOD_GILT: RGB = (220, 176, 72)
FOOD_WAGON: RGB = (138, 104, 72)
FOOD_WHEEL: RGB = (176, 66, 44)
FOOD_STEEL: RGB = (146, 154, 166)
FOOD_APPLE: RGB = (204, 42, 38)
FOOD_GREEN_APPLE: RGB = (156, 192, 70)
FOOD_PEACH: RGB = (244, 158, 92)
FOOD_CHERRY: RGB = (150, 20, 42)
FOOD_STEAM: RGB = (244, 244, 246)


def _food_horse(p: Painter, x: float, y: float, facing: int = 1, coat: RGB = FOOD_BAY) -> None:
    """A heavy draught horse in harness standing on (x, y) along the lit walls, its head the way
    they run down to the front if `facing` is 1 and up to the back if -1: 2.6 m from nose to tail
    and 1.65 m to the withers, a bay with a dark mane and tail, a white blaze, white feathered
    feet, and a black collar with brass hames."""
    m = METRE

    def at(s: float, z: float, w: float = 0) -> Point:
        px, py = _iso(x, y, facing * s * m, w * m)
        return px, py - z * m

    def oval(s: float, z: float, ls: float, lz: float, w: float) -> list[Point]:
        turns = np.linspace(0, 2 * math.pi, 20, endpoint=False)
        return [at(s + ls * math.cos(a), z + lz * math.sin(a), w) for a in turns]

    dark, hoof = shade(coat, 0.62), (52, 46, 42)
    for s, w in ((0.42, 0.22), (-0.5, 0.22)):
        _bar(p, at(s, 1.05, w), at(s + 0.04, 0.12, w), 0.24 * m, dark)
        _bar(p, at(s + 0.04, 0.24, w), at(s + 0.04, 0, w), 0.24 * m, shade(FOOD_FEATHER, 0.8))
    p.polygon(
        [at(-0.82, 1.5), at(-1.02, 1.1), at(-1.06, 0.62), at(-0.94, 0.66), at(-0.78, 1.18)],
        FOOD_MANE,
    )
    p.polygon(oval(0, 1.22, 0.86, 0.4, 0.2), shade(coat, 0.7))
    p.polygon(oval(-0.56, 1.3, 0.38, 0.38, -0.12), coat)
    p.polygon(oval(0.5, 1.24, 0.4, 0.4, -0.12), coat)
    p.polygon(oval(0, 1.2, 0.8, 0.38, -0.14), coat)
    p.polygon(oval(0, 1.0, 0.66, 0.14, -0.16), shade(coat, 0.84))
    p.polygon(oval(-0.2, 1.5, 0.62, 0.08, -0.16), shade(coat, 1.18))
    head = [
        (0.32, 1.6),
        (0.66, 1.94),
        (0.96, 2.18),
        (1.1, 2.2),
        (1.46, 1.72),
        (1.6, 1.52),
        (1.56, 1.36),
        (1.34, 1.36),
        (1.1, 1.5),
        (0.98, 1.3),
        (0.88, 0.98),
        (0.42, 0.98),
    ]
    p.polygon([at(s, z, -0.1) for s, z in head], coat)
    p.polygon([at(s, z, -0.1) for s, z in ((0.98, 2.14), (1.0, 2.4), (1.14, 2.16))], dark)
    _bar(p, at(0.3, 1.7, -0.1), at(1.02, 2.24, -0.1), 0.18 * m, FOOD_MANE)
    _bar(p, at(1.24, 2.02, -0.11), at(1.54, 1.52, -0.11), 0.09 * m, FOOD_FEATHER)
    p.polygon(oval(1.54, 1.42, 0.08, 0.08, -0.11), shade(coat, 0.55))
    p.polygon(oval(1.14, 1.92, 0.06, 0.06, -0.11), (24, 20, 20))
    _bar(p, at(0.6, 1.98, -0.12), at(0.9, 1.02, -0.12), 0.26 * m, (36, 32, 32))
    _bar(p, at(0.64, 2.1, -0.13), at(0.84, 1.36, -0.13), 0.07 * m, FOOD_GILT)
    p.polygon(oval(0.05, 1.56, 0.2, 0.09, -0.15), (40, 34, 32))
    _bar(p, at(0.78, 1.32, -0.15), at(-0.85, 1.22, -0.15), 0.07 * m, (40, 34, 32))
    for s, knee in ((0.56, 0.0), (-0.6, -0.12)):
        hock = (s + knee, 0.56)
        _bar(p, at(s, 1.1, -0.2), at(*hock, -0.2), 0.27 * m, coat)
        _bar(p, at(*hock, -0.2), at(s, 0.12, -0.2), 0.22 * m, coat)
        p.polygon(oval(s, 0.16, 0.15, 0.16, -0.2), FOOD_FEATHER)
        _bar(p, at(s - 0.1, 0.03, -0.2), at(s + 0.12, 0.03, -0.2), 0.08 * m, hoof)


def _food_wheel(p: Painter, x: float, y: float, r: float, paint: RGB) -> None:
    """A cart wheel `r` units across its middle at (x, y), turning along the lit walls: an iron
    tyre on a felloe in `paint`, six spokes and a hub."""

    def rim(k: float) -> list[Point]:
        turns = np.linspace(0, 2 * math.pi, 20, endpoint=False)
        return [(x + k * math.cos(a), y + k * math.cos(a) / 2 - k * math.sin(a)) for a in turns]

    p.polygon(rim(r), (50, 44, 40))
    p.polygon(rim(0.84 * r), paint)
    p.polygon(rim(0.66 * r), shade(paint, 0.45))
    for k in range(3):
        a = k * math.pi / 3
        dx, dy = 0.7 * r * math.cos(a), 0.7 * r * (math.cos(a) / 2 - math.sin(a))
        _bar(p, (x - dx, y - dy), (x + dx, y + dy), 0.14 * r, paint)
    p.polygon(rim(0.2 * r), shade(paint, 0.6))


def _food_cart(
    p: Painter,
    x: float,
    y: float,
    length: float,
    side: float,
    ground: RGB,
    cargo: Callable[[Callable[[float, float], Point]], None],
) -> None:
    """A four-wheeled farm cart `length` m long and 1.7 m wide standing on (x, y) along the lit
    walls, its bed a metre up with plank sides `side` m high, red wheels, and a horse in its
    shafts ahead of it down towards the front; `cargo` draws its load, given where on the bed's
    floor a point `s` m along it from its middle and `w` m across it lies."""
    m = METRE
    half = length / 2

    def at(s: float, z: float, w: float = 0) -> Point:
        px, py = _iso(x, y, s * m, w * m)
        return px, py - z * m

    def face(s0: float, s1: float, w0: float, w1: float, z0: float, z1: float) -> list[Point]:
        return [at(s0, z0, w0), at(s1, z0, w1), at(s1, z1, w1), at(s0, z1, w0)]

    p.polygon(
        [
            at(-half - 0.4, 0, -1.1),
            at(half + 3.6, 0, -1.1),
            at(half + 3.6, 0, 0.9),
            at(-half, 0, 1),
        ],
        shade(ground, 0.8),
    )
    for s in (-half + 0.65, half - 0.65):
        _food_wheel(p, *at(s, 0.6, 0.9), 0.6 * m, shade(FOOD_WHEEL, 0.75))
    _bar(p, at(half - 0.2, 0.9, 0.42), at(half + 2.5, 1.32, 0.42), 0.1 * m, shade(FOOD_WAGON, 0.6))
    p.polygon(face(-half, half, 0.85, 0.85, 1.0, 1.0 + side), shade(FOOD_WAGON, 0.82))
    p.polygon(face(-half, -half, -0.85, 0.85, 1.0, 1.0 + side), shade(FOOD_WAGON, 0.95))
    p.polygon(
        [at(-half, 1.0, -0.85), at(half, 1.0, -0.85), at(half, 1.0, 0.85), at(-half, 1.0, 0.85)],
        shade(FOOD_WAGON, 0.6),
    )
    cargo(lambda s, w: at(s, 1.0, w))
    p.polygon(face(-half, half, -0.85, -0.85, 0.82, 1.0 + side), FOOD_WAGON)
    for z in np.arange(1.0, 1.0 + side - 0.1, 0.24):
        _bar(p, at(-half, z, -0.86), at(half, z, -0.86), 0.04 * m, shade(FOOD_WAGON, 0.7))
    _bar(p, at(-half, 0.86, -0.86), at(half, 0.86, -0.86), 0.12 * m, shade(FOOD_WAGON, 0.55))
    for s in (-half + 0.08, half - 0.08):
        _bar(p, at(s, 0.82, -0.87), at(s, 1.0 + side, -0.87), 0.1 * m, shade(FOOD_WAGON, 0.65))
    p.polygon(face(half, half, -0.85, 0.85, 0.82, 1.0 + side), shade(FOOD_WAGON, 0.74))
    for s in (-half + 0.65, half - 0.65):
        _food_wheel(p, *at(s, 0.6, -1.0), 0.6 * m, FOOD_WHEEL)
    _food_horse(p, *at(half + 1.55, 0, 0))
    _bar(p, at(half - 0.2, 0.9, -0.42), at(half + 2.5, 1.32, -0.42), 0.1 * m, FOOD_WAGON)


def _food_basket(p: Painter, x: float, y: float, load: RGB) -> None:
    """A round wicker basket 0.8 m across and half a metre high standing on (x, y), heaped with
    `load`."""
    m = METRE
    r = 0.4 * m
    _turned(p, x, y, [(0, 0.32 * m), (0.5 * m, r)], FOOD_WICKER)
    rim = y - 0.5 * m
    p.ellipse(x - r, rim - r / 2, x + r, rim + r / 2, shade(FOOD_WICKER, 0.8))
    p.blob(x - 0.86 * r, rim - 0.7 * r, x + 0.86 * r, rim + 0.36 * r, load)
    for dx, dy in ((-0.4, -0.3), (0.3, -0.1), (0, -0.55)):
        cx, cy = x + dx * r, rim + dy * r
        p.blob(cx - 0.2 * r, cy - 0.16 * r, cx + 0.2 * r, cy + 0.16 * r, shade(load, 1.22))


def _food_cases(p: Painter, x: float, y: float, rows: tuple[int, ...], first: int = 0) -> None:
    """Slatted cases of tins 1.1 by 0.8 m and 0.7 m high stacked on (x, y): a row of `rows[0]`
    of them running along the lit walls, `rows[1]` on those and so on, the tins inside showing
    between the slats silver with labels in red, yellow and green by turns from `first`, and
    rows of silver lids on top."""
    m = METRE
    long, wide, tall = 1.1 * m, 0.8 * m, 0.7 * m
    n = first
    for level, count in enumerate(rows):
        for k in range(count):
            cx, cy = _iso(x, y, (k + level * 0.5) * long, 0)
            top = p.hall(cx, cy - level * tall, long, wide, tall, FOOD_CASE, FOOD_TIN)
            left, front, right, back = top
            label = FOOD_LABELS[n % len(FOOD_LABELS)]
            n += 1
            for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
                _patch(p, a, b, (0.07, 0.93), (0.18 * tall, 0.8 * tall), shade(FOOD_TIN, tone))
                _patch(p, a, b, (0.07, 0.93), (0.3 * tall, 0.68 * tall), shade(label, tone))
                for t in (0.36, 0.64):
                    _bar(
                        p,
                        _on_wall(a, b, t, 0.18 * tall),
                        _on_wall(a, b, t, 0.8 * tall),
                        0.05 * m,
                        shade(label, 0.6 * tone),
                    )
            for i in range(3):
                for j in range(2):
                    near = _on_wall(left, front, (i + 0.5) / 3, 0)
                    far = _on_wall(back, right, (i + 0.5) / 3, 0)
                    lx, ly = _on_wall(near, far, (j + 0.5) / 2, 0)
                    p.ellipse(lx - 0.15 * m, ly - 0.08 * m, lx + 0.15 * m, ly + 0.08 * m, FOOD_TIN)


def _food_tinplate(p: Painter, x: float, y: float, layers: int) -> None:
    """Bundles of bright tinplate sheets 2 by 1 m from the rolling mill, for making the tins,
    stacked `layers` deep on timber bearers on (x, y) along the lit walls, each bundle strapped
    twice and its sheet edges glinting along its shaded end."""
    m = METRE
    for along in (-0.6, 0.6):
        p.hall(*_iso(x, y, along * m, 0), 0.25 * m, 1.2 * m, 0.15 * m, shade(TIMBER, 0.6), None)
    for k in range(layers):
        lift = 0.15 * m + k * 0.4 * m
        left, front, right, back = p.hall(
            x, y - lift, 2 * m, 1 * m, 0.34 * m, shade(FOOD_TIN, 0.92), FOOD_TIN
        )
        for t in (0.25, 0.75):
            _bar(
                p,
                _on_wall(left, front, t, 0),
                _on_wall(left, front, t, 0.34 * m),
                0.06 * m,
                (70, 72, 80),
            )
            _bar(
                p, _on_wall(left, front, t, 0), _on_wall(back, right, t, 0), 0.06 * m, (70, 72, 80)
            )
        for d in (0.1, 0.2):
            _bar(
                p,
                _on_wall(front, right, 0, d * m),
                _on_wall(front, right, 1, d * m),
                0.03 * m,
                shade(FOOD_TIN, 0.6),
            )
        if k < layers - 1:
            _patch(p, left, front, (0, 1), (0.34 * m, 0.4 * m), shade(TIMBER, 0.6))


def _food_window(
    p: Painter,
    a: Point,
    b: Point,
    length: float,
    t: float,
    wide: float,
    span: tuple[float, float],
    glass: RGB = MILL_GLASS,
) -> None:
    """A tall round-headed window of many panes in white-painted bars, framed in red brick on
    the render with a cream stone sill, `wide` across about `t` along the wall under the edge
    from `a` to `b`, which is `length` units long, from `span[0]` down the wall to `span[1]`."""
    m = METRE
    head, foot = span
    half = wide / 2 / length
    tone = 1.0 if glass == MILL_GLASS else 0.78
    p.polygon(
        _round_head(a, b, length, t, wide + 0.6 * m, head - 0.32 * m, foot),
        shade(FOOD_BRICK, tone),
    )
    p.polygon(_round_head(a, b, length, t, wide, head, foot), glass)
    bar = shade(FOOD_CREAM, tone)
    _bar(p, _on_wall(a, b, t, head), _on_wall(a, b, t, foot), 0.07 * m, bar)
    for drop in np.arange(head + wide / 2, foot - 0.3 * m, 0.7 * m):
        _bar(p, _on_wall(a, b, t - half, drop), _on_wall(a, b, t + half, drop), 0.06 * m, bar)
    _patch(
        p, a, b, (t - half * 1.4, t + half * 1.4), (foot, foot + 0.28 * m), shade(FOOD_CREAM, tone)
    )


def _food_sign(p: Painter, a: Point, b: Point, length: float, t: float, drop: float) -> None:
    """A painted name board 6.4 by 2.4 m on the wall under the edge from `a` to `b`, which is
    `length` units long, centred `t` along it with its top `drop` down: deep green in a gilt
    frame, lettered in cream on three lines between gilt rules."""
    m = METRE

    def at(s: float, d: float) -> Point:
        return _on_wall(a, b, t + s * m / length, drop + d * m)

    def quad(s0: float, s1: float, d0: float, d1: float) -> list[Point]:
        return [at(s0, d0), at(s1, d0), at(s1, d1), at(s0, d1)]

    p.polygon(quad(-3.2, 3.2, 0, 2.4), FOOD_GILT)
    p.polygon(quad(-3.02, 3.02, 0.18, 2.22), FOOD_GREEN)
    for d, reach, wide in ((0.62, 2.5, 0.24), (1.2, 2.7, 0.3), (1.76, 1.6, 0.2)):
        for k in range(round(reach * 2.4)):
            s0 = -reach + k * reach * 2 / round(reach * 2.4)
            _bar(
                p,
                at(s0 + 0.08, d),
                at(s0 + reach * 2 / round(reach * 2.4) - 0.14, d),
                wide * m,
                FOOD_CREAM,
            )
    for d in (0.92, 1.48):
        _bar(p, at(-1.2, d), at(1.2, d), 0.06 * m, FOOD_GILT)


def _food_steam(p: Painter, x: float, y: float, size: float = 1) -> None:
    """A few small puffs of steam rising and drifting to the right from (x, y)."""
    m = METRE * size
    for k, (dx, dy, r) in enumerate(((0, 0, 0.45), (0.5, -0.9, 0.7), (1.2, -2.0, 0.95))):
        cx, cy = x + dx * m, y + dy * m
        tone = FOOD_STEAM if k % 2 else shade(FOOD_STEAM, 0.9)
        p.blob(cx - r * m, cy - r * m, cx + r * m, cy + r * m, tone)


def _food_retort(p: Painter, x: float, y: float, length: float, r: float, steam: bool) -> None:
    """A steel retort that cooks the sealed tins under steam: a pressure cylinder `length` m long
    and `r` m in radius lying along the lit walls from (x, y) on two brick saddles, riveted in
    three rings, its domed door facing the shaded side clamped by a ring of bolts round a
    handwheel, a gauge and safety valve on top, blowing off steam if `steam`."""
    m = METRE
    zc = r + 0.45

    def at(s: float, w: float, z: float) -> Point:
        px, py = _iso(x, y, s * m, w * m)
        return px, py - z * m

    def rim(s: float, k: float) -> list[Point]:
        turns = np.linspace(0, 2 * math.pi, 20, endpoint=False)
        return [at(s, k * math.cos(a), zc + k * math.sin(a)) for a in turns]

    for s in (0.9, length - 0.9):
        p.hall(*at(s, 0, 0), 0.5 * m, 1.6 * r * m, zc * m, shade(FOOD_BRICK, 0.8), None)
    turns = np.linspace(0, 2 * math.pi, 19)
    bands = sorted(
        ((turns[k], turns[k + 1]) for k in range(len(turns) - 1)),
        key=lambda b: math.sin(b[0]) - math.cos(b[0]),
    )
    for a0, a1 in bands:
        mid = (a0 + a1) / 2
        tone = 0.86 - 0.12 * math.cos(mid) + 0.24 * math.sin(mid)
        p.polygon(
            [
                at(0, r * math.cos(a0), zc + r * math.sin(a0)),
                at(length, r * math.cos(a0), zc + r * math.sin(a0)),
                at(length, r * math.cos(a1), zc + r * math.sin(a1)),
                at(0, r * math.cos(a1), zc + r * math.sin(a1)),
            ],
            shade(FOOD_STEEL, tone),
        )
    seam = np.linspace(math.pi / 4, 5 * math.pi / 4, 9)
    for s in (length / 3, 2 * length / 3):
        ring = [at(s, r * math.cos(a), zc + r * math.sin(a)) for a in seam]
        for k in range(len(ring) - 1):
            _bar(p, ring[k], ring[k + 1], 0.05 * m, shade(FOOD_STEEL, 0.62))
    p.polygon(rim(length, r), shade(FOOD_STEEL, 0.74))
    p.polygon(rim(length + 0.12, 0.9 * r), shade(FOOD_STEEL, 0.82))
    p.polygon(rim(length + 0.22, 0.62 * r), shade(FOOD_STEEL, 0.9))
    for a in np.linspace(0, 2 * math.pi, 10, endpoint=False):
        bx, by = at(length + 0.1, 0.92 * r * math.cos(a), zc + 0.92 * r * math.sin(a))
        p.blob(bx - 0.07 * m, by - 0.07 * m, bx + 0.07 * m, by + 0.07 * m, (58, 60, 66))
    wheel = (176, 60, 44)
    p.polygon(rim(length + 0.32, 0.32), wheel)
    p.polygon(rim(length + 0.32, 0.22), shade(FOOD_STEEL, 0.88))
    for a in (math.pi / 4, 3 * math.pi / 4):
        _bar(
            p,
            at(length + 0.32, 0.3 * math.cos(a), zc + 0.3 * math.sin(a)),
            at(length + 0.32, -0.3 * math.cos(a), zc - 0.3 * math.sin(a)),
            0.07 * m,
            wheel,
        )
    vx, vy = at(length * 0.72, 0, zc + r)
    _bar(p, (vx, vy + 0.1 * m), (vx, vy - 0.7 * m), 0.14 * m, (70, 72, 80))
    p.blob(vx - 0.16 * m, vy - 0.9 * m, vx + 0.16 * m, vy - 0.6 * m, (70, 72, 80))
    gx, gy = at(length * 0.35, 0, zc + r)
    _bar(p, (gx, gy), (gx, gy - 0.4 * m), 0.08 * m, (70, 72, 80))
    p.ellipse(gx - 0.18 * m, gy - 0.75 * m, gx + 0.18 * m, gy - 0.4 * m, (236, 234, 226))
    if steam:
        _food_steam(p, vx + 0.1 * m, vy - 1.3 * m)


def _food_conveyor(p: Painter, x: float, y: float, run: float, top: float, foot: float) -> None:
    """A belt conveyor carrying a file of bright silver tins out of the hall: 0.9 m wide, from
    over (x, y) `run` m along the lit walls, sloping down from `top` m high to `foot` m, its steel
    frame on two braced trestles."""
    m = METRE

    def at(s: float, w: float, z: float) -> Point:
        px, py = _iso(x, y, s * m, w * m)
        return px, py - z * m

    def high(s: float) -> float:
        return top + (foot - top) * s / run

    for s in (run * 0.45, run * 0.88):
        for w in (0.4, -0.4):
            _bar(
                p,
                at(s, w, 0),
                at(s, w, high(s) - 0.3),
                0.14 * m,
                shade(FOOD_STEEL, 0.6 if w > 0 else 0.8),
            )
        _bar(p, at(s, -0.4, 0.3), at(s, 0.4, high(s) - 0.5), 0.07 * m, shade(FOOD_STEEL, 0.7))
    p.polygon(
        [at(0, -0.45, top), at(run, -0.45, foot), at(run, 0.45, foot), at(0, 0.45, top)],
        (66, 64, 66),
    )
    for s in np.arange(0.2, run, 0.6):
        _bar(p, at(s, -0.45, high(s)), at(s, 0.45, high(s)), 0.05 * m, (96, 94, 96))
    p.polygon(
        [
            at(0, -0.45, top),
            at(run, -0.45, foot),
            at(run, -0.45, foot - 0.3),
            at(0, -0.45, top - 0.3),
        ],
        FOOD_STEEL,
    )
    p.polygon(
        [
            at(run, -0.45, foot),
            at(run, 0.45, foot),
            at(run, 0.45, foot - 0.3),
            at(run, -0.45, foot - 0.3),
        ],
        shade(FOOD_STEEL, 0.74),
    )
    for s in np.arange(0.35, run - 0.15, 0.48):
        cx, cy = at(s, 0, high(s))
        half, tall = 0.17 * m, 0.36 * m
        p.polygon(
            [
                (cx - half, cy),
                (cx, cy + half / 2),
                (cx, cy + half / 2 - tall),
                (cx - half, cy - tall),
            ],
            FOOD_TIN,
        )
        p.polygon(
            [
                (cx, cy + half / 2),
                (cx + half, cy),
                (cx + half, cy - tall),
                (cx, cy + half / 2 - tall),
            ],
            shade(FOOD_TIN, 0.74),
        )
        p.blob(
            cx - half, cy - tall - half / 2, cx + half, cy - tall + half / 2, shade(FOOD_TIN, 1.12)
        )


def _food_tank(p: Painter, x: float, y: float) -> None:
    """The works' water tank on a braced steel trestle 8 m high standing on (x, y), made and
    painted like a giant tin can: a silver drum 3.2 m across and 3.4 m high with rolled rims and
    beaded ribs, a lid with its seam, and a bright red label banded in yellow with a ripe peach
    painted on it; a ladder climbs one leg."""
    m = METRE
    r, legs, tall = 1.6, 8.0, 3.4
    rs = r * math.sqrt(2) * m
    steel = (64, 70, 80)

    def at(u: float, v: float, z: float = 0) -> Point:
        px, py = _iso(x, y, u * m, v * m)
        return px, py - z * m

    def leg(u: float, v: float) -> tuple[Point, Point]:
        return at(u * 1.3, v * 1.3, 0), at(u, v, legs)

    feet = {(du, dv): leg(du, dv) for du in (-1.1, 1.1) for dv in (-1.1, 1.1)}
    order = sorted(feet, key=lambda k: k[0] - k[1])

    def brace(k0: tuple[float, float], k1: tuple[float, float], tone: float) -> None:
        (f0, t0), (f1, t1) = feet[k0], feet[k1]
        mid0, mid1 = _on_wall(f0, t0, 0.5, 0), _on_wall(f1, t1, 0.5, 0)
        for a, b in ((f0, mid1), (mid0, f1), (mid0, t1), (t0, mid1), (mid0, mid1)):
            _bar(p, a, b, 0.07 * m, shade(steel, tone))

    brace((-1.1, 1.1), (-1.1, -1.1), 1.2)
    brace((-1.1, 1.1), (1.1, 1.1), 1.0)
    for k in order[:-1]:
        _bar(p, *feet[k], 0.22 * m, steel)
    brace((-1.1, -1.1), (1.1, -1.1), 1.5)
    brace((1.1, -1.1), (1.1, 1.1), 1.25)
    foot, head = feet[order[-1]]
    _bar(p, foot, head, 0.22 * m, steel)
    for side in (-0.25, 0.25):
        _bar(
            p,
            (foot[0] + side * m - 0.2 * m, foot[1]),
            (head[0] + side * m - 0.2 * m, head[1]),
            0.05 * m,
            (40, 42, 48),
        )
    for k in range(1, 16):
        rx, ry = _on_wall(foot, head, k / 16, 0)
        _bar(p, (rx - 0.45 * m, ry), (rx + 0.05 * m, ry), 0.05 * m, (40, 42, 48))
    cx, cy = at(0, 0, legs)
    p.blob(
        cx - rs - 0.3 * m, cy - rs / 2 - 0.2 * m, cx + rs + 0.3 * m, cy + rs / 2 + 0.2 * m, steel
    )
    _turned(p, cx, cy, [(0, rs), (tall * m, rs)], FOOD_TIN)
    split = math.acos(1 / 3)

    def band(h0: float, h1: float, colour: RGB) -> None:
        for a0, a1, tone in ((split, math.pi, 1.0), (0, split, 0.76)):
            turns = np.linspace(a0, a1, 8)
            low = [(cx + rs * math.cos(a), cy - h0 * m + rs / 2 * math.sin(a)) for a in turns]
            high = [(cx + rs * math.cos(a), cy - h1 * m + rs / 2 * math.sin(a)) for a in turns]
            p.polygon(low + high[::-1], shade(colour, tone))

    band(0, 0.22, shade(FOOD_TIN, 0.8))
    band(0.85, 2.6, FOOD_LABELS[0])
    band(1.55, 1.85, FOOD_LABELS[1])
    for h in (0.5, 2.95):
        band(h, h + 0.08, shade(FOOD_TIN, 0.78))
    band(tall - 0.22, tall, shade(FOOD_TIN, 0.86))
    a = 2.3
    px, py = cx + rs * 0.78 * math.cos(a), cy - 1.72 * m + rs / 2 * 0.9 * math.sin(a)
    p.blob(px - 0.5 * m, py - 0.5 * m, px + 0.5 * m, py + 0.5 * m, FOOD_PEACH)
    p.blob(px - 0.42 * m, py - 0.42 * m, px + 0.05 * m, py + 0.1 * m, shade(FOOD_PEACH, 1.12))
    p.polygon(
        [
            (px + 0.05 * m, py - 0.45 * m),
            (px + 0.6 * m, py - 0.85 * m),
            (px + 0.35 * m, py - 0.35 * m),
        ],
        (70, 140, 54),
    )
    top = cy - tall * m
    p.ellipse(cx - rs, top - rs / 2, cx + rs, top + rs / 2, shade(FOOD_TIN, 1.1))
    p.ellipse(cx - rs * 0.86, top - rs * 0.43, cx + rs * 0.86, top + rs * 0.43, FOOD_TIN)


def food_plant() -> Sprite:
    """A Victorian cannery, putting up fruit and vegetables in tins: a hall 18 by 12 m with walls
    9 m high, rendered cream over a red-brick plinth between red-brick pilasters, tall
    round-headed windows framed in brick, a painted name board, and a silver corrugated-iron roof
    with a glazed clerestory along its ridge. A belt carries a file of bright tins out of its
    gable end down to cases waiting in the yard, beside a battery of four steel retorts on brick
    saddles cooking the sealed tins, their bolted doors facing out and steam blowing off their
    valves, and over them stands the works' water tank on a steel trestle, made and painted like
    a giant tin can with a red label and a peach on it. A farm cart stands at the cart door heaped
    with apples, peaches and cabbages, its horse in the shafts, baskets of cherries, peaches,
    tomatoes and carrots set down by the door; along the hall a loading dock is stacked with
    slatted cases of tins labelled red, yellow and green, and bundles of bright tinplate for
    making the tins wait in the yard."""
    m = METRE
    p = Painter(40, 27, (17, 16))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 11.5) * m, (v - 4) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    length, depth, high = 18 * m, 12 * m, 9 * m
    p.polygon(
        [at(0, -5.8), at(21, -5.8), at(28.5, -1.5), at(28.5, 14), at(18, 14), at(0, 4)],
        FOOD_SETTS,
    )
    foot = _footing(*at(9, 6), length, depth)
    _bed(p, foot)
    p.shadow(16, 6, 4)
    _food_tank(p, *at(25.6, 12.6))
    top = p.hall(*at(9, 6), length, depth, high, FOOD_RENDER, None)
    left, front, right, _ = top
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (high - 0.9 * m, high), shade(FOOD_BRICK, 0.92 * tone))
        _patch(p, a, b, (0, 1), (0, 0.55 * m), shade(FOOD_BRICK, tone))
        _patch(p, a, b, (0, 1), (0.55 * m, 0.75 * m), shade(FOOD_CREAM, tone))
        _patch(p, a, b, (0, 1), (2.85 * m, 3.05 * m), shade(FOOD_BRICK, tone))
        for t in (0.012, 0.988):
            _bar(
                p,
                _on_wall(a, b, t, 0.5 * m),
                _on_wall(a, b, t, high),
                0.45 * m,
                shade(FOOD_BRICK, tone),
            )
    _stains(p, left, front, high, (0.05, 0.29, 0.62, 0.96), shade(FOOD_RENDER, 0.72), 1.4 * m)
    bays = 6
    for k in range(1, bays):
        t = k / bays
        _bar(
            p,
            _on_wall(left, front, t, 3.05 * m),
            _on_wall(left, front, t, high - 0.9 * m),
            0.4 * m,
            FOOD_BRICK,
        )
    for k in (0, 2, 3):
        _food_window(p, left, front, length, (k + 0.5) / bays, 1.6 * m, (3.7 * m, 7.6 * m))
    for k in (4, 5):
        t = (k + 0.5) / bays
        _food_window(p, left, front, length, t, 1.4 * m, (3.7 * m, 5.0 * m))
        _patch(p, left, front, (t - 0.05, t + 0.05), (5.5 * m, high - 1.1 * m), FOOD_GREEN)
        for s in (t - 0.017, t + 0.017):
            _bar(
                p,
                _on_wall(left, front, s, 5.5 * m),
                _on_wall(left, front, s, high - 1.1 * m),
                0.05 * m,
                shade(FOOD_GREEN, 0.7),
            )
        _bar(
            p,
            _on_wall(left, front, t - 0.06, 5.35 * m),
            _on_wall(left, front, t + 0.06, 5.35 * m),
            0.3 * m,
            FOOD_BRICK,
        )
    t = 1.5 / bays
    p.polygon(_round_head(left, front, length, t, 3.4 * m, 4.1 * m, high), FOOD_BRICK)
    p.polygon(_round_head(left, front, length, t, 2.8 * m, 4.4 * m, high), DOORWAY)
    for s in (t - 0.095, t + 0.072):
        _patch(p, left, front, (s, s + 0.023), (5.6 * m, high - 0.05 * m), FOOD_GREEN)
    for k in range(3):
        _food_window(
            p,
            front,
            right,
            depth,
            (k + 0.5) / 3,
            1.6 * m,
            (3.7 * m, 7.6 * m),
            shade(MILL_GLASS, 0.85),
        )
    _patch(p, front, right, (0.08, 0.22), (2.5 * m, 4.0 * m), shade(FOOD_BRICK, 0.78))
    _patch(p, front, right, (0.1, 0.2), (2.75 * m, 4.0 * m), DOORWAY)
    _food_sign(p, left, front, length, 0.5, 0.45 * m)
    rise = 3.4 * m
    p.gable(top, rise, FOOD_IRON, FOOD_RENDER)
    for along in np.linspace(0.015, 0.985, 36):
        _bar(
            p,
            _on_roof(top, rise, along, 0),
            _on_roof(top, rise, along, 1),
            0.08 * m,
            shade(FOOD_IRON, 0.84),
        )
    _bar(p, _on_roof(top, rise, 0, 0), _on_roof(top, rise, 1, 0), 0.16 * m, shade(FOOD_IRON, 0.7))
    gx, gy = _on_wall(front, right, 0.5, -rise * 0.42)
    ring = [
        (gx + 0.9 * m * math.cos(a), gy - 0.45 * m * math.cos(a) - 0.9 * m * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)
    ]
    p.polygon(ring, shade(FOOD_BRICK, 0.78))
    p.polygon(
        [(gx + (px - gx) * 0.7, gy + (py - gy) * 0.7) for px, py in ring], shade(MILL_GLASS, 0.85)
    )
    ridge_back, ridge_front = _ridge(top, rise)
    rx, ry = _on_wall(ridge_back, ridge_front, 0.5, 0)
    lantern = p.hall(rx, ry + 0.75 * m, 14 * m, 2.6 * m, 2.1 * m, FOOD_CREAM, None)
    l_left, l_front, l_right, _ = lantern
    for a, b, tone in ((l_left, l_front, 1.0), (l_front, l_right, 0.78)):
        _patch(p, a, b, (0.02, 0.98), (0.25 * m, 1.3 * m), shade(NORTH_LIGHT, 0.8 * tone))
        for s in np.linspace(0.02, 0.98, 16 if tone == 1.0 else 3):
            _bar(
                p,
                _on_wall(a, b, s, 0.25 * m),
                _on_wall(a, b, s, 1.3 * m),
                0.07 * m,
                shade(FOOD_CREAM, tone),
            )
    p.gable(lantern, 0.9 * m, FOOD_IRON, FOOD_CREAM)
    for k, v in enumerate((10.6, 8.4, 6.2, 4.0)):
        _food_retort(p, *at(19.6, v), 5.5, 0.85, k in (1, 3))
    for v in (10.6, 8.4, 6.2, 4.0):
        _bar(p, up(21.2, v, 2.1), up(21.2, v, 2.75), 0.12 * m, (78, 80, 88))
    _bar(p, up(21.2, 3.2, 2.75), up(21.2, 11.6, 2.75), 0.2 * m, (78, 80, 88))
    _bar(p, up(21.2, 3.2, 2.75), up(18, 3.2, 4.2), 0.2 * m, (78, 80, 88))
    _food_conveyor(p, *at(18, 1.8), 4.8, 5.3, 1.6)
    _food_cases(p, *at(23.4, 1.3), (2, 1), 1)
    _food_tinplate(p, *at(20.4, -2.4), 3)
    _food_tinplate(p, *at(23.2, -1.8), 2)
    for k in (2, 1, 0):
        p.hall(
            *at(10.5 - 0.35 * (k + 0.5), -1.2),
            0.35 * m,
            2 * m,
            0.27 * (3 - k) * m,
            shade(TIMBER, 0.8),
            shade(TIMBER, 1.1),
        )
    dock = p.hall(
        *at(14.25, -1.2), 7.5 * m, 2.4 * m, 1.1 * m, shade(FOOD_BRICK, 0.9), shade(TIMBER, 1.15)
    )
    for s in np.linspace(0.05, 0.95, 10):
        _bar(
            p,
            _on_wall(dock[0], dock[3], s, 0),
            _on_wall(dock[1], dock[2], s, 0),
            0.05 * m,
            shade(TIMBER, 0.85),
        )
    _food_cases(p, *up(11.2, -1.4, 1.1), (3, 2, 1))
    _food_cases(p, *up(15.2, -1.0, 1.1), (2, 1), 1)
    _food_cases(p, *up(17.0, -1.6, 1.1), (1,), 2)

    def produce(on_bed: Callable[[float, float], Point]) -> None:
        for z, spots, w_spots in (
            (0.18, np.arange(-1.4, 1.45, 0.4), (0.52, 0.17, -0.18, -0.53)),
            (0.42, np.arange(-1.0, 1.05, 0.4), (0.3, -0.05, -0.4)),
        ):
            for w in w_spots:
                for k, s in enumerate(spots):
                    cx, cy = on_bed(s, w)
                    cy -= z * m
                    if s > 0.5:
                        size, colour = 0.3, shade(FOOD_CABBAGE, 1.0 if k % 2 else 0.86)
                    elif s > -0.5:
                        size, colour = 0.22, shade(FOOD_PEACH, 1.0 if k % 2 else 0.9)
                    else:
                        size, colour = (
                            0.21,
                            FOOD_APPLE if (k + round(w * 3)) % 3 else FOOD_GREEN_APPLE,
                        )
                    p.ellipse(
                        cx - size * m, cy - size * m, cx + size * m, cy + size * 0.8 * m, colour
                    )
                    p.blob(
                        cx - size * 0.6 * m,
                        cy - size * 0.7 * m,
                        cx,
                        cy - size * 0.15 * m,
                        shade(colour, 1.22),
                    )

    for u, v, load in (
        (2.5, -0.8, FOOD_CHERRY),
        (1.6, -1.3, FOOD_PEACH),
        (6.6, -1.1, FOOD_TOMATO),
        (2.8, -1.9, FOOD_CARROT),
    ):
        _food_basket(p, *at(u, v), load)
    _food_cart(p, *at(4.4, -3.6), 3.4, 0.5, FOOD_SETTS, produce)
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


WINE_WALL: RGB = (238, 224, 192)
WINE_ROOF: RGB = (118, 40, 66)
WINE: RGB = (124, 30, 52)
GRAPE: RGB = (92, 44, 104)
VINE: RGB = (78, 130, 50)
VINE_POST: RGB = (110, 84, 58)
DUG: RGB = (124, 100, 70)
TURF: RGB = (104, 146, 70)


def _cask_rack(p: Painter, x: float, y: float, count: int, tiers: int) -> None:
    """Oak casks on their sides with their heads to the lit side, `count` of them side by side
    along the lit walls from (x, y) on each of `tiers` tiers of a timber rack: two rails under
    every tier on posts at either end, the casks of each tier resting on them."""
    m = METRE
    gap, step = 0.9, 1.0
    wood = shade(TIMBER, 0.85)
    ends = (-0.6, (count - 1) * gap + 0.6)
    p.soft(
        [(*_cask_at(x, y, ends[0], 0.6), *_cask_at(x, y, ends[1] + 0.8, -0.8))],
        (34, 30, 24),
        0.4,
        0.2,
    )
    for u in ends:
        _bar(
            p,
            _cask_at(x, y, u, 0.45),
            _cask_at(x, y, u, 0.45, tiers * step + 0.15),
            0.14 * m,
            shade(wood, 0.8),
        )
    for tier in range(tiers):
        z = tier * step + 0.12
        for v in (0.32, -0.32):
            _bar(p, _cask_at(x, y, ends[0], v, z), _cask_at(x, y, ends[1], v, z), 0.12 * m, wood)
        for k in range(count):
            _cask(p, *_cask_at(x, y, k * gap, 0, z + 0.04))
    for u in ends:
        _bar(
            p,
            _cask_at(x, y, u, -0.45),
            _cask_at(x, y, u, -0.45, tiers * step + 0.15),
            0.14 * m,
            wood,
        )


def _wine_press(p: Painter, x: float, y: float) -> None:
    """A wooden basket press standing on (x, y): a square tray on a low trestle with a spout
    running red into a little tub, a round basket of oak slats bound by two iron hoops and full
    of crushed purple grapes, and an iron screw rising from it with a lever across its top."""
    m = METRE
    tray = p.hall(x, y, 1.9 * m, 1.9 * m, 0.55 * m, shade(TIMBER, 0.9), shade(TIMBER, 0.7))
    left, front, right, _ = tray
    spout = _on_wall(left, front, 0.75, 0.2 * m)
    _bar(p, spout, (spout[0] - 0.3 * m, spout[1] + 0.25 * m), 0.14 * m, shade(TIMBER, 0.7))
    tub = (spout[0] - 0.4 * m, spout[1] + 0.9 * m)
    _turned(p, *tub, [(0, 0.3 * m), (0.35 * m, 0.33 * m)], OAK, WINE)
    _bar(p, (spout[0] - 0.36 * m, spout[1] + 0.3 * m), (tub[0], tub[1] - 0.35 * m), 0.06 * m, WINE)
    cx, cy = (left[0] + right[0]) / 2, (left[1] + right[1]) / 2
    r = 0.62 * m
    _turned(p, cx, cy, [(0, r), (1.0 * m, r)], shade(OAK, 1.12), GRAPE)
    for a in np.linspace(0.2, math.pi - 0.2, 7):
        sx, sy = cx + r * math.cos(a), cy + r / 2 * math.sin(a)
        _bar(p, (sx, sy - 0.04 * m), (sx, sy - 0.98 * m), 0.06 * m, shade(OAK, 0.55))
    for drop in (0.2, 0.8):
        hoop = [
            (cx + r * math.cos(a), cy - drop * m + r / 2 * math.sin(a))
            for a in np.linspace(0, math.pi, 9)
        ]
        for a, b in itertools.pairwise(hoop):
            _bar(p, a, b, 0.08 * m, CASK_HOOP)
    p.blob(cx - 0.3 * m, cy - 1.12 * m, cx + 0.1 * m, cy - 0.94 * m, shade(GRAPE, 1.3))
    _bar(p, (cx, cy - 1.0 * m), (cx, cy - 2.2 * m), 0.14 * m, (58, 56, 58))
    _bar(
        p, (cx - 0.9 * m, cy - 2.0 * m), (cx + 0.9 * m, cy - 2.3 * m), 0.12 * m, shade(TIMBER, 0.8)
    )


def _wine_cart(p: Painter, x: float, y: float) -> None:
    """A two-wheeled farm cart standing along the lit walls on (x, y), its shafts down on the
    ground ahead of it, its slatted bed heaped with bunches of purple grapes."""
    m = METRE
    p.soft([(*_cask_at(x, y, -1.4, 0.8), *_cask_at(x, y, 2.8, -0.9))], (34, 30, 24), 0.4, 0.3)
    _brew_wheel(p, x, y, 0, 0.75, 0.62, TIMBER)
    for v in (0.45, -0.45):
        _bar(
            p,
            _cask_at(x, y, 1.1, v, 0.85),
            _cask_at(x, y, 2.6, v, 0.0),
            0.1 * m,
            shade(TIMBER, 0.75),
        )
    bed = p.hall(*_cask_at(x, y, 0, 0, 0.7), 2.4 * m, 1.4 * m, 0.5 * m, TIMBER, shade(GRAPE, 0.7))
    left, front, right, back = bed
    _band(p, left, front, 0.25 * m, shade(TIMBER, 0.7))
    for k, (t, s) in enumerate(
        (
            (0.15, 0.3),
            (0.35, 0.6),
            (0.55, 0.3),
            (0.75, 0.65),
            (0.3, 0.85),
            (0.6, 0.9),
            (0.85, 0.4),
            (0.45, 0.15),
        )
    ):
        gx, gy = _on_wall(_on_wall(left, front, t, 0), _on_wall(back, right, t, 0), s, 0)
        tone = shade(GRAPE, 1.2 if k % 3 == 0 else (0.9 if k % 3 == 1 else 1.05))
        p.blob(gx - 0.35 * m, gy - 0.45 * m, gx + 0.35 * m, gy + 0.1 * m, tone)
        p.blob(gx - 0.2 * m, gy - 0.4 * m, gx, gy - 0.25 * m, shade(GRAPE, 1.5))
    _brew_wheel(p, x, y, 0, -0.85, 0.62, TIMBER)


def _wine_vines(p: Painter, x: float, y: float, posts: int) -> None:
    """A row of vines trained along wires on posts 2 m apart from (x, y) along the lit walls,
    rooted in a strip of dug earth: leafy green cordons along the wires and purple bunches of
    grapes hanging under them."""
    m = METRE
    span = 2 * (posts - 1)
    p.polygon(
        [
            _cask_at(x, y, -0.6, 0.5),
            _cask_at(x, y, span + 0.6, 0.5),
            _cask_at(x, y, span + 0.6, -0.5),
            _cask_at(x, y, -0.6, -0.5),
        ],
        DUG,
    )
    for k in range(posts):
        _bar(p, _cask_at(x, y, 2 * k, 0), _cask_at(x, y, 2 * k, 0, 1.7), 0.12 * m, VINE_POST)
    _bar(p, _cask_at(x, y, 0, 0, 1.55), _cask_at(x, y, span, 0, 1.55), 0.04 * m, (90, 90, 90))
    for k in range(posts - 1):
        for j in range(4):
            u = 2 * k + 0.3 + 0.48 * j
            lx, ly = _cask_at(x, y, u, 0, 1.25 + 0.15 * (j % 2))
            tone = shade(VINE, 1.15 if j % 2 else 0.9)
            p.blob(lx - 0.42 * m, ly - 0.38 * m, lx + 0.42 * m, ly + 0.3 * m, tone)
        for u in (2 * k + 0.6, 2 * k + 1.5):
            gx, gy = _cask_at(x, y, u, -0.15, 0.85)
            p.polygon(
                [
                    (gx - 0.18 * m, gy - 0.25 * m),
                    (gx + 0.18 * m, gy - 0.25 * m),
                    (gx, gy + 0.25 * m),
                ],
                GRAPE,
            )
            p.blob(gx - 0.12 * m, gy - 0.22 * m, gx, gy - 0.1 * m, shade(GRAPE, 1.5))
        stem = _cask_at(x, y, 2 * k + 1, 0)
        _bar(p, stem, (stem[0], stem[1] - 1.1 * m), 0.1 * m, shade(VINE_POST, 0.8))


def _wine_window(
    p: Painter, a: Point, b: Point, length: float, t: float, wide: float, drop: tuple[float, float]
) -> None:
    """A window `wide` across about `t` along the wall under the edge from `a` to `b`, which is
    `length` units long, from `drop[0]` to `drop[1]` down it, between two wine-red shutters
    folded back against the wall."""
    m = METRE
    half = wide / 2 / length
    _patch(p, a, b, (t - half, t + half), drop, (54, 50, 60))
    _bar(p, _on_wall(a, b, t, drop[0]), _on_wall(a, b, t, drop[1]), 0.07 * m, WINE_WALL)
    for side in (-1, 1):
        s0, s1 = sorted((t + side * half * 1.05, t + side * half * 2))
        _patch(p, a, b, (s0, s1), (drop[0] - 0.05 * m, drop[1] + 0.05 * m), WINE)
        mid = (drop[0] + drop[1]) / 2
        _bar(p, _on_wall(a, b, s0, mid), _on_wall(a, b, s1, mid), 0.05 * m, shade(WINE, 0.65))
    _bar(
        p,
        _on_wall(a, b, t - half * 1.1, drop[1] + 0.1 * m),
        _on_wall(a, b, t + half * 1.1, drop[1] + 0.1 * m),
        0.15 * m,
        shade(WINE_WALL, 1.05),
    )


def winery() -> Sprite:
    """A cream-rendered winery 15 by 7 m under a wine-red tiled roof, its windows between
    wine-red shutters, a wine-red arched door with a vine trained over it and a little chimney;
    a cellar vaulted into a grassy bank at its end behind a stone front with a wine-red arched
    door; oak casks lying on a timber rack before it; a basket press stained with spilt wine and
    a cart heaped with purple grapes in the yard, and three rows of vines hung with grapes on
    their wires over strips of dug earth."""
    m = METRE
    p = Painter(46, 34, (22, 21))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 4) * m, (v - 1.75) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon([at(0, -6.5), at(14, -6.5), at(14, 0), at(0, 0)], (186, 168, 132))
    wx, wy = at(4.6, -3.3)
    p.blob(wx - 1.6 * m, wy - 0.5 * m, wx + 1.4 * m, wy + 0.8 * m, (126, 58, 74), 170)
    _bed(p, _footing(*at(7.5, 5.5), 15 * m, 7 * m), margin=0.8)
    p.shadow(12, 3.5, 3)
    length, depth, high = 15 * m, 7 * m, 6 * m
    top = p.hall(*at(7.5, 5.5), length, depth, high, WINE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, high, (196, 184, 160), 0.5 * m)
    for t in (0.12, 0.3, 0.82):
        _wine_window(p, left, front, length, t, 1.1 * m, (2.0 * m, 3.8 * m))
    _wine_window(p, front, right, depth, 0.5, 1.1 * m, (2.0 * m, 3.8 * m))
    t = 0.56
    p.polygon(_round_head(left, front, length, t, 2.2 * m, 2.6 * m, high), shade(WINE_WALL, 0.8))
    p.polygon(_round_head(left, front, length, t, 1.8 * m, 2.8 * m, high), WINE)
    _bar(
        p,
        _on_wall(left, front, t, 2.8 * m),
        _on_wall(left, front, t, high),
        0.06 * m,
        shade(WINE, 0.6),
    )
    for k, s in enumerate(np.linspace(t - 0.09, t + 0.09, 7)):
        vx, vy = _on_wall(
            left, front, s, 2.2 * m - 0.4 * m * math.sin(math.pi * (s - t + 0.09) / 0.18)
        )
        p.blob(
            vx - 0.5 * m,
            vy - 0.35 * m,
            vx + 0.5 * m,
            vy + 0.35 * m,
            shade(VINE, 1.1 if k % 2 else 0.9),
        )
    for s in (t - 0.07, t + 0.05):
        gx, gy = _on_wall(left, front, s, 2.5 * m)
        p.polygon([(gx - 0.16 * m, gy), (gx + 0.16 * m, gy), (gx, gy + 0.45 * m)], GRAPE)
    for s in (t - 0.1, t + 0.1):
        _bar(
            p,
            _on_wall(left, front, s, 2.2 * m),
            _on_wall(left, front, s, high),
            0.12 * m,
            shade(VINE_POST, 0.8),
        )
    p.gable(top, 3.5 * m, WINE_ROOF, WINE_WALL)
    _roof_rows(p, top, 3.5 * m, WINE_ROOF, (0.25, 0.5, 0.75))
    cx, cy = _on_roof(top, 3.5 * m, 0.72, 0.85)
    p.hall(cx, cy + 0.3 * m, 0.9 * m, 0.9 * m, 1.8 * m, shade(WINE_WALL, 0.92), (80, 70, 66))

    bank = shade(TURF, 0.86)
    cellar = p.hall(*at(18.5, 4), 6 * m, 6 * m, 2 * m, bank, None)
    _, c_front, c_right, _ = cellar
    _vault(p, cellar, 2 * m, TURF, STONE, (), shade(TURF, 0.95))
    _patch(p, c_front, c_right, (0, 1), (0, 2 * m), shade(STONE, 0.86))
    for k in range(4):
        _band(p, c_front, c_right, (0.5 * k) * m, shade(STONE, 0.7))
    p.polygon(
        _round_head(c_front, c_right, 6 * m, 0.5, 2.6 * m, -0.6 * m, 2 * m), shade(STONE, 1.1)
    )
    p.polygon(_round_head(c_front, c_right, 6 * m, 0.5, 2 * m, -0.3 * m, 2 * m), WINE)
    _bar(
        p,
        _on_wall(c_front, c_right, 0.5, -0.3 * m),
        _on_wall(c_front, c_right, 0.5, 2 * m),
        0.06 * m,
        shade(WINE, 0.6),
    )

    _cask_rack(p, *at(9.2, 0.9), 4, 2)
    _wine_press(p, *at(4.2, -2.6))
    _wine_cart(p, *at(7.4, -4.8))
    for v in (-1.4, -3.9, -6.4):
        _wine_vines(p, *at(-9, v), 5)
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


BRICK_KILN: RGB = (166, 82, 56)
BRICK_FIRED: RGB = (200, 88, 56)
BRICK_ORANGE: RGB = (216, 124, 72)
BRICK_GREEN: RGB = (220, 202, 162)
BRICK_CLAY: RGB = (190, 160, 114)
BRICK_PANTILE: RGB = (196, 98, 58)
BRICK_DUST: RGB = (158, 112, 86)
BRICK_GLOW: RGB = (246, 150, 54)


def _brick_lattice(
    p: Painter, a: Point, b: Point, cells: int, courses: int, tall: float, gap: RGB
) -> None:
    """The open lattice of a hack of green bricks set to dry on the wall under the edge from `a`
    to `b`, `tall` units high: `courses` courses of `cells` bricks each, every other one left out
    so the air goes through, the gaps in `gap`."""
    for c in range(courses):
        for j in range(cells):
            if (c + j) % 2:
                _patch(
                    p,
                    a,
                    b,
                    ((j + 0.12) / cells, (j + 0.88) / cells),
                    (tall * (c + 0.15) / courses, tall * (c + 0.85) / courses),
                    gap,
                )


def _brick_hack(p: Painter, x: float, y: float, long: float, wide: float, tall: float) -> None:
    """A hack of green bricks drying on (x, y): `long` m along the lit walls, `wide` m across and
    `tall` m high, of cream unfired clay stacked in an open lattice that shows on both its faces
    and its top."""
    m = METRE
    left, front, right, back = p.hall(
        x, y, long * m, wide * m, tall * m, BRICK_GREEN, shade(BRICK_GREEN, 1.06)
    )
    courses = max(2, round(tall / 0.3))
    _brick_lattice(p, left, front, round(long / 0.3), courses, tall * m, (118, 100, 80))
    _brick_lattice(p, front, right, round(wide / 0.3), courses, tall * m, (92, 78, 64))
    for k in range(1, round(long / 0.3)):
        t = k / round(long / 0.3)
        _bar(
            p,
            _on_wall(left, front, t, 0),
            _on_wall(back, right, t, 0),
            0.04 * m,
            shade(BRICK_GREEN, 0.8),
        )


def _brick_pallet(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A neat cube of a thousand-odd fired bricks, 1.2 by 1 m and 1.1 m high, stacked on a
    timber pallet on (x, y), its courses and staggered joints showing on its faces."""
    m = METRE
    p.hall(x, y, 1.3 * m, 1.1 * m, 0.15 * m, shade(TIMBER, 0.7), shade(TIMBER, 0.85))
    left, front, right, back = p.hall(
        x, y - 0.15 * m, 1.2 * m, 1 * m, 1.1 * m, colour, shade(colour, 1.12)
    )
    for a, b, tone in ((left, front, 0.72), (front, right, 0.6)):
        for c in range(1, 5):
            drop = 1.1 * m * c / 5
            _bar(p, _on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop), 0.05 * m, shade(colour, tone))
        for c in range(5):
            for t in (0.25, 0.75) if c % 2 else (0.5,):
                _bar(
                    p,
                    _on_wall(a, b, t, 1.1 * m * c / 5),
                    _on_wall(a, b, t, 1.1 * m * (c + 1) / 5),
                    0.04 * m,
                    shade(colour, tone),
                )
    _bar(
        p,
        _on_wall(left, back, 0.5, 0),
        _on_wall(front, right, 0.5, 0),
        0.04 * m,
        shade(colour, 0.8),
    )


def _brick_tub(p: Painter, x: float, y: float) -> None:
    """A narrow-gauge tipping tub on (x, y) on the track along the lit walls: an iron V-skip 1.3 m
    long on a low frame and four small wheels, heaped with raw clay."""
    m = METRE

    def at(s: float, z: float, w: float = 0) -> Point:
        px, py = _iso(x, y, s * m, w * m)
        return px, py - z * m

    iron = (92, 92, 98)
    for s in (-0.4, 0.4):
        p.polygon(
            [
                at(s + 0.18 * math.cos(a), 0.2 + 0.18 * math.sin(a), 0.35)
                for a in np.linspace(0, 2 * math.pi, 10)
            ],
            (50, 48, 50),
        )
    p.polygon(
        [at(-0.65, 0.35, -0.3), at(0.65, 0.35, -0.3), at(0.65, 0.35, 0.3), at(-0.65, 0.35, 0.3)],
        (60, 58, 62),
    )
    body = [at(-0.7, 1.0, -0.5), at(0.7, 1.0, -0.5), at(0.5, 0.4, -0.1), at(-0.5, 0.4, -0.1)]
    p.polygon(
        [at(0.7, 1.0, -0.5), at(0.7, 1.0, 0.5), at(0.5, 0.4, 0.1), at(0.5, 0.4, -0.1)],
        shade(iron, 0.7),
    )
    p.polygon(body, iron)
    cx, cy = at(0, 1.0, 0)
    p.blob(cx - 0.75 * m, cy - 0.35 * m, cx + 0.75 * m, cy + 0.3 * m, BRICK_CLAY)
    p.blob(cx - 0.4 * m, cy - 0.3 * m, cx + 0.2 * m, cy, shade(BRICK_CLAY, 1.12))
    for s in (-0.4, 0.4):
        p.polygon(
            [
                at(s + 0.18 * math.cos(a), 0.2 + 0.18 * math.sin(a), -0.35)
                for a in np.linspace(0, 2 * math.pi, 10)
            ],
            (50, 48, 50),
        )


def brickworks() -> Sprite:
    """A Victorian brickworks round a Hoffmann ring kiln: a long low oval of sooty brick 26 by
    11 m, its battered walls pierced all round by arched wickets, some bricked up for the firing
    and one glowing, the feeding floor on its crown under a pantiled hipped roof on open posts,
    and its tall round chimney rising from the middle. Before it an open drying shed shelters a
    hack of cream green bricks stacked in an open lattice, with two more hacks out in the yard;
    beside it a clay pit is dug into the ground, a puddle on its floor, with tipping tubs on a
    narrow-gauge track at its edge, and in a yard red with brick dust stand neat pallets of fired
    red bricks by a small brick office."""
    m = METRE
    p = Painter(41, 48, (18, 35))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10.5) * m, (v - 0.5) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon(
        [at(-10, -1), at(-3, -8.5), at(33.5, -8.5), at(33.5, -0.5), at(24, 9), at(-10, 10)],
        BRICK_DUST,
    )
    u0, u1, v0, v1, d = -9.5, -1.5, -1.0, 9.0, 2.2
    p.polygon([at(u0, v0), at(u0, v1), at(u0 + d, v1 - d), at(u0 + d, v0)], shade(BRICK_CLAY, 0.62))
    p.polygon([at(u0, v1), at(u1, v1), at(u1, v1 - d), at(u0 + d, v1 - d)], shade(BRICK_CLAY, 0.92))
    for q, tone in ((0.3, 0.8), (0.55, 1.04), (0.8, 0.78)):
        _bar(p, at(u0 + q * d, v1 - q * d), at(u1, v1 - q * d), 0.12 * m, shade(BRICK_CLAY, tone))
    _bar(p, at(u0, v1), at(u1, v1), 0.25 * m, (88, 104, 56))
    _bar(p, at(u0, v0), at(u0, v1), 0.25 * m, (88, 104, 56))
    p.polygon(
        [at(u0 + d, v0), at(u1, v0), at(u1, v1 - d), at(u0 + d, v1 - d)], shade(BRICK_CLAY, 0.8)
    )
    px, py = at(-4.5, 3)
    p.blob(px - 2.2 * m, py - 0.8 * m, px + 2.2 * m, py + 0.8 * m, PUDDLE)
    p.blob(px - 1.4 * m, py - 0.5 * m, px - 0.2 * m, py - 0.1 * m, shade(PUDDLE, 1.3))
    for w in (-0.3, 0.3):
        _bar(p, at(-1.5, -2 + w), at(5, -2 + w), 0.08 * m, (70, 66, 64))
    _bed(p, _footing(*at(13, 5.5), 26 * m, 11 * m))
    p.shadow(16, 6, 5)
    high, r = 3.6 * m, 5.5
    f = r * math.sqrt(2) * m
    ends = (at(5.5, 5.5), at(20.5, 5.5))
    batter = [(0, f + 0.35 * m), (0.9 * m, f), (high, f)]
    _turned(p, *ends[0], batter, BRICK_KILN)
    b_left, b_front, _, _ = p.hall(*at(13, 5.5), 15 * m, 11 * m, high, BRICK_KILN, None)
    _turned(p, *ends[1], batter, BRICK_KILN)
    _patch(p, b_left, b_front, (0, 1), (high - 0.9 * m, high), shade(BRICK_KILN, 0.84))
    lid = (110, 92, 84)
    for ex, ey in ends:
        p.blob(ex - f, ey - high - f / 2, ex + f, ey - high + f / 2, lid)
    p.polygon([up(5.5, 0, 3.6), up(20.5, 0, 3.6), up(20.5, 11, 3.6), up(5.5, 11, 3.6)], lid)

    def wicket(x: float, y: float, kind: int) -> None:
        def arch(half: float, crown: float) -> list[Point]:
            turns = np.linspace(0, math.pi, 7)
            return [
                (x - half, y),
                (x + half, y),
                *[(x + half * math.cos(a), y - crown + half - half * math.sin(a)) for a in turns],
            ]

        _bar(p, (x, y - 2.0 * m), (x + 0.2 * m, y - high + 0.1 * m), 0.5 * m, (92, 54, 44))
        p.polygon(arch(0.78 * m, 2.0 * m), shade(BRICK_KILN, 1.12))
        fill = (shade(BRICK_KILN, 0.96), DOORWAY, BRICK_GLOW)[kind]
        p.polygon(arch(0.58 * m, 1.8 * m), fill)
        if kind == 0:
            for drop in (0.35, 0.75, 1.15):
                _bar(
                    p,
                    (x - 0.55 * m, y - drop * m),
                    (x + 0.55 * m, y - drop * m),
                    0.05 * m,
                    shade(fill, 0.75),
                )
        if kind == 2:
            p.blob(x - 0.32 * m, y - 1.3 * m, x + 0.32 * m, y - 0.1 * m, (255, 214, 120))

    for k, kind in enumerate((1, 0, 1, 2, 0, 1)):
        wicket(*_on_wall(b_left, b_front, (k + 0.5) / 6, high), kind)
    for (ex, ey), turns in ((ends[0], (2.75, 2.45)), (ends[1], (1.95, 1.2, 0.5))):
        for kind, a in zip((0, 1, 0), turns, strict=False):
            wicket(ex + f * math.cos(a), ey + f / 2 * math.sin(a), kind)
    store = p.hall(*up(13, 5.5, 3.6), 19 * m, 7.6 * m, 1.3 * m, (50, 40, 36), None)
    for a, b, posts in ((store[0], store[1], 9), (store[1], store[2], 3)):
        for k in range(posts + 1):
            px, py = _on_wall(a, b, k / posts, 1.3 * m)
            _bar(p, (px, py), (px, py - 1.3 * m), 0.22 * m, shade(TIMBER, 0.8))
    tile = BRICK_PANTILE
    ru0, ru1, rv0, rv1, eave, rise = 2.9, 23.1, 1.3, 9.7, 4.9, 2.8
    run = (rv1 - rv0) / 2
    r_left, r_front = up(ru0, rv0, eave), up(ru1, rv0, eave)
    r_right, r_back = up(ru1, rv1, eave), up(ru0, rv1, eave)
    ridge_back, ridge_front = (
        up(ru0 + run, rv0 + run, eave + rise),
        up(ru1 - run, rv0 + run, eave + rise),
    )
    p.polygon([r_back, r_right, ridge_front, ridge_back], shade(tile, 0.68), shade(tile, 0.45))
    p.polygon([r_left, r_back, ridge_back], shade(tile, 0.9), shade(tile, 0.55))
    p.polygon([r_front, r_right, ridge_front], shade(tile, 0.74), shade(tile, 0.5))
    p.polygon([r_left, r_front, ridge_front, ridge_back], tile, shade(tile, 0.6))
    for u in np.linspace(ru0 + 0.5, ru1 - 0.5, 40):
        reach = min(u - ru0, ru1 - u, run)
        _bar(
            p,
            up(u, rv0, eave),
            up(u, rv0 + reach, eave + rise * reach / run),
            0.06 * m,
            shade(tile, 0.86),
        )
    _bar(p, r_left, r_front, 0.14 * m, shade(tile, 0.7))
    _bar(p, ridge_back, ridge_front, 0.22 * m, shade(tile, 1.2))
    _tall_chimney(p, *up(13, 5.5, eave + rise - 1.0), 44 * m)
    _brick_tub(p, *at(0.2, -2))
    _brick_tub(p, *at(2.1, -2))
    shed = _footing(*up(6, -5.5, 2.8), 17 * m, 4 * m)
    s_left, s_front, s_right, s_back = shed
    for k in range(7):
        px, py = _on_wall(s_back, s_right, k / 6, 2.8 * m)
        _bar(p, (px, py), (px, py - 2.8 * m), 0.2 * m, shade(TIMBER, 0.6))
    _brick_hack(p, *at(6, -5.5), 15, 1.6, 1.5)
    for k in range(7):
        px, py = _on_wall(s_left, s_front, k / 6, 2.8 * m)
        _bar(p, (px, py), (px, py - 2.8 * m), 0.2 * m, shade(TIMBER, 0.75))
    p.gable(shed, 1.2 * m, shade(tile, 0.8), TIMBER)
    _roof_rows(p, shed, 1.2 * m, shade(tile, 0.8))
    for u, v in ((16.6, -6), (18.9, -6)):
        _brick_hack(p, *at(u, v), 2, 1.4, 1.3)
    office = p.hall(*at(23.5, -6.5), 5 * m, 4 * m, 4 * m, BRICK_ORANGE, None)
    o_left, o_front, o_right, _ = office
    _plinth(p, office, 4 * m, BRICK_ORANGE, 0.5 * m)
    _bands(p, o_left, o_front, o_right, 0.3 * m, BRICK_GREEN, shade(BRICK_GREEN, 0.8))
    _food_window(p, o_left, o_front, 5 * m, 0.25, 1.1 * m, (1.2 * m, 2.9 * m))
    _patch(p, o_left, o_front, (0.58, 0.8), (1.3 * m, 4 * m), DOOR)
    _patch(p, o_left, o_front, (0.56, 0.82), (1.05 * m, 1.3 * m), BRICK_GREEN)
    _food_window(
        p, o_front, o_right, 4 * m, 0.5, 1.1 * m, (1.2 * m, 2.9 * m), shade(MILL_GLASS, 0.85)
    )
    p.gable(office, 2.2 * m, tile, BRICK_ORANGE)
    _roof_rows(p, office, 2.2 * m, tile)
    cx, cy = _on_roof(office, 2.2 * m, 0.2, 0.8)
    p.hall(cx, cy, 0.6 * m, 0.6 * m, 1.4 * m, BRICK_FIRED, (60, 52, 50))
    pallets = [
        (28, -1.5, BRICK_FIRED),
        (30, -1.5, BRICK_ORANGE),
        (32, -1.5, BRICK_FIRED),
        (28, -4, BRICK_ORANGE),
        (30, -4, BRICK_FIRED),
        (32, -4, BRICK_FIRED),
        (28, -6.5, BRICK_FIRED),
        (30, -6.5, BRICK_FIRED),
    ]
    for u, v, colour in sorted(pallets, key=lambda e: e[0] - e[1]):
        _brick_pallet(p, *at(u, v), colour)
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
GLASS_YARD: RGB = (150, 140, 126)
GLASS_SLATE: RGB = (78, 82, 94)
GLASS_COBALT: RGB = (46, 86, 178)
GLASS_SEA: RGB = (72, 168, 148)
GLASS_RUBY: RGB = (180, 42, 66)
GLASS_WICKER: RGB = (180, 140, 84)
GLASS_GLINT: RGB = (244, 252, 252)


def _glass_gathers(p: Painter, x: float, y: float) -> None:
    """Two blowpipes drawn out of a furnace mouth whose sill is at (x, y), each with a gather
    of molten glass glowing white-hot on its end."""
    m = METRE
    for k, (dx, dy) in enumerate(((-2.8, 0.6), (-2.3, -0.5))):
        end = (x + dx * m, y + dy * m)
        _bar(p, (x - 0.2 * m, y - (1.4 - 0.5 * k) * m), end, 0.1 * m, (50, 46, 46))
        p.blob(
            end[0] - 0.32 * m, end[1] - 0.26 * m, end[0] + 0.32 * m, end[1] + 0.26 * m, GLASS_GLOW
        )
        p.blob(end[0] - 0.18 * m, end[1] - 0.16 * m, end[0] + 0.12 * m, end[1] + 0.1 * m, GLASS_HOT)


def _glass_roof(p: Painter, top: Top, rise: float, ribs: int) -> None:
    """A pitched roof glazed in aqua panes on a block's top, `rise` high, its glazing bars
    running down it `ribs` times along, a purlin across them and long glints of light."""
    m = METRE
    p.gable(top, rise, shade(GLASS_AQUA, 0.9), GLASS_BRICK)
    for k in range(1, ribs):
        t = k / ribs
        _bar(
            p, _on_roof(top, rise, t, 0), _on_roof(top, rise, t, 1), 0.08 * m, shade(GLASS_SEA, 0.7)
        )
    _bar(
        p, _on_roof(top, rise, 0, 0.5), _on_roof(top, rise, 1, 0.5), 0.08 * m, shade(GLASS_SEA, 0.7)
    )
    for t0, t1, up in ((0.05, 0.4, 0.7), (0.5, 0.75, 0.3), (0.12, 0.3, 0.25)):
        _bar(
            p,
            _on_roof(top, rise, t0, up),
            _on_roof(top, rise, t1, up + 0.15),
            0.12 * m,
            GLASS_GLINT,
        )


def _glass_cone(p: Painter, x: float, y: float, r: float, tall: float) -> None:
    """An English glass cone of brick `r` across its foot and `tall` high standing on (x, y):
    its walls curving in as they rise, bound with iron hoops, sooted black towards its open top
    and smoking, the furnace mouth glowing in a deep arch at its foot with gathers of molten glass
    drawn out of it on blowpipes, and its light spilling over the ground before it."""
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
    _glass_gathers(p, mx, my)
    for dx, dy in ((-2, 0.9), (-0.6, 1.4), (-2.8, 1.8)):
        p.blob(
            mx + (dx - 0.12) * m,
            my + (dy - 0.08) * m,
            mx + (dx + 0.12) * m,
            my + (dy + 0.08) * m,
            GLASS_GLOW,
        )


def _glass_house(p: Painter, x: float, y: float, size: tuple[float, float, float]) -> Top:
    """A brick glasshouse where the glass is blown, `size` (length, depth and height to its eaves)
    on (x, y): tall arched windows glowing orange from the furnaces inside under cream
    dressings, a roof glazed in aqua panes catching the light, and a glass lantern along its
    ridge, its panes lit warm by the furnaces under a little glazed roof."""
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
    _glass_roof(p, top, 2.2 * m, max(6, round(length / m)))
    ridge_back, ridge_front = _ridge(top, 2.2 * m)
    vx, vy = _on_wall(ridge_back, ridge_front, 0.5, 0)
    lantern = p.hall(vx, vy + 0.5 * m, length * 0.6, 1.8 * m, 1.3 * m, GLASS_AQUA, None)
    l_left, l_front, l_right, _ = lantern
    panes = max(4, round(length * 0.6 / m))
    for k in range(panes):
        t0, t1 = k / panes, (k + 1) / panes
        if k % 3 == 1:
            _patch(p, l_left, l_front, (t0, t1), (0, 1.3 * m), shade(GLASS_HOT, 0.95))
        _bar(
            p,
            _on_wall(l_left, l_front, t0, 0),
            _on_wall(l_left, l_front, t0, 1.3 * m),
            0.07 * m,
            shade(GLASS_SEA, 0.6),
        )
    _patch(p, l_front, l_right, (0, 1), (0, 1.3 * m), shade(GLASS_AQUA, 0.78))
    _bar(
        p,
        _on_wall(l_left, l_front, 0.1, 0.3 * m),
        _on_wall(l_left, l_front, 0.3, 0.3 * m),
        0.1 * m,
        GLASS_GLINT,
    )
    _glass_roof(p, lantern, 0.7 * m, panes)
    return top


def _glass_lehr(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """A long low lehr shed `length` by `depth` on (x, y) where new glass cools slowly: brick
    walls with small windows glowing amber, a slate roof with a long aqua skylight glinting in
    it and a row of short flues along its ridge."""
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
    sky = [
        _on_roof(top, 1.4 * m, 0.06, 0.25),
        _on_roof(top, 1.4 * m, 0.94, 0.25),
        _on_roof(top, 1.4 * m, 0.94, 0.8),
        _on_roof(top, 1.4 * m, 0.06, 0.8),
    ]
    p.polygon(sky, GLASS_AQUA, shade(GLASS_SEA, 0.6))
    for k in range(1, bays * 2):
        t = 0.06 + 0.88 * k / (bays * 2)
        _bar(
            p,
            _on_roof(top, 1.4 * m, t, 0.25),
            _on_roof(top, 1.4 * m, t, 0.8),
            0.06 * m,
            shade(GLASS_SEA, 0.7),
        )
    _bar(
        p,
        _on_roof(top, 1.4 * m, 0.1, 0.55),
        _on_roof(top, 1.4 * m, 0.35, 0.65),
        0.1 * m,
        GLASS_GLINT,
    )
    ridge_back, ridge_front = _ridge(top, 1.4 * m)
    for k in range(max(2, bays // 2)):
        fx, fy = _on_wall(ridge_back, ridge_front, (k + 0.5) / max(2, bays // 2), 0)
        p.hall(fx, fy + 0.3 * m, 0.7 * m, 0.7 * m, 1.3 * m, shade(GLASS_BRICK, 0.85), GLASS_SOOT)


def _glass_packing(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """A boarded packing house and office `length` by `depth` on (x, y): a show window of
    coloured glass on two shelves, vases and bottles in cobalt, ruby, amber, green and aqua
    glowing in the light, a blue glass globe lamp hung out from its corner, its wide door open on
    straw and crates, under a felted roof."""
    m = METRE
    wall, high = (150, 118, 84), 4 * m
    top = p.hall(x, y, length, depth, high, wall, None)
    left, front, right, _ = top
    _plinth(p, top, high, wall, 0.4 * m)
    for a, b, tone in ((left, front, 0.8), (front, right, 0.62)):
        for t in np.linspace(0.04, 0.96, 12):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, high), 0.04 * m, shade(wall, tone))
    _patch(p, left, front, (0.05, 0.47), (0.7 * m, 3 * m), (240, 236, 226))
    _patch(p, left, front, (0.07, 0.45), (0.9 * m, 2.85 * m), (232, 226, 206))
    colours = (GLASS_COBALT, GLASS_RUBY, GLASS_AMBER, GLASS_GREEN, GLASS_AQUA, GLASS_SEA)
    for row, shelf in enumerate((1.75 * m, 2.75 * m)):
        _bar(
            p,
            _on_wall(left, front, 0.07, shelf),
            _on_wall(left, front, 0.45, shelf),
            0.08 * m,
            (120, 92, 66),
        )
        for k in range(5):
            t = 0.1 + 0.33 * (k + 0.5) / 5
            vx, vy = _on_wall(left, front, t, shelf)
            colour = colours[(k + 2 * row) % len(colours)]
            tall = (0.55 + 0.2 * ((k + row) % 2)) * m
            p.blob(vx - 0.16 * m, vy - tall, vx + 0.16 * m, vy, colour)
            _bar(p, (vx, vy - tall), (vx, vy - tall - 0.15 * m), 0.1 * m, colour)
            _bar(
                p,
                (vx - 0.06 * m, vy - tall * 0.8),
                (vx - 0.06 * m, vy - tall * 0.5),
                0.04 * m,
                GLASS_GLINT,
            )
    _patch(p, left, front, (0.55, 0.92), (1.2 * m, high), (52, 44, 40))
    _patch(p, left, front, (0.57, 0.9), (high - 0.7 * m, high), (226, 196, 120))
    _patch(p, left, front, (0.6, 0.74), (high - 1.5 * m, high - 0.5 * m), GOODS_CRATE)
    _patch(p, front, right, (0.35, 0.65), (1 * m, 2 * m), shade(WINDOW, 0.8))
    p.gable(top, 1.6 * m, (70, 66, 64), wall)
    _roof_rows(p, top, 1.6 * m, (70, 66, 64), (0.5,))
    bx, by = _on_wall(left, front, 0.01, 0.4 * m)
    end = (bx - 1.2 * m, by + 0.6 * m)
    _bar(p, (bx, by), end, 0.1 * m, (44, 40, 40))
    _bar(p, end, (end[0], end[1] + 0.5 * m), 0.05 * m, (44, 40, 40))
    gx, gy = end[0], end[1] + 1 * m
    p.blob(gx - 0.45 * m, gy - 0.45 * m, gx + 0.45 * m, gy + 0.45 * m, GLASS_COBALT)
    p.blob(gx - 0.3 * m, gy - 0.32 * m, gx - 0.02 * m, gy - 0.06 * m, shade(GLASS_COBALT, 1.6))


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
            p, (gx - 0.05 * m, gy - 0.25 * m), (gx - 0.05 * m, gy - 0.1 * m), 0.06 * m, GLASS_GLINT
        )


def _glass_baskets(p: Painter, x: float, y: float, bottles: tuple[RGB, ...]) -> None:
    """Round wicker baskets heaped with new bottles along the lit walls from (x, y), one basket
    for each of `bottles`, the bottles of its colour glinting."""
    m = METRE
    for k, colour in enumerate(bottles):
        bx, by = _iso(x, y, k * 1.2 * m, 0)
        r, tall = 0.5 * m, 0.5 * m
        p.polygon(
            [(bx - r, by - tall), (bx + r, by - tall), (bx + 0.8 * r, by), (bx - 0.8 * r, by)],
            GLASS_WICKER,
        )
        p.polygon(
            [
                (bx + 0.3 * r, by - tall),
                (bx + r, by - tall),
                (bx + 0.8 * r, by),
                (bx + 0.25 * r, by),
            ],
            shade(GLASS_WICKER, 0.78),
        )
        for f in (0.3, 0.65):
            _bar(
                p,
                (bx - r * (0.8 + 0.2 * f), by - tall * f),
                (bx + r * (0.8 + 0.2 * f), by - tall * f),
                0.05 * m,
                shade(GLASS_WICKER, 0.66),
            )
        p.blob(bx - r, by - tall - r * 0.45, bx + r, by - tall + r * 0.45, shade(colour, 0.7))
        for dx, dy in ((-0.45, -0.1), (0.05, -0.25), (0.4, 0.0), (-0.1, 0.15)):
            cx, cy = bx + dx * r, by - tall + dy * r - 0.1 * m
            _bar(p, (cx - 0.22 * m, cy + 0.05 * m), (cx + 0.22 * m, cy - 0.05 * m), 0.2 * m, colour)
        _bar(
            p,
            (bx - 0.25 * r, by - tall - 0.2 * m),
            (bx + 0.05 * r, by - tall - 0.25 * m),
            0.05 * m,
            GLASS_GLINT,
        )


def _glass_carboys(p: Painter, x: float, y: float, colours: tuple[RGB, ...]) -> None:
    """A row of carboys along the lit walls from (x, y), one for each of `colours`: a round
    bottle 1.3 m tall in that glass sitting in a low woven wicker jacket, its shoulder and neck
    showing above and glinting."""
    m = METRE
    for k, colour in enumerate(colours):
        cx, cy = _iso(x, y, k * 1.1 * m, 0)
        p.blob(cx - 0.48 * m, cy - 1.15 * m, cx + 0.48 * m, cy - 0.1 * m, colour)
        p.blob(cx - 0.34 * m, cy - 1.05 * m, cx - 0.02 * m, cy - 0.7 * m, shade(colour, 1.4))
        _bar(p, (cx, cy - 1.05 * m), (cx, cy - 1.38 * m), 0.2 * m, shade(colour, 0.9))
        p.blob(cx - 0.13 * m, cy - 1.46 * m, cx + 0.13 * m, cy - 1.33 * m, shade(colour, 0.75))
        p.polygon(
            [
                (cx - 0.5 * m, cy - 0.45 * m),
                (cx + 0.5 * m, cy - 0.45 * m),
                (cx + 0.42 * m, cy),
                (cx - 0.42 * m, cy),
            ],
            GLASS_WICKER,
        )
        p.polygon(
            [
                (cx + 0.16 * m, cy - 0.45 * m),
                (cx + 0.5 * m, cy - 0.45 * m),
                (cx + 0.42 * m, cy),
                (cx + 0.14 * m, cy),
            ],
            shade(GLASS_WICKER, 0.78),
        )
        _bar(
            p,
            (cx - 0.47 * m, cy - 0.22 * m),
            (cx + 0.47 * m, cy - 0.22 * m),
            0.05 * m,
            shade(GLASS_WICKER, 0.66),
        )
        _bar(p, (cx - 0.24 * m, cy - 1.0 * m), (cx - 0.14 * m, cy - 0.8 * m), 0.07 * m, GLASS_GLINT)


def _glass_panes(p: Painter, x: float, y: float, long: float, tall: float) -> None:
    """A big A-frame rack `long` metres along the lit walls and `tall` high standing on (x, y),
    stacked on both sides with sheets of new window glass: cool aqua faces, sea-green edges
    showing where they lie one on another, and long glints of light."""
    m = METRE
    half = long * m / 2
    a, b = (x - half, y - half / 2), (x + half, y + half / 2)
    top = tall * m
    _bar(p, (a[0] + 0.5 * m, a[1] - 0.25 * m), (a[0], a[1] - top), 0.16 * m, shade(TIMBER, 0.85))
    for lean, tone in ((0.5, 0.78), (-0.5, 1.0)):
        for k in range(3):
            off = (lean - 0.12 * k * (1 if lean > 0 else -1)) * m
            sheet = [
                (a[0] + off, a[1]),
                (b[0] + off, b[1]),
                (b[0], b[1] - top * (0.96 - 0.03 * k)),
                (a[0], a[1] - top * (0.96 - 0.03 * k)),
            ]
            p.polygon(sheet, shade(GLASS_AQUA, tone - 0.05 * k), shade(GLASS_SEA, 0.7))
            _bar(p, sheet[3], sheet[2], 0.12 * m, shade(GLASS_SEA, tone))
    for t in (0.15, 0.45, 0.7):
        sx, sy = a[0] + 2 * half * t - 0.3 * m, a[1] + half * t
        _bar(p, (sx, sy - top * 0.2), (sx + 0.6 * m, sy - top * 0.75), 0.1 * m, GLASS_GLINT)
    _bar(p, (b[0] - 0.5 * m, b[1] - 0.25 * m), (b[0], b[1] - top), 0.16 * m, TIMBER)
    _bar(p, (b[0] + 0.5 * m, b[1] - 0.25 * m), (b[0], b[1] - top), 0.16 * m, shade(TIMBER, 0.8))
    _bar(p, (a[0], a[1] - top), (b[0], b[1] - top), 0.12 * m, shade(TIMBER, 0.9))


def _glass_cullet(
    p: Painter, x: float, y: float, r: float, tall: float, colours: tuple[RGB, ...]
) -> None:
    """A heap of cullet, broken glass waiting to be melted again, `r` across its foot and `tall`
    high on (x, y): shards in each of `colours` heaped over a darker mass of the first,
    glinting."""
    m = METRE
    _glass_heap(p, x, y, r, tall, shade(colours[0], 0.88))
    rng = np.random.default_rng(len(colours) * 7 + round(r * 10))
    for k in range(int(r / m * 14)):
        f = math.sqrt(rng.uniform(0, 1))
        side = rng.uniform(-1, 1)
        sx = x + side * r * f
        sy = y - tall * (1 - f) + r * 0.3 * f * rng.uniform(-0.6, 0.6)
        s = rng.uniform(0.15, 0.3) * m
        colour = colours[k % len(colours)]
        p.polygon(
            [(sx - s, sy), (sx, sy - s * 1.2), (sx + s, sy + s * 0.2)],
            shade(colour, 1.0 if side < 0.3 else 0.8),
        )
    for _ in range(5):
        f = rng.uniform(0.2, 0.9)
        gx, gy = x + rng.uniform(-0.6, 0.3) * r * f, y - tall * (1 - f)
        _bar(p, (gx - 0.12 * m, gy), (gx + 0.12 * m, gy), 0.06 * m, GLASS_GLINT)
        _bar(p, (gx, gy - 0.12 * m), (gx, gy + 0.12 * m), 0.06 * m, GLASS_GLINT)


def _glass_heap(p: Painter, x: float, y: float, r: float, tall: float, colour: RGB) -> None:
    """A cone of loose stuff `r` across its foot and `tall` high on (x, y), lit on its left."""
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
    iron and sooted at its smoking top, its furnace mouth glowing in a deep arch with gathers of
    molten glass drawn out on blowpipes; joined to it a brick glasshouse whose arched windows
    glow with the furnaces under a roof glazed in aqua and a glass lantern; a long lehr shed with
    amber windows and an aqua skylight; a packing house with a show window of coloured glass and
    a blue glass globe lamp; and a yard of big racks of window panes glinting aqua and sea green,
    crates and baskets of bottles in green, amber and cobalt, carboys in wicker, and heaps of
    coloured cullet."""
    m = METRE
    p = Painter(44, 45, (21, 32))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 16) * m, (v - 6) * m)

    _glass_yard(
        p, [(at(14, 5), 13), (at(5, -1), 7), (at(26, 2), 8), (at(31, 7), 6), (at(20, -4), 7)]
    )
    for u, v, length, depth in ((19, 8.5, 10, 9), (30, 8, 12, 5), (31, -1, 7, 5)):
        _bed(p, _footing(*at(u, v), length * m, depth * m), margin=0.8)
    _glass_cone(p, *at(8, 8), 7 * m, 24 * m)
    _glass_house(p, *at(19, 8.5), (10 * m, 9 * m, 6.5 * m))
    _glass_lehr(p, *at(30, 8), 12 * m, 5 * m)
    _glass_packing(p, *at(31, -1), 7 * m, 5 * m)
    _glass_cullet(p, *at(-1, 2), 2.8 * m, 2.2 * m, (GLASS_AQUA, GLASS_SEA, GLASS_GREEN))
    _glass_cullet(p, *at(1.5, -2), 2 * m, 1.6 * m, (GLASS_COBALT, GLASS_AMBER, GLASS_AQUA))
    _glass_panes(p, *at(9.5, -3.4), 4.5, 2.6)
    _glass_panes(p, *at(15, -4.2), 4.5, 2.6)
    _glass_crates(p, *at(21, -1.8), (GLASS_GREEN, GLASS_AMBER, GLASS_COBALT))
    _glass_crates(p, *at(21.6, -3.4), (GLASS_COBALT, GLASS_GREEN, GLASS_AMBER))
    _glass_carboys(p, *at(18.5, -5.2), (GLASS_GREEN, GLASS_AMBER, GLASS_GREEN, GLASS_COBALT))
    _glass_baskets(p, *at(24, -5.4), (GLASS_AMBER, GLASS_GREEN, GLASS_COBALT))
    return p.sprite()


BREW_BRICK: RGB = (208, 150, 82)
FOAM: RGB = (246, 238, 214)
HOP: RGB = (78, 124, 52)
MALT_TILE: RGB = (126, 78, 50)
BREW_ROOM: RGB = (66, 44, 32)
KETTLE: RGB = (226, 132, 64)
SHEAF: RGB = (232, 186, 72)
BAY: RGB = (122, 70, 40)
HORSE_MANE: RGB = (42, 34, 32)
FEATHER: RGB = (238, 234, 224)
SETTS: RGB = (156, 146, 128)
DRAY_WHEEL: RGB = (180, 106, 50)
LUCAM: RGB = (150, 104, 64)


def _brew_window(
    p: Painter, a: Point, b: Point, length: float, t: float, wide: float, span: tuple[float, float]
) -> None:
    """A tall round-headed brewhouse window `wide` across about `t` along the wall under the edge
    from `a` to `b`, which is `length` units long, from `span[0]` down the wall to `span[1]`,
    under a cream arch with a stone sill: through it a copper brew kettle gleams in the warm
    room, its dome rising across the lower half and its vapour pipe climbing out of sight, behind
    slim white glazing bars."""
    m = METRE
    head, foot = span
    half = wide / 2 / length
    p.polygon(_round_head(a, b, length, t, wide + 0.7 * m, head - 0.35 * m, head + wide / 2), FOAM)
    p.polygon(_round_head(a, b, length, t, wide, head, foot), BREW_ROOM)
    crown = head + (foot - head) * 0.38
    _bar(
        p,
        _on_wall(a, b, t + half * 0.15, crown),
        _on_wall(a, b, t + half * 0.15, head),
        0.45 * m,
        KETTLE,
    )
    _bar(
        p,
        _on_wall(a, b, t + half * 0.35, crown),
        _on_wall(a, b, t + half * 0.35, head),
        0.12 * m,
        shade(KETTLE, 0.7),
    )

    def dome(k: float) -> float:
        return crown + 1.3 * m * (1 - math.sqrt(max(0.0, 1 - k * k)))

    rim = [_on_wall(a, b, t + half * k, dome(k)) for k in np.linspace(-1, 1, 13)]
    p.polygon([_on_wall(a, b, t - half, foot), *rim, _on_wall(a, b, t + half, foot)], KETTLE)
    p.polygon(
        [
            *[_on_wall(a, b, t + half * k, dome(k)) for k in np.linspace(0.3, 1, 6)],
            _on_wall(a, b, t + half, foot),
            _on_wall(a, b, t + half * 0.3, foot),
        ],
        shade(KETTLE, 0.78),
    )
    _bar(
        p,
        _on_wall(a, b, t - half * 0.5, dome(-0.5) + 0.25 * m),
        _on_wall(a, b, t - half * 0.5, foot - 0.3 * m),
        0.28 * m,
        shade(KETTLE, 1.25),
    )
    _bar(
        p,
        _on_wall(a, b, t - half, crown + 2.2 * m),
        _on_wall(a, b, t + half, crown + 2.2 * m),
        0.12 * m,
        shade(KETTLE, 0.6),
    )
    bar = shade(FOAM, 0.94)
    _bar(p, _on_wall(a, b, t, head + 0.3 * m), _on_wall(a, b, t, foot), 0.07 * m, bar)
    for drop in (head + wide / 2, (crown + foot) / 2 + 0.6 * m):
        _bar(p, _on_wall(a, b, t - half, drop), _on_wall(a, b, t + half, drop), 0.07 * m, bar)
    p.polygon(
        [
            _on_wall(a, b, t - half * 1.3, foot),
            _on_wall(a, b, t + half * 1.3, foot),
            _on_wall(a, b, t + half * 1.3, foot + 0.3 * m),
            _on_wall(a, b, t - half * 1.3, foot + 0.3 * m),
        ],
        FOAM,
    )


def _brew_louvres(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: tuple[float, float]
) -> None:
    """A louvred opening in a brew tower's cooling floor on the wall under the edge from `a` to
    `b`: cream boards slanting across a dark opening in a cream frame."""
    m = METRE
    t0, t1 = span
    _patch(p, a, b, (t0 - 0.03, t1 + 0.03), (drop[0] - 0.25 * m, drop[1] + 0.25 * m), FOAM)
    _patch(p, a, b, span, drop, (52, 42, 36))
    for d in np.arange(drop[0] + 0.25 * m, drop[1], 0.42 * m):
        _bar(p, _on_wall(a, b, t0, d), _on_wall(a, b, t1, d), 0.2 * m, shade(FOAM, 0.86))


def _brew_tankard(p: Painter, x: float, y: float, size: float) -> None:
    """A gilt tankard `size` tall with its middle at (x, y), hooped like a little cask, its handle
    to the right, crowned with a head of cream foam running down its side."""
    s = size
    _bar(p, (x + 0.3 * s, y - 0.22 * s), (x + 0.52 * s, y - 0.18 * s), 0.11 * s, shade(GILT, 0.8))
    _bar(p, (x + 0.52 * s, y - 0.18 * s), (x + 0.52 * s, y + 0.14 * s), 0.11 * s, shade(GILT, 0.8))
    _bar(p, (x + 0.52 * s, y + 0.14 * s), (x + 0.3 * s, y + 0.24 * s), 0.11 * s, shade(GILT, 0.8))
    body = [
        (x - 0.32 * s, y - 0.3 * s),
        (x + 0.32 * s, y - 0.3 * s),
        (x + 0.35 * s, y + 0.46 * s),
        (x - 0.35 * s, y + 0.46 * s),
    ]
    p.polygon(body, GILT)
    p.polygon(
        [(x + 0.1 * s, y - 0.3 * s), body[1], body[2], (x + 0.12 * s, y + 0.46 * s)],
        shade(GILT, 0.82),
    )
    _bar(p, (x - 0.2 * s, y - 0.25 * s), (x - 0.2 * s, y + 0.4 * s), 0.07 * s, shade(GILT, 1.2))
    for drop in (-0.12, 0.33):
        _bar(
            p,
            (x - 0.34 * s, y + drop * s),
            (x + 0.34 * s, y + drop * s),
            0.06 * s,
            shade(GILT, 0.6),
        )
    p.blob(x - 0.4 * s, y - 0.5 * s, x + 0.08 * s, y - 0.22 * s, FOAM)
    p.blob(x - 0.12 * s, y - 0.48 * s, x + 0.4 * s, y - 0.24 * s, shade(FOAM, 0.94))
    p.blob(x - 0.42 * s, y - 0.34 * s, x - 0.24 * s, y - 0.06 * s, FOAM)


def _brew_sign(p: Painter, x: float, y: float) -> None:
    """A brewer's sign hung from a wrought-iron bracket fixed to a wall at (x, y), reaching out
    towards the lit side: a hop-green board in a gilt frame painted with a golden tankard."""
    m = METRE
    iron = (44, 40, 40)
    end = (x - 2.4 * m, y + 1.2 * m)
    _bar(p, (x, y), end, 0.18 * m, iron)
    _bar(p, (x, y + 1 * m), (x - 1.2 * m, y + 0.6 * m), 0.1 * m, iron)
    cx = end[0] + 0.8 * m
    top = end[1] + 0.45 * m
    for dx in (-1.1, 1.1):
        _bar(p, (cx + dx * m, top), (cx + dx * 0.4 * m, end[1]), 0.06 * m, iron)
    p.rect(cx - 1.5 * m, top, cx + 1.5 * m, top + 2.6 * m, GILT)
    p.rect(cx - 1.3 * m, top + 0.2 * m, cx + 1.3 * m, top + 2.4 * m, HOP)
    _brew_tankard(p, cx - 0.1 * m, top + 1.35 * m, 1.7 * m)


def _brew_sheaf(p: Painter, x: float, y: float, tall: float) -> None:
    """A gilt sheaf of barley `tall` high standing on (x, y): its stalks spread a little at the
    foot, bound with a band a third of the way up and fanning out above it, every one ending in
    a bearded ear."""
    t = tall
    band = y - 0.36 * t
    for k in np.linspace(-1, 1, 7):
        tone = shade(SHEAF, 1.12 - 0.22 * (k + 1) / 2)
        foot = (x + 0.2 * k * t, y)
        tie = (x + 0.05 * k * t, band)
        tip = (x + 0.36 * k * t, y - (0.78 - 0.1 * abs(k)) * t)
        _bar(p, foot, tie, 0.045 * t, shade(tone, 0.9))
        _bar(p, tie, tip, 0.045 * t, tone)
        dx, dy = tip[0] - tie[0], tip[1] - tie[1]
        reach = math.hypot(dx, dy)
        ux, uy = dx / reach, dy / reach
        ear_end = (tip[0] + ux * 0.22 * t, tip[1] + uy * 0.22 * t)
        _bar(p, tip, ear_end, 0.08 * t, shade(tone, 1.06))
        _bar(p, ear_end, (ear_end[0] + ux * 0.12 * t, ear_end[1] + uy * 0.12 * t), 0.02 * t, tone)
    _bar(p, (x - 0.1 * t, band), (x + 0.1 * t, band), 0.07 * t, shade(SHEAF, 0.62))


def _brew_wheel(
    p: Painter, x: float, y: float, u: float, v: float, r: float, paint: RGB = DRAY_WHEEL
) -> None:
    """A spoked cart wheel `r` metres round standing on the ground `u` metres along the lit walls
    and `v` along the shaded walls from (x, y), turned along the lit walls: an iron tyre on a rim
    painted `paint`, eight spokes and a hub."""
    m = METRE
    turns = np.linspace(0, 2 * math.pi, 25)

    def at(k: float, turn: float) -> Point:
        return _cask_at(x, y, u + k * r * math.cos(turn), v, r + k * r * math.sin(turn))

    p.polygon([at(1, a) for a in turns] + [at(0.8, a) for a in turns[::-1]], paint)
    for a0, a1 in itertools.pairwise(turns):
        _bar(p, at(1, a0), at(1, a1), 0.08 * m, (48, 44, 42))
    for k in range(4):
        turn = k * math.pi / 4
        _bar(p, at(0.8, turn), at(0.8, turn + math.pi), 0.07 * m, shade(paint, 0.9))
    hx, hy = at(0, 0)
    p.ellipse(hx - 0.14 * m, hy - 0.14 * m, hx + 0.14 * m, hy + 0.14 * m, shade(paint, 0.7))


def _brew_horse(p: Painter, x: float, y: float, facing: int = 1) -> None:
    """A bay Shire horse 2.6 m long standing on (x, y), turned along the lit walls with its head
    towards the shaded side, or the lit side with `facing` -1: a deep rounded body on four heavy
    legs with white feathered feet, an arched neck with a black mane, a long head with a white
    blaze, a black collar with brass hames, a cart saddle and a black tail."""

    def at(u: float, z: float, v: float = 0) -> Point:
        return _cask_at(x, y, facing * u, v, z)

    def shape(points: list[tuple[float, float]], colour: RGB, v: float = 0) -> None:
        p.polygon([at(u, z, v) for u, z in points], colour)

    def oval(u: float, z: float, long: float, high: float, colour: RGB, v: float = 0) -> None:
        shape(
            [
                (u + long * math.cos(a), z + high * math.sin(a))
                for a in np.linspace(0, 2 * math.pi, 17)
            ],
            colour,
            v,
        )

    def leg(u: float, hind: bool, v: float, coat: RGB, feather: RGB) -> None:
        if hind:
            shape(
                [
                    (u - 0.22, 1.12),
                    (u + 0.18, 1.08),
                    (u + 0.06, 0.6),
                    (u + 0.08, 0.3),
                    (u - 0.08, 0.3),
                    (u - 0.14, 0.62),
                ],
                coat,
                v,
            )
        else:
            shape([(u - 0.13, 1.05), (u + 0.13, 1.05), (u + 0.08, 0.3), (u - 0.08, 0.3)], coat, v)
        shape([(u - 0.11, 0.36), (u + 0.11, 0.36), (u + 0.17, 0.02), (u - 0.17, 0.02)], feather, v)
        shape([(u - 0.15, 0.06), (u + 0.15, 0.06), (u + 0.15, 0), (u - 0.15, 0)], (52, 46, 42), v)

    far, near = 0.2, -0.2
    dark = shade(BAY, 0.7)
    leg(0.66, False, far, dark, shade(FEATHER, 0.8))
    leg(-0.6, True, far, dark, shade(FEATHER, 0.8))
    shape([(-0.98, 1.62), (-1.1, 1.3), (-1.08, 0.78), (-0.94, 0.8), (-0.9, 1.4)], HORSE_MANE)
    oval(0, 1.32, 0.95, 0.42, dark, far)
    oval(-0.6, 1.4, 0.44, 0.44, BAY, near / 2)
    oval(0.6, 1.34, 0.42, 0.42, BAY, near / 2)
    oval(0, 1.32, 0.95, 0.42, BAY, near / 2)
    oval(-0.1, 1.6, 0.7, 0.1, shade(BAY, 1.18), near / 2)
    shape(
        [
            (0.42, 1.2),
            (0.48, 1.72),
            (0.78, 2.04),
            (1.02, 2.18),
            (1.16, 2.06),
            (1.02, 1.7),
            (0.96, 1.2),
        ],
        BAY,
    )
    shape(
        [(0.46, 1.74), (0.76, 2.08), (1.02, 2.24), (1.06, 2.14), (0.8, 1.98), (0.54, 1.66)],
        HORSE_MANE,
    )
    shape(
        [
            (0.98, 2.16),
            (1.16, 2.22),
            (1.24, 2.1),
            (1.6, 1.62),
            (1.56, 1.48),
            (1.42, 1.46),
            (1.32, 1.56),
            (1.0, 1.9),
        ],
        BAY,
        near / 2,
    )
    shape([(1.04, 2.14), (1.06, 2.36), (1.14, 2.16)], shade(BAY, 0.8), near / 2)
    shape([(1.14, 2.1), (1.2, 2.1), (1.52, 1.6), (1.46, 1.56)], FEATHER, near / 2)
    shape([(0.4, 1.18), (0.56, 1.14), (0.72, 1.84), (0.56, 1.9)], (40, 36, 34), near)
    shape([(0.5, 1.22), (0.56, 1.21), (0.68, 1.86), (0.62, 1.88)], GILT, near)
    shape([(-0.02, 1.68), (0.36, 1.68), (0.32, 1.86), (0.02, 1.86)], (40, 36, 34), near)
    shape([(0.14, 1.7), (0.2, 1.7), (0.2, 0.94), (0.14, 0.94)], (40, 36, 34), near)
    leg(0.58, False, near, BAY, FEATHER)
    leg(-0.66, True, near, BAY, FEATHER)


def _brew_dray(p: Painter, x: float, y: float) -> None:
    """A brewer's dray drawn up along the lit walls from (x, y) with a Shire horse in its shafts:
    a hop-green bed 3.4 m long lined in gold on four honey-painted spoked wheels, the front pair
    smaller, loaded with a pyramid of oak casks, with a seat for the drayman at its front."""
    m = METRE
    p.soft(
        [(*_cask_at(x, y, -0.5, 0.6), *_cask_at(x, y, 7, -1.2))],
        (34, 30, 24),
        0.4,
        0.4,
    )
    for u, r in ((0.7, 0.6), (2.8, 0.45)):
        _brew_wheel(p, x, y, u, 0.75, r)
    _bar(
        p,
        _cask_at(x, y, 3.3, 0.45, 1.0),
        _cask_at(x, y, 5.6, 0.45, 1.35),
        0.12 * m,
        shade(TIMBER, 0.6),
    )
    _brew_horse(p, *_cask_at(x, y, 5.1, 0))
    bed = p.hall(*_cask_at(x, y, 1.7, 0, 0.75), 3.4 * m, 1.6 * m, 0.35 * m, HOP, shade(TIMBER, 1.1))
    left, front, _, _ = bed
    _band(p, left, front, 0.17 * m, GILT)
    _cask_stack(p, *_cask_at(x, y, 0.45, 0, 1.1), (3, 2), shadow=False)
    seat = p.hall(*_cask_at(x, y, 3.05, 0, 1.1), 0.5 * m, 1.3 * m, 0.45 * m, HOP, (60, 44, 36))
    _band(p, seat[0], seat[1], 0.2 * m, GILT)
    for u, r in ((0.7, 0.6), (2.8, 0.45)):
        _brew_wheel(p, x, y, u, -0.85, r)
    _bar(
        p,
        _cask_at(x, y, 3.3, -0.45, 1.0),
        _cask_at(x, y, 5.6, -0.45, 1.35),
        0.12 * m,
        shade(TIMBER, 0.8),
    )


def brewery() -> Sprite:
    """A Victorian tower brewery of warm honey brick dressed in cream: a brewhouse 18 by 10 m with
    walls 9 m high under a roof of malt-brown tiles, the copper brew kettles gleaming through its
    three tall arched windows and steam rising from a copper vapour stack on its ridge, a hop-green
    door and a hop-green sign with a golden tankard hung by it; a brew tower 18 m high with a
    louvred cooling floor under its pyramid roof, crowned by a gilt sheaf of barley, and a cask
    door at its foot by a neat pyramid of oak casks; a tall malt store with taking-in doors under a
    timber lucam, a sack of malt on the hoist; and a dray loaded with casks in the cobbled yard,
    a Shire horse in its shafts."""
    m = METRE
    p = Painter(46, 42, (23, 22))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10) * m, (v - 3.5) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon(
        [
            at(-1, -6.5),
            at(15, -6.5),
            at(18, -2.5),
            at(27, -2.5),
            at(27, 8.5),
            at(23, 8.5),
            at(-1, 0),
        ],
        SETTS,
    )
    malt_foot = _footing(*at(-5, 7), 8 * m, 8 * m)
    _bed(p, malt_foot, margin=0.8)
    _bed(p, _footing(*at(9, 5), 18 * m, 10 * m), margin=0.8)
    p.shadow(13, 4, 4)

    malt = p.hall(*at(-5, 7), 8 * m, 8 * m, 11 * m, BREW_BRICK, None)
    k_left, k_front, k_right, _ = malt
    _plinth(p, malt, 11 * m, BREW_BRICK, 0.6 * m)
    for a, b in ((k_left, k_front), (k_front, k_right)):
        for t in (0.03, 0.97):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, 10.4 * m), 0.35 * m, FOAM)
        _bar(p, _on_wall(a, b, 0, 0.2 * m), _on_wall(a, b, 1, 0.2 * m), 0.35 * m, FOAM)
    for head in (1.6 * m, 5.2 * m):
        _patch(p, k_left, k_front, (0.38, 0.62), (head - 0.25 * m, head + 2.6 * m), FOAM)
        _patch(p, k_left, k_front, (0.4, 0.6), (head, head + 2.4 * m), HOP)
        _bar(
            p,
            _on_wall(k_left, k_front, 0.5, head),
            _on_wall(k_left, k_front, 0.5, head + 2.4 * m),
            0.06 * m,
            shade(HOP, 0.7),
        )
        for t in (0.17, 0.83):
            _mill_window(p, k_left, k_front, 8 * m, t, 1 * m, (head + 0.5 * m, head + 2 * m))
    _patch(p, k_left, k_front, (0.36, 0.64), (8.4 * m, 11 * m), FOAM)
    _patch(p, k_left, k_front, (0.38, 0.62), (8.6 * m, 11 * m), HOP)
    for t in (0.17, 0.83):
        _mill_window(p, k_left, k_front, 8 * m, t, 1 * m, (8.8 * m, 10.2 * m))
    _mill_window(
        p, k_front, k_right, 8 * m, 0.7, 1 * m, (1.6 * m, 3.4 * m), shade(MILL_GLASS, 0.85)
    )
    _gable_across(p, malt, 4 * m, MALT_TILE, BREW_BRICK)
    lucam = p.hall(*up(-5, 2.4, 9.2), 2 * m, 1.2 * m, 2.6 * m, LUCAM, None)
    for k in range(1, 6):
        _band(p, lucam[0], lucam[1], 2.6 * m * k / 6, shade(LUCAM, 0.82))
    _patch(p, lucam[0], lucam[1], (0.25, 0.75), (0.5 * m, 2.6 * m), (52, 42, 36))
    _gable_across(p, lucam, 1 * m, MALT_TILE, LUCAM)
    beam = _on_wall(lucam[0], lucam[1], 0.5, -0.2 * m)
    rope_end = (beam[0], beam[1] + 3.6 * m)
    _bar(p, beam, rope_end, 0.05 * m, (60, 52, 46))
    p.blob(
        rope_end[0] - 0.45 * m,
        rope_end[1],
        rope_end[0] + 0.45 * m,
        rope_end[1] + 1.1 * m,
        (206, 182, 132),
    )

    length, depth, high = 18 * m, 10 * m, 9 * m
    top = p.hall(*at(9, 5), length, depth, high, BREW_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, BREW_BRICK, 0.7 * m)
    for a, b in ((left, front), (front, right)):
        _bar(p, _on_wall(a, b, 0, 0.22 * m), _on_wall(a, b, 1, 0.22 * m), 0.45 * m, FOAM)
        _bar(p, _on_wall(a, b, 0, 1.0 * m), _on_wall(a, b, 1, 1.0 * m), 0.15 * m, shade(FOAM, 0.9))
    for k in range(6):
        _bar(
            p,
            _on_wall(left, front, k / 5, 0.45 * m),
            _on_wall(left, front, k / 5, high - 0.7 * m),
            0.5 * m,
            shade(BREW_BRICK, 0.88),
        )
    for t in (0.3, 0.5, 0.7):
        _brew_window(p, left, front, length, t, 2.3 * m, (1.6 * m, 7.8 * m))
    for drop in ((1.6 * m, 3.8 * m), (5.2 * m, 7.6 * m)):
        _mill_window(p, left, front, length, 0.9, 1.4 * m, drop)
    t = 0.1
    p.polygon(_round_head(left, front, length, t, 2.8 * m, 4.3 * m, high), FOAM)
    p.polygon(_round_head(left, front, length, t, 2.3 * m, 4.6 * m, high), HOP)
    _bar(
        p,
        _on_wall(left, front, t, 4.6 * m),
        _on_wall(left, front, t, high),
        0.08 * m,
        shade(HOP, 0.6),
    )
    for side in (-0.035, 0.035):
        _bar(
            p,
            _on_wall(left, front, t + side, 6.4 * m),
            _on_wall(left, front, t + side, 6.6 * m),
            0.1 * m,
            GILT,
        )
    ox, oy = _on_wall(left, front, t, 2.6 * m)
    p.ellipse(ox - 0.9 * m, oy - 0.9 * m, ox + 0.9 * m, oy + 0.9 * m, FOAM)
    p.ellipse(ox - 0.6 * m, oy - 0.6 * m, ox + 0.6 * m, oy + 0.6 * m, MILL_GLASS)
    p.gable(top, 4 * m, MALT_TILE, BREW_BRICK)
    _roof_rows(p, top, 4 * m, MALT_TILE, (0.25, 0.5, 0.75))
    vx, vy = _on_roof(top, 4 * m, 0.55, 1)
    _turned(p, vx, vy + 0.5 * m, [(0, 0.45 * m), (3.2 * m, 0.4 * m)], KETTLE)
    _turned(p, vx, vy - 3 * m, [(0, 0.95 * m), (0.7 * m, 0)], shade(KETTLE, 0.9))
    p.soft(
        [
            (vx - 0.6 * m, vy - 4.6 * m, vx + 0.8 * m, vy - 3.4 * m),
            (vx + 0.2 * m, vy - 5.8 * m, vx + 1.8 * m, vy - 4.4 * m),
            (vx + 1.2 * m, vy - 7 * m, vx + 2.6 * m, vy - 5.8 * m),
        ],
        (248, 246, 240),
        0.75,
        0.3,
    )

    tall, wide, deep = 17 * m, 5.5 * m, 6 * m
    tower = p.hall(*at(21, 4.5), wide, deep, tall, BREW_BRICK, None)
    t_left, t_front, t_right, _ = tower
    _plinth(p, tower, tall, BREW_BRICK, 0.7 * m)
    for a, b in ((t_left, t_front), (t_front, t_right)):
        for t in (0.03, 0.97):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, tall - 0.7 * m), 0.4 * m, FOAM)
        for drop in (0.22 * m, 3.6 * m, 7.8 * m, 11.8 * m):
            _bar(p, _on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop), 0.3 * m, FOAM)
    _brew_louvres(p, t_left, t_front, (0.24, 0.76), (1 * m, 3 * m))
    _brew_louvres(p, t_front, t_right, (0.24, 0.76), (1 * m, 3 * m))
    for drop in ((4.6 * m, 6.9 * m), (8.6 * m, 10.9 * m), (12.6 * m, 14.9 * m)):
        _mill_window(p, t_left, t_front, wide, 0.5, 1.3 * m, drop)
    for drop in ((4.6 * m, 6.9 * m), (8.6 * m, 10.9 * m)):
        _mill_window(p, t_front, t_right, deep, 0.5, 1.2 * m, drop, shade(MILL_GLASS, 0.85))
    p.polygon(_round_head(t_front, t_right, deep, 0.5, 3.2 * m, 12.6 * m, tall), FOAM)
    p.polygon(_round_head(t_front, t_right, deep, 0.5, 2.7 * m, 12.9 * m, tall), shade(HOP, 0.8))
    _bar(
        p,
        _on_wall(t_front, t_right, 0.5, 12.9 * m),
        _on_wall(t_front, t_right, 0.5, tall),
        0.08 * m,
        shade(HOP, 0.5),
    )
    apex = ((t_left[0] + t_right[0]) / 2, (t_left[1] + t_right[1]) / 2 - 3.5 * m)
    p.polygon([t_left, t_front, apex], MALT_TILE, shade(MALT_TILE, 0.6))
    p.polygon([t_front, t_right, apex], shade(MALT_TILE, 0.72), shade(MALT_TILE, 0.5))
    for f in (0.35, 0.7):
        p.line(
            [
                _on_wall(t_left, apex, f, 0),
                _on_wall(t_front, apex, f, 0),
                _on_wall(t_right, apex, f, 0),
            ],
            shade(MALT_TILE, 0.82),
        )
    p.ellipse(apex[0] - 0.35 * m, apex[1] - 0.6 * m, apex[0] + 0.35 * m, apex[1] + 0.1 * m, GILT)
    _brew_sheaf(p, apex[0], apex[1] - 0.4 * m, 3 * m)

    for u, v in ((24.6, 6.6), (25.5, 7.2)):
        _cask(p, *at(u, v), "up")
    _cask_stack(p, *at(24.2, 0.6), (4, 3, 2, 1))
    _brew_dray(p, *at(1.5, -4.6))
    _brew_sign(p, *_on_wall(left, front, 0.015, 2.2 * m))
    return p.sprite()


TAR: RGB = (46, 46, 50)
WHISKY: RGB = (192, 120, 42)
PEAT: RGB = (88, 64, 46)
GRAVEL: RGB = (176, 166, 144)
STILL_GLASS: RGB = (60, 72, 88)

# The outline of an onion-shaped copper pot still, as its half-width at each height in metres
# from its foot: the pot, the boil ball over its shoulder and the tall swan neck.
STILL_PROFILE: list[tuple[float, float]] = [
    (0, 0.8),
    (0.4, 1.0),
    (0.95, 1.0),
    (1.35, 0.76),
    (1.55, 0.42),
    (1.75, 0.6),
    (2.1, 0.62),
    (2.4, 0.38),
    (2.65, 0.22),
    (3.9, 0.17),
]


def _still_pot(p: Painter, x: float, y: float, reach: float) -> None:
    """An onion-shaped copper pot still 3.9 m high standing on its brick firebox at (x, y): a
    gleaming pot with a dark seam, a boil ball and a tall swan neck, its lyne arm running down
    `reach` units across to the right, or to the left if `reach` is below 0."""
    m = METRE
    box = p.box(x, y, 1.1 * m, 0.5 * m, (120, 66, 48), (88, 52, 40))
    _patch(p, box[0], box[1], (0.35, 0.75), (0.15 * m, 0.45 * m), (250, 160, 60))
    foot = y - 0.5 * m
    _turned(p, x, foot, [(h * m, w * m) for h, w in STILL_PROFILE], KETTLE)
    for height, half in ((0.95, 1.0), (1.55, 0.42)):
        cy = foot - height * m
        _bar(p, (x - half * m, cy), (x, cy + half * m / 2), 0.1 * m, shade(KETTLE, 0.62))
        _bar(p, (x, cy + half * m / 2), (x + half * m, cy), 0.1 * m, shade(KETTLE, 0.5))
    for low, high, across, wide in ((0.25, 0.85, 0.55, 0.24), (1.8, 2.05, 0.34, 0.16)):
        _bar(
            p,
            (x - across * m, foot - low * m),
            (x - across * m, foot - high * m),
            wide * m,
            shade(KETTLE, 1.32),
        )
    top = (x, foot - 3.85 * m)
    end = (x + reach, top[1] + abs(reach) * 0.3)
    _bar(p, top, end, 0.28 * m, shade(KETTLE, 0.92))
    _bar(p, (top[0], top[1] - 0.1 * m), (end[0], end[1] - 0.1 * m), 0.08 * m, shade(KETTLE, 1.3))


def _still_worm(p: Painter, x: float, y: float) -> None:
    """A worm tub standing on (x, y): a round oak vat 2.4 m across and 1.8 m high bound with iron
    hoops, the coils of the copper worm showing in the cold water inside."""
    m = METRE
    r = 1.2 * m
    _turned(p, x, y, [(0, r), (1.8 * m, r * 1.04)], OAK, (58, 74, 84))
    for drop in (0.35, 1.2):
        _ring(p, x, y - drop * m, r * (1 + 0.02 * drop), CASK_HOOP)
    cy = y - 1.8 * m
    for k, scale in enumerate((0.72, 0.48)):
        ring = [
            (x + r * scale * math.cos(a), cy + 0.08 * m * k + r * scale / 2 * math.sin(a))
            for a in np.linspace(0, 2 * math.pi, 19)
        ]
        for a, b in itertools.pairwise(ring):
            _bar(p, a, b, 0.12 * m, shade(KETTLE, 1.05 - 0.15 * k))


def _still_pagoda(p: Painter, x: float, y: float) -> None:
    """The pagoda ventilator crowning a malt kiln at (x, y): a whitewashed louvred lantern 2.4 m
    across under a slate cap whose sides sweep up concave from flared, upturned eaves to a point,
    with a little finial."""
    m = METRE
    lantern = p.box(x, y, 1.2 * m, 1.5 * m, WHITEWASH, None)
    left, front, right, _ = lantern
    for a, b, tone in ((left, front, 0.6), (front, right, 0.5)):
        for drop in (0.4, 0.75, 1.1):
            _bar(
                p,
                _on_wall(a, b, 0.15, drop * m),
                _on_wall(a, b, 0.85, drop * m),
                0.16 * m,
                shade(WHITEWASH, tone),
            )
    cx, cy = (left[0] + right[0]) / 2, (left[1] + right[1]) / 2
    apex = (cx, cy - 2.8 * m)

    def eave(corner: Point) -> Point:
        return cx + (corner[0] - cx) * 1.7, cy + (corner[1] - cy) * 1.7 + 0.35 * m

    def sweep(a: Point) -> list[Point]:
        return [
            (a[0] + (apex[0] - a[0]) * (1 - (1 - s) ** 2), a[1] + (apex[1] - a[1]) * s**2)
            for s in np.linspace(0, 1, 7)
        ]

    l, f, r = eave(left), eave(front), eave(right)
    p.polygon([l, f, *sweep(f)[1:], *sweep(l)[::-1][1:-1]], STILL_SLATE)
    p.polygon([f, r, *sweep(r)[1:], *sweep(f)[::-1][1:-1]], shade(STILL_SLATE, 0.72))
    for corner in (l, f, r):
        _bar(p, corner, (corner[0], corner[1] - 0.3 * m), 0.15 * m, shade(STILL_SLATE, 1.4))
    _bar(p, apex, (apex[0], apex[1] - 0.9 * m), 0.12 * m, (60, 60, 64))
    p.ellipse(apex[0] - 0.2 * m, apex[1] - 1.1 * m, apex[0] + 0.2 * m, apex[1] - 0.7 * m, GILT)


def _still_peat(p: Painter, x: float, y: float) -> None:
    """A neat stack of cut peat 3.2 m long, 1.6 m deep and 1.2 m high on (x, y), built of dark
    turfs laid in courses like bricks with gaps between them, a few more turfs lying on top."""
    m = METRE
    gap = shade(PEAT, 0.6)
    stack = p.hall(x, y, 3.2 * m, 1.6 * m, 1.2 * m, gap, shade(PEAT, 1.2))
    left, front, right, back = stack
    for a, b, count, tone in ((left, front, 6, 1.05), (front, right, 3, 0.82)):
        for row in range(4):
            d0, d1 = row * 0.3 * m + 0.04 * m, (row + 1) * 0.3 * m - 0.04 * m
            shift = 0.5 * (row % 2)
            for k in range(count + 1):
                t0 = max(0.0, (k - shift) / count + 0.02)
                t1 = min(1.0, (k + 1 - shift) / count - 0.02)
                if t1 > t0:
                    _patch(p, a, b, (t0, t1), (d0, d1), shade(PEAT, tone * (1.0 + 0.1 * (k % 2))))
    for t, s in ((0.25, 0.3), (0.6, 0.55)):
        cx, cy = _on_wall(_on_wall(left, front, t, 0), _on_wall(back, right, t, 0), s, 0)
        p.hall(cx, cy, 0.55 * m, 0.3 * m, 0.25 * m, shade(PEAT, 1.1), shade(PEAT, 1.3))


def _still_window(
    p: Painter, a: Point, b: Point, length: float, t: float, wide: float, drop: tuple[float, float]
) -> None:
    """A small window `wide` across about `t` along the wall under the edge from `a` to `b`, which
    is `length` units long, from `drop[0]` to `drop[1]` down it: dark panes in a black frame with
    a white glazing bar across, over a white sill."""
    m = METRE
    half = wide / 2 / length
    edge = 0.12 * m / length
    _patch(p, a, b, (t - half - edge, t + half + edge), (drop[0] - 0.12 * m, drop[1]), TAR)
    _patch(p, a, b, (t - half, t + half), drop, STILL_GLASS)
    mid = (drop[0] + drop[1]) / 2
    _bar(p, _on_wall(a, b, t - half, mid), _on_wall(a, b, t + half, mid), 0.07 * m, WHITEWASH)
    _bar(p, _on_wall(a, b, t, drop[0]), _on_wall(a, b, t, drop[1]), 0.07 * m, WHITEWASH)
    _bar(
        p,
        _on_wall(a, b, t - half - edge, drop[1] + 0.1 * m),
        _on_wall(a, b, t + half + edge, drop[1] + 0.1 * m),
        0.18 * m,
        shade(WHITEWASH, 0.9),
    )


def distillery() -> Sprite:
    """A Highland whisky distillery whitewashed over a tarred black plinth: a long malt barn 18 by
    7 m under slate with two rows of small black-framed windows, whisky-amber doors and a black
    board lettered in gold; a malt kiln at its end under a steep slate roof crowned with a
    pagoda ventilator; a still house with its amber doors folded open on two onion-shaped copper
    pot stills gleaming over their fires, their lyne arms running off to a worm tub of oak beside
    it; a slim brick stack trailing peat smoke, a neat stack of cut peat, and oak casks in a
    pyramid along the barn and stood on end by the still house."""
    m = METRE
    p = Painter(44, 40, (19, 28))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 9) * m, (v - 2) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon(
        [at(-6, -3.5), at(16, -3.5), at(15, -6), at(25.5, -6), at(25.5, 9), at(-6, 9)], GRAVEL
    )
    _bed(p, _footing(*at(9, 5.5), 18 * m, 7 * m), margin=0.8)
    _bed(p, _footing(*at(-3, 5.5), 6 * m, 6 * m), margin=0.8)
    _bed(p, _footing(*at(19.5, -1), 7 * m, 6 * m), margin=0.8)
    p.shadow(13, 4, 4)
    sx, sy = at(21, 6)
    stack = p.hall(sx, sy, 1.3 * m, 1.3 * m, 20 * m * CHIMNEY_SHARE, BRICK, (52, 46, 46))
    _bands(p, stack[0], stack[1], stack[2], 0.8 * m, WHITEWASH, shade(WHITEWASH, 0.78))
    cap = stack[1][1] - 1.3 * m / 4
    p.soft(
        [
            (sx - 0.4 * m, cap - 2.2 * m, sx + 1.2 * m, cap - 0.8 * m),
            (sx + 0.6 * m, cap - 3.8 * m, sx + 2.6 * m, cap - 2.2 * m),
        ],
        (176, 178, 186),
        0.7,
        0.35,
    )

    kiln_high = 8 * m
    kiln = p.hall(*at(-3, 5.5), 6 * m, 6 * m, kiln_high, WHITEWASH, None)
    k_left, k_front, k_right, _ = kiln
    for a, b, tone in ((k_left, k_front, 1.0), (k_front, k_right, 0.78)):
        _patch(p, a, b, (0, 1), (kiln_high - 0.8 * m, kiln_high), shade(TAR, tone))
    _still_window(p, k_left, k_front, 6 * m, 0.5, 1 * m, (2 * m, 3.2 * m))
    _patch(p, k_left, k_front, (0.3, 0.7), (5.2 * m, kiln_high), WHISKY)
    _bar(
        p,
        _on_wall(k_left, k_front, 0.5, 5.2 * m),
        _on_wall(k_left, k_front, 0.5, kiln_high),
        0.06 * m,
        shade(WHISKY, 0.6),
    )
    apex = ((k_left[0] + k_right[0]) / 2, (k_left[1] + k_right[1]) / 2 - 5.5 * m)
    p.polygon([k_left, k_front, apex], STILL_SLATE, shade(STILL_SLATE, 0.6))
    p.polygon([k_front, k_right, apex], shade(STILL_SLATE, 0.72), shade(STILL_SLATE, 0.5))
    for f in (0.3, 0.6):
        p.line(
            [
                _on_wall(k_left, apex, f, 0),
                _on_wall(k_front, apex, f, 0),
                _on_wall(k_right, apex, f, 0),
            ],
            shade(STILL_SLATE, 0.82),
        )
    _still_pagoda(p, apex[0], apex[1] + 1.4 * m)

    length, depth, high = 18 * m, 7 * m, 7 * m
    top = p.hall(*at(9, 5.5), length, depth, high, WHITEWASH, None)
    left, front, right, _ = top
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (high - 0.8 * m, high), shade(TAR, tone))
        _patch(p, a, b, (0, 1), (0, 0.25 * m), shade(STILL_SLATE, tone))
    _stains(p, left, front, high - 0.8 * m, (0.12, 0.4, 0.66, 0.92), shade(WHITEWASH, 0.86), 1 * m)
    for k in range(7):
        t = 0.06 + 0.125 * k
        if k in (4, 5):
            continue
        _still_window(p, left, front, length, t, 0.9 * m, (1.4 * m, 2.5 * m))
        _still_window(p, left, front, length, t, 0.9 * m, (4.0 * m, 5.1 * m))
    for k in (4, 5):
        _still_window(p, left, front, length, 0.06 + 0.125 * k, 0.9 * m, (1.4 * m, 2.5 * m))
    _patch(p, left, front, (0.5, 0.75), (2.9 * m, 3.8 * m), TAR)
    for k in range(6):
        t0 = 0.52 + 0.036 * k
        _bar(
            p,
            _on_wall(left, front, t0, 3.35 * m),
            _on_wall(left, front, t0 + 0.024, 3.35 * m),
            0.3 * m,
            GILT,
        )
    _patch(p, left, front, (0.56, 0.69), (4.2 * m, high - 0.1 * m), shade(WHISKY, 0.85))
    _patch(p, left, front, (0.57, 0.68), (4.4 * m, high - 0.1 * m), WHISKY)
    _bar(
        p,
        _on_wall(left, front, 0.625, 4.4 * m),
        _on_wall(left, front, 0.625, high),
        0.06 * m,
        shade(WHISKY, 0.6),
    )
    p.gable(top, 3.5 * m, STILL_SLATE, WHITEWASH)
    _roof_rows(p, top, 3.5 * m, STILL_SLATE, (0.33, 0.66))
    _cask_stack(p, *at(2.8, 1.1), (4, 3, 2, 1))

    s_high, s_len, s_deep = 6.5 * m, 7 * m, 6 * m
    still = p.hall(*at(19.5, -1), s_len, s_deep, s_high, WHITEWASH, None)
    s_left, s_front, s_right, _ = still
    for a, b, tone in ((s_left, s_front, 1.0), (s_front, s_right, 0.78)):
        _patch(p, a, b, (0, 1), (s_high - 0.8 * m, s_high), shade(TAR, tone))
    opening = _round_head(s_left, s_front, s_len, 0.5, 5.4 * m, 0.7 * m, s_high)
    p.polygon(opening, (40, 34, 32))
    for t0, t1 in ((0.0, 0.1), (0.9, 1.0)):
        _patch(p, s_left, s_front, (t0, t1), (2.6 * m, s_high - 0.1 * m), WHISKY)
        _bar(
            p,
            _on_wall(s_left, s_front, (t0 + t1) / 2, 2.6 * m),
            _on_wall(s_left, s_front, (t0 + t1) / 2, s_high),
            0.06 * m,
            shade(WHISKY, 0.6),
        )
    for t, reach in ((0.3, -1.1 * m), (0.7, 1.3 * m)):
        _still_pot(p, *_on_wall(s_left, s_front, t, s_high - 0.15 * m), reach)
    _still_window(p, s_front, s_right, s_deep, 0.5, 1 * m, (1.6 * m, 2.8 * m))
    p.gable(still, 3 * m, STILL_SLATE, WHITEWASH)
    _roof_rows(p, still, 3 * m, STILL_SLATE, (0.5,))
    wx, wy = at(24.4, -1.4)
    pipe_from = _on_wall(s_front, s_right, 0.3, 2.2 * m)
    _bar(p, pipe_from, (wx - 0.3 * m, wy - 1.9 * m), 0.2 * m, KETTLE)
    _still_worm(p, wx, wy)
    _still_peat(p, *at(-4, -0.6))
    for u, v in ((14.4, -3.4), (15.4, -4.6)):
        _cask(p, *at(u, v), "up")
    return p.sprite()


PLASTER: RGB = (240, 236, 226)
NAMAKO: RGB = (84, 88, 98)
CHARRED: RGB = (44, 42, 44)
KAWARA: RGB = (72, 76, 88)
INDIGO: RGB = (38, 56, 112)
SUGI: RGB = (88, 116, 52)
KOMO: RGB = (226, 206, 150)
SAKE_RED: RGB = (192, 40, 40)
LATTICE: RGB = (176, 132, 84)


def _cask_straw(p: Painter, x: float, y: float) -> None:
    """A komodaru standing on (x, y): a cask of sake 0.8 m across and 0.85 m high wrapped in
    straw matting with a pale straw lid, bound top and bottom with straw rope laced down its
    sides, and branded on its front with a red label lettered in black and white."""
    m = METRE
    r, h = 0.56 * m, 0.85 * m
    _turned(p, x, y, [(0, r), (h, r)], KOMO, shade(KOMO, 1.1))

    def round_at(a: float, z: float) -> Point:
        return x + r * math.cos(a), y - z + r / 2 * math.sin(a)

    rope = shade(KOMO, 0.66)
    arc = np.linspace(0.05, math.pi - 0.05, 9)
    for z in (0.1 * m, 0.76 * m):
        for a0, a1 in itertools.pairwise(arc):
            _bar(p, round_at(a0, z), round_at(a1, z), 0.06 * m, rope)
    for a in (0.35, 2.8):
        _bar(p, round_at(a, 0.1 * m), round_at(a, 0.76 * m), 0.05 * m, rope)
    label = np.linspace(1.05, 2.1, 6)
    p.polygon(
        [round_at(a, 0.68 * m) for a in label] + [round_at(a, 0.2 * m) for a in label[::-1]],
        SAKE_RED,
    )
    for a, z0, z1, tone in ((1.45, 0.6, 0.3, (30, 26, 26)), (1.75, 0.6, 0.38, PLASTER)):
        _bar(p, round_at(a, z0 * m), round_at(a, z1 * m), 0.07 * m, tone)


def _cask_straw_stack(p: Painter, x: float, y: float, base: int) -> None:
    """Komodaru (_cask_straw) stood on end in a row of `base` along the lit walls from (x, y),
    with one fewer standing on top of them over the gaps."""
    gap = 0.82
    p.soft(
        [(*_cask_at(x, y, -0.6, 0.5), *_cask_at(x, y, (base - 1) * gap + 1, -0.6))],
        (34, 30, 24),
        0.4,
        0.2,
    )
    for level in range(2):
        for k in range(base - level):
            _cask_straw(p, *_cask_at(x, y, (k + 0.5 * level) * gap, 0, 0.86 * level))


def _sake_namako(
    p: Painter, a: Point, b: Point, length: float, drop: tuple[float, float], tone: float
) -> None:
    """A namako wall along the foot of the wall under the edge from `a` to `b`, which is `length`
    units long, from `drop[0]` to `drop[1]` down it: grey tiles set on the diagonal, their joints
    raised in white plaster in a lattice of diamonds, under a black timber rail."""
    m = METRE
    d0, d1 = drop
    _patch(p, a, b, (0, 1), drop, shade(NAMAKO, tone))
    joint, side = shade(PLASTER, tone), 1.0 * m
    for c in np.arange(-(d1 - d0), length + (d1 - d0), side):
        for sign in (1, -1):
            if sign > 0:
                lo, hi = max(d0, d0 - c), min(d1, d0 + length - c)
            else:
                lo, hi = max(d0, d0 + c - length), min(d1, d0 + c)
            if hi - lo < 0.05 * m:
                continue
            ends = [(c + sign * (d - d0), d) for d in (lo, hi)]
            _bar(
                p,
                _on_wall(a, b, ends[0][0] / length, ends[0][1]),
                _on_wall(a, b, ends[1][0] / length, ends[1][1]),
                0.13 * m,
                joint,
            )
    _bar(p, _on_wall(a, b, 0, d0), _on_wall(a, b, 1, d0), 0.22 * m, shade(CHARRED, tone))


def _sake_roof(p: Painter, top: Top, rise: float) -> None:
    """A heavy roof of dark kawara tiles on a block's top: rows of round tiles running down its
    lit side from a thick ridge capped at both ends with raised ridge tiles."""
    m = METRE
    p.gable(top, rise, KAWARA, PLASTER)
    left, front, _, _ = top
    count = max(4, round(math.dist(left, front) / (0.55 * m)))
    for k in range(1, count):
        t = k / count
        _bar(
            p,
            _on_roof(top, rise, t, 0.02),
            _on_roof(top, rise, t, 0.96),
            0.08 * m,
            shade(KAWARA, 0.78),
        )
    _bar(p, _on_roof(top, rise, 0, 0), _on_roof(top, rise, 1, 0), 0.22 * m, shade(KAWARA, 0.7))
    near, far = _ridge(top, rise)
    _bar(p, (near[0], near[1] - 0.2 * m), (far[0], far[1] - 0.2 * m), 0.55 * m, shade(KAWARA, 0.62))
    for end in (near, far):
        p.polygon(
            [
                (end[0] - 0.35 * m, end[1]),
                (end[0] + 0.35 * m, end[1]),
                (end[0] + 0.45 * m, end[1] - 0.8 * m),
                (end[0] - 0.25 * m, end[1] - 0.95 * m),
            ],
            shade(KAWARA, 0.55),
        )


def _sake_ball(p: Painter, x: float, y: float) -> None:
    """A sugidama hung by a cord from the eaves at (x, y): a ball of fresh green cedar twigs 1.1 m
    across, bristling at its rim and lit on its upper left."""
    m = METRE
    _bar(p, (x, y), (x, y + 0.5 * m), 0.06 * m, (60, 50, 40))
    cx, cy, r = x, y + 1.05 * m, 0.55 * m
    for a in np.linspace(0, 2 * math.pi, 14)[:-1]:
        tx, ty = cx + r * 1.02 * math.cos(a), cy + r * 1.02 * math.sin(a)
        p.blob(tx - 0.14 * m, ty - 0.14 * m, tx + 0.14 * m, ty + 0.14 * m, shade(SUGI, 0.8))
    p.blob(cx - r, cy - r, cx + r, cy + r, SUGI)
    p.blob(cx - 0.1 * m, cy - 0.05 * m, cx + r * 0.95, cy + r * 0.95, shade(SUGI, 0.82))
    p.blob(cx - r * 0.75, cy - r * 0.75, cx + 0.05 * m, cy, shade(SUGI, 1.22))
    for dx, dy in ((-0.2, 0.15), (0.2, -0.25), (0.15, 0.3), (-0.3, -0.1)):
        tx, ty = cx + dx * m, cy + dy * m
        p.blob(tx - 0.06 * m, ty - 0.06 * m, tx + 0.06 * m, ty + 0.06 * m, shade(SUGI, 0.7))


def _sake_noren(p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float) -> None:
    """An indigo noren hung from a rod over a doorway on the wall under the edge from `a` to `b`,
    from `span[0]` to `span[1]` along it and `drop` down it: three panels a metre long with slits
    between them and a white crest on the middle one."""
    m = METRE
    t0, t1 = span
    _bar(p, _on_wall(a, b, t0 - 0.02, drop), _on_wall(a, b, t1 + 0.02, drop), 0.12 * m, LATTICE)
    _patch(p, a, b, span, (drop + 0.05 * m, drop + 1.1 * m), INDIGO)
    for k in (1, 2):
        t = t0 + (t1 - t0) * k / 3
        _bar(
            p,
            _on_wall(a, b, t, drop + 0.25 * m),
            _on_wall(a, b, t, drop + 1.1 * m),
            0.07 * m,
            shade(INDIGO, 0.55),
        )
    cx, cy = _on_wall(a, b, (t0 + t1) / 2, drop + 0.55 * m)
    p.blob(cx - 0.2 * m, cy - 0.2 * m, cx + 0.2 * m, cy + 0.2 * m, PLASTER)


def sake_brewery() -> Sprite:
    """A sake brewery: a long kura storehouse 18 by 7 m with walls 7 m high of white plaster over
    a namako base of grey tiles in a lattice of white joints, small high windows behind thick
    white shutters, under a heavy roof of dark kawara tiles with a thick ridge; before it a shop
    of black charred timber under a white band and a tiled roof, an indigo noren over its
    lattice door and a green sugidama hung under its eaves for the new brew; and straw-wrapped
    komodaru with red labels stood in a neat stack before the kura and by the shop."""
    m = METRE
    p = Painter(40, 34, (17, 23))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 6) * m, (v - 3) * m)

    p.polygon([at(-7, -4.5), at(12, -4.5), at(12, 1.5), at(-7, 1.5)], (204, 196, 176))
    _bed(p, _footing(*at(9, 5.5), 18 * m, 7 * m), margin=0.6)
    p.shadow(12, 4, 3)
    length, depth, high = 18 * m, 7 * m, 7 * m
    top = p.hall(*at(9, 5.5), length, depth, high, PLASTER, None)
    left, front, right, _ = top
    _sake_namako(p, left, front, length, (high - 2.4 * m, high), 1.0)
    _sake_namako(p, front, right, depth, (high - 2.4 * m, high), 0.78)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _bar(
            p,
            _on_wall(a, b, 0, 0.15 * m),
            _on_wall(a, b, 1, 0.15 * m),
            0.3 * m,
            shade(CHARRED, tone),
        )
    for t in (0.42, 0.6, 0.78, 0.94):
        _patch(p, left, front, (t - 0.035, t + 0.035), (1.2 * m, 2.4 * m), shade(PLASTER, 0.72))
        _patch(p, left, front, (t - 0.025, t + 0.025), (1.35 * m, 2.25 * m), (40, 38, 42))
        for k in (-1, 0, 1):
            s = t + 0.012 * k
            _bar(
                p,
                _on_wall(left, front, s, 1.35 * m),
                _on_wall(left, front, s, 2.25 * m),
                0.06 * m,
                (110, 106, 100),
            )
    _patch(p, front, right, (0.4, 0.6), (1.2 * m, 2.6 * m), shade(PLASTER, 0.6))
    _sake_roof(p, top, 3.2 * m)

    _cask_straw_stack(p, *at(5.6, 1.0), 5)

    s_len, s_deep, s_high = 7 * m, 4.8 * m, 4.6 * m
    shop = p.hall(*at(-2.5, -0.6), s_len, s_deep, s_high, CHARRED, None)
    h_left, h_front, h_right, _ = shop
    for a, b, length_, tone in ((h_left, h_front, s_len, 1.0), (h_front, h_right, s_deep, 0.78)):
        _patch(p, a, b, (0, 1), (0, 0.9 * m), shade(PLASTER, tone))
        for k in range(1, round(length_ / (0.35 * m))):
            t = k / round(length_ / (0.35 * m))
            _bar(
                p,
                _on_wall(a, b, t, 0.9 * m),
                _on_wall(a, b, t, s_high),
                0.04 * m,
                shade(CHARRED, 1.5 * tone),
            )
        _bar(
            p,
            _on_wall(a, b, 0, 0.95 * m),
            _on_wall(a, b, 1, 0.95 * m),
            0.12 * m,
            shade(LATTICE, 0.6 * tone + 0.2),
        )
    _patch(p, h_left, h_front, (0.4, 0.72), (1.9 * m, s_high), shade(LATTICE, 0.6))
    for k in range(9):
        s = 0.41 + 0.0375 * k
        _bar(
            p,
            _on_wall(h_left, h_front, s, 1.9 * m),
            _on_wall(h_left, h_front, s, s_high),
            0.06 * m,
            LATTICE,
        )
    _sake_noren(p, h_left, h_front, (0.42, 0.7), 1.6 * m)
    _patch(p, h_left, h_front, (0.08, 0.3), (1.6 * m, 3.2 * m), (40, 34, 30))
    for k in range(7):
        s = 0.09 + 0.033 * k
        _bar(
            p,
            _on_wall(h_left, h_front, s, 1.6 * m),
            _on_wall(h_left, h_front, s, 3.2 * m),
            0.06 * m,
            LATTICE,
        )
    _sake_roof(p, shop, 2.2 * m)
    _sake_ball(p, *_on_wall(h_left, h_front, 0.86, 0.25 * m))
    for u, v in ((2.2, -3.4), (3.1, -2.6)):
        _cask_straw(p, *at(u, v))
    return p.sprite()


TAN_OCHRE: RGB = (206, 162, 98)
TAN_OAK: RGB = (94, 68, 48)
TAN_SLAT: RGB = (140, 106, 74)
TAN_ROOF: RGB = (124, 70, 50)
TAN_MUD: RGB = (104, 86, 62)
TAN_LIME: RGB = (234, 234, 224)
TAN_BARK: RGB = (132, 76, 46)
TAN_HIDES: tuple[RGB, ...] = ((232, 204, 160), (204, 152, 96), (162, 100, 60))
TAN_LIQUORS: tuple[RGB, ...] = ((82, 52, 32), (112, 72, 42), (146, 100, 58))


def _tan_louvres(
    p: Painter,
    a: Point,
    b: Point,
    bays: int,
    span: tuple[float, float],
    tone: float,
) -> None:
    """A storey of drying loft on the wall under the edge from `a` to `b`, from `span[0]` to `span[1]` down it: oak posts dividing it into `bays`, and between them
    slatted louvres with dark gaps, pale hides hung to dry showing through them."""
    m = METRE
    head, foot = span
    _patch(p, a, b, (0, 1), span, shade((50, 38, 30), tone))
    for k in range(bays):
        t0, t1 = k / bays, (k + 1) / bays
        for j in range(3):
            s0 = t0 + (t1 - t0) * (0.12 + 0.28 * j)
            s1 = s0 + (t1 - t0) * 0.22
            hide = TAN_HIDES[(k + j) % len(TAN_HIDES)]
            _patch(
                p,
                a,
                b,
                (s0, s1),
                (head + 0.3 * m, foot - 0.4 * m - 0.3 * m * ((k + j) % 2)),
                shade(hide, tone * 0.8),
            )
    rows = max(2, round((foot - head) / (0.42 * m)))
    for r in range(rows):
        d0 = head + (foot - head) * r / rows
        d1 = d0 + (foot - head) / rows * 0.58
        _patch(p, a, b, (0, 1), (d0, d1), shade(TAN_SLAT, tone))
        _bar(
            p,
            _on_wall(a, b, 0, d0 + 0.03 * m),
            _on_wall(a, b, 1, d0 + 0.03 * m),
            0.05 * m,
            shade(TAN_SLAT, tone * 1.25),
        )
    for k in range(bays + 1):
        t = k / bays
        _bar(p, _on_wall(a, b, t, head), _on_wall(a, b, t, foot), 0.3 * m, shade(TAN_OAK, tone))
    for d in (head, foot):
        _bar(p, _on_wall(a, b, 0, d), _on_wall(a, b, 1, d), 0.3 * m, shade(TAN_OAK, tone))


def _tan_pit(
    p: Painter,
    at: Callable[[float, float], Point],
    u: tuple[float, float],
    v: tuple[float, float],
    liquor: RGB,
) -> None:
    """A tanning pit sunk in the ground from `u[0]` to `u[1]` along and `v[0]` to `v[1]` across,
    where `at` places a spot on the ground: a kerb of oak planks round it and the dark liquor
    of bark, or the milky lime, standing a little below the kerb."""
    m = METRE
    (u0, u1), (v0, v1) = u, v
    kerb = 0.28
    p.polygon(
        [
            at(u0 - kerb, v0 - kerb),
            at(u1 + kerb, v0 - kerb),
            at(u1 + kerb, v1 + kerb),
            at(u0 - kerb, v1 + kerb),
        ],
        (150, 118, 84),
    )
    p.polygon([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)], (46, 36, 30))
    drop = 0.35 * m
    near_l, near_r, far_r, far_l = at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)
    p.polygon(
        [
            near_l,
            near_r,
            (far_r[0], far_r[1] + drop),
            (far_l[0], far_l[1] + drop),
        ],
        liquor,
    )
    sheen = shade(liquor, 1.3 if liquor != TAN_LIME else 0.94)
    sx, sy = at(u0 + (u1 - u0) * 0.3, v0 + (v1 - v0) * 0.55)
    p.blob(
        sx - 0.5 * m, sy - 0.08 * m + drop * 0.5, sx + 0.3 * m, sy + 0.12 * m + drop * 0.5, sheen
    )
    _bar(p, near_l, near_r, 0.12 * m, (176, 142, 102))


def _tan_hide(
    p: Painter,
    at: Callable[[float, float], Point],
    cs: float,
    cd: float,
    hs: float,
    hd: float,
    hide: RGB,
) -> list[Point]:
    """The outline of a hide spread flat, its middle at (`cs`, `cd`) on a face where `at(s, d)`
    gives the point `s` across and `d` down it, `hs` and `hd` its half width and height: a
    broad back, four legs reaching to the corners, the neck at its top and the tail at its foot,
    lit pale along its spine. Returns the tips of its legs."""
    half = [
        (0.0, -1.0),
        (0.2, -0.94),
        (0.36, -0.72),
        (0.78, -0.98),
        (0.92, -0.82),
        (0.62, -0.5),
        (0.66, -0.05),
        (0.62, 0.34),
        (0.86, 0.62),
        (0.96, 0.86),
        (0.7, 0.86),
        (0.4, 0.74),
        (0.16, 0.92),
        (0.06, 1.0),
    ]
    outline = half + [(-x, y) for x, y in reversed(half)]
    p.polygon([at(cs + x * hs, cd + y * hd) for x, y in outline], hide, shade(hide, 0.62))
    p.polygon(
        [
            at(cs + x * hs * 0.35, cd + y * hd * 0.8)
            for x, y in ((0, -1), (1, -0.6), (1, 0.7), (0, 1), (-1, 0.7), (-1, -0.6))
        ],
        shade(hide, 1.1),
    )
    return [
        at(cs + x * hs, cd + y * hd)
        for x, y in ((0.85, -0.92), (-0.85, -0.92), (0.92, 0.86), (-0.92, 0.86))
    ]


def _tan_frame(
    p: Painter, x: float, y: float, wide: float, tall: float, hide: RGB, slope: float = 0.5
) -> None:
    """A hide laced with cords into an upright oak frame `wide` across and `tall` high standing
    on (x, y), its left foot, along a face running `slope` down for every unit across, to dry
    and stretch, dripping onto the ground."""
    m = METRE
    tone = 1.0 if slope > 0 else 0.8

    def at(s: float, d: float) -> Point:
        return x + s, y + s * slope - tall + d

    p.blob(
        x + wide * 0.15,
        y + wide * slope * 0.5 - 0.3 * m,
        x + wide * 0.85,
        y + wide * slope * 0.5 + 0.4 * m,
        (84, 74, 52),
        150,
    )
    rail = 0.18 * m
    for s in (0, wide):
        _bar(p, at(s, -0.25 * m), at(s, tall), rail, shade(TAN_OAK, tone * 1.1))
    for d in (0.1 * m, tall - 0.45 * m):
        _bar(p, at(-0.15 * m, d), at(wide + 0.15 * m, d), rail, shade(TAN_OAK, tone * 1.2))
    tips = _tan_hide(p, at, wide / 2, tall * 0.48, wide * 0.4, tall * 0.36, shade(hide, tone))
    cord = (60, 48, 40)
    corners = [at(wide, 0.1 * m), at(0, 0.1 * m), at(wide, tall - 0.45 * m), at(0, tall - 0.45 * m)]
    for tip, corner in zip(tips, corners, strict=True):
        _bar(p, tip, corner, 0.05 * m, cord)
    for s in (0.25, 0.5, 0.75):
        _bar(p, at(wide * s, tall * 0.48 - tall * 0.34), at(wide * s, 0.1 * m), 0.04 * m, cord)
        _bar(
            p,
            at(wide * s, tall * 0.48 + tall * 0.33),
            at(wide * s, tall - 0.45 * m),
            0.04 * m,
            cord,
        )


def _tan_bark(p: Painter, x: float, y: float, half: float, tall: float) -> None:
    """A heap of oak bark for the tan liquor, `half` across either way and `tall` high on
    (x, y): a rounded mound of rusty brown lit on its left, rough curls of bark lying over it
    and slipped round its foot."""
    m = METRE
    p.blob(
        x - half * 1.2, y - half * 0.35, x + half * 1.25, y + half * 0.5, shade(TAN_BARK, 0.62), 170
    )
    profile = [
        (0.0, half),
        (tall * 0.35, half * 0.82),
        (tall * 0.7, half * 0.52),
        (tall * 0.92, half * 0.22),
        (tall, 0.0),
    ]
    split = math.acos(1 / 3)
    lit_arc = [
        (x + half * math.cos(a), y + half / 2 * math.sin(a)) for a in np.linspace(split, math.pi, 6)
    ]
    dark_arc = [
        (x + half * math.cos(a), y + half / 2 * math.sin(a)) for a in np.linspace(0, split, 4)
    ]
    left = [(x - w, y - h) for h, w in profile]
    middle = [(x + w / 3, y - h) for h, w in profile]
    right = [(x + w, y - h) for h, w in profile]
    p.polygon(lit_arc + left + middle[::-1], TAN_BARK)
    p.polygon(dark_arc + middle + right[::-1], shade(TAN_BARK, 0.74))
    for k in range(16):
        h = tall * (0.06 + 0.8 * ((k * 5) % 16) / 16)
        w = half * (1 - h / tall) ** 0.8
        across = ((k * 7) % 11) / 10 * 2 - 1
        px, py = x + w * across * 0.85, y - h + w * 0.25 * (1 - across * across)
        tone = 1.25 if across < 0.2 else 0.95
        _bar(
            p,
            (px - 0.28 * m, py + 0.08 * m),
            (px + 0.28 * m, py - 0.06 * m),
            0.13 * m,
            shade(TAN_BARK, tone),
        )
    for dx in (-1.1, 0.9, 1.2):
        px, py = x + dx * half, y + 0.25 * half
        _bar(p, (px - 0.3 * m, py), (px + 0.3 * m, py - 0.1 * m), 0.13 * m, shade(TAN_BARK, 0.9))


def _tan_cart(p: Painter, x: float, y: float) -> None:
    """A two-wheeled cart 3.2 m long at (x, y) along the lit walls, piled with raw hides still in
    their hair, red-brown, black and white, one hanging over its side, its shafts down on the
    ground ahead of it."""
    m = METRE
    p.blob(x - 2.4 * m, y - 0.9 * m, x + 2.6 * m, y + 1.2 * m, (60, 56, 52), 100)
    for v in (-0.45, 0.45):
        sx, sy = _iso(x, y, 1.6 * m, v * m)
        _bar(p, (sx, sy - 0.9 * m), _iso(sx, sy, 2.4 * m, 0), 0.14 * m, shade(TAN_OAK, 1.2))
    body = p.hall(x, y - 0.9 * m, 3.2 * m, 1.7 * m, 0.6 * m, (128, 92, 60), (96, 70, 48))
    left, front, _, _ = body
    for t in (0.33, 0.66):
        _bar(p, _on_wall(left, front, t, 0), _on_wall(left, front, t, 0.6 * m), 0.08 * m, TAN_OAK)
    pile = ((150, 86, 56), (62, 50, 44), (196, 130, 82), (226, 214, 196), (122, 66, 44))
    for k, colour in enumerate(pile):
        along, across = 0.5 + 0.95 * (k % 3), 0.5 + 0.7 * (k // 3)
        cx, cy = _iso(*left, along * m, across * m)
        lift = (0.15 + 0.3 * (k // 3)) * m
        p.blob(cx - 1.1 * m, cy - lift - 0.55 * m, cx + 1.1 * m, cy - lift + 0.45 * m, colour)
        p.blob(
            cx - 0.7 * m,
            cy - lift - 0.5 * m,
            cx + 0.3 * m,
            cy - lift - 0.1 * m,
            shade(colour, 1.15),
        )
    drape = (172, 104, 64)
    a, b = _on_wall(left, front, 0.4, 0), _on_wall(left, front, 0.78, 0)
    p.polygon(
        [
            a,
            b,
            (b[0], b[1] + 0.8 * m),
            (b[0] - 0.4 * m, b[1] + 1.0 * m),
            (a[0] + 0.3 * m, a[1] + 0.9 * m),
        ],
        drape,
        shade(drape, 0.6),
    )
    wx, wy = _on_wall(left, front, 0.5, 0.7 * m)
    p.ellipse(wx - 0.75 * m, wy - 0.75 * m, wx + 0.75 * m, wy + 0.75 * m, (60, 48, 40))
    p.ellipse(wx - 0.55 * m, wy - 0.55 * m, wx + 0.55 * m, wy + 0.55 * m, (128, 92, 60))
    for k in range(3):
        turn = k * math.pi / 3
        dx, dy = 0.5 * m * math.cos(turn), 0.5 * m * math.sin(turn)
        _bar(p, (wx - dx, wy - dy), (wx + dx, wy + dy), 0.09 * m, (70, 52, 38))
    p.blob(wx - 0.14 * m, wy - 0.14 * m, wx + 0.14 * m, wy + 0.14 * m, (44, 38, 34))


def tannery() -> Sprite:
    """A tanyard. The tannery, 20 by 10 m, has a beamhouse of ochre-washed walls with arched
    doors and barred windows, and jettied over it a tall timber storey of louvred drying lofts,
    two floors of slatted oak with pale hides glimpsed hanging through them, under a steep
    chestnut-tiled roof with two louvred vents on its ridge. Before it twelve tanning pits
    are sunk in the trodden mud in two rows, kerbed in oak planks and brimming with bark liquor
    from pale to near black, the two by the door milky white with lime. Hides stretched to dry
    in upright oak frames stand in a row beside it, a cart piled with raw hides waits at the
    door, and heaps of oak bark for the liquor rise behind."""
    m = METRE
    p = Painter(46, 36, (23, 23))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 11.5) * m, (v - 1.5) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    length, depth = 20 * m, 10 * m
    low, high, rise = 4.2 * m, 9.6 * m, 4.2 * m
    p.polygon(
        [at(-1, -7.4), at(25, -7.4), at(25, 0.6), at(21, 0.6), at(21, -0.6), at(-1, -0.6)], TAN_MUD
    )
    p.soft([(*at(-1.5, -5), *at(4, -1.5))], TAN_LIME, 0.35, 0.6)
    p.shadow(15, 5, 3)
    _tan_bark(p, *at(23.5, 11.5), 3.0 * m, 3.0 * m)
    _tan_bark(p, *at(26.5, 9), 2.2 * m, 2.0 * m)
    foot = _footing(*at(10, 5), length, depth)
    _bed(p, foot)
    beam = p.hall(*at(10, 5), length, depth, low, TAN_OCHRE, None)
    left, front, right, _ = beam
    _plinth(p, beam, low, TAN_OCHRE, 0.6 * m)
    _stains(p, left, front, low, (0.04, 0.16, 0.5, 0.62, 0.95), shade(TAN_OCHRE, 0.72), 1.4 * m)
    for t in (0.14, 0.62):
        p.polygon(_round_head(left, front, length, t, 3 * m, 0.7 * m, low), shade(TAN_OCHRE, 0.78))
        p.polygon(_round_head(left, front, length, t, 2.6 * m, 0.9 * m, low), (52, 40, 34))
        _patch(p, left, front, (t - 0.06, t - 0.002), (1.9 * m, low), (110, 74, 48))
    for t in (0.3, 0.42, 0.78, 0.9):
        _patch(p, left, front, (t - 0.03, t + 0.03), (1.0 * m, 2.6 * m), (52, 44, 40))
        for s in (t - 0.012, t + 0.012):
            _bar(
                p,
                _on_wall(left, front, s, 1.0 * m),
                _on_wall(left, front, s, 2.6 * m),
                0.07 * m,
                (40, 34, 32),
            )
    _patch(p, front, right, (0.35, 0.65), (1.0 * m, low), (52, 40, 34))
    loft = p.hall(*up(10, 5, 4.2), length + 0.8 * m, depth + 0.8 * m, high - low, TAN_OAK, None)
    l_left, l_front, l_right, _ = loft
    mid = (high - low) / 2
    for a, b, bays, tone in ((l_left, l_front, 8, 1.0), (l_front, l_right, 4, 0.78)):
        _tan_louvres(p, a, b, bays, (0.2 * m, mid), tone)
        _tan_louvres(p, a, b, bays, (mid, high - low - 0.1 * m), tone)
    p.gable(loft, rise, TAN_ROOF, TAN_OAK)
    _roof_rows(p, loft, rise, TAN_ROOF, (0.2, 0.4, 0.6, 0.8))
    back, front_ridge = _ridge(loft, rise)
    for along in (0.3, 0.7):
        vx, vy = _on_wall(back, front_ridge, along, 0)
        vent = p.hall(vx, vy + 0.5 * m, 4.5 * m, 2 * m, 1.9 * m, (40, 32, 28), None)
        v_left, v_front, v_right, _ = vent
        for a, b, tone in ((v_left, v_front, 1.0), (v_front, v_right, 0.78)):
            for r in range(4):
                d = (0.25 + 0.3 * r) * m
                _patch(p, a, b, (0, 1), (d, d + 0.17 * m), shade(TAN_SLAT, tone * 1.1))
        p.gable(vent, 1.0 * m, shade(TAN_ROOF, 0.9), (40, 32, 28))
    _tan_cart(p, *at(5.3, -1.5))
    for k in range(2):
        for j in range(6):
            u0 = 2.0 + 3.1 * j
            liquor = TAN_LIME if j == 0 else TAN_LIQUORS[(j + k) % 3]
            v0 = -4.4 - 2.7 * k
            _tan_pit(p, at, (u0, u0 + 2.3), (v0, v0 + 1.7), liquor)
    for k, (u, v) in enumerate(((20.2, -6.8), (21.4, -3.3), (22.6, 0.2), (23.8, 3.7))):
        _tan_frame(p, *at(u, v), 3.6 * m, 3.7 * m, TAN_HIDES[k % 3])
    return p.sprite()


SHOE_BRICK: RGB = (180, 100, 66)
SHOE_TRIM: RGB = (238, 222, 186)
SHOE_SLATE: RGB = (78, 74, 84)
SHOE_LEATHER: RGB = (150, 96, 58)
SHOE_CARTON: RGB = (206, 176, 128)
SHOE_LAMPLIT: RGB = (226, 196, 134)


def _shoe_window(
    p: Painter,
    a: Point,
    b: Point,
    length: float,
    t: float,
    wide: float,
    span: tuple[float, float],
    tone: float = 1.0,
) -> None:
    """A tall factory window `wide` across about `t` along the wall under the edge from `a` to
    `b`, which is `length` units long, from `span[0]` down the wall to `span[1]`: a cream lintel
    with a keystone and a cream sill, small panes in an iron frame, and through the lower ones a
    lamplit room with a bench of stitching machines along it."""
    m = METRE
    head, foot = span
    half = wide / 2 / length
    trim, glass = shade(SHOE_TRIM, tone), shade((58, 66, 82), tone)
    _patch(p, a, b, (t - half, t + half), span, glass)
    room = head + (foot - head) * 0.38
    _patch(p, a, b, (t - half, t + half), (room, foot), shade(SHOE_LAMPLIT, tone))
    bench = head + (foot - head) * 0.8
    _patch(p, a, b, (t - half, t + half), (bench, foot), shade((92, 60, 42), tone))
    for k in range(2):
        s = t - half + 2 * half * (0.28 + 0.44 * k)
        x, y = _on_wall(a, b, s, bench)
        p.polygon(
            [
                (x - 0.22 * m, y),
                (x - 0.22 * m, y - 0.3 * m),
                (x + 0.22 * m, y - 0.38 * m),
                (x + 0.22 * m, y),
            ],
            shade((46, 42, 44), tone),
        )
    bar = shade((40, 40, 46), tone)
    _bar(p, _on_wall(a, b, t, head), _on_wall(a, b, t, foot), 0.08 * m, bar)
    for k in (1, 2):
        d = head + (foot - head) * k / 3
        _bar(p, _on_wall(a, b, t - half, d), _on_wall(a, b, t + half, d), 0.07 * m, bar)
    lip = 0.25 * m / length
    _patch(p, a, b, (t - half - lip, t + half + lip), (head - 0.45 * m, head), trim)
    _patch(
        p,
        a,
        b,
        (t - 0.12 * m / length, t + 0.12 * m / length),
        (head - 0.55 * m, head + 0.1 * m),
        shade(trim, 1.04),
    )
    _patch(p, a, b, (t - half - lip * 0.6, t + half + lip * 0.6), (foot, foot + 0.22 * m), trim)


def _shoe_boot(p: Painter, x: float, y: float, tall: float) -> None:
    """A gilt boot `tall` high standing on (x, y), turned side on to the viewer with its toe to
    the right: a laced shaft, a dark sole and a heel, lit along its back and shaft."""
    w = tall
    outline = (
        (0.04, 0.0),
        (0.46, 0.0),
        (0.46, 0.5),
        (0.62, 0.58),
        (0.84, 0.66),
        (0.97, 0.75),
        (1.0, 0.84),
        (0.97, 0.9),
        (0.4, 0.9),
        (0.32, 0.86),
        (0.25, 0.9),
        (0.25, 1.0),
        (0.0, 1.0),
        (0.0, 0.82),
        (0.05, 0.58),
        (0.0, 0.32),
    )
    left = x - w * 0.5

    def pt(s: float, d: float) -> Point:
        return left + s * w, y - tall + d * tall

    p.polygon([pt(s, d) for s, d in outline], GILT, shade(GILT, 0.55))
    p.polygon([pt(0.4, 0.9), pt(0.97, 0.9), pt(0.99, 0.84), pt(1.0, 0.95), pt(0.4, 0.96)], OXBLOOD)
    p.polygon([pt(0.0, 0.9), pt(0.25, 0.9), pt(0.25, 1.0), pt(0.0, 1.0)], OXBLOOD)
    p.polygon([pt(0.04, 0.0), pt(0.46, 0.0), pt(0.46, 0.08), pt(0.03, 0.08)], shade(GILT, 0.78))
    p.polygon([pt(0.07, 0.12), pt(0.16, 0.12), pt(0.14, 0.8), pt(0.07, 0.8)], shade(GILT, 1.25))
    for k in range(4):
        d = 0.18 + 0.1 * k
        _bar(p, pt(0.36, d), pt(0.46, d + 0.04), tall * 0.025, OXBLOOD)


def _shoe_cartons(p: Painter, x: float, y: float, along: int, across: int, layers: int) -> None:
    """Cartons of shoes, each 0.6 by 0.4 m and 0.3 m high, stacked `layers` deep on (x, y) in a
    block `along` of them by `across`, a label on each end."""
    m = METRE
    long, wide, high = 0.6 * m, 0.4 * m, 0.3 * m
    left, front, right, _ = p.hall(
        x, y, along * long, across * wide, layers * high, SHOE_CARTON, shade(SHOE_CARTON, 1.12)
    )
    edge = shade(SHOE_CARTON, 0.7)
    for k in range(1, layers):
        _bar(
            p,
            _on_wall(left, front, 0, k * high),
            _on_wall(left, front, 1, k * high),
            0.04 * m,
            edge,
        )
        _bar(
            p,
            _on_wall(front, right, 0, k * high),
            _on_wall(front, right, 1, k * high),
            0.04 * m,
            shade(edge, 0.8),
        )
    for k in range(1, along):
        _bar(
            p,
            _on_wall(left, front, k / along, 0),
            _on_wall(left, front, k / along, layers * high),
            0.04 * m,
            edge,
        )
    for k in range(layers):
        for j in range(along):
            t = (j + 0.5) / along
            _patch(
                p,
                left,
                front,
                (t - 0.2 / along, t + 0.2 / along),
                (k * high + 0.1 * m, k * high + 0.2 * m),
                (244, 240, 228),
            )


def _shoe_stack(p: Painter, x: float, y: float, tall: float) -> None:
    """A square brick boiler stack drawn for `tall` units and lowered (CHIMNEY_SHARE): a plinth
    with a cream band, a shaft tapering from 2 m square to 1.4 m with cream bands, a corbelled
    black cap, smoking."""
    m = METRE
    tall *= CHIMNEY_SHARE
    base = 2.6 * m
    plinth = p.hall(x, y, 2.6 * m, 2.6 * m, base, SHOE_BRICK, None)
    _bands(p, plinth[0], plinth[1], plinth[2], 0.3 * m, SHOE_TRIM, shade(SHOE_TRIM, 0.78))
    foot, head = y - base, y - tall
    r0, r1 = 1.0 * m, 0.7 * m
    lit = [(x - 2 * r0, foot), (x, foot + r0), (x, head + r1), (x - 2 * r1, head)]
    dark = [(x, foot + r0), (x + 2 * r0, foot), (x + 2 * r1, head), (x, head + r1)]
    p.polygon(lit, SHOE_BRICK, shade(SHOE_BRICK, 0.6))
    p.polygon(dark, shade(SHOE_BRICK, 0.76), shade(SHOE_BRICK, 0.5))
    for up_ in (0.35, 0.7, 0.93):
        h = (foot - head) * up_
        r = r0 + (r1 - r0) * up_
        yy = foot - h
        _bar(p, (x - 2 * r, yy), (x, yy + r), 0.3 * m, SHOE_TRIM)
        _bar(p, (x, yy + r), (x + 2 * r, yy), 0.3 * m, shade(SHOE_TRIM, 0.78))
    p.hall(
        x,
        head + r1 * 0.5 + 0.2 * m,
        2 * r1 * 1.25,
        2 * r1 * 1.25,
        0.7 * m,
        (60, 52, 50),
        (30, 28, 30),
    )
    _smoke(p, x + 1, head - 4, (210, 208, 212), 2)


def shoe_factory() -> Sprite:
    """A boot and shoe factory of warm brick, 26 by 11 m and three storeys high: even rows of
    tall windows of small panes under cream lintels between pilasters, lamplit rooms of
    stitching benches glimpsed through them, cream bands at every floor and a slate roof lit by
    a strip of glazing, its middle bays rising in a cream-trimmed gable that carries a gilt boot
    on its top. Beside it stands a two-storey leather store with stacked loading doors, a hoist
    and a cart of hides from the tannery under it, a boiler house behind with a tall square
    stack smoking, and along the front a loading dock stacked with crates and cartons of
    shoes."""
    m = METRE
    p = Painter(52, 34, (26, 22))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 15.9) * m, (v - 7.1) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    length, depth, high, rise = 26 * m, 11 * m, 12.6 * m, 3 * m
    wall = SHOE_BRICK
    p.polygon(
        [at(14.5, -4.5), at(36, -4.5), at(36, 8), at(26, 8), at(26, 0), at(14.5, 0)],
        (150, 140, 124),
    )
    p.shadow(17.5, 5.5, 2.5)
    boiler = p.hall(*at(30, 11), 6 * m, 5 * m, 5 * m, wall, None)
    _plinth(p, boiler, 5 * m, wall, 0.5 * m)
    _patch(p, boiler[1], boiler[2], (0.3, 0.7), (1.5 * m, 5 * m), (50, 40, 38))
    p.gable(boiler, 1.8 * m, SHOE_SLATE, wall)
    _shoe_stack(p, *at(33.4, 10.6), 34 * m)
    foot = _footing(*at(13, 5.5), length, depth)
    _bed(p, foot)
    top = p.hall(*at(13, 5.5), length, depth, high, wall, None)
    left, front, right, _ = top
    _plinth(p, top, high, wall, 0.6 * m)
    _stains(p, left, front, high, (0.03, 0.3, 0.97), shade(wall, 0.68), 1.4 * m)
    bays = 9
    for k in range(bays + 1):
        t = k / bays
        _bar(
            p,
            _on_wall(left, front, t, 0.6 * m),
            _on_wall(left, front, t, high),
            0.55 * m,
            shade(wall, 0.88),
        )
    for k in range(5):
        t = k / 4
        _bar(
            p,
            _on_wall(front, right, t, 0.6 * m),
            _on_wall(front, right, t, high),
            0.55 * m,
            shade(wall, 0.7),
        )
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        for d, w in ((0, 0.6), (4.35, 0.25), (8.55, 0.25)):
            _patch(p, a, b, (0, 1), (d * m, (d + w) * m), shade(SHOE_TRIM, tone))
    floors = ((1.2, 3.75), (5.4, 7.95), (9.6, 12.0))
    for k in range(bays):
        t = (k + 0.5) / bays
        for j, (h0, f0) in enumerate(floors):
            if j == 2 and k in (1, 6, 7):
                continue
            if j == 2 and k == 3:
                continue
            _shoe_window(p, left, front, length, t, 1.7 * m, (h0 * m, f0 * m))
    for k in range(4):
        t = (k + 0.5) / 4
        for h0, f0 in floors:
            _shoe_window(p, front, right, depth, t, 1.5 * m, (h0 * m, f0 * m), 0.78)
    t = 1.5 / bays
    p.polygon(_round_head(left, front, length, t, 2.4 * m, high - 3.9 * m, high), SHOE_TRIM)
    p.polygon(_round_head(left, front, length, t, 1.9 * m, high - 3.6 * m, high), OXBLOOD)
    _patch(
        p,
        left,
        front,
        (t - 0.0365, t + 0.0365),
        (high - 3.6 * m, high - 2.9 * m),
        shade(WINDOW, 1.1),
    )
    _bar(
        p,
        _on_wall(left, front, t, high - 2.6 * m),
        _on_wall(left, front, t, high),
        0.08 * m,
        shade(OXBLOOD, 0.6),
    )
    for k in (6, 7):
        t = (k + 0.5) / bays
        _patch(p, left, front, (t - 0.042, t + 0.042), (high - 3.9 * m, high - 1.2 * m), SHOE_TRIM)
        _patch(p, left, front, (t - 0.036, t + 0.036), (high - 3.7 * m, high - 1.2 * m), OXBLOOD)
        _bar(
            p,
            _on_wall(left, front, t, high - 3.7 * m),
            _on_wall(left, front, t, high - 1.2 * m),
            0.08 * m,
            shade(OXBLOOD, 0.6),
        )
    p.gable(top, rise, SHOE_SLATE, wall)
    _roof_rows(p, top, rise, SHOE_SLATE, (0.3, 0.6))
    glazing = [
        _on_roof(top, rise, 0.05, 0.62),
        _on_roof(top, rise, 0.95, 0.62),
        _on_roof(top, rise, 0.95, 0.88),
        _on_roof(top, rise, 0.05, 0.88),
    ]
    p.polygon(glazing, (150, 186, 206), (90, 110, 126))
    for k in range(1, 18):
        _bar(
            p,
            _on_roof(top, rise, 0.05 + 0.9 * k / 18, 0.62),
            _on_roof(top, rise, 0.05 + 0.9 * k / 18, 0.88),
            0.07 * m,
            (96, 116, 132),
        )
    _bar(p, left, front, 0.3 * m, shade(SHOE_TRIM, 0.9))
    g0, g1 = 3.5 / bays, 5.5 / bays
    gable = [
        _on_wall(left, front, g0, 0),
        _on_wall(left, front, g0, -1.2 * m),
        _on_wall(left, front, g0 + 0.02, -1.2 * m),
        _on_wall(left, front, (g0 + g1) / 2, -3.6 * m),
        _on_wall(left, front, g1 - 0.02, -1.2 * m),
        _on_wall(left, front, g1, -1.2 * m),
        _on_wall(left, front, g1, 0),
    ]
    p.polygon(gable, wall, shade(wall, 0.6))
    p.polygon(
        [
            gable[2],
            gable[3],
            gable[4],
            _on_wall(left, front, g1 - 0.02, -0.9 * m),
            _on_wall(left, front, (g0 + g1) / 2, -3.2 * m),
            _on_wall(left, front, g0 + 0.02, -0.9 * m),
        ],
        SHOE_TRIM,
    )
    _patch(p, left, front, (g0 + 0.02, g1 - 0.02), (-0.6 * m, 0.9 * m), SHOE_TRIM)
    _patch(p, left, front, (g0 + 0.03, g1 - 0.03), (-0.45 * m, 0.75 * m), OXBLOOD)
    for k in range(6):
        s0 = g0 + 0.035 + (g1 - g0 - 0.07) * k / 6
        _patch(p, left, front, (s0 + 0.004, s0 + 0.016), (-0.2 * m, 0.5 * m), GILT)
    bx, by = _on_wall(left, front, (g0 + g1) / 2, -3.6 * m)
    _shoe_boot(p, bx, by + 0.2 * m, 3.6 * m)
    store = p.hall(*at(31, 4), 7 * m, 7 * m, 8 * m, shade(wall, 0.94), None)
    s_left, s_front, s_right, _ = store
    _plinth(p, store, 8 * m, wall, 0.5 * m)
    for a, b, tone in ((s_left, s_front, 0.94), (s_front, s_right, 0.74)):
        _patch(p, a, b, (0, 1), (3.9 * m, 4.15 * m), shade(SHOE_TRIM, tone))
    for d0, d1 in ((1.2 * m, 3.6 * m), (8 * m - 3.2 * m, 8 * m)):
        _patch(p, s_left, s_front, (0.36, 0.64), (d0 - 0.2 * m, d1), SHOE_TRIM)
        _patch(p, s_left, s_front, (0.38, 0.62), (d0, d1), shade(SHOE_LEATHER, 0.8))
        _bar(
            p,
            _on_wall(s_left, s_front, 0.5, d0),
            _on_wall(s_left, s_front, 0.5, d1),
            0.08 * m,
            shade(SHOE_LEATHER, 0.5),
        )
    for t in (0.14, 0.86):
        _shoe_window(p, s_left, s_front, 7 * m, t, 1.0 * m, (1.4 * m, 3.4 * m), 0.94)
    for t in (0.3, 0.7):
        _shoe_window(p, s_front, s_right, 7 * m, t, 1.2 * m, (1.4 * m, 3.4 * m), 0.74)
        _shoe_window(p, s_front, s_right, 7 * m, t, 1.2 * m, (5.0 * m, 7.0 * m), 0.74)
    _gable_across(p, store, 3 * m, SHOE_SLATE, shade(wall, 0.94))
    hx, hy = _on_wall(s_left, s_front, 0.5, -2.4 * m)
    _patch(p, s_left, s_front, (0.42, 0.58), (-1.9 * m, 0), shade(SHOE_LEATHER, 0.7))
    beam_end = _iso(hx, hy, 0, -1.6 * m)
    _bar(p, (hx, hy), beam_end, 0.3 * m, (62, 52, 48))
    _bar(p, beam_end, (beam_end[0], beam_end[1] + 3.6 * m), 0.05 * m, (40, 36, 34))
    p.blob(
        beam_end[0] - 0.6 * m,
        beam_end[1] + 3.5 * m,
        beam_end[0] + 0.6 * m,
        beam_end[1] + 4.6 * m,
        (156, 96, 60),
    )
    dock_wall = shade(wall, 0.9)
    deck = p.hall(*at(20.4, -1.3), 10.4 * m, 2.6 * m, 1.2 * m, dock_wall, (150, 120, 88))
    for k in range(1, 8):
        _bar(
            p,
            _on_wall(deck[0], deck[1], k / 8, 0),
            _on_wall(deck[3], deck[2], k / 8, 0),
            0.04 * m,
            (120, 94, 68),
        )
    _crates(p, *up(17.4, -1.2, 1.2), (2, 1))
    _shoe_cartons(p, *up(21.2, -1.4, 1.2), 3, 3, 4)
    _shoe_cartons(p, *up(23.6, -1.0, 1.2), 2, 3, 3)
    _crates(p, *at(26.6, -3), (1,))
    _tan_cart(p, *at(31, -2))
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


SUGAR_BRICK: RGB = (186, 94, 62)
SUGAR_CREAM: RGB = (234, 216, 172)
SUGAR_WHITE: RGB = (246, 246, 242)
SUGAR_BLUE: RGB = (48, 84, 168)
SUGAR_BEET: RGB = (218, 194, 150)
SUGAR_GROUND: RGB = (148, 132, 106)
SUGAR_BALLAST: RGB = (126, 120, 112)
SUGAR_RAIL: RGB = (84, 82, 88)
SUGAR_SLEEPER: RGB = (90, 66, 48)
SUGAR_WAGON: RGB = (128, 74, 54)


def _sugar_window(
    p: Painter, a: Point, b: Point, length: float, t: float, wide: float, span: tuple[float, float]
) -> None:
    """A sash window with a segmental head under a cream brick arch and a white sill, its frame
    and bars painted sugar white, `wide` across about `t` along the wall under the edge from `a`
    to `b`, which is `length` units long, from `span[0]` down the wall to `span[1]`; darker on
    the shaded wall."""
    m = METRE
    head, foot = span
    tone = 1.0 if b[1] > a[1] else 0.8

    def outline(half: float, top: float, rise: float) -> list[Point]:
        crown = [
            _on_wall(a, b, t + half * math.cos(turn), top + rise - rise * math.sin(turn))
            for turn in np.linspace(0, math.pi, 7)
        ]
        return [_on_wall(a, b, t - half, foot), _on_wall(a, b, t + half, foot), *crown]

    half = wide / 2 / length
    pad = 0.18 * m / length
    p.polygon(outline(half + pad, head - 0.3 * m, 0.4 * m), shade(SUGAR_CREAM, tone))
    p.polygon(outline(half + pad * 0.5, head - 0.08 * m, 0.3 * m), shade(SUGAR_WHITE, tone))
    p.polygon(outline(half, head, 0.25 * m), shade(MILL_GLASS, tone))
    bar = shade(SUGAR_WHITE, tone)
    _bar(p, _on_wall(a, b, t, head), _on_wall(a, b, t, foot), 0.08 * m, bar)
    middle = (head + foot) / 2 + 0.1 * m
    _bar(p, _on_wall(a, b, t - half, middle), _on_wall(a, b, t + half, middle), 0.12 * m, bar)
    _patch(
        p,
        a,
        b,
        (t - half - pad, t + half + pad),
        (foot, foot + 0.22 * m),
        shade(SUGAR_WHITE, tone * 0.95),
    )


def _sugar_loaf(p: Painter, x: float, y: float, tall: float) -> None:
    """A sugarloaf `tall` units high standing on (x, y): a tall cone of white sugar wrapped in
    blue paper up to near its rounded tip, lit on the left."""
    half = tall * 0.2
    wrap = tall * 0.74
    tip = (x, y - tall)
    p.polygon([(x - half, y), (x, y + half * 0.4), tip], SUGAR_BLUE)
    p.polygon([(x, y + half * 0.4), (x + half, y), tip], shade(SUGAR_BLUE, 0.7))
    top = 1 - wrap / tall
    p.polygon(
        [(x - half * top, y - wrap), (x + half * top, y - wrap), tip],
        SUGAR_WHITE,
    )
    p.polygon([(x + half * top * 0.2, y - wrap), (x + half * top, y - wrap), tip], (204, 206, 210))
    p.line(
        [(x - half * 0.55, y - tall * 0.3), (x - half * 0.3, y - tall * 0.6)],
        shade(SUGAR_BLUE, 1.35),
    )


def _sugar_bags(p: Painter, x: float, y: float, rows: tuple[int, ...]) -> None:
    """White sacks of sugar 0.9 m long stacked on (x, y): a row of `rows[0]` of them lying along
    the lit walls, `rows[1]` on those and so on, plump on top, each stamped in blue."""
    m = METRE
    long, wide, tall = 0.9 * m, 0.6 * m, 0.34 * m
    for level, count in enumerate(rows):
        for k in range(count):
            cx, cy = _iso(x, y, (k + level * 0.5) * long, 0)
            left, front, right, _ = p.hall(
                cx, cy - level * tall, long, wide, tall, shade(SUGAR_WHITE, 0.92), SUGAR_WHITE
            )
            mx, my = _on_wall(left, right, 0.5, 0)
            p.blob(mx - 0.42 * m, my - 0.2 * m, mx + 0.42 * m, my + 0.14 * m, SUGAR_WHITE)
            _patch(p, left, front, (0.35, 0.65), (0.1 * m, 0.24 * m), SUGAR_BLUE)


def _sugar_beet(p: Painter, x: float, y: float, length: float, depth: float, high: float) -> None:
    """A long clamp of pale sugar beet `length` m along the lit walls and `depth` m across,
    heaped `high` m up on (x, y): sloping lit along its front, hipped at its shaded end, and
    lumpy with roots."""
    m = METRE
    rng = np.random.default_rng(7)

    def at(s: float, w: float, z: float) -> Point:
        px, py = _iso(x, y, s * m, w * m)
        return px, py - z * m

    half, side, ridge = length / 2, depth / 2, length / 2 - depth / 2
    p.polygon(
        [at(-half, -side, 0), at(half, -side, 0), at(ridge, 0, high), at(-ridge, 0, high)],
        SUGAR_BEET,
    )
    p.polygon([at(half, -side, 0), at(half, side, 0), at(ridge, 0, high)], shade(SUGAR_BEET, 0.76))
    for _ in range(int(length * depth * 2.2)):
        up = rng.uniform(0.05, 0.95)
        s = rng.uniform(-half + side * up, half - side * up)
        w = -side * (1 - up)
        px, py = at(s, w, high * up)
        tone = rng.choice([0.78, 0.9, 1.08, 1.18])
        r = rng.uniform(0.16, 0.24) * m
        p.blob(px - r, py - r * 0.7, px + r, py + r * 0.7, shade(SUGAR_BEET, tone))
    for _ in range(int(depth * depth * 2.2)):
        up = rng.uniform(0.05, 0.9)
        w = rng.uniform(-side, side) * (1 - up)
        px, py = at(half - side * up, w, high * up)
        r = rng.uniform(0.16, 0.24) * m
        tone = rng.choice([0.62, 0.7, 0.8, 0.88])
        p.blob(px - r, py - r * 0.7, px + r, py + r * 0.7, shade(SUGAR_BEET, tone))


def _sugar_siding(p: Painter, a: Point, b: Point) -> None:
    """A short railway siding on cinder ballast running along the lit walls from `a` to `b` on
    the ground: sleepers every 0.7 m, two rails, and a red buffer stop at `a`."""
    m = METRE
    run = math.dist(a, b) / math.hypot(1, 0.5) / m

    def at(s: float, w: float, z: float = 0) -> Point:
        px, py = _iso(*_on_wall(a, b, s / run, 0), 0, w * m)
        return px, py - z * m

    p.polygon([at(-0.4, -1.5), at(run, -1.5), at(run, 1.5), at(-0.4, 1.5)], SUGAR_BALLAST)
    for s in np.arange(0.3, run, 0.7):
        _bar(p, at(s, -1.25), at(s, 1.25), 0.24 * m, SUGAR_SLEEPER)
    for w in (0.72, -0.72):
        _bar(p, at(-0.2, w, 0.1), at(run, w, 0.1), 0.12 * m, SUGAR_RAIL)
        _bar(p, at(-0.2, w, 0.14), at(run, w, 0.14), 0.04 * m, (170, 170, 176))
    for w in (0.7, -0.7):
        _bar(p, at(0, w), at(0, w, 0.9), 0.2 * m, (60, 52, 48))
    p.polygon([at(0, -1.2, 0.7), at(0, 1.2, 0.7), at(0, 1.2, 1.1), at(0, -1.2, 1.1)], (186, 52, 40))
    p.polygon(
        [at(0, -1.2, 0.7), at(0.25, -1.2, 0.7), at(0.25, -1.2, 1.1), at(0, -1.2, 1.1)],
        (210, 70, 54),
    )


def _sugar_wagon(p: Painter, x: float, y: float) -> None:
    """An open wooden goods wagon 6 m long on the siding at (x, y), its red-oxide planked sides a
    metre high on a dark underframe and four spoked wheels, heaped with sugar beet."""
    m = METRE

    def at(s: float, z: float, w: float = 0) -> Point:
        px, py = _iso(x, y, s * m, w * m)
        return px, py - z * m

    for s in (-1.8, 1.8):
        _food_wheel(p, *at(s, 0.48, 0.72), 0.48 * m, (70, 66, 66))
    left, front, right, back = p.hall(*at(0, 0.9), 6 * m, 2.4 * m, 1 * m, SUGAR_WAGON, None)
    p.polygon([left, front, right, back], shade(SUGAR_BEET, 0.7))
    for d in (0.34, 0.67):
        _bar(
            p,
            _on_wall(left, front, 0, d * m),
            _on_wall(left, front, 1, d * m),
            0.05 * m,
            shade(SUGAR_WAGON, 0.7),
        )
    for t in (0.015, 0.5, 0.985):
        _bar(
            p, _on_wall(left, front, t, 0), _on_wall(left, front, t, 1 * m), 0.14 * m, (52, 48, 48)
        )
    _bar(
        p, _on_wall(left, front, 0, 1 * m), _on_wall(left, front, 1, 1 * m), 0.24 * m, (52, 48, 48)
    )
    for s in np.linspace(-2.6, 2.6, 9):
        for w in (0.7, 0.2, -0.3, -0.75):
            lift = 0.4 * (1 - abs(w + 0.05) / 0.8) * (1 - abs(s) / 3.4)
            cx, cy = at(s + w * 0.25, 1.95 + lift, w)
            tone = 1.12 if (round(s * 3) + round(w * 5)) % 2 else 0.9
            p.blob(
                cx - 0.28 * m, cy - 0.2 * m, cx + 0.28 * m, cy + 0.18 * m, shade(SUGAR_BEET, tone)
            )
    for s in (-1.8, 1.8):
        _food_wheel(p, *at(s, 0.48, -0.85), 0.48 * m, (70, 66, 66))


def _sugar_stack(p: Painter, x: float, y: float, tall: float) -> None:
    """A square brick boiler stack drawn for `tall` units and lowered (CHIMNEY_SHARE), standing
    on (x, y): tapering from 2.6 m a side at its foot to 1.6 m, banded in cream brick, with a
    corbelled cream crown and a dark mouth, smoking."""
    m = METRE
    tall *= CHIMNEY_SHARE

    def corners(z: float) -> tuple[Point, Point, Point]:
        side = (2.6 + (1.6 - 2.6) * z / tall) * m
        return (x - side, y - z), (x, y + side / 2 - z), (x + side, y - z)

    def faces(z0: float, z1: float, lit: RGB, dark: RGB) -> None:
        (l0, f0, r0), (l1, f1, r1) = corners(z0), corners(z1)
        p.polygon([l0, f0, f1, l1], lit)
        p.polygon([f0, r0, r1, f1], dark)

    faces(0, tall, SUGAR_BRICK, shade(SUGAR_BRICK, 0.76))
    faces(0, 1.2 * m, shade(SUGAR_BRICK, 0.8), shade(SUGAR_BRICK, 0.62))
    for up in (0.3, 0.6):
        faces(tall * up, tall * up + 0.45 * m, SUGAR_CREAM, shade(SUGAR_CREAM, 0.78))
    faces(tall - 2.4 * m, tall - 1.6 * m, SUGAR_CREAM, shade(SUGAR_CREAM, 0.78))
    crown = p.box(x, y - tall + 1.2 * m, 2.1 * m, 1.2 * m, SUGAR_CREAM, (40, 34, 34))
    p.box(x, y - tall + 1.2 * m, 2.1 * m, 0.35 * m, shade(SUGAR_CREAM, 0.85), None)
    cx, cy = (crown[0][0] + crown[2][0]) / 2, (crown[0][1] + crown[2][1]) / 2
    _smoke(p, cx + 1, cy - 3, (206, 206, 212), 2)


def sugar_refinery() -> Sprite:
    """A Victorian sugar refinery: a block 20 by 12 m and six storeys, 18 m to its cornice, in
    warm red brick banded in cream at every floor, with rows of white-painted sash windows under
    cream arches, blue taking-in doors stacked up one bay under a hoist with a sack swinging from
    it, and a slate roof with a louvred ventilator along its ridge. At its end rises the
    cream-brick char house, a tower for the bone-char filters, louvred at its top under a slate
    pyramid, with a boiler house at its foot and behind that a square tapering stack banded and
    crowned in cream, smoking. Before it a siding brings wagons of pale sugar beet to a long beet
    clamp, white sacks of sugar wait by its door, and a dray stands loaded with sugarloaves in
    their blue paper."""
    m = METRE
    p = Painter(43, 42, (18, 31))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 12) * m, (v - 2) * m)

    length, depth, high = 20 * m, 12 * m, 18 * m
    p.polygon([at(-0.5, -8), at(28, -8), at(28, 0), at(-0.5, 0)], SUGAR_GROUND)
    foot = _footing(*at(10, 6), length, depth)
    _bed(p, foot)
    _bed(p, _footing(*at(23.5, 3.5), 7 * m, 7 * m), margin=1)
    p.shadow(16, 6, 6)
    _sugar_stack(p, *at(29.6, 9.6), 56 * m)
    top = p.hall(*at(10, 6), length, depth, high, SUGAR_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, SUGAR_BRICK, 1.0 * m)
    _stains(p, left, front, high, (0.03, 0.27, 0.52, 0.94), shade(SUGAR_BRICK, 0.7), 1.6 * m)
    floors = ((1.0, 2.6), (3.8, 5.4), (6.6, 8.2), (9.4, 11.0), (12.2, 13.8), (15.0, 16.8))
    for a, b, tone, bays in ((left, front, 1.0, 8), (front, right, 0.8, 5)):
        _patch(p, a, b, (0, 1), (0, 0.6 * m), shade(SUGAR_CREAM, tone))
        _patch(p, a, b, (0, 1), (0.6 * m, 0.75 * m), shade(SUGAR_BRICK, 0.7 * tone))
        for drop in (3.25, 6.05, 8.85, 11.65):
            _patch(p, a, b, (0, 1), (drop * m, (drop + 0.2) * m), shade(SUGAR_CREAM, tone))
        _patch(p, a, b, (0, 1), (14.3 * m, 14.65 * m), shade(SUGAR_CREAM, tone))
        span = length if bays == 8 else depth
        for k in range(bays):
            t = (k + 0.5) / bays
            for head, foot_at in floors:
                if bays == 8 and k == 6:
                    continue
                _sugar_window(p, a, b, span, t, 1.1 * m, (head * m, foot_at * m))
    t = 6.5 / 8
    for head, foot_at in floors[:5]:
        _patch(
            p,
            left,
            front,
            (t - 0.034, t + 0.034),
            ((head - 0.1) * m, (foot_at + 0.3) * m),
            SUGAR_CREAM,
        )
        _patch(p, left, front, (t - 0.028, t + 0.028), (head * m, (foot_at + 0.3) * m), SUGAR_BLUE)
        _bar(
            p,
            _on_wall(left, front, t, head * m),
            _on_wall(left, front, t, (foot_at + 0.3) * m),
            0.05 * m,
            shade(SUGAR_BLUE, 0.65),
        )
    p.polygon(_round_head(left, front, length, t, 2.6 * m, 14.9 * m, high), SUGAR_CREAM)
    p.polygon(_round_head(left, front, length, t, 2.1 * m, 15.15 * m, high), DOORWAY)
    _patch(p, left, front, (t - 0.05, t - 0.004), (15.9 * m, high), SUGAR_BLUE)
    rise = 4.5 * m
    p.gable(top, rise, MILL_SLATE, SUGAR_BRICK)
    _roof_rows(p, top, rise, MILL_SLATE, (0.25, 0.5, 0.75))
    ridge_back, ridge_front = _ridge(top, rise)
    vx, vy = _on_wall(ridge_back, ridge_front, 0.42, 0)
    vent = p.hall(vx, vy + 0.5 * m, 10 * m, 1.6 * m, 1.5 * m, shade(SUGAR_CREAM, 0.9), None)
    for a, b, tone in ((vent[0], vent[1], 1.0), (vent[1], vent[2], 0.8)):
        for k in range(1, 4):
            drop = 1.0 * m * k / 4
            _bar(
                p,
                _on_wall(a, b, 0, drop),
                _on_wall(a, b, 1, drop),
                0.12 * m,
                shade(SUGAR_CREAM, 0.55 * tone),
            )
    p.gable(vent, 0.7 * m, MILL_SLATE, shade(SUGAR_CREAM, 0.9))
    hx, hy = _on_wall(left, front, t, 0)
    hood = p.hall(hx - 0.4 * m, hy + 1.4 * m, 2.4 * m, 1.6 * m, 2.4 * m, SUGAR_BRICK, None)
    h_left, h_front, h_right, h_back = hood
    _patch(p, h_left, h_front, (0.2, 0.8), (0.9 * m, 2.4 * m), SUGAR_BLUE)
    near, far = _on_wall(h_left, h_front, 0.5, -1.2 * m), _on_wall(h_back, h_right, 0.5, -1.2 * m)
    p.polygon([h_front, h_right, far, near], shade(MILL_SLATE, 0.72), shade(MILL_SLATE, 0.5))
    p.polygon([h_left, h_back, far, near], MILL_SLATE, shade(MILL_SLATE, 0.6))
    p.polygon([h_left, h_front, near], SUGAR_BRICK, shade(SUGAR_BRICK, 0.55))
    beam = _on_wall(h_left, h_front, 0.5, 0.7 * m)
    tip = (beam[0] - 1.4 * m, beam[1] + 0.7 * m)
    _bar(p, beam, tip, 0.22 * m, (54, 50, 50))
    _bar(p, tip, (tip[0], tip[1] + 3.4 * m), 0.06 * m, (60, 54, 50))
    sx, sy = tip[0], tip[1] + 4.4 * m
    p.ellipse(sx - 0.38 * m, sy - 1.1 * m, sx + 0.38 * m, sy + 0.1 * m, SUGAR_WHITE)
    _bar(
        p,
        (sx - 0.2 * m, sy - 0.9 * m),
        (sx + 0.2 * m, sy - 0.9 * m),
        0.08 * m,
        shade(SUGAR_WHITE, 0.7),
    )
    tower = p.hall(*at(23.5, 3.5), 7 * m, 7 * m, 26 * m, SUGAR_CREAM, None)
    t_left, t_front, t_right, _ = tower
    _plinth(p, tower, 26 * m, SUGAR_CREAM, 1.0 * m)
    for a, b, tone in ((t_left, t_front, 1.0), (t_front, t_right, 0.8)):
        brick = shade(SUGAR_BRICK, tone)
        _patch(p, a, b, (0, 1), (0, 0.7 * m), brick)
        for drop in (5.2, 10.6, 16.0, 21.4):
            _patch(p, a, b, (0, 1), (drop * m, (drop + 0.35) * m), brick)
        for s in (0.03, 0.97):
            _bar(p, _on_wall(a, b, s, 0.7 * m), _on_wall(a, b, s, 25 * m), 0.4 * m, brick)
        _patch(p, a, b, (0.2, 0.8), (1.4 * m, 4.4 * m), (52, 48, 52))
        for d in np.arange(1.7, 4.3, 0.45):
            _bar(
                p,
                _on_wall(a, b, 0.2, d * m),
                _on_wall(a, b, 0.8, d * m),
                0.12 * m,
                shade(SUGAR_CREAM, 0.85 * tone),
            )
        for head in (6.4, 11.8, 17.2):
            for s in (0.34, 0.66):
                _sugar_window(p, a, b, 7 * m, s, 1.0 * m, (head * m, (head + 2.8) * m))
    _sugar_window(p, t_front, t_right, 7 * m, 0.5, 1.0 * m, (22.6 * m, 24.4 * m))
    p.polygon(_round_head(t_left, t_front, 7 * m, 0.5, 1.8 * m, 22.6 * m, 26 * m), DOORWAY)
    apex = ((t_left[0] + t_right[0]) / 2, (t_left[1] + t_right[1]) / 2 - 4.5 * m)
    p.polygon([t_left, t_front, apex], MILL_SLATE, shade(MILL_SLATE, 0.6))
    p.polygon([t_front, t_right, apex], shade(MILL_SLATE, 0.72), shade(MILL_SLATE, 0.5))
    _bar(p, apex, (apex[0], apex[1] - 1.6 * m), 0.12 * m, (50, 50, 56))
    p.blob(apex[0] - 0.3 * m, apex[1] - 1.9 * m, apex[0] + 0.3 * m, apex[1] - 1.3 * m, FOOD_GILT)
    boiler = p.hall(*at(29.5, 4.5), 5 * m, 6 * m, 6 * m, SUGAR_BRICK, None)
    b_left, b_front, b_right, _ = boiler
    _plinth(p, boiler, 6 * m, SUGAR_BRICK, 0.7 * m)
    _bands(p, b_left, b_front, b_right, 0.3 * m, SUGAR_CREAM, shade(SUGAR_CREAM, 0.8))
    _sugar_window(p, b_left, b_front, 5 * m, 0.5, 1.4 * m, (1.4 * m, 3.6 * m))
    p.polygon(_round_head(b_front, b_right, 6 * m, 0.5, 2.2 * m, 2.4 * m, 6 * m), DOORWAY)
    p.gable(boiler, 2 * m, MILL_SLATE, SUGAR_BRICK)
    _roof_rows(p, boiler, 2 * m, MILL_SLATE)
    _sugar_beet(p, *at(5.2, -2.6), 10, 3.4, 2.2)
    _sugar_bags(p, *at(13.6, -1.0), (3, 2, 1))
    _sugar_bags(p, *at(17.2, -1.4), (2, 1))

    def loaves(on_bed: Callable[[float, float], Point]) -> None:
        for w in (0.42, -0.3):
            for s in np.linspace(-1.2, 1.2, 5):
                _sugar_loaf(p, *on_bed(s + w * 0.3, w), 1.0 * m)

    _food_cart(p, *at(13.4, -4.0), 3.4, 0.3, SUGAR_GROUND, loaves)
    _sugar_siding(p, at(0, -6.6), at(14, -6.6))
    _sugar_wagon(p, *at(5.6, -6.6))
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


def _works_office(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """A works office block four storeys high, `length` by `depth`, standing on (x, y): red
    brick on a stone plinth with cream quoins, string courses and a cornice, rows of
    round-headed windows, a slate roof, a central bay standing forward under a pediment with a
    round window over an arched entrance up steps, and a clock tower rising from its roof to a
    slate pyramid and a gilt vane."""
    m = METRE
    storey, storeys = 3.8 * m, 4
    high = storey * storeys
    top = p.hall(x, y, length, depth, high, MILL_BRICK, None)
    left, front, right, _ = top

    def dress(
        a: Point, b: Point, span: float, bays: int, skip: tuple[float, float], tone: float
    ) -> None:
        _patch(p, a, b, (0, 1), (high - 1.1 * m, high), shade(WORKS_STONE, tone))
        _patch(p, a, b, (0, 1), (0, 0.7 * m), shade(DRESSING, tone))
        for k in range(1, storeys):
            _bar(
                p,
                _on_wall(a, b, 0, k * storey),
                _on_wall(a, b, 1, k * storey),
                0.25 * m,
                shade(DRESSING, tone),
            )
        for t in (0.015, 0.985):
            _bar(
                p,
                _on_wall(a, b, t, 0.7 * m),
                _on_wall(a, b, t, high - 1.1 * m),
                0.6 * m,
                shade(DRESSING, tone),
            )
        for k in range(bays):
            t = (k + 0.5) / bays
            if skip[0] < t < skip[1]:
                continue
            for s in range(storeys):
                head = s * storey + 0.9 * m
                tall = 2.6 * m if s == storeys - 1 else 2.1 * m
                _mill_window(
                    p, a, b, span, t, 1.1 * m, (head, head + tall), shade(MILL_GLASS, tone)
                )

    dress(left, front, length, max(3, round(length / (2.8 * m))), (0.34, 0.66), 1.0)
    dress(front, right, depth, max(2, round(depth / (2.8 * m))), (2, 2), 0.85)
    p.gable(top, 3 * m, MILL_SLATE, MILL_BRICK)
    _roof_rows(p, top, 3 * m, MILL_SLATE)
    cx, cy = _iso(x, y, 0, -(depth / 2 + 0.5 * m))
    bay = p.hall(cx, cy, length * 0.32, 1.6 * m, high + 0.6 * m, shade(MILL_BRICK, 1.04), None)
    b_left, b_front, b_right, _ = bay
    span = length * 0.32
    _patch(p, b_left, b_front, (0, 1), (high - 0.5 * m, high + 0.6 * m), WORKS_STONE)
    _patch(p, b_front, b_right, (0, 1), (high - 0.5 * m, high + 0.6 * m), shade(WORKS_STONE, 0.8))
    for k in range(1, storeys):
        _bar(
            p,
            _on_wall(b_left, b_front, 0, k * storey + 0.6 * m),
            _on_wall(b_left, b_front, 1, k * storey + 0.6 * m),
            0.25 * m,
            DRESSING,
        )
    for t in (0.04, 0.96):
        _bar(
            p,
            _on_wall(b_left, b_front, t, 0),
            _on_wall(b_left, b_front, t, high),
            0.6 * m,
            DRESSING,
        )
    for s in range(storeys - 1):
        head = s * storey + 1.5 * m
        for t in (0.3, 0.7):
            _mill_window(p, b_left, b_front, span, t, 1.1 * m, (head, head + 2.1 * m))
    p.polygon(
        _round_head(b_left, b_front, span, 0.5, 3 * m, high - 3.8 * m, high + 0.6 * m), DRESSING
    )
    p.polygon(
        _round_head(b_left, b_front, span, 0.5, 2.4 * m, high - 3.5 * m, high + 0.6 * m), DOOR
    )
    _bar(
        p,
        _on_wall(b_left, b_front, 0.5, high - 2.4 * m),
        _on_wall(b_left, b_front, 0.5, high + 0.6 * m),
        0.08 * m,
        shade(DOOR, 0.6),
    )
    for k in range(3):
        sx, sy = _iso(cx, cy, 0, -(1.2 + 0.5 * k) * m)
        p.hall(
            sx,
            sy,
            4.4 * m + k * 0.8 * m,
            0.5 * m,
            (0.6 - 0.2 * k) * m,
            WORKS_STONE,
            shade(WORKS_STONE, 1.1),
        )
    _bands(p, b_left, b_front, b_right, 0.3 * m, DRESSING, shade(DRESSING, 0.78))
    _gable_across(p, bay, 2.8 * m, MILL_SLATE, shade(MILL_BRICK, 1.04))
    ox, oy = _on_wall(b_left, b_front, 0.5, -1 * m)
    p.ellipse(ox - 0.75 * m, oy - 0.75 * m, ox + 0.75 * m, oy + 0.75 * m, DRESSING)
    p.ellipse(ox - 0.5 * m, oy - 0.5 * m, ox + 0.5 * m, oy + 0.5 * m, MILL_GLASS)
    tx, ty = _iso(x, y, 0, 0.5 * m)
    rise = high + 1.6 * m
    tower = p.hall(tx, ty - rise, 4.6 * m, 4.6 * m, 9 * m, MILL_BRICK, None)
    t_left, t_front, t_right, _ = tower
    for a, b in ((t_left, t_front), (t_front, t_right)):
        for drop in (0.3 * m, 4.6 * m):
            _bar(p, _on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop), 0.3 * m, DRESSING)
        for t in (0.04, 0.96):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, 9 * m), 0.35 * m, DRESSING)
        _works_clock(p, *_on_wall(a, b, 0.5, 2.5 * m), 1.2 * m)
        _mill_window(p, a, b, 4.6 * m, 0.5, 1.1 * m, (5.6 * m, 8 * m))
    _works_pyramid(p, tower, 3.6 * m)


def mega_factory() -> Sprite:
    """A whole works quarter: a four-storey red-brick office block with a pediment over its
    entrance and a clock tower rising from its roof, three long halls under north-light roofs
    of glazed teeth with rows of round-headed windows under cream arches and arched loading
    doors with hoists, two tall round chimneys smoking, a brick water tower carrying a riveted
    iron tank, a siding along the front with vans and open wagons of crated goods, and crates
    and girders stacked in the yard."""
    m = METRE
    p = Painter(66, 66, (33, 46))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 24) * m, (v - 9) * m)

    p.polygon([at(2, -10), at(55, -10), at(55, 0), at(2, 0)], CINDERS)
    halls = (
        ((13, 21), (26, 12, 10), 7, 3),
        ((35, 22), (10, 12, 10), 3, 1),
        ((38, 6.5), (28, 13, 10.5), 8, 2),
    )
    for (u, v), (length, depth, _), _, _ in halls:
        _bed(p, _footing(*at(u, v), length * m, depth * m), margin=0.6)
    _bed(p, _footing(*at(11, 6), 22 * m, 12 * m), margin=0.6)
    p.shadow(26, 7, 3)
    _works_water_tower(p, *at(27.5, 31))
    for (u, v), (length, depth, high), teeth, door in halls[:2]:
        _works_hall(p, *at(u, v), (length * m, depth * m, high * m), teeth, door)
    _tall_chimney(p, *at(28, 14), 70 * m, 1.45)
    _tall_chimney(p, *at(47, 15), 62 * m, 1.3)
    _works_office(p, *at(11, 6), 22 * m, 12 * m)
    (u, v), (length, depth, high), teeth, door = halls[2]
    _works_hall(p, *at(u, v), (length * m, depth * m, high * m), teeth, door)
    _works_siding(p, at(3, -7), at(54, -7))
    for u, covered in ((10, True), (19, False), (28, True)):
        _works_wagon(p, *at(u, -7), covered)
    _crates(p, *at(38, -3), (3, 2, 1))
    _crates(p, *at(42.5, -2.7), (2, 2))
    _girders(p, *at(49, -3), 6, 3)
    return p.sprite()


OIL_YELLOW: RGB = (232, 184, 40)
TANK_WHITE: RGB = (226, 226, 222)


OIL_STEEL: RGB = (88, 92, 100)
OIL_CRUDE: RGB = (64, 62, 64)
OIL_RUST: RGB = (150, 84, 50)
OIL_AMBER: RGB = (236, 164, 44)
OIL_KEROSENE: RGB = (240, 204, 104)
OIL_ALUMINIUM: RGB = (168, 170, 168)
OIL_GRAVEL: RGB = (150, 142, 126)
OIL_PAD: RGB = (172, 168, 158)
OIL_BRICK: RGB = (150, 82, 60)
OIL_FIRE: RGB = (252, 156, 52)
OIL_FLAME: RGB = (250, 140, 40)
OIL_GLOW: RGB = (255, 232, 150)
OIL_BALLAST: RGB = (124, 116, 104)
OIL_SLEEPER: RGB = (90, 70, 54)
OIL_RAIL: RGB = (150, 150, 156)
OIL_EARTH: RGB = (138, 120, 88)
OIL_WEATHERED: RGB = (98, 92, 88)
OIL_WAGON: RGB = (44, 42, 44)
OIL_BLACK: RGB = (40, 38, 40)
OIL_FRAME: RGB = (66, 68, 74)
OIL_HANDRAIL: RGB = (190, 186, 176)
OIL_PIPES: tuple[RGB, ...] = (OIL_BLACK, OIL_RUST, OIL_AMBER, (150, 150, 150))


def _oil_band(
    p: Painter,
    x: float,
    y: float,
    half: float,
    thick: float,
    colour: RGB,
    span: tuple[float, float] = (0, math.pi),
) -> None:
    """A band `thick` high round the front of an upright round thing whose axis passes through
    (x, y), `half` across, from `span[0]` to `span[1]` of the way round (0 is its right, a half
    turn its left)."""
    turns = np.linspace(span[0], span[1], 13)
    upper = [(x + half * math.cos(a), y + half / 2 * math.sin(a) - thick) for a in turns]
    lower = [(x + half * math.cos(a), y + half / 2 * math.sin(a)) for a in turns[::-1]]
    p.polygon(upper + lower, colour)


def _oil_dome(p: Painter, x: float, y: float, half: float, rise: float, roof: RGB) -> None:
    """A shallow domed roof `half` across and `rise` high on a round wall whose top is centred at
    (x, y): dark at its rim, lit towards its crown, with a glint on its left."""
    steps = 7
    for k in range(steps + 1):
        f = k / steps
        r = half * math.sqrt(1 - f * 0.94)
        cy = y - rise * f
        tone = shade(roof, 0.82 + 0.38 * f)
        p.blob(x - r, cy - r / 2, x + r, cy + r / 2, tone)
    p.blob(
        x - half * 0.62, y - rise - half * 0.22, x - half * 0.1, y - rise * 0.4, shade(roof, 1.32)
    )


def _oil_column(
    p: Painter, x: float, y: float, wide: float, tall: float, metal: RGB = OIL_STEEL
) -> None:
    """A distillation column `wide` metres across and `tall` high standing on (x, y): a darker
    skirt at its foot, a shell of steel with a sheen down its lit side, ringed with catwalks every
    six metres and climbed by a caged ladder, rust running from under each catwalk, a dished head
    lit on top with a davit, and a vapour line falling from its head down its shaded side."""
    m = METRE
    r = wide * m / math.sqrt(2)
    skirt, h = 1.8 * m, tall * m
    _turned(p, x, y, [(0, r * 1.12), (skirt, r * 1.12)], shade(metal, 0.62))
    head = [(0, r), (h - skirt, r), (h - skirt + 0.3 * r, 0.84 * r), (h - skirt + 0.55 * r, 0)]
    _turned(p, x, y - skirt, head, metal)
    _bar(p, (x - 0.5 * r, y - skirt), (x - 0.5 * r, y - h + 0.1 * r), 0.55 * m, shade(metal, 1.2))
    _bar(p, (x - 0.56 * r, y - skirt), (x - 0.56 * r, y - h), 0.2 * m, shade(metal, 1.55))
    p.blob(x - 0.7 * r, y - h - 0.4 * r, x - 0.05 * r, y - h + 0.05 * r, shade(metal, 1.5))
    for z in np.arange(skirt + 3 * m, h - 1 * m, 3 * m):
        _oil_band(p, x, y - z, r, 0.1 * m, shade(metal, 0.8))
    decks = list(np.arange(6 * m, h - 2 * m, 6 * m)) + [h - 0.5 * m]
    wall = r + 1.1 * m
    ladder = 0.72 * math.pi
    lx = x + (r + 0.35 * m) * math.cos(ladder)
    ly = y + (r + 0.35 * m) / 2 * math.sin(ladder)
    for side in (-0.22, 0.22):
        _bar(
            p,
            (lx + side * m, ly - skirt),
            (lx + side * m, ly - decks[-1]),
            0.12 * m,
            shade(metal, 1.45),
        )
    for n, z in enumerate(decks):
        for a, reach in ((0.4, 2.2), (0.88, 1.5), (0.18, 1.2)):
            a = (a + 0.23 * n) % 1
            reach *= 0.6 + 0.4 * ((n * 7 + round(a * 10)) % 3) / 2
            sx = x + r * math.cos(a * math.pi)
            sy = y + r / 2 * math.sin(a * math.pi) - z
            _bar(p, (sx, sy), (sx, sy + reach * m), 0.22 * m, shade(OIL_RUST, 0.85))
        _oil_band(p, x, y - z, wall, 0.35 * m, shade(metal, 0.5))
        _oil_band(p, x, y - z - 0.35 * m, wall, 0.1 * m, shade(metal, 1.35))
        _oil_band(p, x, y - z - 1.1 * m, wall, 0.09 * m, shade(metal, 1.25))
        for a in (0.08, 0.3, 0.55, 0.8, 0.95):
            px = x + wall * math.cos(a * math.pi)
            py = y + wall / 2 * math.sin(a * math.pi) - z
            _bar(p, (px, py - 0.35 * m), (px, py - 1.1 * m), 0.08 * m, shade(metal, 1.2))
    for a, reach, colour in ((0.42, 0.72, (116, 118, 120)), (0.6, 0.45, shade(OIL_RUST, 0.8))):
        out = r + 0.5 * m
        px = x + out * math.cos(a * math.pi)
        py = y + out / 2 * math.sin(a * math.pi)
        sx = x + r * math.cos(a * math.pi)
        sy = y + r / 2 * math.sin(a * math.pi)
        z = reach * h
        _oil_pipe(
            p,
            [(sx, sy - z), (px, py - z), (px, py - 0.6 * m), (px - 0.8 * m, py + 0.2 * m)],
            colour,
            0.4,
        )
    crown = y - h - 0.55 * r
    _bar(p, (x, crown), (x, crown - 2.2 * m), 0.16 * m, shade(metal, 0.7))
    _bar(p, (x, crown - 2.2 * m), (x - 1.4 * m, crown - 1.8 * m), 0.14 * m, shade(metal, 0.7))
    vx = x + r + 0.7 * m
    line = shade(metal, 0.75)
    _bar(p, (x + 0.3 * r, crown + 0.2 * r), (x + 0.3 * r, crown - 0.8 * m), 0.4 * m, line)
    _bar(p, (x + 0.3 * r, crown - 0.8 * m), (vx, crown - 0.4 * m), 0.4 * m, line)
    _bar(p, (vx, crown - 0.4 * m), (vx, y - 5 * m), 0.4 * m, line)
    _bar(
        p, (vx - 0.12 * m, crown - 0.4 * m), (vx - 0.12 * m, y - 5 * m), 0.1 * m, shade(metal, 1.3)
    )


def _oil_crude_tank(p: Painter, x: float, y: float, wide: float, tall: float) -> None:
    """A big squat crude tank `wide` metres across and `tall` high standing on (x, y) inside a low
    earth bund: black steel courses lit down the left with a sheen, rust running from its rim, a
    shallow domed roof lit on its crown with a handrail round its edge and a vent at its top, and
    a stair winding up its front from the right to the roof on its left."""
    m = METRE
    r = wide * m / math.sqrt(2)
    h = tall * m
    out = r + 3.2 * m
    p.blob(
        x - out - 1.2 * m,
        y - out / 2 - 0.4 * m,
        x + out + 1.2 * m,
        y + out / 2 + 0.8 * m,
        shade(OIL_EARTH, 0.8),
    )
    p.blob(x - out, y - out / 2, x + out, y + out / 2 + 0.2 * m, shade(OIL_EARTH, 1.12))
    p.blob(
        x - out + 1.2 * m,
        y - out / 2 + 0.9 * m,
        x + out - 1.2 * m,
        y + out / 2 - 0.2 * m,
        shade(OIL_EARTH, 0.72),
    )
    _oil(p, x - r * 0.6, y + r * 0.4, 1.4 * m)
    _turned(p, x, y, [(0, r), (h, r)], OIL_CRUDE)
    for z in np.arange(1.9 * m, h - 0.5 * m, 1.9 * m):
        _oil_band(p, x, y - z, r, 0.1 * m, shade(OIL_CRUDE, 0.7))
    sheen = [(x - 0.62 * r, 0.9 * m, 1.35), (x - 0.5 * r, 0.3 * m, 1.75)]
    for sx, wide_bar, tone in sheen:
        _bar(p, (sx, y + 0.39 * r), (sx, y + 0.39 * r - h), wide_bar, shade(OIL_CRUDE, tone))
    for a, reach in ((0.15, 0.5), (0.33, 0.35), (0.5, 0.6), (0.62, 0.3), (0.86, 0.45)):
        sx = x + r * math.cos(a * math.pi)
        sy = y + r / 2 * math.sin(a * math.pi) - h
        _bar(p, (sx, sy), (sx, sy + reach * h), 0.35 * m, shade(OIL_RUST, 0.85))
    roof = (112, 106, 102)
    _oil_band(p, x, y - h + 0.3 * m, r, 0.5 * m, shade(OIL_CRUDE, 0.5))
    _oil_dome(p, x, y - h, r, 0.16 * wide * m, roof)
    _oil_band(p, x, y - h - 1 * m, r * 0.98, 0.1 * m, (196, 194, 184))
    for a in (0.1, 0.35, 0.6, 0.9):
        px = x + 0.98 * r * math.cos(a * math.pi)
        py = y + 0.98 * r / 2 * math.sin(a * math.pi) - h
        _bar(p, (px, py), (px, py - 1 * m), 0.1 * m, (196, 194, 184))
    vent = y - h - 0.16 * wide * m
    _turned(
        p,
        x,
        vent + 0.2 * m,
        [(0, 0.7 * m), (0.9 * m, 0.7 * m), (1.2 * m, 0.9 * m)],
        shade(roof, 0.7),
        shade(roof, 0.5),
    )
    stair = (192, 188, 176)
    a0, a1 = 0.18 * math.pi, 0.97 * math.pi
    turns = np.linspace(a0, a1, 24)
    rise = np.linspace(0, h, 24)
    out = r + 1.2 * m

    def at(a: float, z: float, half: float) -> Point:
        return x + half * math.cos(a), y + half / 2 * math.sin(a) - z

    p.polygon(
        [at(a, z, r) for a, z in zip(turns, rise, strict=True)]
        + [at(a, z, out) for a, z in zip(turns[::-1], rise[::-1], strict=True)],
        shade(stair, 0.62),
    )
    p.polygon(
        [at(a, z + 0.3 * m, out) for a, z in zip(turns, rise, strict=True)]
        + [at(a, z, out) for a, z in zip(turns[::-1], rise[::-1], strict=True)],
        stair,
    )
    p.line([at(a, z + 1.1 * m, out) for a, z in zip(turns, rise, strict=True)], stair)
    for a, z in list(zip(turns, rise, strict=True))[::3]:
        _bar(p, at(a, z, out), at(a, z + 1.1 * m, out), 0.1 * m, stair)


def _oil_product_tank(p: Painter, x: float, y: float, wide: float, tall: float, band: RGB) -> None:
    """A product tank `wide` metres across and `tall` high standing on (x, y): weathered aluminium
    paint lit on the left, a band in the colour of what it holds, a low coned roof lit on top
    and a caged ladder up its lit side."""
    m = METRE
    r = wide * m / math.sqrt(2)
    h = tall * m
    _turned(p, x, y, [(0, r), (h, r)], OIL_ALUMINIUM)
    _bar(
        p,
        (x - 0.55 * r, y + 0.4 * r),
        (x - 0.55 * r, y + 0.4 * r - h),
        0.5 * m,
        shade(OIL_ALUMINIUM, 1.18),
    )
    for z in np.arange(1.8 * m, h, 1.8 * m):
        _oil_band(p, x, y - z, r, 0.08 * m, shade(OIL_ALUMINIUM, 0.82))
    _oil_band(p, x, y - h * 0.72, r, 0.9 * m, band)
    _oil_band(p, x, y - h * 0.72 - 0.9 * m, r, 0.15 * m, shade(band, 1.2))
    _oil_dome(p, x, y - h, r, 0.14 * wide * m, shade(OIL_ALUMINIUM, 0.95))
    a = 0.78 * math.pi
    lx, ly = x + (r + 0.3 * m) * math.cos(a), y + (r + 0.3 * m) / 2 * math.sin(a)
    for side in (-0.25, 0.25):
        _bar(p, (lx + side * m, ly), (lx + side * m, ly - h - 1 * m), 0.12 * m, (70, 70, 74))


def _oil_rack(
    p: Painter, a: Point, b: Point, across: Point, high: float, pipes: tuple[RGB, ...]
) -> None:
    """A pipe rack on the ground from `a` to `b` reaching `across` to either side of that line,
    `high` metres tall: steel bents every six metres with a beam at the top and one halfway, and
    `pipes` laid side by side on both tiers, each lit along its top."""
    m = METRE
    if across[1] > 0:
        across = (-across[0], -across[1])
    length = math.hypot(b[0] - a[0], (b[1] - a[1]) * 2) / m
    bents = max(2, round(length / 6) + 1)
    steel = (70, 72, 78)
    tiers = (high * 0.55 * m, high * m)

    def off(pt: Point, k: float, z: float) -> Point:
        return pt[0] + across[0] * k, pt[1] + across[1] * k - z

    for k in (1, -1):
        for i in range(bents):
            foot = _on_wall(a, b, i / (bents - 1), 0)
            _bar(
                p,
                off(foot, k, 0),
                off(foot, k, tiers[1]),
                0.35 * m,
                shade(steel, 1.15 if k < 0 else 0.9),
            )
        if k == 1:
            for z in tiers:
                for j in range(len(pipes)):
                    t = 0.8 - 1.6 * j / max(1, len(pipes) - 1)
                    colour = pipes[(j + (z == tiers[1])) % len(pipes)]
                    _bar(p, off(a, t, z + 0.25 * m), off(b, t, z + 0.25 * m), 0.45 * m, colour)
                    _bar(
                        p,
                        off(a, t, z + 0.42 * m),
                        off(b, t, z + 0.42 * m),
                        0.12 * m,
                        shade(colour, 1.7),
                    )
    for i in range(bents):
        foot = _on_wall(a, b, i / (bents - 1), 0)
        for z in tiers:
            _bar(p, off(foot, 1, z), off(foot, -1, z), 0.3 * m, steel)
    for z in tiers:
        _bar(p, off(a, -1, z), off(b, -1, z), 0.25 * m, shade(steel, 1.3))


def _oil_stack(p: Painter, x: float, y: float, wide: float, tall: float) -> None:
    """A steel stack `wide` metres across standing on (x, y), drawn for `tall` metres and lowered
    (CHIMNEY_SHARE): black steel lit down its left, banded, scorched rust at its top under a lit
    lip."""
    m = METRE
    r = wide * m / math.sqrt(2)
    h = tall * m * CHIMNEY_SHARE
    _turned(p, x, y, [(0, r * 1.15), (h, r * 0.9)], OIL_CRUDE)
    _bar(p, (x - 0.5 * r, y), (x - 0.45 * r, y - h), 0.25 * m, shade(OIL_CRUDE, 1.6))
    _turned(p, x, y - h + 2.5 * m, [(0, r * 0.92), (2.5 * m, r * 0.9)], shade(OIL_RUST, 0.8))
    for z in np.arange(4 * m, h - 3 * m, 4 * m):
        _oil_band(p, x, y - z, r * (1.15 - 0.25 * z / h), 0.15 * m, shade(OIL_CRUDE, 0.6))
    _turned(p, x, y - h, [(0, r * 1.1), (0.5 * m, r * 1.1)], shade(OIL_CRUDE, 0.8), (30, 28, 30))


def _oil_furnace(p: Painter, x: float, y: float, long: float = 9) -> None:
    """A pipe still's furnace standing on (x, y), `long` by 5 m and 6 m tall: brick walls held by
    dark steel buckstays, fire doors glowing along the foot of its lit wall, a convection box on
    its roof and a slim steel stack rising from the back of it."""
    m = METRE
    length, depth, high = long * m, 5 * m, 6 * m
    top = p.hall(x, y, length, depth, high, OIL_BRICK, shade(OIL_STEEL, 0.9))
    left, front, right, _ = top
    stay = (52, 52, 56)
    for a, b, n in ((left, front, round(long / 1.5)), (front, right, 4)):
        for k in range(n):
            t = (k + 0.5) / n
            _bar(p, _on_wall(a, b, t, 0.2 * m), _on_wall(a, b, t, high), 0.32 * m, stay)
        _bar(p, _on_wall(a, b, 0, 0.25 * m), _on_wall(a, b, 1, 0.25 * m), 0.35 * m, stay)
        _bar(p, _on_wall(a, b, 0, high * 0.45), _on_wall(a, b, 1, high * 0.45), 0.2 * m, stay)
    doors = round(long / 1.8)
    for k in range(doors):
        t = (k + 1) / (doors + 1)
        _patch(
            p,
            left,
            front,
            (t - 0.36 / long, t + 0.36 / long),
            (high - 1.6 * m, high - 0.6 * m),
            (40, 30, 28),
        )
        _patch(
            p,
            left,
            front,
            (t - 0.22 / long, t + 0.22 / long),
            (high - 1.4 * m, high - 0.8 * m),
            OIL_FIRE,
        )
    sx, sy = _iso(x, y - high, (long / 2 - 1.9) * m, 1.6 * m)
    _oil_stack(p, sx, sy, 1.3, 34)
    box = p.hall(
        x - 0.8 * m,
        y - high,
        (long - 4) * m,
        3 * m,
        2.6 * m,
        shade(OIL_STEEL, 1.05),
        shade(OIL_STEEL, 1.3),
    )
    for a, b in ((box[0], box[1]), (box[1], box[2])):
        for t in (0.25, 0.5, 0.75):
            _bar(
                p, _on_wall(a, b, t, 0), _on_wall(a, b, t, 2.6 * m), 0.15 * m, shade(OIL_STEEL, 0.6)
            )


def _oil_frame(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """An open steel structure `length` by `depth` metres standing on (x, y), three decks high at
    5, 10 and 14 m: plated decks with lit edges and handrails, posts at its corners, crossed
    braces on its lit face, two insulated heat exchangers lying on its top deck and a drum on the
    middle one, with pumps under it."""
    m = METRE
    steel, rail = OIL_FRAME, OIL_HANDRAIL
    decks = (5, 10, 14)
    foot = _footing(x, y, length * m, depth * m)
    left, front, right, back = foot

    def up(pt: Point, z: float) -> Point:
        return pt[0], pt[1] - z * m

    for pt in (back, left, right):
        _bar(p, pt, up(pt, decks[-1] + 1.1), 0.4 * m, shade(steel, 0.8))
    for u in (0.3, 0.7):
        px, py = _on_wall(left, front, u, 0)
        cx, cy = _iso(px, py, 0, depth * m * 0.5)
        _turned(p, cx, cy, [(0, 0.7 * m), (1.1 * m, 0.7 * m), (1.4 * m, 0.4 * m)], OIL_STEEL)
    for n, z in enumerate(decks):
        plate = [up(pt, z) for pt in (left, front, right, back)]
        p.polygon(plate, shade(steel, 0.85), shade(steel, 0.6))
        if n == 1:
            cx, cy = _iso(*up(back, z), length * m * 0.35, -depth * m * 0.5)
            _turned(p, cx, cy, [(0, 1.1 * m), (2.4 * m, 1.1 * m), (2.8 * m, 0.6 * m)], OIL_RUST)
            _bar(
                p, (cx - 0.6 * m, cy), (cx - 0.6 * m, cy - 2.6 * m), 0.25 * m, shade(OIL_RUST, 1.4)
            )
        if n == 2:
            for v in (0.3, 0.7):
                a = _iso(*up(left, z + 0.9), length * m * 0.12, depth * m * v)
                b = _iso(*up(left, z + 0.9), length * m * 0.85, depth * m * v)
                _oil_lying(p, a, b, 0.75, OIL_ALUMINIUM)
        for a, b, tone in ((left, front, 1.3), (front, right, 1.05)):
            _bar(p, up(a, z), up(b, z), 0.4 * m, shade(steel, tone))
            _bar(p, up(a, z + 1.1), up(b, z + 1.1), 0.1 * m, rail)
    for pt in (left, front, right):
        _bar(
            p, pt, up(pt, decks[-1] + 1.1), 0.4 * m, shade(steel, 1.2 if pt is not right else 0.95)
        )
    lows = (0, *decks)
    for k in range(len(decks)):
        z0, z1 = lows[k], lows[k + 1]
        _bar(p, up(left, z0), up(front, z1), 0.18 * m, shade(steel, 1.1))
        _bar(p, up(front, z0), up(left, z1), 0.18 * m, shade(steel, 1.1))


def _oil_cracker(p: Painter, x: float, y: float) -> None:
    """A catalytic cracker standing on (x, y): an open steel structure 9 by 7 m and 14 m high
    carrying a fat regenerator of rusted steel and a tall slim reactor in aluminium paint, joined
    by a sloping riser, the regenerator's flue climbing to a stack of its own."""
    m = METRE
    _oil_frame(p, x, y, 9, 7)
    deck = 14 * m
    gx, gy = _iso(x, y - deck, -1.5 * m, 1 * m)
    rx, ry = _iso(x, y - deck, 2.5 * m, -0.5 * m)
    big = 6 * m / math.sqrt(2)
    _oil_stack(p, *_iso(gx, gy, -1 * m, 3.4 * m), 1.2, 30)
    shell = (104, 88, 78)
    _turned(
        p,
        gx,
        gy,
        [(0, big * 0.7), (1.5 * m, big), (9 * m, big), (11 * m, big * 0.55), (11.6 * m, 0)],
        shell,
    )
    _bar(
        p, (gx - 0.5 * big, gy - 1.5 * m), (gx - 0.5 * big, gy - 9 * m), 0.4 * m, shade(shell, 1.4)
    )
    for a in (0.2, 0.45, 0.7, 0.9):
        sx = gx + big * math.cos(a * math.pi)
        sy = gy + big / 2 * math.sin(a * math.pi)
        _bar(p, (sx, sy - 9 * m), (sx, sy - (6 + 2 * a) * m), 0.3 * m, shade(OIL_RUST, 0.85))
    for z in (4 * m, 8 * m):
        _oil_band(p, gx, gy - z, big + 0.8 * m, 0.3 * m, shade(OIL_STEEL, 0.55))
        _oil_band(p, gx, gy - z - 1 * m, big + 0.8 * m, 0.08 * m, OIL_HANDRAIL)
    _oil_pipe(p, [(gx + 0.3 * big, gy - 10.5 * m), (rx - 0.3 * m, ry - 15 * m)], OIL_ALUMINIUM, 0.9)
    slim = 3.2 * m / math.sqrt(2)
    _turned(
        p,
        rx,
        ry,
        [(0, slim), (17 * m, slim), (18.2 * m, slim * 0.55), (18.6 * m, 0)],
        OIL_ALUMINIUM,
    )
    _bar(
        p, (rx - 0.5 * slim, ry), (rx - 0.5 * slim, ry - 17 * m), 0.3 * m, shade(OIL_ALUMINIUM, 1.2)
    )
    for z in (5 * m, 10 * m, 15 * m):
        _oil_band(p, rx, ry - z, slim, 0.1 * m, shade(OIL_ALUMINIUM, 0.7))
        _oil_band(p, rx, ry - z, slim + 0.9 * m, 0.3 * m, shade(OIL_STEEL, 0.55))
        _oil_band(p, rx, ry - z - 1 * m, slim + 0.9 * m, 0.08 * m, OIL_HANDRAIL)


def _oil_sphere(p: Painter, x: float, y: float, wide: float) -> None:
    """A ball tank of liquid gas `wide` metres round standing on (x, y) on a ring of legs: aluminium
    paint lit on its upper left with a band round its middle, braced legs and a stair."""
    m = METRE
    rho = wide / 2
    half, tall = rho * math.sqrt(2) * m, rho * math.sqrt(1.5) * m
    cy = y - (rho + 1.2) * m
    legs = [k * math.pi / 4 for k in range(8)]
    steel = (96, 98, 104)
    for a in legs:
        lx, ly = x + 0.8 * half * math.cos(a), y + 0.8 * half / 2 * math.sin(a)
        if math.sin(a) < 0:
            _bar(p, (lx, ly), (lx, cy), 0.4 * m, shade(steel, 0.8))
    p.ellipse(x - half, cy - tall, x + half, cy + tall, shade(OIL_ALUMINIUM, 0.8))
    p.blob(x - half * 0.95, cy - tall * 0.97, x + half * 0.7, cy + tall * 0.75, OIL_ALUMINIUM)
    p.blob(
        x - half * 0.8, cy - tall * 0.9, x + half * 0.2, cy + tall * 0.1, shade(OIL_ALUMINIUM, 1.12)
    )
    p.blob(
        x - half * 0.6,
        cy - tall * 0.75,
        x - half * 0.15,
        cy - tall * 0.35,
        shade(OIL_ALUMINIUM, 1.32),
    )
    _oil_band(p, x, cy + 0.15 * m, half, 0.3 * m, OIL_AMBER)
    for a in legs:
        lx, ly = x + 0.8 * half * math.cos(a), y + 0.8 * half / 2 * math.sin(a)
        if math.sin(a) >= 0:
            _bar(p, (lx, ly), (lx, cy + half / 2 * math.sin(a)), 0.45 * m, steel)
    for k in range(4):
        a, b = legs[k], legs[k + 1]
        ax, ay = x + 0.8 * half * math.cos(a), y + 0.8 * half / 2 * math.sin(a)
        bx, by = x + 0.8 * half * math.cos(b), y + 0.8 * half / 2 * math.sin(b)
        _bar(p, (ax, ay - 0.3 * m), (bx, by - (rho * 0.9) * m), 0.12 * m, shade(steel, 1.2))
    sx = x - 1.05 * half
    _bar(p, (sx + 2.5 * m, y + 0.5 * m), (sx, cy - 0.2 * tall), 0.3 * m, (176, 172, 162))


def _oil_flare(p: Painter, x: float, y: float, tall: float) -> None:
    """A modest flare stack `tall` metres high standing on (x, y): a slim steel pipe held by guy
    wires, a small platform under its tip and a bright flame burning off gas at its top."""
    m = METRE
    h = tall * m
    wire = (120, 120, 124)
    for dx, dy in ((-4, 1.2), (3.6, 1.6), (0.8, -2.4)):
        _bar(p, (x + dx * m, y + dy * m), (x, y - h * 0.62), 0.1 * m, wire)
    _bar(p, (x, y), (x, y - h), 0.6 * m, (64, 64, 70))
    _bar(p, (x - 0.15 * m, y), (x - 0.15 * m, y - h), 0.18 * m, (130, 130, 138))
    for z in (h * 0.3, h * 0.55, h * 0.8):
        _bar(p, (x - 0.3 * m, y - z), (x + 0.3 * m, y - z), 0.7 * m, (196, 64, 52))
    _oil_band(p, x, y - h + 1.5 * m, 1.2 * m, 0.25 * m, (90, 90, 96))
    tip = y - h
    p.soft([(x - 2.4 * m, tip - 5 * m, x + 2.6 * m, tip + 0.6 * m)], OIL_FLAME, 0.45, 1.2)
    p.polygon([(x - 1.2 * m, tip), (x + 1.3 * m, tip), (x + 0.6 * m, tip - 4.2 * m)], OIL_FLAME)
    p.blob(x - 1.2 * m, tip - 2.4 * m, x + 1.3 * m, tip + 0.2 * m, OIL_FLAME)
    p.blob(x - 0.6 * m, tip - 1.6 * m, x + 0.7 * m, tip, OIL_GLOW)
    p.blob(x - 0.25 * m, tip - 0.9 * m, x + 0.3 * m, tip, (255, 252, 236))


def _oil_siding(p: Painter, a: Point, length: float) -> None:
    """A siding on ballast running `length` metres along the lit walls from `a`, standard gauge,
    with sleepers every metre and a buffer stop at its end."""
    m = METRE

    def at(s: float, k: float, z: float = 0) -> Point:
        px, py = _iso(a[0], a[1], s * m, k * m)
        return px, py - z * m

    p.polygon([at(0, -1.9), at(length, -1.9), at(length, 1.9), at(0, 1.9)], OIL_BALLAST)
    p.polygon([at(0, 1.9), at(length, 1.9), at(length, 1.4), at(0, 1.4)], shade(OIL_BALLAST, 0.85))
    for s in np.arange(0.5, length, 0.9):
        _bar(p, at(s, -1.3), at(s, 1.3), 0.28 * m, OIL_SLEEPER)
    for k in (-0.72, 0.72):
        _bar(p, at(0, k), at(length, k), 0.14 * m, OIL_RAIL)
        _bar(p, at(length - 0.2, k), at(length - 0.2, k, 1.1), 0.25 * m, (52, 50, 52))
    _bar(p, at(length - 0.2, -1, 0.9), at(length - 0.2, 1, 0.9), 0.45 * m, (196, 60, 48))


def _oil_lying(p: Painter, a: Point, b: Point, radius: float, colour: RGB) -> None:
    """A round barrel lying with its axis from `a` to `b` along the lit walls, `radius` metres
    round: lit along its top, its near end shaded."""
    m = METRE
    turns = np.linspace(0, 2 * math.pi, 24, endpoint=False)

    def ring(c: Point, rad: float) -> list[Point]:
        return [
            (c[0] + rad * m * math.cos(t), c[1] - rad * m / 2 * math.cos(t) - rad * m * math.sin(t))
            for t in turns
        ]

    hull = _oil_hull(ring(a, radius) + ring(b, radius))
    p.polygon(hull, colour, shade(colour, 0.55))
    for t, wide, tone in ((0.62, 0.5, 1.5), (0.7, 0.18, 2.1)):
        da = (
            radius * m * math.cos(t * math.pi),
            -radius * m / 2 * math.cos(t * math.pi) - radius * m * math.sin(t * math.pi),
        )
        _bar(
            p,
            (a[0] + da[0], a[1] + da[1]),
            (b[0] + da[0], b[1] + da[1]),
            wide * m,
            shade(colour, tone),
        )
    p.polygon(ring(b, radius), shade(colour, 0.78), shade(colour, 0.55))


def _oil_hull(points: list[Point]) -> list[Point]:
    """The convex hull of `points`, round it in order."""
    pts = sorted(set(points))

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper: list[Point] = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return lower[:-1] + upper[:-1]


def _oil_wagon(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A four-wheeled rail tank wagon 8 m long standing on the track at (x, y) along the lit
    walls: a dark frame on two axles with buffers, a barrel lit along its top with a dome and a
    walkway."""
    m = METRE

    def at(u: float, v: float, z: float) -> Point:
        px, py = _iso(x, y, u * m, v * m)
        return px, py - z * m

    frame = (38, 36, 38)
    for u in (-2.2, 2.2):
        for v in (0.75, -0.75):
            cx, cy = at(u, v, 0.48)
            p.ellipse(
                cx - 0.5 * m,
                cy - 0.48 * m,
                cx + 0.5 * m,
                cy + 0.48 * m,
                (44, 42, 44) if v > 0 else (60, 58, 60),
            )
            if v < 0:
                p.blob(cx - 0.15 * m, cy - 0.15 * m, cx + 0.15 * m, cy + 0.15 * m, (120, 118, 116))
    fx, fy = at(0, 0, 0.95)
    p.hall(fx, fy, 8 * m, 2.3 * m, 0.45 * m, frame, shade(frame, 1.4))
    for u in (-4.2, 4.2):
        for v in (-0.6, 0.6):
            _bar(p, at(u * 0.95, v, 1.2), at(u, v, 1.2), 0.3 * m, (70, 68, 70))
    _oil_lying(p, at(-3.6, 0, 2.5), at(3.6, 0, 2.5), 1.15, colour)
    dx, dy = at(0, 0, 3.55)
    _turned(p, dx, dy, [(0, 0.55 * m), (0.6 * m, 0.55 * m), (0.8 * m, 0.3 * m)], colour)
    p.blob(dx - 0.45 * m, dy - 0.9 * m, dx - 0.05 * m, dy - 0.5 * m, shade(colour, 2.2))
    _bar(p, at(-3, -0.5, 3.9), at(3, -0.5, 3.9), 0.1 * m, (150, 148, 140))


def _oil_loading_rack(p: Painter, a: Point, length: float, arms: tuple[float, ...]) -> None:
    """A loading rack `length` metres long beside a siding, along the lit walls from `a`: steel
    posts every four metres carrying a walkway 4.5 m up with its handrail, a header of black and
    amber pipes, loading arms swung down at `arms` metres along it, a stair at its end and an
    amber sample lamp glowing on its last post."""
    m = METRE

    def at(u: float, v: float, z: float) -> Point:
        px, py = _iso(a[0], a[1], u * m, v * m)
        return px, py - z * m

    steel, rail = OIL_FRAME, OIL_HANDRAIL
    posts = np.linspace(0, length, max(2, round(length / 4) + 1))
    for u in posts:
        _bar(p, at(u, 0, 0), at(u, 0, 5.6), 0.35 * m, steel)
    for z, colour in ((3.2, OIL_BLACK), (3.9, OIL_AMBER)):
        _bar(p, at(0, 0.1, z), at(length, 0.1, z), 0.45 * m, colour)
        _bar(p, at(0, 0.1, z + 0.2), at(length, 0.1, z + 0.2), 0.12 * m, shade(colour, 1.6))
    p.polygon(
        [at(0, 0.2, 4.6), at(length, 0.2, 4.6), at(length, -1.2, 4.6), at(0, -1.2, 4.6)],
        shade(steel, 0.85),
    )
    _bar(p, at(0, -1.2, 4.45), at(length, -1.2, 4.45), 0.3 * m, shade(steel, 1.5))
    for v in (0.2, -1.2):
        _bar(p, at(0, v, 5.6), at(length, v, 5.6), 0.1 * m, rail)
    for u in posts:
        _bar(p, at(u, -1.2, 4.6), at(u, -1.2, 5.6), 0.1 * m, rail)
    for u in arms:
        _bar(p, at(u, -0.4, 4.4), at(u + 0.4, -2.2, 4.9), 0.25 * m, OIL_BLACK)
        _bar(p, at(u + 0.4, -2.2, 4.9), at(u + 0.4, -3.0, 4.0), 0.25 * m, OIL_BLACK)
        _bar(p, at(u + 0.4, -2.2, 4.95), at(u + 0.4, -2.2, 5.5), 0.12 * m, rail)
    run = 4.4
    for k in range(8):
        f = (k + 0.5) / 8
        _bar(
            p,
            at(length + 0.2 + run * f, -1.1, 4.6 * (1 - f)),
            at(length + 0.2 + run * f, -0.1, 4.6 * (1 - f)),
            0.18 * m,
            (150, 148, 140),
        )
    for v in (-1.1, -0.1):
        _bar(p, at(length + 0.2, v, 4.6), at(length + 0.2 + run, v, 0), 0.16 * m, steel)
    _bar(p, at(length + 0.2, -1.1, 5.6), at(length + 0.2 + run, -1.1, 1), 0.1 * m, rail)
    lx, ly = at(length, 0, 5.9)
    p.soft([(lx - 1.3 * m, ly - 1.3 * m, lx + 1.3 * m, ly + 1.3 * m)], OIL_AMBER, 0.55, 0.8)
    p.blob(lx - 0.4 * m, ly - 0.4 * m, lx + 0.4 * m, ly + 0.4 * m, OIL_GLOW)


def _oil_pipe(p: Painter, points: list[Point], colour: RGB, wide: float = 0.45) -> None:
    """A pipe `wide` metres thick run through `points`, lit along its top."""
    m = METRE
    lift = 0.3 * wide * m
    for k in range(len(points) - 1):
        _bar(p, points[k], points[k + 1], wide * m, colour)
    for k in range(len(points) - 1):
        (ax, ay), (bx, by) = points[k], points[k + 1]
        _bar(p, (ax, ay - lift), (bx, by - lift), lift, shade(colour, 1.7))


def _oil_link(
    p: Painter, a: Point, b: Point, gap: tuple[float, float], z: float, colour: RGB | None = None
) -> None:
    """A pipe in `colour` `z` metres up between two round things standing on `a` and `b`, from the
    skin of the first, `gap[0]` metres from its axis, to the skin of the second, `gap[1]` from
    its; or with no `colour` a catwalk bridge of steel grating with handrails on both sides."""
    m = METRE
    dx, dy = b[0] - a[0], b[1] - a[1]
    du, dv = (dx + 2 * dy) / 2, (dx - 2 * dy) / 2
    run = math.hypot(du, dv) or 1.0
    du, dv = du / run, dv / run

    def along(pt: Point, k: float, side: float, up: float) -> Point:
        px, py = _iso(pt[0], pt[1], (du * k - dv * side) * m, (dv * k + du * side) * m)
        return px, py - up * m

    start, end = (a, gap[0]), (b, -gap[1])
    if colour is not None:
        _oil_pipe(p, [along(*start, 0, z), along(*end, 0, z)], colour)
        return
    steel, rail = OIL_FRAME, OIL_HANDRAIL
    far = 1 if du + dv > 0 else -1
    _bar(p, along(*start, 0.6 * far, z + 1.05), along(*end, 0.6 * far, z + 1.05), 0.1 * m, rail)
    p.polygon(
        [along(*start, 0.6, z), along(*end, 0.6, z), along(*end, -0.6, z), along(*start, -0.6, z)],
        shade(steel, 0.9),
    )
    _bar(
        p,
        along(*start, -0.6 * far, z - 0.15),
        along(*end, -0.6 * far, z - 0.15),
        0.3 * m,
        shade(steel, 1.45),
    )
    _bar(p, along(*start, -0.6 * far, z + 1.05), along(*end, -0.6 * far, z + 1.05), 0.1 * m, rail)


def _oil_yard(p: Painter, corners: list[Point]) -> None:
    """The gravelled ground of a refinery over the four `corners` on the ground, with a dark lip."""
    m = METRE
    p.polygon([(x, y + 0.7 * m) for x, y in corners], shade(OIL_GRAVEL, 0.62))
    p.polygon(corners, OIL_GRAVEL, shade(OIL_GRAVEL, 0.85))


def refinery() -> Sprite:
    """An oil refinery, 31 by 27 m on a gravelled yard: a dense tangle of five distillation
    columns of dark steel and rust from 20 to 34 m tall, ringed with catwalks, climbed by ladders
    and laced with pipes and pipe racks round an open steel structure of heat exchangers; a
    brick furnace with fire doors glowing and its slim steel stack; one big black crude tank under
    a shallow dome with a stair winding up it inside its earth bund; two product tanks banded
    amber; a loading rack with its amber lamp over two of three tank wagons on its siding; and a
    modest flare burning at the back."""
    m = METRE
    p = Painter(56, 76, (28, 56))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 15.5) * m, (v - 13.5) * m)

    _oil_yard(p, [at(0, 0), at(31, 0), at(31, 27), at(0, 27)])
    for u0, v0, u1, v1 in ((16.5, 11.5, 26.5, 26.5), (14.5, 4.5, 26.5, 11.5)):
        p.polygon([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)], OIL_PAD, shade(OIL_PAD, 0.8))
    for u, v, r in ((17, 6.5, 1.6), (25, 19, 1.2), (12.5, 25, 1.4), (30, 5.5, 1.3)):
        _oil(p, *at(u, v), r * m)
    _oil_flare(p, *at(3, 24.5), 30)
    a, b, c, d, e = at(14.5, 23), at(19.5, 24.5), at(27.5, 25), at(24, 22.5), at(25, 18.5)
    _oil_column(p, *a, 4.2, 24)
    _oil_link(p, a, b, (2.1, 1.7), 11, OIL_BLACK)
    _oil_column(p, *b, 3.4, 34)
    _oil_column(p, *c, 2.4, 22)
    _oil_link(p, b, d, (1.7, 1.5), 14, OIL_RUST)
    _oil_link(p, b, d, (1.7, 1.5), 18)
    _oil_column(p, *d, 3, 29, OIL_WEATHERED)
    _oil_link(p, d, c, (1.5, 1.2), 12)
    _oil_rack(
        p,
        at(14, 19.5),
        at(23, 19.5),
        _iso(0, 0, 0, 1.6 * m),
        7,
        OIL_PIPES,
    )
    _oil_rack(
        p, at(15, 7.5), at(15, 25), _iso(0, 0, 1.4 * m, 0), 5.5, (OIL_BLACK, OIL_AMBER, OIL_RUST)
    )
    _oil_crude_tank(p, *at(6.5, 15), 12, 8)
    _oil_link(p, d, e, (1.5, 1.3), 6)
    _oil_column(p, *e, 2.6, 20, OIL_WEATHERED)
    _oil_frame(p, *at(21.5, 15), 7, 5)
    _oil_product_tank(p, *at(29, 17.5), 6.5, 8, OIL_AMBER)
    _oil_furnace(p, *at(21, 8.5))
    _oil_product_tank(p, *at(28.5, 10), 5.5, 6.5, OIL_KEROSENE)
    _oil_loading_rack(p, at(3.5, 5), 14.5, (3.5, 12.5))
    _oil_siding(p, at(0, 2), 31)
    for u in (7, 16, 25.5):
        _oil_wagon(p, *at(u, 2), OIL_WAGON)
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
