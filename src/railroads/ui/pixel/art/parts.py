"""Pieces several kinds of sprite are built from."""

import math
from functools import partial

import numpy as np

from railroads.ui.pixel.art.painter import (
    BARREL,
    CROSS,
    DOORWAY,
    MILITARY_FLAG,
    RGB,
    WINDOW,
    Painter,
    Point,
    Top,
    shade,
)


def _flag(p: Painter, x: float, y: float, height: float, colour: RGB = MILITARY_FLAG) -> None:
    p.line([(x, y), (x, y - height)], (70, 66, 62))
    p.polygon(
        [(x + 1, y - height), (x + 7, y - height + 2), (x + 1, y - height + 4)],
        colour,
    )


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


SILL: RGB = (240, 236, 226)
DOOR: RGB = (104, 66, 44)
LIT: RGB = (236, 206, 120)


def _band(p: Painter, a: Point, b: Point, drop: float, colour: RGB) -> None:
    """A line right along the wall under the edge from `a` to `b`, `drop` down it."""
    p.line([_on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop)], colour)


def _bands(p: Painter, a: Point, b: Point, c: Point, drop: float, lit: RGB, shaded: RGB) -> None:
    """Bands `drop` down the lit wall under `a`-`b` and then the shaded wall under `b`-`c`."""
    _band(p, a, b, drop, lit)
    _band(p, b, c, drop, shaded)


def _plinth(
    p: Painter,
    top: Top,
    height: float,
    wall: RGB,
    depth: float = 0.8,
    lit: float = 0.7,
    dark: float = 0.55,
) -> None:
    """A darker band `depth` high along the foot of a block's lit and shaded walls, `height`
    below its top."""
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (height - depth, height), shade(wall, lit))
    _patch(p, front, right, (0, 1), (height - depth, height), shade(wall, dark))


# The plinth of shops, landmarks and barracks, a touch lighter on the shaded wall.
_pale_plinth = partial(_plinth, dark=0.52)


def _ridge(top: Top, rise: float) -> tuple[Point, Point]:
    """The back and front ends of the ridge of a gable `rise` high on a box's top (Painter.gable)."""
    left, front, right, back = top
    return (
        ((left[0] + back[0]) / 2, (left[1] + back[1]) / 2 - rise),
        ((front[0] + right[0]) / 2, (front[1] + right[1]) / 2 - rise),
    )


def _tiles(
    p: Painter, top: Top, rise: float, roof: RGB, rows: tuple[float, ...] = (0.34, 0.68)
) -> None:
    """Rows of tiles, slates or thatch across the lit side of a gable roof."""
    left, front, _, _ = top
    ridge_back, ridge_front = _ridge(top, rise)
    for f in rows:
        a = (left[0] + (ridge_back[0] - left[0]) * f, left[1] + (ridge_back[1] - left[1]) * f)
        b = (front[0] + (ridge_front[0] - front[0]) * f, front[1] + (ridge_front[1] - front[1]) * f)
        p.line([a, b], shade(roof, 0.8))


def _window(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float, fill: RGB = WINDOW
) -> None:
    """A window with a light sill under it on the wall under the edge from `a` to `b`."""
    _patch(p, a, b, span, (drop, drop + 1.2), fill)
    p.line([_on_wall(a, b, span[0], drop + 1.5), _on_wall(a, b, span[1], drop + 1.5)], SILL)


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


def _vault(
    p: Painter,
    top: Top,
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


def _heap(p: Painter, x: float, y: float, half: float, height: float, colour: RGB) -> None:
    """A conical heap of loose stuff, lit on the left."""
    foot = (x + half * 0.3, y + half * 0.3)
    p.polygon([(x - half, y), (x - 1, y - height), foot], colour, shade(colour, 0.62))
    p.polygon(
        [(x - 1, y - height), (x + 1, y - height), (x + half, y), foot],
        shade(colour, 0.8),
        shade(colour, 0.55),
    )


def _iso(x: float, y: float, u: float, v: float) -> Point:
    """The point `u` pixels across along a lit wall and `v` along a shaded one from (x, y)."""
    return x + u + v, y + u / 2 - v / 2


def _gable_across(p: Painter, top: Top, rise: float, roof: RGB, wall: RGB) -> None:
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


# How many painter units a metre takes in a sprite drawn at BUILDING_SCALE, measured by the
# house: its 8 m wall is 6 units drawn at HOME_SCALE (railroads.ui.pixel.art.registry).
METRE = 6 * 0.413 / 8 / 0.7


def _bar(p: Painter, a: Point, b: Point, wide: float, colour: RGB) -> None:
    """A straight bar `wide` units thick from `a` to `b`, thinner than Painter.line can draw."""
    (ax, ay), (bx, by) = a, b
    length = math.hypot(bx - ax, by - ay) or 1.0
    nx, ny = -(by - ay) / length * wide / 2, (bx - ax) / length * wide / 2
    p.polygon(
        [(ax + nx, ay + ny), (bx + nx, by + ny), (bx - nx, by - ny), (ax - nx, ay - ny)], colour
    )


def _legs(p: Painter, x: float, y: float, half: float, low: float, m: float, colour: RGB) -> None:
    """Four legs under an animal's body `half` long either side of (x, y), `low` tall."""
    for dx, dy in ((-0.78, -0.12), (0.62, -0.12), (-0.62, 0.12), (0.78, 0.12)):
        foot = (x + dx * half, y + dy * m)
        _bar(p, foot, (foot[0], foot[1] - low), 0.13 * m, colour)


def _cow(
    p: Painter, x: float, y: float, hide: RGB, patch: RGB, metre: float = METRE, facing: int = -1
) -> None:
    """A cow standing at (x, y) on the ground, at its true size beside the houses, `metre` units
    to a metre: 2.4 m long and 1.4 m to the shoulder, on four legs, with patches on its hide,
    a head with a pale muzzle and short horns, and a tail; facing left, or right with `facing`
    1."""
    m, f = metre, facing
    half, high, belly = 0.95 * m, 1.4 * m, 0.7 * m
    p.blob(x - 1.25 * half, y - 0.22 * m, x + 1.25 * half, y + 0.3 * m, (62, 84, 40))
    _legs(p, x, y, half, belly, m, shade(hide, 0.55))
    p.blob(x - half, y - high, x + half, y - belly + 0.12 * m, hide)
    p.blob(
        x - 0.9 * half, y - belly - 0.3 * m, x + 0.9 * half, y - belly + 0.12 * m, shade(hide, 0.8)
    )
    p.blob(
        x - 0.8 * half, y - high + 0.02 * m, x + 0.5 * half, y - high + 0.3 * m, shade(hide, 1.12)
    )
    p.blob(x - 0.1 * m, y - high + 0.15 * m, x + 0.55 * m, y - belly - 0.05 * m, patch)
    p.blob(x - 0.85 * m, y - high + 0.1 * m, x - 0.45 * m, y - high + 0.45 * m, patch)
    hx = x + f * (half + 0.22 * m)
    p.blob(hx - 0.3 * m, y - high - 0.08 * m, hx + 0.3 * m, y - high + 0.5 * m, hide)
    p.blob(
        hx + f * 0.12 * m - 0.17 * m,
        y - high + 0.26 * m,
        hx + f * 0.12 * m + 0.17 * m,
        y - high + 0.52 * m,
        (214, 176, 164),
    )
    for side in (-0.2, 0.2):
        _bar(
            p,
            (hx + side * m, y - high - 0.02 * m),
            (hx + side * 1.5 * m, y - high - 0.2 * m),
            0.07 * m,
            (236, 226, 200),
        )
    tail = x - f * half
    _bar(
        p,
        (tail, y - high + 0.15 * m),
        (tail - f * 0.12 * m, y - belly - 0.1 * m),
        0.08 * m,
        shade(hide, 0.6),
    )


def _pig(p: Painter, x: float, y: float, metre: float = METRE, facing: int = -1) -> None:
    """A pink pig standing at (x, y), 1.4 m long and 0.7 m high: a round body lit on top, short
    legs, a snout, a flopped ear and a curled tail."""
    m, f = metre, facing
    skin: RGB = (234, 168, 160)
    half, high, belly = 0.65 * m, 0.75 * m, 0.25 * m
    p.blob(x - 1.2 * half, y - 0.18 * m, x + 1.2 * half, y + 0.22 * m, (92, 74, 54))
    _legs(p, x, y, half * 0.9, belly, m, shade(skin, 0.6))
    p.blob(x - half, y - high, x + half, y - belly + 0.05 * m, skin)
    p.blob(
        x - 0.85 * half,
        y - belly - 0.18 * m,
        x + 0.85 * half,
        y - belly + 0.05 * m,
        shade(skin, 0.82),
    )
    p.blob(
        x - 0.7 * half, y - high + 0.02 * m, x + 0.4 * half, y - high + 0.22 * m, shade(skin, 1.08)
    )
    hx = x + f * (half + 0.05 * m)
    p.blob(hx - 0.24 * m, y - high + 0.12 * m, hx + 0.24 * m, y - belly - 0.05 * m, skin)
    snout = hx + f * 0.24 * m
    p.blob(
        snout - 0.09 * m,
        y - high + 0.3 * m,
        snout + 0.09 * m,
        y - high + 0.48 * m,
        shade(skin, 0.85),
    )
    p.polygon(
        [
            (hx - 0.05 * m, y - high + 0.14 * m),
            (hx + f * 0.18 * m, y - high + 0.08 * m),
            (hx + f * 0.1 * m, y - high + 0.3 * m),
        ],
        shade(skin, 0.75),
    )
    tail = x - f * half
    _bar(
        p,
        (tail, y - high + 0.2 * m),
        (tail - f * 0.1 * m, y - high + 0.1 * m),
        0.06 * m,
        shade(skin, 0.7),
    )


def _hen(
    p: Painter, x: float, y: float, feathers: RGB, metre: float = METRE, facing: int = -1
) -> None:
    """A hen pecking about at (x, y), 0.45 m long: a plump body, a tail of feathers held up, a
    red comb and wattle and a yellow beak."""
    m, f = metre, facing
    p.blob(x - 0.22 * m, y - 0.05 * m, x + 0.22 * m, y + 0.08 * m, (96, 80, 60))
    _bar(p, (x, y), (x, y - 0.12 * m), 0.04 * m, (214, 160, 60))
    p.blob(x - 0.22 * m, y - 0.42 * m, x + 0.2 * m, y - 0.1 * m, feathers)
    p.blob(x - 0.16 * m, y - 0.4 * m, x + 0.06 * m, y - 0.28 * m, shade(feathers, 1.08))
    p.polygon(
        [
            (x - f * 0.14 * m, y - 0.3 * m),
            (x - f * 0.3 * m, y - 0.55 * m),
            (x - f * 0.22 * m, y - 0.25 * m),
        ],
        shade(feathers, 0.8),
    )
    hx = x + f * 0.2 * m
    p.blob(hx - 0.09 * m, y - 0.52 * m, hx + 0.09 * m, y - 0.36 * m, feathers)
    p.blob(hx - 0.06 * m, y - 0.6 * m, hx + 0.06 * m, y - 0.5 * m, (206, 40, 36))
    p.polygon(
        [
            (hx + f * 0.07 * m, y - 0.46 * m),
            (hx + f * 0.16 * m, y - 0.43 * m),
            (hx + f * 0.07 * m, y - 0.4 * m),
        ],
        (230, 176, 50),
    )


def _paddock(x: float, y: float, half: float) -> Top:
    """The left, back, right and front corners of a fenced patch of ground centred at (x, y)."""
    return (x - half, y), (x, y - half / 2), (x + half, y), (x, y + half / 2)


def _fence(p: Painter, a: Point, b: Point, metre: float | None = None) -> None:
    """A post-and-rail fence along the ground from `a` to `b`: four posts and a rail far out, or
    at `metre` units to a metre a fence at its true size, a post every 2.5 m, 1.2 m tall, with two
    rails."""
    if metre is None:
        for k in range(4):
            px, py = _on_wall(a, b, k / 3, 0)
            p.line([(px, py), (px, py - 3)], (122, 90, 60))
        p.line([_on_wall(a, b, 0, -2), _on_wall(a, b, 1, -2)], (206, 176, 126))
        return
    m = metre
    span = math.hypot(b[0] - a[0], (b[1] - a[1]) * 2) / m
    posts = max(2, round(span / 2.5) + 1)
    post, rail = (112, 84, 56), (196, 166, 118)
    for k in range(posts):
        px, py = _on_wall(a, b, k / (posts - 1), 0)
        _bar(p, (px, py), (px, py - 1.2 * m), 0.14 * m, post)
    for drop in (0.55, 1.05):
        _bar(p, _on_wall(a, b, 0, -drop * m), _on_wall(a, b, 1, -drop * m), 0.09 * m, rail)


def _plate(p: Painter, fill: RGB, half: float = 21, sides: int = 4) -> None:
    """The paved or planted ground of a site filling its whole cell: a slab with a dark lip, a
    diamond when it has four `sides` and nearly an oval when it has many."""
    rim = [
        (half * math.cos(a), half / 2 * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, sides + 1)[:-1]
    ]
    p.polygon([(x, y + 1) for x, y in rim], shade(fill, 0.68))
    p.polygon(rim, fill, shade(fill, 0.85))
