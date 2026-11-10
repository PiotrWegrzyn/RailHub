"""Pieces several kinds of sprite are built from."""

import math

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


def _plate(p: Painter, fill: RGB, half: float = 21, sides: int = 4) -> None:
    """The paved or planted ground of a site filling its whole cell: a slab with a dark lip, a
    diamond when it has four `sides` and nearly an oval when it has many."""
    rim = [
        (half * math.cos(a), half / 2 * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, sides + 1)[:-1]
    ]
    p.polygon([(x, y + 1) for x, y in rim], shade(fill, 0.68))
    p.polygon(rim, fill, shade(fill, 0.85))
