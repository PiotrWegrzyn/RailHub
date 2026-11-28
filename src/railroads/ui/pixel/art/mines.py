"""Mines, wells, pits, quarries, logging camps and sawmills."""

import itertools
import math
from collections.abc import Callable

import numpy as np

from railroads.ui.pixel.art.farms import (
    FOOTING,
    WORN,
    _boards,
    _chimney,
    _door,
    _footing,
    _plinth,
    _ruts,
    _strewn,
    _track,
    _worn,
)
from railroads.ui.pixel.art.painter import (
    BARREL,
    BRICK,
    DOORWAY,
    METAL,
    RGB,
    TIMBER,
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
    DOOR,
    METRE,
    _bar,
    _heap,
    _iso,
    _on_wall,
    _patch,
    _ridge,
    _tiles,
    _turned,
    _window,
)

RAIL: RGB = (138, 136, 144)
SLEEPER: RGB = (92, 70, 50)


def sawmill() -> Sprite:
    """A boarded steam sawmill with a door, a window and a chimney, its circular saw
    showing through the open side, with a stack of logs cut to show their rings, a heap of
    sawdust and a pile of sawn planks."""
    p = Painter(28, 24, (13, 19))
    sawdust = (222, 194, 140)
    _ruts(p, [(5, 3.2), (9, 4.4), (13, 5.4)])
    _track(p, [(-5.2, 3), (-6, 4.6), (-5.4, 6)])
    _worn(p, -5.2, 3, 2.2)
    p.blob(-2.4, 1.8, 6.4, 6, WORN)
    p.blob(-1, 2.4, 5.4, 5.4, shade(sawdust, 0.92))
    p.blob(2.6, 0.2, 14.4, 3.4, FOOTING)
    p.blob(-11.4, 1.6, -3.6, 5, FOOTING)
    p.shadow(11, 3)
    roof = (96, 84, 70)
    top = p.box(-3, 0, 7, 6, TIMBER, None)
    left, front, right, _ = top
    _footing(p, top, 6)
    _boards(p, top, TIMBER, (1.5, 3, 4.5))
    _plinth(p, top, TIMBER, 6)
    _window(p, left, front, (0.12, 0.3), 1.2)
    _door(p, left, front, (0.58, 0.78), 1.8, 6)
    _patch(p, front, right, (0.12, 0.72), (1.4, 6), DOORWAY)
    bx, by = _on_wall(front, right, 0.42, 3.6)
    p.ellipse(bx - 1.8, by - 1.8, bx + 1.8, by + 1.8, METAL)
    p.point(bx - 1, by - 1, shade(METAL, 1.3))
    p.point(bx, by, shade(METAL, 0.45))
    p.line([_on_wall(front, right, 0.12, 5), _on_wall(front, right, 0.72, 5)], (138, 96, 58))
    p.gable(top, 4, roof, TIMBER)
    _tiles(p, top, 4, roof)
    _chimney(p, top, 4, 0.2, 2.5)
    _log_pile(p, 6.0, 0.2, (4, 3, 2), 7)
    plank = (200, 166, 116)
    p.hall(-8, 2.4, 5, 1.6, 1.4, plank, shade(plank, 1.1))
    p.line([(-11, 1.9), (-6, 4.4)], shade(plank, 0.7))
    _heap(p, 1.5, 4.5, 2, 1.8, sawdust)
    _strewn(p, 1.4, 3.6, 4.4, shade(sawdust, 1.06), 7)
    return p.sprite()


def _log_pile(p: Painter, x: float, y: float, rows: tuple[int, ...], long: float) -> None:
    """Felled trunks `long` metres long and 0.8 m thick stacked in a pyramid at (x, y), `rows`
    trunks a row from the bottom up, lying along the ground to the front right with their cut
    ends showing their rings."""
    m = METRE
    thick = 0.8 * m
    log, end = (128, 88, 52), (220, 184, 130)
    for row, count in enumerate(rows):
        for k in reversed(range(count)):
            bx = x - (k + row / 2) * thick * 0.9
            by = y + (k + row / 2) * thick * 0.45 - row * thick * 0.8
            ex, ey = bx + long * m, by + long * m / 2
            p.polygon(
                [(bx, by - thick), (ex, ey - thick), (ex, ey), (bx, by)], log, shade(log, 0.65)
            )
            p.line([(bx, by - thick * 0.6), (ex, ey - thick * 0.6)], shade(log, 1.15))
            p.ellipse(ex - thick / 2, ey - thick, ex + thick / 2, ey, end)
            p.point(ex, ey - thick / 2, shade(end, 0.7))


def _cabin(
    p: Painter, x: float, y: float, long: float, deep: float, roof: RGB, chimney: bool
) -> tuple[float, float, float, float]:
    """A log cabin `long` by `deep` metres standing at (x, y) under a shingle `roof`, with a door
    and a window, and a stone chimney if `chimney`; returns where its chimney top is, or its ridge
    middle, and the ground under its lit corner."""
    m = METRE
    high, rise = 2.6 * m, 2.0 * m
    wall = shade(TRUNK, 1.18)
    top = p.hall(x, y, long * m, deep * m, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high)
    _boards(p, top, wall, (high / 4, high / 2, high * 3 / 4))
    for corner in (left, front, right):
        p.line([corner, (corner[0], corner[1] + high)], shade(wall, 0.7))
    _window(p, left, front, (0.15, 0.3), 0.7 * m)
    _door(p, left, front, (0.55, 0.72), high - 1.9 * m, high)
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof)
    if chimney:
        cx, cy = _chimney(p, top, rise, 0.8, 1.4, (128, 124, 116))
    else:
        cx, cy = _ridge(top, rise)[1]
    return cx, cy, front[0], front[1] + high


def logging_camp() -> Sprite:
    """A logging camp in a clearing of fresh stumps: a long log bunkhouse with a smoking stone
    chimney and a cook shack beside it, an open shed over a sawhorse and a grindstone, felled
    trunks stacked high on a log deck and more loaded on a timber wagon waiting for the road,
    and an axe left in a stump."""
    p = Painter(36, 26, (17, 17))
    m = METRE
    _ruts(p, [(5, 4.4), (9, 5.6), (14, 6.8)])
    _worn(p, -1, 2.6, 4.6)
    _worn(p, 6, 2.8, 5)
    p.shadow(12, 3)
    shingle = (84, 70, 52)
    sx, sy, _, _ = _cabin(p, -6.5, -3.0, 10, 5, shingle, True)
    smoke = (214, 214, 218)
    for i in range(3):
        r = 0.9 + 0.4 * i
        p.blob(
            sx + 0.8 * i - r, sy - 1.6 * i - 1 - r, sx + 0.8 * i + r, sy - 1.6 * i - 1 + r, smoke
        )
    _cabin(p, -12.5, 1.6, 5, 4, shade(shingle, 1.1), True)
    post, roof = (96, 74, 50), (110, 92, 70)
    shed = [(-3.6, 3.2), (0.4, 1.2), (0.4, 4.4), (-3.6, 6.4)]
    for px, py in shed:
        p.line([(px, py), (px, py - 2.4 * m)], post)
    p.polygon([(x, y - 2.4 * m) for x, y in shed], roof, shade(roof, 0.65))
    p.line([(-2.6, 4.8), (-1.0, 4.0)], (150, 112, 70))
    p.line([(-2.4, 4.0), (-1.2, 5.0)], (150, 112, 70))
    p.ellipse(-0.6, 3.2, 0.4, 4.4, (150, 146, 140))
    _log_pile(p, 7.0, -1.4, (5, 4, 3, 2), 8)
    wx, wy = 1.8, 5.4
    wagon = (110, 82, 56)
    p.hall(wx + 2.4, wy + 1.0, 6 * m, 1.8 * m, 0.4 * m, wagon, shade(wagon, 1.1))
    _log_pile(p, wx, wy - 0.2, (3, 2), 7)
    for cx in (wx + 0.8, wx + 4.6):
        p.ellipse(cx - 0.6 * m, wy + 1.4, cx + 0.6 * m, wy + 1.4 + 1.2 * m, (70, 52, 38))
    log, end = (128, 88, 52), (220, 184, 130)
    for x, y in ((-14.6, 4.6), (-7.2, 5.8), (12.6, -3.6), (-15.6, -1.6), (13.8, 3.4), (-9.6, 6.6)):
        p.rect(x - 0.4, y - 0.6, x + 0.4, y, shade(log, 0.9))
        p.ellipse(x - 0.5, y - 0.9, x + 0.5, y - 0.4, end)
    p.line([(-7.0, 5.0), (-6.3, 4.3)], (112, 84, 54))
    p.rect(-7.5, 4.8, -6.8, 5.1, METAL)
    return p.sprite()


OIL_BLACK: RGB = (34, 30, 30)
OILY_EARTH: RGB = (112, 96, 76)
DERRICK: RGB = (104, 80, 60)
TANK_WOOD: RGB = (132, 92, 58)
GOLD: RGB = (226, 180, 70)


def _oil_drop(p: Painter, x: float, y: float, size: float) -> None:
    """A drop of black oil `size` units tall with its foot at (x, y), rimmed in gold, with a
    golden glint on its left."""
    for grow, colour in ((1.18, GOLD), (1.0, OIL_BLACK)):
        s = size * grow
        cy = y - size * 0.36
        p.blob(x - 0.36 * s, cy - 0.36 * s, x + 0.36 * s, cy + 0.36 * s, colour)
        p.polygon(
            [(x - 0.33 * s, cy - 0.12 * s), (x + 0.33 * s, cy - 0.12 * s), (x, cy - 0.66 * s)],
            colour,
        )
    _bar(
        p, (x - 0.16 * size, y - 0.25 * size), (x - 0.2 * size, y - 0.48 * size), 0.08 * size, GOLD
    )


def _oil_puddle(p: Painter, x: float, y: float, half: float) -> None:
    """A puddle of spilt crude `half` across on the ground at (x, y), black with a dull sheen."""
    p.blob(x - half, y - half / 2, x + half, y + half / 2, OIL_BLACK)
    p.blob(x - half * 0.5, y - half * 0.25, x - half * 0.05, y - half * 0.05, (70, 64, 84))


def _derrick(p: Painter, x: float, y: float, tall: float) -> None:
    """A timber oil derrick `tall` units high over a well at (x, y): a plank floor, four raking
    legs braced crosswise storey by storey and blackened with oil at their feet, a crown block
    at its top, and beside it a walking beam rocking on a samson post, worked by a pitman rod
    from a band wheel."""
    m = METRE
    p.hall(x, y, 4.6 * m, 4.6 * m, 0.3 * m, shade(DERRICK, 0.9), shade(DERRICK, 1.1))
    _oil_puddle(p, x - 0.4 * m, y - 0.5 * m, 1 * m)
    storeys = tuple(np.linspace(0, 1, max(3, round(tall / m / 5)) + 1))
    _trestle(
        p,
        _corners(x, y - 0.3 * m, 3.6 * m, 3.6 * m),
        _corners(x, y, 1 * m, 1 * m, tall),
        storeys,
        DERRICK,
    )
    p.hall(x, y - tall, 1.5 * m, 1.5 * m, 0.7 * m, shade(DERRICK, 0.8), shade(DERRICK, 1.15))
    _bar(p, (x, y - 0.3 * m), (x, y - 1.6 * m), 0.3 * m, (60, 58, 60))
    post = _iso(x, y, 3.4 * m, 0)
    for side in (-0.5, 0.5):
        foot = _iso(*post, 0, side * m)
        _bar(p, foot, (post[0], post[1] - 3.6 * m), 0.22 * m, shade(DERRICK, 0.85))
    beam_near, beam_far = (x, y - 3.4 * m), _iso(post[0], post[1] - 3.9 * m, 3 * m, 0)
    _bar(p, beam_near, beam_far, 0.4 * m, shade(DERRICK, 1.05))
    _bar(p, (x, y - 3.4 * m), (x, y - 1.6 * m), 0.08 * m, (60, 58, 60))
    wx, wy = _iso(post[0], post[1], 3.4 * m, 0)
    _bar(p, beam_far, (wx + 0.3 * m, wy - 1.4 * m), 0.14 * m, shade(DERRICK, 0.8))
    r = 1.2 * m
    p.ellipse(wx - r, wy - 1.4 * m - r, wx + r, wy - 1.4 * m + r, shade(DERRICK, 0.75))
    p.blob(
        wx - 0.7 * r,
        wy - 1.4 * m - 0.7 * r,
        wx + 0.7 * r,
        wy - 1.4 * m + 0.7 * r,
        shade(DERRICK, 0.55),
    )
    p.blob(
        wx - 0.15 * r,
        wy - 1.4 * m - 0.15 * r,
        wx + 0.15 * r,
        wy - 1.4 * m + 0.15 * r,
        shade(DERRICK, 1.1),
    )


def _wood_tank(p: Painter, x: float, y: float, r: float, tall: float) -> None:
    """A round wooden storage tank `r` across and `tall` high on (x, y): upright staves bound
    with three iron hoops, lit on its left, under a low conical roof of tarred boards."""
    m = METRE
    split = math.acos(1 / 3)
    front = [(x + r * math.cos(a), y + r / 2 * math.sin(a)) for a in np.linspace(math.pi, 0, 13)]
    lit = [pt for pt in front if pt[0] <= x + r / 3 + 1e-6]
    dark = [pt for pt in front if pt[0] >= x + r / 3 - 1e-6]
    p.polygon([(x - r, y - tall), *lit, (x + r / 3, y - tall + r / 2 * math.sin(split))], TANK_WOOD)
    p.polygon(
        [(x + r / 3, y - tall + r / 2 * math.sin(split)), *dark, (x + r, y - tall)],
        shade(TANK_WOOD, 0.74),
    )
    for k in range(1, 8):
        a = math.pi * k / 8
        sx, sy = x + r * math.cos(a), y + r / 2 * math.sin(a)
        _bar(
            p, (sx, sy), (sx, sy - tall), 0.04 * m, shade(TANK_WOOD, 0.8 if sx < x + r / 3 else 0.6)
        )
    for f in (0.15, 0.5, 0.85):
        _bar_line = [(px, py - tall * f) for px, py in front]
        for a, b in itertools.pairwise(_bar_line):
            _bar(p, a, b, 0.1 * m, (52, 50, 54))
    roof = [
        (x + r * 1.06 * math.cos(a), y - tall + r * 0.53 * math.sin(a))
        for a in np.linspace(math.pi, 0, 13)
    ]
    apex = (x, y - tall - r * 0.45)
    p.polygon([*roof, apex], (58, 52, 50))
    p.polygon([roof[0], *roof[:7], apex], (70, 64, 60))


def _oil_barrels(p: Painter, x: float, y: float, count: int) -> None:
    """A row of `count` wooden oil barrels 0.9 m high standing along the lit walls from (x, y),
    each bound with two hoops and stained black at its top."""
    m = METRE
    for k in range(count):
        bx, by = _iso(x, y, k * 0.75 * m, 0)
        r, tall = 0.32 * m, 0.9 * m
        p.polygon(
            [
                (bx - r, by - tall),
                (bx + r, by - tall),
                (bx + r * 1.1, by - tall / 2),
                (bx + r, by),
                (bx - r, by),
                (bx - r * 1.1, by - tall / 2),
            ],
            BARREL,
        )
        p.polygon(
            [
                (bx + r / 3, by - tall),
                (bx + r, by - tall),
                (bx + r * 1.1, by - tall / 2),
                (bx + r, by),
                (bx + r / 3, by),
            ],
            shade(BARREL, 0.75),
        )
        for f in (0.22, 0.78):
            _bar(
                p,
                (bx - r * 1.05, by - tall * f),
                (bx + r * 1.05, by - tall * f),
                0.06 * m,
                (50, 44, 40),
            )
        p.blob(bx - r, by - tall - r / 2, bx + r, by - tall + r / 2, (60, 50, 44))


def _engine_house(p: Painter, x: float, y: float) -> None:
    """A boarded engine house 6 by 4 m with a door, a window and a sign of a black-gold drop of
    oil, and a tall iron stack smoking dark from its boiler."""
    m = METRE
    wall, high = (126, 104, 82), 3.4 * m
    sx, sy = _iso(x, y, 0.5 * m, 3.4 * m)
    _bar(p, (sx, sy), (sx, sy - 11 * m), 0.5 * m, (52, 50, 52))
    _bar(p, (sx - 0.12 * m, sy), (sx - 0.12 * m, sy - 11 * m), 0.12 * m, (84, 82, 84))
    for k, (dx, dy, rr) in enumerate(((0.6, -12.2, 1.1), (1.6, -13.8, 1.5))):
        p.blob(
            sx + (dx - rr) * m,
            sy + (dy - rr) * m,
            sx + (dx + rr) * m,
            sy + (dy + rr) * m,
            shade((120, 116, 118), 1 + 0.08 * k),
        )
    top = p.hall(x, y, 6 * m, 4 * m, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.7)
    for a, b, tone in ((left, front, 0.8), (front, right, 0.62)):
        for t in np.linspace(0.05, 0.95, 12):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, high), 0.04 * m, shade(wall, tone))
    _patch(p, left, front, (0.58, 0.82), (1 * m, high), (58, 48, 42))
    _patch(p, left, front, (0.12, 0.34), (0.9 * m, 1.9 * m), WINDOW)
    _patch(p, left, front, (0.39, 0.53), (0.45 * m, 1.75 * m), (236, 226, 196))
    dx, dy = _on_wall(left, front, 0.46, 1.6 * m)
    _oil_drop(p, dx, dy, 1 * m)
    p.gable(top, 1.4 * m, (62, 58, 56), wall)
    _tiles(p, top, 1.4 * m, (62, 58, 56), (0.5,))


def oil_well() -> Sprite:
    """A small early oil field on trodden earth soaked black in puddles: four timber derricks of
    different heights standing close together, each with its walking beam rocking on a samson
    post beside it, a boarded engine house with a tall iron stack smoking and a sign
    of a black-gold drop of oil, a short run of pipe on trestles to three round wooden storage
    tanks bound with hoops, and a row of oil barrels."""
    m = METRE
    p = Painter(38, 36, (21, 26))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 8) * m, (v - 4) * m)

    _ground(p, [(at(6, 6), 13), (at(-3, -2), 6), (at(17, 2), 7), (at(9, -7), 7)], OILY_EARTH)
    for (u, v), half in (((3, 3), 1.6), ((12, 7), 1.2), ((-1, 5), 1), ((9, -4), 1.4)):
        _oil_puddle(p, *at(u, v), half * m)
    _derrick(p, *at(8, 13), 25 * m)
    _derrick(p, *at(0, 9), 21 * m)
    _derrick(p, *at(15, 9), 18 * m)
    for k, ((u, v), r, tall) in enumerate(
        (((-6, 1), 2.2, 3.2), ((-6.5, -4.5), 2, 2.8), ((-1.5, -6), 1.8, 2.6))
    ):
        _wood_tank(p, *at(u, v), r * m, (tall + 0.1 * k) * m)
    for (u0, v0), (u1, v1) in ((((4, -0.5)), (-3.5, -0.5)), ((-3.5, -0.5), (-3.5, -4))):
        a, b = at(u0, v0), at(u1, v1)
        for t in (0.25, 0.75):
            tx, ty = _on_wall(a, b, t, 0)
            _bar(p, (tx, ty), (tx, ty - 0.6 * m), 0.18 * m, shade(DERRICK, 0.8))
        _bar(p, (a[0], a[1] - 0.7 * m), (b[0], b[1] - 0.7 * m), 0.24 * m, (52, 50, 54))
    _derrick(p, *at(6, 2), 16 * m)
    _engine_house(p, *at(17, 1))
    _oil_barrels(p, *at(5, -8), 6)
    return p.sprite()


def _corners(x: float, y: float, length: float, depth: float, z: float = 0) -> Top:
    """The left, front, right and back corners of an oblong `length` by `depth` centred on the
    ground at (x, y), raised `z` above it, as Painter.hall lays them."""
    front = (x + (length - depth) / 2, y + (length + depth) / 4 - z)
    left = (front[0] - length, front[1] - length / 2)
    right = (front[0] + depth, front[1] - depth / 2)
    return left, front, right, (left[0] + depth, left[1] - depth / 2)


_WOBBLE = (0.0, 0.08, -0.05, 0.1, 0.02, -0.08, 0.05, 0.12, -0.04, 0.06, -0.1, 0.03, 0.07, -0.06)


def _outline(
    at: Callable[[float, float], Point], u: float, v: float, ru: float, rv: float, phase: int
) -> list[Point]:
    """An uneven oval on the ground round (`u`, `v`) on the plan `at` lays out, `ru` and `rv`
    metres out along its two ways, its rim wobbling in and out, `phase` steps along the wobble."""
    n = len(_WOBBLE)
    points = []
    for k in range(n):
        a, w = 2 * math.pi * k / n, 1 + _WOBBLE[(k + phase) % n]
        points.append(at(u + ru * w * math.cos(a), v + rv * w * math.sin(a)))
    return points


def _shrunk(rim: list[Point], share: float) -> list[Point]:
    """The outline `rim` drawn in towards its middle by `share` of the way."""
    cx = sum(x for x, _ in rim) / len(rim)
    cy = sum(y for _, y in rim) / len(rim)
    return [(x + (cx - x) * share, y + (cy - y) * share) for x, y in rim]


def _dug(
    p: Painter,
    rims: list[list[Point]],
    depths: list[float],
    slopes: list[float],
    colours: tuple[RGB, RGB, RGB],
) -> None:
    """A hole dug in steps into flat ground. Each step is cut from its edge on the ground plan,
    `rims[k]`, `depths[k]` units down, its face sloping in `slopes[k]` of the way to its middle,
    to a bench of the next step; the last is the floor. Only the far faces show, lit by how they
    face the light and banded with layers, over benches lit from above; the ground before the
    hole, in `colours[2]`, hides what lies under its near edge. `colours` are the faces, the
    benches and the ground round the hole."""
    face, bench, ground = colours
    m = METRE
    total = sum(depths)
    p.soft(
        [(x - total * 1.4, y - total * 0.2, x + total * 1.4, y + total * 1.3) for x, y in rims[0]],
        ground,
        1,
        0.5,
    )
    drop = 0.0
    for rim, depth, slope in zip(rims, depths, slopes, strict=True):
        top = [(x, y + drop) for x, y in rim]
        low = [(x, y + drop + depth) for x, y in _shrunk(rim, slope)]
        cx = sum(x for x, _ in top) / len(top)
        cy = sum(y for _, y in top) / len(top)
        ring = list(zip(top, top[1:] + top[:1], low, low[1:] + low[:1], strict=True))
        for a, b, c, d in ring:
            nx, ny = -(b[1] - a[1]), b[0] - a[0]
            if nx * (cx - (a[0] + b[0]) / 2) + ny * (cy - (a[1] + b[1]) / 2) < 0:
                nx, ny = -nx, -ny
            if ny <= 0:
                continue
            tone = 0.94 - 0.24 * nx / math.hypot(nx, ny)
            p.polygon([a, b, d, c], shade(face, tone))
            for f in (0.35, 0.7):
                _bar(
                    p,
                    _on_wall(a, c, f, 0),
                    _on_wall(b, d, f, 0),
                    0.08 * m,
                    shade(face, tone * 0.86),
                )
            _bar(p, a, b, 0.12 * m, shade(face, tone * 1.12))
        p.polygon(low, bench)
        drop += depth
    rim = rims[0]
    cx = sum(x for x, _ in rim) / len(rim)
    cy = sum(y for _, y in rim) / len(rim)
    for a, b in zip(rim, rim[1:] + rim[:1], strict=True):
        nx, ny = -(b[1] - a[1]), b[0] - a[0]
        if nx * (cx - (a[0] + b[0]) / 2) + ny * (cy - (a[1] + b[1]) / 2) < 0:
            nx, ny = -nx, -ny
        if ny < 0:
            p.polygon([a, b, (b[0], b[1] + drop + 0.2), (a[0], a[1] + drop + 0.2)], ground)
            _bar(p, a, b, 0.14 * m, shade(ground, 1.12))


def _rails_m(p: Painter, a: Point, b: Point, across: Point) -> None:
    """A narrow-gauge track 0.6 m wide along the ground from `a` to `b`, `across` being how
    far a metre goes over the ground square to it: steel rails on timber sleepers."""
    m = METRE
    length = math.hypot(b[0] - a[0], 2 * (b[1] - a[1])) / m
    for k in range(max(2, round(length / 0.8))):
        t = (k + 0.5) / max(2, round(length / 0.8))
        x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        _bar(
            p,
            (x - across[0] * 0.55, y - across[1] * 0.55),
            (x + across[0] * 0.55, y + across[1] * 0.55),
            0.2 * m,
            SLEEPER,
        )
    for side in (-0.3, 0.3):
        dx, dy = across[0] * side, across[1] * side
        _bar(p, (a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy), 0.08 * m, RAIL)


def _skip(p: Painter, x: float, y: float, load: RGB, lumps: RGB | None = None) -> None:
    """A narrow-gauge tipping wagon 1.6 m long standing on its rails at (x, y), its iron
    V-shaped tub heaped with `load`, flecked with `lumps` if given."""
    m = METRE
    iron = (96, 92, 92)
    for t in (0.2, 0.8):
        wx, wy = _iso(x, y, (t - 0.5) * 1.4 * m, -0.3 * m)
        p.ellipse(wx - 0.22 * m, wy - 0.4 * m, wx + 0.22 * m, wy, (40, 38, 40))
    p.hall(x, y - 0.3 * m, 1.4 * m, 0.8 * m, 0.2 * m, shade(iron, 0.7), None)
    left, front, right, back = _corners(x, y, 1.6 * m, 1.1 * m, 1.1 * m)
    floor = _corners(x, y, 1.0 * m, 0.6 * m, 0.5 * m)
    p.polygon([floor[1], floor[2], right, front], shade(iron, 0.75))
    p.polygon([floor[0], floor[1], front, left], iron)
    p.polygon([left, front, right, back], shade(load, 0.92))
    p.blob(left[0] + 0.2 * m, back[1] - 0.25 * m, right[0] - 0.2 * m, front[1] - 0.05 * m, load)
    if lumps is not None:
        for dx, dy in ((-0.3, -0.25), (0.2, -0.35), (0.1, -0.1)):
            p.blob(
                x + (dx - 0.12) * m,
                y - 1.2 * m + (dy - 0.1) * m,
                x + (dx + 0.12) * m,
                y - 1.2 * m + (dy + 0.1) * m,
                lumps,
            )


def _wheelbarrow(p: Painter, x: float, y: float, load: RGB) -> None:
    """A wooden wheelbarrow 1.6 m long standing on (x, y), its wheel to the right, heaped with
    `load`."""
    m = METRE
    wood = (126, 90, 58)
    _bar(p, (x - 0.4 * m, y - 0.5 * m), (x - 1.3 * m, y - 0.1 * m), 0.1 * m, shade(wood, 0.8))
    _bar(p, (x - 0.3 * m, y - 0.4 * m), (x - 0.3 * m, y), 0.08 * m, shade(wood, 0.7))
    p.polygon(
        [
            (x - 0.6 * m, y - 0.85 * m),
            (x + 0.6 * m, y - 0.85 * m),
            (x + 0.4 * m, y - 0.35 * m),
            (x - 0.4 * m, y - 0.35 * m),
        ],
        wood,
    )
    p.blob(x - 0.55 * m, y - 1.1 * m, x + 0.55 * m, y - 0.7 * m, load)
    p.ellipse(x + 0.45 * m, y - 0.5 * m, x + 0.85 * m, y - 0.05 * m, (50, 44, 40))


def _hut(
    p: Painter, x: float, y: float, size: tuple[float, float, float], wall: RGB, roof: RGB
) -> Top:
    """A boarded hut `size` (length, depth and wall height in metres) standing on (x, y): a door
    and a window on its lit side, a window on its end, a felted roof and a stove pipe."""
    m = METRE
    length, depth, high = (s * m for s in size)
    top = p.hall(x, y, length, depth, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.6)
    for a, b, tone in ((left, front, 0.8), (front, right, 0.62)):
        for drop in (0.6 * m, 1.2 * m, 1.8 * m):
            _bar(p, _on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop), 0.05 * m, shade(wall, tone))
    _patch(p, left, front, (0.58, 0.82), (0.5 * m, high), DOOR)
    _patch(p, left, front, (0.16, 0.4), (0.6 * m, 1.4 * m), WINDOW)
    _patch(p, front, right, (0.35, 0.65), (0.6 * m, 1.4 * m), shade(WINDOW, 0.8))
    p.gable(top, 1 * m, roof, wall)
    sx, sy = _on_roof(top, 1 * m, 0.25, 0.6)
    _bar(p, (sx, sy), (sx, sy - 1.2 * m), 0.18 * m, (60, 58, 58))
    return top


def _on_roof(top: Top, rise: float, along: float, up: float) -> Point:
    """The point on the lit side of a gable roof `along` of the way from its back to its front and
    `up` of the way from its eave to its ridge."""
    left, front, right, back = top
    near = _on_wall(left, front, along, 0)
    ridge = _on_wall(
        ((left[0] + back[0]) / 2, (left[1] + back[1]) / 2 - rise),
        ((front[0] + right[0]) / 2, (front[1] + right[1]) / 2 - rise),
        along,
        0,
    )
    return near[0] + (ridge[0] - near[0]) * up, near[1] + (ridge[1] - near[1]) * up


CLAY: RGB = (192, 102, 60)
CLAY_BENCH: RGB = (208, 140, 88)
CLAY_GROUND: RGB = (150, 108, 74)
GREEN_BRICK: RGB = (182, 150, 116)
PIT_WATER: RGB = (124, 134, 134)


def _drying_shed(p: Painter, x: float, y: float) -> None:
    """An open drying shed 12 by 4 m for green bricks: a felted roof on timber posts over long
    hacks of unfired bricks stacked crosswise to let the air through."""
    m = METRE
    length, depth, eave = 12 * m, 4 * m, 2.6 * m
    left, front, right, back = _corners(x, y, length, depth)
    post = shade(TIMBER, 0.75)
    for t in (0, 0.33, 0.66, 1):
        bx, by = _on_wall(back, right, t, 0)
        _bar(p, (bx, by), (bx, by - eave), 0.22 * m, shade(post, 0.85))
    for v in (-0.9, 0.9):
        hx, hy = _iso(x, y, 0, v * m)
        hack = p.hall(hx, hy, 10.5 * m, 1 * m, 1.3 * m, GREEN_BRICK, shade(GREEN_BRICK, 1.1))
        h_left, h_front, _, _ = hack
        for drop in (0.33 * m, 0.66 * m, 1 * m):
            _bar(
                p,
                _on_wall(h_left, h_front, 0, drop),
                _on_wall(h_left, h_front, 1, drop),
                0.05 * m,
                shade(GREEN_BRICK, 0.72),
            )
        for k in range(1, 24):
            t = k / 24
            _bar(
                p,
                _on_wall(h_left, h_front, t, 0.1 * m),
                _on_wall(h_left, h_front, t, 1.2 * m),
                0.05 * m,
                shade(GREEN_BRICK, 0.8),
            )
    for t in (0, 0.33, 0.66, 1):
        fx, fy = _on_wall(left, front, t, 0)
        _bar(p, (fx, fy), (fx, fy - eave), 0.22 * m, post)
    top = _corners(x, y, length + 0.8 * m, depth + 0.8 * m, eave)
    p.gable(top, 1.4 * m, (84, 78, 74), TIMBER)
    _tiles(p, top, 1.4 * m, (84, 78, 74), (0.5,))


def _mound(p: Painter, x: float, y: float, half: float, tall: float, colour: RGB) -> None:
    """A lumpy heap of dug earth `half` across and `tall` high standing on (x, y), heaped from
    three rounded dumps each lit on its upper left, with a few clods rolled off its foot."""
    for dx, dy, size, lift in ((-0.45, -0.1, 0.6, 0.75), (0.4, 0.05, 0.55, 0.6), (0, 0.15, 0.7, 1)):
        cx, cy, h, t = x + dx * half, y + dy * half * 0.5, half * size, tall * lift
        p.blob(cx - h, cy - t, cx + h, cy + h * 0.3, shade(colour, 0.8))
        p.blob(cx - h, cy - t, cx + h * 0.3, cy + h * 0.15, colour)
        p.blob(cx - h * 0.6, cy - t * 0.95, cx - h * 0.05, cy - t * 0.5, shade(colour, 1.1))
    for dx, dy in ((-1.05, 0.15), (0.9, 0.3), (0.2, 0.42)):
        cx, cy = x + dx * half, y + dy * half * 0.5
        p.blob(
            cx - half * 0.08,
            cy - half * 0.06,
            cx + half * 0.08,
            cy + half * 0.05,
            shade(colour, 0.9),
        )


def _cone(p: Painter, x: float, y: float, r: float, tall: float, colour: RGB) -> None:
    """A cone of loose stuff `r` across its foot and `tall` high standing on (x, y), lit on its
    left and shaded on its right."""
    apex = (x + r * 0.05, y - tall)
    split = math.radians(65)
    lit = [(x + r * math.cos(a), y + r / 2 * math.sin(a)) for a in np.linspace(math.pi, split, 9)]
    dark = [(x + r * math.cos(a), y + r / 2 * math.sin(a)) for a in np.linspace(split, 0, 5)]
    p.polygon([*lit, apex], colour)
    p.polygon([*dark, apex], shade(colour, 0.8))
    _bar(
        p,
        (x - r * 0.45, y - tall * 0.3),
        (apex[0] - r * 0.05, apex[1] + tall * 0.12),
        r * 0.08,
        shade(colour, 1.08),
    )


def _ground(p: Painter, spots: list[tuple[Point, float]], colour: RGB) -> None:
    """Bare ground of `colour` spread over the grass round each of `spots`, an oval `half`
    across, solid in its middle and soft at its rim."""
    m = METRE
    p.soft(
        [
            (x - half * m, y - half * m / 2, x + half * m, y + half * m / 2)
            for (x, y), half in spots
        ],
        colour,
        1,
        0.5,
    )


def clay_pit() -> Sprite:
    """A clay digging on a brickfield: an uneven pit cut in three steps into terracotta clay,
    its faces banded with layers, rain water standing grey on its floor, a plank barrow run up
    its side and a loaded wheelbarrow on a bench, a narrow-gauge track out of it with two tipping
    wagons heaped with clay, rounded heaps of dug clay weathering, an open drying shed of
    unfired bricks and the digger's boarded hut."""
    m = METRE
    p = Painter(40, 31, (19, 18))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 9) * m, (v - 8) * m)

    _ground(
        p,
        [(at(10, 5.5), 13), (at(2, 12), 8), (at(2, 2), 7), (at(17, 1), 7), (at(-4, 8), 6)],
        CLAY_GROUND,
    )
    rim = _outline(at, 10, 5.5, 9.5, 8, 0)
    second = _outline(at, 10.3, 6, 6.8, 5.6, 3)
    floor = _outline(at, 10.6, 6.3, 3.8, 3.2, 6)
    _dug(
        p,
        [rim, second, floor],
        [1.7 * m, 1.6 * m, 0.4 * m],
        [0.06, 0.06, 0.15],
        (CLAY, CLAY_BENCH, CLAY_GROUND),
    )
    wx, wy = at(10.5, 6.5)
    p.blob(wx - 2.6 * m, wy + 3.4 * m, wx + 2.4 * m, wy + 4.8 * m, PIT_WATER)
    _bar(
        p, (wx - 1.4 * m, wy + 4 * m), (wx + 0.2 * m, wy + 4 * m), 0.12 * m, shade(PIT_WATER, 1.35)
    )
    low, high = at(15.5, 6), at(19.2, 6)
    p.polygon(
        [
            (low[0], low[1] + 3.3 * m),
            (low[0] + 0.6 * m, low[1] + 3 * m),
            (high[0] + 0.6 * m, high[1] - 0.3 * m),
            high,
        ],
        shade(TIMBER, 1.15),
        shade(TIMBER, 0.7),
    )
    bx, by = at(6.5, 3)
    _wheelbarrow(p, bx, by + 1.7 * m, CLAY)
    _drying_shed(p, *at(3, 18))
    _rails_m(p, at(-6.5, 7), at(2.5, 7), (m, m / 2))
    for u in (-4.5, -1.8):
        _skip(p, *at(u, 7), CLAY)
    for (u, v), half, tall in (((-1, 1), 3, 1.4), ((2, -3), 2.2, 1.0), ((21.5, 1), 2.4, 1.1)):
        _mound(p, *at(u, v), half * m, tall * m, shade(CLAY, 1.04))
    _hut(p, *at(21.5, 8.5), (3.5, 3, 2.4), (126, 96, 66), (70, 66, 64))
    return p.sprite()


SALT: RGB = (240, 238, 234)
PINK_SALT: RGB = (232, 190, 196)
GREY_SALT: RGB = (204, 202, 200)
SALT_YARD: RGB = (170, 160, 140)
SHINGLE: RGB = (86, 74, 66)
TILE_RED: RGB = (172, 82, 60)


def _crystal(p: Painter, x: float, y: float, size: float) -> None:
    """A cube of pink rock salt `size` units on an edge seen corner on, its top face standing on
    (x, y): three faces in three pinks, crisp pale edges and a glint of light."""
    s = size
    top = [(x - s, y - s / 2), (x, y - s), (x + s, y - s / 2), (x, y)]
    p.polygon([(x - s, y - s / 2), (x, y), (x, y + s), (x - s, y + s / 2)], shade(PINK_SALT, 0.96))
    p.polygon([(x, y), (x + s, y - s / 2), (x + s, y + s / 2), (x, y + s)], shade(PINK_SALT, 0.78))
    p.polygon(top, shade(PINK_SALT, 1.1))
    for a, b in (
        (top[0], top[1]),
        (top[1], top[2]),
        (top[0], top[3]),
        (top[3], top[2]),
        (top[3], (x, y + s)),
    ):
        _bar(p, a, b, s * 0.06, (252, 240, 242))
    _bar(p, (x - s * 0.55, y - s * 0.5), (x - s * 0.15, y - s * 0.5), s * 0.1, (255, 255, 255))
    _bar(p, (x - s * 0.35, y - s * 0.7), (x - s * 0.35, y - s * 0.3), s * 0.1, (255, 255, 255))


def _salt_heap(p: Painter, x: float, y: float, r: float, tall: float) -> None:
    """A heap of mined rock salt `r` across its foot and `tall` high on (x, y): off-white, with
    lumps of pink and grey salt in it and pale lumps rolled off round its foot."""
    m = METRE
    _cone(p, x, y, r, tall, SALT)
    lumps = (
        (-0.5, -0.25, PINK_SALT),
        (0.2, -0.5, GREY_SALT),
        (-0.2, -0.1, PINK_SALT),
        (0.45, -0.15, GREY_SALT),
        (-0.05, -0.7, PINK_SALT),
        (-0.75, 0.05, GREY_SALT),
        (1.05, 0.15, SALT),
        (-1.1, 0.2, PINK_SALT),
        (0.3, 0.45, SALT),
    )
    for k, (dx, dy, colour) in enumerate(lumps):
        cx, cy = x + dx * r, y + dy * tall
        w = (0.3 + 0.1 * (k % 3)) * m
        p.blob(cx - w, cy - w * 0.7, cx + w, cy + w * 0.5, colour)
        p.blob(cx - w * 0.6, cy - w * 0.6, cx, cy - w * 0.1, shade(colour, 1.06))


def _salt_block(p: Painter, x: float, y: float) -> None:
    """A block of rock salt hewn round like a barrel, 1.2 m across and 1.3 m high, standing on
    (x, y), grey-white with a pale top."""
    m = METRE
    r, tall = 0.6 * m, 1.3 * m
    p.polygon(
        [(x - r, y - tall), (x + r / 3, y - tall), (x + r / 3, y + r * 0.47), (x - r, y)], GREY_SALT
    )
    p.polygon(
        [(x + r / 3, y - tall), (x + r, y - tall), (x + r, y), (x + r / 3, y + r * 0.47)],
        shade(GREY_SALT, 0.8),
    )
    p.blob(x - r, y - tall - r / 2, x + r, y - tall + r / 2, shade(GREY_SALT, 1.12))
    _bar(p, (x - r, y - tall * 0.5), (x + r, y - tall * 0.5), 0.06 * m, shade(GREY_SALT, 0.86))


def _salt_tub(p: Painter, x: float, y: float) -> None:
    """A wooden mine tub 1.5 m long on its rails at (x, y), heaped with white rock salt flecked
    pink."""
    m = METRE
    for t in (0.2, 0.8):
        wx, wy = _iso(x, y, (t - 0.5) * 1.3 * m, -0.3 * m)
        p.ellipse(wx - 0.2 * m, wy - 0.38 * m, wx + 0.2 * m, wy, (40, 38, 40))
    left, front, right, back = p.hall(
        x, y - 0.3 * m, 1.5 * m, 0.9 * m, 0.7 * m, (128, 96, 66), None
    )
    _bar(
        p,
        _on_wall(left, front, 0, 0.35 * m),
        _on_wall(left, front, 1, 0.35 * m),
        0.07 * m,
        (80, 62, 46),
    )
    p.polygon([left, front, right, back], shade(SALT, 0.9))
    p.blob(left[0] + 0.15 * m, back[1] - 0.35 * m, right[0] - 0.15 * m, front[1] - 0.05 * m, SALT)
    for dx, dy in ((-0.3, -0.25), (0.25, -0.15)):
        cx, cy = x + dx * m, y - 1.2 * m + dy * m
        p.blob(cx - 0.12 * m, cy - 0.1 * m, cx + 0.12 * m, cy + 0.1 * m, PINK_SALT)


def _trestle(p: Painter, ground: Top, head: Top, storeys: tuple[float, ...], wood: RGB) -> None:
    """A braced timber frame: four legs raking in from the corners `ground` to the corners
    `head`, tied at each of `storeys` of the way up and braced crosswise between them, its back
    members darker and drawn first."""
    m = METRE

    def on(k: int, f: float) -> Point:
        return _on_wall(ground[k], head[k], f, 0)

    for a, b in ((0, 3), (3, 2)):
        for f0, f1 in itertools.pairwise(storeys):
            _bar(p, on(a, f0), on(b, f1), 0.14 * m, shade(wood, 0.62))
            _bar(p, on(b, f0), on(a, f1), 0.14 * m, shade(wood, 0.62))
    _bar(p, ground[3], head[3], 0.3 * m, shade(wood, 0.66))
    for k, tone in ((0, 0.95), (2, 0.78)):
        _bar(p, ground[k], head[k], 0.3 * m, shade(wood, tone))
    for a, b, tone in ((0, 1, 0.92), (1, 2, 0.76)):
        for f in storeys[1:]:
            _bar(p, on(a, f), on(b, f), 0.16 * m, shade(wood, tone))
        for f0, f1 in itertools.pairwise(storeys):
            _bar(p, on(a, f0), on(b, f1), 0.14 * m, shade(wood, tone))
            _bar(p, on(b, f0), on(a, f1), 0.14 * m, shade(wood, tone))
    _bar(p, ground[1], head[1], 0.3 * m, wood)


def _shaft_tower(p: Painter, x: float, y: float) -> Point:
    """A timber headframe over a shaft at (x, y): a boarded collar round the shaft, four raking
    legs 13 m high braced crosswise storey by storey, and a boarded sheave house on top under a
    steep shingled roof, its winding wheel turning in its open lit side. Returns the top of the
    wheel."""
    m = METRE
    tall, wood = 13 * m, (132, 98, 66)
    ground = _corners(x, y, 4.6 * m, 4.6 * m, 0.7 * m)
    head = _corners(x, y, 2.6 * m, 2.6 * m, tall)
    collar = p.hall(x, y, 5.4 * m, 5.4 * m, 0.7 * m, shade(wood, 0.9), shade(wood, 1.05))
    c_left, c_front, c_right, c_back = collar
    hole = [_on_wall(c_left, c_right, f, 0) for f in (0.3, 0.7)]
    p.polygon(
        [hole[0], _on_wall(c_back, c_front, 0.3, 0), hole[1], _on_wall(c_back, c_front, 0.7, 0)],
        (40, 34, 30),
    )
    _trestle(p, ground, head, (0.0, 0.33, 0.66, 1.0), wood)
    cabin = p.hall(x, y - tall, 3.2 * m, 3.2 * m, 2.6 * m, shade(wood, 1.08), None)
    left, front, right, _ = cabin
    for a, b, tone in ((left, front, 0.82), (front, right, 0.64)):
        for t in np.linspace(0.1, 0.9, 6):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, 2.6 * m), 0.05 * m, shade(wood, tone))
    _patch(p, left, front, (0.15, 0.85), (0.4 * m, 2.6 * m), (46, 38, 34))
    wx, wy = _on_wall(left, front, 0.5, 1.4 * m)
    r = 1.05 * m
    p.ellipse(wx - r, wy - r, wx + r, wy + r, (70, 66, 64))
    p.blob(wx - 0.82 * r, wy - 0.82 * r, wx + 0.82 * r, wy + 0.82 * r, (46, 38, 34))
    for k in range(4):
        a = k * math.pi / 4
        dx, dy = 0.85 * r * math.cos(a), 0.85 * r * math.sin(a)
        _bar(p, (wx - dx, wy - dy), (wx + dx, wy + dy), 0.08 * m, (96, 92, 90))
    p.blob(wx - 0.15 * r, wy - 0.15 * r, wx + 0.15 * r, wy + 0.15 * r, (150, 146, 140))
    _bar(p, (wx - r * 0.95, wy), (wx - r * 0.95, wy + 3 * m), 0.06 * m, (60, 58, 58))
    p.gable(cabin, 2.4 * m, SHINGLE, shade(wood, 1.08))
    _tiles(p, cabin, 2.4 * m, SHINGLE, (0.33, 0.66))
    return wx + r * 0.2, wy - r


def _winding_house(p: Painter, x: float, y: float) -> Top:
    """A whitewashed winding house 8 by 6 m on a stone plinth: arched windows, a door, a red
    tiled roof with a pink salt crystal set in its white gable, and a brick chimney smoking."""
    m = METRE
    high = 4.5 * m
    top = p.hall(x, y, 8 * m, 6 * m, high, WHITEWASH, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.8)
    _plinth(p, top, WHITEWASH, high, 0.6 * m)
    _patch(p, left, front, (0, 1), (0, 0.3 * m), shade(WHITEWASH, 0.82))
    _patch(p, front, right, (0, 1), (0, 0.3 * m), shade(WHITEWASH, 0.62))
    for t in (0.15, 0.38, 0.85):
        _patch(p, left, front, (t - 0.05, t + 0.05), (1 * m, 2.8 * m), WINDOW)
        p.polygon(
            [
                _on_wall(left, front, t - 0.05, 1 * m),
                _on_wall(left, front, t + 0.05, 1 * m),
                _on_wall(left, front, t, 0.6 * m),
            ],
            WINDOW,
        )
    _patch(p, left, front, (0.56, 0.7), (1.8 * m, high), DOOR)
    for t in (0.3, 0.7):
        _patch(p, front, right, (t - 0.07, t + 0.07), (1 * m, 2.8 * m), shade(WINDOW, 0.8))
    p.gable(top, 2.6 * m, TILE_RED, WHITEWASH)
    _tiles(p, top, 2.6 * m, TILE_RED, (0.25, 0.5, 0.75))
    gx, gy = _on_wall(front, right, 0.5, -1 * m)
    _crystal(p, gx, gy, 0.6 * m)
    sx, sy = _on_roof(top, 2.6 * m, 0.8, 0.7)
    stack = p.hall(sx, sy, 0.9 * m, 0.9 * m, 2.6 * m, BRICK, (50, 46, 46))
    p.blob(
        sx - 0.6 * m, stack[1][1] - 2.6 * m, sx + 1.2 * m, stack[1][1] - 0.8 * m, (226, 226, 230)
    )
    return top


def _boiling_house(p: Painter, x: float, y: float) -> None:
    """A salt boiling house 9 by 5 m of weathered boards under a long shingled roof, steam from
    its pans rising white out of a louvred vent along its ridge, and a brick chimney."""
    m = METRE
    wall, high = (150, 132, 110), 4 * m
    top = p.hall(x, y, 9 * m, 5 * m, high, wall, None)
    left, front, right, _ = top
    _footing(p, top, high, reach=0.8)
    for a, b, tone in ((left, front, 0.82), (front, right, 0.64)):
        for t in np.linspace(0.05, 0.95, 14):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, high), 0.05 * m, shade(wall, tone))
    _patch(p, left, front, (0.4, 0.58), (1.6 * m, high), (60, 50, 44))
    _patch(p, front, right, (0.3, 0.7), (1 * m, 2.2 * m), shade(WINDOW, 0.8))
    p.gable(top, 2.4 * m, SHINGLE, wall)
    _tiles(p, top, 2.4 * m, SHINGLE, (0.5,))
    rx, ry = _on_roof(top, 2.4 * m, 0.5, 1)
    vent = p.hall(rx, ry + 0.4 * m, 4 * m, 1 * m, 0.9 * m, shade(wall, 0.9), None)
    for k in (1, 2):
        _bar(
            p,
            _on_wall(vent[0], vent[1], 0, 0.3 * k * m),
            _on_wall(vent[0], vent[1], 1, 0.3 * k * m),
            0.08 * m,
            (70, 60, 54),
        )
    p.gable(vent, 0.5 * m, SHINGLE, shade(wall, 0.9))
    for k, (dx, dy, r) in enumerate(((-1.2, -2.2, 1.1), (0.2, -3.4, 1.4), (1.6, -4.8, 1.7))):
        p.blob(
            rx + (dx - r) * m,
            ry + (dy - r) * m,
            rx + (dx + r) * m,
            ry + (dy + r) * m,
            shade((240, 240, 244), 1 - 0.03 * k),
        )
    sx, sy = _on_roof(top, 2.4 * m, 0.15, 0.5)
    p.hall(sx, sy, 0.9 * m, 0.9 * m, 3.4 * m, BRICK, (50, 46, 46))


def salt_works() -> Sprite:
    """A salt mine: a timber headframe over the shaft with its winding wheel turning in a boarded
    sheave house under a shingled roof, the rope running to a whitewashed winding house with a red
    roof, a smoking chimney and a pink salt crystal set in its gable, mine tubs of white
    rock salt on a narrow track out of the shaft to a heap of white and pink rock salt, salt
    blocks hewn round like barrels, and a boarded boiling house with steam rising off its pans."""
    m = METRE
    p = Painter(38, 40, (20, 28))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10) * m, (v - 5) * m)

    _ground(p, [(at(8, 2), 10), (at(2, -3), 7), (at(17, 4), 8), (at(-3, 9), 7)], SALT_YARD)
    p.soft([(*_iso(*at(1, -3), -3 * m, 0), *_iso(*at(4, -1), 0, 0))], (214, 210, 200), 0.8, 0.8)
    _boiling_house(p, *at(-3.5, 9.5))
    wheel = _shaft_tower(p, *at(8, 6))
    house = _winding_house(p, *at(17.5, 5))
    rope = _on_wall(house[0], house[1], 0.05, 0.6 * m)
    _bar(p, wheel, rope, 0.06 * m, (84, 80, 78))
    _rails_m(p, at(6.5, 3), at(6.5, -8), (m, m / 2))
    for v in (-1.5, -5):
        _salt_tub(p, *at(6.5, v))
    _salt_heap(p, *at(1, -3), 3.6 * m, 3.4 * m)
    for u, v in ((11.5, -2.5), (12.8, -1.8), (12, -4)):
        _salt_block(p, *at(u, v))
    return p.sprite()


def limestone_quarry() -> Sprite:
    """A limestone quarry cut into a grassy hill: a tall pale face of cream rock streaked by
    rain and split into its beds, a ragged lip of turf and bushes along its top, a lower bench
    and screes of broken stone at its foot on a dusty floor, a round lime kiln smoking beside
    it, a derrick crane lowering a block, and cut blocks waiting by a wagon on a track out."""
    p = Painter(34, 36, (17, 25))
    stone = (226, 220, 200)
    dust = (206, 200, 180)
    grass = (110, 142, 74)
    p.blob(-16, -3, 16, 9, WORN)
    floor = [(-14, 3), (-9, -1.5), (6, -4), (14, 0), (10, 6), (-4, 8)]
    p.polygon(floor, dust, shade(dust, 0.8))
    _ruts(p, [(-9, 5.6), (-12.5, 7.6), (-16, 9.4)], dust)
    hill = [
        (-14, -1),
        (-12, -10),
        (-5, -16),
        (4, -17.5),
        (12, -14),
        (16, -6),
        (14.5, -0.5),
        (6, -4),
    ]
    p.polygon(hill, grass)
    p.polygon(
        [(-12, -10), (-5, -16), (4, -17.5), (12, -14), (6, -12.4), (-4, -11.6)], shade(grass, 1.1)
    )
    face = [
        (-11, -2.2),
        (-10, -8.5),
        (-7, -10),
        (-4, -11.6),
        (0, -12),
        (3, -11),
        (6, -12.4),
        (9, -11),
        (11, -8.2),
        (12, -4.6),
        (6, -4.2),
        (-9, -1.8),
    ]
    p.polygon(face, stone)
    p.polygon(
        [(-10.6, -4.2), (-3, -5.6), (5, -6), (11.8, -5.2), (12, -4.6), (6, -4.2), (-9, -1.8)],
        shade(stone, 0.86),
    )
    p.polygon(
        [
            (-10, -8.5),
            (-7, -10),
            (-4, -11.6),
            (0, -12),
            (3, -11),
            (6, -12.4),
            (9, -11),
            (11, -8.2),
            (10, -9.2),
            (5, -10.6),
            (-1, -10.4),
            (-7, -9),
        ],
        shade(stone, 1.06),
    )
    side = [(11, -8.2), (12, -4.6), (14.5, -0.6), (14.2, -5.2)]
    p.polygon(side, shade(stone, 0.66))
    for x in (-9, -6.5, -3.5, -1, 2, 4.5, 7.5, 10):
        top = -11.2 if -5 < x < 8 else -8.6
        p.line([(x, top + 0.6), (x + 0.4, -3.2 + abs(x) * 0.05)], shade(stone, 0.74))
    for y in (-9.5, -7, -4.8):
        p.line([(-10.2, y + 0.9), (-3, y - 0.2), (5, y - 0.6), (11.4, y + 0.6)], shade(stone, 0.7))
    p.polygon(
        [(-9, -1.8), (6, -4.2), (12, -4.6), (11, -3.4), (5, -2.8), (-8, -0.6)], (150, 144, 124)
    )
    for x, y in ((-6, -8.2), (1, -9.8), (7, -6.2), (-2, -5.4)):
        p.blob(x - 1.2, y - 0.3, x + 1.2, y + 0.5, shade(stone, 0.86))
    lip = [
        (-11, -2.2),
        (-10, -8.5),
        (-7, -10),
        (-4, -11.6),
        (0, -12),
        (3, -11),
        (6, -12.4),
        (9, -11),
        (11, -8.2),
    ]
    p.line([(x, y - 0.4) for x, y in lip], shade(grass, 0.75))
    for x, y in ((-8.5, -9.6), (-2.5, -12.8), (4, -12.4), (9.5, -12), (13, -7.5), (-11.5, -6)):
        p.blob(x - 1.4, y - 1.2, x + 1.4, y + 0.6, shade(grass, 0.7))
        p.blob(x - 0.9, y - 1.2, x + 0.5, y - 0.2, shade(grass, 0.95))
    for x, y, half, h in (
        (-7, -0.4, 3.4, 3),
        (7.5, -1.6, 3.6, 3.4),
        (1, -1.4, 2.6, 2),
        (-1.5, 1.6, 1.8, 1.2),
    ):
        _heap(p, x, y, half, h, shade(stone, 0.9))
    _strewn(p, 0, 4.4, 7, shade(stone, 0.8), 7)
    kx, ky = -12, 4.6
    _turned(p, kx, ky, [(0, 2.6), (3.6, 2.3), (4.6, 1.6)], (176, 162, 140), cap=(70, 64, 60))
    _patch(p, (kx - 2.4, ky - 0.4), (kx, ky + 0.9), (0.3, 0.75), (2.2, 3.6), (52, 40, 34))
    p.point(kx - 1.2, ky - 0.6, (240, 150, 60))
    for k, (dx, dy, r) in enumerate(((0.4, -6.4, 1.2), (1.4, -8, 1.6), (2.8, -9.8, 1.9))):
        grey = shade((206, 204, 200), 1 - 0.06 * k)
        p.blob(kx + dx - r, ky + dy - r, kx + dx + r, ky + dy + r, grey)
    for x, y, h in ((9.5, 3.5, 2), (11.5, 4.6, 2), (10.5, 2.4, 1.6)):
        block = p.box(x, y, 1.6, h, stone, shade(stone, 1.08))
        _footing(p, block, h, shade(dust, 0.62))
    p.blob(-8, 4.6, -1, 8, shade(dust, 0.72))
    p.hall(-4.5, 5.4, 4, 2, 0.9, TIMBER, shade(TIMBER, 1.15))
    p.box(-4.5, 4.6, 1.3, 1.4, stone, shade(stone, 1.08))
    for x, y in ((-6.6, 5.9), (-3, 7.6)):
        p.ellipse(x - 0.8, y - 0.8, x + 0.8, y + 0.8, (66, 60, 56))
    p.line([(-7.6, 5), (-9.6, 4)], TRUNK)
    p.line([(4, -2), (4, -15)], TRUNK)
    p.line([(4, -15), (8, -4)], (110, 104, 96))
    p.line([(4, -14), (-3, -10)], TRUNK)
    p.line([(4, -8), (-2, -10.2)], shade(TRUNK, 1.2))
    p.line([(-3, -10), (-3, -6)], (60, 56, 54))
    p.rect(-4, -6, -2, -4.6, stone)
    return p.sprite()


SAND: RGB = (232, 204, 138)
SAND_BENCH: RGB = (240, 218, 162)
SAND_GROUND: RGB = (206, 184, 128)
TUFT: RGB = (96, 132, 62)


def _sand_screen(p: Painter, x: float, y: float) -> None:
    """A sand screen standing on (x, y): a framed mesh 2 m wide leant back on two props, fine
    sand heaped before it and stones rolled off behind."""
    m = METRE
    frame = shade(TIMBER, 0.9)
    _heap(p, x - 0.2 * m, y + 0.6 * m, 1.6 * m, 1 * m, shade(SAND, 1.04))
    for dx in (-0.9, 0.9):
        _bar(
            p,
            (x + dx * m + 0.9 * m, y - 0.9 * m),
            (x + dx * m + 0.2 * m, y - 2.4 * m),
            0.12 * m,
            shade(frame, 0.8),
        )
    corners = [
        (x - 1 * m, y),
        (x + 1 * m, y),
        (x + 1.5 * m, y - 2.6 * m),
        (x - 0.5 * m, y - 2.6 * m),
    ]
    p.polygon(corners, (120, 116, 108), frame)
    for k in range(1, 6):
        t = k / 6
        _bar(
            p,
            _on_wall(corners[0], corners[3], t, 0),
            _on_wall(corners[1], corners[2], t, 0),
            0.04 * m,
            (170, 166, 158),
        )
        _bar(
            p,
            _on_wall(corners[0], corners[1], t, 0),
            _on_wall(corners[3], corners[2], t, 0),
            0.04 * m,
            (170, 166, 158),
        )
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        _bar(p, a, b, 0.14 * m, frame)
    for dx, dy in ((1.4, -0.4), (1.9, -0.1), (1.6, 0.2)):
        p.blob(
            x + (dx - 0.18) * m,
            y + (dy - 0.14) * m,
            x + (dx + 0.18) * m,
            y + (dy + 0.14) * m,
            (150, 144, 132),
        )


def _tip_cart(p: Painter, x: float, y: float, load: RGB) -> None:
    """A two-wheeled tipping cart 2.4 m long standing on (x, y) along the lit walls, its box
    heaped with `load`, a big spoked wheel on its near side and its shafts down on the ground."""
    m = METRE
    wood = (138, 96, 60)
    for v in (-0.35, 0.35):
        sx, sy = _iso(x, y, 1.1 * m, v * m)
        _bar(p, (sx, sy - 0.9 * m), _iso(sx, sy, 2.2 * m, 0), 0.12 * m, shade(wood, 0.75))
    box = p.hall(x, y - 0.6 * m, 2.4 * m, 1.4 * m, 0.7 * m, wood, shade(load, 0.9))
    left, front, right, back = box
    p.blob(left[0] + 0.4 * m, back[1] - 0.4 * m, right[0] - 0.4 * m, front[1] - 0.1 * m, load)
    _bar(
        p,
        _on_wall(left, front, 0, 0.35 * m),
        _on_wall(left, front, 1, 0.35 * m),
        0.06 * m,
        shade(wood, 0.7),
    )
    wx, wy = _on_wall(left, front, 0.5, 0.75 * m)
    p.ellipse(wx - 0.65 * m, wy - 0.65 * m, wx + 0.65 * m, wy + 0.65 * m, (58, 46, 38))
    p.blob(wx - 0.45 * m, wy - 0.45 * m, wx + 0.45 * m, wy + 0.45 * m, shade(wood, 0.95))
    for k in range(3):
        a = k * math.pi / 3
        dx, dy = 0.45 * m * math.cos(a), 0.45 * m * math.sin(a)
        _bar(p, (wx - dx, wy - dy), (wx + dx, wy + dy), 0.08 * m, (70, 52, 38))


def _hopper(p: Painter, x: float, y: float, load: RGB) -> None:
    """A boarded loading bin 2 m square on four legs 2.4 m high standing on (x, y), heaped with
    `load`, with a plank run up to it out of the pit behind and a chute sloping down from its
    lit side to fill a cart."""
    m = METRE
    wood = shade(TIMBER, 0.95)
    left, front, right, back = _corners(x, y, 2 * m, 2 * m)
    for corner, tone in ((back, 0.7), (right, 0.8), (left, 0.9)):
        _bar(p, corner, (corner[0], corner[1] - 2.4 * m), 0.22 * m, shade(wood, tone))
    run = _iso(x, y, 0, 4 * m)
    _bar(p, (run[0], run[1] + 1 * m), (back[0], back[1] - 3.4 * m), 0.6 * m, shade(wood, 1.15))
    box = p.hall(x, y - 2.4 * m, 2 * m, 2 * m, 1.3 * m, wood, shade(load, 0.92))
    b_left, b_front, b_right, b_back = box
    for a, b, tone in ((b_left, b_front, 0.8), (b_front, b_right, 0.62)):
        for drop in (0.43 * m, 0.86 * m):
            _bar(p, _on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop), 0.05 * m, shade(wood, tone))
    p.blob(b_left[0] + 0.2 * m, b_back[1] - 0.5 * m, b_right[0] - 0.2 * m, b_front[1], load)
    _bar(p, front, (front[0], front[1] - 2.4 * m), 0.22 * m, wood)
    mouth = _on_wall(b_left, b_front, 0.5, 1.1 * m)
    out = (mouth[0] - 1.6 * m, mouth[1] + 1.8 * m)
    p.polygon(
        [
            (mouth[0] - 0.5 * m, mouth[1] - 0.25 * m),
            (mouth[0] + 0.5 * m, mouth[1] + 0.25 * m),
            (out[0] + 0.5 * m, out[1] + 0.25 * m),
            (out[0] - 0.5 * m, out[1] - 0.25 * m),
        ],
        shade(METAL, 0.85),
        shade(METAL, 0.55),
    )
    p.blob(out[0] - 0.4 * m, out[1] - 0.1 * m, out[0] + 0.2 * m, out[1] + 0.6 * m, load)


def sand_pit() -> Sprite:
    """A sand quarry: an uneven pit dug in two benches into pale gold sand with sloping faces
    and a puddle on its floor, tufts of grass along its edges, a belt conveyor climbing out of it
    to a chute over a tipping cart, a sand screen leant on its props with sifted sand before it,
    cones of sand heaped up, and a small boarded office."""
    m = METRE
    p = Painter(42, 30, (21, 18))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10) * m, (v - 7) * m)

    _ground(
        p,
        [(at(11, 6), 13), (at(23, 1), 7), (at(2, -2), 7), (at(4, 14), 6), (at(20, 12), 6)],
        SAND_GROUND,
    )
    rim = _outline(at, 10.5, 6.5, 10.5, 8.5, 2)
    second = _outline(at, 10.8, 6.8, 7.2, 5.6, 5)
    floor = _outline(at, 11, 7, 3.6, 2.8, 9)
    _dug(
        p,
        [rim, second, floor],
        [1.6 * m, 1.6 * m, 0.3 * m],
        [0.16, 0.18, 0.2],
        (SAND, SAND_BENCH, SAND_GROUND),
    )
    wx, wy = at(10.5, 7.5)
    p.blob(wx - 1.8 * m, wy + 2.6 * m, wx + 2.2 * m, wy + 3.8 * m, (118, 160, 186))
    _bar(p, (wx - 0.8 * m, wy + 3.1 * m), (wx + 0.6 * m, wy + 3.1 * m), 0.1 * m, (176, 210, 228))
    for u, v in (
        (-0.5, 8),
        (1.5, 2),
        (6, -1.5),
        (14, -2),
        (19.5, 1.5),
        (21.5, 8.5),
        (17, 14),
        (6, 14.5),
    ):
        x, y = at(u, v)
        for dx, dy, r in ((-0.6, 0, 0.5), (0.3, -0.2, 0.6), (0.9, 0.2, 0.4)):
            p.blob(
                x + (dx - r) * m,
                y + (dy - r) * m,
                x + (dx + r) * m,
                y + (dy + r * 0.3) * m,
                shade(TUFT, 0.9 + 0.2 * r),
            )
    _hut(p, *at(6, 18), (4, 3, 2.4), (150, 118, 82), (84, 80, 78))
    _hopper(p, *at(23, 4), SAND)
    _tip_cart(p, *at(21.8, 0.6), SAND)
    for (u, v), r, tall in (((25.5, -3.5), 3, 3.2), ((20.5, -5), 2.2, 2.2), ((0, -3.5), 2.6, 2.6)):
        _cone(p, *at(u, v), r * m, tall * m, SAND)
    _sand_screen(p, *at(4.5, -6))
    return p.sprite()
