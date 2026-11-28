"""Mines, wells, pits, quarries, logging camps and sawmills."""

import itertools
import math
from collections.abc import Callable

import numpy as np

from railroads.ui.pixel.art.collieries import (
    _beam,
    _block,
    _face,
    _fir,
    _land,
    _p,
    _rails,
    _speckle,
    _swell,
    _yard,
)
from railroads.ui.pixel.art.commerce import _planks
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
from railroads.ui.pixel.art.industry import _food_cart, _food_horse
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


CLAY_WET: RGB = (164, 84, 52)
CLAY_OCHRE: RGB = (196, 138, 74)
CLAY_SKIN: RGB = (222, 178, 144)
CLAY_SHIRTS: tuple[RGB, ...] = ((236, 232, 222), (96, 122, 170), (176, 72, 58), (214, 204, 170))


def _clay_hand(p: Painter, x: float, y: float, shirt: RGB, pose: str, facing: int = 1) -> None:
    """A brickfield hand standing on (x, y), drawn a fifth over his true size so he reads among
    the works: cord trousers, a shirt of `shirt` with its sleeves rolled, a flat cap; bent over
    a spade cutting clay if `pose` is "dig", leaning on the handles of a barrow ahead of him if
    "push", or upright working at a table if "stand", facing right or with `facing` -1 left."""
    s = 1.2 * METRE
    f = facing
    trousers, cap = (78, 66, 58), (54, 50, 48)
    lean = {"dig": 0.38, "push": 0.42, "stand": 0.06}[pose]
    hip = (x, y - 0.9 * s)
    shoulder = (x + f * lean * s, y - (1.4 if pose != "stand" else 1.5) * s)
    for d in (-0.12, 0.12):
        foot = (x + d * s + (f * 0.12 * s if pose == "push" and d * f < 0 else 0), y)
        _bar(p, (hip[0] + d * 0.5 * s, hip[1]), foot, 0.17 * s, trousers)
        p.blob(
            foot[0] - 0.1 * s,
            foot[1] - 0.06 * s,
            foot[0] + 0.1 * s + f * 0.06 * s,
            foot[1] + 0.03 * s,
            (40, 34, 30),
        )
    p.polygon(
        [
            (hip[0] - 0.17 * s, hip[1]),
            (hip[0] + 0.17 * s, hip[1]),
            (shoulder[0] + 0.2 * s, shoulder[1]),
            (shoulder[0] - 0.2 * s, shoulder[1]),
        ],
        shirt,
    )
    p.polygon(
        [
            (hip[0] + 0.02 * s, hip[1]),
            (hip[0] + 0.17 * s, hip[1]),
            (shoulder[0] + 0.2 * s, shoulder[1]),
            (shoulder[0] + 0.04 * s, shoulder[1]),
        ],
        shade(shirt, 0.84),
    )
    _bar(
        p,
        (hip[0] - 0.17 * s, hip[1] + 0.03 * s),
        (hip[0] + 0.17 * s, hip[1] + 0.03 * s),
        0.06 * s,
        (60, 46, 38),
    )
    hand = {
        "dig": (x + f * 0.62 * s, y - 0.95 * s),
        "push": (x + f * 0.62 * s, y - 0.72 * s),
        "stand": (x + f * 0.38 * s, y - 0.98 * s),
    }[pose]
    for d in (-0.12, 0.12):
        elbow = ((shoulder[0] + hand[0]) / 2 + d * 0.3 * s, (shoulder[1] + hand[1]) / 2 + 0.05 * s)
        _bar(p, (shoulder[0] + d * s, shoulder[1] + 0.04 * s), elbow, 0.11 * s, shirt)
        _bar(p, elbow, hand, 0.09 * s, CLAY_SKIN)
    if pose == "dig":
        blade = (x + f * 0.9 * s, y - 0.02 * s)
        _bar(p, (x + f * 0.45 * s, y - 1.25 * s), blade, 0.06 * s, TIMBER)
        p.polygon(
            [
                (blade[0] - 0.1 * s, blade[1] - 0.22 * s),
                (blade[0] + 0.1 * s, blade[1] - 0.22 * s),
                (blade[0] + 0.08 * s, blade[1] + 0.04 * s),
                (blade[0] - 0.08 * s, blade[1] + 0.04 * s),
            ],
            METAL,
        )
        p.blob(
            blade[0] - 0.14 * s,
            blade[1] - 0.3 * s,
            blade[0] + 0.12 * s,
            blade[1] - 0.12 * s,
            CLAY_WET,
        )
    hx, hy = shoulder[0] + f * 0.07 * s, shoulder[1] - 0.19 * s
    p.blob(hx - 0.12 * s, hy - 0.12 * s, hx + 0.12 * s, hy + 0.13 * s, CLAY_SKIN)
    p.blob(hx - 0.14 * s, hy - 0.17 * s, hx + 0.13 * s, hy - 0.03 * s, cap)
    _bar(p, (hx, hy - 0.06 * s), (hx + f * 0.2 * s, hy - 0.04 * s), 0.06 * s, cap)


def _clay_pug_mill(p: Painter, x: float, y: float) -> None:
    """A horse-driven pug mill on (x, y): a tub of iron-hooped staves 1.8 m across with its
    upright shaft, worked clay squeezing out at its foot, and a long sweep from the shaft's
    head over the back of a draught horse plodding round a ring trodden into the ground."""
    m = METRE
    r = 3.2 * m
    p.soft(
        [(x - r - 0.8 * m, y - r / 2 - 0.4 * m, x + r + 0.8 * m, y + r / 2 + 0.4 * m)],
        shade(CLAY_GROUND, 0.82),
        0.9,
        0.5,
    )
    p.blob(x - r + 0.8 * m, y - r / 2 + 0.4 * m, x + r - 0.8 * m, y + r / 2 - 0.4 * m, CLAY_GROUND)
    horse = _iso(x, y, 0, -r)
    p.blob(x - 1.4 * m, y + 0.4 * m, x - 0.2 * m, y + 1 * m, CLAY_WET)
    tub, tall = 1.0 * m, 2 * m
    p.polygon(
        [(x - tub, y - tall), (x + tub / 3, y - tall), (x + tub / 3, y + tub * 0.47), (x - tub, y)],
        TIMBER,
    )
    p.polygon(
        [(x + tub / 3, y - tall), (x + tub, y - tall), (x + tub, y), (x + tub / 3, y + tub * 0.47)],
        shade(TIMBER, 0.72),
    )
    for f in (0.2, 0.55, 0.85):
        _bar(p, (x - tub, y - tall * f), (x + tub, y - tall * f), 0.08 * m, (52, 48, 48))
    p.blob(x - tub, y - tall - tub / 2, x + tub, y - tall + tub / 2, CLAY_WET)
    head = (x, y - tall - 0.9 * m)
    _bar(p, (x, y - tall), head, 0.16 * m, shade(TIMBER, 0.7))
    _food_horse(p, *horse, 1)
    _bar(p, head, (horse[0] - 0.2 * m, horse[1] - 1.5 * m), 0.26 * m, shade(TIMBER, 1.1))
    _bar(
        p,
        (horse[0] - 0.2 * m, horse[1] - 1.5 * m),
        (horse[0] + 0.4 * m, horse[1] - 1.2 * m),
        0.08 * m,
        (40, 34, 32),
    )


def _clay_lean_to(p: Painter, x: float, y: float, moulder: RGB) -> None:
    """A moulding lean-to on (x, y): a boarded roof 5 by 3 m sloping back on four posts over a
    bench where a moulder strikes bricks in wooden moulds from a lump of clay, a row of new
    bricks set out on a board beside him and a pallet of them on the ground."""
    m = METRE
    ground = _corners(x, y, 5 * m, 3 * m)
    high = _corners(x, y, 5 * m, 3 * m, 3.3 * m)
    low = _corners(x, y, 5 * m, 3 * m, 2.5 * m)
    for k, corner in ((3, high[3]), (2, high[2])):
        _bar(
            p,
            ground[k],
            (ground[k][0], low[k][1] if k in (2, 3) else corner[1]),
            0.18 * m,
            shade(TIMBER, 0.7),
        )
    _clay_hand(p, *_iso(x, y, -0.6 * m, 0.4 * m), moulder, "stand", 1)
    table = p.hall(
        *_iso(x, y, 0, -0.3 * m),
        2.6 * m,
        0.9 * m,
        0.9 * m,
        shade(TIMBER, 0.95),
        shade(TIMBER, 1.15),
    )
    t_left, t_front, t_right, t_back = table
    p.blob(
        t_left[0] + 0.3 * m,
        t_back[1] - 0.35 * m,
        t_left[0] + 1.2 * m,
        t_front[1] - 0.6 * m,
        CLAY_WET,
    )
    for k in range(3):
        mx, my = _on_wall(
            _on_wall(t_left, t_back, 0.5, 0), _on_wall(t_front, t_right, 0.5, 0), 0.45 + 0.17 * k, 0
        )
        p.hall(
            mx,
            my,
            0.4 * m,
            0.25 * m,
            0.12 * m,
            GREEN_BRICK if k else shade(TIMBER, 0.75),
            shade(GREEN_BRICK, 1.1) if k else CLAY_WET,
        )
    for k in range(4):
        bx, by = _iso(*_iso(x, y, 1.9 * m, -1.9 * m), k * 0.4 * m, 0)
        p.hall(bx, by, 0.35 * m, 0.6 * m, 0.25 * m, GREEN_BRICK, shade(GREEN_BRICK, 1.1))
    for k in (0, 1):
        _bar(p, ground[k], (ground[k][0], high[k][1]), 0.18 * m, shade(TIMBER, 0.85))
    roof = [high[0], high[1], low[2], low[3]]
    p.polygon(roof, (110, 96, 82), (80, 70, 60))
    for t in (0.25, 0.5, 0.75):
        _bar(
            p,
            _on_wall(high[0], high[1], t, 0),
            _on_wall(low[3], low[2], t, 0),
            0.05 * m,
            (90, 78, 66),
        )


def _clay_hacks(p: Painter, x: float, y: float, rows: int, long: float) -> None:
    """Long hacks of green bricks drying in the open on (x, y): `rows` rows `long` metres long
    running along the lit walls, the bricks set crosswise with gaps for the air, each row capped
    with boards on edge to keep off the rain."""
    m = METRE
    for k in reversed(range(rows)):
        hx, hy = _iso(x, y, 0, k * 1.6 * m)
        hack = p.hall(hx, hy, long * m, 0.7 * m, 1.1 * m, GREEN_BRICK, shade(GREEN_BRICK, 1.08))
        h_left, h_front, h_right, h_back = hack
        for drop in (0.28 * m, 0.55 * m, 0.82 * m):
            _bar(
                p,
                _on_wall(h_left, h_front, 0, drop),
                _on_wall(h_left, h_front, 1, drop),
                0.04 * m,
                shade(GREEN_BRICK, 0.72),
            )
        for j in range(1, round(long * 2.4)):
            t = j / round(long * 2.4)
            _bar(
                p,
                _on_wall(h_left, h_front, t, 0.08 * m),
                _on_wall(h_left, h_front, t, 1.02 * m),
                0.04 * m,
                shade(GREEN_BRICK, 0.62),
            )
        ridge = [_on_wall(h_left, h_back, 0.5, -0.4 * m), _on_wall(h_front, h_right, 0.5, -0.4 * m)]
        board = (128, 112, 96)
        p.polygon([h_left, h_front, ridge[1], ridge[0]], board, shade(board, 0.6))
        p.polygon([h_front, h_right, ridge[1]], shade(board, 0.72))
        _bar(p, ridge[0], ridge[1], 0.06 * m, shade(board, 1.25))


def _clay_tipped(p: Painter, x: float, y: float) -> None:
    """A narrow-gauge tipping wagon at the end of its rails at (x, y), its V-shaped tub rolled
    over on its cradle to the lit side and clay sliding out of it onto a heap."""
    m = METRE
    iron = (96, 92, 92)
    _mound(p, *_iso(x, y, 0, -1.4 * m), 1.4 * m, 0.8 * m, CLAY)
    for t in (0.2, 0.8):
        wx, wy = _iso(x, y, (t - 0.5) * 1.4 * m, -0.3 * m)
        p.ellipse(wx - 0.22 * m, wy - 0.4 * m, wx + 0.22 * m, wy, (40, 38, 40))
    p.hall(x, y - 0.3 * m, 1.4 * m, 0.8 * m, 0.2 * m, shade(iron, 0.7), None)
    lip = [_iso(x, y - 0.8 * m, u * m, -0.9 * m) for u in (-0.8, 0.8)]
    back = [_iso(x, y - 1.4 * m, u * m, 0.2 * m) for u in (-0.8, 0.8)]
    keel = [_iso(x, y - 0.5 * m, u * m, 0.3 * m) for u in (-0.6, 0.6)]
    p.polygon([back[0], back[1], keel[1], keel[0]], shade(iron, 0.75))
    p.polygon([keel[0], keel[1], lip[1], lip[0]], iron)
    p.polygon([lip[0], lip[1], back[1], back[0]], shade(CLAY_WET, 0.8))
    p.polygon([lip[0], lip[1], _iso(*lip[1], 0, -0.7 * m), _iso(*lip[0], 0, -0.7 * m)], CLAY_WET)


def _clay_butt(p: Painter, x: float, y: float) -> None:
    """A water butt on (x, y): a hooped barrel a metre high brimming with rain water."""
    m = METRE
    r, tall = 0.42 * m, 1 * m
    p.polygon(
        [(x - r, y - tall), (x + r / 3, y - tall), (x + r / 3, y + r * 0.47), (x - r, y)], BARREL
    )
    p.polygon(
        [(x + r / 3, y - tall), (x + r, y - tall), (x + r, y), (x + r / 3, y + r * 0.47)],
        shade(BARREL, 0.72),
    )
    for f in (0.2, 0.8):
        _bar(p, (x - r, y - tall * f), (x + r, y - tall * f), 0.06 * m, (52, 48, 48))
    p.blob(x - r, y - tall - r / 2, x + r, y - tall + r / 2, shade(BARREL, 0.6))
    p.blob(x - r * 0.8, y - tall - r * 0.38, x + r * 0.8, y - tall + r * 0.38, PIT_WATER)


def _clay_tools(p: Painter, x: float, y: float) -> None:
    """Two spades and a pick leant against a wall whose foot is at (x, y)."""
    m = METRE
    for k, (dx, kind) in enumerate(((0, "spade"), (0.4, "pick"), (0.8, "spade"))):
        foot = (x + dx * m, y + 0.1 * m * k)
        top = (foot[0] + 0.35 * m, foot[1] - 1.5 * m)
        _bar(p, foot, top, 0.07 * m, TIMBER)
        if kind == "spade":
            p.polygon(
                [
                    (foot[0] - 0.12 * m, foot[1]),
                    (foot[0] + 0.1 * m, foot[1]),
                    (foot[0] + 0.12 * m, foot[1] - 0.3 * m),
                    (foot[0] - 0.08 * m, foot[1] - 0.3 * m),
                ],
                METAL,
            )
        else:
            _bar(
                p,
                (top[0] - 0.35 * m, top[1] + 0.12 * m),
                (top[0] + 0.3 * m, top[1] - 0.06 * m),
                0.08 * m,
                shade(METAL, 0.8),
            )


def _clay_load(at: Callable[[float, float], Point], p: Painter) -> None:
    """Wet clay heaped on a cart's bed, laid out by `at`."""
    m = METRE
    for s, w, r in ((-0.6, 0, 0.75), (0.4, 0.1, 0.7), (0, -0.2, 0.6)):
        cx, cy = at(s, w)
        p.blob(cx - r * m, cy - 0.65 * m, cx + r * m, cy + 0.1 * m, CLAY_WET)
        p.blob(cx - r * 0.6 * m, cy - 0.6 * m, cx, cy - 0.3 * m, shade(CLAY_WET, 1.12))


def _clay_ramp(p: Painter, a: Point, b: Point) -> None:
    """A cart road 2.6 m wide worn into the clay from `a` at a pit's edge out to `b`, darker
    where the wheels have cut two ruts along it."""
    m = METRE
    w = (1.3 * m, -0.65 * m)
    p.polygon(
        [
            (a[0] - w[0], a[1] - w[1]),
            (a[0] + w[0], a[1] + w[1]),
            (b[0] + w[0], b[1] + w[1]),
            (b[0] - w[0], b[1] - w[1]),
        ],
        shade(CLAY_GROUND, 0.88),
    )
    for side in (-0.5, 0.5):
        _bar(
            p,
            (a[0] + w[0] * side, a[1] + w[1] * side),
            (b[0] + w[0] * side, b[1] + w[1] * side),
            0.18 * m,
            shade(CLAY_GROUND, 0.7),
        )


def clay_pit() -> Sprite:
    """A working clay pit on a brickfield. An uneven pit cut in four benches into terracotta
    clay, its wet faces banded with layers, puddles standing on its benches and floor, spoil
    strewn about; diggers in shirtsleeves cutting clay with spades on the benches and a hand
    wheeling a barrow up a plank run; a horse cart heaped with clay coming up out of it; a
    narrow-gauge line of tubs from its edge to a tub tipped over by the pug mill, where a horse
    plods round on the sweep; a moulding lean-to where a moulder strikes bricks in wooden moulds
    at his bench; long hacks of cream green bricks drying under board caps beside the drying
    shed; and the digger's hut with tools leant on it, stacked planks and a water butt."""
    m = METRE
    p = Painter(44, 36, (22, 21))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 2) * m, (v - 8) * m)

    def low(u: float, v: float, depth: float) -> Point:
        x, y = at(u, v)
        return x, y + depth * m

    _ground(
        p,
        [
            (at(10, 6), 13),
            (at(-7, 4), 7),
            (at(-4, 15), 9),
            (at(18, 0), 8),
            (at(21.5, 6), 5),
            (at(6, 21), 8),
        ],
        CLAY_GROUND,
    )
    rim = _outline(at, 10, 6, 10, 8, 4)
    first = _outline(at, 10.6, 6.6, 7.6, 6.2, 9)
    second = _outline(at, 10.2, 7.2, 5.4, 4.3, 1)
    floor = _outline(at, 10.8, 7, 3.2, 2.4, 11)
    _dug(
        p,
        [rim, first, second, floor],
        [1.4 * m, 1.2 * m, 1.3 * m, 0.4 * m],
        [0.05, 0.05, 0.08, 0.15],
        (CLAY_WET, CLAY_OCHRE, CLAY_GROUND),
    )
    for (u, v, depth), r in (
        ((11, 7.5, 4.3), 2.2),
        ((14.5, 12, 1.4), 1.1),
        ((6.5, 10.5, 2.6), 0.8),
    ):
        x, y = low(u, v, depth)
        p.blob(x - r * m, y - r * m / 2, x + r * m, y + r * m / 2, PIT_WATER)
        _bar(p, (x - r * 0.5 * m, y), (x + r * 0.1 * m, y), 0.1 * m, shade(PIT_WATER, 1.35))
    for u, v, depth in ((8, 13, 1.4), (12.5, 12.8, 1.4), (9.5, 10.6, 2.6), (13, 9.6, 2.6)):
        x, y = low(u, v, depth)
        p.blob(x - 0.4 * m, y - 0.25 * m, x + 0.4 * m, y + 0.1 * m, shade(CLAY_WET, 1.1))
    _clay_hand(p, *low(9, 12.6, 1.4), CLAY_SHIRTS[0], "dig", 1)
    _clay_hand(p, *low(14, 11.8, 1.4), CLAY_SHIRTS[1], "dig", -1)
    _clay_hand(p, *low(11.5, 9.8, 2.6), CLAY_SHIRTS[2], "dig", 1)
    foot, head = low(5.5, 9, 3.9), at(1.5, 11)
    _bar(p, foot, head, 0.5 * m, shade(TIMBER, 1.2))
    _bar(
        p, (foot[0], foot[1] + 0.12 * m), (head[0], head[1] + 0.12 * m), 0.1 * m, shade(TIMBER, 0.7)
    )
    mx, my = (foot[0] * 0.45 + head[0] * 0.55), (foot[1] * 0.45 + head[1] * 0.55)
    _wheelbarrow(p, mx - 0.6 * m, my - 0.3 * m, CLAY_WET)
    _clay_hand(p, mx + 0.7 * m, my + 0.4 * m, CLAY_SHIRTS[3], "push", -1)
    for u, v, half, tall in ((-1, 0, 2.6, 1.3), (20, -1, 2.2, 1.1), (21.5, 12, 1.8, 0.9)):
        _mound(p, *at(u, v), half * m, tall * m, shade(CLAY, 1.04))
    for u, v, r in ((3, -1.5, 0.35), (4.5, -2.4, 0.3), (2, -2.8, 0.25), (5.5, -1.2, 0.3)):
        x, y = at(u, v)
        p.blob(x - r * m, y - r * 0.6 * m, x + r * m, y + r * 0.4 * m, CLAY_WET)
    _clay_ramp(p, at(11.5, -0.5), at(17.5, -4.5))
    _drying_shed(p, *at(4, 22))
    _clay_hacks(p, *at(-6, 16.5), 3, 9)
    _clay_lean_to(p, *at(-6, 10), CLAY_SHIRTS[1])
    _rails_m(p, at(-9.5, 3.5), at(1.5, 3.5), (m, -m / 2))
    for u in (-5.6, -3.2, -0.8):
        _skip(p, *at(u, 3.5), CLAY_WET)
    _clay_pug_mill(p, *at(-12, 4.5))
    _clay_tipped(p, *at(-8.6, 3.5))
    hx, hy = at(21, 8)
    _hut(p, hx, hy, (3.5, 3, 2.4), (126, 96, 66), (70, 66, 64))
    _clay_tools(p, *_iso(hx, hy, -1.2 * m, -1.75 * m))
    _clay_butt(p, *at(22.8, 5.6))
    _planks(p, *at(13, 19.5), 4, 0.8)
    _food_cart(p, *at(19, -4.5), 3.2, 0.5, CLAY_GROUND, lambda put: _clay_load(put, p))
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


def _quarry_lumps(p: Painter, spots: list[tuple[float, float, float, float]], rock: RGB) -> None:
    """Broken stones lying at each (u, v, z, size in metres) of `spots`: angular lumps lit on
    their upper left and shaded below, in tones of `rock`."""
    m = METRE
    for k, (u, v, z, size) in enumerate(spots):
        x, y = _p(u, v, z)
        r = size * m
        tone = (0.92, 1.0, 0.86, 1.04, 0.96)[k % 5]
        p.polygon(
            [
                (x - r, y),
                (x - r * 0.7, y - r * 0.75),
                (x + r * 0.2, y - r * 0.95),
                (x + r, y - r * 0.35),
                (x + r * 0.7, y + r * 0.25),
                (x - r * 0.3, y + r * 0.3),
            ],
            shade(rock, tone * 0.78),
        )
        p.polygon(
            [
                (x - r, y),
                (x - r * 0.7, y - r * 0.75),
                (x + r * 0.2, y - r * 0.95),
                (x + r * 0.5, y - r * 0.4),
                (x - r * 0.2, y - r * 0.1),
            ],
            shade(rock, tone * 1.04),
        )


def _quarry_face(
    p: Painter,
    height: Callable[[float, float], float],
    start: tuple[float, float],
    along: tuple[float, float],
    long: float,
    rock: RGB,
    ledges: tuple[float, ...],
    drills: tuple[float, ...] = (),
) -> None:
    """A sawn face of limestone standing upright where the land shaped by `height` drops
    sheer, running `long` metres from (u, v) `start` the way `along` points, (1, 0) or (0, 1),
    with the dug ground before it on its right: column by column in `rock` a shade lighter or
    darker, crossed by the beds of the stone, streaked grey down from its top by the rain,
    scored by rows of drill holes at each of `drills` metres along it, damp at its foot with
    broken stone fallen before it, and lit sharp along its top edge where it stands at one of
    `ledges`, or capped by a ragged band of soil and grass where the hillside comes down to it."""
    m = METRE
    au, av = along
    fu, fv = av, -au
    su, sv = start[0] + fu * 0.125, start[1] + fv * 0.125
    cols = []
    for k in range(round(long / 0.25) + 1):
        u, v = su + au * k * 0.25, sv + av * k * 0.25
        cols.append((u, v, height(u - fu * 0.3, v - fv * 0.3), height(u + fu * 0.3, v + fv * 0.3)))
    beds = (0.0, 0.8, 1.5, 2.4, 3.6, 4.3, 5.2, 6.1, 7.0, 7.6, 8.5, 9.4, 10.3, 11.1, 99.0)
    tones = (1.0, 0.93, 1.02, 0.9, 0.97, 1.03, 0.92, 0.99, 0.94, 1.02, 0.91, 0.98, 1.0, 0.95)
    soil, grass = (116, 96, 70), (96, 126, 62)
    fallen = []
    for (ua, va, ta, fa), (ub, vb, tb, fb) in itertools.pairwise(cols):
        if min(ta - fa, tb - fb) < 0.25:
            continue
        broad = 1 + 0.035 * _swell(ua * 0.7 + va * 0.7, 1.9, 3.5)
        for (z0, z1), tone in zip(itertools.pairwise(beds), tones, strict=False):
            lo_a, hi_a = max(fa, z0), min(ta, z1)
            lo_b, hi_b = max(fb, z0), min(tb, z1)
            if hi_a <= lo_a and hi_b <= lo_b:
                continue
            colour = shade(rock, broad * tone)
            p.polygon(
                [
                    _p(ua, va, lo_a),
                    _p(ub, vb, lo_b),
                    _p(ub, vb, max(lo_b, hi_b)),
                    _p(ua, va, max(lo_a, hi_a)),
                ],
                colour,
                colour,
            )
        streak = (ta - fa) * max(0.0, 1.6 * _swell(ua * 1.9 + va * 1.9, 3.7, 1.4) - 0.7)
        if streak > 0.2:
            p.polygon(
                [_p(ua, va, ta - streak), _p(ub, vb, tb - streak), _p(ub, vb, tb), _p(ua, va, ta)],
                shade(rock, broad * 0.84),
            )
        for z in beds[1:-1]:
            if fa + 0.2 < z < ta - 0.2 and fb + 0.2 < z < tb - 0.2:
                wide = 0.1 if z in (3.6, 7.0) else 0.05
                _bar(p, _p(ua, va, z), _p(ub, vb, z), wide * m, shade(rock, broad * 0.74))
        if _speckle(ua * 0.29, va * 0.29) > 0.72:
            k = int((1 + _speckle(va * 0.41, ua * 0.41)) * 5)
            z0, z1 = max(fa, beds[k % 12]), min(ta, beds[k % 12 + 2])
            if z1 - z0 > 0.4:
                _bar(p, _p(ua, va, z0), _p(ua, va, z1), 0.06 * m, shade(rock, 0.66))
        _bar(p, _p(ua, va, fa + 0.2), _p(ub, vb, fb + 0.2), 0.4 * m, shade(rock, broad * 0.8))
        if any(abs(ta - level) < 0.05 for level in ledges):
            _bar(p, _p(ua, va, ta), _p(ub, vb, tb), 0.14 * m, shade(rock, 1.12))
        else:
            cap = 0.35 + 0.3 * (1 + _swell(ua * 2.3, va * 2.3, 1.1))
            p.polygon(
                [_p(ua, va, ta), _p(ub, vb, tb), _p(ub, vb, tb - cap), _p(ua, va, ta - cap)], soil
            )
            if _speckle(ua * 1.7, va * 1.7) > 0.2:
                x, y = _p(ua, va, ta)
                p.blob(x - 0.3 * m, y - 0.25 * m, x + 0.35 * m, y + 0.3 * m, shade(grass, 0.9))
        if _speckle(ua * 0.71, va * 0.71) > 0.35:
            out = 0.4 + 0.8 * (1 + _speckle(ua * 1.3, va * 0.9)) / 2
            size = 0.16 + 0.12 * (1 + _speckle(va * 2.1, ua * 1.7)) / 2
            fallen.append((ua + fu * out, va + fv * out, fa, size))
    _quarry_lumps(p, fallen, shade(rock, 0.95))
    for at in drills:
        k = round(at / 0.25)
        if k >= len(cols):
            continue
        u, v, top, foot = cols[k]
        if top - foot < 1:
            continue
        for j in range(5):
            gu, gv = u + au * 0.35 * j, v + av * 0.35 * j
            _bar(p, _p(gu, gv, foot + 0.15), _p(gu, gv, top - 0.15), 0.06 * m, shade(rock, 0.68))


def _quarry_ladder(p: Painter, u: float, face: float, high: float) -> None:
    """A timber ladder leant against the face standing at `face` along v, at `u`, from the
    floor up past the ledge `high` metres up: two rails and a rung every 0.3 m."""
    timber = (150, 112, 74)
    foot, top = (u, face - 1.0, 0.0), (u, face - 0.2, high + 0.6)
    for du, tone in ((0.0, 1.0), (0.5, 0.8)):
        _beam(
            p, (foot[0] + du, foot[1], 0), (top[0] + du, top[1], top[2]), 0.09, shade(timber, tone)
        )
    for k in range(1, int(top[2] / 0.3)):
        t = k * 0.3 / top[2]
        rv = foot[1] + (top[1] - foot[1]) * t
        _beam(p, (u, rv, top[2] * t), (u + 0.5, rv, top[2] * t), 0.06, shade(timber, 0.9))


def _quarry_block(
    p: Painter, u: float, v: float, z: float, size: tuple[float, float, float], stone: RGB
) -> Top:
    """A squared block of limestone `size` (along u, along v, high) in metres resting at (u, v)
    `z` up, its top lit, its lit arris bright and the marks of the drill down its lit side."""
    m = METRE
    long, deep, high = size
    top = _block(p, u, v, long, deep, high, stone, shade(stone, 1.08), z)
    left, front, right, _ = top
    for t in (0.3, 0.62):
        _bar(
            p,
            _on_wall(left, front, t, 0.12 * m),
            _on_wall(left, front, t, high * m - 0.12 * m),
            0.05 * m,
            shade(stone, 0.84),
        )
    _bar(p, left, front, 0.06 * m, shade(stone, 1.15))
    _bar(p, front, right, 0.05 * m, shade(stone, 1.0))
    return top


def _quarry_derrick(p: Painter, u: float, v: float, mast: float, stone: RGB) -> None:
    """A stiff-leg derrick of timber standing at (u, v): an upright mast `mast` metres tall held
    by two raking stiff legs to the ends of the sills it stands on, a hand winch at its foot, and
    a long jib slung from its foot out to the right, a squared block hanging on its hook."""
    timber, iron, rope = (132, 98, 64), (66, 62, 60), (44, 40, 38)
    back, near = (u - 3.0, v + 2.5), (u - 3.5, v - 3.0)
    _beam(p, (u, v, 0.15), (*back, 0.15), 0.32, shade(timber, 0.7))
    _beam(p, (u, v, mast - 0.4), (*back, 0.3), 0.3, shade(timber, 0.82))
    _beam(p, (u, v, 0), (u, v, mast), 0.42, timber)
    _beam(p, (u - 0.12, v - 0.12, 0), (u - 0.12, v - 0.12, mast), 0.12, shade(timber, 1.2))
    _beam(p, (u, v, mast), (u, v, mast + 0.5), 0.16, iron)
    _beam(p, (u, v, 0.15), (*near, 0.15), 0.32, shade(timber, 0.75))
    _beam(p, (u, v, mast - 0.4), (*near, 0.3), 0.3, shade(timber, 0.95))
    for du in (-0.45, 0.45):
        _beam(p, (u + 0.6 + du, v - 1.0, 0), (u + 0.6 + du, v - 1.0, 0.9), 0.1, iron)
    _beam(p, (u + 0.1, v - 1.0, 0.65), (u + 1.1, v - 1.0, 0.65), 0.42, shade(rope, 1.2))
    _beam(p, (u + 0.15, v - 1.0, 0.75), (u + 1.05, v - 1.0, 0.75), 0.08, shade(iron, 1.5))
    _beam(p, (u + 1.15, v - 1.0, 0.65), (u + 1.25, v - 1.5, 0.95), 0.06, iron)
    tip = (u + 5.0, v + 2.5, mast * 0.72)
    _beam(p, (u, v, mast - 0.1), tip, 0.07, rope)
    _beam(p, (u, v, 0.9), tip, 0.36, shade(timber, 1.02))
    _beam(p, (u, v - 0.2, 0.9), (tip[0], tip[1] - 0.2, tip[2]), 0.1, shade(timber, 1.25))
    hang = (tip[0], tip[1], 1.6)
    _beam(p, tip, (hang[0], hang[1], hang[2] + 1.6), 0.06, rope)
    for du, dv in ((-0.55, -0.3), (0.55, 0.3)):
        _beam(
            p,
            (hang[0], hang[1], hang[2] + 1.6),
            (hang[0] + du, hang[1] + dv, hang[2] + 0.9),
            0.05,
            rope,
        )
    _quarry_block(p, hang[0], hang[1], hang[2], (1.8, 1.1, 0.9), stone)


def _quarry_wagon(p: Painter, u: float, v: float, rock: RGB) -> None:
    """A side-tipping wagon of the tramway at (u, v), running along u: a V-shaped iron body on a
    cradle over four small wheels, heaped with broken limestone."""
    m = METRE
    iron = (92, 86, 82)
    wheel = (34, 32, 34)
    for du in (-0.45, 0.45):
        x, y = _p(u + du, v + 0.3, 0.22)
        p.blob(x - 0.22 * m, y - 0.22 * m, x + 0.22 * m, y + 0.22 * m, wheel)
    _block(p, u, v, 1.4, 0.7, 0.2, shade(iron, 0.6), None, 0.3)
    for du in (-0.45, 0.45):
        x, y = _p(u + du, v - 0.3, 0.22)
        p.blob(x - 0.22 * m, y - 0.22 * m, x + 0.22 * m, y + 0.22 * m, wheel)
        p.blob(x - 0.07 * m, y - 0.07 * m, x + 0.07 * m, y + 0.07 * m, (120, 116, 112))
    for du in (-0.55, 0.55):
        _beam(p, (u + du, v - 0.3, 0.5), (u + du, v - 0.5, 0.75), 0.12, shade(iron, 0.7))
    lo, hi, a, b = 0.6, 1.4, 0.25, 0.65
    _face(
        p,
        [(u - 0.8, v - a, lo), (u + 0.8, v - a, lo), (u + 0.8, v - b, hi), (u - 0.8, v - b, hi)],
        iron,
    )
    _face(
        p,
        [(u + 0.8, v - a, lo), (u + 0.8, v + a, lo), (u + 0.8, v + b, hi), (u + 0.8, v - b, hi)],
        shade(iron, 0.75),
    )
    _beam(p, (u - 0.8, v - b, hi), (u + 0.8, v - b, hi), 0.06, shade(iron, 1.4))
    p.polygon(
        [
            _p(u - 0.8, v - b, hi),
            _p(u + 0.8, v - b, hi),
            _p(u + 0.8, v + b, hi),
            _p(u - 0.8, v + b, hi),
        ],
        shade(rock, 0.7),
    )
    _quarry_lumps(
        p,
        [
            (u - 0.45, v + 0.25, hi, 0.28),
            (u + 0.15, v + 0.3, hi + 0.05, 0.3),
            (u + 0.55, v + 0.05, hi, 0.25),
            (u - 0.35, v - 0.3, hi + 0.05, 0.28),
            (u + 0.25, v - 0.3, hi + 0.1, 0.27),
            (u - 0.05, v, hi + 0.2, 0.3),
        ],
        rock,
    )


def _quarry_kiln(p: Painter, u: float, v: float, stone: RGB, lime: RGB) -> None:
    """A lime kiln at (u, v): a squat tower of coursed rubble limestone 5.6 m square at its
    foot, battered in to 4.4 m at its top 5 m up, with dressed quoins at its corners and a low
    parapet round the dark mouth of its pot, a tall arched draw-eye in its lit side spilling
    white burnt lime out over the ground, and pale smoke drifting off it."""
    m = METRE
    a, b, h = 2.8, 2.2, 5.0

    def at(s: float, z: float, side: int) -> tuple[float, float, float]:
        half = a + (b - a) * z / h
        if side == 0:
            return u + s * half, v - half, z
        return u + half, v + s * half, z

    _face(p, [at(-1, 0, 0), at(1, 0, 0), at(1, h, 0), at(-1, h, 0)], stone)
    _face(p, [at(-1, 0, 1), at(1, 0, 1), at(1, h, 1), at(-1, h, 1)], shade(stone, 0.76))
    for side, tone in ((0, 1.0), (1, 0.76)):
        colour = shade(stone, tone)
        for k, z in enumerate(np.arange(0.5, h, 0.5)):
            _bar(p, _p(*at(-1, z, side)), _p(*at(1, z, side)), 0.05 * m, shade(colour, 0.8))
            for s in np.arange(-0.9 + 0.17 * (k % 2), 0.95, 0.34):
                _bar(
                    p, _p(*at(s, z - 0.5, side)), _p(*at(s, z, side)), 0.04 * m, shade(colour, 0.84)
                )
        for k, z in enumerate(np.arange(0, h, 0.5)):
            for s0, s1 in ((-1, -0.86 + 0.06 * (k % 2)), (0.86 - 0.06 * (k % 2), 1)):
                _face(
                    p,
                    [
                        at(s0, z, side),
                        at(s1, z, side),
                        at(s1, z + 0.45, side),
                        at(s0, z + 0.45, side),
                    ],
                    shade(colour, 1.08),
                )
    rim, wall = b - 0.45, 0.6
    p.polygon(
        [_p(u - b, v - b, h), _p(u + b, v - b, h), _p(u + b, v + b, h), _p(u - b, v + b, h)],
        shade(stone, 1.08),
        shade(stone, 0.7),
    )
    p.polygon(
        [
            _p(u - rim, v - rim, h),
            _p(u + rim, v - rim, h),
            _p(u + rim, v + rim, h),
            _p(u - rim, v + rim, h),
        ],
        shade(stone, 0.62),
    )
    _face(
        p,
        [(u - b, v - b, h), (u + b, v - b, h), (u + b, v - b, h + wall), (u - b, v - b, h + wall)],
        shade(stone, 1.04),
    )
    _face(
        p,
        [(u + b, v - b, h), (u + b, v + b, h), (u + b, v + b, h + wall), (u + b, v - b, h + wall)],
        shade(stone, 0.8),
    )
    cx, cy = _p(u, v, h)
    r = 1.3 * m
    p.blob(cx - r, cy - r / 2, cx + r, cy + r / 2, (52, 44, 42))
    p.blob(cx - r * 0.5, cy - r * 0.15, cx + r * 0.4, cy + r * 0.3, (150, 74, 44))
    arch = [
        at(-0.34 * math.cos(t), 2.3 + 1.1 * math.sin(t), 0) for t in np.linspace(0, math.pi, 11)
    ]
    p.polygon([_p(*c) for c in [at(-0.34, 0, 0), *arch, at(0.34, 0, 0)]], shade(stone, 1.12))
    inner = [
        at(-0.25 * math.cos(t), 2.2 + 0.8 * math.sin(t), 0) for t in np.linspace(0, math.pi, 9)
    ]
    p.polygon([_p(*c) for c in [at(-0.25, 0, 0), *inner, at(0.25, 0, 0)]], (56, 44, 40))
    gx, gy = _p(*at(0, 0.4, 0))
    p.blob(gx - 0.45 * m, gy - 0.5 * m, gx + 0.45 * m, gy + 0.15 * m, (150, 70, 40))
    fx, fy = _p(*at(0, 0, 0))
    p.polygon(
        [
            (fx - 0.9 * m, fy - 0.2 * m),
            (fx + 0.7 * m, fy - 0.55 * m),
            (fx + 1.3 * m, fy + 1.3 * m),
            (fx - 1.9 * m, fy + 1.7 * m),
        ],
        shade(lime, 0.92),
    )
    p.blob(fx - 1.6 * m, fy - 0.1 * m, fx + 1.0 * m, fy + 1.7 * m, lime)
    _quarry_lumps(
        p,
        [
            (u - 0.6, v - a - 1.6, 0, 0.2),
            (u + 0.5, v - a - 1.4, 0, 0.16),
            (u - 0.1, v - a - 2.1, 0, 0.18),
            (u + 0.9, v - a - 0.9, 0, 0.15),
        ],
        lime,
    )
    _wheelbarrow(p, *_p(u + 2.2, v - a - 1.6), lime)
    p.soft(
        [
            (cx - 1.2 * m, cy - 3.0 * m, cx + 1.0 * m, cy - 0.9 * m),
            (cx - 0.4 * m, cy - 5.0 * m, cx + 2.2 * m, cy - 2.6 * m),
            (cx + 1.0 * m, cy - 7.0 * m, cx + 4.0 * m, cy - 4.6 * m),
            (cx + 2.8 * m, cy - 8.6 * m, cx + 5.6 * m, cy - 6.6 * m),
        ],
        (238, 236, 230),
        0.72,
        0.6 * m,
    )


def _quarry_crusher(p: Painter, u: float, v: float, chips: RGB) -> None:
    """A stone crusher in a tall boarded house 5 by 4 m on a base of limestone, its boards and
    corrugated roof whitened by the dust: a flywheel turning on its end, a stove pipe letting off
    steam from the engine, and a chute down its lit side pouring crushed stone onto a pale heap."""
    m = METRE
    wall, base = (164, 148, 124), (200, 192, 170)
    roof = (148, 150, 150)
    low = _block(p, u, v, 5, 4, 2.6, base, None)
    b_left, b_front, b_right, _ = low
    for a, b, tone in ((b_left, b_front, 0.84), (b_front, b_right, 0.66)):
        for drop in (0.65, 1.3, 1.95):
            _bar(
                p,
                _on_wall(a, b, 0, drop * m),
                _on_wall(a, b, 1, drop * m),
                0.05 * m,
                shade(base, tone),
            )
    _patch(p, b_left, b_front, (0.55, 0.85), (0.6 * m, 2.6 * m), DOORWAY)
    house = _block(p, u, v, 5, 4, 4.2, wall, None, 2.6)
    left, front, right, _ = house
    for a, b, tone in ((left, front, 0.82), (front, right, 0.66)):
        for t in np.linspace(0.08, 0.92, 9):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, 4.2 * m), 0.06 * m, shade(wall, tone))
        _bar(
            p,
            _on_wall(a, b, 0, 3.9 * m),
            _on_wall(a, b, 1, 3.9 * m),
            0.3 * m,
            shade(wall, tone * 1.18),
        )
    _patch(p, left, front, (0.15, 0.35), (1.0 * m, 2.2 * m), WINDOW)
    _patch(p, front, right, (0.4, 0.65), (0.8 * m, 2.0 * m), shade(WINDOW, 0.8))
    p.gable(house, 1.6 * m, roof, wall)
    _tiles(p, house, 1.6 * m, roof, (0.2, 0.4, 0.6, 0.8))
    sx, sy = _on_roof(house, 1.6 * m, 0.75, 0.5)
    _bar(p, (sx, sy), (sx, sy - 2.2 * m), 0.25 * m, (60, 58, 58))
    p.soft(
        [
            (sx - 0.6 * m, sy - 3.6 * m, sx + 0.8 * m, sy - 2.2 * m),
            (sx + 0.1 * m, sy - 4.8 * m, sx + 1.9 * m, sy - 3.2 * m),
        ],
        (240, 240, 238),
        0.7,
        0.4 * m,
    )
    wx, wy = _p(u + 2.6, v + 0.6, 1.6)
    r = 1.2 * m
    p.ellipse(wx - r * 0.45, wy - r, wx + r * 0.45, wy + r, (64, 62, 62))
    p.ellipse(wx - r * 0.3, wy - r * 0.78, wx + r * 0.3, wy + r * 0.78, shade(base, 0.62))
    for k in range(3):
        t = k * math.pi / 3
        _bar(
            p,
            (wx - 0.3 * r * math.cos(t), wy - 0.78 * r * math.sin(t)),
            (wx + 0.3 * r * math.cos(t), wy + 0.78 * r * math.sin(t)),
            0.08 * m,
            (64, 62, 62),
        )
    _bar(p, (wx, wy - r), _on_wall(front, right, 0.7, 1.2 * m), 0.07 * m, (40, 36, 34))
    mouth = _p(u + 0.4, v - 2.0, 3.2)
    out = _p(u + 1.6, v - 4.2, 1.4)
    _bar(p, mouth, out, 0.8 * m, (84, 80, 78))
    _bar(p, mouth, out, 0.45 * m, shade(chips, 0.9))
    _cone(p, *_p(u + 1.4, v - 5.2), 2.6 * m, 2.2 * m, chips)


def _quarry_office(p: Painter, u: float, v: float, stone: RGB) -> None:
    """The quarry office: a little house of dressed limestone 5 by 4 m under a slate roof, with
    sash windows in painted frames and a door up a step on its lit side, a dark name board over
    the door, a window in its end and a chimney."""
    m = METRE
    slate, paint = (84, 90, 104), (236, 232, 220)
    top = _block(p, u, v, 5, 4, 3.4, stone, None)
    left, front, right, _ = top
    for a, b, tone in ((left, front, 0.86), (front, right, 0.68)):
        for drop in (0.85, 1.7, 2.55):
            _bar(
                p,
                _on_wall(a, b, 0, drop * m),
                _on_wall(a, b, 1, drop * m),
                0.04 * m,
                shade(stone, tone),
            )
    _patch(p, left, front, (0, 1), (2.9 * m, 3.4 * m), shade(stone, 0.8))
    _patch(p, front, right, (0, 1), (2.9 * m, 3.4 * m), shade(stone, 0.6))
    for t in (0.22, 0.78):
        _patch(p, left, front, (t - 0.11, t + 0.11), (0.8 * m, 2.3 * m), paint)
        _patch(p, left, front, (t - 0.08, t + 0.08), (0.9 * m, 2.2 * m), WINDOW)
        _bar(
            p,
            _on_wall(left, front, t - 0.08, 1.55 * m),
            _on_wall(left, front, t + 0.08, 1.55 * m),
            0.06 * m,
            paint,
        )
    _patch(p, left, front, (0.42, 0.58), (1.0 * m, 3.3 * m), DOOR)
    _patch(p, left, front, (0.36, 0.64), (0.25 * m, 0.8 * m), (52, 58, 66))
    for t in (0.42, 0.47, 0.52, 0.57):
        p.point(*_on_wall(left, front, t, 0.5 * m), paint)
    _block(p, u - 0.0, v - 2.25, 1.2, 0.5, 0.2, shade(stone, 0.9), shade(stone, 1.05))
    _patch(p, front, right, (0.3, 0.7), (0.8 * m, 2.3 * m), paint)
    _patch(p, front, right, (0.35, 0.65), (0.9 * m, 2.2 * m), shade(WINDOW, 0.8))
    p.gable(top, 1.8 * m, slate, stone)
    _tiles(p, top, 1.8 * m, slate, (0.33, 0.66))
    cx, cy = _on_roof(top, 1.8 * m, 0.15, 0.6)
    p.hall(cx, cy, 0.7 * m, 0.7 * m, 1.6 * m, stone, (50, 46, 46))


def limestone_quarry() -> Sprite:
    """A limestone quarry cut into a green hill in three benches: sheer cream faces sawn square,
    crossed by the beds of the stone, scored by drill holes and lit sharp along their tops, pale
    ledges between them, and the hillside's grass and soil coming down ragged to their lip; a
    stiff-leg derrick swinging a squared block over the dusty floor, blocks stacked by it, side-
    tipping wagons of broken stone on a tramway to the crusher, a squat stone lime kiln spilling
    white lime from its draw-eye under pale smoke, and the quarry office."""
    p = Painter(80, 70, (40, 45))
    rock = (224, 212, 184)
    shaded_rock = (178, 174, 162)
    ledge = (236, 234, 224)
    dust = (214, 208, 190)
    grass, scrub, dry = (102, 134, 66), (84, 110, 54), (126, 138, 80)
    soil = (122, 100, 72)
    outcrop = (176, 174, 160)
    stone = (214, 206, 184)
    lime = (248, 246, 240)
    peak = 10.4
    u0, v1, bench, lift, return_ = -9.0, -1.0, 4.5, 3.2, 1.6

    def hill(u: float, v: float) -> float:
        wob = _swell(u, v, 6)
        front = (v - (v1 - 3.0 - 1.2 * wob)) / 8.0
        back = (v1 + 19.0 + 1.5 * wob - v) / 9.0
        left = (u + 19.0 + 2.0 * wob) / 7.0
        right = (10.0 + 2.0 * wob - u) / 8.0
        rise = 1.0
        for t in (front, back, left, right):
            t = min(1.0, max(0.0, t))
            rise *= t * t * (3 - 2 * t)
        return peak * rise * (1 + 0.06 * _swell(u * 1.3, v * 1.3, 4))

    def cut(u: float, v: float) -> float:
        d = max(v - v1, (u0 - u) * bench / return_)
        if d < 0:
            return 0.0
        if d >= 2 * bench:
            return 99.0
        return lift * (1 + int(d // bench))

    def height(u: float, v: float) -> float:
        return min(hill(u, v), cut(u, v))

    def paint(u: float, v: float, z: float, slope: float) -> RGB | None:
        if z < 0.15:
            return None
        grain = 1 + 0.025 * _speckle(u, v) + 0.03 * _swell(u, v, 2.2)
        if cut(u, v) < hill(u, v) - 0.05:
            return shade(rock if slope > 2 else ledge, grain)
        reach = 0.7 + 0.5 * _swell(u * 1.7, v * 1.7, 2)
        if any(
            cut(u + du, v + dv) < hill(u + du, v + dv) - 0.05
            for du, dv in ((reach, 0), (0, -reach), (reach * 0.7, -reach * 0.7))
        ):
            return shade(soil, grain)
        if _swell(u * 0.8 + 5, v * 0.9, 2.6) > 0.78 and z > 1.5:
            return shade(outcrop, grain * (0.92 if _speckle(u * 3, v * 3) > 0.5 else 1.0))
        if _swell(u * 1.2, v * 1.3, 4) > 0.55:
            return shade(scrub, grain)
        if _swell(u * 0.9 + 3, v * 1.1, 5) > 0.6:
            return shade(dry, grain)
        return shade(grass, grain)

    _yard(p, [(0, -6, 14, 9), (-8, -9, 7, 5), (9, -4, 6, 6)], dust, 0.92)
    _yard(
        p,
        [(6.0, -9.5, 1.2, 0.8), (8.0, -12.0, 1.2, 0.8), (10.0, -14.5, 1.1, 0.7)],
        shade(dust, 0.86),
        0.6,
    )
    _land(p, (-22, 13, -6, 21), height, paint, 0.25)
    ledges = (lift, 2 * lift)
    for k in (2, 1, 0):
        back = u0 - k * return_
        _quarry_face(
            p,
            height,
            (back, v1 + k * bench),
            (1, 0),
            22,
            rock,
            ledges,
            (2.5, 9.0) if k == 0 else (5.5,),
        )
        _quarry_face(p, height, (back, -6), (0, 1), v1 + k * bench + 6, shaded_rock, ledges)
    for fu, fv in ((-16.5, 4.0), (-14.0, 9.0), (-6.0, 14.5), (4.0, 13.5), (7.0, 5.5)):
        _fir(p, fu, fv, height(fu, fv) - 0.2, 5.5 + 1.5 * _speckle(fu, fv))
    _quarry_lumps(
        p,
        [
            (-1.5, -7.5, 0, 0.3),
            (-0.6, -9.5, 0, 0.25),
            (7.5, -8.5, 0, 0.28),
            (9.0, -11.0, 0, 0.22),
            (-5.5, -14.0, 0, 0.2),
            (3.5, -12.5, 0, 0.26),
        ],
        stone,
    )
    _quarry_ladder(p, -3.0, v1, lift)
    _cone(p, *_p(6.5, -1.6), 2.0 * METRE, 1.5 * METRE, shade(stone, 0.95))
    _quarry_lumps(
        p,
        [(5.2, -2.0, 0, 0.25), (7.8, -2.4, 0, 0.22), (6.4, -2.9, 0, 0.2), (6.9, -1.0, 1.0, 0.2)],
        stone,
    )
    _rails(p, (-0.5, -3.5), (8.5, -3.5), gauge=0.6, ballast=None)
    _quarry_crusher(p, 10.5, 0.5, (204, 200, 188))
    for su in (4.0, 6.0):
        _quarry_wagon(p, su, -3.5, stone)
    for bu, bv, z, size in (
        (1.0, -6.4, 0.0, (1.8, 1.1, 0.9)),
        (2.9, -6.4, 0.0, (1.8, 1.1, 0.9)),
        (1.9, -6.4, 0.9, (1.8, 1.1, 0.9)),
        (2.6, -8.4, 0.0, (1.4, 1.0, 0.8)),
    ):
        _quarry_block(p, bu, bv, z, size, stone)
    _quarry_derrick(p, -5, -5, 9, stone)
    _quarry_kiln(p, -8.5, -10.5, stone, lime)
    _quarry_office(p, 3, -14, stone)
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
