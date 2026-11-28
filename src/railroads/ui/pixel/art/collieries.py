"""Mines as whole workings, each built round what it digs and wearing its colours: the coal
mine's lattice headframe and brick engine house over black heaps, the iron mine's red timber
shaft tower and stone roasting kilns, the copper mine's terraced hillside and smelter under green
roofs, the gold mine's stamp mill stepping down its slope and the diamond mine's great round pit
of blue ground.

They are laid out in metres on the ground (METRE): `u` along the lit walls, towards the front on
the right, `v` along the shaded walls, towards the back on the right, and `z` up."""

import itertools
import math
from collections.abc import Callable

import numpy as np
from PIL import ImageChops

from railroads.ui.pixel.art.commerce import GOLD, _diamond, _glint
from railroads.ui.pixel.art.industry import (
    DRESSING,
    MILL_BRICK,
    MILL_GLASS,
    MILL_SLATE,
    _mill_window,
    _roof_rows,
    _round_head,
    _smoke,
    _stains,
)
from railroads.ui.pixel.art.painter import (
    CHIMNEY_SHARE,
    DOORWAY,
    PINES,
    RGB,
    TRUNK,
    WINDOW,
    Painter,
    Point,
    Sprite,
    Top,
    fineness,
    shade,
)
from railroads.ui.pixel.art.parts import (
    DOOR,
    METRE,
    _bar,
    _on_wall,
    _patch,
    _plinth,
    _turned,
)

RAIL: RGB = (150, 148, 152)
SLEEPER: RGB = (84, 66, 50)
BALLAST: RGB = (112, 104, 94)
TUB: RGB = (96, 84, 76)
WHEEL: RGB = (34, 32, 34)
ROPE: RGB = (44, 42, 44)
CORRUGATED: RGB = (132, 136, 140)
PLANKS: RGB = (128, 96, 66)

# Where the light falls on the land from, as (u, v, z): from above, over the lit walls' side.
SUN = np.array((-0.3, -0.6, 1.0)) / np.linalg.norm((-0.3, -0.6, 1.0))


def _p(u: float, v: float, z: float = 0.0) -> Point:
    """Where the point `u`, `v` metres along the ground and `z` up shows on the sprite."""
    return (u + v) * METRE, ((u - v) / 2 - z) * METRE


def _beam(
    p: Painter,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    wide: float,
    colour: RGB,
) -> None:
    """A straight beam or rope `wide` metres thick from point `a` to `b`, each (u, v, z)."""
    _bar(p, _p(*a), _p(*b), wide * METRE, colour)


def _block(
    p: Painter,
    u: float,
    v: float,
    length: float,
    depth: float,
    high: float,
    wall: RGB,
    roof: RGB | None,
    z: float = 0.0,
) -> Top:
    """A block centred on (`u`, `v`), `length` metres along `u`, `depth` along `v` and `high`
    tall, standing `z` above the ground."""
    m = METRE
    x, y = _p(u, v, z)
    return p.hall(x, y, length * m, depth * m, high * m, wall, roof)


def _ground(p: Painter, corners: list[tuple[float, float]], colour: RGB) -> None:
    """A flat patch of ground through the (u, v) `corners`."""
    p.polygon([_p(u, v) for u, v in corners], colour)


def _yard(
    p: Painter, patches: list[tuple[float, float, float, float]], colour: RGB, alpha: float = 0.9
) -> None:
    """Trodden or dusty ground melting into the land round it: an oval for every (u, v, half
    across, half deep) of `patches`, in metres."""
    m = METRE
    blobs = []
    for u, v, half_u, half_v in patches:
        x, y = _p(u, v)
        blobs.append((x - half_u * m, y - half_v * m / 2, x + half_u * m, y + half_v * m / 2))
    p.soft(blobs, colour, alpha, 1.2 * m)


def _rails(
    p: Painter,
    a: tuple[float, float],
    b: tuple[float, float],
    gauge: float = 1.435,
    ballast: RGB | None = BALLAST,
) -> None:
    """A track from (u, v) `a` to `b`: two rails `gauge` metres apart on sleepers, bedded in
    `ballast` if given."""
    (au, av), (bu, bv) = a, b
    long = math.hypot(bu - au, bv - av)
    nu, nv = -(bv - av) / long, (bu - au) / long
    half = gauge / 2
    if ballast is not None:
        bed = half + 1.1
        _ground(
            p,
            [
                (au + nu * bed, av + nv * bed),
                (bu + nu * bed, bv + nv * bed),
                (bu - nu * bed, bv - nv * bed),
                (au - nu * bed, av - nv * bed),
            ],
            ballast,
        )
    tie = half + 0.45
    for k in range(int(long / 0.8)):
        t = (k + 0.5) * 0.8 / long
        u, v = au + (bu - au) * t, av + (bv - av) * t
        _beam(p, (u + nu * tie, v + nv * tie, 0), (u - nu * tie, v - nv * tie, 0), 0.26, SLEEPER)
    for side in (half, -half):
        _beam(
            p,
            (au + nu * side, av + nv * side, 0.1),
            (bu + nu * side, bv + nv * side, 0.1),
            0.12,
            RAIL,
        )


def _tub(
    p: Painter,
    u: float,
    v: float,
    load: RGB,
    along_u: bool = True,
    z: float = 0.0,
    glint: RGB | None = None,
) -> None:
    """A mine tub of riveted iron, 1.5 m long and a metre wide, on four small wheels at (u, v),
    heaped with `load` and glinting with `glint` if given."""
    m = METRE
    length, depth = (1.5, 1.0) if along_u else (1.0, 1.5)
    for du, dv in ((-0.45, -0.5), (0.45, -0.5)) if along_u else ((0.5, -0.45), (0.5, 0.45)):
        x, y = _p(u + du, v + dv, z + 0.18)
        p.blob(x - 0.2 * m, y - 0.2 * m, x + 0.2 * m, y + 0.2 * m, WHEEL)
    top = _block(p, u, v, length, depth, 0.75, TUB, shade(TUB, 0.6), z + 0.2)
    left, front, _, _ = top
    _patch(p, left, front, (0, 1), (0.08 * m, 0.2 * m), shade(TUB, 1.3))
    x, y = _p(u, v, z + 0.95)
    p.blob(x - 0.62 * m, y - 0.55 * m, x + 0.62 * m, y + 0.28 * m, load)
    p.blob(x - 0.42 * m, y - 0.5 * m, x + 0.1 * m, y - 0.1 * m, shade(load, 1.3))
    if glint is not None:
        p.blob(x - 0.2 * m, y - 0.3 * m, x + 0.05 * m, y - 0.1 * m, glint)


def _wagon(p: Painter, u: float, v: float, load: RGB, body: RGB) -> None:
    """A four-wheeled open railway wagon 6.5 m long along `v` with planked sides at (u, v),
    heaped up with `load`."""
    m = METRE
    length, depth = 2.6, 6.5
    for du, dv in ((1.3, -1.8), (1.3, 1.8)):
        x, y = _p(u + du, v + dv, 0.45)
        p.blob(x - 0.5 * m, y - 0.5 * m, x + 0.5 * m, y + 0.5 * m, WHEEL)
        p.blob(x - 0.15 * m, y - 0.15 * m, x + 0.15 * m, y + 0.15 * m, (120, 116, 112))
    _block(p, u, v, length - 0.4, depth - 0.4, 0.35, (36, 34, 36), None, 0.7)
    top = _block(p, u, v, length, depth, 1.4, body, shade(load, 0.7), 1.0)
    left, front, right, _ = top
    for a, b, tone in ((left, front, 0.8), (front, right, 0.62)):
        for drop in (0.47, 0.94):
            _bar(
                p,
                _on_wall(a, b, 0, drop * m),
                _on_wall(a, b, 1, drop * m),
                0.06 * m,
                shade(body, tone),
            )
        for t in (0.02, 0.5, 0.98):
            _bar(
                p,
                _on_wall(a, b, t, 0),
                _on_wall(a, b, t, 1.4 * m),
                0.12 * m,
                shade(body, tone * 0.8),
            )
    _load(p, top, depth, 0.7, load)


def _load(p: Painter, top: Top, long: float, rise: float, load: RGB) -> None:
    """Loose `load` heaped `rise` metres over the open top of a wagon or bin, in a ridge along
    its longer side `long` metres long, lit on the left, with lumps on it."""
    m = METRE
    left, front, right, back = top
    along_lit = math.dist(left, front) >= math.dist(front, right)
    if along_lit:
        near, far = (left, front), (back, right)
    else:
        near, far = (front, right), (left, back)
    mid = [
        ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - rise * m) for a, b in zip(near, far, strict=False)
    ]
    inset = 0.4 / max(long, 1.0)
    ridge = [_on_wall(mid[0], mid[1], inset, 0), _on_wall(mid[0], mid[1], 1 - inset, 0)]
    p.polygon([near[0], near[1], ridge[1], ridge[0]], shade(load, 1.12))
    p.polygon([far[0], far[1], ridge[1], ridge[0]], shade(load, 0.8))
    p.polygon([near[1], far[1], ridge[1]], shade(load, 0.68))
    p.polygon([near[0], far[0], ridge[0]], shade(load, 0.95))
    for t in (0.2, 0.45, 0.7, 0.88):
        x, y = _on_wall(ridge[0], ridge[1], t, 0.2 * m)
        p.blob(
            x - 0.22 * m,
            y - 0.14 * m,
            x + 0.22 * m,
            y + 0.14 * m,
            shade(load, 1.3 if t < 0.5 else 0.7),
        )


def _cone(
    p: Painter,
    u: float,
    v: float,
    half: float,
    high: float,
    colour: RGB,
    lumps: RGB | None = None,
    power: float = 1.25,
) -> None:
    """A heap of loose stuff `half` metres across its foot and `high` tall, tipped at (u, v):
    slopes lying at their angle of rest up to a rounded top, lit on the left, with lumps of
    `lumps` rolled down it and lying round its foot if given; with a higher `power` it rounds
    into a hill."""
    m = METRE
    x, y = _p(u, v)
    w, h = half * m, high * m

    def crest(s: float) -> Point:
        return x + s * w, y - h * (1 - abs(s) ** power)

    split = 0.12
    lit_crest = [crest(s) for s in np.linspace(-1, split, 12)]
    dark_crest = [crest(s) for s in np.linspace(split, 1, 10)]
    turn = math.radians(62)
    front = (x + w * math.cos(turn), y + w / 2 * math.sin(turn))
    lit_foot = [
        (x + w * math.cos(a), y + w / 2 * math.sin(a)) for a in np.linspace(turn, math.pi, 8)
    ]
    dark_foot = [(x + w * math.cos(a), y + w / 2 * math.sin(a)) for a in np.linspace(0, turn, 6)]
    p.polygon([*lit_crest, front, *lit_foot[::-1][:-1]], colour)
    p.polygon([*dark_crest, *dark_foot[:-1][::-1], front], shade(colour, 0.74))
    p.polygon(
        [crest(-0.95), crest(-0.45), (x - 0.42 * w, y + 0.12 * w), (x - 0.9 * w, y + 0.1 * w)],
        shade(colour, 1.1),
    )
    p.polygon([crest(-0.3), crest(split), (x + 0.1 * w, y - 0.25 * h)], shade(colour, 1.16))
    if lumps is None:
        return
    for k, (s, up) in enumerate(
        ((-0.7, 0.18), (-0.35, 0.42), (0.05, 0.6), (0.4, 0.3), (0.72, 0.12), (-0.1, 0.2))
    ):
        cx, cy = x + s * w, y - h * (1 - abs(s) ** 1.25) * up + 0.1 * w * (1 - up)
        r = 0.32 * m
        p.blob(cx - r, cy - r * 0.7, cx + r, cy + r * 0.7, shade(lumps, 1.0 if k % 2 else 0.8))
    for s, d in ((-1.05, 0.05), (0.9, 0.2), (-0.5, 0.42), (0.35, 0.46)):
        cx, cy = x + s * w, y + d * w
        p.blob(cx - 0.28 * m, cy - 0.18 * m, cx + 0.28 * m, cy + 0.18 * m, shade(lumps, 0.85))


def _sheave(
    p: Painter, u: float, v: float, z: float, radius: float, colour: RGB, spokes: int = 8
) -> None:
    """A spoked winding wheel standing upright on its axle across `v` at (u, v, z), its rim
    turning along `u`, so the rope runs off it along the ground."""

    def rim(t: float, r: float) -> tuple[float, float, float]:
        return u + r * math.cos(t), v, z + r * math.sin(t)

    turns = np.linspace(0, 2 * math.pi, 25)
    for k in range(spokes):
        t = k * 2 * math.pi / spokes
        _beam(p, (u, v, z), rim(t, radius), 0.12, shade(colour, 0.8))
    for a, b in itertools.pairwise(turns):
        _beam(p, rim(a, radius), rim(b, radius), 0.34, colour)
        _beam(p, rim(a, radius - 0.12), rim(b, radius - 0.12), 0.08, shade(colour, 1.5))
    x, y = _p(u, v, z)
    p.blob(x - 0.3 * METRE, y - 0.3 * METRE, x + 0.3 * METRE, y + 0.3 * METRE, shade(colour, 0.6))


def _stack(p: Painter, u: float, v: float, tall: float, brick: RGB, band: RGB, smoke: RGB) -> None:
    """A round brick chimney `tall` metres high, lowered by CHIMNEY_SHARE, on a square plinth,
    ringed with iron bands, with a band of `band` under its corbelled crown and `smoke` rising."""
    m = METRE
    tall *= CHIMNEY_SHARE
    x, y = _p(u, v)
    base = 3.2
    _block(p, u, v, 3.4, 3.4, base, brick, shade(brick, 1.1))
    foot = y - base * m
    shaft = (tall - base) * m
    r0, r1 = 1.45 * m, 0.9 * m
    _turned(p, x, foot, [(0, r0), (shaft, r1)], brick)

    def ring(up: float, wide: float, colour: RGB) -> None:
        half = r0 + (r1 - r0) * up / shaft
        cy = foot - up
        _bar(p, (x - half, cy), (x, cy + half / 2), wide, colour)
        _bar(p, (x, cy + half / 2), (x + half, cy), wide, shade(colour, 0.8))

    for up in (0.3, 0.55, 0.78):
        ring(shaft * up, 0.15 * m, (52, 46, 46))
    ring(shaft - 1.4 * m, 0.55 * m, band)
    _turned(
        p,
        x,
        foot - shaft,
        [(0, r1), (0.5 * m, r1 * 1.25), (1.2 * m, r1 * 1.25)],
        shade(brick, 0.5),
        (26, 24, 26),
    )
    _smoke(p, x + 1, foot - shaft - 4, smoke, 2)


def _lattice_headframe(
    p: Painter, u: float, v: float, high: float, steel: RGB, wheel: RGB, reach: float
) -> None:
    """A riveted steel headframe over the shaft at (u, v): four legs braced in crosses rising
    `high` metres to a railed sheave deck carrying two winding wheels side by side, propped by
    two raking legs leaning back `reach` metres towards the engine house along -u."""
    bu, bv, tu, tv = 2.3, 2.5, 1.2, 1.4
    levels = [0.0, high * 0.3, high * 0.57, high * 0.8, high]

    def leg(su: float, sv: float, z: float) -> tuple[float, float, float]:
        f = z / high
        return u + su * (bu + (tu - bu) * f), v + sv * (bv + (tv - bv) * f), z

    dark, lit = shade(steel, 0.7), shade(steel, 1.25)

    def face(a: tuple[float, float], b: tuple[float, float], colour: RGB) -> None:
        for z0, z1 in itertools.pairwise(levels):
            _beam(p, leg(*a, z0), leg(*b, z1), 0.16, colour)
            _beam(p, leg(*b, z0), leg(*a, z1), 0.16, colour)
            _beam(p, leg(*a, z1), leg(*b, z1), 0.22, colour)

    for side in (1, -1):
        foot = (u - reach, v + side * 2.6, 0.0)
        head = (u - tu, v + side * tv, high - 0.6)
        _beam(p, foot, head, 0.5, dark if side > 0 else steel)
        for f in (0.35, 0.65):
            on = tuple(fa + (ha - fa) * f for fa, ha in zip(foot, head, strict=False))
            _beam(p, on, leg(-1, side, on[2]), 0.14, dark)
    face((-1, 1), (1, 1), dark)
    face((-1, -1), (-1, 1), dark)
    for su, sv in ((-1, 1), (1, 1), (-1, -1)):
        _beam(p, leg(su, sv, 0), leg(su, sv, high), 0.4, dark)
    deck = high + 0.2
    _block(p, u, v, 2 * tu + 1.2, 2 * tv + 1.2, 0.5, steel, shade(steel, 1.15), high - 0.3)
    for sv in (0.9, -0.9):
        _sheave(p, u, v + sv, deck + 2.6, 2.4, wheel)
        _beam(p, (u - 1.4, v + sv, deck), (u, v + sv, deck + 2.6), 0.22, steel)
        _beam(p, (u + 1.4, v + sv, deck), (u, v + sv, deck + 2.6), 0.22, steel)
    for su, sv in ((-1, -1), (1, -1), (1, 1)):
        x0, z0 = u + su * (tu + 0.6), deck + 0.5
        _beam(p, (x0, v + sv * (tv + 0.6), z0), (x0, v + sv * (tv + 0.6), z0 + 1.0), 0.08, lit)
    _beam(
        p,
        (u - tu - 0.6, v - tv - 0.6, deck + 1.5),
        (u + tu + 0.6, v - tv - 0.6, deck + 1.5),
        0.08,
        lit,
    )
    _beam(
        p,
        (u + tu + 0.6, v - tv - 0.6, deck + 1.5),
        (u + tu + 0.6, v + tv + 0.6, deck + 1.5),
        0.08,
        lit,
    )
    face((1, -1), (1, 1), steel)
    face((-1, -1), (1, -1), lit)
    _beam(p, leg(1, -1, 0), leg(1, -1, high), 0.45, steel)
    _beam(p, leg(-1, -1, 0), leg(-1, -1, high), 0.45, lit)
    _beam(p, leg(1, 1, 0), leg(1, 1, high), 0.4, shade(steel, 0.85))


def _brick_house(
    p: Painter,
    u: float,
    v: float,
    length: float,
    depth: float,
    high: float,
    rise: float,
    brick: RGB,
    roof: RGB,
    lit_windows: tuple[float, ...],
    end_windows: tuple[float, ...],
) -> Top:
    """A brick house of `length` by `depth` metres and `high` to its eaves under a slate gable
    roof `rise` high, its ridge along `u` so its gable end faces along it: tall round-headed
    windows under cream arches at `lit_windows` along its lit wall and `end_windows` along its
    gable end, a dark plinth, cream bands and soot creeping up its walls."""
    m = METRE
    top = _block(p, u, v, length, depth, high, brick, None)
    left, front, right, _ = top
    _plinth(p, top, high * m, brick, 0.6 * m)
    _stains(p, left, front, high * m, (0.06, 0.35, 0.64, 0.95), shade(brick, 0.66), 1.6 * m)
    for a, b in ((left, front), (front, right)):
        _bar(p, _on_wall(a, b, 0, 0.25 * m), _on_wall(a, b, 1, 0.25 * m), 0.5 * m, DRESSING)
    for t in lit_windows:
        _mill_window(p, left, front, length * m, t, 1.6 * m, (1.6 * m, (high - 1.4) * m))
    for t in end_windows:
        _mill_window(
            p,
            front,
            right,
            depth * m,
            t,
            1.6 * m,
            (1.6 * m, (high - 1.4) * m),
            shade(MILL_GLASS, 0.85),
        )
    p.gable(top, rise * m, roof, brick)
    _roof_rows(p, top, rise * m, roof, (0.25, 0.5, 0.75))
    return top


def _emblem(p: Painter, x: float, y: float, size: float, tools: RGB) -> None:
    """The miners' sign, a hammer and a pick crossed, in `tools`, `size` units tall, centred at
    (x, y)."""
    k = size * 0.36
    head = size * 0.24
    _bar(p, (x - k, y + k), (x + k, y - k), size * 0.1, tools)
    _bar(p, (x + k, y + k), (x - k, y - k), size * 0.1, tools)
    hx, hy = x + k, y - k
    _bar(
        p, (hx - head * 0.7, hy - head * 0.7), (hx + head * 0.7, hy + head * 0.7), size * 0.2, tools
    )
    px, py = x - k, y - k
    p.polygon(
        [
            (px - head * 1.1, py + head * 0.9),
            (px - head * 0.1, py - head * 0.25),
            (px + head * 1.1, py - head * 0.9),
            (px + head * 0.1, py + head * 0.25),
        ],
        tools,
    )


def coal_mine() -> Sprite:
    """A colliery: a riveted steel headframe 25 m tall with two winding wheels, its ropes running
    back to a red-brick winding-engine house with tall round-headed windows and the miners' sign
    in its gable, a boiler house and a tall smoking chimney behind, tubs of coal running on a
    gantry to a timber screening shed over the siding where a wagon is loaded with black coal, a
    black spoil heap behind it all, a lamp room, and a cinder yard dark with coal dust."""
    m = METRE
    p = Painter(52, 60, (26, 44))
    coal, glint = (38, 36, 40), (120, 122, 132)
    spoil = (62, 58, 60)
    brick = shade(MILL_BRICK, 0.92)
    _yard(p, [(-8, 2, 18, 14), (6, -6, 16, 12), (2, 14, 12, 8)], (128, 116, 98), 0.9)
    _yard(p, [(11, -6, 4, 12), (-1, -4, 4, 6), (-2, 10, 7, 4)], (70, 66, 64), 0.6)
    _cone(p, -4, 18, 10, 11, spoil, shade(spoil, 1.25))
    _cone(p, 4, 17, 5, 5, shade(spoil, 0.9), shade(spoil, 1.2))
    trestle = (70, 60, 52)
    _beam(p, (11, 3, 9), (-3, 16, 10.5), 0.35, trestle)
    for k in range(5):
        t = (k + 0.5) / 5
        bu, bv, bz = 11 - 14 * t, 3 + 13 * t, 9 + 1.5 * t
        _beam(p, (bu, bv, 0), (bu, bv, bz), 0.18, trestle)
    _tub(p, 0.5, 13.3, coal, along_u=False, z=10.3)
    _stack(p, -22, 13, 42, brick, DRESSING, (92, 90, 94))
    boiler = _block(p, -17, 11, 8, 6, 4.5, brick, None)
    _plinth(p, boiler, 4.5 * m, brick, 0.5 * m)
    p.gable(boiler, 2 * m, MILL_SLATE, brick)
    _roof_rows(p, boiler, 2 * m, MILL_SLATE, (0.5,))
    engine = _brick_house(p, -16, 2.5, 10, 9, 7.5, 3.2, brick, MILL_SLATE, (0.25, 0.75), (0.5,))
    _, e_front, e_right, _ = engine
    gx, gy = _on_wall(e_front, e_right, 0.5, -1.4 * m)
    _emblem(p, gx, gy, 1.6 * m, (226, 214, 186))
    for sv in (-0.9, 0.9):
        hx, hy = _p(-11, 2.5 + sv, 6.4)
        p.blob(hx - 0.3 * m, hy - 0.3 * m, hx + 0.3 * m, hy + 0.3 * m, DOORWAY)
    high = 25
    _lattice_headframe(p, 0, 0, high, (100, 100, 106), (64, 62, 66), 9)
    top = high + 0.2 + 2.6
    for sv in (-0.9, 0.9):
        _beam(p, (-11, 2.5 + sv, 6.4), (-1.6, sv, top + 1.8), 0.1, ROPE)
        _beam(p, (2.4, sv, top), (2.4, sv, 4), 0.08, ROPE)
    shed = _block(p, 1.5, 0, 7, 7, 4.2, CORRUGATED, None)
    s_left, s_front, _, _ = shed
    for k in range(1, 12):
        t = k / 12
        _bar(
            p,
            _on_wall(s_left, s_front, t, 0),
            _on_wall(s_left, s_front, t, 4.2 * m),
            0.06 * m,
            shade(CORRUGATED, 0.82),
        )
    _patch(p, s_left, s_front, (0.4, 0.65), (1.4 * m, 4.2 * m), DOORWAY)
    p.gable(shed, 1.6 * m, shade(CORRUGATED, 0.9), CORRUGATED)
    _rails(p, (11.6, 12), (11.6, -18))
    gantry = (88, 70, 54)
    for gu in (5.6, 7.6):
        for sv in (-1, 1):
            _beam(p, (gu, sv, 0), (gu, sv, 5.0 + (gu - 5) * 0.3), 0.26, gantry)
    _beam(p, (5, -1.1, 5.2), (8.4, -1.1, 6.2), 0.3, gantry)
    _beam(p, (5, 1.1, 5.2), (8.4, 1.1, 6.2), 0.3, shade(gantry, 0.8))
    _tub(p, 6.2, 0, coal, z=5.7, glint=glint)
    screens = (84, 74, 66)
    legs = (60, 56, 56)
    for su, sv in ((-3.6, 4), (3.6, 4), (-3.6, -4), (3.6, -4)):
        _beam(p, (12 + su, sv, 0), (12 + su, sv, 5.0), 0.5, legs)
    _wagon(p, 11.6, -1, coal, (76, 64, 56))
    top = _block(p, 12, 0, 7.4, 8.4, 6.4, screens, None, 5.0)
    left, front, right, _ = top
    for a, b, tone in ((left, front, 1.0), (front, right, 0.8)):
        for k in range(1, 10):
            _bar(
                p,
                _on_wall(a, b, k / 10, 0),
                _on_wall(a, b, k / 10, 6.4 * m),
                0.07 * m,
                shade(screens, 0.72 * tone),
            )
    _patch(p, left, front, (0.15, 0.85), (1.2 * m, 2.2 * m), WINDOW)
    _patch(p, front, right, (0.2, 0.45), (1.2 * m, 2.2 * m), shade(WINDOW, 0.8))
    _patch(p, front, right, (0.6, 0.85), (1.2 * m, 2.2 * m), shade(WINDOW, 0.8))
    p.gable(top, 2.4 * m, (58, 56, 60), screens)
    _roof_rows(p, top, 2.4 * m, (58, 56, 60), (0.5,))
    chute = _p(11.6, -4.2, 5.2)
    _bar(p, chute, (chute[0] - 0.6 * m, chute[1] + 1.4 * m), 0.9 * m, (50, 48, 50))
    _wagon(p, 11.6, -11, coal, (112, 64, 48))
    _rails(p, (1.0, -2.5), (1.0, -10.5), gauge=0.6, ballast=None)
    for k, sv in enumerate((-4.0, -5.8, -7.6, -9.4)):
        _tub(
            p,
            1.0,
            sv,
            coal if k % 3 else shade(TUB, 0.8),
            along_u=False,
            glint=glint if k == 1 else None,
        )
    office = _block(p, -7, -10, 8, 6, 4.6, brick, None)
    o_left, o_front, o_right, _ = office
    _plinth(p, office, 4.6 * m, brick, 0.5 * m)
    for t in (0.18, 0.42, 0.82):
        _mill_window(p, o_left, o_front, 8 * m, t, 1.1 * m, (1.2 * m, 3.6 * m))
    _patch(p, o_left, o_front, (0.56, 0.68), (1.6 * m, 4.6 * m), DOOR)
    _mill_window(
        p, o_front, o_right, 6 * m, 0.5, 1.1 * m, (1.2 * m, 3.6 * m), shade(MILL_GLASS, 0.85)
    )
    p.gable(office, 2.2 * m, MILL_SLATE, brick)
    _roof_rows(p, office, 2.2 * m, MILL_SLATE, (0.5,))
    return p.sprite()


def _face(p: Painter, corners: list[tuple[float, float, float]], colour: RGB) -> None:
    """A flat face through the (u, v, z) `corners`."""
    p.polygon([_p(*c) for c in corners], colour, shade(colour, 0.6))


def _pyramid(p: Painter, top: Top, rise: float, roof: RGB) -> Point:
    """A pyramid roof `rise` units high over a block's top; returns its peak."""
    left, front, right, back = top
    peak = ((left[0] + right[0]) / 2, (back[1] + front[1]) / 2 - rise)
    p.polygon([left, front, peak], roof, shade(roof, 0.6))
    p.polygon([front, right, peak], shade(roof, 0.72), shade(roof, 0.5))
    return peak


def _boards(p: Painter, top: Top, high: float, colour: RGB, every: float = 0.9) -> None:
    """Upright boards on a block's lit and shaded walls, `every` metres apart, `high` tall."""
    m = METRE
    left, front, right, _ = top
    for a, b, tone in ((left, front, 0.82), (front, right, 0.66)):
        long = math.dist(a, b) / m * 0.9
        count = max(2, round(long / every))
        for k in range(1, count):
            _bar(
                p,
                _on_wall(a, b, k / count, 0),
                _on_wall(a, b, k / count, high * m),
                0.06 * m,
                shade(colour, tone),
            )


def _trim(p: Painter, top: Top, high: float, colour: RGB) -> None:
    """Painted corner boards and an eaves board round a block's lit and shaded walls."""
    m = METRE
    left, front, right, _ = top
    for a, b in ((left, front), (front, right)):
        _bar(p, _on_wall(a, b, 0, 0.15 * m), _on_wall(a, b, 1, 0.15 * m), 0.3 * m, colour)
    for corner in (left, front, right):
        _bar(p, corner, (corner[0], corner[1] + high * m), 0.35 * m, colour)


def _framed(
    p: Painter,
    a: Point,
    b: Point,
    span: tuple[float, float],
    drop: float,
    tall: float,
    frame: RGB,
    glass: RGB = WINDOW,
) -> None:
    """A window in a painted frame on the wall under the edge from `a` to `b`, `tall` metres
    high from `drop` down, with a glazing bar across it."""
    m = METRE
    t0, t1 = span
    pad = 0.05 * (t1 - t0) / 0.1
    _patch(p, a, b, (t0 - pad, t1 + pad), ((drop - 0.2) * m, (drop + tall + 0.2) * m), frame)
    _patch(p, a, b, span, (drop * m, (drop + tall) * m), glass)
    _bar(
        p,
        _on_wall(a, b, (t0 + t1) / 2, drop * m),
        _on_wall(a, b, (t0 + t1) / 2, (drop + tall) * m),
        0.07 * m,
        frame,
    )


def _kiln(p: Painter, u: float, v: float, high: float, stone: RGB, ore: RGB) -> None:
    """A round stone roasting kiln `high` metres tall at (u, v), battered in towards its top,
    bound with iron hoops and streaked with rust, its mouth heaped with `ore` and a glowing
    arched draw-hole at its foot."""
    m = METRE
    x, y = _p(u, v)
    r0, r1 = 3.0 * m, 2.0 * m
    _turned(p, x, y, [(0, r0), (1.2 * m, r0 * 0.96), (high * m, r1)], stone, shade(ore, 0.7))
    p.blob(x - r1 * 0.7, y - high * m - r1 * 0.35, x + r1 * 0.7, y - high * m + r1 * 0.3, ore)
    for up in (0.3, 0.55, 0.8):
        h = high * m * up
        half = r0 + (r1 - r0) * up
        _bar(p, (x - half, y - h), (x, y - h + half / 2), 0.14 * m, (70, 60, 56))
        _bar(p, (x, y - h + half / 2), (x + half, y - h), 0.14 * m, (54, 46, 44))
    for k in range(1, 9):
        h = high * m * k / 9
        half = r0 + (r1 - r0) * k / 9
        _bar(
            p, (x - half, y - h), (x - half * 0.2, y - h + half * 0.4), 0.05 * m, shade(stone, 0.8)
        )
    for t, reach in ((-0.55, 0.22), (-0.1, 0.34), (0.35, 0.18)):
        _bar(
            p,
            (x + t * r1, y - high * m + 0.4 * m),
            (x + t * (r1 + (r0 - r1) * reach), y - high * m * (1 - reach)),
            0.16 * m,
            shade(ore, 1.15),
        )
    ax, ay = x - 0.35 * r0, y + 0.42 * r0
    p.polygon(
        [
            (ax - 0.75 * m, ay),
            (ax - 0.75 * m, ay - 1.3 * m),
            (ax - 0.4 * m, ay - 1.9 * m),
            (ax + 0.4 * m, ay - 1.9 * m),
            (ax + 0.75 * m, ay - 1.3 * m),
            (ax + 0.75 * m, ay),
        ],
        (238, 220, 186),
    )
    p.polygon(
        [
            (ax - 0.5 * m, ay),
            (ax - 0.5 * m, ay - 1.2 * m),
            (ax - 0.25 * m, ay - 1.6 * m),
            (ax + 0.25 * m, ay - 1.6 * m),
            (ax + 0.5 * m, ay - 1.2 * m),
            (ax + 0.5 * m, ay),
        ],
        (52, 34, 28),
    )
    p.blob(ax - 0.4 * m, ay - 0.7 * m, ax + 0.4 * m, ay, (240, 136, 52))
    p.blob(ax - 0.2 * m, ay - 0.5 * m, ax + 0.2 * m, ay - 0.1 * m, (255, 210, 110))


def iron_mine() -> Sprite:
    """An iron mine painted red as its ore: a shaft tower boarded and painted rust red with white
    trim, open at its top round the winding wheel under a pyramid roof, a red winding house joined
    to it by a sloping rope gallery, a battery of three round stone roasting kilns glowing at
    their draw-holes under a charging deck where a tub of ore waits, a boarded ore bin on stone
    piers with a chute over the siding filling a wagon with red ore, rust-red heaps of ore, tubs of it on a tramway,
    and a corrugated shed streaked with rust, all on ground stained red by the ore."""
    m = METRE
    p = Painter(48, 56, (24, 38))
    ore, glint = (150, 64, 44), (204, 118, 82)
    red, trim = (154, 54, 40), (234, 228, 214)
    stone = (184, 160, 120)
    iron = (98, 102, 110)
    rust = (150, 84, 50)
    _yard(p, [(-6, 2, 18, 14), (6, -6, 16, 12), (-2, 14, 10, 8)], (156, 116, 86), 0.88)
    _yard(p, [(4, -10, 7, 5), (13, -2, 4, 10), (-2, 13, 5, 6)], (136, 70, 50), 0.55)
    deck = (112, 86, 60)
    for dv in (6, 11, 17, 21):
        _beam(p, (-4.6, dv, 0), (-4.6, dv, 8.6), 0.35, shade(deck, 0.8))
    _block(p, -4.6, 13, 2.6, 17, 0.5, deck, shade(deck, 1.2), 8.6)
    for k in range(9):
        dv = 4.8 + k * 2
        _beam(p, (-3.4, dv, 9.1), (-3.4, dv, 10.1), 0.08, shade(deck, 1.3))
    _beam(p, (-3.4, 4.6, 10.1), (-3.4, 21.4, 10.1), 0.08, shade(deck, 1.3))
    _tub(p, -4.6, 16.5, ore, along_u=False, z=9.1, glint=glint)
    for kv in (20, 14, 8):
        _kiln(p, -1.6, kv, 8.6, stone, ore)
    house = _block(p, -15, 2, 8, 7, 5.5, red, None)
    h_left, h_front, h_right, _ = house
    _boards(p, house, 5.5, red)
    _trim(p, house, 5.5, trim)
    _framed(p, h_left, h_front, (0.2, 0.36), 1.5, 2.2, trim)
    _framed(p, h_left, h_front, (0.62, 0.78), 1.5, 2.2, trim)
    _framed(p, h_front, h_right, (0.35, 0.6), 1.5, 2.2, trim, shade(WINDOW, 0.8))
    p.gable(house, 2.6 * m, iron, red)
    _roof_rows(p, house, 2.6 * m, iron, (0.5,))
    g0, g1 = (-11.2, 2, 5.6), (-7.6, 1, 12.5)
    w, h = 1.1, 2.0
    _face(
        p,
        [
            (g0[0], g0[1] - w, g0[2]),
            (g1[0], g1[1] - w, g1[2]),
            (g1[0], g1[1] - w, g1[2] + h),
            (g0[0], g0[1] - w, g0[2] + h),
        ],
        red,
    )
    _face(
        p,
        [
            (g0[0], g0[1] - w, g0[2] + h),
            (g1[0], g1[1] - w, g1[2] + h),
            (g1[0], g1[1] + w, g1[2] + h),
            (g0[0], g0[1] + w, g0[2] + h),
        ],
        iron,
    )
    _beam(p, (g0[0], g0[1] - w, g0[2] + 0.2), (g1[0], g1[1] - w, g1[2] + 0.2), 0.2, trim)
    _beam(
        p, (g0[0] + 1.6, g0[1] - w, 0), (g0[0] + 1.6, g0[1] - w, g0[2] + 2.9), 0.3, shade(red, 0.7)
    )
    body = 15.0
    tower = _block(p, -4, 1, 7, 7, body, red, None)
    t_left, t_front, t_right, _ = tower
    _boards(p, tower, body, red)
    _trim(p, tower, body, trim)
    for drop in (3.0, 7.5):
        _framed(p, t_left, t_front, (0.4, 0.6), drop, 2.0, trim)
        _framed(p, t_front, t_right, (0.4, 0.6), drop, 2.0, trim, shade(WINDOW, 0.8))
    _patch(p, t_left, t_front, (0.32, 0.68), ((body - 3.6) * m, (body - 0.3) * m), trim)
    _patch(p, t_left, t_front, (0.35, 0.65), ((body - 3.4) * m, (body - 0.5) * m), DOORWAY)
    ex, ey = _on_wall(t_left, t_front, 0.5, 9.6 * m)
    _emblem(p, ex, ey, 2.0 * m, trim)
    _block(p, -4, 1, 7.2, 7.2, 0.4, trim, (58, 50, 48), body)
    post = shade(red, 0.75)
    corners = ((-3.4, -3.4), (3.4, -3.4), (3.4, 3.4), (-3.4, 3.4))
    for cu, cv in corners[2:]:
        _beam(p, (-4 + cu, 1 + cv, body), (-4 + cu, 1 + cv, body + 5.6), 0.4, post)
    _sheave(p, -4, 1, body + 2.9, 2.4, (150, 150, 156), spokes=6)
    _beam(p, (-4 + 2.4, 1, body + 2.9), (-4 + 2.4, 1, body + 0.4), 0.1, ROPE)
    for cu, cv in corners[:2]:
        _beam(p, (-4 + cu, 1 + cv, body), (-4 + cu, 1 + cv, body + 5.6), 0.4, post)
    _beam(p, (-7.4, -2.4, body + 0.4), (-5.4, -2.4, body + 5.4), 0.16, post)
    _beam(p, (-0.6, -2.4, body + 0.4), (-2.6, -2.4, body + 5.4), 0.16, post)
    _beam(p, (-7.4, -2.4, body + 1.2), (-0.6, -2.4, body + 1.2), 0.3, trim)
    cap = _block(p, -4, 1, 8, 8, 0.6, trim, None, body + 5.6)
    _pyramid(p, cap, 4.0 * m, iron)
    tip = _p(-4, 1, body + 10.4)
    _bar(p, tip, (tip[0], tip[1] - 1.6 * m), 0.12 * m, (50, 48, 50))
    shed = _block(p, -11, -9, 9, 6, 4.2, iron, None)
    s_left, s_front, s_right, _ = shed
    for a, b, tone in ((s_left, s_front, 0.82), (s_front, s_right, 0.64)):
        for k in range(1, 14):
            _bar(
                p,
                _on_wall(a, b, k / 14, 0),
                _on_wall(a, b, k / 14, 4.2 * m),
                0.05 * m,
                shade(iron, tone),
            )
    for t, reach in ((0.08, 2.8), (0.3, 1.8), (0.52, 3.2), (0.86, 2.2)):
        _bar(
            p,
            _on_wall(s_left, s_front, t, 0.4 * m),
            _on_wall(s_left, s_front, t, reach * m),
            0.3 * m,
            rust,
        )
    _bar(
        p,
        _on_wall(s_front, s_right, 0.5, 0.4 * m),
        _on_wall(s_front, s_right, 0.5, 2.6 * m),
        0.3 * m,
        shade(rust, 0.8),
    )
    _patch(p, s_left, s_front, (0.6, 0.85), (1.2 * m, 4.2 * m), DOORWAY)
    p.gable(shed, 1.4 * m, rust, iron)
    _roof_rows(p, shed, 1.4 * m, shade(rust, 1.1), (0.33, 0.66))
    _rails(p, (13, 8), (13, -16))
    _wagon(p, 13, 3, ore, (86, 78, 74))
    bin_stone = shade(stone, 0.92)
    for sv in (-6.4, -1.6):
        _block(p, 9.6, sv, 3.2, 1.2, 5.0, bin_stone, shade(bin_stone, 1.1))
    hopper = _block(p, 9.6, -4, 4.4, 6, 3.8, PLANKS, None, 5.0)
    _boards(p, hopper, 3.8, PLANKS)
    p.gable(hopper, 1.6 * m, iron, PLANKS)
    chute = _p(11.8, -4, 5.2)
    _bar(p, chute, _p(13.4, -4, 2.8), 1.0 * m, (70, 66, 66))
    _bar(p, _p(13.4, -4, 2.8), _p(13.4, -4, 2.6), 0.6 * m, ore)
    _wagon(p, 13, -6, ore, (122, 56, 42))
    _cone(p, 5, -9, 4.6, 4.2, ore, shade(ore, 1.25))
    _cone(p, 9, -13, 3.2, 2.8, shade(ore, 0.92))
    _cone(p, 1, -12, 2.4, 2.0, shade(ore, 1.1))
    _rails(p, (-0.2, -2.6), (5.6, -2.6), gauge=0.6, ballast=None)
    for k, tu in enumerate((1.2, 3.0, 4.8)):
        _tub(p, tu, -2.6, ore, glint=glint if k == 1 else None)
    return p.sprite()


def _land(
    p: Painter,
    area: tuple[float, float, float, float],
    height: Callable[[float, float], float],
    paint: Callable[[float, float, float, float], RGB | None],
    step: float = 1.0,
) -> None:
    """Land shaped by `height`, the metres up at every (u, v), over `area` (from u, to u, from
    v, to v), in cells `step` metres a side painted from the back forward: each in the colour
    `paint` gives it from its middle's u, v, height and steepness, lit by the sun (SUN), or left
    out where it gives none."""
    u0, u1, v0, v1 = area
    us = np.arange(u0, u1 + step / 2, step)
    vs = np.arange(v0, v1 + step / 2, step)
    z = np.array([[height(u, v) for v in vs] for u in us])
    cells = sorted(
        ((i, j) for i in range(len(us) - 1) for j in range(len(vs) - 1)),
        key=lambda c: us[c[0]] - vs[c[1]],
    )
    for i, j in cells:
        z00, z10, z01, z11 = z[i, j], z[i + 1, j], z[i, j + 1], z[i + 1, j + 1]
        du = (z10 - z00 + z11 - z01) / (2 * step)
        dv = (z01 - z00 + z11 - z10) / (2 * step)
        colour = paint(
            us[i] + step / 2, vs[j] + step / 2, (z00 + z10 + z01 + z11) / 4, math.hypot(du, dv)
        )
        if colour is None:
            continue
        normal = np.array((-du, -dv, 1.0))
        light = 0.62 + 0.42 * float(normal @ SUN) / float(np.linalg.norm(normal))
        tone = shade(colour, light)
        corners = [
            _p(us[i], vs[j], z00),
            _p(us[i + 1], vs[j], z10),
            _p(us[i + 1], vs[j + 1], z11),
            _p(us[i], vs[j + 1], z01),
        ]
        p.polygon(corners, tone, tone)


def _sunk(p: Painter, opening: list[Point], dig: Callable[[Painter], None]) -> None:
    """What `dig` draws below the ground, seen only through the `opening` in it: the hole's rim
    hides the rest."""
    k = fineness()
    width, height = p.image.width / k, p.image.height / k
    anchor = (p.anchor[0] / k, p.anchor[1] / k)
    below = Painter(width, height, anchor)
    dig(below)
    hole = Painter(width, height, anchor)
    hole.polygon(opening, (255, 255, 255))
    cover = ImageChops.multiply(below.image.getchannel("A"), hole.image.getchannel("A"))
    below.image.putalpha(cover)
    p.image.alpha_composite(below.image)


def _benched(d: float, radii: tuple[float, ...], lift: float, lean: float = 1.0) -> float:
    """How high the ground stands `d` metres from the floor of a pit dug in benches: level with
    the floor within `radii[0]`, then rising `lift` metres up a face `lean` metres wide at each of
    `radii` to a level ledge."""
    z = 0.0
    for r in radii:
        if d <= r:
            return z
        if d <= r + lean:
            t = (d - r) / lean
            return z + lift * t * t * (3 - 2 * t)
        z += lift
    return z


def _speckle(u: float, v: float) -> float:
    """A steady scatter of -1 to 1 over the ground, the same at the same (u, v) every time."""
    return math.sin(u * 12.9898 + v * 78.233) * 43758.5453 % 2 - 1


def _swell(u: float, v: float, size: float) -> float:
    """A smooth rise and fall of -1 to 1 over the ground, about `size` metres from crest to
    crest, the same at the same (u, v) every time."""
    k = math.pi / size
    return (
        math.sin(u * k + 1.3) * math.cos(v * k * 0.8 - 0.7) + math.sin((u + v) * k * 0.6 + 2.1)
    ) / 2


def _fir(p: Painter, u: float, v: float, z: float, high: float) -> None:
    """A fir `high` metres tall standing on the ground at (u, v, z): a short trunk under tiers of
    drooping boughs, dark, lit on the left."""
    m = METRE
    x, y = _p(u, v, z)
    dark, mid, light = PINES[0]
    _bar(p, (x, y), (x, y - 0.25 * high * m), 0.22 * m, TRUNK)
    tiers = 4
    for k in range(tiers):
        foot = y - (0.15 + 0.2 * k) * high * m
        tip = foot - (0.42 - 0.04 * k) * high * m
        half = (0.3 - 0.06 * k) * high * m
        p.polygon([(x - half, foot), (x, tip), (x + half, foot), (x, foot + 0.15 * half)], mid)
        p.polygon(
            [
                (x, tip),
                (x + half, foot),
                (x, foot + 0.15 * half),
                (x + 0.1 * half, foot - 0.3 * half),
            ],
            dark,
        )
        p.polygon([(x - half, foot), (x, tip), (x - 0.35 * half, foot - 0.2 * half)], light)


def _lantern(
    p: Painter,
    top: Top,
    rise: float,
    roof: RGB,
    wall: RGB,
    span: tuple[float, float] = (0.15, 0.85),
) -> None:
    """A raised lantern along the ridge of a gable roof `rise` high, louvred on its lit side."""
    m = METRE
    ridge_back, ridge_front = (
        ((top[0][0] + top[3][0]) / 2, (top[0][1] + top[3][1]) / 2 - rise),
        ((top[1][0] + top[2][0]) / 2, (top[1][1] + top[2][1]) / 2 - rise),
    )
    a = _on_wall(ridge_back, ridge_front, span[0], 0)
    b = _on_wall(ridge_back, ridge_front, span[1], 0)
    side = 0.9 * m
    lit = [(a[0] - side, a[1] + side / 2), (b[0] - side, b[1] + side / 2)]
    high = 1.3 * m
    p.polygon(
        [lit[0], lit[1], (lit[1][0], lit[1][1] - high), (lit[0][0], lit[0][1] - high)],
        shade(wall, 0.9),
    )
    for t in (0.2, 0.4, 0.6, 0.8):
        x, y = _on_wall(lit[0], lit[1], t, 0)
        _bar(p, (x, y - 0.2 * m), (x, y - high + 0.2 * m), 0.12 * m, (60, 58, 56))
    p.polygon(
        [
            b,
            (b[0] + side, b[1] - side / 2),
            (b[0] + side, b[1] - side / 2 - high),
            (b[0], b[1] - high),
        ],
        shade(wall, 0.7),
    )
    near = [(lit[0][0], lit[0][1] - high), (lit[1][0], lit[1][1] - high)]
    far = [(a[0] + side, a[1] - side / 2 - high), (b[0] + side, b[1] - side / 2 - high)]
    crest = [(a[0], a[1] - high - 0.7 * m), (b[0], b[1] - high - 0.7 * m)]
    p.polygon([far[0], far[1], crest[1], crest[0]], shade(roof, 0.7))
    p.polygon([near[0], near[1], crest[1], crest[0]], roof, shade(roof, 0.6))


def _shed(
    p: Painter,
    u: float,
    v: float,
    length: float,
    depth: float,
    high: float,
    rise: float,
    wall: RGB,
    roof: RGB,
    z: float = 0.0,
    windows: tuple[float, ...] = (),
    glass: RGB = WINDOW,
) -> Top:
    """A boarded shed `high` to its eaves under a gable roof `rise` high with a lantern on its
    ridge, standing `z` up, with windows at `windows` along its lit wall."""
    m = METRE
    top = _block(p, u, v, length, depth, high, wall, None, z)
    left, front, right, _ = top
    _boards(p, top, high, wall, 1.2)
    for a, b in ((left, front), (front, right)):
        _bar(p, _on_wall(a, b, 0, 0.2 * m), _on_wall(a, b, 1, 0.2 * m), 0.3 * m, shade(wall, 0.7))
    for t in windows:
        _patch(
            p, left, front, (t - 0.04, t + 0.04), (1.0 * m, (1.0 + min(2.2, high - 2)) * m), glass
        )
    p.gable(top, rise * m, roof, wall)
    _roof_rows(p, top, rise * m, roof, (0.33, 0.66))
    _lantern(p, top, rise * m, roof, wall)
    return top


def _ingots(p: Painter, u: float, v: float, metal: RGB, rows: tuple[int, ...]) -> None:
    """Cast bars of metal stacked crosswise on (u, v), `rows[0]` in the bottom layer and so on,
    lit on top."""
    for level, count in enumerate(rows):
        along = level % 2 == 0
        for k in range(count):
            off = (k - (count - 1) / 2) * 0.55
            du, dv = (0, off) if along else (off, 0)
            length, depth = (1.6, 0.42) if along else (0.42, 1.6)
            _block(p, u + du, v + dv, length, depth, 0.3, metal, shade(metal, 1.3), level * 0.3)


def copper_mine() -> Sprite:
    """A copper mine in a hillside: a pit cut in benches into the hill, orange with the ore and
    veined green with malachite, tubs of orange ore coming out of it on a tramway past heaps of
    orange and green ore, a concentrator stepping down the hill's flank in two boarded sheds on
    stone terraces under roofs gone green with verdigris, each with a louvred lantern, a brick
    smelter at its foot glowing at its furnace door under a lantern of new copper, a tall brick
    stack smoking yellow behind the hill, and bright copper bars stacked by a wagon of ore on the
    siding."""
    m = METRE
    p = Painter(50, 60, (25, 40))
    ore, malachite = (200, 112, 62), (70, 152, 112)
    verdigris = (96, 160, 136)
    copper = (218, 130, 70)
    rock, ledge = (180, 106, 64), (206, 158, 112)
    grass, scrub = (112, 136, 70), (82, 106, 54)
    stone = (170, 156, 130)
    wall = (212, 194, 158)
    hill_u, hill_v, pit_u, pit_v = -9.0, 10.0, -9.0, 6.0
    terraces = ((3, 13, 10, 8, 7.0), (8, 6, 10, 8, 3.6))

    def hill(u: float, v: float) -> float:
        d = math.hypot((u - hill_u) / 1.0, (v - hill_v) / 1.1) / 17
        return 15 * max(0.0, 1 - d * d) ** 1.1 * (1 + 0.08 * _swell(u, v, 9))

    def cut(u: float, v: float) -> float:
        a, b = u - pit_u, v - pit_v
        toward, across = (a - b) / math.sqrt(2), (a + b) / math.sqrt(2)
        if toward > 0:
            toward *= 0.35
        return _benched(math.hypot(toward, across), (4.0, 6.8, 9.6, 12.4), 3.2, 1.8)

    def height(u: float, v: float) -> float:
        z = min(hill(u, v), cut(u, v))
        for tu, tv, length, depth, level in terraces:
            if abs(u - tu) <= length / 2 and abs(v - tv) <= depth / 2:
                return level
        return z

    def paint(u: float, v: float, z: float, slope: float) -> RGB | None:
        if z < 0.2:
            return None
        for tu, tv, length, depth, _ in terraces:
            if abs(u - tu) <= length / 2 + 1 and abs(v - tv) <= depth / 2 + 1:
                return stone if slope > 1.2 else shade(stone, 1.1)
        grain = 1 + 0.025 * _speckle(u, v) + 0.04 * _swell(u, v, 3)
        if cut(u, v) < hill(u, v) - 0.3:
            if slope < 0.9:
                return shade(ledge, grain)
            return shade(malachite if _swell(u * 2.3, v * 1.9, 3) > 0.72 else rock, grain)
        if _swell(u * 1.3, v * 1.1, 3) > 0.6 and z > 2:
            return shade(scrub, grain)
        return shade(grass, grain)

    _yard(p, [(-4, -6, 16, 10), (10, -8, 12, 10), (-9, 2, 6, 6)], (184, 146, 104), 0.9)
    _stack(p, -4, 24, 40, shade(MILL_BRICK, 0.95), DRESSING, (214, 208, 170))
    _land(p, (-28, 14, -10, 30), height, paint, 0.25)
    for fu, fv in ((-22, 4), (-19, -1.5), (-14, 20), (-8, 22), (-23, 12)):
        _fir(p, fu, fv, height(fu, fv) - 0.2, 6.0 + 1.5 * _speckle(fu, fv))
    _rails(p, (-9, 3), (-9, -10), gauge=0.6, ballast=None)
    for k, tv in enumerate((-2.0, -4.0, -6.0)):
        _tub(
            p,
            -9,
            tv,
            ore if k != 1 else malachite,
            along_u=False,
            glint=(240, 176, 110) if k == 0 else None,
        )
    _cone(p, -4, -11, 3.6, 3.2, ore, shade(ore, 1.2))
    _cone(p, -10, -14, 2.8, 2.4, malachite, shade(malachite, 1.2))
    _cone(p, 0, -15, 2.2, 1.8, shade(ore, 0.9))
    _rails(p, (19, 6), (19, -16))
    _shed(p, 3, 13, 9, 7, 5.0, 2.4, wall, verdigris, 7.0, (0.25, 0.5, 0.75))
    _shed(p, 8, 6, 9, 7, 5.0, 2.4, wall, verdigris, 3.6, (0.25, 0.5, 0.75))
    smelter = _block(p, 12, -2, 9, 7, 6.0, shade(MILL_BRICK, 1.02), None)
    s_left, s_front, s_right, _ = smelter
    _plinth(p, smelter, 6 * m, MILL_BRICK, 0.6 * m)
    _stains(p, s_left, s_front, 6 * m, (0.1, 0.4, 0.9), shade(MILL_BRICK, 0.66), 1.4 * m)
    for t in (0.2, 0.8):
        _mill_window(p, s_left, s_front, 9 * m, t, 1.3 * m, (1.4 * m, 4.4 * m))
    p.polygon(_round_head(s_left, s_front, 9 * m, 0.5, 2.4 * m, 2.6 * m, 6 * m), DRESSING)
    p.polygon(_round_head(s_left, s_front, 9 * m, 0.5, 1.9 * m, 2.9 * m, 6 * m), (60, 34, 26))
    gx, gy = _on_wall(s_left, s_front, 0.5, 5.4 * m)
    p.blob(gx - 0.8 * m, gy - 1.2 * m, gx + 0.8 * m, gy + 0.3 * m, (240, 140, 50))
    p.blob(gx - 0.4 * m, gy - 0.8 * m, gx + 0.4 * m, gy, (255, 214, 120))
    _mill_window(
        p, s_front, s_right, 7 * m, 0.5, 1.3 * m, (1.4 * m, 4.4 * m), shade(MILL_GLASS, 0.85)
    )
    p.gable(smelter, 2.6 * m, verdigris, MILL_BRICK)
    _roof_rows(p, smelter, 2.6 * m, verdigris, (0.33, 0.66))
    _lantern(p, smelter, 2.6 * m, copper, MILL_BRICK, (0.3, 0.7))
    ex, ey = _on_wall(s_front, s_right, 0.5, -1.0 * m)
    _emblem(p, ex, ey, 1.4 * m, copper)
    _wagon(p, 19, -1, ore, (92, 80, 72))
    _ingots(p, 15, -10, copper, (3, 3, 2))
    _ingots(p, 12.5, -10.5, copper, (3, 2))
    return p.sprite()


def _trestle(
    p: Painter,
    points: list[tuple[float, float, float]],
    ground: Callable[[float, float], float],
    timber: RGB,
    every: float = 2.5,
) -> None:
    """Timber legs standing on the `ground` every `every` metres under a run through the (u, v,
    z) `points`, braced in crosses."""
    for a, b in itertools.pairwise(points):
        long = math.dist(a[:2], b[:2])
        count = max(1, round(long / every))
        for k in range(count + 1):
            t = k / count
            u, v, z = (a[i] + (b[i] - a[i]) * t for i in range(3))
            foot = ground(u, v)
            if z - foot < 0.3:
                continue
            _beam(p, (u, v - 0.5, foot), (u, v - 0.4, z), 0.2, timber)
            _beam(p, (u, v + 0.5, foot), (u, v + 0.4, z), 0.2, shade(timber, 0.8))
            if z - foot > 4:
                _beam(
                    p,
                    (u, v - 0.5, (foot + z) / 2),
                    (u, v + 0.5, (foot + z) / 2),
                    0.1,
                    shade(timber, 0.85),
                )


def _flume(p: Painter, points: list[tuple[float, float, float]], timber: RGB, water: RGB) -> None:
    """A boarded trough carried along the (u, v, z) `points`, full of running `water`."""
    for a, b in itertools.pairwise(points):
        _face(
            p,
            [
                (a[0], a[1] - 0.5, a[2]),
                (b[0], b[1] - 0.5, b[2]),
                (b[0], b[1] - 0.5, b[2] + 0.8),
                (a[0], a[1] - 0.5, a[2] + 0.8),
            ],
            timber,
        )
        _face(
            p,
            [
                (a[0], a[1] - 0.5, a[2] + 0.75),
                (b[0], b[1] - 0.5, b[2] + 0.75),
                (b[0], b[1] + 0.5, b[2] + 0.75),
                (a[0], a[1] + 0.5, a[2] + 0.75),
            ],
            water,
        )
        _beam(
            p,
            (a[0], a[1] - 0.5, a[2] + 0.8),
            (b[0], b[1] - 0.5, b[2] + 0.8),
            0.1,
            shade(timber, 1.25),
        )
        _beam(
            p,
            (a[0], a[1] + 0.5, a[2] + 0.8),
            (b[0], b[1] + 0.5, b[2] + 0.8),
            0.1,
            shade(timber, 0.8),
        )


def _stamps(p: Painter, top: Top, high: float, iron: RGB) -> None:
    """The open lit side of a stamp mill's battery: a dark bay showing a row of iron stamps, each
    a tall stem with a tappet on it and a shoe in the mortar box at its foot, lifted by cams on a
    shaft running across them."""
    m = METRE
    left, front, _, _ = top
    _patch(p, left, front, (0.1, 0.9), (0.8 * m, (high - 0.4) * m), (40, 34, 30))
    _patch(p, left, front, (0.12, 0.88), ((high - 2.0) * m, (high - 0.6) * m), shade(iron, 0.7))
    _bar(
        p,
        _on_wall(left, front, 0.1, 2.4 * m),
        _on_wall(left, front, 0.9, 2.4 * m),
        0.45 * m,
        shade(iron, 1.2),
    )
    for k in range(10):
        t = 0.15 + 0.7 * k / 9
        lift = 0.5 * m if k % 2 else 0.0
        _bar(
            p,
            _on_wall(left, front, t, 1.2 * m - lift),
            _on_wall(left, front, t, (high - 1.4) * m - lift),
            0.16 * m,
            (176, 176, 180),
        )
        x, y = _on_wall(left, front, t, 3.2 * m - lift)
        p.rect(x - 0.18 * m, y - 0.25 * m, x + 0.18 * m, y + 0.25 * m, iron)
    _bar(
        p,
        _on_wall(left, front, 0.1, 3.0 * m),
        _on_wall(left, front, 0.9, 3.0 * m),
        0.3 * m,
        (70, 66, 64),
    )


def gold_mine() -> Sprite:
    """A gold mine on a hillside: a timber headframe with one winding wheel on the crest beside
    its hoist house, tubs of ore on a trestle down to a stamp mill stepping down the slope in
    three boarded houses on timber cribbing, its battery open on its lit side to show a row of
    stamps, a flume of water on trestles running along the hill to a wheel that drives them,
    pale tailings spread in a fan below it, and an assay office with a false front and a gilt
    sign at the foot of the hill."""
    m = METRE
    p = Painter(50, 58, (25, 38))
    timber, weathered = (150, 116, 80), (132, 120, 104)
    shingle = (106, 92, 80)
    rust = (150, 96, 64)
    tailings = (214, 204, 176)
    water = (104, 156, 196)
    grass, scrub = (112, 136, 70), (82, 106, 54)
    gold = (236, 190, 64)
    honey = (198, 158, 74)
    levels = ((-8.0, 4.0, 7.0, 8.0, 9.5), (-1.5, 4.0, 7.0, 8.0, 5.5), (5.0, 4.0, 7.0, 8.0, 1.6))

    def hill(u: float, v: float) -> float:
        across = max(0.0, 1 - ((v - 5) / 11) ** 2)
        return 13 * math.exp(-(((u + 13) / 7.5) ** 2)) * across * (1 + 0.06 * _swell(u, v, 8))

    def height(u: float, v: float) -> float:
        for lu, lv, length, depth, level in levels:
            if abs(u - lu) <= length / 2 + 0.3 and abs(v - lv) <= depth / 2 + 0.3:
                return level
        return hill(u, v)

    def paint(u: float, v: float, z: float, slope: float) -> RGB | None:
        if z < 0.2:
            return None
        for lu, lv, length, depth, _ in levels:
            if abs(u - lu) <= length / 2 + 1 and abs(v - lv) <= depth / 2 + 1:
                return shade(timber, 0.8) if slope > 1.0 else (150, 134, 104)
        grain = 1 + 0.025 * _speckle(u, v) + 0.04 * _swell(u, v, 3)
        if _swell(u * 1.3, v * 1.1, 3) > 0.6 and z > 2:
            return shade(scrub, grain)
        return shade(grass, grain)

    _yard(p, [(-4, 0, 16, 12), (12, 0, 9, 8), (6, -10, 8, 6)], (170, 150, 112), 0.85)
    _yard(p, [(12, 2, 7, 8), (17, 0, 5, 6), (10, -2, 5, 4)], tailings, 0.95)
    for a, b in (((8.5, 5), (14, 6.5)), ((14, 6.5), (19, 3.5))):
        _beam(p, (*a, 0), (*b, 0), 0.5, shade(water, 1.1))
    _land(p, (-27, 10, -8, 18), height, paint, 0.35)
    for fu, fv in ((-19, 0), (-13, -5.5), (-21, 6), (-11, -4), (-18, 10.5)):
        _fir(p, fu, fv, hill(fu, fv) - 0.2, 6.5 + 1.5 * _speckle(fu, fv))
    _cone(p, 14, 1, 4.6, 2.6, tailings, shade(tailings, 0.9))
    _cone(p, 17, 4.5, 3.0, 1.7, shade(tailings, 0.96))
    crest = hill(-13, 5)
    hoist = _block(p, -15, -1, 6, 5, 3.6, weathered, None, hill(-15, -1) - 0.4)
    _boards(p, hoist, 3.6, weathered)
    p.gable(hoist, 1.8 * m, rust, weathered)
    _roof_rows(p, hoist, 1.8 * m, rust, (0.5,))
    head = (-13.0, 5.0, crest + 11)
    for su, sv in ((-1.6, 2), (1.6, 2), (-1.6, -2), (1.6, -2)):
        _beam(
            p,
            (head[0] + su * 1.4, head[1] + sv, crest),
            (head[0] + su * 0.5, head[1] + sv * 0.5, head[2]),
            0.4,
            shade(timber, 0.9 if sv > 0 else 1.0),
        )
    for z in (crest + 3.5, crest + 7):
        f = (z - crest) / 11
        bu, bv = 1.6 * 1.4 * (1 - f) + 0.8 * f, 2 * (1 - f) + f
        _beam(p, (head[0] - bu, head[1] - bv, z), (head[0] + bu, head[1] - bv, z), 0.25, timber)
        _beam(
            p,
            (head[0] + bu, head[1] - bv, z),
            (head[0] + bu, head[1] + bv, z),
            0.25,
            shade(timber, 0.8),
        )
    _beam(
        p,
        (head[0] - 2.2, head[1] - 2, crest),
        (head[0] + 0.8, head[1] - 1, head[2] - 4),
        0.18,
        timber,
    )
    _beam(
        p,
        (head[0] - 9, head[1], crest - 3),
        (head[0] - 0.6, head[1], head[2]),
        0.5,
        shade(timber, 0.85),
    )
    _sheave(p, head[0], head[1], head[2] + 1.6, 1.8, (90, 84, 78), spokes=6)
    _beam(p, (head[0] - 1.8, head[1], head[2] + 1.6), (-15, -1, hill(-15, -1) + 3), 0.08, ROPE)
    bin_top = levels[0][4]
    run = [(head[0] + 1.5, head[1] - 1, crest + 3.5), (-8.0, 2.5, bin_top + 6.2)]
    _trestle(p, run, hill, timber)
    _beam(p, (*run[0][:2], run[0][2]), (*run[1][:2], run[1][2]), 0.35, shade(timber, 0.9))
    _tub(p, -10.5, 3.2, (128, 116, 100), z=bin_top + 5.5, glint=gold)
    flume = [(-7.0, 13.5, hill(-7, 13.5) + 0.4), (2.6, 13.5, 9.6), (2.6, 9.0, 9.4)]
    _trestle(p, flume, hill, shade(timber, 0.85), 3.2)
    _flume(p, flume, timber, water)
    upper = _block(p, -8, 4, 6.5, 7.5, 5.5, weathered, None, bin_top)
    _boards(p, upper, 5.5, weathered)
    p.gable(upper, 2.2 * m, shingle, weathered)
    _roof_rows(p, upper, 2.2 * m, shingle, (0.33, 0.66))
    battery_z, high = levels[1][4], 8.0
    battery = _block(p, -1.5, 4, 6.5, 7.5, high, timber, None, battery_z)
    _, b_front, b_right, _ = battery
    _boards(p, battery, high, timber)
    _stamps(p, battery, high, (84, 80, 78))
    _framed(p, b_front, b_right, (0.3, 0.45), 2.0, 1.6, shade(timber, 1.25), shade(WINDOW, 0.8))
    _framed(p, b_front, b_right, (0.6, 0.75), 2.0, 1.6, shade(timber, 1.25), shade(WINDOW, 0.8))
    p.gable(battery, 2.6 * m, honey, timber)
    _roof_rows(p, battery, 2.6 * m, honey, (0.25, 0.5, 0.75))
    ex, ey = _on_wall(b_front, b_right, 0.5, -1.0 * m)
    _emblem(p, ex, ey, 1.5 * m, gold)
    _lantern(p, battery, 2.6 * m, honey, timber, (0.25, 0.75))
    wheel_u, wheel_v, wheel_z, r = 2.4, 5.6, battery_z + 4.2, 3.0
    wheel = (78, 62, 50)
    _beam(p, (2.6, 8.8, 10.0), (2.5, 7.4, wheel_z + r + 0.3), 0.6, shade(water, 1.2))
    for k in range(12):
        t = k * math.pi / 6
        _beam(
            p,
            (wheel_u, wheel_v, wheel_z),
            (wheel_u, wheel_v + r * math.cos(t), wheel_z + r * math.sin(t)),
            0.16,
            wheel,
        )
    turns = np.linspace(0, 2 * math.pi, 25)
    for a, b in itertools.pairwise(turns):
        _beam(
            p,
            (wheel_u, wheel_v + r * math.cos(a), wheel_z + r * math.sin(a)),
            (wheel_u, wheel_v + r * math.cos(b), wheel_z + r * math.sin(b)),
            0.45,
            wheel,
        )
        _beam(
            p,
            (wheel_u, wheel_v + (r - 0.15) * math.cos(a), wheel_z + (r - 0.15) * math.sin(a)),
            (wheel_u, wheel_v + (r - 0.15) * math.cos(b), wheel_z + (r - 0.15) * math.sin(b)),
            0.1,
            shade(timber, 1.25),
        )
    x, y = _p(wheel_u, wheel_v, wheel_z)
    p.ellipse(x - 0.4 * m, y - 0.4 * m, x + 0.4 * m, y + 0.4 * m, (60, 58, 58))
    sx, sy = _p(wheel_u, wheel_v + r * 0.7, wheel_z - r * 0.8)
    p.blob(sx - 0.7 * m, sy - 0.4 * m, sx + 0.7 * m, sy + 0.5 * m, (214, 232, 244))
    lower = _block(p, 5, 4, 6.5, 7.5, 4.4, weathered, None, levels[2][4])
    l_left, l_front, l_right, _ = lower
    _boards(p, lower, 4.4, weathered)
    for t in (0.2, 0.45, 0.7):
        _framed(p, l_left, l_front, (t - 0.06, t + 0.06), 1.2, 1.6, shade(weathered, 1.3))
    _patch(p, l_front, l_right, (0.3, 0.7), (1.4 * m, 4.4 * m), (52, 44, 40))
    p.gable(lower, 2.0 * m, rust, weathered)
    _roof_rows(p, lower, 2.0 * m, rust, (0.33, 0.66))
    office_wall = (178, 150, 110)
    office = _block(p, 7, -10, 6, 5, 3.6, office_wall, None)
    _boards(p, office, 3.6, office_wall)
    p.gable(office, 1.8 * m, shingle, office_wall)
    face = _block(p, 7, -12.6, 6.4, 0.3, 5.6, office_wall, shade(office_wall, 1.2))
    f_left, f_front, _, _ = face
    _boards(p, face, 5.6, office_wall)
    _patch(p, f_left, f_front, (0, 1), (0, 0.35 * m), shade(office_wall, 0.7))
    _patch(p, f_left, f_front, (0.1, 0.9), (0.7 * m, 1.9 * m), (40, 62, 50))
    _patch(p, f_left, f_front, (0.1, 0.9), (0.7 * m, 0.82 * m), gold)
    _patch(p, f_left, f_front, (0.1, 0.9), (1.78 * m, 1.9 * m), gold)
    for k in range(6):
        t = 0.2 + k * 0.12
        _bar(
            p,
            _on_wall(f_left, f_front, t, 1.05 * m),
            _on_wall(f_left, f_front, t + 0.07, 1.05 * m),
            0.32 * m,
            gold,
        )
    _framed(p, f_left, f_front, (0.14, 0.34), 2.7, 1.8, (232, 222, 196))
    _patch(p, f_left, f_front, (0.5, 0.7), (2.5 * m, 5.6 * m), DOOR)
    _framed(p, f_left, f_front, (0.76, 0.92), 2.7, 1.8, (232, 222, 196))
    _bar(
        p,
        _on_wall(f_left, f_front, 0, 5.3 * m),
        _on_wall(f_left, f_front, 1, 5.3 * m),
        0.3 * m,
        (232, 222, 196),
    )
    _rails(p, (8.5, 0.2), (8.5, -6), gauge=0.6, ballast=None)
    _tub(p, 8.5, -2.4, (128, 116, 100), along_u=False, glint=gold)
    return p.sprite()


def _staging(p: Painter, u: float, v: float, high: float, timber: RGB) -> None:
    """A braced timber staging `high` metres tall on a pit's rim at (u, v), for the haulage ropes
    to run from its top."""
    for su, sv in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        _beam(
            p,
            (u + su * 1.4, v + sv * 1.4, 0),
            (u + su * 0.7, v + sv * 0.7, high),
            0.3,
            shade(timber, 0.85 if sv > 0 else 1.0),
        )
    for z in (high * 0.35, high * 0.7):
        f = z / high
        w = 1.4 - 0.7 * f
        _beam(p, (u - w, v - w, z), (u + w, v - w, z), 0.18, timber)
        _beam(p, (u + w, v - w, z), (u + w, v + w, z), 0.18, shade(timber, 0.8))
        _beam(p, (u - w, v - w, z - high * 0.35), (u + w, v - w, z), 0.1, timber)
    _block(p, u, v, 2.2, 2.2, 0.4, timber, shade(timber, 1.2), high)


def diamond_mine() -> Sprite:
    """A diamond mine like the Big Hole: a deep round pit cut in benches through yellow ground
    into the blue kimberlite below, a pool at its floor, a steel headgear on its far rim and
    timber stagings round it, their haulage ropes strung down into the pit in a web, tubs of
    blue ground on the rim, the blue ground spread out in rows to weather on the floors before
    a whitewashed sorting house under a pale blue roof with a small gilt diamond on its gable,
    all glinting white here and there."""
    m = METRE
    p = Painter(50, 56, (25, 36))
    yellow, blue = (196, 172, 116), (108, 128, 150)
    blue_ledge, floor = (140, 158, 176), (80, 94, 112)
    dust = (190, 184, 164)
    pool = (64, 112, 132)
    ice = (206, 236, 250)
    pale_blue = (150, 174, 194)
    whitewash, roof = (240, 238, 230), (122, 148, 172)
    steel, timber = (92, 98, 108), (134, 104, 74)
    cu, cv = -3.0, 4.0
    radii, lift, lean = (3.0, 5.0, 7.0, 9.0, 11.0), 3.4, 1.4
    deep = lift * len(radii)
    rim = radii[-1] + lean

    def height(u: float, v: float) -> float:
        d = math.hypot(u - cu, v - cv)
        if d >= rim:
            return 0.0
        return _benched(d, radii, lift, lean) - deep

    def paint(u: float, v: float, z: float, slope: float) -> RGB | None:
        d = math.hypot(u - cu, v - cv)
        if d > rim:
            return None
        grain = 1 + 0.03 * _speckle(u, v) + 0.03 * _swell(u, v, 2.5)
        if d < radii[0]:
            return pool if d < radii[0] - 0.9 else floor
        if z > -lift - 0.2:
            return shade(yellow, grain)
        if slope < 0.9:
            return shade(blue_ledge, grain)
        if _speckle(u * 3.1, v * 2.7) > 0.965:
            return ice
        return shade(blue, grain)

    _yard(p, [(cu, cv, rim + 7, rim + 7), (3, -14, 10, 7)], dust, 0.85)
    ring = [
        _p(cu + (rim + 2.6) * math.cos(t), cv + (rim + 2.6) * math.sin(t))
        for t in np.linspace(0, 2 * math.pi, 49)
    ]
    p.polygon(ring, shade(dust, 0.96))
    opening = [
        _p(cu + rim * math.cos(t), cv + rim * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 49)
    ]
    p.polygon(opening, shade(dust, 1.04))
    hu, hv, high = cu - rim * 0.72 - 1.4, cv + rim * 0.72 + 1.4, 15.0
    stagings = (
        (cu - 1, cv + rim + 1.5, 7.0),
        (cu + rim + 1.5, cv + 1, 6.0),
        (cu - rim - 1.2, cv - 1.5, 6.5),
    )
    tops = [(su, sv, sz + 0.6) for su, sv, sz in stagings] + [(hu + 1.6, hv, high + 2.0)]
    ends = [
        (cu + 0.5, cv + 2.6, -deep),
        (cu - 1.5, cv + 1.0, -deep),
        (cu + 1.2, cv + 0.4, -deep),
        (cu - 0.4, cv + 1.6, -deep),
    ]
    ropes = []
    for top, end in zip(tops, ends, strict=False):
        t = top[2] / (top[2] - end[2])
        ropes.append((top, tuple(a + (b - a) * t for a, b in zip(top, end, strict=False)), end))

    def dig(below: Painter) -> None:
        _land(below, (cu - rim, cu + rim, cv - rim, cv + rim), height, paint, 0.2)
        for k, (_, ground, end) in enumerate(ropes):
            _beam(below, ground, end, 0.07, (40, 40, 44))
            if k % 2 == 0:
                at = tuple(a + (b - a) * 0.4 for a, b in zip(ground, end, strict=False))
                _tub(below, at[0], at[1], pale_blue, z=at[2] - 1.2, glint=ice)

    _sunk(p, opening, dig)
    for su, sv, sz in stagings:
        _staging(p, su, sv, sz, timber)
    for su, sv in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        _beam(
            p,
            (hu + su * 1.8, hv + sv * 1.8, 0),
            (hu + su * 0.6, hv + sv * 0.6, high),
            0.35,
            shade(steel, 0.8 if sv > 0 else 1.1),
        )
    for z in (4.0, 8.0, 12.0):
        w = 1.8 - 1.2 * z / high
        _beam(p, (hu - w, hv - w, z), (hu + w, hv - w, z), 0.2, steel)
        _beam(p, (hu + w, hv - w, z), (hu + w, hv + w, z), 0.2, shade(steel, 0.8))
        _beam(p, (hu - w, hv - w, z - 4), (hu + w * 0.9, hv - w, z), 0.12, shade(steel, 1.2))
    _beam(p, (hu - 7, hv + 4, 0), (hu - 0.5, hv + 0.4, high - 0.5), 0.4, shade(steel, 0.75))
    _block(p, hu, hv, 2.4, 2.4, 0.5, steel, shade(steel, 1.3), high)
    _sheave(p, hu, hv, high + 2.0, 1.6, (70, 74, 82), spokes=6)
    for top, ground, _ in ropes:
        _beam(p, top, ground, 0.07, (40, 40, 44))
    _ground(p, [(-3, -19), (6, -19), (6, -12), (-3, -12)], shade(dust, 0.9))
    for row in range(3):
        for k in range(4):
            _cone(
                p,
                -1.4 + k * 2.4,
                -13.6 - row * 2.2,
                1.05,
                0.8,
                shade(pale_blue, 0.92 + 0.05 * ((row + k) % 2)),
                power=1.8,
            )
    _cone(p, -14, -2, 3.0, 2.6, pale_blue, shade(pale_blue, 1.2))
    _cone(p, -10, -8, 2.2, 1.8, shade(pale_blue, 0.94))
    for gu, gv in ((-14.5, -1.0), (-13.0, -2.6), (-9.6, -8.4), (-1.0, -14.4), (3.6, -16.0)):
        x, y = _p(gu, gv, 1.0)
        _glint(p, x, y, 0.35 * m)
    _rails(p, (-8, -10), (-2, -10), gauge=0.6, ballast=None)
    for k, tu in enumerate((-6.6, -4.6)):
        _tub(p, tu, -10, pale_blue, glint=ice if k == 0 else None)
    house = _block(p, 15, -4, 11, 6, 5.0, whitewash, None)
    h_left, h_front, h_right, _ = house
    _plinth(p, house, 5 * m, (190, 186, 176), 0.6 * m)
    for t in (0.14, 0.32, 0.5, 0.68, 0.86):
        _mill_window(p, h_left, h_front, 11 * m, t, 1.2 * m, (1.2 * m, 3.8 * m), (88, 124, 150))
    _mill_window(p, h_front, h_right, 6 * m, 0.5, 1.3 * m, (1.2 * m, 3.8 * m), (76, 108, 132))
    p.gable(house, 2.4 * m, roof, whitewash)
    _roof_rows(p, house, 2.4 * m, roof, (0.33, 0.66))
    _lantern(p, house, 2.4 * m, roof, whitewash, (0.2, 0.8))
    gx, gy = _on_wall(h_front, h_right, 0.5, -1.0 * m)
    p.ellipse(gx - 0.9 * m, gy - 0.9 * m, gx + 0.9 * m, gy + 0.9 * m, GOLD)
    p.ellipse(gx - 0.7 * m, gy - 0.7 * m, gx + 0.7 * m, gy + 0.7 * m, (36, 52, 92))
    _diamond(p, gx, gy - 0.42 * m, 1.0 * m)
    return p.sprite()
