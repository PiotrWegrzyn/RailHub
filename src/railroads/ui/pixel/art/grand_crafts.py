"""Big and mega works of the crafts grown into industry, each drawn as a building of its own:
yellow-brick cotton mills of Łódź with dyed cloth drying, carpet halls hung with rugs, tanneries
of drying lofts and tan pits, shoe factories under giant boots, furniture works in lumber yards,
paper mills turning waterwheels, newspaper press houses under big signs, glowing forges of tool
works and great engine halls of machine works under gantry cranes."""

import math
from collections.abc import Callable
from itertools import pairwise

from railroads.ui.pixel.art.commerce import _lamp_post
from railroads.ui.pixel.art.crafts import (
    BUCKLE,
    GILT,
    SADDLE,
    SADDLE_TAN,
    SADDLER_BAY,
    SADDLER_BRICK,
    SADDLER_COBBLES,
    SADDLER_GLASS,
    SADDLER_GREEN,
    SADDLER_IRON,
    SADDLER_LACQUER,
    SADDLER_LAMPLIT,
    SADDLER_PAVING,
    SADDLER_SIGN,
    SADDLER_SLATE,
    SADDLER_STONE,
    SADDLER_STUCCO,
    SADDLER_TILE,
    WEAVE_CREAM,
    WEAVE_INDIGO,
    WEAVE_MADDER,
    WEAVE_PLASTER,
    WEAVE_ROOF,
    WEAVE_RUGS,
    WEAVE_SAFFRON,
    WEAVE_TIMBER,
    WEAVE_YARD,
    _broken_roof,
    _hip,
    _saddle_arcade,
    _saddle_balconette,
    _saddle_bench_window,
    _saddle_bridles,
    _saddle_carriage,
    _saddle_collar,
    _saddle_cupola,
    _saddle_display,
    _saddle_head,
    _saddle_horse,
    _saddle_sash,
    _saddle_stable_door,
    _saddle_stack,
    _saddle_trestle,
    _saddle_trough,
    _weave_belfry,
    _weave_chimney,
    _weave_hanging,
    _weave_loom,
    _weave_rug,
    _weave_skeins,
    _weave_vat,
)
from railroads.ui.pixel.art.industry import (
    BUNDLE,
    GLAZING,
    JOINER_WALL,
    OXBLOOD,
    SHOE_BRICK,
    SHOE_LAMPLIT,
    SHOE_LEATHER,
    SHOE_SLATE,
    SHOE_TRIM,
    SMOKE,
    SOIL,
    TAN_HIDES,
    TAN_LIME,
    TAN_LIQUORS,
    TAN_MUD,
    TAN_OAK,
    TAN_OCHRE,
    TAN_ROOF,
    TAN_SLAT,
    _anvil,
    _bed,
    _bundle,
    _chair,
    _chimney,
    _crates,
    _door,
    _drying_cloth,
    _footing,
    _forge_stack,
    _gear,
    _lengths,
    _locomotive,
    _north_lights,
    _oil,
    _on_roof,
    _plank_walls,
    _puddle,
    _ring,
    _roof_rows,
    _round_head,
    _sawtooth,
    _shoe_boot,
    _shoe_cartons,
    _shoe_stack,
    _shoe_window,
    _skylight,
    _smoke,
    _smoking_chimney,
    _spill,
    _spoked_wheel,
    _spool,
    _stains,
    _tan_bark,
    _tan_cart,
    _tan_frame,
    _tan_louvres,
    _tan_pit,
    _track,
    _window,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    GLASS,
    RGB,
    VERDIGRIS,
    WINDOW,
    Painter,
    Point,
    Sprite,
    Top,
    shade,
)
from railroads.ui.pixel.art.parts import (
    BEAST,
    METRE,
    _band,
    _bands,
    _bar,
    _barrel,
    _flag,
    _gable_across,
    _heap,
    _iso,
    _on_wall,
    _patch,
    _plinth,
    _ridge,
    _tiles,
    _turned,
    _vault,
    _windows,
)

MILL_WALL: RGB = (206, 164, 92)
MILL_ROOF: RGB = (80, 76, 84)
MILL_DOOR: RGB = (60, 44, 36)
CLOTH_BALE: RGB = (232, 220, 196)

GOLD: RGB = (232, 190, 80)

PAVING: RGB = (176, 170, 158)

BOARD: RGB = (226, 190, 134)
JOINER_ROOF: RGB = (98, 74, 58)
SAWDUST: RGB = (222, 196, 140)
LEG: RGB = (110, 80, 56)

PAPER_WALL: RGB = (192, 168, 136)
PAPER_ROOF: RGB = (100, 86, 80)
WATER: RGB = (84, 140, 196)
BANK: RGB = (96, 88, 62)
WHEEL: RGB = (98, 72, 50)
FOAM: RGB = (226, 238, 244)
ROLL: RGB = (240, 240, 234)
LOG: RGB = (196, 160, 112)
STEAM: RGB = (232, 232, 236)

PRESS_WALL: RGB = (220, 204, 172)
PRESS_ROOF: RGB = (110, 106, 116)
PAGE: RGB = (242, 240, 230)
INK: RGB = (40, 40, 48)
PRINT: RGB = (150, 150, 160)
MASTHEAD: RGB = (170, 40, 40)

FORGE_WALL: RGB = (92, 84, 82)
FORGE_ROOF: RGB = (60, 60, 68)
FORGE_GROUND: RGB = (62, 58, 56)
GLOW: RGB = (238, 132, 40)
HOT: RGB = (255, 208, 112)
COAL: RGB = (48, 46, 50)
SIGNBOARD: RGB = (214, 202, 176)
BAR: RGB = (120, 122, 130)
HANDLE: RGB = (150, 112, 74)

ENGINE_ROOF: RGB = (82, 84, 96)
BRASS: RGB = (222, 180, 74)
CRANE: RGB = (214, 168, 40)
GRAVEL: RGB = (140, 134, 120)
CASTING: RGB = (88, 90, 98)
RAIL: RGB = (96, 92, 90)
SLEEPER: RGB = (110, 84, 60)
DOORWAY_PRESS: RGB = (54, 46, 48)
METAL_GREEN: RGB = (60, 100, 80)


def _rails(p: Painter, a: Point, b: Point, ties: int = 6) -> None:
    """A short rail siding on sleepers across the ground from `a` to `b`."""
    for k in range(ties + 1):
        x, y = _on_wall(a, b, k / ties, 0)
        p.line([(x - 1, y - 0.5), (x + 1, y + 0.5)], SLEEPER)
    for off in (-0.6, 0.6):
        p.line([(a[0] + off, a[1] + off / 2), (b[0] + off, b[1] + off / 2)], RAIL)


def _wagon(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A low open goods wagon on (x, y), its side lit."""
    p.rect(x - 3.5, y - 3, x + 3.5, y - 1, colour)
    p.line([(x - 3, y - 2.2), (x + 3, y - 2.2)], shade(colour, 1.25))
    for wx in (x - 2, x + 2):
        p.ellipse(wx - 0.9, y - 1.5, wx + 0.9, y + 0.3, (40, 38, 40))


def _cart(p: Painter, x: float, y: float) -> None:
    """A two-wheeled wooden handcart standing on (x, y), its shafts down on the ground."""
    p.rect(x - 3, y - 3, x + 3, y - 1.5, (130, 92, 60))
    p.line([(x + 3, y - 2), (x + 6, y)], (110, 80, 56))
    p.ellipse(x - 1.6, y - 2.4, x + 0.6, y + 0.2, (70, 54, 40))


def _mill_block(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, storeys: int
) -> Top:
    """A yellow-brick mill of `storeys` floors of small arched windows over a dark plinth stained
    by the damp, with a cornice and string courses, its flat roof behind a parapet."""
    top = p.hall(x, y, length, depth, height, MILL_WALL, MILL_ROOF)
    left, front, right, _ = top
    lintel = shade(MILL_WALL, 1.3)
    _plinth(p, top, height, MILL_WALL)
    _stains(p, left, front, height, (0.04, 0.3, 0.56, 0.82), shade(MILL_WALL, 0.6), 2)
    _stains(p, front, right, height, (0.2, 0.8), shade(MILL_WALL, 0.5), 2)
    _bands(p, left, front, right, 1, shade(MILL_WALL, 1.2), shade(MILL_WALL, 0.95))
    p.line([left, front, right], shade(MILL_ROOF, 1.3))
    floor = (height - 2.5) / storeys
    drops = tuple(2.8 + floor * f for f in range(storeys))
    n = max(3, round(length / 3.2))
    spans = tuple((k + 0.5) / n for k in range(n))
    _windows(p, left, front, spans, drops, 2, arch=lintel)
    m = max(2, round(depth / 3.4))
    side = tuple((k + 0.5) / m for k in range(m))
    _windows(p, front, right, side, drops, 2, shade(WINDOW, 0.75), lintel)
    for f in range(1, storeys):
        _bands(p, left, front, right, drops[f] - 1.3, shade(MILL_WALL, 0.88), shade(MILL_WALL, 0.7))
    return top


def _mill_tower(p: Painter, x: float, y: float, half: float, tall: float, dome: bool) -> None:
    """A square yellow-brick stair tower standing on (x, y) with narrow arched windows, a
    battlemented parapet and either a green copper dome or a dark pointed cap."""
    top = p.box(x, y, half, tall, MILL_WALL, MILL_ROOF)
    left, front, right, _ = top
    lintel = shade(MILL_WALL, 1.3)
    _plinth(p, top, tall, MILL_WALL)
    drops = tuple(3 + 4.2 * k for k in range(int((tall - 4) / 4.2)))
    _windows(p, left, front, (0.5,), drops, 2, arch=lintel)
    _windows(p, front, right, (0.5,), drops, 2, shade(WINDOW, 0.75), lintel)
    _bands(p, left, front, right, 1.2, shade(MILL_WALL, 0.8), shade(MILL_WALL, 0.65))
    for a, b in ((left, front), (front, right)):
        for t in (0.0, 0.5, 1.0):
            mx, my = _on_wall(a, b, t, 0)
            p.rect(mx - 0.6, my - 1.4, mx + 0.6, my, shade(MILL_WALL, 0.95))
    cx, cy = x, y - tall
    if dome:
        _turned(
            p,
            cx,
            cy,
            [(0, half * 0.8), (half * 0.6, half * 0.75), (half * 1.1, half * 0.4), (half * 1.4, 0)],
            VERDIGRIS,
        )
        p.line([(cx, cy - half * 1.4), (cx, cy - half * 1.4 - 2.5)], GOLD)
    else:
        apex = (cx, cy - half * 1.6)
        p.polygon([left, front, apex], MILL_ROOF, shade(MILL_ROOF, 0.6))
        p.polygon([front, right, apex], shade(MILL_ROOF, 0.75), shade(MILL_ROOF, 0.5))
        p.line([apex, (apex[0], apex[1] - 2)], (70, 66, 62))


def _weaving_shed(p: Painter, x: float, y: float, length: float, depth: float, teeth: int) -> Top:
    """A low yellow-brick weaving shed under a sawtooth roof glazed along its teeth."""
    top = p.hall(x, y, length, depth, 5, MILL_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 5, MILL_WALL)
    _stains(p, front, right, 5, (0.15, 0.9), shade(MILL_WALL, 0.5), 1.2)
    n = max(3, round(length / 3))
    _windows(
        p,
        left,
        front,
        tuple((k + 0.5) / n for k in range(n)),
        (2.5,),
        1,
        arch=shade(MILL_WALL, 1.3),
    )
    _sawtooth(p, top, teeth, 4, (104, 100, 108), GLAZING, MILL_WALL)
    return top


def _bales(p: Painter, x: float, y: float, count: int) -> None:
    """Bales of grey cloth piled on the ground from (x, y)."""
    for k in range(count):
        bx, by = x + 2.6 * (k % 3), y + 1.3 * (k % 3) - 2 * (k // 3)
        p.box(bx, by, 1.6, 1.8, CLOTH_BALE, shade(CLOTH_BALE, 1.06))
        p.line([(bx - 1.6, by - 0.9), (bx, by - 0.1)], shade(CLOTH_BALE, 0.7))


def big_textile_mill() -> Sprite:
    """A bigger yellow-brick cotton mill of Łódź: a long block of five storeys of small arched
    windows behind a stair tower with a dark pointed cap, a long sawtooth weaving shed in front of
    it, a tall banded chimney smoking, two racks of cloth in every dye hung out to dry and bales of
    cloth by the door."""
    p = Painter(52, 74, (25, 56))
    mill_foot, shed_foot = _footing(-4, -3, 22, 9), _footing(11, 6, 12, 9)
    _bed(p, mill_foot)
    _bed(p, shed_foot)
    p.shadow(20, 5, reach=4)
    _track(p, shed_foot[1], shed_foot[2], (0.35, 0.7), 3)
    cap = _chimney(p, 15, -3, -42, shade(MILL_WALL, 0.92))
    _ring(p, 15, -3 + (cap + 3) * 0.45, 1.5, shade(MILL_WALL, 0.7))
    _smoke(p, 16, cap - 3, puffs=3)
    top = _mill_block(p, -4, -3, 22, 9, 20, 5)
    left, front, _, _ = top
    p.line([_on_wall(left, front, 0.66, 1.5), _on_wall(left, front, 0.66, 19.6)], (70, 64, 66))
    tx, ty = _on_wall(mill_foot[0], mill_foot[1], 0.42, 1.6)
    _mill_tower(p, tx, ty, 3.2, 25, dome=False)
    _patch(p, mill_foot[0], mill_foot[1], (0.36, 0.48), (-3, 0), MILL_DOOR)
    _weaving_shed(p, 11, 6, 12, 9, 4)
    _spool(p, *_on_wall(left, front, 0.16, 4))
    _drying_cloth(p, (-24, 2), (-14, 7), 8)
    _drying_cloth(p, (-17, 8.5), (-6, 14), 7)
    _bales(p, 2, 12, 3)
    return p.sprite()


def mega_textile_mill() -> Sprite:
    """A Łódź mill palace: a great yellow-brick mill of six storeys of small arched windows with
    battlemented corner towers under green copper domes and a taller gate tower before its middle,
    long sawtooth weaving sheds beside it, two tall banded chimneys smoking, racks of cloth in
    every dye hung out to dry in rows before it and bales of cloth stacked by the door."""
    p = Painter(64, 82, (31, 62))
    mill_foot, shed_foot = _footing(-6, -5, 28, 10), _footing(18, 5, 12, 11)
    _bed(p, mill_foot)
    _bed(p, shed_foot)
    p.shadow(25, 6, reach=4)
    _track(p, shed_foot[1], shed_foot[2], (0.3, 0.6), 3)
    for x, foot, tall in ((14, -10, 38), (22, -6, 40)):
        cap = _chimney(p, x, foot, foot - tall, shade(MILL_WALL, 0.92))
        _ring(p, x, foot - (foot - cap) * 0.4, 1.5, shade(MILL_WALL, 0.7))
        _smoke(p, x + 1, cap - 3, puffs=3)
    left_corner = mill_foot[0]
    _mill_tower(p, left_corner[0] + 1, left_corner[1] + 0.2, 3.4, 29, dome=True)
    top = _mill_block(p, -6, -5, 28, 10, 24, 6)
    left, front, _, _ = top
    for t in (0.3, 0.72):
        p.line([_on_wall(left, front, t, 1.5), _on_wall(left, front, t, 23.6)], (70, 64, 66))
    _mill_tower(p, mill_foot[1][0], mill_foot[1][1] - 0.2, 3.4, 29, dome=True)
    gx, gy = _on_wall(mill_foot[0], mill_foot[1], 0.5, 2)
    _mill_tower(p, gx, gy, 4, 33, dome=False)
    _patch(p, mill_foot[0], mill_foot[1], (0.44, 0.56), (-2, 2), MILL_DOOR)
    _weaving_shed(p, 18, 5, 12, 11, 4)
    _spool(p, *_on_wall(left, front, 0.22, 5))
    _drying_cloth(p, (-30, -2), (-20, 3), 8)
    _drying_cloth(p, (-26, 5), (-14, 11), 8)
    _drying_cloth(p, (-18, 12), (-6, 18), 7)
    _bales(p, 5, 14, 5)
    return p.sprite()


def _pavement(p: Painter, foot: Top, joints: int) -> None:
    _bed(p, foot, PAVING, 2.4)
    _bed(p, foot, SOIL, 0.8)
    for k in range(1, joints):
        t = k / joints
        p.line(
            [_on_wall(foot[0], foot[1], t, 0.8), _on_wall(foot[0], foot[1], t, 2.2)],
            shade(PAVING, 0.8),
        )


DYE_BRICK: RGB = (176, 96, 70)


def _weave_line(
    p: Painter,
    at: Callable[[float, float], Point],
    u: tuple[float, float],
    v: float,
    tall: float,
    rugs: tuple[tuple[float, float, int], ...],
) -> None:
    """A line strung along the lit walls from `u[0]` to `u[1]` at `v`, `tall` metres up on two
    posts, where `at` places a spot on the ground, with carpets thrown over it to air, each
    `(start, width, colourway)` along it, hanging a metre short of the ground."""
    m = METRE

    def up(uu: float, z: float) -> Point:
        x, y = at(uu, v)
        return x, y - z * m

    post = (108, 80, 54)
    for uu in u:
        _bar(p, at(uu, v), up(uu, tall + 0.2), 0.14 * m, post)
    _bar(p, up(u[0] - 0.1, tall), up(u[1] + 0.1, tall), 0.06 * m, (90, 86, 80))
    for start, wide, k in rugs:
        _weave_hanging(
            p, *up(start, tall), wide * m, (tall - 0.6) * m, WEAVE_RUGS[k % len(WEAVE_RUGS)]
        )


def _weave_gallery(
    p: Painter,
    up: Callable[[float, float, float], Point],
    u: tuple[float, float],
    floor: float,
    rugs: tuple[tuple[float, float, int], ...],
) -> None:
    """A timber gallery 1.3 m deep along a lit wall from `u[0]` to `u[1]`, its floor `floor`
    metres up, where `up(u, v, z)` places a point, on struts, with a slatted balustrade and
    carpets thrown over its rail, each `(start, width, colourway)` along it."""
    m = METRE
    rail = floor + 1.0
    mid = (u[0] + u[1]) / 2
    p.hall(
        *up(mid, -0.65, floor - 0.25),
        (u[1] - u[0]) * m,
        1.3 * m,
        0.25 * m,
        WEAVE_TIMBER,
        shade(WEAVE_TIMBER, 1.25),
    )
    span = u[1] - u[0]
    for k in range(round(span / 3.5) + 1):
        uu = u[0] + 0.3 + (span - 0.6) * k / round(span / 3.5)
        _bar(
            p,
            up(uu, -0.05, floor - 1.0),
            up(uu, -1.2, floor - 0.2),
            0.14 * m,
            shade(WEAVE_TIMBER, 0.8),
        )
    for k in range(round(span * 2) + 1):
        uu = u[0] + 0.2 + (span - 0.4) * k / round(span * 2)
        _bar(p, up(uu, -1.25, floor), up(uu, -1.25, rail), 0.07 * m, shade(WEAVE_TIMBER, 0.9))
    _bar(
        p,
        up(u[0] + 0.1, -1.25, rail),
        up(u[1] - 0.1, -1.25, rail),
        0.16 * m,
        shade(WEAVE_TIMBER, 1.2),
    )
    for start, wide, k in rugs:
        _weave_hanging(
            p, *up(start, -1.3, rail), wide * m, 2.0 * m, WEAVE_RUGS[k % len(WEAVE_RUGS)]
        )


def _weave_sash(
    p: Painter,
    a: Point,
    b: Point,
    length: float,
    t: float,
    span: tuple[float, float],
    tone: float = 1.0,
) -> None:
    """A window 1.4 m wide in an indigo frame about `t` along the wall under the edge from `a`
    to `b`, which is `length` units long, from `span[0]` to `span[1]` down it, with a cream
    glazing cross."""
    m = METRE
    half = 0.7 * m / length
    _patch(
        p,
        a,
        b,
        (t - half - 0.15 * m / length, t + half + 0.15 * m / length),
        (span[0] - 0.15 * m, span[1] + 0.1 * m),
        shade(WEAVE_INDIGO, tone),
    )
    _patch(p, a, b, (t - half, t + half), span, shade(WINDOW, 1.05 * tone))
    _bar(
        p,
        _on_wall(a, b, t, span[0]),
        _on_wall(a, b, t, span[1]),
        0.08 * m,
        shade(WEAVE_CREAM, tone),
    )
    d = span[0] + (span[1] - span[0]) * 0.35
    _bar(
        p,
        _on_wall(a, b, t - half, d),
        _on_wall(a, b, t + half, d),
        0.08 * m,
        shade(WEAVE_CREAM, tone),
    )


def _weave_vent(p: Painter, top: Top, rise: float, along: float, long: float, roof: RGB) -> None:
    """A louvred vent `long` long astride a ridge at `along` its length, steam of the dye
    coppers rising from it."""
    m = METRE
    back, front = _ridge(top, rise)
    x, y = _on_wall(back, front, along, 0)
    vent = p.hall(x, y + 0.5 * m, long, 1.4 * m, 1.3 * m, (46, 38, 36), None)
    for a, b, tone in ((vent[0], vent[1], 1.0), (vent[1], vent[2], 0.78)):
        for r in range(3):
            d = (0.2 + 0.32 * r) * m
            _patch(p, a, b, (0, 1), (d, d + 0.16 * m), shade((196, 180, 160), tone))
    p.gable(vent, 0.7 * m, shade(roof, 0.9), (46, 38, 36))
    p.soft(
        [
            (x - 1.2 * m, y - 4.6 * m, x + 1.6 * m, y - 2.6 * m),
            (x - 0.2 * m, y - 6.8 * m, x + 2.8 * m, y - 4.4 * m),
        ],
        (246, 244, 240),
        0.8,
        0.4,
    )


def _weave_glazing(
    p: Painter,
    eave: tuple[Point, Point],
    end: tuple[Point, Point],
    up_slope: tuple[float, float],
    along: tuple[float, float],
    lights: int,
) -> None:
    """A strip of roof lights over a loom hall on the roof slope from the eave `eave[0]` up to
    the ridge `eave[1]` at its near end and from `end[0]` to `end[1]` at its far end, from
    `up_slope[0]` to `up_slope[1]` of the way up and `along[0]` to `along[1]` of the way along,
    in `lights` panes between iron bars."""
    m = METRE

    def at(t: float, f: float) -> Point:
        return _on_wall(_on_wall(eave[0], end[0], t, 0), _on_wall(eave[1], end[1], t, 0), f, 0)

    (f0, f1), (t0, t1) = up_slope, along
    p.polygon([at(t0, f0), at(t1, f0), at(t1, f1), at(t0, f1)], (156, 188, 204), (90, 104, 116))
    for k in range(1, lights):
        t = t0 + (t1 - t0) * k / lights
        _bar(p, at(t, f0), at(t, f1), 0.1 * m, (80, 88, 96))


def big_carpet_weaver() -> Sprite:
    """A carpet manufactory round a courtyard hung with carpets. At its back the manufactory of
    warm cream plaster, 28 by 9 m and three storeys high under a madder-red roof with a bell
    turret and two chimneys: looms showing through big indigo-framed windows below, a timber
    gallery hung with carpets along its first floor and small windows above. Along the
    courtyard's side the long loom hall, its gable to the street, its looms lit through tall
    windows. Across the courtyard three lines of carpets air in madder, indigo, saffron and cream,
    and at its corner the brick dye house steams from its louvred ridge, with vats of dye and
    skeins of wool drying before it."""
    m = METRE
    p = Painter(52, 36, (26, 22))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 18.5) * m, (v - 7.5) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon([at(8, -7), at(39, -7), at(39, 13), at(8, 13)], WEAVE_YARD)
    for (u, v, r), colour in zip(
        ((28.6, -1.2, 1.4), (31.4, -1.6, 1.6), (34.4, -1.0, 1.2), (24, 6, 1.0)),
        (WEAVE_MADDER, WEAVE_INDIGO, WEAVE_SAFFRON, WEAVE_INDIGO),
        strict=True,
    ):
        x, y = at(u, v)
        p.blob(x - r * m, y - r * m / 2, x + r * m, y + r * m / 2, shade(colour, 0.8), 90)
    p.shadow(21, 6, 3)
    wall, roof = WEAVE_PLASTER, WEAVE_ROOF
    length, depth, high, rise = 28 * m, 9 * m, 11.6 * m, 4.0 * m
    foot = _footing(*at(22, 17.5), length, depth)
    _bed(p, foot, margin=0.6)
    top = p.hall(*at(22, 17.5), length, depth, high, wall, None)
    left, front, right, _ = top
    _plinth(p, top, high, wall, 0.6 * m)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        for z in (4.4, 8.2):
            _patch(
                p,
                a,
                b,
                (0, 1),
                (high - (z + 0.25) * m, high - z * m),
                shade(WEAVE_SAFFRON, 0.92 * tone),
            )
        _patch(p, a, b, (0, 1), (0, 0.35 * m), shade(wall, 0.82 * tone))
    for k in range(6):
        t = (k + 0.5) / 6
        _weave_loom(
            p, left, front, length, t, 2.8 * m, (high - 3.4 * m, high - 0.7 * m), WEAVE_RUGS[k % 4]
        )
        _weave_sash(p, left, front, length, t, (high - 7.7 * m, high - 5.2 * m))
        _weave_sash(p, left, front, length, t, (high - 11.1 * m, high - 9.1 * m))
    for t in (0.25, 0.75):
        for z0, z1 in ((3.4, 1.0), (7.6, 5.2), (11.0, 9.0)):
            _weave_sash(p, front, right, 9 * m, t, (high - z0 * m, high - z1 * m), 0.78)
    p.gable(top, rise, roof, wall)
    _tiles(p, top, rise, roof, (0.2, 0.4, 0.6, 0.8))
    _weave_chimney(p, top, rise, 0.15, 1.8 * m)
    _weave_chimney(p, top, rise, 0.85, 1.8 * m)
    back, ridge_front = _ridge(top, rise)
    _weave_belfry(p, *_on_wall(back, ridge_front, 0.5, 0), wall, roof)
    _weave_gallery(
        p,
        up,
        (9.0, 35.0),
        4.6,
        ((10.0, 2.6, 0), (14.4, 3.0, 1), (19.2, 2.6, 2), (24.0, 3.0, 3), (29.0, 2.8, 1)),
    )
    hall_wall = shade(wall, 0.96)
    loom = p.hall(*at(4, 7.5), 8 * m, 29 * m, 6 * m, hall_wall, None)
    l_left, l_front, l_right, _ = loom
    _plinth(p, loom, 6 * m, hall_wall, 0.6 * m)
    for k in range(7):
        t = (k + 0.5) / 7
        _patch(
            p,
            l_front,
            l_right,
            (t - 0.045, t + 0.045),
            (1.0 * m, 5.0 * m),
            shade(WEAVE_INDIGO, 0.78),
        )
        _patch(
            p,
            l_front,
            l_right,
            (t - 0.036, t + 0.036),
            (1.2 * m, 4.9 * m),
            shade((236, 206, 140), 0.78),
        )
        _patch(
            p,
            l_front,
            l_right,
            (t - 0.036, t + 0.036),
            (3.2 * m, 4.6 * m),
            shade(WEAVE_RUGS[k % 4][0], 0.8),
        )
        _patch(
            p,
            l_front,
            l_right,
            (t - 0.036, t + 0.036),
            (3.2 * m, 3.45 * m),
            shade(WEAVE_RUGS[k % 4][1], 0.8),
        )
        for s in (t - 0.03, t + 0.03):
            _bar(
                p,
                _on_wall(l_front, l_right, s, 1.2 * m),
                _on_wall(l_front, l_right, s, 4.9 * m),
                0.12 * m,
                shade(WEAVE_TIMBER, 0.78),
            )
    _patch(p, l_left, l_front, (0.3, 0.7), (1.6 * m, 6 * m), shade(WEAVE_INDIGO, 0.9))
    _patch(p, l_left, l_front, (0.34, 0.66), (1.8 * m, 6 * m), (60, 48, 44))
    _gable_across(p, loom, 3.4 * m, roof, hall_wall)
    near = _on_wall(l_left, l_front, 0.5, -3.4 * m)
    far = _on_wall(loom[3], l_right, 0.5, -3.4 * m)
    _weave_glazing(p, (l_front, near), (l_right, far), (0.3, 0.75), (0.04, 0.96), 9)
    _weave_line(
        p,
        at,
        (9.5, 30.5),
        9.5,
        3.0,
        ((10.2, 3.2, 0), (14.2, 3.4, 2), (18.4, 3.0, 1), (22.2, 3.6, 3), (26.6, 3.2, 2)),
    )
    _weave_line(
        p,
        at,
        (9.5, 30.5),
        5.0,
        3.0,
        ((10.0, 3.6, 1), (14.4, 3.2, 3), (18.6, 3.6, 0), (23.0, 3.0, 2), (26.8, 3.2, 1)),
    )
    _weave_line(p, at, (9.5, 22.5), 0.5, 3.0, ((10.2, 3.4, 2), (14.4, 3.0, 0), (18.2, 3.6, 3)))
    for (u, v), liquor in zip(
        ((21.6, -3.0), (24.2, -3.3), (26.8, -2.9)),
        ((150, 30, 40), (36, 44, 104), (226, 160, 40)),
        strict=True,
    ):
        _weave_vat(p, *at(u, v), liquor)
    for u in (13.0, 21.4):
        _bar(p, at(u, -5.4), up(u, -5.4, 2.6), 0.14 * m, (108, 80, 54))
    _bar(p, up(12.8, -5.4, 2.55), up(21.6, -5.4, 2.55), 0.1 * m, (130, 96, 64))
    _weave_skeins(
        p,
        up(13.2, -5.4, 2.55),
        up(21.2, -5.4, 2.55),
        16,
        1.2 * m,
        (WEAVE_MADDER, WEAVE_SAFFRON, WEAVE_INDIGO, (196, 70, 52)),
    )
    dye = p.hall(*at(35.5, -3.5), 6 * m, 5 * m, 3.8 * m, DYE_BRICK, None)
    d_left, d_front, d_right, _ = dye
    _plinth(p, dye, 3.8 * m, DYE_BRICK, 0.4 * m)
    _patch(p, d_left, d_front, (0.28, 0.72), (0.9 * m, 3.8 * m), (50, 40, 38))
    _patch(p, d_left, d_front, (0.34, 0.66), (2.9 * m, 3.8 * m), (230, 130, 50))
    _patch(p, d_front, d_right, (0.3, 0.7), (1.0 * m, 2.6 * m), shade(WINDOW, 0.8))
    p.gable(dye, 1.8 * m, (110, 104, 108), DYE_BRICK)
    _tiles(p, dye, 1.8 * m, (110, 104, 108))
    _weave_vent(p, dye, 1.8 * m, 0.5, 3.4 * m, (110, 104, 108))
    return p.sprite()


def _weave_roll(p: Painter, x: float, y: float, long: float, colours: tuple[RGB, RGB, RGB]) -> None:
    """A carpet rolled up and lying along the lit walls on (x, y), `long` units long and half a
    metre thick: its back showing along the roll and its far end a spiral of border and field."""
    m = METRE
    r = 0.25 * m
    field, border, _ = colours
    end = _iso(x, y, long, 0)
    p.polygon(
        [(x, y), end, (end[0], end[1] - 2 * r), (x, y - 2 * r)],
        shade(border, 0.8),
        shade(border, 0.5),
    )
    _bar(p, (x, y - 1.5 * r), (end[0], end[1] - 1.5 * r), 0.1 * m, shade(border, 1.15))
    p.ellipse(end[0] - r * 0.8, end[1] - 2 * r, end[0] + r * 0.8, end[1], border)
    p.blob(end[0] - r * 0.5, end[1] - 1.6 * r, end[0] + r * 0.5, end[1] - 0.4 * r, field)
    p.blob(end[0] - r * 0.2, end[1] - 1.2 * r, end[0] + r * 0.2, end[1] - 0.8 * r, border)


def mega_carpet_weaver() -> Sprite:
    """A grand carpet house, the palace of a carpet manufactory. Its front, 34 by 12 m, is of
    cream plaster between madder pilasters under hipped madder-red roofs: three storeys of
    indigo-framed windows, a carpet thrown over the balcony of every window of the first floor,
    and in the middle a pavilion of four storeys under a tall roof crowned by the bell turret,
    down whose front hangs one great showpiece carpet, bordered and medallioned, two floors
    high. Behind it the loom hall under a glazed north-light roof; beside it the brick dye works
    steaming from its louvred ridges over a yard of dye vats, racks of red, indigo and saffron
    wool and carpets airing on a line; and on the paved forecourt between lamp posts a wagon
    loaded with rolled carpets."""
    m = METRE
    p = Painter(66, 42, (33, 27))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 26.5) * m, (v - 3.5) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    wall, roof = WEAVE_PLASTER, WEAVE_ROOF
    paving = (188, 180, 164)
    p.polygon([at(-1, -7.5), at(35.5, -7.5), at(35.5, 0), at(-1, 0)], paving)
    for k in range(1, 6):
        v = -7.5 + 7.5 * k / 6
        _bar(p, at(-1, v), at(35.5, v), 0.05 * m, shade(paving, 0.88))
    p.polygon([at(35.5, -8.5), at(51, -8.5), at(51, 12), at(35.5, 12)], WEAVE_YARD)
    for (u, v, r), colour in zip(
        ((39.4, -3.4, 1.4), (42.2, -3.8, 1.6), (45.0, -3.3, 1.3), (47.8, -3.6, 1.2)),
        (WEAVE_MADDER, WEAVE_INDIGO, WEAVE_SAFFRON, WEAVE_MADDER),
        strict=True,
    ):
        x, y = at(u, v)
        p.blob(x - r * m, y - r * m / 2, x + r * m, y + r * m / 2, shade(colour, 0.8), 90)
    p.shadow(25, 7, 3)
    hall = p.hall(*at(34, 20.5), 20 * m, 11 * m, 6.5 * m, DYE_BRICK, None)
    _plinth(p, hall, 6.5 * m, DYE_BRICK, 0.5 * m)
    for k in range(4):
        t = (k + 0.5) / 4
        _patch(p, hall[1], hall[2], (t - 0.08, t + 0.08), (1.2 * m, 5.0 * m), shade(WINDOW, 0.8))
    _north_lights(p, hall, 6, 2.6 * m, (96, 90, 98), DYE_BRICK)
    dye = p.hall(*at(43, 5.5), 10 * m, 11 * m, 5 * m, DYE_BRICK, None)
    d_left, d_front, d_right, _ = dye
    _plinth(p, dye, 5 * m, DYE_BRICK, 0.5 * m)
    for t in (0.2, 0.5, 0.8):
        p.polygon(
            _round_head(d_left, d_front, 10 * m, t, 1.8 * m, 1.2 * m, 3.8 * m), (236, 206, 150)
        )
        p.polygon(_round_head(d_left, d_front, 10 * m, t, 1.4 * m, 1.4 * m, 3.7 * m), (60, 46, 42))
        x, y = _on_wall(d_left, d_front, t, 3.7 * m)
        p.blob(x - 0.5 * m, y - 0.6 * m, x + 0.5 * m, y, (230, 130, 50))
    for t in (0.25, 0.75):
        _patch(p, d_front, d_right, (t - 0.08, t + 0.08), (1.2 * m, 3.6 * m), shade(WINDOW, 0.8))
    p.gable(dye, 2.4 * m, (110, 104, 108), DYE_BRICK)
    _tiles(p, dye, 2.4 * m, (110, 104, 108))
    _weave_vent(p, dye, 2.4 * m, 0.28, 3.0 * m, (110, 104, 108))
    _weave_vent(p, dye, 2.4 * m, 0.72, 3.0 * m, (110, 104, 108))
    for v0 in (-1.2,):
        for u in (37.2, 49.2):
            _bar(p, at(u, v0), up(u, v0, 2.7), 0.14 * m, (108, 80, 54))
        _bar(p, up(37.0, v0, 2.65), up(49.4, v0, 2.65), 0.1 * m, (130, 96, 64))
        _weave_skeins(
            p,
            up(37.4, v0, 2.65),
            up(49.0, v0, 2.65),
            24,
            1.3 * m,
            (WEAVE_MADDER, WEAVE_SAFFRON, WEAVE_INDIGO, (196, 70, 52), (60, 120, 90)),
        )
    for (u, v), liquor in zip(
        ((39.4, -4.0), (42.2, -4.4), (45.0, -3.9), (47.8, -4.2)),
        ((150, 30, 40), (36, 44, 104), (226, 160, 40), (150, 30, 40)),
        strict=True,
    ):
        _weave_vat(p, *at(u, v), liquor)
    _weave_line(p, at, (37.0, 50.0), -7.2, 3.0, ((37.6, 3.6, 0), (41.6, 3.2, 2), (45.4, 3.8, 1)))
    high = 12.4 * m

    def wing(u0: float, u1: float) -> None:
        length, depth = (u1 - u0) * m, 12 * m
        bays = round((u1 - u0) / 3.4)
        _bed(p, _footing(*at((u0 + u1) / 2, 6), length, depth), margin=0.6)
        top = p.hall(*at((u0 + u1) / 2, 6), length, depth, high, wall, None)
        left, front, right, _ = top
        _plinth(p, top, high, wall, 0.7 * m)
        for a, b, size, count, tone in (
            (left, front, length, bays, 1.0),
            (front, right, depth, 3, 0.78),
        ):
            for z in (4.4, 8.6):
                _patch(
                    p,
                    a,
                    b,
                    (0, 1),
                    (high - (z + 0.3) * m, high - z * m),
                    shade(WEAVE_SAFFRON, 0.92 * tone),
                )
            _patch(p, a, b, (0, 1), (0, 0.6 * m), shade(WEAVE_CREAM, 0.95 * tone))
            for k in range(count + 1):
                t = k / count
                _bar(
                    p,
                    _on_wall(a, b, t, 0.6 * m),
                    _on_wall(a, b, t, high - 0.7 * m),
                    0.5 * m,
                    shade(WEAVE_MADDER, 0.9 * tone),
                )
            for k in range(count):
                t = (k + 0.5) / count
                p.polygon(
                    _round_head(a, b, size, t, 1.8 * m, high - 3.6 * m, high - 0.9 * m),
                    shade(WEAVE_INDIGO, tone),
                )
                p.polygon(
                    _round_head(a, b, size, t, 1.4 * m, high - 3.4 * m, high - 1.0 * m),
                    shade(WINDOW, 1.05 * tone),
                )
                _weave_sash(p, a, b, size, t, (high - 7.9 * m, high - 5.2 * m), tone)
                _weave_sash(p, a, b, size, t, (high - 11.6 * m, high - 9.6 * m), tone)
        for k in range(bays):
            t = (k + 0.5) / bays
            x0, y0 = _on_wall(left, front, t - 1.2 * m / length, high - 5.0 * m)
            x1, y1 = _on_wall(left, front, t + 1.2 * m / length, high - 5.0 * m)
            p.polygon(
                [(x0, y0), (x1, y1), _iso(x1, y1, 0, -0.8 * m), _iso(x0, y0, 0, -0.8 * m)],
                shade(WEAVE_CREAM, 0.92),
            )
            hx, hy = _iso(x0, y0 - 1.0 * m, 0.1 * m, -0.8 * m)
            _bar(p, (hx - 0.05 * m, hy - 0.05 * m), _iso(hx, hy, 2.3 * m, 0), 0.1 * m, (52, 46, 44))
            _weave_hanging(
                p, hx, hy, 2.2 * m, 1.7 * m, WEAVE_RUGS[(k + round(u0)) % len(WEAVE_RUGS)]
            )
        _hip(p, top, 4.4 * m, 0.3, roof, (0.33, 0.66))
        rb, rf = _ridge(top, 4.4 * m)
        x, y = _on_wall(rb, rf, 0.5, 0.8 * m)
        p.hall(x, y + 0.4 * m, 1.0 * m, 1.0 * m, 2.4 * m, wall, None)
        p.hall(x, y - 2.0 * m, 1.4 * m, 1.4 * m, 0.3 * m, shade(WEAVE_MADDER, 0.9), (50, 40, 40))

    wing(0, 13.4)
    plen, tall = 9.2 * m, 16.4 * m
    _bed(p, _footing(*at(18, 4.5), plen, 15 * m), margin=0.6)
    pav = p.hall(*at(18, 4.5), plen, 15 * m, tall, wall, None)
    p_left, p_front, p_right, _ = pav
    _plinth(p, pav, tall, wall, 0.7 * m)
    for a, b, tone in ((p_left, p_front, 1.0), (p_front, p_right, 0.78)):
        for t in (0, 1):
            _bar(
                p,
                _on_wall(a, b, t, 0.6 * m),
                _on_wall(a, b, t, tall),
                0.7 * m,
                shade(WEAVE_MADDER, 0.9 * tone),
            )
        _patch(p, a, b, (0, 1), (0, 0.6 * m), shade(WEAVE_CREAM, 0.95 * tone))
        _patch(p, a, b, (0, 1), (tall - 4.7 * m, tall - 4.4 * m), shade(WEAVE_SAFFRON, 0.92 * tone))
    for t in (0.1, 0.45, 0.8):
        _weave_sash(p, p_front, p_right, 15 * m, t, (0.9 * m, 2.9 * m), 0.78)
    p.polygon(
        _round_head(p_left, p_front, plen, 0.5, 3.4 * m, tall - 4.2 * m, tall),
        shade(WEAVE_MADDER, 0.9),
    )
    p.polygon(
        _round_head(p_left, p_front, plen, 0.5, 2.8 * m, tall - 3.9 * m, tall), (236, 206, 140)
    )
    _patch(p, p_left, p_front, (0.37, 0.63), (tall - 2.6 * m, tall), WEAVE_INDIGO)
    _bar(
        p,
        _on_wall(p_left, p_front, 0.5, tall - 2.6 * m),
        _on_wall(p_left, p_front, 0.5, tall),
        0.08 * m,
        GILT,
    )
    rug_w, rug_l = 6.6 * m, 9.6 * m

    def face(s: float, d: float) -> Point:
        return _on_wall(p_left, p_front, (plen - rug_w) / 2 / plen + s / plen, 1.0 * m + d)

    _bar(p, face(-0.3 * m, -0.1 * m), face(rug_w + 0.3 * m, -0.1 * m), 0.2 * m, GILT)
    _weave_rug(p, face, rug_w, rug_l, (WEAVE_MADDER, WEAVE_INDIGO, WEAVE_SAFFRON))
    _hip(p, pav, 5.0 * m, 0.42, roof, (0.33, 0.66))
    rb, rf = _ridge(pav, 5.0 * m)
    _weave_belfry(p, *_on_wall(rb, rf, 0.5, 0), wall, roof)
    wing(22.6, 34)
    for u in (1.0, 33.0):
        _lamp_post(p, *at(u, -2.6), 4.5, 2)
    wx, wy = at(24.0, -5.6)
    bed = p.hall(wx, wy - 1.6 * m, 5.4 * m, 2.4 * m, 0.4 * m, (128, 92, 60), (150, 112, 74))
    for k in range(3):
        for j in range(2 - (k // 2)):
            rx, ry = _iso(bed[0][0], bed[0][1], 0.3 * m, (0.4 + 0.75 * k) * m)
            _weave_roll(p, rx, ry - j * 0.5 * m, 4.8 * m, WEAVE_RUGS[(k + j) % 4])
    for t in (0.2, 0.8):
        _spoked_wheel(p, *_on_wall(bed[0], bed[1], t, 0.6 * m + 0.4 * m), 0.8 * m)
    _saddle_horse(p, *_iso(wx, wy, -5.0 * m, 0), SADDLER_BAY, -1, None)
    return p.sprite()


def _leather_loft_block(
    p: Painter,
    x: float,
    y: float,
    length: float,
    depth: float,
    low: float,
    lofts: int,
    rise: float,
    vents: int,
    hipped: bool = False,
) -> Top:
    """A tannery block standing on (x, y), `length` by `depth` units: a beamhouse `low` high of
    ochre-washed walls, jettied over it `lofts` storeys of louvred drying loft with hides
    glimpsed through the slats, under a steep chestnut-tiled roof `rise` high, hipped if
    `hipped`, with `vents` louvred vents on its ridge. Returns the corners of the beamhouse's
    top."""
    m = METRE
    _bed(p, _footing(x, y, length, depth), margin=0.6)
    beam = p.hall(x, y, length, depth, low, TAN_OCHRE, None)
    left, front, _, _ = beam
    _plinth(p, beam, low, TAN_OCHRE, 0.6 * m)
    _stains(p, left, front, low, (0.04, 0.2, 0.5, 0.7, 0.95), shade(TAN_OCHRE, 0.72), 1.4 * m)
    storey = 2.7 * m
    loft = p.hall(x, y - low, length + 0.8 * m, depth + 0.8 * m, storey * lofts, TAN_OAK, None)
    l_left, l_front, l_right, _ = loft
    for a, b, size, tone in ((l_left, l_front, length, 1.0), (l_front, l_right, depth, 0.78)):
        bays = max(2, round(size / (2.6 * m)))
        for k in range(lofts):
            _tan_louvres(p, a, b, bays, (0.2 * m + storey * k, storey * (k + 1) - 0.1 * m), tone)
    if hipped:
        _hip(p, loft, rise, 0.36, TAN_ROOF, (0.25, 0.5, 0.75))
    else:
        p.gable(loft, rise, TAN_ROOF, TAN_OAK)
        _tiles(p, loft, rise, TAN_ROOF, (0.2, 0.4, 0.6, 0.8))
    back, front_ridge = _ridge(loft, rise)
    for k in range(vents):
        vx, vy = _on_wall(back, front_ridge, (k + 0.5) / vents, 0)
        vent = p.hall(
            vx, vy + 0.5 * m, min(4.5 * m, length / vents * 0.6), 2 * m, 1.9 * m, (40, 32, 28), None
        )
        v_left, v_front, v_right, _ = vent
        for a, b, tone in ((v_left, v_front, 1.0), (v_front, v_right, 0.78)):
            for r in range(4):
                d = (0.25 + 0.3 * r) * m
                _patch(p, a, b, (0, 1), (d, d + 0.17 * m), shade(TAN_SLAT, tone * 1.1))
        p.gable(vent, 1.0 * m, shade(TAN_ROOF, 0.9), (40, 32, 28))
    return beam


def _leather_doors(
    p: Painter,
    a: Point,
    b: Point,
    length: float,
    spots: tuple[float, ...],
    low: float,
    windows: tuple[float, ...],
) -> None:
    """Arched beamhouse doors at `spots` and barred windows at `windows` along the wall under the
    edge from `a` to `b`, which is `length` units long, its foot `low` down it."""
    m = METRE
    for t in spots:
        p.polygon(_round_head(a, b, length, t, 3 * m, low - 3.5 * m, low), shade(TAN_OCHRE, 0.78))
        p.polygon(_round_head(a, b, length, t, 2.6 * m, low - 3.3 * m, low), (52, 40, 34))
        _patch(
            p,
            a,
            b,
            (t - 1.3 * m / length, t - 0.02 * m / length),
            (low - 2.3 * m, low),
            (110, 74, 48),
        )
    for t in windows:
        _patch(
            p,
            a,
            b,
            (t - 0.6 * m / length, t + 0.6 * m / length),
            (low - 3.2 * m, low - 1.6 * m),
            (52, 44, 40),
        )
        for s in (t - 0.25 * m / length, t + 0.25 * m / length):
            _bar(
                p,
                _on_wall(a, b, s, low - 3.2 * m),
                _on_wall(a, b, s, low - 1.6 * m),
                0.07 * m,
                (40, 34, 32),
            )


def _leather_gin(p: Painter, x: float, y: float) -> None:
    """A horse-gin bark mill standing on (x, y): a round house 6 m across of ochre-washed walls
    under a conical tiled roof with a little louvred cap, a dark door, and outside it a sweep
    turning its shaft with a horse walking round at its end, bark heaped by its door."""
    m = METRE
    r = 3.0 * m
    _turned(p, x, y, [(0, r), (3.0 * m, r)], TAN_OCHRE)
    _turned(
        p,
        x,
        y - 3.0 * m,
        [(0, r * 1.18), (0.2 * m, r * 1.18), (3.6 * m, 0.45 * m), (3.6 * m, 0)],
        TAN_ROOF,
    )
    _turned(
        p, x, y - 6.4 * m, [(0, 0.5 * m), (0.8 * m, 0.5 * m)], (40, 32, 28), shade(TAN_ROOF, 0.9)
    )
    p.polygon([(x - 0.7 * m, y - 7.2 * m), (x + 0.7 * m, y - 7.2 * m), (x, y - 8.0 * m)], TAN_ROOF)
    p.polygon(
        [
            (x - 2.2 * m, y + 0.6 * m),
            (x - 1.0 * m, y + 1.2 * m),
            (x - 1.0 * m, y - 1.3 * m),
            (x - 2.2 * m, y - 1.9 * m),
        ],
        (52, 40, 34),
    )
    hx, hy = _iso(x, y, 1.0 * m, -4.6 * m)
    _bar(p, (x + 0.6 * r, y - 1.2 * m), (hx, hy - 1.4 * m), 0.18 * m, shade(TAN_OAK, 1.2))
    _saddle_horse(p, hx - 0.4 * m, hy, (96, 70, 52), -1, None)


def big_tannery() -> Sprite:
    """A tanyard of two drying-loft blocks. The long tannery, 22 by 10 m, has a beamhouse of
    ochre-washed walls under two floors of louvred oak drying lofts with hides hanging behind the
    slats; beside it a taller block of three loft floors. Before them long rows of tanning pits
    kerbed in oak, bark liquor from pale to near black and the first ones milky with lime, a cart
    of raw hides at the beamhouse door, hides stretched to dry in a row of oak frames, and a
    round horse-gin bark mill under a conical roof grinding oak bark from the heaps beside it."""
    m = METRE
    p = Painter(58, 40, (29, 25))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 22) * m, (v - 2) * m)

    p.polygon([at(-1, -11), at(37, -11), at(37, 6), at(-1, 6)], TAN_MUD)
    lx, ly = at(2.0, -4.0)
    p.soft([(lx - 4 * m, ly - 2 * m, lx + 4 * m, ly + 2 * m)], TAN_LIME, 0.35, 0.6)
    p.shadow(20, 6, 3)
    _tan_bark(p, *at(41, 17), 3.0 * m, 3.0 * m)
    b1 = _leather_loft_block(p, *at(11, 12), 22 * m, 10 * m, 4.2 * m, 2, 4.2 * m, 2)
    _leather_doors(p, b1[0], b1[1], 22 * m, (0.12, 0.6), 4.2 * m, (0.28, 0.4, 0.76, 0.88))
    _patch(p, b1[1], b1[2], (0.35, 0.65), (1.0 * m, 4.2 * m), (52, 40, 34))
    b2 = _leather_loft_block(p, *at(29.5, 14), 11 * m, 8 * m, 4.0 * m, 3, 3.6 * m, 1)
    _leather_doors(p, b2[0], b2[1], 11 * m, (0.3,), 4.0 * m, (0.7,))
    _tan_bark(p, *at(41, 9.5), 2.4 * m, 2.2 * m)
    _leather_gin(p, *at(41, 4))
    _tan_bark(p, *at(45.5, 2.5), 2.2 * m, 2.0 * m)
    _tan_cart(p, *at(3.6, 5.2))
    for k in range(3):
        for j in range(8):
            u0 = 1.0 + 3.1 * j
            liquor = TAN_LIME if j == 0 else TAN_LIQUORS[(j + k) % 3]
            v0 = 2.6 - 2.7 * k
            _tan_pit(p, at, (u0, u0 + 2.3), (v0, v0 + 1.7), liquor)
    for k in range(6):
        u, v = 27.0 + 1.6 * k, -9.4 + 2.6 * k
        _tan_frame(p, *at(u, v), 3.4 * m, 3.6 * m, TAN_HIDES[k % 3])
    return p.sprite()


def _leather_wheel(p: Painter, x: float, y: float, r: float) -> None:
    """An undershot waterwheel `r` units across its middle turning on an axle at (x, y), seen
    edge on along the shaded walls: an oak rim and spokes, paddles round it and white water
    breaking under it."""
    m = METRE

    def rim(a: float, k: float = 1.0) -> Point:
        return x + math.cos(a) * r * k, y - math.cos(a) * r * k / 2 - math.sin(a) * r * k

    oak = shade(TAN_OAK, 1.15)
    turns = [2 * math.pi * k / 16 for k in range(17)]
    for a0, a1 in pairwise(turns):
        _bar(p, rim(a0), rim(a1), 0.28 * m, oak)
    for k in range(8):
        a = 2 * math.pi * k / 8
        _bar(p, (x, y), rim(a, 0.95), 0.14 * m, shade(oak, 0.85))
    for k in range(16):
        a = 2 * math.pi * k / 16 + 0.1
        _bar(p, rim(a, 0.92), rim(a, 1.12), 0.3 * m, shade(oak, 0.7))
    p.blob(x - 0.3 * m, y - 0.3 * m, x + 0.3 * m, y + 0.3 * m, (50, 46, 44))
    wx, wy = rim(-math.pi / 2)
    p.soft([(wx - 1.6 * m, wy - 0.6 * m, wx + 1.6 * m, wy + 0.6 * m)], (236, 244, 248), 0.8, 0.3)


def mega_tannery() -> Sprite:
    """A great leather works. In the middle stands its drying tower, a beamhouse of
    ochre-washed walls under four storeys of louvred oak drying lofts with hides hanging behind
    the slats, crowned by louvred vents; behind it the long beamhouse under a louvred lantern,
    and beside it the two-loft tan house. Before them three long rows of tanning pits kerbed in
    oak, the first milky with lime beside its white heaps, carts of raw hides waiting, two
    staggered rows of hides stretched in oak frames, and on the mill race the bark mill, its
    undershot waterwheel turning, with heaps of oak bark round it."""
    m = METRE
    p = Painter(66, 46, (33, 30))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 27) * m, (v - 2) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon([at(-1, -8), at(48, -8), at(48, 8), at(-1, 8)], TAN_MUD)
    water, stone = (78, 120, 150), (150, 142, 128)
    p.polygon([at(52, -6), at(55.5, -6), at(55.5, 12), at(52, 12)], stone)
    p.polygon([at(52.5, -6), at(55, -6), at(55, 12), at(52.5, 12)], water)
    for v in (-4, 2, 8):
        _bar(p, at(53.1, v), at(53.1, v + 2.4), 0.08 * m, shade(water, 1.3))
    lx, ly = at(3.0, -5.0)
    p.soft([(lx - 4 * m, ly - 2 * m, lx + 4 * m, ly + 2 * m)], TAN_LIME, 0.4, 0.6)
    p.shadow(24, 7, 3)
    beam = p.hall(*at(14, 20), 28 * m, 10 * m, 5 * m, TAN_OCHRE, None)
    _plinth(p, beam, 5 * m, TAN_OCHRE, 0.6 * m)
    p.gable(beam, 3.2 * m, TAN_ROOF, TAN_OCHRE)
    _tiles(p, beam, 3.2 * m, TAN_ROOF, (0.3, 0.6))
    rb, rf = _ridge(beam, 3.2 * m)
    lantern = p.hall(*_on_wall(rb, rf, 0.5, 0.5 * m), 22 * m, 2.2 * m, 1.5 * m, (40, 32, 28), None)
    for a, b, tone in ((lantern[0], lantern[1], 1.0), (lantern[1], lantern[2], 0.78)):
        for r in range(3):
            d = (0.2 + 0.35 * r) * m
            _patch(p, a, b, (0, 1), (d, d + 0.18 * m), shade(TAN_SLAT, tone * 1.1))
    p.gable(lantern, 0.8 * m, shade(TAN_ROOF, 0.9), (40, 32, 28))
    house = _leather_loft_block(p, *at(10, 9), 18 * m, 9 * m, 4.2 * m, 2, 4.0 * m, 2)
    _leather_doors(p, house[0], house[1], 18 * m, (0.15, 0.65), 4.2 * m, (0.35, 0.47, 0.85))
    tower = _leather_loft_block(p, *at(29, 10), 13 * m, 11 * m, 4.6 * m, 4, 6.0 * m, 1, hipped=True)
    _leather_doors(p, tower[0], tower[1], 13 * m, (0.5,), 4.6 * m, (0.18, 0.82))
    _patch(p, tower[1], tower[2], (0.35, 0.65), (1.0 * m, 4.6 * m), (52, 40, 34))
    for (u, v), half in (((39, 16), 3.2), ((42.5, 12.5), 2.6), ((45.5, 15.5), 2.4)):
        _tan_bark(p, *at(u, v), half * m, half * m)
    mill = p.hall(*at(47.5, 4), 8 * m, 8 * m, 6 * m, TAN_OCHRE, None)
    m_left, m_front, m_right, _ = mill
    _plinth(p, mill, 6 * m, TAN_OCHRE, 0.6 * m)
    _stains(p, m_front, m_right, 6 * m, (0.2, 0.5, 0.8), shade(TAN_OCHRE, 0.6), 1.6 * m)
    p.polygon(_round_head(m_left, m_front, 8 * m, 0.4, 2.6 * m, 2.8 * m, 6 * m), (52, 40, 34))
    _patch(p, m_left, m_front, (0.7, 0.86), (1.2 * m, 2.6 * m), (52, 44, 40))
    _patch(p, m_front, m_right, (0.4, 0.6), (1.0 * m, 2.6 * m), (52, 44, 40))
    p.gable(mill, 3.4 * m, TAN_ROOF, TAN_OCHRE)
    _tiles(p, mill, 3.4 * m, TAN_ROOF, (0.33, 0.66))
    _leather_wheel(p, *up(53.8, 4, 2.6), 3.2 * m)
    _tan_bark(p, *at(42.5, -1.5), 2.4 * m, 2.2 * m)
    for k in range(3):
        for j in range(9):
            u0 = 1.0 + 3.1 * j
            liquor = TAN_LIME if j < 2 else TAN_LIQUORS[(j + k) % 3]
            v0 = 2.6 - 2.7 * k
            _tan_pit(p, at, (u0, u0 + 2.3), (v0, v0 + 1.7), liquor)
    for u, v in ((1.0, -6.8), (3.6, -7.6)):
        hx, hy = at(u, v)
        p.blob(hx - 1.6 * m, hy - 1.2 * m, hx + 1.6 * m, hy + 0.4 * m, shade(TAN_LIME, 0.92))
        p.blob(hx - 1.0 * m, hy - 1.4 * m, hx + 0.6 * m, hy - 0.6 * m, TAN_LIME)
    _tan_cart(p, *at(3.6, 5.6))
    _tan_cart(p, *at(15.6, 5.6))
    for row, (u0, v0) in enumerate(((31.0, -9.0), (35.0, -9.6))):
        for k in range(5):
            u, v = u0 + 1.5 * k, v0 + 2.6 * k
            _tan_frame(p, *at(u, v), 3.4 * m, 3.6 * m, TAN_HIDES[(k + row) % 3])
    return p.sprite()


def _leather_front(
    p: Painter,
    top: Top,
    length: float,
    depth: float,
    high: float,
    floors: int,
    bays: int,
    sides: int,
) -> None:
    """The walls of a brick shoe works `high` units up with `floors` floors: a plinth, cream
    bands at every floor, pilasters between `bays` bays along its lit wall and `sides` along its
    shaded one, and in each bay a tall window of small panes with stitching benches lit behind
    it."""
    m = METRE
    left, front, right, _ = top
    wall = SHOE_BRICK
    _plinth(p, top, high, wall, 0.6 * m)
    storey = (high - 0.6 * m) / floors
    for a, b, count, tone in ((left, front, bays, 0.88), (front, right, sides, 0.7)):
        for k in range(count + 1):
            t = k / count
            _bar(p, _on_wall(a, b, t, 0.6 * m), _on_wall(a, b, t, high), 0.5 * m, shade(wall, tone))
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (0, 0.6 * m), shade(SHOE_TRIM, tone))
        for f in range(1, floors):
            d = 0.6 * m + storey * f
            _patch(p, a, b, (0, 1), (d - 0.12 * m, d + 0.12 * m), shade(SHOE_TRIM, tone))
    for a, b, size, count, tone in (
        (left, front, length, bays, 1.0),
        (front, right, depth, sides, 0.78),
    ):
        for k in range(count):
            t = (k + 0.5) / count
            for f in range(floors):
                head = 0.6 * m + storey * f + 0.9 * m
                _shoe_window(
                    p,
                    a,
                    b,
                    size,
                    t,
                    min(1.8 * m, size / count * 0.55),
                    (head, head + storey - 1.7 * m),
                    tone,
                )


def _leather_van(p: Painter, x: float, y: float, body: RGB) -> None:
    """A closed delivery van of the shoe works standing at (x, y), side on and heading left, a
    horse in harness before it, its box body painted `body` with a gilt band along it, on big
    spoked wheels; drawn twice its size like its horse (BEAST)."""
    m = BEAST

    def at(s: float, h: float) -> Point:
        return x + s * m, y - h * m

    p.blob(x - 2.6 * m, y - 0.25 * m, x + 1.6 * m, y + 0.35 * m, (50, 48, 44), 120)
    p.polygon([at(-1.0, 0.75), at(-1.0, 2.2), at(1.2, 2.2), at(1.2, 0.75)], body, shade(body, 0.6))
    p.polygon([at(-1.05, 2.2), at(1.25, 2.2), at(1.2, 2.32), at(-1.0, 2.32)], shade(body, 0.7))
    p.polygon([at(-0.9, 1.75), at(1.1, 1.75), at(1.1, 1.62), at(-0.9, 1.62)], GILT)
    for s in (-0.3, 0.4):
        _bar(p, at(s, 0.8), at(s, 1.55), 0.03 * m, shade(body, 0.7))
    p.polygon([at(-1.0, 1.4), at(-1.4, 1.35), at(-1.45, 1.9), at(-1.0, 1.95)], shade(body, 0.8))
    _spoked_wheel(p, *at(-0.6, 0.45), 0.45 * m)
    _spoked_wheel(p, *at(0.75, 0.55), 0.55 * m)
    _bar(p, at(-1.0, 0.8), at(-1.7, 1.05), 0.06 * m, (40, 36, 34))
    _saddle_horse(p, *at(-2.8, 0), SADDLER_BAY, -1, None, m)


def big_shoe_factory() -> Sprite:
    """A boot and shoe works. Its long works of warm brick, 38 by 12 m and four storeys high,
    has rows of tall windows of small panes between pilasters and cream bands, stitching benches
    lit behind them, and a slate roof with a strip of glazing; at its corner rises a tower of
    eight storeys with cream quoins and a balustrade, and on it stands a great gilt boot. Behind its far
    end a square stack smokes. Beside it the leather store of three storeys, its loading doors
    stacked under a hoist over a cart of hides from the tannery, and along the works' front the
    dispatch dock stacked with crates and cartons of shoes, a van of the works backed up to it."""
    m = METRE
    p = Painter(56, 50, (28, 34))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 19.5) * m, (v - 6) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    wall = SHOE_BRICK
    paving = (160, 150, 134)
    p.polygon([at(-12, -4.5), at(42, -4.5), at(42, 2), at(-12, 2)], paving)
    p.shadow(22, 7, 3)
    _shoe_stack(p, *at(37, 19), 42 * m)
    store = p.hall(*at(-8, 9), 12 * m, 10 * m, 11.5 * m, shade(wall, 0.94), None)
    s_left, s_front, s_right, _ = store
    _plinth(p, store, 11.5 * m, wall, 0.5 * m)
    for a, b, tone in ((s_left, s_front, 0.94), (s_front, s_right, 0.74)):
        for d in (4.0, 7.7):
            _patch(p, a, b, (0, 1), (d * m, (d + 0.25) * m), shade(SHOE_TRIM, tone))
    for d0, d1 in ((8.3, 11.5), (4.6, 7.5), (0.8, 3.7)):
        _patch(p, s_left, s_front, (0.4, 0.6), ((d0 - 0.2) * m, d1 * m), SHOE_TRIM)
        _patch(p, s_left, s_front, (0.42, 0.58), (d0 * m, d1 * m), shade(SHOE_LEATHER, 0.8))
        _bar(
            p,
            _on_wall(s_left, s_front, 0.5, d0 * m),
            _on_wall(s_left, s_front, 0.5, d1 * m),
            0.08 * m,
            shade(SHOE_LEATHER, 0.5),
        )
    for t in (0.15, 0.85):
        for h0 in (1.4, 5.1, 8.8):
            _shoe_window(p, s_left, s_front, 12 * m, t, 1.2 * m, (h0 * m, (h0 + 2.0) * m), 0.94)
    for t in (0.25, 0.75):
        for h0 in (1.4, 5.1, 8.8):
            _shoe_window(p, s_front, s_right, 10 * m, t, 1.2 * m, (h0 * m, (h0 + 2.0) * m), 0.74)
    _gable_across(p, store, 3.4 * m, SHOE_SLATE, shade(wall, 0.94))
    hx, hy = _on_wall(s_left, s_front, 0.5, -2.6 * m)
    _patch(p, s_left, s_front, (0.43, 0.57), (-2.1 * m, 0), shade(SHOE_LEATHER, 0.7))
    beam_end = _iso(hx, hy, 0, -1.8 * m)
    _bar(p, (hx, hy), beam_end, 0.3 * m, (62, 52, 48))
    _bar(p, beam_end, (beam_end[0], beam_end[1] + 4.6 * m), 0.05 * m, (40, 36, 34))
    p.blob(
        beam_end[0] - 0.7 * m,
        beam_end[1] + 4.5 * m,
        beam_end[0] + 0.7 * m,
        beam_end[1] + 5.7 * m,
        (156, 96, 60),
    )
    _tan_cart(p, *at(-7.5, 2.6))
    length, depth, high, rise = 38 * m, 12 * m, 17.4 * m, 3.2 * m
    _bed(p, _footing(*at(19, 12), length, depth), margin=0.6)
    top = p.hall(*at(19, 12), length, depth, high, wall, None)
    _leather_front(p, top, length, depth, high, 4, 12, 4)
    p.gable(top, rise, SHOE_SLATE, wall)
    _tiles(p, top, rise, SHOE_SLATE, (0.3, 0.6))
    glazing = [
        _on_roof(top, rise, 0.04, 0.62),
        _on_roof(top, rise, 0.96, 0.62),
        _on_roof(top, rise, 0.96, 0.88),
        _on_roof(top, rise, 0.04, 0.88),
    ]
    p.polygon(glazing, (150, 186, 206), (90, 110, 126))
    for k in range(1, 24):
        _bar(
            p,
            _on_roof(top, rise, 0.04 + 0.92 * k / 24, 0.62),
            _on_roof(top, rise, 0.04 + 0.92 * k / 24, 0.88),
            0.07 * m,
            (96, 116, 132),
        )
    tower_h = 31 * m
    tower = p.hall(*at(37.5, 5.5), 7 * m, 7 * m, tower_h, wall, None)
    t_left, t_front, t_right, _ = tower
    _plinth(p, tower, tower_h, wall, 0.6 * m)
    for a, b, tone in ((t_left, t_front, 1.0), (t_front, t_right, 0.78)):
        for t in (0.03, 0.97):
            _bar(
                p,
                _on_wall(a, b, t, 0),
                _on_wall(a, b, t, tower_h),
                0.45 * m,
                shade(SHOE_TRIM, tone),
            )
        for f in range(7):
            head = 1.4 * m + f * 3.9 * m
            _shoe_window(p, a, b, 7 * m, 0.5, 1.6 * m, (head, head + 2.4 * m), tone)
        _patch(p, a, b, (0, 1), (0, 0.7 * m), shade(SHOE_TRIM, tone))
        _patch(p, a, b, (0, 1), (-1.1 * m, 0), shade(SHOE_TRIM, tone * 0.94))
        for k in range(14):
            s = (k + 0.5) / 14
            _bar(
                p,
                _on_wall(a, b, s, -1.0 * m),
                _on_wall(a, b, s, -0.2 * m),
                0.12 * m,
                shade(SHOE_TRIM, tone * 0.8),
            )
    p.polygon(
        _round_head(t_left, t_front, 7 * m, 0.5, 3.4 * m, tower_h - 4.4 * m, tower_h), SHOE_TRIM
    )
    p.polygon(
        _round_head(t_left, t_front, 7 * m, 0.5, 2.8 * m, tower_h - 4.1 * m, tower_h), OXBLOOD
    )
    _bar(
        p,
        _on_wall(t_left, t_front, 0.5, tower_h - 2.8 * m),
        _on_wall(t_left, t_front, 0.5, tower_h),
        0.08 * m,
        shade(OXBLOOD, 0.6),
    )
    cx = (t_left[0] + t_right[0]) / 2
    cy = (t_left[1] + t_right[1]) / 2
    p.hall(cx, cy - 1.1 * m + 1.2 * m, 3.0 * m, 3.0 * m, 1.2 * m, SHOE_TRIM, shade(SHOE_TRIM, 1.05))
    _shoe_boot(p, cx, cy - 1.1 * m, 6.4 * m)
    deck = p.hall(*at(17, 3.4), 26 * m, 2.8 * m, 1.2 * m, shade(wall, 0.9), (150, 120, 88))
    for k in range(1, 14):
        _bar(
            p,
            _on_wall(deck[0], deck[1], k / 14, 0),
            _on_wall(deck[3], deck[2], k / 14, 0),
            0.04 * m,
            (120, 94, 68),
        )
    _crates(p, *up(5.0, 3.2, 1.2), (3, 2))
    _shoe_cartons(p, *up(10.6, 3.0, 1.2), 3, 3, 4)
    _shoe_cartons(p, *up(13.4, 3.4, 1.2), 2, 3, 3)
    _crates(p, *up(23.0, 3.2, 1.2), (2, 1))
    _shoe_cartons(p, *up(27.4, 3.0, 1.2), 3, 3, 5)
    _leather_van(p, *at(19.5, -2.6), OXBLOOD)
    return p.sprite()


SHOE_CONCRETE: RGB = (214, 208, 196)
SHOE_STEEL: RGB = (66, 74, 86)


def _leather_grid(
    p: Painter, x: float, y: float, length: float, depth: float, floors: int, storey: float
) -> Top:
    """A modern shoe works standing on (x, y), `length` by `depth` units and `floors` storeys
    `storey` units high: a frame of pale concrete on a grid of 6 m bays, red brick under the
    sills and wide steel windows of many panes filling the rest, the benches lit behind them,
    and a flat roof behind a concrete parapet with roof lights. Returns the corners of its
    top."""
    m = METRE
    high = floors * storey + 0.8 * m
    _bed(p, _footing(x, y, length, depth), margin=0.6)
    top = p.hall(x, y, length, depth, high, SHOE_BRICK, (92, 88, 90))
    left, front, right, back = top
    for a, b, size, tone in ((left, front, length, 1.0), (front, right, depth, 0.78)):
        bays = max(1, round(size / (6.15 * m)))
        for f in range(floors):
            d0 = 0.8 * m + storey * f
            _patch(p, a, b, (0, 1), (d0 - 0.45 * m, d0), shade(SHOE_CONCRETE, tone))
            glass = (d0 + 0.2 * m, d0 + storey - 1.0 * m)
            _patch(p, a, b, (0, 1), glass, shade((150, 170, 184), tone))
            _patch(
                p,
                a,
                b,
                (0, 1),
                (glass[0] + (glass[1] - glass[0]) * 0.45, glass[1]),
                shade(SHOE_LAMPLIT, tone * 0.95),
            )
            _patch(p, a, b, (0, 1), (glass[1] - 0.35 * m, glass[1]), shade((92, 60, 42), tone))
            panes = bays * 6
            for k in range(1, panes):
                _bar(
                    p,
                    _on_wall(a, b, k / panes, glass[0]),
                    _on_wall(a, b, k / panes, glass[1]),
                    0.06 * m,
                    shade(SHOE_STEEL, tone),
                )
            for d in (glass[0] + (glass[1] - glass[0]) * 0.4,):
                _bar(
                    p, _on_wall(a, b, 0, d), _on_wall(a, b, 1, d), 0.06 * m, shade(SHOE_STEEL, tone)
                )
        _patch(p, a, b, (0, 1), (high - 0.8 * m, high), shade(SHOE_CONCRETE, tone * 0.92))
        for k in range(bays + 1):
            t = k / bays
            _bar(
                p,
                _on_wall(a, b, t, 0),
                _on_wall(a, b, t, high),
                0.5 * m,
                shade(SHOE_CONCRETE, tone),
            )
    rim = [(px, py) for px, py in (left, front, right, back)]
    p.polygon(rim, (92, 88, 90), shade(SHOE_CONCRETE, 0.8))
    _bar(p, left, front, 0.25 * m, SHOE_CONCRETE)
    _bar(p, front, right, 0.25 * m, shade(SHOE_CONCRETE, 0.9))
    for k in range(3):
        cx, cy = _on_wall(
            _on_wall(left, back, 0.5, 0), _on_wall(front, right, 0.5, 0), (k + 0.5) / 3, 0
        )
        p.hall(cx, cy, length / 5, 2.0 * m, 0.8 * m, shade(SHOE_CONCRETE, 0.9), (150, 186, 206))
    return top


def _leather_wagon(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A covered goods wagon 9 m long standing on rails at (x, y) along the lit walls, its
    planked body painted `colour` with a sliding door, a curved grey roof and dark buffers and
    wheels."""
    m = METRE
    left, front, right, back = p.hall(x, y - 1.0 * m, 9 * m, 2.8 * m, 2.8 * m, colour, None)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        for k in range(1, 12):
            _bar(
                p,
                _on_wall(a, b, k / 12, 0.1 * m),
                _on_wall(a, b, k / 12, 2.7 * m),
                0.04 * m,
                shade(colour, 0.8 * tone),
            )
    _patch(p, left, front, (0.38, 0.62), (0.2 * m, 2.6 * m), shade(colour, 0.85))
    _bar(
        p,
        _on_wall(left, front, 0.5, 0.2 * m),
        _on_wall(left, front, 0.5, 2.6 * m),
        0.06 * m,
        shade(colour, 0.6),
    )
    _patch(p, left, front, (0.42, 0.58), (1.0 * m, 1.5 * m), GILT)
    roof = shade((150, 150, 150), 0.9)
    p.polygon([left, front, right, back], roof, shade(roof, 0.7))
    _bar(
        p,
        _on_wall(left, back, 0.5, -0.25 * m),
        _on_wall(front, right, 0.5, -0.25 * m),
        0.6 * m,
        shade(roof, 1.1),
    )
    for t in (0.18, 0.82):
        wx, wy = _on_wall(left, front, t, 3.25 * m)
        p.ellipse(wx - 0.5 * m, wy - 0.5 * m, wx + 0.5 * m, wy + 0.5 * m, (40, 38, 38))
    for t, dy in ((0, 1.9), (1, 1.9)):
        bx, by = _on_wall(left, front, t, dy * m)
        p.blob(bx - 0.25 * m, by - 0.25 * m, bx + 0.25 * m, by + 0.25 * m, (40, 38, 38))


def _leather_rails(
    p: Painter, at: Callable[[float, float], Point], u: tuple[float, float], v: float
) -> None:
    """A siding of standard gauge along the lit walls from `u[0]` to `u[1]` at `v`, where `at`
    places a spot on the ground: timber sleepers on ballast under two steel rails."""
    m = METRE
    ballast = (140, 132, 120)
    p.polygon([at(u[0], v - 1.6), at(u[1], v - 1.6), at(u[1], v + 1.6), at(u[0], v + 1.6)], ballast)
    for k in range(int((u[1] - u[0]) / 0.7)):
        uu = u[0] + 0.35 + 0.7 * k
        _bar(p, at(uu, v - 1.25), at(uu, v + 1.25), 0.22 * m, (104, 82, 62))
    for side in (-0.72, 0.72):
        _bar(p, at(u[0], v + side), at(u[1], v + side), 0.1 * m, (70, 70, 76))


def mega_shoe_factory() -> Sprite:
    """A shoe-factory quarter of the modern age. Two long works of five storeys, frames of pale
    concrete on a grid of bays filled with red brick and wide steel windows, the stitching rooms
    lit behind every one, flat roofs set with roof lights; between them at the corner the works'
    headquarters, a tower of thirteen storeys on the same grid, its corner a band of oxblood and a
    great gilt boot standing on its roof. A square stack smokes behind. Along the front a
    siding runs past the dispatch platform, cartons and crates of shoes stacked on it and goods
    wagons in the works' oxblood loading at it."""
    m = METRE
    p = Painter(70, 72, (35, 54))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 25) * m, (v - 8) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    paving = (176, 170, 160)
    p.polygon([at(0, -7.5), at(50, -7.5), at(50, 2), at(0, 2)], paving)
    p.shadow(26, 8, 3)
    _shoe_stack(p, *at(6, 33), 56 * m)
    _leather_grid(p, *at(18, 25), 36 * m, 12 * m, 6, 4.0 * m)
    _leather_grid(p, *at(18, 9), 36 * m, 12 * m, 5, 4.0 * m)
    tower = _leather_grid(p, *at(44, 10), 11 * m, 11 * m, 13, 3.6 * m)
    t_left, t_front, t_right, _ = tower
    high = 13 * 3.6 * m + 0.8 * m
    _bar(
        p,
        _on_wall(t_front, t_right, 0.0, 0),
        _on_wall(t_front, t_right, 0.0, high),
        1.2 * m,
        OXBLOOD,
    )
    for k in range(12):
        d = 0.8 * m + 3.6 * m * k + 1.0 * m
        _patch(p, t_left, t_front, (0.96, 1.0), (d, d + 1.6 * m), GILT)
    cx = (t_left[0] + t_right[0]) / 2
    cy = (t_left[1] + t_right[1]) / 2
    p.hall(
        cx + 1.2 * m,
        cy + 1.2 * m,
        4.0 * m,
        3.0 * m,
        3.0 * m,
        SHOE_CONCRETE,
        shade(SHOE_CONCRETE, 1.05),
    )
    p.hall(cx, cy - 0.2 * m, 3.4 * m, 3.4 * m, 0.8 * m, OXBLOOD, shade(OXBLOOD, 1.2))
    _shoe_boot(p, cx, cy - 1.0 * m, 7.5 * m)
    p.hall(*at(20, -0.8), 38 * m, 3.4 * m, 1.1 * m, SHOE_CONCRETE, shade(SHOE_CONCRETE, 0.9))
    _shoe_cartons(p, *up(4.0, -0.4, 1.1), 3, 3, 4)
    _crates(p, *up(8.0, -0.8, 1.1), (3, 2))
    _shoe_cartons(p, *up(13.0, -0.4, 1.1), 4, 3, 5)
    _shoe_cartons(p, *up(31.0, -0.4, 1.1), 3, 3, 3)
    _crates(p, *up(35.0, -0.8, 1.1), (2, 1))
    _leather_rails(p, at, (0, 50), -5.0)
    _leather_wagon(p, *at(9.5, -5.0), OXBLOOD)
    _leather_wagon(p, *at(20.5, -5.0), shade(OXBLOOD, 1.1))
    _leather_wagon(p, *at(36.5, -5.0), OXBLOOD)
    return p.sprite()


def _boards(p: Painter, x: float, y: float, length: float, depth: float, h: float) -> None:
    """A stack of sawn boards on bearers, the gaps between them showing."""
    for t in (0.15, 0.85):
        bx, by = _on_wall(*_footing(x, y, length, depth)[:2], t, 0)
        p.rect(bx - 0.5, by - 1, bx + 0.5, by + 0.5, LEG)
    stack = p.hall(x, y - 0.8, length, depth, h, BOARD, shade(BOARD, 1.12))
    gap = shade(BOARD, 0.5)
    for drop in range(2, int(h), 2):
        for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
            _band(p, a, b, drop, gap)


def _furniture(p: Painter, x: float, y: float) -> None:
    """A table, a chair and a tall wardrobe stood out on the ground from (x, y)."""
    wardrobe = (130, 86, 54)
    p.rect(x - 2, y - 7, x + 1, y, wardrobe)
    p.line([(x - 0.5, y - 6.5), (x - 0.5, y - 0.5)], shade(wardrobe, 0.6))
    p.point(x - 1, y - 3.5, GOLD)
    p.point(x, y - 3.5, GOLD)
    p.rect(x + 2.5, y - 2.6, x + 7, y - 2, BOARD)
    for lx in (x + 3, x + 6.5):
        p.line([(lx, y - 2), (lx, y)], LEG)
    _chair(p, x + 8, y + 0.5, 0.6)


def big_furniture_maker() -> Sprite:
    """A bigger joiner's works: a two-storey plank-walled workshop with a loft door under a hoist
    beam and a big red chair on its ridge, an open saw shed beside it, a yard of sawn boards
    stacked on bearers, heaps of sawdust and a table, chair and wardrobe stood out by the door."""
    p = Painter(50, 48, (23, 34))
    foot = _footing(-4, -2, 16, 8)
    _bed(p, foot)
    p.blob(2, -6, 24, 4, SOIL)
    p.shadow(18, 4)
    _track(p, foot[0], foot[1], (0.62, 0.86), 3)
    for x, y, h in ((14, -6, 5), (17, -1, 4), (10, 2, 3)):
        _boards(p, x, y, 8, 3, h)
    top = p.hall(-4, -2, 16, 8, 11, JOINER_WALL, None)
    left, front, right, _ = top
    _plank_walls(p, top, 11)
    _windows(p, left, front, (0.12, 0.3, 0.48), (2.5,), 2)
    _window(p, left, front, (0.08, 0.22), 7)
    _patch(p, left, front, (0.6, 0.76), (1.6, 4.6), (56, 40, 30))
    _door(p, left, front, (0.62, 0.86), 6.6, 11, (56, 40, 30), shade(JOINER_WALL, 1.3))
    p.line([_on_wall(left, front, 0.74, 6.8), _on_wall(left, front, 0.74, 11)], (88, 64, 44))
    _window(p, front, right, (0.3, 0.6), 2, shade(WINDOW, 0.8))
    _window(p, front, right, (0.3, 0.6), 6.5, shade(WINDOW, 0.8))
    p.gable(top, 5, JOINER_ROOF, JOINER_WALL)
    _roof_rows(p, top, 5, JOINER_ROOF)
    hx, hy = _on_wall(left, front, 0.68, 0)
    p.line([(hx, hy - 1), (hx - 3, hy - 0.5)], LEG)
    p.line([(hx - 2.6, hy - 0.5), (hx - 2.6, hy + 3)], (70, 66, 62))
    rx, ry = _on_roof(top, 5, 0.35, 1)
    _chair(p, rx - 3, ry + 0.5, 1.3)
    p.blob(-18, 6, -10, 9, SAWDUST)
    p.blob(-13, 9, -7, 11, shade(SAWDUST, 0.92))
    for x in (-16, -12):
        p.line([(x - 1, 8), (x, 5), (x + 1, 8)], LEG)
    p.line([(-17, 4), (-10, 6.5)], BOARD)
    p.line([(-17, 5), (-10, 7.5)], shade(BOARD, 0.7))
    _furniture(p, 4, 11)
    return p.sprite()


def mega_furniture_maker() -> Sprite:
    """A great furniture factory: a long plank-walled hall under a glazed sawtooth roof with a giant
    red chair raised on its roof, a timber drying kiln tower with a louvred top, a banded steam
    stack smoking, a lumber yard of sawn boards stacked high, heaps of sawdust and a row of
    finished tables, chairs and wardrobes out front by a cart."""
    p = Painter(64, 76, (32, 60))
    foot = _footing(-6, -4, 24, 10)
    _bed(p, foot)
    p.blob(4, -12, 30, 6, SOIL)
    p.shadow(24, 5)
    _track(p, foot[0], foot[1], (0.78, 0.94), 3.5)
    _smoking_chimney(p, 2, -14, -44, shade(BRICK, 0.85), smoke=STEAM)
    kiln = p.box(16, -10, 4, 16, shade(JOINER_WALL, 0.9), None)
    _plank_walls(p, kiln, 16)
    for a, b in ((kiln[0], kiln[1]), (kiln[1], kiln[2])):
        _patch(p, a, b, (0.15, 0.85), (1.2, 3.6), (70, 54, 40))
        for t in (0.3, 0.5, 0.7):
            p.line(
                [_on_wall(a, b, t - 0.08, 1.4), _on_wall(a, b, t + 0.08, 3.4)], shade(BOARD, 0.8)
            )
    p.gable(kiln, 3, JOINER_ROOF, JOINER_WALL)
    _smoke(p, 17, -32, STEAM, 2)
    for x, y, h in ((22, -6, 7), (20, 1, 5), (25, 2, 4), (13, 5, 3)):
        _boards(p, x, y, 8, 3, h)
    top = p.hall(-6, -4, 24, 10, 10, JOINER_WALL, None)
    left, front, right, _ = top
    _plank_walls(p, top, 10)
    _windows(p, left, front, tuple(0.06 + 0.12 * k for k in range(6)), (2.4, 5.4), 2)
    _door(p, left, front, (0.78, 0.94), 4, 10, (56, 40, 30), shade(JOINER_WALL, 1.3))
    p.line([_on_wall(left, front, 0.86, 4.2), _on_wall(left, front, 0.86, 10)], (88, 64, 44))
    _windows(p, front, right, (0.25, 0.5, 0.75), (2.4, 5.4), 2, shade(WINDOW, 0.8))
    _sawtooth(p, top, 5, 4, JOINER_ROOF, GLAZING, JOINER_WALL)
    rx, ry = _on_wall(top[0], top[2], 0.45, -3)
    for dx in (-2, 4):
        p.line([(rx + dx, ry + 2), (rx + dx, ry - 1)], (70, 66, 62))
    _chair(p, rx - 4, ry, 2.2)
    p.blob(3, 9, 15, 14, SAWDUST)
    _heap(p, 9, 12, 3.5, 3, SAWDUST)
    for x, y in ((-22, 1.5), (-13, 6)):
        _furniture(p, x, y)
    _cart(p, 22, 12)
    for k in range(2):
        p.line([(20 + 2 * k, 9), (24 + 2 * k, 10.5)], BOARD)
    return p.sprite()


def _river(p: Painter, a: Point, b: Point, width: float) -> None:
    """A stream between muddy banks flowing over the ground from `a` to `b`, `width` across."""
    (ax, ay), (bx, by) = a, b
    p.polygon(
        [
            (ax - 1, ay - width / 2 - 1),
            (bx + 1, by - width / 2),
            (bx - 1, by + width / 2 + 1.5),
            (ax - 1, ay + width / 2 + 1),
        ],
        BANK,
    )
    p.polygon(
        [
            (ax, ay - width / 2),
            (bx, by - width / 2 + 0.6),
            (bx - 1, by + width / 2),
            (ax, ay + width / 2),
        ],
        WATER,
    )
    for t in (0.2, 0.55):
        x, y = _on_wall(a, b, t, -0.5)
        p.line([(x, y), (x + 3, y + 1.5)], (150, 196, 226))


def _waterwheel(p: Painter, cx: float, cy: float, r: float) -> None:
    """A wooden waterwheel turning in the stream at (cx, cy), churning foam below it."""
    p.ellipse(cx - r * 0.6, cy - r, cx + r * 0.6, cy + r * 0.8, WHEEL)
    p.ellipse(cx - r * 0.4, cy - r * 0.8, cx + r * 0.4, cy + r * 0.6, shade(WHEEL, 1.3))
    p.line([(cx, cy - r * 0.8), (cx, cy + r * 0.6)], WHEEL)
    p.line([(cx - r * 0.4, cy - 0.1 * r), (cx + r * 0.4, cy - 0.1 * r)], WHEEL)
    for a in range(8):
        ang = a * math.pi / 4
        p.point(
            cx + r * 0.6 * math.cos(ang), cy - 0.1 * r + r * 0.9 * math.sin(ang), shade(WHEEL, 0.6)
        )
    for dx, dy in ((-1.5, r * 0.8), (1, r * 0.9 + 0.5), (2.5, r * 0.8 + 1)):
        p.point(cx + dx, cy + dy, FOAM)


def _rolls(p: Painter, x: float, y: float, rows: int) -> None:
    """A pyramid of paper rolls standing on end from (x, y), `rows` deep."""
    for row in range(rows):
        for k in range(rows - row):
            rx = x + 3 * k + 1.5 * row
            ry = y - 1.5 * k - 3 * row + 0.75 * row
            p.ellipse(rx - 1.5, ry - 3, rx + 1.5, ry, ROLL)
            p.point(rx, ry - 1.5, shade(ROLL, 0.55))


def _logs(p: Painter, x: float, y: float, rows: int) -> None:
    """A pile of pulpwood logs with their pale ends showing, among scattered bark."""
    for dx, dy in ((-1, 1.5), (3, 2), (6, 0.8)):
        p.point(x + dx, y + dy, (150, 112, 74))
    for row in range(rows):
        for k in range(rows + 1 - row):
            lx, ly = x + 2 * k + row, y - 1 * k - 1.8 * row
            p.ellipse(lx - 1, ly - 2, lx + 1, ly, LOG)
            p.point(lx, ly - 1, (150, 112, 74))


def _paper_mill_hall(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, storeys: int
) -> Top:
    top = p.hall(x, y, length, depth, height, PAPER_WALL, None)
    left, front, right, _ = top
    lintel = shade(PAPER_WALL, 1.2)
    _plinth(p, top, height, PAPER_WALL)
    _stains(p, left, front, height, (0.45, 0.65, 0.85), shade(PAPER_WALL, 0.6), 2.2)
    _stains(p, front, right, height, (0.15, 0.85), shade(PAPER_WALL, 0.5), 1.6)
    n = max(3, round(length / 3.4))
    drops = tuple(2 + 3.6 * f for f in range(storeys))
    _windows(p, left, front, tuple((k + 0.5) / n for k in range(n)), drops, 2, arch=lintel)
    _windows(p, front, right, (0.3, 0.75), drops, 2, shade(WINDOW, 0.8), lintel)
    _door(p, front, right, (0.4, 0.65), height - 4, height, (70, 56, 48), lintel)
    return top


def big_paper_mill() -> Sprite:
    """A bigger riverside paper mill: a long two-storey mill turning two waterwheels in its stream,
    a drying loft of louvres along its ridge, a tall banded stack smoking, a pyramid of paper rolls
    in the yard and a big pile of pulpwood logs among scattered bark."""
    p = Painter(54, 68, (27, 54))
    _river(p, (-25, -6), (2, 7.5), 5)
    foot = _footing(1, -6, 18, 9)
    _bed(p, foot)
    p.blob(10, 2, 24, 10, (120, 96, 66))
    p.shadow(16, 4)
    _track(p, foot[1], foot[2], (0.4, 0.65), 4)
    _smoking_chimney(p, 11, -8, -38)
    top = _paper_mill_hall(p, 1, -6, 18, 9, 10, 2)
    left, front, _, _ = top
    p.gable(top, 5, PAPER_ROOF, PAPER_WALL)
    _roof_rows(p, top, 5, PAPER_ROOF, (0.34,))
    for t in (0.2, 0.35, 0.5, 0.65, 0.8):
        a, b = _on_roof(top, 5, t - 0.04, 0.6), _on_roof(top, 5, t + 0.04, 0.95)
        p.line([a, b], shade(PAPER_ROOF, 1.35))
    for t in (0.2, 0.55):
        cx, cy = _on_wall(left, front, t, 8)
        _waterwheel(p, cx, cy, 4)
    _rolls(p, 6, 10, 3)
    _logs(p, 17, 6, 3)
    return p.sprite()


def mega_paper_mill() -> Sprite:
    """A great paper works on its river: a long mill turning three waterwheels, a tall machine hall
    under a glazed sawtooth roof behind, two round pulp digesters banded in iron, two tall stacks
    smoking and steam rising, pyramids of paper rolls in the yard and a great pile of pulpwood logs
    among scattered bark."""
    p = Painter(64, 80, (32, 64))
    _river(p, (-30, -8), (2, 9), 6)
    foot = _footing(-1, -8, 24, 10)
    _bed(p, foot)
    _bed(p, _footing(16, -12, 12, 9))
    p.blob(8, 2, 30, 14, (120, 96, 66))
    p.shadow(22, 5)
    for x, foot_y, tall in ((4, -18, 34), (24, -12, 34)):
        _smoking_chimney(p, x, foot_y, foot_y - tall)
    for x, y in ((20, -14), (25, -11.5)):
        _turned(
            p,
            x,
            y,
            [(0, 2.4), (14, 2.4), (16, 1.4)],
            shade(PAPER_WALL, 0.85),
            shade(PAPER_WALL, 0.7),
        )
        for h in (4, 9):
            _ring(p, x, y - h, 2.4, (90, 84, 82))
    _smoke(p, 21, -32, STEAM, 2)
    mh = p.hall(16, -6, 12, 8, 9, PAPER_WALL, None)
    _plinth(p, mh, 9, PAPER_WALL)
    _windows(p, mh[0], mh[1], (0.2, 0.4, 0.6, 0.8), (2.5,), 4, arch=shade(PAPER_WALL, 1.2))
    _windows(p, mh[1], mh[2], (0.3, 0.7), (2.5,), 4, shade(WINDOW, 0.8))
    _sawtooth(p, mh, 3, 3, PAPER_ROOF, GLAZING, PAPER_WALL)
    _track(p, mh[1], (mh[1][0] + 8, mh[1][1] - 4), (0.3, 0.6), 4)
    top = _paper_mill_hall(p, -1, -8, 24, 10, 13, 3)
    left, front, _, _ = top
    p.gable(top, 6, PAPER_ROOF, PAPER_WALL)
    _roof_rows(p, top, 6, PAPER_ROOF, (0.34,))
    for t in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
        a, b = _on_roof(top, 6, t - 0.03, 0.6), _on_roof(top, 6, t + 0.03, 0.95)
        p.line([a, b], shade(PAPER_ROOF, 1.35))
    for t in (0.15, 0.45, 0.75):
        cx, cy = _on_wall(left, front, t, 11)
        _waterwheel(p, cx, cy, 4.5)
    _rolls(p, 4, 14, 3)
    _rolls(p, 13, 10, 2)
    _logs(p, 20, 9, 4)
    return p.sprite()


def _newspaper(p: Painter, x: float, y: float, half: float, tall: float) -> None:
    """A newspaper page held up on posts with its feet at (x, y) as a sign: a red masthead, a
    headline, a picture and columns of print."""
    for dx in (-half * 0.5, half * 0.5):
        p.line([(x + dx, y), (x + dx, y - 3)], (70, 66, 62))
    t, b = y - 3 - tall, y - 3
    p.rect(x - half, t, x + half, b, PAGE)
    p.rect(x - half + 1, t + 1, x + half - 1, t + 2.2, MASTHEAD)
    p.line([(x - half + 1.5, t + 3.6), (x + half - 1.5, t + 3.6)], INK)
    if tall > 9:
        p.line([(x - half + 1.5, t + 4.8), (x + half * 0.4, t + 4.8)], INK)
    p.rect(x - half + 1.5, t + tall * 0.55, x - 1, b - 1.5, PRINT)
    for yy in range(int(t + tall * 0.55), int(b - 1), 2):
        p.line([(x + 1, yy), (x + half - 1.5, yy)], PRINT)


def _bundles(p: Painter, x: float, y: float, count: int) -> None:
    for k in range(count):
        _bundle(p, x + 3.4 * (k % 3) - 1.7 * (k // 3), y - 1.7 * (k % 3) - 1.6 * (k // 3))


def _press_house(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, storeys: int
) -> Top:
    top = p.hall(x, y, length, depth, height, PRESS_WALL, PRESS_ROOF)
    left, front, right, _ = top
    lintel = shade(PRESS_WALL, 1.12)
    _plinth(p, top, height, PRESS_WALL)
    _stains(p, front, right, height, (0.12, 0.38, 0.62, 0.88), shade(PRESS_WALL, 0.56), 1.5)
    _band(p, left, front, 1, lintel)
    p.line([left, front, right], shade(PRESS_ROOF, 1.3))
    n = max(4, round(length / 3))
    spans = tuple((k + 0.5) / n for k in range(n))
    floor = (height - 2.5) / storeys
    for f in range(storeys):
        drop = 2.5 + floor * f
        _windows(
            p, left, front, spans, (drop,), max(2, round(floor) - 2), arch=shade(PRESS_WALL, 0.75)
        )
        _windows(
            p,
            front,
            right,
            (0.25, 0.5, 0.75),
            (drop,),
            max(2, round(floor) - 2),
            shade(WINDOW, 0.75),
        )
        if f:
            _bands(
                p, left, front, right, drop - 1.2, shade(PRESS_WALL, 0.85), shade(PRESS_WALL, 0.68)
            )
    return top


def big_printing_works() -> Sprite:
    """A bigger printing house: three storeys of tall arched windows over a dark plinth with a
    great newspaper sign held up on its roof, a glazed press hall beside it, a pavement of flags
    with a path to the door, a cart loaded with bundles of papers, more bundles piled up and ink
    barrels by a stain of spilt ink."""
    p = Painter(48, 54, (23, 38))
    foot = _footing(-2, -1, 16, 9)
    _pavement(p, foot, 5)
    _bed(p, _footing(14, -2, 8, 7))
    p.shadow(15, 4)
    _track(p, foot[0], foot[1], (0.62, 0.8), 2.6, shade(PAVING, 0.9), ruts=False)
    _oil(p, 10, 8, 1.6)
    annex = p.hall(14, -2, 8, 7, 6, PRESS_WALL, None)
    _plinth(p, annex, 6, PRESS_WALL)
    _patch(p, annex[1], annex[2], (0.2, 0.8), (2, 5), DOORWAY_PRESS)
    p.gable(annex, 3, PRESS_ROOF, PRESS_WALL)
    _skylight(p, annex, 3, (0.1, 0.9))
    top = _press_house(p, -2, -1, 16, 9, 16, 3)
    left, front, _, _ = top
    _door(p, left, front, (0.62, 0.78), 11.5, 16, (70, 56, 48), shade(PRESS_WALL, 0.75))
    rx, ry = _on_wall(top[0], top[2], 0.5, 0)
    _newspaper(p, rx, ry, 9, 10)
    cx, cy = 8, 12
    p.rect(cx - 4, cy - 2.6, cx + 4, cy - 1, (130, 92, 60))
    for k in range(3):
        bundle = p.box(cx - 2.5 + 2.6 * k, cy - 2.4, 1.4, 1.4, BUNDLE, shade(BUNDLE, 1.04))
        p.line([bundle[1], (bundle[1][0], bundle[1][1] + 1.4)], (150, 110, 80))
    for wx in (cx - 2.5, cx + 2.5):
        p.ellipse(wx - 1.1, cy - 1.5, wx + 1.1, cy + 0.7, (70, 54, 40))
    _bundles(p, 15, 7, 4)
    for x, y in ((-16, 6), (-13.5, 7.5)):
        _barrel(p, x, y)
    return p.sprite()


def mega_printing_works() -> Sprite:
    """A great newspaper press house: a long building of four storeys of tall arched windows with a
    clock over its door and a huge newspaper sign raised on a frame on its roof, a glazed press
    hall where the rotary presses roll, a flag flying, rolls of newsprint delivered on a wagon,
    bundles of papers stacked high and carts waiting to take them away."""
    p = Painter(64, 74, (32, 52))
    foot = _footing(-4, -3, 24, 10)
    _pavement(p, foot, 7)
    _bed(p, _footing(18, -4, 12, 9))
    p.shadow(22, 5)
    _track(p, foot[0], foot[1], (0.7, 0.84), 3, shade(PAVING, 0.9), ruts=False)
    _oil(p, 14, 12, 1.8)
    hall = p.hall(18, -4, 12, 9, 8, PRESS_WALL, None)
    _plinth(p, hall, 8, PRESS_WALL)
    _patch(p, hall[0], hall[1], (0.1, 0.9), (1.6, 6), GLASS)
    for t in (0.2, 0.45, 0.7):
        cx, cy = _on_wall(hall[0], hall[1], t + 0.05, 4.4)
        p.ellipse(cx - 1.2, cy - 1.2, cx + 1.2, cy + 1.2, INK)
        p.point(cx, cy, PRINT)
    p.line([_on_wall(hall[0], hall[1], 0.1, 3), _on_wall(hall[0], hall[1], 0.9, 3)], PAGE)
    _door(p, hall[1], hall[2], (0.2, 0.8), 3, 8, DOORWAY_PRESS)
    _sawtooth(p, hall, 4, 3, PRESS_ROOF, GLAZING, PRESS_WALL)
    top = _press_house(p, -4, -3, 24, 10, 22, 4)
    left, front, right, _ = top
    _door(p, left, front, (0.44, 0.56), 17, 22, (70, 56, 48), shade(PRESS_WALL, 0.75))
    cx, cy = _on_wall(left, front, 0.5, 14.4)
    p.ellipse(cx - 1.6, cy - 1.6, cx + 1.6, cy + 1.6, (248, 246, 236))
    p.line([(cx, cy), (cx, cy - 1.1)], INK)
    p.line([(cx, cy), (cx + 1, cy)], INK)
    rx, ry = _on_wall(top[0], top[2], 0.5, 0)
    for dx in (-8, 8):
        p.line([(rx + dx, ry + 1), (rx + dx, ry - 4)], (70, 66, 62))
    p.line([(rx - 8, ry - 2), (rx + 8, ry - 2)], (70, 66, 62))
    _newspaper(p, rx, ry - 2, 13, 14)
    _flag(p, right[0] - 1, right[1], 7, MASTHEAD)
    _rails(p, (15, 16.5), (29, 9.5), 7)
    wx, wy = 22, 13
    p.rect(wx - 5, wy - 2.6, wx + 5, wy - 1, (70, 72, 80))
    for k in range(3):
        rx2 = wx - 3 + 3 * k
        p.ellipse(rx2 - 1.4, wy - 5.4, rx2 + 1.4, wy - 2.4, ROLL)
        p.point(rx2, wy - 3.9, shade(ROLL, 0.55))
    for wxx in (wx - 3, wx + 3):
        p.ellipse(wxx - 1, wy - 1.6, wxx + 1, wy + 0.4, (40, 38, 40))
    _bundles(p, -20, 6, 6)
    _bundles(p, 4, 14, 3)
    for x, y in ((-10, 14), (-3, 17)):
        p.rect(x - 4, y - 2.6, x + 4, y - 1, (130, 92, 60))
        for j in range(2):
            bundle = p.box(x - 1.4 + 2.8 * j, y - 2.4, 1.3, 1.3, BUNDLE, shade(BUNDLE, 1.04))
            p.line([bundle[1], (bundle[1][0], bundle[1][1] + 1.3)], (150, 110, 80))
        p.ellipse(x - 1.2, y - 1.5, x + 1.2, y + 0.7, (70, 54, 40))
    return p.sprite()


def _forge_hall(
    p: Painter,
    x: float,
    y: float,
    length: float,
    depth: float,
    height: float,
    doors: tuple[float, ...],
) -> Top:
    """A dark forge hall with doors open on glowing hearths along its lit wall."""
    top = p.hall(x, y, length, depth, height, FORGE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, height, FORGE_WALL)
    _stains(p, front, right, height, (0.15, 0.5, 0.85), shade(FORGE_WALL, 0.55), 1.8)
    w = 0.34 * 13 / length
    for t in doors:
        _patch(p, left, front, (t - w / 2, t + w / 2), (height - 4.5, height), GLOW)
        _patch(p, left, front, (t - w / 4, t + w / 4), (height - 2.5, height), HOT)
        p.line(
            [
                _on_wall(left, front, t - w / 2 - 0.02, height - 5.1),
                _on_wall(left, front, t + w / 2 + 0.02, height - 5.1),
            ],
            shade(FORGE_WALL, 1.5),
        )
    _windows(p, front, right, (0.3, 0.7), (3,), 1, (240, 150, 60), shade(FORGE_WALL, 1.5))
    return top


def _tools(p: Painter, x: float, y: float, count: int) -> None:
    """A rack of finished shovels and picks leant on a rail standing on (x, y)."""
    p.line([(x - 0.5, y - 4), (x + count * 1.6, y - 4 + count * 0.8)], LEG)
    for k in range(count):
        tx, ty = x + 1.6 * k, y + 0.8 * k
        p.line([(tx, ty - 5), (tx, ty)], HANDLE)
        if k % 2:
            p.line([(tx - 1, ty - 5), (tx + 1, ty - 4.6)], BAR)
        else:
            p.rect(tx - 0.6, ty - 1, tx + 0.6, ty, BAR)


def _sparks(p: Painter, points: tuple[Point, ...]) -> None:
    for k, (x, y) in enumerate(points):
        p.point(x, y, HOT if k % 2 else GLOW)


def big_tool_works() -> Sprite:
    """A bigger tool works: a long dark forge hall with two doors open on glowing hearths and an
    anvil painted on a board on its wall, a trip-hammer shed glowing beside it, two banded stacks
    smoking and sparking, a heap of coal spilling over the sooty ground, a quenching tub, racks of
    finished shovels and picks and a pile of iron bars."""
    p = Painter(48, 58, (24, 42))
    foot = _footing(-3, -2, 18, 9)
    _bed(p, foot, FORGE_GROUND, 1.6)
    _bed(p, _footing(13, -5, 7, 6), FORGE_GROUND, 1.4)
    _spill(p, -15, 4, 3.5, COAL)
    p.shadow(17, 4)
    _track(p, foot[0], foot[1], (0.62, 0.86), 3, (96, 88, 82))
    shed = p.hall(13, -5, 7, 6, 5, shade(FORGE_WALL, 1.1), None)
    _patch(p, shed[1], shed[2], (0.25, 0.75), (1.5, 5), GLOW)
    _patch(p, shed[1], shed[2], (0.4, 0.6), (3, 5), HOT)
    p.gable(shed, 2.5, FORGE_ROOF, FORGE_WALL)
    top = _forge_hall(p, -3, -2, 18, 9, 8, (0.62, 0.86))
    left, front, _, _ = top
    _patch(p, left, front, (0.04, 0.42), (0.6, 6.4), SIGNBOARD)
    _patch(p, left, front, (0.04, 0.42), (5.6, 6.4), shade(SIGNBOARD, 0.8))
    _anvil(p, *_on_wall(left, front, 0.23, 3.4), 1.1)
    p.gable(top, 4, FORGE_ROOF, FORGE_WALL)
    _roof_rows(p, top, 4, FORGE_ROOF)
    for along in (0.3, 0.75):
        x, y = _on_roof(top, 4, along, 0.9)
        cap = _forge_stack(p, x, y + 1, y - 9)
        _smoke(p, x + 1, cap - 4, (96, 94, 98))
    _sparks(p, ((-6, -22), (-3, -24), (2, -21), (5, -25), (-1, -27)))
    _heap(p, -15, 4, 3.5, 3, COAL)
    _puddle(p, 4, 5.6, 2.2)
    p.ellipse(2, 3.5, 6, 4.5, (70, 96, 120))
    _tools(p, -9, 10, 6)
    _lengths(p, 13, 5, BAR, 3)
    _barrel(p, 7, 10)
    return p.sprite()


def mega_tool_works() -> Sprite:
    """A great tool works: a long dark forge hall under a sawtooth roof glowing from the hearths
    inside, three doors open on the fire, a tall steam-hammer house with its door glowing, three
    banded stacks smoking and throwing sparks, a giant anvil sign on a frame, a great heap of coal,
    a rail siding with a wagon of coal, racks of finished shovels and picks, quenching tubs and
    piles of iron bars."""
    p = Painter(62, 76, (31, 56))
    foot = _footing(-6, -5, 26, 10)
    _bed(p, foot, FORGE_GROUND, 2)
    _bed(p, _footing(14, -10, 8, 8), FORGE_GROUND, 1.4)
    p.blob(-28, 1, -8, 11, FORGE_GROUND)
    p.blob(-14, 7, 6, 16, FORGE_GROUND)
    _spill(p, -22, 5, 4.5, COAL)
    p.shadow(23, 5)
    for k, x in enumerate((-12, -2, 8)):
        _smoking_chimney(p, x, -14 + k, -42 + k, (74, 58, 52), smoke=(96, 94, 98))
    _sparks(p, ((-14, -46), (-10, -44), (-4, -47), (0, -45), (6, -46), (10, -44), (-8, -49)))
    house = p.hall(14, -10, 8, 8, 16, shade(FORGE_WALL, 1.08), None)
    _plinth(p, house, 16, FORGE_WALL)
    _patch(p, house[0], house[1], (0.25, 0.75), (8, 16), GLOW)
    _patch(p, house[0], house[1], (0.38, 0.62), (11, 16), HOT)
    _windows(p, house[1], house[2], (0.3, 0.7), (3, 8), 2, (240, 150, 60))
    p.gable(house, 3, FORGE_ROOF, FORGE_WALL)
    top = _forge_hall(p, -6, -5, 26, 10, 10, (0.5, 0.7, 0.9))
    left, front, _, _ = top
    _windows(p, left, front, (0.06, 0.14, 0.22, 0.3, 0.38), (2.5,), 3, GLOW, shade(FORGE_WALL, 1.6))
    _sawtooth(p, top, 4, 3, FORGE_ROOF, shade(GLOW, 0.9), FORGE_WALL)
    sx, sy = _on_wall(foot[0], foot[1], 0.2, 4)
    for dx in (-5, 5):
        p.line([(sx + dx, sy), (sx + dx, sy - 9)], LEG)
    p.rect(sx - 6, sy - 15, sx + 6, sy - 7, SIGNBOARD)
    _anvil(p, sx, sy - 11, 1.6)
    _heap(p, -22, 5, 4.5, 4, COAL)
    p.point(-23, 2, (104, 104, 116))
    _rails(p, (6, 18), (28, 7), 9)
    _wagon(p, 18, 13, (90, 70, 56))
    p.blob(15, 9, 21, 11.5, COAL)
    _puddle(p, -6, 12, 2.4)
    p.ellipse(-8, 10, -4, 11, (70, 96, 120))
    _tools(p, -2, 13, 6)
    _tools(p, -14, 12, 5)
    _lengths(p, 24, 0, BAR, 3)
    _lengths(p, 18, 3, BAR, 2)
    return p.sprite()


def _engine_hall(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> Top:
    """A brick engine hall over a dark plinth stained by the damp, with tall arched windows."""
    top = p.hall(x, y, length, depth, height, BRICK, None)
    left, front, right, _ = top
    lintel = shade(BRICK, 1.3)
    _plinth(p, top, height, BRICK)
    _stains(p, left, front, height, (0.04, 0.21, 0.39, 0.57, 0.73), shade(BRICK, 0.62), 2)
    _stains(p, front, right, height, (0.2, 0.8), shade(BRICK, 0.5), 2)
    _bands(p, left, front, right, 1, lintel, shade(BRICK, 1.05))
    n = max(4, round(length / 3.6))
    spans = tuple((k + 0.5) / (n + 1) for k in range(n))
    _windows(p, left, front, spans, (2.5,), round(height * 0.5), arch=lintel)
    _windows(p, left, front, spans, (height * 0.5 + 3.2,), 1, shade(BRICK, 1.15))
    _door(p, left, front, (n / (n + 1) + 0.02, 0.97), height * 0.45, height, (58, 48, 44), lintel)
    p.line(
        [
            _on_wall(left, front, (n / (n + 1) + 0.99) / 2, height * 0.45),
            _on_wall(left, front, (n / (n + 1) + 0.99) / 2, height),
        ],
        (44, 36, 34),
    )
    return top


def _gantry(p: Painter, feet: tuple[Point, Point], height: float) -> None:
    """A yellow gantry crane on two legs standing on rails, its beam across their tops."""
    for x, y in feet:
        p.line([(x, y), (x, y - height)], CRANE)
        p.line([(x + 0.6, y), (x + 0.6, y - height)], shade(CRANE, 0.75))
        p.line([(x - 1.5, y), (x, y - 4)], shade(CRANE, 0.7))
        p.rect(x - 1.5, y - 0.5, x + 1.5, y + 0.5, (70, 70, 76))
    (x0, y0), (x1, y1) = feet
    p.line([(x0 - 1, y0 - height), (x1 + 1, y1 - height)], CRANE)
    p.line([(x0 - 1, y0 - height + 1), (x1 + 1, y1 - height + 1)], shade(CRANE, 0.7))
    for k in range(1, 4):
        a = _on_wall((x0, y0 - height), (x1, y1 - height), (k - 0.5) / 3, 0)
        b = _on_wall((x0, y0 - height), (x1, y1 - height), k / 3, 1)
        p.line([a, b], shade(CRANE, 0.85))


def _wheel(p: Painter, cx: float, cy: float, rx: float, ry: float, colour: RGB) -> None:
    """A spoked iron wheel seen a little edge on, centred on (cx, cy), open between its spokes."""
    for inset in (0, 0.7):
        ring = [
            (
                cx + (rx - inset) * math.cos(a * math.pi / 8),
                cy + (ry - inset) * math.sin(a * math.pi / 8),
            )
            for a in range(17)
        ]
        p.line(ring, colour if inset else shade(colour, 0.7))
    for a in range(4):
        ang = a * math.pi / 4
        dx, dy = rx * 0.9 * math.cos(ang), ry * 0.9 * math.sin(ang)
        p.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], colour)
    p.point(cx, cy, BRASS)


def _flywheel(p: Painter, x: float, y: float, r: float) -> None:
    """A stationary steam engine on (x, y): a great spoked flywheel beside its green cylinder on
    an iron bed, a brass rod between them."""
    p.rect(x - 1, y - 2, x + 6, y, CASTING)
    _wheel(p, x - r * 0.15, y - r, r * 0.7, r, shade(CASTING, 0.9))
    p.rect(x + 2, y - 4.5, x + 6, y - 2, METAL_GREEN)
    p.line([(x + 2, y - 3.6), (x + 6, y - 3.6)], shade(METAL_GREEN, 1.4))
    p.line([(x - r * 0.15, y - r), (x + 2, y - 3)], BRASS)


def _boiler(p: Painter, x: float, y: float) -> None:
    """A new locomotive boiler lying on blocks at (x, y), its smokebox end dark."""
    p.rect(x - 1, y - 1, x + 1, y, (110, 84, 60))
    p.rect(x + 5, y - 1, x + 7, y, (110, 84, 60))
    p.polygon(
        [(x - 2, y - 5), (x + 7, y - 0.5), (x + 7, y - 4.5), (x - 2, y - 9)],
        CASTING,
        shade(CASTING, 0.6),
    )
    p.line([(x - 2, y - 7.6), (x + 7, y - 3.1)], shade(CASTING, 1.4))
    for t in (0.3, 0.65):
        p.line(
            [(x - 2 + 9 * t, y - 9 + 4.5 * t), (x - 2 + 9 * t, y - 5 + 4.5 * t)],
            shade(CASTING, 0.75),
        )
    p.ellipse(x + 5.4, y - 4.8, x + 8.4, y - 0.4, (40, 38, 42))


def big_machine_works() -> Sprite:
    """A bigger machine works: a long brick engine hall with tall arched windows, a glazed roof
    light and a great brass gear on its end wall, a lower erecting shop beside it, a tall banded
    chimney smoking, and a gravelled yard where a gantry crane on rails lifts a casting over a
    boiler lying on blocks, with pipes piled up and oil stains."""
    p = Painter(54, 70, (26, 54))
    foot = _footing(-6, -3, 20, 10)
    _bed(p, foot)
    _bed(p, _footing(12, -9, 10, 7))
    p.polygon([(4, 9), (17, 2.5), (26, 7), (13, 13.5)], GRAVEL)
    p.shadow(20, 5)
    _smoking_chimney(p, 2, -14, -40, shade(BRICK, 0.85))
    shop = p.hall(12, -9, 10, 7, 7, shade(BRICK, 0.95), None)
    _plinth(p, shop, 7, BRICK)
    _windows(p, shop[0], shop[1], (0.25, 0.5, 0.75), (2.4,), 3, arch=shade(BRICK, 1.3))
    _door(p, shop[1], shop[2], (0.25, 0.75), 2.6, 7, (58, 48, 44))
    p.gable(shop, 3, ENGINE_ROOF, BRICK)
    _track(p, foot[0], foot[1], (0.82, 0.96), 4)
    for dx, dy in ((0, 0), (7, -3.5)):
        p.line([(6 + dx, 10 + dy), (12 + dx, 13 + dy)], RAIL)
    _oil(p, 17, 8, 1.4)
    top = _engine_hall(p, -6, -3, 20, 10, 12)
    _, front, right, _ = top
    p.gable(top, 6, ENGINE_ROOF, BRICK)
    _skylight(p, top, 6, (0.06, 0.94))
    _gear(p, *_on_wall(front, right, 0.5, 6), 3.2, BRASS)
    _lengths(p, -19, 4, (96, 100, 112), 3)
    _boiler(p, 12, 10)
    _gantry(p, ((8, 11), (19, 5.5)), 18)
    p.rect(12.5, -10, 14.5, -8.5, (60, 64, 74))
    p.line([(13.5, -8.5), (13.5, -4)], (70, 64, 58))
    p.rect(11, -4, 16, -0.5, CASTING)
    p.line([(11.5, -3), (15.5, -3)], shade(CASTING, 1.3))
    return p.sprite()


def mega_machine_works() -> Sprite:
    """A great machine works: a long brick engine hall with tall arched windows under a raised
    glazed clerestory, a giant brass gear on its gable, two tall banded chimneys smoking, a big
    gantry crane spanning the yard on rails lifting a flywheel, a stationary steam engine and a
    new locomotive out front on a rail siding, a boiler lying on blocks and pipes piled up."""
    p = Painter(64, 80, (31, 60))
    foot = _footing(-8, -6, 28, 11)
    _bed(p, foot)
    p.polygon([(-2, 10), (16, 1), (31, 8.5), (13, 17.5)], GRAVEL)
    p.shadow(25, 6)
    for x, foot_y, tall in ((6, -18, 30), (12, -15, 32)):
        cap = _chimney(p, x, foot_y, foot_y - tall, shade(BRICK, 0.85))
        _ring(p, x, foot_y - (foot_y - cap) * 0.4, 1.5, shade(BRICK, 0.6))
        _smoke(p, x + 1, cap - 3, SMOKE, 3)
    _track(p, foot[0], foot[1], (0.86, 0.97), 4)
    _oil(p, 20, 12, 1.6)
    _oil(p, 8, 10, 1.2)
    top = _engine_hall(p, -8, -6, 28, 11, 15)
    _, front, right, _ = top
    p.gable(top, 4, ENGINE_ROOF, BRICK)
    clere = p.hall(-8, -6 - 15 - 3, 24, 4, 3, shade(BRICK, 0.95), None)
    _patch(p, clere[0], clere[1], (0.03, 0.97), (0.6, 2.4), GLAZING)
    for t in (0.2, 0.4, 0.6, 0.8):
        p.line(
            [_on_wall(clere[0], clere[1], t, 0.6), _on_wall(clere[0], clere[1], t, 2.4)],
            shade(BRICK, 0.8),
        )
    p.gable(clere, 2, ENGINE_ROOF, BRICK)
    _gear(p, *_on_wall(front, right, 0.5, 6.5), 4, BRASS)
    _rails(p, (-26, 6), (-2, 18), 10)
    _locomotive(p, -12, 13.5)
    _flywheel(p, 0, 9, 4)
    _boiler(p, 18, 13)
    _lengths(p, -24, -2, (96, 100, 112), 3)
    _gantry(p, ((10, 16), (26, 8)), 22)
    p.rect(16, -13, 18.5, -11.5, (60, 64, 74))
    p.line([(17.2, -11.5), (17.2, -7)], (70, 64, 58))
    _wheel(p, 17.2, -4.1, 2.4, 3, shade(CASTING, 0.9))
    return p.sprite()


def big_saddler() -> Sprite:
    """A whole saddlery round a cobbled stable yard. At its back a saddle-tan stucco house of
    two storeys and an attic, 22 by 10 m, under a tiled roof between brick gable stacks: an
    oxblood shopfront with brass rails and lamplit windows of saddles at one end, big windows
    over the benches of the stitching and cutting rooms along the rest, and a black horse collar
    with brass hames hung over the yard on an iron bracket. Along the yard's side a brick stable
    block under slate with a louvred ventilator on its ridge, horses looking out over its
    oxblood half doors, and at its end the harness room, its gable to the street, with bridles
    and collars hung on pegs under a little pent roof. In the yard a bay is being saddled at a
    post, saddles stand on trestles, a stone trough and pump, and sides of leather are stacked
    at the workshop door."""
    m = METRE
    p = Painter(54, 32, (27, 21))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 17) * m, (v - 4.5) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon([at(8, -7.5), at(31, -7.5), at(31, 10), at(8, 10)], SADDLER_COBBLES)
    for k in range(1, 9):
        v = -7.5 + 17.5 * k / 9
        _bar(p, at(8, v), at(31, v), 0.05 * m, shade(SADDLER_COBBLES, 0.86))
    _bar(p, at(19, -7.5), at(19, 10), 0.6 * m, shade(SADDLER_COBBLES, 0.8))
    p.shadow(19, 6, 3)
    wall, high, rise = SADDLER_STUCCO, 8.4 * m, 4.4 * m
    _saddle_stack(p, *up(8.6, 15, 8.4 + 2.6), 1.4 * m, 4.4 * m, SADDLER_BRICK)
    foot = _footing(*at(19.5, 15), 23 * m, 10 * m)
    _bed(p, foot, margin=0.6)
    top = p.hall(*at(19.5, 15), 23 * m, 10 * m, high, wall, None)
    left, front, right, _ = top
    length = 23 * m
    _plinth(p, top, high, wall, 0.6 * m)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (0, 0.35 * m), shade(SADDLER_STONE, tone))
        _patch(p, a, b, (0, 1), (high - 4.3 * m, high - 4.05 * m), shade(SADDLER_STONE, tone))
    for k, u in enumerate((2.0, 5.6, 9.2)):
        _saddle_bench_window(
            p,
            left,
            front,
            length,
            u / 23,
            2.6 * m,
            (high - 3.4 * m, high - 0.8 * m),
            (SADDLE, SADDLE_TAN)[k % 2],
        )
        _saddle_bench_window(
            p,
            left,
            front,
            length,
            u / 23,
            2.6 * m,
            (0.9 * m, high - 5.0 * m),
            (SADDLE_TAN, SADDLE)[k % 2],
        )
    t0, t1 = 12.2 / 23, 22.4 / 23
    _patch(p, left, front, (t0, t1), (high - 4.0 * m, high), OXBLOOD)
    _patch(p, left, front, (t0, t1), (high - 4.0 * m, high - 3.3 * m), shade(OXBLOOD, 1.15))
    for k in range(12):
        t = t0 + 0.02 + 0.033 * k
        _patch(p, left, front, (t, t + 0.018), (high - 3.8 * m, high - 3.5 * m), BUCKLE)
    _bar(
        p,
        _on_wall(left, front, t0, high - 4.0 * m),
        _on_wall(left, front, t1, high - 4.0 * m),
        0.1 * m,
        BUCKLE,
    )
    _saddle_display(
        p, left, front, length, (t0 + 0.02, t0 + 0.17), (high - 3.1 * m, high - 0.75 * m), 0
    )
    _saddle_display(
        p, left, front, length, (t1 - 0.17, t1 - 0.02), (high - 3.1 * m, high - 0.75 * m), 1
    )
    tm = (t0 + t1) / 2
    _patch(p, left, front, (tm - 0.04, tm + 0.04), (high - 3.1 * m, high), shade(OXBLOOD, 0.8))
    _patch(
        p, left, front, (tm - 0.03, tm + 0.03), (high - 2.9 * m, high - 1.7 * m), SADDLER_LAMPLIT
    )
    for u in (13.4, 16.0, 19.8, 22.0):
        _saddle_sash(p, left, front, length, u / 23, (0.8 * m, 3.4 * m), 1.0)
    for t in (0.3, 0.7):
        _saddle_sash(p, front, right, 10 * m, t, (0.8 * m, 3.4 * m), 0.78)
        _saddle_sash(p, front, right, 10 * m, t, (high - 3.2 * m, high - 0.9 * m), 0.78)
    p.gable(top, rise, SADDLER_TILE, wall)
    _tiles(p, top, rise, SADDLER_TILE, (0.2, 0.4, 0.6, 0.8))
    for along in (0.22, 0.5, 0.78):
        dx, dy = _on_roof(top, rise, along, 0.35)
        dormer = p.hall(dx, dy + 0.4 * m, 1.8 * m, 1.6 * m, 1.6 * m, wall, None)
        _patch(p, dormer[0], dormer[1], (0.22, 0.78), (0.3 * m, 1.4 * m), shade(WINDOW, 1.05))
        _gable_across(p, dormer, 0.9 * m, SADDLER_TILE, wall)
    _saddle_stack(p, *up(30.6, 15, 8.4 + 2.2), 1.4 * m, 4.4 * m, SADDLER_BRICK)
    bx, by = _on_wall(left, front, 11.3 / 23, 1.0 * m)
    tip = _iso(bx, by, 0, -1.8 * m)
    _bar(p, (bx, by), tip, 0.12 * m, SADDLER_IRON)
    _bar(p, (bx, by + 0.6 * m), _iso(bx, by, 0, -0.9 * m), 0.08 * m, SADDLER_IRON)
    _saddle_collar(p, tip[0], tip[1] + 0.1 * m, wall, 2.0)
    stable_wall = SADDLER_BRICK
    stable = p.hall(*at(4, 2), 8 * m, 18 * m, 4.6 * m, stable_wall, None)
    st_left, st_front, st_right, _ = stable
    _plinth(p, stable, 4.6 * m, stable_wall, 0.5 * m)
    for a, b, tone in ((st_left, st_front, 1.0), (st_front, st_right, 0.78)):
        _patch(p, a, b, (0, 1), (0, 0.3 * m), shade(SADDLER_STONE, tone))
    for t, coat in (
        (0.38, (70, 46, 34)),
        (0.56, None),
        (0.74, (196, 168, 126)),
        (0.92, (40, 34, 32)),
    ):
        _saddle_stable_door(p, st_front, st_right, 18 * m, t, 4.6 * m, coat, 0.78)
    _patch(p, st_front, st_right, (0.06, 0.16), (1.6 * m, 4.6 * m), (34, 28, 26))
    _saddle_bridles(p, st_front, st_right, (0.17, 0.25), 1.2 * m, 3)
    for t in (0.275, 0.315):
        _saddle_collar(
            p, *_on_wall(st_front, st_right, t, 1.1 * m), shade(SADDLER_BRICK, 0.78), 1.3
        )
    eave = [_on_wall(st_front, st_right, t, 0.7 * m) for t in (0.16, 0.34)]
    p.polygon(
        [eave[0], eave[1], _iso(*eave[1], 1.0 * m, 0), _iso(*eave[0], 1.0 * m, 0)],
        shade(SADDLER_SLATE, 1.1),
        shade(SADDLER_SLATE, 0.6),
    )
    for e in eave:
        _bar(
            p,
            (e[0] + 1.0 * m, e[1] + 0.5 * m),
            (e[0] + 1.0 * m, e[1] + 3.9 * m),
            0.1 * m,
            SADDLER_SIGN,
        )
    _saddle_sash(p, st_left, st_front, 8 * m, 0.5, (1.0 * m, 3.0 * m), 1.0)
    _gable_across(p, stable, 3.4 * m, SADDLER_SLATE, stable_wall)
    _patch(p, st_left, st_front, (0.4, 0.6), (-2.0 * m, -0.4 * m), shade(OXBLOOD, 0.9))
    cx, cy = _on_wall(
        _on_wall(st_left, st_front, 0.5, -3.4 * m),
        _on_wall(stable[3], st_right, 0.5, -3.4 * m),
        0.55,
        0,
    )
    _saddle_cupola(p, cx, cy, SADDLER_STONE, SADDLER_SLATE)
    hx, hy = at(14.5, 3.0)
    _bar(p, at(12.2, 3.6), up(12.2, 3.6, 1.6), 0.16 * m, SADDLER_SIGN)
    _saddle_horse(p, hx, hy, SADDLER_BAY, -1, None)
    _saddle_trestle(p, *at(16.8, 1.0), SADDLE)
    _saddle_trestle(p, *at(25.2, 5.4), SADDLE_TAN)
    _saddle_trestle(p, *at(27.6, 4.2), SADDLE)
    _saddle_trough(p, *at(24.5, -3.2))
    for k, leather in enumerate((SADDLE_TAN, SADDLE, shade(SADDLE, 0.8))):
        p.hall(*up(10.2, 8.4, 0.25 * k), 2.2 * m, 1.4 * m, 0.25 * m, leather, shade(leather, 1.15))
    for u0, u1 in ((8.2, 17.6), (20.4, 30.8)):
        coping = p.hall(
            *up((u0 + u1) / 2, -7.6, 0),
            (u1 - u0) * m,
            0.45 * m,
            1.2 * m,
            SADDLER_BRICK,
            SADDLER_STONE,
        )
        _patch(p, coping[0], coping[1], (0, 1), (0, 0.15 * m), SADDLER_STONE)
    for u in (17.6, 20.4):
        p.hall(*at(u, -7.6), 0.8 * m, 0.8 * m, 2.4 * m, SADDLER_STONE, shade(SADDLER_STONE, 1.05))
        bx, by = up(u, -7.6, 2.4)
        p.blob(
            bx - 0.32 * m, by - 0.6 * m, bx + 0.32 * m, by + 0.02 * m, shade(SADDLER_STONE, 1.04)
        )
        p.blob(
            bx - 0.18 * m, by - 0.52 * m, bx + 0.02 * m, by - 0.3 * m, shade(SADDLER_STONE, 1.12)
        )
    return p.sprite()


def mega_saddler() -> Sprite:
    """A great leather goods house with its carriage works. The house is a palace of trade in
    pale limestone, 22 by 13 m, every storey different: a rusticated ground floor of round-headed
    shop windows framed in oxblood and brass showing saddles, trunks and bridles in the
    lamplight under a glass marquise over its doors; a piano nobile of tall arched windows on
    wrought-iron balconies; a floor of square workroom windows; a cornice and balustrade, and
    over them a slate mansard of round dormers, its middle rising to a pediment that carries a
    great gilt horse's head. Beside it a carriage showroom of iron and glass under a glazed
    vault, broughams standing in it; behind, a brick carriage works under a lantern roof with
    its smithy's iron stack smoking, and wheels leaning in the wheelwright's yard. On the paved
    forecourt between lamp posts a black brougham waits with a horse in polished harness."""
    m = METRE
    p = Painter(66, 44, (33, 29))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 25.75) * m, (v - 4.25) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    stone, glass = SADDLER_STONE, SADDLER_GLASS
    p.polygon(
        [
            at(-1, -8.5),
            at(23.5, -8.5),
            at(23.5, -3),
            at(49.5, -3),
            at(49.5, 15),
            at(44.5, 15),
            at(44.5, 0),
            at(-1, 0),
        ],
        SADDLER_PAVING,
    )
    for k in range(1, 7):
        v = -8.5 + 8.5 * k / 7
        _bar(p, at(-1, v), at(23.5 if v < -3 else 44.5, v), 0.05 * m, shade(SADDLER_PAVING, 0.88))
    p.polygon([at(44.5, 0), at(49.5, 0), at(49.5, 15), at(44.5, 15)], SADDLER_COBBLES)
    p.shadow(25, 7, 3)
    works = p.hall(*at(35.5, 19.5), 19 * m, 9 * m, 6 * m, SADDLER_BRICK, None)
    _, w_front, w_right, _ = works
    _plinth(p, works, 6 * m, SADDLER_BRICK, 0.5 * m)
    p.polygon(
        _round_head(w_front, w_right, 9 * m, 0.5, 4.2 * m, 0.9 * m, 6 * m), shade(stone, 0.78)
    )
    p.polygon(_round_head(w_front, w_right, 9 * m, 0.5, 3.6 * m, 1.2 * m, 6 * m), (40, 32, 30))
    p.gable(works, 2.8 * m, SADDLER_SLATE, SADDLER_BRICK)
    rb, rf = _ridge(works, 2.8 * m)
    lantern = p.hall(
        *_on_wall(rb, rf, 0.5, 0.5 * m), 14 * m, 2.4 * m, 1.6 * m, shade(stone, 0.9), None
    )
    for a, b, tone in ((lantern[0], lantern[1], 1.0), (lantern[1], lantern[2], 0.78)):
        _patch(p, a, b, (0.03, 0.97), (0.3 * m, 1.2 * m), shade(glass, 0.8 * tone))
    p.gable(lantern, 0.9 * m, SADDLER_SLATE, shade(stone, 0.9))
    sx, sy = up(41.5, 22.5, 0)
    _turned(p, sx, sy, [(0, 0.45 * m), (13 * m, 0.38 * m)], SADDLER_IRON)
    for k in (0.35, 0.7):
        _bar(
            p,
            (sx - 0.4 * m, sy - 13 * m * k),
            (sx + 0.4 * m, sy - 13 * m * k + 0.2 * m),
            0.1 * m,
            (70, 66, 66),
        )
    p.polygon(
        [(sx - 0.7 * m, sy - 13 * m), (sx + 0.7 * m, sy - 13 * m), (sx, sy - 13.8 * m)],
        SADDLER_IRON,
    )
    p.soft(
        [
            (sx - 0.6 * m, sy - 16 * m, sx + 1.4 * m, sy - 14 * m),
            (sx + 0.4 * m, sy - 18.4 * m, sx + 3.2 * m, sy - 15.8 * m),
        ],
        (200, 198, 202),
        0.75,
        0.4,
    )
    length, depth, high = 22 * m, 13 * m, 14 * m
    foot = _footing(*at(11, 6.5), length, depth)
    _bed(p, foot, margin=0.6)
    top = p.hall(*at(11, 6.5), length, depth, high, stone, None)
    left, front, right, _ = top
    for a, b, size, tone, bays in ((left, front, length, 1.0, 6), (front, right, depth, 0.78, 3)):
        for k in range(8):
            d = high - 0.65 * m * (k + 1)
            _bar(p, _on_wall(a, b, 0, d), _on_wall(a, b, 1, d), 0.06 * m, shade(stone, 0.78 * tone))
        _patch(p, a, b, (0, 1), (high - 5.8 * m, high - 5.1 * m), shade(OXBLOOD, tone))
        _patch(p, a, b, (0, 1), (high - 5.1 * m, high - 4.9 * m), shade(BUCKLE, tone))
        for k in range(bays * 4):
            t = (k + 0.5) / (bays * 4)
            if k % 4 in (1, 2):
                _patch(
                    p,
                    a,
                    b,
                    (t - 0.25 / (bays * 4), t + 0.25 / (bays * 4)),
                    (high - 5.6 * m, high - 5.3 * m),
                    shade(GILT, tone),
                )
        for d0, d1, tone2 in (
            (0, 0.9 * m, 1.06),
            (4.3 * m, 4.6 * m, 1.04),
            (high - 8.6 * m, high - 8.3 * m, 1.04),
        ):
            _patch(p, a, b, (0, 1), (d0, d1), shade(stone, tone * tone2))
        for k in range(bays):
            t = (k + 0.5) / bays
            entrance = a is left and k == 3
            if not entrance:
                _saddle_arcade(
                    p, a, b, size, t, 2.4 * m, (high - 4.4 * m, high - 0.2 * m), k % 3, tone
                )
            p.polygon(
                _round_head(a, b, size, t, 1.7 * m, high - 8.4 * m, high - 5.9 * m),
                shade(stone, 0.86 * tone),
            )
            p.polygon(
                _round_head(a, b, size, t, 1.4 * m, high - 8.2 * m, high - 6.0 * m),
                shade(WINDOW, tone),
            )
            _bar(
                p,
                _on_wall(a, b, t, high - 7.6 * m),
                _on_wall(a, b, t, high - 6.0 * m),
                0.06 * m,
                shade(stone, tone),
            )
            _saddle_balconette(p, a, b, size, t, 1.9 * m, high - 6.0 * m, tone)
            _patch(
                p,
                a,
                b,
                (t - 0.65 * m / size, t + 0.65 * m / size),
                (1.4 * m, 3.0 * m),
                shade(stone, 0.86 * tone),
            )
            _patch(
                p,
                a,
                b,
                (t - 0.5 * m / size, t + 0.5 * m / size),
                (1.55 * m, 2.9 * m),
                shade(WINDOW, tone),
            )
    t = 3.5 / 6
    p.polygon(
        _round_head(left, front, length, t, 2.8 * m, high - 4.6 * m, high), shade(OXBLOOD, 0.9)
    )
    p.polygon(_round_head(left, front, length, t, 2.2 * m, high - 4.2 * m, high), SADDLER_LAMPLIT)
    _patch(p, left, front, (t - 0.045, t + 0.045), (high - 2.6 * m, high), shade(OXBLOOD, 0.7))
    _bar(
        p,
        _on_wall(left, front, t, high - 2.6 * m),
        _on_wall(left, front, t, high),
        0.06 * m,
        BUCKLE,
    )
    marquise = [
        _on_wall(left, front, t - 0.09, high - 4.8 * m),
        _on_wall(left, front, t + 0.09, high - 4.8 * m),
    ]
    outer = [_iso(px, py, 0, -2.4 * m) for px, py in marquise]
    p.polygon(
        [
            marquise[0],
            marquise[1],
            (outer[1][0], outer[1][1] + 0.4 * m),
            (outer[0][0], outer[0][1] + 0.4 * m),
        ],
        shade(glass, 1.05),
        SADDLER_IRON,
    )
    _bar(
        p,
        (outer[0][0], outer[0][1] + 0.4 * m),
        (outer[1][0], outer[1][1] + 0.4 * m),
        0.14 * m,
        SADDLER_IRON,
    )
    for px, py in outer:
        _bar(p, (px, py + 0.4 * m), (px, py + 0.4 * m + 0.6 * m), 0.06 * m, SADDLER_IRON)
    _patch(p, left, front, (0, 1), (0, 0.5 * m), shade(stone, 1.05))
    _patch(p, front, right, (0, 1), (0, 0.5 * m), shade(stone, 0.8))
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (-0.6 * m, 0), shade(stone, 0.62 * tone))
        _bar(
            p,
            _on_wall(a, b, 0, -0.6 * m),
            _on_wall(a, b, 1, -0.6 * m),
            0.22 * m,
            shade(stone, 1.05 * tone),
        )
        for k in range(45):
            s = (k + 0.5) / 45
            _bar(
                p,
                _on_wall(a, b, s, -0.6 * m),
                _on_wall(a, b, s, 0),
                0.12 * m,
                shade(stone, 0.9 * tone),
            )
    roof_top = tuple((px, py - 0.6 * m) for px, py in (left, front, right, top[3]))
    inner = _broken_roof(p, roof_top, 3.4 * m, 1.3 * m, 1.2 * m, 0.2, SADDLER_SLATE)
    cx = (inner[0][0] + inner[2][0]) / 2
    cy = (inner[0][1] + inner[2][1]) / 2
    for along in (-6.5, 6.5):
        fx, fy = _iso(cx, cy, along * m, 0.6 * m)
        p.hall(fx, fy, 1.1 * m, 0.8 * m, 2.6 * m, stone, None)
        p.hall(fx, fy - 2.6 * m, 1.5 * m, 1.2 * m, 0.35 * m, shade(stone, 1.06), shade(stone, 1.1))
        p.hall(fx, fy - 2.95 * m, 0.9 * m, 0.6 * m, 0.3 * m, shade(stone, 0.9), (44, 38, 38))
    for k in range(6):
        if k == 3:
            continue
        s = (k + 0.5) / 6
        dx, dy = _on_wall(
            _on_wall(roof_top[0], inner[0], 0.5, 0), _on_wall(roof_top[1], inner[1], 0.5, 0), s, 0
        )
        dormer = p.hall(dx, dy + 0.5 * m, 1.6 * m, 1.2 * m, 1.8 * m, stone, None)
        d_left, d_front, d_right, d_back = dormer
        p.polygon(
            _round_head(d_left, d_front, 1.6 * m, 0.5, 0.9 * m, 0.25 * m, 1.6 * m),
            shade(WINDOW, 1.05),
        )
        crown = _on_wall(d_left, d_front, 0.5, -0.55 * m)
        p.polygon(
            [d_left, crown, d_front, d_right, (crown[0] + 1.2 * m, crown[1] - 0.6 * m), d_back],
            shade(SADDLER_SLATE, 1.15),
            shade(SADDLER_SLATE, 0.6),
        )
        _bar(p, d_left, crown, 0.12 * m, shade(stone, 1.04))
        _bar(p, crown, d_front, 0.12 * m, shade(stone, 1.04))
    pa, pb = _on_wall(left, front, 3.0 / 6, -0.6 * m), _on_wall(left, front, 4.0 / 6, -0.6 * m)
    apex = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 - 2.4 * m)
    p.polygon(
        [
            (pa[0], pa[1] + 0.6 * m),
            (pb[0], pb[1] + 0.6 * m),
            (pb[0], pb[1] - 0.6 * m),
            apex,
            (pa[0], pa[1] - 0.6 * m),
        ],
        stone,
        shade(stone, 0.6),
    )
    p.polygon(
        [
            (pa[0] + 0.5 * m, pa[1] - 0.3 * m),
            (pb[0] - 0.5 * m, pb[1] - 0.3 * m),
            (apex[0], apex[1] + 0.6 * m),
        ],
        shade(stone, 0.86),
    )
    _patch(p, left, front, (3.15 / 6, 3.85 / 6), (-0.4 * m, 0.4 * m), OXBLOOD)
    for k in range(5):
        s = 3.2 / 6 + k * 0.12 / 6
        _patch(p, left, front, (s + 0.01 / 6, s + 0.07 / 6), (-0.25 * m, 0.25 * m), GILT)
    p.hall(apex[0], apex[1] + 0.2 * m, 1.6 * m, 1.6 * m, 0.8 * m, stone, shade(stone, 1.05))
    _saddle_head(p, apex[0], apex[1] - 0.55 * m, 4.0 * m)
    show = p.hall(*at(34.5, 7.5), 20 * m, 11 * m, 7 * m, SADDLER_GREEN, None)
    h_left, h_front, h_right, _ = show
    for a, b, tone in ((h_left, h_front, 1.0), (h_front, h_right, 0.78)):
        _patch(p, a, b, (0, 1), (0.6 * m, 6.4 * m), shade((156, 128, 96), tone))
        _patch(p, a, b, (0, 1), (0.6 * m, 2.4 * m), shade(glass, 0.9 * tone))
        _patch(p, a, b, (0, 1), (5.2 * m, 6.4 * m), shade(SADDLER_LAMPLIT, 0.85 * tone))
    for s, coat in ((4.5, OXBLOOD), (12.5, SADDLER_LACQUER)):
        cx, cy = _on_wall(h_left, h_front, s / 20, 6.6 * m)
        body = shade(coat, 1.0)
        p.polygon(
            [
                (cx - 1.6 * m, cy - 1.2 * m),
                (cx - 1.7 * m, cy - 3.6 * m),
                (cx + 1.4 * m, cy - 3.6 * m),
                (cx + 1.8 * m, cy - 2.0 * m),
                (cx + 1.6 * m, cy - 1.2 * m),
            ],
            body,
        )
        p.polygon(
            [
                (cx - 1.2 * m, cy - 3.3 * m),
                (cx - 0.2 * m, cy - 3.3 * m),
                (cx - 0.2 * m, cy - 2.5 * m),
                (cx - 1.2 * m, cy - 2.5 * m),
            ],
            (120, 140, 150),
        )
        _spoked_wheel(p, cx - 1.2 * m, cy - 1.0 * m, 1.0 * m)
        _spoked_wheel(p, cx + 1.4 * m, cy - 0.8 * m, 0.8 * m)
    for a, b, panes, tone in ((h_left, h_front, 14, 1.0), (h_front, h_right, 7, 0.78)):
        for k in range(panes + 1):
            t = k / panes
            _bar(
                p,
                _on_wall(a, b, t, 0),
                _on_wall(a, b, t, 7 * m),
                0.18 * m,
                shade(SADDLER_GREEN, tone),
            )
        for d in (0.6, 2.4, 5.2):
            _bar(
                p,
                _on_wall(a, b, 0, d * m),
                _on_wall(a, b, 1, d * m),
                0.12 * m,
                shade(SADDLER_GREEN, tone),
            )
        _patch(p, a, b, (0, 1), (6.4 * m, 7 * m), shade(SADDLER_GREEN, 0.9 * tone))
        _patch(p, a, b, (0, 1), (0, 0.4 * m), shade(stone, 0.8 * tone))
    _vault(
        p, show, 3.2 * m, glass, SADDLER_GREEN, tuple(k / 10 for k in range(1, 10)), SADDLER_GREEN
    )
    for u in (1.0, 21.0):
        _lamp_post(p, *at(u, -2.2), 4.5, 2)
    _saddle_carriage(p, *at(8.0, -4.8), (62, 40, 32))
    for u, v in ((46.0, 4.0), (46.6, 6.2), (47.2, 8.4)):
        wx, wy = up(u, v, 1.1)
        _spoked_wheel(p, wx, wy, 1.1 * m)
    _saddle_trestle(p, *at(47.5, 11.5), SADDLE_TAN)
    return p.sprite()
