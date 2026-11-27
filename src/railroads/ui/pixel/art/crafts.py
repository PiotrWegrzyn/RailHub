"""Town crafts and the town's grand buildings: potters, tile works, foundries,
sausage makers, the makers of luxury goods, manors, theatres and hospitals."""

import numpy as np

from railroads.ui.pixel.art.churches import _ribs
from railroads.ui.pixel.art.commerce import PAVING, _lamp_post
from railroads.ui.pixel.art.homes import _chimney_on, _foot
from railroads.ui.pixel.art.industry import _round_head
from railroads.ui.pixel.art.painter import (
    BRICK,
    GLASS,
    LEAVES,
    METAL,
    RGB,
    TRUNK,
    VERDIGRIS,
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
    _bands,
    _bar,
    _gable_across,
    _iso,
    _on_wall,
    _patch,
    _plinth,
    _ridge,
    _tiles,
    _turned,
    _window,
    _windows,
)

ROOF: RGB = (150, 70, 52)
MOLTEN: RGB = (240, 128, 36)
SLATE: RGB = (88, 84, 92)


def _shop(p: Painter, x: float, y: float, wall: RGB, roof: RGB, sign: RGB) -> None:
    """A small workshop: a door, a window, a sign over the door, a tiled roof and a chimney."""
    top = p.box(x, y, 5, 4, wall, None)
    left, front, right, _ = top
    _patch(p, left, front, (0.45, 0.65), (1.3, 4), DOOR)
    _window(p, front, right, (0.3, 0.6), 1.0, shade(WINDOW, 0.8))
    p.line([_on_wall(left, front, 0.35, 0.8), _on_wall(left, front, 0.75, 0.8)], sign)
    _foot(p, top, 4)
    p.gable(top, 3, roof, wall)
    _tiles(p, top, 3, roof)
    _chimney_on(p, top, 3, 0.3, 1.4)


def _hall(p: Painter, wall: RGB, roof: RGB, length: float = 16) -> tuple:
    top = p.hall(0, 0, length, 8, 6, wall, None)
    left, front, _, _ = top
    for f in (0.15, 0.4, 0.65, 0.88):
        _patch(p, left, front, (f - 0.05, f + 0.05), (2, 4), shade(WINDOW, 0.85))
    _foot(p, top, 6)
    p.gable(top, 3.5, roof, wall)
    return top


def potter() -> Sprite:
    """A potter's cottage washed terracotta like its pots, with a round brick kiln beside it and
    pots set out to dry."""
    p = Painter(26, 22, (12, 16))
    p.shadow(9, 2.5)
    _shop(p, -3, 0, (204, 128, 88), ROOF, (176, 104, 70))
    p.ellipse(4, -6, 11, 2, shade(BRICK, 0.9))
    p.rect(6, -9, 9, -5, shade(BRICK, 0.8))
    for x in (-10, -7.5, 9, 11.5):
        p.ellipse(x - 1, 2.5, x + 1, 5, (176, 104, 70))
    return p.sprite()


def tile_works() -> Sprite:
    """A brick works shed with a bottle kiln and stacks of glazed blue stove tiles."""
    p = Painter(30, 26, (14, 19))
    p.shadow(12, 3)
    _hall(p, BRICK, SLATE, 14)
    p.ellipse(8, -12, 13, -2, shade(BRICK, 0.85))
    p.rect(9.5, -16, 11.5, -10, shade(BRICK, 0.75))
    for x in (-11, -8):
        p.rect(x, 2, x + 2.4, 5, (70, 120, 170))
        p.line([(x, 3.5), (x + 2.4, 3.5)], (236, 236, 226))
    return p.sprite()


def foundry() -> Sprite:
    """A soot-dark foundry hall with its door glowing, a tall iron cupola furnace beside it with
    fire and sparks at its top, molten iron running from its tap into a ladle and castings laid
    out on the sand in front."""
    p = Painter(30, 28, (14, 20))
    p.shadow(12, 3)
    top = _hall(p, shade(BRICK, 0.75), SLATE, 15)
    left, front, _, _ = top
    _patch(p, left, front, (0.45, 0.6), (2, 6), MOLTEN)
    p.rect(9, -18, 11, -4, (60, 60, 64))
    p.point(10, -19, MOLTEN)
    p.blob(-12, 3, -4, 6, (214, 196, 146))
    for x in (-10, -7):
        p.ellipse(x - 1.2, 3.4, x + 1.2, 5.4, (74, 76, 82))
    iron = (66, 62, 64)
    p.rect(11, -9, 15, 5, iron)
    for y in (-5, 0):
        p.line([(11, y), (15, y)], shade(iron, 1.4))
    p.ellipse(10.6, -10.6, 15.4, -8.4, MOLTEN)
    p.blob(11.8, -10, 14.2, -9, (255, 226, 120))
    for x, y in ((12, -13), (14.5, -14.5), (13, -16.5), (11, -15)):
        p.point(x, y, (255, 200, 80))
    p.polygon([(11, 3), (8, 4.6), (8.6, 5.4), (11, 4.4)], (40, 36, 36))
    p.blob(6, 4.4, 9.4, 6.6, MOLTEN)
    p.point(7.4, 5.2, (255, 226, 120))
    return p.sprite()


def sausage_maker() -> Sprite:
    """A butcher's shop washed the red of smoked ham, with a smokehouse chimney and sausages hung
    over its door."""
    p = Painter(24, 22, (11, 16))
    p.shadow(9, 2.5)
    _shop(p, -1, 0, (190, 106, 92), ROOF, (130, 80, 50))
    p.rect(6, -9, 8, 0, shade(BRICK, 0.8))
    for x in (-6, -4.5, -3):
        p.line([(x, -1), (x, 1.5)], (170, 74, 54))
    return p.sprite()


def _luxury_shop(sign: RGB, goods: RGB) -> Painter:
    p = Painter(26, 26, (14, 20))
    p.shadow(8, 2.5)
    top = p.box(0, 0, 5, 6, (214, 200, 176), None)
    left, front, right, _ = top
    _patch(p, left, front, (0.15, 0.85), (2, 4.6), shade(WINDOW, 0.9))
    for f in (0.3, 0.55):
        p.point(*_on_wall(left, front, f, 3.4), goods)
    _patch(p, left, front, (0.15, 0.85), (5, 5.8), sign)
    _window(p, front, right, (0.3, 0.6), 1.2, shade(WINDOW, 0.8))
    _foot(p, top, 6)
    p.gable(top, 3, SLATE, (214, 200, 176))
    return p


def carpet_weaver() -> Sprite:
    """A townhouse workshop with a red sign and carpets in reds and blues hung in its window, and
    two patterned rugs hung over a line beside it."""
    p = _luxury_shop((150, 40, 40), (60, 90, 170))
    pole = (110, 84, 60)
    rugs = (((-10, -0.5), (176, 40, 44), (232, 190, 80)), ((-6, 1.5), (50, 80, 160), (220, 70, 60)))
    for x, y in ((-12.5, -1.5), (-3.5, 3)):
        p.line([(x, y), (x, y - 8)], pole)
    p.line([(-12.5, -9.5), (-3.5, -5)], shade(pole, 1.2))
    for (x, y), field, border in rugs:
        p.polygon([(x - 2, y - 8), (x + 2, y - 6), (x + 2, y), (x - 2, y - 2)], border)
        p.polygon(
            [(x - 1.2, y - 6.8), (x + 1.2, y - 5.6), (x + 1.2, y - 1), (x - 1.2, y - 2.2)], field
        )
        p.point(x, y - 4, border)
    return p.sprite()


def clock_maker() -> Sprite:
    """A townhouse shop with a brass sign and a clock face showing in its window, under a little
    brass clock turret with a big white face and a gilt spire, the colours of its clocks."""
    brass = (190, 150, 68)
    p = _luxury_shop(brass, (232, 190, 70))
    p.rect(-2.5, -15, 2.5, -9, brass)
    p.polygon([(-3.2, -15), (0, -19), (3.2, -15)], (214, 176, 80), shade(brass, 0.6))
    p.ellipse(-2, -14.4, 2, -10.4, (246, 244, 236))
    p.line([(0, -12.4), (0, -14)], (30, 30, 34))
    p.line([(0, -12.4), (1.2, -12.4)], (30, 30, 34))
    return p.sprite()


def mirror_maker() -> Sprite:
    """A townhouse shop with a gilded sign and mirrors glinting in its window, and tall mirrors
    in gilt frames leant against its wall glinting in the sun."""
    p = _luxury_shop((200, 160, 60), GLASS)
    gilt = (214, 170, 60)
    for x, y in ((-9, 1), (-6, 2.5)):
        p.rect(x - 1.6, y - 6, x + 1.6, y, gilt)
        p.rect(x - 0.8, y - 5.2, x + 0.8, y - 0.8, (176, 214, 236))
        p.line([(x - 0.6, y - 2), (x + 0.6, y - 4.4)], (250, 252, 255))
    p.point(-10.5, -6.5, (255, 255, 255))
    return p.sprite()


MANOR_ROOF: RGB = (158, 72, 54)
SHUTTER: RGB = (58, 96, 70)
LAWN: RGB = (104, 156, 74)
GRAVEL: RGB = (214, 204, 178)
BLOOMS: tuple[RGB, ...] = ((214, 70, 84), (244, 214, 96), (236, 236, 240))


def _broken_roof(
    p: Painter, top: Top, low: float, step: float, rise: float, inset: float, roof: RGB
) -> Top:
    """A Polish broken roof on a block's top: steep slopes rising `low` all round to a break
    `step` in from the eaves, then a gentler hipped roof `rise` high over the rest, cut back
    `inset` of its length at either end. Returns the corners of the break."""
    left, front, right, back = top
    inner = (
        (left[0] + 2 * step, left[1] - low),
        (front[0], front[1] - step - low),
        (right[0] - 2 * step, right[1] - low),
        (back[0], back[1] + step - low),
    )
    i_left, i_front, i_right, i_back = inner
    edge = shade(roof, 0.55)
    p.polygon([back, right, i_right, i_back], shade(roof, 0.66), edge)
    p.polygon([left, back, i_back, i_left], shade(roof, 1.05), edge)
    p.polygon([left, front, i_front, i_left], shade(roof, 0.92), edge)
    for f in (0.33, 0.66):
        p.line([_on_wall(left, i_left, f, 0), _on_wall(front, i_front, f, 0)], shade(roof, 0.74))
    p.polygon([front, right, i_right, i_front], shade(roof, 0.7), edge)
    _hip(p, inner, rise, inset, shade(roof, 1.08), (0.45,))
    return inner


def _shuttered(
    p: Painter, a: Point, b: Point, length: float, t: float, span: tuple[float, float], glass: RGB
) -> None:
    """A tall window with white frames and glazing bars and green shutters folded back either
    side of it, about `t` along the wall under the edge from `a` to `b`, which is `length` units
    long, from `span[0]` down the wall to `span[1]`."""
    m = METRE
    half = 0.55 * m / length
    leaf = 0.5 * m / length
    _patch(p, a, b, (t - half - leaf, t - half), span, SHUTTER)
    _patch(p, a, b, (t + half, t + half + leaf), span, SHUTTER)
    _patch(p, a, b, (t - half, t + half), (span[0] - 0.1 * m, span[1] + 0.1 * m), WHITEWASH)
    _patch(p, a, b, (t - half * 0.8, t + half * 0.8), span, glass)
    _bar(p, _on_wall(a, b, t, span[0]), _on_wall(a, b, t, span[1]), 0.08 * m, WHITEWASH)
    middle = span[0] + (span[1] - span[0]) * 0.38
    _bar(p, _on_wall(a, b, t - half, middle), _on_wall(a, b, t + half, middle), 0.08 * m, WHITEWASH)


def _lime(p: Painter, x: float, y: float, k: int = 0) -> None:
    """An old lime tree some 9 m tall standing on (x, y): a dark trunk and a round crown of
    leaves heaped in lumps, lit on its top left."""
    m = METRE
    dark, mid, light = LEAVES[k % len(LEAVES)]
    p.blob(x - 2.6 * m, y - 0.8 * m, x + 3 * m, y + 1 * m, (62, 90, 44), 150)
    _bar(p, (x, y), (x, y - 3.6 * m), 0.5 * m, TRUNK)
    p.blob(x - 3 * m, y - 9.2 * m, x + 3 * m, y - 3 * m, dark)
    for dx, dy, r, colour in (
        (-1.2, -7.6, 1.8, mid),
        (1.1, -6.8, 1.9, mid),
        (-0.2, -8.4, 1.6, mid),
        (-1.5, -8.2, 1.0, light),
        (0.2, -8.9, 0.9, light),
        (-0.9, -6.4, 0.9, light),
    ):
        p.blob(x + (dx - r) * m, y + (dy - r) * m, x + (dx + r) * m, y + (dy + r) * m, colour)


def manor() -> Sprite:
    """A Polish manor house, a dwór, 18 by 10 m: whitewashed walls on a dark plinth, symmetrical
    tall windows with green shutters, a porch of four white columns under a triangular pediment
    with a round window in it, steps up to its door, a tall broken roof of red tiles with two
    white chimneys, a round lawn with a flower bed in a gravel drive before it and two old lime
    trees at its sides."""
    m = METRE
    p = Painter(34, 28, (17, 19))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 9) * m, (v - 2) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    _lime(p, *at(-2.5, 8), 1)
    _lime(p, *at(21, 9), 0)
    p.blob(*at(1.5, -8.8), *at(16.5, -8.8), GRAVEL)
    gx, gy = at(9, -7)
    p.blob(gx - 7.4 * m, gy - 3.7 * m, gx + 7.4 * m, gy + 3.7 * m, GRAVEL)
    p.blob(gx - 4.6 * m, gy - 2.3 * m, gx + 4.6 * m, gy + 2.3 * m, LAWN)
    p.blob(gx - 2 * m, gy - 1 * m, gx + 2 * m, gy + 1 * m, (94, 74, 54))
    for k, (dx, dy) in enumerate(
        ((-1.2, -0.2), (0, -0.6), (1.2, -0.2), (-0.6, 0.4), (0.6, 0.4), (0, 0))
    ):
        p.blob(
            gx + (dx - 0.35) * m,
            gy + (dy - 0.3) * m,
            gx + (dx + 0.35) * m,
            gy + (dy + 0.3) * m,
            BLOOMS[k % 3],
        )
    p.polygon([at(8, 0), at(10, 0), at(10, -3), at(8, -3)], GRAVEL)
    p.shadow(12, 3.5, 3)
    high = 5.5 * m
    top = p.hall(*at(9, 5), 18 * m, 10 * m, high, WHITEWASH, None)
    left, front, right, _ = top
    _plinth(p, top, high, (176, 170, 158), 0.7 * m)
    _foot(p, top, high)
    _bands(p, left, front, right, 0.3 * m, shade(WHITEWASH, 0.82), shade(WHITEWASH, 0.62))
    for u in (1.6, 4, 14, 16.4):
        _shuttered(p, left, front, 18 * m, u / 18, (1.3 * m, 3.8 * m), WINDOW)
    for t in (0.3, 0.7):
        _shuttered(p, front, right, 10 * m, t, (1.3 * m, 3.8 * m), shade(WINDOW, 0.8))
    porch = shade(WHITEWASH, 0.74)
    _patch(p, left, front, (6.2 / 18, 11.8 / 18), (0.9 * m, high - 0.7 * m), porch)
    for u in (7.3, 10.7):
        _shuttered(p, left, front, 18 * m, u / 18, (1.3 * m, 3.8 * m), shade(WINDOW, 0.85))
    _patch(p, left, front, (8.4 / 18, 9.6 / 18), (1.4 * m, high - 0.7 * m), DOOR)
    _bar(
        p,
        _on_wall(left, front, 0.5, 1.4 * m),
        _on_wall(left, front, 0.5, high - 0.7 * m),
        0.08 * m,
        shade(DOOR, 0.6),
    )
    _broken_roof(p, top, 2.6 * m, 1.6 * m, 3.4 * m, 0.3, MANOR_ROOF)
    for u in (4.5, 13.5):
        cx, cy = up(u, 5, 5.5 + 6)
        p.hall(cx, cy + 1.2 * m, 0.9 * m, 0.9 * m, 2.2 * m, WHITEWASH, shade(WHITEWASH, 0.9))
        p.hall(cx, cy - 1 * m, 1.1 * m, 1.1 * m, 0.3 * m, (70, 64, 62), (50, 46, 46))
    step = shade(WHITEWASH, 0.9)
    p.hall(*at(9, -1.5), 6 * m, 3 * m, 0.7 * m, step, shade(step, 1.06))
    p.hall(*at(9, -3.3), 3 * m, 0.6 * m, 0.35 * m, step, shade(step, 1.06))
    for v in (-0.6, -2.6):
        for u in (6.6, 11.4) if v > -1 else (6.6, 8.2, 9.8, 11.4):
            _column(p, *up(u, v, 0.7), 4.3 * m)
    ent = p.hall(*up(9, -1.5, 5), 6 * m, 3 * m, 0.7 * m, WHITEWASH, None)
    e_left, e_front, e_right, _ = ent
    _bands(p, e_left, e_front, e_right, 0.2 * m, shade(WHITEWASH, 0.85), shade(WHITEWASH, 0.65))
    _gable_across(p, ent, 2.4 * m, MANOR_ROOF, WHITEWASH)
    ox, oy = _on_wall(e_left, e_front, 0.5, -0.9 * m)
    p.ellipse(ox - 0.45 * m, oy - 0.5 * m, ox + 0.45 * m, oy + 0.5 * m, WINDOW)
    return p.sprite()


THEATRE_WALL: RGB = (238, 222, 186)
COLUMN: RGB = (248, 245, 236)
GILT: RGB = (228, 182, 70)
CURTAIN: RGB = (150, 34, 44)
POSTERS: tuple[RGB, ...] = ((204, 52, 48), (240, 200, 72), (64, 104, 178), (236, 232, 220))


def _hip(
    p: Painter, top: Top, rise: float, inset: float, roof: RGB, rows: tuple[float, ...] = ()
) -> None:
    """A hipped roof `rise` high on a block's top: its ridge runs along the lit wall, cut back
    `inset` of its length from either end, and its two ends slope down to the eaves too, with
    courses of tiles or seams across its lit side at each of `rows` of the way up it."""
    left, front, right, back = top
    near, far = _ridge(top, rise)
    ridge_back = _on_wall(near, far, inset, 0)
    ridge_front = _on_wall(near, far, 1 - inset, 0)
    edge = shade(roof, 0.55)
    p.polygon([back, right, ridge_front, ridge_back], shade(roof, 0.7), edge)
    p.polygon([left, back, ridge_back], shade(roof, 1.12), edge)
    p.polygon([left, front, ridge_front, ridge_back], roof, edge)
    for f in rows:
        p.line(
            [_on_wall(left, ridge_back, f, 0), _on_wall(front, ridge_front, f, 0)],
            shade(roof, 0.82),
        )
    p.polygon([front, right, ridge_front], shade(roof, 0.78), edge)
    p.line([ridge_back, ridge_front], shade(roof, 1.3))


def _column(p: Painter, x: float, y: float, tall: float, colour: RGB = COLUMN) -> None:
    """A round column of the classical orders standing on (x, y), `tall` units high and 0.8 m
    across, tapering a little, on a square base and under a square capital, lit on the left."""
    m = METRE
    r = 0.4 * m
    p.box(x, y, r * 1.5, 0.35 * m, colour, shade(colour, 1.04))
    foot, head, split = y - 0.35 * m, y - tall + 0.55 * m, x + r / 3
    p.polygon([(x - r, foot), (split, foot), (split, head), (x - 0.86 * r, head)], colour)
    p.polygon(
        [(split, foot), (x + r, foot), (x + 0.86 * r, head), (split, head)], shade(colour, 0.8)
    )
    _bar(p, (x - 0.45 * r, foot), (x - 0.4 * r, head), 0.1 * m, shade(colour, 1.08))
    p.box(x, y - tall + 0.55 * m, r * 1.6, 0.55 * m, colour, shade(colour, 1.04))


def _poster_pillar(p: Painter, x: float, y: float) -> None:
    """A round advertising pillar 3 m tall on (x, y), pasted with bright playbills and capped
    with a little green dome and a gilt knob."""
    m = METRE
    r, tall = 0.75 * m, 3 * m
    _turned(p, x, y, [(0, r), (tall, r)], POSTERS[3])
    for k, (t0, t1) in enumerate(((-0.95, -0.35), (-0.3, 0.3))):
        p.polygon(
            [
                (x + r * t0, y - 0.5 * m + r * 0.5 * (1 - t0**2) ** 0.5),
                (x + r * t1, y - 0.5 * m + r * 0.5 * (1 - t1**2) ** 0.5),
                (x + r * t1, y - 2.5 * m + r * 0.5 * (1 - t1**2) ** 0.5),
                (x + r * t0, y - 2.5 * m + r * 0.5 * (1 - t0**2) ** 0.5),
            ],
            POSTERS[k],
        )
    _turned(
        p,
        x,
        y - tall,
        [(0, r * 1.15), (0.25 * m, r * 1.15), (0.8 * m, 0.4 * r), (1 * m, 0)],
        VERDIGRIS,
    )
    p.blob(x - 0.15 * m, y - tall - 1.3 * m, x + 0.15 * m, y - tall - 1 * m, GILT)


def _playbills(p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float) -> None:
    """A board of playbills in a gilt frame on the wall under the edge from `a` to `b`, from
    `span[0]` to `span[1]` of the way along it, its top `drop` down it."""
    m = METRE
    t0, t1 = span
    _patch(p, a, b, span, (drop, drop + 2.4 * m), GILT)
    _patch(p, a, b, (t0 + 0.004, t1 - 0.004), (drop + 0.15 * m, drop + 2.25 * m), (46, 40, 44))
    for k in range(3):
        s0 = t0 + (t1 - t0) * (0.06 + 0.31 * k)
        _patch(p, a, b, (s0, s0 + (t1 - t0) * 0.26), (drop + 0.3 * m, drop + 2.1 * m), POSTERS[k])
        _patch(
            p,
            a,
            b,
            (s0 + (t1 - t0) * 0.04, s0 + (t1 - t0) * 0.22),
            (drop + 0.5 * m, drop + 0.8 * m),
            POSTERS[3],
        )


def theatre() -> Sprite:
    """A grand town theatre of cream stucco 18 by 18 m and 13 m high: a portico of six white
    columns under a pediment with a gilt lyre, its frieze lettered in gold, steps up to three red
    doors, tall arched windows over a rusticated ground floor, a balustrade and cornice, a
    hipped green copper roof crowned by a ribbed dome with a lantern, the tall stage house behind,
    lanterns either side of the steps, playbills in gilt frames and a poster pillar on the
    paved forecourt."""
    m = METRE
    p = Painter(45, 38, (22, 26))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 7) * m, (v - 9) * m)

    wall, high = THEATRE_WALL, 13 * m
    p.polygon([at(-5, -8), at(20, -8), at(20, 18), at(18, 18), at(18, 0), at(-5, 0)], PAVING)
    p.shadow(15, 5, 4)
    fly = p.hall(*at(9, 21.5), 14 * m, 7 * m, 18 * m, wall, None)
    _plinth(p, fly, 18 * m, wall, 0.8 * m)
    for drop in (0.5 * m, 6 * m):
        _bands(p, fly[0], fly[1], fly[2], drop, shade(wall, 0.9), shade(wall, 0.7))
    _windows(p, fly[1], fly[2], (0.3, 0.7), (2 * m,), 2.4 * m, shade(WINDOW, 0.8))
    p.gable(fly, 2.5 * m, VERDIGRIS, wall)
    _tiles(p, fly, 2.5 * m, VERDIGRIS)
    top = p.hall(*at(9, 9), 18 * m, 18 * m, high, wall, None)
    left, front, right, _ = top
    _plinth(p, top, high, wall, 0.8 * m)
    for drop in np.arange(8.6 * m, high - 0.8 * m, 0.9 * m):
        _bands(p, left, front, right, drop, shade(wall, 0.86), shade(wall, 0.66))
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (0, 0.6 * m), shade(wall, 0.84 * tone))
        _patch(p, a, b, (0, 1), (0.6 * m, 1.6 * m), shade(COLUMN, tone))
        for k in range(1, 40):
            _bar(
                p,
                _on_wall(a, b, k / 40, 0.75 * m),
                _on_wall(a, b, k / 40, 1.5 * m),
                0.1 * m,
                shade(wall, 0.7 * tone),
            )
        _patch(p, a, b, (0, 1), (2.4 * m, 2.8 * m), shade(wall, 0.84 * tone))
        _patch(p, a, b, (0, 1), (7.8 * m, 8.2 * m), shade(wall, 0.88 * tone))
    for t in (0.125, 0.875):
        p.polygon(_round_head(left, front, high, t, 2 * m, 3.6 * m, 7.4 * m), shade(wall, 0.8))
        p.polygon(_round_head(left, front, high, t, 1.6 * m, 3.8 * m, 7.3 * m), WINDOW)
        _playbills(p, left, front, (t - 0.08, t + 0.08), 9.4 * m)
    for k in range(6):
        t = (k + 0.5) / 6
        p.polygon(_round_head(front, right, high, t, 2 * m, 3.6 * m, 7.4 * m), shade(wall, 0.62))
        p.polygon(_round_head(front, right, high, t, 1.6 * m, 3.8 * m, 7.3 * m), shade(WINDOW, 0.8))
        p.polygon(
            _round_head(front, right, high, t, 1.4 * m, 9.2 * m, 11.6 * m), shade(WINDOW, 0.8)
        )
    _hip(p, top, 3.5 * m, 0.38, VERDIGRIS, (0.3, 0.6))
    dx, dy = at(9, 9)
    base = dy - 15.5 * m
    _turned(p, dx, base, [(0, 3.6 * m), (2.4 * m, 3.6 * m)], wall)
    for t in (-0.75, -0.25, 0.3):
        ox = dx + 3.6 * m * t
        oy = base - 1.2 * m + 1.8 * m * (1 - t * t) ** 0.5
        p.ellipse(ox - 0.35 * m, oy - 0.5 * m, ox + 0.35 * m, oy + 0.5 * m, shade(WINDOW, 1.1))
    dome = [
        (0, 4 * m),
        (0.4 * m, 4 * m),
        (1.6 * m, 3.7 * m),
        (2.9 * m, 2.9 * m),
        (3.9 * m, 1.7 * m),
        (4.5 * m, 0.6 * m),
    ]
    crown = base - 2.4 * m
    _turned(p, dx, crown, dome, VERDIGRIS)
    _ribs(p, dx, crown, dome, (1.1, 1.75, 2.4), shade(VERDIGRIS, 1.3))
    _turned(p, dx, crown - 4.4 * m, [(0, 0.7 * m), (1.2 * m, 0.7 * m), (1.7 * m, 0)], wall)
    p.blob(dx - 0.3 * m, crown - 6.5 * m, dx + 0.3 * m, crown - 5.9 * m, GILT)
    recess = shade(wall, 0.62)
    _patch(p, left, front, (0.25, 0.75), (1.6 * m, high), recess)
    for t in (0.36, 0.5, 0.64):
        p.polygon(_round_head(left, front, high, t, 2 * m, 8 * m, high - 1.2 * m), GILT)
        p.polygon(_round_head(left, front, high, t, 1.7 * m, 8.3 * m, high - 1.2 * m), CURTAIN)
        _bar(
            p,
            _on_wall(left, front, t, 9.4 * m),
            _on_wall(left, front, t, high - 1.2 * m),
            0.08 * m,
            shade(CURTAIN, 0.6),
        )

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    stone = shade(COLUMN, 0.92)
    p.hall(*at(9, -2), 10 * m, 4 * m, 1.2 * m, stone, shade(COLUMN, 0.96))
    carpet = [(0.0, -4.0, 1.2)]
    for k, v in enumerate((-4.7, -5.4)):
        p.hall(*at(9, v + 0.35), (10.6 + 1.2 * k) * m, 0.7 * m, (0.8 - 0.4 * k) * m, stone, None)
        carpet.append((v + 0.7, v, 0.8 - 0.4 * k))
    for near, far, z in carpet:
        p.polygon([up(8.2, near, z), up(9.8, near, z), up(9.8, far, z), up(8.2, far, z)], CURTAIN)
        p.polygon(
            [up(8.2, far, z), up(9.8, far, z), up(9.8, far, z - 0.4), up(8.2, far, z - 0.4)],
            shade(CURTAIN, 0.8),
        )
    p.polygon([up(8.2, -5.4, 0), up(9.8, -5.4, 0), up(10, -7.2, 0), up(8, -7.2, 0)], CURTAIN)
    for v in (-1.2, -3.4):
        for u in (5, 13) if v > -2 else (5, 6.6, 8.2, 9.8, 11.4, 13):
            _column(p, *up(u, v, 1.2), 8.4 * m)
    ent = p.hall(*up(9, -2, 9.6), 10 * m, 4 * m, 1.6 * m, wall, None)
    e_left, e_front, e_right, _ = ent
    _bands(p, e_left, e_front, e_right, 0.25 * m, COLUMN, shade(COLUMN, 0.8))
    _patch(p, e_left, e_front, (0.2, 0.8), (0.65 * m, 1.15 * m), GILT)
    _gable_across(p, ent, 2.8 * m, VERDIGRIS, wall)
    lx, ly = _on_wall(e_left, e_front, 0.5, -0.9 * m)
    p.ellipse(lx - 0.8 * m, ly - 0.9 * m, lx + 0.8 * m, ly + 0.5 * m, GILT)
    p.ellipse(lx - 0.45 * m, ly - 0.55 * m, lx + 0.45 * m, ly + 0.2 * m, wall)
    for x, y in (_on_wall(e_left, e_front, 0.5, -2.8 * m), e_left, e_front, left, front):
        p.polygon([(x - 0.35 * m, y), (x + 0.35 * m, y), (x, y - 1.2 * m)], GILT)
        p.blob(x - 0.3 * m, y - 1.5 * m, x + 0.3 * m, y - 0.9 * m, GILT)
    for x, y in (at(3.2, -6.2), at(14.8, -6.2)):
        _lamp_post(p, x, y, 4, 2)
    _poster_pillar(p, *at(-3, -5))
    return p.sprite()


def hospital() -> Sprite:
    """A long whitewashed hospital of two storeys with rows of windows, a red cross over its
    door and a flag on its roof."""
    p = Painter(32, 28, (15, 20))
    p.shadow(13, 3.5)
    top = p.hall(0, 0, 20, 8, 8, WHITEWASH, None)
    left, front, _, _ = top
    for f in (0.1, 0.22, 0.34, 0.66, 0.78, 0.9):
        for height in ((2, 3.4), (5, 6.4)):
            _patch(p, left, front, (f - 0.03, f + 0.03), height, GLASS)
    _patch(p, left, front, (0.45, 0.55), (1.5, 4.5), DOOR)
    x, y = _on_wall(left, front, 0.5, 6.4)
    p.rect(x - 1.5, y - 0.5, x + 1.5, y + 0.5, (200, 40, 40))
    p.rect(x - 0.5, y - 1.5, x + 0.5, y + 1.5, (200, 40, 40))
    _foot(p, top, 8)
    p.gable(top, 3, SLATE, WHITEWASH)
    p.line([(8, -14), (8, -8)], METAL)
    p.rect(8, -14, 11, -12, (200, 40, 40))
    return p.sprite()


# The dark timber, pale plaster and black tiles of the workshops of East Asia.
DARK_TIMBER: RGB = (112, 82, 58)
PLASTER: RGB = (236, 230, 214)
BLACK_TILE: RGB = (58, 60, 68)


def _vat(p: Painter, x: float, y: float, half: float = 2.5, tall: float = 4) -> None:
    """A big wooden vat standing on (x, y), bound with two hoops of split bamboo."""
    wood: RGB = (150, 104, 62)
    p.rect(x - half, y - tall, x + half, y, wood)
    p.ellipse(x - half, y - tall - 1, x + half, y - tall + 1, shade(wood, 0.62))
    for drop in (1, tall - 1):
        p.line([(x - half, y - drop), (x + half, y - drop)], (176, 160, 96))


def soy_sauce_brewery() -> Sprite:
    """A dark-timbered brewhouse under black tiles, with big hooped vats of soy mash at its side
    and a cedar ball for a sign over its door."""
    p = Painter(28, 26, (13, 19))
    p.shadow(11, 3)
    _shop(p, -2, 0, DARK_TIMBER, BLACK_TILE, (236, 226, 200))
    _vat(p, 7, 4)
    _vat(p, 10.5, 2, 2, 3.5)
    p.blob(-6.5, -2.5, -4.5, -0.5, (96, 112, 52))
    return p.sprite()


def tofu_maker() -> Sprite:
    """A small plastered tofu shop with a dark sign, steam rising from its chimney, and white
    blocks of tofu soaking in a tub of water by its door."""
    p = Painter(24, 24, (11, 18))
    p.shadow(9, 2.5)
    _shop(p, -1, 0, PLASTER, BLACK_TILE, (60, 50, 44))
    p.blob(3, -13, 6, -10, (232, 234, 238))
    p.blob(4, -16, 7, -13, (220, 222, 228))
    tub: RGB = (150, 104, 62)
    p.ellipse(-10, 1, -4, 4, tub)
    p.ellipse(-9, 1.4, -5, 3.2, (150, 196, 220))
    for x in (-8.2, -6.6):
        p.rect(x, 1.8, x + 1, 2.6, (250, 248, 238))
    return p.sprite()


def miso_maker() -> Sprite:
    """A storehouse washed the warm brown of miso, on a dark skirt, where it ripens, its tubs of it
    weighted down with stones outside and a sack of rice by the door."""
    p = Painter(26, 24, (12, 18))
    p.shadow(10, 2.5)
    _shop(p, -1, 0, (196, 146, 92), BLACK_TILE, (150, 104, 62))
    for x, y in ((-9, 3), (-5.5, 4.5)):
        _vat(p, x, y, 2, 2.5)
        p.blob(x - 1.2, y - 4, x + 1.2, y - 2.4, (150, 150, 146))
    p.ellipse(7, 2, 10, 5, (226, 216, 190))
    return p.sprite()


def fish_dryer() -> Sprite:
    """A boarded shed by the shore with racks of split fish hung out to dry on poles before it."""
    p = Painter(28, 22, (13, 16))
    p.shadow(11, 2.5)
    _shop(p, 3, -1, (140, 110, 80), SLATE, (110, 140, 170))
    pole: RGB = (120, 84, 50)
    fish: RGB = (190, 196, 200)
    for x0, y0 in ((-12, 0), (-6, 3)):
        p.line([(x0, y0), (x0, y0 - 6)], pole)
        p.line([(x0 + 5, y0 + 2.5), (x0 + 5, y0 - 3.5)], pole)
        p.line([(x0, y0 - 6), (x0 + 5, y0 - 3.5)], shade(pole, 1.2))
        for k in range(4):
            fx, fy = x0 + 0.8 + 1.2 * k, y0 - 5.4 + 0.6 * k
            p.line([(fx, fy), (fx, fy + 2.4)], fish if k % 2 else shade(fish, 0.8))
    return p.sprite()
