"""Town crafts and the town's grand buildings: potters, tile works, foundries,
sausage makers, the makers of luxury goods, manors, theatres and hospitals."""

import math

import numpy as np

from railroads.ui.pixel.art.churches import _ribs
from railroads.ui.pixel.art.commerce import PAVING, _lamp_post
from railroads.ui.pixel.art.homes import _chimney_on, _foot, _outward
from railroads.ui.pixel.art.industry import (
    DRESSING,
    _footing,
    _mill_window,
    _round_head,
    _tall_chimney,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    GLASS,
    LEAVES,
    METAL,
    RGB,
    TIMBER,
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
    BEAST,
    DOOR,
    METRE,
    _bands,
    _bar,
    _fence,
    _gable_across,
    _iso,
    _on_wall,
    _patch,
    _pig,
    _plinth,
    _ridge,
    _tiles,
    _turned,
    _vault,
    _window,
    _windows,
)

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


WOOD_SMOKE: RGB = (220, 220, 226)


def _craft_beam(
    p: Painter,
    a: Point,
    b: Point,
    start: tuple[float, float],
    end: tuple[float, float],
    wide: float,
    colour: RGB,
) -> None:
    """A straight timber or bar `wide` thick on the wall under the edge from `a` to `b`, from
    `start` to `end`, each a point so far along the wall and so far down it."""
    _bar(p, _on_wall(a, b, *start), _on_wall(a, b, *end), wide, colour)


def _craft_band(p: Painter, top: Top, drop: float, wide: float, colour: RGB) -> None:
    """A band `wide` deep round a block's lit wall in `colour` and its shaded wall a shade
    darker, its middle `drop` down from its top, such as a string course or a hoop."""
    left, front, right, _ = top
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (drop - wide / 2, drop + wide / 2), shade(colour, tone))


def _craft_casement(
    p: Painter,
    a: Point,
    b: Point,
    length: float,
    t: float,
    wide: float,
    span: tuple[float, float],
    glass: RGB,
    frame: RGB = WHITEWASH,
) -> None:
    """A casement window `wide` across about `t` along the wall under the edge from `a` to `b`,
    which is `length` units long, from `span[0]` down the wall to `span[1]`: four panes in a
    light frame."""
    m = METRE
    half, rim = wide / 2 / length, 0.12 * m / length
    _patch(
        p, a, b, (t - half - rim, t + half + rim), (span[0] - 0.12 * m, span[1] + 0.12 * m), frame
    )
    _patch(p, a, b, (t - half, t + half), span, glass)
    _craft_beam(p, a, b, (t, span[0]), (t, span[1]), 0.08 * m, frame)
    middle = (span[0] + span[1]) / 2
    _craft_beam(p, a, b, (t - half, middle), (t + half, middle), 0.08 * m, frame)


def _craft_hoop(p: Painter, x: float, y: float, half: float, wide: float, colour: RGB) -> None:
    """A band `wide` deep round the front of an upright round thing `half` across at its middle
    (x, y), such as an iron hoop round a kiln."""
    arc = [
        (x + half * math.cos(a), y + half / 2 * math.sin(a)) for a in np.linspace(0, math.pi, 13)
    ]
    p.polygon(arc + [(ax, ay - wide) for ax, ay in reversed(arc)], colour)


def _craft_mound(p: Painter, x: float, y: float, half: float, tall: float, colour: RGB) -> None:
    """A rounded heap of clay or sand `half` across and `tall` high on (x, y), lit on its left,
    with a few lumps showing on it."""
    split = math.acos(0.2)
    dome = [
        (x + half * math.cos(a), y - tall * math.sin(a) ** 0.8) for a in np.linspace(math.pi, 0, 15)
    ]
    front = [
        (x + half * math.cos(b), y + half / 2 * math.sin(b)) for b in np.linspace(0, math.pi, 13)
    ]
    p.polygon(dome + front, shade(colour, 0.76))
    lit_dome = [
        (x + half * math.cos(a), y - tall * math.sin(a) ** 0.8)
        for a in np.linspace(math.pi, split, 10)
    ]
    lit_front = [
        (x + half * math.cos(b), y + half / 2 * math.sin(b)) for b in np.linspace(split, math.pi, 8)
    ]
    p.polygon(lit_dome + lit_front, colour)
    p.blob(x - half * 0.6, y - tall * 0.95, x - half * 0.05, y - tall * 0.55, shade(colour, 1.1))
    for dx, dy, k in ((-0.5, -0.3, 1.1), (0.1, -0.15, 0.85), (0.45, -0.4, 0.7), (-0.2, -0.6, 0.9)):
        q = half * 0.1
        cx, cy = x + dx * half, y + dy * tall
        p.blob(cx - q, cy - q * 0.7, cx + q, cy + q * 0.7, shade(colour, k))


def _craft_smoke(p: Painter, x: float, y: float, size: float, puffs: int = 3) -> None:
    """Wood smoke seeping up from (x, y) and drifting to the right in soft puffs growing to
    `size` across."""
    blobs = []
    for k in range(puffs):
        r = size * (0.35 + 0.3 * k / max(1, puffs - 1))
        cx, cy = x + size * 0.45 * k, y - size * 0.8 * k
        blobs.append((cx - r, cy - r * 0.8, cx + r, cy + r * 0.8))
    p.soft(blobs, WOOD_SMOKE, 0.6, size * 0.15)


def _craft_log_pile(p: Painter, x: float, y: float, long: float, high: float) -> None:
    """A pile of split beech logs `long` along the lit walls and `high` tall on (x, y), stacked
    for the fire with their pale cut ends towards the front and dark gaps between them."""
    m = METRE
    bark = (78, 58, 44)
    left, front, _, _ = p.hall(x, y, long, 0.9 * m, high, bark, shade(bark, 1.25))
    rows = max(2, round(high / (0.38 * m)))
    per = max(3, round(long / (0.4 * m)))
    q = 0.19 * m
    for r in range(rows):
        shift = 0.5 * (r % 2)
        for k in range(per - (r % 2)):
            t = (k + 0.5 + shift) / per
            cx, cy = _on_wall(left, front, t, (r + 0.5) * high / rows)
            wood = (222, 190, 140) if (k + r) % 3 else (204, 166, 114)
            p.ellipse(cx - q, cy - q, cx + q, cy + q, wood)


TERRACOTTA: RGB = (198, 108, 68)
POTTER_WALL: RGB = (238, 216, 170)
PANTILE: RGB = (192, 94, 58)
KILN_BRICK: RGB = (160, 90, 64)
COBALT: RGB = (44, 80, 172)
CELADON: RGB = (72, 146, 96)
CREAM_GLAZE: RGB = (242, 232, 204)
GREENWARE: RGB = (224, 212, 188)
CLAY: RGB = (156, 150, 140)
BEATEN_EARTH: RGB = (180, 160, 122)
PLANK: RGB = (150, 112, 74)


def _craft_arch(x: float, y: float, wide: float, tall: float) -> list[Point]:
    """The outline of a round-headed opening `wide` across and `tall` high facing the viewer, the
    middle of its foot at (x, y)."""
    r = wide / 2
    arc = [
        (x + r * math.cos(a), y - tall + r - r * math.sin(a)) for a in np.linspace(0, math.pi, 9)
    ]
    return [(x - r, y), (x + r, y), *arc]


def _craft_pot(
    p: Painter, x: float, y: float, tall: float, colour: RGB, glazed: bool = False
) -> None:
    """A round-bellied pot `tall` high standing on (x, y), lit on the left, with a short neck and
    its dark mouth open at the top, and a glint of light on its glaze if `glazed`."""
    t = tall
    belly = 0.42 * t
    p.blob(x - belly, y - 0.8 * t, x + belly, y, shade(colour, 0.74))
    p.blob(x - belly, y - 0.8 * t, x + belly * 0.4, y - 0.06 * t, colour)
    p.polygon(
        [
            (x - 0.2 * t, y - 0.7 * t),
            (x + 0.2 * t, y - 0.7 * t),
            (x + 0.25 * t, y - t),
            (x - 0.25 * t, y - t),
        ],
        shade(colour, 0.9),
    )
    p.blob(x - 0.25 * t, y - 1.06 * t, x + 0.25 * t, y - 0.94 * t, shade(colour, 0.42))
    if glazed:
        p.blob(x - 0.3 * t, y - 0.66 * t, x - 0.12 * t, y - 0.42 * t, shade(colour, 1.7))


def _craft_trestle(p: Painter, x: float, y: float, long: float, high: float) -> Top:
    """A board `long` along the lit walls laid on two trestles `high` tall standing on (x, y);
    returns the top of the board."""
    m = METRE
    for t in (-0.4, 0.4):
        fx, fy = _iso(x, y, t * long, 0)
        for dv in (-0.3, 0.3):
            _bar(p, _iso(fx, fy, 0, dv * m), (fx, fy - high), 0.1 * m, shade(PLANK, 0.7))
    return p.hall(x, y - high, long, 0.7 * m, 0.12 * m, PLANK, shade(PLANK, 1.15))


def _craft_pantiles(p: Painter, top: Top, rise: float, roof: RGB) -> None:
    """The rolls of a roof of pantiles running down the lit side of a gable roof from its ridge
    to its eaves, with a row of ridge tiles along its top."""
    m = METRE
    left, front, _, _ = top
    ridge_back, ridge_front = _ridge(top, rise)
    rolls = max(4, round(abs(front[0] - left[0]) / (0.55 * m)))
    for k in range(rolls + 1):
        t = k / rolls
        _bar(
            p,
            _on_wall(ridge_back, ridge_front, t, 0),
            _on_wall(left, front, t, 0),
            0.13 * m,
            shade(roof, 0.82),
        )
    _bar(p, ridge_back, ridge_front, 0.3 * m, shade(roof, 0.9))
    _bar(p, left, front, 0.2 * m, shade(roof, 0.7))


def _craft_ware_stall(p: Painter, x: float, y: float) -> None:
    """A potter's stall 4.5 m long on (x, y): a high board at the back set out with big glazed
    plates standing on edge and a low one before it with jugs and pots, in cobalt blue, green,
    cream and terracotta."""
    m = METRE
    glazes = (COBALT, CELADON, CREAM_GLAZE, TERRACOTTA, COBALT, CELADON)
    for row, (v, high) in enumerate(((1.0, 1.4), (0, 0.8))):
        left, front, _, _ = _craft_trestle(p, *_iso(x, y, 0, v * m), 4.5 * m, high * m)
        count = 5
        for k in range(count):
            cx, cy = _on_wall(left, front, (k + 0.5) / count, 0)
            cx, cy = cx + 0.25 * m, cy - 0.1 * m
            colour = glazes[(k + row * 3) % len(glazes)]
            if row == 0:
                p.ellipse(cx - 0.5 * m, cy - 1.0 * m, cx + 0.5 * m, cy, colour)
                p.ellipse(
                    cx - 0.26 * m, cy - 0.76 * m, cx + 0.26 * m, cy - 0.24 * m, shade(colour, 1.22)
                )
                p.blob(cx - 0.36 * m, cy - 0.86 * m, cx - 0.2 * m, cy - 0.7 * m, shade(colour, 1.7))
            else:
                _craft_pot(p, cx, cy, (0.95 if k % 2 else 0.7) * m, colour, True)


def _craft_bottle_kiln(p: Painter, x: float, y: float) -> None:
    """A potter's bottle kiln of brick 6 m across and 8.6 m tall standing on (x, y): a squat
    belly bound with iron hoops narrowing to a sooted neck, an arched firing mouth glowing low at
    its front and a wisp of smoke from its top."""
    m = METRE
    profile = [
        (0, 2.8 * m),
        (1.6 * m, 2.95 * m),
        (3.4 * m, 2.75 * m),
        (5.2 * m, 2.0 * m),
        (6.6 * m, 1.15 * m),
        (8.2 * m, 0.85 * m),
        (8.6 * m, 0.95 * m),
    ]
    rim = (44, 36, 34)
    _turned(p, x, y, profile, KILN_BRICK, rim)
    neck = [(h - 5.6 * m, w) for h, w in profile if h >= 6.6 * m]
    _turned(p, x, y - 5.6 * m, [(0, 1.75 * m), *neck], shade(KILN_BRICK, 0.66), rim)
    for h, half in ((1.0 * m, 2.92 * m), (2.4 * m, 2.92 * m), (3.8 * m, 2.6 * m)):
        _craft_hoop(p, x, y - h, half, 0.16 * m, (64, 60, 62))
    mx, my = x - 0.9 * m, y + 1.32 * m
    p.polygon(_craft_arch(mx, my, 1.5 * m, 2.1 * m), shade(KILN_BRICK, 0.6))
    p.polygon(_craft_arch(mx, my, 1.0 * m, 1.7 * m), (56, 40, 34))
    p.blob(mx - 0.42 * m, my - 0.9 * m, mx + 0.42 * m, my, MOLTEN)
    p.blob(mx - 0.22 * m, my - 0.55 * m, mx + 0.16 * m, my - 0.12 * m, (255, 214, 120))
    _craft_smoke(p, x + 0.5 * m, y - 10 * m, 1.8 * m, 2)


def potter() -> Sprite:
    """A village pottery: a workshop 9 by 6 m washed in warm cream under a roof of terracotta
    pantiles, with green shutters, glazed pots on its window sill and its blue doors swung open on
    the potter's wheel inside, a pot half thrown on it before a shelf of unfired ware; behind it a
    brick bottle kiln bound with iron hoops, its mouth glowing and a wisp of smoke from its neck,
    and firewood stacked by it; boards of unfired pots drying in rows before the workshop, a heap
    of grey clay, and a stall of finished ware glazed cobalt blue, green and cream, with big jars
    standing by it."""
    m = METRE
    p = Painter(27, 20, (13, 13))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 9) * m, (v - 5.5) * m)

    p.polygon(
        [at(-1, 0), at(19.5, 0), at(19.5, 12), at(10, 12), at(10, 4.5), at(-1, 4.5)], BEATEN_EARTH
    )
    _craft_bottle_kiln(p, *at(13.5, 9.5))
    _craft_log_pile(p, *at(17.6, 8), 2.4 * m, 1.4 * m)
    high, rise = 5 * m, 4 * m
    top = p.hall(*at(4.5, 7.5), 9 * m, 6 * m, high, POTTER_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, high, (180, 156, 120), 0.5 * m)
    _foot(p, top, high)
    _craft_band(p, top, 0.2 * m, 0.4 * m, shade(POTTER_WALL, 0.84))
    _shuttered(p, left, front, 9 * m, 0.2, (1.2 * m, 3.1 * m), WINDOW)
    for t, colour in ((0.165, COBALT), (0.235, CELADON)):
        _craft_pot(p, *_on_wall(left, front, t, 3.2 * m), 0.55 * m, colour, True)
    _shuttered(p, front, right, 6 * m, 0.5, (1.2 * m, 3.1 * m), shade(WINDOW, 0.8))
    t0, t1 = 0.5, 0.76
    _patch(p, left, front, (t0 - 0.02, t1 + 0.02), (1.2 * m, high), shade(PLANK, 0.8))
    _patch(p, left, front, (t0, t1), (1.4 * m, high), (58, 44, 38))
    _patch(p, left, front, (t0, t1), (3.2 * m, high), (156, 116, 80))
    for k in range(3):
        sx, sy = _on_wall(left, front, t0 + (t1 - t0) * (0.25 + 0.25 * k), 2.6 * m)
        _craft_pot(p, sx, sy, 0.45 * m, GREENWARE)
    wx, wy = _on_wall(left, front, (t0 + t1) / 2, high - 0.3 * m)
    _bar(p, (wx, wy), (wx, wy - 0.8 * m), 0.16 * m, (52, 46, 44))
    p.ellipse(wx - 0.65 * m, wy - 1.05 * m, wx + 0.65 * m, wy - 0.7 * m, (74, 66, 62))
    _craft_pot(p, wx, wy - 0.9 * m, 0.7 * m, (206, 128, 86))
    leaf = (t1 - t0) / 2
    _patch(p, left, front, (t1, t1 + leaf), (1.4 * m, high), shade(COBALT, 0.9))
    for f in (0.33, 0.66):
        s0 = t1 + leaf * f
        _craft_beam(p, left, front, (s0, 1.4 * m), (s0, high), 0.06 * m, shade(COBALT, 0.6))
    out = _outward(left, front, leaf * 9 * m)
    a, b = _on_wall(left, front, t0, 1.4 * m), _on_wall(left, front, t0, high)
    a_out, b_out = (a[0] + out[0], a[1] + out[1]), (b[0] + out[0], b[1] + out[1])
    p.polygon([a, b, b_out, a_out], COBALT)
    for f in (0.33, 0.66):
        _bar(p, _on_wall(a, a_out, f, 0), _on_wall(b, b_out, f, 0), 0.06 * m, shade(COBALT, 0.6))
    p.gable(top, rise, PANTILE, POTTER_WALL)
    _craft_pantiles(p, top, rise, PANTILE)
    ridge_back, ridge_front = _ridge(top, rise)
    cx, cy = _on_wall(ridge_back, ridge_front, 0.8, 0)
    p.hall(cx, cy + 0.4 * m, 0.8 * m, 0.8 * m, 1.4 * m, KILN_BRICK, (60, 50, 48))
    _craft_mound(p, *at(10.2, 4.4), 1.5 * m, 1.3 * m, CLAY)
    for v in (2.8, 1.2):
        board, b_front, _, _ = _craft_trestle(p, *at(4.5, v), 6.5 * m, 0.9 * m)
        for k in range(10):
            px, py = _on_wall(board, b_front, (k + 0.5) / 10, 0)
            _craft_pot(p, px + 0.25 * m, py - 0.1 * m, (0.5 if k % 2 else 0.4) * m, GREENWARE)
    _craft_ware_stall(p, *at(13.5, 2.2))
    for u, v, tall, colour in (
        (17.4, 4.2, 1.4, COBALT),
        (18.4, 2.9, 1.1, CELADON),
        (17.0, 1.9, 0.9, CREAM_GLAZE),
        (18.2, 1.0, 1.2, COBALT),
    ):
        _craft_pot(p, *at(u, v), tall * m, colour, True)
    return p.sprite()


ROOF_TILE: RGB = (194, 98, 60)
RAW_TILE: RGB = (214, 178, 136)
STOVE_GREEN: RGB = (58, 136, 92)
TILE_WHITE: RGB = (244, 244, 238)
CREAM_CLAY: RGB = (232, 212, 162)
OFFICE_WALL: RGB = (236, 226, 204)
WORKS_BRICK: RGB = (164, 84, 58)
WORKS_YARD: RGB = (148, 132, 104)


def _craft_open_shed(
    p: Painter, x: float, y: float, length: float, depth: float, eaves: float, rise: float
) -> None:
    """An open drying shed `length` along the lit walls and `depth` deep centred at (x, y):
    a roof of red tiles on timber posts over racks of raw tiles drying on slatted shelves, open
    on every side to the wind, with weatherboards in its gable."""
    m = METRE
    left, front, right, back = _footing(x, y, length, depth)
    p.polygon([left, front, right, back], (112, 98, 76))
    posts = max(2, round(length / (3 * m)))
    post = shade(TIMBER, 0.75)

    def stand(a: Point, b: Point) -> None:
        for k in range(posts + 1):
            fx, fy = _on_wall(a, b, k / posts, 0)
            _bar(p, (fx, fy), (fx, fy - eaves), 0.22 * m, post)

    stand(back, right)
    for v in (0.68, 0.3):
        rx, ry = _on_wall(left, back, v, 0)
        fx, fy = _on_wall(front, right, v, 0)
        cx, cy = (rx + fx) / 2, (ry + fy) / 2
        rack = p.hall(cx, cy, length - 1.2 * m, 0.9 * m, eaves - 0.5 * m, (92, 72, 56), None)
        r_left, r_front, _, _ = rack
        shelves = 5
        for k in range(shelves):
            d0 = (eaves - 0.5 * m) * k / shelves + 0.1 * m
            d1 = d0 + (eaves - 0.5 * m) / shelves - 0.2 * m
            _patch(
                p,
                r_left,
                r_front,
                (0.01, 0.99),
                (d0, d1),
                RAW_TILE if k % 2 else shade(RAW_TILE, 0.92),
            )
            for t in np.linspace(0.02, 0.98, 26):
                _craft_beam(p, r_left, r_front, (t, d0), (t, d1), 0.05 * m, shade(RAW_TILE, 0.72))
        for t in np.linspace(0, 1, posts * 2 + 1):
            _craft_beam(p, r_left, r_front, (t, 0), (t, eaves - 0.5 * m), 0.14 * m, (84, 64, 50))
    stand(left, front)
    stand(front, right)
    top = (
        (left[0], left[1] - eaves),
        (front[0], front[1] - eaves),
        (right[0], right[1] - eaves),
        (back[0], back[1] - eaves),
    )
    _craft_beam(p, top[0], top[1], (0, 0), (1, 0), 0.3 * m, post)
    p.gable(top, rise, ROOF_TILE, shade(TIMBER, 0.9))
    _tiles(p, top, rise, ROOF_TILE, (0.25, 0.5, 0.75))
    _, ridge_front = _ridge(top, rise)
    for f in (0.2, 0.4, 0.6, 0.8):
        foot = _on_wall(top[1], top[2], f, 0)
        head = (
            _on_wall(top[1], ridge_front, 2 * f, 0)
            if f <= 0.5
            else _on_wall(ridge_front, top[2], 2 * f - 1, 0)
        )
        _bar(p, foot, head, 0.08 * m, shade(TIMBER, 0.65))


def _craft_tile_stack(p: Painter, x: float, y: float, colour: RGB, high: float) -> None:
    """A stack of finished roof tiles on a pallet at (x, y), 1.6 by 1.1 m and `high` tall: rows
    of tiles laid on edge, their curved ends showing in courses on its lit side."""
    m = METRE
    p.hall(x, y, 1.8 * m, 1.3 * m, 0.15 * m, shade(TIMBER, 0.8), shade(TIMBER, 0.95))
    top = p.hall(x, y - 0.15 * m, 1.6 * m, 1.1 * m, high, colour, shade(colour, 1.12))
    left, front, right, back = top
    rows = max(2, round(high / (0.3 * m)))
    for k in range(1, rows):
        _craft_band(p, top, high * k / rows, 0.07 * m, shade(colour, 0.68))
    for t in np.linspace(0.1, 0.9, 6):
        _bar(
            p,
            _on_wall(left, back, t, 0),
            _on_wall(front, right, t, 0),
            0.06 * m,
            shade(colour, 0.82),
        )


def _craft_hack(p: Painter, x: float, y: float, long: float) -> None:
    """A hack on (x, y): a low wall of raw tiles `long` along the lit walls set on edge in an open
    lattice to dry in the wind, under a cap of boards."""
    m = METRE
    high = 1.1 * m
    left, front, right, _ = p.hall(x, y, long, 0.8 * m, high, RAW_TILE, None)
    rows = 4
    for k in range(rows):
        d0 = high * k / rows
        for t in np.linspace(0.03, 0.97, round(long / (0.35 * m))):
            shift = 0.5 * (k % 2) * 0.35 * m / long
            _craft_beam(
                p,
                left,
                front,
                (t + shift, d0 + 0.05 * m),
                (t + shift, d0 + high / rows - 0.05 * m),
                0.08 * m,
                shade(RAW_TILE, 0.62),
            )
    _patch(p, front, right, (0, 1), (0, high), shade(RAW_TILE, 0.7))
    boards = (130, 112, 92)
    cap = p.hall(x, y - high, long + 0.3 * m, 1.3 * m, 0.12 * m, boards, None)
    p.gable(cap, 0.4 * m, boards, shade(boards, 0.9))


def _craft_tile_panel(
    p: Painter,
    a: Point,
    b: Point,
    length: float,
    span: tuple[float, float],
    drop: tuple[float, float],
) -> None:
    """A panel of glazed tiles set into the wall under the edge from `a` to `b`, which is `length`
    units long, from `span[0]` to `span[1]` along it and `drop[0]` to `drop[1]` down it: a cobalt
    border round white tiles each painted with a blue flower, a sampler of the works' best."""
    m = METRE
    t0, t1 = span
    d0, d1 = drop
    _patch(p, a, b, span, drop, COBALT)
    rim = 0.2 * m
    s0, s1 = t0 + rim / length, t1 - rim / length
    e0, e1 = d0 + rim, d1 - rim
    cols = max(2, round((s1 - s0) * length / (0.55 * m)))
    rows = max(2, round((e1 - e0) / (0.55 * m)))
    for i in range(cols):
        for j in range(rows):
            u0, u1 = s0 + (s1 - s0) * i / cols, s0 + (s1 - s0) * (i + 1) / cols
            v0, v1 = e0 + (e1 - e0) * j / rows, e0 + (e1 - e0) * (j + 1) / rows
            _patch(p, a, b, (u0, u1), (v0, v1), TILE_WHITE)
            cx, cy = _on_wall(a, b, (u0 + u1) / 2, (v0 + v1) / 2)
            q = 0.17 * m
            if (i + j) % 2:
                p.blob(cx - q, cy - q, cx + q, cy + q, COBALT)
                p.blob(cx - q / 2.5, cy - q / 2.5, cx + q / 2.5, cy + q / 2.5, TILE_WHITE)
            else:
                for dx, dy in ((-q, 0), (q, 0), (0, -q), (0, q)):
                    p.blob(
                        cx + dx - q / 2, cy + dy - q / 2, cx + dx + q / 2, cy + dy + q / 2, COBALT
                    )


def _craft_pug_mill(p: Painter, x: float, y: float) -> None:
    """A horse-turned pug mill on (x, y), where the clay is kneaded: a hooped wooden tub 1.6 m
    across and 1.7 m tall full of clay, an iron shaft standing out of it with a long sweep for the
    horse, a ring of trodden clay round it and a wad of kneaded clay pushed out at its foot."""
    m = METRE
    p.blob(x - 4.4 * m, y - 2.2 * m, x + 4.4 * m, y + 2.2 * m, (136, 118, 90))
    p.blob(x - 3.4 * m, y - 1.7 * m, x + 3.4 * m, y + 1.7 * m, WORKS_YARD)
    tub = (140, 102, 68)
    _turned(p, x, y, [(0, 0.8 * m), (1.7 * m, 0.75 * m)], tub, CREAM_CLAY)
    for h in (0.35 * m, 1.3 * m):
        _craft_hoop(p, x, y - h, 0.79 * m, 0.13 * m, (58, 56, 60))
    p.blob(x - 0.7 * m, y + 0.05 * m, x + 0.1 * m, y + 0.55 * m, CREAM_CLAY)
    iron = (56, 54, 58)
    _bar(p, (x, y - 1.7 * m), (x, y - 2.8 * m), 0.2 * m, iron)
    end = _iso(x, y - 2.5 * m, -3.8 * m, 0.4 * m)
    _bar(p, (x, y - 2.6 * m), end, 0.2 * m, shade(TIMBER, 0.8))
    _bar(p, end, (end[0], end[1] + 0.8 * m), 0.12 * m, shade(TIMBER, 0.7))
    _bar(
        p,
        (end[0] - 0.5 * m, end[1] + 0.8 * m),
        (end[0] + 0.5 * m, end[1] + 0.8 * m),
        0.12 * m,
        shade(TIMBER, 0.7),
    )


def _craft_kiln_block(p: Painter, x: float, y: float) -> None:
    """A tile kiln 8 by 6 m and 4 m high of brick standing on (x, y): a barrel vault of brick
    bound with iron ribs, a cream string course, two arched firing mouths in its lit wall, one of
    them glowing, and iron tie rods along its walls."""
    m = METRE
    high = 4 * m
    top = p.hall(x, y, 8 * m, 6 * m, high, WORKS_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, WORKS_BRICK, 0.5 * m)
    _craft_band(p, top, 0.45 * m, 0.3 * m, CREAM_GLAZE)
    soot = (70, 56, 50)
    for t, glow in ((0.3, False), (0.7, True)):
        p.polygon(
            _round_head(left, front, 8 * m, t, 1.6 * m, high - 2.2 * m, high),
            shade(WORKS_BRICK, 0.62),
        )
        p.polygon(_round_head(left, front, 8 * m, t, 1.1 * m, high - 1.9 * m, high), soot)
        if glow:
            p.polygon(_round_head(left, front, 8 * m, t, 0.8 * m, high - 1.4 * m, high), MOLTEN)
            fx, fy = _on_wall(left, front, t, high - 0.5 * m)
            p.blob(fx - 0.25 * m, fy - 0.35 * m, fx + 0.25 * m, fy + 0.1 * m, (255, 220, 130))
    for t in (0.1, 0.5, 0.9):
        _craft_beam(p, left, front, (t, 0.5 * m), (t, high - 0.6 * m), 0.14 * m, (60, 58, 60))
    _patch(p, front, right, (0.3, 0.7), (high - 1.8 * m, high), shade(soot, 0.8))
    _vault(p, top, 1.6 * m, shade(WORKS_BRICK, 0.8), WORKS_BRICK, rib=(70, 64, 64))


def tile_works() -> Sprite:
    """A roof-tile and stove-tile works: a long drying shed 18 by 6 m open to the wind under a
    roof of its own red tiles, with racks of raw tiles drying on its shelves; a brick kiln with
    a glowing firing mouth under an earthed vault and a tall round chimney smoking; an office of
    cream render under a roof of green glazed tiles with a panel of blue-and-white glazed tiles
    set in its wall; a heap of cream clay, a horse-turned pug mill, two hacks of raw tiles
    drying under caps of boards, and pallets of finished roof tiles and green stove tiles stacked
    in the yard."""
    m = METRE
    p = Painter(38, 32, (19, 22))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 14.5) * m, (v - 7.5) * m)

    p.polygon(
        [at(-0.5, -1), at(29.5, -1), at(29.5, 15.5), at(-0.5, 15.5)],
        WORKS_YARD,
        shade(WORKS_YARD, 0.85),
    )
    _tall_chimney(p, *at(23.5, 16.5), 26 * m)
    _craft_open_shed(p, *at(9, 12), 18 * m, 6 * m, 3.2 * m, 2.2 * m)
    _craft_kiln_block(p, *at(25, 12))
    _craft_mound(p, *at(3.5, 6.5), 3.2 * m, 2.6 * m, CREAM_CLAY)
    for v in (7.2, 5.4):
        _craft_hack(p, *at(12, v), 7 * m)
    _craft_pug_mill(p, *at(5, 2.5))
    high = 4.6 * m
    office = p.hall(*at(25, 3.5), 6 * m, 5 * m, high, OFFICE_WALL, None)
    left, front, right, _ = office
    _plinth(p, office, high, (176, 168, 150), 0.5 * m)
    _foot(p, office, high)
    _craft_tile_panel(p, left, front, 6 * m, (0.08, 0.5), (0.9 * m, 3.6 * m))
    _patch(p, left, front, (0.62, 0.8), (1.6 * m, high), DOOR)
    _craft_casement(p, front, right, 5 * m, 0.5, 1.3 * m, (1.2 * m, 2.8 * m), shade(WINDOW, 0.8))
    p.gable(office, 3 * m, STOVE_GREEN, OFFICE_WALL)
    _tiles(p, office, 3 * m, STOVE_GREEN, (0.2, 0.4, 0.6, 0.8))
    for u, v, colour, high in (
        (11, 2.8, ROOF_TILE, 1.2),
        (13.5, 2.8, ROOF_TILE, 1.5),
        (16, 2.8, ROOF_TILE, 1.0),
        (18.5, 2.8, STOVE_GREEN, 0.9),
        (12, 0.6, ROOF_TILE, 0.9),
        (14.5, 0.6, STOVE_GREEN, 1.2),
        (17, 0.6, ROOF_TILE, 1.4),
    ):
        _craft_tile_stack(p, *at(u, v), colour, high * m)
    return p.sprite()


SOOTED_BRICK: RGB = (150, 76, 56)
CAST_IRON: RGB = (98, 100, 106)
FURNACE_IRON: RGB = (82, 80, 84)
WHITE_HOT: RGB = (255, 238, 176)
MOULDING_SAND: RGB = (206, 174, 122)
BLACK_SAND: RGB = (76, 70, 64)
FOUNDRY_YARD: RGB = (124, 114, 100)
FURNACE_GLOW: RGB = (150, 88, 56)


def _craft_pour(p: Painter, a: Point, b: Point, length: float, t: float, foot: float) -> None:
    """What shows through a foundry's open door about `t` along the wall under the edge from `a`
    to `b`, which is `length` units long, its floor `foot` down it: the hall lit orange from
    within, a crane beam, a ladle hung from it by its bail and tipped to pour a thin white-hot
    stream into a mould on the sand floor, and a row of moulds glowing where they were filled."""
    m = METRE
    half = 1.9 * m / length
    gx, gy = _on_wall(a, b, t, foot - 1.4 * m)
    p.soft([(gx - 1.9 * m, gy - 1.8 * m, gx + 1.9 * m, gy + 1.2 * m)], (214, 104, 44), 0.7, 0.6)
    _craft_beam(
        p, a, b, (t - half, foot - 4.6 * m), (t + half, foot - 4.6 * m), 0.3 * m, (34, 28, 28)
    )
    hx, hy = _on_wall(a, b, t + half * 0.3, foot - 4.5 * m)
    _bar(p, (hx, hy), (hx, hy + 1.0 * m), 0.08 * m, (30, 26, 26))
    cx, cy = hx - 0.1 * m, hy + 2.3 * m
    turn = -0.5
    c, s = math.cos(turn), math.sin(turn)

    def tipped(u: float, v: float) -> Point:
        return cx + (u * c - v * s) * m, cy + (u * s + v * c) * m

    for side in (-0.75, 0.75):
        _bar(p, (hx, hy + 1.0 * m), tipped(side, -0.5), 0.07 * m, (30, 26, 26))
    p.polygon(
        [tipped(-0.7, -0.6), tipped(0.7, -0.6), tipped(0.5, 0.6), tipped(-0.5, 0.6)], (46, 42, 44)
    )
    p.polygon(
        [tipped(-0.62, -0.58), tipped(0.62, -0.58), tipped(0.58, -0.36), tipped(-0.58, -0.36)],
        MOLTEN,
    )
    p.polygon(
        [tipped(-0.62, -0.58), tipped(-0.1, -0.58), tipped(-0.12, -0.44), tipped(-0.6, -0.44)],
        WHITE_HOT,
    )
    _bar(p, tipped(-0.5, -0.1), tipped(0.4, 0.2), 0.06 * m, shade((46, 42, 44), 1.5))
    lip = tipped(-0.72, -0.6)
    mould = _on_wall(a, b, t - half * 0.25, foot - 0.55 * m)
    _bar(p, lip, (lip[0] - 0.1 * m, lip[1] + 0.25 * m), 0.2 * m, WHITE_HOT)
    _bar(
        p, (lip[0] - 0.1 * m, lip[1] + 0.25 * m), (lip[0] - 0.12 * m, mould[1]), 0.12 * m, WHITE_HOT
    )
    sand = (150, 86, 48)
    _craft_beam(p, a, b, (t - half, foot - 0.3 * m), (t + half, foot - 0.3 * m), 0.5 * m, sand)
    for k, dx in enumerate((-1.0, -0.3, 0.4, 1.1)):
        mx, my = _on_wall(a, b, t + half * dx * 0.8, foot - 0.35 * m)
        p.rect(mx - 0.45 * m, my - 0.45 * m, mx + 0.45 * m, my, (54, 48, 46))
        glow = (255, 236, 176) if k == 1 else MOLTEN if k == 0 else (120, 60, 36)
        p.blob(mx - 0.16 * m, my - 0.56 * m, mx + 0.16 * m, my - 0.34 * m, glow)
    sx, sy = lip[0] - 0.12 * m, mould[1]
    p.blob(sx - 0.35 * m, sy - 0.22 * m, sx + 0.35 * m, sy + 0.12 * m, (255, 200, 96))
    for dx, dy in ((-0.5, -0.4), (0.4, -0.55), (-0.2, -0.7)):
        q = 0.08 * m
        p.blob(sx + dx * m - q, sy + dy * m - q, sx + dx * m + q, sy + dy * m + q, (255, 220, 120))


def _craft_monitor(p: Painter, top: Top, rise: float, wall: RGB, roof: RGB) -> None:
    """A raised ventilator along the ridge of a gable roof `rise` high on a block's top: a long
    box louvred on its sides under a little roof of its own, letting the heat out."""
    m = METRE
    ridge_back, ridge_front = _ridge(top, rise)
    long = abs(ridge_front[0] - ridge_back[0]) * 0.72
    x, y = (ridge_back[0] + ridge_front[0]) / 2, (ridge_back[1] + ridge_front[1]) / 2
    high = 1.6 * m
    box = p.hall(x, y + 0.5 * m, long, 2.6 * m, high, wall, None)
    left, front, right, _ = box
    louvre = (44, 38, 38)
    for k in range(3):
        drop = (0.25 + 0.38 * k) * m
        _craft_beam(p, left, front, (0.03, drop), (0.97, drop), 0.16 * m, louvre)
        _craft_beam(p, front, right, (0.1, drop), (0.9, drop), 0.16 * m, shade(louvre, 0.8))
    for t in np.linspace(0, 1, 7):
        _craft_beam(p, left, front, (t, 0), (t, high), 0.12 * m, shade(wall, 0.8))
    p.gable(box, 0.8 * m, roof, wall)
    p.blob(x - 0.3 * m, y - 2.8 * m, x + 0.3 * m, y - 2.4 * m, shade(roof, 1.3))


def _craft_seam(p: Painter, x: float, y: float, half: float) -> None:
    """A riveted seam round the front of an iron shaft `half` across at its middle (x, y): a dark
    lapped joint with a row of bright rivet heads under it."""
    m = METRE
    _craft_hoop(p, x, y, half, 0.1 * m, shade(FURNACE_IRON, 0.7))
    q = 0.07 * m
    for a in np.linspace(0.25, math.pi - 0.25, 7):
        rx, ry = x + half * math.cos(a), y + half / 2 * math.sin(a) + 0.12 * m
        p.blob(rx - q, ry - q, rx + q, ry + q, shade(FURNACE_IRON, 1.45))


def _craft_cupola(p: Painter, x: float, y: float) -> None:
    """A cupola furnace standing on (x, y): a riveted iron shaft 2.4 m across on four legs, rising
    some 15 m to the glow and sparks at its spark arrester, a charging floor round it at 6.5 m on posts
    with a rail and a ladder up to it, its charging door glowing, and at its foot the tap hole
    glowing over a spout into a ladle on a bogie."""
    m = METRE
    r, deck = 1.2 * m, 6.5 * m
    side = 4.4 * m
    corners = [
        _iso(x, y, du * side / 2, dv * side / 2) for du, dv in ((-1, 1), (1, 1), (-1, -1), (1, -1))
    ]
    timber = (92, 74, 58)
    for cx, cy in corners[:2]:
        _bar(p, (cx, cy), (cx, cy - deck), 0.2 * m, timber)
    for dx, dy in ((-0.7, -0.3), (0.7, -0.3), (-0.7, 0.3), (0.7, 0.3)):
        lx, ly = x + dx * r, y + dy * r
        _bar(p, (lx, ly), (lx, ly - 1.4 * m), 0.2 * m, (60, 58, 62))
    rim = (40, 38, 40)
    _turned(p, x, y - 1.4 * m, [(0, r), (deck - 1.4 * m, r)], FURNACE_IRON, rim)
    for h in (2.8 * m, 4.2 * m, 5.6 * m):
        _craft_seam(p, x, y - h, r)
    tx, ty = x - 0.2 * r, y + 0.45 * r - 1.9 * m
    p.blob(tx - 0.3 * m, ty - 0.3 * m, tx + 0.3 * m, ty + 0.3 * m, MOLTEN)
    p.blob(tx - 0.12 * m, ty - 0.12 * m, tx + 0.12 * m, ty + 0.12 * m, WHITE_HOT)
    spout = _iso(tx, ty + 0.4 * m, 0, -1.6 * m)
    _bar(p, (tx, ty + 0.2 * m), spout, 0.3 * m, (60, 56, 56))
    _bar(p, (tx, ty + 0.1 * m), spout, 0.1 * m, MOLTEN)
    lx, ly = _iso(x, y, 0, -2.4 * m)
    for dx in (-0.5, 0.5):
        p.ellipse(
            lx + dx * m - 0.25 * m,
            ly - 0.25 * m,
            lx + dx * m + 0.25 * m,
            ly + 0.25 * m,
            (50, 48, 50),
        )
    _turned(p, lx, ly - 0.3 * m, [(0, 0.45 * m), (0.9 * m, 0.6 * m)], (56, 52, 54), MOLTEN)
    floor = p.hall(x, y - deck, side, side, 0.3 * m, timber, shade(timber, 1.2))
    for k in range(1, 6):
        _bar(
            p,
            _on_wall(floor[0], floor[3], k / 6, 0),
            _on_wall(floor[1], floor[2], k / 6, 0),
            0.06 * m,
            shade(timber, 0.9),
        )
    _turned(p, x, y - deck - 0.3 * m, [(0, r), (6 * m, r)], FURNACE_IRON, rim)
    for h in (1.5 * m, 3 * m, 4.5 * m):
        _craft_seam(p, x, y - deck - h, r)
    dx, dy = x - 0.35 * r, y - deck - 0.3 * m + 0.4 * r
    p.rect(dx - 0.45 * m, dy - 1.3 * m, dx + 0.45 * m, dy - 0.2 * m, (60, 36, 28))
    p.blob(dx - 0.3 * m, dy - 0.9 * m, dx + 0.3 * m, dy - 0.3 * m, MOLTEN)
    head = y - deck - 6.3 * m
    _turned(
        p,
        x,
        head,
        [(0, r), (0.4 * m, 1.6 * m), (1.8 * m, 1.6 * m), (2.4 * m, 0.3 * m)],
        shade(FURNACE_IRON, 0.85),
        None,
    )
    p.blob(x - 0.8 * m, head - 1.0 * m, x + 0.8 * m, head - 0.5 * m, MOLTEN)
    for sx, sy in ((-0.6, -3.2), (0.8, -3.8), (0.1, -4.6), (1.4, -4.9), (-0.4, -5.6)):
        q = 0.14 * m
        p.blob(x + sx * m - q, head + sy * m - q, x + sx * m + q, head + sy * m + q, (255, 200, 90))
    rail = (54, 52, 56)
    for a, b in ((corners[2], corners[3]), (corners[3], corners[1])):
        top_a, top_b = (a[0], a[1] - deck - 1.3 * m), (b[0], b[1] - deck - 1.3 * m)
        _bar(p, top_a, top_b, 0.1 * m, rail)
        for f in np.linspace(0, 1, 4):
            px, py = _on_wall(a, b, f, -deck)
            _bar(p, (px, py), (px, py - 1.3 * m), 0.08 * m, rail)
    for cx, cy in corners[2:]:
        _bar(p, (cx, cy), (cx, cy - deck), 0.2 * m, timber)
    lx, ly = _on_wall(corners[2], corners[3], 0.2, 0)
    for side_x in (-0.35, 0.35):
        _bar(p, (lx + side_x * m, ly), (lx + side_x * m, ly - deck - 1 * m), 0.1 * m, rail)
    for k in range(1, 13):
        _bar(
            p,
            (lx - 0.35 * m, ly - k * 0.55 * m),
            (lx + 0.35 * m, ly - k * 0.55 * m),
            0.07 * m,
            rail,
        )


def _craft_cogwheel(p: Painter, x: float, y: float, r: float, teeth: int = 14) -> None:
    """A cast-iron cogwheel `r` across its middle at (x, y) standing on edge: teeth round its rim,
    six spokes and a boss, lit on its upper left."""
    for k in range(teeth):
        a = 2 * math.pi * k / teeth
        c, s = math.cos(a), math.sin(a)
        w = 0.5 * math.pi * r / teeth
        p.polygon(
            [
                (x + c * r * 0.9 - s * w, y + s * r * 0.9 + c * w),
                (x + c * r * 1.15 - s * w * 0.7, y + s * r * 1.15 + c * w * 0.7),
                (x + c * r * 1.15 + s * w * 0.7, y + s * r * 1.15 - c * w * 0.7),
                (x + c * r * 0.9 + s * w, y + s * r * 0.9 - c * w),
            ],
            shade(CAST_IRON, 0.8),
        )
    p.ellipse(x - r, y - r, x + r, y + r, CAST_IRON)
    p.blob(x - 0.78 * r, y - 0.78 * r, x + 0.78 * r, y + 0.78 * r, shade(CAST_IRON, 0.55))
    for k in range(3):
        a = k * math.pi / 3
        dx, dy = 0.8 * r * math.cos(a), 0.8 * r * math.sin(a)
        _bar(p, (x - dx, y - dy), (x + dx, y + dy), 0.16 * r, CAST_IRON)
    p.ellipse(x - 0.25 * r, y - 0.25 * r, x + 0.25 * r, y + 0.25 * r, shade(CAST_IRON, 1.1))
    _bar(
        p,
        (x - 0.7 * r, y - 0.75 * r),
        (x - 0.1 * r, y - 1.02 * r),
        0.08 * r,
        shade(CAST_IRON, 1.35),
    )


def _craft_pipes(p: Painter, x: float, y: float, long: float) -> None:
    """Cast-iron pipes 0.6 m across and `long` metres long stacked three and two on timber bearers
    at (x, y), lying along the lit walls with their dark bores towards the shaded side."""
    m = METRE
    for along in (-0.3, 0.3):
        bx, by = _iso(x, y, along * long * m, 0)
        p.hall(bx, by, 0.3 * m, 2.4 * m, 0.25 * m, shade(TIMBER, 0.6), None)
    r = 0.3 * m
    for row, count in enumerate((3, 2)):
        for k in range(count):
            v = (k - (count - 1) / 2) * 2 * r * 1.05
            cx, cy = _iso(x, y - 0.25 * m - r - row * 1.7 * r, 0, -v)
            near, far = _iso(cx, cy, long * m / 2, 0), _iso(cx, cy, -long * m / 2, 0)
            _bar(p, far, near, 2 * r, CAST_IRON)
            _bar(
                p,
                (far[0], far[1] - 0.55 * r),
                (near[0], near[1] - 0.55 * r),
                0.5 * r,
                shade(CAST_IRON, 1.25),
            )
            p.ellipse(
                near[0] - r * 0.9,
                near[1] - r,
                near[0] + r * 0.9,
                near[1] + r,
                shade(CAST_IRON, 0.9),
            )
            p.blob(
                near[0] - r * 0.5,
                near[1] - r * 0.6,
                near[0] + r * 0.5,
                near[1] + r * 0.6,
                (34, 32, 34),
            )


def _craft_bell(p: Painter, x: float, y: float) -> None:
    """A cast-iron church bell 1.3 m high hung in a timber frame on (x, y), lit on its left."""
    m = METRE
    timber = shade(TIMBER, 0.8)
    for dx in (-1.1, 1.1):
        _bar(p, (x + dx * m, y + 0.2 * m), (x + dx * 0.9 * m, y - 2.6 * m), 0.2 * m, timber)
    _bar(p, (x - 1.2 * m, y - 2.6 * m), (x + 1.2 * m, y - 2.6 * m), 0.25 * m, timber)
    top = y - 2.4 * m
    p.polygon(
        [
            (x - 0.25 * m, top),
            (x + 0.25 * m, top),
            (x + 0.45 * m, top + 0.3 * m),
            (x + 0.5 * m, top + 0.9 * m),
            (x + 0.75 * m, top + 1.3 * m),
            (x - 0.75 * m, top + 1.3 * m),
            (x - 0.5 * m, top + 0.9 * m),
            (x - 0.45 * m, top + 0.3 * m),
        ],
        shade(CAST_IRON, 0.85),
    )
    p.polygon(
        [
            (x - 0.25 * m, top),
            (x, top),
            (x, top + 1.3 * m),
            (x - 0.75 * m, top + 1.3 * m),
            (x - 0.5 * m, top + 0.9 * m),
            (x - 0.45 * m, top + 0.3 * m),
        ],
        CAST_IRON,
    )
    _bar(
        p,
        (x - 0.75 * m, top + 1.2 * m),
        (x + 0.75 * m, top + 1.2 * m),
        0.12 * m,
        shade(CAST_IRON, 1.3),
    )
    p.blob(x - 0.12 * m, top + 1.2 * m, x + 0.12 * m, top + 1.5 * m, shade(CAST_IRON, 0.5))


def _craft_wheelset(p: Painter, x: float, y: float) -> None:
    """A pair of cast-iron railway wheels on their axle standing on (x, y), the axle running back
    along the shaded walls."""
    m = METRE
    r = 0.75 * m
    far = _iso(x, y, 0, 1.5 * m)
    for wx, wy in (far, (x, y)):
        cy = wy - r
        p.ellipse(wx - 0.55 * r, cy - r, wx + 0.55 * r, cy + r, shade(CAST_IRON, 0.75))
        p.ellipse(wx - 0.5 * r, cy - 0.92 * r, wx + 0.42 * r, cy + 0.92 * r, CAST_IRON)
        p.blob(wx - 0.12 * r, cy - 0.15 * r, wx + 0.12 * r, cy + 0.15 * r, shade(CAST_IRON, 0.5))
        if (wx, wy) == far:
            _bar(p, (far[0], far[1] - r), (x, y - r), 0.18 * m, shade(CAST_IRON, 0.6))


def _craft_flasks(p: Painter, x: float, y: float, layers: int) -> None:
    """Moulding boxes stacked `layers` high on (x, y): iron frames 1.4 by 1 m packed with sand."""
    m = METRE
    for k in range(layers):
        top = p.hall(x, y - k * 0.45 * m, 1.4 * m, 1.0 * m, 0.4 * m, (70, 66, 66), MOULDING_SAND)
        _craft_band(p, top, 0.2 * m, 0.08 * m, (104, 100, 100))


def foundry() -> Sprite:
    """An iron foundry: a casting hall 18 by 9 m of soot-stained red brick with a louvred monitor
    along its slate ridge, round-headed windows glowing dimly and a great open door through which
    a ladle hung from a crane pours a thin white-hot stream into moulds on the sand floor, its
    light spilling out over the yard; beside it a riveted cupola furnace on legs with a charging
    floor and ladder, its tap glowing into a ladle and sparks at its top; heaps of tan
    and black moulding sand and stacked moulding boxes; and a yard of grey castings: a cogwheel,
    pipes, a bell in its frame, street lamps and railway wheels."""
    m = METRE
    p = Painter(37, 26, (19, 16))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 14.5) * m, (v - 7.5) * m)

    p.polygon(
        [at(-0.5, -1), at(29, -1), at(29, 8), at(25, 8), at(25, 15.5), at(-0.5, 15.5)],
        FOUNDRY_YARD,
        shade(FOUNDRY_YARD, 0.85),
    )
    length, depth, high = 18 * m, 9 * m, 7 * m
    top = p.hall(*at(9, 10.5), length, depth, high, SOOTED_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, SOOTED_BRICK, 0.6 * m)
    soot = shade(SOOTED_BRICK, 0.58)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (0, 1.2 * m), shade(soot, tone))
        _patch(p, a, b, (0, 1), (1.2 * m, 1.8 * m), shade(soot, tone * 1.25))
        _patch(p, a, b, (0, 1), (2.6 * m, 2.9 * m), shade(CREAM_GLAZE, tone * 0.85))
    for k in range(7):
        t = k / 6
        _craft_beam(p, left, front, (t, 1.2 * m), (t, high), 0.45 * m, shade(SOOTED_BRICK, 0.85))
    door = 0.64
    for t, reach in ((0.05, 3.2), (0.22, 2.4), (0.4, 3.6), (0.85, 2.8), (0.97, 3.4)):
        _craft_beam(p, left, front, (t, 1.8 * m), (t, reach * m), 0.25 * m, shade(soot, 1.3))
    _patch(p, left, front, (door - 0.17, door + 0.17), (1.2 * m, 1.6 * m), shade(soot, 0.9))
    for k in range(6):
        t = (k + 0.5) / 6
        if abs(t - door) < 0.12:
            continue
        _mill_window(p, left, front, length, t, 1.4 * m, (3.4 * m, 5.9 * m), FURNACE_GLOW)
    for k in range(3):
        t = (k + 0.5) / 3
        _mill_window(
            p, front, right, depth, t, 1.4 * m, (3.4 * m, 5.9 * m), shade(FURNACE_GLOW, 0.7)
        )
    p.polygon(_round_head(left, front, length, door, 6.0 * m, 1.3 * m, high), DRESSING)
    p.polygon(_round_head(left, front, length, door, 5.0 * m, 1.9 * m, high), (40, 32, 30))
    _craft_pour(p, left, front, length, door, high)
    p.gable(top, 3.4 * m, SLATE, SOOTED_BRICK)
    _tiles(p, top, 3.4 * m, SLATE, (0.3, 0.6))
    _craft_monitor(p, top, 3.4 * m, shade(SOOTED_BRICK, 0.8), SLATE)
    gx, gy = _on_wall(left, front, door, high)
    p.soft([(gx - 3.8 * m, gy - 0.4 * m, gx + 2.2 * m, gy + 2.6 * m)], (255, 166, 76), 0.42, 0.8)
    _craft_cupola(p, *at(21.5, 11))
    _craft_mound(p, *at(3, 3.5), 2.4 * m, 2.0 * m, MOULDING_SAND)
    _craft_mound(p, *at(5.5, 1), 1.8 * m, 1.4 * m, BLACK_SAND)
    _craft_flasks(p, *at(8.5, 2.5), 3)
    _craft_flasks(p, *at(10.5, 1), 2)
    _craft_bell(p, *at(17.5, 3))
    _craft_cogwheel(p, *at(21.5, 3.5), 1.5 * m)
    _craft_pipes(p, *at(26, 6), 4)
    for u, v in ((24.5, 3.2), (26.5, 2.4)):
        _lamp_post(p, *at(u, v), 4.2)
    _craft_wheelset(p, *at(27, 0))
    return p.sprite()


SAUSAGE: RGB = (184, 58, 52)
HAM: RGB = (160, 74, 58)
BUTCHER_PLASTER: RGB = (240, 228, 200)
OAK: RGB = (84, 56, 42)
BUTCHER_ROOF: RGB = (132, 64, 54)
WHITE_TILE: RGB = (236, 238, 234)
STRIPES: tuple[RGB, RGB] = ((198, 54, 62), (248, 222, 222))
SMOKED_BRICK: RGB = (128, 84, 64)
SHOP_GLOW: RGB = (250, 228, 182)
IRON: RGB = (46, 42, 42)
COBBLES: RGB = (176, 168, 152)
YARD: RGB = (150, 132, 100)
MUD: RGB = (118, 92, 64)


def _craft_framing(
    p: Painter, a: Point, b: Point, length: float, drop: tuple[float, float], colour: RGB
) -> None:
    """Oak framing on a plastered wall under the edge from `a` to `b`, which is `length` units
    long, from `drop[0]` down it to `drop[1]`: a plate and a sill, a post every metre and a half, a
    rail halfway and a brace rising from it to the plate in every other bay."""
    m = METRE
    d0, d1 = drop
    bays = max(2, round(length / (1.5 * m)))
    wide = 0.22 * m
    for d in (d0 + wide / 2, d1 - wide / 2):
        _craft_beam(p, a, b, (0, d), (1, d), wide, colour)
    middle = (d0 + d1) / 2
    _craft_beam(p, a, b, (0, middle), (1, middle), wide * 0.8, colour)
    for k in range(bays + 1):
        t = k / bays
        _craft_beam(p, a, b, (t, d0), (t, d1), wide, colour)
    for k in range(bays):
        t0, t1 = k / bays, (k + 1) / bays
        if k % 2 == 0:
            _craft_beam(p, a, b, (t0, middle), (t1, d0), wide * 0.8, colour)
        elif k == bays - 1:
            _craft_beam(p, a, b, (t1, middle), (t0, d0), wide * 0.8, colour)


def _craft_meat_window(
    p: Painter,
    a: Point,
    b: Point,
    length: float,
    span: tuple[float, float],
    drop: tuple[float, float],
) -> None:
    """A butcher's window in a white frame on the wall under the edge from `a` to `b`, which is
    `length` units long, from `span[0]` to `span[1]` along it and `drop[0]` to `drop[1]` down it:
    lit warm inside, with sausages of reds and pinks and hams hung from a rail and a white marble
    counter along its foot laid with pink joints."""
    m = METRE
    t0, t1 = span
    d0, d1 = drop
    rim = 0.15 * m / length
    _patch(p, a, b, (t0 - rim, t1 + rim), (d0 - 0.15 * m, d1 + 0.15 * m), WHITE_TILE)
    _patch(p, a, b, span, drop, SHOP_GLOW)
    rail = d0 + (d1 - d0) * 0.42
    _craft_beam(p, a, b, (t0, rail), (t1, rail), 0.08 * m, (96, 84, 78))
    count = max(3, round((t1 - t0) * length / (0.36 * m)))
    step = (t1 - t0) / count
    links = (SAUSAGE, (214, 120, 112), shade(SAUSAGE, 0.78))
    for k in range(count):
        x, y = _on_wall(a, b, t0 + step * (k + 0.5), rail)
        if k % 4 == 2:
            _bar(p, (x, y), (x, y + 0.2 * m), 0.05 * m, (120, 110, 100))
            p.ellipse(x - 0.24 * m, y + 0.15 * m, x + 0.24 * m, y + 0.85 * m, HAM)
            p.blob(x - 0.15 * m, y + 0.28 * m, x - 0.01 * m, y + 0.55 * m, shade(HAM, 1.35))
        else:
            low = y + (0.55 + 0.2 * (k % 2)) * m
            _bar(p, (x, y), (x, low), 0.17 * m, links[k % 3])
            p.blob(
                x - 0.09 * m, low - 0.06 * m, x + 0.09 * m, low + 0.08 * m, shade(links[k % 3], 0.8)
            )
    counter = d1 - 0.2 * m
    _patch(p, a, b, span, (counter, d1), (228, 224, 216))


def _craft_awning(
    p: Painter,
    a: Point,
    b: Point,
    span: tuple[float, float],
    drop: float,
    reach: float,
    colours: tuple[RGB, RGB],
) -> None:
    """A striped canvas awning sloping out `reach` from the wall under the edge from `a` to `b`,
    from `span[0]` to `span[1]` along it, fixed `drop` down it, with a scalloped valance."""
    m = METRE
    t0, t1 = span
    dx, dy = _outward(a, b, reach)
    fall = reach * 0.5
    stripes = max(4, round(abs(b[0] - a[0]) * (t1 - t0) / (0.5 * m)))
    for k in range(stripes):
        s0, s1 = t0 + (t1 - t0) * k / stripes, t0 + (t1 - t0) * (k + 1) / stripes
        (x0, y0), (x1, y1) = _on_wall(a, b, s0, drop), _on_wall(a, b, s1, drop)
        o0, o1 = (x0 + dx, y0 + dy + fall), (x1 + dx, y1 + dy + fall)
        colour = colours[k % 2]
        p.polygon([(x0, y0), (x1, y1), o1, o0], colour)
        hem = 0.3 * m
        p.polygon([o0, o1, (o1[0], o1[1] + hem), (o0[0], o0[1] + hem)], shade(colour, 0.88))
        mx, my = (o0[0] + o1[0]) / 2, (o0[1] + o1[1]) / 2 + hem
        half = abs(o1[0] - o0[0]) / 2
        p.blob(mx - half, my - 0.15 * m, mx + half, my + 0.15 * m, shade(colour, 0.88))


def _craft_gilt_pig(p: Painter, x: float, y: float, long: float) -> None:
    """A pig cut out of gilt sheet in profile, `long` from snout to tail with the middle of its
    back at (x, y), facing left: a round body on short legs, a blunt snout, an ear flopped
    forward and a curled tail, lit along its back."""
    s = long
    dark = shade(GILT, 0.7)
    for dx, tone in ((-0.28, GILT), (-0.14, dark), (0.18, GILT), (0.3, dark)):
        _bar(p, (x + dx * s, y + 0.34 * s), (x + dx * s, y + 0.5 * s), 0.08 * s, tone)
    p.ellipse(x - 0.42 * s, y, x + 0.4 * s, y + 0.44 * s, GILT)
    p.blob(x - 0.36 * s, y + 0.26 * s, x + 0.34 * s, y + 0.42 * s, shade(GILT, 0.86))
    p.ellipse(x - 0.58 * s, y + 0.07 * s, x - 0.26 * s, y + 0.35 * s, GILT)
    p.ellipse(x - 0.66 * s, y + 0.14 * s, x - 0.53 * s, y + 0.3 * s, shade(GILT, 0.9))
    p.polygon(
        [(x - 0.36 * s, y + 0.06 * s), (x - 0.52 * s, y + 0.04 * s), (x - 0.44 * s, y + 0.18 * s)],
        dark,
    )
    p.blob(x - 0.52 * s, y + 0.13 * s, x - 0.48 * s, y + 0.17 * s, (90, 62, 24))
    _bar(p, (x + 0.38 * s, y + 0.14 * s), (x + 0.47 * s, y + 0.04 * s), 0.05 * s, GILT)
    _bar(p, (x + 0.47 * s, y + 0.04 * s), (x + 0.5 * s, y + 0.12 * s), 0.05 * s, GILT)
    _bar(p, (x - 0.26 * s, y + 0.05 * s), (x + 0.24 * s, y + 0.04 * s), 0.05 * s, (252, 232, 150))


def _craft_pig_sign(p: Painter, x: float, y: float) -> None:
    """A pork butcher's sign: a gilt pig hung on two chains from a wrought-iron bracket fixed to a
    wall at (x, y), reaching out towards the lit side and braced by a stay under it."""
    m = METRE
    end = (x - 2.8 * m, y + 1.4 * m)
    _bar(p, (x, y), end, 0.16 * m, IRON)
    _bar(p, (x, y + 1.1 * m), (x - 1.4 * m, y + 0.7 * m), 0.1 * m, IRON)
    for t in (0.25, 0.8):
        cx, cy = x + (end[0] - x) * t, y + (end[1] - y) * t
        _bar(p, (cx, cy), (cx, cy + 0.9 * m - t * 0.6 * m), 0.05 * m, IRON)
    mx = x + (end[0] - x) * 0.52
    my = y + (end[1] - y) * 0.52
    _craft_gilt_pig(p, mx, my + 0.45 * m, 2.6 * m)


def _craft_smokehouse(p: Painter, x: float, y: float) -> None:
    """A smokehouse 4 by 4 m and 6 m to its eaves of smoke-browned brick standing on (x, y): soot
    creeping down from its eaves, a riveted iron door, a steep pyramid roof of dark tiles and a
    louvred lantern on it with wood smoke seeping out between the slats."""
    m = METRE
    high = 6 * m
    top = p.hall(x, y, 4 * m, 4 * m, high, SMOKED_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, SMOKED_BRICK, 0.5 * m)
    for drop in (1.6 * m, 3.2 * m, 4.6 * m):
        _craft_band(p, top, drop, 0.1 * m, shade(SMOKED_BRICK, 0.86))
    soot = shade(SMOKED_BRICK, 0.62)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (0, 0.5 * m), shade(soot, tone * 0.9))
        _patch(p, a, b, (0, 1), (0.5 * m, 1.1 * m), shade(soot, tone * 1.15))
        for t, reach in ((0.15, 2.0), (0.5, 2.6), (0.85, 1.6)):
            _craft_beam(p, a, b, (t, 1 * m), (t, reach * m), 0.12 * m, shade(soot, tone * 1.1))
    door = (70, 68, 70)
    _patch(p, left, front, (0.28, 0.72), (high - 2.3 * m, high), door)
    for drop in (high - 1.9 * m, high - 0.6 * m):
        _craft_beam(p, left, front, (0.28, drop), (0.62, drop), 0.14 * m, shade(door, 0.6))
    _craft_beam(
        p, left, front, (0.66, high - 1.3 * m), (0.66, high - 1 * m), 0.12 * m, (180, 170, 150)
    )
    _patch(p, front, right, (0.35, 0.65), (1.4 * m, 2.4 * m), shade(door, 0.7))
    roof = (78, 64, 60)
    apex = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2 - 2.8 * m)
    p.polygon([left, front, apex], roof, shade(roof, 0.6))
    p.polygon([front, right, apex], shade(roof, 0.7), shade(roof, 0.5))
    for f in (0.35, 0.7):
        _bar(p, _on_wall(left, apex, f, 0), _on_wall(front, apex, f, 0), 0.1 * m, shade(roof, 0.8))
    lx, ly = apex[0], apex[1] + 0.9 * m
    lantern = p.hall(lx, ly, 1.4 * m, 1.4 * m, 1.1 * m, (104, 86, 74), None)
    l_left, l_front, l_right, _ = lantern
    for k in range(3):
        drop = (0.25 + 0.3 * k) * m
        _craft_beam(p, l_left, l_front, (0.1, drop), (0.9, drop), 0.12 * m, (40, 34, 32))
        _craft_beam(p, l_front, l_right, (0.1, drop), (0.9, drop), 0.12 * m, (34, 28, 26))
    cap = ((l_left[0] + l_right[0]) / 2, (l_left[1] + l_right[1]) / 2 - 1 * m)
    p.polygon([(l_left[0] - 0.3 * m, l_left[1]), l_front, cap], roof, shade(roof, 0.6))
    p.polygon([l_front, (l_right[0] + 0.3 * m, l_right[1]), cap], shade(roof, 0.7))
    _craft_smoke(p, cap[0] + 0.6 * m, cap[1] - 1.2 * m, 2.2 * m)


def _craft_sty(p: Painter, x: float, y: float) -> None:
    """A low pigsty 3 by 2.5 m of whitewashed stone standing on (x, y), under a roof of weathered
    boards, with a dark low door."""
    m = METRE
    stone = (226, 220, 206)
    top = p.hall(x, y, 3 * m, 2.5 * m, 1.6 * m, stone, None)
    left, front, _, _ = top
    _patch(p, left, front, (0.55, 0.85), (0.5 * m, 1.6 * m), (54, 44, 40))
    _plinth(p, top, 1.6 * m, stone, 0.3 * m)
    boards = (126, 104, 82)
    p.gable(top, 0.9 * m, boards, stone)
    _tiles(p, top, 0.9 * m, boards, (0.5,))


def sausage_maker() -> Sprite:
    """A village pork butcher's 9 by 7 m: a half-timbered house of cream plaster and dark oak
    under a red-brown tiled roof with a dormer, its shop front in white tiles under a red fascia
    lettered in gold, windows hung with sausages of reds and pinks and hams over a marble counter,
    the big one under a red and pink striped awning, and a gilt pig hung from a wrought-iron
    bracket at its corner; behind it a smokehouse of smoke-browned brick with wood smoke seeping
    from the louvres of its lantern and a pile of beech logs before it, and beside them a pen of
    churned mud where two pink pigs root by a trough and a whitewashed sty, drawn twice their size
    (BEAST)."""
    m = METRE
    p = Painter(26, 21, (12, 13))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 8) * m, (v - 5.5) * m)

    p.polygon([at(9, 4), at(17.5, 4), at(17.5, 12), at(9, 12)], YARD, shade(YARD, 0.85))
    p.polygon([at(-1.5, 1), at(10.5, 1), at(10.5, 4), at(-1.5, 4)], COBBLES, shade(COBBLES, 0.85))
    high = 5.6 * m
    top = p.hall(*at(4.5, 7.5), 9 * m, 7 * m, high, BUTCHER_PLASTER, None)
    left, front, right, _ = top
    _plinth(p, top, high, (170, 160, 146), 0.5 * m)
    _foot(p, top, high)
    _craft_framing(p, left, front, 9 * m, (0, 2.6 * m), OAK)
    _craft_framing(p, front, right, 7 * m, (0, high - 0.5 * m), shade(OAK, 0.8))
    _patch(p, left, front, (0, 1), (2.6 * m, high - 0.5 * m), WHITE_TILE)
    for drop in (3.6 * m, 4.4 * m):
        _craft_beam(p, left, front, (0, drop), (1, drop), 0.04 * m, shade(WHITE_TILE, 0.88))
    _patch(p, left, front, (0.03, 0.97), (2.6 * m, 3.05 * m), (150, 36, 44))
    for t0, t1 in ((0.62, 0.68), (0.7, 0.75), (0.77, 0.86), (0.88, 0.94)):
        _craft_beam(p, left, front, (t0, 2.83 * m), (t1, 2.83 * m), 0.16 * m, GILT)
    _craft_meat_window(p, left, front, 9 * m, (0.12, 0.5), (3.35 * m, high - 0.5 * m))
    _patch(p, left, front, (0.6, 0.73), (3.25 * m, high), DOOR)
    _patch(p, left, front, (0.62, 0.71), (3.45 * m, 4.3 * m), SHOP_GLOW)
    _craft_meat_window(p, left, front, 9 * m, (0.8, 0.94), (3.35 * m, high - 0.5 * m))
    for t in (0.25, 0.75):
        _craft_casement(p, left, front, 9 * m, t, 1.2 * m, (0.7 * m, 1.9 * m), WINDOW)
    for t in (0.3, 0.72):
        _craft_casement(p, front, right, 7 * m, t, 1.1 * m, (1.2 * m, 2.4 * m), shade(WINDOW, 0.8))
    _craft_casement(p, front, right, 7 * m, 0.5, 1.1 * m, (3.4 * m, 4.6 * m), shade(WINDOW, 0.8))
    _craft_awning(p, left, front, (0.09, 0.53), 3.05 * m, 0.7 * m, STRIPES)
    p.gable(top, 4.2 * m, BUTCHER_ROOF, BUTCHER_PLASTER)
    _tiles(p, top, 4.2 * m, BUTCHER_ROOF, (0.25, 0.5, 0.75))
    dormer = p.hall(
        *_on_wall(left, front, 0.55, -1.2 * m), 1.8 * m, 1.6 * m, 1.6 * m, BUTCHER_PLASTER, None
    )
    d_left, d_front, _, _ = dormer
    _craft_casement(p, d_left, d_front, 1.8 * m, 0.5, 1.0 * m, (0.4 * m, 1.3 * m), WINDOW)
    _gable_across(p, dormer, 1.0 * m, BUTCHER_ROOF, BUTCHER_PLASTER)
    ridge_back, ridge_front = _ridge(top, 4.2 * m)
    mid = ((front[0] + right[0]) / 2, (front[1] + right[1]) / 2)
    _bar(p, ridge_front, mid, 0.22 * m, shade(OAK, 0.8))
    for t in (0.25, 0.75):
        foot = _on_wall(front, right, t, 0)
        _bar(p, foot, _on_wall(ridge_front, foot, 0.5, 0), 0.18 * m, shade(OAK, 0.8))
    cx, cy = _on_wall(ridge_back, ridge_front, 0.3, 0)
    p.hall(cx, cy + 0.4 * m, 0.9 * m, 0.9 * m, 1.6 * m, BRICK, (60, 50, 48))
    _craft_pig_sign(p, *_on_wall(left, front, 0.03, 0.9 * m))
    _craft_smokehouse(p, *at(14, 9.5))
    _craft_log_pile(p, *at(14, 6.8), 3.4 * m, 1.6 * m)
    pen = (at(10.5, -0.5), at(18.5, -0.5), at(18.5, 5), at(10.5, 5))
    near_left, near_right, far_right, far_left = pen
    p.polygon(list(pen), MUD, shade(MUD, 0.8))
    wx, wy = at(13.5, 1.8)
    p.soft([(wx - 2.2 * m, wy - 0.9 * m, wx + 2.2 * m, wy + 0.9 * m)], shade(MUD, 0.8), 0.7, 0.6)
    _fence(p, near_left, far_left, METRE)
    _fence(p, far_left, far_right, METRE)
    _craft_sty(p, *at(17, 3.75))
    tx, ty = at(13.5, 3.6)
    p.hall(tx, ty, 2.2 * m, 0.6 * m, 0.5 * m, (122, 90, 60), (176, 150, 96))
    _pig(p, *at(14.5, 1.6), BEAST, 1)
    _pig(p, *at(12.2, 0.6), BEAST, -1)
    _fence(p, near_left, near_right, METRE)
    _fence(p, near_right, far_right, METRE)
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
