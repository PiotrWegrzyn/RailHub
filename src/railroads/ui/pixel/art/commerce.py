"""Car dealers, jewellers, taverns, department stores, hotels, builder's yards and airports."""

import numpy as np

from railroads.ui.pixel.art.churches import _ribs
from railroads.ui.pixel.art.homes import _doorstep, _foot, _outward, _worn
from railroads.ui.pixel.art.painter import (
    BARREL,
    BRICK,
    CAR_COLOURS,
    DOORWAY,
    GLASS,
    RGB,
    STONE,
    TIMBER,
    VERDIGRIS,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import (
    DOOR,
    LIT,
    METRE,
    _band,
    _bands,
    _bar,
    _fence,
    _flag,
    _gable_across,
    _iso,
    _on_wall,
    _paddock,
    _pale_plinth,
    _patch,
    _tiles,
    _turned,
    _windows,
)

PAVING: RGB = (190, 184, 170)
POST: RGB = (52, 50, 54)
PLANK: RGB = (214, 178, 118)


def _out(a: Point, b: Point, t: float, height: float, reach: float) -> Point:
    """The point on the ground `reach` out from the foot of the wall under the edge from `a` to
    `b`, `t` of the way along it, `height` below that edge."""
    x, y = _on_wall(a, b, t, height)
    dx, dy = _outward(a, b, reach)
    return x + dx, y + dy


def _cask(p: Painter, x: float, y: float) -> None:
    """A small cask standing on end on the ground at (x, y)."""
    p.ellipse(x - 1.5, y - 4, x + 1.5, y, BARREL)
    p.line([(x - 1.5, y - 1.5), (x + 1.5, y - 1.5)], shade(BARREL, 0.55))
    p.line([(x - 1, y - 3.2), (x + 1, y - 3.2)], shade(BARREL, 1.25))


def _street_lamp(p: Painter, x: float, y: float, height: float = 7) -> None:
    """A black street lamp with a lit glass head, its foot at (x, y)."""
    p.line([(x - 0.5, y), (x + 0.5, y)], POST)
    p.line([(x, y), (x, y - height)], POST)
    p.rect(x - 0.7, y - height - 1.8, x + 0.7, y - height, LIT)
    p.point(x - 0.5, y - height - 2.6, POST)


def _lamp_post(p: Painter, x: float, y: float, tall: float = 4, arms: int = 1) -> None:
    """A black cast-iron lamp post `tall` metres high standing on (x, y) with a lit glass lantern
    on top, or with `arms` 2 a lantern hung either side of a cross arm."""
    m = METRE
    top = y - tall * m
    p.blob(x - 0.3 * m, y - 0.15 * m, x + 0.3 * m, y + 0.15 * m, POST)
    _bar(p, (x, y), (x, top), 0.18 * m, POST)
    heads = [(x, top)]
    if arms == 2:
        _bar(p, (x - 0.8 * m, top + 0.4 * m), (x + 0.8 * m, top + 0.4 * m), 0.12 * m, POST)
        heads = [(x - 0.8 * m, top + 0.9 * m), (x + 0.8 * m, top + 0.9 * m), (x, top)]
    for hx, hy in heads:
        p.polygon(
            [
                (hx - 0.25 * m, hy),
                (hx + 0.25 * m, hy),
                (hx + 0.35 * m, hy - 0.75 * m),
                (hx - 0.35 * m, hy - 0.75 * m),
            ],
            LIT,
        )
        p.polygon(
            [(hx - 0.45 * m, hy - 0.75 * m), (hx + 0.45 * m, hy - 0.75 * m), (hx, hy - 1.05 * m)],
            POST,
        )


SHOWROOM: RGB = (240, 240, 234)
DEALER_RED: RGB = (204, 40, 46)
CHROME: RGB = (204, 210, 216)
PLATE_GLASS: RGB = (132, 178, 204)
TYRE: RGB = (34, 34, 38)


def _auto(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A car 4.4 m long and 1.8 m wide parked on (x, y) nose out towards the lit side: its
    grille, bumper and headlamps on its lit end, its windscreen and side windows in a chrome-
    trimmed cabin, and its wheels along its shaded side, lit on its top."""
    m = METRE
    p.blob(x - 2.2 * m, y - 1.2 * m, x + 2.6 * m, y + 1.4 * m, (60, 60, 64), 110)
    body = p.hall(x, y - 0.3 * m, 1.8 * m, 4.4 * m, 0.75 * m, colour, shade(colour, 1.12))
    left, front, right, _ = body
    for t in (0.22, 0.78):
        wx, wy = _on_wall(front, right, t, 0.75 * m)
        p.ellipse(wx - 0.38 * m, wy - 0.3 * m, wx + 0.38 * m, wy + 0.45 * m, TYRE)
        p.blob(wx - 0.14 * m, wy - 0.04 * m, wx + 0.14 * m, wy + 0.2 * m, CHROME)
    _bar(p, _on_wall(left, front, 0, 0.6 * m), _on_wall(left, front, 1, 0.6 * m), 0.14 * m, CHROME)
    _patch(p, left, front, (0.32, 0.68), (0.2 * m, 0.5 * m), shade(colour, 0.5))
    for t in (0.14, 0.86):
        hx, hy = _on_wall(left, front, t, 0.3 * m)
        p.blob(hx - 0.16 * m, hy - 0.14 * m, hx + 0.16 * m, hy + 0.14 * m, (252, 246, 210))
    cx, cy = _iso(x, y, 0, 0.35 * m)
    cabin = p.hall(cx, cy - 1.05 * m, 1.6 * m, 2.3 * m, 0.6 * m, colour, shade(colour, 1.18))
    c_left, c_front, c_right, _ = cabin
    _patch(p, c_left, c_front, (0.08, 0.92), (0.08 * m, 0.55 * m), shade(PLATE_GLASS, 0.8))
    _patch(p, c_front, c_right, (0.06, 0.46), (0.1 * m, 0.55 * m), shade(PLATE_GLASS, 0.62))
    _patch(p, c_front, c_right, (0.54, 0.94), (0.1 * m, 0.55 * m), shade(PLATE_GLASS, 0.62))


def _pennant_flag(p: Painter, x: float, y: float, tall: float, colour: RGB) -> None:
    """A white flagpole `tall` metres high on (x, y) flying a flag of `colour` with a white
    stripe, its top knob gilt."""
    m = METRE
    top = y - tall * m
    _bar(p, (x, y), (x, top), 0.14 * m, (236, 236, 232))
    p.polygon(
        [
            (x, top + 0.2 * m),
            (x + 2 * m, top + 0.6 * m),
            (x + 1.9 * m, top + 1.8 * m),
            (x, top + 1.4 * m),
        ],
        colour,
    )
    _bar(p, (x, top + 0.8 * m), (x + 1.95 * m, top + 1.2 * m), 0.25 * m, (246, 244, 238))
    p.blob(x - 0.15 * m, top - 0.25 * m, x + 0.15 * m, top + 0.05 * m, (228, 182, 70))


def car_dealer() -> Sprite:
    """A car dealer's showroom 20 by 14 m: a white hall walled in plate glass on chrome mullions
    with new cars on show inside, a bold red fascia lettered in white along its flat roof, a
    workshop door at its side, a row of new cars in bright colours parked nose out on its paved
    forecourt, a tall red pylon sign with a white badge, and flags flying along its roof."""
    m = METRE
    p = Painter(42, 32, (21, 21))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10) * m, (v - 4) * m)

    high = 6 * m
    p.polygon([at(-2, -9), at(23, -9), at(23, 14), at(20, 14), at(20, 0), at(-2, 0)], PAVING)
    for u in range(0, 23, 4):
        _bar(p, at(u, -9), at(u, -6.5), 0.12 * m, (246, 244, 238))
    p.shadow(13, 4, 3)
    top = p.hall(*at(10, 7), 20 * m, 14 * m, high, SHOWROOM, shade(SHOWROOM, 0.82))
    left, front, right, _ = top
    _pale_plinth(p, top, high, SHOWROOM, 0.5 * m)
    _foot(p, top, high)
    _patch(p, left, front, (0.03, 0.97), (1.7 * m, high - 0.5 * m), PLATE_GLASS)
    _patch(p, front, right, (0.04, 0.55), (1.7 * m, high - 0.5 * m), shade(PLATE_GLASS, 0.78))
    _patch(p, left, front, (0.03, 0.97), (high - 1.2 * m, high - 0.5 * m), shade(PLATE_GLASS, 0.72))
    for k, t in ((4, 0.2), (1, 0.8)):
        colour = CAR_COLOURS[k]
        span = (t - 0.1, t + 0.1)
        _patch(p, left, front, span, (high - 1.4 * m, high - 0.75 * m), colour)
        _patch(
            p,
            left,
            front,
            (t - 0.055, t + 0.06),
            (high - 1.95 * m, high - 1.4 * m),
            shade(colour, 0.9),
        )
        _patch(
            p,
            left,
            front,
            (t - 0.045, t + 0.05),
            (high - 1.85 * m, high - 1.45 * m),
            shade(PLATE_GLASS, 1.1),
        )
        for s in (t - 0.065, t + 0.065):
            wx, wy = _on_wall(left, front, s, high - 0.75 * m)
            p.ellipse(wx - 0.35 * m, wy - 0.35 * m, wx + 0.35 * m, wy + 0.35 * m, TYRE)
    for t in np.linspace(0.03, 0.97, 7):
        _bar(
            p,
            _on_wall(left, front, t, 1.7 * m),
            _on_wall(left, front, t, high - 0.5 * m),
            0.12 * m,
            CHROME,
        )
    for t in np.linspace(0.04, 0.55, 3):
        _bar(
            p,
            _on_wall(front, right, t, 1.7 * m),
            _on_wall(front, right, t, high - 0.5 * m),
            0.12 * m,
            shade(CHROME, 0.8),
        )
    _patch(p, front, right, (0.64, 0.94), (2 * m, high), (150, 152, 158))
    for drop in np.arange(2.3 * m, high, 0.45 * m):
        _bar(
            p,
            _on_wall(front, right, 0.64, drop),
            _on_wall(front, right, 0.94, drop),
            0.06 * m,
            (120, 122, 128),
        )
    _patch(p, left, front, (0.42, 0.58), (2.6 * m, high - 0.5 * m), shade(PLATE_GLASS, 0.6))
    _bar(
        p,
        _on_wall(left, front, 0.5, 2.6 * m),
        _on_wall(left, front, 0.5, high - 0.5 * m),
        0.1 * m,
        CHROME,
    )
    for a, b, tone in ((left, front, 1.0), (front, right, 0.8)):
        _patch(p, a, b, (0, 1), (-0.6 * m, 1.5 * m), shade(DEALER_RED, tone))
        _bar(
            p, _on_wall(a, b, 0, 1.5 * m), _on_wall(a, b, 1, 1.5 * m), 0.15 * m, shade(CHROME, tone)
        )
    for t0, t1 in ((0.12, 0.2), (0.24, 0.3), (0.34, 0.46), (0.5, 0.56), (0.6, 0.72), (0.76, 0.86)):
        _bar(
            p,
            _on_wall(left, front, t0, 0.45 * m),
            _on_wall(left, front, t1, 0.45 * m),
            0.55 * m,
            (250, 248, 244),
        )
    for u, v in ((5, 9), (13, 9)):
        x, y = at(u, v)
        p.hall(x, y - high, 5 * m, 3 * m, 0.5 * m, shade(SHOWROOM, 0.9), PLATE_GLASS)
    for t, colour in ((0.06, CAR_COLOURS[1]), (0.38, DEALER_RED), (0.7, CAR_COLOURS[2])):
        x, y = _on_wall(left, front, t, -0.6 * m)
        _pennant_flag(p, x, y, 4, colour)
    _pennant_flag(p, *_on_wall(front, right, 0.5, -0.6 * m), 4, DEALER_RED)
    px, py = at(-1.5, -8)
    _bar(p, (px, py), (px, py - 7 * m), 0.35 * m, (90, 92, 98))
    p.rect(px - 1.6 * m, py - 10.4 * m, px + 1.6 * m, py - 6.6 * m, DEALER_RED)
    p.ellipse(px - 1.1 * m, py - 9.6 * m, px + 1.1 * m, py - 7.4 * m, (250, 248, 244))
    p.ellipse(px - 0.7 * m, py - 9.1 * m, px + 0.7 * m, py - 7.9 * m, shade(DEALER_RED, 0.9))
    for u, k in ((2.6, 0), (7.2, 3), (11.8, 2), (16.4, 1)):
        _auto(p, *at(u, -4.5), CAR_COLOURS[k])
    return p.sprite()


JEWEL_FRONT: RGB = (40, 42, 70)
GOLD: RGB = (234, 188, 64)
DIAMOND: RGB = (204, 240, 255)
VELVET: RGB = (118, 30, 50)
GLOW: RGB = (252, 234, 176)
SPARK: RGB = (255, 255, 255)


def _glint(p: Painter, x: float, y: float, size: float, colour: RGB = SPARK) -> None:
    """A four-pointed glint of light at (x, y), `size` units from its middle to its tips."""
    _bar(p, (x - size, y), (x + size, y), size * 0.35, colour)
    _bar(p, (x, y - size), (x, y + size), size * 0.35, colour)


def _diamond(p: Painter, x: float, y: float, wide: float) -> None:
    """A brilliant-cut diamond `wide` across with the middle of its table at (x, y): a crown of
    facets over a pavilion narrowing to its point, lit on the left and glinting."""
    h = wide / 2
    girdle, point = y + wide * 0.28, y + wide * 0.9
    p.polygon([(x - h * 0.55, y), (x + h * 0.55, y), (x + h, girdle), (x - h, girdle)], DIAMOND)
    p.polygon([(x, girdle), (x + h, girdle), (x, point)], shade(DIAMOND, 0.72))
    p.polygon([(x - h, girdle), (x, girdle), (x, point)], shade(DIAMOND, 0.9))
    edge = shade(DIAMOND, 0.55)
    for a, b in (
        ((x - h * 0.55, y), (x - h * 0.3, girdle)),
        ((x + h * 0.55, y), (x + h * 0.3, girdle)),
        ((x - h * 0.3, girdle), (x, point)),
        ((x + h * 0.3, girdle), (x, point)),
        ((x - h, girdle), (x + h, girdle)),
    ):
        _bar(p, a, b, wide * 0.04, edge)
    _glint(p, x - h * 0.35, y + wide * 0.1, wide * 0.22)


def _jewels(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: tuple[float, float]
) -> None:
    """A display window in a gilt frame on the wall under the edge from `a` to `b`: lit warm
    inside, with shelves of dark red velvet laid with gold rings and chains and diamonds
    glittering, from `span[0]` to `span[1]` along it and `drop[0]` to `drop[1]` down it."""
    m = METRE
    t0, t1 = span
    d0, d1 = drop
    _patch(p, a, b, span, (d0 - 0.15 * m, d1 + 0.15 * m), GOLD)
    _patch(p, a, b, (t0 + 0.015, t1 - 0.015), drop, GLOW)
    shelves = (d0 + (d1 - d0) * 0.45, d1 - 0.1 * m)
    for shelf in shelves:
        _patch(p, a, b, (t0 + 0.015, t1 - 0.015), (shelf - 0.25 * m, shelf), VELVET)
    count = max(2, round((t1 - t0) * 16))
    for k in range(count):
        t = t0 + (t1 - t0) * (k + 0.5) / count
        for row, shelf in enumerate(shelves):
            x, y = _on_wall(a, b, t, shelf - 0.38 * m)
            if (k + row) % 2:
                p.blob(x - 0.2 * m, y - 0.2 * m, x + 0.2 * m, y + 0.12 * m, GOLD)
                p.blob(x - 0.08 * m, y - 0.1 * m, x + 0.08 * m, y + 0.02 * m, GLOW)
            else:
                _glint(p, x, y - 0.05 * m, 0.22 * m)
    for k in range(count - 1):
        t = t0 + (t1 - t0) * (k + 1) / count
        x, y = _on_wall(a, b, t, d0 + 0.3 * m)
        _bar(p, (x - 0.25 * m, y), (x + 0.25 * m, y + 0.5 * m), 0.06 * m, GOLD)


def _street_clock(p: Painter, x: float, y: float) -> None:
    """A jeweller's gilt street clock on a fluted dark post 4.5 m tall standing on (x, y), a
    round white face in a gold rim on top under a gilt finial."""
    m = METRE
    p.blob(x - 0.4 * m, y - 0.2 * m, x + 0.4 * m, y + 0.2 * m, POST)
    p.rect(x - 0.3 * m, y - 1 * m, x + 0.3 * m, y, POST)
    _bar(p, (x, y - 1 * m), (x, y - 3.6 * m), 0.28 * m, POST)
    _bar(p, (x - 0.06 * m, y - 1 * m), (x - 0.06 * m, y - 3.6 * m), 0.06 * m, GOLD)
    cy = y - 4.4 * m
    p.ellipse(x - 0.95 * m, cy - 0.95 * m, x + 0.95 * m, cy + 0.95 * m, GOLD)
    p.ellipse(x - 0.7 * m, cy - 0.7 * m, x + 0.7 * m, cy + 0.7 * m, (250, 248, 240))
    _bar(p, (x, cy), (x, cy - 0.55 * m), 0.1 * m, (30, 30, 34))
    _bar(p, (x, cy), (x + 0.4 * m, cy + 0.15 * m), 0.1 * m, (30, 30, 34))
    p.polygon([(x - 0.3 * m, cy - 0.9 * m), (x + 0.3 * m, cy - 0.9 * m), (x, cy - 1.6 * m)], GOLD)


def jeweller() -> Sprite:
    """A jeweller's townhouse 8 by 8 m and two storeys high, its front painted a deep night
    blue and trimmed in gold: two lit display windows in gilt frames glittering with gold rings,
    chains and diamonds on red velvet, a glazed door under a gold fascia, gilt-framed windows
    above, a gold cornice, a dark mansard roof with a gilded dormer, a great cut diamond on
    a gilt stand on its roof, and a gilt street clock on the pavement before it."""
    m = METRE
    p = Painter(26, 30, (13, 21))
    high = 9 * m
    _worn(p, (-6, -high), (0, 3 - high), (-0.05, 1.05), high, 3, PAVING)
    p.shadow(9, 3)
    top = p.box(0, 0, 6, high, JEWEL_FRONT, (50, 54, 72))
    left, front, right, _ = top
    _pale_plinth(p, top, high, JEWEL_FRONT, 0.5 * m)
    _foot(p, top, high)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.8)):
        _patch(p, a, b, (0, 1), (0, 0.45 * m), shade(GOLD, tone))
        _patch(p, a, b, (0, 1), (3.5 * m, 4.6 * m), shade(GOLD, tone))
        _patch(p, a, b, (0.04, 0.96), (3.7 * m, 4.4 * m), shade(JEWEL_FRONT, 0.8 * tone))
        for t in (0.02, 0.98):
            _bar(
                p,
                _on_wall(a, b, t, 0.45 * m),
                _on_wall(a, b, t, high),
                0.3 * m,
                shade(GOLD, tone * 0.9),
            )
    for t0, t1 in ((0.12, 0.22), (0.26, 0.4), (0.44, 0.52), (0.56, 0.72), (0.76, 0.88)):
        _bar(
            p,
            _on_wall(left, front, t0, 4.05 * m),
            _on_wall(left, front, t1, 4.05 * m),
            0.35 * m,
            GOLD,
        )
    _jewels(p, left, front, (0.07, 0.36), (5.4 * m, 8 * m))
    _jewels(p, left, front, (0.64, 0.93), (5.4 * m, 8 * m))
    _jewels(p, front, right, (0.12, 0.88), (5.4 * m, 8 * m))
    _patch(p, left, front, (0.42, 0.58), (5 * m, high), GOLD)
    _patch(p, left, front, (0.44, 0.56), (5.2 * m, high), (30, 30, 44))
    _patch(p, left, front, (0.46, 0.54), (5.4 * m, high - 0.6 * m), shade(GLOW, 0.75))
    kx, ky = _on_wall(left, front, 0.53, 7.4 * m)
    p.blob(kx - 0.12 * m, ky - 0.12 * m, kx + 0.12 * m, ky + 0.12 * m, GOLD)
    _doorstep(p, left, front, (0.4, 0.6), high, shade(PAVING, 1.1))
    for a, b, spans, glass in (
        (left, front, (0.25, 0.75), (60, 72, 104)),
        (front, right, (0.3, 0.7), (48, 58, 86)),
    ):
        for t in spans:
            _patch(p, a, b, (t - 0.1, t + 0.1), (0.9 * m, 3.2 * m), GOLD)
            _patch(p, a, b, (t - 0.08, t + 0.08), (1.05 * m, 3.05 * m), glass)
            _bar(p, _on_wall(a, b, t, 1.05 * m), _on_wall(a, b, t, 3.05 * m), 0.08 * m, GOLD)
            _patch(p, a, b, (t - 0.12, t + 0.12), (0.6 * m, 0.85 * m), GOLD)
    _patch(p, left, front, (0.4, 0.6), (1.6 * m, 2.6 * m), GOLD)
    cx, cy = _on_wall(left, front, 0.5, 2.1 * m)
    p.ellipse(cx - 0.4 * m, cy - 0.4 * m, cx + 0.4 * m, cy + 0.4 * m, JEWEL_FRONT)
    _glint(p, cx, cy, 0.3 * m, DIAMOND)
    slate = (50, 54, 72)
    roof = p.box(0, -high, 5.2, 2.2 * m, slate, shade(slate, 1.2))
    _bands(p, roof[0], roof[1], roof[2], 0.1 * m, GOLD, shade(GOLD, 0.8))
    p.line([roof[0], roof[3], roof[2]], GOLD)
    dormer = p.hall(*_on_wall(left, front, 0.5, -0.8 * m), 1.6 * m, 1 * m, 1.6 * m, slate, None)
    _patch(p, dormer[0], dormer[1], (0.1, 0.9), (0.1 * m, 1.4 * m), GOLD)
    _patch(p, dormer[0], dormer[1], (0.22, 0.78), (0.3 * m, 1.4 * m), (60, 72, 104))
    _gable_across(p, dormer, 0.8 * m, slate, GOLD)
    sy = -high - 2.2 * m
    for dx in (-1, 1):
        _bar(p, (dx * 1.1 * m, sy + 0.3 * m), (dx * 0.2 * m, sy - 1.6 * m), 0.2 * m, GOLD)
    _diamond(p, 0, sy - 1.4 * m - 0.9 * 4.4 * m, 4.4 * m)
    _glint(p, 1.8 * m, sy - 4.6 * m, 0.6 * m)
    _street_clock(p, *_out(left, front, 0.12, high, 2.2))
    return p.sprite()


def airport() -> Sprite:
    """An apron with a marked runway and edge lights, a parked airliner, a hangar with a great
    sliding door, and a control tower with a glazed cab and a mast."""
    p = Painter(42, 34, (21, 24))
    p.polygon([(-20, 1), (-1, 9), (20, 0), (1, -8)], (112, 114, 108), (88, 90, 86))
    p.line([(-15, 2), (-1, 8)], (232, 232, 224))
    for k in range(4):
        x0 = 1 + 4.5 * k
        p.line([(x0, 6.2 - x0 * 0.43), (x0 + 2.5, 6.2 - (x0 + 2.5) * 0.43)], (232, 232, 224))
    for x in (-19, -14, -9, -4, 3, 9, 15):
        y = 1 + (x + 20) * 8 / 19 if x < -1 else 9 - (x + 1) * 9 / 21
        p.point(x, y - 1, (250, 214, 96))
    hangar = (208, 210, 212)
    top = p.box(-5, -2, 7, 7, hangar, (146, 150, 158))
    left, front, right, _ = top
    _pale_plinth(p, top, 7, hangar, 0.6)
    _foot(p, top, 7)
    _patch(p, left, front, (0.15, 0.85), (2.2, 7), (120, 124, 132))
    for t in (0.32, 0.5, 0.68):
        p.line([_on_wall(left, front, t, 2.4), _on_wall(left, front, t, 7)], (96, 100, 108))
    _patch(p, front, right, (0.15, 0.85), (1, 2), shade(GLASS, 0.8))
    p.line([_on_wall(top[0], top[3], 0.5, 0), _on_wall(front, right, 0.5, 0)], (170, 174, 180))
    white, tail = (240, 240, 236), (196, 56, 52)
    cx, cy = 8, 3.5
    p.polygon(
        [(cx - 4.5, cy + 2), (cx - 0.5, cy - 0.5), (cx + 1.5, cy + 0.5), (cx - 3.5, cy + 2.5)],
        shade(white, 0.85),
    )
    p.polygon(
        [(cx + 0.5, cy - 0.5), (cx + 3, cy - 2.5), (cx + 4, cy - 2.5), (cx + 2, cy + 0.5)],
        shade(white, 0.85),
    )
    p.polygon([(cx - 5, cy - 3), (cx + 4, cy + 1.5), (cx + 4, cy + 2.5), (cx - 5, cy - 2)], white)
    p.line([(cx - 5, cy - 2), (cx + 4, cy + 2.5)], shade(white, 0.65))
    p.point(cx + 3, cy + 1, GLASS)
    p.polygon([(cx - 5, cy - 3), (cx - 5, cy - 6), (cx - 3, cy - 2)], tail)
    tug = (240, 196, 64)
    p.box(cx - 7, cy + 3.5, 1.4, 1.4, tug, shade(tug, 1.12))
    p.point(cx - 7, cy + 1.5, (36, 36, 40))
    p.line([(cx - 5.6, cy + 3.5), (cx - 4, cy + 2.6)], POST)
    sock = (240, 120, 40)
    p.line([(-15, 0.5), (-15, -7)], POST)
    p.polygon([(-14.5, -7), (-11, -6.4), (-11, -5.6), (-14.5, -5.4)], sock)
    p.line([(-13, -6.7), (-13, -5.4)], (240, 240, 236))
    tower = (192, 194, 198)
    p.polygon([(7, -1), (12, -1), (11, 0.5), (6, 0.5)], shade(tower, 0.82))
    p.rect(8, -18, 11, -1, tower)
    p.line([(8.5, -0.5), (11.5, -0.5)], shade(tower, 0.5))
    for y in (-14, -10, -6):
        p.point(9, y, WINDOW)
    p.rect(9, -3, 10, -1, DOORWAY)
    p.rect(6, -22, 13, -18, GLASS)
    p.line([(9, -22), (9, -18)], shade(GLASS, 0.6))
    p.line([(11, -22), (11, -18)], shade(GLASS, 0.6))
    p.line([(7, -21), (8, -21)], shade(GLASS, 1.4))
    p.rect(5.5, -23.5, 13.5, -22, shade(tower, 0.8))
    p.line([(10, -24), (10, -28)], (70, 70, 76))
    p.point(10, -29, (230, 60, 50))
    return p.sprite()


def tavern() -> Sprite:
    """A half-timbered inn on a stone plinth with a lit window, a door under a lantern, a sign hung
    from a bracket, a beer barrel by the door, and a tiled roof with a smoking chimney."""
    p = Painter(25, 26, (12, 20))
    p.shadow(9, 3)
    wall, beam = (234, 222, 192), (92, 64, 44)
    _worn(p, (-7, -7), (0, -3.5), (0.5, 0.95), 7, 4)
    top = p.box(0, 0, 7, 7, wall, None)
    left, front, right, _ = top
    for a, b in ((left, front), (front, right)):
        for t in (0, 0.5, 1):
            p.line([_on_wall(a, b, t, 0), _on_wall(a, b, t, 7)], beam)
        _band(p, a, b, 3.5, beam)
        p.line([_on_wall(a, b, 0, 0.5), _on_wall(a, b, 0.5, 3.5)], beam)
    _pale_plinth(p, top, 7, STONE, 1)
    _patch(p, left, front, (0.1, 0.38), (4.4, 5.8), LIT)
    p.line([_on_wall(left, front, 0.24, 4.4), _on_wall(left, front, 0.24, 5.8)], beam)
    _patch(p, left, front, (0.66, 0.88), (1.2, 2.6), WINDOW)
    _patch(p, front, right, (0.64, 0.86), (1.2, 2.6), shade(WINDOW, 0.75))
    _patch(p, front, right, (0.12, 0.36), (4.4, 5.8), shade(LIT, 0.8))
    _patch(p, left, front, (0.6, 0.84), (4, 7), (70, 48, 34))
    p.point(*_on_wall(left, front, 0.72, 3.2), (250, 214, 96))
    _foot(p, top, 7)
    _doorstep(p, left, front, (0.58, 0.86), 7, shade(STONE, 1.15))
    roof = (126, 74, 54)
    p.gable(top, 5, roof, wall)
    _tiles(p, top, 5, roof)
    p.rect(1, -14, 2.5, -10, shade(STONE, 0.8))
    p.blob(1.5, -17, 4, -14.5, (214, 214, 218))
    p.line([(-7, -5), (-10, -5)], (52, 48, 46))
    p.rect(-11, -4, -8, -1, (170, 46, 42))
    p.point(-10, -3, (240, 200, 90))
    _cask(p, *_out(left, front, 0.44, 7, 1.8))
    return p.sprite()


def sushi_hut() -> Sprite:
    """A small wooden eating house on a stone step, a red cloth curtain hung over its door, a
    paper lantern glowing beside it, a slatted window and a dark tiled roof with a sign board
    under the eaves."""
    p = Painter(22, 24, (11, 18))
    p.shadow(7, 2.5)
    wall, slat = (178, 132, 84), (120, 84, 52)
    _worn(p, (-5, -5), (0, -2.5), (0.4, 0.95), 5, 2.5, PAVING)
    top = p.box(0, 0, 6, 6, wall, None)
    left, front, right, _ = top
    for a, b in ((left, front), (front, right)):
        for t in (0.0, 1.0):
            p.line([_on_wall(a, b, t, 0), _on_wall(a, b, t, 6)], slat)
        _band(p, a, b, 0.6, slat)
    _pale_plinth(p, top, 6, STONE, 0.8)
    _patch(p, left, front, (0.1, 0.38), (2, 4), LIT)
    for t in (0.17, 0.24, 0.31):
        p.line([_on_wall(left, front, t, 2), _on_wall(left, front, t, 4)], slat)
    _patch(p, left, front, (0.5, 0.86), (1.4, 6), (66, 46, 36))
    curtain = (196, 48, 42)
    _patch(p, left, front, (0.5, 0.86), (1.4, 3.4), curtain)
    for t in (0.62, 0.74):
        p.line([_on_wall(left, front, t, 1.4), _on_wall(left, front, t, 3.4)], shade(curtain, 0.7))
    p.point(*_on_wall(left, front, 0.56, 2), (244, 236, 220))
    _patch(p, front, right, (0.25, 0.75), (1.6, 3.6), shade(LIT, 0.78))
    _foot(p, top, 6)
    _doorstep(p, left, front, (0.48, 0.88), 6, shade(STONE, 1.15))
    roof = (64, 62, 70)
    p.gable(top, 4, roof, wall)
    _tiles(p, top, 4, roof, (0.3, 0.6, 0.9))
    _patch(p, left, front, (0.3, 0.7), (-0.6, 0.6), (236, 226, 200))
    lantern = (230, 66, 46)
    lx, ly = _out(left, front, 0.44, 1.2, 1)
    p.line([(lx, ly - 0.8), (lx, ly)], POST)
    p.blob(lx - 0.9, ly, lx + 0.9, ly + 2, lantern)
    p.point(lx - 0.4, ly + 0.6, (255, 196, 120))
    return p.sprite()


STORE_STONE: RGB = (236, 204, 160)
STORE_GREEN: RGB = (34, 118, 80)
STORE_CREAM: RGB = (248, 240, 222)
BANNER: RGB = (192, 40, 52)
FASHIONS: tuple[RGB, ...] = (
    (214, 52, 70),
    (60, 110, 196),
    (240, 196, 60),
    (60, 160, 110),
    (226, 120, 170),
    (150, 80, 170),
)


def _striped_awning(
    p: Painter,
    a: Point,
    b: Point,
    span: tuple[float, float],
    drop: float,
    reach: float,
    stripes: int,
) -> None:
    """A green and cream striped awning sloping out `reach` from the wall under the edge from `a`
    to `b`, from `span[0]` to `span[1]` along it and hung from `drop` down it, with a valance of
    the same stripes along its front."""
    t0, t1 = span
    dx, dy = _outward(a, b, reach)
    fall = reach * 0.9
    for k in range(stripes):
        s0, s1 = t0 + (t1 - t0) * k / stripes, t0 + (t1 - t0) * (k + 1) / stripes
        colour = (STORE_GREEN, STORE_CREAM)[k % 2]
        near = [_on_wall(a, b, s0, drop), _on_wall(a, b, s1, drop)]
        out = [(x + dx, y + dy + fall) for x, y in near]
        p.polygon([near[0], near[1], out[1], out[0]], colour)
        p.polygon(
            [
                out[0],
                out[1],
                (out[1][0], out[1][1] + reach * 0.35),
                (out[0][0], out[0][1] + reach * 0.35),
            ],
            shade(colour, 0.85),
        )


def _mannequins(
    p: Painter, a: Point, b: Point, span: tuple[float, float], foot: float, tall: float
) -> None:
    """Dressed figures in a shop window on the wall under the edge from `a` to `b`, spread from
    `span[0]` to `span[1]` along it and standing `foot` down it, `tall` high, each in a dress or
    coat of its own colour under a hat."""
    t0, t1 = span
    count = max(2, round((t1 - t0) * 12))
    for k in range(count):
        t = t0 + (t1 - t0) * (k + 0.5) / count
        colour = FASHIONS[(k * 2 + round(t0 * 10)) % len(FASHIONS)]
        x, y = _on_wall(a, b, t, foot)
        p.polygon(
            [(x - 0.35, y), (x + 0.35, y), (x + 0.2, y - tall * 0.7), (x - 0.2, y - tall * 0.7)],
            colour,
        )
        p.blob(x - 0.22, y - tall * 0.92, x + 0.22, y - tall * 0.68, (232, 196, 170))
        p.polygon(
            [(x - 0.4, y - tall * 0.86), (x + 0.4, y - tall * 0.86), (x, y - tall)],
            shade(colour, 0.7),
        )


def department_store() -> Sprite:
    """A wide, ornate store of three storeys in warm sandstone with cream cornices: great display
    windows lit and dressed with figures in bright fashions under green and cream striped
    awnings, a glazed entrance in a gilt frame under a green sign lettered in gold, arched windows
    above each under its own little striped awning, long red banners with gold edges hung down
    its corners, a green copper dome ribbed in gold under a flag, pennants on its corners and
    lamps before it."""
    p = Painter(32, 44, (16, 33))
    p.shadow(11, 3.5)
    wall = STORE_STONE
    _worn(p, (-10, -13), (0, -8), (-0.05, 1.05), 13, 2.5, PAVING)
    top = p.box(0, 0, 10, 13, wall, (128, 118, 124))
    left, front, right, _ = top
    display = (252, 238, 206)
    _patch(p, left, front, (0.05, 0.92), (9, 12), display)
    _patch(p, front, right, (0.08, 0.95), (9, 12), shade(display, 0.82))
    _mannequins(p, left, front, (0.05, 0.36), 11.9, 2.1)
    _mannequins(p, left, front, (0.6, 0.92), 11.9, 2.1)
    _mannequins(p, front, right, (0.08, 0.95), 11.9, 2.1)
    for a, b, spans, tone in (
        (left, front, (0.05, 0.3, 0.68, 0.92), 0.85),
        (front, right, (0.08, 0.36, 0.64, 0.95), 0.65),
    ):
        for t in spans:
            p.line([_on_wall(a, b, t, 9), _on_wall(a, b, t, 12)], shade(wall, tone))
    _patch(p, left, front, (0.38, 0.58), (9, 12.2), GOLD)
    _patch(p, left, front, (0.4, 0.56), (9.4, 12.2), DOORWAY)
    _patch(p, left, front, (0.42, 0.54), (9.8, 12.2), shade(display, 0.7))
    p.line([_on_wall(left, front, 0.48, 9.8), _on_wall(left, front, 0.48, 12.2)], GOLD)
    _pale_plinth(p, top, 13, wall, 0.8)
    _foot(p, top, 13)
    _doorstep(p, left, front, (0.36, 0.6), 13, shade(PAVING, 1.1))
    for a, b, tone in ((left, front, 1.0), (front, right, 0.8)):
        _patch(p, a, b, (0, 1), (0, 1.2), shade(STORE_CREAM, tone))
        _band(p, a, b, 1.2, shade(wall, 0.7 * tone))
        _patch(p, a, b, (0, 1), (8, 8.8), shade(STORE_CREAM, tone))
    _patch(p, left, front, (0.3, 0.66), (7.1, 8.5), GOLD)
    _patch(p, left, front, (0.32, 0.64), (7.3, 8.3), STORE_GREEN)
    for t0, t1 in ((0.35, 0.4), (0.42, 0.46), (0.48, 0.54), (0.56, 0.61)):
        p.line([_on_wall(left, front, t0, 7.8), _on_wall(left, front, t1, 7.8)], GOLD)
    _striped_awning(p, left, front, (0.05, 0.36), 8.9, 1.1, 6)
    _striped_awning(p, left, front, (0.6, 0.92), 8.9, 1.1, 6)
    _striped_awning(p, front, right, (0.08, 0.95), 8.9, 1.1, 12)
    for a, b, spans, banner, glass in (
        (left, front, (0.12, 0.3, 0.48, 0.66), 0.85, WINDOW),
        (front, right, (0.34, 0.52, 0.7, 0.88), 0.15, shade(WINDOW, 0.75)),
    ):
        for row in (2.5, 5.5):
            for k, t in enumerate(spans):
                lit = (k + round(row)) % 3 == 0
                _patch(p, a, b, (t - 0.045, t + 0.045), (row - 0.3, row + 2.2), STORE_CREAM)
                _patch(p, a, b, (t - 0.03, t + 0.03), (row, row + 2), LIT if lit else glass)
                _striped_awning(p, a, b, (t - 0.05, t + 0.05), row - 0.2, 0.7, 3)
        _patch(p, a, b, (banner - 0.05, banner + 0.05), (1.6, 7.6), GOLD)
        _patch(p, a, b, (banner - 0.04, banner + 0.04), (1.6, 7.2), BANNER)
        cx, cy = _on_wall(a, b, banner, 3.4)
        p.blob(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5, GOLD)
    p.line([left, front, right], shade(wall, 1.15))
    for (x, y), colour in ((left, (60, 110, 196)), (right, (240, 196, 60))):
        p.rect(x - 0.5, y - 2, x + 0.5, y, STORE_CREAM)
        p.line([(x, y - 2), (x, y - 6)], POST)
        p.polygon([(x, y - 6), (x + 3, y - 5.2), (x, y - 4.4)], colour)
    _turned(p, 0, -8, [(0, 3), (3, 3)], STORE_CREAM)
    for dx in (-2, 0):
        p.line([(dx, -9), (dx, -10)], WINDOW)
    dome = [(0, 3.4), (2, 2.8), (3.5, 1.6), (4.5, 0)]
    _turned(p, 0, -11, dome, VERDIGRIS)
    _ribs(p, 0, -11, dome, (1.1, 1.9, 2.6), GOLD)
    p.blob(-0.5, -16.2, 0.5, -15.2, GOLD)
    _flag(p, 0, -15, 8, BANNER)
    for t in (0, 1):
        _street_lamp(p, *_out(left, front, t, 13, 1.8), 7)
    return p.sprite()


def hotel() -> Sprite:
    """A tall, elegant hotel: a rusticated ground floor with a canopy over its door and a potted
    tree, rows of windows with a few lit, a cornice, a mansard roof with lit dormers and chimneys,
    and its name lit up down a sign at the corner."""
    p = Painter(27, 46, (12, 37))
    p.shadow(8, 3, reach=6)
    wall, slate = (226, 198, 176), (84, 88, 104)
    s, height = 7, 22
    _worn(p, (-s, -height), (0, s / 2 - height), (-0.05, 1.05), height, 3, PAVING)
    top = p.box(0, 0, s, height, wall, None)
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (18.6, height), shade(wall, 0.88))
    _patch(p, front, right, (0, 1), (18.6, height), shade(wall, 0.68))
    _pale_plinth(p, top, height, wall, 0.8)
    for a, b, tone in ((left, front, 0.75), (front, right, 0.58)):
        for drop in (19.8, 21):
            _band(p, a, b, drop, shade(wall, tone))
    lit_spots = {(0.45, 5), (0.92, 11), (0.2, 14), (0.7, 2)}
    for drop in tuple(2 + 3 * f for f in range(6)):
        for t in (0.2, 0.45, 0.7, 0.92):
            colour = LIT if (t, drop) in lit_spots else WINDOW
            _windows(p, left, front, (t,), (drop,), 2, colour)
    _windows(
        p,
        front,
        right,
        (0.25, 0.5, 0.75),
        tuple(2 + 3 * f for f in range(6)),
        2,
        shade(WINDOW, 0.75),
    )
    _windows(p, front, right, (0.5,), (19.5,), 1.5, shade(LIT, 0.75))
    for a, b, tone in ((left, front, 1.15), (front, right, 0.9)):
        _band(p, a, b, 0.6, shade(wall, tone))
        _band(p, a, b, 18.4, shade(wall, tone))
    p.box(0, -height, s - 1, 3, slate, shade(slate, 0.8))
    for x in (-3, 0, 3):
        p.point(x, -height - 1 + abs(x) * 0.5, (236, 214, 140))
    for x in (-3, 3):
        p.rect(x - 0.7, -height - 5.8, x + 0.7, -height - 4, shade(BRICK, 0.85))
    _patch(p, left, front, (0.42, 0.62), (19, 22), (70, 52, 44))
    p.point(*_on_wall(left, front, 0.52, 20), LIT)
    awning = (170, 44, 48)
    _foot(p, top, height)
    _worn(p, left, front, (0.45, 0.59), height, 3, awning)
    _doorstep(p, left, front, (0.38, 0.66), height, shade(PAVING, 1.1))
    a, b = _on_wall(left, front, 0.36, 18), _on_wall(left, front, 0.68, 18)
    p.polygon([a, b, (b[0] - 3, b[1] + 2), (a[0] - 3, a[1] + 2)], awning, shade(awning, 0.6))
    p.line([(a[0] - 3, a[1] + 2), (b[0] - 3, b[1] + 2)], (240, 214, 120))
    p.line([(a[0] - 3, a[1] + 2), (a[0] - 3, a[1] + 5)], (60, 56, 54))
    p.line([(b[0] - 3, b[1] + 2), (b[0] - 3, b[1] + 5)], (60, 56, 54))
    for t in (0.14, 0.88):
        x, y = _on_wall(left, front, t, 22)
        x, y = x - 1.5, y + 1
        p.rect(x - 0.5, y - 1, x + 0.5, y, (140, 96, 70))
        p.blob(x - 1.2, y - 3.4, x + 1.2, y - 0.8, (54, 108, 44))
    sign = (40, 48, 92)
    p.rect(-9, -19, -7, -8, sign)
    for y in (-17, -15, -13, -11):
        p.point(-8, y, (250, 214, 96))
    return p.sprite()


RUT: RGB = (142, 138, 128)
# The paving under a lot, the colour of the paved yards round town buildings
# (railroads.ui.pixel.ground_palette.YARD_COLOURS).
APRON: RGB = (164, 160, 148)
GRAVEL: RGB = (186, 180, 166)
SACK: RGB = (214, 210, 198)
PANE: RGB = (156, 200, 216)
SAND: RGB = (222, 196, 132)
CART: RGB = (150, 70, 52)


def _seam(p: Painter, a: Point, b: Point, drop: float, colour: RGB) -> None:
    """A thin line right along the wall under the edge from `a` to `b`, `drop` down it, as fine
    at every size as the thing it is drawn on."""
    _bar(p, _on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop), 0.07 * METRE, colour)


def _pallet(p: Painter, x: float, y: float, tall: float, load: RGB, layers: int) -> None:
    """A pallet a metre square standing on (x, y) with a load `tall` metres high on it in
    `layers` layers, the pallet's timber showing at its foot."""
    m = METRE
    p.hall(x, y, m, m, 0.15 * m, shade(TIMBER, 0.85), None)
    left, front, right, _ = p.hall(x, y - 0.15 * m, m, m, tall * m, load, shade(load, 1.12))
    for k in range(1, layers):
        drop = tall * m * k / layers
        _seam(p, left, front, drop, shade(load, 0.68))
        _seam(p, front, right, drop, shade(load, 0.55))


def _planks(p: Painter, x: float, y: float, long: float, high: float) -> None:
    """A stack of sawn planks `long` metres long and `high` high on bearers at (x, y), its
    layers showing on its long side and the plank ends on its short one."""
    m = METRE
    for along in (-0.35, 0.35):
        p.hall(
            x + along * long * m,
            y + along * long * m / 2,
            0.25 * m,
            1.5 * m,
            0.2 * m,
            shade(TIMBER, 0.6),
            None,
        )
    left, front, right, _ = p.hall(
        x, y - 0.2 * m, long * m, 1.2 * m, high * m, PLANK, shade(PLANK, 1.15)
    )
    layers = max(2, round(high / 0.25))
    for k in range(1, layers):
        _seam(p, left, front, high * m * k / layers, shade(PLANK, 0.72))
        _seam(p, front, right, high * m * k / layers, shade(PLANK, 0.5))
    for t in (1 / 3, 2 / 3):
        _bar(
            p,
            _on_wall(front, right, t, 0),
            _on_wall(front, right, t, high * m),
            0.07 * m,
            shade(PLANK, 0.5),
        )


def _sand(p: Painter, x: float, y: float, half: float, tall: float) -> None:
    """A heap of building sand `half` across and `tall` high on (x, y), its top rounded and its
    left side lit."""
    h, t = half, tall
    top = [(x - 0.45 * h, y - 0.8 * t), (x - 0.15 * h, y - t), (x + 0.2 * h, y - 0.95 * t)]
    rim = [(x - h, y), *top, (x + 0.55 * h, y - 0.7 * t), (x + h, y), (x, y + 0.4 * h)]
    p.polygon(rim, shade(SAND, 0.8), shade(SAND, 0.6))
    p.polygon([(x - h, y), *top, (x + 0.1 * h, y + 0.36 * h)], SAND)
    p.blob(x - 0.55 * h, y - 0.85 * t, x - 0.15 * h, y - 0.6 * t, shade(SAND, 1.1))


def _glass_rack(p: Painter, x: float, y: float) -> None:
    """An A-frame rack 3 m long with panes of glass leaning on both its sides, at (x, y)."""
    m = METRE
    long, tall = 3 * m, 2 * m
    a, b = (x - long / 2, y - long / 4), (x + long / 2, y + long / 4)
    _bar(p, (a[0] + 0.35 * m, a[1] - 0.2 * m), (a[0], a[1] - tall), 0.14 * m, TIMBER)
    for lean, tone in ((-0.35, 0.82), (0.35, 1.0)):
        p.polygon(
            [
                (a[0] + lean * m, a[1]),
                (b[0] + lean * m, b[1]),
                (b[0], b[1] - tall * 0.95),
                (a[0], a[1] - tall * 0.95),
            ],
            shade(PANE, tone),
            shade(PANE, 0.62),
        )
    for t in (0.2, 0.55):
        sx, sy = a[0] + long * t, a[1] + long * t / 2
        _bar(
            p,
            (sx + 0.1 * m, sy - tall * 0.25),
            (sx + 0.5 * m, sy - tall * 0.7),
            0.08 * m,
            (238, 248, 252),
        )
    _bar(p, (b[0] - 0.35 * m, b[1] - 0.2 * m), (b[0], b[1] - tall), 0.14 * m, TIMBER)
    _bar(p, (b[0] + 0.35 * m, b[1] - 0.2 * m), (b[0], b[1] - tall), 0.14 * m, shade(TIMBER, 0.8))


def _hand_cart(p: Painter, x: float, y: float) -> None:
    """A two-wheeled builder's hand cart 1.8 m long at (x, y), its shafts down on the ground."""
    m = METRE
    left, front, _, _ = p.hall(x, y - 0.6 * m, 1.8 * m, 1.1 * m, 0.4 * m, CART, shade(TIMBER, 0.9))
    _bar(p, left, (left[0] - 1.3 * m, left[1] + 1.5 * m), 0.12 * m, shade(TIMBER, 0.75))
    wx, wy = front[0] - 0.5 * m, front[1] + 0.1 * m
    p.ellipse(wx - 0.45 * m, wy - 0.45 * m, wx + 0.45 * m, wy + 0.55 * m, (54, 50, 48))


def _ruts(p: Painter, a: Point, b: Point) -> None:
    """The two wheel ruts carts have worn into the ground from `a` to `b`, a cart's width
    apart."""
    m = METRE
    (ax, ay), (bx, by) = a, b
    length = ((bx - ax) ** 2 + (2 * (by - ay)) ** 2) ** 0.5 or 1.0
    dx, dy = -(by - ay) * 2 / length * 0.7 * m, (bx - ax) / length * 0.35 * m
    for side in (-1, 1):
        _bar(p, (ax + side * dx, ay + side * dy), (bx + side * dx, by + side * dy), 0.3 * m, RUT)


def _spill(p: Painter, x: float, y: float, half: float, colour: RGB) -> None:
    """A flat patch of something spilt or tipped on the ground at (x, y), `half` across."""
    p.blob(x - half, y - half / 2, x + half, y + half / 2, colour)
    p.blob(x - half * 0.5, y - half * 0.3, x + half * 0.2, y + half * 0.05, shade(colour, 1.06))


def _apron(p: Painter, half: float) -> None:
    """The paving a lot stands on, an oval `half` across and half that down, solid in its
    middle and thinning out towards its uneven rim, so the ground round it shows through and
    the slabs give out into the grass."""
    for share, alpha in ((1.0, 50), (0.92, 100), (0.84, 160), (0.76, 220), (0.7, 255)):
        reach = half * share
        p.blob(-reach, -reach / 2, reach, reach / 2, APRON, alpha)
    for k, (u, v) in enumerate(_RIM):
        x, y = u * half * 0.86, v * half * 0.43
        size = half * (0.1 + 0.03 * (k % 3))
        p.blob(x - size, y - size / 2, x + size, y + size / 2, APRON, 120)


# Where an apron's rim reaches out, as shares of its half width across and half height down.
_RIM = (
    (-0.96, 0.18),
    (-0.7, -0.66),
    (-0.2, -0.95),
    (0.42, -0.86),
    (0.9, -0.36),
    (0.88, 0.42),
    (0.4, 0.88),
    (-0.3, 0.92),
    (-0.82, 0.5),
)


def builders_yard() -> Sprite:
    """A builder's yard behind a post-and-rail fence on a paved lot: a boarded timber store
    12 by 6 m with planks and sacks in its open front, a small office with a yellow sign,
    pallets of bricks and of cement sacks, stacks of sawn planks, a rack of glass panes, a heap
    of sand and a hand cart, wheel ruts worn through its gate, all at their true size beside
    the houses."""
    p = Painter(30, 22, (15, 14))
    m = METRE
    _apron(p, 13.6)
    yard = _paddock(0, 0, 9.6)
    left_corner, back_corner, right_corner, front_corner = yard
    for x, y, half in ((-2.4, 2.7, 1.9), (4.4, 1.9, 2), (-6.2, 0.6, 1.4)):
        _spill(p, x, y, half, GRAVEL)
    _spill(p, 2.2, -1.4, 2.2, shade(SAND, 0.94))
    gate = 0.42
    middle = _on_wall(left_corner, front_corner, gate + 0.09, 0)
    _ruts(p, (middle[0] - 4.2, middle[1] + 2.1), (middle[0] + 4, middle[1] - 2))
    _fence(p, left_corner, back_corner, m)
    _fence(p, back_corner, right_corner, m)
    _sand(p, 1.6, -2.5, 1.9, 1.5)
    office = (226, 216, 190)
    top = p.hall(4.9, -1.2, 5 * m, 4.5 * m, 2.7 * m, office, None)
    left, front, right, _ = top
    _foot(p, top, 2.7 * m)
    _patch(p, left, front, (0.58, 0.8), (0.6 * m, 2.7 * m), DOOR)
    _patch(p, left, front, (0.15, 0.42), (0.6 * m, 1.5 * m), WINDOW)
    _patch(p, front, right, (0.3, 0.7), (0.6 * m, 1.5 * m), shade(WINDOW, 0.8))
    p.gable(top, 1.4 * m, (150, 60, 46), office)
    _patch(p, left, front, (0.1, 0.9), (-0.9 * m, 0.1 * m), (236, 194, 60))
    store = (156, 118, 82)
    high = 3.4 * m
    top = p.hall(-2.5, -0.9, 12 * m, 6 * m, high, store, None)
    left, front, right, _ = top
    _foot(p, top, high)
    for k in range(1, 7):
        _bar(
            p,
            _on_wall(front, right, k / 7, 0),
            _on_wall(front, right, k / 7, high),
            0.1 * m,
            shade(store, 0.62),
        )
    _patch(p, left, front, (0.08, 0.92), (0.5 * m, high), DOORWAY)
    for k in range(4):
        drop = high - (0.5 + 0.45 * k) * m
        _patch(p, left, front, (0.12, 0.6), (drop - 0.35 * m, drop), shade(PLANK, 1 - 0.06 * k))
    _patch(p, left, front, (0.66, 0.88), (2 * m, high), shade(SACK, 0.8))
    for t in (0.08, 0.5, 0.92):
        _bar(
            p,
            _on_wall(left, front, t, 0),
            _on_wall(left, front, t, high),
            0.18 * m,
            shade(store, 0.75),
        )
    roof = (92, 96, 104)
    p.gable(top, 1.8 * m, roof, store)
    _tiles(p, top, 1.8 * m, roof, (0.5,))
    _hand_cart(p, 1.9, 0.8)
    _planks(p, 5.8, 1.2, 6, 1.2)
    _planks(p, 3.6, 2.6, 5, 0.8)
    for row, col, tall in ((0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 1, 1), (1, 2, 1)):
        bx = -5.9 + (col + row) * 1.15 * m
        by = 0.7 + (col - row) * 0.58 * m
        _pallet(p, bx, by, tall, BRICK, 2 * tall)
    for k, tall in enumerate((1.2, 0.9, 1.2)):
        _pallet(p, -3.2 + k * 1.15 * m, 2.6 + k * 0.58 * m, tall, SACK, round(tall / 0.3))
    _glass_rack(p, 0.6, 2.4)
    _fence(p, left_corner, _on_wall(left_corner, front_corner, gate, 0), m)
    _fence(p, _on_wall(left_corner, front_corner, gate + 0.18, 0), front_corner, m)
    _fence(p, front_corner, right_corner, m)
    return p.sprite()
