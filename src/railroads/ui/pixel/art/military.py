"""Outposts, bases and headquarters."""

from railroads.ui.pixel.art.homes import TRODDEN, _doorstep, _foot, _worn
from railroads.ui.pixel.art.painter import (
    DOORWAY,
    METAL,
    MILITARY_FLAG,
    MILITARY_ROOF,
    MILITARY_YARD,
    RGB,
    STONE,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import _flag, _on_wall, _patch

type Top = tuple[Point, Point, Point, Point]

SANDBAG: RGB = (196, 178, 124)

OLIVE: RGB = (110, 118, 74)

FLAGS: RGB = (176, 170, 150)


def _plinth(p: Painter, top: Top, height: float, wall: RGB, depth: float = 0.8) -> None:
    """A darker band along the foot of a box's two front walls."""
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (height - depth, height), shade(wall, 0.7))
    _patch(p, front, right, (0, 1), (height - depth, height), shade(wall, 0.52))


def _barracks(p: Painter, x: float, y: float, half: float, height: float) -> None:
    """A low barrack hut with a door and windows under a flat roof with a vent along it."""
    top = p.box(x, y, half, height, MILITARY_YARD, MILITARY_ROOF)
    left, front, right, back = top
    _plinth(p, top, height, MILITARY_YARD)
    _foot(p, top, height)
    _patch(p, left, front, (0.15, 0.35), (1.8, 3), WINDOW)
    _patch(p, left, front, (0.58, 0.8), (2, height), DOORWAY)
    _doorstep(p, left, front, (0.54, 0.84), height, shade(STONE, 1.1))
    for span in ((0.2, 0.4), (0.6, 0.8)):
        _patch(p, front, right, span, (1.8, 3), shade(WINDOW, 0.75))
    p.line([left, front, right], shade(MILITARY_YARD, 1.12))
    p.line(
        [_on_wall(left, back, 0.5, 0), _on_wall(front, right, 0.5, 0)], shade(MILITARY_ROOF, 1.3)
    )


def _sandbags(p: Painter, a: Point, b: Point, spans: tuple[float, ...]) -> None:
    """A row of sandbags along the edge from `a` to `b`, one at each of `spans` along it."""
    for t in spans:
        x, y = _on_wall(a, b, t, 0)
        p.ellipse(x - 1.2, y - 1.4, x + 1.2, y, SANDBAG)


def _crate(p: Painter, x: float, y: float) -> None:
    p.box(x, y, 1.5, 1.6, OLIVE, shade(OLIVE, 1.2))


def outpost() -> Sprite:
    """A hut with a door, windows and a stove pipe behind an earth bank topped with sandbags and an
    opening at its front, with ammunition crates, under a flag."""
    p = Painter(26, 30, (13, 23))
    p.shadow(10, 3)
    earth = (122, 94, 62)
    _worn(p, (-10, -2), (0, 3), (0.62, 0.86), 2, 3.5)
    bank = p.box(0, 0, 10, 2, earth, MILITARY_YARD)
    left, front, right, back = bank
    _foot(p, bank, 2)
    _patch(p, left, front, (0.64, 0.84), (0, 2), shade(TRODDEN, 0.9))
    _worn(p, (-5, -7), (0, -4.5), (0.56, 0.84), 6, 1.8, shade(TRODDEN, 1.05))
    _sandbags(p, back, left, (0.3, 0.6))
    _sandbags(p, back, right, (0.4, 0.7))
    _crate(p, -6, -1)
    top = p.box(0, -1, 5, 6, MILITARY_YARD, MILITARY_ROOF)
    hut_left, hut_front, hut_right, _ = top
    _plinth(p, top, 6, MILITARY_YARD)
    _patch(p, hut_left, hut_front, (0.15, 0.38), (1.8, 3), WINDOW)
    _patch(p, hut_left, hut_front, (0.58, 0.82), (2.2, 6), DOORWAY)
    _patch(p, hut_front, hut_right, (0.3, 0.55), (1.8, 3), shade(WINDOW, 0.75))
    _foot(p, top, 6)
    _doorstep(p, hut_left, hut_front, (0.56, 0.84), 6, (150, 116, 76))
    p.line([hut_left, hut_front, hut_right], shade(MILITARY_YARD, 1.12))
    p.rect(1.5, -12, 2.5, -8, shade(METAL, 0.7))
    p.blob(1.5, -15, 4, -12.5, (200, 200, 204))
    _crate(p, 4.5, 1)
    _sandbags(p, left, front, (0.15, 0.35, 0.55))
    _sandbags(p, front, right, (0.45, 0.65, 0.85))
    _flag(p, 7, -2, 16)
    return p.sprite()


def base() -> Sprite:
    """Barracks with doors and windows in a walled yard with a gate and a barrier, a stack of
    ammunition crates, and a watchtower with a searchlight."""
    p = Painter(34, 34, (17, 26))
    p.shadow(14, 4)
    _worn(p, (-15, -3), (0, 4.5), (0.58, 0.8), 3, 4)
    yard = p.box(0, 0, 15, 3, STONE, MILITARY_YARD)
    left, front, right, _ = yard
    _foot(p, yard, 3)
    for a, b, tone in ((left, front, 0.8), (front, right, 0.62)):
        p.line([_on_wall(a, b, 0, 1.5), _on_wall(a, b, 1, 1.5)], shade(STONE, tone))
    p.line([left, front, right], shade(STONE, 1.15))
    _patch(p, left, front, (0.6, 0.78), (0.4, 3), DOORWAY)
    for t, colour in ((0.56, MILITARY_FLAG), (0.64, (240, 240, 236)), (0.72, MILITARY_FLAG)):
        p.line([_on_wall(left, front, t, 1.4), _on_wall(left, front, t + 0.08, 1.4)], colour)
    _barracks(p, -6, -2, 4, 6)
    _barracks(p, 5, 0, 4, 6)
    for x, y in ((-0.3, 0.6), (-1.5, 1.5), (0.8, 2.8)):
        _crate(p, x, y)
    tower = shade(STONE, 0.9)
    p.polygon([(8.5, -2), (12.5, -2), (12, -1), (8, -1)], shade(MILITARY_YARD, 0.7))
    p.rect(9, -15, 12, -2, tower)
    for y in (-11, -7):
        p.line([(9, y), (12, y)], shade(tower, 0.75))
    p.line([(10.5, -13), (10.5, -2)], shade(tower, 0.7))
    p.rect(8.5, -15.5, 12.5, -14, shade(tower, 0.8))
    p.line([(9.5, -14.5), (11.5, -14.5)], DOORWAY)
    p.rect(8, -18, 13, -15, MILITARY_ROOF)
    p.rect(12.5, -17, 14, -15.5, (250, 240, 180))
    _flag(p, -11, -2, 20)
    return p.sprite()


def headquarters() -> Sprite:
    """A great stone block on a plinth: rows of windows with sills between cornices, a columned
    portico with a pediment over the door, a parapet and a mast on the roof, in a walled yard
    with a gate and a sentry box, under a tall flag."""
    p = Painter(40, 46, (19, 36))
    p.shadow(17, 5)
    _worn(p, (-18, -3), (0, 6), (0.5, 0.62), 3, 4)
    yard = p.box(0, 0, 18, 3, STONE, MILITARY_YARD)
    left, front, right, _ = yard
    _foot(p, yard, 3)
    p.line([left, front, right], shade(STONE, 1.15))
    _patch(p, left, front, (0.5, 0.62), (0.4, 3), DOORWAY)
    wall = (178, 172, 146)
    height = 15
    top = p.box(0, -2, 11, height, wall, MILITARY_ROOF)
    left, front, right, _ = top
    _plinth(p, top, height, wall)
    _foot(p, top, height)
    _worn(p, left, front, (0.42, 0.58), height, 2.6, FLAGS)
    sill = shade(wall, 1.2)
    for drop in (2, 5.5, 9, 12.3):
        for span in ((0.06, 0.16), (0.22, 0.32), (0.68, 0.78), (0.84, 0.94)):
            _patch(p, left, front, span, (drop, drop + 1.6), WINDOW)
            p.line(
                [
                    _on_wall(left, front, span[0], drop + 2),
                    _on_wall(left, front, span[1], drop + 2),
                ],
                sill,
            )
        for span in ((0.12, 0.22), (0.34, 0.44), (0.56, 0.66), (0.78, 0.88)):
            _patch(p, front, right, span, (drop, drop + 1.6), shade(WINDOW, 0.75))
    for a, b, tone in ((left, front, 1.15), (front, right, 0.88)):
        for drop in (0.6, 11):
            p.line([_on_wall(a, b, 0, drop), _on_wall(a, b, 1, drop)], shade(wall, tone))
    portico = shade(wall, 1.25)
    _patch(p, left, front, (0.35, 0.65), (5, height), shade(wall, 0.45))
    _patch(p, left, front, (0.46, 0.54), (11.4, height), DOORWAY)
    p.point(*_on_wall(left, front, 0.5, 9.5), shade(WINDOW, 1.2))
    for t in (0.36, 0.45, 0.55, 0.64):
        p.line([_on_wall(left, front, t, 6), _on_wall(left, front, t, height)], portico)
    _patch(p, left, front, (0.34, 0.66), (5, 6), portico)
    p.polygon(
        [
            _on_wall(left, front, 0.33, 5),
            _on_wall(left, front, 0.5, 2.4),
            _on_wall(left, front, 0.67, 5),
        ],
        portico,
        shade(wall, 0.7),
    )
    p.point(*_on_wall(left, front, 0.5, 4), shade(wall, 0.8))
    _doorstep(p, left, front, (0.33, 0.67), height, shade(wall, 1.15))
    p.line([left, front, right], shade(wall, 1.2))
    p.line([(6, -19), (6, -27)], shade(METAL, 0.7))
    p.line([(5, -25), (7, -25)], shade(METAL, 0.7))
    _flag(p, 0, -22, 14)
    sentry = p.box(-12, 3, 1.4, 4, OLIVE, shade(OLIVE, 1.25))
    _patch(p, sentry[0], sentry[1], (0.2, 0.8), (1, 2), DOORWAY)
    return p.sprite()
