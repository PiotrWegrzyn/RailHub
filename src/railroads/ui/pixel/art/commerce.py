"""Ports, shops, taverns, department stores, hotels and airports."""

from railroads.ui.pixel.art.painter import GLASS, RGB, STONE, WINDOW, Painter, Sprite, shade
from railroads.ui.pixel.art.parts import _barrel, _flag, _on_wall, _patch, _turned, _windows


def port() -> Sprite:
    p = Painter(30, 30, (14, 22))
    p.polygon([(-13, 0), (0, 6.5), (13, 0), (0, -6.5)], STONE, shade(STONE, 0.6))
    p.line([(-13, 1), (0, 7.5), (13, 1)], shade(STONE, 0.5))
    p.shadow(7, 2)
    top = p.box(-4, 0, 6, 6, (196, 186, 166), None)
    p.gable(top, 3, (64, 92, 124), (196, 186, 166))
    crane = (214, 168, 40)
    p.line([(7, 1), (7, -18)], crane)
    p.line([(6, 1), (6, -18)], shade(crane, 0.7))
    p.line([(4, -18), (14, -16)], crane)
    p.line([(13, -16), (13, -9)], (70, 64, 58))
    p.rect(12, -9, 14, -7, (120, 70, 50))
    return p.sprite()


def shop(wall: RGB, awning: RGB, goods: RGB) -> Sprite:
    """A smart city shop with a bright awning and its wares in the window."""
    p = Painter(24, 26, (11, 20))
    p.shadow(9, 3)
    p.box(0, 0, 8, 12, wall, shade(wall, 0.7))
    p.polygon([(-8, -6), (0, -2), (0, -5), (-8, -9)], awning, shade(awning, 0.6))
    p.rect(-6, -5, -2, -1, WINDOW)
    p.point(-4, -3, goods)
    p.point(-3, -2, goods)
    return p.sprite()


def airport() -> Sprite:
    """A runway, a hangar and a control tower."""
    p = Painter(42, 34, (21, 24))
    p.polygon([(-20, 1), (-1, 9), (20, 0), (1, -8)], (112, 114, 108), (88, 90, 86))
    p.line([(-15, 2), (-1, 8)], (232, 232, 224))
    p.box(-5, -2, 7, 7, (208, 210, 212), (146, 150, 158))
    p.rect(8, -18, 11, -1, (192, 194, 198))
    p.rect(6, -22, 13, -18, GLASS)
    return p.sprite()


def tavern() -> Sprite:
    """A half-timbered inn with a sign hung from a bracket and a beer barrel by the door."""
    p = Painter(25, 26, (12, 20))
    p.shadow(9, 3)
    wall, beam = (234, 222, 192), (92, 64, 44)
    top = p.box(0, 0, 7, 7, wall, None)
    left, front, right, _ = top
    for a, b in ((left, front), (front, right)):
        for t in (0, 0.5, 1):
            p.line([_on_wall(a, b, t, 0), _on_wall(a, b, t, 7)], beam)
        p.line([_on_wall(a, b, 0, 3.5), _on_wall(a, b, 1, 3.5)], beam)
        p.line([_on_wall(a, b, 0, 0.5), _on_wall(a, b, 0.5, 3.5)], beam)
    _windows(p, left, front, (0.25,), (5,), 1)
    _patch(p, left, front, (0.62, 0.84), (4, 7), (70, 48, 34))
    p.gable(top, 5, (126, 74, 54), wall)
    p.line([(-7, -5), (-10, -5)], (52, 48, 46))
    p.rect(-11, -4, -8, -1, (170, 46, 42))
    p.point(-10, -3, (240, 200, 90))
    _barrel(p, -5, 4)
    return p.sprite()


def department_store() -> Sprite:
    """A wide, ornate store of three storeys with great display windows, a corner dome and a flag."""
    p = Painter(28, 42, (13, 32))
    p.shadow(11, 3.5)
    wall = (234, 216, 184)
    left, front, right, _ = p.box(0, 0, 10, 13, wall, (128, 118, 124))
    display = (150, 196, 222)
    _patch(p, left, front, (0.05, 0.92), (9, 12), display)
    _patch(p, front, right, (0.08, 0.95), (9, 12), shade(display, 0.75))
    for a, b in ((left, front), (front, right)):
        p.line([_on_wall(a, b, 0, 1), _on_wall(a, b, 1, 1)], shade(wall, 1.12))
        p.line([_on_wall(a, b, 0, 8), _on_wall(a, b, 1, 8)], shade(wall, 0.7))
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2.5, 5.5), 2)
    _windows(p, front, right, (0.34, 0.52, 0.7, 0.88), (2.5, 5.5), 2, shade(WINDOW, 0.75))
    _turned(p, 0, -8, [(0, 3), (3, 3)], wall)
    _turned(p, 0, -11, [(0, 3.4), (2, 2.8), (3.5, 1.6), (4.5, 0)], (98, 150, 132))
    _flag(p, 0, -15, 8)
    return p.sprite()


def hotel() -> Sprite:
    """A tall, elegant hotel with rows of windows under a mansard roof, a canopy over its door and
    its name lit up down a sign at the corner."""
    p = Painter(27, 46, (12, 37))
    p.shadow(8, 3, reach=6)
    wall, slate = (226, 198, 176), (84, 88, 104)
    s, height = 7, 22
    left, front, right, _ = p.box(0, 0, s, height, wall, None)
    _windows(p, left, front, (0.2, 0.45, 0.7, 0.92), tuple(2 + 3 * f for f in range(6)), 2)
    _windows(
        p,
        front,
        right,
        (0.25, 0.5, 0.75),
        tuple(2 + 3 * f for f in range(7)),
        2,
        shade(WINDOW, 0.75),
    )
    p.box(0, -height, s - 1, 3, slate, shade(slate, 0.8))
    for x in (-3, 0, 3):
        p.point(x, -height - 1 + abs(x) * 0.5, (236, 214, 140))
    _patch(p, left, front, (0.42, 0.62), (19, 22), (70, 52, 44))
    awning = (170, 44, 48)
    a, b = _on_wall(left, front, 0.36, 18), _on_wall(left, front, 0.68, 18)
    p.polygon([a, b, (b[0] - 3, b[1] + 2), (a[0] - 3, a[1] + 2)], awning, shade(awning, 0.6))
    p.line([(a[0] - 3, a[1] + 2), (a[0] - 3, a[1] + 5)], (60, 56, 54))
    sign = (40, 48, 92)
    p.rect(-9, -19, -7, -8, sign)
    for y in (-17, -15, -13, -11):
        p.point(-8, y, (250, 214, 96))
    return p.sprite()
