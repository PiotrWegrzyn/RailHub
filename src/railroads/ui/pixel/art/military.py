"""Outposts, bases and headquarters."""

from railroads.ui.pixel.art.painter import (
    MILITARY_ROOF,
    MILITARY_YARD,
    STONE,
    WINDOW,
    Painter,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import _flag


def outpost() -> Sprite:
    """A hut behind an earth bank, under a flag."""
    p = Painter(26, 30, (13, 23))
    p.shadow(10, 3)
    p.box(0, 0, 10, 2, (122, 94, 62), MILITARY_YARD)
    p.box(0, -1, 5, 6, MILITARY_YARD, MILITARY_ROOF)
    _flag(p, 7, -2, 16)
    return p.sprite()


def base() -> Sprite:
    """Barracks in a walled yard with a watchtower."""
    p = Painter(34, 34, (17, 26))
    p.shadow(14, 4)
    p.box(0, 0, 15, 3, STONE, MILITARY_YARD)
    p.box(-6, -2, 4, 6, MILITARY_YARD, MILITARY_ROOF)
    p.box(5, 0, 4, 6, MILITARY_YARD, MILITARY_ROOF)
    p.rect(9, -15, 12, -2, shade(STONE, 0.9))
    p.rect(8, -18, 13, -15, MILITARY_ROOF)
    _flag(p, -11, -2, 20)
    return p.sprite()


def headquarters() -> Sprite:
    """A great stone block with rows of windows, in a walled yard, under a tall flag."""
    p = Painter(40, 46, (19, 36))
    p.shadow(17, 5)
    p.box(0, 0, 18, 3, STONE, MILITARY_YARD)
    p.box(0, -2, 11, 15, (178, 172, 146), MILITARY_ROOF)
    for row in range(3):
        for col in range(3):
            p.point(-9 + 3 * col, -13 + 4 * row + 1.5 * col, WINDOW)
    _flag(p, 0, -22, 14)
    return p.sprite()
