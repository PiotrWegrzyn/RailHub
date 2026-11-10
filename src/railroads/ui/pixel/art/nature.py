"""Trees and rocks."""

from railroads.ui.pixel.art.painter import (
    RGB,
    ROCK,
    TRUNK,
    Painter,
    Sprite,
    shade,
)


def pine(colors: tuple[RGB, RGB, RGB]) -> Sprite:
    dark, mid, light = colors
    p = Painter(11, 16, (5, 14))
    p.shadow(3, 1.5)
    p.line([(0, 0), (0, -2)], TRUNK)
    for i, half in enumerate((4, 3, 2)):
        base = -2 - 3.5 * i
        p.polygon([(-half, base), (half, base), (0, base - 5)], mid, dark)
        p.line([(-half + 1, base - 1), (-1, base - 4)], light)
    return p.sprite()


def leafy(colors: tuple[RGB, RGB, RGB]) -> Sprite:
    dark, mid, light = colors
    p = Painter(11, 14, (5, 12))
    p.shadow(3, 1.5)
    p.line([(0, 0), (0, -3)], TRUNK)
    p.ellipse(-4, -10, 4, -3, mid)
    p.blob(-3, -9, 0, -6, light)
    p.point(2, -5, dark)
    p.point(-1, -4, dark)
    return p.sprite()


def rocks() -> Sprite:
    p = Painter(14, 10, (7, 7))
    p.shadow(5, 1.5)
    p.polygon([(-6, 1), (-4, -4), (0, -6), (3, -2), (2, 1)], ROCK, shade(ROCK, 0.55))
    p.polygon([(-4, -4), (0, -6), (-1, -2)], shade(ROCK, 1.25))
    p.polygon([(1, 1), (3, -3), (6, -1), (5, 2)], shade(ROCK, 0.9), shade(ROCK, 0.5))
    return p.sprite()
