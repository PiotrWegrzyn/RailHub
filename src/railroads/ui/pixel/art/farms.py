"""Farms, orchards, plantations, pastures and the other fields."""

import itertools

from railroads.ui.pixel.art.painter import (
    FRUIT,
    HIVE,
    METAL,
    RGB,
    THATCH,
    TRUNK,
    WHITEWASH,
    WINDOW,
    Painter,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import _cow, _fence, _on_wall, _paddock, _patch, _turned, _windows


def farm() -> Sprite:
    p = Painter(22, 22, (9, 17))
    p.shadow(8, 3)
    barn = (172, 62, 46)
    top = p.box(-1, 0, 7, 6, barn, None)
    p.gable(top, 4, (112, 98, 84), barn)
    p.rect(-3, -4, -1, -1, shade(barn, 0.6))
    p.rect(7, -12, 10, 1, (206, 206, 200))
    p.ellipse(7, -14, 10, -11, METAL)
    return p.sprite()


def orchard(fruit: list[RGB] = FRUIT) -> Sprite:
    """A fruit grower's cottage, with crates of the fruit stacked beside it; the trees themselves
    are planted on the ground (railroads.ui.pixel.world)."""
    p = Painter(22, 20, (9, 15))
    p.shadow(8, 3)
    wall, crate = (226, 208, 166), (156, 108, 62)
    top = p.box(-1, 0, 7, 5, wall, None)
    p.gable(top, 4, (124, 82, 54), wall)
    p.rect(-4, -3, -3, -1, shade(wall, 0.55))
    p.point(-6, -2, WINDOW)
    p.point(3, -2, shade(WINDOW, 0.8))
    p.rect(6, -2, 10, 1, crate)
    p.rect(7, -4, 9, -2, crate)
    for x, colour in ((7, 0), (9, 1), (8, 2)):
        p.point(x, -3 if x == 8 else -2, fruit[colour % len(fruit)])
    return p.sprite()


def date_palms() -> Sprite:
    """Three date palms on a patch of sand, their fronds hung with dates."""
    p = Painter(32, 26, (16, 20))
    p.polygon([(-14, 0), (0, 7), (14, 0), (0, -7)], (214, 192, 134), (186, 164, 108))
    for x, y in ((-7, 1), (2, -2), (8, 2)):
        p.line([(x, y), (x + 1, y - 6), (x + 1, y - 11)], (120, 86, 54))
        top = (x + 1, y - 12)
        for dx, dy in ((-5, 2), (5, 2), (-4, -2), (4, -2), (0, -4)):
            p.line([top, (top[0] + dx, top[1] + dy)], (64, 128, 56))
        p.point(top[0] - 1, top[1] + 1, (110, 62, 36))
        p.point(top[0] + 1, top[1] + 1, (110, 62, 36))
    return p.sprite()


def bananas() -> Sprite:
    """Rows of broad-leaved banana plants with yellow bunches."""
    p = Painter(32, 22, (16, 14))
    p.polygon([(-14, 0), (0, 7), (14, 0), (0, -7)], (74, 120, 48), (60, 100, 40))
    for row in range(2):
        for col in range(3):
            x = -8 + 7 * col + 3 * row - 2
            y = -1 + 3.5 * row - 1.5 * col + 1
            p.line([(x, y), (x, y - 4)], (96, 120, 52))
            p.blob(x - 4, y - 8, x, y - 4, (52, 116, 44))
            p.blob(x, y - 9, x + 4, y - 5, (66, 140, 54))
            p.point(x + 1, y - 3, (236, 206, 64))
    return p.sprite()


def sugar() -> Sprite:
    """A field of tall sugar cane in rows."""
    p = Painter(30, 18, (15, 11))
    p.polygon([(-13, 0), (0, 6), (13, 0), (0, -6)], (132, 112, 72), (112, 94, 60))
    for row in range(4):
        for col in range(5):
            x = -9 + 4 * col + 2 * row - 3
            y = -3 + 2 * row - 1 * col + 2
            p.line([(x, y), (x, y - 4)], (118, 176, 82))
            p.point(x - 1, y - 4, (166, 206, 104))
    return p.sprite()


def fishery() -> Sprite:
    """A hut with racks of drying fish and a small boat."""
    p = Painter(30, 22, (15, 16))
    p.shadow(12, 3)
    top = p.box(-5, 0, 6, 5, (132, 112, 88), None)
    p.gable(top, 3, (84, 106, 130), (132, 112, 88))
    p.polygon([(3, 3), (13, 3), (11, 6), (5, 6)], (150, 62, 48), (92, 40, 34))
    p.line([(8, 3), (8, -6)], (88, 64, 44))
    for y in (-3, -1):
        p.line([(3, y), (6, y)], (120, 170, 196))
    return p.sprite()


def pasture() -> Sprite:
    """A shed on hilly grazing with sheep around it."""
    p = Painter(30, 22, (15, 16))
    top = p.box(-5, 0, 5, 4, (152, 124, 92), None)
    p.gable(top, 3, (110, 80, 60), (152, 124, 92))
    for x, y in ((3, -2), (8, 1), (4, 4), (11, -3), (-9, 5)):
        p.blob(x - 2, y - 1, x + 2, y + 1, (240, 236, 224))
        p.point(x + 2, y - 1, (50, 46, 44))
    return p.sprite()


def plantation(wall: RGB, roof: RGB) -> Sprite:
    """The small barn that stands on a field or grove."""
    p = Painter(18, 16, (9, 11))
    p.shadow(6, 2)
    top = p.box(0, 0, 5, 4, wall, None)
    p.gable(top, 3, roof, wall)
    return p.sprite()


def beekeeper() -> Sprite:
    """Straw hives in rows on a patch of flowers."""
    p = Painter(28, 18, (14, 11))
    p.polygon([(-12, 0), (0, 6), (12, 0), (0, -6)], (128, 168, 84), (104, 142, 70))
    for x, y in ((-6, 1), (0, 4), (0, -2), (6, 1)):
        p.ellipse(x - 2.5, y - 6, x + 2.5, y, HIVE)
        p.point(x, y - 1, shade(HIVE, 0.4))
    for x, y in ((-9, 0), (-3, -3), (4, -5), (9, -2), (3, 4)):
        p.point(x, y, (240, 226, 120))
    return p.sprite()


def incense_grove() -> Sprite:
    """Gnarled resin trees on pale ground."""
    p = Painter(28, 20, (14, 13))
    p.polygon([(-12, 0), (0, 6), (12, 0), (0, -6)], (190, 170, 130), (160, 142, 106))
    for x, y in ((-6, 0), (0, 3), (6, -1)):
        p.line([(x, y), (x - 1, y - 3), (x + 1, y - 5)], TRUNK)
        p.blob(x - 3, y - 9, x + 3, y - 5, (108, 124, 76))
        p.point(x - 1, y - 2, (200, 160, 210))
    return p.sprite()


def potato_farm() -> Sprite:
    """A low fieldstone cottage under thatch, with sacks of potatoes stood by its door."""
    p = Painter(26, 24, (12, 16))
    p.shadow(10, 3)
    wall = (172, 166, 150)
    top = p.hall(-2, -1, 11, 6, 4, wall, None)
    left, front, *_ = top
    for t, drop in ((0.15, 1), (0.4, 3), (0.85, 2), (0.6, 0.5)):
        p.point(*_on_wall(left, front, t, drop), shade(wall, 0.75))
    _windows(p, left, front, (0.25,), (2,), 1)
    _patch(p, left, front, (0.55, 0.72), (1, 4), (80, 62, 48))
    p.gable(top, 4, THATCH, wall)
    sack = (192, 162, 112)
    for x, y in ((8, 0), (5, 3), (9, 3.5)):
        p.ellipse(x - 1.5, y - 4, x + 1.5, y, sack)
        p.point(x, y - 4, shade(sack, 0.5))
    for x, y in ((3, 5), (4, 6), (6, 6)):
        p.point(x, y, (150, 104, 58))
    return p.sprite()


def cattle_ranch() -> Sprite:
    """A wooden cattle barn beside a fenced paddock with brown and black-and-white cows in it."""
    p = Painter(34, 26, (16, 18))
    p.shadow(12, 3.5)
    wood = (136, 86, 56)
    top = p.box(-8, -2, 6, 6, wood, None)
    left, front, *_ = top
    _patch(p, left, front, (0.45, 0.8), (1.5, 6), (236, 230, 214))
    _patch(p, left, front, (0.52, 0.73), (2.5, 6), shade(wood, 0.7))
    p.gable(top, 4, (86, 74, 70), wood)
    left, back, right, front = _paddock(6, 2, 9)
    _fence(p, left, back)
    _fence(p, back, right)
    _cow(p, 6, 0, (138, 84, 50), (96, 58, 36))
    _cow(p, 10, 4, (240, 238, 232), (40, 38, 42))
    _cow(p, 3, 4, (40, 38, 42), (240, 238, 232))
    _fence(p, left, front)
    _fence(p, front, right)
    return p.sprite()


def flax_field() -> Sprite:
    """A small whitewashed cottage with sheaves of flax stood up in stooks to dry beside it."""
    p = Painter(28, 24, (13, 16))
    p.shadow(10, 3)
    top = p.box(-6, -1, 5, 4, WHITEWASH, None)
    _windows(p, top[0], top[1], (0.3,), (2,), 1)
    _patch(p, top[0], top[1], (0.6, 0.8), (1, 4), (92, 70, 52))
    p.gable(top, 3, (128, 96, 70), WHITEWASH)
    straw = (204, 188, 120)
    for row in range(2):
        for col in range(3):
            x, y = 1 + 4 * col + 2 * row, -1 + 4 * row - col
            p.polygon([(x - 2, y), (x, y - 6), (x, y + 1)], straw, shade(straw, 0.6))
            p.polygon([(x, y - 6), (x + 2, y), (x, y + 1)], shade(straw, 0.78), shade(straw, 0.55))
            p.line([(x - 1, y - 3), (x + 1, y - 3)], (128, 104, 60))
    for x, y in ((-1, 5), (2, 6), (11, 6)):
        p.point(x, y, (110, 140, 230))
    return p.sprite()


def hop_garden() -> Sprite:
    """Tall poles strung with green hop bines, and a round oast house with a white cowl."""
    p = Painter(34, 40, (17, 30))
    pole, bine = (122, 96, 66), (88, 152, 58)
    for r in range(2):
        poles = [(-11 + 4 * k - 4 * r, -4 + 2 * k + 2 * r) for k in range(3)]
        tops = [(x, y - 16) for x, y in poles]
        for (x, y), (tx, ty) in zip(poles, tops, strict=True):
            p.line([(x, y), (tx, ty)], pole)
            p.line([(x - 1, y - 2), (x - 1, y - 12)], bine)
            p.line([(x + 1, y - 1), (x + 1, y - 14)], shade(bine, 0.85))
            for k in range(4):
                p.point(x + (k % 2) * 2 - 1, y - 3 - 3 * k, shade(bine, 1.4))
        p.line(tops, (70, 64, 58))
        for (x0, y0), (x1, y1) in itertools.pairwise(tops):
            p.line([(x0 + 1, y0 + 2), ((x0 + x1) / 2, (y0 + y1) / 2 + 3), (x1 - 1, y1 + 2)], bine)
    oast = (176, 92, 64)
    _turned(p, 10, 0, [(0, 4), (9, 4)], oast)
    _turned(p, 10, -9, [(0, 4.5), (8, 0.5)], (110, 76, 64))
    cowl = (244, 242, 236)
    p.polygon([(9, -17), (9, -20), (11, -22), (12, -19), (11, -17)], cowl, shade(cowl, 0.6))
    p.line([(12, -21), (14, -22)], cowl)
    stow = p.box(4, 4, 4, 4, (228, 222, 206), None)
    p.gable(stow, 3, (112, 82, 66), (228, 222, 206))
    return p.sprite()
