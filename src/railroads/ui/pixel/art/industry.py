"""Factories and works of every kind."""

import math

import numpy as np

from railroads.ui.pixel.art.painter import (
    BRICK,
    CAR_COLOURS,
    CLOTH,
    COPPER,
    METAL,
    RGB,
    STONE,
    TIMBER,
    WHITEWASH,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import (
    _barrel,
    _cow,
    _fence,
    _heap,
    _iso,
    _on_wall,
    _paddock,
    _patch,
    _plate,
    _turned,
    _vault,
    _windows,
)


def factory() -> Sprite:
    """A small brick works with one chimney."""
    p = Painter(20, 28, (9, 22))
    p.shadow(7, 2.5)
    p.box(0, 0, 7, 6, BRICK, (96, 94, 100))
    for k in range(2):
        x, y = -5 + 5 * k, -6 + 1.2 * k
        p.polygon([(x, y), (x + 5, y - 2.5), (x + 5, y + 1)], METAL, shade(METAL, 0.55))
    p.rect(3, -16, 5, -6, shade(BRICK, 0.8))
    p.line([(3, -14), (5, -14)], (220, 220, 214))
    for i, (x, y, r) in enumerate([(4, -18, 1.5), (6, -20, 2)]):
        p.blob(x - r, y - r, x + r, y + r, (226, 226, 230) if i % 2 else (204, 204, 210))
    return p.sprite()


def big_factory() -> Sprite:
    p = Painter(30, 40, (14, 33))
    p.shadow(12, 4)
    p.box(0, 0, 12, 8, BRICK, (96, 94, 100))
    for k in range(3):
        x = -9 + 6 * k
        y = -8 + 1.5 * k
        p.polygon([(x, y), (x + 6, y - 3), (x + 6, y + 1.5)], METAL, shade(METAL, 0.55))
        p.polygon([(x, y), (x, y + 1.5 - 3), (x + 6, y - 3)], shade(METAL, 1.15))
    for x in (-6, 4):
        p.rect(x, -24, x + 2, -9, shade(BRICK, 0.8))
        p.line([(x, -22), (x + 2, -22)], (220, 220, 214))
    for i, (x, y, r) in enumerate(
        [(-5, -27, 2), (-3, -30, 2.5), (0, -32, 3), (5, -27, 2), (7, -30, 2.5)]
    ):
        p.blob(x - r, y - r, x + r, y + r, (226, 226, 230) if i % 2 else (204, 204, 210))
    return p.sprite()


def works(wall: RGB, roof: RGB, sign: RGB, stacks: int) -> Sprite:
    """A works: a long hall in its own colours with chimneys and a sign."""
    p = Painter(32, 40, (15, 32))
    p.shadow(13, 4)
    p.box(0, 0, 12, 8, wall, roof)
    p.rect(-9, -5, -4, -2, sign)
    p.point(-3, -3, WINDOW)
    p.point(5, -2, WINDOW)
    for k in range(stacks):
        x = 2 + 4 * k
        p.rect(x, -22, x + 2, -9, shade(BRICK, 0.8))
        p.blob(x - 1, -26, x + 3, -22, (150, 148, 150))
    return p.sprite()


def _sawtooth(
    p: Painter,
    top: tuple[Point, Point, Point, Point],
    teeth: int,
    rise: float,
    roof: RGB,
    glass: RGB,
    wall: RGB,
) -> None:
    """A sawtooth roof over a box's top: `teeth` ridges across it, each sloping down towards the
    front from a strip of glazing along its crest."""
    left, front, right, back = top
    for k in range(teeth):
        t0, t1 = k / teeth, (k + 1) / teeth
        near_up, far_up = _on_wall(left, front, t0, -rise), _on_wall(back, right, t0, -rise)
        near, far = _on_wall(left, front, t1, 0), _on_wall(back, right, t1, 0)
        p.polygon([_on_wall(left, front, t0, 0), near_up, near], wall, shade(wall, 0.55))
        p.polygon([near_up, far_up, far, near], roof, shade(roof, 0.6))
        p.line([near_up, far_up], glass)


def _chimney(p: Painter, x: float, foot: float, top: float, colour: RGB = BRICK) -> None:
    """A round brick chimney narrowing to a dark cap."""
    p.polygon(
        [(x - 2, foot), (x - 1, top), (x + 1, top), (x + 2, foot)], colour, shade(colour, 0.6)
    )
    p.line([(x - 1, foot - 1), (x - 1, top + 1)], shade(colour, 1.2))
    p.rect(x - 2, top - 1, x + 2, top, shade(colour, 0.7))


def _smoke(p: Painter, x: float, y: float, colour: RGB = (150, 148, 150), puffs: int = 3) -> None:
    for i in range(puffs):
        r = 1.5 + 0.6 * i
        cx, cy = x + 1.5 * i, y - 2.5 * i
        p.blob(cx - r, cy - r, cx + r, cy + r, colour if i % 2 else shade(colour, 0.86))


def _gear(p: Painter, x: float, y: float, radius: float, colour: RGB) -> None:
    for k in range(8):
        a = k * math.pi / 4
        tx, ty = x + (radius + 0.8) * math.cos(a), y + (radius + 0.8) * math.sin(a)
        p.blob(tx - 1, ty - 1, tx + 1, ty + 1, shade(colour, 0.8))
    p.ellipse(x - radius, y - radius, x + radius, y + radius, colour)
    p.blob(x - 1, y - 1, x + 1, y + 1, shade(colour, 0.45))


def _bale(p: Painter, x: float, y: float, colour: RGB) -> None:
    p.box(x, y, 2, 2, colour, shade(colour, 1.15))


def barrel_maker() -> Sprite:
    """A cooper's shed with finished barrels stood outside."""
    p = Painter(28, 22, (13, 16))
    p.shadow(11, 3)
    top = p.box(-3, 0, 7, 5, TIMBER, None)
    p.gable(top, 3, (118, 84, 54), TIMBER)
    for x, y in ((7, -1), (11, 1), (6, 3)):
        _barrel(p, x, y)
    return p.sprite()


def bakery() -> Sprite:
    """A baker's house with a brick oven chimney and a loaf hung over the door."""
    p = Painter(18, 22, (8, 17))
    p.shadow(7, 2.5)
    wall = (232, 214, 176)
    top = p.box(0, 0, 6, 5, wall, None)
    p.gable(top, 4, (150, 72, 52), wall)
    p.rect(2, -13, 4, -6, BRICK)
    for x, y, r in ((3, -15, 1.2), (4, -17, 1.6)):
        p.blob(x - r, y - r, x + r, y + r, (224, 222, 226))
    p.rect(-4, -3, -3, -1, shade(wall, 0.55))
    p.ellipse(-6, -6, -3, -4, (196, 128, 58))
    p.point(3, -2, WINDOW)
    return p.sprite()


def food_plant() -> Sprite:
    """A cannery: a pale hall with a green roof and two tall silos."""
    p = Painter(26, 32, (12, 25))
    p.shadow(10, 3)
    wall = (222, 214, 192)
    p.box(-2, 0, 8, 6, wall, (86, 128, 84))
    for x in (6, 10):
        p.rect(x - 2, -18, x + 1, 1, shade(METAL, 1.1))
        p.ellipse(x - 2, -20, x + 1, -16, shade(METAL, 1.25))
    p.point(-6, -2, WINDOW)
    p.point(-3, -1, WINDOW)
    return p.sprite()


def steel_mill() -> Sprite:
    """A dark mill with a blast furnace glowing at its foot and a plume of smoke."""
    p = Painter(32, 46, (14, 38))
    p.shadow(12, 4)
    iron = (82, 80, 84)
    p.box(0, 0, 11, 9, iron, (60, 58, 62))
    p.rect(4, -26, 9, -6, shade(iron, 0.85))
    p.polygon([(4, -26), (6.5, -30), (9, -26)], shade(iron, 0.7))
    p.rect(-9, -12, -6, -2, (238, 132, 40))
    p.point(-8, -10, (255, 214, 120))
    p.rect(-3, -30, -1, -9, shade(BRICK, 0.7))
    for i, (x, y, r) in enumerate([(-2, -33, 2), (0, -36, 3), (3, -39, 3.5)]):
        p.blob(x - r, y - r, x + r, y + r, (150, 148, 150) if i % 2 else (124, 122, 126))
    return p.sprite()


def textile_mill() -> Sprite:
    """A red-brick cotton mill of Łódź: a long block of storeys with rows of small arched windows,
    a sawtooth weaving shed at its end, a tall round chimney and bales of cloth by the door."""
    p = Painter(37, 55, (17, 44))
    p.shadow(15, 5, reach=4)
    wall, roof = (178, 78, 58), (80, 76, 84)
    _chimney(p, 11, 0, -32, shade(wall, 0.92))
    _smoke(p, 12, -35)
    left, front, right, _ = p.hall(-4, -1, 14, 8, 15, wall, roof)
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], shade(wall, 1.2))
    lintel = shade(wall, 1.3)
    _windows(p, left, front, (0.1, 0.3, 0.5, 0.7, 0.9), (3.5, 7, 10.5), 2, arch=lintel)
    _windows(p, left, front, (0.1, 0.3, 0.5), (14,), 1, arch=lintel)
    _windows(p, front, right, (0.3, 0.7), (3.5, 7), 2, shade(WINDOW, 0.75), lintel)
    _patch(p, left, front, (0.68, 0.84), (12, 15), (60, 44, 36))
    shed = p.box(7, 4.5, 8, 5, wall, None)
    _sawtooth(p, shed, 3, 4, (104, 100, 108), (160, 196, 218), wall)
    _windows(p, shed[0], shed[1], (0.2, 0.5, 0.8), (2.5,), 1, arch=lintel)
    for x, y, colour in ((-7, 6, CLOTH[0]), (-3.5, 7.5, CLOTH[1]), (-7, 4, CLOTH[2])):
        _bale(p, x, y, colour)
    return p.sprite()


def furniture_maker() -> Sprite:
    """A joiner's plank-walled workshop with sawn boards stacked in its yard and a chair standing
    on its roof for a sign."""
    p = Painter(30, 32, (14, 23))
    p.shadow(12, 3.5)
    wall, board = (176, 134, 88), (226, 190, 134)
    for x, y, h in ((9, -3, 6), (7, 2, 4)):
        stack = p.hall(x, y, 8, 3, h, board, shade(board, 1.12))
        gap = shade(board, 0.5)
        for drop in range(2, h, 2):
            p.line(
                [_on_wall(stack[0], stack[1], 0, drop), _on_wall(stack[0], stack[1], 1, drop)], gap
            )
            p.line(
                [_on_wall(stack[1], stack[2], 0, drop), _on_wall(stack[1], stack[2], 1, drop)], gap
            )
    top = p.box(-4, 0, 7, 6, wall, None)
    left, front, *_ = top
    for drop in (2, 4):
        p.line([_on_wall(left, front, 0, drop), _on_wall(left, front, 1, drop)], shade(wall, 0.86))
    _patch(p, left, front, (0.45, 0.82), (1.5, 6), (56, 40, 30))
    p.point(*_on_wall(left, front, 0.22, 2.5), WINDOW)
    p.gable(top, 4, (98, 74, 58), wall)
    chair = (156, 62, 44)
    x0, y0 = -6, -10
    p.polygon(
        [
            (x0 + dx, y0 + dy)
            for dx, dy in (
                (0, 0),
                (0, -7),
                (1, -7),
                (1, -3),
                (5, -3),
                (5, 0),
                (4, 0),
                (4, -2),
                (1, -2),
                (1, 0),
            )
        ],
        chair,
    )
    return p.sprite()


def tool_works() -> Sprite:
    """A dark forge hall with its door open on a glowing hearth, an anvil on a board over it and a
    short smoking stack."""
    p = Painter(30, 38, (14, 29))
    p.shadow(12, 4)
    wall = (92, 84, 82)
    top = p.hall(-1, 0, 13, 9, 7, wall, None)
    left, front, right, _ = top
    _patch(p, left, front, (0.52, 0.86), (2.5, 7), (238, 132, 40))
    _patch(p, left, front, (0.6, 0.78), (4.5, 7), (255, 208, 112))
    _patch(p, left, front, (0.03, 0.47), (0.5, 6), (214, 202, 176))
    cx, cy = _on_wall(left, front, 0.25, 3.5)
    p.polygon(
        [
            (cx + dx, cy + dy)
            for dx, dy in (
                (-3, -1),
                (2, -1),
                (2, 0),
                (1, 0),
                (1, 1),
                (2, 2),
                (-2, 2),
                (-1, 1),
                (-1, 0),
                (-2, 0),
            )
        ],
        (40, 40, 46),
    )
    _windows(p, front, right, (0.3, 0.7), (3,), 1, (240, 150, 60))
    p.gable(top, 4, (60, 60, 68), wall)
    p.rect(-1, -17, 1, -10, (74, 58, 52))
    p.point(0, -17, (255, 170, 60))
    _smoke(p, 1, -20, (96, 94, 98))
    return p.sprite()


def winery() -> Sprite:
    """A low cream winery under red tiles, with a cellar door dug into a grassy bank, barrels
    outside and vines trained on a trellis."""
    p = Painter(34, 30, (17, 19))
    p.shadow(13, 3.5)
    wall = (238, 224, 192)
    top = p.hall(0, -2, 12, 7, 5, wall, None)
    left, front, *_ = top
    p.gable(top, 3, (178, 86, 58), wall)
    _windows(p, left, front, (0.2, 0.45, 0.8), (2,), 1)
    for t in (0.2, 0.45, 0.8):
        p.point(*_on_wall(left, front, t - 0.07, 2), (90, 120, 70))
    _patch(p, left, front, (0.56, 0.68), (2, 5), (110, 74, 50))
    vine, grape, post = (78, 130, 50), (112, 50, 124), (110, 84, 58)
    for start in ((-16, 1), (-13, 5)):
        ends = [(start[0] + 3 * k, start[1] + 1.5 * k) for k in range(4)]
        for x, y in ends:
            p.line([(x, y), (x, y - 5)], post)
        p.line([(x, y - 4) for x, y in ends], vine)
        p.line([(x, y - 5) for x, y in ends], shade(vine, 1.25))
        for x, y in ends[:-1]:
            p.point(x + 1.5, y - 2.5, grape)
    cellar = p.box(10, 3, 4, 3, (176, 168, 150), (104, 140, 70))
    _patch(p, cellar[0], cellar[1], (0.3, 0.75), (1, 3), (42, 34, 30))
    p.point(*_on_wall(cellar[0], cellar[1], 0.5, 0.5), shade(STONE, 1.3))
    for x, y in ((3, 6), (6, 7.5)):
        _barrel(p, x, y)
    return p.sprite()


def chemical_works() -> Sprite:
    """A ball tank on legs, pale tanks with green tops and a pipe rack beside a small control
    house, with green fumes rising from a vent stack."""
    p = Painter(32, 47, (15, 37))
    p.shadow(13, 4)
    body, green = (208, 214, 198), (104, 176, 84)
    p.rect(10, -26, 11, -2, METAL)
    for i, (x, y, r) in enumerate(((11, -28, 1.5), (12, -31, 2), (14, -34, 2.5))):
        p.blob(x - r, y - r, x + r, y + r, (178, 222, 120) if i % 2 else (146, 204, 96))
    _turned(p, 4, -4, [(0, 3), (13, 3)], body, cap=green)
    for x in (-11, -9, -6):
        p.line([(x, 0), (x, -6)], shade(METAL, 0.7))
    p.ellipse(-13, -15, -4, -6, body)
    p.blob(-11, -13, -8, -10, shade(body, 1.15))
    p.line([(-13, -10), (-4, -10)], green)
    p.line([(-4, -9), (2, -9)], shade(METAL, 0.8))
    p.line([(-4, -7), (8, -7)], green)
    _turned(p, 8, 1, [(0, 2.5), (9, 2.5)], body, cap=green)
    left, front, *_ = p.box(-3, 4, 5, 4, (196, 200, 186), (96, 116, 84))
    _windows(p, left, front, (0.3, 0.7), (1.5,), 1, (110, 200, 120))
    return p.sprite()


def electronics_plant() -> Sprite:
    """A modern box of white panels and bands of cyan glass, with a dish and an aerial on its flat
    roof."""
    p = Painter(32, 38, (15, 28))
    p.shadow(13, 4)
    wall, glass, accent = (226, 232, 236), (84, 176, 206), (80, 216, 222)
    left, front, right, _ = p.hall(0, 0, 15, 9, 8, wall, (172, 180, 190))
    for d0, d1 in ((1.5, 3), (4.5, 6.5)):
        _patch(p, left, front, (0.04, 0.96), (d0, d1), glass)
        _patch(p, front, right, (0.06, 0.94), (d0, d1), shade(glass, 0.72))
    p.line([_on_wall(left, front, 0.04, 4.5), _on_wall(left, front, 0.96, 4.5)], accent)
    _patch(p, left, front, (0.42, 0.58), (6.5, 8), (52, 70, 84))
    p.line([(-4, -8), (-4, -11)], METAL)
    p.ellipse(-8, -16, -1, -11, (236, 238, 240))
    p.blob(-6, -15, -3, -13, (176, 186, 196))
    p.point(-4, -14, accent)
    p.line([(5, -7), (5, -18)], METAL)
    p.line([(3, -15), (7, -15)], METAL)
    p.point(5, -19, (230, 60, 60))
    p.rect(7, -11, 9, -9, (190, 198, 206))
    return p.sprite()


def machine_works() -> Sprite:
    """A big brick engine hall with tall arched windows and a great gear on its end wall, beside a
    yard with a gantry crane lifting a casting."""
    p = Painter(38, 42, (18, 32))
    p.shadow(16, 5)
    top = p.hall(-2, -2, 16, 9, 10, BRICK, None)
    left, front, right, _ = top
    lintel = shade(BRICK, 1.3)
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2.5,), 5, arch=lintel)
    _patch(p, left, front, (0.8, 0.94), (5, 10), (58, 48, 44))
    p.gable(top, 5, (82, 84, 96), BRICK)
    _gear(p, *_on_wall(front, right, 0.5, 5), 2.5, (222, 180, 74))
    crane = (214, 168, 40)
    feet, height = ((11, 6), (17, 3)), 15
    for x, y in feet:
        p.line([(x, y), (x, y - height)], crane)
        p.line([(x - 1, y), (x, y - 3)], shade(crane, 0.7))
    p.line([(feet[0][0] - 1, feet[0][1] - height), (feet[1][0] + 1, feet[1][1] - height)], crane)
    p.line([(14, -11), (14, -4)], (70, 64, 58))
    p.rect(12, -4, 16, -1, (88, 90, 98))
    return p.sprite()


def brickworks() -> Sprite:
    """A round Hoffmann kiln ringed with arched firing doors around its tall central chimney, and
    stacks of new red bricks."""
    p = Painter(34, 54, (16, 44))
    p.shadow(14, 5)
    kiln = (152, 98, 76)
    x, y, r = -2, -1, 10
    _turned(p, x, y, [(0, r), (6, r)], kiln)
    for a in (0.35, 0.95, 1.55, 2.15, 2.75):
        ax, ay = x + r * math.cos(a), y + r / 2 * math.sin(a)
        p.rect(ax - 1, ay - 4, ax + 1, ay, (52, 36, 32))
    _turned(p, x, y - 6, [(0, r + 1), (3, r - 4)], (96, 86, 86), cap=(118, 106, 104))
    _chimney(p, x, y - 8, y - 35, shade(BRICK, 0.9))
    _smoke(p, x + 1, y - 38, puffs=2)
    for bx, by in ((13, 2), (8.5, 5.5)):
        stack = p.hall(bx, by, 4, 3, 3, BRICK, shade(BRICK, 1.25))
        for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
            p.line([_on_wall(a, b, 0, 1.5), _on_wall(a, b, 1, 1.5)], shade(BRICK, 1.3))
    return p.sprite()


def cement_works() -> Sprite:
    """Dusty grey silos and a preheater tower, with a long rotary kiln tube on piers before them."""
    p = Painter(40, 50, (20, 40))
    p.shadow(16, 5)
    grey, dust = (190, 188, 182), (214, 212, 206)
    tower = p.box(9, -6, 3.5, 22, shade(grey, 0.9), shade(grey, 0.75))
    for drop in (4, 9, 14, 19):
        p.line(
            [_on_wall(tower[0], tower[1], 0, drop), _on_wall(tower[0], tower[1], 1, drop)],
            shade(grey, 0.6),
        )
    p.rect(8, -32, 10, -28, shade(grey, 0.7))
    _smoke(p, 9, -35, dust, 2)
    for sx, sy in ((-6, -7), (-10, -5), (-14, -3)):
        _turned(p, sx, sy, [(0, 2.5), (16, 2.5)], grey, cap=shade(grey, 1.1))
    tube = (134, 128, 120)
    a, b = (-12, 5), (6, -4)
    for t in (0.15, 0.5, 0.85):
        px, py = _on_wall(a, b, t, 0)
        p.rect(px - 1, py - 2 - 2 * t, px + 1, py + 1, shade(grey, 0.8))
    lift = 4
    p.polygon(
        [
            (a[0], a[1] - lift - 3),
            (b[0], b[1] - lift - 5),
            (b[0], b[1] - lift),
            (a[0], a[1] - lift),
        ],
        tube,
        shade(tube, 0.55),
    )
    p.line([(a[0], a[1] - lift - 3), (b[0], b[1] - lift - 5)], shade(tube, 1.3))
    for t in (0.3, 0.6):
        px, py = _on_wall(a, b, t, -lift)
        p.line([(px, py - 3 - t), (px, py)], shade(tube, 0.6))
    p.ellipse(a[0] - 2, a[1] - lift - 3.5, a[0] + 1, a[1] - lift + 0.5, shade(tube, 0.8))
    for x, y, r in ((-14, 4, 2), (-10, 7, 1.5), (-16, 1, 1.5)):
        p.blob(x - r, y - r, x + r, y + r, dust)
    return p.sprite()


def glassworks() -> Sprite:
    """A tall brick glass cone with its furnace mouth glowing at the foot, beside sheets of new
    glass leant against a shed."""
    p = Painter(32, 50, (15, 40))
    p.shadow(13, 4)
    cone = (162, 88, 64)
    x = -3
    _turned(p, x, 0, [(0, 9), (7, 7.5), (20, 4), (28, 3)], cone, cap=(54, 40, 36))
    _smoke(p, x + 1, -31)
    p.ellipse(x - 3, -6, x + 1, 0, (240, 132, 40))
    p.rect(x - 3, -3, x + 1, 2, (240, 132, 40))
    p.blob(x - 2, -3, x, 1, (255, 214, 120))
    for gx, gy in ((x - 4, 4), (x - 1, 4), (x - 3, 5)):
        p.point(gx, gy, (236, 150, 60))
    top = p.box(10, 2, 4, 4, (196, 180, 156), None)
    p.gable(top, 2, (96, 84, 80), (196, 180, 156))
    pane = (176, 222, 234)
    for k in range(3):
        px, py = 5 + 2 * k, 7 - k
        p.polygon(
            [(px, py), (px + 1, py - 5), (px + 3, py - 6), (px + 2, py - 1)], pane, shade(pane, 0.6)
        )
        p.point(px + 1, py - 3, (246, 252, 255))
    return p.sprite()


def brewery() -> Sprite:
    """A brick brewhouse with a copper-domed tower, barrels at its door and a hop cone for a sign."""
    p = Painter(32, 44, (15, 34))
    p.shadow(13, 4)
    top = p.hall(-4, -1, 12, 8, 8, BRICK, None)
    left, front, *_ = top
    _windows(p, left, front, (0.45, 0.65, 0.85), (2,), 2, arch=shade(BRICK, 1.3))
    _patch(p, left, front, (0.04, 0.34), (1, 5), (222, 210, 180))
    hop = (110, 170, 70)
    hx, hy = _on_wall(left, front, 0.19, 3)
    p.polygon(
        [(hx, hy - 1.5), (hx + 1.5, hy), (hx, hy + 1.5), (hx - 1.5, hy)], hop, shade(hop, 0.6)
    )
    p.gable(top, 4, (84, 76, 80), BRICK)
    tower = shade(BRICK, 1.08)
    p.box(5, 3, 4, 15, tower, None)
    p.line([(1, -12), (5, -10), (9, -12)], shade(tower, 1.25))
    _windows(p, (1, -12), (5, -10), (0.5,), (3, 8), 2, arch=shade(BRICK, 1.35))
    _turned(p, 5, -12, [(0, 4.5), (2, 4.2), (4, 3), (5.5, 1.5), (6, 0)], COPPER)
    p.blob(2, -16, 4, -14, shade(COPPER, 1.3))
    p.line([(5, -18), (5, -20)], (90, 80, 70))
    for x, y in ((-7, 6), (-3, 8)):
        _barrel(p, x, y)
    return p.sprite()


def distillery() -> Sprite:
    """A whitewashed still house with a copper pot still at its side and a tall thin stack."""
    p = Painter(32, 50, (15, 41))
    p.shadow(13, 4)
    p.rect(3, -33, 4, -4, (150, 80, 60))
    p.line([(3, -31), (4, -31)], (230, 228, 222))
    _smoke(p, 4, -36, (190, 190, 194), 2)
    top = p.hall(-4, -1, 12, 7, 6, WHITEWASH, None)
    left, front, *_ = top
    _windows(p, left, front, (0.15, 0.35, 0.8), (2,), 2, (44, 46, 52))
    _patch(p, left, front, (0.5, 0.66), (2.5, 6), (60, 48, 42))
    p.gable(top, 4, (68, 72, 82), WHITEWASH)
    _turned(p, 8, 5, [(0, 3), (2, 4), (4, 3.5), (6, 1.5), (7, 1)], COPPER)
    p.blob(5, -1, 7, 2, shade(COPPER, 1.3))
    p.line([(8, -2), (8, -4), (12, -2)], shade(COPPER, 0.8))
    _barrel(p, 12, 4)
    return p.sprite()


def tannery() -> Sprite:
    """Low tanning sheds, with hides stretched to dry on wooden frames and a pit of tan liquor."""
    p = Painter(34, 30, (17, 18))
    p.shadow(12, 3.5)
    wall = (184, 160, 124)
    for x, y in ((-2, -6), (-7, -2)):
        top = p.hall(x, y, 10, 5, 4, wall, None)
        _patch(p, top[0], top[1], (0.3, 0.7), (1, 4), (70, 56, 44))
        p.gable(top, 3, (96, 82, 74), wall)
    p.ellipse(-6, 4, 0, 7, (138, 116, 86))
    p.blob(-5, 4.5, -1, 6.5, (96, 66, 40))
    frame = (104, 76, 50)
    for k, hide in enumerate(((198, 150, 100), (162, 108, 68), (214, 180, 130))):
        x, y = 2 + 4 * k, 6 - 2 * k
        p.line([(x, y), (x, y - 7)], frame)
        p.line([(x + 4, y - 1), (x + 4, y - 8)], frame)
        p.line([(x, y - 7), (x + 4, y - 8)], frame)
        p.polygon(
            [(x + 1, y - 6), (x + 3, y - 7), (x + 3.5, y - 2), (x + 2, y - 1.5), (x + 0.5, y - 1)],
            hide,
            shade(hide, 0.65),
        )
    return p.sprite()


def shoe_factory() -> Sprite:
    """A narrow workshop of four storeys with a big boot hung out from its wall as a sign."""
    p = Painter(24, 38, (12, 29))
    p.shadow(8, 2.5)
    wall = (204, 170, 116)
    s, floors = 6, 4
    top = p.box(1, 0, s, 4 * floors, wall, None)
    left, front, right, _ = top
    _windows(p, left, front, (0.3, 0.7), tuple(1.5 + 4 * f for f in range(floors - 1)), 2)
    _windows(
        p,
        front,
        right,
        (0.3, 0.7),
        tuple(1.5 + 4 * f for f in range(floors)),
        2,
        shade(WINDOW, 0.75),
    )
    _patch(p, left, front, (0.55, 0.8), (12.5, 16), (84, 60, 46))
    p.gable(top, 4, (124, 66, 52), wall)
    p.line([(-5, -12), (-8, -12)], (60, 56, 54))
    boot = (140, 54, 40)
    p.polygon(
        [(-9, -11), (-5, -11), (-5, -3), (-11, -3), (-11, -5), (-10, -6), (-9, -6)],
        boot,
        shade(boot, 0.5),
    )
    p.line([(-8, -10), (-8, -6)], shade(boot, 1.4))
    p.line([(-11, -3), (-5, -3)], (40, 32, 30))
    return p.sprite()


def plastics_works() -> Sprite:
    """White tanks with red and blue tops and a white hall under a blue roof, tied together by
    coloured pipes, under a red-banded stack."""
    p = Painter(32, 40, (15, 31))
    p.shadow(13, 4)
    white, red, blue = (232, 234, 236), (214, 58, 54), (64, 108, 196)
    p.rect(-2, -29, 0, -4, white)
    for y in (-27, -24):
        p.line([(-2, y), (0, y)], red)
    _turned(p, 6, -3, [(0, 3), (12, 3)], white, cap=red)
    _turned(p, 11, 2, [(0, 2.5), (8, 2.5)], white, cap=blue)
    left, front, *_ = p.box(-5, 2, 7, 5, white, blue)
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], red)
    _windows(p, left, front, (0.25, 0.5, 0.75), (2.5,), 1, shade(blue, 0.8))
    p.line([(1, -5), (3, -6), (3, -10)], blue)
    p.line([(2, -2), (8, -2), (9, -4)], red)
    return p.sprite()


def appliance_factory() -> Sprite:
    """A long modern factory with a big roller door, and its fridges and washing machines stacked
    outside in white."""
    p = Painter(36, 35, (17, 23))
    p.shadow(15, 4)
    wall, stripe = (216, 220, 224), (64, 116, 196)
    left, front, right, _ = p.hall(1, -3, 18, 8, 7, wall, (150, 160, 174))
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], stripe)
    p.line([_on_wall(front, right, 0, 1), _on_wall(front, right, 1, 1)], shade(stripe, 0.8))
    door = (170, 174, 180)
    _patch(p, left, front, (0.55, 0.88), (2.5, 7), door)
    for drop in (3.5, 5, 6.5):
        p.line(
            [_on_wall(left, front, 0.55, drop), _on_wall(left, front, 0.88, drop)], shade(door, 0.7)
        )
    _windows(p, left, front, (0.1, 0.2, 0.3, 0.4), (3,), 1, (110, 160, 200))
    white = (244, 246, 248)
    for x, y, h, washer in (
        (15, 1, 6, False),
        (12, 4, 4, True),
        (12, 0, 4, True),
        (9, 7, 4, True),
    ):
        crate = p.box(x, y, 2.5, h, white, shade(white, 0.9))
        if washer:
            cx, cy = _on_wall(crate[0], crate[1], 0.5, 2.5)
            p.blob(cx - 1, cy - 1, cx + 1, cy + 1, (64, 88, 118))
        else:
            p.line(
                [_on_wall(crate[0], crate[1], 0, 2), _on_wall(crate[0], crate[1], 1, 2)],
                shade(white, 0.55),
            )
    return p.sprite()


def paper_mill() -> Sprite:
    """A riverside paper mill turning a waterwheel in its stream, with a tall stack and rolls of
    paper stacked in the yard."""
    p = Painter(38, 51, (19, 41))
    water = (84, 140, 196)
    p.polygon([(-17, -3), (-1, 5), (-4, 7.5), (-17, 1)], water)
    p.line([(-14, -0.5), (-10, 1.5)], (150, 196, 226))
    p.shadow(11, 4)
    _chimney(p, 8, -4, -30)
    _smoke(p, 9, -33)
    wall = (192, 168, 136)
    top = p.hall(-1, -3, 12, 8, 9, wall, None)
    left, front, *_ = top
    _windows(p, left, front, (0.55, 0.75, 0.92), (2, 5.5), 2)
    p.gable(top, 4, (100, 86, 80), wall)
    cx, cy = _on_wall(left, front, 0.25, 7)
    wheel = (98, 72, 50)
    p.ellipse(cx - 3, cy - 5, cx + 3, cy + 4, wheel)
    p.ellipse(cx - 2, cy - 4, cx + 2, cy + 3, shade(wheel, 1.3))
    p.line([(cx, cy - 4), (cx, cy + 3)], wheel)
    p.line([(cx - 2, cy - 0.5), (cx + 2, cy - 0.5)], wheel)
    p.polygon([(cx - 4, cy + 2), (cx + 4, cy + 5), (cx + 4, cy + 6), (cx - 4, cy + 3)], water)
    roll = (240, 240, 234)
    for x, y in ((6, 6), (9, 4.5), (12, 3), (7.5, 2.5), (10.5, 1), (9, -1)):
        p.ellipse(x - 1.5, y - 3, x + 1.5, y, roll)
        p.point(x, y - 1.5, shade(roll, 0.55))
    return p.sprite()


def printing_works() -> Sprite:
    """A town printing house with tall windows and a newspaper held up on its roof as a sign."""
    p = Painter(26, 36, (12, 27))
    p.shadow(9, 3)
    wall = (220, 204, 172)
    left, front, right, _ = p.box(0, 0, 8, 12, wall, (110, 106, 116))
    p.line([_on_wall(left, front, 0, 1), _on_wall(left, front, 1, 1)], shade(wall, 1.12))
    _windows(p, left, front, (0.2, 0.45, 0.7, 0.9), (2.5,), 4)
    _windows(p, left, front, (0.2, 0.45, 0.9), (8,), 3)
    _patch(p, left, front, (0.58, 0.8), (8, 12), (70, 56, 48))
    _windows(p, front, right, (0.25, 0.5, 0.75), (2.5, 8), 3, shade(WINDOW, 0.75))
    for x in (-3, 3):
        p.line([(x, -14), (x, -17)], (70, 66, 62))
    page = (242, 240, 230)
    p.rect(-7, -25, 7, -17, page)
    p.line([(-5, -23), (5, -23)], (40, 40, 48))
    p.rect(-5, -21, -2, -19, (150, 150, 160))
    for y in (-21, -19):
        p.line([(0, y), (5, y)], (150, 150, 160))
    return p.sprite()


def sugar_refinery() -> Sprite:
    """A tall brick refinery with a big square chimney, and white sacks and sugar loaves in blue
    paper at its foot."""
    p = Painter(34, 61, (15, 51))
    p.shadow(13, 4, reach=5)
    wall = (184, 110, 74)
    stack = p.box(8, -3, 2, 34, shade(wall, 0.9), (60, 52, 50))
    for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
        p.line([_on_wall(a, b, 0, 2), _on_wall(a, b, 1, 2)], shade(wall, 1.2))
    _smoke(p, 9, -40, puffs=4)
    left, front, right, _ = p.box(-3, 0, 8, 18, wall, (92, 86, 94))
    _windows(p, left, front, (0.2, 0.45, 0.7, 0.92), (2, 5.5, 9, 12.5), 2)
    _windows(p, front, right, (0.3, 0.7), (2, 5.5, 9, 12.5, 16), 2, shade(WINDOW, 0.75))
    _patch(p, left, front, (0.55, 0.75), (15, 18), (70, 50, 40))
    white, blue = (240, 240, 236), (70, 100, 180)
    for x, y in ((9, 4), (11, 2.5), (10, 1)):
        p.ellipse(x - 2, y - 3, x + 2, y, white)
    for x, y in ((3, 7), (6, 7.5)):
        p.polygon([(x - 1.5, y), (x, y - 6), (x + 1.5, y)], white, shade(white, 0.7))
        p.polygon([(x - 1.5, y), (x - 1, y - 2), (x + 1, y - 2), (x + 1.5, y)], blue)
    return p.sprite()


def porcelain_works() -> Sprite:
    """Two small brick bottle ovens beside a workshop, with blue-and-white ware set out on shelves."""
    p = Painter(34, 36, (17, 27))
    p.shadow(13, 4)
    oven = (172, 116, 86)
    bottle = [(0, 4.5), (4, 5), (8, 4), (11, 2), (13, 1.8)]
    for x, y in ((-4, -6), (-10, -3)):
        _turned(p, x, y, bottle, oven, cap=(60, 46, 42))
    wall = (230, 218, 192)
    top = p.hall(5, -2, 10, 6, 5, wall, None)
    _windows(p, top[0], top[1], (0.25, 0.5, 0.75), (2,), 1)
    p.gable(top, 3, (122, 88, 70), wall)
    frame, china, blue = (120, 88, 58), (246, 246, 250), (54, 86, 176)
    for x in (-8, 2):
        p.line([(x, 6 + (x + 8) * 0.1), (x, -1)], frame)
    for y in (5, 2):
        p.line([(-8, y), (2, y + 1)], frame)
        for k in range(5):
            x = -7 + 2 * k
            p.line([(x, y - 1 + k * 0.2), (x, y - 2 + k * 0.2)], china if k % 2 else blue)
    return p.sprite()


def meat_factory() -> Sprite:
    """A brick slaughterhouse with a ham hung out as its sign, a ramp up from a cattle pen, and a
    white cold store with a refrigeration unit and vent on its roof."""
    p = Painter(36, 36, (17, 25))
    p.shadow(15, 4.5)
    top = p.hall(0, -3, 14, 8, 8, BRICK, None)
    left, front, *_ = top
    _windows(p, left, front, (0.6, 0.75, 0.9), (2,), 2, arch=shade(BRICK, 1.3))
    _patch(p, left, front, (0.3, 0.45), (3, 6), (56, 44, 40))
    p.gable(top, 4, (86, 80, 88), BRICK)
    bracket = _on_wall(left, front, 0.08, 2)
    p.line([bracket, (bracket[0] - 3, bracket[1] + 1)], (60, 56, 54))
    ham = (196, 76, 66)
    hx, hy = bracket[0] - 3, bracket[1] + 4
    p.ellipse(hx - 3, hy - 2, hx + 1, hy + 2, ham)
    p.point(hx - 2, hy - 1, shade(ham, 1.3))
    p.line([(hx + 1, hy - 1), (hx + 2, hy - 2)], (244, 238, 224))
    p.point(hx + 3, hy - 3, (244, 238, 224))
    ramp = [_on_wall(left, front, t, 6) for t in (0.3, 0.45)]
    p.polygon(
        [ramp[0], ramp[1], (ramp[1][0] - 4, ramp[1][1] + 4), (ramp[0][0] - 4, ramp[0][1] + 4)],
        TIMBER,
        shade(TIMBER, 0.6),
    )
    pen_left, pen_back, pen_right, pen_front = _paddock(-10, 3, 5)
    _fence(p, pen_left, pen_back)
    _fence(p, pen_back, pen_right)
    _cow(p, -8, 4, (138, 84, 50), (96, 58, 36))
    _fence(p, pen_left, pen_front)
    _fence(p, pen_front, pen_right)
    cold = (234, 236, 238)
    store = p.box(11, 3, 5, 6, cold, shade(cold, 0.88))
    for t in (0.25, 0.5, 0.75):
        p.line(
            [_on_wall(store[0], store[1], t, 0.5), _on_wall(store[0], store[1], t, 6)],
            shade(cold, 0.85),
        )
    _patch(p, store[1], store[2], (0.3, 0.7), (2, 6), (150, 170, 186))
    p.box(12, -3.5, 2, 2, METAL, shade(METAL, 1.1))
    p.blob(11, -7, 13, -5, (60, 62, 70))
    p.line([(8, -4), (8, -9)], METAL)
    p.rect(7, -10, 9, -9, shade(METAL, 0.8))
    return p.sprite()


def _car(p: Painter, x: float, y: float, colour: RGB) -> None:
    p.box(x, y, 2, 1.5, colour, shade(colour, 1.2))
    p.point(x, y - 2, (172, 214, 234))


def _plane(p: Painter, x: float, y: float, size: float = 1) -> None:
    """An airliner standing at (x, y), its nose pointing the way the lit walls run; `size` sets its
    length and wingspan but not how thick it is."""
    body, stripe = (240, 242, 246), (70, 130, 210)
    edge = shade(body, 0.55)
    half = 8 * size

    def at(u: float, v: float, h: float) -> Point:
        px, py = _iso(x, y, u, v)
        return px, py - h

    p.polygon([at(-1.5, -half, 2), at(2, -half, 2), at(2.5, half, 2), at(-1, half, 2)], body, edge)
    tail = -half
    p.polygon(
        [
            at(tail - 0.5, -3, 3.5),
            at(tail + 1.5, -3, 3.5),
            at(tail + 1.5, 3, 3.5),
            at(tail - 0.5, 3, 3.5),
        ],
        body,
        edge,
    )
    p.polygon(
        [
            at(tail, -1.5, 1.5),
            at(half, -1.5, 1.5),
            at(half + 1.5, 0, 2.5),
            at(half, -1.5, 3.5),
            at(tail, -1.5, 3.5),
        ],
        shade(body, 0.9),
        edge,
    )
    p.polygon(
        [
            at(tail, -1.5, 3.5),
            at(half, -1.5, 3.5),
            at(half + 1.5, 0, 2.5),
            at(half, 1.5, 3.5),
            at(tail, 1.5, 3.5),
        ],
        body,
        edge,
    )
    p.line([at(tail + 1, -1.5, 2.5), at(half - 1, -1.5, 2.5)], stripe)
    p.polygon([at(tail, 0, 3.5), at(tail + 3, 0, 3.5), at(tail, 0, 9)], stripe, shade(stripe, 0.6))


def _tank_vehicle(p: Painter, x: float, y: float, olive: RGB) -> None:
    track = (50, 50, 54)
    base = p.hall(x, y, 9, 5, 2, track, None)
    for t in (0.15, 0.38, 0.62, 0.85):
        p.point(*_on_wall(base[0], base[1], t, 1), (130, 130, 136))
    hull = (150, 158, 108)
    p.hall(x, y - 2, 8, 4.5, 2, hull, shade(hull, 1.12))
    tx, ty = x - 1, y - 4
    p.box(tx, ty, 2.5, 2, shade(hull, 0.92), shade(hull, 1.18))
    p.line([(tx + 1, ty - 2), (tx + 8, ty + 1.5)], shade(olive, 0.5))
    p.line([(tx + 1, ty - 3), (tx + 8, ty + 0.5)], shade(hull, 0.85))


def _pylon(p: Painter, x: float, y: float, away: float) -> None:
    """A lattice pylon standing on (x, y), its cables running `away` pixels off sideways."""
    steel, cable = (110, 112, 120), (52, 52, 58)
    p.line([(x - 3, y), (x, y - 19)], steel)
    p.line([(x + 3, y), (x, y - 19)], steel)
    for dy, half in ((-15, 4), (-11, 3)):
        p.line([(x - half, y + dy), (x + half, y + dy)], steel)
    p.line([(x - 2, y - 4), (x + 2, y - 8)], steel)
    p.line([(x + 2, y - 4), (x - 2, y - 8)], steel)
    side = 1 if away > 0 else -1
    for dy, half in ((-15, 4), (-11, 3)):
        tip = (x + side * half, y + dy)
        p.line([tip, (tip[0] + away / 2, tip[1] + 1.5), (tip[0] + away, tip[1] - 1)], cable)


def _cooling_tower(p: Painter, x: float, y: float) -> None:
    concrete = (198, 196, 188)
    _turned(
        p, x, y, [(0, 7.5), (6, 5.8), (11, 4.6), (15, 4.9), (17, 5.4)], concrete, cap=(70, 70, 74)
    )
    for i, (dx, dy, r) in enumerate(((0, -19, 3.5), (3, -23, 4), (7, -26, 3.5))):
        p.blob(
            x + dx - r,
            y + dy - r,
            x + dx + r,
            y + dy + r,
            (240, 240, 242) if i % 2 else (224, 226, 230),
        )


def car_factory() -> Sprite:
    """A car works filling its cell: a long assembly hall under an arched roof beside a
    sawtoothed press shop, with new cars parked in rows on the lot out front."""
    p = Painter(50, 40, (24, 26))
    _plate(p, (166, 166, 162))
    wall = (156, 160, 170)
    press_wall = shade(wall, 0.92)
    press = p.hall(*_iso(0, 0, 3, 8), 14, 5, 6, press_wall, None)
    _sawtooth(p, press, 4, 3, (104, 110, 126), (160, 196, 218), press_wall)
    main = p.hall(*_iso(0, 0, 0, 0.5), 20, 7, 7, wall, None)
    left, front, right, _ = main
    _patch(p, left, front, (0.03, 0.97), (1.5, 3), (96, 136, 168))
    _patch(p, front, right, (0.2, 0.8), (1.5, 7), (64, 66, 74))
    _vault(p, main, 5, (98, 108, 130), wall)
    for row, v in enumerate((-5.5, -8.5)):
        for k in range(6):
            colour = CAR_COLOURS[(k + 2 * row) % len(CAR_COLOURS)]
            _car(p, *_iso(0, 0, -9 + 3.6 * k + row, v), colour)
    return p.sprite()


def plane_factory() -> Sprite:
    """An aircraft works filling its cell: a great hangar under an arched roof with its doors open,
    a long assembly shed behind, and a new airliner standing on the dark apron in front."""
    p = Painter(50, 40, (24, 26))
    _plate(p, (176, 176, 170))
    apron = (112, 114, 118)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-10.5, -10.5), (10.5, -10.5), (10.5, -1), (-10.5, -1))],
        apron,
    )
    p.line([_iso(0, 0, -10, -2.5), _iso(0, 0, 10, -2.5)], (226, 196, 70))
    wall = (200, 206, 214)
    shed = p.hall(*_iso(0, 0, 4, 9), 12, 3.5, 6, shade(wall, 0.9), (140, 150, 166))
    _windows(p, shed[0], shed[1], tuple(k / 7 for k in range(1, 7)), (2,), 1, (96, 136, 168))
    hangar = p.hall(*_iso(0, 0, -4, 3.5), 12, 9, 7, wall, None)
    left, front, right, _ = hangar
    _windows(p, left, front, tuple(k / 8 for k in range(1, 8)), (2,), 2, (96, 136, 168))
    _patch(p, front, right, (0.08, 0.92), (0.5, 7), (44, 48, 58))
    tail = [_on_wall(front, right, t, d) for t, d in ((0.55, 7), (0.62, 2), (0.68, 7))]
    p.polygon(tail, (70, 130, 210))
    _vault(p, hangar, 7, (150, 160, 176), wall)
    _plane(p, *_iso(0, 0, 4.5, -3.5), 0.75)
    return p.sprite()


def tank_factory() -> Sprite:
    """A tank works filling its cell: low armoured assembly halls in olive drab with slit windows
    and riveted bands, and a new tank standing on its churned-up proving ground."""
    p = Painter(50, 36, (24, 22))
    _plate(p, (150, 142, 112))
    olive = (104, 112, 80)
    roof = shade(olive, 0.72)
    mud = (112, 92, 66)
    p.polygon([_iso(0, 0, 0, -2), _iso(0, 0, 10, -2), _iso(0, 0, 10, -10), _iso(0, 0, 0, -10)], mud)
    for v in (-5, -7.5):
        p.line([_iso(0, 0, 1, v), _iso(0, 0, 9, v)], shade(mud, 0.7))
    for (u, v), length, depth, height, tone in (
        ((0, 7), 20, 6, 7, 1.0),
        ((-5.5, -2), 9, 8, 6, 0.9),
    ):
        top = p.hall(*_iso(0, 0, u, v), length, depth, height, shade(olive, tone), roof)
        for a, b in ((top[0], top[1]), (top[1], top[2])):
            p.line([_on_wall(a, b, 0, 1), _on_wall(a, b, 1, 1)], shade(olive, 1.3))
            p.line(
                [_on_wall(a, b, 0, height - 1), _on_wall(a, b, 1, height - 1)], shade(olive, 0.7)
            )
            for k in range(1, 6):
                t = k / 6
                p.line([_on_wall(a, b, t - 0.04, 2.5), _on_wall(a, b, t + 0.04, 2.5)], (34, 36, 30))
        _patch(p, top[0], top[1], (0.6, 0.85), (3.5, height - 1), shade(olive, 0.55))
    _tank_vehicle(p, *_iso(0, 0, 4.5, -6), olive)
    return p.sprite()


def mega_factory() -> Sprite:
    """A great works filling its cell: three long brick halls under sawtooth roofs, a row of tall
    chimneys behind them and a round gasholder."""
    p = Painter(50, 63, (24, 51))
    _plate(p, (150, 146, 140))
    for u in (-9, -4, 1, 6):
        x, y = _iso(0, 0, u, 10)
        _chimney(p, x, y - 4, y - 30, shade(BRICK, 0.8))
        _smoke(p, x + 1, y - 33, (214, 214, 218))
    for v, height, tone in ((6, 8, 0.9), (0, 9, 1.0), (-6, 7, 1.08)):
        wall = shade(BRICK, tone)
        top = p.hall(*_iso(0, 0, -1, v), 18, 5, height, wall, None)
        _windows(
            p, top[0], top[1], tuple(k / 8 for k in range(1, 8)), (2.5,), 2, arch=shade(BRICK, 1.3)
        )
        _sawtooth(p, top, 5, 3, METAL, (170, 200, 220), wall)
    gx, gy = _iso(0, 0, 9.5, 2)
    _turned(p, gx, gy, [(0, 4.5), (8, 4.5)], (118, 126, 118), cap=(138, 146, 138))
    for a in np.linspace(0.3, math.pi - 0.3, 5):
        fx, fy = gx + 5 * math.cos(a), gy + 2.5 * math.sin(a)
        p.line([(fx, fy), (fx, fy - 10)], (86, 92, 86))
    return p.sprite()


def refinery() -> Sprite:
    """An oil refinery filling its cell: tall silver distillation columns ringed with walkways,
    squat white storage tanks, pipe racks and a flare burning on a tall stack."""
    p = Painter(50, 58, (24, 46))
    _plate(p, (168, 164, 156))
    metal, white = (176, 178, 184), (224, 226, 228)
    fx, fy = _iso(0, 0, 9, 9)
    p.line([(fx, fy), (fx, fy - 36)], (90, 90, 96))
    p.line([(fx + 1, fy), (fx + 1, fy - 36)], (124, 124, 130))
    p.blob(fx - 2, fy - 43, fx + 3, fy - 36, (246, 150, 40))
    p.blob(fx - 0.5, fy - 40, fx + 1.5, fy - 37, (255, 220, 120))
    for u, v in ((-9, 8), (-3, 9)):
        _turned(p, *_iso(0, 0, u, v), [(0, 4.5), (4, 4.5)], white, cap=shade(white, 1.08))
    for u, v, r, h in ((-7, 2, 2, 18), (-2, 2, 2.5, 30), (3, 3, 3, 24)):
        x, y = _iso(0, 0, u, v)
        _turned(p, x, y, [(0, r), (h, r), (h + 1.5, r * 0.6)], metal)
        for ring in range(6, h, 6):
            p.line([(x - r - 1, y - ring), (x + r + 1, y - ring)], shade(metal, 0.6))
    rack = (96, 96, 104)
    for u in (-9, -3, 3, 9):
        x, y = _iso(0, 0, u, -2)
        p.line([(x, y), (x, y - 6)], rack)
    for h, colour in ((6, rack), (4.5, (150, 120, 60))):
        a, b = _iso(0, 0, -9, -2), _iso(0, 0, 9, -2)
        p.line([(a[0], a[1] - h), (b[0], b[1] - h)], colour)
    for u, v in ((-8, -6), (-1, -7.5), (6, -7)):
        _turned(p, *_iso(0, 0, u, v), [(0, 3.5), (3.5, 3.5)], white, cap=shade(white, 1.08))
    return p.sprite()


def power_plant() -> Sprite:
    """A coal power station filling its cell: a big brick turbine hall with tall windows, two
    concrete cooling towers letting off steam, a tall chimney, a coal heap fed by a conveyor and a
    pylon carrying power lines away."""
    p = Painter(50, 70, (24, 58))
    _plate(p, (160, 156, 148))
    cx, cy = _iso(0, 0, -9, 9)
    _chimney(p, cx, cy - 4, cy - 38, shade(BRICK, 0.9))
    _smoke(p, cx + 1, cy - 41, (124, 122, 126))
    _cooling_tower(p, *_iso(0, 0, -2, 9.5))
    _pylon(p, *_iso(0, 0, -8, -8), -4)
    top = p.hall(*_iso(0, 0, -3, -0.5), 14, 7, 12, BRICK, None)
    left, front, right, _ = top
    window = (88, 112, 140)
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2,), 7, window, shade(BRICK, 1.3))
    _patch(p, left, front, (0.8, 0.94), (7, 12), (58, 48, 44))
    _windows(p, front, right, (0.3, 0.7), (2,), 7, shade(window, 0.8), shade(BRICK, 1.2))
    p.gable(top, 4, (78, 78, 88), BRICK)
    _cooling_tower(p, *_iso(0, 0, 8.5, 4))
    hx, hy = _iso(0, 0, 5, -7)
    p.line([(hx - 1, hy - 5), _on_wall(left, front, 0.9, 4)], (70, 70, 78))
    p.line([(hx - 1, hy - 4), _on_wall(left, front, 0.9, 5)], (110, 110, 118))
    _heap(p, hx, hy, 5, 5, (48, 46, 50))
    for dx, dy in ((-2, -2), (0, -3), (2, -1), (-1, 0)):
        p.point(hx + dx, hy + dy, (104, 104, 116))
    return p.sprite()


def shipyard() -> Sprite:
    """A shipyard filling its cell: a big ship's hull on a slipway running down into the water,
    a gantry crane straddling it and a fabrication hall behind."""
    p = Painter(50, 46, (24, 32))
    _plate(p, STONE)
    water = (72, 128, 184)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((7.5, -10.5), (10.5, -10.5), (10.5, 10.5), (7.5, 10.5))],
        water,
    )
    for v in (-7, 5):
        p.line([_iso(0, 0, 8.5, v), _iso(0, 0, 9.5, v + 2)], (150, 196, 226))
    hall_wall = (176, 166, 152)
    hall = p.hall(*_iso(0, 0, -3, 7), 14, 6, 9, hall_wall, None)
    _windows(p, hall[0], hall[1], (0.15, 0.35, 0.55, 0.75), (2,), 3)
    _patch(p, hall[1], hall[2], (0.2, 0.8), (3, 9), shade(hall_wall, 0.5))
    p.gable(hall, 4, (92, 96, 110), hall_wall)
    slip = shade(STONE, 0.78)
    p.polygon([_iso(0, 0, u, v) for u, v in ((-10, -6.5), (9, -6.5), (9, 0.5), (-10, 0.5))], slip)
    crane = (214, 168, 40)
    for u in (-2,):
        far = _iso(0, 0, u, 1.5)
        p.line([far, (far[0], far[1] - 20)], crane)
    hull, band, deck = (150, 58, 46), (40, 40, 46), (196, 190, 178)
    ship = p.hall(*_iso(0, 0, -2, -3), 14, 5, 6, hull, deck)
    for a, b in ((ship[0], ship[1]), (ship[1], ship[2])):
        _patch(p, a, b, (0, 1), (0, 2), band)
    front, right = ship[1], ship[2]
    tip = _iso(*_on_wall(front, right, 0.5, 0), 4, 0)
    p.polygon([front, tip, (tip[0], tip[1] + 5), (front[0], front[1] + 6)], hull, shade(hull, 0.55))
    p.polygon([front, tip, (tip[0], tip[1] + 2), (front[0], front[1] + 2)], band)
    p.polygon([front, right, tip], deck, shade(deck, 0.6))
    bx, by = _iso(0, 0, -7, -3)
    bridge = p.box(bx, by - 6, 2.5, 4, (238, 236, 230), shade(STONE, 1.15))
    _windows(p, bridge[0], bridge[1], (0.3, 0.7), (1,), 1)
    p.rect(bx - 1, by - 14, bx + 1, by - 10, (196, 60, 48))
    near = _iso(0, 0, -2, -7.5)
    far = _iso(0, 0, -2, 1.5)
    p.line([near, (near[0], near[1] - 20)], crane)
    p.line([(near[0], near[1] - 20), (far[0], far[1] - 20)], crane)
    p.line([(near[0] + 1, near[1] - 19), (far[0] + 1, far[1] - 19)], shade(crane, 0.7))
    hook = _iso(0, 0, -2, -3)
    p.line([(hook[0], hook[1] - 19), (hook[0], hook[1] - 12)], (70, 64, 58))
    p.rect(hook[0] - 1, hook[1] - 12, hook[0] + 1, hook[1] - 10, (88, 90, 98))
    return p.sprite()
