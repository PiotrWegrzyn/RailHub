"""Shops, taverns, department stores, hotels, builder's yards and airports."""

from railroads.ui.pixel.art.homes import _doorstep, _foot, _outward, _worn
from railroads.ui.pixel.art.painter import (
    BARREL,
    BRICK,
    CLOTH,
    DOORWAY,
    GLASS,
    RGB,
    STONE,
    TIMBER,
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
    SILL,
    _band,
    _bar,
    _fence,
    _flag,
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
LEAF: RGB = (60, 112, 48)
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


def _bush(p: Painter, x: float, y: float, size: float = 1.6) -> None:
    """A small round bush standing on the ground at (x, y)."""
    p.ellipse(x - size, y - 1.6 * size, x + size, y, LEAF)
    p.point(x - size * 0.5, y - 1.2 * size, shade(LEAF, 1.35))


def shop(wall: RGB, awning: RGB, goods: RGB) -> Sprite:
    """A smart city shop: a striped awning over a big window full of its wares and a door, a floor
    of windows with sills above, a cornice and a chimney."""
    p = Painter(24, 26, (11, 20))
    p.shadow(9, 3)
    height = 12
    _worn(p, (-8, -height), (0, 4 - height), (-0.05, 1.05), height, 2.5, PAVING)
    top = p.box(0, 0, 8, height, wall, shade(wall, 0.7))
    left, front, right, _ = top
    _pale_plinth(p, top, height, wall)
    _foot(p, top, height)
    for a, b, tone in ((left, front, 1.15), (front, right, 0.88)):
        _band(p, a, b, 0.6, shade(wall, tone))
    glazing = WINDOW if sum(wall) > 300 else LIT
    for span in ((0.14, 0.34), (0.56, 0.76)):
        _patch(p, left, front, span, (1.6, 3.6), glazing)
        p.line([_on_wall(left, front, span[0], 4), _on_wall(left, front, span[1], 4)], SILL)
    for span in ((0.2, 0.4), (0.6, 0.8)):
        for drop in (1.6, 6.6):
            _patch(p, front, right, span, (drop, drop + 2), shade(WINDOW, 0.75))
    _patch(p, left, front, (0, 1), (4.6, 6.6), awning)
    for t in (0.2, 0.4, 0.6, 0.8):
        p.line([_on_wall(left, front, t, 4.8), _on_wall(left, front, t, 6.4)], SILL)
    _band(p, left, front, 6.8, shade(awning, 0.6))
    _patch(p, left, front, (0.06, 0.62), (7.6, 11), glazing)
    for t in (0.16, 0.42):
        p.line([_on_wall(left, front, t, 9.8), _on_wall(left, front, t + 0.1, 9.8)], goods)
        p.point(*_on_wall(left, front, t + 0.05, 8.8), goods)
    _patch(p, left, front, (0.72, 0.9), (7.6, height), DOOR)
    p.point(*_on_wall(left, front, 0.86, 10), shade(SILL, 0.8))
    _doorstep(p, left, front, (0.68, 0.94), height, PAVING)
    p.rect(3, -15, 5, -12.5, shade(wall, 0.6))
    p.line([(3, -15.5), (5, -15.5)], shade(wall, 0.4))
    _bush(p, *_out(left, front, 1, height, 1.2), 1.4)
    _street_lamp(p, *_out(left, front, 0, height, 1.5), 8)
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


def department_store() -> Sprite:
    """A wide, ornate store of three storeys: great display windows dressed with goods, a glazed
    entrance under a gold sign, arched windows above, a corner dome and a flag."""
    p = Painter(28, 42, (13, 32))
    p.shadow(11, 3.5)
    wall = (234, 216, 184)
    _worn(p, (-10, -13), (0, -8), (-0.05, 1.05), 13, 2.5, PAVING)
    top = p.box(0, 0, 10, 13, wall, (128, 118, 124))
    left, front, right, _ = top
    display = (150, 196, 222)
    _patch(p, left, front, (0.05, 0.92), (9, 12), display)
    _patch(p, front, right, (0.08, 0.95), (9, 12), shade(display, 0.75))
    for k, t in enumerate((0.12, 0.22, 0.62, 0.74, 0.84)):
        p.line([_on_wall(left, front, t, 11), _on_wall(left, front, t + 0.04, 11)], CLOTH[k % 3])
    for t in (0.3, 0.68):
        p.line([_on_wall(left, front, t, 9), _on_wall(left, front, t, 12)], shade(wall, 0.85))
    for t in (0.36, 0.64):
        p.line([_on_wall(front, right, t, 9), _on_wall(front, right, t, 12)], shade(wall, 0.65))
    _patch(p, left, front, (0.4, 0.56), (9.4, 12.2), DOORWAY)
    _pale_plinth(p, top, 13, wall, 0.8)
    _foot(p, top, 13)
    _doorstep(p, left, front, (0.36, 0.6), 13, shade(PAVING, 1.1))
    for a, b in ((left, front), (front, right)):
        _band(p, a, b, 1, shade(wall, 1.12))
        _band(p, a, b, 8, shade(wall, 0.7))
    sign = (196, 156, 60)
    _patch(p, left, front, (0.32, 0.64), (7.6, 8.6), sign)
    _windows(p, left, front, (0.12, 0.3, 0.48, 0.66), (2.5, 5.5), 2, arch=shade(wall, 0.72))
    _windows(
        p,
        front,
        right,
        (0.34, 0.52, 0.7, 0.88),
        (2.5, 5.5),
        2,
        shade(WINDOW, 0.75),
        shade(wall, 0.6),
    )
    p.line([left, front, right], shade(wall, 1.15))
    for x, y in (left, right):
        p.rect(x - 0.5, y - 2, x + 0.5, y, shade(wall, 0.9))
    _turned(p, 0, -8, [(0, 3), (3, 3)], wall)
    for dx in (-2, 0):
        p.line([(dx, -9), (dx, -10)], WINDOW)
    _turned(p, 0, -11, [(0, 3.4), (2, 2.8), (3.5, 1.6), (4.5, 0)], (98, 150, 132))
    _flag(p, 0, -15, 8)
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


GRAVEL: RGB = (176, 166, 142)
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


def builders_yard() -> Sprite:
    """A builder's yard of trodden gravel behind a post-and-rail fence: a boarded timber store
    14 by 7 m with planks and sacks in its open front, a small office with a yellow sign,
    pallets of bricks and of cement sacks, stacks of sawn planks, a rack of glass panes, a heap
    of sand and a hand cart, all at their true size beside the houses."""
    p = Painter(30, 22, (15, 14))
    m = METRE
    yard = _paddock(0, 0, 12.5)
    p.polygon(list(yard), GRAVEL, shade(GRAVEL, 0.8))
    left_corner, back_corner, right_corner, front_corner = yard
    _fence(p, left_corner, back_corner, m)
    _fence(p, back_corner, right_corner, m)
    _sand(p, 2, -3.2, 2.2, 1.7)
    office = (226, 216, 190)
    top = p.hall(6.2, -1.6, 5 * m, 4.5 * m, 2.7 * m, office, None)
    left, front, right, _ = top
    _foot(p, top, 2.7 * m)
    _patch(p, left, front, (0.58, 0.8), (0.6 * m, 2.7 * m), DOOR)
    _patch(p, left, front, (0.15, 0.42), (0.6 * m, 1.5 * m), WINDOW)
    _patch(p, front, right, (0.3, 0.7), (0.6 * m, 1.5 * m), shade(WINDOW, 0.8))
    p.gable(top, 1.4 * m, (150, 60, 46), office)
    _patch(p, left, front, (0.1, 0.9), (-0.9 * m, 0.1 * m), (236, 194, 60))
    store = (156, 118, 82)
    high = 3.4 * m
    top = p.hall(-3.2, -1, 14 * m, 7 * m, high, store, None)
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
    _hand_cart(p, 2.4, 0.9)
    _planks(p, 7.6, 1.5, 6, 1.2)
    _planks(p, 4.6, 3.3, 5, 0.8)
    for row, col, tall in ((0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 1, 1), (1, 2, 1)):
        bx = -7.6 + (col + row) * 1.15 * m
        by = 0.9 + (col - row) * 0.58 * m
        _pallet(p, bx, by, tall, BRICK, 2 * tall)
    for k, tall in enumerate((1.2, 0.9, 1.2)):
        _pallet(p, -4 + k * 1.15 * m, 3.3 + k * 0.58 * m, tall, SACK, round(tall / 0.3))
    _glass_rack(p, 0.8, 3)
    gate = 0.42
    _fence(p, left_corner, _on_wall(left_corner, front_corner, gate, 0), m)
    _fence(p, _on_wall(left_corner, front_corner, gate + 0.18, 0), front_corner, m)
    _fence(p, front_corner, right_corner, m)
    return p.sprite()
