"""The one-off buildings a game may get (railroads.sim.novelties)."""

import math

import numpy as np

from railroads.ui.pixel.art.homes import SOIL, _doorstep, _foot, _worn
from railroads.ui.pixel.art.painter import (
    CAR_COLOURS,
    CHURCH_ROOF,
    CROSS,
    DIGITS,
    DOORWAY,
    LEAVES,
    METAL,
    RGB,
    STONE,
    TRUNK,
    VERDIGRIS,
    WHITEWASH,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import (
    _cross,
    _flag,
    _gable_across,
    _heap,
    _iso,
    _on_wall,
    _patch,
    _plate,
    _portal,
    _turned,
    _vault,
    _windows,
)

type Top = tuple[Point, Point, Point, Point]

LAMP: RGB = (250, 236, 170)

PAVING: RGB = (196, 190, 176)

GRAVEL: RGB = (214, 206, 176)


def _plinth(p: Painter, top: Top, height: float, wall: RGB, depth: float = 0.8) -> None:
    """A darker band along the foot of a box's two front walls."""
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (height - depth, height), shade(wall, 0.7))
    _patch(p, front, right, (0, 1), (height - depth, height), shade(wall, 0.52))


def _tiles(p: Painter, top: Top, rise: float, roof: RGB, rows: tuple[float, ...] = (0.5,)) -> None:
    """Rows of tiles across the lit side of a gable roof (Painter.gable)."""
    left, front, right, back = top
    ridge_back = ((left[0] + back[0]) / 2, (left[1] + back[1]) / 2 - rise)
    ridge_front = ((front[0] + right[0]) / 2, (front[1] + right[1]) / 2 - rise)
    for f in rows:
        a = (left[0] + (ridge_back[0] - left[0]) * f, left[1] + (ridge_back[1] - left[1]) * f)
        b = (front[0] + (ridge_front[0] - front[0]) * f, front[1] + (ridge_front[1] - front[1]) * f)
        p.line([a, b], shade(roof, 0.8))


def _lamp(p: Painter, x: float, y: float, height: float = 5) -> None:
    p.line([(x, y), (x, y - height)], (60, 60, 66))
    p.rect(x - 0.5, y - height - 1.5, x + 0.5, y - height, LAMP)


def _people(p: Painter, spots: list[Point]) -> None:
    """Visitors, each a dot of colour over a darker one, at each of `spots`."""
    for k, (x, y) in enumerate(spots):
        colour = CAR_COLOURS[k % len(CAR_COLOURS)]
        p.point(x, y - 2, colour)
        p.point(x, y - 1, shade(colour, 0.6))


def _oval(
    x: float, y: float, radius: float, rise: float, start: float, end: float, steps: int = 17
) -> list[Point]:
    """Points on the ground circle of `radius` about (x, y), raised by `rise`, from angle `start`
    to `end`: angle 0 is to the right and a quarter turn is to the front."""
    return [
        (x + radius * math.cos(a), y + radius / 2 * math.sin(a) - rise)
        for a in np.linspace(start, end, steps)
    ]


def _tree(p: Painter, x: float, y: float, size: float = 3) -> None:
    _, mid, light = LEAVES[0]
    p.line([(x, y), (x, y - size)], TRUNK)
    p.ellipse(x - size, y - 3 * size, x + size, y - size, mid)
    p.blob(x - size + 1, y - 3 * size + 1, x, y - 2 * size, light)


def _tub(p: Painter, x: float, y: float) -> None:
    """A flower tub standing on the ground at (x, y)."""
    p.rect(x - 1, y - 1.5, x + 1, y, (150, 100, 66))
    p.blob(x - 1.4, y - 3.2, x + 1.4, y - 1, (70, 128, 52))
    p.point(x, y - 2.6, (226, 92, 92))


def _digit(p: Painter, x: float, y: float, digit: int, colour: RGB) -> None:
    """A digit three pixels wide and five tall with its top left at (x, y)."""
    for row, bits in enumerate(DIGITS[digit]):
        for col, bit in enumerate(bits):
            if bit == "1":
                p.point(x + col, y + row, colour)


def _car(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A little car on the ground at (x, y), its bonnet to the front left."""
    body = p.hall(x, y, 4, 2, 1.2, colour, shade(colour, 1.15))
    cab = p.hall(x + 0.4, y - 1.4, 2, 1.6, 1, shade(colour, 1.1), colour)
    _patch(p, cab[0], cab[1], (0.1, 0.9), (0.2, 0.9), shade(WINDOW, 1.4))
    for t in (0.2, 0.8):
        p.point(*_on_wall(body[0], body[1], t, 1), (36, 36, 40))


def _water_tower(p: Painter, x: float, y: float, height: float = 14) -> None:
    legs, tank = (96, 88, 80), (176, 150, 118)
    for dx in (-3, 3):
        p.line([(x + dx, y), (x + dx * 0.7, y - height)], legs)
    p.line([(x, y + 1.5), (x, y - height)], shade(legs, 0.8))
    p.line([(x - 2.5, y - height / 2), (x + 2.5, y - height / 2)], legs)
    _turned(p, x, y - height, [(0, 3.5), (5, 3.5)], tank)
    _turned(p, x, y - height - 5, [(0, 4), (2.5, 0)], shade(tank, 0.7))


def space_center() -> Sprite:
    """A space centre: a tall white rocket with fins and a red nose on its launch pad over a
    flame trench, beside a red gantry tower; a huge vehicle assembly hall with a striped door and
    a flag painted on; and a low mission-control building with a door and a dish."""
    p = Painter(50, 52, (24, 40))
    _plate(p, (184, 182, 176))
    p.line([_iso(0, 0, -2, 4), _iso(0, 0, 5, 4)], (150, 148, 142))
    vab = p.hall(*_iso(0, 0, -6, 4), 8, 8, 22, (224, 226, 230), (178, 182, 190))
    _plinth(p, vab, 22, (224, 226, 230))
    _foot(p, vab, 22)
    _patch(p, vab[0], vab[1], (0.3, 0.7), (3, 22), (70, 74, 84))
    for drop in (7, 12, 17):
        p.line(
            [_on_wall(vab[0], vab[1], 0.3, drop), _on_wall(vab[0], vab[1], 0.7, drop)],
            (96, 100, 110),
        )
    for a, b in ((vab[0], vab[1]), (vab[1], vab[2])):
        p.line([_on_wall(a, b, 0, 1.5), _on_wall(a, b, 1, 1.5)], (60, 90, 170))
    _patch(p, vab[1], vab[2], (0.25, 0.75), (4, 7), (60, 90, 170))
    for drop in (5, 6.2):
        p.line(
            [_on_wall(vab[1], vab[2], 0.25, drop), _on_wall(vab[1], vab[2], 0.75, drop)],
            (200, 64, 60),
        )
    for t in (0.2, 0.5, 0.8):
        p.line([_on_wall(vab[1], vab[2], t, 10), _on_wall(vab[1], vab[2], t, 20)], (190, 192, 198))
    px, py = _iso(0, 0, 6, 4)
    pad = p.box(px, py, 5, 2, (150, 148, 144), (170, 168, 162))
    _foot(p, pad, 2)
    _patch(p, pad[0], pad[1], (0.3, 0.7), (0.4, 2), DOORWAY)
    steel, white = (196, 84, 52), (242, 242, 238)
    for x in (px + 3, px + 5):
        p.line([(x, py - 3), (x, py - 33)], steel)
    for y in range(-6, -33, -4):
        p.line([(px + 3, py + y), (px + 5, py + y - 2)], steel)
    p.line([(px + 1, py - 25), (px + 3, py - 25)], steel)
    for dx in (-2.5, 2.5):
        _turned(p, px + dx, py - 2, [(0, 1.2), (12, 1.2), (14, 0)], white)
    for dx in (-3.5, 3.5):
        p.polygon(
            [(px + dx * 0.4, py - 6), (px + dx, py - 2), (px + dx * 0.4, py - 2)], (60, 60, 68)
        )
    _turned(p, px, py - 2, [(0, 2.5), (22, 2.5), (26, 1.6), (29, 0)], white)
    _turned(p, px, py - 26, [(0, 1.6), (3, 0)], (200, 64, 60))
    p.point(px - 1, py - 18, (60, 90, 170))
    p.line([(px - 2, py - 14), (px + 2, py - 14)], (40, 40, 46))
    p.line([(px - 2, py - 22), (px + 2, py - 22)], (40, 40, 46))
    control = p.hall(*_iso(0, 0, 0, -6), 13, 5, 4, (204, 208, 214), (150, 156, 166))
    _plinth(p, control, 4, (204, 208, 214), 0.6)
    _patch(p, control[0], control[1], (0.05, 0.95), (1, 2.5), (90, 150, 196))
    _patch(p, control[0], control[1], (0.58, 0.68), (1.4, 4), DOORWAY)
    _foot(p, control, 4)
    _doorstep(p, control[0], control[1], (0.55, 0.71), 4, (214, 212, 206))
    _patch(p, control[1], control[2], (0.2, 0.8), (1, 2.5), shade((90, 150, 196), 0.75))
    dx, dy = _iso(0, 0, -4, -6)
    p.line([(dx, dy - 4), (dx, dy - 7)], METAL)
    p.ellipse(dx - 3, dy - 11, dx + 2, dy - 7, (236, 238, 240))
    p.line([(dx - 0.5, dy - 9), (dx + 0.5, dy - 12)], (110, 110, 116))
    return p.sprite()


def pear_hq() -> Sprite:
    """A modern campus: a great ring of glass offices with solar panels all round its roof,
    around a wooded park, with paths out to the trees all round it."""
    p = Painter(50, 38, (24, 24))
    _plate(p, (112, 160, 80), 20, 24)
    gravel = (214, 206, 176)
    for a in (0.35, math.pi / 2 + 0.25, math.pi - 0.35):
        p.polygon(
            _oval(0, 0, 14, 0, a - 0.09, a + 0.09, 3) + _oval(0, 0, 19.5, 0, a + 0.07, a - 0.07, 3),
            gravel,
        )
    for x, y in ((-16, -2), (-9, -7), (0, -9), (9, -7), (16, -2)):
        _tree(p, x, y)
    big, small, h = 16, 10, 5
    roof, glass = (226, 230, 232), (120, 176, 204)
    p.polygon(_oval(0, 0, big, h, 0, 2 * math.pi, 33), roof, shade(roof, 0.7))
    p.polygon(_oval(0, 0, small, h, 0, 2 * math.pi, 25), shade(glass, 0.72))
    p.polygon(
        _oval(0, 0, small, 0, math.pi, 2 * math.pi) + _oval(0, 0, small, h, 0, math.pi),
        (98, 152, 70),
    )
    for x, y in ((-5, -2), (1, -3), (5, -1), (-1, -0.5)):
        _tree(p, x, y, 2)
    p.polygon(
        _oval(0, 0, big, h, 0, math.pi) + _oval(0, 0, small, h, math.pi, 0), roof, shade(roof, 0.75)
    )
    panel = (66, 92, 140)
    for a in np.linspace(0, 2 * math.pi, 29)[:-1]:
        p.line(_oval(0, 0, 13, h, a, a + 0.13, 2), panel)
    split = math.acos(1 / 3)
    p.polygon(_oval(0, 0, big, 0, split, math.pi) + _oval(0, 0, big, h, math.pi, split), glass)
    p.polygon(
        _oval(0, 0, big, 0, 0, split, 9) + _oval(0, 0, big, h, split, 0, 9), shade(glass, 0.78)
    )
    for a in np.linspace(0.2, math.pi - 0.2, 12):
        bx, by = big * math.cos(a), big / 2 * math.sin(a)
        p.line([(bx, by - 0.5), (bx, by - h + 0.5)], shade(glass, 1.3))
    p.line(_oval(0, 0, big, 0, 0, math.pi), shade(glass, 0.55))
    p.line(_oval(0, 0.6, big, 0, 0.1, math.pi - 0.1), SOIL)
    door = math.pi / 2 + 0.25
    p.polygon(
        _oval(0, 0, big, 0, door - 0.1, door + 0.1, 3)
        + _oval(0, 0, big, 3.5, door + 0.1, door - 0.1, 3),
        (52, 74, 96),
    )
    p.line(_oval(0, 0, big, 3.5, door - 0.12, door + 0.12, 3), (240, 240, 236))
    for x, y in ((-18, 3), (-12, 7), (12, 7), (18, 3), (0, 10)):
        _tree(p, x, y)
    return p.sprite()


def royal_palace() -> Sprite:
    """A baroque royal palace: long symmetrical wings with rows of tall arched windows under a
    roof with chimneys, end pavilions and a central pavilion with a pediment over steps under a
    green dome, behind a formal garden with a gravel walk and a fountain, closed off by railings
    with a gate."""
    p = Painter(50, 40, (24, 27))
    _plate(p, (206, 198, 166))
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-1.3, -10), (1.3, -10), (1.3, 1), (-1.3, 1))],
        (226, 220, 196),
    )
    hedge, bed = (70, 120, 56), (98, 154, 72)
    for u in (-6, 6):
        corners = [
            _iso(0, 0, u + du, -5 + dv)
            for du, dv in ((-3.5, -3.5), (3.5, -3.5), (3.5, 3.5), (-3.5, 3.5))
        ]
        p.polygon(corners, bed, hedge)
        p.line([_iso(0, 0, u - 3.5, -5), _iso(0, 0, u + 3.5, -5)], hedge)
    fx, fy = _iso(0, 0, 0, -5)
    p.ellipse(fx - 3, fy - 1.5, fx + 3, fy + 1.5, (96, 160, 206))
    p.line([(fx, fy), (fx, fy - 4)], (226, 240, 250))
    railing = (54, 56, 62)
    for side in (-1, 1):
        start, end = _iso(0, 0, side * 1.8, -9.6), _iso(0, 0, side * 9.8, -9.6)
        p.line([(start[0], start[1] - 2), (end[0], end[1] - 2)], railing)
        for k in range(5):
            x, y = _on_wall(start, end, k / 4, 0)
            p.line([(x, y), (x, y - 2)], railing)
        p.rect(start[0] - 0.6, start[1] - 3.5, start[0] + 0.6, start[1], (226, 220, 196))
        p.point(start[0] - 0.5, start[1] - 4.5, CROSS)
    wall, roof = (236, 214, 160), (96, 106, 126)
    wing = p.hall(*_iso(0, 0, 0, 6), 21, 5, 7, wall, None)
    _plinth(p, wing, 7, wall)
    _foot(p, wing, 7)
    _windows(
        p,
        wing[0],
        wing[1],
        tuple(k / 17 for k in range(1, 17)),
        (1.5, 4.5),
        2,
        arch=shade(wall, 0.75),
    )
    p.gable(wing, 3, roof, wall)
    _tiles(p, wing, 3, roof)
    for t in (0.2, 0.8):
        x, y = _on_wall(wing[0], wing[1], t, 0)
        x, y = x + 2.5, y - 3
        p.rect(x - 0.6, y - 2, x + 0.6, y + 0.5, shade(wall, 0.8))
    dome_x, dome_y = _iso(0, 0, 0, 5)
    _turned(p, dome_x, dome_y - 9, [(0, 3.5), (3, 3.5)], wall)
    _turned(p, dome_x, dome_y - 12, [(0, 4), (2, 3.4), (3.5, 2.2), (5, 0)], VERDIGRIS)
    _flag(p, dome_x, dome_y - 17, 6)
    for u in (-9, 9):
        pavilion = p.hall(*_iso(0, 0, u, 4.5), 3.5, 7, 9, wall, None)
        _windows(p, pavilion[0], pavilion[1], (0.5,), (1.5, 4.5), 2)
        _windows(p, pavilion[1], pavilion[2], (0.3, 0.7), (1.5, 4.5), 2, shade(WINDOW, 0.75))
        _plinth(p, pavilion, 9, wall)
        _foot(p, pavilion, 9)
        p.gable(pavilion, 3, roof, wall)
    centre = p.hall(*_iso(0, 0, 0, 2.5), 6, 3, 9, shade(wall, 1.04), None)
    _windows(p, centre[0], centre[1], (0.25, 0.5, 0.75), (1.5, 4.5), 2)
    _plinth(p, centre, 9, wall)
    _foot(p, centre, 9)
    _portal(p, centre[0], centre[1], (0.38, 0.62), 9, 2)
    near, far = _on_wall(centre[0], centre[1], 0.3, 9), _on_wall(centre[0], centre[1], 0.7, 9)
    p.polygon(
        [near, far, (far[0] - 1.5, far[1] + 0.75), (near[0] - 1.5, near[1] + 0.75)],
        (236, 230, 214),
        (176, 170, 156),
    )
    _gable_across(p, centre, 3, roof, shade(wall, 1.04))
    tympanum = _on_wall(centre[0], centre[1], 0.5, -1)
    p.point(tympanum[0] - 0.5, tympanum[1], CROSS)
    return p.sprite()


def salt_cathedral() -> Sprite:
    """A salt mine's head: a boarded timber shaft house on a stone plinth with a tall tower over
    the shaft and a headframe wheel, white heaps of salt, a mine cart of salt on rails, and a
    little white chapel with a cross marking the cathedral carved in the salt below."""
    p = Painter(50, 46, (24, 34))
    _plate(p, (178, 172, 154))
    timber, shingle = (138, 98, 62), (92, 70, 54)
    wx, wy = _iso(0, 0, 4, 8)
    frame = (74, 64, 58)
    p.line([(wx - 3, wy), (wx, wy - 22), (wx + 3, wy)], frame)
    p.line([(wx - 2, wy - 8), (wx + 2, wy - 8)], frame)
    p.ellipse(wx - 3, wy - 26, wx + 3, wy - 20, frame)
    p.blob(wx - 1.5, wy - 24.5, wx + 1.5, wy - 21.5, (178, 172, 154))
    house = p.hall(*_iso(0, 0, -3, 3), 12, 7, 6, timber, None)
    _worn(p, house[0], house[1], (0.45, 0.72), 6, 4)
    for k in range(1, 12):
        t = k / 12
        p.line(
            [_on_wall(house[0], house[1], t, 0.5), _on_wall(house[0], house[1], t, 6)],
            shade(timber, 0.85),
        )
    for k in range(1, 7):
        t = k / 7
        p.line(
            [_on_wall(house[1], house[2], t, 0.5), _on_wall(house[1], house[2], t, 6)],
            shade(timber, 0.66),
        )
    _plinth(p, house, 6, STONE, 1)
    _foot(p, house, 6)
    _windows(p, house[0], house[1], (0.15, 0.35, 0.8), (2,), 2, (240, 220, 150))
    _portal(p, house[0], house[1], (0.5, 0.66), 6, 3)
    p.gable(house, 5, shingle, timber)
    _tiles(p, house, 5, shingle, (0.34, 0.68))
    p.line([(wx, wy - 23), _on_wall(house[2], house[3], 0.5, -6)], (52, 52, 58))
    tx, ty = _iso(0, 0, -3, 3)
    tower = p.box(tx, ty - 6, 3, 10, shade(timber, 1.1), None)
    _windows(p, tower[0], tower[1], (0.5,), (2,), 3, (54, 46, 48))
    apex = (tx, ty - 22)
    p.polygon([tower[0], tower[1], apex], shingle, shade(shingle, 0.55))
    p.polygon([tower[1], tower[2], apex], shade(shingle, 0.7), shade(shingle, 0.45))
    white = (240, 242, 246)
    rail = (96, 90, 86)
    for u in (-2.4, -1.2):
        p.line([_iso(0, 0, u, -0.5), _iso(0, 0, u, -9.5)], rail)
    for u, v, half, height in ((7, -2, 4, 6), (3, -7, 3.5, 5), (8, -7, 3, 4)):
        _heap(p, *_iso(0, 0, u, v), half, height, white)
    cart = p.hall(*_iso(0, 0, -1.8, -5), 2, 3, 2, (84, 80, 80), None)
    cx, cy = (cart[0][0] + cart[2][0]) / 2, (cart[0][1] + cart[2][1]) / 2
    p.blob(cx - 2, cy - 1.5, cx + 2, cy + 1, white)
    cx, cy = _iso(0, 0, -7, -6)
    _worn(p, (cx - 3, cy - 4), (cx, cy - 2.5), (0.3, 0.7), 4, 3.5, GRAVEL)
    chapel = p.box(cx, cy, 3, 4, WHITEWASH, None)
    p.gable(chapel, 3, CHURCH_ROOF, WHITEWASH)
    _plinth(p, chapel, 4, WHITEWASH, 0.6)
    _foot(p, chapel, 4)
    _portal(p, chapel[0], chapel[1], (0.35, 0.65), 4, 2)
    p.point(*_on_wall(chapel[1], chapel[2], 0.5, 1.5), (62, 78, 132))
    ridge = _on_wall(chapel[1], chapel[2], 0.5, -3)
    _cross(p, ridge[0], ridge[1])
    return p.sprite()


def spa_resort() -> Sprite:
    """A spa resort: an elegant white spa house under a green roof and a cupola with windows, a
    long colonnade for taking the waters, a pool with a fountain, benches and lamps beside it,
    and a small park of trees."""
    p = Painter(50, 40, (24, 27))
    _plate(p, (114, 162, 82))
    gravel = (214, 206, 176)
    p.polygon([_iso(0, 0, u, v) for u, v in ((-8, -9.5), (8, -9.5), (8, 0), (-8, 0))], gravel)
    for u, v in ((-9.5, 9), (9.5, 9), (-9.5, 3)):
        _tree(p, *_iso(0, 0, u, v))
    cream = (242, 234, 216)
    house = p.hall(*_iso(0, 0, 0, 6.5), 16, 6, 8, cream, None)
    _plinth(p, house, 8, cream, 0.8)
    _foot(p, house, 8)
    _windows(
        p, house[0], house[1], tuple(k / 9 for k in range(1, 9)), (2,), 3, arch=shade(cream, 0.8)
    )
    p.gable(house, 4, VERDIGRIS, cream)
    _tiles(p, house, 4, VERDIGRIS, (0.4, 0.75))
    for t in (0.25, 0.75):
        x, y = _on_wall(house[0], house[1], t, 0)
        p.rect(x + 1.5, y - 4.5, x + 2.5, y - 2, shade(cream, 0.85))
    cx, cy = _iso(0, 0, 0, 6.5)
    _turned(p, cx, cy - 11, [(0, 2), (2, 2)], cream)
    p.line([(cx - 1, cy - 11.5), (cx - 1, cy - 12.5)], WINDOW)
    _turned(p, cx, cy - 13, [(0, 2.4), (1.5, 1.6), (2.5, 0)], VERDIGRIS)
    walk = p.hall(*_iso(0, 0, 0, 1.5), 18, 2.5, 4, (88, 80, 76), VERDIGRIS)
    _foot(p, walk, 4)
    for k in range(10):
        t = (k + 0.5) / 10
        p.line([_on_wall(walk[0], walk[1], t, 0.5), _on_wall(walk[0], walk[1], t, 4)], cream)
    pool = [_iso(0, 0, u, v) for u, v in ((-6, -8), (6, -8), (6, -3), (-6, -3))]
    p.polygon(pool, (92, 172, 214), (240, 240, 236))
    px, py = _iso(0, 0, 0, -5.5)
    p.ellipse(px - 2, py - 1, px + 2, py + 1, (240, 240, 236))
    p.line([(px, py), (px, py - 5)], (226, 242, 252))
    for dx in (-2, 2):
        p.point(px + dx, py - 3, (226, 242, 252))
    bench = (120, 84, 56)
    for u in (-4.5, 2.5):
        a, b = _iso(0, 0, u, -9), _iso(0, 0, u + 2, -9)
        p.line([(a[0], a[1] - 1), (b[0], b[1] - 1)], bench)
        p.line([a, (a[0], a[1] - 1)], shade(bench, 0.6))
        p.line([b, (b[0], b[1] - 1)], shade(bench, 0.6))
    for u in (-7, 7):
        _lamp(p, *_iso(0, 0, u, -2))
    for u, v in ((-9.5, -6), (9.5, -6), (9.5, -1)):
        _tree(p, *_iso(0, 0, u, v))
    return p.sprite()


def worlds_fair() -> Sprite:
    """A world's fair: a vast exhibition palace of iron and glass on a stone plinth under
    barrel-vaulted glass roofs, its taller middle crowned with flags, with lamps and visitors on
    the square before its door."""
    p = Painter(50, 46, (24, 32))
    _plate(p, (206, 200, 186))
    wall, roof, iron = (184, 220, 234), (214, 236, 244), (86, 98, 118)
    main = p.hall(0, 0, 21, 9, 8, wall, None)
    for k in range(1, 7):
        p.line([_on_wall(main[0], main[1], k / 7, 0.5), _on_wall(main[0], main[1], k / 7, 8)], iron)
    for k in range(1, 3):
        p.line(
            [_on_wall(main[1], main[2], k / 3, 0.5), _on_wall(main[1], main[2], k / 3, 8)],
            shade(iron, 0.8),
        )
    for a, b in ((main[0], main[1]), (main[1], main[2])):
        p.line([_on_wall(a, b, 0, 4), _on_wall(a, b, 1, 4)], iron)
    _plinth(p, main, 8, STONE, 1)
    _foot(p, main, 8)
    _vault(p, main, 5, roof, wall, (0.15, 0.3, 0.7, 0.85), iron)
    centre = p.hall(0, 0, 7, 12, 13, wall, None)
    _plinth(p, centre, 13, STONE, 1)
    _foot(p, centre, 13)
    for k in range(1, 4):
        p.line(
            [_on_wall(centre[0], centre[1], k / 4, 0.5), _on_wall(centre[0], centre[1], k / 4, 13)],
            iron,
        )
    p.line([_on_wall(centre[0], centre[1], 0, 6), _on_wall(centre[0], centre[1], 1, 6)], iron)
    _vault(p, centre, 6, roof, wall, (0.25, 0.5, 0.75), iron)
    _portal(p, centre[0], centre[1], (0.3, 0.7), 13, 5)
    _doorstep(p, centre[0], centre[1], (0.25, 0.75), 13, (226, 222, 212))
    for t in (0.1, 0.9):
        x, y = _on_wall(centre[0], centre[1], t, 13)
        _tub(p, x - 2, y + 1)
    poles = (
        _on_wall(centre[0], centre[3], 0.5, -6),
        _on_wall(main[0], main[3], 0.5, -5),
        _on_wall(main[1], main[2], 0.5, -5),
    )
    for (x, y), colour in zip(poles, ((214, 58, 54), (60, 110, 200), (240, 200, 70)), strict=True):
        _flag(p, x, y, 7, colour)
    for x, y in ((-12, 3.5), (-1, 9)):
        _lamp(p, x, y)
    _people(p, [(-9, 5), (-6.5, 6.5), (-4, 6), (-7, 8.5), (-2.5, 8), (-11, 7)])
    return p.sprite()


def film_studio() -> Sprite:
    """A film studio: big sound-stage sheds with their numbers painted on, a door under a red
    recording light and a great sliding door; a water tower on legs, a star's car, and the studio
    gate with its sign."""
    p = Painter(50, 44, (24, 31))
    _plate(p, (150, 150, 146))
    stage, paint = (218, 206, 180), (60, 56, 60)
    for (u, v), length, depth, height, number in (((6.5, 6), 8, 8, 9, 2), ((-4, 5), 12, 9, 11, 1)):
        top = p.hall(*_iso(0, 0, u, v), length, depth, height, stage, None)
        _plinth(p, top, height, stage)
        _foot(p, top, height)
        p.gable(top, 3, (120, 116, 112), stage)
        _tiles(p, top, 3, (120, 116, 112))
        x, y = _on_wall(top[0], top[1], 0.5, 2.5)
        p.rect(x - 3, y - 1, x + 3, y + 6, (240, 236, 226))
        _digit(p, x - 1, y, number, paint)
        _patch(p, top[0], top[1], (0.12, 0.24), (height - 3.5, height), DOORWAY)
        _doorstep(p, top[0], top[1], (0.09, 0.27), height, (196, 194, 188))
        p.point(*_on_wall(top[0], top[1], 0.17, height - 5), (236, 40, 36))
        _worn(p, top[1], top[2], (0.25, 0.75), height, 2.5, (128, 128, 124))
        _patch(p, top[1], top[2], (0.25, 0.75), (3, height), shade(stage, 0.55))
        for t in (0.375, 0.5, 0.625):
            p.line(
                [_on_wall(top[1], top[2], t, 3), _on_wall(top[1], top[2], t, height)],
                shade(stage, 0.45),
            )
    _water_tower(p, *_iso(0, 0, 7, -4))
    gx, gy = _iso(0, 0, -5, -8)
    pillar = (230, 220, 196)
    for dx in (-5, 5):
        p.rect(gx + dx - 1, gy + dx / 2 - 8, gx + dx + 1, gy + dx / 2, pillar)
    p.polygon(
        [(gx - 6, gy - 11.5), (gx + 6, gy - 5.5), (gx + 6, gy - 8.5), (gx - 6, gy - 14.5)],
        (200, 64, 56),
        shade((200, 64, 56), 0.6),
    )
    p.line([(gx - 4, gy - 11), (gx + 4, gy - 7)], (250, 214, 96))
    p.box(*_iso(0, 0, 0, -9), 2, 3, (240, 236, 226), shade(STONE, 1.1))
    _car(p, -4, 6.5, CAR_COLOURS[0])
    return p.sprite()


def amusement_park() -> Sprite:
    """An amusement park: a big Ferris wheel with coloured gondolas, a roller coaster looping the
    loop with cars on it, a carousel, striped stalls along its paths, a bunch of balloons and
    visitors."""
    p = Painter(50, 50, (24, 37))
    _plate(p, (116, 164, 84))
    gravel = (214, 206, 176)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-10.5, -2), (10.5, -2), (10.5, -4), (-10.5, -4))], gravel
    )
    rail, post = (214, 56, 56), (236, 236, 232)
    track = [(-1, 9, 2), (2, 9, 10), (4, 9, 3), (6, 8, 3), (8, 6, 7), (10, 3, 2), (9, 0, 2)]
    points = []
    for u, v, h in track:
        x, y = _iso(0, 0, u, v)
        p.line([(x, y), (x, y - h)], post)
        points.append((x, y - h))
    p.line(points, rail)
    for f, colour in ((0.25, (240, 200, 70)), (0.6, (60, 110, 200))):
        x0, y0 = points[5]
        x1, y1 = points[6]
        x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        p.rect(x - 1, y - 2, x + 1, y, colour)
    lx, ly = _iso(0, 0, 6, 8)
    loop = [
        (lx + 4 * math.cos(a), ly - 7 + 5 * math.sin(a))
        for a in np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, 25)
    ]
    p.line(loop, rail)
    wx, wy = _iso(0, 0, -5, 4)
    hub = (wx, wy - 14)
    frame = (120, 120, 132)
    p.line([(wx - 5, wy), hub, (wx + 5, wy)], frame)
    rim = [
        (hub[0] + 11 * math.cos(a), hub[1] + 11 * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, 33)
    ]
    for a in np.linspace(0, math.pi, 5)[:-1]:
        p.line(
            [
                (hub[0] - 11 * math.cos(a), hub[1] - 11 * math.sin(a)),
                (hub[0] + 11 * math.cos(a), hub[1] + 11 * math.sin(a)),
            ],
            frame,
        )
    p.line(rim, (236, 236, 240))
    for k, a in enumerate(np.linspace(0, 2 * math.pi, 9)[:-1]):
        gx, gy = hub[0] + 11 * math.cos(a), hub[1] + 11 * math.sin(a)
        p.rect(gx - 1, gy, gx + 1, gy + 2, CAR_COLOURS[k % len(CAR_COLOURS)])
    p.ellipse(hub[0] - 1, hub[1] - 1, hub[0] + 1, hub[1] + 1, (236, 236, 240))
    for (u, v), awning in (
        ((-8, -7), (214, 56, 56)),
        ((-3, -8.5), (60, 110, 200)),
        ((9, -6), (240, 200, 70)),
    ):
        stall = p.box(*_iso(0, 0, u, v), 2.5, 3, (236, 230, 214), None)
        _foot(p, stall, 3)
        _patch(p, stall[0], stall[1], (0.2, 0.8), (1, 1.6), shade(awning, 0.9))
        p.polygon(
            [
                _on_wall(stall[0], stall[3], 0, -1.5),
                stall[0],
                stall[1],
                stall[2],
                _on_wall(stall[3], stall[2], 1, -1.5),
                (stall[3][0], stall[3][1] - 1.5),
            ],
            awning,
            shade(awning, 0.6),
        )
        p.line([stall[0], stall[1], stall[2]], (250, 250, 248))
    cx, cy = _iso(0, 0, 3, -7)
    _turned(p, cx, cy, [(0, 4), (1, 4)], (230, 220, 196))
    for a in np.linspace(0.4, math.pi - 0.4, 4):
        px = cx + 3.5 * math.cos(a)
        p.line([(px, cy + 1.7 * math.sin(a)), (px, cy + 1.7 * math.sin(a) - 4)], (240, 200, 70))
    _turned(p, cx, cy - 5, [(0, 4.5), (3.5, 0)], (214, 56, 56))
    _flag(p, cx, cy - 8.5, 3, (240, 200, 70))
    _people(p, [_iso(0, 0, u, v) for u, v in ((-5, -10), (-1, -9.5), (6, -9.5), (9, -9))])
    bx, by = _iso(0, 0, -9.5, -3)
    p.line([(bx, by), (bx, by - 5)], (236, 236, 232))
    for (dx, dy), colour in zip(((-1, -6), (1, -6.5), (0, -8)), CAR_COLOURS, strict=False):
        p.blob(bx + dx - 1, by + dy - 1.2, bx + dx + 1, by + dy + 1.2, colour)
    return p.sprite()


def olympic_stadium() -> Sprite:
    """An Olympic stadium: an oval bowl of stands around a red running track and a green field
    with goals, a scoreboard showing the rings at the far end, floodlight masts round its rim and
    the flame burning in a cauldron on a tall stand."""
    p = Painter(52, 44, (25, 30))
    big, small, h = 20, 12, 7
    p.polygon(_oval(0, 0.5, big + 2.5, 0, 0, 2 * math.pi, 41), PAVING, shade(PAVING, 0.8))
    for a in (math.pi / 2 - 0.5, math.pi / 2 + 0.5):
        p.polygon(
            _oval(0, 0.5, big, 0, a - 0.12, a + 0.12, 3)
            + _oval(0, 0.5, big + 4, 0, a + 0.08, a - 0.08, 3),
            PAVING,
        )
    stand, seats = (196, 196, 202), ((198, 64, 60), (64, 96, 176))
    p.polygon(_oval(0, 0, big, h, 0, 2 * math.pi, 41), stand, shade(stand, 0.7))
    for k in range(1, 6):
        f = k / 6
        p.line(
            _oval(0, 0, big + (small - big) * f, h * (1 - f), math.pi, 2 * math.pi, 25),
            seats[k % 2],
        )

    def mast(angle: float) -> None:
        mx, my = big * math.cos(angle), big / 2 * math.sin(angle) - h
        p.line([(mx, my), (mx, my - 13)], (120, 120, 128))
        p.rect(mx - 1.5, my - 16, mx + 1.5, my - 13, (250, 246, 210))

    for angle in (math.pi + 0.7, 2 * math.pi - 0.7):
        mast(angle)
    for dx in (-3, 3):
        p.line([(dx, -big / 2 - h), (dx, -big / 2 - h - 2)], (70, 70, 80))
    p.rect(-5, -big / 2 - h - 6, 5, -big / 2 - h - 2, (40, 42, 50))
    rings = ((60, 110, 200), (230, 230, 230), (214, 56, 54), (240, 200, 70), (60, 160, 100))
    for k, colour in enumerate(rings):
        p.point(-3 + 1.5 * k, -big / 2 - h - 5 + (k % 2), colour)
    p.polygon(_oval(0, 0, small, 0, 0, 2 * math.pi, 33), (192, 82, 62))
    p.line(_oval(0, 0, small - 1.5, 0, 0, 2 * math.pi, 33), (236, 200, 190))
    p.polygon(_oval(0, 0, small - 3, 0, 0, 2 * math.pi, 33), (92, 156, 68))
    p.line([(0, -3.5), (0, 3.5)], (220, 240, 210))
    for x in (-7.5, 7.5):
        p.line([(x, -1), (x, 0)], (250, 250, 248))
    split = math.acos(1 / 3)
    p.polygon(
        _oval(0, 0, big, 0, split, math.pi) + _oval(0, 0, big, h, math.pi, split),
        shade(stand, 0.92),
    )
    p.polygon(
        _oval(0, 0, big, 0, 0, split, 9) + _oval(0, 0, big, h, split, 0, 9), shade(stand, 0.74)
    )
    for a in np.linspace(0.25, math.pi - 0.25, 10):
        ax, ay = big * math.cos(a), big / 2 * math.sin(a)
        p.line([(ax, ay - 1), (ax, ay - 3)], (70, 70, 80))
    p.line(_oval(0, 0, big, h, 0, math.pi, 25), shade(stand, 1.15))
    p.line(_oval(0, 0.6, big, 0, 0.15, math.pi - 0.15, 25), shade(PAVING, 0.55))
    for angle in (0.7, math.pi - 0.7):
        mast(angle)
    cx, cy = -big - 1, -h
    p.polygon([(cx - 2, cy + 5), (cx + 2, cy + 5), (cx + 1, cy + 6), (cx - 3, cy + 6)], SOIL)
    p.rect(cx - 1, cy - 12, cx + 1, cy + 5, (150, 150, 158))
    p.ellipse(cx - 3, cy - 14, cx + 3, cy - 11, CROSS)
    p.blob(cx - 2.5, cy - 20, cx + 2.5, cy - 13, (246, 140, 40))
    p.blob(cx - 1, cy - 18, cx + 1, cy - 14, (255, 220, 110))
    return p.sprite()
