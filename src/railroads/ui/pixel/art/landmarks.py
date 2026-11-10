"""The one-off buildings a game may get (railroads.sim.novelties)."""

import math

import numpy as np

from railroads.ui.pixel.art.painter import (
    CAR_COLOURS,
    CHURCH_ROOF,
    CROSS,
    DIGITS,
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


def _digit(p: Painter, x: float, y: float, digit: int, colour: RGB) -> None:
    """A digit three pixels wide and five tall with its top left at (x, y)."""
    for row, bits in enumerate(DIGITS[digit]):
        for col, bit in enumerate(bits):
            if bit == "1":
                p.point(x + col, y + row, colour)


def _water_tower(p: Painter, x: float, y: float, height: float = 14) -> None:
    legs, tank = (96, 88, 80), (176, 150, 118)
    for dx in (-3, 3):
        p.line([(x + dx, y), (x + dx * 0.7, y - height)], legs)
    p.line([(x, y + 1.5), (x, y - height)], shade(legs, 0.8))
    p.line([(x - 2.5, y - height / 2), (x + 2.5, y - height / 2)], legs)
    _turned(p, x, y - height, [(0, 3.5), (5, 3.5)], tank)
    _turned(p, x, y - height - 5, [(0, 4), (2.5, 0)], shade(tank, 0.7))


def space_center() -> Sprite:
    """A space centre: a tall white rocket on its launch pad beside a red gantry tower, a huge
    vehicle assembly hall and a low mission-control building with a dish."""
    p = Painter(50, 52, (24, 40))
    _plate(p, (184, 182, 176))
    p.line([_iso(0, 0, -2, 4), _iso(0, 0, 5, 4)], (150, 148, 142))
    vab = p.hall(*_iso(0, 0, -6, 4), 8, 8, 22, (224, 226, 230), (178, 182, 190))
    _patch(p, vab[0], vab[1], (0.3, 0.7), (3, 22), (70, 74, 84))
    for a, b in ((vab[0], vab[1]), (vab[1], vab[2])):
        p.line([_on_wall(a, b, 0, 1.5), _on_wall(a, b, 1, 1.5)], (60, 90, 170))
    px, py = _iso(0, 0, 6, 4)
    p.box(px, py, 5, 2, (150, 148, 144), (170, 168, 162))
    steel, white = (196, 84, 52), (242, 242, 238)
    for x in (px + 3, px + 5):
        p.line([(x, py - 3), (x, py - 33)], steel)
    for y in range(-6, -33, -4):
        p.line([(px + 3, py + y), (px + 5, py + y - 2)], steel)
    p.line([(px + 1, py - 25), (px + 3, py - 25)], steel)
    for dx in (-2.5, 2.5):
        _turned(p, px + dx, py - 2, [(0, 1.2), (12, 1.2), (14, 0)], white)
    _turned(p, px, py - 2, [(0, 2.5), (22, 2.5), (26, 1.6), (29, 0)], white)
    p.line([(px - 2, py - 14), (px + 2, py - 14)], (40, 40, 46))
    p.line([(px - 2, py - 22), (px + 2, py - 22)], (40, 40, 46))
    control = p.hall(*_iso(0, 0, 0, -6), 13, 5, 4, (204, 208, 214), (150, 156, 166))
    _patch(p, control[0], control[1], (0.05, 0.95), (1, 2.5), (90, 150, 196))
    dx, dy = _iso(0, 0, -4, -6)
    p.line([(dx, dy - 4), (dx, dy - 7)], METAL)
    p.ellipse(dx - 3, dy - 11, dx + 2, dy - 7, (236, 238, 240))
    return p.sprite()


def pear_hq() -> Sprite:
    """A modern campus: a great ring of glass offices around a wooded park, with trees all round
    it."""
    p = Painter(50, 38, (24, 24))
    _plate(p, (112, 160, 80), 20, 24)
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
    split = math.acos(1 / 3)
    p.polygon(_oval(0, 0, big, 0, split, math.pi) + _oval(0, 0, big, h, math.pi, split), glass)
    p.polygon(
        _oval(0, 0, big, 0, 0, split, 9) + _oval(0, 0, big, h, split, 0, 9), shade(glass, 0.78)
    )
    for a in np.linspace(0.2, math.pi - 0.2, 12):
        bx, by = big * math.cos(a), big / 2 * math.sin(a)
        p.line([(bx, by - 0.5), (bx, by - h + 0.5)], shade(glass, 1.3))
    p.line(_oval(0, 0, big, 0, 0, math.pi), shade(glass, 0.55))
    for x, y in ((-18, 3), (-12, 7), (12, 7), (18, 3), (0, 10)):
        _tree(p, x, y)
    return p.sprite()


def royal_palace() -> Sprite:
    """A baroque royal palace: long symmetrical wings with rows of tall windows, end pavilions and
    a central pavilion with a pediment under a green dome, behind a formal garden with a
    fountain."""
    p = Painter(50, 40, (24, 27))
    _plate(p, (206, 198, 166))
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
    wall, roof = (236, 214, 160), (96, 106, 126)
    wing = p.hall(*_iso(0, 0, 0, 6), 21, 5, 7, wall, None)
    _windows(p, wing[0], wing[1], tuple(k / 17 for k in range(1, 17)), (1.5, 4.5), 2)
    p.gable(wing, 3, roof, wall)
    dome_x, dome_y = _iso(0, 0, 0, 5)
    _turned(p, dome_x, dome_y - 9, [(0, 3.5), (3, 3.5)], wall)
    _turned(p, dome_x, dome_y - 12, [(0, 4), (2, 3.4), (3.5, 2.2), (5, 0)], VERDIGRIS)
    _flag(p, dome_x, dome_y - 17, 6)
    for u in (-9, 9):
        pavilion = p.hall(*_iso(0, 0, u, 4.5), 3.5, 7, 9, wall, None)
        _windows(p, pavilion[0], pavilion[1], (0.5,), (1.5, 4.5), 2)
        _windows(p, pavilion[1], pavilion[2], (0.3, 0.7), (1.5, 4.5), 2, shade(WINDOW, 0.75))
        p.gable(pavilion, 3, roof, wall)
    centre = p.hall(*_iso(0, 0, 0, 2.5), 6, 3, 9, shade(wall, 1.04), None)
    _windows(p, centre[0], centre[1], (0.25, 0.5, 0.75), (1.5, 4.5), 2)
    _portal(p, centre[0], centre[1], (0.38, 0.62), 9, 2)
    _gable_across(p, centre, 3, roof, shade(wall, 1.04))
    return p.sprite()


def salt_cathedral() -> Sprite:
    """A salt mine's head: a timber shaft house with a tall tower over the shaft and a headframe
    wheel, white heaps of salt, and a little white chapel with a cross marking the cathedral
    carved in the salt below."""
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
    _windows(p, house[0], house[1], (0.15, 0.35, 0.8), (2,), 2, (240, 220, 150))
    _portal(p, house[0], house[1], (0.5, 0.66), 6, 3)
    p.gable(house, 5, shingle, timber)
    p.line([(wx, wy - 23), _on_wall(house[2], house[3], 0.5, -6)], (52, 52, 58))
    tx, ty = _iso(0, 0, -3, 3)
    tower = p.box(tx, ty - 6, 3, 10, shade(timber, 1.1), None)
    _windows(p, tower[0], tower[1], (0.5,), (2,), 3, (54, 46, 48))
    apex = (tx, ty - 22)
    p.polygon([tower[0], tower[1], apex], shingle, shade(shingle, 0.55))
    p.polygon([tower[1], tower[2], apex], shade(shingle, 0.7), shade(shingle, 0.45))
    white = (240, 242, 246)
    for u, v, half, height in ((7, -2, 4, 6), (3, -7, 3.5, 5), (8, -7, 3, 4)):
        _heap(p, *_iso(0, 0, u, v), half, height, white)
    chapel = p.box(*_iso(0, 0, -7, -6), 3, 4, WHITEWASH, None)
    p.gable(chapel, 3, CHURCH_ROOF, WHITEWASH)
    _portal(p, chapel[0], chapel[1], (0.35, 0.65), 4, 2)
    ridge = _on_wall(chapel[1], chapel[2], 0.5, -3)
    _cross(p, ridge[0], ridge[1])
    return p.sprite()


def spa_resort() -> Sprite:
    """A spa resort: an elegant white spa house under a green roof and cupola, a long colonnade
    for taking the waters, a pool with a fountain and a small park of trees."""
    p = Painter(50, 40, (24, 27))
    _plate(p, (114, 162, 82))
    gravel = (214, 206, 176)
    p.polygon([_iso(0, 0, u, v) for u, v in ((-8, -9.5), (8, -9.5), (8, 0), (-8, 0))], gravel)
    for u, v in ((-9.5, 9), (9.5, 9), (-9.5, 3)):
        _tree(p, *_iso(0, 0, u, v))
    cream = (242, 234, 216)
    house = p.hall(*_iso(0, 0, 0, 6.5), 16, 6, 8, cream, None)
    _windows(
        p, house[0], house[1], tuple(k / 9 for k in range(1, 9)), (2,), 3, arch=shade(cream, 0.8)
    )
    p.gable(house, 4, VERDIGRIS, cream)
    cx, cy = _iso(0, 0, 0, 6.5)
    _turned(p, cx, cy - 11, [(0, 2), (2, 2)], cream)
    _turned(p, cx, cy - 13, [(0, 2.4), (1.5, 1.6), (2.5, 0)], VERDIGRIS)
    walk = p.hall(*_iso(0, 0, 0, 1.5), 18, 2.5, 4, (88, 80, 76), VERDIGRIS)
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
    for u, v in ((-9.5, -6), (9.5, -6), (9.5, -1)):
        _tree(p, *_iso(0, 0, u, v))
    return p.sprite()


def worlds_fair() -> Sprite:
    """A world's fair: a vast exhibition palace of iron and glass under barrel-vaulted glass roofs,
    its taller middle crowned with flags."""
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
    _vault(p, main, 5, roof, wall, (0.15, 0.3, 0.7, 0.85), iron)
    centre = p.hall(0, 0, 7, 12, 13, wall, None)
    for k in range(1, 4):
        p.line(
            [_on_wall(centre[0], centre[1], k / 4, 0.5), _on_wall(centre[0], centre[1], k / 4, 13)],
            iron,
        )
    p.line([_on_wall(centre[0], centre[1], 0, 6), _on_wall(centre[0], centre[1], 1, 6)], iron)
    _vault(p, centre, 6, roof, wall, (0.25, 0.5, 0.75), iron)
    _portal(p, centre[0], centre[1], (0.3, 0.7), 13, 5)
    poles = (
        _on_wall(centre[0], centre[3], 0.5, -6),
        _on_wall(main[0], main[3], 0.5, -5),
        _on_wall(main[1], main[2], 0.5, -5),
    )
    for (x, y), colour in zip(poles, ((214, 58, 54), (60, 110, 200), (240, 200, 70)), strict=True):
        _flag(p, x, y, 7, colour)
    return p.sprite()


def film_studio() -> Sprite:
    """A film studio: big sound-stage sheds with their numbers painted on, a water tower on legs,
    and the studio gate with its sign."""
    p = Painter(50, 44, (24, 31))
    _plate(p, (150, 150, 146))
    stage, paint = (218, 206, 180), (60, 56, 60)
    for (u, v), length, depth, height, number in (((6.5, 6), 8, 8, 9, 2), ((-4, 5), 12, 9, 11, 1)):
        top = p.hall(*_iso(0, 0, u, v), length, depth, height, stage, None)
        p.gable(top, 3, (120, 116, 112), stage)
        x, y = _on_wall(top[0], top[1], 0.5, 2.5)
        p.rect(x - 3, y - 1, x + 3, y + 6, (240, 236, 226))
        _digit(p, x - 1, y, number, paint)
        _patch(p, top[1], top[2], (0.25, 0.75), (3, height), shade(stage, 0.55))
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
    return p.sprite()


def amusement_park() -> Sprite:
    """An amusement park: a big Ferris wheel with coloured gondolas, a roller coaster looping the
    loop, a carousel and striped stalls along its paths."""
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
    for (u, v), awning in (
        ((-8, -7), (214, 56, 56)),
        ((-3, -8.5), (60, 110, 200)),
        ((9, -6), (240, 200, 70)),
    ):
        stall = p.box(*_iso(0, 0, u, v), 2.5, 3, (236, 230, 214), None)
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
    return p.sprite()


def olympic_stadium() -> Sprite:
    """An Olympic stadium: an oval bowl of stands around a red running track and a green field,
    with floodlight masts round its rim and the flame burning in a cauldron on a tall stand."""
    p = Painter(52, 44, (25, 30))
    big, small, h = 20, 12, 7
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
    p.polygon(_oval(0, 0, small, 0, 0, 2 * math.pi, 33), (192, 82, 62))
    p.line(_oval(0, 0, small - 1.5, 0, 0, 2 * math.pi, 33), (236, 200, 190))
    p.polygon(_oval(0, 0, small - 3, 0, 0, 2 * math.pi, 33), (92, 156, 68))
    p.line([(0, -3.5), (0, 3.5)], (220, 240, 210))
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
    for angle in (0.7, math.pi - 0.7):
        mast(angle)
    cx, cy = -big - 1, -h
    p.rect(cx - 1, cy - 12, cx + 1, cy + 5, (150, 150, 158))
    p.ellipse(cx - 3, cy - 14, cx + 3, cy - 11, CROSS)
    p.blob(cx - 2.5, cy - 20, cx + 2.5, cy - 13, (246, 140, 40))
    p.blob(cx - 1, cy - 18, cx + 1, cy - 14, (255, 220, 110))
    return p.sprite()
