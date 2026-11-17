"""Works built as 3D models (railroads.ui.pixel.art.built) after their painted sprites: factories,
mills, forges, pits and kilns, laid out with the painted sprite's own coordinates."""

import math

from railroads.ui.pixel import train_parts as parts
from railroads.ui.pixel.art.built import (
    DOOR,
    along,
    below,
    blob,
    box_at,
    door,
    hall_at,
    heap,
    lathe,
    patch,
    roofed_hall,
    sawtooth,
    side_window,
    spot,
    stack,
    strut,
    tall,
    turned,
    upright,
    walls,
    wide,
    windows_at,
)
from railroads.ui.pixel.art.painter import BRICK, DOORWAY, METAL, RGB, TIMBER, WINDOW, shade
from railroads.ui.pixel.models import Model

GLAZING: RGB = (170, 200, 214)
MOLTEN: RGB = (250, 170, 60)
GLOW: RGB = (240, 150, 60)
CINDER: RGB = (64, 60, 60)
SOOT_GROUND: RGB = (70, 66, 64)
WATER: RGB = (92, 148, 196)
CLAY: RGB = (190, 108, 64)
SAND: RGB = (230, 206, 140)
LIMESTONE: RGB = (222, 218, 202)
GREY: RGB = (190, 188, 182)
TILE_BLUE: RGB = (70, 120, 170)


def _end_door(
    m: Model,
    u: tuple[float, float],
    v: tuple[float, float],
    span: tuple[float, float],
    z: tuple[float, float],
    color: RGB,
) -> None:
    """A door on the shaded end wall `span` of the way along it."""
    ends = sorted((v[1] - (v[1] - v[0]) * span[0], v[1] - (v[1] - v[0]) * span[1]))
    parts.front_plate(m, u[1] + 0.0006, (ends[0], ends[1]), z, color)


def factory() -> Model:
    """A small brick works with one chimney: a sawtooth roof glazed along its teeth, a row of
    arched windows and a big door in its end."""
    m = Model()
    u, v = box_at(0, 0, 7)
    high = roofed_hall(m, u, v, 6, BRICK, None)
    windows_at(m, u, v, 6, (0.15, 0.38, 0.62, 0.85), (2,), 2, WINDOW, shade(BRICK, 1.35))
    _end_door(m, u, v, (0.2, 0.5), (0.0, tall(3.5)), DOORWAY)
    sawtooth(m, u, v, high, 2, tall(2.5), METAL, GLAZING, BRICK)
    stack(m, 3, -6, -16, shade(BRICK, 0.8), puffs=1)
    return m


def textile_mill() -> Model:
    """A yellow-brick cotton mill of Łódź: a long block of storeys with rows of small arched
    windows under a flat roof, a weaving shed with a sawtooth roof and a tall smoking chimney."""
    m = Model()
    wall, roof = (206, 164, 92), (80, 76, 84)
    lintel = shade(BRICK, 1.2)
    stack(m, 11, 0, -32, shade(wall, 0.92), 2.4, puffs=2)
    u, v = hall_at(-4, -1, 14, 8)
    roofed_hall(m, u, v, 15, wall, None, flat=roof)
    windows_at(
        m, u, v, 15, (0.1, 0.3, 0.5, 0.7, 0.9), (3.5, 7, 10.5), 2, WINDOW, lintel, (0.3, 0.7)
    )
    windows_at(m, u, v, 15, (0.1, 0.3, 0.5), (14,), 1, WINDOW, lintel)
    su, sv = box_at(7, 4.5, 8)
    high = roofed_hall(m, su, sv, 5, wall, None)
    _end_door(m, su, sv, (0.4, 0.75), (0.0, tall(3)), (60, 44, 36))
    windows_at(m, su, sv, 5, (0.2, 0.5, 0.8), (2.5,), 1, WINDOW, lintel)
    sawtooth(m, su, sv, high, 3, tall(4), (104, 100, 108), (160, 196, 218), wall)
    return m


def tool_works() -> Model:
    """A dark forge hall on sooty ground with its door open on a glowing hearth, a forge stack
    glowing at its cap, a heap of coal and a quenching tub."""
    m = Model()
    blob(m, 0, 2, 14, 5, SOOT_GROUND, 0.0004)
    wall, roof = (92, 84, 82), (60, 60, 68)
    u, v = hall_at(-1, 0, 13, 9)
    roofed_hall(m, u, v, 7, wall, roof, 4)
    _end_door(m, u, v, (0.3, 0.7), below(7, 3, 4), GLOW)
    m.panel((along(u, 0.4), along(u, 0.65)), (0.0, tall(4.4)), v[1] + 0.0006, (48, 36, 30))
    high = stack(m, 0, -10, -17, (74, 58, 52))
    su, sv = spot(1, -10)
    m.box((su - 0.012, su + 0.012), (sv - 0.012, sv + 0.012), (high, high + 0.006), MOLTEN)
    heap(m, -9, 5, 3, 2.5, (48, 46, 50))
    tu, tv = spot(4, 3)
    upright(m, tu, tv, wide(2), (0.0, tall(1.4)), (110, 84, 58))
    upright(m, tu, tv, wide(1.7), (tall(1.4), tall(1.45)), (70, 96, 120))
    return m


def tannery() -> Model:
    """Low tanning sheds with doors and windows under tiled roofs, slatted drying lofts, vats of
    tan liquor sunk in the yard and a heap of hides."""
    m = Model()
    wall, roof = (184, 160, 124), (96, 82, 74)
    for x, y in ((-3, -3), (6, 2)):
        u, v = hall_at(x, y, 10, 5)
        high = roofed_hall(m, u, v, 4, wall, roof, 3)
        windows_at(m, u, v, 4, (0.85,), (1.2,), 1.2)
        side_window(m, u, v, (0.35, 0.65), 4, 1.2)
        door(m, (along(u, 0.4), along(u, 0.6)), tall(2.8), v[1], DOOR)
        with m.fine():
            for t in (0.1, 0.25):
                at = along(u, t)
                m.panel(
                    (at - 0.004, at + 0.004),
                    (high - tall(1.4), high - tall(0.3)),
                    v[1] + 0.0012,
                    shade(wall, 0.6),
                )
    for x in (-5, -2):
        vu, vv = spot(x, 5.5)
        upright(m, vu, vv, wide(1.6), (0.0, tall(1)), (110, 84, 58))
        upright(m, vu, vv, wide(1.4), (tall(1), tall(1.02)), (138, 116, 86))
    heap(m, -12, 7, 3, 2.5, (140, 92, 58))
    return m


def foundry() -> Model:
    """A soot-dark foundry hall with its door glowing, a tall iron cupola furnace beside it with
    molten iron at its mouth, a stack and moulds cooling on the ground."""
    m = Model()
    blob(m, 0, 2, 13, 5, SOOT_GROUND, 0.0004)
    wall, roof = (88, 80, 78), (58, 58, 64)
    u, v = hall_at(-3, 0, 12, 8)
    roofed_hall(m, u, v, 6, wall, roof, 3.5)
    m.panel((along(u, 0.45), along(u, 0.7)), (0.0, tall(4)), v[1] + 0.0006, GLOW)
    windows_at(m, u, v, 6, (0.15,), (2,), 1.6, GLOW)
    stack(m, 9, -4, -18, (60, 60, 64))
    turned(m, 13, 3, [(0, 2.2), (12, 2.2)], (66, 62, 64), MOLTEN)
    for x in (-1, 2, 5):
        mu, mv = spot(x, 4.4)
        m.box((mu - 0.01, mu + 0.01), (mv - 0.01, mv + 0.01), (0.0, tall(1)), (74, 76, 82))
    return m


def tile_works() -> Model:
    """A brick works shed with a bottle kiln and stacks of glazed blue stove tiles."""
    m = Model()
    wall, roof = (196, 176, 150), (110, 84, 70)
    u, v = hall_at(-4, 0, 10, 6)
    roofed_hall(m, u, v, 4, wall, roof, 3)
    door(m, (along(u, 0.4), along(u, 0.6)), tall(2.8), v[1], DOOR)
    windows_at(m, u, v, 4, (0.15, 0.85), (1,), 1.2, ends=(0.5,))
    turned(
        m, 10.5, -2, [(0, 2.5), (5, 2.5), (8, 1.2), (14, 1)], shade(BRICK, 0.85), shade(BRICK, 0.6)
    )
    for x in (-10, -7.5):
        tu, tv = spot(x + 1.2, 5)
        m.box((tu - 0.02, tu + 0.02), (tv - 0.015, tv + 0.015), (0.0, tall(3)), TILE_BLUE)
    return m


def _pit(
    m: Model, points: list[tuple[float, float]], color: RGB, steps: int, water: RGB | None
) -> None:
    """A pit dug in steps into the ground through the painted sprite's `points`, darker at each
    step down, with water in its bottom if given."""
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    for k in range(steps):
        t = 1 - k / steps * 0.75
        patch(
            m,
            [(cx + (x - cx) * t, cy + (y - cy) * t) for x, y in points],
            shade(color, 1 - 0.12 * k),
            0.0005 + k * 0.0003,
        )
    if water is not None:
        t = 0.2
        patch(
            m,
            [(cx + (x - cx) * t, cy + (y - cy) * t) for x, y in points],
            water,
            0.0005 + steps * 0.0003,
        )


def _shed(m: Model, x: float, y: float, half: float, wall: RGB, roof: RGB) -> None:
    u, v = box_at(x, y, half)
    roofed_hall(m, u, v, 4, wall, roof, 3)
    windows_at(m, u, v, 4, (0.27,), (1,), 1.2, ends=(0.3,))
    door(m, (along(u, 0.55), along(u, 0.85)), tall(2.7), v[1], (70, 56, 46))


def clay_pit() -> Model:
    """An open pit dug in steps into orange clay with a puddle and a ladder in it, a heap of clay
    and a shed."""
    m = Model()
    _pit(m, [(-12, 2), (-2, -3), (6, 1), (-4, 7)], CLAY, 3, (132, 120, 96))
    heap(m, -9, -3, 5, 6, shade(CLAY, 1.05))
    a, b = spot(-4, 2), spot(-2, 1)
    strut(m, (a[0], a[1], 0.0), (b[0], b[1], tall(4)), 0.003, (150, 112, 74))
    _shed(m, 10, -1, 4, (164, 150, 128), (96, 84, 76))
    return m


def sand_pit() -> Model:
    """Pale yellow heaps of sand beside a pit dug in steps with rippled water in its bottom, fed
    by a belt, with a sieve and a small heap by it."""
    m = Model()
    _pit(m, [(-12, 0), (-4, -4), (4, 0), (-4, 4)], SAND, 3, WATER)
    heap(m, 9, -1, 5, 8, SAND)
    heap(m, 6, 6, 4, 5, shade(SAND, 1.04))
    heap(m, -12.2, 4.6, 2.4, 1.6, shade(SAND, 1.06))
    a, b = spot(-1.5, 0), spot(7, -3)
    strut(m, (a[0], a[1], tall(1)), (b[0], b[1], tall(7)), 0.006, (70, 70, 78))
    su, sv = spot(-8, 6)
    m.box((su - 0.015, su + 0.015), (sv - 0.008, sv + 0.008), (tall(1.5), tall(2)), (150, 112, 74))
    for du in (-0.012, 0.012):
        m.box(
            (su + du - 0.002, su + du + 0.002),
            (sv - 0.002, sv + 0.002),
            (0.0, tall(1.5)),
            (150, 112, 74),
        )
    return m


def limestone_quarry() -> Model:
    """A pale limestone face cut in steps into a grassy hill, its courses split into blocks, with
    blocks cut out on the floor, a cart of them and a crane."""
    m = Model()
    for k, (x, y, h) in enumerate(((-2, -5, 9), (0, -3, 6), (2, -1, 3))):
        u, v = hall_at(x, y, 16, 5)
        walls(m, u, v, tall(h), LIMESTONE)
        m.box(
            u, v, (tall(h), tall(h) + 0.004), (122, 156, 84) if k == 0 else shade(LIMESTONE, 1.08)
        )
        with m.fine():
            for j in range(1, int(h / 1.5)):
                m.panel(
                    u,
                    (tall(j * 1.5) - 0.001, tall(j * 1.5) + 0.001),
                    v[1] + 0.0008,
                    shade(LIMESTONE, 0.8),
                )
    for x, y in ((6, 4), (9, 2.5), (3, 5.5)):
        bu, bv = box_at(x, y, 2.5)
        m.box(bu, bv, (0.0, tall(2.5)), LIMESTONE)
    cu, cv = hall_at(-8, 4, 5, 2.4)
    m.box(cu, cv, (tall(0.6), tall(1.6)), TIMBER)
    m.box(*box_at(-8, 3, 1.6), (tall(1.6), tall(3.4)), LIMESTONE)
    a = spot(8, -2)
    strut(m, (a[0], a[1], 0.0), (a[0], a[1], tall(12)), 0.006, shade(TIMBER, 1.2))
    b = spot(2, -2)
    strut(m, (a[0], a[1], tall(11.5)), (b[0], b[1], tall(9)), 0.004, shade(TIMBER, 1.2))
    return m


def glassworks() -> Model:
    """A tall banded brick glass cone sooted at its top, its furnace mouth glowing under an arch
    at its foot, a shed for the glass, crates of panes and a heap of cullet."""
    m = Model()
    cone_color = (162, 88, 64)
    turned(m, -2, 0, [(0, 9), (7, 7.5), (20, 4), (28, 3)], cone_color, (54, 40, 36))
    cu, cv = spot(-2, 0)
    with m.fine():
        for h in (7, 14):
            upright(
                m,
                cu,
                cv,
                wide(9 - h * 0.3) + 0.002,
                (tall(h), tall(h) + 0.006),
                shade(cone_color, 1.3),
            )
    m.box((cu + wide(8.6), cu + wide(9.2)), (cv - 0.02, cv + 0.02), (0.0, tall(5)), (240, 132, 40))
    su, sv = box_at(10, 2, 4)
    roofed_hall(m, su, sv, 4, (196, 180, 156), (96, 84, 80), 2)
    windows_at(m, su, sv, 4, (0.5,), (1.2,), 1.2)
    _end_door(m, su, sv, (0.3, 0.7), (0.0, tall(2.8)), DOORWAY)
    for x, y in ((6, 6), (9, 5.4)):
        pu, pv = box_at(x, y, 1.4)
        m.box(pu, pv, (0.0, tall(2)), (150, 112, 74))
        m.box(pu, pv, (tall(2), tall(2.1)), (176, 222, 234))
    heap(m, -12, 6, 2.5, 2, (170, 210, 196))
    return m


def brickworks() -> Model:
    """A round Hoffmann kiln ringed with arched firing doors, around its tall banded smoking
    chimney, with stacks of new bricks and a heap of clay."""
    m = Model()
    kiln = (152, 98, 76)
    x, y, r = 0, 0, 10
    turned(m, x, y, [(0, r), (6, r)], kiln)
    ku, kv = spot(x, y)
    lathe_cap = [(tall(6), wide(r + 1)), (tall(9), wide(r - 4))]
    lathe(m, ku, kv, lathe_cap, (96, 86, 86), (118, 106, 104))
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        du, dv = wide(r) * math.cos(a), wide(r) * math.sin(a)
        m.box(
            (ku + du - 0.008, ku + du + 0.008),
            (kv + dv - 0.008, kv + dv + 0.008),
            (0.0, tall(4)),
            (52, 36, 32),
        )
    turned(m, x, y - 4, [(0, 2), (35 * 0.6 - 4, 1.4)], shade(BRICK, 0.9), (60, 50, 46))
    for bx, by in ((10, 5), (13, 3)):
        bu, bv = hall_at(bx, by, 4, 3)
        m.box(bu, bv, (0.0, tall(3)), BRICK)
    heap(m, -12, 7, 3, 2.5, (176, 122, 86))
    return m


def machine_works() -> Model:
    """A big brick engine hall over a dark plinth with tall arched windows, a big door, a gantry
    crane over a yard of gravel with machine parts set out on it."""
    m = Model()
    lintel = shade(BRICK, 1.3)
    blob(m, 10, 2, 7, 3, (140, 134, 120), 0.0005)
    u, v = hall_at(-2, -2, 16, 9)
    roofed_hall(m, u, v, 10, BRICK, (82, 84, 96), 5)
    windows_at(m, u, v, 10, (0.12, 0.3, 0.48, 0.66), (2.5,), 5, WINDOW, lintel)
    windows_at(m, u, v, 10, (0.12, 0.3, 0.48, 0.66), (8,), 1, shade(BRICK, 1.15))
    m.panel((along(u, 0.8), along(u, 0.94)), (0.0, tall(5)), v[1] + 0.0006, (58, 48, 44))
    crane = (214, 168, 40)
    legs = [spot(9, -2), spot(16, 1.5), spot(5, 1), spot(12, 4.5)]
    for lu, lv in legs:
        m.box((lu - 0.004, lu + 0.004), (lv - 0.004, lv + 0.004), (0.0, tall(10)), crane)
    for a, b in ((legs[0], legs[1]), (legs[2], legs[3])):
        strut(m, (a[0], a[1], tall(10)), (b[0], b[1], tall(10)), 0.008, crane)
    for x, y in ((12, 2), (14, 3)):
        mu, mv = box_at(x, y, 1.5)
        m.box(mu, mv, (0.0, tall(1.5)), (88, 90, 98))
    return m


def steel_mill() -> Model:
    """A dark mill on blackened ground with glowing windows, a blast furnace glowing at its foot
    and a tall stack, a heap of ore and ingots stacked by it."""
    m = Model()
    blob(m, 0, 2, 15, 6, CINDER, 0.0004)
    iron = (82, 80, 84)
    glow = (186, 104, 48)
    u, v = box_at(0, 0, 11)
    roofed_hall(m, u, v, 9, iron, None, flat=(60, 58, 62))
    windows_at(m, u, v, 9, (0.52, 0.66, 0.8, 0.94), (2.5,), 3, glow, ends=(0.3, 0.55))
    turned(m, 6.5, -6, [(0, 2.6), (13, 2.6), (17, 1.8), (20, 1.4)], (96, 88, 86), (40, 36, 36))
    heap(m, 13, 5, 3.5, 3, (150, 76, 54))
    gu, gv = spot(-7.5, -2)
    m.box((gu - 0.03, gu + 0.03), (gv - 0.01, gv + 0.01), (0.0, tall(6)), (238, 132, 40))
    stack(m, -3, -9, -30, shade(BRICK, 0.7), puffs=2)
    iu, iv = hall_at(-3, 7, 4, 2)
    m.box(iu, iv, (0.0, tall(1.2)), (128, 128, 136))
    return m


def cement_works() -> Model:
    """Dusty grey banded silos joined to a windowed preheater tower by a conveyor, with a long
    rotary kiln on piers and a dusty yard."""
    m = Model()
    blob(m, 0, 2, 16, 6, (176, 174, 164), 0.0004)
    tu, tv = box_at(9, -6, 3.5)
    walls(m, tu, tv, tall(22), shade(GREY, 0.9))
    m.box(tu, tv, (tall(22), tall(22) + 0.004), shade(GREY, 0.75))
    windows_at(m, tu, tv, 22, (0.5,), (4, 10, 16), 2)
    for sx, sy in ((-6, -3), (-2, -5), (-10, -1)):
        turned(m, sx, sy, [(0, 2.5), (16, 2.5)], GREY, shade(GREY, 1.1))
        su, sv = spot(sx, sy)
        with m.fine():
            for h in (5, 10):
                upright(m, su, sv, wide(2.5) + 0.0015, (tall(h), tall(h) + 0.004), shade(GREY, 0.8))
    a, b = spot(-2, -5), spot(9, -6)
    strut(m, (a[0], a[1], tall(16)), (b[0], b[1], tall(18)), 0.008, (134, 128, 120))
    ka, kb = spot(-8, 6), spot(8, 3)
    strut(m, (ka[0], ka[1], tall(3)), (kb[0], kb[1], tall(4.5)), 0.04, (134, 128, 120))
    for t in (0.2, 0.5, 0.8):
        pu, pv = ka[0] + (kb[0] - ka[0]) * t, ka[1] + (kb[1] - ka[1]) * t
        m.box(
            (pu - 0.008, pu + 0.008),
            (pv - 0.008, pv + 0.008),
            (0.0, tall(3) + (tall(1.5)) * t),
            shade(GREY, 0.8),
        )
    return m
