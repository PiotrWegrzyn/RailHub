"""More works built as 3D models (railroads.ui.pixel.art.built) after their painted sprites:
workshops, chemical and modern plants, a paper mill, a refinery, a car works and coke ovens,
laid out with the painted sprite's own coordinates."""

import math

from railroads.ui.pixel import train_parts as parts
from railroads.ui.pixel.art.built import (
    DOOR,
    along,
    barrel,
    blob,
    box_at,
    cornice,
    dome,
    door,
    hall_at,
    heap,
    patch,
    roofed_hall,
    sawtooth,
    spot,
    stack,
    strut,
    tall,
    turned,
    upright,
    wide,
    windows_at,
)
from railroads.ui.pixel.art.painter import BRICK, DOORWAY, GLASS, METAL, RGB, TIMBER, WINDOW, shade
from railroads.ui.pixel.models import Model

PAVING: RGB = (176, 170, 158)
PAD: RGB = (170, 170, 164)


def _end_door(
    m: Model,
    u: tuple[float, float],
    v: tuple[float, float],
    span: tuple[float, float],
    z: tuple[float, float],
    color: RGB,
) -> None:
    ends = sorted((v[1] - (v[1] - v[0]) * span[0], v[1] - (v[1] - v[0]) * span[1]))
    parts.front_plate(m, u[1] + 0.0006, (ends[0], ends[1]), z, color)


def shoe_factory() -> Model:
    """A narrow workshop of four storeys with cornices between its floors, windows with sills, a
    door, a tiled roof with a chimney and a boot hung out as its sign."""
    m = Model()
    blob(m, -3, 4, 6, 2.4, PAVING, 0.0005)
    wall, roof = (204, 170, 116), (124, 66, 52)
    u, v = box_at(1, 0, 5)
    high = roofed_hall(m, u, v, 16, wall, roof, 4)
    for floor in range(4):
        windows_at(m, u, v, 16, (0.25, 0.75), (1.2 + 4 * floor,), 1.6, ends=(0.3, 0.7))
        if floor:
            cornice(m, u, v, tall(16 - 4 * floor + 0.5), shade(wall, 1.15))
    door(m, (along(u, 0.55), along(u, 0.8)), tall(3.5), v[1], (84, 60, 46))
    stack(m, 2, -16, -24, BRICK, 1.4)
    bu = along(u, 0.1)
    m.box(
        (bu - 0.002, bu + 0.002),
        (v[1], v[1] + 0.03),
        (high - tall(2), high - tall(1.8)),
        (60, 56, 54),
    )
    m.box(
        (bu - 0.006, bu + 0.006),
        (v[1] + 0.02, v[1] + 0.034),
        (high - tall(5), high - tall(2)),
        (140, 54, 40),
    )
    return m


def plastics_works() -> Model:
    """White banded tanks with red and blue tops and a ladder on a concrete pad, a tall white
    vent stack and a low white works with blue windows."""
    m = Model()
    patch(m, [(-14, 2), (2, -6), (16, 2), (0, 9)], PAD)
    white, red, blue = (232, 234, 236), (214, 58, 54), (64, 108, 196)
    vu, vv = spot(-1, -4)
    m.box((vu - 0.012, vu + 0.012), (vv - 0.012, vv + 0.012), (0.0, tall(25)), white)
    m.box((vu - 0.016, vu + 0.016), (vv - 0.016, vv + 0.016), (tall(25), tall(26)), (70, 70, 76))
    turned(m, 6, -3, [(0, 3), (12, 3)], white, red)
    turned(m, 11, 2, [(0, 2.5), (8, 2.5)], white, blue)
    u, v = box_at(-5, 2, 7)
    roofed_hall(m, u, v, 5, white, None, flat=blue)
    windows_at(m, u, v, 5, (0.25, 0.5, 0.75), (2.5,), 1, shade(blue, 0.8))
    _end_door(m, u, v, (0.3, 0.65), (0.0, tall(3)), (70, 74, 84))
    return m


def appliance_factory() -> Model:
    """A long modern factory on a grey plinth on a concrete apron, with a big roller door, rows of
    windows under a blue stripe, a vent on its roof and white boxed appliances stacked out front."""
    m = Model()
    patch(m, [(-16, 2), (2, -8), (20, 1), (2, 11)], (180, 180, 174))
    wall, stripe = (216, 220, 224), (64, 116, 196)
    u, v = hall_at(1, -3, 18, 8)
    high = roofed_hall(m, u, v, 7, wall, None, flat=(150, 160, 174))
    cornice(m, u, v, high - tall(1), stripe)
    windows_at(m, u, v, 7, (0.1, 0.2, 0.3, 0.4), (3,), 2, (110, 160, 200), ends=(0.3, 0.6))
    m.panel((along(u, 0.55), along(u, 0.85)), (0.0, tall(4.5)), v[1] + 0.0006, (170, 174, 180))
    with m.fine():
        for k in range(1, 8):
            z = tall(4.5 * k / 8)
            m.panel(
                (along(u, 0.55), along(u, 0.85)),
                (z - 0.0008, z + 0.0008),
                v[1] + 0.0012,
                (140, 144, 150),
            )
    ru, rv = box_at(4, -6, 1.5)
    m.box(ru, rv, (high, high + tall(1.5)), (190, 196, 204))
    for x, y, h in ((10, 5, 3), (13, 3.5, 2), (8, 7, 2)):
        cu, cv = box_at(x, y, 2.5)
        m.box(cu, cv, (0.0, tall(h)), (244, 246, 248))
    return m


def paper_mill() -> Model:
    """A riverside paper mill on a dark plinth turning a waterwheel in its stream, with arched
    windows, a smoking chimney and rolls of paper stacked by its door."""
    m = Model()
    water = (84, 140, 196)
    patch(m, [(-18, 2), (-10, -2), (0, 9), (-8, 12)], (96, 88, 62))
    patch(m, [(-16, 3), (-10, 0), (-1, 9), (-7, 11)], water, 0.001)
    lintel = shade(BRICK, 1.25)
    stack(m, 8, -4, -30, BRICK, 2, puffs=2)
    wall, roof = (192, 168, 136), (100, 86, 80)
    u, v = hall_at(-1, -3, 12, 8)
    roofed_hall(m, u, v, 9, wall, roof, 4)
    windows_at(m, u, v, 9, (0.55, 0.75, 0.92), (2, 5.5), 2, WINDOW, lintel, (0.3, 0.75))
    _end_door(m, u, v, (0.4, 0.65), (0.0, tall(4)), (70, 56, 48))
    wu = along(u, 0.15)
    m.cylinder_across(
        wu,
        wide(4.5),
        wide(4.5) * 0.9,
        (v[1] + 0.004, v[1] + 0.03),
        (98, 72, 50),
        shade((98, 72, 50), 1.3),
    )
    for x, y in ((9, 6), (11, 5), (10, 7.5)):
        ru, rv = spot(x, y)
        upright(m, ru, rv, wide(1.5), (0.0, tall(3)), (240, 240, 234))
        upright(m, ru, rv, wide(0.6), (tall(3), tall(3.05)), (196, 160, 112))
    return m


def printing_works() -> Model:
    """A town printing house with tall arched windows over a dark plinth, a door under a lintel
    and a great printed sheet on its roof for a sign."""
    m = Model()
    blob(m, -3, 5, 7, 2.6, PAVING, 0.0005)
    wall = (220, 204, 172)
    u, v = box_at(0, 0, 8)
    high = roofed_hall(m, u, v, 12, wall, None, flat=(110, 106, 116))
    cornice(m, u, v, tall(5), shade(wall, 0.85))
    windows_at(
        m, u, v, 12, (0.2, 0.45, 0.7, 0.9), (2.5,), 4, WINDOW, shade(wall, 0.75), (0.25, 0.5, 0.75)
    )
    windows_at(m, u, v, 12, (0.2, 0.45, 0.9), (8,), 3, ends=(0.25, 0.5, 0.75))
    door(m, (along(u, 0.58), along(u, 0.8)), tall(3.5), v[1], (70, 56, 48))
    su = (u[0] + u[1]) / 2
    sv = (v[0] + v[1]) / 2
    for du in (-0.04, 0.04):
        m.box(
            (su + du - 0.003, su + du + 0.003),
            (sv - 0.003, sv + 0.003),
            (high, high + tall(3)),
            (60, 56, 54),
        )
    m.box(
        (su - 0.07, su + 0.07),
        (sv - 0.002, sv + 0.002),
        (high + tall(3), high + tall(8)),
        (242, 240, 230),
    )
    with m.fine():
        m.box(
            (su - 0.05, su - 0.02),
            (sv + 0.002, sv + 0.003),
            (high + tall(5), high + tall(6.4)),
            (150, 150, 160),
        )
    return m


def porcelain_works() -> Model:
    """Two small brick bottle ovens bound with iron hoops on sooty ground, a whitewashed works
    with windows under a tiled roof and racks of white and blue china drying."""
    m = Model()
    blob(m, -7, 1, 9, 4, (90, 84, 80), 0.0004)
    oven = (172, 116, 86)
    bottle = [(0, 4), (4, 4.5), (8, 3), (10, 1.6), (14, 1.4)]
    for x, y in ((-9, -2), (-3, 1)):
        turned(m, x, y, bottle, oven, (60, 46, 42))
        bu, bv = spot(x, y)
        with m.fine():
            for h in (2, 6):
                upright(m, bu, bv, wide(4.4) + 0.002, (tall(h), tall(h) + 0.004), (70, 70, 76))
    wall, roof = (230, 218, 192), (122, 88, 70)
    u, v = hall_at(5, -2, 10, 6)
    roofed_hall(m, u, v, 5, wall, roof, 3)
    windows_at(m, u, v, 5, (0.2, 0.45, 0.85), (1.4,), 1.6)
    _end_door(m, u, v, (0.35, 0.7), (0.0, tall(3.4)), DOOR)
    for k, (x, y) in enumerate(((6, 6), (10, 4))):
        ru, rv = hall_at(x, y, 4, 1)
        m.box(ru, rv, (0.0, tall(2.6)), (120, 88, 58))
        with m.fine():
            for j in range(3):
                at = ru[0] + (ru[1] - ru[0]) * (j + 0.5) / 3
                upright(
                    m,
                    at,
                    rv[1],
                    0.006,
                    (tall(2.6), tall(3.4)),
                    (246, 246, 250) if (j + k) % 2 else (54, 86, 176),
                )
    heap(m, -13, 6, 2, 1.5, (220, 220, 226))
    return m


def refinery() -> Model:
    """An oil refinery filling its cell: tall silver distillation columns ringed with walkways,
    white storage tanks, a pipe rack between them and a flare stack."""
    m = Model()
    patch(m, [(-22, 0), (0, -11), (22, 0), (0, 11)], (168, 164, 156))
    metal, white = (176, 178, 184), (224, 226, 228)
    for x, y, r, h in ((-6, -3, 2.2, 26), (-1, -5, 1.8, 30), (4, -3, 2, 22)):
        turned(m, x, y, [(0, r), (h, r), (h + 1.5, r * 0.6)], metal)
        cu, cv = spot(x, y)
        with m.fine():
            for k in range(1, 4):
                upright(
                    m,
                    cu,
                    cv,
                    wide(r) + 0.004,
                    (tall(h * k / 4), tall(h * k / 4) + 0.003),
                    (96, 96, 104),
                )
    for x, y in ((-12, 4), (-6, 6), (10, 3), (14, 6)):
        turned(m, x, y, [(0, 4.5), (4, 4.5)], white, shade(white, 1.08))
    a, b = spot(-14, -1), spot(14, -1)
    for dz in (tall(5), tall(6)):
        strut(m, (a[0], a[1], dz), (b[0], b[1], dz), 0.005, (96, 96, 104))
    fu, fv = spot(16, -4)
    m.box((fu - 0.005, fu + 0.005), (fv - 0.005, fv + 0.005), (0.0, tall(28)), (150, 150, 156))
    m.box((fu - 0.007, fu + 0.007), (fv - 0.007, fv + 0.007), (tall(28), tall(30)), (250, 160, 60))
    return m


def chemical_works() -> Model:
    """A ball tank on legs, banded pale tanks with green tops and a ladder, a vent pipe and a
    pipe rack, a small control house on a concrete pad."""
    m = Model()
    patch(m, [(-15, 2), (0, -6), (15, 2), (0, 9)], (156, 156, 148))
    body, green = (208, 214, 198), (104, 176, 84)
    pu, pv = spot(10.5, -2)
    m.box((pu - 0.004, pu + 0.004), (pv - 0.004, pv + 0.004), (0.0, tall(24)), METAL)
    turned(m, 4, -4, [(0, 3), (13, 3)], body, green)
    bu, bv = spot(-8.5, -3)
    radius = wide(4.5)
    for a in range(4):
        ang = math.pi / 4 + a * math.pi / 2
        lu, lv = bu + radius * 0.7 * math.cos(ang), bv + radius * 0.7 * math.sin(ang)
        m.box((lu - 0.003, lu + 0.003), (lv - 0.003, lv + 0.003), (0.0, tall(6)), METAL)
    upright(m, bu, bv, radius, (tall(6) + radius * 0.2, tall(6) + radius * 0.8), body)
    dome(m, bu, bv, radius, (tall(6) + radius * 0.8, tall(6) + radius * 1.6), body)
    turned(m, 8, 1, [(0, 2.5), (9, 2.5)], body, green)
    u, v = box_at(-3, 4, 5)
    roofed_hall(m, u, v, 4, (196, 200, 186), None, flat=(96, 116, 84))
    door(m, (along(u, 0.6), along(u, 0.85)), tall(2.8), v[1], DOORWAY)
    return m


def electronics_plant() -> Model:
    """A modern box of white panels and bands of cyan glass on a paved apron, with a satellite
    dish and vents on its roof and flags out front."""
    m = Model()
    patch(m, [(-16, 2), (1, -8), (18, 1), (1, 11)], (186, 188, 184))
    wall, glass, accent = (226, 232, 236), (84, 176, 206), (80, 216, 222)
    u, v = hall_at(0, 0, 15, 9)
    high = roofed_hall(m, u, v, 8, wall, None, flat=(172, 180, 190))
    for drop in (2, 5):
        z0, z1 = tall(8 - drop - 2), tall(8 - drop)
        m.panel((along(u, 0.05), along(u, 0.95)), (z0, z1), v[1] + 0.0006, glass)
        parts.front_plate(m, u[1] + 0.0006, (v[0] + 0.01, v[1] - 0.01), (z0, z1), shade(glass, 0.8))
    m.panel((along(u, 0.4), along(u, 0.6)), (0.0, tall(2)), v[1] + 0.0009, accent)
    du, dv = spot(-2, -2)
    upright(m, du, dv, 0.004, (high, high + tall(2)), (190, 198, 206))
    dome(m, du, dv, wide(3.5), (high + tall(2), high + tall(3.4)), (236, 238, 240))
    for x, y in ((8, -9), (11, -7)):
        ru, rv = box_at(x, y, 1)
        m.box(ru, rv, (high, high + tall(2)), (190, 198, 206))
    for x, color in ((-12, accent), (-9, (64, 108, 196))):
        fu, fv = spot(x, 5)
        m.box((fu - 0.002, fu + 0.002), (fv - 0.002, fv + 0.002), (0.0, tall(9)), (200, 200, 206))
        m.box((fu - 0.022, fu - 0.002), (fv - 0.001, fv + 0.001), (tall(7), tall(9)), color)
    return m


def car_factory() -> Model:
    """A car works filling its cell: a long assembly hall on a dark plinth under an arched roof,
    a press shop with a sawtooth roof and new cars parked in rows on the lot."""
    m = Model()
    patch(m, [(-24, 0), (0, -12), (24, 0), (0, 12)], (166, 166, 162))
    wall, press_wall = (156, 160, 170), (176, 172, 168)
    pu, pv = hall_at(9, -5, 14, 5)
    high = roofed_hall(m, pu, pv, 6, press_wall, None)
    windows_at(m, pu, pv, 6, (), (2,), 2, ends=(0.3, 0.7))
    sawtooth(m, pu, pv, high, 4, tall(3), (104, 110, 126), (160, 196, 218), press_wall)
    u, v = hall_at(-2, 0, 20, 7)
    high = roofed_hall(m, u, v, 7, wall, None)
    m.vault(u, (v[1] - v[0]) / 2 + 0.006, (high, high + tall(5)), (98, 108, 130))
    windows_at(m, u, v, 7, (0.1, 0.25, 0.4, 0.55, 0.7, 0.85), (2,), 2, shade(GLASS, 0.8))
    colors = [(196, 58, 54), (60, 90, 170), (230, 230, 226), (40, 40, 44)]
    for row in range(2):
        for k in range(5):
            cu, cv = spot(-14 + 4 * k + 2 * row, 6 + 2 * row - k * 0.6)
            m.box(
                (cu - 0.02, cu + 0.02),
                (cv - 0.01, cv + 0.01),
                (0.004, tall(1.2)),
                colors[(k + row) % 4],
            )
    return m


def barrel_maker() -> Model:
    """A cooper's shed with a wide door, a window and a smoking stove pipe, a pile of staves and
    finished barrels by its door."""
    m = Model()
    roof = (118, 84, 54)
    u, v = box_at(-3, 0, 7)
    roofed_hall(m, u, v, 5, TIMBER, roof, 3)
    windows_at(m, u, v, 5, (0.2,), (1.6,), 1.4, ends=(0.5,))
    door(m, (along(u, 0.46), along(u, 0.84)), tall(3.2), v[1], DOOR)
    stack(m, -6, -6, -12, (70, 70, 76), 1, puffs=1)
    su, sv = hall_at(-9.5, 4, 5, 2)
    m.box(su, sv, (0.0, tall(1.6)), (206, 168, 116))
    for x, y in ((7, 3), (9, 1.5), (8.5, 4.5)):
        bu, bv = spot(x, y)
        barrel(m, bu, bv, wide(1.3), tall(2.6), (150, 100, 56))
    return m


def coke_plant() -> Model:
    """A low battery of brick coke ovens with their doors in a row under a flat top, a tall
    chimney smoking above it, a quench tower and a heap of grey coke beside."""
    m = Model()
    blob(m, 0, 3, 15, 5, (70, 66, 64), 0.0004)
    u, v = hall_at(-1, 1, 18, 7)
    roofed_hall(m, u, v, 4, BRICK, None, flat=(70, 66, 64))
    with m.fine():
        for k in range(8):
            at = along(u, (k + 0.5) / 8)
            m.panel((at - 0.006, at + 0.006), (0.0, tall(3)), v[1] + 0.0008, (52, 40, 36))
    stack(m, 9, -2, -18, shade(BRICK, 0.85), 1.6, puffs=2)
    qu, qv = box_at(-11, -4, 2)
    m.box(qu, qv, (0.0, tall(12)), (120, 116, 110))
    heap(m, 12, 6, 3, 3, (70, 70, 76))
    return m
