"""A winery, shops, a fishery, a hop garden, a beekeeper's, a potter's, a sugar refinery, a
furniture maker's, a flax farm and a sugar cane field built as 3D models
(railroads.ui.pixel.art.built) after their painted sprites, laid out with the painted sprite's
own coordinates."""

import itertools

from railroads.ui.pixel import train_parts as parts
from railroads.ui.pixel.art.built import (
    DOOR,
    along,
    arch_window,
    barrel,
    below,
    blob,
    box_at,
    chimney,
    cone,
    cornice,
    dome,
    door,
    flat_roof,
    footing,
    gable_roof,
    hall_at,
    patch,
    plinth,
    side_window,
    smoke,
    spot,
    strut,
    tall,
    upright,
    walls,
    wide,
    window,
)
from railroads.ui.pixel.art.painter import (
    BARREL,
    BRICK,
    GLASS,
    HIVE,
    RGB,
    STONE,
    WHITEWASH,
    WINDOW,
    shade,
)
from railroads.ui.pixel.models import Model

VINE: RGB = (78, 130, 50)
GRAPE: RGB = (112, 50, 124)
POST: RGB = (110, 84, 58)
SOIL: RGB = (118, 96, 66)
SHINGLE: RGB = (198, 184, 146)
WATER: RGB = (92, 148, 196)
HULL: RGB = (150, 62, 48)
NET: RGB = (120, 170, 196)
BINE: RGB = (88, 152, 58)
HOP: RGB = (180, 214, 112)
OAST: RGB = (176, 92, 64)
KILN: RGB = (110, 76, 64)
COWL: RGB = (238, 236, 228)
HONEY: RGB = (214, 150, 40)
CLAY: RGB = (176, 104, 70)
BEET: RGB = (200, 170, 128)
SUGAR_WHITE: RGB = (240, 240, 236)
SUGAR_BLUE: RGB = (70, 100, 180)
BOARD: RGB = (214, 178, 124)
SAWDUST: RGB = (222, 196, 140)
STRAW: RGB = (214, 184, 104)
FLAX: RGB = (88, 128, 84)
FLAX_FLOWER: RGB = (96, 124, 214)
CANE: RGB = (118, 176, 82)
CANE_GROUND: RGB = (132, 112, 72)


def _cottage(
    m: Model, x: float, y: float, half: float, height: float, wall: RGB, roof: RGB, rise: float
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """A small house with a window, a door and an end window under a gable roof."""
    u, v = box_at(x, y, half)
    high = tall(height)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    plinth(m, u, v, tall(0.8), shade(wall, 0.75))
    window(m, (along(u, 0.14), along(u, 0.34)), below(height, 1, 1.2), v[1], WINDOW)
    door(m, (along(u, 0.56), along(u, 0.76)), tall(height - 1.5), v[1], DOOR)
    side_window(m, u, v, (0.4, 0.66), height, 1)
    gable_roof(m, u, v, (high, high + tall(rise)), roof, wall)
    return u, v, high


def winery() -> Model:
    """A long whitewashed winery under a red roof with a row of windows, a stone cellar dug into
    the hill with grass on its roof, barrels by its door, and vines trained on trellises."""
    m = Model()
    wall, roof = (238, 224, 192), (178, 86, 58)
    for k in range(3):
        x, y = -15 + 3 * k, -1 + 1.5 * k
        patch(m, [(x - 1, y), (x + 9, y + 4.5), (x + 9, y + 6), (x - 1, y + 1.5)], SOIL)
        for t in range(4):
            px, py = x + 2.5 * t, y + 1.25 * t + 1
            pu, pv = spot(px, py)
            m.box((pu - 0.002, pu + 0.002), (pv - 0.002, pv + 0.002), (0.0, tall(4)), POST)
            dome(m, pu, pv, wide(1.2), (tall(2.2), tall(4.4)), VINE)
            with m.fine():
                dome(m, pu + 0.004, pv + 0.006, wide(0.5), (tall(2.4), tall(3.2)), GRAPE)
    u, v = hall_at(0, -2, 12, 7)
    high = tall(5)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    plinth(m, u, v, tall(0.8), shade(wall, 0.8))
    for t in (0.12, 0.28, 0.44, 0.84):
        window(m, (along(u, t - 0.03), along(u, t + 0.03)), below(5, 1.6, 1.6), v[1], WINDOW)
    door(m, (along(u, 0.56), along(u, 0.68)), tall(3), v[1], (110, 74, 50))
    side_window(m, u, v, (0.4, 0.6), 5, 1.6)
    gable_roof(m, u, v, (high, high + tall(3)), roof, wall)
    cu, cv = box_at(10, 3, 4)
    walls(m, cu, cv, tall(3), (176, 168, 150))
    m.box(cu, cv, (tall(3), tall(3) + 0.006), (104, 140, 70))
    m.panel((along(cu, 0.3), along(cu, 0.75)), (0.0, tall(2)), cv[1] + 0.0006, (42, 34, 30))
    for x, y in ((6, 6), (8.5, 5)):
        bu, bv = spot(x, y)
        barrel(m, bu, bv, wide(1.3), tall(2.6), BARREL)
    blob(m, -1.2, 9.6, 1.3, 0.6, (110, 52, 76), 0.001)
    return m


def shop(wall: RGB, awning: RGB, goods: RGB) -> Model:
    """A smart city shop: a striped awning over a big window full of its wares and a door, a
    floor of windows with sills above, a cornice and a chimney."""
    m = Model()
    height = 12
    u, v = box_at(0, 0, 8)
    high = tall(height)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    flat_roof(m, u, v, high, shade(wall, 0.7), wall)
    m.panel((along(u, 0.08), along(u, 0.6)), (tall(1), tall(5)), v[1] + 0.0006, GLASS)
    with m.fine():
        for t in (0.2, 0.35, 0.5):
            at = along(u, t)
            m.box((at - 0.008, at + 0.008), (v[1] - 0.004, v[1]), (tall(1), tall(2.4)), goods)
    door(m, (along(u, 0.7), along(u, 0.86)), tall(4.4), v[1], (60, 50, 46))
    step = (along(u, 0.04), along(u, 0.64))
    m.box(step, (v[1], v[1] + 0.03), (tall(5.4), tall(5.8)), awning)
    with m.fine():
        for k in range(6):
            a = step[0] + (step[1] - step[0]) * k / 6
            if k % 2:
                m.box(
                    (a, a + (step[1] - step[0]) / 6),
                    (v[1] + 0.001, v[1] + 0.031),
                    (tall(5.8), tall(5.82) + 0.001),
                    (240, 236, 228),
                )
    for t in (0.2, 0.5, 0.8):
        window(m, (along(u, t - 0.08), along(u, t + 0.08)), (tall(7.4), tall(9.8)), v[1], WINDOW)
    for t in (0.3, 0.7):
        ends = sorted((v[1] - (v[1] - v[0]) * (t - 0.12), v[1] - (v[1] - v[0]) * (t + 0.12)))
        parts.front_plate(
            m, u[1] + 0.0006, (ends[0], ends[1]), (tall(7.4), tall(9.8)), shade(WINDOW, 0.75)
        )
        parts.front_plate(
            m, u[1] + 0.0006, (ends[0], ends[1]), (tall(2), tall(4.6)), shade(WINDOW, 0.75)
        )
    cornice(m, u, v, high - 0.006, shade(wall, 1.12))
    chimney(m, along(u, 0.25), (v[0] + v[1]) / 2, (high, high + tall(3)), shade(wall, 0.6))
    return m


def fishery() -> Model:
    """A boarded fisherman's cottage under a blue-grey roof on a shingle shore, nets hung to dry
    on poles, a rowing boat drawn up on the beach and a basket of fish."""
    m = Model()
    patch(m, [(-2, 7), (7, 1.5), (15, -1.5), (15, 7)], SHINGLE)
    patch(m, [(8, 7), (15, 1), (15, 7)], WATER, 0.0012)
    _cottage(m, -5, 0, 6, 5, (132, 112, 88), (84, 106, 130), 3)
    u, v = box_at(-5, 0, 6)
    chimney(m, along(u, 0.25), (v[0] + v[1]) / 2, (tall(7), tall(9.5)), STONE)
    for x, y in ((2, 1), (6, -1)):
        pu, pv = spot(x, y)
        m.box((pu - 0.003, pu + 0.003), (pv - 0.003, pv + 0.003), (0.0, tall(6)), (88, 64, 44))
    a, b = spot(2, 1), spot(6, -1)
    strut(m, (a[0], a[1], tall(5.6)), (b[0], b[1], tall(5.6)), 0.003, (88, 64, 44))
    with m.fine():
        for t in (0.25, 0.5, 0.75):
            nu, nv = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            m.box((nu - 0.012, nu + 0.012), (nv - 0.001, nv + 0.001), (tall(2), tall(5.6)), NET)
    hu, hv = hall_at(8, 4.5, 10, 2.4)
    m.box(hu, hv, (0.0, tall(1.6)), HULL)
    m.box(
        (hu[0] + 0.006, hu[1] - 0.006),
        (hv[0] + 0.004, hv[1] - 0.004),
        (tall(1.2), tall(1.62)),
        (92, 40, 34),
    )
    bu, bv = spot(-8, 5)
    upright(m, bu, bv, wide(1.6), (0.0, tall(2)), (176, 132, 76))
    dome(m, bu, bv, wide(1.5), (tall(2), tall(2.8)), NET)
    return m


def hop_garden() -> Model:
    """Tall poles strung with wires and green hop bines hung with cones, a round brick oast with
    a white cowl on its kiln, a store with a door and pockets of dried hops beside it."""
    m = Model()
    poles = [(-12 + 4 * k, -2 + 2 * k) for k in range(4)] + [
        (-8 + 4 * k, -4 + 2 * k) for k in range(3)
    ]
    for x, y in poles:
        blob(m, x, y + 0.3, 2, 0.9, (98, 80, 54), 0.0008)
        pu, pv = spot(x, y)
        m.box((pu - 0.003, pu + 0.003), (pv - 0.003, pv + 0.003), (0.0, tall(16)), (122, 96, 66))
        for k, (du, dv) in enumerate(((-0.006, 0.0), (0.006, 0.004))):
            strut(
                m,
                (pu + du, pv + dv, 0.0),
                (pu + du * 0.5, pv + dv * 0.5, tall(13 + k)),
                0.004,
                shade(BINE, 1 - 0.15 * k),
            )
        with m.fine():
            for k in range(4):
                dome(
                    m,
                    pu + (0.006 if k % 2 else -0.006),
                    pv,
                    0.004,
                    (tall(4 + 3 * k), tall(5 + 3 * k)),
                    HOP,
                )
    for (x0, y0), (x1, y1) in itertools.pairwise(poles[:4]):
        a, b = spot(x0, y0), spot(x1, y1)
        strut(m, (a[0], a[1], tall(15.5)), (b[0], b[1], tall(15.5)), 0.0015, (90, 90, 96))
    ou, ov = spot(10, 0)
    radius = wide(4)
    blob(m, 10.2, 0.5, 5, 2.1, (62, 70, 40), 0.0006)
    upright(m, ou, ov, radius, (0.0, tall(9)), OAST)
    m.box(
        (ou + radius - 0.002, ou + radius + 0.002),
        (ov - 0.008, ov + 0.008),
        (0.0, tall(4.4)),
        (60, 40, 34),
    )
    cone(m, ou, ov, radius * 1.1, tall(8), KILN, tall(9))
    upright(m, ou, ov, wide(1), (tall(16.6), tall(18.6)), COWL)
    m.box((ou - 0.004, ou + 0.012), (ov - 0.006, ov + 0.006), (tall(17.4), tall(19.4)), COWL)
    su, sv = box_at(4, 4, 4)
    walls(m, su, sv, tall(4), (222, 210, 186))
    door(m, (along(su, 0.3), along(su, 0.6)), tall(2.7), sv[1])
    side_window(m, su, sv, (0.35, 0.65), 4, 1)
    gable_roof(m, su, sv, (tall(4), tall(7)), (112, 82, 66), (222, 210, 186))
    for x, y in ((9, 5.5), (11, 4.5)):
        pu, pv = spot(x, y)
        upright(m, pu, pv, wide(1.2), (0.0, tall(4)), (232, 222, 196))
        dome(m, pu, pv, wide(1.2), (tall(4), tall(5)), (232, 222, 196))
    return m


def beekeeper() -> Model:
    """A patch of flowering meadow with straw skeps on it, a stand of honey jars beside them."""
    m = Model()
    patch(m, [(-12, 0), (0, 6), (12, 0), (0, -6)], (128, 168, 84))
    with m.fine():
        for k, (x, y) in enumerate(((-3, -3), (4, -4), (8, -1), (-8, -1), (-5, 3), (5, 3))):
            blob(
                m,
                x,
                y,
                0.8,
                0.4,
                ((240, 226, 120), (238, 238, 232), (214, 120, 160))[k % 3],
                0.0012,
            )
    for x, y in ((0, -2), (-6, 1), (6, 1), (0, 4)):
        hu, hv = spot(x, y)
        m.box((hu - 0.012, hu + 0.012), (hv - 0.012, hv + 0.012), (0.0, tall(0.6)), (122, 90, 60))
        upright(m, hu, hv, wide(2.4), (tall(0.6), tall(2.6)), HIVE)
        dome(m, hu, hv, wide(2.4), (tall(2.6), tall(6.5)), HIVE)
        with m.fine():
            for h in (1.6, 3.2):
                upright(
                    m, hu, hv, wide(2.4) + 0.0012, (tall(h), tall(h) + 0.003), shade(HIVE, 0.75)
                )
            m.box(
                (hu + wide(2.2), hu + wide(2.5)),
                (hv - 0.004, hv + 0.004),
                (tall(0.7), tall(1.4)),
                (60, 40, 20),
            )
    su, sv = hall_at(9.8, 4.4, 5, 1.4)
    m.box(su, sv, (0.0, tall(0.8)), shade((122, 90, 60), 1.2))
    for x, y in ((8.6, 4.4), (11, 3.2)):
        ju, jv = spot(x, y - 0.8)
        m.box((ju - 0.008, ju + 0.008), (jv - 0.008, jv + 0.008), (tall(0.8), tall(2.8)), HONEY)
    return m


def potter() -> Model:
    """A potter's cottage beside a bottle kiln of brick, with pots set out to dry."""
    m = Model()
    _cottage(m, -6, -1, 5, 4, WHITEWASH, (150, 70, 52), 3)
    u, v = box_at(-6, -1, 5)
    chimney(m, along(u, 0.3), (v[0] + v[1]) / 2, (tall(5.5), tall(8)), BRICK)
    ku, kv = spot(7.5, 0)
    radius = wide(3.5)
    upright(m, ku, kv, radius, (0.0, tall(3)), shade(BRICK, 0.9))
    dome(m, ku, kv, radius, (tall(3), tall(7)), shade(BRICK, 0.9))
    upright(m, ku, kv, wide(1.5), (tall(6), tall(11)), shade(BRICK, 0.8))
    m.box(
        (ku + radius - 0.002, ku + radius + 0.002),
        (kv - 0.01, kv + 0.01),
        (0.0, tall(2)),
        (50, 36, 32),
    )
    for x in (-10, -7.5, 9, 11.5):
        pu, pv = spot(x, 4)
        upright(m, pu, pv, wide(0.9), (0.0, tall(1.6)), CLAY)
        dome(m, pu, pv, wide(0.9), (tall(1.6), tall(2.4)), shade(CLAY, 1.1))
    return m


def sugar_refinery() -> Model:
    """A tall many-windowed refinery under a flat roof beside a tall brick stack, a heap of
    beet by its door and white sugar loaves in blue paper set out on the ground."""
    m = Model()
    wall = (232, 226, 210)
    lintel = shade(BRICK, 1.3)
    u, v = box_at(-3, 0, 8)
    high = tall(18)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    plinth(m, u, v, tall(1), shade(wall, 0.75))
    flat_roof(m, u, v, high, (92, 86, 94), wall)
    for drop in (2.5, 6, 9.5, 13):
        for t in (0.2, 0.45, 0.7, 0.92):
            arch_window(
                m,
                (along(u, t - 0.05), along(u, t + 0.05)),
                (tall(18 - drop - 2), tall(18 - drop - 0.6)),
                v[1],
                WINDOW,
                lintel,
            )
        for t in (0.3, 0.7):
            ends = sorted((v[1] - (v[1] - v[0]) * (t - 0.06), v[1] - (v[1] - v[0]) * (t + 0.06)))
            parts.front_plate(
                m,
                u[1] + 0.0006,
                (ends[0], ends[1]),
                (tall(18 - drop - 2), tall(18 - drop)),
                shade(WINDOW, 0.75),
            )
    door(m, (along(u, 0.55), along(u, 0.75)), tall(3), v[1], (70, 50, 40))
    cornice(m, u, v, high - 0.008, shade(wall, 1.1))
    su, sv = box_at(8, -3, 2)
    m.box(su, sv, (0.0, tall(34 * 0.6)), shade(BRICK, 0.95))
    m.box(
        (su[0] - 0.003, su[1] + 0.003),
        (sv[0] - 0.003, sv[1] + 0.003),
        (tall(34 * 0.6), tall(34 * 0.6) + 0.008),
        (60, 52, 50),
    )
    smoke(m, (su[0] + su[1]) / 2, (sv[0] + sv[1]) / 2, tall(34 * 0.6) + 0.01, 2)
    for x, y in ((-13, 5.5), (-11.5, 4.6)):
        bu, bv = spot(x, y)
        cone(m, bu, bv, wide(2.5), tall(2), BEET)
    for x, y in ((9, 4), (11, 2.5), (10, 1)):
        lu, lv = spot(x, y)
        upright(m, lu, lv, wide(1.4), (0.0, tall(2)), SUGAR_BLUE)
        cone(m, lu, lv, wide(1.3), tall(6), SUGAR_WHITE, tall(2))
    for x, y in ((3, 7), (6, 7.5)):
        pu, pv = spot(x, y)
        upright(m, pu, pv, wide(2), (0.0, tall(1.4)), SUGAR_WHITE)
        dome(m, pu, pv, wide(2), (tall(1.4), tall(3)), SUGAR_WHITE)
    return m


def furniture_maker() -> Model:
    """A timber workshop under a pitched roof with a big door, a chair on its ridge for a sign,
    stacks of seasoning boards and sawdust swept out by the door."""
    m = Model()
    blob(m, 8, 1, 7, 3, (112, 86, 58), 0.0005)
    wall, roof = (150, 108, 70), (104, 84, 70)
    u, v = box_at(-4, 0, 7)
    high = tall(6)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    plinth(m, u, v, tall(0.8), shade(wall, 0.72))
    window(m, (along(u, 0.1), along(u, 0.32)), below(6, 1.8, 1.4), v[1], WINDOW)
    door(m, (along(u, 0.45), along(u, 0.82)), tall(4.2), v[1], (56, 40, 30))
    side_window(m, u, v, (0.3, 0.6), 6, 1.8)
    gable_roof(m, u, v, (high, high + tall(4)), roof, wall)
    cu, cv = along(u, 0.4), (v[0] + v[1]) / 2
    chair = (170, 52, 44)
    m.box(
        (cu - 0.012, cu + 0.012),
        (cv - 0.012, cv + 0.012),
        (high + tall(4.4), high + tall(5)),
        chair,
    )
    m.box(
        (cu + 0.008, cu + 0.012),
        (cv - 0.012, cv + 0.012),
        (high + tall(5), high + tall(7.4)),
        chair,
    )
    for du in (-0.01, 0.01):
        for dv in (-0.01, 0.01):
            m.box(
                (cu + du - 0.002, cu + du + 0.002),
                (cv + dv - 0.002, cv + dv + 0.002),
                (high + tall(3.6), high + tall(4.4)),
                chair,
            )
    for x, y, h in ((9, -1, 6), (11, 2, 4)):
        bu, bv = hall_at(x, y - 0.8, 8, 3)
        m.box(bu, bv, (0.0, tall(h)), BOARD)
        with m.fine():
            for j in range(1, int(h)):
                m.panel(bu, (tall(j) - 0.001, tall(j) + 0.001), bv[1] + 0.0006, shade(BOARD, 0.75))
    blob(m, -9.5, 7.5, 3, 1, SAWDUST, 0.001)
    return m


def flax_field() -> Model:
    """A small whitewashed cottage with a tiled roof and a chimney, a strip of blue-flowered flax
    before it, and sheaves of flax stood up in stooks to dry beside it on the stubble."""
    m = Model()
    stubble = (178, 168, 106)
    patch(m, [(-2.5, -0.5), (8, -5.5), (15.5, -1.5), (5, 4)], stubble)
    _cottage(m, -6, -1, 5, 4, WHITEWASH, (128, 96, 70), 3)
    u, v = box_at(-6, -1, 5)
    chimney(m, along(u, 0.3), (v[0] + v[1]) / 2, (tall(5.5), tall(8)), BRICK)
    patch(m, [(-11, 3), (-3, 7), (-5, 8), (-13, 4)], FLAX, 0.0012)
    with m.fine():
        for k in range(8):
            x, y = -11.5 + k, 3.8 + 0.5 * k
            blob(m, x, y, 0.5, 0.3, FLAX_FLOWER, 0.0016)
    for x, y in ((4, 0), (8, -2), (7, 2), (11, 0), (10, -3)):
        su, sv = spot(x, y)
        cone(m, su, sv, wide(2.2), tall(6), STRAW)
    return m


def sugar() -> Model:
    """A field of tall sugar cane in rows on bare earth, a stand of it cut at one end."""
    m = Model()
    patch(m, [(-13, 0), (0, 6), (13, 0), (0, -6)], CANE_GROUND)
    for row in range(5):
        for k in range(6):
            x, y = -9 + 3 * k + row * 1.2 - 2.4, -3 + 1.5 * k - row * 0.6 + 1.2
            if abs(x) / 13 + abs(y) / 6 > 0.92:
                continue
            cu, cv = spot(x, y)
            cut = k == 5 and row > 2
            high = tall(2.4 if cut else 7)
            color = (176, 182, 104) if cut else shade(CANE, 0.62 + 0.08 * row)
            for du, dv in ((-0.006, 0.0), (0.006, 0.003), (0.0, -0.005)):
                strut(
                    m, (cu + du, cv + dv, 0.0), (cu + du * 1.6, cv + dv * 1.6, high), 0.004, color
                )
            if not cut:
                cone(m, cu, cv, wide(1.6), tall(2), shade(color, 1.15), high - tall(1))
    return m
