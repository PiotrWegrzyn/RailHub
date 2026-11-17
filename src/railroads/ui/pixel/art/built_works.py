"""Works and town buildings built as 3D models (railroads.ui.pixel.art.built) after their painted
sprites: the same parts in the same places and colours, laid out with the painted sprite's own
coordinates."""

from railroads.ui.pixel import train_parts as parts
from railroads.ui.pixel.art.built import (
    FOOTING,
    WORN,
    along,
    barrel,
    below,
    blob,
    box_at,
    chimney,
    cone,
    dome,
    door,
    footing,
    gable_roof,
    hall_at,
    log,
    plinth,
    sack,
    side_window,
    smoke,
    spot,
    tall,
    track,
    upright,
    walls,
    wide,
    window,
    worn,
)
from railroads.ui.pixel.art.industry import STILL_SLATE
from railroads.ui.pixel.art.painter import (
    BARREL,
    BRICK,
    COPPER,
    DOORWAY,
    METAL,
    RGB,
    STONE,
    TIMBER,
    WHITEWASH,
    WINDOW,
    chimney_top,
    shade,
)
from railroads.ui.pixel.art.provisions import BAKERY_ROOF, CRUST, FLOUR
from railroads.ui.pixel.models import Model

AWNING: list[RGB] = [(178, 52, 46), (238, 232, 220)]
LOAF: RGB = (214, 150, 62)
SHOP_GLASS: RGB = (120, 156, 180)
SAWDUST: RGB = (222, 194, 140)
LOG: RGB = (138, 96, 58)
LOG_END: RGB = (214, 176, 122)
PLANK: RGB = (200, 166, 116)
PEAT: RGB = (112, 84, 60)
LIT: RGB = (246, 214, 128)
BEAM: RGB = (92, 64, 44)


def _awning(m: Model, u: tuple[float, float], side: float, z: float, out: float = 0.03) -> None:
    """A striped awning sloping out from the wall `side` along `u`, its top at `z`."""
    count = 6
    step = (u[1] - u[0]) / count
    for k in range(count):
        a = u[0] + k * step
        corners_low = z - 0.025
        m.box((a, a + step), (side, side + out), (corners_low, corners_low + 0.006), AWNING[k % 2])
        m.box(
            (a, a + step),
            (side + out - 0.004, side + out),
            (corners_low - 0.012, corners_low),
            AWNING[k % 2],
        )


def bakery() -> Model:
    """A honey-coloured baker's house under a red roof, a striped awning over its window of
    loaves, a gold loaf hung by the door, a domed bread oven glowing beside it and a tall
    chimney smoking."""
    m = Model()
    u, v = box_at(0, 0, 7)
    high = tall(7)
    footing(m, u, v)
    walls(m, u, v, high, CRUST)
    plinth(m, u, v, tall(0.9), shade(CRUST, 0.75))
    shop = (along(u, 0.09), along(u, 0.53))
    m.panel(shop, (tall(0.9), tall(4.4)), v[1] + 0.0006, SHOP_GLASS)
    with m.fine():
        for k in range(4):
            lu = shop[0] + (shop[1] - shop[0]) * (k + 0.5) / 4
            m.panel((lu - 0.006, lu + 0.006), (tall(1.2), tall(2)), v[1] + 0.0012, LOAF)
    _awning(m, (along(u, 0.06), along(u, 0.56)), v[1], tall(7 - 1.8))
    door(m, (along(u, 0.62), along(u, 0.78)), tall(4), v[1], (120, 70, 44))
    side_window(m, u, v, (0.3, 0.65), 7, 2)
    gable_roof(m, u, v, (high, high + tall(5)), BAKERY_ROOF, CRUST)
    cu, cv = spot(-1, 0)
    chimney(m, cu, cv, (high + tall(3), tall(19)), BRICK, 0.012)
    smoke(m, cu, cv, tall(19) + 0.01, 2)
    lu = along(u, 0.02)
    m.box((lu - 0.002, lu + 0.002), (v[1], v[1] + 0.022), (tall(5.5), tall(5.7)), (60, 56, 54))
    upright(m, lu, v[1] + 0.02, 0.006, (tall(3.2), tall(5.3)), LOAF)
    ou, ov = spot(8, 3)
    dome(m, ou, ov, wide(3), (0.0, tall(4.2)), shade(BRICK, 0.82))
    with m.fine():
        m.box(
            (ou + wide(2.6), ou + wide(3.1)), (ov - 0.01, ov + 0.01), (0.0, tall(2)), (50, 36, 32)
        )
    blob(m, -6, 3.7, 3.4, 1, FLOUR, 0.001)
    for k in range(2):
        sack(m, -7 + 1.6 * k, 3.5 + 0.5 * (k % 2), (222, 208, 178))
    return m


def lumber_mill() -> Model:
    """A boarded steam sawmill with a door, a window and a chimney, its circular saw showing
    through the open side, with a stack of logs cut to show their rings, a heap of sawdust and a
    pile of sawn planks."""
    m = Model()
    track(m, [(5, 3.2), (9, 4.4), (13, 5.4)], 1.4)
    track(m, [(-5.2, 3), (-6, 4.6), (-5.4, 6)])
    worn(m, -5.2, 3, 2.2)
    blob(m, 2, 3.9, 4.4, 2.1, WORN, 0.0004)
    blob(m, 2.2, 3.9, 3.2, 1.5, shade(SAWDUST, 0.92), 0.0006)
    blob(m, 8.5, 1.8, 5.9, 1.6, FOOTING, 0.0004)
    blob(m, -7.5, 3.3, 3.9, 1.7, FOOTING, 0.0004)
    roof = (96, 84, 70)
    u, v = box_at(-3, 0, 7)
    high = tall(6)
    footing(m, u, v)
    walls(m, u, v, high, TIMBER)
    plinth(m, u, v, tall(0.9), shade(TIMBER, 0.72))
    with m.fine():
        for drop in (1.5, 3, 4.5):
            z = tall(6 - drop)
            m.panel(u, (z - 0.0015, z + 0.0015), v[1] + 0.0008, shade(TIMBER, 0.75))
            parts.front_plate(m, u[1] + 0.0008, v, (z - 0.0015, z + 0.0015), shade(TIMBER, 0.6))
    window(m, (along(u, 0.12), along(u, 0.3)), below(6, 1.2, 1.2), v[1], WINDOW)
    door(m, (along(u, 0.58), along(u, 0.78)), tall(4.2), v[1])
    opening = sorted((v[1] - (v[1] - v[0]) * 0.12, v[1] - (v[1] - v[0]) * 0.72))
    parts.front_plate(m, u[1] + 0.0006, (opening[0], opening[1]), (0.0, tall(4.6)), DOORWAY)
    middle = (opening[0] + opening[1]) / 2
    parts.front_ring(m, u[1] + 0.0012, tall(2.4), wide(1.8), METAL, middle, 18)
    with m.fine():
        parts.front_ring(m, u[1] + 0.0018, tall(2.4), wide(0.4), shade(METAL, 0.45), middle, 10)
    gable_roof(m, u, v, (high, high + tall(4)), roof, TIMBER)
    chimney(m, along(u, 0.2), (v[0] + v[1]) / 2, (high + tall(2), high + tall(6.5)), BRICK)
    radius = wide(1.1)
    for k, (y, z) in enumerate(((1.8, radius), (0.6, radius), (-0.6, radius))):
        log(m, (5.5 + 0.3 * k, y), (12.5 + 0.3 * k, y), z, radius, LOG, LOG_END)
    for y in (1.2, 0.0):
        log(m, (5.8, y), (12.8, y), radius * 2.7, radius, LOG, LOG_END)
    pu, pv = hall_at(-8, 2.4, 5, 1.6)
    m.box(pu, pv, (0.0, tall(1.4)), PLANK)
    with m.fine():
        m.panel(pu, (tall(0.7) - 0.001, tall(0.7) + 0.001), pv[1] + 0.0006, shade(PLANK, 0.7))
    hu, hv = spot(1.5, 4.5)
    cone(m, hu, hv, wide(2), tall(1.8), SAWDUST)
    return m


def distillery() -> Model:
    """A whitewashed still house on a stone plinth, its foot greyed by the damp, with a
    pagoda-roofed kiln vent, a cart track worn out of its door past a stack of peat, a copper pot
    still at its side, casks outside and a tall thin banded stack."""
    m = Model()
    blob(m, 9, 7.5, 1.6, 0.8, (96, 116, 120), 0.001)
    blob(m, -11, 6, 2.5, 1.2, shade(PEAT, 0.8), 0.0006)
    u, v = hall_at(-4, -1, 12, 7)
    high = tall(6)
    footing(m, u, v)
    walls(m, u, v, high, WHITEWASH)
    plinth(m, u, v, tall(1), STONE)
    for t in (0.15, 0.35, 0.8):
        window(m, (along(u, t - 0.06), along(u, t + 0.06)), below(6, 2, 2), v[1], (44, 46, 52))
    door(m, (along(u, 0.5), along(u, 0.66)), tall(3.5), v[1], (60, 48, 42))
    ends = sorted((v[1] - (v[1] - v[0]) * 0.44, v[1] - (v[1] - v[0]) * 0.56))
    parts.front_plate(m, u[1] + 0.0006, (ends[0], ends[1]), below(6, 2, 2), (44, 46, 52))
    gable_roof(m, u, v, (high, high + tall(4)), STILL_SLATE, WHITEWASH)
    vu, vv = along(u, 0.35), (v[0] + v[1]) / 2
    top = high + tall(4)
    m.box((vu - 0.016, vu + 0.016), (vv - 0.016, vv + 0.016), (top - 0.01, top + 0.03), WHITEWASH)
    m.roof(
        (vu - 0.026, vu + 0.026),
        (vv - 0.026, vv + 0.026),
        (top + 0.03, top + 0.06),
        STILL_SLATE,
        WHITEWASH,
    )
    su, sv = spot(3.5, 0)
    cap = -chimney_top(-4, -33)
    upright(m, su, sv, wide(0.6), (0.0, tall(cap)), (150, 80, 60))
    with m.fine():
        for h in (cap * 0.45, cap * 0.8):
            upright(m, su, sv, wide(0.6) + 0.0015, (tall(h), tall(h) + 0.004), (230, 228, 222))
    upright(m, su, sv, wide(1), (tall(cap), tall(cap) + 0.008), (90, 52, 42))
    smoke(m, su, sv, tall(cap) + 0.01, 2, (190, 190, 194))
    pu, pv = spot(8, 5)
    for (h0, r0), (h1, r1) in zip(
        [(0, 3), (2, 4), (4, 3.5), (6, 1.5)], [(2, 4), (4, 3.5), (6, 1.5), (7, 1)], strict=True
    ):
        upright(m, pu, pv, wide((r0 + r1) / 2) * 0.8, (tall(h0), tall(h1)), COPPER)
    dome(m, pu, pv, wide(1) * 0.8, (tall(7), tall(7.6)), shade(COPPER, 1.2))
    hu, hv = spot(-11, 6)
    cone(m, hu, hv, wide(2.5), tall(2), PEAT)
    bu, bv = spot(12, 4)
    barrel(m, bu, bv, wide(1.3), tall(2.6), BARREL)
    log(m, (9.5, 7.2), (12.5, 7.2), wide(1.2), wide(1.2), BARREL, shade(BARREL, 0.8))
    return m


def tavern() -> Model:
    """A half-timbered inn on a stone plinth with a lit window, a door under a lantern, a sign
    hung from a bracket, a beer barrel by the door, and a tiled roof with a smoking chimney."""
    m = Model()
    wall, roof = (234, 222, 192), (126, 74, 54)
    worn(m, -4, -2, 4)
    u, v = box_at(0, 0, 7)
    high = tall(7)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    plinth(m, u, v, tall(1), STONE)
    with m.fine():
        side = v[1] + 0.0009
        for t in (0.0, 0.5, 1.0):
            at = along(u, t)
            m.panel((at - 0.003, at + 0.003), (0.0, high), side, BEAM)
            ac = v[1] - (v[1] - v[0]) * t
            parts.front_plate(m, u[1] + 0.0009, (ac - 0.003, ac + 0.003), (0.0, high), BEAM)
        mid = tall(3.5)
        m.panel(u, (mid - 0.003, mid + 0.003), side, BEAM)
        parts.front_plate(m, u[1] + 0.0009, v, (mid - 0.003, mid + 0.003), BEAM)
        parts.bar(m, (along(u, 0.0), high - tall(0.5)), (along(u, 0.5), mid), 0.005, side, BEAM)
    m.panel((along(u, 0.1), along(u, 0.38)), below(7, 4.4, 1.4), v[1] + 0.0006, LIT)
    m.panel((along(u, 0.66), along(u, 0.88)), below(7, 1.2, 1.4), v[1] + 0.0006, WINDOW)
    ends = sorted((v[1] - (v[1] - v[0]) * 0.64, v[1] - (v[1] - v[0]) * 0.86))
    parts.front_plate(m, u[1] + 0.0006, (ends[0], ends[1]), below(7, 1.2, 1.4), shade(WINDOW, 0.8))
    ends = sorted((v[1] - (v[1] - v[0]) * 0.12, v[1] - (v[1] - v[0]) * 0.36))
    parts.front_plate(m, u[1] + 0.0006, (ends[0], ends[1]), below(7, 4.4, 1.4), shade(LIT, 0.85))
    door(m, (along(u, 0.6), along(u, 0.84)), tall(3), v[1], (70, 48, 34))
    lu = along(u, 0.72)
    m.box((lu - 0.003, lu + 0.003), (v[1], v[1] + 0.006), (tall(3.6), tall(4.1)), (250, 214, 96))
    gable_roof(m, u, v, (high, high + tall(5)), roof, wall)
    cu, cv = spot(1.75, 0)
    chimney(m, cu, cv, (high + tall(2), tall(14)), shade(STONE, 0.8), 0.011)
    smoke(m, cu, cv, tall(14), 1)
    su = along(u, 0.04)
    m.box((su - 0.002, su + 0.002), (v[1], v[1] + 0.03), (tall(5.2), tall(5.4)), (52, 48, 46))
    m.box(
        (su - 0.002, su + 0.002), (v[1] + 0.014, v[1] + 0.032), (tall(2.6), tall(5)), (170, 46, 42)
    )
    with m.fine():
        m.box(
            (su - 0.0028, su + 0.0028),
            (v[1] + 0.02, v[1] + 0.026),
            (tall(3.6), tall(4)),
            (240, 200, 90),
        )
    bu = along(u, 0.44)
    barrel(m, bu, v[1] + 0.03, wide(1.3), tall(2.4), BARREL)
    return m
