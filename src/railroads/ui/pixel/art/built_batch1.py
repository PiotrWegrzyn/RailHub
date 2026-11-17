"""Livestock farms, a butcher's, a slaughterhouse, a brewery, a dairy and pitheads built as 3D
models (railroads.ui.pixel.art.built) after their painted sprites, laid out with the painted
sprite's own coordinates."""

from railroads.ui.pixel import train_parts as parts
from railroads.ui.pixel.art.built import (
    DOOR,
    WORN,
    along,
    arch_window,
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
    paddock,
    patch,
    plinth,
    side_window,
    smoke,
    spot,
    strut,
    tall,
    track,
    upright,
    walls,
    wide,
    window,
    worn,
)
from railroads.ui.pixel.art.crafts import ROOF
from railroads.ui.pixel.art.farms import EARTH, MUD
from railroads.ui.pixel.art.painter import (
    BRICK,
    COPPER,
    DOORWAY,
    METAL,
    RGB,
    TIMBER,
    WHITEWASH,
    WINDOW,
    shade,
)
from railroads.ui.pixel.models import Model

HAY: RGB = (220, 186, 96)
TRIM: RGB = (236, 230, 214)
TROUGH: RGB = (122, 90, 60)
WATER: RGB = (92, 148, 196)
SKIN: RGB = (232, 166, 160)
DAIRY_BLUE: RGB = (52, 92, 170)
MILK: RGB = (236, 238, 240)
COLD: RGB = (234, 236, 238)
SAUSAGE: RGB = (170, 74, 54)
HAM: RGB = (196, 76, 66)
SLATE: RGB = (86, 80, 88)
BREW_ROOF: RGB = (84, 76, 80)
HOP: RGB = (110, 170, 70)
GRAIN: RGB = (176, 150, 92)
RAIL: RGB = (150, 152, 160)
SLEEPER: RGB = (96, 72, 52)
TUB: RGB = (128, 98, 80)


def _boards(
    m: Model,
    u: tuple[float, float],
    v: tuple[float, float],
    high: float,
    wall: RGB,
    drops: tuple[float, ...],
    height_px: float,
) -> None:
    """The seams between the boards of a timber wall, `drops` pixels down both front walls."""
    with m.fine():
        for drop in drops:
            z = tall(height_px - drop)
            m.panel(u, (z - 0.0012, z + 0.0012), v[1] + 0.0008, shade(wall, 0.8))
            parts.front_plate(m, u[1] + 0.0008, v, (z - 0.0012, z + 0.0012), shade(wall, 0.66))


def _barn(
    m: Model,
    x: float,
    y: float,
    half: float,
    height: float,
    wood: RGB,
    roof: RGB,
    rise: float,
    drops: tuple[float, ...],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """A boarded barn on a plinth under a tiled gable roof; returns its footprint and height."""
    u, v = box_at(x, y, half)
    high = tall(height)
    footing(m, u, v)
    walls(m, u, v, high, wood)
    plinth(m, u, v, tall(0.9), shade(wood, 0.72))
    _boards(m, u, v, high, wood, drops, height)
    gable_roof(m, u, v, (high, high + tall(rise)), roof, wood)
    return u, v, high


def _cow(m: Model, x: float, y: float, hide: RGB, patches: RGB) -> None:
    """A cow standing on the painted sprite's (x, y), its head to the left."""
    u, v = spot(x, y)
    half_u, half_v = wide(3) * 0.75, wide(1.2) * 0.75
    legs = tall(1)
    for du in (-half_u * 0.7, half_u * 0.7):
        for dv in (-half_v * 0.6, half_v * 0.6):
            m.box(
                (u + du - 0.003, u + du + 0.003),
                (v + dv - 0.003, v + dv + 0.003),
                (0.0, legs),
                shade(hide, 0.6),
            )
    m.box((u - half_u, u + half_u), (v - half_v, v + half_v), (legs, legs + tall(2.6)), hide)
    with m.fine():
        m.panel(
            (u - half_u * 0.2, u + half_u * 0.5),
            (legs + tall(0.8), legs + tall(2)),
            v + half_v + 0.0006,
            patches,
        )
    hu = u - half_u - wide(1)
    m.box(
        (hu - wide(0.9), hu + wide(0.9)),
        (v - half_v * 0.7, v + half_v * 0.7),
        (legs + tall(1.2), legs + tall(3.2)),
        hide,
    )
    with m.fine():
        m.box(
            (hu - 0.002, hu + 0.002),
            (v - half_v, v + half_v),
            (legs + tall(3.2), legs + tall(3.6)),
            (232, 222, 196),
        )


def _pig(m: Model, x: float, y: float) -> None:
    """A pink pig standing on the painted sprite's (x, y)."""
    u, v = spot(x, y)
    radius = wide(1.6)
    legs = tall(0.8)
    for du in (-radius * 0.6, radius * 0.6):
        m.box(
            (u + du - 0.003, u + du + 0.003), (v - 0.003, v + 0.003), (0.0, legs), shade(SKIN, 0.6)
        )
    upright(m, u, v, radius, (legs, legs + tall(1.4)), SKIN)
    dome(m, u, v, radius, (legs + tall(1.4), legs + tall(2.6)), SKIN)
    m.box(
        (u - radius - wide(1.2), u - radius + 0.004),
        (v - wide(0.6), v + wide(0.6)),
        (legs + tall(0.6), legs + tall(1.8)),
        shade(SKIN, 0.92),
    )


def _hen(m: Model, x: float, y: float, feathers: RGB) -> None:
    """A hen pecking about on the painted sprite's (x, y)."""
    u, v = spot(x, y)
    radius = wide(1.1)
    upright(m, u, v, radius, (0.0, tall(1.1)), feathers)
    dome(m, u, v, radius, (tall(1.1), tall(2)), feathers)
    m.box(
        (u - radius - wide(0.6), u - radius + 0.003),
        (v - wide(0.4), v + wide(0.4)),
        (tall(1.6), tall(2.8)),
        feathers,
    )
    with m.fine():
        m.box(
            (u - radius - wide(0.5), u - radius),
            (v - wide(0.2), v + wide(0.2)),
            (tall(2.8), tall(3.3)),
            (200, 50, 44),
        )


def _yard(m: Model, ground: RGB) -> None:
    blob(m, 6, 2.6, 8, 3, WORN, 0.0004)
    blob(m, 5.5, 2.8, 4.5, 1.8, ground, 0.0007)
    track(m, [(-10.2, 1.2), (-6.6, 3), (-2, 3.4)])


def cattle_ranch() -> Model:
    """A boarded cattle barn on a plinth with cross-braced doors, a hayloft hatch and a tiled
    roof, a haystack, and a fenced paddock with a water trough and cows in it."""
    m = Model()
    _yard(m, shade(EARTH, 0.92))
    blob(m, 10.4, 0.6, 3.6, 1.8, shade(MUD, 1.25), 0.0009)
    worn(m, -10.4, 1.4, 2.8)
    wood = (136, 86, 56)
    u, v, _ = _barn(m, -8, -2, 6, 6, wood, (86, 74, 70), 4, (1.5, 3, 4.5))
    doors = (along(u, 0.45), along(u, 0.8))
    m.panel(doors, (0.0, tall(4.5)), v[1] + 0.0006, TRIM)
    m.panel((along(u, 0.52), along(u, 0.73)), (0.0, tall(3.5)), v[1] + 0.0012, shade(wood, 0.7))
    with m.fine():
        a, b = along(u, 0.52), along(u, 0.73)
        parts.bar(m, (a, 0.0), (b, tall(3.5)), 0.003, v[1] + 0.0018, TRIM)
        parts.bar(m, (b, 0.0), (a, tall(3.5)), 0.003, v[1] + 0.0018, TRIM)
    side_window(m, u, v, (0.35, 0.65), 6, 1.4)
    hu, hv = spot(-12.5, 3.5)
    cone(m, hu, hv, wide(2.6), tall(3.5), HAY)
    paddock(m, 6, 2, 9)
    tu, tv = hall_at(10.5, 0, 4, 1.4)
    m.box(tu, tv, (0.0, tall(1.2)), TROUGH)
    m.box(
        (tu[0] + 0.002, tu[1] - 0.002),
        (tv[0] + 0.002, tv[1] - 0.002),
        (tall(1.2), tall(1.2) + 0.001),
        WATER,
    )
    _cow(m, 6, 0, (138, 84, 50), (96, 58, 36))
    _cow(m, 10, 4, (240, 238, 232), (40, 38, 42))
    _cow(m, 3, 4, (40, 38, 42), (240, 238, 232))
    return m


def pig_farm() -> Model:
    """A low boarded sty with a tiled roof over a plinth, a heap of straw, and a fenced pen of
    trampled mud with a feeding trough and pink pigs in it."""
    m = Model()
    blob(m, 6, 2.6, 8, 3, WORN, 0.0004)
    blob(m, 7, 1.4, 6, 3, shade(MUD, 1.25), 0.0007)
    blob(m, 7, 1.4, 4.5, 2.1, MUD, 0.0009)
    track(m, [(-10.2, 1.2), (-6.6, 3), (-2, 3.4)])
    worn(m, -10.4, 1.4, 2.8)
    wood = (150, 104, 72)
    u, v, _ = _barn(m, -8, -2, 6, 4.4, wood, (150, 80, 60), 3, (1.5, 3))
    m.panel((along(u, 0.5), along(u, 0.75)), (0.0, tall(3)), v[1] + 0.0006, (60, 40, 30))
    hu, hv = spot(-12.5, 3.5)
    cone(m, hu, hv, wide(2.4), tall(3), HAY)
    paddock(m, 6, 2, 9)
    tu, tv = hall_at(9, 0.2, 4, 1.4)
    m.box(tu, tv, (0.0, tall(1.2)), TROUGH)
    for x, y in ((4, 1), (9, 4), (2, 4.5)):
        _pig(m, x, y)
    return m


def chicken_farm() -> Model:
    """A whitewashed henhouse on stilts with a ramp up to its hatch and a tiled roof, nesting
    boxes along its side, and a fenced yard of scratched earth with hens pecking about."""
    m = Model()
    _yard(m, shade(EARTH, 0.95))
    u, v = box_at(-7, -1, 5)
    footing(m, u, v)
    for uu in u:
        for vv in v:
            m.box((uu - 0.003, uu + 0.003), (vv - 0.003, vv + 0.003), (0.0, tall(1)), TIMBER)
    walls_low = tall(1)
    m.footprints.append((u, v))
    m.box(u, v, (walls_low, tall(4)), WHITEWASH)
    _boards(m, u, v, tall(4), WHITEWASH, (2,), 4)
    m.panel((along(u, 0.55), along(u, 0.75)), (walls_low, tall(2)), v[1] + 0.0006, (70, 50, 36))
    hatch = along(u, 0.65)
    parts.bar(m, (hatch, walls_low), (hatch - 0.01, 0.0), 0.006, v[1] + 0.012, (150, 110, 70))
    nest = (v[1] - (v[1] - v[0]) * 0.8, v[1] - (v[1] - v[0]) * 0.2)
    m.box((u[1], u[1] + 0.012), nest, (tall(1.6), tall(2.4)), (150, 110, 70))
    gable_roof(m, u, v, (tall(4), tall(7)), (150, 70, 52), WHITEWASH)
    paddock(m, 6, 2, 8)
    for x, y, feathers in (
        (3, 1, (244, 240, 230)),
        (7, 3, (176, 108, 60)),
        (9, 0.5, (244, 240, 230)),
        (5, 4.5, (244, 240, 230)),
        (10, 3.6, (120, 70, 40)),
    ):
        _hen(m, x, y, feathers)
    return m


def sausage_maker() -> Model:
    """A half-timbered butcher's shop with a smokehouse chimney and sausages hung over its door."""
    m = Model()
    wall = (226, 210, 180)
    u, v = box_at(-1, 0, 5)
    high = tall(4)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    door(m, (along(u, 0.45), along(u, 0.65)), tall(2.7), v[1], DOOR)
    side_window(m, u, v, (0.3, 0.6), 4, 1)
    m.panel((along(u, 0.35), along(u, 0.75)), (tall(3.1), tall(3.3)), v[1] + 0.0008, (130, 80, 50))
    gable_roof(m, u, v, (high, high + tall(3)), ROOF, wall)
    chimney(m, along(u, 0.3), (v[0] + v[1]) / 2, (high + tall(1.5), high + tall(4.4)), BRICK)
    su, sv = spot(7, -1)
    m.box((su - 0.012, su + 0.012), (sv - 0.012, sv + 0.012), (0.0, tall(9.5)), shade(BRICK, 0.8))
    smoke(m, su, sv, tall(9.5), 1)
    for t in (0.3, 0.42, 0.54):
        at = along(u, t)
        m.box(
            (at - 0.002, at + 0.002), (v[1] + 0.002, v[1] + 0.006), (tall(1.6), tall(2.9)), SAUSAGE
        )
    return m


def meat_factory() -> Model:
    """A brick slaughterhouse with two rows of arched windows, a louvred vent on its ridge and a
    ham hung out as its sign, a ramp up from a cattle pen trampled to mud, and a white cold
    store with an insulated door and a refrigeration unit on its roof."""
    m = Model()
    lintel = shade(BRICK, 1.3)
    patch(m, [(-14.6, 3), (-10, 0.7), (-5.4, 3), (-10, 5.3)], (112, 94, 66))
    u, v = hall_at(0, -3, 14, 8)
    high = tall(8)
    footing(m, u, v)
    walls(m, u, v, high, BRICK)
    plinth(m, u, v, tall(1), shade(BRICK, 0.7))
    for t in (0.6, 0.75, 0.9):
        arch_window(
            m, (along(u, t - 0.04), along(u, t + 0.04)), (tall(4), tall(5.4)), v[1], WINDOW, lintel
        )
        arch_window(
            m,
            (along(u, t - 0.04), along(u, t + 0.04)),
            (tall(1.4), tall(2.4)),
            v[1],
            WINDOW,
            lintel,
        )
    m.panel((along(u, 0.3), along(u, 0.45)), (tall(2), tall(5)), v[1] + 0.0006, (56, 44, 40))
    gable_roof(m, u, v, (high, high + tall(4)), SLATE, BRICK)
    ridge = (v[0] + v[1]) / 2
    m.box(
        (along(u, 0.4), along(u, 0.7)),
        (ridge - 0.008, ridge + 0.008),
        (high + tall(4), high + tall(5.4)),
        (176, 168, 160),
    )
    ramp = (along(u, 0.3), along(u, 0.45))
    strut(
        m,
        ((ramp[0] + ramp[1]) / 2, v[1] + 0.05, 0.0),
        ((ramp[0] + ramp[1]) / 2, v[1], tall(2)),
        0.03,
        TIMBER,
    )
    paddock(m, -10, 3, 5)
    _cow(m, -8, 4, (138, 84, 50), (96, 58, 36))
    su, sv = box_at(11, 3, 5)
    store_high = tall(6)
    footing(m, su, sv)
    walls(m, su, sv, store_high, COLD)
    m.box(su, sv, (store_high, store_high + 0.004), shade(COLD, 0.88))
    with m.fine():
        for t in (0.25, 0.5, 0.75):
            at = along(su, t)
            m.panel(
                (at - 0.0012, at + 0.0012),
                (tall(0.6), tall(5.5)),
                sv[1] + 0.0008,
                shade(COLD, 0.85),
            )
    ends = sorted((sv[1] - (sv[1] - sv[0]) * 0.3, sv[1] - (sv[1] - sv[0]) * 0.7))
    parts.front_plate(m, su[1] + 0.0006, (ends[0], ends[1]), (0.0, tall(4)), (150, 170, 186))
    mu, mv = box_at(12, -3.5, 2)
    m.box(mu, mv, (store_high, store_high + tall(2)), METAL)
    bu = along(u, 0.08)
    m.box((bu - 0.002, bu + 0.002), (v[1], v[1] + 0.03), (tall(5.8), tall(6)), (60, 56, 54))
    dome(m, bu, v[1] + 0.03, 0.012, (tall(3.4), tall(5.6)), HAM)
    return m


def brewery() -> Model:
    """A brick brewhouse with arched windows and door, a copper-domed tower with a loading door,
    barrels at its door, a heap of spent grain and a hop cone for a sign."""
    m = Model()
    lintel = shade(BRICK, 1.3)
    blob(m, -5, 7.5, 1.6, 0.8, (110, 96, 70), 0.001)
    u, v = hall_at(-4, -1, 12, 8)
    high = tall(8)
    footing(m, u, v)
    walls(m, u, v, high, BRICK)
    plinth(m, u, v, tall(1), shade(BRICK, 0.7))
    for t in (0.45, 0.65, 0.85):
        arch_window(
            m, (along(u, t - 0.04), along(u, t + 0.04)), (tall(4), tall(5.4)), v[1], WINDOW, lintel
        )
    for t in (0.45, 0.85):
        arch_window(
            m,
            (along(u, t - 0.04), along(u, t + 0.04)),
            (tall(1.2), tall(2.4)),
            v[1],
            WINDOW,
            lintel,
        )
    arch_window(m, (along(u, 0.6), along(u, 0.71)), (0.0, tall(2.6)), v[1], DOORWAY, lintel)
    m.panel((along(u, 0.04), along(u, 0.34)), below(8, 1, 4), v[1] + 0.0006, (222, 210, 180))
    hu = along(u, 0.19)
    m.box((hu - 0.006, hu + 0.006), (v[1] + 0.001, v[1] + 0.004), (tall(4.2), tall(6.2)), HOP)
    gable_roof(m, u, v, (high, high + tall(4)), BREW_ROOF, BRICK)
    chimney(m, along(u, 0.25), (v[0] + v[1]) / 2, (high + tall(2), high + tall(8)), BRICK)
    tower = shade(BRICK, 1.08)
    tu, tv = box_at(5, 3, 4)
    tower_high = tall(15)
    footing(m, tu, tv)
    walls(m, tu, tv, tower_high, tower)
    plinth(m, tu, tv, tall(1), shade(tower, 0.7))
    for low in (tall(3), tall(8)):
        arch_window(
            m,
            (along(tu, 0.42), along(tu, 0.58)),
            (tower_high - low - tall(2), tower_high - low),
            tv[1],
            WINDOW,
            shade(BRICK, 1.35),
        )
    ends = sorted((tv[1] - (tv[1] - tv[0]) * 0.25, tv[1] - (tv[1] - tv[0]) * 0.75))
    parts.front_plate(m, tu[1] + 0.0006, (ends[0], ends[1]), (0.0, tall(4)), (70, 50, 40))
    cu, cv = spot(5, 3)
    radius = (tu[1] - tu[0]) / 2
    dome(m, cu, cv, radius * 1.05, (tower_high, tower_high + tall(6)), COPPER)
    upright(m, cu, cv, 0.002, (tower_high + tall(6), tower_high + tall(8)), (90, 80, 70))
    gu, gv = spot(13, 6.5)
    cone(m, gu, gv, wide(2.2), tall(1.6), GRAIN)
    for x, y in ((-7, 6), (-3, 8)):
        bu, bv = spot(x, y)
        barrel(m, bu, bv, wide(1.3), tall(2.6), (150, 100, 56))
    return m


def dairy() -> Model:
    """A whitewashed dairy under a blue roof with blue trim and door, a blue sign with a drop of
    milk, a row of milk churns by its door and two cows in a fenced yard beside it."""
    m = Model()
    blob(m, -9, 1, 6.5, 3, WORN, 0.0004)
    paddock(m, -9, 1, 6.5)
    _cow(m, -9, 0, (240, 238, 232), (40, 38, 42))
    _cow(m, -6, 2.5, (138, 84, 50), (96, 58, 36))
    u, v = box_at(4, -1, 6)
    high = tall(6)
    footing(m, u, v)
    walls(m, u, v, high, WHITEWASH)
    plinth(m, u, v, tall(0.8), DAIRY_BLUE)
    m.panel(u, (high - 0.004, high), v[1] + 0.0008, DAIRY_BLUE)
    parts.front_plate(m, u[1] + 0.0008, v, (high - 0.004, high), DAIRY_BLUE)
    m.panel((along(u, 0.24), along(u, 0.4)), below(6, 1.8, 1.6), v[1] + 0.0008, DAIRY_BLUE)
    with m.fine():
        at = along(u, 0.32)
        m.panel((at - 0.002, at + 0.002), below(6, 2.2, 0.8), v[1] + 0.0014, MILK)
    door(m, (along(u, 0.64), along(u, 0.86)), tall(4), v[1], DAIRY_BLUE)
    side_window(m, u, v, (0.3, 0.65), 6, 1.8)
    gable_roof(m, u, v, (high, high + tall(4)), DAIRY_BLUE, WHITEWASH)
    for k in range(3):
        cu, cv = spot(6 + 1.6 * k, 5 - 0.8 * k)
        upright(m, cu, cv, 0.008, (0.0, tall(2.2)), MILK)
        upright(m, cu, cv, 0.005, (tall(2.2), tall(2.8)), shade(MILK, 0.85))
    return m


def _pithead(heap: RGB, glint: RGB, wheel: RGB) -> Model:
    """A pithead: a spoil heap of what it digs, a tub on rails between it and the shaft, a
    winding house with a chimney and a braced headframe with its spoked wheel and cable."""
    m = Model()
    dust = shade(heap, 1.5)
    track(m, [(7.8, 3.4), (8.8, 5), (9.8, 7)])
    worn(m, 8, 4, 2.6, dust)
    blob(m, -5, 2.6, 7, 2.8, shade(heap, 1.15), 0.0005)
    hu, hv = spot(-5.4, 2.4)
    cone(m, hu, hv, wide(5.8), tall(7.8), heap)
    with m.fine():
        for x, y in ((-7, -2), (-5, -3.4), (-5, 0), (-2.4, 0.6), (-8.6, 1.4)):
            gu, gv = spot(x, y + 2)
            m.box(
                (gu - 0.002, gu + 0.002),
                (gv - 0.002, gv + 0.002),
                (tall(1.5), tall(1.5) + 0.004),
                glint,
            )
    patch(m, [(0, -1), (3, -2.5), (6, -1), (3, 0.5)], (40, 36, 38))
    (ra, rb) = spot(1.5, 0.5), spot(-6, 4.2)
    for offset in (-0.006, 0.006):
        strut(m, (ra[0], ra[1] + offset, 0.002), (rb[0], rb[1] + offset, 0.002), 0.003, RAIL)
    tu, tv = spot(-2.5, 3.5)
    m.box((tu - 0.014, tu + 0.014), (tv - 0.01, tv + 0.01), (0.004, 0.024), TUB)
    dome(m, tu, tv, 0.012, (0.024, 0.032), heap)
    wall = (150, 132, 112)
    u, v = box_at(6, 1, 5)
    high = tall(4)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    plinth(m, u, v, tall(0.8), shade(wall, 0.75))
    window(m, (along(u, 0.55), along(u, 0.85)), below(4, 1, 1.2), v[1], WINDOW)
    side_window(m, u, v, (0.62, 0.85), 4, 1)
    ends = sorted((v[1] - (v[1] - v[0]) * 0.2, v[1] - (v[1] - v[0]) * 0.45))
    parts.front_plate(m, u[1] + 0.0006, (ends[0], ends[1]), (0.0, tall(2.7)), DOOR)
    gable_roof(m, u, v, (high, high + tall(2)), (92, 72, 62), wall)
    chimney(m, along(u, 0.85), (v[0] + v[1]) / 2, (high + tall(1), high + tall(4)), BRICK)
    frame = (74, 64, 58)
    fu, fv = spot(3, -1)
    top = tall(19)
    legs = [
        (fu - 0.05, fv - 0.03),
        (fu + 0.03, fv - 0.05),
        (fu - 0.03, fv + 0.05),
        (fu + 0.05, fv + 0.03),
    ]
    for lu, lv in legs:
        strut(m, (lu, lv, 0.0), (fu, fv, top), 0.007, frame)
    for h in (tall(7), tall(13)):
        t = h / top
        ring = [(lu + (fu - lu) * t, lv + (fv - lv) * t, h) for lu, lv in legs]
        for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
            strut(m, a, b, 0.004, frame)
    wu, wv = spot(5.5, -1)
    strut(m, (wu, wv, 0.0), (fu, fv, top * 0.85), 0.006, frame)
    radius = wide(2)
    m.cylinder_across(
        fu, radius, top + radius * 0.6, (fv - 0.004, fv + 0.004), wheel, shade(wheel, 0.75)
    )
    cable_end = spot(8.5, -1)
    strut(m, (fu, fv, top + radius), (cable_end[0], cable_end[1], high), 0.0015, (130, 130, 136))
    return m


def mine() -> Model:
    """A coal mine: a pithead over a black heap of coal."""
    return _pithead((52, 48, 50), (104, 100, 104), (96, 86, 78))


def shaft_mine(heap: RGB, glint: RGB) -> Model:
    """A pithead over a heap of what it digs, glinting with its ore."""
    return _pithead(heap, glint, glint)
