"""Farmsteads built as 3D models (railroads.ui.pixel.art.built) after their painted sprites in
railroads.ui.pixel.art.farms: the same parts in the same places and colours, laid out with the
painted sprite's own coordinates (built.spot, tall, box_at, hall_at)."""

from railroads.ui.pixel.art.built import (
    across,
    along,
    below,
    blob,
    box_at,
    chimney,
    cone,
    crate,
    dome,
    door,
    end_window,
    fence,
    footing,
    gable_roof,
    hall_at,
    lit_window,
    patch,
    plinth,
    sack,
    side_window,
    spot,
    tall,
    track,
    upright,
    walls,
    wide,
    worn,
)
from railroads.ui.pixel.art.farms import (
    FOOTING,
    SACK,
    WICKER,
)
from railroads.ui.pixel.art.painter import FRUIT, METAL, RGB, STONE, THATCH, shade
from railroads.ui.pixel.models import Model

BARN: RGB = (172, 62, 46)
BARN_ROOF: RGB = (112, 98, 84)
HAY: RGB = (220, 186, 96)
TRIM: RGB = (232, 226, 210)
SILO: RGB = (210, 208, 200)
LADDER: RGB = (200, 164, 108)
POTATO_LEAF: RGB = (84, 132, 56)
SOIL: RGB = (112, 86, 58)
TUBER: RGB = (150, 104, 58)


def farm() -> Model:
    """A farm: a red barn with cross-braced doors and a hayloft door under its roof, a round silo
    with hoops and a ladder, a haystack shedding straw and a bit of fence, with the yard trodden
    bare before the barn doors and a cart track running off from it."""
    m = Model()
    track(m, [(-4.6, 3.4), (-4.2, 4.8), (-3.4, 6)], 1.4)
    worn(m, -4.8, 3.6, 3.4)
    for k, (dx, dy) in enumerate(((-0.9, 0.2), (0.7, -0.4), (-0.2, 0.7), (1.0, 0.5), (-1.2, -0.3))):
        blob(m, -8.2 + dx * 2.4, 3.6 + dy * 1.2, 0.5, 0.3, shade(HAY, 0.85 if k % 2 else 1), 0.0011)
    hu, hv = spot(-8, 3)
    cone(m, hu - 0.004, hv, wide(2.2), tall(3.2), HAY)
    u, v = box_at(-1, 0, 7)
    high = tall(6)
    footing(m, u, v)
    walls(m, u, v, high, BARN)
    with m.fine():
        plinth(m, u, v, tall(0.8), shade(BARN, 0.8))
    doors = (along(u, 0.3), along(u, 0.72))
    m.panel(doors, (0.0, tall(4.4)), v[1] + 0.0006, shade(BARN, 0.75))
    m.panel(doors, (0.0, tall(4.4)), v[0] - 0.0006, shade(BARN, 0.75))
    with m.fine():
        side = v[1] + 0.0012
        z0, z1 = 0.0, tall(4.4)
        for a, b in ((doors[0], doors[0] + 0.004), (doors[1] - 0.004, doors[1])):
            m.panel((a, b), (z0, z1), side, TRIM)
        m.panel(doors, (z1 - 0.004, z1), side, TRIM)
        m.panel(doors, (z0, z0 + 0.004), side, TRIM)
        mid = (doors[0] + doors[1]) / 2
        m.panel((mid - 0.002, mid + 0.002), (z0, z1), side, TRIM)
    gable_roof(m, u, v, (high, high + tall(4)), BARN_ROOF, BARN)
    loft = sorted((across(v, 0.45), across(v, 0.55)))
    end_window(m, u[1], (loft[0], loft[1]), (high + tall(0.6), high + tall(1.8)), (60, 36, 28))
    su, sv = spot(8.5, 1)
    radius, silo_high = wide(1.8) * 1.2, tall(12)
    blob(m, 6.2 + 2.4, 0.6, 2.6, 1.2, FOOTING, 0.0007)
    upright(m, su, sv, radius, (0.0, silo_high), SILO)
    dome(m, su, sv, radius, (silo_high, silo_high + tall(2.7)), METAL)
    with m.fine():
        for h in (3.5, 7, 10.5):
            upright(m, su, sv, radius + 0.0015, (tall(h), tall(h) + 0.003), shade(SILO, 0.7))
    lu, lv = spot(9.8, 1)
    m.box((lu - 0.002, lu + 0.002), (lv - 0.008, lv - 0.004), (0.0, tall(11)), shade(METAL, 0.62))
    fence(m, (1, 6), (8, 2.5))
    return m


def orchard(fruit: list[RGB] = FRUIT) -> Model:
    """A fruit grower's cottage on a plinth, with a door, windows with sills, a tiled roof and a
    chimney, a picking ladder leant on its wall, crates of the fruit stacked on trodden earth
    beside it and a basket of it by the door, where a path runs off."""
    m = Model()
    track(m, [(-4.4, 2.6), (-4.6, 4.4), (-3.8, 6)])
    worn(m, -4.4, 2.8, 2.4)
    worn(m, 9.4, 2.2, 3.4)
    wall, roof = (226, 208, 166), (124, 82, 54)
    u, v = box_at(-1, 0, 7)
    high = tall(5)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    plinth(m, u, v, tall(0.9), shade(wall, 0.75))
    lit_window(m, u, v, (0.12, 0.3), 5, 1)
    lit_window(m, u, v, (0.78, 0.94), 5, 1)
    door(m, (along(u, 0.48), along(u, 0.64)), tall(5 - 1.3), v[1])
    side_window(m, u, v, (0.35, 0.62), 5, 1)
    gable_roof(m, u, v, (high, high + tall(4)), roof, wall)
    chimney(m, along(u, 0.3), (v[0] + v[1]) / 2, (high + tall(2), high + tall(6)), (168, 86, 60))
    lu = u[1] + 0.006
    for side in (across(v, 0.12), across(v, 0.2)):
        m.box((lu, lu + 0.004), (side - 0.002, side + 0.002), (0.0, tall(5.2)), LADDER)
    for x, y, k in ((10.5, 1, 1), (8.5, 2.6, 0), (8.5, 1, 2)):
        crate(m, x, y, fruit[k % len(fruit)])
    for k, (x, y) in enumerate(((11.5, 3.2), (6.4, 3.6), (12.4, 1.8), (-2.6, 4.4))):
        blob(m, x, y, 0.5, 0.3, fruit[k % len(fruit)], 0.0012)
    bu, bv = spot(-6.5, 4.4)
    upright(m, bu, bv, wide(1.8), (0.0, tall(2.5)), WICKER)
    dome(m, bu, bv, wide(1.8), (tall(2.5), tall(3.5)), fruit[0])
    return m


def potato_farm() -> Model:
    """A low fieldstone cottage on a plinth, with a window, a door and a stone chimney under
    ridged thatch, rows of potato plants before it and sacks and a crate of potatoes by its
    door."""
    m = Model()
    track(m, [(-3.4, 1.8), (-1.8, 4.4), (0.4, 8)])
    worn(m, -3.4, 2.2, 2.4)
    worn(m, 8, 2.6, 4)
    wall = (172, 166, 150)
    u, v = hall_at(-2, -1, 11, 6)
    high = tall(4)
    footing(m, u, v)
    walls(m, u, v, high, wall)
    plinth(m, u, v, tall(0.8), shade(wall, 0.75))
    with m.fine():
        for t, drop in ((0.15, 1), (0.4, 3), (0.85, 2), (0.6, 0.5)):
            at = along(u, t)
            m.panel((at - 0.004, at + 0.004), below(4, drop, 0.6), v[1] + 0.0008, shade(wall, 0.75))
    lit_window(m, u, v, (0.2, 0.32), 4, 1)
    door(m, (along(u, 0.55), along(u, 0.72)), tall(4 - 1.2), v[1], (80, 62, 48))
    side_window(m, u, v, (0.4, 0.62), 4, 1)
    gable_roof(m, u, v, (high, high + tall(4)), THATCH, wall, overhang=0.02)
    chimney(m, along(u, 0.7), (v[0] + v[1]) / 2, (high + tall(2), high + tall(6)), STONE)
    patch(m, [(-10, 1.5), (-3, 5), (-6, 6.5), (-13, 3)], SOIL)
    for row in range(2):
        x0, y0 = -11 - 1.2 * row, 2.5 + 0.6 * row
        for k in range(4):
            pu, pv = spot(x0 + 1.8 * k + 0.8, y0 + 0.9 * k - 0.5)
            dome(m, pu, pv, wide(0.9), (0.0, tall(1.2)), POTATO_LEAF)
    for x, y in ((8, 0), (5, 3), (9, 3.5)):
        sack(m, x, y, SACK)
    crate(m, 12, 3, TUBER, 1.4)
    for x, y in ((3, 5), (4, 6), (6, 6), (-7.6, 6), (-4.4, 5.4)):
        blob(m, x, y, 0.5, 0.3, TUBER, 0.0012)
    return m
