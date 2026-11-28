"""Big and mega wineries, breweries, distilleries, sake breweries, canneries, sugar refineries,
meat works, sausage makers and cooperages, each drawn as a building of its own: a cream château
with round towers among terraced vines, a honey-brick brewery of brew towers and copper kettle
houses crowned with a gilt sheaf, a whitewashed distillery of pagoda kilns, copper stills and
bonded warehouses, a walled sake brewery of white kura and great cedar vats, a cannery quarter
round a water tank made like a tin can, a red-brick refinery banded in sugar white and blue among
its beet, a brick slaughterhouse with a white cold store, a sausage works with smokehouses, and a
cooperage of stave yards and cask pyramids."""

import itertools
import math
from collections.abc import (
    Callable,
)

import numpy as np

from railroads.ui.pixel.art.crafts import (
    BUTCHER_PLASTER,
    BUTCHER_ROOF,
    COBBLES,
    IRON,
    SHOP_GLOW,
    STRIPES,
    WHITE_TILE,
    _craft_awning,
    _craft_band,
    _craft_beam,
    _craft_casement,
    _craft_framing,
    _craft_gilt_pig,
    _craft_log_pile,
    _craft_meat_window,
    _craft_smokehouse,
    _craft_sty,
    _hip,
)
from railroads.ui.pixel.art.crafts import (
    GILT as CRAFT_GILT,
)
from railroads.ui.pixel.art.crafts import (
    OAK as CRAFT_OAK,
)
from railroads.ui.pixel.art.crafts import (
    YARD as CRAFT_YARD,
)
from railroads.ui.pixel.art.industry import (
    BREW_BRICK,
    CHARRED,
    COLD_TILE,
    COOPER_ROOF,
    COOPER_WALL,
    DRESSING,
    FOAM,
    FOOD_APPLE,
    FOOD_BRICK,
    FOOD_CASE,
    FOOD_CHERRY,
    FOOD_CREAM,
    FOOD_GILT,
    FOOD_GREEN,
    FOOD_GREEN_APPLE,
    FOOD_IRON,
    FOOD_LABELS,
    FOOD_PEACH,
    FOOD_RENDER,
    FOOD_SETTS,
    FOOD_STEEL,
    FOOD_TIN,
    FOOD_TOMATO,
    GILT,
    GRAVEL,
    HOP,
    KETTLE,
    LATTICE,
    MALT_TILE,
    MILL_GLASS,
    MILL_SLATE,
    NAMAKO,
    NORTH_LIGHT,
    OAK,
    PLASTER,
    PUDDLE,
    SETTS,
    STILL_SLATE,
    STRAW,
    SUGAR_BEET,
    SUGAR_BLUE,
    SUGAR_BRICK,
    SUGAR_CREAM,
    SUGAR_GROUND,
    SUGAR_WHITE,
    SUGI,
    TAR,
    TURF,
    WHISKY,
    WINE,
    WINE_ROOF,
    YARD,
    _bed,
    _brew_dray,
    _brew_louvres,
    _brew_sheaf,
    _brew_sign,
    _brew_wheel,
    _brew_window,
    _butcher_sign,
    _cask,
    _cask_at,
    _cask_hoops,
    _cask_rack,
    _cask_raised,
    _cask_stack,
    _cask_staves,
    _cask_straw,
    _cold_door,
    _cooper_shavings,
    _food_basket,
    _food_cart,
    _food_cases,
    _food_retort,
    _food_sign,
    _food_tank,
    _food_tinplate,
    _food_wheel,
    _food_window,
    _footing,
    _meat_wagon,
    _mill_window,
    _north_lights,
    _on_roof,
    _roof_rows,
    _round_head,
    _sake_ball,
    _sake_namako,
    _sake_noren,
    _sake_roof,
    _stains,
    _still_pagoda,
    _still_peat,
    _still_pot,
    _still_window,
    _still_worm,
    _sugar_awning,
    _sugar_bags,
    _sugar_beet,
    _sugar_board,
    _sugar_display,
    _sugar_loaf,
    _sugar_siding,
    _sugar_stack,
    _sugar_wagon,
    _sugar_window,
    _wine_cart,
    _wine_press,
    _wine_vines,
    _wine_window,
    _works_siding,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    DIGITS,
    DOORWAY,
    METAL,
    RGB,
    STONE,
    TIMBER,
    WHITEWASH,
    WINDOW,
    Painter,
    Point,
    Sprite,
    Top,
    shade,
)
from railroads.ui.pixel.art.parts import (
    BEAST,
    DOOR,
    METRE,
    _band,
    _bands,
    _bar,
    _cow,
    _fence,
    _gable_across,
    _iso,
    _on_wall,
    _patch,
    _pig,
    _plinth,
    _ridge,
    _tiles,
    _turned,
    _vault,
)


def _cart(p: Painter, x: float, y: float, load: tuple[RGB, ...] = ()) -> None:
    """A two-wheeled wooden cart standing on (x, y) with kegs or crates of `load` on its bed."""
    p.polygon(
        [(x - 4, y - 3), (x + 2, y), (x + 4, y - 1), (x - 2, y - 4)], TIMBER, shade(TIMBER, 0.6)
    )
    p.line([(x + 2, y), (x + 6, y + 2)], shade(TIMBER, 0.6))
    for k, colour in enumerate(load):
        cx, cy = x - 1.5 + 2.2 * k, y - 2.4 + 1.1 * k
        p.ellipse(cx - 1.3, cy - 3.2, cx + 1.3, cy, colour)
        p.line([(cx - 1.2, cy - 1.4), (cx + 1.2, cy - 1.4)], shade(colour, 0.55))
    p.ellipse(x - 1, y - 2.5, x + 1.6, y + 0.5, (70, 54, 40))
    p.point(x + 0.3, y - 1, shade(TIMBER, 1.2))


def big_winery() -> Sprite:
    """A winery grown into a château: a two-storey house of cream stone under a hipped wine-red
    roof with dormers, its windows between wine-red shutters and its door up steps under a
    pediment with a gilt oculus, a round corner tower under a tall wine-red cone, an open press
    shed with a basket press and racked casks, a cellar vaulted into a grassy bank, grape carts
    and casks in the gravel court and a vineyard of six rows beside it."""
    m = METRE
    p = Painter(62, 48, (28, 30))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 9) * m, (v - 3) * m)

    p.polygon([at(-1, -5), at(25, -5), at(25, 4), at(-1, 4)], (206, 192, 160))
    p.shadow(18, 5, 5)
    for k, v in enumerate((9.5, 7, 4.5, 2, -0.5, -3)):
        _wine_vines(p, *at(-15 + 0.5 * (k % 2), v), 7)
    _drink_chateau(p, *at(9.5, 9), 19, 10, 2, 7)
    _drink_round_tower(p, *at(19.5, 3.6), 2.6, 13, 5.5)
    _drink_press_shed(p, *at(29, 9.5), 9, 6)
    _drink_cellar(p, *at(28, 0.5), 6, 6)
    _cask_rack(p, *at(4, 2.2), 6, 2)
    _wine_cart(p, *at(12, -1.6))
    _wine_cart(p, *at(17.5, -3.8))
    wx, wy = at(23, -1)
    p.blob(wx - 1.4 * m, wy - 0.5 * m, wx + 1.4 * m, wy + 0.7 * m, (126, 58, 74), 150)
    return p.sprite()


def mega_winery() -> Sprite:
    """A grand château of cream stone among its vines: a two-storey house with dormers in its
    hipped wine-red roof, its windows between wine-red shutters and its door up steps under a
    pediment with a gilt oculus, flanked by two round towers under tall wine-red cones with gilt
    vanes; terraces of vines stepping up the hillside behind, a stone fountain in the gravel
    court, two cellars vaulted into grassy banks, an open press shed with a basket press, casks
    racked in the court and grape carts, and a vineyard spreading before it."""
    m = METRE
    p = Painter(72, 56, (34, 34))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 9) * m, (v - 3) * m)

    for v, high in ((18.5, 5.4), (13.3, 3.6), (8.1, 1.8)):
        _drink_terrace(p, *at(-10, v), 16, high, 2)
    p.polygon([at(-1, -6), at(27, -6), at(27, 4), at(-1, 4)], (206, 192, 160))
    p.shadow(22, 6, 6)
    for k, v in enumerate((3, 0.5, -2, -4.5)):
        _wine_vines(p, *at(-17 + 0.5 * (k % 2), v), 6)
    _drink_cellar(p, *at(31, 11), 6, 6)
    _drink_chateau(p, *at(12, 10), 22, 10, 2, 9)
    _drink_press_shed(p, *at(32, 3), 8, 6)
    for u in (0.5, 23.5):
        _drink_round_tower(p, *at(u, 4.4), 3, 15, 6.5)
    _drink_fountain(p, *at(12, -0.8))
    _cask_rack(p, *at(2, 1.6), 5, 2)
    _cask_rack(p, *at(18.5, 1.6), 5, 2)
    _wine_cart(p, *at(5, -4.2))
    _wine_cart(p, *at(20, -4.4))
    _drink_cellar(p, *at(31.5, -4), 6, 5)
    return p.sprite()


def big_brewery() -> Sprite:
    """A brewery grown into a works of honey brick: a long brewhouse 24 by 10 m with four copper
    kettles gleaming through its tall arched windows and a vapour stack steaming on its ridge, a
    brew tower 20 m high crowned with a gilt sheaf of barley, a square malt kiln under a white
    cowl, a hop-green sign by the door, two drays loaded with casks in the cobbled yard, and a
    cooperage yard with a cooper's shop, casks being raised and long stacks of oak casks."""
    m = METRE
    p = Painter(62, 54, (28, 32))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10) * m, (v - 2) * m)

    p.polygon([at(-10, -8), at(24, -8), at(36, -3), at(36, 13), at(22, 13), at(-10, 2)], SETTS)
    _bed(p, _footing(*at(4, 8), 24 * m, 10 * m), margin=0.8)
    p.shadow(20, 6, 5)
    _drink_kiln(p, *at(-12.5, 8.5), 7, 9, BREW_BRICK, MALT_TILE, HOP, pagoda=False)
    house = _drink_brewhouse(p, *at(4, 8), 24, 10, 11, 4)
    _drink_cooper_shed(p, *at(30, 9.5), 9, 6, 4.5)
    _drink_tower(p, *at(19.25, 7.5), 6.5, 7, 20, 3.2)
    _brew_sign(p, *_on_wall(house[0], house[1], 0.012, 2.4 * m))
    _cask_staves(p, *at(26, 2.5), 8)
    _cask_staves(p, *at(26, 4.2), 6)
    for u, v in ((29, 4.6), (31, 3.4)):
        _cask_raised(p, *at(u, v))
    _cask_raised(p, *at(33, 4.8), fire=True)
    for k, rows in enumerate(((8, 7, 6), (7, 6))):
        _cask_stack(p, *at(25 + 1.2 * k, -0.4 - 2.2 * k), rows)
    _brew_dray(p, *at(-6, -2.2))
    _brew_dray(p, *at(5, -5.6))
    for u, v in ((-8, -6.2), (-7, -7)):
        _cask(p, *at(u, v), "up")
    return p.sprite()


def mega_brewery() -> Sprite:
    """A great brewery of honey brick spread along its own siding: twin malt kilns under white
    cowls, a brewhouse with three copper kettles gleaming through its arched windows, a grand
    brew tower 26 m high under a copper-domed cupola carrying a great gilt sheaf of barley, a
    kettle house whose copper hoods rise through its roof letting off steam, a second brew tower
    with a gilt sheaf, drays in the cobbled yard, a cooperage yard of long cask stacks, and cask
    wagons lined in gold waiting on the siding."""
    m = METRE
    p = Painter(72, 76, (34, 50))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10) * m, (v - 2) * m)

    p.polygon([at(-16, -7), at(42, -7), at(42, 13), at(30, 13), at(-16, 1)], SETTS)
    _bed(p, _footing(*at(10, 8), 52 * m, 10 * m), margin=0.8)
    p.shadow(26, 7, 6)
    for v in (11.5, 4.5):
        _drink_kiln(p, *at(-19.5, v), 6.5, 9, BREW_BRICK, MALT_TILE, HOP, pagoda=False)
    house = _drink_brewhouse(p, *at(-6, 8), 20, 10, 12, 3)
    kettle = p.hall(*at(23, 8), 18 * m, 10 * m, 12 * m, BREW_BRICK, None)
    k_left, k_front, k_right, _ = kettle
    _plinth(p, kettle, 12 * m, BREW_BRICK, 0.7 * m)
    for a, b in ((k_left, k_front), (k_front, k_right)):
        _bar(p, _on_wall(a, b, 0, 0.22 * m), _on_wall(a, b, 1, 0.22 * m), 0.45 * m, FOAM)
    for k in range(5):
        _bar(
            p,
            _on_wall(k_left, k_front, k / 4, 0.45 * m),
            _on_wall(k_left, k_front, k / 4, 11.3 * m),
            0.5 * m,
            shade(BREW_BRICK, 0.88),
        )
    for t in (0.125, 0.375, 0.625, 0.875):
        _brew_window(p, k_left, k_front, 18 * m, t, 2.6 * m, (1.4 * m, 10.8 * m))
    for t in (0.3, 0.7):
        _mill_window(
            p, k_front, k_right, 10 * m, t, 1.4 * m, (1.6 * m, 5 * m), shade(MILL_GLASS, 0.85)
        )
        _mill_window(
            p, k_front, k_right, 10 * m, t, 1.4 * m, (6.6 * m, 10.4 * m), shade(MILL_GLASS, 0.85)
        )
    p.gable(kettle, 4 * m, MALT_TILE, BREW_BRICK)
    _roof_rows(p, kettle, 4 * m, MALT_TILE, (0.25, 0.5, 0.75))
    for t in (0.2, 0.5, 0.8):
        dx, dy = _on_roof(kettle, 4 * m, t, 0.75)
        _drink_copper_dome(p, dx, dy + 0.6 * m, 2.2)
    _drink_tower(p, *at(36, 8), 6, 6.5, 20, 2.6)
    grand = at(8, 8)
    apex = _drink_tower(p, *grand, 8, 8, 26, 0)
    g_left, g_front = _footing(*grand, 8 * m, 8 * m)[:2]
    _drink_clock(p, _drink_up(*g_left, 26), _drink_up(*g_front, 26), 0.5, 5.4 * m, 1.5)
    _drink_cupola(p, apex[0], apex[1] + 1.2 * m, 7)
    _brew_sign(p, *_on_wall(house[0], house[1], 0.012, 2.4 * m))
    for k, rows in enumerate(((9, 8, 7), (8, 7, 6))):
        _cask_stack(p, *at(31 + 1.4 * k, -0.2 - 2.4 * k), rows)
    _brew_dray(p, *at(-12, -1.6))
    _brew_dray(p, *at(-1, -4.2))
    _brew_dray(p, *at(12, -2.0))
    _drink_siding(p, *at(-12, -9.5), 55)
    for u in (-10, -2.8, 4.4, 20):
        _drink_cask_wagon(p, *at(u, -9.5))
    return p.sprite()


def big_distillery() -> Sprite:
    """A distillery grown into a works, whitewashed over tarred plinths under slate: a malt kiln
    crowned with a pagoda ventilator, a long malt barn with a black board lettered in gold, a
    still house with its amber doors open on three onion-shaped copper stills and worm tubs at
    its end, a low bonded warehouse with numbered black doors, a peat yard of neat stacks, and
    oak casks in long pyramids in the gravel yard."""
    m = METRE
    p = Painter(62, 50, (28, 32))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 7) * m, (v - 2) * m)

    p.polygon([at(-15, -9), at(31, -9), at(31, 3), at(-15, 3)], GRAVEL)
    p.shadow(20, 6, 5)
    _drink_kiln(p, *at(-9, 9), 6, 8, WHITEWASH, STILL_SLATE, WHISKY, pagoda=True)
    _drink_malt_barn(p, *at(3, 8.5), 18, 7, 7)
    _drink_still_house(p, *at(19, 7), 14, 8, 7.5, 3)
    for v in (4.6, 8.6):
        _still_worm(p, *at(27.8, v))
    _drink_bond(p, *at(-3, -3), 22, 7, 5.5, (1, 2))
    for u, v in ((11, -5.5), (14.4, -7.4)):
        _still_peat(p, *at(u, v))
    _cask_stack(p, *at(19, -2.2), (6, 5, 4))
    _cask_stack(p, *at(22, -6.4), (5, 4))
    for u, v in ((26.6, -3.4), (27.4, -4.4), (13.6, -1.6)):
        _cask(p, *at(u, v), "up")
    return p.sprite()


def mega_distillery() -> Sprite:
    """A great distillery whitewashed over tarred plinths under slate: twin malt kilns crowned
    with pagoda ventilators over a long malt barn lettered in gold, a grand still house whose
    amber doors stand open on a row of four onion-shaped copper stills with worm tubs at its end,
    two bonded warehouses with numbered black doors closing the yard, a peat yard of neat stacks
    and long pyramids of oak casks waiting on the gravel."""
    m = METRE
    p = Painter(72, 56, (32, 34))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 8) * m, (v - 2) * m)

    p.polygon([at(-15, -9.5), at(39, -9.5), at(39, 10), at(-15, 1)], GRAVEL)
    p.shadow(24, 7, 6)
    for u in (-12, -5):
        _drink_kiln(p, *at(u, 12), 6.5, 9, WHITEWASH, STILL_SLATE, WHISKY, pagoda=True)
    _drink_malt_barn(p, *at(0, 9), 16, 8, 8)
    _drink_still_house(p, *at(17, 8), 18, 9, 9, 4)
    for v in (5.4, 10.4):
        _still_worm(p, *at(27.6, v))
    _drink_bond(p, *at(-5, -4.5), 20, 8, 6, (1, 2))
    for u, v in ((9, -6.6), (12.4, -8.0), (9.6, -3.8)):
        _still_peat(p, *at(u, v))
    _cask_stack(p, *at(15.5, -1.6), (8, 7, 6))
    _cask_stack(p, *at(17.5, -6.2), (7, 6, 5))
    for u, v in ((28.6, -3.6), (29.4, -4.6), (27.4, -5.8)):
        _cask(p, *at(u, v), "up")
    _drink_bond(p, *at(35.5, 1), 7, 18, 6, (3,), shaded=True)
    return p.sprite()


BALLAST: RGB = (138, 130, 120)

RAIL: RGB = (90, 88, 92)

SLEEPER: RGB = (96, 70, 50)


def _rails(p: Painter, start: Point, end: Point) -> None:
    """A railway siding on gravel ballast from `start` to `end` along the ground, laid on
    sleepers."""
    (x0, y0), (x1, y1) = start, end
    gauge = (1.4, 0.7) if (x1 - x0) * (y1 - y0) < 0 else (-1.4, 0.7)
    p.polygon(
        [
            (x0 - gauge[0] * 0.6, y0 - gauge[1] * 0.6),
            (x1 - gauge[0] * 0.6, y1 - gauge[1] * 0.6),
            (x1 + gauge[0] * 1.6, y1 + gauge[1] * 1.6),
            (x0 + gauge[0] * 1.6, y0 + gauge[1] * 1.6),
        ],
        BALLAST,
    )
    steps = round(abs(x1 - x0) / 2)
    for k in range(steps + 1):
        sx, sy = x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps
        p.line(
            [
                (sx - gauge[0] * 0.3, sy - gauge[1] * 0.3),
                (sx + gauge[0] * 1.3, sy + gauge[1] * 1.3),
            ],
            SLEEPER,
        )
    for g in (0, 1):
        dx, dy = gauge[0] * g, gauge[1] * g
        p.line([(x0 + dx, y0 + dy), (x1 + dx, y1 + dy)], RAIL)


def _wagon(p: Painter, x: float, y: float, length: float, colour: RGB, roof: RGB) -> Top:
    """A goods wagon standing on rails running back to the right, centred on (x, y): a box body
    on dark wheels, its long side shaded."""
    for t in (0.2, 0.8):
        wx, wy = x - length / 2 + length * t, y + length / 4 - length * t / 2
        p.ellipse(wx - 0.6, wy - 1.6, wx + 1.4, wy + 0.4, (40, 38, 40))
    top = p.hall(x, y - 1.2, 3, length, 4.5, colour, roof)
    _, front, right, _ = top
    _band(p, front, right, 2.2, shade(colour, 0.6))
    _patch(p, front, right, (0.4, 0.6), (0.6, 4.2), shade(colour, 0.65))
    return top


type At3 = Callable[[float, float, float], Point]

FOOD_VAN: RGB = (122, 64, 50)


def _food_corrugate(p: Painter, a: Point, b: Point, c: Point, d: Point, ribs: int) -> None:
    """Ribs of corrugated iron across a roof slope from its eave `a`-`b` up to its crest `d`-`c`."""
    m = METRE
    for k in range(1, ribs):
        t = k / ribs
        _bar(p, _on_wall(a, b, t, 0), _on_wall(d, c, t, 0), 0.08 * m, shade(FOOD_IRON, 0.84))


def _food_walls(p: Painter, top: Top, high: float) -> None:
    """Dress a cannery's cream render: a red-brick plinth and cornice with a cream band under it,
    a brick string course and brick quoins at its corners, on its lit and shaded walls."""
    m = METRE
    left, front, right, _ = top
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (high - 0.9 * m, high), shade(FOOD_BRICK, 0.92 * tone))
        _patch(p, a, b, (0, 1), (0, 0.55 * m), shade(FOOD_BRICK, tone))
        _patch(p, a, b, (0, 1), (0.55 * m, 0.75 * m), shade(FOOD_CREAM, tone))
        _patch(p, a, b, (0, 1), (2.85 * m, 3.05 * m), shade(FOOD_BRICK, tone))
        for t in (0.012, 0.988):
            _bar(
                p,
                _on_wall(a, b, t, 0.5 * m),
                _on_wall(a, b, t, high),
                0.45 * m,
                shade(FOOD_BRICK, tone),
            )
    _stains(p, left, front, high, (0.05, 0.29, 0.62, 0.96), shade(FOOD_RENDER, 0.72), 1.4 * m)


def _food_hall(
    p: Painter,
    x: float,
    y: float,
    length: float,
    depth: float,
    high: float,
    cart_door: int | None,
    sign: bool,
) -> Top:
    """A cannery hall `length` by `depth` m with walls `high` m, centred on (x, y), its long lit
    wall in bays 3 m wide between red-brick pilasters: rendered cream over a red-brick plinth,
    tall round-headed windows framed in brick, a round-headed cart door with green leaves folded
    back in bay `cart_door`, a painted name board over its middle if `sign`, windows down its
    shaded gable end, and a silver corrugated-iron roof with a glazed clerestory along its
    ridge."""
    m = METRE
    top = p.hall(x, y, length * m, depth * m, high * m, FOOD_RENDER, None)
    left, front, right, _ = top
    _food_walls(p, top, high * m)
    bays = round(length / 3)
    head, foot = 3.7 * m, (high - 1.4) * m
    for k in range(1, bays):
        _bar(
            p,
            _on_wall(left, front, k / bays, 3.05 * m),
            _on_wall(left, front, k / bays, (high - 0.9) * m),
            0.4 * m,
            FOOD_BRICK,
        )
    for k in range(bays):
        t = (k + 0.5) / bays
        if k == cart_door:
            p.polygon(
                _round_head(left, front, length * m, t, 3.4 * m, 4.1 * m, high * m), FOOD_BRICK
            )
            p.polygon(_round_head(left, front, length * m, t, 2.8 * m, 4.4 * m, high * m), DOORWAY)
            for s in (t - 1.55 / length, t + 1.25 / length):
                _patch(
                    p, left, front, (s, s + 0.3 / length), (5.6 * m, (high - 0.05) * m), FOOD_GREEN
                )
            continue
        _food_window(p, left, front, length * m, t, 1.6 * m, (head, foot))
    panes = round(depth / 4)
    for k in range(panes):
        _food_window(
            p,
            front,
            right,
            depth * m,
            (k + 0.5) / panes,
            1.6 * m,
            (head, foot),
            shade(MILL_GLASS, 0.85),
        )
    if sign:
        _food_sign(p, left, front, length * m, 0.5, 0.45 * m)
    rise = 3.4 * m
    p.gable(top, rise, FOOD_IRON, FOOD_RENDER)
    for along in np.linspace(0.015, 0.985, round(length * 2)):
        _bar(
            p,
            _on_roof(top, rise, along, 0),
            _on_roof(top, rise, along, 1),
            0.08 * m,
            shade(FOOD_IRON, 0.84),
        )
    _bar(p, _on_roof(top, rise, 0, 0), _on_roof(top, rise, 1, 0), 0.16 * m, shade(FOOD_IRON, 0.7))
    ridge_back, ridge_front = _ridge(top, rise)
    rx, ry = _on_wall(ridge_back, ridge_front, 0.5, 0)
    lantern = p.hall(rx, ry + 0.75 * m, (length - 4) * m, 2.6 * m, 2.1 * m, FOOD_CREAM, None)
    l_left, l_front, l_right, _ = lantern
    for a, b, tone, n in ((l_left, l_front, 1.0, round(length)), (l_front, l_right, 0.78, 3)):
        _patch(p, a, b, (0.02, 0.98), (0.25 * m, 1.3 * m), shade(NORTH_LIGHT, 0.8 * tone))
        for s in np.linspace(0.02, 0.98, n):
            _bar(
                p,
                _on_wall(a, b, s, 0.25 * m),
                _on_wall(a, b, s, 1.3 * m),
                0.07 * m,
                shade(FOOD_CREAM, tone),
            )
    p.gable(lantern, 0.9 * m, FOOD_IRON, FOOD_CREAM)
    return top


def _food_gable_hall(
    p: Painter, x: float, y: float, wide: float, long: float, high: float, sign: bool
) -> Top:
    """A cannery hall `wide` m across and `long` m deep with walls `high` m, centred on (x, y),
    its gable end turned to the lit side: rendered cream with red-brick plinth, cornice, quoins
    and coped gable, two tall round-headed windows either side of a tall arched loading door
    with a hatch over it, a round window in the gable and a painted name board under it if
    `sign`, round-headed windows down its shaded side, and a corrugated-iron roof with glazed
    lights."""
    m = METRE
    top = p.hall(x, y, wide * m, long * m, high * m, FOOD_RENDER, None)
    left, front, right, back = top
    _food_walls(p, top, high * m)
    head, foot = 3.7 * m, (high - 1.4) * m
    for t in (0.17, 0.83):
        _food_window(p, left, front, wide * m, t, 1.5 * m, (head, foot))
    door = (high - 4.6) * m
    p.polygon(
        _round_head(left, front, wide * m, 0.5, 3.6 * m, door - 0.3 * m, high * m), FOOD_BRICK
    )
    p.polygon(_round_head(left, front, wide * m, 0.5, 3.0 * m, door, high * m), FOOD_GREEN)
    _bar(
        p,
        _on_wall(left, front, 0.5, door + 1.5 * m),
        _on_wall(left, front, 0.5, high * m),
        0.06 * m,
        shade(FOOD_GREEN, 0.6),
    )
    hatch = (0.5 - 0.8 / wide, 0.5 + 0.8 / wide)
    _patch(
        p,
        left,
        front,
        (hatch[0] - 0.25 / wide, hatch[1] + 0.25 / wide),
        (head - 0.45 * m, head + 1.2 * m),
        FOOD_BRICK,
    )
    _patch(p, left, front, hatch, (head - 0.2 * m, head + 1.0 * m), DOORWAY)
    bays = round(long / 3)
    for k in range(bays):
        _food_window(
            p,
            front,
            right,
            long * m,
            (k + 0.5) / bays,
            1.4 * m,
            (head, foot),
            shade(MILL_GLASS, 0.85),
        )
    rise = wide * 0.32 * m
    near, far = _on_wall(left, front, 0.5, -rise), _on_wall(back, right, 0.5, -rise)
    p.polygon([left, back, far, near], shade(FOOD_IRON, 1.04), shade(FOOD_IRON, 0.6))
    p.polygon([front, right, far, near], shade(FOOD_IRON, 0.76), shade(FOOD_IRON, 0.5))
    _food_corrugate(p, front, right, far, near, round(long * 2))
    _food_corrugate(p, left, back, far, near, round(long * 2))
    for t0 in (0.15, 0.45, 0.75):
        p.polygon(
            [
                _on_wall(_on_wall(front, near, 0.3, 0), _on_wall(right, far, 0.3, 0), t0, 0),
                _on_wall(_on_wall(front, near, 0.3, 0), _on_wall(right, far, 0.3, 0), t0 + 0.12, 0),
                _on_wall(
                    _on_wall(front, near, 0.75, 0), _on_wall(right, far, 0.75, 0), t0 + 0.12, 0
                ),
                _on_wall(_on_wall(front, near, 0.75, 0), _on_wall(right, far, 0.75, 0), t0, 0),
            ],
            shade(NORTH_LIGHT, 0.72),
        )
    p.polygon([left, front, near], FOOD_RENDER, shade(FOOD_RENDER, 0.6))
    for a, b in ((left, near), (near, front)):
        _bar(p, a, b, 0.4 * m, FOOD_BRICK)
    _bar(p, near, far, 0.2 * m, shade(FOOD_IRON, 1.15))
    gx, gy = _on_wall(left, front, 0.5, -rise * 0.45)
    ring = [
        (gx + 1.0 * m * math.cos(a), gy + 0.5 * m * math.cos(a) - 1.0 * m * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)
    ]
    p.polygon(ring, FOOD_BRICK)
    p.polygon([(gx + (px - gx) * 0.7, gy + (py - gy) * 0.7) for px, py in ring], MILL_GLASS)
    if sign:
        _food_sign(p, left, front, wide * m, 0.5, 0.2 * m)
    return top


def _food_can_shop(p: Painter, x: float, y: float, length: float, depth: float) -> Top:
    """The tin shop where the cans are made, `length` by `depth` m and 5.5 m to its eaves, centred
    on (x, y): red brick banded in cream with small round-headed windows and a wide door, under
    a north-light roof of glazed teeth."""
    m = METRE
    high = 5.5 * m
    top = p.hall(x, y, length * m, depth * m, high, FOOD_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, FOOD_BRICK, 0.6 * m)
    _bands(p, left, front, right, 0.3 * m, FOOD_CREAM, shade(FOOD_CREAM, 0.78))
    bays = round(length / 3)
    for k in range(bays):
        t = (k + 0.5) / bays
        if k == bays - 1:
            _patch(p, left, front, (t - 0.9 / length, t + 0.9 / length), (2.0 * m, high), DOORWAY)
            _patch(
                p,
                left,
                front,
                (t - 1.0 / length, t + 1.0 / length),
                (1.75 * m, 2.0 * m),
                FOOD_CREAM,
            )
            continue
        _food_window(p, left, front, length * m, t, 1.2 * m, (1.4 * m, 4.0 * m))
    _food_window(
        p, front, right, depth * m, 0.5, 1.2 * m, (1.4 * m, 4.0 * m), shade(MILL_GLASS, 0.85)
    )
    _north_lights(p, top, round(depth / 3), 1.8 * m, MILL_SLATE, FOOD_BRICK)
    return top


def _food_office(p: Painter, x: float, y: float, length: float, depth: float) -> Top:
    """The company's office block, `length` by `depth` m and three storeys, 12 m to its cornice,
    centred on (x, y): cream render over a rusticated red-brick ground floor, brick quoins and
    string courses, rows of sash windows, a pedimented green door, the company's long name board
    across its front and a hipped slate roof with a gilt clock turret."""
    m = METRE
    high = 12 * m
    top = p.hall(x, y, length * m, depth * m, high, FOOD_RENDER, None)
    left, front, right, back = top
    for a, b, tone, n in ((left, front, 1.0, round(length / 2.4)), (front, right, 0.78, 3)):
        brick = shade(FOOD_BRICK, tone)
        _patch(p, a, b, (0, 1), (high - 4 * m, high), brick)
        for d in np.arange(8.4, 11.9, 0.7):
            _bar(p, _on_wall(a, b, 0, d * m), _on_wall(a, b, 1, d * m), 0.05 * m, shade(brick, 0.8))
        _patch(p, a, b, (0, 1), (0, 0.7 * m), brick)
        _patch(p, a, b, (0, 1), (0.7 * m, 0.9 * m), shade(FOOD_CREAM, tone))
        _patch(p, a, b, (0, 1), (4.0 * m, 4.25 * m), brick)
        for t in (0.015, 0.985):
            _bar(p, _on_wall(a, b, t, 0.7 * m), _on_wall(a, b, t, high - 4 * m), 0.5 * m, brick)
        span = length if n > 3 else depth
        glass = MILL_GLASS if tone == 1.0 else shade(MILL_GLASS, 0.85)
        for k in range(n):
            t = (k + 0.5) / n
            for head, foot in ((1.3, 3.4), (4.9, 7.1)):
                _food_window(p, a, b, span * m, t, 1.1 * m, (head * m, foot * m), glass)
            if not (tone == 1.0 and k == n // 2):
                _food_window(p, a, b, span * m, t, 1.1 * m, (8.8 * m, 10.9 * m), glass)
    n_lit = round(length / 2.4)
    t = (n_lit // 2 + 0.5) / n_lit
    p.polygon(
        [
            _on_wall(left, front, t - 1.4 / length, 8.3 * m),
            _on_wall(left, front, t + 1.4 / length, 8.3 * m),
            _on_wall(left, front, t, 7.3 * m),
        ],
        FOOD_CREAM,
    )
    _patch(p, left, front, (t - 1.2 / length, t + 1.2 / length), (8.3 * m, 8.6 * m), FOOD_CREAM)
    _patch(p, left, front, (t - 0.8 / length, t + 0.8 / length), (8.6 * m, high), FOOD_GREEN)
    _food_sign(p, left, front, length * m, 0.5, 4.3 * m)
    rise = 3.2 * m
    run = depth / 2
    r_left, r_front, r_right, r_back = left, front, right, back
    ridge_back = _iso(*_on_wall(r_left, r_back, 0.5, -rise), run * m, 0)
    ridge_front = _iso(*_on_wall(r_front, r_right, 0.5, -rise), -run * m, 0)
    p.polygon([r_back, r_right, ridge_front, ridge_back], shade(MILL_SLATE, 0.7))
    p.polygon([r_left, r_back, ridge_back], shade(MILL_SLATE, 0.9))
    p.polygon([r_front, r_right, ridge_front], shade(MILL_SLATE, 0.74), shade(MILL_SLATE, 0.5))
    p.polygon([r_left, r_front, ridge_front, ridge_back], MILL_SLATE, shade(MILL_SLATE, 0.6))
    _bar(p, ridge_back, ridge_front, 0.18 * m, shade(MILL_SLATE, 1.3))
    for t_c in (0.15, 0.85):
        cx, cy = _on_wall(ridge_back, ridge_front, t_c, 0)
        p.hall(cx, cy + 0.6 * m, 0.9 * m, 0.9 * m, 2.0 * m, FOOD_BRICK, (50, 44, 44))
    cx, cy = _on_wall(ridge_back, ridge_front, 0.5, 0)
    turret = p.hall(cx, cy + 0.5 * m, 2.2 * m, 2.2 * m, 2.6 * m, FOOD_CREAM, None)
    fx, fy = _on_wall(turret[0], turret[1], 0.5, 1.2 * m)
    p.ellipse(fx - 0.7 * m, fy - 0.7 * m, fx + 0.7 * m, fy + 0.7 * m, (248, 246, 236))
    _bar(p, (fx, fy), (fx, fy - 0.5 * m), 0.1 * m, (30, 30, 34))
    _bar(p, (fx, fy), (fx + 0.35 * m, fy + 0.1 * m), 0.1 * m, (30, 30, 34))
    apex = (cx, turret[0][1] - 2.0 * m)
    p.polygon([turret[0], turret[1], apex], shade(FOOD_GREEN, 1.2))
    p.polygon([turret[1], turret[2], apex], FOOD_GREEN)
    _bar(p, apex, (apex[0], apex[1] - 0.8 * m), 0.1 * m, FOOD_GILT)
    return top


def _food_belt(
    p: Painter, at: At3, start: tuple[float, float, float], end: tuple[float, float, float]
) -> None:
    """A belt conveyor 0.9 m wide carrying a file of bright silver tins from `start` down to
    `end`, each a point (u, v, z) in metres that `at` puts on the sprite, running along one of
    the two ground axes on a steel frame held by braced trestles."""
    m = METRE
    (u0, v0, z0), (u1, v1, z1) = start, end
    along_u = abs(u1 - u0) >= abs(v1 - v0)
    run = math.hypot(u1 - u0, v1 - v0)

    def point(s: float, w: float, lift: float = 0, ground: bool = False) -> Point:
        u, v = u0 + (u1 - u0) * s, v0 + (v1 - v0) * s
        if along_u:
            v += w
        else:
            u += w
        return at(u, v, 0 if ground else z0 + (z1 - z0) * s + lift)

    for s in (0.42, 0.84):
        for w in (0.4, -0.4):
            _bar(p, point(s, w, ground=True), point(s, w, -0.3), 0.14 * m, shade(FOOD_STEEL, 0.7))
        _bar(p, point(s, -0.4, ground=True), point(s, 0.4, -0.6), 0.07 * m, shade(FOOD_STEEL, 0.6))
    p.polygon([point(0, -0.45), point(1, -0.45), point(1, 0.45), point(0, 0.45)], (66, 64, 66))
    for s in np.arange(0.2 / run, 1, 0.6 / run):
        _bar(p, point(s, -0.45), point(s, 0.45), 0.05 * m, (96, 94, 96))
    side, tone = (-0.45, 1.0) if along_u else (0.45, 0.78)
    p.polygon(
        [point(0, side), point(1, side), point(1, side, -0.3), point(0, side, -0.3)],
        shade(FOOD_STEEL, tone),
    )
    for s in np.arange(0.35 / run, 1 - 0.15 / run, 0.48 / run):
        cx, cy = point(s, 0)
        half, tall = 0.17 * m, 0.36 * m
        p.polygon(
            [
                (cx - half, cy),
                (cx, cy + half / 2),
                (cx, cy + half / 2 - tall),
                (cx - half, cy - tall),
            ],
            FOOD_TIN,
        )
        p.polygon(
            [
                (cx, cy + half / 2),
                (cx + half, cy),
                (cx + half, cy - tall),
                (cx, cy + half / 2 - tall),
            ],
            shade(FOOD_TIN, 0.74),
        )
        p.blob(
            cx - half, cy - tall - half / 2, cx + half, cy - tall + half / 2, shade(FOOD_TIN, 1.12)
        )


def _food_van(p: Painter, x: float, y: float) -> None:
    """A covered goods van 7 m long on the siding at (x, y), along the lit walls: red-oxide
    planking under an arched grey roof on four spoked wheels, its sliding door run back on the
    lit side showing cases of labelled tins stacked inside, lettered in white."""
    m = METRE

    def at(s: float, z: float, w: float = 0) -> Point:
        px, py = _iso(x, y, s * m, w * m)
        return px, py - z * m

    for s in (-2.2, 2.2):
        _food_wheel(p, *at(s, 0.48, 0.75), 0.48 * m, (70, 66, 66))
    left, front, right, back = p.hall(*at(0, 0.9), 7 * m, 2.6 * m, 2.6 * m, FOOD_VAN, None)
    for t in np.linspace(0.04, 0.96, 14):
        _bar(
            p,
            _on_wall(left, front, t, 0.1 * m),
            _on_wall(left, front, t, 2.5 * m),
            0.04 * m,
            shade(FOOD_VAN, 0.8),
        )
    for t in (0.015, 0.985):
        _bar(
            p,
            _on_wall(left, front, t, 0),
            _on_wall(left, front, t, 2.6 * m),
            0.14 * m,
            (52, 48, 48),
        )
    _bar(
        p,
        _on_wall(left, front, 0, 2.6 * m),
        _on_wall(left, front, 1, 2.6 * m),
        0.24 * m,
        (52, 48, 48),
    )
    _patch(p, left, front, (0.36, 0.64), (0.35 * m, 2.5 * m), (40, 34, 32))
    for k, t in enumerate((0.39, 0.47, 0.55)):
        _patch(p, left, front, (t, t + 0.07), (1.3 * m, 2.5 * m), FOOD_CASE)
        _patch(p, left, front, (t + 0.01, t + 0.06), (1.65 * m, 2.1 * m), FOOD_LABELS[k])
    _patch(p, left, front, (0.66, 0.94), (0.3 * m, 2.5 * m), shade(FOOD_VAN, 0.9))
    for t0, t1 in ((0.08, 0.16), (0.18, 0.3)):
        _bar(
            p,
            _on_wall(left, front, t0, 0.8 * m),
            _on_wall(left, front, t1, 0.8 * m),
            0.18 * m,
            (236, 232, 220),
        )
    p.gable((left, front, right, back), 0.45 * m, (150, 150, 152), FOOD_VAN)
    for s in (-2.2, 2.2):
        _food_wheel(p, *at(s, 0.48, -0.9), 0.48 * m, (70, 66, 66))


def _food_heap(p: Painter, x: float, y: float, r: float, colours: tuple[RGB, ...]) -> None:
    """A heap of fruit `r` m across its foot tipped out on the ground at (x, y), mounded in rings
    of fruit in `colours` by turns, each lit on its top."""
    m = METRE
    p.blob(x - r * m, y - r * m * 0.5, x + r * m, y + r * m * 0.5, shade(colours[0], 0.55))
    n = 0
    for level in range(4):
        ring = max(0.0, r * (1 - level * 0.28) - 0.2)
        lift = level * 0.26 * r
        count = round(ring * 4)
        for k in range(count + 1):
            a = math.pi * k / count if count else math.pi / 2
            fx = x + ring * m * math.cos(a)
            fy = y + ring * m * 0.5 * math.sin(a) - lift * m
            colour = colours[n % len(colours)]
            n += 1
            p.ellipse(fx - 0.2 * m, fy - 0.2 * m, fx + 0.2 * m, fy + 0.16 * m, colour)
            p.blob(fx - 0.13 * m, fy - 0.16 * m, fx, fy - 0.04 * m, shade(colour, 1.25))


def _food_fruit(
    p: Painter, on_bed: Callable[[float, float], Point], layers: tuple[RGB, ...]
) -> None:
    """A cart's load of fruit heaped on its bed in bands of `layers`' colours from back to
    front."""
    m = METRE
    for z, spots, w_spots in (
        (0.18, np.arange(-1.4, 1.45, 0.4), (0.52, 0.17, -0.18, -0.53)),
        (0.42, np.arange(-1.0, 1.05, 0.4), (0.3, -0.05, -0.4)),
    ):
        for w in w_spots:
            for k, s in enumerate(spots):
                cx, cy = on_bed(s, w)
                cy -= z * m
                band = min(len(layers) - 1, int((s + 1.5) / 3.0 * len(layers)))
                colour = shade(layers[band], 1.0 if (k + round(w * 3)) % 2 else 0.88)
                p.ellipse(cx - 0.21 * m, cy - 0.21 * m, cx + 0.21 * m, cy + 0.17 * m, colour)
                p.blob(cx - 0.13 * m, cy - 0.15 * m, cx, cy - 0.03 * m, shade(colour, 1.22))


def big_food_plant() -> Sprite:
    """A big cannery: a long cream cannery hall with a clerestory and a painted name board, a
    taller second hall turning its gabled front to the yard, and beside them the brick tin shop
    under its north lights where the cans are made, with bundles of bright tinplate at its door.
    A row of steel retorts steams in the yard; a long belt carries a file of tins from the second
    hall over the yard to a van at the siding, where wagons of fruit wait, a farm cart brings
    more, and behind it all stands a big water tank on a steel trestle, made and painted like a
    giant tin can."""
    m = METRE
    p = Painter(55, 35, (24, 19))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 15) * m, (v - 6) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon(
        [at(-11, 6), at(-11, 18), at(36, 23), at(40, 19), at(40, -8.5), at(-1, -8.5)],
        FOOD_SETTS,
    )
    for cu, cv, lu, dv in ((-5.5, 12.5, 9, 9), (10, 11, 20, 12), (28, 12, 12, 18)):
        _bed(p, _footing(*at(cu, cv), lu * m, dv * m))
    p.shadow(20, 6, 6)
    _food_can_shop(p, *at(-5.5, 12.5), 9, 9)
    _food_hall(p, *at(10, 11), 20, 12, 9, 1, True)
    _food_gable_hall(p, *at(28, 12), 12, 18, 11, False)
    _food_tank(p, *at(37.6, 19.4), 1.3, 10)
    _food_tinplate(p, *at(-8.2, 5.6), 3)
    _food_tinplate(p, *at(-5.6, 6.4), 2)
    _food_tinplate(p, *at(-3.2, 5.2), 2)
    for k, v in enumerate((3.0, 0.8, -1.4)):
        _food_retort(p, *at(12.6, v), 6, 0.9, k != 1)
    for v in (3.0, 0.8, -1.4):
        _bar(p, up(14.2, v, 2.2), up(14.2, v, 2.95), 0.12 * m, (78, 80, 88))
    _bar(p, up(14.2, -2.2, 2.95), up(14.2, 4.9, 2.95), 0.2 * m, (78, 80, 88))
    _food_belt(p, up, (28, 2.9, 6.8), (28, -5.4, 3.9))
    _sugar_siding(p, at(4, -6.6), at(40, -6.6))
    for u in (8.6, 15.4):
        _sugar_wagon(p, *at(u, -6.6), (FOOD_APPLE, FOOD_GREEN_APPLE, FOOD_PEACH))
    _food_van(p, *at(27.6, -6.6))
    _food_cases(p, *up(32.2, -4.4, 0), (2, 1), 1)

    def fruit(on_bed: Callable[[float, float], Point]) -> None:
        _food_fruit(p, on_bed, (FOOD_APPLE, FOOD_PEACH, FOOD_CHERRY))

    for u, v, load in ((0.4, -0.6, FOOD_CHERRY), (2.0, 0.6, FOOD_PEACH), (5.2, 0.2, FOOD_APPLE)):
        _food_basket(p, *at(u, v), load)
    _food_cart(p, *at(1.4, -2.6), 3.4, 0.5, FOOD_SETTS, fruit)
    return p.sprite()


def mega_food_plant() -> Sprite:
    """A great cannery quarter: three long cream halls side by side turn their coped gables to a
    loading dock, the middle one tallest and bearing the company's name, and beside them stands
    the company's three-storey office with its long name board and a gilt clock turret. A belt
    brings tins out of the middle hall to the dock, where cases of labelled tins are stacked for
    the vans waiting on the siding beside wagons of fruit just come in; farm carts bring more and
    tip it out in heaps, and over the whole quarter towers its landmark, the water tank on a tall
    steel trestle, made and painted like a giant tin can."""
    m = METRE
    p = Painter(62, 41, (29, 23))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 14) * m, (v - 7) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon(
        [at(-14, -1), at(-14, 10), at(0, 25), at(36, 25), at(40, 21), at(40, -11), at(-3, -11)],
        FOOD_SETTS,
    )
    for k in range(3):
        _bed(p, _footing(*at(5.5 + 12 * k, 14), 11 * m, 20 * m))
    _bed(p, _footing(*at(-6.6, 4), 11.4 * m, 9 * m))
    p.shadow(24, 7, 6)
    for k, high in enumerate((9.0, 11.0, 9.0)):
        _food_gable_hall(p, *at(5.5 + 12 * k, 14), 11, 20, high, k == 1)
    _food_tank(p, *at(38.6, 20.4), 1.7, 18)
    _food_office(p, *at(-6.6, 4), 11.4, 9)
    dock = p.hall(
        *at(17.5, 2.6), 35 * m, 2.6 * m, 1.1 * m, shade(FOOD_BRICK, 0.9), shade(TIMBER, 1.15)
    )
    for s in np.linspace(0.02, 0.98, 36):
        _bar(
            p,
            _on_wall(dock[0], dock[3], s, 0),
            _on_wall(dock[1], dock[2], s, 0),
            0.05 * m,
            shade(TIMBER, 0.85),
        )
    for u, rows, first in (
        (2.5, (3, 2, 1), 0),
        (9.5, (2, 1), 1),
        (24.0, (3, 2), 2),
        (30.5, (2, 2, 1), 0),
    ):
        _food_cases(p, *up(u, 2.2, 1.1), rows, first)
    _food_belt(p, up, (17.5, 3.9, 7.0), (17.5, -0.4, 4.1))
    _sugar_siding(p, at(1, -1.2), at(41, -1.2))
    for u, load in ((4.2, (FOOD_APPLE, FOOD_GREEN_APPLE)), (11.0, (FOOD_PEACH, FOOD_CHERRY))):
        _sugar_wagon(p, *at(u, -1.2), load)
    _food_van(p, *at(17.9, -1.2))
    _food_van(p, *at(26.0, -1.2))
    for u, v, r, colours in (
        (2.5, -6.5, 1.8, (FOOD_APPLE, FOOD_GREEN_APPLE)),
        (8.0, -8.5, 1.5, (FOOD_PEACH,)),
        (13.0, -6.0, 1.3, (FOOD_CHERRY, FOOD_APPLE)),
    ):
        _food_heap(p, *at(u, v), r, colours)

    def fruit(on_bed: Callable[[float, float], Point]) -> None:
        _food_fruit(p, on_bed, (FOOD_PEACH, FOOD_APPLE, FOOD_GREEN_APPLE))

    def cherries(on_bed: Callable[[float, float], Point]) -> None:
        _food_fruit(p, on_bed, (FOOD_CHERRY, FOOD_TOMATO, FOOD_CHERRY))

    for u, v, load in (
        (-3.2, -4.6, FOOD_CHERRY),
        (-2.2, -5.6, FOOD_PEACH),
        (16.4, -5.2, FOOD_APPLE),
    ):
        _food_basket(p, *at(u, v), load)
    _food_cart(p, *at(-2.0, -8.2), 3.4, 0.5, FOOD_SETTS, fruit)
    _food_cart(p, *at(18.0, -8.6), 3.4, 0.5, FOOD_SETTS, cherries)
    return p.sprite()


SUGAR_RENDER: RGB = (238, 236, 228)

SUGAR_WATER: RGB = (78, 124, 150)


def _sugar_block(
    p: Painter,
    x: float,
    y: float,
    length: float,
    depth: float,
    high: float,
    door: int | None,
    board: tuple[float, float] | None,
    white: bool,
) -> Top:
    """A refinery block `length` by `depth` m and `high` m to its cornice, centred on (x, y), in
    bays 2.5 m wide: of warm red brick banded in white at every floor or, if `white`, rendered
    white and banded in sugarloaf blue; rows of sash windows in white surrounds, a ground floor
    rendered white with blue shutters, a blue name board across `board` of it, and in bay `door`
    a white-arched dispatch door with blue leaves and blue taking-in doors stacked over it up to
    a hoist in a hood on the eaves with a sack on its rope; a slate roof with a louvred white
    ventilator along its ridge."""
    m = METRE
    wall, band = (SUGAR_RENDER, SUGAR_BLUE) if white else (SUGAR_BRICK, SUGAR_WHITE)
    top = p.hall(x, y, length * m, depth * m, high * m, wall, None)
    left, front, right, _ = top
    _plinth(p, top, high * m, SUGAR_BLUE if white else wall, 1.0 * m, 0.6, 0.45)
    if not white:
        _stains(p, left, front, high * m, (0.03, 0.27, 0.52, 0.94), shade(wall, 0.7), 1.6 * m)
    ground = high - 4.4
    heads = [1.2 + 2.8 * k for k in range(int((ground - 2.0) / 2.8) + 1)]
    lit_bays = round(length / 2.5)
    for a, b, tone, bays, span in (
        (left, front, 1.0, lit_bays, length),
        (front, right, 0.8, round(depth / 2.4), depth),
    ):
        stripe = shade(band, tone)
        _patch(p, a, b, (0, 1), (0, 0.8 * m), stripe)
        _patch(p, a, b, (0, 1), (0.8 * m, 0.95 * m), shade(wall, 0.62 * tone))
        for h in heads[:-1]:
            _patch(p, a, b, (0, 1), ((h + 2.1) * m, (h + 2.32) * m), stripe)
        _patch(p, a, b, (0, 1), (ground * m, (high - 0.95) * m), shade(SUGAR_WHITE, 0.94 * tone))
        _patch(p, a, b, (0, 1), (ground * m, (ground + 0.25) * m), shade(band, 0.85 * tone))
        for k in range(bays):
            t = (k + 0.5) / bays
            if tone == 1.0 and k == door:
                continue
            for h in heads:
                _sugar_window(p, a, b, span * m, t, 1.1 * m, (h * m, (h + 1.5) * m))
            _sugar_window(p, a, b, span * m, t, 1.1 * m, ((high - 2.5) * m, (high - 1.0) * m))
            if tone == 1.0:
                for side in (-1, 1):
                    edge = t + side * 0.55 / span
                    leaf = side * 0.5 / span
                    _patch(
                        p,
                        a,
                        b,
                        (min(edge, edge + leaf), max(edge, edge + leaf)),
                        ((high - 2.7) * m, (high - 0.9) * m),
                        SUGAR_BLUE,
                    )
    if board is not None:
        _sugar_board(p, left, front, board, (ground + 0.15) * m, 1.15 * m)
    rise = depth * 0.375 * m
    p.gable(top, rise, MILL_SLATE, wall)
    _roof_rows(p, top, rise, MILL_SLATE, (0.25, 0.5, 0.75))
    ridge_back, ridge_front = _ridge(top, rise)
    vx, vy = _on_wall(ridge_back, ridge_front, 0.42, 0)
    vent = p.hall(vx, vy + 0.5 * m, length * 0.45 * m, 1.6 * m, 1.5 * m, SUGAR_WHITE, None)
    for a, b, tone in ((vent[0], vent[1], 1.0), (vent[1], vent[2], 0.8)):
        for k in range(1, 4):
            _bar(
                p,
                _on_wall(a, b, 0, k * 0.25 * m),
                _on_wall(a, b, 1, k * 0.25 * m),
                0.12 * m,
                shade(SUGAR_WHITE, 0.6 * tone),
            )
    p.gable(vent, 0.7 * m, MILL_SLATE, SUGAR_WHITE)
    if door is None:
        return top
    t = (door + 0.5) / lit_bays
    half = 1.45 / length
    for h in heads[1:]:
        _patch(
            p,
            left,
            front,
            (t - half * 0.5, t + half * 0.5),
            ((h - 0.12) * m, (h + 1.82) * m),
            SUGAR_WHITE,
        )
        _patch(p, left, front, (t - half * 0.4, t + half * 0.4), (h * m, (h + 1.8) * m), SUGAR_BLUE)
        _bar(
            p,
            _on_wall(left, front, t, h * m),
            _on_wall(left, front, t, (h + 1.8) * m),
            0.05 * m,
            shade(SUGAR_BLUE, 0.65),
        )
    p.polygon(
        _round_head(left, front, length * m, t, 2.7 * m, (ground + 1.0) * m, high * m), SUGAR_WHITE
    )
    p.polygon(
        _round_head(left, front, length * m, t, 2.2 * m, (ground + 1.25) * m, high * m), DOORWAY
    )
    for s in (t - 1.1 / length, t + 0.66 / length):
        _patch(
            p,
            left,
            front,
            (s, s + 0.44 / length),
            ((ground + 2.3) * m, (high - 0.05) * m),
            SUGAR_BLUE,
        )
    hx, hy = _on_wall(left, front, t, 0)
    hood = p.hall(hx - 0.4 * m, hy + 1.4 * m, 2.4 * m, 1.6 * m, 2.4 * m, wall, None)
    h_left, h_front, h_right, h_back = hood
    _patch(p, h_left, h_front, (0.15, 0.85), (0.8 * m, 2.4 * m), SUGAR_WHITE)
    _patch(p, h_left, h_front, (0.22, 0.78), (0.9 * m, 2.4 * m), SUGAR_BLUE)
    near, far = _on_wall(h_left, h_front, 0.5, -1.2 * m), _on_wall(h_back, h_right, 0.5, -1.2 * m)
    p.polygon([h_front, h_right, far, near], shade(MILL_SLATE, 0.72), shade(MILL_SLATE, 0.5))
    p.polygon([h_left, h_back, far, near], MILL_SLATE, shade(MILL_SLATE, 0.6))
    p.polygon([h_left, h_front, near], SUGAR_WHITE, shade(SUGAR_WHITE, 0.6))
    beam = _on_wall(h_left, h_front, 0.5, 0.7 * m)
    tip = (beam[0] - 1.4 * m, beam[1] + 0.7 * m)
    _bar(p, beam, tip, 0.22 * m, (54, 50, 50))
    _bar(p, tip, (tip[0], tip[1] + 3.4 * m), 0.06 * m, (60, 54, 50))
    sx, sy = tip[0], tip[1] + 4.4 * m
    p.ellipse(sx - 0.38 * m, sy - 1.1 * m, sx + 0.38 * m, sy + 0.1 * m, SUGAR_WHITE)
    _bar(p, (sx - 0.2 * m, sy - 0.6 * m), (sx + 0.2 * m, sy - 0.6 * m), 0.16 * m, SUGAR_BLUE)
    return top


def _sugar_char_house(p: Painter, x: float, y: float, side: float, high: float) -> Top:
    """A char house for the bone-char filters, a cream-brick tower `side` m square and `high` m to
    its eaves centred on (x, y): banded in red brick every five and a half metres with white
    corner strips, pairs of tall white-framed windows, louvred openings at its top, a blue door
    in a white arch, and a slate pyramid roof with a gilt ball on its point."""
    m = METRE
    tower = p.hall(x, y, side * m, side * m, high * m, SUGAR_CREAM, None)
    t_left, t_front, t_right, _ = tower
    _plinth(p, tower, high * m, SUGAR_CREAM, 1.0 * m)
    for a, b, tone in ((t_left, t_front, 1.0), (t_front, t_right, 0.8)):
        brick = shade(SUGAR_BRICK, tone)
        _patch(p, a, b, (0, 1), (0, 0.7 * m), shade(SUGAR_WHITE, tone))
        _patch(p, a, b, (0, 1), (0.7 * m, 0.85 * m), brick)
        for drop in np.arange(5.2, high - 3, 5.4):
            _patch(p, a, b, (0, 1), (drop * m, (drop + 0.35) * m), brick)
            if drop + 4.4 < high - 1.2:
                for s in (0.34, 0.66):
                    _sugar_window(
                        p, a, b, side * m, s, 1.0 * m, ((drop + 1.2) * m, (drop + 4.0) * m)
                    )
        for s in (0.03, 0.97):
            _bar(p, _on_wall(a, b, s, 0.85 * m), _on_wall(a, b, s, (high - 1) * m), 0.4 * m, brick)
        _patch(p, a, b, (0.2, 0.8), (1.4 * m, 4.4 * m), (52, 48, 52))
        for d in np.arange(1.7, 4.3, 0.45):
            _bar(
                p,
                _on_wall(a, b, 0.2, d * m),
                _on_wall(a, b, 0.8, d * m),
                0.12 * m,
                shade(SUGAR_WHITE, 0.9 * tone),
            )
    door = (high - 3.4) * m
    p.polygon(
        _round_head(t_left, t_front, side * m, 0.5, 2.2 * m, door - 0.3 * m, high * m), SUGAR_WHITE
    )
    p.polygon(_round_head(t_left, t_front, side * m, 0.5, 1.8 * m, door, high * m), SUGAR_BLUE)
    apex = ((t_left[0] + t_right[0]) / 2, (t_left[1] + t_right[1]) / 2 - side * 0.64 * m)
    p.polygon([t_left, t_front, apex], MILL_SLATE, shade(MILL_SLATE, 0.6))
    p.polygon([t_front, t_right, apex], shade(MILL_SLATE, 0.72), shade(MILL_SLATE, 0.5))
    _bar(p, apex, (apex[0], apex[1] - 1.6 * m), 0.12 * m, (50, 50, 56))
    p.blob(apex[0] - 0.3 * m, apex[1] - 1.9 * m, apex[0] + 0.3 * m, apex[1] - 1.3 * m, FOOD_GILT)
    return tower


def _sugar_store(p: Painter, x: float, y: float, length: float, depth: float, high: float) -> Top:
    """The sugarloaf store, a whitewashed warehouse `length` by `depth` m and `high` m to its
    eaves centred on (x, y), its lit front an arcade of round-headed openings where racks of
    sugarloaves in blue paper stand on two shelves, a blue plinth and cornice band, blue doors at
    its end and a slate roof."""
    m = METRE
    top = p.hall(x, y, length * m, depth * m, high * m, SUGAR_RENDER, None)
    left, front, right, _ = top
    for a, b, tone in ((left, front, 1.0), (front, right, 0.8)):
        _patch(p, a, b, (0, 1), (0, 0.6 * m), shade(SUGAR_BLUE, tone))
        _patch(p, a, b, (0, 1), ((high - 0.7) * m, high * m), shade(SUGAR_BLUE, 0.8 * tone))
    arches = round(length / 3)
    for k in range(arches):
        t = (k + 0.5) / arches
        p.polygon(
            _round_head(left, front, length * m, t, 2.5 * m, 1.6 * m, (high - 0.7) * m),
            shade(SUGAR_RENDER, 0.8),
        )
        p.polygon(
            _round_head(left, front, length * m, t, 2.1 * m, 1.8 * m, (high - 0.7) * m),
            (58, 52, 56),
        )
        for shelf in (high - 3.6, high - 1.3):
            _bar(
                p,
                _on_wall(left, front, t - 1.05 / length, shelf * m),
                _on_wall(left, front, t + 1.05 / length, shelf * m),
                0.12 * m,
                shade(TIMBER, 0.9),
            )
            for j in (-0.65, 0, 0.65):
                _sugar_loaf(p, *_on_wall(left, front, t + j / length, (shelf - 0.06) * m), 0.85 * m)
    for s in (0.3, 0.7):
        _patch(
            p,
            front,
            right,
            (s - 0.12, s + 0.12),
            ((high - 3.4) * m, (high - 0.6) * m),
            shade(SUGAR_BLUE, 0.8),
        )
    rise = depth * 0.35 * m
    p.gable(top, rise, MILL_SLATE, SUGAR_RENDER)
    _roof_rows(p, top, rise, MILL_SLATE)
    return top


def _sugar_sky_board(
    p: Painter, top: Top, depth: float, rise: float, span: tuple[float, float], tall: float
) -> None:
    """A great name board in sugarloaf blue lettered in white, `tall` m high, standing on steel
    legs on the lit slope of a roof over `top`, `depth` m deep with a ridge `rise` units high,
    from `span[0]` to `span[1]` of the way along it."""
    m = METRE
    left, front, _, _ = top
    back = 2.4
    lift = rise * back / (depth / 2)

    def shifted(p0: Point, z: float) -> Point:
        px, py = _iso(*p0, 0, back * m)
        return px, py - lift - z * m

    a, b = shifted(left, 0.7 + tall), shifted(front, 0.7 + tall)
    for t in np.linspace(span[0] + 0.04, span[1] - 0.04, 5):
        foot = _on_wall(shifted(left, 0), shifted(front, 0), t, 0)
        _bar(p, foot, _on_wall(a, b, t, tall * m), 0.16 * m, (56, 60, 70))
    _sugar_board(p, a, b, span, 0, tall * m)


def _sugar_flume(p: Painter, at: At3, start: tuple[float, float], end: tuple[float, float]) -> None:
    """A beet flume running from `start` to `end`, points (u, v) on the ground along one of the
    two ground axes that `at` puts on the sprite: a concrete trough 1.4 m wide with walls half a
    metre high, the water in it carrying washed sugar beet into the works."""
    m = METRE
    (u0, v0), (u1, v1) = start, end
    along_u = abs(u1 - u0) >= abs(v1 - v0)
    concrete = (196, 192, 182)

    def point(s: float, w: float, z: float) -> Point:
        u, v = u0 + (u1 - u0) * s, v0 + (v1 - v0) * s
        return at(u + (0 if along_u else w), v + (w if along_u else 0), z)

    far, near = (0.7, -0.7) if along_u else (-0.7, 0.7)
    p.polygon(
        [point(0, far, 0.5), point(1, far, 0.5), point(1, near, 0.5), point(0, near, 0.5)], concrete
    )
    p.polygon(
        [
            point(0, far * 0.75, 0.3),
            point(1, far * 0.75, 0.3),
            point(1, near * 0.75, 0.3),
            point(0, near * 0.75, 0.3),
        ],
        SUGAR_WATER,
    )
    p.polygon(
        [
            point(0, far * 0.75, 0.5),
            point(1, far * 0.75, 0.5),
            point(1, far * 0.75, 0.3),
            point(0, far * 0.75, 0.3),
        ],
        shade(concrete, 0.8),
    )
    run = math.hypot(u1 - u0, v1 - v0)
    for k, s in enumerate(np.arange(0.4 / run, 1, 0.55 / run)):
        bx, by = point(s, 0.25 * ((k * 7) % 3 - 1) * 0.6, 0.32)
        p.blob(
            bx - 0.22 * m,
            by - 0.13 * m,
            bx + 0.22 * m,
            by + 0.13 * m,
            shade(SUGAR_BEET, 1.0 + 0.1 * (k % 2)),
        )
    p.polygon(
        [point(0, near, 0), point(1, near, 0), point(1, near, 0.5), point(0, near, 0.5)],
        shade(concrete, 0.88 if along_u else 0.74),
    )


def big_sugar_refinery() -> Sprite:
    """A big sugar refinery: a long red-brick block of six storeys, banded and windowed in white
    over a ground floor rendered white with blue shutters and a long blue name board, its blue
    taking-in doors stacked under a hoist, rises between two cream char-house towers, and the
    square stack of its boilers rises behind it banded in white. Beside it stands the white
    sugarloaf store, racks of sugarloaves in blue paper showing through its arcade; a striped
    blue-and-white awning shades the dispatch door, where more loaves stand on a stepped stand,
    sacks wait and a dray is loaded, and before it a beet train stands at the siding by two long
    beet clamps."""
    m = METRE
    p = Painter(53, 58, (24, 42))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 17) * m, (v - 2) * m)

    p.polygon(
        [at(-7.5, 0), at(-0.5, -11.5), at(44.5, -11.5), at(44.5, 9.5), at(-7.5, 9.5)], SUGAR_GROUND
    )
    for cu, cv, lu, dv in (
        (14, 6, 28, 12),
        (-3.5, 3.5, 7, 7),
        (31.5, 3.5, 7, 7),
        (39.5, 4.5, 9, 9),
    ):
        _bed(p, _footing(*at(cu, cv), lu * m, dv * m))
    p.shadow(20, 7, 6)
    _sugar_stack(p, *at(9, 15.5), 56 * m)
    _sugar_char_house(p, *at(-3.5, 3.5), 7, 26)
    main = _sugar_block(p, *at(14, 6), 28, 12, 18, 8, (0.03, 0.66), False)
    _sugar_char_house(p, *at(31.5, 3.5), 7, 24)
    _sugar_store(p, *at(39.5, 4.5), 9, 9, 8)
    _sugar_beet(p, *at(5.4, -2.6), 10, 3.4, 2.2)
    _sugar_beet(p, *at(-2.6, -4.2), 6, 3.0, 1.8)
    t = 8.5 / 11
    _sugar_awning(p, main[0], main[1], (t - 0.055, t + 0.055), 14.2 * m, 1.7, 1.0)
    _sugar_display(p, *at(16.2, -1.6), 3)
    _sugar_bags(p, *at(23.8, -1.2), (2, 1))
    _sugar_bags(p, *at(25.0, -2.6), (3, 2, 1))

    def loaves(on_bed: Callable[[float, float], Point]) -> None:
        for w in (0.42, -0.3):
            for s in np.linspace(-1.2, 1.2, 5):
                _sugar_loaf(p, *on_bed(s + w * 0.3, w), 1.0 * m)

    _food_cart(p, *at(17.0, -4.8), 3.4, 0.3, SUGAR_GROUND, loaves)
    _sugar_siding(p, at(1, -9.2), at(44, -9.2))
    for u in (6.0, 12.6, 19.2, 25.8):
        _sugar_wagon(p, *at(u, -9.2))
    return p.sprite()


def mega_sugar_refinery() -> Sprite:
    """A great sugar refinery: a seven-storey block rendered white and banded in sugarloaf blue
    carries the company's grand blue name board on its roof, between a taller red-brick block
    banded in white and a cream char-house tower with a lower red-brick block beyond, all rising
    over white ground floors with blue shutters, and two square stacks banded in white rise
    behind them. A long siding brings trains of beet to a beet yard of long clamps, where a
    concrete flume carries the washed beet into the works; at the white block's dispatch door a
    striped awning shades a stand of sugarloaves in blue paper, sacks stamped in blue and a dray
    of loaves."""
    m = METRE
    p = Painter(59, 61, (27, 44))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 14) * m, (v - 2) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon(
        [at(-12, 3), at(-12, -4), at(-4, -15), at(40, -15), at(40, 13), at(-12, 16)],
        SUGAR_GROUND,
    )
    for cu, cv, lu, dv in ((-6, 10, 12, 12), (12, 6, 24, 12), (27.5, 3.5, 7, 7), (35.5, 8, 9, 12)):
        _bed(p, _footing(*at(cu, cv), lu * m, dv * m))
    p.shadow(22, 8, 6)
    _sugar_stack(p, *at(6, 17.5), 60 * m)
    _sugar_stack(p, *at(33, 16.5), 52 * m)
    _sugar_block(p, *at(-6, 10), 12, 12, 24, 2, None, False)
    main = _sugar_block(p, *at(12, 6), 24, 12, 22, 7, (0.03, 0.6), True)
    _sugar_sky_board(p, main, 12, 12 * 0.375 * m, (0.12, 0.88), 3.2)
    _sugar_char_house(p, *at(27.5, 3.5), 7, 28)
    _sugar_block(p, *at(35.5, 8), 9, 12, 16, None, None, False)
    _sugar_beet(p, *at(-6.5, -3.2), 6, 3.2, 2.0)
    _sugar_flume(p, up, (-1.5, -11.6), (-1.5, 4.0))
    _sugar_beet(p, *at(2.8, -8.6), 5, 3.0, 1.8)
    _sugar_beet(p, *at(3.4, -3.6), 6, 3.0, 2.0)
    t = 7.5 / 10
    _sugar_awning(p, main[0], main[1], (t - 0.065, t + 0.065), 17.4 * m, 1.7, 1.0)
    _sugar_display(p, *at(14.2, -1.6), 3)
    _sugar_bags(p, *at(21.6, -1.2), (2, 1))
    _sugar_bags(p, *at(22.4, -2.8), (3, 2, 1))

    def loaves(on_bed: Callable[[float, float], Point]) -> None:
        for w in (0.42, -0.3):
            for s in np.linspace(-1.2, 1.2, 5):
                _sugar_loaf(p, *on_bed(s + w * 0.3, w), 1.0 * m)

    _food_cart(p, *at(15.0, -5.0), 3.4, 0.3, SUGAR_GROUND, loaves)
    _sugar_siding(p, at(-2.5, -13.2), at(40, -13.2))
    for u in (1.6, 8.2, 14.8, 21.4):
        _sugar_wagon(p, *at(u, -13.2))
    return p.sprite()


MEAT_ROOF: RGB = (86, 80, 88)

MUD: RGB = (112, 94, 66)

HAM: RGB = (196, 76, 66)

BONE: RGB = (244, 238, 224)

COLD: RGB = (234, 236, 238)

COLD_DOOR: RGB = (150, 170, 186)

STEER: tuple[RGB, RGB] = ((138, 84, 50), (96, 58, 36))


MEAT_VAN: RGB = (238, 238, 232)

MEAT_STRIPE: RGB = (176, 40, 44)


def _meat_hall(
    p: Painter, x: float, y: float, size: tuple[float, float, float], doors: tuple[int, ...]
) -> Top:
    """A red-brick slaughter hall `size` (length, depth and wall height) standing on (x, y): two
    tidy rows of round-headed windows under cream arches between pilasters, cream bands at its
    eaves and between its floors, arched timber doors in the bays `doors` of its lit side, and a
    slate roof with a louvred vent along its ridge."""
    m = METRE
    length, depth, high = size
    top = p.hall(x, y, length, depth, high, BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, BRICK, 0.6 * m)
    _stains(p, left, front, high, (0.04, 0.37, 0.63, 0.97), shade(BRICK, 0.66), 1.2 * m)
    for a, b in ((left, front), (front, right)):
        for drop, wide in ((0, 0.45 * m), (high * 0.48, 0.25 * m)):
            _bar(
                p,
                _on_wall(a, b, 0, drop + wide / 2),
                _on_wall(a, b, 1, drop + wide / 2),
                wide,
                DRESSING,
            )
    bays = max(3, round(length / (3 * m)))
    for k in range(bays + 1):
        _bar(
            p,
            _on_wall(left, front, k / bays, 0.45 * m),
            _on_wall(left, front, k / bays, high - 0.6 * m),
            0.45 * m,
            shade(BRICK, 0.86),
        )
    upper, lower = (high * 0.14, high * 0.41), (high * 0.59, high * 0.86)
    for k in range(bays):
        t = (k + 0.5) / bays
        _mill_window(p, left, front, length, t, 1.2 * m, upper)
        if k in doors:
            p.polygon(_round_head(left, front, length, t, 2.6 * m, high * 0.55, high), DRESSING)
            p.polygon(_round_head(left, front, length, t, 2.2 * m, high * 0.58, high), DOORWAY)
            _patch(
                p,
                left,
                front,
                (t - 1.1 * m / length, t - 0.04 * m / length),
                (high * 0.68, high),
                DOOR,
            )
        else:
            _mill_window(p, left, front, length, t, 1.3 * m, lower)
    side = max(2, round(depth / (3.5 * m)))
    for k in range(side):
        t = (k + 0.5) / side
        _mill_window(p, front, right, depth, t, 1.2 * m, upper, shade(MILL_GLASS, 0.85))
        _mill_window(p, front, right, depth, t, 1.3 * m, lower, shade(MILL_GLASS, 0.85))
    p.gable(top, 3 * m, MILL_SLATE, BRICK)
    _roof_rows(p, top, 3 * m, MILL_SLATE)
    ridge_back, ridge_front = _ridge(top, 3 * m)
    vx, vy = _on_wall(ridge_back, ridge_front, 0.5, 0)
    vent = p.hall(vx, vy + 0.4 * m, length * 0.5, 1.4 * m, 1.3 * m, (178, 172, 164), None)
    for k in range(1, 4):
        drop = 1.3 * m * k / 4
        _bar(
            p,
            _on_wall(vent[0], vent[1], 0, drop),
            _on_wall(vent[0], vent[1], 1, drop),
            0.1 * m,
            (100, 96, 94),
        )
    p.gable(vent, 0.7 * m, MILL_SLATE, (178, 172, 164))
    return top


def _meat_cold_store(
    p: Painter, x: float, y: float, size: tuple[float, float, float], plants: int
) -> Top:
    """A white-tiled cold store `size` (length, depth and height) standing on (x, y): glazed
    tiles on a red-brick plinth under a red-brick cornice, a few small windows, insulated doors
    on both its faces, and on its flat roof `plants` refrigerating plants with their fans and a
    vent pipe."""
    m = METRE
    length, depth, high = size
    top = p.hall(x, y, length, depth, high, COLD_TILE, shade(COLD_TILE, 0.86))
    left, front, right, back = top
    for a, b, span, tone in ((left, front, length, 0.9), (front, right, depth, 0.72)):
        rows = int((high - 1.6 * m) / (0.45 * m))
        for k in range(rows):
            drop = 0.9 * m + k * 0.45 * m
            _bar(
                p,
                _on_wall(a, b, 0, drop),
                _on_wall(a, b, 1, drop),
                0.03 * m,
                shade(COLD_TILE, tone),
            )
        cols = max(4, round(span / m * 1.2))
        for k in range(1, cols):
            _bar(
                p,
                _on_wall(a, b, k / cols, 0.9 * m),
                _on_wall(a, b, k / cols, high - 0.7 * m),
                0.03 * m,
                shade(COLD_TILE, tone),
            )
        _patch(p, a, b, (0, 1), (0, 0.8 * m), shade(BRICK, 1.0 if tone > 0.8 else 0.8))
        _patch(p, a, b, (0, 1), (high - 0.7 * m, high), shade(BRICK, 0.9 if tone > 0.8 else 0.72))
        floors = max(1, round((high - 4 * m) / (3.6 * m)))
        for f in range(floors):
            drop = 1.6 * m + f * 3.6 * m
            for t in (0.2, 0.8) if span > 6 * m else (0.5,):
                _patch(
                    p,
                    a,
                    b,
                    (t - 0.5 * m / span, t + 0.5 * m / span),
                    (drop, drop + 1.1 * m),
                    shade(MILL_GLASS, tone * 1.1),
                )
    _cold_door(p, left, front, (0.42, 0.58), high - 3.3 * m, high - 1.1 * m)
    _cold_door(p, front, right, (0.32, 0.62), high - 3.3 * m, high)
    for k in range(plants):
        ux, uy = _on_wall(
            _on_wall(left, back, 0.5, 0), _on_wall(front, right, 0.5, 0), (k + 0.5) / plants, 0
        )
        plant = p.hall(ux, uy, 2.4 * m, 1.6 * m, 1.3 * m, METAL, shade(METAL, 1.1))
        fx, fy = _on_wall(plant[3], plant[1], 0.5, 0)
        p.ellipse(fx - 0.7 * m, fy - 0.35 * m, fx + 0.7 * m, fy + 0.35 * m, (60, 62, 70))
    vx, vy = _on_wall(back, right, 0.8, 0)
    _bar(p, (vx - 1 * m, vy), (vx - 1 * m, vy - 2.4 * m), 0.25 * m, METAL)
    return top


def _meat_lairs(p: Painter, x: float, y: float, pens: int, beasts: tuple[int, ...]) -> None:
    """Cattle lairs on (x, y): a row of `pens` fenced pens 6 m square along the lit walls,
    trampled to mud and strewn with straw, a water trough in each, with `beasts` brown steers
    standing in each pen in turn, under a long open lair roof on posts along their backs."""
    m = METRE

    def at(u: float, v: float) -> Point:
        return _iso(x, y, u * m, v * m)

    width = pens * 6
    p.polygon([at(0, 0), at(width, 0), at(width, 8), at(0, 8)], MUD)
    for k in range(pens * 3):
        u, v = 1 + (k * 2.3) % (width - 1.5), 1.2 + (k * 1.7) % 4.6
        sx, sy = at(u, v)
        p.blob(sx - 0.8 * m, sy - 0.4 * m, sx + 0.8 * m, sy + 0.4 * m, STRAW, 160)
    roof_lo, roof_hi = 2.6 * m, 3.4 * m
    for u in range(0, width + 1, 3):
        bx, by = at(u, 8)
        _bar(p, (bx, by), (bx, by - roof_hi), 0.18 * m, shade(TIMBER, 0.7))
    for k in range(pens + 1):
        _fence(p, at(k * 6, 6), at(k * 6, 0), 1.3 * METRE)
    _fence(p, at(0, 6), at(width, 6), 1.3 * METRE)
    for k in range(pens):
        tx, ty = at(k * 6 + 3, 5.2)
        trough = p.hall(tx, ty, 2.4 * m, 0.7 * m, 0.6 * m, shade(TIMBER, 0.85), PUDDLE)
        _bar(
            p,
            _on_wall(trough[3], trough[2], 0.2, 0),
            _on_wall(trough[3], trough[2], 0.6, 0),
            0.06 * m,
            shade(PUDDLE, 1.4),
        )
        for j in range(beasts[k % len(beasts)]):
            cx, cy = at(k * 6 + 1.8 + 2.4 * j, 2.4 + 1.4 * (j % 2))
            _cow(p, cx, cy, *STEER, BEAST, 1 if (j + k) % 2 else -1)
    for u in range(0, width + 1, 3):
        fx, fy = at(u, 6)
        _bar(p, (fx, fy), (fx, fy - roof_lo), 0.18 * m, shade(TIMBER, 0.85))
    roof = [at(0, 6), at(width, 6), at(width, 8), at(0, 8)]
    lo = [(px, py - roof_lo) for px, py in roof[:2]]
    hi = [(px, py - roof_hi) for px, py in roof[2:]]
    p.polygon([lo[0], lo[1], hi[0], hi[1]], MILL_SLATE, shade(MILL_SLATE, 0.6))
    for k in range(1, 3):
        f = k / 3
        _bar(
            p,
            (lo[0][0] + (hi[1][0] - lo[0][0]) * f, lo[0][1] + (hi[1][1] - lo[0][1]) * f),
            (lo[1][0] + (hi[0][0] - lo[1][0]) * f, lo[1][1] + (hi[0][1] - lo[1][1]) * f),
            0.06 * m,
            shade(MILL_SLATE, 0.8),
        )
    for k in range(pens):
        _fence(p, at(k * 6, 0), at(k * 6 + 6, 0), 1.3 * METRE)


def _meat_van(p: Painter, x: float, y: float) -> None:
    """A white refrigerator van 7.5 m long standing on a siding at (x, y) along the lit walls:
    insulated walls with a red band lettered in white, a door, ice hatches on its pale roof,
    spoked wheels and buffers."""
    m = METRE
    for t in (0.22, 0.78):
        wx, wy = _iso(x, y, (t - 0.5) * 7.5 * m, 0.7 * m)
        p.ellipse(wx - 0.5 * m, wy - 1 * m, wx + 0.5 * m, wy, (40, 38, 40))
    p.hall(x, y - 0.6 * m, 7.5 * m, 2.2 * m, 0.4 * m, (44, 42, 44), None)
    for t in (0.22, 0.78):
        wx, wy = _iso(x, y, (t - 0.5) * 7.5 * m, -0.7 * m)
        p.ellipse(wx - 0.5 * m, wy - 1 * m, wx + 0.5 * m, wy, (36, 34, 36))
        p.blob(wx - 0.15 * m, wy - 0.65 * m, wx + 0.15 * m, wy - 0.35 * m, (120, 118, 120))
    for u in (-3.95, 3.95):
        bx, by = _iso(x, y, u * m, 0)
        _bar(
            p,
            (bx, by - 1.1 * m),
            _iso(bx, by - 1.1 * m, 0.3 * m if u > 0 else -0.3 * m, 0),
            0.25 * m,
            (40, 38, 40),
        )
    body = p.hall(x, y - 1 * m, 7.5 * m, 2.6 * m, 2.4 * m, MEAT_VAN, None)
    left, front, _, _ = body
    for t in [k / 14 for k in range(1, 14)]:
        _bar(
            p,
            _on_wall(left, front, t, 0),
            _on_wall(left, front, t, 2.4 * m),
            0.03 * m,
            shade(MEAT_VAN, 0.86),
        )
    _patch(p, left, front, (0, 1), (0.7 * m, 1.2 * m), MEAT_STRIPE)
    for t0, t1 in ((0.08, 0.2), (0.24, 0.3), (0.62, 0.74), (0.78, 0.9)):
        _bar(
            p,
            _on_wall(left, front, t0, 0.95 * m),
            _on_wall(left, front, t1, 0.95 * m),
            0.18 * m,
            MEAT_VAN,
        )
    _patch(p, left, front, (0.38, 0.58), (0.3 * m, 2.3 * m), shade(MEAT_VAN, 0.9))
    _bar(
        p,
        _on_wall(left, front, 0.48, 0.3 * m),
        _on_wall(left, front, 0.48, 2.3 * m),
        0.05 * m,
        shade(MEAT_VAN, 0.6),
    )
    p.gable(body, 0.4 * m, (214, 212, 206), MEAT_VAN)
    for t in (0.15, 0.85):
        hx, hy = _on_wall(
            _on_wall(body[0], body[3], 0.5, -0.4 * m),
            _on_wall(body[1], body[2], 0.5, -0.4 * m),
            t,
            0,
        )
        p.hall(hx, hy + 0.15 * m, 0.7 * m, 0.7 * m, 0.25 * m, (150, 150, 146), (170, 170, 166))


def _meat_dock(p: Painter, x: float, y: float, length: float) -> Top:
    """A loading dock `length` metres long and 1.2 m high on (x, y) along the lit walls: a
    brick face under a deck of planks."""
    m = METRE
    deck = p.hall(x, y, length * m, 2.4 * m, 1.2 * m, shade(BRICK, 0.9), shade(TIMBER, 1.1))
    for k in range(1, round(length * 1.2)):
        t = k / round(length * 1.2)
        _bar(
            p,
            _on_wall(deck[0], deck[3], t, 0),
            _on_wall(deck[1], deck[2], t, 0),
            0.05 * m,
            shade(TIMBER, 0.8),
        )
    return deck


def _meat_gilt_ox(p: Painter, x: float, y: float, size: float) -> None:
    """A gilt ox head in relief seen from the front, its forehead at (x, y) and `size` units from
    its brow to its muzzle: long face, short ears and curving horns, all in gold lit on the
    left."""

    def at(points: list[tuple[float, float]]) -> list[Point]:
        return [(x + u * size, y + v * size) for u, v in points]

    for side in (-1, 1):
        p.polygon(
            at(
                [
                    (side * 0.28, -0.3),
                    (side * 0.6, -0.42),
                    (side * 0.84, -0.72),
                    (side * 0.8, -0.48),
                    (side * 0.58, -0.26),
                    (side * 0.3, -0.16),
                ]
            ),
            shade(GILT, 1.12 if side < 0 else 0.86),
        )
        ex, ey = x + side * 0.48 * size, y - 0.08 * size
        p.blob(
            ex - 0.16 * size, ey - 0.08 * size, ex + 0.16 * size, ey + 0.09 * size, shade(GILT, 0.8)
        )
    p.polygon(
        at(
            [
                (-0.36, -0.36),
                (0.36, -0.36),
                (0.38, -0.1),
                (0.26, 0.3),
                (0.2, 0.56),
                (-0.2, 0.56),
                (-0.26, 0.3),
                (-0.38, -0.1),
            ]
        ),
        GILT,
    )
    p.polygon(
        at([(0.06, -0.36), (0.36, -0.36), (0.38, -0.1), (0.26, 0.3), (0.2, 0.56), (0.06, 0.56)]),
        shade(GILT, 0.82),
    )
    for side in (-1, 1):
        ex = x + side * 0.17 * size
        p.blob(ex - 0.05 * size, y - 0.09 * size, ex + 0.05 * size, y, shade(GILT, 0.55))
    p.blob(x - 0.22 * size, y + 0.38 * size, x + 0.22 * size, y + 0.6 * size, shade(GILT, 0.92))
    _bar(
        p,
        (x - 0.2 * size, y - 0.25 * size),
        (x - 0.12 * size, y + 0.2 * size),
        0.05 * size,
        (255, 236, 160),
    )


def _meat_office(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """A meat company's office block three storeys high, `length` by `depth`, on (x, y): red
    brick on a stone plinth with cream quoins and string courses, rows of round-headed windows,
    a slate roof, and a central bay standing forward under a pediment bearing a gilt ox head,
    over an arched entrance up stone steps."""
    m = METRE
    storey, storeys = 3.8 * m, 3
    high = storey * storeys
    top = p.hall(x, y, length, depth, high, BRICK, None)
    left, front, right, _ = top
    for a, b, span, tone, skip in (
        (left, front, length, 1.0, (0.36, 0.64)),
        (front, right, depth, 0.85, (2, 2)),
    ):
        _patch(p, a, b, (0, 1), (high - 1 * m, high), shade(STONE, tone * 1.1))
        _patch(p, a, b, (0, 1), (0, 0.6 * m), shade(DRESSING, tone))
        for k in range(1, storeys):
            _bar(
                p,
                _on_wall(a, b, 0, k * storey),
                _on_wall(a, b, 1, k * storey),
                0.25 * m,
                shade(DRESSING, tone),
            )
        for t in (0.015, 0.985):
            _bar(
                p,
                _on_wall(a, b, t, 0.6 * m),
                _on_wall(a, b, t, high - 1 * m),
                0.55 * m,
                shade(DRESSING, tone),
            )
        bays = max(2, round(span / (2.8 * m)))
        for k in range(bays):
            t = (k + 0.5) / bays
            if skip[0] < t < skip[1]:
                continue
            for s in range(storeys):
                head = s * storey + 0.9 * m
                _mill_window(
                    p, a, b, span, t, 1.1 * m, (head, head + 2.2 * m), shade(MILL_GLASS, tone)
                )
    p.gable(top, 3 * m, MILL_SLATE, BRICK)
    _roof_rows(p, top, 3 * m, MILL_SLATE)
    cx, cy = _iso(x, y, 0, -(depth / 2 + 0.5 * m))
    span = length * 0.3
    bay = p.hall(cx, cy, span, 1.6 * m, high + 0.6 * m, shade(BRICK, 1.04), None)
    b_left, b_front, b_right, _ = bay
    _patch(p, b_left, b_front, (0, 1), (high - 0.4 * m, high + 0.6 * m), STONE)
    _patch(p, b_front, b_right, (0, 1), (high - 0.4 * m, high + 0.6 * m), shade(STONE, 0.8))
    for k in range(1, storeys):
        _bar(
            p,
            _on_wall(b_left, b_front, 0, k * storey + 0.6 * m),
            _on_wall(b_left, b_front, 1, k * storey + 0.6 * m),
            0.25 * m,
            DRESSING,
        )
    for t in (0.05, 0.95):
        _bar(
            p,
            _on_wall(b_left, b_front, t, 0),
            _on_wall(b_left, b_front, t, high),
            0.55 * m,
            DRESSING,
        )
    for s in range(storeys - 1):
        head = s * storey + 1.5 * m
        _mill_window(p, b_left, b_front, span, 0.5, 1.2 * m, (head, head + 2.2 * m))
    p.polygon(
        _round_head(b_left, b_front, span, 0.5, 2.8 * m, high - 3.6 * m, high + 0.6 * m), DRESSING
    )
    p.polygon(
        _round_head(b_left, b_front, span, 0.5, 2.2 * m, high - 3.3 * m, high + 0.6 * m), DOOR
    )
    for k in range(3):
        sx, sy = _iso(cx, cy, 0, -(1.2 + 0.5 * k) * m)
        p.hall(sx, sy, 4 * m + k * 0.8 * m, 0.5 * m, (0.6 - 0.2 * k) * m, STONE, shade(STONE, 1.1))
    _bands(p, b_left, b_front, b_right, 0.3 * m, DRESSING, shade(DRESSING, 0.78))
    _gable_across(p, bay, 2.8 * m, MILL_SLATE, shade(BRICK, 1.04))
    ox, oy = _on_wall(b_left, b_front, 0.5, -1.3 * m)
    _meat_gilt_ox(p, ox, oy, 1.9 * m)


def big_meat_factory() -> Sprite:
    """A big meat works round a yard: two red-brick slaughter halls with tidy rows of
    round-headed windows under cream arches and slate roofs with louvred vents, the butcher's
    sign of a white ox head on an oxblood board hung from a bracket, cattle lairs of four pens of
    brown steers under a long lair roof, a larger white-tiled cold store with insulated doors and
    two refrigerating plants on its roof, a loading dock with a meat wagon at it, and a siding
    with white refrigerator vans."""
    m = METRE
    p = Painter(60, 50, (31, 34))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 18) * m, (v - 4) * m)

    p.polygon([at(18, -9), at(44, -9), at(44, 0), at(18, 0)], (150, 140, 126))
    for (u, v), length, depth in (((9, 4.5), 20, 9), ((26, 11), 8, 18), ((37, 6), 10, 9)):
        _bed(p, _footing(*at(u, v), length * m, depth * m), margin=0.6)
    p.shadow(22, 5.5, 2)
    _meat_hall(p, *at(26, 11), (8 * m, 18 * m, 9 * m), (1,))
    hall = _meat_hall(p, *at(9, 4.5), (20 * m, 9 * m, 9 * m), (1, 4))
    _butcher_sign(p, *_on_wall(hall[0], hall[1], 0.02, 1.6 * m))
    _meat_cold_store(p, *at(37, 6), (10 * m, 9 * m, 9 * m), 2)
    _meat_lairs(p, *at(-3, -12), 4, (2, 1))
    _meat_dock(p, *at(31.5, -1.2), 21)
    _meat_wagon(p, *at(24, -4))
    _works_siding(p, at(20, -7.5), at(44, -7.5))
    for u in (30, 38.5):
        _meat_van(p, *at(u, -7.5))
    return p.sprite()


def mega_meat_factory() -> Sprite:
    """A meat-packing quarter: a three-storey red-brick office block whose pediment bears a gilt
    ox head, long cattle lairs of pens full of brown steers under a lair roof before it, a long
    red-brick slaughter hall with tidy rows of round-headed windows and a louvred ridge vent, a
    tall white-tiled cold-store block of five floors with refrigerating plants on its roof beside
    a lower one, a loading dock along them, and a rail dock where refrigerator vans wait on the
    siding."""
    m = METRE
    p = Painter(64, 62, (32, 42))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 22) * m, (v - 4) * m)

    p.polygon([at(17, -10), at(54, -10), at(54, 0), at(17, 0)], (150, 140, 126))
    for (u, v), length, depth in (
        ((7, 5), 14, 10),
        ((25, 5), 18, 10),
        ((41, 6), 10, 12),
        ((49.5, 5), 5, 10),
    ):
        _bed(p, _footing(*at(u, v), length * m, depth * m), margin=0.6)
    p.shadow(25, 6, 2)
    _meat_cold_store(p, *at(49.5, 5), (5 * m, 10 * m, 7 * m), 1)
    _meat_cold_store(p, *at(41, 6), (10 * m, 12 * m, 18 * m), 2)
    _meat_hall(p, *at(25, 5), (18 * m, 10 * m, 10 * m), (1, 4))
    _meat_office(p, *at(7, 5), 14 * m, 10 * m)
    _meat_lairs(p, *at(-1.5, -13), 3, (2, 3, 2))
    _meat_dock(p, *at(34.5, -1.2), 35)
    _works_siding(p, at(18, -7.5), at(54, -7.5))
    for u in (22, 30.5, 39, 47.5):
        _meat_van(p, *at(u, -7.5))
    return p.sprite()


def big_sausage_maker() -> Sprite:
    """A smoked-meats works: on the street a pork butcher's house of two half-timbered storeys
    over its shop, the shop front in white tiles with windows of sausages and hams under red and
    pink awnings; behind it a curing hall of smoke-browned brick; two smokehouses with wood smoke
    seeping from their louvred lanterns and beech logs stacked before them; and a delivery yard
    behind a brick wall, entered through a gateway with the gilt pig standing on its iron arch,
    where a white meat wagon waits to be loaded and pink pigs root in a pen by their sty."""
    m = METRE
    p = Painter(56, 46, (28, 30))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 20) * m, (v - 12) * m)

    p.polygon([at(-1, -2), at(41, -2), at(41, 26), at(-1, 26)], COBBLES, shade(COBBLES, 0.85))
    p.polygon([at(15, 0), at(41, 0), at(41, 15), at(15, 15)], CRAFT_YARD)
    _craft_curing_hall(p, *at(16, 20.5), 22 * m, 9 * m, 7 * m)
    _craft_smokehouse(p, *at(31.5, 22), 7)
    _craft_smokehouse(p, *at(37, 22), 7)
    for u in (31.5, 37):
        _craft_log_pile(p, *at(u, 18.8), 3.2 * m, 1.5 * m)
    _meat_wagon(p, *at(23, 9))
    pen = (at(30, 2), at(40, 2), at(40, 11), at(30, 11))
    near_left, near_right, far_right, far_left = pen
    p.polygon(list(pen), MUD, shade(MUD, 0.8))
    _fence(p, near_left, far_left, METRE)
    _fence(p, far_left, far_right, METRE)
    _craft_sty(p, *at(38, 9.5))
    _pig(p, *at(35, 6.5), BEAST, 1)
    _pig(p, *at(32.5, 4.5), BEAST, -1)
    _pig(p, *at(36.5, 3.5), BEAST, -1)
    _fence(p, near_left, near_right, METRE)
    _fence(p, near_right, far_right, METRE)
    _craft_butcher_house(p, *at(7, 5.5), 13 * m, 8 * m, 2)
    _craft_street_wall(p, at(15, -0.5), at(19, -0.5))
    _craft_street_wall(p, at(24, -0.5), at(41, -0.5))
    _craft_pig_gate(p, *at(21.5, -0.5), 5, 3.0)
    return p.sprite()


def mega_sausage_maker() -> Sprite:
    """A great smoked-meats works: on the street a grand pork butcher's house of three
    half-timbered storeys over its shop, its windows of sausages and hams under red and pink
    awnings, and a great gilt pig standing on its roof; behind it a long curing hall of
    smoke-browned brick fronted by a row of four tall smokehouses, wood smoke rising from all
    their lanterns, with beech logs stacked by them; and a delivery yard behind a brick wall,
    entered through a gateway with the gilt pig on its iron arch, where two white meat wagons are
    loaded and pink pigs root in a pen by their sty."""
    m = METRE
    p = Painter(66, 56, (33, 38))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 23.5) * m, (v - 14.5) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon([at(0, -2), at(47.5, -2), at(47.5, 30.5), at(0, 30.5)], COBBLES, shade(COBBLES, 0.85))
    p.polygon([at(18, 0), at(47.5, 0), at(47.5, 18), at(18, 18)], CRAFT_YARD)
    _craft_curing_hall(p, *at(22, 25.5), 34 * m, 9 * m, 8 * m)
    for u in (11, 18, 25, 32):
        _craft_smokehouse(p, *at(u, 18.5), 10)
        _craft_log_pile(p, *at(u + 3.4, 15.5), 2.6 * m, 1.4 * m)
    _craft_smokehouse(p, *at(43, 25), 8)
    _meat_wagon(p, *at(26, 10))
    _meat_wagon(p, *at(30.5, 7.5))
    pen = (at(37, 2), at(47, 2), at(47, 12), at(37, 12))
    near_left, near_right, far_right, far_left = pen
    p.polygon(list(pen), MUD, shade(MUD, 0.8))
    _fence(p, near_left, far_left, METRE)
    _fence(p, far_left, far_right, METRE)
    _craft_sty(p, *at(45, 10.5))
    for u, v, facing in ((42, 7.5, 1), (39.5, 5, -1), (43.5, 4, -1), (40.5, 9, 1)):
        _pig(p, *at(u, v), BEAST, facing)
    _fence(p, near_left, near_right, METRE)
    _fence(p, near_right, far_right, METRE)
    house = _craft_butcher_house(p, *at(8.5, 6), 16 * m, 9 * m, 3)
    px, py = _on_wall(*_ridge(house, 9 * m * 0.6), 0.5, 0)
    p.hall(px, py + 0.3 * m, 1.8 * m, 1.0 * m, 0.7 * m, CRAFT_GILT, shade(CRAFT_GILT, 1.15))
    _craft_gilt_pig(p, px, py - 0.4 * m - 2.5 * m, 5 * m)
    _craft_street_wall(p, at(18, -0.5), at(22, -0.5))
    _craft_street_wall(p, at(27, -0.5), at(47.5, -0.5))
    _craft_pig_gate(p, *at(24.5, -0.5), 5, 3.0)
    return p.sprite()


def big_barrel_maker() -> Sprite:
    """A cooperage grown into a works: a long weatherboarded oak cooper's shop under an iron-grey
    roof with its doors open on a cask inside and hoops leant against its wall, an iron-clad steam
    shop breathing steam with an upright boiler beside it, a stave yard of seasoning cribs, casks
    being raised and fired in the yard, and long pyramids of finished oak casks."""
    m = METRE
    p = Painter(60, 42, (27, 26))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 8) * m, (v - 3) * m)

    p.polygon([at(-16, -6), at(31, -6), at(31, 4), at(-6, 4), at(-6, 10), at(-16, 10)], YARD)
    _cooper_shavings(
        p, [at(u, v) for u, v in ((6, 2), (8.4, 0.4), (19, -1), (21.5, -3.4), (3, -0.6), (12, 2.6))]
    )
    p.shadow(18, 5, 5)
    for row in range(3):
        for k in range(5):
            _cask_staves(p, *at(-14 + 1.8 * k, 8.6 - 2.0 * row), 16 - 5 * ((k + row) % 2))
    shop = _drink_cooper_shed(p, *at(9, 8), 18, 8, 5)
    _cask_hoops(p, *_on_wall(shop[0], shop[1], 0.06, 5 * m), (0.5, 0.45, 0.42, 0.5, 0.45))
    _drink_steam_shop(p, *at(23, 8.5), 8, 6)
    _drink_boiler(p, *at(28.6, 6))
    for u, v in ((17.6, 1.8), (19.6, 0.4)):
        _cask_raised(p, *at(u, v))
    _cask_raised(p, *at(23, -0.6), fire=True)
    _cask_raised(p, *at(25.4, -2.6), fire=True)
    _cask_stack(p, *at(1, 1.2), (12, 11, 10))
    _cask_stack(p, *at(2.5, -3.2), (10, 9))
    for u, v in ((-6, -2.4), (-5.2, -3.4), (-6.8, -4.2)):
        _cask(p, *at(u, v), "up")
    return p.sprite()


def mega_barrel_maker() -> Sprite:
    """A great cooperage works of oak and iron: a long cooper's hall under an iron-grey roof with
    a great oak cask hooped in gilt standing on its ridge as its sign, an iron-clad steam shop
    breathing steam beside two upright boilers, a stave yard of seasoning cribs beside a stack of
    oak logs, casks raised and fired in the yard, long pyramids of finished casks, and a siding
    of wagons loaded with casks along its front."""
    m = METRE
    p = Painter(70, 56, (32, 34))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 9) * m, (v - 2) * m)

    p.polygon([at(-18, -7), at(36, -7), at(36, 4), at(-6, 4), at(-6, 11), at(-18, 11)], YARD)
    _cooper_shavings(
        p,
        [
            at(u, v)
            for u, v in ((8, 2), (10.4, 0.4), (22, -1), (24.5, -3.4), (5, -0.6), (14, 2.6), (28, 1))
        ],
    )
    p.shadow(22, 6, 6)
    _drink_logs(p, *at(-18.5, 13), (5, 4, 3))
    for row in range(4):
        for k in range(5):
            _cask_staves(p, *at(-16 + 1.8 * k, 9.4 - 2.0 * row), 18 - 5 * ((k + row) % 2))
    hall = _drink_cooper_shed(p, *at(9, 9), 26, 9, 7)
    _cask_hoops(p, *_on_wall(hall[0], hall[1], 0.05, 7 * m), (0.5, 0.45, 0.42, 0.5, 0.45, 0.5))
    rise = 0.38 * 9 * m
    sx, sy = _on_roof(hall, rise, 0.5, 1)
    _drink_sign_cask(p, sx, sy + 0.3 * m)
    _drink_steam_shop(p, *at(28, 9.5), 9, 7)
    for u, v in ((34, 7), (34.4, 11)):
        _drink_boiler(p, *at(u, v))
    for u, v in ((20, 1.8), (22, 0.4)):
        _cask_raised(p, *at(u, v))
    for u, v in ((26, -0.6), (28.4, -2.6), (30.4, -0.8)):
        _cask_raised(p, *at(u, v), fire=True)
    _cask_stack(p, *at(-1, 1.4), (14, 13, 12))
    _cask_stack(p, *at(0.5, -3.0), (13, 12, 11))
    _drink_siding(p, *at(-16, -9.5), 52)
    for u in (-12, -4.8, 2.4, 9.6, 22):
        _drink_cask_wagon(p, *at(u, -9.5), shade(COOPER_WALL, 0.8))
    return p.sprite()


COWL_WHITE: RGB = (240, 236, 226)

SIDING: RGB = (150, 146, 138)

SLEEPERS: RGB = (92, 70, 52)

STEEL: RGB = (176, 178, 184)

WAGON_GREEN: RGB = (62, 98, 60)


def _drink_up(x: float, y: float, z: float) -> Point:
    """The point `z` metres above (x, y)."""
    return x, y - z * METRE


def _drink_pyramid(p: Painter, top: Top, rise: float, roof: RGB) -> Point:
    """A pyramid roof `rise` high on a square block's top, coursed across its two sides in
    sight; returns its apex."""
    left, front, right, _ = top
    apex = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2 - rise)
    p.polygon([left, front, apex], roof, shade(roof, 0.6))
    p.polygon([front, right, apex], shade(roof, 0.72), shade(roof, 0.5))
    for f in (0.3, 0.6):
        for a, b, tone in ((left, front, 0.82), (front, right, 0.62)):
            _bar(
                p, _on_wall(a, apex, f, 0), _on_wall(b, apex, f, 0), 0.08 * METRE, shade(roof, tone)
            )
    return apex


def _drink_steam(p: Painter, x: float, y: float, puffs: int = 3) -> None:
    """A soft white wisp of steam rising and drifting right from (x, y)."""
    m = METRE
    p.soft(
        [
            (
                x - 0.6 * m + k * 1.0 * m,
                y - (1.2 + 1.2 * k) * m,
                x + (0.8 + k * 1.1) * m,
                y - 1.2 * k * m,
            )
            for k in range(puffs)
        ],
        (248, 246, 240),
        0.72,
        0.3,
    )


def _drink_cowl(p: Painter, x: float, y: float) -> None:
    """The white weatherboarded cowl turning on top of a malt kiln at (x, y): a round drum with a
    hood leaning over to the right on its dark mouth, and a vane fin behind keeping it to the
    wind."""
    m = METRE
    _turned(p, x, y, [(0, 0.8 * m), (1.4 * m, 0.75 * m)], COWL_WHITE)
    for drop in (0.45, 0.9):
        _bar(
            p,
            (x - 0.78 * m, y - drop * m),
            (x + 0.78 * m, y - drop * m),
            0.04 * m,
            shade(COWL_WHITE, 0.8),
        )
    hood = [
        (x - 0.75 * m, y - 1.4 * m),
        (x - 0.6 * m, y - 2.3 * m),
        (x - 0.1 * m, y - 2.7 * m),
        (x + 0.6 * m, y - 2.55 * m),
        (x + 1.15 * m, y - 1.95 * m),
        (x + 0.75 * m, y - 1.4 * m),
    ]
    p.polygon(hood, COWL_WHITE, shade(COWL_WHITE, 0.6))
    p.polygon(
        [(x + 0.45 * m, y - 1.45 * m), (x + 1.1 * m, y - 1.98 * m), (x + 0.75 * m, y - 1.4 * m)],
        (52, 46, 42),
    )
    p.polygon(
        [
            (x - 0.35 * m, y - 2.55 * m),
            (x - 1.7 * m, y - 3.0 * m),
            (x - 1.7 * m, y - 2.35 * m),
            (x - 0.5 * m, y - 2.1 * m),
        ],
        shade(COWL_WHITE, 0.88),
        shade(COWL_WHITE, 0.6),
    )


def _drink_kiln(
    p: Painter,
    x: float,
    y: float,
    side: float,
    high: float,
    wall: RGB,
    roof: RGB,
    door: RGB,
    pagoda: bool,
) -> None:
    """A square malt kiln `side` metres a side and `high` to its eaves on (x, y), dressed in
    cream when of brick and tarred at its foot when whitewashed, with a door painted `door` and a
    window, under a steep pyramid roof crowned by a pagoda ventilator or a white cowl."""
    m = METRE
    top = p.hall(x, y, side * m, side * m, high * m, wall, None)
    left, front, right, _ = top
    if wall == WHITEWASH:
        for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
            _patch(p, a, b, (0, 1), (high * m - 0.8 * m, high * m), shade(TAR, tone))
    else:
        _plinth(p, top, high * m, wall, 0.6 * m)
        for a, b in ((left, front), (front, right)):
            for t in (0.03, 0.97):
                _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, high * m - 0.6 * m), 0.35 * m, FOAM)
            _bar(p, _on_wall(a, b, 0, 0.2 * m), _on_wall(a, b, 1, 0.2 * m), 0.3 * m, FOAM)
    _patch(p, left, front, (0.32, 0.68), (high * m - 3 * m, high * m), door)
    _bar(
        p,
        _on_wall(left, front, 0.5, high * m - 3 * m),
        _on_wall(left, front, 0.5, high * m),
        0.06 * m,
        shade(door, 0.6),
    )
    _still_window(p, left, front, side * m, 0.5, 1 * m, (1.8 * m, 3.0 * m))
    _still_window(p, front, right, side * m, 0.5, 1 * m, (1.8 * m, 3.0 * m))
    apex = _drink_pyramid(p, top, side * 0.85 * m, roof)
    if pagoda:
        _still_pagoda(p, apex[0], apex[1] + 1.4 * m)
    else:
        _drink_cowl(p, apex[0], apex[1] + 0.8 * m)


def _drink_brewhouse(
    p: Painter, x: float, y: float, length: float, depth: float, high: float, kettles: int
) -> Top:
    """A brewhouse of honey brick `length` by `depth` metres and `high` to its eaves on (x, y),
    under a cream cornice and a malt-brown tiled roof: a hop-green arched door, `kettles` tall
    arched windows with copper kettles gleaming through them between pilasters, two storeys of
    windows at its end, and a copper vapour stack trailing steam on its ridge."""
    m = METRE
    length, depth, high = length * m, depth * m, high * m
    top = p.hall(x, y, length, depth, high, BREW_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, BREW_BRICK, 0.7 * m)
    for a, b in ((left, front), (front, right)):
        _bar(p, _on_wall(a, b, 0, 0.22 * m), _on_wall(a, b, 1, 0.22 * m), 0.45 * m, FOAM)
        _bar(p, _on_wall(a, b, 0, 1.0 * m), _on_wall(a, b, 1, 1.0 * m), 0.15 * m, shade(FOAM, 0.9))
    bays = kettles + 2
    for k in range(bays + 1):
        _bar(
            p,
            _on_wall(left, front, k / bays, 0.45 * m),
            _on_wall(left, front, k / bays, high - 0.7 * m),
            0.5 * m,
            shade(BREW_BRICK, 0.88),
        )
    for k in range(1, kettles + 1):
        _brew_window(p, left, front, length, (k + 0.5) / bays, 2.3 * m, (1.6 * m, high - 1.2 * m))
    t = 0.5 / bays
    p.polygon(_round_head(left, front, length, t, 2.8 * m, high - 4.7 * m, high), FOAM)
    p.polygon(_round_head(left, front, length, t, 2.3 * m, high - 4.4 * m, high), HOP)
    _bar(
        p,
        _on_wall(left, front, t, high - 4.4 * m),
        _on_wall(left, front, t, high),
        0.08 * m,
        shade(HOP, 0.6),
    )
    ox, oy = _on_wall(left, front, t, 2.6 * m)
    p.ellipse(ox - 0.9 * m, oy - 0.9 * m, ox + 0.9 * m, oy + 0.9 * m, FOAM)
    p.ellipse(ox - 0.6 * m, oy - 0.6 * m, ox + 0.6 * m, oy + 0.6 * m, MILL_GLASS)
    t = (bays - 0.5) / bays
    for drop in ((1.6 * m, high / 2 - 1.2 * m), (high / 2 + 0.6 * m, high - 1.4 * m)):
        _mill_window(p, left, front, length, t, 1.4 * m, drop)
    for t in (0.3, 0.7):
        _mill_window(
            p,
            front,
            right,
            depth,
            t,
            1.3 * m,
            (1.6 * m, high / 2 - 1.2 * m),
            shade(MILL_GLASS, 0.85),
        )
        _mill_window(
            p,
            front,
            right,
            depth,
            t,
            1.3 * m,
            (high / 2 + 0.6 * m, high - 1.4 * m),
            shade(MILL_GLASS, 0.85),
        )
    rise = 0.4 * depth
    p.gable(top, rise, MALT_TILE, BREW_BRICK)
    _roof_rows(p, top, rise, MALT_TILE, (0.25, 0.5, 0.75))
    vx, vy = _on_roof(top, rise, 0.55, 1)
    _turned(p, vx, vy + 0.5 * m, [(0, 0.45 * m), (3.2 * m, 0.4 * m)], KETTLE)
    _turned(p, vx, vy - 3 * m, [(0, 0.95 * m), (0.7 * m, 0)], shade(KETTLE, 0.9))
    _drink_steam(p, vx, vy - 3.4 * m)
    return top


def _drink_tower(
    p: Painter, x: float, y: float, wide: float, deep: float, tall: float, sheaf: float
) -> Point:
    """A brew tower of honey brick `wide` by `deep` metres and `tall` to its eaves on (x, y):
    cream quoins and string courses, a window on every floor, a louvred cooling floor at its top,
    a hop-green cask door at its foot, and a pyramid of malt-brown tiles crowned by a gilt ball,
    with a gilt sheaf of barley `sheaf` metres tall on it if `sheaf`. Returns the apex."""
    m = METRE
    wide_u, deep_u, tall_u = wide * m, deep * m, tall * m
    top = p.hall(x, y, wide_u, deep_u, tall_u, BREW_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, tall_u, BREW_BRICK, 0.7 * m)
    floors = [3.6 + 4 * k for k in range(int((tall - 4.6) // 4))]
    for a, b in ((left, front), (front, right)):
        for t in (0.03, 0.97):
            _bar(p, _on_wall(a, b, t, 0), _on_wall(a, b, t, tall_u - 0.7 * m), 0.4 * m, FOAM)
        for drop in (0.22, *floors):
            _bar(p, _on_wall(a, b, 0, drop * m), _on_wall(a, b, 1, drop * m), 0.3 * m, FOAM)
        _brew_louvres(p, a, b, (0.24, 0.76), (1 * m, 3 * m))
    for drop in floors[:-1]:
        head = (drop + 0.9) * m
        _mill_window(p, left, front, wide_u, 0.5, 1.3 * m, (head, head + 2.3 * m))
        if drop + 4 < tall - 5:
            _mill_window(
                p,
                front,
                right,
                deep_u,
                0.5,
                1.2 * m,
                (head, head + 2.3 * m),
                shade(MILL_GLASS, 0.85),
            )
    p.polygon(_round_head(front, right, deep_u, 0.5, 3.2 * m, tall_u - 4.4 * m, tall_u), FOAM)
    p.polygon(
        _round_head(front, right, deep_u, 0.5, 2.7 * m, tall_u - 4.1 * m, tall_u), shade(HOP, 0.8)
    )
    _bar(
        p,
        _on_wall(front, right, 0.5, tall_u - 4.1 * m),
        _on_wall(front, right, 0.5, tall_u),
        0.08 * m,
        shade(HOP, 0.5),
    )
    apex = _drink_pyramid(p, top, max(3.5, wide * 0.6) * m, MALT_TILE)
    p.ellipse(apex[0] - 0.35 * m, apex[1] - 0.6 * m, apex[0] + 0.35 * m, apex[1] + 0.1 * m, GILT)
    if sheaf:
        _brew_sheaf(p, apex[0], apex[1] - 0.4 * m, sheaf * m)
    return apex


def _drink_copper_dome(p: Painter, x: float, y: float, r: float) -> None:
    """The gleaming copper hood of a brew kettle `r` metres round standing up through a roof at
    (x, y), with a dark seam, a light on its shoulder and a vent pipe on its crown letting off
    steam."""
    m = METRE
    profile = [
        (0, r),
        (0.3 * r, 0.97 * r),
        (0.6 * r, 0.82 * r),
        (0.85 * r, 0.52 * r),
        (r, 0.12 * r),
    ]
    _turned(p, x, y, [(h * m, w * m) for h, w in profile], KETTLE)
    arc = np.linspace(0.1, math.pi - 0.1, 9)
    for a0, a1 in itertools.pairwise(arc):
        _bar(
            p,
            (x + r * m * math.cos(a0), y - 0.15 * r * m + r * m / 2 * math.sin(a0)),
            (x + r * m * math.cos(a1), y - 0.15 * r * m + r * m / 2 * math.sin(a1)),
            0.12 * m,
            shade(KETTLE, 0.62),
        )
    p.blob(
        x - 0.62 * r * m, y - 0.82 * r * m, x - 0.2 * r * m, y - 0.5 * r * m, shade(KETTLE, 1.35)
    )
    _turned(p, x, y - r * m, [(0, 0.22 * m), (1.6 * m, 0.2 * m)], shade(KETTLE, 0.95))
    _turned(p, x, y - r * m - 1.8 * m, [(0, 0.5 * m), (0.4 * m, 0)], shade(KETTLE, 0.85))
    _drink_steam(p, x, y - r * m - 2.2 * m, 2)


def _drink_cupola(p: Painter, x: float, y: float, sheaf: float) -> None:
    """A cream cupola on a tower's apex at (x, y): a round drum of arched openings on a moulded
    base under a copper dome, carrying a great gilt sheaf of barley `sheaf` metres tall on a gilt
    ball."""
    m = METRE
    _turned(p, x, y + 0.6 * m, [(0, 2.2 * m), (0.7 * m, 2.2 * m)], shade(FOAM, 0.9), FOAM)
    _turned(p, x, y, [(0, 1.8 * m), (3 * m, 1.8 * m)], FOAM)
    for a in np.linspace(0.4, math.pi - 0.4, 5):
        ox, oy = x + 1.8 * m * math.cos(a), y + 0.9 * m * math.sin(a)
        _bar(p, (ox, oy - 0.6 * m), (ox, oy - 2.4 * m), 0.45 * m, (60, 46, 36))
    _turned(p, x, y - 3 * m, [(0, 2.1 * m), (0.35 * m, 2.05 * m)], shade(FOAM, 0.92), FOAM)
    _turned(
        p,
        x,
        y - 3.35 * m,
        [(0, 1.85 * m), (0.8 * m, 1.65 * m), (1.5 * m, 1.15 * m), (2 * m, 0.45 * m), (2.2 * m, 0)],
        KETTLE,
    )
    p.blob(x - 1.2 * m, y - 4.8 * m, x - 0.5 * m, y - 4.1 * m, shade(KETTLE, 1.35))
    p.ellipse(x - 0.45 * m, y - 6.1 * m, x + 0.45 * m, y - 5.2 * m, GILT)
    _brew_sheaf(p, x, y - 5.8 * m, sheaf * m)


def _drink_clock(p: Painter, a: Point, b: Point, t: float, drop: float, r: float) -> None:
    """A clock `r` metres round on the wall under the edge from `a` to `b`, `t` along it and
    `drop` down it to its middle: a cream dial in a gilt ring with dark hands."""
    m = METRE
    cx, cy = _on_wall(a, b, t, drop)
    p.ellipse(cx - r * m, cy - r * m, cx + r * m, cy + r * m, GILT)
    p.ellipse(cx - 0.8 * r * m, cy - 0.8 * r * m, cx + 0.8 * r * m, cy + 0.8 * r * m, FOAM)
    _bar(p, (cx, cy), (cx, cy - 0.65 * r * m), 0.12 * m, (40, 34, 30))
    _bar(p, (cx, cy), (cx + 0.45 * r * m, cy + 0.15 * r * m), 0.12 * m, (40, 34, 30))


def _drink_siding(p: Painter, x: float, y: float, length: float) -> None:
    """A railway siding `length` metres long running along the lit walls from (x, y): grey
    ballast, sleepers every 0.7 m and two steel rails at standard gauge."""
    m = METRE
    p.polygon(
        [
            _cask_at(x, y, 0, 1.5),
            _cask_at(x, y, length, 1.5),
            _cask_at(x, y, length, -1.5),
            _cask_at(x, y, 0, -1.5),
        ],
        SIDING,
    )
    for u in np.arange(0.35, length, 0.7):
        _bar(p, _cask_at(x, y, u, 1.15), _cask_at(x, y, u, -1.15), 0.22 * m, SLEEPERS)
    for v in (0.72, -0.72):
        _bar(p, _cask_at(x, y, 0, v, 0.08), _cask_at(x, y, length, v, 0.08), 0.12 * m, STEEL)


def _drink_cask_wagon(p: Painter, x: float, y: float, body: RGB = WAGON_GREEN) -> None:
    """A low-sided wagon 6.6 m long standing on a siding at (x, y), its sides painted `body` and
    lined in gold, on two axles of iron wheels with buffers at its ends, loaded with two tiers of
    oak casks."""
    m = METRE
    for u in (1.6, 5.0):
        _brew_wheel(p, x, y, u, 0.8, 0.46, (64, 60, 62))
    p.hall(*_cask_at(x, y, 3.3, 0, 0.55), 6.6 * m, 2.2 * m, 0.3 * m, (46, 44, 46), None)
    deck = p.hall(
        *_cask_at(x, y, 3.3, 0, 0.85), 6.6 * m, 2.5 * m, 0.5 * m, body, shade(TIMBER, 1.05)
    )
    left, front, _, _ = deck
    _band(p, left, front, 0.25 * m, GILT)
    for t in (0.25, 0.5, 0.75):
        _bar(
            p,
            _on_wall(left, front, t, 0),
            _on_wall(left, front, t, 0.5 * m),
            0.05 * m,
            shade(body, 0.7),
        )
    _cask_stack(p, *_cask_at(x, y, 0.65, 0, 1.35), (6, 5), shadow=False)
    for u in (1.6, 5.0):
        _brew_wheel(p, x, y, u, -0.8, 0.46, (64, 60, 62))
    for u in (-0.15, 6.75):
        for v in (0.6, -0.6):
            bx, by = _cask_at(x, y, u, v, 0.75)
            p.blob(bx - 0.18 * m, by - 0.18 * m, bx + 0.18 * m, by + 0.18 * m, (40, 38, 40))


def _drink_cooper_shed(
    p: Painter, x: float, y: float, length: float, depth: float, high: float
) -> Top:
    """A weatherboarded oak cooper's shop `length` by `depth` metres and `high` to its eaves on
    (x, y) under an iron-grey roof: its boards lapped, a stone footing, wide doors standing open
    on a cask in the dark inside and a row of small windows."""
    m = METRE
    length, depth, high = length * m, depth * m, high * m
    top = p.hall(x, y, length, depth, high, COOPER_WALL, None)
    left, front, right, _ = top
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        for k in range(1, round(high / (0.45 * m))):
            drop = k * 0.45 * m
            _bar(
                p,
                _on_wall(a, b, 0, drop),
                _on_wall(a, b, 1, drop),
                0.05 * m,
                shade(COOPER_WALL, 0.78 * tone),
            )
        _patch(p, a, b, (0, 1), (high - 0.4 * m, high), shade(STONE, 0.8 * tone))
        for t in (0.0, 1.0):
            _bar(
                p,
                _on_wall(a, b, t, 0),
                _on_wall(a, b, t, high),
                0.2 * m,
                shade(COOPER_WALL, 0.6 * tone),
            )
    door = 3.4 * m / length
    t0 = 0.55 - door / 2
    _patch(p, left, front, (t0, t0 + door), (1.0 * m, high - 0.05 * m), (36, 30, 28))
    _cask(p, *_on_wall(left, front, 0.55, high - 0.3 * m), "up")
    for s0 in (t0 - door / 4, t0 + door):
        _patch(
            p,
            left,
            front,
            (s0, s0 + door / 4),
            (1.0 * m, high - 0.05 * m),
            shade(COOPER_WALL, 1.12),
        )
    windows = max(1, round(length / (4.5 * m)))
    for k in range(windows):
        t = (k + 0.5) / windows
        if abs(t - 0.55) > door / 2 + 0.06:
            half = 0.5 * m / length
            _patch(p, left, front, (t - half, t + half), (1.4 * m, 2.6 * m), (54, 58, 70))
            _bar(
                p,
                _on_wall(left, front, t, 1.4 * m),
                _on_wall(left, front, t, 2.6 * m),
                0.06 * m,
                shade(COOPER_WALL, 1.25),
            )
    _patch(p, front, right, (0.4, 0.6), (1.4 * m, 2.6 * m), shade((54, 58, 70), 0.85))
    p.gable(top, 0.38 * depth, COOPER_ROOF, COOPER_WALL)
    _roof_rows(p, top, 0.38 * depth, COOPER_ROOF, (0.5,))
    return top


def _drink_tarred(p: Painter, top: Top, high: float) -> None:
    """A whitewashed block's tarred black plinth 0.8 m high and the dark slate edge under its
    eaves, on its lit and shaded walls."""
    m = METRE
    left, front, right, _ = top
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (high - 0.8 * m, high), shade(TAR, tone))
        _patch(p, a, b, (0, 1), (0, 0.25 * m), shade(STILL_SLATE, tone))


def _drink_amber_door(
    p: Painter, a: Point, b: Point, span: tuple[float, float], top: float, foot: float
) -> None:
    """A pair of whisky-amber doors on the wall under the edge from `a` to `b`, `span` along it
    and from `top` to `foot` down it, in a black frame."""
    m = METRE
    t0, t1 = span
    _patch(p, a, b, (t0 - 0.01, t1 + 0.01), (top - 0.2 * m, foot), TAR)
    _patch(p, a, b, span, (top, foot), WHISKY)
    _bar(
        p,
        _on_wall(a, b, (t0 + t1) / 2, top),
        _on_wall(a, b, (t0 + t1) / 2, foot),
        0.06 * m,
        shade(WHISKY, 0.6),
    )


def _drink_malt_barn(
    p: Painter, x: float, y: float, length: float, depth: float, high: float
) -> Top:
    """A whitewashed malt barn `length` by `depth` metres and `high` to its eaves on (x, y) over a
    tarred plinth: two rows of small black-framed windows, whisky-amber doors and a black board
    lettered in gold, under slate."""
    m = METRE
    length, depth, high = length * m, depth * m, high * m
    top = p.hall(x, y, length, depth, high, WHITEWASH, None)
    left, front, right, _ = top
    _drink_tarred(p, top, high)
    count = max(3, round(length / (2.4 * m)))
    door = count - 2
    for k in range(count):
        t = (k + 0.5) / count
        _still_window(p, left, front, length, t, 0.9 * m, (1.4 * m, 2.5 * m))
        if k not in (door, door + 1):
            _still_window(p, left, front, length, t, 0.9 * m, (high - 3.0 * m, high - 1.9 * m))
    t0, t1 = door / count + 0.02, (door + 2) / count - 0.02
    _drink_amber_door(p, left, front, (t0 + 0.02, t1 - 0.02), high - 3.4 * m, high - 0.05 * m)
    _patch(p, left, front, (t0, t1), (2.9 * m, 3.8 * m), TAR)
    for k in range(6):
        s = t0 + (t1 - t0) * (0.08 + 0.15 * k)
        _bar(
            p,
            _on_wall(left, front, s, 3.35 * m),
            _on_wall(left, front, s + (t1 - t0) * 0.1, 3.35 * m),
            0.3 * m,
            GILT,
        )
    _still_window(p, front, right, depth, 0.5, 1 * m, (1.6 * m, 2.8 * m))
    p.gable(top, 0.5 * depth, STILL_SLATE, WHITEWASH)
    _roof_rows(p, top, 0.5 * depth, STILL_SLATE, (0.33, 0.66))
    return top


def _drink_still_house(
    p: Painter, x: float, y: float, length: float, depth: float, high: float, stills: int
) -> Top:
    """A whitewashed still house `length` by `depth` metres and `high` to its eaves on (x, y) over
    a tarred plinth, under slate with a louvred copper vent along its ridge: `stills` tall arched
    openings in its long wall, their whisky-amber doors folded back, each on an onion-shaped
    copper pot still gleaming over its fire, the lyne arms reaching away."""
    m = METRE
    length, depth, high = length * m, depth * m, high * m
    top = p.hall(x, y, length, depth, high, WHITEWASH, None)
    left, front, right, _ = top
    _drink_tarred(p, top, high)
    wide = 3.6 * m
    for k in range(stills):
        t = (k + 0.5) / stills
        p.polygon(_round_head(left, front, length, t, wide, high - 6.2 * m, high), (40, 34, 32))
        for side in (-1, 1):
            s0 = t + side * (wide / 2 / length)
            s1 = s0 + side * 0.8 * m / length
            _patch(
                p, left, front, tuple(sorted((s0, s1))), (high - 4.6 * m, high - 0.05 * m), WHISKY
            )
        _still_pot(p, *_on_wall(left, front, t, high - 0.15 * m), (1 if k % 2 else -1) * 1.1 * m)
    for t in (0.3, 0.7):
        _still_window(p, front, right, depth, t, 1 * m, (1.6 * m, 2.8 * m))
    rise = 0.42 * depth
    p.gable(top, rise, STILL_SLATE, WHITEWASH)
    _roof_rows(p, top, rise, STILL_SLATE, (0.5,))
    near, far = _ridge(top, rise)
    a, b = _on_wall(near, far, 0.2, 0), _on_wall(near, far, 0.8, 0)
    vent = [a, b, (b[0], b[1] - 1.1 * m), (a[0], a[1] - 1.1 * m)]
    p.polygon(vent, KETTLE, shade(KETTLE, 0.6))
    for k in range(1, 3):
        _bar(
            p,
            _on_wall(a, b, 0, -0.37 * k * m),
            _on_wall(a, b, 1, -0.37 * k * m),
            0.1 * m,
            shade(KETTLE, 0.62),
        )
    _bar(
        p, (a[0] - 0.4 * m, a[1] - 1.2 * m), (b[0] + 0.4 * m, b[1] - 1.2 * m), 0.3 * m, STILL_SLATE
    )
    return top


def _drink_bond(
    p: Painter,
    x: float,
    y: float,
    length: float,
    depth: float,
    high: float,
    doors: tuple[int, ...],
    shaded: bool = False,
) -> Top:
    """A bonded warehouse `length` by `depth` metres and `high` to its eaves on (x, y): low walls
    of whitewashed stone over a tarred plinth under slate, small barred windows, and black doors
    numbered in white, one for each of `doors`, on its lit wall or on its shaded wall if
    `shaded`."""
    m = METRE
    length_u, depth_u, high_u = length * m, depth * m, high * m
    top = p.hall(x, y, length_u, depth_u, high_u, WHITEWASH, None)
    left, front, right, _ = top
    _drink_tarred(p, top, high_u)
    a, b, wall = (front, right, depth_u) if shaded else (left, front, length_u)
    tone = 0.78 if shaded else 1.0
    slots = len(doors) * 2 + 1
    for k in range(slots):
        t = (k + 0.5) / slots
        if k % 2:
            number = doors[k // 2]
            half = 1.1 * m / wall
            _patch(
                p,
                a,
                b,
                (t - half - 0.01, t + half + 0.01),
                (high_u - 3.4 * m, high_u),
                shade(STONE, tone),
            )
            _patch(
                p,
                a,
                b,
                (t - half, t + half),
                (high_u - 3.2 * m, high_u - 0.05 * m),
                shade((34, 34, 38), tone),
            )
            cell = 0.22 * m
            for row, line in enumerate(DIGITS[number]):
                for col, mark in enumerate(line):
                    if mark == "1":
                        s = t + (col - 1.5) * cell / wall
                        d = high_u - 2.9 * m + row * cell
                        _patch(p, a, b, (s, s + cell / wall), (d, d + cell), shade(WHITEWASH, tone))
        else:
            for s in (t - 0.25 / slots, t + 0.25 / slots):
                _patch(
                    p,
                    a,
                    b,
                    (s - 0.35 * m / wall, s + 0.35 * m / wall),
                    (high_u - 4.4 * m, high_u - 3.6 * m),
                    shade(TAR, tone),
                )
    other = (left, front, length_u) if shaded else (front, right, depth_u)
    _still_window(
        p, other[0], other[1], other[2], 0.5, 0.8 * m, (high_u - 4.4 * m, high_u - 3.5 * m)
    )
    rise = 0.32 * depth_u
    p.gable(top, rise, STILL_SLATE, WHITEWASH)
    _roof_rows(p, top, rise, STILL_SLATE, (0.5,))
    return top


CHATEAU_STONE: RGB = (226, 210, 178)


def _drink_cellar(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """A wine cellar `length` by `depth` metres vaulted into a grassy bank on (x, y), its stone
    end opened by a wine-red arched door."""
    m = METRE
    bank = shade(TURF, 0.86)
    cellar = p.hall(x, y, length * m, depth * m, 2 * m, bank, None)
    _, front, right, _ = cellar
    _vault(p, cellar, 2 * m, TURF, STONE, (), shade(TURF, 0.95))
    _patch(p, front, right, (0, 1), (0, 2 * m), shade(STONE, 0.86))
    for k in range(4):
        _band(p, front, right, (0.5 * k) * m, shade(STONE, 0.7))
    p.polygon(
        _round_head(front, right, depth * m, 0.5, 2.6 * m, -0.6 * m, 2 * m), shade(STONE, 1.1)
    )
    p.polygon(_round_head(front, right, depth * m, 0.5, 2 * m, -0.3 * m, 2 * m), WINE)
    _bar(
        p,
        _on_wall(front, right, 0.5, -0.3 * m),
        _on_wall(front, right, 0.5, 2 * m),
        0.06 * m,
        shade(WINE, 0.6),
    )


def _drink_round_tower(p: Painter, x: float, y: float, r: float, tall: float, cone: float) -> None:
    """A round tower of cream stone `r` metres round and `tall` high on (x, y), with a band of
    stone at each floor and shuttered windows up its front, under a tall cone of wine-red tiles
    with a gilt vane."""
    m = METRE
    _turned(p, x, y, [(0, r * m), (tall * m, r * m)], CHATEAU_STONE)
    arc = np.linspace(0.05, math.pi - 0.05, 9)
    for h in np.arange(4.0, tall - 0.5, 4.0):
        for a0, a1 in itertools.pairwise(arc):
            _bar(
                p,
                (x + r * m * math.cos(a0), y - h * m + r * m / 2 * math.sin(a0)),
                (x + r * m * math.cos(a1), y - h * m + r * m / 2 * math.sin(a1)),
                0.2 * m,
                shade(CHATEAU_STONE, 0.86),
            )
    wx = x - 0.25 * r * m
    for h in np.arange(2.0, tall - 1.5, 4.0):
        wy = y - h * m + 0.4 * r * m
        p.rect(wx - 0.45 * m, wy - 2.0 * m, wx + 0.45 * m, wy, (54, 50, 60))
        for side in (-1, 1):
            p.rect(
                wx + side * 0.5 * m - 0.3 * m, wy - 2.0 * m, wx + side * 0.5 * m + 0.3 * m, wy, WINE
            )
    eave = y - tall * m
    _turned(
        p, x, eave + 0.3 * m, [(0, (r + 0.5) * m), (0.3 * m, (r + 0.5) * m)], shade(WINE_ROOF, 0.8)
    )
    _turned(p, x, eave, [(0, (r + 0.5) * m), (cone * m, 0)], WINE_ROOF)
    for f in (0.3, 0.6):
        h = cone * f
        w = (r + 0.5) * (1 - f)
        for a0, a1 in itertools.pairwise(arc):
            _bar(
                p,
                (x + w * m * math.cos(a0), eave - h * m + w * m / 2 * math.sin(a0)),
                (x + w * m * math.cos(a1), eave - h * m + w * m / 2 * math.sin(a1)),
                0.08 * m,
                shade(WINE_ROOF, 0.8),
            )
    tip = (x, eave - cone * m)
    _bar(p, tip, (tip[0], tip[1] - 1.4 * m), 0.1 * m, (60, 56, 54))
    p.polygon(
        [
            (tip[0], tip[1] - 1.3 * m),
            (tip[0] + 1.0 * m, tip[1] - 1.1 * m),
            (tip[0], tip[1] - 0.8 * m),
        ],
        GILT,
    )
    p.ellipse(tip[0] - 0.2 * m, tip[1] - 1.75 * m, tip[0] + 0.2 * m, tip[1] - 1.35 * m, GILT)


def _drink_dormer(p: Painter, x: float, y: float) -> None:
    """A cream dormer standing on a roof at (x, y) with a dark window and a little wine-red
    gable."""
    m = METRE
    top = p.hall(x, y, 1.4 * m, 1.0 * m, 1.5 * m, CHATEAU_STONE, None)
    _patch(p, top[0], top[1], (0.25, 0.75), (0.3 * m, 1.3 * m), (54, 50, 60))
    p.gable(top, 0.8 * m, WINE_ROOF, CHATEAU_STONE)


def _drink_chateau(
    p: Painter, x: float, y: float, length: float, depth: float, storeys: int, bays: int
) -> Top:
    """A château of cream stone `length` by `depth` metres on (x, y), `storeys` storeys of four
    and a half metres: stone quoins and bands, `bays` tall windows a storey between wine-red
    shutters, a wine-red arched door up steps under a pediment with a gilt oculus in its middle,
    and a hipped roof of wine-red tiles with dormers and cream chimneys."""
    m = METRE
    high = storeys * 4.5 * m
    length_u, depth_u = length * m, depth * m
    top = p.hall(x, y, length_u, depth_u, high, CHATEAU_STONE, None)
    left, front, right, _ = top
    _plinth(p, top, high, CHATEAU_STONE, 0.6 * m)
    for a, b in ((left, front), (front, right)):
        for t in (0.015, 0.985):
            _bar(
                p,
                _on_wall(a, b, t, 0),
                _on_wall(a, b, t, high),
                0.5 * m,
                shade(CHATEAU_STONE, 1.08),
            )
        _bar(
            p,
            _on_wall(a, b, 0, 0.25 * m),
            _on_wall(a, b, 1, 0.25 * m),
            0.5 * m,
            shade(CHATEAU_STONE, 1.1),
        )
        for k in range(1, storeys):
            drop = k * 4.5 * m
            _bar(
                p,
                _on_wall(a, b, 0, drop),
                _on_wall(a, b, 1, drop),
                0.3 * m,
                shade(CHATEAU_STONE, 0.86),
            )
    middle = bays // 2
    for k in range(bays):
        t = (k + 0.5) / bays
        for s in range(storeys):
            if k == middle and s == storeys - 1:
                continue
            head = s * 4.5 * m + 1.0 * m
            _wine_window(p, left, front, length_u, t, 1.1 * m, (head, head + 2.4 * m))
    t = (middle + 0.5) / bays
    p.polygon(
        _round_head(left, front, length_u, t, 2.4 * m, high - 3.6 * m, high),
        shade(CHATEAU_STONE, 0.82),
    )
    p.polygon(_round_head(left, front, length_u, t, 1.9 * m, high - 3.3 * m, high), WINE)
    _bar(
        p,
        _on_wall(left, front, t, high - 3.3 * m),
        _on_wall(left, front, t, high),
        0.06 * m,
        shade(WINE, 0.6),
    )
    p.hall(
        *_on_wall(left, front, t, high + 0.9 * m),
        3.2 * m,
        1.4 * m,
        0.5 * m,
        shade(STONE, 1.1),
        shade(STONE, 1.2),
    )
    for k in range(3):
        t = (k + 0.5) / 3
        for s in range(storeys):
            head = s * 4.5 * m + 1.0 * m
            _wine_window(p, front, right, depth_u, t, 1.1 * m, (head, head + 2.4 * m))
    rise = 0.5 * depth_u
    _hip(p, top, rise, 0.18, WINE_ROOF, (0.35, 0.7))
    t = (middle + 0.5) / bays
    half = 1.6 * m / length_u
    base_l, base_r = (
        _on_wall(left, front, t - half * 1.6, 0),
        _on_wall(left, front, t + half * 1.6, 0),
    )
    peak = _on_wall(left, front, t, -2.6 * m)
    p.polygon([base_l, base_r, peak], CHATEAU_STONE, shade(CHATEAU_STONE, 0.6))
    _bar(p, base_l, peak, 0.25 * m, shade(CHATEAU_STONE, 1.1))
    _bar(p, peak, base_r, 0.25 * m, shade(CHATEAU_STONE, 1.1))
    ox, oy = _on_wall(left, front, t, -0.9 * m)
    p.ellipse(ox - 0.6 * m, oy - 0.6 * m, ox + 0.6 * m, oy + 0.6 * m, GILT)
    for k in range(bays):
        if abs(k - middle) < 1:
            continue
        dx, dy = _on_roof(top, rise, (k + 0.5) / bays, 0.25)
        _drink_dormer(p, dx, dy)
    for t in (0.2, 0.8):
        cx, cy = _on_roof(top, rise, t, 0.95)
        p.hall(cx, cy + 0.4 * m, 0.9 * m, 1.4 * m, 2 * m, CHATEAU_STONE, (80, 70, 66))
    return top


def _drink_terrace(p: Painter, x: float, y: float, length: float, high: float, rows: int) -> None:
    """A terrace of vines `length` metres long cut into a hillside on (x, y): a stone retaining
    wall `high` metres tall in front of a grassy step 2.6 m deep planted with `rows` rows of
    vines along it."""
    m = METRE
    depth = 2.6 * rows
    top = p.hall(x, y, length * m, depth * m, high * m, STONE, shade(TURF, 0.9))
    left, front, right, back = top
    for k in range(1, max(2, round(high / 0.6))):
        _band(p, left, front, k * 0.6 * m, shade(STONE, 0.84))
    _bar(
        p,
        _on_wall(left, front, 0, 0.1 * m),
        _on_wall(left, front, 1, 0.1 * m),
        0.25 * m,
        shade(STONE, 1.15),
    )
    posts = max(2, round((length - 1) / 2))
    for k in range(rows):
        f = (rows - k - 0.5) / rows
        start = _on_wall(_on_wall(left, back, f, 0), _on_wall(front, right, f, 0), 0.04, 0)
        _wine_vines(p, *start, posts)


def _drink_press_shed(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """An open press shed `length` by `depth` metres on (x, y): a wine-red tiled roof on stout
    timber posts over a back wall of cream stone, sheltering a wooden basket press and casks on
    a rack."""
    m = METRE
    high = 4.2 * m
    foot = _footing(x, y, length * m, depth * m)
    left, front, right, back = foot
    p.polygon([left, front, right, back], shade(STONE, 0.9))
    p.hall(*_on_wall(back, right, 0.5, 0), length * m, 0.6 * m, high, CHATEAU_STONE, None)
    _wine_press(p, *_on_wall(_on_wall(left, front, 0.3, 0), _on_wall(back, right, 0.3, 0), 0.45, 0))
    _cask_rack(
        p, *_on_wall(_on_wall(left, front, 0.58, 0), _on_wall(back, right, 0.58, 0), 0.55, 0), 4, 2
    )
    for t in (0.02, 0.98):
        for corner in (_on_wall(left, front, t, 0), _on_wall(back, right, t, 0)):
            _bar(p, corner, (corner[0], corner[1] - high), 0.3 * m, shade(TIMBER, 0.75))
    roof = tuple((c[0], c[1] - high) for c in foot)
    p.gable(roof, 0.45 * depth * m, WINE_ROOF, CHATEAU_STONE)
    _roof_rows(p, roof, 0.45 * depth * m, WINE_ROOF, (0.35, 0.7))


def _drink_fountain(p: Painter, x: float, y: float) -> None:
    """A round stone fountain on (x, y): a basin 5 m across of water with a two-tier column in
    its middle spilling water."""
    m = METRE
    _turned(p, x, y, [(0, 2.5 * m), (0.6 * m, 2.5 * m)], STONE, (110, 150, 176))
    p.blob(x - 1.6 * m, y - 1.0 * m, x - 0.2 * m, y - 0.6 * m, (170, 204, 224))
    _turned(p, x, y - 0.6 * m, [(0, 0.3 * m), (1.4 * m, 0.25 * m)], shade(STONE, 1.1))
    _turned(
        p,
        x,
        y - 2.0 * m,
        [(0, 0.2 * m), (0.2 * m, 1.1 * m), (0.35 * m, 1.1 * m)],
        shade(STONE, 1.1),
        (110, 150, 176),
    )
    _turned(p, x, y - 2.35 * m, [(0, 0.15 * m), (0.9 * m, 0.1 * m)], shade(STONE, 1.1))
    p.soft([(x - 0.5 * m, y - 3.9 * m, x + 0.5 * m, y - 3.0 * m)], (226, 240, 248), 0.8, 0.15)


IRON_CLAD: RGB = (128, 132, 140)

BOILER: RGB = (48, 48, 54)


def _drink_steam_shop(p: Painter, x: float, y: float, length: float, depth: float) -> Top:
    """A cooper's steam shop `length` by `depth` metres on (x, y), clad in ribbed iron sheets
    over a timber sill: a wide door standing open on casks softening in the steam inside, a
    louvred vent along its ridge breathing steam, and an upright iron boiler at its end with a
    thin iron flue."""
    m = METRE
    high = 5 * m
    length_u, depth_u = length * m, depth * m
    top = p.hall(x, y, length_u, depth_u, high, IRON_CLAD, None)
    left, front, right, _ = top
    for a, b, wall, tone in ((left, front, length_u, 1.0), (front, right, depth_u, 0.78)):
        for k in range(1, round(wall / (0.5 * m))):
            t = k / round(wall / (0.5 * m))
            _bar(
                p,
                _on_wall(a, b, t, 0),
                _on_wall(a, b, t, high),
                0.05 * m,
                shade(IRON_CLAD, 0.82 * tone),
            )
        _patch(p, a, b, (0, 1), (high - 0.5 * m, high), shade(COOPER_WALL, 0.8 * tone))
    _patch(p, left, front, (0.3, 0.7), (1.2 * m, high - 0.05 * m), (44, 40, 40))
    for t in (0.4, 0.6):
        _cask(p, *_on_wall(left, front, t, high - 0.3 * m), "up")
    p.soft(
        [(*_on_wall(left, front, 0.32, 1.4 * m), *_on_wall(left, front, 0.68, 3.4 * m))],
        (240, 240, 236),
        0.55,
        0.4,
    )
    rise = 0.4 * depth_u
    p.gable(top, rise, COOPER_ROOF, IRON_CLAD)
    _roof_rows(p, top, rise, COOPER_ROOF, (0.5,))
    near, far = _ridge(top, rise)
    a, b = _on_wall(near, far, 0.15, 0), _on_wall(near, far, 0.85, 0)
    p.polygon([a, b, (b[0], b[1] - 1.0 * m), (a[0], a[1] - 1.0 * m)], shade(IRON_CLAD, 1.1))
    for k in (1, 2):
        _bar(
            p,
            _on_wall(a, b, 0, -0.33 * k * m),
            _on_wall(a, b, 1, -0.33 * k * m),
            0.1 * m,
            shade(IRON_CLAD, 0.6),
        )
    _bar(
        p, (a[0] - 0.4 * m, a[1] - 1.1 * m), (b[0] + 0.4 * m, b[1] - 1.1 * m), 0.3 * m, COOPER_ROOF
    )
    mid = _on_wall(a, b, 0.5, -1.2 * m)
    _drink_steam(p, *mid, 3)
    return top


def _drink_boiler(p: Painter, x: float, y: float) -> None:
    """An upright iron boiler 3.4 m high standing on (x, y), riveted in bands, with a firedoor
    glowing at its foot and a thin iron flue rising from its dome trailing smoke."""
    m = METRE
    _turned(p, x, y, [(0, 0.9 * m), (3 * m, 0.9 * m), (3.4 * m, 0.5 * m)], BOILER)
    arc = np.linspace(0.1, math.pi - 0.1, 8)
    for h in (0.8, 1.8, 2.7):
        for a0, a1 in itertools.pairwise(arc):
            _bar(
                p,
                (x + 0.9 * m * math.cos(a0), y - h * m + 0.45 * m * math.sin(a0)),
                (x + 0.9 * m * math.cos(a1), y - h * m + 0.45 * m * math.sin(a1)),
                0.06 * m,
                shade(BOILER, 1.6),
            )
    p.rect(x - 0.6 * m, y - 0.6 * m, x - 0.1 * m, y - 0.15 * m, (246, 150, 54))
    _turned(p, x, y - 3.4 * m, [(0, 0.2 * m), (4 * m, 0.18 * m)], BOILER)
    p.soft(
        [
            (x - 0.4 * m, y - 9.0 * m, x + 1.0 * m, y - 7.6 * m),
            (x + 0.4 * m, y - 10.6 * m, x + 2.2 * m, y - 9.0 * m),
        ],
        (176, 174, 176),
        0.6,
        0.3,
    )


def _drink_logs(p: Painter, x: float, y: float, rows: tuple[int, ...]) -> None:
    """Oak logs 4 m long stacked on (x, y) along the lit walls, `rows[0]` side by side on the
    ground and fewer nestled on top, their pale sawn ends ringed on the shaded side."""
    m = METRE
    r, long = 0.3, 4.0
    bark, end = (104, 80, 58), (214, 182, 132)
    for level, count in enumerate(rows):
        for k in range(count):
            v = -(k + 0.5 * level) * 2 * r
            z = r + level * 1.7 * r
            top = [_cask_at(x, y, u, v, z + r) for u in (0, long)]
            low = [_cask_at(x, y, u, v - r, z) for u in (long, 0)]
            p.polygon([*top, *low], bark)
            _bar(p, *top, 0.06 * m, shade(bark, 1.3))
            cx, cy = _cask_at(x, y, long, v - 0.15, z)
            p.ellipse(cx - r * m * 0.8, cy - r * m, cx + r * m * 0.8, cy + r * m, end)
            p.blob(cx - 0.08 * m, cy - 0.08 * m, cx + 0.08 * m, cy + 0.08 * m, shade(end, 0.7))


def _drink_sign_cask(p: Painter, x: float, y: float) -> None:
    """A great oak cask 2.6 m high stood on end on a roof's ridge at (x, y) as the cooperage's
    sign, its staves lit on the left and bound with four gilt hoops."""
    m = METRE
    profile = [(0, 0.85), (0.6, 1.0), (1.3, 1.08), (2.0, 1.0), (2.6, 0.85)]
    _turned(p, x, y, [(h * m, w * m) for h, w in profile], OAK, shade(OAK, 1.2))
    arc = np.linspace(0.05, math.pi - 0.05, 10)
    for h in (0.3, 0.85, 1.75, 2.3):
        w = float(np.interp(h, [a for a, _ in profile], [b for _, b in profile]))
        for a0, a1 in itertools.pairwise(arc):
            _bar(
                p,
                (x + w * m * math.cos(a0), y - h * m + w * m / 2 * math.sin(a0)),
                (x + w * m * math.cos(a1), y - h * m + w * m / 2 * math.sin(a1)),
                0.14 * m,
                GILT,
            )
    p.blob(x - 0.75 * m, y - 1.9 * m, x - 0.45 * m, y - 0.6 * m, shade(OAK, 1.25))


CEDAR: RGB = (176, 120, 78)

BAMBOO: RGB = (150, 146, 84)

RICE_STRAW: RGB = (218, 196, 132)


def _drink_kura(
    p: Painter, x: float, y: float, length: float, depth: float, high: float, door: bool = True
) -> Top:
    """A kura storehouse `length` by `depth` metres and `high` to its eaves on (x, y): white
    plaster over a namako base of grey tiles in a lattice of white joints with a black rail on
    it, a black band under its eaves, rows of small barred windows behind thick white shutters,
    a heavy plastered door if `door`, and a roof of dark kawara tiles with a thick ridge."""
    m = METRE
    length_u, depth_u, high_u = length * m, depth * m, high * m
    top = p.hall(x, y, length_u, depth_u, high_u, PLASTER, None)
    left, front, right, _ = top
    _sake_namako(p, left, front, length_u, (high_u - 2.4 * m, high_u), 1.0)
    _sake_namako(p, front, right, depth_u, (high_u - 2.4 * m, high_u), 0.78)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _bar(
            p,
            _on_wall(a, b, 0, 0.15 * m),
            _on_wall(a, b, 1, 0.15 * m),
            0.3 * m,
            shade(CHARRED, tone),
        )
    rows = [1.2 * m] + ([4.2 * m] if high >= 9 else [])
    count = max(2, round(length / 4))
    for k in range(count):
        t = (k + 0.5) / count
        half = 0.6 * m / length_u
        for drop in rows:
            if door and k == count // 2 and drop > 2 * m:
                continue
            _patch(
                p,
                left,
                front,
                (t - half * 1.4, t + half * 1.4),
                (drop, drop + 1.2 * m),
                shade(PLASTER, 0.72),
            )
            _patch(
                p,
                left,
                front,
                (t - half, t + half),
                (drop + 0.15 * m, drop + 1.05 * m),
                (40, 38, 42),
            )
            for j in (-1, 0, 1):
                s = t + half * 0.5 * j
                _bar(
                    p,
                    _on_wall(left, front, s, drop + 0.15 * m),
                    _on_wall(left, front, s, drop + 1.05 * m),
                    0.06 * m,
                    (110, 106, 100),
                )
    if door:
        t = (count // 2 + 0.5) / count
        half = 1.2 * m / length_u
        _patch(
            p,
            left,
            front,
            (t - half * 1.3, t + half * 1.3),
            (high_u - 3.6 * m, high_u),
            shade(PLASTER, 0.8),
        )
        _patch(
            p, left, front, (t - half, t + half), (high_u - 3.3 * m, high_u), shade(PLASTER, 0.92)
        )
        _bar(
            p,
            _on_wall(left, front, t, high_u - 3.3 * m),
            _on_wall(left, front, t, high_u),
            0.08 * m,
            shade(PLASTER, 0.6),
        )
    _patch(p, front, right, (0.4, 0.6), (1.2 * m, 2.6 * m), shade(PLASTER, 0.6))
    _sake_roof(p, top, 0.45 * depth_u)
    return top


def _drink_sake_shop(
    p: Painter, x: float, y: float, length: float, depth: float, ball: float
) -> Top:
    """A sake brewer's shop `length` by `depth` metres on (x, y), 4.6 m to its eaves: walls of
    black charred timber boards over a white plaster footing, a lattice door under an indigo
    noren, a lattice window, a tiled roof, and a sugidama `ball` times the usual size hung under
    its eaves."""
    m = METRE
    s_len, s_deep, s_high = length * m, depth * m, 4.6 * m
    shop = p.hall(x, y, s_len, s_deep, s_high, CHARRED, None)
    left, front, right, _ = shop
    for a, b, wall, tone in ((left, front, s_len, 1.0), (front, right, s_deep, 0.78)):
        _patch(p, a, b, (0, 1), (0, 0.9 * m), shade(PLASTER, tone))
        boards = round(wall / (0.35 * m))
        for k in range(1, boards):
            _bar(
                p,
                _on_wall(a, b, k / boards, 0.9 * m),
                _on_wall(a, b, k / boards, s_high),
                0.04 * m,
                shade(CHARRED, 1.5 * tone),
            )
        _bar(
            p,
            _on_wall(a, b, 0, 0.95 * m),
            _on_wall(a, b, 1, 0.95 * m),
            0.12 * m,
            shade(LATTICE, 0.6 * tone + 0.2),
        )
    d0, d1 = 0.55 - 1.6 * m / s_len, 0.55 + 1.6 * m / s_len
    _patch(p, left, front, (d0, d1), (1.9 * m, s_high), shade(LATTICE, 0.6))
    for k in range(9):
        s = d0 + (d1 - d0) * (k + 0.5) / 9
        _bar(
            p,
            _on_wall(left, front, s, 1.9 * m),
            _on_wall(left, front, s, s_high),
            0.06 * m,
            LATTICE,
        )
    _sake_noren(p, left, front, (d0 + 0.01, d1 - 0.01), 1.6 * m)
    w0, w1 = 0.08, 0.08 + 2.4 * m / s_len
    _patch(p, left, front, (w0, w1), (1.6 * m, 3.2 * m), (40, 34, 30))
    for k in range(7):
        s = w0 + (w1 - w0) * (k + 0.5) / 7
        _bar(
            p,
            _on_wall(left, front, s, 1.6 * m),
            _on_wall(left, front, s, 3.2 * m),
            0.06 * m,
            LATTICE,
        )
    _sake_roof(p, shop, 0.45 * s_deep)
    bx, by = _on_wall(left, front, min(0.95, d1 + 1.0 * m / s_len), 0.25 * m)
    if ball > 1:
        _bar(p, (bx, by), (bx, by + 0.5 * (ball - 1) * m), 0.06 * m, (60, 50, 40))
        by += 0.5 * (ball - 1) * m
        cx, cy, r = bx, by + 1.05 * ball * m, 0.55 * ball * m
        _bar(p, (bx, by), (bx, by + 0.5 * m), 0.06 * m, (60, 50, 40))
        for a in np.linspace(0, 2 * math.pi, 18)[:-1]:
            tx, ty = cx + r * 1.02 * math.cos(a), cy + r * 1.02 * math.sin(a)
            p.blob(tx - 0.16 * m, ty - 0.16 * m, tx + 0.16 * m, ty + 0.16 * m, shade(SUGI, 0.8))
        p.blob(cx - r, cy - r, cx + r, cy + r, SUGI)
        p.blob(cx - 0.1 * m, cy - 0.05 * m, cx + r * 0.95, cy + r * 0.95, shade(SUGI, 0.82))
        p.blob(cx - r * 0.75, cy - r * 0.75, cx + 0.05 * m, cy, shade(SUGI, 1.22))
    else:
        _sake_ball(p, bx, by)
    return shop


def _drink_drum(
    p: Painter,
    x: float,
    y: float,
    length: float,
    r: float,
    body: RGB,
    bands: tuple[float, ...],
    band: RGB,
    open_end: bool,
) -> None:
    """A drum `length` metres long and `r` round lying on its side on (x, y) with its end to the
    lit side, like a straw rice bale or a great cedar vat: its body lit along the top, bound with
    `bands` (offsets along it) in `band`, and its end closed in pale straw or, if `open_end`,
    open on its dark inside."""
    m = METRE
    half = length / 2
    rim = np.linspace(0, 2 * math.pi, 25)

    def at(v: float, rr: float, a: float) -> Point:
        return _cask_at(x, y, rr * math.cos(a), v, r + rr * math.sin(a))

    for v in np.linspace(half, -half, 7):
        centre = at(v, 0, 0)
        for a0, a1 in itertools.pairwise(rim):
            mid = (a0 + a1) / 2
            tone = 0.86 - 0.08 * math.cos(mid) + 0.18 * math.sin(mid)
            p.polygon([centre, at(v, r, a0), at(v, r, a1)], shade(body, tone))
    for v in bands:
        for a0, a1 in itertools.pairwise(rim):
            if math.cos(a0) + math.sin(a0) > 0 and math.cos(a1) + math.sin(a1) > 0:
                _bar(p, at(v, r * 1.01, a0), at(v, r * 1.01, a1), 0.12 * m, band)
    p.polygon([at(-half, r, a) for a in rim], shade(body, 0.8))
    inside = (52, 38, 30) if open_end else shade(body, 1.15)
    p.polygon([at(-half, r * 0.86, a) for a in rim], inside)
    if open_end:
        p.polygon([at(-half + 0.6, r * 0.86, a) for a in rim[6:19]], shade(body, 0.6))


def _drink_komodaru(p: Painter, x: float, y: float, rows: tuple[int, ...]) -> None:
    """Komodaru (_cask_straw) stood on end in a stepped pyramid from (x, y): `rows[0]` along the
    lit walls on the ground and each row above set back over the gaps of the one below."""
    gap = 0.82
    p.soft(
        [(*_cask_at(x, y, -0.6, 0.5), *_cask_at(x, y, (rows[0] - 1) * gap + 1, -0.6))],
        (34, 30, 24),
        0.4,
        0.2,
    )
    for level, count in enumerate(rows):
        for k in range(count):
            _cask_straw(p, *_cask_at(x, y, (k + 0.5 * level) * gap, 0, 0.86 * level))


def _drink_bales(p: Painter, x: float, y: float, rows: tuple[int, ...]) -> None:
    """Straw bales of rice 0.9 m long lying with their ends to the lit side in a pile from
    (x, y): `rows[0]` side by side on the ground and fewer nestled on top."""
    gap, r = 0.62, 0.3
    for level, count in enumerate(rows):
        for k in range(count):
            u = (k + 0.5 * level) * gap
            bx, by = _cask_at(x, y, u, 0, level * 1.7 * r)
            _drink_drum(p, bx, by, 0.9, r, RICE_STRAW, (0.25, -0.25), shade(RICE_STRAW, 0.6), False)


def big_sake_brewery() -> Sprite:
    """A sake brewery grown into a works: two kura storehouses of white plaster over namako bases
    of grey tiles under heavy roofs of dark kawara, a shop of black charred timber with an
    indigo noren over its door and a green sugidama under its eaves, a stepped stack of red-
    labelled komodaru, rice bales piled by the kura door and two great cedar vats drying on their
    sides in the yard."""
    m = METRE
    p = Painter(60, 40, (27, 26))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10) * m, (v - 3) * m)

    p.polygon(
        [at(-11, -6), at(34, -6), at(34, 3), at(20, 3), at(20, 5.5), at(-11, 5.5)], (204, 196, 176)
    )
    p.shadow(18, 5, 4)
    _drink_kura(p, *at(9, 9.5), 20, 7, 7)
    _drink_kura(p, *at(27.5, 6.5), 13, 7, 6.5, door=False)
    _drink_komodaru(p, *at(2, 4.6), (6, 5))
    _drink_bales(p, *at(13, 4.4), (5, 4, 3))
    _drink_sake_shop(p, *at(-5, 2.4), 8, 5, 1)
    for u in (22.5, 26.5):
        _drink_drum(p, *at(u, -2.4), 2.2, 1.2, CEDAR, (0.7, -0.7), BAMBOO, True)
    for u, v in ((5, -1.6), (5.9, -2.6)):
        _cask_straw(p, *at(u, v))
    return p.sprite()


def _drink_wall(p: Painter, x: float, y: float, length: float) -> None:
    """A compound wall `length` metres long running along the lit walls from (x, y), 2.6 m high:
    white plaster over a namako base, capped with a little roof of dark kawara tiles."""
    m = METRE
    top = p.hall(*_cask_at(x, y, length / 2, 0), length * m, 0.7 * m, 2.6 * m, PLASTER, None)
    left, front, right, _ = top
    _sake_namako(p, left, front, length * m, (1.3 * m, 2.6 * m), 1.0)
    _patch(p, front, right, (0, 1), (1.3 * m, 2.6 * m), shade(NAMAKO, 0.78))
    _sake_roof(p, top, 0.5 * m)


def _drink_gate(p: Painter, x: float, y: float, ball: float) -> None:
    """A roofed gate 5.6 m wide standing on (x, y) in a compound wall: two stout dark posts with
    a beam across, its leaves folded back, under a heavy gabled roof of dark kawara tiles, with a
    great sugidama `ball` times the usual size hung from the beam in its middle."""
    m = METRE
    post = (58, 50, 46)
    for u in (-2.2, 2.2):
        for v in (0.5, -0.5):
            _bar(
                p,
                _cask_at(x, y, u, v),
                _cask_at(x, y, u, v, 4.0),
                0.32 * m,
                shade(post, 1.0 if v < 0 else 0.8),
            )
    _bar(p, _cask_at(x, y, -3.0, -0.5, 4.5), _cask_at(x, y, 3.0, -0.5, 4.5), 0.45 * m, post)
    for u0, u1 in ((-2.5, -1.3), (1.3, 2.5)):
        p.polygon(
            [
                _cask_at(x, y, u0, -0.6, 0),
                _cask_at(x, y, u1, -0.6, 0),
                _cask_at(x, y, u1, -0.6, 3.4),
                _cask_at(x, y, u0, -0.6, 3.4),
            ],
            shade(CHARRED, 1.6),
        )
    roof = p.hall(*_cask_at(x, y, 0, 0, 4.9), 7.6 * m, 3.4 * m, 0.6 * m, shade(post, 0.9), None)
    _sake_roof(p, roof, 1.8 * m)
    bx, by = _cask_at(x, y, 0.3, -0.5, 4.5)
    cx, cy, r = bx, by + (0.5 + 0.55 * ball) * m, 0.55 * ball * m
    _bar(p, (bx, by), (bx, by + 0.5 * m), 0.06 * m, (60, 50, 40))
    for a in np.linspace(0, 2 * math.pi, 18)[:-1]:
        tx, ty = cx + r * 1.02 * math.cos(a), cy + r * 1.02 * math.sin(a)
        p.blob(tx - 0.16 * m, ty - 0.16 * m, tx + 0.16 * m, ty + 0.16 * m, shade(SUGI, 0.8))
    p.blob(cx - r, cy - r, cx + r, cy + r, SUGI)
    p.blob(cx - 0.1 * m, cy - 0.05 * m, cx + r * 0.95, cy + r * 0.95, shade(SUGI, 0.82))
    p.blob(cx - r * 0.75, cy - r * 0.75, cx + 0.05 * m, cy, shade(SUGI, 1.22))


def mega_sake_brewery() -> Sprite:
    """A great sake brewery behind its own wall: a tall kura storehouse two rows of windows high
    of white plaster over a namako base under a heavy roof of dark kawara, a second kura beside
    it, a long shop of black charred timber with an indigo noren over its door and a sugidama
    under its eaves, stepped pyramids of red-labelled komodaru, piles of rice bales and a row of
    great cedar vats drying on their sides in the yard, all closed by a white wall capped with
    tiles and entered through a roofed gate hung with a great green sugidama."""
    m = METRE
    p = Painter(70, 50, (32, 32))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 10) * m, (v - 3) * m)

    p.polygon([at(-15, -9), at(36, -9), at(36, 6), at(-15, 6)], (204, 196, 176))
    p.shadow(22, 6, 5)
    _drink_kura(p, *at(10, 11), 24, 8, 9.5)
    _drink_kura(p, *at(30, 8), 11, 7, 7, door=False)
    _drink_komodaru(p, *at(13, 5.2), (8, 7, 6, 5))
    _drink_bales(p, *at(1, 5.2), (6, 5, 4, 3))
    _drink_sake_shop(p, *at(-8, 1.6), 12, 6, 1)
    for u in (22, 26, 30):
        _drink_drum(p, *at(u, 0.6), 2.2, 1.2, CEDAR, (0.7, -0.7), BAMBOO, True)
    _drink_komodaru(p, *at(16, -3.4), (6, 5, 4))
    _drink_wall(p, *at(-14, -9), 22)
    _drink_wall(p, *at(14, -9), 22)
    _drink_gate(p, *at(11, -9), 2.2)
    return p.sprite()


CURING_BRICK: RGB = (152, 92, 70)

GATE_BRICK: RGB = (162, 86, 62)

COPING: RGB = (206, 198, 184)


def _craft_butcher_house(
    p: Painter, x: float, y: float, long: float, deep: float, floors: int
) -> Top:
    """A pork butcher's house on the street `long` along the lit walls and `deep`, of `floors`
    storeys over its shop: the shop front in white tiles under a red fascia lettered in gold,
    windows hung with sausages and hams under red and pink striped awnings either side of a
    glazed door, half-timbered upper floors of cream plaster and dark oak with casement windows,
    and a steep red-brown roof with dormers and chimneys. Returns its top."""
    m = METRE
    shop, storey = 3.2 * m, 2.8 * m
    upper = floors * storey
    high = upper + shop
    top = p.hall(x, y, long, deep, high, BUTCHER_PLASTER, None)
    left, front, right, _ = top
    _plinth(p, top, high, (170, 160, 146), 0.5 * m)
    for f in range(floors):
        drop = (f * storey, (f + 1) * storey)
        _craft_framing(p, left, front, long, drop, CRAFT_OAK)
        _craft_framing(p, front, right, deep, drop, shade(CRAFT_OAK, 0.8))
    _craft_framing(p, front, right, deep, (upper, high - 0.5 * m), shade(CRAFT_OAK, 0.8))
    _patch(p, left, front, (0, 1), (upper, high - 0.5 * m), WHITE_TILE)
    _patch(p, left, front, (0.02, 0.98), (upper, upper + 0.45 * m), (150, 36, 44))
    for t0 in np.arange(0.36, 0.62, 0.06):
        _craft_beam(
            p,
            left,
            front,
            (t0, upper + 0.22 * m),
            (t0 + 0.04, upper + 0.22 * m),
            0.16 * m,
            CRAFT_GILT,
        )
    door = 1.0 * m / long
    for span in ((0.05, 0.5 - door * 1.6), (0.5 + door * 1.6, 0.95)):
        _craft_meat_window(p, left, front, long, span, (upper + 0.8 * m, high - 0.5 * m))
        _craft_awning(
            p, left, front, (span[0] - 0.01, span[1] + 0.01), upper + 0.45 * m, 0.7 * m, STRIPES
        )
    _patch(p, left, front, (0.5 - door, 0.5 + door), (upper + 0.6 * m, high), DOOR)
    _patch(
        p,
        left,
        front,
        (0.5 - door * 0.7, 0.5 + door * 0.7),
        (upper + 0.8 * m, upper + 1.7 * m),
        SHOP_GLOW,
    )
    count = max(2, round(long / (2.6 * m)))
    for f in range(floors):
        head = f * storey + 0.7 * m
        for k in range(count):
            _craft_casement(
                p, left, front, long, (k + 0.5) / count, 1.1 * m, (head, head + 1.3 * m), WINDOW
            )
        for t in (0.3, 0.7):
            _craft_casement(
                p, front, right, deep, t, 1.0 * m, (head, head + 1.3 * m), shade(WINDOW, 0.8)
            )
    rise = deep * 0.6
    p.gable(top, rise, BUTCHER_ROOF, BUTCHER_PLASTER)
    _tiles(p, top, rise, BUTCHER_ROOF, (0.25, 0.5, 0.75))
    ridge_back, ridge_front = _ridge(top, rise)
    _bar(
        p,
        ridge_front,
        ((front[0] + right[0]) / 2, (front[1] + right[1]) / 2),
        0.22 * m,
        shade(CRAFT_OAK, 0.8),
    )
    for t in (0.25, 0.75):
        foot = _on_wall(front, right, t, 0)
        _bar(p, foot, _on_wall(ridge_front, foot, 0.5, 0), 0.18 * m, shade(CRAFT_OAK, 0.8))
    dormers = max(1, round(long / (6 * m)))
    for k in range(dormers):
        d = p.hall(
            *_on_wall(left, front, (k + 0.5) / dormers, -1.4 * m),
            1.8 * m,
            1.6 * m,
            1.6 * m,
            BUTCHER_PLASTER,
            None,
        )
        _craft_casement(p, d[0], d[1], 1.8 * m, 0.5, 1.0 * m, (0.4 * m, 1.3 * m), WINDOW)
        _gable_across(p, d, 1.0 * m, BUTCHER_ROOF, BUTCHER_PLASTER)
    for along in (0.2, 0.8):
        cx, cy = _on_wall(ridge_back, ridge_front, along, 0)
        p.hall(cx, cy + 0.4 * m, 0.9 * m, 0.9 * m, 1.6 * m, BRICK, (60, 50, 48))
    return top


def _craft_curing_hall(
    p: Painter, x: float, y: float, long: float, deep: float, high: float
) -> Top:
    """The works where the meat is cured and the sausage made, `long` along the lit walls and
    `deep`, `high` to its eaves, of smoke-browned brick on (x, y): cream string courses, rows of
    white-framed windows, wide doors open onto the yard, and a louvred vent along its tiled
    ridge. Returns its top."""
    m = METRE
    top = p.hall(x, y, long, deep, high, CURING_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, CURING_BRICK, 0.6 * m)
    for drop in (0.4 * m, high - 3.4 * m):
        _craft_band(p, top, drop, 0.3 * m, COPING)
    floors = max(1, round(high / (3.6 * m)))
    count = max(2, round(long / (2.4 * m)))
    doors = (round(count * 0.3), round(count * 0.7))
    for f in range(floors):
        head = high - (f + 1) * (high / floors) + 0.9 * m
        for k in range(count):
            if f == 0 and k in doors:
                continue
            _craft_casement(
                p, left, front, long, (k + 0.5) / count, 1.1 * m, (head, head + 1.6 * m), WINDOW
            )
        for k in range(2):
            _craft_casement(
                p,
                front,
                right,
                deep,
                (k + 0.5) / 2,
                1.1 * m,
                (head, head + 1.6 * m),
                shade(WINDOW, 0.8),
            )
    for k in doors:
        t, half = (k + 0.5) / count, 1.2 * m / long
        _patch(p, left, front, (t - half, t + half), (high - 3.0 * m, high), (58, 48, 44))
        _patch(p, left, front, (t - half, t + half), (high - 1.4 * m, high), (122, 96, 80))
        _patch(
            p,
            left,
            front,
            (t - half * 1.3, t + half * 1.3),
            (high - 3.3 * m, high - 3.0 * m),
            COPING,
        )
    rise = deep * 0.4
    p.gable(top, rise, BUTCHER_ROOF, CURING_BRICK)
    _tiles(p, top, rise, BUTCHER_ROOF, (0.3, 0.6))
    ridge_back, ridge_front = _ridge(top, rise)
    vx, vy = _on_wall(ridge_back, ridge_front, 0.5, 0)
    vent = p.hall(vx, vy + 0.4 * m, long * 0.5, 1.2 * m, 0.9 * m, shade(CURING_BRICK, 0.9), None)
    for d in (0.3 * m, 0.6 * m):
        _craft_beam(p, vent[0], vent[1], (0.02, d), (0.98, d), 0.1 * m, (60, 52, 48))
    p.gable(vent, 0.5 * m, BUTCHER_ROOF, CURING_BRICK)
    return top


def _craft_street_wall(p: Painter, a: Point, b: Point) -> None:
    """A low brick yard wall 1.6 m high on the ground from `a` to `b`, with a stone coping."""
    m = METRE
    high = 1.6 * m
    p.polygon([a, b, (b[0], b[1] - high), (a[0], a[1] - high)], GATE_BRICK, shade(GATE_BRICK, 0.6))
    for k in range(1, 6):
        _bar(
            p,
            _on_wall(a, b, 0, k * 0.27 * m),
            _on_wall(a, b, 1, k * 0.27 * m),
            0.04 * m,
            shade(GATE_BRICK, 0.85),
        )
    _bar(p, (a[0], a[1] - high), (b[0], b[1] - high), 0.25 * m, COPING)


def _craft_pig_gate(p: Painter, x: float, y: float, wide: float, pig: float) -> None:
    """A yard gateway `wide` metres across on the ground at (x, y) along the lit walls: two brick
    piers 3.4 m tall under stone caps, a wrought-iron arch between them and the gilt pig `pig`
    metres long standing on its crown."""
    m = METRE
    high = 3.4 * m
    tops = []
    for side in (-1, 1):
        px, py = _iso(x, y, side * wide / 2 * m, 0)
        pier = p.hall(px, py, 0.9 * m, 0.9 * m, high, GATE_BRICK, None)
        _craft_band(p, pier, 0.6 * m, 0.2 * m, shade(GATE_BRICK, 0.8))
        p.hall(px, py - high, 1.2 * m, 1.2 * m, 0.3 * m, COPING, shade(COPING, 1.05))
        tops.append((px, py - high - 0.3 * m))
    a, b = tops
    arc = [
        (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f - 1.8 * m * math.sin(math.pi * f))
        for f in np.linspace(0, 1, 9)
    ]
    inner = [(ax, ay + 0.45 * m) for ax, ay in arc]
    for k in range(8):
        _bar(p, arc[k], arc[k + 1], 0.18 * m, IRON)
        _bar(p, inner[k], inner[k + 1], 0.1 * m, IRON)
        if 0 < k < 8:
            _bar(p, arc[k], inner[k], 0.06 * m, IRON)
    crown = arc[4]
    _bar(p, crown, (crown[0], crown[1] - 0.4 * m), 0.12 * m, IRON)
    _craft_gilt_pig(p, crown[0], crown[1] - 0.4 * m - 0.5 * pig * m, pig * m)
