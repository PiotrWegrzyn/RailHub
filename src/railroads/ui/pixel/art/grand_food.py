"""Big and mega wineries, breweries, distilleries, canneries, sugar refineries, meat works,
sausage makers and cooperages, each drawn as a building of its own: a cream château among
terraced vines, a brick brewhouse under a copper dome, a whitewashed still house among bonded
warehouses, a cannery quarter round a water tank made like a tin can, a red-brick refinery banded in
sugar white and blue among its beet,
a brick slaughterhouse with a white cold store, a half-timbered sausage works with smokehouses and
a timber cooperage with barrels firing over the coals."""

import math
from collections.abc import Callable

import numpy as np

from railroads.ui.pixel.art.industry import (
    COLD_TILE,
    DRESSING,
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
    MILL_GLASS,
    MILL_SLATE,
    NORTH_LIGHT,
    PUDDLE,
    STILL_SLATE,
    STRAW,
    SUGAR_BEET,
    SUGAR_BLUE,
    SUGAR_BRICK,
    SUGAR_CREAM,
    SUGAR_GROUND,
    SUGAR_WHITE,
    _bed,
    _butcher_sign,
    _cold_door,
    _door,
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
    _heap,
    _meat_wagon,
    _mill_window,
    _north_lights,
    _on_roof,
    _pagoda,
    _pipe,
    _puddle,
    _ring,
    _roof_rows,
    _round_head,
    _smoke,
    _smoking_chimney,
    _spill,
    _stack,
    _stains,
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
    _track,
    _window,
    _works_siding,
)
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
    Painter,
    Point,
    Sprite,
    Top,
    chimney_top,
    shade,
)
from railroads.ui.pixel.art.parts import (
    BEAST,
    DOOR,
    METRE,
    _band,
    _bands,
    _bar,
    _barrel,
    _cow,
    _fence,
    _gable_across,
    _iso,
    _on_wall,
    _patch,
    _plinth,
    _ridge,
    _turned,
    _windows,
)

WINE_WALL: RGB = (238, 224, 192)
WINE_ROOF: RGB = (178, 86, 58)
WINE_STAIN: RGB = (110, 52, 76)
VINE: RGB = (78, 130, 50)
GRAPE: RGB = (112, 50, 124)
POST: RGB = (110, 84, 58)
DUG: RGB = (118, 96, 66)
CELLAR: RGB = (176, 168, 150)
TURF: RGB = (104, 140, 70)
SHUTTER: RGB = (90, 120, 70)


def _cask(p: Painter, x: float, y: float) -> None:
    """A cask lying on its side on (x, y), hooped."""
    p.ellipse(x - 2.5, y - 3, x + 2.5, y, BARREL)
    for dx in (-1, 1):
        p.line([(x + dx, y - 2.8), (x + dx, y - 0.2)], shade(BARREL, 0.55))
    p.point(x - 2, y - 2.2, shade(BARREL, 1.3))


def _casks(p: Painter, x: float, y: float, base: int) -> None:
    """A pyramid of casks lying on their sides, `base` along the bottom, its foot on (x, y)."""
    for row in range(base):
        for k in range(base - row):
            _cask(p, x + 5 * k + 2.5 * row, y - 2.6 * row)


def _keg(p: Painter, x: float, y: float) -> None:
    """A small upright keg standing on (x, y)."""
    p.ellipse(x - 1.4, y - 3.6, x + 1.4, y, BARREL)
    p.line([(x - 1.3, y - 1.2), (x + 1.3, y - 1.2)], shade(BARREL, 0.55))
    p.line([(x - 1.3, y - 2.5), (x + 1.3, y - 2.5)], shade(BARREL, 0.55))


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


def _vines(p: Painter, x: float, y: float, posts: int) -> None:
    """A row of vines trained along a trellis over a strip of dug earth, from (x, y) down along
    the ground, hung with grapes."""
    span = 3 * (posts - 1)
    p.polygon(
        [(x - 1, y), (x + span, y + span / 2), (x + span, y + span / 2 + 1.5), (x - 1, y + 1.5)],
        DUG,
    )
    ends = [(x + 3 * k, y + 1.5 * k) for k in range(posts)]
    for ex, ey in ends:
        p.line([(ex, ey), (ex, ey - 5)], POST)
    p.line([(ex, ey - 4) for ex, ey in ends], VINE)
    p.line([(ex, ey - 5) for ex, ey in ends], shade(VINE, 1.25))
    for ex, ey in ends[:-1]:
        p.point(ex + 1.5, ey - 2.5, GRAPE)
        p.point(ex + 1, ey - 3.5, shade(VINE, 0.8))


def _cellar(p: Painter, x: float, y: float, half: float, doors: tuple[float, ...]) -> None:
    """A wine cellar dug into a grassy bank on (x, y), its stone front opened by dark arched doors
    at each of `doors` along it."""
    top = p.box(x, y, half, 3, CELLAR, TURF)
    left, front, _, back = top
    for t in doors:
        _patch(p, left, front, (t - 0.12, t + 0.12), (1, 3), (42, 34, 30))
        p.line(
            [_on_wall(left, front, t - 0.14, 0.5), _on_wall(left, front, t + 0.14, 0.5)],
            shade(STONE, 1.3),
        )
    p.point(*_on_wall(left, back, 0.5, -0.5), shade(TURF, 1.25))


def _shutters(p: Painter, a: Point, b: Point, spans: tuple[float, ...], drop: float) -> None:
    """Windows with green shutters folded back either side, on the wall under the edge from `a`
    to `b`."""
    for t in spans:
        _window(p, a, b, (t - 0.025, t + 0.025), drop, tall=1.6)
        for side in (-0.05, 0.05):
            p.point(*_on_wall(a, b, t + side, drop + 0.2), SHUTTER)


def _round_tower(p: Painter, x: float, y: float, r: float, tall: float, cone: float) -> None:
    """A round cream tower standing on (x, y) under a pointed cone of red tiles, with windows up
    its lit side and a gold vane on top."""
    _turned(p, x, y, [(0, r), (tall, r)], WINE_WALL)
    for h in (tall * 0.3, tall * 0.65):
        p.line([(x - r * 0.4, y - h), (x - r * 0.4, y - h - 1.4)], WINDOW)
    p.ellipse(x - r - 0.8, y - tall - r / 2, x + r + 0.8, y - tall + r / 2 + 0.4, WINE_ROOF)
    apex = (x, y - tall - cone)
    p.polygon([(x - r - 0.8, y - tall), apex, (x + r / 3, y - tall + r / 2)], WINE_ROOF)
    p.polygon(
        [(x + r / 3, y - tall + r / 2), apex, (x + r + 0.8, y - tall)], shade(WINE_ROOF, 0.72)
    )
    p.line(
        [(x - r * 0.5, y - tall - 0.5), (x - 0.5, y - tall - cone + 1.5)], shade(WINE_ROOF, 1.25)
    )
    p.line([apex, (apex[0], apex[1] - 2)], (60, 56, 54))
    p.point(apex[0] + 0.5, apex[1] - 2.5, (230, 188, 70))


def _terrace(p: Painter, x: float, y: float, length: float, height: float) -> None:
    """A terrace of vines cut into a hillside: a stone retaining wall `height` high round a green
    strip on (x, y) planted with two rows of trellised vines hung with grapes."""
    top = p.hall(x, y, length, 6, height, CELLAR, shade(VINE, 0.8))
    left, front, right, back = top
    _band(p, left, front, 0.6, shade(CELLAR, 1.12))
    for t in (0.2, 0.5, 0.8):
        p.point(*_on_wall(left, front, t, height * 0.6), shade(CELLAR, 0.8))
    posts = round(length / 3)
    for depth in (0.3, 0.75):
        ends = []
        for k in range(posts + 1):
            a = _on_wall(left, front, k / posts, 0)
            b = _on_wall(back, right, k / posts, 0)
            ends.append((a[0] + (b[0] - a[0]) * depth, a[1] + (b[1] - a[1]) * depth))
        for ex, ey in ends:
            p.line([(ex, ey), (ex, ey - 3)], POST)
        p.line([(ex, ey - 2) for ex, ey in ends], VINE)
        p.line([(ex, ey - 3) for ex, ey in ends], shade(VINE, 1.3))
        for ex, ey in ends[:-1]:
            p.point(ex + 1.5, ey - 1.5, GRAPE)


def big_winery() -> Sprite:
    """A two-storey cream winery under red tiles with green-shuttered windows and an arched door,
    a square cream press tower behind it under a tiled pyramid, three rows of trellised vines
    before it, a cellar dug into a grassy bank with two dark doors, and a pyramid of casks on
    ground stained with spilt wine."""
    p = Painter(52, 45, (27, 29))
    foot = _footing(2, -3, 18, 8)
    _bed(p, foot)
    p.blob(4, 6, 14, 10, (112, 92, 70))
    p.shadow(17, 4.5)
    _track(p, foot[0], foot[1], (0.62, 0.76), 4, ruts=False)
    tower = p.box(12, -8, 4, 14, WINE_WALL, None)
    tl, tf, tr, _ = tower
    _plinth(p, tower, 14, WINE_WALL)
    _shutters(p, tl, tf, (0.5,), 2)
    _window(p, tf, tr, (0.4, 0.6), 2, shade(WINDOW, 0.85), 1.6)
    apex = (tf[0], (tl[1] + tr[1]) / 2 - 5)
    p.polygon([tl, tf, apex], WINE_ROOF, shade(WINE_ROOF, 0.6))
    p.polygon([tf, tr, apex], shade(WINE_ROOF, 0.72), shade(WINE_ROOF, 0.5))
    top = p.hall(2, -3, 18, 8, 9, WINE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 9, WINE_WALL)
    _stains(p, left, front, 9, (0.05, 0.3, 0.9), shade(WINE_WALL, 0.66), 1.4)
    _shutters(p, left, front, (0.12, 0.3, 0.48, 0.86), 1.6)
    _shutters(p, left, front, (0.12, 0.3, 0.48), 5.4)
    _door(p, left, front, (0.62, 0.76), 4.4, 9, (110, 74, 50), shade(WINE_WALL, 0.8))
    _window(p, front, right, (0.35, 0.65), 2, shade(WINDOW, 0.85), 1.6)
    p.gable(top, 4, WINE_ROOF, WINE_WALL)
    _roof_rows(p, top, 4, WINE_ROOF)
    x, y = _on_roof(top, 4, 0.2, 0.9)
    p.rect(x - 0.6, y - 3, x + 0.6, y, shade(WINE_WALL, 0.9))
    for x, y in ((-24, -2), (-21, 2.5), (-18, 7)):
        _vines(p, x, y, 4)
    _cellar(p, 17, 4, 5, (0.3, 0.7))
    _casks(p, -2, 13, 3)
    p.blob(10, 10.5, 13, 12, WINE_STAIN)
    _barrel(p, 12, 10)
    return p.sprite()


def mega_winery() -> Sprite:
    """A wine château: a long cream house of two storeys under red tiles with green-shuttered
    windows and a round tower at its corner under a pointed red cone, a hillside of three vine
    terraces walled in stone rising behind it, cellars dug into a grassy bank with three arched
    doors, a pyramid of casks and a cart of casks on ground stained with wine, and a row of vines
    before it all."""
    p = Painter(63, 56, (32, 37))
    for x, y, h in ((-6, -17, 10), (-11, -14.5, 7), (-16, -12, 4)):
        _terrace(p, x, y, 14, h)
    foot = _footing(6, -2, 22, 9)
    _bed(p, foot)
    p.blob(-2, 8, 10, 12, (112, 92, 70))
    p.shadow(22, 5)
    _track(p, foot[0], foot[1], (0.52, 0.64), 5, ruts=False)
    top = p.hall(6, -2, 22, 9, 11, WINE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 11, WINE_WALL)
    _stains(p, left, front, 11, (0.25, 0.45, 0.9), shade(WINE_WALL, 0.66), 1.4)
    _shutters(p, left, front, (0.28, 0.4, 0.74, 0.86), 2)
    _shutters(p, left, front, (0.28, 0.4, 0.58, 0.74, 0.86), 6.2)
    _patch(p, left, front, (0.52, 0.64), (5.6, 6.2), (230, 188, 70))
    _door(p, left, front, (0.53, 0.63), 7, 11, (110, 74, 50), shade(WINE_WALL, 0.8))
    _windows(p, front, right, (0.3, 0.7), (2, 6.2), 2, shade(WINDOW, 0.85))
    p.gable(top, 5, WINE_ROOF, WINE_WALL)
    _roof_rows(p, top, 5, WINE_ROOF, (0.25, 0.5, 0.75))
    for along in (0.3, 0.75):
        x, y = _on_roof(top, 5, along, 0.95)
        p.rect(x - 0.6, y - 3, x + 0.6, y, shade(WINE_WALL, 0.9))
    _round_tower(p, -9, -4, 3.6, 18, 8)
    _vines(p, -29, 2, 5)
    _vines(p, -26, 7, 4)
    _cellar(p, 22, 6, 6, (0.2, 0.5, 0.8))
    _casks(p, -14, 15, 3)
    p.blob(2, 13, 7, 15, WINE_STAIN)
    _cart(p, 8, 14, (BARREL, BARREL, BARREL))
    return p.sprite()


BREW_ROOF: RGB = (84, 76, 80)
HOP: RGB = (110, 170, 70)
GRAIN: RGB = (176, 150, 92)
COWL: RGB = (236, 232, 222)
STEAM: RGB = (236, 236, 238)


def _copper_dome(p: Painter, x: float, y: float, r: float) -> None:
    """A green-tinged copper dome on (x, y), banded, glinting and topped with a vent pipe."""
    rise = r * 1.35
    _turned(
        p,
        x,
        y,
        [
            (0, r),
            (rise * 0.33, r * 0.93),
            (rise * 0.66, r * 0.67),
            (rise * 0.9, r * 0.33),
            (rise, 0),
        ],
        COPPER,
    )
    _ring(p, x, y - rise * 0.25, r * 0.95, shade(COPPER, 0.7))
    p.blob(x - r * 0.65, y - rise * 0.8, x - r * 0.2, y - rise * 0.4, shade(COPPER, 1.3))
    p.line([(x, y - rise), (x, y - rise - 2)], (90, 80, 70))


def _kiln(p: Painter, x: float, y: float, half: float, tall: float) -> None:
    """A square brick malt kiln standing on (x, y) under a steep slate pyramid, a white cowl on
    its top letting out a wisp of steam."""
    top = p.box(x, y, half, tall, BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, tall, BRICK)
    _windows(p, left, front, (0.5,), (tall * 0.3,), 2, WINDOW, shade(BRICK, 1.35))
    apex = (front[0], (left[1] + right[1]) / 2 - half * 1.6)
    p.polygon([left, front, apex], BREW_ROOF, shade(BREW_ROOF, 0.6))
    p.polygon([front, right, apex], shade(BREW_ROOF, 0.72), shade(BREW_ROOF, 0.5))
    cx, cy = apex
    p.rect(cx - 1, cy - 3, cx + 1, cy + 0.5, COWL)
    p.polygon([(cx - 1.6, cy - 3), (cx, cy - 4.6), (cx + 2.2, cy - 3.4)], shade(COWL, 0.8))
    _smoke(p, cx + 2, cy - 6, STEAM, 2)


def _hop_sign(p: Painter, x: float, y: float, size: float) -> None:
    """A green hop cone hung on a wall at (x, y)."""
    p.polygon(
        [(x, y - 1.5 * size), (x + 1.4 * size, y), (x, y + 1.8 * size), (x - 1.4 * size, y)],
        HOP,
        shade(HOP, 0.6),
    )
    p.line([(x - 0.8 * size, y - 0.2), (x + 0.8 * size, y + 0.4)], shade(HOP, 0.75))
    p.point(x - 0.5 * size, y - 0.6 * size, shade(HOP, 1.3))


def _brewhouse(p: Painter, x: float, y: float, length: float, height: float) -> Top:
    """A brick brewhouse over a dark plinth: rows of arched windows, a door under a light lintel and
    a cream painted sign with a hop cone on it, under a dark slate gable."""
    lintel = shade(BRICK, 1.3)
    top = p.hall(x, y, length, 9, height, BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, height, BRICK)
    _stains(p, left, front, height, (0.04, 0.38, 0.62, 0.9), shade(BRICK, 0.62), 1.8)
    rows = tuple(2 + 3.4 * k for k in range(int((height - 3) // 3.4)))
    _windows(p, left, front, (0.45, 0.58, 0.72, 0.86), rows, 2, arch=lintel)
    _door(p, left, front, (0.62, 0.7), height - 3.4, height, DOORWAY, lintel)
    _patch(p, left, front, (0.04, 0.36), (1, 5.2), (222, 210, 180))
    _patch(p, left, front, (0.04, 0.36), (1, 1.5), shade(BRICK, 0.8))
    for drop in (3.6, 4.2):
        p.line(
            [_on_wall(left, front, 0.2, drop), _on_wall(left, front, 0.34, drop)], (150, 130, 96)
        )
    _hop_sign(p, *_on_wall(left, front, 0.1, 3), 1)
    _windows(p, front, right, (0.3, 0.7), rows, 2, shade(WINDOW, 0.8), shade(BRICK, 1.1))
    p.gable(top, 4.5, BREW_ROOF, BRICK)
    _roof_rows(p, top, 4.5, BREW_ROOF)
    return top


def _brew_tower(p: Painter, x: float, y: float, half: float, tall: float) -> Top:
    """A tall brick brew tower with arched windows up it and a hoist door near its top."""
    tower = shade(BRICK, 1.08)
    top = p.box(x, y, half, tall, tower, None)
    left, front, right, _ = top
    _plinth(p, top, tall, tower)
    _stains(p, front, right, tall, (0.1, 0.9), shade(tower, 0.5), 1.6)
    _bands(p, left, front, right, 0.8, shade(tower, 1.25), shade(tower, 1.0))
    drops = tuple(3 + 5 * k for k in range(int((tall - 6) // 5)))
    _windows(p, left, front, (0.5,), drops, 2, arch=shade(BRICK, 1.35))
    _windows(p, front, right, (0.5,), drops[:1], 2, shade(WINDOW, 0.8), shade(BRICK, 1.1))
    _door(p, front, right, (0.3, 0.7), tall - 4, tall, (70, 50, 40), shade(BRICK, 1.1))
    return top


def big_brewery() -> Sprite:
    """A big brewery: a long brick brewhouse with two rows of arched windows and a hop cone on its
    sign, a tall brick tower under a copper dome, a malt kiln behind under a slate pyramid with a
    white cowl steaming, a dray loaded with kegs, barrels by the door on a path wet with beer and a
    heap of spent grain."""
    p = Painter(48, 53, (24, 37))
    foot = _footing(-6, -1, 18, 9)
    tower_foot = _footing(8, 4, 5, 5)
    _bed(p, foot)
    _bed(p, tower_foot)
    p.shadow(18, 4.5)
    _track(p, foot[0], foot[1], (0.62, 0.7), 4, ruts=False)
    _track(p, tower_foot[1], tower_foot[2], (0.3, 0.7), 4)
    _kiln(p, 15, -7, 4, 11)
    top = _brewhouse(p, -6, -1, 18, 9)
    x, y = _on_roof(top, 4.5, 0.3, 0.85)
    cap = _stack(p, x - 1, y, y - 6, BRICK)
    _smoke(p, x + 0.5, cap - 2, STEAM, 2)
    tower = _brew_tower(p, 8, 4, 5, 19)
    tl, tf, tr, _ = tower
    _copper_dome(p, tf[0], (tl[1] + tr[1]) / 2, 5.2)
    _puddle(p, -8, 9, 1.8)
    for x, y in ((-12, 7), (-5, 9.5)):
        _barrel(p, x, y)
    _heap(p, 18, 6, 2.6, 2, GRAIN)
    _cart(p, 13, 11, (BARREL, BARREL, BARREL))
    return p.sprite()


def mega_brewery() -> Sprite:
    """A great brewery: a long three-storey brick brewhouse with rows of arched windows and a hop
    cone on its sign, a tall brew tower crowned with a big copper dome, two malt kilns behind under
    slate pyramids with white cowls steaming, a tall chimney smoking, casks stacked by the door, a
    dray loaded with kegs, a big hop cone hung from the tower and a heap of spent grain."""
    p = Painter(61, 80, (30, 59))
    foot = _footing(-6, 0, 24, 9)
    tower_foot = _footing(11, 6, 6, 6)
    _bed(p, foot)
    _bed(p, tower_foot)
    _spill(p, 22, 8, 2.8, GRAIN)
    p.shadow(24, 5.5)
    _track(p, foot[0], foot[1], (0.62, 0.7), 5, ruts=False)
    _track(p, tower_foot[1], tower_foot[2], (0.3, 0.7), 5)
    _smoking_chimney(p, 20, -6, -44, smoke=(200, 200, 204), puffs=4, close=True)
    for x, y in ((-18, -12), (-9, -15)):
        _kiln(p, x, y, 4.5, 14)
    top = _brewhouse(p, -6, 0, 24, 13)
    tower = _brew_tower(p, 11, 6, 6, 28)
    tl, tf, tr, _ = tower
    _copper_dome(p, tf[0], (tl[1] + tr[1]) / 2, 6.6)
    dx, dy = tf[0], (tl[1] + tr[1]) / 2 - 6.6 * 1.35
    p.rect(dx - 1.4, dy - 3, dx + 1.4, dy + 0.5, (222, 210, 180))
    p.line([(dx - 0.4, dy - 2.2), (dx - 0.4, dy - 0.6)], WINDOW)
    p.ellipse(dx - 2, dy - 4.4, dx + 2, dy - 2.4, COPPER)
    p.line([(dx, dy - 4.4), (dx, dy - 6.5)], (230, 188, 70))
    left, front, _, _ = top
    bx, by = _on_wall(left, front, 0.02, 2)
    p.line([(bx, by), (bx - 3.5, by)], (60, 56, 54))
    _hop_sign(p, bx - 3.5, by + 3.5, 1.6)
    _puddle(p, -9, 11, 2)
    _casks(p, -24, 9, 3)
    _heap(p, 22, 8, 2.8, 2.2, GRAIN)
    _cart(p, 3, 15, (BARREL, BARREL, BARREL))
    _cart(p, 14, 16, (BARREL, BARREL))
    return p.sprite()


PEAT: RGB = (112, 84, 60)
DAMP: RGB = (204, 204, 194)
TAR: RGB = (44, 46, 52)


def _still_house(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> Top:
    """A whitewashed house on a stone plinth, its foot greyed by the damp, with dark windows and
    a door under a slate gable."""
    top = p.hall(x, y, length, depth, height, WHITEWASH, None)
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (height - 1, height), STONE)
    _patch(p, front, right, (0, 1), (height - 1, height), shade(STONE, 0.78))
    spans = tuple((k + 0.5) / 8 for k in range(8))
    for t in spans:
        p.point(*_on_wall(left, front, t, height - 0.6), shade(STONE, 0.75))
    _stains(p, left, front, height - 1, spans[::2], DAMP, 1.4)
    _stains(p, front, right, height - 1, (0.2, 0.75), shade(WHITEWASH, 0.66), 1.4)
    return top


def _pot_still(p: Painter, x: float, y: float, size: float, worm: float) -> None:
    """A copper pot still standing on (x, y), its swan neck bent over into a lyne arm running
    `worm` along to a wooden worm tub."""
    s = size
    _turned(
        p,
        x,
        y,
        [(0, 3 * s), (2 * s, 4 * s), (4 * s, 3.5 * s), (6 * s, 1.5 * s), (8 * s, 1 * s)],
        COPPER,
    )
    p.blob(x - 3 * s, y - 6 * s, x - 1 * s, y - 3 * s, shade(COPPER, 1.3))
    _ring(p, x, y - 3 * s, 3.6 * s, shade(COPPER, 0.7))
    p.line([(x, y - 8 * s), (x, y - 9.5 * s), (x + worm, y - 7 * s)], shade(COPPER, 0.8))
    tx, ty = x + worm + 1, y + 1
    p.line([(x + worm, y - 7 * s), (x + worm, ty - 4)], shade(COPPER, 0.8))
    p.ellipse(tx - 2.2, ty - 4, tx + 2.2, ty, BARREL)
    p.line([(tx - 2, ty - 2), (tx + 2, ty - 2)], shade(BARREL, 0.55))
    p.blob(x - 4 * s, y - 0.5, x + 4 * s, y + 1.5, (98, 126, 146))


def _warehouse(p: Painter, x: float, y: float, length: float, height: float) -> None:
    """A long whitewashed bonded warehouse under a tarred roof, its small windows barred and a
    black band of lettering along its wall."""
    top = _still_house(p, x, y, length, 6, height)
    left, front, right, _ = top
    _patch(p, left, front, (0.15, 0.85), (0.8, 2), TAR)
    for k in range(7):
        t = 0.2 + k * 0.1
        p.point(*_on_wall(left, front, t, 1.2), shade(WHITEWASH, 0.9))
    _windows(p, left, front, (0.1, 0.3, 0.5, 0.7, 0.9), (3.2,), 1, TAR)
    _door(p, front, right, (0.3, 0.7), height - 3.4, height - 1, TAR)
    p.gable(top, 3, TAR, WHITEWASH)
    _roof_rows(p, top, 3, TAR, (0.5,))


def _thin_stack(p: Painter, x: float, foot: float, top: float) -> None:
    """A tall thin brick stack banded in white, smoking, lowered (chimney_top)."""
    top = chimney_top(foot, top)
    colour = (150, 80, 60)
    p.rect(x, top, x + 1.6, foot, colour)
    for f in (0.06, 0.4):
        y = top + (foot - top) * f
        p.line([(x, y), (x + 1.6, y)], (230, 228, 222))
    p.rect(x - 0.5, top - 1, x + 2.1, top, (90, 52, 42))
    _smoke(p, x + 1, top - 3, (190, 190, 194), 3)


def big_distillery() -> Sprite:
    """A big distillery: a long whitewashed still house on a stone plinth greyed by the damp, two
    pagoda-roofed kiln vents on its slate roof, a tall thin banded stack smoking, a whitewashed
    bonded warehouse behind under a tarred roof, two copper pot stills at its side running to
    their worm tubs over wet ground, casks stacked out front and a heap of peat."""
    p = Painter(51, 70, (24, 55))
    foot = _footing(-6, -1, 18, 8)
    _bed(p, foot)
    _spill(p, -18, 6, 2.6, PEAT)
    p.shadow(19, 4.5)
    _track(p, foot[0], foot[1], (0.48, 0.62), 4)
    _thin_stack(p, 4, -6, -42)
    _warehouse(p, 14, -6, 14, 7)
    top = _still_house(p, -6, -1, 18, 8, 8)
    left, front, right, _ = top
    _windows(p, left, front, (0.1, 0.25, 0.4, 0.72, 0.87), (2,), 2, TAR)
    _windows(p, left, front, (0.1, 0.25, 0.4, 0.72, 0.87), (5.4,), 1, TAR)
    _door(p, left, front, (0.5, 0.62), 4, 8, (60, 48, 42), shade(WHITEWASH, 0.8))
    _windows(p, front, right, (0.3, 0.7), (2,), 2, TAR)
    p.gable(top, 4.5, STILL_SLATE, WHITEWASH)
    _roof_rows(p, top, 4.5, STILL_SLATE, (0.5,))
    for along in (0.25, 0.6):
        _pagoda(p, *_on_roof(top, 4.5, along, 1), 1.1)
    _pot_still(p, 11, 7, 1, 4)
    _pot_still(p, 18, 4, 0.8, 3)
    _heap(p, -18, 6, 2.6, 2.2, PEAT)
    _casks(p, -9, 12, 2)
    _barrel(p, 3, 11)
    return p.sprite()


def mega_distillery() -> Sprite:
    """A great distillery: a tall whitewashed still house whose big windows show the copper stills
    gleaming inside, a malt house beside it with three pagoda-roofed kiln vents, two long bonded
    warehouses under tarred roofs behind, a tall banded stack smoking, a big copper pot still
    outside running to its worm tub, a pyramid of casks, a cart of casks and stacks of peat."""
    p = Painter(64, 81, (31, 59))
    _spill(p, -24, 8, 3, PEAT)
    _warehouse(p, -10, -16, 20, 7)
    _warehouse(p, 16, -10, 18, 7)
    _thin_stack(p, 8, -10, -46)
    malt_foot = _footing(-14, -2, 14, 8)
    foot = _footing(7, 3, 16, 9)
    _bed(p, malt_foot)
    _bed(p, foot)
    p.shadow(24, 5.5)
    _track(p, malt_foot[0], malt_foot[1], (0.55, 0.7), 4)
    _track(p, foot[0], foot[1], (0.42, 0.58), 5)
    malt = _still_house(p, -14, -2, 14, 8, 7)
    ml, mf, mr, _ = malt
    _windows(p, ml, mf, (0.15, 0.32, 0.8), (2,), 2, TAR)
    _door(p, ml, mf, (0.55, 0.7), 3, 7, (60, 48, 42), shade(WHITEWASH, 0.8))
    _windows(p, mf, mr, (0.5,), (2,), 2, TAR)
    p.gable(malt, 4, STILL_SLATE, WHITEWASH)
    _roof_rows(p, malt, 4, STILL_SLATE, (0.5,))
    for along in (0.15, 0.48, 0.81):
        _pagoda(p, *_on_roof(malt, 4, along, 1), 1.1)
    top = _still_house(p, 7, 3, 16, 9, 12)
    left, front, right, _ = top
    for t in (0.22, 0.52):
        _patch(p, left, front, (t - 0.11, t + 0.11), (2, 8), (54, 50, 52))
        sx, sy = _on_wall(left, front, t, 8)
        p.ellipse(sx - 2, sy - 4, sx + 2, sy, COPPER)
        p.line([(sx, sy - 4), (sx, sy - 5.5)], COPPER)
        p.point(sx - 1, sy - 3, shade(COPPER, 1.35))
        p.line([_on_wall(left, front, t - 0.11, 1.4), _on_wall(left, front, t + 0.11, 1.4)], STONE)
    _door(p, left, front, (0.74, 0.86), 7, 12, (60, 48, 42), shade(WHITEWASH, 0.8))
    _windows(p, front, right, (0.25, 0.5, 0.75), (2, 6.5), 2, TAR)
    p.gable(top, 5, STILL_SLATE, WHITEWASH)
    _roof_rows(p, top, 5, STILL_SLATE, (0.5,))
    _pagoda(p, *_on_roof(top, 5, 0.5, 1), 1.3)
    _pot_still(p, 23, 6, 1.2, 4)
    _heap(p, -24, 8, 3, 2.6, PEAT)
    _heap(p, -19, 10, 2.2, 1.8, PEAT)
    _casks(p, -12, 15, 3)
    _cart(p, 8, 17, (BARREL, BARREL, BARREL))
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


def _louvre(p: Painter, top: Top, rise: float, span: tuple[float, float]) -> None:
    """A long louvred vent along a gable's ridge from `span[0]` to `span[1]` of the way along it."""
    ridge_back, ridge_front = _ridge(top, rise)
    a, b = (
        _on_wall(ridge_back, ridge_front, span[0], 0),
        _on_wall(ridge_back, ridge_front, span[1], 0),
    )
    p.polygon(
        [(a[0], a[1] - 1.8), (b[0], b[1] - 1.8), b, a], (176, 168, 160), shade(MEAT_ROOF, 0.8)
    )
    p.line([(a[0] - 0.5, a[1] - 2.3), (b[0] + 0.5, b[1] - 2.3)], shade(MEAT_ROOF, 0.8))
    steps = max(2, round((b[0] - a[0]) / 2))
    for k in range(steps + 1):
        x, y = _on_wall(a, b, k / steps, 0)
        p.point(x, y - 1, shade(MEAT_ROOF, 0.9))


def _ham(p: Painter, x: float, y: float, size: float = 1) -> None:
    """A ham hung from a bracket at (x, y) as a sign."""
    s = size
    p.line([(x, y), (x - 3 * s, y + 1)], (60, 56, 54))
    hx, hy = x - 3 * s, y + 1 + 3 * s
    p.ellipse(hx - 3 * s, hy - 2 * s, hx + 1 * s, hy + 2 * s, HAM)
    p.point(hx - 2 * s, hy - 1 * s, shade(HAM, 1.3))
    p.line([(hx + 1 * s, hy - 1 * s), (hx + 2 * s, hy - 2 * s)], BONE)
    p.point(hx + 2.6 * s, hy - 2.6 * s, BONE)


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


SAUSAGE_WALL: RGB = (226, 210, 180)
SAUSAGE_ROOF: RGB = (150, 70, 52)
FRAME: RGB = (96, 64, 44)
SAUSAGE: RGB = (170, 74, 54)
SIGN_BOARD: RGB = (130, 80, 50)
SMOKEHOUSE: RGB = (134, 69, 48)
WOOD_SMOKE: RGB = (176, 170, 164)


def _half_timbered(p: Painter, top: Top, height: float, bays: int) -> None:
    """Dark timber framing on a cream wall: posts between `bays` bays, a rail at each floor and
    a brace across every other bay."""
    left, front, right, _ = top
    floors = max(1, round(height / 5))
    for k in range(bays + 1):
        t = k / bays
        p.line([_on_wall(left, front, t, 0), _on_wall(left, front, t, height)], FRAME)
    for f in range(floors + 1):
        d = height * f / floors
        _bands(p, left, front, right, d, FRAME, shade(FRAME, 0.8))
    for k in range(0, bays, 2):
        t0, t1 = k / bays, (k + 1) / bays
        for f in range(floors):
            d0, d1 = height * f / floors, height * (f + 1) / floors
            p.line([_on_wall(left, front, t0, d1), _on_wall(left, front, t1, d0)], FRAME)
    for t in (0.5, 1):
        p.line([_on_wall(front, right, t, 0), _on_wall(front, right, t, height)], shade(FRAME, 0.8))


def _garland(p: Painter, a: Point, b: Point, links: int) -> None:
    """A string of sausages hung in loops between `a` and `b`."""
    for k in range(links):
        x0, y0 = _on_wall(a, b, k / links, 0)
        x1, y1 = _on_wall(a, b, (k + 1) / links, 0)
        mid = ((x0 + x1) / 2, (y0 + y1) / 2 + 1.6)
        p.line([(x0, y0), mid, (x1, y1)], SAUSAGE)
        p.point(mid[0], mid[1], shade(SAUSAGE, 1.3))


def _rack(p: Painter, x: float, y: float, length: float, tall: float) -> None:
    """A drying rack of two posts and a pole running down along the ground from (x, y), hung
    with sausages."""
    end = (x + length, y + length / 2)
    for px, py in ((x, y), end):
        p.line([(px, py), (px, py - tall)], FRAME)
    a, b = (x, y - tall), (end[0], end[1] - tall)
    p.line([a, b], shade(FRAME, 1.3))
    steps = round(length / 1.6)
    for k in range(1, steps):
        sx, sy = _on_wall(a, b, k / steps, 0)
        p.line([(sx, sy), (sx, sy + 3.4)], SAUSAGE if k % 2 else (200, 120, 84))


def _smokehouse(p: Painter, x: float, y: float, half: float, tall: float) -> None:
    """A square brick smokehouse standing on (x, y) with a small iron door, under a little slate
    cap smoking with wood smoke."""
    top = p.box(x, y, half, tall, SMOKEHOUSE, None)
    left, front, right, _ = top
    _plinth(p, top, tall, SMOKEHOUSE)
    _stains(p, left, front, tall, (0.2, 0.8), shade(SMOKEHOUSE, 0.65), 2)
    _patch(p, front, right, (0.3, 0.7), (tall - 3.4, tall - 0.4), (54, 50, 52))
    _bands(p, left, front, right, 1, (54, 48, 46), (44, 40, 40))
    cx, cy = front[0], (left[1] + right[1]) / 2
    p.rect(cx - half * 0.5, cy - 2.4, cx + half * 0.5, cy, shade(SMOKEHOUSE, 0.8))
    p.polygon(
        [(cx - half * 0.8, cy - 2.4), (cx, cy - 4), (cx + half * 0.8, cy - 2.4)], (88, 84, 92)
    )
    _smoke(p, cx + 0.5, cy - 6.5, WOOD_SMOKE, 3)


def _sausage_house(p: Painter, x: float, y: float, length: float, height: float) -> Top:
    """A half-timbered butcher's house: a lit shop window of sausages under a brown sign board, a
    door with sausages hung over it, and a red tiled gable."""
    top = p.hall(x, y, length, 8, height, SAUSAGE_WALL, None)
    left, front, right, _ = top
    _half_timbered(p, top, height, round(length / 3))
    _patch(p, left, front, (0.08, 0.5), (height - 4.6, height - 3.8), SIGN_BOARD)
    _patch(p, left, front, (0.1, 0.48), (height - 3.4, height - 1), (244, 210, 136))
    for t in (0.16, 0.26, 0.36):
        p.line(
            [_on_wall(left, front, t, height - 3.2), _on_wall(left, front, t, height - 1.6)],
            SAUSAGE,
        )
    _door(p, left, front, (0.6, 0.74), height - 4, height, (104, 66, 44))
    _garland(p, _on_wall(left, front, 0.56, height - 5), _on_wall(left, front, 0.78, height - 5), 3)
    _plinth(p, top, height, SAUSAGE_WALL)
    _window(p, front, right, (0.2, 0.4), 2, shade(WINDOW, 0.85))
    p.gable(top, 4.5, SAUSAGE_ROOF, SAUSAGE_WALL)
    _roof_rows(p, top, 4.5, SAUSAGE_ROOF)
    return top


def big_sausage_maker() -> Sprite:
    """A big butcher's: a two-storey half-timbered house under red tiles with a lit shop window of
    sausages under its sign and sausages strung over its door, a square brick smokehouse beside it
    smoking with wood smoke, and a rack of sausages drying outside."""
    p = Painter(43, 48, (21, 34))
    foot = _footing(-3, 0, 14, 8)
    _bed(p, foot)
    _bed(p, _footing(12, -3, 4, 4), margin=1)
    p.shadow(16, 4)
    _track(p, foot[0], foot[1], (0.6, 0.74), 3.5, ruts=False)
    _smokehouse(p, 12, -3, 4, 14)
    top = _sausage_house(p, -3, 0, 14, 10)
    x, y = _on_roof(top, 4.5, 0.25, 0.9)
    p.rect(x - 1, y - 3, x + 1, y + 0.5, BRICK)
    _rack(p, 4, 6, 10, 6)
    _ham(p, *_on_wall(top[0], top[1], 0.02, 2), 0.8)
    return p.sprite()


def mega_sausage_maker() -> Sprite:
    """A great sausage works: a long half-timbered hall of two storeys under red tiles with shop
    windows of sausages, a giant sausage on its ridge for a sign, two tall brick smokehouses behind
    smoking with wood smoke, racks of sausages drying in the yard, a ham hung out by the door and
    a cart of crates."""
    p = Painter(56, 58, (28, 42))
    foot = _footing(-5, 0, 22, 9)
    _bed(p, foot)
    for x, y in ((10, -8), (16, -5)):
        _bed(p, _footing(x, y, 4.5, 4.5), margin=1)
    p.shadow(22, 5)
    _track(p, foot[0], foot[1], (0.6, 0.74), 5, ruts=False)
    for x, y in ((10, -8), (16, -5)):
        _smokehouse(p, x, y, 4.2, 17)
    top = _sausage_house(p, -5, 0, 22, 12)
    left, front, _, _ = top
    for along in (0.15, 0.85):
        x, y = _on_roof(top, 4.5, along, 0.9)
        p.rect(x - 1, y - 3, x + 1, y + 0.5, BRICK)
    rb, rf = _ridge(top, 4.5)
    links = []
    for k in range(6):
        t = 0.32 + 0.36 * k / 5
        sx, sy = _on_wall(rb, rf, t, -3.5)
        links.append((sx, sy - 1.6 * (1 - (2 * k / 5 - 1) ** 2)))
    for x, y in (links[1], links[4]):
        p.line([(x, y), (x, y + 3.5)], (60, 56, 54))
    for x, y in links:
        p.blob(x - 2.2, y - 2, x + 2.2, y + 2, (90, 40, 34))
    for x, y in links:
        p.blob(x - 1.8, y - 1.6, x + 1.8, y + 1.4, (212, 112, 82))
    for x, y in links[1:-1]:
        p.point(x - 0.6, y - 1.2, (240, 170, 130))
    for x, y in (links[0], links[-1]):
        p.point(x + (-2 if x == links[0][0] else 1.8), y, (60, 56, 54))
    _ham(p, *_on_wall(left, front, 0.02, 2), 1)
    _rack(p, 4, 7, 10, 6)
    _rack(p, 10, 3, 10, 6)
    _cart(p, -14, 11, ((150, 112, 74), (150, 112, 74)))
    return p.sprite()


COOPER_ROOF: RGB = (118, 84, 54)
STAVES: RGB = (206, 168, 116)
SHAVINGS: RGB = (226, 200, 150)
BLOCK: RGB = (128, 94, 62)
FLAME: RGB = (246, 140, 40)
FLAME_HOT: RGB = (255, 220, 110)
LOG: RGB = (120, 86, 56)


def _cooper_shed(p: Painter, x: float, y: float, length: float, height: float) -> Top:
    """A timber cooper's shed over a dark plinth with planked walls, windows and a wide double
    door under a brown gable."""
    top = p.hall(x, y, length, 8, height, TIMBER, None)
    left, front, right, _ = top
    for drop in range(2, int(height), 2):
        _band(p, front, right, drop, shade(TIMBER, 0.66))
        _band(p, left, front, drop, shade(TIMBER, 0.85))
    _plinth(p, top, height, TIMBER)
    _stains(p, left, front, height, (0.05, 0.3, 0.95), shade(TIMBER, 0.6), 1.2)
    for t in (0.1, 0.24):
        _window(p, left, front, (t - 0.05, t + 0.05), 1.6)
    _door(p, left, front, (0.42, 0.72), 2, height, DOORWAY, shade(TIMBER, 1.3))
    p.line(
        [_on_wall(left, front, 0.57, 2), _on_wall(left, front, 0.57, height)], shade(TIMBER, 0.55)
    )
    _windows(p, front, right, (0.3, 0.6), (1.6,), 2, shade(WINDOW, 0.8))
    p.gable(top, 4, COOPER_ROOF, TIMBER)
    _roof_rows(p, top, 4, COOPER_ROOF)
    return top


def _firing(p: Painter, x: float, y: float) -> None:
    """A barrel being fired: an open hooped barrel standing over a ring of glowing coals with
    flames licking up inside it and sparks rising."""
    p.blob(x - 3, y - 1, x + 3, y + 1.4, (60, 46, 40))
    p.blob(x - 2.4, y - 0.6, x + 2.4, y + 1, FLAME)
    p.ellipse(x - 2, y - 6, x + 2, y, shade(BARREL, 0.85))
    for h in (1.6, 4):
        p.line([(x - 2, y - h), (x + 2, y - h)], (60, 56, 54))
    p.ellipse(x - 1.8, y - 6.8, x + 1.8, y - 5.2, (40, 30, 26))
    p.polygon(
        [(x - 1.2, y - 6), (x - 0.4, y - 9), (x + 0.4, y - 7), (x + 1, y - 9.5), (x + 1.4, y - 6)],
        FLAME,
    )
    p.point(x, y - 7, FLAME_HOT)
    for dx, dy in ((-1, -11), (1.5, -12), (0, -13.5)):
        p.point(x + dx, y + dy, FLAME_HOT if dy % 2 else FLAME)


def _staves(p: Painter, x: float, y: float, rows: int) -> None:
    """A stacked pile of pale oak staves on (x, y), their ends banded."""
    pile = p.hall(x, y, 5, 2.4, 1.2 * rows, STAVES, shade(STAVES, 1.12))
    for r in range(1, rows):
        _band(p, pile[0], pile[1], 1.2 * r, shade(STAVES, 0.7))
    _band(p, pile[1], pile[2], 0.6, shade(STAVES, 0.6))


def _logs(p: Painter, x: float, y: float, count: int) -> None:
    """Oak logs piled on (x, y), their pale cut ends showing."""
    for k in range(count):
        row = 0 if k < count - count // 3 else 1
        lx, ly = x + 2.2 * k - (count - count // 3) * 2.2 * row + 1.1 * row, y - 1.8 * row
        p.ellipse(lx - 1.2, ly - 2.2, lx + 1.2, ly, LOG)
        p.point(lx, ly - 1.2, SHAVINGS)


def _shavings(p: Painter, points: tuple[Point, ...]) -> None:
    for x, y in points:
        p.point(x, y, SHAVINGS)


def big_barrel_maker() -> Sprite:
    """A big cooperage: a long planked timber shed under a brown roof with a wide double door and
    two stove pipes smoking, a barrel being fired over glowing coals outside, a pyramid of
    finished casks, piles of staves and oak logs, a chopping block and shavings scattered on the
    footpath."""
    p = Painter(47, 40, (23, 26))
    foot = _footing(-5, -1, 16, 8)
    _bed(p, foot)
    p.shadow(18, 4)
    _track(p, foot[0], foot[1], (0.42, 0.72), 4, ruts=False)
    top = _cooper_shed(p, -5, -1, 16, 8)
    for along in (0.25, 0.7):
        x, y = _on_roof(top, 4, along, 0.8)
        _pipe(p, x, y, 3)
        _smoke(p, x + 0.5, y - 5, puffs=2)
    _shavings(p, ((-8, 7.5), (-5, 8.4), (-10, 6.6), (-3, 7.6), (-7, 9.4)))
    _logs(p, 13, -2, 4)
    _firing(p, 14, 6)
    _staves(p, -17, 3, 2)
    p.ellipse(-4, 8, -1, 9.6, BLOCK)
    p.line([(-2.5, 8.6), (-1.5, 6.8)], (90, 88, 92))
    _casks(p, 2, 11, 3)
    return p.sprite()


def mega_barrel_maker() -> Sprite:
    """A great cooperage: a long two-storey planked timber hall under a brown roof with a louvred
    lantern along its ridge and a brick chimney smoking, an open-sided shed beside it on posts
    stacked with barrels, three barrels being fired over glowing coals in a row, a big pyramid of
    finished casks, piles of staves and oak logs and a cart loaded with barrels."""
    p = Painter(61, 59, (30, 42))
    foot = _footing(-7, -2, 22, 9)
    _bed(p, foot)
    p.shadow(24, 4.5)
    _track(p, foot[0], foot[1], (0.42, 0.72), 5, ruts=False)
    _logs(p, 18, -9, 6)
    shed_foot = _footing(16, 0, 10, 7)
    _bed(p, shed_foot, margin=0.8)
    sl, sf, sr, sb = shed_foot
    for k in range(3):
        _keg(p, sb[0] - 1 + 2.6 * k, sb[1] + 2.2 + 1.3 * k)
    for cx, cy in ((sl[0] + 3, sl[1] + 0.5), (sl[0] + 6, sl[1] + 2)):
        _barrel(p, cx, cy)
    shed_top = tuple((x, y - 7) for x, y in shed_foot)
    for x, y in (sl, sf, sr):
        p.line([(x, y), (x, y - 7)], shade(TIMBER, 0.7))
    p.gable(shed_top, 3, COOPER_ROOF, TIMBER)
    _roof_rows(p, shed_top, 3, COOPER_ROOF, (0.5,))
    top = _cooper_shed(p, -7, -2, 22, 11)
    left, front, _, _ = top
    _windows(p, left, front, (0.1, 0.24, 0.84, 0.94), (6,), 2, shade(WINDOW, 1.1))
    _louvre(p, top, 4, (0.3, 0.75))
    _smoking_chimney(p, -18, -17, -30, close=True)
    _shavings(p, ((-9, 8.5), (-6, 9.4), (-12, 7.6), (-4, 8.6), (-8, 10.4), (-1, 10)))
    for x, y in ((8, 10), (13, 7.5), (18, 5)):
        _firing(p, x, y)
    _staves(p, -24, 2, 3)
    _staves(p, -20, 5, 2)
    _casks(p, -16, 14, 4)
    _cart(p, 22, 12, (BARREL, BARREL, BARREL))
    return p.sprite()
