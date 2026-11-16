"""Big and mega wineries, breweries, distilleries, canneries, sugar refineries, meat works,
sausage makers and cooperages, each drawn as a building of its own: a cream château among
terraced vines, a brick brewhouse under a copper dome, a whitewashed still house among bonded
warehouses, a cannery with silos and a rail siding, a sugar-white refinery towering over its beet,
a brick slaughterhouse with a white cold store, a half-timbered sausage works with smokehouses and
a timber cooperage with barrels firing over the coals."""

from railroads.ui.pixel.art.industry import (
    GLAZING,
    STILL_SLATE,
    _bed,
    _chimney,
    _crate,
    _door,
    _footing,
    _heap,
    _on_roof,
    _pagoda,
    _pipe,
    _puddle,
    _ring,
    _roof_rows,
    _smoke,
    _spill,
    _stack,
    _stains,
    _track,
    _window,
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
    shade,
)
from railroads.ui.pixel.art.parts import (
    _band,
    _bands,
    _barrel,
    _cow,
    _fence,
    _on_wall,
    _paddock,
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
    _stack(p, x - 1, y, y - 6, BRICK)
    _smoke(p, x + 0.5, y - 8, STEAM, 2)
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
    _chimney(p, 20, -6, -44, BRICK)
    _smoke(p, 20.5, -46, (200, 200, 204), 4)
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
    """A tall thin brick stack banded in white, smoking."""
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


CANNERY_WALL: RGB = (222, 214, 192)
CANNERY_ROOF: RGB = (86, 128, 84)
LABEL: RGB = (196, 58, 48)
LABEL_TEXT: RGB = (240, 230, 210)
SILO: RGB = (165, 167, 176)
TIN: RGB = (206, 210, 218)
BALLAST: RGB = (138, 130, 120)
RAIL: RGB = (90, 88, 92)
SLEEPER: RGB = (96, 70, 50)


def _silos(p: Painter, feet: tuple[Point, ...], tall: float, r: float = 2.2) -> None:
    """A row of tall round banded steel silos standing on `feet`, joined across their tops by a
    railed catwalk."""
    for x, y in feet:
        _turned(p, x, y, [(0, r), (tall, r), (tall + 1.5, r * 0.5)], SILO, cap=shade(METAL, 1.25))
        for h in (tall * 0.33, tall * 0.66):
            _ring(p, x, y - h, r, shade(SILO, 0.7))
        p.line([(x - r * 0.5, y - 1), (x - r * 0.5, y - tall + 1)], shade(SILO, 1.15))
    (x0, y0), (x1, y1) = feet[0], feet[-1]
    p.line([(x0, y0 - tall - 1.5), (x1, y1 - tall - 1.5)], shade(METAL, 0.6))
    p.line([(x0, y0 - tall - 2.6), (x1, y1 - tall - 2.6)], shade(METAL, 0.75))


def _cannery(p: Painter, x: float, y: float, length: float, height: float) -> Top:
    """A pale cannery hall under a flat green roof behind a light parapet: rows of windows, a long
    red label with a white stripe for a sign, a loading door under a lintel and a row of
    glazed roof lights."""
    top = p.hall(x, y, length, 9, height, CANNERY_WALL, CANNERY_ROOF)
    left, front, right, back = top
    _plinth(p, top, height, CANNERY_WALL)
    _stains(p, left, front, height, (0.04, 0.24, 0.46, 0.7), shade(CANNERY_WALL, 0.62))
    _stains(p, front, right, height, (0.6, 0.9), shade(CANNERY_WALL, 0.5))
    p.line([left, front, right], shade(CANNERY_ROOF, 1.3))
    _patch(p, left, front, (0.05, 0.6), (0.8, 2.4), LABEL)
    p.line([_on_wall(left, front, 0.1, 1.6), _on_wall(left, front, 0.55, 1.6)], LABEL_TEXT)
    rows = tuple(3.4 + 3.6 * k for k in range(int((height - 4) // 3.6)))
    spans = tuple(0.06 + 0.08 * k for k in range(7))
    _windows(p, left, front, spans, rows, 2, WINDOW, shade(CANNERY_WALL, 1.1))
    _door(p, left, front, (0.66, 0.86), height - 4.4, height, DOORWAY, shade(CANNERY_WALL, 1.12))
    _windows(p, front, right, (0.25, 0.5, 0.75), rows, 2, shade(WINDOW, 0.8))
    for k in range(3):
        a = _on_wall(left, back, 0.5, 0)
        t = (k + 0.5) / 3
        lx, ly = _on_wall(left, front, t, 0)
        p.hall(
            lx + (a[0] - left[0]),
            ly + (a[1] - left[1]) + 0.5,
            length / 4,
            2,
            0.8,
            shade(CANNERY_ROOF, 0.8),
            GLAZING,
        )
    return top


def _tin_sign(p: Painter, x: float, y: float, r: float) -> None:
    """A giant tin can with a red label stood on legs on a roof at (x, y) as a sign."""
    for dx in (-r * 0.6, r * 0.6):
        p.line([(x + dx, y), (x + dx, y - 2)], (70, 66, 64))
    _turned(p, x, y - 2, [(0, r), (r * 1.8, r)], LABEL, cap=TIN)
    _ring(p, x, y - 2.2, r, TIN)
    _ring(p, x, y - 2 - r * 0.9, r, LABEL_TEXT)
    p.line([(x - r * 0.6, y - 3), (x - r * 0.6, y - 1.6 - r * 1.6)], shade(LABEL, 1.3))


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


def big_food_plant() -> Sprite:
    """A big cannery: a long two-storey pale hall under a flat green roof with glazed roof lights,
    a long red label for its sign, rows of windows and a loading door with cart ruts, three tall
    banded silos behind joined by a catwalk with a chute to the roof, crates of tins stacked out
    front and a cart of crates."""
    p = Painter(51, 47, (24, 32))
    foot = _footing(-6, 0, 18, 9)
    _bed(p, foot)
    p.blob(4, -5, 24, 3, (78, 74, 52))
    p.shadow(19, 4.5)
    _track(p, foot[0], foot[1], (0.66, 0.86), 4)
    _silos(p, ((9, -5), (13.5, -3), (18, -1)), 22)
    p.line([(9, -27), (4, -16), (2, -12)], shade(METAL, 0.7))
    _cannery(p, -6, 0, 18, 9)
    _puddle(p, 15, 4, 1.6)
    for x, y in ((-17, 3), (-14, 4.6), (-15.5, 1.8)):
        _crate(p, x, y)
    _crate(p, -15.5, 2.6)
    _cart(p, 6, 10, ((150, 112, 74), (150, 112, 74)))
    return p.sprite()


def mega_food_plant() -> Sprite:
    """A great cannery: a long three-storey pale hall under a flat green roof with roof lights and a
    giant red-labelled tin can for a sign on it, four tall banded silos behind joined by a catwalk
    with a chute running down to the roof, a rail siding along its side with a goods wagon at the
    loading door, and stacks of crates of tins."""
    p = Painter(64, 58, (30, 40))
    foot = _footing(-8, -2, 24, 10)
    _bed(p, foot)
    p.blob(8, -9, 29, 1, (78, 74, 52))
    p.shadow(24, 5)
    _track(p, foot[0], foot[1], (0.7, 0.88), 5)
    _silos(p, ((12, -9), (16.5, -7), (21, -5), (25.5, -3)), 26, 2.4)
    p.line([(12, -36), (6, -24), (2, -16)], shade(METAL, 0.7))
    p.line([(12.6, -36), (6.6, -24), (2.6, -16)], shade(METAL, 0.5))
    top = _cannery(p, -8, -2, 24, 12)
    _, front, _, back = top
    cx, cy = (front[0] + back[0]) / 2 - 2, (front[1] + back[1]) / 2
    _tin_sign(p, cx, cy, 3.2)
    _rails(p, (3, 12), (29, -1))
    _wagon(p, 14, 6, 11, (150, 66, 48), (110, 104, 100))
    for x, y in ((-24, 3), (-21, 4.6), (-22.5, 1.8), (-18, 6.2)):
        _crate(p, x, y)
    _crate(p, -22.5, 2.6)
    _crate(p, -21, 1)
    _cart(p, -6, 13, ((150, 112, 74), (150, 112, 74)))
    return p.sprite()


SUGAR_WALL: RGB = (232, 226, 210)
SUGAR_ROOF: RGB = (92, 86, 94)
BEET: RGB = (200, 170, 128)
SUGAR: RGB = (240, 240, 236)
SUGAR_PAPER: RGB = (70, 100, 180)


def _square_stack(p: Painter, x: float, y: float, half: float, tall: float) -> None:
    """A tall square brick chimney standing on (x, y), banded white near its top and dark lower
    down, smoking."""
    stack = p.box(x, y, half, tall, shade(BRICK, 0.95), (60, 52, 50))
    for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
        for drop, colour in ((2, (220, 220, 214)), (tall * 0.4, shade(SUGAR_WALL, 0.65))):
            _band(p, a, b, drop, colour)
    _smoke(p, x + 1, y - tall - 5, puffs=4)


def _refinery(p: Painter, top: Top, height: float, spans: tuple[float, ...], door: bool) -> None:
    """Dress a sugar-white block: a dark plinth stained by the damp, a cornice, rows of arched
    windows up its walls every three and a half, and a door if `door`."""
    left, front, right, _ = top
    lintel = shade(SUGAR_WALL, 1.1)
    _plinth(p, top, height, SUGAR_WALL)
    _stains(p, left, front, height, (0.08, 0.33, 0.85), shade(SUGAR_WALL, 0.6), 2)
    _stains(p, front, right, height, (0.15, 0.5, 0.85), shade(SUGAR_WALL, 0.5), 2)
    _bands(p, left, front, right, 0.8, lintel, shade(SUGAR_WALL, 0.95))
    p.line([left, front, right], shade(SUGAR_ROOF, 1.3))
    rows = tuple(2.5 + 3.5 * k for k in range(int((height - 4) // 3.5)))
    _windows(p, left, front, spans, rows, 2, arch=lintel)
    _windows(p, front, right, (0.3, 0.7), rows, 2, shade(WINDOW, 0.75))
    if door:
        _door(p, left, front, (0.55, 0.72), height - 3.4, height, (70, 50, 40), lintel)


def _loaves(p: Painter, x: float, y: float, count: int) -> None:
    """A row of white sugar loaves wrapped in blue paper at their foot, from (x, y)."""
    for k in range(count):
        lx, ly = x + 3 * k, y + 0.6 * (k % 2)
        p.polygon([(lx - 1.5, ly), (lx, ly - 6), (lx + 1.5, ly)], SUGAR, shade(SUGAR, 0.7))
        p.polygon([(lx - 1.5, ly), (lx - 1, ly - 2), (lx + 1, ly - 2), (lx + 1.5, ly)], SUGAR_PAPER)


def _sugar_sacks(p: Painter, x: float, y: float, count: int) -> None:
    """White sacks of sugar piled at (x, y)."""
    for k in range(count):
        sx, sy = x + 2.2 * (k % 3) - 1.1 * (k // 3), y - 1.6 * (k // 3) + 0.5 * (k % 2)
        p.ellipse(sx - 2, sy - 3, sx + 2, sy, SUGAR)
        p.point(sx, sy - 2.5, shade(SUGAR, 0.7))


def _beet(p: Painter, x: float, y: float, half: float, height: float) -> None:
    """A heap of sugar beet spreading over the ground at (x, y)."""
    _spill(p, x, y, half, BEET)
    _heap(p, x, y, half, height, BEET)
    for dx, dy in ((-1, -0.5), (0.5, -1.2), (-half * 0.6, -0.2)):
        p.point(x + dx, y + dy, (226, 206, 170))


def big_sugar_refinery() -> Sprite:
    """A big sugar refinery: a tall sugar-white block over a dark plinth stained by the damp with
    rows of arched windows and a cornice, a lower white boiling house beside it, a big banded
    square chimney smoking, a cart track worn out of its door past two heaps of sugar beet, and
    white sacks and sugar loaves in blue paper at its foot."""
    p = Painter(53, 80, (27, 65))
    foot = _footing(-4, -1, 18, 18)
    wing_foot = _footing(10, 4, 10, 8)
    _bed(p, foot)
    _bed(p, wing_foot)
    _bed(p, _footing(12, -7, 5, 5), margin=1)
    p.shadow(18, 4.5, reach=5)
    _track(p, foot[0], foot[1], (0.55, 0.72), 4)
    _square_stack(p, 12, -7, 2.5, 40)
    top = p.box(-4, -1, 9, 22, SUGAR_WALL, SUGAR_ROOF)
    _refinery(p, top, 22, (0.2, 0.45, 0.7, 0.92), True)
    wing = p.hall(10, 4, 10, 8, 11, SUGAR_WALL, SUGAR_ROOF)
    _refinery(p, wing, 11, (0.25, 0.5, 0.75), False)
    _door(p, wing[1], wing[2], (0.35, 0.65), 7, 11, (70, 50, 40), shade(SUGAR_WALL, 1.1))
    _beet(p, -17, 5, 2.8, 2.4)
    _beet(p, -11, 9, 2.4, 2)
    _sugar_sacks(p, 4, 12, 5)
    _loaves(p, 14, 11, 3)
    return p.sprite()


def mega_sugar_refinery() -> Sprite:
    """A towering sugar refinery: a great sugar-white block of nine storeys over a dark plinth with
    rows of arched windows, a cornice and a blue name band, a big sugar loaf standing on its roof,
    two lower white wings either side, two big banded square chimneys smoking behind, heaps of
    sugar beet with a conveyor climbing from them into the works, and rows of white sacks and
    sugar loaves in blue paper."""
    p = Painter(64, 83, (33, 67))
    _bed(p, _footing(5, -12, 5, 5), margin=1)
    _bed(p, _footing(17, -8, 5, 5), margin=1)
    _square_stack(p, 5, -12, 2.5, 37)
    _square_stack(p, 17, -8, 2.5, 33)
    foot = _footing(-4, -2, 16, 14)
    left_foot = _footing(-18, 3, 12, 8)
    right_foot = _footing(12, 5, 12, 8)
    for f in (foot, left_foot, right_foot):
        _bed(p, f)
    p.shadow(24, 5.5, reach=4)
    _track(p, right_foot[0], right_foot[1], (0.5, 0.7), 5)
    top = p.hall(-4, -2, 16, 14, 32, SUGAR_WALL, SUGAR_ROOF)
    left, front, right, back = top
    _refinery(p, top, 32, (0.14, 0.32, 0.5, 0.68, 0.86), False)
    _patch(p, left, front, (0.1, 0.9), (1.6, 3.2), SUGAR_PAPER)
    _patch(p, front, right, (0.1, 0.9), (1.6, 3.2), shade(SUGAR_PAPER, 0.78))
    for k in range(6):
        p.point(*_on_wall(left, front, 0.2 + 0.12 * k, 2.2), SUGAR)
    sx, sy = (front[0] + back[0]) / 2, (front[1] + back[1]) / 2 + 1
    p.polygon([(sx - 2.6, sy), (sx, sy - 10), (sx + 2.6, sy)], SUGAR, shade(SUGAR, 0.7))
    p.polygon([(sx, sy - 10), (sx + 2.6, sy), (sx + 0.8, sy + 0.6)], shade(SUGAR, 0.85))
    p.polygon([(sx - 2.6, sy), (sx - 1.8, sy - 3), (sx + 1.8, sy - 3), (sx + 2.6, sy)], SUGAR_PAPER)
    wing = p.hall(-18, 3, 12, 8, 14, SUGAR_WALL, SUGAR_ROOF)
    _refinery(p, wing, 14, (0.2, 0.4, 0.6, 0.8), False)
    p.line([(-25, 6), (-16, -10)], (90, 88, 92))
    p.line([(-24, 6.5), (-15, -9.5)], shade(BEET, 0.7))
    wing = p.hall(12, 5, 12, 8, 16, SUGAR_WALL, SUGAR_ROOF)
    _refinery(p, wing, 16, (0.2, 0.4, 0.6, 0.8), True)
    _beet(p, -25, 8, 3.2, 2.8)
    _beet(p, -19, 11, 2.8, 2.4)
    _sugar_sacks(p, -6, 13, 6)
    _sugar_sacks(p, 2, 13, 3)
    _loaves(p, 18, 13, 4)
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


def _slaughterhouse(p: Painter, x: float, y: float, length: float, height: float) -> Top:
    """A brick slaughterhouse over a dark plinth with rows of arched windows, a dark loading door
    with a timber ramp coming down from it and a louvred vent along its slate ridge."""
    lintel = shade(BRICK, 1.3)
    top = p.hall(x, y, length, 9, height, BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, height, BRICK)
    _stains(p, left, front, height, (0.06, 0.2, 0.55, 0.85), shade(BRICK, 0.62), 1.8)
    rows = tuple(2 + 3.4 * k for k in range(int((height - 3) // 3.4)))
    _windows(p, left, front, (0.1, 0.2, 0.55, 0.67, 0.79, 0.91), rows, 2, arch=lintel)
    _windows(p, front, right, (0.3, 0.7), rows, 2, shade(WINDOW, 0.8), shade(BRICK, 1.1))
    _patch(p, left, front, (0.3, 0.45), (height - 5, height - 2), (56, 44, 40))
    p.line(
        [_on_wall(left, front, 0.28, height - 5.6), _on_wall(left, front, 0.47, height - 5.6)],
        lintel,
    )
    ramp = [_on_wall(left, front, t, height - 2) for t in (0.3, 0.45)]
    p.polygon(
        [ramp[0], ramp[1], (ramp[1][0] - 4, ramp[1][1] + 3), (ramp[0][0] - 4, ramp[0][1] + 3)],
        TIMBER,
        shade(TIMBER, 0.6),
    )
    for k in (1, 2, 3):
        p.line(
            [(ramp[0][0] - k, ramp[0][1] + 0.75 * k), (ramp[1][0] - k, ramp[1][1] + 0.75 * k)],
            shade(TIMBER, 0.75),
        )
    p.gable(top, 4.5, MEAT_ROOF, BRICK)
    _roof_rows(p, top, 4.5, MEAT_ROOF)
    _louvre(p, top, 4.5, (0.2, 0.8))
    return top


def _pen(p: Painter, x: float, y: float, half: float, cows: tuple[Point, ...]) -> None:
    """A cattle pen trampled to mud centred at (x, y), fenced, with brown steers in it."""
    corners = _paddock(x, y, half)
    pen_left, pen_back, pen_right, pen_front = corners
    p.polygon(list(corners), MUD)
    for dx, dy in ((-2, 0), (2, 2), (0, -1.4), (-half * 0.6, 1)):
        p.point(x + dx, y + dy, shade(MUD, 0.75))
    _fence(p, pen_left, pen_back)
    _fence(p, pen_back, pen_right)
    for cx, cy in cows:
        _cow(p, cx, cy, *STEER)
    _fence(p, pen_left, pen_front)
    _fence(p, pen_front, pen_right)


def _cold_store(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> Top:
    """A white cold store with panelled walls and insulated grey-blue doors, refrigeration units
    with dark fans and a vent pipe on its flat roof."""
    top = p.hall(x, y, length, depth, height, COLD, shade(COLD, 0.88))
    left, front, right, back = top
    _plinth(p, top, height, COLD, 0.6)
    _stains(p, front, right, height, (0.15, 0.85), shade(COLD, 0.66), 1.2)
    panels = max(3, round(length / 3))
    for k in range(1, panels):
        t = k / panels
        p.line(
            [_on_wall(left, front, t, 0.5), _on_wall(left, front, t, height - 0.6)],
            shade(COLD, 0.85),
        )
    _patch(p, front, right, (0.3, 0.7), (height - 4.4, height), COLD_DOOR)
    p.line(
        [_on_wall(front, right, 0.5, height - 4.4), _on_wall(front, right, 0.5, height)],
        shade(COLD, 0.6),
    )
    p.point(*_on_wall(front, right, 0.62, height - 2.2), (60, 62, 70))
    if length > 10:
        _patch(p, left, front, (0.4, 0.6), (height - 4.4, height), COLD_DOOR)
    units = max(1, round(length / 7))
    for k in range(units):
        ux, uy = _on_wall(left, front, (k + 0.5) / units, 0)
        ux, uy = ux + depth * 0.45, uy - depth * 0.22
        p.box(ux, uy, 2, 2, METAL, shade(METAL, 1.1))
        p.blob(ux - 1, uy - 3.6, ux + 1, uy - 1.8, (60, 62, 70))
    vx, vy = _on_wall(back, right, 0.7, 0)
    p.line([(vx - 2, vy), (vx - 2, vy - 5)], METAL)
    p.rect(vx - 3, vy - 6, vx - 1, vy - 5, shade(METAL, 0.8))
    return top


def big_meat_factory() -> Sprite:
    """A big meat works: a long brick slaughterhouse with two rows of arched windows, a louvred
    vent along its ridge, a chimney smoking and a ham hung out as its sign, a timber ramp down to
    a cattle pen trampled to mud with three steers in it, and a white cold store with insulated
    doors and refrigeration units on its roof, a track worn out of it past a puddle of melt
    water."""
    p = Painter(53, 55, (26, 42))
    foot = _footing(-2, -4, 18, 9)
    store_foot = _footing(15, 4, 8, 7)
    _bed(p, foot)
    _bed(p, store_foot, margin=1)
    p.shadow(20, 5)
    _track(p, store_foot[1], store_foot[2], (0.3, 0.7), 3.5)
    _puddle(p, 21, 9.5, 1.5)
    top = _slaughterhouse(p, -2, -4, 18, 9)
    left, front, _, _ = top
    _chimney(p, -6, -18, -30, BRICK)
    _smoke(p, -5.5, -32, (200, 200, 204), 3)
    _ham(p, *_on_wall(left, front, 0.04, 2), 1)
    _pen(p, -17, 5, 6, ((-19, 4), (-15, 6.5), (-15, 2.5)))
    _cold_store(p, 15, 4, 8, 7, 8)
    return p.sprite()


def mega_meat_factory() -> Sprite:
    """A great meat-packing works: a long two-storey brick slaughterhouse with rows of arched
    windows, a long louvred vent along its ridge and a big ham hung out as its sign, a tall chimney
    smoking, timber ramps down to two cattle pens full of steers, a big white cold store with
    insulated doors and a row of refrigeration units on its roof, and a white refrigerated van on
    a rail siding at its door."""
    p = Painter(64, 78, (32, 59))
    foot = _footing(-4, -6, 26, 10)
    store_foot = _footing(16, 0, 13, 9)
    _bed(p, foot)
    _bed(p, store_foot, margin=1)
    p.shadow(25, 5.5)
    _chimney(p, -14, -18, -44, BRICK)
    _smoke(p, -13.5, -46, (200, 200, 204), 4)
    top = _slaughterhouse(p, -4, -6, 26, 12)
    left, front, _, _ = top
    _ham(p, *_on_wall(left, front, 0.03, 2), 1.4)
    _cold_store(p, 16, 0, 13, 9, 12)
    _rails(p, (5, 15), (26, 4.5))
    van = _wagon(p, 18, 9, 11, COLD, shade(COLD, 0.88))
    for t in (0.25, 0.75):
        _patch(p, van[1], van[2], (t - 0.08, t + 0.08), (0.8, 1.8), COLD_DOOR)
    _pen(p, -19, 6, 7, ((-21, 5), (-17, 3), (-17, 8), (-23, 7.5)))
    _pen(p, -6, 11, 5, ((-7, 11), (-4, 12.5)))
    _puddle(p, 25, 11, 1.5)
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
    _chimney(p, -18, -17, -30, BRICK)
    _smoke(p, -17.5, -32, puffs=3)
    _shavings(p, ((-9, 8.5), (-6, 9.4), (-12, 7.6), (-4, 8.6), (-8, 10.4), (-1, 10)))
    for x, y in ((8, 10), (13, 7.5), (18, 5)):
        _firing(p, x, y)
    _staves(p, -24, 2, 3)
    _staves(p, -20, 5, 2)
    _casks(p, -16, 14, 4)
    _cart(p, 22, 12, (BARREL, BARREL, BARREL))
    return p.sprite()
