"""Big and mega clock makers, mirror makers and steel mills, each drawn as a building of its own:
clock towers with gilded domes, halls of mirrors under a glass dome, and steelworks of blast
furnaces and converters throwing fire."""

from railroads.ui.pixel.art.industry import (
    STEEL_COKE,
    STEEL_GROUND,
    STEEL_LIME,
    STEEL_ORE,
    _bed,
    _door,
    _footing,
    _ring,
    _roof_rows,
    _steel_bessemer,
    _steel_bins,
    _steel_cooling_bed,
    _steel_dust_catcher,
    _steel_flat,
    _steel_furnace,
    _steel_hopper,
    _steel_hot_blast,
    _steel_hot_car,
    _steel_incline,
    _steel_ingots,
    _steel_shed,
    _steel_siding,
    _steel_slag,
    _steel_stack,
    _steel_stock,
    _steel_stove,
    _steel_teeming,
    _steel_trestle,
    _tall_chimney,
    _window,
)
from railroads.ui.pixel.art.painter import (
    RGB,
    SLATE,
    VERDIGRIS,
    WINDOW,
    Painter,
    Point,
    Sprite,
    Top,
    shade,
)
from railroads.ui.pixel.art.parts import (
    METRE,
    _band,
    _bands,
    _iso,
    _on_wall,
    _patch,
    _plinth,
    _turned,
    _windows,
)

STONE_WALL: RGB = (228, 216, 188)
GREEN: RGB = (40, 90, 60)
GOLD: RGB = (230, 188, 70)
FACE: RGB = (248, 246, 236)
HANDS: RGB = (30, 30, 34)

MIRROR_WALL: RGB = (234, 228, 216)
GILT: RGB = (214, 170, 60)
MIRROR: RGB = (176, 214, 238)
GLINT: RGB = (252, 254, 255)
GLASSHOUSE: RGB = (196, 226, 240)


def _clock_face(p: Painter, x: float, y: float, r: float) -> None:
    """A white clock face with a gold rim and its hands at ten past ten, centred on (x, y)."""
    p.ellipse(x - r - 0.6, y - r - 0.6, x + r + 0.6, y + r + 0.6, GOLD)
    p.ellipse(x - r, y - r, x + r, y + r, FACE)
    p.line([(x, y), (x - r * 0.55, y - r * 0.45)], HANDS)
    p.line([(x, y), (x + r * 0.7, y - r * 0.4)], HANDS)


def _clock_tower(p: Painter, x: float, y: float, half: float, tall: float, face: float) -> Top:
    """A square stone clock tower standing on (x, y) with a clock on both its walls near the top
    and a green copper spire with a gold finial."""
    top = p.box(x, y, half, tall, STONE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, tall, STONE_WALL)
    for f in (0.35, 0.65):
        _windows(p, left, front, (0.5,), (tall * f,), 2, shade(WINDOW, 1.1))
    for a, b in ((left, front), (front, right)):
        cx, cy = _on_wall(a, b, 0.5, face + 1.4)
        _clock_face(p, cx, cy, face)
    _bands(p, left, front, right, 0.6, GREEN, shade(GREEN, 0.8))
    apex = (front[0], (left[1] + right[1]) / 2 - tall * 0.45)
    p.polygon([left, front, apex], VERDIGRIS, shade(VERDIGRIS, 0.6))
    p.polygon([front, right, apex], shade(VERDIGRIS, 0.75), shade(VERDIGRIS, 0.5))
    p.line([(apex[0], apex[1]), (apex[0], apex[1] - 2)], GOLD)
    p.point(apex[0], apex[1] - 2.5, GOLD)
    return top


def _pocket_watch(p: Painter, x: float, y: float) -> None:
    """A big gold pocket watch hung from a bracket at (x, y) as a sign."""
    p.line([(x, y), (x - 3, y)], HANDS)
    p.line([(x - 3, y), (x - 3, y + 1)], GOLD)
    _clock_face(p, x - 3, y + 3.4, 1.8)


def big_clock_maker() -> Sprite:
    """A two-storey stone watch workshop with a green sign over its shop window of gold watches,
    a big pocket watch hung out as a sign, and a square clock tower behind it under a copper
    spire."""
    p = Painter(36, 50, (17, 40))
    p.shadow(13, 3.5)
    _clock_tower(p, 2, -5, 3.4, 22, 1.9)
    foot = _footing(0, 0, 14, 7)
    _bed(p, foot)
    top = p.hall(0, 0, 14, 7, 9, STONE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 9, STONE_WALL)
    _windows(p, left, front, (0.15, 0.35, 0.55, 0.75, 0.92), (1.6,), 2, shade(WINDOW, 1.1))
    _patch(p, left, front, (0.08, 0.6), (4.4, 5.4), GREEN)
    _patch(p, left, front, (0.1, 0.58), (5.8, 8), shade(WINDOW, 0.9))
    for t in (0.18, 0.3, 0.42, 0.52):
        p.point(*_on_wall(left, front, t, 6.8), GOLD)
    _door(p, left, front, (0.68, 0.84), 5, 9, GREEN)
    _window(p, front, right, (0.3, 0.7), 2, shade(WINDOW, 0.85))
    p.gable(top, 4, SLATE, STONE_WALL)
    _roof_rows(p, top, 4, SLATE)
    _pocket_watch(p, *_on_wall(left, front, 0.02, 3.5))
    return p.sprite()


def mega_clock_maker() -> Sprite:
    """A grand watch manufactory: a long stone hall of three storeys of windows under a slate roof
    with green trim, and a tall clock tower before its middle with big clocks on its walls and a
    gilded dome, its foot set in a garden with a gravel path."""
    p = Painter(50, 62, (25, 50))
    p.blob(-14, 1, 4, 9, (120, 168, 84))
    p.shadow(19, 5)
    foot = _footing(0, 0, 24, 9)
    _bed(p, foot)
    top = p.hall(0, 0, 24, 9, 12, STONE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 12, STONE_WALL)
    spans = (0.06, 0.16, 0.26, 0.58, 0.68, 0.78, 0.88, 0.96)
    for drop in (1.6, 5.2, 8.6):
        _windows(p, left, front, spans, (drop,), 2, shade(WINDOW, 1.1), shade(STONE_WALL, 1.1))
    _windows(p, front, right, (0.25, 0.5, 0.75), (1.6, 5.2, 8.6), 2, shade(WINDOW, 0.85))
    _bands(p, left, front, right, 0.6, GREEN, shade(GREEN, 0.8))
    p.gable(top, 5, SLATE, STONE_WALL)
    _roof_rows(p, top, 5, SLATE, (0.25, 0.5, 0.75))
    gx, gy = _on_wall(foot[0], foot[1], 0.42, 0)
    p.polygon(
        [(gx - 2, gy + 1), (gx + 2, gy + 3), (gx - 6, gy + 8), (gx - 10, gy + 6)], (206, 196, 170)
    )
    tower = _clock_tower(p, gx, gy + 1.5, 4, 30, 2.6)
    tl, tf, tr, _ = tower
    _door(p, tl, tf, (0.3, 0.7), 26, 30, GREEN)
    dome_x, dome_y = tf[0], (tl[1] + tr[1]) / 2 + 0.5
    _turned(p, dome_x, dome_y, [(0, 3.4), (2, 3.2), (4, 2.2), (5, 1)], GOLD, shade(GOLD, 1.15))
    p.line([(dome_x, dome_y - 5), (dome_x, dome_y - 8)], GOLD)
    p.point(dome_x, dome_y - 8.5, GLINT)
    for x, y in ((-18, -8), (17, -8)):
        p.rect(x - 1, y - 6, x + 1, y, STONE_WALL)
        p.polygon([(x - 1.6, y - 6), (x, y - 9), (x + 1.6, y - 6)], VERDIGRIS)
    return p.sprite()


def _mirror_window(
    p: Painter, a: Point, b: Point, t: float, half: float, drops: tuple[float, float]
) -> None:
    """A tall arched mirror in a gilt frame on the wall under the edge from `a` to `b`, glinting."""
    d0, d1 = drops
    _patch(p, a, b, (t - half - 0.02, t + half + 0.02), (d0 - 0.6, d1 + 0.4), GILT)
    _patch(p, a, b, (t - half, t + half), (d0, d1), MIRROR)
    p.point(*_on_wall(a, b, t, d0 - 1), GILT)
    p.line(
        [_on_wall(a, b, t - half * 0.6, d1 - 1), _on_wall(a, b, t + half * 0.5, d0 + 0.8)], GLINT
    )


def _leaning_mirror(p: Painter, x: float, y: float, tall: float = 6) -> None:
    """A tall mirror in a gilt frame leant on the ground at (x, y), the sky glinting in it."""
    p.rect(x - 1.6, y - tall, x + 1.6, y, GILT)
    p.rect(x - 0.8, y - tall + 0.8, x + 0.8, y - 0.8, MIRROR)
    p.line([(x - 0.6, y - 2), (x + 0.6, y - tall + 1.6)], GLINT)


def _glasshouse(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> None:
    """A glazed workshop: pale glass walls with white glazing bars under a ridged glass roof."""
    top = p.hall(x, y, length, depth, height, GLASSHOUSE, None)
    left, front, right, _ = top
    for t in (0.2, 0.4, 0.6, 0.8):
        p.line([_on_wall(left, front, t, 0), _on_wall(left, front, t, height)], GLINT)
    for t in (0.33, 0.66):
        p.line([_on_wall(front, right, t, 0), _on_wall(front, right, t, height)], GLINT)
    p.gable(top, 2.5, shade(GLASSHOUSE, 1.05), GLASSHOUSE)
    _band(p, left, front, height - 0.6, shade(GILT, 0.8))


def big_mirror_maker() -> Sprite:
    """A pale mirror works with a row of tall arched mirrors in gilt frames for a front, a gilded
    cornice, a glasshouse workshop beside it, and mirrors leant outside glinting in the sun."""
    p = Painter(38, 36, (18, 26))
    p.shadow(14, 3.5)
    _glasshouse(p, 10, -3, 7, 6, 5)
    foot = _footing(-1, 1, 14, 7)
    _bed(p, foot)
    top = p.hall(-1, 1, 14, 7, 9, MIRROR_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 9, MIRROR_WALL)
    for t in (0.14, 0.34, 0.54, 0.74):
        _mirror_window(p, left, front, t, 0.06, (2, 7.2))
    _door(p, left, front, (0.86, 0.96), 4.5, 9, GILT)
    _mirror_window(p, front, right, 0.5, 0.15, (2, 6.5))
    _bands(p, left, front, right, 0.6, GILT, shade(GILT, 0.8))
    p.gable(top, 4, SLATE, MIRROR_WALL)
    _roof_rows(p, top, 4, SLATE)
    for x, y in ((-12, 5), (-9, 6.5), (2, 8)):
        _leaning_mirror(p, x, y)
    for x, y in ((-6, -10), (8, -6), (-14, -2)):
        p.point(x, y, GLINT)
    return p.sprite()


def mega_mirror_maker() -> Sprite:
    """A hall of mirrors: a long pale palace of a works whose front is a row of tall arched mirrors
    between gilt pilasters, under a slate roof crowned with a glass dome banded in gold, with a
    glasshouse wing behind, a cart of mirrors at its door and glints all over it."""
    p = Painter(52, 50, (26, 37))
    p.shadow(20, 5)
    _glasshouse(p, 12, -6, 10, 7, 6)
    foot = _footing(-2, 1, 24, 9)
    _bed(p, foot)
    top = p.hall(-2, 1, 24, 9, 12, MIRROR_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 12, MIRROR_WALL)
    steps = 7
    for k in range(steps):
        t = (k + 0.5) / steps
        _mirror_window(p, left, front, t, 0.042, (2.4, 10))
        if k:
            p.line(
                [_on_wall(left, front, k / steps, 1), _on_wall(left, front, k / steps, 11)], GILT
            )
    for t in (0.3, 0.7):
        _mirror_window(p, front, right, t, 0.1, (2.4, 9.5))
    _bands(p, left, front, right, 0.6, GILT, shade(GILT, 0.8))
    p.gable(top, 5, SLATE, MIRROR_WALL)
    _roof_rows(p, top, 5, SLATE, (0.25, 0.5, 0.75))
    dx, dy = _on_wall(left, right, 0.5, -3.5)
    _turned(
        p, dx, dy, [(0, 4.5), (2.5, 4.3), (5, 3), (6.5, 1.2)], GLASSHOUSE, shade(GLASSHOUSE, 1.1)
    )
    for h in (0.5, 2.5):
        _ring(p, dx, dy - h, 4.4, GILT)
    p.line([(dx, dy - 7), (dx, dy - 10)], GILT)
    p.point(dx, dy - 10.5, GLINT)
    p.line([(dx - 2.5, dy - 2), (dx - 1, dy - 5)], GLINT)
    cx, cy = _on_wall(foot[0], foot[1], 0.2, 3.5)
    p.rect(cx - 4, cy - 2, cx + 3, cy, (120, 84, 56))
    for k in range(3):
        _leaning_mirror(p, cx - 2.5 + 2.4 * k, cy - 1.5, 5)
    for wx in (cx - 3, cx + 2):
        p.ellipse(wx - 1, cy - 0.5, wx + 1, cy + 1.5, (70, 54, 40))
    for x, y in ((-20, -12), (-6, -20), (10, -16), (20, -6), (-12, -6), (4, -4)):
        p.point(x, y, GLINT)
    return p.sprite()


def big_steel_mill() -> Sprite:
    """A steelworks grown from the ironworks: two blast furnaces of light riveted steel with their
    headgears and arching uptakes, each fed by a skip incline from a grid of stock bins heaped
    with ore, coke and limestone where hopper wagons stand on the siding behind, a downcomer
    falling to the dust catcher, four domed hot-blast stoves on their main and three steel stacks
    smoking, two red brick Bessemer sheds open on a converter blowing fire and sparks and another
    pouring its steel into a ladle, a long rolling mill under a rust-red roof with a glazed
    clerestory, its door glowing over a bogie of red-hot ingots, and on the siding in front a
    hot-metal car, ingots teemed from a ladle, a wagon of rails and stacks of blue-grey beams."""
    m = METRE
    p = Painter(58, 70, (29, 42))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 26) * m, (v - 9) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    ground = [(-2, 1), (8, -7.5), (60, -7.5), (58, 12), (44, 28), (10, 28), (-6, 14)]
    p.polygon([at(u, v) for u, v in ground], STEEL_GROUND)
    p.soft([(*at(-4, 10), *at(16, 0))], shade(STEEL_ORE, 0.9), 0.35, 1.5)
    p.shadow(22, 6, 2)
    for u in (21, 28, 35):
        _steel_stack(p, *at(u, 26.5), 48 * m)
    dust = _steel_dust_catcher(p, *at(21.5, 21))
    heads = []
    for u in (4.5, 15.5):
        fx, fy = at(u, 17.5)
        heads.append((fy, _steel_furnace(p, fx, fy, 19 * m, 4.6 * m, dust)))
    for u in (27, 32.5, 38, 43.5):
        _steel_stove(p, *at(u, 19.5), 15 * m, 2.9 * m)
    _steel_hot_blast(p, [up(44, 16, 7), up(18.5, 16, 7), up(17, 16, 8.2)])
    _steel_siding(p, *at(-6, 13.5), 24)
    for u, load in ((-1, STEEL_ORE), (6, STEEL_COKE), (13, STEEL_ORE)):
        _steel_hopper(p, *at(u, 13.5), load)
    for (fy, head), u in zip(heads, (3.5, 13), strict=True):
        under = (head[0], fy - 1.5 * m)
        _steel_incline(p, at(u, 10.6), under, under[1] - head[1], _iso(0, 0, 1.7 * m, 0))
    _steel_bins(
        p,
        *at(-1.5, 10.6),
        (
            (STEEL_ORE, STEEL_COKE, STEEL_ORE, STEEL_LIME),
            (STEEL_COKE, STEEL_ORE, STEEL_LIME, STEEL_COKE),
        ),
    )
    _steel_bessemer(p, *at(20.5, 5), 7 * m, 10 * m, 10 * m, 4)
    _steel_bessemer(p, *at(27.5, 5), 7 * m, 10 * m, 10 * m, 5, pouring=True)
    _steel_shed(p, *at(44.5, 5.5), 25 * m, 11 * m, 9 * m, 3.3 * m, 0.72)
    for u in (38, 46.5):
        _steel_stock(p, *at(u, -1.8), 7.5, 2.2, 3)
    _steel_siding(p, *at(7, -5.5), 55)
    _steel_hot_car(p, *at(12, -5.5))
    _steel_teeming(p, *at(24, -5.5))
    _steel_ingots(p, *at(50.5, -5.5), 3)
    _steel_flat(p, *at(57, -5.5))
    return p.sprite()


def mega_steel_mill() -> Sprite:
    """A great integrated steelworks: a row of three blast furnaces of light riveted steel with
    their headgears and arching uptakes, six domed hot-blast stoves and four steel stacks smoking
    behind them, each furnace fed by a skip incline from a grid of stock bins under a high-line
    trestle where hopper wagons tip their ore and coke, a slag tip glowing where a pot is
    tipped down it, a Bessemer shop of three red brick bays open on converters blowing fire and
    showering sparks or pouring into a ladle, a long rolling mill under a rust-red roof with a
    glazed clerestory and its tall brick chimney, its door glowing over red-hot ingots and a
    cooling bed of fresh rails, and a busy siding in front: hot-metal cars, ingots teemed from a
    ladle, a wagon of rails and a stack of blue-grey beams."""
    m = METRE
    p = Painter(64, 80, (33, 46))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 28) * m, (v - 10) * m)

    ground = [(-6, 4), (6, -8), (64, -8), (62, 12), (40, 32), (0, 32), (-9, 18)]
    p.polygon([at(u, v) for u, v in ground], STEEL_GROUND)
    p.soft([(*at(-6, 12), *at(12, 2))], shade(STEEL_ORE, 0.9), 0.35, 1.5)
    p.shadow(25, 7, 1)
    for u in (12, 18, 24, 30):
        _steel_stack(p, *at(u, 30), 50 * m)
    for u in (16, 21, 26, 31, 36, 41):
        _steel_stove(p, *at(u, 25), 15 * m, 2.8 * m)
    dusts = [_steel_dust_catcher(p, *at(u, 22)) for u in (5.5, 16, 26.5)]
    heads = []
    for u, dust in zip((0, 10.5, 21), dusts, strict=True):
        fx, fy = at(u, 18.5)
        heads.append((fy, _steel_furnace(p, fx, fy, 19 * m, 4.6 * m, dust)))
    for (fy, head), u in zip(heads, (0, 9.5, 19), strict=True):
        under = (head[0], fy - 1.5 * m)
        _steel_incline(p, at(u, 12.4), under, under[1] - head[1], _iso(0, 0, 1.7 * m, 0))
    deck = _steel_trestle(p, *at(-4, 14.2), 24, 6)
    for k, load in enumerate((STEEL_ORE, STEEL_COKE, STEEL_ORE)):
        _steel_hopper(p, *_iso(deck[0], deck[1], (k - 1) * 7 * m, 0), load)
    _steel_slag(p, *at(-6, 9), 4.5 * m, 5.5 * m)
    _steel_bins(
        p,
        *at(-2.5, 12.4),
        (
            (STEEL_ORE, STEEL_COKE, STEEL_ORE, STEEL_LIME, STEEL_ORE),
            (STEEL_COKE, STEEL_ORE, STEEL_LIME, STEEL_COKE, STEEL_ORE),
        ),
    )
    for k, u in enumerate((24, 30, 36)):
        _steel_bessemer(p, *at(u, 5), 6 * m, 10 * m, 10.5 * m, 6 + k, pouring=k == 1)
    sx, sy = at(56, 15)
    _tall_chimney(p, sx, sy, 52 * m)
    _steel_shed(p, *at(51, 5.5), 24 * m, 11 * m, 9.5 * m, 3.4 * m, 0.7)
    _steel_stock(p, *at(41.5, -1.8), 5, 2.2, 3)
    _steel_cooling_bed(p, *at(49.5, -2.1), 9, 6)
    _steel_siding(p, *at(5, -5.5), 60)
    for u in (9.5, 15):
        _steel_hot_car(p, *at(u, -5.5))
    _steel_teeming(p, *at(28, -5.5))
    _steel_flat(p, *at(46, -5.5))
    _steel_ingots(p, *at(56, -5.5), 3)
    return p.sprite()
