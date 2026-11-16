"""Big and mega clock makers, mirror makers and steel mills, each drawn as a building of its own:
clock towers with gilded domes, halls of mirrors under a glass dome, and steelworks of blast
furnaces and converters throwing fire."""

from railroads.ui.pixel.art.grand_fire import WHITE_HOT
from railroads.ui.pixel.art.industry import (
    SMOKE,
    _bed,
    _chimney,
    _door,
    _footing,
    _ring,
    _roof_rows,
    _sawtooth,
    _smoke,
    _spill,
    _stains,
    _window,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    RGB,
    SLATE,
    VERDIGRIS,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import (
    _band,
    _bands,
    _heap,
    _on_wall,
    _patch,
    _plinth,
    _turned,
    _windows,
)

type Top = tuple[Point, Point, Point, Point]

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

IRON: RGB = (74, 72, 78)
RUST: RGB = (130, 76, 56)
STOVE: RGB = (140, 138, 144)
CINDER: RGB = (58, 54, 56)
MOLTEN: RGB = (246, 132, 36)
ORE: RGB = (150, 76, 54)


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


def _blast_furnace(p: Painter, x: float, y: float, tall: float) -> None:
    """A round banded blast furnace standing on (x, y), its throat glowing at the top, a stream of
    molten iron running from its tap at the foot."""
    _turned(
        p,
        x,
        y,
        [(0, 3), (tall * 0.25, 3), (tall * 0.75, 2.2), (tall, 1.6)],
        shade(IRON, 0.9),
        (40, 34, 32),
    )
    for f in (0.2, 0.45, 0.7):
        _ring(p, x, y - tall * f, 3 - f, shade(IRON, 0.6))
    p.blob(x - 1.2, y - tall - 1.2, x + 1.2, y - tall + 0.4, MOLTEN)
    p.point(x, y - tall - 0.6, WHITE_HOT)
    p.blob(x - 1.6, y - 1.6, x + 0.6, y + 0.4, MOLTEN)


def _stove(p: Painter, x: float, y: float, tall: float) -> None:
    """A tall grey steel hot-blast stove with a domed top banded in rust, standing on (x, y)."""
    _turned(p, x, y, [(0, 1.8), (tall - 1.5, 1.8), (tall, 1)], STOVE, shade(STOVE, 0.7))
    for f in (0.3, 0.65):
        _ring(p, x, y - tall * f, 1.8, RUST)


def _sparks(p: Painter, points: tuple[tuple[float, float], ...]) -> None:
    for k, (x, y) in enumerate(points):
        p.point(x, y, WHITE_HOT if k % 2 else MOLTEN)


def _casting_hall(
    p: Painter, x: float, y: float, length: float, depth: float, height: float
) -> Top:
    """A dark iron casting hall with a row of windows glowing from the fire inside and its door
    open on molten iron."""
    top = p.hall(x, y, length, depth, height, IRON, None)
    left, front, right, _ = top
    _plinth(p, top, height, IRON)
    _stains(p, front, right, height, (0.2, 0.5, 0.8), shade(IRON, 0.55), 2)
    spans = tuple((k + 0.5) / 7 for k in range(7))
    _windows(p, left, front, spans, (2,), 3, shade(MOLTEN, 0.85), shade(IRON, 1.6))
    _patch(p, front, right, (0.3, 0.7), (height - 4, height), MOLTEN)
    _patch(p, front, right, (0.4, 0.6), (height - 3, height), WHITE_HOT)
    return top


def _ladle_car(p: Painter, x: float, y: float) -> None:
    """A ladle car brimming with molten iron, standing on (x, y)."""
    p.rect(x - 2.4, y - 1, x + 2.4, y, (40, 38, 40))
    p.ellipse(x - 2, y - 4, x + 2, y - 0.6, shade(IRON, 0.8))
    p.blob(x - 1.4, y - 4.2, x + 1.4, y - 3, MOLTEN)


def big_steel_mill() -> Sprite:
    """A steelworks on blackened ground: two blast furnaces glowing at their throats with a row of
    rust-red hot-blast stoves, a dark casting hall whose windows glow with fire, molten iron
    running into ladle cars, two banded chimneys smoking and sparks flying."""
    p = Painter(46, 60, (22, 46))
    foot = _footing(2, 1, 18, 8)
    _bed(p, foot, CINDER, 3)
    _spill(p, -15, 4, 3, ORE)
    p.shadow(17, 4.5)
    for x, y in ((9, -9), (12, -7.5), (15, -6)):
        _stove(p, x, y, 18)
    for x, y in ((-4, -6), (4, -7)):
        _blast_furnace(p, x, y, 22)
    _chimney(p, -10, -6, -34, shade(BRICK, 0.75))
    _smoke(p, -9.5, -36, SMOKE, 4)
    top = _casting_hall(p, 2, 1, 18, 8, 7)
    p.gable(top, 3, (60, 58, 62), IRON)
    _heap(p, -15, 4, 3, 3, ORE)
    for x in (12, 16):
        _ladle_car(p, x, 9)
    p.line([(8, 8), (11, 7.4)], MOLTEN)
    _sparks(p, ((-6, -32), (-2, -33), (3, -34), (6, -31), (-4, -36), (5, -37)))
    return p.sprite()


def mega_steel_mill() -> Sprite:
    """A great integrated steelworks: three blast furnaces with their stoves, a Bessemer converter
    tipped and throwing a fountain of fire and sparks, a long rolling mill under a sawtooth roof
    glowing from inside, a row of four tall chimneys smoking, glowing slag on the cinders and
    ladle cars of molten iron on the tracks."""
    p = Painter(58, 72, (29, 54))
    foot = _footing(0, 2, 26, 9)
    _bed(p, foot, CINDER, 4)
    _spill(p, -21, 6, 3.5, ORE)
    p.shadow(23, 6)
    for k, x in enumerate((-14, -6, 2, 10)):
        _chimney(p, x, -14 + k, -48 + k, shade(BRICK, 0.72))
        _smoke(p, x + 0.5, -50 + k, SMOKE, 4)
    for x, y in ((14, -12), (17, -10.5), (20, -9), (23, -7.5)):
        _stove(p, x, y, 20)
    for x, y in ((-10, -8), (-2, -9), (6, -10)):
        _blast_furnace(p, x, y, 26)
    top = _casting_hall(p, 0, 2, 26, 9, 9)
    _sawtooth(p, top, 4, 3, (64, 62, 66), shade(MOLTEN, 0.9), IRON)
    cx, cy = 17, 9
    p.ellipse(cx - 3, cy - 7, cx + 3, cy - 1, shade(IRON, 0.85))
    p.rect(cx - 4, cy - 4, cx + 4, cy - 3, (40, 38, 40))
    p.polygon([(cx - 1, cy - 7), (cx - 6, cy - 16), (cx - 2, cy - 18), (cx + 2, cy - 7)], MOLTEN)
    p.polygon(
        [(cx - 0.5, cy - 7.5), (cx - 4, cy - 14), (cx - 2, cy - 15), (cx + 1, cy - 7.5)], WHITE_HOT
    )
    _sparks(
        p,
        (
            (cx - 7, cy - 18),
            (cx - 4, cy - 21),
            (cx, cy - 20),
            (cx - 8, cy - 14),
            (cx + 2, cy - 16),
            (cx - 6, cy - 23),
            (cx + 3, cy - 22),
            (cx - 9, cy - 19),
        ),
    )
    _heap(p, -21, 6, 3.5, 3, ORE)
    p.blob(-14, 9, -7, 12, shade(MOLTEN, 0.75))
    p.blob(-12, 9.6, -9, 11, MOLTEN)
    for x in (4, 9):
        _ladle_car(p, x, 13)
    _sparks(p, ((-10, -38), (-2, -40), (6, -41), (-6, -43), (2, -44)))
    return p.sprite()
