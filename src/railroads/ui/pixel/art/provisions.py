"""Bakeries and dairies, small, big and mega. A bakery is a warm honey-coloured house under a red
roof, with a striped awning over its shop window, a domed bread oven and a tall chimney smoking;
a dairy is whitewashed with a blue roof and blue trim, milk churns out front and cows in a fenced
yard beside it."""

from railroads.ui.pixel.art.industry import (
    _bed,
    _chimney,
    _door,
    _footing,
    _ring,
    _roof_rows,
    _sack,
    _smoke,
    _track,
    _window,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    RGB,
    WHITEWASH,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import _cow, _fence, _on_wall, _paddock, _patch, _plinth

CRUST: RGB = (236, 180, 96)
BAKERY_ROOF: RGB = (172, 66, 46)
AWNING: tuple[RGB, RGB] = ((200, 46, 42), (246, 240, 228))
LOAF: RGB = (204, 130, 52)
SMOKE: RGB = (232, 230, 232)
EMBERS: RGB = (250, 150, 48)
FLOUR: RGB = (226, 220, 204)
MILK: RGB = (246, 246, 240)
DAIRY_BLUE: RGB = (58, 104, 178)
CHURN: RGB = (204, 208, 216)
HIDE: RGB = (242, 240, 234)
PATCH: RGB = (40, 36, 36)

type Top = tuple[Point, Point, Point, Point]


def _awning(p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float) -> None:
    """A red and white striped awning over the wall under the edge from `a` to `b`, `span` of the
    way along it, hanging from `drop` down it, with a dark scalloped hem."""
    t0, t1 = span
    stripes = max(3, round((t1 - t0) * 10))
    for k in range(stripes):
        s0 = t0 + (t1 - t0) * k / stripes
        s1 = t0 + (t1 - t0) * (k + 1) / stripes
        _patch(p, a, b, (s0, s1), (drop, drop + 2.4), AWNING[k % 2])
    p.line([_on_wall(a, b, t0, drop + 2.6), _on_wall(a, b, t1, drop + 2.6)], shade(AWNING[0], 0.6))


def _shop_window(p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float) -> None:
    """A lit shop window of loaves under an awning."""
    _patch(p, a, b, span, (drop, drop + 1.8), (244, 210, 136))
    t0, t1 = span
    for k in range(3):
        p.point(*_on_wall(a, b, t0 + (t1 - t0) * (k + 0.6) / 3.4, drop + 1), LOAF)


def _oven(p: Painter, x: float, y: float, half: float) -> None:
    """A domed brick bread oven on the ground at (x, y), the fire glowing in its dark mouth."""
    dome = shade(BRICK, 0.82)
    p.ellipse(x - half, y - half * 1.2, x + half, y + half * 0.4, dome)
    p.line([(x - half * 0.6, y - half * 0.8), (x - half * 0.1, y - half * 1.1)], shade(dome, 1.3))
    p.blob(x - 1.4, y - 1.8, x + 0.6, y + 0.3, (50, 36, 32))
    p.point(x - 0.6, y - 0.6, EMBERS)


def _loaf_sign(p: Painter, x: float, y: float, size: float = 1.0) -> None:
    """A golden loaf hung from a bracket at (x, y)."""
    p.line([(x, y), (x - 2.5 * size, y)], (60, 56, 54))
    p.line([(x - 2.5 * size, y), (x - 2.5 * size, y + 1)], (60, 56, 54))
    p.ellipse(x - 4.5 * size, y + 1, x - 0.5 * size, y + 1 + 2.6 * size, LOAF)
    p.line([(x - 3.5 * size, y + 2), (x - 1.8 * size, y + 1.6)], shade(LOAF, 1.35))


def _gable_sign(p: Painter, x: float, y: float) -> None:
    """A big golden loaf standing on a roof's ridge over (x, y) on a post."""
    p.line([(x, y), (x, y - 2)], (60, 56, 54))
    p.ellipse(x - 4, y - 5.5, x + 4, y - 1.5, LOAF)
    for dx in (-2, 0, 2):
        p.line([(x + dx - 0.5, y - 4.5), (x + dx + 0.5, y - 3)], shade(LOAF, 1.35))


def _flour(p: Painter, x: float, y: float, sacks: int) -> None:
    """Flour sacks piled at (x, y) with flour spilt round them."""
    p.blob(x - 2.5, y - 0.8, x + 2.5 + sacks, y + 1.2, FLOUR)
    for k in range(sacks):
        _sack(p, x + 1.6 * k, y + 0.5 * (k % 2), (222, 208, 178))


def _bakery_front(p: Painter, top: Top, height: float, span: tuple[float, float]) -> None:
    left, front, _, _ = top
    _plinth(p, top, height, CRUST)
    _awning(p, left, front, span, height - 5.2)
    _shop_window(p, left, front, (span[0] + 0.03, span[1] - 0.03), height - 2.6)
    _door(p, left, front, (span[1] + 0.06, span[1] + 0.22), height - 4, height, (120, 70, 44))


def bakery() -> Sprite:
    """A honey-coloured baker's house under a red roof, a striped awning over its window of loaves,
    a gold loaf hung by the door, a domed bread oven glowing beside it and a tall chimney smoking."""
    p = Painter(26, 34, (12, 27))
    foot = _footing(0, 0, 7, 7)
    _bed(p, foot, margin=1)
    p.shadow(10, 3)
    _track(p, foot[0], foot[1], (0.64, 0.86), 3, (190, 172, 136), ruts=False)
    top = p.box(0, 0, 7, 7, CRUST, None)
    left, front, right, _ = top
    _bakery_front(p, top, 7, (0.06, 0.56))
    _window(p, front, right, (0.3, 0.65), 2, shade(WINDOW, 0.85))
    p.gable(top, 5, BAKERY_ROOF, CRUST)
    _roof_rows(p, top, 5, BAKERY_ROOF)
    _chimney(p, -1, -12, -19, BRICK)
    _smoke(p, -0.5, -21, SMOKE, 3)
    _loaf_sign(p, *_on_wall(left, front, 0.02, 1.5))
    _oven(p, 8, 3, 3)
    _flour(p, -7, 3.5, 2)
    return p.sprite()


def big_bakery() -> Sprite:
    """A long two-storey bakery under a red roof, its whole shop front under a striped awning,
    two oven chimneys smoking, a gold loaf on its ridge and flour sacks piled by the door."""
    p = Painter(36, 44, (17, 34))
    foot = _footing(0, 0, 16, 8)
    _bed(p, foot)
    p.shadow(14, 4)
    _track(p, foot[0], foot[1], (0.72, 0.88), 4, (190, 172, 136), ruts=False)
    top = p.hall(0, 0, 16, 8, 10, CRUST, None)
    left, front, right, _ = top
    _bakery_front(p, top, 10, (0.04, 0.66))
    for t in (0.12, 0.32, 0.52, 0.8):
        _window(p, left, front, (t - 0.05, t + 0.05), 1.6, shade(WINDOW, 1.1))
    _window(p, front, right, (0.3, 0.7), 2, shade(WINDOW, 0.85))
    p.gable(top, 5, BAKERY_ROOF, CRUST)
    _roof_rows(p, top, 5, BAKERY_ROOF)
    for x, foot_y in ((-6, -16), (2, -12)):
        _chimney(p, x, foot_y, foot_y - 8, BRICK)
        _smoke(p, x + 0.5, foot_y - 10, SMOKE, 3)
    _gable_sign(p, -2, -14)
    _oven(p, 12, 5, 3)
    _flour(p, -13, 3, 3)
    return p.sprite()


def mega_bakery() -> Sprite:
    """A great bread works: a long honey-coloured hall with rows of windows and a shop under a
    striped awning at its end, three smoking oven stacks, two white flour silos behind it, a big
    gold loaf over the door and flour sacks stacked out front."""
    p = Painter(48, 52, (24, 40))
    for x, y in ((9, -9), (15, -6)):
        _silo(p, x, y, 16, MILK, BAKERY_ROOF)
    foot = _footing(-2, 0, 22, 10)
    _bed(p, foot)
    p.shadow(19, 5)
    _track(p, foot[0], foot[1], (0.78, 0.92), 5, (190, 172, 136), ruts=False)
    top = p.hall(-2, 0, 22, 10, 10, CRUST, None)
    left, front, right, _ = top
    _plinth(p, top, 10, CRUST)
    for t in (0.08, 0.22, 0.36, 0.5, 0.64):
        _window(p, left, front, (t - 0.04, t + 0.04), 1.6, shade(WINDOW, 1.1))
        _window(p, left, front, (t - 0.04, t + 0.04), 5, shade(WINDOW, 1.1))
    _awning(p, left, front, (0.72, 0.98), 4.8)
    _shop_window(p, left, front, (0.74, 0.96), 7.4)
    _door(p, front, right, (0.3, 0.6), 5, 10, (120, 70, 44))
    p.gable(top, 6, BAKERY_ROOF, CRUST)
    _roof_rows(p, top, 6, BAKERY_ROOF, (0.25, 0.5, 0.75))
    for x, foot_y in ((-14, -20), (-6, -16), (2, -12)):
        _chimney(p, x, foot_y, foot_y - 9, BRICK)
        _smoke(p, x + 0.5, foot_y - 11, SMOKE, 3)
    _gable_sign(p, 6, -10)
    _flour(p, -20, 3, 4)
    _flour(p, 6, 10, 3)
    return p.sprite()


def _silo(p: Painter, x: float, y: float, tall: float, wall: RGB, band: RGB) -> None:
    """A round silo or tank standing on (x, y) under a domed cap, banded in `band`."""
    p.rect(x - 2.5, y - tall, x + 2.5, y, wall)
    p.line([(x - 1.5, y - 0.5), (x - 1.5, y - tall + 0.5)], shade(wall, 1.1))
    p.line([(x + 1.5, y - 0.5), (x + 1.5, y - tall + 0.5)], shade(wall, 0.82))
    p.ellipse(x - 2.5, y - tall - 1.5, x + 2.5, y - tall + 1.2, shade(wall, 0.92))
    for k in (0.3, 0.7):
        _ring(p, x, y - tall * k, 2.5, band)


def _churns(p: Painter, x: float, y: float, count: int) -> None:
    """A row of milk churns standing on the ground from (x, y)."""
    for k in range(count):
        cx, cy = x + 2.2 * k, y + 0.6 * (k % 2)
        p.rect(cx - 1, cy - 3.2, cx + 1, cy, CHURN)
        p.rect(cx - 0.6, cy - 4.2, cx + 0.6, cy - 3.2, shade(CHURN, 0.9))
        p.line([(cx - 1, cy - 1.6), (cx + 1, cy - 1.6)], shade(CHURN, 0.7))
        p.point(cx - 0.4, cy - 2.6, shade(CHURN, 1.2))


def _cow_yard(p: Painter, x: float, y: float, half: float, cows: tuple[Point, ...]) -> None:
    """A fenced yard of trodden grass centred at (x, y) with black-and-white cows in it."""
    left, back, right, near = _paddock(x, y, half)
    p.polygon([left, back, right, near], (118, 150, 70))
    _fence(p, left, back)
    _fence(p, back, right)
    for cx, cy in cows:
        _cow(p, cx, cy, HIDE, PATCH)
    _fence(p, left, near)
    _fence(p, near, right)


def _dairy_trim(p: Painter, top: Top, height: float) -> None:
    """Blue trim round a whitewashed dairy: a band under its eaves and a blue plinth."""
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (0, 0.9), DAIRY_BLUE)
    _patch(p, front, right, (0, 1), (0, 0.9), shade(DAIRY_BLUE, 0.78))
    _patch(p, left, front, (0, 1), (height - 1, height), shade(DAIRY_BLUE, 0.9))
    _patch(p, front, right, (0, 1), (height - 1, height), shade(DAIRY_BLUE, 0.7))


def _milk_sign(p: Painter, a: Point, b: Point, t: float, drop: float, half: float = 0.1) -> None:
    """A blue board with a white drop of milk on it, on the wall under the edge from `a` to `b`,
    `half` of the way along it either side of `t`."""
    _patch(p, a, b, (t - half, t + half), (drop, drop + 2.6), DAIRY_BLUE)
    x, y = _on_wall(a, b, t, drop + 1.4)
    p.blob(x - 0.8, y - 0.6, x + 0.8, y + 0.9, MILK)
    p.point(x, y - 1, MILK)


def dairy() -> Sprite:
    """A whitewashed dairy under a blue roof with blue trim and door, a blue sign with a drop of
    milk, a row of milk churns by its door and two cows in a fenced yard beside it."""
    p = Painter(32, 26, (17, 19))
    _cow_yard(p, -9, 1, 6.5, ((-9, 0), (-6, 2.5)))
    foot = _footing(4, -1, 6, 6)
    _bed(p, foot, margin=1)
    p.shadow(9, 2.5, 2)
    top = p.box(4, -1, 6, 6, WHITEWASH, None)
    left, front, right, _ = top
    _dairy_trim(p, top, 6)
    _milk_sign(p, left, front, 0.32, 1.8, 0.08)
    _door(p, left, front, (0.64, 0.86), 2, 6, DAIRY_BLUE)
    _window(p, front, right, (0.3, 0.65), 1.8, shade(WINDOW, 0.85))
    p.gable(top, 4, DAIRY_BLUE, WHITEWASH)
    _roof_rows(p, top, 4, DAIRY_BLUE)
    _churns(p, 6, 5, 3)
    return p.sprite()


def big_dairy() -> Sprite:
    """A long whitewashed dairy hall under a blue roof with blue trim and doors, a white milk tank
    banded in blue behind it, churns lined up out front and three cows in a fenced yard."""
    p = Painter(40, 36, (21, 26))
    _silo(p, 10, -6, 13, MILK, DAIRY_BLUE)
    _cow_yard(p, -11, 2, 8, ((-13, 1), (-9, -1), (-9, 4)))
    foot = _footing(4, 0, 14, 7)
    _bed(p, foot)
    p.shadow(12, 3.5)
    top = p.hall(4, 0, 14, 7, 8, WHITEWASH, None)
    left, front, right, _ = top
    _dairy_trim(p, top, 8)
    for t in (0.12, 0.3):
        _window(p, left, front, (t - 0.05, t + 0.05), 2.4, shade(WINDOW, 1.1))
    _milk_sign(p, left, front, 0.52, 2)
    _door(p, left, front, (0.72, 0.9), 3, 8, DAIRY_BLUE)
    _door(p, front, right, (0.3, 0.7), 3, 8, DAIRY_BLUE)
    p.gable(top, 5, DAIRY_BLUE, WHITEWASH)
    _roof_rows(p, top, 5, DAIRY_BLUE)
    _churns(p, 4, 9, 4)
    return p.sprite()


def mega_dairy() -> Sprite:
    """A great creamery: a long whitewashed hall under a blue roof with rows of windows, a big blue
    sign with a drop of milk, three tall white milk tanks banded in blue, churns stacked on the
    loading ramp and cows in a fenced yard."""
    p = Painter(48, 46, (24, 34))
    for x, y in ((6, -11), (11, -9), (16, -6)):
        _silo(p, x, y, 20, MILK, DAIRY_BLUE)
    _cow_yard(p, -14, 4, 8, ((-16, 3), (-12, 1), (-12, 6)))
    foot = _footing(2, 0, 20, 9)
    _bed(p, foot)
    p.shadow(17, 4.5)
    top = p.hall(2, 0, 20, 9, 10, WHITEWASH, None)
    left, front, right, _ = top
    _dairy_trim(p, top, 10)
    for t in (0.08, 0.2, 0.62, 0.74, 0.88):
        _window(p, left, front, (t - 0.04, t + 0.04), 2, shade(WINDOW, 1.1))
        _window(p, left, front, (t - 0.04, t + 0.04), 5.6, shade(WINDOW, 1.1))
    _milk_sign(p, left, front, 0.41, 2.2)
    _patch(p, left, front, (0.32, 0.5), (5.4, 10), DAIRY_BLUE)
    _door(p, front, right, (0.25, 0.75), 4, 10, DAIRY_BLUE)
    p.gable(top, 5, DAIRY_BLUE, WHITEWASH)
    _roof_rows(p, top, 5, DAIRY_BLUE, (0.25, 0.5, 0.75))
    _churns(p, 0, 11, 5)
    return p.sprite()
