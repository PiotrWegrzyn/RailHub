"""Mines, wells, pits, quarries and lumber mills."""

from railroads.ui.pixel.art.painter import RGB, TIMBER, TRUNK, Painter, Sprite, shade
from railroads.ui.pixel.art.parts import _heap, _on_wall, _patch


def mine() -> Sprite:
    p = Painter(28, 30, (12, 24))
    p.shadow(11, 3)
    heap = (52, 48, 50)
    p.polygon([(-11, 2), (-6, -5), (-1, -3), (1, 3)], heap, shade(heap, 0.6))
    p.polygon([(-6, -5), (-4, -4), (-8, 0)], shade(heap, 1.4))
    top = p.box(6, 1, 5, 4, (150, 132, 112), None)
    p.gable(top, 2, (92, 72, 62), (150, 132, 112))
    frame = (74, 64, 58)
    p.line([(0, -1), (3, -20), (6, -1)], frame)
    p.line([(1, -8), (5, -8)], frame)
    p.line([(1, -14), (5, -14)], frame)
    p.ellipse(1, -23, 5, -19, (96, 86, 78))
    return p.sprite()


def lumber_mill() -> Sprite:
    p = Painter(28, 24, (13, 19))
    p.shadow(11, 3)
    top = p.box(-3, 0, 7, 6, TIMBER, None)
    p.gable(top, 4, (96, 84, 70), TIMBER)
    log = (138, 96, 58)
    for k in range(3):
        y = 2 - 2 * k
        x = 5 + k
        p.rect(x, y - 2, x + 7, y, log)
        p.point(x + 7, y - 1, (214, 176, 122))
    p.point(6, -6, (60, 44, 30))
    return p.sprite()


def shaft_mine(heap: RGB, glint: RGB) -> Sprite:
    """A pithead over a heap of what it digs."""
    p = Painter(28, 30, (12, 24))
    p.shadow(11, 3)
    p.polygon([(-11, 2), (-6, -5), (-1, -3), (1, 3)], heap, shade(heap, 0.6))
    for x, y in ((-7, -2), (-4, -3), (-5, 0), (-2, 0)):
        p.point(x, y, glint)
    top = p.box(6, 1, 5, 4, (150, 132, 112), None)
    p.gable(top, 2, (92, 72, 62), (150, 132, 112))
    frame = (74, 64, 58)
    p.line([(0, -1), (3, -20), (6, -1)], frame)
    p.line([(1, -8), (5, -8)], frame)
    p.line([(1, -14), (5, -14)], frame)
    p.ellipse(1, -23, 5, -19, glint)
    return p.sprite()


def oil_well() -> Sprite:
    """A timber derrick beside a storage tank."""
    p = Painter(28, 36, (12, 30))
    p.shadow(10, 3)
    frame = (70, 58, 48)
    p.line([(-7, 1), (-2, -26), (3, 1)], frame)
    for y in (-6, -12, -18):
        half = 4 + y / 6
        p.line([(-2 - half, y), (-2 + half, y)], frame)
    p.ellipse(5, -6, 13, -1, (40, 38, 44))
    p.rect(5, -4, 13, 2, (52, 50, 58))
    p.ellipse(5, -1, 13, 3, (40, 38, 44))
    return p.sprite()


def salt_works() -> Sprite:
    """White heaps of salt beside a low shed."""
    p = Painter(30, 22, (15, 16))
    p.shadow(12, 3)
    white = (232, 234, 238)
    p.polygon([(-13, 3), (-8, -6), (-3, 3)], white, shade(white, 0.7))
    p.polygon([(-5, 4), (-1, -3), (3, 4)], white, shade(white, 0.7))
    top = p.box(8, 1, 5, 4, (170, 152, 124), None)
    p.gable(top, 2, (112, 92, 74), (170, 152, 124))
    return p.sprite()


def _pit(p: Painter, x: float, y: float, half: float, ground: RGB, floor: RGB) -> None:
    """A hole dug into the ground at (x, y), ringed with spoil, showing its far walls and floor."""
    w, h = half, half / 2
    p.polygon(
        [(x - w - 3, y), (x, y + h + 1.5), (x + w + 3, y), (x, y - h - 1.5)], shade(ground, 1.12)
    )
    low = (x - w + 3, y + 1.5), (x, y - h + 3), (x + w - 3, y + 1.5)
    p.polygon([(x - w, y), (x, y - h), low[1], low[0]], shade(ground, 0.72))
    p.polygon([(x, y - h), (x + w, y), low[2], low[1]], ground)
    p.polygon([low[0], low[1], low[2], (x, y + h)], floor)


def clay_pit() -> Sprite:
    """An open pit dug into orange clay, with a heap of it, a tool shed and a loaded wheelbarrow."""
    p = Painter(32, 26, (16, 16))
    clay = (190, 108, 64)
    _heap(p, -9, -3, 5, 6, shade(clay, 1.05))
    _pit(p, -2, 2, 8, clay, shade(clay, 0.55))
    top = p.box(10, -1, 4, 4, (164, 150, 128), None)
    p.gable(top, 3, (96, 84, 76), (164, 150, 128))
    _patch(p, top[0], top[1], (0.5, 0.8), (1, 4), (70, 56, 46))
    barrow = (92, 118, 150)
    x, y = 6, 8
    p.line([(x - 3, y - 2), (x, y - 3)], TRUNK)
    p.polygon(
        [(x - 1, y - 4), (x + 5, y - 4), (x + 4, y - 1), (x, y - 1)], barrow, shade(barrow, 0.55)
    )
    p.line([(x, y - 4), (x + 4, y - 4)], shade(clay, 1.1))
    p.point(x + 2, y - 5, clay)
    p.rect(x + 4, y - 1, x + 5, y, (40, 38, 40))
    return p.sprite()


def limestone_quarry() -> Sprite:
    """A pale limestone face cut in steps into a grassy hill, with sawn blocks waiting below."""
    p = Painter(34, 34, (17, 24))
    p.shadow(13, 4)
    stone = (222, 218, 202)
    terrace = shade(stone, 0.9)
    for (x, y, h), top in (
        ((3, -6, 11), (118, 148, 80)),
        ((-1, -4, 7), terrace),
        ((-5, -2, 3), terrace),
    ):
        left, front, *_ = p.hall(x, y, 16, 5, h, stone, top)
        for drop in range(3, h, 3):
            p.line(
                [_on_wall(left, front, 0.05, drop), _on_wall(left, front, 0.95, drop)],
                shade(stone, 0.8),
            )
    for x, y, h in ((6, 5, 3), (10, 3, 3), (10, 0, 2.5)):
        p.box(x, y, 2.5, h, stone, shade(stone, 1.08))
    p.line([(8, -13), (8, -23)], TRUNK)
    p.line([(8, -22), (0, -17)], TRUNK)
    p.line([(0, -17), (0, -12)], (60, 56, 54))
    p.rect(-1, -12, 1, -10, stone)
    return p.sprite()


def sand_pit() -> Sprite:
    """Pale yellow heaps of sand beside a pit with water in its bottom, fed by a belt conveyor."""
    p = Painter(34, 28, (17, 18))
    sand = (230, 206, 140)
    _pit(p, -5, 2, 8, sand, (92, 148, 196))
    p.line([(-7, 3), (-3, 3)], (150, 196, 226))
    _heap(p, 9, -1, 5, 8, sand)
    _heap(p, 6, 6, 4, 5, shade(sand, 1.04))
    belt = (70, 70, 78)
    p.line([(-1, -1), (7, -9)], belt)
    p.line([(-1, 0), (7, -8)], shade(belt, 1.4))
    for x, y in ((1, -2), (4, -5)):
        p.line([(x, y), (x, y + 4)], shade(belt, 1.2))
        p.point(x + 1, y - 1, sand)
    return p.sprite()
