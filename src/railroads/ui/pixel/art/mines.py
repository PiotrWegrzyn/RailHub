"""Mines, wells, pits, quarries and lumber mills."""

import itertools

from railroads.ui.pixel.art.farms import (
    DOOR,
    EARTH,
    FOOTING,
    SACK,
    WORN,
    _boards,
    _chimney,
    _door,
    _footing,
    _mud,
    _plinth,
    _ruts,
    _sack,
    _strewn,
    _tiles,
    _track,
    _window,
    _worn,
)
from railroads.ui.pixel.art.painter import (
    BARREL,
    DOORWAY,
    METAL,
    RGB,
    TIMBER,
    TRUNK,
    WINDOW,
    Painter,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import _heap, _on_wall, _patch, _turned

RAIL: RGB = (138, 136, 144)
SLEEPER: RGB = (92, 70, 50)


def _rails(p: Painter, x0: float, y0: float, x1: float, y1: float, sleepers: int = 4) -> None:
    """A narrow track of two rails on sleepers running along the ground from (x0, y0) to
    (x1, y1), its second rail 1.5 across to the right."""
    for k in range(sleepers):
        t = (k + 0.5) / sleepers
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        p.line([(x - 0.5, y - 0.25), (x + 2, y + 1)], SLEEPER)
    p.line([(x0, y0), (x1, y1)], RAIL)
    p.line([(x0 + 1.5, y0 + 0.75), (x1 + 1.5, y1 + 0.75)], RAIL)


def _tub(p: Painter, x: float, y: float, load: RGB, glint: RGB | None = None) -> None:
    """A small iron mine tub standing on its wheels at (x, y), heaped with `load`."""
    tub = (128, 98, 80)
    p.point(x - 1.2, y, (36, 34, 36))
    p.point(x + 0.6, y + 0.6, (36, 34, 36))
    p.box(x, y - 0.5, 1.8, 1.6, tub, None)
    p.blob(x - 1.6, y - 3.3, x + 1.6, y - 1.6, load)
    if glint is not None:
        p.point(x - 0.5, y - 3, glint)


def _barrow(p: Painter, x: float, y: float, body: RGB, load: RGB) -> None:
    """A wheelbarrow at (x, y) with its handles to the left, loaded with `load`."""
    p.line([(x - 3, y - 2), (x, y - 3)], TRUNK)
    p.polygon([(x - 1, y - 4), (x + 5, y - 4), (x + 4, y - 1), (x, y - 1)], body, shade(body, 0.55))
    p.line([(x, y - 4), (x + 4, y - 4)], shade(load, 1.1))
    p.point(x + 2, y - 5, load)
    p.rect(x + 4, y - 1, x + 5, y, (40, 38, 40))


def _spade(p: Painter, x: float, y: float) -> None:
    """A spade stuck upright in the ground at (x, y)."""
    p.line([(x, y), (x, y - 4)], TRUNK)
    p.line([(x - 0.5, y - 4), (x + 0.5, y - 4)], TRUNK)
    p.rect(x - 0.5, y - 0.5, x + 0.5, y + 1, METAL)


def _pithead(p: Painter, heap: RGB, glint: RGB, wheel: RGB) -> None:
    """A pithead: a spoil heap of what it digs, a tub on rails between it and the shaft, a
    winding house with a chimney and a braced headframe with its spoked wheel and cable."""
    dust = shade(heap, 1.5)
    _track(p, [(7.8, 3.4), (8.8, 5), (9.8, 7)], 1, dust)
    _worn(p, 8, 4, 2.6, dust)
    p.blob(-11.8, -0.2, 2, 5.4, shade(heap, 1.15))
    _strewn(p, -7, 4.2, 5.4, heap, 7)
    p.shadow(11, 3)
    _heap(p, -5.4, 2.4, 5.8, 7.8, heap)
    p.line([(-6.4, -5), (-9.4, 0.4)], shade(heap, 1.35))
    p.line([(-4.4, -4.6), (-2, -0.6)], shade(heap, 0.62))
    for x, y in ((-7, -2), (-5, -3.4), (-5, 0), (-2.4, 0.6), (-8.6, 1.4)):
        p.point(x, y, glint)
    p.polygon([(0, -1), (3, -2.5), (6, -1), (3, 0.5)], (40, 36, 38), shade(TIMBER, 0.7))
    _rails(p, 1.5, 0.5, -6, 4.2)
    _tub(p, -2.5, 3.5, heap, glint)
    wall = (150, 132, 112)
    top = p.box(6, 1, 5, 4, wall, None)
    left, front, right, _ = top
    _footing(p, top, 4)
    _plinth(p, top, wall, 4, 0.8)
    _window(p, left, front, (0.55, 0.85), 1)
    _door(p, front, right, (0.2, 0.45), 1.3, 4, DOOR)
    _window(p, front, right, (0.62, 0.85), 1, shade(WINDOW, 0.8))
    p.gable(top, 2, (92, 72, 62), wall)
    _tiles(p, top, 2, (92, 72, 62), (0.5,))
    _chimney(p, top, 2, 0.85, 1.5)
    frame = (74, 64, 58)
    p.line([(0, -1), (3, -20), (6, -1)], frame)
    p.line([(1.1, -8), (4.9, -8)], frame)
    p.line([(2.1, -14), (3.9, -14)], frame)
    p.line([(0.4, -3), (4.9, -8)], shade(frame, 1.3))
    p.line([(5.6, -3), (1.1, -8)], shade(frame, 1.3))
    p.line([(4, -18), (9, -6)], frame)
    p.line([(5, -21), (8.5, -6)], (130, 130, 136))
    p.ellipse(1, -23, 5, -19, wheel)
    spoke = shade(wheel, 0.6)
    p.line([(1.6, -21), (4.4, -21)], spoke)
    p.line([(3, -22.4), (3, -19.6)], spoke)
    p.point(3, -21, shade(wheel, 1.3))


def mine() -> Sprite:
    """A coal mine: a pithead over a black heap of coal."""
    p = Painter(28, 30, (12, 24))
    _pithead(p, (52, 48, 50), (104, 100, 104), (96, 86, 78))
    return p.sprite()


def shaft_mine(heap: RGB, glint: RGB) -> Sprite:
    """A pithead over a heap of what it digs, glinting with its ore."""
    p = Painter(28, 30, (12, 24))
    _pithead(p, heap, glint, glint)
    return p.sprite()


def lumber_mill() -> Sprite:
    """A boarded steam sawmill with a door, a window and a chimney, its circular saw
    showing through the open side, with a stack of logs cut to show their rings, a heap of
    sawdust and a pile of sawn planks."""
    p = Painter(28, 24, (13, 19))
    sawdust = (222, 194, 140)
    _ruts(p, [(5, 3.2), (9, 4.4), (13, 5.4)])
    _track(p, [(-5.2, 3), (-6, 4.6), (-5.4, 6)])
    _worn(p, -5.2, 3, 2.2)
    p.blob(-2.4, 1.8, 6.4, 6, WORN)
    p.blob(-1, 2.4, 5.4, 5.4, shade(sawdust, 0.92))
    p.blob(2.6, 0.2, 14.4, 3.4, FOOTING)
    p.blob(-11.4, 1.6, -3.6, 5, FOOTING)
    p.shadow(11, 3)
    roof = (96, 84, 70)
    top = p.box(-3, 0, 7, 6, TIMBER, None)
    left, front, right, _ = top
    _footing(p, top, 6)
    _boards(p, top, TIMBER, (1.5, 3, 4.5))
    _plinth(p, top, TIMBER, 6)
    _window(p, left, front, (0.12, 0.3), 1.2)
    _door(p, left, front, (0.58, 0.78), 1.8, 6)
    _patch(p, front, right, (0.12, 0.72), (1.4, 6), DOORWAY)
    bx, by = _on_wall(front, right, 0.42, 3.6)
    p.ellipse(bx - 1.8, by - 1.8, bx + 1.8, by + 1.8, METAL)
    p.point(bx - 1, by - 1, shade(METAL, 1.3))
    p.point(bx, by, shade(METAL, 0.45))
    p.line([_on_wall(front, right, 0.12, 5), _on_wall(front, right, 0.72, 5)], (138, 96, 58))
    p.gable(top, 4, roof, TIMBER)
    _tiles(p, top, 4, roof)
    _chimney(p, top, 4, 0.2, 2.5)
    log, end = (138, 96, 58), (214, 176, 122)
    for x, y in ((6, -2), (5, 0), (5.5, -2.2), (4, 2), (4.5, -0.2)):
        p.rect(x, y - 2, x + 7, y, log)
        p.line([(x + 0.5, y - 1), (x + 6, y - 1)], shade(log, 0.8))
        p.ellipse(x + 6, y - 2.2, x + 8.2, y + 0.2, end)
        p.point(x + 7, y - 1, shade(end, 0.7))
    plank = (200, 166, 116)
    p.hall(-8, 2.4, 5, 1.6, 1.4, plank, shade(plank, 1.1))
    p.line([(-11, 1.9), (-6, 4.4)], shade(plank, 0.7))
    _heap(p, 1.5, 4.5, 2, 1.8, sawdust)
    _strewn(p, 1.4, 3.6, 4.4, shade(sawdust, 1.06), 7)
    return p.sprite()


def oil_well() -> Sprite:
    """A timber derrick on a plank floor with braces, a crown block and a pipe to a riveted
    storage tank with a ladder, and barrels of oil beside it."""
    p = Painter(28, 36, (12, 30))
    p.shadow(10, 3)
    frame = (70, 58, 48)
    floor = (122, 96, 66)
    oil = (44, 42, 46)
    _track(p, [(-2, 3.6), (-1.6, 5), (-0.6, 7)], 1)
    _worn(p, -9.2, 4.2, 3, shade(EARTH, 0.8))
    p.blob(-10, -1.4, 6, 5.2, shade(EARTH, 0.72))
    p.blob(-7, -0.4, 2.6, 4.6, (70, 64, 58))
    p.blob(2, 1.4, 5.6, 3.2, (70, 64, 58))
    p.blob(4.2, -1.4, 13.8, 3.6, (150, 146, 136))
    p.blob(4.6, -0.6, 13.4, 3, shade((150, 146, 136), 0.8))
    p.polygon([(-8, 1), (-2, 4), (4, 1), (-2, -2)], floor, shade(floor, 0.6))
    p.line([(-5, 2.5), (1, -0.5)], shade(floor, 0.8))
    p.blob(-4, 0, 0, 2, oil)
    p.point(-3, 0.6, (120, 116, 128))
    for x, y in ((-5.4, 2.2), (0.6, 2.4), (-1, -0.6)):
        p.point(x, y, shade(oil, 1.3))
    p.line([(-2, -2), (-2, -26)], shade(frame, 0.7))
    p.line([(-7, 1), (-2, -26), (3, 1)], frame)
    rungs = (1, -6, -12, -18)
    for y0, y1 in itertools.pairwise(rungs):
        h0, h1 = 5 * (26 + y0) / 27, 5 * (26 + y1) / 27
        if y0 < 1:
            p.line([(-2 - h0, y0), (-2 + h0, y0)], frame)
        p.line([(-2 - h0, y0), (-2 + h1, y1)], shade(frame, 1.3))
        p.line([(-2 + h0, y0), (-2 - h1, y1)], shade(frame, 1.3))
    p.line([(-2 - 1.5, -18), (-2 + 1.5, -18)], frame)
    p.rect(-3, -28, -1, -26, shade(frame, 1.2))
    p.line([(0, 2), (6, 1.5)], (36, 34, 40))
    tank = (52, 50, 58)
    _turned(p, 9, 1, [(0, 4), (7, 4)], tank, (40, 38, 44))
    for h in (2.5, 5):
        p.line([(5, 1 - h), (13, 1 - h)], shade(tank, 1.35))
    p.line([(6, 2), (6, -6)], METAL)
    for y in (0, -2, -4):
        p.point(5, y, METAL)
    for x, y in ((-10, 3), (-8, 5)):
        p.ellipse(x - 1.4, y - 3.5, x + 1.4, y, BARREL)
        p.line([(x - 1.4, y - 1.5), (x + 1.4, y - 1.5)], shade(BARREL, 0.55))
    return p.sprite()


def salt_works() -> Sprite:
    """White heaps of salt raked up on a pan of drying brine, beside a low shed with a door, a
    window and a tiled roof, and sacks of salt."""
    p = Painter(30, 22, (15, 16))
    p.shadow(12, 3)
    white, brine = (232, 234, 238), (176, 206, 222)
    bank = (176, 160, 128)
    _track(p, [(5.8, 3.4), (2.4, 4.8), (-1, 5), (-3, 6.5)], 1, shade(bank, 1.08))
    _worn(p, 5.6, 3.8, 2.4, shade(bank, 1.08))
    p.polygon([(-15, -1), (-4, 4.6), (7, -1), (-4, -6.6)], bank, shade(bank, 0.8))
    p.polygon([(-14, -1), (-4, 4), (6, -1), (-4, -6)], brine, shade(brine, 0.7))
    for k in range(1, 4):
        p.line([(-14 + 2.5 * k, -1 - 1.25 * k), (-4 + 2.5 * k, 4 - 1.25 * k)], shade(brine, 0.85))
    for x, y in ((-10, -2), (1, -3), (-2, 1)):
        p.point(x, y, white)
    _heap(p, -8, 2, 5, 8, white)
    _heap(p, -1, 3, 4, 6, white)
    for x, y in ((-10, -1), (-8, 1), (-2, 1)):
        p.point(x, y, shade(white, 0.86))
    _strewn(p, -4.4, 3.4, 4.4, white, 6)
    wall, roof = (170, 152, 124), (112, 92, 74)
    top = p.box(8, 1, 5, 4, wall, None)
    left, front, right, _ = top
    _footing(p, top, 4)
    _plinth(p, top, wall, 4, 0.8)
    _door(p, left, front, (0.55, 0.78), 1.3, 4)
    _window(p, left, front, (0.15, 0.38), 1)
    _window(p, front, right, (0.35, 0.65), 1, shade(WINDOW, 0.8))
    p.gable(top, 2, roof, wall)
    _tiles(p, top, 2, roof, (0.5,))
    _sack(p, 13.5, 4.5, shade(white, 0.92), 3, 1.2)
    _sack(p, 3, 5, shade(SACK, 1.1), 3, 1.2)
    return p.sprite()


def _pit(p: Painter, x: float, y: float, half: float, ground: RGB, floor: RGB) -> None:
    """A hole dug into the ground at (x, y), ringed with spoil, showing its far walls, dug in
    steps, and its floor."""
    w, h = half, half / 2
    p.polygon(
        [(x - w - 3, y), (x, y + h + 1.5), (x + w + 3, y), (x, y - h - 1.5)], shade(ground, 1.12)
    )
    low = (x - w + 3, y + 1.5), (x, y - h + 3), (x + w - 3, y + 1.5)
    p.polygon([(x - w, y), (x, y - h), low[1], low[0]], shade(ground, 0.72))
    p.polygon([(x, y - h), (x + w, y), low[2], low[1]], ground)
    for f in (0.35, 0.7):
        mid = (x, y - h + 3 * f)
        p.line([(x - w + 3 * f, y + 1.5 * f), mid], shade(ground, 0.6))
        p.line([mid, (x + w - 3 * f, y + 1.5 * f)], shade(ground, 0.82))
    p.polygon([low[0], low[1], low[2], (x, y + h)], floor)


def clay_pit() -> Sprite:
    """An open pit dug in steps into orange clay with a puddle and a ladder in it, a heap of clay
    with a spade in it, a tool shed with a door and a window, and a loaded wheelbarrow."""
    p = Painter(32, 26, (16, 16))
    clay = (190, 108, 64)
    _track(p, [(8.6, 1.4), (7.4, 3.2), (6, 4)], 1)
    _worn(p, 8.6, 1.6, 2.2)
    _ruts(p, [(5.4, 5), (8, 6.6), (10.6, 8.4)], shade(clay, 0.85))
    _mud(p, 5, 5.2, 2.2)
    p.blob(-14.6, -3.6, -3, 0.8, shade(clay, 0.9))
    _strewn(p, -9, -1.2, 5.6, clay, 7)
    _heap(p, -9, -3, 5, 6, shade(clay, 1.05))
    _spade(p, -6, -4)
    _pit(p, -2, 2, 8, clay, shade(clay, 0.55))
    p.blob(-4, 3, 0, 4.5, (120, 120, 112))
    p.line([(-3.5, 3.4), (-1.5, 3.4)], (170, 176, 170))
    ladder = (150, 112, 74)
    p.line([(3, -2), (2, 3)], ladder)
    p.line([(4.5, -1.5), (3.5, 3.5)], ladder)
    for t in (0.25, 0.6):
        p.line([(3 - t, -2 + 5 * t), (4.5 - t, -1.5 + 5 * t)], ladder)
    wall = (164, 150, 128)
    top = p.box(10, -1, 4, 4, wall, None)
    left, front, right, _ = top
    _footing(p, top, 4)
    _plinth(p, top, wall, 4, 0.8)
    _window(p, left, front, (0.15, 0.4), 1)
    _window(p, front, right, (0.2, 0.42), 1, shade(WINDOW, 0.8))
    p.gable(top, 3, (96, 84, 76), wall)
    _tiles(p, top, 3, (96, 84, 76), (0.5,))
    _door(p, left, front, (0.55, 0.85), 1.3, 4, (70, 56, 46))
    _barrow(p, 6, 8, (92, 118, 150), clay)
    return p.sprite()


def limestone_quarry() -> Sprite:
    """A pale limestone face cut in steps into a grassy hill, its courses split into blocks, with
    a derrick crane lowering a block, and sawn blocks waiting below beside a wagon."""
    p = Painter(34, 34, (17, 24))
    stone = (222, 218, 202)
    dust = (200, 196, 176)
    p.blob(-15, -0.4, 15, 9, WORN)
    p.blob(-12.4, 0.8, 12.6, 8, dust)
    _ruts(p, [(-9, 5.4), (-12, 7.4), (-15.6, 9.4)], dust)
    p.shadow(13, 4)
    terrace = shade(stone, 0.9)
    for (x, y, h), top in (
        ((3, -6, 11), (118, 148, 80)),
        ((-1, -4, 7), terrace),
        ((-5, -2, 3), terrace),
    ):
        face = p.hall(x, y, 16, 5, h, stone, top)
        left, front, *_ = face
        if h == 3:
            _footing(p, face, h, shade(dust, 0.62))
        joint = shade(stone, 0.8)
        for drop in range(3, h, 3):
            p.line([_on_wall(left, front, 0.05, drop), _on_wall(left, front, 0.95, drop)], joint)
        for row, drop in enumerate(range(0, h, 3)):
            for t in (0.2, 0.45, 0.7) if row % 2 else (0.32, 0.58, 0.84):
                bottom = min(drop + 3, h)
                p.line(
                    [_on_wall(left, front, t, drop + 0.5), _on_wall(left, front, t, bottom)], joint
                )
    for x, y in ((-12, 0), (-9, 2.5), (1, 5)):
        p.point(x, y, shade(stone, 0.75))
    _strewn(p, -1, 4.4, 7, stone, 7)
    for x, y, h in ((10, 0, 2.5), (10, 3, 3), (6, 5, 3)):
        block = p.box(x, y, 2.5, h, stone, shade(stone, 1.08))
        _footing(p, block, h, shade(dust, 0.62))
    p.blob(-12, 3.4, -4, 7.2, shade(dust, 0.7))
    p.hall(-8, 4, 5, 2.4, 1, TIMBER, shade(TIMBER, 1.15))
    p.box(-8, 3, 1.6, 1.8, stone, shade(stone, 1.08))
    for x, y in ((-10.5, 4.6), (-6.5, 6.6)):
        p.ellipse(x - 1, y - 1, x + 1, y + 1, (66, 60, 56))
    p.line([(-11.7, 3.4), (-14, 2.2)], TRUNK)
    p.line([(8, -12), (8, -23)], TRUNK)
    p.line([(8, -23), (12, -13)], (110, 104, 96))
    p.line([(8, -22), (0, -17)], TRUNK)
    p.line([(8, -15), (2, -17.2)], shade(TRUNK, 1.2))
    p.rect(7, -14, 9, -12, shade(TRUNK, 1.2))
    p.line([(0, -17), (0, -12)], (60, 56, 54))
    p.rect(-1, -12, 1, -10, stone)
    return p.sprite()


def sand_pit() -> Sprite:
    """Pale yellow heaps of sand beside a pit dug in steps with rippled water in its bottom, fed
    by a belt conveyor on legs, with a sieve, a spade and a loaded wheelbarrow."""
    p = Painter(34, 28, (17, 18))
    sand = (230, 206, 140)
    water = (92, 148, 196)
    _pit(p, -5, 2, 8, sand, water)
    p.line([(-7, 3), (-3, 3)], (150, 196, 226))
    p.line([(-5, 4.5), (-3, 4.5)], (150, 196, 226))
    _ruts(p, [(3, 5.6), (8, 8.4), (13.6, 10.6)], shade(sand, 0.92))
    p.blob(2.6, -3, 16, 3, shade(sand, 0.9))
    p.blob(0.4, 4.2, 12, 9, shade(sand, 0.9))
    _strewn(p, 9, 1.8, 6.6, sand, 7)
    _heap(p, 9, -1, 5, 8, sand)
    _spade(p, 12, -1)
    _strewn(p, 5.6, 7.4, 5, shade(sand, 1.04), 6)
    _heap(p, 6, 6, 4, 5, shade(sand, 1.04))
    belt = (70, 70, 78)
    for x, y in ((1, -2), (4, -5)):
        p.blob(x - 0.8, y + 3.4, x + 2.2, y + 4.6, FOOTING)
        p.line([(x, y), (x, y + 4)], shade(belt, 1.2))
        p.line([(x + 1, y - 1), (x + 1, y + 3)], shade(belt, 0.9))
    p.line([(-1, -1), (7, -9)], belt)
    p.line([(-1, 0), (7, -8)], shade(belt, 1.4))
    for x, y in ((0, -2), (2.5, -4.5), (5, -7)):
        p.point(x, y - 0.5, sand)
    p.rect(-2, -1, -1, 1, shade(belt, 1.2))
    _heap(p, -12.2, 4.6, 2.4, 1.6, shade(sand, 1.06))
    sieve = (150, 112, 74)
    p.polygon([(-15, 3), (-12, -2), (-10, -1), (-13, 4)], shade(sieve, 0.6), sieve)
    for k in range(1, 3):
        p.line([(-15 + k, 3 + 0.5 * k), (-12 + k, -2 + 0.5 * k)], shade(METAL, 0.9))
    _barrow(p, 11, 9, (92, 118, 150), sand)
    return p.sprite()
