"""Churches, basilicas, cathedrals and St. Paul's."""

import math

import numpy as np

from railroads.ui.pixel.art.homes import _foot, _worn
from railroads.ui.pixel.art.painter import (
    CHURCH_ROOF,
    CHURCH_WALL,
    CROSS,
    DOORWAY,
    LEAD,
    RGB,
    SLATE,
    STAINED,
    VERDIGRIS,
    WHITEWASH,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import (
    _cross,
    _gable_across,
    _iso,
    _on_wall,
    _patch,
    _portal,
    _turned,
    _windows,
)

type Top = tuple[Point, Point, Point, Point]

GRAVEL: RGB = (204, 194, 168)
FLAGS: RGB = (190, 184, 170)
YEW: RGB = (40, 82, 46)


def _lerp(a: Point, b: Point, f: float) -> Point:
    return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f


def _plinth(p: Painter, top: Top, height: float, wall: RGB, depth: float = 1) -> None:
    """A darker band of stone along the foot of a box's two front walls."""
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (height - depth, height), shade(wall, 0.72))
    _patch(p, front, right, (0, 1), (height - depth, height), shade(wall, 0.54))


def _tiles(
    p: Painter, top: Top, rise: float, roof: RGB, rows: tuple[float, ...] = (0.34, 0.68)
) -> None:
    """Rows of tiles across the lit side of a gable roof (Painter.gable)."""
    left, front, right, back = top
    ridge_back = _lerp(left, back, 0.5)[0], _lerp(left, back, 0.5)[1] - rise
    ridge_front = _lerp(front, right, 0.5)[0], _lerp(front, right, 0.5)[1] - rise
    for f in rows:
        p.line([_lerp(left, ridge_back, f), _lerp(front, ridge_front, f)], shade(roof, 0.8))


def _courses(
    p: Painter, top: Top, rise: float, roof: RGB, rows: tuple[float, ...] = (0.34, 0.68)
) -> None:
    """Rows of tiles or lead along the lit side of a roof with its gable to the front
    (_gable_across)."""
    left, front, right, back = top
    near, far = _on_wall(left, front, 0.5, -rise), _on_wall(back, right, 0.5, -rise)
    for f in rows:
        p.line([_lerp(left, near, f), _lerp(back, far, f)], shade(roof, 0.8))


def _grave(p: Painter, x: float, y: float, cross: bool = False) -> None:
    """A headstone, or a stone cross, over a low mound of earth."""
    stone = (196, 192, 184)
    p.blob(x - 1.5, y - 0.5, x + 1.5, y + 1, (84, 112, 52))
    if cross:
        p.line([(x, y), (x, y - 3)], stone)
        p.line([(x - 1, y - 2), (x + 1, y - 2)], stone)
        return
    p.rect(x - 1, y - 2, x + 1, y, stone)
    p.line([(x - 0.5, y - 2.5), (x + 0.5, y - 2.5)], stone)
    p.point(x + 1, y - 1, shade(stone, 0.7))


def _forecourt(
    p: Painter, a: Point, b: Point, span: tuple[float, float], height: float, reach: float
) -> None:
    """Flagstones laid on the ground before a door in the wall under the edge from `a` to `b`,
    with joints across them."""
    _worn(p, a, b, span, height, reach, FLAGS)
    for f in (0.4, 0.75):
        near = _on_wall(a, b, span[0] - 0.25 * (span[1] - span[0]) * f, height)
        far = _on_wall(a, b, span[1] + 0.25 * (span[1] - span[0]) * f, height)
        p.line(
            [
                (near[0] - reach * f, near[1] + reach * f / 2),
                (far[0] - reach * f, far[1] + reach * f / 2),
            ],
            shade(FLAGS, 0.85),
        )


def _yew(p: Painter, x: float, y: float, height: float = 5) -> None:
    """A dark clipped yew standing on the ground at (x, y)."""
    p.ellipse(x - 1.5, y - height, x + 1.5, y, YEW)
    p.line([(x - 0.5, y - height + 1), (x - 0.5, y - 2)], shade(YEW, 1.4))


def _rose(p: Painter, x: float, y: float, radius: float) -> None:
    """A round rose window of stained glass."""
    p.ellipse(x - radius, y - radius, x + radius, y + radius, STAINED)
    p.blob(x - radius / 2, y - radius / 2, x + radius / 2, y + radius / 2, shade(STAINED, 1.7))
    p.point(x, y, CROSS)


def _ribs(
    p: Painter,
    x: float,
    y: float,
    profile: list[tuple[float, float]],
    angles: tuple[float, ...],
    colour: RGB,
) -> None:
    """Ribs up the front of a dome drawn with _turned at (x, y) with `profile`, at each of
    `angles` round it, a quarter turn being straight to the front."""
    for a in angles:
        p.line([(x + w * math.cos(a), y - h + w / 2 * math.sin(a)) for h, w in profile], colour)


def _tower(
    p: Painter,
    x: float,
    y: float,
    half: float,
    height: float,
    spire: float,
    roof: RGB,
    wall: RGB = CHURCH_WALL,
) -> tuple[Point, Point, Point, Point]:
    """A square church tower on a stone plinth, with a string course under its arched belfry
    openings, lancets lower down a tall one, and a pyramid spire with a gold cross. Returns the
    corners of its top."""
    top = p.box(x, y, half, height, wall, None)
    left, front, right, _ = top
    _plinth(p, top, height, wall)
    _foot(p, top, height)
    openings = (0.5,) if half < 3 else (0.33, 0.67)
    _windows(p, left, front, openings, (2,), 3, DOORWAY)
    _windows(p, front, right, openings, (2,), 3, shade(DOORWAY, 0.8))
    if height > 18:
        _windows(p, left, front, (0.5,), (height * 0.42,), 3, STAINED, shade(wall, 0.7))
        _windows(p, front, right, (0.5,), (height * 0.42,), 3, shade(STAINED, 0.8))
    for a, b, tone in ((left, front, 0.8), (front, right, 0.62)):
        p.line([_on_wall(a, b, 0, 5.8), _on_wall(a, b, 1, 5.8)], shade(wall, tone))
    p.line([left, front, right], shade(wall, 1.12))
    apex = (x, y - height - spire)
    p.polygon([left, front, apex], roof, shade(roof, 0.55))
    p.polygon([front, right, apex], shade(roof, 0.7), shade(roof, 0.45))
    p.line([front, apex], shade(roof, 1.3))
    if spire > 8:
        for f in (0.3, 0.6):
            p.line([_lerp(left, apex, f), _lerp(front, apex, f)], shade(roof, 0.8))
    _cross(p, apex[0], apex[1] + 1)
    return top


def _steps(p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float) -> None:
    """Light stone steps on the ground before the wall under the edge from `a` to `b`."""
    near, far = _on_wall(a, b, span[0], drop), _on_wall(a, b, span[1], drop)
    out = [(far[0] - 1.5, far[1] + 0.75), (near[0] - 1.5, near[1] + 0.75)]
    p.polygon([near, far, *out], (226, 222, 212), (170, 166, 158))


def church() -> Sprite:
    """A whitewashed village church on a stone plinth: a nave under a steep tiled roof with tall
    arched windows, a bell tower with a round-arched door, a slate spire topped by a gold cross,
    and graves in the churchyard."""
    p = Painter(30, 46, (14, 36))
    p.shadow(12, 3.5)
    for x, y, cross in ((-12, 5, False), (-9, 6.5, True), (-6, 8, False)):
        _grave(p, x, y, cross)
    _yew(p, -13, 1.5)
    top = p.hall(-3, -1, 12, 7, 7, WHITEWASH, None)
    left, front, *_ = top
    _plinth(p, top, 7, WHITEWASH)
    _foot(p, top, 7)
    for t in (0.3, 0.54):
        p.line([_on_wall(left, front, t, 2), _on_wall(left, front, t, 7)], shade(WHITEWASH, 0.8))
    _windows(p, left, front, (0.18, 0.42, 0.66), (1.5,), 4, STAINED, shade(WHITEWASH, 0.7))
    p.gable(top, 6, CHURCH_ROOF, WHITEWASH)
    _tiles(p, top, 6, CHURCH_ROOF, (0.25, 0.5, 0.75))
    tower = _tower(p, 4, 2.5, 3.5, 14, 11, SLATE, WHITEWASH)
    _portal(p, tower[0], tower[1], (0.3, 0.7), 14, 4)
    _worn(p, tower[0], tower[1], (0.3, 0.7), 14, 5.5, GRAVEL)
    _steps(p, tower[0], tower[1], (0.25, 0.75), 14)
    _rose(p, *_on_wall(tower[0], tower[1], 0.5, 7), 1)
    return p.sprite()


def basilica() -> Sprite:
    """A Romanesque basilica: twin bell towers with crosses flank a front with a rose window over
    a great arched door up a flight of steps, before a tall nave lit by arched clerestory windows
    under a ribbed copper-green dome."""
    p = Painter(40, 52, (19, 40))
    p.shadow(16, 5)
    cx, cy = 1, -2
    depth, nave, aisle = 16, 7, 4
    side = (nave + aisle) / 2
    left_aisle = p.hall(
        *_iso(cx, cy, -side, 0), aisle, depth, 8, CHURCH_WALL, shade(CHURCH_ROOF, 0.9)
    )
    _plinth(p, left_aisle, 8, CHURCH_WALL)
    _foot(p, left_aisle, 8)
    _windows(p, left_aisle[0], left_aisle[1], (0.5,), (2,), 3, STAINED, shade(CHURCH_WALL, 0.7))
    _tower(p, *_iso(cx, cy, -5.5, -5.5), 2.5, 20, 4, CHURCH_ROOF)
    top = p.hall(cx, cy, nave, depth, 14, CHURCH_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 14, CHURCH_WALL)
    _foot(p, top, 14)
    _portal(p, left, front, (0.25, 0.75), 14, 6)
    p.point(*_on_wall(left, front, 0.5, 9.5), CROSS)
    _forecourt(p, left, front, (0.1, 0.9), 14, 5)
    _steps(p, left, front, (0.15, 0.85), 14)
    _windows(
        p,
        front,
        right,
        (0.45, 0.6, 0.75, 0.9),
        (2,),
        3,
        shade(STAINED, 0.8),
        shade(CHURCH_WALL, 0.6),
    )
    p.line(
        [_on_wall(front, right, 0, 6.5), _on_wall(front, right, 1, 6.5)], shade(CHURCH_WALL, 0.65)
    )
    _gable_across(p, top, 4, CHURCH_ROOF, CHURCH_WALL)
    _courses(p, top, 4, CHURCH_ROOF)
    _rose(p, *_on_wall(left, front, 0.5, 2.5), 2)
    dome_x, dome_y = _iso(cx, cy, 0, 4.5)
    _turned(p, dome_x, dome_y - 16, [(0, 3.5), (4, 3.5)], CHURCH_WALL)
    _windows(p, (dome_x - 3, dome_y - 20), (dome_x + 3, dome_y - 20), (0.2, 0.5), (1,), 2, STAINED)
    dome = [(0, 4), (2, 3.6), (3.5, 2.6), (4.5, 1.2), (5, 0)]
    _turned(p, dome_x, dome_y - 20, dome, VERDIGRIS)
    _ribs(p, dome_x, dome_y - 20, dome, (1.15, 2.0, 2.7), shade(VERDIGRIS, 1.3))
    _cross(p, dome_x, dome_y - 25)
    right_aisle = p.hall(
        *_iso(cx, cy, side, 0), aisle, depth, 8, CHURCH_WALL, shade(CHURCH_ROOF, 0.9)
    )
    _plinth(p, right_aisle, 8, CHURCH_WALL)
    _foot(p, right_aisle, 8)
    _windows(
        p,
        right_aisle[1],
        right_aisle[2],
        (0.45, 0.65, 0.85),
        (2,),
        3,
        shade(STAINED, 0.8),
        shade(CHURCH_WALL, 0.6),
    )
    _windows(p, right_aisle[0], right_aisle[1], (0.5,), (2,), 3, STAINED, shade(CHURCH_WALL, 0.7))
    _tower(p, *_iso(cx, cy, 5.5, -5.5), 2.5, 20, 4, CHURCH_ROOF)
    return p.sprite()


def cathedral() -> Sprite:
    """A Gothic cathedral: two tall spires with crosses flank a great rose window over a pointed
    door up a flight of steps, before a long slate-roofed nave held up by flying buttresses with
    pinnacles and lit by tall pointed windows."""
    p = Painter(50, 66, (24, 53))
    p.shadow(20, 6)
    cx, cy = 0, -3
    depth, nave, aisle = 22, 8, 4
    side = (nave + aisle) / 2
    roof = SLATE
    left_aisle = p.hall(*_iso(cx, cy, -side, 0), aisle, depth, 9, CHURCH_WALL, shade(roof, 0.9))
    _plinth(p, left_aisle, 9, CHURCH_WALL)
    _foot(p, left_aisle, 9)
    _windows(p, left_aisle[0], left_aisle[1], (0.5,), (2,), 4, STAINED, shade(CHURCH_WALL, 0.7))
    _tower(p, *_iso(cx, cy, -5.5, -8.5), 3, 27, 15, roof)
    top = p.hall(cx, cy, nave, depth, 17, CHURCH_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 17, CHURCH_WALL)
    _foot(p, top, 17)
    _portal(p, left, front, (0.3, 0.7), 17, 6)
    p.point(*_on_wall(left, front, 0.5, 12.5), CROSS)
    _forecourt(p, left, front, (0.05, 0.95), 17, 5.5)
    _steps(p, left, front, (0.2, 0.8), 17)
    _windows(
        p,
        front,
        right,
        (0.4, 0.52, 0.64, 0.76, 0.88),
        (2,),
        5,
        shade(STAINED, 0.8),
        shade(CHURCH_WALL, 0.6),
    )
    _gable_across(p, top, 7, roof, CHURCH_WALL)
    _courses(p, top, 7, roof, (0.25, 0.5, 0.75))
    _rose(p, *_on_wall(left, front, 0.5, 4.5), 2.5)
    right_aisle = p.hall(*_iso(cx, cy, side, 0), aisle, depth, 9, CHURCH_WALL, shade(roof, 0.9))
    _plinth(p, right_aisle, 9, CHURCH_WALL)
    _foot(p, right_aisle, 9)
    _windows(
        p,
        right_aisle[1],
        right_aisle[2],
        (0.46, 0.64, 0.82),
        (2,),
        5,
        shade(STAINED, 0.8),
        shade(CHURCH_WALL, 0.6),
    )
    _windows(p, right_aisle[0], right_aisle[1], (0.5,), (2,), 4, STAINED, shade(CHURCH_WALL, 0.7))
    buttress = shade(CHURCH_WALL, 0.82)
    for v in (-2, 2, 6, 10):
        foot_x, foot_y = _iso(cx, cy, nave / 2 + aisle + 0.5, v)
        wall_x, wall_y = _iso(cx, cy, nave / 2, v)
        p.rect(foot_x - 1, foot_y - 12, foot_x, foot_y, buttress)
        p.line([(foot_x, foot_y - 11), (wall_x + 1, wall_y - 15)], buttress)
        p.line([(foot_x - 1, foot_y - 12), (wall_x, wall_y - 16)], shade(buttress, 1.15))
        p.line([(foot_x - 1, foot_y - 12), (foot_x - 1, foot_y - 15)], shade(buttress, 1.15))
        p.point(foot_x - 1, foot_y - 16, shade(buttress, 1.2))
    _tower(p, *_iso(cx, cy, 5.5, -8.5), 3, 27, 15, roof)
    return p.sprite()


def st_pauls() -> Sprite:
    """A Baroque cathedral like St Paul's: a great ribbed lead dome on a columned drum, crowned by
    a lantern and a gold cross, over a long balustraded body fronted by a columned portico up a
    flight of steps between two west towers, one with a clock."""
    p = Painter(60, 74, (29, 61))
    p.shadow(25, 7)
    cx, cy = 3, -3
    stone = CHURCH_WALL

    def west_tower(u: float, clock: bool) -> None:
        tx, ty = _iso(cx, cy, u, -11)
        tower = p.box(tx, ty, 3, 23, stone, None)
        _plinth(p, tower, 23, stone)
        _foot(p, tower, 23)
        _windows(p, tower[0], tower[1], (0.33, 0.67), (3,), 3, DOORWAY)
        _windows(p, tower[1], tower[2], (0.33, 0.67), (3,), 3, shade(DOORWAY, 0.8))
        for a, b, tone in ((tower[0], tower[1], 0.8), (tower[1], tower[2], 0.62)):
            p.line([_on_wall(a, b, 0, 7), _on_wall(a, b, 1, 7)], shade(stone, tone))
        if clock:
            x, y = _on_wall(tower[0], tower[1], 0.5, 10)
            p.ellipse(x - 1.5, y - 1.5, x + 1.5, y + 1.5, (240, 236, 222))
            p.line([(x, y), (x, y - 1)], DOORWAY)
        _turned(p, tx, ty - 23, [(0, 2.5), (3, 2.5)], stone)
        _turned(p, tx, ty - 26, [(0, 2.8), (2, 2), (3, 1), (3.5, 0)], LEAD)
        _cross(p, tx, ty - 28)

    west_tower(-9.5, False)
    top = p.hall(cx, cy, 14, 28, 13, stone, None)
    _, front, right, _ = top
    _plinth(p, top, 13, stone)
    _foot(p, top, 13)
    for t in range(1, 10):
        p.line(
            [_on_wall(front, right, t / 10, 1), _on_wall(front, right, t / 10, 13)],
            shade(stone, 0.68),
        )
    _windows(p, front, right, (0.15, 0.35, 0.55, 0.75, 0.95), (3,), 4, shade(DOORWAY, 0.8))
    p.line([_on_wall(front, right, 0, 8.5), _on_wall(front, right, 1, 8.5)], shade(stone, 0.6))
    _gable_across(p, top, 3, LEAD, stone)
    p.line([front, right], shade(stone, 1.12))
    dome_x, dome_y = _iso(cx, cy, 0, 5)
    drum = dome_y - 14
    _turned(p, dome_x, drum, [(0, 9), (8, 9)], stone)
    for a in np.linspace(0.25, math.pi - 0.25, 9):
        col_x, col_y = dome_x + 9 * math.cos(a), drum + 4.5 * math.sin(a)
        p.line([(col_x, col_y - 1), (col_x, col_y - 7)], shade(stone, 0.62))
    p.line(
        [
            (dome_x + 9 * math.cos(a), drum - 8 + 4.5 * math.sin(a))
            for a in np.linspace(0, math.pi, 13)
        ],
        shade(stone, 1.12),
    )
    dome = [(0, 10), (3, 9.6), (6, 8.3), (9, 5.8), (11, 2.8), (12, 0)]
    _turned(p, dome_x, drum - 8, dome, LEAD)
    p.blob(dome_x - 7, drum - 17, dome_x - 2, drum - 11, shade(LEAD, 1.3))
    _ribs(p, dome_x, drum - 8, dome, (0.55, 1.05, 1.57, 2.1, 2.6), shade(LEAD, 0.8))
    _turned(p, dome_x, drum - 19, [(0, 1.8), (4, 1.8)], stone)
    p.line([(dome_x - 0.5, drum - 20), (dome_x - 0.5, drum - 22)], DOORWAY)
    _turned(p, dome_x, drum - 23, [(0, 2), (1.5, 1), (2, 0)], LEAD)
    _cross(p, dome_x, drum - 24, 7)
    portico = p.hall(*_iso(cx, cy, 0, -15.5), 10, 3, 9, stone, None)
    _forecourt(p, portico[0], portico[1], (0.05, 0.95), 9, 4.5)
    _steps(p, portico[0], portico[1], (0.05, 0.95), 9)
    for t in range(1, 10, 2):
        p.line(
            [
                _on_wall(portico[0], portico[1], t / 10, 1),
                _on_wall(portico[0], portico[1], t / 10, 9),
            ],
            DOORWAY,
        )
    _gable_across(p, portico, 3, LEAD, stone)
    tympanum = _on_wall(portico[0], portico[1], 0.5, -1)
    p.point(tympanum[0] - 0.5, tympanum[1], shade(stone, 0.7))
    west_tower(9.5, True)
    return p.sprite()
