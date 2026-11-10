"""Churches, basilicas, cathedrals and St. Paul's."""

import math

import numpy as np

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
    _portal,
    _turned,
    _windows,
)


def _rose(p: Painter, x: float, y: float, radius: float) -> None:
    """A round rose window of stained glass."""
    p.ellipse(x - radius, y - radius, x + radius, y + radius, STAINED)
    p.blob(x - radius / 2, y - radius / 2, x + radius / 2, y + radius / 2, shade(STAINED, 1.7))
    p.point(x, y, CROSS)


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
    """A square church tower with arched belfry openings under a pyramid spire and a gold cross.
    Returns the corners of its top."""
    top = p.box(x, y, half, height, wall, None)
    left, front, right, _ = top
    openings = (0.5,) if half < 3 else (0.33, 0.67)
    _windows(p, left, front, openings, (2,), 3, DOORWAY)
    _windows(p, front, right, openings, (2,), 3, shade(DOORWAY, 0.8))
    p.line([left, front, right], shade(wall, 1.12))
    apex = (x, y - height - spire)
    p.polygon([left, front, apex], roof, shade(roof, 0.55))
    p.polygon([front, right, apex], shade(roof, 0.7), shade(roof, 0.45))
    p.line([front, apex], shade(roof, 1.3))
    _cross(p, apex[0], apex[1] + 1)
    return top


def church() -> Sprite:
    """A whitewashed village church: a nave under a steep tiled roof with tall arched windows,
    and a bell tower with a round-arched door and a slate spire topped by a gold cross."""
    p = Painter(30, 46, (14, 36))
    p.shadow(12, 3.5)
    top = p.hall(-3, -1, 12, 7, 7, WHITEWASH, None)
    left, front, *_ = top
    _windows(p, left, front, (0.18, 0.42, 0.66), (1.5,), 4, STAINED)
    p.gable(top, 6, CHURCH_ROOF, WHITEWASH)
    tower = _tower(p, 4, 2.5, 3.5, 14, 11, SLATE, WHITEWASH)
    _portal(p, tower[0], tower[1], (0.3, 0.7), 14, 4)
    _rose(p, *_on_wall(tower[0], tower[1], 0.5, 7), 1)
    return p.sprite()


def basilica() -> Sprite:
    """A Romanesque basilica: twin bell towers with crosses flank a front with a rose window over
    a great arched door, before a tall nave lit by clerestory windows under a copper-green dome."""
    p = Painter(40, 52, (19, 40))
    p.shadow(16, 5)
    cx, cy = 1, -2
    depth, nave, aisle = 16, 7, 4
    side = (nave + aisle) / 2
    p.hall(*_iso(cx, cy, -side, 0), aisle, depth, 8, CHURCH_WALL, shade(CHURCH_ROOF, 0.9))
    _tower(p, *_iso(cx, cy, -5.5, -5.5), 2.5, 20, 4, CHURCH_ROOF)
    top = p.hall(cx, cy, nave, depth, 14, CHURCH_WALL, None)
    left, front, right, _ = top
    _portal(p, left, front, (0.25, 0.75), 14, 6)
    _windows(p, front, right, (0.45, 0.6, 0.75, 0.9), (2,), 3, shade(STAINED, 0.8))
    _gable_across(p, top, 4, CHURCH_ROOF, CHURCH_WALL)
    _rose(p, *_on_wall(left, front, 0.5, 2.5), 2)
    dome_x, dome_y = _iso(cx, cy, 0, 4.5)
    _turned(p, dome_x, dome_y - 16, [(0, 3.5), (4, 3.5)], CHURCH_WALL)
    _windows(p, (dome_x - 3, dome_y - 20), (dome_x + 3, dome_y - 20), (0.2, 0.5), (1,), 2, STAINED)
    _turned(p, dome_x, dome_y - 20, [(0, 4), (2, 3.6), (3.5, 2.6), (4.5, 1.2), (5, 0)], VERDIGRIS)
    _cross(p, dome_x, dome_y - 25)
    right_aisle = p.hall(
        *_iso(cx, cy, side, 0), aisle, depth, 8, CHURCH_WALL, shade(CHURCH_ROOF, 0.9)
    )
    _windows(p, right_aisle[1], right_aisle[2], (0.45, 0.65, 0.85), (2,), 3, shade(STAINED, 0.8))
    _tower(p, *_iso(cx, cy, 5.5, -5.5), 2.5, 20, 4, CHURCH_ROOF)
    return p.sprite()


def cathedral() -> Sprite:
    """A Gothic cathedral: two tall spires with crosses flank a great rose window over a pointed
    door, before a long nave held up by flying buttresses and lit by tall pointed windows."""
    p = Painter(50, 66, (24, 53))
    p.shadow(20, 6)
    cx, cy = 0, -3
    depth, nave, aisle = 22, 8, 4
    side = (nave + aisle) / 2
    roof = SLATE
    p.hall(*_iso(cx, cy, -side, 0), aisle, depth, 9, CHURCH_WALL, shade(roof, 0.9))
    _tower(p, *_iso(cx, cy, -5.5, -8.5), 3, 27, 15, roof)
    top = p.hall(cx, cy, nave, depth, 17, CHURCH_WALL, None)
    left, front, right, _ = top
    _portal(p, left, front, (0.3, 0.7), 17, 6)
    _windows(p, front, right, (0.4, 0.52, 0.64, 0.76, 0.88), (2,), 5, shade(STAINED, 0.8))
    _gable_across(p, top, 7, roof, CHURCH_WALL)
    _rose(p, *_on_wall(left, front, 0.5, 4.5), 2.5)
    right_aisle = p.hall(*_iso(cx, cy, side, 0), aisle, depth, 9, CHURCH_WALL, shade(roof, 0.9))
    _windows(p, right_aisle[1], right_aisle[2], (0.46, 0.64, 0.82), (2,), 5, shade(STAINED, 0.8))
    buttress = shade(CHURCH_WALL, 0.82)
    for v in (-2, 2, 6, 10):
        foot_x, foot_y = _iso(cx, cy, nave / 2 + aisle + 0.5, v)
        wall_x, wall_y = _iso(cx, cy, nave / 2, v)
        p.rect(foot_x - 1, foot_y - 12, foot_x, foot_y, buttress)
        p.line([(foot_x, foot_y - 11), (wall_x + 1, wall_y - 15)], buttress)
        p.line([(foot_x - 1, foot_y - 12), (wall_x, wall_y - 16)], shade(buttress, 1.15))
        p.line([(foot_x - 1, foot_y - 12), (foot_x - 1, foot_y - 14)], shade(buttress, 1.15))
    _tower(p, *_iso(cx, cy, 5.5, -8.5), 3, 27, 15, roof)
    return p.sprite()


def st_pauls() -> Sprite:
    """A Baroque cathedral like St Paul's: a great lead dome on a columned drum, crowned by a
    lantern and a gold cross, over a long body fronted by a columned portico between two west
    towers."""
    p = Painter(60, 74, (29, 61))
    p.shadow(25, 7)
    cx, cy = 3, -3
    stone = CHURCH_WALL

    def west_tower(u: float) -> None:
        tx, ty = _iso(cx, cy, u, -11)
        tower = p.box(tx, ty, 3, 23, stone, None)
        _windows(p, tower[0], tower[1], (0.33, 0.67), (3,), 3, DOORWAY)
        _windows(p, tower[1], tower[2], (0.33, 0.67), (3,), 3, shade(DOORWAY, 0.8))
        _turned(p, tx, ty - 23, [(0, 2.5), (3, 2.5)], stone)
        _turned(p, tx, ty - 26, [(0, 2.8), (2, 2), (3, 1), (3.5, 0)], LEAD)
        _cross(p, tx, ty - 28)

    west_tower(-9.5)
    top = p.hall(cx, cy, 14, 28, 13, stone, None)
    _, front, right, _ = top
    for t in range(1, 10):
        p.line(
            [_on_wall(front, right, t / 10, 1), _on_wall(front, right, t / 10, 13)],
            shade(stone, 0.68),
        )
    _windows(p, front, right, (0.15, 0.35, 0.55, 0.75, 0.95), (3,), 4, shade(DOORWAY, 0.8))
    _gable_across(p, top, 3, LEAD, stone)
    dome_x, dome_y = _iso(cx, cy, 0, 5)
    drum = dome_y - 14
    _turned(p, dome_x, drum, [(0, 9), (8, 9)], stone)
    for a in np.linspace(0.25, math.pi - 0.25, 9):
        col_x, col_y = dome_x + 9 * math.cos(a), drum + 4.5 * math.sin(a)
        p.line([(col_x, col_y - 1), (col_x, col_y - 7)], shade(stone, 0.62))
    _turned(p, dome_x, drum - 8, [(0, 10), (3, 9.6), (6, 8.3), (9, 5.8), (11, 2.8), (12, 0)], LEAD)
    p.blob(dome_x - 7, drum - 17, dome_x - 2, drum - 11, shade(LEAD, 1.3))
    _turned(p, dome_x, drum - 19, [(0, 1.8), (4, 1.8)], stone)
    _turned(p, dome_x, drum - 23, [(0, 2), (1.5, 1), (2, 0)], LEAD)
    _cross(p, dome_x, drum - 24, 7)
    portico = p.hall(*_iso(cx, cy, 0, -15.5), 10, 3, 9, stone, None)
    for t in range(1, 10, 2):
        p.line(
            [
                _on_wall(portico[0], portico[1], t / 10, 1),
                _on_wall(portico[0], portico[1], t / 10, 9),
            ],
            DOORWAY,
        )
    _gable_across(p, portico, 3, LEAD, stone)
    west_tower(9.5)
    return p.sprite()
