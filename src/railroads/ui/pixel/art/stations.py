"""Stations at each level."""

import numpy as np

from railroads.ui.pixel.art.painter import (
    BRICK,
    NAME_BOARD,
    PLATFORM,
    PLATFORM_EDGE,
    STATION_ROOF,
    STATION_WALL,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import (
    _flag,
    _gable_across,
    _iso,
    _on_wall,
    _patch,
    _vault,
    _windows,
)


def _station_ground(p: Painter, half: float = 15) -> None:
    """The paved platform of a station filling its cell, with a white edge along the track."""
    p.polygon([(-half, 1), (0, half / 2 + 1), (half, 1), (0, -half / 2 + 1)], shade(PLATFORM, 0.62))
    p.polygon(
        [(-half, 0), (0, half / 2), (half, 0), (0, -half / 2)], PLATFORM, shade(PLATFORM, 0.8)
    )
    p.line([(-half + 2, 0), (-1, half / 2 - 1.5)], PLATFORM_EDGE)


def _name_board(p: Painter, x: float, y: float, half: float = 3) -> None:
    """A blue station name board on two posts, its foot at (x, y)."""
    post = (52, 50, 54)
    for dx in (-half + 1, half - 1):
        p.line([(x + dx, y), (x + dx, y - 4)], post)
    p.rect(x - half, y - 7, x + half, y - 4, NAME_BOARD)
    for dx in range(-round(half) + 1, round(half), 1):
        if dx % 2 == 0 or abs(dx) < half - 1:
            p.point(x + dx, y - 5.5, (236, 240, 250))


def _lamp(p: Painter, x: float, y: float, height: float = 7) -> None:
    p.line([(x, y), (x, y - height)], (40, 40, 46))
    p.rect(x - 1, y - height - 2, x + 1, y - height, (255, 226, 120))


def _clock(p: Painter, x: float, y: float, radius: float = 1.5) -> None:
    p.ellipse(x - radius, y - radius, x + radius, y + radius, (246, 244, 236))
    p.line([(x, y), (x, y - radius + 0.5)], (30, 30, 34))
    p.point(x + 1, y, (30, 30, 34))


def _canopy(
    p: Painter, x: float, y: float, length: float, depth: float, height: float
) -> tuple[Point, Point, Point, Point]:
    """A platform canopy on posts: a flat dark red roof `height` up with a cream valance."""
    top = p.hall(x, y - height, length, depth, 1.5, STATION_WALL, STATION_ROOF)
    p.line([_on_wall(top[0], top[1], 0, 1.5), _on_wall(top[0], top[1], 1, 1.5)], STATION_ROOF)
    for t in np.linspace(0.08, 0.92, max(2, round(length / 4))):
        fx, fy = _on_wall(top[0], top[1], t, height + 1.5)
        p.line([(fx, fy), (fx, fy - height)], (60, 58, 62))
    return top


def station() -> Sprite:
    """A small country station: a cream station house with a dark red roof and brick base, a
    canopy over the platform, a blue name board and a lamp by the white platform edge."""
    p = Painter(34, 38, (16, 27))
    _station_ground(p)
    house = p.hall(*_iso(0, 0, 1, 3), 9, 5, 6, STATION_WALL, None)
    left, front, *_ = house
    _patch(p, left, front, (0, 1), (4.5, 6), BRICK)
    _windows(p, left, front, (0.2, 0.75), (1.5,), 2)
    _patch(p, left, front, (0.42, 0.58), (2, 6), (88, 60, 44))
    p.gable(house, 4, STATION_ROOF, STATION_WALL)
    chimney = _on_wall(house[0], house[3], 0.5, -5)
    p.rect(chimney[0] - 1, chimney[1] - 3, chimney[0] + 1, chimney[1], BRICK)
    _canopy(p, *_iso(0, 0, 0, -1.5), 10, 3, 5)
    _name_board(p, *_iso(0, 0, -5, -4.5))
    _lamp(p, *_iso(0, 0, 5.5, -5))
    return p.sprite()


def station_2() -> Sprite:
    """A town station: a two-storey cream building under a dark red roof with a clock in its
    central gable, and a long canopy over the platform with name boards and lamps."""
    p = Painter(34, 42, (16, 30))
    _station_ground(p)
    house = p.hall(*_iso(0, 0, 0, 4), 13, 5, 10, STATION_WALL, None)
    left, front, right, _ = house
    _patch(p, left, front, (0, 1), (9, 10), BRICK)
    _windows(p, left, front, (0.1, 0.25, 0.4, 0.6, 0.75, 0.9), (1.5, 5.5), 2)
    _patch(p, left, front, (0.44, 0.56), (6.5, 10), (88, 60, 44))
    _windows(p, front, right, (0.3, 0.7), (1.5, 5.5), 2, shade(WINDOW, 0.75))
    p.gable(house, 4, STATION_ROOF, STATION_WALL)
    gx, gy = _iso(0, 0, 0, 1.5)
    gable = p.hall(gx, gy, 5, 1, 10, STATION_WALL, None)
    _gable_across(p, gable, 4, STATION_ROOF, STATION_WALL)
    _clock(p, *_on_wall(gable[0], gable[1], 0.5, -1.5))
    _canopy(p, *_iso(0, 0, 0, -2.5), 14, 3, 6)
    for u in (-5.5, 2):
        _name_board(p, *_iso(0, 0, u, -5))
    for u in (-1.5, 6):
        _lamp(p, *_iso(0, 0, u, -5.5))
    return p.sprite()


def station_3() -> Sprite:
    """A grand terminal: a cream head building with a clock tower under a dark red spire, before
    an arched train shed of iron and glass over the platforms, with name boards and lamps."""
    p = Painter(38, 58, (18, 45))
    _station_ground(p)
    house = p.hall(*_iso(0, 0, 0, 5.5), 15, 4, 11, STATION_WALL, None)
    left, front, *_ = house
    _patch(p, left, front, (0, 1), (10, 11), BRICK)
    _windows(p, left, front, tuple(k / 10 for k in range(1, 10)), (1.5, 5.5), 2)
    p.gable(house, 3, STATION_ROOF, STATION_WALL)
    tx, ty = _iso(0, 0, -5, 5)
    tower = p.box(tx, ty, 2.5, 22, STATION_WALL, None)
    for a, b in ((tower[0], tower[1]), (tower[1], tower[2])):
        p.line([_on_wall(a, b, 0, 1), _on_wall(a, b, 1, 1)], shade(STATION_WALL, 1.08))
        _windows(p, a, b, (0.5,), (8, 13), 2)
    _clock(p, *_on_wall(tower[0], tower[1], 0.5, 4), 1.6)
    apex = (tx, ty - 31)
    p.polygon([tower[0], tower[1], apex], STATION_ROOF, shade(STATION_ROOF, 0.55))
    p.polygon([tower[1], tower[2], apex], shade(STATION_ROOF, 0.7), shade(STATION_ROOF, 0.45))
    _flag(p, apex[0], apex[1], 4, NAME_BOARD)
    glass, iron = (186, 220, 236), (70, 82, 100)
    shed = p.hall(*_iso(0, 0, 0, -1.5), 15, 8, 5, STATION_WALL, None)
    sl, sf, sr, _ = shed
    for t in np.linspace(0.07, 0.93, 7):
        p.line([_on_wall(sl, sf, t, 0.5), _on_wall(sl, sf, t, 5)], iron)
    _patch(p, sl, sf, (0, 1), (0, 1), STATION_ROOF)
    _vault(p, shed, 7, glass, glass, tuple(k / 8 for k in range(1, 8)), iron)
    _patch(p, sf, sr, (0.25, 0.75), (1.5, 5), (52, 56, 66))
    board = _on_wall(sf, sr, 0.5, 0.5)
    p.rect(board[0] - 3, board[1] - 2, board[0] + 3, board[1] + 1, NAME_BOARD)
    for dx in (-2, 0, 2):
        p.point(board[0] + dx, board[1] - 0.5, (236, 240, 250))
    _name_board(p, *_iso(0, 0, -4, -6.5))
    for u in (-0.5, 5):
        _lamp(p, *_iso(0, 0, u, -6.5))
    return p.sprite()
