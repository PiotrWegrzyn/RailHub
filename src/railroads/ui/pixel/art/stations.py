"""Stations at each level: drawn in the pixel style for squares, and as small models turned
along their track for hexes."""

import math
from collections.abc import Callable

import numpy as np

from railroads.sim.grids import HexGrid
from railroads.ui.pixel.art.painter import (
    BRICK,
    NAME_BOARD,
    RGB,
    STATION_ROOF,
    STATION_WALL,
    WINDOW,
    Painter,
    Point,
    Sprite,
    fineness,
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
from railroads.ui.pixel.models import Face, Model, draw_model
from railroads.ui.pixel.projection import PixelProjection


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


# The station houses on hexes, built as models (railroads.ui.pixel.models) and drawn turned along
# the track through them: for each way track runs on the screen, the heading the house's length
# lies along, chosen so its front, with the platform canopy, faces the track. The house stands
# behind track running across or slanting and to the left of track running up and down
# (railroads.ui.pixel.track_art.station_side).
TURNED_LINES = {"across": 0.0, "falling": 60.0, "rising": 300.0, "upright": 270.0}
# The hex map's look (railroads.ui.map_view.LOOKS): not turned, a cell wide per cell.
_HEX_PROJECTION = PixelProjection.of(HexGrid(rows=1, cols=1, km_per_cell=1.0), 0.0, 1.0)
POST: RGB = (60, 58, 62)
DOOR: RGB = (88, 60, 44)
CLOCK: RGB = (246, 244, 236)
GLASS: RGB = (186, 220, 236)
IRON: RGB = (70, 82, 100)


def _house(m: Model, half: float, depth: float, height: float, storeys: int) -> None:
    """A cream house with a brick base, windows on each storey, a door in the middle of its
    front and a dark red pitched roof."""
    m.box((-half, half), (-depth, depth), (0.0, 0.03), BRICK)
    m.box((-half, half), (-depth, depth), (0.03, height), STATION_WALL)
    m.roof(
        (-half - 0.02, half + 0.02),
        (-depth - 0.02, depth + 0.02),
        (height, height + 0.1),
        STATION_ROOF,
        STATION_WALL,
    )
    count = max(2, round(half / 0.06))
    for storey in range(storeys):
        low = 0.065 + storey * 0.1
        for k in range(count):
            u = -half + (k + 0.5) * 2 * half / count
            if storey == 0 and abs(u) < 0.04:
                continue
            m.panel((u - 0.018, u + 0.018), (low, low + 0.045), depth + 0.0005, WINDOW)
    m.panel((-0.025, 0.025), (0.03, 0.12), depth + 0.001, DOOR)


def _shelter(
    m: Model, half: float, front: float, reach: float, height: float, roof: RGB = STATION_ROOF
) -> None:
    """A flat canopy over the platform in front of the house, on a row of posts."""
    m.box((-half, half), (front, front + reach), (height, height + 0.015), roof)
    for k in range(max(2, round(half / 0.12)) + 1):
        u = -half + 0.02 + k * (2 * half - 0.04) / max(2, round(half / 0.12))
        m.box((u - 0.006, u + 0.006), (front + reach - 0.012, front + reach), (0.0, height), POST)


def _board(m: Model, u: float, v: float) -> None:
    """A blue name board on two posts."""
    m.box((u - 0.035, u + 0.035), (v - 0.004, v + 0.004), (0.055, 0.09), NAME_BOARD)
    for side in (-0.028, 0.028):
        m.box((u + side - 0.004, u + side + 0.004), (v - 0.004, v + 0.004), (0.0, 0.055), POST)


def country_station() -> Model:
    """A small country station: a cream house with a chimney and a canopy over the platform."""
    m = Model()
    _house(m, 0.19, 0.075, 0.15, 1)
    m.box((0.08, 0.11), (-0.02, 0.02), (0.2, 0.3), BRICK)
    _shelter(m, 0.15, 0.075, 0.07, 0.12)
    _board(m, -0.26, 0.12)
    return m


def town_station() -> Model:
    """A town station: a long two-storey house with a clock in a gable over its middle and a
    long canopy over the platform."""
    m = Model()
    _house(m, 0.29, 0.085, 0.24, 2)
    m.box((-0.07, 0.07), (0.085, 0.1), (0.0, 0.27), STATION_WALL)
    m.faces.append(
        Face([(-0.07, 0.1, 0.27), (0.07, 0.1, 0.27), (0.0, 0.1, 0.34)], STATION_WALL, (0, 1, 0))
    )
    m.panel((-0.02, 0.02), (0.19, 0.23), 0.1005, CLOCK)
    _shelter(m, 0.3, 0.1, 0.08, 0.13)
    for u in (-0.36, 0.36):
        _board(m, u, 0.14)
    return m


def grand_station() -> Model:
    """A grand terminal: a long house with a clock tower under a pointed roof, and a glass
    train shed on iron posts over the platform."""
    m = Model()
    _house(m, 0.34, 0.09, 0.28, 2)
    m.box((-0.3, -0.2), (-0.05, 0.05), (0.0, 0.56), STATION_WALL)
    m.panel((-0.275, -0.225), (0.43, 0.48), 0.0505, CLOCK)
    m.spire(-0.25, 0.06, (0.56, 0.74), STATION_ROOF)
    m.roof((-0.4, 0.4), (0.09, 0.23), (0.17, 0.24), GLASS, IRON)
    for k in range(7):
        u = -0.38 + k * 0.76 / 6
        m.box((u - 0.006, u + 0.006), (0.215, 0.227), (0.0, 0.17), IRON)
    for u in (-0.44, 0.44):
        _board(m, u, 0.16)
    return m


def turned(build: Callable[[], Model], line: str) -> Callable[[], Sprite]:
    """The sprite of a station house built by `build`, turned along track running `line`."""

    def draw() -> Sprite:
        heading = math.radians(TURNED_LINES[line])
        return Sprite(*draw_model(build(), heading, _HEX_PROJECTION.matrix, fineness()))

    return draw


TURNED_STATIONS = {
    f"{name}_{line}": turned(build, line)
    for name, build in (
        ("station", country_station),
        ("station_2", town_station),
        ("station_3", grand_station),
    )
    for line in TURNED_LINES
}
