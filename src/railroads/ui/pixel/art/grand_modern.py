"""Big and mega works of modern heavy industry, each drawn as a site of its own: refineries of
columns, cracking towers and tank farms under burning flares, chemical and plastics works of ball
tanks, silos and pipe racks, glass electronics campuses, appliance and car plants with their goods
lined up outside, and the arms works: gun shops with their proving range, armoured tank halls
round a churned-up proving ground and aircraft hangars on a runway."""

import math

import numpy as np

from railroads.ui.pixel.art.industry import (
    _band,
    _bed,
    _car,
    _chimney,
    _door,
    _drum,
    _footing,
    _oil,
    _plane,
    _plinth,
    _puddle,
    _ring,
    _sack,
    _sawtooth,
    _smoke,
    _stains,
    _tank_vehicle,
    _track,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    CAR_COLOURS,
    DOORWAY,
    GLASS,
    LEAVES,
    METAL,
    MILITARY_YARD,
    RGB,
    TRUNK,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import (
    _flag,
    _iso,
    _on_wall,
    _patch,
    _plate,
    _turned,
    _vault,
    _windows,
)

type Top = tuple[Point, Point, Point, Point]

COLUMN: RGB = (176, 178, 184)
TANK_WHITE: RGB = (224, 226, 228)
RACK: RGB = (96, 96, 104)
VALVE: RGB = (206, 60, 50)
FLAME: RGB = (246, 150, 40)
WHITE_HOT: RGB = (255, 220, 120)
STEAM: RGB = (236, 236, 240)
BALLAST: RGB = (128, 120, 108)
SLEEPER: RGB = (96, 74, 56)
RAIL: RGB = (58, 58, 64)
WAGON: RGB = (112, 112, 120)

CHEM_PAD: RGB = (156, 156, 148)
CHEM_BODY: RGB = (208, 214, 198)
CHEM_GREEN: RGB = (104, 176, 84)
FUMES: tuple[RGB, RGB] = ((146, 204, 96), (178, 222, 120))
CHEM_HOUSE: RGB = (196, 200, 186)
CHEM_ROOF: RGB = (96, 116, 84)
DRUM_BLUE: RGB = (64, 108, 196)
DRUM_YELLOW: RGB = (236, 196, 60)

PLASTIC_WHITE: RGB = (232, 234, 236)
PLASTIC_RED: RGB = (214, 58, 54)
PLASTIC_BLUE: RGB = (64, 108, 196)
PLASTIC_PAD: RGB = (170, 170, 164)


def _siding(p: Painter, u0: float, u1: float, v: float) -> None:
    """A railway siding on ballast running along the lit walls from `u0` to `u1` at `v`."""
    p.polygon([_iso(0, 0, u, v + dv) for u, dv in ((u0, -1), (u1, -1), (u1, 1), (u0, 1))], BALLAST)
    u = u0 + 0.5
    while u < u1:
        p.line([_iso(0, 0, u, v - 0.8), _iso(0, 0, u, v + 0.8)], SLEEPER)
        u += 1.4
    for dv in (-0.5, 0.5):
        p.line([_iso(0, 0, u0, v + dv), _iso(0, 0, u1, v + dv)], RAIL)


def _tank_wagon(p: Painter, u: float, v: float, colour: RGB) -> None:
    """A rail tank wagon standing on a siding at (`u`, `v`)."""
    x, y = _iso(0, 0, u, v)
    p.hall(x, y, 5, 1.6, 1, (44, 42, 46), None)
    top = p.hall(x, y - 1, 4.6, 1.8, 2.2, colour, shade(colour, 1.15))
    p.point(*_on_wall(top[0], top[3], 0.5, -0.6), shade(colour, 0.6))


def _column(p: Painter, x: float, y: float, r: float, tall: float, metal: RGB = COLUMN) -> None:
    """A tall silver distillation column standing on (x, y), ringed with walkways every few pixels
    and climbed by a ladder on its shaded side."""
    _turned(p, x, y, [(0, r), (tall, r), (tall + 1.5, r * 0.6)], metal)
    for ring in range(6, int(tall), 6):
        p.line([(x - r - 1, y - ring), (x + r + 1, y - ring)], shade(metal, 0.6))
    p.line([(x + r - 0.5, y - 1), (x + r - 0.5, y - tall)], shade(metal, 0.62))


def _oil_tank(p: Painter, x: float, y: float, r: float = 4.5, tall: float = 4) -> None:
    """A squat white storage tank standing on (x, y) with a stair up its side."""
    _turned(p, x, y, [(0, r), (tall, r)], TANK_WHITE, cap=shade(TANK_WHITE, 1.08))
    p.line([(x - r + 0.5, y - 0.5), (x - 1, y - tall - 0.5)], shade(TANK_WHITE, 0.65))


def _floating_tank(p: Painter, x: float, y: float, r: float, tall: float) -> None:
    """A wide white crude tank standing on (x, y), its floating roof sunk inside its rim and a red
    band round it."""
    _turned(p, x, y, [(0, r), (tall, r)], TANK_WHITE, cap=shade(TANK_WHITE, 0.82))
    p.blob(x - r + 1, y - tall - r / 2 + 1.2, x + r - 1, y - tall + r / 2 - 0.2, (170, 172, 176))
    _ring(p, x, y - tall * 0.5, r, VALVE)
    p.line([(x - r + 0.5, y - 0.5), (x - r * 0.3, y - tall - 0.5)], shade(TANK_WHITE, 0.62))


def _flare(p: Painter, x: float, y: float, tall: float, size: float = 1.0) -> None:
    """A tall red-and-white flare stack standing on (x, y), burning off gas in a big flame."""
    p.line([(x, y), (x, y - tall)], (90, 90, 96))
    p.line([(x + 1, y), (x + 1, y - tall)], (124, 124, 130))
    for f in (0.94, 0.82, 0.7):
        p.line([(x, y - tall * f), (x + 1, y - tall * f)], VALVE)
    s = size
    top = y - tall
    p.polygon(
        [(x - 2 * s, top - 2 * s), (x - 0.5 * s, top - 9 * s), (x + 3 * s, top - 3 * s)], FLAME
    )
    p.blob(x - 2 * s, top - 6 * s, x + 3 * s, top + 0.5, FLAME)
    p.blob(x - 0.5 * s, top - 4 * s, x + 1.5 * s, top - 0.5, WHITE_HOT)


def _rack(p: Painter, u0: float, u1: float, v: float, tall: float, pipes: tuple[RGB, ...]) -> None:
    """A pipe rack on posts running along the lit walls from `u0` to `u1` at `v`, carrying a run
    of pipes in `pipes` with red valves on them."""
    u = u0
    while u <= u1 + 0.01:
        x, y = _iso(0, 0, u, v)
        p.line([(x, y), (x, y - tall)], RACK)
        u += 4
    a, b = _iso(0, 0, u0, v), _iso(0, 0, u1, v)
    for k, colour in enumerate((RACK, *pipes)):
        h = tall - 1.3 * k
        p.line([(a[0], a[1] - h), (b[0], b[1] - h)], colour)
    for k in range(int((u1 - u0) / 6)):
        x, y = _iso(0, 0, u0 + 3 + 6 * k, v)
        p.point(x, y - tall + 0.5, VALVE)


def _cracker(p: Painter, x: float, y: float, frame_tall: float, size: float = 1.0) -> None:
    """A catalytic cracker standing on (x, y): an open steel frame with a fat regenerator and a
    tall slim reactor on top, joined by a sloping riser and climbed by stairs."""
    s = size
    w = 4 * s
    ground = {"l": (x - w, y), "f": (x, y + w / 2), "r": (x + w, y), "b": (x, y - w / 2)}

    def up(k: str, h: float) -> Point:
        gx, gy = ground[k]
        return gx, gy - h

    steel = (88, 90, 98)
    levels = [frame_tall * f for f in (0.35, 0.7, 1.0)]
    p.line([ground["b"], up("b", frame_tall)], shade(steel, 0.8))
    for h in levels:
        p.line([up("l", h), up("b", h), up("r", h)], shade(steel, 0.8))
    p.polygon(
        [up("l", frame_tall), up("b", frame_tall), up("r", frame_tall), up("f", frame_tall)],
        (120, 122, 128),
        steel,
    )
    tx, ty = up("f", frame_tall)
    ty -= w / 2
    _turned(p, tx - 1.4 * s, ty, [(0, 2.6 * s), (7 * s, 2.6 * s), (9 * s, 1.2 * s)], COLUMN)
    _ring(p, tx - 1.4 * s, ty - 3.5 * s, 2.6 * s, shade(COLUMN, 0.6))
    rx = tx + 2.2 * s
    _turned(p, rx, ty + 0.5, [(0, 1.5 * s), (16 * s, 1.5 * s), (17.5 * s, 0.8 * s)], COLUMN)
    for h in (5, 10):
        _ring(p, rx, ty + 0.5 - h * s, 1.5 * s, shade(COLUMN, 0.6))
    p.line([(tx - 1.4 * s, ty - 9 * s), (rx, ty - 14 * s)], (150, 120, 60))
    for k in "lfr":
        p.line([ground[k], up(k, frame_tall)], steel)
    for h in levels:
        p.line([up("l", h), up("f", h), up("r", h)], steel)
    p.line([up("l", 0), up("f", levels[0])], steel)
    p.line([up("f", levels[0]), up("r", levels[1])], steel)
    p.line([up("l", levels[1]), up("f", levels[2])], steel)


def _heater(p: Painter, x: float, y: float) -> None:
    """A boxy fired heater standing on (x, y), its burners glowing along its foot and two thin
    stacks on top."""
    wall = (190, 182, 170)
    top = p.box(x, y, 2.5, 7, wall, shade(wall, 1.1))
    _patch(p, top[0], top[1], (0.15, 0.85), (5.6, 6.4), FLAME)
    _band(p, top[0], top[1], 2.5, shade(wall, 0.7))
    for dx in (-1, 1):
        p.line([(x + dx, y - 7.5), (x + dx, y - 14)], (110, 110, 116))
        p.point(x + dx, y - 14.5, (70, 70, 76))


def big_refinery() -> Sprite:
    """A bigger oil refinery: a cluster of four silver distillation columns ringed with walkways, a
    catalytic cracker on its steel frame, a fired heater glowing at its foot, a pipe rack with red
    valves, a tank farm of squat white tanks behind a bund wall, a tank wagon on the siding and two
    flares burning on their tall red-and-white stacks."""
    p = Painter(52, 66, (25, 51))
    plate = (168, 164, 156)
    _plate(p, plate, 24)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-11.5, -11.5), (0, -11.5), (0, -4), (-11.5, -4))],
        shade(plate, 0.92),
        shade(plate, 0.6),
    )
    for u, v in ((3, -7), (6, -9.5), (7, 1), (-9, 4)):
        _oil(p, *_iso(0, 0, u, v), 1.2)
    for u, v, tall, size in ((11, 11, 40, 1.1), (7, 12, 32, 0.9)):
        _flare(p, *_iso(0, 0, u, v), tall, size)
    for u, v, r, h in ((-9, 6, 2, 18), (-4, 9, 2.5, 36), (-6, 4, 3, 28), (-1, 6, 2, 22)):
        _column(p, *_iso(0, 0, u, v), r, h)
    _cracker(p, *_iso(0, 0, 7, 3), 10)
    _heater(p, *_iso(0, 0, 1, -1))
    _rack(p, -11, 11, -3, 6, ((150, 120, 60), (110, 150, 180)))
    bund = shade(plate, 0.75)
    p.line(
        [_iso(0, 0, u, v) for u, v in ((-11, -11), (0, -11), (0, -4.5), (-11, -4.5), (-11, -11))],
        bund,
    )
    for u, v in ((-7.5, -6.5), (-2.5, -7), (-8, -10), (-3, -10.5)):
        _oil_tank(p, *_iso(0, 0, u, v), 3.4, 3.5)
    _siding(p, 2, 11.5, -9)
    _tank_wagon(p, 7, -9, WAGON)
    return p.sprite()


def mega_refinery() -> Sprite:
    """A great oil refinery, the showpiece of the map: a row of silver distillation columns, the
    tallest towering over it all with steam rising from it, two catalytic crackers on their steel
    frames, fired heaters glowing, three flares burning, pipe racks with red valves, a tank farm of
    wide crude tanks with floating roofs, two ball tanks of gas and a train of tank wagons on the
    siding."""
    p = Painter(62, 81, (30, 64))
    plate = (168, 164, 156)
    _plate(p, plate, 29)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-14, -14), (-1, -14), (-1, -5), (-14, -5))],
        shade(plate, 0.92),
        shade(plate, 0.6),
    )
    for u, v in ((-4, 2), (6, -6), (9, 5), (-11, 3)):
        _oil(p, *_iso(0, 0, u, v), 1.3)
    for u, v, tall, size in ((14, 13, 46, 1.2), (9, 14, 38, 1.0), (14, 8, 30, 0.8)):
        _flare(p, *_iso(0, 0, u, v), tall, size)
    for u, v, r, h in (
        (-6, 13, 2, 22),
        (-11, 10, 2.5, 30),
        (-2, 11, 2.5, 30),
        (-7, 8.5, 3.5, 42),
        (-12, 5.5, 2, 20),
        (-3, 6, 2.5, 26),
    ):
        _column(p, *_iso(0, 0, u, v), r, h)
    sx, sy = _iso(0, 0, -7, 8.5)
    _smoke(p, sx + 1, sy - 47, STEAM, 3)
    _cracker(p, *_iso(0, 0, 9, 6), 12, 1.1)
    _cracker(p, *_iso(0, 0, 4, 1), 9)
    for u in (-9, -6):
        _heater(p, *_iso(0, 0, u, 0.5))
    _rack(p, -14, 14, -3, 7, ((150, 120, 60), (110, 150, 180), (70, 70, 76)))
    for u, v in ((5, -7), (10.5, -6)):
        x, y = _iso(0, 0, u, v)
        for dx in (-2.5, 0, 2.5):
            p.line([(x + dx, y), (x + dx, y - 3)], shade(METAL, 0.7))
        p.ellipse(x - 3.5, y - 9.5, x + 3.5, y - 2.5, TANK_WHITE)
        p.blob(x - 2.4, y - 8.6, x - 0.4, y - 6.6, shade(TANK_WHITE, 1.08))
        p.line([(x - 3.5, y - 6), (x + 3.5, y - 6)], VALVE)
    bund = shade(plate, 0.75)
    p.line(
        [_iso(0, 0, u, v) for u, v in ((-14, -14), (-1, -14), (-1, -5), (-14, -5), (-14, -14))],
        bund,
    )
    for u, v in ((-10.5, -8), (-4.5, -8), (-10.5, -12), (-4.5, -12)):
        _floating_tank(p, *_iso(0, 0, u, v), 4.6, 4)
    _siding(p, 1, 14.5, -12)
    for k in range(2):
        _tank_wagon(p, 4.5 + 5.5 * k, -12, WAGON)
    return p.sprite()


def _sphere(p: Painter, x: float, y: float, r: float, legs: float) -> None:
    """A pale ball tank banded in green held up on steel legs over (x, y)."""
    leg = shade(METAL, 0.7)
    for dx in (-r * 0.6, 0, r * 0.6):
        p.line([(x + dx, y), (x + dx, y - legs - 1)], leg)
        p.point(x + dx, y, shade(METAL, 0.5))
    p.line([(x - r * 0.6, y - legs * 0.5), (x + r * 0.6, y - legs * 0.9)], shade(METAL, 0.6))
    cy = y - legs - r + 1.5
    p.ellipse(x - r, cy - r, x + r, cy + r, CHEM_BODY)
    p.blob(x - r * 0.6, cy - r * 0.7, x - r * 0.1, cy - r * 0.2, shade(CHEM_BODY, 1.15))
    p.line([(x - r, cy + 0.5), (x + r, cy + 0.5)], CHEM_GREEN)


def _green_tank(p: Painter, x: float, y: float, r: float, tall: float) -> None:
    """A pale banded chemical tank with a green top and a ladder, standing on (x, y)."""
    _turned(p, x, y, [(0, r), (tall, r)], CHEM_BODY, cap=CHEM_GREEN)
    for f in (0.3, 0.7):
        _ring(p, x, y - tall * f, r, shade(CHEM_BODY, 0.75))
    p.line([(x + r * 0.6, y - 1), (x + r * 0.6, y - tall)], shade(METAL, 0.7))
    k = 3.0
    while k < tall:
        p.line([(x + r * 0.6 - 0.5, y - k), (x + r * 0.6 + 0.5, y - k)], shade(METAL, 0.7))
        k += 3


def _vent(p: Painter, x: float, y: float, tall: float, puffs: int = 3) -> None:
    """A thin red-banded vent stack standing on (x, y) with green fumes rising from it."""
    p.rect(x, y - tall, x + 1, y, METAL)
    for f in (0.92, 0.8):
        p.line([(x, y - tall * f), (x + 1, y - tall * f)], VALVE)
    for i in range(puffs):
        r = 1.5 + 0.5 * i
        cx, cy = x + 1 + 1.5 * i, y - tall - 2 - 3 * i
        p.blob(cx - r, cy - r, cx + r, cy + r, FUMES[i % 2])


def _reactor(p: Painter, x: float, y: float, r: float, tall: float) -> None:
    """A tall pale reactor column standing on (x, y), ringed with walkways and topped with a green
    dome."""
    _turned(p, x, y, [(0, r), (tall, r), (tall + 1.5, r * 0.5)], CHEM_BODY, cap=CHEM_GREEN)
    for ring in range(5, int(tall), 5):
        p.line([(x - r - 1, y - ring), (x + r + 1, y - ring)], shade(METAL, 0.6))


def _drums(p: Painter, x: float, y: float, colours: tuple[RGB, ...]) -> None:
    """Drums of chemicals standing in a staggered row from (x, y) on a stained patch."""
    p.blob(x - 2, y - 0.5, x + 2.6 * len(colours), y + 2.5, (130, 150, 90))
    for k, colour in enumerate(colours):
        _drum(p, x + 2.4 * k, y + 1.2 * (k % 2), colour)


def _lab(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> Top:
    """A pale control house or lab block under a flat green roof with green-lit windows."""
    top = p.hall(x, y, length, depth, height, CHEM_HOUSE, CHEM_ROOF)
    left, front, right, _ = top
    _plinth(p, top, height, CHEM_HOUSE)
    _stains(p, front, right, height, (0.25, 0.75), shade(CHEM_HOUSE, 0.55), 1.2)
    rows = (1.2,) if height < 6 else (1.2, 3.6)
    for drop in rows:
        _patch(p, left, front, (0.08, 0.6), (drop, drop + 1.2), (110, 200, 120))
    _door(p, left, front, (0.7, 0.88), height - 2.8, height, DOORWAY)
    p.point(*_on_wall(front, right, 0.5, 1.5), DRUM_YELLOW)
    return top


def big_chemical_works() -> Sprite:
    """A bigger chemical works on a stained concrete pad: two pale ball tanks on legs, a tall
    reactor column with a green dome, green-topped tanks with ladders, a pipe rack of green and
    grey pipes with a red valve, a lab block with green-lit windows, drums of chemicals and two
    red-banded vent stacks breathing green fumes."""
    p = Painter(46, 60, (22, 48))
    p.polygon([(-21, -4), (2, -15), (21, -4), (-1, 8)], CHEM_PAD, shade(CHEM_PAD, 0.8))
    for x0, y0, x1, y1 in ((-8, -6, -2, -3), (10, 2, 15, 4.5), (-15, -3, -10, -1)):
        p.blob(x0, y0, x1, y1, (142, 168, 92))
    p.shadow(19, 5)
    _vent(p, 14, -6, 30)
    _vent(p, -2, -13, 22, 2)
    _sphere(p, -12, -6, 4.5, 6)
    _sphere(p, -4, -10, 4, 5)
    _reactor(p, 5, -8, 2.2, 22)
    _green_tank(p, 11, -3, 3, 12)
    _green_tank(p, 16, 0, 2.5, 9)
    leg = shade(METAL, 0.7)
    for x in (-15, -9, -3, 3):
        p.line([(x, -1), (x, -7)], leg)
    p.line([(-15, -7), (3, -7)], leg)
    p.line([(-15, -6), (9, -6)], CHEM_GREEN)
    p.line([(-15, -4.8), (9, -4.8)], shade(METAL, 0.85))
    p.ellipse(-7, -8, -5, -6, VALVE)
    _green_tank(p, 8, 3, 2.5, 8)
    _lab(p, -6, 5, 12, 5, 7)
    _track(p, *_footing(-6, 5, 12, 5)[:2], (0.7, 0.88), 3, ruts=False)
    _drums(p, -17, 6, (DRUM_BLUE, DRUM_YELLOW, DRUM_BLUE, DRUM_YELLOW))
    for x, y in ((9, 8), (11.5, 7)):
        _drum(p, x, y, DRUM_BLUE)
    return p.sprite()


def mega_chemical_works() -> Sprite:
    """A great chemical complex: a row of three pale ball tanks on legs, a steel process frame of
    reactor columns with green domes, a tank farm of green-topped tanks, a long pipe rack of green,
    yellow and grey pipes with red valves, a flare and three red-banded vent stacks breathing green
    fumes, a lab block with green-lit windows, drums of chemicals and tank wagons on the siding."""
    p = Painter(59, 65, (29, 48))
    pad = [_iso(0, 0, u, v) for u, v in ((-14, -14), (14, -14), (14, 14), (-14, 14))]
    p.polygon([(x, y + 1) for x, y in pad], shade(CHEM_PAD, 0.68))
    p.polygon(pad, CHEM_PAD, shade(CHEM_PAD, 0.8))
    for u, v in ((-6, -4), (8, 2), (-4, 6), (3, -9)):
        x, y = _iso(0, 0, u, v)
        p.blob(x - 3, y - 1.2, x + 3, y + 1.4, (142, 168, 92))
    _flare(p, *_iso(0, 0, 13, 13), 38, 0.8)
    for u, v, tall in ((8, 13, 34), (-4, 13, 28), (12, 9, 26)):
        _vent(p, *_iso(0, 0, u, v), tall)
    fx, fy = _iso(0, 0, 1, 8)
    steel = (88, 90, 98)
    for dx in (-8, 8):
        p.line([(fx + dx, fy - 3), (fx + dx, fy - 22)], shade(steel, 0.8))
    for h in (8, 15, 22):
        p.line([(fx - 8, fy - 3 - h + 3), (fx + 8, fy - 3 - h + 3)], shade(steel, 0.8))
    for dx, dy, r, tall in ((-4, 0, 2.2, 26), (4, 0, 2, 22), (0, 2, 3, 34)):
        _reactor(p, fx + dx, fy + dy, r, tall)
    for dx in (-8.5, 8.5):
        p.line([(fx + dx, fy + 3), (fx + dx, fy - 19)], steel)
    for h in (8, 15):
        p.line([(fx - 8.5, fy + 3 - h), (fx + 8.5, fy + 3 - h)], steel)
    for u, v, r, t in ((12, 3, 3, 13), (9, -1.5, 2.5, 10), (13, -2, 3, 11)):
        _green_tank(p, *_iso(0, 0, u, v), r, t)
    for k, v in enumerate((9, 3.5, -2)):
        _sphere(p, *_iso(0, 0, -12, v), 4.4 - 0.2 * k, 6)
    _rack(p, -8, 9, -0.5, 7, (CHEM_GREEN, DRUM_YELLOW, shade(METAL, 0.85)))
    _lab(p, *_iso(0, 0, 4, -7), 10, 5, 8)
    _lab(p, *_iso(0, 0, -6, -7), 5, 4, 5)
    x, y = _iso(0, 0, 10, -8)
    _drums(p, x, y, (DRUM_BLUE, DRUM_YELLOW, DRUM_BLUE, DRUM_YELLOW))
    _siding(p, -14, 14, -12.5)
    for k, colour in enumerate((CHEM_BODY, WAGON, CHEM_BODY)):
        _tank_wagon(p, -9 + 5.5 * k, -12.5, colour)
    return p.sprite()


def _silo(p: Painter, x: float, y: float, r: float, tall: float, cap: RGB) -> None:
    """A tall white pellet silo standing on (x, y) with a coloured top and dark bands."""
    _turned(p, x, y, [(0, r), (tall, r), (tall + 1.2, r * 0.6)], PLASTIC_WHITE, cap=cap)
    _ring(p, x, y - tall * 0.3, r, shade(PLASTIC_WHITE, 0.75))
    _ring(p, x, y - 0.6, r, shade(PLASTIC_WHITE, 0.62))
    _ring(p, x, y - tall * 0.7, r, shade(PLASTIC_WHITE, 0.75))


def _plastic_stack(p: Painter, x: float, y: float, tall: float) -> None:
    """A tall white stack standing on (x, y) with red bands under a dark cap."""
    p.rect(x - 1, y - tall, x + 1, y, PLASTIC_WHITE)
    for d in (2, 5):
        p.line([(x - 1, y - tall + d), (x + 1, y - tall + d)], PLASTIC_RED)
    p.rect(x - 1.5, y - tall - 1, x + 1.5, y - tall, (70, 70, 76))
    _smoke(p, x + 0.5, y - tall - 3, STEAM, 2)


def _plastics_hall(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, doors: int
) -> Top:
    """A white hall on a grey plinth under a blue roof, with a red stripe, small blue windows and
    roller doors on its end and its side."""
    top = p.hall(x, y, length, depth, height, PLASTIC_WHITE, PLASTIC_BLUE)
    left, front, right, _ = top
    _plinth(p, top, height, PLASTIC_WHITE)
    _stains(p, front, right, height, (0.15, 0.85), shade(PLASTIC_WHITE, 0.6), 1.2)
    _band(p, left, front, 1, PLASTIC_RED)
    _band(p, front, right, 1, shade(PLASTIC_RED, 0.8))
    spans = tuple((k + 0.5) / 8 for k in range(8))
    _windows(p, left, front, spans[: 8 - 2 * doors], (2.5,), 1, shade(PLASTIC_BLUE, 0.8))
    for k in range(doors):
        t0 = 1 - (k + 1) * 0.22
        _door(p, left, front, (t0, t0 + 0.14), height - 3.5, height, (70, 74, 84))
    _door(p, front, right, (0.3, 0.7), height - 3.5, height, (70, 74, 84))
    for drop in (height - 2.5, height - 1.5):
        p.line(
            [_on_wall(front, right, 0.3, drop), _on_wall(front, right, 0.7, drop)], (90, 94, 104)
        )
    p.line([left, front, right], shade(PLASTIC_BLUE, 1.3))
    return top


def _pellets(p: Painter, x: float, y: float, sacks: int) -> None:
    """White sacks of pellets piled at (x, y) with red, white and blue pellets spilt round them."""
    for k in range(sacks):
        _sack(p, x + 2.4 * k, y + 0.8 * (k % 2), (236, 236, 240))
    for k, colour in enumerate((PLASTIC_RED, PLASTIC_WHITE, PLASTIC_BLUE, PLASTIC_RED)):
        p.point(x - 1.5 + 2.2 * k, y + 1.8 - 0.4 * (k % 2), colour)


def big_plastics_works() -> Sprite:
    """A bigger plastics works: a long white hall under a blue roof with a red stripe and roller
    doors, three tall white pellet silos with red and blue tops, a white tank, red and blue pipes
    running between them, a tall red-banded stack smoking, and drums and sacks of pellets outside
    with pellets spilt round them."""
    p = Painter(45, 61, (21, 49))
    p.polygon([(-2, -4), (10, -10), (21, 0), (9, 6)], PLASTIC_PAD, shade(PLASTIC_PAD, 0.8))
    foot = _footing(-6, 3, 14, 8)
    _bed(p, foot)
    p.blob(-18, 7, -10, 9.6, (110, 116, 84))
    p.shadow(19, 5)
    _track(p, foot[1], foot[2], (0.3, 0.7), 3.5)
    _plastic_stack(p, 0, -6, 34)
    for k, (x, y) in enumerate(((7, -8), (11, -6), (15, -4))):
        _silo(p, x, y, 2.4, 18, (PLASTIC_RED, PLASTIC_BLUE)[k % 2])
    p.line([(5, -24), (17, -20)], (130, 132, 138))
    p.line([(5, -24), (2, -16), (2, -10)], PLASTIC_BLUE)
    _turned(p, 12, 2, [(0, 3), (9, 3)], PLASTIC_WHITE, cap=PLASTIC_BLUE)
    _ring(p, 12, -3, 3, shade(PLASTIC_WHITE, 0.75))
    p.line([(4, -1), (10, -1), (11, -3)], PLASTIC_RED)
    _plastics_hall(p, -6, 3, 14, 8, 7, 1)
    p.line([(6, -3), (6, -1), (9, 0)], PLASTIC_BLUE)
    for x, y, colour in ((-17, 8, PLASTIC_BLUE), (-14.5, 9, PLASTIC_RED), (-12, 8, PLASTIC_BLUE)):
        _drum(p, x, y, colour)
    _pellets(p, 7, 8.5, 4)
    return p.sprite()


def mega_plastics_works() -> Sprite:
    """A great plastics plant: a long white extrusion hall under a blue sawtooth roof with a red
    stripe and roller doors, a white cracker column, a battery of six white pellet silos with red
    and blue tops fed over a conveyor bridge, two tall red-banded stacks smoking, coloured pipes
    everywhere, a lorry at the door and drums, sacks of pellets and stacks of red, blue and yellow
    crates outside."""
    p = Painter(59, 75, (29, 58))
    pad = [_iso(0, 0, u, v) for u, v in ((-14, -14), (14, -14), (14, 14), (-14, 14))]
    p.polygon([(x, y + 1) for x, y in pad], shade(PLASTIC_PAD, 0.68))
    p.polygon(pad, PLASTIC_PAD, shade(PLASTIC_PAD, 0.8))
    for u, v in ((-4, 8), (9, -3)):
        _oil(p, *_iso(0, 0, u, v), 1.2)
    for u, v, tall in ((-4, 13, 40), (2, 13.5, 34)):
        _plastic_stack(p, *_iso(0, 0, u, v), tall)
    cx, cy = _iso(0, 0, -11, 10)
    _turned(p, cx, cy, [(0, 2.5), (28, 2.5), (30, 1.2)], PLASTIC_WHITE, cap=PLASTIC_RED)
    for ring in range(6, 28, 6):
        p.line([(cx - 3.5, cy - ring), (cx + 3.5, cy - ring)], shade(METAL, 0.6))
    silos = [(u, v) for v in (11, 7) for u in (6, 9.5, 13)]
    for k, (u, v) in enumerate(silos):
        _silo(p, *_iso(0, 0, u, v), 2.5, 20, (PLASTIC_RED, PLASTIC_BLUE)[k % 2])
    a, b = _iso(0, 0, 4, 9), _iso(0, 0, -4, 3)
    p.line([(a[0], a[1] - 22), (b[0], b[1] - 12)], (120, 122, 128))
    p.line([(a[0], a[1] - 21), (b[0], b[1] - 11)], (150, 152, 158))
    p.line([(b[0], b[1] - 12), (b[0], b[1] - 4)], (120, 122, 128))
    top = _plastics_hall(p, *_iso(0, 0, -3, 1), 24, 9, 9, 2)
    _sawtooth(p, top, 5, 2.5, PLASTIC_BLUE, (190, 214, 236), PLASTIC_WHITE)
    for u, v, colour in ((8, 2, PLASTIC_RED), (8, 0.5, PLASTIC_BLUE)):
        x0, y0 = _iso(0, 0, u, v)
        x1, y1 = _iso(0, 0, 13, v + 2)
        p.line([(x0, y0 - 3), (x1, y1 - 3), (x1, y1 - 6)], colour)
    x, y = _iso(0, 0, 11, -3)
    _turned(p, x, y, [(0, 3), (9, 3)], PLASTIC_WHITE, cap=PLASTIC_BLUE)
    _ring(p, x, y - 4.5, 3, shade(PLASTIC_WHITE, 0.75))
    trailer = _iso(0, 0, 2.5, -7.5)
    p.hall(trailer[0], trailer[1], 2.6, 6, 3.5, PLASTIC_WHITE, shade(PLASTIC_WHITE, 1.02))
    cab = _iso(0, 0, 2.5, -11.5)
    cab_top = p.hall(cab[0], cab[1], 2.6, 2, 3, PLASTIC_RED, shade(PLASTIC_RED, 1.15))
    _patch(p, cab_top[0], cab_top[1], (0.1, 0.9), (0.5, 1.5), (150, 190, 220))
    for k, colour in enumerate((PLASTIC_RED, PLASTIC_BLUE, DRUM_YELLOW, (60, 160, 100))):
        x, y = _iso(0, 0, -12 + 2.6 * k, -9)
        for lift in (0, 1.6):
            p.box(x, y - lift, 1.2, 1.6, colour, shade(colour, 1.2))
    x, y = _iso(0, 0, -3, -12)
    _pellets(p, x, y, 4)
    x, y = _iso(0, 0, 10, -7)
    for k, colour in enumerate((PLASTIC_BLUE, PLASTIC_RED, PLASTIC_BLUE)):
        _drum(p, x + 2.2 * k, y + 1.1 * (k % 2), colour)
    return p.sprite()


def _lorry(p: Painter, x: float, y: float, body: RGB, cab: RGB) -> None:
    """A lorry backed up to a door, its box trailer centred on (x, y) and its cab out in front."""
    p.hall(x, y, 2.6, 6, 3.5, body, shade(body, 1.04))
    cx, cy = _iso(x, y, 0, -4)
    top = p.hall(cx, cy, 2.6, 2, 3, cab, shade(cab, 1.15))
    _patch(p, top[0], top[1], (0.1, 0.9), (0.5, 1.5), (150, 190, 220))


def _tree(p: Painter, x: float, y: float, k: int = 0) -> None:
    """A round leafy tree standing on (x, y)."""
    dark, mid, light = LEAVES[k % len(LEAVES)]
    p.line([(x, y), (x, y - 2)], TRUNK)
    p.blob(x - 2.2, y - 6, x + 2.2, y - 1.5, dark)
    p.blob(x - 1.8, y - 6, x + 1.2, y - 2.6, mid)
    p.point(x - 1, y - 5, light)


def _cars(p: Painter, u0: float, v: float, count: int, k: int = 0) -> None:
    """A row of `count` new cars parked along the lit walls from (`u0`, `v`) in bright colours."""
    for n in range(count):
        _car(p, *_iso(0, 0, u0 + 3.4 * n, v), CAR_COLOURS[(n + k) % len(CAR_COLOURS)])


EL_WALL: RGB = (226, 232, 236)
EL_GLASS: RGB = (84, 176, 206)
EL_ACCENT: RGB = (80, 216, 222)
EL_ROOF: RGB = (172, 180, 190)
PAVING: RGB = (186, 188, 184)
LAWN: RGB = (86, 150, 64)
DISH: RGB = (236, 238, 240)
TOWER_GLASS: RGB = (118, 194, 222)


def _glass_bands(p: Painter, top: Top, height: float, bands: int) -> None:
    """Bands of cyan glass with light mullions right round a white block `height` tall, and a cyan
    accent line under the top band."""
    left, front, right, _ = top
    step = (height - 1.5) / bands
    for k in range(bands):
        d0 = 1.5 + step * k
        d1 = d0 + step * 0.55
        _patch(p, left, front, (0.04, 0.96), (d0, d1), EL_GLASS)
        _patch(p, front, right, (0.06, 0.94), (d0, d1), shade(EL_GLASS, 0.72))
        for t in (0.2, 0.4, 0.6, 0.8):
            p.line(
                [_on_wall(left, front, t, d0), _on_wall(left, front, t, d1)], shade(EL_GLASS, 1.3)
            )
        p.line([_on_wall(front, right, 0.5, d0), _on_wall(front, right, 0.5, d1)], EL_GLASS)
        p.line([_on_wall(left, front, 0.1, d0), _on_wall(left, front, 0.2, d0)], (200, 240, 250))
    accent = 1.5 + step * 0.55 + 0.6
    p.line([_on_wall(left, front, 0.04, accent), _on_wall(left, front, 0.96, accent)], EL_ACCENT)
    p.line([left, front, right], shade(EL_WALL, 1.1))


def _entrance(p: Painter, left: Point, front: Point, t: float, height: float) -> None:
    """A dark glass entrance under a canopy with a lit cyan sign above it, at `t` along the lit
    wall of a block `height` tall."""
    _patch(p, left, front, (t - 0.08, t + 0.08), (height - 1.6, height), (52, 70, 84))
    p.line(
        [
            _on_wall(left, front, t - 0.11, height - 1.9),
            _on_wall(left, front, t + 0.11, height - 1.9),
        ],
        (120, 130, 140),
    )
    _patch(p, left, front, (t - 0.1, t + 0.1), (height - 4.6, height - 3.6), EL_ACCENT)


def _dish(p: Painter, x: float, y: float, s: float = 1.0) -> None:
    """A white satellite dish on a post standing on (x, y), tilted to the sky."""
    p.line([(x, y), (x, y - 3 * s)], METAL)
    p.ellipse(x - 4 * s, y - 8 * s, x + 3 * s, y - 3 * s, DISH)
    p.blob(x - 2 * s, y - 7 * s, x + 1 * s, y - 5 * s, (176, 186, 196))
    p.point(x - 0.5 * s, y - 6 * s, EL_ACCENT)


def _mast(p: Painter, x: float, y: float, tall: float) -> None:
    """A thin aerial mast standing on (x, y) with crossbars and a red light on top."""
    p.line([(x, y), (x, y - tall)], METAL)
    for f in (0.55, 0.8):
        half = 2 * (1 - f) + 0.8
        p.line([(x - half, y - tall * f), (x + half, y - tall * f)], METAL)
    p.point(x, y - tall - 1, (230, 60, 60))


def _plant_boxes(p: Painter, top: Top, spots: tuple[tuple[float, float], ...]) -> None:
    """Grey air-handling boxes with round fans on a flat roof, at `spots` (along, across) on it."""
    left, front, right, back = top
    for along, across in spots:
        x, y = _on_wall(
            _on_wall(left, back, across, 0), _on_wall(front, right, across, 0), along, 0
        )
        p.box(x, y, 1.6, 1.4, (190, 198, 206), (214, 220, 226))
        p.blob(x - 0.8, y - 2, x + 0.8, y - 1.2, (110, 120, 130))


def big_electronics_plant() -> Sprite:
    """A bigger electronics plant: a tall white block wrapped in three bands of cyan glass with a
    lit sign and an entrance under a canopy, a long low clean-room wing behind with fans on its
    roof, a dish and an aerial on top, flags flying on a trimmed lawn with trees, and staff cars
    parked by it."""
    p = Painter(48, 42, (24, 30))
    foot = _footing(-2, 2, 18, 10)
    _bed(p, foot, PAVING, 1.6)
    _bed(p, _footing(10, -5, 12, 7), PAVING, 1.6)
    p.blob(-21, 2, -9, 9, LAWN)
    p.shadow(18, 5)
    _track(p, foot[0], foot[1], (0.35, 0.55), 4, PAVING, ruts=False)
    for k, (x, y) in enumerate(((-19, 4), (-14, 1.5))):
        _tree(p, x, y, k)
    wing = p.hall(10, -5, 12, 7, 7, EL_WALL, EL_ROOF)
    _plinth(p, wing, 7, EL_WALL)
    _glass_bands(p, wing, 7, 1)
    _plant_boxes(p, wing, ((0.25, 0.5), (0.55, 0.5), (0.85, 0.5)))
    top = p.hall(-2, 2, 18, 10, 12, EL_WALL, EL_ROOF)
    left, front, _, _ = top
    _plinth(p, top, 12, EL_WALL)
    _glass_bands(p, top, 12, 3)
    _entrance(p, left, front, 0.45, 12)
    _dish(p, -10, -14, 1.2)
    _mast(p, -1, -12, 16)
    _plant_boxes(p, top, ((0.7, 0.35), (0.85, 0.6)))
    for k, (x, y) in enumerate(((-17, 7), (-14, 8.5), (-11, 10))):
        _flag(p, x, y, 10, (EL_ACCENT, DRUM_BLUE, EL_ACCENT)[k])
    _tree(p, -20, 8, 2)
    for k, (x, y) in enumerate(((11, 8.5), (14.5, 6.8), (18, 5))):
        _car(p, x, y, CAR_COLOURS[k + 1])
    return p.sprite()


def mega_electronics_plant() -> Sprite:
    """A great electronics campus: a tall tower clad all over in cyan glass with a lit crown and an
    aerial mast, two long white clean-room wings banded in glass with rows of fans on their roofs
    and a big dish, a lawn with a pond, trees and a row of flags, and a car park of staff cars."""
    p = Painter(59, 71, (29, 54))
    plate = [_iso(0, 0, u, v) for u, v in ((-14, -14), (14, -14), (14, 14), (-14, 14))]
    p.polygon([(x, y + 1) for x, y in plate], shade(PAVING, 0.7))
    p.polygon(plate, PAVING, shade(PAVING, 0.85))
    p.polygon([_iso(0, 0, u, v) for u, v in ((-13, -13), (-1, -13), (-1, -6), (-13, -6))], LAWN)
    x, y = _iso(0, 0, -6, -10)
    p.blob(x - 4, y - 1.5, x + 4, y + 2, (98, 126, 146))
    p.line([(x - 2, y), (x + 1, y - 0.5)], (170, 200, 220))
    tower = p.box(*_iso(0, 0, -4, 8), 6, 34, TOWER_GLASS, EL_ROOF)
    left, front, right, _ = tower
    for t in (0.2, 0.4, 0.6, 0.8):
        p.line([_on_wall(left, front, t, 0), _on_wall(left, front, t, 34)], shade(TOWER_GLASS, 1.2))
        p.line(
            [_on_wall(front, right, t, 0), _on_wall(front, right, t, 34)], shade(TOWER_GLASS, 0.9)
        )
    for drop in range(5, 34, 3):
        _band(p, left, front, drop, shade(TOWER_GLASS, 0.85))
        _band(p, front, right, drop, shade(TOWER_GLASS, 0.66))
    _patch(p, left, front, (0, 1), (0, 2.5), EL_WALL)
    _patch(p, front, right, (0, 1), (0, 2.5), shade(EL_WALL, 0.78))
    _patch(p, left, front, (0.25, 0.75), (0.6, 1.9), EL_ACCENT)
    p.line([(left[0] + 3, left[1] + 6), (left[0] + 5, left[1] + 16)], (220, 246, 252))
    _mast(p, *_on_wall(left, right, 0.5, 0), 12)
    wing_a = p.hall(*_iso(0, 0, 8, 7), 12, 9, 9, EL_WALL, EL_ROOF)
    _plinth(p, wing_a, 9, EL_WALL)
    _glass_bands(p, wing_a, 9, 2)
    _dish(p, *_on_wall(_on_wall(wing_a[0], wing_a[3], 0.5, 0), wing_a[1], 0.4, 0), 1.4)
    _plant_boxes(p, wing_a, ((0.75, 0.3), (0.75, 0.7)))
    wing_b = p.hall(*_iso(0, 0, 2, -1), 22, 8, 8, EL_WALL, EL_ROOF)
    lb, fb, _, _ = wing_b
    _plinth(p, wing_b, 8, EL_WALL)
    _glass_bands(p, wing_b, 8, 2)
    _entrance(p, lb, fb, 0.3, 8)
    _plant_boxes(p, wing_b, tuple((t, 0.5) for t in (0.15, 0.35, 0.55, 0.75, 0.92)))
    for k, (u, v) in enumerate(((-12, -6), (-9, -13), (-3, -13.5), (-13, -11))):
        _tree(p, *_iso(0, 0, u, v), k)
    for k in range(4):
        _flag(p, *_iso(0, 0, -11 + 2.6 * k, -4.6), 11, (EL_ACCENT, DRUM_BLUE)[k % 2])
    _cars(p, 2, -7, 4)
    _cars(p, 2, -10.5, 4, 2)
    _cars(p, 3, -13.5, 3, 1)
    return p.sprite()


AP_WALL: RGB = (216, 220, 224)
AP_STRIPE: RGB = (64, 116, 196)
AP_APRON: RGB = (180, 180, 174)
AP_DOOR: RGB = (170, 174, 180)
AP_GLASS: RGB = (110, 160, 200)
AP_ROOF: RGB = (150, 160, 174)
GOODS: RGB = (244, 246, 248)
HAZARD: RGB = (232, 196, 60)


def _roller_door(
    p: Painter, left: Point, front: Point, span: tuple[float, float], height: float
) -> None:
    """A big grey roller door with yellow posts either side, on the lit wall of a hall `height`
    tall."""
    _patch(
        p,
        left,
        front,
        (span[0] - 0.02, span[1] + 0.02),
        (height - 5, height - 4.5),
        shade(AP_WALL, 0.7),
    )
    _patch(p, left, front, span, (height - 4.5, height), AP_DOOR)
    for k in range(1, 4):
        drop = height - 4.5 + 1.1 * k
        p.line(
            [_on_wall(left, front, span[0], drop), _on_wall(left, front, span[1], drop)],
            shade(AP_DOOR, 0.7),
        )
    for t in (span[0] - 0.02, span[1] + 0.02):
        p.line([_on_wall(left, front, t, height - 1.5), _on_wall(left, front, t, height)], HAZARD)


def _appliance_hall(
    p: Painter, x: float, y: float, length: float, depth: float, height: float
) -> Top:
    """A long modern factory hall on a grey plinth with a blue stripe round it."""
    top = p.hall(x, y, length, depth, height, AP_WALL, AP_ROOF)
    left, front, right, _ = top
    _plinth(p, top, height, AP_WALL)
    _stains(p, front, right, height, (0.2, 0.5, 0.8), shade(AP_WALL, 0.62), 1.2)
    _band(p, left, front, 1, AP_STRIPE)
    _band(p, front, right, 1, shade(AP_STRIPE, 0.8))
    p.line([left, front, right], shade(AP_WALL, 1.12))
    return top


def _white_good(p: Painter, x: float, y: float, tall: float, washer: bool) -> None:
    """A white fridge, or a washing machine with its round dark window, standing on (x, y)."""
    box = p.box(x, y, 2.5, tall, GOODS, shade(GOODS, 0.9))
    if washer:
        cx, cy = _on_wall(box[0], box[1], 0.5, 2.5)
        p.blob(cx - 1, cy - 1, cx + 1, cy + 1, (64, 88, 118))
        p.point(cx - 0.5, cy - 0.5, (150, 176, 200))
    else:
        p.line([_on_wall(box[0], box[1], 0, 2), _on_wall(box[0], box[1], 1, 2)], shade(GOODS, 0.55))
        p.point(*_on_wall(box[0], box[1], 0.85, 3.5), shade(GOODS, 0.5))


def _goods_stack(p: Painter, x: float, y: float) -> None:
    """Fridges and washing machines stacked on the ground from (x, y)."""
    for dx, dy, tall, washer in (
        (6, -3, 6, False),
        (3, 0, 4, True),
        (3, -4, 4, True),
        (0, 3, 4, True),
    ):
        _white_good(p, x + dx, y + dy, tall, washer)


def _billboard(p: Painter, x: float, y: float, s: float = 1.0) -> None:
    """A big blue board on legs standing on a roof at (x, y), facing the way the lit walls do, with
    a white fridge and a white washing machine painted on it."""

    def at(t: float, h: float) -> Point:
        return x + t * s, y + t * s / 2 - h

    for t in (-3, 3):
        p.line([at(t, 0), at(t, 2.5)], (90, 96, 106))
    p.polygon(
        [at(-5, 2), at(5, 2), at(5, 2 + 7 * s), at(-5, 2 + 7 * s)], AP_STRIPE, shade(AP_STRIPE, 0.6)
    )
    p.polygon([at(-3.5, 3), at(-1, 3), at(-1, 3 + 5 * s), at(-3.5, 3 + 5 * s)], GOODS)
    p.line([at(-3.5, 3 + 3.4 * s), at(-1, 3 + 3.4 * s)], shade(AP_STRIPE, 0.8))
    p.polygon([at(0.5, 3), at(3.5, 3), at(3.5, 3 + 3.6 * s), at(0.5, 3 + 3.6 * s)], GOODS)
    cx, cy = at(2, 3 + 1.6 * s)
    p.blob(cx - 0.9 * s, cy - 0.9 * s, cx + 0.9 * s, cy + 0.9 * s, (64, 88, 118))


def big_appliance_factory() -> Sprite:
    """A bigger appliance works on a concrete apron: a long white factory hall with a blue stripe,
    two roller doors and a row of windows, a tall warehouse behind with a big blue sign, plant
    boxes on its roof, a lorry backed up to a door and fridges and washing machines stacked
    outside in white."""
    p = Painter(47, 40, (23, 27))
    foot = _footing(-1, 0, 22, 9)
    _bed(p, foot, AP_APRON, 2.4)
    _bed(p, foot, (120, 120, 116), 0.7)
    _bed(p, _footing(10, -7, 10, 8), AP_APRON, 1.2)
    p.shadow(19, 5)
    _track(p, foot[0], foot[1], (0.55, 0.95), 6, AP_APRON)
    _oil(p, -6, 5, 1.2)
    store = _appliance_hall(p, 10, -7, 10, 8, 14)
    _windows(p, store[0], store[1], (0.2, 0.4, 0.6, 0.8), (3, 7), 2, AP_GLASS)
    _patch(p, store[0], store[1], (0.1, 0.9), (9.5, 12), AP_STRIPE)
    p.line(
        [_on_wall(store[0], store[1], 0.2, 10.8), _on_wall(store[0], store[1], 0.8, 10.8)], GOODS
    )
    top = _appliance_hall(p, -1, 0, 22, 9, 8)
    left, front, right, _ = top
    for span in ((0.55, 0.72), (0.78, 0.95)):
        _roller_door(p, left, front, span, 8)
    spans = (0.06, 0.15, 0.24, 0.33, 0.42)
    _windows(p, left, front, spans, (3,), 2, AP_GLASS)
    _windows(p, left, front, spans, (5.4,), 1, shade(AP_WALL, 1.1))
    _windows(p, front, right, (0.3, 0.6), (3,), 2, shade(AP_GLASS, 0.8))
    _plant_boxes(p, top, ((0.55, 0.5), (0.75, 0.5)))
    _billboard(
        p, *_on_wall(_on_wall(left, top[3], 0.5, 0), _on_wall(front, right, 0.5, 0), 0.22, 0)
    )
    _lorry(p, *_iso(*_on_wall(foot[0], foot[1], 0.865, 0), 0, -4), GOODS, AP_STRIPE)
    _goods_stack(p, 10, 2)
    _white_good(p, 18, 3, 4, False)
    return p.sprite()


def mega_appliance_factory() -> Sprite:
    """A great appliance plant on a wide apron: a long white factory hall with a blue stripe and
    three roller doors, a high-bay warehouse behind it, a big rooftop board painted with a fridge
    and a washing machine, plant boxes on the roofs, two lorries backed up to the doors and rows
    of fridges and washing machines lined up outside in white."""
    p = Painter(59, 46, (29, 29))
    _plate(p, AP_APRON, 28)
    for u, v in ((-6, -9), (4, -11)):
        _oil(p, *_iso(0, 0, u, v), 1.2)
    store = _appliance_hall(p, *_iso(0, 0, 4, 9), 18, 9, 18)
    sl, sf, sr, _ = store
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        p.line([_on_wall(sl, sf, t, 2), _on_wall(sl, sf, t, 16)], shade(AP_WALL, 0.88))
    _patch(p, sl, sf, (0.08, 0.92), (3, 6), AP_STRIPE)
    p.line([_on_wall(sl, sf, 0.2, 4.5), _on_wall(sl, sf, 0.8, 4.5)], GOODS)
    _windows(p, sf, sr, (0.3, 0.7), (4, 9, 14), 2, shade(AP_GLASS, 0.8))
    top = _appliance_hall(p, *_iso(0, 0, -1, 0), 28, 10, 9)
    left, front, right, back = top
    for span in ((0.5, 0.64), (0.68, 0.82), (0.86, 0.98)):
        _roller_door(p, left, front, span, 9)
    spans = (0.05, 0.13, 0.21, 0.29, 0.37, 0.45)
    _windows(p, left, front, spans, (3,), 2, AP_GLASS)
    _windows(p, left, front, spans, (5.8,), 1, shade(AP_WALL, 1.1))
    _windows(p, front, right, (0.25, 0.5, 0.75), (3,), 2, shade(AP_GLASS, 0.8))
    _plant_boxes(p, top, ((0.6, 0.35), (0.6, 0.7), (0.8, 0.35), (0.8, 0.7)))
    _billboard(
        p, *_on_wall(_on_wall(left, back, 0.5, 0), _on_wall(front, right, 0.5, 0), 0.25, 0), 1.3
    )
    ground = _footing(*_iso(0, 0, -1, 0), 28, 10)
    for t in (0.75, 0.92):
        _lorry(p, *_iso(*_on_wall(ground[0], ground[1], t, 0), 0, -4), GOODS, AP_STRIPE)
    for v, tall, washer in ((-8.5, 6, False), (-12.5, 4, True)):
        for k in range(4):
            _white_good(p, *_iso(0, 0, -13 + 2.5 * k, v), tall, washer)
    x, y = _iso(0, 0, 11, -9)
    _goods_stack(p, x - 4, y)
    return p.sprite()


CAR_LOT: RGB = (166, 166, 162)
CAR_WALL: RGB = (156, 160, 170)
CAR_VAULT: RGB = (98, 108, 130)
CAR_SAW: RGB = (104, 110, 126)
CAR_SIGN: RGB = (196, 58, 54)


def _assembly_hall(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, sign: bool
) -> Top:
    """A long car assembly hall on a dark plinth under an arched roof, with a band of mullioned
    glass along it and a wide dark door in its end under a red sign if `sign`."""
    top = p.hall(x, y, length, depth, height, CAR_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, height, CAR_WALL)
    _patch(p, left, front, (0.03, 0.97), (1.5, 3), (96, 136, 168))
    steps = round(length / 2.6)
    for k in range(1, steps):
        t = 0.03 + 0.94 * k / steps
        p.line([_on_wall(left, front, t, 1.5), _on_wall(left, front, t, 3)], (150, 184, 206))
    _patch(p, front, right, (0.2, 0.8), (height - 5.5, height), (64, 66, 74))
    p.line(
        [_on_wall(front, right, 0.5, height - 5.5), _on_wall(front, right, 0.5, height)],
        (84, 86, 96),
    )
    _vault(p, top, depth * 0.7, CAR_VAULT, CAR_WALL)
    if sign:
        sx, sy = _on_wall(front, right, 0.5, -2.4)
        p.rect(sx - 3, sy - 1.2, sx + 3, sy + 1.2, CAR_SIGN)
        p.line([(sx - 2, sy), (sx + 2, sy)], (244, 240, 230))
    return top


def _press_shop(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> Top:
    """A press shop under a glazed sawtooth roof, with windows in its end."""
    wall = shade(CAR_WALL, 0.92)
    top = p.hall(x, y, length, depth, height, wall, None)
    _plinth(p, top, height, wall)
    _windows(p, top[1], top[2], (0.3, 0.7), (2,), 2, shade(GLASS, 0.8))
    _sawtooth(p, top, round(length / 3.5), 3, CAR_SAW, (160, 196, 218), wall)
    return top


def _paint_shop(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> None:
    """A tall windowless paint shop with two thin white stacks steaming on its roof."""
    wall = (200, 204, 210)
    top = p.hall(x, y, length, depth, height, wall, shade(wall, 0.8))
    _plinth(p, top, height, wall)
    _band(p, top[0], top[1], 1.2, CAR_SIGN)
    _band(p, top[1], top[2], 1.2, shade(CAR_SIGN, 0.8))
    for t in (0.35, 0.7):
        sx, sy = _on_wall(_on_wall(top[0], top[3], 0.5, 0), _on_wall(top[1], top[2], 0.5, 0), t, 0)
        p.rect(sx - 0.6, sy - 9, sx + 0.6, sy, (236, 236, 236))
        p.line([(sx - 0.6, sy - 7.5), (sx + 0.6, sy - 7.5)], CAR_SIGN)
        _smoke(p, sx + 0.5, sy - 11, STEAM, 2)


def big_car_factory() -> Sprite:
    """A bigger car works filling its cell: a long assembly hall under an arched roof with a band
    of mullioned glass and a red sign on its end, a sawtoothed press shop and a tall paint shop
    with steaming stacks behind, and three rows of new cars parked on the lot out front."""
    p = Painter(51, 49, (25, 34))
    _plate(p, CAR_LOT, 24)
    under = shade(CAR_LOT, 0.76)
    _bed(p, _footing(*_iso(0, 0, 5, 9.5), 12, 5), under, 0.8)
    _bed(p, _footing(*_iso(0, 0, -6, 9), 9, 5), under, 0.8)
    _bed(p, _footing(*_iso(0, 0, 0, 2), 24, 8), under, 0.8)
    for u in (-12, -8.5, -5, -1.5, 2, 5.5, 9, 12):
        p.line([_iso(0, 0, u, -4.4), _iso(0, 0, u, -11.8)], shade(CAR_LOT, 1.1))
    for u, v in ((-4.8, -7), (2.4, -5.6), (9, -9)):
        _oil(p, *_iso(0, 0, u, v), 1.1)
    _paint_shop(p, *_iso(0, 0, -6, 9), 9, 5, 9)
    _press_shop(p, *_iso(0, 0, 5, 9.5), 12, 5, 7)
    _assembly_hall(p, *_iso(0, 0, 0, 2), 24, 8, 8, True)
    p.line([_iso(0, 0, -12, -3.6), _iso(0, 0, 12, -3.6)], shade(CAR_LOT, 1.15))
    for row, v in enumerate((-5.5, -8.3, -11.1)):
        _cars(p, -11 + row, v, 7 - row, 2 * row)
    return p.sprite()


def _oval(cu: float, cv: float, ru: float, rv: float) -> list[Point]:
    """Points round an oval on the ground centred at (`cu`, `cv`) with half-axes `ru` and `rv`."""
    return [
        _iso(0, 0, cu + ru * math.cos(a), cv + rv * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, 25)
    ]


def mega_car_factory() -> Sprite:
    """A great car plant: two long assembly halls under arched roofs with bands of mullioned glass
    and a red sign, a sawtoothed press shop and a paint shop with steaming stacks behind, a tall
    red sign on a pylon, an oval test track with cars lapping it, and a big car park full of new
    cars."""
    p = Painter(61, 57, (30, 40))
    _plate(p, CAR_LOT, 29)
    cu, cv = 7.5, -8.6
    p.polygon(_oval(cu, cv, 6.4, 5.4), (84, 86, 92))
    p.polygon(_oval(cu, cv, 3.8, 2.8), (118, 150, 70))
    mid = _oval(cu, cv, 5.1, 4.1)
    for k in range(0, 24, 2):
        p.line([mid[k], mid[k + 1]], (230, 230, 226))
    for u in (-14, -11, -8, -5, -2):
        p.line([_iso(0, 0, u, -3.8), _iso(0, 0, u, -14)], shade(CAR_LOT, 1.1))
    under = shade(CAR_LOT, 0.76)
    halls = (((-8, 11.5), 10, 5), ((5, 12), 16, 5), ((0, 6.5), 26, 6), ((0, 0), 26, 6))
    for (u, v), length, depth in halls:
        _bed(p, _footing(*_iso(0, 0, u, v), length, depth), under, 0.8)
    _paint_shop(p, *_iso(0, 0, -8, 11.5), 10, 5, 12)
    _press_shop(p, *_iso(0, 0, 5, 12), 16, 5, 9)
    _assembly_hall(p, *_iso(0, 0, 0, 6.5), 26, 6, 11, False)
    _assembly_hall(p, *_iso(0, 0, 0, 0), 26, 6, 8, True)
    for row, v in enumerate((-5.4, -8.2, -11, -13.6)):
        _cars(p, -13.5 + 0.5 * (row % 2), v, 4, row)
    px, py = _iso(0, 0, -0.5, -4.5)
    p.line([(px, py), (px, py - 15)], (90, 92, 100))
    p.line([(px + 1, py), (px + 1, py - 15)], (130, 132, 140))
    p.rect(px - 3, py - 21, px + 4, py - 15, CAR_SIGN)
    p.line([(px - 1.5, py - 18), (px + 2.5, py - 18)], (244, 240, 230))
    for k, colour in ((1, CAR_COLOURS[0]), (8, CAR_COLOURS[2]), (15, CAR_COLOURS[1])):
        _car(p, *mid[k], colour)
    return p.sprite()


GUN_WALL: RGB = (120, 104, 78)
GUN_ROOF: RGB = (84, 74, 58)
GUN_SIGN: RGB = (190, 56, 48)
GUN_STEEL: RGB = (92, 96, 104)
GUN_OLIVE: RGB = (96, 102, 70)
AMMO: RGB = (140, 122, 80)
EARTH: RGB = (130, 108, 76)
GLAZING: RGB = (160, 196, 218)
WINDOW_DARK: RGB = (58, 70, 92)
CRANE: RGB = (214, 168, 52)


def _gun_hall(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> Top:
    """A drab works hall on a dark plinth, its flat roof behind a parapet with roof lights, rows of
    windows with sills and a loading door under a lintel."""
    top = p.hall(x, y, length, depth, height, GUN_WALL, GUN_ROOF)
    left, front, right, back = top
    _plinth(p, top, height, GUN_WALL)
    _stains(p, left, front, height, (0.06, 0.3, 0.62, 0.88), shade(GUN_WALL, 0.6))
    _band(p, left, front, 1, shade(GUN_WALL, 1.15))
    _band(p, front, right, 1, shade(GUN_WALL, 0.9))
    spans = tuple(0.08 + k * 0.1 for k in range(9))
    _windows(p, left, front, spans, (2.5,), 2, WINDOW_DARK, shade(GUN_WALL, 1.25))
    _windows(p, left, front, spans, (height - 3.2,), 1, shade(GUN_WALL, 1.15))
    _door(p, front, right, (0.25, 0.6), height - 4.5, height, DOORWAY, shade(GUN_WALL, 1.25))
    p.line([left, front, right], shade(GUN_ROOF, 1.3))
    middle = _on_wall(left, back, 0.5, 0), _on_wall(front, right, 0.5, 0)
    for t in (0.3, 0.65):
        p.hall(*_on_wall(*middle, t, 0), length * 0.25, 2, 0.8, shade(GUN_ROOF, 0.8), GLAZING)
    return top


def _gun_shop(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> Top:
    """A tall gun shop under a pitched roof with two rows of windows and its great end door open
    on the dark inside."""
    wall = shade(GUN_WALL, 0.92)
    top = p.hall(x, y, length, depth, height, wall, None)
    left, front, right, _ = top
    _plinth(p, top, height, wall)
    spans = tuple(0.08 + k * (0.84 / 7) for k in range(8))
    _windows(p, left, front, spans, (2.5, height * 0.5), 3, WINDOW_DARK, shade(GUN_WALL, 1.2))
    _patch(p, front, right, (0.18, 0.82), (height * 0.35, height), (44, 40, 36))
    p.gable(top, depth * 0.7, GUN_ROOF, wall)
    return top


def _gun_sign(p: Painter, x: float, y: float, s: float = 1.0) -> None:
    """A framed red sign with a white bar on it, standing on a roof at (x, y)."""
    p.rect(x - 3.5 * s, y - 3.5 * s, x + 3.5 * s, y, shade(GUN_SIGN, 0.6))
    p.rect(x - 3 * s, y - 3 * s, x + 3 * s, y - 0.5, GUN_SIGN)
    p.line([(x - 2 * s, y - 2 * s), (x + 2 * s, y - 1.5 * s)], (240, 240, 236))


def _barrels(p: Painter, x: float, y: float, rows: int, long: float = 7) -> None:
    """Long gun barrels stacked `rows` high on timber bearers at (x, y), lying along the lit walls
    with their dark muzzles towards the front."""
    for u in (0.8, long - 0.8):
        p.line([_iso(x, y, u, -0.5), _iso(x, y, u, 1.2 * (4 - 1) + 0.5)], SLEEPER)
    for row in range(rows):
        for k in range(4 - row):
            v = 1.2 * k + 0.6 * row
            bx, by = _iso(x, y, 0, v)
            fx, fy = _iso(x, y, long, v)
            lift = 1 + 1.1 * row
            p.line([(bx, by - lift), (fx, fy - lift)], GUN_STEEL)
            p.line([(bx, by - lift - 0.6), (fx - 1, fy - lift - 1)], shade(GUN_STEEL, 1.35))
            p.point(fx, fy - lift - 0.5, (34, 34, 38))


def _field_gun(p: Painter, x: float, y: float, s: float = 1.0) -> Point:
    """A field gun in olive drab standing on (x, y), its trail on the ground behind it, a shield
    over its two wheels and its long barrel laid along the lit walls; returns its muzzle."""
    trail = shade(GUN_OLIVE, 0.7)
    p.line([_iso(x, y, -4 * s, 0), (x, y - 1)], trail)
    p.line([_iso(x, y, -4 * s, 0.6), (x + 0.5, y - 1.3)], trail)
    for dv in (-1.4, 1.4):
        wx, wy = _iso(x, y, 0, dv * s)
        p.ellipse(wx - 1.1 * s, wy - 2.2 * s, wx + 1.1 * s, wy + 0.2, (52, 46, 40))
        p.point(wx, wy - 1, (130, 120, 96))
    mx, my = _iso(x, y, 7 * s, 0)
    my -= 2.5 * s + 1.2 * s
    p.line([(x, y - 2.5 * s), (mx, my)], GUN_STEEL)
    p.line([(x, y - 3.2 * s), (mx, my - 0.6)], shade(GUN_STEEL, 1.3))
    p.polygon(
        [
            (x - 1.2 * s, y - 4.5 * s),
            (x + 1.2 * s, y - 3.5 * s),
            (x + 1.2 * s, y - 1),
            (x - 1.2 * s, y - 2),
        ],
        GUN_OLIVE,
        shade(GUN_OLIVE, 0.6),
    )
    return mx, my


def _ammo(p: Painter, x: float, y: float) -> None:
    """Ammunition crates stacked two high on (x, y)."""
    for dx, dy, lift in ((0, 0, 0), (3, 1.5, 0), (1.5, 0.75, 1.6)):
        p.box(x + dx, y + dy - lift, 1.5, 1.6, AMMO, shade(AMMO, 1.2))


def big_gun_factory() -> Sprite:
    """A bigger gun works: a long drab hall with a flat roof behind a parapet, roof lights and a
    red sign, a tall gun shop behind under a pitched roof with its great door open, two banded
    chimneys smoking, gun barrels stacked on bearers in the yard, a field gun standing ready and
    ammunition crates piled up."""
    p = Painter(48, 62, (24, 48))
    foot = _footing(-4, 1, 17, 9)
    _bed(p, foot)
    _bed(p, _footing(7, -8, 13, 7))
    yard = [(2, 8), (13, 2.5), (21, 6.5), (10, 12)]
    p.polygon(yard, (140, 130, 104))
    p.shadow(19, 5)
    for x, tall in ((-10, 26), (0, 20)):
        _chimney(p, x, -10, -10 - tall, shade(BRICK, 0.8))
        _smoke(p, x + 1, -13 - tall, (150, 148, 150), 3)
    _gun_shop(p, 7, -8, 13, 7, 11)
    top = _gun_hall(p, -4, 1, 17, 9, 8)
    left, front, _, _ = top
    _gun_sign(p, *_on_wall(left, front, 0.3, -0.5), 1.2)
    _barrels(p, 8, 5, 3)
    _field_gun(p, -14, 9.5, 1.1)
    _ammo(p, -21, 5)
    return p.sprite()


def mega_gun_factory() -> Sprite:
    """A great arsenal: a long drab works hall with roof lights and a red sign, a towering gun shop
    under a pitched roof with a gantry crane before it lifting a great barrel, three banded
    chimneys smoking, a proving range where a big gun fires down at a target on an earth bank in a
    burst of flame and smoke, field guns lined up, barrels stacked and ammunition crates piled."""
    p = Painter(59, 74, (29, 57))
    yard = [_iso(0, 0, u, v) for u, v in ((-14, -14), (14, -14), (14, 14), (-14, 14))]
    p.polygon([(x, y + 1) for x, y in yard], shade(MILITARY_YARD, 0.68))
    p.polygon(yard, MILITARY_YARD, shade(MILITARY_YARD, 0.85))
    p.polygon([_iso(0, 0, u, v) for u, v in ((-2, -14), (14, -14), (14, -7), (-2, -7))], EARTH)
    for v in (-9, -12):
        p.line([_iso(0, 0, 0, v), _iso(0, 0, 10, v)], shade(EARTH, 0.82))
    for k, u in enumerate((-12, -7, -2)):
        x, y = _iso(0, 0, u, 13)
        _chimney(p, x, y, y - 32 + 2 * k, shade(BRICK, 0.8))
        _smoke(p, x + 1, y - 35 + 2 * k, (150, 148, 150), 3)
    _gun_shop(p, *_iso(0, 0, -1, 8), 22, 9, 15)
    gx, gy = _iso(0, 0, 11, 3)
    for dv in (-3.5, 3.5):
        lx, ly = _iso(gx, gy, 0, dv)
        p.line([(lx - 1.5, ly + 0.8), (lx, ly - 18)], CRANE)
        p.line([(lx + 1.5, ly - 0.8), (lx, ly - 18)], CRANE)
    a, b = _iso(gx, gy, 0, -3.5), _iso(gx, gy, 0, 3.5)
    p.polygon(
        [(a[0], a[1] - 18), (b[0], b[1] - 18), (b[0], b[1] - 19.5), (a[0], a[1] - 19.5)],
        CRANE,
        shade(CRANE, 0.6),
    )
    p.line([(gx, gy - 18), (gx, gy - 11)], (40, 40, 44))
    s0, s1 = _iso(gx, gy - 9.5, -5, 0), _iso(gx, gy - 9.5, 5, 0)
    p.line([s0, s1], GUN_STEEL)
    p.line([(s0[0], s0[1] - 0.8), (s1[0], s1[1] - 0.8)], shade(GUN_STEEL, 1.35))
    p.line([(s0[0], s0[1] + 0.8), (s1[0] - 0.5, s1[1] + 0.4)], shade(GUN_STEEL, 0.7))
    top = _gun_hall(p, *_iso(0, 0, -6, -1), 16, 9, 9)
    left, front, _, _ = top
    _gun_sign(p, *_on_wall(left, front, 0.35, -0.5), 1.4)
    _barrels(p, *_iso(0, 0, 4, -3.5), 3)
    bx, by = _iso(0, 0, 12, -10.5)
    p.polygon(
        [(bx - 3, by + 2), (bx - 1, by - 4), (bx + 4, by - 6), (bx + 5, by - 1)],
        shade(EARTH, 0.95),
        shade(EARTH, 0.6),
    )
    for r, colour in ((2.2, (240, 240, 236)), (1.4, GUN_SIGN), (0.6, (240, 240, 236))):
        p.blob(bx + 1.2 - r, by - 2.5 - r, bx + 1.2 + r, by - 2.5 + r, colour)
    mx, my = _field_gun(p, *_iso(0, 0, -3, -10.5), 1.3)
    p.blob(mx - 1, my - 2.5, mx + 4, my + 2, FLAME)
    p.blob(mx, my - 1, mx + 2, my + 1, WHITE_HOT)
    _smoke(p, mx - 3, my - 3, (206, 204, 200), 3)
    for k in range(3):
        _field_gun(p, *_iso(0, 0, -13 + 4.5 * k, -6.5))
    _ammo(p, *_iso(0, 0, -12, -12))
    return p.sprite()


TANK_YARD: RGB = (150, 142, 112)
OLIVE: RGB = (104, 112, 80)
MUD: RGB = (112, 92, 66)


def _armoured_hall(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, tone: float
) -> Top:
    """A low armoured assembly hall in olive drab with riveted bands, slit windows, a big door and
    vents on its roof."""
    wall = shade(OLIVE, tone)
    roof = shade(OLIVE, 0.72)
    top = p.hall(x, y, length, depth, height, wall, roof)
    for a, b in ((top[0], top[1]), (top[1], top[2])):
        p.line([_on_wall(a, b, 0, 1), _on_wall(a, b, 1, 1)], shade(OLIVE, 1.3))
        p.line([_on_wall(a, b, 0, height - 1), _on_wall(a, b, 1, height - 1)], shade(OLIVE, 0.7))
        slits = max(3, round(math.dist(a, b) / 3))
        for k in range(1, slits):
            t = k / slits
            p.line([_on_wall(a, b, t - 0.03, 2.5), _on_wall(a, b, t + 0.03, 2.5)], (34, 36, 30))
    _patch(p, top[0], top[1], (0.6, 0.85), (3.5, height - 1), shade(OLIVE, 0.55))
    _plinth(p, top, height, wall, 0.6)
    left, front, right, back = top
    middle = _on_wall(left, back, 0.5, 0), _on_wall(front, right, 0.5, 0)
    for t in (0.25, 0.5, 0.75):
        vx, vy = _on_wall(*middle, t, 0)
        p.box(vx, vy, 1.2, 1.2, shade(OLIVE, 0.8), shade(OLIVE, 1.05))
    return top


def _mound(p: Painter, x: float, y: float, half: float, tall: float) -> None:
    """A churned earth mound on the proving ground centred on (x, y), scored with tank tracks."""
    p.polygon(
        [
            (x - half, y),
            (x - half * 0.3, y - tall),
            (x + half * 0.3, y - tall),
            (x + half, y),
            (x, y + half / 2),
        ],
        shade(MUD, 1.15),
        shade(MUD, 0.7),
    )
    p.polygon([(x + half * 0.3, y - tall), (x + half, y), (x, y + half / 2)], shade(MUD, 0.88))
    for dx in (-0.25, 0.05):
        p.line(
            [(x + half * dx - half * 0.4, y + half * 0.1), (x + half * dx, y - tall + 0.5)],
            shade(MUD, 0.8),
        )


def _proving_ground(p: Painter, u0: float, u1: float, v0: float, v1: float) -> None:
    """A stretch of churned mud between `u0`-`u1` and `v0`-`v1`, scored with tank tracks."""
    p.polygon([_iso(0, 0, u, v) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))], MUD)
    v = v0 + 1.5
    while v < v1 - 1:
        p.line([_iso(0, 0, u0 + 1, v), _iso(0, 0, u1 - 1, v)], shade(MUD, 0.72))
        p.line([_iso(0, 0, u0 + 1, v + 0.8), _iso(0, 0, u1 - 1, v + 0.8)], shade(MUD, 0.72))
        v += 3.4


def big_tank_factory() -> Sprite:
    """A bigger tank works: three low armoured halls in olive drab with slit windows, riveted bands
    and vents on their roofs, two banded stacks behind, a new tank parked before the halls and
    another churning across the proving ground by an earth mound, with ammunition crates and fuel
    drums stacked by."""
    p = Painter(53, 50, (26, 35))
    _plate(p, TANK_YARD, 24)
    for u in (4, 9):
        x, y = _iso(0, 0, u, 12)
        _chimney(p, x, y, y - 22, shade(BRICK, 0.8))
        _smoke(p, x + 1, y - 25, (150, 148, 150), 2)
    _proving_ground(p, -2, 12, -12, -1)
    under = shade(TANK_YARD, 0.76)
    halls = (((0, 9.5), 22, 5, 8, 0.95), ((-6.5, 2.5), 11, 7, 7, 1.0), ((5.5, 3.5), 9, 5, 6, 0.9))
    for (u, v), length, depth, _, _ in halls:
        _bed(p, _footing(*_iso(0, 0, u, v), length, depth), under, 0.8)
    for (u, v), length, depth, height, tone in halls:
        _armoured_hall(p, *_iso(0, 0, u, v), length, depth, height, tone)
    _puddle(p, *_iso(0, 0, 1, -3), 1.4)
    _puddle(p, *_iso(0, 0, 9, -11), 1.2)
    _ammo(p, *_iso(0, 0, -11, -6))
    _tank_vehicle(p, *_iso(0, 0, -5, -5), shade(OLIVE, 0.92))
    _mound(p, *_iso(0, 0, 9.5, -4), 4, 3)
    _tank_vehicle(p, *_iso(0, 0, 4, -8.5), OLIVE)
    for u, v in ((-11, -10), (-9.5, -11), (-12, -11.5)):
        _drum(p, *_iso(0, 0, u, v), (70, 76, 60))
    return p.sprite()


def mega_tank_factory() -> Sprite:
    """A great tank works: long armoured halls in olive drab under a sawtooth roof with slit
    windows and riveted bands, three banded stacks smoking, a wide proving ground of churned mud
    and an earth mound where new tanks are put through their paces, one climbing the mound, a
    finished tank parked up, a tank loaded on a flat wagon at the siding, a flag flying and
    ammunition crates and fuel drums stacked by."""
    p = Painter(62, 62, (30, 45))
    _plate(p, TANK_YARD, 29)
    for k, u in enumerate((0, 5, 10)):
        x, y = _iso(0, 0, u, 14)
        _chimney(p, x, y, y - 26 + 2 * k, shade(BRICK, 0.8))
        _smoke(p, x + 1, y - 29 + 2 * k, (150, 148, 150), 3)
    _proving_ground(p, -1, 14.5, -14.5, 0.5)
    under = shade(TANK_YARD, 0.76)
    halls = (((0.5, 11.5), 27, 5, 9, 0.95), ((-8, 4.5), 12, 8, 8, 1.0), ((7, 5.5), 13, 5, 7, 0.9))
    for (u, v), length, depth, _, _ in halls:
        _bed(p, _footing(*_iso(0, 0, u, v), length, depth), under, 0.8)
    for k, ((u, v), length, depth, height, tone) in enumerate(halls):
        top = _armoured_hall(p, *_iso(0, 0, u, v), length, depth, height, tone)
        if k == 0:
            _sawtooth(p, top, 7, 2.5, shade(OLIVE, 0.72), (150, 170, 140), shade(OLIVE, tone))
    for u, v in ((2, -2), (12, -13), (1, -13)):
        _puddle(p, *_iso(0, 0, u, v), 1.4)
    _flag(p, *_iso(0, 0, -2, 0), 14)
    _ammo(p, *_iso(0, 0, -13, -1))
    for u, v in ((-13.5, -3), (-12.5, -4)):
        _drum(p, *_iso(0, 0, u, v), (70, 76, 60))
    _tank_vehicle(p, *_iso(0, 0, -7, -5.5), shade(OLIVE, 0.92))
    _siding(p, -14.5, -1, -12.5)
    x, y = _iso(0, 0, -8, -12.5)
    p.hall(x, y, 10, 2.4, 1.2, (60, 58, 56), shade((60, 58, 56), 1.3))
    _tank_vehicle(p, x, y - 1.2, OLIVE)
    mx, my = _iso(0, 0, 10, -4)
    _mound(p, mx, my, 5.5, 4.5)
    _tank_vehicle(p, mx - 0.5, my - 3.5, OLIVE)
    _tank_vehicle(p, *_iso(0, 0, 3.5, -9.5), OLIVE)
    return p.sprite()


PLANE_YARD: RGB = (176, 176, 170)
APRON: RGB = (112, 114, 118)
HANGAR_WALL: RGB = (200, 206, 214)
HANGAR_ROOF: RGB = (150, 160, 176)
AIR_BLUE: RGB = (70, 130, 210)
WINDSOCK: RGB = (236, 120, 40)


def _hangar(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, tail: bool
) -> Top:
    """A great hangar on a dark plinth under an arched roof with a row of windows and a blue sign,
    its sliding doors pushed open on its end, a blue tail fin showing inside if `tail`."""
    top = p.hall(x, y, length, depth, height, HANGAR_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, height, HANGAR_WALL)
    _windows(p, left, front, tuple(k / 8 for k in range(1, 8)), (2,), 2, (96, 136, 168))
    _patch(p, left, front, (0.3, 0.7), (height - 2.7, height - 1.2), AIR_BLUE)
    p.line(
        [_on_wall(left, front, 0.36, height - 2), _on_wall(left, front, 0.64, height - 2)],
        (240, 242, 246),
    )
    _patch(p, front, right, (0.08, 0.92), (0.5, height), (44, 48, 58))
    for span in ((0, 0.12), (0.88, 1)):
        _patch(p, front, right, span, (0.5, height), shade(HANGAR_WALL, 0.7))
    for t in (0.06, 0.94):
        p.line(
            [_on_wall(front, right, t, 0.5), _on_wall(front, right, t, height)],
            shade(HANGAR_WALL, 0.55),
        )
    if tail:
        fin = [_on_wall(front, right, t, d) for t, d in ((0.5, height), (0.58, 2), (0.65, height))]
        p.polygon(fin, AIR_BLUE)
    _vault(p, top, depth * 0.75, HANGAR_ROOF, HANGAR_WALL)
    return top


def _windsock(p: Painter, x: float, y: float) -> None:
    """An orange-and-white windsock on its pole standing on (x, y)."""
    p.line([(x, y), (x, y - 7)], (90, 90, 96))
    p.polygon([(x, y - 7), (x + 4, y - 6), (x, y - 5)], WINDSOCK)
    p.line([(x + 2, y - 6.5), (x + 2, y - 5.5)], (244, 244, 240))


def _tug(p: Painter, x: float, y: float) -> None:
    """A small yellow aircraft tug standing on (x, y)."""
    p.box(x, y, 1.5, 1.5, HAZARD, shade(HAZARD, 1.15))


def big_plane_factory() -> Sprite:
    """A bigger aircraft works: two great hangars under arched roofs side by side with their doors
    pushed open, one with a blue tail inside, a long assembly shed behind, and two new airliners
    standing on the wide apron by a windsock and a tug."""
    p = Painter(51, 39, (25, 24))
    _plate(p, PLANE_YARD, 24)
    p.polygon([_iso(0, 0, u, v) for u, v in ((-1, -12), (12, -12), (12, 9), (-1, 9))], APRON)
    p.line([_iso(0, 0, 0, -1), _iso(0, 0, 11.5, -1)], HAZARD)
    under = shade(PLANE_YARD, 0.76)
    halls = (((-2, 11), 18, 3.5), ((-6, 4.5), 11, 9), ((-6, -5.5), 11, 9))
    for (u, v), length, depth in halls:
        _bed(p, _footing(*_iso(0, 0, u, v), length, depth), under, 0.8)
    for u, v in ((4, 6), (8, -8), (2, -10)):
        _oil(p, *_iso(0, 0, u, v), 1.2)
    shed = p.hall(*_iso(0, 0, -2, 11), 18, 3.5, 6, shade(HANGAR_WALL, 0.9), (140, 150, 166))
    _plinth(p, shed, 6, shade(HANGAR_WALL, 0.9))
    _windows(p, shed[0], shed[1], tuple(k / 10 for k in range(1, 10)), (2,), 1, (96, 136, 168))
    _hangar(p, *_iso(0, 0, -6, 4.5), 11, 9, 8, True)
    _plane(p, *_iso(0, 0, 6, 4.5), 0.75)
    _hangar(p, *_iso(0, 0, -6, -5.5), 11, 9, 8, False)
    _plane(p, *_iso(0, 0, 6, -5.5), 0.75)
    _windsock(p, *_iso(0, 0, 10.5, -11))
    _tug(p, *_iso(0, 0, 1.5, -10))
    return p.sprite()


def mega_plane_factory() -> Sprite:
    """A great aircraft works on its own airfield: two huge hangars under arched roofs with their
    doors pushed open on airliners being built, a long assembly shed behind, a control tower with
    a glass cab, a wide apron of new airliners and a runway with white markings where one stands
    ready to take off, by a windsock and a tug."""
    p = Painter(61, 47, (30, 30))
    _plate(p, PLANE_YARD, 29)
    p.polygon(
        [_iso(0, 0, u, v) for u, v in ((-14.5, -14.5), (14.5, -14.5), (14.5, -9), (-14.5, -9))],
        (84, 86, 92),
    )
    for k in range(7):
        u = -12 + 4 * k
        p.line([_iso(0, 0, u, -11.75), _iso(0, 0, u + 2, -11.75)], (240, 240, 236))
    for dv in (-13.8, -13, -10.5, -9.7):
        p.line([_iso(0, 0, -14, dv), _iso(0, 0, -12.5, dv)], (240, 240, 236))
    p.polygon([_iso(0, 0, u, v) for u, v in ((-3, -8), (14, -8), (14, 10), (-3, 10))], APRON)
    p.line([_iso(0, 0, -2, 0.5), _iso(0, 0, 13.5, 0.5)], HAZARD)
    under = shade(PLANE_YARD, 0.76)
    halls = (((-2, 13), 22, 3.5), ((-8, 5.5), 13, 11), ((-8, -3.5), 13, 7))
    for (u, v), length, depth in halls:
        _bed(p, _footing(*_iso(0, 0, u, v), length, depth), under, 0.8)
    shed = p.hall(*_iso(0, 0, -2, 13), 22, 3.5, 7, shade(HANGAR_WALL, 0.9), (140, 150, 166))
    _plinth(p, shed, 7, shade(HANGAR_WALL, 0.9))
    _windows(p, shed[0], shed[1], tuple(k / 12 for k in range(1, 12)), (2,), 2, (96, 136, 168))
    tx, ty = _iso(0, 0, 12, 11)
    p.rect(tx - 1.2, ty - 18, tx + 1.2, ty, (210, 212, 216))
    p.polygon(
        [(tx - 3.5, ty - 18), (tx + 3.5, ty - 18), (tx + 3, ty - 21.5), (tx - 3, ty - 21.5)],
        (96, 160, 196),
        (60, 70, 80),
    )
    p.rect(tx - 3.5, ty - 23, tx + 3.5, ty - 21.5, (90, 94, 104))
    p.line([(tx, ty - 23), (tx, ty - 26)], METAL)
    p.point(tx, ty - 27, (230, 60, 60))
    _hangar(p, *_iso(0, 0, -8, 5.5), 13, 11, 10, True)
    _plane(p, *_iso(0, 0, 5, 5.5), 0.9)
    _hangar(p, *_iso(0, 0, -8, -3.5), 13, 7, 8, True)
    _plane(p, *_iso(0, 0, 5, -4), 0.75)
    _plane(p, *_iso(0, 0, 0, -11.75), 0.85)
    _windsock(p, *_iso(0, 0, 13, -7))
    _tug(p, *_iso(0, 0, -1, -6.5))
    return p.sprite()
