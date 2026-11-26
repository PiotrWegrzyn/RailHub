"""Stations at each level, and the platforms added to the biggest, built as small models
(railroads.ui.pixel.art.built) like the other modelled buildings: lit, outlined and standing in a
soft shadow, with window frames, roof tiles and boards once drawn big enough to show them.

They grow from a wooden halt by the line, through a brick station with a canopy, a yellow town
station with wings and a red brick one between two domed towers, to a stone terminus with a clock
tower and a glass train shed over its platform. On hexes each is drawn turned along the track
through it; on squares turned as the other buildings are.

Models are in cells: u along the track, v across it towards the track, z up."""

import math
from collections.abc import Callable
from functools import cache, partial

import numpy as np
import numpy.typing as npt

from railroads.ui.pixel.art.built import (
    HEADING,
    PROJECTION,
    chimney,
    cornice,
    dome,
    door,
    footing,
    gable_roof,
    modelled,
    stretched,
    upright,
    walls,
    window,
)
from railroads.ui.pixel.art.painter import BRICK, NAME_BOARD, RGB, VERDIGRIS, Sprite, shade
from railroads.ui.pixel.models import Face, Model
from railroads.ui.pixel.track_art import BANK, SET_BACK

# The station houses on hexes, drawn turned along the track through them: for each way track
# runs on the screen, the heading the house's length lies along, chosen so its front, with the
# platform canopy, faces the track. The house stands behind track running across or slanting and
# to the left of track running up and down (railroads.ui.pixel.track_art.station_side).
# How big a platform's house is, as a share of the grand station it belongs to.
PLATFORM_HOUSE = 0.6
TURNED_LINES = {"across": 0.0, "falling": 60.0, "rising": 300.0, "upright": 270.0}

TIMBER: RGB = (200, 178, 134)
TIMBER_TRIM: RGB = (104, 66, 44)
SHINGLE: RGB = (112, 72, 60)
STONE: RGB = (196, 188, 172)
CREAM: RGB = (230, 220, 196)
OCHRE: RGB = (228, 192, 116)
TILE: RGB = (154, 70, 50)
STATION_BRICK: RGB = (172, 90, 62)
ASHLAR: RGB = (216, 206, 184)
ROOF_SLATE: RGB = (92, 96, 110)
IRON: RGB = (74, 80, 92)
GLASS: RGB = (176, 204, 218)
COLUMN: RGB = (138, 150, 140)
CANOPY: RGB = (118, 112, 110)
VALANCE: RGB = (236, 230, 214)
PLATFORM_STONE: RGB = (178, 172, 160)
CLOCK: RGB = (246, 242, 230)
PANE: RGB = (64, 78, 100)
# How wide a station's shadow is beside its footprint: its long low wings cast less than a tall
# building's.
SHADOW = 0.6
# How much of the platform's edge by the track shows in front of a station house standing on
# its platform, in cells of the picture's width.
PLATFORM_EDGE = 0.01
# How far from the middle of a hex its edges lie, in cells, and which ways they face.
APOTHEM = 0.5
HEX_EDGES = np.array(
    [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 60)]
)
# How much smaller a station house is tried again when it does not fit inside its hex.
FIT_STEP = 0.98


def _slope_normal(rise: float, run: float, way: tuple[float, float]) -> tuple[float, float, float]:
    """The way out of a roof face rising `rise` over `run`, sloping down towards `way` (u, v)."""
    length = math.hypot(rise, run) or 1.0
    return (way[0] * rise / length, way[1] * rise / length, run / length)


def hip_roof(
    m: Model,
    u: tuple[float, float],
    v: tuple[float, float],
    z: tuple[float, float],
    color: RGB,
    overhang: float = 0.01,
) -> None:
    """A roof hipped at both ends over walls `u` by `v`, from eaves `z[0]` to a ridge `z[1]`
    along the middle, with rows of tiles and a ridge up close."""
    (u0, u1), (v0, v1) = (u[0] - overhang, u[1] + overhang), (v[0] - overhang, v[1] + overhang)
    eaves, ridge = z
    rise, half = ridge - eaves, (v1 - v0) / 2
    hip = min(half, (u1 - u0) / 2)
    middle = (v0 + v1) / 2
    m.faces += [
        Face(
            [
                (u0, v1, eaves),
                (u1, v1, eaves),
                (u1 - hip, middle, ridge),
                (u0 + hip, middle, ridge),
            ],
            color,
            _slope_normal(rise, half, (0.0, 1.0)),
        ),
        Face(
            [
                (u1, v0, eaves),
                (u0, v0, eaves),
                (u0 + hip, middle, ridge),
                (u1 - hip, middle, ridge),
            ],
            color,
            _slope_normal(rise, half, (0.0, -1.0)),
        ),
        Face(
            [(u1, v1, eaves), (u1, v0, eaves), (u1 - hip, middle, ridge)],
            color,
            _slope_normal(rise, hip, (1.0, 0.0)),
        ),
        Face(
            [(u0, v0, eaves), (u0, v1, eaves), (u0 + hip, middle, ridge)],
            color,
            _slope_normal(rise, hip, (-1.0, 0.0)),
        ),
    ]
    with m.fine():
        rows = max(2, round(rise / 0.012))
        for k in range(1, rows):
            t = k / rows
            high = eaves + rise * t
            ends = (u0 + hip * t + 0.002, u1 - hip * t - 0.002)
            for edge in (v0, v1):
                at = edge + (middle - edge) * t
                m.box(ends, (at - 0.0012, at + 0.0012), (high, high + 0.0014), shade(color, 0.72))
        if u1 - u0 > 2 * hip + 0.004:
            m.box(
                (u0 + hip, u1 - hip),
                (middle - 0.004, middle + 0.004),
                (ridge - 0.002, ridge + 0.004),
                shade(color, 0.8),
            )


def pavilion_roof(m: Model, u: float, half: float, z: tuple[float, float], color: RGB) -> None:
    """A pointed four-sided roof over a square `half` either way of (`u`, 0), its eaves out a
    little, with a finial up close."""
    m.spire(u, half + 0.006, z, color)
    with m.fine():
        upright(m, u, 0.0, 0.003, (z[1] - 0.004, z[1] + 0.02), IRON)


def canopy(
    m: Model,
    u: tuple[float, float],
    front: float,
    reach: float,
    z: tuple[float, float],
    roof: RGB = CANOPY,
    posts: RGB | None = None,
) -> None:
    """A platform canopy leaning from the front wall at `front` out over the platform by `reach`:
    its roof falling from `z[1]` at the wall to `z[0]` at its edge, with a pale valance along
    that edge, held up by a row of `posts` under it or, without them, by brackets on the wall."""
    (u0, u1), (low, high) = u, z
    edge = front + reach
    m.faces.append(
        Face(
            [(u0, front, high), (u1, front, high), (u1, edge, low), (u0, edge, low)],
            roof,
            _slope_normal(high - low, reach, (0.0, 1.0)),
        )
    )
    drop = 0.012
    m.box((u0, u1), (edge - 0.002, edge), (low - drop, low), VALANCE)
    for end, out in ((u0, -1.0), (u1, 1.0)):
        m.faces.append(
            Face(
                [(end, front, high), (end, edge, low), (end, edge, low - drop), (end, front, low)],
                VALANCE,
                (out, 0.0, 0.0),
            )
        )
    count = max(1, round((u1 - u0) / 0.16))
    for k in range(count + 1):
        at = u0 + 0.012 + k * (u1 - u0 - 0.024) / count
        if posts is not None:
            m.box((at - 0.003, at + 0.003), (edge - 0.01, edge - 0.004), (0.0, low - drop), posts)
            continue
        with m.fine():
            brace = [
                (at, front, low - 0.034),
                (at, front, low - 0.026),
                (at, edge - 0.01, low),
                (at, edge - 0.016, low),
            ]
            m.faces.append(Face(brace, roof, (1.0, 0.0, 0.0)))
            m.faces.append(Face(brace[::-1], roof, (-1.0, 0.0, 0.0)))
    with m.fine():
        for k in range(count * 3):
            at = u0 + (k + 0.5) * (u1 - u0) / (count * 3)
            m.box(
                (at - 0.002, at + 0.002),
                (edge - 0.0026, edge + 0.0004),
                (low - drop, low - drop + 0.004),
                shade(VALANCE, 0.75),
            )


def boards(m: Model, u: tuple[float, float], v: float, height: float, color: RGB) -> None:
    """Upright battens up close down both long walls `v` out, every few hundredths of a cell."""
    with m.fine():
        count = round((u[1] - u[0]) / 0.014)
        for k in range(1, count):
            at = u[0] + k * (u[1] - u[0]) / count
            m.panel((at - 0.0012, at + 0.0012), (0.014, height), v + 0.0004, shade(color, 0.8))


def quoins(
    m: Model, u: tuple[float, float], v: tuple[float, float], height: float, color: RGB
) -> None:
    """Pale stone at the corners of a block, up close."""
    (u0, u1), (v0, v1) = u, v
    with m.fine():
        for cu in (u0, u1):
            for cv in (v0, v1):
                m.box((cu - 0.004, cu + 0.004), (cv - 0.004, cv + 0.004), (0.014, height), color)


def row_of_windows(
    m: Model,
    u: tuple[float, float],
    v: float,
    lows: tuple[float, ...],
    tall: float = 0.04,
    gap: float = 0.05,
    skip_middle: float = 0.0,
    glass: RGB = PANE,
) -> None:
    """Windows evenly along both long walls `v` out, a row at each of `lows`, leaving a space of
    `skip_middle` either side of the middle on the lowest row for a door."""
    count = max(1, round((u[1] - u[0]) / gap))
    for row, low in enumerate(lows):
        for k in range(count):
            at = u[0] + (k + 0.5) * (u[1] - u[0]) / count
            if row == 0 and abs(at) < skip_middle:
                continue
            window(m, (at - 0.011, at + 0.011), (low, low + tall), v, glass)


def name_board(m: Model, u: float, v: float, z: float, half: float = 0.03) -> None:
    """The station's blue name board on the wall at the end `u`, its face `v` across."""
    out = 0.004 if u > 0 else -0.004
    m.box((min(u, u + out), max(u, u + out)), (v - half, v + half), (z, z + 0.026), NAME_BOARD)
    with m.fine():
        for k in range(-2, 3):
            m.box(
                (min(u, u + 1.4 * out), max(u, u + 1.4 * out)),
                (v + k * half / 3 - 0.003, v + k * half / 3 + 0.003),
                (z + 0.011, z + 0.015),
                (236, 240, 250),
            )


def clock(m: Model, u: float, v: float, z: float, half: float = 0.014) -> None:
    """A clock face on both long walls `v` out, `z` up at its middle."""
    m.panel((u - half, u + half), (z - half, z + half), v + 0.0008, CLOCK)
    with m.fine():
        m.panel((u - 0.0015, u + 0.0015), (z - 0.001, z + half * 0.7), v + 0.0014, IRON)
        m.panel((u - 0.0015, u + half * 0.6), (z - 0.0015, z + 0.0015), v + 0.0014, IRON)


def block(
    m: Model,
    u: tuple[float, float],
    v: tuple[float, float],
    height: float,
    wall: RGB,
    trim: RGB,
) -> None:
    """A block of walls on a stone plinth, with a ledge of `trim` under its eaves."""
    footing(m, u, v)
    walls(m, u, v, height, wall, STONE)
    cornice(m, u, v, height - 0.008, trim)


def wooden_halt() -> Model:
    """A small wooden station: a board-and-batten waiting room under a shingled roof, a lower
    goods shed at one end, a chimney and a name board."""
    m = Model()
    u, v, high = (-0.16, 0.12), (-0.06, 0.06), 0.11
    footing(m, (-0.16, 0.22), v)
    walls(m, u, v, high, TIMBER, STONE)
    boards(m, u, v[1], high, TIMBER)
    row_of_windows(m, u, v[1], (0.04,), gap=0.07, skip_middle=0.03)
    door(m, (-0.016, 0.016), 0.085, v[1], TIMBER_TRIM)
    gable_roof(m, u, v, (high, high + 0.085), SHINGLE, TIMBER, 0.016)
    shed_u, shed_v, shed_high = (0.12, 0.22), (-0.05, 0.05), 0.08
    walls(m, shed_u, shed_v, shed_high, shade(TIMBER, 0.85), STONE)
    boards(m, shed_u, shed_v[1], shed_high, shade(TIMBER, 0.85))
    door(m, (0.15, 0.19), 0.065, shed_v[1], TIMBER_TRIM)
    gable_roof(m, shed_u, shed_v, (shed_high, shed_high + 0.06), shade(SHINGLE, 0.9), TIMBER)
    chimney(m, -0.06, -0.02, (high + 0.04, high + 0.12), BRICK)
    name_board(m, -0.176, 0.0, 0.07)
    return m


def brick_station() -> Model:
    """A brick station: a long block under a hipped slate roof with a little lookout tower over
    its middle, a one-storey wing at one end and a canopy on iron posts over the platform."""
    m = Model()
    u, v, high = (-0.18, 0.14), (-0.075, 0.075), 0.14
    block(m, u, v, high, STATION_BRICK, ASHLAR)
    quoins(m, u, v, high, ASHLAR)
    row_of_windows(m, u, v[1], (0.05,), tall=0.05, gap=0.055, skip_middle=0.03)
    door(m, (-0.018, 0.018), 0.1, v[1])
    hip_roof(m, u, v, (high, high + 0.08), ROOF_SLATE, 0.014)
    tower = (-0.035, 0.035)
    block(m, tower, tower, high + 0.15, STATION_BRICK, ASHLAR)
    window(m, (-0.012, 0.012), (high + 0.09, high + 0.125), 0.035, PANE)
    pavilion_roof(m, 0.0, 0.035, (high + 0.15, high + 0.23), ROOF_SLATE)
    wing_u, wing_v, wing_high = (0.14, 0.25), (-0.06, 0.06), 0.1
    block(m, wing_u, wing_v, wing_high, STATION_BRICK, ASHLAR)
    row_of_windows(m, wing_u, wing_v[1], (0.04,), gap=0.055)
    gable_roof(m, wing_u, wing_v, (wing_high, wing_high + 0.06), ROOF_SLATE, STATION_BRICK)
    chimney(m, -0.12, 0.0, (high + 0.04, high + 0.11), STATION_BRICK)
    canopy(m, (-0.13, 0.13), v[1], 0.08, (high - 0.035, high - 0.01), ROOF_SLATE, COLUMN)
    name_board(m, 0.254, 0.0, 0.05)
    return m


def town_station() -> Model:
    """A town station in yellow render: a two-storey middle under a tiled hipped roof with a
    clock in a gable over its door, a one-storey wing either side and a long canopy over the
    platform."""
    m = Model()
    u, v, high = (-0.11, 0.11), (-0.09, 0.09), 0.24
    block(m, u, v, high, OCHRE, CREAM)
    quoins(m, u, v, high, CREAM)
    cornice(m, u, v, 0.125, CREAM)
    row_of_windows(m, u, v[1], (0.05, 0.155), tall=0.05, gap=0.05, skip_middle=0.03)
    door(m, (-0.02, 0.02), 0.105, v[1])
    hip_roof(m, u, v, (high, high + 0.1), TILE, 0.014)
    m.faces.append(
        Face(
            [
                (-0.06, v[1] + 0.001, high),
                (0.06, v[1] + 0.001, high),
                (0.0, v[1] + 0.001, high + 0.06),
            ],
            CREAM,
            (0.0, 1.0, 0.0),
        )
    )
    clock(m, 0.0, v[1] + 0.001, high + 0.022)
    for wing_u in ((-0.31, -0.11), (0.11, 0.31)):
        wing_v, wing_high = (-0.075, 0.075), 0.14
        block(m, wing_u, wing_v, wing_high, OCHRE, CREAM)
        row_of_windows(m, wing_u, wing_v[1], (0.05,), tall=0.05, gap=0.05)
        hip_roof(m, wing_u, wing_v, (wing_high, wing_high + 0.065), TILE, 0.012)
    for at in (-0.2, 0.2):
        chimney(m, at, 0.0, (0.18, 0.25), shade(OCHRE, 0.85))
    canopy(m, (-0.25, 0.25), 0.075, 0.09, (0.1, 0.125), TILE, COLUMN)
    name_board(m, -0.324, 0.0, 0.06)
    name_board(m, 0.324, 0.0, 0.06)
    return m


def grand_station() -> Model:
    """A grand red brick station trimmed in pale stone: a high hall with a tall arched window
    between two towers under green copper domes, two-storey wings either side and a long canopy
    over the platform."""
    m = Model()
    u, v, high = (-0.1, 0.1), (-0.1, 0.1), 0.3
    block(m, u, v, high, STATION_BRICK, ASHLAR)
    gable_roof(m, (-0.1, 0.1), v, (high, high + 0.12), ROOF_SLATE, STATION_BRICK, 0.012)
    m.panel((-0.045, 0.045), (0.11, 0.24), v[1] + 0.0008, ASHLAR)
    m.panel((-0.035, 0.035), (0.12, 0.235), v[1] + 0.0014, GLASS)
    with m.fine():
        for at in (-0.012, 0.012):
            m.panel((at - 0.0015, at + 0.0015), (0.12, 0.235), v[1] + 0.002, IRON)
        m.panel((-0.035, 0.035), (0.175, 0.178), v[1] + 0.002, IRON)
    door(m, (-0.03, 0.03), 0.1, v[1])
    clock(m, 0.0, v[1], 0.27, 0.016)
    for at in (-0.135, 0.135):
        tower = (at - 0.04, at + 0.04)
        block(m, tower, (-0.045, 0.045), 0.4, STATION_BRICK, ASHLAR)
        quoins(m, tower, (-0.045, 0.045), 0.4, ASHLAR)
        for low in (0.06, 0.17, 0.29):
            window(m, (at - 0.012, at + 0.012), (low, low + 0.05), 0.045, PANE)
        m.box((at - 0.046, at + 0.046), (-0.051, 0.051), (0.4, 0.412), ASHLAR)
        dome(m, at, 0.0, 0.045, (0.412, 0.49), VERDIGRIS)
        with m.fine():
            upright(m, at, 0.0, 0.003, (0.48, 0.53), IRON)
    for wing_u in ((-0.39, -0.175), (0.175, 0.39)):
        wing_v, wing_high = (-0.08, 0.08), 0.22
        block(m, wing_u, wing_v, wing_high, STATION_BRICK, ASHLAR)
        cornice(m, wing_u, wing_v, 0.11, ASHLAR)
        row_of_windows(m, wing_u, wing_v[1], (0.04, 0.14), tall=0.05, gap=0.045)
        hip_roof(m, wing_u, wing_v, (wing_high, wing_high + 0.08), ROOF_SLATE, 0.012)
        chimney(m, sum(wing_u) / 2, 0.0, (0.26, 0.33), STATION_BRICK)
    canopy(m, (-0.3, 0.3), 0.08, 0.1, (0.11, 0.14), ROOF_SLATE, COLUMN)
    return m


def branch_station() -> Model:
    """A platform added to a grand station: the grand station in small, its towers, wings and
    canopy PLATFORM_HOUSE times the size."""
    return stretched(grand_station(), PLATFORM_HOUSE, PLATFORM_HOUSE)


def terminus() -> Model:
    """A central terminus of pale stone: a long three-storey front with a tall clock tower in its
    middle and a pavilion at either end, all under slate, and an arched train shed of iron and
    glass over its platform before it."""
    m = Model()
    u, v, high = (-0.4, 0.4), (-0.07, 0.07), 0.32
    block(m, u, v, high, ASHLAR, CREAM)
    for low in (0.11, 0.215):
        cornice(m, u, v, low, CREAM)
    row_of_windows(m, u, v[1], (0.04, 0.14, 0.24), tall=0.05, gap=0.05)
    hip_roof(m, u, v, (high, high + 0.09), ROOF_SLATE, 0.012)
    for at in (-0.37, 0.37):
        pavilion = (at - 0.06, at + 0.06)
        block(m, pavilion, (-0.08, 0.08), 0.38, ASHLAR, CREAM)
        quoins(m, pavilion, (-0.08, 0.08), 0.38, CREAM)
        pavilion_roof(m, at, 0.06, (0.38, 0.47), ROOF_SLATE)
    tower = (-0.055, 0.055)
    block(m, tower, tower, 0.56, ASHLAR, CREAM)
    quoins(m, tower, tower, 0.56, CREAM)
    clock(m, 0.0, 0.055, 0.49, 0.02)
    m.box((-0.06, 0.06), (-0.06, 0.06), (0.56, 0.575), CREAM)
    pavilion_roof(m, 0.0, 0.06, (0.575, 0.72), ROOF_SLATE)
    back = -0.09
    m.move(0, 0.0, back)
    m.footprints = [(fu, (fv[0] + back, fv[1] + back)) for fu, fv in m.footprints]
    shed_half, shed_v = 0.13, 0.11
    start = len(m.faces)
    m.vault((-0.38, 0.38), shed_half, (0.1, 0.25), GLASS)
    with m.fine():
        for k in range(9):
            at = -0.38 + k * 0.76 / 8
            m.vault((at - 0.004, at + 0.004), shed_half + 0.003, (0.1, 0.253), IRON)
    for k in range(5):
        at = -0.37 + k * 0.74 / 4
        side = shed_half
        m.box((at - 0.005, at + 0.005), (side - 0.005, side + 0.005), (0.0, 0.1), COLUMN)
    m.move(start, 0.0, shed_v)
    return m


def island_platform() -> Model:
    """A platform added to a station: a stone island under a long ridged canopy on a row of iron
    posts down its middle, with a name board on a post at either end."""
    m = Model()
    m.footprints.append(((-0.33, 0.33), (-0.08, 0.08)))
    m.box((-0.4, 0.4), (-0.07, 0.07), (0.0, 0.03), PLATFORM_STONE)
    gable_roof(m, (-0.33, 0.33), (-0.08, 0.08), (0.13, 0.155), CANOPY, CANOPY, 0.0)
    m.box((-0.33, 0.33), (-0.082, 0.082), (0.118, 0.13), VALANCE)
    for k in range(6):
        at = -0.3 + k * 0.6 / 5
        m.box((at - 0.005, at + 0.005), (-0.005, 0.005), (0.03, 0.118), IRON)
    for at in (-0.37, 0.37):
        m.box((at - 0.003, at + 0.003), (-0.003, 0.003), (0.03, 0.09), IRON)
        name_board(m, at, 0.0, 0.09, 0.04)
    return m


# Each station's model.
LEVELS: dict[str, Callable[[], Model]] = {
    "station": wooden_halt,
    "station_2": brick_station,
    "station_3": town_station,
    "station_4": grand_station,
    "station_5": terminus,
    "platform": branch_station,
}


def by_platform(build: Callable[[], Model], heading: float) -> Model:
    """The station built by `build`, turned `heading`, standing on its platform inside its own
    hex: what reaches furthest towards the track, a canopy, wall or train shed, by the edge of the
    platform along the track with that edge showing in front of it, its middle by the middle of
    the hex along the track, and as big as it can be (_fit) so that whichever way its track runs,
    none of it reaches past the hex's edges."""
    m = build()
    scale, front = _fit(build)
    m = stretched(m, scale, scale)
    slide, shift = _centred(heading), _front_at(heading) - front * scale
    m.move(0, slide, shift)
    m.footprints = [
        ((u[0] + slide, u[1] + slide), (v[0] + shift, v[1] + shift)) for u, v in m.footprints
    ]
    return m


def _plan(build: Callable[[], Model]) -> npt.NDArray[np.float64]:
    """Where the corners of a model's faces drawn at every zoom lie on the ground, u by v."""
    return np.array([c[:2] for face in build().faces if not face.fine for c in face.corners])


@cache
def _fit(build: Callable[[], Model]) -> tuple[float, float]:
    """How big a station house is drawn, at most its full size, so that standing on its
    platform (by_platform) its plan lies inside its hex whichever way its track runs, and how far
    towards the track from where it stands it reaches at full size: by its faces drawn at every
    zoom, so it is the same size at all of them."""
    plan = _plan(build)
    front = float(plan[:, 1].max())
    scale = 1.0
    while scale > FIT_STEP**100 and not all(
        _inside(plan, scale, front, math.radians(heading)) for heading in TURNED_LINES.values()
    ):
        scale *= FIT_STEP
    return scale, front


def _inside(plan: npt.NDArray[np.float64], scale: float, front: float, heading: float) -> bool:
    """Whether a house's plan, drawn `scale` times its size and turned `heading` by its
    platform, lies inside its hex."""
    u = plan[:, 0] * scale + _centred(heading)
    v = plan[:, 1] * scale + _front_at(heading) - front * scale
    cos, sin = math.cos(heading), math.sin(heading)
    x, y = _stands_at(heading)
    ground = np.stack([x + u * cos - v * sin, y + u * sin + v * cos], axis=1)
    return bool((ground @ HEX_EDGES.T).max() <= APOTHEM + 1e-9)


def _centred(heading: float) -> float:
    """How far along its track, in cells, a house turned `heading` moves to stand by the middle
    of its hex: set back square to the track across the picture (_stands_at), it stands off the
    middle along track slanting across the map."""
    return -float(_stands_at(heading) @ (math.cos(heading), math.sin(heading)))


def _front_at(heading: float) -> float:
    """How far across from where a house turned `heading` stands, in cells, the edge of its
    platform by the track lies, less PLATFORM_EDGE (railroads.ui.pixel.track_art, which needs
    this module's package first): the cells across the track fill more or less of the picture
    by the way it runs."""
    gap = SET_BACK - BANK / 2 - PLATFORM_EDGE
    return gap * _cell() / _reach(heading)


def _cell() -> float:
    """How many pixels wide a cell is in the picture."""
    return float(np.hypot(*PROJECTION.matrix[:, 0]))


def _towards(heading: float) -> tuple[npt.NDArray[np.float64], float]:
    """Which way across the picture, square to track turned `heading`, a house's front faces,
    and how many pixels a cell across that track takes that way."""
    plane = PROJECTION.matrix
    along = plane @ (math.cos(heading), math.sin(heading))
    across = plane @ (-math.sin(heading), math.cos(heading))
    square = np.array([-along[1], along[0]]) / float(np.hypot(*along))
    reach = float(across @ square)
    return (square if reach > 0 else -square), abs(reach)


def _reach(heading: float) -> float:
    return _towards(heading)[1]


def _stands_at(heading: float) -> npt.NDArray[np.float64]:
    """Where a house turned `heading` stands from the middle of its hex, in cells: SET_BACK cells
    of the picture's width from its track square across the picture, away from the track
    (railroads.ui.pixel.scenery)."""
    facing, _ = _towards(heading)
    return np.linalg.solve(PROJECTION.matrix, -facing * SET_BACK * _cell())


def turned(build: Callable[[], Model], heading: float) -> Callable[[], Sprite]:
    """The sprite of a station built by `build`, turned `heading` radians and standing on its
    platform."""
    return modelled(partial(by_platform, build, heading), shadow=SHADOW, heading=heading)


SQUARE_STATIONS = {name: turned(build, HEADING) for name, build in LEVELS.items()}

TURNED_STATIONS = {
    f"{name}_{line}": turned(build, math.radians(heading))
    for name, build in LEVELS.items()
    for line, heading in TURNED_LINES.items()
}
