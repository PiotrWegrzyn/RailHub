"""Ports (railroads.sim.ports) built as small lit models (railroads.ui.pixel.art.built) at the
works' scale, filling their cell with their own ground and only their piers reaching out of it
over the water beside it: a port is a fishing harbour of a shingle beach, a trodden yard, a
boathouse and a cottage with its garden, a big port a walled yard of gravel with a brick
warehouse, a harbour office, cranes and a stone pier, and a mega port a fenced container terminal
with a long warehouse and stacks of ribbed containers by two concrete piers, each with a gantry
crane over the water. A way runs from each out to the land at its cell's far edge, and trees and
bushes stand round it, so it sits among the land beside it. Each is drawn turned towards each way
the water can lie beside its cell (PORT_HEADINGS), its piers running out along u.

Models are in the works' cells: u and v across the ground, z up, the cell's middle at the
origin."""

import itertools
import math
import random
from collections.abc import Callable

from railroads.ui.pixel.art.built import (
    SOIL,
    WINDOW,
    WORN,
    chimney,
    cone,
    dome,
    door,
    gable_roof,
    modelled,
    upright,
    walls,
    window,
)
from railroads.ui.pixel.art.harbour import (
    GLASS,
    TRUCK_BLUE,
    TRUCK_YELLOW,
    box_wagon,
    container_lorry,
    container_ship,
    dump_truck,
    fishing_boat,
    flat_wagon,
    horse_cart,
    open_wagon,
    placed,
    rowing_boat,
    steamer,
)
from railroads.ui.pixel.art.painter import BRICK, RGB, Sprite, shade
from railroads.ui.pixel.models import Face, Model

PLANKS: RGB = (150, 112, 74)
PILES: RGB = (88, 64, 46)
SHED: RGB = (176, 150, 112)
SHED_ROOF: RGB = (96, 100, 106)
SLATE: RGB = (80, 86, 98)
QUAY: RGB = (150, 146, 136)
CONCRETE: RGB = (188, 186, 178)
EDGE: RGB = (226, 192, 40)
BOLLARD: RGB = (52, 50, 54)
DOORWAY: RGB = (52, 42, 36)
BARREL: RGB = (126, 82, 50)
CRATE: RGB = (156, 108, 62)
CRANE: RGB = (214, 168, 40)
GANTRY: RGB = (196, 74, 52)
CABIN: RGB = (232, 230, 222)
CABLE: RGB = (60, 58, 56)
HALL: RGB = (150, 166, 178)
HALL_ROOF: RGB = (112, 126, 138)
GRAVEL: RGB = (150, 142, 124)
COBBLES: RGB = (164, 158, 148)
QUAY_WALL: RGB = (112, 108, 100)
STONE_WALL: RGB = (196, 188, 172)
ASPHALT: RGB = (92, 94, 98)
LAMP: RGB = (250, 230, 160)
NET: RGB = (118, 138, 128)
SACK: RGB = (204, 186, 146)
DARK_TIMBER: RGB = (122, 96, 70)
FISH: RGB = (190, 196, 200)
LOGS: RGB = (140, 102, 66)
RUT: RGB = (84, 70, 52)
PUDDLE: RGB = (92, 104, 112)
SHINE: RGB = (150, 166, 176)
COAL: RGB = (40, 40, 44)
SAND: RGB = (214, 190, 132)
WHITE_LINE: RGB = (226, 226, 220)
SHINGLE: RGB = (200, 190, 164)
YARD_EARTH: RGB = (156, 132, 92)
WHITEWASH: RGB = (236, 232, 218)
RED_ROOF: RGB = (172, 74, 50)
BOARDS: RGB = (156, 54, 42)
LEAVES: RGB = (78, 124, 50)
CABBAGE: RGB = (120, 160, 90)
TRUNK: RGB = (92, 70, 50)
FENCE: RGB = (176, 146, 104)
PICKET: RGB = (226, 222, 208)
BOAT_BLUE: RGB = (58, 104, 158)
BOAT_WHITE: RGB = (226, 222, 210)
BOAT_GREEN: RGB = (70, 126, 86)
SMOKE: RGB = (214, 214, 218)
STEEL_POST: RGB = (150, 154, 158)
CONTAINERS: tuple[RGB, ...] = (
    (176, 64, 52),
    (52, 104, 156),
    (64, 132, 84),
    (204, 150, 48),
    (130, 130, 136),
)

# The ways a port is drawn turned, its piers running that way on the map, in degrees from the
# map's x axis: every way a cell beside it lies on hexes.
PORT_HEADINGS = tuple(range(0, 360, 60))

# A container's length, width and height, in cells.
CONTAINER = (0.075, 0.028, 0.028)


def _deck(
    m: Model,
    u: tuple[float, float],
    half: float,
    v: float,
    height: float,
    deck: RGB,
    piles: RGB | None,
    every: float,
) -> None:
    """A pier along u from `u[0]` to `u[1]`, `half` either side of `v`, its deck `height` up:
    on piles every `every` if `piles` are given, else built solid, with planks or slabs across it
    up close and bollards along its sides."""
    v0, v1 = v - half, v + half
    if piles is None:
        m.box(u, (v0, v1), (0.0, height), deck)
    else:
        m.box(u, (v0, v1), (height - 0.006, height), deck)
        count = max(1, round((u[1] - u[0]) / every))
        for k in range(count + 1):
            at = u[0] + (u[1] - u[0]) * k / count
            for side in (v0 + 0.004, v1 - 0.004):
                m.box((at - 0.004, at + 0.004), (side - 0.004, side + 0.004), (0.0, height), piles)
    with m.fine():
        slats = max(2, round((u[1] - u[0]) / 0.02))
        for k in range(1, slats):
            at = u[0] + (u[1] - u[0]) * k / slats
            m.box((at - 0.0008, at + 0.0008), (v0, v1), (height, height + 0.0006), shade(deck, 0.8))
        for k in range(1, 5):
            at = u[0] + (u[1] - u[0]) * k / 5
            for side in (v0 + 0.005, v1 - 0.005):
                upright(m, at, side, 0.0035, (height, height + 0.01), BOLLARD)


def _crate(m: Model, u: float, v: float, z: float = 0.0, half: float = 0.012) -> None:
    m.box((u - half, u + half), (v - half, v + half), (z, z + 2 * half), CRATE)


def _barrel(m: Model, u: float, v: float, z: float = 0.0) -> None:
    upright(m, u, v, 0.008, (z, z + 0.022), BARREL)


def _container(m: Model, u: float, v: float, z: float, color: RGB, along_u: bool = True) -> None:
    """A shipping container standing at (u, v) on `z`: a long ribbed steel box with doors at one
    end up close."""
    length, width, height = CONTAINER
    du, dv = (length / 2, width / 2) if along_u else (width / 2, length / 2)
    m.box((u - du, u + du), (v - dv, v + dv), (z, z + height), color)
    with m.fine():
        ribs = 7
        for k in range(1, ribs):
            if along_u:
                at = u - du + 2 * du * k / ribs
                for side in (v - dv - 0.0005, v + dv + 0.0005):
                    m.box(
                        (at - 0.0012, at + 0.0012),
                        (side - 0.0003, side + 0.0003),
                        (z + 0.002, z + height - 0.002),
                        shade(color, 0.72),
                    )
            else:
                at = v - dv + 2 * dv * k / ribs
                for side in (u - du - 0.0005, u + du + 0.0005):
                    m.box(
                        (side - 0.0003, side + 0.0003),
                        (at - 0.0012, at + 0.0012),
                        (z + 0.002, z + height - 0.002),
                        shade(color, 0.72),
                    )
        m.box(
            (u - du, u + du),
            (v - dv, v + dv),
            (z + height, z + height + 0.0008),
            shade(color, 1.15),
        )


def _stack(
    m: Model, u: float, v: float, rows: int, tiers: tuple[int, ...], seed: int, along_u: bool
) -> None:
    """Containers in `rows` rows across from (u, v), each row `tiers[row]` high."""
    _, width, height = CONTAINER
    for row in range(rows):
        for tier in range(tiers[row % len(tiers)]):
            color = CONTAINERS[(seed + row * 3 + tier * 2) % len(CONTAINERS)]
            if along_u:
                _container(m, u, v + row * (width + 0.004), tier * height, color, True)
            else:
                _container(m, u + row * (width + 0.004), v, tier * height, color, False)


def crane(reach: float, high: float, load: RGB) -> Model:
    """A slewing crane on a portal of four legs, its lattice tower rising to a cabin under a
    long jib reaching `reach` out along u, a counterweight behind and a load on its hook."""
    m = Model()
    for lu in (-0.022, 0.022):
        for lv in (-0.022, 0.022):
            m.box((lu - 0.004, lu + 0.004), (lv - 0.004, lv + 0.004), (0.0, 0.05), CRANE)
    m.box((-0.028, 0.028), (-0.028, 0.028), (0.05, 0.062), shade(CRANE, 0.8))
    m.box((-0.01, 0.01), (-0.01, 0.01), (0.062, high), CRANE)
    with m.fine():
        rungs = 10
        for k in range(rungs):
            z = 0.062 + (high - 0.062) * k / rungs
            m.box((-0.0108, 0.0108), (-0.0108, 0.0108), (z, z + 0.002), shade(CRANE, 0.65))
    m.box((-0.02, 0.022), (-0.016, 0.016), (high - 0.04, high - 0.008), CABIN)
    with m.fine():
        m.panel((0.0, 0.018), (high - 0.034, high - 0.016), 0.0166, GLASS)
    m.box((-0.1, reach), (-0.006, 0.006), (high, high + 0.012), CRANE)
    with m.fine():
        bays = max(2, round(reach / 0.03))
        for k in range(bays):
            at = reach * k / bays
            m.box(
                (at, at + 0.002), (-0.0064, 0.0064), (high + 0.012, high + 0.016), shade(CRANE, 0.6)
            )
    m.box((-0.002, 0.002), (-0.002, 0.002), (high, high + 0.06), CRANE)
    for side in (-0.0015, 0.0015):
        m.slope(
            (0.0, high + 0.06), (reach * 0.8, high + 0.012), (side - 0.0008, side + 0.0008), CABLE
        )
    m.box((-0.11, -0.07), (-0.016, 0.016), (high - 0.02, high + 0.008), shade(CRANE, 0.45))
    hook = reach - 0.015
    m.box((hook - 0.0015, hook + 0.0015), (-0.0015, 0.0015), (0.09, high), CABLE)
    m.box((hook - 0.016, hook + 0.016), (-0.016, 0.016), (0.06, 0.09), load)
    return m


def gantry(reach: float, high: float, half: float) -> Model:
    """A container gantry crane on four legs `half` either side of the u axis, its boom reaching
    `reach` out along u over the water, with its machinery house, a trolley and a container
    hanging under it."""
    m = Model()
    for lu in (-0.04, 0.04):
        for lv in (-half, half):
            m.box((lu - 0.006, lu + 0.006), (lv - 0.006, lv + 0.006), (0.0, high), GANTRY)
        m.box(
            (lu - 0.008, lu + 0.008), (-half - 0.008, half + 0.008), (0.0, 0.01), shade(GANTRY, 0.5)
        )
    for z in (high * 0.35, high * 0.7):
        for lv in (-half, half):
            m.box((-0.04, 0.04), (lv - 0.004, lv + 0.004), (z, z + 0.008), shade(GANTRY, 0.8))
    for lv in (-half, half):
        m.box((-0.16, reach), (lv - 0.005, lv + 0.005), (high, high + 0.016), GANTRY)
        with m.fine():
            bays = max(2, round((reach + 0.16) / 0.035))
            for k in range(bays):
                at = -0.16 + (reach + 0.16) * k / bays
                m.box(
                    (at, at + 0.003),
                    (lv - 0.0054, lv + 0.0054),
                    (high + 0.016, high + 0.03),
                    shade(GANTRY, 0.7),
                )
    m.box((-0.16, reach), (-half, half), (high + 0.016, high + 0.02), shade(GANTRY, 0.75))
    m.box((-0.16, -0.07), (-half, half), (high + 0.02, high + 0.05), CABIN)
    m.box((-0.02, 0.03), (-0.008, 0.008), (high + 0.02, high + 0.09), GANTRY)
    trolley = reach * 0.7
    m.box((trolley - 0.018, trolley + 0.018), (-0.018, 0.018), (high - 0.014, high), CABIN)
    for side in (-0.01, 0.01):
        m.box(
            (trolley - 0.0012, trolley + 0.0012),
            (side - 0.0012, side + 0.0012),
            (0.1, high - 0.014),
            CABLE,
        )
    _container(m, trolley, 0.0, 0.07, CONTAINERS[1], True)
    return m


def _apothem() -> float:
    """How far the middle of a port's cell lies from its edges in the model's units: half a cell,
    as the model is drawn at the works' scale (railroads.ui.pixel.art.registry.BUILDING_SCALE)."""
    from railroads.ui.pixel.art.registry import BUILDING_SCALE

    return 0.5 / BUILDING_SCALE


def _across(u: float, apothem: float) -> float:
    """How far either side of the u axis the cell reaches `u` along it: the cell's hexagon has an
    edge across the u axis at each end, which faces the water."""
    return max(apothem - abs(u) / 2, 0.0) * 2 / math.sqrt(3)


def _slab(m: Model, points: list[tuple[float, float]], z: float, color: RGB, side: RGB) -> None:
    """A flat slab `z` high over the ground through `points` round it, with its sides; a patch
    lying on the ground where `z` is next to nothing."""
    m.faces.append(Face([(u, v, z) for u, v in points], color, (0.0, 0.0, 1.0)))
    if z < 0.002:
        return
    for (u0, v0), (u1, v1) in zip(points, points[1:] + points[:1], strict=True):
        length = math.hypot(u1 - u0, v1 - v0) or 1.0
        normal = ((v1 - v0) / length, -(u1 - u0) / length, 0.0)
        m.faces.append(Face([(u0, v0, 0.0), (u1, v1, 0.0), (u1, v1, z), (u0, v0, z)], side, normal))


def _rect(u: tuple[float, float], v: tuple[float, float]) -> list[tuple[float, float]]:
    (u0, u1), (v0, v1) = u, v
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


def _ellipse(
    u: float, v: float, ru: float, rv: float, sides: int = 12
) -> list[tuple[float, float]]:
    return [
        (u + ru * math.cos(2 * math.pi * k / sides), v + rv * math.sin(2 * math.pi * k / sides))
        for k in range(sides)
    ]


def _yard(m: Model, u: tuple[float, float], v: tuple[float, float], surface: RGB) -> None:
    """A rectangular yard lying on the ground, its edge worn into the grass round it as the
    other works' yards are (railroads.ui.pixel.art.built.worn)."""
    (u0, u1), (v0, v1) = u, v
    rim = 0.035
    _slab(m, _rect((u0 - rim, u1 + rim), (v0 - rim, v1 + rim)), 0.0004, WORN, WORN)
    _slab(m, _rect(u, v), 0.0007, surface, surface)


def _ruts(m: Model, points: list[tuple[float, float]]) -> None:
    """Two wheel ruts along the ground through `points`, a little apart."""
    for side in (-0.012, 0.012):
        for (ua, va), (ub, vb) in itertools.pairwise(points):
            length = math.hypot(ub - ua, vb - va) or 1.0
            nu, nv = -(vb - va) / length, (ub - ua) / length
            off_u, off_v = nu * side, nv * side
            width = 0.0035
            m.faces.append(
                Face(
                    [
                        (ua + off_u + nu * width, va + off_v + nv * width, 0.0009),
                        (ub + off_u + nu * width, vb + off_v + nv * width, 0.0009),
                        (ub + off_u - nu * width, vb + off_v - nv * width, 0.0009),
                        (ua + off_u - nu * width, va + off_v - nv * width, 0.0009),
                    ],
                    RUT,
                    (0.0, 0.0, 1.0),
                )
            )


def _puddle(m: Model, u: float, v: float, ru: float, rv: float) -> None:
    _slab(m, _ellipse(u, v, ru, rv), 0.001, PUDDLE, PUDDLE)
    _slab(m, _ellipse(u - ru * 0.3, v - rv * 0.3, ru * 0.35, rv * 0.25, 8), 0.0011, SHINE, SHINE)


def _gravel(m: Model, u: tuple[float, float], v: tuple[float, float], seed: int) -> None:
    """A yard of gravel, up close strewn with bigger stones."""
    _yard(m, u, v, GRAVEL)
    (u0, u1), (v0, v1) = u, v
    rng = random.Random(seed)
    with m.fine():
        for _ in range(int((u1 - u0) * (v1 - v0) * 320)):
            pu, pv = rng.uniform(u0 + 0.01, u1 - 0.01), rng.uniform(v0 + 0.01, v1 - 0.01)
            tone = shade(GRAVEL, rng.choice((0.82, 0.9, 1.12)))
            _slab(m, _ellipse(pu, pv, 0.003, 0.0025, 5), 0.0012, tone, tone)


def _quay(m: Model, start: float, half: float, top: RGB, wall: RGB) -> float:
    """A paved quay from `start` along u to the shore at the cell's edge, `half` either side, with
    its wall facing the water and, up close, rows of slabs; returns where its edge stands."""
    edge = _apothem() * 0.97
    _slab(m, _rect((start, edge), (-half, half)), 0.012, top, wall)
    with m.fine():
        rows = 6
        for k in range(1, rows):
            at = start + (edge - start) * k / rows
            m.box((at - 0.0006, at + 0.0006), (-half, half), (0.012, 0.0124), shade(top, 0.85))
        for k in range(1, 12):
            at = -half + 2 * half * k / 12
            m.box((start, edge), (at - 0.0005, at + 0.0005), (0.012, 0.0123), shade(top, 0.9))
    return edge


def _rails(m: Model, u: float, v: tuple[float, float], z: float = 0.0) -> None:
    """A siding along v at u: sleepers on ballast and, up close, two rails."""
    m.box((u - 0.016, u + 0.016), v, (z, z + 0.002), (78, 72, 66))
    with m.fine():
        for side in (u - 0.006, u + 0.006):
            m.box((side - 0.0015, side + 0.0015), v, (z + 0.002, z + 0.0045), (150, 146, 140))


def _lamp(m: Model, u: float, v: float, high: float = 0.09) -> None:
    upright(m, u, v, 0.0025, (0.0, high), (52, 48, 46))
    with m.fine():
        m.box((u - 0.005, u + 0.005), (v - 0.005, v + 0.005), (high - 0.002, high + 0.008), LAMP)


def _net_rack(m: Model, u: float, v: tuple[float, float]) -> None:
    """Fishing nets hung to dry on a rack of two posts and a pole."""
    for at in v:
        m.box((u - 0.003, u + 0.003), (at - 0.003, at + 0.003), (0.0, 0.05), PILES)
    m.box((u - 0.002, u + 0.002), v, (0.046, 0.05), PILES)
    m.box((u - 0.0008, u + 0.0008), (v[0] + 0.004, v[1] - 0.004), (0.012, 0.046), NET)


def _sacks(m: Model, u: float, v: float) -> None:
    for k, (du, dv) in enumerate(((0.0, 0.0), (0.014, 0.004), (0.006, 0.014))):
        m.box(
            (u + du - 0.007, u + du + 0.007),
            (v + dv - 0.005, v + dv + 0.005),
            (0.0, 0.012 + 0.002 * (k % 2)),
            SACK,
        )


def _moved(m: Model, at: tuple[float, float], build: Callable[[Model], None]) -> None:
    """A building built round the model's middle, as windows and doors on both its long walls
    need, and moved to stand at `at`."""
    faces, footprints = len(m.faces), len(m.footprints)
    build(m)
    m.move(faces, *at)
    du, dv = at
    m.footprints[footprints:] = [
        ((u0 + du, u1 + du), (v0 + dv, v1 + dv)) for (u0, u1), (v0, v1) in m.footprints[footprints:]
    ]


def _warehouse(m: Model) -> None:
    u, v, high = (-0.14, 0.14), (-0.15, 0.15), 0.1
    walls(m, u, v, high, BRICK, shade(BRICK, 0.62))
    gable_roof(m, u, v, (high, high + 0.055), SLATE, BRICK)
    for at in ((-0.12, -0.07), (0.07, 0.12)):
        window(m, at, (0.05, 0.08), v[1], (58, 70, 92))
    m.panel((-0.035, 0.035), (0.0, 0.07), v[1] + 0.0006, DOORWAY)


def _office(m: Model) -> None:
    u, v, high = (-0.07, 0.07), (-0.07, 0.07), 0.07
    walls(m, u, v, high, STONE_WALL, shade(STONE_WALL, 0.7))
    gable_roof(m, u, v, (high, high + 0.04), SLATE, STONE_WALL)
    window(m, (-0.04, -0.01), (0.03, 0.05), v[1], (58, 70, 92))
    m.panel((0.015, 0.045), (0.0, 0.05), v[1] + 0.0006, DOORWAY)


def _smokehouse(m: Model) -> None:
    u, v, high = (-0.05, 0.05), (-0.045, 0.045), 0.05
    walls(m, u, v, high, DARK_TIMBER, shade(DARK_TIMBER, 0.7))
    gable_roof(m, u, v, (high, high + 0.03), SHED_ROOF, DARK_TIMBER, 0.008)
    upright(m, 0.03, 0.0, 0.007, (high, high + 0.05), (120, 110, 100))


def _fish_rack(m: Model, u: float, v: float) -> None:
    """A rack of fish hung to dry on a pole between two posts."""
    for at in (v - 0.05, v + 0.05):
        m.box((u - 0.003, u + 0.003), (at - 0.003, at + 0.003), (0.0, 0.045), PILES)
    m.box((u - 0.002, u + 0.002), (v - 0.05, v + 0.05), (0.042, 0.046), PILES)
    for k in range(9):
        at = v - 0.04 + 0.08 * k / 8
        m.box((u - 0.0016, u + 0.0016), (at - 0.002, at + 0.002), (0.024, 0.042), FISH)


def _woodpile(m: Model, u: float, v: float, along: float, across: float, high: float) -> None:
    m.box((u - along, u + along), (v - across, v + across), (0.0, high), LOGS)
    with m.fine():
        for k in range(1, 4):
            z = high * k / 4
            m.box(
                (u - along - 0.001, u + along + 0.001),
                (v - across, v + across),
                (z - 0.0008, z + 0.0008),
                shade(LOGS, 0.7),
            )


def _inside(u: float, v: float, room: float = 0.97) -> tuple[float, float]:
    """The point (u, v) drawn in towards the cell's middle until it lies `room` of the way out
    to the cell's edge at most, so nothing laid on the ground sticks out of the hexagon."""
    apothem = _apothem()
    out = max(abs(u), abs(u) / 2 + abs(v) * math.sqrt(3) / 2) / apothem
    if out <= room:
        return u, v
    return u * room / out, v * room / out


def _blotch(
    m: Model,
    middle: tuple[float, float],
    reach: tuple[float, float],
    seed: int,
    z: float,
    color: RGB,
) -> None:
    """A patch on the ground round `middle`, `reach` out along u and v, its outline a rounded
    oblong frayed a little at random, kept inside the cell."""
    rng = random.Random(seed)
    points = []
    sides = 28
    for k in range(sides):
        a = 2 * math.pi * k / sides
        c, s = math.cos(a), math.sin(a)
        fray = rng.uniform(0.9, 1.0)
        u = middle[0] + reach[0] * fray * math.copysign(abs(c) ** 0.5, c)
        v = middle[1] + reach[1] * fray * math.copysign(abs(s) ** 0.5, s)
        points.append(_inside(u, v))
    m.faces.append(Face([(u, v, z) for u, v in points], color, (0.0, 0.0, 1.0)))


def _worn_patch(
    m: Model, middle: tuple[float, float], reach: tuple[float, float], seed: int, color: RGB
) -> None:
    """A patch of `color` worn into the grass: a frayed rim of bare earth round it."""
    _blotch(m, middle, (reach[0] + 0.04, reach[1] + 0.04), seed + 100, 0.0004, WORN)
    _blotch(m, middle, reach, seed, 0.0007, color)


def _strip(m: Model, points: list[tuple[float, float]], half: float, z: float, color: RGB) -> None:
    """A way `half` either side of the line through `points` lying on the ground: a track, a
    road, rounded at its bends."""
    for (ua, va), (ub, vb) in itertools.pairwise(points):
        length = math.hypot(ub - ua, vb - va) or 1.0
        nu, nv = -(vb - va) / length * half, (ub - ua) / length * half
        corners = [(ua + nu, va + nv), (ub + nu, vb + nv), (ub - nu, vb - nv), (ua - nu, va - nv)]
        m.faces.append(Face([(u, v, z) for u, v in corners], color, (0.0, 0.0, 1.0)))
    for u, v in points[1:-1]:
        _slab(m, _ellipse(u, v, half, half, 10), z, color, color)


def _beam(
    m: Model,
    a: tuple[float, float],
    b: tuple[float, float],
    z: tuple[float, float],
    half: float,
    color: RGB,
) -> None:
    """A long thin block from `a` to `b` on the ground, `half` either side of the line, from
    `z[0]` to `z[1]`: a fence rail, a wall."""
    (ua, va), (ub, vb) = a, b
    length = math.hypot(ub - ua, vb - va) or 1.0
    du, dv = (ub - ua) / length, (vb - va) / length
    nu, nv = -dv, du
    z0, z1 = z
    p = [
        (ua + nu * half, va + nv * half),
        (ub + nu * half, vb + nv * half),
        (ub - nu * half, vb - nv * half),
        (ua - nu * half, va - nv * half),
    ]
    m.faces.append(Face([(u, v, z1) for u, v in p], color, (0.0, 0.0, 1.0)))
    for (u0, v0), (u1, v1), normal in (
        (p[0], p[1], (nu, nv, 0.0)),
        (p[2], p[3], (-nu, -nv, 0.0)),
        (p[1], p[2], (du, dv, 0.0)),
        (p[3], p[0], (-du, -dv, 0.0)),
    ):
        m.faces.append(
            Face([(u0, v0, z0), (u1, v1, z0), (u1, v1, z1), (u0, v0, z1)], color, normal)
        )


def _fence(
    m: Model, points: list[tuple[float, float]], high: float = 0.03, color: RGB = FENCE
) -> None:
    """A post-and-rail fence along the ground through `points`."""
    for a, b in itertools.pairwise(points):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        posts = max(1, round(length / 0.045))
        for k in range(posts + 1):
            t = k / posts
            u, v = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            m.box(
                (u - 0.0025, u + 0.0025), (v - 0.0025, v + 0.0025), (0.0, high), shade(color, 0.7)
            )
        for rail in (high * 0.45, high * 0.9):
            _beam(m, a, b, (rail - 0.003, rail), 0.0012, color)


def _wall(m: Model, points: list[tuple[float, float]], high: float, color: RGB) -> None:
    """A low wall of stone along the ground through `points`, coped along its top."""
    for a, b in itertools.pairwise(points):
        _beam(m, a, b, (0.0, high), 0.008, color)
        _beam(m, a, b, (high, high + 0.004), 0.01, shade(color, 0.85))


def _lumps(
    m: Model, u: float, v: float, z: float, radius: float, color: RGB, seed: int, count: int
) -> None:
    """A heap of leaves `radius` round over (u, v) from `z`: rounded lumps of foliage of a few
    shades, the lower ones out round its rim and one on top."""
    rng = random.Random(seed)
    for k in range(count):
        a = 2 * math.pi * k / count + rng.uniform(-0.3, 0.3)
        out = radius * rng.uniform(0.45, 0.6)
        size = radius * rng.uniform(0.5, 0.65)
        lu, lv = u + out * math.cos(a), v + out * math.sin(a)
        low = z + rng.uniform(0.0, radius * 0.3)
        dome(m, lu, lv, size, (low, low + size * 1.3), shade(color, rng.uniform(0.82, 1.08)))
    top = z + radius * 0.35
    dome(m, u, v, radius * 0.6, (top, top + radius * 1.1), shade(color, 1.12))


def _bush(m: Model, u: float, v: float, radius: float, color: RGB = LEAVES) -> None:
    _lumps(m, u, v, 0.0, radius, shade(color, 0.92), int(u * 1000 + v * 77), 5)


def _tree(m: Model, u: float, v: float, radius: float, color: RGB = LEAVES) -> None:
    """A leafy tree: a trunk forking under a crown of lumps of leaves."""
    trunk = radius * 1.2
    upright(m, u, v, radius * 0.12, (0.0, trunk + radius * 0.5), TRUNK)
    seed = int(u * 1000 + v * 77)
    _lumps(m, u, v, trunk, radius, shade(color, 0.9), seed, 7)
    _lumps(
        m,
        u - radius * 0.1,
        v + radius * 0.1,
        trunk + radius * 0.7,
        radius * 0.75,
        color,
        seed + 1,
        5,
    )


def _garden(m: Model, u: tuple[float, float], v: tuple[float, float]) -> None:
    """A kitchen garden: dug earth in rows of green plants across u, a picket fence round it."""
    (u0, u1), (v0, v1) = u, v
    _slab(m, _rect(u, v), 0.0008, SOIL, SOIL)
    rows = max(2, round((u1 - u0) / 0.024))
    for k in range(rows):
        at = u0 + (u1 - u0) * (k + 0.5) / rows
        for plant in range(max(2, round((v1 - v0) / 0.022))):
            pv = v0 + 0.012 + plant * 0.022
            if pv < v1 - 0.008:
                dome(m, at, pv, 0.007, (0.0, 0.01), CABBAGE if k % 2 else LEAVES)
    _fence(
        m, [(u0 - 0.01, v0 - 0.01), (u1 + 0.01, v0 - 0.01), (u1 + 0.01, v1 + 0.01)], 0.02, PICKET
    )
    _fence(
        m, [(u1 + 0.01, v1 + 0.01), (u0 - 0.01, v1 + 0.01), (u0 - 0.01, v0 - 0.01)], 0.02, PICKET
    )


def _road_out(
    m: Model, points: list[tuple[float, float]], half: float, color: RGB, rim: RGB = WORN
) -> None:
    """A way from the port out to the cell's landward edge through `points`, its first point on
    that edge, so it runs on into the land beyond, worn into the grass along its sides."""
    _strip(m, points, half + 0.014, 0.0005, rim)
    _strip(m, points, half, 0.0009, color)


def _cottage(m: Model) -> None:
    """A fisherman's cottage: whitewashed walls under a red tiled roof, a chimney smoking, its
    door and windows on the long walls."""
    u, v, high = (-0.08, 0.08), (-0.05, 0.05), 0.055
    walls(m, u, v, high, WHITEWASH, shade(WHITEWASH, 0.75))
    gable_roof(m, u, v, (high, high + 0.045), RED_ROOF, WHITEWASH, 0.012)
    door(m, (-0.012, 0.012), 0.04, v[1])
    for at in ((-0.06, -0.035), (0.035, 0.06)):
        window(m, at, (0.022, 0.04), v[1], WINDOW)
    chimney(m, 0.05, 0.0, (high + 0.02, high + 0.06), (196, 190, 180))
    for k, (du, dz, r) in enumerate(((0.0, 0.07, 0.008), (0.008, 0.085, 0.011))):
        dome(m, 0.05 + du, -du, r, (high + dz, high + dz + r * 1.4), shade(SMOKE, 1 - 0.04 * k))


def _boathouse(m: Model) -> None:
    """A boathouse of red-painted boards with white corners, its wide doors open to the water at
    its end along +u."""
    u, v, high = (-0.1, 0.1), (-0.055, 0.055), 0.055
    walls(m, u, v, high, BOARDS, shade(BOARDS, 0.7))
    gable_roof(m, u, v, (high, high + 0.04), SHED_ROOF, BOARDS, 0.01)
    m.box((u[1] - 0.0002, u[1] + 0.0016), (-0.035, 0.035), (0.0, 0.045), DOORWAY)
    with m.fine():
        for cu in u:
            for cv in v:
                m.box((cu - 0.003, cu + 0.003), (cv - 0.003, cv + 0.003), (0.0, high), WHITEWASH)
        for k in range(1, 12):
            at = u[0] + (u[1] - u[0]) * k / 12
            m.panel(
                (at - 0.0006, at + 0.0006),
                (0.004, high - 0.002),
                v[1] + 0.0008,
                shade(BOARDS, 0.75),
            )
    window(m, (-0.05, -0.02), (0.022, 0.04), v[1], WINDOW)


def _chain_link(m: Model, points: list[tuple[float, float]], high: float = 0.04) -> None:
    """A wire fence on steel posts along the ground through `points`."""
    for a, b in itertools.pairwise(points):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        posts = max(1, round(length / 0.06))
        for k in range(posts + 1):
            t = k / posts
            u, v = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            upright(m, u, v, 0.0022, (0.0, high), STEEL_POST)
        for wire in (high * 0.35, high * 0.68, high):
            _beam(m, a, b, (wire - 0.0016, wire), 0.0008, STEEL_POST)


def _greenery(m: Model, seed: int) -> None:
    """Trees and bushes in the corners of the cell either side of a port's yard, where its
    cell reaches furthest out along v, so the port stands among the land round it."""
    rng = random.Random(seed)
    for side in (-1, 1):
        for u, v, kind in (
            (-0.16, 0.6, "tree"),
            (0.06, 0.62, "tree"),
            (-0.34, 0.56, "bush"),
            (-0.04, 0.72, "bush"),
            (0.2, 0.58, "bush"),
            (-0.08, 0.53, "bush"),
        ):
            u, v = _inside(u + rng.uniform(-0.03, 0.03), side * (v + rng.uniform(-0.03, 0.02)), 0.9)
            if kind == "tree":
                _tree(m, u, v, rng.uniform(0.05, 0.065), shade(LEAVES, rng.uniform(0.9, 1.1)))
            else:
                _bush(m, u, v, rng.uniform(0.022, 0.032))


def _beached(m: Model, u: float, v: float, angle: float, color: RGB) -> None:
    """A rowing boat pulled up on the beach, lying on its keel."""
    placed(m, rowing_boat(color), (u, v), angle)


def port() -> Model:
    """A fishing harbour: a beach of shingle along the shore with boats pulled up on it, a
    wooden jetty on piles out over the water with fishing boats moored at it, a trodden yard
    with a red boathouse, racks of fish and nets drying, barrels and crates, a whitewashed
    cottage with its kitchen garden behind, and a cart track out to the land, a fence along it
    and trees and bushes round about."""
    m = Model()
    edge = _apothem() * 0.96
    _blotch(m, (0.5, 0.0), (0.13, 0.46), 3, 0.0006, SHINGLE)
    rng = random.Random(4)
    with m.fine():
        for _ in range(160):
            pu, pv = rng.uniform(0.39, 0.61), rng.uniform(-0.38, 0.38)
            pu, pv = _inside(pu, pv, 0.95)
            tone = shade(SHINGLE, rng.choice((0.78, 0.88, 1.1)))
            _slab(m, _ellipse(pu, pv, 0.004, 0.003, 5), 0.0011, tone, tone)
    _worn_patch(m, (0.04, -0.02), (0.26, 0.24), 5, YARD_EARTH)
    track = [(-_apothem(), 0.14), (-0.46, 0.1), (-0.2, 0.02), (0.1, -0.02), (0.38, 0.02)]
    _road_out(m, track, 0.032, YARD_EARTH)
    _ruts(m, track[1:])
    _fence(m, [(-0.68, 0.22), (-0.46, 0.19), (-0.26, 0.12)])
    _puddle(m, -0.1, 0.06, 0.03, 0.018)
    _moved(m, (0.26, -0.24), _boathouse)
    _moved(m, (-0.34, -0.2), _cottage)
    _garden(m, (-0.5, -0.32), (-0.46, -0.3))
    _woodpile(m, -0.2, -0.3, 0.03, 0.015, 0.025)
    _moved(m, (-0.02, 0.24), _smokehouse)
    _fish_rack(m, 0.18, 0.2)
    _fish_rack(m, 0.3, 0.26)
    _net_rack(m, -0.04, (-0.2, -0.12))
    for k, (bu, bv) in enumerate(((0.36, -0.12), (0.38, -0.09), (0.35, -0.07), (0.1, -0.2))):
        _barrel(m, bu, bv)
        if k == 3:
            _crate(m, bu + 0.03, bv + 0.01)
    _crate(m, 0.34, 0.12)
    _crate(m, 0.36, 0.14, 0.024)
    placed(m, horse_cart(), (-0.3, 0.06), 10)
    _beached(m, 0.48, -0.28, 75, BOAT_BLUE)
    _beached(m, 0.52, -0.18, 95, BOAT_WHITE)
    _beached(m, 0.5, 0.28, 105, BOAT_GREEN)
    _deck(m, (0.44, edge + 0.36), 0.026, 0.06, 0.014, PLANKS, PILES, 0.07)
    placed(m, fishing_boat(BOAT_BLUE), (edge + 0.2, 0.12), 0)
    placed(m, fishing_boat(BOAT_WHITE), (edge + 0.28, -0.0), 180)
    _tree(m, -0.12, 0.5, 0.06)
    _tree(m, 0.06, 0.6, 0.05, shade(LEAVES, 1.1))
    _tree(m, -0.56, -0.08, 0.055)
    _tree(m, -0.06, -0.58, 0.05)
    for bu, bv, r in (
        (-0.3, 0.42, 0.03),
        (-0.22, 0.56, 0.025),
        (0.22, 0.46, 0.028),
        (-0.5, 0.32, 0.03),
        (0.14, -0.5, 0.026),
        (-0.2, -0.5, 0.022),
    ):
        _bush(m, bu, bv, r)
    return m


def big_port() -> Model:
    """A merchant harbour filling its cell: a cobbled quay along the shore with a siding of
    wagons along it, a brick warehouse and a stone harbour office on gravel behind, heaps of coal
    and sand with dump trucks taking them, timber stacked, two cranes, one on the quay and one on
    the stone pier lifting cargo out of a steamer moored alongside, crates, barrels, sacks and
    lamps."""
    m = Model()
    _gravel(m, (-0.48, 0.27), (-0.46, 0.48), 2)
    edge = _quay(m, 0.26, 0.42, COBBLES, QUAY_WALL)
    _ruts(m, [(-0.4, 0.02), (-0.1, 0.1), (0.2, 0.06)])
    _rails(m, 0.34, (-0.42, 0.42), 0.012)
    for at, part in ((-0.36, open_wagon(COAL)), (-0.24, open_wagon(COAL)), (-0.08, box_wagon())):
        placed(m, part, (0.34, at), 90)
    _moved(m, (-0.3, -0.24), _warehouse)
    _moved(m, (-0.3, 0.26), _office)
    cone(m, 0.0, -0.38, 0.08, 0.055, COAL)
    cone(m, 0.1, 0.36, 0.08, 0.05, SAND)
    placed(m, dump_truck(TRUCK_YELLOW, COAL), (0.1, -0.3), 100)
    placed(m, dump_truck(TRUCK_BLUE, SAND), (-0.02, 0.24), -60)
    _woodpile(m, -0.1, 0.42, 0.06, 0.03, 0.04)
    _woodpile(m, -0.22, 0.42, 0.05, 0.03, 0.03)
    _deck(m, (edge, edge + 0.8), 0.05, 0.18, 0.014, QUAY, None, 0.1)
    placed(m, steamer(), (edge + 0.42, 0.18 + 0.05 + 0.05), 0)
    placed(m, crane(0.3, 0.4, CRATE), (edge - 0.1, -0.1), 0)
    placed(m, crane(0.18, 0.36, (120, 96, 70)), (edge + 0.32, 0.18), 90)
    for cu, cv in ((0.5, 0.02), (0.53, 0.04), (0.51, 0.06), (0.56, 0.02)):
        _crate(m, cu, cv, 0.012)
    _barrel(m, 0.6, 0.22, 0.012)
    _barrel(m, 0.62, 0.25, 0.012)
    _sacks(m, 0.48, 0.3)
    for side in (-1, 1):
        _lamp(m, edge - 0.1, side * _across(edge, _apothem()) * 0.6)
    _road_out(m, [(-_apothem(), 0.03), (-0.6, 0.03), (-0.46, 0.035)], 0.045, COBBLES)
    _wall(m, [(0.24, -0.49), (-0.5, -0.49), (-0.5, -0.02)], 0.03, STONE_WALL)
    _wall(m, [(-0.5, 0.09), (-0.5, 0.49), (0.24, 0.49)], 0.03, STONE_WALL)
    for gate in (-0.02, 0.09):
        m.box((-0.512, -0.488), (gate - 0.012, gate + 0.012), (0.0, 0.05), shade(STONE_WALL, 0.9))
        _lamp(m, -0.53, gate + (0.03 if gate > 0 else -0.03), 0.08)
    _greenery(m, 6)
    return m


def mega_port() -> Model:
    """A container terminal filling its cell: a broad concrete quay along the shore with yellow
    edges and a siding of container wagons, its yard of asphalt marked in lanes and stacked
    with blocks of ribbed containers in many colours, lorries carrying containers, a long steel
    warehouse, tall lamp masts, and two concrete piers with a container ship moored between
    them, their two gantry cranes reaching over it."""
    m = Model()
    _yard(m, (-0.54, 0.3), (-0.45, 0.45), ASPHALT)
    with m.fine():
        for lane in (-0.28, 0.06):
            m.box((lane - 0.002, lane + 0.002), (-0.44, 0.44), (0.0007, 0.0013), WHITE_LINE)
        for lane in (-0.22, 0.38):
            for k in range(8):
                at = -0.5 + k * 0.1
                m.box((at, at + 0.05), (lane - 0.002, lane + 0.002), (0.0007, 0.0013), WHITE_LINE)
    edge = _quay(m, 0.3, 0.42, CONCRETE, shade(CONCRETE, 0.7))
    with m.fine():
        m.box((edge - 0.012, edge), (-0.42, 0.42), (0.012, 0.0128), EDGE)
    _rails(m, 0.2, (-0.45, 0.45))
    for k, at in enumerate((-0.42, -0.3, -0.18, 0.3, 0.42)):
        placed(m, flat_wagon(CONTAINERS[k % len(CONTAINERS)]), (0.2, at), 90)
    u, v, high = (-0.52, -0.36), (-0.43, -0.02), 0.1
    walls(m, u, v, high, HALL, shade(HALL, 0.6))
    gable_roof(m, u, v, (high, high + 0.025), HALL_ROOF, HALL, 0.008)
    with m.fine():
        for k in range(1, 12):
            at = v[0] + (v[1] - v[0]) * k / 12
            m.box(
                (u[1] - 0.0004, u[1] + 0.0012),
                (at - 0.0012, at + 0.0012),
                (0.004, high - 0.004),
                shade(HALL, 0.75),
            )
    for at in (-0.38, -0.24, -0.1):
        m.box((u[1] - 0.0002, u[1] + 0.0016), (at - 0.03, at + 0.03), (0.0, 0.07), DOORWAY)
    for k, (su, sv) in enumerate(
        (
            (-0.46, 0.06),
            (-0.36, 0.06),
            (-0.26, 0.06),
            (-0.16, 0.06),
            (-0.46, 0.24),
            (-0.36, 0.24),
            (-0.26, 0.24),
            (-0.2, -0.44),
            (-0.1, -0.44),
        )
    ):
        _stack(m, su, sv, 5 if sv > 0 else 4, (3, 2, 4, 1, 3), k, True)
    placed(m, container_lorry(CONTAINERS[3]), (-0.02, -0.12), 90)
    placed(m, container_lorry(CONTAINERS[0]), (0.08, 0.3), -90)
    placed(m, container_lorry(CONTAINERS[2]), (-0.12, 0.42), 0)
    for lamp in ((-0.3, -0.42), (-0.06, 0.43), (0.05, -0.32), (0.05, 0.12)):
        _lamp(m, *lamp, 0.22)
    for middle in (-0.24, 0.24):
        _deck(m, (edge, edge + 0.86), 0.065, middle, 0.012, CONCRETE, None, 0.1)
        with m.fine():
            for side in (middle - 0.063, middle + 0.063):
                m.box((edge, edge + 0.86), (side - 0.002, side + 0.002), (0.012, 0.0128), EDGE)
    placed(m, container_ship(CONTAINERS), (edge + 0.45, 0.0), 0)
    placed(m, gantry(0.36, 0.46, 0.04), (edge + 0.3, -0.24), 90)
    placed(m, gantry(0.36, 0.46, 0.04), (edge + 0.6, 0.24), -90)
    _road_out(m, [(-_apothem(), 0.02), (-0.53, 0.02)], 0.04, ASPHALT)
    with m.fine():
        for at in (-0.69, -0.62):
            m.box((at, at + 0.035), (0.018, 0.022), (0.0009, 0.0014), WHITE_LINE)
    _chain_link(m, [(0.28, -0.48), (-0.56, -0.48), (-0.56, -0.03)])
    _chain_link(m, [(-0.56, 0.07), (-0.56, 0.48), (0.28, 0.48)])
    walls(m, (-0.62, -0.58), (-0.08, -0.04), 0.04, CABIN)
    m.box((-0.625, -0.575), (-0.085, -0.035), (0.04, 0.046), HALL_ROOF)
    with m.fine():
        m.panel((-0.615, -0.59), (0.02, 0.034), -0.0394, GLASS)
    for k in range(6):
        at = -0.03 + 0.016 * k
        _beam(m, (-0.6, at), (-0.6, at + 0.016), (0.026, 0.031), 0.0018, GANTRY if k % 2 else CABIN)
    _greenery(m, 7)
    return m


def port_sprite(kind: str, heading: int) -> str:
    """The sprite of a port of a kind ("port", "big_port" or "mega_port") whose piers run
    `heading` degrees on the map."""
    return f"{kind}_{heading}"


_MODELS: dict[str, Callable[[], Model]] = {
    "port": port,
    "big_port": big_port,
    "mega_port": mega_port,
}

PORTS: dict[str, Callable[[], Sprite]] = {
    **{
        name: modelled(build, shadow=0.0, heading=math.radians(PORT_HEADINGS[1]))
        for name, build in _MODELS.items()
    },
    **{
        port_sprite(name, heading): modelled(build, shadow=0.0, heading=math.radians(heading))
        for name, build in _MODELS.items()
        for heading in PORT_HEADINGS
    },
}
