"""Ports (railroads.sim.ports) built as small lit models (railroads.ui.pixel.art.built) at the
works' scale, filling their cell to its edges with a yard and, but for the fishing harbour, a quay
along the shore, and only their piers reaching out of it over the water beside it: a port is a
timber shed and a short wooden jetty on piles, a big port a brick warehouse, a harbour office, a
crane and a stone pier, and a mega port a container terminal, a long warehouse and stacks of
ribbed containers by two concrete piers, each with a gantry crane over the water. Each is drawn
turned towards each way the water can lie beside its cell (PORT_HEADINGS), its piers running out
along u, so it stands in the middle of its cell, its yard filling it.

Models are in the works' cells: u and v across the ground, z up, the cell's middle at the
origin."""

import math
from collections.abc import Callable

from railroads.ui.pixel.art.built import cone, gable_roof, modelled, upright, walls, window
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
HULL: RGB = (92, 62, 44)
CRANE: RGB = (214, 168, 40)
GANTRY: RGB = (196, 74, 52)
CABIN: RGB = (232, 230, 222)
CABLE: RGB = (60, 58, 56)
HALL: RGB = (150, 166, 178)
HALL_ROOF: RGB = (112, 126, 138)
WORN_EARTH: RGB = (138, 120, 86)
GRAVEL: RGB = (150, 142, 124)
COBBLES: RGB = (164, 158, 148)
QUAY_WALL: RGB = (112, 108, 100)
STONE_WALL: RGB = (196, 188, 172)
ASPHALT: RGB = (92, 94, 98)
LAMP: RGB = (250, 230, 160)
NET: RGB = (70, 86, 80)
SACK: RGB = (204, 186, 146)
DARK_TIMBER: RGB = (122, 96, 70)
FISH: RGB = (190, 196, 200)
LOGS: RGB = (140, 102, 66)
TUFT: RGB = (92, 128, 62)
TRACK: RGB = (160, 138, 100)
COAL: RGB = (40, 40, 44)
SAND: RGB = (214, 190, 132)
WHITE_LINE: RGB = (226, 226, 220)
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


def _boat(m: Model, u: float, v: float, length: float, color: RGB = HULL) -> None:
    """A rowing boat moored at (u, v), lying along u."""
    m.box((u - length / 2, u + length / 2), (v - 0.012, v + 0.012), (0.0, 0.01), color)
    with m.fine():
        m.box(
            (u - length / 2 + 0.006, u + length / 2 - 0.006),
            (v - 0.008, v + 0.008),
            (0.01, 0.0104),
            shade(color, 1.4),
        )


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


def _yard(m: Model, color: RGB, inset: float = 0.94) -> None:
    """The port's yard lying on the ground, the cell's hexagon a little inside its edges."""
    radius = _apothem() * inset * 2 / math.sqrt(3)
    corners = [
        (radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)))
        for a in range(-30, 330, 60)
    ]
    _slab(m, corners, 0.0006, color, color)


def _quay(m: Model, start: float, top: RGB, wall: RGB) -> float:
    """A paved quay from `start` along u to the cell's edge over the water, as wide as the cell
    there, with its wall and, up close, slabs along it; returns where its edge stands."""
    edge = _apothem() * 0.97
    near, far = _across(start, edge / 0.97) * 0.97, _across(edge, edge / 0.97) * 0.97
    _slab(m, [(start, -near), (edge, -far), (edge, far), (start, near)], 0.012, top, wall)
    with m.fine():
        rows = 6
        for k in range(1, rows):
            at = start + (edge - start) * k / rows
            wide = _across(at, edge / 0.97) * 0.97
            m.box((at - 0.0006, at + 0.0006), (-wide, wide), (0.012, 0.0124), shade(top, 0.85))
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


def _shed(m: Model) -> None:
    u, v, high = (-0.08, 0.08), (-0.07, 0.07), 0.06
    walls(m, u, v, high, SHED, shade(SHED, 0.7))
    gable_roof(m, u, v, (high, high + 0.04), SHED_ROOF, SHED, 0.01)
    m.panel((-0.04, 0.02), (0.0, 0.045), v[1] + 0.0006, DOORWAY)


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


def _grass(m: Model, points: list[tuple[float, float]]) -> None:
    _slab(m, points, 0.0009, TUFT, TUFT)


def port() -> Model:
    """A fishing harbour: a yard of trodden earth filling its cell, worn into a track from the
    shed to the shore, with tufts of grass at its edges; a timber shed, a smokehouse, racks of
    fish drying, nets, a woodpile, crates and barrels, a horse and cart, a boat pulled up on the
    beach, and a wooden jetty on piles running out from a boardwalk, fishing boats moored at
    it."""
    m = Model()
    _yard(m, WORN_EARTH)
    _grass(m, [(-0.62, -0.1), (-0.5, -0.32), (-0.36, -0.2), (-0.46, 0.02)])
    _grass(m, [(-0.2, 0.5), (0.0, 0.62), (0.08, 0.5), (-0.1, 0.42)])
    _slab(m, [(-0.1, -0.2), (0.5, -0.04), (0.5, 0.06), (-0.1, -0.1)], 0.0012, TRACK, TRACK)
    edge = _apothem() * 0.96
    _slab(
        m,
        [(edge - 0.2, -0.12), (edge, -0.12), (edge, 0.14), (edge - 0.2, 0.14)],
        0.006,
        PLANKS,
        PILES,
    )
    _moved(m, (-0.24, -0.3), _shed)
    _moved(m, (-0.36, 0.14), _smokehouse)
    _fish_rack(m, -0.08, 0.26)
    _fish_rack(m, 0.06, 0.32)
    _net_rack(m, -0.34, (-0.02, -0.1))
    _woodpile(m, -0.46, -0.08, 0.04, 0.02, 0.03)
    _barrel(m, -0.1, -0.18)
    _barrel(m, -0.08, -0.15)
    _barrel(m, -0.11, -0.14)
    _crate(m, -0.14, -0.13)
    _crate(m, 0.5, 0.1, 0.006)
    _crate(m, 0.48, 0.08, 0.006)
    _crate(m, 0.52, 0.12, 0.006)
    for k in range(4):
        _barrel(m, 0.36 + 0.022 * k, 0.2 + 0.01 * (k % 2))
    _fish_rack(m, 0.2, 0.36)
    upright(m, -0.02, 0.0, 0.016, (0.0, 0.016), (150, 146, 136))
    upright(m, -0.02, 0.0, 0.011, (0.0, 0.0165), (40, 60, 70))
    placed(m, rowing_boat(), (-0.12, 0.48), 120)
    _woodpile(m, 0.04, -0.42, 0.05, 0.025, 0.035)
    placed(m, horse_cart(), (0.22, -0.1), 15)
    placed(m, rowing_boat(), (0.46, -0.4), 35)
    _deck(m, (edge, edge + 0.5), 0.024, 0.02, 0.014, PLANKS, PILES, 0.08)
    placed(m, fishing_boat(), (edge + 0.26, 0.075), 0)
    placed(m, fishing_boat(), (edge + 0.36, -0.035), 180)
    return m


def big_port() -> Model:
    """A merchant harbour filling its cell: a cobbled quay along the shore with a siding of
    wagons along it, a brick warehouse and a stone harbour office on gravel behind, heaps of coal
    and sand with dump trucks taking them, timber stacked, two cranes, one on the quay and one on
    the stone pier lifting cargo out of a steamer moored alongside, crates, barrels, sacks and
    lamps."""
    m = Model()
    _yard(m, GRAVEL)
    _grass(m, [(-0.64, 0.05), (-0.5, -0.24), (-0.44, 0.0)])
    edge = _quay(m, 0.26, COBBLES, QUAY_WALL)
    reach = _across(0.34, _apothem()) * 0.85
    _rails(m, 0.34, (-reach, reach), 0.012)
    for at, part in ((-0.36, open_wagon(COAL)), (-0.24, open_wagon(COAL)), (-0.08, box_wagon())):
        placed(m, part, (0.34, at), 90)
    _moved(m, (-0.3, -0.24), _warehouse)
    _moved(m, (-0.3, 0.26), _office)
    cone(m, 0.02, -0.46, 0.09, 0.06, COAL)
    cone(m, 0.1, 0.36, 0.08, 0.05, SAND)
    placed(m, dump_truck(TRUCK_YELLOW, COAL), (0.1, -0.3), 100)
    placed(m, dump_truck(TRUCK_BLUE, SAND), (-0.02, 0.24), -60)
    _woodpile(m, -0.1, 0.5, 0.06, 0.03, 0.04)
    _woodpile(m, -0.22, 0.5, 0.05, 0.03, 0.03)
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
    return m


def mega_port() -> Model:
    """A container terminal filling its cell: a broad concrete quay along the shore with yellow
    edges and a siding of container wagons, its yard of asphalt marked in lanes and stacked
    with blocks of ribbed containers in many colours, lorries carrying containers, a long steel
    warehouse, tall lamp masts, and two concrete piers with a container ship moored between
    them, their two gantry cranes reaching over it."""
    m = Model()
    _yard(m, ASPHALT, 0.97)
    with m.fine():
        for lane in (-0.28, 0.06):
            m.box((lane - 0.002, lane + 0.002), (-0.55, 0.55), (0.0006, 0.0012), WHITE_LINE)
    edge = _quay(m, 0.3, CONCRETE, shade(CONCRETE, 0.7))
    with m.fine():
        far = _across(edge, _apothem()) * 0.97
        m.box((edge - 0.012, edge), (-far, far), (0.012, 0.0128), EDGE)
    reach = _across(0.2, _apothem()) * 0.85
    _rails(m, 0.2, (-reach, reach))
    for k, at in enumerate((-0.42, -0.3, -0.18, 0.3, 0.42)):
        placed(m, flat_wagon(CONTAINERS[k % len(CONTAINERS)]), (0.2, at), 90)
    u, v, high = (-0.54, -0.36), (-0.46, -0.02), 0.1
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
    placed(m, container_lorry(CONTAINERS[2]), (-0.12, 0.5), 0)
    for lamp in ((-0.3, -0.5), (-0.06, 0.6), (0.05, -0.32), (0.05, 0.12)):
        _lamp(m, *lamp, 0.22)
    for middle in (-0.24, 0.24):
        _deck(m, (edge, edge + 0.86), 0.065, middle, 0.012, CONCRETE, None, 0.1)
        with m.fine():
            for side in (middle - 0.063, middle + 0.063):
                m.box((edge, edge + 0.86), (side - 0.002, side + 0.002), (0.012, 0.0128), EDGE)
    placed(m, container_ship(CONTAINERS), (edge + 0.45, 0.0), 0)
    placed(m, gantry(0.36, 0.46, 0.04), (edge + 0.3, -0.24), 90)
    placed(m, gantry(0.36, 0.46, 0.04), (edge + 0.6, 0.24), -90)
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
        name: modelled(build, shadow=0.8, heading=math.radians(PORT_HEADINGS[1]))
        for name, build in _MODELS.items()
    },
    **{
        port_sprite(name, heading): modelled(build, shadow=0.8, heading=math.radians(heading))
        for name, build in _MODELS.items()
        for heading in PORT_HEADINGS
    },
}
