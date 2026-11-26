"""Ports (railroads.sim.ports) built as small lit models (railroads.ui.pixel.art.built), true to
the map's size so their piers reach out over the water beside them: a port is a timber shed and a
long wooden jetty on piles, a big port a brick warehouse, a crane and a stone pier, and a mega
port a container terminal, a long warehouse and stacks of ribbed containers by two concrete piers,
each with a gantry crane over the water. Each is drawn turned towards each way the water can lie
beside its cell (PORT_HEADINGS), its piers running out along u.

Models are in cells: u and v across the ground, z up, the cell's middle at the origin."""

import math
from collections.abc import Callable

from railroads.ui.pixel.art.built import gable_roof, modelled, upright, walls, window
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


def _crane(m: Model, u: float, v: float, reach: float, high: float, load: RGB) -> None:
    """A slewing crane standing at (u, v), its jib reaching `reach` out along u over the water,
    a cabin under the jib, a counterweight behind it and a load on its hook."""
    m.box((u - 0.02, u + 0.02), (v - 0.02, v + 0.02), (0.0, 0.012), shade(CRANE, 0.6))
    m.box((u - 0.006, u + 0.006), (v - 0.006, v + 0.006), (0.012, high), CRANE)
    with m.fine():
        rungs = 6
        for k in range(1, rungs):
            z = 0.012 + (high - 0.012) * k / rungs
            m.box(
                (u - 0.0068, u + 0.0068),
                (v - 0.0068, v + 0.0068),
                (z, z + 0.002),
                shade(CRANE, 0.7),
            )
    m.box((u - 0.012, u + 0.012), (v - 0.011, v + 0.011), (high - 0.03, high - 0.008), CABIN)
    m.box((u - 0.07, u + reach), (v - 0.004, v + 0.004), (high, high + 0.008), CRANE)
    m.box(
        (u - 0.075, u - 0.045),
        (v - 0.012, v + 0.012),
        (high - 0.016, high + 0.004),
        shade(CRANE, 0.45),
    )
    hook = u + reach - 0.01
    m.box((hook - 0.0012, hook + 0.0012), (v - 0.0012, v + 0.0012), (0.06, high), CABLE)
    m.box((hook - 0.012, hook + 0.012), (v - 0.012, v + 0.012), (0.036, 0.06), load)


def _gantry(m: Model, u: float, v: float, half: float, reach: float, high: float) -> None:
    """A container gantry crane straddling a pier at (u, v), its legs `half` either side, its
    boom `reach` out along u over the water, with its machinery house, its trolley and a
    container hanging under it."""
    legs = ((u - 0.03, v - half), (u - 0.03, v + half), (u + 0.03, v - half), (u + 0.03, v + half))
    for lu, lv in legs:
        m.box((lu - 0.005, lu + 0.005), (lv - 0.005, lv + 0.005), (0.0, high), GANTRY)
    for side in (v - half, v + half):
        m.box(
            (u - 0.035, u + 0.035),
            (side - 0.004, side + 0.004),
            (high * 0.45, high * 0.45 + 0.008),
            shade(GANTRY, 0.8),
        )
        m.box((u - 0.12, u + reach), (side - 0.004, side + 0.004), (high, high + 0.012), GANTRY)
    m.box(
        (u - 0.12, u + reach),
        (v - half, v + half),
        (high + 0.012, high + 0.016),
        shade(GANTRY, 0.75),
    )
    m.box((u - 0.12, u - 0.05), (v - half, v + half), (high + 0.016, high + 0.04), CABIN)
    trolley = u + reach * 0.7
    m.box((trolley - 0.014, trolley + 0.014), (v - 0.014, v + 0.014), (high - 0.012, high), CABIN)
    for side in (v - 0.008, v + 0.008):
        m.box(
            (trolley - 0.001, trolley + 0.001),
            (side - 0.001, side + 0.001),
            (0.075, high - 0.012),
            CABLE,
        )
    _container(m, trolley, v, 0.045, CONTAINERS[1], True)


def _ground(m: Model, u: tuple[float, float], v: tuple[float, float], color: RGB, z: float) -> None:
    """A flat patch of yard lying on the ground, `z` up so the later ones lie over the earlier."""
    (u0, u1), (v0, v1) = u, v
    corners = [(u0, v0, z), (u1, v0, z), (u1, v1, z), (u0, v1, z)]
    m.faces.append(Face(corners, color, (0.0, 0.0, 1.0)))


def _quay(m: Model, u: tuple[float, float], v: tuple[float, float], top: RGB, wall: RGB) -> None:
    """A paved quay along the shore, `u[1]` its edge over the water, with its stone wall facing
    the water and, up close, a row of slabs along it."""
    (u0, u1), (v0, v1) = u, v
    m.box((u0, u1), (v0, v1), (0.0, 0.012), top)
    m.box((u1 - 0.012, u1), (v0, v1), (0.0, 0.0126), wall)
    with m.fine():
        rows = max(2, round((v1 - v0) / 0.03))
        for k in range(1, rows):
            at = v0 + (v1 - v0) * k / rows
            m.box((u0, u1 - 0.012), (at - 0.0006, at + 0.0006), (0.012, 0.0124), shade(top, 0.85))


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


def port() -> Model:
    """A fishing harbour: a timber shed on a yard of trodden earth by the shore, nets drying on a
    rack, barrels and crates, a boat pulled up on the beach, and a short wooden jetty on piles
    running out from a boardwalk on the shore beside the shed, a rowing boat moored at its end."""
    m = Model()
    _ground(m, (-0.3, 0.3), (-0.32, 0.18), WORN_EARTH, 0.0006)
    _ground(m, (0.16, 0.3), (-0.06, 0.12), PLANKS, 0.004)
    _moved(m, (-0.14, -0.21), _shed)
    _barrel(m, -0.04, -0.1)
    _barrel(m, -0.02, -0.08)
    _crate(m, -0.08, -0.08)
    _net_rack(m, -0.12, (0.0, 0.1))
    _boat(m, 0.12, -0.22, 0.08)
    _deck(m, (0.28, 0.63), 0.024, 0.03, 0.014, PLANKS, PILES, 0.08)
    _boat(m, 0.56, 0.075, 0.07)
    return m


def big_port() -> Model:
    """A merchant harbour: a cobbled quay along the shore with a siding running along it, a brick
    warehouse and a stone harbour office behind, a crane on the quay's edge lifting a crate over
    the water, crates, barrels and sacks, lamps, and a stone pier running out from the quay with a
    boat moored alongside."""
    m = Model()
    _ground(m, (-0.42, 0.0), (-0.42, 0.4), GRAVEL, 0.0006)
    _quay(m, (-0.02, 0.32), (-0.42, 0.4), COBBLES, QUAY_WALL)
    _rails(m, 0.04, (-0.42, 0.4), 0.012)
    _moved(m, (-0.24, -0.23), _warehouse)
    _moved(m, (-0.23, 0.19), _office)
    _deck(m, (0.3, 0.87), 0.05, 0.16, 0.014, QUAY, None, 0.1)
    _crane(m, 0.22, -0.16, 0.2, 0.22, CRATE)
    for cu, cv in ((0.12, -0.04), (0.14, -0.02), (0.13, -0.035)):
        _crate(m, cu, cv, 0.012)
    _barrel(m, 0.2, 0.02, 0.012)
    _barrel(m, 0.22, 0.04, 0.012)
    _sacks(m, 0.12, 0.26)
    _lamp(m, 0.26, -0.32)
    _lamp(m, 0.26, 0.3)
    _boat(m, 0.7, 0.23, 0.1, (60, 76, 96))
    return m


def mega_port() -> Model:
    """A container terminal: a broad concrete quay along the shore with yellow edges, its yard of
    asphalt behind stacked with ribbed containers in many colours by a long steel warehouse, a
    siding, tall lamp masts, and two concrete piers running out from the quay side by side, each
    with a gantry crane lifting a container."""
    m = Model()
    _ground(m, (-0.46, 0.0), (-0.46, 0.46), ASPHALT, 0.0006)
    _quay(m, (-0.02, 0.32), (-0.46, 0.46), CONCRETE, shade(CONCRETE, 0.7))
    with m.fine():
        m.box((0.306, 0.32), (-0.46, 0.46), (0.012, 0.0128), EDGE)
    _rails(m, -0.04, (-0.46, 0.46))
    u, v, high = (-0.44, -0.26), (-0.42, 0.02), 0.1
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
    for at in (-0.32, -0.2, -0.08):
        m.box((u[1] - 0.0002, u[1] + 0.0016), (at - 0.03, at + 0.03), (0.0, 0.07), DOORWAY)
    _stack(m, -0.36, 0.1, 4, (3, 2, 3, 1), 0, True)
    _stack(m, -0.26, 0.1, 4, (2, 3, 1, 2), 2, True)
    _stack(m, -0.16, 0.1, 4, (1, 3, 2, 3), 4, True)
    _stack(m, 0.08, -0.4, 3, (2, 1, 2), 1, True)
    _stack(m, 0.18, -0.4, 2, (1, 2), 3, True)
    for lamp in ((-0.1, -0.44), (-0.1, 0.44), (0.24, 0.0)):
        _lamp(m, *lamp, 0.2)
    for middle in (-0.22, 0.22):
        _deck(m, (0.3, 0.9), 0.065, middle, 0.012, CONCRETE, None, 0.1)
        with m.fine():
            for side in (middle - 0.063, middle + 0.063):
                m.box((0.3, 0.9), (side - 0.002, side + 0.002), (0.012, 0.0128), EDGE)
    _gantry(m, 0.48, -0.22, 0.05, 0.34, 0.28)
    _gantry(m, 0.62, 0.22, 0.05, 0.3, 0.28)
    _container(m, 0.4, 0.22, 0.012, CONTAINERS[2], True)
    _container(m, 0.74, -0.22, 0.012, CONTAINERS[3], True)
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
