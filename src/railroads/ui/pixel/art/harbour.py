"""What stands and floats in a port (railroads.ui.pixel.art.ports), built as small lit models
along u from their middle: ships moored at the piers, from a sailing fishing boat to a steamer
and a container ship, and what carries goods on land, a horse and cart, dump trucks, container
lorries and railway wagons.

Models are in the works' cells: u and v across the ground, z up."""

import math
from collections.abc import Callable
from dataclasses import replace

from railroads.ui.pixel.art.built import cone, upright
from railroads.ui.pixel.art.painter import RGB, shade
from railroads.ui.pixel.models import Face, Model

WOOD: RGB = (128, 92, 60)
DARK_WOOD: RGB = (86, 60, 42)
HORSE: RGB = (112, 72, 44)
MANE: RGB = (52, 36, 28)
TYRE: RGB = (34, 34, 36)
STEEL: RGB = (120, 124, 130)
WHITE: RGB = (232, 230, 222)
GLASS: RGB = (120, 150, 176)
RED_BOTTOM: RGB = (150, 52, 44)
HULL_BLACK: RGB = (44, 46, 52)
HULL_BLUE: RGB = (44, 72, 112)
FUNNEL: RGB = (34, 32, 34)
FUNNEL_BAND: RGB = (186, 52, 40)
SAIL: RGB = (226, 218, 196)
TRUCK_YELLOW: RGB = (222, 170, 40)
TRUCK_BLUE: RGB = (52, 92, 150)
TRUCK_RED: RGB = (176, 56, 44)
BOGIE: RGB = (48, 46, 46)
DECK: RGB = (170, 146, 108)
TAN_SAIL: RGB = (156, 62, 44)
HATCH: RGB = (74, 116, 86)


def placed(m: Model, part: Model, at: tuple[float, float], angle: float = 0.0) -> None:
    """Adds `part`, turned `angle` degrees about its middle and moved to `at`, to `m`."""
    turn = math.radians(angle)
    c, s = math.cos(turn), math.sin(turn)
    du, dv = at
    for face in part.faces:
        corners = [(u * c - v * s + du, u * s + v * c + dv, z) for u, v, z in face.corners]
        nu, nv, nz = face.normal
        m.faces.append(
            replace(face, corners=corners, normal=(nu * c - nv * s, nu * s + nv * c, nz))
        )


def _hull(m: Model, length: float, half: float, high: float, color: RGB, bottom: RGB) -> None:
    """A ship's hull `length` long along u, `half` either side, its deck `high` over the water,
    with a pointed bow at +u, a red bottom along the waterline and a deck of planks."""
    stern, bow = -length / 2, length / 2 - half * 1.6
    m.box((stern, bow), (-half, half), (0.0, high * 0.3), bottom)
    m.box((stern, bow), (-half, half), (high * 0.3, high), color)
    m.nose((bow, length / 2), (0.0, high * 0.6, high), half, color)
    deck = [
        (stern, -half, high),
        (bow, -half, high),
        (length / 2, 0.0, high * 0.99),
        (bow, half, high),
        (stern, half, high),
    ]
    m.faces.append(Face(deck, DECK, (0.0, 0.0, 1.0)))
    with m.fine():
        m.box((stern, bow), (-half - 0.0006, half + 0.0006), (high - 0.004, high - 0.0028), WHITE)


def fishing_boat(color: RGB = WOOD) -> Model:
    """A small fishing boat painted `color` with a mast, a furled sail and a wheelhouse."""
    m = Model()
    _hull(m, 0.09, 0.018, 0.016, color, RED_BOTTOM)
    m.box((-0.003, 0.003), (-0.002, 0.002), (0.016, 0.09), DARK_WOOD)
    m.box((-0.03, 0.0), (-0.003, 0.003), (0.03, 0.036), SAIL)
    m.box((-0.03, -0.012), (-0.01, 0.01), (0.016, 0.034), WHITE)
    with m.fine():
        m.panel((-0.028, -0.014), (0.024, 0.03), 0.0106, GLASS)
    return m


def rowing_boat(color: RGB = WOOD) -> Model:
    """A rowing boat painted `color` outside, its thwarts across it."""
    m = Model()
    _hull(m, 0.07, 0.014, 0.012, color, DARK_WOOD)
    with m.fine():
        for at in (-0.015, 0.008):
            m.box((at - 0.003, at + 0.003), (-0.013, 0.013), (0.012, 0.014), DARK_WOOD)
    return m


def sailing_barge() -> Model:
    """A flat sailing barge with a black hull: green covers over its two holds, a tall mast with
    its red-brown sails set, the big spritsail aft of it on the sprit, a topsail over it and a
    foresail out to the bow, a mizzen at the stern and its leeboard down on one side."""
    m = Model()
    length, half, high = 0.34, 0.042, 0.024
    _hull(m, length, half, high, HULL_BLACK, RED_BOTTOM)
    for middle in (-0.07, 0.06):
        m.box((middle - 0.045, middle + 0.045), (-0.03, 0.03), (high, high + 0.008), HATCH)
        with m.fine():
            for k in range(1, 4):
                at = middle - 0.045 + 0.09 * k / 4
                m.box(
                    (at - 0.001, at + 0.001),
                    (-0.031, 0.031),
                    (high + 0.008, high + 0.0086),
                    shade(HATCH, 0.7),
                )
    m.box((-0.0025, 0.0025), (-0.0025, 0.0025), (high, 0.3), DARK_WOOD)
    _sail(m, [(-0.004, 0.045), (-0.13, 0.045), (-0.15, 0.27), (-0.004, 0.22)], TAN_SAIL, 0.0)
    _sail(m, [(-0.004, 0.225), (-0.14, 0.272), (-0.004, 0.294)], shade(TAN_SAIL, 1.1), 0.001)
    _sail(m, [(0.004, 0.035), (0.165, 0.03), (0.004, 0.2)], shade(TAN_SAIL, 0.92), 0.0)
    m.slope((-0.004, 0.05), (-0.155, 0.275), (-0.0015, 0.0015), DARK_WOOD)
    stern = -length / 2
    m.box((stern + 0.011, stern + 0.016), (-0.002, 0.002), (high, 0.13), DARK_WOOD)
    _sail(
        m,
        [
            (stern + 0.011, 0.045),
            (stern - 0.035, 0.045),
            (stern - 0.04, 0.115),
            (stern + 0.011, 0.12),
        ],
        TAN_SAIL,
        0.0,
    )
    m.box((-0.03, 0.02), (half, half + 0.004), (0.0, high), DARK_WOOD)
    return m


def _sail(m: Model, points: list[tuple[float, float]], color: RGB, v: float) -> None:
    """A sail of cloth set along the hull through `points`, each a (u, z) corner, `v` off its
    middle, seen from either side."""
    _flat(m, [(u, v, z) for u, z in points], color)


def _flat(m: Model, corners: list[tuple[float, float, float]], color: RGB) -> None:
    """A thin flat piece upright along u through `corners`, seen from either side."""
    if m.skipping:
        return
    m.faces.append(Face(corners, color, (0.0, -1.0, 0.0)))
    m.faces.append(Face(corners[::-1], color, (0.0, 1.0, 0.0)))


def steamer() -> Model:
    """A cargo steamer: a black hull with a red bottom, two holds with derricks, a white
    bridge amidships and a black funnel with a red band."""
    m = Model()
    length, half, high = 0.5, 0.045, 0.05
    _hull(m, length, half, high, HULL_BLACK, RED_BOTTOM)
    m.box((-0.06, 0.03), (-0.032, 0.032), (high, high + 0.035), WHITE)
    m.box((-0.03, 0.035), (-0.036, 0.036), (high + 0.035, high + 0.05), WHITE)
    with m.fine():
        for at in (-0.05, -0.02, 0.01):
            m.panel((at, at + 0.018), (high + 0.012, high + 0.022), 0.0326, GLASS)
    upright(m, -0.09, 0.0, 0.014, (high, high + 0.09), FUNNEL)
    upright(m, -0.09, 0.0, 0.0145, (high + 0.06, high + 0.072), FUNNEL_BAND)
    for hold in (-0.18, 0.11):
        m.box((hold - 0.04, hold + 0.04), (-0.026, 0.026), (high, high + 0.01), DARK_WOOD)
        m.box((hold + 0.05, hold + 0.054), (-0.002, 0.002), (high, high + 0.12), WOOD)
        m.box((hold - 0.02, hold + 0.05), (-0.0015, 0.0015), (high + 0.07, high + 0.074), WOOD)
    return m


def container_ship(colors: tuple[RGB, ...]) -> Model:
    """A container ship: a long blue hull with a red bottom, rows of containers in many colours
    stacked over its deck, a tall white bridge at the stern and its funnel."""
    m = Model()
    length, half, high = 0.86, 0.075, 0.05
    _hull(m, length, half, high, HULL_BLUE, RED_BOTTOM)
    box_u, box_v, box_z = 0.072, 0.024, 0.026
    for bay in range(8):
        u = -0.26 + bay * (box_u + 0.006)
        for row in range(5):
            v = -0.054 + row * (box_v + 0.003)
            tiers = 2 + (bay + row) % 3
            for tier in range(tiers):
                color = colors[(bay * 3 + row * 2 + tier) % len(colors)]
                z = high + tier * box_z
                m.box((u, u + box_u), (v, v + box_v), (z, z + box_z), color)
                with m.fine():
                    for k in range(1, 6):
                        at = u + box_u * k / 6
                        for side in (v - 0.0005, v + box_v + 0.0005):
                            m.box(
                                (at - 0.0008, at + 0.0008),
                                (side - 0.0003, side + 0.0003),
                                (z + 0.002, z + box_z - 0.002),
                                shade(color, 0.7),
                            )
    m.box((-0.42, -0.33), (-0.06, 0.06), (high, high + 0.11), WHITE)
    m.box((-0.36, -0.31), (-0.075, 0.075), (high + 0.11, high + 0.13), WHITE)
    with m.fine():
        for at in (-0.355, -0.34, -0.325):
            m.panel((at, at + 0.01), (high + 0.115, high + 0.125), 0.0756, GLASS)
        for z in (high + 0.03, high + 0.06, high + 0.09):
            m.box((-0.3302, -0.3298), (-0.05, 0.05), (z, z + 0.008), GLASS)
    m.box((-0.45, -0.42), (-0.016, 0.016), (high, high + 0.15), FUNNEL_BAND)
    return m


def _ring(v: float, z: float, half: float, high: float) -> list[tuple[float, float]]:
    """An oval cross section round (v, z), `half` wide either side and `high` up and down,
    closed by its first point again."""
    points = [
        (v + half * math.sin(2 * math.pi * k / 10), z + high * math.cos(2 * math.pi * k / 10))
        for k in range(10)
    ]
    return [*points, points[0]]


def _horse(m: Model) -> None:
    """A bay horse standing along u, its head at +u: a rounded barrel of a body on four slim
    legs with dark hooves, an arched neck with its mane up to a long head, its ears pricked, and a
    dark tail."""
    m.loft(
        [
            (0.021, _ring(0.0, 0.022, 0.001, 0.001)),
            (0.023, _ring(0.0, 0.0215, 0.0042, 0.0052)),
            (0.028, _ring(0.0, 0.021, 0.0054, 0.0066)),
            (0.038, _ring(0.0, 0.0208, 0.0054, 0.0068)),
            (0.045, _ring(0.0, 0.0215, 0.0046, 0.0062)),
            (0.0475, _ring(0.0, 0.023, 0.0024, 0.0034)),
        ],
        HORSE,
    )
    m.loft(
        [
            (0.042, _ring(0.0, 0.024, 0.003, 0.0048)),
            (0.047, _ring(0.0, 0.0295, 0.0024, 0.0038)),
            (0.0515, _ring(0.0, 0.035, 0.002, 0.0028)),
        ],
        HORSE,
    )
    m.loft(
        [
            (0.0505, _ring(0.0, 0.036, 0.0022, 0.0028)),
            (0.055, _ring(0.0, 0.034, 0.0019, 0.0024)),
            (0.0605, _ring(0.0, 0.03, 0.0014, 0.0017)),
        ],
        HORSE,
    )
    for side in (-0.0011, 0.0011):
        m.box((0.0505, 0.0515), (side - 0.0005, side + 0.0005), (0.0375, 0.04), HORSE)
    m.slope((0.0415, 0.0292), (0.0515, 0.0382), (-0.0007, 0.0007), MANE)
    m.slope((0.0215, 0.024), (0.0185, 0.011), (-0.0009, 0.0009), MANE)
    for lu in (0.0255, 0.0425):
        for side in (-0.0026, 0.0026):
            m.box(
                (lu - 0.0011, lu + 0.0011), (side - 0.0011, side + 0.0011), (0.0015, 0.0165), HORSE
            )
            m.box((lu - 0.0013, lu + 0.0013), (side - 0.0013, side + 0.0013), (0.0, 0.0015), MANE)
    m.box((0.0425, 0.045), (-0.0034, 0.0034), (0.022, 0.0285), DARK_WOOD)


def _cart_wheel(m: Model, u: float, v: float, radius: float) -> None:
    """A spoked wooden cart wheel standing at `u`, `v` across, `radius` round: a rim of felloes,
    eight spokes and a hub."""
    rim, sides = radius * 0.16, 16
    for k in range(sides):
        a0, a1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
        outer = [(u + radius * math.cos(a), radius + radius * math.sin(a)) for a in (a0, a1)]
        inner = [
            (u + (radius - rim) * math.cos(a), radius + (radius - rim) * math.sin(a))
            for a in (a1, a0)
        ]
        _flat(m, [(pu, v, pz) for pu, pz in outer + inner], DARK_WOOD)
    for k in range(8):
        a = 2 * math.pi * k / 8
        du, dz = math.cos(a), math.sin(a)
        nu, nz = -dz * radius * 0.05, du * radius * 0.05
        hub, out = radius * 0.2, radius - rim
        _flat(
            m,
            [
                (u + du * hub + nu, v, radius + dz * hub + nz),
                (u + du * out + nu, v, radius + dz * out + nz),
                (u + du * out - nu, v, radius + dz * out - nz),
                (u + du * hub - nu, v, radius + dz * hub - nz),
            ],
            WOOD,
        )
    m.box(
        (u - radius * 0.22, u + radius * 0.22),
        (v - 0.002, v + 0.002),
        (radius * 0.78, radius * 1.22),
        DARK_WOOD,
    )


def horse_cart() -> Model:
    """A horse drawing a two-wheeled cart loaded with sacks."""
    m = Model()
    m.box((-0.04, 0.0), (-0.013, 0.013), (0.012, 0.016), WOOD)
    for side in (-0.014, 0.014):
        m.box((-0.04, 0.0), (side - 0.0015, side + 0.0015), (0.016, 0.026), DARK_WOOD)
    for side in (-0.016, 0.016):
        _cart_wheel(m, -0.022, side, 0.011)
    m.box((-0.026, -0.018), (-0.016, 0.016), (0.0102, 0.0118), DARK_WOOD)
    m.box((-0.036, -0.006), (-0.01, 0.01), (0.016, 0.026), (204, 186, 146))
    for side in (-0.0068, 0.0068):
        m.slope((0.0, 0.016), (0.044, 0.022), (side - 0.0008, side + 0.0008), DARK_WOOD)
    _horse(m)
    return m


def dump_truck(color: RGB, load: RGB) -> Model:
    """A dump truck: a cab with a windscreen and a tipping body heaped with its load."""
    m = Model()
    m.box((-0.05, 0.04), (-0.014, 0.014), (0.01, 0.014), BOGIE)
    m.wheels([-0.035, -0.018, 0.028], 0.009, TYRE, 0.014)
    m.box((0.018, 0.045), (-0.016, 0.016), (0.014, 0.042), color)
    m.slope((0.045, 0.024), (0.04, 0.04), (-0.014, 0.014), GLASS)
    m.box((-0.05, 0.014), (-0.018, 0.018), (0.016, 0.034), shade(color, 0.85))
    cone(m, -0.018, 0.0, 0.018, 0.016, load, 0.034)
    return m


def container_lorry(color: RGB) -> Model:
    """A lorry: a tractor with a cab and a trailer carrying a container."""
    m = Model()
    m.box((-0.07, 0.05), (-0.013, 0.013), (0.01, 0.014), BOGIE)
    m.wheels([-0.06, -0.045, 0.012, 0.036], 0.009, TYRE, 0.013)
    m.box((0.026, 0.056), (-0.016, 0.016), (0.014, 0.046), TRUCK_RED)
    m.slope((0.056, 0.026), (0.052, 0.044), (-0.014, 0.014), GLASS)
    m.box((-0.074, 0.02), (-0.015, 0.015), (0.014, 0.042), color)
    with m.fine():
        for k in range(1, 9):
            at = -0.074 + 0.094 * k / 9
            for side in (-0.0156, 0.0156):
                m.box(
                    (at - 0.0008, at + 0.0008),
                    (side - 0.0004, side + 0.0004),
                    (0.016, 0.04),
                    shade(color, 0.7),
                )
    return m


def wagon(build: Callable[[Model], None]) -> Model:
    """A railway wagon on two bogies, whatever `build` puts on its floor."""
    m = Model()
    m.box((-0.05, 0.05), (-0.016, 0.016), (0.008, 0.013), BOGIE)
    m.wheels([-0.036, -0.024, 0.024, 0.036], 0.0055, STEEL, 0.012)
    build(m)
    return m


def open_wagon(load: RGB) -> Model:
    def build(m: Model) -> None:
        m.box((-0.05, 0.05), (-0.017, 0.017), (0.013, 0.03), (96, 74, 58))
        cone(m, 0.0, 0.0, 0.018, 0.012, load, 0.026)

    return wagon(build)


def flat_wagon(color: RGB) -> Model:
    def build(m: Model) -> None:
        m.box((-0.046, 0.046), (-0.015, 0.015), (0.013, 0.04), color)
        with m.fine():
            for k in range(1, 9):
                at = -0.046 + 0.092 * k / 9
                for side in (-0.0156, 0.0156):
                    m.box(
                        (at - 0.0008, at + 0.0008),
                        (side - 0.0004, side + 0.0004),
                        (0.015, 0.038),
                        shade(color, 0.7),
                    )

    return wagon(build)


def box_wagon() -> Model:
    def build(m: Model) -> None:
        m.box((-0.05, 0.05), (-0.017, 0.017), (0.013, 0.044), (132, 70, 52))
        m.roof((-0.052, 0.052), (-0.018, 0.018), (0.044, 0.05), (90, 90, 96), (132, 70, 52))
        with m.fine():
            m.panel((-0.012, 0.012), (0.015, 0.04), 0.0176, (100, 54, 40))

    return wagon(build)
