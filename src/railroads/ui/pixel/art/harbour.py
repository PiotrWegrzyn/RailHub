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


def horse_cart() -> Model:
    """A horse drawing a two-wheeled cart loaded with sacks."""
    m = Model()
    m.box((-0.045, 0.0), (-0.016, 0.016), (0.012, 0.016), WOOD)
    for side in (-0.017, 0.017):
        m.box((-0.045, 0.0), (side - 0.0015, side + 0.0015), (0.016, 0.026), DARK_WOOD)
    m.wheels([-0.022], 0.011, DARK_WOOD, 0.02)
    m.box((-0.04, -0.006), (-0.012, 0.012), (0.016, 0.028), (204, 186, 146))
    for side in (-0.01, 0.01):
        m.box((0.0, 0.05), (side - 0.0012, side + 0.0012), (0.014, 0.016), DARK_WOOD)
    m.box((0.02, 0.05), (-0.006, 0.006), (0.016, 0.03), HORSE)
    for lu in (0.023, 0.047):
        for side in (-0.004, 0.004):
            m.box((lu - 0.002, lu + 0.002), (side - 0.002, side + 0.002), (0.0, 0.017), HORSE)
    m.box((0.046, 0.056), (-0.003, 0.003), (0.026, 0.04), HORSE)
    m.box((0.052, 0.066), (-0.0035, 0.0035), (0.034, 0.042), HORSE)
    m.box((0.044, 0.052), (-0.0015, 0.0015), (0.03, 0.043), MANE)
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
