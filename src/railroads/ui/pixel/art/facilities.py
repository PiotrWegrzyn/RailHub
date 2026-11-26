"""The facilities that stand on the map beside a station (railroads.sim.facilities), built as
small lit models (railroads.ui.pixel.art.built) like the stations: an engine shed as a roundhouse
of brick stalls fanned round a turntable, with an engine on its bridge and a brick workshop wing,
filling its whole cell, its lead track running out to the station beside it, drawn turned
towards each way a station can lie (SHED_HEADINGS); and cattle pens, a small fenced pen and shed.

Models are in cells: u and v across the ground, z up."""

import math
from collections.abc import Callable
from dataclasses import replace

from railroads.sim.sheds import (
    ROUNDHOUSE_SIZE,
    STALL_FROM,
    STALL_HEADINGS,
    TURNTABLE,
    TURNTABLE_RADIUS,
)
from railroads.ui.pixel.art.built import (
    chimney,
    footing,
    gable_roof,
    modelled,
    paddock,
    spot,
    upright,
    walls,
    window,
)
from railroads.ui.pixel.art.painter import BRICK, RGB, Sprite, shade
from railroads.ui.pixel.art.stations import PANE, ROOF_SLATE, STONE, TIMBER, TIMBER_TRIM
from railroads.ui.pixel.models import Face, Model

ASH: RGB = (112, 106, 94)
PIT: RGB = (86, 82, 76)
SLEEPERS: RGB = (70, 58, 48)
RAIL: RGB = (150, 146, 140)
DECK: RGB = (120, 110, 96)
OPENING: RGB = (34, 30, 30)
PIER: RGB = (132, 70, 50)
SKYLIGHT: RGB = (176, 204, 218)
WORKSHOP: RGB = (176, 82, 58)
ENGINE: RGB = (36, 36, 40)
WHEEL: RGB = (176, 40, 36)
CAB: RGB = (60, 112, 66)
STRAW: RGB = (204, 178, 98)

# The roundhouse, its lead track running out along u, built to the measures trains stand in it by
# (railroads.sim.sheds): how big the yard round its turntable is, and where its lead track ends:
# in the middle of the cell beside it, a cell away.
YARD_RADIUS = 0.2
LEAD_END = 1.0 / ROUNDHOUSE_SIZE
# The ways an engine shed is drawn turned, its lead track running that way on the map, in
# degrees from the map's x axis: every way a cell beside it lies on hexes and on squares.
SHED_HEADINGS = tuple(range(0, 360, 30))
# How much smaller cattle pens are drawn than built.
PENS_SIZE = 0.6


def _turned_into(m: Model, part: Model, angle: float, du: float, dv: float) -> None:
    """Adds `part`, turned `angle` radians about its origin and moved to (du, dv), to `m`."""
    c, s = math.cos(angle), math.sin(angle)
    for face in part.faces:
        corners = [(u * c - v * s + du, u * s + v * c + dv, z) for u, v, z in face.corners]
        nu, nv, nz = face.normal
        m.faces.append(
            replace(face, corners=corners, normal=(nu * c - nv * s, nu * s + nv * c, nz))
        )
    for (u0, u1), (v0, v1) in part.footprints:
        points = [(u * c - v * s + du, u * s + v * c + dv) for u in (u0, u1) for v in (v0, v1)]
        us, vs = zip(*points, strict=True)
        m.footprints.append(((min(us), max(us)), (min(vs), max(vs))))


def _disc(m: Model, middle: tuple[float, float], radius: float, color: RGB, z: float) -> None:
    """A round patch lying on the ground."""
    sides = 28
    u, v = middle
    corners = [
        (
            u + radius * math.cos(2 * math.pi * k / sides),
            v + radius * math.sin(2 * math.pi * k / sides),
            z,
        )
        for k in range(sides)
    ]
    m.faces.append(Face(corners, color, (0.0, 0.0, 1.0)))


def _strip(
    m: Model, a: tuple[float, float], b: tuple[float, float], half: float, color: RGB, z: float
) -> None:
    """A flat strip `half` either side of the line from `a` to `b`, `z` above the ground."""
    (ua, va), (ub, vb) = a, b
    length = math.hypot(ub - ua, vb - va) or 1.0
    nu, nv = -(vb - va) / length * half, (ub - ua) / length * half
    corners = [
        (ua + nu, va + nv, z),
        (ub + nu, vb + nv, z),
        (ub - nu, vb - nv, z),
        (ua - nu, va - nv, z),
    ]
    m.faces.append(Face(corners, color, (0.0, 0.0, 1.0)))


def _round(angle: float, radius: float) -> tuple[float, float]:
    """The point `radius` out from the turntable's middle at `angle`."""
    return TURNTABLE[0] + radius * math.cos(angle), TURNTABLE[1] + radius * math.sin(angle)


def _stall(high: float) -> Model:
    """One brick stall, its open door at u = 0 facing the turntable, under a slate roof with
    skylights along its ridge."""
    m = Model()
    u, v = (0.0, 0.13), (-0.034, 0.034)
    walls(m, u, v, high, BRICK, STONE)
    gable_roof(m, u, v, (high, high + 0.035), ROOF_SLATE, shade(BRICK, 0.9), 0.008)
    m.box((-0.002, 0.0), (-0.026, 0.026), (0.008, high - 0.01), OPENING)
    for side in ((-0.036, -0.026), (0.026, 0.036)):
        m.box((-0.004, 0.0), side, (0.0, high), PIER)
    with m.fine():
        for k in range(3):
            low = 0.04 + k * 0.03
            m.box((low, low + 0.015), (-0.005, 0.005), (high + 0.02, high + 0.034), SKYLIGHT)
    return m


def _engine() -> Model:
    """A small steam engine and its tender along u: a black boiler on red wheels, a green cab."""
    m = Model()
    m.box((-0.07, 0.03), (-0.016, 0.016), (0.008, 0.022), WHEEL)
    m.cylinder_along((-0.07, 0.0), 0.016, 0.036, ENGINE, ENGINE)
    upright(m, -0.058, 0.0, 0.005, (0.05, 0.066), ENGINE)
    m.box((0.0, 0.026), (-0.018, 0.018), (0.02, 0.064), CAB)
    m.box((-0.002, 0.028), (-0.02, 0.02), (0.064, 0.07), ENGINE)
    m.box((0.032, 0.07), (-0.016, 0.016), (0.012, 0.048), ENGINE)
    return m


def _track(m: Model, a: tuple[float, float], b: tuple[float, float]) -> None:
    """A piece of track on the ground from `a` to `b`: sleepers and, up close, two rails."""
    _strip(m, a, b, 0.014, SLEEPERS, 0.0012)
    with m.fine():
        (ua, va), (ub, vb) = a, b
        length = math.hypot(ub - ua, vb - va) or 1.0
        nu, nv = -(vb - va) / length, (ub - ua) / length
        for side in (-0.007, 0.007):
            shift = (nu * side, nv * side)
            _strip(
                m,
                (ua + shift[0], va + shift[1]),
                (ub + shift[0], vb + shift[1]),
                0.0018,
                RAIL,
                0.0018,
            )


def _workshop() -> Model:
    """The roundhouse's brick workshop wing along u, with windows down both sides and a
    chimney."""
    m = Model()
    u, v, high = (0.1, 0.32), (-0.05, 0.05), 0.1
    footing(m, u, v)
    walls(m, u, v, high, WORKSHOP, STONE)
    for k in range(4):
        at = 0.125 + k * 0.05
        window(m, (at - 0.012, at + 0.012), (0.03, 0.075), v[1], PANE)
    gable_roof(m, u, v, (high, high + 0.05), ROOF_SLATE, WORKSHOP, 0.01)
    chimney(m, 0.28, 0.0, (0.12, 0.2), BRICK)
    return m


def roundhouse() -> Model:
    """An engine shed filling its cell: brick stalls fanned round the back of a turntable in a
    yard of ash, their tracks running in to it, an engine on the turntable's bridge, the lead
    track running out along u into the station's cell beside it, and a brick workshop wing with a chimney
    beside it."""
    m = Model()
    _disc(m, TURNTABLE, YARD_RADIUS, ASH, 0.0004)
    _disc(m, TURNTABLE, TURNTABLE_RADIUS, PIT, 0.0008)
    _disc(m, TURNTABLE, TURNTABLE_RADIUS * 0.85, shade(PIT, 0.8), 0.001)
    for angle in STALL_HEADINGS:
        _track(m, _round(angle, TURNTABLE_RADIUS), _round(angle, STALL_FROM))
    _track(m, _round(0.0, TURNTABLE_RADIUS), (LEAD_END, 0.0))
    bridge = math.radians(20)
    ends = (_round(bridge + math.pi, TURNTABLE_RADIUS), _round(bridge, TURNTABLE_RADIUS))
    _strip(m, *ends, 0.018, DECK, 0.006)
    _turned_into(m, _engine(), bridge, *TURNTABLE)
    for angle in STALL_HEADINGS:
        _turned_into(m, _stall(0.075), angle, *_round(angle, STALL_FROM))
    _turned_into(m, _workshop(), 0.0, 0.0, 0.18)
    return m


def cattle_pens() -> Model:
    """A fenced pen of trodden straw by a small timber shed with a loading ramp."""
    m = Model()
    for k, color in enumerate((STRAW, shade(STRAW, 0.85))):
        half = 5.0 - k * 1.5
        corners = [(-half, 0), (0, -half / 2), (half, 0), (0, half / 2)]
        ground = [(*spot(px, py), 0.0006 + k * 0.0003) for px, py in corners]
        m.faces.append(Face(ground, color, (0.0, 0.0, 1.0)))
    paddock(m, 0, 0, 5.0)
    u, v, high = (0.1, 0.22), (-0.2, -0.1), 0.09
    walls(m, u, v, high, TIMBER, STONE)
    gable_roof(m, u, v, (high, high + 0.05), shade(TIMBER_TRIM, 1.2), TIMBER, 0.01)
    m.box((0.22, 0.28), (-0.17, -0.13), (0.0, 0.03), TIMBER_TRIM)
    return m


def shed_sprite(heading: int) -> str:
    """The sprite of an engine shed whose lead track runs `heading` degrees on the map."""
    return f"engine_shed_{heading}"


PENS_SPRITE = "cattle_pens_annex"
FACILITIES: dict[str, Callable[[], Sprite]] = {
    shed_sprite(heading): modelled(
        roundhouse, across=ROUNDHOUSE_SIZE, shadow=0.7, heading=math.radians(heading)
    )
    for heading in SHED_HEADINGS
}
FACILITIES[PENS_SPRITE] = modelled(cattle_pens, across=PENS_SIZE, shadow=0.6)
