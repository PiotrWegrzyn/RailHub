"""The facilities that stand on the map beside a station (railroads.sim.facilities), built as
small lit models (railroads.ui.pixel.art.built) like the stations: an engine shed as a roundhouse
after the model railway ones, bigger than its cell, one curved hall of brick bays fanned round a
turntable, 3, 6 or 9 of them by its level (railroads.sim.sheds.BUILT), each with white-framed
doors swung open, white pilasters and arched windows, a grey roof falling towards the doors with a
skylight and a smoke pipe; the turntable with its bridge and a green operator's hut, rails running
from it into every bay and out along the lead, and, once the station puts one up, a tall sand
tower under a red pointed roof by the lead; drawn turned towards each way its lead can run (SHED_HEADINGS); and cattle pens, no
longer put on the map, a small fenced pen and shed.

Models are in cells: u and v across the ground, z up."""

import math
from collections.abc import Callable
from dataclasses import replace
from functools import partial

from railroads.sim.sheds import (
    BUILT,
    ROUNDHOUSE_SIZE,
    STALL_FROM,
    STALL_HEADINGS,
    STALL_TO,
    TURNTABLE,
    TURNTABLE_RADIUS,
)
from railroads.ui.pixel.art.built import (
    cone,
    gable_roof,
    modelled,
    paddock,
    spot,
    upright,
    walls,
    window,
)
from railroads.ui.pixel.art.painter import BRICK, RGB, Sprite, shade
from railroads.ui.pixel.art.stations import STONE, TIMBER, TIMBER_TRIM
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
ARCH: RGB = (190, 170, 140)
SOOT: RGB = (52, 48, 46)
TANK: RGB = (70, 78, 84)
COAL: RGB = (30, 30, 32)
ROOF_GREY: RGB = (128, 140, 146)
RIB: RGB = (104, 116, 122)
FRAME_WHITE: RGB = (232, 232, 226)
GLASS: RGB = (150, 178, 196)
DOOR_DARK: RGB = (38, 38, 40)
HAZARD: RGB = (226, 192, 40)
PILASTER: RGB = (214, 206, 196)
BALLAST: RGB = (78, 72, 66)
OCHRE: RGB = (214, 184, 120)
RED_ROOF: RGB = (172, 64, 46)
LAMP: RGB = (250, 230, 160)

# The roundhouse, its lead track running out along u, built to the measures trains stand in it by
# (railroads.sim.sheds): where its lead track ends, at the edge of its cell, where track laid to it
# joins.
LEAD_END = 0.5 / ROUNDHOUSE_SIZE
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


# How high a bay's walls stand at its doors and at its back, how high its doors are, and how
# wide each door leaf is as a share of the doorway.
BAY_LOW = 0.07
BAY_HIGH = 0.105
DOOR_TOP = 0.056
# The slope of a bay's sides: each is a slice of the fan round the turntable, as wide as the
# angle between stalls, so the bays stand side by side as one curved hall.
SPREAD = math.tan((STALL_HEADINGS[1] - STALL_HEADINGS[0]) / 2)


def _normal(corners: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = corners[:3]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / length, ny / length, nz / length


def _quad(
    m: Model,
    corners: list[tuple[float, float, float]],
    color: RGB,
    out: tuple[float, float, float],
) -> None:
    """A flat face through `corners`, lit as facing `out`'s way."""
    if m.skipping:
        return
    nx, ny, nz = _normal(corners)
    if nx * out[0] + ny * out[1] + nz * out[2] < 0:
        nx, ny, nz = -nx, -ny, -nz
    m.faces.append(Face(corners, color, (nx, ny, nz)))


def _side(m: Model, depth: float, sign: float) -> None:
    """One side wall of a bay, on its `sign` side, from its doors back, rising with the roof,
    with white pilasters and an arched window, which show on the hall's end walls."""
    w0, w1 = STALL_FROM * SPREAD, STALL_TO * SPREAD
    out = (0.0, sign, 0.0)
    corners = [
        (0.0, sign * w0, 0.0),
        (depth, sign * w1, 0.0),
        (depth, sign * w1, BAY_HIGH),
        (0.0, sign * w0, BAY_LOW),
    ]
    _quad(m, corners, BRICK, out)
    lift = 0.0008

    def on_wall(u: float, z: float) -> tuple[float, float, float]:
        w = (STALL_FROM + u) * SPREAD
        return (u, sign * (w + lift), z)

    def eave(u: float) -> float:
        return BAY_LOW + (BAY_HIGH - BAY_LOW) * u / depth

    with m.fine():
        _quad(
            m,
            [on_wall(0, 0), on_wall(depth, 0), on_wall(depth, 0.01), on_wall(0, 0.01)],
            STONE,
            out,
        )
        for at in (0.01, depth * 0.5, depth - 0.008):
            top = eave(at) - 0.002
            _quad(
                m,
                [
                    on_wall(at - 0.004, 0.01),
                    on_wall(at + 0.004, 0.01),
                    on_wall(at + 0.004, top),
                    on_wall(at - 0.004, top),
                ],
                PILASTER,
                out,
            )
        for mid in (depth * 0.27, depth * 0.73):
            low, high = 0.022, eave(mid) - 0.016
            a, b = mid - 0.02, mid + 0.02
            _quad(
                m,
                [on_wall(a, low), on_wall(b, low), on_wall(b, high), on_wall(a, high)],
                FRAME_WHITE,
                out,
            )
            _quad(
                m,
                [
                    on_wall(a + 0.003, low + 0.003),
                    on_wall(b - 0.003, low + 0.003),
                    on_wall(b - 0.003, high - 0.006),
                    on_wall(a + 0.003, high - 0.006),
                ],
                GLASS,
                out,
            )
            arch = high + 0.004
            _quad(
                m,
                [on_wall(a, high), on_wall(b, high), on_wall(mid, arch)] + [on_wall(mid, arch)],
                FRAME_WHITE,
                out,
            )
    with m.finest():
        for mid in (depth * 0.27, depth * 0.73):
            low, high = 0.025, eave(mid) - 0.022
            for k in (1, 2):
                z = low + (high - low) * k / 3
                _quad(
                    m,
                    [
                        on_wall(mid - 0.017, z),
                        on_wall(mid + 0.017, z),
                        on_wall(mid + 0.017, z + 0.0012),
                        on_wall(mid - 0.017, z + 0.0012),
                    ],
                    FRAME_WHITE,
                    out,
                )
            _quad(
                m,
                [
                    on_wall(mid - 0.0006, low),
                    on_wall(mid + 0.0006, low),
                    on_wall(mid + 0.0006, high),
                    on_wall(mid - 0.0006, high),
                ],
                FRAME_WHITE,
                out,
            )


def _back(m: Model, depth: float, w1: float) -> None:
    """A bay's back wall, the hall's outer wall: brick on a stone plinth between white
    pilasters, with a tall arched window in white frames and a stone band under the eaves."""
    back = (1.0, 0.0, 0.0)
    _quad(
        m,
        [(depth, -w1, 0.0), (depth, w1, 0.0), (depth, w1, BAY_HIGH), (depth, -w1, BAY_HIGH)],
        BRICK,
        back,
    )
    face = depth + 0.0008
    with m.fine():
        _quad(
            m,
            [(face, -w1, 0.0), (face, w1, 0.0), (face, w1, 0.011), (face, -w1, 0.011)],
            STONE,
            back,
        )
        _quad(
            m,
            [
                (face, -w1, BAY_HIGH - 0.008),
                (face, w1, BAY_HIGH - 0.008),
                (face, w1, BAY_HIGH),
                (face, -w1, BAY_HIGH),
            ],
            STONE,
            back,
        )
        for edge in (-w1, w1 - 0.007):
            _quad(
                m,
                [
                    (face, edge, 0.011),
                    (face, edge + 0.007, 0.011),
                    (face, edge + 0.007, BAY_HIGH - 0.008),
                    (face, edge, BAY_HIGH - 0.008),
                ],
                PILASTER,
                back,
            )
        low, high, half = 0.024, BAY_HIGH - 0.026, w1 * 0.42
        _quad(
            m,
            [(face, -half, low), (face, half, low), (face, half, high), (face, -half, high)],
            FRAME_WHITE,
            back,
        )
        _quad(
            m,
            [
                (face, -half, high),
                (face, half, high),
                (face, 0.0, high + half * 0.6),
                (face, 0.0, high + half * 0.6),
            ],
            FRAME_WHITE,
            back,
        )
        glass = face + 0.0004
        inner = half - 0.003
        _quad(
            m,
            [
                (glass, -inner, low + 0.003),
                (glass, inner, low + 0.003),
                (glass, inner, high),
                (glass, -inner, high),
            ],
            GLASS,
            back,
        )
        _quad(
            m,
            [
                (glass, -inner, high),
                (glass, inner, high),
                (glass, 0.0, high + inner * 0.5),
                (glass, 0.0, high + inner * 0.5),
            ],
            GLASS,
            back,
        )
    with m.finest():
        bar = face + 0.0008
        for k in (1, 2, 3):
            z = low + (high - low) * k / 4
            _quad(
                m,
                [
                    (bar, -inner, z),
                    (bar, inner, z),
                    (bar, inner, z + 0.0012),
                    (bar, -inner, z + 0.0012),
                ],
                FRAME_WHITE,
                back,
            )
        for at in (-inner / 3, inner / 3):
            _quad(
                m,
                [
                    (bar, at - 0.0006, low),
                    (bar, at + 0.0006, low),
                    (bar, at + 0.0006, high),
                    (bar, at - 0.0006, high),
                ],
                FRAME_WHITE,
                back,
            )


def _leaf(m: Model, hinge: float, sign: float) -> None:
    """One door leaf hinged at the side `sign` of a doorway `hinge` out from its middle, swung
    open out towards the turntable: glazed in white frames above, dark below, striped where
    engines pass."""
    width = hinge
    angle = math.radians(105)
    end = (-width * math.sin(angle), sign * (hinge - width * math.cos(angle)))
    start = (0.0, sign * hinge)
    out = (-math.cos(angle), sign * math.sin(angle) * 0.0 - sign, 0.0)

    def leaf(z0: float, z1: float, color: RGB, inset: float = 0.0) -> None:
        sx, sy = start[0] + (end[0] - start[0]) * inset, start[1] + (end[1] - start[1]) * inset
        ex, ey = end[0] - (end[0] - start[0]) * inset, end[1] - (end[1] - start[1]) * inset
        for flip in (1.0, -1.0):
            _quad(
                m,
                [(sx, sy, z0), (ex, ey, z0), (ex, ey, z1), (sx, sy, z1)],
                color,
                (out[0] * flip, out[1] * flip, 0.0),
            )

    leaf(0.004, DOOR_TOP * 0.55, DOOR_DARK)
    leaf(DOOR_TOP * 0.55, DOOR_TOP, FRAME_WHITE)
    with m.fine():
        leaf(DOOR_TOP * 0.6, DOOR_TOP - 0.004, GLASS, 0.12)
    with m.finest():
        leaf(0.012, 0.018, HAZARD, 0.35)
        leaf(0.024, 0.03, HAZARD, 0.35)


def _bay() -> Model:
    """One bay of the roundhouse from its doors at u = 0, facing the turntable, back to STALL_TO:
    brick walls and white pilasters, a dark doorway framed in white with its two doors swung
    open, a grey ribbed roof falling towards the doors with a white-framed skylight, and a smoke
    pipe over where an engine's chimney stands."""
    m = Model()
    depth = STALL_TO - STALL_FROM
    w0, w1 = STALL_FROM * SPREAD, STALL_TO * SPREAD
    m.footprints.append(((0.0, depth), (-w1, w1)))
    for sign in (-1.0, 1.0):
        _side(m, depth, sign)
    _back(m, depth, w1)
    door = w0 - 0.006
    front = (-1.0, 0.0, 0.0)
    _quad(
        m,
        [(0, -w0, DOOR_TOP), (0, w0, DOOR_TOP), (0, w0, BAY_LOW), (0, -w0, BAY_LOW)],
        BRICK,
        front,
    )
    for sign in (-1.0, 1.0):
        _quad(
            m,
            [
                (0, sign * door, 0),
                (0, sign * w0, 0),
                (0, sign * w0, DOOR_TOP),
                (0, sign * door, DOOR_TOP),
            ],
            BRICK,
            front,
        )
    _quad(
        m,
        [
            (-0.0006, -door, 0.0),
            (-0.0006, door, 0.0),
            (-0.0006, door, DOOR_TOP),
            (-0.0006, -door, DOOR_TOP),
        ],
        OPENING,
        front,
    )
    with m.fine():
        _quad(
            m,
            [
                (-0.0012, -door - 0.003, DOOR_TOP),
                (-0.0012, door + 0.003, DOOR_TOP),
                (-0.0012, door + 0.003, DOOR_TOP + 0.005),
                (-0.0012, -door - 0.003, DOOR_TOP + 0.005),
            ],
            FRAME_WHITE,
            front,
        )
        for sign in (-1.0, 1.0):
            _quad(
                m,
                [
                    (-0.0012, sign * door, 0.0),
                    (-0.0012, sign * (door + 0.003), 0.0),
                    (-0.0012, sign * (door + 0.003), DOOR_TOP),
                    (-0.0012, sign * door, DOOR_TOP),
                ],
                FRAME_WHITE,
                front,
            )
    for sign in (-1.0, 1.0):
        _leaf(m, door, sign)
    eave, ridge, over = BAY_LOW + 0.004, BAY_HIGH + 0.004, 0.01
    roof = [
        (-over, -(w0 + 0.002), eave - 0.003),
        (-over, w0 + 0.002, eave - 0.003),
        (depth + 0.004, w1 + 0.002, ridge),
        (depth + 0.004, -(w1 + 0.002), ridge),
    ]
    _quad(m, roof, ROOF_GREY, (-0.3, 0.0, 1.0))

    def on_roof(t: float, s: float, rise: float = 0.0012) -> tuple[float, float, float]:
        u = -over + (depth + 0.004 + over) * t
        w = (w0 + 0.002) + (w1 - w0) * t
        z = eave - 0.003 + (ridge - eave + 0.003) * t + rise
        return (u, s * w, z)

    with m.fine():
        for k in range(1, 6):
            s = -1 + 2 * k / 6
            _quad(
                m,
                [
                    on_roof(0.02, s - 0.012),
                    on_roof(0.98, s - 0.012),
                    on_roof(0.98, s + 0.012),
                    on_roof(0.02, s + 0.012),
                ],
                RIB,
                (-0.3, 0.0, 1.0),
            )
        _quad(
            m,
            [
                on_roof(0.4, -0.5, 0.004),
                on_roof(0.72, -0.55, 0.004),
                on_roof(0.72, 0.55, 0.004),
                on_roof(0.4, 0.5, 0.004),
            ],
            FRAME_WHITE,
            (-0.3, 0.0, 1.0),
        )
        _quad(
            m,
            [
                on_roof(0.43, -0.4, 0.0052),
                on_roof(0.69, -0.45, 0.0052),
                on_roof(0.69, 0.45, 0.0052),
                on_roof(0.43, 0.4, 0.0052),
            ],
            GLASS,
            (-0.3, 0.0, 1.0),
        )
    with m.finest():
        for t in (0.51, 0.6):
            _quad(
                m,
                [
                    on_roof(t, -0.45, 0.0062),
                    on_roof(t + 0.012, -0.45, 0.0062),
                    on_roof(t + 0.012, 0.45, 0.0062),
                    on_roof(t, 0.45, 0.0062),
                ],
                FRAME_WHITE,
                (-0.3, 0.0, 1.0),
            )
    pipe_u = depth * 0.8
    top = BAY_LOW + (BAY_HIGH - BAY_LOW) * pipe_u / depth
    upright(m, pipe_u, 0.0, 0.0055, (top, top + 0.05), SOOT)
    m.box((pipe_u - 0.009, pipe_u + 0.009), (-0.009, 0.009), (top + 0.05, top + 0.055), SOOT)
    return m


def _track(m: Model, a: tuple[float, float], b: tuple[float, float]) -> None:
    """A piece of track on the ground from `a` to `b`: sleepers and, up close, two rails."""
    _strip(m, a, b, 0.016, BALLAST, 0.0008)
    _strip(m, a, b, 0.012, SLEEPERS, 0.0012)
    with m.fine():
        (ua, va), (ub, vb) = a, b
        length = math.hypot(ub - ua, vb - va) or 1.0
        nu, nv = -(vb - va) / length, (ub - ua) / length
        for side in (-0.006, 0.006):
            shift = (nu * side, nv * side)
            _strip(
                m,
                (ua + shift[0], va + shift[1]),
                (ub + shift[0], vb + shift[1]),
                0.0016,
                RAIL,
                0.0018,
            )


def _turntable(m: Model) -> None:
    """The turntable: a dark pit with a rail round its edge, its bridge across it with railings
    and an engine on it, and a green operator's hut at one end."""
    _disc(m, TURNTABLE, TURNTABLE_RADIUS + 0.008, STONE, 0.0006)
    _disc(m, TURNTABLE, TURNTABLE_RADIUS, PIT, 0.0008)
    _disc(m, TURNTABLE, TURNTABLE_RADIUS * 0.9, shade(PIT, 0.75), 0.001)
    with m.fine():
        _disc(m, TURNTABLE, TURNTABLE_RADIUS * 0.84, RAIL, 0.0012)
        _disc(m, TURNTABLE, TURNTABLE_RADIUS * 0.81, shade(PIT, 0.75), 0.0014)
    bridge = math.radians(20)
    ends = (_round(bridge + math.pi, TURNTABLE_RADIUS), _round(bridge, TURNTABLE_RADIUS))
    _strip(m, *ends, 0.02, DECK, 0.008)
    begin = len(m.faces)
    m.box((-TURNTABLE_RADIUS, TURNTABLE_RADIUS), (-0.02, 0.02), (0.0, 0.008), shade(DECK, 0.8))
    with m.fine():
        for side in (-0.019, 0.019):
            m.box(
                (-TURNTABLE_RADIUS, TURNTABLE_RADIUS),
                (side - 0.0012, side + 0.0012),
                (0.008, 0.02),
                RAIL,
            )
            for k in range(7):
                at = -TURNTABLE_RADIUS + 2 * TURNTABLE_RADIUS * k / 6
                m.box(
                    (at - 0.0012, at + 0.0012), (side - 0.0012, side + 0.0012), (0.008, 0.02), RAIL
                )
    hut = TURNTABLE_RADIUS - 0.012
    m.box((hut - 0.022, hut + 0.004), (-0.024, -0.004), (0.008, 0.05), CAB)
    m.box((hut - 0.025, hut + 0.007), (-0.027, -0.001), (0.05, 0.056), shade(CAB, 0.6))
    with m.fine():
        m.box((hut - 0.018, hut), (-0.0245, -0.0235), (0.028, 0.044), GLASS)
    _turn(m, begin, bridge, *TURNTABLE)
    _turned_into(m, _engine(), bridge, *TURNTABLE)


def _turn(m: Model, begin: int, angle: float, du: float, dv: float) -> None:
    """Turns the faces added since there were `begin` of them `angle` radians about the origin
    and moves them to (du, dv)."""
    part = Model()
    part.faces = m.faces[begin:]
    del m.faces[begin:]
    _turned_into(m, part, angle, du, dv)


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


def _rod(
    m: Model,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    half: float,
    color: RGB,
) -> None:
    """A square rod `half` either side of the straight line from `a` to `b`, sloping as it
    likes."""
    (au, av, az), (bu, bv, bz) = a, b
    du, dv, dz = bu - au, bv - av, bz - az
    length = math.sqrt(du * du + dv * dv + dz * dz) or 1.0
    du, dv, dz = du / length, dv / length, dz / length
    side = (-dv, du, 0.0)
    norm = math.hypot(side[0], side[1]) or 1.0
    side = (side[0] / norm * half, side[1] / norm * half, 0.0)
    up = (
        dv * side[2] - dz * side[1],
        dz * side[0] - du * side[2],
        du * side[1] - dv * side[0],
    )
    scale = half / (math.sqrt(up[0] ** 2 + up[1] ** 2 + up[2] ** 2) or 1.0)
    up = (up[0] * scale, up[1] * scale, up[2] * scale)

    def at(p: tuple[float, float, float], s1: float, s2: float) -> tuple[float, float, float]:
        return (
            p[0] + side[0] * s1 + up[0] * s2,
            p[1] + side[1] * s1 + up[1] * s2,
            p[2] + side[2] * s1 + up[2] * s2,
        )

    for (s1, s2), (t1, t2) in (
        ((-1, -1), (1, -1)),
        ((1, -1), (1, 1)),
        ((1, 1), (-1, 1)),
        ((-1, 1), (-1, -1)),
    ):
        out = (
            side[0] * (s1 + t1) + up[0] * (s2 + t2),
            side[1] * (s1 + t1) + up[1] * (s2 + t2),
            side[2] * (s1 + t1) + up[2] * (s2 + t2),
        )
        _quad(m, [at(a, s1, s2), at(a, t1, t2), at(b, t1, t2), at(b, s1, s2)], color, out)


def _sand_tower() -> Model:
    """A tall sand tower: an ochre shaft on a stone footing with a string course and slit
    windows, a wider storey holding the dried sand with windows all round under a stone band, a
    red pointed roof, and one sand pipe sloping down over the lead track to fill engines' sand
    boxes."""
    m = Model()
    walls(m, (-0.03, 0.03), (-0.03, 0.03), 0.2, OCHRE, STONE)
    m.box((-0.04, 0.04), (-0.04, 0.04), (0.2, 0.27), OCHRE)
    m.box((-0.043, 0.043), (-0.043, 0.043), (0.198, 0.205), STONE)
    m.box((-0.042, 0.042), (-0.042, 0.042), (0.265, 0.272), STONE)
    cone(m, 0.0, 0.0, 0.06, 0.075, RED_ROOF, 0.272)
    _rod(m, (0.0, 0.04, 0.21), (0.0, 0.165, 0.13), 0.0045, SOOT)
    upright(m, 0.0, 0.165, 0.0045, (0.105, 0.134), SOOT)
    with m.fine():
        m.box((-0.0312, 0.0312), (-0.0312, 0.0312), (0.1, 0.104), STONE)
        for at in (-0.022, 0.0, 0.022):
            window(m, (at - 0.007, at + 0.007), (0.215, 0.255), 0.04, GLASS, FRAME_WHITE)
        for low in (0.05, 0.13):
            window(m, (-0.006, 0.006), (low, low + 0.032), 0.03, GLASS, FRAME_WHITE)
    turned = Model()
    for face in m.faces:
        turned.faces.append(face)
    _turned_into(m, turned, math.pi / 2, 0.0, 0.0)
    return m


def _lamp(m: Model, u: float, v: float) -> None:
    """A yard lamp on a post."""
    upright(m, u, v, 0.0025, (0.0, 0.09), SOOT)
    with m.fine():
        m.box((u - 0.005, u + 0.005), (v - 0.005, v + 0.005), (0.088, 0.098), LAMP)


def _rails_stack(m: Model, u: float, v: float) -> None:
    """Spare rails laid by the track on a pair of sleepers."""
    with m.fine():
        for k in range(4):
            off = k * 0.008
            m.box((u - 0.07, u + 0.07), (v + off - 0.0015, v + off + 0.0015), (0.0008, 0.004), PIER)
        for at in (u - 0.04, u + 0.04):
            m.box((at - 0.004, at + 0.004), (v - 0.006, v + 0.03), (0.0, 0.002), SLEEPERS)


def roundhouse(level: int = 3, sanded: bool = True) -> Model:
    """An engine shed of `level` after the model railway ones: its built bays (BUILT) standing
    side by side as one curved hall round the back of its turntable, rails running from the
    turntable into each and out along the lead track along u, a dark ballast apron round the
    turntable, yard lamps, spare rails, and, when the station has one, a tall sand tower by the
    lead."""
    m = Model()
    _disc(m, TURNTABLE, STALL_FROM + 0.004, BALLAST, 0.0004)
    built = BUILT[level]
    for k in built:
        _track(m, _round(STALL_HEADINGS[k], TURNTABLE_RADIUS), _round(STALL_HEADINGS[k], STALL_TO))
    _track(m, _round(0.0, TURNTABLE_RADIUS), (LEAD_END, 0.0))
    _turntable(m)
    for k in built:
        angle = STALL_HEADINGS[k]
        _turned_into(m, _bay(), angle, *_round(angle, STALL_FROM))
    if sanded:
        _turned_into(m, _sand_tower(), 0.0, 0.2, -0.17)
    _lamp(m, 0.06, 0.13)
    _lamp(m, 0.06, -0.12)
    _rails_stack(m, 0.2, 0.12)
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


def shed_sprite(heading: int, level: int = 1, sanded: bool = False) -> str:
    """The sprite of an engine shed of `level` whose lead track runs `heading` degrees on the
    map, with a sand tower by it if `sanded`."""
    return f"engine_shed_{level}_{heading}" + ("_sanded" if sanded else "")


PENS_SPRITE = "cattle_pens_annex"
FACILITIES: dict[str, Callable[[], Sprite]] = {
    shed_sprite(heading, level, sanded): modelled(
        partial(roundhouse, level, sanded),
        across=ROUNDHOUSE_SIZE,
        shadow=0.7,
        heading=math.radians(heading),
    )
    for level in BUILT
    for sanded in (False, True)
    for heading in SHED_HEADINGS
}
FACILITIES[PENS_SPRITE] = modelled(cattle_pens, across=PENS_SIZE, shadow=0.6)
