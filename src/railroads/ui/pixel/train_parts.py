"""Small parts that give train models (railroads.ui.pixel.trains) their detail when they are drawn
big: spoked wheels, buffers and couplings, rods, rails, rivets, louvres, ladders and the like.
Every face they add is fine (Model.fine), so the far zooms keep clean outlines; one lying flat on
a bigger face is painted straight after it (models.decals), never hidden under it.

Sizes are in model cells, like the models: u along the track, v across it, z up from the rails.
"""

import math

from railroads.ui.pixel.models import RGB, Face, Model, keeping_fine, shade

IRON: RGB = (40, 40, 46)
STEEL: RGB = (150, 166, 186)
BRIGHT: RGB = (214, 218, 224)
BUFFER_BEAM: RGB = (168, 42, 34)
LAMP: RGB = (246, 226, 150)
TAIL_LAMP: RGB = (210, 40, 36)
# How far apart layers of parts lying flat on one another are.
LAYER = 0.0006


def _side(v: float) -> float:
    return 1.0 if v > 0 else -1.0


def ring(
    m: Model,
    u: float,
    z: float,
    radius: float,
    v: float,
    color: RGB,
    points: int = 14,
    squash: float = 1.0,
) -> None:
    """A flat round plate on one side of the model, `v` out from its middle, facing out."""
    if not keeping_fine():
        return
    corners = []
    for k in range(points):
        angle = 2 * math.pi * k / points
        corners.append((u + radius * math.cos(angle), v, z + radius * squash * math.sin(angle)))
    m.faces.append(Face(corners, color, (0, _side(v), 0)))


def plate(m: Model, u: tuple[float, float], z: tuple[float, float], v: float, color: RGB) -> None:
    """A flat patch on one side of the model only, `v` out from its middle, facing out."""
    if not keeping_fine():
        return
    (u0, u1), (z0, z1) = u, z
    corners = [(u0, v, z0), (u1, v, z0), (u1, v, z1), (u0, v, z1)]
    m.faces.append(Face(corners, color, (0, _side(v), 0)))


def front_plate(
    m: Model, u: float, v: tuple[float, float], z: tuple[float, float], color: RGB
) -> None:
    """A flat patch on an end of the model at `u`, facing out along the track."""
    if not keeping_fine():
        return
    (v0, v1), (z0, z1) = v, z
    corners = [(u, v0, z0), (u, v1, z0), (u, v1, z1), (u, v0, z1)]
    m.faces.append(Face(corners, color, (_side(u), 0, 0)))


def front_ring(
    m: Model, u: float, z: float, radius: float, color: RGB, v: float = 0.0, points: int = 14
) -> None:
    """A flat round plate on an end of the model at `u`, facing out along the track."""
    if not keeping_fine():
        return
    corners = []
    for k in range(points):
        angle = 2 * math.pi * k / points
        corners.append((u, v + radius * math.cos(angle), z + radius * math.sin(angle)))
    m.faces.append(Face(corners, color, (_side(u), 0, 0)))


def bar(
    m: Model,
    a: tuple[float, float],
    b: tuple[float, float],
    thick: float,
    v: float,
    color: RGB,
) -> None:
    """A flat bar `thick` tall from `a` to `b`, each a (u, z) point, on one side `v` out."""
    if not keeping_fine():
        return
    (au, az), (bu, bz) = a, b
    du, dz = bu - au, bz - az
    length = math.hypot(du, dz) or 1.0
    nu, nz = -dz / length * thick / 2, du / length * thick / 2
    corners = [
        (au - nu, v, az - nz),
        (bu - nu, v, bz - nz),
        (bu + nu, v, bz + nz),
        (au + nu, v, az + nz),
    ]
    m.faces.append(Face(corners, color, (0, _side(v), 0)))


def both(v: float) -> tuple[float, float]:
    return v, -v


def spoked_wheel(
    m: Model,
    u: float,
    radius: float,
    v: float,
    color: RGB,
    hub: RGB = STEEL,
    spokes: int = 8,
    weight: bool = False,
) -> None:
    """The face of a wheel on both sides, over the box models.Model.wheels makes `v` out: a
    rim round spokes, a hub, and for a driving wheel a counterweight between spokes."""
    if not keeping_fine():
        return
    with m.fine():
        for side in both(v + 0.012):
            out = _side(side)
            z = radius
            ring(m, u, z, radius * 0.98, side + out * LAYER, color, 16)
            ring(m, u, z, radius * 0.8, side + out * 2 * LAYER, shade(color, 0.5), 16)
            for k in range(spokes):
                angle = 2 * math.pi * (k + 0.5) / spokes
                end = (u + radius * 0.8 * math.cos(angle), z + radius * 0.8 * math.sin(angle))
                bar(m, (u, z), end, max(0.008, radius * 0.16), side + out * 3 * LAYER, color)
            if weight:
                corners = []
                for k in range(7):
                    angle = math.radians(200 + 140 * k / 6)
                    corners.append(
                        (
                            u + radius * 0.78 * math.cos(angle),
                            side + out * 4 * LAYER,
                            z + radius * 0.78 * math.sin(angle),
                        )
                    )
                m.faces.append(Face(corners, color, (0, out, 0)))
            ring(m, u, z, max(0.008, radius * 0.24), side + out * 5 * LAYER, hub, 10)


def disc_wheel(m: Model, u: float, radius: float, v: float, color: RGB) -> None:
    """The face of a solid wheel on both sides: a tyre, a dished disc and a bright hub."""
    if not keeping_fine():
        return
    with m.fine():
        for side in both(v + 0.012):
            out = _side(side)
            ring(m, u, radius, radius * 0.98, side + out * LAYER, shade(color, 1.25), 14)
            ring(m, u, radius, radius * 0.78, side + out * 2 * LAYER, color, 14)
            ring(m, u, radius, radius * 0.3, side + out * 3 * LAYER, STEEL, 10)


def axle_box(m: Model, u: float, z: float, v: float, color: RGB = IRON) -> None:
    """An axle box over a wheel's hub with a leaf spring above it, on both sides."""
    if not keeping_fine():
        return
    with m.fine():
        for side in both(v):
            out = _side(side)
            inner, outer = sorted((side, side + out * 0.014))
            m.box((u - 0.018, u + 0.018), (inner, outer), (z - 0.016, z + 0.016), color)
            m.box((u - 0.04, u + 0.04), (inner, outer), (z + 0.018, z + 0.026), shade(color, 1.3))


def bogie_frame(
    m: Model, middle: float, span: float, z: float, v: float, color: RGB = IRON
) -> None:
    """The side frame of a bogie round its wheels: a beam with a dip in the middle and an axle
    box and spring over each end, on both sides."""
    if not keeping_fine():
        return
    with m.fine():
        for side in both(v):
            out = _side(side)
            inner, outer = sorted((side, side + out * 0.01))
            m.box((middle - span, middle + span), (inner, outer), (z, z + 0.014), color)
            m.box((middle - 0.03, middle + 0.03), (inner, outer), (z - 0.02, z), shade(color, 0.85))
            for end in (middle - span + 0.02, middle + span - 0.02):
                m.box(
                    (end - 0.016, end + 0.016),
                    (inner, outer + out * 0.004),
                    (z - 0.026, z - 0.002),
                    shade(color, 1.2),
                )


def buffers(
    m: Model,
    end: float,
    z: float,
    beam: RGB | None = BUFFER_BEAM,
    width: float = 0.075,
    heads: RGB = STEEL,
) -> None:
    """A buffer beam across the end at `end`, or none where the model has one, two round
    buffers on it and a hook between."""
    if not keeping_fine():
        return
    out = _side(end)
    with m.fine():
        if beam is not None:
            m.box(
                tuple(sorted((end, end + out * 0.012))),
                (-width, width),
                (z - 0.022, z + 0.022),
                beam,
            )
        else:
            end -= out * 0.012
        for v in (-0.05, 0.05):
            m.cylinder_along(tuple(sorted((end + out * 0.012, end + out * 0.03))), 0.008, z, IRON)
            m.faces[-12:] = [
                Face([(c[0], c[1] + v, c[2]) for c in f.corners], f.color, f.normal)
                for f in m.faces[-12:]
            ]
            front_ring(m, end + out * 0.032, z, 0.016, heads, v, 12)
        m.box(
            tuple(sorted((end + out * 0.012, end + out * 0.028))),
            (-0.008, 0.008),
            (z - 0.008, z + 0.006),
            IRON,
        )


def coupler(m: Model, end: float, z: float) -> None:
    """A knuckle coupler sticking out of the end at `end`, for cars and modern engines."""
    if not keeping_fine():
        return
    out = _side(end)
    with m.fine():
        m.box(tuple(sorted((end, end + out * 0.028))), (-0.012, 0.012), (z - 0.01, z + 0.008), IRON)
        m.box(
            tuple(sorted((end + out * 0.02, end + out * 0.034))),
            (-0.02, 0.02),
            (z - 0.012, z + 0.01),
            IRON,
        )


def handrail(
    m: Model, u: tuple[float, float], z: float, v: float, color: RGB = BRIGHT, posts: int = 3
) -> None:
    """A thin rail along both sides `v` out at height `z`, on short posts."""
    if not keeping_fine():
        return
    u0, u1 = u
    with m.fine():
        for side in both(v):
            inner, outer = sorted((side, side + _side(side) * 0.006))
            m.box((u0, u1), (inner, outer), (z, z + 0.006), color)
            for k in range(posts):
                at = u0 + (u1 - u0) * (k + 0.5) / posts
                inside = side - _side(side) * 0.012
                m.box(
                    (at - 0.003, at + 0.003),
                    tuple(sorted((inside, outer))),
                    (z - 0.003, z + 0.003),
                    color,
                )


def rivets(m: Model, u: tuple[float, float], z: float, v: float, color: RGB, count: int) -> None:
    """A row of rivet heads along both sides at height `z`."""
    if not keeping_fine():
        return
    u0, u1 = u
    with m.fine():
        for side in both(v):
            for k in range(count):
                at = u0 + (u1 - u0) * (k + 0.5) / count
                plate(m, (at - 0.004, at + 0.004), (z - 0.004, z + 0.004), side, color)


def louvres(
    m: Model,
    u: tuple[float, float],
    z: tuple[float, float],
    v: float,
    color: RGB,
    count: int,
    upright: bool = False,
) -> None:
    """A grille of slats on both sides: dark behind, lighter slats across it."""
    if not keeping_fine():
        return
    (u0, u1), (z0, z1) = u, z
    with m.fine():
        for side in both(v):
            out = _side(side)
            plate(m, (u0, u1), (z0, z1), side, shade(color, 0.45))
            for k in range(count):
                if upright:
                    at = u0 + (u1 - u0) * (k + 0.5) / count
                    plate(m, (at - 0.003, at + 0.003), (z0, z1), side + out * LAYER, color)
                else:
                    at = z0 + (z1 - z0) * (k + 0.5) / count
                    plate(m, (u0, u1), (at - 0.003, at + 0.002), side + out * LAYER, color)


def lines(
    m: Model,
    u: tuple[float, float],
    z: tuple[float, float],
    v: float,
    color: RGB,
    count: int,
    upright: bool = False,
) -> None:
    """Thin lines on both sides, between planks or panels: `count` of them evenly between."""
    if not keeping_fine():
        return
    (u0, u1), (z0, z1) = u, z
    with m.fine():
        for side in both(v):
            for k in range(1, count + 1):
                if upright:
                    at = u0 + (u1 - u0) * k / (count + 1)
                    plate(m, (at - 0.002, at + 0.002), (z0, z1), side, color)
                else:
                    at = z0 + (z1 - z0) * k / (count + 1)
                    plate(m, (u0, u1), (at - 0.002, at + 0.002), side, color)


def ribs(
    m: Model, u: tuple[float, float], z: tuple[float, float], v: float, color: RGB, count: int
) -> None:
    """Upright stiffening ribs standing out of both sides."""
    if not keeping_fine():
        return
    (u0, u1), (z0, z1) = u, z
    with m.fine():
        for side in both(v):
            inner, outer = sorted((side, side + _side(side) * 0.006))
            for k in range(count):
                at = u0 + (u1 - u0) * k / max(1, count - 1)
                m.box((at - 0.004, at + 0.004), (inner, outer), (z0, z1), color)


def ladder(m: Model, u: float, z: tuple[float, float], v: float, color: RGB = BRIGHT) -> None:
    """A ladder up both sides at `u`: two rails and rungs."""
    if not keeping_fine():
        return
    z0, z1 = z
    with m.fine():
        for side in both(v):
            out = _side(side)
            for at in (u - 0.014, u + 0.014):
                plate(m, (at - 0.0025, at + 0.0025), (z0, z1), side + out * LAYER, color)
            rungs = max(2, round((z1 - z0) / 0.025))
            for k in range(rungs):
                at = z0 + (z1 - z0) * (k + 0.5) / rungs
                plate(
                    m, (u - 0.014, u + 0.014), (at - 0.002, at + 0.002), side + out * LAYER, color
                )


def end_ladder(m: Model, end: float, z: tuple[float, float], color: RGB = BRIGHT) -> None:
    """A ladder up the end of the model at `end`."""
    if not keeping_fine():
        return
    z0, z1 = z
    out = _side(end)
    with m.fine():
        for v in (-0.03, 0.03):
            front_plate(m, end + out * LAYER, (v - 0.003, v + 0.003), (z0, z1), color)
        rungs = max(2, round((z1 - z0) / 0.025))
        for k in range(rungs):
            at = z0 + (z1 - z0) * (k + 0.5) / rungs
            front_plate(m, end + out * LAYER, (-0.03, 0.03), (at - 0.002, at + 0.002), color)


def brake_wheel(m: Model, end: float, z: float, v: float = 0.04) -> None:
    """A hand-brake wheel on the end of a car at `end`."""
    if not keeping_fine():
        return
    out = _side(end)
    with m.fine():
        m.box(tuple(sorted((end, end + out * 0.006))), (v - 0.003, v + 0.003), (z - 0.06, z), IRON)
        front_ring(m, end + out * 0.007, z, 0.02, IRON, v, 10)
        front_ring(m, end + out * 0.0075, z, 0.012, shade(IRON, 0.5), v, 10)


def lamp(m: Model, end: float, z: float, v: float, color: RGB = LAMP, size: float = 0.012) -> None:
    """A lamp on the end at `end`: a dark rim round a lit glass."""
    if not keeping_fine():
        return
    out = _side(end)
    with m.fine():
        front_ring(m, end + out * LAYER, z, size, IRON, v, 10)
        front_ring(m, end + out * 2 * LAYER, z, size * 0.7, color, v, 10)


def window_frames(
    m: Model,
    u: tuple[float, float],
    z: tuple[float, float],
    v: float,
    frame: RGB,
    glass: RGB,
    panes: int = 1,
) -> None:
    """A framed window on both sides, split into `panes` panes with a glint on each."""
    if not keeping_fine():
        return
    (u0, u1), (z0, z1) = u, z
    with m.fine():
        for side in both(v):
            out = _side(side)
            plate(m, (u0, u1), (z0, z1), side, frame)
            width = (u1 - u0) / panes
            for k in range(panes):
                a = u0 + k * width + 0.004
                b = a + width - 0.008
                plate(m, (a, b), (z0 + 0.004, z1 - 0.004), side + out * LAYER, glass)
                plate(
                    m,
                    (a + 0.003, a + 0.003 + (b - a) * 0.3),
                    (z1 - 0.004 - (z1 - z0) * 0.35, z1 - 0.006),
                    side + out * 2 * LAYER,
                    shade(glass, 1.6),
                )


def stack(
    m: Model, u: float, z: tuple[float, float], radius: float, color: RGB, v: float = 0.0
) -> None:
    """A short exhaust stack or roof fan standing up, with a dark mouth on top."""
    if not keeping_fine():
        return
    with m.fine():
        m.cylinder_up(u, radius, z, color)
        if v:
            m.faces[-11:] = [
                Face([(c[0], c[1] + v, c[2]) for c in f.corners], f.color, f.normal)
                for f in m.faces[-11:]
            ]
        mouth = []
        for k in range(10):
            angle = 2 * math.pi * k / 10
            mouth.append(
                (
                    u + radius * 0.65 * math.cos(angle),
                    v + radius * 0.65 * math.sin(angle),
                    z[1] + LAYER,
                )
            )
        m.faces.append(Face(mouth, shade(color, 0.35), (0, 0, 1)))


def roof_fan(m: Model, u: float, roof: float, radius: float, color: RGB) -> None:
    """A round fan grille lying on a roof: a rim, dark blades and a hub."""
    if not keeping_fine():
        return
    with m.fine():
        m.cylinder_up(u, radius, (roof, roof + 0.008), shade(color, 0.8))
        top = roof + 0.008 + LAYER
        for size, tint in ((0.85, shade(color, 0.4)), (0.3, color)):
            corners = []
            for k in range(12):
                angle = 2 * math.pi * k / 12
                corners.append(
                    (u + radius * size * math.cos(angle), radius * size * math.sin(angle), top)
                )
            m.faces.append(Face(corners, tint, (0, 0, 1)))
            top += LAYER


def roof_plate(
    m: Model, u: tuple[float, float], v: tuple[float, float], z: float, color: RGB
) -> None:
    """A flat patch lying on a roof at height `z`: a hatch, a walkway, a vent."""
    if not keeping_fine():
        return
    (u0, u1), (v0, v1) = u, v
    m.faces.append(Face([(u0, v0, z), (u1, v0, z), (u1, v1, z), (u0, v1, z)], color, (0, 0, 1)))


def pantograph(m: Model, u: float, roof: float, color: RGB, raised: bool = True) -> None:
    """A pantograph on its base frame on two insulators: lower and upper arms meeting at a knee,
    and a pan head with horns at both ends across the top."""
    if not keeping_fine():
        return
    with m.fine():
        for v in (-0.035, 0.035):
            m.cylinder_up(u - 0.06, 0.007, (roof, roof + 0.016), (230, 230, 220))
            m.faces[-11:] = [
                Face([(c[0], c[1] + v, c[2]) for c in f.corners], f.color, f.normal)
                for f in m.faces[-11:]
            ]
            m.cylinder_up(u + 0.06, 0.007, (roof, roof + 0.016), (230, 230, 220))
            m.faces[-11:] = [
                Face([(c[0], c[1] + v, c[2]) for c in f.corners], f.color, f.normal)
                for f in m.faces[-11:]
            ]
        base = roof + 0.016
        m.box((u - 0.075, u + 0.075), (-0.045, 0.045), (base, base + 0.008), color)
        high = base + (0.1 if raised else 0.05)
        knee = (u + 0.05, base + (high - base) * 0.5)
        for v in (-0.03, 0.03):
            m.slope((u - 0.05, base + 0.008), knee, (v - 0.004, v + 0.004), color)
            m.slope(knee, (u - 0.005, high), (v - 0.003, v + 0.003), color)
        m.box((u - 0.012, u + 0.012), (-0.075, 0.075), (high, high + 0.008), color)
        for side in (-1, 1):
            m.slope(
                (u - 0.004, high + 0.004),
                (u - 0.004, high - 0.008),
                tuple(sorted((side * 0.075, side * 0.09))),
                color,
            )


def lumps(
    m: Model,
    u: tuple[float, float],
    v: tuple[float, float],
    z: float,
    color: RGB,
    size: float = 0.022,
    rise: float = 0.012,
) -> None:
    """A heap of lumps over the patch `u` by `v` at height `z`, higher in the middle, in
    lighter and darker shades of `color`: coal, ore, potatoes."""
    if not keeping_fine():
        return
    (u0, u1), (v0, v1) = u, v
    across = max(1, round((v1 - v0) / size))
    along = max(1, round((u1 - u0) / size))
    with m.fine():
        for i in range(along):
            for j in range(across):
                a = u0 + (u1 - u0) * i / along
                b = v0 + (v1 - v0) * j / across
                middle = 1 - abs((i + 0.5) / along - 0.5) - abs((j + 0.5) / across - 0.5)
                high = z + rise * (0.3 + middle + 0.35 * ((i * 7 + j * 3) % 4) / 3)
                tint = shade(color, (0.8, 1.15, 0.95, 1.3, 0.9)[(i * 3 + j * 5) % 5])
                inset = size * 0.12
                m.box(
                    (a + inset, a + (u1 - u0) / along - inset),
                    (b + inset, b + (v1 - v0) / across - inset),
                    (z, high),
                    tint,
                )
