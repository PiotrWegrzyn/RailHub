"""Buildings built as small 3D models (railroads.ui.pixel.models) and drawn from them, shaded by
the light like the trains and the stations on hexes, instead of painted pixel by pixel: walls on
a plinth, pitched or flat roofs, windows, doors, chimneys and cornices. A model is drawn at
whatever fineness the sheet asks (railroads.ui.pixel.art.painter.drawn_at), with its fine details
once it is drawn big enough to show them, so the closer the zoom the more it shows.

Models are in cells, u and v across the ground and z up, seen from the hex map's look turned
HEADING, so their two front walls face the viewer as the painted buildings' do."""

import itertools
import math
from collections.abc import Callable
from dataclasses import replace

from PIL import Image, ImageDraw

from railroads.sim.grids import HexGrid
from railroads.ui.pixel.art.painter import RGB, SHADOW, Sprite, fineness, shade
from railroads.ui.pixel.models import HEIGHT_PER_WIDTH, Face, Model, draw_model
from railroads.ui.pixel.projection import PixelProjection

# The hex map's look (railroads.ui.iso_camera.LOOKS): not turned, a cell wide per cell.
PROJECTION = PixelProjection.of(HexGrid(rows=1, cols=1, km_per_cell=1.0), 0.0, 1.0)
# How a building is turned: its u axis runs down to the right of the screen, so its long front
# wall faces the viewer on the left and its end wall on the right, lit from the top left.
HEADING = math.radians(45.0)
# From what fineness models are drawn with their fine details: windows' frames and sills, roof
# tiles, doors' panels and the like, which only clutter a smaller picture.
FINE_FROM = 1.5
# Where buildings are lit from on the screen: the left and a little towards the viewer, so the
# front wall on the left is lit and the end wall on the right shaded, as on the painted ones.
LIGHT_FROM = (-1.0, 0.45)
# How high a part must stand to cast a shadow: patches of earth on the ground cast none.
TALL_ENOUGH = 0.02
# How high a flat patch may lie and still count as lying on the ground.
GROUND_HIGH = 0.002

SOIL: RGB = (74, 64, 46)
SILL: RGB = (226, 222, 210)
FRAME: RGB = (238, 236, 228)
DOOR: RGB = (96, 64, 44)
STEP: RGB = (206, 200, 188)
EARTH: RGB = (146, 124, 84)
WORN: RGB = (120, 128, 70)
FOOTING: RGB = (62, 70, 40)
WINDOW: RGB = (58, 70, 92)
CRATE: RGB = (156, 108, 62)


def modelled(
    build: Callable[[], Model],
    across: float = 1.0,
    up: float = 1.0,
    shadow: float = 1.0,
    heading: float = HEADING,
) -> Callable[[], Sprite]:
    """The draw function of a building built by `build`: its model stretched `across` the ground
    and `up`, turned `heading` radians, drawn at the fineness the sheet asks, standing in a soft
    shadow `shadow` times as wide as its footprint. Stretching lets a model be built in handy sizes and match the size and
    tall look of the painted buildings beside it."""

    def draw() -> Sprite:
        pixels = fineness()
        model = _stretched(build(), across, up)
        ground, standing = Model(), Model()
        for face in model.faces:
            (ground if _lying(face) else standing).faces.append(face)
        sprites = [
            Sprite(
                *draw_model(
                    part,
                    heading,
                    PROJECTION.matrix,
                    pixels,
                    fine=pixels >= FINE_FROM,
                    light_from=LIGHT_FROM,
                    depth_test=True,
                    outline=part is standing,
                )
            )
            for part in (ground, standing)
            if part.faces
        ]
        return _shadowed(_stacked(sprites), model, pixels * shadow)

    return draw


def _lying(face: Face) -> bool:
    """Whether a face is a patch lying flat on the ground, which is drawn without an outline."""
    return face.normal[2] > 0.99 and max(corner[2] for corner in face.corners) < GROUND_HIGH


def _stacked(sprites: list[Sprite]) -> Sprite:
    """Sprites laid one over another in order, their anchors on the same spot."""
    left = max(sprite.anchor[0] for sprite in sprites)
    top = max(sprite.anchor[1] for sprite in sprites)
    right = max(sprite.image.width - sprite.anchor[0] for sprite in sprites)
    bottom = max(sprite.image.height - sprite.anchor[1] for sprite in sprites)
    image = Image.new("RGBA", (left + right, top + bottom), (0, 0, 0, 0))
    for sprite in sprites:
        image.alpha_composite(sprite.image, (left - sprite.anchor[0], top - sprite.anchor[1]))
    return Sprite(image, (left, top))


def _stretched(model: Model, across: float, up: float) -> Model:
    """The model `across` times as wide and long and `up` times as tall, its faces still facing
    the right way."""
    if across == 1.0 and up == 1.0:
        return model
    stretched = Model()
    stretched.footprints = [
        ((u0 * across, u1 * across), (v0 * across, v1 * across))
        for (u0, u1), (v0, v1) in model.footprints
    ]
    for face in model.faces:
        nu, nv, nz = face.normal[0] / across, face.normal[1] / across, face.normal[2] / up
        length = math.sqrt(nu * nu + nv * nv + nz * nz) or 1.0
        stretched.faces.append(
            replace(
                face,
                corners=[(u * across, v * across, z * up) for u, v, z in face.corners],
                normal=(nu / length, nv / length, nz / length),
            )
        )
    return stretched


def _shadowed(sprite: Sprite, model: Model, pixels: float) -> Sprite:
    """The sprite with a soft dark shadow on the ground under it, reaching out to the lower right
    away from the light, as the painted buildings have."""
    if model.footprints:
        us = [u for (u0, u1), _ in model.footprints for u in (u0, u1)]
        vs = [v for _, (v0, v1) in model.footprints for v in (v0, v1)]
    else:
        standing = [face for face in model.faces if max(c[2] for c in face.corners) > TALL_ENOUGH]
        us = [c[0] for face in standing for c in face.corners]
        vs = [c[1] for face in standing for c in face.corners]
    if not us:
        return sprite
    reach = max(max(us) - min(us), max(vs) - min(vs))
    scale = PROJECTION.matrix * pixels
    half_w = float(abs(scale[0, 0]) * reach * 0.62) + 1
    half_h = half_w * 0.42
    w, h = sprite.image.size
    ax, ay = sprite.anchor
    pad = math.ceil(half_w * 0.5)
    image = Image.new("RGBA", (w + pad, h + pad), (0, 0, 0, 0))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    cx, cy = ax + half_w * 0.3, ay + half_h * 0.35
    ImageDraw.Draw(overlay).ellipse(
        (cx - half_w, cy - half_h, cx + half_w, cy + half_h), fill=SHADOW
    )
    image.alpha_composite(overlay)
    image.alpha_composite(sprite.image)
    box = image.getchannel("A").getbbox() or (0, 0, 1, 1)
    return Sprite(image.crop(box), (ax - box[0], ay - box[1]))


def walls(
    m: Model,
    u: tuple[float, float],
    v: tuple[float, float],
    height: float,
    color: RGB,
    plinth: RGB | None = None,
) -> None:
    """Walls round `u` by `v` up to `height`, on a low plinth of `plinth` if given, with a dark
    line of earth along their feet up close."""
    m.footprints.append((u, v))
    low = 0.0
    if plinth is not None:
        m.box(u, v, (0.0, 0.014), plinth)
        low = 0.014
    m.box(u, v, (low, height), color)
    with m.fine():
        (u0, u1), (v0, v1) = u, v
        for side in (v0 - 0.0006, v1 + 0.0006):
            m.panel((u0, u1), (0.0, 0.004), side, SOIL)


def gable_roof(
    m: Model,
    u: tuple[float, float],
    v: tuple[float, float],
    z: tuple[float, float],
    color: RGB,
    ends: RGB,
    overhang: float = 0.012,
) -> None:
    """A pitched roof over walls `u` by `v` from eaves `z[0]` to ridge `z[1]`, its eaves and
    verges out by `overhang`, with gables in `ends` and, up close, rows of tiles and a ridge."""
    (u0, u1), (v0, v1) = u, v
    out = (u0 - overhang, u1 + overhang), (v0 - overhang, v1 + overhang)
    m.roof(*out, z, color, ends)
    with m.fine():
        eaves, ridge = z
        middle = (v0 + v1) / 2
        rows = max(2, round((ridge - eaves) / 0.012))
        for k in range(1, rows):
            t = k / rows
            for edge in (out[1][0], out[1][1]):
                v = edge + (middle - edge) * t
                high = eaves + (ridge - eaves) * t
                m.box(out[0], (v - 0.0012, v + 0.0012), (high, high + 0.0014), shade(color, 0.72))
        m.box(
            out[0],
            (middle - 0.004, middle + 0.004),
            (ridge - 0.002, ridge + 0.004),
            shade(color, 0.8),
        )


def window(
    m: Model,
    u: tuple[float, float],
    z: tuple[float, float],
    v: float,
    glass: RGB,
    frame: RGB = FRAME,
) -> None:
    """A window on both long walls `v` out, with a frame, a cross bar and a sill up close."""
    (u0, u1), (z0, z1) = u, z
    m.panel(u, z, v + 0.0006, glass)
    with m.fine():
        m.panel((u0 - 0.003, u1 + 0.003), (z0 - 0.004, z0), v + 0.0012, SILL)
        mid_u, mid_z = (u0 + u1) / 2, (z0 + z1) / 2
        m.panel((mid_u - 0.0012, mid_u + 0.0012), z, v + 0.0012, frame)
        m.panel(u, (mid_z - 0.0012, mid_z + 0.0012), v + 0.0012, frame)
        for edge in ((u0 - 0.0016, u0), (u1, u1 + 0.0016)):
            m.panel(edge, z, v + 0.0012, frame)
        m.panel(u, (z1, z1 + 0.0016), v + 0.0012, frame)


def door(m: Model, u: tuple[float, float], height: float, v: float, color: RGB = DOOR) -> None:
    """A door on both long walls `v` out, with a frame, panels and a doorstep up close."""
    (u0, u1) = u
    m.panel(u, (0.008, height), v + 0.0006, color)
    with m.fine():
        for edge in ((u0 - 0.002, u0), (u1, u1 + 0.002)):
            m.panel(edge, (0.008, height + 0.002), v + 0.0012, FRAME)
        m.panel((u0 - 0.002, u1 + 0.002), (height, height + 0.002), v + 0.0012, FRAME)
        for low in (0.016, 0.016 + (height - 0.016) / 2):
            m.panel(
                (u0 + 0.003, u1 - 0.003),
                (low, low + (height - 0.03) / 2),
                v + 0.0012,
                shade(color, 0.8),
            )
        m.box((u0 - 0.006, u1 + 0.006), (v, v + 0.012), (0.0, 0.008), STEP)
        m.box((u0 - 0.006, u1 + 0.006), (-v - 0.012, -v), (0.0, 0.008), STEP)


def chimney(
    m: Model,
    u: float,
    v: float,
    z: tuple[float, float],
    color: RGB,
    half: float = 0.009,
) -> None:
    """A chimney stack rising through a roof from `z[0]` to `z[1]`, with a cap and up close a
    pot."""
    base, top = z
    m.box((u - half, u + half), (v - half, v + half), (base, top), color)
    m.box(
        (u - half - 0.002, u + half + 0.002),
        (v - half - 0.002, v + half + 0.002),
        (top, top + 0.005),
        shade(color, 0.6),
    )
    with m.fine():
        upright(m, u, v, half * 0.45, (top + 0.005, top + 0.012), (150, 84, 58))


def cornice(
    m: Model, u: tuple[float, float], v: tuple[float, float], height: float, color: RGB
) -> None:
    """A thin ledge round the walls at `height`, standing out a little."""
    (u0, u1), (v0, v1) = u, v
    m.box((u0 - 0.003, u1 + 0.003), (v0 - 0.003, v1 + 0.003), (height, height + 0.005), color)


# Placing parts where the painted sprites draw them: the Painter lays a sprite out in art pixels
# of the map at fineness 1, x to the right and y down from where the sprite stands, on the hex
# map's look, which these turn into the model's cells.
_PLANE = PROJECTION.matrix
_UP = float(math.hypot(*_PLANE[:, 0])) * HEIGHT_PER_WIDTH


def spot(x: float, y: float) -> tuple[float, float]:
    """The point on the ground (u, v) under the painted sprite's point (x, y)."""
    across, down = x / _PLANE[0, 0], y / _PLANE[1, 1]
    cos, sin = math.cos(HEADING), math.sin(HEADING)
    return across * cos + down * sin, -across * sin + down * cos


def tall(pixels: float) -> float:
    """How high a part the painted sprite draws `pixels` tall stands."""
    return pixels / _UP


def wide(pixels: float) -> float:
    """How far across the ground the painted sprite's `pixels` across reach."""
    return pixels / _PLANE[0, 0]


def box_at(
    x: float, y: float, half_width: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The square footprint `u` by `v` of the painted sprite's box (Painter.box) centred at
    (x, y), `half_width` pixels either side."""
    return hall_at(x, y, half_width, half_width)


def hall_at(
    x: float, y: float, length: float, depth: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The oblong footprint `u` by `v` of the painted sprite's hall (Painter.hall) centred at
    (x, y): its lit wall `length` pixels across along u, its shaded wall `depth` along v."""
    front = (x + (length - depth) / 2, y + (length + depth) / 4)
    left = (front[0] - length, front[1] - length / 2)
    right = (front[0] + depth, front[1] - depth / 2)
    us, vs = zip(*(spot(*corner) for corner in (left, front, right)), strict=True)
    return (min(us), max(us)), (min(vs), max(vs))


def patch(m: Model, points: list[tuple[float, float]], color: RGB, z: float = 0.0008) -> None:
    """A flat patch lying on the ground through the painted sprite's `points`: trodden earth,
    a track, a field."""
    corners = [(*spot(x, y), z) for x, y in points]
    m.faces.append(Face(corners, color, (0.0, 0.0, 1.0)))


def blob(
    m: Model, x: float, y: float, half_x: float, half_y: float, color: RGB, z: float = 0.0008
) -> None:
    """A round patch on the ground centred on the painted sprite's (x, y), `half_x` pixels
    across and `half_y` down."""
    patch(
        m,
        [
            (x + half_x * math.cos(a), y + half_y * math.sin(a))
            for a in (2 * math.pi * k / 14 for k in range(14))
        ],
        color,
        z,
    )


def cone(
    m: Model, u: float, v: float, radius: float, height: float, color: RGB, base: float = 0.0
) -> None:
    """A heap standing on (u, v): a cone `radius` round and `height` high, its foot `base` up."""
    sides = 10
    apex = (u, v, base + height)
    slant = math.hypot(radius, height) or 1.0
    for k in range(sides):
        a0, a1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
        mid = (a0 + a1) / 2
        normal = (height / slant * math.cos(mid), height / slant * math.sin(mid), radius / slant)
        corners = [
            (u + radius * math.cos(a0), v + radius * math.sin(a0), base),
            (u + radius * math.cos(a1), v + radius * math.sin(a1), base),
            apex,
        ]
        m.faces.append(Face(corners, color, normal))


def upright(
    m: Model, u: float, v: float, radius: float, z: tuple[float, float], color: RGB
) -> None:
    """A round upright standing at (u, v), `radius` round from `z[0]` to `z[1]`: a silo, a
    barrel, a chimney."""
    begin = len(m.faces)
    m.cylinder_up(0.0, radius, z, color)
    m.move(begin, u, v)


def dome(m: Model, u: float, v: float, radius: float, z: tuple[float, float], color: RGB) -> None:
    """A rounded top `radius` round on (u, v), from `z[0]` up to `z[1]` in its middle."""
    begin = len(m.faces)
    m.cap(0.0, radius, z, color)
    m.move(begin, u, v)


def crate(m: Model, x: float, y: float, load: RGB, half: float = 1.6) -> None:
    """A slatted wooden crate standing on the painted sprite's (x, y), full of `load`."""
    u, v = box_at(x, y, half)
    high = tall(half)
    m.box(u, v, (0.0, high), CRATE)
    m.box(u, v, (high, high + 0.004), load)
    with m.fine():
        for side in (v[0] - 0.0006, v[1] + 0.0006):
            m.panel(u, (high * 0.45, high * 0.55), side, shade(CRATE, 0.7))


def sack(m: Model, x: float, y: float, fill: RGB, tall_px: float = 4, half: float = 1.5) -> None:
    """A full sack standing on the painted sprite's (x, y), tied off at its neck."""
    u, v = spot(x, y)
    radius = wide(half) * 0.8
    upright(m, u, v, radius, (0.0, tall(tall_px) * 0.55), fill)
    dome(m, u, v, radius, (tall(tall_px) * 0.55, tall(tall_px)), fill)
    upright(m, u, v, radius * 0.35, (tall(tall_px) * 0.95, tall(tall_px) * 1.08), shade(fill, 0.6))


def fence(m: Model, a: tuple[float, float], b: tuple[float, float]) -> None:
    """A post-and-rail fence along the ground from the painted sprite's `a` to `b`."""
    (ua, va), (ub, vb) = spot(*a), spot(*b)
    post, rail = (122, 90, 60), (206, 176, 126)
    for k in range(4):
        t = k / 3
        u, v = ua + (ub - ua) * t, va + (vb - va) * t
        m.box((u - 0.003, u + 0.003), (v - 0.003, v + 0.003), (0.0, tall(3)), post)
    length = math.hypot(ub - ua, vb - va) or 1.0
    nu, nv = -(vb - va) / length * 0.002, (ub - ua) / length * 0.002
    high = tall(2)
    m.faces.append(
        Face(
            [
                (ua + nu, va + nv, high),
                (ub + nu, vb + nv, high),
                (ub + nu, vb + nv, high + 0.006),
                (ua + nu, va + nv, high + 0.006),
            ],
            rail,
            _toward(nu, nv),
        )
    )


def _toward(nu: float, nv: float) -> tuple[float, float, float]:
    length = math.hypot(nu, nv) or 1.0
    return (nu / length, nv / length, 0.0)


def plinth(
    m: Model, u: tuple[float, float], v: tuple[float, float], height: float, color: RGB
) -> None:
    """A darker band `height` high along the foot of a block's walls."""
    m.box(
        (u[0] - 0.0008, u[1] + 0.0008),
        (v[0] - 0.0008, v[1] + 0.0008),
        (0.0, height),
        color,
    )


def worn(m: Model, x: float, y: float, half: float, colour: RGB = EARTH) -> None:
    """A patch of bare trodden earth on the ground at the painted sprite's (x, y), `half` across,
    its rim worn into the grass."""
    blob(m, x, y, half, half / 2, WORN, 0.0006)
    blob(m, x, y, half * 0.68, half * 0.3, colour, 0.0009)


def track(m: Model, points: list[tuple[float, float]], wide_px: float = 1.0) -> None:
    """A trodden track `wide_px` across along the ground through the painted sprite's points."""
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / length * wide_px * 0.6, dx / length * wide_px * 0.6
        patch(
            m,
            [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)],
            EARTH,
        )


def footing(m: Model, u: tuple[float, float], v: tuple[float, float]) -> None:
    """A dark damp strip on the ground round a block's walls."""
    (u0, u1), (v0, v1) = u, v
    m.box((u0 - 0.012, u1 + 0.012), (v0 - 0.012, v1 + 0.012), (0.0, 0.0005), FOOTING)


def along(u: tuple[float, float], t: float) -> float:
    """A spot `t` of the way along the lit wall, from its left corner to its front one."""
    return u[0] + (u[1] - u[0]) * t


def across(v: tuple[float, float], t: float) -> float:
    """A spot `t` of the way along the shaded wall, from its front corner to its right one."""
    return v[1] - (v[1] - v[0]) * t


def below(height: float, drop: float, size: float) -> tuple[float, float]:
    """The heights of a patch `drop` pixels below the top of a wall `height` pixels high, `size`
    pixels tall."""
    return tall(height - drop - size), tall(height - drop)


def log(
    m: Model,
    a: tuple[float, float],
    b: tuple[float, float],
    z: float,
    radius: float,
    color: RGB,
    end: RGB,
) -> None:
    """A log lying from the painted sprite's point `a` to `b` with its middle `z` up, its cut
    ends showing `end`."""
    (ua, va), (ub, vb) = spot(*a), spot(*b)
    length = math.hypot(ub - ua, vb - va)
    begin = len(m.faces)
    m.cylinder_along((0.0, length), radius, z, color, end)
    angle = math.atan2(vb - va, ub - ua)
    cos, sin = math.cos(angle), math.sin(angle)
    m.faces[begin:] = [
        replace(
            face,
            corners=[
                (ua + u * cos - v * sin, va + u * sin + v * cos, h) for u, v, h in face.corners
            ],
            normal=(
                face.normal[0] * cos - face.normal[1] * sin,
                face.normal[0] * sin + face.normal[1] * cos,
                face.normal[2],
            ),
        )
        for face in m.faces[begin:]
    ]


def smoke(
    m: Model, u: float, v: float, z: float, puffs: int = 2, color: RGB = (226, 226, 230)
) -> None:
    """Puffs of pale smoke rising from (u, v, z), each bigger and further up."""
    for k in range(puffs):
        radius = 0.022 + 0.012 * k
        low = z + 0.012 + k * 0.045
        upright(m, u + 0.006 * k, v - 0.006 * k, radius, (low, low + radius * 0.4), color)
        dome(
            m, u + 0.006 * k, v - 0.006 * k, radius, (low + radius * 0.4, low + radius * 1.4), color
        )


def barrel(m: Model, u: float, v: float, radius: float, height: float, color: RGB) -> None:
    """A barrel standing on (u, v), its hoops dark up close."""
    upright(m, u, v, radius, (0.0, height), color)
    with m.fine():
        for at in (0.2, 0.8):
            upright(
                m,
                u,
                v,
                radius + 0.0012,
                (height * at - 0.002, height * at + 0.002),
                shade(color, 0.55),
            )
        dome(m, u, v, radius * 0.98, (height, height + 0.002), shade(color, 0.8))


def paddock(m: Model, x: float, y: float, half: float) -> None:
    """A fenced square patch of ground centred on the painted sprite's (x, y)."""
    corners = [(x - half, y), (x, y - half / 2), (x + half, y), (x, y + half / 2)]
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        fence(m, a, b)


def lathe(
    m: Model,
    u: float,
    v: float,
    profile: list[tuple[float, float]],
    color: RGB,
    cap: RGB | None = None,
    sides: int = 16,
) -> None:
    """A round thing on an upright axis at (u, v), like a tank, a kiln or a cone: `profile`
    gives its radius at each height from its foot up; closed with a flat top in `cap` if given."""
    for (z0, r0), (z1, r1) in itertools.pairwise(profile):
        slope = math.atan2(r0 - r1, z1 - z0) if z1 != z0 else math.pi / 2
        for k in range(sides):
            a0, a1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
            mid = (a0 + a1) / 2
            normal = (
                math.cos(mid) * math.cos(slope),
                math.sin(mid) * math.cos(slope),
                math.sin(slope),
            )
            corners = [
                (u + r0 * math.cos(a0), v + r0 * math.sin(a0), z0),
                (u + r0 * math.cos(a1), v + r0 * math.sin(a1), z0),
                (u + r1 * math.cos(a1), v + r1 * math.sin(a1), z1),
                (u + r1 * math.cos(a0), v + r1 * math.sin(a0), z1),
            ]
            if r1 < 1e-6:
                corners = corners[:3]
            m.faces.append(Face(corners, color, normal))
    if cap is not None:
        z, r = profile[-1]
        m.faces.append(
            Face(
                [
                    (
                        u + r * math.cos(2 * math.pi * k / sides),
                        v + r * math.sin(2 * math.pi * k / sides),
                        z,
                    )
                    for k in range(sides)
                ],
                cap,
                (0.0, 0.0, 1.0),
            )
        )


def turned(
    m: Model,
    x: float,
    y: float,
    profile: list[tuple[float, float]],
    color: RGB,
    cap: RGB | None = None,
) -> None:
    """The painted sprite's _turned round thing at (x, y): `profile` gives its half-width in
    pixels at each height in pixels from its foot up."""
    u, v = spot(x, y)
    lathe(m, u, v, [(tall(h), wide(r)) for h, r in profile], color, cap)


def stack(
    m: Model, x: float, foot: float, top: float, color: RGB, width: float = 2, puffs: int = 0
) -> float:
    """The painted sprite's chimney stack drawn from `foot` up to `top` at x (lowered as the
    painted ones are), standing on the ground behind; returns how high its cap is."""
    from railroads.ui.pixel.art.painter import chimney_top

    u, v = spot(x + width / 2, foot)
    high = tall(foot - chimney_top(foot, top))
    half = wide(width) / 2
    m.box((u - half, u + half), (v - half, v + half), (0.0, high), color)
    m.box(
        (u - half - 0.002, u + half + 0.002),
        (v - half - 0.002, v + half + 0.002),
        (high, high + 0.006),
        shade(color, 0.6),
    )
    if puffs:
        smoke(m, u, v, high + 0.006, puffs)
    return high


def sawtooth(
    m: Model,
    u: tuple[float, float],
    v: tuple[float, float],
    high: float,
    teeth: int,
    rise: float,
    roof: RGB,
    glass: RGB,
    wall: RGB,
) -> None:
    """A sawtooth roof over walls `u` by `v` from `high`: `teeth` ridges across it, each sloping
    down towards the front from a strip of glazing along its crest."""
    (u0, u1), (v0, v1) = u, v
    step = (u1 - u0) / teeth
    slant = math.hypot(step, rise) or 1.0
    for k in range(teeth):
        a, b = u0 + step * k, u0 + step * (k + 1)
        m.faces.append(
            Face(
                [(a, v0, high + rise), (b, v0, high), (b, v1, high), (a, v1, high + rise)],
                roof,
                (rise / slant, 0.0, step / slant),
            )
        )
        m.faces.append(
            Face(
                [(a, v0, high), (a, v0, high + rise), (a, v1, high + rise), (a, v1, high)],
                glass,
                (-1.0, 0.0, 0.0),
            )
        )
        for side, out in ((v0, -1.0), (v1, 1.0)):
            m.faces.append(
                Face(
                    [(a, side, high), (b, side, high), (a, side, high + rise)],
                    wall,
                    (0.0, out, 0.0),
                )
            )


def heap(m: Model, x: float, y: float, half: float, height: float, color: RGB) -> None:
    """The painted sprite's heap at (x, y), `half` pixels across and `height` high."""
    u, v = spot(x, y)
    cone(m, u, v, wide(half), tall(height), color)
