"""Buildings built as small 3D models (railroads.ui.pixel.models) and drawn from them, shaded by
the light like the trains and the stations on hexes, instead of painted pixel by pixel: walls on
a plinth, pitched or flat roofs, windows, doors, chimneys and cornices. A model is drawn at
whatever fineness the sheet asks (railroads.ui.pixel.art.painter.drawn_at), with its fine details
once it is drawn big enough to show them, so the closer the zoom the more it shows.

Models are in cells, u and v across the ground and z up, seen from the hex map's look turned
HEADING, so their two front walls face the viewer as the painted buildings' do."""

import math
from collections.abc import Callable
from dataclasses import replace

from PIL import Image, ImageDraw

from railroads.sim.grids import HexGrid
from railroads.ui.pixel import train_parts as parts
from railroads.ui.pixel.art.painter import RGB, SHADOW, Sprite, fineness, shade
from railroads.ui.pixel.models import Model, draw_model
from railroads.ui.pixel.projection import PixelProjection

# The hex map's look (railroads.ui.map_view.LOOKS): not turned, a cell wide per cell.
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

SOIL: RGB = (74, 64, 46)
SILL: RGB = (226, 222, 210)
FRAME: RGB = (238, 236, 228)
DOOR: RGB = (96, 64, 44)
STEP: RGB = (206, 200, 188)


def modelled(
    build: Callable[[], Model], across: float = 1.0, up: float = 1.0, shadow: float = 1.0
) -> Callable[[], Sprite]:
    """The draw function of a building built by `build`: its model stretched `across` the ground
    and `up`, drawn at the fineness the sheet asks, standing in a soft shadow `shadow` times as
    wide as its footprint. Stretching lets a model be built in handy sizes and match the size and
    tall look of the painted buildings beside it."""

    def draw() -> Sprite:
        pixels = fineness()
        model = _stretched(build(), across, up)
        image, anchor = draw_model(
            model,
            HEADING,
            PROJECTION.matrix,
            pixels,
            fine=pixels >= FINE_FROM,
            light_from=LIGHT_FROM,
            depth_test=True,
        )
        return _shadowed(Sprite(image, anchor), model, pixels * shadow)

    return draw


def _stretched(model: Model, across: float, up: float) -> Model:
    """The model `across` times as wide and long and `up` times as tall, its faces still facing
    the right way."""
    if across == 1.0 and up == 1.0:
        return model
    stretched = Model()
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
    us = [c[0] for face in model.faces for c in face.corners if c[2] < 0.002]
    vs = [c[1] for face in model.faces for c in face.corners if c[2] < 0.002]
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


def flat_roof(
    m: Model, u: tuple[float, float], v: tuple[float, float], height: float, color: RGB, wall: RGB
) -> None:
    """A flat roof at `height` behind a low parapet in the wall's colour."""
    (u0, u1), (v0, v1) = u, v
    m.box(u, v, (height, height + 0.004), color)
    for side in ((v0, v0 + 0.006), (v1 - 0.006, v1)):
        m.box(u, side, (height, height + 0.014), shade(wall, 0.95))
    for side in ((u0, u0 + 0.006), (u1 - 0.006, u1)):
        m.box(side, v, (height, height + 0.014), shade(wall, 0.95))


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


def end_window(
    m: Model,
    end: float,
    v: tuple[float, float],
    z: tuple[float, float],
    glass: RGB,
    frame: RGB = FRAME,
) -> None:
    """A window on the end wall at `end`, with a frame, cross bar and sill up close."""
    out = 1.0 if end > 0 else -1.0
    parts.front_plate(m, end + out * 0.0006, v, z, glass)
    if not parts.keeping_fine():
        return
    (v0, v1), (z0, z1) = v, z
    with m.fine():
        at = end + out * 0.0012
        parts.front_plate(m, at, (v0 - 0.003, v1 + 0.003), (z0 - 0.004, z0), SILL)
        mid_v, mid_z = (v0 + v1) / 2, (z0 + z1) / 2
        parts.front_plate(m, at, (mid_v - 0.0012, mid_v + 0.0012), z, frame)
        parts.front_plate(m, at, v, (mid_z - 0.0012, mid_z + 0.0012), frame)


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
        m.cylinder_up(u, half * 0.45, (top + 0.005, top + 0.012), (150, 84, 58))


def cornice(
    m: Model, u: tuple[float, float], v: tuple[float, float], height: float, color: RGB
) -> None:
    """A thin ledge round the walls at `height`, standing out a little."""
    (u0, u1), (v0, v1) = u, v
    m.box((u0 - 0.003, u1 + 0.003), (v0 - 0.003, v1 + 0.003), (height, height + 0.005), color)
