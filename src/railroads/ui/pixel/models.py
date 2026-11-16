"""Small models built of boxes, cylinders and plates, drawn as pixel art with the map's own
projection: faces shaded by how much they face the light from the top left, painted back to
front, then outlined in a dark line like the rest of the art. Trains are built of them
(railroads.ui.pixel.trains), and so are station houses, which need drawing turned along the
track.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from railroads.ui.iso_camera import TILT

type RGB = tuple[int, int, int]
type Vec3 = tuple[float, float, float]

# How many map art pixels tall a cell-wide height is, for a view from 30 degrees above.
HEIGHT_PER_WIDTH = math.sqrt(1 - TILT**2)
OUTLINE: RGB = (22, 18, 18)
# Shades a face can take, from facing away from the light to facing it.
SHADES = (0.7, 0.8, 0.9, 1.0, 1.08)
CYLINDER_SIDES = 10
# Whether models built now keep their fine details: those built for the far zooms leave them out.
_KEEP_FINE = [True]


def keeping_fine() -> bool:
    """Whether models built now keep their fine details."""
    return _KEEP_FINE[-1]


@contextmanager
def without_fine() -> Iterator[None]:
    """Builds models within without their fine details, which saves the time of making them."""
    _KEEP_FINE.append(False)
    try:
        yield
    finally:
        _KEEP_FINE.pop()


@dataclass(frozen=True)
class Face:
    """A flat face of a model; a `fine` one is a small detail drawn only when the model is drawn
    big enough to show it, and a `coarse` one stands in for fine ones when it is not."""

    corners: list[Vec3]
    color: RGB
    normal: Vec3
    fine: bool = False
    coarse: bool = False

    def drawn(self, fine: bool) -> bool:
        """Whether the face is drawn when the model is drawn with its fine details or not."""
        return not self.coarse if fine else not self.fine


class Model:
    """A model built of faces, in cells: u along the track (forward positive), v across it (to
    the left), z up from the rails."""

    def __init__(self) -> None:
        self.faces: list[Face] = []
        # Whether faces added now are left out, as fine details of a model built without them.
        self.skipping = False

    @contextmanager
    def fine(self) -> Iterator[None]:
        """Marks every face added within as a fine detail, or leaves them out of a model built
        without_fine."""
        if not _KEEP_FINE[-1]:
            skipping, self.skipping = self.skipping, True
            try:
                yield
            finally:
                self.skipping = skipping
            return
        start = len(self.faces)
        yield
        self.faces[start:] = [replace(face, fine=True) for face in self.faces[start:]]

    @contextmanager
    def coarse(self) -> Iterator[None]:
        """Marks every face added within as standing in for fine details far out."""
        start = len(self.faces)
        yield
        self.faces[start:] = [replace(face, coarse=True) for face in self.faces[start:]]

    def box(
        self, u: tuple[float, float], v: tuple[float, float], z: tuple[float, float], color: RGB
    ) -> None:
        if self.skipping:
            return
        (u0, u1), (v0, v1), (z0, z1) = u, v, z
        self.faces += [
            Face([(u0, v0, z1), (u1, v0, z1), (u1, v1, z1), (u0, v1, z1)], color, (0, 0, 1)),
            Face([(u1, v0, z0), (u1, v1, z0), (u1, v1, z1), (u1, v0, z1)], color, (1, 0, 0)),
            Face([(u0, v0, z0), (u0, v0, z1), (u0, v1, z1), (u0, v1, z0)], color, (-1, 0, 0)),
            Face([(u0, v1, z0), (u0, v1, z1), (u1, v1, z1), (u1, v1, z0)], color, (0, 1, 0)),
            Face([(u0, v0, z0), (u1, v0, z0), (u1, v0, z1), (u0, v0, z1)], color, (0, -1, 0)),
        ]

    def panel(self, u: tuple[float, float], z: tuple[float, float], v: float, color: RGB) -> None:
        """A flat patch on both sides of the model, `v` out from its middle: windows, doors."""
        if self.skipping:
            return
        (u0, u1), (z0, z1) = u, z
        self.faces.append(
            Face([(u0, v, z0), (u1, v, z0), (u1, v, z1), (u0, v, z1)], color, (0, 1, 0))
        )
        self.faces.append(
            Face([(u0, -v, z0), (u0, -v, z1), (u1, -v, z1), (u1, -v, z0)], color, (0, -1, 0))
        )

    def cylinder_along(
        self, u: tuple[float, float], radius: float, z: float, color: RGB, cap: RGB | None = None
    ) -> None:
        """A cylinder lying along the track, its middle `z` above the rails."""
        if self.skipping:
            return
        u0, u1 = u
        ring = [
            (
                radius * math.cos(2 * math.pi * k / CYLINDER_SIDES),
                radius * math.sin(2 * math.pi * k / CYLINDER_SIDES),
            )
            for k in range(CYLINDER_SIDES)
        ]
        for (a, b), (c, d) in zip(ring, ring[1:] + ring[:1], strict=True):
            mid = ((a + c) / 2, (b + d) / 2)
            length = math.hypot(*mid) or 1
            self.faces.append(
                Face(
                    [(u0, a, z + b), (u1, a, z + b), (u1, c, z + d), (u0, c, z + d)],
                    color,
                    (0, mid[0] / length, mid[1] / length),
                )
            )
        end = cap or color
        self.faces.append(Face([(u1, a, z + b) for a, b in ring], end, (1, 0, 0)))
        self.faces.append(Face([(u0, a, z + b) for a, b in reversed(ring)], end, (-1, 0, 0)))

    def cylinder_up(self, u: float, radius: float, z: tuple[float, float], color: RGB) -> None:
        """A cylinder standing up, like a chimney."""
        if self.skipping:
            return
        z0, z1 = z
        ring = [
            (
                u + radius * math.cos(2 * math.pi * k / CYLINDER_SIDES),
                radius * math.sin(2 * math.pi * k / CYLINDER_SIDES),
            )
            for k in range(CYLINDER_SIDES)
        ]
        for (a, b), (c, d) in zip(ring, ring[1:] + ring[:1], strict=True):
            mid = ((a + c) / 2 - u, (b + d) / 2)
            length = math.hypot(*mid) or 1
            self.faces.append(
                Face(
                    [(a, b, z0), (c, d, z0), (c, d, z1), (a, b, z1)],
                    color,
                    (mid[0] / length, mid[1] / length, 0),
                )
            )
        self.faces.append(Face([(a, b, z1) for a, b in ring], color, (0, 0, 1)))

    def slope(
        self, a: tuple[float, float], b: tuple[float, float], v: tuple[float, float], color: RGB
    ) -> None:
        """A flat plate tilted along the track from `a` to `b`, each a (u, z) point, spanning `v`
        across, like a pantograph arm or a windscreen. It shows from either side."""
        if self.skipping:
            return
        (au, az), (bu, bz), (v0, v1) = a, b, v
        du, dz = bu - au, bz - az
        length = math.hypot(du, dz) or 1
        normal = (-dz / length, 0.0, du / length)
        corners = [(au, v0, az), (bu, v0, bz), (bu, v1, bz), (au, v1, az)]
        self.faces.append(Face(corners, color, normal))
        self.faces.append(Face(corners[::-1], color, (-normal[0], 0.0, -normal[2])))

    def nose(
        self, u: tuple[float, float], z: tuple[float, float, float], v: float, color: RGB
    ) -> None:
        """A streamlined end running from `u[0]` out to its tip at `u[1]`, `v` either side of the
        middle: its floor at `z[0]` and its top sloping down from `z[2]` to `z[1]` at the tip."""
        if self.skipping:
            return
        (u0, u1), (z0, low, high) = u, z
        out = 1.0 if u1 > u0 else -1.0
        du, dz = abs(u1 - u0), high - low
        length = math.hypot(du, dz)
        self.faces += [
            Face(
                [(u0, -v, high), (u1, -v, low), (u1, v, low), (u0, v, high)],
                color,
                (out * dz / length, 0.0, du / length),
            ),
            Face([(u1, -v, z0), (u1, v, z0), (u1, v, low), (u1, -v, low)], color, (out, 0.0, 0.0)),
            Face([(u0, v, z0), (u0, v, high), (u1, v, low), (u1, v, z0)], color, (0.0, 1.0, 0.0)),
            Face(
                [(u0, -v, z0), (u1, -v, z0), (u1, -v, low), (u0, -v, high)],
                color,
                (0.0, -1.0, 0.0),
            ),
        ]

    def wheels(self, positions: list[float], radius: float, color: RGB, v: float) -> None:
        if self.skipping:
            return
        for u in positions:
            self.box((u - radius, u + radius), (v - 0.012, v + 0.012), (0.0, 2 * radius), color)
            self.box((u - radius, u + radius), (-v - 0.012, -v + 0.012), (0.0, 2 * radius), color)

    def roof(
        self,
        u: tuple[float, float],
        v: tuple[float, float],
        z: tuple[float, float],
        color: RGB,
        ends: RGB,
    ) -> None:
        """A pitched roof over `u` by `v`, its eaves at `z[0]` along both long sides and its
        ridge at `z[1]` along the middle, with a gable in `ends` at each end."""
        if self.skipping:
            return
        (u0, u1), (v0, v1), (eaves, ridge) = u, v, z
        middle, rise, half = (v0 + v1) / 2, ridge - eaves, (v1 - v0) / 2
        slant = math.hypot(rise, half) or 1.0
        self.faces += [
            Face(
                [(u0, v1, eaves), (u1, v1, eaves), (u1, middle, ridge), (u0, middle, ridge)],
                color,
                (0.0, rise / slant, half / slant),
            ),
            Face(
                [(u0, middle, ridge), (u1, middle, ridge), (u1, v0, eaves), (u0, v0, eaves)],
                color,
                (0.0, -rise / slant, half / slant),
            ),
            Face([(u1, v0, eaves), (u1, v1, eaves), (u1, middle, ridge)], ends, (1, 0, 0)),
            Face([(u0, v1, eaves), (u0, v0, eaves), (u0, middle, ridge)], ends, (-1, 0, 0)),
        ]

    def spire(self, u: float, half: float, z: tuple[float, float], color: RGB) -> None:
        """A four-sided pointed roof over a square `half` either way of (`u`, 0), rising from
        `z[0]` to its tip at `z[1]`."""
        if self.skipping:
            return
        base, tip = z
        rise = tip - base
        slant = math.hypot(rise, half) or 1.0
        apex = (u, 0.0, tip)
        a, b = rise / slant, half / slant
        self.faces += [
            Face([(u - half, half, base), (u + half, half, base), apex], color, (0, a, b)),
            Face([(u + half, -half, base), (u - half, -half, base), apex], color, (0, -a, b)),
            Face([(u + half, half, base), (u + half, -half, base), apex], color, (a, 0, b)),
            Face([(u - half, -half, base), (u - half, half, base), apex], color, (-a, 0, b)),
        ]


# How far out of a flat face a fine detail may lie and still count as lying on it.
ON_FACE = 0.006


def decals(faces: list[Face]) -> list[tuple[int, int]]:
    """Each fine detail lying flat on a bigger face, square to an axis, and the face it lies on:
    it is painted straight after that face, however far back the middles of the two lie, so it
    is never hidden under it. Each comes after the face it lies on if that is one of them."""
    flat: dict[tuple[int, float], list[tuple[int, float, npt.NDArray[np.float64]]]] = {}
    for index, face in enumerate(faces):
        axes = [k for k in range(3) if abs(face.normal[k]) > 1 - 1e-9]
        if not axes:
            continue
        axis = axes[0]
        corners = np.array(face.corners, dtype=np.float64)
        if np.ptp(corners[:, axis]) > 1e-9:
            continue
        side = float(np.sign(face.normal[axis]))
        others = [k for k in range(3) if k != axis]
        box = np.concatenate([corners[:, others].min(axis=0), corners[:, others].max(axis=0)])
        flat.setdefault((axis, side), []).append((index, side * float(corners[0, axis]), box))
    found = []
    for group in flat.values():
        for index, out, box in group:
            if not faces[index].fine:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            best = None
            for other, under, around in group:
                if other == index or not 0 <= out - under <= ON_FACE:
                    continue
                if (
                    not (around[:2] <= box[:2] + 1e-6).all()
                    or not (around[2:] >= box[2:] - 1e-6).all()
                ):
                    continue
                size = (around[2] - around[0]) * (around[3] - around[1])
                if under == out and size <= area:
                    continue
                if best is None or (under, -size) > (best[1], -best[2]):
                    best = (other, under, size)
            if best is not None:
                found.append((out, -area, index, best[0]))
    return [(index, under) for _, _, index, under in sorted(found)]


def shade(color: RGB, factor: float) -> RGB:
    r, g, b = color
    return (min(255, round(r * factor)), min(255, round(g * factor)), min(255, round(b * factor)))


def draw_model(
    model: Model,
    heading: float,
    matrix: npt.NDArray[np.float64],
    pixels: float = 1.0,
    fine: bool = False,
) -> tuple[Image.Image, tuple[int, int]]:
    """The model seen turned `heading` radians round from the map's x axis, with the map's
    plane-to-art-pixel matrix, `pixels` picture pixels for every art pixel a model cell would
    take, with its fine details if `fine`: shaded, outlined, and where its origin lands in the
    picture."""
    scale = matrix * pixels
    cos, sin = math.cos(heading), math.sin(heading)
    up = float(np.hypot(*scale[:, 0])) * HEIGHT_PER_WIDTH
    # The way to the viewer in the plane: what moves straight down the screen.
    towards = np.linalg.solve(matrix, np.array([0.0, 1.0]))
    towards = towards / np.hypot(*towards)
    view = np.array([towards[0] * TILT, towards[1] * TILT, HEIGHT_PER_WIDTH])
    light_plane = np.linalg.solve(matrix, np.array([-1.0, -1.0]))
    light_plane = light_plane / np.hypot(*light_plane)
    light = np.array([*light_plane * 0.6, 0.8])

    def world(p: Vec3) -> tuple[float, float, float]:
        u, v, z = p
        return u * cos - v * sin, u * sin + v * cos, z

    def to_screen(p: tuple[float, float, float]) -> tuple[float, float]:
        x, y, z = p
        sx, sy = scale @ np.array([x, y])
        return float(sx), float(sy - z * up)

    faces = [face for face in model.faces if face.drawn(fine)]
    depths = [float(np.mean([np.array(world(c)) @ view for c in face.corners])) for face in faces]
    for index, under in decals(faces) if fine else []:
        depths[index] = depths[under] + 1e-6
    drawn = []
    for face, depth in zip(faces, depths, strict=True):
        normal = np.array(world(face.normal))
        if normal @ view <= 1e-6:
            continue
        points = [world(c) for c in face.corners]
        lit = float(np.clip(normal @ light, 0, 1))
        level = SHADES[min(len(SHADES) - 1, int(lit * len(SHADES)))]
        drawn.append((depth, [to_screen(p) for p in points], shade(face.color, level)))
    drawn.sort(key=lambda item: item[0])
    xs = [x for _, pts, _ in drawn for x, _ in pts]
    ys = [y for _, pts, _ in drawn for _, y in pts]
    pad = max(2, round(2 * pixels))
    left, top = math.floor(min(xs)) - pad, math.floor(min(ys)) - pad
    edge = pad + max(1, round(pixels))
    width, height = math.ceil(max(xs)) - left + edge, math.ceil(max(ys)) - top + edge
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    paint = ImageDraw.Draw(image)
    for _, points, color in drawn:
        paint.polygon([(x - left, y - top) for x, y in points], fill=(*color, 255))
    _outline(image)
    return image, (-left, -top)


def _outline(image: Image.Image, width: int = 1) -> None:
    """Rings what is drawn in a dark line `width` pixels wide."""
    pixels = np.asarray(image).copy()
    solid = pixels[..., 3] > 0
    grown = solid
    for _ in range(width):
        around = grown.copy()
        around[1:] |= grown[:-1]
        around[:-1] |= grown[1:]
        around[:, 1:] |= grown[:, :-1]
        around[:, :-1] |= grown[:, 1:]
        grown = around
    edge = grown & ~solid
    pixels[edge] = (*OUTLINE, 255)
    image.paste(Image.fromarray(pixels))
