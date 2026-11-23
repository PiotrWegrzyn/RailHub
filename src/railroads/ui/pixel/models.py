"""Small models built of boxes, cylinders and plates, drawn as pixel art with the map's own
projection: faces shaded by how much they face the light from the top left, painted back to
front, then outlined in a dark line like the rest of the art. Trains are built of them
(railroads.ui.pixel.trains), and so are station houses, which need drawing turned along the
track.
"""

import itertools
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from railroads.ui.iso_camera import TILT
from railroads.ui.pixel.projection import RGB

type Vec3 = tuple[float, float, float]

# How many map art pixels tall a cell-wide height is, for a view from 30 degrees above.
HEIGHT_PER_WIDTH = math.sqrt(1 - TILT**2)
OUTLINE: RGB = (22, 18, 18)
TYRE: RGB = (112, 112, 118)
# Shades a face can take, from facing away from the light to facing it.
SHADES = (0.7, 0.8, 0.9, 1.0, 1.08)
CYLINDER_SIDES = 10
# Drawn fine, close up: cylinders get more sides, so they look round, and faces take any of
# FINE_SHADES shades from FINE_DARK facing away from the light to FINE_LIGHT facing it, with a
# SHINE where they face halfway between the light and the viewer, and faces low down LOW_DARK
# darker, in the shade of the body above, fading out by LOW_HEIGHT above the rails.
FINE_CYLINDER_SIDES = 24
# How much nearer a face drawn later may be counted than one drawn before it at the same pixel,
# in cells: faces meeting at an edge, or lying flat on one another, show the later one.
DEPTH_SLACK = 1e-4
# How far up close a body's top edges are rounded over, and in how many steps.
BODY_ROUND = 0.014
BODY_ARC = 3
FINE_SHADES = 24
FINE_DARK, FINE_LIGHT = 0.62, 1.1
SHINE, SHINE_POWER = 0.4, 24
LOW_DARK, LOW_HEIGHT = 0.25, 0.08
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
    big enough to show it, a `finest` one only when it is drawn bigger still, at the closest
    zooms, and a `coarse` one stands in for fine ones when it is not."""

    corners: list[Vec3]
    color: RGB
    normal: Vec3
    fine: bool = False
    coarse: bool = False
    finest: bool = False

    def drawn(self, fine: bool, finest: bool = False) -> bool:
        """Whether the face is drawn when the model is drawn with its fine details or not, and
        with its finest ones or not."""
        if self.finest and not finest:
            return False
        return not self.coarse if fine else not self.fine


class Model:
    """A model built of faces, in cells: u along the track (forward positive), v across it (to
    the left), z up from the rails."""

    def __init__(self) -> None:
        self.faces: list[Face] = []
        # Whether faces added now are left out, as fine details of a model built without them.
        self.skipping = False
        # How many fine blocks the faces added now are within.
        self.within_fine = 0
        # The footprints, u by v, of what stands as buildings, where they cast their shadow.
        self.footprints: list[tuple[tuple[float, float], tuple[float, float]]] = []

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
        self.within_fine += 1
        try:
            yield
        finally:
            self.within_fine -= 1
        self.faces[start:] = [replace(face, fine=True) for face in self.faces[start:]]

    @contextmanager
    def finest(self) -> Iterator[None]:
        """Marks every face added within as a fine detail drawn only at the closest zooms."""
        start = len(self.faces)
        with self.fine():
            yield
        self.faces[start:] = [replace(face, finest=True) for face in self.faces[start:]]

    @contextmanager
    def coarse(self) -> Iterator[None]:
        """Marks every face added within as standing in for fine details far out."""
        start = len(self.faces)
        yield
        self.faces[start:] = [replace(face, coarse=True) for face in self.faces[start:]]

    def rounded(self, build: Callable[[int], None]) -> None:
        """Builds a round part with `build` given how many sides to give it: CYLINDER_SIDES for
        far out, and when the model keeps fine details, once more at FINE_CYLINDER_SIDES drawn
        only close up."""
        if self.within_fine:
            build(FINE_CYLINDER_SIDES)
            return
        if not _KEEP_FINE[-1]:
            with self.coarse():
                build(CYLINDER_SIDES)
            return
        with self.coarse():
            build(CYLINDER_SIDES)
        with self.fine():
            build(FINE_CYLINDER_SIDES)

    def move(self, start: int, u: float, v: float) -> None:
        """Moves the faces added since there were `start` of them `u` along and `v` across."""
        self.faces[start:] = [
            replace(face, corners=[(a + u, b + v, c) for a, b, c in face.corners])
            for face in self.faces[start:]
        ]

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

    def body(
        self,
        u: tuple[float, float],
        v: tuple[float, float],
        z: tuple[float, float],
        color: RGB,
        round: float = BODY_ROUND,
    ) -> None:
        """A box far out; up close the same box with its top edges rounded `round` over and its
        upright corners and ends' top edges cut back a little, like a pressed or planked body."""
        if self.skipping:
            return
        with self.coarse():
            self.box(u, v, z, color)
        if not _KEEP_FINE[-1]:
            return
        (u0, u1), (v0, v1), (z0, z1) = u, v, z
        r = min(round, (v1 - v0) / 3, (z1 - z0) / 3)
        cut = min(r * 0.6, (u1 - u0) / 6)
        middle = ((u0 + u1) / 2, (v0 + v1) / 2, (z0 + z1) / 2)
        sections = [
            (u0, _rounded_section(v0 + cut, v1 - cut, z0, z1 - cut, r)),
            (u0 + cut, _rounded_section(v0, v1, z0, z1, r)),
            (u1 - cut, _rounded_section(v0, v1, z0, z1, r)),
            (u1, _rounded_section(v0 + cut, v1 - cut, z0, z1 - cut, r)),
        ]
        with self.fine():
            for (ua, pa), (ub, pb) in itertools.pairwise(sections):
                for (va0, za0), (va1, za1), (vb0, zb0), (vb1, zb1) in zip(
                    pa, pa[1:], pb, pb[1:], strict=False
                ):
                    corners = [(ua, va0, za0), (ua, va1, za1), (ub, vb1, zb1), (ub, vb0, zb0)]
                    self.faces.append(Face(corners, color, _normal(corners, middle)))
            for end, out in ((u0, -1.0), (u1, 1.0)):
                points = next(points for at, points in sections if at == end)
                self.faces.append(Face([(end, a, b) for a, b in points], color, (out, 0.0, 0.0)))

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
        if not self.skipping:
            self.rounded(lambda sides: self._cylinder_along(u, radius, z, color, cap, sides))

    def _cylinder_along(
        self,
        u: tuple[float, float],
        radius: float,
        z: float,
        color: RGB,
        cap: RGB | None,
        sides: int,
    ) -> None:
        u0, u1 = u
        ring = [
            (
                radius * math.cos(2 * math.pi * k / sides),
                radius * math.sin(2 * math.pi * k / sides),
            )
            for k in range(sides)
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

    def cylinder_across(
        self, u: float, radius: float, z: float, v: tuple[float, float], color: RGB, cap: RGB
    ) -> None:
        """A cylinder lying across the track, like a wheel, its middle `z` above the rails at
        `u`, from `v[0]` to `v[1]`."""
        if not self.skipping:
            self.rounded(lambda sides: self._cylinder_across(u, radius, z, v, color, cap, sides))

    def _cylinder_across(
        self,
        u: float,
        radius: float,
        z: float,
        v: tuple[float, float],
        color: RGB,
        cap: RGB,
        sides: int,
    ) -> None:
        v0, v1 = v
        ring = [
            (
                u + radius * math.cos(2 * math.pi * k / sides),
                z + radius * math.sin(2 * math.pi * k / sides),
            )
            for k in range(sides)
        ]
        for (a, b), (c, d) in zip(ring, ring[1:] + ring[:1], strict=True):
            mid = ((a + c) / 2 - u, (b + d) / 2 - z)
            length = math.hypot(*mid) or 1
            self.faces.append(
                Face(
                    [(a, v0, b), (c, v0, d), (c, v1, d), (a, v1, b)],
                    color,
                    (mid[0] / length, 0, mid[1] / length),
                )
            )
        self.faces.append(Face([(a, v1, b) for a, b in ring], cap, (0, 1, 0)))
        self.faces.append(Face([(a, v0, b) for a, b in reversed(ring)], cap, (0, -1, 0)))

    def cylinder_up(self, u: float, radius: float, z: tuple[float, float], color: RGB) -> None:
        """A cylinder standing up, like a chimney."""
        if not self.skipping:
            self.rounded(lambda sides: self._cylinder_up(u, radius, z, color, sides))

    def _cylinder_up(
        self, u: float, radius: float, z: tuple[float, float], color: RGB, sides: int
    ) -> None:
        z0, z1 = z
        ring = [
            (
                u + radius * math.cos(2 * math.pi * k / sides),
                radius * math.sin(2 * math.pi * k / sides),
            )
            for k in range(sides)
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
        """Wheels on both sides `v` out, square blocks far out and round with a steel tyre close
        up."""
        if self.skipping:
            return
        with self.coarse():
            for u in positions:
                for side in (v, -v):
                    self.box(
                        (u - radius, u + radius),
                        (side - 0.012, side + 0.012),
                        (0.0, 2 * radius),
                        color,
                    )
        if not _KEEP_FINE[-1]:
            return
        with self.fine():
            for u in positions:
                for side in (v, -v):
                    self.cylinder_across(
                        u, radius, radius, (side - 0.012, side + 0.012), TYRE, color
                    )

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

    def snout(
        self,
        u: tuple[float, float],
        z: tuple[float, float, float],
        v: float,
        color: RGB,
        glass: RGB | None = None,
    ) -> None:
        """A rounded streamlined end, the smooth form of `nose`: from `u[0]` out to its tip at
        `u[1]`, `v` either side of the middle at its root and narrowing a little, its floor at
        `z[0]` and its top curving down from `z[2]` at the root to `z[1]` at the tip, its upper
        edges rounded across; with a windscreen of `glass` over the top near the root."""
        if self.skipping:
            return
        (u0, u1), (floor, tip, top) = u, z
        steps, arc = 8, 6
        sections = []
        for k in range(steps + 1):
            t = k / steps
            high = tip + (top - tip) * math.sqrt(max(0.0, 1 - t**2))
            half = v * (1 - 0.5 * t**2)
            shoulder = floor + (high - floor) * 0.55
            rise = high - shoulder
            profile = [(half, floor), (half, shoulder)]
            profile += [
                (half * math.cos(math.pi * j / arc), shoulder + rise * math.sin(math.pi * j / arc))
                for j in range(1, arc)
            ]
            profile += [(-half, shoulder), (-half, floor)]
            sections.append((u0 + (u1 - u0) * t, profile))
        self.loft(sections, color)
        if glass is None:
            return
        screen = [
            (at, [(a, b + 0.003) for a, b in profile[2 : 2 + arc - 1]])
            for at, profile in sections[1:4]
        ]
        self.loft(screen, glass)
        self.faces.pop()

    def loft(self, sections: list[tuple[float, list[tuple[float, float]]]], color: RGB) -> None:
        """A skin through cross sections along the track, each a `u` and the same number of
        (v, z) points round it, closed off at the last section."""
        if self.skipping:
            return
        for (ua, pa), (ub, pb) in itertools.pairwise(sections):
            middle = sum(z for _, z in pa + pb) / len(pa + pb)
            for (va0, za0), (va1, za1), (vb0, zb0), (vb1, zb1) in zip(
                pa, pa[1:], pb, pb[1:], strict=False
            ):
                corners = [(ua, va0, za0), (ua, va1, za1), (ub, vb1, zb1), (ub, vb0, zb0)]
                normal = _normal(corners, ((ua + ub) / 2, 0.0, middle))
                self.faces.append(Face(corners, color, normal))
        last_u, last = sections[-1]
        out = 1.0 if last_u > sections[0][0] else -1.0
        self.faces.append(Face([(last_u, a, b) for a, b in last], color, (out, 0.0, 0.0)))

    def cap(self, u: float, radius: float, z: tuple[float, float], color: RGB) -> None:
        """A rounded top standing on a circle `radius` round (`u`, 0) at `z[0]`, rising to
        `z[1]` in the middle, like the top of a dome."""
        if not self.skipping:
            self.rounded(lambda sides: self._cap(u, radius, z, color, sides))

    def _cap(self, u: float, radius: float, z: tuple[float, float], color: RGB, sides: int) -> None:
        base, tip = z
        rise = tip - base
        bands = max(2, sides // 6)
        rings = [
            (
                radius * math.cos(math.pi / 2 * k / bands),
                base + rise * math.sin(math.pi / 2 * k / bands),
            )
            for k in range(bands + 1)
        ]
        for (r0, z0), (r1, z1) in itertools.pairwise(rings):
            middle = math.pi / 2 * (rings.index((r0, z0)) + 0.5) / bands
            up, out = math.sin(middle) / rise, math.cos(middle) / radius
            for k in range(sides):
                a0, a1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
                mid = (a0 + a1) / 2
                normal = (out * math.cos(mid), out * math.sin(mid), up)
                length = math.hypot(*normal)
                corners = [
                    (u + r0 * math.cos(a0), r0 * math.sin(a0), z0),
                    (u + r0 * math.cos(a1), r0 * math.sin(a1), z0),
                    (u + r1 * math.cos(a1), r1 * math.sin(a1), z1),
                    (u + r1 * math.cos(a0), r1 * math.sin(a0), z1),
                ]
                self.faces.append(Face(corners, color, tuple(n / length for n in normal)))

    def vault(
        self, u: tuple[float, float], half: float, z: tuple[float, float], color: RGB
    ) -> None:
        """A roof arched across the track over `u`, `half` either side of the middle: its edges
        at `z[0]` and its crown at `z[1]`, with a thin rim under the edges and an arched end at
        each end."""
        if not self.skipping:
            self.rounded(lambda sides: self._vault(u, half, z, color, sides))

    def _vault(
        self, u: tuple[float, float], half: float, z: tuple[float, float], color: RGB, sides: int
    ) -> None:
        (u0, u1), (eaves, crown) = u, z
        rise = crown - eaves
        steps = max(3, sides // 2)
        arc = [
            (
                half * math.cos(math.pi * k / steps),
                eaves + rise * math.sin(math.pi * k / steps),
            )
            for k in range(steps + 1)
        ]
        for (a, b), (c, d) in itertools.pairwise(arc):
            mid = ((a + c) / 2 / half, ((b + d) / 2 - eaves) / rise)
            length = math.hypot(*mid) or 1
            self.faces.append(
                Face(
                    [(u0, a, b), (u0, c, d), (u1, c, d), (u1, a, b)],
                    color,
                    (0, mid[0] / length, mid[1] / length),
                )
            )
        rim = eaves - 0.006
        for side in (half, -half):
            self.faces.append(
                Face(
                    [(u0, side, rim), (u1, side, rim), (u1, side, eaves), (u0, side, eaves)],
                    color,
                    (0, 1.0 if side > 0 else -1.0, 0),
                )
            )
        end = [(a, b) for a, b in reversed(arc)] + [(-half, rim), (half, rim)]
        self.faces.append(Face([(u1, a, b) for a, b in end], color, (1, 0, 0)))
        self.faces.append(Face([(u0, a, b) for a, b in reversed(end)], color, (-1, 0, 0)))

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


def _rounded_section(
    v0: float, v1: float, z0: float, z1: float, r: float
) -> list[tuple[float, float]]:
    """A box's cross-section as (v, z) points from its bottom left corner up over its rounded
    top and down to its bottom right corner."""
    points = [(v0, z0)]
    for centre_v, start in ((v0 + r, math.pi), (v1 - r, math.pi / 2)):
        for k in range(BODY_ARC + 1):
            angle = start - math.pi / 2 * k / BODY_ARC
            points.append((centre_v + r * math.cos(angle), z1 - r + r * math.sin(angle)))
    points.append((v1, z0))
    return points


def _normal(corners: list[Vec3], inside: Vec3) -> Vec3:
    """The way out of a flat face, away from a point inside the shape it bounds."""
    a, b, d = (np.array(c, dtype=np.float64) for c in (corners[0], corners[1], corners[3]))
    normal = np.cross(b - a, d - a)
    length = float(np.linalg.norm(normal))
    if length == 0:
        return (0.0, 0.0, 1.0)
    normal = normal / length
    centre = np.mean(np.array(corners, dtype=np.float64), axis=0)
    if float(normal @ (centre - np.array(inside))) < 0:
        normal = -normal
    return (float(normal[0]), float(normal[1]), float(normal[2]))


# How far out of a flat face a fine detail may lie and still count as lying on it.
ON_FACE = 0.006


def decals(faces: list[Face]) -> list[tuple[int, int]]:
    """Each fine detail lying flat on a bigger face, square to an axis, and the face it lies on:
    it is painted straight after that face, however far back the middles of the two lie, so it
    is never hidden under it. Each comes after the face it lies on if that is one of them."""
    flat: dict[tuple[int, float], list[tuple[int, float, list[float]]]] = {}
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
        box = [*corners[:, others].min(axis=0).tolist(), *corners[:, others].max(axis=0).tolist()]
        flat.setdefault((axis, side), []).append((index, side * float(corners[0, axis]), box))
    found = []
    for group in flat.values():
        outs = np.array([out for _, out, _ in group])
        by_out = np.argsort(outs, kind="stable")
        sorted_outs = outs[by_out]
        for index, out, box in group:
            if not faces[index].fine:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            best = None
            low = int(np.searchsorted(sorted_outs, out - ON_FACE - 1e-12, "left"))
            high = int(np.searchsorted(sorted_outs, out, "right"))
            for position in sorted(by_out[low:high].tolist()):
                other, under, around = group[position]
                if other == index or not 0 <= out - under <= ON_FACE:
                    continue
                if (
                    around[0] > box[0] + 1e-6
                    or around[1] > box[1] + 1e-6
                    or around[2] < box[2] - 1e-6
                    or around[3] < box[3] - 1e-6
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


def fine_shades(
    colors: npt.NDArray[np.float64],
    lit: npt.NDArray[np.float64],
    shine: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
) -> list[list[int]]:
    """Faces' colours shaded close up: by how much each faces the light (`lit`, 0 to 1) and the
    way halfway between the light and the viewer (`shine`), and how high its middle is (`low`)."""
    level = np.floor(lit * (FINE_SHADES - 1) + 0.5) / (FINE_SHADES - 1)
    factor = FINE_DARK + (FINE_LIGHT - FINE_DARK) * level
    factor = factor * (1 - LOW_DARK * np.clip(1 - low / LOW_HEIGHT, 0, 1))
    glint = SHINE * np.clip(shine, 0, 1) ** SHINE_POWER
    rgb = np.clip(np.rint(colors * factor[:, None] + 255 * glint[:, None]), 0, 255)
    return rgb.astype(np.int64).tolist()


def halfway(light: npt.NDArray[np.float64], view: npt.NDArray[np.float64]):
    """The way halfway between the light and the viewer, where faces shine."""
    way = light + view
    return way / np.linalg.norm(way)


def draw_model(
    model: Model,
    heading: float,
    matrix: npt.NDArray[np.float64],
    pixels: float = 1.0,
    fine: bool = False,
    light_from: tuple[float, float] = (-1.0, -1.0),
    depth_test: bool = False,
    outline: bool = True,
) -> tuple[Image.Image, tuple[int, int]]:
    """The model seen turned `heading` radians round from the map's x axis, with the map's
    plane-to-art-pixel matrix, `pixels` picture pixels for every art pixel a model cell would
    take, with its fine details if `fine`, lit from the way `light_from` on the screen: shaded,
    outlined, and where its origin lands in the picture. Faces are drawn back to front; with
    `depth_test` each pixel shows whichever face is nearest there, so a part standing out of a
    big wall, like a cornice or an awning, shows in front of it. Without `outline` it is left
    without its dark rim, as patches lying on the ground are."""
    scale = matrix * pixels
    cos, sin = math.cos(heading), math.sin(heading)
    up = float(np.hypot(*scale[:, 0])) * HEIGHT_PER_WIDTH
    # The way to the viewer in the plane: what moves straight down the screen.
    towards = np.linalg.solve(matrix, np.array([0.0, 1.0]))
    towards = towards / np.hypot(*towards)
    view = np.array([towards[0] * TILT, towards[1] * TILT, HEIGHT_PER_WIDTH])
    light_plane = np.linalg.solve(matrix, np.array(light_from))
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
    shine_way = halfway(light, view)
    drawn = []
    for face, depth in zip(faces, depths, strict=True):
        normal = np.array(world(face.normal))
        if normal @ view <= 1e-6:
            continue
        points = [world(c) for c in face.corners]
        lit = float(np.clip(normal @ light, 0, 1))
        if fine:
            low = float(np.mean([c[2] for c in face.corners]))
            (rgb,) = fine_shades(
                np.array([face.color], dtype=np.float64),
                np.array([lit]),
                np.array([float(normal @ shine_way)]),
                np.array([low]),
            )
            color = (rgb[0], rgb[1], rgb[2])
        else:
            color = shade(face.color, SHADES[min(len(SHADES) - 1, int(lit * len(SHADES)))])
        near = [float(np.array(p) @ view) for p in points]
        drawn.append((depth, [to_screen(p) for p in points], color, near))
    drawn.sort(key=lambda item: item[0])
    xs = [x for _, pts, _, _ in drawn for x, _ in pts]
    ys = [y for _, pts, _, _ in drawn for _, y in pts]
    pad = max(2, round(2 * pixels))
    left, top = math.floor(min(xs)) - pad, math.floor(min(ys)) - pad
    edge = pad + max(1, round(pixels))
    width, height = math.ceil(max(xs)) - left + edge, math.ceil(max(ys)) - top + edge
    if depth_test:
        image = _depth_tested(
            [
                ([(x - left, y - top) for x, y in points], color, near)
                for _, points, color, near in drawn
            ],
            (width, height),
        )
    else:
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        paint = ImageDraw.Draw(image)
        for _, points, color, _ in drawn:
            paint.polygon([(x - left, y - top) for x, y in points], fill=(*color, 255))
    if outline:
        outline_drawing(image)
    return image, (-left, -top)


def _depth_tested(
    faces: list[tuple[list[tuple[float, float]], RGB, list[float]]], size: tuple[int, int]
) -> Image.Image:
    """Faces, each its corners on the picture, its colour and how near the viewer each corner
    is, drawn so each pixel shows the nearest face there: how near is worked out over each face
    from its corners, as it is flat."""
    width, height = size
    nearest = np.full((height, width), -np.inf)
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    for points, color, near in faces:
        xs = np.array([x for x, _ in points])
        ys = np.array([y for _, y in points])
        x0, y0 = max(0, math.floor(xs.min())), max(0, math.floor(ys.min()))
        x1, y1 = min(width - 1, math.ceil(xs.max())), min(height - 1, math.ceil(ys.max()))
        if x1 < x0 or y1 < y0:
            continue
        mask_image = Image.new("L", (x1 - x0 + 1, y1 - y0 + 1), 0)
        ImageDraw.Draw(mask_image).polygon([(x - x0, y - y0) for x, y in points], fill=1)
        mask = np.asarray(mask_image, dtype=bool)
        if not mask.any():
            continue
        plane, *_ = np.linalg.lstsq(
            np.column_stack([xs, ys, np.ones(len(points))]), np.array(near), rcond=None
        )
        grid_y, grid_x = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        depth = plane[0] * (grid_x + 0.5) + plane[1] * (grid_y + 0.5) + plane[2]
        region = nearest[y0 : y1 + 1, x0 : x1 + 1]
        wins = mask & (depth >= region - DEPTH_SLACK)
        region[wins] = depth[wins]
        pixels[y0 : y1 + 1, x0 : x1 + 1][wins] = (*color, 255)
    return Image.fromarray(pixels)


def outline_drawing(image: Image.Image, width: int = 1) -> None:
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
