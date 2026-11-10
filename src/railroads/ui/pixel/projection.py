"""Where map points land in the big picture of the pixel map, in art pixels, and where the picture's chunks sit on screen."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import Grid
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, TILT

CHUNK = 256


# Room left above the map's top corner for the tallest sprites, in art pixels.
TOP_MARGIN = 48


type RGB = tuple[int, int, int]


type Point = tuple[float, float]


type Path = list[Point]


type ToArt = Callable[[Path], Path]


FloatArray = npt.NDArray[np.float64]


IntArray = npt.NDArray[np.int64]


Rgba = npt.NDArray[np.uint8]


def hash01(*keys: npt.ArrayLike) -> FloatArray:
    """A repeatable pseudo-random number from 0 to 1 for each combination of integer keys."""
    h = np.uint64(0x9E3779B97F4A7C15)
    for key in keys:
        h = h ^ np.asarray(key).astype(np.uint64)
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        h = h ^ (h >> np.uint64(31))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


@dataclass(frozen=True)
class PixelProjection:
    """Where map points land in the big picture, in art pixels from its top-left corner."""

    matrix: FloatArray
    origin: tuple[float, float]
    size: tuple[int, int]

    @classmethod
    def of(cls, grid: Grid, angle_deg: float, cell_units: float) -> Self:
        scale = PIXEL_CELL_WIDTH / cell_units
        cos, sin = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        matrix = np.array([[cos, -sin], [sin * TILT, cos * TILT]]) * scale
        width, height = grid.size()
        corners = matrix @ np.array([[0, width, width, 0], [0, 0, height, height]])
        left, top = corners.min(axis=1) - (0, TOP_MARGIN)
        right, bottom = corners.max(axis=1)
        origin = (math.floor(left), math.floor(top))
        return cls(matrix, origin, (math.ceil(right) - origin[0], math.ceil(bottom) - origin[1]))

    def to_pixels(self, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        (a, b), (c, d) = self.matrix
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return a * x + b * y - self.origin[0], c * x + d * y - self.origin[1]

    def to_plane(self, px: npt.ArrayLike, py: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        (a, b), (c, d) = np.linalg.inv(self.matrix)
        px = np.asarray(px, dtype=np.float64) + self.origin[0]
        py = np.asarray(py, dtype=np.float64) + self.origin[1]
        return a * px + b * py, c * px + d * py

    @property
    def chunks(self) -> tuple[int, int]:
        """How many chunks across and down the picture takes."""
        return math.ceil(self.size[0] / CHUNK), math.ceil(self.size[1] / CHUNK)


@dataclass(frozen=True)
class ChunkPlacement:
    """Where the big picture sits on screen at a pixel scale: the screen position of its top-left
    corner, rounded to a whole pixel so art pixels line up with screen pixels, and the chunks
    that show."""

    origin: tuple[int, int]
    scale: int
    chunks: list[tuple[int, int]]

    @classmethod
    def of(
        cls,
        projection: PixelProjection,
        camera_pixel: tuple[float, float],
        scale: int,
        center: tuple[float, float],
        clip: tuple[tuple[float, float], tuple[float, float]],
    ) -> Self:
        """`camera_pixel` is the art pixel at the middle of the view, which sits at `center`."""
        origin = (
            round(center[0] - camera_pixel[0] * scale),
            round(center[1] - camera_pixel[1] * scale),
        )
        (left, top), (right, bottom) = clip
        across, down = projection.chunks
        span = CHUNK * scale
        first_x = max(0, math.floor((left - origin[0]) / span))
        last_x = min(across - 1, math.floor((right - origin[0]) / span))
        first_y = max(0, math.floor((top - origin[1]) / span))
        last_y = min(down - 1, math.floor((bottom - origin[1]) / span))
        chunks = [(x, y) for y in range(first_y, last_y + 1) for x in range(first_x, last_x + 1)]
        return cls(origin, scale, chunks)

    def screen_rect(self, chunk: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        span = CHUNK * self.scale
        left = self.origin[0] + chunk[0] * span
        top = self.origin[1] + chunk[1] * span
        return (left, top), (left + span, top + span)
