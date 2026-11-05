import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.units import km_to_tiles, tiles_to_km


class Terrain(IntEnum):
    WATER = 0
    PLAINS = 1
    HILLS = 2
    MOUNTAINS = 3


# Elevation quantiles separating water | plains | hills | mountains.
TERRAIN_QUANTILES = (0.25, 0.75, 0.92)


@dataclass(frozen=True, slots=True)
class World:
    width: int
    height: int
    terrain: npt.NDArray[np.uint8]

    @classmethod
    def generate(cls, width: int, height: int, seed: int) -> Self:
        rng = np.random.default_rng(seed)
        elevation = fractal_noise(width, height, rng)
        thresholds = np.quantile(elevation, TERRAIN_QUANTILES)
        terrain = np.digitize(elevation, thresholds).astype(np.uint8)
        return cls(width=width, height=height, terrain=terrain)

    @property
    def width_km(self) -> float:
        return tiles_to_km(self.width)

    @property
    def height_km(self) -> float:
        return tiles_to_km(self.height)

    def terrain_at(self, x: int, y: int) -> Terrain:
        return Terrain(int(self.terrain[y, x]))

    def tile_at_km(self, x_km: float, y_km: float) -> tuple[int, int] | None:
        x, y = math.floor(km_to_tiles(x_km)), math.floor(km_to_tiles(y_km))
        if 0 <= x < self.width and 0 <= y < self.height:
            return x, y
        return None


def fractal_noise(
    width: int,
    height: int,
    rng: np.random.Generator,
    octaves: int = 7,
) -> npt.NDArray[np.float64]:
    noise = np.zeros((height, width))
    cell = max(width, height) / 4
    amplitude = 1.0
    for _ in range(octaves):
        noise += amplitude * _value_noise(width, height, cell, rng)
        cell /= 2
        amplitude /= 2
    return noise


def _value_noise(
    width: int, height: int, cell: float, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    ys = np.arange(height) / cell
    xs = np.arange(width) / cell
    y0 = ys.astype(np.intp)
    x0 = xs.astype(np.intp)
    lattice = rng.random((y0[-1] + 2, x0[-1] + 2))
    ty = _smoothstep(ys - y0)[:, np.newaxis]
    tx = _smoothstep(xs - x0)[np.newaxis, :]
    top = lattice[np.ix_(y0, x0)] * (1 - tx) + lattice[np.ix_(y0, x0 + 1)] * tx
    bottom = lattice[np.ix_(y0 + 1, x0)] * (1 - tx) + lattice[np.ix_(y0 + 1, x0 + 1)] * tx
    return top * (1 - ty) + bottom * ty


def _smoothstep(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return t * t * (3 - 2 * t)
