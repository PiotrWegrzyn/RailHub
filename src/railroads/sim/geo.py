import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.units import km_to_tiles

KM_PER_DEGREE_LAT = 111.32

LonLat = tuple[float, float]
FloatGrid = npt.NDArray[np.float64]
BoolGrid = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Projection:
    """Equirectangular projection of a lon/lat box onto the tile grid, north at row 0."""

    west: float
    east: float
    south: float
    north: float

    @property
    def km_per_degree_lon(self) -> float:
        return KM_PER_DEGREE_LAT * float(np.cos(np.radians((self.south + self.north) / 2)))

    @property
    def width_tiles(self) -> int:
        return round(km_to_tiles((self.east - self.west) * self.km_per_degree_lon))

    @property
    def height_tiles(self) -> int:
        return round(km_to_tiles((self.north - self.south) * KM_PER_DEGREE_LAT))

    def to_tile(self, lon: float, lat: float) -> tuple[float, float]:
        x = (lon - self.west) / (self.east - self.west) * self.width_tiles
        y = (self.north - lat) / (self.north - self.south) * self.height_tiles
        return x, y

    def tile_centers(self) -> tuple[FloatGrid, FloatGrid]:
        ys, xs = np.mgrid[0 : self.height_tiles, 0 : self.width_tiles]
        return xs + 0.5, ys + 0.5


def polygon_mask(
    xs: FloatGrid, ys: FloatGrid, projection: Projection, polygon: list[LonLat]
) -> BoolGrid:
    points = [projection.to_tile(lon, lat) for lon, lat in polygon]
    inside = np.zeros(xs.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
        crosses = (y1 > ys) != (y2 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_y = x1 + (ys - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (xs < x_at_y)
    return inside


def distance_to_polyline(
    xs: FloatGrid, ys: FloatGrid, projection: Projection, polyline: list[LonLat]
) -> FloatGrid:
    points = [projection.to_tile(lon, lat) for lon, lat in polyline]
    best = np.full(xs.shape, np.inf)
    for (x1, y1), (x2, y2) in itertools.pairwise(points):
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy or 1.0
        t = np.clip(((xs - x1) * dx + (ys - y1) * dy) / length_sq, 0.0, 1.0)
        best = np.minimum(best, np.hypot(xs - (x1 + t * dx), ys - (y1 + t * dy)))
    return best
