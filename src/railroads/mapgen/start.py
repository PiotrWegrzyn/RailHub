"""Where a new game starts: a random spot in the world that is mostly land."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from railroads.mapgen.area import MapArea
from railroads.mapgen.sources import Shape
from railroads.sim.geo import KM_PER_DEGREE_LAT, Bounds
from railroads.sim.grids import SquareGrid

# Land is measured on a coarse grid, which is plenty to tell sea from land at country size.
LAND_CHECK_KM_PER_CELL = 20.0


@dataclass(frozen=True)
class StartRegion:
    """The x (longitude) and y (latitude) limits a new map must fit inside, and how much of the
    map must be land. The defaults leave out the polar regions."""

    west: float = -180.0
    east: float = 180.0
    south: float = -55.0
    north: float = 65.0
    min_land_share: float = 0.6

    def contains(self, bounds: Bounds) -> bool:
        return (
            self.west <= bounds.west
            and bounds.east <= self.east
            and self.south <= bounds.south
            and bounds.north <= self.north
        )


def pick_start_bounds(
    rng: np.random.Generator,
    region: StartRegion,
    width_km: float,
    height_km: float,
    water: Sequence[Shape],
    attempts: int = 1000,
) -> Bounds:
    """A random map area inside the region with enough land. Spots are spread evenly over the
    globe's surface, so high latitudes are not picked more often than their real size."""
    low, high = np.sin(np.radians([region.south, region.north]))
    half_height = height_km / 2 / KM_PER_DEGREE_LAT
    for _ in range(attempts):
        lat = math.degrees(math.asin(rng.uniform(low, high)))
        lon = rng.uniform(region.west, region.east)
        half_width = width_km / 2 / (KM_PER_DEGREE_LAT * math.cos(math.radians(lat)))
        bounds = Bounds(
            west=lon - half_width,
            east=lon + half_width,
            south=lat - half_height,
            north=lat + half_height,
        )
        if region.contains(bounds) and land_share(bounds, water) >= region.min_land_share:
            return bounds
    raise ValueError("found no spot in the start region with enough land")


def land_share(bounds: Bounds, water: Sequence[Shape]) -> float:
    grid = SquareGrid.covering(bounds.width_km, bounds.height_km, LAND_CHECK_KM_PER_CELL)
    return 1.0 - float(MapArea.of(bounds, grid).cover(water).mean())
