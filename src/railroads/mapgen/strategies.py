"""Strategies: each one works out one property for every cell of the map.

A strategy is called with the map area and the layers worked out before it, so it can build on
them: temperature needs height, vegetation needs to know where the water is.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from railroads.mapgen.area import MapArea
from railroads.mapgen.noise import fractal_noise
from railroads.mapgen.sources import Elevation, Shape
from railroads.sim.geo import Coord
from railroads.sim.grids import BoolGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, Vegetation

type Layer = npt.NDArray[np.generic]


class Strategy(Protocol):
    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer: ...


@dataclass(frozen=True)
class Climate:
    """Yearly mean temperature from latitude and height: at sea level about 27 °C at the equator
    and -20 °C at the poles, and 6 °C colder for every km up. Rough, but enough to tell tropics,
    temperate land, tundra and snowy peaks apart anywhere in the world."""

    equator_c: float = 27.0
    pole_c: float = -20.0
    colder_per_km_up_c: float = 6.0

    def mean_c(self, lat: Coord, height_m: Layer) -> Layer:
        sea_level = self.pole_c + (self.equator_c - self.pole_c) * np.cos(np.radians(lat))
        return sea_level - self.colder_per_km_up_c * np.maximum(height_m, 0) / 1000


@dataclass(frozen=True)
class HeightFromElevation:
    """Average ground height over each cell, in metres."""

    elevation: Elevation

    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        if not self.elevation.covers(area.bounds):
            raise ValueError("the elevation data does not cover the map area")
        heights = self.elevation.sample(*area.sample_lonlat())
        return np.rint(area.samples.per_cell_mean(heights)).astype(np.int16)


@dataclass(frozen=True)
class WaterFromShapes:
    """Sea, lakes and rivers from mapped shapes; everything else is land.

    A cell is sea or lake when at least `min_cover` of it is. A river (all lines sharing its name)
    is kept only if it runs at least `min_river_km` inside the map, which drops the small ones.
    Without sea shapes the sea is taken from height: ground at 0 m or below that connects to the
    edge of the map.
    """

    lakes: Sequence[Shape]
    rivers: Sequence[Shape]
    sea: Sequence[Shape] | None = None
    min_river_km: float = 150.0
    min_cover: float = 0.5

    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        grid = area.grid
        if self.sea is None:
            sea = sea_from_height(layers["height_m"])
        else:
            sea = area.cover(self.sea) >= self.min_cover
        sea = grid.seal_corner_gaps(sea)
        lake = grid.seal_corner_gaps(area.cover(self.lakes) >= self.min_cover) & ~sea
        water = grid.seal_corner_gaps(sea | lake | self._rivers(area))
        terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
        terrain[water] = Terrain.RIVER
        terrain[lake] = Terrain.LAKE
        terrain[sea] = Terrain.SEA
        return terrain

    def _rivers(self, area: MapArea) -> BoolGrid:
        by_name: dict[str, list[Shape]] = defaultdict(list)
        for index, shape in enumerate(self.rivers):
            by_name[shape.name or f"unnamed {index}"].append(shape)
        rivers = np.zeros(area.grid.shape, dtype=bool)
        for shapes in by_name.values():
            if area.length_km(shapes) >= self.min_river_km:
                rivers |= area.trace(shapes)
        return rivers


def sea_from_height(height_m: Layer) -> BoolGrid:
    low = height_m <= 0
    sea = np.zeros_like(low)
    sea[[0, -1], :] = low[[0, -1], :]
    sea[:, [0, -1]] = low[:, [0, -1]]
    while True:
        grown = sea.copy()
        grown[1:] |= sea[:-1]
        grown[:-1] |= sea[1:]
        grown[:, 1:] |= sea[:, :-1]
        grown[:, :-1] |= sea[:, 1:]
        grown &= low
        if np.array_equal(grown, sea):
            return sea
        sea = grown


@dataclass(frozen=True)
class ReliefFromSlopes:
    """Flat, hills or mountains, from the grade of the climb to the steepest neighbouring cell:
    hills from `hills_from_pct`, mountains from `mountains_from_pct`. Then each cell takes the
    middle level of itself and its neighbours, so lone steep cells don't speckle the map and
    ranges have no holes. Water is flat, and the sea counts as ground at 0 m."""

    hills_from_pct: float = 3.0
    mountains_from_pct: float = 8.0

    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        grid = area.grid
        water = layers["terrain"] != Terrain.LAND
        height = np.maximum(layers["height_m"], 0).astype(float)
        rows, cols = grid.neighbour_index_grids()
        climb_m = np.abs(height[rows, cols] - height).max(axis=0)
        grade_pct = climb_m / (grid.km_per_cell * 10)
        relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
        relief[grade_pct >= self.hills_from_pct] = Relief.HILLS
        relief[grade_pct >= self.mountains_from_pct] = Relief.MOUNTAINS
        relief[water] = Relief.FLAT
        around = np.concatenate([relief[None], relief[rows, cols]])
        relief = np.median(around, axis=0).astype(np.uint8)
        relief[water] = Relief.FLAT
        return relief


@dataclass(frozen=True)
class TemperatureFromClimate:
    """How warm each cell feels: cold where the yearly mean is below `cold_below_c`, hot above
    `hot_above_c`, normal in between. Mapped deserts count `desert_warming_c` warmer, since
    cloudless deserts run hotter than the average for their latitude."""

    climate: Climate = Climate()
    cold_below_c: float = 4.0
    hot_above_c: float = 20.0
    deserts: Sequence[Shape] = ()
    desert_warming_c: float = 4.0

    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        _, lat = area.center_lonlat()
        desert = area.cover(self.deserts) >= 0.5
        celsius = self.climate.mean_c(lat, layers["height_m"]) + self.desert_warming_c * desert
        temperature = np.full(area.grid.shape, Temperature.NORMAL, dtype=np.uint8)
        temperature[celsius < self.cold_below_c] = Temperature.COLD
        temperature[celsius > self.hot_above_c] = Temperature.HOT
        return temperature


@dataclass(frozen=True)
class VegetationPlaceholder:
    """A stand-in until real land-cover data is plugged in: random forest patches over
    `forest_share` of the land, and bare ground on water, inside mapped deserts and where it is
    too cold for trees (a yearly mean below `treeline_c`)."""

    deserts: Sequence[Shape] = ()
    seed: int = 1
    forest_share: float = 0.3
    treeline_c: float = 1.0
    climate: Climate = Climate()
    patch_size_cells: float = 12.0

    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        land = layers["terrain"] == Terrain.LAND
        x, y = area.grid.centers()
        noise = fractal_noise(x, y, np.random.default_rng(self.seed), self.patch_size_cells)
        vegetation = np.full(area.grid.shape, Vegetation.GRASS, dtype=np.uint8)
        if land.any():
            forest = noise >= np.quantile(noise[land], 1 - self.forest_share)
            vegetation[forest] = Vegetation.FOREST
        _, lat = area.center_lonlat()
        too_cold = self.climate.mean_c(lat, layers["height_m"]) < self.treeline_c
        desert = area.cover(self.deserts) >= 0.5
        vegetation[~land | too_cold | desert] = Vegetation.BARREN
        return vegetation
