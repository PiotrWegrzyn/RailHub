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
from railroads.mapgen.sources import ElevationGrid, Shape
from railroads.sim.grids import BoolGrid
from railroads.sim.tilemap import Temperature, Terrain, Vegetation

type Layer = npt.NDArray[np.generic]


class Strategy(Protocol):
    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer: ...


@dataclass(frozen=True)
class HeightFromElevation:
    """Average ground height over each cell, in metres."""

    elevation: ElevationGrid

    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer:
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
class TemperatureFromClimate:
    """How warm each cell feels, from a simple climate rule: colder further north and higher up.

    The defaults fit Central Europe, about 9 °C a year at sea level at 51° N. A cell whose yearly
    average is below `cold_below_c` is cold, above `hot_above_c` it is hot.
    """

    sea_level_c: float = 9.0
    reference_lat: float = 51.0
    colder_per_degree_north_c: float = 0.6
    colder_per_km_up_c: float = 6.0
    cold_below_c: float = 4.0
    hot_above_c: float = 16.0

    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        _, lat = area.center_lonlat()
        height_km = np.maximum(layers["height_m"], 0) / 1000
        celsius = (
            self.sea_level_c
            - self.colder_per_degree_north_c * (lat - self.reference_lat)
            - self.colder_per_km_up_c * height_km
        )
        temperature = np.full(area.grid.shape, Temperature.NORMAL, dtype=np.uint8)
        temperature[celsius < self.cold_below_c] = Temperature.COLD
        temperature[celsius > self.hot_above_c] = Temperature.HOT
        return temperature


@dataclass(frozen=True)
class VegetationPlaceholder:
    """A stand-in until real land-cover data is plugged in: random forest patches over
    `forest_share` of the land, bare ground above `barren_above_m`, and nothing on water."""

    seed: int = 1
    forest_share: float = 0.3
    barren_above_m: int = 1300
    patch_size_cells: float = 12.0

    def __call__(self, area: MapArea, layers: Mapping[str, Layer]) -> Layer:
        land = layers["terrain"] == Terrain.LAND
        x, y = area.grid.centers()
        noise = fractal_noise(x, y, np.random.default_rng(self.seed), self.patch_size_cells)
        vegetation = np.full(area.grid.shape, Vegetation.GRASS, dtype=np.uint8)
        if land.any():
            forest = noise >= np.quantile(noise[land], 1 - self.forest_share)
            vegetation[forest] = Vegetation.FOREST
        vegetation[~land | (layers["height_m"] >= self.barren_above_m)] = Vegetation.BARREN
        return vegetation
