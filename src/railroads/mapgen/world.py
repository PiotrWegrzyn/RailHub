"""A coarse picture of the whole world's land, used to show where a map lies."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.mapgen.area import MapArea
from railroads.mapgen.sources import Shape
from railroads.sim.geo import KM_PER_DEGREE_LAT, Bounds, Coord
from railroads.sim.grids import SquareGrid

WHOLE_WORLD = Bounds(west=-180.0, east=180.0, south=-90.0, north=90.0)


@dataclass(frozen=True)
class WorldLand:
    """How much of each cell of a lon/lat grid over the whole world is land, from 0 to 255.
    Rows run from the north pole south, columns from 180°W east."""

    land: npt.NDArray[np.uint8]

    @classmethod
    def bake(cls, water: Sequence[Shape], degrees_per_cell: float) -> Self:
        grid = SquareGrid.covering(
            WHOLE_WORLD.width_km, WHOLE_WORLD.height_km, degrees_per_cell * KM_PER_DEGREE_LAT
        )
        wet = MapArea.of(WHOLE_WORLD, grid).cover(water)
        return cls(np.rint((1.0 - wet) * 255).astype(np.uint8))

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path) as data:
            return cls(data["land"])

    def save(self, path: Path) -> None:
        np.savez_compressed(path, land=self.land)

    @property
    def degrees_per_cell(self) -> float:
        return 360.0 / self.land.shape[1]

    def land_at(self, lon: Coord, lat: Coord) -> npt.NDArray[np.uint8]:
        rows, cols = self.land.shape
        row = np.clip(((90.0 - np.asarray(lat)) / self.degrees_per_cell).astype(int), 0, rows - 1)
        col = np.clip(((np.asarray(lon) + 180.0) / self.degrees_per_cell).astype(int), 0, cols - 1)
        return self.land[row, col]
