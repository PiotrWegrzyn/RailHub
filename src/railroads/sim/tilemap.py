"""A map divided into cells, where every cell has the same set of properties.

Each property is stored as one numpy layer with one value per cell, so a whole map is a few small
arrays. Tile is the view of one cell with all its properties together.
"""

import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.geo import Bounds
from railroads.sim.grids import GRIDS, Cell, Grid

FORMAT_VERSION = 1
LAYERS = ("terrain", "vegetation", "temperature", "height_m")


class Terrain(IntEnum):
    LAND = 0
    SEA = 1
    LAKE = 2
    RIVER = 3


class Vegetation(IntEnum):
    BARREN = 0
    GRASS = 1
    FOREST = 2


class Temperature(IntEnum):
    COLD = 0
    NORMAL = 1
    HOT = 2


@dataclass(frozen=True, slots=True)
class Tile:
    terrain: Terrain
    vegetation: Vegetation
    temperature: Temperature
    height_m: int


@dataclass(frozen=True, slots=True)
class TileMap:
    name: str
    grid: Grid
    bounds: Bounds
    terrain: npt.NDArray[np.uint8]
    vegetation: npt.NDArray[np.uint8]
    temperature: npt.NDArray[np.uint8]
    height_m: npt.NDArray[np.int16]
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for layer in LAYERS:
            values = getattr(self, layer)
            if values.shape != self.grid.shape:
                raise ValueError(f"{layer} is {values.shape} but the grid is {self.grid.shape}")
            values.setflags(write=False)

    def tile(self, cell: Cell) -> Tile:
        return Tile(
            terrain=Terrain(self.terrain[cell]),
            vegetation=Vegetation(self.vegetation[cell]),
            temperature=Temperature(self.temperature[cell]),
            height_m=int(self.height_m[cell]),
        )

    def cell_at_lonlat(self, lon: float, lat: float) -> Cell | None:
        x_km, y_km = self.bounds.to_km(lon, lat)
        row, col = self.grid.cell_at(x_km / self.grid.km_per_cell, y_km / self.grid.km_per_cell)
        cell = int(row), int(col)
        return cell if self.grid.contains(*cell) else None

    def lonlat(self, cell: Cell) -> tuple[float, float]:
        x, y = self.grid.center(*cell)
        km = self.grid.km_per_cell
        lon, lat = self.bounds.to_lonlat(float(x) * km, float(y) * km)
        return float(lon), float(lat)

    def save(self, path: Path) -> None:
        meta = {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "grid": self.grid.kind,
            "km_per_cell": self.grid.km_per_cell,
            "bounds": asdict(self.bounds),
            "sources": list(self.sources),
        }
        layers = {layer: getattr(self, layer) for layer in LAYERS}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **layers)

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"{path} has map format {meta.get('format_version')}, "
                    f"this game reads format {FORMAT_VERSION}"
                )
            layers = {layer: data[layer] for layer in LAYERS}
        rows, cols = layers["terrain"].shape
        grid = GRIDS[meta["grid"]](rows=rows, cols=cols, km_per_cell=meta["km_per_cell"])
        return cls(
            name=meta["name"],
            grid=grid,
            bounds=Bounds(**meta["bounds"]),
            sources=tuple(meta["sources"]),
            **layers,
        )
