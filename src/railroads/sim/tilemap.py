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

FORMAT_VERSION = 2
LAYERS = ("terrain", "vegetation", "temperature", "height_m", "relief")


class Terrain(IntEnum):
    LAND = 0
    SEA = 1
    LAKE = 2
    RIVER = 3


class Vegetation(IntEnum):
    BARREN = 0
    GRASS = 1
    FOREST = 2


class Deposit(IntEnum):
    NONE = 0
    IRON = 1
    COPPER = 2
    GOLD = 3
    DIAMONDS = 4
    OIL = 5
    SALT = 6
    FISH = 7
    WOOL = 8
    GRAPES = 9
    COTTON = 10
    COFFEE = 11
    RUBBER = 12
    # Hidden ground where military outposts can open.
    GARRISON = 13
    INCENSE = 14


class Temperature(IntEnum):
    COLD = 0
    NORMAL = 1
    HOT = 2


class Relief(IntEnum):
    """The shape of the land: flat, hills (slower, dearer track) or mountains (no track)."""

    FLAT = 0
    HILLS = 1
    MOUNTAINS = 2


@dataclass(frozen=True, slots=True)
class Tile:
    terrain: Terrain
    vegetation: Vegetation
    temperature: Temperature
    height_m: int
    relief: Relief = Relief.FLAT


@dataclass(frozen=True, slots=True)
class TileMap:
    name: str
    grid: Grid
    bounds: Bounds
    terrain: npt.NDArray[np.uint8]
    vegetation: npt.NDArray[np.uint8]
    temperature: npt.NDArray[np.uint8]
    height_m: npt.NDArray[np.int16]
    relief: npt.NDArray[np.uint8]
    sources: tuple[str, ...] = ()
    # Each cell's Deposit, placed when a game starts (railroads.mapgen.deposits).
    deposits: npt.NDArray[np.uint8] | None = None

    def __post_init__(self) -> None:
        for layer in LAYERS:
            values = getattr(self, layer)
            if values.shape != self.grid.shape:
                raise ValueError(f"{layer} is {values.shape} but the grid is {self.grid.shape}")
            values.setflags(write=False)

    def set_vegetation(self, cells: npt.NDArray[np.intp], vegetation: Vegetation) -> None:
        """Changes the vegetation of these flat cells, as forests come and go; every other layer
        stays as the map was made. A map sent between processes holds its layers in memory that
        cannot be written to, so it takes a copy of its vegetation first."""
        layer = self.vegetation
        try:
            layer.setflags(write=True)
        except ValueError:
            layer = layer.copy()
            object.__setattr__(self, "vegetation", layer)
        layer.flat[cells] = vegetation
        layer.setflags(write=False)

    def tile(self, cell: Cell) -> Tile:
        return Tile(
            terrain=Terrain(self.terrain[cell]),
            vegetation=Vegetation(self.vegetation[cell]),
            temperature=Temperature(self.temperature[cell]),
            height_m=int(self.height_m[cell]),
            relief=Relief(self.relief[cell]),
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
