"""A map divided into cells, where every cell has the same set of properties.

Each property is stored as one numpy layer with one value per cell, so a whole map is a few small
arrays. Tile is the view of one cell with all its properties together.
"""

import json
import math
from dataclasses import asdict, dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.geo import Bounds
from railroads.sim.grids import GRIDS, Cell, Grid

FORMAT_VERSION = 3
LAYERS = ("terrain", "vegetation", "temperature", "height_m", "relief", "stream")
# The share of a place's land a new game starts on (TileMap.square_middle).
START_SHARE = 0.25
# About the land cells of the map of Poland: the counts of rare buildings each game gets are set
# for a map of this much land and scaled to the land of the others.
REFERENCE_LAND = 121_000


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
    COPPER = 2
    GOLD = 3
    DIAMONDS = 4
    OIL = 5
    SALT = 6
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
    stream: bool = False


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
    # The cells on land next to the player's that the player has not bought (railroads.sim.lots),
    # where nothing grows or is built; None while there are none.
    locked: npt.NDArray[np.bool_] | None = None
    # Where a minor river flows through a land cell, its links to the cells beside it that the
    # stream runs to or comes from, a bit for each of the grid's edge_normals; 0 elsewhere
    # (railroads.mapgen.streams).
    stream: npt.NDArray[np.uint8] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.stream is None:
            object.__setattr__(self, "stream", np.zeros(self.grid.shape, dtype=np.uint8))
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
            stream=bool(self.stream[cell]),
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

    def middle_square(self, share: float = START_SHARE) -> tuple[int, int, int, int]:
        """Where the square in the middle of the map holding this share of its land lies, as its
        top row, left column, rows and columns, or the biggest square the map has room for. It
        splits into four lots (railroads.sim.lots) of whole cells, each starting on an even row
        so every hex keeps its neighbours: its rows come in fours and its columns in twos."""
        step = float(self.grid.center(1, 0)[1] - self.grid.center(0, 0)[1])
        side = math.sqrt(self.grid.rows * self.grid.cols * step * share)
        cols = min(self.grid.cols // 2 * 2, max(2, round(side / 2) * 2))
        rows = min(self.grid.rows // 4 * 4, max(4, round(cols / step / 4) * 4))
        cols = min(cols, max(2, round(rows * step / 2) * 2))
        top = (self.grid.rows - rows) // 2 // 2 * 2
        left = (self.grid.cols - cols) // 2
        return top, left, rows, cols

    def square_middle(self, share: float = START_SHARE) -> Self:
        """The square in the middle of the map holding this share of its land (middle_square),
        its bounds moved with it, so places stay where they are."""
        return self.window(*self.middle_square(share))

    def window(self, top: int, left: int, rows: int, cols: int) -> Self:
        """The `rows` by `cols` cells from row `top` and column `left` on, which may reach past
        the map, where they are sea, flat and barren; its bounds go with it. `top` must be even,
        so every hex keeps its neighbours."""
        if top % 2:
            raise ValueError(f"A window must start on an even row, not {top}")
        grid = self.grid
        step = float(grid.center(1, 0)[1] - grid.center(0, 0)[1])
        x_km, y_km = left * grid.km_per_cell, top * step * grid.km_per_cell
        west, north = self.bounds.to_lonlat(x_km, y_km)
        east, south = self.bounds.to_lonlat(
            x_km + cols * grid.km_per_cell, y_km + rows * step * grid.km_per_cell
        )
        rows_in = slice(max(top, 0), min(top + rows, grid.rows))
        cols_in = slice(max(left, 0), min(left + cols, grid.cols))
        into = (
            slice(rows_in.start - top, rows_in.stop - top),
            slice(cols_in.start - left, cols_in.stop - left),
        )

        def cut(layer: np.ndarray, outside: int) -> np.ndarray:
            part = np.full((rows, cols), outside, dtype=layer.dtype)
            if rows_in.start < rows_in.stop and cols_in.start < cols_in.stop:
                part[into] = layer[rows_in, cols_in]
            return part

        outside = {"terrain": Terrain.SEA, "relief": Relief.FLAT, "height_m": 0}
        outside |= {"vegetation": Vegetation.BARREN, "temperature": Temperature.NORMAL, "stream": 0}
        layers = {layer: cut(getattr(self, layer), outside[layer]) for layer in LAYERS}
        return replace(
            self,
            grid=type(grid)(rows=rows, cols=cols, km_per_cell=grid.km_per_cell),
            bounds=Bounds(
                west=float(west), east=float(east), south=float(south), north=float(north)
            ),
            deposits=None if self.deposits is None else cut(self.deposits, Deposit.NONE),
            locked=None if self.locked is None else cut(self.locked, 1).astype(bool),
            **layers,
        )

    def is_locked(self, cell: Cell) -> bool:
        """Whether a cell lies on land the player has not bought yet (railroads.sim.lots)."""
        return self.locked is not None and bool(self.locked[cell])

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
