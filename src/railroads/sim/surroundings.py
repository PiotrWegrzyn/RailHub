"""What the land around a cell is like, as ladders weigh it: the share of each terrain, vegetation,
temperature and relief, and the deposits, in the cell and around it (Land, Surroundings)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import FlatIndices, FloatArray
from railroads.sim.reach import Neighbourhoods
from railroads.sim.tilemap import Deposit, Relief, Temperature, Terrain, TileMap, Vegetation

# Each land's weight tables (Land._tables), kept with the land they belong to.
_LAND_TABLES: dict[int, tuple] = {}


@dataclass(frozen=True, slots=True)
class Land:
    """How the land changes a ladder's chances: each weight multiplies them. Water weights go by
    the best water next to the cell, LAND standing for none; a minor river through the cell itself
    counts as a river (Surroundings.on_stream) unless minor_rivers is off. The sea is salty, so unless a ladder names it, it
    does no more than dry land; anything else left out weighs 1."""

    vegetation: Mapping[Vegetation, float] = field(default_factory=dict)
    relief: Mapping[Relief, float] = field(default_factory=dict)
    temperature: Mapping[Temperature, float] = field(default_factory=dict)
    water: Mapping[Terrain, float] = field(default_factory=dict)
    # Whether a minor river through the cell counts as a river.
    minor_rivers: bool = True
    # Weight for a cell with mountains next to it.
    beside_mountains: float = 1.0
    # Only where at least one cell next to it is flat open land: not forest, not water.
    forest_edge: bool = False
    # Only where the neighbourhood holds at least this many forest cells.
    least_forest_nearby: int = 0
    # Only on cells over this deposit (railroads.mapgen.deposits).
    deposit: Deposit | None = None
    # Only within this many degrees of the equator, north or south.
    most_latitude: float | None = None
    # Only inside this box of longitude and latitude: west, east, south, north.
    region: tuple[float, float, float, float] | None = None
    # Flat grassland weighs this much more, but where anything stands already only one turn in
    # this many goes ahead.
    open_field: float = 1.0

    @property
    def on_vegetation(self) -> bool:
        """Whether its weights change as forests come and go."""
        return bool(self.vegetation) or self.open_field != 1.0 or self.on_forest_nearby

    @property
    def on_forest_nearby(self) -> bool:
        """Whether a cell's weight changes with the forests around it, not just its own."""
        return self.forest_edge or self.least_forest_nearby > 0

    @property
    def _dry(self) -> float:
        return self.water.get(Terrain.LAND, 1.0)

    def _tables(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """The weight of every vegetation, relief, temperature and water, as arrays indexed by
        their values, worked out once for each land."""
        tables = _LAND_TABLES.get(id(self))
        if tables is None or tables[0] is not self:
            tables = (
                self,
                np.array([self.vegetation.get(v, 1.0) for v in Vegetation]),
                np.array([self.relief.get(r, 1.0) for r in Relief]),
                np.array([self.temperature.get(t, 1.0) for t in Temperature]),
                np.array(
                    [
                        self.water.get(t, self._dry) if t == Terrain.SEA else self.water.get(t, 1.0)
                        for t in Terrain
                    ]
                ),
            )
            _LAND_TABLES[id(self)] = tables
        return tables[1:]

    def weights(
        self,
        tile_map: TileMap,
        cells: npt.NDArray[np.intp] | None = None,
        near: FlatIndices | None = None,
    ) -> FloatArray:
        """The weight of every cell as a grid, or of the given flat cells alone. `near` holds
        every cell's neighbours as flat indices, one past the last cell for off the map."""
        weight = self.weights_in(Surroundings(tile_map, cells, near))
        return weight.reshape(tile_map.grid.shape) if cells is None else weight

    def weights_in(self, around: Surroundings) -> FloatArray:
        """The weight of the cells whose surroundings these are, in their order."""
        vegetation_weights, relief_weights, temperature_weights, water = self._tables()
        wet = around.is_water
        best = np.where(wet, water[around.terrain_around], 0.0).max(axis=0)
        own = around.on_stream if self.minor_rivers else np.zeros_like(around.on_stream)
        best = np.where(own, np.maximum(best, water[Terrain.RIVER]), best)
        by_water = np.where(wet.any(axis=0) | own, best, water[Terrain.LAND])
        weight = (
            vegetation_weights[around.vegetation]
            * relief_weights[around.relief]
            * temperature_weights[around.temperature]
            * by_water
        )
        weight = weight * np.where(around.beside_mountains, self.beside_mountains, 1.0)
        if self.forest_edge:
            weight = weight * around.beside_open_flat
        if self.least_forest_nearby:
            weight = weight * (around.forests >= self.least_forest_nearby)
        if self.deposit is not None:
            deposits = around.deposits
            on_it = deposits is not None and deposits == self.deposit
            weight = weight * on_it
        if self.most_latitude is not None:
            weight = weight * (np.abs(around.latitude) < self.most_latitude)
        if self.region is not None:
            west, east, south, north = self.region
            lon, lat = around.longitude, around.latitude
            weight = weight * ((lon >= west) & (lon <= east) & (lat >= south) & (lat <= north))
        weight = weight * np.where(around.open_flat_grass, self.open_field, 1.0)
        return np.where(around.land, weight, 0.0)


class Surroundings:
    """What lands weigh (Land.weights_in) in some flat cells of a map, or all of them, and around
    them: worked out once, as asked for, for any number of lands. `near` holds every cell's
    neighbours as flat indices, one past the last cell for off the map."""

    def __init__(
        self,
        tile_map: TileMap,
        cells: npt.NDArray[np.intp] | None = None,
        near: FlatIndices | None = None,
    ) -> None:
        size = tile_map.grid.rows * tile_map.grid.cols
        if near is None:
            near = Neighbourhoods(tile_map.grid).index
        self.tile_map = tile_map
        self.chosen = np.arange(size) if cells is None else cells
        self._terrain = tile_map.terrain.ravel()
        self._vegetation = tile_map.vegetation.ravel()
        self._relief = tile_map.relief.ravel()
        around = near[:, self.chosen]
        self.real = around < size
        self.around = np.where(self.real, around, 0)

    @cached_property
    def terrain_around(self) -> npt.NDArray[np.uint8]:
        return self._terrain[self.around]

    @cached_property
    def is_water(self) -> npt.NDArray[np.bool_]:
        return self.real & (self.terrain_around != Terrain.LAND)

    @cached_property
    def on_stream(self) -> npt.NDArray[np.bool_]:
        """Whether a minor river flows through the cell itself: fresh water like a river there,
        though not beside it, so the streams of a delta don't make a whole plain wet."""
        return self.tile_map.stream.ravel()[self.chosen].astype(bool) & self.land

    @cached_property
    def vegetation(self) -> npt.NDArray[np.uint8]:
        return self._vegetation[self.chosen]

    @cached_property
    def relief(self) -> npt.NDArray[np.uint8]:
        return self._relief[self.chosen]

    @cached_property
    def temperature(self) -> npt.NDArray[np.uint8]:
        return self.tile_map.temperature.ravel()[self.chosen]

    @cached_property
    def beside_mountains(self) -> npt.NDArray[np.bool_]:
        beside = self.real & (self._relief[self.around] == Relief.MOUNTAINS)
        return beside.any(axis=0)

    @cached_property
    def forest_around(self) -> npt.NDArray[np.bool_]:
        return self.real & (self._vegetation[self.around] == Vegetation.FOREST) & ~self.is_water

    @cached_property
    def beside_open_flat(self) -> npt.NDArray[np.bool_]:
        open_flat = (
            self.real
            & ~self.is_water
            & ~self.forest_around
            & (self._relief[self.around] == Relief.FLAT)
        )
        return open_flat.any(axis=0)

    @cached_property
    def forests(self) -> npt.NDArray[np.int64]:
        return self.forest_around.sum(axis=0) + (self.vegetation == Vegetation.FOREST)

    @cached_property
    def deposits(self) -> npt.NDArray[np.uint8] | None:
        deposits = self.tile_map.deposits
        return None if deposits is None else deposits.ravel()[self.chosen]

    @cached_property
    def open_flat_grass(self) -> npt.NDArray[np.bool_]:
        return (self.relief == Relief.FLAT) & (self.vegetation == Vegetation.GRASS)

    @cached_property
    def _lonlat(self) -> tuple[FloatArray, FloatArray]:
        rows, cols = np.unravel_index(self.chosen, self.tile_map.grid.shape)
        x, y = self.tile_map.grid.center(rows, cols)
        km = self.tile_map.grid.km_per_cell
        lon, lat = self.tile_map.bounds.to_lonlat(np.asarray(x) * km, np.asarray(y) * km)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    @cached_property
    def longitude(self) -> FloatArray:
        return self._lonlat[0]

    @cached_property
    def latitude(self) -> FloatArray:
        return self._lonlat[1]

    @cached_property
    def land(self) -> npt.NDArray[np.bool_]:
        return self._terrain[self.chosen] == Terrain.LAND
