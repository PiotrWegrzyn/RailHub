"""Real place names for stations, from the GeoNames places in the world's files (world/places/,
see railroads.mapgen.world_files). A station takes the name of the town it stands in, or else of
the closest place within NEAREST_KM. A second station with the same name is numbered
("Łódź [2]"); a station with no place that near, or every station when the game has no world
files, is called Station 1, Station 2 and so on.

A town reaches REACH_KM from its centre plus REACH_KM_PER_SCALE for every SCALE_PEOPLE people
taken as a square root, so a village of 500 reaches about 1.3 km and a city of a million 13 km;
a station within the reach of several towns stands in the one it lies deepest within.
"""

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.mapgen import world_files
from railroads.mapgen.world_files import World
from railroads.sim.geo import KM_PER_DEGREE_LAT, Bounds
from railroads.sim.grids import Cell
from railroads.sim.tilemap import TileMap

REACH_KM = 1.0
REACH_KM_PER_SCALE = 1.2
SCALE_PEOPLE = 10_000
NEAREST_KM = 50.0
# How far outside a map's edges places are still kept, in degrees: a bit more than NEAREST_KM.
MARGIN_DEG = 0.75


@dataclass(frozen=True)
class Places:
    names: tuple[str, ...]
    lon: npt.NDArray[np.float64]
    lat: npt.NDArray[np.float64]
    population: npt.NDArray[np.int64]

    @classmethod
    def near(cls, bounds: Bounds, world: World | None = None) -> Places | None:
        """The places on and around a map from the world's files (`world`, or else the game's
        own), or None without them."""
        world = world_files.world() if world is None else world
        if world is None:
            return None
        wide = Bounds(
            west=bounds.west - MARGIN_DEG,
            east=bounds.east + MARGIN_DEG,
            south=bounds.south - MARGIN_DEG,
            north=bounds.north + MARGIN_DEG,
        )
        return cls.joined([cls.read(data, wide) for data in world.places_files(wide)])

    @classmethod
    def read(cls, data: bytes, bounds: Bounds) -> Places:
        """The places in a places file that lie inside the bounds."""
        with np.load(io.BytesIO(data)) as arrays:
            lon, lat = arrays["lon"].astype(np.float64), arrays["lat"].astype(np.float64)
            kept = np.flatnonzero(
                (lon >= bounds.west)
                & (lon <= bounds.east)
                & (lat >= bounds.south)
                & (lat <= bounds.north)
            )
            ends = arrays["ends"]
            blob = arrays["names"].tobytes()
            starts = np.concatenate([[0], ends[:-1]])
            names = tuple(blob[starts[i] : ends[i]].decode("utf-8") for i in kept.tolist())
            return cls(names, lon[kept], lat[kept], arrays["population"][kept].astype(np.int64))

    @classmethod
    def joined(cls, parts: Sequence[Places]) -> Places:
        return cls(
            tuple(name for part in parts for name in part.names),
            np.concatenate([part.lon for part in parts] or [np.zeros(0)]),
            np.concatenate([part.lat for part in parts] or [np.zeros(0)]),
            np.concatenate([part.population for part in parts] or [np.zeros(0, dtype=np.int64)]),
        )

    def write(self) -> bytes:
        """The places as a places file."""
        encoded = [name.encode("utf-8") for name in self.names]
        out = io.BytesIO()
        np.savez_compressed(
            out,
            names=np.frombuffer(b"".join(encoded), dtype=np.uint8),
            ends=np.cumsum([len(name) for name in encoded], dtype=np.int64),
            lat=self.lat.astype(np.float32),
            lon=self.lon.astype(np.float32),
            population=self.population.astype(np.int64),
        )
        return out.getvalue()

    def town_at(self, lon: float, lat: float, nearest_km: float = NEAREST_KM) -> str | None:
        """The place a point is named after: of the towns whose reach holds it, the one it lies
        deepest within, else the closest place within `nearest_km`; None if there is none."""
        if not self.names:
            return None
        dx = (self.lon - lon) * KM_PER_DEGREE_LAT * math.cos(math.radians(lat))
        dy = (self.lat - lat) * KM_PER_DEGREE_LAT
        distance = np.hypot(dx, dy)
        reach = REACH_KM + REACH_KM_PER_SCALE * np.sqrt(self.population / SCALE_PEOPLE)
        depth = distance / reach
        deepest = int(np.argmin(depth))
        if depth[deepest] <= 1.0:
            return self.names[deepest]
        closest = int(np.argmin(distance))
        return self.names[closest] if distance[closest] <= nearest_km else None


class StationNames:
    """The name of every station on a map, given when it is first asked for and kept."""

    def __init__(self, tile_map: TileMap, places: Places | None) -> None:
        self.tile_map = tile_map
        self.places = places
        self._names: dict[Cell, str] = {}
        self._numbered = 0

    def __call__(self, station: Cell) -> str:
        if station not in self._names:
            self._names[station] = self._new_name(station)
        return self._names[station]

    def move(self, old: Cell, new: Cell) -> None:
        """Keeps a station's name when it moves."""
        if old in self._names:
            self._names[new] = self._names.pop(old)

    def _new_name(self, station: Cell) -> str:
        town = None
        if self.places is not None:
            town = self.places.town_at(*self.tile_map.lonlat(station))
        if town is None:
            self._numbered += 1
            return f"Station {self._numbered}"
        taken = set(self._names.values())
        if town not in taken:
            return town
        number = 2
        while f"{town} [{number}]" in taken:
            number += 1
        return f"{town} [{number}]"
