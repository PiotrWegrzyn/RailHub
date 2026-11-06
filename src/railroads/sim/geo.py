from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

KM_PER_DEGREE_LAT = 111.32

type Coord = float | npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Bounds:
    """A lon/lat box laid flat: x runs in km east of the west edge, y in km south of the north edge.

    The projection is equirectangular around the middle latitude, which is accurate to a few
    percent across a country.
    """

    west: float
    east: float
    south: float
    north: float

    @property
    def km_per_degree_lon(self) -> float:
        return KM_PER_DEGREE_LAT * float(np.cos(np.radians((self.south + self.north) / 2)))

    @property
    def center(self) -> tuple[float, float]:
        return (self.west + self.east) / 2, (self.south + self.north) / 2

    @property
    def width_km(self) -> float:
        return (self.east - self.west) * self.km_per_degree_lon

    @property
    def height_km(self) -> float:
        return (self.north - self.south) * KM_PER_DEGREE_LAT

    def to_km(self, lon: Coord, lat: Coord) -> tuple[Coord, Coord]:
        return (lon - self.west) * self.km_per_degree_lon, (self.north - lat) * KM_PER_DEGREE_LAT

    def to_lonlat(self, x_km: Coord, y_km: Coord) -> tuple[Coord, Coord]:
        return self.west + x_km / self.km_per_degree_lon, self.north - y_km / KM_PER_DEGREE_LAT


def describe_lonlat(lon: float, lat: float) -> str:
    north_south = "N" if lat >= 0 else "S"
    east_west = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{north_south} {abs(lon):.2f}°{east_west}"
