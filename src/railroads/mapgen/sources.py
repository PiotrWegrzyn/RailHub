"""Map data that strategies read: mapped shapes and elevation, all in lon/lat."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import numpy as np
import numpy.typing as npt

from railroads.sim.geo import Bounds, Coord

FloatArray = npt.NDArray[np.float64]


class Elevation(Protocol):
    """Anything that can tell the ground height in metres at lon/lat points."""

    def covers(self, bounds: Bounds) -> bool: ...

    def sample(self, lon: Coord, lat: Coord) -> FloatArray: ...


@dataclass(frozen=True)
class Shape:
    """One mapped thing: a polygon made of rings (holes included), or a set of lines.

    Each part is an (n, 2) array of lon, lat points.
    """

    kind: Literal["polygon", "line"]
    parts: tuple[FloatArray, ...]
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str | None:
        return self.properties.get("name")


def read_geojson(path: Path) -> list[Shape]:
    shapes = []
    for feature in json.loads(path.read_text())["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        kind, parts = _parts(geometry)
        arrays = tuple(np.asarray(part, dtype=float)[:, :2] for part in parts if len(part) >= 2)
        if arrays:
            shapes.append(Shape(kind, arrays, feature.get("properties") or {}))
    return shapes


def _parts(geometry: dict[str, Any]) -> tuple[Literal["polygon", "line"], list[Any]]:
    match geometry["type"]:
        case "Polygon":
            return "polygon", geometry["coordinates"]
        case "MultiPolygon":
            return "polygon", [ring for polygon in geometry["coordinates"] for ring in polygon]
        case "LineString":
            return "line", [geometry["coordinates"]]
        case "MultiLineString":
            return "line", geometry["coordinates"]
    return "line", []


@dataclass(frozen=True)
class ElevationGrid:
    """Heights in metres on a regular lon/lat grid, read between grid points by bilinear
    interpolation."""

    lons: FloatArray
    lats: FloatArray
    heights: FloatArray

    @classmethod
    def from_erddap_csv(cls, path: Path) -> Self:
        rows = np.loadtxt(path, delimiter=",", skiprows=2)
        lats, lons = np.unique(rows[:, 0]), np.unique(rows[:, 1])
        return cls(lons=lons, lats=lats, heights=rows[:, 2].reshape(len(lats), len(lons)))

    def covers(self, bounds: Bounds) -> bool:
        slack = 1e-6
        return bool(
            self.lons[0] <= bounds.west + slack
            and self.lons[-1] >= bounds.east - slack
            and self.lats[0] <= bounds.south + slack
            and self.lats[-1] >= bounds.north - slack
        )

    def sample(self, lon: Coord, lat: Coord) -> FloatArray:
        rows = np.interp(lat, self.lats, np.arange(len(self.lats)))
        cols = np.interp(lon, self.lons, np.arange(len(self.lons)))
        return bilinear(self.heights, rows, cols)


def bilinear(values: npt.NDArray[np.generic], rows: Coord, cols: Coord) -> FloatArray:
    """Values between the points of a 2D array at fractional row and column positions, clamped
    to the array's edges."""
    rows = np.clip(rows, 0, values.shape[0] - 1)
    cols = np.clip(cols, 0, values.shape[1] - 1)
    r0 = np.minimum(np.floor(rows).astype(np.int64), values.shape[0] - 2)
    c0 = np.minimum(np.floor(cols).astype(np.int64), values.shape[1] - 2)
    tr, tc = rows - r0, cols - c0
    top = values[r0, c0] * (1 - tc) + values[r0, c0 + 1] * tc
    bottom = values[r0 + 1, c0] * (1 - tc) + values[r0 + 1, c0 + 1] * tc
    return top * (1 - tr) + bottom * tr
