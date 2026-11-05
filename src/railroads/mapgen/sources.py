"""Map data that strategies read: mapped shapes and elevation grids, all in lon/lat."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt

from railroads.sim.geo import Coord

FloatArray = npt.NDArray[np.float64]


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

    def sample(self, lon: Coord, lat: Coord) -> FloatArray:
        i = np.interp(lat, self.lats, np.arange(len(self.lats)))
        j = np.interp(lon, self.lons, np.arange(len(self.lons)))
        i0 = np.minimum(np.floor(i).astype(np.int64), len(self.lats) - 2)
        j0 = np.minimum(np.floor(j).astype(np.int64), len(self.lons) - 2)
        ti, tj = i - i0, j - j0
        h = self.heights
        top = h[i0, j0] * (1 - tj) + h[i0, j0 + 1] * tj
        bottom = h[i0 + 1, j0] * (1 - tj) + h[i0 + 1, j0 + 1] * tj
        return top * (1 - ti) + bottom * ti
