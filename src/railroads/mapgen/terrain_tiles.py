"""Worldwide heights from Terrarium terrain tiles, gathered once into a local SQLite database by
tools/build_elevation_db.py. The game reads them from the small files under world/ that
tools/build_world_files.py writes from it (railroads.mapgen.world_files).

The tiles are Mapzen's elevation served as web map images. They use the Web Mercator projection:
at zoom z the world is 2**z by 2**z tiles of 256 x 256 pixels, x counting east from 180° W and
y counting south from about 85° N.
"""

import math
import sqlite3
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from railroads.mapgen.sources import FloatArray, bilinear
from railroads.sim.geo import Bounds, Coord

TILE_SIZE = 256
ZOOM = 6
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
# Height wherever the database has no tile: only tiles with land are stored.
OPEN_SEA_M = -100
SOURCE = (
    "Elevation: Mapzen terrain tiles on AWS Open Data, from SRTM, GMTED2010, ETOPO1, EU-DEM "
    "and others (github.com/tilezen/joerd/blob/master/docs/attribution.md)"
)

type HeightTile = npt.NDArray[np.int16]


def world_pixel(lon: Coord, lat: Coord, zoom: int = ZOOM) -> tuple[Coord, Coord]:
    """Position in pixels on the whole-world Web Mercator map at this zoom."""
    size = TILE_SIZE * 2**zoom
    x = (np.asarray(lon, dtype=float) + 180) / 360 * size
    y = (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * size
    return x, y


def world_lonlat(x: Coord, y: Coord, zoom: int = ZOOM) -> tuple[Coord, Coord]:
    size = TILE_SIZE * 2**zoom
    lon = np.asarray(x, dtype=float) / size * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * np.asarray(y, dtype=float) / size))))
    return lon, lat


def decode_terrarium(rgb: npt.NDArray[np.uint8]) -> HeightTile:
    """Heights in metres from a Terrarium tile's pixels: red * 256 + green + blue / 256 - 32768."""
    channels = rgb.astype(np.float64)
    metres = channels[..., 0] * 256 + channels[..., 1] + channels[..., 2] / 256 - 32768
    return np.rint(metres).astype(np.int16)


@dataclass(frozen=True)
class TileElevation:
    """Heights from a block of neighbouring tiles, read between pixels by bilinear interpolation."""

    heights: HeightTile
    first_tile: tuple[int, int]
    zoom: int = ZOOM

    def covers(self, bounds: Bounds) -> bool:
        left, top = self._position(bounds.west, bounds.north)
        right, bottom = self._position(bounds.east, bounds.south)
        rows, cols = self.heights.shape
        return bool(left >= 0 and top >= 0 and right <= cols - 1 and bottom <= rows - 1)

    def sample(self, lon: Coord, lat: Coord) -> FloatArray:
        cols, rows = self._position(lon, lat)
        return bilinear(self.heights, rows, cols)

    def _position(self, lon: Coord, lat: Coord) -> tuple[Coord, Coord]:
        """Fractional column and row in `heights`, with pixel centres on whole numbers."""
        x, y = world_pixel(lon, lat, self.zoom)
        first_x, first_y = self.first_tile
        return x - first_x * TILE_SIZE - 0.5, y - first_y * TILE_SIZE - 0.5


class ElevationDatabase:
    """Terrain tiles of one zoom level in SQLite, each stored as zlib-packed 16-bit metres."""

    def __init__(self, path: Path, zoom: int = ZOOM) -> None:
        self.zoom = zoom
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS tiles "
            "(z INTEGER, x INTEGER, y INTEGER, heights BLOB, PRIMARY KEY (z, x, y))"
        )

    def close(self) -> None:
        self._db.close()

    def put(self, x: int, y: int, heights: HeightTile) -> None:
        packed = zlib.compress(heights.astype("<i2").tobytes())
        self._db.execute(
            "INSERT OR REPLACE INTO tiles VALUES (?, ?, ?, ?)", (self.zoom, x, y, packed)
        )
        self._db.commit()

    def get(self, x: int, y: int) -> HeightTile | None:
        row = self._db.execute(
            "SELECT heights FROM tiles WHERE z = ? AND x = ? AND y = ?", (self.zoom, x, y)
        ).fetchone()
        if row is None:
            return None
        heights = np.frombuffer(zlib.decompress(row[0]), dtype="<i2")
        return heights.reshape(TILE_SIZE, TILE_SIZE).astype(np.int16)

    def stored(self) -> set[tuple[int, int]]:
        rows = self._db.execute("SELECT x, y FROM tiles WHERE z = ?", (self.zoom,)).fetchall()
        return {(x, y) for x, y in rows}

    def elevation_for(self, bounds: Bounds, margin_px: int = 2) -> TileElevation:
        """Heights around the bounds from the stored tiles, and open sea where none is stored."""
        return elevation_from_tiles(bounds, self.get, self.zoom, margin_px)


def elevation_from_tiles(
    bounds: Bounds,
    tile: Callable[[int, int], HeightTile | None],
    zoom: int = ZOOM,
    margin_px: int = 2,
) -> TileElevation:
    """Heights around the bounds from the tiles `tile` gives by x and y, and open sea where it
    gives none."""
    left, top = world_pixel(bounds.west, bounds.north, zoom)
    right, bottom = world_pixel(bounds.east, bounds.south, zoom)
    tiles_across = 2**zoom
    x0 = math.floor((left - margin_px) / TILE_SIZE)
    x1 = math.floor((right + margin_px) / TILE_SIZE)
    y0 = max(0, math.floor((top - margin_px) / TILE_SIZE))
    y1 = min(tiles_across - 1, math.floor((bottom + margin_px) / TILE_SIZE))
    shape = ((y1 - y0 + 1) * TILE_SIZE, (x1 - x0 + 1) * TILE_SIZE)
    heights = np.full(shape, OPEN_SEA_M, dtype=np.int16)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            found = tile(x % tiles_across, y)
            if found is not None:
                row, col = (y - y0) * TILE_SIZE, (x - x0) * TILE_SIZE
                heights[row : row + TILE_SIZE, col : col + TILE_SIZE] = found
    return TileElevation(heights, (x0, y0), zoom)
