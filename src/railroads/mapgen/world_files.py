"""The world's map data for new games in random places and for station names, kept in small files
under world/ (built by tools/build_world_files.py) so a game reads only the few its map needs.

    world/index.npz                 which elevation tiles and squares there are
    world/elevation/{x}_{y}.bin     a zoom-6 Terrarium tile of land heights (terrain_tiles)
    world/shapes/{col}_{row}.npz    Natural Earth sea, lakes and deserts cut to a SQUARE_DEG square,
                                    and every whole river that crosses it
    world/places/{col}_{row}.npz    the GeoNames places in the square (railroads.places)

Squares count columns east from 180° W and rows south from 90° N. The desktop game reads the
files from the world/ folder beside the code or inside the app; the web game, which has no such
folder, downloads each one from world/ beside its page when it is first needed."""

import io
import sys
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt

from railroads.mapgen.presets import NaturalEarth
from railroads.mapgen.sources import Shape
from railroads.mapgen.terrain_tiles import (
    TILE_SIZE,
    HeightTile,
    TileElevation,
    elevation_from_tiles,
)
from railroads.sim.geo import Bounds

if sys.platform == "emscripten":
    import js

SQUARE_DEG = 5.0
INDEX_FILE = "index.npz"
# How far past a map's edges the shapes are still read, in degrees.
SHAPES_MARGIN_DEG = 0.5
# How far a square's shapes reach past its edges, so neighbouring squares overlap and no sliver
# between them is left out.
SQUARE_OVERLAP_DEG = 0.01


def _world_dir() -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    root = Path(bundle) if bundle else Path(__file__).resolve().parents[3]
    return root / "world"


WORLD_DIR = _world_dir()


class WorldFiles(Protocol):
    """Somewhere the world's files are read from by their path under world/."""

    def read(self, name: str) -> bytes | None:
        """The file's bytes, or None if there is no such file."""
        ...


@dataclass(frozen=True)
class FolderFiles:
    folder: Path

    def read(self, name: str) -> bytes | None:
        path = self.folder / name
        return path.read_bytes() if path.is_file() else None


@dataclass(frozen=True)
class WebFiles:
    """The files beside the web page, downloaded as they are asked for. The download waits for
    its answer, which a page allows only for text, so the bytes come as text that keeps each byte
    in its lowest eight bits (the x-user-defined charset)."""

    base_url: str

    def read(self, name: str) -> bytes | None:
        request = js.XMLHttpRequest.new()
        request.open("GET", self.base_url + name, False)
        request.overrideMimeType("text/plain; charset=x-user-defined")
        try:
            request.send()
        except Exception as error:
            raise OSError(f"couldn't download {name}; is the internet connected?") from error
        if request.status == 404:
            return None
        if request.status != 200:
            raise OSError(f"couldn't download {name}: {request.status} {request.statusText}")
        return bytes_from_user_defined(str(request.responseText))


def bytes_from_user_defined(text: str) -> bytes:
    """The bytes a response read as x-user-defined text stands for: each character keeps one in
    its lowest eight bits."""
    codes = np.frombuffer(text.encode("utf-16-le"), dtype="<u2")
    return (codes & 0xFF).astype(np.uint8).tobytes()


@dataclass(frozen=True)
class WorldIndex:
    tiles: frozenset[tuple[int, int]]
    shape_squares: frozenset[tuple[int, int]]
    place_squares: frozenset[tuple[int, int]]

    @classmethod
    def read(cls, data: bytes) -> Self:
        with np.load(io.BytesIO(data)) as arrays:
            return cls(*(_pairs(arrays[key]) for key in ("tiles", "shapes", "places")))

    def write(self) -> bytes:
        out = io.BytesIO()
        np.savez_compressed(
            out,
            tiles=_array(self.tiles),
            shapes=_array(self.shape_squares),
            places=_array(self.place_squares),
        )
        return out.getvalue()


def _pairs(array: npt.NDArray[np.int16]) -> frozenset[tuple[int, int]]:
    return frozenset((int(a), int(b)) for a, b in array.reshape(-1, 2))


def _array(pairs: frozenset[tuple[int, int]]) -> npt.NDArray[np.int16]:
    return np.array(sorted(pairs), dtype=np.int16).reshape(-1, 2)


class World:
    """The world's map data, read from `files` as a map asks for it."""

    def __init__(self, files: WorldFiles) -> None:
        self.files = files

    @cached_property
    def index(self) -> WorldIndex:
        data = self.files.read(INDEX_FILE)
        if data is None:
            raise FileNotFoundError(f"the world's files have no {INDEX_FILE}")
        return WorldIndex.read(data)

    def tile(self, x: int, y: int) -> HeightTile | None:
        if (x, y) not in self.index.tiles:
            return None
        data = self.files.read(tile_file(x, y))
        return None if data is None else unpack_tile(data)

    def elevation_for(self, bounds: Bounds) -> TileElevation:
        return elevation_from_tiles(bounds, self.tile)

    def natural_earth(self, bounds: Bounds) -> NaturalEarth:
        """Sea, lakes, rivers and deserts on and around the bounds."""
        wide = Bounds(
            west=bounds.west - SHAPES_MARGIN_DEG,
            east=bounds.east + SHAPES_MARGIN_DEG,
            south=bounds.south - SHAPES_MARGIN_DEG,
            north=bounds.north + SHAPES_MARGIN_DEG,
        )
        ocean: list[Shape] = []
        lakes: list[Shape] = []
        deserts: list[Shape] = []
        rivers: dict[int, Shape] = {}
        for square in squares_over(wide):
            if square not in self.index.shape_squares:
                continue
            data = self.files.read(shapes_file(*square))
            if data is None:
                continue
            found = read_shapes(data)
            ocean += found.ocean
            lakes += found.lakes
            deserts += found.deserts
            rivers |= dict(zip(found.river_ids, found.rivers, strict=True))
        return NaturalEarth(ocean=ocean, lakes=lakes, rivers=list(rivers.values()), deserts=deserts)

    def places_files(self, bounds: Bounds) -> Iterator[bytes]:
        """The places files of the squares the bounds touch."""
        for square in squares_over(bounds):
            if square in self.index.place_squares:
                data = self.files.read(places_file(*square))
                if data is not None:
                    yield data


@cache
def world() -> World | None:
    """The world's map data: the world/ folder where the game has one, downloaded beside the page
    in a browser, else None."""
    if (WORLD_DIR / INDEX_FILE).is_file():
        return World(FolderFiles(WORLD_DIR))
    if sys.platform == "emscripten":
        return World(WebFiles(str(js.URL.new("world/", js.location.href).href)))
    return None


def tile_file(x: int, y: int) -> str:
    return f"elevation/{x}_{y}.bin"


def shapes_file(col: int, row: int) -> str:
    return f"shapes/{col}_{row}.npz"


def places_file(col: int, row: int) -> str:
    return f"places/{col}_{row}.npz"


def pack_tile(heights: HeightTile) -> bytes:
    return zlib.compress(heights.astype("<i2").tobytes(), 9)


def unpack_tile(data: bytes) -> HeightTile:
    heights = np.frombuffer(zlib.decompress(data), dtype="<i2")
    return heights.reshape(TILE_SIZE, TILE_SIZE).astype(np.int16)


def square_of(lon: float, lat: float) -> tuple[int, int]:
    return int((lon + 180.0) // SQUARE_DEG), int((90.0 - lat) // SQUARE_DEG)


def square_bounds(col: int, row: int) -> Bounds:
    west = -180.0 + col * SQUARE_DEG
    north = 90.0 - row * SQUARE_DEG
    return Bounds(west=west, east=west + SQUARE_DEG, south=north - SQUARE_DEG, north=north)


def squares_over(bounds: Bounds) -> list[tuple[int, int]]:
    """The squares the bounds touch."""
    columns = int(360 / SQUARE_DEG)
    rows = int(180 / SQUARE_DEG)
    west, north = square_of(bounds.west, bounds.north)
    east, south = square_of(bounds.east, bounds.south)
    return [
        (col, row)
        for row in range(max(north, 0), min(south, rows - 1) + 1)
        for col in range(max(west, 0), min(east, columns - 1) + 1)
    ]


@dataclass(frozen=True)
class SquareShapes:
    """One square's sea, lakes and deserts cut to it, and the whole rivers crossing it with the
    number each has across the world, so a river crossing several squares is taken once."""

    ocean: list[Shape]
    lakes: list[Shape]
    deserts: list[Shape]
    rivers: list[Shape]
    river_ids: list[int]


def write_shapes(shapes: SquareShapes) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    for key, found in (
        ("ocean", shapes.ocean),
        ("lakes", shapes.lakes),
        ("deserts", shapes.deserts),
        ("rivers", shapes.rivers),
    ):
        arrays |= _packed(key, found)
    arrays["river_ids"] = np.array(shapes.river_ids, dtype=np.int32)
    arrays["river_names"] = np.array([river.name or "" for river in shapes.rivers], dtype=str)
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


def read_shapes(data: bytes) -> SquareShapes:
    with np.load(io.BytesIO(data)) as arrays:
        names = arrays["river_names"].tolist()
        rivers = [
            Shape("line", river.parts, {"name": name} if name else {})
            for river, name in zip(_unpacked(arrays, "rivers", "line"), names, strict=True)
        ]
        return SquareShapes(
            ocean=_unpacked(arrays, "ocean", "polygon"),
            lakes=_unpacked(arrays, "lakes", "polygon"),
            deserts=_unpacked(arrays, "deserts", "polygon"),
            rivers=rivers,
            river_ids=arrays["river_ids"].tolist(),
        )


def _packed(key: str, shapes: Sequence[Shape]) -> dict[str, np.ndarray]:
    parts = [part for shape in shapes for part in shape.parts]
    points = np.concatenate(parts) if parts else np.zeros((0, 2))
    return {
        f"{key}_points": points.astype(np.float32),
        f"{key}_part_ends": np.cumsum([len(part) for part in parts], dtype=np.int64),
        f"{key}_shape_ends": np.cumsum([len(shape.parts) for shape in shapes], dtype=np.int64),
    }


def _unpacked(arrays: np.lib.npyio.NpzFile, key: str, kind: str) -> list[Shape]:
    points = arrays[f"{key}_points"].astype(np.float64)
    part_ends = arrays[f"{key}_part_ends"].tolist()
    parts = np.split(points, part_ends[:-1]) if part_ends else []
    shapes = []
    start = 0
    for end in arrays[f"{key}_shape_ends"].tolist():
        shapes.append(Shape(kind, tuple(parts[start:end])))  # type: ignore[arg-type]
        start = end
    return shapes
