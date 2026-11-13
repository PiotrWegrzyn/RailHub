"""Cut the world's map data into the small files under world/ that new games read (see
railroads.mapgen.world_files): every tile of data/elevation.sqlite, and the Natural Earth shapes
in data/raw/ and the places in data/places.npz square by square, for the squares between
LOWEST_LAT and HIGHEST_LAT, which hold every map the start region allows with room to spare.

Needs the outputs of tools/build_elevation_db.py and tools/build_places.py and the Natural Earth
downloads (see README.md). Rewrites world/ from scratch.

Run with: uv run python tools/build_world_files.py [--data DIR]
"""

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from railroads.mapgen import world_files
from railroads.mapgen.area import clip_ring
from railroads.mapgen.presets import NaturalEarth
from railroads.mapgen.sources import Shape
from railroads.mapgen.terrain_tiles import ElevationDatabase
from railroads.mapgen.world_files import (
    SQUARE_DEG,
    SQUARE_OVERLAP_DEG,
    SquareShapes,
    WorldIndex,
    pack_tile,
    places_file,
    shapes_file,
    square_bounds,
    square_of,
    tile_file,
    write_shapes,
)
from railroads.newgame import DATA_DIR
from railroads.places import Places
from railroads.sim.geo import Bounds

LOWEST_LAT = -60.0
HIGHEST_LAT = 70.0


def cut_polygons(shapes: Sequence[Shape], boxes: Sequence[np.ndarray], area: Bounds) -> list[Shape]:
    """The polygons cut down to the area; `boxes` holds each shape's rings' lon/lat limits."""
    cut = []
    for shape, rings in zip(shapes, boxes, strict=True):
        near = (
            (rings[:, 0] <= area.east)
            & (rings[:, 2] >= area.west)
            & (rings[:, 1] <= area.north)
            & (rings[:, 3] >= area.south)
        )
        if not near.any():
            continue
        parts = tuple(
            part
            for part, keep in zip(shape.parts, near, strict=True)
            if keep
            for part in [clip_ring(part, area.west, area.south, area.east, area.north)]
            if len(part) >= 3
        )
        if parts:
            cut.append(Shape("polygon", parts))
    return cut


def ring_boxes(shapes: Sequence[Shape]) -> list[np.ndarray]:
    """For each shape, each part's west, south, east and north."""
    return [
        np.array([[*part.min(axis=0), *part.max(axis=0)] for part in shape.parts])
        for shape in shapes
    ]


def overlapping(bounds: Bounds) -> Bounds:
    return Bounds(
        west=bounds.west - SQUARE_OVERLAP_DEG,
        east=bounds.east + SQUARE_OVERLAP_DEG,
        south=bounds.south - SQUARE_OVERLAP_DEG,
        north=bounds.north + SQUARE_OVERLAP_DEG,
    )


def write_tiles(database: ElevationDatabase, out: Path) -> set[tuple[int, int]]:
    tiles = database.stored()
    (out / "elevation").mkdir(parents=True)
    for x, y in sorted(tiles):
        heights = database.get(x, y)
        assert heights is not None
        (out / tile_file(x, y)).write_bytes(pack_tile(heights))
    return tiles


def write_shape_squares(
    world: NaturalEarth, squares: list[tuple[int, int]], out: Path
) -> set[tuple[int, int]]:
    (out / "shapes").mkdir(parents=True)
    layers = {"ocean": world.ocean, "lakes": world.lakes, "deserts": world.deserts}
    boxes = {key: ring_boxes(shapes) for key, shapes in layers.items()}
    river_boxes = [box[:, :2].min(axis=0).tolist() + box[:, 2:].max(axis=0).tolist()
                   for box in ring_boxes(world.rivers)]  # fmt: skip
    written = set()
    for col, row in squares:
        area = overlapping(square_bounds(col, row))
        cut = {key: cut_polygons(layers[key], boxes[key], area) for key in layers}
        river_ids = [
            index
            for index, (west, south, east, north) in enumerate(river_boxes)
            if west <= area.east and east >= area.west and south <= area.north
            and north >= area.south
        ]  # fmt: skip
        if not (any(cut.values()) or river_ids):
            continue
        found = SquareShapes(
            ocean=cut["ocean"],
            lakes=cut["lakes"],
            deserts=cut["deserts"],
            rivers=[world.rivers[index] for index in river_ids],
            river_ids=river_ids,
        )
        (out / shapes_file(col, row)).write_bytes(write_shapes(found))
        written.add((col, row))
    return written


def write_place_squares(places_path: Path, out: Path) -> set[tuple[int, int]]:
    (out / "places").mkdir(parents=True)
    everywhere = Bounds(west=-180.0, east=180.0, south=-90.0, north=90.0)
    places = Places.read(places_path.read_bytes(), everywhere)
    squares = np.array(
        [square_of(lon, lat) for lon, lat in zip(places.lon, places.lat, strict=True)]
    )
    written = set()
    for col, row in {tuple(square) for square in squares.tolist()}:
        if not LOWEST_LAT <= square_bounds(col, row).south < HIGHEST_LAT:
            continue
        kept = np.flatnonzero((squares[:, 0] == col) & (squares[:, 1] == row))
        part = Places(
            tuple(places.names[i] for i in kept.tolist()),
            places.lon[kept],
            places.lat[kept],
            places.population[kept],
        )
        (out / places_file(col, row)).write_bytes(part.write())
        written.add((col, row))
    return written


def size_mb(folder: Path) -> float:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="the source data folder")
    data = parser.parse_args().data
    out = world_files.WORLD_DIR
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    database = ElevationDatabase(data / "elevation.sqlite")
    try:
        tiles = write_tiles(database, out)
    finally:
        database.close()
    print(f"{len(tiles)} elevation tiles, {size_mb(out / 'elevation'):.1f} MB")
    first_row = int((90.0 - HIGHEST_LAT) // SQUARE_DEG)
    last_row = int((90.0 - LOWEST_LAT) // SQUARE_DEG) - 1
    squares = [
        (col, row) for row in range(first_row, last_row + 1) for col in range(int(360 / SQUARE_DEG))
    ]
    shapes = write_shape_squares(NaturalEarth.load(data / "raw"), squares, out)
    print(f"{len(shapes)} squares of shapes, {size_mb(out / 'shapes'):.1f} MB")
    places = write_place_squares(data / "places.npz", out)
    print(f"{len(places)} squares of places, {size_mb(out / 'places'):.1f} MB")
    index = WorldIndex(frozenset(tiles), frozenset(shapes), frozenset(places))
    (out / world_files.INDEX_FILE).write_bytes(index.write())
    print(f"{out} is {size_mb(out):.1f} MB")


if __name__ == "__main__":
    main()
