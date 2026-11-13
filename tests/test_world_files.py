import numpy as np
import pytest

from builders import MemoryFiles
from railroads.mapgen.sources import Shape
from railroads.mapgen.terrain_tiles import OPEN_SEA_M, TILE_SIZE, world_lonlat
from railroads.mapgen.world import WorldLand
from railroads.mapgen.world_files import (
    INDEX_FILE,
    SquareShapes,
    World,
    WorldIndex,
    bytes_from_user_defined,
    pack_tile,
    read_shapes,
    shapes_file,
    squares_over,
    tile_file,
    write_shapes,
)
from railroads.sim.geo import Bounds

POLAND_TILE = (35, 21)
SQUARE = Shape("polygon", (np.array([(10.0, 50.0), (12.0, 50.0), (12.0, 52.0), (10.0, 52.0)]),))


def river(*points: tuple[float, float], name: str = "Vistula") -> Shape:
    return Shape("line", (np.array(points),), {"name": name})


def world_of(files: dict[str, bytes], index: WorldIndex) -> tuple[World, MemoryFiles]:
    memory = MemoryFiles(files | {INDEX_FILE: index.write()})
    return World(memory), memory


def tile_bounds(first: float, last: float) -> Bounds:
    x0, y0 = POLAND_TILE[0] * TILE_SIZE, POLAND_TILE[1] * TILE_SIZE
    west, north = (float(v) for v in world_lonlat(x0 + first, y0 + first))
    east, south = (float(v) for v in world_lonlat(x0 + last, y0 + last))
    return Bounds(west=west, east=east, south=south, north=north)


def test_elevation_comes_from_the_stored_tile_and_sea_elsewhere() -> None:
    heights = np.full((TILE_SIZE, TILE_SIZE), 250, dtype=np.int16)
    world, files = world_of(
        {tile_file(*POLAND_TILE): pack_tile(heights)},
        WorldIndex(frozenset({POLAND_TILE}), frozenset(), frozenset()),
    )
    bounds = tile_bounds(200, 300)
    inside_lon, inside_lat = world_lonlat(
        POLAND_TILE[0] * TILE_SIZE + 210, POLAND_TILE[1] * TILE_SIZE + 210
    )
    outside_lon, outside_lat = world_lonlat(
        POLAND_TILE[0] * TILE_SIZE + 290, POLAND_TILE[1] * TILE_SIZE + 290
    )

    elevation = world.elevation_for(bounds)

    assert elevation.sample(inside_lon, inside_lat) == pytest.approx(250)
    assert elevation.sample(outside_lon, outside_lat) == OPEN_SEA_M
    assert files.asked == {INDEX_FILE, tile_file(*POLAND_TILE)}


def test_square_shapes_read_back_unchanged() -> None:
    shapes = SquareShapes(
        ocean=[SQUARE], lakes=[], deserts=[SQUARE], rivers=[river((10.0, 50.0), (11.0, 51.0))],
        river_ids=[7],
    )  # fmt: skip

    again = read_shapes(write_shapes(shapes))

    assert np.allclose(again.ocean[0].parts[0], SQUARE.parts[0])
    assert again.lakes == []
    assert len(again.deserts) == 1
    assert again.rivers[0].name == "Vistula"
    assert again.river_ids == [7]


def test_a_river_crossing_two_squares_is_taken_once() -> None:
    vistula = river((14.0, 50.0), (16.0, 50.0))
    one = SquareShapes(ocean=[], lakes=[], deserts=[], rivers=[vistula], river_ids=[3])
    two = SquareShapes(ocean=[SQUARE], lakes=[], deserts=[], rivers=[vistula], river_ids=[3])
    squares = squares_over(Bounds(west=14.0, east=16.0, south=49.0, north=51.0))
    world, _ = world_of(
        {shapes_file(*squares[0]): write_shapes(one), shapes_file(*squares[1]): write_shapes(two)},
        WorldIndex(frozenset(), frozenset(squares), frozenset()),
    )

    found = world.natural_earth(Bounds(west=14.5, east=15.5, south=49.5, north=50.5))

    assert len(found.rivers) == 1
    assert len(found.ocean) == 1


def test_squares_over_bounds_are_the_five_degree_squares_they_touch() -> None:
    bounds = Bounds(west=14.0, east=16.0, south=49.0, north=51.0)

    squares = squares_over(bounds)

    assert squares == [(38, 7), (39, 7), (38, 8), (39, 8)]


def test_text_read_as_user_defined_gives_back_the_bytes() -> None:
    sent = bytes(range(256))
    text = "".join(chr(b) if b < 0x80 else chr(0xF700 + b) for b in sent)

    received = bytes_from_user_defined(text)

    assert received == sent


def test_land_share_is_the_land_inside_the_bounds() -> None:
    land = np.zeros((720, 1440), dtype=np.uint8)
    land[:, 720:] = 255
    world = WorldLand(land)

    shares = [
        world.land_share(Bounds(west=10.0, east=14.0, south=40.0, north=44.0)),
        world.land_share(Bounds(west=-14.0, east=-10.0, south=40.0, north=44.0)),
    ]

    assert shares == [1.0, 0.0]
