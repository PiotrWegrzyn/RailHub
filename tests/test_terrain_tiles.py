from pathlib import Path

import numpy as np
import pytest

from railroads.mapgen.terrain_tiles import (
    OPEN_SEA_M,
    TILE_SIZE,
    ElevationDatabase,
    decode_terrarium,
    world_lonlat,
    world_pixel,
)
from railroads.sim.geo import Bounds

POLAND_TILE = (35, 21)


def tile_point(x: float, y: float) -> tuple[float, float]:
    lon, lat = world_lonlat(x, y)
    return float(lon), float(lat)


def rising_east_tile() -> np.ndarray:
    return np.tile(np.arange(TILE_SIZE, dtype=np.int16) * 10, (TILE_SIZE, 1))


@pytest.mark.parametrize(
    ("pixel", "metres"),
    [((128, 0, 0), 0), ((128, 100, 0), 100), ((127, 255, 0), -1), ((130, 0, 128), 512)],
)
def test_terrarium_colours_decode_to_metres(pixel: tuple[int, int, int], metres: int) -> None:
    rgb = np.array([[pixel]], dtype=np.uint8)

    heights = decode_terrarium(rgb)

    assert heights[0, 0] == metres


def test_world_pixel_and_lonlat_are_inverse() -> None:
    x, y = world_pixel(21.01, 52.23)

    lon, lat = world_lonlat(x, y)

    assert (float(lon), float(lat)) == pytest.approx((21.01, 52.23))


def test_warsaw_lies_in_its_zoom_6_tile() -> None:
    x, y = world_pixel(21.01, 52.23)

    tile = (int(x // TILE_SIZE), int(y // TILE_SIZE))

    assert tile == POLAND_TILE


def test_stored_tile_reads_back_unchanged(tmp_path: Path) -> None:
    database = ElevationDatabase(tmp_path / "elevation.sqlite")
    heights = rising_east_tile()
    database.put(*POLAND_TILE, heights)

    stored, again = database.stored(), database.get(*POLAND_TILE)

    assert stored == {POLAND_TILE}
    assert np.array_equal(again, heights)


def test_elevation_is_read_between_the_stored_pixels(tmp_path: Path) -> None:
    database = ElevationDatabase(tmp_path / "elevation.sqlite")
    database.put(*POLAND_TILE, rising_east_tile())
    x0, y0 = POLAND_TILE[0] * TILE_SIZE, POLAND_TILE[1] * TILE_SIZE
    west, north = tile_point(x0 + 64, y0 + 64)
    east, south = tile_point(x0 + 192, y0 + 192)
    bounds = Bounds(west=west, east=east, south=south, north=north)

    elevation = database.elevation_for(bounds)

    assert elevation.covers(bounds)
    assert elevation.sample(*tile_point(x0 + 100.5, y0 + 100.5)) == pytest.approx(1000)


def test_places_without_a_stored_tile_are_open_sea(tmp_path: Path) -> None:
    database = ElevationDatabase(tmp_path / "elevation.sqlite")
    database.put(*POLAND_TILE, rising_east_tile())
    x0, y0 = POLAND_TILE[0] * TILE_SIZE, POLAND_TILE[1] * TILE_SIZE
    west, north = tile_point(x0 + 200, y0 + 100)
    east, south = tile_point(x0 + 300, y0 + 150)

    elevation = database.elevation_for(Bounds(west=west, east=east, south=south, north=north))

    assert elevation.sample(*tile_point(x0 + 290, y0 + 120)) == OPEN_SEA_M
