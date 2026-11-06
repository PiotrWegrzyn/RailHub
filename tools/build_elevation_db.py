"""Build the worldwide elevation database (data/elevation.sqlite) that new games read from.

Downloads, once, every zoom-6 Terrarium terrain tile with land between the start region's
latitudes (about 70-90 MB from AWS Open Data) and stores their heights. An interrupted build
picks up where it stopped. Needs the Natural Earth files in data/raw/ (see README.md) to know
where the land is, and Pillow to read the tiles; the game itself needs neither.

Run with: uv run python tools/build_elevation_db.py
"""

import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from railroads.mapgen.area import MapArea
from railroads.mapgen.presets import NaturalEarth
from railroads.mapgen.terrain_tiles import (
    TILE_SIZE,
    TILE_URL,
    ZOOM,
    ElevationDatabase,
    HeightTile,
    decode_terrarium,
    world_pixel,
)
from railroads.newgame import DATA_DIR, ELEVATION_DB, START_REGION
from railroads.sim.geo import Bounds
from railroads.sim.grids import SquareGrid

LAND_CELL_KM = 50.0
DOWNLOADS_AT_ONCE = 8
ATTEMPTS = 3


def land_tiles(world: NaturalEarth) -> set[tuple[int, int]]:
    """Tiles holding any land between the start region's latitudes, found on a coarse land grid."""
    band = Bounds(-180.0, 180.0, START_REGION.south - 1, START_REGION.north + 1)
    grid = SquareGrid.covering(band.width_km, band.height_km, LAND_CELL_KM)
    area = MapArea.of(band, grid)
    land = area.cover(world.ocean + world.lakes) < 1.0
    lon, lat = (values[land] for values in area.center_lonlat())
    half_lon = (band.east - band.west) / grid.cols / 2
    half_lat = (band.north - band.south) / grid.rows / 2
    tiles_across = 2**ZOOM
    tiles: set[tuple[int, int]] = set()
    for d_lon in (-half_lon, 0.0, half_lon):
        for d_lat in (-half_lat, 0.0, half_lat):
            x, y = world_pixel(lon + d_lon, lat + d_lat)
            columns = (x // TILE_SIZE).astype(int) % tiles_across
            rows = (y // TILE_SIZE).astype(int)
            tiles |= set(zip(columns.tolist(), rows.tolist(), strict=True))
    return {(x, y) for x, y in tiles if 0 <= y < tiles_across}


def fetch_tile(x: int, y: int) -> HeightTile:
    url = TILE_URL.format(z=ZOOM, x=x, y=y)
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                image = Image.open(io.BytesIO(response.read())).convert("RGB")
            return decode_terrarium(np.asarray(image))
        except OSError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    wanted = land_tiles(NaturalEarth.load(DATA_DIR / "raw"))
    database = ElevationDatabase(ELEVATION_DB)
    missing = sorted(wanted - database.stored())
    print(f"{len(wanted)} tiles hold land, {len(missing)} still to download")
    failed = 0
    with ThreadPoolExecutor(DOWNLOADS_AT_ONCE) as pool:
        downloads = {pool.submit(fetch_tile, x, y): (x, y) for x, y in missing}
        for done, download in enumerate(as_completed(downloads), start=1):
            x, y = downloads[download]
            try:
                database.put(x, y, download.result())
            except OSError as error:
                failed += 1
                print(f"tile {x}/{y} failed: {error}")
            if done % 100 == 0 or done == len(missing):
                print(f"{done}/{len(missing)} tiles", flush=True)
    database.close()
    print(f"{ELEVATION_DB} is {ELEVATION_DB.stat().st_size / 1e6:.0f} MB")
    if failed:
        raise SystemExit(f"{failed} tiles failed; run the tool again to retry them")


if __name__ == "__main__":
    main()
