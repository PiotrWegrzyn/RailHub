from pathlib import Path

import numpy as np

from railroads.places import Places, StationNames
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

BOUNDS = Bounds(west=19.0, east=20.0, south=51.5, north=52.0)


def make_map() -> TileMap:
    grid = HexGrid(rows=20, cols=30, km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=BOUNDS,
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )


def places_around(tile_map: TileMap, *towns: tuple[str, tuple[int, int], int]) -> Places:
    names, lons, lats, people = [], [], [], []
    for name, cell, population in towns:
        lon, lat = tile_map.lonlat(cell)
        names.append(name)
        lons.append(lon)
        lats.append(lat)
        people.append(population)
    return Places(tuple(names), np.array(lons), np.array(lats), np.array(people))


def test_a_station_takes_the_name_of_the_town_it_stands_in() -> None:
    tile_map = make_map()
    places = places_around(tile_map, ("Łódź", (10, 10), 650_000), ("Zgierz", (10, 18), 55_000))
    names = StationNames(tile_map, places)

    named = [names((10, 11)), names((10, 18)), names((10, 12))]

    assert named == ["Łódź", "Zgierz", "Łódź [2]"]


def test_a_station_outside_any_town_takes_the_closest_place_within_fifty_km() -> None:
    tile_map = make_map()
    places = places_around(tile_map, ("Wola", (2, 2), 600), ("Góra", (2, 12), 600))
    names = StationNames(tile_map, places)

    named = [names((15, 25)), names((15, 5)), names((2, 2))]

    assert named == ["Góra", "Wola", "Wola [2]"]


def test_a_station_with_no_place_within_fifty_km_is_numbered() -> None:
    tile_map = make_map()
    places = places_around(tile_map, ("Wola", (2, 2), 600))
    far = Places(places.names, places.lon + 2.0, places.lat, places.population)
    names = StationNames(tile_map, far)

    named = [names((15, 25)), names((2, 2))]

    assert named == ["Station 1", "Station 2"]


def test_without_place_names_every_station_is_numbered() -> None:
    names = StationNames(make_map(), None)

    named = [names((1, 1)), names((5, 5)), names((1, 1))]

    assert named == ["Station 1", "Station 2", "Station 1"]


def test_a_station_in_two_towns_reaches_takes_the_one_it_is_deeper_in() -> None:
    tile_map = make_map()
    places = places_around(tile_map, ("City", (10, 10), 1_000_000), ("Suburb", (10, 13), 600))

    town = places.town_at(*tile_map.lonlat((10, 11)))

    assert town == "City"


def test_places_load_only_around_the_map_and_none_without_the_file(tmp_path: Path) -> None:
    names = ["Łódź", "Paris"]
    encoded = [name.encode() for name in names]
    path = tmp_path / "places.npz"
    np.savez_compressed(
        path,
        names=np.frombuffer(b"".join(encoded), dtype=np.uint8),
        ends=np.cumsum([len(e) for e in encoded]),
        lat=np.array([51.76, 48.85], dtype=np.float32),
        lon=np.array([19.46, 2.35], dtype=np.float32),
        population=np.array([650_000, 2_100_000]),
    )

    loaded = Places.load(BOUNDS, path)
    missing = Places.load(BOUNDS, tmp_path / "none.npz")

    assert loaded is not None
    assert loaded.names == ("Łódź",)
    assert missing is None
