from collections import deque

import pytest

from railroads import maps
from railroads.sim.grids import Cell
from railroads.sim.lots import Lots
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap

WARSAW = (21.01, 52.23)
KRAKOW = (19.94, 50.06)


@pytest.fixture(scope="module")
def poland() -> TileMap:
    return maps.load("poland", "hex")


def cell_at(tile_map: TileMap, lon: float, lat: float) -> Cell:
    cell = tile_map.cell_at_lonlat(lon, lat)
    assert cell is not None
    return cell


def nearest_cell(tile_map: TileMap, start: Cell, terrain: Terrain) -> Cell:
    seen, queue = {start}, deque([start])
    while queue:
        cell = queue.popleft()
        if tile_map.terrain[cell] == terrain:
            return cell
        for neighbour in tile_map.grid.neighbours(*cell):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    raise AssertionError(f"no {terrain.name} on the map")


def water_reachable_from(tile_map: TileMap, start: Cell) -> set[Cell]:
    seen, queue = {start}, deque([start])
    while queue:
        for neighbour in tile_map.grid.neighbours(*queue.popleft()):
            if neighbour not in seen and tile_map.terrain[neighbour] != Terrain.LAND:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen


def test_map_covers_poland(poland: TileMap) -> None:
    width, height = poland.grid.size()

    size_km = (width * poland.grid.km_per_cell, height * poland.grid.km_per_cell)

    assert 680 < size_km[0] < 710
    assert 640 < size_km[1] < 670


@pytest.mark.parametrize(
    ("place", "lon", "lat", "expected"),
    [
        ("Łódź", 19.46, 51.76, Terrain.LAND),
        ("Vistula at Kraków", *KRAKOW, Terrain.RIVER),
        ("Szczecin Lagoon", 14.3, 53.8, Terrain.SEA),
        ("Śniardwy lake", 21.73, 53.75, Terrain.LAKE),
    ],
)
def test_known_places_have_their_terrain(
    poland: TileMap, place: str, lon: float, lat: float, expected: Terrain
) -> None:
    tile = poland.tile(cell_at(poland, lon, lat))

    assert tile.terrain is expected, place


def test_tatras_are_cold(poland: TileMap) -> None:
    tile = poland.tile(cell_at(poland, 20.0, 49.23))

    assert tile.temperature is Temperature.COLD


def test_warsaw_lies_in_the_lowlands(poland: TileMap) -> None:
    tile = poland.tile(cell_at(poland, *WARSAW))

    assert 60 < tile.height_m < 150
    assert tile.relief is Relief.FLAT


def test_tatras_are_mountains(poland: TileMap) -> None:
    tile = poland.tile(cell_at(poland, 20.0, 49.23))

    assert tile.relief is Relief.MOUNTAINS


def test_most_of_poland_is_flat(poland: TileMap) -> None:
    land = poland.terrain == Terrain.LAND

    flat_share = (poland.relief[land] == Relief.FLAT).mean()

    assert flat_share > 0.8


def test_vistula_carries_water_from_krakow_to_warsaw(poland: TileMap) -> None:
    upstream = nearest_cell(poland, cell_at(poland, *KRAKOW), Terrain.RIVER)
    downstream = nearest_cell(poland, cell_at(poland, *WARSAW), Terrain.RIVER)

    reached = water_reachable_from(poland, upstream)

    assert downstream in reached


def test_start_reaches_the_mountains(poland: TileMap) -> None:
    lots = Lots.around_middle(poland)
    top, left, rows, cols = window = lots.window()

    start = lots.in_lots(window, lots.owned)

    relief = poland.relief[top : top + rows, left : left + cols][start]
    assert (relief == Relief.MOUNTAINS).sum() > 100
