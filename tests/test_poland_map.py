from collections import deque

import pytest

from railroads import maps
from railroads.sim.grids import Cell
from railroads.sim.tilemap import Temperature, Terrain, TileMap

WARSAW = (21.01, 52.23)


@pytest.fixture(scope="module", params=["square", "hex"])
def poland(request: pytest.FixtureRequest) -> TileMap:
    return maps.load("poland", request.param)


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

    assert 700 < size_km[0] < 800
    assert 650 < size_km[1] < 720


def test_map_file_keeps_its_scale(poland: TileMap) -> None:
    km_per_cell = poland.grid.km_per_cell

    assert km_per_cell == 2.0


@pytest.mark.parametrize(
    ("place", "lon", "lat", "expected"),
    [
        ("Baltic Sea north of Łeba", 17.5, 54.95, Terrain.SEA),
        ("Łódź", 19.46, 51.76, Terrain.LAND),
        ("Żuławy delta, below sea level", 19.12, 54.21, Terrain.LAND),
        ("Szczecin Lagoon", 14.2, 53.75, Terrain.SEA),
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


def test_vistula_carries_water_from_warsaw_to_the_sea(poland: TileMap) -> None:
    river = nearest_cell(poland, cell_at(poland, *WARSAW), Terrain.RIVER)

    reached = water_reachable_from(poland, river)

    assert any(poland.terrain[cell] == Terrain.SEA for cell in reached)


def test_square_map_water_cannot_be_crossed_diagonally() -> None:
    water = maps.load("poland", "square").terrain != Terrain.LAND

    falling = water[:-1, :-1] & water[1:, 1:] & ~water[:-1, 1:] & ~water[1:, :-1]
    rising = water[:-1, 1:] & water[1:, :-1] & ~water[:-1, :-1] & ~water[1:, 1:]

    assert not falling.any()
    assert not rising.any()
