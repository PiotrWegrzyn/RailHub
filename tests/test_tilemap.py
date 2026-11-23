import pickle
from pathlib import Path

import numpy as np
import pytest

from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import (
    LAYERS,
    Relief,
    Temperature,
    Terrain,
    Tile,
    TileMap,
    Vegetation,
)

BOUNDS = Bounds(west=14.0, east=15.0, south=50.0, north=51.0)
SMALL_GRID = SquareGrid(rows=2, cols=3, km_per_cell=2.0)


def make_map(grid: Grid = SMALL_GRID) -> TileMap:
    index = np.arange(grid.rows * grid.cols).reshape(grid.shape)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=BOUNDS,
        terrain=(index % len(Terrain)).astype(np.uint8),
        vegetation=(index % len(Vegetation)).astype(np.uint8),
        temperature=(index % len(Temperature)).astype(np.uint8),
        height_m=(index * 100).astype(np.int16),
        relief=(index % len(Relief)).astype(np.uint8),
        sources=("Test data",),
    )


def test_saved_map_loads_back_unchanged(tmp_path: Path) -> None:
    tile_map = make_map(HexGrid(rows=4, cols=5, km_per_cell=3.0))
    path = tmp_path / "test.npz"
    tile_map.save(path)

    loaded = TileMap.load(path)
    changed = [
        layer
        for layer in LAYERS
        if not np.array_equal(getattr(loaded, layer), getattr(tile_map, layer))
    ]

    assert (loaded.name, loaded.grid, loaded.bounds) == ("Test", tile_map.grid, BOUNDS)
    assert loaded.sources == ("Test data",)
    assert changed == []


def test_loading_a_map_in_another_format_fails(tmp_path: Path) -> None:
    path = tmp_path / "old.npz"
    np.savez(path, meta=np.array('{"format_version": 0}'))

    with pytest.raises(ValueError, match="format"):
        TileMap.load(path)


def test_tile_gathers_every_property_of_one_cell() -> None:
    tile_map = make_map()

    tile = tile_map.tile((1, 2))

    assert tile == Tile(
        Terrain.SEA, Vegetation.FOREST, Temperature.HOT, height_m=500, relief=Relief.MOUNTAINS
    )


def test_layers_cannot_be_changed() -> None:
    tile_map = make_map()

    with pytest.raises(ValueError):
        tile_map.terrain[0, 0] = Terrain.SEA


def test_layer_that_does_not_fit_the_grid_is_rejected() -> None:
    layers = {layer: np.zeros((3, 3), dtype=np.uint8) for layer in LAYERS}

    with pytest.raises(ValueError, match="grid"):
        TileMap(name="Test", grid=SMALL_GRID, bounds=BOUNDS, **layers)


@pytest.mark.parametrize(
    "grid", [SMALL_GRID, HexGrid(rows=3, cols=4, km_per_cell=2.0)], ids=["square", "hex"]
)
def test_lonlat_of_a_cell_leads_back_to_that_cell(grid: Grid) -> None:
    tile_map = make_map(grid)
    lon, lat = tile_map.lonlat((1, 2))

    cell = tile_map.cell_at_lonlat(lon, lat)

    assert cell == (1, 2)


def test_places_off_the_map_have_no_cell() -> None:
    tile_map = make_map()

    cell = tile_map.cell_at_lonlat(30.0, 60.0)

    assert cell is None


def test_a_map_sent_between_processes_can_still_change_its_vegetation() -> None:
    tile_map = pickle.loads(pickle.dumps(make_map()))

    tile_map.set_vegetation(np.array([0]), Vegetation.FOREST)

    assert tile_map.vegetation.flat[0] == Vegetation.FOREST
    assert not tile_map.vegetation.flags.writeable


def test_the_square_in_the_middle_holds_a_quarter_of_the_land_on_an_even_row() -> None:
    whole = make_map(HexGrid(rows=80, cols=60, km_per_cell=2.0))

    square = whole.square_middle()
    rows, cols = square.grid.shape
    top, left = (80 - rows) // 4 * 2, (60 - cols) // 2

    assert rows * cols == pytest.approx(80 * 60 / 4, rel=0.05)
    assert cols == pytest.approx(rows * 0.866, abs=1)
    assert top % 2 == 0
    assert np.array_equal(square.height_m, whole.height_m[top : top + rows, left : left + cols])


def test_the_square_in_the_middle_keeps_every_place_where_it_was() -> None:
    whole = make_map(HexGrid(rows=80, cols=60, km_per_cell=2.0))
    square = whole.square_middle()

    lon, lat = square.lonlat((10, 10))

    assert whole.cell_at_lonlat(lon, lat) == (
        10 + (80 - square.grid.rows) // 4 * 2,
        10 + (60 - square.grid.cols) // 2,
    )
