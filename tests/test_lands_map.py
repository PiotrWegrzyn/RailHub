import numpy as np
import numpy.typing as npt
import pytest

from railroads import maps
from railroads.mapgen.lands import BANDS, COVERS, lands_map, square
from railroads.newgame import EARLIEST_START_YEAR, MapChoice, Settings, make_game
from railroads.sim.tilemap import LAYERS, Relief, Temperature, Terrain, TileMap, Vegetation

LANDS = [(band, cover) for band in BANDS for cover in COVERS]


@pytest.fixture(scope="module")
def place() -> TileMap:
    return maps.load("lands", "hex")


def beside(tile_map: TileMap, cells: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    rows, cols = tile_map.grid.neighbour_index_grids()
    return cells[rows, cols].any(axis=0)


def in_land(tile_map: TileMap, band: Temperature, cover: Vegetation) -> npt.NDArray[np.bool_]:
    return (
        square(tile_map.grid)
        & (tile_map.terrain == Terrain.LAND)
        & (tile_map.temperature == band)
        & (tile_map.vegetation == cover)
    )


def test_the_baked_map_is_the_one_the_maker_draws(place: TileMap) -> None:
    drawn = lands_map()

    assert all((getattr(place, layer) == getattr(drawn, layer)).all() for layer in LAYERS)


def test_each_land_holds_about_a_ninth_of_the_square(place: TileMap) -> None:
    sizes = [int(in_land(place, band, cover).sum()) for band, cover in LANDS]

    assert min(sizes) > 0.9 * max(sizes)


@pytest.mark.parametrize(("band", "cover"), LANDS, ids=lambda x: x.name)
def test_each_land_has_plains_hills_mountains_and_every_kind_of_water_beside_it(
    place: TileMap, band: Temperature, cover: Vegetation
) -> None:
    land = in_land(place, band, cover)
    size = int(land.sum())

    features = {
        "plains": land & (place.relief == Relief.FLAT),
        "hills": land & (place.relief == Relief.HILLS),
        "mountains": land & (place.relief == Relief.MOUNTAINS),
        "by a river": land & beside(place, place.terrain == Terrain.RIVER),
        "by a lake": land & beside(place, place.terrain == Terrain.LAKE),
        "by the sea": land & beside(place, place.terrain == Terrain.SEA),
        "on a minor river": land & (place.stream > 0),
    }

    assert {
        name: int(cells.sum()) > 0.005 * size for name, cells in features.items()
    } == dict.fromkeys(features, True)


def test_a_game_starts_on_the_map_of_every_land() -> None:
    settings = Settings(seed=1, map=MapChoice.LANDS, start_year=EARLIEST_START_YEAR)

    saved = make_game(settings, report=lambda text: None)

    assert saved.game.growth.population() > 0
