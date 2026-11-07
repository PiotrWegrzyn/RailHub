import numpy as np
import pytest

from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Economy, Satisfaction, free_food, levels_of
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.growth import LUMBER_MILLS, MINES
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation


def make_map(size: int = 12) -> TileMap:
    grid = HexGrid(rows=size, cols=size, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    relief[:, -3:] = Relief.HILLS
    relief[:, -1] = Relief.MOUNTAINS
    vegetation = np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
    vegetation[:6, 4:8] = Vegetation.FOREST
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=terrain,
        vegetation=vegetation,
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=relief,
    )


def counts(tile_map: TileMap, *placed: tuple[tuple[int, int], BuildingKind, int]) -> dict:
    grids = {kind: np.zeros(tile_map.grid.shape, dtype=np.int64) for kind in BuildingKind}
    for cell, kind, n in placed:
        grids[kind][cell] += n
    return grids


def level_at(economy: Economy, trade: object, cell: tuple[int, int], kind: BuildingKind) -> int:
    index = np.ravel_multi_index(cell, economy.tile_map.grid.shape)
    position = int(np.searchsorted(trade.cells, index))  # type: ignore[attr-defined]
    return int(trade.levels((kind,))[position])  # type: ignore[attr-defined]


def test_water_feeds_two_and_flat_land_and_forest_one() -> None:
    tile_map = make_map()

    food = free_food(tile_map).reshape(tile_map.grid.shape)

    assert food[5, 0] == 2
    assert food[8, 3] == 1
    assert food[2, 5] == 1
    assert food[8, 9] == 0


def test_a_house_on_open_land_is_fed_by_the_land_around_it() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)

    trade = economy.trade(counts(tile_map, ((8, 3), BuildingKind.HOUSE, 3)))

    assert level_at(economy, trade, (8, 3), BuildingKind.HOUSE) == Satisfaction.SATISFIED


def test_a_crowded_town_goes_hungry_without_farms() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)

    trade = economy.trade(counts(tile_map, ((8, 3), BuildingKind.FLATS, 6)))

    assert level_at(economy, trade, (8, 3), BuildingKind.FLATS) == Satisfaction.UNSATISFIED


def test_a_farm_that_feeds_half_what_it_grows_is_content() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    hungry = counts(tile_map, ((8, 9), BuildingKind.FARM, 1), ((8, 10), BuildingKind.HOUSE, 3))

    trade = economy.trade(hungry)

    assert level_at(economy, trade, (8, 9), BuildingKind.FARM) == Satisfaction.CONTENT


def test_a_factory_makes_only_as_much_as_its_lumber_allows() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(
        tile_map,
        ((8, 3), BuildingKind.FACTORY, 1),
        ((8, 4), BuildingKind.LUMBER_MILL, 0),
        ((8, 2), BuildingKind.TENEMENT, 6),
    )

    without = economy.trade(placed)
    placed[BuildingKind.LUMBER_MILL][8, 4] = 1
    with_lumber = Economy(tile_map).trade(placed)

    index = int(np.searchsorted(without.cells, np.ravel_multi_index((8, 3), tile_map.grid.shape)))
    assert without.gave[BuildingKind.FACTORY][index] == 0
    index = int(
        np.searchsorted(with_lumber.cells, np.ravel_multi_index((8, 3), tile_map.grid.shape))
    )
    assert with_lumber.gave[BuildingKind.FACTORY][index] == pytest.approx(3)


@pytest.mark.parametrize(
    ("share", "level"),
    [(0.2, Satisfaction.UNSATISFIED), (0.5, Satisfaction.CONTENT), (1.0, Satisfaction.SATISFIED)],
)
def test_satisfaction_goes_by_the_share_met(share: float, level: Satisfaction) -> None:
    levels = levels_of(np.array([share]))

    assert levels[0] == level


def test_lumber_mills_only_grow_at_the_edge_of_a_forest() -> None:
    weights = LUMBER_MILLS.land.weights(make_map())

    assert weights[2, 4] > 0
    assert weights[2, 5] == 0


def test_mines_like_hills_beside_mountains_best() -> None:
    weights = MINES.land.weights(make_map())

    assert weights[8, 10] > 2 * weights[8, 9]
