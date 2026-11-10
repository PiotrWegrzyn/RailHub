import numpy as np
import pytest

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.ui.search import BuildingSearch, kinds_matching


def make_growth(*built: tuple[tuple[int, int], BuildingKind]) -> Growth:
    grid = HexGrid(rows=20, cols=20, km_per_cell=2.0)
    tile_map = TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )
    constructions = Constructions(tile_map)
    for cell, kind in built:
        constructions.build(Building(cell, kind))
    return Growth(constructions)


def beekeepers_at(*cells: tuple[int, int]) -> Growth:
    return make_growth(*((cell, BuildingKind.BEEKEEPER) for cell in cells))


def test_an_empty_search_lists_every_kind_of_building() -> None:
    found = kinds_matching("")

    assert set(found) == set(BuildingKind)


def test_a_search_finds_buildings_by_part_of_their_name() -> None:
    found = kinds_matching("cathed")

    assert BuildingKind.CATHEDRAL in found
    assert found[0] in (BuildingKind.CATHEDRAL, BuildingKind.ST_PAULS)


def test_a_search_for_a_good_finds_who_makes_and_wants_it() -> None:
    found = kinds_matching("wax")

    assert BuildingKind.BEEKEEPER in found
    assert BuildingKind.CHURCH in found
    assert BuildingKind.FARM not in found


def test_the_name_matches_come_before_the_goods_matches() -> None:
    found = kinds_matching("wine")

    assert found[0] == BuildingKind.WINERY


def test_a_search_for_nonsense_finds_nothing() -> None:
    assert kinds_matching("zzzz") == []


def test_choosing_a_kind_finds_every_one_nearest_the_view_first() -> None:
    growth = beekeepers_at((1, 1), (10, 10), (18, 18))
    around = growth.constructions.tile_map.grid.center(10, 11)
    search = BuildingSearch()

    search.choose(BuildingKind.BEEKEEPER, growth, (float(around[0]), float(around[1])))

    assert search.cells[0] == (10, 10)
    assert set(search.cells) == {(1, 1), (10, 10), (18, 18)}
    assert search.index is None


def test_stepping_visits_each_building_and_comes_round_again() -> None:
    growth = beekeepers_at((1, 1), (10, 10))
    search = BuildingSearch()
    search.choose(BuildingKind.BEEKEEPER, growth, (0.0, 0.0))

    visited = [search.step(1) for _ in range(3)]

    assert visited == [(1, 1), (10, 10), (1, 1)]


def test_the_first_step_back_goes_to_the_farthest() -> None:
    growth = beekeepers_at((1, 1), (10, 10))
    search = BuildingSearch()
    search.choose(BuildingKind.BEEKEEPER, growth, (0.0, 0.0))

    assert search.step(-1) == (10, 10)


def test_stepping_with_nothing_found_goes_nowhere() -> None:
    search = BuildingSearch()
    search.choose(BuildingKind.CATHEDRAL, make_growth(), (0.0, 0.0))

    assert search.step(1) is None


def test_a_refresh_keeps_the_place_when_others_come_and_go() -> None:
    growth = beekeepers_at((1, 1), (5, 5), (10, 10))
    search = BuildingSearch()
    search.choose(BuildingKind.BEEKEEPER, growth, (0.0, 0.0))
    search.step(1)
    search.step(1)
    growth._take_down(Building((1, 1), BuildingKind.BEEKEEPER))
    growth.constructions.build(Building((15, 15), BuildingKind.BEEKEEPER))
    growth._add((15, 15), BuildingKind.BEEKEEPER, 1)

    search.refresh(growth)

    assert search.current == (5, 5)
    assert search.cells == [(5, 5), (10, 10), (15, 15)]


@pytest.mark.parametrize("taken", [(1, 1), (5, 5)])
def test_a_refresh_with_the_visited_one_gone_stays_on_the_list(taken: tuple[int, int]) -> None:
    growth = beekeepers_at((1, 1), (5, 5))
    search = BuildingSearch()
    search.choose(BuildingKind.BEEKEEPER, growth, (0.0, 0.0))
    search.step(1)
    search.step(1)
    growth._take_down(Building(taken, BuildingKind.BEEKEEPER))

    search.refresh(growth)

    assert search.current is not None
    assert search.current != taken


def test_clearing_forgets_what_was_found() -> None:
    search = BuildingSearch()
    search.choose(BuildingKind.BEEKEEPER, beekeepers_at((1, 1)), (0.0, 0.0))

    search.clear()

    assert search.kind is None
    assert search.cells == []
    assert search.rows.size == 0
