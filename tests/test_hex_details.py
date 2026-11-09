import numpy as np

from railroads.sim.constructions import Building, BuildingKind, Constructions, Station
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.ui.hex_details import describe


def make_map() -> TileMap:
    grid = HexGrid(rows=10, cols=10, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=terrain,
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )


def test_details_list_each_kind_of_building_with_what_it_makes_needs_and_traded() -> None:
    constructions = Constructions(make_map())
    for _ in range(3):
        constructions.build(Building((5, 5), BuildingKind.HOUSE))
    constructions.build(Building((5, 5), BuildingKind.FARM))
    constructions.build(Station((5, 6)))
    growth = Growth(constructions)

    details = describe(growth, (5, 5))

    rows = {row.name: row for row in details.buildings}
    assert rows["house"].count == 3
    assert rows["house"].needs == "3 food"
    assert rows["house"].got == "food 3.0 of 3.0"
    assert rows["house"].satisfaction == "satisfied"
    assert rows["farm"].makes == "6 food"
    assert rows["farm"].gave.startswith("food ")
    assert details.free_food == 1
    assert ("population", 3, 3) in details.points
    assert ("food", 1, 1) in details.points


def test_details_of_an_empty_cell_show_its_land_and_nearby_points() -> None:
    constructions = Constructions(make_map())
    constructions.build(Station((5, 6)))
    growth = Growth(constructions)

    details = describe(growth, (5, 6))

    assert details.buildings == []
    assert details.others == ["station"]
    assert details.satisfaction == "nothing trades here"
    assert details.title.startswith("Cell 5, 6")


def test_a_building_needing_two_goods_lists_each_with_what_it_got() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.TENEMENT))
    growth = Growth(constructions)

    details = describe(growth, (5, 5))

    tenement = details.buildings[0]
    assert tenement.needs == "2 food, 1 goods"
    assert tenement.got == "food 2.0 of 2.0\ngoods 0.0 of 1.0"


def test_a_factory_without_lumber_makes_nothing_and_shows_what_it_could() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.FACTORY))
    growth = Growth(constructions)

    details = describe(growth, (5, 5))

    assert details.buildings[0].gave == "goods 0.0 of 0.0 (up to 3)"
