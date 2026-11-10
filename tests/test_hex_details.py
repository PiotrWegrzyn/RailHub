import numpy as np

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions, Station
from railroads.sim.goods import Good, Need
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.tilemap import Terrain, TileMap
from railroads.ui.hex_details import Flow, describe


def make_map() -> TileMap:
    grid = HexGrid(rows=10, cols=10, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    return flat_map(grid=grid, terrain=terrain)


def test_details_list_each_kind_of_building_with_what_it_makes_needs_and_traded() -> None:
    constructions = Constructions(make_map())
    for _ in range(3):
        constructions.build(Building((5, 5), BuildingKind.HOUSE))
    constructions.build(Building((5, 4), BuildingKind.FARM))
    constructions.build(Station((5, 6)))
    growth = Growth(constructions, GrowthRule(trades=True, ladders=()))

    details = describe(growth, (5, 5))
    farm = describe(growth, (5, 4)).buildings[0]

    (house,) = details.buildings
    assert house.name == "house"
    assert house.count == 3
    assert house.needs == [(Need.FOOD, 30), (Need.BEER, 3), (Need.NEWSPAPERS, 3), (Need.HEATING, 3)]
    assert house.got == [
        Flow(Need.NEWSPAPERS, 0.0, 3.0),
        Flow(Need.HEATING, 0.0, 3.0),
        Flow(Need.BEER, 0.0, 3.0),
        Flow(Need.FOOD, 0.0, 30.0),
    ]
    assert house.satisfaction == "unsatisfied"
    assert farm.makes == [(Good.WHEAT, 5)]
    assert farm.gave == [Flow(Good.WHEAT, 0.0, 5.0)]
    assert ("population", 3, 3) in details.points
    assert ("food", 0, 1) in details.points


def test_details_of_an_empty_cell_show_its_land_and_nearby_points() -> None:
    constructions = Constructions(make_map())
    constructions.build(Station((5, 6)))
    growth = Growth(constructions)

    details = describe(growth, (5, 6))

    assert details.buildings == []
    assert details.others == ["station"]
    assert details.satisfaction == "nothing trades here"
    assert details.title.startswith("Cell 5, 6")


def test_a_building_needing_several_goods_lists_each_with_what_it_got() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.TENEMENT))
    growth = Growth(constructions)

    details = describe(growth, (5, 5))

    tenement = details.buildings[0]
    assert tenement.needs == [
        (Need.FOOD, 60),
        (Need.SALT, 2),
        (Need.GOODS, 20),
        (Need.COFFEE, 2),
        (Need.CLOTHES, 3),
        (Need.FURNITURE, 2),
        (Need.BEER, 3),
        (Need.VODKA, 1),
        (Need.SHOES, 2),
        (Need.BRICKS, 5),
        (Need.REFINED_SUGAR, 2),
        (Need.NEWSPAPERS, 3),
        (Need.HEATING, 3),
    ]
    assert tenement.got == [
        Flow(Need.SALT, 0.0, 2.0),
        Flow(Need.COFFEE, 0.0, 2.0),
        Flow(Need.BRICKS, 0.0, 5.0),
        Flow(Need.NEWSPAPERS, 0.0, 3.0),
        Flow(Need.SHOES, 0.0, 2.0),
        Flow(Need.FURNITURE, 0.0, 2.0),
        Flow(Need.CLOTHES, 0.0, 3.0),
        Flow(Need.HEATING, 0.0, 3.0),
        Flow(Need.BEER, 0.0, 3.0),
        Flow(Need.VODKA, 0.0, 1.0),
        Flow(Need.REFINED_SUGAR, 0.0, 2.0),
        Flow(Need.FOOD, 0.0, 60.0),
        Flow(Need.GOODS, 0.0, 20.0),
    ]


def test_a_factory_without_lumber_makes_nothing_and_shows_what_it_could() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.FACTORY))
    growth = Growth(constructions)

    details = describe(growth, (5, 5))

    assert details.buildings[0].gave == [Flow(Good.GOODS, 0.0, 0.0, 30)]
