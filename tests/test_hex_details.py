import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions, Station
from railroads.sim.goods import Good, Need, bonus_of, needs_of
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.tilemap import Terrain, TileMap
from railroads.ui.hex_details import BuildingRow, Flow, describe, mood_text


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
    assert house.needs == [(Need.FOOD, 30), (Need.HEATING, 3)]
    assert dict(house.bonus) == {
        need: pytest.approx(3 * amount)
        for need, amount in bonus_of(BuildingKind.HOUSE, 1850).items()
    }
    got = {flow.good: (flow.done, flow.wanted) for flow in house.got}
    assert got[Need.FOOD] == (pytest.approx(9.0), 30.0)
    assert got[Need.HEATING] == (0.0, 3.0)
    assert got[Need.BEER] == (0.0, 3.0)
    assert house.satisfaction == "content"
    assert farm.makes == [(Good.WHEAT, 10)]
    assert [(flow.good, flow.done, flow.wanted) for flow in farm.gave] == [
        (Good.WHEAT, pytest.approx(9.0), 10.0)
    ]
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


def test_a_tenement_needs_only_food_and_heating_and_takes_the_rest_as_bonus_goods() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.TENEMENT))
    growth = Growth(constructions)

    details = describe(growth, (5, 5))

    tenement = details.buildings[0]
    assert tenement.needs == [(Need.FOOD, 60), (Need.HEATING, 3)]
    assert dict(tenement.bonus) == bonus_of(BuildingKind.TENEMENT, 1850)
    assert {flow.good for flow in tenement.got} == set(needs_of(BuildingKind.TENEMENT, 1850))
    assert all(flow.done == 0.0 for flow in tenement.got)


def test_a_factory_without_lumber_makes_none_of_the_most_it_could() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.FACTORY))
    growth = Growth(constructions)

    details = describe(growth, (5, 5))

    assert details.buildings[0].gave == [Flow(Good.GOODS, 0.0, 30.0)]


def test_a_building_shows_its_mood_with_its_satisfaction_as_a_percentage() -> None:
    row = BuildingRow("house", 1, [], [], [], [], [], [], "thriving", percent=1.22)
    quiet = BuildingRow("house", 1, [], [], [], [], [], [], "-")

    text, none = mood_text(row), mood_text(quiet)

    assert text == "thriving 122%"
    assert none == "-"
