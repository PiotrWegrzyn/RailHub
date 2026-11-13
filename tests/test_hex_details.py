import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions, Station
from railroads.sim.economy import Satisfaction
from railroads.sim.game import Game
from railroads.sim.goods import Good, Need, bonus_of, needs_of
from railroads.sim.grids import Cell, HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule, Points
from railroads.sim.tilemap import Terrain, TileMap
from railroads.ui import hex_details
from railroads.ui.hex_details import (
    BuildingRow,
    Flow,
    HexDetails,
    describe,
    mood_text,
    recently_described,
)


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


def traded_growth() -> Growth:
    game = Game(Constructions(make_map()))
    game.finances.cash = 1_000_000.0
    for _ in range(3):
        game.growth.build(Building((5, 5), BuildingKind.HOUSE))
    game.growth.build(Building((5, 5), BuildingKind.BAKERY))
    game.growth.build(Building((5, 4), BuildingKind.FARM))
    game.growth.build(Building((4, 6), BuildingKind.FARM))
    game.build_station((5, 6))
    game.build_station((5, 2))
    game.lay_track((5, 6), (5, 2))
    rng = np.random.default_rng(3)
    for _ in range(3):
        game.month(rng)
    return game.growth


def test_a_cell_alone_is_described_as_the_whole_map_has_it() -> None:
    growth = traded_growth()
    cells = [(row, col) for row in range(10) for col in range(10)]

    described = {cell: describe(growth, cell, growth.satisfaction()) for cell in cells}

    trade = growth.trade
    assert trade is not None
    rated = 0
    for cell, details in described.items():
        assert details == describe(growth, cell)
        whole_map = [
            (kind.value, int(growth.points(kind)[cell]), int(growth.nearby(kind)[cell]))
            for kind in Points
            if growth.nearby(kind)[cell]
        ]
        assert details.points == whole_map
        position = int(np.searchsorted(trade.cells, cell[0] * 10 + cell[1]))
        for row in details.buildings:
            if row.percent is None:
                continue
            rated += 1
            assert row.kind is not None
            assert row.percent == float(trade.share((row.kind,))[position])
            level = Satisfaction(int(trade.levels((row.kind,))[position]))
            assert row.satisfaction == level.name.lower()
    assert rated >= 3


def test_cells_described_by_turns_are_each_worked_out_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    growth = traded_growth()
    worked: list[Cell] = []
    plain = hex_details.describe

    def counted(growth: Growth, cell: Cell, satisfaction: object = None) -> HexDetails:
        worked.append(cell)
        return plain(growth, cell)

    monkeypatch.setattr(hex_details, "describe", counted)

    for _ in range(3):
        recently_described(growth, (5, 5))
        recently_described(growth, (5, 4))

    assert worked == [(5, 5), (5, 4)]
