from dataclasses import replace

import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions, Station
from railroads.sim.game import Game
from railroads.sim.goods import Good, Need
from railroads.sim.grids import Cell, HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.ledger import Takings
from railroads.sim.ownership import HOLD_MONTHS
from railroads.sim.points import Points
from railroads.sim.ports import Port, Slot, Tier
from railroads.sim.tilemap import Terrain, TileMap
from railroads.sim.trade import Satisfaction
from railroads.sim.yearly import bonus_of
from railroads.ui import hex_rows
from railroads.ui.hex_rows import (
    BuildingRow,
    Flow,
    HexDetails,
    describe,
    mood_text,
    recently_described,
    result_lines,
    sell_lots,
    share_lines,
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
    got = {flow.good: (flow.done, flow.wanted) for flow in house.got}
    bonus_got = {flow.good: (flow.done, flow.wanted) for flow in house.bonus_got}

    assert house.name == "house"
    assert house.count == 3
    assert house.needs == [(Need.FOOD, 18), (Need.HEATING, 6)]
    assert dict(house.bonus) == {
        Need.FOOD: 18,
        **{
            need: pytest.approx(3 * amount)
            for need, amount in bonus_of(BuildingKind.HOUSE, 1850).items()
        },
    }
    assert got[Need.FOOD] == (pytest.approx(6 * 13 / 21), 18.0)
    assert bonus_got[Need.FOOD] == (0.0, 18.0)
    assert got[Need.HEATING] == (0.0, 6.0)
    assert bonus_got[Need.BEER] == (0.0, pytest.approx(1.8))
    assert house.satisfaction == "short"
    assert farm.makes == [(Good.WHEAT, 6)]
    assert [(flow.good, flow.done, flow.wanted) for flow in farm.gave] == [
        (Good.WHEAT, pytest.approx(6 * 13 / 21), 6.0)
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


def test_a_tenement_must_have_half_its_food_common_goods_and_heating_and_takes_the_rest_as_bonus() -> (
    None
):
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.TENEMENT))
    growth = Growth(constructions)

    details = describe(growth, (5, 5))

    tenement = details.buildings[0]
    assert tenement.needs == [(Need.FOOD, 35), (Need.COMMON, 40), (Need.HEATING, 12)]
    assert dict(tenement.bonus) == {Need.FOOD: 35, **bonus_of(BuildingKind.TENEMENT, 1850)}
    assert {flow.good for flow in tenement.got} == {Need.FOOD, Need.COMMON, Need.HEATING}
    assert {flow.good for flow in tenement.bonus_got} == {
        Need.FOOD,
        *bonus_of(BuildingKind.TENEMENT, 1850),
    }
    assert all(flow.done == 0.0 for flow in [*tenement.got, *tenement.bonus_got])


def test_a_factory_without_planks_makes_none_of_the_most_it_could() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.FACTORY))
    growth = Growth(constructions)

    details = describe(growth, (5, 5))

    assert details.buildings[0].gave == [Flow(Good.GOODS, 0.0, 20.0)]


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

    trade = growth.last_trade
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
    plain = hex_rows.describe

    def counted(growth: Growth, cell: Cell, satisfaction: object = None) -> HexDetails:
        worked.append(cell)
        return plain(growth, cell)

    monkeypatch.setattr(hex_rows, "describe", counted)

    for _ in range(3):
        recently_described(growth, (5, 5))
        recently_described(growth, (5, 4))

    assert worked == [(5, 5), (5, 4)]


def test_a_bakery_must_get_wheat_and_shows_refined_sugar_as_a_bonus() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.BAKERY))
    growth = Growth(constructions, GrowthRule(trades=True, ladders=()))

    (bakery,) = describe(growth, (5, 5)).buildings

    assert [flow.good for flow in bakery.got] == [Need.WHEAT]
    assert [flow.good for flow in bakery.bonus_got] == [Need.REFINED_SUGAR]


def test_details_say_the_company_flies_its_flag_over_a_building_the_player_owns() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 4), BuildingKind.FARM))
    growth = Growth(constructions, GrowthRule(trades=True, ladders=()))
    growth.owners.own((5, 4), BuildingKind.FARM, 300)

    others = describe(growth, (5, 4)).others

    assert others == ["The Trains Corp flies its flag here: you own 300 shares of the farms."]


def farms_row(**fields: object) -> BuildingRow:
    row = BuildingRow("farm", 3, [], [], [], [], [], [], "content", BuildingKind.FARM)
    return replace(row, **fields)


def test_the_shares_window_shows_every_share_the_players_and_those_they_may_sell_now() -> None:
    row = farms_row(owned=300, price=4320.0, sellable=100, sell_from=1862, total=3000)

    lines = {label: value for label, value, _ in share_lines(row)}

    assert lines["In all (3 x 1,000)"] == "3,000"
    assert lines["Yours"] == "300 (10% of what they make)"
    assert lines["Can sell now"] == "100"
    assert lines["Sellable from 1862"] == "200"
    assert lines["Price of 100"] == "$4,320"


def test_results_show_last_year_and_the_forecast_and_a_dash_before_a_whole_year() -> None:
    row = farms_row(total=3000, forecast=Takings(4300.0, 430.0, 80.0))

    lines = {label: (last, forecast) for label, last, forecast, _ in result_lines(row)}

    assert lines["All 3 farms: sold"] == ("-", "$4,300")
    assert lines["Yours: sold to stations"] == ("-", "$430")
    assert lines["Yours in all"] == ("-", "$510")


def test_the_shares_window_says_how_many_years_shares_are_held_before_they_can_be_sold() -> None:
    row = farms_row(owned=100, total=3000)

    tips = {label: tip for label, _, tip in share_lines(row)}

    assert tips["Can sell now"] == f"Shares bought can be sold {HOLD_MONTHS // 12} years later"


@pytest.mark.parametrize(
    ("sellable", "lots"), [(0, [0]), (50, [50]), (300, [100, 200]), (700, [100, 200, 600])]
)
def test_the_sell_buttons_offer_each_lot_the_player_may_sell_or_just_what_they_may(
    sellable: int, lots: list[int]
) -> None:
    offered = sell_lots(sellable)

    assert offered == lots


@pytest.mark.parametrize(("owned", "control"), [(500, "from 600 shares"), (600, "yours")])
def test_the_shares_window_says_from_how_many_shares_the_player_controls_the_buildings(
    owned: int, control: str
) -> None:
    row = farms_row(owned=owned, price=4320.0, total=1000)

    lines = {label: value for label, value, _ in share_lines(row)}

    assert lines["Control"] == control


def test_a_port_shows_only_the_goods_it_gives_itself() -> None:
    constructions = Constructions(make_map())
    for cell in ((2, 1), (7, 1)):
        constructions.build(Building(cell, BuildingKind.PORT))
    growth = Growth(constructions, GrowthRule(trades=True, ladders=()))

    (port,) = describe(growth, (2, 1)).buildings

    assert {flow.good for flow in port.gave} == set(growth.ports.gives_at((2, 1)))


def test_a_farm_says_whom_it_handed_its_wheat_to() -> None:
    constructions = Constructions(make_map())
    for _ in range(3):
        constructions.build(Building((5, 5), BuildingKind.HOUSE))
    constructions.build(Building((5, 4), BuildingKind.FARM))
    growth = Growth(constructions, GrowthRule(trades=True, ladders=()))

    handed = describe(growth, (5, 4)).handed

    assert handed == [(Good.WHEAT, "house x3 at 5, 5", pytest.approx(6 * 13 / 21))]


def test_a_farm_says_it_handed_wheat_to_a_port_taking_it() -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 1), BuildingKind.PORT))
    constructions.build(Building((5, 2), BuildingKind.FARM))
    growth = Growth(constructions, GrowthRule(trades=False, ladders=()))
    growth.ports.at[(5, 1)] = Port([Slot(Good.WHEAT, Tier.COMMON, False)])
    growth.trade_goods()

    handed = describe(growth, (5, 2)).handed

    assert [(good, to) for good, to, _ in handed] == [(Good.WHEAT, "port at 5, 1")]
