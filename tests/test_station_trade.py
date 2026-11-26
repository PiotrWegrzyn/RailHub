import pytest

from builders import boards_with, flat_map, joined_stations, open_boards
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.demand import DEMAND_PER_NEED
from railroads.sim.goods import MAKES, Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import PASSENGERS_PER_POINT, GrowthRule
from railroads.sim.points import Points
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station
from railroads.sim.tiles import Tile, activate
from railroads.sim.trade import input_stock

A_HOUSE = DEMAND_PER_NEED * PASSENGERS_PER_POINT[Points.POPULATION]


def close_stations() -> tuple[Growth, Railway]:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 10))
    build_station(growth, (6, 13))
    railway = Railway(constructions)
    railway.levels = {(6, 10): 2, (6, 13): 2}
    growth.railway = railway
    return growth, railway


def put_up(growth: Growth, cell: tuple[int, int], houses: int) -> None:
    for _ in range(houses):
        growth.build(Building(cell, BuildingKind.HOUSE))


def test_stations_whose_rings_overlap_each_count_every_building_in_their_own_ring() -> None:
    growth, railway = close_stations()
    put_up(growth, (6, 11), 2)
    put_up(growth, (6, 8), 1)
    put_up(growth, (6, 15), 3)

    growth.trade_goods()

    passengers = {
        station: railway.station_demand(station).capacity[Need.PASSENGERS]
        for station in railway.stations()
    }
    assert passengers == {
        (6, 10): pytest.approx(A_HOUSE * 3),
        (6, 13): pytest.approx(A_HOUSE * 5),
    }


def test_a_raised_station_counts_the_buildings_in_its_wider_ring() -> None:
    growth, railway = close_stations()
    put_up(growth, (6, 7), 4)
    growth.trade_goods()
    before = railway.station_demand((6, 10)).capacity.get(Need.PASSENGERS)
    railway.levels[(6, 10)] = 3

    growth.trade_goods()

    assert before is None
    assert railway.station_demand((6, 10)).capacity[Need.PASSENGERS] == pytest.approx(A_HOUSE * 4)


def test_the_cells_in_reach_are_every_cell_any_station_hands_goods_out_to() -> None:
    growth, _ = close_stations()
    grid = growth.grid

    growth.trade_goods()

    reached = set(grid.within((6, 10), 2)) | set(grid.within((6, 13), 2))
    assert growth.station_trade.in_reach == reached


def test_goods_buildings_hand_out_on_their_own_by_a_station_earn_the_player_nothing() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    growth.build(Building((6, 22), BuildingKind.FARM))

    growth.trade_goods()

    assert railway.finances.month["cargo"] == 0.0
    assert railway.station_goods((6, 20)).stockpile.get(Good.WHEAT, 0.0) == pytest.approx(
        MAKES[BuildingKind.FARM][Good.WHEAT] * 8 / 21
    )


def food_wanted_with_a_farm_at(farm: tuple[int, int] | None) -> float:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    if farm is not None:
        growth.build(Building(farm, BuildingKind.FARM))
    growth.trade_goods()
    return railway.station_demand((6, 20)).capacity.get(Need.FOOD, 0.0)


def test_a_station_wants_more_food_the_less_the_farms_around_could_hand_its_houses() -> None:
    farms = [(6, 22), (6, 25), None]

    wanted = [food_wanted_with_a_farm_at(farm) for farm in farms]

    assert wanted[0] < wanted[1] < wanted[2]


@pytest.mark.parametrize(("stocked", "wanted"), [(0.0, True), (None, False)])
def test_a_processor_stocked_up_on_an_input_adds_no_demand_for_it_at_the_station(
    stocked: float | None, wanted: bool
) -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.STEEL_MILL))
    cell = growth.constructions.tile_map.grid.shape[1] * 6 + 21
    full = input_stock(BuildingKind.STEEL_MILL, Need.COAL, growth.year)
    growth.economy.stock[(BuildingKind.STEEL_MILL, Need.COAL)][cell] = (
        full if stocked is None else stocked
    )

    growth.station_trade.measure_demand((6, 20))

    capacity = railway.station_demand((6, 20)).capacity
    assert (capacity.get(Need.COAL, 0.0) > 0) == wanted


def test_a_station_about_to_be_built_counts_the_points_of_the_buildings_it_would_hand_out_to() -> (
    None
):
    growth = Growth(Constructions(flat_map()), GrowthRule(trades=False))
    put_up(growth, (6, 11), 2)
    put_up(growth, (6, 14), 4)

    preview = growth.station_trade.preview((6, 10))

    kind, many, points, value = preview.reached[0]
    assert (kind, many) == (BuildingKind.HOUSE, 2)
    assert (points, value) == growth.rule.points[BuildingKind.HOUSE]
    assert len(preview.reached) == 1


def test_a_station_about_to_be_built_shows_what_its_buildings_want_and_what_is_made_around() -> (
    None
):
    growth = Growth(Constructions(flat_map()), GrowthRule(trades=False))
    put_up(growth, (6, 11), 3)
    growth.build(Building((6, 12), BuildingKind.FARM))

    preview = growth.station_trade.preview((6, 10))

    assert preview.demand.capacity[Need.PASSENGERS] == pytest.approx(A_HOUSE * 3)
    assert preview.demand.capacity[Need.FOOD] > 0
    assert preview.made[Good.WHEAT] > 0
    assert preview.made[Good.PASSENGERS] > 0


def test_a_station_standing_reaches_as_far_as_its_own_reach_goes() -> None:
    growth = Growth(Constructions(flat_map()), GrowthRule(trades=False))
    put_up(growth, (6, 11), 2)
    put_up(growth, (6, 14), 4)

    preview = growth.station_trade.preview((6, 10), hand_out=4)

    assert {kind: many for kind, many, _, _ in preview.reached} == {BuildingKind.HOUSE: 6}


def test_a_station_reaches_out_from_its_platforms_as_well() -> None:
    growth = Growth(Constructions(flat_map()), GrowthRule(trades=False))
    put_up(growth, (6, 11), 2)
    put_up(growth, (6, 14), 4)

    preview = growth.station_trade.preview((6, 10), platforms=((6, 13),))

    assert {kind: many for kind, many, _, _ in preview.reached} == {BuildingKind.HOUSE: 6}


def test_a_processor_keeps_as_much_more_input_in_stock_once_its_demand_tile_goes_up() -> None:
    activate(open_boards())
    before = input_stock(BuildingKind.STEEL_MILL, Need.COAL, 1900)

    activate(boards_with(wanted={Tile.COAL: 2.0}))
    after = input_stock(BuildingKind.STEEL_MILL, Need.COAL, 1900)

    assert after > before
