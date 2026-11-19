import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.goods import Good
from railroads.sim.grids import SquareGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.towns import Towns
from railroads.sim.trains import Stop


def test_a_lone_house_starts_a_town_and_one_next_to_it_joins_it() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))

    towns.add((5, 2), 1.0)
    towns.add((5, 3), 1.0)
    towns.add((5, 5), 1.0)

    assert towns.town[5, 2] == towns.town[5, 3] != 0
    assert towns.town[5, 5] not in (0, towns.town[5, 2])


def test_a_cell_joins_the_town_sending_the_most_passengers_next_to_it() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 2), 1.0)
    towns.add((5, 4), 6.0)

    towns.add((5, 3), 1.0)

    assert towns.town[5, 3] == towns.town[5, 4]


def test_a_building_between_two_towns_paints_its_neighbours_the_busier_ones_colour() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 2), 1.0)
    towns.add((5, 4), 6.0)

    towns.add((5, 3), 1.0)

    assert towns.town[5, 2] == towns.town[5, 3] == towns.town[5, 4]


def test_a_town_outvoted_around_a_new_building_joins_the_winner_whole() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 0), 1.0)
    towns.add((5, 1), 1.0)
    towns.add((5, 3), 6.0)

    towns.add((5, 2), 1.0)

    assert towns.town[5, 0] == towns.town[5, 3]


def test_a_new_building_leaves_towns_without_a_cell_next_to_it_alone() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 0), 1.0)
    towns.add((5, 3), 6.0)
    first = int(towns.town[5, 0])

    towns.add((5, 4), 1.0)

    assert towns.town[5, 0] == first
    assert towns.town[5, 4] == towns.town[5, 3] != first


def test_a_cell_emptied_leaves_its_town() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 2), 1.0)

    towns.add((5, 2), -1.0)

    assert towns.town[5, 2] == 0


def test_a_cell_emptied_joins_the_town_next_to_it_when_built_on_again() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 2), 1.0)
    towns.add((5, 2), -1.0)
    towns.add((5, 3), 18.0)

    towns.add((5, 2), 1.0)

    assert towns.town[5, 2] == towns.town[5, 3]


def test_the_buildings_standing_at_the_start_are_settled_into_towns_the_busiest_first() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    sending = np.zeros((12, 12))
    sending[5, 3], sending[5, 4], sending[5, 8] = 1.0, 6.0, 1.0

    towns.settle(sending)

    assert towns.town[5, 4] == 1
    assert towns.town[5, 3] == 1
    assert towns.town[5, 8] == 2


def test_a_station_belongs_to_the_town_sending_the_most_from_the_cells_around_it() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 2), 1.0)
    towns.add((5, 8), 18.0)
    cells = np.array([5 * 12 + 2, 5 * 12 + 8])

    town = towns.most_in(cells)

    assert town == towns.town[5, 8]


def stations_with_houses(houses: list[tuple[int, int]]) -> tuple[Growth, Railway]:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 3))
    build_station(growth, (6, 9))
    route = plan_route(constructions, (6, 3), (6, 9))
    assert route is not None
    lay_route(growth, route)
    for cell in houses:
        growth.build(Building(cell, BuildingKind.HOUSE))
    railway = Railway(constructions)
    growth.railway = railway
    railway.station_trade = growth.station_trade
    return growth, railway


ONE_TOWN = [(5, column) for column in range(3, 10)]
TWO_TOWNS = [(5, 3), (5, 9)]


@pytest.mark.parametrize(("houses", "boarded"), [(ONE_TOWN, 0.0), (TWO_TOWNS, 10.0)])
def test_passengers_board_only_for_a_stop_in_another_town(
    houses: list[tuple[int, int]], boarded: float
) -> None:
    _, railway = stations_with_houses(houses)
    railway.station_goods((6, 3)).board(Good.PASSENGERS, 10.0)
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.PASSENGERS]), Stop((6, 9))]

    railway.run(0.001)

    assert train.cars[0].load == pytest.approx(boarded)


def test_a_train_waiting_for_a_full_load_does_not_wait_for_passengers_who_wont_board() -> None:
    _, railway = stations_with_houses(ONE_TOWN)
    railway.station_goods((6, 3)).board(Good.PASSENGERS, 10.0)
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.PASSENGERS], wait_for_load=True), Stop((6, 9))]

    railway.run(0.001)

    assert train.status == "Running"
    assert train.cars[0].load == 0.0


@pytest.mark.parametrize(
    ("width", "river", "joined"),
    [(1, False, False), (1, True, True), (2, True, True), (3, True, False)],
)
def test_a_town_spans_a_river_up_to_two_cells_wide(width: int, river: bool, joined: bool) -> None:
    rivers = np.zeros((12, 12), dtype=bool)
    rivers[:, 5 : 5 + width] = river
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0), rivers)

    towns.add((6, 4), 1.0)
    towns.add((6, 5 + width), 1.0)

    assert bool(towns.town[6, 4] == towns.town[6, 5 + width]) is joined
