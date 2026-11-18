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


def test_a_lone_house_starts_a_town_and_one_two_cells_off_joins_it() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))

    towns.add((5, 2), 1.0)
    towns.add((5, 4), 1.0)
    towns.add((5, 7), 1.0)

    assert towns.town[5, 2] == towns.town[5, 4] != 0
    assert towns.town[5, 7] not in (0, towns.town[5, 2])


def test_a_cell_joins_the_town_sending_the_most_passengers_around_it() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 2), 1.0)
    towns.add((5, 6), 6.0)

    towns.add((5, 4), 1.0)

    assert towns.town[5, 4] == towns.town[5, 6]


def test_a_building_between_two_towns_paints_its_neighbourhood_the_busier_ones_colour() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 2), 1.0)
    towns.add((5, 6), 6.0)

    towns.add((5, 4), 1.0)

    assert towns.town[5, 2] == towns.town[5, 4] == towns.town[5, 6]


def test_a_building_going_up_leaves_cells_beyond_its_neighbourhood_alone() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 0), 1.0)
    towns.add((5, 2), 1.0)
    first = int(towns.town[5, 0])
    towns.add((5, 6), 6.0)

    towns.add((5, 4), 1.0)

    assert towns.town[5, 0] == first
    assert towns.town[5, 2] == towns.town[5, 6] != first


def test_a_cell_emptied_leaves_its_town() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 2), 1.0)

    towns.add((5, 2), -1.0)

    assert towns.town[5, 2] == 0


def test_a_cell_emptied_joins_the_town_around_it_when_built_on_again() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((5, 2), 1.0)
    towns.add((5, 2), -1.0)
    towns.add((5, 4), 18.0)

    towns.add((5, 2), 1.0)

    assert towns.town[5, 2] == towns.town[5, 4]


def test_the_buildings_standing_at_the_start_are_settled_into_towns_the_busiest_first() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    sending = np.zeros((12, 12))
    sending[5, 2], sending[5, 4], sending[5, 8] = 1.0, 6.0, 1.0

    towns.settle(sending)

    assert towns.town[5, 4] == 1
    assert towns.town[5, 2] == 1
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


ONE_TOWN = [(5, 3), (5, 5), (5, 7), (5, 9)]
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


def test_the_heart_of_the_biggest_town_is_its_busiest_cell_though_another_town_has_a_busier_one() -> (
    None
):
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))
    towns.add((2, 2), 5.0)
    for cell, sends in (((8, 2), 3.0), ((8, 4), 4.0), ((8, 6), 3.0)):
        towns.add(cell, sends)

    heart = towns.heart_of_biggest()

    assert heart == (8, 4)


def test_a_map_without_towns_has_no_biggest_one() -> None:
    towns = Towns(SquareGrid(rows=12, cols=12, km_per_cell=2.0))

    heart = towns.heart_of_biggest()

    assert heart is None
