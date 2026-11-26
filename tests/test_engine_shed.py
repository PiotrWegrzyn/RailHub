import math

import pytest

from builders import joined_stations
from railroads.sim.constructions import Facility
from railroads.sim.facilities import SHED_PRICES
from railroads.sim.goods import Good
from railroads.sim.growth import Growth
from railroads.sim.railway import Railway
from railroads.sim.routes import add_facility, build_station, lay_route, plan_route
from railroads.sim.sheds import NOSE, stall_way
from railroads.sim.trains import BULK_CAR, FROM_SHED, IN_SHED, Stop, Train
from railroads.ui.pixel.train_layer import _in_shed

STATION, SHED, FAR = (6, 3), (6, 2), (6, 20)


def with_shed(level: int = 3) -> tuple[Growth, Railway]:
    growth, railway = joined_stations()
    railway.levels[STATION] = level
    add_facility(growth, STATION, Facility.ENGINE_SHED, SHED, (0, 6))
    return growth, railway


def waiting_for_wheat(railway: Railway) -> Train:
    train = railway.buy()
    assert train is not None
    train.stops = [Stop(STATION, [Good.WHEAT], wait_for_load=True), Stop(FAR)]
    return train


def run(railway: Railway, times: int) -> None:
    for _ in range(times):
        railway.run(0.05)


def test_a_train_short_of_a_full_load_waits_in_its_stations_engine_shed() -> None:
    _, railway = with_shed()
    train = waiting_for_wheat(railway)

    run(railway, 20)

    assert train.shed == IN_SHED
    assert train.platform == SHED
    assert train.status == f"Waiting in the engine shed at {railway.name_of(STATION)}"


def test_a_train_in_the_shed_comes_back_to_load_once_the_station_holds_enough() -> None:
    _, railway = with_shed()
    train = waiting_for_wheat(railway)
    run(railway, 20)

    railway.station_goods(STATION).stockpile[Good.WHEAT] = BULK_CAR + 10.0
    for _ in range(400):
        railway.run(0.002)
        if train.path and railway.network.cell_at(train.path[-1]) == FAR:
            break

    assert train.shed == ""
    assert [car.load for car in train.cars] == [BULK_CAR]
    assert railway.network.cell_at(train.path[-1]) == FAR


def test_a_train_in_the_shed_stays_while_the_station_holds_too_little() -> None:
    _, railway = with_shed()
    train = waiting_for_wheat(railway)
    run(railway, 20)

    railway.station_goods(STATION).stockpile[Good.WHEAT] = BULK_CAR / 2
    run(railway, 20)

    assert train.shed == IN_SHED
    assert train.cars[0].load == 0.0


def test_without_an_engine_shed_a_train_short_of_a_load_waits_at_the_platform() -> None:
    _, railway = joined_stations()
    train = waiting_for_wheat(railway)

    run(railway, 20)

    assert train.shed == ""
    assert train.status == f"Loading at {railway.name_of(STATION)}"


def test_a_train_going_back_to_load_runs_from_the_shed_to_the_station() -> None:
    _, railway = with_shed()
    train = waiting_for_wheat(railway)
    run(railway, 20)
    railway.station_goods(STATION).stockpile[Good.WHEAT] = BULK_CAR + 10.0

    stall = stall_way(railway.constructions, SHED, train.stall)

    railway.run(0.001)

    assert train.shed == FROM_SHED
    assert train.path[:3] == list(stall)
    assert railway.network.cell_at(train.path[-1]) == STATION


def test_no_way_runs_through_an_engine_shed() -> None:
    growth, railway = with_shed()
    build_station(growth, (6, 0))
    beyond = plan_route(growth.constructions, (6, 0), SHED)
    assert beyond is not None
    lay_route(growth, beyond)

    through = railway.route((6, 0), FAR)
    into = railway.route((6, 0), SHED)

    assert through is None
    assert into is not None


def test_an_engine_shed_put_up_on_track_takes_its_place() -> None:
    growth, _ = joined_stations()
    on_track = (6, 4)

    add_facility(growth, STATION, Facility.ENGINE_SHED, on_track)

    constructions = growth.constructions
    assert constructions.track_at(on_track) == ()
    assert constructions.shed_at(on_track) is not None
    assert constructions.station_ends(on_track) == (0, 6)


def test_an_engine_shed_no_track_joins_turns_to_track_laid_into_its_side() -> None:
    growth, railway = joined_stations()
    shed, north = (5, 3), (1, 3)
    add_facility(growth, STATION, Facility.ENGINE_SHED, shed, (0, 6))
    build_station(growth, north)
    route = plan_route(growth.constructions, north, shed)
    assert route is not None

    lay_route(growth, route)

    assert growth.constructions.station_ends(shed) != (0, 6)
    assert railway.route(north, shed) is not None


def test_a_train_waiting_in_the_shed_stands_with_its_front_out_of_its_stalls_door() -> None:
    _, railway = with_shed()
    train = waiting_for_wheat(railway)

    run(railway, 20)

    back, door, _ = stall_way(railway.constructions, SHED, train.stall)
    assert train.path[:2] == [back, door]
    assert train.along == pytest.approx(math.dist(back, door) + NOSE)


def test_an_engine_shed_of_level_1_holds_three_trains() -> None:
    _, railway = with_shed()
    trains = [waiting_for_wheat(railway) for _ in range(4)]

    run(railway, 60)

    assert sum(train.shed == IN_SHED for train in trains) == 3
    assert len({train.stall for train in trains if train.shed}) == 3


def test_an_engine_shed_raised_to_level_3_holds_nine_trains() -> None:
    _, railway = with_shed()
    railway.finances.earn(10_000_000.0, "test")
    raised = [railway.upgrade_shed(SHED) for _ in range(3)]
    trains = [waiting_for_wheat(railway) for _ in range(10)]

    run(railway, 120)

    assert raised == [True, True, False]
    assert railway.shed_level(SHED) == 3
    assert sum(train.shed == IN_SHED for train in trains) == 9


def test_raising_an_engine_shed_costs_its_price() -> None:
    _, railway = with_shed()
    railway.finances.earn(1_000_000.0, "test")
    before = railway.finances.cash

    railway.upgrade_shed(SHED)

    assert before - railway.finances.cash == SHED_PRICES[0]


def test_an_engine_shed_is_not_raised_without_the_cash() -> None:
    _, railway = with_shed()
    railway.finances.cash = SHED_PRICES[0] - 1.0

    raised = railway.upgrade_shed(SHED)

    assert not raised
    assert railway.shed_level(SHED) == 1


def test_a_train_waiting_in_the_shed_is_inside_it_and_not_drawn() -> None:
    _, railway = with_shed()
    train = waiting_for_wheat(railway)
    run(railway, 20)

    hidden = _in_shed(railway, train, train.path[-1])

    assert hidden


def test_a_train_on_its_way_elsewhere_is_drawn_even_beside_a_shed() -> None:
    _, railway = with_shed()
    train = waiting_for_wheat(railway)

    hidden = _in_shed(railway, train, railway.constructions.tile_map.grid.center(*SHED))

    assert not hidden


def test_a_sand_tower_needs_an_engine_shed() -> None:
    _, railway = joined_stations()
    railway.levels[STATION] = 3

    put_up = railway.put_up(STATION, Facility.SAND_TOWER)

    assert not put_up
    assert Facility.SAND_TOWER not in railway.facilities_at(STATION)


def test_a_sand_tower_stands_by_the_engine_shed() -> None:
    _, railway = with_shed()

    put_up = railway.put_up(STATION, Facility.SAND_TOWER)

    shed = railway.constructions.shed_at(SHED)
    assert put_up
    assert shed is not None and shed.sand_tower
    assert Facility.SAND_TOWER in railway.facilities_at(STATION)


def test_closing_the_sand_tower_takes_it_from_beside_the_shed() -> None:
    _, railway = with_shed()
    railway.put_up(STATION, Facility.SAND_TOWER)

    railway.close_facility(STATION, Facility.SAND_TOWER)

    shed = railway.constructions.shed_at(SHED)
    assert shed is not None and not shed.sand_tower
    assert Facility.SAND_TOWER not in railway.facilities_at(STATION)


def test_a_sand_tower_put_up_in_the_station_before_moves_beside_its_engine_shed() -> None:
    _, railway = with_shed()
    railway.tucked = {STATION: {Facility.SAND_TOWER}}

    railway._supply_facilities()

    shed = railway.constructions.shed_at(SHED)
    assert shed is not None and shed.sand_tower
    assert railway.tucked == {STATION: set()}
    assert Facility.SAND_TOWER in railway.facilities_at(STATION)
