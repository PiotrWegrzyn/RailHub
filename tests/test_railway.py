import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions, Track
from railroads.sim.demand import PASSED_ALONG_PAY, price_of
from railroads.sim.economy import Economy, StationOrder
from railroads.sim.finance import (
    CAR_TRIP_PRICE,
    STATION_UPKEEP,
    STORAGE_PRICE,
    TRACK_UPKEEP,
    TRAIN_PRICE,
)
from railroads.sim.goods import Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import CELLS_PER_MONTH, Railway
from railroads.sim.routes import build_station, lay_route, plan_route, route_price
from railroads.sim.station_goods import STOCKPILE, WAITING, StationGoods
from railroads.sim.trains import CAR_CAPACITY, MAX_CARS, Car, Stop


def joined_stations() -> tuple[Growth, Railway]:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 3))
    build_station(growth, (6, 20))
    route = plan_route(constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    return growth, railway


def test_trains_find_their_way_along_track_between_stations() -> None:
    _, railway = joined_stations()

    way = railway.route((6, 3), (6, 20))

    assert way is not None
    points, speeds = way
    assert len(speeds) == len(points) - 1
    assert railway.route((6, 3), (2, 3)) is None


def test_a_new_train_leaves_its_first_stop_with_that_stops_cars_full() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 75.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT, Good.WHEAT], wait_for_load=False), Stop((6, 20))]

    railway.run(0.1)

    assert train.heading == 1
    assert [car.load for car in train.cars] == [CAR_CAPACITY, 25.0]
    assert railway.position(train) is not None


def test_a_train_waits_at_its_stop_until_new_goods_fill_its_cars() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT]), Stop((6, 20))]
    railway.run(0.1)
    waiting = train.status

    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 60.0
    railway.run(0.1)

    assert waiting == "Loading at 6, 3"
    assert [car.load for car in train.cars] == [CAR_CAPACITY]
    assert train.status == "Running"


def test_a_train_gives_up_waiting_after_a_month() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT]), Stop((6, 20))]

    for _ in range(12):
        railway.run(0.1)

    assert train.status == "Running"
    assert train.cars[0].load == 0.0


def test_at_the_next_stop_it_unloads_and_takes_that_stops_cars() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 50.0
    railway.station_goods((6, 20)).stockpile[Good.COAL] = 150.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT]), Stop((6, 20), [Good.COAL, Good.COAL])]

    railway.run(0.1)
    railway.run(25 / CELLS_PER_MONTH)

    assert train.heading == 0
    assert railway.station_goods((6, 20)).brought[Good.WHEAT] == 50.0
    assert [(car.good, car.load) for car in train.cars] == [(Good.COAL, 50.0), (Good.COAL, 50.0)]


def test_each_car_fills_with_its_own_good_from_the_stockpile() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.LUMBER, Good.LUMBER, Good.WHEAT]), Stop((6, 20))]
    railway.station_goods((6, 3)).stockpile.update({Good.LUMBER: 70.0, Good.COAL: 150.0})

    railway.run(0.01)

    assert [car.load for car in train.cars] == [CAR_CAPACITY, 20.0, 0.0]
    assert railway.station_goods((6, 3)).stockpile == {Good.LUMBER: 0.0, Good.COAL: 150.0}


def test_a_stop_sends_at_most_six_cars() -> None:
    stop = Stop((6, 3))

    added = [stop.add_car(Good.COAL) for _ in range(MAX_CARS + 1)]

    assert added == [True] * MAX_CARS + [False]


def test_without_track_a_train_waits() -> None:
    constructions = Constructions(flat_map())
    railway = Railway(constructions)
    train = railway.buy()
    train.stops = [Stop((6, 3)), Stop((6, 20))]

    railway.run(1.0)

    assert train.status == "No track to its next stop"


def test_stations_hand_goods_to_the_hungriest_buildings_and_gather_spare_ones() -> None:
    tile_map = flat_map()
    economy = Economy(tile_map)
    counts = {kind: np.zeros(tile_map.grid.shape, dtype=np.int64) for kind in BuildingKind}
    counts[BuildingKind.FLATS][6, 10] = 6
    counts[BuildingKind.LUMBER_MILL][6, 25] = 1
    orders = {
        (6, 11): StationOrder({Good.WHEAT: 20.0}, 0.0),
        (6, 24): StationOrder({}, 50.0),
    }

    trade = economy.trade(counts, orders)

    index = int(np.searchsorted(trade.cells, np.ravel_multi_index((6, 25), tile_map.grid.shape)))
    assert trade.handed[(6, 11)][Good.WHEAT] == pytest.approx(20.0)
    assert trade.gathered[(6, 24)][Good.LUMBER] == pytest.approx(20.0)
    assert trade.gave[BuildingKind.LUMBER_MILL][index] == pytest.approx(20.0)
    assert sum(trade.gathered[(6, 24)].values()) <= 50.0


def test_growth_hands_station_goods_out_and_stores_what_was_gathered() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.LUMBER_MILL))

    growth.trade_goods()

    goods = railway.station_goods((6, 20))
    assert goods.stockpile[Good.LUMBER] == pytest.approx(20.0)
    assert goods.stocked <= STOCKPILE


def test_a_station_gathers_what_a_farm_makes() -> None:
    tile_map = flat_map()
    economy = Economy(tile_map)
    counts = {kind: np.zeros(tile_map.grid.shape, dtype=np.int64) for kind in BuildingKind}
    counts[BuildingKind.HOUSE][6, 10] = 1
    counts[BuildingKind.FARM][6, 14] = 1

    trade = economy.trade(counts, {(6, 12): StationOrder({}, 100.0)})

    assert trade.gathered[(6, 12)][Good.WHEAT] == pytest.approx(5.0)


def test_cars_slow_a_train_and_full_cars_slow_it_more() -> None:
    _, railway = joined_stations()
    light, empty, full = railway.buy(), railway.buy(), railway.buy()
    empty.cars = [Car(Good.COAL) for _ in range(MAX_CARS)]
    full.cars = [Car(Good.COAL, CAR_CAPACITY) for _ in range(MAX_CARS)]

    pulls = [train.pull for train in (light, empty, full)]

    assert pulls[0] == 1.0
    assert pulls[0] > pulls[1] > pulls[2] >= 0.5


def test_trains_run_through_a_station_on_the_way() -> None:
    growth, railway = joined_stations()
    build_station(growth, (6, 27))
    route = plan_route(growth.constructions, (6, 20), (6, 27))
    assert route is not None
    lay_route(growth, route)

    way = railway.route((6, 3), (6, 27))

    assert way is not None
    points, _ = way
    assert any(
        np.allclose(point, growth.constructions.tile_map.grid.center(6, 20)) for point in points
    )


def test_cars_set_after_the_stops_still_load_on_the_first_trip() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 10.0
    train = railway.buy()
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    railway.run(0.05)

    train.stops[0].add_car(Good.WHEAT)
    train.cars_changed()
    railway.run(0.05)

    assert [(car.good, car.load) for car in train.cars] == [(Good.WHEAT, 10.0)]


def test_later_car_changes_wait_for_the_next_call() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    railway.run(1.0)
    heading = train.heading

    train.stops[0].add_car(Good.COAL)
    train.cars_changed()

    assert train.arrivals >= 1
    assert train.heading == heading


def test_removing_one_loaded_car_keeps_the_others_full_and_returns_its_goods() -> None:
    _, railway = joined_stations()
    stockpile = railway.station_goods((6, 3)).stockpile
    stockpile[Good.WHEAT] = 400.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT] * MAX_CARS, wait_for_load=False), Stop((6, 20))]
    railway.run(0.001)

    train.stops[0].cars.pop()
    train.cars_changed()
    railway.run(0.001)

    assert [car.load for car in train.cars] == [CAR_CAPACITY] * (MAX_CARS - 1)
    assert stockpile[Good.WHEAT] == pytest.approx(400.0 - (MAX_CARS - 1) * CAR_CAPACITY)
    assert railway.station_goods((6, 3)).brought == {}


def test_a_train_called_back_on_its_first_trip_keeps_what_it_loaded() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 300.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT] * MAX_CARS), Stop((6, 20))]
    railway.run(0.2)
    running = bool(train.path)

    train.stops[0].cars.pop()
    train.cars_changed()
    railway.run(0.001)

    assert running
    assert [car.load for car in train.cars] == [CAR_CAPACITY] * (MAX_CARS - 1)
    assert railway.station_goods((6, 3)).stockpile[Good.WHEAT] == pytest.approx(
        300.0 - (MAX_CARS - 1) * CAR_CAPACITY
    )


def test_a_train_costs_five_thousand_and_sells_for_half() -> None:
    _, railway = joined_stations()
    cash = railway.finances.cash

    train = railway.buy()
    assert train is not None
    after_buying = railway.finances.cash
    railway.sell(train)

    assert after_buying == cash - TRAIN_PRICE
    assert railway.finances.cash == cash - TRAIN_PRICE / 2


def test_without_the_cash_no_train_is_bought() -> None:
    _, railway = joined_stations()
    railway.finances.cash = TRAIN_PRICE - 1

    train = railway.buy()

    assert train is None
    assert railway.trains == []


def test_cargo_earns_money_only_when_the_buildings_around_take_it() -> None:
    growth, railway = joined_stations()
    for cell in [(6, 21)] * 3 + [(5, 20)]:
        growth.build(Building(cell, BuildingKind.HOUSE))
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 50.0
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.WHEAT]), Stop((6, 20))]
    railway.run(0.01)
    railway.run(25 / CELLS_PER_MONTH)
    on_arrival = railway.finances.month["cargo"]

    growth.trade_goods()

    goods = railway.station_goods((6, 20))
    assert on_arrival == 0.0
    assert goods.brought[Good.WHEAT] == pytest.approx(10.0)
    assert 0.0 < railway.finances.month["cargo"] <= 40 * price_of(Good.WHEAT)


def test_each_month_stations_and_track_cost_their_upkeep() -> None:
    growth, railway = joined_stations()
    track = {built.cell for built in growth.constructions if isinstance(built, Track)}
    cash = railway.finances.cash

    railway.turn_month({})

    assert railway.finances.cash == cash - 2 * STATION_UPKEEP - len(track) * TRACK_UPKEEP


def test_station_demand_is_twice_the_needs_of_buildings_in_its_inner_ring() -> None:
    growth, railway = joined_stations()
    for cell in [(6, 21)] * 6 + [(5, 20)] * 6 + [(6, 25)] * 2:
        growth.build(Building(cell, BuildingKind.HOUSE))

    growth.trade_goods()

    demand = railway.station_demand((6, 20))
    assert demand.capacity == {
        Need.FOOD: 240.0,
        Need.BEER: 24.0,
        Need.NEWSPAPERS: 24.0,
        Need.HEATING: 24.0,
        Need.PASSENGERS: 24.0,
        Need.MAIL: 4.0,
    }
    assert demand.record(Need.FOOD).left == pytest.approx(240.0)


def test_a_route_costs_three_hundred_a_cell_on_the_flat() -> None:
    constructions = Constructions(flat_map())
    route = plan_route(constructions, (6, 3), (6, 8))
    assert route is not None

    price = route_price(constructions, route)

    assert price == pytest.approx(300 * len(route.pieces))


def test_stations_send_passengers_by_the_points_around_them_and_a_fifth_as_much_mail() -> None:
    growth, railway = joined_stations()
    for cell, kind in [
        ((6, 21), BuildingKind.HOUSE),
        ((6, 21), BuildingKind.HOUSE),
        ((5, 20), BuildingKind.FACTORY),
        ((7, 19), BuildingKind.FARM),
    ]:
        growth.build(Building(cell, kind))

    growth.trade_goods()

    goods = railway.station_goods((6, 20))
    demand = railway.station_demand((6, 20))
    assert goods.stockpile[Good.PASSENGERS] == 2 * 1 + 10 + 5
    assert goods.stockpile[Good.MAIL] == 3
    assert demand.capacity[Need.PASSENGERS] == 2 * 17


def test_waiting_passengers_have_their_own_room_and_leave_freight_its_stockpile() -> None:
    goods = StationGoods()
    goods.stockpile[Good.COAL] = 40.0

    goods.board(Good.PASSENGERS, 70.0)
    goods.board(Good.PASSENGERS, 70.0)

    assert goods.stockpile[Good.PASSENGERS] == WAITING
    assert goods.stocked == 40.0


def test_passengers_pay_on_arrival_and_go_on_their_way() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.PASSENGERS] = 50.0
    railway.station_demand((6, 20)).capacity = {Need.PASSENGERS: 200.0}
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.PASSENGERS]), Stop((6, 20))]

    railway.run(0.01)
    railway.run(25 / CELLS_PER_MONTH)

    assert railway.finances.month["passengers"] > 0.95 * 50 * price_of(Good.PASSENGERS)
    assert Good.PASSENGERS not in railway.station_goods((6, 20)).brought


def test_raising_a_station_costs_100k_then_500k_and_widens_its_reach() -> None:
    _, railway = joined_stations()
    railway.finances.cash = 700_000.0

    reaches = [railway.reach((6, 20))]
    for _ in range(3):
        railway.upgrade((6, 20))
        reaches.append(railway.reach((6, 20)))

    assert reaches == [(2, 4), (3, 5), (4, 6), (4, 6)]
    assert railway.finances.cash == 100_000.0
    assert railway.upgrade_price((6, 20)) is None


def test_without_the_cash_a_station_stays_as_it_is() -> None:
    _, railway = joined_stations()
    railway.finances.cash = 99_999.0

    raised = railway.upgrade((6, 20))

    assert not raised
    assert railway.level((6, 20)) == 1


def test_each_storage_upgrade_adds_five_hundred_to_freight_and_to_waiting_room() -> None:
    _, railway = joined_stations()
    cash = railway.finances.cash

    railway.add_storage((6, 20))
    railway.add_storage((6, 20))

    goods = railway.station_goods((6, 20))
    assert (goods.limit, goods.waiting_limit) == (STOCKPILE + 1000, WAITING + 1000)
    assert railway.finances.cash == cash - 2 * STORAGE_PRICE


def test_a_raised_station_gathers_from_further_away() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 25), BuildingKind.LUMBER_MILL))
    growth.trade_goods()
    before = railway.station_goods((6, 20)).stockpile.get(Good.LUMBER, 0.0)
    railway.finances.cash = 100_000.0
    railway.upgrade((6, 20))

    growth.trade_goods()

    assert before == 0.0
    assert railway.station_goods((6, 20)).stockpile[Good.LUMBER] == pytest.approx(20.0)


def test_every_car_costs_twenty_each_time_the_train_sets_off() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.COAL, Good.WHEAT], wait_for_load=False), Stop((6, 20))]

    railway.run(0.01)

    assert railway.finances.month["car trips"] == -2 * CAR_TRIP_PRICE


def test_a_mine_in_reach_feeds_a_steel_mill_in_reach_through_the_station() -> None:
    growth, railway = joined_stations()
    for cell, kind in [((6, 24), BuildingKind.MINE), ((6, 21), BuildingKind.STEEL_MILL)]:
        growth.build(Building(cell, kind))

    growth.trade_goods()
    growth.trade_goods()

    mill = np.ravel_multi_index((6, 21), growth.constructions.tile_map.grid.shape)
    assert growth.economy.stock[(BuildingKind.STEEL_MILL, Need.COAL)][mill] == pytest.approx(100.0)
    assert railway.station_goods((6, 20)).stockpile[Good.COAL] == pytest.approx(100.0)


def test_a_train_unloads_only_what_fits_and_keeps_the_rest_aboard() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 10.0
    railway.station_goods((6, 20)).stockpile[Good.COAL] = STOCKPILE - 4.0
    railway.station_demand((6, 20)).capacity = {Need.FOOD: 200.0}
    train = railway.buy()
    assert train is not None
    train.stops = [
        Stop((6, 3), [Good.WHEAT], wait_for_load=False),
        Stop((6, 20), [Good.WHEAT], wait_for_load=False),
    ]

    railway.run(0.01)
    railway.run(25 / CELLS_PER_MONTH)

    assert railway.station_goods((6, 20)).brought[Good.WHEAT] == pytest.approx(4.0)
    assert [car.load for car in train.cars] == [pytest.approx(6.0)]
    assert railway.finances.month["cargo"] == 0.0


def test_goods_the_area_has_no_use_for_can_be_carried_on_by_another_train() -> None:
    _, railway = joined_stations()
    goods = railway.station_goods((6, 3))
    goods.bring(Good.WHEAT, {(6, 20): 75.0})
    railway.station_demand((6, 3)).capacity = {Need.FOOD: 40.0}
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.WHEAT], wait_for_load=False), Stop((6, 20))]

    railway.run(0.01)

    assert train.cars[0].load == pytest.approx(50.0)
    assert train.cars[0].origins == {(6, 20): pytest.approx(50.0)}
    assert goods.brought[Good.WHEAT] == pytest.approx(25.0)


def test_goods_the_area_wants_stay_for_it() -> None:
    _, railway = joined_stations()
    goods = railway.station_goods((6, 3))
    goods.bring(Good.WHEAT, {(6, 20): 4.0})
    railway.station_demand((6, 3)).capacity = {Need.FOOD: 8.0}
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.WHEAT], wait_for_load=False), Stop((6, 20))]

    railway.run(0.01)

    assert train.cars[0].load == 0.0


def test_cargo_back_at_the_station_it_came_from_returns_to_its_stockpile_unpaid() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), []), Stop((6, 20), [])]
    train.cars = [Car(Good.COAL)]
    train.cars[0].fill({(6, 20): 10.0})
    train.called = True
    train.loaded_at = (6, 3)

    railway.run(25 / CELLS_PER_MONTH)

    goods = railway.station_goods((6, 20))
    assert goods.stockpile[Good.COAL] == pytest.approx(10.0)
    assert goods.brought == {}
    assert railway.finances.month["cargo"] == 0.0


def test_gold_reaches_a_jeweller_through_a_hub_and_pays_only_when_he_takes_it() -> None:
    growth, railway = joined_stations()
    build_station(growth, (6, 27))
    route = plan_route(growth.constructions, (6, 20), (6, 27))
    assert route is not None
    lay_route(growth, route)
    for cell, kind in [((5, 5), BuildingKind.GOLD_MINE), ((5, 27), BuildingKind.JEWELLER)]:
        growth.build(Building(cell, kind))
    growth.trade_goods()
    first = railway.buy()
    assert first is not None
    first.stops = [Stop((6, 3), [Good.GOLD], wait_for_load=False), Stop((6, 20))]
    railway.run(0.01)
    railway.run(25 / CELLS_PER_MONTH)
    second = railway.buy()
    assert second is not None
    second.stops = [Stop((6, 20), [Good.GOLD], wait_for_load=False), Stop((6, 27))]
    railway.run(0.01)
    railway.run(12 / CELLS_PER_MONTH)
    before_taken = railway.finances.month["cargo"]

    growth.trade_goods()

    assert before_taken == 0.0
    assert railway.station_goods((6, 27)).brought[Good.GOLD] == pytest.approx(0.0)
    assert 10 * price_of(Good.GOLD) < railway.finances.month["cargo"] <= 20 * price_of(Good.GOLD)


def settle_houses(growth: Growth, cell: tuple[int, int], houses: int) -> None:
    for _ in range(houses):
        growth.build(Building(cell, BuildingKind.HOUSE))


def test_only_stations_that_track_joins_to_another_count_as_connected() -> None:
    growth, railway = joined_stations()
    build_station(growth, (2, 26))

    connected = railway.connected()

    assert connected == {(6, 3), (6, 20)}


def test_a_connected_station_passes_along_what_it_gathered_for_a_share_of_the_price() -> None:
    growth, railway = joined_stations()
    settle_houses(growth, (6, 21), 4)
    railway.station_goods((6, 20)).stockpile[Good.WHEAT] = 10.0
    growth.station_trade.measure_demand((6, 20))

    growth.trade_goods()

    assert railway.station_goods((6, 20)).stockpile[Good.WHEAT] == pytest.approx(0.0)
    earned = railway.finances.month["goods passed along"]
    assert 0.0 < earned <= 10 * PASSED_ALONG_PAY * price_of(Good.WHEAT)
    assert railway.finances.month["cargo"] == 0.0


def test_a_station_no_track_joins_to_another_passes_nothing_along() -> None:
    growth, railway = joined_stations()
    build_station(growth, (2, 26))
    settle_houses(growth, (2, 27), 4)
    railway.station_goods((2, 26)).stockpile[Good.WHEAT] = 10.0

    growth.trade_goods()

    assert railway.station_goods((2, 26)).stockpile[Good.WHEAT] == 10.0
    assert railway.finances.month["goods passed along"] == 0.0


def test_a_train_can_turn_round_at_a_station_it_passes_through() -> None:
    _, railway = joined_stations()
    entry = 0

    exits = railway.network._exits((6, 20), entry)

    assert entry in exits
    assert railway.network._exits((6, 3), None) == exits
