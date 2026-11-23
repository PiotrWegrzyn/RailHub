import numpy as np
import pytest

from builders import flat_map, joined_stations
from railroads.sim.constructions import Building, BuildingKind, Constructions, Track
from railroads.sim.demand import DEMAND_PER_NEED
from railroads.sim.economy import Economy
from railroads.sim.finance import (
    CAR_TRIP_PRICE,
    STATION_UPKEEP,
    STORAGE_GROWTH,
    STORAGE_PRICE,
    TRACK_UPKEEP,
    TRAIN_PRICE,
    car_trip_price,
)
from railroads.sim.goods import Good, Need, freight_of
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.railway import CELLS_PER_MONTH, Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.station_goods import STOCKPILE, STORAGE_STEP, WAITING, StationGoods
from railroads.sim.trade import SCARCE_WORTH, StationOrder
from railroads.sim.trains import (
    BULK_CAR,
    CAR_CAPACITY,
    ENGINES,
    MAX_CARS,
    SMALL_CAR,
    Car,
    Stop,
    car_size,
)
from railroads.sim.yearly import needs_of


def test_trains_find_their_way_along_track_between_stations() -> None:
    _, railway = joined_stations()

    way = railway.route((6, 3), (6, 20))

    assert way is not None
    points, speeds = way
    assert len(speeds) == len(points) - 1
    assert railway.route((6, 3), (2, 3)) is None


def test_a_new_train_leaves_its_first_stop_with_that_stops_cars_full() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = BULK_CAR + 25.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT, Good.WHEAT], wait_for_load=False), Stop((6, 20))]

    railway.run(0.1)

    assert train.heading == 1
    assert [car.load for car in train.cars] == [BULK_CAR, 25.0]
    assert railway.position(train) is not None


@pytest.mark.parametrize(
    ("good", "load"),
    [
        (Good.GOLD, 5.0),
        (Good.STEEL, 13.0),
        (Good.CATTLE, 8.0),
        (Good.PIGS, 21.0),
        (Good.OIL, 55.0),
        (Good.FISH, 34.0),
        (Good.IRON, 89.0),
        (Good.WHEAT, 89.0),
        (Good.BREAD, 89.0),
        (Good.PASSENGERS, 34.0),
    ],
)
def test_a_car_of_the_planets_day_holds_as_many_units_as_fit_in_a_real_wagon_of_it(
    good: Good, load: float
) -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[good] = 200.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [good], wait_for_load=False), Stop((6, 20))]

    railway.run(0.1)

    assert [car.load for car in train.cars] == [load]
    assert railway.station_goods((6, 3)).stockpile[good] == pytest.approx(200.0 - load)


def test_a_train_waits_at_its_stop_until_new_goods_fill_its_cars() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT], wait_for_load=True), Stop((6, 20))]
    railway.run(0.1)
    waiting = train.status

    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = BULK_CAR + 10.0
    railway.run(0.1)

    assert waiting == "Loading at 6, 3"
    assert [car.load for car in train.cars] == [BULK_CAR]
    assert train.status == "Running"


def test_a_train_waiting_for_a_full_load_waits_however_long_its_cars_take_to_fill() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT], wait_for_load=True), Stop((6, 20))]

    for _ in range(120):
        railway.run(0.1)

    assert train.status == "Loading at 6, 3"
    assert train.path == []


def test_a_train_stops_waiting_as_soon_as_its_stop_no_longer_waits_for_a_full_load() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    stop = Stop((6, 3), [Good.WHEAT], wait_for_load=True)
    train.stops = [stop, Stop((6, 20))]
    railway.run(0.1)
    waiting = train.status

    stop.wait_for_load = False
    railway.run(0.1)

    assert waiting == "Loading at 6, 3"
    assert train.status == "Running"


def test_at_the_next_stop_it_unloads_and_takes_that_stops_cars() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = BULK_CAR
    railway.station_goods((6, 20)).stockpile[Good.OIL] = 150.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT]), Stop((6, 20), [Good.OIL, Good.OIL])]

    railway.run(0.1)
    railway.run(80 / CELLS_PER_MONTH)

    assert train.heading == 0
    assert railway.station_goods((6, 20)).brought[Good.WHEAT] == BULK_CAR
    assert [(car.good, car.load) for car in train.cars] == [(Good.OIL, 55.0), (Good.OIL, 55.0)]


def test_each_car_fills_with_its_own_good_from_the_stockpile() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.PLANKS, Good.PLANKS, Good.WHEAT]), Stop((6, 20))]
    railway.station_goods((6, 3)).stockpile.update(
        {Good.PLANKS: car_size(Good.PLANKS) + 20.0, Good.IRON: 150.0}
    )

    railway.run(0.01)

    assert [car.load for car in train.cars] == [car_size(Good.PLANKS), 20.0, 0.0]
    assert railway.station_goods((6, 3)).stockpile == {Good.PLANKS: 0.0, Good.IRON: 150.0}


def test_a_stop_sends_at_most_max_cars() -> None:
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


def test_a_station_hands_brought_goods_to_buildings_around_and_gathers_the_spare() -> None:
    tile_map = flat_map()
    economy = Economy(tile_map)
    counts = {kind: np.zeros(tile_map.grid.shape, dtype=np.int64) for kind in BuildingKind}
    counts[BuildingKind.FLATS][6, 10] = 6
    counts[BuildingKind.LOGGING_CAMP][6, 25] = 1
    orders = {
        (6, 11): StationOrder({Good.WHEAT: 20.0}, 0.0),
        (6, 24): StationOrder({}, 50.0),
    }

    trade = economy.trade(counts, orders)

    index = int(np.searchsorted(trade.cells, np.ravel_multi_index((6, 25), tile_map.grid.shape)))
    assert trade.handed[(6, 11)][Good.WHEAT] == pytest.approx(20.0)
    assert trade.gathered[(6, 24)][Good.LOGS] == pytest.approx(20.0)
    assert trade.gave[BuildingKind.LOGGING_CAMP][index] == pytest.approx(20.0)
    assert sum(trade.gathered[(6, 24)].values()) <= 50.0


def test_growth_hands_station_goods_out_and_stores_what_was_gathered() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.LOGGING_CAMP))

    growth.trade_goods()

    goods = railway.station_goods((6, 20))
    assert goods.stockpile[Good.LOGS] == pytest.approx(20.0)
    assert goods.held(Good.LOGS) <= STOCKPILE


def test_cars_slow_a_train_and_full_cars_slow_it_more() -> None:
    _, railway = joined_stations()
    railway.finances.earn(1_000_000.0, "test")
    light, empty, full = railway.buy(), railway.buy(), railway.buy()
    cars = ENGINES["planet"].cars
    empty.cars = [Car(Good.MAIL) for _ in range(cars)]
    full.cars = [Car(Good.MAIL, CAR_CAPACITY) for _ in range(cars)]

    pulls = [railway.pull(train) for train in (light, empty, full)]

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


def test_new_cars_at_the_stop_a_train_stands_at_load_there_at_once() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 10.0
    train = railway.buy()
    train.stops = [Stop((6, 3)), Stop((6, 20))]

    train.stops[0].add_car(Good.WHEAT)
    train.cars_changed(0)
    railway.run(0.001)

    assert [(car.good, car.load) for car in train.cars] == [(Good.WHEAT, 10.0)]


@pytest.mark.parametrize("index", [0, 1])
def test_a_running_train_keeps_its_way_when_cars_change_and_takes_them_at_the_next_call(
    index: int,
) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    railway.run(0.05)
    heading, path = train.heading, list(train.path)

    train.stops[index].add_car(Good.COAL)
    train.cars_changed(index)

    assert path
    assert (train.heading, train.path, train.called) == (heading, path, True)
    assert train.cars == []


def test_new_cars_at_another_stop_leave_a_standing_train_be() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 10.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT], wait_for_load=False), Stop((6, 20))]
    railway.run(0.001)
    loaded = [(car.good, car.load) for car in train.cars]

    train.stops[1].add_car(Good.COAL)
    train.cars_changed(1)

    assert loaded == [(Good.WHEAT, 10.0)]
    assert train.called


def test_removing_one_loaded_car_keeps_the_others_full_and_returns_its_goods() -> None:
    _, railway = joined_stations()
    stockpile = railway.station_goods((6, 3)).stockpile
    cars = ENGINES["planet"].cars
    stockpile[Good.WHEAT] = (cars - 1) * BULK_CAR + 10.0
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT] * cars, wait_for_load=True), Stop((6, 20))]
    railway.run(0.001)

    train.stops[0].cars.pop()
    train.cars_changed(0)
    railway.run(0.001)

    assert [car.load for car in train.cars] == [BULK_CAR] * (cars - 1)
    assert stockpile[Good.WHEAT] == pytest.approx(10.0)
    assert railway.station_goods((6, 3)).brought == {}


def test_a_train_costs_its_price_and_sells_for_half() -> None:
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
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = BULK_CAR
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.WHEAT]), Stop((6, 20))]
    railway.run(0.01)
    railway.run(80 / CELLS_PER_MONTH)
    on_arrival = railway.finances.month["cargo"]

    growth.trade_goods()

    goods = railway.station_goods((6, 20))
    assert on_arrival == 0.0
    assert goods.brought[Good.WHEAT] == pytest.approx(BULK_CAR - 80.0)
    assert 0.0 < railway.finances.month["cargo"] <= 80 * freight_of(Good.WHEAT)


def test_each_month_stations_and_track_cost_their_upkeep() -> None:
    growth, railway = joined_stations()
    track = {built.cell for built in growth.constructions if isinstance(built, Track)}
    cash = railway.finances.cash

    railway.turn_month({})

    assert railway.finances.cash == cash - 2 * STATION_UPKEEP[0] - len(track) * TRACK_UPKEEP


def test_station_demand_is_twice_the_needs_of_buildings_in_its_inner_ring() -> None:
    growth, railway = joined_stations()
    for cell in [(6, 21)] * 6 + [(5, 20)] * 6 + [(6, 25)] * 2:
        growth.build(Building(cell, BuildingKind.HOUSE))

    growth.trade_goods()

    demand = railway.station_demand((6, 20))
    inner_ring = 12
    needs = needs_of(BuildingKind.HOUSE, growth.year)
    travellers = growth.station_trade.travellers_in(growth.station_trade.inner_ring((6, 20)))
    assert demand.capacity == pytest.approx(
        {need: DEMAND_PER_NEED * inner_ring * amount for need, amount in needs.items()}
        | {Need(good.value): DEMAND_PER_NEED * amount for good, amount in travellers.items()}
    )
    assert demand.record(Need.FOOD).left == pytest.approx(demand.capacity[Need.FOOD])


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
    assert goods.stockpile[Good.PASSENGERS] == 2 + 10 + 5
    assert goods.stockpile[Good.MAIL] == 3
    assert demand.capacity[Need.PASSENGERS] == 2 * 17


def test_waiting_passengers_have_their_own_room_and_leave_freight_its_stockpile() -> None:
    goods = StationGoods()
    goods.stockpile[Good.COAL] = 40.0

    goods.board(Good.PASSENGERS, 170.0)

    assert goods.stockpile[Good.PASSENGERS] == WAITING
    assert goods.fullest == (Good.COAL, 40.0)


def test_last_months_passengers_and_mail_give_way_to_this_months() -> None:
    goods = StationGoods()
    goods.board(Good.PASSENGERS, 70.0)
    goods.board(Good.MAIL, 14.0)

    goods.board(Good.PASSENGERS, 30.0)
    goods.board(Good.MAIL, 6.0)

    assert goods.stockpile[Good.PASSENGERS] == 30.0
    assert goods.stockpile[Good.MAIL] == 6.0


def test_passengers_pay_on_arrival_and_go_on_their_way() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.PASSENGERS] = 30.0
    railway.station_demand((6, 20)).capacity = {Need.PASSENGERS: 200.0}
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.PASSENGERS]), Stop((6, 20))]

    railway.run(0.01)
    railway.run(80 / CELLS_PER_MONTH)

    assert railway.finances.month["passengers"] > 0.95 * 30 * freight_of(Good.PASSENGERS)
    assert Good.PASSENGERS not in railway.station_goods((6, 20)).brought


def test_a_station_rises_through_five_levels_for_50k_250k_1_5m_and_10m_each_reaching_further() -> (
    None
):
    _, railway = joined_stations()
    railway.finances.cash = 12_000_000.0

    reaches = [railway.reach((6, 20))]
    for _ in range(5):
        railway.upgrade((6, 20))
        reaches.append(railway.reach((6, 20)))

    assert reaches == [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (5, 7)]
    assert railway.finances.cash == 12_000_000.0 - 11_800_000.0
    assert railway.upgrade_price((6, 20)) is None


def test_each_level_adds_as_much_room_as_two_storage_upgrades_without_making_them_dearer() -> None:
    _, railway = joined_stations()
    goods = railway.station_goods((6, 20))
    before, price = goods.limit, railway.storage_price((6, 20))

    railway.upgrade((6, 20))

    assert goods.limit == before + 2 * STORAGE_STEP
    assert railway.storage_price((6, 20)) == price


def test_station_upkeep_grows_with_each_level_and_rises_two_percent_a_year() -> None:
    _, railway = joined_stations()
    railway.first_year = 1850
    railway.year = 1852
    railway.levels[(6, 20)] = 3

    upkeep = railway.station_upkeep((6, 3)), railway.station_upkeep((6, 20))

    assert upkeep == (pytest.approx(150 * 1.02**2), pytest.approx(1200 * 1.02**2))


def test_without_the_cash_a_station_stays_as_it_is() -> None:
    _, railway = joined_stations()
    railway.finances.cash = 49_999.0

    raised = railway.upgrade((6, 20))

    assert not raised
    assert railway.level((6, 20)) == 1


def test_each_storage_upgrade_adds_to_each_good_and_to_the_waiting_room() -> None:
    _, railway = joined_stations()
    cash = railway.finances.cash

    railway.add_storage((6, 20))
    railway.add_storage((6, 20))

    goods = railway.station_goods((6, 20))
    assert (goods.limit, goods.waiting_limit) == (500.0 + 2 * 250, WAITING + 1000)
    assert railway.finances.cash == cash - STORAGE_PRICE * (1 + STORAGE_GROWTH)


def test_each_storage_upgrade_costs_twice_the_one_before() -> None:
    _, railway = joined_stations()
    prices = []

    for _ in range(3):
        prices.append(railway.storage_price((6, 20)))
        railway.add_storage((6, 20))

    assert prices == [STORAGE_PRICE, 2 * STORAGE_PRICE, 4 * STORAGE_PRICE]


def test_a_raised_station_gathers_from_further_away() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 24), BuildingKind.LOGGING_CAMP))
    growth.trade_goods()
    before = railway.station_goods((6, 20)).stockpile.get(Good.LOGS, 0.0)
    railway.finances.cash = 100_000.0
    railway.upgrade((6, 20))

    growth.trade_goods()

    assert before == 0.0
    assert railway.station_goods((6, 20)).stockpile[Good.LOGS] == pytest.approx(20.0)


def test_every_car_costs_fifteen_and_a_dollar_for_every_ten_it_holds_each_time_it_sets_off() -> (
    None
):
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.COAL, Good.WHEAT], wait_for_load=False), Stop((6, 20))]

    railway.run(0.01)

    assert railway.finances.month["car trips"] == -2 * (CAR_TRIP_PRICE + 8)


def test_a_mine_feeds_a_steel_mill_two_cells_off_and_the_station_gathers_the_rest() -> None:
    growth, railway = joined_stations()
    for cell, kind in [((6, 23), BuildingKind.MINE), ((6, 21), BuildingKind.STEEL_MILL)]:
        growth.build(Building(cell, kind))

    growth.trade_goods()

    mill = np.ravel_multi_index((6, 21), growth.constructions.tile_map.grid.shape)
    assert growth.economy.stock[(BuildingKind.STEEL_MILL, Need.COAL)][mill] == pytest.approx(
        100 * 8 / 21
    )
    assert railway.station_goods((6, 20)).stockpile[Good.COAL] == pytest.approx(100 * 13 / 21)


def test_a_train_unloads_only_what_fits_and_keeps_the_rest_aboard() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 10.0
    railway.station_goods((6, 20)).bring(Good.WHEAT, {(6, 3): STOCKPILE - 4.0})
    railway.station_demand((6, 20)).capacity = {Need.FOOD: 2 * STOCKPILE}
    train = railway.buy()
    assert train is not None
    train.stops = [
        Stop((6, 3), [Good.WHEAT], wait_for_load=False),
        Stop((6, 20), [Good.WHEAT], wait_for_load=False),
    ]

    railway.run(0.01)
    railway.run(80 / CELLS_PER_MONTH)

    assert railway.station_goods((6, 20)).held(Good.WHEAT) <= STOCKPILE
    assert [car.load for car in train.cars] == [pytest.approx(6.0)]
    assert railway.finances.month["cargo"] == 0.0


def test_goods_the_area_has_no_use_for_can_be_carried_on_by_another_train() -> None:
    _, railway = joined_stations()
    goods = railway.station_goods((6, 3))
    goods.bring(Good.WHEAT, {(6, 20): BULK_CAR + 25.0})
    railway.station_demand((6, 3)).capacity = {Need.FOOD: 40.0}
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.WHEAT], wait_for_load=False), Stop((6, 20))]

    railway.run(0.01)

    assert train.cars[0].load == pytest.approx(BULK_CAR)
    assert train.cars[0].origins == {(6, 20): pytest.approx(BULK_CAR)}
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

    railway.run(80 / CELLS_PER_MONTH)

    goods = railway.station_goods((6, 20))
    assert goods.stockpile[Good.COAL] == pytest.approx(10.0)
    assert goods.brought == {}
    assert railway.finances.month["cargo"] == 0.0


def test_gold_reaches_a_jeweller_through_a_hub_and_pays_the_moment_he_takes_it() -> None:
    growth, railway = joined_stations()
    railway.finances.earn(1_000_000.0, "test")
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
    railway.run(80 / CELLS_PER_MONTH)
    second = railway.buy()
    assert second is not None
    second.stops = [Stop((6, 20), [Good.GOLD], wait_for_load=False), Stop((6, 27))]
    railway.run(0.01)

    railway.run(40 / CELLS_PER_MONTH)

    assert railway.station_goods((6, 27)).brought[Good.GOLD] == pytest.approx(0.0)
    assert (
        4 * freight_of(Good.GOLD)
        < railway.finances.month["cargo"]
        <= SMALL_CAR * freight_of(Good.GOLD) * SCARCE_WORTH
    )


def settle_houses(growth: Growth, cell: tuple[int, int], houses: int) -> None:
    for _ in range(houses):
        growth.build(Building(cell, BuildingKind.HOUSE))


def test_only_stations_that_track_joins_to_another_count_as_connected() -> None:
    growth, railway = joined_stations()
    build_station(growth, (2, 26))

    connected = railway.connected()

    assert connected == {(6, 3), (6, 20)}


def test_a_station_hands_what_it_gathered_to_the_houses_around_and_earns_its_freight() -> None:
    growth, railway = joined_stations()
    settle_houses(growth, (6, 21), 4)
    railway.station_goods((6, 20)).stockpile[Good.WHEAT] = 10.0
    growth.station_trade.measure_demand((6, 20))

    growth.trade_goods()

    assert railway.station_goods((6, 20)).stockpile[Good.WHEAT] < 10.0
    assert railway.finances.month["cargo"] > 0.0


def test_a_train_can_turn_round_at_a_station_it_passes_through() -> None:
    _, railway = joined_stations()
    entry = 0

    exits = railway.network._exits((6, 20), entry)

    assert entry in exits
    assert railway.network._exits((6, 3), None) == exits


def test_a_station_holds_five_hundred_of_each_good_counting_what_trains_brought() -> None:
    goods = StationGoods()
    goods.stockpile[Good.COAL] = 300.0
    goods.brought[Good.COAL] = 200.0
    goods.stockpile[Good.WHEAT] = 120.0

    rooms = (goods.room(Good.COAL), goods.room(Good.WHEAT), goods.room(Good.PLANKS))

    assert rooms == (0.0, 380.0, 500.0)


def test_a_storage_upgrade_gives_every_good_two_hundred_and_fifty_more_room() -> None:
    _, railway = joined_stations()
    goods = railway.station_goods((6, 20))
    goods.stockpile[Good.COAL] = 500.0

    railway.add_storage((6, 20))

    assert (goods.room(Good.COAL), goods.room(Good.WHEAT)) == (250.0, 750.0)


def test_a_train_unloads_a_good_at_a_station_full_of_another() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 10.0
    railway.station_goods((6, 20)).stockpile[Good.COAL] = STOCKPILE
    railway.station_demand((6, 20)).capacity = {Need.FOOD: 200.0}
    train = railway.buy()
    assert train is not None
    train.stops = [
        Stop((6, 3), [Good.WHEAT], wait_for_load=False),
        Stop((6, 20), [Good.WHEAT], wait_for_load=False),
    ]

    railway.run(0.01)
    railway.run(80 / CELLS_PER_MONTH)

    assert railway.station_goods((6, 20)).brought[Good.WHEAT] == pytest.approx(10.0)


def stations_with_houses(station: Cell, houses: int) -> tuple[Growth, Railway]:
    growth, railway = joined_stations()
    railway.station_trade = growth.station_trade
    for _ in range(houses):
        growth.build(Building((station[0], station[1] + 1), BuildingKind.HOUSE))
    return growth, railway


def test_a_trains_cargo_is_worth_what_the_buildings_at_its_destination_would_pay_for_it() -> None:
    _, railway = stations_with_houses((6, 20), 2)
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.WHEAT]), Stop((6, 20))]
    train.cars = [Car(Good.WHEAT)]
    train.cars[0].fill({(6, 3): 60.0})

    worth = railway.cargo_worth(train)
    railway.station_goods((6, 20)).bring(Good.WHEAT, {(6, 3): 60.0})
    railway.hand_out_now((6, 20), Good.WHEAT)

    assert railway.destination(train) == (6, 20)
    assert worth == [pytest.approx(railway.finances.month["cargo"])]
    assert worth[0] > 0


def test_cargo_is_worth_freight_only_for_what_the_buildings_around_still_want() -> None:
    _, railway = stations_with_houses((6, 20), 1)

    small, big = (
        railway.worth_at((6, 20), [(Good.WHEAT, 5.0)]),
        railway.worth_at((6, 20), [(Good.WHEAT, 500.0)]),
    )
    first, second = railway.worth_at((6, 20), [(Good.WHEAT, 500.0), (Good.WHEAT, 500.0)])

    assert big[0] > small[0]
    assert first == pytest.approx(big[0])
    assert second == 0.0


def test_cargo_going_back_where_it_came_from_is_worth_nothing() -> None:
    _, railway = stations_with_houses((6, 3), 2)
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 20)), Stop((6, 3))]
    train.cars = [Car(Good.WHEAT)]
    train.cars[0].fill({(6, 3): 20.0, (6, 20): 10.0})

    worth = railway.cargo_worth(train)

    assert worth == [pytest.approx(railway.worth_at((6, 3), [(Good.WHEAT, 10.0)])[0])]
    assert worth[0] > 0


def test_a_train_with_one_stop_has_no_destination() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3))]
    train.cars = [Car(Good.WHEAT)]

    worth = railway.cargo_worth(train)

    assert railway.destination(train) is None
    assert worth == [0.0]


def test_a_train_bought_after_another_calls_at_its_stops_with_their_cars_and_flags() -> None:
    _, railway = joined_stations()
    railway.finances.earn(1_000_000.0, "test")
    first = railway.buy()
    first.stops = [
        Stop((6, 3), [Good.WHEAT], wait_for_load=True),
        Stop((6, 12), refuel_only=True),
        Stop((6, 20), [Good.COAL]),
    ]

    second = railway.buy(route_of=first)

    assert [stop.cell for stop in second.stops] == [(6, 3), (6, 12), (6, 20)]
    assert [stop.cars for stop in second.stops] == [[Good.WHEAT], [], [Good.COAL]]
    assert [stop.wait_for_load for stop in second.stops] == [True, False, False]
    assert [stop.refuel_only for stop in second.stops] == [False, True, False]
    assert second.stops[0] is not first.stops[0]


def test_a_train_bought_after_another_leaves_out_the_cars_its_engine_cannot_pull() -> None:
    _, railway = joined_stations()
    railway.finances.earn(1_000_000.0, "test")
    first = railway.buy()
    first.stops = [Stop((6, 3), [Good.WHEAT] * 30), Stop((6, 20))]

    second = railway.buy(route_of=first)

    assert 0 < len(second.stops[0].cars) <= railway.most_cars(second)
    assert len(first.stops[0].cars) == 30


def test_a_train_bought_on_its_own_has_no_stops() -> None:
    _, railway = joined_stations()

    train = railway.buy()

    assert train.stops == []


def food_and_heating(growth: Growth, cell: tuple[int, int]) -> tuple[float, float]:
    index = np.ravel_multi_index(cell, growth.constructions.tile_map.grid.shape)
    position = int(np.searchsorted(growth.last_trade.cells, index))
    food = growth.last_trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)][0][position]
    heating = growth.last_trade.goods_got[(BuildingKind.HOUSE, Need.HEATING)][0][position]
    return float(food), float(heating)


def test_a_town_eats_its_own_farms_food_the_month_it_is_made_and_only_the_rest_waits() -> None:
    growth, railway = joined_stations()
    for cell in ((5, 22), (6, 22), (6, 23), (7, 22)):
        growth.build(Building(cell, BuildingKind.FARM))
    growth.build(Building((6, 21), BuildingKind.HOUSE))

    growth.trade_goods()

    food, _ = food_and_heating(growth, (6, 21))
    assert food == pytest.approx(10 * (13 + 3 * 8) / 21)
    assert railway.station_goods((6, 20)).stockpile[Good.WHEAT] == pytest.approx(4 * 10 - food)


def test_coal_mined_near_a_station_heats_its_own_houses_before_waiting_for_trains() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 23), BuildingKind.MINE))
    growth.build(Building((6, 21), BuildingKind.HOUSE))

    growth.trade_goods()

    _, heating = food_and_heating(growth, (6, 21))
    assert heating == pytest.approx(3.0)
    assert railway.station_goods((6, 20)).stockpile[Good.COAL] == pytest.approx(97.0)


@pytest.mark.parametrize(("farm", "demand"), [(False, 40.0), (True, 2 * (20 - 2 * 10 * 13 / 21))])
def test_a_station_wants_only_what_its_houses_were_left_short_of_by_the_farms_around(
    farm: bool, demand: float
) -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.HOUSE))
    if farm:
        for cell in ((6, 22), (7, 21)):
            growth.build(Building(cell, BuildingKind.FARM))

    growth.trade_goods()

    assert railway.station_demand((6, 20)).capacity.get(Need.FOOD, 0.0) == pytest.approx(demand)


def test_a_new_stop_does_not_wait_for_a_full_load() -> None:
    stop = Stop((6, 3))

    waits = stop.wait_for_load

    assert waits is False


def test_a_held_train_stands_at_its_first_stop_until_let_go() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    train.held = True
    railway.run(0.1)
    stood = not train.path

    railway.let_go()
    railway.run(0.1)

    assert stood
    assert train.path
    assert train.status == "Running"


def test_a_rides_bill_takes_the_fuel_and_every_cars_trip_off_what_its_cargo_earns() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = BULK_CAR
    train = railway.buy()
    train.stops = [Stop((6, 3), [Good.WHEAT, Good.WHEAT]), Stop((6, 20))]
    railway.run(0.001)
    cells = railway.network.route_length((6, 3), (6, 20))
    assert cells is not None

    ride = railway.ride(train, [300.0, 0.0])
    loaded_share = railway.burn(train) / ENGINES[train.engine].burn

    assert ride is not None
    assert ride.cells == pytest.approx(cells, rel=0.05)
    assert (ride.loaded, ride.empty) == (1, 1)
    assert ride.fuel == pytest.approx(railway.fuel_cost(train.engine, ride.cells) * loaded_share)
    assert ride.loaded_cars == pytest.approx(car_trip_price(train.cars[0].capacity))
    assert ride.empty_cars == pytest.approx(car_trip_price(train.cars[0].capacity))
    assert ride.net == pytest.approx(300.0 - ride.fuel - 2 * car_trip_price(train.cars[0].capacity))


def test_a_train_with_one_stop_has_no_ride() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    train.stops = [Stop((6, 3))]

    ride = railway.ride(train)

    assert ride is None
