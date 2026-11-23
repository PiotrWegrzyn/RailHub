import pytest

from builders import joined_stations
from railroads.sim.facilities import COAL_A_MONTH, ENGINE_SHED_SPEED, WAREHOUSE_ROOM, Facility
from railroads.sim.finance import ENGINE_SHED_PRICE
from railroads.sim.goods import Good, Need
from railroads.sim.trains import Stop

STATION = (6, 3)


def test_putting_up_an_engine_shed_costs_its_price_once() -> None:
    _, railway = joined_stations()
    cash = railway.finances.cash

    first = railway.build_facility(STATION, Facility.ENGINE_SHED)
    second = railway.build_facility(STATION, Facility.ENGINE_SHED)

    assert (first, second) == (True, False)
    assert railway.finances.cash == pytest.approx(cash - ENGINE_SHED_PRICE)


@pytest.mark.parametrize(("coal", "works"), [(COAL_A_MONTH, True), (COAL_A_MONTH - 1, False)])
def test_an_engine_shed_works_a_month_only_on_a_months_coal(coal: float, works: bool) -> None:
    _, railway = joined_stations()
    railway.build_facility(STATION, Facility.ENGINE_SHED)
    railway.station_goods(STATION).stockpile[Good.COAL] = coal

    railway.turn_month({})

    assert (Facility.ENGINE_SHED in railway.working_at(STATION)) is works
    left = railway.station_goods(STATION).held(Good.COAL)
    assert left == pytest.approx(0.0 if works else coal)


def test_a_fuelled_train_runs_faster_by_the_engine_shed_speed() -> None:
    railways = [joined_stations()[1] for _ in range(2)]
    trains = []
    for railway in railways:
        railway.finances.earn(1_000_000.0, "test")
        train = railway.buy()
        assert train is not None
        train.stops = [Stop(STATION, wait_for_load=False), Stop((6, 20), wait_for_load=False)]
        railway.run(0.01)
        train.pace = 1.0
        trains.append(train)
    plain, fuelled = trains
    fuelled.fuelled = True
    before = (plain.along, fuelled.along)

    for railway in railways:
        railway.run(0.02)

    gained = (plain.along - before[0], fuelled.along - before[1])
    assert gained[1] == pytest.approx(gained[0] * ENGINE_SHED_SPEED, rel=0.05)


def test_a_station_whose_engine_shed_lacks_coal_wants_it() -> None:
    growth, railway = joined_stations()
    railway.build_facility(STATION, Facility.ENGINE_SHED)
    growth.trade_goods()

    short = growth.station_trade.short_of(STATION)

    assert short[Need.COAL] == pytest.approx(COAL_A_MONTH)


def test_a_working_post_office_makes_mail_delivered_there_pay_half_as_much_again() -> None:
    _, railway = joined_stations()
    railway.build_facility(STATION, Facility.POST_OFFICE)
    railway.station_goods(STATION).stockpile[Good.PAPER] = 50.0
    railway.turn_month({})

    factors = (railway.facility_pay(STATION, Good.MAIL), railway.facility_pay(STATION, Good.COAL))

    assert factors == (pytest.approx(1.5), 1.0)


def test_a_working_warehouse_gives_every_good_more_room() -> None:
    _, railway = joined_stations()
    before = railway.station_goods(STATION).limit
    railway.build_facility(STATION, Facility.WAREHOUSE)
    railway.station_goods(STATION).stockpile[Good.BRICKS] = 50.0
    railway.turn_month({})

    after = railway.station_goods(STATION).limit

    assert after == pytest.approx(before + WAREHOUSE_ROOM)
