import pytest

from builders import joined_stations, make_game
from railroads.sim.constructions import CELL_SPACE, Building, BuildingKind
from railroads.sim.demolition import bulldoze
from railroads.sim.facilities import (
    DELIVERED_SHARE,
    ENGINE_SHED_SPEED,
    USES,
    WAREHOUSE_ROOM,
    Facility,
    monthly_cost,
)
from railroads.sim.game import Game
from railroads.sim.goods import Good, Need
from railroads.sim.growth import Growth
from railroads.sim.railway import Railway
from railroads.sim.routes import add_facility
from railroads.sim.trains import Stop

STATION = (6, 3)
BESIDE = (5, 3)
ON_THE_LINE = (6, 4)
COAL = USES[Facility.ENGINE_SHED][1]


def _with_shed() -> tuple[Growth, Railway]:
    growth, railway = joined_stations()
    add_facility(growth, STATION, Facility.ENGINE_SHED, BESIDE)
    return growth, railway


def _raised_game() -> Game:
    game = make_game(10_000_000.0)
    game.build_station(STATION)
    game.railway.upgrade(STATION)
    game.railway.upgrade(STATION)
    return game


def test_a_facility_costs_its_months_good_at_the_market_price_when_the_station_holds_none() -> None:
    _, railway = _with_shed()

    railway.turn_month({})

    spent = -railway.finances.month["station facilities"]
    assert spent == pytest.approx(COAL * railway.market_price(Good.COAL))


@pytest.mark.parametrize("held", [0.0, 20.0, COAL, COAL + 30.0])
def test_what_the_station_holds_costs_the_delivered_share(held: float) -> None:
    taken = min(held, COAL)

    cost = monthly_cost(Facility.ENGINE_SHED, held, 10.0)

    assert cost == pytest.approx((taken * DELIVERED_SHARE + COAL - taken) * 10.0)


def test_a_facility_takes_what_the_station_holds_and_pays_less_for_it() -> None:
    _, railway = _with_shed()
    railway.station_goods(STATION).stockpile[Good.COAL] = 20.0

    railway.turn_month({})

    spent = -railway.finances.month["station facilities"]
    price = railway.market_price(Good.COAL)
    assert spent == pytest.approx((20.0 * DELIVERED_SHARE + COAL - 20.0) * price)
    assert railway.station_goods(STATION).held(Good.COAL) == pytest.approx(0.0)


def test_an_engine_shed_fills_its_cell_taking_the_buildings_there_down() -> None:
    growth, railway = joined_stations()
    growth.build(Building(BESIDE, BuildingKind.HOUSE))

    add_facility(growth, STATION, Facility.ENGINE_SHED, BESIDE)

    assert growth.constructions.buildings_at(BESIDE) == []
    assert growth.constructions.space_left(BESIDE) == 0
    assert railway.facilities_at(STATION) == {Facility.ENGINE_SHED}


@pytest.mark.parametrize(("houses", "left"), [(4, 4), (CELL_SPACE, CELL_SPACE - 1)])
def test_cattle_pens_take_one_place_in_their_cell(houses: int, left: int) -> None:
    growth, _ = joined_stations()
    for _ in range(houses):
        growth.build(Building(BESIDE, BuildingKind.HOUSE))

    add_facility(growth, STATION, Facility.CATTLE_PENS, BESIDE)

    assert len(growth.constructions.buildings_at(BESIDE)) == left


def test_an_engine_shed_needs_the_station_raised_to_level_3() -> None:
    game = make_game(10_000_000.0)
    game.build_station(STATION)
    early = game.add_facility(STATION, Facility.ENGINE_SHED, BESIDE)
    game.railway.upgrade(STATION)
    game.railway.upgrade(STATION)

    late = game.add_facility(STATION, Facility.ENGINE_SHED, BESIDE)

    assert (early.ok, late.ok) == (False, True)


@pytest.mark.parametrize(("cell", "ok"), [(BESIDE, True), ((4, 3), False), (STATION, False)])
def test_an_engine_shed_stands_next_to_its_station(cell: tuple[int, int], ok: bool) -> None:
    game = _raised_game()

    done = game.add_facility(STATION, Facility.ENGINE_SHED, cell)

    assert done.ok is ok


def test_an_engine_shed_put_up_on_the_line_takes_the_place_of_its_track() -> None:
    game = _raised_game()
    game.lay_track(STATION, (6, 8))

    shed = game.add_facility(STATION, Facility.ENGINE_SHED, ON_THE_LINE)

    assert shed.ok
    assert game.constructions.track_at(ON_THE_LINE) == ()
    assert game.constructions.shed_at(ON_THE_LINE) is not None


def test_cattle_pens_stand_beside_the_track_on_the_line() -> None:
    game = _raised_game()
    game.lay_track(STATION, (6, 8))

    pens = game.add_facility(STATION, Facility.CATTLE_PENS, ON_THE_LINE)

    assert pens.ok
    assert game.constructions.track_at(ON_THE_LINE) != ()


def test_a_station_has_one_facility_of_each_kind() -> None:
    game = _raised_game()
    game.add_facility(STATION, Facility.ENGINE_SHED, BESIDE)

    again = game.add_facility(STATION, Facility.ENGINE_SHED, (7, 3))

    assert again.ok is False


def test_the_other_facilities_go_in_the_station_itself_and_not_on_the_map() -> None:
    game = _raised_game()

    placed = game.add_facility(STATION, Facility.POST_OFFICE, BESIDE)
    put_up = game.railway.put_up(STATION, Facility.POST_OFFICE)

    assert (placed.ok, put_up) == (False, True)
    assert game.constructions.at(BESIDE) == ()
    assert game.railway.facilities_at(STATION) == {Facility.POST_OFFICE}


def test_closing_a_facility_in_the_station_stops_its_cost() -> None:
    _, railway = joined_stations()
    railway.put_up(STATION, Facility.POST_OFFICE)

    railway.close_facility(STATION, Facility.POST_OFFICE)
    railway.turn_month({})

    assert railway.finances.month["station facilities"] == 0.0


def test_bulldozing_a_station_takes_its_facilities_down() -> None:
    growth, railway = _with_shed()

    bulldoze(growth, STATION)

    assert growth.constructions.annex_at(BESIDE) is None
    assert railway.facilities_at(STATION) == set()


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


def test_a_station_wants_what_its_facilities_will_use_beyond_what_it_holds() -> None:
    growth, railway = _with_shed()
    growth.trade_goods()
    railway.station_goods(STATION).stockpile[Good.COAL] = 20.0

    short = growth.station_trade.short_of(STATION)

    assert short[Need.COAL] == pytest.approx(COAL - 20.0)


def test_a_post_office_makes_mail_delivered_there_pay_half_as_much_again() -> None:
    _, railway = joined_stations()
    railway.put_up(STATION, Facility.POST_OFFICE)

    factors = (railway.facility_pay(STATION, Good.MAIL), railway.facility_pay(STATION, Good.COAL))

    assert factors == (pytest.approx(1.5), 1.0)


def test_a_warehouse_gives_every_good_more_room() -> None:
    _, railway = joined_stations()
    before = railway.station_goods(STATION).limit

    railway.put_up(STATION, Facility.WAREHOUSE)

    after = railway.station_goods(STATION).limit
    assert after == pytest.approx(before + WAREHOUSE_ROOM)


@pytest.mark.parametrize("facility", [Facility.ENGINE_SHED, Facility.CATTLE_PENS])
def test_no_station_or_platform_goes_up_on_a_facility(facility: Facility) -> None:
    game = _raised_game()
    for _ in range(2):
        game.railway.upgrade(STATION)
    assert game.add_facility(STATION, facility, BESIDE).ok

    quote = game.station_quote(BESIDE)
    platform = game.platform_quote(BESIDE)
    station = game.build_station(BESIDE)
    added = game.add_platform(STATION, BESIDE)

    assert quote.price is None
    assert platform.price is None
    assert not station.ok
    assert not added.ok
