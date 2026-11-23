import pytest

from builders import QUIET, flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions, Track
from railroads.sim.demolition import (
    STATION_REMOVAL,
    bulldoze,
    bulldoze_price,
    demolition_price,
    station_site_price,
)
from railroads.sim.game import Game
from railroads.sim.goods import Good
from railroads.sim.growth import Growth
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, move_station, plan_route, route_price
from railroads.sim.trains import Stop


def grown(*placed: tuple[tuple[int, int], BuildingKind]) -> Growth:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, QUIET)
    for cell, kind in placed:
        growth.build(Building(cell, kind))
    return growth


@pytest.mark.parametrize(
    ("kind", "price"),
    [
        (BuildingKind.HOUSE, 5_000),
        (BuildingKind.FARM, 5_000),
        (BuildingKind.MINE, 20_000),
        (BuildingKind.FACTORY, 15_000),
        (BuildingKind.MEGA_FACTORY, 135_000),
        (BuildingKind.SKYSCRAPER, 1_000_000),
        (BuildingKind.SPACE_CENTER, None),
        (BuildingKind.ST_PETERS, None),
    ],
)
def test_homes_and_fields_cost_least_to_clear_works_more_and_one_offs_cannot_be_cleared(
    kind: BuildingKind, price: float | None
) -> None:
    worth = demolition_price(kind)

    assert worth == price


def test_a_station_pays_for_every_building_it_moves_and_not_where_a_one_off_stands() -> None:
    growth = grown(
        ((5, 5), BuildingKind.HOUSE),
        ((5, 5), BuildingKind.HOUSE),
        ((5, 9), BuildingKind.ROYAL_PALACE),
    )

    houses = station_site_price(growth.constructions, (5, 5))
    palace = station_site_price(growth.constructions, (5, 9))

    assert houses == 10_000
    assert palace is None
    assert build_station(growth, (5, 9)) is None


def test_track_through_buildings_costs_what_moving_them_aside_does() -> None:
    growth = grown()
    route = plan_route(growth.constructions, (6, 3), (6, 9))
    assert route is not None
    plain = route_price(growth.constructions, route)
    middle = route.cells[1:-1]
    for cell in middle:
        for _ in range(6):
            growth.build(Building(cell, BuildingKind.HOUSE))

    price = route_price(growth.constructions, route)

    assert price == pytest.approx(plain + 5_000 * len(middle))


def test_the_bulldozer_clears_buildings_and_track_for_their_price_and_leaves_one_offs() -> None:
    growth = grown(((5, 5), BuildingKind.MINE), ((5, 5), BuildingKind.HOUSE))
    growth.build(Building((5, 9), BuildingKind.ROYAL_PALACE))
    build_station(growth, (5, 12))

    price = bulldoze_price(growth.constructions, (5, 5))
    bulldoze(growth, (5, 5))
    bulldoze(growth, (5, 9))

    assert price == 25_000
    assert growth.constructions.at((5, 5)) == ()
    assert growth.count(BuildingKind.MINE).sum() == 0
    assert growth.count(BuildingKind.ROYAL_PALACE).sum() == 1
    assert bulldoze_price(growth.constructions, (5, 12)) == STATION_REMOVAL


def test_a_moved_station_keeps_its_trains_name_and_goods_but_starts_its_upgrades_again() -> None:
    growth = grown()
    for cell in ((6, 3), (6, 20)):
        build_station(growth, cell)
    route = plan_route(growth.constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    railway = Railway(growth.constructions)
    name = railway.name_of((6, 20))
    railway.levels[(6, 20)] = 2
    railway.station_goods((6, 20)).stockpile[Good.COAL] = 40.0
    railway.station_goods((6, 20)).storage = 1
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3)), Stop((6, 20))]

    changed = move_station(growth, railway, (6, 20), (8, 22))

    assert changed is not None
    assert railway.stations() == [(6, 3), (8, 22)]
    assert [stop.cell for stop in train.stops] == [(6, 3), (8, 22)]
    assert railway.name_of((8, 22)) == name
    assert railway.level((8, 22)) == 1
    assert railway.station_goods((8, 22)).stockpile[Good.COAL] == 40.0
    assert railway.station_goods((8, 22)).storage == 0


def three_stations_on_a_line() -> Game:
    game = Game(Constructions(flat_map(cols=32)))
    game.finances.cash = 1_000_000.0
    for station in ((6, 3), (6, 14), (6, 26)):
        game.build_station(station)
    game.lay_track((6, 3), (6, 14))
    game.lay_track((6, 14), (6, 26))
    return game


def test_a_bulldozed_station_leaves_every_train_that_called_at_it() -> None:
    game = three_stations_on_a_line()
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3)), Stop((6, 14), [Good.COAL]), Stop((6, 26))]
    game.railway.run(0.05)

    done = game.bulldoze((6, 14))

    assert done.ok
    assert [stop.cell for stop in train.stops] == [(6, 3), (6, 26)]
    assert train.heading == 1
    assert not train.path


def test_a_train_left_with_one_stop_after_a_bulldozing_waits_for_a_second() -> None:
    game = three_stations_on_a_line()
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3)), Stop((6, 14))]

    game.bulldoze((6, 14))
    game.railway.run(0.05)

    assert [stop.cell for stop in train.stops] == [(6, 3)]
    assert not train.path


def junction() -> Growth:
    growth = grown()
    for piece in (
        Track((6, 9), (0, 6)),
        Track((6, 10), (0, 6)),
        Track((6, 11), (0, 6)),
        Track((6, 10), (2, 6)),
        Track((7, 10), (2, 8)),
    ):
        growth.constructions.build(piece)
    return growth


def test_bulldozing_a_branch_takes_up_its_stub_at_the_junction_it_left() -> None:
    growth = junction()

    changed = bulldoze(growth, (7, 10))

    assert growth.constructions.track_at((6, 10)) == (Track((6, 10), (0, 6)),)
    assert (6, 10) in changed


def test_bulldozing_the_line_beyond_a_junction_leaves_the_line_turning_into_the_branch() -> None:
    growth = junction()

    bulldoze(growth, (6, 11))

    assert growth.constructions.track_at((6, 10)) == (Track((6, 10), (2, 6)),)
    assert growth.constructions.track_at((6, 9)) == (Track((6, 9), (0, 6)),)


def test_bulldozing_track_leaves_the_plain_line_beside_it_to_be_laid_on_from() -> None:
    growth = grown()
    for cell in ((6, 9), (6, 10), (6, 11)):
        growth.constructions.build(Track(cell, (0, 6)))

    bulldoze(growth, (6, 11))

    assert growth.constructions.track_at((6, 10)) == (Track((6, 10), (0, 6)),)
