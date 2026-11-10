import numpy as np
import pytest

from railroads.sim.constructions import Building, BuildingKind, Constructions, Station
from railroads.sim.finance import STATION_PRICE, dollars
from railroads.sim.game import Game
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.routes import plan_route, route_price
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.sim.trains import Stop


def make_map() -> TileMap:
    grid = HexGrid(rows=12, cols=30, km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )


def make_game(cash: float = 1_000_000.0) -> Game:
    game = Game(Constructions(make_map()))
    game.finances.cash = cash
    return game


def test_a_station_costs_its_price_and_clearing_its_site() -> None:
    game = make_game()
    for _ in range(2):
        game.growth.build(Building((5, 5), BuildingKind.HOUSE))

    done = game.build_station((5, 5))

    assert done.ok
    assert (5, 5) in done.changed
    assert game.finances.cash == 1_000_000.0 - STATION_PRICE - 1_000
    assert done.message == f"Station built for {dollars(STATION_PRICE + 1_000)}."


def test_nothing_is_built_or_paid_without_the_cash_for_it() -> None:
    game = make_game(cash=100.0)

    done = game.build_station((5, 5))

    assert not done.ok
    assert done.changed == set()
    assert game.finances.cash == 100.0
    assert game.railway.stations() == []


def test_track_costs_what_its_route_does_and_laying_it_again_is_free() -> None:
    game = make_game()
    route = plan_route(game.constructions, (6, 3), (6, 9))
    assert route is not None
    price = route_price(game.constructions, route)

    first = game.lay_track((6, 3), (6, 9))
    again = game.lay_track((6, 3), (6, 9))

    assert first.ok and again.ok
    assert again.message == "That track is laid already."
    assert game.finances.cash == pytest.approx(1_000_000.0 - price)


def test_the_bulldozer_clears_a_cell_for_its_price() -> None:
    game = make_game()
    game.growth.build(Building((5, 5), BuildingKind.MINE))

    done = game.bulldoze((5, 5))
    nothing = game.bulldoze((5, 5))

    assert done.ok
    assert game.finances.cash == 1_000_000.0 - 20_000
    assert not nothing.ok
    assert nothing.message == "Nothing here can be cleared."


def test_a_moved_station_takes_its_trains_and_costs_a_new_one() -> None:
    game = make_game()
    for cell in ((6, 3), (6, 20)):
        game.build_station(cell)
    game.lay_track((6, 3), (6, 20))
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    cash = game.finances.cash

    done = game.move_station((6, 20), (8, 22))

    assert done.ok
    assert [stop.cell for stop in train.stops] == [(6, 3), (8, 22)]
    assert game.finances.cash == cash - STATION_PRICE


def test_a_month_grows_the_map_and_sums_up_what_changed_near_the_stations() -> None:
    game = make_game()
    game.build_station((6, 6))
    for cell in ((6, 7), (5, 6), (7, 6), (6, 5)):
        for _ in range(6):
            game.growth.build(Building(cell, BuildingKind.HOUSE))
    rng = np.random.default_rng(0)

    months = [game.month(rng, "Jan 1850") for _ in range(120)]

    assert any(month.changed for month in months)
    assert any(month.summary is not None for month in months)
    assert all(isinstance(x, Station) for x in game.constructions.at((6, 6)))
