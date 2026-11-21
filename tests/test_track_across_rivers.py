from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from builders import flat_map
from railroads import maps
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.game import Game
from railroads.sim.goods import Good
from railroads.sim.grids import Cell, HexGrid
from railroads.sim.lots import Lots
from railroads.sim.railway import NO_TRACK_ON
from railroads.sim.routes import plan_route_steps
from railroads.sim.steps import finish
from railroads.sim.tilemap import Terrain, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.sim.trains import Stop
from railroads.ui.station_window import UNJOINED, station_warning
from railroads.ui.trains_window import train_warning

NORTH_BANK, SOUTH_BANK = (11, 17), (15, 16)
DZIALOSZYN_RIVER = (186, 108), (188, 118)


def wide_river_by_a_lot_border() -> TileMap:
    grid = HexGrid(rows=16, cols=24, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    for row in range(16):
        terrain[row, row + 2 : row + 5] = Terrain.RIVER
    locked = np.zeros(grid.shape, dtype=bool)
    locked[:, 20:] = True
    return replace(flat_map(grid=grid, terrain=terrain), locked=locked)


def poland_as_a_new_game_crops_it() -> TileMap:
    place = maps.load("poland", "hex")
    lots = Lots.around_middle(place)
    return replace(place.window(*lots.window()), locked=lots.locked())


def two_stations(tile_map: TileMap, start: Cell, end: Cell) -> Game:
    game = Game(Constructions(tile_map))
    game.finances.cash = 10_000_000
    assert game.build_station(start).ok
    assert game.build_station(end).ok
    return game


def lay_as_the_track_tool_does(game: Game, start: Cell, end: Cell) -> None:
    preview = finish(plan_route_steps(game.constructions, start, end))
    assert preview is not None
    assert game.lay_track(start, end, planned=preview).ok


def farms_round(game: Game, station: Cell) -> None:
    for cell in game.tile_map.grid.neighbours(*station):
        if game.constructions.why_not(Building(cell, BuildingKind.FARM)) is None:
            game.growth.build(Building(cell, BuildingKind.FARM))


@pytest.mark.parametrize(
    ("tile_map", "stations"),
    [
        (wide_river_by_a_lot_border, (NORTH_BANK, SOUTH_BANK)),
        (poland_as_a_new_game_crops_it, DZIALOSZYN_RIVER),
    ],
    ids=["wide river by a lot border", "Poland by Dzialoszyn"],
)
def test_track_laid_across_a_river_joins_the_stations_on_both_banks(
    tile_map: Callable[[], TileMap], stations: tuple[Cell, Cell]
) -> None:
    start, end = stations
    game = two_stations(tile_map(), start, end)

    lay_as_the_track_tool_does(game, start, end)

    assert {start, end} <= game.railway.connected()
    assert TrackBed.BRIDGE in {piece.bed for piece in game.constructions.track()}


@pytest.mark.parametrize(
    ("tile_map", "stations"),
    [
        (wide_river_by_a_lot_border, (NORTH_BANK, SOUTH_BANK)),
        (poland_as_a_new_game_crops_it, DZIALOSZYN_RIVER),
    ],
    ids=["wide river by a lot border", "Poland by Dzialoszyn"],
)
def test_a_station_joined_across_a_river_gathers_from_the_farms_round_it(
    tile_map: Callable[[], TileMap], stations: tuple[Cell, Cell]
) -> None:
    start, end = stations
    game = two_stations(tile_map(), start, end)
    farms_round(game, end)
    lay_as_the_track_tool_does(game, start, end)

    game.growth.trade_goods()

    assert game.railway.station_goods(end).stockpile.get(Good.WHEAT, 0.0) > 0


def test_a_station_no_track_joins_to_another_says_it_gathers_and_hands_out_nothing() -> None:
    game = two_stations(wide_river_by_a_lot_border(), NORTH_BANK, SOUTH_BANK)

    warning = station_warning(game.railway, SOUTH_BANK)

    assert warning == UNJOINED


def test_a_station_joined_across_a_river_warns_of_nothing() -> None:
    game = two_stations(wide_river_by_a_lot_border(), NORTH_BANK, SOUTH_BANK)
    lay_as_the_track_tool_does(game, NORTH_BANK, SOUTH_BANK)

    warning = station_warning(game.railway, SOUTH_BANK)

    assert warning == ""


def test_a_train_whose_next_stop_no_track_reaches_says_it_waits_for_track() -> None:
    game = two_stations(wide_river_by_a_lot_border(), NORTH_BANK, SOUTH_BANK)
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop(NORTH_BANK), Stop(SOUTH_BANK)]
    game.railway.run(0.1)

    warning = train_warning(game.railway, train)

    assert warning.startswith(f"{NO_TRACK_ON}, {game.railway.name_of(SOUTH_BANK)}")


def test_a_train_with_track_to_its_next_stop_warns_of_nothing() -> None:
    game = two_stations(wide_river_by_a_lot_border(), NORTH_BANK, SOUTH_BANK)
    lay_as_the_track_tool_does(game, NORTH_BANK, SOUTH_BANK)
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop(NORTH_BANK), Stop(SOUTH_BANK)]
    game.railway.run(0.1)

    warning = train_warning(game.railway, train)

    assert warning == ""


def test_a_station_in_a_river_is_refused_saying_why() -> None:
    game = Game(Constructions(wide_river_by_a_lot_border()))

    done = game.build_station((5, 8))

    assert not done.ok
    assert done.message == "A station can't go there: stations can't stand in a river."
