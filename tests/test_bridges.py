import numpy as np
import pytest

from builders import flat_map, layer, open_boards
from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.finance import track_price
from railroads.sim.game import Game
from railroads.sim.level_perks import Perk, perks
from railroads.sim.routes import Bridging, plan_route
from railroads.sim.signals import Blocks, track_point
from railroads.sim.tilemap import Terrain, TileMap
from railroads.sim.tiles import Board, Tile, activate, bridge_open
from railroads.sim.track_network import TrackNetwork
from railroads.sim.track_rules import BRIDGES, RAMP_RULES, Bridge, TrackBed, carries

WEST, EAST = (7, 2), (7, 17)
NORTH, SOUTH = (1, 9), (13, 9)
CROSSING = (7, 9)


def river_map() -> TileMap:
    terrain = layer(14, 20, Terrain.LAND)
    terrain[:, 9] = Terrain.RIVER
    return flat_map(14, 20, terrain=terrain)


def game_on(tile_map: TileMap) -> Game:
    game = Game(Constructions(tile_map))
    game.finances.cash = 100_000_000.0
    return game


def flyover_game(bridge: Bridge = Bridge.STEEL) -> Game:
    game = game_on(flat_map(14, 20))
    assert game.lay_track(WEST, EAST, TrackWidth.DOUBLE).ok
    route = plan_route(
        game.constructions, NORTH, SOUTH, TrackWidth.DOUBLE, bridging=Bridging(bridge)
    )
    assert route is not None
    activate(open_boards())
    game.railway.year = 1950
    assert game.lay_track(NORTH, SOUTH, TrackWidth.DOUBLE, planned=route, bridge=bridge).ok
    return game


@pytest.mark.parametrize(
    ("bridge", "tracks"),
    [(Bridge.WOOD, 1), (Bridge.STONE, 2), (Bridge.STEEL, 4), (Bridge.CONCRETE, 4)],
)
def test_each_bridge_carries_as_many_tracks_as_it_is_built_for(bridge: Bridge, tracks: int) -> None:
    carried = carries(bridge, tracks), carries(bridge, tracks + 1)

    assert carried == (True, False)


def test_only_steel_and_concrete_bridges_take_wires_and_only_concrete_a_guideway() -> None:
    wired = [bridge for bridge in Bridge if carries(bridge, 1, electric=True)]
    guided = [bridge for bridge in Bridge if carries(bridge, 1, maglev=True)]

    assert (wired, guided) == ([Bridge.STEEL, Bridge.CONCRETE], [Bridge.CONCRETE])


def test_each_newer_bridge_costs_more_and_lets_trains_run_faster() -> None:
    kinds = [BRIDGES[bridge] for bridge in Bridge]

    prices = [kind.price_factor for kind in kinds]
    speeds = [kind.speed_factor for kind in kinds]

    assert prices == sorted(set(prices))
    assert speeds == sorted(set(speeds))


def test_track_left_to_the_planner_crosses_a_river_on_the_cheapest_bridge_carrying_it() -> None:
    constructions = Constructions(river_map())
    bridging = Bridging(None, (Bridge.WOOD, Bridge.STONE))

    route = plan_route(constructions, WEST, EAST, TrackWidth.DOUBLE, bridging=bridging)

    assert route is not None
    assert {piece.bridge for piece in route.pieces if piece.bed == TrackBed.BRIDGE} == {
        Bridge.STONE
    }


def test_no_track_crosses_a_river_on_a_bridge_too_narrow_for_it() -> None:
    constructions = Constructions(river_map())

    route = plan_route(
        constructions, WEST, EAST, TrackWidth.DOUBLE, bridging=Bridging(None, (Bridge.WOOD,))
    )

    assert route is None


def test_a_bridge_still_to_come_cannot_be_built() -> None:
    game = game_on(river_map())

    done = game.lay_track(WEST, EAST, bridge=Bridge.STEEL)

    assert not done.ok
    assert "steel and metals supply 16" in done.message


def test_stone_bridges_open_with_ore_and_stone_demand_and_steel_ones_with_steel_supply() -> None:
    stone_tiles = [
        (board, tile)
        for board in Board
        for tile in Tile
        if Perk.BRIDGE in {perk.kind for perk in perks(board, tile, 8)}
    ]

    assert stone_tiles == [(Board.DEMAND, Tile.ORE)]
    assert bridge_open(Bridge.STONE, 1850)
    assert not bridge_open(Bridge.STEEL, 1850)


def test_steel_track_laid_across_track_goes_over_it_as_a_flyover() -> None:
    game = flyover_game()

    over = [
        piece for piece in game.constructions.track_at(CROSSING) if piece.bed == TrackBed.BRIDGE
    ]

    assert len(over) == 1
    assert game.constructions.is_flyover(over[0])
    assert over[0].bridge == Bridge.STEEL


def test_track_left_to_the_planner_crosses_track_on_the_level() -> None:
    game = game_on(flat_map(14, 20))
    game.lay_track(WEST, EAST, TrackWidth.DOUBLE)

    game.lay_track(NORTH, SOUTH, TrackWidth.DOUBLE)

    beds = {piece.bed for piece in game.constructions.track_at(CROSSING)}
    assert beds == {TrackBed.GROUND}


def test_a_wooden_bridge_cannot_go_over_track() -> None:
    game = game_on(flat_map(14, 20))
    game.lay_track(WEST, EAST)
    under = game.constructions.track_at(CROSSING)[0]
    ends = sorted(game.tile_map.grid.track_ends())
    across = next(
        (a, b)
        for a in ends
        for b in ends
        if a < b
        and {a, b}.isdisjoint(under.ends)
        and game.constructions.why_not(Track(CROSSING, (a, b))) is None
    )

    problem = game.constructions.why_not(
        Track(CROSSING, across, bed=TrackBed.BRIDGE, bridge=Bridge.WOOD)
    )

    assert problem == "A wooden bridge goes only over rivers and lakes"


def test_trains_over_a_flyover_never_hold_up_trains_under_it() -> None:
    game = flyover_game()
    grid = game.tile_map.grid
    over, under = sorted(game.constructions.track_at(CROSSING), key=game.constructions.is_flyover)[
        ::-1
    ]
    blocks = Blocks(game.constructions)

    def way(piece: Track) -> tuple:
        return tuple(track_point(grid, CROSSING, end) for end in piece.ends)

    assert blocks.apart(CROSSING, way(over), way(under))


def test_a_flyover_costs_more_than_the_same_bridge_over_water_for_its_ramps() -> None:
    land, water = flat_map(14, 20), river_map()
    piece = Track(CROSSING, (0, 6), TrackWidth.DOUBLE, TrackBed.BRIDGE, bridge=Bridge.STEEL)

    over_track = track_price(piece, land.tile(CROSSING))
    over_water = track_price(piece, water.tile(CROSSING))

    assert over_track > over_water


def test_trains_slow_down_on_the_ramps_up_to_a_flyover() -> None:
    game = flyover_game()
    network = TrackNetwork(game.constructions)
    ramps = {
        game.tile_map.grid.track_exit(CROSSING, end)[0]
        for piece in game.constructions.track_at(CROSSING)
        if game.constructions.is_flyover(piece)
        for end in piece.ends
    }

    speeds = {network._speed(cell) for cell in ramps}

    assert speeds == {RAMP_RULES.speed_factor}
    assert np.isclose(network._speed(CROSSING), 1.0)
