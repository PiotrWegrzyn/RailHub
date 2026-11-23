"""Moves a running game onto a bigger map, as when the player buys land next to it
(railroads.sim.lots): every cell moves down and right by the rows and columns added above and to
the left (railroads.sim.move), and everything the game keeps by cell or in arrays over the map
moves with it, each part by its own move_onto or moved.

What can be worked out again from what is built (the counts, neighbourhoods, towns, chances and
caches) is worked out afresh on the bigger map; what the game remembers (satisfaction,
stockpiles, starving buildings, the player's shares, stations, trains and their goods) is moved."""

from collections.abc import Iterable

from railroads.sim.constructions import Building
from railroads.sim.game import Game
from railroads.sim.move import Move
from railroads.sim.tilemap import TileMap


def regrid(
    game: Game,
    tile_map: TileMap,
    rows: int,
    cols: int,
    buildings: Iterable[Building] = (),
) -> None:
    """Moves the game onto `tile_map`, its old map `rows` down and `cols` right on it, and puts
    up `buildings` there, already on the new map's cells, as land bought arrives grown: those
    that can stand there as the map now is."""
    game.move_onto(tile_map, Move(game.tile_map.grid, tile_map.grid, rows, cols), buildings)
