from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.game import Game
from railroads.sim.tilemap import Relief, Temperature, Terrain, Tile, Vegetation
from railroads.sim.tiles import Tiles, activate

WEST = (6, 3)
FLAT = Tile(Terrain.LAND, Vegetation.GRASS, Temperature.NORMAL, 0, Relief.FLAT)


def electric_game(year: int) -> Game:
    game = Game(Constructions(flat_map()))
    game.finances.cash = 10_000_000.0
    game.railway.year = year
    activate(Tiles.until(year))
    return game


def test_electric_track_cannot_be_laid_before_it_opens() -> None:
    game = electric_game(1924)
    cash = game.finances.cash

    done = game.lay_track(WEST, (6, 9), electric=True)

    assert not done.ok
    assert "electric engine" in done.message
    assert game.finances.cash == cash
    assert game.constructions.track() == []


def test_electric_track_is_laid_once_it_opens() -> None:
    game = electric_game(1930)

    done = game.lay_track(WEST, (6, 9), electric=True)

    assert done.ok
    assert all(piece.electric for piece in game.constructions.track())
