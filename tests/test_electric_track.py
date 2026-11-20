import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.finance import ELECTRIC_TRACK_FACTOR, track_price
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


@pytest.mark.parametrize("width", [TrackWidth.SINGLE, TrackWidth.DOUBLE])
def test_electric_track_costs_five_times_the_same_track_without_wires(width: TrackWidth) -> None:
    plain = Track((0, 0), (0, 6), width)
    wired = Track((0, 0), (0, 6), width, electric=True)

    price = track_price(wired, FLAT)

    assert price == pytest.approx(ELECTRIC_TRACK_FACTOR * track_price(plain, FLAT))
    assert ELECTRIC_TRACK_FACTOR == 5


def test_electric_track_opens_with_the_first_electric_engine() -> None:
    early = electric_game(1924)
    ready = electric_game(1930)

    opened = [game.railway.electric_open for game in (early, ready)]

    assert opened == [False, True]


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
