import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.finance import ELECTRIC_TRACK_FACTOR, track_price
from railroads.sim.game import Game
from railroads.sim.tech import Research
from railroads.sim.tilemap import Relief, Temperature, Terrain, Tile, Vegetation

WEST = (6, 3)
FLAT = Tile(Terrain.LAND, Vegetation.GRASS, Temperature.NORMAL, 0, Relief.FLAT)


def electric_game(year: int, *taken: str) -> Game:
    game = Game(Constructions(flat_map()))
    game.finances.cash = 10_000_000.0
    game.railway.year = year
    game.railway.research = Research(taken=set(taken))
    return game


@pytest.mark.parametrize("width", [TrackWidth.SINGLE, TrackWidth.DOUBLE])
def test_electric_track_costs_five_times_the_same_track_without_wires(width: TrackWidth) -> None:
    plain = Track((0, 0), (0, 6), width)
    wired = Track((0, 0), (0, 6), width, electric=True)

    price = track_price(wired, FLAT)

    assert price == pytest.approx(ELECTRIC_TRACK_FACTOR * track_price(plain, FLAT))
    assert ELECTRIC_TRACK_FACTOR == 5


def test_electric_track_opens_with_the_first_electric_engine() -> None:
    early = electric_game(1924, "telegraph", "traction")
    unlearned = electric_game(1930, "telegraph")
    ready = electric_game(1930, "telegraph", "traction")

    opened = [game.railway.electric_open for game in (early, unlearned, ready)]

    assert opened == [False, False, True]


def test_electric_track_cannot_be_laid_before_it_opens() -> None:
    game = electric_game(1924, "telegraph", "traction")
    cash = game.finances.cash

    done = game.lay_track(WEST, (6, 9), electric=True)

    assert not done.ok
    assert "Electric traction" in done.message
    assert game.finances.cash == cash
    assert game.constructions.track() == []


def test_electric_track_is_laid_once_it_opens() -> None:
    game = electric_game(1930, "telegraph", "traction")

    done = game.lay_track(WEST, (6, 9), electric=True)

    assert done.ok
    assert all(piece.electric for piece in game.constructions.track())
