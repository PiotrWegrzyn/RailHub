"""The tutorial's game: the place's buildings put up as they are drawn (railroads.tutorial.place),
with nothing grown from a seed, so it starts the same every time."""

import numpy as np

from railroads.newgame import MapChoice, Settings
from railroads.saves import SavedGame
from railroads.sim.clock import GameClock
from railroads.sim.constructions import Constructions
from railroads.sim.finance import START_CASH
from railroads.sim.game import Game
from railroads.sim.lots import Lots
from railroads.sim.stats import History
from railroads.sim.tilemap import Terrain
from railroads.tutorial.place import (
    ASHFORD,
    SQUARE_LEFT,
    SQUARE_SIZE,
    SQUARE_TOP,
    START_YEAR,
    buildings,
    place_map,
    square_map,
    station_names,
)
from railroads.tutorial.steps import Tutorial

SETTINGS = Settings(seed=0, map=MapChoice.TUTORIAL, start_year=START_YEAR)
# The random numbers the tutorial's months grow by, the same every time.
SEED = 1850
# The view opens on Ashford, close enough to see its houses.
OPENING_ZOOM = 4


def make_tutorial() -> SavedGame:
    """A new tutorial game, ready to play."""
    place = place_map()
    tile_map = square_map(place)
    constructions = Constructions(tile_map)
    for building in buildings():
        constructions.build(building)
    game = Game(
        constructions, start_year=START_YEAR, history=History(), names=station_names(tile_map)
    )
    game.finances.cash = START_CASH
    game.lots = Lots.of_square(SQUARE_TOP, SQUARE_LEFT, SQUARE_SIZE, SQUARE_SIZE)
    game.lots.offer(place.terrain == Terrain.LAND)
    x, y = tile_map.grid.center(*ASHFORD)
    return SavedGame(
        game,
        GameClock(first_year=START_YEAR),
        np.random.default_rng(SEED),
        "Tutorial",
        SETTINGS,
        camera=(float(x), float(y), OPENING_ZOOM),
        tutorial=Tutorial(),
    )
