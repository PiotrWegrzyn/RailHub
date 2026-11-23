"""What tests build their games on: a small flat map of grassland, any layer of which a test can
give itself, and a growth rule that grows and trades nothing, for tests that place every building
themselves; boards that open every buyer; world files held in memory; two stations joined by
track; a game played a few months; and a stand-in for the web page."""

import ctypes
from collections.abc import Callable
from functools import cache
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from railroads.newgame import MapChoice, Settings
from railroads.saves import SavedGame
from railroads.sim.clock import END_YEAR, GameClock
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.game import Game
from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.points import Points
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.sim.tiles import FIRST_YEAR, Tile, Tiles, Year
from railroads.sim.trains import Stop
from railroads.ui import web_setup
from railroads.version import VERSION

BOUNDS = Bounds(west=14.0, east=15.0, south=50.0, north=51.0)
QUIET = GrowthRule(trades=False, ladders=(), forests=None)
# A level above every one a demand tile opens a buyer at.
OPEN_LEVEL = 99


def layer(rows: int, cols: int, value: int) -> npt.NDArray[np.uint8]:
    """A map layer holding one value everywhere, for a test to change in places."""
    return np.full((rows, cols), value, dtype=np.uint8)


def flat_map(
    rows: int = 12,
    cols: int = 30,
    *,
    grid: Grid | None = None,
    terrain: npt.NDArray[np.uint8] | None = None,
    vegetation: npt.NDArray[np.uint8] | None = None,
    temperature: npt.NDArray[np.uint8] | None = None,
    height_m: npt.NDArray[np.int16] | None = None,
    relief: npt.NDArray[np.uint8] | None = None,
    deposits: npt.NDArray[np.uint8] | None = None,
) -> TileMap:
    """A map of `rows` by `cols` hexagons 2 km across (or the grid given), flat temperate
    grassland at sea level wherever a layer is not given."""
    grid = HexGrid(rows=rows, cols=cols, km_per_cell=2.0) if grid is None else grid
    rows, cols = grid.shape
    return TileMap(
        name="Test",
        grid=grid,
        bounds=BOUNDS,
        terrain=layer(rows, cols, Terrain.LAND) if terrain is None else terrain,
        vegetation=layer(rows, cols, Vegetation.GRASS) if vegetation is None else vegetation,
        temperature=layer(rows, cols, Temperature.NORMAL) if temperature is None else temperature,
        height_m=np.zeros((rows, cols), dtype=np.int16) if height_m is None else height_m,
        relief=layer(rows, cols, Relief.FLAT) if relief is None else relief,
        deposits=deposits,
    )


class MemoryFiles:
    """World files held in memory, noting which were asked for."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.asked: set[str] = set()

    def read(self, name: str) -> bytes | None:
        self.asked.add(name)
        return self.files.get(name)


@cache
def _world_years() -> dict[int, Year]:
    """The boards of every year when the lowest tiles always go up."""
    world = Tiles.until(END_YEAR + 99)
    return {year: world.of_year(year) for year in range(FIRST_YEAR, END_YEAR + 100)}


def open_boards() -> Tiles:
    """Boards on which every good is made and wanted at its base and earns its base price, and
    every buyer a demand tile opens is open, from the first year of the boards; the supply levels,
    which open engines, and the eras come as they do when the lowest tiles always go up."""
    return boards_with()


def boards_with(
    made: dict[Tile, float] | None = None, wanted: dict[Tile, float] | None = None
) -> Tiles:
    """open_boards, but with some tiles made or wanted so many times their base."""
    world = _world_years()
    tiles = Tiles()
    tiles.years = {
        year: Year(
            world[year].supply,
            MappingProxyType(dict.fromkeys(Tile, OPEN_LEVEL)),
            MappingProxyType(made or {}),
            MappingProxyType(wanted or {}),
            world[year].era,
        )
        for year in range(FIRST_YEAR, END_YEAR + 100)
    }
    return tiles


def joined_stations() -> tuple[Growth, Railway]:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 3))
    build_station(growth, (6, 20))
    route = plan_route(constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    return growth, railway


def played_game(months: int = 3) -> SavedGame:
    game = Game(Constructions(flat_map(rows=16, cols=30)))
    game.finances.cash = 1_000_000.0
    for _ in range(4):
        game.growth.build(Building((8, 15), BuildingKind.HOUSE))
    game.build_station((8, 6))
    game.build_station((8, 24))
    game.build_station((2, 2), (3, 9))
    game.lay_track((8, 6), (8, 24))
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop((8, 6)), Stop((8, 24))]
    clock = GameClock(first_year=1850)
    rng = np.random.default_rng(7)
    for _ in range(months):
        clock.advance(clock.seconds_per_month)
        game.month(rng, clock.date)
        game.time_passed(0.4)
    clock.advance(clock.seconds_per_month / 3)
    settings = Settings(seed=7, map=MapChoice.POLAND, start_year=1850)
    return SavedGame(game, clock, rng, "Test, seed 7", settings, camera=(4.5, 6.0, 2))


def served_at(growth: Growth, cell: tuple[int, int]) -> float:
    return float(growth._served[Points.POPULATION][cell[0] * growth.grid.cols + cell[1]])


SETTINGS = Settings(seed=7, map=MapChoice.POLAND, start_year=1850)


class FakePage:
    """A setup worker as the page runs one: it grows the game when told to work and its bytes
    are copied out of Python's memory by their address, as web/painter.js copies them."""

    def __init__(self, fails: bool = False, version: str = VERSION) -> None:
        self.fails = fails
        self.version = version
        self.settings: bytes | None = None
        self.inbox: list[dict[str, Any]] = []
        self.stopped = 0
        self.started = 0

    def start(self, settings: bytes) -> None:
        self.settings = settings
        self.started += 1

    def work(self) -> None:
        assert self.settings is not None
        if self.fails:
            self.inbox.append({"kind": "dead", "error": "out of memory"})
            return
        report = lambda text: self.inbox.append({"kind": "progress", "text": text})
        advance = lambda done: self.inbox.append({"kind": "advance", "done": done})
        grown = web_setup.grow(self.settings, report, advance, self.see)
        game = np.frombuffer(ctypes.string_at(grown.address, grown.length), dtype=np.uint8)
        self.inbox.append(
            {"kind": "grown", "game": game, "size": grown.size, "version": self.version}
        )

    def see(self, address: int, length: int) -> None:
        pixels = np.frombuffer(ctypes.string_at(address, length), dtype=np.uint8)
        self.inbox.append({"kind": "picture", "pixels": pixels})

    def take(self) -> list[dict[str, Any]]:
        taken, self.inbox = self.inbox, []
        return taken

    def stop(self) -> None:
        self.stopped += 1


def grows_as(saved: SavedGame) -> Callable[..., SavedGame]:
    def made(
        settings: Settings,
        report: Callable[[str], None],
        progress: Callable[[float], None] = lambda done: None,
        look: Callable[[Constructions, int], None] = lambda constructions, year: None,
    ) -> SavedGame:
        report(f"Growing towns for seed {settings.seed}")
        progress(0.5)
        look(saved.game.constructions, 950)
        return saved

    return made
