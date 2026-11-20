"""What tests build their games on: a small flat map of grassland, any layer of which a test can
give itself, and a growth rule that grows and trades nothing, for tests that place every building
themselves; boards that open every buyer; and world files held in memory."""

from functools import cache
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import END_YEAR
from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid
from railroads.sim.ladders import GrowthRule
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.sim.tiles import FIRST_YEAR, Tile, Tiles, Year

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
def _world_eras() -> dict[int, int]:
    """The era of every year when the lowest tiles always go up."""
    world = Tiles.until(END_YEAR + 99)
    return {year: world.of_year(year).era for year in range(FIRST_YEAR, END_YEAR + 100)}


def open_boards() -> Tiles:
    """Boards on which every good is made and wanted at its base and earns its base price, and
    every buyer a demand tile opens is open, from the first year of the boards; the eras come as
    they do when the lowest tiles always go up."""
    return boards_with()


def boards_with(
    made: dict[Tile, float] | None = None, wanted: dict[Tile, float] | None = None
) -> Tiles:
    """open_boards, but with some tiles made or wanted so many times their base."""
    eras = _world_eras()
    tiles = Tiles()
    tiles.years = {
        year: Year(
            MappingProxyType({}),
            MappingProxyType(dict.fromkeys(Tile, OPEN_LEVEL)),
            MappingProxyType(made or {}),
            MappingProxyType(wanted or {}),
            eras[year],
        )
        for year in range(FIRST_YEAR, END_YEAR + 100)
    }
    return tiles
