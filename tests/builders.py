"""What tests build their games on: a small flat map of grassland, any layer of which a test can
give itself, and a growth rule that grows and trades nothing, for tests that place every building
themselves."""

import numpy as np
import numpy.typing as npt

from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid
from railroads.sim.ladders import GrowthRule
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

BOUNDS = Bounds(west=14.0, east=15.0, south=50.0, north=51.0)
QUIET = GrowthRule(trades=False, ladders=(), forests=None)


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
