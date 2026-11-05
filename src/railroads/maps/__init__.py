"""Maps that ship with the game, baked by the scripts in tools/."""

from pathlib import Path

from railroads.sim.grids import GridKind
from railroads.sim.tilemap import TileMap

MAPS_DIR = Path(__file__).parent


def map_file(name: str, grid: GridKind) -> Path:
    return MAPS_DIR / f"{name}-{grid}.npz"


def load(name: str, grid: GridKind) -> TileMap:
    return TileMap.load(map_file(name, grid))
