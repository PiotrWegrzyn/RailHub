from collections.abc import Mapping, Sequence

from railroads.mapgen.area import MapArea
from railroads.mapgen.strategies import Layer, Strategy
from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid
from railroads.sim.tilemap import TileMap


def generate_map(
    name: str,
    bounds: Bounds,
    grid: Grid,
    strategies: Mapping[str, Strategy],
    sources: Sequence[str] = (),
) -> TileMap:
    """Fill in every layer of a map, running the strategies in order so each one can use the
    layers made before it."""
    area = MapArea.of(bounds, grid)
    layers: dict[str, Layer] = {}
    for layer, strategy in strategies.items():
        layers[layer] = strategy(area, layers)
    return TileMap(name=name, grid=grid, bounds=bounds, sources=tuple(sources), **layers)
