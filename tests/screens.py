"""What tests that draw stand on: an imgui context like the one the game's window gives, a
texture standing in for one on the GPU, and pixel worlds of small maps to draw."""

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from imgui_bundle import imgui, implot

from builders import flat_map, layer
from railroads.sim.constructions import (
    Building,
    BuildingKind,
    Constructions,
)
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Relief, Terrain, TileMap, Vegetation
from railroads.ui.map_view import LOOKS
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.world import PixelWorld


class FakeTexture:
    def texture_id(self) -> int:
        return 1


@contextmanager
def gui_context(
    size: tuple[float, float] = (1280.0, 800.0), *, plots: bool = False, textures: bool = True
) -> Iterator[None]:
    """An imgui context with a screen this big, drawing textures if `textures`, and an implot
    one beside it if `plots`."""
    gui = imgui.create_context()
    plotting = implot.create_context() if plots else None
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(*size)
    if textures:
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    try:
        yield
    finally:
        if plotting is not None:
            implot.destroy_context(plotting)
        imgui.destroy_context(gui)


ROWS, COLS = 20, 30


def mountain_map() -> TileMap:
    relief = layer(ROWS, COLS, Relief.FLAT)
    relief[:, 18:] = Relief.MOUNTAINS
    relief[:, 10:13] = Relief.HILLS
    heights = np.where(relief == Relief.MOUNTAINS, 1500, 200).astype(np.int16)
    return flat_map(ROWS, COLS, relief=relief, height_m=heights)


def world_over_mountains() -> tuple[PixelWorld, tuple[int, int], tuple[int, int]]:
    tile_map = mountain_map()
    look = LOOKS[tile_map.grid.kind]
    world = PixelWorld(
        tile_map,
        Constructions(tile_map),
        SpriteSheet.load(),
        look.angle_deg,
        look.cell_units,
        fine=SpriteSheet.load_fine(),
    )
    grid = tile_map.grid
    chunks = []
    for col in (25, 3):
        px, py = world.projection.to_pixels(*grid.center(10, col))
        chunks.append((int(px) // 256, int(py) // 256))
    return world, chunks[0], chunks[1]


PLAIN_LOOKS = {HexGrid: (0.0, 1.0), SquareGrid: (-45.0, float(np.sqrt(2)))}


def make_tile_map(
    grid_type: type[Grid] = HexGrid,
    hills_from_col: int | None = None,
    rise_per_col: float = 0.0,
    mountains_from_col: int | None = None,
) -> TileMap:
    grid = grid_type(rows=16, cols=12, km_per_cell=2.0)
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    if hills_from_col is not None:
        relief[:, hills_from_col:] = Relief.HILLS
    if mountains_from_col is not None:
        relief[:, mountains_from_col:] = Relief.MOUNTAINS
    heights = np.broadcast_to(np.arange(grid.cols) * rise_per_col, grid.shape).astype(np.int16)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    vegetation = np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
    vegetation[:8, 6:] = Vegetation.FOREST
    return flat_map(
        grid=grid, terrain=terrain, vegetation=vegetation, height_m=heights, relief=relief
    )


def make_world(sheet: SpriteSheet, grid_type: type[Grid] = HexGrid, houses: int = 3) -> PixelWorld:
    tile_map = make_tile_map(grid_type)
    constructions = Constructions(tile_map)
    for _ in range(houses):
        constructions.build(Building((10, 4), BuildingKind.HOUSE))
    constructions.build(Building((4, 8), BuildingKind.LOGGING_CAMP))
    constructions.build(Building((12, 3), BuildingKind.FARM))
    return PixelWorld(tile_map, constructions, sheet, *PLAIN_LOOKS[grid_type])
