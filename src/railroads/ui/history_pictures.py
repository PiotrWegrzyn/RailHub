"""Pictures of the middle of a new game's map as its towns grow through history
(railroads.mapgen.seeding), every PICTURE_YEARS from HISTORY_START, for the waiting screen to
show: the pixel map as the game shows it at OPENING_SCALE, one art pixel to a screen point.

They are drawn as the game draws the map where the graphics card draws its ground
(railroads.ui.pixel.ground_shader): the CPU paints only the fields, woods, track and buildings over
the ground (PixelWorld.render with `over_ground`), which takes it well under half the time. Each
picture brings the cells' biomes for the graphics card to draw the ground under it, and the first
also what of the map's ground never changes. Where the graphics card can't, the pictures are
painted whole, ground and all.

Nothing here needs imgui, so a web worker growing the towns can draw them too."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import HISTORY_START
from railroads.sim.constructions import Constructions
from railroads.sim.grids import FloatArray, Grid
from railroads.ui.iso_camera import LOOKS
from railroads.ui.pixel.projection import CHUNK, ChunkPlacement, PixelProjection, Rgba
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.world import PixelWorld

PICTURE_YEARS = 100
OPENING_SCALE = 1


@dataclass(frozen=True)
class GroundLook:
    """What of a map's ground the graphics card needs that stays as it is through history: where
    map points land in the pixel map, the map's size in cells, each cell's deposit and hills, and
    the light on its slopes with the water and mountains round it (Ground.shading)."""

    projection: PixelProjection
    shape: tuple[int, int]
    deposits: npt.NDArray[np.integer] | None
    hills: npt.NDArray[np.bool_]
    shading: FloatArray


@dataclass(frozen=True)
class Picture:
    """The middle of the map in a year of its history: its RGBA pixels, one to a screen point of
    the window it fills, and where art pixel (0, 0) of the pixel map lies in it. With `biomes`, the
    pixels leave the ground out for the graphics card to draw under them from the cells' biomes
    and the map's GroundLook, which the first picture brings."""

    year: int
    pixels: Rgba
    origin: tuple[int, int]
    biomes: npt.NDArray[np.integer] | None = None
    ground: GroundLook | None = None


def opening_center(grid: Grid, size: tuple[float, float]) -> tuple[float, float]:
    """The map point a new game opens on in a window of `size` screen points: as near the map's
    middle as puts the window's top-left corner on a chunk's, so the fewest chunks fill it."""
    look = LOOKS[grid.kind]
    projection = PixelProjection.of(grid, look.angle_deg, look.cell_units)
    px, py = projection.to_pixels(*(length / 2 for length in grid.size()))
    left = round((float(px) - size[0] / 2) / CHUNK) * CHUNK
    top = round((float(py) - size[1] / 2) / CHUNK) * CHUNK
    x, y = projection.to_plane(left + size[0] / 2, top + size[1] / 2)
    return float(x), float(y)


def middle_picture(
    world: PixelWorld, size: tuple[int, int], over_ground: bool = False
) -> tuple[Rgba, tuple[int, int]]:
    """The middle of the world's map as a new game opens on it (opening_center), filling a window
    of `size` screen points, clear past its edges, and where art pixel (0, 0) lies in it; with
    `over_ground`, its ground left out where nothing rises."""
    width, height = size
    px, py = world.projection.to_pixels(*opening_center(world.tile_map.grid, size))
    placement = ChunkPlacement.of(
        world.projection,
        (float(px), float(py)),
        OPENING_SCALE,
        (width / 2, height / 2),
        ((0.0, 0.0), (float(width), float(height))),
    )
    picture = np.zeros((height, width, 4), dtype=np.uint8)
    for chunk in placement.chunks:
        (left, top), (right, bottom) = placement.screen_rect(chunk)
        shown_x = slice(max(left, 0), min(right, width))
        shown_y = slice(max(top, 0), min(bottom, height))
        if shown_x.start >= shown_x.stop or shown_y.start >= shown_y.stop:
            continue
        pixels = world.render(*chunk, over_ground=over_ground)
        picture[shown_y, shown_x] = pixels[
            shown_y.start - top : shown_y.stop - top, shown_x.start - left : shown_x.stop - left
        ]
    return picture, placement.origin


class HistoryPictures:
    """Looks at a map as its history grows (railroads.mapgen.seeding.seed_steps' `look`) and,
    every PICTURE_YEARS from HISTORY_START, shows `seen` a Picture of its middle `size` points
    big: with its ground left for the graphics card where `over_ground`, otherwise whole."""

    def __init__(
        self,
        size: tuple[int, int],
        seen: Callable[[Picture], None],
        over_ground: bool = False,
    ) -> None:
        self.size = size
        self.seen = seen
        self.over_ground = over_ground
        self._world: PixelWorld | None = None

    def __call__(self, constructions: Constructions, year: int) -> None:
        if (year - HISTORY_START) % PICTURE_YEARS:
            return
        world = self._world
        ground = None
        if world is None or world.constructions is not constructions:
            tile_map = constructions.tile_map
            look = LOOKS[tile_map.grid.kind]
            world = PixelWorld(tile_map, constructions, _sheet(), look.angle_deg, look.cell_units)
            self._world = world
            ground = GroundLook(
                world.projection,
                tile_map.grid.shape,
                tile_map.deposits,
                world.ground.hills,
                world.ground.shading,
            )
        else:
            world.buildings_changed(())
        pixels, origin = middle_picture(world, self.size, self.over_ground)
        if not self.over_ground:
            self.seen(Picture(year, pixels, origin))
            return
        self.seen(Picture(year, pixels, origin, world.ground.biomes.copy(), ground))


@cache
def _sheet() -> SpriteSheet:
    return SpriteSheet.load()
