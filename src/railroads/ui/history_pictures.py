"""Pictures of the middle of a new game's map as its towns grow through history
(railroads.mapgen.seeding), every PICTURE_YEARS, for the waiting screen to show: the pixel map as
the game shows it at OPENING_SCALE, one art pixel to a screen point, over a plain background.

Nothing here needs imgui, so a web worker growing the towns can draw them too."""

from collections.abc import Callable
from functools import cache

import numpy as np

from railroads.sim.clock import HISTORY_START
from railroads.sim.constructions import Constructions
from railroads.ui.iso_camera import LOOKS
from railroads.ui.pixel.projection import RGB, ChunkPlacement, Rgba
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.world import PixelWorld

PICTURE_YEARS = 100
OPENING_SCALE = 1
BACKGROUND: RGB = (28, 26, 23)


def middle_picture(world: PixelWorld, size: tuple[int, int], background: RGB = BACKGROUND) -> Rgba:
    """The middle of the world's map filling a window of `size` screen points, opaque."""
    width, height = size
    middle = [length / 2 for length in world.tile_map.grid.size()]
    px, py = world.projection.to_pixels(*middle)
    placement = ChunkPlacement.of(
        world.projection,
        (float(px), float(py)),
        OPENING_SCALE,
        (width / 2, height / 2),
        ((0.0, 0.0), (float(width), float(height))),
    )
    canvas = np.empty((height, width, 3), dtype=np.float32)
    canvas[:] = background
    for chunk in placement.chunks:
        (left, top), (right, bottom) = placement.screen_rect(chunk)
        shown_x = slice(max(left, 0), min(right, width))
        shown_y = slice(max(top, 0), min(bottom, height))
        if shown_x.start >= shown_x.stop or shown_y.start >= shown_y.stop:
            continue
        pixels = world.render(*chunk)[
            shown_y.start - top : shown_y.stop - top, shown_x.start - left : shown_x.stop - left
        ]
        alpha = pixels[..., 3:].astype(np.float32) / 255
        part = canvas[shown_y, shown_x]
        canvas[shown_y, shown_x] = part + (pixels[..., :3] - part) * alpha
    picture = np.full((height, width, 4), 255, dtype=np.uint8)
    picture[..., :3] = np.rint(canvas)
    return picture


class HistoryPictures:
    """Looks at a map as its history grows (railroads.mapgen.seeding.seed_steps' `look`) and,
    every PICTURE_YEARS from HISTORY_START, tells `seen` the year and a picture of its middle
    `size` points big."""

    def __init__(
        self,
        size: tuple[int, int],
        seen: Callable[[int, Rgba], None],
    ) -> None:
        self.size = size
        self.seen = seen
        self._world: PixelWorld | None = None

    def __call__(self, constructions: Constructions, year: int) -> None:
        if (year - HISTORY_START) % PICTURE_YEARS:
            return
        world = self._world
        if world is None or world.constructions is not constructions:
            tile_map = constructions.tile_map
            look = LOOKS[tile_map.grid.kind]
            world = PixelWorld(tile_map, constructions, _sheet(), look.angle_deg, look.cell_units)
            self._world = world
        else:
            world.buildings_changed(())
        self.seen(year, middle_picture(world, self.size))


@cache
def _sheet() -> SpriteSheet:
    return SpriteSheet.load()
