"""The map as pixel art, drawn a square chunk at a time.

The view never turns, so every point of the map has a fixed place in one big picture, at one art
pixel per screen pixel when PIXEL_CELL_WIDTH pixels make a cell. That picture is far too big to
hold at once, so it is drawn in CHUNK x CHUNK pieces as the view reaches them: first the ground,
every pixel coloured from its cell with a little texture, then everything standing on it (trees,
rocks, buildings) from back to front. Textures come from a hash of the pixel's place, so chunks
meet without seams and look the same each time they are drawn.

The ground, the track, what stands where and where it all lands in the picture each have their
own module (railroads.ui.pixel.ground, track_art, scenery and projection); this one puts a chunk
together from them.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import Constructions
from railroads.sim.grids import Cell
from railroads.sim.tilemap import TileMap
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH
from railroads.ui.pixel.art import TINT_GREYS
from railroads.ui.pixel.ground import Ground
from railroads.ui.pixel.projection import CHUNK, FloatArray, IntArray, PixelProjection, Rgba
from railroads.ui.pixel.scenery import Scenery
from railroads.ui.pixel.sheet import FINE, SpriteSheet
from railroads.ui.pixel.track_art import TrackArt

# How far in art pixels a sprite can reach from where it stands.
SPRITE_REACH = 48
# How much darker a tinted sprite's outline is than the darkest shade of its ground, and how much
# lighter its highlight than the lightest.
TINT_OUTLINE = 0.72
TINT_HIGHLIGHT = 1.12


class PixelWorld:
    """Draws chunks of one map as pixel art."""

    def __init__(
        self,
        tile_map: TileMap,
        constructions: Constructions,
        sheet: SpriteSheet,
        angle_deg: float,
        cell_units: float,
        fine: SpriteSheet | None = None,
    ) -> None:
        self.tile_map = tile_map
        self.constructions = constructions
        self.sheet = sheet
        self.projection = PixelProjection.of(tile_map.grid, angle_deg, cell_units)
        # Each station's level (railroads.sim.railway), which picks how grand it looks.
        self.station_levels: Mapping[Cell, int] = {}
        self.scenery = Scenery.of(tile_map, constructions, self.projection, sheet)
        # The ground and the track over it (railroads.ui.pixel.ground, railroads.ui.pixel.track_art).
        self.ground = Ground(tile_map, constructions, self.projection, angle_deg)
        self.track = TrackArt(tile_map, constructions, self.projection)
        # The sprites at each resolution chunks can be drawn at: the sheet, and the same sprites
        # drawn finer if there are any.
        self._sprites = {1: _Sprites.of(sheet)}
        if fine is not None:
            if fine.names != sheet.names:
                raise ValueError("The fine sprite sheet must hold the same sprites in order")
            self._sprites[FINE] = _Sprites.of(fine)
        self._tints: dict[tuple[int, int, int], npt.NDArray[np.float32]] = {}

    @property
    def satisfaction(self) -> npt.NDArray[np.int8]:
        """Each cell's Satisfaction, or -1 where nothing trades, which the ground glows with."""
        return self.ground.satisfaction

    @satisfaction.setter
    def satisfaction(self, levels: npt.NDArray[np.int8]) -> None:
        self.ground.satisfaction = levels

    @property
    def resolutions(self) -> tuple[int, ...]:
        """The resolutions chunks can be drawn at, in pixels per art pixel."""
        return tuple(self._sprites)

    def buildings_changed(self, cells: Iterable[Cell]) -> set[tuple[int, int]]:
        """Puts the ground and scenery up again after buildings or forests changed in `cells`;
        returns the chunks that look different now."""
        self.ground.refresh()
        self.scenery = Scenery.of(
            self.tile_map, self.constructions, self.projection, self.sheet, self.station_levels
        )
        return self._chunks_around(cells)

    def satisfaction_changed(self, levels: npt.NDArray[np.int8]) -> set[tuple[int, int]]:
        """Takes each cell's new satisfaction; returns the chunks whose glow changed."""
        changed = np.argwhere(levels != self.satisfaction)
        self.satisfaction = levels.copy()
        return self._chunks_around([(int(r), int(c)) for r, c in changed])

    def _chunks_around(self, cells: Iterable[Cell]) -> set[tuple[int, int]]:
        cells = list(cells)
        if not cells:
            return set()
        rows, cols = np.array(cells).T
        px, py = self.projection.to_pixels(*self.tile_map.grid.center(rows, cols))
        reach = PIXEL_CELL_WIDTH + SPRITE_REACH
        chunks: set[tuple[int, int]] = set()
        for x, y in zip(px.tolist(), py.tolist(), strict=True):
            for cx in range(int((x - reach) // CHUNK), int((x + reach) // CHUNK) + 1):
                for cy in range(int((y - reach) // CHUNK), int((y + reach) // CHUNK) + 1):
                    chunks.add((cx, cy))
        return chunks

    def render(self, chunk_x: int, chunk_y: int, resolution: int = 1) -> Rgba:
        """One chunk as RGBA pixels, `resolution` of them for every art pixel each way; off the
        map it is clear. Above 1 the ground is blown up and the track and everything standing
        on it are drawn that much finer."""
        left, top = chunk_x * CHUNK, chunk_y * CHUNK
        rgb, alpha = self.ground.render(left, top)
        canvas = np.concatenate([rgb / 255, alpha[..., None]], axis=-1)
        if resolution > 1:
            canvas = canvas.repeat(resolution, axis=0).repeat(resolution, axis=1)
        self._lay_track(canvas, left, top, resolution)
        self._stand_scenery(canvas, left, top, resolution)
        alpha = canvas[..., 3:]
        canvas[..., :3] /= np.maximum(alpha, 1e-6)
        return np.ascontiguousarray(np.rint(np.clip(canvas, 0, 1) * 255).astype(np.uint8))

    def _lay_track(self, canvas: FloatArray, left: int, top: int, resolution: int) -> None:
        """Draws the track crossing the chunk over its ground (TrackArt.layer)."""
        pieces = self.constructions.track()
        if not pieces:
            return
        track = self.track.layer(pieces, left, top, CHUNK, CHUNK, resolution)
        alpha = track[..., 3:] * (canvas[..., 3:] > 0)
        canvas[..., :3] = canvas[..., :3] * (1 - alpha) + track[..., :3] * alpha

    def _stand_scenery(self, canvas: FloatArray, left: int, top: int, resolution: int) -> None:
        scenery, drawn, k = self.scenery, self._sprites[resolution], resolution
        x0 = (scenery.x - left) * k - drawn.anchors[scenery.sprite, 0]
        y0 = (scenery.y - top) * k - drawn.anchors[scenery.sprite, 1]
        w, h = drawn.sizes[scenery.sprite, 0], drawn.sizes[scenery.sprite, 1]
        span = CHUNK * k
        overlapping = (x0 < span) & (x0 + w > 0) & (y0 < span) & (y0 + h > 0)
        for i in np.flatnonzero(overlapping).tolist():
            sprite, tint = int(scenery.sprite[i]), int(scenery.tint[i])
            pixels = (
                drawn.premultiplied[sprite] if tint < 0 else self._tinted(sprite, tint, resolution)
            )
            _blend(canvas, pixels, int(x0[i]), int(y0[i]))

    def _tinted(self, sprite: int, biome: int, resolution: int = 1) -> npt.NDArray[np.float32]:
        """A sprite with its pure greys coloured with a biome's shades, premultiplied."""
        key = (sprite, biome, resolution)
        if key not in self._tints:
            ramp = self.ground.ramps[biome]
            colors = np.vstack([ramp[0] * TINT_OUTLINE, ramp, ramp[3] * TINT_HIGHLIGHT]) / 255
            pixels = self._sprites[resolution].pixels[sprite].copy()
            grey = (pixels[..., 3] == 1) & (pixels[..., 0] == pixels[..., 1])
            grey &= pixels[..., 1] == pixels[..., 2]
            levels = np.rint(pixels[..., 0] * 255).astype(np.int64)
            level = np.abs(levels[..., None] - np.array(TINT_GREYS)).argmin(axis=-1)
            pixels[grey, :3] = np.clip(colors[level[grey]], 0, 1)
            self._tints[key] = np.concatenate(
                [pixels[..., :3] * pixels[..., 3:], pixels[..., 3:]], axis=-1
            )
        return self._tints[key]


@dataclass(frozen=True)
class _Sprites:
    """A sheet's sprites as the renderer uses them: each one's pixels, premultiplied too, its
    width and height, and its anchor."""

    pixels: list[npt.NDArray[np.float32]]
    premultiplied: list[npt.NDArray[np.float32]]
    sizes: IntArray
    anchors: IntArray

    @classmethod
    def of(cls, sheet: SpriteSheet) -> Self:
        premultiplied = [
            np.concatenate([p[..., :3] * p[..., 3:], p[..., 3:]], axis=-1) for p in sheet.pixels
        ]
        return cls(sheet.pixels, premultiplied, sheet.sizes, sheet.anchors)


def _blend(canvas: FloatArray, sprite: npt.NDArray[np.float32], x: int, y: int) -> None:
    """Lays a premultiplied RGBA sprite over the canvas with its top-left corner at (x, y)."""
    h, w = sprite.shape[:2]
    cx0, cy0 = max(x, 0), max(y, 0)
    cx1, cy1 = min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
    if cx0 >= cx1 or cy0 >= cy1:
        return
    part = sprite[cy0 - y : cy1 - y, cx0 - x : cx1 - x]
    under = canvas[cy0:cy1, cx0:cx1]
    under *= 1 - part[..., 3:]
    under += part
