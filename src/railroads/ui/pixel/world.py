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

from railroads.sim.constructions import Constructions, Track
from railroads.sim.grids import Cell
from railroads.sim.tilemap import Relief, TileMap
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH
from railroads.ui.pixel.ground import Ground
from railroads.ui.pixel.projection import CHUNK, FloatArray, IntArray, PixelProjection, Rgba
from railroads.ui.pixel.relief import MOST_LIFT, Elevation, raise_picture, raycast
from railroads.ui.pixel.scenery import Scenery
from railroads.ui.pixel.sheet import FINE, SpriteSheet
from railroads.ui.pixel.track_art import TrackArt

# How far in art pixels a sprite can reach from where it stands.
SPRITE_REACH = 48
# How many art pixels further down the map than track the ground showing where it is drawn may
# lie before the track counts as hidden behind it: the track's own width over gentle slopes.
HIDDEN_BEYOND = 2.0


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
        # The height of every point, which raises hills and mountains (railroads.ui.pixel.relief).
        self.elevation = Elevation(tile_map, self.projection)
        self.scenery = Scenery.of(
            tile_map, constructions, self.projection, sheet, elevation=self.elevation
        )
        # The ground and the track over it (railroads.ui.pixel.ground, railroads.ui.pixel.track_art).
        self.ground = Ground(tile_map, constructions, self.projection, angle_deg, self.elevation)
        self.track = TrackArt(tile_map, constructions, self.projection, self.elevation)
        # The sprites at each resolution chunks can be drawn at: the sheet, and the same sprites
        # drawn finer if there are any.
        self._sprites = {1: _Sprites.of(sheet)}
        if fine is not None:
            if fine.names != sheet.names:
                raise ValueError("The fine sprite sheet must hold the same sprites in order")
            self._sprites[FINE] = _Sprites.of(fine)

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
            self.tile_map,
            self.constructions,
            self.projection,
            self.sheet,
            self.station_levels,
            self.elevation,
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
        reach = PIXEL_CELL_WIDTH + SPRITE_REACH + MOST_LIFT
        chunks: set[tuple[int, int]] = set()
        for x, y in zip(px.tolist(), py.tolist(), strict=True):
            for cx in range(int((x - reach) // CHUNK), int((x + reach) // CHUNK) + 1):
                for cy in range(int((y - reach) // CHUNK), int((y + reach) // CHUNK) + 1):
                    chunks.add((cx, cy))
        return chunks

    def rises(self, chunk_x: int, chunk_y: int) -> bool:
        """Whether any hills or mountains rise in a chunk, or in the ground below it that it
        shows lifted."""
        return self.elevation.rises_in(chunk_x * CHUNK, chunk_y * CHUNK, CHUNK, CHUNK + MOST_LIFT)

    def render(
        self, chunk_x: int, chunk_y: int, resolution: int = 1, fine_relief: bool = False
    ) -> Rgba:
        """One chunk as RGBA pixels, `resolution` of them for every art pixel each way; off the
        map it is clear. Above 1 the ground is blown up and the track and everything standing
        on it are drawn that much finer. Where hills and mountains rise, the ground with its
        track is lifted, so the chunk shows ground from below it too; with `fine_relief` their
        heights, lift and colours are worked out at the full resolution too, which is smoother
        and slower."""
        if fine_relief and resolution > 1:
            return self._render_fine(chunk_x, chunk_y, resolution)
        left, top = chunk_x * CHUNK, chunk_y * CHUNK
        raised = self.elevation.area(left, top, CHUNK, CHUNK + MOST_LIFT)
        offset = None if raised is None else raycast(raised.lift, CHUNK, CHUNK)
        rows = CHUNK if offset is None else CHUNK + int(offset.max()) + 1
        rgb, alpha = self.ground.render(left, top, rows, raised)
        canvas = np.concatenate([rgb / 255, alpha[..., None]], axis=-1)
        if resolution > 1:
            canvas = canvas.repeat(resolution, axis=0).repeat(resolution, axis=1)
        if offset is None:
            self._lay_track(canvas, left, top, resolution)
        else:
            canvas = raise_picture(canvas, offset, resolution)
            seen = offset.repeat(resolution, axis=0).repeat(resolution, axis=1)
            self._lay_lifted_track(canvas, left, top, resolution, seen)
        self._stand_scenery(canvas, left, top, resolution)
        alpha = canvas[..., 3:]
        canvas[..., :3] /= np.maximum(alpha, 1e-6)
        return np.ascontiguousarray(np.rint(np.clip(canvas, 0, 1) * 255).astype(np.uint8))

    def _render_fine(self, chunk_x: int, chunk_y: int, k: int) -> Rgba:
        left, top = chunk_x * CHUNK, chunk_y * CHUNK
        raised = self.elevation.area(left, top, CHUNK, CHUNK + MOST_LIFT + 1, k)
        if raised is None:
            return self.render(chunk_x, chunk_y, k)
        offset = raycast(raised.lift, CHUNK * k, CHUNK * k)
        rows = -(-(CHUNK * k + int(offset.max()) + 1) // k)
        rgb, alpha = self.ground.render(left, top, rows)
        rgb = rgb.repeat(k, axis=0).repeat(k, axis=1)
        alpha = alpha.repeat(k, axis=0).repeat(k, axis=1)
        py, px = np.mgrid[0 : rows * k, 0 : CHUNK * k]
        x, y = self.projection.to_plane(left + (px + 0.5) / k, top + (py + 0.5) / k)
        grid = self.tile_map.grid
        r, c = grid.cell_at(x, y)
        r, c = np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1)
        rocky = self.tile_map.relief[r, c] == Relief.MOUNTAINS
        self.elevation.colour(rgb, raised.top(rows * k), x, y, rocky)
        rgb = np.clip(rgb, 0, 255) * alpha[..., None]
        canvas = np.concatenate([rgb / 255, alpha[..., None]], axis=-1)
        canvas = raise_picture(canvas, offset, 1)
        self._lay_lifted_track(canvas, left, top, k, offset / k)
        self._stand_scenery(canvas, left, top, k)
        alpha = canvas[..., 3:]
        canvas[..., :3] /= np.maximum(alpha, 1e-6)
        return np.ascontiguousarray(np.rint(np.clip(canvas, 0, 1) * 255).astype(np.uint8))

    def _lay_track(self, canvas: FloatArray, left: int, top: int, resolution: int) -> None:
        """Draws the track crossing the chunk over its ground (TrackArt.layer)."""
        pieces = self.constructions.track()
        if not pieces:
            return
        rows = canvas.shape[0] // resolution
        track = self.track.layer(pieces, left, top, CHUNK, rows, resolution)
        alpha = track[..., 3:] * (canvas[..., 3:] > 0)
        canvas[..., :3] = canvas[..., :3] * (1 - alpha) + track[..., :3] * alpha

    def _lay_lifted_track(
        self, canvas: FloatArray, left: int, top: int, resolution: int, seen: FloatArray
    ) -> None:
        """Draws the track crossing the chunk over its raised ground, lifted with it, where the
        ground shown, `seen` art pixels further down the map in each pixel, is not in front
        of it."""
        pieces = self.constructions.track()
        if not pieces:
            return
        track, lift = self.track.lifted(pieces, left, top, CHUNK, CHUNK, resolution)
        shown = (seen <= lift + HIDDEN_BEYOND) & (canvas[..., 3] > 0)
        alpha = track[..., 3:] * shown[..., None]
        canvas[..., :3] = canvas[..., :3] * (1 - alpha) + track[..., :3] * alpha

    def _stand_scenery(self, canvas: FloatArray, left: int, top: int, resolution: int) -> None:
        scenery, drawn, k = self.scenery, self._sprites[resolution], resolution
        x0 = (scenery.x - left) * k - drawn.anchors[scenery.sprite, 0]
        y0 = (scenery.y - scenery.lift - top) * k - drawn.anchors[scenery.sprite, 1]
        w, h = drawn.sizes[scenery.sprite, 0], drawn.sizes[scenery.sprite, 1]
        span = CHUNK * k
        overlapping = (x0 < span) & (x0 + w > 0) & (y0 < span) & (y0 + h > 0)
        for i in np.flatnonzero(overlapping).tolist():
            _blend(canvas, drawn.premultiplied[int(scenery.sprite[i])], int(x0[i]), int(y0[i]))

    def planned_track(
        self, pieces: list[Track], left: int, top: int, width: int, height: int, resolution: int
    ) -> FloatArray:
        """Track not laid yet, over `width` by `height` art pixels from (`left`, `top`), drawn
        as laid track would be: lifted with the ground and hidden behind raised ground in front
        of it."""
        raised = self.elevation.area(left, top, width, height + MOST_LIFT)
        if raised is None:
            return self.track.layer(pieces, left, top, width, height, resolution, stations=False)
        k = resolution
        seen = raycast(raised.lift, height, width).repeat(k, axis=0).repeat(k, axis=1)
        track, lift = self.track.lifted(pieces, left, top, width, height, k, stations=False)
        track[..., 3] *= seen <= lift + HIDDEN_BEYOND
        return track


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
