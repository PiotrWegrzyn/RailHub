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
from railroads.ui.pixel.ground import (
    APRON,
    CANOPY_REACH,
    FIELD_KINDS,
    IN_THE_WOODS,
    MILL_CLEARING,
    YARD_REACH,
    Ground,
    art_pixels,
    glow_strength,
)
from railroads.ui.pixel.lattice import Lattice
from railroads.ui.pixel.projection import CHUNK, FloatArray, IntArray, PixelProjection, Rgba
from railroads.ui.pixel.relief import MOST_LIFT, Elevation, lighting, raise_picture, raycast
from railroads.ui.pixel.scenery import BUILDING_SPRITES, Scenery
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
        station_levels: Mapping[Cell, int] | None = None,
    ) -> None:
        self.tile_map = tile_map
        self.constructions = constructions
        self.sheet = sheet
        self.projection = PixelProjection.of(tile_map.grid, angle_deg, cell_units)
        # Each station's level (railroads.sim.railway), which picks how grand it looks.
        self.station_levels: Mapping[Cell, int] = {} if station_levels is None else station_levels
        # The height of every point, which raises hills and mountains (railroads.ui.pixel.relief).
        self.elevation = Elevation(tile_map, self.projection)
        self.scenery = Scenery.of(
            tile_map,
            constructions,
            self.projection,
            sheet,
            self.station_levels,
            elevation=self.elevation,
        )
        # The ground and the track over it (railroads.ui.pixel.ground, railroads.ui.pixel.track_art).
        self.ground = Ground(tile_map, constructions, self.projection, angle_deg, self.elevation)
        self._shade_trees()
        self.track = TrackArt(tile_map, constructions, self.projection, self.elevation)
        # Every piece of track, kept until track or stations change.
        self._pieces: list[Track] = []
        self._pieces_version = -1
        # The sprites at each resolution chunks can be drawn at: the sheet, and the same sprites
        # drawn finer if there are any.
        self._sprites = {1: _Sprites.of(sheet)}
        if fine is not None:
            if fine.names != sheet.names:
                raise ValueError("The fine sprite sheet must hold the same sprites in order")
            self._sprites[FINE] = _Sprites.of(fine)
        # How many times buildings and satisfaction have changed, so copies of the world painting
        # elsewhere (railroads.ui.pixel.copies) know when to catch up.
        self.buildings_version = 0
        self.satisfaction_version = 0

    @property
    def satisfaction(self) -> npt.NDArray[np.int8]:
        """Each cell's Satisfaction, or -1 where nothing trades, which the ground glows with."""
        return self.ground.satisfaction

    @satisfaction.setter
    def satisfaction(self, levels: npt.NDArray[np.int8]) -> None:
        self.ground.satisfaction = levels
        self.satisfaction_version += 1

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
        self._shade_trees()
        self.buildings_version += 1
        return self._chunks_around(cells)

    def _shade_trees(self) -> None:
        """Tells the ground where trees stand, so the grass round them lies in their shade."""
        trees = np.isin(
            self.scenery.sprite, [*self.sheet.variants("pine"), *self.sheet.variants("leafy")]
        )
        self.ground.trees = (
            self.scenery.x[trees].astype(np.float32),
            self.scenery.y[trees].astype(np.float32),
        )
        farmsteads = {BUILDING_SPRITES[kind] for kind in FIELD_KINDS if kind in BUILDING_SPRITES}
        kinds = np.array(
            ["farmyard" if name in farmsteads else _yard_of(name) for name in self.sheet.names]
        )
        reaches = np.array(
            [
                YARD_REACH.get(kind, max(6.0, float(width) * YARD_SHARE))
                for kind, width in zip(kinds, self.sheet.sizes[:, 0], strict=True)
            ],
            dtype=np.float32,
        )
        yards = kinds[self.scenery.sprite]
        built = yards != ""
        self.ground.buildings = (
            self.scenery.x[built].astype(np.float32),
            self.scenery.y[built].astype(np.float32),
            yards[built],
            reaches[self.scenery.sprite][built],
        )
        woodland = [self.sheet.index(BUILDING_SPRITES[kind]) for kind in IN_THE_WOODS]
        mills = np.isin(self.scenery.sprite, woodland)
        self.ground.clearings = (
            self.scenery.x[mills].astype(np.float32),
            self.scenery.y[mills].astype(np.float32),
            reaches[self.scenery.sprite][mills] * MILL_CLEARING,
        )

    def satisfaction_changed(self, levels: npt.NDArray[np.int8]) -> set[tuple[int, int]]:
        """Takes each cell's new satisfaction; returns the chunks whose glow changed. Most
        changes of level, such as from not trading to unsatisfied, leave the glow as it was."""
        changed = np.argwhere(glow_strength(levels) != glow_strength(self.satisfaction))
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
        self,
        chunk_x: int,
        chunk_y: int,
        resolution: int = 1,
        fine_relief: bool = False,
        over_ground: bool = False,
    ) -> Rgba:
        """One chunk as RGBA pixels, `resolution` of them for every art pixel each way; off the
        map it is clear. Above 1 the ground is blown up and the track and everything standing
        on it are drawn that much finer. Where hills and mountains rise, the ground with its
        track is lifted, so the chunk shows ground from below it too; with `fine_relief` their
        heights, lift and colours are worked out at the full resolution too, which is smoother
        and slower. With `over_ground` a chunk where nothing rises leaves its ground out, clear
        but for the shade of trees, to be laid over ground the graphics card draws
        (railroads.ui.pixel.ground_shader)."""
        if fine_relief and resolution > 1:
            return self._render_fine(chunk_x, chunk_y, resolution)
        if over_ground and not self.rises(chunk_x, chunk_y):
            return self._render_over_ground(chunk_x, chunk_y, resolution)
        left, top = chunk_x * CHUNK, chunk_y * CHUNK
        may_rise = self.elevation.may_rise(left, top, CHUNK, CHUNK + MOST_LIFT)
        points = self._points(left, top, CHUNK + MOST_LIFT + 1 if may_rise else CHUNK)
        raised = None
        if may_rise:
            raised = self.elevation.area(left, top, CHUNK, CHUNK + MOST_LIFT, points=points)
        offset = None if raised is None else raycast(raised.lift, CHUNK, CHUNK)
        # The ground goes a row past the furthest any pixel shows, but stops with the raised
        # ground, MOST_LIFT rows past the chunk, which is as far as any pixel shows.
        rows = CHUNK if offset is None else CHUNK + min(int(offset.max()) + 1, MOST_LIFT)
        rgb, alpha = self.ground.render(left, top, rows, raised, points)
        canvas = _blown_up(rgb / 255, alpha, resolution)
        lit = None if raised is None else lighting(raised.top(rows))
        self.ground.fields(canvas, left, top, resolution, lit, points=points)
        self.ground.yards(canvas, left, top, resolution)
        self.ground.canopy(canvas, left, top, resolution, points=points)
        if offset is None:
            self._lay_track(canvas, left, top, resolution)
        else:
            canvas = raise_picture(canvas, offset, resolution)
            self._lay_lifted_track(canvas, left, top, resolution, offset, resolution)
        self._stand_scenery(canvas, left, top, resolution)
        return _finished(canvas)

    def _points(self, left: int, top: int, rows: int) -> Lattice:
        """The plane points and cells of the art pixels a chunk's layers work out from: CHUNK
        across and `rows` down from (`left`, `top`), with the ground's apron round them and room
        above and to the left for the shadows of woods."""
        before = max(APRON, CANOPY_REACH)
        return Lattice(
            self.projection,
            left - before,
            top - before,
            CHUNK + before + APRON,
            rows + before + APRON,
            grid=self.tile_map.grid,
        )

    def _render_over_ground(self, chunk_x: int, chunk_y: int, k: int) -> Rgba:
        left, top = chunk_x * CHUNK, chunk_y * CHUNK
        points = self._points(left, top, CHUNK)
        inside, shade = self.ground.under(left, top, CHUNK, points)
        shown = inside.repeat(k, axis=0).repeat(k, axis=1)
        canvas = np.zeros((CHUNK * k, CHUNK * k, 4), dtype=np.float64)
        art_pixels(canvas, k)[..., 3] = shade[:, None, :, None]
        self.ground.fields(canvas, left, top, k, shown=shown, points=points)
        self.ground.yards(canvas, left, top, k, shown=shown)
        self.ground.canopy(canvas, left, top, k, shown=shown, points=points)
        self._lay_track(canvas, left, top, k, shown=shown)
        self._stand_scenery(canvas, left, top, k)
        return _finished(canvas)

    def _render_fine(self, chunk_x: int, chunk_y: int, k: int) -> Rgba:
        left, top = chunk_x * CHUNK, chunk_y * CHUNK
        fine = Lattice(
            self.projection, left, top, CHUNK, CHUNK + MOST_LIFT + 1, k, self.tile_map.grid
        )
        raised = self.elevation.area(left, top, CHUNK, CHUNK + MOST_LIFT + 1, k, fine)
        if raised is None:
            return self.render(chunk_x, chunk_y, k)
        offset = raycast(raised.lift, CHUNK * k, CHUNK * k)
        rows = -(-(CHUNK * k + int(offset.max()) + 1) // k)
        points = self._points(left, top, rows)
        rgb, alpha = self.ground.render(left, top, rows, points=points)
        sown = _blown_up(rgb / 255, alpha, k)
        alpha = sown[..., 3].copy()
        self.ground.fields(sown, left, top, k)
        self.ground.yards(sown, left, top, k)
        rgb = sown[..., :3] * 255
        x, y = fine.plane(left, top, CHUNK, rows)
        r, c, _ = fine.cells(left, top, CHUNK, rows)
        rocky = self.tile_map.relief[r, c] == Relief.MOUNTAINS
        self.elevation.colour(rgb, raised.top(rows * k), x, y, rocky)
        rgb = np.clip(rgb, 0, 255) * alpha[..., None]
        canvas = np.concatenate([rgb / 255, alpha[..., None]], axis=-1)
        self.ground.canopy(canvas, left, top, k, points=points)
        canvas = raise_picture(canvas, offset, 1)
        self._lay_lifted_track(canvas, left, top, k, offset / k)
        self._stand_scenery(canvas, left, top, k)
        return _finished(canvas)

    def _track_pieces(self) -> list[Track]:
        version = self.constructions.railway_version
        if version != self._pieces_version:
            self._pieces = self.constructions.track()
            self._pieces_version = version
        return self._pieces

    def _lay_track(
        self,
        canvas: FloatArray,
        left: int,
        top: int,
        resolution: int,
        shown: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        """Draws the track crossing the chunk over its ground (TrackArt.layer), where the
        picture shows the map: where `shown`, or else where it isn't clear."""
        pieces = self._track_pieces()
        if not pieces:
            return
        rows = canvas.shape[0] // resolution
        track = self.track.layer(pieces, left, top, CHUNK, rows, resolution)
        box = _drawn_box(track)
        if box is None:
            return
        part, under = track[box], canvas[box]
        on_map = under[..., 3:] > 0 if shown is None else shown[box][..., None]
        alpha = part[..., 3:] * on_map
        under[..., :3] = under[..., :3] * (1 - alpha) + part[..., :3] * alpha
        under[..., 3:] = under[..., 3:] * (1 - alpha) + alpha

    def _lay_lifted_track(
        self,
        canvas: FloatArray,
        left: int,
        top: int,
        resolution: int,
        seen: FloatArray,
        blow: int = 1,
    ) -> None:
        """Draws the track crossing the chunk over its raised ground, lifted with it, where the
        ground shown, `seen` art pixels further down the map in each pixel (of a picture `blow`
        times coarser than the chunk's), is not in front of it."""
        pieces = self._track_pieces()
        if not pieces:
            return
        track, lift = self.track.lifted(pieces, left, top, CHUNK, CHUNK, resolution)
        box = _drawn_box(track)
        if box is None:
            return
        part, under = track[box], canvas[box]
        down, across = (np.arange(span.start, span.stop) // blow for span in box)
        seen = seen[down[:, None], across[None, :]]
        shown = (seen <= lift[box] + HIDDEN_BEYOND) & (under[..., 3] > 0)
        alpha = part[..., 3:] * shown[..., None]
        under[..., :3] = under[..., :3] * (1 - alpha) + part[..., :3] * alpha

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
        art = self.track.planning(pieces)
        raised = self.elevation.area(left, top, width, height + MOST_LIFT)
        if raised is None:
            return art.layer(pieces, left, top, width, height, resolution, stations=False)
        k = resolution
        seen = raycast(raised.lift, height, width).repeat(k, axis=0).repeat(k, axis=1)
        track, lift = art.lifted(pieces, left, top, width, height, k, stations=False)
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


# Words in the names of works' sprites, which stand in a gravel yard.
WORKS = (
    "factory",
    "works",
    "mill",
    "plant",
    "refinery",
    "brewery",
    "distillery",
    "tannery",
    "bakery",
    "winery",
    "maker",
    "shipyard",
    "brickworks",
    "glassworks",
)


# How far the ground round a building reaches, as a share of its sprite's width.
YARD_SHARE = 0.42
CHURCHES = ("church", "basilica", "cathedral", "st_pauls")
LANDMARKS = (
    "royal_palace",
    "spa_resort",
    "worlds_fair",
    "pear_hq",
    "space_center",
    "film_studio",
    "amusement_park",
    "olympic_stadium",
    "salt_cathedral",
)
TOWN = (
    "tenement_",
    "flats_",
    "skyscraper",
    "tavern",
    "department_store",
    "hotel",
    "car_dealer",
    "jeweller",
    "headquarters",
    "airport",
    "port",
)
DIGGINGS = (
    "mine",
    "oil_well",
    "salt_works",
    "clay_pit",
    "limestone_quarry",
    "sand_pit",
    "lumber_mill",
    "outpost",
    "base",
)


def _yard_of(name: str) -> str:
    """What ground lies round the building a sprite shows (railroads.ui.pixel.ground.YARD_COLOURS),
    or nothing."""
    if name.startswith("house_"):
        return "cottage"
    if name.startswith(TOWN):
        return "paved"
    if name in CHURCHES or name in LANDMARKS:
        return "flagstones"
    if name.endswith(DIGGINGS) or name in DIGGINGS:
        return "spoil"
    if name in ("date_palms",):
        return "sand"
    if any(word in name for word in WORKS):
        return "gravel"
    return ""


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


def _blown_up(rgb: FloatArray, alpha: FloatArray, k: int) -> FloatArray:
    """RGB and its alpha as one RGBA picture, each pixel blown up to `k` by `k`."""
    picture = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return picture if k == 1 else picture.repeat(k, axis=0).repeat(k, axis=1)


def _finished(canvas: FloatArray) -> Rgba:
    """A premultiplied RGBA picture from 0 to 1 as RGBA bytes, its colours no longer multiplied
    by its alpha; the picture itself is used up working them out."""
    canvas[..., :3] /= np.maximum(canvas[..., 3:], 1e-6)
    np.clip(canvas, 0, 1, out=canvas)
    canvas *= 255
    np.rint(canvas, out=canvas)
    return np.ascontiguousarray(canvas.astype(np.uint8))


def _drawn_box(layer: FloatArray) -> tuple[slice, slice] | None:
    """The smallest box holding every pixel of an RGBA layer that isn't clear; None if all are,
    so blending the layer can skip the rest, which it would leave as it is."""
    drawn = layer[..., 3] != 0
    rows = np.flatnonzero(drawn.any(axis=1))
    if not len(rows):
        return None
    cols = np.flatnonzero(drawn.any(axis=0))
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
