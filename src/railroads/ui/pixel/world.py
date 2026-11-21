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

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import Constructions, Track
from railroads.sim.grids import Cell, FloatArray, IntArray
from railroads.sim.steps import Steps, finish
from railroads.sim.tilemap import Relief, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH
from railroads.ui.pixel.ground import (
    APRON,
    CANOPY_REACH,
    FIELD_KINDS,
    IN_THE_WOODS,
    MILL_CLEARING,
    OVERHANG,
    YARD_REACH,
    Ground,
    art_pixels,
    glow_strength,
)
from railroads.ui.pixel.ground_texture import textured
from railroads.ui.pixel.lattice import Lattice
from railroads.ui.pixel.projection import CHUNK, PixelProjection, Rgba
from railroads.ui.pixel.relief import (
    MOST_LIFT,
    Elevation,
    crossings,
    lighting,
    raise_picture,
    raycast,
)
from railroads.ui.pixel.scenery import BUILDING_SPRITES, Scenery
from railroads.ui.pixel.sheet import FINE, SpriteSheet
from railroads.ui.pixel.track_art import TrackArt

# How far in art pixels a sprite can reach from where it stands.
SPRITE_REACH = 48
# The resolution chunks are drawn at in close views: the ground, fields and woods that fine, and
# what stands on the ground as its pixel art from the finest sheet, each pixel blown up to a square.
CLOSE = 4
# How many art pixels further down the map than track the ground showing where it is drawn may
# lie before the track counts as hidden behind it: the track's own width over gentle slopes.
HIDDEN_BEYOND = 2.0


class PixelWorld:
    """Draws chunks of one map as pixel art."""

    # The mountain cells track tunnels through, with the railway version they were found for.
    _tunnels: tuple[int, frozenset[Cell]] = (-1, frozenset())

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
        finish(
            self._build(tile_map, constructions, sheet, angle_deg, cell_units, fine, station_levels)
        )

    @classmethod
    def built(
        cls,
        tile_map: TileMap,
        constructions: Constructions,
        sheet: SpriteSheet,
        angle_deg: float,
        cell_units: float,
        fine: SpriteSheet | None = None,
        station_levels: Mapping[Cell, int] | None = None,
    ) -> Steps[Self]:
        """The world made a part at a time, for a page to go on drawing frames between them."""
        world = cls.__new__(cls)
        yield from world._build(
            tile_map, constructions, sheet, angle_deg, cell_units, fine, station_levels
        )
        return world

    def _build(
        self,
        tile_map: TileMap,
        constructions: Constructions,
        sheet: SpriteSheet,
        angle_deg: float,
        cell_units: float,
        fine: SpriteSheet | None,
        station_levels: Mapping[Cell, int] | None,
    ) -> Steps[None]:
        self.tile_map = tile_map
        self.constructions = constructions
        self.sheet = sheet
        self.projection = PixelProjection.of(tile_map.grid, angle_deg, cell_units)
        # Each station's level (railroads.sim.railway), which picks how grand it looks.
        self.station_levels: Mapping[Cell, int] = {} if station_levels is None else station_levels
        # The height of every point, which raises hills and mountains (railroads.ui.pixel.relief).
        self.elevation = Elevation(tile_map, self.projection)
        yield
        # The track (railroads.ui.pixel.track_art), with the ground graded under it.
        self.track = TrackArt(tile_map, constructions, self.projection, self.elevation)
        self._graded_version = -1
        self._grade()
        yield
        self.scenery = Scenery.of(
            tile_map,
            constructions,
            self.projection,
            sheet,
            self.station_levels,
            elevation=self.elevation,
        )
        yield
        # The ground under the track (railroads.ui.pixel.ground).
        self.ground = Ground(tile_map, constructions, self.projection, angle_deg, self.elevation)
        yield
        self._shade_trees()
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

    def sprite(self, index: int) -> tuple[int, npt.NDArray[np.float32], tuple[int, int]]:
        """A sprite as the finest sheet holds it: that sheet's resolution, the sprite's RGBA
        pixels from 0 to 1 and its anchor."""
        resolution = max(self._sprites)
        sprites = self._sprites[resolution]
        anchor = sprites.anchors[index]
        return resolution, sprites.pixels[index], (int(anchor[0]), int(anchor[1]))

    @property
    def resolutions(self) -> tuple[int, ...]:
        """The resolutions chunks can be drawn at, in pixels per art pixel."""
        return tuple(self._sprites)

    def buildings_changed(self, cells: Iterable[Cell]) -> set[tuple[int, int]]:
        """Puts the ground and scenery up again after buildings or forests changed in `cells`;
        returns the chunks that look different now."""
        return finish(self.buildings_changed_steps(cells))

    def buildings_changed_steps(self, cells: Iterable[Cell]) -> Steps[set[tuple[int, int]]]:
        """buildings_changed in steps, nothing changing the world between them."""
        self._grade()
        self.ground.refresh()
        yield
        self.scenery = Scenery.of(
            self.tile_map,
            self.constructions,
            self.projection,
            self.sheet,
            self.station_levels,
            self.elevation,
        )
        yield
        self._shade_trees()
        self.buildings_version += 1
        return self._chunks_around(cells)

    def _grade(self) -> None:
        """Grades the ground under the track through hills and mountains (Elevation.grade), if
        track or stations changed since it last was."""
        version = self.constructions.railway_version
        if version == self._graded_version:
            return
        beds = []
        for line in self.track.beds():
            mountains, hills = self.elevation.shares(*line)
            if (mountains > 0).any() or (hills > 0).any():
                beds.append(line)
        self.elevation.grade(beds)
        self._graded_version = version

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

    def underground(self, x: float, y: float) -> bool:
        """Whether a plane point lies in a tunnel: in a mountain cell its track tunnels through,
        where trains go out of sight until they come out of a portal."""
        grid = self.tile_map.grid
        row, col = (int(v) for v in grid.cell_at(x, y))
        if not grid.contains(row, col) or self.tile_map.relief[row, col] != Relief.MOUNTAINS:
            return False
        return any(
            piece.bed == TrackBed.TUNNEL for piece in self.constructions.track_at((row, col))
        )

    def underground_all(self, xs: list[float], ys: list[float]) -> list[bool]:
        """Whether each of many plane points lies in a tunnel (underground), all found at once,
        as trains ask for every engine and car each frame."""
        tunnels = self._tunnel_cells()
        if not tunnels or not xs:
            return [False] * len(xs)
        rows, cols = self.tile_map.grid.cell_at(np.array(xs), np.array(ys))
        return [cell in tunnels for cell in zip(rows.tolist(), cols.tolist(), strict=True)]

    def _tunnel_cells(self) -> frozenset[Cell]:
        version = self.constructions.railway_version
        if self._tunnels[0] != version:
            relief = self.tile_map.relief
            cells = frozenset(
                piece.cell
                for piece in self._track_pieces()
                if piece.bed == TrackBed.TUNNEL and relief[piece.cell] == Relief.MOUNTAINS
            )
            self._tunnels = (version, cells)
        return self._tunnels[1]

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
        surfaces: bool = True,
    ) -> Rgba:
        """One chunk as RGBA pixels, `resolution` of them for every art pixel each way; off the
        map it is clear. Above 1 the ground is blown up and the track and everything standing
        on it are drawn that much finer. Where hills and mountains rise, the ground with its
        track is lifted, so the chunk shows ground from below it too; with `fine_relief` their
        heights, lift and colours are worked out at the full resolution too, which is smoother
        and slower. With `over_ground` the chunk leaves its ground out, clear but for the shade
        of trees, to be laid over ground the graphics card draws and raises
        (railroads.ui.pixel.ground_shader); what lies on raised ground is raised with it. Over
        ground, without `surfaces` the chunk leaves out what lies on the ground too, the shade of
        trees, fields, yards and woods, which the graphics card lays under it
        (GroundShader.dress)."""
        self._grade()
        if fine_relief and resolution > 1:
            return self._render_fine(chunk_x, chunk_y, resolution)
        if over_ground:
            if self.rises(chunk_x, chunk_y):
                return self._render_over_raised(chunk_x, chunk_y, resolution, surfaces)
            return self._render_over_ground(chunk_x, chunk_y, resolution, surfaces)
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
        flat = offset is None
        canvas = _ground_blown_up(rgb / 255, alpha, resolution, left, top, flat)
        lit = None if raised is None else lighting(raised.top(rows))
        self._surfaces(
            canvas,
            left,
            top,
            resolution,
            None,
            lambda layer, k, shown: (
                self.ground.fields(layer, left, top, k, lit, shown=shown, points=points),
                self.ground.yards(layer, left, top, k, shown=shown),
                self.ground.canopy(layer, left, top, k, shown=shown, points=points),
            ),
            flat,
        )
        if offset is None:
            self._stand_scenery(canvas, left, top, resolution, under_track=True)
            self._lay_track(canvas, left, top, resolution)
        else:
            canvas = raise_picture(canvas, offset, resolution, crossings(raised.lift, offset))
            if resolution > 1:
                textured(canvas, left, top, resolution)
            self._stand_scenery(canvas, left, top, resolution, under_track=True)
            self._lay_lifted_track(canvas, left, top, resolution, offset, resolution)
        self._stand_scenery(canvas, left, top, resolution)
        return _finished(canvas)

    def _surfaces(
        self,
        canvas: FloatArray,
        left: int,
        top: int,
        k: int,
        shown: npt.NDArray[np.bool_] | None,
        paint: Callable[[FloatArray, int, npt.NDArray[np.bool_]], object],
        texture: bool = True,
    ) -> None:
        """Lays what `paint` draws over the ground, fields, yards and woods, onto `canvas`, `k`
        pixels to an art pixel, where it shows the map (`shown`, or wherever it isn't clear).
        Above 1 they are drawn at no more than FINE, blended between their pixels up to `k` and
        laid with the ground's texture (railroads.ui.pixel.ground_texture), so they read as the
        same lit surface as the ground under them."""
        on = canvas[..., 3] > 0 if shown is None else shown
        if k == 1:
            paint(canvas, 1, on)
            return
        drawn = min(k, FINE)
        step = k // drawn
        layer = np.zeros((canvas.shape[0] // step, canvas.shape[1] // step, 4))
        paint(layer, drawn, on[::step, ::step])
        if step > 1:
            layer = _smoothly_blown_up(layer[..., :3], layer[..., 3], step)
        if texture:
            textured(layer, left, top, k)
        canvas[...] = layer + canvas * (1 - layer[..., 3:])

    def _points(self, left: int, top: int, rows: int) -> Lattice:
        """The plane points and cells of the art pixels a chunk's layers work out from: CHUNK
        across and `rows` down from (`left`, `top`), with the ground's apron round them and room
        above and to the left for the shadows of woods."""
        before = max(APRON, CANOPY_REACH + OVERHANG)
        return Lattice(
            self.projection,
            left - before,
            top - before,
            CHUNK + before * 2,
            rows + before * 2,
            grid=self.tile_map.grid,
        )

    def _render_over_ground(
        self, chunk_x: int, chunk_y: int, k: int, surfaces: bool = True
    ) -> Rgba:
        left, top = chunk_x * CHUNK, chunk_y * CHUNK
        points = self._points(left, top, CHUNK)
        shown, canvas = self._over(left, top, CHUNK, k, points, surfaces)
        if surfaces:
            self._surfaces_over(canvas, left, top, k, shown, points)
        self._stand_scenery(canvas, left, top, k, under_track=True)
        self._lay_track(canvas, left, top, k, shown=shown)
        self._stand_scenery(canvas, left, top, k)
        return _finished(canvas)

    def _render_over_raised(
        self, chunk_x: int, chunk_y: int, k: int, surfaces: bool = True
    ) -> Rgba:
        left, top = chunk_x * CHUNK, chunk_y * CHUNK
        points = self._points(left, top, CHUNK + MOST_LIFT + 1)
        raised = self.elevation.area(left, top, CHUNK, CHUNK + MOST_LIFT, points=points)
        if raised is None:
            return self._render_over_ground(chunk_x, chunk_y, k, surfaces)
        offset = raycast(raised.lift, CHUNK, CHUNK)
        rows = CHUNK + min(int(offset.max()) + 1, MOST_LIFT)
        shown, canvas = self._over(left, top, rows, k, points, surfaces)
        lit = lighting(raised.top(rows))
        if surfaces:
            self._surfaces_over(canvas, left, top, k, shown, points, lit)
        between = crossings(raised.lift, offset)
        canvas = raise_picture(canvas, offset, k, between)
        on_map = raise_picture(shown[::k, ::k].astype(np.float32), offset, 1, between) > 0
        if k > 1:
            if surfaces:
                textured(canvas, left, top, k)
            on_map = on_map.repeat(k, axis=0).repeat(k, axis=1)
        self._stand_scenery(canvas, left, top, k, under_track=True)
        self._lay_lifted_track(canvas, left, top, k, offset, k, on_map)
        self._stand_scenery(canvas, left, top, k)
        return _finished(canvas)

    def _over(
        self, left: int, top: int, rows: int, k: int, points: Lattice, surfaces: bool
    ) -> tuple[npt.NDArray[np.bool_], FloatArray]:
        """Which pixels of a chunk drawn over ground, CHUNK art pixels across and `rows` down
        from (`left`, `top`), `k` pixels to each, lie on the map, and a clear canvas for it,
        shaded under trees unless the graphics card shades it (without `surfaces`)."""
        canvas = np.zeros((rows * k, CHUNK * k, 4), dtype=np.float64)
        if surfaces:
            inside, shade = self.ground.under(left, top, rows, points)
            art_pixels(canvas, k)[..., 3] = shade[:, None, :, None]
        else:
            inside = points.cells(left, top, CHUNK, rows)[2]
        return inside.repeat(k, axis=0).repeat(k, axis=1), canvas

    def _surfaces_over(
        self,
        canvas: FloatArray,
        left: int,
        top: int,
        k: int,
        shown: npt.NDArray[np.bool_],
        points: Lattice,
        lit: FloatArray | None = None,
    ) -> None:
        """Lays fields, yards and woods over a chunk drawn over ground; raised (with `lit`), they
        take the ground's texture once raised with it."""
        ground = self.ground

        def paint(layer: FloatArray, kk: int, on: npt.NDArray[np.bool_]) -> None:
            ground.fields(layer, left, top, kk, lit, shown=on, points=points)
            ground.yards(layer, left, top, kk, shown=on)
            ground.canopy(layer, left, top, kk, shown=on, points=points)

        self._surfaces(canvas, left, top, k, shown, paint, lit is None)

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
        sown = _ground_blown_up(rgb / 255, alpha, k, left, top, False)
        alpha = sown[..., 3].copy()
        self._surfaces(
            sown,
            left,
            top,
            k,
            None,
            lambda layer, kk, on: (
                self.ground.fields(layer, left, top, kk, shown=on),
                self.ground.yards(layer, left, top, kk, shown=on),
            ),
            False,
        )
        rgb = sown[..., :3] * 255
        x, y = fine.plane(left, top, CHUNK, rows)
        r, c, _ = fine.cells(left, top, CHUNK, rows)
        wr, wc = self.ground.winding(x, y, r, c)
        rocky = self.tile_map.relief[wr, wc] == Relief.MOUNTAINS
        self.elevation.colour(rgb, raised.top(rows * k), x, y, rocky)
        rgb = np.clip(rgb, 0, 255) * alpha[..., None]
        canvas = np.concatenate([rgb / 255, alpha[..., None]], axis=-1)
        self._surfaces(
            canvas,
            left,
            top,
            k,
            None,
            lambda layer, kk, on: self.ground.canopy(layer, left, top, kk, shown=on, points=points),
            False,
        )
        canvas = raise_picture(canvas, offset, 1, crossings(raised.lift, offset))
        textured(canvas, left, top, k)
        self._stand_scenery(canvas, left, top, k, under_track=True)
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
        on_map: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        """Draws the track crossing the chunk over its raised ground, lifted with it, where the
        ground shown, `seen` art pixels further down the map in each pixel (of a picture `blow`
        times coarser than the chunk's), is not in front of it, and the picture shows the map:
        where `on_map`, for a picture of what lies over the ground, or else where it isn't
        clear."""
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
        mapped = under[..., 3] > 0 if on_map is None else on_map[box]
        shown = (seen <= lift[box] + HIDDEN_BEYOND) & mapped
        alpha = part[..., 3:] * shown[..., None]
        under[..., :3] = under[..., :3] * (1 - alpha) + part[..., :3] * alpha
        if on_map is not None:
            under[..., 3:] = under[..., 3:] * (1 - alpha) + alpha

    def _stand_scenery(
        self, canvas: FloatArray, left: int, top: int, resolution: int, under_track: bool = False
    ) -> None:
        """Stands what stands on the ground in the chunk on it, back to front: with
        `under_track` only what is drawn under the track (Scenery.under_track), and otherwise the
        rest."""
        sheet = resolution if resolution in self._sprites else max(self._sprites)
        times = resolution // sheet
        scenery, drawn, k = self.scenery, self._sprites[sheet], resolution
        x0 = (scenery.x - left) * k - drawn.anchors[scenery.sprite, 0] * times
        y0 = (scenery.y - scenery.lift - top) * k - drawn.anchors[scenery.sprite, 1] * times
        w, h = drawn.sizes[scenery.sprite, 0] * times, drawn.sizes[scenery.sprite, 1] * times
        span = CHUNK * k
        overlapping = (x0 < span) & (x0 + w > 0) & (y0 < span) & (y0 + h > 0)
        overlapping &= scenery.under_track == under_track
        for i in np.flatnonzero(overlapping).tolist():
            pixels = drawn.premultiplied[int(scenery.sprite[i])]
            if times > 1:
                pixels = pixels.repeat(times, axis=0).repeat(times, axis=1)
            _blend(canvas, pixels, int(x0[i]), int(y0[i]))

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
CHURCHES = ("church", "basilica", "cathedral", "st_peters")
LANDMARKS = (
    "royal_palace",
    "spa_resort",
    "worlds_fair",
    "pear_hq",
    "gigafactory",
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
    "logging_camp",
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


def _ground_blown_up(
    rgb: FloatArray, alpha: FloatArray, k: int, left: int, top: int, texture: bool = True
) -> FloatArray:
    """The ground's RGB and alpha as one RGBA picture `k` pixels for every art pixel from
    (`left`, `top`): blown up to squares at 1, otherwise blended between its pixels and, with
    `texture`, laid with the ground's texture (railroads.ui.pixel.ground_texture), as the ground
    shader draws. Raised ground is laid with it only once it is raised, so climbing slopes don't
    stretch it into streaks."""
    if k == 1:
        return _blown_up(rgb, alpha, k)
    picture = _smoothly_blown_up(rgb, alpha, k)
    if texture:
        textured(picture, left, top, k)
    return picture


def _smoothly_blown_up(rgb: FloatArray, alpha: FloatArray, k: int) -> FloatArray:
    """RGB and its alpha as one RGBA picture `k` times as big each way, blended between the
    middles of its pixels instead of each blown up to a square: the ground up close."""
    picture = np.concatenate([rgb, alpha[..., None]], axis=-1)
    for axis in (0, 1):
        n = picture.shape[axis]
        at = np.clip((np.arange(n * k) + 0.5) / k - 0.5, 0, n - 1)
        low = np.floor(at).astype(np.intp)
        high = np.minimum(low + 1, n - 1)
        share = (at - low).reshape([-1 if a == axis else 1 for a in range(3)])
        picture = (
            np.take(picture, low, axis=axis) * (1 - share)
            + np.take(picture, high, axis=axis) * share
        )
    return picture


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
