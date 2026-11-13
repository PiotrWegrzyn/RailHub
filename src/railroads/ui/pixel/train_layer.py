"""Shows trains as pixel art over the pixel map: each engine, tender and car drawn from the train
sprite sheet (railroads.ui.pixel.trains), turned to the track, lined up on the map's own art
pixels and lifted with the ground where hills and mountains rise, with puffs of smoke from the
chimney. On double track each train rides the track on its right, as the track art draws it, so
trains going opposite ways pass on different tracks."""

import bisect
import itertools
import math
from dataclasses import dataclass

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.sim.constructions import Track
from railroads.sim.railway import Railway
from railroads.sim.trains import Point, Train
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, IsoCamera, Vec2
from railroads.ui.pixel.models import HEIGHT_PER_WIDTH
from railroads.ui.pixel.track_art import BEND, TrackArt
from railroads.ui.pixel.trains import (
    CAR_HALF,
    COUPLING,
    DETAIL,
    ENGINES,
    SCALE,
    Look,
    TrainSprites,
    train_sprites,
)
from railroads.ui.pixel.world import PixelWorld

SMOKE = (0.88, 0.88, 0.9)
# The biggest texture a graphics card takes on each side; a bigger one draws black.
MAX_TEXTURE = 16384
PUFFS = 5


class TrainLayer:
    def __init__(self, world: PixelWorld) -> None:
        self.world = world
        matrix = world.projection.matrix.ravel()
        self.sprites: TrainSprites = train_sprites(tuple(float(v) for v in matrix))
        self._textures: dict[int, tuple[hello_imgui.TextureGpu, int]] = {}
        self.lanes = Lanes(world.track)

    def draw(
        self, draw_list: imgui.ImDrawList, railway: Railway, camera: IsoCamera, center: Vec2
    ) -> None:
        projection = self.world.projection
        scale = camera.pixel_scale
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        camera_pixel = projection.to_pixels(camera.center_x, camera.center_y)
        origin = (
            round(center[0] - float(camera_pixel[0]) * scale),
            round(center[1] - float(camera_pixel[1]) * scale),
        )
        texture, _ = self._texture(scale * density)
        height, width = self.sprites.image.shape[:2]
        per_pixel = scale / DETAIL
        found = []
        for train in railway.trains:
            for look, back in self._parts(train):
                placed = railway.position(train, back)
                if placed is not None:
                    shift = self.lanes.sideways(train.path, train.along - back)
                    found.append((placed, shift, look))
        parts = []
        if found:
            xs = np.array([placed[0][0] for placed, _, _ in found])
            ys = np.array([placed[0][1] for placed, _, _ in found])
            ax, ay = projection.to_pixels(xs, ys)
            ay = ay - self.world.elevation.lift(xs, ys)
            for k, (((_, _), (dx, dy)), shift, look) in enumerate(found):
                sx = origin[0] + round((float(ax[k]) + shift[0]) * DETAIL) * per_pixel
                sy = origin[1] + round((float(ay[k]) + shift[1]) * DETAIL) * per_pixel
                parts.append((sy, sx, look, math.atan2(dy, dx)))
        parts.sort(key=lambda part: part[0])
        ref = imgui.ImTextureRef(texture.texture_id())
        for sy, sx, look, heading in parts:
            px, py, w, h, anchor_x, anchor_y = self.sprites.place(look, heading)
            left = sx - anchor_x * per_pixel
            top = sy - anchor_y * per_pixel
            draw_list.add_image(
                ref,
                (left, top),
                (left + w * per_pixel, top + h * per_pixel),
                (px / width, py / height),
                ((px + w) / width, (py + h) / height),
            )
        self._smoke(draw_list, railway, origin, scale)

    def _parts(self, train: Train) -> list[tuple[object, float]]:
        """Each part of the train and how far its middle is behind the train's front, in cells."""
        model = ENGINES[train.engine]
        parts: list[tuple[object, float]] = [((Look.ENGINE, train.engine), model.half)]
        back = 2 * model.half
        if model.tender is not None:
            back += COUPLING + model.tender_half
            parts.append(((Look.TENDER, train.engine), back))
            back += model.tender_half
        for car in train.cars:
            back += COUPLING + CAR_HALF
            parts.append(((car.good, car.load > 0), back))
            back += CAR_HALF
        return [(look, middle * SCALE) for look, middle in parts]

    def _smoke(
        self,
        draw_list: imgui.ImDrawList,
        railway: Railway,
        origin: tuple[int, int],
        scale: int,
    ) -> None:
        """Puffs of smoke rising from each chimney and drifting back over its train. The ground
        under every chimney and where every puff lies in the picture are worked out for all the
        trains at once: on hills a lift takes nearly as long for one point as for many."""
        chimneys = [
            chimney
            for train in railway.trains
            if (chimney := self._chimney(railway, train)) is not None
        ]
        if not chimneys:
            return
        projection = self.world.projection
        lifts = self.world.elevation.lift(
            np.array([chimney.x for chimney in chimneys]),
            np.array([chimney.y for chimney in chimneys]),
        )
        up = float(np.hypot(*projection.matrix[:, 0])) * HEIGHT_PER_WIDTH
        puffs = []
        for index, chimney in enumerate(chimneys):
            length = math.hypot(chimney.dx, chimney.dy) or 1.0
            ux, uy = chimney.dx / length, chimney.dy / length
            for k in range(PUFFS):
                age = (k + chimney.phase) / PUFFS
                px, py = chimney.x - ux * age * 0.5 * SCALE, chimney.y - uy * age * 0.5 * SCALE
                puffs.append((index, age, px, py))
        ax, ay = projection.to_pixels(
            np.array([puff[2] for puff in puffs]), np.array([puff[3] for puff in puffs])
        )
        for (index, age, _, _), x, y in zip(puffs, ax.tolist(), ay.tolist(), strict=True):
            chimney = chimneys[index]
            shift = chimney.shift
            sx = origin[0] + (x + shift[0]) * scale
            rise = (chimney.height + age * 0.35 * SCALE) * up
            sy = origin[1] + (y + shift[1] - float(lifts[index]) - rise) * scale
            radius = (1.5 + age * 4.0) * SCALE * scale
            alpha = 0.75 * (1 - age)
            color = imgui.color_convert_float4_to_u32(imgui.ImVec4(*SMOKE, alpha))
            draw_list.add_circle_filled((sx, sy), radius, color)

    def _chimney(self, railway: Railway, train: Train) -> _Chimney | None:
        """Where a train's chimney is on the map, if it has one and is on the map."""
        model = ENGINES[train.engine]
        if model.chimney is None:
            return None
        back = (model.half - model.chimney[0]) * SCALE
        placed = railway.position(train, back)
        if placed is None:
            return None
        (x, y), (dx, dy) = placed
        shift = self.lanes.sideways(train.path, train.along - back)
        phase = (train.along * 6.0) % 1.0
        return _Chimney(x, y, dx, dy, shift, model.chimney[1] * SCALE, phase)

    def _texture(self, device_scale: int) -> tuple[hello_imgui.TextureGpu, int]:
        """The sprite sheet blown up by a whole number so it shows crisp at this many device
        pixels per art pixel, and by how much it was blown up."""
        height, width = self.sprites.image.shape[:2]
        fits = max(1, MAX_TEXTURE // max(height, width))
        factor = max(1, min(device_scale // DETAIL, fits))
        if factor not in self._textures:
            big = np.repeat(np.repeat(self.sprites.image, factor, axis=0), factor, axis=1)
            texture = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(big))
            self._textures[factor] = (texture, factor)
        return self._textures[factor]


@dataclass(frozen=True, slots=True)
class _Chimney:
    """A train's chimney: where it is on the map, which way the train runs, how far across the
    picture the train is moved onto its track, how high the chimney stands in model cells, and
    how far along its cycle the smoke is."""

    x: float
    y: float
    dx: float
    dy: float
    shift: Point
    height: float
    phase: float


@dataclass(frozen=True)
class _Stretch:
    """One stretch of a train's path, from the middle of a cell to an edge or back: how far into
    the path it starts and how long it is, the piece of track it runs along, whether it runs in
    to the middle, whether its edge is the piece's first end, how long each half of the piece is,
    and which way it runs across the picture and the stretch it meets in the middle runs."""

    start: float
    length: float
    piece: Track | None
    inward: bool
    at_first: bool
    halves: tuple[float, float]
    way: Point
    turn: Point | None


class Lanes:
    """Where on double track trains ride: each a little to the side of its path's centre line,
    on the track the track art draws there (TrackArt.riding_side)."""

    def __init__(self, track: TrackArt) -> None:
        self.track = track
        self._paths: dict[int, tuple[list[Point], int, list[_Stretch], list[float]]] = {}

    def sideways(self, path: list[Point], along: float) -> Point:
        """How far across the picture, in art pixels, a train `along` cells into its path is
        moved off the path's centre line to ride the track drawn there: square to the way it
        goes, turning smoothly with the track round the middle of a cell."""
        stretches, starts = self._stretches(path)
        if not stretches:
            return 0.0, 0.0
        stretch = stretches[max(0, bisect.bisect_right(starts, along) - 1)]
        if stretch.piece is None:
            return 0.0, 0.0
        t = min(1.0, max(0.0, (along - stretch.start) / (stretch.length or 1.0)))
        from_middle = (1 - t if stretch.inward else t) * stretch.length
        first, second = stretch.halves
        if stretch.at_first:
            share, forward = (first - from_middle) / (first + second), stretch.inward
        else:
            share, forward = (first + from_middle) / (first + second), not stretch.inward
        side = self.track.riding_side(stretch.piece, share, forward)
        if side == 0.0:
            return 0.0, 0.0
        dx, dy = stretch.way
        if stretch.turn is not None and from_middle < BEND:
            mix = 0.5 * (1 - from_middle / BEND)
            dx, dy = (1 - mix) * dx + mix * stretch.turn[0], (1 - mix) * dy + mix * stretch.turn[1]
        length = math.hypot(dx, dy) or 1.0
        return -dy / length * side * PIXEL_CELL_WIDTH, dx / length * side * PIXEL_CELL_WIDTH

    def _stretches(self, path: list[Point]) -> tuple[list[_Stretch], list[float]]:
        """The path's stretches and how far into it each starts, worked out again only when
        track or stations have changed."""
        version = self.track.constructions.railway_version
        kept = self._paths.get(id(path))
        if kept is not None and kept[0] is path and kept[1] == version:
            return kept[2], kept[3]
        if len(self._paths) > 256:
            self._paths.clear()
        stretches = self._lay_out(path)
        starts = [stretch.start for stretch in stretches]
        self._paths[id(path)] = (path, version, stretches, starts)
        return stretches, starts

    def _lay_out(self, path: list[Point]) -> list[_Stretch]:
        """The stretches of a path that runs from cell middle to cell edge and on
        (TrackNetwork.route)."""
        grid = self.track.tile_map.grid
        constructions = self.track.constructions
        steps = grid.track_steps()
        matrix = self.track.projection.matrix
        ways = []
        for a, b in itertools.pairwise(path):
            dx, dy = matrix @ (b[0] - a[0], b[1] - a[1])
            length = math.hypot(dx, dy) or 1.0
            ways.append((float(dx / length), float(dy / length)))
        stretches = []
        start = 0.0
        for index, (a, b) in enumerate(itertools.pairwise(path)):
            length = math.dist(a, b)
            row, col = grid.cell_at((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            cell = (int(row), int(col))
            cx, cy = (float(v) for v in grid.center(*cell))
            inward = math.dist(b, (cx, cy)) < math.dist(a, (cx, cy))
            edge = a if inward else b
            point = _point(steps, edge[0] - cx, edge[1] - cy)
            beyond = index + 2 if inward else index - 1
            met = index + 1 if inward else index - 1
            piece: Track | None = None
            if constructions.station_at(cell) is not None:
                piece = self.track.station_piece(cell)
            elif 0 <= beyond < len(path):
                other = _point(steps, path[beyond][0] - cx, path[beyond][1] - cy)
                piece = next(
                    (p for p in constructions.track_at(cell) if set(p.ends) == {point, other}),
                    None,
                )
            if piece is not None and point not in piece.ends:
                piece = None
            halves = (0.5, 0.5)
            if piece is not None:
                first, second = (float(np.hypot(*steps[end])) / 2 for end in piece.ends)
                halves = (first, second)
            stretches.append(
                _Stretch(
                    start=start,
                    length=length,
                    piece=piece,
                    inward=inward,
                    at_first=piece is not None and point == piece.ends[0],
                    halves=halves,
                    way=ways[index],
                    turn=ways[met] if 0 <= met < len(ways) else None,
                )
            )
            start += length
        return stretches


def _point(steps: np.ndarray, dx: float, dy: float) -> int:
    """The track point this far from a cell's middle lies at."""
    return int(np.argmin(np.hypot(steps[:, 0] / 2 - dx, steps[:, 1] / 2 - dy)))
