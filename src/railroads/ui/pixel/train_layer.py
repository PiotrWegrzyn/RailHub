"""Shows trains as pixel art over the pixel map: each engine, tender and car drawn from the train
sprite sheet (railroads.ui.pixel.trains), turned to the track, lined up on the map's own art
pixels and lifted with the ground where hills and mountains rise, with puffs of smoke from the
chimney, and out of sight in tunnels. On double track each train rides the track on its right, as the track art draws it, so
trains going opposite ways pass on different tracks."""

import bisect
import itertools
import math
import time
from collections import OrderedDict
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from imgui_bundle import hello_imgui, imgui
from PIL import Image

from railroads.sim.constructions import Track, TrackWidth
from railroads.sim.grids import Cell
from railroads.sim.railway import Railway
from railroads.sim.signals import blocks_of
from railroads.sim.trains import Point, Train
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, IsoCamera, Vec2
from railroads.ui.pixel.art.registry import Sprite
from railroads.ui.pixel.models import HEIGHT_PER_WIDTH
from railroads.ui.pixel.relief import MOST_LIFT
from railroads.ui.pixel.track_art import BEND, TrackArt
from railroads.ui.pixel.trains import (
    CAR_HALF,
    COUPLING,
    DETAIL,
    ENGINES,
    HEADINGS,
    SCALE,
    FinePainter,
    GrownTrainSprites,
    Look,
    TrainSprites,
    fine_pictures,
    turned,
)

type Pixels = npt.NDArray[np.uint8]
from railroads.ui.pixel.world import PixelWorld

# Coal smoke leaves the chimney dark and thins out paler as it drifts back.
SMOKE = (0.22, 0.21, 0.21)
THIN_SMOKE = (0.6, 0.6, 0.62)
# The biggest texture made for trains on each side: many graphics cards and browsers take no
# bigger, and draw a bigger one black.
MAX_TEXTURE = 4096
# How wide a page of one look's pictures is at the least, in sheet pixels, how many bytes the
# pages, of every zoom shown, may take before those not drawn lately are let go, and how long a
# frame may spend drawing pages: a look whose page is not drawn yet shows from the sheet meanwhile.
PAGE_WIDTH = 512
PAGES_BUDGET = 128 * 1024 * 1024
PAGES_TIME = 0.006
PUFFS = 5


class Texture(Protocol):
    def texture_id(self) -> int: ...


class TrainLayer:
    def __init__(self, world: PixelWorld, trains: Iterable[Train] = ()) -> None:
        """A layer drawing trains on `world`, with the looks of `trains` drawn up front and the
        others drawn as they are wanted or as frames have time to spare."""
        self.world = world
        matrix = world.projection.matrix.ravel()
        self._matrix = tuple(float(v) for v in matrix)
        looks = (look for train in trains for look, _ in self._parts(train))
        self.sprites = GrownTrainSprites(self._matrix, looks)
        # The sheet's texture, its size and the looks on it.
        self._sheet: Texture | None = None
        self._sheet_shape = (0, 0)
        self._sheet_looks: set[object] = set()
        self._paging_until = 0.0
        # Each look's pages by how many times as fine as the sheet they are drawn, kept for every
        # zoom shown, so coming back to a zoom shows them at once.
        self._pages: OrderedDict[tuple[object, int], _Page] = OrderedDict()
        # Pages being drawn a few pictures a frame: each look's painter and pictures so far.
        self._painting: dict[tuple[object, int], tuple[FinePainter, list[Sprite]]] = {}
        self.lanes = Lanes(world.track)

    def draw(
        self, draw_list: imgui.ImDrawList, railway: Railway, camera: IsoCamera, center: Vec2
    ) -> None:
        projection = self.world.projection
        scale = camera.pixel_scale or camera.cell_width / PIXEL_CELL_WIDTH
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        camera_pixel = projection.to_pixels(camera.center_x, camera.center_y)
        origin = (
            round(center[0] - float(camera_pixel[0]) * scale),
            round(center[1] - float(camera_pixel[1]) * scale),
        )
        factor = self._blown_up_by(int(scale * density))
        per_pixel = scale / DETAIL
        snap = DETAIL * factor
        in_sight = self._in_sight(railway, origin, scale)
        seen = []
        for train in in_sight:
            for look, back in self._parts(train):
                placed = railway.position(train, back)
                if placed is not None:
                    seen.append((train, back, look, placed))
        hidden = self.world.underground_all(
            [placed[0][0] for *_, placed in seen], [placed[0][1] for *_, placed in seen]
        )
        four_lanes = blocks_of(railway.constructions).four_lanes
        found = []
        for (train, back, look, placed), under in zip(seen, hidden, strict=True):
            if not under:
                shift = self.lanes.sideways(
                    train.path, train.along - back, self.lanes.of(train, four_lanes)
                )
                found.append((placed, shift, look))
        parts = []
        if found:
            xs = np.array([placed[0][0] for placed, _, _ in found])
            ys = np.array([placed[0][1] for placed, _, _ in found])
            ax, ay = projection.to_pixels(xs, ys)
            ay = ay - self.world.elevation.lift(xs, ys)
            for k, (((_, _), (dx, dy)), shift, look) in enumerate(found):
                sx = origin[0] + round((float(ax[k]) + shift[0]) * snap) * scale / snap
                sy = origin[1] + round((float(ay[k]) + shift[1]) * snap) * scale / snap
                parts.append((sy, sx, look, math.atan2(dy, dx)))
        parts.sort(key=lambda part: part[0])
        wanted = {look for _, _, look, _ in parts}
        for look in wanted:
            self.sprites.draw(look)
        drawn: set[tuple[object, int]] = set()
        self._paging_until = time.perf_counter() + PAGES_TIME
        for sy, sx, look, heading in parts:
            page, fine = (None, factor) if factor == 1 else self._page_or_nearest(look, factor)
            if page is None:
                px, py, w, h, anchor_x, anchor_y = self.sprites.place(look, heading)
                texture, (width, height), per = (
                    self._whole_sheet(wanted),
                    self._sheet_shape,
                    per_pixel,
                )
            else:
                px, py, w, h, anchor_x, anchor_y = page.spots[turned(heading)]
                texture, (width, height), per = page.texture, page.size, per_pixel / fine
                drawn.add((look, fine))
            left = sx - anchor_x * per
            top = sy - anchor_y * per
            draw_list.add_image(
                imgui.ImTextureRef(texture.texture_id()),
                (left, top),
                (left + w * per, top + h * per),
                (px / width, py / height),
                ((px + w) / width, (py + h) / height),
            )
        self._let_go(drawn)
        self._smoke(draw_list, railway, in_sight, origin, scale)
        self.sprites.draw_ahead(self._paging_until)

    @staticmethod
    def _parts(train: Train) -> list[tuple[object, float]]:
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
            parts.append((Look.DINING if car.dining else (car.good, car.load > 0), back))
            back += CAR_HALF
        return [(look, middle * SCALE) for look, middle in parts]

    def _in_sight(self, railway: Railway, origin: tuple[int, int], scale: float) -> list[Train]:
        """The running trains whose front is close enough to the screen for some of the train
        to show: a train reaches at most its length, and the ground's lift, from its front."""
        fronts = []
        for train in railway.trains:
            placed = railway.position(train)
            if placed is not None:
                fronts.append((train, placed[0]))
        if not fronts:
            return []
        projection = self.world.projection
        ax, ay = projection.to_pixels(
            np.array([at[0] for _, at in fronts]), np.array([at[1] for _, at in fronts])
        )
        per_cell = float(np.abs(projection.matrix).sum(axis=0).max()) * scale
        size = imgui.get_io().display_size
        found = []
        for (train, _), x, y in zip(fronts, ax.tolist(), ay.tolist(), strict=True):
            reach = (train.length + 2) * per_cell + MOST_LIFT * scale
            sx, sy = origin[0] + x * scale, origin[1] + y * scale
            if -reach <= sx <= size.x + reach and -reach <= sy <= size.y + reach:
                found.append(train)
        return found

    def _smoke(
        self,
        draw_list: imgui.ImDrawList,
        railway: Railway,
        trains: list[Train],
        origin: tuple[int, int],
        scale: float,
    ) -> None:
        """Puffs of smoke rising from each chimney and drifting back over its train. The ground
        under every chimney and where every puff lies in the picture are worked out for all the
        trains at once: on hills a lift takes nearly as long for one point as for many."""
        chimneys = self._chimneys(railway, trains)
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
            shade = (
                dark + (thin - dark) * age for dark, thin in zip(SMOKE, THIN_SMOKE, strict=True)
            )
            color = imgui.color_convert_float4_to_u32(imgui.ImVec4(*shade, alpha))
            draw_list.add_circle_filled((sx, sy), radius, color)

    def _chimneys(self, railway: Railway, trains: list[Train]) -> list[_Chimney]:
        """Where each of these trains' chimney is on the map, for those that have one in the
        open."""
        seen = []
        for train in trains:
            model = ENGINES[train.engine]
            if model.chimney is None:
                continue
            back = (model.half - model.chimney[0]) * SCALE
            placed = railway.position(train, back)
            if placed is not None:
                seen.append((train, back, placed))
        hidden = self.world.underground_all(
            [placed[0][0] for *_, placed in seen], [placed[0][1] for *_, placed in seen]
        )
        four_lanes = blocks_of(railway.constructions).four_lanes
        chimneys = []
        for (train, back, ((x, y), (dx, dy))), under in zip(seen, hidden, strict=True):
            if under:
                continue
            model = ENGINES[train.engine]
            assert model.chimney is not None
            shift = self.lanes.sideways(
                train.path, train.along - back, self.lanes.of(train, four_lanes)
            )
            phase = (train.along * 6.0) % 1.0
            chimneys.append(_Chimney(x, y, dx, dy, shift, model.chimney[1] * SCALE, phase))
        return chimneys

    def _blown_up_by(self, device_scale: int) -> int:
        """How many times as fine as the sheet each way the pictures are drawn to show crisp at
        this many device pixels per art pixel: imgui samples every texture smoothly, so a
        texture must hold a texel for every device pixel. Pages for other zooms are kept."""
        return max(1, min(device_scale // DETAIL, MAX_TEXTURE // PAGE_WIDTH))

    def _page_or_nearest(self, look: object, factor: int) -> tuple[_Page | None, int]:
        """A look's page as fine as this zoom wants, or while it is drawn the page of the
        nearest zoom that has one, and how fine it is; None, with nothing drawn yet."""
        page = self._page(look, factor)
        if page is not None:
            return page, factor
        kept = [fine for kept_look, fine in self._pages if kept_look == look]
        if not kept:
            return None, factor
        fine = min(kept, key=lambda each: (abs(each - factor), -each))
        self._pages.move_to_end((look, fine))
        return self._pages[(look, fine)], fine

    def _whole_sheet(self, wanted: set[object]) -> Texture:
        """The sprite sheet at its own size, for zooms that need it no bigger, made again with
        every look drawn so far once it lacks one of those `wanted`."""
        if self._sheet is None or not wanted <= self._sheet_looks:
            image = np.ascontiguousarray(self.sprites.image[: self.sprites.height])
            self._sheet = hello_imgui.create_texture_gpu_from_rgba_data(image)
            self._sheet_shape = (image.shape[1], image.shape[0])
            self._sheet_looks = {look for look, _ in self.sprites.places}
        return self._sheet

    def _page(self, look: object, factor: int) -> _Page | None:
        """A texture of one look's pictures every way round, drawn `factor` times as fine as the
        sheet with their fine details, or None until it is drawn: a few pictures each frame,
        while the frame has not spent its PAGES_TIME. Only the looks on screen get one: the
        whole sheet that fine on a sharp screen is too big a texture for many graphics cards,
        which draw it black."""
        key = (look, factor)
        page = self._pages.get(key)
        if page is not None:
            self._pages.move_to_end(key)
            return page
        if time.perf_counter() > self._paging_until:
            return None
        if key not in self._painting:
            self._painting[key] = (FinePainter(self._matrix, look, factor), [])
        painter, pictures = self._painting[key]
        while len(pictures) < HEADINGS and time.perf_counter() <= self._paging_until:
            pictures.append(painter.picture(len(pictures)))
        if len(pictures) < HEADINGS:
            return None
        del self._painting[key]
        width = max(PAGE_WIDTH * factor, *(p.image.size[0] for p in pictures))
        spots = {}
        x = y = row = 0
        for index, picture in enumerate(pictures):
            w, h = picture.image.size
            if x + w > width:
                x, y, row = 0, y + row + 1, 0
            spots[index] = (x, y, w, h, *picture.anchor)
            x += w + 1
            row = max(row, h)
        pixels = np.zeros((y + row, width, 4), dtype=np.uint8)
        for index, picture in enumerate(pictures):
            x, y, w, h, _, _ = spots[index]
            pixels[y : y + h, x : x + w] = np.asarray(picture.image)
        texture = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(pixels))
        page = self._pages[key] = _Page(texture, (width, pixels.shape[0]), spots, pixels.nbytes)
        return page

    def _let_go(self, drawn: set[tuple[object, int]]) -> None:
        """Lets go of the pages not drawn lately, of any zoom, while they take more than
        PAGES_BUDGET."""
        held = sum(page.bytes for page in self._pages.values())
        for key in list(self._pages):
            if held <= PAGES_BUDGET:
                break
            if key not in drawn:
                held -= self._pages.pop(key).bytes


@dataclass(frozen=True)
class _Page:
    """One look's pictures every way round on a texture of their own: the texture, its size,
    each picture's x, y, width, height and anchor on it, and its bytes."""

    texture: Texture
    size: tuple[int, int]
    spots: dict[int, tuple[int, int, int, int, int, int]]
    bytes: int


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
        self._held: dict[str, tuple[list[Point] | None, dict[Cell, int]]] = {}

    def of(self, train: Train, four_lanes: Container[Cell]) -> dict[Cell | None, int]:
        """The track of quadruple track the train takes in each cell it holds (lanes_of), a
        cell keeping the one it was drawn on after quadruple track while the train holds it and
        after it lets go of it, along the same path, and the station it sets off from the one it
        stood on."""
        if len(self._held) > 1024:
            self._held.clear()
        path, kept = self._held.get(train.name, (None, {}))
        if path is not train.path:
            kept = {}
        lanes, sure = _held_lanes(train, four_lanes, kept)
        behind = {cell: lane for cell, lane in kept.items() if cell not in lanes}
        lanes |= behind
        sure |= behind
        if train.path:
            row, col = self.track.tile_map.grid.cell_at(*train.path[0])
            start = (int(row), int(col))
            if start not in lanes:
                lanes[start] = sure[start] = kept.get(start, lanes[None])
        self._held[train.name] = (train.path, sure)
        return lanes

    def sideways(
        self, path: list[Point], along: float, lanes: Mapping[Cell | None, int] | None = None
    ) -> Point:
        """How far across the picture, in art pixels, a train `along` cells into its path is
        moved off the path's centre line to ride the track drawn there: square to the way it
        goes, turning smoothly with the track round the middle of a cell."""
        stretches, starts = self._stretches(path)
        if not stretches:
            return 0.0, 0.0
        index = max(0, bisect.bisect_right(starts, along) - 1)
        stretch = stretches[index]
        if stretch.piece is None:
            return 0.0, 0.0
        t = min(1.0, max(0.0, (along - stretch.start) / (stretch.length or 1.0)))
        from_middle = (1 - t if stretch.inward else t) * stretch.length
        first, second = stretch.halves
        if stretch.at_first:
            share, forward = (first - from_middle) / (first + second), stretch.inward
        else:
            share, forward = (first + from_middle) / (first + second), not stretch.inward
        lanes = lanes or {}
        piece = stretch.piece
        lane = _lane_at(stretches, index, lanes)
        side = self.track.riding_side(piece, share, forward, lane)
        choices = self.track.riding_choices(piece, share, forward, lane)
        if choices is not None and len(choices[1]) > 1:
            met = self._met_side(stretches, index, choices[0], lanes)
            if met is not None:
                side = min(choices[1], key=lambda choice: abs(choice[1] - met))[0]
        elif piece.width == TrackWidth.QUADRUPLE:
            first, second = (
                self._end_lane(stretches, index, end, forward, lane, lanes) for end in piece.ends
            )
            if first != lane or second != lane:
                side = (1 - share) * self.track.riding_side(
                    piece, share, forward, first
                ) + share * self.track.riding_side(piece, share, forward, second)
        if side == 0.0:
            return 0.0, 0.0
        dx, dy = stretch.way
        if stretch.turn is not None and from_middle < BEND:
            mix = 0.5 * (1 - from_middle / BEND)
            dx, dy = (1 - mix) * dx + mix * stretch.turn[0], (1 - mix) * dy + mix * stretch.turn[1]
        length = math.hypot(dx, dy) or 1.0
        return -dy / length * side * PIXEL_CELL_WIDTH, dx / length * side * PIXEL_CELL_WIDTH

    def _end_lane(
        self,
        stretches: list[_Stretch],
        index: int,
        point: int,
        forward: bool,
        lane: int,
        lanes: Mapping[Cell | None, int],
    ) -> int:
        """Which track of quadruple track a train rides at the piece's end `point`: where it
        meets track there that does not branch off other, the one that track runs on from, so
        it crosses over to the one it takes on the piece rather than jumping at the edge, and
        from other quadruple track only where it comes from it; elsewhere the one it takes."""
        piece = stretches[index].piece
        assert piece is not None
        entering = point == (piece.ends[0] if forward else piece.ends[1])
        met = self._met_side(stretches, index, point, lanes, plain=True, wide=entering)
        if met is None:
            return lane
        share = 0.0 if point == piece.ends[0] else 1.0
        return min(
            (lane, 1 - lane),
            key=lambda each: abs(self.track.riding_side(piece, share, forward, each) - met),
        )

    def _met_side(
        self,
        stretches: list[_Stretch],
        index: int,
        point: int,
        lanes: Mapping[Cell | None, int],
        plain: bool = False,
        wide: bool = False,
    ) -> float | None:
        """How far to the left of its way a train rides on the track it meets the stretch's piece
        from, or goes on to, across that piece's end `point`; None where its path does not cross
        that end, or with `plain` where the track there branches off other, or is quadruple
        unless `wide`."""
        stretch = stretches[index]
        piece = stretch.piece
        assert piece is not None
        edge = piece.ends[0] if stretch.at_first else piece.ends[1]
        if edge != point:
            index = index + 1 if stretch.inward else index - 1
            if not 0 <= index < len(stretches) or stretches[index].piece != piece:
                return None
        beyond = index - 1 if stretches[index].inward else index + 1
        if not 0 <= beyond < len(stretches):
            return None
        other = stretches[beyond]
        if other.piece is None:
            return None
        share = 0.0 if other.at_first else 1.0
        forward = other.inward if other.at_first else not other.inward
        lane = _lane_at(stretches, beyond, lanes)
        if plain and (
            (other.piece.width == TrackWidth.QUADRUPLE and not wide)
            or self.track.riding_choices(other.piece, share, forward, lane) is not None
        ):
            return None
        return self.track.riding_side(other.piece, share, forward, lane)

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


def _lane_at(stretches: list[_Stretch], index: int, lanes: Mapping[Cell | None, int]) -> int:
    """The track of quadruple track a train rides on the stretch's piece: the one it holds there,
    or else the one it holds on the nearest cell behind it on its path, or ahead, so a car on a
    cell the train has let go of keeps to the track it rode there."""
    for near in itertools.chain(range(index, -1, -1), range(index + 1, len(stretches))):
        piece = stretches[near].piece
        if piece is not None and piece.cell in lanes:
            return lanes[piece.cell]
    return lanes.get(None, 0)


def lanes_of(
    train: Train, four_lanes: Container[Cell], kept: Mapping[Cell, int] | None = None
) -> dict[Cell | None, int]:
    """The track of quadruple track the train takes in each cell it holds (railroads.sim.signals):
    where trains going one way keep to two tracks the one it holds, and through junctions and
    where quadruple track ends the one it held last before, or else the one `kept` there, or
    else the one it holds next; and under None the one it keeps to through stations, which no
    train holds: the one it last took."""
    return _held_lanes(train, four_lanes, kept or {})[0]


def _held_lanes(
    train: Train, four_lanes: Container[Cell], kept: Mapping[Cell, int]
) -> tuple[dict[Cell | None, int], dict[Cell, int]]:
    """lanes_of, and those of its cells that come from a track of quadruple track the train
    holds there or before them, or that it has come onto, which keep theirs."""
    lanes: dict[Cell | None, int] = {}
    sure: dict[Cell, int] = {}
    carried = None
    before: list[Cell] = []
    for claim in sorted(train.claims, key=lambda claim: claim[3]):
        cell, lane = claim[0], claim[6]
        if cell in four_lanes:
            carried = lane
            lanes |= {each: kept.get(each, lane) for each in before}
            before = []
            lanes[cell] = sure[cell] = lane
        elif carried is None:
            before.append(cell)
        else:
            lanes[cell] = sure[cell] = carried
    lanes |= {each: kept.get(each, train.lane) for each in before}
    entered = {claim[0] for claim in train.claims if claim[3] <= train.along}
    sure |= {cell: lanes[cell] for cell in entered if cell in lanes and cell not in sure}
    lanes[None] = train.lane if carried is None else carried
    return lanes, sure


def _point(steps: np.ndarray, dx: float, dy: float) -> int:
    """The track point this far from a cell's middle lies at."""
    return int(np.argmin(np.hypot(steps[:, 0] / 2 - dx, steps[:, 1] / 2 - dy)))


# Which way on the screen an engine shown on its own faces: to the right and a little down, so
# its side and a bit of its front show.
SHOWN_FACING = (0.95, 0.3)


def engine_picture(
    sprites: TrainSprites | GrownTrainSprites,
    matrix: npt.NDArray[np.float64],
    engine: str,
    times: int = 1,
) -> Pixels:
    """An engine with its tender, as the map draws it, facing SHOWN_FACING: RGBA at the size of
    the sprite sheet, or `times` as many pixels each way with its fine details."""
    return train_picture(sprites, matrix, Train("", engine=engine), times)


def train_picture(
    sprites: TrainSprites | GrownTrainSprites,
    matrix: npt.NDArray[np.float64],
    train: Train,
    times: int = 1,
    shown_facing: tuple[float, float] = SHOWN_FACING,
) -> Pixels:
    """A whole train, its engine, tender and cars, as the map draws it, facing `shown_facing` on
    the screen: RGBA at the size of the sprite sheet, or `times` as many pixels each way with its
    fine details."""
    facing = np.array(shown_facing) / math.hypot(*shown_facing)

    def on_screen(heading: float) -> float:
        way = matrix @ (math.cos(heading), math.sin(heading))
        return float(way @ facing / (np.hypot(*way) or 1.0))

    heading = max((2 * math.pi * k / HEADINGS for k in range(HEADINGS)), key=on_screen)
    ahead = matrix @ (math.cos(heading), math.sin(heading)) * DETAIL * times
    key = tuple(float(v) for v in matrix.ravel())
    placed = []
    for look, back in TrainLayer._parts(train):
        if times == 1:
            x, y, w, h, anchor_x, anchor_y = sprites.place(look, heading)
            pixels = sprites.image[y : y + h, x : x + w]
        else:
            (picture,) = fine_pictures(key, look, times, [turned(heading)])
            pixels = np.asarray(picture.image)
            anchor_x, anchor_y = picture.anchor
        left = -ahead[0] * back - anchor_x
        top = -ahead[1] * back - anchor_y
        placed.append((-ahead[1] * back, top, left, pixels))
    low_x = math.floor(min(left for _, _, left, _ in placed))
    low_y = math.floor(min(top for _, top, _, _ in placed))
    high_x = math.ceil(max(left + p.shape[1] for _, _, left, p in placed))
    high_y = math.ceil(max(top + p.shape[0] for _, top, _, p in placed))
    canvas = Image.new("RGBA", (high_x - low_x, high_y - low_y), (0, 0, 0, 0))
    for _, top, left, pixels in sorted(placed, key=lambda part: part[0]):
        part = Image.fromarray(np.ascontiguousarray(pixels), "RGBA")
        spot = (round(left) - low_x, round(top) - low_y)
        canvas.alpha_composite(part, spot)
    return np.asarray(canvas).copy()
