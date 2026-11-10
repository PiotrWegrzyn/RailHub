"""Track drawn into the pixel map: a gravel bed with wooden sleepers and two steel rails, a timber deck with railings over water, dark portals where it goes into mountains, and masts holding wires over electric track."""

import itertools
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.tilemap import Relief, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH
from railroads.ui.pixel.projection import RGB, FloatArray, Path, PixelProjection, Point, ToArt

# Track, in cells: how wide the gravel bed and a bridge deck are, the gap between the rails, how
# long and wide a sleeper is and how far apart sleepers lie, and a tunnel portal's width. The two
# tracks of double track lie DOUBLE_SPACING apart, and every piece rounds its turn over the last
# BEND either side of the middle of its cell.
TRACK_BED = 0.3


BRIDGE_WIDTH = 0.32


GAUGE = 0.15


SLEEPER_LENGTH = 0.25


SLEEPER_WIDTH = 0.045


SLEEPER_GAP = 0.125


PORTAL = 0.32


DOUBLE_SPACING = 0.28


BEND = 0.2


# Overhead wires on electric track: how far beside the track their masts stand and how far
# apart, in cells, and how tall the masts are and how high the wires hang, in art pixels.
MAST_SIDE = 0.16


MAST_GAP = 1.0


MAST_HEIGHT = 7


WIRE_HEIGHT = 5


BALLAST: RGB = (118, 110, 98)


SLEEPER: RGB = (84, 60, 40)


RAIL: RGB = (178, 176, 172)


BRIDGE_DECK: RGB = (126, 86, 50)


BRIDGE_SHADOW: RGB = (62, 42, 26)


RAILING: RGB = (70, 48, 30)


PORTAL_STONE: RGB = (118, 112, 104)


PORTAL_DARK: RGB = (20, 18, 18)


MAST: RGB = (96, 96, 104)


WIRE: RGB = (40, 40, 46)


def _band(
    start: tuple[float, float], end: tuple[float, float], half_width: float, overlap: bool = True
) -> list[tuple[float, float]]:
    """The corners of a strip from `start` to `end`, `half_width` either side; with `overlap`, a
    little longer at both ends so strips meet without gaps."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy * half_width, ux * half_width
    grow = half_width * 0.5 if overlap else 0.0
    sx, sy = start[0] - ux * grow, start[1] - uy * grow
    ex, ey = end[0] + ux * grow, end[1] + uy * grow
    return [(sx + nx, sy + ny), (ex + nx, ey + ny), (ex - nx, ey - ny), (sx - nx, sy - ny)]


def _along(
    middle: tuple[float, float], start: tuple[float, float], end: tuple[float, float], half: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """A short stretch of the line from `start` to `end`, `half` either side of `middle`, turned
    across it: the line a sleeper lies along."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    return (middle[0] - ux * half, middle[1] - uy * half), (
        middle[0] + ux * half,
        middle[1] + uy * half,
    )


def _offset(path: Path, side: float | list[float]) -> Path:
    """The path moved `side` to its left, by one amount or by one for each of its points,
    joined at each bend where the moved lines meet."""
    sides = side if isinstance(side, list) else [side] * len(path)
    normals = []
    for (x0, y0), (x1, y1) in itertools.pairwise(path):
        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        normals.append((-(y1 - y0) / length, (x1 - x0) / length))
    moved = []
    for index, ((x, y), amount) in enumerate(zip(path, sides, strict=True)):
        before = normals[max(0, index - 1)]
        after = normals[min(len(normals) - 1, index)]
        mx, my = before[0] + after[0], before[1] + after[1]
        dot = (mx * before[0] + my * before[1]) or 1.0
        moved.append((x + mx / dot * amount, y + my / dot * amount))
    return moved


def _bend(start: Point, middle: Point, end: Point) -> Path:
    """The line from `start` through `middle` to `end`, rounded off over the last BEND before
    and after the middle instead of turning sharply there."""
    (ax, ay), (cx, cy), (bx, by) = start, middle, end
    la, lb = math.hypot(ax - cx, ay - cy), math.hypot(bx - cx, by - cy)
    ua, ub = ((ax - cx) / la, (ay - cy) / la), ((bx - cx) / lb, (by - cy) / lb)
    if ua[0] * ub[0] + ua[1] * ub[1] < -0.999:
        return [start, middle, end]
    p = (cx + ua[0] * BEND, cy + ua[1] * BEND)
    q = (cx + ub[0] * BEND, cy + ub[1] * BEND)
    curve = [
        (
            (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * cx + t**2 * q[0],
            (1 - t) ** 2 * p[1] + 2 * (1 - t) * t * cy + t**2 * q[1],
        )
        for t in (0.25, 0.5, 0.75)
    ]
    return [start, p, *curve, q, end]


def _lengths(path: Path) -> list[float]:
    """How far along the path each of its points lies."""
    return [0.0, *itertools.accumulate(math.dist(a, b) for a, b in itertools.pairwise(path))]


def _point_at(path: Path, lengths: list[float], along: float) -> tuple[Point, Point]:
    """The point `along` the path and the way the path runs there, as a unit vector."""
    for (a, b), (start, stop) in zip(
        itertools.pairwise(path), itertools.pairwise(lengths), strict=True
    ):
        if along <= stop or b == path[-1]:
            span = (stop - start) or 1.0
            t = min(1.0, max(0.0, (along - start) / span))
            point = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            return point, ((b[0] - a[0]) / span, (b[1] - a[1]) / span)
    raise ValueError("A path needs two points")


def _sleepers(lane: Path, art_length: float) -> list[Path]:
    """The sleepers under one track, evenly along it, each lying across the way it runs. They
    are spaced by the track's length in the picture, `art_length` pixels, so they stay apart
    where the view foreshortens it."""
    lengths = _lengths(lane)
    count = max(1, round(art_length / (SLEEPER_GAP * PIXEL_CELL_WIDTH)))
    sleepers = []
    for k in range(count):
        (x, y), (ux, uy) = _point_at(lane, lengths, (k + 0.5) * lengths[-1] / count)
        half = SLEEPER_WIDTH / 2
        start, end = (x - ux * half, y - uy * half), (x + ux * half, y + uy * half)
        sleepers.append(_band(start, end, SLEEPER_LENGTH / 2, overlap=False))
    return sleepers


@dataclass(frozen=True)
class _Laid:
    """A piece of track as it is drawn: the middle of its cell, its centre line through the
    cell, and the middle of each of its tracks."""

    piece: Track
    middle: Point
    line: Path
    lanes: list[Path]


class TrackArt:
    """Draws the track of one map's constructions in the pixel map's projection."""

    def __init__(
        self, tile_map: TileMap, constructions: Constructions, projection: PixelProjection
    ) -> None:
        self.tile_map = tile_map
        self.constructions = constructions
        self.projection = projection

    def layer(
        self,
        pieces: list[Track],
        left: int,
        top: int,
        width: int,
        height: int,
        resolution: int = 1,
    ) -> FloatArray:
        """These pieces of track as RGBA pixels from 0 to 1, over the art pixels from `left`,
        `top` across `width` by `height`, `resolution` pixels for each art pixel: a gravel bed with wooden sleepers and two steel rails
        on the ground, a timber deck with railings over water, and dark portals where it goes
        into mountains. It is drawn layer by layer, beds, then sleepers, then rails, so the
        pieces meeting in a cell make one junction or crossing. Double track is two tracks side
        by side that close up where they run into single track, and electric track has masts
        beside it holding wires over each track."""
        grid = self.tile_map.grid
        steps = grid.track_steps() / 2
        reach = PIXEL_CELL_WIDTH * 2
        k = resolution
        size = ((width + 2 * reach) * k, (height + 2 * reach) * k)
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        paint = ImageDraw.Draw(layer)
        projection = self.projection
        shift = (left - reach, top - reach)

        def art(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
            xs, ys = projection.to_pixels([p[0] for p in points], [p[1] for p in points])
            return [
                ((float(x) - shift[0]) * k, (float(y) - shift[1]) * k)
                for x, y in zip(xs, ys, strict=True)
            ]

        laid: list[_Laid] = []
        for piece in pieces:
            cx, cy = grid.center(*piece.cell)
            px, py = projection.to_pixels(cx, cy)
            if not (
                left - reach < px < left + width + reach and top - reach < py < top + height + reach
            ):
                continue
            a, b = (steps[point] for point in piece.ends)
            middle = (float(cx), float(cy))
            line = _bend((cx + a[0], cy + a[1]), middle, (cx + b[0], cy + b[1]))
            laid.append(_Laid(piece, middle, line, self._lanes(piece, line)))
        self._lay_bridges(paint, art, [each for each in laid if each.piece.bed == TrackBed.BRIDGE])
        for each in laid:
            if each.piece.bed == TrackBed.GROUND:
                for lane in each.lanes:
                    for segment in itertools.pairwise(lane):
                        paint.polygon(art(_band(*segment, TRACK_BED / 2)), fill=(*BALLAST, 255))
        for each in laid:
            if each.piece.bed == TrackBed.TUNNEL:
                self._lay_portals(paint, art, each)
        laid = [each for each in laid if each.piece.bed != TrackBed.TUNNEL]
        for each in laid:
            for lane in each.lanes:
                for sleeper in _sleepers(lane, _lengths(art(lane))[-1] / k):
                    paint.polygon(art(sleeper), fill=(*SLEEPER, 255))
        for each in laid:
            for lane in each.lanes:
                for side in (-GAUGE / 2, GAUGE / 2):
                    paint.line(art(_offset(lane, side)), fill=(*RAIL, 255), width=1)
        for each in laid:
            if each.piece.electric:
                self._hang_wires(paint, art, each, k)
        pixels = np.asarray(layer, dtype=np.float32) / 255
        return pixels[reach * k : (reach + height) * k, reach * k : (reach + width) * k]

    def _lanes(self, piece: Track, line: Path) -> list[Path]:
        """The middle of each track of the piece. Double track has one either side of its
        centre line, closing up to it at an end that runs into single track. Single track keeps
        to its centre line, except where it branches off double track in the same cell: there
        it runs on along the outer track on the side it turns to, and closes in to its centre
        line by its other end."""
        lengths = _lengths(line)
        half_length = lengths[-1] / 2
        if piece.width == TrackWidth.SINGLE:
            shifts = [self._branch_shift(piece, point) for point in piece.ends]
            if not any(shifts):
                return [line]
            widths = [
                shifts[0] * min(1.0, (lengths[-1] - along) / half_length)
                + shifts[1] * min(1.0, along / half_length)
                for along in lengths
            ]
            return [_offset(line, widths)]
        wide_start, wide_end = (self._double_beyond(piece, point) for point in piece.ends)
        widths = []
        for along in lengths:
            share = 1.0
            if not wide_start:
                share = min(share, along / half_length)
            if not wide_end:
                share = min(share, (lengths[-1] - along) / half_length)
            widths.append(DOUBLE_SPACING / 2 * share)
        return [_offset(line, [-width for width in widths]), _offset(line, widths)]

    def _branch_shift(self, piece: Track, point: int) -> float:
        """How far to the left of its centre line single track starts at this end, to leave
        from the outer track of full double track it branches off there: half the spacing to
        whichever side it turns away to, or nothing."""
        steps = self.tile_map.grid.track_steps()
        mine = piece.ends[1] if piece.ends[0] == point else piece.ends[0]
        for other in self.constructions.track_at(piece.cell):
            if other.width != TrackWidth.DOUBLE or point not in other.ends:
                continue
            if not self._double_beyond(other, point):
                continue
            theirs = other.ends[1] if other.ends[0] == point else other.ends[0]
            ux, uy = -steps[point]
            turn = sum(
                sign * (ux * steps[end][1] - uy * steps[end][0])
                for sign, end in ((1, mine), (-1, theirs))
            )
            side = math.copysign(DOUBLE_SPACING / 2, turn)
            return side if point == piece.ends[0] else -side
        return 0.0

    def _double_beyond(self, piece: Track, point: int) -> bool:
        """Whether track leaving the piece by this point runs on as double track, or into a
        station or nowhere, rather than into single track."""
        grid = self.tile_map.grid
        beyond, entry = grid.track_exit(piece.cell, point)
        if not grid.contains(*beyond):
            return True
        joined = [other for other in self.constructions.track_at(beyond) if entry in other.ends]
        return not joined or any(other.width == TrackWidth.DOUBLE for other in joined)

    def _lay_bridges(self, paint: ImageDraw.ImageDraw, art: ToArt, laid: list[_Laid]) -> None:
        for half, color in (
            (BRIDGE_WIDTH / 2 + 0.03, BRIDGE_SHADOW),
            (BRIDGE_WIDTH / 2, BRIDGE_DECK),
        ):
            for each in laid:
                for lane in each.lanes:
                    for segment in itertools.pairwise(lane):
                        paint.polygon(art(_band(*segment, half)), fill=(*color, 255))
        for each in laid:
            outer = each.lanes[0], each.lanes[-1]
            for lane, side in zip(outer, (-BRIDGE_WIDTH / 2, BRIDGE_WIDTH / 2), strict=True):
                paint.line(art(_offset(lane, side)), fill=(*RAILING, 255), width=1)

    def _lay_portals(self, paint: ImageDraw.ImageDraw, art: ToArt, laid: _Laid) -> None:
        """A dark arch where the tunnel meets track out in the open."""
        grid = self.tile_map.grid
        piece, middle = laid.piece, laid.middle
        half = PORTAL / 2 + (DOUBLE_SPACING / 2 if piece.width == TrackWidth.DOUBLE else 0.0)
        for point, end in zip(piece.ends, (laid.line[0], laid.line[-1]), strict=True):
            beyond, _ = grid.track_exit(piece.cell, point)
            if not grid.contains(*beyond) or self.tile_map.relief[beyond] == Relief.MOUNTAINS:
                continue
            inner = (
                middle[0] + (end[0] - middle[0]) * 0.6,
                middle[1] + (end[1] - middle[1]) * 0.6,
            )
            paint.polygon(art(_band(inner, end, half + 0.03)), fill=(*PORTAL_STONE, 255))
            paint.polygon(art(_band(inner, end, half)), fill=(*PORTAL_DARK, 255))

    def _hang_wires(
        self, paint: ImageDraw.ImageDraw, art: ToArt, laid: _Laid, resolution: int = 1
    ) -> None:
        """A wire over each track, and masts at even steps along the piece on the far side of
        it, each with an arm reaching over the track to hold the wires."""
        (_, _), (c, d) = self.projection.matrix
        line, lanes = laid.line, laid.lanes
        lengths = _lengths(line)
        count = max(1, round(lengths[-1] / MAST_GAP))
        for lane in lanes:
            wire = [(x, y - WIRE_HEIGHT * resolution) for x, y in art(lane)]
            paint.line(wire, fill=(*WIRE, 255), width=1)
        for k in range(count):
            along = (k + 0.5) * lengths[-1] / count
            _, (ux, uy) = _point_at(line, lengths, along)
            far = (
                (-uy, ux)
                if c * -uy + d * ux < 0 or (c * -uy + d * ux == 0 and ux < 0)
                else (uy, -ux)
            )
            overs = []
            for lane in lanes:
                lane_lengths = _lengths(lane)
                over, _ = _point_at(lane, lane_lengths, along * lane_lengths[-1] / lengths[-1])
                overs.append(over)
            overs.sort(key=lambda over: over[0] * far[0] + over[1] * far[1])
            near, back = overs[0], overs[-1]
            base = (back[0] + far[0] * MAST_SIDE, back[1] + far[1] * MAST_SIDE)
            (bx, by), (wx, wy) = art([base, near])
            top = (bx, by - MAST_HEIGHT * resolution)
            paint.line([(bx, by), top], fill=(*MAST, 255), width=1)
            paint.line([top, (wx, wy - WIRE_HEIGHT * resolution)], fill=(*MAST, 255), width=1)
