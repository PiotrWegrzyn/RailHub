"""Track drawn into the pixel map, built upright on the screen so it keeps one width and look
whichever way it runs: a dark bank round a gravel bed, dark sleepers lying square to the rails
and two light rails, the same on both sides, with a faint shadow round it. Over water it runs on
a timber deck with railings, where it goes into mountains a dark portal opens, and over electric
track masts hold wires. Through a station it runs straight between the station's two ends, past
a platform on the station's side.

Each layer is drawn several times finer than it is shown and averaged down, so its edges and
rails come out smooth instead of stepped. Where hills and mountains rise, track is drawn lifted
with the ground point by point, so it climbs smoothly, and the lift of every pixel it covers is
kept, so the map can hide it behind raised ground in front of it (railroads.ui.pixel.world)."""

import itertools
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.grids import Cell
from railroads.sim.tilemap import Relief, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH
from railroads.ui.pixel.art.painter import PLATFORM, PLATFORM_EDGE
from railroads.ui.pixel.projection import RGB, FloatArray, Path, PixelProjection, Point
from railroads.ui.pixel.relief import MOST_LIFT, Elevation

# Track, in cells of the picture's width, the same whichever way it runs: how wide the dark bank
# and the gravel bed on it are, the gap between the rails, how long and wide a sleeper is and how
# far apart sleepers lie, a bridge deck's and a tunnel portal's width, and how far beyond the
# bank its shadow reaches. The two tracks of double track lie DOUBLE_SPACING apart, and every
# piece rounds its turn over the last BEND either side of the middle of its cell.
BANK = 0.175
TRACK_BED = 0.15
GAUGE = 0.072
SLEEPER_LENGTH = 0.14
SLEEPER_WIDTH = 0.026
SLEEPER_GAP = 0.05
BRIDGE_WIDTH = 0.24
PORTAL = 0.24
SHADOW_REACH = 0.03
DOUBLE_SPACING = 0.22
BEND = 0.2
# How dark the shadow round the track is, from 0 to 1.
SHADOW = 0.35
# How far apart track is followed over the ground to lift it, in cells.
STEP = 1 / 12
# How many times finer than shown, each way, track is drawn before it is averaged down, by the
# resolution it is shown at: enough to smooth it, without drawing a huge picture up close.
SMOOTHING = {1: 4, 2: 2}
# How many pieces of track are kept laid out, with where they land in the picture and how far
# they are lifted, before all are forgotten and laid out again as they are drawn: more than the
# chunks round a view of a busy railway draw, and few enough that a big railway's pieces do not
# fill memory.
PIECES_KEPT = 1 << 11
# A station's platform beside its track, and how far from the track's middle the station house
# stands, in cells of the picture's width.
PLATFORM_WIDTH = 0.12
SET_BACK = 0.34

# Overhead wires on electric track: how far beside the track their masts stand and how far
# apart, in cells, and how tall the masts are and how high the wires hang, in art pixels.
MAST_SIDE = 0.13
MAST_GAP = 1.0
MAST_HEIGHT = 6
WIRE_HEIGHT = 4

BANK_EDGE: RGB = (58, 52, 44)
GRAVEL: RGB = (98, 90, 80)
SLEEPER: RGB = (46, 30, 20)
RAIL: RGB = (238, 236, 230)
RAIL_UNDER: RGB = (24, 22, 20)
BRIDGE_DECK: RGB = (126, 86, 50)
BRIDGE_SHADOW: RGB = (62, 42, 26)
RAILING: RGB = (70, 48, 30)
PORTAL_STONE: RGB = (118, 112, 104)
PORTAL_DARK: RGB = (20, 18, 18)
MAST: RGB = (96, 96, 104)
WIRE: RGB = (40, 40, 46)


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


def _strip(path: Path, left: float, right: float | None = None) -> Path:
    """The outline of a strip along the path, from `left` to its left out to `right` (by
    default as far to its right)."""
    right = -left if right is None else right
    return _offset(path, left) + _offset(path, right)[::-1]


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
        for t in (0.2, 0.4, 0.6, 0.8)
    ]
    return [start, p, *curve, q, end]


def _dense(path: Path, step: float) -> Path:
    """The path with points put in along each stretch, no more than `step` apart."""
    dense = [path[0]]
    for (x0, y0), (x1, y1) in itertools.pairwise(path):
        count = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / step))
        dense += [
            (x0 + (x1 - x0) * t / count, y0 + (y1 - y0) * t / count) for t in range(1, count + 1)
        ]
    return dense


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


def _sleepers(lane: Path, cell: float) -> list[Path]:
    """The sleepers under one track on the screen, evenly along it, each lying square to the
    way it runs there; `cell` is a cell's width in the picture's pixels."""
    lengths = _lengths(lane)
    count = max(1, round(lengths[-1] / (SLEEPER_GAP * cell)))
    half_width, half_length = SLEEPER_WIDTH * cell / 2, SLEEPER_LENGTH * cell / 2
    sleepers = []
    for k in range(count):
        (x, y), (ux, uy) = _point_at(lane, lengths, (k + 0.5) * lengths[-1] / count)
        along, across = (ux * half_width, uy * half_width), (-uy * half_length, ux * half_length)
        sleepers.append(
            [
                (x - along[0] + across[0], y - along[1] + across[1]),
                (x + along[0] + across[0], y + along[1] + across[1]),
                (x + along[0] - across[0], y + along[1] - across[1]),
                (x - along[0] - across[0], y - along[1] - across[1]),
            ]
        )
    return sleepers


def station_side(projection: PixelProjection, steps: FloatArray, run: tuple[int, int]) -> Point:
    """Which way, as a unit vector across the picture, a station's house stands from its main
    run of track: away from the viewer, or to the left of track running up and down."""
    (ax, ay), (bx, by) = (projection.matrix @ steps[point] for point in run)
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    if abs(ny) < 1e-6:
        return (-1.0, 0.0)
    return (nx, ny) if ny < 0 else (-nx, -ny)


def station_line(constructions: Constructions, cell: Cell) -> str:
    """Which way the track through a station runs on the screen, as the station house is drawn
    turned to it (railroads.ui.pixel.art.stations.TURNED_LINES): across, falling to the right,
    rising to the right or up and down."""
    grid = constructions.tile_map.grid
    steps = grid.track_steps()
    a, b = constructions.station_ends(cell)
    dx, dy = steps[b] - steps[a]
    angle = math.degrees(math.atan2(dy, dx)) % 180
    lines = {"across": 0.0, "falling": 60.0, "upright": 90.0, "rising": 120.0}
    return min(
        lines, key=lambda line: min(abs(angle - lines[line]), 180 - abs(angle - lines[line]))
    )


def station_house(constructions: Constructions, projection: PixelProjection, cell: Cell) -> Point:
    """How far across the picture, in art pixels, a station's house stands from the middle of
    its cell: set back from the track through it, clear of both tracks where that is double."""
    grid = constructions.tile_map.grid
    steps = grid.track_steps()
    run = constructions.station_ends(cell)
    side = station_side(projection, steps, run)
    double = any(
        piece.width == TrackWidth.DOUBLE
        for point in run
        for piece in _joined(constructions, cell, point)
    )
    cells = SET_BACK + (DOUBLE_SPACING / 2 if double else 0.0)
    width = float(np.hypot(*(projection.matrix @ steps[0])))
    return side[0] * cells * width, side[1] * cells * width


def _joined(constructions: Constructions, cell: Cell, point: int) -> list[Track]:
    """The track that track leaving the cell by this point runs onto."""
    grid = constructions.tile_map.grid
    beyond, entry = grid.track_exit(cell, point)
    if not grid.contains(*beyond):
        return []
    return [piece for piece in constructions.track_at(beyond) if entry in piece.ends]


@dataclass(frozen=True, eq=False)
class _Laid:
    """A piece of track as it is drawn: the piece, its centre line through its cell in the
    plane as a row of x and a row of y, how far to its left each of its tracks lies at each
    point of that line, in cells, and the side its station's platform is on if it runs through
    a station."""

    piece: Track
    line: FloatArray
    sides: list[list[float]]
    platform: Point | None = None


class TrackArt:
    """Draws the track of one map's constructions in the pixel map's projection."""

    def __init__(
        self,
        tile_map: TileMap,
        constructions: Constructions,
        projection: PixelProjection,
        elevation: Elevation | None = None,
        planned: tuple[Track, ...] = (),
    ) -> None:
        self.tile_map = tile_map
        self.constructions = constructions
        self.projection = projection
        self.elevation = elevation
        # How each piece of track opens up or shifts at its ends (`_ends`), each piece as it is
        # drawn (`_laid`), and where the points of its line land in the picture and how far they
        # are lifted, kept until track or stations change, and the last three for no more than
        # PIECES_KEPT pieces.
        self._shapes: dict[Track, tuple[float, float]] = {}
        self._laid_pieces: dict[tuple[Track, Point | None], tuple[int, _Laid]] = {}
        self._placed: dict[tuple[Track, Point | None], FloatArray] = {}
        self._lifts: dict[tuple[Track, Point | None], FloatArray] = {}
        self._shapes_version = -1
        # The middle of each cell in the picture, and the steps to the cells track leaves
        # each cell for, which never change.
        self._centres: dict[Cell, tuple[float, float]] = {}
        self._steps = tile_map.grid.track_steps()
        self._planned: dict[Cell, list[Track]] = {}
        for piece in planned:
            self._planned.setdefault(piece.cell, []).append(piece)

    def _keep_current(self) -> None:
        """Forgets what was kept about pieces of track before track or stations last changed."""
        if self._shapes_version != self.constructions.railway_version:
            self._shapes.clear()
            self._forget_laid()
            self._shapes_version = self.constructions.railway_version

    def _forget_laid(self) -> None:
        """Forgets the pieces laid out, where they land in the picture and how far they lift."""
        self._laid_pieces.clear()
        self._placed.clear()
        self._lifts.clear()

    def planning(self, pieces: list[Track]) -> TrackArt:
        """Track art that draws track as it will be once these pieces are laid, each in place
        of built track between the same two points of its cell, so planned double track widens
        and narrows by the planned track round it rather than by what is built there now."""
        return TrackArt(
            self.tile_map, self.constructions, self.projection, self.elevation, tuple(pieces)
        )

    def _track_at(self, cell: Cell) -> list[Track]:
        built = self.constructions.track_at(cell)
        planned = self._planned.get(cell)
        if not planned:
            return list(built)
        ends = {piece.ends for piece in planned}
        return [piece for piece in built if piece.ends not in ends] + planned

    def layer(
        self,
        pieces: list[Track],
        left: int,
        top: int,
        width: int,
        height: int,
        resolution: int = 1,
        stations: bool = True,
    ) -> FloatArray:
        """These pieces of track, and with `stations` the track through stations, as RGBA
        pixels from 0 to 1 over the art pixels from `left`, `top` across `width` by `height`,
        `resolution` pixels for each art pixel, lying flat on the picture of the ground. It is
        drawn layer by layer, banks, then beds, then sleepers, then rails, so the pieces meeting
        in a cell make one junction or crossing. Double track is two tracks side by side that
        close up where they run into single track."""
        return self._draw(pieces, left, top, width, height, resolution, stations, False)[0]

    def lifted(
        self,
        pieces: list[Track],
        left: int,
        top: int,
        width: int,
        height: int,
        resolution: int = 1,
        stations: bool = True,
    ) -> tuple[FloatArray, FloatArray]:
        """The same track as `layer`, each point of it lifted with the ground there, and how
        many art pixels the track in each pixel is lifted."""
        return self._draw(pieces, left, top, width, height, resolution, stations, True)

    def _draw(
        self,
        pieces: list[Track],
        left: int,
        top: int,
        width: int,
        height: int,
        resolution: int,
        stations: bool,
        lifted: bool,
    ) -> tuple[FloatArray, FloatArray]:
        k = resolution
        smoothing = SMOOTHING.get(k, 1)
        s = k * smoothing
        reach = PIXEL_CELL_WIDTH * 2
        below = MOST_LIFT if lifted else 0
        near = (left - reach, top - reach, left + width + reach, top + height + reach + below)
        laid = [self._laid(piece) for piece in pieces if self._near(piece.cell, near)]
        if stations:
            laid += self._through_stations(near)
        if not laid:
            empty = np.zeros((height * k, width * k, 4), dtype=np.float32)
            return empty, np.zeros((height * k, width * k), dtype=np.float32)
        size = (width * s, height * s)
        picture = Image.new("RGBA", size, (0, 0, 0, 0))
        shadow = Image.new("L", size, 0)
        heights = Image.new("F", size, 0.0)
        paint, shade, measure = (
            ImageDraw.Draw(picture),
            ImageDraw.Draw(shadow),
            ImageDraw.Draw(heights),
        )
        cell = PIXEL_CELL_WIDTH * s
        raise_up = lifted and self.elevation is not None

        def screen(each: _Laid) -> tuple[Path, list[float]]:
            xs, ys = self._placed_line(each)
            lifts = self._lifted_line(each) if raise_up else [0.0] * len(xs)
            on_screen = [
                ((x - left) * s, (y - lift - top) * s)
                for x, y, lift in zip(xs, ys, lifts, strict=True)
            ]
            return on_screen, lifts

        drawn = []
        for each in laid:
            line, lifts = screen(each)
            lanes = [_offset(line, [side * cell for side in sides]) for sides in each.sides]
            drawn.append((each, line, lanes))
            if raise_up:
                self._measure(measure, each, line, lanes, lifts, cell, s)
        ground = [item for item in drawn if item[0].piece.bed != TrackBed.TUNNEL]
        for each, _, lanes in ground:
            half = (BRIDGE_WIDTH if each.piece.bed == TrackBed.BRIDGE else BANK) / 2
            for lane in lanes:
                shade.polygon(_strip(lane, (half + SHADOW_REACH) * cell), fill=255)
        for each, _, lanes in drawn:
            if each.piece.bed == TrackBed.BRIDGE:
                self._lay_deck(paint, lanes, cell, s)
        for each, _, lanes in ground:
            if each.piece.bed == TrackBed.GROUND:
                for lane in lanes:
                    paint.polygon(_strip(lane, BANK / 2 * cell), fill=(*BANK_EDGE, 255))
        for each, line, lanes in ground:
            if each.platform is not None:
                self._lay_platform(paint, line, lanes, each.platform, cell, s)
        for each, _, lanes in ground:
            if each.piece.bed == TrackBed.GROUND:
                for lane in lanes:
                    paint.polygon(_strip(lane, TRACK_BED / 2 * cell), fill=(*GRAVEL, 255))
        for each, line, lanes in drawn:
            if each.piece.bed == TrackBed.TUNNEL:
                self._lay_portals(paint, each, line, lanes, cell)
        for _, _, lanes in ground:
            for lane in lanes:
                for sleeper in _sleepers(lane, cell):
                    paint.polygon(sleeper, fill=(*SLEEPER, 255))
        rail = smoothing
        for _, _, lanes in ground:
            for lane in lanes:
                for side in (-GAUGE / 2 * cell, GAUGE / 2 * cell):
                    rail_line = _offset(lane, side)
                    paint.line(rail_line, fill=(*RAIL_UNDER, 255), width=2 * rail, joint="curve")
        for _, _, lanes in ground:
            for lane in lanes:
                for side in (-GAUGE / 2 * cell, GAUGE / 2 * cell):
                    paint.line(_offset(lane, side), fill=(*RAIL, 255), width=rail, joint="curve")
        for each, line, lanes in ground:
            if each.piece.electric:
                self._hang_wires(paint, line, lanes, cell, s)
        return _averaged(picture, shadow, smoothing), _highest(heights, smoothing)

    def _measure(
        self,
        measure: ImageDraw.ImageDraw,
        laid: _Laid,
        line: Path,
        lanes: list[Path],
        lifts: list[float],
        cell: float,
        s: int,
    ) -> None:
        """Marks how far each stretch of the piece is lifted over every pixel it can cover,
        masts and wires too."""
        half = (max(BANK, BRIDGE_WIDTH, PORTAL) / 2 + _lane_spread(lanes) + SHADOW_REACH) * cell
        rises = (0, (MAST_HEIGHT + 1) * s) if laid.piece.electric else (0,)
        if laid.piece.electric:
            half += MAST_SIDE * cell
        for (a, b), (la, lb) in zip(
            itertools.pairwise(line), itertools.pairwise(lifts), strict=True
        ):
            for rise in rises:
                stretch = _strip([(a[0], a[1] - rise), (b[0], b[1] - rise)], half)
                measure.polygon(stretch, fill=(la + lb) / 2)

    def _near(self, cell: Cell, near: tuple[int, int, int, int]) -> bool:
        centre = self._centres.get(cell)
        if centre is None:
            px, py = self.projection.to_pixels(*self.tile_map.grid.center(*cell))
            centre = self._centres[cell] = (float(px), float(py))
        return near[0] < centre[0] < near[2] and near[1] < centre[1] < near[3]

    def _laid(self, piece: Track, platform: Point | None = None) -> _Laid:
        self._keep_current()
        # Each piece is kept with the railway it was laid out for, so one laid out while the
        # railway changed, as the painter's thread can, is laid out again.
        version = self._shapes_version
        kept = self._laid_pieces.get((piece, platform))
        if kept is not None and kept[0] == version:
            return kept[1]
        grid = self.tile_map.grid
        steps = self._steps / 2
        cx, cy = (float(v) for v in grid.center(*piece.cell))
        a, b = (steps[point] for point in piece.ends)
        line = _dense(_bend((cx + a[0], cy + a[1]), (cx, cy), (cx + b[0], cy + b[1])), STEP)
        plane = np.array([[x for x, _ in line], [y for _, y in line]])
        laid = _Laid(piece, plane, self._sides(piece, line), platform)
        if len(self._laid_pieces) >= PIECES_KEPT:
            self._forget_laid()
        self._laid_pieces[piece, platform] = (version, laid)
        return laid

    def _placed_line(self, laid: _Laid) -> tuple[list[float], list[float]]:
        """Where the points of a piece's centre line land in the picture."""
        placed = self._placed.get((laid.piece, laid.platform))
        if placed is None:
            placed = np.stack(self.projection.to_pixels(*laid.line))
            self._placed[laid.piece, laid.platform] = placed
        xs, ys = placed.tolist()
        return xs, ys

    def _lifted_line(self, laid: _Laid) -> list[float]:
        """How many art pixels each point of a piece's centre line is lifted with the ground."""
        lifts = self._lifts.get((laid.piece, laid.platform))
        if lifts is None:
            assert self.elevation is not None
            lifts = self._lifts[laid.piece, laid.platform] = self.elevation.lift(*laid.line)
        return lifts.tolist()

    def _through_stations(self, near: tuple[int, int, int, int]) -> list[_Laid]:
        """The track through each station near the area, straight between its two ends with
        its platform beside it: double where double track comes in, electric where electric
        track comes in at both ends."""
        laid = []
        for cell in self.constructions.stations():
            if not self._near(cell, near):
                continue
            piece = self.station_piece(cell)
            laid.append(self._laid(piece, station_side(self.projection, self._steps, piece.ends)))
        return laid

    def station_piece(self, cell: Cell) -> Track:
        """The track drawn through a station, straight between its two ends: double where
        double track comes in, electric where electric track comes in at both ends."""
        run = self.constructions.station_ends(cell)
        joined = [_joined(self.constructions, cell, point) for point in run]
        electric = all(any(piece.electric for piece in found) for found in joined)
        double = any(piece.width == TrackWidth.DOUBLE for found in joined for piece in found)
        width = TrackWidth.DOUBLE if double else TrackWidth.SINGLE
        return Track(cell, run, width, TrackBed.GROUND, electric)

    def riding_side(self, piece: Track, share: float, forward: bool) -> float:
        """How far a train `share` of the way along the piece, from its first end to its
        second, rides to the left of the way it goes (as `_offset` moves a line), in cells: on
        double track the track to that side of the centre line, so trains going opposite ways
        keep to different tracks, and on single track the track itself, which leaves the
        centre line only where it branches off double track."""
        sides = [lane[0] for lane in self._sides_at(piece, [share])]
        if piece.width == TrackWidth.DOUBLE:
            return sides[1] if forward else -sides[0]
        return sides[0] if forward else -sides[0]

    def _sides(self, piece: Track, line: Path) -> list[list[float]]:
        """How far to the left of the centre line each track of the piece lies at each of its
        points, in cells. Double track has one track either side of its centre line, closing up
        to it at an end that runs into single track. Single track keeps to its centre line,
        except where it branches off double track in the same cell: there it runs on along the
        outer track on the side it turns to, and closes in to its centre line by its other
        end."""
        lengths = _lengths(line)
        return self._sides_at(piece, [along / lengths[-1] for along in lengths])

    def _sides_at(self, piece: Track, shares: list[float]) -> list[list[float]]:
        """The same as `_sides`, at these shares of the way along the piece from its first end
        to its second."""
        start, end = self._ends(piece)
        if piece.width == TrackWidth.SINGLE:
            return [
                [start * min(1.0, 2 * (1 - share)) + end * min(1.0, 2 * share) for share in shares]
            ]
        wide_start, wide_end = start > 0, end > 0
        widths = []
        for share in shares:
            open_by = 1.0
            if not wide_start:
                open_by = min(open_by, 2 * share)
            if not wide_end:
                open_by = min(open_by, 2 * (1 - share))
            widths.append(DOUBLE_SPACING / 2 * open_by)
        return [[-width for width in widths], widths]

    def _ends(self, piece: Track) -> tuple[float, float]:
        """At each end of the piece, for single track how far to the left of its centre line
        it starts there (`_branch_shift`), and for double track 1 if it runs on as double
        track there (`_double_beyond`) or else 0."""
        self._keep_current()
        shape = self._shapes.get(piece)
        if shape is None:
            if piece.width == TrackWidth.SINGLE:
                start, end = (self._branch_shift(piece, point) for point in piece.ends)
            else:
                start, end = (float(self._double_beyond(piece, point)) for point in piece.ends)
            shape = self._shapes[piece] = (start, end)
        return shape

    def _branch_shift(self, piece: Track, point: int) -> float:
        """How far to the left of its centre line single track starts at this end, to leave
        from the outer track of full double track it branches off there: half the spacing to
        whichever side it turns away to, or nothing."""
        steps = self._steps
        mine = piece.ends[1] if piece.ends[0] == point else piece.ends[0]
        for other in self._track_at(piece.cell):
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
        joined = [other for other in self._track_at(beyond) if entry in other.ends]
        return not joined or any(other.width == TrackWidth.DOUBLE for other in joined)

    def _lay_deck(self, paint: ImageDraw.ImageDraw, lanes: list[Path], cell: float, s: int) -> None:
        """A timber deck under the tracks with a railing along each side."""
        spread = (_lane_spread(lanes) + BRIDGE_WIDTH / 2) * cell
        middle = lanes[0] if len(lanes) == 1 else _middle(lanes)
        paint.polygon(_strip(middle, spread + 0.03 * cell), fill=(*BRIDGE_SHADOW, 255))
        paint.polygon(_strip(middle, spread), fill=(*BRIDGE_DECK, 255))
        for side in (-spread, spread):
            paint.line(_offset(middle, side), fill=(*RAILING, 255), width=s, joint="curve")

    def _lay_platform(
        self,
        paint: ImageDraw.ImageDraw,
        line: Path,
        lanes: list[Path],
        side: Point,
        cell: float,
        s: int,
    ) -> None:
        """A paved platform along the station's side of its track, its white edge by the
        track."""
        (x0, y0), (x1, y1) = line[0], line[-1]
        nx, ny = -(y1 - y0), x1 - x0
        sign = 1.0 if nx * side[0] + ny * side[1] > 0 else -1.0
        near = (_lane_spread(lanes) + BANK / 2) * cell
        far = near + PLATFORM_WIDTH * cell
        paint.polygon(_strip(line, sign * near, sign * far), fill=(*PLATFORM, 255))
        paint.line(_offset(line, sign * (near + s / 2)), fill=(*PLATFORM_EDGE, 255), width=s)

    def _lay_portals(
        self, paint: ImageDraw.ImageDraw, laid: _Laid, line: Path, lanes: list[Path], cell: float
    ) -> None:
        """A dark arch where the tunnel meets track out in the open."""
        grid = self.tile_map.grid
        piece = laid.piece
        middle = line[len(line) // 2]
        half = (PORTAL / 2 + _lane_spread(lanes)) * cell
        for point, end in zip(piece.ends, (line[0], line[-1]), strict=True):
            beyond, _ = grid.track_exit(piece.cell, point)
            if not grid.contains(*beyond) or self.tile_map.relief[beyond] == Relief.MOUNTAINS:
                continue
            inner = (
                middle[0] + (end[0] - middle[0]) * 0.6,
                middle[1] + (end[1] - middle[1]) * 0.6,
            )
            mouth = [inner, end]
            paint.polygon(_strip(mouth, half + 0.03 * cell), fill=(*PORTAL_STONE, 255))
            paint.polygon(_strip(mouth, half), fill=(*PORTAL_DARK, 255))

    def _hang_wires(
        self, paint: ImageDraw.ImageDraw, line: Path, lanes: list[Path], cell: float, s: int
    ) -> None:
        """A wire over each track, and masts at even steps along the piece on the far side of
        it from the viewer, each with an arm reaching over the track to hold the wires."""
        lengths = _lengths(line)
        count = max(1, round(lengths[-1] / (MAST_GAP * cell)))
        for lane in lanes:
            wire = [(x, y - WIRE_HEIGHT * s) for x, y in lane]
            paint.line(wire, fill=(*WIRE, 255), width=s)
        for k in range(count):
            along = (k + 0.5) * lengths[-1] / count
            _, (ux, uy) = _point_at(line, lengths, along)
            far = (-uy, ux) if ux < 0 else (uy, -ux)
            overs = []
            for lane in lanes:
                lane_lengths = _lengths(lane)
                over, _ = _point_at(lane, lane_lengths, along * lane_lengths[-1] / lengths[-1])
                overs.append(over)
            overs.sort(key=lambda over: over[0] * far[0] + over[1] * far[1])
            near, back = overs[0], overs[-1]
            bx, by = back[0] + far[0] * MAST_SIDE * cell, back[1] + far[1] * MAST_SIDE * cell
            top = (bx, by - MAST_HEIGHT * s)
            paint.line([(bx, by), top], fill=(*MAST, 255), width=s)
            paint.line([top, (near[0], near[1] - WIRE_HEIGHT * s)], fill=(*MAST, 255), width=s)


def _lane_spread(lanes: list[Path]) -> float:
    """How far the outer tracks lie from the middle of the piece, in cells."""
    return 0.0 if len(lanes) == 1 else DOUBLE_SPACING / 2


def _middle(lanes: list[Path]) -> Path:
    return [
        ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(lanes[0], lanes[-1], strict=True)
    ]


def _highest(heights: Image.Image, smoothing: int) -> FloatArray:
    """The highest lift marked in each pixel once the fine picture is averaged down; nothing is
    marked outside the box round what was, which stays 0."""
    width, height = heights.size
    out = np.zeros((height // smoothing, width // smoothing), dtype=np.float32)
    box = heights.getbbox()
    if box is None:
        return out
    x0, y0, x1, y1 = _whole_pixels(box, smoothing)
    fine = np.asarray(heights.crop((x0, y0, x1, y1)), dtype=np.float32)
    h, w = fine.shape[0] // smoothing, fine.shape[1] // smoothing
    top, left = y0 // smoothing, x0 // smoothing
    out[top : top + h, left : left + w] = fine.reshape(h, smoothing, w, smoothing).max(axis=(1, 3))
    return out


def _whole_pixels(box: tuple[int, int, int, int], smoothing: int) -> tuple[int, int, int, int]:
    """A box of the fine picture grown to whole pixels of the picture averaged down."""
    x0, y0, x1, y1 = box
    return (
        x0 // smoothing * smoothing,
        y0 // smoothing * smoothing,
        -(-x1 // smoothing) * smoothing,
        -(-y1 // smoothing) * smoothing,
    )


def _averaged(picture: Image.Image, shadow: Image.Image, smoothing: int) -> FloatArray:
    """The fine picture with its shadow under it, averaged down `smoothing` times each way, as
    RGBA from 0 to 1.

    Each pixel sums the fine pixels behind it row by row, in the order numpy's mean over them
    adds them up, but with the channels laid one after another, so each addition runs along a
    whole row of pixels at once."""
    width, height = picture.size
    out = np.zeros((height // smoothing, width // smoothing, 4), dtype=np.float32)
    box = picture.getbbox()
    shadow_box = shadow.getbbox()
    if box is None and shadow_box is None:
        return out
    boxes = [b for b in (box, shadow_box) if b is not None]
    x0, y0, x1, y1 = _whole_pixels(
        (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ),
        smoothing,
    )
    channels = np.asarray(picture.crop((x0, y0, x1, y1))).transpose(2, 0, 1)
    fine = channels.astype(np.float32)
    fine /= 255
    dark = np.asarray(shadow.crop((x0, y0, x1, y1)), dtype=np.float32) / 255 * SHADOW
    alpha = fine[3].copy()
    fine[:3] *= alpha
    np.maximum(alpha, dark, out=fine[3])
    h, w = fine.shape[1] // smoothing, fine.shape[2] // smoothing
    blocks = fine.reshape(4, h, smoothing, w, smoothing)
    coarse = blocks[:, :, 0, :, 0].copy()
    for down, across in itertools.product(range(smoothing), repeat=2):
        if down or across:
            coarse += blocks[:, :, down, :, across]
    coarse /= smoothing * smoothing
    coarse[:3] /= np.maximum(coarse[3], 1e-6)
    top, left = y0 // smoothing, x0 // smoothing
    out[top : top + h, left : left + w] = coarse.transpose(1, 2, 0)
    return out
