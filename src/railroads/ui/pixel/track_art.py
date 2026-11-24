"""Track drawn into the pixel map, built upright on the screen so it keeps one width and look
whichever way it runs: a dark bank round a gravel bed, dark sleepers lying square to the rails
and two light rails, the same on both sides, with a faint shadow round it. Over water it runs on
a timber deck with railings, and over electric track masts hold wires. A tunnel is out of sight
but for a stone portal where it comes out of the mountains facing the viewer. Through a station it
runs straight between the station's two ends, past a platform on the station's side.

Each layer is drawn several times finer than it is shown and averaged down, so its edges and
rails come out smooth instead of stepped. Where hills and mountains rise, track is laid out flat
and then lifted with the ground point by point, on the bed graded for it (Elevation.grade), so it
climbs smoothly and keeps its tracks apart however steep the climb, and the lift of every pixel
it covers is kept, so the map can hide it behind raised ground in front of it
(railroads.ui.pixel.world)."""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.grids import Cell, FloatArray, Grid
from railroads.sim.tilemap import Relief, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH
from railroads.ui.pixel.art.painter import PLATFORM, PLATFORM_EDGE
from railroads.ui.pixel.projection import RGB, Path, PixelProjection, Point
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
# How tall a tunnel portal stands, in cells; how narrow it looks from the side, as a share of its
# width seen from the front; and how far its track may run away from the viewer into it, as the
# share of the way up the picture, for it to be seen.
PORTAL_HEIGHT = 0.26
PORTAL_NARROWEST = 0.6
PORTAL_FACING = 0.5
SHADOW_REACH = 0.03
DOUBLE_SPACING = 0.18
# How far the outer tracks of quadruple track lie from its middle, in cells: as far from the inner
# tracks as these lie from each other, close enough that two lines of it in cells beside each
# other keep apart.
QUADRUPLE_OUTER = 1.5 * DOUBLE_SPACING
BEND = 0.2
# How far a rail may wander across the picture, in pixels it is shown at, and still count as
# running straight across or down it, to be kept to one row or column of pixels (_crisp).
STRAIGHT_WITHIN = 0.25
# How dark the shadow round the track is, from 0 to 1.
SHADOW = 0.35
# How far apart track is followed over the ground to lift it, in cells.
STEP = 1 / 12
# How many times finer than shown, each way, track is drawn before it is averaged down, by the
# resolution it is shown at: enough to smooth it, without drawing a huge picture up close.
SMOOTHING = {1: 4, 2: 2}
# How wide a rail is, in art pixels, though never narrower than a pixel shown.
RAIL_WIDTH = 0.25
# How many pieces of track are kept laid out, with where they land in the picture and how far
# they are lifted, before all are forgotten and laid out again as they are drawn: more than the
# chunks round a view of a busy railway draw, and few enough that a big railway's pieces do not
# fill memory.
PIECES_KEPT = 1 << 11
# A station's platform beside its track, and how far from the track's middle the station house
# stands, in cells of the picture's width.
PLATFORM_WIDTH = 0.12
# How far apart the two tracks either side of an island platform lie in a station on quadruple
# track, in cells, and how far into the station's cell from each end they have opened out to that.
ISLAND_TRACKS = 0.3
ISLAND_FROM = 0.22
SET_BACK = 0.34
# How gently a track branching off other track turns out of it: round a circle of this radius,
# in cells, or as wide as there is room for; and how many cells along that track another track of
# it crosses over to the track that turns out for each cell it crosses, as a crossover does.
TURNOUT_RADIUS = 0.6
CROSSOVER_SLOPE = 4.0

# Overhead wires on electric track: how far beside the track their masts stand and how far
# apart, in cells, and how tall the masts are and how high the wires hang, in art pixels.
MAST_SIDE = 0.13
MAST_GAP = 1.0
MAST_HEIGHT = 6
WIRE_HEIGHT = 4
# Maglev track's concrete beam: how wide it is and how wide its darker edges, in cells.
GUIDEWAY_WIDTH = 0.16
GUIDEWAY_RIM = 0.025

BANK_EDGE: RGB = (58, 52, 44)
GRAVEL: RGB = (98, 90, 80)
SLEEPER: RGB = (46, 30, 20)
RAIL: RGB = (238, 236, 230)
RAIL_UNDER: RGB = (24, 22, 20)
BRIDGE_DECK: RGB = (126, 86, 50)
BRIDGE_SHADOW: RGB = (62, 42, 26)
RAILING: RGB = (70, 48, 30)
PORTAL_STONE: RGB = (118, 112, 104)
PORTAL_LIT: RGB = (168, 160, 148)
PORTAL_EDGE: RGB = (58, 54, 50)
PORTAL_DARK: RGB = (20, 18, 18)
MAST: RGB = (96, 96, 104)
GUIDEWAY: RGB = (196, 194, 188)
GUIDEWAY_EDGE: RGB = (112, 110, 108)
GUIDEWAY_GROOVE: RGB = (70, 72, 80)
WIRE: RGB = (40, 40, 46)


def _lay_guideway(
    paint: Canvas, lanes: list[Path], lifts: list[float], cell: float, line: int
) -> None:
    """Maglev track: a pale concrete beam with darker edges and a groove down its middle where
    rails and sleepers would be."""
    for lane in lanes:
        edge = _raised_strip(lane, lifts, GUIDEWAY_WIDTH / 2 * cell)
        paint.polygon(edge, fill=(*GUIDEWAY_EDGE, 255))
        inner = (GUIDEWAY_WIDTH / 2 - GUIDEWAY_RIM) * cell
        paint.polygon(_raised_strip(lane, lifts, inner), fill=(*GUIDEWAY, 255))
        groove = _raised(lane, lifts)
        paint.line(groove, fill=(*GUIDEWAY_GROOVE, 255), width=line, joint="curve")


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
        # Where the line doubles back, the moved lines meet far off: they are kept at most
        # twice as far out as the line is moved.
        dot = mx * before[0] + my * before[1]
        dot = max(0.5, dot) if dot else 1.0
        moved.append((x + mx / dot * amount, y + my / dot * amount))
    return moved


def _raised(path: Path, lifts: list[float]) -> Path:
    """The path with each of its points lifted up the picture by its own lift."""
    return [(x, y - lift) for (x, y), lift in zip(path, lifts, strict=True)]


def _raised_strip(path: Path, lifts: list[float], left: float, right: float | None = None) -> Path:
    """`_strip` laid out flat along the path and then lifted point by point, so it keeps to the
    same side of the path however steeply the lifts carry it up or down the picture."""
    right = -left if right is None else right
    return _Strip(_raised(_offset(path, left), lifts) + _raised(_offset(path, right), lifts)[::-1])


class _Strip(list[Point]):
    """The outline of a strip along a path: its points along one side, then back along the
    other, which the graphics card fills as the quads between the two sides."""


def _bend(start: Point, middle: Point, end: Point, matrix: FloatArray, radius: float) -> Path:
    """The line from `start` through `middle` to `end`, rounded off round the middle instead of
    turning sharply there: on a circle in the picture, which `matrix` turns the map into, from
    BEND before the middle to BEND after it, or further out where the turn looks sharp in the
    picture, so the circle is at least `radius` art pixels round where it can be. Anything laid
    along the line no further than `radius` to its side then turns with it instead of folding
    over itself on the inside of the turn into spikes."""
    centre = np.array(middle)
    arms = [np.asarray(point) - centre for point in (start, end)]
    seen = [matrix @ arm for arm in arms]
    lengths = [float(np.hypot(*arm)) for arm in seen]
    ua, ub = (arm / length for arm, length in zip(seen, lengths, strict=True))
    cos = float(ua @ ub)
    if cos < -0.999:
        return [start, middle, end]
    half = math.acos(max(-1.0, min(1.0, cos))) / 2
    shortest = min(
        BEND * seen_length / float(np.hypot(*arm))
        for arm, seen_length in zip(arms, lengths, strict=True)
    )
    tangent = min(max(shortest, radius / math.tan(half)), 0.95 * min(lengths))
    across = (ua + ub) / float(np.hypot(*(ua + ub)))
    hub = across * tangent / math.cos(half)
    p, q = ua * tangent - hub, ub * tangent - hub
    first = math.atan2(p[1], p[0])
    sweep = (math.atan2(q[1], q[0]) - first + math.pi) % (2 * math.pi) - math.pi
    round_by = tangent * math.tan(half)
    seen_curve = [
        hub + round_by * np.array([math.cos(angle), math.sin(angle)])
        for angle in (first + sweep * k / 8 for k in range(9))
    ]
    back = np.linalg.inv(matrix)
    curve = [tuple(float(v) for v in centre + back @ point) for point in seen_curve]
    return [start, *curve, end]


def centre_line(
    grid: Grid, steps: FloatArray, matrix: FloatArray, piece: Track, radius: float
) -> Path:
    """The middle of a piece of track through its cell, in the plane: from one of its ends
    through the cell's centre to the other, rounded off round the centre (_bend), with points no
    more than STEP apart. `steps` are the grid's track_steps."""
    cx, cy = (float(v) for v in grid.center(*piece.cell))
    a, b = (steps[point] / 2 for point in piece.ends)
    ends = (cx + a[0], cy + a[1]), (cx + b[0], cy + b[1])
    return _dense(_bend(ends[0], (cx, cy), ends[1], matrix, radius), STEP)


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


def _sleepers(
    lane: Path, lifts: list[float], cell: float, kept: tuple[float, float] | None = None
) -> list[Path]:
    """The sleepers under one track on the screen, evenly along it as it lies flat, each lying
    square to the way it runs there and lifted with it, only those `kept` between these two
    distances along it if given; `cell` is a cell's width in the picture's pixels."""
    lengths = _lengths(lane)
    count = max(1, round(lengths[-1] / (SLEEPER_GAP * cell)))
    half_width, half_length = SLEEPER_WIDTH * cell / 2, SLEEPER_LENGTH * cell / 2
    sleepers = []
    for k in range(count):
        middle = (k + 0.5) * lengths[-1] / count
        if kept is not None and not kept[0] <= middle <= kept[1]:
            continue
        (x, y), (ux, uy) = _point_at(lane, lengths, middle)
        y -= float(np.interp(middle, lengths, lifts))
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
    cells = SET_BACK + _spread(station_width(constructions, cell), station=True)
    width = float(np.hypot(*(projection.matrix @ steps[0])))
    return side[0] * cells * width, side[1] * cells * width


def station_width(constructions: Constructions, cell: Cell) -> TrackWidth:
    """How many tracks run through a station: as many as the widest track coming in."""
    return max(
        (
            piece.width
            for point in constructions.station_ends(cell)
            for piece in _joined(constructions, cell, point)
        ),
        default=TrackWidth.SINGLE,
    )


def _spread(width: TrackWidth, station: bool = False) -> float:
    """How far the middle of the outer tracks of track this wide lie from its middle, in cells:
    in a station on quadruple track, beyond its island platforms."""
    half = DOUBLE_SPACING / 2
    if width == TrackWidth.QUADRUPLE:
        return half + ISLAND_TRACKS if station else QUADRUPLE_OUTER
    return half if width == TrackWidth.DOUBLE else 0.0


def track_spread(piece: Track, station: bool = False) -> float:
    """How far the middle of a piece's outer tracks lie from its middle, in cells."""
    return _spread(piece.width, station)


def _eased(share: float) -> float:
    """How far a track moving over from one place to another has got, `share` of the way along:
    setting out and arriving gently, so it runs over in a curve rather than a straight line."""
    share = min(1.0, max(0.0, share))
    return share * share * (3 - 2 * share)


def _turnout(line: Path, start: float, end: float, radius: float) -> Path | None:
    """The way a track runs through a piece branching off other track, the piece's centre line
    lying along `line` in the picture from the end it branches off at: in along the line
    `start` to its left and out of it `end` to its left, running straight on from where it
    comes in and then round in one curve, round `radius` where there is room, onto the straight
    it goes out along. Its first point is where it comes in, its second where it starts to
    turn. None where it cannot turn that way, its way out lying back behind where it comes
    in."""
    (ax, ay), (bx, by) = _offset(line[:2], start)[0], _offset(line[-2:], end)[-1]
    (x0, y0), (x1, y1) = line[0], line[1]
    (x2, y2), (x3, y3) = line[-2], line[-1]
    d0, d1 = (x1 - x0, y1 - y0), (x3 - x2, y3 - y2)
    t, u = _corner((ax, ay), d0, (bx, by), d1) or (0.0, 0.0)
    if t <= 0 or u <= 0:
        return None
    corner = (ax + t * d0[0], ay + t * d0[1])
    return _bend((ax, ay), corner, (bx, by), np.eye(2), radius)


def _corner(a: Point, along: Point, b: Point, onward: Point) -> tuple[float, float] | None:
    """How far on from `a` along `along`, and back from `b` against `onward`, the lines through
    them meet, in lengths of those; None where they run side by side."""
    det = along[1] * onward[0] - along[0] * onward[1]
    if abs(det) < 1e-9:
        return None
    gap = (b[0] - a[0], b[1] - a[1])
    return (
        (gap[1] * onward[0] - gap[0] * onward[1]) / det,
        (along[1] * gap[0] - along[0] * gap[1]) / det,
    )


def _crossover(line: Path, main: float, turning: Path, slope: float) -> Path:
    """The way a track of the track a piece branches off runs over to the track `turning`
    turns out of (`_turnout`) and on along it, the piece's centre line lying along `line` in
    the picture from the end it branches off at and the track coming in `main` to its left:
    straight on, then over in an easy S just before `turning` starts to turn, `slope` along for
    each across where there is room."""
    (x0, y0), (x1, y1) = line[0], line[1]
    length = math.hypot(x1 - x0, y1 - y0) or 1.0
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -dy, dx
    ax, ay = x0 + nx * main, y0 + ny * main
    (rx, ry), (tx, ty) = turning[0], turning[1]
    turn = (tx - rx) * dx + (ty - ry) * dy
    across = (rx - ax) * nx + (ry - ay) * ny
    span = min(turn, abs(across) * slope)
    steps = 12
    over = [
        (
            ax + dx * (turn - span * (1 - k / steps)) + nx * across * _eased(k / steps),
            ay + dy * (turn - span * (1 - k / steps)) + ny * across * _eased(k / steps),
        )
        for k in range(steps + 1)
    ]
    return [(ax, ay), *over, *turning[2:]]


def _sides_along(line: Path, lane: Path) -> list[float]:
    """How far to the left of each point of `line` the lane lies, square to the line there."""
    starts = np.array(lane[:-1])
    runs = np.array(lane[1:]) - starts
    sides = []
    for index, (x, y) in enumerate(line):
        (px, py), (qx, qy) = line[max(0, index - 1)], line[min(len(line) - 1, index + 1)]
        length = math.hypot(qx - px, qy - py) or 1.0
        normal = np.array([-(qy - py) / length, (qx - px) / length])
        det = normal[0] * -runs[:, 1] + runs[:, 0] * normal[1]
        gap = starts - (x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (gap[:, 0] * -runs[:, 1] + runs[:, 0] * gap[:, 1]) / det
            along = (normal[0] * gap[:, 1] - normal[1] * gap[:, 0]) / det
        hits = out[(np.abs(det) > 1e-12) & (along >= -1e-6) & (along <= 1 + 1e-6)]
        if hits.size:
            sides.append(float(hits[np.argmin(np.abs(hits))]))
        else:
            sides.append(sides[-1] if sides else 0.0)
    return sides


def _on_its_way_in(lane: Path, within: float) -> int:
    """How many of the lane's first points lie no further than `within` from the straight line
    it comes in along: where track branching off other track still lies on that track's."""
    (x0, y0), (x1, y1) = lane[0], lane[1]
    length = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    for k, (x, y) in enumerate(lane):
        if abs((x - x0) * uy - (y - y0) * ux) > within:
            return k
    return len(lane)


def _lane_sides(tracks: int, lanes: int | None = None) -> list[float]:
    """How far to the left of the centre line each of `lanes` tracks lies, in cells, where the
    track is `tracks` wide: lanes the narrower track has no room for run into its outer tracks."""
    lanes = tracks if lanes is None else lanes
    if lanes == 1:
        return [0.0]
    half = DOUBLE_SPACING / 2
    if lanes == 2:
        outer = QUADRUPLE_OUTER if tracks >= 4 else half if tracks == 2 else 0.0
        return [-outer, outer]
    if tracks >= 4:
        return [-QUADRUPLE_OUTER, -half, half, QUADRUPLE_OUTER]
    if tracks >= 2:
        return [-half, -half, half, half]
    return [0.0, 0.0, 0.0, 0.0]


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
    forks: tuple[bool, bool] = (False, False)


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
        self._shapes: dict[Track, tuple[list[float], list[float]]] = {}
        self._turnout_lanes: dict[Track, tuple[FloatArray, list[FloatArray]]] = {}
        self._laid_pieces: dict[tuple[Track, Point | None], tuple[int, _Laid]] = {}
        self._placed: dict[tuple[Track, Point | None], FloatArray] = {}
        self._lifts: dict[tuple[Track, Point | None], FloatArray] = {}
        self._shapes_version = -1
        self._grades = -1
        # The middle of each cell in the picture, and the steps to the cells track leaves
        # each cell for, which never change.
        self._centres: dict[Cell, tuple[float, float]] = {}
        self._steps = tile_map.grid.track_steps()
        self._planned: dict[Cell, list[Track]] = {}
        for piece in planned:
            self._planned.setdefault(piece.cell, []).append(piece)

    def _keep_current(self) -> None:
        """Forgets what was kept about pieces of track before track or stations last changed,
        and how far they lift before the ground was last graded for track."""
        if self._shapes_version != self.constructions.railway_version:
            self._shapes.clear()
            self._turnout_lanes.clear()
            self._forget_laid()
            self._shapes_version = self.constructions.railway_version
        grades = 0 if self.elevation is None else self.elevation.grades
        if grades != self._grades:
            self._lifts.clear()
            self._grades = grades

    def beds(self) -> list[FloatArray]:
        """The middle line through the plane of all track laid on the ground, through stations
        too, as a row of x and a row of y each, for the ground to be graded under
        (Elevation.grade)."""
        laid = [
            self._laid(piece)
            for piece in self.constructions.track()
            if piece.bed == TrackBed.GROUND
        ]
        for cell in self.constructions.every_station_cell():
            piece = self.station_piece(cell)
            laid.append(self._laid(piece, station_side(self.projection, self._steps, piece.ends)))
        return [each.line for each in laid]

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
        laid = self._laid_near(pieces, left, top, width, height, stations, lifted)
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
        self._lay_out(paint, shade, measure, laid, left, top, s, smoothing, lifted)
        return _averaged(picture, shadow, smoothing), _highest(heights, smoothing)

    def shapes(
        self,
        pieces: list[Track],
        left: int,
        top: int,
        width: int,
        height: int,
        scale: int,
        quality: int,
        lifted: bool,
        stations: bool = True,
    ) -> TrackShapes | None:
        """The same track as `layer`, or as `lifted` with `lifted`, as triangles for the graphics
        card to draw (TrackShapes), laid out `scale` pixels to an art pixel and `quality` times
        finer still, to be averaged down as the CPU averages its picture; None if no track lies
        near."""
        laid = self._laid_near(pieces, left, top, width, height, stations, lifted)
        if not laid:
            return None
        shapes = _Recorder()
        s = scale * quality
        self._lay_out(
            shapes, shapes.shadow, shapes.heights, laid, left, top, s, quality, lifted, True
        )
        return shapes.shapes(scale, quality)

    def _laid_near(
        self,
        pieces: list[Track],
        left: int,
        top: int,
        width: int,
        height: int,
        stations: bool,
        lifted: bool,
    ) -> list[_Laid]:
        """The pieces, and with `stations` the track through stations, that can show over the
        art pixels from `left`, `top` across `width` by `height`."""
        reach = PIXEL_CELL_WIDTH * 2
        below = MOST_LIFT if lifted else 0
        near = (left - reach, top - reach, left + width + reach, top + height + reach + below)
        laid = [self._laid(piece) for piece in pieces if self._near(piece.cell, near)]
        if stations:
            laid += self._through_stations(near)
        return laid

    def _lay_out(
        self,
        paint: Canvas,
        shade: Canvas,
        measure: Canvas,
        laid: list[_Laid],
        left: int,
        top: int,
        s: int,
        smoothing: int,
        lifted: bool,
        thin_rails: bool = False,
    ) -> None:
        """Draws the laid pieces `s` pixels to an art pixel from (`left`, `top`): the track with
        `paint`, its shadow with `shade` and, raised, how far each stretch is lifted with
        `measure`; `smoothing` pixels make one shown. Rails are RAIL_WIDTH wide, but never
        narrower than a pixel shown unless `thin_rails`, which leaves a rail narrower than a
        pixel far off to show only as faint as its share of the pixel, as everything else
        drawn finely and shrunk does."""
        cell = PIXEL_CELL_WIDTH * s
        raise_up = lifted and self.elevation is not None

        def screen(each: _Laid) -> tuple[Path, list[float]]:
            xs, ys = self._placed_line(each)
            lifts = self._lifted_line(each) if raise_up else [0.0] * len(xs)
            flat = [((x - left) * s, (y - top) * s) for x, y in zip(xs, ys, strict=True)]
            return flat, [lift * s for lift in lifts]

        # Everything is laid out flat on the picture and lifted only then, point by point, so
        # tracks keep their sides and widths on the steepest climbs.
        drawn = []
        for each in laid:
            line, lifts = screen(each)
            lanes = [_offset(line, [side * cell for side in sides]) for sides in each.sides]
            drawn.append((each, line, lanes, lifts))
            if raise_up:
                self._measure(measure, each, line, lanes, lifts, cell, s)
        ground = [item for item in drawn if item[0].piece.bed != TrackBed.TUNNEL]
        for each, _, lanes, lifts in ground:
            half = (BRIDGE_WIDTH if each.piece.bed == TrackBed.BRIDGE else BANK) / 2
            for lane in lanes:
                shade.polygon(_raised_strip(lane, lifts, (half + SHADOW_REACH) * cell), fill=255)
        for each, _, lanes, lifts in drawn:
            if each.piece.bed == TrackBed.BRIDGE:
                self._lay_deck(paint, lanes, lifts, cell, s)
        for each, _, lanes, lifts in ground:
            if each.piece.bed == TrackBed.GROUND:
                for lane in lanes:
                    bank = _raised_strip(lane, lifts, BANK / 2 * cell)
                    paint.polygon(bank, fill=(*BANK_EDGE, 255))
        for each, line, lanes, lifts in ground:
            if each.platform is not None:
                self._lay_platform(paint, line, lanes, lifts, each.platform, cell, s)
        railed = [item for item in ground if not item[0].piece.maglev]
        for each, _, lanes, lifts in railed:
            if each.piece.bed == TrackBed.GROUND:
                for lane in lanes:
                    bed = _raised_strip(lane, lifts, TRACK_BED / 2 * cell)
                    paint.polygon(bed, fill=(*GRAVEL, 255))
        rail = max(1 if thin_rails else smoothing, round(RAIL_WIDTH * s))
        for each, _, lanes, lifts in ground:
            if each.piece.maglev:
                _lay_guideway(paint, lanes, lifts, cell, rail)
        for each, _, lanes, lifts in railed:
            for lane in lanes:
                for sleeper in _sleepers(lane, lifts, cell, self._bare(each, lanes, lane, cell)):
                    paint.polygon(sleeper, fill=(*SLEEPER, 255))
        rails = [
            _crisp(_raised(_offset(lane, side), lifts), smoothing, max(rail, smoothing))
            for _, _, lanes, lifts in railed
            for lane in lanes
            for side in (-GAUGE / 2 * cell, GAUGE / 2 * cell)
        ]
        for rail_line in rails:
            paint.line(rail_line, fill=(*RAIL_UNDER, 255), width=2 * rail, joint="curve")
        for rail_line in rails:
            paint.line(rail_line, fill=(*RAIL, 255), width=rail, joint="curve")
        for each, line, lanes, lifts in drawn:
            if each.piece.bed == TrackBed.TUNNEL:
                for portal in self._portals(each, line, lanes, lifts, cell):
                    _lay_portal(paint, portal, s)
        for each, line, lanes, lifts in ground:
            if each.piece.electric:
                self._hang_wires(paint, line, lanes, lifts, cell, s)

    @staticmethod
    def _bare(laid: _Laid, lanes: list[Path], lane: Path, cell: float) -> tuple[float, float]:
        """How far along the lane its sleepers start and end: where it branches off other
        track, only once it has left that track's, and crossing over to another track of the
        piece, only until it is on it: there the other's sleepers lie already and would only be
        blurred by its own."""
        lengths = _lengths(lane)
        within = GAUGE / 2 * cell
        start = _on_its_way_in(lane, within) if laid.forks[0] else 0
        end = len(lane) - _on_its_way_in(lane[::-1], within) if laid.forks[1] else len(lane)
        onto = [other for other in lanes[: int(laid.piece.width)] if other is not lane]
        if any(laid.forks) and lanes.index(lane) >= int(laid.piece.width) and onto:
            on = [
                min(math.dist(point, other[k]) for other in onto) < within
                for k, point in enumerate(lane)
            ]
            if laid.forks[0]:
                end = min(end, next((k for k in range(start, len(lane)) if on[k]), end))
            else:
                start = max(start, next((k for k in range(end - 1, -1, -1) if on[k]), start))
        return lengths[max(0, start - 1)], lengths[max(0, min(len(lane), end) - 1)]

    def _measure(
        self,
        measure: Canvas,
        laid: _Laid,
        line: Path,
        lanes: list[Path],
        lifts: list[float],
        cell: float,
        s: int,
    ) -> None:
        """Marks how far each stretch of the piece is lifted, in art pixels, over every pixel it
        can cover, masts and wires too; of a tunnel only its portals."""
        if laid.piece.bed == TrackBed.TUNNEL:
            for portal in self._portals(laid, line, lanes, lifts, cell):
                measure.polygon(_arch(portal, s), fill=portal.lift / s)
            return
        half = (max(BANK, BRIDGE_WIDTH) / 2 + _lane_spread(lanes, cell) + SHADOW_REACH) * cell
        rises = (0, (MAST_HEIGHT + 1) * s) if laid.piece.electric else (0,)
        if laid.piece.electric:
            half += MAST_SIDE * cell
        for (a, b), (la, lb) in zip(
            itertools.pairwise(line), itertools.pairwise(lifts), strict=True
        ):
            for rise in rises:
                stretch = _raised_strip([a, b], [la + rise, lb + rise], half)
                measure.polygon(stretch, fill=(la + lb) / 2 / s)

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
        line = self._centre_line(piece)
        sides = self._sides(piece, line)
        forks = tuple(self._branch_lanes(piece, point) is not None for point in piece.ends)
        plane = np.array([[x for x, _ in line], [y for _, y in line]])
        laid = _Laid(piece, plane, sides, platform, (forks[0], forks[1]))
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
        """The track through each station and added platform near the area, straight between its
        two ends with its platform beside it: double where double track comes in, electric where
        electric track comes in at both ends, maglev where only maglev track comes in."""
        laid = []
        for cell in self.constructions.every_station_cell():
            if not self._near(cell, near):
                continue
            piece = self.station_piece(cell)
            laid.append(self._laid(piece, station_side(self.projection, self._steps, piece.ends)))
        return laid

    def station_piece(self, cell: Cell) -> Track:
        """The track drawn through a station, straight between its two ends: double where
        double track comes in, electric where electric track comes in at both ends, maglev where
        only maglev track comes in."""
        run = self.constructions.station_ends(cell)
        joined = [_joined(self.constructions, cell, point) for point in run]
        electric = all(any(piece.electric for piece in found) for found in joined)
        coming_in = [piece for found in joined for piece in found]
        maglev = bool(coming_in) and all(piece.maglev for piece in coming_in)
        width = max((piece.width for found in joined for piece in found), default=TrackWidth.SINGLE)
        return Track(cell, run, width, TrackBed.GROUND, electric and not maglev, maglev)

    def riding_side(self, piece: Track, share: float, forward: bool, lane: int = 0) -> float:
        """How far a train `share` of the way along the piece, from its first end to its
        second, rides to the left of the way it goes (as `_offset` moves a line), in cells: on
        double track the track to that side of the centre line, so trains going opposite ways
        keep to different tracks, and so too where single track splits in two to run into both
        tracks of double track it branches off; on quadruple track the inner track that side
        for `lane` 0, the fast track, and the outer one for lane 1, the slow track; elsewhere on
        single track the track itself."""
        sides = [lane_sides[0] for lane_sides in self._sides_at(piece, [share])]
        if len(sides) == 1:
            return sides[0] if forward else -sides[0]
        if len(sides) == 2:
            return sides[1] if forward else -sides[0]
        if len(sides) == 4:
            return sides[2 + lane] if forward else -sides[1 - lane]
        return 0.0

    def _sides(self, piece: Track, line: Path) -> list[list[float]]:
        """How far to the left of the centre line each track of the piece lies at each of its
        points, in cells, with the tracks branching off other track forking into every track
        of it they join (`_sides_at`)."""
        lengths = _lengths(line)
        return self._sides_at(piece, [along / lengths[-1] for along in lengths], forks=True)

    def _sides_at(
        self, piece: Track, shares: list[float], forks: bool = False
    ) -> list[list[float]]:
        """How far to the left of the centre line each track of the piece lies at these shares
        of the way along it from its first end to its second, in cells. Double track has one
        track either side of its centre line, closing up to it at an end that runs into single
        track; quadruple track two either side, the inner pair DOUBLE_SPACING apart and the
        outer ones as far again beyond them, the outer ones closing up onto the inner where it
        runs into double track and all onto the centre line where it runs into single. Where the
        piece branches off straighter track in its cell each of its tracks leaves from a track
        of that one going its way (`_branch_lanes`), and with `forks` from each of the others
        going its way too, crossing the tracks between, each turning out of it in one curve
        (`_turnouts`)."""
        if any(self._branch_lanes(piece, point) is not None for point in piece.ends):
            along, turnouts = self._turnouts(piece)
            kept = turnouts if forks else turnouts[: int(piece.width)]
            return [np.interp(shares, along, lane).tolist() for lane in kept]
        full = _lane_sides(int(piece.width))
        opening = 2.0
        if piece.width == TrackWidth.QUADRUPLE and self.constructions.station_at(piece.cell):
            half = DOUBLE_SPACING / 2
            full = [-(half + ISLAND_TRACKS), -half, half, half + ISLAND_TRACKS]
            opening = 1 / ISLAND_FROM
        at_start, at_end = self._ends(piece)
        lanes = []
        for index, middle in enumerate(full):
            sides = []
            for share in shares:
                if share < 0.5:
                    side = at_start[index] + (middle - at_start[index]) * _eased(opening * share)
                elif share > 0.5:
                    side = at_end[index] + (middle - at_end[index]) * _eased(opening * (1 - share))
                else:
                    side = middle
                sides.append(side)
            lanes.append(sides)
        return lanes

    def _turnouts(self, piece: Track) -> tuple[FloatArray, list[FloatArray]]:
        """How far to the left of the centre line of a piece branching off other track each of
        its tracks lies, in cells, at the shares of the way along it in the first array: first
        its own tracks, each from the track it leaves from (`_branch_lanes`), then each other
        track of the track it branches off going that way, each running into the track of the
        piece it joins in one curve (`_turnout`)."""
        self._keep_current()
        found = self._turnout_lanes.get(piece)
        if found is not None:
            return found
        at_start, at_end = self._ends(piece)
        branch = self._branch_lanes(piece, piece.ends[0])
        # Laid out from the end it branches off at, where the left is the other side.
        way = 1.0 if branch is not None else -1.0
        if branch is None:
            branch = self._branch_lanes(piece, piece.ends[1])
            at_start, at_end = at_end, at_start
        assert branch is not None
        xs, ys = self.projection.matrix @ np.array(self._centre_line(piece)).T
        line = list(zip(xs.tolist(), ys.tolist(), strict=True))
        if way < 0:
            line.reverse()
        lengths = _lengths(line)
        along = [length / (lengths[-1] or 1.0) for length in lengths]
        cell = way * PIXEL_CELL_WIDTH
        radius = TURNOUT_RADIUS * PIXEL_CELL_WIDTH
        turning = [
            _turnout(line, start * cell, end * cell, radius)
            or _offset(line, [start * cell + (end - start) * cell * _eased(k) for k in along])
            for start, end in zip(at_start, at_end, strict=True)
        ]
        paths = turning + [
            _turnout(line, main * cell, at_end[k] * cell, radius)
            or _crossover(line, main * cell, turning[k], CROSSOVER_SLOPE)
            for k, joins in enumerate(branch)
            for main in joins[1:]
        ]
        if way < 0:
            line.reverse()
        lengths = _lengths(line)
        shares = np.array(lengths) / (lengths[-1] or 1.0)
        lanes = [np.array(_sides_along(line, _dense(path, STEP))) / abs(cell) for path in paths]
        found = self._turnout_lanes[piece] = (shares, lanes)
        return found

    def _centre_line(self, piece: Track) -> Path:
        """The piece's centre line in the plane, rounded off round the middle of its cell enough
        for the furthest of its tracks to turn round without folding over itself."""
        if any(self._branch_lanes(piece, point) is not None for point in piece.ends):
            reach = max(
                abs(side)
                for point in piece.ends
                for joins in (self._branch_lanes(piece, point) or [])
                for side in joins
            )
            reach = max(reach, *(abs(side) for end in self._ends(piece) for side in end))
        else:
            reach = max(
                abs(side) for lane in self._sides_at(piece, [0.0, 0.5, 1.0]) for side in lane
            )
        bed = BRIDGE_WIDTH if piece.bed == TrackBed.BRIDGE else BANK
        radius = (reach + bed / 2 + SHADOW_REACH) * PIXEL_CELL_WIDTH
        return centre_line(self.tile_map.grid, self._steps, self.projection.matrix, piece, radius)

    def _ends(self, piece: Track) -> tuple[list[float], list[float]]:
        """Where each track of the piece lies at each of its ends, in cells to the left of its
        centre line: where it branches off straighter track at least as wide in its cell, on
        that track's outer tracks on the side it turns to (`_branch_lanes`); elsewhere as wide
        as it is, or as the track it runs into there if that is narrower, its outer tracks
        closing onto the inner ones."""
        self._keep_current()
        shape = self._shapes.get(piece)
        if shape is None:
            lanes = int(piece.width)
            ends = []
            for point in piece.ends:
                branch = self._branch_lanes(piece, point)
                if branch is not None:
                    ends.append([joins[0] for joins in branch])
                elif lanes == 1:
                    ends.append([0.0])
                else:
                    ends.append(_lane_sides(min(lanes, self._width_beyond(piece, point)), lanes))
            shape = self._shapes[piece] = (ends[0], ends[1])
        return shape

    def _branch_lanes(self, piece: Track, point: int) -> list[list[float]] | None:
        """Where the piece's tracks leave from at this end where it branches off straighter
        track in its cell at least as wide: for each of its tracks, in order from its right to
        its left, the tracks of that one it joins, each in cells to the left of the piece's centre
        line, the one it leaves from first. Each joins every track there going its way: where
        the piece is as wide its own, so the whole line forks; double off quadruple the pair on
        its side; single all of them. It leaves from the one furthest to the side it turns to,
        the others crossing over to that one (`_turnouts`), so a train off it keeps clear of the
        rest. None where it branches off none."""
        grid = self.tile_map.grid
        turn = grid.track_turn(*piece.ends)
        mains = [
            other
            for other in self._track_at(piece.cell)
            if other != piece
            and point in other.ends
            and other.width >= piece.width
            and grid.track_turn(*other.ends) < turn - 1e-6
        ]
        if not mains:
            return None
        main = min(mains, key=lambda other: grid.track_turn(*other.ends))
        # The tracks of both, to the left of the way from this end into the cell: the main
        # track's where they lie at this end, closed up where it runs into narrower track.
        from_here = 1.0 if point == main.ends[0] else -1.0
        at_end = self._ends(main)[0 if point == main.ends[0] else 1]
        tracks = sorted({round(from_here * lane, 9) for lane in at_end})
        own = _lane_sides(int(piece.width))
        matrix = self.projection.matrix
        on = matrix @ (self._steps[main.other_end(point)] / 2)
        off = matrix @ (self._steps[piece.other_end(point)] / 2)
        side = 1.0 if on[0] * off[1] - on[1] * off[0] > 0 else -1.0
        if len(own) == len(tracks):
            joins = [[lane] for lane in tracks]
        else:
            joins = [
                sorted(
                    [lane for lane in tracks if len(own) == 1 or lane * mine > 0] or tracks,
                    key=lambda lane: -lane * side,
                )
                for mine in own
            ]
        joins = [
            sorted(lanes, key=lambda lane: not self._turns_out(piece, point, main, lane, mine))
            for lanes, mine in zip(joins, own, strict=True)
        ]
        into_piece = 1.0 if point == piece.ends[0] else -1.0
        if into_piece < 0:
            joins.reverse()
        return [[into_piece * lane for lane in lanes] for lanes in joins]

    def _turns_out(self, piece: Track, point: int, main: Track, lane: float, mine: float) -> bool:
        """Whether a track of the piece can turn out of the track `lane` of `main` to the left of
        the way from this end into the cell in one curve, onto where it leaves the cell `mine`
        to the left of its way, as it is seen in the picture: where wide track lies across more
        of the cell than its far edge, its outer track on that side can turn out only by first
        swinging back."""
        grid = self.tile_map.grid
        centre = np.array(grid.center(*piece.cell), dtype=float)
        matrix = self.projection.matrix
        here = matrix @ (centre + self._steps[point] / 2)
        along = matrix @ (self._steps[main.other_end(point)] - self._steps[point]) / 2
        there = matrix @ (centre + self._steps[piece.other_end(point)] / 2)
        onward = matrix @ (self._steps[piece.other_end(point)] / 2)
        start = _offset([tuple(here), tuple(here + along)], lane * PIXEL_CELL_WIDTH)[0]
        end = _offset([tuple(there - onward), tuple(there)], mine * PIXEL_CELL_WIDTH)[-1]
        met = _corner(start, tuple(along), end, tuple(onward))
        return met is not None and met[0] > 0 and met[1] > 0

    def _width_beyond(self, piece: Track, point: int) -> int:
        """How many tracks wide the track leaving the piece by this point runs on: the widest
        track it joins there, or the track through the station it runs into; as wide as the
        piece where it runs nowhere."""
        grid = self.tile_map.grid
        beyond, entry = grid.track_exit(piece.cell, point)
        if not grid.contains(*beyond):
            return int(piece.width)
        if self.constructions.station_at(beyond) is not None:
            return int(self.station_piece(beyond).width)
        joined = [other for other in self._track_at(beyond) if entry in other.ends]
        if not joined:
            return int(piece.width)
        return max(int(other.width) for other in joined)

    def _lay_deck(
        self,
        paint: Canvas,
        lanes: list[Path],
        lifts: list[float],
        cell: float,
        s: int,
    ) -> None:
        """A timber deck under the tracks with a railing along each side."""
        spread = (_lane_spread(lanes, cell) + BRIDGE_WIDTH / 2) * cell
        middle = lanes[0] if len(lanes) == 1 else _middle(lanes)
        deck = _raised_strip(middle, lifts, spread + 0.03 * cell)
        paint.polygon(deck, fill=(*BRIDGE_SHADOW, 255))
        paint.polygon(_raised_strip(middle, lifts, spread), fill=(*BRIDGE_DECK, 255))
        for side in (-spread, spread):
            railing = _raised(_offset(middle, side), lifts)
            paint.line(railing, fill=(*RAILING, 255), width=s, joint="curve")

    def _lay_platform(
        self,
        paint: Canvas,
        line: Path,
        lanes: list[Path],
        lifts: list[float],
        side: Point,
        cell: float,
        s: int,
    ) -> None:
        """A paved platform along the station's side of its track, its white edge by the
        track; on quadruple track instead an island platform between the two tracks each way,
        white edges along both."""
        if len(lanes) == 4:
            self._lay_islands(paint, line, lifts, cell, s)
            return
        (x0, y0), (x1, y1) = line[0], line[-1]
        nx, ny = -(y1 - y0), x1 - x0
        sign = 1.0 if nx * side[0] + ny * side[1] > 0 else -1.0
        near = (_lane_spread(lanes, cell) + BANK / 2) * cell
        far = near + PLATFORM_WIDTH * cell
        platform = _raised_strip(line, lifts, sign * near, sign * far)
        paint.polygon(platform, fill=(*PLATFORM, 255))
        edge = _raised(_offset(line, sign * (near + s / 2)), lifts)
        paint.line(edge, fill=(*PLATFORM_EDGE, 255), width=s)

    def _lay_islands(
        self, paint: Canvas, line: Path, lifts: list[float], cell: float, s: int
    ) -> None:
        """The two island platforms of a station on quadruple track, along the middle of its
        cell where the tracks either side of them have opened out."""
        lengths = _lengths(line)
        total = lengths[-1] or 1.0
        keep = [
            k for k, along in enumerate(lengths) if ISLAND_FROM <= along / total <= 1 - ISLAND_FROM
        ]
        if len(keep) < 2:
            return
        middle = [line[k] for k in keep]
        raised = [lifts[k] for k in keep]
        centre = DOUBLE_SPACING / 2 + ISLAND_TRACKS / 2
        half = (ISLAND_TRACKS - TRACK_BED) / 2
        for sign in (-1.0, 1.0):
            near, far = (centre - half) * cell, (centre + half) * cell
            island = _raised_strip(middle, raised, sign * near, sign * far)
            paint.polygon(island, fill=(*PLATFORM, 255))
            for edge in (near + s / 2, far - s / 2):
                paint.line(
                    _raised(_offset(middle, sign * edge), raised),
                    fill=(*PLATFORM_EDGE, 255),
                    width=s,
                )

    def _portals(
        self, laid: _Laid, line: Path, lanes: list[Path], lifts: list[float], cell: float
    ) -> list[_Portal]:
        """The portals of a tunnel piece, at each end where it comes out of the mountains, that
        face the viewer enough to be seen: those facing away open behind the hill in front."""
        grid = self.tile_map.grid
        piece = laid.piece
        mx, my = line[len(line) // 2]
        width = (PORTAL / 2 + _lane_spread(lanes, cell)) * cell
        portals = []
        for point, index in zip(piece.ends, (0, -1), strict=True):
            beyond, _ = grid.track_exit(piece.cell, point)
            if not grid.contains(*beyond) or self.tile_map.relief[beyond] == Relief.MOUNTAINS:
                continue
            ex, ey = line[index]
            length = math.hypot(ex - mx, ey - my) or 1.0
            ux, uy = (ex - mx) / length, (ey - my) / length
            if uy < -PORTAL_FACING:
                continue
            # The portal stands on the near edge of the track's end, as wide as the track is
            # across the picture there.
            base = (ex, ey - lifts[index] + abs(ux) * width)
            half = width * (PORTAL_NARROWEST + (1 - PORTAL_NARROWEST) * abs(uy))
            portals.append(_Portal(base, half, PORTAL_HEIGHT * cell, lifts[index]))
        return portals

    def _hang_wires(
        self,
        paint: Canvas,
        line: Path,
        lanes: list[Path],
        lifts: list[float],
        cell: float,
        s: int,
    ) -> None:
        """A wire over each track, and masts at even steps along the piece on the far side of
        it from the viewer, each with an arm reaching over the track to hold the wires."""
        lengths = _lengths(line)
        count = max(1, round(lengths[-1] / (MAST_GAP * cell)))
        for lane in lanes:
            wire = [
                (x, y - lift - WIRE_HEIGHT * s) for (x, y), lift in zip(lane, lifts, strict=True)
            ]
            paint.line(wire, fill=(*WIRE, 255), width=s)
        for k in range(count):
            along = (k + 0.5) * lengths[-1] / count
            _, (ux, uy) = _point_at(line, lengths, along)
            lift = float(np.interp(along, lengths, lifts))
            far = (-uy, ux) if ux < 0 else (uy, -ux)
            overs = []
            for lane in lanes:
                lane_lengths = _lengths(lane)
                over, _ = _point_at(lane, lane_lengths, along * lane_lengths[-1] / lengths[-1])
                overs.append(over)
            overs.sort(key=lambda over: over[0] * far[0] + over[1] * far[1])
            near, back = overs[0], overs[-1]
            near = (near[0], near[1] - lift)
            bx, by = back[0] + far[0] * MAST_SIDE * cell, back[1] + far[1] * MAST_SIDE * cell - lift
            top = (bx, by - MAST_HEIGHT * s)
            paint.line([(bx, by), top], fill=(*MAST, 255), width=s)
            paint.line([top, (near[0], near[1] - WIRE_HEIGHT * s)], fill=(*MAST, 255), width=s)


@dataclass(frozen=True)
class _Portal:
    """A tunnel portal as drawn: the middle of its foot in the picture, its half width and its
    height in the picture's pixels, and how far it is lifted with the track."""

    base: Point
    half: float
    height: float
    lift: float


def _arch(portal: _Portal, s: int, inset: float = 0.0) -> Path:
    """The outline of a portal standing upright, straight sides under a round top, or of the
    opening `inset` pixels inside it."""
    (x, y), half, height = portal.base, portal.half - inset, portal.height - inset
    round_by = min(half, height * 0.6)
    shoulder = y - (height - round_by)
    top = [
        (x + half * math.cos(angle), shoulder - round_by * math.sin(angle))
        for angle in (math.pi * (1 - k / 8) for k in range(9))
    ]
    return [(x - half, y), *top, (x + half, y)]


def _lay_portal(paint: Canvas, portal: _Portal, s: int) -> None:
    """A stone arch round a dark opening, lit along its top."""
    stone = max(portal.half * 0.3, float(s))
    outline = _arch(portal, s)
    paint.polygon(outline, fill=(*PORTAL_STONE, 255), outline=(*PORTAL_EDGE, 255), width=s)
    paint.line(outline[1:-1], fill=(*PORTAL_LIT, 255), width=s)
    paint.polygon(_arch(portal, s, stone), fill=(*PORTAL_DARK, 255))


def _crisp(rail: Path, smoothing: int, width: int | None = None) -> Path:
    """A rail running straight across or down the picture moved by less than a pixel so its
    edges lie between the rows or columns of pixels it is shown on, `width` of the picture's
    pixels wide (`smoothing` of them to a pixel shown): one shown pixel wide, it lies on the middle
    of one, so it shows as one bright line instead of two dim ones where it would straddle two."""
    wider = 0 if width is None else width - smoothing
    for axis in (1, 0):
        across = [point[axis] for point in rail]
        if max(across) - min(across) > smoothing * STRAIGHT_WITHIN:
            continue
        mean = sum(across) / len(across)
        shift = (math.floor((mean - wider / 2) / smoothing) + 0.5) * smoothing + wider / 2 - mean
        if axis:
            return [(x, y + shift) for x, y in rail]
        return [(x + shift, y) for x, y in rail]
    return rail


def _lane_spread(lanes: list[Path], cell: float) -> float:
    """How far the outer tracks lie from the middle of the piece at their widest, in cells,
    `cell` being a cell's width on the picture."""
    if len(lanes) == 1:
        return 0.0
    return max(math.dist(a, b) for a, b in zip(lanes[0], lanes[-1], strict=True)) / 2 / cell


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


class Canvas(Protocol):
    """What track art draws with: a picture's ImageDraw, or a _Recorder of triangles."""

    def polygon(
        self,
        xy: Sequence[Point],
        fill: float | tuple[int, ...] | None = None,
        outline: tuple[int, ...] | None = None,
        width: int = 1,
    ) -> None: ...

    def line(
        self,
        xy: Sequence[Point],
        fill: tuple[int, ...] | None = None,
        width: int = 0,
        joint: str | None = None,
    ) -> None: ...


# How many floats each vertex of the track's triangles holds: where it lies, then its colour.
TRIANGLE_FLOATS = 6
# How many floats each vertex of the triangles marking how far track is lifted holds: where it
# lies, then the lift.
LIFT_FLOATS = 3


@dataclass(frozen=True)
class TrackShapes:
    """Track as triangles for the graphics card (TrackArt.shapes), laid out `scale` pixels to an
    art pixel and `quality` times finer still from the corner of the chunk: its shadow and then
    the track itself, each vertex with its colour, from 0 to 1, in the order they are drawn, and
    triangles marking how far each stretch is lifted, in art pixels, over all it can cover."""

    triangles: npt.NDArray[np.float32]
    lifts: npt.NDArray[np.float32]
    scale: int
    quality: int

    def packed(self) -> npt.NDArray[np.uint8]:
        """The shapes as bytes in the form of a chunk's pixels, rows of one pixel of four bytes,
        to go wherever drawn chunks go (railroads.ui.pixel.copies)."""
        head = np.array(
            [len(self.triangles), len(self.lifts), self.scale, self.quality], dtype=np.float32
        )
        floats = np.concatenate([head, self.triangles.ravel(), self.lifts.ravel()])
        return floats.astype(np.float32).view(np.uint8).reshape(-1, 1, 4)

    @classmethod
    def unpacked(cls, packed: npt.NDArray[np.uint8]) -> Self:
        floats = np.ascontiguousarray(packed).reshape(-1).view(np.float32)
        triangles, lifts, scale, quality = (int(v) for v in floats[:4])
        at = 4 + triangles * TRIANGLE_FLOATS
        return cls(
            floats[4:at].reshape(-1, TRIANGLE_FLOATS),
            floats[at : at + lifts * LIFT_FLOATS].reshape(-1, LIFT_FLOATS),
            scale,
            quality,
        )


def no_track() -> npt.NDArray[np.uint8]:
    """Shapes (TrackShapes.packed) of a chunk with no track."""
    empty = np.zeros((0, 1), dtype=np.float32)
    return TrackShapes(empty, empty, 0, 0).packed()


class _Recorder:
    """A canvas recording what track art draws as triangles (TrackShapes): its own drawing,
    `shadow`'s and `heights`'."""

    def __init__(self) -> None:
        self._shadow: list[float] = []
        self._track: list[float] = []
        self._lifts: list[float] = []
        self.shadow = _Marks(self._shadow, None)
        self.heights = _Marks(self._lifts, 0.0)

    def polygon(
        self,
        xy: Sequence[Point],
        fill: float | tuple[int, ...] | None = None,
        outline: tuple[int, ...] | None = None,
        width: int = 1,
    ) -> None:
        if outline is not None:
            _coloured(self._track, _triangles(xy), outline)
            xy = _inset(xy, width)
        assert isinstance(fill, tuple)
        _coloured(self._track, _triangles(xy), fill)

    def line(
        self,
        xy: Sequence[Point],
        fill: tuple[int, ...] | None = None,
        width: int = 0,
        joint: str | None = None,
    ) -> None:
        assert fill is not None
        _coloured(self._track, _wide_line(xy, width, joint == "curve"), fill)

    def shapes(self, scale: int, quality: int) -> TrackShapes:
        shadow = np.array(self._shadow, dtype=np.float32).reshape(-1, 2)
        dark = np.zeros((len(shadow), 4), dtype=np.float32)
        dark[:, 3] = SHADOW
        track = np.array(self._track, dtype=np.float32).reshape(-1, TRIANGLE_FLOATS)
        triangles = np.concatenate([np.hstack([shadow, dark]), track])
        lifts = np.array(self._lifts, dtype=np.float32).reshape(-1, LIFT_FLOATS)
        return TrackShapes(triangles, lifts, scale, quality)


class _Marks:
    """A canvas recording only where polygons lie, as triangles, each vertex with the polygon's
    fill when `valued` is not None."""

    def __init__(self, into: list[float], valued: float | None) -> None:
        self._into = into
        self._valued = valued

    def polygon(
        self,
        xy: Sequence[Point],
        fill: float | tuple[int, ...] | None = None,
        outline: tuple[int, ...] | None = None,
        width: int = 1,
    ) -> None:
        del outline, width
        for x, y in _triangles(xy):
            self._into += (x, y) if self._valued is None else (x, y, float(fill or 0.0))

    def line(
        self,
        xy: Sequence[Point],
        fill: tuple[int, ...] | None = None,
        width: int = 0,
        joint: str | None = None,
    ) -> None:
        raise NotImplementedError


def _coloured(into: list[float], points: list[Point], fill: tuple[int, ...]) -> None:
    r, g, b, a = (v / 255 for v in fill)
    for x, y in points:
        into += (x, y, r, g, b, a)


def _triangles(xy: Sequence[Point]) -> list[Point]:
    """The corners of triangles filling a polygon: a strip (_Strip) as the quads between its two
    sides, any other polygon, which track art draws only convex, as a fan."""
    points = list(xy)
    triangles: list[Point] = []
    if isinstance(xy, _Strip):
        half = len(points) // 2
        side, other = points[:half], points[half:][::-1]
        for i in range(half - 1):
            a, b, c, d = side[i], side[i + 1], other[i + 1], other[i]
            triangles += (a, b, c, a, c, d)
        return triangles
    for i in range(1, len(points) - 1):
        triangles += (points[0], points[i], points[i + 1])
    return triangles


def _wide_line(xy: Sequence[Point], width: float, joined: bool) -> list[Point]:
    """The corners of triangles drawing a line `width` wide along the points, its stretches
    joined across the outside of each bend where `joined`."""
    half = width / 2
    triangles: list[Point] = []
    normals: list[tuple[Point, Point] | None] = []
    for (x0, y0), (x1, y1) in itertools.pairwise(xy):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            continue
        nx, ny = -(y1 - y0) / length * half, (x1 - x0) / length * half
        a, b = (x0 + nx, y0 + ny), (x1 + nx, y1 + ny)
        c, d = (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)
        triangles += (a, b, c, a, c, d)
        normals.append(((x1, y1), (nx, ny)))
    if joined:
        for before, after in itertools.pairwise(normals):
            assert before is not None and after is not None
            (x, y), (n0x, n0y) = before
            n1x, n1y = after[1]
            triangles += ((x, y), (x + n0x, y + n0y), (x + n1x, y + n1y))
            triangles += ((x, y), (x - n0x, y - n0y), (x - n1x, y - n1y))
    return triangles


def _inset(xy: Sequence[Point], by: float) -> list[Point]:
    """A convex polygon with each side moved `by` inwards."""
    points = list(xy)
    area = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True)
    )
    closed = [points[-1], *points, points[0]]
    moved = _offset(closed, by if area > 0 else -by)
    return moved[1:-1]
