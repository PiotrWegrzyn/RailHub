"""The build bar: what a left click on the map builds (a station, track) and the buttons for the
trains and stations windows; and drawing the track, routes, station reach and far-off trains."""

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.demolition import (
    DEMOLITION_PER_POINT,
    SKYSCRAPER_DEMOLITION,
    STATION_REMOVAL,
)
from railroads.sim.finance import (
    DOUBLE_TRACK_FACTOR,
    ELECTRIC_TRACK_PRICE,
    ELECTRIC_TRACK_UPKEEP,
    STATION_PRICE,
    TRACK_PRICE,
    TRACK_UPKEEP,
    dollars,
)
from railroads.sim.goods import Good
from railroads.sim.grids import Cell, Grid
from railroads.sim.ladders import Points
from railroads.sim.railway import Railway
from railroads.sim.routes import Route
from railroads.sim.tilemap import Relief
from railroads.sim.track_rules import BRIDGE_RULES, RELIEF_RULES, TUNNEL_RULES, TrackBed
from railroads.ui.iso_camera import IsoCamera, Vec2
from railroads.ui.stations_window import StationsWindow
from railroads.ui.theme import BORDER, BRASS, GOOD_COLORS, color_u32, toggle_button
from railroads.ui.trains_window import TrainsWindow

BALLAST = (0.36, 0.3, 0.24, 1.0)
RAIL = (0.82, 0.8, 0.76, 1.0)
DECK = (0.42, 0.26, 0.12, 1.0)
TUNNEL = (0.18, 0.17, 0.17, 1.0)
WIRE = (0.12, 0.14, 0.2, 1.0)
# How far apart the two tracks of double track are drawn, as a share of a cell.
DOUBLE_SPACING = 0.28
# Track not laid yet is drawn in this grey, this see-through.
PLANNED_GREY = 0.82
PLANNED_OPACITY = 0.7
FOCUS_FILL = (0.9, 0.72, 0.45, 0.6)
OTHER_FILL = (0.92, 0.6, 0.65, 0.5)
SHARED_FILL = (0.82, 0.86, 0.82, 0.6)
STATION_FILL = (0.27, 0.35, 0.8, 0.85)
CELL_LINE = (0.25, 0.22, 0.2, 0.5)
COLLECTION_RING = (0.4, 0.6, 1.0, 0.95)
BLOCKED = (0.95, 0.3, 0.25, 0.9)
type Rgba = tuple[float, float, float, float]

ENGINE = (0.14, 0.15, 0.17, 1.0)
CAB = (0.62, 0.14, 0.1, 1.0)
SMOKE = (0.85, 0.85, 0.88, 0.9)
# What each car looks like empty, and the colour of what it carries.
CAR_COLORS: dict[Good, Rgba] = dict(GOOD_COLORS)
CAR_BODIES: dict[Good, Rgba] = {
    good: (r * 0.55, g * 0.55, b * 0.55, 1.0) for good, (r, g, b, _) in GOOD_COLORS.items()
}
# Sizes in cells: how far apart the engine and each car are drawn, how long and wide they are,
# and the gap between cars.
CAR_SPACING = 0.36
CAR_LENGTH = 0.34
CAR_WIDTH = 0.16
CAR_GAP = 0.02
OUTLINE = (0.05, 0.05, 0.05, 1.0)
# How far above the bottom of the map the build bar floats, how wide its hints run, and how long
# a message stays up, in seconds.
BAR_MARGIN = 16.0
HINT_WIDTH = 520.0
MESSAGE_SECONDS = 4.0
BUTTON_HEIGHT = 34.0


class Tool(Enum):
    NONE = "None"
    STATION = "Station"
    TRACK = "Track"
    BULLDOZE = "Bulldoze"
    MOVE_STATION = "Move station"


HILLS_PRICE = TRACK_PRICE * RELIEF_RULES[Relief.HILLS].price_factor
BRIDGE_PRICE = TRACK_PRICE * BRIDGE_RULES.price_factor
TUNNEL_PRICE = TRACK_PRICE * TUNNEL_RULES.price_factor
HINTS = {
    Tool.STATION: (
        f"A station costs {dollars(STATION_PRICE)}, and clearing its cell what the buildings there "
        "cost to take down. Click a cell to build one. Orange: where it hands "
        "goods out; pink: other stations; grey: shared; blue line: where it gathers goods. "
        "Right click cancels."
    ),
    Tool.TRACK: (
        f"Click where track starts, then where it goes; starting or ending on track joins it. "
        f"{dollars(TRACK_PRICE)} a cell on the flat, {dollars(HILLS_PRICE)} in hills, "
        f"{dollars(BRIDGE_PRICE)} a bridge, {dollars(TUNNEL_PRICE)} a tunnel; double track "
        f"{DOUBLE_TRACK_FACTOR:g} times as much, and electric {dollars(ELECTRIC_TRACK_PRICE)} "
        f"a cell more. Laying it along your track widens or electrifies that. Buildings in the way "
        "cost what taking them down does. Right click cancels."
    ),
    Tool.BULLDOZE: (
        f"Click a cell to clear it: homes from {dollars(DEMOLITION_PER_POINT[Points.POPULATION])}, "
        f"fields {dollars(DEMOLITION_PER_POINT[Points.FOOD])}, factories and mines from "
        f"{dollars(DEMOLITION_PER_POINT[Points.INDUSTRY])}, a skyscraper "
        f"{dollars(SKYSCRAPER_DEMOLITION)}; track a quarter of its price, a station "
        f"{dollars(STATION_REMOVAL)}. One-off buildings stay. Right click cancels."
    ),
    Tool.MOVE_STATION: (
        f"Click where the station goes: {dollars(STATION_PRICE)} and clearing the cell. Its trains, "
        "name and goods go with it; its level and storage start again. Right click cancels."
    ),
}
TOOL_BUTTONS = (
    (Tool.STATION, f"{fa.ICON_FA_BUILDING}  Station", "Build a station (1)"),
    (Tool.TRACK, f"{fa.ICON_FA_ROAD}  Track", "Lay track (2)"),
    (Tool.BULLDOZE, f"{fa.ICON_FA_TRASH}  Bulldoze", "Clear cells, for a price (3)"),
)
WIDTH_BUTTONS = (
    (
        TrackWidth.SINGLE,
        "Single",
        f"One track: {dollars(TRACK_PRICE)} a cell, {dollars(TRACK_UPKEEP)} a month",
    ),
    (
        TrackWidth.DOUBLE,
        "Double",
        (
            f"Two tracks side by side: {dollars(TRACK_PRICE * DOUBLE_TRACK_FACTOR)} a cell, "
            f"${TRACK_UPKEEP * DOUBLE_TRACK_FACTOR:.2f} a month"
        ),
    ),
)
ELECTRIC_TIP = (
    f"Wires over the track for electric engines: {dollars(ELECTRIC_TRACK_PRICE)} a cell more, "
    f"{dollars(ELECTRIC_TRACK_UPKEEP)} a month more ({DOUBLE_TRACK_FACTOR:g} times that on double "
    "track). Lay it over your track to electrify that."
)


@dataclass
class BuildMenu:
    """The build bar along the bottom of the map: what a left click builds, and the buttons that
    open the trains and stations windows."""

    tool: Tool = Tool.NONE
    # Where the next piece of track starts, once the first click is made.
    track_from: Cell | None = None
    # The route to the cell under the mouse, worked out when that cell changes.
    preview: Route | None = None
    preview_to: Cell | None = None
    message: str = ""
    message_at: float = -MESSAGE_SECONDS
    # What the previewed track would cost.
    preview_price: float | None = None
    # What the tool would cost in the cell under the mouse, and what it would do there.
    hover_note: str = ""
    # The station being moved, with the move tool.
    moving: Cell | None = None
    # Which of the grid's ways the player turned the station being placed to, by its place in
    # MapTools.station_ways; None while it takes the way of the cell under the pointer.
    station_turn: int | None = None
    # What the track tool lays: single or double track, and whether it is electric.
    track_width: TrackWidth = TrackWidth.SINGLE
    electric: bool = False
    trains: TrainsWindow = field(default_factory=TrainsWindow)
    stations: StationsWindow = field(default_factory=StationsWindow)

    def draw(self, top_left: Vec2, bottom_right: Vec2) -> None:
        x = (top_left[0] + bottom_right[0]) / 2
        imgui.set_next_window_pos(
            (x, bottom_right[1] - BAR_MARGIN), imgui.Cond_.always.value, (0.5, 1.0)
        )
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_nav.value
        )
        imgui.push_style_var(imgui.StyleVar_.window_rounding.value, 12.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding.value, imgui.ImVec2(10, 8))
        imgui.begin("##build bar", flags=flags)
        self._draw_hint()
        for tool, label, tooltip in TOOL_BUTTONS:
            if toggle_button(label, self.tool == tool, tooltip, (0, BUTTON_HEIGHT)):
                self.choose(Tool.NONE if self.tool == tool else tool)
            imgui.same_line()
        if self.tool == Tool.TRACK:
            self._draw_track_options()
        _vertical_rule()
        imgui.same_line()
        if toggle_button(
            f"{fa.ICON_FA_TRAIN}  Trains", self.trains.open, "Your trains (T)", (0, BUTTON_HEIGHT)
        ):
            self.trains.open = not self.trains.open
        imgui.same_line()
        if toggle_button(
            f"{fa.ICON_FA_CUBES}  Stations",
            self.stations.open,
            "What your stations hold (S)",
            (0, BUTTON_HEIGHT),
        ):
            self.stations.open = not self.stations.open
        imgui.end()
        imgui.pop_style_var(2)

    def _draw_track_options(self) -> None:
        _vertical_rule()
        imgui.same_line()
        for width, label, tooltip in WIDTH_BUTTONS:
            if toggle_button(label, self.track_width == width, tooltip, (0, BUTTON_HEIGHT)):
                self.lay(width, self.electric)
            imgui.same_line()
        label = f"{fa.ICON_FA_BOLT}  Electric"
        if toggle_button(label, self.electric, ELECTRIC_TIP, (0, BUTTON_HEIGHT)):
            self.lay(self.track_width, not self.electric)
        imgui.same_line()

    def lay(self, width: TrackWidth, electric: bool) -> None:
        """Sets what the track tool lays, and plans the previewed track again."""
        self.track_width, self.electric = width, electric
        self.preview_to = None

    def _draw_hint(self) -> None:
        recent = time.monotonic() - self.message_at < MESSAGE_SECONDS
        if not recent and self.tool not in HINTS:
            return
        imgui.push_text_wrap_pos(HINT_WIDTH)
        if recent:
            imgui.text_colored(imgui.ImVec4(*BRASS), self.message)
        if self.tool in HINTS:
            imgui.text_disabled(HINTS[self.tool])
        if self.tool == Tool.TRACK and self.preview is not None and self.preview_price is not None:
            imgui.text_colored(imgui.ImVec4(*BRASS), f"This track: {dollars(self.preview_price)}")
        if self.hover_note and self.tool in (Tool.STATION, Tool.BULLDOZE, Tool.MOVE_STATION):
            imgui.text_colored(imgui.ImVec4(*BRASS), self.hover_note)
        imgui.pop_text_wrap_pos()
        imgui.spacing()

    def say(self, message: str) -> None:
        """Shows a message above the build bar for a few seconds."""
        self.message = message
        self.message_at = time.monotonic()

    def choose(self, tool: Tool) -> None:
        self.tool = tool
        self.stop()

    def stop(self) -> None:
        self.moving = None
        self.station_turn = None
        self.hover_note = ""
        self.track_from = None
        self.preview = None
        self.preview_to = None
        self.preview_price = None


def _vertical_rule() -> None:
    """A thin upright line between groups of buttons."""
    position = imgui.get_cursor_screen_pos()
    imgui.dummy((1, BUTTON_HEIGHT))
    imgui.get_window_draw_list().add_line(
        (position.x, position.y + 4),
        (position.x, position.y + BUTTON_HEIGHT - 4),
        color_u32(BORDER),
        1.0,
    )


def draw_trains(
    draw_list: imgui.ImDrawList, railway: Railway, camera: IsoCamera, center: Vec2
) -> None:
    """Draws each running train along the track: the engine in front, then its cars, each a box
    turned to the track and shaped by its good; empty cars show their bare bodies."""
    for train in railway.trains:
        parts: list[Good | None] = [None, *(car.good for car in train.cars)]
        loads = [1.0, *(car.load for car in train.cars)]
        for index in reversed(range(len(parts))):
            placed = railway.position(train, back=index * CAR_SPACING)
            if placed is None:
                continue
            (x, y), (dx, dy) = placed
            length = math.hypot(dx, dy) or 1.0
            box = _Box(x, y, dx / length, dy / length, camera, center)
            if parts[index] is None:
                _draw_engine(draw_list, box)
            else:
                _draw_car(draw_list, box, parts[index], loads[index] > 0)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _Box:
    """A spot on the track and the way it runs, for drawing shapes along it in screen space."""

    x: float
    y: float
    ux: float
    uy: float
    camera: IsoCamera
    center: Vec2

    def quad(self, front: float, back: float, half_width: float) -> list[imgui.ImVec2]:
        """The corners of a box from `back` to `front` along the track (cells from this spot,
        forward positive), `half_width` cells either side."""
        sx, sy = -self.uy * half_width, self.ux * half_width
        corners = [
            (self.x + self.ux * front + sx, self.y + self.uy * front + sy),
            (self.x + self.ux * front - sx, self.y + self.uy * front - sy),
            (self.x + self.ux * back - sx, self.y + self.uy * back - sy),
            (self.x + self.ux * back + sx, self.y + self.uy * back + sy),
        ]
        return [imgui.ImVec2(*self.camera.to_screen(px, py, self.center)) for px, py in corners]

    def point(self, along: float, side: float = 0.0) -> imgui.ImVec2:
        px = self.x + self.ux * along - self.uy * side
        py = self.y + self.uy * along + self.ux * side
        return imgui.ImVec2(*self.camera.to_screen(px, py, self.center))

    @property
    def scale(self) -> float:
        return self.camera.cell_width


def _fill(draw_list: imgui.ImDrawList, quad: list[imgui.ImVec2], color: Rgba) -> None:
    draw_list.add_quad_filled(*quad, _color(color))
    draw_list.add_quad(*quad, _color(OUTLINE), 1.0)


def _draw_engine(draw_list: imgui.ImDrawList, box: _Box) -> None:
    half = CAR_LENGTH / 2
    _fill(draw_list, box.quad(half, -half, CAR_WIDTH * 0.55), ENGINE)
    _fill(draw_list, box.quad(-half * 0.25, -half, CAR_WIDTH * 0.7), CAB)
    chimney = box.point(half * 0.65)
    draw_list.add_circle_filled(chimney, max(1.5, box.scale * 0.035), _color(OUTLINE))
    draw_list.add_circle_filled(
        box.point(half * 0.65 + 0.06, 0.05), max(1.5, box.scale * 0.03), _color(SMOKE)
    )


def _draw_car(draw_list: imgui.ImDrawList, box: _Box, good: Good, full: bool) -> None:
    half = CAR_LENGTH / 2 - CAR_GAP
    body = CAR_BODIES[good]
    _fill(draw_list, box.quad(half, -half, CAR_WIDTH * 0.6), body)
    if not full:
        return
    load = CAR_COLORS[good]
    match good:
        case Good.LUMBER | Good.STEEL:
            for side in (-0.3, 0.0, 0.3):
                a = box.point(half * 0.9, CAR_WIDTH * side)
                b = box.point(-half * 0.9, CAR_WIDTH * side)
                draw_list.add_line(a, b, _color(load), max(1.5, box.scale * 0.03))
        case Good.GOODS:
            _fill(draw_list, box.quad(half * 0.9, -half * 0.9, CAR_WIDTH * 0.5), load)
        case _:
            draw_list.add_quad_filled(
                *box.quad(half * 0.8, -half * 0.8, CAR_WIDTH * 0.4), _color(load)
            )


def draw_station_reach(
    draw_list: imgui.ImDrawList,
    grid: Grid,
    focus: Cell,
    stations: list[Cell],
    camera: IsoCamera,
    center: Vec2,
    reach: Callable[[Cell], tuple[int, int]],
) -> None:
    """Shades the cells stations reach, cell by cell as in Civilization: every station's area
    where it hands goods out, the focused one (being placed, under the mouse or open) in its own
    colour, cells two stations share in grey, the border of where the focused station gathers
    goods, and the stations' own cells in solid blue. `reach` says how far each station hands
    goods out and gathers them."""
    owners: dict[Cell, set[Cell]] = {}
    for station in {*stations, focus}:
        for cell in grid.within(station, reach(station)[0]):
            owners.setdefault(cell, set()).add(station)
    for cell, by in owners.items():
        fill = SHARED_FILL if len(by) > 1 else FOCUS_FILL if focus in by else OTHER_FILL
        _hex(draw_list, grid, cell, camera, center, fill, CELL_LINE)
    for station in {*stations, focus}:
        _hex(draw_list, grid, station, camera, center, STATION_FILL, CELL_LINE)
    for a, b in _reach_border(grid, focus, reach(focus)[1]):
        draw_list.add_line(
            imgui.ImVec2(*camera.to_screen(*a, center)),
            imgui.ImVec2(*camera.to_screen(*b, center)),
            _color(COLLECTION_RING),
            2.5,
        )


def _hex(
    draw_list: imgui.ImDrawList,
    grid: Grid,
    cell: Cell,
    camera: IsoCamera,
    center: Vec2,
    fill: Rgba,
    line: Rgba,
) -> None:
    points = [imgui.ImVec2(*camera.to_screen(x, y, center)) for x, y in grid.corners(*cell)]
    draw_list.add_convex_poly_filled(points, _color(fill))
    draw_list.add_polyline(points, _color(line), 1.0, imgui.ImDrawFlags_.closed.value)


@lru_cache(maxsize=64)
def _reach_border(
    grid: Grid, station: Cell, steps: int
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """The border of the cells within `steps` of a station, worked out once: an open station
    window draws it every frame, and finding it takes hundreds of lookups."""
    return tuple(_border(grid, set(grid.within(station, steps))))


def _border(grid: Grid, cells: set[Cell]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The edges of these cells that face a cell outside them."""
    edges = []
    for row, col in cells:
        corners = [(float(x), float(y)) for x, y in grid.corners(row, col)]
        x, y = grid.center(row, col)
        for a, b in itertools.pairwise([*corners, corners[0]]):
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            across = grid.cell_at(2 * mid[0] - x, 2 * mid[1] - y)
            if (int(across[0]), int(across[1])) not in cells:
                edges.append((a, b))
    return edges


def draw_track(
    draw_list: imgui.ImDrawList,
    constructions: Constructions,
    camera: IsoCamera,
    center: Vec2,
    planned: list[Track] | None = None,
) -> None:
    """Draws every piece of track, or only the planned pieces greyed out: ballast with a rail on
    top, two rails on double track, a wooden deck on bridges and a dark line through tunnels, with
    a wire beside electric track, thicker the closer the view. All the beds go down before any
    rail, so pieces that meet in a cell make one junction or crossing."""
    width = max(1.5, camera.cell_width / 10)
    laid = constructions.track()
    pieces = [
        (piece, _piece_points(piece, constructions, camera, center))
        for piece in (laid if planned is None else planned)
    ]
    shade = _color if planned is None else _greyed
    for piece, points in pieces:
        bed = width * (1.6 + (1.4 if piece.width == TrackWidth.DOUBLE else 0.0))
        if piece.bed == TrackBed.BRIDGE:
            draw_list.add_polyline(points, shade(DECK), bed + width * 1.4, 0)
        if piece.bed == TrackBed.TUNNEL:
            draw_list.add_polyline(points, shade(TUNNEL), bed, 0)
        else:
            draw_list.add_polyline(points, shade(BALLAST), bed, 0)
    spacing = camera.cell_width * DOUBLE_SPACING / 2
    for piece, points in pieces:
        if piece.bed == TrackBed.TUNNEL:
            continue
        sides = (-spacing, spacing) if piece.width == TrackWidth.DOUBLE else (0.0,)
        for side in sides:
            draw_list.add_polyline(_parallel(points, side), shade(RAIL), max(1.0, width * 0.5), 0)
    for piece, points in pieces:
        if piece.electric and piece.bed != TrackBed.TUNNEL:
            lifted = [imgui.ImVec2(point.x, point.y - width * 1.2) for point in points]
            draw_list.add_polyline(lifted, shade(WIRE), max(1.0, width * 0.35), 0)


def _greyed(rgba: tuple[float, float, float, float]) -> int:
    """A colour faded to a see-through grey, for track not laid yet."""
    r, g, b, a = rgba
    light = 0.3 * r + 0.59 * g + 0.11 * b
    grey = 0.4 * light + 0.6 * PLANNED_GREY
    return _color((grey, grey, grey, a * PLANNED_OPACITY))


def _piece_points(
    piece: Track, constructions: Constructions, camera: IsoCamera, center: Vec2
) -> list[imgui.ImVec2]:
    grid = constructions.tile_map.grid
    cx, cy = grid.center(*piece.cell)
    # Halfway along each step, where the piece meets the one in the next cell.
    offsets = grid.track_steps() / 2
    start, end = (offsets[point] for point in piece.ends)
    plane = [(cx + start[0], cy + start[1]), (cx, cy), (cx + end[0], cy + end[1])]
    return [imgui.ImVec2(*camera.to_screen(x, y, center)) for x, y in plane]


def _parallel(points: list[imgui.ImVec2], side: float) -> list[imgui.ImVec2]:
    """The line moved `side` screen pixels to its left, joined where the moved lines meet."""
    if not side:
        return points
    normals = []
    for a, b in itertools.pairwise(points):
        length = math.hypot(b.x - a.x, b.y - a.y) or 1.0
        normals.append((-(b.y - a.y) / length, (b.x - a.x) / length))
    moved = []
    for index, point in enumerate(points):
        before = normals[max(0, index - 1)]
        after = normals[min(len(normals) - 1, index)]
        mx, my = before[0] + after[0], before[1] + after[1]
        dot = (mx * before[0] + my * before[1]) or 1.0
        moved.append(imgui.ImVec2(point.x + mx / dot * side, point.y + my / dot * side))
    return moved


def _color(rgba: tuple[float, float, float, float]) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))
