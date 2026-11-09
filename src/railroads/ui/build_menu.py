"""The build menu: what a left click on the map builds (a station, track), the trains window with
each train's cars and each station's stockpile, and drawing the track and trains."""

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum

from imgui_bundle import imgui

from railroads.sim.constructions import Constructions, Track
from railroads.sim.economy import COLLECTION_RADIUS, DELIVERY_RADIUS, Good
from railroads.sim.grids import Cell, Grid
from railroads.sim.railway import CAR_CAPACITY, MAX_CARS, STOCKPILE, Railway, Stop, Train
from railroads.sim.routes import Route
from railroads.sim.track_rules import TrackBed
from railroads.ui.iso_camera import IsoCamera, Vec2

BALLAST = (0.36, 0.3, 0.24, 1.0)
RAIL = (0.82, 0.8, 0.76, 1.0)
DECK = (0.42, 0.26, 0.12, 1.0)
TUNNEL = (0.18, 0.17, 0.17, 1.0)
PREVIEW = (1.0, 0.85, 0.2, 0.9)
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
CAR_BODIES: dict[Good, Rgba] = {
    Good.FOOD: (0.52, 0.42, 0.22, 1.0),
    Good.LUMBER: (0.36, 0.3, 0.26, 1.0),
    Good.COAL: (0.2, 0.2, 0.22, 1.0),
    Good.STEEL: (0.36, 0.3, 0.26, 1.0),
    Good.GOODS: (0.5, 0.18, 0.14, 1.0),
}
CAR_COLORS: dict[Good, Rgba] = {
    Good.FOOD: (0.95, 0.8, 0.25, 1.0),
    Good.LUMBER: (0.55, 0.35, 0.18, 1.0),
    Good.COAL: (0.25, 0.25, 0.28, 1.0),
    Good.STEEL: (0.55, 0.65, 0.8, 1.0),
    Good.GOODS: (0.85, 0.3, 0.25, 1.0),
}
# Sizes in cells: how far apart the engine and each car are drawn, how long and wide they are,
# and the gap between cars.
CAR_SPACING = 0.36
CAR_LENGTH = 0.34
CAR_WIDTH = 0.16
CAR_GAP = 0.02
OUTLINE = (0.05, 0.05, 0.05, 1.0)
MENU_POS = (12.0, 420.0)
HINT_WIDTH = 360.0
TRAINS_POS = (12.0, 540.0)
TRAINS_SIZE = (520.0, 220.0)


class Tool(Enum):
    NONE = "None"
    STATION = "Station"
    TRACK = "Track"


HINTS = {
    Tool.NONE: "Choose what to build. Right click any cell to see what is there.",
    Tool.STATION: (
        "Click a cell to build a station; what stands there moves aside. Orange: where it hands "
        "goods out; pink: other stations; grey: shared; blue line: where it gathers goods. "
        "Right click cancels."
    ),
    Tool.TRACK: "Click where track starts, then where it goes. Right click cancels.",
}


@dataclass
class BuildMenu:
    tool: Tool = Tool.NONE
    # Where the next piece of track starts, once the first click is made.
    track_from: Cell | None = None
    # The route to the cell under the mouse, worked out when that cell changes.
    preview: Route | None = None
    preview_to: Cell | None = None
    message: str = ""
    laid: list[str] = field(default_factory=list)

    def draw(self) -> None:
        imgui.set_next_window_pos(MENU_POS, imgui.Cond_.first_use_ever.value)
        imgui.begin("Build", flags=imgui.WindowFlags_.always_auto_resize.value)
        for tool in Tool:
            if imgui.radio_button(tool.value, self.tool == tool):
                self.choose(tool)
            imgui.same_line()
        imgui.new_line()
        imgui.push_text_wrap_pos(HINT_WIDTH)
        imgui.text_disabled(HINTS[self.tool])
        imgui.pop_text_wrap_pos()
        if self.message:
            imgui.text(self.message)
        imgui.end()

    def choose(self, tool: Tool) -> None:
        self.tool = tool
        self.stop()

    def stop(self) -> None:
        self.track_from = None
        self.preview = None
        self.preview_to = None


def draw_trains_window(railway: Railway) -> None:
    """The trains window: buy a train, set its stops and cars, or sell it; and what each station
    holds."""
    imgui.set_next_window_pos(TRAINS_POS, imgui.Cond_.first_use_ever.value)
    imgui.set_next_window_size(TRAINS_SIZE, imgui.Cond_.first_use_ever.value)
    imgui.begin("Trains")
    stations = railway.stations()
    if imgui.button("Buy a train"):
        railway.buy()
    if len(stations) < 2:
        imgui.same_line()
        imgui.text_disabled("Build two stations joined by track for it to run.")
    for train in list(railway.trains):
        imgui.push_id(train.name)
        load = _goods_text(train.load)
        header = f"{train.name}: {train.status}" + (f" ({load})" if load else "")
        if imgui.collapsing_header(header, imgui.TreeNodeFlags_.default_open.value):
            _draw_stops(train, stations)
            _draw_cars(train)
            if imgui.button("Sell"):
                railway.sell(train)
        imgui.pop_id()
    if stations and imgui.collapsing_header("Stations", imgui.TreeNodeFlags_.default_open.value):
        for station in stations:
            goods = railway.station_goods(station)
            imgui.bullet_text(
                f"{station_name(station)}: stockpile {goods.stocked:.0f}/{STOCKPILE:.0f}"
            )
            imgui.indent()
            imgui.text_disabled(f"Stockpile: {_goods_text(goods.stockpile) or 'empty'}")
            imgui.text_disabled(f"To hand out: {_goods_text(goods.brought) or 'nothing'}")
            imgui.unindent()
    imgui.end()


def _draw_cars(train: Train) -> None:
    """What the train pulls now and carries."""
    if not train.cars:
        imgui.text_disabled("Pulling no cars.")
        return
    imgui.text("Pulling:")
    for car in train.cars:
        imgui.same_line()
        imgui.color_button(f"##{id(car)}", imgui.ImVec4(*CAR_COLORS[car.good]), 0, (12, 12))
        imgui.same_line()
        imgui.text(f"{car.load:.0f}/{CAR_CAPACITY:.0f}")


def _goods_text(goods: dict[Good, float]) -> str:
    return ", ".join(
        f"{amount:.0f} {good.value}" for good, amount in goods.items() if amount >= 0.5
    )


def _draw_stops(train: Train, stations: list[Cell]) -> None:
    """The stops in order, each with the cars the train pulls on to the next stop."""
    imgui.text("Stops, and the cars it pulls to the next one:")
    for index, stop in enumerate(list(train.stops)):
        imgui.push_id(index)
        following = train.stops[(index + 1) % len(train.stops)] if len(train.stops) > 1 else None
        towards = f" to {station_name(following.cell)}" if following else ""
        imgui.bullet_text(f"{index + 1}. {station_name(stop.cell)}")
        imgui.same_line()
        if imgui.small_button("remove stop"):
            train.stops.pop(index)
            train.restart()
            imgui.pop_id()
            break
        imgui.same_line()
        _, stop.wait_for_load = imgui.checkbox("wait for full load", stop.wait_for_load)
        imgui.indent()
        imgui.text_disabled(f"Cars{towards} ({len(stop.cars)}/{MAX_CARS}):")
        for car_index, good in enumerate(list(stop.cars)):
            imgui.same_line()
            imgui.push_id(car_index)
            imgui.color_button("##car", imgui.ImVec4(*CAR_COLORS[good]), 0, (12, 12))
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"{good.value} car; click to remove")
            if imgui.is_item_clicked():
                stop.cars.pop(car_index)
                train.cars_changed()
            imgui.pop_id()
        if len(stop.cars) < MAX_CARS:
            imgui.same_line()
            imgui.set_next_item_width(130)
            if imgui.begin_combo("##add", "add a car"):
                for good in Good:
                    if imgui.selectable(f"{good.value} car", False)[0] and stop.add_car(good):
                        train.cars_changed()
                imgui.end_combo()
        imgui.unindent()
        imgui.pop_id()
    choices = [s for s in stations if not train.stops or train.stops[-1].cell != s]
    if choices and imgui.begin_combo("Add stop", "choose a station"):
        for station in choices:
            if imgui.selectable(station_name(station), False)[0]:
                cars = list(train.stops[-1].cars) if train.stops else []
                train.stops.append(Stop(station, cars))
                train.restart()
        imgui.end_combo()


def station_name(cell: Cell) -> str:
    return f"Station at {cell[0]}, {cell[1]}"


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
) -> None:
    """Shades the cells stations reach, cell by cell as in Civilization: every station's area
    where it hands goods out, the focused one (being placed or under the mouse) in its own
    colour, cells two stations share in grey, the border of where the focused station gathers
    goods, and the stations' own cells in solid blue."""
    owners: dict[Cell, set[Cell]] = {}
    for station in {*stations, focus}:
        for cell in grid.within(station, DELIVERY_RADIUS):
            owners.setdefault(cell, set()).add(station)
    for cell, by in owners.items():
        fill = SHARED_FILL if len(by) > 1 else FOCUS_FILL if focus in by else OTHER_FILL
        _hex(draw_list, grid, cell, camera, center, fill, CELL_LINE)
    for station in {*stations, focus}:
        _hex(draw_list, grid, station, camera, center, STATION_FILL, CELL_LINE)
    gathered = set(grid.within(focus, COLLECTION_RADIUS))
    for a, b in _border(grid, gathered):
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
) -> None:
    """Draws every piece of track: ballast with a rail on top, a wooden deck on bridges and a dark
    line through tunnels, thicker the closer the view."""
    width = max(1.5, camera.cell_width / 10)
    for built in constructions:
        if isinstance(built, Track):
            _draw_piece(draw_list, built, constructions, camera, center, width)


def draw_route(
    draw_list: imgui.ImDrawList,
    route: Route,
    constructions: Constructions,
    camera: IsoCamera,
    center: Vec2,
) -> None:
    width = max(1.5, camera.cell_width / 12)
    for piece in route.pieces:
        points = _piece_points(piece, constructions, camera, center)
        draw_list.add_polyline(points, _color(PREVIEW), width, 0)


def _draw_piece(
    draw_list: imgui.ImDrawList,
    piece: Track,
    constructions: Constructions,
    camera: IsoCamera,
    center: Vec2,
    width: float,
) -> None:
    points = _piece_points(piece, constructions, camera, center)
    if piece.bed == TrackBed.BRIDGE:
        draw_list.add_polyline(points, _color(DECK), width * 3.0, 0)
    if piece.bed == TrackBed.TUNNEL:
        draw_list.add_polyline(points, _color(TUNNEL), width * 1.6, 0)
        return
    draw_list.add_polyline(points, _color(BALLAST), width * 1.6, 0)
    draw_list.add_polyline(points, _color(RAIL), max(1.0, width * 0.5), 0)


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


def _color(rgba: tuple[float, float, float, float]) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))
