"""The build bar: what a left click on the map builds (a station, track) and the buttons for the
trains and stations windows; and drawing the track, routes, station reach and far-off trains."""

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import numpy as np
from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.kept import KeptFor
from railroads.sim.anger import ANGER_PER_BUILDING, ANGER_SURCHARGE
from railroads.sim.constructions import Constructions, Facility, Track, TrackWidth
from railroads.sim.demolition import (
    DEMOLITION_PER_POINT,
    SKYSCRAPER_DEMOLITION,
    STATION_REMOVAL,
    TRACK_REMOVAL_SHARE,
)
from railroads.sim.facilities import REACH as FACILITY_REACH
from railroads.sim.finance import (
    ELECTRIC_TRACK_FACTOR,
    ELECTRIC_TRACK_UPKEEP,
    MAGLEV_TRACK_PRICE,
    MAGLEV_TRACK_UPKEEP,
    STATION_PRICE,
    TRACK_PRICE,
    TRACK_UPKEEP,
    TRACK_WIDTH_FACTORS,
    TRACK_WIDTH_UPKEEP_FACTORS,
)
from railroads.sim.grids import Cell, FloatArray, Grid
from railroads.sim.points import Points
from railroads.sim.railway import PLATFORM_REACH, QUADRUPLE_ERA
from railroads.sim.routes import Route
from railroads.sim.tilemap import Relief
from railroads.sim.tiles import bridge_opened_from
from railroads.sim.track_rules import (
    BRIDGES,
    RAMP_PRICE_FACTOR,
    RAMP_RULES,
    RELIEF_RULES,
    STREAM_FACTOR,
    TUNNEL_RULES,
    Bridge,
    TrackBed,
)
from railroads.ui.iso_camera import IsoCamera, Vec2
from railroads.ui.ornate import DEEP, bar_buttons, diamond, frame_window
from railroads.ui.spotlight import spot
from railroads.ui.stations_window import StationsWindow
from railroads.ui.theme import (
    BRASS,
    OVERLAY_FLAGS,
    Rgba,
    colored,
    rgba_u32,
    tip,
    toggle_button,
)
from railroads.ui.trains_window import TrainsWindow
from railroads.words import dollars, share_of, times

BALLAST = (0.36, 0.3, 0.24, 1.0)
RAIL = (0.82, 0.8, 0.76, 1.0)
DECK = (0.42, 0.26, 0.12, 1.0)
TUNNEL = (0.18, 0.17, 0.17, 1.0)
WIRE = (0.12, 0.14, 0.2, 1.0)
GUIDEWAY = (0.74, 0.74, 0.72, 1.0)
GUIDEWAY_EDGE = (0.42, 0.42, 0.44, 1.0)
# How far apart the two tracks of double track are drawn zoomed out, as a share of a cell.
FLAT_DOUBLE_SPACING = 0.28
# Track not laid yet is drawn in this grey, this see-through.
PLANNED_GREY = 0.82
PLANNED_OPACITY = 0.7
FOCUS_FILL = (0.9, 0.72, 0.45, 0.6)
OTHER_FILL = (0.92, 0.6, 0.65, 0.5)
SHARED_FILL = (0.82, 0.86, 0.82, 0.6)
STATION_FILL = (0.27, 0.35, 0.8, 0.85)
CELL_LINE = (0.25, 0.22, 0.2, 0.5)
COLLECTION_RING = (0.4, 0.6, 1.0, 0.95)

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
    ADD_PLATFORM = "Add platform"
    ADD_FACILITY = "Add facility"


# The tools that place a station or a platform, which Q or the mouse wheel turns.
STATION_TOOLS = frozenset((Tool.STATION, Tool.MOVE_STATION, Tool.ADD_PLATFORM))
# The tools that note what they would cost in the cell under the mouse.
PRICED_TOOLS = STATION_TOOLS | {Tool.BULLDOZE, Tool.ADD_FACILITY}

HILLS_PRICE = TRACK_PRICE * RELIEF_RULES[Relief.HILLS].price_factor
BRIDGE_PRICE = TRACK_PRICE * BRIDGES[Bridge.WOOD].price_factor
TUNNEL_PRICE = TRACK_PRICE * TUNNEL_RULES.price_factor
HINTS = {
    Tool.STATION: (
        f"A station costs {dollars(STATION_PRICE)}, and clearing its cell what moving the buildings "
        "there aside costs. Click a cell to build one. Orange: where it hands "
        "goods out; pink: other stations; grey: shared; blue line: where it gathers goods. "
        "Right click cancels."
    ),
    Tool.TRACK: (
        f"Click where track starts, then where it goes; starting or ending on track joins it. "
        f"{dollars(TRACK_PRICE)} a cell on the flat, {dollars(HILLS_PRICE)} in hills, {times(STREAM_FACTOR)} that "
        f"across a minor river, {dollars(BRIDGE_PRICE)} a wooden bridge and more for newer ones, "
        f"{dollars(TUNNEL_PRICE)} a tunnel; "
        f"double track {TRACK_WIDTH_FACTORS[TrackWidth.DOUBLE]:g} times as much, quadruple "
        f"{TRACK_WIDTH_FACTORS[TrackWidth.QUADRUPLE]:g} times, and wires "
        f"{ELECTRIC_TRACK_FACTOR - 1:g} times a single track's price again for each track. Laying it "
        "along your track widens or electrifies that. Buildings in the way move aside, from "
        f"{dollars(DEMOLITION_PER_POINT[Points.POPULATION])} a house or field, and {share_of(ANGER_SURCHARGE)} that "
        "again for each point of anger people feel at your company, your own buildings "
        "excepted. Right click cancels."
    ),
    Tool.BULLDOZE: (
        f"Click a cell to clear it: homes from {dollars(DEMOLITION_PER_POINT[Points.POPULATION])}, "
        f"fields {dollars(DEMOLITION_PER_POINT[Points.FOOD])}, factories and mines from "
        f"{dollars(DEMOLITION_PER_POINT[Points.INDUSTRY])}, a skyscraper "
        f"{dollars(SKYSCRAPER_DEMOLITION)}, each not your own {share_of(ANGER_SURCHARGE)} that again for each point "
        f"of anger people feel at your company, and raising it; track {share_of(TRACK_REMOVAL_SHARE)} "
        f"of its price, a station {dollars(STATION_REMOVAL)}. One-off buildings stay. Right click cancels."
    ),
    Tool.ADD_PLATFORM: (
        f"Click a free cell up to {PLATFORM_REACH} cells from the station for its new platform; "
        "Q or the mouse wheel turns its track. Clearing the cell costs what moving the buildings "
        "there aside does. Right click cancels."
    ),
    Tool.ADD_FACILITY: (
        f"Click a cell up to {FACILITY_REACH} cells from the station or a platform for its "
        "engine shed: it fills the cell, "
        "taking the place of any track there with its own, which Q or the mouse wheel turns and "
        "track is laid to as to a station; trains waiting for a load wait in it. "
        "Clearing it costs what moving the buildings in the way aside does. Right click cancels."
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
            f"Two tracks side by side: "
            f"{dollars(TRACK_PRICE * TRACK_WIDTH_FACTORS[TrackWidth.DOUBLE])} a cell, "
            f"{dollars(TRACK_UPKEEP * TRACK_WIDTH_UPKEEP_FACTORS[TrackWidth.DOUBLE], cents=True)} a month"
        ),
    ),
    (
        TrackWidth.QUADRUPLE,
        "Quadruple",
        (
            f"Four tracks, two each way, from era {QUADRUPLE_ERA}: fast trains keep to the inner "
            f"pair, slow ones to the outer, and each pair runs into its own platform where a "
            f"station has one beside it: "
            f"{dollars(TRACK_PRICE * TRACK_WIDTH_FACTORS[TrackWidth.QUADRUPLE])} a cell, "
            f"{dollars(TRACK_UPKEEP * TRACK_WIDTH_UPKEEP_FACTORS[TrackWidth.QUADRUPLE], cents=True)} a month"
        ),
    ),
)
ELECTRIC_TIP = (
    f"Wires over the track for electric engines, once electric engines are on sale: "
    f"{ELECTRIC_TRACK_FACTOR - 1:g} times a single track's price more for each track, "
    f"{dollars(ELECTRIC_TRACK_UPKEEP)} a month more for each track. Lay it over your track to "
    "electrify that."
)
AUTO_BRIDGE_TIP = (
    "Rivers and lakes are crossed on the cheapest bridge that carries the track; track is "
    "crossed on the level."
)


def bridge_tip(bridge: Bridge) -> str:
    """What a kind of bridge is like, for its tooltip."""
    kind = BRIDGES[bridge]
    most = "one track" if kind.tracks == 1 else f"up to {kind.tracks} tracks"
    text = (
        f"A {kind.name} bridge: {most}, {dollars(TRACK_PRICE * kind.price_factor)} a cell for "
        f"one track, trains at {kind.speed_factor:.0%} of their speed."
    )
    if kind.over_track:
        ramps = dollars(TRACK_PRICE * RAMP_PRICE_FACTOR)
        text += (
            f" Laid across your track it goes over it as a flyover, so trains on both run at "
            f"once; the ramps on both sides cost {ramps} more a track and slow trains to "
            f"{RAMP_RULES.speed_factor:.0%}."
        )
    else:
        text += " It goes only over rivers and lakes."
    if kind.maglev:
        text += " The only bridge for maglev guideways."
    elif kind.electric:
        text += " Takes overhead wires."
    opened = bridge_opened_from(bridge)
    if opened:
        text += f" Comes with {opened}."
    return text


MAGLEV_TIP = (
    f"A concrete guideway for maglev engines, which run on nothing else, and nothing else runs "
    f"on it: {dollars(MAGLEV_TRACK_PRICE)} a cell more, {dollars(MAGLEV_TRACK_UPKEEP)} a month "
    f"more ({TRACK_WIDTH_FACTORS[TrackWidth.DOUBLE]:g} times that on double track). Laid over "
    "your track it takes "
    "the place of the rails."
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
    # What the previewed route was planned for: where it runs, the track it lays and the
    # constructions' version then.
    preview_for: tuple[object, ...] = ()
    message: str = ""
    message_at: float = -MESSAGE_SECONDS
    # What the previewed track would cost, and how many buildings it would move aside.
    preview_price: float | None = None
    preview_moved: int = 0
    preview_angering: int = 0
    # What the tool would cost in the cell under the mouse, and what it would do there.
    hover_note: str = ""
    # The station being moved, with the move tool, or added to.
    moving: Cell | None = None
    # The facility being put up beside it.
    facility: Facility | None = None
    # Which of the grid's ways the player turned the station being placed to, by its place in
    # MapTools.station_ways; None while it takes the way of the cell under the pointer.
    station_turn: int | None = None
    # What the track tool lays: single or double track, and whether it is electric or maglev.
    track_width: TrackWidth = TrackWidth.SINGLE
    electric: bool = False
    maglev: bool = False
    # Whether maglev track can be laid yet (Railway.maglev_open).
    maglev_open: bool = False
    # Whether electric track can be laid yet (Railway.electric_open).
    electric_open: bool = False
    # Whether quadruple track can be laid yet (Railway.quadruple_open).
    quadruple_open: bool = False
    # The kind of bridge the track tool builds, None for the cheapest that carries the track,
    # and the kinds that can be built yet (Railway.bridges_open).
    bridge: Bridge | None = None
    bridges_open: tuple[Bridge, ...] = (Bridge.WOOD,)
    trains: TrainsWindow = field(default_factory=TrainsWindow)
    stations: StationsWindow = field(default_factory=StationsWindow)

    def draw(self, top_left: Vec2, bottom_right: Vec2) -> None:
        x = (top_left[0] + bottom_right[0]) / 2
        imgui.set_next_window_pos(
            (x, bottom_right[1] - BAR_MARGIN), imgui.Cond_.always.value, (0.5, 1.0)
        )
        flags = OVERLAY_FLAGS | imgui.WindowFlags_.no_move.value
        imgui.push_style_var(imgui.StyleVar_.window_rounding.value, 12.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding.value, imgui.ImVec2(10, 8))
        imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(*DEEP))
        imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(0, 0, 0, 0))
        imgui.begin("##build bar", flags=flags)
        imgui.pop_style_color(2)
        frame_window()
        with bar_buttons():
            self._draw_bar()
        imgui.end()
        imgui.pop_style_var(2)

    def _draw_bar(self) -> None:
        self._draw_hint()
        for tool, label, tooltip in TOOL_BUTTONS:
            if toggle_button(label, self.tool == tool, tooltip, (0, BUTTON_HEIGHT)):
                self.choose(Tool.NONE if self.tool == tool else tool)
            spot(f"tool:{tool.value}")
            imgui.same_line()
        if self.tool == Tool.TRACK:
            self._draw_track_options()
        _vertical_rule()
        imgui.same_line()
        if toggle_button(
            f"{fa.ICON_FA_TRAIN}  Trains", self.trains.open, "Your trains (T)", (0, BUTTON_HEIGHT)
        ):
            self.toggle_trains()
        spot("trains")
        imgui.same_line()
        if toggle_button(
            f"{fa.ICON_FA_CUBES}  Stations",
            self.stations.open,
            "What your stations hold (S)",
            (0, BUTTON_HEIGHT),
        ):
            self.stations.open = not self.stations.open

    def _draw_track_options(self) -> None:
        _vertical_rule()
        imgui.same_line()
        if not self.quadruple_open and self.track_width == TrackWidth.QUADRUPLE:
            self.lay(TrackWidth.DOUBLE, self.electric, self.maglev)
        for width, label, tooltip in WIDTH_BUTTONS:
            if width == TrackWidth.QUADRUPLE and not self.quadruple_open:
                continue
            if toggle_button(label, self.track_width == width, tooltip, (0, BUTTON_HEIGHT)):
                self.lay(width, self.electric, self.maglev)
            imgui.same_line()
        if self.electric_open:
            label = f"{fa.ICON_FA_BOLT}  Electric"
            if toggle_button(label, self.electric, ELECTRIC_TIP, (0, BUTTON_HEIGHT)):
                self.lay(self.track_width, not self.electric)
            imgui.same_line()
        elif self.electric:
            self.lay(self.track_width, False, self.maglev)
        if not self.maglev_open:
            if self.maglev:
                self.lay(self.track_width, self.electric)
        else:
            label = f"{fa.ICON_FA_MAGNET}  Maglev"
            if toggle_button(label, self.maglev, MAGLEV_TIP, (0, BUTTON_HEIGHT)):
                self.lay(self.track_width, False, not self.maglev)
            imgui.same_line()
        self._draw_bridge_picker()

    def _draw_bridge_picker(self) -> None:
        """The kind of bridge the track tool builds, a button like the bar's others opening a list
        of Auto and every kind, those not built yet greyed out (their tips say what brings them)."""
        if self.bridge is not None and self.bridge not in self.bridges_open:
            self.pick_bridge(None)
        shown = "Auto" if self.bridge is None else BRIDGES[self.bridge].name.capitalize()
        picking = imgui.is_popup_open("##bridges")
        tooltip = AUTO_BRIDGE_TIP if self.bridge is None else bridge_tip(self.bridge)
        label = f"{fa.ICON_FA_ROAD}  Bridge: {shown}"
        if toggle_button(label, picking, tooltip, (0, BUTTON_HEIGHT)):
            imgui.open_popup("##bridges")
        if imgui.begin_popup("##bridges"):
            if imgui.selectable("Auto", self.bridge is None)[0]:
                self.pick_bridge(None)
            tip(AUTO_BRIDGE_TIP)
            for bridge in Bridge:
                can = bridge in self.bridges_open
                flags = 0 if can else imgui.SelectableFlags_.disabled.value
                name = BRIDGES[bridge].name.capitalize()
                if imgui.selectable(f"{name}##{bridge.name}", self.bridge == bridge, flags)[0]:
                    self.pick_bridge(bridge)
                tip(bridge_tip(bridge), disabled=True)
            imgui.end_popup()
        imgui.same_line()

    def pick_bridge(self, bridge: Bridge | None) -> None:
        """Sets the kind of bridge the track tool builds, and plans the previewed track again."""
        self.bridge = bridge
        self.preview_to = None

    def lay(self, width: TrackWidth, electric: bool, maglev: bool = False) -> None:
        """Sets what the track tool lays, and plans the previewed track again. Maglev track
        carries no wires, so choosing one leaves out the other."""
        self.track_width, self.electric, self.maglev = width, electric and not maglev, maglev
        self.preview_to = None

    def _draw_hint(self) -> None:
        recent = time.monotonic() - self.message_at < MESSAGE_SECONDS
        if not recent and self.tool not in HINTS:
            return
        imgui.push_text_wrap_pos(HINT_WIDTH)
        if recent:
            colored(BRASS, self.message)
        if self.tool in HINTS:
            imgui.text_disabled(HINTS[self.tool])
        if self.tool == Tool.TRACK and self.preview is not None and self.preview_price is not None:
            moving = moving_note(self.preview_moved, self.preview_angering)
            colored(BRASS, f"This track: {dollars(self.preview_price)}{moving}")
        if self.hover_note and self.tool in PRICED_TOOLS:
            colored(BRASS, self.hover_note)
        imgui.pop_text_wrap_pos()
        imgui.spacing()

    def say(self, message: str) -> None:
        """Shows a message above the build bar for a few seconds."""
        self.message = message
        self.message_at = time.monotonic()

    def toggle_trains(self) -> None:
        """Opens the trains window, putting the tool away so a click on the map adds a stop
        rather than building, or closes it."""
        self.trains.open = not self.trains.open
        if self.trains.open:
            self.choose(Tool.NONE)

    def choose(self, tool: Tool) -> None:
        self.tool = tool
        self.stop()

    @property
    def placing_track(self) -> bool:
        """Whether the tool places something with its own way of track through its cell, which Q
        or the mouse wheel turns: a station, a platform or an engine shed."""
        return self.tool in STATION_TOOLS or (
            self.tool == Tool.ADD_FACILITY and self.facility == Facility.ENGINE_SHED
        )

    @property
    def busy(self) -> bool:
        """Whether a piece of track, a station being moved or added to, or a facility is half
        built."""
        return self.track_from is not None or self.moving is not None or self.facility is not None

    def stop(self) -> None:
        self.moving = None
        self.facility = None
        self.station_turn = None
        self.hover_note = ""
        self.track_from = None
        self.preview = None
        self.preview_to = None
        self.preview_for = ()
        self.preview_price = None
        self.preview_moved = 0
        self.preview_angering = 0


def moving_note(moved: int, angering: int) -> str:
    """What to add to a price about the buildings it moves aside, each not the player's own
    raising people's anger with the company."""
    return _anger_note(moved, angering, "moving", "aside")


def taking_down_note(taken: int, angering: int) -> str:
    """What to add to the bulldozer's price about the buildings it takes down, each not the
    player's own raising people's anger with the company."""
    return _anger_note(taken, angering, "taking", "down")


def _anger_note(buildings: int, angering: int, doing: str, where: str) -> str:
    if not buildings:
        return ""
    many = "building" if buildings == 1 else "buildings"
    anger = f" (anger +{angering * ANGER_PER_BUILDING})" if angering else ""
    return f", {doing} {buildings} {many} {where}{anger}"


def _vertical_rule() -> None:
    """A brass diamond between groups of buttons."""
    diamond(BUTTON_HEIGHT)


def draw_station_reach(
    draw_list: imgui.ImDrawList,
    grid: Grid,
    focus: Cell,
    stations: list[Cell],
    camera: IsoCamera,
    center: Vec2,
    reach: Callable[[Cell], tuple[int, int]],
    platforms: Callable[[Cell], list[Cell]] = lambda _: [],
) -> None:
    """Shades the cells stations reach, from their own cells and their `platforms`, cell by
    cell as in Civilization: every station's area
    where it hands goods out, the focused one (being placed, under the mouse or open) in its own
    colour, cells two stations share in grey, the border of where the focused station gathers
    goods, and the stations' own cells in solid blue. `reach` says how far each station hands
    goods out and gathers them. Every corner is put on screen in one go: on raised ground each
    is lifted, which takes nearly as long for one point as for all of them."""
    owners: dict[Cell, set[Cell]] = {}
    shown = {*stations, focus}
    for station in shown:
        for own in (station, *platforms(station)):
            for cell in grid.within(own, reach(station)[0]):
                owners.setdefault(cell, set()).add(station)
    hexes = [
        (
            grid.corners(*cell),
            SHARED_FILL if len(by) > 1 else FOCUS_FILL if focus in by else OTHER_FILL,
        )
        for cell, by in owners.items()
    ]
    hexes += [
        (grid.corners(*own), STATION_FILL)
        for station in shown
        for own in (station, *platforms(station))
    ]
    border = reach_border(grid, (focus, *platforms(focus)), reach(focus)[1])
    plane = [corner for corners, _ in hexes for corner in corners]
    plane += [end for edge in border for end in edge]
    xs, ys = (np.array(values, dtype=np.float64) for values in zip(*plane, strict=True))
    sx, sy = camera.to_screen(xs, ys, center)
    points = [imgui.ImVec2(x, y) for x, y in zip(sx.tolist(), sy.tolist(), strict=True)]
    start = 0
    for corners, fill in hexes:
        polygon = points[start : start + len(corners)]
        start += len(corners)
        draw_list.add_convex_poly_filled(polygon, rgba_u32(fill))
        draw_list.add_polyline(polygon, rgba_u32(CELL_LINE), 1.0, imgui.ImDrawFlags_.closed.value)
    for a, b in itertools.batched(points[start:], 2, strict=True):
        draw_list.add_line(a, b, rgba_u32(COLLECTION_RING), 2.5)


@lru_cache(maxsize=64)
def reach_border(
    grid: Grid, cells: tuple[Cell, ...], steps: int
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """The border of the cells within `steps` of a station's cells, worked out once: an open
    station window draws it every frame, and finding it takes hundreds of lookups. The same
    shape of cells moved by an even number of rows has the same border moved with it, so a
    station tool hovering over the map finds each shape's border only once."""
    near = {near for cell in cells for near in grid.within(cell, steps)}
    top = min(row for row, _ in near)
    shift = (top - top % 2, min(col for _, col in near))
    shape = frozenset((row - shift[0], col - shift[1]) for row, col in near)
    origin, moved = grid.center(0, 0), grid.center(*shift)
    dx, dy = float(moved[0] - origin[0]), float(moved[1] - origin[1])
    return tuple(
        ((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)) for a, b in _shape_border(grid, shape)
    )


@lru_cache(maxsize=64)
def _shape_border(
    grid: Grid, shape: frozenset[Cell]
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(_border(grid, set(shape)))


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
    top, two rails on double track, a pale concrete beam instead of rails on maglev track, a
    wooden deck on bridges and a dark line through tunnels, with a wire beside electric track,
    thicker the closer the view. All the beds go down before any rail, so pieces that meet in a
    cell make one junction or crossing."""
    display = imgui.get_io().display_size
    key = (
        constructions,
        constructions.railway_version,
        camera.center_x,
        camera.center_y,
        camera.zoom_level,
        camera.angle_deg,
        camera.cell_units,
        tuple(center),
        (display.x, display.y),
    )
    global _DRAWN_TRACK
    if planned is None and _DRAWN_TRACK is not None and _DRAWN_TRACK[0] == key:
        lines = _DRAWN_TRACK[1]
    else:
        lines = _track_lines(constructions, camera, center, planned)
        if planned is None:
            _DRAWN_TRACK = (key, lines)
    for points, color, thickness in lines:
        draw_list.add_polyline(points, color, thickness, 0)


# The lines draw_track drew the laid track with last, and what it drew them for: while the view
# and the railway stay as they are, the next frame draws the same lines without working them out.
_DRAWN_TRACK: tuple[tuple, list[tuple[list[imgui.ImVec2], int, float]]] | None = None


def _track_lines(
    constructions: Constructions,
    camera: IsoCamera,
    center: Vec2,
    planned: list[Track] | None,
) -> list[tuple[list[imgui.ImVec2], int, float]]:
    """The lines draw_track draws, in the order it draws them: their points, colour and
    thickness."""
    lines: list[tuple[list[imgui.ImVec2], int, float]] = []
    width = max(1.5, camera.cell_width / 10)
    if planned is None:
        laid, plane = _laid_track(constructions)
    else:
        laid, plane = planned, _track_plane(planned, constructions.tile_map.grid)
    pieces, xs, ys = _on_screen(laid, plane, camera, center, width * 4)
    colors: dict[tuple[float, float, float, float], int] = {}

    def shade(rgba: tuple[float, float, float, float]) -> int:
        color = colors.get(rgba)
        if color is None:
            color = colors[rgba] = rgba_u32(rgba) if planned is None else _greyed(rgba)
        return color

    for piece, points in pieces:
        bed = width * (1.6 + 1.4 * (int(piece.width) - 1))
        if piece.bed == TrackBed.BRIDGE:
            lines.append((points, shade(DECK), bed + width * 1.4))
        if piece.bed == TrackBed.TUNNEL:
            lines.append((points, shade(TUNNEL), bed))
        else:
            lines.append((points, shade(BALLAST), bed))
    spacing = camera.cell_width * FLAT_DOUBLE_SPACING / 2
    lanes: dict[float, list[list[imgui.ImVec2]]] = {}

    def lane_of(index: int, side: float) -> list[imgui.ImVec2]:
        if side not in lanes:
            lanes[side] = _parallel_lines(xs, ys, side)
        return lanes[side][index]

    for index, (piece, points) in enumerate(pieces):
        if piece.bed == TrackBed.TUNNEL:
            continue
        sides = {
            TrackWidth.SINGLE: (0.0,),
            TrackWidth.DOUBLE: (-spacing, spacing),
            TrackWidth.QUADRUPLE: (-3 * spacing, -spacing, spacing, 3 * spacing),
        }[piece.width]
        for side in sides:
            lane = points if not side else lane_of(index, side)
            if piece.maglev:
                lines.append((lane, shade(GUIDEWAY_EDGE), width * 1.3))
                lines.append((lane, shade(GUIDEWAY), width * 0.9))
            else:
                lines.append((lane, shade(RAIL), max(1.0, width * 0.5)))
    for piece, points in pieces:
        if piece.electric and piece.bed != TrackBed.TUNNEL:
            lifted = [imgui.ImVec2(point.x, point.y - width * 1.2) for point in points]
            lines.append((lifted, shade(WIRE), max(1.0, width * 0.35)))
    return lines


def _greyed(rgba: Rgba) -> int:
    """A colour faded to a see-through grey, for track not laid yet."""
    r, g, b, a = rgba
    light = 0.3 * r + 0.59 * g + 0.11 * b
    grey = 0.4 * light + 0.6 * PLANNED_GREY
    return rgba_u32((grey, grey, grey, a * PLANNED_OPACITY))


# The laid track of each Constructions and where its pieces run on the map, kept until the
# railway changes.
_LAID: KeptFor[Constructions, tuple[list[Track], FloatArray]] = KeptFor(weak=True)


def _laid_track(constructions: Constructions) -> tuple[list[Track], FloatArray]:
    def laid() -> tuple[list[Track], FloatArray]:
        pieces = list(constructions.track())
        return pieces, _track_plane(pieces, constructions.tile_map.grid)

    return _LAID.get(constructions, constructions.railway_version, laid)


def _track_plane(pieces: list[Track], grid: Grid) -> FloatArray:
    """Where each piece runs on the map, as an (n, 3, 2) array: from halfway along the step into
    the cell it comes from, where it meets the piece there, through the cell's centre to halfway
    along the step to the cell it leads to."""
    if not pieces:
        return np.zeros((0, 3, 2))
    rows, cols = np.array([piece.cell for piece in pieces]).T
    centers = np.stack(grid.center(rows, cols), axis=-1)
    offsets = grid.track_steps() / 2
    ends = np.array([piece.ends for piece in pieces])
    return np.stack([centers + offsets[ends[:, 0]], centers, centers + offsets[ends[:, 1]]], 1)


def _on_screen(
    pieces: list[Track], plane: FloatArray, camera: IsoCamera, center: Vec2, margin: float
) -> tuple[list[tuple[Track, list[imgui.ImVec2]]], FloatArray, FloatArray]:
    """The pieces that show on the screen, each with its line's points on it, and those points'
    x and y as rows of an array each."""
    xs, ys = camera.to_screen(plane[..., 0], plane[..., 1], center)
    display = imgui.get_io().display_size
    shown = (
        (xs.max(axis=1) >= -margin)
        & (xs.min(axis=1) <= display.x + margin)
        & (ys.max(axis=1) >= -margin)
        & (ys.min(axis=1) <= display.y + margin)
    )
    indices = np.flatnonzero(shown).tolist()
    xs, ys = xs[shown], ys[shown]
    found = [
        (pieces[index], [imgui.ImVec2(x, y) for x, y in zip(row_x, row_y, strict=True)])
        for index, row_x, row_y in zip(indices, xs.tolist(), ys.tolist(), strict=True)
    ]
    return found, xs, ys


def _parallel_lines(xs: FloatArray, ys: FloatArray, side: float) -> list[list[imgui.ImVec2]]:
    """Each line, its points' x and y a row of `xs` and `ys`, moved `side` screen pixels to its
    left, joined where the moved lines meet: all of them at once, as there are thousands of
    pieces of track on screen zoomed out."""
    if not len(xs):
        return []
    dx, dy = np.diff(xs, axis=1), np.diff(ys, axis=1)
    length = np.hypot(dx, dy)
    length[length == 0] = 1.0
    nx, ny = -dy / length, dx / length
    before = np.concatenate([nx[:, :1], nx], axis=1), np.concatenate([ny[:, :1], ny], axis=1)
    after = np.concatenate([nx, nx[:, -1:]], axis=1), np.concatenate([ny, ny[:, -1:]], axis=1)
    mx, my = before[0] + after[0], before[1] + after[1]
    dot = mx * before[0] + my * before[1]
    dot[dot == 0] = 1.0
    moved_x = (xs + mx / dot * side).tolist()
    moved_y = (ys + my / dot * side).tolist()
    return [
        [imgui.ImVec2(x, y) for x, y in zip(row_x, row_y, strict=True)]
        for row_x, row_y in zip(moved_x, moved_y, strict=True)
    ]
