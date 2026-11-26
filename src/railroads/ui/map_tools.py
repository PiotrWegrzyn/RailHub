"""What a click on the map does with the build tool chosen (railroads.ui.build_menu): building,
laying track, bulldozing and moving stations through the game (railroads.sim.game), the price
note for the cell under the pointer, and the planned track shown once the pointer rests."""

import time
from typing import TYPE_CHECKING

from imgui_bundle import imgui

from railroads.sim.constructions import Facility, Track
from railroads.sim.facilities import REACH as FACILITY_REACH
from railroads.sim.grids import Cell
from railroads.sim.railway import PLATFORM_REACH
from railroads.sim.routes import Route, plan_route_steps, station_way
from railroads.sim.steps import Steps, finish
from railroads.ui.build_menu import PRICED_TOOLS, Tool, moving_note, taking_down_note
from railroads.words import dollars

if TYPE_CHECKING:
    from railroads.sim.game import Outcome
    from railroads.ui.map_view import MapView

# How long the pointer must rest on a cell, in seconds, before the track planned to it is shown.
PREVIEW_REST_S = 0.12
# How long planning the previewed track may take of each frame, in seconds; a long route is
# planned over several frames rather than stalling one.
PREVIEW_STEP_S = 0.006
_UNFINISHED = object()


def _work_on[T](steps: Steps[T], seconds: float) -> T | object:
    """Takes steps for about `seconds`, at least one; returns what they came to, or
    _UNFINISHED."""
    until = time.perf_counter() + seconds
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value
        if time.perf_counter() >= until:
            return _UNFINISHED


class MapTools:
    def __init__(self, view: MapView) -> None:
        self._view = view
        # The cell the pointer rests on while laying track, and since when, so the planned route
        # is worked out only once it stops there.
        self._resting_on: Cell | None = None
        self._resting_since = 0.0
        # The track being planned to the cell the pointer rests on, a few steps a frame, and
        # what it is planned for (_track_wanted).
        self._planning: tuple[tuple[object, ...], Steps[Route | None]] | None = None

    def click(self, cell: Cell) -> None:
        view = self._view
        build, game = view.build, view.game
        match build.tool:
            case Tool.NONE:
                station = view.constructions.main_station(cell)
                annex = view.constructions.annex_at(cell)
                if station is None and annex is not None:
                    station = annex.station
                if station is not None:
                    view.open_station(station)
            case Tool.STATION:
                done = game.build_station(cell, self.station_ends(cell))
                if self._said(done):
                    view.buildings_changed(done.changed)
            case Tool.BULLDOZE:
                self._bulldoze(cell)
            case Tool.MOVE_STATION:
                self._move_station(cell)
            case Tool.ADD_PLATFORM:
                self._add_platform(cell)
            case Tool.ADD_FACILITY:
                self._add_facility(cell)
            case Tool.TRACK:
                if build.track_from is None:
                    build.track_from = cell
                    build.say("Now click where the track goes.")
                    return
                planned = self._planned(cell)
                done = game.lay_track(
                    build.track_from,
                    cell,
                    build.track_width,
                    build.electric,
                    build.maglev,
                    planned,
                )
                if not self._said(done):
                    return
                view.buildings_changed(done.changed)
                build.track_from = cell
                build.preview, build.preview_to = None, None

    def _said(self, done: Outcome) -> bool:
        """Tells the player how a build went; whether it was done."""
        self._view.build.say(done.message)
        return done.ok

    def _bulldoze(self, cell: Cell) -> None:
        view = self._view
        done = view.game.bulldoze(cell)
        if not self._said(done):
            return
        if view.station_window.station == cell:
            view.station_window.station = None
        view.buildings_changed(done.changed)

    def _move_station(self, cell: Cell) -> None:
        view = self._view
        build = view.build
        if build.moving is None:
            build.say("The station can't go there.")
            return
        done = view.game.move_station(build.moving, cell, self.station_ends(cell))
        if not self._said(done):
            return
        build.choose(Tool.NONE)
        view.station_window.station = cell
        view.buildings_changed(done.changed)

    def _add_platform(self, cell: Cell) -> None:
        view = self._view
        build = view.build
        if build.moving is None:
            return
        station = build.moving
        done = view.game.add_platform(station, cell, self.station_ends(cell))
        if not self._said(done):
            return
        view.buildings_changed(done.changed)
        if view.railway.platforms_left(station):
            build.say(f"{done.message} Click where the next one goes.")
            return
        build.choose(Tool.NONE)

    def _add_facility(self, cell: Cell) -> None:
        view = self._view
        build = view.build
        if build.moving is None or build.facility is None:
            return
        ends = self.station_ends(cell) if build.placing_track else None
        done = view.game.add_facility(build.moving, build.facility, cell, ends)
        if not self._said(done):
            return
        view.buildings_changed(done.changed)
        build.choose(Tool.NONE)

    def start_adding_facility(self, station: Cell, facility: Facility) -> None:
        """Lets the next click choose where a facility beside a station goes."""
        build = self._view.build
        build.choose(Tool.ADD_FACILITY)
        build.moving, build.facility = station, facility
        build.say(
            f"Click a cell up to {FACILITY_REACH} cells from the station for its {facility.value}; "
            "Q or the mouse wheel turns it."
        )

    def start_adding_platform(self, station: Cell) -> None:
        """Lets the next click choose where a station's new platform goes."""
        build = self._view.build
        build.choose(Tool.ADD_PLATFORM)
        build.moving = station
        build.say(
            f"Click a free cell up to {PLATFORM_REACH} cells from the station for its new platform; "
            "Q or the mouse wheel turns it."
        )

    def start_moving(self, station: Cell) -> None:
        """Lets the next click choose where a station goes."""
        build = self._view.build
        build.choose(Tool.MOVE_STATION)
        build.moving = station
        build.say("Click where the station goes.")

    def station_ways(self) -> list[tuple[int, int]]:
        """Every way a station's track can run through its cell, turning round in order."""
        grid = self._view.constructions.tile_map.grid
        count = len(grid.track_points())
        ends = grid.track_ends()
        return [(a, (a + count // 2) % count) for a in ends if a < (a + count // 2) % count]

    def station_ends(self, cell: Cell) -> tuple[int, int]:
        """The way a station placed in the cell would run: as the player turned it, or else the
        way it takes there by itself (railroads.sim.routes.station_way)."""
        build = self._view.build
        if build.station_turn is None:
            return station_way(self._view.constructions, cell)
        ways = self.station_ways()
        return ways[build.station_turn % len(ways)]

    def turn_station(self, cell: Cell | None, by: int = 1) -> None:
        """Turns the station being placed `by` ways on, back for a negative number."""
        ways = self.station_ways()
        current = self.station_ends(cell) if cell is not None else ways[-1]
        self._view.build.station_turn = (ways.index(current) + by) % len(ways)

    def update_hover_note(self, cell: Cell | None) -> None:
        """What the chosen tool would cost in the cell under the mouse."""
        build, game = self._view.build, self._view.game
        if cell is None or build.tool not in PRICED_TOOLS:
            build.hover_note = ""
            return
        if build.tool == Tool.BULLDOZE:
            clearing = game.clearing_quote(cell)
            build.hover_note = (
                ""
                if clearing.price is None
                else f"Clearing this cell: {dollars(clearing.price)}"
                f"{taking_down_note(clearing.moved, clearing.angering)}"
            )
            return
        if build.tool == Tool.ADD_FACILITY:
            if build.moving is None or build.facility is None:
                build.hover_note = ""
                return
            quote = game.facility_quote(build.moving, build.facility, cell)
            name = build.facility.value.capitalize()
            build.hover_note = (
                f"{name} can't go here: {quote.problem}."
                if quote.price is None
                else f"{name} here"
                + (
                    f": {dollars(quote.price)} to clear the cell{moving_note(quote.moved, quote.angering)}."
                    if quote.price
                    else "."
                )
            )
            return
        if build.tool == Tool.ADD_PLATFORM:
            platform = game.platform_quote(cell)
            build.hover_note = (
                f"A platform can't go here: {platform.problem}."
                if platform.price is None and platform.problem
                else "A platform can't go here."
                if platform.price is None
                else f"New platform here: {dollars(platform.price)}"
                f"{moving_note(platform.moved, platform.angering)}. Q or the mouse wheel turns it."
            )
            return
        quote = game.station_quote(cell)
        build.hover_note = (
            f"A station can't go here: {quote.problem}."
            if quote.price is None
            else f"Here: {dollars(quote.price)}"
            + (
                f" ({dollars(quote.site)} of it clearing the cell{moving_note(quote.moved, quote.angering)})"
                if quote.site
                else ""
            )
            + ". Q or the mouse wheel turns it."
        )

    def _planned(self, cell: Cell) -> Route | None:
        """The route already planned, or being planned, for track to `cell` as the tool lays it
        now, if there is one."""
        build, wanted = self._view.build, self._track_wanted(cell)
        if build.preview_for == wanted:
            return build.preview
        if self._planning is not None and self._planning[0] == wanted:
            steps, self._planning = self._planning[1], None
            return finish(steps)
        return None

    def _track_wanted(self, cell: Cell) -> tuple[object, ...]:
        """What track from where it starts to `cell` would be planned for, so a route planned
        for the same is laid as it is."""
        build = self._view.build
        return (
            build.track_from,
            cell,
            build.track_width,
            build.electric,
            build.maglev,
            self._view.constructions.version,
        )

    def update_preview(self, cell: Cell | None) -> None:
        build, constructions = self._view.build, self._view.constructions
        if build.placing_track and cell is not None:
            build.preview = Route([cell], [Track(cell, self.station_ends(cell))], 0.0)
            build.preview_to, build.preview_price = cell, None
            return
        if build.tool != Tool.TRACK or build.track_from is None or cell is None:
            build.preview, build.preview_to = None, None
            return
        if cell != build.preview_to:
            now = imgui.get_time()
            if self._resting_on != cell:
                self._resting_on, self._resting_since = cell, now
            if now - self._resting_since < PREVIEW_REST_S:
                return
            wanted = self._track_wanted(cell)
            if self._planning is None or self._planning[0] != wanted:
                steps = plan_route_steps(
                    constructions,
                    build.track_from,
                    cell,
                    build.track_width,
                    build.electric,
                    build.maglev,
                )
                self._planning = wanted, steps
            route = _work_on(self._planning[1], PREVIEW_STEP_S)
            if route is _UNFINISHED:
                return
            self._planning = None
            build.preview_to, build.preview_for, build.preview = cell, wanted, route
            quote = None if route is None else self._view.game.track_quote(route)
            build.preview_price = None if quote is None else quote.price
            build.preview_moved = 0 if quote is None else quote.moved
            build.preview_angering = 0 if quote is None else quote.angering
