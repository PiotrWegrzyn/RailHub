"""What a click on the map does with the build tool chosen (railroads.ui.build_menu): building,
laying track, bulldozing and moving stations through the game (railroads.sim.game), the price
note for the cell under the pointer, and the planned track shown once the pointer rests."""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from railroads.sim.constructions import Station, Track
from railroads.sim.demolition import bulldoze_price, station_site_price
from railroads.sim.finance import STATION_PRICE, dollars
from railroads.sim.grids import Cell
from railroads.sim.routes import Route, plan_route, route_price, station_way
from railroads.ui.build_menu import Tool

if TYPE_CHECKING:
    from railroads.ui.map_view import MapView

# How long the pointer must rest on a cell, in seconds, before the track planned to it is shown.
PREVIEW_REST_S = 0.12


class MapTools:
    def __init__(self, view: MapView) -> None:
        self._view = view
        # The cell the pointer rests on while laying track, and since when, so the planned route
        # is worked out only once it stops there.
        self._resting_on: Cell | None = None
        self._resting_since = 0.0

    def click(self, cell: Cell) -> None:
        view = self._view
        build, game = view.build, view.game
        match build.tool:
            case Tool.NONE:
                if any(isinstance(built, Station) for built in view.constructions.at(cell)):
                    view.open_station(cell)
            case Tool.STATION:
                done = game.build_station(cell, self.station_ends(cell))
                build.say(done.message)
                if done.ok:
                    view.buildings_changed(done.changed)
            case Tool.BULLDOZE:
                self._bulldoze(cell)
            case Tool.MOVE_STATION:
                self._move_station(cell)
            case Tool.TRACK:
                if build.track_from is None:
                    build.track_from = cell
                    build.say("Now click where the track goes.")
                    return
                done = game.lay_track(
                    build.track_from, cell, build.track_width, build.electric, build.maglev
                )
                build.say(done.message)
                if not done.ok:
                    return
                view.buildings_changed(done.changed)
                build.track_from = cell
                build.preview, build.preview_to = None, None

    def _bulldoze(self, cell: Cell) -> None:
        view = self._view
        done = view.game.bulldoze(cell)
        view.build.say(done.message)
        if not done.ok:
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
        build.say(done.message)
        if not done.ok:
            return
        build.choose(Tool.NONE)
        view.station_window.station = cell
        view.buildings_changed(done.changed)

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

    def turn_station(self, cell: Cell | None) -> None:
        """Turns the station being placed to the next way."""
        ways = self.station_ways()
        current = self.station_ends(cell) if cell is not None else ways[-1]
        self._view.build.station_turn = (ways.index(current) + 1) % len(ways)

    def update_hover_note(self, cell: Cell | None) -> None:
        """What the chosen tool would cost in the cell under the mouse."""
        build, constructions = self._view.build, self._view.constructions
        if cell is None or build.tool not in (Tool.STATION, Tool.BULLDOZE, Tool.MOVE_STATION):
            build.hover_note = ""
            return
        if build.tool == Tool.BULLDOZE:
            price = bulldoze_price(constructions, cell)
            build.hover_note = "" if price is None else f"Clearing this cell: {dollars(price)}"
            return
        site = station_site_price(constructions, cell)
        build.hover_note = (
            "A station can't go here"
            if site is None
            else f"Here: {dollars(STATION_PRICE + site)}"
            + (f" ({dollars(site)} of it clearing the cell)" if site else "")
            + ". Q turns it."
        )

    def update_preview(self, cell: Cell | None) -> None:
        build, constructions = self._view.build, self._view.constructions
        if build.tool in (Tool.STATION, Tool.MOVE_STATION) and cell is not None:
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
            build.preview_to = cell
            build.preview = plan_route(
                constructions,
                build.track_from,
                cell,
                build.track_width,
                build.electric,
                build.maglev,
            )
            build.preview_price = (
                None if build.preview is None else route_price(constructions, build.preview)
            )
