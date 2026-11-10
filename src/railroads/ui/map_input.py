"""The mouse and keys on the map view: dragging pans, the wheel and -/+ zoom, a press without a
drag is a click for the build tools (railroads.ui.map_tools), a right click picks a cell or
station or cancels building, and letter keys open windows and choose tools."""

from typing import TYPE_CHECKING, Literal

from imgui_bundle import imgui

from railroads.sim.constructions import Station
from railroads.ui.build_menu import Tool
from railroads.ui.iso_camera import Vec2
from railroads.ui.minimap import MinimapLayout

if TYPE_CHECKING:
    from railroads.ui.map_view import MapView

# A press that moves the mouse less than this many pixels is a click, not a drag.
CLICK_SLOP = 4.0
TOOL_KEYS = (
    (imgui.Key._1, Tool.STATION),
    (imgui.Key._2, Tool.TRACK),
    (imgui.Key._3, Tool.BULLDOZE),
)


def key_pressed(*keys: imgui.Key) -> bool:
    return any(imgui.is_key_pressed(key, repeat=False) for key in keys)


class MapInput:
    def __init__(self, view: MapView) -> None:
        self._view = view
        self.over_minimap = False
        self._dragged = 0.0
        # Which minimap the current drag started on, if any.
        self._pressed_on: Literal["world", "overview"] | None = None
        self._wheel = 0.0
        self._overview_wheel = 0.0

    def handle_mouse(self, center: Vec2, minimap: MinimapLayout | None) -> None:
        """Reads the mouse over the map, the invisible button drawn last."""
        view = self._view
        camera, grid = view.camera, view.tile_map.grid
        io = imgui.get_io()
        self.over_minimap = minimap is not None and minimap.contains(io.mouse_pos.x, io.mouse_pos.y)
        if imgui.is_item_activated():
            self._dragged = 0.0
            self._pressed_on = (
                None
                if minimap is None or not self.over_minimap
                else "overview"
                if minimap.overview is not None
                and minimap.overview.contains(io.mouse_pos.x, io.mouse_pos.y)
                else "world"
            )
        if imgui.is_item_active():
            self._dragged += abs(io.mouse_delta.x) + abs(io.mouse_delta.y)
            if self._pressed_on is None:
                camera.pan(io.mouse_delta.x, io.mouse_delta.y)
            elif self._pressed_on == "overview" and minimap is not None:
                x, y = minimap.overview_to_map(io.mouse_pos.x, io.mouse_pos.y)
                camera.center_x, camera.center_y = x, y
        on_overview = (
            minimap is not None
            and minimap.overview is not None
            and minimap.overview.contains(io.mouse_pos.x, io.mouse_pos.y)
        )
        if on_overview and imgui.is_item_hovered():
            self._overview_wheel += io.mouse_wheel
            steps = int(self._overview_wheel)
            if steps:
                self._overview_wheel -= steps
                view.minimap.zoom_by(steps)
        if not (self._pressed_on == "overview" and imgui.is_item_active()):
            view.minimap.focus = (camera.center_x, camera.center_y)
        hovered = imgui.is_item_hovered() and not self.over_minimap
        if hovered:
            self._wheel += io.mouse_wheel
            steps = int(self._wheel)
            if steps:
                self._wheel -= steps
                camera.zoom(steps, (io.mouse_pos.x, io.mouse_pos.y), center)
        view.hovered = None
        if hovered:
            x, y = camera.to_plane(io.mouse_pos.x, io.mouse_pos.y, center)
            row, col = grid.cell_at(x, y)
            cell = int(row), int(col)
            if grid.contains(*cell):
                view.hovered = cell
        clicked = imgui.is_item_deactivated() and self._dragged < CLICK_SLOP
        if clicked and self._pressed_on is None and view.hovered is not None:
            view.tools.click(view.hovered)
        if imgui.is_item_clicked(imgui.MouseButton_.right.value):
            self._right_click()
        if key_pressed(imgui.Key.escape):
            view.build.stop()
        view.tools.update_preview(view.hovered)
        view.tools.update_hover_note(view.hovered)

    def _right_click(self) -> None:
        view = self._view
        hovered = view.hovered
        if view.build.tool != Tool.NONE:
            view.build.choose(Tool.NONE)
            view.build.say("Building cancelled.")
        elif hovered is not None and any(
            isinstance(built, Station) for built in view.constructions.at(hovered)
        ):
            opened = view.station_window.station == hovered
            view.station_window.station = None if opened else hovered
            view.selected = None
        elif hovered is not None:
            view.selected = None if view.selected == hovered else hovered
            view.station_window.station = None

    def handle_keys(self) -> None:
        view = self._view
        if imgui.get_io().want_text_input:
            return
        if key_pressed(imgui.Key.minus, imgui.Key.keypad_subtract):
            view.camera.zoom(-1, (0, 0), (0, 0))
        if key_pressed(imgui.Key.equal, imgui.Key.keypad_add):
            view.camera.zoom(1, (0, 0), (0, 0))
        if key_pressed(imgui.Key.m):
            view.minimap.shown = not view.minimap.shown
        if key_pressed(imgui.Key.w):
            view.minimap.world_shown = not view.minimap.world_shown
        for key, tool in TOOL_KEYS:
            if key_pressed(key):
                view.build.choose(Tool.NONE if view.build.tool == tool else tool)
        if key_pressed(imgui.Key.t):
            view.build.trains.open = not view.build.trains.open
        if key_pressed(imgui.Key.s):
            view.build.stations.open = not view.build.stations.open
        if key_pressed(imgui.Key.f):
            view.toggle_search()
        if view.search.kind is not None:
            for key, by in ((imgui.Key.left_arrow, -1), (imgui.Key.right_arrow, 1)):
                if imgui.is_key_pressed(key) and (cell := view.search.step(by)) is not None:
                    view.visit(cell)
