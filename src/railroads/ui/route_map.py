"""The map in the trains window: the ground seen from above with the track and stations on it,
the chosen train's stops numbered. The wheel zooms it around the mouse, dragging moves it, and
clicking a station adds it to the train's stops. While the player hovers a good to load, the
stations whose buildings want that good are ringed in its colour."""

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.constructions import TrackWidth
from railroads.sim.goods import Good
from railroads.sim.grids import Cell
from railroads.sim.railway import Railway
from railroads.sim.trains import Stop, Train
from railroads.ui.iso_camera import Vec2
from railroads.ui.minimap import Panel
from railroads.ui.theme import BRASS, GOOD_COLORS, color_u32, draw_good_icon

# How many times as close as the whole map the map can show.
MOST_ZOOM = 24.0
ZOOM_STEP = 1.25
# How far, in pixels, from a station's spot a click still picks it.
PICK_REACH = 9.0
# How far, in pixels, the mouse moves before a press is a drag rather than a click.
DRAG_START = 3.0
# How much room round a train's stops is shown when the map frames them, as a share of the
# distance they spread over.
FRAME_MARGIN = 0.6
HIGHLIGHT_ICON = 16.0
# How far inside the map's edge a station off it that wants the hovered good is pointed at.
EDGE_INSET = 10.0

SEA = (0.12, 0.17, 0.24, 1.0)
TRACK = (0.2, 0.15, 0.11, 0.95)
STATION = (0.27, 0.35, 0.8, 1.0)
STATION_EDGE = (0.08, 0.08, 0.12, 0.9)
HOVER_EDGE = (1.0, 1.0, 1.0, 1.0)
LABEL_SHADOW = (0.0, 0.0, 0.0, 0.75)
DIMMED = (0.0, 0.0, 0.0, 0.35)

# The picture of the ground (railroads.ui.ground) and its size in map units.
type GroundPicture = Callable[[], tuple[imgui.ImTextureRef, Vec2]]


@dataclass(frozen=True, slots=True)
class MapFrame:
    """Where the map shows in its panel: the map point at the panel's top-left corner and how
    many pixels a map unit takes."""

    panel: Panel
    origin: Vec2
    scale: float

    @classmethod
    def of(cls, panel: Panel, map_size: Vec2, zoom: float, focus: Vec2 | None = None) -> MapFrame:
        """`zoom` times as close as the whole map fitting the panel, centred on `focus` but kept
        on the map, and centred on the map along a side it doesn't fill."""
        width, height = map_size
        scale = min(panel.width / width, panel.height / height) * zoom
        cx, cy = (width / 2, height / 2) if focus is None else focus
        return cls(
            panel,
            (_start(cx, panel.width / scale, width), _start(cy, panel.height / scale, height)),
            scale,
        )

    @property
    def center(self) -> Vec2:
        return (
            self.origin[0] + self.panel.width / self.scale / 2,
            self.origin[1] + self.panel.height / self.scale / 2,
        )

    def to_screen(self, x: float, y: float) -> Vec2:
        return (
            self.panel.left + (x - self.origin[0]) * self.scale,
            self.panel.top + (y - self.origin[1]) * self.scale,
        )

    def to_map(self, screen_x: float, screen_y: float) -> Vec2:
        return (
            self.origin[0] + (screen_x - self.panel.left) / self.scale,
            self.origin[1] + (screen_y - self.panel.top) / self.scale,
        )


def _start(center: float, span: float, length: float) -> float:
    if span >= length:
        return (length - span) / 2
    return min(max(center - span / 2, 0.0), length - span)


def station_at(spots: Mapping[Cell, Vec2], x: float, y: float, reach: float) -> Cell | None:
    """The station whose spot is nearest a point on screen, if one is within `reach` of it."""
    near = [
        (math.hypot(sx - x, sy - y), station)
        for station, (sx, sy) in spots.items()
        if math.hypot(sx - x, sy - y) <= reach
    ]
    return min(near)[1] if near else None


def edge_spot(panel: Panel, x: float, y: float, inset: float) -> Vec2:
    """Where a point outside the panel shows at its edge: the spot `inset` inside the edge on
    the way from the panel's middle to the point."""
    cx, cy = panel.left + panel.width / 2, panel.top + panel.height / 2
    dx, dy = x - cx, y - cy
    reach_x = panel.width / 2 - inset
    reach_y = panel.height / 2 - inset
    share = min(
        reach_x / abs(dx) if dx else math.inf,
        reach_y / abs(dy) if dy else math.inf,
        1.0,
    )
    return cx + dx * share, cy + dy * share


def wanted_at(railway: Railway, good: Good) -> dict[Cell, float]:
    """The stations whose buildings want a good, each with how much of it they could use in
    a month."""
    wants = {
        station: railway.station_demand(station).monthly_want(good)
        for station in railway.stations()
    }
    return {station: want for station, want in wants.items() if want > 0}


def can_stop_at(train: Train, station: Cell) -> bool:
    """Whether a station can be added as the train's next stop: any but the one it ends with."""
    return not train.stops or train.stops[-1].cell != station


class RouteMap:
    def __init__(self) -> None:
        self.zoom = 1.0
        # The map point the map is centred on, or None for the middle of the map.
        self.focus: Vec2 | None = None
        # The good hovered among those to load this frame, whose demand the map shows.
        self.good: Good | None = None
        # The train the map was last framed on, to frame the next one chosen.
        self._framed: str | None = None
        self._dragged = False

    def frame(self, railway: Railway, train: Train, map_size: Vec2, panel: Panel) -> None:
        """Centres the map on the train's stops, close enough to show them all."""
        self._framed = train.name
        grid = railway.constructions.tile_map.grid
        points = [grid.center(*stop.cell) for stop in train.stops]
        if not points:
            self.zoom, self.focus = 1.0, None
            return
        xs, ys = [float(x) for x, _ in points], [float(y) for _, y in points]
        self.focus = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        spread_x = (max(xs) - min(xs)) * (1 + FRAME_MARGIN) + 8
        spread_y = (max(ys) - min(ys)) * (1 + FRAME_MARGIN) + 8
        wanted = min(panel.width / spread_x, panel.height / spread_y)
        self.zoom = min(max(wanted / _fit(panel, map_size), 1.0), MOST_ZOOM)

    def zoom_at(self, frame: MapFrame, map_size: Vec2, x: float, y: float, steps: float) -> None:
        """Zooms in (steps above 0) or out, keeping the map point under (x, y) where it is."""
        under = frame.to_map(x, y)
        self.zoom = min(max(self.zoom * ZOOM_STEP**steps, 1.0), MOST_ZOOM)
        panel = frame.panel
        scale = _fit(panel, map_size) * self.zoom
        origin = (under[0] - (x - panel.left) / scale, under[1] - (y - panel.top) / scale)
        focus = (origin[0] + panel.width / scale / 2, origin[1] + panel.height / scale / 2)
        self.focus = MapFrame.of(panel, map_size, self.zoom, focus).center

    def draw(self, railway: Railway, train: Train, ground: GroundPicture) -> None:
        """Draws a row of zoom buttons and a hint, and the map below them in the space left."""
        texture, map_size = ground()
        wanted = wanted_at(railway, self.good) if self.good is not None else {}
        steps = self._draw_buttons(wanted)
        low = imgui.get_cursor_screen_pos()
        width, height = imgui.get_content_region_avail()
        panel = Panel(low.x, low.y, max(width, 40.0), max(height, 40.0))
        if self._framed != train.name:
            self.frame(railway, train, map_size, panel)
        frame = MapFrame.of(panel, map_size, self.zoom, self.focus)
        if steps:
            self.zoom_at(frame, map_size, panel.left + width / 2, panel.top + height / 2, steps)
        imgui.invisible_button("route map", (panel.width, panel.height))
        hovering, clicked = self._handle_mouse(frame, map_size)
        frame = MapFrame.of(panel, map_size, self.zoom, self.focus)
        spots = self._station_spots(railway, frame)
        mouse = imgui.get_io().mouse_pos
        hovered = station_at(spots, mouse.x, mouse.y, PICK_REACH) if hovering else None
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect((panel.left, panel.top), (panel.right, panel.bottom), True)
        draw_list.add_rect_filled(
            (panel.left, panel.top), (panel.right, panel.bottom), color_u32(SEA)
        )
        draw_list.add_image(texture, frame.to_screen(0, 0), frame.to_screen(*map_size))
        self._draw_track(draw_list, railway, frame)
        if self.good is not None:
            draw_list.add_rect_filled(
                (panel.left, panel.top), (panel.right, panel.bottom), color_u32(DIMMED)
            )
        self._draw_stations(draw_list, railway, train, spots, wanted, hovered, frame)
        draw_list.pop_clip_rect()
        if hovered is not None:
            self._station_tooltip(railway, train, hovered, wanted)
            if clicked and can_stop_at(train, hovered):
                add_stop(train, hovered)

    def _draw_buttons(self, wanted: Mapping[Cell, float]) -> int:
        """The zoom buttons and a line on what the map shows; returns the zoom steps pressed."""
        steps = 0
        if imgui.small_button(f"{fa.ICON_FA_SEARCH_PLUS}##zoom in"):
            steps = 2
        if imgui.is_item_hovered():
            imgui.set_tooltip("Zoom in, or turn the wheel over the map")
        imgui.same_line(0, 3)
        if imgui.small_button(f"{fa.ICON_FA_SEARCH_MINUS}##zoom out"):
            steps = -2
        if imgui.is_item_hovered():
            imgui.set_tooltip("Zoom out; drag the map to move it")
        imgui.same_line(0, 10)
        if self.good is None:
            imgui.text_disabled("Click a station to add it")
        elif wanted:
            imgui.text(f"{self.good.value} is wanted at {_count(len(wanted))}")
        else:
            imgui.text(f"No station wants {self.good.value} yet")
        return steps

    def _handle_mouse(self, frame: MapFrame, map_size: Vec2) -> tuple[bool, bool]:
        """Zooms and moves the map by the mouse over it; returns whether the mouse is over the
        map, and whether it was clicked without being dragged."""
        io = imgui.get_io()
        hovering = imgui.is_item_hovered()
        if imgui.is_item_activated():
            self._dragged = False
        clicked = imgui.is_item_deactivated() and hovering and not self._dragged
        if hovering and io.mouse_wheel:
            self.zoom_at(frame, map_size, io.mouse_pos.x, io.mouse_pos.y, io.mouse_wheel)
        if imgui.is_item_active() and imgui.is_mouse_dragging(0, DRAG_START):
            self._dragged = True
            x, y = frame.center
            focus = (x - io.mouse_delta.x / frame.scale, y - io.mouse_delta.y / frame.scale)
            self.focus = MapFrame.of(frame.panel, map_size, self.zoom, focus).center
        return hovering, clicked

    @staticmethod
    def _station_spots(railway: Railway, frame: MapFrame) -> dict[Cell, Vec2]:
        grid = railway.constructions.tile_map.grid
        spots = {}
        for station in railway.stations():
            x, y = grid.center(*station)
            spots[station] = frame.to_screen(float(x), float(y))
        return spots

    @staticmethod
    def _draw_track(draw_list: imgui.ImDrawList, railway: Railway, frame: MapFrame) -> None:
        grid = railway.constructions.tile_map.grid
        offsets = grid.track_steps() / 2
        width = min(max(frame.scale * 0.18, 1.0), 4.0)
        color = color_u32(TRACK)
        for piece in railway.constructions.track():
            cx, cy = grid.center(*piece.cell)
            start, end = (offsets[point] for point in piece.ends)
            points = [
                imgui.ImVec2(*frame.to_screen(cx + start[0], cy + start[1])),
                imgui.ImVec2(*frame.to_screen(cx, cy)),
                imgui.ImVec2(*frame.to_screen(cx + end[0], cy + end[1])),
            ]
            double = 1.8 if piece.width == TrackWidth.DOUBLE else 1.0
            draw_list.add_polyline(points, color, width * double, 0)

    def _draw_stations(
        self,
        draw_list: imgui.ImDrawList,
        railway: Railway,
        train: Train,
        spots: Mapping[Cell, Vec2],
        wanted: Mapping[Cell, float],
        hovered: Cell | None,
        frame: MapFrame,
    ) -> None:
        half = min(max(frame.scale * 0.35, 3.5), 8.0)
        stops: dict[Cell, list[int]] = {}
        for number, stop in enumerate(train.stops, 1):
            stops.setdefault(stop.cell, []).append(number)
        for station, (x, y) in spots.items():
            fill = BRASS if station in stops else STATION
            draw_list.add_rect_filled((x - half, y - half), (x + half, y + half), color_u32(fill))
            edge = HOVER_EDGE if station == hovered else STATION_EDGE
            thick = 2.0 if station == hovered else 1.0
            draw_list.add_rect(
                (x - half, y - half), (x + half, y + half), color_u32(edge), 0.0, thick
            )
        if self.good is not None:
            self._draw_wanted(draw_list, railway, spots, wanted, half, frame.panel)
        for station, numbers in stops.items():
            x, y = spots[station]
            text = ",".join(map(str, numbers))
            left = x - half - 3 - imgui.calc_text_size(text).x
            _label(draw_list, (left, y - half - 4), text, BRASS)

    def _draw_wanted(
        self,
        draw_list: imgui.ImDrawList,
        railway: Railway,
        spots: Mapping[Cell, Vec2],
        wanted: Mapping[Cell, float],
        half: float,
        panel: Panel,
    ) -> None:
        """Rings each station that wants the hovered good, and points from the map's edge at
        those outside it."""
        assert self.good is not None
        color = GOOD_COLORS[self.good]
        pulse = 1.5 * (1 + math.sin(time.monotonic() * 5))
        for station in wanted:
            x, y = spots[station]
            if not panel.contains(x, y):
                ex, ey = edge_spot(panel, x, y, EDGE_INSET)
                draw_list.add_line((ex, ey), _toward((ex, ey), (x, y), 14.0), color_u32(color), 3.0)
                draw_list.add_circle_filled((ex, ey), 6.0 + pulse / 2, color_u32(color))
                draw_list.add_circle((ex, ey), 6.0 + pulse / 2, color_u32(LABEL_SHADOW), 0, 1.5)
                continue
            radius = half + 4 + pulse
            draw_list.add_circle((x, y), radius, color_u32(LABEL_SHADOW), 0, 5.0)
            draw_list.add_circle((x, y), radius, color_u32(color), 0, 2.5)
            icon = (x - HIGHLIGHT_ICON / 2, y - radius - HIGHLIGHT_ICON - 2)
            draw_good_icon(draw_list, self.good, icon, HIGHLIGHT_ICON)
            _label(draw_list, (x + radius + 4, y - 7), railway.name_of(station), (1, 1, 1, 1))

    def _station_tooltip(
        self, railway: Railway, train: Train, station: Cell, wanted: Mapping[Cell, float]
    ) -> None:
        imgui.begin_tooltip()
        imgui.text(railway.name_of(station))
        if self.good is not None and station in wanted:
            imgui.text(f"Wants about {wanted[station]:.0f} {self.good.value} a month")
        if can_stop_at(train, station):
            imgui.text_disabled("Click to add it as a stop")
        else:
            imgui.text_disabled("It is already the last stop")
        imgui.end_tooltip()


def add_stop(train: Train, station: Cell) -> None:
    train.stops.append(Stop(station))
    train.restart()


def _fit(panel: Panel, map_size: Vec2) -> float:
    return min(panel.width / map_size[0], panel.height / map_size[1])


def _toward(start: Vec2, end: Vec2, length: float) -> Vec2:
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy) or 1.0
    return start[0] + dx / distance * length, start[1] + dy / distance * length


def _count(stations: int) -> str:
    return "1 station" if stations == 1 else f"{stations} stations"


def _label(
    draw_list: imgui.ImDrawList, at: Vec2, text: str, rgba: tuple[float, float, float, float]
) -> None:
    x, y = at
    draw_list.add_text((x + 1, y + 1), color_u32(LABEL_SHADOW), text)
    draw_list.add_text((x, y), color_u32(rgba), text)
