"""The map in the trains window: the ground seen from above, softened, with the track and stations
on it, the chosen train's stops numbered. The wheel zooms it around the mouse, dragging moves it,
clicking a station adds it to the train's stops, and hovering one sums up what it wants and makes.
While the player hovers a good, the stations around which it is wanted or made are ringed, a ring
for each of three kinds, its shade by how much a month (SHADE_FROM): yellow to red where homes
want it for a general need like food, heating or mail, light to dark violet where some
building wants it for a need only it meets, and light to dark blue where more than LEAST_MADE of it
is made."""

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui
from PIL import Image, ImageFilter

from railroads.sim.constructions import TrackWidth
from railroads.sim.demand import DEMAND_PER_NEED, StationDemand
from railroads.sim.goods import FITS, Good, Need
from railroads.sim.grids import Cell
from railroads.sim.railway import Railway
from railroads.sim.steps import Steps
from railroads.sim.trains import Stop, Train
from railroads.ui.iso_camera import Vec2
from railroads.ui.minimap import Panel
from railroads.ui.station_offers import Offer, Supplies, goods_to_load
from railroads.ui.theme import BRASS, color_u32, draw_good_icon, good_amount, good_icon

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
# How many of what a station wants and makes its summary shows, the most first.
SUMMARY_MOST = 5
# How often, in seconds, where the hovered good is made is worked out again.
MADE_EVERY = 0.5
# How far, in pixels of the halved ground picture, it is blurred so the cells' edges soften.
SOFTEN_RADIUS = 1.2
# How many rows of the halved picture a step of softening halves or blurs (soften_steps).
SOFTEN_ROWS = 160

SEA = (0.12, 0.17, 0.24, 1.0)
TRACK = (0.2, 0.15, 0.11, 0.95)
STATION = (0.27, 0.35, 0.8, 1.0)
STATION_EDGE = (0.08, 0.08, 0.12, 0.9)
HOVER_EDGE = (1.0, 1.0, 1.0, 1.0)
LABEL_SHADOW = (0.0, 0.0, 0.0, 0.75)
DIMMED = (0.0, 0.0, 0.0, 0.35)
NEEDED = (0.95, 0.27, 0.22, 1.0)
MADE = (0.35, 0.65, 1.0, 1.0)

type Rgba = tuple[float, float, float, float]

# Needs homes everywhere have: food and heating, which many goods meet, and the passengers and
# mail every station both sends and wants. Any other need is a special one.
GENERAL_NEEDS = frozenset({Need.FOOD, Need.HEATING, Need.PASSENGERS, Need.MAIL})
# How much of a good made around a station a month marks it as made there.
LEAST_MADE = 10.0
# The amounts a month from which a mark takes its second and third shade.
SHADE_FROM = (40.0, 120.0)
# The three shades, least to most, of each kind of mark.
GENERAL_SHADES: tuple[Rgba, Rgba, Rgba] = (
    (1.0, 0.88, 0.2, 1.0),
    (1.0, 0.55, 0.1, 1.0),
    (0.93, 0.18, 0.15, 1.0),
)
SPECIAL_SHADES: tuple[Rgba, Rgba, Rgba] = (
    (0.84, 0.66, 1.0, 1.0),
    (0.62, 0.36, 0.92, 1.0),
    (0.4, 0.13, 0.66, 1.0),
)
MADE_SHADES: tuple[Rgba, Rgba, Rgba] = (
    (0.6, 0.85, 1.0, 1.0),
    (0.25, 0.56, 0.96, 1.0),
    (0.08, 0.24, 0.72, 1.0),
)

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


@dataclass(frozen=True)
class Marks:
    """How much of a good a station's buildings want a month for general needs and for special
    ones, and how much of it is made around the station a month, if more than LEAST_MADE."""

    general: float = 0.0
    special: float = 0.0
    made: float = 0.0

    def shades(self) -> list[Rgba]:
        """A shade for each kind of mark the station has: general, special, then made."""
        kinds = (
            (self.general, GENERAL_SHADES),
            (self.special, SPECIAL_SHADES),
            (self.made, MADE_SHADES),
        )
        return [shades[shade_of(amount)] for amount, shades in kinds if amount > 0]


def shade_of(amount: float) -> int:
    """Which of a mark's three shades an amount a month takes: the first under SHADE_FROM[0]."""
    return sum(amount >= least for least in SHADE_FROM)


def needed(demand: StationDemand, good: Good) -> tuple[float, float]:
    """How much of a good the buildings around a station could use a month, for general needs
    and for special ones."""
    general = special = 0.0
    for need, capacity in demand.capacity.items():
        if good in FITS[need]:
            if need in GENERAL_NEEDS:
                general += capacity / DEMAND_PER_NEED
            else:
                special += capacity / DEMAND_PER_NEED
    return general, special


def marks_at(railway: Railway, good: Good, made: Mapping[Cell, Offer]) -> dict[Cell, Marks]:
    """The stations a good is wanted or made at, given where a train can load it."""
    found = {}
    for station in railway.stations():
        general, special = needed(railway.station_demand(station), good)
        offer = made.get(station)
        making = offer.made if offer is not None and offer.made > LEAST_MADE else 0.0
        if general > 0 or special > 0 or making > 0:
            found[station] = Marks(general, special, making)
    return found


def made_at(railway: Railway, good: Good, supplies: Supplies) -> dict[Cell, Offer]:
    """The stations a train can load a good at, made around them or waiting there."""
    found = {}
    for station in railway.stations():
        for offer in goods_to_load(railway, station, supplies(station)):
            if offer.good == good:
                found[station] = offer
    return found


@dataclass(frozen=True)
class Summary:
    """What a station wants, need by need with the demand left, and what a train can load
    there, good by good with how much is made a month, or waits if none is made; the most
    first, at most SUMMARY_MOST of each, and how many more there are."""

    wants: list[tuple[Need, float]] = field(default_factory=list)
    makes: list[tuple[Good, float]] = field(default_factory=list)
    more_wants: int = 0
    more_makes: int = 0


def summary_of(railway: Railway, station: Cell, made: Mapping[Good, float]) -> Summary:
    demand = railway.station_demand(station)
    left = [(need, demand.record(need).left) for need in Need]
    wants = sorted(((n, a) for n, a in left if a >= 0.5), key=lambda w: -w[1])
    offers = goods_to_load(railway, station, made)
    amounts = [(o.good, o.made if o.made > 0 else o.waiting) for o in offers]
    makes = sorted(((g, a) for g, a in amounts if a >= 0.5), key=lambda m: -m[1])
    return Summary(
        wants[:SUMMARY_MOST],
        makes[:SUMMARY_MOST],
        max(0, len(wants) - SUMMARY_MOST),
        max(0, len(makes) - SUMMARY_MOST),
    )


def soften(rgba: np.ndarray, radius: float = SOFTEN_RADIUS) -> np.ndarray:
    """The ground picture at half its size and blurred a little, so the map's cells don't show
    as hard blocks; the renderer stretches it smoothly back over the map."""
    picture = Image.fromarray(np.ascontiguousarray(rgba), "RGBA").reduce(2)
    blurred = picture.filter(ImageFilter.GaussianBlur(radius))
    return np.ascontiguousarray(np.asarray(blurred, dtype=np.uint8))


def soften_steps(rgba: np.ndarray, radius: float = SOFTEN_RADIUS) -> Steps[np.ndarray]:
    """soften a step at a time, with the same picture at the end: SOFTEN_ROWS rows of the half
    made a step, then as many blurred. The picture must stay as it is until the last step."""
    whole = Image.fromarray(np.ascontiguousarray(rgba), "RGBA")
    # Halving takes the pixels two by two, so bands of an even number of rows halve apart.
    half = Image.new("RGBA", ((whole.width + 1) // 2, (whole.height + 1) // 2))
    for top in range(0, whole.height, 2 * SOFTEN_ROWS):
        bottom = min(top + 2 * SOFTEN_ROWS, whole.height)
        half.paste(whole.crop((0, top, whole.width, bottom)).reduce(2), (0, top // 2))
        yield
    width, height = half.size
    soft = np.empty((height, width, 4), dtype=np.uint8)
    blur = ImageFilter.GaussianBlur(radius)
    # A band is blurred with as many rows more on either side as the blur reaches, so it comes
    # out as it does within the whole picture.
    reach = blur_reach(radius)
    for top in range(0, height, SOFTEN_ROWS):
        bottom = min(top + SOFTEN_ROWS, height)
        start, end = max(0, top - reach), min(height, bottom + reach)
        band = half.crop((0, start, width, end)).filter(blur)
        soft[top:bottom] = np.asarray(band, dtype=np.uint8)[top - start : bottom - start]
        yield
    return soft


def blur_reach(radius: float) -> int:
    """How many rows away a Gaussian blur of `radius` takes colour from: Pillow blurs in three box
    blurs, each with a radius no larger than the Gaussian one and reaching a row past it."""
    return 3 * (math.floor(radius) + 1)


def can_stop_at(train: Train, station: Cell) -> bool:
    """Whether a station can be added as the train's next stop: any but the one it ends with."""
    return not train.stops or train.stops[-1].cell != station


class RouteMap:
    def __init__(self) -> None:
        self.zoom = 1.0
        # The map point the map is centred on, or None for the middle of the map.
        self.focus: Vec2 | None = None
        # The good hovered among those to load this frame, whose demand and makers the map shows.
        self.good: Good | None = None
        # Where the hovered good was last found to be made: when, which good, and the stations.
        self._made: tuple[float, Good, dict[Cell, Offer]] | None = None
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

    def made(self, railway: Railway, good: Good, supplies: Supplies) -> dict[Cell, Offer]:
        """Where a good can be loaded, kept for a moment so hovering it stays cheap."""
        now = time.monotonic()
        kept = self._made
        if kept is None or kept[1] != good or now - kept[0] > MADE_EVERY:
            kept = (now, good, made_at(railway, good, supplies))
            self._made = kept
        return kept[2]

    def draw(
        self, railway: Railway, train: Train, ground: GroundPicture, supplies: Supplies
    ) -> None:
        """Draws a row of zoom buttons and a hint, and the map below them in the space left."""
        texture, map_size = ground()
        good = self.good
        marks = {} if good is None else marks_at(railway, good, self.made(railway, good, supplies))
        steps = self._draw_buttons(marks)
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
        self._draw_stations(draw_list, railway, train, spots, marks, hovered, frame)
        draw_list.pop_clip_rect()
        if hovered is not None:
            self._station_tooltip(railway, train, hovered, supplies)
            if clicked and can_stop_at(train, hovered):
                add_stop(train, hovered)

    def _draw_buttons(self, marks: Mapping[Cell, Marks]) -> int:
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
            return steps
        good_icon(self.good)
        low, high = f"{SHADE_FROM[0]:.0f}", f"{SHADE_FROM[1]:.0f}"
        counts = (
            (
                "needed",
                GENERAL_SHADES,
                sum(m.general > 0 for m in marks.values()),
                "Wanted for food or heating",
            ),
            (
                "special",
                SPECIAL_SHADES,
                sum(m.special > 0 for m in marks.values()),
                "Wanted by buildings that need just this",
            ),
            ("made", MADE_SHADES, sum(m.made > 0 for m in marks.values()), "Made around"),
        )
        for label, shades, count, why in counts:
            imgui.same_line(0, 8)
            imgui.text_colored(imgui.ImVec4(*shades[1]), f"{label} {count}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"{why}: lightest under {low} a month, middle {low}-{high}, darkest {high}+"
                )
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
        marks: Mapping[Cell, Marks],
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
            self._draw_marks(draw_list, railway, spots, marks, half, frame.panel)
        for station, numbers in stops.items():
            x, y = spots[station]
            text = ",".join(map(str, numbers))
            left = x - half - 3 - imgui.calc_text_size(text).x
            _label(draw_list, (left, y - half - 4), text, BRASS)

    def _draw_marks(
        self,
        draw_list: imgui.ImDrawList,
        railway: Railway,
        spots: Mapping[Cell, Vec2],
        marks: Mapping[Cell, Marks],
        half: float,
        panel: Panel,
    ) -> None:
        """Rings each station the hovered good is wanted or made at, a ring for each kind of
        mark, and points from the map's edge at those off it."""
        assert self.good is not None
        pulse = 1.5 * (1 + math.sin(time.monotonic() * 5))
        for station, mark in marks.items():
            x, y = spots[station]
            colors = mark.shades()
            if not panel.contains(x, y):
                ex, ey = edge_spot(panel, x, y, EDGE_INSET)
                tip = _toward((ex, ey), (x, y), 14.0)
                draw_list.add_line((ex, ey), tip, color_u32(colors[0]), 3.0)
                outer = 5.0 + 2.5 * len(colors) + pulse / 2
                for ring, color in enumerate(colors):
                    draw_list.add_circle_filled((ex, ey), outer - ring * 2.5, color_u32(color))
                draw_list.add_circle((ex, ey), outer, color_u32(LABEL_SHADOW), 0, 1.5)
                continue
            radius = half + 4 + pulse
            for ring, color in enumerate(colors):
                r = radius + ring * 4.0
                draw_list.add_circle((x, y), r, color_u32(LABEL_SHADOW), 0, 5.0)
                draw_list.add_circle((x, y), r, color_u32(color), 0, 2.5)
            outer = radius + (len(colors) - 1) * 4.0
            icon = (x - HIGHLIGHT_ICON / 2, y - outer - HIGHLIGHT_ICON - 2)
            draw_good_icon(draw_list, self.good, icon, HIGHLIGHT_ICON)
            _label(draw_list, (x + outer + 4, y - 7), railway.name_of(station), (1, 1, 1, 1))

    @staticmethod
    def _station_tooltip(railway: Railway, train: Train, station: Cell, supplies: Supplies) -> None:
        """A short summary of a station: what it wants and what a train can load there."""
        summary = summary_of(railway, station, supplies(station))
        imgui.begin_tooltip()
        imgui.text(railway.name_of(station))
        _summary_row("Wants", NEEDED, summary.wants, summary.more_wants)
        _summary_row("Makes", MADE, summary.makes, summary.more_makes)
        if can_stop_at(train, station):
            imgui.text_disabled("Click to add it as a stop")
        else:
            imgui.text_disabled("It is already the last stop")
        imgui.end_tooltip()


def _summary_row(
    title: str,
    rgba: tuple[float, float, float, float],
    items: list[tuple[Need, float]] | list[tuple[Good, float]],
    more: int,
) -> None:
    imgui.text_colored(imgui.ImVec4(*rgba), title)
    imgui.same_line(56)
    if not items:
        imgui.text_disabled("nothing")
        return
    for index, (good, amount) in enumerate(items):
        if index:
            imgui.same_line(0, 8)
        good_amount(good, f"{amount:.0f}")
    if more:
        imgui.same_line(0, 8)
        imgui.text_disabled(f"+{more}")


def add_stop(train: Train, station: Cell) -> None:
    train.stops.append(Stop(station))
    train.restart()


def _fit(panel: Panel, map_size: Vec2) -> float:
    return min(panel.width / map_size[0], panel.height / map_size[1])


def _toward(start: Vec2, end: Vec2, length: float) -> Vec2:
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy) or 1.0
    return start[0] + dx / distance * length, start[1] + dy / distance * length


def _label(
    draw_list: imgui.ImDrawList, at: Vec2, text: str, rgba: tuple[float, float, float, float]
) -> None:
    x, y = at
    draw_list.add_text((x + 1, y + 1), color_u32(LABEL_SHADOW), text)
    draw_list.add_text((x, y), color_u32(rgba), text)
