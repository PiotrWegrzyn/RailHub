"""Draws one TileMap. Zoomed out, the ground is one smooth texture with the cell grid and
buildings as dots on top; close up, it is pixel art (see railroads.ui.pixel). The hovered cell, a
compass and the minimap go over both."""

import math
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np
from imgui_bundle import hello_imgui, imgui
from imgui_bundle import icons_fontawesome_4 as fa

from railroads import maps
from railroads.mapgen.world import WorldLand
from railroads.places import Places, StationNames
from railroads.sim.autoplay import StandInPlayer, play
from railroads.sim.clock import START_YEAR
from railroads.sim.constructions import (
    Building,
    Construction,
    Constructions,
    Station,
    Track,
)
from railroads.sim.demolition import bulldoze, bulldoze_price, station_site_price
from railroads.sim.economy import Satisfaction
from railroads.sim.finance import STATION_PRICE, dollars
from railroads.sim.grids import HEX_RADIUS, HEX_ROW_STEP, Cell, GridKind
from railroads.sim.growth import Growth
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, move_station, plan_route, route_price
from railroads.sim.stats import History
from railroads.sim.summaries import Reach, snapshot, summarize
from railroads.sim.tilemap import Relief, Terrain, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.ui.biomes import biome_of
from railroads.ui.build_menu import (
    BuildMenu,
    Tool,
    draw_station_reach,
    draw_track,
    draw_trains,
)
from railroads.ui.buildings import BuildingSpots, draw_building_dots
from railroads.ui.ground import GroundLayout, cell_colors
from railroads.ui.hex_details import describe, draw_hex_details
from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS, IsoCamera, Vec2
from railroads.ui.minimap import Minimap, MinimapLayout, view_corners
from railroads.ui.pixel.layer import PixelLayer
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.track_preview import TrackPreview
from railroads.ui.pixel.train_layer import TrainLayer
from railroads.ui.pixel.world import GLOW, PixelWorld, glow_strength
from railroads.ui.search import BuildingSearch, pulse
from railroads.ui.station_window import StationWindow
from railroads.ui.summaries_panel import SummariesPanel
from railroads.ui.theme import (
    BRASS,
    GONE,
    GROWN,
    MOODS,
    OVERLAY,
    TEXT,
    color_u32,
    good_icon,
    toggle_button,
)

BACKGROUND_COLOR = (0.08, 0.09, 0.11, 1.0)
# How big the goods' icons are in the card of the hovered cell's buildings.
HOVER_ICON = 16.0
# How long the pointer must rest on a cell, in seconds, before the track planned to it is shown.
PREVIEW_REST_S = 0.12
GRID_COLOR = (0.0, 0.0, 0.0, 0.25)
HOVER_COLOR = (1.0, 1.0, 1.0, 0.9)
SELECTED_COLOR = (1.0, 0.85, 0.2, 1.0)
# A press that moves the mouse less than this many pixels is a click, not a drag.
CLICK_SLOP = 4.0
DETAILS_POS = (16.0, 96.0)
STATUS_MARGIN = 10.0
# Station names show from this zoom level in, this many cell widths above the station's middle.
NAMES_FROM_ZOOM = 2
NAME_LIFT = 0.55
NAME_PADDING = 6.0
STATUS_PADDING = 10.0
DETAILS_SIZE = (680.0, 340.0)
COMPASS_COLOR = (1.0, 1.0, 1.0, 0.85)
FOUND_COLOR = (0.3, 0.95, 1.0, 1.0)
# From this cell width, buildings found are outlined; further out each is a small square.
OUTLINE_FOUND_FROM = 16
MOST_OUTLINES = 1500
MOST_FOUND_MARKS = 6000
MOST_MINIMAP_MARKS = 3000
# Stepping to a building found zooms in at least this far, so it can be seen.
VISIT_ZOOM = 4
# Slopes are lit from the top-left of the screen.
LIGHT_ON_SCREEN = (-1.0, -1.0)
# Population points a cell needs to show as a town on the minimap.
MINIMAP_TOWN_POINTS = 12
# Lower left, upper left, top and upper right corner: three edges per hexagon draw the whole grid.
_HEX_EDGE_ANGLES = np.radians([150, 210, 270, 330])


@dataclass(frozen=True)
class GridLook:
    """How a grid sits on screen. Squares are diamonds with north at the top-left, like RT2;
    hexagons have north up."""

    angle_deg: float
    cell_units: float
    # Cell widths in pixels from which the grid shows: first baked into the ground picture, while
    # there are too many cells to outline one by one, then as sharp lines drawn on top.
    baked_lines_from: int | None
    drawn_lines_from: int


LOOKS: dict[GridKind, GridLook] = {
    "square": GridLook(-45.0, math.sqrt(2), baked_lines_from=None, drawn_lines_from=16),
    "hex": GridLook(0.0, 1.0, baked_lines_from=8, drawn_lines_from=24),
}


def _color(rgba: tuple[float, float, float, float]) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))


def _construction_name(construction: Construction) -> str:
    match construction:
        case Building():
            return construction.kind.value
        case Station():
            return "station"
        case Track():
            bed = "" if construction.bed == TrackBed.GROUND else f" {construction.bed.name.lower()}"
            electric = " electric" if construction.electric else ""
            return f"{construction.width.name.lower()}{electric} track{bed}"


@cache
def _sprite_sheet() -> SpriteSheet:
    return SpriteSheet.load()


@cache
def _fine_sprite_sheet() -> SpriteSheet:
    return SpriteSheet.load_fine()


def _key_pressed(*keys: imgui.Key) -> bool:
    return any(imgui.is_key_pressed(key, repeat=False) for key in keys)


class MapView:
    def __init__(
        self,
        tile_map: TileMap,
        constructions: Constructions | None = None,
        world: WorldLand | None = None,
        start_year: int = START_YEAR,
        history: History | None = None,
    ) -> None:
        self.tile_map = tile_map
        self.constructions = Constructions(tile_map) if constructions is None else constructions
        self.growth = Growth(self.constructions, start_year=start_year, history=history)
        self._satisfaction = self.growth.satisfaction()
        self.minimap = Minimap(
            tile_map, maps.world_land() if world is None else world, self._town_cells()
        )
        self._spots = BuildingSpots.of(self.constructions)
        self._look = LOOKS[tile_map.grid.kind]
        width, height = tile_map.grid.size()
        self.camera = IsoCamera(
            center_x=width / 2,
            center_y=height / 2,
            angle_deg=self._look.angle_deg,
            cell_units=self._look.cell_units,
        )
        self._layout: GroundLayout | None = None
        self._pixels: PixelLayer | None = None
        self._trains: TrainLayer | None = None
        self._preview: TrackPreview | None = None
        self._textures: dict[bool, hello_imgui.TextureGpu] = {}
        self._hovered: Cell | None = None
        # The cell whose details are open, picked by clicking without dragging.
        self.selected: Cell | None = None
        self.build = BuildMenu()
        names = StationNames(self.tile_map, Places.load(self.tile_map.bounds))
        self.railway = Railway(self.constructions, names=names)
        self.railway.year = start_year
        self.railway.history = self.growth.history
        self.growth.railway = self.railway
        # Plays in the player's place in a simulated game (railroads.sim.autoplay).
        self.stand_in: StandInPlayer | None = None
        # What came of the last Buy pressed in the hex details.
        self._bought = ""
        # The station levels the map shows, to redraw stations raised since.
        self._shown_levels: dict[Cell, int] = {}
        # The cell the pointer rests on while laying track, and since when, so the planned route
        # is worked out only once it stops there.
        self._resting_on: Cell | None = None
        self._resting_since = 0.0
        self._dragged = 0.0
        self._over_minimap = False
        # Which minimap the current drag started on, if any.
        self._pressed_on: Literal["world", "overview"] | None = None
        self._wheel = 0.0
        self._map_rect: tuple[Vec2, Vec2] = ((0.0, 0.0), (0.0, 0.0))
        self.summaries = SummariesPanel()
        self.station_window = StationWindow()
        self.search = BuildingSearch()
        self._reach = Reach(self.tile_map.grid)

    def gui(self) -> None:
        if self.railway.levels != self._shown_levels:
            raised = {
                cell
                for cell, level in self.railway.levels.items()
                if self._shown_levels.get(cell) != level
            }
            self._shown_levels = dict(self.railway.levels)
            self._buildings_changed(raised)
        self._handle_keys()
        self._draw_map()
        self._draw_details()
        self.build.draw(*self._map_rect)
        self.build.trains.draw(self.railway)
        self.build.stations.draw(self.railway, self.look_at, self.open_station)
        self.station_window.draw(
            self.railway, self.look_at, self.growth.measure_demand, self.start_moving
        )
        self.summaries.draw(*self._map_rect, self.look_at, self.railway.name_of)
        self.search.draw(self.growth, self.visit, self._view_center)

    def _view_center(self) -> Vec2:
        return self.camera.center_x, self.camera.center_y

    def visit(self, cell: Cell) -> None:
        """Takes the view to a cell, close enough to see what stands there."""
        self.camera.zoom_level = max(self.camera.zoom_level, VISIT_ZOOM)
        self.look_at(cell)

    def toggle_search(self) -> None:
        if self.search.open:
            self.search.close()
        else:
            self.search.open = True
            self.search.refresh(self.growth)

    def open_station(self, station: Cell) -> None:
        """Opens a station's details in place of any cell's."""
        self.station_window.station = station
        self.selected = None

    def look_at(self, cell: Cell) -> None:
        """Centres the view on a cell."""
        x, y = self.tile_map.grid.center(*cell)
        self.camera.center_x, self.camera.center_y = float(x), float(y)

    def time_passed(self, months: float) -> None:
        """Moves the trains on by this much of a month."""
        self.railway.run(months)

    def _draw_details(self) -> None:
        if self.selected is None:
            return
        imgui.set_next_window_pos(DETAILS_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(DETAILS_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, still_open = imgui.begin("Hex details", True)
        if expanded:
            bought = draw_hex_details(describe(self.growth, self.selected))
            if bought is not None:
                self._bought = self.growth.buy(self.selected, bought) or f"You own a {bought.value}"
            if self._bought:
                imgui.text_disabled(self._bought)
        imgui.end()
        if not still_open:
            self.selected = None
            self._bought = ""

    def month_passed(self, rng: np.random.Generator, date: str = "") -> None:
        """Grows the towns by a month, shows what changed, and sums up what changed within
        reach of the stations."""
        reach = self._reach.of(self.railway.stations(), lambda s: self.railway.reach(s)[1])
        before = snapshot(self.growth, reach)
        if self.stand_in is None:
            changed = self.growth.month(rng)
        else:
            changed = self.stand_in.month(rng)
        if (summary := summarize(date, self.growth, reach, before)) is not None:
            self.summaries.add(summary)
        satisfaction = self.growth.satisfaction()
        moods_changed = not np.array_equal(satisfaction, self._satisfaction)
        self._satisfaction = satisfaction
        if moods_changed or self.growth.forests_changed:
            self._textures.clear()
            if self._pixels is not None:
                self._pixels.refresh(self._pixels.world.satisfaction_changed(satisfaction))
        if not changed:
            return
        self._spots = BuildingSpots.of(self.constructions)
        self.minimap.show_towns(self._town_cells())
        if self.search.open:
            self.search.refresh(self.growth)
        if self._pixels is not None:
            self._pixels.refresh(self._pixels.world.buildings_changed(changed))

    def _town_cells(self) -> list[Cell]:
        """Cells big enough to show as towns on the minimap."""
        town = self.growth.points() >= MINIMAP_TOWN_POINTS
        return [(int(row), int(col)) for row, col in np.argwhere(town)]

    def draw_money(self) -> None:
        """The cash, on the current line, with this month's and last month's books on hover."""
        finances = self.railway.finances
        imgui.align_text_to_frame_padding()
        color = BRASS if finances.cash >= 0 else GONE
        imgui.text_colored(
            imgui.ImVec4(*color), f"{fa.ICON_FA_MONEY_BILL}  {dollars(finances.cash)}"
        )
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            for title, books in (
                ("This month", finances.month),
                ("Last month", finances.last_month),
            ):
                imgui.text(f"{title}: {dollars(sum(books.values()))}")
                for what, amount in sorted(books.items(), key=lambda item: -item[1]):
                    imgui.text_colored(
                        imgui.ImVec4(*(GROWN if amount >= 0 else GONE)), f"   {dollars(amount)}"
                    )
                    imgui.same_line()
                    imgui.text_disabled(what)
            imgui.end_tooltip()

    def draw_controls(self) -> None:
        """The view's buttons, on the current line: zoom, the two minimaps and the search."""
        if imgui.button(f"{fa.ICON_FA_SEARCH_MINUS}"):
            self.camera.zoom(-1, (0, 0), (0, 0))
        if imgui.is_item_hovered():
            imgui.set_tooltip("Zoom out (-)")
        imgui.same_line(0, 4)
        if imgui.button(f"{fa.ICON_FA_SEARCH_PLUS}"):
            self.camera.zoom(1, (0, 0), (0, 0))
        if imgui.is_item_hovered():
            imgui.set_tooltip("Zoom in (+)")
        imgui.same_line()
        if toggle_button(f"{fa.ICON_FA_MAP}", self.minimap.shown, "Map overview (M)"):
            self.minimap.shown = not self.minimap.shown
        imgui.same_line(0, 4)
        if toggle_button(f"{fa.ICON_FA_GLOBE}", self.minimap.world_shown, "World map (W)"):
            self.minimap.world_shown = not self.minimap.world_shown
        imgui.same_line()
        if toggle_button(f"{fa.ICON_FA_SEARCH}", self.search.open, "Find buildings (F)"):
            self.toggle_search()

    def _handle_keys(self) -> None:
        if imgui.get_io().want_text_input:
            return
        if _key_pressed(imgui.Key.minus, imgui.Key.keypad_subtract):
            self.camera.zoom(-1, (0, 0), (0, 0))
        if _key_pressed(imgui.Key.equal, imgui.Key.keypad_add):
            self.camera.zoom(1, (0, 0), (0, 0))
        if _key_pressed(imgui.Key.m):
            self.minimap.shown = not self.minimap.shown
        if _key_pressed(imgui.Key.w):
            self.minimap.world_shown = not self.minimap.world_shown
        for key, tool in (
            (imgui.Key._1, Tool.STATION),
            (imgui.Key._2, Tool.TRACK),
            (imgui.Key._3, Tool.BULLDOZE),
        ):
            if _key_pressed(key):
                self.build.choose(Tool.NONE if self.build.tool == tool else tool)
        if _key_pressed(imgui.Key.t):
            self.build.trains.open = not self.build.trains.open
        if _key_pressed(imgui.Key.s):
            self.build.stations.open = not self.build.stations.open
        if _key_pressed(imgui.Key.f):
            self.toggle_search()
        if self.search.kind is not None:
            for key, by in ((imgui.Key.left_arrow, -1), (imgui.Key.right_arrow, 1)):
                if imgui.is_key_pressed(key) and (cell := self.search.step(by)) is not None:
                    self.visit(cell)

    def _draw_status(self, draw_list: imgui.ImDrawList, top_left: Vec2, bottom_right: Vec2) -> None:
        """What is under the mouse, in a strip along the top of the map."""
        text = self._status()
        size = imgui.calc_text_size(text)
        x, y = top_left[0] + STATUS_MARGIN, top_left[1] + STATUS_MARGIN
        right = min(x + size.x + 2 * STATUS_PADDING, bottom_right[0] - 80)
        draw_list.add_rect_filled(
            (x, y), (right, y + size.y + 2 * STATUS_PADDING * 0.6), color_u32(OVERLAY), 6.0
        )
        draw_list.push_clip_rect((x, y), (right - STATUS_PADDING / 2, bottom_right[1]), True)
        draw_list.add_text((x + STATUS_PADDING, y + STATUS_PADDING * 0.6), color_u32(TEXT), text)
        draw_list.pop_clip_rect()
        self._draw_hovered_buildings((x, y + size.y + 2 * STATUS_PADDING * 0.6 + STATUS_MARGIN / 2))

    def _draw_hovered_buildings(self, top_left: Vec2) -> None:
        """The buildings in the cell under the mouse, one to a line, in a small card below the
        status strip: how satisfied each is, and icons of what it makes and what it needs."""
        if self._hovered is None or self._over_minimap:
            return
        rows = describe(self.growth, self._hovered).buildings
        if not rows:
            return
        imgui.set_next_window_pos(top_left)
        imgui.set_next_window_bg_alpha(OVERLAY[3])
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_nav.value
            | imgui.WindowFlags_.no_inputs.value
        )
        if imgui.begin("##hovered buildings", None, flags):
            for row in rows:
                imgui.text(row.name if row.count == 1 else f"{row.name} x{row.count}")
                if row.satisfaction in MOODS:
                    imgui.same_line(0, 8)
                    imgui.text_colored(imgui.ImVec4(*MOODS[row.satisfaction]), row.satisfaction)
                for goods, label in ((row.makes, "makes"), (row.needs, "needs")):
                    if not goods:
                        continue
                    imgui.same_line(0, 10)
                    imgui.text_disabled(label)
                    for good, _ in goods:
                        imgui.same_line(0, 2)
                        good_icon(good, size=HOVER_ICON)
        imgui.end()

    def _status(self) -> str:
        zoom = f"Zoom {self.camera.zoom_level + 1}/{len(ZOOM_CELL_WIDTHS)}"
        if self._over_minimap:
            return f"{zoom}. Click or drag on the map overview to look there."
        if self._hovered is None:
            return f"{zoom}. Drag to pan, scroll or -/+ to zoom."
        tile = self.tile_map.tile(self._hovered)
        row, col = self._hovered
        hills = " hills" if tile.relief == Relief.HILLS else ""
        built = self._built_note(self._hovered)
        level = int(self._satisfaction[row, col])
        mood = f"; {Satisfaction(level).name.lower()}" if level >= 0 else ""
        return f"{zoom}. {biome_of(tile).name.title()}{hills}{built}{mood}"

    def _built_note(self, cell: Cell) -> str:
        built = self.constructions.at(cell)
        if not built:
            return ""
        counts = Counter(_construction_name(construction) for construction in built)
        listed = ", ".join(name if n == 1 else f"{name} x{n}" for name, n in counts.items())
        left = self.constructions.space_left(cell)
        return f"; {listed}, " + (f"room for {left} more" if left else "full")

    def _ground_layout(self) -> GroundLayout:
        if self._layout is None:
            self._layout = GroundLayout.of(self.tile_map.grid)
        return self._layout

    def _ground_texture(self, with_lines: bool) -> hello_imgui.TextureGpu:
        """The smooth ground picture used when zoomed out, with or without grid lines."""
        if with_lines not in self._textures:
            light = self.camera.map_direction(*LIGHT_ON_SCREEN)
            colors = cell_colors(self.tile_map)
            strength = glow_strength(self._satisfaction)[..., None]
            land = (self.tile_map.terrain == Terrain.LAND)[..., None]
            tinted = colors[..., :3] * (1 - strength) + np.array(GLOW) * strength
            colors[..., :3] = np.where(land, tinted, colors[..., :3]).astype(np.uint8)
            rgba = self._ground_layout().paint(colors, self.tile_map.relief, light, with_lines)
            self._textures[with_lines] = hello_imgui.create_texture_gpu_from_rgba_data(rgba)
        return self._textures[with_lines]

    def _draw_map(self) -> None:
        origin = imgui.get_cursor_screen_pos()
        size = imgui.get_content_region_avail()
        if size.x <= 0 or size.y <= 0:
            return
        imgui.invisible_button("map", size)
        center = (origin.x + size.x / 2, origin.y + size.y / 2)
        top_left = (origin.x, origin.y)
        bottom_right = (origin.x + size.x, origin.y + size.y)
        self._map_rect = (top_left, bottom_right)
        map_size = self._ground_layout().size
        shown, world_shown = self.minimap.shown, self.minimap.world_shown
        minimap = (
            MinimapLayout.of(map_size, bottom_right, shown, world_shown)
            if shown or world_shown
            else None
        )
        self._handle_input(center, minimap)

        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(top_left, bottom_right, True)
        draw_list.add_rect_filled(top_left, bottom_right, _color(BACKGROUND_COLOR))
        if self.camera.pixel_scale:
            self._draw_ground(draw_list, center, self._ground_texture(False))
            self._pixel_layer().draw(draw_list, self.camera, center, (top_left, bottom_right))
        else:
            self._draw_smooth_map(draw_list, center, top_left, bottom_right)
        if self.camera.pixel_scale:
            self._train_layer().draw(draw_list, self.railway, self.camera, center)
        else:
            draw_track(draw_list, self.constructions, self.camera, center)
            draw_trains(draw_list, self.railway, self.camera, center)
        if self.build.preview is not None:
            if self.camera.pixel_scale:
                self._track_preview().draw(draw_list, self.build.preview, self.camera, center)
            else:
                draw_track(
                    draw_list, self.constructions, self.camera, center, self.build.preview.pieces
                )
        if self.build.track_from is not None:
            self._draw_outline(draw_list, center, self.build.track_from, SELECTED_COLOR)
        if self._hovered is not None:
            self._draw_outline(draw_list, center, self._hovered)
            hovering_station = any(
                isinstance(built, Station) for built in self.constructions.at(self._hovered)
            )
            if self.build.tool == Tool.STATION or hovering_station:
                stations = self.railway.stations() if self.build.tool == Tool.STATION else []
                draw_station_reach(
                    draw_list,
                    self.tile_map.grid,
                    self._hovered,
                    stations,
                    self.camera,
                    center,
                    self.railway.reach,
                )
        opened = self.station_window.station
        if (
            opened is not None
            and (
                self._hovered is None
                or self._hovered == opened
                or not any(
                    isinstance(built, Station) for built in self.constructions.at(self._hovered)
                )
            )
            and self.build.tool != Tool.STATION
        ):
            draw_station_reach(
                draw_list, self.tile_map.grid, opened, [], self.camera, center, self.railway.reach
            )
        if self.selected is not None:
            self._draw_outline(draw_list, center, self.selected, SELECTED_COLOR)
        for cell in self.summaries.outlined():
            self._draw_outline(draw_list, center, cell, BRASS)
        self._draw_found(draw_list, center, top_left, bottom_right)
        self._draw_station_names(draw_list, center)
        self._draw_compass(draw_list, (bottom_right[0] - 36, top_left[1] + 36))
        self._draw_status(draw_list, top_left, bottom_right)
        if minimap is not None:
            corners = view_corners(self.camera, center, top_left, bottom_right)
            ground = imgui.ImTextureRef(self._ground_texture(False).texture_id())
            textures = (self.minimap.world_texture(), ground)
            self.minimap.draw(draw_list, minimap, textures, corners)
            self._draw_found_on_overview(draw_list, minimap)
        draw_list.pop_clip_rect()

    def _draw_found(
        self, draw_list: imgui.ImDrawList, center: Vec2, top_left: Vec2, bottom_right: Vec2
    ) -> None:
        """Marks every building the search found that is on screen, and the one being visited."""
        search = self.search
        if search.kind is None or not search.cells:
            return
        width = self.camera.cell_width
        grid = self.tile_map.grid
        sx, sy = self.camera.to_screen(*grid.center(search.rows, search.cols), center)
        on_screen = np.flatnonzero(
            (sx > top_left[0] - width)
            & (sx < bottom_right[0] + width)
            & (sy > top_left[1] - width)
            & (sy < bottom_right[1] + width)
        )
        if width >= OUTLINE_FOUND_FROM:
            for index in on_screen[:: max(1, on_screen.size // MOST_OUTLINES)].tolist():
                self._draw_outline(draw_list, center, search.cells[index], FOUND_COLOR)
        else:
            color = _color(FOUND_COLOR)
            shown = on_screen[:: max(1, on_screen.size // MOST_FOUND_MARKS)]
            for x, y in zip(sx[shown].tolist(), sy[shown].tolist(), strict=True):
                draw_list.add_rect_filled((x - 2, y - 2), (x + 2, y + 2), color)
        if search.current is not None:
            self._draw_visited(draw_list, center, search.current)

    def _draw_visited(self, draw_list: imgui.ImDrawList, center: Vec2, cell: Cell) -> None:
        """A bright outline round the building being visited and an arrow bobbing over it."""
        self._draw_outline(draw_list, center, cell, HOVER_COLOR)
        x, y = self.camera.to_screen(*self.tile_map.grid.center(*cell), center)
        tip = float(y) - self.camera.cell_width * 0.55 - 10 * pulse(imgui.get_time())
        draw_list.add_triangle_filled(
            (float(x), tip),
            (float(x) - 10, tip - 16),
            (float(x) + 10, tip - 16),
            _color(FOUND_COLOR),
        )

    def _draw_found_on_overview(self, draw_list: imgui.ImDrawList, minimap: MinimapLayout) -> None:
        search = self.search
        panel = minimap.overview
        if search.kind is None or panel is None or not search.cells:
            return
        step = max(1, len(search.cells) // MOST_MINIMAP_MARKS)
        xs, ys = self.tile_map.grid.center(search.rows[::step], search.cols[::step])
        color = _color(FOUND_COLOR)
        sx, sy = minimap.overview_point(xs, ys)
        for x, y in zip(sx.tolist(), sy.tolist(), strict=True):
            draw_list.add_rect_filled((x - 1.5, y - 1.5), (x + 1.5, y + 1.5), color)

    def _draw_smooth_map(
        self, draw_list: imgui.ImDrawList, center: Vec2, top_left: Vec2, bottom_right: Vec2
    ) -> None:
        cell_width = self.camera.cell_width
        drawn_lines = cell_width >= self._look.drawn_lines_from
        baked_from = self._look.baked_lines_from
        baked_lines = not drawn_lines and baked_from is not None and cell_width >= baked_from
        self._draw_ground(draw_list, center, self._ground_texture(baked_lines))
        if drawn_lines:
            self._draw_grid(draw_list, center, top_left, bottom_right)
        draw_building_dots(draw_list, self.camera, center, (top_left, bottom_right), self._spots)

    def _train_layer(self) -> TrainLayer:
        if self._trains is None:
            self._trains = TrainLayer(self._pixel_layer().world)
        return self._trains

    def _track_preview(self) -> TrackPreview:
        if self._preview is None:
            self._preview = TrackPreview(self._pixel_layer().world)
        return self._preview

    def _pixel_layer(self) -> PixelLayer:
        if self._pixels is None:
            world = PixelWorld(
                self.tile_map,
                self.constructions,
                _sprite_sheet(),
                self._look.angle_deg,
                self._look.cell_units,
                fine=_fine_sprite_sheet(),
            )
            world.satisfaction = self._satisfaction.copy()
            world.station_levels = self.railway.levels
            self._pixels = PixelLayer(world)
            self._buildings_changed(set(self.railway.levels))
        return self._pixels

    def _handle_input(self, center: Vec2, minimap: MinimapLayout | None) -> None:
        io = imgui.get_io()
        self._over_minimap = minimap is not None and minimap.contains(
            io.mouse_pos.x, io.mouse_pos.y
        )
        if imgui.is_item_activated():
            self._dragged = 0.0
            self._pressed_on = (
                None
                if minimap is None or not self._over_minimap
                else "overview"
                if minimap.overview is not None
                and minimap.overview.contains(io.mouse_pos.x, io.mouse_pos.y)
                else "world"
            )
        if imgui.is_item_active():
            self._dragged += abs(io.mouse_delta.x) + abs(io.mouse_delta.y)
            if self._pressed_on is None:
                self.camera.pan(io.mouse_delta.x, io.mouse_delta.y)
            elif self._pressed_on == "overview" and minimap is not None:
                x, y = minimap.overview_to_map(io.mouse_pos.x, io.mouse_pos.y)
                self.camera.center_x, self.camera.center_y = x, y
        hovered = imgui.is_item_hovered() and not self._over_minimap
        if hovered:
            self._wheel += io.mouse_wheel
            steps = int(self._wheel)
            if steps:
                self._wheel -= steps
                self.camera.zoom(steps, (io.mouse_pos.x, io.mouse_pos.y), center)
        self._hovered = None
        if hovered:
            x, y = self.camera.to_plane(io.mouse_pos.x, io.mouse_pos.y, center)
            row, col = self.tile_map.grid.cell_at(x, y)
            cell = int(row), int(col)
            if self.tile_map.grid.contains(*cell):
                self._hovered = cell
        clicked = imgui.is_item_deactivated() and self._dragged < CLICK_SLOP
        if clicked and self._pressed_on is None and self._hovered is not None:
            self._click(self._hovered)
        if imgui.is_item_clicked(imgui.MouseButton_.right.value):
            if self.build.tool != Tool.NONE:
                self.build.choose(Tool.NONE)
                self.build.say("Building cancelled.")
            elif self._hovered is not None and any(
                isinstance(built, Station) for built in self.constructions.at(self._hovered)
            ):
                opened = self.station_window.station == self._hovered
                self.station_window.station = None if opened else self._hovered
                self.selected = None
            elif self._hovered is not None:
                self.selected = None if self.selected == self._hovered else self._hovered
                self.station_window.station = None
        if _key_pressed(imgui.Key.escape):
            self.build.stop()
        self._update_preview()
        self._update_hover_note()

    def _click(self, cell: Cell) -> None:
        build = self.build
        match build.tool:
            case Tool.NONE:
                if any(isinstance(built, Station) for built in self.constructions.at(cell)):
                    self.open_station(cell)
            case Tool.STATION:
                finances = self.railway.finances
                site = station_site_price(self.constructions, cell)
                if site is None:
                    build.say("A station can't go there.")
                    return
                if not finances.can_afford(STATION_PRICE + site):
                    build.say(
                        f"This station costs {dollars(STATION_PRICE + site)}; "
                        f"you have {dollars(finances.cash)}."
                    )
                    return
                changed = build_station(self.growth, cell)
                if not changed:
                    build.say("A station can't go there.")
                    return
                finances.spend(STATION_PRICE, "stations")
                if site:
                    finances.spend(site, "demolition")
                build.say(f"Station built for {dollars(STATION_PRICE + site)}.")
                self.growth.measure_demand(cell)
                self._buildings_changed(changed)
            case Tool.BULLDOZE:
                self._bulldoze(cell)
            case Tool.MOVE_STATION:
                self._move_station(cell)
            case Tool.TRACK:
                if build.track_from is None:
                    build.track_from = cell
                    build.say("Now click where the track goes.")
                    return
                route = plan_route(
                    self.constructions, build.track_from, cell, build.track_width, build.electric
                )
                if route is None:
                    build.say("No way for track to get there.")
                    return
                laid = len(route.changes(self.constructions))
                price = route_price(self.constructions, route)
                finances = self.railway.finances
                if not finances.can_afford(price):
                    build.say(
                        f"This track costs {dollars(price)}; you have {dollars(finances.cash)}."
                    )
                    return
                self._buildings_changed(lay_route(self.growth, route))
                finances.spend(price, "track")
                build.say(
                    f"Laid {laid} pieces of track for {dollars(price)}."
                    if laid
                    else "That track is laid already."
                )
                build.track_from = cell
                build.preview, build.preview_to = None, None

    def _bulldoze(self, cell: Cell) -> None:
        build, finances = self.build, self.railway.finances
        price = bulldoze_price(self.constructions, cell)
        if price is None:
            build.say("Nothing here can be cleared.")
            return
        if not finances.can_afford(price):
            build.say(f"Clearing this costs {dollars(price)}; you have {dollars(finances.cash)}.")
            return
        changed = bulldoze(self.growth, cell)
        finances.spend(price, "demolition")
        build.say(f"Cleared for {dollars(price)}.")
        if self.station_window.station == cell:
            self.station_window.station = None
        self._buildings_changed(changed)

    def _move_station(self, cell: Cell) -> None:
        build, finances = self.build, self.railway.finances
        old = build.moving
        site = station_site_price(self.constructions, cell)
        if old is None or site is None:
            build.say("The station can't go there.")
            return
        price = STATION_PRICE + site
        if not finances.can_afford(price):
            build.say(f"Moving it there costs {dollars(price)}; you have {dollars(finances.cash)}.")
            return
        changed = move_station(self.growth, self.railway, old, cell)
        if changed is None:
            build.say("The station can't go there.")
            return
        finances.spend(STATION_PRICE, "stations")
        if site:
            finances.spend(site, "demolition")
        build.say(f"Station moved for {dollars(price)}.")
        build.choose(Tool.NONE)
        self.station_window.station = cell
        self.growth.measure_demand(cell)
        self._buildings_changed(changed)

    def start_moving(self, station: Cell) -> None:
        """Lets the next click choose where a station goes."""
        self.build.choose(Tool.MOVE_STATION)
        self.build.moving = station
        self.build.say("Click where the station goes.")

    def _update_hover_note(self) -> None:
        """What the chosen tool would cost in the cell under the mouse."""
        build, cell = self.build, self._hovered
        if cell is None or build.tool not in (Tool.STATION, Tool.BULLDOZE, Tool.MOVE_STATION):
            build.hover_note = ""
            return
        if build.tool == Tool.BULLDOZE:
            price = bulldoze_price(self.constructions, cell)
            build.hover_note = "" if price is None else f"Clearing this cell: {dollars(price)}"
            return
        site = station_site_price(self.constructions, cell)
        build.hover_note = (
            "A station can't go here"
            if site is None
            else f"Here: {dollars(STATION_PRICE + site)}"
            + (f" ({dollars(site)} of it clearing the cell)" if site else "")
        )

    def _update_preview(self) -> None:
        build = self.build
        if build.tool != Tool.TRACK or build.track_from is None or self._hovered is None:
            build.preview, build.preview_to = None, None
            return
        if self._hovered != build.preview_to:
            now = imgui.get_time()
            if self._resting_on != self._hovered:
                self._resting_on, self._resting_since = self._hovered, now
            if now - self._resting_since < PREVIEW_REST_S:
                return
            build.preview_to = self._hovered
            build.preview = plan_route(
                self.constructions,
                build.track_from,
                self._hovered,
                build.track_width,
                build.electric,
            )
            build.preview_price = (
                None if build.preview is None else route_price(self.constructions, build.preview)
            )

    def simulate(self, years: int, rng: np.random.Generator) -> None:
        """Lets a stand-in player build stations and serve the towns for `years`, printing the
        map every ten years, and goes on playing for the player from there."""
        self.stand_in = play(self.growth, self.railway, years, rng)
        self._satisfaction = self.growth.satisfaction()
        self._buildings_changed(set())

    def _buildings_changed(self, changed: set[Cell]) -> None:
        """Shows buildings, track and stations that changed outside the monthly growth."""
        self._spots = BuildingSpots.of(self.constructions)
        self.minimap.show_towns(self._town_cells())
        if self.search.open:
            self.search.refresh(self.growth)
        if self._pixels is not None:
            self._pixels.refresh(self._pixels.world.buildings_changed(changed))

    def _draw_ground(
        self, draw_list: imgui.ImDrawList, center: Vec2, texture: hello_imgui.TextureGpu
    ) -> None:
        width, height = self._ground_layout().size
        to_screen = self.camera.to_screen
        draw_list.add_image_quad(
            imgui.ImTextureRef(texture.texture_id()),
            to_screen(0, 0, center),
            to_screen(width, 0, center),
            to_screen(width, height, center),
            to_screen(0, height, center),
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
        )

    def _visible_area(self, center: Vec2, top_left: Vec2, bottom_right: Vec2) -> tuple[float, ...]:
        corners = [
            self.camera.to_plane(x, y, center)
            for x in (top_left[0], bottom_right[0])
            for y in (top_left[1], bottom_right[1])
        ]
        xs, ys = [corner[0] for corner in corners], [corner[1] for corner in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def _draw_grid(
        self, draw_list: imgui.ImDrawList, center: Vec2, top_left: Vec2, bottom_right: Vec2
    ) -> None:
        x0, y0, x1, y1 = self._visible_area(center, top_left, bottom_right)
        if self.tile_map.grid.kind == "square":
            self._draw_square_lines(draw_list, center, (x0, y0, x1, y1))
        else:
            self._draw_hex_lines(draw_list, center, (x0, y0, x1, y1))

    def _draw_square_lines(
        self, draw_list: imgui.ImDrawList, center: Vec2, visible: tuple[float, ...]
    ) -> None:
        grid = self.tile_map.grid
        x0, y0, x1, y1 = visible
        c0, c1 = max(0, math.floor(x0)), min(grid.cols, math.ceil(x1))
        r0, r1 = max(0, math.floor(y0)), min(grid.rows, math.ceil(y1))
        if c0 >= c1 or r0 >= r1:
            return
        color = _color(GRID_COLOR)
        to_screen = self.camera.to_screen
        for x in range(c0, c1 + 1):
            draw_list.add_line(to_screen(x, r0, center), to_screen(x, r1, center), color)
        for y in range(r0, r1 + 1):
            draw_list.add_line(to_screen(c0, y, center), to_screen(c1, y, center), color)

    def _draw_hex_lines(
        self, draw_list: imgui.ImDrawList, center: Vec2, visible: tuple[float, ...]
    ) -> None:
        grid = self.tile_map.grid
        x0, y0, x1, y1 = visible
        r0, r1 = (
            max(0, math.floor(y0 / HEX_ROW_STEP) - 1),
            min(grid.rows, math.ceil(y1 / HEX_ROW_STEP) + 1),
        )
        c0, c1 = max(0, math.floor(x0) - 1), min(grid.cols, math.ceil(x1) + 1)
        if c0 >= c1 or r0 >= r1:
            return
        rows, cols = np.mgrid[r0:r1, c0:c1]
        cx, cy = grid.center(rows.ravel(), cols.ravel())
        px = cx[:, None] + HEX_RADIUS * np.cos(_HEX_EDGE_ANGLES)
        py = cy[:, None] + HEX_RADIUS * np.sin(_HEX_EDGE_ANGLES)
        sx, sy = self.camera.to_screen(px, py, center)
        color = _color(GRID_COLOR)
        for points in np.stack([sx, sy], axis=2).tolist():
            draw_list.add_polyline([imgui.ImVec2(x, y) for x, y in points], color, 1.0, 0)

    def _draw_outline(
        self,
        draw_list: imgui.ImDrawList,
        center: Vec2,
        cell: Cell,
        color: tuple[float, float, float, float] = HOVER_COLOR,
    ) -> None:
        points = [
            imgui.ImVec2(*self.camera.to_screen(x, y, center))
            for x, y in self.tile_map.grid.corners(*cell)
        ]
        draw_list.add_polyline(points, _color(color), 2.0, imgui.ImDrawFlags_.closed.value)

    def _draw_station_names(self, draw_list: imgui.ImDrawList, center: Vec2) -> None:
        """Each station's name on a dark tag above it, once zoomed in to NAMES_FROM_ZOOM."""
        if self.camera.zoom_level < NAMES_FROM_ZOOM:
            return
        width = self.camera.cell_width
        grid = self.tile_map.grid
        for station in self.railway.stations():
            name = self.railway.name_of(station)
            x, y = self.camera.to_screen(*grid.center(*station), center)
            size = imgui.calc_text_size(name)
            left = x - size.x / 2 - NAME_PADDING
            bottom = y - width * NAME_LIFT
            top = bottom - size.y - NAME_PADDING
            draw_list.add_rect_filled(
                (left, top), (x + size.x / 2 + NAME_PADDING, bottom), color_u32(OVERLAY), 5.0
            )
            draw_list.add_text((x - size.x / 2, top + NAME_PADDING / 2), color_u32(TEXT), name)

    def _draw_compass(self, draw_list: imgui.ImDrawList, anchor: Vec2) -> None:
        here = self.camera.to_screen(self.camera.center_x, self.camera.center_y, (0, 0))
        north = self.camera.to_screen(self.camera.center_x, self.camera.center_y - 1, (0, 0))
        dx, dy = north[0] - here[0], north[1] - here[1]
        length = math.hypot(dx, dy)
        dx, dy = dx / length, dy / length
        color = _color(COMPASS_COLOR)
        tip = (anchor[0] + dx * 14, anchor[1] + dy * 14)
        draw_list.add_line((anchor[0] - dx * 14, anchor[1] - dy * 14), tip, color, 2.0)
        draw_list.add_triangle_filled(
            tip,
            (tip[0] - dx * 7 - dy * 5, tip[1] - dy * 7 + dx * 5),
            (tip[0] - dx * 7 + dy * 5, tip[1] - dy * 7 - dx * 5),
            color,
        )
        label = imgui.calc_text_size("N")
        position = (anchor[0] + dx * 25 - label.x / 2, anchor[1] + dy * 25 - label.y / 2)
        draw_list.add_text(position, color, "N")
