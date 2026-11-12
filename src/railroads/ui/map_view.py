"""Draws one TileMap. Zoomed out, the ground is one smooth texture with the cell grid and
buildings as dots on top; close up, it is pixel art (see railroads.ui.pixel). The hovered cell, a
compass and the minimap go over both."""

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads import maps
from railroads.mapgen.world import WorldLand
from railroads.sim.constructions import (
    Constructions,
    Station,
)
from railroads.sim.game import Game
from railroads.sim.geo import Coord
from railroads.sim.grids import Cell, GridKind
from railroads.sim.growth import Growth
from railroads.sim.railway import Railway
from railroads.sim.tilemap import Terrain, TileMap
from railroads.ui.build_menu import (
    BuildMenu,
    Tool,
    draw_station_reach,
    draw_track,
    draw_trains,
)
from railroads.ui.buildings import BuildingSpots, draw_building_dots
from railroads.ui.ground import GroundLayout, GroundPictures, Pixels, cell_colors
from railroads.ui.hex_details import describe, draw_hex_details
from railroads.ui.iso_camera import IsoCamera, Vec2
from railroads.ui.map_input import MapInput
from railroads.ui.map_overlays import (
    SELECTED_COLOR,
    draw_compass,
    draw_found,
    draw_found_on_overview,
    draw_grid_lines,
    draw_hovered_buildings,
    draw_outline,
    draw_station_names,
    draw_status,
    status_text,
    visible_area,
)
from railroads.ui.map_tools import MapTools
from railroads.ui.minimap import Minimap, MinimapLayout, view_corners
from railroads.ui.pixel.ground import GLOW, glow_strength
from railroads.ui.pixel.layer import PixelLayer
from railroads.ui.pixel.relief import Elevation
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.track_preview import TrackPreview
from railroads.ui.pixel.train_layer import TrainLayer
from railroads.ui.pixel.world import PixelWorld
from railroads.ui.search import BuildingSearch
from railroads.ui.station_window import StationWindow
from railroads.ui.summaries_panel import SummariesPanel
from railroads.ui.theme import (
    BRASS,
)

BACKGROUND_COLOR = (0.08, 0.09, 0.11, 1.0)
DETAILS_POS = (16.0, 96.0)
DETAILS_SIZE = (680.0, 340.0)
# Stepping to a building found zooms in at least this far, so it can be seen.
VISIT_ZOOM = 4
# Slopes are lit from the top-left of the screen.
LIGHT_ON_SCREEN = (-1.0, -1.0)
# Population points a cell needs to show as a town on the minimap.
MINIMAP_TOWN_POINTS = 12


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


@cache
def _sprite_sheet() -> SpriteSheet:
    return SpriteSheet.load()


@cache
def _fine_sprite_sheet() -> SpriteSheet:
    return SpriteSheet.load_fine()


class OnGround:
    """A camera that puts map points on the raised ground of the pixel map: like the camera it
    stands for, but each point lifted by the ground under it (railroads.ui.pixel.relief)."""

    def __init__(self, camera: IsoCamera, elevation: Elevation) -> None:
        self._camera = camera
        self._elevation = elevation

    def __getattr__(self, name: str) -> object:
        return getattr(self._camera, name)

    def to_screen(self, x: Coord, y: Coord, screen_center: Vec2) -> tuple[Coord, Coord]:
        sx, sy = self._camera.to_screen(x, y, screen_center)
        lift = self._elevation.lift(x, y) * self._camera.pixel_scale
        if np.ndim(lift) == 0:
            return float(sx), float(sy - lift)
        return sx, sy - lift


class MapView:
    def __init__(self, game: Game, world: WorldLand | None = None) -> None:
        self.game = game
        tile_map = game.tile_map
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
        self._pictures: GroundPictures | None = None
        self._pixels: PixelLayer | None = None
        self._trains: TrainLayer | None = None
        self._preview: TrackPreview | None = None
        self._textures: dict[bool, hello_imgui.TextureGpu] = {}
        self.hovered: Cell | None = None
        # The cell whose details are open, picked by clicking without dragging.
        self.selected: Cell | None = None
        self.build = BuildMenu()
        # What came of the last Buy pressed in the hex details.
        self._bought = ""
        # The station levels the map shows, to redraw stations raised since.
        self._shown_levels: dict[Cell, int] = {}
        # What the build tools do with clicks, and the mouse and keys (railroads.ui.map_tools,
        # railroads.ui.map_input).
        self.tools = MapTools(self)
        self.input = MapInput(self)
        self._map_rect: tuple[Vec2, Vec2] = ((0.0, 0.0), (0.0, 0.0))
        self.summaries = SummariesPanel()
        self.station_window = StationWindow()
        self.search = BuildingSearch()

    @property
    def tile_map(self) -> TileMap:
        return self.game.tile_map

    @property
    def constructions(self) -> Constructions:
        return self.game.constructions

    @property
    def growth(self) -> Growth:
        return self.game.growth

    @property
    def railway(self) -> Railway:
        return self.game.railway

    def gui(self) -> None:
        if self.railway.levels != self._shown_levels:
            raised = {
                cell
                for cell, level in self.railway.levels.items()
                if self._shown_levels.get(cell) != level
            }
            self._shown_levels = dict(self.railway.levels)
            self.buildings_changed(raised)
        self.input.handle_keys()
        self._draw_map()
        self._draw_details()
        self.build.draw(*self._map_rect)
        self.build.trains.draw(self.railway, self.growth.station_trade.supplies)
        self.build.stations.draw(self.railway, self.look_at, self.open_station)
        self.station_window.draw(
            self.railway,
            self.look_at,
            self.growth.station_trade.measure_demand,
            self.tools.start_moving,
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
        self.game.time_passed(months)

    def _draw_details(self) -> None:
        if self.selected is None:
            return
        imgui.set_next_window_pos(DETAILS_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(DETAILS_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, still_open = imgui.begin("Hex details", True)
        if expanded:
            bought = draw_hex_details(describe(self.growth, self.selected))
            if bought is not None:
                self._bought = (
                    self.growth.owners.buy(self.selected, bought) or f"You own a {bought.value}"
                )
            if self._bought:
                imgui.text_disabled(self._bought)
        imgui.end()
        if not still_open:
            self.selected = None
            self._bought = ""

    def month_passed(self, rng: np.random.Generator, date: str = "") -> None:
        """Lets a month of the game pass and shows what changed."""
        month = self.game.month(rng, date)
        changed = month.changed
        if month.summary is not None:
            self.summaries.add(month.summary)
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

    def _ground_layout(self) -> GroundLayout:
        if self._layout is None:
            self._layout = GroundLayout.of(self.tile_map.grid)
        return self._layout

    def _ground_pictures(self) -> GroundPictures:
        if self._pictures is None:
            self._pictures = GroundPictures(self._ground_layout(), self.tile_map.relief)
        return self._pictures

    def _ground_texture(self, with_lines: bool) -> hello_imgui.TextureGpu:
        """The smooth ground picture used when zoomed out, with or without grid lines."""
        if with_lines not in self._textures:
            light = self.camera.map_direction(*LIGHT_ON_SCREEN)
            rgba = self._ground_pictures().picture(self._ground_colors(), light, with_lines)
            self._textures[with_lines] = hello_imgui.create_texture_gpu_from_rgba_data(rgba)
        return self._textures[with_lines]

    def _ground_colors(self) -> Pixels:
        """Every cell's colour on the ground picture, tinted towards GLOW on land that is served."""
        colors = cell_colors(self.tile_map)
        strength = glow_strength(self._satisfaction).ravel()
        land = (self.tile_map.terrain == Terrain.LAND).ravel()
        glowing = np.flatnonzero(land & (strength > 0))
        share = strength[glowing, None]
        flat = colors.reshape(-1, 4)
        tinted = flat[glowing, :3] * (1 - share) + np.array(GLOW) * share
        flat[glowing, :3] = tinted.astype(np.uint8)
        return colors

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
            MinimapLayout.of(
                map_size, bottom_right, shown, world_shown, self.minimap.zoom, self.minimap.focus
            )
            if shown or world_shown
            else None
        )
        self.input.handle_mouse(center, minimap)

        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(top_left, bottom_right, True)
        draw_list.add_rect_filled(top_left, bottom_right, _color(BACKGROUND_COLOR))
        if self.camera.pixel_scale:
            self._draw_ground(draw_list, center, self._ground_texture(False))
            self._pixel_layer().draw(draw_list, self.camera, center, (top_left, bottom_right))
        else:
            self._draw_smooth_map(draw_list, center, top_left, bottom_right)
        if self.camera.pixel_scale:
            if self.railway.trains:
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
        marks = self._marks_camera()
        if self.build.track_from is not None:
            draw_outline(
                draw_list,
                marks,
                self.tile_map.grid,
                center,
                self.build.track_from,
                SELECTED_COLOR,
            )
        if self.hovered is not None:
            draw_outline(draw_list, marks, self.tile_map.grid, center, self.hovered)
            hovering_station = any(
                isinstance(built, Station) for built in self.constructions.at(self.hovered)
            )
            if self.build.tool == Tool.STATION or hovering_station:
                stations = self.railway.stations() if self.build.tool == Tool.STATION else []
                draw_station_reach(
                    draw_list,
                    self.tile_map.grid,
                    self.hovered,
                    stations,
                    marks,
                    center,
                    self.railway.reach,
                )
        opened = self.station_window.station
        if (
            opened is not None
            and (
                self.hovered is None
                or self.hovered == opened
                or not any(
                    isinstance(built, Station) for built in self.constructions.at(self.hovered)
                )
            )
            and self.build.tool != Tool.STATION
        ):
            draw_station_reach(
                draw_list, self.tile_map.grid, opened, [], marks, center, self.railway.reach
            )
        if self.selected is not None:
            draw_outline(
                draw_list, marks, self.tile_map.grid, center, self.selected, SELECTED_COLOR
            )
        for cell in self.summaries.outlined():
            draw_outline(draw_list, marks, self.tile_map.grid, center, cell, BRASS)
        draw_found(
            draw_list, marks, self.tile_map.grid, self.search, center, top_left, bottom_right
        )
        draw_station_names(draw_list, marks, self.tile_map.grid, self.railway, center)
        draw_compass(draw_list, self.camera, (bottom_right[0] - 36, top_left[1] + 36))
        self._draw_status(draw_list, top_left, bottom_right)
        if minimap is not None:
            corners = view_corners(self.camera, center, top_left, bottom_right)
            ground = imgui.ImTextureRef(self._ground_texture(False).texture_id())
            textures = (self.minimap.world_texture(), ground)
            self.minimap.draw(draw_list, minimap, textures, corners)
            draw_found_on_overview(draw_list, self.tile_map.grid, self.search, minimap)
        draw_list.pop_clip_rect()

    def _draw_smooth_map(
        self, draw_list: imgui.ImDrawList, center: Vec2, top_left: Vec2, bottom_right: Vec2
    ) -> None:
        cell_width = self.camera.cell_width
        drawn_lines = cell_width >= self._look.drawn_lines_from
        baked_from = self._look.baked_lines_from
        baked_lines = not drawn_lines and baked_from is not None and cell_width >= baked_from
        self._draw_ground(draw_list, center, self._ground_texture(baked_lines))
        if drawn_lines:
            visible = visible_area(self.camera, center, top_left, bottom_right)
            draw_grid_lines(draw_list, self.camera, self.tile_map.grid, center, visible)
        draw_building_dots(draw_list, self.camera, center, (top_left, bottom_right), self._spots)

    def _marks_camera(self) -> IsoCamera | OnGround:
        """The camera marks on the map are drawn with: lifted onto raised ground in pixel art."""
        if not self.camera.pixel_scale:
            return self.camera
        return OnGround(self.camera, self._pixel_layer().world.elevation)

    def ground_under(self, screen_x: float, screen_y: float, center: Vec2) -> tuple[float, float]:
        """The map point showing at a spot on screen, on raised ground where it rises."""
        x, y = self.camera.to_plane(screen_x, screen_y, center)
        if not self.camera.pixel_scale:
            return x, y
        world = self._pixel_layer().world
        px, py = world.projection.to_pixels(x, y)
        return world.elevation.ground_at(float(px), float(py))

    def _draw_status(self, draw_list: imgui.ImDrawList, top_left: Vec2, bottom_right: Vec2) -> None:
        """What is under the mouse, in a strip along the top of the map, and the card of the
        hovered cell's buildings below it."""
        over_minimap = self.input.over_minimap
        text = status_text(
            self.camera,
            self.tile_map,
            self.constructions,
            self._satisfaction,
            self.hovered,
            over_minimap,
        )
        below = draw_status(draw_list, top_left, bottom_right, text)
        if self.hovered is not None and not over_minimap:
            draw_hovered_buildings(self.growth, self.hovered, below)

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
            self.buildings_changed(set(self.railway.levels))
        return self._pixels

    def simulate(self, years: int, rng: np.random.Generator) -> None:
        """Lets a stand-in player build stations and serve the towns for `years`, printing the
        map every ten years, and goes on playing for the player from there."""
        self.game.simulate(years, rng)
        self._satisfaction = self.growth.satisfaction()
        self.buildings_changed(set())

    def buildings_changed(self, changed: set[Cell]) -> None:
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
