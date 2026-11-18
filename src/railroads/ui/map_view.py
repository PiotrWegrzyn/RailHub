"""Draws one TileMap. Zoomed out, the ground is one smooth texture with the cell grid and
buildings as dots on top; close up, it is pixel art (see railroads.ui.pixel). The hovered cell, a
compass and the minimap go over both."""

from enum import Enum
from functools import cache

import numpy as np
import numpy.typing as npt
from imgui_bundle import hello_imgui, imgui
from imgui_bundle import icons_fontawesome_4 as fa

from railroads import maps
from railroads.mapgen.world import WorldLand
from railroads.sim.constructions import (
    Constructions,
    Station,
)
from railroads.sim.economy import Trade
from railroads.sim.game import Game
from railroads.sim.geo import Coord
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.railway import Railway
from railroads.sim.steps import Steps, finish
from railroads.sim.tilemap import Terrain, TileMap
from railroads.ui.build_menu import (
    BuildMenu,
    Tool,
    draw_station_reach,
    draw_track,
    draw_trains,
)
from railroads.ui.building_spots import BuildingSpots
from railroads.ui.buildings import draw_building_dots
from railroads.ui.ground import GroundLayout, GroundPictures, Pixels, cell_colors
from railroads.ui.hex_details import draw_hex_details, recently_described
from railroads.ui.iso_camera import LOOKS, IsoCamera, Vec2
from railroads.ui.land_offers import LandOffers
from railroads.ui.map_input import MapInput
from railroads.ui.map_overlays import (
    MOODS_LEGEND,
    SELECTED_COLOR,
    TOWNS_LEGEND,
    consumers_in_reach,
    draw_compass,
    draw_found,
    draw_found_on_overview,
    draw_grid_lines,
    draw_hand_out,
    draw_hovered_buildings,
    draw_moods,
    draw_outline,
    draw_owned_flags,
    draw_station_names,
    draw_status,
    least_satisfied,
    mood_tint,
    status_text,
    tint_colours,
    town_cells,
    visible_area,
)
from railroads.ui.map_tools import MapTools
from railroads.ui.minimap import Minimap, MinimapLayout, view_corners
from railroads.ui.pixel.ground import GLOW, glow_strength
from railroads.ui.pixel.ground_shader import (
    GroundShader,
    Screen,
    View,
    cell_info,
    free_textures,
    screen_of,
    view_of,
)
from railroads.ui.pixel.layer import PixelLayer
from railroads.ui.pixel.relief import Elevation
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.track_preview import TrackPreview
from railroads.ui.pixel.train_layer import TrainLayer, engine_picture
from railroads.ui.pixel.world import PixelWorld
from railroads.ui.route_map import soften, soften_steps
from railroads.ui.search import BuildingSearch
from railroads.ui.spotlight import pulsing_brass, spot
from railroads.ui.station_window import StationWindow
from railroads.ui.summaries_panel import SummariesPanel
from railroads.ui.theme import BRASS, OVERLAY_FLAGS, SIDE_WINDOW_POS, rgba_u32, toggle_button

BACKGROUND_COLOR = (0.08, 0.09, 0.11, 1.0)
DETAILS_SIZE = (680.0, 340.0)
# Stepping to a building found zooms in at least this far, so it can be seen.
VISIT_ZOOM = 4
# Slopes are lit from the top-left of the screen.
LIGHT_ON_SCREEN = (-1.0, -1.0)
# Population points a cell needs to show as a town on the minimap.
MINIMAP_TOWN_POINTS = 12
# How far above the minimap, or the map's corner without one, the lens buttons sit.
LENS_GAP = 8.0
# How thick the outline is round a cell the tutorial wants clicked.
SPOT_WIDTH = 4.0


class Lens(Enum):
    """What the map can be tinted by: how satisfied buildings are, or which town they are in."""

    SATISFACTION = "satisfaction"
    TOWNS = "towns"


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
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            lift = self._elevation.lift_at(float(x), float(y)) * self._camera.pixel_scale
            return float(sx), float(sy - lift)
        return sx, sy - self._elevation.lift(x, y) * self._camera.pixel_scale


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
        # Each engine's picture for the buy screen, by engine and how many times it is blown up:
        # its texture and its size in art pixels.
        self._engine_pictures: dict[tuple[str, int], tuple[hello_imgui.TextureGpu, Vec2]] = {}
        self._preview: TrackPreview | None = None
        self._textures: dict[bool, hello_imgui.TextureGpu] = {}
        # Which of those the frame drawn last showed, the ones a month paints again in its colours.
        self._textures_shown: set[bool] = set()
        # How many times the cells' colours on the ground pictures have changed.
        self._ground_version = 0
        # The ground picture softened for the trains window's map, with the version of the colours
        # it shows: its pixels, softened with the map or anew a step a frame (_soften_on), and its
        # texture.
        self._soft: tuple[int, Pixels] | None = None
        self._softening: Steps[tuple[int, Pixels]] | None = None
        self._soft_texture: tuple[int, hello_imgui.TextureGpu] | None = None
        # The ground shader for the pixel map, once tried (railroads.ui.pixel.ground_shader), and
        # how many times the cells' colours have changed, so it takes them up again.
        self._shader: GroundShader | None = None
        self._shader_tried = False
        self._colors_version = 0
        self.hovered: Cell | None = None
        # The art pixel last looked up under the mouse and the ground point showing there, as the
        # mouse often rests while frames go on.
        self._under: tuple[tuple[float, float], tuple[float, float]] | None = None
        # The cell whose details are open, picked by clicking without dragging.
        self.selected: Cell | None = None
        # The buildings that take what the selected cell makes, and the cell, buildings and year
        # they were found for.
        self._consumers: tuple[tuple[Cell, int, int], tuple[Cell, ...]] | None = None
        # The cells Shift tints by how satisfied their buildings are, their packed colours, and
        # the month's trade and buildings they were worked out for.
        self._moods: tuple[tuple[Trade | None, int], npt.NDArray[np.intp], list[int]] | None = None
        # The lens turned on above the minimap, if any, and the town tints with the buildings
        # they were worked out for.
        self.lens: Lens | None = None
        self._towns: tuple[int, npt.NDArray[np.intp], list[int]] | None = None
        # The top of the minimap on screen, which the lens buttons sit above.
        self._minimap_top: float | None = None
        self.build = BuildMenu()
        # What came of the last Buy pressed in the hex details.
        self._bought = ""
        # The station levels the map shows, to redraw stations raised since.
        self._shown_levels: dict[Cell, int] = dict(self.railway.levels)
        # What the build tools do with clicks, and the mouse and keys (railroads.ui.map_tools,
        # railroads.ui.map_input).
        self.tools = MapTools(self)
        self.input = MapInput(self)
        self._map_rect: tuple[Vec2, Vec2] = ((0.0, 0.0), (0.0, 0.0))
        self.summaries = SummariesPanel()
        self.station_window = StationWindow()
        self.search = BuildingSearch()
        # The land around the map the player can buy (railroads.ui.land_offers).
        self.land = LandOffers()
        # The cells the tutorial wants clicked, marked on the map (railroads.ui.spotlight).
        self.spotlit: list[Cell] = []

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
        self._textures_shown.clear()
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
        self.build.maglev_open = self.railway.maglev_open
        self.build.draw(*self._map_rect)
        self._draw_lenses()
        self.build.trains.draw(
            self.railway,
            self.growth.station_trade.supplies,
            self._ground_picture,
            self._engine_picture,
        )
        self.build.stations.draw(self.railway, self.look_at, self.open_station)
        self.station_window.draw(
            self.railway,
            self.look_at,
            self.growth.station_trade.measure_demand,
            self.tools.start_moving,
            self.growth.station_trade.supplies,
            self.tools.start_adding_platform,
        )
        self.summaries.draw(*self._map_rect, self.look_at, self.railway.name_of)
        self.search.draw(self.growth, self.visit, self._view_center)
        self.land.confirm(self.game.lots, self.game.finances)

    def _view_center(self) -> Vec2:
        return self.camera.center_x, self.camera.center_y

    def visit(self, cell: Cell) -> None:
        """Takes the view to a cell, close enough to see what stands there."""
        self.camera.zoom_level = max(self.camera.zoom_level, VISIT_ZOOM)
        self.look_at(cell)

    def visit_biggest_town(self) -> None:
        """Takes the view to the cell with the most people around it, and of those to the one with
        the most people on it."""
        near = self.growth.nearby()
        on = np.where(near == near.max(), self.growth.points(), -1)
        row, col = np.unravel_index(int(np.argmax(on)), on.shape)
        self.visit((int(row), int(col)))

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
        imgui.set_next_window_pos(SIDE_WINDOW_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(DETAILS_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, still_open = imgui.begin("Hex details", True)
        if expanded:
            details = recently_described(self.growth, self.selected, self._satisfaction)
            pressed = draw_hex_details(details)
            if pressed is not None:
                kind, buying = pressed
                owners = self.growth.owners
                if buying:
                    said = owners.buy(self.selected, kind)
                else:
                    said = owners.sell(self.selected, kind)
                held = int(owners.shares(kind)[self.selected])
                self._bought = said or f"You own {held} shares of the {kind.value}s here"
            if self._bought:
                imgui.text_disabled(self._bought)
        imgui.end()
        if not still_open:
            self.selected = None
            self._bought = ""

    def month_passed(self, rng: np.random.Generator, date: str = "") -> None:
        """Lets a month of the game pass and shows what changed."""
        finish(self.month_steps(rng, date))

    def month_steps(self, rng: np.random.Generator, date: str = "") -> Steps[None]:
        """month_passed in a few steps, which the app spreads over frames so no frame stalls on
        the whole month; nothing may change the game between them."""
        month = yield from self.game.month_steps(rng, date)
        yield
        changed = month.changed
        if month.summary is not None:
            self.summaries.add(month.summary)
        satisfaction = self.growth.satisfaction()
        moods_changed = not np.array_equal(satisfaction, self._satisfaction)
        self._satisfaction = satisfaction
        if moods_changed or self.growth.forests_changed:
            self._ground_version += 1
            self._colors_version += 1
            if self._pixels is not None:
                self._pixels.refresh(self._pixels.world.satisfaction_changed(satisfaction))
            # The ground pictures the last frame showed are painted again in the new colours in
            # steps of their own, the old ones showing until then, rather than all on one frame.
            # The others are dropped, to be painted when they next show.
            for with_lines in set(self._textures) - self._textures_shown:
                del self._textures[with_lines]
            for with_lines in list(self._textures):
                yield
                self._textures.pop(with_lines, None)
                self._ground_texture(with_lines)
        if not changed:
            return
        yield
        self._spots = BuildingSpots.of(self.constructions)
        self.minimap.show_towns(self._town_cells())
        if self.search.open:
            self.search.refresh(self.growth)
        if self._pixels is not None:
            yield
            chunks = self._pixels.world.buildings_changed(changed)
            # Copies of the world painting elsewhere hear of the change on a frame of its own;
            # asking for the changed chunks on this one would add that to it.
            yield
            self._pixels.catch_up()
            self._pixels.refresh(chunks)
            self._colors_version += 1

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
            self._pictures = GroundPictures(
                self._ground_layout(), self.tile_map.relief, self.tile_map.grid
            )
        return self._pictures

    def _ground_picture(self) -> tuple[imgui.ImTextureRef, Vec2]:
        """The ground picture softened for the trains window's map, and its size in map units.
        Once the cells' colours change it is softened anew a step each frame it is asked for,
        the old picture showing meanwhile."""
        if self._soft is None:
            self._soft = (self._ground_version, soften(self._plain_picture()))
        elif self._soft[0] != self._ground_version:
            self._soften_on()
        version, pixels = self._soft
        if self._soft_texture is None or self._soft_texture[0] != version:
            texture = hello_imgui.create_texture_gpu_from_rgba_data(pixels)
            self._soft_texture = (version, texture)
        texture_id = self._soft_texture[1].texture_id()
        return imgui.ImTextureRef(texture_id), self._ground_layout().size

    def _soften_on(self) -> None:
        """Takes the next step of softening the ground picture in the cells' new colours."""
        if self._softening is None:
            self._softening = self._softened()
        try:
            next(self._softening)
        except StopIteration as done:
            self._soft = done.value
            self._softening = None

    def _softened(self) -> Steps[tuple[int, Pixels]]:
        version = self._ground_version
        colors = self._ground_colors()
        yield
        light = self.camera.map_direction(*LIGHT_ON_SCREEN)
        # A copy, as the kept picture is painted over in place when the colours change again.
        rgba = self._ground_pictures().picture(colors, light, False).copy()
        yield
        pixels = yield from soften_steps(rgba)
        return version, pixels

    def _plain_picture(self) -> Pixels:
        """The ground picture without grid lines, in the cells' colours now."""
        light = self.camera.map_direction(*LIGHT_ON_SCREEN)
        return self._ground_pictures().picture(self._ground_colors(), light, False)

    def _ground_texture(self, with_lines: bool) -> hello_imgui.TextureGpu:
        """The smooth ground picture used when zoomed out, with or without grid lines."""
        if with_lines not in self._textures:
            light = self.camera.map_direction(*LIGHT_ON_SCREEN)
            rgba = self._ground_pictures().picture(self._ground_colors(), light, with_lines)
            self._textures[with_lines] = hello_imgui.create_texture_gpu_from_rgba_data(rgba)
        self._textures_shown.add(with_lines)
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
        self._minimap_top = (
            None
            if minimap is None
            else min(panel.top for panel in (minimap.overview, minimap.world) if panel is not None)
        )
        self.input.handle_mouse(center, minimap)

        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(top_left, bottom_right, True)
        draw_list.add_rect_filled(top_left, bottom_right, rgba_u32(BACKGROUND_COLOR))
        if self.camera.pixel_scale:
            shaded = self._draw_shaded_ground(draw_list, center, top_left, bottom_right)
            if not shaded:
                self._draw_ground(draw_list, center, self._ground_texture(False))
            layer = self._pixel_layer()
            layer.over_ground = shaded
            layer.blow_up = self._shader.blow_up if shaded and self._shader is not None else None
            layer.dress = self._shader.dress if shaded and self._shader is not None else None
            ready = layer.draw(
                draw_list, self.camera, center, (top_left, bottom_right), ground_below=shaded
            )
            if not ready:
                # Keeps frames coming while chunks are drawn, so they show without input.
                hello_imgui.request_refresh()
        else:
            self._draw_smooth_map(draw_list, center, top_left, bottom_right)
        lots = self.game.lots
        if lots is not None:
            world = self.minimap.world_texture()
            cash = self.game.finances.cash
            self.land.draw(draw_list, self.camera, center, self.tile_map, lots, world, cash)
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
        tints = (
            self._mood_tints()
            if self._showing_moods()
            else self._town_tints()
            if self.lens == Lens.TOWNS
            else None
        )
        if tints is not None and not self._draw_shaded_tints(
            draw_list, center, top_left, bottom_right, *tints
        ):
            draw_moods(
                draw_list, marks, self.tile_map.grid, center, (top_left, bottom_right), *tints
            )
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
            hovered_station = self.constructions.main_station(self.hovered)
            if self.build.tool == Tool.STATION or hovered_station is not None:
                stations = self.railway.stations() if self.build.tool == Tool.STATION else []
                draw_station_reach(
                    draw_list,
                    self.tile_map.grid,
                    self.hovered if hovered_station is None else hovered_station,
                    stations,
                    marks,
                    center,
                    self.railway.reach,
                    self.constructions.platforms,
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
                draw_list,
                self.tile_map.grid,
                opened,
                [],
                marks,
                center,
                self.railway.reach,
                self.constructions.platforms,
            )
        if self.selected is not None:
            consumers = self._consumers_of(self.selected)
            if consumers:
                draw_hand_out(
                    draw_list, marks, self.tile_map.grid, center, self.selected, consumers
                )
            draw_outline(
                draw_list, marks, self.tile_map.grid, center, self.selected, SELECTED_COLOR
            )
        for cell in self.summaries.outlined():
            draw_outline(draw_list, marks, self.tile_map.grid, center, cell, BRASS)
        for cell in self.spotlit:
            draw_outline(
                draw_list, marks, self.tile_map.grid, center, cell, pulsing_brass(), SPOT_WIDTH
            )
        if self.spotlit:
            hello_imgui.request_refresh()
        draw_found(
            draw_list, marks, self.tile_map.grid, self.search, center, top_left, bottom_right
        )
        draw_owned_flags(draw_list, marks, self.tile_map.grid, self.growth.owners, center)
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

    def _consumers_of(self, cell: Cell) -> tuple[Cell, ...]:
        """consumers_in_reach of the cell, found again only once its buildings or the year
        changed: the hex details are drawn every frame."""
        key = (cell, self.constructions.version, self.growth.year)
        if self._consumers is None or self._consumers[0] != key:
            self._consumers = key, consumers_in_reach(self.constructions, cell, self.growth.year)
        return self._consumers[1]

    def _draw_lenses(self) -> None:
        """Buttons over the bottom-right corner of the map, above the minimap, that turn a lens
        on or off: one at a time."""
        _, (right, bottom) = self._map_rect
        top = self._minimap_top if self._minimap_top is not None else bottom
        imgui.set_next_window_pos(
            (right - LENS_GAP, top - LENS_GAP), imgui.Cond_.always.value, (1.0, 1.0)
        )
        imgui.begin("##lenses", flags=OVERLAY_FLAGS | imgui.WindowFlags_.no_move.value)
        for lens, label, tooltip in (
            (
                Lens.SATISFACTION,
                f"{fa.ICON_FA_SMILE}  Satisfaction",
                "Tint the land by how satisfied its buildings are (or hold Shift)",
            ),
            (
                Lens.TOWNS,
                f"{fa.ICON_FA_USERS}  Towns",
                "Tint the land by town; passengers and mail only travel between towns",
            ),
        ):
            if lens != Lens.SATISFACTION:
                imgui.same_line()
            if toggle_button(label, self.lens == lens, tooltip):
                self.lens = None if self.lens == lens else lens
            spot(f"lens:{lens.value}")
        imgui.end()

    def _showing_moods(self) -> bool:
        """Whether the satisfaction lens is on, or Shift is held for it, not while typing."""
        io = imgui.get_io()
        return self.lens == Lens.SATISFACTION or (io.key_shift and not io.want_text_input)

    def _town_tints(self) -> tuple[npt.NDArray[np.intp], list[int]]:
        """town_cells, worked out again only once the buildings changed."""
        version = self.constructions.version
        if self._towns is None or self._towns[0] != version:
            self._towns = (version, *town_cells(self.growth.towns))
        return self._towns[1], self._towns[2]

    def _mood_tints(self) -> tuple[npt.NDArray[np.intp], list[int]]:
        """least_satisfied's cells and their tints, worked out again only once the month's trade
        or the buildings changed."""
        key = (self.growth.trade, self.constructions.version)
        if self._moods is None or self._moods[0][0] is not key[0] or self._moods[0][1] != key[1]:
            cells, shares = least_satisfied(self.growth.trade, self.growth.assumed_shares)
            colors = [rgba_u32(mood_tint(share)) for share in shares.tolist()]
            self._moods = key, cells, colors
        return self._moods[1], self._moods[2]

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
        pixel = (float(px), float(py))
        if self._under is None or self._under[0] != pixel:
            self._under = (pixel, world.elevation.ground_at(*pixel))
        return self._under[1]

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
        if self._showing_moods():
            text = f"{text} {MOODS_LEGEND}"
        elif self.lens == Lens.TOWNS:
            text = f"{text} {TOWNS_LEGEND}"
        below = draw_status(draw_list, top_left, bottom_right, text)
        if self.hovered is not None and not over_minimap:
            draw_hovered_buildings(self.growth, self.hovered, below, self._satisfaction)

    def _train_layer(self) -> TrainLayer:
        if self._trains is None:
            self._trains = TrainLayer(self._pixel_layer().world, self.railway.trains)
        return self._trains

    def _engine_picture(self, engine: str, times: int) -> tuple[imgui.ImTextureRef, Vec2]:
        """An engine as the map draws it, drawn `times` times as fine each way with its fine
        details, for the buy screen, and its size in the sheet's pixels."""
        kept = self._engine_pictures.get((engine, times))
        if kept is None:
            layer = self._train_layer()
            pixels = engine_picture(layer.sprites, layer.world.projection.matrix, engine, times)
            texture = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(pixels))
            kept = (texture, (pixels.shape[1] / times, pixels.shape[0] / times))
            self._engine_pictures[(engine, times)] = kept
        return imgui.ImTextureRef(kept[0].texture_id()), kept[1]

    def _track_preview(self) -> TrackPreview:
        if self._preview is None:
            self._preview = TrackPreview(self._pixel_layer().world)
        return self._preview

    def prepare(self, size: Vec2, density: int, chunks: bool = True) -> None:
        """Builds what the map needs before its first frame, in a window of `size` screen pixels
        with `density` device pixels to each, so that work can run while a game loads: the
        ground picture with and without grid lines and its softened copy for the trains window,
        the pixel world, the trains' sprites and, with `chunks`, the chunks the first frame
        shows. Nothing here touches the graphics card."""
        finish(self.prepare_steps(size, density, chunks))

    def prepare_steps(self, size: Vec2, density: int, chunks: bool = True) -> Steps[None]:
        """prepare a part at a time, for a page to go on drawing frames between them."""
        plain = self._plain_picture()
        yield
        self._soft = (self._ground_version, (yield from soften_steps(plain)))
        light = self.camera.map_direction(*LIGHT_ON_SCREEN)
        self._ground_pictures().picture(self._ground_colors(), light, True)
        yield
        if self._pixels is None:
            yield from self._pixel_layer_steps()
        layer = self._pixel_layer()
        if self.railway.trains:
            yield
            self._train_layer()
        if chunks and self.camera.pixel_scale:
            yield
            center = (size[0] / 2, size[1] / 2)
            layer.prepare(self.camera, center, ((0.0, 0.0), size), density)

    def _pixel_layer(self) -> PixelLayer:
        if self._pixels is None:
            finish(self._pixel_layer_steps())
        assert self._pixels is not None
        return self._pixels

    def _pixel_layer_steps(self) -> Steps[None]:
        world = yield from PixelWorld.built(
            self.tile_map,
            self.constructions,
            _sprite_sheet(),
            self._look.angle_deg,
            self._look.cell_units,
            fine=_fine_sprite_sheet(),
            station_levels=self.railway.levels,
        )
        world.satisfaction = self._satisfaction.copy()
        self._pixels = PixelLayer(world)

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
            self._colors_version += 1

    def _draw_shaded_ground(
        self, draw_list: imgui.ImDrawList, center: Vec2, top_left: Vec2, bottom_right: Vec2
    ) -> bool:
        """Draws the pixel map's ground with the shader, if it runs here; returns whether it
        did."""
        shader = self._ground_shader()
        if shader is None:
            return False
        world = self._pixel_layer().world
        if shader.cells_version != self._colors_version:
            ground = world.ground
            info = cell_info(
                ground.biomes,
                self.tile_map.deposits,
                ground.hills,
                glow_strength(ground.satisfaction),
                ground.mountains,
            )
            shader.set_cells(info, self._colors_version)
        if shader.surfaces_version != self._colors_version:
            shader.set_surfaces(world.ground, self._colors_version)
        elevation = world.elevation
        grades = elevation.grades
        if shader.bed_version != grades:
            shader.set_bed(elevation.bed, grades)
        view, screen = self._shaded_view(center, top_left, bottom_right)
        texture = shader.picture(view, screen, blend=True)
        draw_list.add_image(imgui.ImTextureRef(texture), top_left, bottom_right)
        free_textures()
        return True

    def _shaded_view(self, center: Vec2, top_left: Vec2, bottom_right: Vec2) -> tuple[View, Screen]:
        """The art pixels of the pixel map showing in the map area, and where they go on screen."""
        camera_pixel = self._pixel_layer().world.projection.to_pixels(
            self.camera.center_x, self.camera.center_y
        )
        scale = self.camera.pixel_scale
        origin = (
            round(center[0] - float(camera_pixel[0]) * scale),
            round(center[1] - float(camera_pixel[1]) * scale),
        )
        size = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        return view_of(origin, scale, top_left, size), screen_of(
            origin, scale, top_left, size, density
        )

    def _draw_shaded_tints(
        self,
        draw_list: imgui.ImDrawList,
        center: Vec2,
        top_left: Vec2,
        bottom_right: Vec2,
        cells: npt.NDArray[np.intp],
        colors: list[int],
    ) -> bool:
        """Tints the cells in their packed colours with the shader, each pixel by the cell it
        shows, if it runs here and the cells are hexagons; returns whether it did."""
        shader = self._ground_shader() if self.tile_map.grid.kind == "hex" else None
        if shader is None:
            return False
        if shader.tints_version is not colors:
            shader.set_tints(tint_colours(self.tile_map.grid.shape, cells, colors), colors)
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        size = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
        pixels = (round(size[0] * density), round(size[1] * density))
        e, f = self.camera.to_plane(top_left[0], top_left[1], center)
        a, c = self.camera.to_plane(top_left[0] + 1 / density, top_left[1], center)
        b, d = self.camera.to_plane(top_left[0], top_left[1] + 1 / density, center)
        flat = (a - e, b - e, c - f, d - f, e, f)
        raised = None
        if self.camera.pixel_scale:
            raised = self._shaded_view(center, top_left, bottom_right)
        texture = shader.tinted(pixels, flat, raised)
        draw_list.add_image(imgui.ImTextureRef(texture), top_left, bottom_right)
        return True

    def _ground_shader(self) -> GroundShader | None:
        """The ground shader, made the first time the pixel map shows in a real window."""
        if not self._shader_tried:
            self._shader_tried = True
            if hello_imgui.is_using_hello_imgui():
                world = self._pixel_layer().world
                self._shader = GroundShader.make(
                    world.projection,
                    self.tile_map.grid.shape,
                    world.ground.shading,
                    world.elevation.heights,
                )
        return self._shader

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
