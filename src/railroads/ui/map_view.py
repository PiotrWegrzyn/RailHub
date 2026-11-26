"""Draws one TileMap. Zoomed out, the ground is one smooth texture with the cell grid and
buildings as dots on top; close up, it is pixel art (see railroads.ui.pixel). The hovered cell, a
compass and the minimap go over both."""

import math
from enum import Enum
from functools import cache

import numpy as np
import numpy.typing as npt
from imgui_bundle import hello_imgui, imgui
from imgui_bundle import icons_fontawesome_4 as fa

from railroads import maps
from railroads.kept import Kept, KeptFor
from railroads.mapgen.world import WorldLand
from railroads.sim.constructions import (
    BuildingKind,
    Constructions,
    Station,
)
from railroads.sim.game import Game
from railroads.sim.geo import Coord
from railroads.sim.goods import Good
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.ladder_types import Ladder
from railroads.sim.railway import Railway
from railroads.sim.station_trade import StationPreview
from railroads.sim.steps import Steps, finish
from railroads.sim.tilemap import Terrain, TileMap
from railroads.sim.year_caches import boards_changed
from railroads.ui.build_menu import (
    BuildMenu,
    Tool,
    draw_station_reach,
    draw_track,
)
from railroads.ui.building_spots import BuildingSpots
from railroads.ui.buildings import dot_looks, draw_building_dots
from railroads.ui.goods_search import draw_cell_good, draw_goods_legend
from railroads.ui.ground_picture import GroundLayout, GroundPictures, Pixels, cell_colors
from railroads.ui.growth_window import GrowthWindow, draw_cell_odds, draw_legend
from railroads.ui.hex_details import draw_hex_details, draw_shares
from railroads.ui.hex_rows import HexDetails, recently_described
from railroads.ui.iso_camera import LOOKS, PIXEL_CELL_WIDTH, ZOOM_CELL_WIDTHS, IsoCamera, Vec2
from railroads.ui.land_offers import LandOffers
from railroads.ui.map_input import MapInput
from railroads.ui.map_overlays import (
    MOODS_LEGEND,
    SELECTED_COLOR,
    TOWNS_LEGEND,
    cell_moods,
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
    draw_starving,
    draw_station_names,
    draw_status,
    mood_tint,
    status_text,
    tint_colours,
    town_cells,
    visible_area,
    with_legend,
)
from railroads.ui.map_tools import MapTools
from railroads.ui.minimap import FRAME_PAD, Minimap, MinimapLayout, view_corners
from railroads.ui.ornate import DEEP, bar_buttons, frame_window, keep_in_front, ornate_window
from railroads.ui.pixel.dots_shader import DotShader, dot_vertices
from railroads.ui.pixel.gl_util import free_textures
from railroads.ui.pixel.ground import glow_strength
from railroads.ui.pixel.ground_palette import GLOW
from railroads.ui.pixel.ground_shader import (
    GroundShader,
    Screen,
    View,
    cell_info,
    screen_of,
    view_of,
)
from railroads.ui.pixel.layer import PixelLayer
from railroads.ui.pixel.relief import Elevation
from railroads.ui.pixel.sheet import ByteSheet, SpriteSheet
from railroads.ui.pixel.signal_layer import SignalLayer
from railroads.ui.pixel.stand_shader import StandShader
from railroads.ui.pixel.track_preview import TrackPreview
from railroads.ui.pixel.train_layer import TrainLayer, engine_picture
from railroads.ui.pixel.world import PixelWorld
from railroads.ui.port_window import PortWindow
from railroads.ui.route_map import soften, soften_steps
from railroads.ui.search import BuildingSearch
from railroads.ui.spotlight import pulsing_brass, spot
from railroads.ui.station_preview import draw_station_preview
from railroads.ui.station_window import StationWindow, draw_station_card
from railroads.ui.summaries_panel import SummariesPanel
from railroads.ui.theme import BRASS, OVERLAY_FLAGS, SIDE_WINDOW_POS, rgba_u32, toggle_button

BACKGROUND_COLOR = (0.08, 0.09, 0.11, 1.0)
# How long the mouse rests on a station, in seconds, before its card shows.
STATION_CARD_DELAY = 1.0
DETAILS_SIZE = (460.0, 420.0)
SHARES_SIZE = (400.0, 400.0)
# Stepping to a building found zooms in at least this far, so it can be seen.
VISIT_ZOOM = ZOOM_CELL_WIDTHS.index(2 * PIXEL_CELL_WIDTH)
# Slopes are lit from the top-left of the screen.
LIGHT_ON_SCREEN = (-1.0, -1.0)
# Population points a cell needs to show as a town on the minimap.
MINIMAP_TOWN_POINTS = 12
# How far above the minimap, or the map's corner without one, the lens buttons sit.
LENS_GAP = 8.0
# How thick the outline is round a cell the tutorial wants clicked.
SPOT_WIDTH = 4.0


class Lens(Enum):
    """What the map can be tinted by: how satisfied buildings are, which town they are in, or
    the chance of a kind of building coming up (railroads.ui.growth_window), or where a good is made
    and wanted (railroads.ui.goods_search)."""

    SATISFACTION = "satisfaction"
    TOWNS = "towns"
    GROWTH = "growth"
    GOODS = "goods"


@cache
def _sprite_sheet() -> SpriteSheet:
    return SpriteSheet.load()


@cache
def _fine_sprite_sheet() -> SpriteSheet:
    return SpriteSheet.load_fine()


@cache
def _finest_sprite_sheet() -> ByteSheet:
    return SpriteSheet.load_finest()


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
        self._spots_kept: BuildingSpots | None = None
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
        self._signals: SignalLayer | None = None
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
        # The shaders standing what stands on the map over that ground, made with it
        # (railroads.ui.pixel.stand_shader).
        self._stand: StandShader | None = None
        self._shader_tried = False
        self._colors_version = 0
        # The shader drawing the building dots zoomed out, once tried.
        self._dots: DotShader | None = None
        self._dots_tried = False
        self.hovered: Cell | None = None
        # The art pixel last looked up under the mouse and the ground point showing there, as the
        # mouse often rests while frames go on.
        self._under: tuple[tuple[float, float], tuple[float, float]] | None = None
        # The cell whose details are open, picked by clicking without dragging.
        self.selected: Cell | None = None
        # The buildings that take what the selected cell makes.
        self._consumers: Kept[tuple[Cell, ...]] = Kept()
        # The cells Shift tints by how satisfied their buildings are, their packed colours, and
        # those where a kind of building gets nothing.
        self._moods: Kept[tuple[npt.NDArray[np.intp], list[int], npt.NDArray[np.intp]]] = Kept()
        # The lens turned on above the minimap, if any, and the town tints.
        self.lens: Lens | None = None
        self._towns: Kept[tuple[npt.NDArray[np.intp], list[int]]] = Kept()
        # The top of the minimap on screen, which the lens buttons sit above.
        self._minimap_top: float | None = None
        # The left and right of the map the lens buttons sit over, if one is shown.
        self._minimap_span: tuple[float, float] | None = None
        self.build = BuildMenu()
        # What came of the last Buy or Sell pressed in the shares window, and the kind of building
        # in the selected cell whose shares it shows, if open.
        self._bought = ""
        self.shares_of: tuple[Cell, BuildingKind] | None = None
        # The station levels the map shows, to redraw stations raised since.
        self._shown_levels: dict[Cell, int] = dict(self.railway.levels)
        # What the build tools do with clicks, and the mouse and keys (railroads.ui.map_tools,
        # railroads.ui.map_input).
        self.tools = MapTools(self)
        self.input = MapInput(self)
        self._map_rect: tuple[Vec2, Vec2] = ((0.0, 0.0), (0.0, 0.0))
        # What a station built in the cell under the mouse would reach, and what each station
        # standing reaches.
        self._station_previewed: Kept[StationPreview] = Kept()
        self._built_previewed: KeptFor[Cell, StationPreview] = KeptFor()
        # The station the mouse rests on, and since when, in imgui's seconds.
        self._resting: tuple[Cell, float] | None = None
        self.summaries = SummariesPanel()
        self.station_window = StationWindow()
        self.port_window = PortWindow()
        self.search = BuildingSearch()
        # The growth window and the kind the growth lens shows (railroads.ui.growth_window).
        self.growth_window = GrowthWindow()
        # The top of the lens buttons on screen, which the growth lens's legend sits above.
        self._lenses_top: float | None = None
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
        self.build.electric_open = self.railway.electric_open
        self.build.quadruple_open = self.railway.quadruple_open
        self.build.bridges_open = self.railway.bridges_open
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
            self._built_station_preview,
            self.tools.start_adding_platform,
            self.tools.start_adding_facility,
            self.buildings_changed,
        )
        self.port_window.draw(self.growth, self.look_at, self.open_shares)
        self.summaries.draw(*self._map_rect, self.look_at, self.railway.name_of)
        self.search.draw(self.growth, self.visit, self._view_center, self._good_picked)
        self.growth_window.draw(self.growth, self.visit, self._growth_picked)
        self._draw_growth_legend()
        self._draw_goods_legend()
        self.land.confirm(self.game.lots, self.game.finances)

    @property
    def _spots(self) -> BuildingSpots:
        """Where every building stands (BuildingSpots), worked out the first time it is wanted,
        which a game loading does on a step of its own (prepare_steps)."""
        if self._spots_kept is None:
            self._spots_kept = BuildingSpots.of(self.constructions)
        return self._spots_kept

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

    def toggle_growth(self) -> None:
        self.growth_window.open = not self.growth_window.open

    def _growth_picked(self) -> None:
        """Turns the growth lens on for the kind just picked in the growth window."""
        self.lens = Lens.GROWTH

    def shows_growth(self) -> bool:
        """Whether the growth or the goods lens is on, which right-clicks pick cells for."""
        return self.lens in (Lens.GROWTH, Lens.GOODS)

    def _good_picked(self) -> None:
        """Turns the goods lens on for the good just picked in the search window."""
        self.lens = Lens.GOODS

    def _goods_lens(self) -> Good | None:
        """The good the goods lens shows, if it is on with one picked."""
        return self.search.goods.good if self.lens == Lens.GOODS else None

    def _draw_goods_legend(self) -> None:
        """The goods lens's scales over the map, above the lens buttons."""
        good = self._goods_lens()
        tints = self.search.goods.tints(self.growth)
        if good is None or tints is None or self._lenses_top is None:
            return
        self._begin_legend()
        draw_goods_legend(good, tints, self.search.goods.gaps)
        imgui.end()

    def _begin_legend(self) -> None:
        """Begins a lens's legend window over the map, above the lens buttons."""
        _, (right, _) = self._map_rect
        imgui.set_next_window_pos(
            (right - LENS_GAP, (self._lenses_top or 0.0) - LENS_GAP),
            imgui.Cond_.always.value,
            (1.0, 1.0),
        )
        imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(*DEEP))
        imgui.begin("##lenslegend", flags=OVERLAY_FLAGS | imgui.WindowFlags_.no_move.value)
        imgui.pop_style_color()
        frame_window()

    def _growth_lens(self) -> Ladder | None:
        """The kind of building the growth lens shows, if it is on with one picked."""
        return self.growth_window.ladder if self.lens == Lens.GROWTH else None

    def _draw_growth_legend(self) -> None:
        """The growth lens's scale over the map, above the lens buttons."""
        ladder = self._growth_lens()
        if ladder is None or self._lenses_top is None:
            return
        self._begin_legend()
        draw_legend(ladder, self.growth_window.odds.tints(self.growth, ladder))
        imgui.end()

    def close_windows(self) -> None:
        """Closes the windows over the map: trains, stations, a station's, a cell's details and
        its shares, a month's summary, the search and the growth window."""
        self.growth_window.open = False
        self.build.trains.open = self.build.trains.buy.open = False
        self.build.stations.open = False
        self.station_window.station = None
        self.port_window.cell = None
        self.selected = None
        self.shares_of = None
        self._bought = ""
        self.summaries.open = None
        if self.search.open:
            self.search.close()

    def open_station(self, station: Cell) -> None:
        """Opens a station's details in place of any cell's."""
        self.station_window.station = station
        self.selected = None

    def open_shares(self, cell: Cell, kind: BuildingKind) -> None:
        """Opens a cell's details with the shares window of a kind of building there."""
        self.selected = cell
        self.shares_of = (cell, kind)
        self._bought = ""

    def look_at(self, cell: Cell) -> None:
        """Centres the view on a cell."""
        x, y = self.tile_map.grid.center(*cell)
        self.camera.center_x, self.camera.center_y = float(x), float(y)

    @property
    def map_rect(self) -> tuple[Vec2, Vec2]:
        """The map's top left and bottom right corners on the screen, as last drawn."""
        return self._map_rect

    def time_passed(self, months: float) -> None:
        """Moves the trains on by this much of a month."""
        self.game.time_passed(months)

    def _draw_details(self) -> None:
        if self.selected is None:
            return
        imgui.set_next_window_pos(SIDE_WINDOW_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(DETAILS_SIZE, imgui.Cond_.first_use_ever.value)
        details = recently_described(self.growth, self.selected, self._satisfaction)
        with ornate_window("Hex details", "details") as (expanded, still_open):
            keep_in_front()
            if expanded:
                pressed = draw_hex_details(details)
                if pressed is not None:
                    self.shares_of = (self.selected, pressed)
                    self._bought = ""
                if (good := self._goods_lens()) is not None:
                    draw_cell_good(self.growth, good, self.selected)
                draw_cell_odds(
                    self.growth, self.growth_window.odds, self.selected, self._growth_lens()
                )
        if not still_open:
            self.selected = None
            self.shares_of = None
            self._bought = ""
            return
        self._draw_shares(details)

    def _draw_shares(self, details: HexDetails) -> None:
        """The shares window of the kind of building in the selected cell whose Shares button
        was pressed, beside the details."""
        cell = self.selected
        if self.shares_of is None or self.shares_of[0] != cell:
            self.shares_of = None
            return
        row = next((r for r in details.buildings if r.kind == self.shares_of[1]), None)
        if cell is None or row is None or row.kind is None:
            self.shares_of = None
            return
        imgui.set_next_window_pos(
            (SIDE_WINDOW_POS[0] + DETAILS_SIZE[0] + 8, SIDE_WINDOW_POS[1]),
            imgui.Cond_.first_use_ever.value,
        )
        imgui.set_next_window_size(SHARES_SIZE, imgui.Cond_.first_use_ever.value)
        with ornate_window("Shares", "shares") as (expanded, still_open):
            keep_in_front()
            if expanded:
                pressed = draw_shares(row, f"{cell[0]}, {cell[1]}")
                if pressed is not None:
                    owners = self.growth.owners
                    kind = row.kind
                    buying, lot = pressed
                    trade = owners.buy if buying else owners.sell
                    said = trade(cell, kind, lot)
                    held = int(owners.shares(kind)[cell])
                    self._bought = said or f"You own {held:,} shares of the {kind.value}s here"
                if self._bought:
                    imgui.text_disabled(self._bought)
        if not still_open:
            self.shares_of = None
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
        for headline in month.headlines:
            self.summaries.add(headline)
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
        self._spots_kept = BuildingSpots.of(self.constructions)
        self.minimap.show_towns(self._town_cells())
        if self.search.open:
            self.search.refresh(self.growth)
        if self._pixels is not None:
            yield
            chunks = yield from self._pixels.world.buildings_changed_steps(changed)
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
        under = None if minimap is None else minimap.overview or minimap.world
        self._minimap_top = None if under is None else under.top
        self._minimap_span = None if under is None else (under.left, under.right)
        self.input.handle_mouse(center, minimap)

        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(top_left, bottom_right, True)
        draw_list.add_rect_filled(top_left, bottom_right, rgba_u32(BACKGROUND_COLOR))
        if self.camera.pixel_scale:
            shaded = self._draw_shaded_ground(draw_list, center, top_left, bottom_right)
            if not shaded:
                self._draw_ground(draw_list, center, self._ground_texture(False))
            layer = self.pixel_layer()
            layer.over_ground = shaded
            layer.stand = self._stand.stand if shaded and self._stand is not None else None
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
            signals = self._signal_layer()
            signals.draw(draw_list, self.railway, self.camera, center, near=False)
            if self.railway.trains:
                self._train_layer().draw(draw_list, self.railway, self.camera, center)
            signals.draw(draw_list, self.railway, self.camera, center, near=True)
        else:
            draw_track(draw_list, self.constructions, self.camera, center)
            if self.railway.trains:
                self._train_layer().draw(draw_list, self.railway, self.camera, center)
        marks = self._marks_camera()
        tints = (
            self._mood_tints()
            if self._showing_moods()
            else self._town_tints()
            if self.lens == Lens.TOWNS
            else self._goods_tints()
            if self.lens == Lens.GOODS
            else self._growth_tints()
        )
        if tints is not None and not self._draw_shaded_tints(
            draw_list, center, top_left, bottom_right, *tints
        ):
            draw_moods(
                draw_list, marks, self.tile_map.grid, center, (top_left, bottom_right), *tints
            )
        if self._showing_moods():
            draw_starving(
                draw_list,
                marks,
                self.tile_map.grid,
                center,
                (top_left, bottom_right),
                self._starving_cells(),
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
        if self.build.preview is not None:
            if self.camera.pixel_scale:
                self._track_preview().draw(draw_list, self.build.preview, self.camera, center)
            else:
                draw_track(
                    draw_list, self.constructions, self.camera, center, self.build.preview.pieces
                )
            if self.hovered is not None and self.build.placing_track:
                draw_outline(draw_list, marks, self.tile_map.grid, center, self.hovered)
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
        if not self._draw_shaded_dots(draw_list, center, top_left, bottom_right):
            clip = (top_left, bottom_right)
            draw_building_dots(draw_list, self.camera, center, clip, self._spots)

    def _draw_shaded_dots(
        self, draw_list: imgui.ImDrawList, center: Vec2, top_left: Vec2, bottom_right: Vec2
    ) -> bool:
        """Draws the building dots with the graphics card, if it can here, into a texture laid
        on whole device pixels; returns whether it did."""
        shader = self._dot_shader()
        if shader is None:
            return False
        spots = self._spots
        if shader.dots_version is not spots:
            shader.set_dots(dot_vertices(spots.x, spots.y, *dot_looks(spots)), spots)
        density = imgui.get_io().display_framebuffer_scale.x
        left, top = math.floor(top_left[0] * density), math.floor(top_left[1] * density)
        right = math.ceil(bottom_right[0] * density)
        bottom = math.ceil(bottom_right[1] * density)
        corner = (left / density, top / density)
        e, f = self.camera.to_screen(0.0, 0.0, center)
        a, c = self.camera.to_screen(1.0, 0.0, center)
        b, d = self.camera.to_screen(0.0, 1.0, center)
        to_screen = (a - e, b - e, c - f, d - f, e - corner[0], f - corner[1])
        unit = self.camera.cell_width / self.camera.cell_units
        texture = shader.draw((right - left, bottom - top), density, to_screen, unit)
        draw_list.add_image(
            imgui.ImTextureRef(texture), corner, (right / density, bottom / density)
        )
        return True

    def _consumers_of(self, cell: Cell) -> tuple[Cell, ...]:
        """consumers_in_reach of the cell, found again only once its buildings, the year or the
        boards changed: the hex details are drawn every frame."""
        key = (cell, self.constructions.version, self.growth.year, boards_changed())
        return self._consumers.get(
            key, lambda: consumers_in_reach(self.constructions, cell, self.growth.year)
        )

    def _draw_lenses(self) -> None:
        """Buttons over the bottom-right corner of the map, above the minimap, that turn a lens
        on or off: one at a time."""
        _, (right, bottom) = self._map_rect
        top = self._minimap_top if self._minimap_top is not None else bottom
        flags = OVERLAY_FLAGS | imgui.WindowFlags_.no_move.value
        if self._minimap_span is None:
            imgui.set_next_window_pos(
                (right - LENS_GAP, top - LENS_GAP), imgui.Cond_.always.value, (1.0, 1.0)
            )
        else:
            left, edge = self._minimap_span
            imgui.set_next_window_pos(
                (edge + FRAME_PAD, top - FRAME_PAD - LENS_GAP),
                imgui.Cond_.always.value,
                (1.0, 1.0),
            )
            height = imgui.get_frame_height() + 2 * imgui.get_style().window_padding.y
            imgui.set_next_window_size((edge - left + 2 * FRAME_PAD, height))
            flags &= ~imgui.WindowFlags_.always_auto_resize.value
        imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(*DEEP))
        imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(0, 0, 0, 0))
        imgui.begin("##lenses", flags=flags)
        imgui.pop_style_color(2)
        self._lenses_top = imgui.get_window_pos().y
        frame_window()
        with bar_buttons():
            self._draw_lens_buttons(self._minimap_span is not None)
        imgui.end()

    def _draw_lens_buttons(self, fill: bool) -> None:
        """The lens buttons side by side, sharing the window's width between them if `fill`, and
        showing only their icons where their names would not fit."""
        spacing = imgui.get_style().item_spacing.x
        count = len(Lens)
        width = (
            (imgui.get_content_region_avail().x - spacing * (count - 1)) / count if fill else 0.0
        )
        buttons = (
            (
                Lens.SATISFACTION,
                fa.ICON_FA_SMILE,
                "Satisfaction",
                "Tint the land by how satisfied its buildings are (or hold Shift)",
            ),
            (
                Lens.TOWNS,
                fa.ICON_FA_USERS,
                "Towns",
                "Tint the land by town; passengers and mail only travel between towns",
            ),
            (
                Lens.GOODS,
                fa.ICON_FA_CUBES,
                "Goods",
                "Tint the land by where a good is made and wanted; pick it in the search window",
            ),
            (
                Lens.GROWTH,
                fa.ICON_FA_LEAF,
                "Growth",
                "Tint the land by the chance of a kind of building coming up in a year",
            ),
        )
        padding = 2 * imgui.get_style().frame_padding.x
        named = all(
            imgui.calc_text_size(f"{icon}  {name}").x + padding <= width or not fill
            for _, icon, name, _ in buttons
        )
        for lens, icon, name, tooltip in buttons:
            label = f"{icon}  {name}" if named else icon
            if not named:
                tooltip = f"{name}: {tooltip}"
            if lens != Lens.SATISFACTION:
                imgui.same_line()
            if toggle_button(label, self.lens == lens, tooltip, (width, 0)):
                self.lens = None if self.lens == lens else lens
                if self.lens == Lens.GROWTH and self.growth_window.ladder is None:
                    self.growth_window.open = True
                if self.lens == Lens.GOODS and self.search.goods.good is None:
                    if not self.search.open:
                        self.toggle_search()
                    self.search.goods_to_front = True
            spot(f"lens:{lens.value}")

    def _showing_moods(self) -> bool:
        """Whether the satisfaction lens is on, or Shift is held for it, not while typing."""
        io = imgui.get_io()
        return self.lens == Lens.SATISFACTION or (io.key_shift and not io.want_text_input)

    def _goods_tints(self) -> tuple[npt.NDArray[np.intp], list[int]] | None:
        """The goods lens's cells and tints, if it is on with a good picked."""
        if self._goods_lens() is None:
            return None
        tints = self.search.goods.tints(self.growth)
        return None if tints is None else (tints.cells, tints.colors)

    def _growth_tints(self) -> tuple[npt.NDArray[np.intp], list[int]] | None:
        """The growth lens's cells and tints, if it is on with a kind picked."""
        ladder = self._growth_lens()
        if ladder is None:
            return None
        tints = self.growth_window.odds.tints(self.growth, ladder)
        return tints.cells, tints.colors

    def _town_tints(self) -> tuple[npt.NDArray[np.intp], list[int]]:
        """town_cells, worked out again only once the buildings changed."""
        return self._towns.get(self.constructions.version, lambda: town_cells(self.growth.towns))

    def _mood_tints(self) -> tuple[npt.NDArray[np.intp], list[int]]:
        """cell_moods's cells and their tints, worked out again only once the month's trade
        or the buildings changed."""
        return self._cell_moods()[:2]

    def _starving_cells(self) -> npt.NDArray[np.intp]:
        """The cells cell_moods finds a kind of building getting nothing in."""
        return self._cell_moods()[2]

    def _cell_moods(self) -> tuple[npt.NDArray[np.intp], list[int], npt.NDArray[np.intp]]:
        def moods() -> tuple[npt.NDArray[np.intp], list[int], npt.NDArray[np.intp]]:
            cells, shares, starving = cell_moods(self.growth.last_trade, self.growth.assumed_shares)
            return cells, [rgba_u32(mood_tint(share)) for share in shares.tolist()], cells[starving]

        return self._moods.get((self.growth.last_trade, self.constructions.version), moods)

    def _marks_camera(self) -> IsoCamera | OnGround:
        """The camera marks on the map are drawn with: lifted onto raised ground in pixel art."""
        if not self.camera.pixel_scale:
            return self.camera
        return OnGround(self.camera, self.pixel_layer().world.elevation)

    def ground_under(self, screen_x: float, screen_y: float, center: Vec2) -> tuple[float, float]:
        """The map point showing at a spot on screen, on raised ground where it rises."""
        x, y = self.camera.to_plane(screen_x, screen_y, center)
        if not self.camera.pixel_scale:
            return x, y
        world = self.pixel_layer().world
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
            text = with_legend(text, MOODS_LEGEND)
        elif self.lens == Lens.TOWNS:
            text = with_legend(text, TOWNS_LEGEND)
        elif (good := self._goods_lens()) is not None:
            text = with_legend(
                text, f"Goods: where {good.value} is made (amber) and wanted (blue)."
            )
        elif (ladder := self._growth_lens()) is not None:
            text = with_legend(text, f"Growth: the chance of a {ladder.name} a year.")
        below = draw_status(draw_list, top_left, bottom_right, text)
        if self.hovered is None or over_minimap:
            return
        if self.build.tool in (Tool.STATION, Tool.MOVE_STATION):
            draw_station_preview(self._station_preview(self.hovered), below)
            return
        station = self.rested_station(imgui.get_time())
        if station is not None:
            draw_station_card(self.railway, station, below)
        else:
            draw_hovered_buildings(self.growth, self.hovered, below, self._satisfaction)

    def rested_station(self, now: float) -> Cell | None:
        """The station under the mouse once it has rested there STATION_CARD_DELAY seconds."""
        station = None if self.hovered is None else self.constructions.main_station(self.hovered)
        if station is None:
            self._resting = None
            return None
        if self._resting is None or self._resting[0] != station:
            self._resting = (station, now)
        return station if now - self._resting[1] >= STATION_CARD_DELAY else None

    def _built_station_preview(self, station: Cell) -> StationPreview:
        """What a station standing reaches, with its own reach and platforms, worked out again
        once the buildings, its reach or its platforms change or a month on."""
        delivery, collection = self.railway.reach(station)
        platforms = tuple(self.constructions.platforms(station))
        key = (self.constructions.version, delivery, collection, platforms, self.growth.last_trade)
        return self._built_previewed.get(
            station,
            key,
            lambda: self.growth.station_trade.preview(station, delivery, collection, platforms),
        )

    def _station_preview(self, cell: Cell) -> StationPreview:
        """What a station built in a cell would reach, worked out again only for another cell,
        once the buildings change or a month on."""
        key = (cell, self.constructions.version, self.growth.last_trade)
        return self._station_previewed.get(key, lambda: self.growth.station_trade.preview(cell))

    def _signal_layer(self) -> SignalLayer:
        if self._signals is None:
            self._signals = SignalLayer(self.pixel_layer().world)
        return self._signals

    def _train_layer(self) -> TrainLayer:
        if self._trains is None:
            self._trains = TrainLayer(self.pixel_layer().world, self.railway.trains)
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
            self._preview = TrackPreview(self.pixel_layer().world)
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
        if self._spots_kept is None:
            self._spots_kept = BuildingSpots.of(self.constructions)
        yield
        plain = self._plain_picture()
        yield
        self._soft = (self._ground_version, (yield from soften_steps(plain)))
        light = self.camera.map_direction(*LIGHT_ON_SCREEN)
        self._ground_pictures().picture(self._ground_colors(), light, True)
        yield
        if self._pixels is None:
            yield from self._pixel_layer_steps()
        layer = self.pixel_layer()
        if self.railway.trains:
            yield
            self._train_layer()
        if chunks and self.camera.pixel_scale:
            yield
            center = (size[0] / 2, size[1] / 2)
            standing = hello_imgui.is_using_hello_imgui()
            layer.prepare(self.camera, center, ((0.0, 0.0), size), density, standing)

    def pixel_layer(self) -> PixelLayer:
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
            finest=_finest_sprite_sheet(),
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
        self._spots_kept = BuildingSpots.of(self.constructions)
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
        world = self.pixel_layer().world
        if shader.cells_version != self._colors_version:
            ground = world.ground
            info = cell_info(
                ground.biomes,
                self.tile_map.deposits,
                ground.hills,
                glow_strength(ground.satisfaction),
                ground.mountains,
                ground.streams,
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
        camera_pixel = self.pixel_layer().world.projection.to_pixels(
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

    def _dot_shader(self) -> DotShader | None:
        """The building dots' shader, made the first time the map shows zoomed out in a real
        window."""
        if not self._dots_tried:
            self._dots_tried = True
            if hello_imgui.is_using_hello_imgui():
                self._dots = DotShader.make()
        return self._dots

    def ready_shaders(self) -> None:
        """Makes the shaders and the world's texture the first frame would, where the game is
        loaded on the frames of a real window: compiling them takes a frame of its own."""
        if hello_imgui.is_using_hello_imgui():
            self._ground_shader()
            self._dot_shader()
            self.minimap.world_texture()

    def _ground_shader(self) -> GroundShader | None:
        """The ground shader, made the first time the pixel map shows in a real window."""
        if not self._shader_tried:
            self._shader_tried = True
            if hello_imgui.is_using_hello_imgui():
                world = self.pixel_layer().world
                self._shader = GroundShader.make(
                    world.projection,
                    self.tile_map.grid.shape,
                    world.ground.shading,
                    world.elevation.heights,
                )
                self._stand = StandShader.make(self._shader, SpriteSheet.load_bytes)
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
