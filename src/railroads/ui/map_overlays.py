"""What the map view draws over the map: cell outlines and grid lines, station names, the compass,
the buildings a search found, the status strip saying what is under the pointer, the card of
the hovered cell's buildings and, while Shift is held, how satisfied buildings are."""

import itertools
import math
from collections import Counter

import numpy as np
import numpy.typing as npt
from imgui_bundle import imgui

from railroads.sim.constructions import Building, Construction, Constructions, Station, Track
from railroads.sim.economy import LOCAL_REACH, FloatCells, Satisfaction, Trade
from railroads.sim.goods import HANDED_OUT, makes_of, needs_of
from railroads.sim.grids import HEX_RADIUS, HEX_ROW_STEP, Cell, Grid
from railroads.sim.growth import Growth
from railroads.sim.ownership import Owners
from railroads.sim.railway import Railway
from railroads.sim.tilemap import Relief, TileMap
from railroads.sim.towns import Towns
from railroads.sim.track_rules import TrackBed
from railroads.ui.biomes import biome_of
from railroads.ui.build_menu import reach_border
from railroads.ui.hex_details import mood_text, recently_described
from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS, IsoCamera, Vec2
from railroads.ui.minimap import MinimapLayout
from railroads.ui.search import BuildingSearch, pulse
from railroads.ui.theme import (
    MOODS,
    OVERLAY,
    OVERLAY_FLAGS,
    TEXT,
    Rgba,
    color_u32,
    good_icon,
    rgba_u32,
    town_tint,
)

GRID_COLOR = (0.0, 0.0, 0.0, 0.25)
HOVER_COLOR = (1.0, 1.0, 1.0, 0.9)
SELECTED_COLOR = (1.0, 0.85, 0.2, 1.0)
COMPASS_COLOR = (1.0, 1.0, 1.0, 0.85)
FOUND_COLOR = (0.3, 0.95, 1.0, 1.0)
# The selected cell's hand-out range and the buildings in it that take what it makes.
HAND_OUT_RING = (1.0, 0.85, 0.3, 0.95)
CONSUMER_FILL = (0.25, 0.85, 0.3, 0.35)
CONSUMER_LINE = (0.3, 0.95, 0.35, 0.95)
# What Shift lays over the land under buildings by how satisfied they are: thriving, at least
# half way there, short of half and getting nothing at all.
SATISFIED_TINT = (0.2, 0.85, 0.25, 0.5)
HALF_SATISFIED_TINT = (0.95, 0.88, 0.15, 0.5)
LITTLE_SATISFIED_TINT = (1.0, 0.5, 0.1, 0.5)
UNSATISFIED_TINT = (0.9, 0.12, 0.1, 0.5)
MOODS_LEGEND = "Satisfaction: green 100%, yellow 50%+, orange under 50%, red nothing."
TOWNS_LEGEND = "Towns: each colour is one town; passengers and mail only travel between towns."
# From this cell width the tint fills each cell's hexagon; further out it is a square.
TINT_HEXES_FROM = 8
# From this cell width, buildings found are outlined; further out each is a small square.
OUTLINE_FOUND_FROM = 16
MOST_OUTLINES = 1500
MOST_FOUND_MARKS = 6000
MOST_MINIMAP_MARKS = 3000
# Station names show from this zoom level in, this many cell widths above the station's middle.
NAMES_FROM_ZOOM = 2
# The flag over the player's buildings: from how wide a cell it shows, how tall it stands and how
# far right of the cell's middle, both in cell widths, and its colours.
FLAGS_FROM_WIDTH = 24.0
FLAG_HEIGHT = 0.45
FLAG_RIGHT = 0.3
FLAG_POLE = (0.15, 0.12, 0.1, 1.0)
FLAG_CLOTH = (0.14, 0.3, 0.72, 1.0)
FLAG_STRIPE = (0.93, 0.78, 0.3, 1.0)
NAME_LIFT = 0.55
NAME_PADDING = 6.0
STATUS_MARGIN = 10.0
STATUS_PADDING = 10.0
# How big the goods' icons are in the card of the hovered cell's buildings.
HOVER_ICON = 16.0
# Lower left, upper left, top and upper right corner: three edges per hexagon draw the whole grid.
_HEX_EDGE_ANGLES = np.radians([150, 210, 270, 330])


def draw_outline(
    draw_list: imgui.ImDrawList,
    camera: IsoCamera,
    grid: Grid,
    center: Vec2,
    cell: Cell,
    rgba: Rgba = HOVER_COLOR,
    thickness: float = 2.0,
) -> None:
    points = [imgui.ImVec2(*camera.to_screen(x, y, center)) for x, y in grid.corners(*cell)]
    draw_list.add_polyline(points, rgba_u32(rgba), thickness, imgui.ImDrawFlags_.closed.value)


def visible_area(
    camera: IsoCamera, center: Vec2, top_left: Vec2, bottom_right: Vec2
) -> tuple[float, ...]:
    """The part of the map plane on screen: its least and greatest x and y."""
    corners = [
        camera.to_plane(x, y, center)
        for x in (top_left[0], bottom_right[0])
        for y in (top_left[1], bottom_right[1])
    ]
    xs, ys = [corner[0] for corner in corners], [corner[1] for corner in corners]
    return min(xs), min(ys), max(xs), max(ys)


def draw_grid_lines(
    draw_list: imgui.ImDrawList, camera: IsoCamera, grid: Grid, center: Vec2, visible: tuple
) -> None:
    if grid.kind == "square":
        _draw_square_lines(draw_list, camera, grid, center, visible)
    else:
        _draw_hex_lines(draw_list, camera, grid, center, visible)


def _draw_square_lines(
    draw_list: imgui.ImDrawList, camera: IsoCamera, grid: Grid, center: Vec2, visible: tuple
) -> None:
    x0, y0, x1, y1 = visible
    c0, c1 = max(0, math.floor(x0)), min(grid.cols, math.ceil(x1))
    r0, r1 = max(0, math.floor(y0)), min(grid.rows, math.ceil(y1))
    if c0 >= c1 or r0 >= r1:
        return
    line = rgba_u32(GRID_COLOR)
    to_screen = camera.to_screen
    for x in range(c0, c1 + 1):
        draw_list.add_line(to_screen(x, r0, center), to_screen(x, r1, center), line)
    for y in range(r0, r1 + 1):
        draw_list.add_line(to_screen(c0, y, center), to_screen(c1, y, center), line)


def _draw_hex_lines(
    draw_list: imgui.ImDrawList, camera: IsoCamera, grid: Grid, center: Vec2, visible: tuple
) -> None:
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
    sx, sy = camera.to_screen(px, py, center)
    line = rgba_u32(GRID_COLOR)
    for points in np.stack([sx, sy], axis=2).tolist():
        draw_list.add_polyline([imgui.ImVec2(x, y) for x, y in points], line, 1.0, 0)


def draw_station_names(
    draw_list: imgui.ImDrawList, camera: IsoCamera, grid: Grid, railway: Railway, center: Vec2
) -> None:
    """Each station's name on a dark tag above it, once zoomed in to NAMES_FROM_ZOOM."""
    if camera.zoom_level < NAMES_FROM_ZOOM:
        return
    width = camera.cell_width
    for station in railway.stations():
        name = railway.name_of(station)
        x, y = camera.to_screen(*grid.center(*station), center)
        size = imgui.calc_text_size(name)
        left = x - size.x / 2 - NAME_PADDING
        bottom = y - width * NAME_LIFT
        top = bottom - size.y - NAME_PADDING
        draw_list.add_rect_filled(
            (left, top), (x + size.x / 2 + NAME_PADDING, bottom), color_u32(OVERLAY), 5.0
        )
        draw_list.add_text((x - size.x / 2, top + NAME_PADDING / 2), color_u32(TEXT), name)


def draw_owned_flags(
    draw_list: imgui.ImDrawList, camera: IsoCamera, grid: Grid, owners: Owners, center: Vec2
) -> None:
    """A small flag in the company's colours over every cell where the player owns a building,
    once a cell is FLAGS_FROM_WIDTH pixels wide."""
    width = camera.cell_width
    if width < FLAGS_FROM_WIDTH:
        return
    pole = rgba_u32(FLAG_POLE)
    cloth = rgba_u32(FLAG_CLOTH)
    stripe = rgba_u32(FLAG_STRIPE)
    height = width * FLAG_HEIGHT
    for cell in owners.cells():
        x, y = camera.to_screen(*grid.center(*cell), center)
        foot = (x + width * FLAG_RIGHT, y)
        top = (foot[0], y - height)
        draw_list.add_line(foot, top, pole, max(1.0, width / 40))
        far = (top[0] + height * 0.6, top[1] + height * 0.35)
        draw_list.add_rect_filled(top, far, cloth)
        middle = top[1] + height * 0.35 / 2
        draw_list.add_line((top[0], middle), (far[0], middle), stripe, max(1.0, height * 0.08))


def draw_compass(draw_list: imgui.ImDrawList, camera: IsoCamera, anchor: Vec2) -> None:
    here = camera.to_screen(camera.center_x, camera.center_y, (0, 0))
    north = camera.to_screen(camera.center_x, camera.center_y - 1, (0, 0))
    dx, dy = north[0] - here[0], north[1] - here[1]
    length = math.hypot(dx, dy)
    dx, dy = dx / length, dy / length
    line = rgba_u32(COMPASS_COLOR)
    tip = (anchor[0] + dx * 14, anchor[1] + dy * 14)
    draw_list.add_line((anchor[0] - dx * 14, anchor[1] - dy * 14), tip, line, 2.0)
    draw_list.add_triangle_filled(
        tip,
        (tip[0] - dx * 7 - dy * 5, tip[1] - dy * 7 + dx * 5),
        (tip[0] - dx * 7 + dy * 5, tip[1] - dy * 7 - dx * 5),
        line,
    )
    label = imgui.calc_text_size("N")
    position = (anchor[0] + dx * 25 - label.x / 2, anchor[1] + dy * 25 - label.y / 2)
    draw_list.add_text(position, line, "N")


def draw_found(
    draw_list: imgui.ImDrawList,
    camera: IsoCamera,
    grid: Grid,
    search: BuildingSearch,
    center: Vec2,
    top_left: Vec2,
    bottom_right: Vec2,
) -> None:
    """Marks every building the search found that is on screen, and the one being visited."""
    if search.kind is None or not search.cells:
        return
    width = camera.cell_width
    sx, sy = camera.to_screen(*grid.center(search.rows, search.cols), center)
    on_screen = np.flatnonzero(
        (sx > top_left[0] - width)
        & (sx < bottom_right[0] + width)
        & (sy > top_left[1] - width)
        & (sy < bottom_right[1] + width)
    )
    if width >= OUTLINE_FOUND_FROM:
        for index in on_screen[:: max(1, on_screen.size // MOST_OUTLINES)].tolist():
            draw_outline(draw_list, camera, grid, center, search.cells[index], FOUND_COLOR)
    else:
        mark = rgba_u32(FOUND_COLOR)
        shown = on_screen[:: max(1, on_screen.size // MOST_FOUND_MARKS)]
        for x, y in zip(sx[shown].tolist(), sy[shown].tolist(), strict=True):
            draw_list.add_rect_filled((x - 2, y - 2), (x + 2, y + 2), mark)
    if search.current is not None:
        _draw_visited(draw_list, camera, grid, center, search.current)


def _draw_visited(
    draw_list: imgui.ImDrawList, camera: IsoCamera, grid: Grid, center: Vec2, cell: Cell
) -> None:
    """A bright outline round the building being visited and an arrow bobbing over it."""
    draw_outline(draw_list, camera, grid, center, cell, HOVER_COLOR)
    x, y = camera.to_screen(*grid.center(*cell), center)
    tip = float(y) - camera.cell_width * 0.55 - 10 * pulse(imgui.get_time())
    draw_list.add_triangle_filled(
        (float(x), tip),
        (float(x) - 10, tip - 16),
        (float(x) + 10, tip - 16),
        rgba_u32(FOUND_COLOR),
    )


def draw_found_on_overview(
    draw_list: imgui.ImDrawList, grid: Grid, search: BuildingSearch, minimap: MinimapLayout
) -> None:
    panel = minimap.overview
    if search.kind is None or panel is None or not search.cells:
        return
    step = max(1, len(search.cells) // MOST_MINIMAP_MARKS)
    xs, ys = grid.center(search.rows[::step], search.cols[::step])
    mark = rgba_u32(FOUND_COLOR)
    sx, sy = minimap.overview_point(xs, ys)
    for x, y in zip(sx.tolist(), sy.tolist(), strict=True):
        if panel.contains(x, y):
            draw_list.add_rect_filled((x - 1.5, y - 1.5), (x + 1.5, y + 1.5), mark)


def status_text(
    camera: IsoCamera,
    tile_map: TileMap,
    constructions: Constructions,
    satisfaction: np.ndarray,
    hovered: Cell | None,
    over_minimap: bool,
) -> str:
    """What the status strip says: the zoom, and what is under the pointer."""
    zoom = f"Zoom {camera.zoom_level + 1}/{len(ZOOM_CELL_WIDTHS)}"
    if over_minimap:
        return f"{zoom}. Click or drag on the map overview to look there, scroll to zoom it."
    if hovered is None:
        return f"{zoom}. Drag to pan, scroll or -/+ to zoom."
    tile = tile_map.tile(hovered)
    row, col = hovered
    hills = " hills" if tile.relief == Relief.HILLS else ""
    built = _built_note(constructions, hovered)
    level = int(satisfaction[row, col])
    mood = f"; {Satisfaction(level).name.lower()}" if level >= 0 else ""
    return f"{zoom}. {biome_of(tile).name.title()}{hills}{built}{mood}"


def _built_note(constructions: Constructions, cell: Cell) -> str:
    built = constructions.at(cell)
    if not built:
        return ""
    counts = Counter(_construction_name(construction) for construction in built)
    listed = ", ".join(name if n == 1 else f"{name} x{n}" for name, n in counts.items())
    left = constructions.space_left(cell)
    return f"; {listed}, " + (f"room for {left} more" if left else "full")


def _construction_name(construction: Construction) -> str:
    match construction:
        case Building():
            return construction.kind.value
        case Station():
            return "station"
        case Track():
            bed = "" if construction.bed == TrackBed.GROUND else f" {construction.bed.name.lower()}"
            kind = (
                " maglev" if construction.maglev else " electric" if construction.electric else ""
            )
            return f"{construction.width.name.lower()}{kind} track{bed}"


def draw_status(draw_list: imgui.ImDrawList, top_left: Vec2, bottom_right: Vec2, text: str) -> Vec2:
    """The status text in a strip along the top of the map; returns where the space below it
    starts."""
    size = imgui.calc_text_size(text)
    x, y = top_left[0] + STATUS_MARGIN, top_left[1] + STATUS_MARGIN
    right = min(x + size.x + 2 * STATUS_PADDING, bottom_right[0] - 80)
    draw_list.add_rect_filled(
        (x, y), (right, y + size.y + 2 * STATUS_PADDING * 0.6), color_u32(OVERLAY), 6.0
    )
    draw_list.push_clip_rect((x, y), (right - STATUS_PADDING / 2, bottom_right[1]), True)
    draw_list.add_text((x + STATUS_PADDING, y + STATUS_PADDING * 0.6), color_u32(TEXT), text)
    draw_list.pop_clip_rect()
    return x, y + size.y + 2 * STATUS_PADDING * 0.6 + STATUS_MARGIN / 2


def draw_hovered_buildings(
    growth: Growth, cell: Cell, top_left: Vec2, satisfaction: np.ndarray | None = None
) -> None:
    """The buildings in the cell under the mouse, one to a line, in a small card below the
    status strip: how satisfied each is, and icons of what it makes, what it demands and its
    bonus goods. `satisfaction` is every cell's, if known (railroads.ui.hex_details.describe)."""
    rows = recently_described(growth, cell, satisfaction).buildings
    if not rows:
        return
    imgui.set_next_window_pos(top_left)
    imgui.set_next_window_bg_alpha(OVERLAY[3])
    flags = OVERLAY_FLAGS | imgui.WindowFlags_.no_inputs.value
    if imgui.begin("##hovered buildings", None, flags):
        for row in rows:
            imgui.text(row.name if row.count == 1 else f"{row.name} x{row.count}")
            if row.satisfaction in MOODS:
                imgui.same_line(0, 8)
                imgui.text_colored(imgui.ImVec4(*MOODS[row.satisfaction]), mood_text(row))
            for goods, label in ((row.makes, "makes"), (row.needs, "needs"), (row.bonus, "bonus")):
                if not goods:
                    continue
                imgui.same_line(0, 10)
                imgui.text_disabled(label)
                for good, _ in goods:
                    imgui.same_line(0, 2)
                    good_icon(good, size=HOVER_ICON)
    imgui.end()


def consumers_in_reach(constructions: Constructions, cell: Cell, year: int) -> tuple[Cell, ...]:
    """The cells within LOCAL_REACH of `cell` that have a building needing a good the buildings
    in `cell` make: where those goods can go on their own (railroads.sim.economy)."""
    made = {
        good
        for building in constructions.buildings_at(cell)
        for good in makes_of(building.kind, year)
    }
    needs = {need for need, goods in HANDED_OUT.items() if made.intersection(goods)}
    if not needs:
        return ()
    return tuple(
        near
        for near in constructions.tile_map.grid.within(cell, LOCAL_REACH)
        if any(
            needs.intersection(needs_of(building.kind, year))
            for building in constructions.buildings_at(near)
        )
    )


def draw_hand_out(
    draw_list: imgui.ImDrawList,
    camera: IsoCamera,
    grid: Grid,
    center: Vec2,
    cell: Cell,
    consumers: tuple[Cell, ...],
) -> None:
    """The border of how far the goods made in `cell` go on their own, and the cells in it whose
    buildings take any of them in green."""
    hexes = [grid.corners(*near) for near in consumers]
    border = reach_border(grid, (cell,), LOCAL_REACH)
    plane = [corner for corners in hexes for corner in corners]
    plane += [end for edge in border for end in edge]
    if not plane:
        return
    xs, ys = (np.array(values, dtype=np.float64) for values in zip(*plane, strict=True))
    sx, sy = camera.to_screen(xs, ys, center)
    points = [imgui.ImVec2(x, y) for x, y in zip(sx.tolist(), sy.tolist(), strict=True)]
    start = 0
    for corners in hexes:
        polygon = points[start : start + len(corners)]
        start += len(corners)
        draw_list.add_convex_poly_filled(polygon, rgba_u32(CONSUMER_FILL))
        draw_list.add_polyline(
            polygon, rgba_u32(CONSUMER_LINE), 2.0, imgui.ImDrawFlags_.closed.value
        )
    for a, b in itertools.batched(points[start:], 2, strict=True):
        draw_list.add_line(a, b, rgba_u32(HAND_OUT_RING), 2.5)


def least_satisfied(
    trade: Trade | None, assumed: FloatCells | None = None
) -> tuple[npt.NDArray[np.intp], FloatCells]:
    """The flat cells whose buildings want anything, and how satisfied the least satisfied kind
    of building in each is (Trade.share); `assumed` as Growth.assumed_shares, where it is given."""
    if trade is None:
        return np.zeros(0, dtype=np.intp), np.zeros(0)
    least = np.full(trade.cells.size, np.nan)
    for kind in trade.got:
        least = np.fmin(least, trade.share((kind,), empty=np.nan))
    wanting = ~np.isnan(least)
    cells, shares = trade.cells[wanting], least[wanting]
    if assumed is not None:
        given = assumed[cells]
        shares = np.where(np.isnan(given), shares, given)
    return cells, shares


def town_cells(towns: Towns) -> tuple[npt.NDArray[np.intp], list[int]]:
    """The flat cells that belong to a town, and each one's packed town colour."""
    cells = np.flatnonzero((towns.town.ravel() > 0) & (towns.sending.ravel() > 0))
    colors = [rgba_u32(town_tint(town)) for town in towns.town.ravel()[cells].tolist()]
    return cells, colors


def mood_tint(share: float) -> Rgba:
    """The tint of a cell whose least satisfied building is at `share`, as levels_of judges it."""
    if share >= 1.0 - 1e-9:
        return SATISFIED_TINT
    if share >= 0.5:
        return HALF_SATISFIED_TINT
    if share > 0.0:
        return LITTLE_SATISFIED_TINT
    return UNSATISFIED_TINT


def tint_colours(
    shape: tuple[int, int], cells: npt.NDArray[np.intp], colors: list[int]
) -> npt.NDArray[np.uint8]:
    """Each cell's tint as (rows, cols, 4) RGBA bytes from the flat `cells` and their packed
    colours, clear where a cell has none, for the graphics card to tint them
    (railroads.ui.pixel.ground_shader.GroundShader.set_tints)."""
    tints = np.zeros((shape[0] * shape[1], 4), dtype=np.uint8)
    tints[cells] = np.array(colors, dtype=np.uint32).view(np.uint8).reshape(-1, 4)
    return tints.reshape(*shape, 4)


def draw_moods(
    draw_list: imgui.ImDrawList,
    camera: IsoCamera,
    grid: Grid,
    center: Vec2,
    clip: tuple[Vec2, Vec2],
    cells: npt.NDArray[np.intp],
    colors: list[int],
) -> None:
    """Fills each of the flat `cells` on screen in its packed colour: its hexagon, or a square
    once cells are smaller than TINT_HEXES_FROM. The corners of every cell drawn are put on screen
    in one go, as in draw_hand_out."""
    if not cells.size:
        return
    width = camera.cell_width
    cx, cy = grid.center(*np.divmod(cells, grid.cols))
    sx, sy = camera.to_screen(cx, cy, center)
    (left, top), (right, bottom) = clip
    shown = np.flatnonzero(
        (sx > left - width) & (sx < right + width) & (sy > top - width) & (sy < bottom + width)
    )
    if width < TINT_HEXES_FROM:
        half = max(1.0, width / 2)
        for i, x, y in zip(shown.tolist(), sx[shown].tolist(), sy[shown].tolist(), strict=True):
            draw_list.add_rect_filled((x - half, y - half), (x + half, y + half), colors[i])
        return
    offsets = np.array(grid.corners(0, 0)) - np.array(grid.center(0, 0))
    px = np.asarray(cx)[shown, None] + offsets[:, 0]
    py = np.asarray(cy)[shown, None] + offsets[:, 1]
    hx, hy = camera.to_screen(px.ravel(), py.ravel(), center)
    corners = np.stack([hx, hy], axis=1).reshape(shown.size, len(offsets), 2).tolist()
    for i, polygon in zip(shown.tolist(), corners, strict=True):
        draw_list.add_convex_poly_filled([imgui.ImVec2(x, y) for x, y in polygon], colors[i])
