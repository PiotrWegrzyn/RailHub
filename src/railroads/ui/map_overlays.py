"""What the map view draws over the map: cell outlines and grid lines, station names, the compass,
the buildings a search found, the status strip saying what is under the pointer and the card of
the hovered cell's buildings."""

import math
from collections import Counter

import numpy as np
from imgui_bundle import imgui

from railroads.sim.constructions import Building, Construction, Constructions, Station, Track
from railroads.sim.economy import Satisfaction
from railroads.sim.grids import HEX_RADIUS, HEX_ROW_STEP, Cell, Grid
from railroads.sim.growth import Growth
from railroads.sim.railway import Railway
from railroads.sim.tilemap import Relief, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.ui.biomes import biome_of
from railroads.ui.hex_details import mood_text, recently_described
from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS, IsoCamera, Vec2
from railroads.ui.minimap import MinimapLayout
from railroads.ui.search import BuildingSearch, pulse
from railroads.ui.theme import MOODS, OVERLAY, TEXT, color_u32, good_icon

type Rgba = tuple[float, float, float, float]

GRID_COLOR = (0.0, 0.0, 0.0, 0.25)
HOVER_COLOR = (1.0, 1.0, 1.0, 0.9)
SELECTED_COLOR = (1.0, 0.85, 0.2, 1.0)
COMPASS_COLOR = (1.0, 1.0, 1.0, 0.85)
FOUND_COLOR = (0.3, 0.95, 1.0, 1.0)
# From this cell width, buildings found are outlined; further out each is a small square.
OUTLINE_FOUND_FROM = 16
MOST_OUTLINES = 1500
MOST_FOUND_MARKS = 6000
MOST_MINIMAP_MARKS = 3000
# Station names show from this zoom level in, this many cell widths above the station's middle.
NAMES_FROM_ZOOM = 2
NAME_LIFT = 0.55
NAME_PADDING = 6.0
STATUS_MARGIN = 10.0
STATUS_PADDING = 10.0
# How big the goods' icons are in the card of the hovered cell's buildings.
HOVER_ICON = 16.0
# Lower left, upper left, top and upper right corner: three edges per hexagon draw the whole grid.
_HEX_EDGE_ANGLES = np.radians([150, 210, 270, 330])


def color(rgba: Rgba) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))


def draw_outline(
    draw_list: imgui.ImDrawList,
    camera: IsoCamera,
    grid: Grid,
    center: Vec2,
    cell: Cell,
    rgba: Rgba = HOVER_COLOR,
) -> None:
    points = [imgui.ImVec2(*camera.to_screen(x, y, center)) for x, y in grid.corners(*cell)]
    draw_list.add_polyline(points, color(rgba), 2.0, imgui.ImDrawFlags_.closed.value)


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
    line = color(GRID_COLOR)
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
    line = color(GRID_COLOR)
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


def draw_compass(draw_list: imgui.ImDrawList, camera: IsoCamera, anchor: Vec2) -> None:
    here = camera.to_screen(camera.center_x, camera.center_y, (0, 0))
    north = camera.to_screen(camera.center_x, camera.center_y - 1, (0, 0))
    dx, dy = north[0] - here[0], north[1] - here[1]
    length = math.hypot(dx, dy)
    dx, dy = dx / length, dy / length
    line = color(COMPASS_COLOR)
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
        mark = color(FOUND_COLOR)
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
        color(FOUND_COLOR),
    )


def draw_found_on_overview(
    draw_list: imgui.ImDrawList, grid: Grid, search: BuildingSearch, minimap: MinimapLayout
) -> None:
    panel = minimap.overview
    if search.kind is None or panel is None or not search.cells:
        return
    step = max(1, len(search.cells) // MOST_MINIMAP_MARKS)
    xs, ys = grid.center(search.rows[::step], search.cols[::step])
    mark = color(FOUND_COLOR)
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
