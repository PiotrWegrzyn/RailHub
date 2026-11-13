import json

import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import (
    Building,
    BuildingKind,
    Constructions,
    Station,
    Track,
    TrackWidth,
)
from railroads.sim.economy import Satisfaction
from railroads.sim.grids import Cell, Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Relief, Terrain, TileMap, Vegetation
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, TILT
from railroads.ui.pixel import track_art
from railroads.ui.pixel.art import SPRITES, draw_sheet
from railroads.ui.pixel.art.stations import TURNED_LINES
from railroads.ui.pixel.ground import AROUND, CROWN_GAP, _depths, _nearest_crown
from railroads.ui.pixel.lattice import Lattice
from railroads.ui.pixel.layer import resolution_for
from railroads.ui.pixel.noise import noise, table_hash
from railroads.ui.pixel.projection import CHUNK, ChunkPlacement, PixelProjection, hash01
from railroads.ui.pixel.relief import MOST_LIFT, raycast
from railroads.ui.pixel.sheet import FINE, SHEET_INDEX, SpriteSheet
from railroads.ui.pixel.track_art import RAIL
from railroads.ui.pixel.track_preview import ghost
from railroads.ui.pixel.world import PixelWorld

LOOKS = {HexGrid: (0.0, 1.0), SquareGrid: (-45.0, float(np.sqrt(2)))}


@pytest.fixture(scope="module")
def sheet() -> SpriteSheet:
    image, places = draw_sheet()
    return SpriteSheet.from_image(image, places)


def make_tile_map(
    grid_type: type[Grid] = HexGrid,
    hills_from_col: int | None = None,
    rise_per_col: float = 0.0,
    mountains_from_col: int | None = None,
) -> TileMap:
    grid = grid_type(rows=16, cols=12, km_per_cell=2.0)
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    if hills_from_col is not None:
        relief[:, hills_from_col:] = Relief.HILLS
    if mountains_from_col is not None:
        relief[:, mountains_from_col:] = Relief.MOUNTAINS
    heights = np.broadcast_to(np.arange(grid.cols) * rise_per_col, grid.shape).astype(np.int16)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    vegetation = np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
    vegetation[:8, 6:] = Vegetation.FOREST
    return flat_map(
        grid=grid, terrain=terrain, vegetation=vegetation, height_m=heights, relief=relief
    )


def make_world(sheet: SpriteSheet, grid_type: type[Grid] = HexGrid, houses: int = 3) -> PixelWorld:
    tile_map = make_tile_map(grid_type)
    constructions = Constructions(tile_map)
    for _ in range(houses):
        constructions.build(Building((10, 4), BuildingKind.HOUSE))
    constructions.build(Building((4, 8), BuildingKind.LUMBER_MILL))
    constructions.build(Building((12, 3), BuildingKind.FARM))
    return PixelWorld(tile_map, constructions, sheet, *LOOKS[grid_type])


def test_sheet_holds_every_sprite_once_without_overlaps() -> None:
    image, places = draw_sheet()

    boxes = [(x, y, x + w, y + h) for x, y, w, h, _, _ in places.values()]
    overlaps = [
        (a, b)
        for i, a in enumerate(boxes)
        for b in boxes[i + 1 :]
        if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
    ]
    assert list(places) == list(SPRITES)
    assert overlaps == []
    assert all(x1 <= image.width and y1 <= image.height for _, _, x1, y1 in boxes)


def test_every_sprite_stands_on_an_anchor_inside_it() -> None:
    _, places = draw_sheet()

    outside = [
        name for name, (_, _, w, h, ax, ay) in places.items() if not (0 <= ax < w and 0 <= ay < h)
    ]

    assert outside == []


def test_shipped_sheet_has_the_sprites_the_game_draws() -> None:
    shipped = SpriteSheet.load()

    assert shipped.names == list(SPRITES)
    assert list(json.loads(SHEET_INDEX.read_text())) == list(SPRITES)
    assert all(pixels[..., 3].max() == 1.0 for pixels in shipped.pixels)


def test_variants_are_found_by_name(sheet: SpriteSheet) -> None:
    houses = sheet.variants("house")

    assert [sheet.names[i] for i in houses] == [f"house_{i}" for i in range(12)]
    assert sheet.variants("farm") == []


def test_hash_is_repeatable_and_spread_between_zero_and_one() -> None:
    keys = np.arange(10_000)

    first, again, other = hash01(keys, 3), hash01(keys, 3), hash01(keys, 4)

    assert (first == again).all()
    assert (first != other).mean() > 0.99
    assert 0 <= first.min() and first.max() < 1
    assert abs(first.mean() - 0.5) < 0.02


@pytest.mark.parametrize("grid_type", [HexGrid, SquareGrid], ids=["hex", "square"])
def test_projection_turns_pixels_back_into_the_same_map_points(grid_type: type[Grid]) -> None:
    projection = PixelProjection.of(make_tile_map(grid_type).grid, *LOOKS[grid_type])

    px, py = projection.to_pixels(3.25, 7.5)

    assert projection.to_plane(px, py) == pytest.approx((3.25, 7.5))


@pytest.mark.parametrize("grid_type", [HexGrid, SquareGrid], ids=["hex", "square"])
def test_the_whole_map_fits_in_the_picture(grid_type: type[Grid]) -> None:
    grid = make_tile_map(grid_type).grid
    projection = PixelProjection.of(grid, *LOOKS[grid_type])
    width, height = grid.size()

    px, py = projection.to_pixels([0, width, width, 0], [0, 0, height, height])

    assert (px >= 0).all() and (px <= projection.size[0]).all()
    assert (py >= 0).all() and (py <= projection.size[1]).all()


def test_hexagons_are_one_pixel_cell_wide_and_squares_two_to_one_diamonds() -> None:
    hexes = PixelProjection.of(make_tile_map(HexGrid).grid, *LOOKS[HexGrid])
    squares = PixelProjection.of(make_tile_map(SquareGrid).grid, *LOOKS[SquareGrid])

    hex_step = np.subtract(hexes.to_pixels(1, 0), hexes.to_pixels(0, 0))
    east = np.subtract(squares.to_pixels(1, 0), squares.to_pixels(0, 0))

    assert hex_step == pytest.approx((PIXEL_CELL_WIDTH, 0))
    assert east == pytest.approx((PIXEL_CELL_WIDTH / 2, -PIXEL_CELL_WIDTH / 2 * TILT))


def test_woods_stand_as_a_canopy_not_trees_and_everything_is_sorted_back_to_front(
    sheet: SpriteSheet,
) -> None:
    world = make_world(sheet)
    scenery = world.scenery

    trees = np.isin(scenery.sprite, sheet.variants("pine") + sheet.variants("leafy"))
    x, y = world.projection.to_plane(scenery.x[trees], scenery.y[trees])
    rows, cols = world.tile_map.grid.cell_at(x, y)
    in_forest = world.tile_map.vegetation[rows, cols] == Vegetation.FOREST
    assert not in_forest.any()
    assert (np.diff(scenery.y) >= 0).all()


def woods_and_meadow(world: PixelWorld) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = world.tile_map.grid
    px, py = world.projection.to_pixels(*grid.center(3, 9))
    chunk = (int(px) // CHUNK, int(py) // CHUNK)
    picture = world.render(*chunk)
    left, top = chunk[0] * CHUNK, chunk[1] * CHUNK
    iy, ix = np.mgrid[0:CHUNK, 0:CHUNK]
    r, c = grid.cell_at(*world.projection.to_plane(left + ix + 0.5, top + iy + 0.5))
    inside = grid.contains(r, c)
    r, c = np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1)
    forest = inside & (world.tile_map.vegetation[r, c] == Vegetation.FOREST)
    meadow = inside & (world.tile_map.vegetation[r, c] == Vegetation.GRASS) & (r >= 9)
    return picture, forest, meadow


def test_woods_are_drawn_darker_than_meadow_with_crowns_and_floor(sheet: SpriteSheet) -> None:
    world = make_world(sheet, houses=0)

    picture, forest, meadow = woods_and_meadow(world)

    woods = picture[forest][:, :3].astype(int)
    grass = picture[meadow][:, :3].astype(int)
    assert woods.sum(axis=1).mean() < 0.75 * grass.sum(axis=1).mean()
    assert len(np.unique(woods, axis=0)) > 20


def test_woods_clear_round_what_is_built_in_them(sheet: SpriteSheet) -> None:
    world = make_world(sheet, houses=0)
    grid = world.tile_map.grid
    canvas = np.ones((CHUNK, CHUNK, 4), dtype=np.float32)
    px, py = world.projection.to_pixels(*grid.center(4, 8))
    left, top = int(px) - CHUNK // 2, int(py) - CHUNK // 2

    world.ground.canopy(canvas, left, top, 1)

    mill = canvas[CHUNK // 2 - 2 : CHUNK // 2 + 3, CHUNK // 2 - 2 : CHUNK // 2 + 3, :3]
    fx, fy = world.projection.to_pixels(*grid.center(2, 10))
    forest = canvas[int(fy) - top, int(fx) - left, :3]
    assert (mill == 1).all()
    assert (forest < 1).all()


def test_every_building_gets_its_sprite(sheet: SpriteSheet) -> None:
    scenery = make_world(sheet).scenery

    names = [sheet.names[i] for i in scenery.sprite]

    assert sum(name.startswith("house_") for name in names) == 3
    assert names.count("lumber_mill") == 1
    assert names.count("farm") == 1


def test_chunks_are_drawn_the_same_every_time(sheet: SpriteSheet) -> None:
    world = make_world(sheet)

    first, again = world.render(0, 1), world.render(0, 1)

    assert first.shape == (CHUNK, CHUNK, 4)
    assert (first == again).all()


def test_beyond_the_map_chunks_are_clear(sheet: SpriteSheet) -> None:
    world = make_world(sheet)
    across, down = world.projection.chunks

    pixels = world.render(across, down)

    assert (pixels[..., 3] == 0).all()


def test_sea_is_blue_and_land_is_green(sheet: SpriteSheet) -> None:
    world = make_world(sheet)
    grid = world.tile_map.grid
    sea = np.round(world.projection.to_pixels(*grid.center(8, 0))).astype(int)
    land = np.round(world.projection.to_pixels(*grid.center(14, 9))).astype(int)

    def color(point: np.ndarray) -> np.ndarray:
        chunk = world.render(point[0] // CHUNK, point[1] // CHUNK)
        return chunk[point[1] % CHUNK, point[0] % CHUNK, :3].astype(int)

    r, g, b = color(sea)
    lr, lg, lb = color(land)
    assert b > g > r
    assert lg > lr and lg > lb


def test_buildings_show_in_the_chunk_they_stand_in(sheet: SpriteSheet) -> None:
    with_houses, without = make_world(sheet, houses=6), make_world(sheet, houses=0)
    px, py = with_houses.projection.to_pixels(*with_houses.tile_map.grid.center(10, 4))
    chunk = (int(px) // CHUNK, int(py) // CHUNK)

    changed = with_houses.render(*chunk) != without.render(*chunk)

    assert changed.any(axis=-1).sum() > 100


def test_placement_covers_the_view_with_whole_pixel_chunks() -> None:
    projection = PixelProjection.of(SquareGrid(rows=200, cols=200, km_per_cell=2.0), 0.0, 1.0)
    clip = ((0.0, 0.0), (1280.0, 800.0))

    placement = ChunkPlacement.of(projection, (3000.3, 2000.7), 2, (640.0, 400.0), clip)

    rects = [placement.screen_rect(chunk) for chunk in placement.chunks]
    assert placement.origin == (round(640 - 6000.6), round(400 - 4001.4))
    assert min(left for (left, _), _ in rects) <= 0
    assert max(right for _, (right, _) in rects) >= 1280
    assert min(top for (_, top), _ in rects) <= 0
    assert max(bottom for _, (_, bottom) in rects) >= 800
    assert all(right - left == 2 * CHUNK for (left, _), (right, _) in rects)


def test_placement_leaves_out_chunks_past_the_picture() -> None:
    projection = PixelProjection.of(SquareGrid(rows=10, cols=10, km_per_cell=2.0), 0.0, 1.0)

    placement = ChunkPlacement.of(projection, (0.0, 0.0), 1, (640.0, 400.0), ((0, 0), (1280, 800)))

    assert placement.chunks == [(0, 0), (1, 0)]


def make_hilly_world(sheet: SpriteSheet, rise_per_col: float = 0.0) -> PixelWorld:
    tile_map = make_tile_map(hills_from_col=6, rise_per_col=rise_per_col)
    return PixelWorld(tile_map, Constructions(tile_map), sheet, *LOOKS[HexGrid])


def test_mountains_have_no_sprites_and_no_loose_rocks(sheet: SpriteSheet) -> None:
    tile_map = make_tile_map(hills_from_col=4, mountains_from_col=8)
    world = PixelWorld(tile_map, Constructions(tile_map), sheet, *LOOKS[HexGrid])
    scenery = world.scenery

    x, y = world.projection.to_plane(scenery.x, scenery.y)
    rows, cols = tile_map.grid.cell_at(x, y)
    relief = tile_map.relief[rows, cols]
    rocks = scenery.sprite == sheet.index("rocks")
    assert sheet.variants("hill") == [] and sheet.variants("mountain") == []
    assert not (rocks & (relief == Relief.MOUNTAINS)).any()


def test_slopes_facing_the_light_are_lighter_than_slopes_facing_away(sheet: SpriteSheet) -> None:
    rising_east, rising_west = make_hilly_world(sheet, 120.0), make_hilly_world(sheet, -120.0)

    facing_light = rising_east.ground._relief_light[
        rising_east.ground._relief_light.shape[0] // 2, 20:30
    ]
    facing_away = rising_west.ground._relief_light[
        rising_west.ground._relief_light.shape[0] // 2, 20:30
    ]

    assert facing_light.mean() > 0.3
    assert facing_away.mean() < -0.3


def test_new_buildings_put_up_the_scenery_again_and_name_the_chunks_to_redraw(
    sheet: SpriteSheet,
) -> None:
    world = make_world(sheet)
    cell = (6, 3)
    world.constructions.build(Building(cell, BuildingKind.FLATS))
    px, py = world.projection.to_pixels(*world.tile_map.grid.center(*cell))

    chunks = world.buildings_changed([cell])

    assert (int(px // CHUNK), int(py // CHUNK)) in chunks
    assert np.isin(world.scenery.sprite, sheet.variants("flats")).sum() == 1


def track_chunk(
    sheet: SpriteSheet,
    pieces: list[Track],
    stations: tuple[Station, ...] = (),
    at: Cell = (10, 3),
) -> tuple[PixelWorld, np.ndarray]:
    tile_map = make_tile_map()
    constructions = Constructions(tile_map)
    for piece in pieces:
        constructions.build(piece)
    for station in stations:
        constructions.build(station)
    world = PixelWorld(tile_map, constructions, sheet, *LOOKS[HexGrid])
    px, py = world.projection.to_pixels(*tile_map.grid.center(*at))
    return world, world.render(int(px // CHUNK), int(py // CHUNK))


def rails(chunk: np.ndarray) -> np.ndarray:
    """The pixels lit mostly by a rail: on grass and dark track, only rails are this light."""
    return chunk[..., :3].min(axis=-1) > 140


def rails_across(chunk: np.ndarray, column: int, rows: slice = slice(None)) -> int:
    """How many rails a column of the chunk crosses, within `rows`: how often it turns light
    and dark again."""
    light = chunk[rows, column, :3].min(axis=-1).astype(int)
    peaks = (light[1:-1] > light[:-2]) & (light[1:-1] >= light[2:]) & (light[1:-1] > 115)
    return int(peaks.sum())


def at_cell(world: PixelWorld, cell: Cell, dx: float = 0.0) -> tuple[int, int]:
    """Where a point `dx` cells right of a cell's centre lies in its chunk, as a column and row."""
    x, y = world.tile_map.grid.center(*cell)
    px, py = world.projection.to_pixels(x + dx, y)
    return int(px) % CHUNK, int(py) % CHUNK


def test_track_crossing_in_a_cell_keeps_the_rails_of_both(sheet: SpriteSheet) -> None:
    across = Track((10, 3), (0, 6))
    alone = rails(track_chunk(sheet, [across])[1])

    crossed = rails(track_chunk(sheet, [across, Track((10, 3), (3, 9))])[1])

    assert (crossed | ~alone).all()
    assert crossed.sum() > alone.sum()


def test_double_track_lays_two_tracks_side_by_side(sheet: SpriteSheet) -> None:
    world, single = track_chunk(sheet, [Track((10, 3), (0, 6))])

    _, double = track_chunk(sheet, [Track((10, 3), (0, 6), TrackWidth.DOUBLE)])

    column, _ = at_cell(world, (10, 3))
    assert rails_across(single, column) == 2
    assert rails_across(double, column) == 4


def test_electric_track_hangs_wires_over_it(sheet: SpriteSheet) -> None:
    plain = track_chunk(sheet, [Track((10, 3), (0, 6))])[1]

    electric = track_chunk(sheet, [Track((10, 3), (0, 6), electric=True)])[1]

    wired = (electric != plain).any(axis=-1)
    track_rows = np.flatnonzero((plain != track_chunk(sheet, [])[1]).any(axis=-1).any(axis=1))
    assert wired.any(axis=1)[: track_rows.min()].any()


def test_track_keeps_one_width_on_the_screen_whichever_way_it_runs(sheet: SpriteSheet) -> None:
    bare = track_chunk(sheet, [])[1]

    across = track_chunk(sheet, [Track((10, 3), (0, 6))])[1]
    upright = track_chunk(sheet, [Track((10, 3), (3, 9))])[1]

    across_width = ((across != bare).any(axis=-1)).any(axis=1).sum()
    upright_width = ((upright != bare).any(axis=-1)).any(axis=0).sum()
    assert abs(across_width - upright_width) <= 2


def test_track_into_a_station_runs_on_through_it_past_its_platform(sheet: SpriteSheet) -> None:
    line = [Track((10, c), (0, 6)) for c in (2, 3)]
    world, chunk = track_chunk(sheet, line, stations=(Station((10, 4), (0, 6)),))
    _, bare = track_chunk(sheet, line)

    column, row = at_cell(world, (10, 4), 0.4)

    track = slice(row - 2, row + 3)
    assert rails_across(chunk, column, track) == 2
    assert rails_across(bare, column, track) == 0
    assert (chunk[row - 8 : row, column, :3].min(axis=-1) > 170).any()


def test_a_station_house_stands_back_from_its_track_turned_along_it(sheet: SpriteSheet) -> None:
    across, _ = track_chunk(sheet, [], stations=(Station((10, 4), (0, 6)),))
    upright, _ = track_chunk(sheet, [], stations=(Station((10, 4), (3, 9)),))

    houses = [
        (int(world.scenery.x[found][0]), int(world.scenery.y[found][0]))
        for world, name in ((across, "station_across"), (upright, "station_upright"))
        if (found := world.scenery.sprite == sheet.index(name)).any()
    ]

    x, y = (float(v) for v in across.projection.to_pixels(*across.tile_map.grid.center(10, 4)))
    assert len(houses) == 2
    assert houses[0][1] < y - 6 and abs(houses[0][0] - x) <= 1
    assert houses[1][0] < x - 6 and abs(houses[1][1] - y) <= 1


def test_a_station_house_is_drawn_turned_each_way_track_runs_through_it(sheet: SpriteSheet) -> None:
    looks = [sheet.index(f"station_{line}") for line in TURNED_LINES]

    sizes = {tuple(sheet.sizes[index]) for index in looks}

    assert len(sizes) == len(TURNED_LINES)


def test_planned_track_shows_as_the_real_track_faded_to_a_see_through_grey() -> None:
    pixels = np.array([[[0.33, 0.24, 0.16, 1.0], [0.7, 0.7, 0.68, 1.0], [0.0, 0.0, 0.0, 0.0]]])

    faded = ghost(pixels).astype(int)

    spread = faded[0, :2, :3].max(axis=1) - faded[0, :2, :3].min(axis=1)
    assert (spread < 20).all()
    assert faded[0, 1, 0] > faded[0, 0, 0] + 60
    assert 0 < faded[0, 0, 3] < 255
    assert faded[0, 2, 3] == 0


@pytest.fixture(scope="module")
def fine_sheet() -> SpriteSheet:
    image, places = draw_sheet(resolution=FINE)
    return SpriteSheet.from_image(image, places)


def test_the_fine_sheet_holds_the_same_sprites_twice_as_big_on_the_same_spots(
    sheet: SpriteSheet, fine_sheet: SpriteSheet
) -> None:
    shipped = SpriteSheet.load_fine()

    assert shipped.names == fine_sheet.names == sheet.names
    assert np.abs(fine_sheet.sizes - FINE * sheet.sizes).max() <= 2
    assert np.abs(fine_sheet.anchors - FINE * sheet.anchors).max() <= 2


def test_a_fine_chunk_keeps_the_ground_and_draws_what_stands_on_it_finer(
    sheet: SpriteSheet, fine_sheet: SpriteSheet
) -> None:
    tile_map = make_tile_map()
    constructions = Constructions(tile_map)
    world = PixelWorld(tile_map, constructions, sheet, *LOOKS[HexGrid], fine=fine_sheet)
    bare = world.render(0, 1)
    for _ in range(6):
        constructions.build(Building((10, 4), BuildingKind.HOUSE))
    world.buildings_changed([(10, 4)])
    px, py = world.projection.to_pixels(*tile_map.grid.center(10, 4))
    chunk = (int(px) // CHUNK, int(py) // CHUNK)

    coarse, fine = world.render(*chunk), world.render(*chunk, FINE)

    blown_up = coarse.repeat(FINE, axis=0).repeat(FINE, axis=1)
    assert world.resolutions == (1, FINE)
    assert fine.shape == (CHUNK * FINE, CHUNK * FINE, 4)
    assert (world.render(0, 1, FINE) == bare.repeat(FINE, axis=0).repeat(FINE, axis=1)).all()
    assert (fine != blown_up).any(axis=-1).sum() > 100


def test_fine_track_keeps_one_pixel_rails_over_twice_the_pixels(sheet: SpriteSheet) -> None:
    tile_map = make_tile_map()
    world = PixelWorld(tile_map, Constructions(tile_map), sheet, *LOOKS[HexGrid])
    piece = Track((10, 3), (0, 6))
    px, py = world.projection.to_pixels(*tile_map.grid.center(10, 3))
    left, top = int(px) - 32, int(py) - 32

    coarse = world.track.layer([piece], left, top, 64, 64)
    fine = world.track.layer([piece], left, top, 64, 64, FINE)

    rails = [
        np.all(np.abs(layer[..., :3] - np.array(RAIL) / 255) < 0.01, axis=-1)
        for layer in (coarse, fine)
    ]
    assert fine.shape == (128, 128, 4)
    assert rails[0].sum() < rails[1].sum() < 3 * rails[0].sum()


@pytest.mark.parametrize(("screen", "resolution"), [(1, 1), (2, 2), (3, 1), (4, 2), (8, 2)])
def test_chunks_are_drawn_fine_where_they_blow_up_evenly(screen: int, resolution: int) -> None:
    chosen = resolution_for(screen, (1, FINE))

    assert chosen == resolution


def test_meadow_grass_is_shaded_smoothly_rather_than_in_a_few_steps(sheet: SpriteSheet) -> None:
    world = make_world(sheet, houses=0)
    px, py = world.projection.to_pixels(*world.tile_map.grid.center(12, 4))
    left, top = int(px) - 40, int(py) - 20

    rgb, _ = world.ground.render(left, top, 40)

    shades = np.unique(rgb[:, :80].round().astype(int).reshape(-1, 3), axis=0)
    assert len(shades) > 40


def test_grass_lies_darker_in_the_shade_of_a_tree(sheet: SpriteSheet) -> None:
    world = make_world(sheet, houses=0)
    px, py = world.projection.to_pixels(*world.tile_map.grid.center(12, 4))
    left, top = int(px) - 40, int(py) - 20
    world.ground.trees = (np.zeros(0, np.float32), np.zeros(0, np.float32))
    open_grass, _ = world.ground.render(left, top, 40)

    world.ground.trees = (np.array([px], np.float32), np.array([py], np.float32))
    shaded, _ = world.ground.render(left, top, 40)

    foot = (20 + 1, 40 + 3)
    assert shaded[foot].sum() < 0.85 * open_grass[foot].sum()
    assert np.array_equal(shaded[:, :20], open_grass[:, :20])


def test_only_a_change_of_glow_marks_chunks_to_draw_again(sheet: SpriteSheet) -> None:
    world = make_world(sheet)
    levels = world.satisfaction.copy()
    levels[3, 3] = Satisfaction.UNSATISFIED
    unseen = world.satisfaction_changed(levels)
    levels = levels.copy()
    levels[3, 3] = Satisfaction.THRIVING

    seen = world.satisfaction_changed(levels)

    assert unseen == set()
    assert seen


def test_a_chunk_drawn_over_ground_laid_on_its_ground_matches_the_whole_chunk(
    sheet: SpriteSheet,
) -> None:
    world = make_world(sheet)
    world.ground.trees = (np.zeros(0), np.zeros(0))
    rgb, alpha = world.ground.render(0, 0, CHUNK)

    whole = world.render(0, 0)
    over = world.render(0, 0, over_ground=True)

    cover = over[..., 3:] / 255
    laid = np.rint((over[..., :3] / 255 * cover + rgb / 255 * (1 - cover)) * 255)
    difference = np.abs(laid - whole[..., :3])[alpha > 0]
    assert (over[..., 3] < 255).any()
    assert difference.max() <= 1


def woods_sampled_at_every_pixel(
    world: PixelWorld, left: int, top: int, width: int, rows: int
) -> np.ndarray:
    grid = world.tile_map.grid
    iy, ix = np.mgrid[top : top + rows, left : left + width]
    x, y = world.projection.to_plane(ix + 0.5, iy + 0.5)
    r, c = grid.cell_at(x, y)
    r, c = np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1)
    share = np.zeros(x.shape, dtype=np.float32)
    for dx, dy in AROUND:
        sr, sc = grid.cell_at(x + dx, y + dy)
        sr, sc = np.clip(sr, 0, grid.rows - 1), np.clip(sc, 0, grid.cols - 1)
        share += world.ground._woods[sr, sc]
    share /= len(AROUND)
    edge = noise(x, y, 0.55, 61) * 0.55 + noise(x, y, 0.16, 62) * 0.3
    land = world.tile_map.terrain[r, c] == Terrain.LAND
    return land & (share + (edge - 0.42) * 0.9 > 0.5)


@pytest.mark.parametrize("grid_type", [HexGrid, SquareGrid], ids=["hex", "square"])
def test_woods_cover_what_the_cells_round_each_pixel_say_without_looking_at_all(
    sheet: SpriteSheet, grid_type: type[Grid]
) -> None:
    world = make_world(sheet, grid_type)
    grid = world.tile_map.grid

    for cell in ((4, 6), (0, 11)):
        px, py = world.projection.to_pixels(*grid.center(*cell))
        left, top, cols, rows = int(px) - 110, int(py) - 100, 220, 200
        points = Lattice(world.projection, left, top, cols, rows, grid=grid)
        r, c, _ = points.cells(left, top, cols, rows)

        woods = world.ground._wooded(points, (left, top, cols, rows), r, c)

        expected = woods_sampled_at_every_pixel(world, left, top, cols, rows)
        assert expected.any()
        assert np.array_equal(woods, expected)


def nearest_crown_by_hashing(px: np.ndarray, py: np.ndarray, gap: float) -> tuple:
    cx, cy = np.floor(px / gap).astype(np.int64), np.floor(py / gap).astype(np.int64)
    best = np.full(px.shape, np.inf, dtype=np.float32)
    best_dx, best_dy = np.zeros(px.shape, np.float32), np.zeros(px.shape, np.float32)
    best_x, best_y = cx.copy(), cy.copy()
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            gx, gy = cx + ox, cy + oy
            jx = (gx + 0.15 + 0.7 * table_hash(gx, gy, 51)) * gap
            jy = (gy + 0.15 + 0.7 * table_hash(gx, gy, 52)) * gap
            dx, dy = (px - jx).astype(np.float32), (py - jy).astype(np.float32)
            distance = dx * dx + dy * dy
            closer = distance < best
            best = np.where(closer, distance, best)
            best_dx, best_dy = np.where(closer, dx, best_dx), np.where(closer, dy, best_dy)
            best_x, best_y = np.where(closer, gx, best_x), np.where(closer, gy, best_y)
    return np.sqrt(best), best_dx, best_dy, best_x, best_y


def test_the_nearest_crown_is_the_one_found_by_hashing_the_squares_round_each_point() -> None:
    rng = np.random.default_rng(8)
    px = np.floor(rng.uniform(-300, 900, 20_000)) + 0.5
    py = np.floor(rng.uniform(-50, 700, 20_000)) + 0.5

    found = _nearest_crown(px, py, CROWN_GAP)

    for value, expected in zip(found, nearest_crown_by_hashing(px, py, CROWN_GAP), strict=True):
        assert value.dtype == expected.dtype
        assert np.array_equal(value, expected)


def test_a_chunk_showing_ground_lifted_as_far_as_ground_goes_is_drawn(
    sheet: SpriteSheet, fine_sheet: SpriteSheet
) -> None:
    grid = HexGrid(rows=16, cols=12, km_per_cell=1.0)
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    relief[:, 6:] = Relief.MOUNTAINS
    heights = np.broadcast_to(np.where(np.arange(grid.cols) >= 6, 1500, 0), grid.shape)
    tile_map = flat_map(grid=grid, height_m=heights.astype(np.int16), relief=relief)
    constructions = Constructions(tile_map)
    world = PixelWorld(tile_map, constructions, sheet, *LOOKS[HexGrid], fine=fine_sheet)
    raised = world.elevation.area(0, 0, CHUNK, CHUNK + MOST_LIFT)

    pictures = [world.render(0, 0, k) for k in (1, FINE)]

    assert raised is not None and raycast(raised.lift, CHUNK, CHUNK).max() == MOST_LIFT
    assert [picture.shape[0] for picture in pictures] == [CHUNK, FINE * CHUNK]
    assert all((picture[..., 3] == 255).any() for picture in pictures)


def test_track_drawn_again_after_its_neighbour_changes_matches_track_drawn_fresh(
    sheet: SpriteSheet,
) -> None:
    double = Track((10, 3), (0, 6), TrackWidth.DOUBLE)
    single = Track((10, 4), (0, 6))
    world, before = track_chunk(sheet, [double, single])
    widened = Track((10, 4), (0, 6), TrackWidth.DOUBLE)

    world.constructions.remove(single)
    world.constructions.build(widened)
    px, py = world.projection.to_pixels(*world.tile_map.grid.center(10, 3))
    again = world.render(int(px // CHUNK), int(py // CHUNK))

    _, fresh = track_chunk(sheet, [double, widened])
    assert not np.array_equal(before, fresh)
    assert np.array_equal(again, fresh)


@pytest.mark.parametrize("k", [1, 2])
def test_the_points_of_a_patch_of_fields_are_those_of_a_box_round_it_cut_to_it(
    sheet: SpriteSheet, k: int
) -> None:
    ground = make_world(sheet).ground
    left, top = 30, 50
    around = ground._patch_points((0, 48 * k, 0, 90 * k), left, top, k, None)

    alone = ground._patch_points((10 * k, 34 * k, 20 * k, 60 * k), left, top, k, None)

    place = (slice(10 * k, 34 * k), slice(20 * k, 60 * k))
    for values, cut in zip(alone, around, strict=True):
        assert np.array_equal(values, cut[place])


def test_track_is_drawn_the_same_when_fewer_pieces_are_kept_than_are_drawn(
    sheet: SpriteSheet, monkeypatch: pytest.MonkeyPatch
) -> None:
    pieces = [Track((10, col), (0, 6)) for col in range(2, 7)]
    _, all_kept = track_chunk(sheet, pieces)
    monkeypatch.setattr(track_art, "PIECES_KEPT", 2)

    world, few_kept = track_chunk(sheet, pieces)

    assert 0 < len(world.track._laid_pieces) <= 2
    assert np.array_equal(few_kept, all_kept)


def depths_by_rolling(field: np.ndarray, most: int) -> np.ndarray:
    depth = np.zeros(field.shape, dtype=np.int16)
    inside = field.copy()
    for step in range(1, most + 1):
        inside = inside & np.roll(inside, 1, 0) & np.roll(inside, -1, 0)
        inside = inside & np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
        depth[inside] = step
    return depth[field]


@pytest.mark.parametrize("shape", [(1, 1), (2, 7), (24, 40), (48, 80)])
def test_depths_into_a_field_are_those_of_shrinking_it_with_wrapping_edges(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(sum(shape))
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]

    for _ in range(40):
        cy, cx = rng.uniform(-3, shape[0] + 3), rng.uniform(-3, shape[1] + 3)
        reach = rng.uniform(0.5, max(shape))
        field = (rows - cy) ** 2 + ((cols - cx) / 1.7) ** 2 < reach**2
        field &= rng.random(shape) < 0.97
        if not field.any():
            continue

        depths = _depths(field, 4)

        assert np.array_equal(depths, depths_by_rolling(field, 4))
