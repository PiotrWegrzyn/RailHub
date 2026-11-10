import json

import numpy as np
import pytest

from railroads.sim.constructions import Building, BuildingKind, Constructions, Track, TrackWidth
from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.ui.biomes import Biome
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, TILT
from railroads.ui.pixel.art import SPRITES, draw_sheet, relief
from railroads.ui.pixel.layer import resolution_for
from railroads.ui.pixel.projection import CHUNK, ChunkPlacement, PixelProjection, hash01
from railroads.ui.pixel.sheet import FINE, SHEET_INDEX, SpriteSheet
from railroads.ui.pixel.track_art import RAIL, WIRE
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
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=terrain,
        vegetation=vegetation,
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=heights,
        relief=relief,
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


def test_trees_grow_only_in_unbuilt_forest_and_everything_is_sorted_back_to_front(
    sheet: SpriteSheet,
) -> None:
    world = make_world(sheet)
    scenery = world.scenery

    trees = np.isin(scenery.sprite, sheet.variants("pine") + sheet.variants("leafy"))
    x, y = world.projection.to_plane(scenery.x[trees], scenery.y[trees])
    rows, cols = world.tile_map.grid.cell_at(x, y)
    in_forest = world.tile_map.vegetation[rows, cols] == Vegetation.FOREST
    in_mill_cell = (rows == 4) & (cols == 8)
    assert in_forest.mean() > 0.95
    assert not in_mill_cell.any()
    assert (np.diff(scenery.y) >= 0).all()


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


def test_hill_sprites_stand_only_on_hill_cells_tinted_by_their_biome(sheet: SpriteSheet) -> None:
    world = make_hilly_world(sheet)
    scenery = world.scenery

    hills = np.isin(scenery.sprite, sheet.variants("hill"))
    x, y = world.projection.to_plane(scenery.x[hills], scenery.y[hills])
    rows, cols = world.tile_map.grid.cell_at(x, y)
    on_hills = world.tile_map.relief[rows, cols] == Relief.HILLS
    assert hills.sum() > 20
    assert on_hills.mean() > 0.9
    assert (scenery.tint[hills] >= 0).all()
    assert (scenery.tint[~hills] == -1).all()


def test_tinted_hills_take_the_shades_of_their_ground(sheet: SpriteSheet) -> None:
    world = make_hilly_world(sheet)
    hill = sheet.variants("hill")[0]
    grass, desert = int(Biome.GRASSLAND), int(Biome.DESERT)

    on_grass, on_sand = world._tinted(hill, grass), world._tinted(hill, desert)

    opaque = sheet.pixels[hill][..., 3] == 1
    grass_rgb, sand_rgb = on_grass[opaque, :3].mean(axis=0), on_sand[opaque, :3].mean(axis=0)
    assert grass_rgb[1] > grass_rgb[0] and grass_rgb[1] > grass_rgb[2]
    assert sand_rgb[0] > sand_rgb[2]


def test_mountain_sprites_stand_only_on_mountain_cells_without_loose_rocks(
    sheet: SpriteSheet,
) -> None:
    tile_map = make_tile_map(hills_from_col=4, mountains_from_col=8)
    world = PixelWorld(tile_map, Constructions(tile_map), sheet, *LOOKS[HexGrid])
    scenery = world.scenery

    x, y = world.projection.to_plane(scenery.x, scenery.y)
    rows, cols = tile_map.grid.cell_at(x, y)
    relief = tile_map.relief[rows, cols]
    mountains = np.isin(scenery.sprite, sheet.variants("mountain"))
    rocks = scenery.sprite == sheet.index("rocks")
    assert mountains.sum() > 40
    assert (relief[mountains] == Relief.MOUNTAINS).mean() > 0.9
    assert (scenery.tint[mountains] == int(Biome.MOUNTAIN)).mean() > 0.9
    assert not (rocks & (relief == Relief.MOUNTAINS)).any()


def test_snow_on_mountain_tops_keeps_its_colour_when_tinted(sheet: SpriteSheet) -> None:
    world = make_hilly_world(sheet)
    mountain = sheet.variants("mountain")[0]
    pixels = sheet.pixels[mountain]
    snow = (pixels[..., 3] == 1) & (pixels[..., 2] > pixels[..., 0])

    tinted = world._tinted(mountain, int(Biome.DESERT))

    assert snow.sum() > 5
    np.testing.assert_allclose(tinted[snow, :3], pixels[snow, :3])


@pytest.mark.parametrize("peaked", [False, True])
def test_relief_sprites_are_lit_on_the_left_and_shaded_on_the_right(peaked: bool) -> None:
    sprite = relief(28, 10, 0, peaked=peaked)

    pixels = np.asarray(sprite.image, dtype=np.float64)
    middle = sprite.anchor[0]
    opaque, grey = pixels[..., 3] == 255, pixels[..., 0]
    left = grey[:, :middle][opaque[:, :middle]].mean()
    right = grey[:, middle:][opaque[:, middle:]].mean()
    assert left > right + 20


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


def track_pixels(
    sheet: SpriteSheet, pieces: list[Track], color: tuple[int, int, int]
) -> np.ndarray:
    tile_map = make_tile_map()
    constructions = Constructions(tile_map)
    for piece in pieces:
        constructions.build(piece)
    world = PixelWorld(tile_map, constructions, sheet, *LOOKS[HexGrid])
    px, py = world.projection.to_pixels(*tile_map.grid.center(*pieces[0].cell))
    chunk = world.render(int(px // CHUNK), int(py // CHUNK))
    return (chunk[..., :3] == color).all(axis=-1)


def test_track_crossing_in_a_cell_keeps_the_rails_of_both(sheet: SpriteSheet) -> None:
    across = Track((10, 3), (0, 6))
    alone = track_pixels(sheet, [across], RAIL)

    crossed = track_pixels(sheet, [across, Track((10, 3), (3, 9))], RAIL)

    assert (crossed | ~alone).all()
    assert crossed.sum() > alone.sum()


def test_double_track_lays_two_tracks_side_by_side(sheet: SpriteSheet) -> None:
    single = track_pixels(sheet, [Track((10, 3), (0, 6))], RAIL)

    double = track_pixels(sheet, [Track((10, 3), (0, 6), TrackWidth.DOUBLE)], RAIL)

    assert double.sum() > 1.6 * single.sum()


def test_electric_track_hangs_wires_over_it(sheet: SpriteSheet) -> None:
    plain = track_pixels(sheet, [Track((10, 3), (0, 6))], WIRE)

    electric = track_pixels(sheet, [Track((10, 3), (0, 6), electric=True)], WIRE)

    assert plain.sum() == 0
    assert electric.sum() > 0


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
