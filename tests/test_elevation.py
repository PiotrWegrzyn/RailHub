import numpy as np

from builders import flat_map, layer
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.tilemap import Relief, TileMap
from railroads.ui.iso_camera import IsoCamera
from railroads.ui.map_view import LOOKS, OnGround
from railroads.ui.pixel.projection import PixelProjection
from railroads.ui.pixel.relief import (
    MOST_LIFT,
    SNOW_LIT,
    Elevation,
    Raised,
    raise_picture,
    raycast,
)
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.world import PixelWorld

ROWS, COLS = 20, 30


def mountain_map() -> TileMap:
    relief = layer(ROWS, COLS, Relief.FLAT)
    relief[:, 18:] = Relief.MOUNTAINS
    relief[:, 10:13] = Relief.HILLS
    heights = np.where(relief == Relief.MOUNTAINS, 1500, 200).astype(np.int16)
    return flat_map(ROWS, COLS, relief=relief, height_m=heights)


def elevation_of(tile_map: TileMap) -> Elevation:
    grid = tile_map.grid
    angle, units = LOOKS[grid.kind].angle_deg, LOOKS[grid.kind].cell_units
    return Elevation(tile_map, PixelProjection.of(grid, angle, units))


def test_mountains_rise_hills_roll_and_plains_stay_flat() -> None:
    tile_map = mountain_map()
    elevation = elevation_of(tile_map)
    grid = tile_map.grid

    mountain, hill, plain = (grid.center(10, c) for c in (25, 11, 4))
    lifts = [float(elevation.lift(*point)) for point in (mountain, hill, plain)]

    assert lifts[0] > 8
    assert 0 <= lifts[1] < lifts[0]
    assert lifts[2] == 0


def test_each_pixel_shows_the_nearest_ground_raised_up_to_it() -> None:
    lift = np.zeros((10, 1), dtype=np.float32)
    lift[6] = 4

    offset = raycast(lift, 6, 1)

    assert offset[:, 0].tolist() == [0, 0, 4, 3, 2, 1]


def test_a_raised_picture_takes_each_pixel_from_the_ground_below_it() -> None:
    picture = np.arange(8, dtype=np.float64).reshape(8, 1, 1).repeat(2, axis=1)
    offset = np.array([[2], [0]])

    raised = raise_picture(picture, offset, 2)

    assert raised[:, 0, 0].tolist() == [4, 5, 2, 3]


def test_the_point_under_the_mouse_on_a_mountain_lies_further_down_the_map() -> None:
    tile_map = mountain_map()
    elevation = elevation_of(tile_map)
    x, y = tile_map.grid.center(10, 25)
    px, py = elevation.projection.to_pixels(x, y)
    lift = float(elevation.lift(x, y))

    seen = elevation.ground_at(float(px), float(py) - lift)

    assert seen[1] > y - 0.3
    assert abs(seen[0] - x) < 1e-6


def test_snow_lies_high_on_mountains_and_not_on_hills() -> None:
    elevation = elevation_of(mountain_map())
    ones = np.ones((1, 2), dtype=np.float32)
    raised = Raised(
        lift=ones * 20,
        rise=np.array([[1800, 120]], dtype=np.float32),
        mountains=np.array([[1, 0]], dtype=np.float32),
        hills=np.array([[0, 1]], dtype=np.float32),
        shade=ones * 0.8,
        steep=ones * 0.05,
    )
    rgb = np.full((1, 2, 3), 100, dtype=np.float32)

    elevation.colour(rgb, raised, ones * 1.0, ones * 1.0, np.array([[True, False]]))

    assert np.abs(rgb[0, 0] - SNOW_LIT).max() < 40
    assert rgb[0, 1, 1] < 160


def test_buildings_on_mountains_stand_on_the_raised_ground() -> None:
    tile_map = mountain_map()
    constructions = Constructions(tile_map)
    constructions.build(Building((10, 25), BuildingKind.MINE))
    constructions.build(Building((10, 4), BuildingKind.HOUSE))
    sheet = SpriteSheet.load()

    look = LOOKS[tile_map.grid.kind]

    world = PixelWorld(tile_map, constructions, sheet, look.angle_deg, look.cell_units)

    mine = world.scenery.sprite == sheet.index("mine")
    houses = np.isin(world.scenery.sprite, sheet.variants("house"))
    assert world.scenery.lift[mine].min() > 8
    assert (world.scenery.lift[houses] == 0).all()


def test_marks_on_the_map_follow_the_raised_ground() -> None:
    tile_map = mountain_map()
    elevation = elevation_of(tile_map)
    camera = IsoCamera(15.0, 8.0, 0.0, 1.0, zoom_level=5)
    x, y = tile_map.grid.center(10, 25)

    flat = camera.to_screen(x, y, (0.0, 0.0))
    raised = OnGround(camera, elevation).to_screen(x, y, (0.0, 0.0))

    assert raised[0] == flat[0]
    assert flat[1] - raised[1] == float(elevation.lift(x, y)) * camera.pixel_scale
    assert 0 < flat[1] - raised[1] <= MOST_LIFT * camera.pixel_scale


def world_over_mountains() -> tuple[PixelWorld, tuple[int, int], tuple[int, int]]:
    tile_map = mountain_map()
    look = LOOKS[tile_map.grid.kind]
    world = PixelWorld(
        tile_map,
        Constructions(tile_map),
        SpriteSheet.load(),
        look.angle_deg,
        look.cell_units,
        fine=SpriteSheet.load_fine(),
    )
    grid = tile_map.grid
    chunks = []
    for col in (25, 3):
        px, py = world.projection.to_pixels(*grid.center(10, col))
        chunks.append((int(px) // 256, int(py) // 256))
    return world, chunks[0], chunks[1]


def test_fine_relief_draws_raised_ground_finely_and_flat_ground_as_before() -> None:
    world, mountains, plains = world_over_mountains()

    fast, fine = world.render(*mountains, 2), world.render(*mountains, 2, fine_relief=True)

    assert fine.shape == fast.shape
    assert (fine != fast).any(axis=-1).mean() > 0.05
    assert world.rises(*mountains)
    assert not world.rises(*plains)
    np.testing.assert_array_equal(world.render(*plains, 2, True), world.render(*plains, 2))


def test_a_lone_mountain_cell_rises_to_a_peak_of_its_own() -> None:
    relief = layer(ROWS, COLS, Relief.FLAT)
    relief[10, 15] = Relief.MOUNTAINS
    heights = np.where(relief == Relief.MOUNTAINS, 1300, 150).astype(np.int16)
    elevation = elevation_of(flat_map(ROWS, COLS, relief=relief, height_m=heights))
    x, y = flat_map(ROWS, COLS).grid.center(10, 15)

    assert float(elevation.lift(x, y)) > 20
    assert float(elevation.lift(x + 1.0, y)) < 5


def test_a_remembered_lift_is_the_same_as_working_it_out() -> None:
    elevation = elevation_of(mountain_map())
    points = [(16 + x * 0.7, 4 + y * 0.9) for x in range(12) for y in range(12)]

    first = [elevation.lift_at(x, y) for x, y in points]
    again = [elevation.lift_at(x, y) for x, y in points]

    assert first == again == [float(elevation.lift(x, y)) for x, y in points]
    assert any(lift > 0 for lift in first)
