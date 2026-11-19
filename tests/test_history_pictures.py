import numpy as np

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.grids import HexGrid
from railroads.ui.history_pictures import HistoryPictures, Picture, middle_picture, opening_center
from railroads.ui.iso_camera import LOOKS
from railroads.ui.pixel.projection import CHUNK, PixelProjection
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.world import PixelWorld


def small_constructions() -> Constructions:
    return Constructions(flat_map(grid=HexGrid(rows=12, cols=12, km_per_cell=2.0)))


def small_world() -> PixelWorld:
    constructions = small_constructions()
    look = LOOKS["hex"]
    return PixelWorld(
        constructions.tile_map, constructions, SpriteSheet.load(), look.angle_deg, look.cell_units
    )


def test_the_picture_shows_the_map_in_its_middle_and_nothing_past_its_edges() -> None:
    world = small_world()

    picture, _ = middle_picture(world, (1000, 800))

    assert picture.shape == (800, 1000, 4)
    assert picture[0, 0, 3] == 0
    assert picture[400, 500, 3] == 255


def test_the_opening_view_lies_near_the_middle_with_its_corner_on_a_chunks() -> None:
    grid = HexGrid(rows=300, cols=300, km_per_cell=2.0)
    look = LOOKS["hex"]
    projection = PixelProjection.of(grid, look.angle_deg, look.cell_units)

    center = opening_center(grid, (1280, 800))

    px, py = projection.to_pixels(*center)
    middle = projection.to_pixels(*(size / 2 for size in grid.size()))
    assert abs(px - middle[0]) <= CHUNK / 2
    assert abs(py - middle[1]) <= CHUNK / 2
    assert round(px - 640) % CHUNK == 0
    assert round(py - 400) % CHUNK == 0


def test_over_the_ground_the_picture_leaves_open_land_clear_for_the_graphics_card() -> None:
    world = small_world()

    whole, _ = middle_picture(world, (1000, 800))
    over, _ = middle_picture(world, (1000, 800), over_ground=True)

    assert whole[400, 500, 3] == 255
    assert over[400, 500, 3] < 255


def test_a_picture_is_taken_every_hundred_years_from_850_showing_what_was_built_since() -> None:
    constructions = small_constructions()
    seen: list[Picture] = []
    pictures = HistoryPictures((400, 300), seen.append)

    pictures(constructions, 850)
    pictures(constructions, 900)
    for row in range(4, 8):
        for col in range(4, 8):
            constructions.build(Building((row, col), BuildingKind.HOUSE))
    pictures(constructions, 950)

    assert [picture.year for picture in seen] == [850, 950]
    assert not np.array_equal(seen[0].pixels, seen[1].pixels)
    assert seen[0].biomes is None


def test_pictures_over_the_ground_bring_the_biomes_and_the_first_the_ground_that_stays() -> None:
    constructions = small_constructions()
    seen: list[Picture] = []
    pictures = HistoryPictures((400, 300), seen.append, over_ground=True)

    pictures(constructions, 850)
    pictures(constructions, 950)

    first, second = seen
    assert first.biomes is not None and second.biomes is not None
    assert first.biomes.shape == constructions.tile_map.grid.shape
    assert first.ground is not None
    assert first.ground.shape == constructions.tile_map.grid.shape
    assert second.ground is None
