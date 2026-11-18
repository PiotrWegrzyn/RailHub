import numpy as np

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.grids import HexGrid
from railroads.ui.history_pictures import BACKGROUND, HistoryPictures, middle_picture
from railroads.ui.iso_camera import LOOKS
from railroads.ui.pixel.projection import Rgba
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.world import PixelWorld


def small_constructions() -> Constructions:
    return Constructions(flat_map(grid=HexGrid(rows=12, cols=12, km_per_cell=2.0)))


def test_the_picture_shows_the_map_in_its_middle_and_the_background_past_its_edges() -> None:
    constructions = small_constructions()
    look = LOOKS["hex"]
    world = PixelWorld(
        constructions.tile_map, constructions, SpriteSheet.load(), look.angle_deg, look.cell_units
    )

    picture = middle_picture(world, (1000, 800))

    assert picture.shape == (800, 1000, 4)
    assert (picture[..., 3] == 255).all()
    assert tuple(picture[0, 0, :3]) == BACKGROUND
    assert tuple(picture[400, 500, :3]) != BACKGROUND


def test_a_picture_is_taken_every_fifty_years_showing_what_was_built_since() -> None:
    constructions = small_constructions()
    seen: list[tuple[int, Rgba]] = []
    pictures = HistoryPictures((400, 300), lambda year, picture: seen.append((year, picture)))

    pictures(constructions, 850)
    pictures(constructions, 875)
    for row in range(4, 8):
        for col in range(4, 8):
            constructions.build(Building((row, col), BuildingKind.HOUSE))
    pictures(constructions, 900)

    assert [year for year, _ in seen] == [850, 900]
    assert not np.array_equal(seen[0][1], seen[1][1])
