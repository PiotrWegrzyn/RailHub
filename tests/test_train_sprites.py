import math

import numpy as np
import numpy.typing as npt
import pytest

from railroads.sim.goods import Good
from railroads.sim.grids import HexGrid
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, ZOOM_CELL_WIDTHS
from railroads.ui.map_view import LOOKS, GridLook
from railroads.ui.pixel.models import draw_model
from railroads.ui.pixel.projection import PixelProjection
from railroads.ui.pixel.train_layer import MAX_TEXTURE
from railroads.ui.pixel.trains import (
    DETAIL,
    HEADINGS,
    MODELS,
    SCALE,
    Look,
    TrainSprites,
    train_sprites,
)

# Between them built of every kind of face: boxes, plates, cylinders lying and standing, slopes,
# noses and logs moved across their car. On the tank wagon some faces lie equally far back, so
# which is painted last hangs on their depths being summed exactly as draw_model sums them.
EVERY_KIND_OF_FACE = {
    "planet": Look.ENGINE,
    "express": (Look.ENGINE, "express"),
    "pacific tender": (Look.TENDER, "pacific"),
    "lumber": (Good.LUMBER, True),
    "oil": (Good.OIL, True),
    "tank wagon": (Good.TANKS, True),
}


def projection() -> PixelProjection:
    return PixelProjection.of(HexGrid(rows=10, cols=10, km_per_cell=2.0), 0.0, 1.0)


def picture(sprites: TrainSprites, look: object, index: int) -> npt.NDArray[np.uint8]:
    x, y, width, height, _, _ = sprites.places[(look, index)]
    return sprites.image[y : y + height, x : x + width]


def test_every_engine_and_car_is_drawn_for_every_heading() -> None:
    sprites = train_sprites(tuple(float(v) for v in projection().matrix.ravel()))

    looks = {look for look, _ in sprites.places}

    assert looks == set(MODELS)
    assert len(sprites.places) == len(MODELS) * HEADINGS


def test_a_car_running_east_is_longer_than_it_is_tall_and_outlined() -> None:
    sprites = train_sprites(tuple(float(v) for v in projection().matrix.ravel()))

    pixels = picture(sprites, (Good.GOODS, True), 0)

    solid = pixels[..., 3] > 0
    rows, cols = np.nonzero(solid)
    assert np.ptp(cols) > np.ptp(rows)
    assert tuple(pixels[rows.min(), cols[rows == rows.min()][0], :3]) == (22, 18, 18)


def test_a_full_coal_car_shows_its_load() -> None:
    sprites = train_sprites(tuple(float(v) for v in projection().matrix.ravel()))

    empty = picture(sprites, (Good.COAL, False), 0)
    full = picture(sprites, (Good.COAL, True), 0)

    assert (full[..., 3] > 0).sum() > (empty[..., 3] > 0).sum()


@pytest.mark.parametrize("grid_look", LOOKS.values(), ids=LOOKS.keys())
@pytest.mark.parametrize("look", EVERY_KIND_OF_FACE.values(), ids=EVERY_KIND_OF_FACE.keys())
def test_the_sheet_holds_the_pictures_draw_model_draws(look: object, grid_look: GridLook) -> None:
    grid = HexGrid(rows=10, cols=10, km_per_cell=2.0)
    matrix = PixelProjection.of(grid, grid_look.angle_deg, grid_look.cell_units).matrix
    drawn = [
        draw_model(MODELS[look](), 2 * math.pi * index / HEADINGS, matrix, DETAIL * SCALE)
        for index in range(HEADINGS)
    ]

    sprites = train_sprites(tuple(float(v) for v in matrix.ravel()))

    for index, (image, anchor) in enumerate(drawn):
        assert np.array_equal(picture(sprites, look, index), np.asarray(image))
        assert sprites.places[(look, index)][4:] == anchor


def test_the_nearest_heading_is_picked() -> None:
    sprites = train_sprites(tuple(float(v) for v in projection().matrix.ravel()))

    east = sprites.place(Look.ENGINE, 0.01)
    almost_east = sprites.place(Look.ENGINE, 2 * math.pi - 0.01)

    assert east == almost_east


def test_the_sheet_blown_up_for_the_closest_zoom_on_a_sharp_screen_still_fits_a_texture() -> None:
    projection = PixelProjection.of(HexGrid(rows=10, cols=10, km_per_cell=2.0), 0.0, 1.0)
    closest = ZOOM_CELL_WIDTHS[-1] // PIXEL_CELL_WIDTH * 2

    sprites = train_sprites(tuple(float(v) for v in projection.matrix.ravel()))

    height, width = sprites.image.shape[:2]
    assert max(height, width) * (closest // DETAIL) <= MAX_TEXTURE
