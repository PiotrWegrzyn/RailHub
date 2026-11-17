import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from railroads.sim.goods import Good
from railroads.sim.grids import HexGrid
from railroads.ui.map_view import LOOKS, GridLook
from railroads.ui.pixel.models import SHADES, Model, draw_model, shade, without_fine
from railroads.ui.pixel.projection import PixelProjection
from railroads.ui.pixel.trains import (
    DETAIL,
    HEADINGS,
    MODELS,
    SCALE,
    Look,
    TrainSprites,
    fine_pictures,
    train_sprites,
)

# Between them built of every kind of face: boxes, plates, cylinders lying and standing, slopes,
# noses and logs moved across their car. On the tank wagon some faces lie equally far back, so
# which is painted last hangs on their depths being summed exactly as draw_model sums them.
EVERY_KIND_OF_FACE = {
    "planet": (Look.ENGINE, "planet"),
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

    east = sprites.place((Look.ENGINE, "planet"), 0.01)
    almost_east = sprites.place((Look.ENGINE, "planet"), 2 * math.pi - 0.01)

    assert east == almost_east


@pytest.mark.parametrize("look", EVERY_KIND_OF_FACE.values(), ids=EVERY_KIND_OF_FACE.keys())
def test_fine_pictures_are_the_pictures_draw_model_draws_fine(look: object) -> None:
    matrix = projection().matrix
    drawn = [
        draw_model(MODELS[look](), 2 * math.pi * index / HEADINGS, matrix, DETAIL * SCALE * 2, True)
        for index in (0, 5, 13)
    ]

    pictures = fine_pictures(tuple(float(v) for v in matrix.ravel()), look, 2, (0, 5, 13))

    for (image, anchor), sprite in zip(drawn, pictures, strict=True):
        assert np.array_equal(np.asarray(sprite.image), np.asarray(image))
        assert sprite.anchor == anchor


def test_fine_pictures_are_as_many_times_as_big_as_the_sheets() -> None:
    key = tuple(float(v) for v in projection().matrix.ravel())
    sprites = train_sprites(key)
    _, _, width, height, _, _ = sprites.places[((Good.COAL, True), 0)]

    (picture,) = fine_pictures(key, (Good.COAL, True), 4, (0,))

    assert 3.5 * width < picture.image.size[0] < 4.5 * width
    assert 3.5 * height < picture.image.size[1] < 4.5 * height


def boxed(detail: Callable[[Model], None]) -> Model:
    model = Model()
    model.box((-0.2, 0.2), (-0.08, 0.08), (0.0, 0.4), (90, 90, 90))
    detail(model)
    return model


def colours(image: Image.Image) -> set[tuple[int, ...]]:
    return {tuple(c) for c in np.asarray(image)[..., :3].reshape(-1, 3).tolist()}


def reds(image: Image.Image) -> set[tuple[int, ...]]:
    return {(r, g, b) for r, g, b in colours(image) if r > 2 * (g + b) + 40}


def low_plate(model: Model) -> None:
    with model.fine():
        model.panel((-0.18, 0.18), (0.01, 0.05), 0.0806, (250, 0, 0))


def test_a_fine_detail_is_drawn_only_when_the_model_is_drawn_fine() -> None:
    model = boxed(low_plate)

    far, _ = draw_model(model, 0.3, projection().matrix, 4.0)
    near, _ = draw_model(model, 0.3, projection().matrix, 4.0, fine=True)

    assert not reds(far)
    assert reds(near)


def test_a_fine_detail_low_on_a_tall_side_is_painted_over_that_side() -> None:
    model = boxed(low_plate)

    for heading in (0.3, 1.9, 3.4, 5.0):
        near, _ = draw_model(model, heading, projection().matrix, 4.0, fine=True)

        assert reds(near)


def test_a_coarse_face_stands_in_only_far_out() -> None:
    def stand_in(model: Model) -> None:
        with model.coarse():
            model.box((-0.1, 0.1), (-0.05, 0.05), (0.4, 0.5), (0, 0, 250))

    model = boxed(stand_in)
    blue = {shade((0, 0, 250), level) for level in SHADES}

    far, _ = draw_model(model, 0.3, projection().matrix, 4.0)
    near, _ = draw_model(model, 0.3, projection().matrix, 4.0, fine=True)

    assert blue & colours(far)
    assert not blue & colours(near)


def test_a_model_built_without_fine_details_has_only_the_rest() -> None:
    make = MODELS[(Look.ENGINE, "pacific")]
    full = make()

    with without_fine():
        bare = make()

    assert not any(face.fine for face in bare.faces)
    assert bare.faces == [face for face in full.faces if not face.fine]


@pytest.mark.parametrize("good", list(Good), ids=[good.name for good in Good])
def test_every_full_car_carries_more_than_a_bare_frame(good: Good) -> None:
    frame = Model()
    frame.box((-0.24, 0.24), (-0.07, 0.07), (0.035, 0.06), (0, 0, 0))
    frame.wheels([0.17, -0.17], 0.035, (0, 0, 0), 0.075)

    full = MODELS[(good, True)]()

    assert len([face for face in full.faces if not face.fine]) > len(
        [face for face in frame.faces if not face.fine]
    )
