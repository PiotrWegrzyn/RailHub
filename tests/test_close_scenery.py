from dataclasses import replace

import numpy as np
import pytest
from test_pixel_world import make_world

from railroads.ui.pixel.art import MODELS, _scale_of, draw_sheet, fine_sprite, homes, industry
from railroads.ui.pixel.art.built import HEADING, PROJECTION
from railroads.ui.pixel.art.nature import DRAWN_AT, _drawn_at
from railroads.ui.pixel.art.painter import ROOFS, WALLS, drawn_at
from railroads.ui.pixel.layer import CLOSE_FROM, chunk_resolution
from railroads.ui.pixel.models import Model, draw_model
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.world import CLOSE, _smoothly_blown_up

AWNING = (200, 30, 40)


@pytest.fixture(scope="module")
def sheet() -> SpriteSheet:
    image, places = draw_sheet()
    return SpriteSheet.from_image(image, places)


@pytest.mark.parametrize(
    ("pixel_scale", "density", "expected"),
    [
        (1, 2, 2),
        (2, 2, 2),
        (3, 2, 2),
        (CLOSE_FROM, 2, CLOSE),
        (8, 2, CLOSE),
        (6, 1, 2),
        (4, 1, CLOSE),
    ],
)
def test_close_views_draw_chunks_close_where_that_blows_up_evenly(
    pixel_scale: int, density: int, expected: int
) -> None:
    found = chunk_resolution(pixel_scale, density, (1, 2))

    assert found == expected


def test_with_the_depth_test_a_part_standing_out_of_a_big_wall_shows_in_front_of_it() -> None:
    model = Model()
    model.box((-0.3, 0.3), (-0.05, 0.05), (0.0, 0.6), (200, 200, 200))
    model.box((-0.1, 0.1), (0.05, 0.1), (0.05, 0.07), AWNING)
    model.box((-0.1, 0.1), (-0.1, -0.05), (0.05, 0.07), AWNING)

    image, _ = draw_model(model, HEADING, PROJECTION.matrix, 4, depth_test=True)

    pixels = np.asarray(image)[..., :3].reshape(-1, 3)
    reds = (pixels[:, 0] > 120) & (pixels[:, 1] < 60) & (pixels[:, 2] < 70)
    assert reds.any()


def test_up_close_the_ground_is_blended_between_its_pixels_not_blown_up_to_squares() -> None:
    rgb = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    alpha = np.ones((1, 2))

    blended = _smoothly_blown_up(rgb, alpha, 4)

    assert blended.shape == (4, 8, 4)
    assert ((blended[0, :, 0] > 0.05) & (blended[0, :, 0] < 0.95)).any()
    assert (np.diff(blended[0, :, 0]) >= 0).all()


@pytest.mark.parametrize(("fineness", "expected"), [(1, DRAWN_AT), (2, DRAWN_AT), (16, 16)])
def test_trees_are_drawn_as_fine_as_they_are_shown_up_close(fineness: int, expected: int) -> None:
    with drawn_at(fineness):
        found = _drawn_at()

    assert found == expected


def test_close_chunks_stand_buildings_as_their_pixel_art_blown_up(sheet: SpriteSheet) -> None:
    world = make_world(sheet)
    bare = make_world(sheet)
    bare.scenery = replace(
        bare.scenery,
        x=bare.scenery.x[:0],
        y=bare.scenery.y[:0],
        sprite=bare.scenery.sprite[:0],
        lift=bare.scenery.lift[:0],
    )
    across, down = world.projection.chunks
    chunks = [(x, y) for x in range(across) for y in range(down)]

    close = any((world.render(*c, CLOSE) != bare.render(*c, CLOSE)).any() for c in chunks)

    assert close


def test_homes_are_their_painted_pixel_art_and_the_gasworks_its_model() -> None:
    with drawn_at(_scale_of("house_0") * 4):
        painted_house = homes.house(WALLS[0], ROOFS[0])
    with drawn_at(_scale_of("gasworks") * 4):
        painted_gasworks = industry.gasworks()

    house, gasworks = fine_sprite("house_0", 4), fine_sprite("gasworks", 4)

    assert house.image.tobytes() == painted_house.image.tobytes()
    assert gasworks.image.tobytes() != painted_gasworks.image.tobytes()


def test_every_model_can_be_drawn() -> None:
    with drawn_at(2):
        drawn = {name: build() for name, build in MODELS.items()}

    assert all(sprite.image.width > 0 for sprite in drawn.values())
