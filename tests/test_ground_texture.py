import numpy as np
import pytest

from railroads.ui.pixel.ground_texture import (
    GRAIN,
    PERIOD,
    STROKES,
    noise,
    square_hash,
    textured,
)

GRASS = (0.3, 0.6, 0.25)
WATER = (0.2, 0.35, 0.7)
EARTH = (0.55, 0.42, 0.3)
WHEAT = (0.85, 0.72, 0.34)


@pytest.mark.parametrize("layer", [GRAIN, STROKES])
def test_the_texture_repeats_every_period_so_chunks_meet_without_a_seam(
    layer: tuple[float, float, float],
) -> None:
    x, y = np.meshgrid(np.linspace(0, 20, 37), np.linspace(0, 20, 41))

    here, there = noise(x, y, layer), noise(x + PERIOD, y + PERIOD, layer)

    np.testing.assert_allclose(here, there, atol=1e-6)


def test_grass_is_textured_far_more_than_water() -> None:
    canvas = np.ones((64, 128, 4))
    canvas[:, :64, :3] = GRASS
    canvas[:, 64:, :3] = WATER
    before = canvas.copy()

    textured(canvas, 0, 0, 2)

    change = np.abs(canvas - before)[..., :3]
    assert change[:, :64].std() > 3 * change[:, 64:].std()


def test_the_squares_of_fine_detail_repeat_every_period() -> None:
    i, j = np.meshgrid(np.arange(-8, 40), np.arange(-8, 40))
    size = (0.5, 1.0)

    here = square_hash(i, j, size, 3)
    there = square_hash(i + round(PERIOD / size[0]), j + round(PERIOD / size[1]), size, 3)

    np.testing.assert_array_equal(here, there)


def test_bare_grass_shows_blades_up_close_and_keeps_its_colour_from_afar() -> None:
    close, close_plain, far, far_plain = (np.ones((n, n, 4)) for n in (256, 256, 32, 32))
    for canvas in (close, close_plain, far, far_plain):
        canvas[..., :3] = GRASS

    textured(close, 0, 0, 8, bare=True)
    textured(close_plain, 0, 0, 8)
    textured(far, 0, 0, 1, bare=True)
    textured(far_plain, 0, 0, 1)

    assert np.abs(close - close_plain)[..., 1].mean() > 0.02
    np.testing.assert_allclose(far, far_plain)


def test_flowers_grow_in_grass_and_not_on_earth() -> None:
    grass, earth = np.ones((512, 512, 4)), np.ones((512, 512, 4))
    grass[..., :3] = GRASS
    earth[..., :3] = EARTH

    textured(grass, 0, 0, 8, bare=True)
    textured(earth, 0, 0, 8, bare=True)

    def blooming(canvas: np.ndarray) -> int:
        r, g, b = canvas[..., 0], canvas[..., 1], canvas[..., 2]
        return int(((r > g) & (b > g) | (r > 0.8) & (g > 0.8)).sum())

    assert blooming(grass) > 0
    assert blooming(earth) == 0


def test_ripe_crops_grow_blades_only_where_laid_as_crops() -> None:
    plain, crops = np.ones((64, 64, 4)), np.ones((64, 64, 4))
    plain[..., :3] = WHEAT
    crops[..., :3] = WHEAT

    textured(plain, 0, 0, 8)
    textured(crops, 0, 0, 8, crops=True)

    assert crops[..., 0].std() > 1.2 * plain[..., 0].std()
