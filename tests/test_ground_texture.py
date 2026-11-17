import numpy as np
import pytest

from railroads.ui.pixel.ground_texture import GRAIN, PERIOD, STROKES, noise, textured

GRASS = (0.3, 0.6, 0.25)
WATER = (0.2, 0.35, 0.7)


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
