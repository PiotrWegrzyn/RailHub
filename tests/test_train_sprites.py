import math

import numpy as np

from railroads.sim.economy import Good
from railroads.sim.grids import HexGrid
from railroads.ui.pixel.trains import HEADINGS, MODELS, Look, render, train_sprites
from railroads.ui.pixel.world import PixelProjection


def projection() -> PixelProjection:
    return PixelProjection.of(HexGrid(rows=10, cols=10, km_per_cell=2.0), 0.0, 1.0)


def test_every_engine_and_car_is_drawn_for_every_heading() -> None:
    sprites = train_sprites(tuple(float(v) for v in projection().matrix.ravel()))

    looks = {look for look, _ in sprites.places}

    assert looks == set(MODELS)
    assert len(sprites.places) == len(MODELS) * HEADINGS


def test_a_car_running_east_is_longer_than_it_is_tall_and_outlined() -> None:
    model = MODELS[(Good.GOODS, True)]()

    sprite = render(model, 0.0, projection().matrix)

    pixels = np.asarray(sprite.image)
    solid = pixels[..., 3] > 0
    rows, cols = np.nonzero(solid)
    assert np.ptp(cols) > np.ptp(rows)
    assert tuple(pixels[rows.min(), cols[rows == rows.min()][0], :3]) == (22, 18, 18)


def test_a_full_coal_car_shows_its_load() -> None:
    empty = np.asarray(render(MODELS[(Good.COAL, False)](), 0.0, projection().matrix).image)
    full = np.asarray(render(MODELS[(Good.COAL, True)](), 0.0, projection().matrix).image)

    assert (full[..., 3] > 0).sum() > (empty[..., 3] > 0).sum()


def test_the_nearest_heading_is_picked() -> None:
    sprites = train_sprites(tuple(float(v) for v in projection().matrix.ravel()))

    east = sprites.place(Look.ENGINE, 0.01)
    almost_east = sprites.place(Look.ENGINE, 2 * math.pi - 0.01)

    assert east == almost_east
