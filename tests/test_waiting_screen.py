import numpy as np

from railroads.ui.waiting_screen import (
    HEX_RADIUS,
    Pieces,
)


def test_the_hexes_cover_the_whole_screen() -> None:
    pieces = Pieces.covering(800, 600, np.random.default_rng(1))
    x, y = (grid.ravel() for grid in np.meshgrid(np.linspace(0, 800, 81), np.linspace(0, 600, 61)))

    nearest = np.hypot(x[:, None] - pieces.x[None, :], y[:, None] - pieces.y[None, :]).min(axis=1)

    assert nearest.max() <= HEX_RADIUS


def test_the_hexes_hold_still_at_first_and_all_fall_out_of_sight() -> None:
    pieces = Pieces.covering(800, 600, np.random.default_rng(1))

    x, y, turn = pieces.at(0.0)

    assert np.array_equal(x, pieces.x) and np.array_equal(y, pieces.y)
    assert not turn.any()
    assert not pieces.gone(0.5, 600)
    assert pieces.gone(3.0, 600)
