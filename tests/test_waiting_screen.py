import numpy as np
import pytest

from railroads.ui.waiting_screen import (
    HEX_RADIUS,
    JOKE_SECONDS,
    WORK,
    WORK_SECONDS,
    Pieces,
    waiting_line,
)


@pytest.mark.parametrize(
    ("seconds", "line"),
    [
        (0.0, "Random bullshit coming!"),
        (JOKE_SECONDS + 0.1, "Lerooorryyy jennnnnkinssss"),
        (2 * JOKE_SECONDS + 0.1, WORK[0]),
        (2 * JOKE_SECONDS + WORK_SECONDS + 0.1, WORK[1]),
    ],
)
def test_a_random_place_opens_with_the_jokes_then_the_work(seconds: float, line: str) -> None:
    shown = waiting_line(seconds, random_place=True)

    assert shown == line


def test_a_ready_made_map_cycles_through_the_work_from_the_start() -> None:
    lines = [waiting_line(n * WORK_SECONDS + 0.1, random_place=False) for n in range(len(WORK) + 1)]

    assert lines == [*WORK, WORK[0]]


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
