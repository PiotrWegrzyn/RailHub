import numpy as np
import pytest

from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Trade
from railroads.ui.map_overlays import (
    HALF_SATISFIED_TINT,
    LITTLE_SATISFIED_TINT,
    SATISFIED_TINT,
    UNSATISFIED_TINT,
    least_satisfied,
    mood_tint,
)

HOUSE, FARM = BuildingKind.HOUSE, BuildingKind.FARM


def houses_and_farms() -> Trade:
    nothing = np.zeros(4)
    return Trade(
        np.array([3, 7, 9, 12]),
        got={HOUSE: np.array([10.0, 5.0, 0.0, 0.0]), FARM: nothing},
        needed={HOUSE: np.array([10.0, 10.0, 10.0, 0.0]), FARM: nothing},
        gave={HOUSE: nothing, FARM: np.array([5.0, 5.0, 0.0, 0.0])},
        could_make={HOUSE: nothing, FARM: np.array([5.0, 5.0, 0.0, 0.0])},
    )


@pytest.mark.parametrize(
    ("share", "tint"),
    [
        (1.45, SATISFIED_TINT),
        (1.0, SATISFIED_TINT),
        (0.99, HALF_SATISFIED_TINT),
        (0.5, HALF_SATISFIED_TINT),
        (0.49, LITTLE_SATISFIED_TINT),
        (0.01, LITTLE_SATISFIED_TINT),
        (0.0, UNSATISFIED_TINT),
    ],
)
def test_a_cell_is_tinted_green_yellow_orange_or_red_by_its_satisfaction(
    share: float, tint: tuple[float, ...]
) -> None:
    chosen = mood_tint(share)

    assert chosen == tint


def test_each_cell_wanting_anything_takes_its_least_satisfied_building() -> None:
    trade = houses_and_farms()

    cells, shares = least_satisfied(trade)

    assert cells.tolist() == [3, 7, 9]
    assert shares.tolist() == [1.0, 0.5, 0.0]


def test_cells_whose_buildings_want_nothing_are_not_tinted() -> None:
    nothing = np.zeros(2)
    trade = Trade(
        np.array([4, 5]), {FARM: nothing}, {FARM: nothing}, {FARM: nothing}, {FARM: nothing}
    )

    cells, shares = least_satisfied(trade)

    assert cells.size == 0
    assert shares.size == 0


def test_no_trade_yet_tints_nothing() -> None:
    cells, shares = least_satisfied(None)

    assert cells.size == 0
    assert shares.size == 0


def test_an_assumed_share_stands_in_for_the_trade_where_given() -> None:
    trade = houses_and_farms()
    assumed = np.full(16, np.nan)
    assumed[9] = 0.7

    cells, shares = least_satisfied(trade, assumed)

    assert cells.tolist() == [3, 7, 9]
    assert shares.tolist() == [1.0, 0.5, 0.7]
