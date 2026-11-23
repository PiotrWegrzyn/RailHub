import colorsys
import itertools

import numpy as np
import pytest

from railroads.sim.constructions import BuildingKind
from railroads.sim.grids import SquareGrid
from railroads.sim.towns import Towns
from railroads.sim.trade import Trade
from railroads.ui.map_overlays import (
    HALF_SATISFIED_TINT,
    LITTLE_SATISFIED_TINT,
    MOODS_LEGEND,
    SATISFIED_TINT,
    UNSATISFIED_TINT,
    cell_moods,
    mood_tint,
    tint_colours,
    town_cells,
    with_legend,
)
from railroads.ui.theme import town_tint

HOUSE, FARM, JEWELLER = BuildingKind.HOUSE, BuildingKind.FARM, BuildingKind.JEWELLER


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


def test_each_cell_wanting_anything_is_tinted_by_the_share_of_all_its_buildings_want() -> None:
    trade = houses_and_farms()

    cells, shares, _ = cell_moods(trade)

    assert cells.tolist() == [3, 7, 9]
    assert shares.tolist() == pytest.approx([1.0, 2 / 3, 0.0])


def test_a_small_building_getting_nothing_leaves_a_well_served_cell_green_but_edged() -> None:
    nothing = np.zeros(1)
    trade = Trade(
        np.array([5]),
        got={HOUSE: np.array([95.0]), JEWELLER: nothing},
        needed={HOUSE: np.array([95.0]), JEWELLER: np.array([5.0])},
        gave={HOUSE: nothing, JEWELLER: nothing},
        could_make={HOUSE: nothing, JEWELLER: nothing},
    )

    _, shares, starving = cell_moods(trade)

    assert mood_tint(shares[0]) == HALF_SATISFIED_TINT
    assert starving.tolist() == [True]


def test_only_cells_with_a_building_getting_nothing_are_edged() -> None:
    trade = houses_and_farms()

    cells, _, starving = cell_moods(trade)

    assert cells[starving].tolist() == [9]


def test_cells_whose_buildings_want_nothing_are_not_tinted() -> None:
    nothing = np.zeros(2)
    trade = Trade(
        np.array([4, 5]), {FARM: nothing}, {FARM: nothing}, {FARM: nothing}, {FARM: nothing}
    )

    cells, shares, starving = cell_moods(trade)

    assert cells.size == shares.size == starving.size == 0


def test_no_trade_yet_tints_nothing() -> None:
    cells, shares, starving = cell_moods(None)

    assert cells.size == shares.size == starving.size == 0


def test_an_assumed_share_stands_in_for_the_trade_where_given() -> None:
    trade = houses_and_farms()
    assumed = np.full(16, np.nan)
    assumed[9] = 0.7

    cells, shares, starving = cell_moods(trade, assumed)

    assert cells.tolist() == [3, 7, 9]
    assert shares.tolist() == pytest.approx([1.0, 2 / 3, 0.7])
    assert starving.tolist() == [False, False, False]


def test_the_town_lens_tints_each_town_one_colour_and_leaves_land_without_a_town() -> None:
    towns = Towns(SquareGrid(rows=8, cols=8, km_per_cell=2.0))
    towns.add((1, 1), 1.0)
    towns.add((1, 2), 1.0)
    towns.add((6, 6), 1.0)

    cells, colors = town_cells(towns)

    assert cells.tolist() == [1 * 8 + 1, 1 * 8 + 2, 6 * 8 + 6]
    assert colors[0] == colors[1] != colors[2]


def test_neighbouring_town_numbers_get_unlike_colours() -> None:
    hues = [colorsys.rgb_to_hsv(*town_tint(town)[:3])[0] for town in (1, 2, 3)]

    assert min(abs(a - b) for a, b in itertools.combinations(hues, 2)) > 0.1


def test_tinted_cells_hand_the_graphics_card_their_colours_as_bytes() -> None:
    cells = np.array([1, 5])
    colors = [0x80102030, 0xFF000001]

    tints = tint_colours((2, 3), cells, colors)

    assert tints.shape == (2, 3, 4)
    assert tints[0, 1].tolist() == [0x30, 0x20, 0x10, 0x80]
    assert tints[1, 2].tolist() == [1, 0, 0, 0xFF]
    assert tints[0, 0].tolist() == [0, 0, 0, 0]


def test_a_legend_follows_a_status_without_a_full_stop_as_a_sentence_of_its_own() -> None:
    text = with_legend("Barcin: houses; unsatisfied", MOODS_LEGEND)

    assert text == f"Barcin: houses; unsatisfied. {MOODS_LEGEND}"


def test_a_legend_follows_a_status_ending_in_a_full_stop_after_a_space() -> None:
    text = with_legend("Zoom 5/8.", MOODS_LEGEND)

    assert text == f"Zoom 5/8. {MOODS_LEGEND}"
