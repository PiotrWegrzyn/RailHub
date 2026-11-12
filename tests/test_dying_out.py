from dataclasses import replace

import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.growth import DIES_AFTER_MONTHS, UNDYING, Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.sandbox import HOUSES_ONLY

TRADING = GrowthRule(trades=True, ladders=(), forests=None)


def standing(*placed: tuple[tuple[int, int], BuildingKind]) -> Growth:
    constructions = Constructions(flat_map(20, 30))
    for cell, kind in placed:
        constructions.build(Building(cell, kind))
    return Growth(constructions, TRADING)


def run(growth: Growth, months: int) -> None:
    growth.run(months, np.random.default_rng(0))


def test_a_house_with_nothing_dies_out_after_thirty_years() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))

    run(growth, DIES_AFTER_MONTHS)

    assert growth.counts[BuildingKind.HOUSE].sum() == 0
    assert growth.died_out == 1


def test_a_house_with_nothing_still_stands_just_short_of_thirty_years() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))

    run(growth, DIES_AFTER_MONTHS - 2)

    assert growth.counts[BuildingKind.HOUSE].sum() == 1


def test_a_farm_and_a_house_near_enough_to_trade_both_live_on() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE), ((5, 9), BuildingKind.FARM))

    run(growth, DIES_AFTER_MONTHS + 12)

    assert growth.counts[BuildingKind.HOUSE].sum() == 1
    assert growth.counts[BuildingKind.FARM].sum() == 1


def test_a_farm_and_a_house_too_far_apart_to_trade_both_die_out() -> None:
    growth = standing(((5, 2), BuildingKind.HOUSE), ((5, 25), BuildingKind.FARM))

    run(growth, DIES_AFTER_MONTHS)

    assert growth.died_out == 2


def test_a_house_left_to_starve_counts_its_thirty_years_afresh_once_fed() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))
    run(growth, DIES_AFTER_MONTHS - 12)
    growth.build(Building((5, 6), BuildingKind.FARM))
    run(growth, 1)
    growth.take_down(Building((5, 6), BuildingKind.FARM))

    run(growth, 24)

    assert growth.counts[BuildingKind.HOUSE].sum() == 1


@pytest.mark.parametrize("kind", [BuildingKind.IRON_MINE, BuildingKind.VINEYARD])
def test_buildings_that_make_a_bonus_good_never_die_out(kind: BuildingKind) -> None:
    growth = standing(((5, 5), kind))

    run(growth, DIES_AFTER_MONTHS + 12)

    assert kind in UNDYING
    assert growth.counts[kind].sum() == 1


def test_the_players_own_buildings_die_out_too() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))
    growth.owners.own((5, 5), BuildingKind.HOUSE)

    run(growth, DIES_AFTER_MONTHS)

    assert growth.counts[BuildingKind.HOUSE].sum() == 0


def test_the_neighbourhood_of_starving_houses_is_not_fed_and_empty_land_counts_as_fed() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))
    cells = np.ravel_multi_index(([5, 5, 15], [5, 6, 20]), (20, 30))

    fed = growth.fed(cells)

    assert fed.tolist() == [0.0, 0.0, 1.0]


def test_houses_next_to_a_farm_are_fed_by_the_share_of_food_they_get() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE), ((5, 6), BuildingKind.FARM))
    cells = np.ravel_multi_index(([5], [5]), (20, 30))

    fed = growth.fed(cells)

    assert fed[0] == pytest.approx(8 * 0.9 / 10)


def test_no_house_goes_up_beside_starving_houses() -> None:
    constructions = Constructions(flat_map(20, 30))
    constructions.build(Building((5, 5), BuildingKind.HOUSE))
    eager = replace(HOUSES_ONLY, base_chance=1.0)
    growth = Growth(constructions, GrowthRule(ladders=(eager,), forests=None))
    around = [(5, 5), *growth.grid.neighbours(5, 5)]

    run(growth, 1)

    assert sum(int(growth.counts[BuildingKind.HOUSE][cell]) for cell in around) == 1
    assert growth.counts[BuildingKind.HOUSE].sum() > 100


def test_of_four_houses_beside_one_farm_three_starve() -> None:
    growth = standing(*[((5, 5), BuildingKind.HOUSE)] * 4, ((5, 6), BuildingKind.FARM))
    assert growth.trade is not None

    starving = growth.starving(growth.trade, BuildingKind.HOUSE)

    assert starving.tolist() == [3, 0]


def test_houses_starving_for_thirty_years_die_and_the_fed_one_stays() -> None:
    growth = standing(*[((5, 5), BuildingKind.HOUSE)] * 4, ((5, 6), BuildingKind.FARM))

    run(growth, DIES_AFTER_MONTHS)

    assert growth.counts[BuildingKind.HOUSE][5, 5] == 1
    assert growth.died_out == 3
