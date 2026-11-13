import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.growth import DIE_AFTER_MONTHS, HOMES_DIE_AFTER_MONTHS, UNDYING, Growth
from railroads.sim.ladders import GrowthRule

TRADING = GrowthRule(trades=True, ladders=(), forests=None)


def standing(*placed: tuple[tuple[int, int], BuildingKind]) -> Growth:
    constructions = Constructions(flat_map(20, 30))
    for cell, kind in placed:
        constructions.build(Building(cell, kind))
    return Growth(constructions, TRADING)


def run(growth: Growth, months: int) -> None:
    growth.run(months, np.random.default_rng(0))


def test_a_house_with_nothing_dies_out_after_ten_years() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))

    run(growth, HOMES_DIE_AFTER_MONTHS)

    assert growth.counts[BuildingKind.HOUSE].sum() == 0
    assert growth.died_out == 1


def test_a_house_with_nothing_still_stands_just_short_of_ten_years() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))

    run(growth, HOMES_DIE_AFTER_MONTHS - 2)

    assert growth.counts[BuildingKind.HOUSE].sum() == 1


def test_a_farm_and_a_house_near_enough_to_trade_both_live_on() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE), ((5, 9), BuildingKind.FARM))

    run(growth, DIE_AFTER_MONTHS + 12)

    assert growth.counts[BuildingKind.HOUSE].sum() == 1
    assert growth.counts[BuildingKind.FARM].sum() == 1


def test_a_farm_with_no_one_to_sell_to_stands_idle_and_never_dies() -> None:
    growth = standing(((5, 2), BuildingKind.HOUSE), ((5, 25), BuildingKind.FARM))

    run(growth, DIE_AFTER_MONTHS + 12)

    assert growth.counts[BuildingKind.FARM].sum() == 1
    assert growth.died_out == 1


def test_a_house_left_to_starve_counts_its_ten_years_afresh_once_fed() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))
    run(growth, HOMES_DIE_AFTER_MONTHS - 12)
    growth.build(Building((5, 6), BuildingKind.FARM))
    run(growth, 1)
    growth.take_down(Building((5, 6), BuildingKind.FARM))

    run(growth, 24)

    assert growth.counts[BuildingKind.HOUSE].sum() == 1


@pytest.mark.parametrize("kind", [BuildingKind.IRON_MINE, BuildingKind.VINEYARD])
def test_buildings_that_make_a_bonus_good_never_die_out(kind: BuildingKind) -> None:
    growth = standing(((5, 5), kind))

    run(growth, DIE_AFTER_MONTHS + 12)

    assert kind in UNDYING
    assert growth.counts[kind].sum() == 1


def test_the_players_own_buildings_die_out_too() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))
    growth.owners.own((5, 5), BuildingKind.HOUSE)

    run(growth, HOMES_DIE_AFTER_MONTHS)

    assert growth.counts[BuildingKind.HOUSE].sum() == 0


def test_of_four_houses_beside_one_farm_three_starve() -> None:
    growth = standing(*[((5, 5), BuildingKind.HOUSE)] * 4, ((5, 6), BuildingKind.FARM))
    assert growth.trade is not None

    starving = growth.starving(growth.trade, BuildingKind.HOUSE)

    assert starving.tolist() == [3, 0]


def test_houses_starving_for_ten_years_die_and_the_fed_one_stays() -> None:
    growth = standing(*[((5, 5), BuildingKind.HOUSE)] * 4, ((5, 6), BuildingKind.FARM))

    run(growth, HOMES_DIE_AFTER_MONTHS)

    assert growth.counts[BuildingKind.HOUSE][5, 5] == 1
    assert growth.died_out == 3


def test_a_house_the_stand_in_player_serves_does_not_starve() -> None:
    growth = standing(((5, 5), BuildingKind.HOUSE))
    growth.assumed_shares = np.full(20 * 30, np.nan)
    growth.assumed_shares[5 * 30 + 5] = 1.0

    run(growth, HOMES_DIE_AFTER_MONTHS + 12)

    assert growth.counts[BuildingKind.HOUSE][5, 5] == 1


def test_a_steel_mill_with_coal_but_no_iron_is_unsatisfied_however_full_its_stockpile() -> None:
    growth = standing(((5, 5), BuildingKind.STEEL_MILL), ((5, 7), BuildingKind.MINE))
    run(growth, 12)
    assert growth.trade is not None

    mill = int(np.searchsorted(growth.trade.cells, 5 * 30 + 5))

    share = growth.trade.share((BuildingKind.STEEL_MILL,))

    assert share[mill] == 0.0


def test_a_steel_mill_with_nothing_to_make_steel_from_dies_after_thirty_years() -> None:
    growth = standing(((5, 5), BuildingKind.STEEL_MILL))

    run(growth, DIE_AFTER_MONTHS)

    assert growth.counts[BuildingKind.STEEL_MILL].sum() == 0


def test_a_coal_mine_with_nobody_to_sell_to_stands_idle_and_never_dies() -> None:
    growth = standing(((5, 5), BuildingKind.MINE))

    run(growth, DIE_AFTER_MONTHS + 12)

    assert BuildingKind.MINE in UNDYING
    assert growth.counts[BuildingKind.MINE][5, 5] == 1


def test_one_off_buildings_wait_for_what_they_need_however_long_it_takes() -> None:
    growth = standing(((5, 5), BuildingKind.RAILWAY_WORKS))

    run(growth, DIE_AFTER_MONTHS + 12)

    assert BuildingKind.RAILWAY_WORKS in UNDYING
    assert growth.counts[BuildingKind.RAILWAY_WORKS][5, 5] == 1
