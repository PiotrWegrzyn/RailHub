from dataclasses import replace

import numpy as np
import pytest
from test_growth import build, only

from builders import flat_map
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.growth import Growth, inputs_of
from railroads.sim.ladders import CHANCE_UNIT, LADDERS, SUSHI_HUTS

SURE_SUSHI = replace(
    SUSHI_HUTS, base=50.0 / CHANCE_UNIT, needs={}, apart=None, land_cells_each=None
)
(WINERIES,) = [ladder for ladder in LADDERS if ladder.kinds[0] == BuildingKind.WINERY]
SURE_WINERIES = replace(WINERIES, base=50.0 / CHANCE_UNIT, merges={})


def grown(ladder: object, *standing: BuildingKind, months: int = 1) -> int:
    constructions = Constructions(flat_map())
    for k, kind in enumerate(standing):
        build(constructions, (1, 2 + 3 * k), kind, 1)
    growth = Growth(constructions, only(ladder))
    kind = ladder.kinds[0]  # type: ignore[attr-defined]

    for _ in range(months):
        growth.month(np.random.default_rng(0))

    return int(growth.count(kind).sum())


@pytest.mark.parametrize(
    ("standing", "opens"),
    [
        ((), False),
        ((BuildingKind.FISHERY,), False),
        ((BuildingKind.RICE_PADDY,), False),
        ((BuildingKind.FISHERY, BuildingKind.RICE_PADDY), True),
    ],
)
def test_a_sushi_hut_opens_only_once_the_map_has_a_fishery_and_a_rice_paddy(
    standing: tuple[BuildingKind, ...], opens: bool
) -> None:
    huts = grown(SURE_SUSHI, *standing)

    assert (huts > 0) == opens


@pytest.mark.parametrize(("standing", "opens"), [((), False), ((BuildingKind.VINEYARD,), True)])
def test_a_winery_opens_only_once_a_vineyard_stands_anywhere_on_the_map(
    standing: tuple[BuildingKind, ...], opens: bool
) -> None:
    wineries = grown(SURE_WINERIES, *standing)

    assert (wineries > 0) == opens


def test_a_works_whose_glass_plastic_can_stand_in_for_is_met_by_either_maker() -> None:
    before = inputs_of(BuildingKind.JUICE_WORKS, 1900)

    after = inputs_of(BuildingKind.JUICE_WORKS, 1960)

    assert BuildingKind.PLASTICS_WORKS not in before[1]
    assert {BuildingKind.GLASSWORKS, BuildingKind.PLASTICS_WORKS} <= after[1]


def test_a_boost_is_no_main_input() -> None:
    inputs = inputs_of(BuildingKind.BREWERY, 1850)

    assert inputs == (frozenset({BuildingKind.HOP_GARDEN}),)


def test_buildings_that_take_nothing_in_need_nothing_on_the_map() -> None:
    inputs = inputs_of(BuildingKind.HOUSE, 1850)

    assert inputs == ()


@pytest.mark.parametrize(("standing", "topped_up"), [((), 0), ((BuildingKind.VINEYARD,), 1)])
def test_a_yearly_top_up_puts_up_a_works_only_with_its_input_on_the_map(
    standing: tuple[BuildingKind, ...], topped_up: int
) -> None:
    later = replace(WINERIES, from_year=1, merges={})

    wineries = grown(later, *standing, months=12)

    assert wineries == topped_up


(SAWMILLS,) = [ladder for ladder in LADDERS if ladder.kinds[0] == BuildingKind.SAWMILL]


@pytest.mark.parametrize(("standing", "opens"), [((), False), ((BuildingKind.LOGGING_CAMP,), True)])
def test_a_sawmill_opens_only_once_a_logging_camp_fells_logs_on_the_map(
    standing: tuple[BuildingKind, ...], opens: bool
) -> None:
    sawmills = grown(replace(SAWMILLS, base=50.0 / CHANCE_UNIT, merges={}), *standing)

    assert (sawmills > 0) == opens
