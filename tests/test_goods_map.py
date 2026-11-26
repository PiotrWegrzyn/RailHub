import numpy as np
import pytest
from growing import build

from builders import flat_map
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.goods import Good, Need
from railroads.sim.goods_map import GoodMap, cell_good, good_map, goods_totals, needs_met_by
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.ui.goods_search import GoodTints, good_tints, strength

FARM = (2, 2)
TOWN = (2, 14)
NEAR_FARM = (2, 3)


def traded(*buildings: tuple[tuple[int, int], BuildingKind, int]) -> Growth:
    grid = HexGrid(rows=6, cols=20, km_per_cell=2.0)
    constructions = Constructions(flat_map(grid=grid))
    for cell, kind, count in buildings:
        build(constructions, cell, kind, count)
    return Growth(constructions, GrowthRule(ladders=()))


def flat(growth: Growth, cell: tuple[int, int]) -> int:
    return cell[0] * growth.grid.cols + cell[1]


def wheat(growth: Growth) -> GoodMap:
    assert growth.last_trade is not None
    grid = growth.grid
    return good_map(growth.last_trade, grid.rows * grid.cols, grid.cols, Good.WHEAT)


def test_a_farm_far_from_anyone_is_left_with_all_its_wheat() -> None:
    growth = traded((FARM, BuildingKind.FARM, 1), (TOWN, BuildingKind.HOUSE, 2))

    shown = wheat(growth)

    at = flat(growth, FARM)
    assert shown.made[at] > 0
    assert shown.left[at] == pytest.approx(shown.made[at])
    assert shown.most_left[0].cell == FARM


def test_homes_far_from_any_farm_are_short_of_all_the_food_they_want() -> None:
    growth = traded((FARM, BuildingKind.FARM, 1), (TOWN, BuildingKind.HOUSE, 2))

    shown = wheat(growth)

    at = flat(growth, TOWN)
    assert shown.wanted[at] > 0
    assert shown.short[at] == pytest.approx(shown.wanted[at])
    assert shown.most_short[0].cell == TOWN
    assert shown.most_short[0].need == Need.FOOD


def test_homes_beside_a_farm_get_its_wheat_from_nearby() -> None:
    growth = traded((FARM, BuildingKind.FARM, 1), (NEAR_FARM, BuildingKind.HOUSE, 1))
    assert growth.last_trade is not None

    here = cell_good(growth.last_trade, flat(growth, NEAR_FARM), Good.WHEAT)

    food = next(entry for entry in here.nearby if entry[0] == Need.FOOD)
    _, got, nearby = food
    assert got > 0
    assert nearby == pytest.approx(got)


def test_a_cell_with_nothing_traded_has_no_part_in_a_good() -> None:
    growth = traded((FARM, BuildingKind.FARM, 1))
    assert growth.last_trade is not None

    here = cell_good(growth.last_trade, flat(growth, (5, 19)), Good.WHEAT)

    assert (here.makers, here.buyers, here.nearby) == ((), (), ())


def test_the_maps_totals_add_up_every_cell() -> None:
    growth = traded((FARM, BuildingKind.FARM, 1), (TOWN, BuildingKind.HOUSE, 2))
    assert growth.last_trade is not None
    shown = wheat(growth)

    totals = goods_totals(growth.last_trade)[Good.WHEAT]

    assert totals.made == pytest.approx(shown.made.sum())
    assert totals.left == pytest.approx(shown.left.sum())
    assert totals.wanted == pytest.approx(shown.wanted.sum())
    assert totals.short == pytest.approx(shown.short.sum())


def test_wheat_meets_food() -> None:
    assert Need.FOOD in needs_met_by(Good.WHEAT)


def test_the_lens_scale_runs_two_orders_of_magnitude_down_from_the_most() -> None:
    places = strength(np.array([100.0, 10.0, 1.0, 0.1, 0.0]), 100.0)

    assert places.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0, 0.0])


def test_the_lens_tints_makers_and_buyers_of_a_gap() -> None:
    growth = traded((FARM, BuildingKind.FARM, 1), (TOWN, BuildingKind.HOUSE, 2))

    tints = good_tints(wheat(growth), gaps=True)

    assert isinstance(tints, GoodTints)
    assert set(tints.cells.tolist()) == {flat(growth, FARM), flat(growth, TOWN)}
    assert tints.makers_top > 0
    assert tints.buyers_top > 0
