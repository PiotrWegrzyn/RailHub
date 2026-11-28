from dataclasses import replace

import numpy as np
import pytest
from growing import only

from builders import flat_map
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.events import orders_ended
from railroads.sim.growth import Growth
from railroads.sim.ladders import LADDERS
from railroads.sim.tiles import SUPPLY_ORDERS, Tile, of_year

(MINES,) = [ladder for ladder in LADDERS if ladder.kinds[0] == BuildingKind.MINE]
UNLIKELY_MINES = replace(MINES, base=1e-9, from_year=0)


def mines_after_a_year(start_year: int, orders_share: float = 1.0) -> int:
    growth = Growth(
        Constructions(flat_map()),
        replace(only(UNLIKELY_MINES), orders_share=orders_share),
        start_year=start_year,
    )

    for _ in range(12):
        growth.month(np.random.default_rng(0))

    return int(growth.count(BuildingKind.MINE).sum())


def test_coal_goes_up_a_level_in_1850_as_the_world_grows_by_itself() -> None:
    before, after = of_year(1849).supply[Tile.COAL], of_year(1850).supply[Tile.COAL]

    assert after == before + 1


@pytest.mark.parametrize(("start_year", "mines"), [(1849, 1), (1850, 0)])
def test_a_coal_supply_level_puts_up_a_mine_where_none_would_come_up_on_its_own(
    start_year: int, mines: int
) -> None:
    grown = mines_after_a_year(start_year)

    assert grown == mines


def test_land_bought_puts_up_only_its_share_of_what_supply_levels_order() -> None:
    grown = mines_after_a_year(1849, orders_share=0.0)

    assert grown == 0


def test_steel_levels_put_up_an_iron_mine_and_a_steel_mill_in_turn() -> None:
    orders = SUPPLY_ORDERS[Tile.STEEL]

    lots = [orders.at(level)[: orders.each] for level in (11, 12, 13)]

    assert lots == [
        [(BuildingKind.IRON_MINE,)],
        [(BuildingKind.STEEL_MILL,)],
        [(BuildingKind.IRON_MINE,)],
    ]


def test_a_food_level_starts_where_the_one_before_ended() -> None:
    orders = SUPPLY_ORDERS[Tile.FOOD]

    first, second = orders.at(1)[:2], orders.at(2)[:2]

    assert orders.at(1)[2:4] == second
    assert first != second


def test_the_internet_ends_the_printing_works_mail_levels_put_up() -> None:
    before = orders_ended(BuildingKind.PRINTING_WORKS, 1994)

    after = orders_ended(BuildingKind.PRINTING_WORKS, 1995)

    assert (before, after) == (False, True)


def test_a_farming_level_puts_up_two_farms_drawn_from_every_kind() -> None:
    orders = SUPPLY_ORDERS[Tile.FARMING]

    lots = orders.at(11)[: orders.each]

    assert len(lots) == 2
    assert BuildingKind.FARM in lots[0] and BuildingKind.VINEYARD in lots[1]
