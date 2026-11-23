import pytest

from railroads.sim.demand import DemandRecord
from railroads.sim.goods import HANDED_OUT, Good, Need, freight_of
from railroads.sim.trade import SCARCE_WORTH
from railroads.ui.demand_view import coins_for, demand_lines
from railroads.words import dollars


@pytest.mark.parametrize(
    ("share", "coins"), [(1.0, 3), (0.7, 3), (0.5, 2), (0.34, 2), (0.2, 1), (0.05, 1), (0.01, 0)]
)
def test_coins_show_how_much_of_the_demand_is_left(share: float, coins: int) -> None:
    shown = coins_for(share)

    assert shown == coins


def test_the_food_tooltip_prices_every_food_by_its_freight_the_dearest_first() -> None:
    record = DemandRecord(capacity=100.0, left=50.0)

    lines = demand_lines(Need.FOOD, record)

    named = [line.split()[0] for line in lines[2:]]
    assert named[0] == HANDED_OUT[Need.FOOD][0].value.split()[0]
    assert len(lines) == 2 + len(HANDED_OUT[Need.FOOD])
    assert dollars(freight_of(Good.WHEAT)) in next(line for line in lines if "wheat" in line)


def test_the_food_tooltip_says_freight_pays_up_to_the_scarce_worth() -> None:
    record = DemandRecord(capacity=100.0, left=50.0)

    lines = demand_lines(Need.FOOD, record)
    wheat = next(line for line in lines if "wheat" in line)

    assert f"up to {SCARCE_WORTH:g}x" in lines[0]
    assert wheat.endswith(dollars(freight_of(Good.WHEAT) * SCARCE_WORTH))
