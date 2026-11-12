import pytest

from railroads.ui.demand_view import coins_for


@pytest.mark.parametrize(
    ("share", "coins"), [(1.0, 3), (0.7, 3), (0.5, 2), (0.34, 2), (0.2, 1), (0.05, 1), (0.01, 0)]
)
def test_coins_show_how_much_of_the_demand_is_left(share: float, coins: int) -> None:
    shown = coins_for(share)

    assert shown == coins
