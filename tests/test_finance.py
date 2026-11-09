import pytest

from railroads.sim.finance import START_CASH, Finances, dollars


def test_a_game_starts_with_a_hundred_thousand() -> None:
    finances = Finances()

    assert finances.cash == START_CASH == 100_000


def test_the_books_keep_this_month_and_last_by_what_money_was_for() -> None:
    finances = Finances()
    finances.spend(15_000, "stations")
    finances.earn(200, "cargo")

    finances.close_month()
    finances.spend(100, "station upkeep")

    assert finances.cash == pytest.approx(START_CASH - 15_000 + 200 - 100)
    assert finances.last_month == {"stations": -15_000, "cargo": 200}
    assert finances.profit == -100


def test_spending_past_the_cash_runs_into_debt_but_cannot_be_afforded() -> None:
    finances = Finances(cash=50)

    affordable = finances.can_afford(100)
    finances.spend(100, "station upkeep")

    assert not affordable
    assert finances.cash == -50


@pytest.mark.parametrize(("amount", "shown"), [(15000, "$15,000"), (-1200.4, "-$1,200"), (0, "$0")])
def test_money_shows_in_whole_dollars(amount: float, shown: str) -> None:
    assert dollars(amount) == shown
