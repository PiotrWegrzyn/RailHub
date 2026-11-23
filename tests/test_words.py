import pytest

from railroads.words import counted, either, share_of, times


@pytest.mark.parametrize(("n", "text"), [(7, "seven"), (12, "twelve"), (13, "13"), (1500, "1,500")])
def test_small_counts_are_words_and_big_ones_digits(n: int, text: str) -> None:
    written = counted(n)

    assert written == text


@pytest.mark.parametrize(
    ("share", "text"),
    [
        (0.5, "half"),
        (0.25, "a quarter"),
        (1 / 7, "a seventh"),
        (0.75, "three quarters"),
        (0.35, "35%"),
    ],
)
def test_simple_shares_are_words_and_the_rest_percentages(share: float, text: str) -> None:
    written = share_of(share)

    assert written == text


@pytest.mark.parametrize(
    ("factor", "text"), [(2.0, "twice"), (3, "three times"), (2.5, "2.5 times")]
)
def test_factors_read_as_times(factor: float, text: str) -> None:
    written = times(factor)

    assert written == text


def test_a_choice_joins_its_last_item_with_or() -> None:
    written = either(range(3, 6))

    assert written == "3, 4 or 5"
