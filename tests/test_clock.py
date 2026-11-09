import pytest

from railroads.sim.clock import END_YEAR, SECONDS_PER_MONTH, START_YEAR, GameClock


def test_a_month_turns_over_after_its_seconds_of_play() -> None:
    clock = GameClock()

    before = clock.advance(SECONDS_PER_MONTH - 1)
    after = clock.advance(2)

    assert (before, after) == (0, 1)
    assert clock.seconds == pytest.approx(1)
    assert clock.date == f"February {START_YEAR}"


def test_a_long_pause_turns_over_several_months_into_the_next_year() -> None:
    clock = GameClock()

    turned = clock.advance(SECONDS_PER_MONTH * 14.5)

    assert turned == 14
    assert clock.date == f"March {START_YEAR + 1}"


def test_the_calendar_stops_in_january_of_the_last_year() -> None:
    clock = GameClock(months=(END_YEAR - START_YEAR) * 12 - 2)

    turned = clock.advance(SECONDS_PER_MONTH * 5)
    after_the_end = clock.advance(SECONDS_PER_MONTH * 5)

    assert (turned, after_the_end) == (2, 0)
    assert clock.finished
    assert clock.date == f"January {END_YEAR}"


def test_a_whole_game_lasts_a_hundred_hours_at_normal_speed() -> None:
    clock = GameClock()

    hours = clock.months_left * SECONDS_PER_MONTH / 3600

    assert hours == pytest.approx(100)


def test_at_six_times_speed_a_month_takes_a_sixth_of_its_seconds() -> None:
    clock = GameClock(speed=6)

    turned = clock.advance(SECONDS_PER_MONTH / 6)

    assert turned == 1
    assert clock.months_in(SECONDS_PER_MONTH) == 6.0
