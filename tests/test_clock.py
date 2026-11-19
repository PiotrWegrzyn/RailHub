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


def test_the_calendar_runs_on_past_the_last_year_play_can_begin_in() -> None:
    clock = GameClock(months=(END_YEAR - START_YEAR) * 12 - 2)

    turned = clock.advance(SECONDS_PER_MONTH * 5)

    assert turned == 5
    assert clock.date == f"April {END_YEAR}"
    assert clock.months_in(SECONDS_PER_MONTH) == 1.0


def test_from_the_usual_start_to_the_last_start_year_is_a_hundred_hours_at_normal_speed() -> None:
    months = (END_YEAR - START_YEAR) * 12

    hours = months * SECONDS_PER_MONTH / 3600

    assert hours == pytest.approx(100)


def test_at_six_times_speed_a_month_takes_a_sixth_of_its_seconds() -> None:
    clock = GameClock(speed=6)

    turned = clock.advance(SECONDS_PER_MONTH / 6)

    assert turned == 1
    assert clock.months_in(SECONDS_PER_MONTH) == 6.0


def test_a_clock_starts_where_the_maps_history_left_off() -> None:
    clock = GameClock(first_year=1950)

    shown = clock.date

    assert shown == "January 1950"


@pytest.mark.parametrize(
    ("speed", "steps", "after"), [(1, 1, 2), (6, 1, 10), (10, 1, 10), (2, -1, 1), (1, -1, 1)]
)
def test_the_speed_steps_up_and_down_its_settings_up_to_ten_times(
    speed: int, steps: int, after: int
) -> None:
    clock = GameClock(speed=speed)

    clock.faster(steps)

    assert clock.speed == after
