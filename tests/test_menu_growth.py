import pytest

from railroads.ui.main_menu import MENU_EASING, grown_towards


def test_a_menu_shown_for_the_first_time_is_drawn_at_once_at_its_full_height() -> None:
    shown = 0.0

    height = grown_towards(shown, 300.0, 1 / 60)

    assert height == 300.0


def test_an_opened_fold_lengthens_the_menu_a_share_of_the_way_each_frame() -> None:
    shown = 200.0

    height = grown_towards(shown, 300.0, 1 / 60)

    assert height == pytest.approx(200.0 + 100.0 * MENU_EASING / 60)


def test_the_menu_settles_on_its_height_once_within_half_a_pixel() -> None:
    shown = 299.8

    height = grown_towards(shown, 300.0, 1 / 60)

    assert height == 300.0
