from typing import Any

import pytest

from railroads import newgame
from railroads.newgame import EARLIEST_START_YEAR, MapChoice, Settings, allowed_start_year
from railroads.saves import Saves
from railroads.sim.clock import END_YEAR, HISTORY_START, START_YEAR
from railroads.sim.constructions import Constructions
from railroads.ui.app import parse_args, settings_from
from railroads.ui.main_menu import MainMenu


@pytest.mark.parametrize(
    ("wished", "allowed"),
    [
        (HISTORY_START, EARLIEST_START_YEAR),
        (1749, EARLIEST_START_YEAR),
        (1750, 1750),
        (START_YEAR, START_YEAR),
        (2049, 2049),
        (2100, END_YEAR - 1),
    ],
)
def test_play_begins_no_earlier_than_1750(wished: int, allowed: int) -> None:
    year = allowed_start_year(wished)

    assert year == allowed


def test_the_earliest_start_is_1750_and_the_suggested_one_1850() -> None:
    assert (EARLIEST_START_YEAR, START_YEAR) == (1750, 1850)


def test_a_new_game_wished_to_begin_too_early_begins_in_1750(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grown: list[int] = []

    def quick_growth(tile_map: Any, rng: Any, months: int, history: Any) -> Constructions:
        grown.append(months)
        return Constructions(tile_map)

    monkeypatch.setattr(newgame, "seed_buildings", quick_growth)

    saved = newgame.make_game(
        Settings(seed=1, map=MapChoice.POLAND, start_year=1500), report=lambda _: None
    )

    assert grown == [(1750 - HISTORY_START) * 12]
    assert saved.clock.first_year == 1750
    assert saved.game.growth.year == 1750
    assert saved.settings is not None
    assert saved.settings.start_year == 1750


def test_the_menu_never_sets_up_a_game_before_1750(tmp_path: Any) -> None:
    menu = MainMenu(Saves(tmp_path), start_year=1500)

    settings = menu.settings()

    assert settings.start_year == 1750


def test_the_menu_suggests_1850_to_begin_with(tmp_path: Any) -> None:
    menu = MainMenu(Saves(tmp_path))

    settings = menu.settings()

    assert settings.start_year == 1850


def test_the_command_line_refuses_a_history_that_begins_play_before_1750(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--poland", "--years", "899"])

    error = capsys.readouterr().err
    assert "--years must be at least 900" in error
    assert "1850" in error


def test_the_command_line_allows_play_to_begin_in_1750() -> None:
    args = parse_args(["--poland", "--years", "900"])

    settings = settings_from(args)

    assert settings is not None
    assert settings.start_year == 1750


def test_the_command_line_begins_play_in_1850_by_default() -> None:
    args = parse_args(["--poland"])

    settings = settings_from(args)

    assert settings is not None
    assert settings.start_year == 1850
