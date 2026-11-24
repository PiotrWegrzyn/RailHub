from typing import Any

import pytest

from builders import make_game
from railroads import newgame
from railroads.saves import SavedGame, Saves
from railroads.settings import MapChoice, Settings
from railroads.sim.company import PRICE_FACTORS, START_CASHES, Difficulty
from railroads.sim.constructions import Constructions
from railroads.sim.goods import Good
from railroads.sim.steps import Steps
from railroads.sim.tiles import TILE_GOODS, Board, Tile, Tiles
from railroads.ui.main_menu import MainMenu, difficulty_text


def quick_game(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> SavedGame:
    def quick_growth(
        tile_map: Any, rng: Any, months: int, history: Any, progress: Any, look: Any
    ) -> Steps[Constructions]:
        yield
        return Constructions(tile_map)

    monkeypatch.setattr(newgame, "seed_steps", quick_growth)
    return newgame.make_game(settings, report=lambda _: None)


def test_easy_doubles_prices_and_starts_with_150k_and_hard_keeps_prices_with_80k() -> None:
    rules = {d: (PRICE_FACTORS[d], START_CASHES[d]) for d in Difficulty}

    assert rules == {
        Difficulty.EASY: (2.0, 150_000.0),
        Difficulty.NORMAL: (1.5, 100_000.0),
        Difficulty.HARD: (1.0, 80_000.0),
    }


def test_marked_up_boards_price_every_good_that_many_times_in_years_so_far_and_to_come() -> None:
    plain = Tiles.until(1901)
    tiles = Tiles.until(1900)

    tiles.mark_up(2.0)
    tiles.turn(1901)

    for year in (1900, 1901):
        for good in Good:
            assert tiles.of_year(year).price(good) == pytest.approx(
                2.0 * plain.of_year(year).price(good)
            )


def test_marked_up_boards_show_picks_at_the_marked_up_price() -> None:
    plain = Tiles.until(1900)
    tiles = Tiles.until(1900)
    tile = Tile.COAL

    tiles.mark_up(1.5)

    assert tiles.planned_price(tile) == pytest.approx(1.5 * plain.planned_price(tile))
    assert tiles.price_after(Board.DEMAND, tile) == pytest.approx(
        1.5 * plain.price_after(Board.DEMAND, tile)
    )


def test_the_railway_is_paid_the_marked_up_price() -> None:
    game = make_game()
    good = TILE_GOODS[Tile.COAL][0]
    plain = game.railway.market_price(good), game.railway.price_factor(good)

    game.railway.tiles.mark_up(2.0)

    assert game.railway.market_price(good) == pytest.approx(2.0 * plain[0])
    assert game.railway.price_factor(good) == pytest.approx(2.0 * plain[1])


@pytest.mark.parametrize("difficulty", list(Difficulty))
def test_a_new_game_starts_with_its_difficultys_cash_and_prices(
    monkeypatch: pytest.MonkeyPatch, difficulty: Difficulty
) -> None:
    settings = Settings(seed=1, map=MapChoice.POLAND, difficulty=difficulty)

    saved = quick_game(monkeypatch, settings)

    assert saved.game.finances.cash == START_CASHES[difficulty]
    assert saved.game.railway.tiles.markup == PRICE_FACTORS[difficulty]


def test_cash_set_for_a_new_game_wins_over_the_difficultys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(seed=1, map=MapChoice.POLAND, cash=2_000_000.0, difficulty=Difficulty.HARD)

    saved = quick_game(monkeypatch, settings)

    assert saved.game.finances.cash == 2_000_000.0


def test_picking_a_difficulty_in_the_menu_sets_its_starting_cash(tmp_path: Any) -> None:
    menu = MainMenu(Saves(tmp_path))

    menu.choose_difficulty(Difficulty.EASY)
    settings = menu.settings()

    assert settings.difficulty == Difficulty.EASY
    assert settings.cash == 150_000.0


def test_the_menu_starts_on_normal_with_its_cash(tmp_path: Any) -> None:
    menu = MainMenu(Saves(tmp_path))

    settings = menu.settings()

    assert (settings.difficulty, settings.cash) == (Difficulty.NORMAL, 100_000.0)


@pytest.mark.parametrize(
    ("difficulty", "shown"),
    [
        (Difficulty.EASY, ["$150,000", "2 times their usual prices"]),
        (Difficulty.NORMAL, ["$100,000", "1.5 times their usual prices"]),
        (Difficulty.HARD, ["$80,000", "their usual prices"]),
    ],
)
def test_the_new_game_page_tells_each_difficultys_cash_and_prices(
    difficulty: Difficulty, shown: list[str]
) -> None:
    text = difficulty_text(difficulty)

    assert all(part in text for part in shown)
