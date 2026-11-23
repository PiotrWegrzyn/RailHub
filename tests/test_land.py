import numpy as np
import pytest

from railroads.land import (
    buy_lot_steps,
    lot_middle,
    novelty_chance,
    offered_lots,
    quarter_of_top_ups,
)
from railroads.newgame import make_game
from railroads.saves import SavedGame
from railroads.settings import MapChoice, Settings
from railroads.sim.constructions import BuildingKind
from railroads.sim.lots import START_LOTS
from railroads.sim.novelties import YEARLY_CHANCE
from railroads.sim.steps import finish

SETTINGS = Settings(seed=5, map=MapChoice.POLAND, start_year=1850, cash=2_000_000.0)
LEFT = (-2, 0)


@pytest.fixture(scope="module")
def started() -> SavedGame:
    return make_game(SETTINGS, report=lambda text: None)


@pytest.fixture(scope="module")
def bought(started: SavedGame) -> tuple[SavedGame, dict]:
    game = started.game
    station = next(
        (row, col)
        for row, col in zip(*np.nonzero(game.tile_map.terrain == 0), strict=True)
        if game.build_station((int(row), int(col))).ok
    )
    before = {
        "station": (int(station[0]), int(station[1])),
        "shape": game.tile_map.grid.shape,
        "cash": game.finances.cash,
        "price": game.lots.price(),
        "offered": offered_lots(started),
    }
    saved = finish(buy_lot_steps(started, LEFT, report=lambda text: None))
    return saved, before


def test_a_new_game_offers_the_twelve_lots_around_its_honeycomb_with_land(
    started: SavedGame,
) -> None:
    lots = started.game.lots

    offered = offered_lots(started)

    assert lots is not None
    assert lots.owned == set(START_LOTS)
    assert 9 <= len(offered) <= 12
    assert LEFT in offered


def test_a_lot_bought_is_paid_for_and_widens_the_map(bought: tuple[SavedGame, dict]) -> None:
    saved, before = bought
    game = saved.game
    lots = game.lots

    rows, cols = game.tile_map.grid.shape

    assert lots is not None
    assert game.finances.cash == pytest.approx(before["cash"] - before["price"])
    assert before["price"] == 377_000
    assert lots.price() == 610_000
    assert rows == before["shape"][0]
    assert cols > before["shape"][1]


def test_what_stood_on_the_map_moves_right_by_as_much_as_the_map_widened_on_the_left(
    bought: tuple[SavedGame, dict],
) -> None:
    saved, before = bought
    row, col = before["station"]
    widened = saved.game.tile_map.grid.cols - before["shape"][1]

    stations = saved.game.railway.stations()

    assert stations == [(row, col + widened)]


def test_the_lot_bought_arrives_grown_and_the_lots_not_bought_stay_locked(
    bought: tuple[SavedGame, dict],
) -> None:
    saved, _ = bought
    game = saved.game
    lots = game.lots
    assert lots is not None

    inside = lots.in_lots(lots.window(), {LEFT})
    houses = game.growth.count(BuildingKind.HOUSE)
    locked = game.tile_map.locked

    assert houses[inside].sum() > 0
    assert locked is not None
    assert locked.any()
    assert not locked[inside].any()
    assert lots.lot_at(*lot_middle(lots, LEFT)) == LEFT


def test_the_bought_game_plays_on(bought: tuple[SavedGame, dict]) -> None:
    saved, _ = bought
    rng = np.random.default_rng(2)

    for _ in range(2):
        saved.game.month(rng)

    assert saved.game.growth.population() > 0


def test_land_bought_keeps_a_quarter_of_the_buildings_the_map_keeps_at_least_one_of() -> None:
    every = frozenset().union(*(quarter_of_top_ups(np.random.default_rng(s)) for s in range(60)))

    picked = quarter_of_top_ups(np.random.default_rng(1))

    assert BuildingKind.OIL_WELL in every
    assert BuildingKind.HOUSE not in every
    assert len(picked) == round(len(every) / 4)


def test_a_lot_not_offered_cant_be_bought(started: SavedGame) -> None:
    with pytest.raises(ValueError, match="isn't for sale"):
        finish(buy_lot_steps(started, (7, 7), report=lambda text: None))


@pytest.mark.parametrize("years", [1, 10, 100])
def test_a_novelty_comes_up_in_land_bought_at_a_quarter_of_its_chance_over_the_years_played(
    years: int,
) -> None:
    chance = novelty_chance(years)

    assert chance == pytest.approx((1 - (1 - YEARLY_CHANCE) ** years) / 4)
    assert chance <= 0.25


def test_no_novelty_comes_up_in_land_bought_before_its_era() -> None:
    assert novelty_chance(0) == 0.0
    assert novelty_chance(-5) == 0.0
