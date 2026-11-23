import numpy as np
import pytest

from builders import flat_map
from railroads.newgame import Settings
from railroads.sim.clock import END_YEAR
from railroads.sim.company import (
    BANKRUPT_MONTHS,
    BOOK_SHARE,
    TARGETS,
    CompanyValue,
    Difficulty,
    Goal,
    company_value,
)
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.facilities import FACILITY_PRICES, Facility
from railroads.sim.finance import LEVEL_PRICES, STATION_PRICE, STORAGE_PRICE
from railroads.sim.game import Game
from railroads.sim.routes import plan_route, route_price
from railroads.sim.trains import ENGINES, FIRST_ENGINE


def make_game(cash: float = 10_000_000.0) -> Game:
    game = Game(Constructions(flat_map()))
    game.finances.cash = cash
    return game


def test_a_company_with_nothing_built_is_worth_its_cash() -> None:
    game = make_game(123_456.0)

    value = company_value(game.railway)

    assert value == CompanyValue(cash=123_456.0)
    assert value.total == 123_456.0


def test_debt_takes_from_what_the_company_is_worth() -> None:
    game = make_game()
    game.build_station((5, 5))
    game.finances.cash = -50_000.0

    value = company_value(game.railway)

    assert value.total == pytest.approx(-50_000.0 + STATION_PRICE * BOOK_SHARE)


def test_a_station_counts_for_half_of_what_it_and_its_upgrades_cost() -> None:
    game = make_game()
    game.build_station((5, 5))
    railway = game.railway
    railway.upgrade((5, 5))
    railway.add_storage((5, 5))
    railway.build_facility((5, 5), Facility.ENGINE_SHED)

    value = company_value(railway)

    cost = STATION_PRICE + LEVEL_PRICES[0] + STORAGE_PRICE + FACILITY_PRICES[Facility.ENGINE_SHED]
    assert value.stations == pytest.approx(cost * BOOK_SHARE)
    assert BOOK_SHARE == 0.5


def test_track_and_trains_count_for_half_of_what_they_cost() -> None:
    game = make_game()
    game.build_station((6, 3))
    game.build_station((6, 12))
    route = plan_route(game.constructions, (6, 3), (6, 12))
    assert route is not None
    track_cost = route_price(game.constructions, route)
    game.lay_track((6, 3), (6, 12), planned=route)
    game.railway.buy(FIRST_ENGINE)

    value = company_value(game.railway)

    assert value.track == pytest.approx(track_cost * BOOK_SHARE)
    assert value.trains == pytest.approx(ENGINES[FIRST_ENGINE].price * BOOK_SHARE)


def test_shares_count_at_what_they_would_fetch_today() -> None:
    game = make_game()
    game.growth.build(Building((5, 5), BuildingKind.FARM))
    owners = game.growth.owners
    owners.own((5, 5), BuildingKind.FARM, 300)

    value = company_value(game.railway)

    lot = owners.lot_price((5, 5), BuildingKind.FARM)
    assert lot is not None and lot > 0
    assert value.shares == pytest.approx(3 * lot)


def test_each_difficulty_sets_a_higher_target_than_the_one_before() -> None:
    goals = [Goal(difficulty) for difficulty in Difficulty]

    targets = [goal.target for goal in goals]

    assert targets == [TARGETS[d] for d in Difficulty]
    assert targets == sorted(targets)
    assert len(set(targets)) == 3


def test_a_new_game_is_set_for_the_normal_target_unless_chosen() -> None:
    default = Settings(seed=1)
    chosen = Settings(seed=1, difficulty=Difficulty.HARD)

    goals = Goal(default.difficulty), Goal(chosen.difficulty)

    assert [goal.difficulty for goal in goals] == [Difficulty.NORMAL, Difficulty.HARD]


def test_the_player_is_told_once_when_the_target_is_first_reached() -> None:
    goal = Goal(Difficulty.EASY)
    target = goal.target

    goal.begin_month(1900, 0, CompanyValue(cash=target - 1))
    goal.begin_month(1900, 1, CompanyValue(cash=target))
    goal.begin_month(1900, 2, CompanyValue(cash=target * 2))

    assert goal.reached == "February 1900"
    assert [news.title for news in goal.news] == ["Target reached"]
    assert goal.met is None


@pytest.mark.parametrize(("worth", "met"), [(1.0, True), (0.5, False)])
def test_on_new_year_of_the_end_year_the_target_is_judged_and_play_goes_on(
    worth: float, met: bool
) -> None:
    goal = Goal(Difficulty.NORMAL)

    goal.begin_month(END_YEAR - 1, 11, CompanyValue(cash=goal.target * worth))
    goal.begin_month(END_YEAR, 0, CompanyValue(cash=goal.target * worth))
    goal.begin_month(END_YEAR, 1, CompanyValue(cash=goal.target * worth))

    assert goal.met is met
    assert [n.title for n in goal.news][-1] == ("Target met" if met else "Target missed")
    assert sum(n.title.startswith("Target m") for n in goal.news) == 1
    assert goal.bankrupt is None
    assert len(goal.values) == 3


def test_the_player_is_warned_as_soon_as_the_cash_goes_below_nothing() -> None:
    goal = Goal()

    goal.watch(100.0)
    goal.watch(-1.0)
    goal.watch(-2.0)

    assert [news.title for news in goal.news] == ["Cash below nothing"]
    assert str(BANKRUPT_MONTHS) in goal.news[0].text
    assert goal.months_left == BANKRUPT_MONTHS


def test_each_month_begun_in_debt_brings_bankruptcy_a_month_nearer() -> None:
    goal = Goal()

    for month in range(BANKRUPT_MONTHS - 1):
        goal.begin_month(1850, month, CompanyValue(cash=-1.0))

    assert goal.months_left == 1
    assert goal.bankrupt is None


def test_cash_back_above_nothing_starts_the_countdown_again() -> None:
    goal = Goal()
    goal.watch(-1.0)
    for month in range(BANKRUPT_MONTHS - 1):
        goal.begin_month(1850, month, CompanyValue(cash=-1.0))

    goal.begin_month(1850, 6, CompanyValue(cash=10.0))
    goal.watch(-1.0)

    assert goal.months_left == BANKRUPT_MONTHS
    assert goal.bankrupt is None
    assert [news.title for news in goal.news] == ["Cash below nothing", "Cash below nothing"]


def test_six_months_in_a_row_begun_in_debt_end_the_game() -> None:
    goal = Goal()

    for month in range(BANKRUPT_MONTHS + 2):
        goal.begin_month(1850, month, CompanyValue(cash=-1.0))

    assert goal.bankrupt == "June 1850"
    assert goal.news[-1].ends_game
    assert len(goal.values) == BANKRUPT_MONTHS


def test_a_game_notes_its_worth_each_month_and_goes_bankrupt_in_debt() -> None:
    game = make_game(-1_000.0)
    rng = np.random.default_rng(0)

    for _ in range(BANKRUPT_MONTHS):
        game.month(rng)

    assert [(year, month) for year, month, _ in game.goal.values] == [
        (1850, month) for month in range(1, BANKRUPT_MONTHS + 1)
    ]
    assert game.goal.bankrupt == "July 1850"
