import numpy as np
import pytest

from builders import flat_map, make_game
from railroads.sim.anger import Anger, cooled, cooling_year, surcharge
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.demolition import moving_price
from railroads.sim.finance import STATION_PRICE
from railroads.sim.game import Game
from railroads.sim.routes import plan_route
from railroads.ui.reports_window import anger_lines


@pytest.mark.parametrize(
    ("anger", "after"),
    [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (7, 4), (10, 6)],
)
def test_anger_cools_by_the_ceiling_of_log2_of_one_more(anger: int, after: int) -> None:
    result = cooled(anger)

    assert result == after


@pytest.mark.parametrize(("anger", "factor"), [(0, 1.0), (1, 1.5), (2, 2.0), (4, 3.0)])
def test_each_point_of_anger_adds_half_the_price_without_compounding(
    anger: int, factor: float
) -> None:
    result = surcharge(anger)

    assert result == factor


def test_moving_a_house_or_a_farm_costs_five_thousand_times_the_surcharge() -> None:
    house = moving_price(BuildingKind.HOUSE, 2)
    farm = moving_price(BuildingKind.FARM, 0)
    one_off = moving_price(BuildingKind.ST_PETERS, 2)

    assert house == 10_000
    assert farm == 5_000
    assert one_off is None


def test_anger_cools_only_on_the_first_of_january_of_years_divisible_by_five() -> None:
    anger = Anger(3)

    off_year = anger.new_year(1853)
    on_year = anger.new_year(1855)

    assert off_year == 0
    assert on_year == 2
    assert anger.value == 1
    assert cooling_year(1851) == 1855
    assert cooling_year(1855) == 1855


def test_a_station_moves_the_houses_in_its_cell_aside_and_raises_anger_for_each() -> None:
    game = make_game()
    for _ in range(2):
        game.growth.build(Building((5, 5), BuildingKind.HOUSE))
    quote = game.station_quote((5, 5))

    done = game.build_station((5, 5))

    houses = game.growth.count(BuildingKind.HOUSE)
    assert done.ok
    assert quote.moved == 2
    assert game.anger.value == 2
    assert houses.sum() == 2
    assert houses[5, 5] == 0
    assert game.finances.cash == 1_000_000.0 - STATION_PRICE - 10_000


def test_anger_makes_the_next_station_site_dearer() -> None:
    game = make_game()
    game.anger.value = 2
    game.growth.build(Building((5, 5), BuildingKind.HOUSE))

    quote = game.station_quote((5, 5))

    assert quote.site == 10_000
    assert quote.price == STATION_PRICE + 10_000


def test_track_moves_buildings_aside_and_its_quote_counts_them() -> None:
    game = make_game()
    route = plan_route(game.constructions, (6, 3), (6, 9))
    assert route is not None
    for cell in route.cells[1:-1]:
        for _ in range(6):
            game.growth.build(Building(cell, BuildingKind.HOUSE))
    quote = game.track_quote(route)
    houses = int(game.growth.count(BuildingKind.HOUSE).sum())

    done = game.lay_track((6, 3), (6, 9), planned=route)

    assert done.ok
    assert quote.moved == len(route.cells) - 2
    assert game.anger.value == quote.moved
    assert game.growth.count(BuildingKind.HOUSE).sum() == houses


def test_a_new_year_divisible_by_five_cools_anger_and_tells_the_player() -> None:
    game = Game(Constructions(flat_map()), start_year=1854)
    game.anger.value = 3
    rng = np.random.default_rng(0)

    for _ in range(12):
        game.month(rng)

    assert game.anger.value == 1
    assert any("falls from 3 to 1" in news.text for news in game.goal.news)


def test_the_reports_show_the_anger_and_when_it_cools() -> None:
    anger = Anger(3)

    title, text = anger_lines(anger, 1851)

    assert title == "Anger with the company: 3"
    assert "2.5 times" in text
    assert "1 January 1855, to 1" in text
