from collections.abc import Iterator

import pytest
from imgui_bundle import imgui, implot
from screens import gui_context

from railroads.sim.company import CompanyValue, Goal
from railroads.sim.finance import Finances
from railroads.sim.goods import Good
from railroads.sim.stats import History
from railroads.sim.tiles import Tile, Tiles
from railroads.ui.reports_window import (
    MOST_YEAR_TICKS,
    ReportsWindow,
    books_rows,
    period_name,
    price_rows,
    profits,
    report,
    tile_price_line,
    worth_line,
    worth_unit,
    year_ticks,
)


@pytest.fixture
def frame() -> Iterator[None]:
    with gui_context(plots=True):
        imgui.new_frame()
        yield
        imgui.render()


def made_history() -> History:
    history = History()
    history.deliver(1901, Good.COAL, 3)
    history.deliver(1903, Good.COAL)
    history.deliver(1903, Good.WOOL, 2)
    history.close_year(1901, 1, 100)
    history.close_year(1902, 2, 120)
    return history


def test_the_report_runs_from_the_first_delivery_to_this_year_with_today_last() -> None:
    history = made_history()

    shown = report(history, 1903, stations=4, population=150)
    coal = shown.goods.index(Good.COAL)

    assert shown.delivery_years == [1901, 1902, 1903]
    assert shown.goods == [g for g in Good if g in (Good.COAL, Good.WOOL)]
    assert shown.cars[coal].tolist() == [3, 0, 1]
    assert shown.years == [1901, 1902, 1903]
    assert shown.stations == [1, 2, 4]
    assert shown.population == [100, 120, 150]


def test_stations_owned_start_the_year_play_began_while_population_goes_back_further() -> None:
    history = made_history()

    shown = report(history, 1903, stations=4, population=150, begun=1902)

    assert shown.station_years == [1902, 1903]
    assert shown.stations == [2, 4]
    assert shown.years == [1901, 1902, 1903]
    assert shown.population == [100, 120, 150]


def test_with_nothing_delivered_the_report_holds_only_this_year() -> None:
    shown = report(History(), 1850, stations=0, population=10)

    assert shown.goods == []
    assert shown.cars.shape == (0, 1)
    assert shown.years == [1850]


@pytest.mark.parametrize("history", [made_history(), History()])
def test_the_window_draws_its_charts(frame: None, history: History) -> None:
    window = ReportsWindow()
    window.open = True

    window.draw(history, 1903, 4, 150)

    assert window.open
    assert (implot.get_colormap_index("Goods") != -1) == bool(history.delivered)


def months_of_books() -> Finances:
    finances = Finances()
    for month in range(14):
        finances.earn(100.0, "cargo")
        finances.spend(10.0 * month, "car trips")
        finances.close_month(1850 if month < 11 else 1851)
    finances.earn(50.0, "cargo")
    return finances


@pytest.mark.parametrize(
    ("length", "count", "last"),
    [(1, 15, (1851, 2, 50.0)), (3, 5, (1851, 0, 250.0)), (12, 2, (1851, 0, 250.0))],
)
def test_the_books_add_up_over_months_quarters_or_years_with_the_month_on_the_books_last(
    length: int, count: int, last: tuple[int, int, float]
) -> None:
    finances = months_of_books()

    periods = finances.periods(length, 1851)
    year, index, books = periods[-1]

    assert len(periods) == count
    assert (year, index, books["cargo"]) == last


def test_each_period_is_named_by_its_month_quarter_or_year() -> None:
    names = [period_name(1851, 2, 1), period_name(1851, 2, 3), period_name(1851, 0, 12)]

    assert names == ["Mar 1851", "Q3 1851", "1851"]


def test_profit_is_what_came_in_less_what_went_out_at_the_start_of_each_period() -> None:
    periods = [(1850, 0, {"cargo": 300.0, "car trips": -100.0}), (1850, 1, {"track": -50.0})]

    starts, amounts = profits(periods, 3)

    assert starts == [1850.0, 1850.25]
    assert amounts == [200.0, -50.0]


def test_the_books_list_costs_first_then_income() -> None:
    rows = books_rows([{"cargo": 500.0, "track": -20.0}, {"cargo": 100.0, "trains": -900.0}])

    assert [what for what, _ in rows] == ["trains", "track", "cargo"]


def test_the_books_give_each_period_its_amount_oldest_first() -> None:
    books = [{"cargo": 10.0}, {"cargo": 20.0, "coal": -5.0}, {"coal": -7.0}]

    rows = dict(books_rows(books))

    assert rows == {"coal": [0.0, -5.0, -7.0], "cargo": [10.0, 20.0, 0.0]}


@pytest.mark.parametrize("length", [1, 3, 12])
def test_the_window_draws_the_books_and_profit_for_each_period(frame: None, length: int) -> None:
    window = ReportsWindow()
    window.open = True
    window.period = length

    window.draw(made_history(), 1851, 4, 150, finances=months_of_books())

    assert window.period == length


def test_the_worth_chart_has_a_point_at_the_start_of_every_month() -> None:
    goal = Goal()
    for year, month, cash in (
        (1850, 10, 100.0),
        (1850, 11, 250.0),
        (1851, 0, -50.0),
        (1851, 1, 400.0),
    ):
        goal.begin_month(year, month, CompanyValue(cash=cash))

    xs, ys = worth_line(goal)

    assert xs == pytest.approx([1850 + 10 / 12, 1850 + 11 / 12, 1851.0, 1851 + 1 / 12])
    assert ys == [100.0, 250.0, -50.0, 400.0]


@pytest.mark.parametrize("months", [0, 3])
def test_the_window_draws_the_worth_against_the_target(frame: None, months: int) -> None:
    goal = Goal()
    for month in range(months):
        goal.begin_month(1850, month, CompanyValue(cash=1_000.0 * month))
    window = ReportsWindow()
    window.open = True

    window.draw(made_history(), 1850, 4, 150, goal=goal)

    assert window.open
    assert len(goal.values) == months


def test_the_price_list_shows_every_tile_the_cheapest_first() -> None:
    tiles = Tiles.until(1900)

    rows = price_rows(tiles, 1900)

    assert rows[0] == ("Textiles and household", "98%", "18", "9")
    assert rows[-1][0] == "Chemicals and building"


def test_a_tiles_price_line_runs_year_by_year_from_when_it_came() -> None:
    tiles = Tiles.until(1850)

    xs, ys = tile_price_line(tiles, Tile.STEEL)

    assert xs[0] == 1843.0
    assert xs[-1] == 1850.0
    assert ys[-1] == pytest.approx(100.0)


def test_the_window_draws_the_prices(frame: None) -> None:
    window = ReportsWindow()
    window.open = True

    window.draw(made_history(), 1903, 4, 150, tiles=Tiles.until(1903))

    assert window.open


def test_a_worth_under_a_million_is_charted_in_thousands() -> None:
    scale, unit = worth_unit([45_000.0, 60_067.0])

    assert (scale, unit) == (1000.0, "$ thousand")


def test_a_worth_of_a_million_or_more_is_charted_in_millions() -> None:
    scale, unit = worth_unit([900_000.0, 2_500_000.0])

    assert (scale, unit) == (1_000_000.0, "$ million")


def test_a_plot_over_two_years_names_each_whole_year_once() -> None:
    ticks = year_ticks([1850.0, 1850.5, 1851.25, 1851.9])

    assert ticks == [1850.0, 1851.0, 1852.0]


def test_a_plot_over_two_centuries_names_no_more_than_ten_years() -> None:
    ticks = year_ticks([1850.0, 2050.0])

    assert len(ticks) <= MOST_YEAR_TICKS
    assert ticks[0] == 1850.0
    assert all(tick.is_integer() for tick in ticks)


def test_a_plot_with_nothing_yet_names_no_years() -> None:
    ticks = year_ticks([])

    assert ticks == []
