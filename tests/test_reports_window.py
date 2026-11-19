from collections.abc import Iterator

import pytest
from imgui_bundle import imgui, implot

from railroads.sim.company import CompanyValue, Goal
from railroads.sim.finance import Finances
from railroads.sim.goods import Good, Need
from railroads.sim.goods_market import GoodsMarket, NeedMonth
from railroads.sim.stats import History
from railroads.ui.reports_window import (
    ReportsWindow,
    books_rows,
    market_line,
    market_rows,
    period_name,
    profits,
    report,
    worth_line,
)


@pytest.fixture
def frame() -> Iterator[None]:
    context = imgui.create_context()
    plots = implot.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    imgui.new_frame()
    yield
    imgui.render()
    implot.destroy_context(plots)
    imgui.destroy_context(context)


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

    assert shown.delivery_years == [1901, 1902, 1903]
    assert shown.goods == [g for g in Good if g in (Good.COAL, Good.WOOL)]
    coal = shown.goods.index(Good.COAL)
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

    assert len(periods) == count
    year, index, books = periods[-1]
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
    rows = books_rows({"cargo": 500.0, "track": -20.0}, {"cargo": 100.0, "trains": -900.0})

    assert [what for what, _, _ in rows] == ["trains", "track", "cargo"]


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


def made_market() -> GoodsMarket:
    market = GoodsMarket()
    market.history = {
        Need.HEATING: [
            NeedMonth(Need.HEATING, 1850, 100.0, 300.0, 1.0),
            NeedMonth(Need.HEATING, 1850, 120.0, 300.0, 0.8),
        ],
        Need.FOOD: [
            NeedMonth(Need.FOOD, 1850, 900.0, 800.0, 1.0),
            NeedMonth(Need.FOOD, 1850, 1000.0, 800.0, 1.25),
        ],
    }
    return market


def test_the_markets_list_every_need_the_most_wanted_first() -> None:
    market = made_market()

    rows = market_rows(market)

    assert rows == [("Food", "125%", "1,000", "800"), ("Heating", "80%", "120", "300")]


def test_a_markets_line_runs_month_by_month_in_percent() -> None:
    months = made_market().history[Need.HEATING]

    xs, ys = market_line(months)

    assert xs == pytest.approx([1850.0, 1850 + 1 / 12])
    assert ys == pytest.approx([100.0, 80.0])


@pytest.mark.parametrize("market", [made_market(), GoodsMarket()])
def test_the_window_draws_the_markets(frame: None, market: GoodsMarket) -> None:
    window = ReportsWindow()
    window.open = True

    window.draw(made_history(), 1903, 4, 150, market=market)

    assert window.open
