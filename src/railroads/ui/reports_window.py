"""The reports window: the company's worth each month against its target (railroads.sim.company),
the cars delivered each year, stacked by what they carried, the stations
the player had each year and the population of the whole map (railroads.sim.stats), each fuel's
price (railroads.sim.fuel_market) and what goods earn on the market (railroads.sim.goods_market).

The year under way shows as it stands so far, with today's stations and population.
"""

from dataclasses import dataclass

import numpy as np
from imgui_bundle import imgui, implot

from railroads.sim.clock import END_YEAR
from railroads.sim.company import Goal
from railroads.sim.finance import Finances, dollars
from railroads.sim.fuel_market import Fuel, FuelMarket, FuelMonth, electrified
from railroads.sim.goods import Good
from railroads.sim.goods_market import GoodsMarket, NeedMonth
from railroads.sim.stats import History
from railroads.ui.ornate import begin_ornate, end_ornate
from railroads.ui.theme import GOOD_COLORS, GROWN, WINDOW_TOP

WINDOW_SIZE = (720.0, 640.0)
PLOT_HEIGHT = 180.0
BAR_WIDTH = 0.8
GOODS_COLORMAP = "Goods"
GOODS = list(Good)
MILLION = 1_000_000.0
# How many of the most wanted needs the markets tab draws over the years.
MARKET_LINES = 6


@dataclass(frozen=True)
class Report:
    # The years with deliveries, from the first to this one, and the cars of each good in each
    # (goods by years), for the goods ever delivered.
    delivery_years: list[int]
    goods: list[Good]
    cars: np.ndarray
    # The years the population is known for, ending with this one, and the stations owned in
    # those of them since play began.
    years: list[int]
    stations: list[int]
    population: list[int]
    station_years: list[int]


def report(
    history: History, year: int, stations: int, population: int, begun: int | None = None
) -> Report:
    """What the reports show in `year`, with the stations and population standing now; the
    stations only from `begun`, the year play began, as there were none before."""
    delivered = {y: cars for y, cars in history.delivered.items() if y <= year}
    first = min(delivered, default=year)
    delivery_years = list(range(first, year + 1))
    goods = [g for g in GOODS if any(cars[g] for cars in delivered.values())]
    cars = np.array(
        [[delivered.get(y, {}).get(good, 0) for y in delivery_years] for good in goods],
        dtype=np.float64,
    ).reshape(len(goods), len(delivery_years))
    known = {y: (history.stations[y], history.population[y]) for y in history.stations if y < year}
    known[year] = (stations, population)
    years = sorted(known)
    return Report(
        delivery_years,
        goods,
        cars,
        years,
        [known[y][0] for y in years if begun is None or y >= begun],
        [known[y][1] for y in years],
        [y for y in years if begun is None or y >= begun],
    )


def _goods_colormap() -> implot.Colormap:
    """A colormap with every good's colour, in the order of Good, added once per plot context."""
    found = implot.get_colormap_index(GOODS_COLORMAP)
    if found != -1:
        return found
    colors = np.array([GOOD_COLORS[good] for good in GOODS], dtype=np.float32)
    return implot.add_colormap(GOODS_COLORMAP, colors, qual=True)


# The lengths the books can be shown over, in months, and what each is called.
PERIODS = {1: "Monthly", 3: "Quarterly", 12: "Yearly"}
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
PROFIT = (0.86, 0.66, 0.32, 1.0)
LOSS = (0.92, 0.46, 0.36, 1.0)


def period_name(year: int, index: int, length: int) -> str:
    """What a period of `length` months is called: Mar 1851, Q1 1851 or 1851."""
    if length == 1:
        return f"{MONTH_NAMES[index]} {year}"
    if length == 3:
        return f"Q{index + 1} {year}"
    return str(year)


def books_rows(before: dict[str, float], now: dict[str, float]) -> list[tuple[str, float, float]]:
    """What money went in and out on, in the period before and this one so far: costs first,
    the dearest at the top, then income, the biggest first."""
    rows = [(what, before.get(what, 0.0), now.get(what, 0.0)) for what in {*before, *now}]
    return sorted(rows, key=lambda row: (row[1] + row[2] >= 0, abs(row[1] + row[2]) * -1))


def profits(
    periods: list[tuple[int, int, dict[str, float]]], length: int
) -> tuple[list[float], list[float]]:
    """Each period's profit, placed at its start as years with months as twelfths of them."""
    starts = [year + index * length / 12 for year, index, _ in periods]
    return starts, [sum(books.values()) for _, _, books in periods]


def price_line(months: list[FuelMonth]) -> tuple[list[float], list[float]]:
    """A fuel's price month by month, as years with the months as twelfths of them, and prices."""
    xs, seen = [], {}
    for month in months:
        xs.append(month.year + seen.get(month.year, 0) / 12)
        seen[month.year] = seen.get(month.year, 0) + 1
    return xs, [month.price for month in months]


def fuel_breakdown(fuel: Fuel, month: FuelMonth) -> list[tuple[str, str]]:
    """What moved a fuel's price in a month, as labels and values."""
    rows = [
        (
            "Price",
            f"{dollars(month.price)} a unit" if month.price >= 1 else f"${month.price:.2f} a unit",
        ),
    ]
    if fuel == Fuel.ELECTRICITY:
        demand, supply = month.buildings + month.trains, month.network + month.natural
        rows += [
            ("Homes and factories electrified", f"{electrified(month.year):.0%}"),
            ("Wanted by them", f"{month.buildings:,.0f}"),
            ("Used by your trains", f"{month.trains:,.0f}"),
            ("Made by the map's power plants", f"{month.network:,.0f}"),
            ("Fuel those plants got", f"{month.plants_fed:.0%}"),
            ("Of it by your own plants", f"{month.own_power:,.0f}"),
            ("From the world around", f"{month.natural:,.0f}"),
            ("Demand against supply", f"{demand:,.0f} / {supply:,.0f}"),
        ]
        return rows
    demand, supply = month.buildings + month.trains, month.network + month.natural
    rows += [
        ("Wanted by buildings", f"{month.buildings:,.0f}"),
        ("Bought by your trains", f"{month.trains:,.0f}"),
        ("Handed out by your stations", f"{month.network:,.0f}"),
        ("From the world around", f"{month.natural:,.0f}"),
        ("Demand against supply", f"{demand:,.0f} / {supply:,.0f}"),
        (
            "Against the first month",
            f"x{month.pull / month.first_pull:.2f}" if month.first_pull else "-",
        ),
    ]
    return rows


def _begin_year_plot(plot_id: str, height: float, unit: str) -> bool:
    """Starts a plot the window's width over the years, both axes fitted to what it shows;
    whether it is drawn (then end it with implot.end_plot)."""
    if not implot.begin_plot(plot_id, (-1.0, height)):
        return False
    flags = implot.AxisFlags_.auto_fit.value
    implot.setup_axes("", unit, flags, flags)
    implot.setup_axis_format(implot.ImAxis_.x1.value, "%.0f")
    return True


def _draw_fuel(market: FuelMarket) -> None:
    """Each fuel's price over the years, what moved it last month, and what the trains bought of
    it this month."""
    opened = [fuel for fuel in Fuel if market.history[fuel]]
    if not opened:
        imgui.text_disabled("Fuel prices start after the first month")
        return
    imgui.text("Fuel prices, a unit")
    if _begin_year_plot("##fuel prices", PLOT_HEIGHT * 1.4, "$"):
        for fuel in opened:
            xs, ys = price_line(market.history[fuel])
            implot.plot_line(
                fuel.value.capitalize(),
                np.array(xs, dtype=np.float64),
                np.array(ys, dtype=np.float64),
            )
        implot.end_plot()
    for fuel in opened:
        imgui.separator_text(fuel.value.capitalize())
        last = market.history[fuel][-1]
        if imgui.begin_table(f"fuel {fuel.name}", 2):
            for label, value in fuel_breakdown(fuel, last):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(label)
                imgui.table_next_column()
                imgui.text(value)
            imgui.end_table()
        imgui.text(
            f"Your trains this month: {market.bought(fuel):,.0f} "
            f"({market.from_stations[fuel]:,.0f} at the market price"
            f"{'' if fuel == Fuel.ELECTRICITY else ' from stations'}), "
            f"{dollars(market.spent[fuel])}; last month {market.last_bought[fuel]:,.0f}"
        )


def market_line(months: list[NeedMonth]) -> tuple[list[float], list[float]]:
    """A need's price, as a share of the prices goods list, month by month, as years with the
    months as twelfths of them."""
    xs, seen = [], {}
    for month in months:
        xs.append(month.year + seen.get(month.year, 0) / 12)
        seen[month.year] = seen.get(month.year, 0) + 1
    return xs, [100.0 * month.price for month in months]


def market_rows(market: GoodsMarket) -> list[tuple[str, str, str, str]]:
    """Every need on the market, the most wanted first: its name, what its goods earn as a share
    of their prices, and what the map wanted and could make of it last month."""
    rows = []
    for need, months in sorted(market.history.items(), key=lambda item: -item[1][-1].wanted):
        last = months[-1]
        rows.append(
            (
                need.value.capitalize(),
                f"{last.price:.0%}",
                f"{last.wanted:,.0f}",
                f"{last.made:,.0f}",
            )
        )
    return rows


def _draw_markets(market: GoodsMarket) -> None:
    """What goods earn on the market over the years, against the prices they list, the most
    wanted needs drawn, and every need's market last month."""
    if not market.history:
        imgui.text_disabled("Markets open after the first month")
        return
    imgui.text_wrapped(
        "What goods earn against the prices they list: more where the map wants more of them "
        "than it makes, less where it makes more than it wants."
    )
    wanted = sorted(market.history.items(), key=lambda item: -item[1][-1].wanted)
    if _begin_year_plot("##markets", PLOT_HEIGHT * 1.4, "%"):
        for need, months in wanted[:MARKET_LINES]:
            xs, ys = market_line(months)
            implot.plot_line(
                need.value.capitalize(),
                np.array(xs, dtype=np.float64),
                np.array(ys, dtype=np.float64),
            )
        implot.end_plot()
    flags = imgui.TableFlags_.row_bg.value
    if imgui.begin_table("markets", 4, flags):
        for heading in ("Need", "Price", "Wanted", "Made"):
            imgui.table_setup_column(heading)
        imgui.table_headers_row()
        for row in market_rows(market):
            imgui.table_next_row()
            for value in row:
                imgui.table_next_column()
                imgui.text(value)
        imgui.end_table()


def worth_line(goal: Goal) -> tuple[list[float], list[float]]:
    """The company's worth at the start of each month, by the year and the part of it gone."""
    return [year + month / 12 for year, month, _ in goal.values], [v for _, _, v in goal.values]


def _draw_worth(goal: Goal) -> None:
    """The company's worth each month against its target, in millions."""
    imgui.text(
        f"Company worth, against the {goal.difficulty.value.lower()} target of "
        f"{dollars(goal.target)} for 1 January {END_YEAR}"
    )
    if not goal.values:
        imgui.text_disabled("Shown from the start of the next month")
        return
    if not _begin_year_plot("##worth", PLOT_HEIGHT, "$ million"):
        return
    xs, ys = worth_line(goal)
    implot.setup_axis_format(implot.ImAxis_.y1.value, "%g")
    spec = implot.Spec(flags=implot.ItemFlags_.no_legend.value)
    worth = np.array(ys, dtype=np.float64) / MILLION
    implot.plot_line("Worth", np.array(xs, dtype=np.float64), worth, spec)
    implot.plot_line(
        "Target",
        np.array([xs[0], max(xs[-1], END_YEAR)], dtype=np.float64),
        np.array([goal.target, goal.target], dtype=np.float64) / MILLION,
        implot.Spec(line_color=imgui.ImVec4(*GROWN), flags=implot.ItemFlags_.no_legend.value),
    )
    implot.end_plot()


class ReportsWindow:
    def __init__(self) -> None:
        self.open = False
        # How many months the books are shown over: 1, 3 or 12.
        self.period = 12

    def _draw_books(self, finances: Finances, year: int) -> None:
        """Buttons to show the books by month, quarter or year; a table of what money went in
        and out on, in the period before and this one so far; and each period's profit."""
        for length, name in PERIODS.items():
            if length != 1:
                imgui.same_line()
            if imgui.radio_button(name, self.period == length):
                self.period = length
        periods = finances.periods(self.period, year)
        now = periods[-1]
        before = periods[-2] if len(periods) > 1 else None
        imgui.text("Money in and out")
        rows = books_rows({} if before is None else before[2], now[2])
        if rows and imgui.begin_table("books", 3, imgui.TableFlags_.row_bg.value):
            imgui.table_setup_column("")
            imgui.table_setup_column(
                "" if before is None else period_name(before[0], before[1], self.period)
            )
            imgui.table_setup_column(f"{period_name(now[0], now[1], self.period)} so far")
            imgui.table_headers_row()
            totals = [0.0, 0.0]
            for what, last, current in rows:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(what.capitalize())
                for index, amount in enumerate((last, current)):
                    totals[index] += amount
                    imgui.table_next_column()
                    imgui.text(dollars(amount) if amount else "")
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Profit")
            for amount in totals:
                imgui.table_next_column()
                imgui.text(dollars(amount))
            imgui.end_table()
        else:
            imgui.text_disabled("Nothing yet")
        imgui.spacing()
        self._draw_profits(periods)

    def _draw_profits(self, periods: list[tuple[int, int, dict[str, float]]]) -> None:
        """Each period's profit as a bar, brass above nothing and red below."""
        imgui.text(f"Profit, {PERIODS[self.period].lower()}")
        if not _begin_year_plot("##profit", PLOT_HEIGHT, "$"):
            return
        starts, amounts = profits(periods, self.period)
        xs = np.array(starts, dtype=np.float64) + self.period / 24
        ys = np.array(amounts, dtype=np.float64)
        width = self.period / 12 * BAR_WIDTH
        for name, colour, shown in (
            ("Profit", PROFIT, np.where(ys >= 0, ys, 0.0)),
            ("Loss", LOSS, np.where(ys < 0, ys, 0.0)),
        ):
            implot.plot_bars(
                name,
                xs,
                shown,
                width,
                implot.Spec(
                    fill_color=imgui.ImVec4(*colour), flags=implot.ItemFlags_.no_legend.value
                ),
            )
        implot.end_plot()

    def draw(
        self,
        history: History,
        year: int,
        stations: int,
        population: int,
        begun: int | None = None,
        finances: Finances | None = None,
        fuel: FuelMarket | None = None,
        goal: Goal | None = None,
        market: GoodsMarket | None = None,
    ) -> None:
        if not self.open:
            return
        imgui.set_next_window_pos((60.0, WINDOW_TOP), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, self.open = begin_ornate("Reports")
        if expanded and imgui.begin_tab_bar("reports"):
            if imgui.begin_tab_item("Overview")[0]:
                shown = report(history, year, stations, population, begun)
                if goal is not None:
                    _draw_worth(goal)
                if finances is not None:
                    self._draw_books(finances, year)
                self._draw_deliveries(shown)
                self._draw_line("Stations owned", "stations", shown.station_years, shown.stations)
                self._draw_line("Population", "population", shown.years, shown.population)
                imgui.end_tab_item()
            if fuel is not None and imgui.begin_tab_item("Fuel")[0]:
                _draw_fuel(fuel)
                imgui.end_tab_item()
            if market is not None and imgui.begin_tab_item("Markets")[0]:
                _draw_markets(market)
                imgui.end_tab_item()
            imgui.end_tab_bar()
        end_ornate()

    def _draw_deliveries(self, shown: Report) -> None:
        imgui.text("Cars delivered, by what they carried")
        if not _begin_year_plot("##deliveries", PLOT_HEIGHT * 1.4, "cars"):
            return
        implot.setup_legend(implot.Location_.north_west.value, implot.LegendFlags_.outside.value)
        if shown.goods:
            implot.push_colormap(_goods_colormap())
            labels = [good.value.capitalize() for good in shown.goods]
            # Each good keeps its own colour by drawing every good, those never delivered hidden.
            rows = {good: row for good, row in zip(shown.goods, shown.cars, strict=True)}
            empty = np.zeros(len(shown.delivery_years))
            values = np.concatenate([rows.get(good, empty) for good in GOODS])
            names = [
                labels[shown.goods.index(good)] if good in rows else f"##{good.name}"
                for good in GOODS
            ]
            implot.plot_bar_groups(
                names,
                values,
                group_size=BAR_WIDTH,
                shift=shown.delivery_years[0],
                spec=implot.Spec(flags=implot.BarGroupsFlags_.stacked.value),
            )
            implot.pop_colormap()
        implot.end_plot()

    def _draw_line(self, title: str, key: str, years: list[int], values: list[int]) -> None:
        imgui.text(title)
        if not _begin_year_plot(f"##{key}", PLOT_HEIGHT, ""):
            return
        xs = np.array(years, dtype=np.float64)
        ys = np.array(values, dtype=np.float64)
        implot.plot_stairs(title, xs, ys, implot.Spec(flags=implot.ItemFlags_.no_legend.value))
        implot.end_plot()
