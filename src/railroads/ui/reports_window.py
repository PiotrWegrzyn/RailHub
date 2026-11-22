"""The reports window: the company's worth each month against its target (railroads.sim.company),
the cars delivered each year, stacked by what they carried, the stations
the player had each year and the population of the whole map (railroads.sim.stats), each fuel's
price (railroads.sim.fuel_market) and what a unit of every tile's goods earns (railroads.sim.tiles).

The year under way shows as it stands so far, with today's stations and population.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from imgui_bundle import imgui, implot

from railroads.sim.clock import END_YEAR
from railroads.sim.company import Goal
from railroads.sim.finance import Finances, dollars
from railroads.sim.fuel_market import Fuel, FuelMarket, FuelMonth, electrified
from railroads.sim.goods import Good
from railroads.sim.stats import History
from railroads.sim.tiles import TILE_GOODS, Tile, Tiles
from railroads.ui.ornate import begin_ornate, end_ornate
from railroads.ui.theme import GOOD_COLORS, GROWN, WINDOW_TOP

WINDOW_SIZE = (720.0, 640.0)
PLOT_HEIGHT = 180.0
BAR_WIDTH = 0.8
GOODS_COLORMAP = "Goods"
GOODS = list(Good)
MILLION = 1_000_000.0
# The most years a plot names along its axis.
MOST_YEAR_TICKS = 10


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


def profit_cells(totals: list[float], had_before: bool) -> list[str]:
    """The profit row's cells for the period before and this one, the first left blank when the
    game has no period before this one to show."""
    return [dollars(totals[0]) if had_before else "", dollars(totals[1])]


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


def year_ticks(years: Sequence[float]) -> list[float]:
    """Where a plot over these years marks its axis: on whole years only, so no year is named
    twice, every year or every few so that no more than MOST_YEAR_TICKS are named."""
    if not years:
        return []
    first, last = math.floor(min(years)), math.ceil(max(years))
    step = (last - first) // MOST_YEAR_TICKS + 1
    return [float(year) for year in range(first, last + 1, step)]


def _begin_year_plot(plot_id: str, height: float, unit: str, years: Sequence[float]) -> bool:
    """Starts a plot the window's width over the `years` it shows, both axes fitted to what it
    shows and the years marked by year_ticks; whether it is drawn (then end it with
    implot.end_plot)."""
    if not implot.begin_plot(plot_id, (-1.0, height)):
        return False
    flags = implot.AxisFlags_.auto_fit.value
    implot.setup_axes("", unit, flags, flags)
    ticks = year_ticks(years)
    if ticks:
        labels = [f"{tick:.0f}" for tick in ticks]
        implot.setup_axis_ticks(implot.ImAxis_.x1.value, ticks, labels)
    return True


def _draw_fuel(market: FuelMarket) -> None:
    """Each fuel's price over the years, what moved it last month, and what the trains bought of
    it this month."""
    opened = [fuel for fuel in Fuel if market.history[fuel]]
    if not opened:
        imgui.text_disabled("Fuel prices start after the first month")
        return
    imgui.text("Fuel prices, a unit")
    lines = {fuel: price_line(market.history[fuel]) for fuel in opened}
    years = [x for xs, _ in lines.values() for x in xs]
    if _begin_year_plot("##fuel prices", PLOT_HEIGHT * 1.4, "$", years):
        for fuel, (xs, ys) in lines.items():
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


def tile_price_line(tiles: Tiles, tile: Tile) -> tuple[list[float], list[float]]:
    """What a unit of a tile's goods earned each year it was on the boards, as a share of its
    base price."""
    xs, ys = [], []
    for year, boards in sorted(tiles.years.items()):
        if tile in boards.made:
            xs.append(float(year))
            ys.append(100.0 * boards.price(TILE_GOODS[tile][0]))
    return xs, ys


def price_rows(tiles: Tiles, year: int) -> list[tuple[str, str, str, str]]:
    """Every tile on the boards in a year, the cheapest first: its name, what a unit of its goods
    earns as a share of its base price, and its supply and demand levels."""
    boards = tiles.of_year(year)
    rows = sorted(boards.supply, key=lambda tile: boards.price(TILE_GOODS[tile][0]))
    return [
        (
            tile.value.capitalize(),
            f"{boards.price(TILE_GOODS[tile][0]):.0%}",
            str(boards.supply[tile]),
            str(boards.demand[tile]),
        )
        for tile in rows
    ]


def _draw_prices(tiles: Tiles, year: int) -> None:
    """What every tile's goods earned over the years against their base prices, and the boards
    now."""
    imgui.text_wrapped(
        "What a unit of every tile's goods earns against its base price: less the higher its "
        "supply, more the higher its demand. Prices change only as tiles go up, on 1 January."
    )
    lines = {tile: tile_price_line(tiles, tile) for tile in tiles.of_year(year).supply}
    years = [x for xs, _ in lines.values() for x in xs]
    if _begin_year_plot("##prices", PLOT_HEIGHT * 1.4, "%", years):
        for tile, (xs, ys) in lines.items():
            implot.plot_line(
                tile.value.capitalize(),
                np.array(xs, dtype=np.float64),
                np.array(ys, dtype=np.float64),
            )
        implot.end_plot()
    flags = imgui.TableFlags_.row_bg.value
    if imgui.begin_table("prices", 4, flags):
        for heading in ("Tile", "Price", "Supply", "Demand"):
            imgui.table_setup_column(heading)
        imgui.table_headers_row()
        for row in price_rows(tiles, year):
            imgui.table_next_row()
            for value in row:
                imgui.table_next_column()
                imgui.text(value)
        imgui.end_table()


def worth_line(goal: Goal) -> tuple[list[float], list[float]]:
    """The company's worth at the start of each month, by the year and the part of it gone."""
    return [year + month / 12 for year, month, _ in goal.values], [v for _, _, v in goal.values]


def _draw_worth(goal: Goal) -> None:
    """The company's worth each month against its target, in millions, or in thousands while it
    is worth less than a million; the chart fits the worth, the target showing once it is
    near."""
    imgui.text(
        f"Company worth, against the {goal.difficulty.value.lower()} target of "
        f"{dollars(goal.target)} for 1 January {END_YEAR}"
    )
    if not goal.values:
        imgui.text_disabled("Shown from the start of the next month")
        return
    xs, ys = worth_line(goal)
    scale, unit = worth_unit(ys)
    if not _begin_year_plot("##worth", PLOT_HEIGHT, unit, xs):
        return
    implot.setup_axis_format(implot.ImAxis_.y1.value, "%g")
    spec = implot.Spec(flags=implot.ItemFlags_.no_legend.value)
    worth = np.array(ys, dtype=np.float64) / scale
    implot.plot_line("Worth", np.array(xs, dtype=np.float64), worth, spec)
    implot.plot_line(
        "Target",
        np.array([xs[0], max(xs[-1], END_YEAR)], dtype=np.float64),
        np.array([goal.target, goal.target], dtype=np.float64) / scale,
        implot.Spec(
            line_color=imgui.ImVec4(*GROWN),
            flags=implot.ItemFlags_.no_legend.value | implot.ItemFlags_.no_fit.value,
        ),
    )
    implot.end_plot()


def worth_unit(values: list[float]) -> tuple[float, str]:
    """What the worth chart divides its values by, and its axis label: millions, or thousands
    while every value is under a million."""
    if max((abs(value) for value in values), default=0.0) < MILLION:
        return 1000.0, "$ thousand"
    return MILLION, "$ million"


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
            for cell in profit_cells(totals, before is not None):
                imgui.table_next_column()
                imgui.text(cell)
            imgui.end_table()
        else:
            imgui.text_disabled("Nothing yet")
        imgui.spacing()
        self._draw_profits(periods)

    def _draw_profits(self, periods: list[tuple[int, int, dict[str, float]]]) -> None:
        """Each period's profit as a bar, brass above nothing and red below."""
        imgui.text(f"Profit, {PERIODS[self.period].lower()}")
        starts, amounts = profits(periods, self.period)
        if not _begin_year_plot("##profit", PLOT_HEIGHT, "$", starts):
            return
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
        tiles: Tiles | None = None,
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
            if tiles is not None and imgui.begin_tab_item("Prices")[0]:
                _draw_prices(tiles, year)
                imgui.end_tab_item()
            imgui.end_tab_bar()
        end_ornate()

    def _draw_deliveries(self, shown: Report) -> None:
        imgui.text("Cars delivered, by what they carried")
        if not _begin_year_plot("##deliveries", PLOT_HEIGHT * 1.4, "cars", shown.delivery_years):
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
        if not _begin_year_plot(f"##{key}", PLOT_HEIGHT, "", years):
            return
        xs = np.array(years, dtype=np.float64)
        ys = np.array(values, dtype=np.float64)
        implot.plot_stairs(title, xs, ys, implot.Spec(flags=implot.ItemFlags_.no_legend.value))
        implot.end_plot()
