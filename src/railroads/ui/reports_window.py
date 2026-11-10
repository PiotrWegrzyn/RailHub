"""The reports window: the cars delivered each year, stacked by what they carried, the stations
the player had each year and the population of the whole map (railroads.sim.stats).

The year under way shows as it stands so far, with today's stations and population.
"""

from dataclasses import dataclass

import numpy as np
from imgui_bundle import imgui, implot

from railroads.sim.goods import Good
from railroads.sim.stats import History
from railroads.ui.theme import GOOD_COLORS

WINDOW_SIZE = (720.0, 640.0)
WINDOW_TOP = 56.0
PLOT_HEIGHT = 180.0
BAR_WIDTH = 0.8
GOODS_COLORMAP = "Goods"
GOODS = list(Good)


@dataclass(frozen=True)
class Report:
    # The years with deliveries, from the first to this one, and the cars of each good in each
    # (goods by years), for the goods ever delivered.
    delivery_years: list[int]
    goods: list[Good]
    cars: np.ndarray
    # The years the stations and population are known for, ending with this one.
    years: list[int]
    stations: list[int]
    population: list[int]


def report(history: History, year: int, stations: int, population: int) -> Report:
    """What the reports show in `year`, with the stations and population standing now."""
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
        [known[y][0] for y in years],
        [known[y][1] for y in years],
    )


def _goods_colormap() -> implot.Colormap:
    """A colormap with every good's colour, in the order of Good, added once per plot context."""
    found = implot.get_colormap_index(GOODS_COLORMAP)
    if found != -1:
        return found
    colors = np.array([GOOD_COLORS[good] for good in GOODS], dtype=np.float32)
    return implot.add_colormap(GOODS_COLORMAP, colors, qual=True)


class ReportsWindow:
    def __init__(self) -> None:
        self.open = False

    def draw(self, history: History, year: int, stations: int, population: int) -> None:
        if not self.open:
            return
        imgui.set_next_window_pos((60.0, WINDOW_TOP), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, self.open = imgui.begin("Reports", self.open)
        if expanded:
            shown = report(history, year, stations, population)
            self._draw_deliveries(shown)
            self._draw_line("Stations owned", "stations", shown.years, shown.stations)
            self._draw_line("Population", "population", shown.years, shown.population)
        imgui.end()

    def _draw_deliveries(self, shown: Report) -> None:
        imgui.text("Cars delivered, by what they carried")
        if not implot.begin_plot("##deliveries", (-1.0, PLOT_HEIGHT * 1.4)):
            return
        flags = implot.AxisFlags_.auto_fit.value
        implot.setup_axes("", "cars", flags, flags)
        implot.setup_axis_format(implot.ImAxis_.x1.value, "%.0f")
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
        if not implot.begin_plot(f"##{key}", (-1.0, PLOT_HEIGHT)):
            return
        flags = implot.AxisFlags_.auto_fit.value
        implot.setup_axes("", "", flags, flags)
        implot.setup_axis_format(implot.ImAxis_.x1.value, "%.0f")
        xs = np.array(years, dtype=np.float64)
        ys = np.array(values, dtype=np.float64)
        implot.plot_stairs(title, xs, ys, implot.Spec(flags=implot.ItemFlags_.no_legend.value))
        implot.end_plot()
