"""The goods tab of the search window and the goods lens: where a good is made and where it is
wanted, from the last month of trade (railroads.sim.goods_map). The tab lists every good with what
was left over and short of on the map; picking one tints the land by it, its makers in amber and
its buyers in blue, by the gaps (what was left that nothing took, and what buyers were short of)
or by the whole amounts made and wanted, and lists the biggest of each to step to. A cell's
details then show what its buildings made, wanted and got of it, and from where."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.kept import Kept
from railroads.sim.goods import Good
from railroads.sim.goods_map import GoodMap, GoodTotals, Spot, cell_good, good_map, goods_totals
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.ui.ornate import section
from railroads.ui.theme import (
    TEXT,
    Rgba,
    color_u32,
    draw_good_icon,
    good_icon,
    rgba_u32,
)

ROW_HEIGHT = 22.0
ROW_ICON = 16.0
FOOTER_HEIGHT = 230.0
# The colours of a good's makers and of its buyers on the lens, and how clear the least and the
# most of either are, by a logarithmic scale over SCALE_SPAN down from the most on the map.
MAKERS = (1.0, 0.68, 0.12)
BUYERS = (0.25, 0.55, 1.0)
LEAST_ALPHA = 0.15
MOST_ALPHA = 0.75
SCALE_SPAN = 100.0
LEGEND_WIDTH = 220.0
LEGEND_BAR = 10.0
# How many of the biggest makers and buyers the tab lists.
LISTED = 5


def amount_text(amount: float) -> str:
    """An amount a month, whole from 100 up, to a tenth below."""
    return f"{amount:,.0f}" if amount >= 100 else f"{amount:,.1f}"


def strength(amounts: npt.NDArray[np.float64], most: float) -> npt.NDArray[np.float64]:
    """Where amounts fall on the lens's scale, 0 at most / SCALE_SPAN or less and 1 at most."""
    if most <= 0:
        return np.zeros_like(amounts)
    places = np.log10(np.maximum(amounts, 1e-12) / most) / math.log10(SCALE_SPAN) + 1
    return np.clip(places, 0.0, 1.0)


def tint(colour: tuple[float, float, float], place: float) -> int:
    alpha = LEAST_ALPHA + (MOST_ALPHA - LEAST_ALPHA) * place
    return rgba_u32((*colour, alpha))


@dataclass(frozen=True, slots=True)
class GoodTints:
    """The goods lens's flat cells, their packed colours, and the most made or left over and the
    most wanted or short of in a cell, its scales' tops."""

    cells: npt.NDArray[np.intp]
    colors: list[int]
    makers_top: float
    buyers_top: float


def good_tints(shown: GoodMap, gaps: bool) -> GoodTints:
    """Each cell making or wanting the good, in the colour of whichever it does more of for its
    scale, as clear as how much."""
    makers = shown.left if gaps else shown.made
    buyers = shown.short if gaps else shown.wanted
    makers_top, buyers_top = float(makers.max(initial=0.0)), float(buyers.max(initial=0.0))
    made_place, wanted_place = strength(makers, makers_top), strength(buyers, buyers_top)
    cells = np.flatnonzero((makers > 0) | (buyers > 0))
    colors = [
        tint(MAKERS, made) if made >= wanted else tint(BUYERS, wanted)
        for made, wanted in zip(
            made_place[cells].tolist(), wanted_place[cells].tolist(), strict=True
        )
    ]
    return GoodTints(cells, colors, makers_top, buyers_top)


@dataclass
class GoodsSearch:
    """The goods tab: the query, the good picked and whether the lens shows the gaps."""

    query: str = ""
    good: Good | None = None
    gaps: bool = True
    _totals: Kept[dict[Good, GoodTotals]] = field(default_factory=Kept)
    _map: Kept[GoodMap] = field(default_factory=Kept)
    _tints: Kept[GoodTints] = field(default_factory=Kept)

    def totals(self, growth: Growth) -> dict[Good, GoodTotals]:
        trade = growth.last_trade
        if trade is None:
            return {}
        return self._totals.get(trade, lambda: goods_totals(trade))

    def shown(self, growth: Growth) -> GoodMap | None:
        """The picked good on the map, as last month traded it."""
        trade, good = growth.last_trade, self.good
        if trade is None or good is None:
            return None
        grid = growth.grid
        return self._map.get(
            (trade, good), lambda: good_map(trade, grid.rows * grid.cols, grid.cols, good)
        )

    def tints(self, growth: Growth) -> GoodTints | None:
        shown = self.shown(growth)
        if shown is None:
            return None
        return self._tints.get((shown, self.gaps), lambda: good_tints(shown, self.gaps))

    def draw(self, growth: Growth, visit: Callable[[Cell], None], pick: Callable[[], None]) -> None:
        """The tab: `visit` shows a cell, `pick` turns the goods lens on once a good is picked."""
        totals = self.totals(growth)
        imgui.set_next_item_width(-1)
        hint = f"{fa.ICON_FA_SEARCH}  A good"
        _, self.query = imgui.input_text_with_hint("##findgood", hint, self.query)
        words = self.query.lower().split()
        rows = sorted(
            (t for t in totals.values() if all(word in t.good.value for word in words)),
            key=lambda t: t.good.value,
        )
        height = -FOOTER_HEIGHT if self.good is not None else 0.0
        imgui.begin_child("goodrows", (0, height), imgui.ChildFlags_.borders.value)
        _draw_header()
        if not totals:
            imgui.text_disabled("No month of trade yet.")
        elif not rows:
            imgui.text_disabled("No good goes by that name.")
        for row in rows:
            picked = row.good == self.good
            if imgui.selectable(f"##{row.good.value}", picked, 0, (0, ROW_HEIGHT))[0]:
                self.good = None if picked else row.good
                if self.good is not None:
                    pick()
            if imgui.is_item_hovered():
                imgui.set_tooltip(_row_tip(row))
            _draw_row(row)
        imgui.end_child()
        if self.good is not None:
            self._draw_picked(growth, totals.get(self.good), visit)

    def _draw_picked(
        self, growth: Growth, totals: GoodTotals | None, visit: Callable[[Cell], None]
    ) -> None:
        assert self.good is not None
        good_icon(self.good)
        imgui.same_line()
        imgui.text(self.good.value.capitalize())
        imgui.same_line()
        if totals is not None:
            imgui.text_disabled(
                f"made {amount_text(totals.made)}, wanted {amount_text(totals.wanted)} a month"
            )
        if imgui.radio_button("Gaps", self.gaps):
            self.gaps = True
        imgui.same_line()
        if imgui.radio_button("All made and wanted", not self.gaps):
            self.gaps = False
        imgui.same_line()
        if imgui.small_button(f"{fa.ICON_FA_TIMES} Clear"):
            self.good = None
            return
        shown = self.shown(growth)
        if shown is None:
            return
        makers = shown.most_left if self.gaps else shown.most_made
        buyers = shown.most_short if self.gaps else shown.most_wanted
        imgui.text_colored(imgui.ImVec4(*MAKERS, 1.0), "Left over" if self.gaps else "Made")
        _draw_spots(makers, visit, "makers")
        imgui.text_colored(imgui.ImVec4(*BUYERS, 1.0), "Short of" if self.gaps else "Wanted")
        _draw_spots(buyers, visit, "buyers")


def _row_tip(row: GoodTotals) -> str:
    return (
        f"{row.good.value.capitalize()}, a month on the map:\n"
        f"made {amount_text(row.made)}, of it left over {amount_text(row.left)}\n"
        f"wanted {amount_text(row.wanted)}, short of {amount_text(row.short)}\n"
        "Wanted counts every need it meets, whichever good would fill it."
    )


def _draw_header() -> None:
    """The columns' names, over where _draw_row puts them."""
    low = imgui.get_cursor_screen_pos()
    right = low.x + imgui.get_content_region_avail().x - 6
    draw_list = imgui.get_window_draw_list()
    imgui.text_disabled("a month on the map")
    for text, colour, end in (("left over", MAKERS, right - 70), ("short", BUYERS, right)):
        width = imgui.calc_text_size(text).x
        draw_list.add_text((end - width, low.y), color_u32((*colour, 1.0)), text)


def _draw_row(row: GoodTotals) -> None:
    """A good's row over its selectable: its icon and name, what was left over and short of."""
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()
    text_y = low.y + (ROW_HEIGHT - imgui.get_text_line_height()) / 2
    draw_good_icon(draw_list, row.good, (low.x + 2, low.y + (ROW_HEIGHT - ROW_ICON) / 2), ROW_ICON)
    draw_list.add_text((low.x + ROW_ICON + 8, text_y), color_u32(TEXT), row.good.value)
    short = amount_text(row.short)
    left = amount_text(row.left)
    right = high.x - 6
    width = imgui.calc_text_size(short).x
    draw_list.add_text((right - width, text_y), color_u32((*BUYERS, 1.0)), short)
    width_left = imgui.calc_text_size(left).x
    draw_list.add_text((right - 70 - width_left, text_y), color_u32((*MAKERS, 1.0)), left)


def _draw_spots(spots: tuple[Spot, ...], visit: Callable[[Cell], None], key: str) -> None:
    if not spots:
        imgui.text_disabled("  none")
    for index, spot in enumerate(spots[:LISTED]):
        need = "" if spot.need is None else f" ({spot.need.value})"
        label = f"{spot.kind.value}{need}: {amount_text(spot.amount)}"
        if imgui.small_button(f"{fa.ICON_FA_CROSSHAIRS}##{key}{index}"):
            visit(spot.cell)
        imgui.same_line()
        imgui.text(label)


def draw_goods_legend(good: Good, tints: GoodTints, gaps: bool) -> None:
    """The goods lens's legend inside the current window: the good and both scales."""
    good_icon(good)
    imgui.same_line()
    imgui.text(f"{good.value.capitalize()}: {'gaps' if gaps else 'made and wanted'} a month")
    for colour, name, top in (
        (MAKERS, "left over" if gaps else "made", tints.makers_top),
        (BUYERS, "short of" if gaps else "wanted", tints.buyers_top),
    ):
        _scale_bar(colour)
        imgui.text_disabled(f"{name}: up to {amount_text(top)} in a cell")
    imgui.text_disabled("Right-click a cell to see its part.")


def _scale_bar(colour: tuple[float, float, float]) -> None:
    draw_list = imgui.get_window_draw_list()
    low = imgui.get_cursor_screen_pos()
    steps = 24
    width = LEGEND_WIDTH / steps
    for step in range(steps):
        alpha = LEAST_ALPHA + (MOST_ALPHA - LEAST_ALPHA) * step / (steps - 1)
        left = low.x + step * width
        draw_list.add_rect_filled(
            (left, low.y), (left + width + 1, low.y + LEGEND_BAR), rgba_u32((*colour, alpha))
        )
    imgui.dummy((LEGEND_WIDTH, LEGEND_BAR))


def draw_cell_good(growth: Growth, good: Good, cell: Cell) -> None:
    """A good in a cell's details: what its buildings made and had left, what they wanted, got
    and were short of, and how much of what they got came from sellers nearby or stations."""
    trade = growth.last_trade
    section(f"{good.value} here, last month")
    if trade is None:
        imgui.text_disabled("No month of trade yet.")
        return
    shown = cell_good(trade, cell[0] * growth.grid.cols + cell[1], good)
    if not shown.makers and not shown.buyers:
        imgui.text_disabled("Nothing here makes or wants it.")
    for kind, made, left in shown.makers:
        _line(MAKERS, f"{kind.value} made {amount_text(made)}, left over {amount_text(left)}")
    for kind, need, wanted, got in shown.buyers:
        short = max(wanted - got, 0.0)
        _line(
            BUYERS,
            f"{kind.value} wants {amount_text(wanted)} {need.value}, got {amount_text(got)},"
            f" short of {amount_text(short)}",
        )
    for need, got, nearby in shown.nearby:
        imgui.text_disabled(
            f"{need.value}: {amount_text(nearby)} from sellers nearby,"
            f" {amount_text(max(got - nearby, 0.0))} from stations"
        )


def _line(colour: Rgba | tuple[float, float, float], text: str) -> None:
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*colour[:3], 1.0))
    imgui.bullet()
    imgui.text_wrapped(text)
    imgui.pop_style_color()
