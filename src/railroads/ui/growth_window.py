"""How likely every kind of building is to come up (railroads.sim.growth_odds), shown three ways:
the growth window listing every ladder's first kind with how many a year the map may get and why
none if so, and the one-off buildings this game may get (railroads.sim.novelties) with theirs, the growth lens tinting the land by the chance of the kind picked there in a year with
a scale beside it, and the chances of the cell picked in its details, with why not."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.kept import Kept
from railroads.sim.grids import Cell, FloatArray
from railroads.sim.growth import Growth, ladder_open
from railroads.sim.growth_odds import (
    LadderOdds,
    ladder_odds,
    monthly_chances,
    turn_instead,
    why_not_here,
    yearly,
)
from railroads.sim.ladder_types import Ladder
from railroads.sim.novelties import NoveltyOdds
from railroads.ui.ornate import begin_ornate, end_ornate, section
from railroads.ui.theme import (
    BRASS,
    GROWN,
    SIDE_WINDOW_POS,
    TEXT,
    TEXT_DIM,
    Rgba,
    color_u32,
    colored,
    rgba_u32,
)

WINDOW_SIZE = (380.0, 520.0)
ROW_HEIGHT = 22.0
FOOTER_HEIGHT = 150.0
# The lens's scale runs from the likeliest cell's chance down this many times lower.
SCALE_SPAN = 1000.0
# The colours of the scale, from the least likely to the likeliest, and of land that suits a kind
# but lacks what it needs nearby.
SCALE = (
    (0.2, 0.3, 0.75, 0.55),
    (0.1, 0.65, 0.75, 0.55),
    (0.35, 0.85, 0.3, 0.6),
    (0.97, 0.85, 0.2, 0.65),
    (1.0, 0.4, 0.12, 0.7),
)
MISSING_NEED = (0.55, 0.55, 0.55, 0.45)
LEGEND_WIDTH = 260.0
LEGEND_BAR = 12.0
# Smaller chances read as under this percentage.
LEAST_PERCENT = 0.0001
# How many of the likeliest kinds a cell's details list.
LIKELIEST_SHOWN = 5


def per_year_text(per_year: float) -> str:
    """How many a year, at most, in words: a count a year, or once in so many years."""
    if per_year <= 0:
        return "none"
    if per_year >= 1:
        return f"up to {per_year:,.1f} a year"
    years = 1 / per_year
    if years >= 1000:
        return "under once in 1,000 years"
    return f"up to once in {years:,.0f} years" if years >= 1.5 else "up to about one a year"


def percent_text(chance: float) -> str:
    """A chance as a percentage to two digits that count, "under 0.0001%" below that."""
    if chance <= 0:
        return "0%"
    if chance >= 0.995:
        return "100%"
    percent = chance * 100
    if percent >= 10:
        return f"{percent:.0f}%"
    if percent < LEAST_PERCENT:
        return f"under {LEAST_PERCENT:f}".rstrip("0") + "%"
    digits = 1 - math.floor(math.log10(percent))
    return f"{percent:.{digits}f}%"


def scale_colour(at: float) -> Rgba:
    """The scale's colour at a share of the way from its bottom (0) to its top (1)."""
    at = min(max(at, 0.0), 1.0) * (len(SCALE) - 1)
    low = min(int(at), len(SCALE) - 2)
    part = at - low
    return tuple(a + (b - a) * part for a, b in zip(SCALE[low], SCALE[low + 1], strict=True))  # type: ignore[return-value]


def scale_place(chances: FloatArray, top: float) -> FloatArray:
    """Where yearly chances fall on the scale, by their logarithm, from top / SCALE_SPAN to top."""
    return np.clip(np.log10(np.maximum(chances, 1e-300) / top) / np.log10(SCALE_SPAN) + 1, 0, 1)


@dataclass(frozen=True, slots=True)
class Tints:
    """A lens's flat cells, their packed colours, and the top of its scale."""

    cells: npt.NDArray[np.intp]
    colors: list[int]
    top: float


def lens_tints(growth: Growth, ladder: Ladder, monthly: FloatArray) -> Tints:
    """The growth lens's tints for a ladder: each cell with a chance by where on the scale it
    falls, and the land that suits it but lacks what it needs grey."""
    chances = yearly(monthly)
    likely = np.flatnonzero(chances > 0)
    top = float(chances.max()) if likely.size else 0.0
    places = scale_place(chances[likely], top) if likely.size else np.empty(0)
    steps = 64
    palette = [rgba_u32(scale_colour(step / (steps - 1))) for step in range(steps)]
    colors = [palette[round(place * (steps - 1))] for place in places.tolist()]
    missing = np.flatnonzero((growth.land_weights(ladder) > 0) & (chances <= 0))
    cells = np.concatenate([likely, missing])
    return Tints(cells, colors + [rgba_u32(MISSING_NEED)] * missing.size, top)


class GrowthOdds:
    """Every ladder's chances this month and its odds on the map, worked out again only once a
    month has passed or the buildings changed."""

    def __init__(self) -> None:
        self._kept: Kept[dict[int, tuple[FloatArray, LadderOdds]]] = Kept()
        self._tints: Kept[Tints] = Kept()
        self._cell: Kept[CellOdds] = Kept()
        self._one_offs: Kept[list[NoveltyOdds]] = Kept()

    def all(self, growth: Growth) -> dict[int, tuple[FloatArray, LadderOdds]]:
        """Each ladder's monthly chances and odds, by the ladder's id."""
        key = (growth, growth.months, growth.constructions.version)

        def work_out() -> dict[int, tuple[FloatArray, LadderOdds]]:
            every = {}
            for ladder in growth.rule.ladders:
                monthly = monthly_chances(growth, ladder)
                every[id(ladder)] = (monthly, ladder_odds(growth, ladder, monthly))
            return every

        return self._kept.get(key, work_out)

    def one_offs(self, growth: Growth) -> list[NoveltyOdds]:
        """The one-off buildings this game may get, with their chances and why none now."""
        novelties = growth.novelties
        if novelties is None:
            return []
        key = (growth, growth.months, growth.constructions.version)
        return self._one_offs.get(key, lambda: novelties.odds(growth))

    def of(self, growth: Growth, ladder: Ladder) -> tuple[FloatArray, LadderOdds]:
        return self.all(growth)[id(ladder)]

    def tints(self, growth: Growth, ladder: Ladder) -> Tints:
        monthly = self.of(growth, ladder)[0]
        return self._tints.get((monthly, ladder), lambda: lens_tints(growth, ladder, monthly))

    def cell(self, growth: Growth, cell: Cell, lens: Ladder | None) -> CellOdds:
        every = self.all(growth)
        key = (every, cell[0], cell[1], lens)
        return self._cell.get(key, lambda: cell_odds(growth, self, cell, lens))


@dataclass
class GrowthWindow:
    """The growth window, and the ladder the growth lens shows."""

    open: bool = False
    query: str = ""
    ladder: Ladder | None = None
    odds: GrowthOdds = field(default_factory=GrowthOdds)

    def draw(self, growth: Growth, visit: Callable[[Cell], None], pick: Callable[[], None]) -> None:
        """The window, if open: `visit` shows a cell, `pick` turns the growth lens on once a
        kind is picked."""
        if not self.open:
            return
        imgui.set_next_window_pos(SIDE_WINDOW_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, still_open = begin_ornate(f"{fa.ICON_FA_LEAF}  Growth###growth")
        if expanded:
            every = self.odds.all(growth)
            self._draw_list(growth, every, pick)
            if self.ladder is not None:
                self._draw_picked(every[id(self.ladder)][1], visit)
        end_ornate()
        if not still_open:
            self.open = False

    def _draw_list(
        self,
        growth: Growth,
        every: dict[int, tuple[FloatArray, LadderOdds]],
        pick: Callable[[], None],
    ) -> None:
        imgui.set_next_item_width(-1)
        _, self.query = imgui.input_text_with_hint("##growthfind", "Find a building", self.query)
        words = self.query.lower().split()
        rows = [
            odds
            for _, odds in every.values()
            if all(word in odds.ladder.name.lower() for word in words)
        ]
        rows.sort(key=lambda odds: (-odds.per_year, odds.ladder.name))
        height = -FOOTER_HEIGHT if self.ladder is not None else 0.0
        imgui.begin_child("growthrows", (0, height), imgui.ChildFlags_.borders.value)
        _wrapped(TEXT_DIM, "New ones the map may get; pick one to see where on the map.")
        for odds in rows:
            picked = odds.ladder is self.ladder
            if imgui.selectable(f"##{odds.ladder.name}", picked, 0, (0, ROW_HEIGHT))[0]:
                self.ladder = None if picked else odds.ladder
                if self.ladder is not None:
                    pick()
            if imgui.is_item_hovered():
                imgui.set_tooltip(_row_tip(odds))
            _draw_row(odds)
        _draw_one_offs(self.odds.one_offs(growth))
        imgui.end_child()

    def _draw_picked(self, odds: LadderOdds, visit: Callable[[Cell], None]) -> None:
        imgui.text(odds.ladder.name.capitalize())
        imgui.same_line()
        imgui.text_disabled(f"{odds.standing:,} standing, {per_year_text(odds.per_year)}")
        if odds.why_none is not None:
            _wrapped(BRASS, odds.why_none)
        for way in odds.other_ways:
            _bullet(way)
        imgui.begin_disabled(odds.best is None)
        if imgui.button(f"{fa.ICON_FA_CROSSHAIRS}  Likeliest cell") and odds.best is not None:
            visit(odds.best)
        imgui.end_disabled()
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_TIMES}  Clear"):
            self.ladder = None


def _draw_one_offs(one_offs: list[NoveltyOdds]) -> None:
    """The one-off buildings this game may get: each one's chance, when, and why none now."""
    if not one_offs:
        return
    section("One-off buildings this game may get")
    for odds in one_offs:
        name = odds.kind.value.capitalize()
        colour = TEXT_DIM if odds.why_none is not None else TEXT
        _wrapped(colour, f"{name}: {percent_text(odds.chance)} {odds.when}")
        if odds.why_none is not None:
            _wrapped(BRASS, odds.why_none)


def _row_tip(odds: LadderOdds) -> str:
    lines = [f"{odds.ladder.name.capitalize()}: {per_year_text(odds.per_year)}"]
    if odds.why_none is not None:
        lines.append(odds.why_none)
    lines += odds.other_ways
    return "\n".join(lines)


def _draw_row(odds: LadderOdds) -> None:
    """Draws a row over its selectable: the kind's name, how many stand and how many a year."""
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()
    text_y = low.y + (ROW_HEIGHT - imgui.get_text_line_height()) / 2
    growing = odds.per_year > 0
    draw_list.add_text(
        (low.x + 4, text_y), color_u32(TEXT if growing else TEXT_DIM), odds.ladder.name
    )
    number = f"{odds.standing:,}"
    draw_list.add_text((low.x + 170, text_y), color_u32(TEXT_DIM), number)
    rate = _short_rate(odds.per_year)
    width = imgui.calc_text_size(rate).x
    colour = GROWN if growing else TEXT_DIM
    draw_list.add_text((high.x - width - 6, text_y), color_u32(colour), rate)


def _short_rate(per_year: float) -> str:
    if per_year <= 0:
        return "-"
    if per_year >= 1:
        return f"{per_year:,.1f}/yr"
    years = 1 / per_year
    return "1/1000+ yr" if years >= 1000 else f"1/{years:,.0f} yr"


def draw_legend(ladder: Ladder, tints: Tints) -> None:
    """The growth lens's legend inside the current window: the kind, the scale of a cell's
    chance a year with its ends, and what grey means."""
    imgui.text(f"{ladder.name.capitalize()}: chance a year")
    draw_list = imgui.get_window_draw_list()
    low = imgui.get_cursor_screen_pos()
    steps = 32
    width = LEGEND_WIDTH / steps
    for step in range(steps):
        colour = color_u32((*scale_colour(step / (steps - 1))[:3], 1.0))
        left = low.x + step * width
        draw_list.add_rect_filled((left, low.y), (left + width + 1, low.y + LEGEND_BAR), colour)
    imgui.dummy((LEGEND_WIDTH, LEGEND_BAR))
    if tints.top > 0:
        bottom = percent_text(tints.top / SCALE_SPAN)
        top = percent_text(tints.top)
        imgui.text_disabled(bottom if bottom.startswith("under") else f"{bottom} or less")
        imgui.same_line(
            LEGEND_WIDTH - imgui.calc_text_size(top).x + imgui.get_style().window_padding.x
        )
        imgui.text_disabled(top)
    else:
        imgui.text_disabled("No cell has any chance now.")
    swatch = imgui.get_cursor_screen_pos()
    size = imgui.get_text_line_height()
    draw_list.add_rect_filled(
        swatch, (swatch.x + size, swatch.y + size), color_u32((*MISSING_NEED[:3], 1.0))
    )
    imgui.dummy((size, size))
    imgui.same_line()
    imgui.text_disabled("suits the land, but can't come up now")
    imgui.text_disabled("Right-click a cell to see why.")


@dataclass(frozen=True, slots=True)
class CellOdds:
    """What a cell's details show of its chances: the growth lens's kind with its chance a year,
    why none and what a turn does instead, the likeliest kinds here and the kinds whose land suits
    the cell but that can't come up, with the first reason, those already open first."""

    lens: tuple[str, float, tuple[str, ...], str | None] | None
    likeliest: tuple[tuple[str, float], ...]
    blocked: tuple[tuple[str, str], ...]


def cell_odds(growth: Growth, odds: GrowthOdds, cell: Cell, lens: Ladder | None) -> CellOdds:
    flat = cell[0] * growth.grid.cols + cell[1]
    every = odds.all(growth)
    shown = None
    if lens is not None:
        chance = float(yearly(every[id(lens)][0][flat : flat + 1])[0])
        reasons = () if chance > 0 else tuple(why_not_here(growth, lens, cell))
        shown = (lens.name, chance, reasons, turn_instead(growth, lens, cell))
    likeliest = []
    blocked = []
    year = growth.year
    for monthly, ladder_odds_ in every.values():
        ladder = ladder_odds_.ladder
        chance = float(yearly(monthly[flat : flat + 1])[0])
        if chance > 0:
            likeliest.append((ladder.name, chance))
        elif growth.land_weights(ladder)[flat] > 0:
            reasons = why_not_here(growth, ladder, cell)
            blocked.append((not ladder_open(ladder, year), ladder.name, reasons[:1]))
    likeliest.sort(key=lambda pair: -pair[1])
    return CellOdds(
        lens=shown,
        likeliest=tuple(likeliest[:LIKELIEST_SHOWN]),
        blocked=tuple((name, reason[0] if reason else "") for _, name, reason in sorted(blocked)),
    )


def draw_cell_odds(growth: Growth, odds: GrowthOdds, cell: Cell, lens: Ladder | None) -> None:
    """A cell's chances inside its details (CellOdds)."""
    shown = odds.cell(growth, cell, lens)
    if shown.lens is not None:
        name, chance, reasons, instead = shown.lens
        section(f"Chance of a {name}")
        if chance > 0:
            colored(GROWN, f"{percent_text(chance)} a year")
            if chance < 0.5:
                imgui.same_line()
                imgui.text_disabled(f"(once in ~{1 / chance:,.0f} years)")
        else:
            colored(BRASS, "None now:")
            for reason in reasons or ("Its chance here is too small to count.",):
                _bullet(reason)
        if instead is not None:
            _wrapped(TEXT_DIM, instead)
    section("Likeliest new buildings here")
    if not shown.likeliest:
        imgui.text_disabled("Nothing can come up here now.")
    for name, chance in shown.likeliest:
        imgui.bullet_text(f"{name.capitalize()}: {percent_text(chance)} a year")
    blocked = shown.blocked
    if blocked and imgui.tree_node_ex(f"Can't come up here ({len(blocked)})###blocked"):
        for name, reason in blocked:
            _bullet(f"{name.capitalize()}: {reason}")
        imgui.tree_pop()


def _bullet(text: str) -> None:
    """A bullet whose text wraps at the window's edge."""
    imgui.bullet()
    imgui.text_wrapped(text)


def _wrapped(colour: Rgba, text: str) -> None:
    """Text in a colour, wrapping at the window's edge."""
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*colour))
    imgui.text_wrapped(text)
    imgui.pop_style_color()
