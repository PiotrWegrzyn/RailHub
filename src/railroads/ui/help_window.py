"""The help window: every building with what it makes and wants in a month, and every good with
what it earns and who makes and wants it."""

from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.constructions import BUILDING_RULES, BuildingKind
from railroads.sim.demand import price_of
from railroads.sim.finance import dollars
from railroads.sim.goods import FITS, GOOD_YEARS, MAKES, NEED_YEARS, NEEDS, Good, Need
from railroads.sim.ladders import LADDERS, PASSENGERS_PER_POINT, POINTS, Points
from railroads.ui.search import goods_of
from railroads.ui.theme import BRASS, good_amount, good_icon

WINDOW_SIZE = (860.0, 600.0)
WINDOW_TOP = 56.0


@dataclass(frozen=True)
class BuildingHelp:
    kind: BuildingKind
    group: Points
    makes: tuple[tuple[Good, float], ...]
    # Each need, how much of it a month, and the year it starts to be felt (0 for always).
    wants: tuple[tuple[Need, float, int], ...]
    travellers: int
    space: int
    from_year: int


@dataclass(frozen=True)
class GoodHelp:
    good: Good
    price: float
    from_year: int
    made_by: tuple[str, ...]
    wanted_by: tuple[str, ...]


def _opening_year(kind: BuildingKind) -> int:
    return next((ladder.from_year for ladder in LADDERS if kind in ladder.kinds), 0)


def building_help() -> list[BuildingHelp]:
    """Every kind of building, grouped by the points it gives."""
    rows = [
        BuildingHelp(
            kind,
            POINTS[kind][0],
            tuple(MAKES.get(kind, {}).items()),
            tuple(
                (need, amount, NEED_YEARS.get(need, 0))
                for need, amount in NEEDS.get(kind, {}).items()
            ),
            PASSENGERS_PER_POINT[POINTS[kind][0]] * POINTS[kind][1],
            BUILDING_RULES[kind].space,
            _opening_year(kind),
        )
        for kind in BuildingKind
    ]
    order = list(Points)
    return sorted(rows, key=lambda row: order.index(row.group))


def good_help() -> list[GoodHelp]:
    """Every good, with its price where all the demand for it is left."""
    rows = []
    for good in Good:
        needs = {need for need, goods in FITS.items() if good in goods}
        rows.append(
            GoodHelp(
                good,
                price_of(good),
                GOOD_YEARS.get(good, 0),
                tuple(kind.value for kind, made in MAKES.items() if good in made),
                tuple(kind.value for kind, wanted in NEEDS.items() if needs & set(wanted)),
            )
        )
    return rows


@dataclass
class HelpWindow:
    open: bool = False
    query: str = ""

    def draw(self) -> None:
        if not self.open:
            return
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(
            (display.x / 2, WINDOW_TOP), imgui.Cond_.first_use_ever.value, (0.5, 0.0)
        )
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        title = f"{fa.ICON_FA_QUESTION_CIRCLE}  Buildings and goods###help"
        expanded, self.open = imgui.begin(title, self.open)
        if expanded:
            imgui.set_next_item_width(-1)
            hint = f"{fa.ICON_FA_SEARCH}  Filter by building or good"
            _, self.query = imgui.input_text_with_hint("##help filter", hint, self.query)
            if imgui.begin_tab_bar("help tabs"):
                if imgui.begin_tab_item("Buildings")[0]:
                    self._draw_buildings()
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Goods")[0]:
                    self._draw_goods()
                    imgui.end_tab_item()
                imgui.end_tab_bar()
        imgui.end()

    def _shown(self, *names: str) -> bool:
        words = self.query.lower().split()
        return all(any(word in name.lower() for name in names) for word in words)

    def _draw_buildings(self) -> None:
        if not imgui.begin_table("all buildings", 6, _TABLE_FLAGS):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_setup_column("Building")
        imgui.table_setup_column("Makes a month", imgui.TableColumnFlags_.width_stretch.value, 1.0)
        imgui.table_setup_column("Wants a month", imgui.TableColumnFlags_.width_stretch.value, 1.6)
        imgui.table_setup_column("Travellers")
        imgui.table_setup_column("Space")
        imgui.table_setup_column("Opens")
        imgui.table_headers_row()
        group = None
        for row in building_help():
            if not self._shown(row.kind.value, *goods_of(row.kind)):
                continue
            if row.group != group:
                group = row.group
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_colored(imgui.ImVec4(*BRASS), group.value.capitalize())
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(row.kind.value)
            imgui.table_next_column()
            _draw_makes(row)
            imgui.table_next_column()
            _draw_wants(row)
            imgui.table_next_column()
            imgui.text(f"{row.travellers:,}" if row.travellers else "-")
            imgui.table_next_column()
            imgui.text(f"{row.space}/6")
            imgui.table_next_column()
            imgui.text(str(row.from_year) if row.from_year else "always")
        imgui.end_table()

    def _draw_goods(self) -> None:
        if not imgui.begin_table("all goods", 5, _TABLE_FLAGS):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_setup_column("Good")
        imgui.table_setup_column("Base price")
        imgui.table_setup_column("Appears")
        imgui.table_setup_column("Made by", imgui.TableColumnFlags_.width_stretch.value, 1.0)
        imgui.table_setup_column("Wanted by", imgui.TableColumnFlags_.width_stretch.value, 1.4)
        imgui.table_headers_row()
        for row in good_help():
            if not self._shown(row.good.value, *row.made_by, *row.wanted_by):
                continue
            imgui.table_next_row()
            imgui.table_next_column()
            good_icon(row.good)
            imgui.same_line(0, 6)
            imgui.text(row.good.value)
            imgui.table_next_column()
            imgui.text(dollars(row.price))
            imgui.table_next_column()
            imgui.text(str(row.from_year) if row.from_year else "always")
            imgui.table_next_column()
            imgui.text_wrapped(", ".join(row.made_by) or _NO_MAKER.get(row.good, "-"))
            imgui.table_next_column()
            imgui.text_wrapped(", ".join(row.wanted_by) or _NO_BUYER.get(row.good, "-"))
        imgui.end_table()


_TABLE_FLAGS = (
    imgui.TableFlags_.row_bg.value
    | imgui.TableFlags_.borders_inner_h.value
    | imgui.TableFlags_.scroll_y.value
    | imgui.TableFlags_.sizing_fixed_fit.value
)
_NO_MAKER = {Good.PASSENGERS: "every building", Good.MAIL: "every building"}
_NO_BUYER = {Good.PASSENGERS: "every building", Good.MAIL: "every building"}


def _draw_makes(row: BuildingHelp) -> None:
    if not row.makes:
        imgui.text_disabled("-")
    for index, (good, amount) in enumerate(row.makes):
        if index:
            imgui.same_line(0, 10)
        good_amount(good, f"{amount:g}")


def _draw_wants(row: BuildingHelp) -> None:
    if not row.wants:
        imgui.text_disabled("-")
    for index, (need, amount, year) in enumerate(row.wants):
        if index:
            imgui.same_line(0, 10)
        good_amount(need, f"{amount:g}")
        if year:
            imgui.same_line(0, 2)
            imgui.text_disabled(f"({year})")
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Wanted from {year}")
