"""The help window: every building with what it makes and wants in a month, every good with
what it earns and who makes and wants it, every engine with what it can do, when and with
which tech it is sold, and the real engines it is drawn after, how towns work, the target
set for the company and going bankrupt, and why a game starts on a quarter of its place."""

from dataclasses import dataclass
from functools import cache

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.clock import END_YEAR, HISTORY_START
from railroads.sim.company import BANKRUPT_MONTHS, BOOK_SHARE, TARGETS, Difficulty
from railroads.sim.constructions import BUILDING_RULES, BuildingKind
from railroads.sim.economy import SCARCE_WORTH
from railroads.sim.finance import dollars
from railroads.sim.goods import (
    FITS,
    FREIGHT_SHARE,
    GOOD_YEARS,
    MAKES,
    NEEDS,
    Good,
    Need,
    freight_of,
    price_of,
)
from railroads.sim.ladders import LADDERS, PASSENGERS_PER_POINT, POINTS, Points
from railroads.sim.lots import FIRST_FIBONACCI, PRICE_UNIT, fibonacci
from railroads.sim.ownership import EXPERIENCE_PER_POINT
from railroads.sim.tech import NODES, Tree, path_to
from railroads.sim.tiles import opened_from
from railroads.sim.trains import ENGINES, Engine, successor
from railroads.ui.buy_train import fuel_use
from railroads.ui.engine_origins import origins
from railroads.ui.ornate import begin_ornate, end_ornate
from railroads.ui.search import goods_of
from railroads.ui.theme import BRASS, LIST_TABLE_FLAGS, WINDOW_TOP, good_amount, good_icon, tip

WINDOW_SIZE = (860.0, 600.0)


def lot_price(bought: int) -> float:
    """What the next lot costs once this many have been bought (railroads.sim.lots)."""
    return float(fibonacci(FIRST_FIBONACCI + bought) * PRICE_UNIT)


TOWNS_HELP = (
    (
        "Every place sending passengers belongs to a town, shown in its own colour on the Towns lens"
        " above the minimap. When a building goes up, it and every building next to it take the"
        " colour sending the most passengers among them; a building with no coloured neighbour"
        " starts a new town. Any other town with a building there joins that colour whole, so towns"
        " that grow into each other merge and two towns never touch."
    ),
    (
        "A station takes the colour of the town around it when it is built and again every year;"
        " its window shows the colour and any other station of yours in that town."
    ),
    (
        "Passengers and mail only board for a stop in another town, so two stations in one town"
        " earn nothing carrying them to each other. A train waiting for a full load doesn't wait for"
        " them either."
    ),
)
GOAL_HELP = (
    (
        "Your company is worth its cash, less any debt, plus the shares you own at what they"
        f" would fetch today, plus {BOOK_SHARE:.0%} of what your stations, with their levels,"
        " storage and facilities, your track and your trains would cost today. The top bar shows it against"
        " your target, with what it is made of on hover, and the Reports window charts it every"
        " month."
    ),
    (
        f"The new game page sets the target for 1 January {END_YEAR}: Easy"
        f" {dollars(TARGETS[Difficulty.EASY])}, Normal {dollars(TARGETS[Difficulty.NORMAL])} or"
        f" Hard {dollars(TARGETS[Difficulty.HARD])}. The game tells you when your company is"
        f" first worth that much, and on 1 January {END_YEAR} whether it still is. Play goes on"
        " after that."
    ),
    (
        "Upkeep is charged whatever the cash, so it can go below nothing. The moment it does, the"
        " game warns you, and the top bar counts down the months left. If"
        f" {BANKRUPT_MONTHS} months in a row begin with the cash below nothing, the company goes"
        " bankrupt and the game ends; a month begun above nothing starts the count again."
    ),
)
MAP_HELP = (
    (
        "A game starts on a honeycomb of seven big hexagons, or lots, in the middle of its place,"
        " holding a quarter of it: one lot in the middle and six around it, each about 140 km"
        " across on Poland."
    ),
    (
        f"Every hex of the map grows every month, from the year {HISTORY_START} before play and"
        " all through it, so the time to set up a new game and to run each month grows with the"
        " map: twice the land takes about twice as long. A quarter of the land sets up and runs about three"
        " times as fast as the whole place: Poland's towns grow to 1850 in about 3 seconds"
        " instead of 10 on a fast computer, and a month takes about 25 ms instead of 70. In the"
        " browser both take about twice as long."
    ),
    (
        "The land around it can be bought a lot at a time, each a seventh of the land a game"
        " starts on. Any lot sharing a side with yours is for sale, twelve at the start: past the"
        " map's edge it shows its rivers, forests and mountains, without the buildings, behind a"
        f" chain and a padlock with its price. The first costs {dollars(lot_price(0))} and each"
        f" after it the next Fibonacci number of thousands: {dollars(lot_price(1))},"
        f" {dollars(lot_price(2))} and so on."
    ),
    (
        f"Click a lot and confirm to buy it. It then grows on its own from {HISTORY_START} to this"
        " year, as the rest of the map did, while the waiting screen shows, and the view opens on it. It"
        " gets only a quarter of the buildings every map is sure to have, picked at random, and"
        " each one-off building the game hasn't got yet comes up there at a quarter of the"
        " chance it had over the years played, at most. Each lot bought makes a month a few"
        " milliseconds longer."
    ),
)
ENGINES_NOTE = (
    "An engine is sold from its year once its tech is taken, until a newer one of its line comes."
    " Tech is taken in the tech window (R): every level a tree reaches gives a point for one node"
    " whose requirements are taken. A tree gains experience only from deliveries: what cargo of"
    " its goods earns as buildings take it out of your stations, and for passengers and mail"
    " what they pay as they get off. Every point of the buildings of a tree you own adds"
    f" {EXPERIENCE_PER_POINT:.0%} to it. Engines are drawn after real ones, linked to their"
    " Wikipedia pages."
)
GOODS_NOTE = (
    f"A good's freight is what carrying a unit of it earns: {FREIGHT_SHARE:.0%} of its market"
    f" price, up to {SCARCE_WORTH:g} times that to a building that went short over the last"
    " quarter, paid for everything your stations hand out, what trains brought and what they"
    " gathered. Passengers and mail pay their freight times the share of demand left where they"
    " get off. What your own buildings put into your stations is sold to them at the market price."
)


@dataclass(frozen=True)
class BuildingHelp:
    kind: BuildingKind
    group: Points
    makes: tuple[tuple[Good, float], ...]
    # Each need, how much of it a month, and from when it is felt ("" for always): a year, or the
    # demand tile level that opens it.
    wants: tuple[tuple[Need, float, str], ...]
    travellers: int
    space: int
    from_year: int


@dataclass(frozen=True)
class GoodHelp:
    good: Good
    price: float
    freight: float
    from_year: int
    made_by: tuple[str, ...]
    wanted_by: tuple[str, ...]


@dataclass(frozen=True)
class EngineHelp:
    name: str
    engine: Engine
    # The year its sale ends and the engine that ends it, if one does.
    sold_until: tuple[int, str] | None
    # The nodes to take for it in each tree, each after those it requires.
    tech: tuple[tuple[Tree, tuple[str, ...]], ...]
    # The real engines it is drawn after, as names and Wikipedia links.
    origins: tuple[tuple[str, str], ...]


def _opening_year(kind: BuildingKind) -> int:
    return next((ladder.from_year for ladder in LADDERS if kind in ladder.kinds), 0)


@cache
def building_help() -> tuple[BuildingHelp, ...]:
    """Every kind of building, grouped by the points it gives; worked out once, as the open
    window draws them all every frame."""
    rows = [
        BuildingHelp(
            kind,
            POINTS[kind][0],
            tuple(MAKES.get(kind, {}).items()),
            tuple(
                (need, amount, opened_from(kind, need))
                for need, amount in NEEDS.get(kind, {}).items()
            ),
            PASSENGERS_PER_POINT[POINTS[kind][0]] * POINTS[kind][1],
            BUILDING_RULES[kind].space,
            _opening_year(kind),
        )
        for kind in BuildingKind
    ]
    order = list(Points)
    return tuple(sorted(rows, key=lambda row: order.index(row.group)))


@cache
def good_help() -> tuple[GoodHelp, ...]:
    """Every good, with its market price and what carrying it earns; worked out once."""
    rows = []
    for good in Good:
        needs = {need for need, goods in FITS.items() if good in goods}
        rows.append(
            GoodHelp(
                good,
                price_of(good),
                freight_of(good),
                GOOD_YEARS.get(good, 0),
                tuple(kind.value for kind, made in MAKES.items() if good in made),
                tuple(kind.value for kind, wanted in NEEDS.items() if needs & set(wanted)),
            )
        )
    return tuple(rows)


def _tech_path(engine: Engine) -> tuple[tuple[Tree, tuple[str, ...]], ...]:
    path = path_to(engine.needs)
    trees = dict.fromkeys(NODES[key][0] for key in path)
    return tuple(
        (tree, tuple(NODES[key][1].name for key in path if NODES[key][0] == tree)) for tree in trees
    )


@cache
def engine_help() -> tuple[EngineHelp, ...]:
    """Every engine by the year it is first sold; worked out once."""
    rows = []
    for name, engine in sorted(ENGINES.items(), key=lambda item: item[1].year):
        newer = successor(name)
        rows.append(
            EngineHelp(
                name,
                engine,
                (ENGINES[newer].year, ENGINES[newer].title) if newer else None,
                _tech_path(engine),
                origins(name),
            )
        )
    return tuple(rows)


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
        title = f"{fa.ICON_FA_QUESTION_CIRCLE}  Buildings, goods and engines###help"
        expanded, self.open = begin_ornate(title)
        if expanded:
            imgui.set_next_item_width(-1)
            hint = f"{fa.ICON_FA_SEARCH}  Filter by building, good or engine"
            _, self.query = imgui.input_text_with_hint("##help filter", hint, self.query)
            if imgui.begin_tab_bar("help tabs"):
                if imgui.begin_tab_item("Buildings")[0]:
                    self._draw_buildings()
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Goods")[0]:
                    self._draw_goods()
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Engines")[0]:
                    self._draw_engines()
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Towns")[0]:
                    _draw_paragraphs(TOWNS_HELP)
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Goal")[0]:
                    _draw_paragraphs(GOAL_HELP)
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Map")[0]:
                    _draw_paragraphs(MAP_HELP)
                    imgui.end_tab_item()
                imgui.end_tab_bar()
        end_ornate()

    def _shown(self, *names: str) -> bool:
        words = self.query.lower().split()
        return all(any(word in name.lower() for name in names) for word in words)

    def _draw_buildings(self) -> None:
        if not imgui.begin_table("all buildings", 6, LIST_TABLE_FLAGS):
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
        imgui.text_wrapped(GOODS_NOTE)
        if not imgui.begin_table("all goods", 6, LIST_TABLE_FLAGS):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_setup_column("Good")
        imgui.table_setup_column("Market price")
        imgui.table_setup_column("Freight")
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
            imgui.text(dollars(row.freight))
            imgui.table_next_column()
            imgui.text(str(row.from_year) if row.from_year else "always")
            imgui.table_next_column()
            imgui.text_wrapped(", ".join(row.made_by) or _NO_MAKER.get(row.good, "-"))
            imgui.table_next_column()
            imgui.text_wrapped(", ".join(row.wanted_by) or _NO_BUYER.get(row.good, "-"))
        imgui.end_table()

    def _draw_engines(self) -> None:
        imgui.text_wrapped(ENGINES_NOTE)
        if not imgui.begin_table("all engines", 9, LIST_TABLE_FLAGS):
            return
        imgui.table_setup_scroll_freeze(1, 1)
        imgui.table_setup_column("Engine")
        imgui.table_setup_column("On sale")
        imgui.table_setup_column("Tech to take", imgui.TableColumnFlags_.width_stretch.value, 1.2)
        imgui.table_setup_column("Speed")
        imgui.table_setup_column("Cars")
        imgui.table_setup_column("Price")
        imgui.table_setup_column("Burns a cell")
        imgui.table_setup_column("Range")
        imgui.table_setup_column("Drawn after", imgui.TableColumnFlags_.width_stretch.value, 1.0)
        imgui.table_headers_row()
        for row in engine_help():
            engine = row.engine
            names = (engine.title, engine.kind.value, engine.power.value)
            if not self._shown(*names, *(name for name, _ in row.origins)):
                continue
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(engine.title)
            imgui.text_disabled(f"{engine.kind.value}, {engine.power.value}")
            imgui.table_next_column()
            _draw_sale(row)
            imgui.table_next_column()
            _draw_tech(row)
            imgui.table_next_column()
            imgui.text(f"{engine.kmh:.0f} km/h")
            imgui.text_disabled(f"full in {engine.pickup:g} s")
            imgui.table_next_column()
            imgui.text(f"{engine.cars} + dining" if engine.express else str(engine.cars))
            imgui.table_next_column()
            imgui.text(dollars(engine.price))
            imgui.table_next_column()
            imgui.text(fuel_use(engine).removesuffix(" a cell"))
            imgui.table_next_column()
            imgui.text(f"{engine.tank:.0f} cells" if engine.tank else "no limit")
            imgui.table_next_column()
            _draw_origins(row)
        imgui.end_table()


def _draw_paragraphs(paragraphs: tuple[str, ...]) -> None:
    for paragraph in paragraphs:
        imgui.text_wrapped(paragraph)
        imgui.spacing()


def _draw_sale(row: EngineHelp) -> None:
    if row.sold_until is None:
        imgui.text(f"{row.engine.year} on")
        return
    year, title = row.sold_until
    imgui.text(f"{row.engine.year}-{year}")
    tip(f"Sold until the {title} can be bought")


def _draw_tech(row: EngineHelp) -> None:
    if not row.tech:
        imgui.text_disabled("none")
    for tree, nodes in row.tech:
        imgui.text_wrapped(f"{tree.value.capitalize()}: {' > '.join(nodes)}")


def _draw_origins(row: EngineHelp) -> None:
    if not row.origins:
        imgui.text_disabled("our own design")
    for name, url in row.origins:
        imgui.text_link_open_url(name, url)
        tip(f"{name}\n{url}")


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
    for index, (need, amount, since) in enumerate(row.wants):
        if index:
            imgui.same_line(0, 10)
        good_amount(need, f"{amount:g}")
        if since:
            imgui.same_line(0, 2)
            imgui.text_disabled(f"({since})")
            tip(f"Wanted from {since}")
