"""What the details of a hex show, worked out without drawing: its buildings, what each makes,
gets and stocks, and the player's shares in them."""

import math
import time
import weakref
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from railroads.sim.amounts import EPSILON
from railroads.sim.clock import MONTHS_PER_YEAR
from railroads.sim.constructions import Annex, Building, BuildingKind, Construction, Station, Track
from railroads.sim.goods import BONUS_PART, RECIPES, Good, Need
from railroads.sim.grids import Cell, FloatArray
from railroads.sim.growth import Growth
from railroads.sim.ledger import Takings
from railroads.sim.ownership import CONTROL, HOLD_MONTHS, LOT, LOTS, SHARES
from railroads.sim.points import Points
from railroads.sim.tilemap import Deposit, TileMap
from railroads.sim.trade import Satisfaction, Trade
from railroads.sim.yearly import bonus_of, boosts_of, demands_of, makes_of
from railroads.ui.biomes import biome_of
from railroads.words import dollars


@dataclass(frozen=True)
class Flow:
    """How much of a good buildings got or gave out of what they wanted to; for goods made, also
    the most they could make with all their inputs, where that is more."""

    good: Good | Need
    done: float
    wanted: float
    up_to: float | None = None

    @property
    def text(self) -> str:
        line = f"{self.done:.1f} of {self.wanted:.1f}"
        return line if self.up_to is None else f"{line} (up to {self.up_to:g})"


@dataclass(frozen=True)
class HexDetails:
    title: str
    land: str
    satisfaction: str
    # Each kind of points: in the cell and in its neighbourhood.
    points: list[tuple[str, int, int]]
    buildings: list[BuildingRow]
    others: list[str]


# How long the player must hold shares before they may sell them.
HELD = f"Shares bought can be sold {HOLD_MONTHS // MONTHS_PER_YEAR} years later"


# How long a cell's description is shown again before it is worked out anew, and how many cells'
# descriptions are kept: the hovered cell's and the open details' at least. Working one out takes
# a millisecond or so, too much for every frame the mouse rests on a cell.
FRESH_S = 0.25


KEPT = 4


_last: list[tuple[weakref.ref[Growth], Cell, float, HexDetails]] = []


def recently_described(
    growth: Growth, cell: Cell, satisfaction: npt.NDArray[np.int8] | None = None
) -> HexDetails:
    """The cell's description, worked out again only if none of the last few is of this cell
    or it is older than FRESH_S; `satisfaction` as for describe."""
    now = time.monotonic()
    for owner, last_cell, at, details in _last:
        if owner() is growth and last_cell == cell and now - at < FRESH_S:
            return details
    details = describe(growth, cell, satisfaction)
    others = [kept for kept in _last if kept[1] != cell]
    _last[:] = [(weakref.ref(growth), cell, now, details), *others[: KEPT - 1]]
    return details


def describe(
    growth: Growth, cell: Cell, satisfaction: npt.NDArray[np.int8] | None = None
) -> HexDetails:
    """Everything the panel shows about one cell. `satisfaction`, every cell's as
    Growth.satisfaction has it now, saves working it out for the whole map again."""
    constructions = growth.constructions
    tile_map = constructions.tile_map
    tile = tile_map.tile(cell)
    flat = int(np.ravel_multi_index(cell, tile_map.grid.shape))
    moods = growth.satisfaction() if satisfaction is None else satisfaction
    level = int(moods[cell])
    here = points_here(growth, cell)
    points = [
        (kind.value, here[kind], int(growth.nearby(kind)[cell]))
        for kind in Points
        if growth.nearby(kind)[cell]
    ]
    built = constructions.at(cell)
    kinds = Counter(b.kind for b in built if isinstance(b, Building))
    trade = growth.last_trade
    position = None
    if trade is not None:
        found = int(np.searchsorted(trade.cells, flat))
        if found < trade.cells.size and trade.cells[found] == flat:
            position = found
    rows = []
    for kind, count in kinds.items():
        got: list[Flow] = []
        bonus_got: list[Flow] = []
        gave: list[Flow] = []
        stock: list[tuple[Need, float]] = []
        mood = "-"
        percent = None
        if trade is not None and position is not None and kind in trade.got:
            extra = {**bonus_of(kind, growth.year), **boosts_of(kind, growth.year)}
            flows = _per_good(trade.goods_got, kind, position)
            got = [flow for flow in flows if flow.good not in extra]
            bonus_got = [flow for flow in flows if flow.good in extra]
            got, bonus_got = _split_bonus_part(kind, count, got, bonus_got)
            if kind in RECIPES:
                stock = _stock(growth, kind, flat)
            gave = _per_good(trade.goods_gave, kind, position, makes_of(kind, growth.year), count)
            alone = one_cell(trade, position, kind)
            mood = Satisfaction(int(alone.levels((kind,))[0])).name.lower()
            percent = float(alone.share((kind,))[0])
        rows.append(
            BuildingRow(
                kind.value,
                count,
                [(good, amount * count) for good, amount in makes_of(kind, growth.year).items()],
                [
                    (need, (amount - BONUS_PART.get((kind, need), 0.0)) * count)
                    for need, amount in demands_of(kind, growth.year, growth.economy.east).items()
                ],
                [
                    *(
                        (need, extra * count)
                        for (of, need), extra in BONUS_PART.items()
                        if of == kind and need in demands_of(kind, growth.year)
                    ),
                    *(
                        (need, amount * count)
                        for need, amount in bonus_of(kind, growth.year, growth.economy.east).items()
                    ),
                ],
                got,
                gave,
                stock,
                mood,
                kind,
                int(growth.owners.shares(kind)[cell]),
                growth.owners.lot_price(cell, kind),
                percent,
                bonus_got,
                growth.owners.sellable(cell, kind),
                growth.owners.sellable_from(cell, kind),
                growth.owners.total(cell, kind),
                growth.ledger.last_year(kind, flat),
                growth.ledger.forecast(kind, flat),
            )
        )
    return HexDetails(
        title=f"Cell {cell[0]}, {cell[1]}: {biome_of(tile).name.title()}",
        land=(
            f"{tile.terrain.name.lower()}, {tile.relief.name.lower()}, "
            f"{tile.vegetation.name.lower()}, {tile.temperature.name.lower()}, {tile.height_m} m"
            + _deposit_note(tile_map, cell)
        ),
        satisfaction=Satisfaction(level).name.lower() if level >= 0 else "nothing trades here",
        points=points,
        buildings=rows,
        others=[_other(c, growth) for c in built if not isinstance(c, Building)]
        + _flag_note(growth, rows),
    )


def points_here(growth: Growth, cell: Cell) -> Counter[Points]:
    """Each kind of points one cell's buildings give, summed as Growth.points sums them for
    every cell, without going over the whole map."""
    here: Counter[Points] = Counter()
    for kind, (given, value) in growth.rule.points.items():
        count = int(growth.count(kind)[cell])
        if count:
            here[given] += value * count
    return here


def one_cell(trade: Trade, position: int, kind: BuildingKind) -> Trade:
    """The month's trade of one kind of building in the cell at `position` among the traded
    cells, alone: what Trade.share and Trade.levels work out for it there, without going over
    every cell."""
    at = slice(position, position + 1)
    return replace(
        trade,
        cells=trade.cells[at],
        got={kind: trade.got[kind][at]},
        needed={kind: trade.needed[kind][at]},
        gave={kind: trade.gave[kind][at]},
        could_make={kind: trade.could_make[kind][at]},
        bonus={kind: trade.bonus[kind][at]} if kind in trade.bonus else {},
    )


def sell_lots(sellable: int) -> list[int]:
    """How many shares each Sell button sells: each of LOTS the player may sell now, or just
    what they may if that is fewer than the smallest; 0 for a button that can't sell yet."""
    return [lot for lot in LOTS if lot <= sellable] or [min(LOT, sellable)]


def share_lines(row: BuildingRow) -> list[tuple[str, str, str]]:
    """The shares window's lines: what each is, its value and what it means."""
    part = row.owned / row.total if row.total else 0.0
    lines = [
        (
            f"In all ({row.count} x {SHARES:,})",
            f"{row.total:,}",
            f"Every building is {SHARES:,} shares, bought and sold {LOT}, 200 or 600 at a time",
        ),
        (
            "Yours",
            f"{row.owned:,} ({part:.0%} of what they make)",
            "Your part of what they make goes into your stations around them first",
        ),
        control_line(row),
        ("Can sell now", f"{row.sellable:,}", HELD),
    ]
    if row.held_back:
        lines.append((f"Sellable from {row.sell_from}", f"{row.held_back:,}", ""))
    if row.price is not None:
        lines.append(
            (
                f"Price of {LOT}",
                dollars(row.price),
                "Higher the more satisfied they were over the last quarter",
            )
        )
    return lines


def control_line(row: BuildingRow) -> tuple[str, str, str]:
    """Whether the player controls the buildings, or how many more shares would give control."""
    needed = math.ceil(CONTROL * row.total)
    meaning = (
        f"With {CONTROL:.0%} of the shares your flag flies big here and only then do the boosts only "
        "your buildings take work on them, on all of them; more shares only raise your part"
    )
    if row.total and row.owned >= needed:
        return ("Control", "yours", meaning)
    return ("Control", f"from {needed:,} shares", meaning)


def result_lines(row: BuildingRow) -> list[tuple[str, str, str, str]]:
    """The financial results, a line at a time: what it is, last year's money and the forecast
    for the next, and what it means."""

    def money(takings: Takings | None, pick: Callable[[Takings], float]) -> str:
        return "-" if takings is None else dollars(pick(takings))

    def line(label: str, pick: Callable[[Takings], float], tip: str) -> tuple[str, str, str, str]:
        return label, money(row.last_year, pick), money(row.forecast, pick), tip

    plural = f"The {row.name}" if row.count == 1 else f"All {row.count} {row.name}s"
    return [
        line(
            f"{plural}: sold",
            lambda t: t.output,
            "What they sold of what they made, at its market price; next year at the pace of "
            "the last quarter",
        ),
        line(
            "Yours: sold to stations",
            lambda t: t.sold,
            "Your part of what they made that went into your stations, paid at its market price",
        ),
        line(
            "Yours: freight, about",
            lambda t: t.freight,
            "About what carrying your part on earns: its freight before what the buyer's "
            "satisfaction adds",
        ),
        line("Yours in all", lambda t: t.yours, ""),
    ]


def _flag_note(growth: Growth, rows: list[BuildingRow]) -> list[str]:
    owned = [
        f"{row.owned} shares of the {row.name}s"
        + (", which you control" if row.owned >= CONTROL * row.total else "")
        for row in rows
        if row.owned
    ]
    if not owned:
        return []
    return [f"{growth.owners.company} flies its flag here: you own {', '.join(owned)}."]


def mood_text(row: BuildingRow) -> str:
    """How satisfied a building is, with its satisfaction as a percentage where it trades."""
    if row.percent is None:
        return row.satisfaction
    return f"{row.satisfaction} {row.percent:.0%}"


def _deposit_note(tile_map: TileMap, cell: Cell) -> str:
    """Names the deposit under a cell, if there is one."""
    if tile_map.deposits is None or tile_map.deposits[cell] in (Deposit.NONE, Deposit.GARRISON):
        return ""
    return f"; {Deposit(int(tile_map.deposits[cell])).name.lower()} deposit"


def _stock(growth: Growth, kind: BuildingKind, flat: int) -> list[tuple[Need, float]]:
    """What a kind of processor holds of each input, waiting to be used."""
    stockpiles = growth.economy.stock
    return [
        (need, float(stockpiles[(kind, need)][flat]) if (kind, need) in stockpiles else 0.0)
        for need in RECIPES[kind].per_unit
    ]


def _split_bonus_part(
    kind: BuildingKind, count: int, got: list[Flow], bonus_got: list[Flow]
) -> tuple[list[Flow], list[Flow]]:
    """Moves the part of a need a kind takes only as a bonus (BONUS_PART), such as a home's second
    half of food, out of what it must get and into its bonus goods."""
    must, extras = [], []
    for flow in got:
        spare = BONUS_PART.get((kind, flow.good), 0.0) * count  # type: ignore[arg-type]
        if not spare:
            must.append(flow)
            continue
        wanted = max(flow.wanted - spare, 0.0)
        must.append(Flow(flow.good, min(flow.done, wanted), wanted))
        extras.append(Flow(flow.good, min(max(flow.done - wanted, 0.0), spare), spare))
    return must, extras + bonus_got


def _per_good(
    goods: Mapping[tuple[BuildingKind, Good], tuple[FloatArray, FloatArray]]
    | Mapping[tuple[BuildingKind, Need], tuple[FloatArray, FloatArray]],
    kind: BuildingKind,
    position: int,
    most: Mapping[Good, float] | None = None,
    count: int = 1,
) -> list[Flow]:
    """How much of each good the buildings got or gave, out of what they needed or made; for
    goods made, also the most they could make with all their inputs."""
    flows = []
    for (of, good), (done, wanted) in goods.items():
        if of != kind or (wanted[position] <= 0 and not most):
            continue
        limit = most.get(good, 0) * count if most else 0
        flows.append(
            Flow(
                good,
                float(done[position]),
                float(wanted[position]),
                limit if limit > wanted[position] + EPSILON else None,
            )
        )
    return flows


def _other(construction: Construction, growth: Growth) -> str:
    match construction:
        case Station():
            if growth.railway is None:
                return "station"
            return f"station: {growth.railway.name_of(construction.cell)}"
        case Annex():
            if growth.railway is None:
                return construction.facility.value
            name = growth.railway.name_of(construction.station)
            return f"{construction.facility.value} of {name}"
        case Track():
            kind = (
                ", maglev" if construction.maglev else ", electric" if construction.electric else ""
            )
            return (
                f"track ({construction.width.name.lower()}, {construction.bed.name.lower()}{kind})"
            )
    return str(construction)


@dataclass(frozen=True)
class BuildingRow:
    name: str
    count: int
    makes: list[tuple[Good, float]]
    # What it must get to be satisfied, and the bonus goods that raise it above that.
    needs: list[tuple[Need, float]]
    bonus: list[tuple[Need, float]]
    got: list[Flow]
    gave: list[Flow]
    # What processors hold of each input, waiting to be used.
    stock: list[tuple[Need, float]]
    satisfaction: str
    kind: BuildingKind | None = None
    # How many shares of them the player owns, and what LOT shares cost or fetch now; None for
    # what cannot be bought.
    owned: int = 0
    price: float | None = None
    # Its satisfaction last month, past 1 with bonus goods; None where it does not trade.
    percent: float | None = None
    # How the bonus goods and boosts it can do without went; `got` holds what it must get.
    bonus_got: list[Flow] = field(default_factory=list)
    # How many of the player's shares may be sold now, and the year the next held back may be.
    sellable: int = 0
    sell_from: int | None = None
    # How many shares they are in all, SHARES each; what they took in money last year (None
    # before a whole year of trade) and what they would at the pace of the last quarter.
    total: int = 0
    last_year: Takings | None = None
    forecast: Takings | None = None

    @property
    def held_back(self) -> int:
        """How many of the player's shares may not be sold yet."""
        return self.owned - self.sellable
