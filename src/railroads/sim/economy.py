"""Trade: how the goods of railroads.sim.goods reach the buildings that need them each month, and
how satisfied that leaves each kind of building in each cell.

Each processor keeps a stockpile of up to INPUT_STOCK of every input for every building, taking in
what reaches it even while it waits for another input, and works from it.

Buildings hand out what they make on their own first, everywhere on the map (LOCAL_REACH): every
cell's spare goods go first to the buildings in the same cell, then to those one cell away, and so
on outwards up to LOCAL_REACH cells. Nothing is lost on the way, but a cell further away can take
less of a seller's month (reach_share): all of it from the same cell, then 13, 8, 5, 3, 2 and 1
twenty-firsts of it from one to six cells away (REACH_SHARES). Buildings the same distance from a
seller share what it has by how much they still want.

Stations come after (railroads.sim.railway), those joined by track to another station: each hands
what trains brought to buildings within its delivery radius still short of a need, unsatisfied
ones first, then content ones, nearest first, unless the player keeps the good for trains there
(StationOrder.kept). Once every need has been met, each station takes the spare goods no building
took that buildings made within its collection radius into its stockpile, while it has room, for
trains to carry away; it never hands them out around it.

Nothing is made that has nowhere to go: a processor uses up only the inputs for what left it; the
rest waits in its stockpile. Needs are handed out in the order of PIPELINE, so the inputs of a
processor come before what it makes: canneries take their fruit before houses eat what is left.

A building's satisfaction is the share of what it demands that it got and of what it makes that
reached a buyer; a processor's is only the share of all it could make that reached a buyer, so one
whose stockpile holds one input but not the other is unsatisfied however full it is, plus
BONUS_PER_GOOD for each of its bonus goods (railroads.sim.goods) it got all of, and that share of
it for part: a house fed and warmed that also got all its beer and half its newspapers is at 145%.
Food is a must for homes: one that got none is unsatisfied whatever else it got, so homes can't
live on firewood or beer alone. A building is unsatisfied at nothing, thriving from 100% and
content between. Its satisfaction adds to the
chances of more growing in its cell, its points times its satisfaction, with no ceiling above
100%.
"""

import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import cache

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import START_YEAR
from railroads.sim.constructions import GREW_FROM, BuildingKind
from railroads.sim.goods import (
    BOOST_MOST,
    HANDED_OUT,
    HOMES,
    LAND_BOOST_MOST,
    MAKES,
    NEEDS,
    PIPELINE,
    RECIPES,
    Good,
    Need,
    bonus_of,
    boosts_of,
    makes_of,
    needs_of,
)
from railroads.sim.grids import Cell, Grid, flat_within_any
from railroads.sim.steps import Steps, finish
from railroads.sim.tilemap import TileMap

FloatCells = npt.NDArray[np.float64]


class Satisfaction(IntEnum):
    UNSATISFIED = 0
    CONTENT = 1
    THRIVING = 2


DELIVERY_RADIUS = 2

COLLECTION_RADIUS = 3

# The most of a seller's month a cell can take, by how many cells away from it it is: all of it in
# the seller's own cell, then falling by the Fibonacci numbers, 13, 8, 5, 3, 2 and 1 twenty-firsts,
# and nothing from LOCAL_REACH + 1 cells away.
REACH_SHARES = (21 / 21, 13 / 21, 8 / 21, 5 / 21, 3 / 21, 2 / 21, 1 / 21)
LOCAL_REACH = len(REACH_SHARES) - 1
# The cells of -1 around the map that let _Month._pairs look LOCAL_REACH cells out from any cell
# without checking for the edge.
_PAD = LOCAL_REACH


def reach_share(away: int) -> float:
    """The most of a seller's month a cell `away` cells from it can take."""
    return REACH_SHARES[away] if 0 <= away <= LOCAL_REACH else 0.0


# About how long a month's trade works before it pauses for the next frame.
STEP_SECONDS = 0.008

# The most of each input a processor keeps in stock, for every building; a big or mega works
# keeps as much more as it makes (input_stock).
INPUT_STOCK = 300.0


@cache
def input_stock(kind: BuildingKind) -> float:
    """The most of each input one processor of a kind keeps in stock."""
    return INPUT_STOCK * RECIPES[kind].most / RECIPES[GREW_FROM.get(kind, kind)].most


# How much each bonus good a building gets all of adds to its satisfaction.
BONUS_PER_GOOD = 0.3


def levels_of(share: FloatCells) -> npt.NDArray[np.int8]:
    """Satisfaction for a share of needs met or goods given away."""
    levels = np.full(share.shape, Satisfaction.CONTENT, dtype=np.int8)
    levels[share <= 0.0] = Satisfaction.UNSATISFIED
    levels[share >= 1.0 - 1e-9] = Satisfaction.THRIVING
    return levels


class Neighbourhoods:
    """Every cell's neighbours, as flat indices."""

    def __init__(self, grid: Grid) -> None:
        rows, cols = grid.neighbour_index_grids()
        own = np.indices(grid.shape)
        real = ((rows != own[0]) | (cols != own[1])).reshape(len(rows), -1)
        index = np.ravel_multi_index((rows, cols), grid.shape).reshape(len(rows), -1)
        # Off the map, a neighbour points at one extra cell past the last.
        self.index = np.where(real, index, grid.rows * grid.cols)


@dataclass(frozen=True, slots=True, eq=False)
class Rings:
    """The steps to every cell up to LOCAL_REACH cells away, nearest first, as row and column
    steps for even rows and for odd rows, and where each distance starts among them."""

    rows: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    cols: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    starts: npt.NDArray[np.intp]


@cache
def rings_of(grid: Grid) -> Rings:
    reach = LOCAL_REACH
    middle = 2 * reach + 2
    roomy = replace(grid, rows=2 * middle + 2, cols=2 * middle + 2)
    rows, cols, away = [], [], None
    for row in (middle, middle + 1):
        steps = sorted(roomy.within((row, middle), reach).items(), key=lambda item: item[1])
        rows.append(np.array([r - row for (r, _), _ in steps], dtype=np.intp))
        cols.append(np.array([c - middle for (_, c), _ in steps], dtype=np.intp))
        away = np.array([d for _, d in steps], dtype=np.intp)
    assert away is not None
    starts = np.searchsorted(away, np.arange(LOCAL_REACH + 2))
    return Rings((rows[0], rows[1]), (cols[0], cols[1]), starts)


@dataclass(frozen=True)
class StationOrder:
    """What one station brings to a month's trade: goods trains left there to hand out, and the
    most of each good it holds, gathered and brought together."""

    brought: Mapping[Good, float]
    limit: float
    # How many cells out it hands goods out to, and gathers them from.
    delivery: int = DELIVERY_RADIUS
    collection: int = COLLECTION_RADIUS
    # Freight it gathered before, which takes up its room.
    stockpile: Mapping[Good, float] = field(default_factory=dict)
    # Goods kept for trains: none of what trains brought of them is handed out around it.
    kept: frozenset[Good] = frozenset()
    # The platforms added to it, whose cells it reaches out from as well as its own.
    platforms: tuple[Cell, ...] = ()


# What Trade.share works out, by the trade's id and kinds, while the trade lasts: a trade is
# saved with the game, so it keeps none of it itself.
_SHARES: dict[int, dict[tuple[BuildingKind, ...], tuple[FloatCells, npt.NDArray[np.bool_]]]] = {}


@dataclass(frozen=True)
class Trade:
    """One month of trade on the cells with buildings (`cells`, flat indices): per kind of
    building and cell, how much of its demands it got and how much of what it can make it gave
    away, both summed over goods, and how many of its bonus goods it got, each counted by the
    share of it that came."""

    cells: npt.NDArray[np.intp]
    got: Mapping[BuildingKind, FloatCells]
    needed: Mapping[BuildingKind, FloatCells]
    gave: Mapping[BuildingKind, FloatCells]
    could_make: Mapping[BuildingKind, FloatCells]
    # Per station, the goods it handed out and the goods it gathered.
    handed: Mapping[Cell, Mapping[Good, float]] = field(default_factory=dict)
    gathered: Mapping[Cell, Mapping[Good, float]] = field(default_factory=dict)
    # Per kind of building and need: what it took in this month, from anywhere, and what it
    # needs; per kind and good: what it gave away and the most it could make.
    goods_got: Mapping[tuple[BuildingKind, Need], tuple[FloatCells, FloatCells]] = field(
        default_factory=dict
    )
    goods_gave: Mapping[tuple[BuildingKind, Good], tuple[FloatCells, FloatCells]] = field(
        default_factory=dict
    )
    # Per station, how much of what trains brought it handed out, by need and good.
    handed_for: Mapping[Cell, Mapping[tuple[Need, Good], float]] = field(default_factory=dict)
    bonus: Mapping[BuildingKind, FloatCells] = field(default_factory=dict)
    # What each bonus good a building gets all of adds to its satisfaction, with the player's tech.
    bonus_per_good: float = BONUS_PER_GOOD
    # Per need, what each cell's buildings got of it from sellers nearby rather than stations.
    local: Mapping[Need, FloatCells] = field(default_factory=dict)

    def done(self, kinds: tuple[BuildingKind, ...]) -> FloatCells:
        return sum(
            (self.got[k] + self.gave[k] for k in kinds if k in self.got),
            start=np.zeros(self.cells.size),
        )

    def wanted(self, kinds: tuple[BuildingKind, ...]) -> FloatCells:
        return sum(
            (self.needed[k] + self.could_make[k] for k in kinds if k in self.got),
            start=np.zeros(self.cells.size),
        )

    def share(self, kinds: tuple[BuildingKind, ...], empty: float = 1.0) -> FloatCells:
        """The satisfaction of each cell's buildings of these kinds, from 0 up and past 1 with
        bonus goods. Where kinds share a cell, each counts by how much it wants; cells where they
        want nothing get `empty`."""
        share, wanting = self._shares(kinds)
        return np.where(wanting, share, empty)

    def _shares(self, kinds: tuple[BuildingKind, ...]) -> tuple[FloatCells, npt.NDArray[np.bool_]]:
        """share before cells that want nothing get theirs, and those that want anything,
        worked out once for each trade and kinds and kept in _SHARES while the trade lasts."""
        kept = _SHARES.get(id(self))
        if kept is None:
            kept = _SHARES[id(self)] = {}
            weakref.finalize(self, _SHARES.pop, id(self), None)
        found = kept.get(kinds)
        if found is not None:
            return found
        total, weight = np.zeros(self.cells.size), np.zeros(self.cells.size)
        for kind in kinds:
            if kind not in self.got:
                continue
            wanted = self.needed[kind] + self.could_make[kind]
            done = self.got[kind] + self.gave[kind]
            own = np.where(wanted > 0, done / np.where(wanted > 0, wanted, 1.0), 0.0)
            if kind in self.bonus:
                own = own + self.bonus_per_good * self.bonus[kind]
            total += wanted * own
            weight += wanted
        wanting = weight > 0
        found = kept[kinds] = (total / np.where(wanting, weight, 1.0), wanting)
        return found

    def levels(self, kinds: tuple[BuildingKind, ...] | None = None) -> npt.NDArray[np.int8]:
        """Satisfaction of each cell's buildings of these kinds, or of all of them; cells with
        none count as thriving."""
        return levels_of(self.share(kinds if kinds is not None else tuple(self.needed)))


# Every processor's kind and each input it ever takes in.
STOCKPILES = tuple(
    (kind, need) for kind, recipe in RECIPES.items() for need in Need if need in recipe.every_need
)


class Stockpiles(dict[tuple[BuildingKind, Need], FloatCells]):
    """What the processors of each kind in each cell hold of each input: a map-sized array for
    each kind and input some of them hold, and none for the rest, which holds nothing anywhere:
    asked for, it gets an empty one."""

    def __init__(self, cells: int) -> None:
        super().__init__()
        self.cells = cells

    def __missing__(self, key: tuple[BuildingKind, Need]) -> FloatCells:
        stock = self[key] = np.zeros(self.cells)
        return stock


class Economy:
    """Lets stations hand goods out on one map each month, on the cells with buildings that
    trade, and keeps every processor's stockpile of inputs from month to month."""

    # What buildings that are no processors took this month as goods reached a station by them,
    # by kind and need, then cell: the month's trade counts it as got and hands out only the rest.
    # None in saves from before stations handed goods out at once.
    taken: dict[tuple[BuildingKind, Need], dict[int, float]] | None = None

    def __init__(self, tile_map: TileMap) -> None:
        self.tile_map = tile_map
        self.cells = tile_map.grid.rows * tile_map.grid.cols
        self.neighbourhoods = Neighbourhoods(tile_map.grid)
        self.kinds = tuple(dict.fromkeys([*MAKES, *NEEDS]))
        # Each cell's satisfaction last month, which decides who is served first.
        self.levels = np.full(self.cells, Satisfaction.UNSATISFIED, dtype=np.int8)
        # The year, which decides what buildings need, and what each bonus good a building gets
        # adds to its satisfaction (railroads.sim.growth sets it from the player's tech).
        self.year = START_YEAR
        self.bonus_per_good = BONUS_PER_GOOD
        # What the processors of each kind in each cell hold of each input.
        self.stock = Stockpiles(self.cells)

    def trade(
        self,
        counts: Mapping[BuildingKind, npt.NDArray[np.int64]],
        stations: Mapping[Cell, StationOrder] | None = None,
        cells: npt.NDArray[np.intp] | None = None,
    ) -> Trade:
        """Lets the stations hand out and gather one month of goods among the buildings counted
        in each cell, and the buildings sell what is left to each other. `cells`, the flat cells
        with any buildings, saves finding them again."""
        return finish(self.trade_steps(counts, stations, cells))

    def trade_steps(
        self,
        counts: Mapping[BuildingKind, npt.NDArray[np.int64]],
        stations: Mapping[Cell, StationOrder] | None = None,
        cells: npt.NDArray[np.intp] | None = None,
    ) -> Steps[Trade]:
        """trade, pausing about every STEP_SECONDS; nothing else may change the game between."""
        return _Month(self, counts, stations or {}, cells).run_steps()

    def _around(
        self,
        station: Cell,
        cells: npt.NDArray[np.intp],
        radius: float,
        platforms: tuple[Cell, ...] = (),
    ) -> tuple[npt.NDArray[np.intp], FloatCells]:
        """The traded cells within `radius` steps of a station or any of its platforms, as
        positions among `cells`, nearest first, and how many steps away each is."""
        within = flat_within_any(self.tile_map.grid, (station, *platforms), int(radius))
        if not cells.size:
            return cells, np.zeros(0)
        flat = within.cells
        positions = np.searchsorted(cells, flat)
        found = (positions < cells.size) & (cells[np.minimum(positions, cells.size - 1)] == flat)
        positions, steps = positions[found], within.steps[found]
        order = np.argsort(steps, kind="stable")
        return positions[order], steps[order]

    @staticmethod
    def _hand_out(
        amount: float,
        near: tuple[npt.NDArray[np.intp], FloatCells],
        demand: FloatCells,
        received: FloatCells,
        priority: npt.NDArray[np.int8],
    ) -> float:
        """Hands goods out to the cells still short of them, the least satisfied first and then
        the nearest; returns how much went."""
        positions, _ = near
        order = positions[np.argsort(priority[positions], kind="stable")]
        left = amount
        for position in order.tolist():
            if left <= 0:
                break
            give = min(left, demand[position] - received[position])
            if give > 0:
                received[position] += give
                left -= give
        return amount - left

    @staticmethod
    def _gather(
        room: float,
        far: tuple[npt.NDArray[np.intp], FloatCells],
        supply: FloatCells,
        made: FloatCells,
        sold: FloatCells,
    ) -> float:
        """Takes spare goods that buildings made from the cells around, evenly, up to the room
        left; returns how much was taken."""
        positions, _ = far
        spare = np.maximum(supply[positions] - sold[positions], 0.0)
        spare = np.minimum(spare, made[positions])
        total = float(spare.sum())
        if total <= 0 or room <= 0:
            return 0.0
        take = min(room, total)
        sold[positions] += spare * (take / total)
        return take


@cache
def wants_of(kind: BuildingKind, year: int) -> frozenset[Need]:
    """The needs a kind wants in a given year: what a processor takes in or is boosted by, and
    what the rest need."""
    recipe = RECIPES.get(kind)
    if recipe is None:
        return frozenset(needs_of(kind, year))
    return frozenset(recipe.takes(year)) | frozenset(boosts_of(kind, year))


class _Month:
    """One month of trade on the cells with buildings, worked out need by need in the order of
    the chain: what each building is offered, takes in, makes and gives away."""

    def __init__(
        self,
        economy: Economy,
        counts: Mapping[BuildingKind, npt.NDArray[np.int64]],
        stations: Mapping[Cell, StationOrder],
        cells: npt.NDArray[np.intp] | None = None,
    ) -> None:
        self.economy = economy
        self.stations = stations
        flat = {k: counts[k].ravel() for k in economy.kinds if k in counts}
        if cells is None:
            built = np.zeros(economy.cells, dtype=bool)
            for n in flat.values():
                built |= n > 0
            cells = np.flatnonzero(built)
        self.cells = cells
        in_cells = ((k, n[cells]) for k, n in flat.items())
        self.count = {k: n.astype(np.float64) for k, n in in_cells if n.any()}
        self.wanted = {kind: wants_of(kind, economy.year) for kind in self.count}
        self.zero = np.zeros(cells.size)
        self.priority = economy.levels[cells]
        # What each kind of processor holds of each input, worked out as the trade begins.
        self.stock: dict[tuple[BuildingKind, Need], FloatCells] = {}
        # What buildings offer of each good, by kind and in all, and how much of it went.
        self.offers: dict[tuple[BuildingKind, Good], FloatCells] = {}
        self.offered = {good: self.zero.copy() for good in Good}
        self.sold = {good: self.zero.copy() for good in Good}
        # Per need, what buildings got of it from sellers nearby rather than stations.
        self.local: dict[Need, FloatCells] = {}
        # Where each traded cell stands among them, -1 for the rest, and per good the pairs of
        # sellers and buyers near enough to trade.
        self.position = np.full(economy.cells, -1, dtype=np.intp)
        self.position[cells] = np.arange(cells.size)
        self.pairs: dict[Good, tuple[npt.NDArray[np.intp], ...]] = {}
        self.padded: tuple[npt.NDArray[np.intp], tuple[npt.NDArray[np.intp], ...]] | None = None
        # What buildings had taken as goods reached their station when their need was served:
        # what they take after that, while the month's trade is still going, counts for the next.
        self.counted: dict[tuple[BuildingKind, Need], dict[int, float]] = {}
        for kind in MAKES:
            if kind in RECIPES or kind not in self.count:
                continue
            for good, amount in makes_of(kind, economy.year).items():
                self.offers[(kind, good)] = self.count[kind] * amount
                self.offered[good] += self.count[kind] * amount
        # Per kind of processor, what it makes from its inputs alone, and the share it has of each
        # of its boosts; per kind making goods from the land, the share it has of its boosts.
        self.uses: dict[BuildingKind, dict[Need, FloatCells]] = {}
        self.raw_boost: dict[BuildingKind, FloatCells] = {}
        self.received: dict[tuple[BuildingKind, Need], FloatCells] = {}
        self.brought = {station: dict(order.brought) for station, order in stations.items()}
        self.handed: dict[Cell, dict[Good, float]] = {station: {} for station in stations}
        self.handed_for: dict[Cell, dict[tuple[Need, Good], float]] = {
            station: {} for station in stations
        }
        self.gathered: dict[Cell, dict[Good, float]] = {station: {} for station in stations}
        self.room = {
            station: {
                good: max(
                    0.0, order.limit - order.brought.get(good, 0.0) - order.stockpile.get(good, 0.0)
                )
                for good in Good
            }
            for station, order in stations.items()
        }
        self.reach = {
            station: (
                economy._around(station, cells, order.delivery, order.platforms),
                economy._around(station, cells, order.collection, order.platforms),
            )
            for station, order in stations.items()
        }

    def run_steps(self) -> Steps[Trade]:
        yield
        economy, cells = self.economy, self.cells
        self.stock = {
            (kind, need): np.minimum(
                economy.stock[(kind, need)][cells],
                input_stock(kind) * self.count.get(kind, self.zero),
            )
            if (kind, need) in economy.stock
            else np.zeros(cells.size)
            for kind, need in STOCKPILES
        }
        yield
        paused = time.perf_counter()
        for step in PIPELINE:
            if isinstance(step, BuildingKind):
                self._produce(step)
                continue
            self._serve(step)
            if time.perf_counter() - paused > STEP_SECONDS:
                yield
                paused = time.perf_counter()
        for good in Good:
            self._gather(good)
        return (yield from self._settle())

    def _wants(self, need: Need) -> dict[BuildingKind, FloatCells]:
        """What each kind wants of a need: consumers what they use in a month, processors the
        room left in their stockpile."""
        wants = {}
        for kind, n in self.count.items():
            if need not in self.wanted[kind]:
                continue
            if kind in RECIPES:
                wants[kind] = np.maximum(input_stock(kind) * n - self.stock[(kind, need)], 0.0)
            else:
                wants[kind] = n * NEEDS[kind][need]
        return wants

    def _taken(self, kind: BuildingKind, need: Need) -> FloatCells:
        """What buildings of a kind took of a need this month as goods reached their station."""
        taken = self.zero.copy()
        for cell, amount in ((self.economy.taken or {}).get((kind, need)) or {}).items():
            position = self.position[cell]
            if position >= 0:
                taken[position] += amount
        return taken

    def _serve(self, need: Need) -> None:
        """Lets the goods meeting a need reach the buildings wanting it, the most valuable first:
        first what buildings hand out on their own, then what trains brought to stations, within
        their delivery radius, unless the good is kept for trains. Buildings want only what they
        did not take already this month."""
        wants = self._wants(need)
        if not wants:
            return
        early = {kind: self._taken(kind, need) for kind in wants if kind not in RECIPES}
        for kind in early:
            self.counted[(kind, need)] = dict((self.economy.taken or {}).get((kind, need)) or {})
        for kind, taken in early.items():
            wants[kind] = np.maximum(wants[kind] - taken, 0.0)
        demand = sum(wants.values(), start=self.zero.copy())
        received = self.zero.copy()
        for good in HANDED_OUT[need]:
            self._sell_nearby(good, demand, received)
        self.local[need] = received.copy()
        for station in self.stations:
            near, _ = self.reach[station]
            for good in HANDED_OUT[need]:
                brought = self.brought[station].get(good, 0.0)
                if brought <= 0 or good in self.stations[station].kept:
                    continue
                given = self.economy._hand_out(brought, near, demand, received, self.priority)
                if given <= 0:
                    continue
                self.brought[station][good] = brought - given
                self.handed[station][good] = self.handed[station].get(good, 0.0) + given
                handed_for = self.handed_for[station]
                handed_for[(need, good)] = handed_for.get((need, good), 0.0) + given
        met = np.where(demand > 0, received / np.where(demand > 0, demand, 1.0), 0.0)
        for kind, want in wants.items():
            got = met * want
            key = (kind, need)
            got = got + early.get(kind, 0.0)
            self.received[key] = self.received.get(key, self.zero) + got
            if kind in RECIPES:
                self.stock[key] = self.stock[key] + got
            else:
                self._boost_from_land(kind, need, got)

    def _taken_since_served(self) -> dict[tuple[BuildingKind, Need], dict[int, float]]:
        """What buildings took as goods reached their station after their need was served this
        month, which the next month counts as taken."""
        later: dict[tuple[BuildingKind, Need], dict[int, float]] = {}
        for key, taken in (self.economy.taken or {}).items():
            counted = self.counted.get(key, {})
            left = {
                cell: amount - counted.get(cell, 0.0)
                for cell, amount in taken.items()
                if amount - counted.get(cell, 0.0) > 1e-9
            }
            if left:
                later[key] = left
        return later

    def _boost_from_land(self, kind: BuildingKind, need: Need, got: FloatCells) -> None:
        """Lets what a kind making goods from the land got of one of its boosts make it make more
        of each of its goods, before any of them is sold: up to its own LAND_BOOST_MOST more, or
        its part of BOOST_MOST shared with the other boosts."""
        boosts = boosts_of(kind, self.economy.year)
        if need not in boosts:
            return
        wanted = self.count[kind] * boosts[need]
        share = np.minimum(got / np.where(wanted > 0, wanted, 1.0), 1.0) * (wanted > 0)
        shared = [boost for boost in boosts if boost not in LAND_BOOST_MOST]
        most = LAND_BOOST_MOST.get(need, BOOST_MOST / max(len(shared), 1))
        before = self.raw_boost.get(kind, self.zero)
        self.raw_boost[kind] = before + share * most
        for good, amount in makes_of(kind, self.economy.year).items():
            more = self.count[kind] * amount * most * share
            self.offers[(kind, good)] = self.offers[(kind, good)] + more
            self.offered[good] = self.offered[good] + more

    def _sell_nearby(self, good: Good, demand: FloatCells, received: FloatCells) -> None:
        """Lets the cells with spare goods hand them out to the buildings still short of them, the
        nearest first, each cell taking from a seller at most its reach_share of the seller's
        month."""
        spare = np.maximum(self.offered[good] - self.sold[good], 0.0)
        left = demand - received
        if not (spare > 1e-9).any() or not (left > 1e-9).any():
            return
        sellers, buyers, starts = self._pairs(good)
        size = self.cells.size
        remaining = spare.copy()
        month = self.offered[good]
        for away in range(LOCAL_REACH + 1):
            ring = slice(starts[away], starts[away + 1])
            seller, buyer = sellers[ring], buyers[ring]
            wanting = left[buyer]
            live = (wanting > 1e-9) & (remaining[seller] > 1e-9)
            if not live.any():
                continue
            seller, buyer, wanting = seller[live], buyer[live], wanting[live]
            share = wanting / np.bincount(seller, wanting, minlength=size)[seller]
            sent = np.minimum(remaining[seller] * share, month[seller] * reach_share(away))
            arriving = np.bincount(buyer, sent, minlength=size)[buyer]
            sent *= np.minimum(1.0, left[buyer] / arriving)
            remaining -= np.bincount(seller, sent, minlength=size)
            left -= np.bincount(buyer, sent, minlength=size)
            if remaining.max() <= 1e-9 or left.max() <= 1e-9:
                break
        self.sold[good] += spare - remaining
        received[:] = demand - left

    def _pairs(self, good: Good) -> tuple[npt.NDArray[np.intp], ...]:
        """Every cell making a good with every traded cell up to LOCAL_REACH cells from it, as
        positions among the traded cells, the nearest pairs first, and where the pairs of
        each distance start among them."""
        known = self.pairs.get(good)
        if known is not None:
            return known
        grid = self.economy.tile_map.grid
        rings = rings_of(grid)
        sellers = np.flatnonzero(self.offered[good] > 0)
        padded, offsets = self._padded()
        rows, cols = np.divmod(self.cells[sellers], grid.cols)
        at = (rows + _PAD) * (grid.cols + 2 * _PAD) + cols + _PAD
        reached = at + np.where(rows % 2 == 1, offsets[1][:, None], offsets[0][:, None])
        buyers = padded[reached]
        step, seller = np.nonzero(buyers >= 0)
        found = (sellers[seller], buyers[step, seller], np.searchsorted(step, rings.starts))
        self.pairs[good] = found
        return found

    def _padded(self) -> tuple[npt.NDArray[np.intp], tuple[npt.NDArray[np.intp], ...]]:
        """Where each traded cell stands among them on the map with _PAD cells of -1 around it,
        as flat indices, and the flat steps on it to every cell up to LOCAL_REACH cells away from
        a cell on an even and an odd row, nearest first."""
        if self.padded is None:
            grid = self.economy.tile_map.grid
            rings = rings_of(grid)
            width = grid.cols + 2 * _PAD
            padded = np.full((grid.rows + 2 * _PAD, width), -1, dtype=np.intp)
            padded[_PAD:-_PAD, _PAD:-_PAD] = self.position.reshape(grid.shape)
            offsets = tuple(rings.rows[odd] * width + rings.cols[odd] for odd in (0, 1))
            self.padded = (padded.ravel(), offsets)
        return self.padded

    def _gather(self, good: Good) -> None:
        """Lets every station take the spare units of a good no building took into its
        stockpile, for trains. Where nothing is left or a station has no room, it takes
        nothing."""
        offered = self.offered[good]
        if not offered.any():
            return
        for station in self.stations:
            room = self.room[station][good]
            if room <= 0:
                continue
            _, far = self.reach[station]
            taken = self.economy._gather(room, far, offered, offered, self.sold[good])
            if taken > 0:
                self.gathered[station][good] = self.gathered[station].get(good, 0.0) + taken
                self.room[station][good] -= taken

    def _produce(self, kind: BuildingKind) -> None:
        """Offers what a kind of processor could make from its stockpile: what its inputs allow,
        the inputs standing in for others first as they make the most, and up to the recipe's
        boost_most more by the share it has of its boosts. Keeps how much of each input all of
        that takes."""
        if kind not in self.count:
            return
        recipe = RECIPES[kind]
        year = self.economy.year
        inputs = recipe.inputs(year)
        room = self.count[kind] * recipe.most
        left = {need: self.stock[(kind, need)].copy() for need in inputs}
        uses = {need: self.zero.copy() for need in recipe.takes(year)}
        can = self.zero.copy()
        for swapped, (swap, times) in recipe.swaps_in(year).items():
            made = room.copy()
            for need, per_unit in inputs.items():
                held = self.stock[(kind, swap)] if need == swapped else left[need]
                made = np.minimum(made, held / (per_unit / times))
            for need, per_unit in inputs.items():
                taken = made * per_unit / times
                uses[swap if need == swapped else need] += taken
                if need != swapped:
                    left[need] -= taken
            can += made
            room = room - made
        plain = room
        for need, per_unit in inputs.items():
            plain = np.minimum(plain, left[need] / per_unit)
        for need, per_unit in inputs.items():
            uses[need] += plain * per_unit
        can += plain
        boosts = boosts_of(kind, year)
        if boosts:
            shares = {
                need: np.minimum(
                    self.stock[(kind, need)] / np.where(can > 0, can * recipe.boosts[need], 1.0),
                    1.0,
                )
                * (can > 0)
                for need in boosts
            }
            for need, share in shares.items():
                uses[need] = can * share * recipe.boosts[need]
            can = can * (1.0 + recipe.boost_most * sum(shares.values()) / len(shares))
        self.uses[kind] = uses
        self.offers[(kind, recipe.makes)] = can
        self.offered[recipe.makes] += can

    def _went(self, good: Good) -> FloatCells:
        """The share of all of a good on offer that went, to buildings or stations, in each
        cell."""
        offered = self.offered[good]
        on_offer = offered > 0
        went = np.where(on_offer, self.sold[good] / np.where(on_offer, offered, 1.0), 0.0)
        return np.minimum(went, 1.0)

    def _settle(self) -> Steps[Trade]:
        """Counts what every building got and gave, uses up the inputs of what the processors
        made, and keeps their stockpiles for next month, pausing once between."""
        got = {k: self.zero.copy() for k in self.count}
        needed = {k: self.zero.copy() for k in self.count}
        gave = {k: self.zero.copy() for k in self.count}
        could_make = {k: self.zero.copy() for k in self.count}
        bonus: dict[BuildingKind, FloatCells] = {}
        goods_got = {}
        for (kind, need), received in self.received.items():
            wanted = self.count[kind] * NEEDS[kind][need]
            goods_got[(kind, need)] = (received, wanted)
            if need in bonus_of(kind, self.economy.year):
                part = np.minimum(received, wanted) / np.where(wanted > 0, wanted, 1.0)
                bonus[kind] = bonus.get(kind, self.zero) + part
                continue
            if kind in RECIPES or need in boosts_of(kind, self.economy.year):
                continue
            got[kind] += np.minimum(received, wanted)
            needed[kind] += wanted
        for kind in HOMES:
            if kind not in self.count or Need.FOOD not in NEEDS.get(kind, {}):
                continue
            food = self.received.get((kind, Need.FOOD), self.zero)
            hungry = (self.count[kind] > 0) & (food <= 0)
            got[kind] = np.where(hungry, 0.0, got[kind])
            if kind in bonus:
                bonus[kind] = np.where(hungry, 0.0, bonus[kind])
        goods_gave = {}
        shares: dict[Good, FloatCells] = {}
        for (kind, good), offer in self.offers.items():
            if good not in shares:
                shares[good] = self._went(good)
            sent = given = offer * shares[good]
            most = self.count[kind] * MAKES[kind][good]
            goods_gave[(kind, good)] = (given, most)
            gave[kind] += given
            could_make[kind] += most
            recipe = RECIPES.get(kind)
            if recipe is not None:
                went = np.where(offer > 0, sent / np.where(offer > 0, offer, 1.0), 0.0)
                for need, amount in self.uses.get(kind, {}).items():
                    key = (kind, need)
                    self.stock[key] = np.maximum(self.stock[key] - went * amount, 0.0)
        yield
        stockpiles = self.economy.stock
        for key, held in self.stock.items():
            if not held.any():
                stockpiles.pop(key, None)
                continue
            stock = stockpiles[key]
            stock[:] = 0.0
            stock[self.cells] = held
        self.economy.taken = self._taken_since_served()
        trade = Trade(
            self.cells,
            got,
            needed,
            gave,
            could_make,
            self.handed,
            self.gathered,
            goods_got,
            goods_gave,
            self.handed_for,
            bonus,
            self.economy.bonus_per_good,
            self.local,
        )
        self.economy.levels[:] = Satisfaction.UNSATISFIED
        self.economy.levels[self.cells] = trade.levels()
        return trade
