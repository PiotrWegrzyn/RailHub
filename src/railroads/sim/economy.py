"""Trade: how the goods of railroads.sim.goods reach the buildings that need them each month, and
how satisfied that leaves each kind of building in each cell.

Each processor keeps a stockpile of up to INPUT_STOCK of every input for every building, taking in
what reaches it even while it waits for another input, and works from it.

Stations come first (railroads.sim.railway), those joined by track to another station. Before a
need is met, a station takes every spare unit of the goods meeting it that buildings made within
its collection radius into its stockpile, while it has room for that good. It then hands what
trains brought to buildings within its delivery radius, unsatisfied ones first, then content ones,
nearest first, and then its stockpile, freshly gathered goods included. Goods the player keeps for
trains at a station (StationOrder.kept) are not handed out at all, brought or gathered.

What stations leave, buildings sell on their own (LOCAL_REACH): every cell's spare goods go first
to the buildings in the same cell, then to those one cell away, and so on outwards, and each cell
of the way loses LOSS_PER_STEP of what was sent, so at LOCAL_REACH cells nothing arrives. Buildings
the same distance from a seller share what it has by how much they still want.

Nothing is made that has nowhere to go: a processor uses up only the inputs for what left it; the
rest waits in its stockpile. Needs are handed out in the order of PIPELINE, so the inputs of a
processor come before what it makes: canneries take their fruit before houses eat what is left.

A building's satisfaction is the share of what it demands that it got and of what it makes that
reached a buyer; a processor's is only the share of all it could make that reached a buyer, so one
whose stockpile holds one input but not the other is unsatisfied however full it is, plus
BONUS_PER_GOOD for each of its bonus goods (railroads.sim.goods) it got all of, and that share of
it for part: a house fed and warmed that also got all its beer and half its newspapers is at 145%.
It is unsatisfied at nothing, thriving from 100% and content between. Its satisfaction adds to the
chances of more growing in its cell, MOST_SERVED times its points at 100% and that much more above
it (served_multipliers).
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import cache

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import START_YEAR
from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import (
    BOOST_MOST,
    FITS,
    HANDED_OUT,
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
from railroads.sim.grids import Cell, Grid, flat_within
from railroads.sim.steps import Steps, finish
from railroads.sim.tilemap import TileMap

FloatCells = npt.NDArray[np.float64]


class Satisfaction(IntEnum):
    UNSATISFIED = 0
    CONTENT = 1
    THRIVING = 2


DELIVERY_RADIUS = 2

COLLECTION_RADIUS = 4

# How many cells away buildings sell their goods on their own, and the share of what was sent that
# each cell of the way loses: at LOCAL_REACH cells nothing arrives.
LOCAL_REACH = 10
LOSS_PER_STEP = 0.1

# About how long a month's trade works before it pauses for the next frame.
STEP_SECONDS = 0.008

# The most of each input a processor keeps in stock, for every building.
INPUT_STOCK = 300.0

# How many times over a building that gets all it needs and gives away all it makes counts its
# points towards more growing in its own cell, on top of the plain points every building gives
# its neighbourhood; one that manages part of it counts that share of this.
MOST_SERVED = 10.0
# How much each bonus good a building gets all of adds to its satisfaction.
BONUS_PER_GOOD = 0.3


def levels_of(share: FloatCells) -> npt.NDArray[np.int8]:
    """Satisfaction for a share of needs met or goods given away."""
    levels = np.full(share.shape, Satisfaction.CONTENT, dtype=np.int8)
    levels[share <= 0.0] = Satisfaction.UNSATISFIED
    levels[share >= 1.0 - 1e-9] = Satisfaction.THRIVING
    return levels


def served_multipliers(share: FloatCells) -> FloatCells:
    """How many times over buildings count their points as served, for their satisfaction."""
    return MOST_SERVED * np.maximum(share, 0.0)


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
    """The steps to every cell fewer than LOCAL_REACH cells away, nearest first, as row and
    column steps for even rows and for odd rows, and where each distance starts among them."""

    rows: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    cols: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    starts: npt.NDArray[np.intp]


@cache
def rings_of(grid: Grid) -> Rings:
    reach = LOCAL_REACH - 1
    middle = 2 * reach + 2
    roomy = replace(grid, rows=2 * middle + 2, cols=2 * middle + 2)
    rows, cols, away = [], [], None
    for row in (middle, middle + 1):
        steps = sorted(roomy.within((row, middle), reach).items(), key=lambda item: item[1])
        rows.append(np.array([r - row for (r, _), _ in steps], dtype=np.intp))
        cols.append(np.array([c - middle for (_, c), _ in steps], dtype=np.intp))
        away = np.array([d for _, d in steps], dtype=np.intp)
    assert away is not None
    starts = np.searchsorted(away, np.arange(LOCAL_REACH + 1))
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
    # Freight it gathered before, which it also hands out to the buildings around it.
    stockpile: Mapping[Good, float] = field(default_factory=dict)
    # Goods kept for trains: none of them is handed out around it, brought or gathered.
    kept: frozenset[Good] = frozenset()


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
    # Per station, the goods it handed out of its own stockpile, and how much of them by need.
    supplied: Mapping[Cell, Mapping[Good, float]] = field(default_factory=dict)
    supplied_for: Mapping[Cell, Mapping[tuple[Need, Good], float]] = field(default_factory=dict)
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
        return np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), empty)

    def levels(self, kinds: tuple[BuildingKind, ...] | None = None) -> npt.NDArray[np.int8]:
        """Satisfaction of each cell's buildings of these kinds, or of all of them; cells with
        none count as thriving."""
        return levels_of(self.share(kinds if kinds is not None else tuple(self.needed)))


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
        self.stock: dict[tuple[BuildingKind, Need], FloatCells] = {
            (kind, need): np.zeros(self.cells)
            for kind, recipe in RECIPES.items()
            for need in recipe.per_unit
        }

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
        self, station: Cell, cells: npt.NDArray[np.intp], radius: float
    ) -> tuple[npt.NDArray[np.intp], FloatCells]:
        """The traded cells within `radius` steps of a station, as positions among `cells`,
        nearest first, and how many steps away each is."""
        within = flat_within(self.tile_map.grid, station, int(radius))
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
        self.count = {k: n[cells].astype(np.float64) for k, n in flat.items() if n[cells].any()}
        self.zero = np.zeros(cells.size)
        self.priority = economy.levels[cells]
        for kind, recipe in RECIPES.items():
            for need in {**recipe.per_unit, **recipe.boosts}:
                economy.stock.setdefault((kind, need), np.zeros(economy.cells))
        self.stock = {
            (kind, need): np.minimum(
                economy.stock[(kind, need)][cells], INPUT_STOCK * self.count.get(kind, self.zero)
            )
            for kind, need in economy.stock
        }
        # What buildings offer of each good, by kind and in all, and how much of it went.
        self.offers: dict[tuple[BuildingKind, Good], FloatCells] = {}
        self.offered = {good: self.zero.copy() for good in Good}
        self.sold = {good: self.zero.copy() for good in Good}
        # Of what sellers sent to buildings nearby, what was lost on the way, and per need what
        # buildings got that way.
        self.lost = {good: self.zero.copy() for good in Good}
        self.local: dict[Need, FloatCells] = {}
        # Where each traded cell stands among them, -1 for the rest, and per good the pairs of
        # sellers and buyers near enough to trade.
        self.position = np.full(economy.cells, -1, dtype=np.intp)
        self.position[cells] = np.arange(cells.size)
        self.pairs: dict[Good, tuple[npt.NDArray[np.intp], ...]] = {}
        for kind in MAKES:
            if kind in RECIPES or kind not in self.count:
                continue
            for good, amount in makes_of(kind, economy.year).items():
                self.offers[(kind, good)] = self.count[kind] * amount
                self.offered[good] += self.count[kind] * amount
        # Per kind of processor, what it makes from its inputs alone, and the share it has of each
        # of its boosts; per kind making goods from the land, the share it has of its boosts.
        self.boosted: dict[BuildingKind, tuple[FloatCells, dict[Need, FloatCells]]] = {}
        self.raw_boost: dict[BuildingKind, FloatCells] = {}
        self.received: dict[tuple[BuildingKind, Need], FloatCells] = {}
        self.brought = {station: dict(order.brought) for station, order in stations.items()}
        self.stockpiled = {station: dict(order.stockpile) for station, order in stations.items()}
        self.supplied: dict[Cell, dict[Good, float]] = {station: {} for station in stations}
        self.supplied_for: dict[Cell, dict[tuple[Need, Good], float]] = {
            station: {} for station in stations
        }
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
                economy._around(station, cells, order.delivery),
                economy._around(station, cells, order.collection),
            )
            for station, order in stations.items()
        }

    def run_steps(self) -> Steps[Trade]:
        paused = time.perf_counter()
        for step in PIPELINE:
            if isinstance(step, BuildingKind):
                self._produce(step)
                continue
            for good in FITS[step]:
                self._gather(good)
            self._serve(step)
            if time.perf_counter() - paused > STEP_SECONDS:
                yield
                paused = time.perf_counter()
        return self._settle()

    def _wants(self, need: Need) -> dict[BuildingKind, FloatCells]:
        """What each kind wants of a need: consumers what they use in a month, processors the
        room left in their stockpile."""
        wants = {}
        for kind, n in self.count.items():
            recipe = RECIPES.get(kind)
            year = self.economy.year
            if recipe is not None and (
                need in recipe.inputs(year) or need in boosts_of(kind, year)
            ):
                wants[kind] = np.maximum(INPUT_STOCK * n - self.stock[(kind, need)], 0.0)
            elif recipe is None and need in needs_of(kind, self.economy.year):
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
        """Lets stations hand out the goods meeting a need, the most valuable first: of each
        good what trains brought, then their stockpile, unless the good is kept for trains. Buildings
        want only what they did not take already this month."""
        wants = self._wants(need)
        if not wants:
            return
        early = {kind: self._taken(kind, need) for kind in wants if kind not in RECIPES}
        for kind, taken in early.items():
            wants[kind] = np.maximum(wants[kind] - taken, 0.0)
        demand = sum(wants.values(), start=self.zero.copy())
        received = self.zero.copy()
        for station in self.stations:
            near, _ = self.reach[station]
            for good in HANDED_OUT[need]:
                if good in self.stations[station].kept:
                    continue
                brought = self.brought[station].get(good, 0.0)
                if brought > 0:
                    given = self.economy._hand_out(brought, near, demand, received, self.priority)
                    self.brought[station][good] = brought - given
                    self.handed[station][good] = self.handed[station].get(good, 0.0) + given
                    handed_for = self.handed_for[station]
                    handed_for[(need, good)] = handed_for.get((need, good), 0.0) + given
                held = self.stockpiled[station].get(good, 0.0)
                if held <= 0:
                    continue
                given = self.economy._hand_out(held, near, demand, received, self.priority)
                self.stockpiled[station][good] = held - given
                self.supplied[station][good] = self.supplied[station].get(good, 0.0) + given
                supplied_for = self.supplied_for[station]
                supplied_for[(need, good)] = supplied_for.get((need, good), 0.0) + given
        from_stations = received.copy()
        for good in HANDED_OUT[need]:
            self._sell_nearby(good, demand, received)
        self.local[need] = received - from_stations
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

    def _boost_from_land(self, kind: BuildingKind, need: Need, got: FloatCells) -> None:
        """Lets what a kind making goods from the land got of one of its boosts make it make more
        of each of its goods, before any of them is sold."""
        boosts = boosts_of(kind, self.economy.year)
        if need not in boosts:
            return
        wanted = self.count[kind] * boosts[need]
        share = np.minimum(got / np.where(wanted > 0, wanted, 1.0), 1.0) * (wanted > 0)
        share = share / len(boosts)
        before = self.raw_boost.get(kind, self.zero)
        self.raw_boost[kind] = before + share
        for good, amount in makes_of(kind, self.economy.year).items():
            more = self.count[kind] * amount * BOOST_MOST * share
            self.offers[(kind, good)] = self.offers[(kind, good)] + more
            self.offered[good] = self.offered[good] + more

    def _sell_nearby(self, good: Good, demand: FloatCells, received: FloatCells) -> None:
        """Lets the cells with spare goods sell them to the buildings still short of them, the
        nearest first, losing LOSS_PER_STEP of what is sent for every cell of the way."""
        spare = np.maximum(self.offered[good] - self.sold[good], 0.0)
        left = demand - received
        if not (spare > 1e-9).any() or not (left > 1e-9).any():
            return
        sellers, buyers, starts = self._pairs(good)
        size = self.cells.size
        remaining = spare.copy()
        lost = self.zero.copy()
        for away in range(LOCAL_REACH):
            keep = 1.0 - LOSS_PER_STEP * away
            ring = slice(starts[away], starts[away + 1])
            seller, buyer = sellers[ring], buyers[ring]
            wanting = left[buyer]
            live = (wanting > 1e-9) & (remaining[seller] > 1e-9)
            if not live.any():
                continue
            seller, buyer, wanting = seller[live], buyer[live], wanting[live]
            share = wanting / np.bincount(seller, wanting, minlength=size)[seller]
            sent = remaining[seller] * share
            arriving = np.bincount(buyer, sent * keep, minlength=size)[buyer]
            taken = np.minimum(1.0, left[buyer] / arriving)
            sent *= taken
            remaining -= np.bincount(seller, sent, minlength=size)
            lost += np.bincount(seller, sent * (1.0 - keep), minlength=size)
            left -= np.bincount(buyer, sent * keep, minlength=size)
            if remaining.max() <= 1e-9 or left.max() <= 1e-9:
                break
        self.sold[good] += spare - remaining
        self.lost[good] += lost
        received[:] = demand - left

    def _pairs(self, good: Good) -> tuple[npt.NDArray[np.intp], ...]:
        """Every cell making a good with every traded cell fewer than LOCAL_REACH cells from it,
        as positions among the traded cells, the nearest pairs first, and where the pairs of
        each distance start among them."""
        known = self.pairs.get(good)
        if known is not None:
            return known
        grid = self.economy.tile_map.grid
        rings = rings_of(grid)
        sellers = np.flatnonzero(self.offered[good] > 0)
        rows, cols = np.divmod(self.cells[sellers], grid.cols)
        odd = (rows % 2 == 1)[:, None]
        reached_rows = rows[:, None] + np.where(odd, rings.rows[1], rings.rows[0])
        reached_cols = cols[:, None] + np.where(odd, rings.cols[1], rings.cols[0])
        inside = (
            (reached_rows >= 0)
            & (reached_rows < grid.rows)
            & (reached_cols >= 0)
            & (reached_cols < grid.cols)
        )
        flat = np.where(inside, reached_rows * grid.cols + reached_cols, 0)
        buyers = np.where(inside, self.position[flat], -1)
        step, seller = np.nonzero((buyers >= 0).T)
        found = (sellers[seller], buyers[seller, step], np.searchsorted(step, rings.starts))
        self.pairs[good] = found
        return found

    def _gather(self, good: Good) -> None:
        """Lets every station take spare units of a good buildings made into its stockpile, where
        they can be handed out at once."""
        for station in self.stations:
            _, far = self.reach[station]
            taken = self.economy._gather(
                self.room[station][good],
                far,
                self.offered[good],
                self.offered[good],
                self.sold[good],
            )
            if taken > 0:
                self.gathered[station][good] = self.gathered[station].get(good, 0.0) + taken
                self.room[station][good] -= taken
                stockpiled = self.stockpiled[station]
                stockpiled[good] = stockpiled.get(good, 0.0) + taken

    def _produce(self, kind: BuildingKind) -> None:
        """Offers what a kind of processor could make from its stockpile: what its inputs allow,
        and up to BOOST_MOST more by the share it has of its boosts."""
        if kind not in self.count:
            return
        recipe = RECIPES[kind]
        can = self.count[kind] * recipe.most
        for need, per_unit in recipe.inputs(self.economy.year).items():
            can = np.minimum(can, self.stock[(kind, need)] / per_unit)
        boosts = boosts_of(kind, self.economy.year)
        if boosts:
            shares = {
                need: np.minimum(
                    self.stock[(kind, need)] / np.where(can > 0, can * recipe.boosts[need], 1.0),
                    1.0,
                )
                * (can > 0)
                for need in boosts
            }
            self.boosted[kind] = (can, shares)
            can = can * (1.0 + BOOST_MOST * sum(shares.values()) / len(shares))
        self.offers[(kind, recipe.makes)] = can
        self.offered[recipe.makes] += can

    def _given(self, offer: FloatCells, good: Good, went: FloatCells) -> FloatCells:
        """How much of an offer went, at the share of all the good on offer that went."""
        offered = self.offered[good]
        share = np.where(offered > 0, went / np.where(offered > 0, offered, 1.0), 0.0)
        return offer * np.minimum(share, 1.0)

    def _settle(self) -> Trade:
        """Counts what every building got and gave, uses up the inputs of what the processors
        made, and keeps their stockpiles for next month."""
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
        goods_gave = {}
        for (kind, good), offer in self.offers.items():
            sent = self._given(offer, good, self.sold[good])
            given = self._given(offer, good, self.sold[good] - self.lost[good])
            most = self.count[kind] * MAKES[kind][good]
            goods_gave[(kind, good)] = (given, most)
            gave[kind] += given
            could_make[kind] += most
            recipe = RECIPES.get(kind)
            if recipe is not None:
                went = np.where(offer > 0, sent / np.where(offer > 0, offer, 1.0), 0.0)
                base, shares = self.boosted.get(kind, (offer, {}))
                used = {
                    **{need: base * per for need, per in recipe.inputs(self.economy.year).items()},
                    **{need: base * share * recipe.boosts[need] for need, share in shares.items()},
                }
                for need, amount in used.items():
                    key = (kind, need)
                    self.stock[key] = np.maximum(self.stock[key] - went * amount, 0.0)
        for key, stock in self.economy.stock.items():
            stock[:] = 0.0
            stock[self.cells] = self.stock[key]
        self.economy.taken = {}
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
            self.supplied,
            self.supplied_for,
            bonus,
            self.economy.bonus_per_good,
            self.local,
        )
        self.economy.levels[:] = Satisfaction.UNSATISFIED
        self.economy.levels[self.cells] = trade.levels()
        return trade
