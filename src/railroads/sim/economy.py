"""Trade: how stations pass the goods of railroads.sim.goods on each month, and how satisfied
that leaves each kind of building in each cell.

Each processor keeps a stockpile of up to INPUT_STOCK of every input for every building, taking in
what reaches it even while it waits for another input, and works from it.

Goods only ever move through stations (railroads.sim.railway), and only through those joined by
track to another station. A station hands what trains brought to buildings within its delivery
radius, unsatisfied ones first, then content ones, nearest first, and then what it gathered
before, so a mine in its reach feeds a steel mill in its reach; once a good has been handed out it
takes every spare unit buildings made within its collection radius into its stockpile while it has
room. Nothing is made that has nowhere to go: every building makes only what a station takes, and
a processor uses up only the inputs for what it made; the rest waits in its stockpile. Needs are
handed out in the order of PIPELINE, so the inputs of a processor come before what it makes:
canneries take their fruit before houses eat what is left.

A building's satisfaction is the share of what it demands that it got and of what it makes that
it gave away, plus BONUS_PER_GOOD for each of its bonus goods (railroads.sim.goods) it got all of,
and that share of it for part: a house fed and warmed that also got all its beer and half its
newspapers is at 145%. It is unsatisfied at nothing, thriving from 100% and content between.
Its satisfaction adds to the chances of more growing in its cell, MOST_SERVED times its points
at 100% and that much more above it (served_multipliers).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import START_YEAR
from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import (
    FITS,
    FOOD_INPUTS,
    HANDED_OUT,
    MAKES,
    NEEDS,
    PIPELINE,
    RECIPES,
    Good,
    Need,
    bonus_of,
    makes_of,
    needs_of,
)
from railroads.sim.grids import Cell, Grid
from railroads.sim.tilemap import TileMap

FloatCells = npt.NDArray[np.float64]


class Satisfaction(IntEnum):
    UNSATISFIED = 0
    CONTENT = 1
    THRIVING = 2


DELIVERY_RADIUS = 2

COLLECTION_RADIUS = 4

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


@dataclass(frozen=True)
class StationOrder:
    """What one station brings to a month's trade: goods trains left there to hand out, and how
    many more goods its stockpile can take in."""

    brought: Mapping[Good, float]
    room: float
    # How many cells out it hands goods out to, and gathers them from.
    delivery: int = DELIVERY_RADIUS
    collection: int = COLLECTION_RADIUS
    # Freight it gathered before, which it also hands out to the buildings around it.
    stockpile: Mapping[Good, float] = field(default_factory=dict)


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
        in each cell. `cells`, the flat cells with any buildings, saves finding them again."""
        return _Month(self, counts, stations or {}, cells).run()

    def _around(
        self, station: Cell, cells: npt.NDArray[np.intp], radius: float
    ) -> tuple[npt.NDArray[np.intp], FloatCells]:
        """The traded cells within `radius` steps of a station, as positions among `cells`,
        nearest first, and how many steps away each is."""
        grid = self.tile_map.grid
        reached = grid.within(station, int(radius))
        flat = np.array([np.ravel_multi_index(c, grid.shape) for c in reached], dtype=np.intp)
        steps = np.array(list(reached.values()), dtype=np.float64)
        positions = np.searchsorted(cells, flat)
        found = (positions < cells.size) & (cells[np.minimum(positions, cells.size - 1)] == flat)
        positions, steps = positions[found], steps[found]
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
        for kind in MAKES:
            if kind in RECIPES or kind not in self.count:
                continue
            for good, amount in makes_of(kind, economy.year).items():
                self.offers[(kind, good)] = self.count[kind] * amount
                self.offered[good] += self.count[kind] * amount
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
        self.room = {station: order.room for station, order in stations.items()}
        self.reach = {
            station: (
                economy._around(station, cells, order.delivery),
                economy._around(station, cells, order.collection),
            )
            for station, order in stations.items()
        }

    def run(self) -> Trade:
        for step in PIPELINE:
            if isinstance(step, BuildingKind):
                self._produce(step)
                continue
            self._serve(step)
            if step not in FOOD_INPUTS:
                for good in FITS[step]:
                    self._gather(good)
        return self._settle()

    def _wants(self, need: Need) -> dict[BuildingKind, FloatCells]:
        """What each kind wants of a need: consumers what they use in a month, processors the
        room left in their stockpile."""
        wants = {}
        for kind, n in self.count.items():
            recipe = RECIPES.get(kind)
            if recipe is not None and need in recipe.inputs(self.economy.year):
                wants[kind] = np.maximum(INPUT_STOCK * n - self.stock[(kind, need)], 0.0)
            elif recipe is None and need in needs_of(kind, self.economy.year):
                wants[kind] = n * NEEDS[kind][need]
        return wants

    def _serve(self, need: Need) -> None:
        """Lets stations hand out the goods meeting a need, the most valuable first: of each
        good what trains brought, then what they gathered before."""
        wants = self._wants(need)
        if not wants:
            return
        demand = sum(wants.values(), start=self.zero.copy())
        received = self.zero.copy()
        for station in self.stations:
            near, _ = self.reach[station]
            for good in HANDED_OUT[need]:
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
        met = np.where(demand > 0, received / np.where(demand > 0, demand, 1.0), 0.0)
        for kind, want in wants.items():
            got = met * want
            key = (kind, need)
            self.received[key] = self.received.get(key, self.zero) + got
            if kind in RECIPES:
                self.stock[key] = self.stock[key] + got

    def _gather(self, good: Good) -> None:
        """Lets every station take spare units of a good buildings made into its stockpile."""
        for station in self.stations:
            _, far = self.reach[station]
            taken = self.economy._gather(
                self.room[station], far, self.offered[good], self.offered[good], self.sold[good]
            )
            if taken > 0:
                self.gathered[station][good] = self.gathered[station].get(good, 0.0) + taken
                self.room[station] -= taken

    def _produce(self, kind: BuildingKind) -> None:
        """Offers what a kind of processor could make from its stockpile."""
        if kind not in self.count:
            return
        recipe = RECIPES[kind]
        can = self.count[kind] * recipe.most
        for need, per_unit in recipe.inputs(self.economy.year).items():
            can = np.minimum(can, self.stock[(kind, need)] / per_unit)
        self.offers[(kind, recipe.makes)] = can
        self.offered[recipe.makes] += can

    def _given(self, offer: FloatCells, good: Good) -> FloatCells:
        """How much of an offer went, at the share of all the good on offer that was sold."""
        offered = self.offered[good]
        share = np.where(offered > 0, self.sold[good] / np.where(offered > 0, offered, 1.0), 0.0)
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
            having = self.stock[(kind, need)] if kind in RECIPES else received
            got[kind] += np.minimum(having, wanted)
            needed[kind] += wanted
        goods_gave = {}
        for (kind, good), offer in self.offers.items():
            given = self._given(offer, good)
            most = offer if kind in RECIPES else self.count[kind] * MAKES[kind][good]
            goods_gave[(kind, good)] = (given, most)
            gave[kind] += given
            could_make[kind] += most
            recipe = RECIPES.get(kind)
            if recipe is not None:
                for need, per_unit in recipe.inputs(self.economy.year).items():
                    key = (kind, need)
                    self.stock[key] = np.maximum(self.stock[key] - given * per_unit, 0.0)
        for key, stock in self.economy.stock.items():
            stock[:] = 0.0
            stock[self.cells] = self.stock[key]
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
        )
        self.economy.levels[:] = Satisfaction.UNSATISFIED
        self.economy.levels[self.cells] = trade.levels()
        return trade
