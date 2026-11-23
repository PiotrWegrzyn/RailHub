"""What a month of trade takes in and gives out: the satisfaction of buildings, what stations bring
to it (StationOrder), the shares the player owns (OwnedShare), what came of it (Trade) and the
stockpiles stations keep (Stockpiles)."""

import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from railroads.sim.amounts import EPSILON, divided, ratio
from railroads.sim.constructions import BuildingKind
from railroads.sim.events import rise_of
from railroads.sim.goods import (
    RECIPES,
    Good,
    Need,
)
from railroads.sim.grids import Cell, FloatArray
from railroads.sim.move import Move
from railroads.sim.year_caches import year_cached


class Satisfaction(IntEnum):
    """How much of what they want buildings get, in words that only describe them: nothing,
    short of half, at least half and all of it."""

    UNSATISFIED = 0
    SHORT = 1
    CONTENT = 2
    THRIVING = 3


# The share of what they want from which buildings are content rather than short.
HALF_SATISFIED = 0.5


# How many cells out a station hands goods out to, and gathers spare goods from, before it is
# raised (railroads.sim.railway.Railway.reach).
DELIVERY_RADIUS = 1


COLLECTION_RADIUS = 3


# How many months of the most it uses of each input a processor keeps in stock, for every
# building.
STOCK_MONTHS = 2.0


@year_cached
def input_stock(kind: BuildingKind, need: Need, year: int) -> float:
    """The most of an input one processor of a kind keeps in stock in a given year: STOCK_MONTHS
    of what it uses of it making the most it can, with every boost that lets it take in more and
    as much more as the input's demand tiles make it take (rise_of); an input standing in for
    another as much as that one."""
    recipe = RECIPES[kind]
    rise = rise_of(kind, need, year)
    if need in recipe.boosts:
        return STOCK_MONTHS * recipe.most * recipe.boosts[need] * rise
    stands_for = {swap: replaced for replaced, (swap, _) in recipe.swaps.items()}
    used = recipe.per_unit[stands_for.get(need, need)]
    return STOCK_MONTHS * recipe.most * (1.0 + sum(recipe.takes_more.values())) * used * rise


# How much each bonus good a building gets all of adds to its satisfaction.
BONUS_PER_GOOD = 0.3


# What a unit handed out on its own is worth, times its price, when its buyers got nothing over
# the last quarter; it falls evenly to the price itself as they get all they want.
SCARCE_WORTH = 3.0


def worth_to(satisfaction: FloatArray) -> FloatArray:
    """What a unit handed out on its own is worth, times its price, to buyers this satisfied."""
    return 1.0 + (SCARCE_WORTH - 1.0) * (1.0 - np.clip(satisfaction, 0.0, 1.0))


def levels_of(share: FloatArray) -> npt.NDArray[np.int8]:
    """Satisfaction for a share of needs met or goods given away, as the satisfaction lens
    colours it: nothing, short of HALF_SATISFIED, at least that, and all of it."""
    levels = np.full(share.shape, Satisfaction.CONTENT, dtype=np.int8)
    levels[share < HALF_SATISFIED] = Satisfaction.SHORT
    levels[share <= 0.0] = Satisfaction.UNSATISFIED
    levels[share >= 1.0 - EPSILON] = Satisfaction.THRIVING
    return levels


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
    # The platforms added to it, whose cells it reaches out from as well as its own.
    platforms: tuple[Cell, ...] = ()


def stores_together(first: Mapping[Good, float], second: Mapping[Good, float]) -> dict[Good, float]:
    """Two stores of goods taken together."""
    total = dict(first)
    for good, amount in second.items():
        total[good] = total.get(good, 0.0) + amount
    return total


@dataclass(frozen=True)
class OwnedShare:
    """The buildings of a kind the player owns in each cell and the player's share of what they
    make: that part goes to the stations in reach before anything is handed out around them."""

    owned: npt.NDArray[np.float64]
    share: float
    # The boosts only the player's own take (OWNED_BOOSTS) that their tiles opened.
    takes: frozenset[Need] = frozenset()
    # How many buildings in each cell take those boosts, counted as `owned` is: all of them where
    # the player controls them (railroads.sim.ownership.CONTROL); None for just the player's own.
    taking: npt.NDArray[np.float64] | None = None


# What Trade.share works out, by the trade's id and kinds, while the trade lasts: a trade is
# saved with the game, so it keeps none of it itself.
_SHARES: dict[int, dict[tuple[BuildingKind, ...], tuple[FloatArray, npt.NDArray[np.bool_]]]] = {}


@dataclass(frozen=True)
class Trade:
    """One month of trade on the cells with buildings (`cells`, flat indices): per kind of
    building and cell, how much of its demands it got and how much of what it can make it gave
    away, both summed over goods, and how many of its bonus goods it got, each counted by the
    share of it that came."""

    cells: npt.NDArray[np.intp]
    got: Mapping[BuildingKind, FloatArray]
    needed: Mapping[BuildingKind, FloatArray]
    gave: Mapping[BuildingKind, FloatArray]
    could_make: Mapping[BuildingKind, FloatArray]
    # Per station, the goods it handed out and the goods it gathered.
    handed: Mapping[Cell, Mapping[Good, float]] = field(default_factory=dict)
    gathered: Mapping[Cell, Mapping[Good, float]] = field(default_factory=dict)
    # Per kind of building and need: what it took in this month, from anywhere, and what it
    # needs; per kind and good: what it gave away and the most it could make.
    goods_got: Mapping[tuple[BuildingKind, Need], tuple[FloatArray, FloatArray]] = field(
        default_factory=dict
    )
    goods_gave: Mapping[tuple[BuildingKind, Good], tuple[FloatArray, FloatArray]] = field(
        default_factory=dict
    )
    # Per station, how much of what trains brought it handed out, by need and good.
    handed_for: Mapping[Cell, Mapping[tuple[Need, Good], float]] = field(default_factory=dict)
    bonus: Mapping[BuildingKind, FloatArray] = field(default_factory=dict)
    # What each bonus good a building gets all of adds to its satisfaction.
    bonus_per_good: float = BONUS_PER_GOOD
    # Per need, what each cell's buildings got of it from sellers nearby rather than stations.
    local: Mapping[Need, FloatArray] = field(default_factory=dict)
    # Per good, what a unit that left each cell was worth on average, times its price: more the
    # less satisfied the buildings it was handed out to were over the last quarter, and the
    # price itself for what went to stations. Goods missing here were worth their price.
    worth: Mapping[Good, FloatArray] = field(default_factory=dict)
    # Per station, what it took of the player's share of their own buildings' goods, to hand
    # out as if a train had brought it.
    owned_brought: Mapping[Cell, Mapping[Good, float]] = field(default_factory=dict)
    # Per station, what it handed out by need and good, each unit counted by what it was worth to
    # the building that took it (worth_to): what freight pays for.
    handed_worth: Mapping[Cell, Mapping[tuple[Need, Good], float]] = field(default_factory=dict)
    # Per kind of building and good, what of the player's share of it each cell sent into the
    # stations around.
    owned_sent: Mapping[tuple[BuildingKind, Good], FloatArray] = field(default_factory=dict)

    def done(self, kinds: tuple[BuildingKind, ...]) -> FloatArray:
        return sum(
            (self.got[k] + self.gave[k] for k in kinds if k in self.got),
            start=np.zeros(self.cells.size),
        )

    def wanted(self, kinds: tuple[BuildingKind, ...]) -> FloatArray:
        return sum(
            (self.needed[k] + self.could_make[k] for k in kinds if k in self.got),
            start=np.zeros(self.cells.size),
        )

    def share(self, kinds: tuple[BuildingKind, ...], empty: float = 1.0) -> FloatArray:
        """The satisfaction of each cell's buildings of these kinds, from 0 up and past 1 with
        bonus goods. Where kinds share a cell, each counts by how much it wants; cells where they
        want nothing get `empty`."""
        share, wanting = self._shares(kinds)
        return np.where(wanting, share, empty)

    def _shares(self, kinds: tuple[BuildingKind, ...]) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
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
            own = ratio(done, wanted)
            if kind in self.bonus:
                own = own * (1.0 + self.bonus_per_good * self.bonus[kind])
            total += wanted * own
            weight += wanted
        wanting = weight > 0
        found = kept[kinds] = (divided(total, weight), wanting)
        return found

    def levels(self, kinds: tuple[BuildingKind, ...] | None = None) -> npt.NDArray[np.int8]:
        """Satisfaction of each cell's buildings of these kinds, or of all of them; cells with
        none count as thriving."""
        return levels_of(self.share(kinds if kinds is not None else tuple(self.needed)))

    def moved(self, move: Move) -> Trade:
        """This trade on the moved cells: its arrays follow its cells, or the map."""
        size = move.old.rows * move.old.cols

        def values(value: Any) -> Any:
            if isinstance(value, tuple):
                return tuple(values(part) for part in value)
            if isinstance(value, np.ndarray) and value.shape[-1:] == (size,):
                return move.flat_cells(value)
            return value

        def arrays[K](by_key: Mapping[K, Any]) -> dict[K, Any]:
            return {key: values(value) for key, value in by_key.items()}

        return replace(
            self,
            cells=move.flats(self.cells),
            got=arrays(self.got),
            needed=arrays(self.needed),
            gave=arrays(self.gave),
            could_make=arrays(self.could_make),
            handed=move.keys(self.handed),
            gathered=move.keys(self.gathered),
            goods_got=arrays(self.goods_got),
            goods_gave=arrays(self.goods_gave),
            handed_for=move.keys(self.handed_for),
            bonus=arrays(self.bonus),
            local=arrays(self.local),
            worth=arrays(self.worth),
            owned_brought=move.keys(self.owned_brought),
        )


# Every processor's kind and each input it ever takes in.
STOCKPILES = tuple(
    (kind, need) for kind, recipe in RECIPES.items() for need in Need if need in recipe.every_need
)


class Stockpiles(dict[tuple[BuildingKind, Need], FloatArray]):
    """What the processors of each kind in each cell hold of each input: a map-sized array for
    each kind and input some of them hold, and none for the rest, which holds nothing anywhere:
    asked for, it gets an empty one."""

    def __init__(self, cells: int) -> None:
        super().__init__()
        self.cells = cells

    def __missing__(self, key: tuple[BuildingKind, Need]) -> FloatArray:
        stock = self[key] = np.zeros(self.cells)
        return stock
