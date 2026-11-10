"""The buildings the player owns (railroads.sim.growth keeps one Owners): how many of each kind in
each cell, what buying one costs, and the share of what they sell that the player is paid.

A building costs PAYBACK_YEARS of its owner's share at full output, OWNER_SHARE of what it sells,
less what the player's tech takes off, down to MOST_DISCOUNT of the price. Tech in a building's
tree can raise the share it pays and how many buildings' worth it makes. When buildings leave a
cell, the player's go last.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Trade
from railroads.sim.goods import FITS, makes_of, price_of
from railroads.sim.grids import Cell
from railroads.sim.ladders import IntGrid
from railroads.sim.tech import SHOPS, Effect, Tree, tree_of

if TYPE_CHECKING:
    from railroads.sim.growth import Growth

# The player's share of what a building they own sells to their stations, and what it costs to
# buy: this many years of that share at full output.
OWNER_SHARE = 0.5
PAYBACK_YEARS = 5
# However much tech cuts the price of buildings, they cost at least this share of it.
MOST_DISCOUNT = 0.5


class Owners:
    def __init__(self, growth: Growth) -> None:
        self._growth = growth
        self._owned: dict[BuildingKind, IntGrid] = {}
        # What each kind the player owns earned them last month.
        self.income: dict[BuildingKind, float] = {}

    def owned(self, kind: BuildingKind) -> IntGrid:
        """How many buildings of a kind the player owns in each cell."""
        return self._owned.get(kind, np.zeros(self._growth.grid.shape, dtype=np.int64))

    def kinds(self) -> list[BuildingKind]:
        """The kinds of building the player owns any of."""
        return [kind for kind, grid in self._owned.items() if grid.any()]

    def share(self, kind: BuildingKind) -> float:
        """The player's share of what one building of a kind they own sells, with their tech."""
        return OWNER_SHARE + self.boost(kind, Effect.SHARE)

    def output_factor(self, kind: BuildingKind) -> float:
        """How many buildings' worth one of a kind the player owns makes, with their tech."""
        return 1.0 + self.boost(kind, Effect.PRODUCTION)

    def own(self, cell: Cell, kind: BuildingKind) -> bool:
        """Makes one more of the kind in the cell the player's, if any is left to own."""
        grid = self._owned.setdefault(kind, np.zeros(self._growth.grid.shape, dtype=np.int64))
        if grid[cell] >= self._growth.counts[kind][cell]:
            return False
        grid[cell] += 1
        return True

    def price_to_buy(self, kind: BuildingKind) -> float | None:
        """What one building of a kind costs the player; None for what cannot be bought."""
        growth = self._growth
        makes = makes_of(kind, growth.year)
        if not makes or kind not in growth.counts:
            return None
        month = sum(amount * price_of(good) for good, amount in makes.items())
        railway = growth.railway
        discount = 0.0 if railway is None else railway.research.bonus(Effect.DISCOUNT)
        return 12 * PAYBACK_YEARS * OWNER_SHARE * month * max(MOST_DISCOUNT, 1.0 - discount)

    def buy(self, cell: Cell, kind: BuildingKind) -> str | None:
        """Buys one building of a kind in a cell; why not, if it cannot be bought."""
        price = self.price_to_buy(kind)
        if price is None:
            return f"A {kind.value} cannot be bought"
        if self.owned(kind)[cell] >= self._growth.counts[kind][cell]:
            return f"You already own every {kind.value} here"
        railway = self._growth.railway
        if railway is None or not railway.finances.can_afford(price):
            return "Not enough cash"
        railway.finances.spend(price, "buildings bought")
        self.own(cell, kind)
        return None

    def lose(self, cell: Cell, kind: BuildingKind, gone: int) -> int:
        """Counts `gone` buildings of a kind leaving a cell, the player's last; returns how
        many of them the player owned."""
        grid = self._owned.get(kind)
        if grid is None or not grid[cell]:
            return 0
        unowned = int(self._growth.counts[kind][cell] - grid[cell])
        owned = max(0, gone - unowned)
        grid[cell] -= owned
        return owned

    def boost(self, kind: BuildingKind, effect: Effect) -> float:
        """What the player's tech adds, of one effect, to the buildings of a kind they own."""
        tree = tree_of(kind)
        railway = self._growth.railway
        if railway is None or tree is None:
            return 0.0
        return railway.research.bonus(effect, tree)

    def working_counts(self) -> Mapping[BuildingKind, npt.NDArray[np.int64 | np.float64]]:
        """The buildings as they trade: each the player owns counts as more than one, by what
        their production tech adds."""
        counts_now = self._growth.counts
        if not self._owned:
            return counts_now
        counts: dict[BuildingKind, npt.NDArray[np.int64 | np.float64]] = dict(counts_now)
        for kind, grid in self._owned.items():
            boost = self.boost(kind, Effect.PRODUCTION)
            if boost and grid.any():
                counts[kind] = counts_now[kind] + boost * grid
        return counts

    def pay(self, trade: Trade) -> None:
        """Pays the player their share of what the buildings they own sold this month, which
        is also experience in their trees, and counts what shops around the player's stations
        took as commerce experience."""
        self.income = {}
        railway = self._growth.railway
        if railway is None:
            return
        research = railway.research
        counts = self._growth.counts
        for (kind, good), (given, _) in trade.goods_gave.items():
            grid = self._owned.get(kind)
            if grid is None:
                continue
            owned = grid.ravel()[trade.cells].astype(np.float64)
            if not owned.any():
                continue
            boost = self.boost(kind, Effect.PRODUCTION)
            count = counts[kind].ravel()[trade.cells] + boost * owned
            theirs = owned * (1.0 + boost) / np.maximum(count, 1.0)
            share = OWNER_SHARE + self.boost(kind, Effect.SHARE)
            earned = float((given * theirs).sum()) * price_of(good) * share
            self.income[kind] = self.income.get(kind, 0.0) + earned
        for kind, earned in self.income.items():
            tree = tree_of(kind)
            if tree is not None:
                research.earn(tree, earned)
        total = sum(self.income.values())
        if total > 0:
            railway.finances.earn(total, "owned buildings")
        for (kind, need), (got, _) in trade.goods_got.items():
            if kind in SHOPS:
                research.earn(Tree.COMMERCE, float(got.sum()) * price_of(FITS[need][0]))
