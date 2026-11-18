"""The shares of buildings the player owns (railroads.sim.growth keeps one Owners): how many of
each kind in each cell, and what buying or selling them fetches.

Every building is SHARES shares, bought and sold LOT at a time. A building pays nothing of its
own: the player's part of what it makes, by their shares, goes into the stations around it
before anything is handed out, as if a train had brought it, so the stations hand it out to the
buildings in their reach and it pays as a delivery does (railroads.sim.economy,
railroads.sim.station_trade). A whole building costs PAYBACK_YEARS of all it makes at full
output at full satisfaction, from two thirds of that with none to four thirds at 200%
(PRICE_BY_SATISFACTION), by how satisfied its cell was over the last quarter, less what the
player's tech takes off, down to MOST_DISCOUNT of the price; shares sell at the price they buy
at. A power plant of any kind, which makes cheap electricity for the player's trains
(railroads.sim.fuel_market), costs POWER_PLANT_PRICE. Tech in a building's tree can raise how
many buildings' worth it makes. When buildings leave a cell, the player's shares go last.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import OwnedShare
from railroads.sim.goods import Need, makes_of, price_of
from railroads.sim.grids import Cell
from railroads.sim.ladders import IntGrid
from railroads.sim.tech import Effect, Tree, tree_of

if TYPE_CHECKING:
    from railroads.sim.growth import Growth

# Every building is SHARES shares, bought and sold LOT at a time: the player's shares of it are
# their part of what it makes, which goes to the stations around before anything is handed out.
SHARES = 1000
LOT = 100
# What a whole building costs at full satisfaction: this many years of all it makes at full
# output, at its goods' market prices, so its shares pay a tenth of their price back a year.
PAYBACK_YEARS = 10
# What a building's price is worth, times its price at full satisfaction, at each satisfaction
# over the last quarter, in between in a straight line.
PRICE_BY_SATISFACTION = ((0.0, 2 / 3), (1.0, 1.0), (2.0, 4 / 3))
# Every whole building the player owns adds this much to the experience its tree gets from
# deliveries for each of its points, and the shares of one their part of that.
EXPERIENCE_PER_POINT = 0.01
# The name the player's company trades under, whose flag flies over the buildings it owns.
COMPANY = "The Trains Corp"
# However much tech cuts the price of shares, they cost at least this share of it.
MOST_DISCOUNT = 0.5
POWER_PLANT_PRICE = 2_000_000.0
POWER_PLANTS = frozenset(
    {BuildingKind.POWER_PLANT, BuildingKind.DIESEL_POWER_PLANT, BuildingKind.GAS_POWER_PLANT}
)


def satisfaction_price(satisfaction: float) -> float:
    """How many times its price at full satisfaction a building costs at this satisfaction."""
    points, factors = zip(*PRICE_BY_SATISFACTION, strict=True)
    return float(np.interp(satisfaction, points, factors))


class Owners:
    def __init__(self, growth: Growth) -> None:
        self._growth = growth
        self._shares: dict[BuildingKind, IntGrid] = {}
        self.company = COMPANY

    def shares(self, kind: BuildingKind) -> IntGrid:
        """How many shares of the buildings of a kind the player owns in each cell."""
        return self._shares.get(kind, np.zeros(self._growth.grid.shape, dtype=np.int64))

    def owned(self, kind: BuildingKind) -> npt.NDArray[np.float64]:
        """How many buildings' worth of a kind the player owns in each cell, by their shares."""
        return self.shares(kind) / SHARES

    def count(self) -> float:
        """How many buildings' worth the player owns in all."""
        return sum(int(grid.sum()) for grid in self._shares.values()) / SHARES

    def cells(self) -> list[Cell]:
        """The cells where the player owns shares of any building."""
        held = np.zeros(self._growth.grid.shape, dtype=bool)
        for grid in self._shares.values():
            held |= grid > 0
        return [(int(row), int(col)) for row, col in np.argwhere(held)]

    def kinds(self) -> list[BuildingKind]:
        """The kinds of building the player owns shares of."""
        return [kind for kind, grid in self._shares.items() if grid.any()]

    def output_factor(self, kind: BuildingKind) -> float:
        """How many buildings' worth one of a kind the player owns makes, with their tech."""
        return 1.0 + self.boost(kind, Effect.PRODUCTION)

    def own(self, cell: Cell, kind: BuildingKind, shares: int = SHARES) -> int:
        """Makes up to this many more shares of the kind in the cell the player's, as many as
        are left to own; returns how many."""
        grid = self._shares.setdefault(kind, np.zeros(self._growth.grid.shape, dtype=np.int64))
        added = max(0, min(shares, SHARES * int(self._growth.counts[kind][cell]) - int(grid[cell])))
        grid[cell] += added
        return added

    def price_to_buy(self, kind: BuildingKind) -> float | None:
        """What a whole building of a kind costs at full satisfaction, with the player's tech;
        None for what cannot be bought."""
        growth = self._growth
        if kind not in growth.counts:
            return None
        railway = growth.railway
        tree = tree_of(kind)
        discount = 0.0 if railway is None else railway.research.bonus(Effect.DISCOUNT, tree)
        if tree is None and railway is not None:
            discount = railway.research.bonus(Effect.DISCOUNT, Tree.COMMERCE)
        kept = max(MOST_DISCOUNT, 1.0 - discount)
        if kind in POWER_PLANTS:
            return POWER_PLANT_PRICE * kept
        makes = makes_of(kind, growth.year)
        if not makes:
            return None
        month = sum(amount * price_of(good) for good, amount in makes.items())
        return 12 * PAYBACK_YEARS * month * kept

    def lot_price(self, cell: Cell, kind: BuildingKind) -> float | None:
        """What LOT shares of a building of a kind in a cell cost, or fetch, now: by how
        satisfied the cell was over the last quarter; None for what cannot be bought."""
        whole = self.price_to_buy(kind)
        if whole is None:
            return None
        if kind in POWER_PLANTS:
            return whole * LOT / SHARES
        satisfaction = float(self._growth.economy.last_quarter()[self._flat(cell)])
        return whole * LOT / SHARES * satisfaction_price(satisfaction)

    def buy(self, cell: Cell, kind: BuildingKind) -> str | None:
        """Buys LOT shares of a building of a kind in a cell; why not, if they cannot be bought."""
        price = self.lot_price(cell, kind)
        if price is None:
            return f"A {kind.value} cannot be bought"
        if self.shares(kind)[cell] + LOT > SHARES * self._growth.counts[kind][cell]:
            return f"You already own every share of the {kind.value}s here"
        railway = self._growth.railway
        if railway is None or not railway.finances.pay(price, "shares bought"):
            return "Not enough cash"
        self.own(cell, kind, LOT)
        return None

    def sell(self, cell: Cell, kind: BuildingKind) -> str | None:
        """Sells LOT shares, or the last few, of a building of a kind in a cell at their price
        now; why not, if there are none to sell."""
        held = int(self.shares(kind)[cell])
        price = self.lot_price(cell, kind)
        railway = self._growth.railway
        if not held or price is None or railway is None:
            return f"You own no shares of a {kind.value} here"
        sold = min(LOT, held)
        railway.finances.earn(price * sold / LOT, "shares sold")
        self._shares[kind][cell] -= sold
        return None

    def lose(self, cell: Cell, kind: BuildingKind, gone: int) -> int:
        """Counts `gone` buildings of a kind leaving a cell, the player's shares last; returns
        how many shares of them the player owned."""
        grid = self._shares.get(kind)
        if grid is None or not grid[cell]:
            return 0
        left = SHARES * max(0, int(self._growth.counts[kind][cell]) - gone)
        lost = max(0, int(grid[cell]) - left)
        grid[cell] -= lost
        return lost

    def taking(self, kind: BuildingKind, need: Need) -> npt.NDArray[np.float64] | None:
        """How many buildings' worth of a kind the player owns in each cell, as flat cells
        counted as they trade, if their tech opened a boost only their own take; None if not."""
        railway = self._growth.railway
        grid = self._shares.get(kind)
        if railway is None or grid is None or need not in railway.research.takes(kind):
            return None
        return grid.ravel() / SHARES * (1.0 + self.boost(kind, Effect.PRODUCTION))

    def boost(self, kind: BuildingKind, effect: Effect) -> float:
        """What the player's tech adds, of one effect, to the buildings of a kind they own."""
        tree = tree_of(kind)
        railway = self._growth.railway
        if railway is None or tree is None:
            return 0.0
        return railway.research.bonus(effect, tree)

    def working_counts(self) -> Mapping[BuildingKind, npt.NDArray[np.int64 | np.float64]]:
        """The buildings as they trade: the player's part of each counts as more, by what their
        production tech adds."""
        counts_now = self._growth.counts
        if not self._shares:
            return counts_now
        counts: dict[BuildingKind, npt.NDArray[np.int64 | np.float64]] = dict(counts_now)
        for kind, grid in self._shares.items():
            boost = self.boost(kind, Effect.PRODUCTION)
            if boost and grid.any():
                counts[kind] = counts_now[kind] + boost * grid / SHARES
        return counts

    def first_for_stations(self) -> dict[BuildingKind, OwnedShare]:
        """Per kind the player owns shares of, how many buildings' worth they own in each cell,
        as they trade: all they make goes to the stations around before anything is handed
        out. And the boosts only their own take that their tech opened."""
        railway = self._growth.railway
        return {
            kind: OwnedShare(
                grid.ravel() / SHARES * (1.0 + self.boost(kind, Effect.PRODUCTION)),
                1.0,
                frozenset() if railway is None else railway.research.takes(kind),
            )
            for kind, grid in self._shares.items()
            if grid.any()
        }

    def experience_factor(self, tree: Tree) -> float:
        """How many times what deliveries earn a tree gets as experience: once, and
        EXPERIENCE_PER_POINT more for each point of the buildings of that tree the player owns,
        by their shares."""
        points = self._growth.rule.points
        owned = sum(
            points[kind][1] * int(grid.sum()) / SHARES
            for kind, grid in self._shares.items()
            if kind in points and tree_of(kind) == tree
        )
        return 1.0 + EXPERIENCE_PER_POINT * owned

    def _flat(self, cell: Cell) -> int:
        return int(np.ravel_multi_index(cell, self._growth.grid.shape))
