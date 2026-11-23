"""The shares of buildings the player owns (railroads.sim.growth keeps one Owners): how many of
each kind in each cell, and what buying or selling them fetches.

Every building is SHARES shares, bought and sold LOTS at a time. Holding CONTROL of the shares of
the buildings of a kind in a cell gives the player control of them: only then do the boosts only their
own take (railroads.sim.goods.OWNED_BOOSTS) work on them, on the whole of them, and the player
still takes their part of all they make; past that, more shares only raise that part. A building pays nothing of its own: the player's part of what it makes, by their shares, goes into the stations around it
before anything is handed out, as if a train had brought it, so the stations hand it out to the
buildings in their reach and it pays as a delivery does (railroads.sim.economy,
railroads.sim.station_trade). A whole building costs PAYBACK_YEARS of all it makes at full
output at full satisfaction, at the prices on the boards (railroads.sim.tiles), from two thirds of
that with none to twice that at 200% (PRICE_BY_SATISFACTION), by how satisfied the buildings of
its kind in its cell were over the last quarter (railroads.sim.ledger); shares sell at the price
they buy at. A power plant of any kind, which makes cheap electricity for the player's trains
(railroads.sim.fuel_market), costs POWER_PLANT_PRICE. When buildings leave a cell, the player's
shares go last.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import MONTHS_PER_YEAR
from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import Need, price_of
from railroads.sim.grids import Cell, IntArray
from railroads.sim.tiles import of_year
from railroads.sim.trade import OwnedShare
from railroads.sim.yearly import makes_of, owned_boosts

if TYPE_CHECKING:
    from railroads.sim.growth import Growth

# Every building is SHARES shares, bought and sold LOT at a time: the player's shares of it are
# their part of what it makes, which goes to the stations around before anything is handed out.
SHARES = 1000
LOT = 100
# How many shares the player may buy or sell at once, LOT the fewest.
LOTS = (100, 200, 600)
# The part of the shares that gives control: a big flag over the buildings, and the boosts only
# the player's buildings take working on the whole of them.
CONTROL = 0.6
# What a whole building costs at full satisfaction: this many years of all it makes at full
# output, at its goods' market prices, so its shares pay a tenth of their price back a year.
PAYBACK_YEARS = 10
# What a building's price is worth, times its price at full satisfaction, at each satisfaction
# over the last quarter, in between in a straight line.
PRICE_BY_SATISFACTION = ((0.0, 2 / 3), (1.0, 1.0), (2.0, 2.0))
# The satisfaction from which a building's price rises no more.
MOST_PRICED = PRICE_BY_SATISFACTION[-1][0]
# Shares bought can be sold only this many months later, so they can't be flipped for quick gains.
HOLD_MONTHS = 60
# The name the player's company trades under, whose flag flies over the buildings it owns.
COMPANY = "The Trains Corp"
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
        self._shares: dict[BuildingKind, IntArray] = {}
        # Per kind and cell, the shares bought there and the month from which each lot may be sold.
        self._held: dict[tuple[BuildingKind, Cell], list[tuple[int, int]]] = {}
        self.company = COMPANY

    def shares(self, kind: BuildingKind) -> IntArray:
        """How many shares of the buildings of a kind the player owns in each cell."""
        return self._shares.get(kind, np.zeros(self._growth.grid.shape, dtype=np.int64))

    def total(self, cell: Cell, kind: BuildingKind) -> int:
        """How many shares the buildings of a kind in a cell are in all, SHARES for each."""
        counts = self._growth.counts
        return SHARES * int(counts[kind][cell]) if kind in counts else 0

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

    def own(self, cell: Cell, kind: BuildingKind, shares: int = SHARES) -> int:
        """Makes up to this many more shares of the kind in the cell the player's, as many as
        are left to own; returns how many."""
        grid = self._shares.setdefault(kind, np.zeros(self._growth.grid.shape, dtype=np.int64))
        added = max(0, min(shares, SHARES * int(self._growth.counts[kind][cell]) - int(grid[cell])))
        grid[cell] += added
        return added

    def price_to_buy(self, kind: BuildingKind) -> float | None:
        """What a whole building of a kind costs at full satisfaction; None for what cannot be
        bought."""
        growth = self._growth
        if kind not in growth.counts:
            return None
        if kind in POWER_PLANTS:
            return POWER_PLANT_PRICE
        makes = makes_of(kind, growth.year)
        if not makes:
            return None
        boards = of_year(growth.year)
        month = sum(amount * price_of(good) * boards.price(good) for good, amount in makes.items())
        return MONTHS_PER_YEAR * PAYBACK_YEARS * month

    def lot_price(self, cell: Cell, kind: BuildingKind) -> float | None:
        """What LOT shares of a building of a kind in a cell cost, or fetch, now: by how
        satisfied the buildings of that kind there were over the last quarter, not the rest of
        the cell; None for what cannot be bought."""
        whole = self.price_to_buy(kind)
        if whole is None:
            return None
        if kind in POWER_PLANTS:
            return whole * LOT / SHARES
        satisfaction = self._growth.ledger.satisfaction(kind, self._flat(cell))
        return whole * LOT / SHARES * satisfaction_price(satisfaction)

    def buy(self, cell: Cell, kind: BuildingKind, lot: int = LOT) -> str | None:
        """Buys `lot` shares of a building of a kind in a cell; why not, if they cannot be
        bought."""
        price = self.lot_price(cell, kind)
        if price is None:
            return f"A {kind.value} cannot be bought"
        if self.shares(kind)[cell] + lot > self.total(cell, kind):
            return f"There are not {lot} shares of the {kind.value}s here left to buy"
        railway = self._growth.railway
        if railway is None or not railway.finances.pay(price * lot / LOT, "shares bought"):
            return "Not enough cash"
        self.own(cell, kind, lot)
        lots = self._held.setdefault((kind, cell), [])
        lots.append((self._growth.months + HOLD_MONTHS, lot))
        return None

    def controls(self, cell: Cell, kind: BuildingKind) -> bool:
        """Whether the player holds CONTROL of the shares of the buildings of a kind in a cell."""
        total = self.total(cell, kind)
        return total > 0 and int(self.shares(kind)[cell]) >= CONTROL * total

    def controls_any(self, cell: Cell) -> bool:
        """Whether the player controls the buildings of any kind in a cell."""
        return any(self.controls(cell, kind) for kind in self.kinds())

    def sellable(self, cell: Cell, kind: BuildingKind) -> int:
        """How many of the player's shares of a kind in a cell they may sell now: all but those
        bought less than HOLD_MONTHS ago."""
        now = self._growth.months
        locked = sum(shares for until, shares in self._held.get((kind, cell), ()) if until > now)
        return max(0, int(self.shares(kind)[cell]) - locked)

    def sellable_from(self, cell: Cell, kind: BuildingKind) -> int | None:
        """The year the next of the player's shares of a kind in a cell they can't sell yet may
        be sold; None if none are held back."""
        now = self._growth.months
        until = [until for until, _ in self._held.get((kind, cell), ()) if until > now]
        if not until:
            return None
        return self._growth.start_year + min(until) // MONTHS_PER_YEAR

    def sell(self, cell: Cell, kind: BuildingKind, lot: int = LOT) -> str | None:
        """Sells `lot` shares, or the last few, of a building of a kind in a cell at their price
        now, of those bought at least HOLD_MONTHS ago; why not, if there are none to sell."""
        held = int(self.shares(kind)[cell])
        price = self.lot_price(cell, kind)
        railway = self._growth.railway
        if not held or price is None or railway is None:
            return f"You own no shares of a {kind.value} here"
        free = self.sellable(cell, kind)
        if not free:
            return f"These shares can be sold from {self.sellable_from(cell, kind)}"
        sold = min(lot, free)
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
        self._hold_at_most(cell, kind, int(grid[cell]))
        return lost

    def _hold_at_most(self, cell: Cell, kind: BuildingKind, shares: int) -> None:
        """Lets the shares held back in a cell be no more than the player owns there, the newest
        let go first."""
        lots = self._held.get((kind, cell))
        if not lots:
            return
        kept: list[tuple[int, int]] = []
        for until, held in sorted(lots):
            take = min(held, shares)
            if take > 0:
                kept.append((until, take))
            shares -= take
        self._held[(kind, cell)] = kept

    def taking(self, kind: BuildingKind, need: Need) -> npt.NDArray[np.float64] | None:
        """How many buildings of a kind take a boost only the player's own take, if their tile
        opened it, in each cell as flat cells counted as they trade: all of them where the player
        controls them, none elsewhere; None if the tile has not opened it."""
        railway = self._growth.railway
        grid = self._shares.get(kind)
        if railway is None or grid is None or need not in owned_boosts(kind, self._growth.year):
            return None
        return self._controlled(kind, grid).ravel()

    def _controlled(self, kind: BuildingKind, grid: IntArray) -> npt.NDArray[np.float64]:
        """How many buildings of a kind in each cell the player controls: all there, or none."""
        counts = self._growth.counts[kind]
        controlled = (counts > 0) & (grid >= CONTROL * SHARES * counts)
        return np.where(controlled, counts, 0).astype(np.float64)

    def first_for_stations(self) -> dict[BuildingKind, OwnedShare]:
        """Per kind the player owns shares of, how many buildings' worth they own in each cell,
        as they trade: all they make goes to the stations around before anything is handed
        out. And the boosts only their own take that their tiles opened."""
        railway = self._growth.railway
        shares: dict[BuildingKind, OwnedShare] = {}
        for kind, grid in self._shares.items():
            if not grid.any():
                continue
            shares[kind] = OwnedShare(
                grid.ravel() / SHARES,
                1.0,
                frozenset() if railway is None else owned_boosts(kind, self._growth.year),
                self._controlled(kind, grid).ravel(),
            )
        return shares

    def _flat(self, cell: Cell) -> int:
        return int(np.ravel_multi_index(cell, self._growth.grid.shape))
