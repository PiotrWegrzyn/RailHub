"""What the buildings of each kind in each cell took in money, month by month (railroads.sim.growth
keeps one Ledger): what all of them sold of what they made, at its market price, and of the
player's share of them, what went into the player's stations, which the player is paid its
market price for (railroads.sim.station_trade), and about what carrying it on earns, its freight
(railroads.sim.goods.freight_of) before what the buyer's satisfaction adds.

The ledger keeps this year so far, the last year and the last QUARTER months, so a building's
results show last year's money and a forecast for the next: its last quarter, for a whole year.
It also keeps how satisfied the buildings of each kind in each cell were in each of those months,
which their shares are priced by (railroads.sim.ownership)."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import QUARTER
from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Trade
from railroads.sim.goods import freight_of, price_of

# What a month's takings of each cell hold, in this order.
OUTPUT, SOLD, FREIGHT = range(3)
MEASURES = 3


@dataclass(frozen=True)
class Takings:
    """A year's money of the buildings of a kind in a cell: what all of them sold of what they
    made, and of the player's share of them, what they sold to the player's stations and about
    what its freight earns."""

    output: float
    sold: float
    freight: float

    @property
    def yours(self) -> float:
        return self.sold + self.freight


def _empty(*shape: int) -> npt.NDArray[np.float64]:
    return np.zeros((*shape, 0))


@dataclass
class _Books:
    """The takings of the buildings of one kind, in the flat cells they have ever had any in,
    sorted: this year's so far, last year's and each of the last QUARTER months', the latest last."""

    cells: npt.NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    year: npt.NDArray[np.float64] = field(default_factory=lambda: _empty(MEASURES))
    last: npt.NDArray[np.float64] = field(default_factory=lambda: _empty(MEASURES))
    recent: npt.NDArray[np.float64] = field(default_factory=lambda: _empty(QUARTER, MEASURES))
    # Each of the last QUARTER months, how satisfied the buildings were, and 1 where they traded.
    moods: npt.NDArray[np.float64] = field(default_factory=lambda: _empty(QUARTER))
    seen: npt.NDArray[np.float64] = field(default_factory=lambda: _empty(QUARTER))

    def place(self, cells: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Where each of these flat cells stands in the books, making room for those new to them."""
        joined = np.union1d(self.cells, cells)
        if joined.size != self.cells.size:
            at = np.searchsorted(joined, self.cells)
            for name in ("year", "last", "recent", "moods", "seen"):
                old = getattr(self, name)
                grown = np.zeros((*old.shape[:-1], joined.size))
                grown[..., at] = old
                setattr(self, name, grown)
            self.cells = joined
        return np.searchsorted(self.cells, cells)

    def find(self, flat: int) -> int | None:
        at = int(np.searchsorted(self.cells, flat))
        return at if at < self.cells.size and self.cells[at] == flat else None

    def close_year(self) -> None:
        self.last = self.year
        self.year = np.zeros_like(self.last)
        kept = (
            (self.last != 0).any(axis=0)
            | (self.recent != 0).any(axis=(0, 1))
            | (self.seen != 0).any(axis=0)
        )
        self.cells = self.cells[kept]
        self.last, self.year = self.last[:, kept], self.year[:, kept]
        self.recent = self.recent[..., kept]
        self.moods, self.seen = self.moods[:, kept], self.seen[:, kept]


class Ledger:
    def __init__(self) -> None:
        self._books: dict[BuildingKind, _Books] = {}
        # The year, counted from the start, the books are open for, and how many months of trade
        # went into it and into the last; and how many of the last QUARTER months had trade.
        self._year: int | None = None
        self._months = 0
        self._last_months = 0
        self._recent_months = 0

    def record(self, trade: Trade, month: int) -> None:
        """Counts a month of trade, `month` months from the start, into the books."""
        year = month // 12
        if self._year is not None and year != self._year:
            self._last_months = self._months if year == self._year + 1 else 0
            self._months = 0
            for books in self._books.values():
                books.close_year()
        self._year = year
        self._months += 1
        self._recent_months = min(QUARTER, self._recent_months + 1)
        for books in self._books.values():
            books.recent = np.roll(books.recent, -1, axis=0)
            books.recent[-1] = 0.0
            for name in ("moods", "seen"):
                rows = np.roll(getattr(books, name), -1, axis=0)
                rows[-1] = 0.0
                setattr(books, name, rows)
        takings = _takings(trade)
        for kind in takings.keys() | trade.got.keys():
            taken = takings.get(kind)
            moods = trade.share((kind,), np.nan) if kind in trade.got else None
            found = np.zeros(trade.cells.size, dtype=bool)
            if taken is not None:
                found |= (taken != 0).any(axis=0)
            if moods is not None:
                found |= ~np.isnan(moods)
            if not found.any():
                continue
            books = self._books.setdefault(kind, _Books())
            at = books.place(trade.cells[found])
            if taken is not None:
                books.year[:, at] += taken[:, found]
                books.recent[-1][:, at] = taken[:, found]
            if moods is not None:
                mood = moods[found]
                books.moods[-1][at] = np.nan_to_num(mood)
                books.seen[-1][at] = ~np.isnan(mood)

    def last_year(self, kind: BuildingKind, flat: int) -> Takings | None:
        """What the buildings of a kind in a flat cell took last year; None if trade did not run
        through all of it."""
        if self._last_months < 12:
            return None
        return self._read(kind, flat, lambda books, at: books.last[:, at])

    def this_year(self, kind: BuildingKind, flat: int) -> Takings:
        """What they took this year so far."""
        taken = self._read(kind, flat, lambda books, at: books.year[:, at])
        return Takings(0.0, 0.0, 0.0) if taken is None else taken

    def forecast(self, kind: BuildingKind, flat: int) -> Takings | None:
        """What they would take over the next year at the pace of the last quarter; None before
        any trade."""
        if not self._recent_months:
            return None
        months = self._recent_months
        taken = self._read(kind, flat, lambda books, at: books.recent[:, :, at].sum(axis=0))
        if taken is None:
            return Takings(0.0, 0.0, 0.0)
        scale = 12 / months
        return Takings(taken.output * scale, taken.sold * scale, taken.freight * scale)

    def satisfaction(self, kind: BuildingKind, flat: int) -> float:
        """How satisfied the buildings of a kind in a flat cell were over the last quarter, on
        average over the months they traded; full before they traded at all."""
        books = self._books.get(kind)
        at = None if books is None else books.find(flat)
        if books is None or at is None:
            return 1.0
        months = float(books.seen[:, at].sum())
        return float(books.moods[:, at].sum()) / months if months else 1.0

    def _read(
        self,
        kind: BuildingKind,
        flat: int,
        pick: Callable[[_Books, int], npt.NDArray[np.float64]],
    ) -> Takings | None:
        books = self._books.get(kind)
        at = None if books is None else books.find(flat)
        if books is None or at is None:
            return None
        values = pick(books, at)
        return Takings(float(values[OUTPUT]), float(values[SOLD]), float(values[FREIGHT]))


def _takings(trade: Trade) -> dict[BuildingKind, npt.NDArray[np.float64]]:
    """Per kind of building, a month's takings of each traded cell: rows OUTPUT, SOLD and
    FREIGHT."""
    takings: dict[BuildingKind, npt.NDArray[np.float64]] = {}

    def row(kind: BuildingKind) -> npt.NDArray[np.float64]:
        if kind not in takings:
            takings[kind] = np.zeros((MEASURES, trade.cells.size))
        return takings[kind]

    for (kind, good), (given, _) in trade.goods_gave.items():
        if given.any():
            row(kind)[OUTPUT] += given * price_of(good)
    for (kind, good), sent in trade.owned_sent.items():
        if sent.any():
            row(kind)[SOLD] += sent * price_of(good)
            row(kind)[FREIGHT] += sent * freight_of(good)
    return takings
