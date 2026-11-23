"""The player's money: the cash a game starts with, what building costs, the upkeep charged
every month, and a ledger of the month so far and the month before.

Building needs the cash for it; upkeep is charged whatever the cash, so it can run into debt.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from railroads.sim.amounts import EPSILON
from railroads.sim.constructions import Track, TrackWidth
from railroads.sim.tilemap import Tile
from railroads.sim.track_rules import track_rules_for

START_CASH = 100_000.0
# Track costs TRACK_PRICE per cell times its bed's cost factor (railroads.sim.track_rules), and
# electric track ELECTRIC_TRACK_FACTOR times that, or a maglev guideway MAGLEV_TRACK_PRICE more;
# double track costs DOUBLE_TRACK_FACTOR as much.
TRACK_PRICE = 600.0
ELECTRIC_TRACK_FACTOR = 5.0
MAGLEV_TRACK_PRICE = 4_000.0
DOUBLE_TRACK_FACTOR = 3.0
STATION_PRICE = 15_000.0
TRAIN_PRICE = 10_000.0
# What selling a train gives back, as a share of its price.
TRAIN_RESALE = 0.5
# What each car costs every time a train leaves a stop with it, loaded or not: CAR_TRIP_PRICE,
# and CAR_PLACE_PRICE more for every CAR_PLACES it holds.
CAR_TRIP_PRICE = 15.0
CAR_PLACE_PRICE = 1.0
CAR_PLACES = 10
# What raising a station to each level above the first costs; and the first storage upgrade,
# each one after costing STORAGE_GROWTH times the one before (storage_price).
LEVEL_PRICES = (50_000.0, 250_000.0, 1_500_000.0, 10_000_000.0)
STORAGE_PRICE = 10_000.0
STORAGE_GROWTH = 2.0
# What each facility at a station costs (railroads.sim.facilities).
ENGINE_SHED_PRICE = 40_000.0
SAND_TOWER_PRICE = 25_000.0
POST_OFFICE_PRICE = 30_000.0
REFRESHMENT_ROOM_PRICE = 30_000.0
MARKET_HALL_PRICE = 50_000.0
CATTLE_PENS_PRICE = 35_000.0
WAREHOUSE_PRICE = 60_000.0
# What a station of each level costs to keep a month, every station alike, rising by
# STATION_UPKEEP_RISE a year, compounded, from the year play began.
STATION_UPKEEP = (150.0, 400.0, 1_200.0, 3_500.0, 9_000.0)
STATION_UPKEEP_RISE = 0.02
# Every cell of track costs TRACK_UPKEEP a month, ELECTRIC_TRACK_UPKEEP more where it is
# electric, MAGLEV_TRACK_UPKEEP more where it is maglev, and DOUBLE_TRACK_UPKEEP_FACTOR as much
# where it is double.
TRACK_UPKEEP = 2.0
DOUBLE_TRACK_UPKEEP_FACTOR = 1.8
ELECTRIC_TRACK_UPKEEP = 40.0
MAGLEV_TRACK_UPKEEP = 60.0


def car_trip_price(places: float) -> float:
    """What a car holding this much costs each time a train leaves a stop with it."""
    return CAR_TRIP_PRICE + CAR_PLACE_PRICE * (int(places) // CAR_PLACES)


def storage_price(upgrades: int) -> float:
    """What the next storage upgrade costs a station that has had this many."""
    return STORAGE_PRICE * STORAGE_GROWTH**upgrades


def track_price(piece: Track, tile: Tile) -> float:
    """What laying this piece of track in this cell costs."""
    price = TRACK_PRICE * track_rules_for(tile, piece.bed).price_factor
    price *= ELECTRIC_TRACK_FACTOR if piece.electric else 1.0
    return _dressed(price, piece, 0.0, MAGLEV_TRACK_PRICE)


def _dressed(
    base: float, piece: Track, electric: float, maglev: float, double: float = DOUBLE_TRACK_FACTOR
) -> float:
    """A piece's base cost with what its wires or guideway add, the whole doubled up for double
    track."""
    cost = base + (electric if piece.electric else 0.0) + (maglev if piece.maglev else 0.0)
    return cost * (double if piece.width == TrackWidth.DOUBLE else 1.0)


def track_upkeep(pieces: Iterable[Track]) -> float:
    """What this track costs every month: each cell it runs through is charged once, for the
    dearest piece in it."""
    dearest: dict[tuple[int, int], float] = {}
    for piece in pieces:
        upkeep = _dressed(
            TRACK_UPKEEP,
            piece,
            ELECTRIC_TRACK_UPKEEP,
            MAGLEV_TRACK_UPKEEP,
            DOUBLE_TRACK_UPKEEP_FACTOR,
        )
        dearest[piece.cell] = max(upkeep, dearest.get(piece.cell, 0.0))
    return sum(dearest.values())


@dataclass
class Finances:
    cash: float = START_CASH
    # Money in (above 0) and out (below 0) this month and last month, by what it was for.
    month: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    last_month: dict[str, float] = field(default_factory=dict)
    # Each year's money in and out, by what it was for, as months close; and the year of the
    # month on the books now, None until the first month closes.
    years: dict[int, dict[str, float]] = field(default_factory=dict)
    books_year: int | None = None
    # Each month's money in and out as months close: its year, its month in that year from 0,
    # and what it went on.
    months: list[tuple[int, int, dict[str, float]]] = field(default_factory=list)

    def can_afford(self, price: float) -> bool:
        return price <= self.cash + EPSILON

    def pay(self, price: float, what: str) -> bool:
        """Spends the price if there is cash enough; whether it was paid."""
        if not self.can_afford(price):
            return False
        self.spend(price, what)
        return True

    def spend(self, amount: float, what: str) -> None:
        self.cash -= amount
        self.month[what] -= amount

    def earn(self, amount: float, what: str) -> None:
        self.cash += amount
        self.month[what] += amount

    def close_month(self, year: int | None = None) -> None:
        """Starts a new month in the ledger, in `year` if given, adding the month just ended to
        its year's books."""
        filed = self.books_year if self.books_year is not None else year
        if filed is not None:
            books = self.years.setdefault(filed, {})
            for what, amount in self.month.items():
                books[what] = books.get(what, 0.0) + amount
            self.months.append((filed, self._month_number(filed), dict(self.month)))
        if year is not None:
            self.books_year = year
        self.last_month = dict(self.month)
        self.month = defaultdict(float)

    def _month_number(self, year: int) -> int:
        """Which month of `year` the month on the books now is, from 0."""
        if self.months and self.months[-1][0] == year:
            return self.months[-1][1] + 1
        return 0

    def periods(self, length: int, year: int) -> list[tuple[int, int, dict[str, float]]]:
        """Money in and out over every period of `length` months (1, 3 or 12) so far, oldest
        first: its year, which of that year's periods it is from 0, and what it went on. The last
        holds the month on the books now, in `year` until a month has closed."""
        found: dict[tuple[int, int], dict[str, float]] = {}
        filed = [*self.months]
        now = self.books_year if self.books_year is not None else year
        filed.append((now, self._month_number(now), self.month))
        for at, month, books in filed:
            period = found.setdefault((at, month // length), {})
            for what, amount in books.items():
                period[what] = period.get(what, 0.0) + amount
        return [(at, index, books) for (at, index), books in sorted(found.items())]

    def year_books(self, year: int) -> dict[str, float]:
        """A year's money in and out by what it was for, with this month's if it is on the
        books now."""
        books = dict(self.years.get(year, {}))
        if year == self.books_year or (self.books_year is None and year not in self.years):
            for what, amount in self.month.items():
                books[what] = books.get(what, 0.0) + amount
        return books


def dollars(amount: float) -> str:
    """An amount of money as the game shows it: $15,000, or -$1,200 in debt."""
    sign = "-" if amount < -0.5 else ""
    return f"{sign}${abs(amount):,.0f}"
