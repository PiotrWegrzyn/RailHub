"""The player's money: the cash a game starts with, what building costs, the upkeep charged
every month, and a ledger of the month so far and the month before.

Building needs the cash for it; upkeep is charged whatever the cash, so it can run into debt.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from railroads.sim.constructions import Track, TrackWidth
from railroads.sim.tilemap import Tile
from railroads.sim.track_rules import track_rules_for

START_CASH = 80_000.0
# Track costs TRACK_PRICE per cell times its bed's cost factor (railroads.sim.track_rules), and
# electrifying it ELECTRIC_TRACK_PRICE more; double track costs DOUBLE_TRACK_FACTOR as much.
TRACK_PRICE = 300.0
ELECTRIC_TRACK_PRICE = 450.0
DOUBLE_TRACK_FACTOR = 1.8
STATION_PRICE = 15_000.0
TRAIN_PRICE = 5_000.0
# What selling a train gives back, as a share of its price.
TRAIN_RESALE = 0.5
# What each car costs every time a train leaves a stop with it.
CAR_TRIP_PRICE = 20.0
# What raising a station to level 2, and then to level 3, costs; and each 50 more storage.
LEVEL_PRICES = (100_000.0, 500_000.0)
STORAGE_PRICE = 10_000.0
STATION_UPKEEP = 200.0
# Every cell of track costs TRACK_UPKEEP a month, ELECTRIC_TRACK_UPKEEP more where it is
# electric, and DOUBLE_TRACK_FACTOR as much where it is double.
TRACK_UPKEEP = 2.0
ELECTRIC_TRACK_UPKEEP = 3.0


def track_price(piece: Track, tile: Tile) -> float:
    """What laying this piece of track in this cell costs."""
    price = TRACK_PRICE * track_rules_for(tile, piece.bed).price_factor
    if piece.electric:
        price += ELECTRIC_TRACK_PRICE
    return price * (DOUBLE_TRACK_FACTOR if piece.width == TrackWidth.DOUBLE else 1.0)


def track_upkeep(pieces: Iterable[Track]) -> float:
    """What this track costs every month: each cell it runs through is charged once, for the
    dearest piece in it."""
    dearest: dict[tuple[int, int], float] = {}
    for piece in pieces:
        upkeep = TRACK_UPKEEP + (ELECTRIC_TRACK_UPKEEP if piece.electric else 0.0)
        upkeep *= DOUBLE_TRACK_FACTOR if piece.width == TrackWidth.DOUBLE else 1.0
        dearest[piece.cell] = max(upkeep, dearest.get(piece.cell, 0.0))
    return sum(dearest.values())


@dataclass
class Finances:
    cash: float = START_CASH
    # Money in (above 0) and out (below 0) this month and last month, by what it was for.
    month: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    last_month: dict[str, float] = field(default_factory=dict)

    def can_afford(self, price: float) -> bool:
        return price <= self.cash + 1e-9

    def spend(self, amount: float, what: str) -> None:
        self.cash -= amount
        self.month[what] -= amount

    def earn(self, amount: float, what: str) -> None:
        self.cash += amount
        self.month[what] += amount

    def close_month(self) -> None:
        """Starts a new month in the ledger."""
        self.last_month = dict(self.month)
        self.month = defaultdict(float)

    @property
    def profit(self) -> float:
        """This month's money in less money out so far."""
        return sum(self.month.values())


def dollars(amount: float) -> str:
    """An amount of money as the game shows it: $15,000, or -$1,200 in debt."""
    sign = "-" if amount < -0.5 else ""
    return f"{sign}${abs(amount):,.0f}"
