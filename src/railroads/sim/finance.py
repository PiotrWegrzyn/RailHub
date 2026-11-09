"""The player's money: the cash a game starts with, what building costs, the upkeep charged
every month, and a ledger of the month so far and the month before.

Building needs the cash for it; upkeep is charged whatever the cash, so it can run into debt.
"""

from collections import defaultdict
from dataclasses import dataclass, field

START_CASH = 100_000.0
# Track costs TRACK_PRICE per cell times its bed's cost factor (railroads.sim.track_rules).
TRACK_PRICE = 100.0
STATION_PRICE = 15_000.0
TRAIN_PRICE = 5_000.0
# What selling a train gives back, as a share of its price.
TRAIN_RESALE = 0.5
STATION_UPKEEP = 100.0
TRACK_UPKEEP = 1.0


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
