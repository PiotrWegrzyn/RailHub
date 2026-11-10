"""What the buildings around a station want, and what cargo is worth there.

Demand is kept by need (railroads.sim.goods.Need): food, fruit, coal and so on. Every month a
station's capacity for each need is DEMAND_PER_NEED times what the buildings in its inner ring
need of it. Demand left is the capacity less what was used up of it, averaged over the last
MONTHS_REMEMBERED months, so a flood of cargo lowers the price for a few months before it climbs
back.

Freight trains brought earns its price (railroads.sim.goods.price_of) times the share
of demand left when buildings take it, and uses that need up. Goods a station gathered itself and
passes along to the buildings around it use up demand the same way, and earn PASSED_ALONG_PAY of
what a train delivery would have earned. Passengers and mail earn the same as they get off,
against the need they meet.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from railroads.sim.goods import FITS, Good, Need, price_of

DEMAND_PER_NEED = 2.0
MONTHS_REMEMBERED = 3
PASSED_ALONG_PAY = 0.3


@dataclass(frozen=True, slots=True)
class DemandRecord:
    """One good's demand at a station: its capacity and how much of it is left."""

    capacity: float
    left: float

    @property
    def share(self) -> float:
        return self.left / self.capacity if self.capacity > 0 else 0.0


@dataclass
class StationDemand:
    capacity: dict[Need, float] = field(default_factory=dict)
    # What was used up of each need in each of the last months, this month last.
    used: dict[Need, list[float]] = field(default_factory=dict)
    # Each need's demand as last month ended.
    last_month: dict[Need, DemandRecord] = field(default_factory=dict)

    def record(self, need: Need) -> DemandRecord:
        capacity = self.capacity.get(need, 0.0)
        used = sum(self.used.get(need, ())) / MONTHS_REMEMBERED
        return DemandRecord(capacity, max(0.0, capacity - used))

    def best_need(self, good: Good) -> Need | None:
        """Of the needs here a good meets, the one with the most demand left."""
        fits = [need for need in self.capacity if good in FITS[need] and self.capacity[need] > 0]
        return max(fits, key=lambda need: self.record(need).share, default=None)

    def monthly_want(self, good: Good) -> float:
        """How much of a good the buildings around could use in a month."""
        wants = (amount for need, amount in self.capacity.items() if good in FITS[need])
        return sum(wants) / DEMAND_PER_NEED

    def deliver(self, good: Good, amount: float) -> float:
        """Takes in passengers or mail getting off; returns what they earned, each at the share
        of demand left of the need it meets best as it came in."""
        earned = 0.0
        left = amount
        while left > 1e-9:
            unit = min(1.0, left)
            need = self.best_need(good)
            if need is None:
                break
            earned += self.pay(need, good, unit)
            left -= unit
        return earned

    def pay(self, need: Need, good: Good, amount: float, rate: float = 1.0) -> float:
        """What buildings taking this much of a good for a need earn, unit by unit, each at the
        share of demand left as it is taken, and at `rate` of the full price."""
        earned = 0.0
        left = amount
        while left > 1e-9:
            unit = min(1.0, left)
            earned += price_of(good) * unit * self.record(need).share * rate
            self._use(need, unit)
            left -= unit
        return earned

    def turn_month(self, capacity: Mapping[Need, float]) -> None:
        """Ends the month and starts the next, with the new capacity."""
        needs = set(self.capacity) | set(capacity) | set(self.used)
        self.last_month = {need: self.record(need) for need in needs if self.capacity.get(need)}
        self.capacity = dict(capacity)
        for need in needs:
            months = self.used.setdefault(need, [])
            months.append(0.0)
            del months[:-MONTHS_REMEMBERED]

    def _use(self, need: Need, amount: float) -> None:
        months = self.used.setdefault(need, [0.0])
        if not months:
            months.append(0.0)
        months[-1] += amount


def capacity_of(needs: Mapping[Need, float]) -> dict[Need, float]:
    """The demand capacity for what the buildings in a ring need each month."""
    return {need: DEMAND_PER_NEED * amount for need, amount in needs.items() if amount > 0}
