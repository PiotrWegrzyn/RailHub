"""What the buildings around a station want, and what cargo trains bring there is worth.

Every month a station's capacity for each good is DEMAND_PER_NEED times what the buildings in
its inner ring (DELIVERY_RADIUS) need of it. What they used up of it is what they consumed from
any source, the land, their neighbours or the station, plus every unit trains unloaded there.
Demand left is the capacity less the use averaged over the last MONTHS_REMEMBERED months, so a
town fed by itself has half its demand left, and a flood of cargo lowers the price for a few
months before it climbs back.

Each unit unloaded earns CARGO_PRICE times the share of demand left as it arrives.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from railroads.sim.economy import Good

DEMAND_PER_NEED = 2.0
MONTHS_REMEMBERED = 3
CARGO_PRICE = 20.0


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
    capacity: dict[Good, float] = field(default_factory=dict)
    # What was used up of each good in each of the last months, this month last.
    used: dict[Good, list[float]] = field(default_factory=dict)
    # Each good's demand as last month ended.
    last_month: dict[Good, DemandRecord] = field(default_factory=dict)

    def record(self, good: Good) -> DemandRecord:
        capacity = self.capacity.get(good, 0.0)
        used = sum(self.used.get(good, ())) / MONTHS_REMEMBERED
        return DemandRecord(capacity, max(0.0, capacity - used))

    def deliver(self, good: Good, amount: float) -> float:
        """Takes in cargo a train unloaded; returns what it earned, each unit at the share of
        demand left as it came in."""
        earned = 0.0
        left = amount
        while left > 1e-9:
            unit = min(1.0, left)
            earned += CARGO_PRICE * unit * self.record(good).share
            self._use(good, unit)
            left -= unit
        return earned

    def turn_month(self, capacity: Mapping[Good, float], consumed: Mapping[Good, float]) -> None:
        """Ends the month and starts the next: the new capacity, and what the buildings around
        consumed from anywhere but the station as it began."""
        goods = set(self.capacity) | set(capacity) | set(self.used)
        self.last_month = {good: self.record(good) for good in goods if self.capacity.get(good)}
        self.capacity = dict(capacity)
        for good in goods:
            months = self.used.setdefault(good, [])
            months.append(consumed.get(good, 0.0))
            del months[:-MONTHS_REMEMBERED]

    def _use(self, good: Good, amount: float) -> None:
        months = self.used.setdefault(good, [0.0])
        if not months:
            months.append(0.0)
        months[-1] += amount


def capacity_of(needs: Mapping[Good, float]) -> dict[Good, float]:
    """The demand capacity for what the buildings in a ring need each month."""
    return {good: DEMAND_PER_NEED * amount for good, amount in needs.items() if amount > 0}
