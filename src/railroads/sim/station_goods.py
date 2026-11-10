"""What a station holds: goods trains brought, waiting to be handed out, and its stockpile of spare
goods from around it and passengers and mail waiting for trains. The stockpile holds STOCKPILE
freight and WAITING passengers and mail, and each storage upgrade adds STORAGE_STEP to both."""

from dataclasses import dataclass, field

from railroads.sim.goods import TRAVELLERS, Good
from railroads.sim.trains import Parcels, take_parcels

# Most freight a station's stockpile holds, of all kinds together, and most passengers, and mail,
# waiting there.
STOCKPILE = 1000.0


WAITING = 100.0


STORAGE_STEP = 500


@dataclass
class StationGoods:
    """What a station holds: goods trains brought, waiting to be handed out, and its stockpile of
    spare goods from around it, waiting for trains."""

    brought: dict[Good, float] = field(default_factory=dict)
    stockpile: dict[Good, float] = field(default_factory=dict)
    # How many storage upgrades the station has had.
    storage: int = 0
    # Where each good trains brought came from.
    origins: dict[Good, Parcels] = field(default_factory=dict)
    # Room tech adds to every station's stockpile and waiting room (railroads.sim.tech).
    extra_room: float = 0.0
    extra_waiting: float = 0.0

    @property
    def limit(self) -> float:
        """The most freight the stockpile holds."""
        return STOCKPILE + STORAGE_STEP * self.storage + self.extra_room

    @property
    def waiting_limit(self) -> float:
        """The most passengers, and the most mail, that wait here."""
        return WAITING + STORAGE_STEP * self.storage + self.extra_waiting

    @property
    def room(self) -> float:
        return max(0.0, self.limit - self.stocked)

    @property
    def stocked(self) -> float:
        """How much freight the station holds, gathered and brought together; passengers and
        mail wait apart."""
        held = [*self.stockpile.items(), *self.brought.items()]
        return sum(amount for good, amount in held if good not in TRAVELLERS)

    def bring(self, good: Good, parcels: Parcels) -> None:
        """Takes in cargo a train unloaded."""
        known = self.origins.setdefault(good, {})
        for origin, amount in parcels.items():
            known[origin] = known.get(origin, 0.0) + amount
            self.brought[good] = self.brought.get(good, 0.0) + amount

    def release(self, good: Good, amount: float) -> Parcels:
        """Lets go of cargo trains brought, handed out or loaded onto another train."""
        held = self.brought.get(good, 0.0)
        amount = min(amount, held)
        self.brought[good] = held - amount
        return take_parcels(self.origins.setdefault(good, {}), amount, held)

    def board(self, good: Good, amount: float) -> None:
        """Lets passengers or mail come to the station to wait, up to its waiting limit."""
        self.stockpile[good] = min(self.waiting_limit, self.stockpile.get(good, 0.0) + amount)
