"""What a station holds: goods trains brought, waiting to be handed out, and its stockpile of spare
goods from around it and passengers and mail waiting for trains. The station holds up to STOCKPILE
of each kind of freight, gathered and brought together, so one good piling up leaves the others
their room, and WAITING passengers and mail. Each storage upgrade adds STORAGE_STEP to every kind of
freight and WAITING_STEP to the passengers and mail."""

from dataclasses import dataclass, field

from railroads.sim.goods import TRAVELLERS, Good
from railroads.sim.trains import Parcels, take_parcels

# Most of each kind of freight a station holds, and most passengers, and mail, waiting there.
STOCKPILE = 500.0
WAITING = 100.0
# What each storage upgrade adds to them.
STORAGE_STEP = 250
WAITING_STEP = 500


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
        """The most of each kind of freight the station holds."""
        return STOCKPILE + STORAGE_STEP * self.storage + self.extra_room

    @property
    def waiting_limit(self) -> float:
        """The most passengers, and the most mail, that wait here."""
        return WAITING + WAITING_STEP * self.storage + self.extra_waiting

    def held(self, good: Good) -> float:
        """How much of a good the station holds, gathered and brought together."""
        return self.stockpile.get(good, 0.0) + self.brought.get(good, 0.0)

    def room(self, good: Good) -> float:
        """How much more of a kind of freight the station can take."""
        return max(0.0, self.limit - self.held(good))

    @property
    def fullest(self) -> tuple[Good | None, float]:
        """The kind of freight the station holds most of, and how much; passengers and mail
        wait apart."""
        freight = {*self.stockpile, *self.brought} - set(TRAVELLERS)
        if not freight:
            return None, 0.0
        good = max(freight, key=lambda g: (self.held(g), g.value))
        return good, self.held(good)

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
