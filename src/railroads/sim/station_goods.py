"""What a station holds: goods trains brought, waiting to be handed out, and its stockpile of spare
goods from around it and passengers and mail waiting for trains. The station holds up to STOCKPILE
of each kind of freight, gathered and brought together, so one good piling up leaves the others
their room, and WAITING passengers and mail, who wait only a month before the next month's come
in their place. Each storage upgrade adds STORAGE_STEP to every kind of
freight and WAITING_STEP to the passengers and mail."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from railroads.sim.clock import QUARTER
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
    # The storage steps raising the station's level has added, which don't make bought storage
    # dearer.
    level_steps: int = 0
    # Goods kept for trains, which the station doesn't hand out to the buildings around it. A
    # frozenset, so saves from before it existed load with none kept.
    kept: frozenset[Good] = frozenset()
    # What came to the station in each of the last QUARTER months, the latest last: the freight
    # it gathered and the passengers and mail who came to wait.
    months_got: list[dict[Good, float]] = field(default_factory=list)

    def count_month(self, got: Mapping[Good, float]) -> None:
        """Counts what came to the station this month, forgetting months over a quarter ago."""
        self.months_got = [*self.months_got[1 - QUARTER :], dict(got)]

    def last_quarter(self) -> dict[Good, float]:
        """What came to the station over the last QUARTER months, good by good."""
        total: dict[Good, float] = {}
        for month in self.months_got:
            for good, amount in month.items():
                total[good] = total.get(good, 0.0) + amount
        return total

    @property
    def limit(self) -> float:
        """The most of each kind of freight the station holds."""
        return STOCKPILE + STORAGE_STEP * (self.storage + self.level_steps) + self.extra_room

    @property
    def waiting_limit(self) -> float:
        """The most passengers, and the most mail, that wait here."""
        return WAITING + WAITING_STEP * (self.storage + self.level_steps) + self.extra_waiting

    def held(self, good: Good) -> float:
        """How much of a good the station holds, gathered and brought together."""
        return self.stockpile.get(good, 0.0) + self.brought.get(good, 0.0)

    def room(self, good: Good) -> float:
        """How much more of a kind of freight the station can take."""
        return max(0.0, self.limit - self.held(good))

    def freight(self) -> dict[Good, float]:
        """Every kind of freight the station holds and how much, gathered and brought together:
        one stockpile, which it hands out and loads onto trains alike."""
        goods = {*self.stockpile, *self.brought} - set(TRAVELLERS)
        return {good: self.held(good) for good in goods if self.held(good) > 0}

    def waiting(self) -> dict[Good, float]:
        """The passengers and mail waiting at the station."""
        return {good: self.stockpile[good] for good in TRAVELLERS if self.stockpile.get(good)}

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

    def keep_for_trains(self, good: Good, kept: bool = True) -> None:
        """Keeps a good for trains, or lets the station hand it out around it again."""
        self.kept = self.kept | {good} if kept else self.kept - {good}

    def board(self, good: Good, amount: float) -> None:
        """Lets the month's passengers or mail come to the station to wait, up to its waiting
        limit, in place of last month's: those no train took have gone their own way."""
        self.stockpile[good] = min(self.waiting_limit, amount)
