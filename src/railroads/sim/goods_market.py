"""The market for goods: what carrying each good earns falls as the world fills up with goods,
and moves with what the map wants of what it meets against what the map makes of that, a tug of
war like the one for fuel (railroads.sim.fuel_market).

People want more every decade (railroads.sim.events.RISE) and the land gives more still
(MAKE_RISE), so a unit of a good earns less every decade, five times less by the end of the game
(price_fall): the player has to carry ever more to earn as much.

Every need is a market. Each month it counts what the map's buildings wanted of it, and the most
the map's makers of the goods meeting it could make: what the land gives, and what processors'
inputs let them make. How hard demand pulls against supply is measured against the first
FIRST_MONTHS of play in which the map makes any of it, on average, in which goods earn their
prices (railroads.sim.goods.PRICES); after that they earn the ELASTICITY power of how many times
as hard it pulls, between LEAST_FACTOR and MOST_FACTOR, and the price moves each month FOLLOW of
the way there, so it changes smoothly: a good the map makes too much of gets cheap, and towns
growing to want it make it dear again. A good meeting several needs is priced by all of them,
each by how much of it was wanted. Passengers and mail have no market: they pay by the demand at
their station.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from railroads.sim.economy import Trade
from railroads.sim.events import price_fall
from railroads.sim.goods import FITS, TRAVELLERS, Good, Need

ELASTICITY = 0.5
FIRST_MONTHS = 36
LEAST_FACTOR = 0.25
MOST_FACTOR = 3.0
FOLLOW = 0.25

# The needs each good meets.
MEETS: Mapping[Good, tuple[Need, ...]] = MappingProxyType(
    {
        good: tuple(need for need, goods in FITS.items() if good in goods)
        for good in Good
        if good not in TRAVELLERS
    }
)


@dataclass(frozen=True, slots=True)
class NeedMonth:
    """One month of a need's market: the need, the year, what the map's buildings wanted of it
    and what its makers could make, and its price factor after it."""

    need: Need
    year: int
    wanted: float
    made: float
    factor: float

    @property
    def price(self) -> float:
        """What its goods earn, as a share of their prices: its factor, as much less as goods have
        grown plentiful by its year."""
        return self.factor * price_fall(self.year)


def signals_of(trade: Trade) -> dict[Need, tuple[float, float]]:
    """What a month's trade says about each need: what the map's buildings wanted of it and the
    most the makers of the goods meeting it could make."""
    wanted: dict[Need, float] = {}
    for (_, need), (_, want) in trade.goods_got.items():
        wanted[need] = wanted.get(need, 0.0) + float(want.sum())
    made: dict[Good, float] = {}
    for (_, good), (_, most) in trade.goods_gave.items():
        made[good] = made.get(good, 0.0) + float(most.sum())
    return {
        need: (want, sum(made.get(good, 0.0) for good in FITS.get(need, ())))
        for need, want in wanted.items()
        if want > 0
    }


@dataclass
class GoodsMarket:
    """Every need's price factor now, how hard demand pulled against supply in its first months,
    how much of it was wanted last month and its months so far."""

    factor: dict[Need, float] = field(default_factory=lambda: dict.fromkeys(Need, 1.0))
    year: int = 0
    first_pull: dict[Need, float] = field(default_factory=dict)
    made_months: dict[Need, int] = field(default_factory=dict)
    wanted: dict[Need, float] = field(default_factory=dict)
    history: dict[Need, list[NeedMonth]] = field(default_factory=dict)

    def factor_of(self, good: Good) -> float:
        """How many times its price a good earns now: as much less as goods have grown plentiful
        (railroads.sim.events.price_fall), times the factors of the needs it meets, each by how
        much of it was wanted. Passengers and mail earn their price."""
        if good in TRAVELLERS:
            return 1.0
        fall = price_fall(self.year)
        needs = MEETS.get(good, ())
        weights = [self.wanted.get(need, 0.0) for need in needs]
        total = sum(weights)
        if total <= 0:
            return fall
        market = sum(
            weight * self.factor[need] for need, weight in zip(needs, weights, strict=True)
        )
        return fall * market / total

    def month(self, year: int, trade: Trade) -> None:
        """Closes a month: moves each need's price factor with what the map wanted of it against
        what it could make, and keeps the month in its history."""
        self.year = year
        for need, (wanted, made) in signals_of(trade).items():
            months = self.history.setdefault(need, [])
            self.wanted[need] = wanted
            if made <= 0:
                months.append(NeedMonth(need, year, wanted, made, self.factor[need]))
                continue
            pull = wanted / made
            seen = self.made_months.get(need, 0)
            self.made_months[need] = seen + 1
            if seen < FIRST_MONTHS:
                first = self.first_pull.get(need, 0.0)
                self.first_pull[need] = (first * seen + pull) / (seen + 1)
            else:
                ratio = pull / self.first_pull[need] if self.first_pull[need] > 0 else 1.0
                target = min(MOST_FACTOR, max(LEAST_FACTOR, ratio**ELASTICITY))
                self.factor[need] += FOLLOW * (target - self.factor[need])
            months.append(NeedMonth(need, year, wanted, made, self.factor[need]))
