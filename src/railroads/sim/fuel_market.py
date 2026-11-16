"""The market for what engines burn: coal for steam, diesel fuel for diesels and electricity for
electric and maglev engines (railroads.sim.railway buys it as trains set off).

Every fuel's price, from the year it comes in, moves only with what the map wants of it against
what it gets, a tug of war. Coal and diesel fuel are wanted
by the buildings that need them and by the player's trains, which wanted what they bought last
month; they are supplied by what the player's stations hand out to those buildings, and a little,
NATURAL_SUPPLY of what the map's mines and refineries can make, by the world around. How hard
demand pulls against supply is measured against the first month of play, so a map with little
coal starts at the same price as one with much, and the price moves each month FOLLOW of the way
to where demand and supply put it, so it changes smoothly.

Electricity always has a sensible price, BASE_PRICE, which moves a little, between
ELECTRICITY_LEAST and ELECTRICITY_MOST of it, with supply against demand. It is wanted by the
player's electric and maglev trains and by the map's homes and factories, ELECTRICITY_PER_POINT for
each of their points times the share of them electrified, which grows by year from ELECTRIFICATION
to ELECTRIFIED; the buildings only pull the price, they need no power delivered. It is made by the
map's coal, diesel and gas power plants (PLANT_FUEL, railroads.sim.power_plants), POWER_PLANT_OUTPUT
a month each at full fuel and less on less. Power plants the player owns are counted among the
map's, and their trains use what those make
first, at OWN_POWER_SHARE of the price.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import numpy as np

from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Trade
from railroads.sim.goods import FITS, Good, Need, price_of
from railroads.sim.goods import NEEDS as BUILDING_NEEDS
from railroads.sim.trains import Power


class Fuel(Enum):
    COAL = "coal"
    DIESEL = "diesel fuel"
    ELECTRICITY = "electricity"


FUEL_OF: Mapping[Power, Fuel] = MappingProxyType(
    {
        Power.STEAM: Fuel.COAL,
        Power.DIESEL: Fuel.DIESEL,
        Power.ELECTRIC: Fuel.ELECTRICITY,
        Power.MAGLEV: Fuel.ELECTRICITY,
    }
)
# The good a fuel is, which trains can take from a station's stockpile; electricity is no good.
GOOD_OF: Mapping[Fuel, Good] = MappingProxyType({Fuel.COAL: Good.COAL, Fuel.DIESEL: Good.FUEL})
# Each kind of power plant and what it burns.
PLANT_FUEL: Mapping[BuildingKind, Need] = MappingProxyType(
    {
        BuildingKind.POWER_PLANT: Need.COAL,
        BuildingKind.DIESEL_POWER_PLANT: Need.FUEL,
        BuildingKind.GAS_POWER_PLANT: Need.GAS,
    }
)
# What a unit of each fuel costs at first, and the year its market opens.
BASE_PRICE: Mapping[Fuel, float] = MappingProxyType(
    {Fuel.COAL: price_of(Good.COAL), Fuel.DIESEL: price_of(Good.FUEL), Fuel.ELECTRICITY: 5.0}
)
OPENS: Mapping[Fuel, int] = MappingProxyType(
    {Fuel.COAL: 1804, Fuel.DIESEL: 1935, Fuel.ELECTRICITY: 1925}
)
# The buildings' needs each good fuel meets.
NEEDS: Mapping[Fuel, frozenset[Need]] = MappingProxyType(
    {
        fuel: frozenset(need for need, goods in FITS.items() if good in goods)
        for fuel, good in GOOD_OF.items()
    }
)
NATURAL_SUPPLY = 0.25
# How strongly the price follows demand against supply: the share it rises by is this power of
# how many times as hard demand pulls as in the first month; and the least and most that makes.
ELASTICITY = 0.5
LEAST_FACTOR = 0.4
MOST_FACTOR = 4.0
FOLLOW = 0.25
ELECTRICITY_LEAST = 0.7
ELECTRICITY_MOST = 1.5
ELECTRICITY_PER_POINT = 10.0
# Homes and factories are electrified bit by bit, from none in the first year to all in the second.
ELECTRIFICATION = (1880, 2030)
POWER_PLANT_OUTPUT = 3_000.0
OWN_POWER_SHARE = 0.2
# Fuel bought anywhere but out of a station's stockpile costs this much more.
RETAIL_MARKUP = 0.5


@dataclass(frozen=True, slots=True)
class FuelMonth:
    """One month of a fuel's market: the year, its price, what
    the buildings and the player's trains wanted, what the player's stations handed out and the
    world supplied, how hard demand pulled against supply then and in the first month, and for
    electricity what share of their fuel the map's power plants got and what the player's own
    made."""

    year: int
    price: float
    buildings: float = 0.0
    trains: float = 0.0
    network: float = 0.0
    natural: float = 0.0
    pull: float = 1.0
    first_pull: float = 1.0
    plants_fed: float = 0.0
    own_power: float = 0.0


@dataclass(frozen=True, slots=True)
class Signals:
    """What moved a fuel's market this month (FuelMonth)."""

    buildings: float = 0.0
    network: float = 0.0
    natural: float = 0.0
    plants_fed: float = 0.0
    own_power: float = 0.0


def electrified(year: int) -> float:
    """The share of the map's homes and factories that use electricity in a year."""
    first, last = ELECTRIFICATION
    return min(1.0, max(0.0, (year - first) / (last - first)))


def signals_of(
    trade: Trade,
    owned: Mapping[BuildingKind, np.ndarray] | None = None,
    points: float = 0.0,
    year: int = OPENS[Fuel.ELECTRICITY],
) -> dict[Fuel, Signals]:
    """What a month's trade says about each fuel: what the buildings needed of it, what the
    player's stations handed out of it to them and what the map's makers of it could make; for
    electricity what the map's homes and factories, worth `points` together, want of it in a
    year, what the map's power plants made and what share of their fuel they got, and what the
    ones the player owns (a count per cell of each kind, on the map's grid) made."""
    signals = {}
    for fuel, good in GOOD_OF.items():
        needs = NEEDS[fuel]
        wanted = sum(
            float(want.sum()) for (_, need), (_, want) in trade.goods_got.items() if need in needs
        )
        handed = sum(
            amount
            for by_station in (trade.handed_for, trade.supplied_for)
            for station in by_station.values()
            for (need, given), amount in station.items()
            if given == good and need in needs
        )
        makers = sum(
            float(most.sum()) for (_, made), (_, most) in trade.goods_gave.items() if made == good
        )
        signals[fuel] = Signals(wanted, handed, NATURAL_SUPPLY * makers)
    made, own, got_all, wanted_all = 0.0, 0.0, 0.0, 0.0
    for kind, need in PLANT_FUEL.items():
        got, want = trade.goods_got.get((kind, need), (None, None))
        if got is None or want is None or float(want.sum()) <= 0:
            continue
        per_plant = BUILDING_NEEDS[kind][need]
        made += POWER_PLANT_OUTPUT * float(np.minimum(got, want).sum()) / per_plant
        got_all += float(np.minimum(got, want).sum())
        wanted_all += float(want.sum())
        grid = None if owned is None else owned.get(kind)
        if grid is not None:
            share = np.where(want > 0, np.minimum(1.0, got / np.where(want > 0, want, 1.0)), 0.0)
            counts = np.asarray(grid).ravel()[trade.cells]
            own += POWER_PLANT_OUTPUT * float((counts * share).sum())
    fed = got_all / wanted_all if wanted_all > 0 else 0.0
    homes = ELECTRICITY_PER_POINT * points * electrified(year)
    signals[Fuel.ELECTRICITY] = Signals(homes, made, 0.0, fed, own)
    return signals


@dataclass
class FuelMarket:
    """Every fuel's price now, its months so far, and what the player's trains bought of it this
    month."""

    # Where demand and supply have moved each fuel's price, as a factor of its inflated price.
    factor: dict[Fuel, float] = field(default_factory=lambda: dict.fromkeys(Fuel, 1.0))
    first_pull: dict[Fuel, float] = field(default_factory=dict)
    history: dict[Fuel, list[FuelMonth]] = field(default_factory=lambda: {f: [] for f in Fuel})
    # This month: how much the trains bought, out of stations' stockpiles and at retail, and
    # what it cost; and what is left of the player's own power plants' electricity.
    from_stations: dict[Fuel, float] = field(default_factory=lambda: dict.fromkeys(Fuel, 0.0))
    at_retail: dict[Fuel, float] = field(default_factory=lambda: dict.fromkeys(Fuel, 0.0))
    spent: dict[Fuel, float] = field(default_factory=lambda: dict.fromkeys(Fuel, 0.0))
    own_power: float = 0.0
    # What the trains bought last month.
    last_bought: dict[Fuel, float] = field(default_factory=lambda: dict.fromkeys(Fuel, 0.0))

    def price(self, fuel: Fuel) -> float:
        """What a unit of a fuel costs on the market now."""
        return BASE_PRICE[fuel] * self.factor[fuel]

    def bought(self, fuel: Fuel) -> float:
        return self.from_stations[fuel] + self.at_retail[fuel]

    def buy(
        self, fuel: Fuel, amount: float, from_station: float = 0.0, markup: float = 0.0
    ) -> float:
        """What `amount` of a fuel costs a train setting off: `from_station` of it out of the
        station's stockpile at the market price, electricity from the player's own power plants
        first at OWN_POWER_SHARE of it, and the rest at retail, `markup` more."""
        price = self.price(fuel)
        cost = 0.0
        if fuel == Fuel.ELECTRICITY:
            own = min(amount, self.own_power)
            self.own_power -= own
            cost += own * price * OWN_POWER_SHARE
            self.from_stations[fuel] += own
            rest = amount - own
            cost += rest * price
            self.at_retail[fuel] += rest
        else:
            station = min(amount, from_station)
            cost += station * price + (amount - station) * price * (1.0 + markup)
            self.from_stations[fuel] += station
            self.at_retail[fuel] += amount - station
        self.spent[fuel] += cost
        return cost

    def month(self, year: int, signals: Mapping[Fuel, Signals]) -> None:
        """Closes a month: moves each fuel's price with what its demand and supply were, keeps the
        month in its history and starts the next."""
        for fuel in Fuel:
            got = signals.get(fuel, Signals())
            trains = self.bought(fuel)
            if year < OPENS[fuel]:
                self._reset(fuel, trains)
                continue
            if fuel == Fuel.ELECTRICITY:
                demand = got.buildings + trains
                supply = got.network + got.natural
                pull = demand / supply if supply > 0 else ELECTRICITY_MOST
                first = 1.0
                target = min(ELECTRICITY_MOST, max(ELECTRICITY_LEAST, pull**ELASTICITY))
            else:
                demand = got.buildings + trains
                supply = max(got.network + got.natural, 1e-6)
                pull = demand / supply if demand > 0 else 0.0
                first = self.first_pull.setdefault(fuel, pull if pull > 0 else 1.0)
                ratio = pull / first if first > 0 else 1.0
                target = min(MOST_FACTOR, max(LEAST_FACTOR, ratio**ELASTICITY))
            self.factor[fuel] += FOLLOW * (target - self.factor[fuel])
            self.history[fuel].append(
                FuelMonth(
                    year,
                    self.price(fuel),
                    got.buildings,
                    trains,
                    got.network,
                    got.natural,
                    pull,
                    first,
                    got.plants_fed,
                    got.own_power,
                )
            )
            self._reset(fuel, trains)
        self.own_power = signals.get(Fuel.ELECTRICITY, Signals()).own_power

    def _reset(self, fuel: Fuel, trains: float) -> None:
        self.last_bought[fuel] = trains
        self.from_stations[fuel] = self.at_retail[fuel] = self.spent[fuel] = 0.0
