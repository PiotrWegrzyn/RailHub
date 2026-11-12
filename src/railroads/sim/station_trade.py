"""Where growth meets the player's stations (railroads.sim.growth keeps one StationTrade): what
each station offers the month's trade, settling what it handed out, gathered and was paid for,
the passengers and mail the buildings around it send, and the experience the player gets when
buildings rise within its reach.

A station sends a piece of mail for every MAIL_PER_PASSENGERS passengers. A building rising in a
station's reach gives the experience of TRAIN_LOAD units of its goods, a full train of the first
engine, for each of its stars.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind
from railroads.sim.demand import capacity_of
from railroads.sim.economy import COLLECTION_RADIUS, DELIVERY_RADIUS, StationOrder, Trade
from railroads.sim.goods import (
    FITS,
    NEEDS,
    RECIPES,
    TRAVELLERS,
    Good,
    Need,
    makes_of,
    needs_of,
    price_of,
)
from railroads.sim.grids import Cell, flat_within
from railroads.sim.ladders import PASSENGERS_PER_POINT
from railroads.sim.tech import GOOD_TREES, Tree
from railroads.sim.trains import CAR_CAPACITY, MAX_CARS

if TYPE_CHECKING:
    from railroads.sim.growth import Growth

MAIL_PER_PASSENGERS = 5
TRAIN_LOAD = MAX_CARS * CAR_CAPACITY


class StationTrade:
    def __init__(self, growth: Growth) -> None:
        self._growth = growth
        # The cells the player's stations hand goods out to, and the experience growth there
        # has given since the game began: where, what it became and how much.
        self.in_reach: set[Cell] = set()
        self.rewards: list[tuple[Cell, BuildingKind, float]] = []

    def orders(self) -> dict[Cell, StationOrder]:
        railway = self._growth.railway
        if railway is None:
            return {}
        orders = {}
        for station in sorted(railway.connected()):
            goods = railway.station_goods(station)
            delivery, collection = railway.reach(station)
            freight = {g: a for g, a in goods.stockpile.items() if g not in TRAVELLERS}
            orders[station] = StationOrder(
                dict(goods.brought), goods.limit, delivery, collection, freight
            )
        return orders

    def settle(self, trade: Trade) -> None:
        railway = self._growth.railway
        if railway is None:
            return
        for station, handed in trade.handed.items():
            goods = railway.station_goods(station)
            for good, amount in handed.items():
                goods.release(good, amount)
        for station, supplied in trade.supplied.items():
            goods = railway.station_goods(station)
            for good, amount in supplied.items():
                goods.stockpile[good] = max(0.0, goods.stockpile.get(good, 0.0) - amount)
        for station, gathered in trade.gathered.items():
            goods = railway.station_goods(station)
            for good, amount in gathered.items():
                goods.stockpile[good] = goods.stockpile.get(good, 0.0) + amount
        capacity = {}
        stations = railway.stations()
        for station, counts in zip(stations, self._ring_counts(stations), strict=True):
            travellers = self._travellers(counts)
            goods = railway.station_goods(station)
            for good, amount in travellers.items():
                goods.board(good, amount)
            capacity[station] = capacity_of(self._needs(counts) | as_needs(travellers))
        railway.year = self._growth.year
        railway.turn_month(capacity)
        for station, handed_for in trade.handed_for.items():
            for (need, good), amount in handed_for.items():
                railway.pay_for(station, need, good, amount)
        for station, supplied_for in trade.supplied_for.items():
            for (need, good), amount in supplied_for.items():
                railway.pay_for_passing(station, need, good, amount)

    def measure_demand(self, station: Cell) -> None:
        """Sets a new station's demand capacity from the buildings around it now, so cargo pays
        before its first month turns over."""
        railway = self._growth.railway
        if railway is not None:
            (counts,) = self._ring_counts([station])
            needs = self._needs(counts) | as_needs(self._travellers(counts))
            railway.station_demand(station).capacity = capacity_of(needs)

    def inner_ring(self, station: Cell) -> npt.NDArray[np.intp]:
        """The cells a station hands goods out to, as flat indices that are kept and so can't be
        written to."""
        railway = self._growth.railway
        radius = DELIVERY_RADIUS if railway is None else railway.reach(station)[0]
        return flat_within(self._growth.grid, station, radius).cells

    def supplies(self, station: Cell) -> dict[Good, float]:
        """What the station has to send on each month: the goods buildings make within the
        cells it gathers from, the most each processor there makes, and the passengers and mail
        the buildings it hands out to send."""
        railway = self._growth.railway
        radius = COLLECTION_RADIUS if railway is None else railway.reach(station)[1]
        cells = flat_within(self._growth.grid, station, radius).cells
        made: dict[Good, float] = {}
        for kind, count in self._growth.counts.items():
            n = float(count.ravel()[cells].sum())
            if n <= 0:
                continue
            goods = dict(makes_of(kind, self._growth.year))
            if kind in RECIPES:
                goods[RECIPES[kind].makes] = RECIPES[kind].most
            for good, amount in goods.items():
                made[good] = made.get(good, 0.0) + n * amount
        for good, amount in self.travellers_in(self.inner_ring(station)).items():
            if amount > 0:
                made[good] = amount
        return made

    def travellers_in(self, cells: npt.NDArray[np.intp]) -> dict[Good, float]:
        """The passengers and mail the buildings in these cells send, and want, in a month."""
        counts = self._growth.counts
        return self._travellers(
            {kind: int(count.ravel()[cells].sum()) for kind, count in counts.items()}
        )

    def _ring_counts(self, stations: list[Cell]) -> list[dict[BuildingKind, int]]:
        """How many buildings of each kind stand in each station's inner ring. The rings are laid
        end to end and summed a kind at a time for all the stations at once; none is empty, as
        each holds its station's own cell."""
        if not stations:
            return []
        rings = [self.inner_ring(station) for station in stations]
        starts = np.cumsum([0, *(ring.size for ring in rings[:-1])])
        cells = np.concatenate(rings)
        counts = self._growth.counts
        sums = [np.add.reduceat(count.ravel()[cells], starts).tolist() for count in counts.values()]
        return [dict(zip(counts, summed, strict=True)) for summed in zip(*sums, strict=True)]

    def _travellers(self, counts: Mapping[BuildingKind, int]) -> dict[Good, float]:
        """The passengers and mail these many buildings of each kind send, and want, in a month."""
        passengers = 0
        for kind, count in counts.items():
            points, value = self._growth.rule.points[kind]
            passengers += PASSENGERS_PER_POINT.get(points, 0) * value * count
        return {
            Good.PASSENGERS: float(passengers),
            Good.MAIL: float(passengers // MAIL_PER_PASSENGERS),
        }

    def _needs(self, counts: Mapping[BuildingKind, int]) -> dict[Need, float]:
        """What these many buildings of each kind need in a month."""
        year = self._growth.year
        needs: dict[Need, float] = {}
        for kind in NEEDS:
            if kind not in counts:
                continue
            count = float(counts[kind])
            for need, amount in needs_of(kind, year).items():
                needs[need] = needs.get(need, 0.0) + count * amount
        return needs

    def update_reach(self) -> None:
        """Works out again which cells the player's stations hand goods out to."""
        railway, cols = self._growth.railway, self._growth.grid.cols
        stations = [] if railway is None else railway.stations()
        self.in_reach = {
            divmod(index, cols)
            for station in stations
            for index in self.inner_ring(station).tolist()
        }

    def reward(self, cell: Cell, kind: BuildingKind) -> None:
        """Gives the player experience when a building in their stations' reach rises: as much
        as a train full of its goods would earn, for every star it has."""
        railway = self._growth.railway
        if railway is None or cell not in self.in_reach:
            return
        stars = self._growth.stars(kind)
        if stars < 2:
            return
        gained = 0.0
        for tree, worth in self._load_worth(kind).items():
            xp = stars * TRAIN_LOAD * worth
            railway.research.earn(tree, xp)
            gained += xp
        self.rewards.append((cell, kind, gained))

    def _load_worth(self, kind: BuildingKind) -> dict[Tree, float]:
        """What one unit of a kind's goods is worth to each tree: what it makes, or for those
        that make nothing, the mix of what they need."""
        year = self._growth.year
        makes = makes_of(kind, year)
        if makes:
            goods = {good: amount for good, amount in makes.items()}
        else:
            goods = {FITS[n][0]: amount for n, amount in needs_of(kind, year).items()}
        total = sum(goods.values())
        worth: dict[Tree, float] = {}
        for good, amount in goods.items():
            tree = GOOD_TREES[good]
            worth[tree] = worth.get(tree, 0.0) + amount / total * price_of(good)
        return worth


def as_needs(travellers: dict[Good, float]) -> dict[Need, float]:
    return {Need(good.value): amount for good, amount in travellers.items()}
