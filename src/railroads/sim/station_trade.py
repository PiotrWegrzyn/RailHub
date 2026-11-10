"""Where growth meets the player's stations (railroads.sim.growth keeps one StationTrade): what
each station offers the month's trade, settling what it handed out, gathered and was paid for,
the passengers and mail the buildings around it send, and the experience the player gets when
buildings rise within its reach.

A station sends a piece of mail for every MAIL_PER_PASSENGERS passengers. A building rising in a
station's reach gives the experience of TRAIN_LOAD units of its goods, a full train of the first
engine, for each of its stars.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind
from railroads.sim.demand import capacity_of
from railroads.sim.economy import DELIVERY_RADIUS, StationOrder, Trade
from railroads.sim.goods import FITS, NEEDS, TRAVELLERS, Good, Need, makes_of, needs_of, price_of
from railroads.sim.grids import Cell
from railroads.sim.ladders import PASSENGERS_PER_POINT
from railroads.sim.railway import CAR_CAPACITY, MAX_CARS
from railroads.sim.tech import GOOD_TREES, Tree

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
                dict(goods.brought), goods.room, delivery, collection, freight
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
        for station in railway.stations():
            ring = self.inner_ring(station)
            travellers = self.travellers_in(ring)
            goods = railway.station_goods(station)
            for good, amount in travellers.items():
                goods.board(good, amount)
            capacity[station] = capacity_of(self.needs_in(ring) | as_needs(travellers))
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
            ring = self.inner_ring(station)
            needs = self.needs_in(ring) | as_needs(self.travellers_in(ring))
            railway.station_demand(station).capacity = capacity_of(needs)

    def inner_ring(self, station: Cell) -> npt.NDArray[np.intp]:
        """The cells a station hands goods out to, as flat indices."""
        railway, grid = self._growth.railway, self._growth.grid
        radius = DELIVERY_RADIUS if railway is None else railway.reach(station)[0]
        reached = grid.within(station, radius)
        return np.array([np.ravel_multi_index(c, grid.shape) for c in reached])

    def travellers_in(self, cells: npt.NDArray[np.intp]) -> dict[Good, float]:
        """The passengers and mail the buildings in these cells send, and want, in a month."""
        passengers = 0
        for kind, count in self._growth.counts.items():
            points, value = self._growth.rule.points[kind]
            passengers += (
                PASSENGERS_PER_POINT.get(points, 0) * value * int(count.ravel()[cells].sum())
            )
        return {
            Good.PASSENGERS: float(passengers),
            Good.MAIL: float(passengers // MAIL_PER_PASSENGERS),
        }

    def needs_in(self, cells: npt.NDArray[np.intp]) -> dict[Need, float]:
        """What the buildings in these cells need in a month."""
        counts, year = self._growth.counts, self._growth.year
        needs: dict[Need, float] = {}
        for kind in NEEDS:
            if kind not in counts:
                continue
            count = float(counts[kind].ravel()[cells].sum())
            for need, amount in needs_of(kind, year).items():
                needs[need] = needs.get(need, 0.0) + count * amount
        return needs

    def update_reach(self) -> None:
        """Works out again which cells the player's stations hand goods out to."""
        railway, grid = self._growth.railway, self._growth.grid
        reached: set[Cell] = set()
        if railway is not None:
            for station in railway.stations():
                reached |= set(grid.within(station, railway.reach(station)[0]))
        self.in_reach = reached

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
