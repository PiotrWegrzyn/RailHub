"""Where growth meets the player's stations (railroads.sim.growth keeps one StationTrade): what
each station offers the month's trade, settling what it handed out, gathered and was paid for,
the passengers and mail the buildings around it send.

A station sends a piece of mail for every MAIL_PER_PASSENGERS passengers.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind
from railroads.sim.demand import capacity_of
from railroads.sim.economy import (
    COLLECTION_RADIUS,
    DELIVERY_RADIUS,
    FloatCells,
    StationOrder,
    Trade,
    distribute,
    input_stock,
    ring_starts,
    wants_of,
    worth_to,
)
from railroads.sim.events import travel_factor
from railroads.sim.facilities import USES as FACILITY_USES
from railroads.sim.goods import (
    BONUS_PART,
    FITS,
    NEEDS,
    OWNED_BOOSTS,
    PIPELINE,
    RECIPES,
    TRAVELLERS,
    Good,
    Need,
    makes_of,
    needs_of,
)
from railroads.sim.grids import Cell, Within, flat_within_any
from railroads.sim.ladders import PASSENGERS_PER_POINT
from railroads.sim.power_plants import powered_points
from railroads.sim.station_goods import StationGoods
from railroads.sim.steps import Steps, finish

if TYPE_CHECKING:
    from railroads.sim.growth import Growth

MAIL_PER_PASSENGERS = 5
# How far from a building the town it stands in is looked for.
TOWN_NEAR = 2
# The needs each good meets, in the order a month's trade hands them out.
MEETS: dict[Good, tuple[Need, ...]] = {
    good: tuple(step for step in PIPELINE if isinstance(step, Need) and good in FITS[step])
    for good in Good
}


class StationTrade:
    def __init__(self, growth: Growth) -> None:
        self._growth = growth
        # The cells the player's stations hand goods out to.
        self.in_reach: set[Cell] = set()
        # What the buildings of each kind took of each need, cell by cell, as trains arrived this
        # month.
        self._arrived: dict[tuple[BuildingKind, Need], dict[int, float]] = {}

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
                dict(goods.brought),
                goods.limit,
                delivery,
                collection,
                freight,
                goods.kept,
                tuple(self._growth.constructions.platforms(station)),
            )
        return orders

    def settle(self, trade: Trade) -> None:
        finish(self.settle_steps(trade))

    def settle_steps(self, trade: Trade) -> Steps[None]:
        """settle in a few steps, which a caller can spread over frames; nothing else may change
        the game between them."""
        self._arrived = {}
        railway = self._growth.railway
        if railway is None:
            return
        for station, owned in trade.owned_brought.items():
            goods = railway.station_goods(station)
            for good, amount in owned.items():
                goods.bring(good, {station: amount})
        for station, handed in trade.handed.items():
            goods = railway.station_goods(station)
            for good, amount in handed.items():
                from_trains = min(amount, goods.brought.get(good, 0.0))
                goods.release(good, from_trains)
                if amount > from_trains:
                    goods.stockpile[good] = max(
                        0.0, goods.stockpile.get(good, 0.0) - (amount - from_trains)
                    )
        for station, gathered in trade.gathered.items():
            goods = railway.station_goods(station)
            for good, amount in gathered.items():
                goods.stockpile[good] = goods.stockpile.get(good, 0.0) + amount
        railway.fuel_month(trade, powered_points(self._growth))
        yield
        capacity = {}
        stations = railway.stations()
        for station, counts in zip(stations, self._ring_counts(stations), strict=True):
            travellers = self._travellers(counts)
            goods = railway.station_goods(station)
            for good, amount in travellers.items():
                goods.board(good, amount)
            got = dict(trade.gathered.get(station, {}))
            goods.count_month(got | {good: goods.stockpile[good] for good in travellers})
            needs = self._short_after_local(station, self._needs(counts, station), trade)
            capacity[station] = capacity_of(needs | as_needs(travellers))
        railway.year = self._growth.year
        railway.turn_month(capacity)
        yield
        for owned in trade.owned_brought.values():
            for good, amount in owned.items():
                railway.sell_own(good, amount)
        for station, handed_for in trade.handed_for.items():
            worths = trade.handed_worth.get(station, {})
            for (need, good), amount in handed_for.items():
                railway.pay_for(station, need, good, amount, worths.get((need, good), amount))

    def _short_after_local(
        self, station: Cell, needs: dict[Need, float], trade: Trade | None
    ) -> dict[Need, float]:
        """What the buildings a station hands goods out to need, less what they got this month
        from the goods buildings around handed out on their own."""
        if trade is None or not trade.cells.size or not trade.local:
            return needs
        ring = self.inner_ring(station)
        positions = np.searchsorted(trade.cells, ring)
        inside = (positions < trade.cells.size) & (
            trade.cells[np.minimum(positions, trade.cells.size - 1)] == ring
        )
        positions = positions[inside]
        short = {}
        for need, amount in needs.items():
            local = trade.local.get(need)
            got = float(local[positions].sum()) if local is not None else 0.0
            short[need] = max(0.0, amount - got)
        return short

    def hand_out_now(self, station: Cell, good: Good) -> None:
        """Hands a good trains have just brought to a station to the buildings around it that
        still want it this month, at once, unless the good is kept for trains: the nearest
        first, as in a month's trade (railroads.sim.economy.distribute). Each takes up to what it
        needs in a month, less what it took already; a processor up to the room in its
        stockpile. They pay as they take."""
        growth, railway = self._growth, self._growth.railway
        if railway is None or good in TRAVELLERS:
            return
        goods = railway.station_goods(station)
        if good in goods.kept:
            return
        cells, bounds = self._in_reach(station)
        economy = growth.economy
        if economy.taken is None:
            economy.taken = {}
        for need in MEETS[good]:
            have = goods.brought.get(good, 0.0)
            if have <= 0:
                return
            wants = self._wants_at(cells, need)
            if not wants:
                continue
            got, demand = self._share_out(cells, bounds, need, have, wants)
            total = float(got.sum())
            weighed = float((got * worth_to(economy.last_quarter()[cells])).sum())
            rate = weighed / total if total > 0 else 1.0
            if self._give(station, goods, need, good, total, rate) <= 0:
                continue
            met = np.where(demand > 0, got / np.where(demand > 0, demand, 1.0), 0.0)
            for kind, want in wants.items():
                took = want * met
                for at in np.flatnonzero(took > 0).tolist():
                    cell = int(cells[at])
                    arrived = self._arrived.setdefault((kind, need), {})
                    arrived[cell] = arrived.get(cell, 0.0) + float(took[at])
                    if kind in RECIPES:
                        economy.stock[(kind, need)][cell] += took[at]
                    else:
                        taken = economy.taken.setdefault((kind, need), {})
                        taken[cell] = taken.get(cell, 0.0) + float(took[at])

    def would_take(self, station: Cell, cargo: list[tuple[Good, float]]) -> list[float]:
        """What each load of cargo would be worth to the buildings around a station if a train
        unloaded it there now, one load after another, changing nothing: the units they would
        take at once, as hand_out_now hands them out, each weighed by what it is worth to the
        building taking it (worth_to), what pay_for pays freight on. Passengers and mail, goods
        kept for trains and what doesn't fit in the station are worth nothing here."""
        railway = self._growth.railway
        if railway is None:
            return [0.0 for _ in cargo]
        goods = railway.station_goods(station)
        cells, bounds = self._in_reach(station)
        worth = worth_to(self._growth.economy.last_quarter()[cells])
        assigned: dict[Need, FloatCells] = {}
        room: dict[Good, float] = {}
        weighed = []
        for good, amount in cargo:
            if good in TRAVELLERS or good in goods.kept:
                weighed.append(0.0)
                continue
            have = min(amount, room.setdefault(good, goods.room(good)))
            room[good] -= have
            units = 0.0
            for need in MEETS[good]:
                if have <= 1e-9:
                    break
                wants = self._wants_at(cells, need)
                if not wants:
                    continue
                demand = sum(wants.values(), start=np.zeros(cells.size))
                before = assigned.get(need, np.zeros(cells.size))
                left = np.maximum(1.0 - before / np.where(demand > 0, demand, 1.0), 0.0)
                got, _ = self._share_out(
                    cells, bounds, need, have, {kind: want * left for kind, want in wants.items()}
                )
                assigned[need] = before + got
                have -= float(got.sum())
                units += float((got * worth).sum())
            weighed.append(units)
        return weighed

    def _in_reach(self, station: Cell) -> tuple[npt.NDArray[np.intp], tuple[int, ...]]:
        """The cells a station hands goods out to, nearest first, and where each distance
        starts among them."""
        railway = self._growth.railway
        assert railway is not None
        within = self._within(station, railway.reach(station)[0])
        order = np.argsort(within.steps, kind="stable")
        return within.cells[order], ring_starts(within.steps[order])

    def _share_out(
        self,
        cells: npt.NDArray[np.intp],
        bounds: tuple[int, ...],
        need: Need,
        have: float,
        wants: dict[BuildingKind, FloatCells],
    ) -> tuple[FloatCells, FloatCells]:
        """How much of `have` each of these cells takes for a need, by what its buildings still
        want: the player's own buildings' musts first, then every must, then any bonus, the
        nearest first; and what they wanted in all."""
        demand = sum(wants.values(), start=np.zeros(cells.size))
        left = np.array([have], dtype=np.float64)
        sellers, buyers = np.zeros(cells.size, dtype=np.intp), np.arange(cells.size)
        mine = self._owned_wants(cells, wants)
        must = sum(
            (
                np.maximum(want - self._spare(kind, need, cells), 0.0)
                for kind, want in wants.items()
            ),
            start=np.zeros(cells.size),
        )
        got = np.zeros(cells.size)
        for target in (np.minimum(mine, must), must, demand):
            short = np.maximum(target - got, 0.0)
            wanted = short.copy()
            distribute(sellers, buyers, bounds, left, short)
            got += wanted - short
        return got, demand

    def _spare(self, kind: BuildingKind, need: Need, cells: npt.NDArray[np.intp]) -> FloatCells:
        """The part of a need the buildings of a kind in these cells take only as a bonus
        (BONUS_PART), handed out after everyone has had what they must."""
        extra = BONUS_PART.get((kind, need))
        if extra is None:
            return np.zeros(cells.size)
        return self._growth.counts[kind].ravel()[cells].astype(np.float64) * extra

    def _owned_wants(
        self, cells: npt.NDArray[np.intp], wants: dict[BuildingKind, FloatCells]
    ) -> FloatCells:
        """The part of these wants that is the player's own buildings', which a station hands
        out before anyone else's."""
        growth = self._growth
        mine = np.zeros(cells.size)
        for kind, want in wants.items():
            owned = growth.owners.owned(kind).ravel()[cells]
            if owned.any():
                count = growth.counts[kind].ravel()[cells]
                mine += want * np.minimum(owned / np.maximum(count, 1), 1.0)
        return mine

    def _wants_at(self, cells: npt.NDArray[np.intp], need: Need) -> dict[BuildingKind, FloatCells]:
        """What each kind of building in these cells still wants of a need this month: up to
        what it needs in a month, less what it took already; a processor up to the room in its
        stockpile."""
        growth, economy = self._growth, self._growth.economy
        taken_by = economy.taken or {}
        wants: dict[BuildingKind, FloatCells] = {}
        for kind, count in growth.counts.items():
            if need not in wants_of(kind, growth.year):
                continue
            n = count.ravel()[cells].astype(np.float64)
            if (kind, need) in OWNED_BOOSTS:
                taking = growth.owners.taking(kind, need)
                n = np.zeros(cells.size) if taking is None else taking[cells]
            if not n.any():
                continue
            if kind in RECIPES:
                want = input_stock(kind, need) * n - economy.stock[(kind, need)][cells]
            else:
                taken = taken_by.get((kind, need), {})
                had = np.array([taken.get(int(cell), 0.0) for cell in cells.tolist()])
                want = n * needs_of(kind, growth.year, economy.east)[need] - had
            short = self._short_last_month(cells, kind, need)
            if short is not None:
                arrived = self._arrived.get((kind, need), {})
                got = np.array([arrived.get(int(cell), 0.0) for cell in cells.tolist()])
                want = np.minimum(want, short - got)
            wants[kind] = np.where(n > 0, np.maximum(want, 0.0), 0.0)
        return wants

    def _short_last_month(
        self, cells: npt.NDArray[np.intp], kind: BuildingKind, need: Need
    ) -> FloatCells | None:
        """What the buildings of a kind in these cells were left short of a need last month after
        the goods handed out around them on their own: all a station may hand them as trains
        arrive, so a train never takes the place of the buildings nearby. Cells that did not
        trade last month are not held back; None before any trade."""
        trade = self._growth.trade
        if trade is None or (kind, need) not in trade.goods_got or not trade.cells.size:
            return None
        _, wanted = trade.goods_got[(kind, need)]
        total = sum(
            (w for (_, n), (_, w) in trade.goods_got.items() if n == need),
            start=np.zeros(trade.cells.size),
        )
        local = trade.local.get(need, np.zeros(trade.cells.size))
        fed = np.minimum(local / np.where(total > 0, total, 1.0), 1.0)
        short = wanted * (1.0 - fed)
        positions = np.searchsorted(trade.cells, cells)
        found = (positions < trade.cells.size) & (
            trade.cells[np.minimum(positions, trade.cells.size - 1)] == cells
        )
        return np.where(found, short[np.minimum(positions, trade.cells.size - 1)], np.inf)

    def _give(
        self,
        station: Cell,
        goods: StationGoods,
        need: Need,
        good: Good,
        want: float,
        rate: float = 1.0,
    ) -> float:
        """Gives up to `want` of a good trains brought to a station, and pays its freight, each
        unit worth `rate` to the buildings that took it; returns how much went."""
        railway = self._growth.railway
        assert railway is not None
        given = min(want, goods.brought.get(good, 0.0))
        if given > 0:
            goods.release(good, given)
            railway.pay_for(station, need, good, given, given * rate)
        return max(given, 0.0)

    @property
    def has_traded(self) -> bool:
        """Whether the buildings have had a month's trade yet, so what they are short of is known."""
        return self._growth.trade is not None

    def short_of(self, station: Cell) -> dict[Need, float]:
        """What the buildings a station hands goods out to are still short of this month, need by
        need: what last month's trade left them wanting, less what they took since; and what its
        facilities lacked this month."""
        short: dict[Need, float] = {}
        railway = self._growth.railway
        if railway is not None:
            for facility in railway.facilities_at(station) - railway.working_at(station):
                good, amount = FACILITY_USES[facility]
                short[Need(good.value)] = short.get(Need(good.value), 0.0) + amount
        trade = self._growth.trade
        if trade is None or not trade.cells.size:
            return short
        ring = self.inner_ring(station)
        positions = np.searchsorted(trade.cells, ring)
        inside = (positions < trade.cells.size) & (
            trade.cells[np.minimum(positions, trade.cells.size - 1)] == ring
        )
        positions, ring = positions[inside], ring[inside]
        taken = self._growth.economy.taken or {}
        for (kind, need), (received, wanted) in trade.goods_got.items():
            lacking = np.maximum(wanted[positions] - received[positions], 0.0)
            since = taken.get((kind, need), {})
            if since:
                lacking = np.maximum(lacking - [since.get(int(c), 0.0) for c in ring], 0.0)
            if lacking.sum() > 0:
                short[need] = short.get(need, 0.0) + float(lacking.sum())
        return short

    def measure_demand(self, station: Cell) -> None:
        """Sets a new station's demand capacity from the buildings around it now, so cargo pays
        before its first month turns over."""
        railway = self._growth.railway
        if railway is not None:
            (counts,) = self._ring_counts([station])
            needs = self._short_after_local(
                station, self._needs(counts, station), self._growth.trade
            )
            needs |= as_needs(self._travellers(counts))
            railway.station_demand(station).capacity = capacity_of(needs)

    def town_of(self, station: Cell) -> int:
        """The town sending the most passengers from the cells a station hands goods out to."""
        return self._growth.towns.most_in(self.inner_ring(station))

    def town_near(self, cell: Cell) -> int:
        """The town sending the most passengers from the cells within TOWN_NEAR of a cell."""
        return self._growth.towns.most_in(
            flat_within_any(self._growth.grid, (cell,), TOWN_NEAR).cells
        )

    def inner_ring(self, station: Cell) -> npt.NDArray[np.intp]:
        """The cells a station hands goods out to, as flat indices that are kept and so can't be
        written to."""
        railway = self._growth.railway
        radius = DELIVERY_RADIUS if railway is None else railway.reach(station)[0]
        return self._within(station, radius).cells

    def _within(self, station: Cell, radius: int) -> Within:
        """The cells within `radius` of a station or any of its platforms."""
        cells = tuple(self._growth.constructions.station_cells(station))
        return flat_within_any(self._growth.grid, cells, radius)

    def supplies(self, station: Cell) -> dict[Good, float]:
        """What the station has to send on: each good made within the cells it gathers from, and
        the passengers and mail the buildings it hands out to send, with how much of each came
        to it over the last quarter, which is only what the buildings around left over."""
        railway = self._growth.railway
        radius = COLLECTION_RADIUS if railway is None else railway.reach(station)[1]
        cells = self._within(station, radius).cells
        made: set[Good] = set()
        for kind, count in self._growth.counts.items():
            if count.ravel()[cells].sum() <= 0:
                continue
            made |= set(makes_of(kind, self._growth.year))
            if kind in RECIPES:
                made.add(RECIPES[kind].makes)
                made |= set(RECIPES[kind].also)
        for good, amount in self.travellers_in(self.inner_ring(station)).items():
            if amount > 0:
                made.add(good)
        got = {} if railway is None else railway.station_goods(station).last_quarter()
        made |= {good for good, amount in got.items() if amount > 0}
        return {good: got.get(good, 0.0) for good in made}

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
        """The passengers and mail these many buildings of each kind send, and want, in a month,
        more or fewer while history makes them travel more or less (railroads.sim.events)."""
        passengers = 0
        for kind, count in counts.items():
            if not count:
                continue
            points, value = self._growth.rule.points[kind]
            passengers += PASSENGERS_PER_POINT.get(points, 0) * value * count
        year = self._growth.year
        return {
            Good.PASSENGERS: float(passengers) * travel_factor(Good.PASSENGERS, year),
            Good.MAIL: float(passengers // MAIL_PER_PASSENGERS) * travel_factor(Good.MAIL, year),
        }

    def _needs(self, counts: Mapping[BuildingKind, int], station: Cell) -> dict[Need, float]:
        """What these many buildings of each kind around a station need in a month; of a boost
        only the player's own take (OWNED_BOOSTS), what theirs there take; of a processor's
        input, no more than the room left in its stockpile, so one stocked up on coal while it
        waits for iron ore wants none."""
        year = self._growth.year
        needs: dict[Need, float] = {}
        ring: npt.NDArray[np.intp] | None = None
        for kind in NEEDS:
            count = counts.get(kind)
            if not count:
                continue
            for need, amount in needs_of(kind, year, self._growth.economy.east).items():
                if (kind, need) in OWNED_BOOSTS:
                    taking = self._growth.owners.taking(kind, need)
                    if taking is None:
                        continue
                    ring = self.inner_ring(station) if ring is None else ring
                    many = float(taking[ring].sum())
                else:
                    many = float(count)
                if many <= 0:
                    continue
                wanted = many * amount
                if kind in RECIPES:
                    ring = self.inner_ring(station) if ring is None else ring
                    wanted = min(wanted, self._stock_room(kind, need, ring))
                if wanted > 0:
                    needs[need] = needs.get(need, 0.0) + wanted
        return needs

    def _stock_room(self, kind: BuildingKind, need: Need, cells: npt.NDArray[np.intp]) -> float:
        """How much more of an input the processors of a kind in these cells have room for."""
        many = self._growth.counts[kind].ravel()[cells].astype(np.float64)
        held = self._growth.economy.stock.get((kind, need))
        stocked = held[cells] if held is not None else 0.0
        return float(np.maximum(input_stock(kind, need) * many - stocked, 0.0).sum())

    def update_reach(self) -> None:
        """Works out again which cells the player's stations hand goods out to."""
        railway, cols = self._growth.railway, self._growth.grid.cols
        stations = [] if railway is None else railway.stations()
        self.in_reach = {
            divmod(index, cols)
            for station in stations
            for index in self.inner_ring(station).tolist()
        }


def as_needs(travellers: dict[Good, float]) -> dict[Need, float]:
    return {Need(good.value): amount for good, amount in travellers.items()}
