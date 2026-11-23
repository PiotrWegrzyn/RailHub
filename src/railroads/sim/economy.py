"""Trade: how the goods of railroads.sim.goods reach the buildings that need them each month, and
how satisfied that leaves each kind of building in each cell.

Each processor keeps a stockpile of STOCK_MONTHS of every input it uses at most for every
building, taking in what reaches it even while it waits for another input, and works from it.

Buildings hand out what they make on their own first, everywhere on the map (LOCAL_REACH): every
cell's spare goods go first to the buildings in the same cell, then to those one cell away, and so
on outwards up to LOCAL_REACH cells. Nothing is lost on the way, but a cell further away can take
less of a seller's month (reach_share): all of it from the same cell, then 13, 8, 5, 3, 2 and 1
twenty-firsts of it from one to six cells away (REACH_SHARES). Buildings the same distance from a
seller share what it has by how much they still want.

Stations come after (railroads.sim.railway), those joined by track to another station: each hands
what it holds, what trains brought and its stockpile, to buildings within its delivery radius
still short of a need, the same way,
nearest first and shared by how much they still want, but taking all they want from any distance.
Both go through one distribute. Once every need has been met, each station takes the spare goods no building
took that buildings made within its collection radius into its stockpile, while it has room, for
trains to carry away or to hand out the next month. The player's share of what the buildings they
own make goes to the stations in reach before anything else: before it is handed out around them.

Nothing is made that has nowhere to go: a processor uses up only the inputs for what left it; the
rest waits in its stockpile. Needs are handed out in the order of PIPELINE, so the inputs of a
processor come before what it makes: canneries take their fruit before houses eat what is left.

A building's satisfaction is the share of what it demands that it got and of what it makes that
reached a buyer; a processor's is only the share of all it could make that reached a buyer, so one
whose stockpile holds one input but not the other is unsatisfied however full it is; that is
raised by BONUS_PER_GOOD of itself for each of its bonus goods (railroads.sim.goods) it got all
of, and that share of it for part: a house fed and warmed that also got all its beer and half its
newspapers is at 145%, one only half fed with all its beer at 65%.
Food is a must for homes: one that got none is unsatisfied whatever else it got, so homes can't
live on firewood or beer alone. A building is unsatisfied at nothing, thriving from 100% and
content between. Its satisfaction adds to the
chances of more growing in its cell, its points times its satisfaction, with no ceiling above
100%. The economy remembers each cell's satisfaction over the last QUARTER months: what is handed
out on its own to buyers who were short is worth more (worth_to), which is what the player's own
buildings are paid by (railroads.sim.ownership).
"""

import time
from collections.abc import Mapping
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from railroads.sim import tiles
from railroads.sim.amounts import EPSILON, divided, ratio
from railroads.sim.clock import QUARTER, START_YEAR
from railroads.sim.constructions import BuildingKind
from railroads.sim.events import in_eastern_bloc, rise_of
from railroads.sim.goods import (
    BONUS_PART,
    BOOST_MOST,
    HANDED_OUT,
    HOMES,
    KIND_BOOST_MOST,
    LAND_BOOST_MOST,
    MAKES,
    NEEDS,
    OWNED_BOOSTS,
    PIPELINE,
    RECIPES,
    Good,
    Need,
)
from railroads.sim.grids import Cell, FloatArray, flat_within_any
from railroads.sim.move import Move
from railroads.sim.reach import REACH_SHARES, RING_PAD, Neighbourhoods, ring_starts, rings_of
from railroads.sim.steps import Steps, finish
from railroads.sim.tilemap import TileMap
from railroads.sim.trade import (
    BONUS_PER_GOOD,
    STOCKPILES,
    OwnedShare,
    Satisfaction,
    StationOrder,
    Stockpiles,
    Trade,
    input_stock,
    stores_together,
    worth_to,
)
from railroads.sim.year_caches import year_cached
from railroads.sim.yearly import bonus_of, bonus_part, boosts_of, makes_of, needs_of

# About how long a month's trade works before it pauses for the next frame.
STEP_SECONDS = 0.008


# Sellers and buyers in pairs, in order of how far apart they are, and where each distance starts.
type _Pairs = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], tuple[int, ...]]


def distribute(
    sellers: npt.NDArray[np.intp],
    buyers: npt.NDArray[np.intp],
    bounds: tuple[int, ...],
    supply: FloatArray,
    short: FloatArray,
    falloff: tuple[FloatArray, tuple[float, ...]] | None = None,
    used: FloatArray | None = None,
) -> tuple[npt.NDArray[np.intp], FloatArray]:
    """Hands goods from sellers to buyers, the nearest first: pairs of a seller and a buyer in
    order of how far apart they are, the pairs of each distance starting at its `bounds`
    (ring_starts). A seller shares what it has among the buyers at the same distance by how short
    they are. With a `falloff` of a month per seller and a share per distance, a buyer takes at most
    that share of the seller's month, less what `used` says already went between the pair this
    month. Takes what went out of `supply`, per seller, and out of `short`, per buyer; returns the
    pairs anything went between and how much."""
    went: list[npt.NDArray[np.intp]] = []
    amounts: list[FloatArray] = []
    for ring, (start, end) in enumerate(pairwise(bounds)):
        seller, buyer = sellers[start:end], buyers[start:end]
        wanting = short[buyer]
        live = (wanting > EPSILON) & (supply[seller] > EPSILON)
        if not live.any():
            continue
        seller, buyer, wanting = seller[live], buyer[live], wanting[live]
        before = None if used is None else used[start:end][live]
        share = wanting / np.bincount(seller, wanting, minlength=supply.size)[seller]
        sent = supply[seller] * share
        if falloff is not None:
            month, shares = falloff
            cap = month[seller] * shares[ring]
            if before is not None:
                cap = np.maximum(cap - before, 0.0)
            sent = np.minimum(sent, cap)
        arriving = np.bincount(buyer, sent, minlength=short.size)[buyer]
        sent *= np.minimum(1.0, short[buyer] / np.maximum(arriving, 1e-12))
        supply -= np.bincount(seller, sent, minlength=supply.size)
        short -= np.bincount(buyer, sent, minlength=short.size)
        went.append(np.flatnonzero(live) + start)
        amounts.append(sent)
        if supply.max() <= EPSILON or short.max() <= EPSILON:
            break
    if not went:
        return np.zeros(0, dtype=np.intp), np.zeros(0)
    return np.concatenate(went), np.concatenate(amounts)


class Economy:
    """Lets stations hand goods out on one map each month, on the cells with buildings that
    trade, and keeps every processor's stockpile of inputs from month to month."""

    # What buildings that are no processors took this month as goods reached a station by them,
    # by kind and need, then cell: the month's trade counts it as got and hands out only the rest.
    # None in saves from before stations handed goods out at once.
    taken: dict[tuple[BuildingKind, Need], dict[int, float]] | None = None

    def __init__(self, tile_map: TileMap) -> None:
        self.tile_map = tile_map
        self.cells = tile_map.grid.rows * tile_map.grid.cols
        self.neighbourhoods = Neighbourhoods(tile_map.grid)
        self.kinds = tuple(dict.fromkeys([*MAKES, *NEEDS]))
        # Each cell's satisfaction last month, which decides who is served first.
        self.levels = np.full(self.cells, Satisfaction.UNSATISFIED, dtype=np.int8)
        # The year, which decides what buildings need, and what each bonus good a building gets
        # adds to its satisfaction (railroads.sim.growth sets it).
        self.year = START_YEAR
        self.bonus_per_good = BONUS_PER_GOOD
        # Whether the map lies in the old Eastern Bloc, where some of history happens only
        # (railroads.sim.events).
        self.east = in_eastern_bloc(tile_map.bounds)
        # What the processors of each kind in each cell hold of each input.
        self.stock = Stockpiles(self.cells)
        # Each cell's satisfaction in each of the last QUARTER months, the latest last.
        self.months: list[FloatArray] = []

    def last_quarter(self) -> FloatArray:
        """Each cell's satisfaction over the last quarter, on average; full before any trade."""
        if not self.months:
            return np.ones(self.cells)
        return np.mean(self.months, axis=0)

    def trade(
        self,
        counts: Mapping[BuildingKind, npt.NDArray[np.int64]],
        stations: Mapping[Cell, StationOrder] | None = None,
        cells: npt.NDArray[np.intp] | None = None,
        owned: Mapping[BuildingKind, OwnedShare] | None = None,
    ) -> Trade:
        """Lets the stations hand out and gather one month of goods among the buildings counted
        in each cell, and the buildings sell what is left to each other. `cells`, the flat cells
        with any buildings, saves finding them again. `owned`, the player's part of the buildings they own, goes to
        stations first."""
        return finish(self.trade_steps(counts, stations, cells, owned))

    def trade_steps(
        self,
        counts: Mapping[BuildingKind, npt.NDArray[np.int64]],
        stations: Mapping[Cell, StationOrder] | None = None,
        cells: npt.NDArray[np.intp] | None = None,
        owned: Mapping[BuildingKind, OwnedShare] | None = None,
    ) -> Steps[Trade]:
        """trade, pausing about every STEP_SECONDS; nothing else may change the game between."""
        return MonthOfTrade(self, counts, stations or {}, cells, owned).run_steps()

    def _around(
        self,
        station: Cell,
        cells: npt.NDArray[np.intp],
        radius: float,
        platforms: tuple[Cell, ...] = (),
    ) -> tuple[npt.NDArray[np.intp], FloatArray]:
        """The traded cells within `radius` steps of a station or any of its platforms, as
        positions among `cells`, nearest first, and how many steps away each is."""
        within = flat_within_any(self.tile_map.grid, (station, *platforms), int(radius))
        if not cells.size:
            return cells, np.zeros(0)
        flat = within.cells
        positions = np.searchsorted(cells, flat)
        found = (positions < cells.size) & (cells[np.minimum(positions, cells.size - 1)] == flat)
        positions, steps = positions[found], within.steps[found]
        order = np.argsort(steps, kind="stable")
        return positions[order], steps[order]

    @staticmethod
    def _gather(
        room: float,
        far: tuple[npt.NDArray[np.intp], FloatArray],
        supply: FloatArray,
        made: FloatArray,
        sold: FloatArray,
    ) -> float:
        """Takes spare goods that buildings made from the cells around, evenly, up to the room
        left; returns how much was taken."""
        positions, _ = far
        spare = np.maximum(supply[positions] - sold[positions], 0.0)
        spare = np.minimum(spare, made[positions])
        total = float(spare.sum())
        if total <= 0 or room <= 0:
            return 0.0
        take = min(room, total)
        sold[positions] += spare * (take / total)
        return take

    def moved_into(self, new: Economy, move: Move) -> Economy:
        """`new`, an economy of the moved map, given what this one remembers, moved."""
        new.levels = move.flat_cells(self.levels, self.levels.dtype.type(new.levels[0]))
        new.year = self.year
        new.bonus_per_good = self.bonus_per_good
        stock = Stockpiles(new.cells)
        for key, held in self.stock.items():
            stock[key] = move.flat_cells(held)
        new.stock = stock
        new.months = [move.flat_cells(month, 1.0) for month in self.months]
        if self.taken is not None:
            new.taken = {
                key: {move.flat(cell): amount for cell, amount in taken.items()}
                for key, taken in self.taken.items()
            }
        return new


@year_cached
def wants_of(kind: BuildingKind, year: int) -> frozenset[Need]:
    """The needs a kind wants in a given year: what a processor takes in or is boosted by, and
    what the rest need."""
    recipe = RECIPES.get(kind)
    if recipe is None:
        return frozenset(needs_of(kind, year))
    return frozenset(recipe.takes(year)) | frozenset(boosts_of(kind, year))


class MonthOfTrade:
    """One month of trade on the cells with buildings, worked out need by need in the order of
    the chain: what each building is offered, takes in, makes and gives away."""

    def __init__(
        self,
        economy: Economy,
        counts: Mapping[BuildingKind, npt.NDArray[np.int64]],
        stations: Mapping[Cell, StationOrder],
        cells: npt.NDArray[np.intp] | None = None,
        owned: Mapping[BuildingKind, OwnedShare] | None = None,
    ) -> None:
        self.economy = economy
        self.stations = stations
        self.owned = owned or {}
        self.first_gathered: set[Good] = set()
        flat = {k: counts[k].ravel() for k in economy.kinds if k in counts}
        if cells is None:
            built = np.zeros(economy.cells, dtype=bool)
            for n in flat.values():
                built |= n > 0
            cells = np.flatnonzero(built)
        self.cells = cells
        in_cells = ((k, n[cells]) for k, n in flat.items())
        self.count = {k: n.astype(np.float64) for k, n in in_cells if n.any()}
        self.wanted = {kind: wants_of(kind, economy.year) for kind in self.count}
        self.zero = np.zeros(cells.size)
        self.station_pairs: _Pairs | None = None
        # What a unit handed out on its own is worth to each cell, times its price, and per good
        # how much more than its price what each cell handed out was worth in all.
        self.worth_to = worth_to(economy.last_quarter()[cells])
        self.worth_above: dict[Good, FloatArray] = {}
        # What each kind of processor holds of each input, worked out as the trade begins.
        self.stock: dict[tuple[BuildingKind, Need], FloatArray] = {}
        # What buildings offer of each good, by kind and in all, and how much of it went; and how
        # much of each kind's offer the player's own buildings made.
        self.offers: dict[tuple[BuildingKind, Good], FloatArray] = {}
        self.owned_made: dict[tuple[BuildingKind, Good], FloatArray] = {}
        self.offered = {good: self.zero.copy() for good in Good}
        self.sold = {good: self.zero.copy() for good in Good}
        # Per need, what buildings got of it from sellers nearby rather than stations.
        self.local: dict[Need, FloatArray] = {}
        # Where each traded cell stands among them, -1 for the rest, and per good the pairs of
        # sellers and buyers near enough to trade.
        self.position = np.full(economy.cells, -1, dtype=np.intp)
        self.position[cells] = np.arange(cells.size)
        self.pairs: dict[Good, tuple[*_Pairs, tuple[float, ...]]] = {}
        self.padded: tuple[npt.NDArray[np.intp], tuple[npt.NDArray[np.intp], ...]] | None = None
        # What buildings had taken as goods reached their station when their need was served:
        # what they take after that, while the month's trade is still going, counts for the next.
        self.counted: dict[tuple[BuildingKind, Need], dict[int, float]] = {}
        for kind in MAKES:
            if kind in RECIPES or kind not in self.count:
                continue
            for good, amount in makes_of(kind, economy.year).items():
                self.offers[(kind, good)] = self.count[kind] * amount
                self.offered[good] += self.count[kind] * amount
                if kind in self.owned:
                    self.owned_made[(kind, good)] = (
                        self._owned_part(kind) * self.count[kind] * amount
                    )
        # Per kind of processor, what it makes from its inputs alone, and the share it has of each
        # of its boosts.
        self.uses: dict[BuildingKind, dict[Need, FloatArray]] = {}
        self.received: dict[tuple[BuildingKind, Need], FloatArray] = {}
        self.brought = {
            station: stores_together(order.brought, order.stockpile)
            for station, order in stations.items()
        }
        self.handed: dict[Cell, dict[Good, float]] = {station: {} for station in stations}
        self.handed_for: dict[Cell, dict[tuple[Need, Good], float]] = {
            station: {} for station in stations
        }
        self.handed_worth: dict[Cell, dict[tuple[Need, Good], float]] = {
            station: {} for station in stations
        }
        self.gathered: dict[Cell, dict[Good, float]] = {station: {} for station in stations}
        self.owned_brought: dict[Cell, dict[Good, float]] = {station: {} for station in stations}
        self.owned_sent: dict[tuple[BuildingKind, Good], FloatArray] = {}
        self.room = {
            station: {
                good: max(
                    0.0, order.limit - order.brought.get(good, 0.0) - order.stockpile.get(good, 0.0)
                )
                for good in Good
            }
            for station, order in stations.items()
        }
        self.reach = {
            station: (
                economy._around(station, cells, order.delivery, order.platforms),
                economy._around(station, cells, order.collection, order.platforms),
            )
            for station, order in stations.items()
        }

    def run_steps(self) -> Steps[Trade]:
        yield
        economy, cells = self.economy, self.cells
        self.stock = {
            (kind, need): np.minimum(
                economy.stock[(kind, need)][cells],
                input_stock(kind, need, economy.year) * self.count.get(kind, self.zero),
            )
            if (kind, need) in economy.stock
            else np.zeros(cells.size)
            for kind, need in STOCKPILES
        }
        yield
        paused = time.perf_counter()
        for step in PIPELINE:
            if isinstance(step, BuildingKind):
                self._produce(step)
                continue
            self._serve(step)
            if time.perf_counter() - paused > STEP_SECONDS:
                yield
                paused = time.perf_counter()
        for good in Good:
            self._gather_owned(good)
            self._gather(good)
        return (yield from self._settle())

    def _wants(self, need: Need) -> dict[BuildingKind, FloatArray]:
        """What each kind wants of a need: consumers what they use in a month, processors the
        room left in their stockpile."""
        wants = {}
        for kind, n in self.count.items():
            if need not in self.wanted[kind]:
                continue
            if kind in RECIPES:
                wants[kind] = np.maximum(
                    input_stock(kind, need, self.economy.year) * n - self.stock[(kind, need)], 0.0
                )
            else:
                wants[kind] = self._count_for(kind, need) * self._amount(kind, need)
        return wants

    def _amount(self, kind: BuildingKind, need: Need) -> float:
        """What one building of a kind wants of a need in a month this year."""
        return needs_of(kind, self.economy.year, self.economy.east).get(need, 0.0)

    def _owned_of_taking(self, kind: BuildingKind) -> FloatArray:
        """The player's part of the buildings of a kind in each cell taking the boosts only their
        own take: all of them, unless the player controls more than they own."""
        share = self.owned[kind]
        if share.taking is None:
            return np.ones(self.cells.size)
        taking = share.taking[self.cells]
        return np.minimum(divided(share.owned[self.cells], taking), 1.0)

    def _owned_part(self, kind: BuildingKind) -> FloatArray:
        """The share of the buildings of a kind in each cell that are the player's, as they
        trade."""
        share = self.owned.get(kind)
        if share is None:
            return self.zero
        owned = share.owned[self.cells]
        count = self.count[kind]
        return np.minimum(divided(owned, count), 1.0)

    def _count_for(self, kind: BuildingKind, need: Need) -> FloatArray:
        """How many buildings of a kind in each cell want a need: all of them, but for a boost
        only the player's own take (OWNED_BOOSTS), only theirs or those they control, once their
        tile opens it."""
        if (kind, need) not in OWNED_BOOSTS:
            return self.count[kind]
        share = self.owned.get(kind)
        if share is None or need not in share.takes:
            return self.zero
        return (share.owned if share.taking is None else share.taking)[self.cells]

    def _taken(self, kind: BuildingKind, need: Need) -> FloatArray:
        """What buildings of a kind took of a need this month as goods reached their station."""
        taken = self.zero.copy()
        for cell, amount in ((self.economy.taken or {}).get((kind, need)) or {}).items():
            position = self.position[cell]
            if position >= 0:
                taken[position] += amount
        return taken

    def _serve(self, need: Need) -> None:
        """Lets the goods meeting a need reach the buildings wanting it, the most valuable first:
        first what buildings hand out on their own, then what trains brought to stations, within
        their delivery radius. Buildings want only what they did not take already this month."""
        wants = self._wants(need)
        if not wants:
            return
        early = {kind: self._taken(kind, need) for kind in wants if kind not in RECIPES}
        for kind in early:
            self.counted[(kind, need)] = dict((self.economy.taken or {}).get((kind, need)) or {})
        for kind, taken in early.items():
            wants[kind] = np.maximum(wants[kind] - taken, 0.0)
        demand = sum(wants.values(), start=self.zero.copy())
        must = self._must(need, wants)
        received = self.zero.copy()
        local = self.zero.copy()
        used: dict[Good, FloatArray] = {}
        for target in (demand,) if must is None else (must, demand):
            before = received.copy()
            for good in HANDED_OUT[need]:
                used[good] = self._sell_nearby(good, target, received, used.get(good))
            local += received - before
            mine = self._owned_short(wants, target, received)
            if mine is not None:
                for good in HANDED_OUT[need]:
                    self._hand_out(need, good, np.minimum(received + mine, target), received)
            for good in HANDED_OUT[need]:
                self._hand_out(need, good, target, received)
        self.local[need] = local
        met = ratio(received, demand)
        for kind, want in wants.items():
            got = met * want
            key = (kind, need)
            got = got + early.get(kind, 0.0)
            self.received[key] = self.received.get(key, self.zero) + got
            if kind in RECIPES:
                self.stock[key] = self.stock[key] + got
            else:
                self._boost_from_land(kind, need, got)

    def _must(self, need: Need, wants: dict[BuildingKind, FloatArray]) -> FloatArray | None:
        """What the buildings wanting a need must get of it in each cell, without the part some
        take only as a bonus (BONUS_PART), which is handed out after everyone has had the rest;
        None where no one wants any of it as a bonus."""
        if not any((kind, need) in BONUS_PART for kind in wants):
            return None
        must = self.zero.copy()
        for kind, want in wants.items():
            extra = bonus_part(kind, need, self.economy.year, self.economy.east)
            spare = 0.0 if extra is None else self._count_for(kind, need) * extra
            must += np.maximum(want - spare, 0.0)
        return must

    def _owned_short(
        self, wants: dict[BuildingKind, FloatArray], demand: FloatArray, received: FloatArray
    ) -> FloatArray | None:
        """What the player's own buildings in each cell are still short of a need after the
        goods handed out on their own, which the stations hand them before anyone else; None
        if they own none that want it."""
        owned = [kind for kind in wants if kind in self.owned]
        if not owned:
            return None
        want = sum((wants[kind] * self._owned_part(kind) for kind in owned), start=self.zero)
        met = ratio(received, demand)
        return want * (1.0 - met)

    def _hand_out(self, need: Need, good: Good, demand: FloatArray, received: FloatArray) -> None:
        """Lets every station hand out what trains brought of a good to the buildings within its
        delivery radius still short of it, the nearest first."""
        stations = list(self.stations)
        brought = np.array([self.brought[s].get(good, 0.0) for s in stations], dtype=np.float64)
        if not (brought > EPSILON).any():
            return
        sellers, buyers, bounds = self._station_pairs()
        left = brought.copy()
        short = demand - received
        went, sent = distribute(sellers, buyers, bounds, left, short)
        received[:] = demand - short
        worth = np.bincount(
            sellers[went], sent * self.worth_to[buyers[went]], minlength=len(stations)
        )
        for station, before, after, weighed in zip(stations, brought, left, worth, strict=True):
            given = float(before - after)
            if given <= 0:
                continue
            worths = self.handed_worth[station]
            worths[(need, good)] = worths.get((need, good), 0.0) + float(weighed)
            self.brought[station][good] = float(after)
            self.handed[station][good] = self.handed[station].get(good, 0.0) + given
            handed_for = self.handed_for[station]
            handed_for[(need, good)] = handed_for.get((need, good), 0.0) + given

    def _station_pairs(self) -> _Pairs:
        """Every station, by its place among the stations, with every traded cell within its
        delivery radius, as positions among the traded cells, the nearest pairs first, and where
        the pairs of each distance start (ring_starts)."""
        if self.station_pairs is None:
            near = [self.reach[station][0] for station in self.stations]
            sellers = np.repeat(np.arange(len(near)), [cells.size for cells, _ in near])
            buyers = np.concatenate([cells for cells, _ in near] or [np.zeros(0, np.intp)])
            away = np.concatenate([steps for _, steps in near] or [np.zeros(0)])
            order = np.argsort(away, kind="stable")
            self.station_pairs = (sellers[order], buyers[order], ring_starts(away[order]))
        return self.station_pairs

    def _taken_since_served(self) -> dict[tuple[BuildingKind, Need], dict[int, float]]:
        """What buildings took as goods reached their station after their need was served this
        month, which the next month counts as taken."""
        later: dict[tuple[BuildingKind, Need], dict[int, float]] = {}
        for key, taken in (self.economy.taken or {}).items():
            counted = self.counted.get(key, {})
            left = {
                cell: amount - counted.get(cell, 0.0)
                for cell, amount in taken.items()
                if amount - counted.get(cell, 0.0) > EPSILON
            }
            if left:
                later[key] = left
        return later

    def _boost_from_land(self, kind: BuildingKind, need: Need, got: FloatArray) -> None:
        """Lets what a kind making goods from the land got of one of its boosts make it make more
        of each of its goods, before any of them is sold: up to its own KIND_BOOST_MOST or
        LAND_BOOST_MOST more, or its part of BOOST_MOST shared with the other boosts; a boost only
        the player's own take makes only those make up to its OWNED_BOOSTS more, for the
        player."""
        boosts = boosts_of(kind, self.economy.year)
        if need not in boosts:
            return
        taking = self._count_for(kind, need)
        wanted = taking * boosts[need]
        share = np.minimum(ratio(got, wanted), 1.0)
        shared = [
            boost
            for boost in boosts
            if boost not in LAND_BOOST_MOST
            and (kind, boost) not in KIND_BOOST_MOST
            and (kind, boost) not in OWNED_BOOSTS
        ]
        most = OWNED_BOOSTS.get(
            (kind, need),
            KIND_BOOST_MOST.get(
                (kind, need), LAND_BOOST_MOST.get(need, BOOST_MOST / max(len(shared), 1))
            ),
        )
        owned_only = (kind, need) in OWNED_BOOSTS
        for good, amount in makes_of(kind, self.economy.year).items():
            more = taking * amount * most * share
            self.offers[(kind, good)] = self.offers[(kind, good)] + more
            self.offered[good] = self.offered[good] + more
            if kind in self.owned:
                mine = more * (
                    self._owned_of_taking(kind) if owned_only else self._owned_part(kind)
                )
                self.owned_made[(kind, good)] = self.owned_made.get((kind, good), self.zero) + mine

    def _sell_nearby(
        self,
        good: Good,
        demand: FloatArray,
        received: FloatArray,
        used: FloatArray | None = None,
    ) -> FloatArray | None:
        """Lets the cells with spare goods hand them out to the buildings still short of them, the
        nearest first, each cell taking from a seller at most its reach_share of the seller's
        month, less what `used` says it took already this month; returns that, per pair, with
        what went now."""
        self._gather_owned(good)
        spare = np.maximum(self.offered[good] - self.sold[good], 0.0)
        left = demand - received
        if not (spare > EPSILON).any() or not (left > EPSILON).any():
            return used
        sellers, buyers, bounds, falloff = self._pairs(good)
        remaining = spare.copy()
        went, sent = distribute(
            sellers, buyers, bounds, remaining, left, (self.offered[good], falloff), used
        )
        used = np.zeros(sellers.size) if used is None else used
        used[went] += sent
        above = np.bincount(
            sellers[went], sent * (self.worth_to[buyers[went]] - 1.0), minlength=self.cells.size
        )
        self.worth_above[good] = self.worth_above.get(good, self.zero) + above
        self.sold[good] += spare - remaining
        received[:] = demand - left
        return used

    def _gather_owned(self, good: Good) -> None:
        """Brings the player's share of what the buildings they own made of a good into the
        stations around, while they have room, before any of it is handed out around them: as
        if a train had brought it, so the stations hand it out and it pays like a delivery."""
        if good in self.first_gathered:
            return
        self.first_gathered.add(good)
        left = self.zero.copy()
        parts: dict[BuildingKind, FloatArray] = {}
        for (kind, made), mine in self.owned_made.items():
            if made == good:
                part = mine * self.owned[kind].share
                parts[kind] = parts.get(kind, self.zero) + part
                left += part
        if not left.any():
            return
        offered = left.copy()
        for station in self.stations:
            room = self.room[station][good]
            if room <= 0:
                continue
            _, far = self.reach[station]
            took = self.zero.copy()
            taken = self.economy._gather(room, far, left, left, took)
            if taken > 0:
                left -= took
                self.sold[good] += took
                self.brought[station][good] = self.brought[station].get(good, 0.0) + taken
                owned = self.owned_brought[station]
                owned[good] = owned.get(good, 0.0) + taken
                self.room[station][good] -= taken
        sent = divided(offered - left, offered)
        for kind, part in parts.items():
            self.owned_sent[(kind, good)] = part * sent

    def _pairs(self, good: Good) -> tuple[*_Pairs, tuple[float, ...]]:
        """Every cell making a good with every traded cell up to LOCAL_REACH cells from it, as
        positions among the traded cells, the nearest pairs first, where the pairs of each
        distance start (ring_starts) and the reach_share of the seller's month each buyer can take."""
        known = self.pairs.get(good)
        if known is not None:
            return known
        grid = self.economy.tile_map.grid
        rings = rings_of(grid)
        sellers = np.flatnonzero(self.offered[good] > 0)
        padded, offsets = self._padded()
        rows, cols = np.divmod(self.cells[sellers], grid.cols)
        at = (rows + RING_PAD) * (grid.cols + 2 * RING_PAD) + cols + RING_PAD
        reached = at + np.where(rows % 2 == 1, offsets[1][:, None], offsets[0][:, None])
        buyers = padded[reached]
        step, seller = np.nonzero(buyers >= 0)
        away = rings.away[step]
        bounds = ring_starts(away)
        shares = tuple(REACH_SHARES[int(away[start])] for start in bounds[:-1])
        found = (sellers[seller], buyers[step, seller], bounds, shares)
        self.pairs[good] = found
        return found

    def _padded(self) -> tuple[npt.NDArray[np.intp], tuple[npt.NDArray[np.intp], ...]]:
        """Where each traded cell stands among them on the map with RING_PAD cells of -1 around it,
        as flat indices, and the flat steps on it to every cell up to LOCAL_REACH cells away from
        a cell on an even and an odd row, nearest first."""
        if self.padded is None:
            grid = self.economy.tile_map.grid
            rings = rings_of(grid)
            width = grid.cols + 2 * RING_PAD
            padded = np.full((grid.rows + 2 * RING_PAD, width), -1, dtype=np.intp)
            padded[RING_PAD:-RING_PAD, RING_PAD:-RING_PAD] = self.position.reshape(grid.shape)
            offsets = tuple(rings.rows[odd] * width + rings.cols[odd] for odd in (0, 1))
            self.padded = (padded.ravel(), offsets)
        return self.padded

    def _gather(self, good: Good) -> None:
        """Lets every station take the spare units of a good no building took into its
        stockpile, for trains. Where nothing is left or a station has no room, it takes
        nothing."""
        offered = self.offered[good]
        if not offered.any():
            return
        for station in self.stations:
            room = self.room[station][good]
            if room <= 0:
                continue
            _, far = self.reach[station]
            taken = self.economy._gather(room, far, offered, offered, self.sold[good])
            if taken > 0:
                self.gathered[station][good] = self.gathered[station].get(good, 0.0) + taken
                self.room[station][good] -= taken

    def _produce(self, kind: BuildingKind) -> None:
        """Offers what a kind of processor could make from its stockpile: what its inputs allow,
        the inputs standing in for others first as they make the most, within the room its
        boosts that let it take in more give it (takes_more), and up to the recipe's boost_most
        more by the share it has of its other boosts; and what that leaves over (also). Keeps
        how much of each input all of that takes. Its supply tile lets it make more from the same
        inputs, and the demand tiles of its inputs make it take more of them for as much
        (railroads.sim.tiles)."""
        if kind not in self.count:
            return
        recipe = RECIPES[kind]
        year = self.economy.year
        made = tiles.of_year(year).make_factor(recipe.makes)
        inputs = {
            need: per_unit * rise_of(kind, need, year) / made
            for need, per_unit in recipe.inputs(year).items()
        }
        per_boost = {
            need: amount * rise_of(kind, need, year) / made
            for need, amount in recipe.boosts.items()
        }
        boosts = boosts_of(kind, year)
        base = self.count[kind] * recipe.most * made
        roomy = {
            need: np.minimum(
                self.stock[(kind, need)] / np.where(base > 0, base * per_boost[need], 1.0),
                1.0,
            )
            * (base > 0)
            for need in boosts
            if need in recipe.takes_more
        }
        room = base * (
            1.0 + sum((recipe.takes_more[need] * share for need, share in roomy.items()), 0.0)
        )
        extra = room - base
        left = {need: self.stock[(kind, need)].copy() for need in inputs}
        uses = {need: self.zero.copy() for need in recipe.takes(year)}
        can = self.zero.copy()
        for swapped, (swap, times) in recipe.swaps_in(year).items():
            made = room.copy()
            for need, per_unit in inputs.items():
                held = self.stock[(kind, swap)] if need == swapped else left[need]
                made = np.minimum(made, held / (per_unit / times))
            for need, per_unit in inputs.items():
                taken = made * per_unit / times
                uses[swap if need == swapped else need] += taken
                if need != swapped:
                    left[need] -= taken
            can += made
            room = room - made
        plain = room
        for need, per_unit in inputs.items():
            plain = np.minimum(plain, left[need] / per_unit)
        for need, per_unit in inputs.items():
            uses[need] += plain * per_unit
        can += plain
        if roomy:
            used = np.clip(divided(can - base, extra), 0.0, 1.0)
            for need, share in roomy.items():
                uses[need] = base * share * per_boost[need] * used
        boosts = {need: amount for need, amount in boosts.items() if need not in roomy}
        if boosts:
            shares = {
                need: np.minimum(
                    self.stock[(kind, need)] / np.where(can > 0, can * per_boost[need], 1.0),
                    1.0,
                )
                * (can > 0)
                for need in boosts
            }
            for need, share in shares.items():
                uses[need] = can * share * per_boost[need]
            can = can * (1.0 + recipe.boost_most * sum(shares.values()) / len(shares))
        self.uses[kind] = uses
        for good, made in ((recipe.makes, can), *((g, can * a) for g, a in recipe.also.items())):
            self.offers[(kind, good)] = made
            self.offered[good] += made
            if kind in self.owned:
                self.owned_made[(kind, good)] = made * self._owned_part(kind)

    def _went(self, good: Good) -> FloatArray:
        """The share of all of a good on offer that went, to buildings or stations, in each
        cell."""
        return np.minimum(ratio(self.sold[good], self.offered[good]), 1.0)

    def _settle(self) -> Steps[Trade]:
        """Counts what every building got and gave, uses up the inputs of what the processors
        made, and keeps their stockpiles for next month, pausing once between."""
        got = {k: self.zero.copy() for k in self.count}
        needed = {k: self.zero.copy() for k in self.count}
        gave = {k: self.zero.copy() for k in self.count}
        could_make = {k: self.zero.copy() for k in self.count}
        bonus: dict[BuildingKind, FloatArray] = {}
        goods_got = {}
        for (kind, need), received in self.received.items():
            wanted = self._count_for(kind, need) * self._amount(kind, need)
            goods_got[(kind, need)] = (received, wanted)
            if need in bonus_of(kind, self.economy.year):
                part = divided(np.minimum(received, wanted), wanted)
                bonus[kind] = bonus.get(kind, self.zero) + part
                continue
            if kind in RECIPES or need in boosts_of(kind, self.economy.year):
                continue
            extra = bonus_part(kind, need, self.economy.year, self.economy.east)
            if extra is not None:
                spare = self._count_for(kind, need) * extra
                wanted = wanted - spare
                over = np.clip(received - wanted, 0.0, spare)
                part = divided(over, spare)
                bonus[kind] = bonus.get(kind, self.zero) + part
            got[kind] += np.minimum(received, wanted)
            needed[kind] += wanted
        for kind in HOMES:
            if kind not in self.count or Need.FOOD not in NEEDS.get(kind, {}):
                continue
            food = self.received.get((kind, Need.FOOD), self.zero)
            hungry = (self.count[kind] > 0) & (food <= 0)
            got[kind] = np.where(hungry, 0.0, got[kind])
            if kind in bonus:
                bonus[kind] = np.where(hungry, 0.0, bonus[kind])
        goods_gave = {}
        shares: dict[Good, FloatArray] = {}
        for (kind, good), offer in self.offers.items():
            if good not in shares:
                shares[good] = self._went(good)
            sent = given = offer * shares[good]
            most = self.count[kind] * makes_of(kind, self.economy.year).get(good, 0.0)
            goods_gave[(kind, good)] = (given, most)
            gave[kind] += given
            could_make[kind] += most
            recipe = RECIPES.get(kind)
            if recipe is not None and good == recipe.makes:
                went = ratio(sent, offer)
                for need, amount in self.uses.get(kind, {}).items():
                    key = (kind, need)
                    self.stock[key] = np.maximum(self.stock[key] - went * amount, 0.0)
        yield
        stockpiles = self.economy.stock
        for key, held in self.stock.items():
            if not held.any():
                stockpiles.pop(key, None)
                continue
            stock = stockpiles[key]
            stock[:] = 0.0
            stock[self.cells] = held
        self.economy.taken = self._taken_since_served()
        trade = Trade(
            self.cells,
            got,
            needed,
            gave,
            could_make,
            self.handed,
            self.gathered,
            goods_got,
            goods_gave,
            self.handed_for,
            bonus,
            self.economy.bonus_per_good,
            self.local,
            {
                good: 1.0 + divided(above, self.sold[good])
                for good, above in self.worth_above.items()
            },
            self.owned_brought,
            self.handed_worth,
            self.owned_sent,
        )
        self.economy.levels[:] = Satisfaction.UNSATISFIED
        self.economy.levels[self.cells] = trade.levels()
        month = np.ones(self.economy.cells)
        month[self.cells] = trade.share(tuple(needed))
        self.economy.months = [*self.economy.months[1 - QUARTER :], month]
        return trade
