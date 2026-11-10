"""The map grows by itself, a month at a time, like a cellular automaton, by the ladders of
railroads.sim.ladders: towns, farms, orchards, mines, lumber mills, ports and every kind of works.

A building's points count as served up to MOST_SERVED times over, by the share of what it needs
and makes that stations handled, each at SERVED_POINT_WORTH of a doubling, so only what stations
supply compounds. A building that finds no room in its cell spills over into the nearest cell
within SPILL_STEPS with the room and land for it that the ladder's caps allow, chosen among cells
equally near at random by how well their land suits it.

On town turns, a neighbourhood with SKYSCRAPER_PEOPLE population and SKYSCRAPER_FLATS blocks of
flats may put up a skyscraper, SKYSCRAPER_CHANCE of the time: a business centre that pulls down the
flats and gives corporate points, whose people move into SKYSCRAPER_HOMES houses in the nearest
cells with room for them.

Every TRADE_EVERY_MONTHS the stations hand out their goods (railroads.sim.economy), which leaves
each kind of building in each cell unsatisfied, content or thriving.

Forests come and go as well (railroads.sim.forests): people, industry and above all lumber mills
clear them, they spread onto open grassland beside them, and the land weighs again wherever they
changed.

A new map runs this for MONTHS_BEFORE_PLAY before play starts; in play it carries on, one month
every few minutes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import START_YEAR
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.demand import capacity_of
from railroads.sim.economy import (
    DELIVERY_RADIUS,
    Economy,
    FloatCells,
    StationOrder,
    Trade,
    levels_of,
    served_multipliers,
)
from railroads.sim.forests import ForestRule, forest_turn, forests_nearby
from railroads.sim.goods import FITS, NEEDS, TRAVELLERS, Good, Need, makes_of, needs_of, price_of
from railroads.sim.grids import Cell
from railroads.sim.ladders import (
    FARMS_CROWDED_OUT,
    GROWTH,
    PASSENGERS_PER_POINT,
    POINTS,
    TOWN_LAND,
    TOWNS,
    Cells,
    FloatGrid,
    GrowthRule,
    IntGrid,
    Ladder,
    Land,
    Points,
)
from railroads.sim.railway import CAR_CAPACITY, MAX_CARS, Railway
from railroads.sim.stats import History
from railroads.sim.tech import GOOD_TREES, SHOPS, Effect, Tree, tree_of
from railroads.sim.tilemap import Relief, Terrain, Vegetation

if TYPE_CHECKING:
    from railroads.sim.novelties import Novelties

# A station sends a piece of mail for every MAIL_PER_PASSENGERS passengers, rounded down.
MAIL_PER_PASSENGERS = 5
MONTHS_BEFORE_PLAY = 1000 * 12
SKYSCRAPER_FLATS = 3
SKYSCRAPER_HOMES = 10
# How many cells away the people of the flats a skyscraper pulls down may find their new homes.
HOMES_STEPS = 6
# Population points that clear a forest as fast as one point of industry.
POPULATION_PER_CLEARING_POINT = 4
# Above this chance, a ladder draws for every cell instead of thinning.
THINNING_UP_TO = 0.05
# What one served point adds to a ladder's chance, as a share of what a doubling of the points
# nearby adds.
SERVED_POINT_WORTH = 2.0
# The player's share of what a building they own sells to their stations, what it costs to buy
# (this many years of that share at full output), and how far away a building pulled down by growth
# goes up again.
OWNER_SHARE = 0.5
PAYBACK_YEARS = 5
# However much tech cuts the price of buildings, they cost at least this share of it.
MOST_DISCOUNT = 0.5
# A building rising in a station's reach gives the experience of this many units of its goods for
# each of its stars: a full train of the first engine.
TRAIN_LOAD = MAX_CARS * CAR_CAPACITY
SKYSCRAPER_STARS = 4
RELOCATE_STEPS = 12
# How many cells away a building that finds no room in its own may go up instead.
SPILL_STEPS = 3


def _allowed(cells: Cells, opened: Cells, land: FloatGrid) -> Cells:
    """The cells a ladder may grow in once these have opened to it, in order. Cells that closed
    stay in until they are a tenth of the rest, harmless with no chance to grow there."""
    if opened.size:
        at = np.searchsorted(cells, opened)
        listed = (at < cells.size) & (cells[np.minimum(at, cells.size - 1)] == opened)
        cells = np.insert(cells, at[~listed], opened[~listed])
    if cells.size and np.count_nonzero(land[cells] <= 0) > cells.size // 10:
        cells = np.flatnonzero(land > 0)
    return cells


@dataclass(slots=True)
class _Place:
    """Where a ladder can grow: its cells as flat indices, every cell's weight for its land,
    and every cell's chance before the ladder's needs, kept up to date as buildings and forests
    come and go."""

    ladder: Ladder
    cells: Cells
    land: FloatGrid
    chances: FloatGrid
    # No cell's chance is above this; a month draws its cells against it.
    highest: float = 0.0


def leaving_order(
    buildings: list[Building], points: Mapping[BuildingKind, tuple[Points, int]] = POINTS
) -> list[Building]:
    """The order buildings leave a cell to make room: the smallest first."""
    return sorted(
        buildings, key=lambda b: (points.get(b.kind, (Points.POPULATION, 0))[1], b.kind.value)
    )


class Growth:
    """The grown buildings on one map, counted per cell, and the rule that grows them."""

    def __init__(
        self,
        constructions: Constructions,
        rule: GrowthRule = GROWTH,
        start_year: int = START_YEAR,
        history: History | None = None,
    ) -> None:
        self.constructions = constructions
        self.rule = rule
        self.start_year = start_year
        # What each year ended with (railroads.sim.stats).
        self.history = History() if history is None else history
        tile_map = constructions.tile_map
        grid = tile_map.grid
        self._grid = grid
        self._counts = {kind: np.zeros(grid.shape, dtype=np.int64) for kind in rule.points}
        for built in constructions:
            if isinstance(built, Building) and built.kind in self._counts:
                self._counts[built.kind][built.cell] += 1
        # How many buildings of any kind stand in each cell.
        self._occupied = sum(self._counts.values(), start=np.zeros(grid.shape, dtype=np.int64))
        # The cells last month's trade was worked out on.
        self._traded = np.zeros(0, dtype=np.intp)
        rows, cols = grid.neighbour_index_grids()
        self._nearby = {
            points: self._neighbourhood_sum(self.points(points), rows, cols) for points in Points
        }
        self.economy = Economy(tile_map)
        self._near = self.economy.neighbourhoods.index
        self._by_points: dict[Points, list[tuple[_Place, float]]] = {p: [] for p in Points}
        # Each cell's points of every kind counted as served, multiplied by how satisfied.
        self._served = {points: np.zeros(grid.rows * grid.cols) for points in Points}
        self._places = []
        everywhere = np.arange(grid.rows * grid.cols)
        # Ladders growing on the same land share its weights, and the cells it allows.
        lands: dict[int, tuple[Land, FloatGrid, Cells]] = {}
        for ladder in rule.ladders:
            shared = lands.get(id(ladder.land))
            if shared is None or shared[0] is not ladder.land:
                weights = ladder.land.weights(tile_map, everywhere, self._near)
                shared = lands[id(ladder.land)] = (
                    ladder.land,
                    weights,
                    np.flatnonzero(weights > 0),
                )
            _, land, cells = shared
            chances = np.full(land.size, ladder.base_chance)
            for points, per_doubling in ladder.chance_per_doubling.items():
                chances += per_doubling * ladder.pull(self._nearby[points].ravel())
            self._places.append(
                _Place(ladder, cells, land, chances * land, float((chances * land).max()))
            )
        for place in self._places:
            for points, per_doubling in place.ladder.chance_per_doubling.items():
                self._by_points[points].append((place, per_doubling))
        # The cells whose forest came or went in the last month.
        self.forests_changed: set[Cell] = set()
        self.trade: Trade | None = None
        # The share of its needs each cell's buildings get in a simulated game, standing in for
        # trains (railroads.sim.autoplay); NaN where trains decide.
        self.assumed_shares: FloatCells | None = None
        # Trains and their stations, once the player has any (railroads.sim.railway).
        self.railway: Railway | None = None
        # The one-off buildings this game may get (railroads.sim.novelties), rolled once a year.
        self.novelties: Novelties | None = None
        # How many of each kind of building in each cell the player owns, and what each kind they
        # own earned them last month.
        self._owned: dict[BuildingKind, IntGrid] = {}
        self.owner_income: dict[BuildingKind, float] = {}
        # The cells the player's stations hand goods out to, and the experience growth there
        # has given since the game began: where, what it became and how much.
        self.in_reach: set[Cell] = set()
        self.growth_rewards: list[tuple[Cell, BuildingKind, float]] = []
        self._months = 0
        self._land_cells = int((tile_map.terrain == Terrain.LAND).sum())
        if rule.trades:
            self.trade_goods()

    def trade_goods(self) -> set[Cell]:
        """Pulls down farms that towns have crowded out, lets stations hand out a month of goods,
        raises buildings with enough people around them and counts what the served buildings add
        to every ladder's chances; returns the cells whose buildings changed."""
        changed = self._crowd_out_farms()
        self.economy.year = self.year
        if self._trades_anything():
            self.trade = self.economy.trade(
                self._working_counts(), self._station_orders(), np.flatnonzero(self._occupied)
            )
            self._settle_stations(self.trade)
            self._pay_owners(self.trade)
            self.in_reach = self._station_reach()
            changed |= self._raise_buildings()
            self._count_served(self.trade)
        else:
            changed |= self._raise_buildings()
        for place in self._places:
            place.highest = float(place.chances.max())
        return changed

    def _trades_anything(self) -> bool:
        """Whether a month's trade can change anything: not before there is a railway or a
        stand-in for one, when no station hands out or gathers goods, once it has been worked
        out the first time."""
        return self.trade is None or self.railway is not None or self.assumed_shares is not None

    def _count_served(self, trade: Trade) -> None:
        """Works out each cell's served points of every kind, and what they add to the chances:
        in the cells traded on, and the cells traded on last month that have no buildings now."""
        served = {points: np.zeros(trade.cells.size) for points in Points}
        assumed = None if self.assumed_shares is None else self.assumed_shares[trade.cells]
        for kind in trade.got:
            done = trade.done((kind,))
            if not done.any() and assumed is None:
                continue
            wanted = trade.wanted((kind,))
            share = done / np.where(wanted > 0, wanted, 1.0)
            if assumed is not None:
                share = np.where(np.isnan(assumed), share, assumed)
            points, value = self.rule.points[kind]
            count = self._counts[kind].ravel()[trade.cells].astype(np.float64)
            boost = self._boost(kind, Effect.GROWTH)
            if boost:
                count += boost * self.owned(kind).ravel()[trade.cells]
            served[points] += served_multipliers(share) * value * count
        emptied = np.setdiff1d(self._traded, trade.cells, assume_unique=True)
        self._traded = trade.cells
        for points, here in served.items():
            old = self._served[points]
            now = np.concatenate([here, np.zeros(emptied.size)])
            every = np.concatenate([trade.cells, emptied])
            moved = now != old[every]
            if not moved.any():
                continue
            cells = every[moved]
            added = SERVED_POINT_WORTH * (now[moved] - old[cells])
            for place, per_doubling in self._by_points[points]:
                place.chances[cells] += per_doubling * added * place.land[cells]
            old[cells] = now[moved]

    @property
    def year(self) -> int:
        return self.start_year + self._months // 12

    def _raise_buildings(self) -> set[Cell]:
        """Turns buildings into the next kind on their ladder wherever enough people have come
        to live around them, the most crowded first and at most one step at a time."""
        changed: set[Cell] = set()
        for ladder in self.rule.ladders:
            for small, big in reversed(list(pairwise(ladder.kinds))):
                rise = ladder.rises.get(small)
                if rise is None or (rise.only_one and self._counts[big].any()):
                    continue
                people = self._nearby[Points.POPULATION]
                ready = np.argwhere((self._counts[small] > 0) & (people >= rise.people)).tolist()
                ready.sort(key=lambda at: -int(people[*at]))
                for row, col in ready[:1] if rise.only_one else ready:
                    cell = (row, col)
                    around = [cell, *self._grid.neighbours(*cell)]
                    changed |= self._replace(cell, around, [(small, 1)], big) or set()
        return changed

    def clear_room(self, cell: Cell, units: int, avoid: frozenset[Cell] = frozenset()) -> set[Cell]:
        """Makes `units` of room in the cell for track or a station, the smallest buildings first
        (railroads.sim.demolition prices them in the same order): each moves to a cell next to it
        with room for it (never into `avoid`) if it can, and otherwise comes down. Returns the
        cells whose buildings changed."""
        changed: set[Cell] = set()
        while self.constructions.space_left(cell) < units:
            here = [b for b in self.constructions.at(cell) if isinstance(b, Building)]
            if not here:
                break
            victim = leaving_order(here, self.rule.points)[0]
            target = self._room_beside(victim, avoid | {cell})
            owned = self._take_down(victim)
            changed.add(cell)
            if target is not None:
                self.constructions.build(Building(target, victim.kind))
                if victim.kind in self._counts:
                    self._add(target, victim.kind, 1)
                if owned:
                    self.own(target, victim.kind)
                changed.add(target)
            elif owned:
                changed |= self._relocate(victim)
        return changed

    def demolish(self, building: Building) -> None:
        """Takes a building down for good, the player's too."""
        self._take_down(building)

    def _room_beside(self, building: Building, avoid: frozenset[Cell]) -> Cell | None:
        """The cell next to the building's with the most room that would take it, if any."""
        options = [
            near
            for near in self._grid.neighbours(*building.cell)
            if near not in avoid
            and self.constructions.why_not(Building(near, building.kind)) is None
        ]
        return max(options, key=self.constructions.space_left, default=None)

    def _take_down(self, building: Building) -> bool:
        """Pulls a building down; returns whether it was one the player owned."""
        owned = self._lose(building.cell, building.kind, 1) > 0
        self.constructions.remove(building)
        if building.kind in self._counts:
            self._add(building.cell, building.kind, -1)
        return owned

    def _crowd_out_farms(self) -> set[Cell]:
        farms = self._counts.get(BuildingKind.FARM)
        if farms is None:
            return set()
        crowded = np.argwhere((farms > 0) & (self._nearby[Points.POPULATION] >= FARMS_CROWDED_OUT))
        changed: set[Cell] = set()
        for row, col in crowded.tolist():
            gone = int(farms[row, col])
            owned = self._lose((row, col), BuildingKind.FARM, gone)
            for _ in range(gone):
                self.constructions.remove(Building((row, col), BuildingKind.FARM))
            self._add((row, col), BuildingKind.FARM, -gone)
            changed.add((row, col))
            for _ in range(owned):
                changed |= self._relocate(Building((row, col), BuildingKind.FARM))
        return changed

    def _station_orders(self) -> dict[Cell, StationOrder]:
        if self.railway is None:
            return {}
        orders = {}
        for station in sorted(self.railway.connected()):
            goods = self.railway.station_goods(station)
            delivery, collection = self.railway.reach(station)
            freight = {g: a for g, a in goods.stockpile.items() if g not in TRAVELLERS}
            orders[station] = StationOrder(
                dict(goods.brought), goods.room, delivery, collection, freight
            )
        return orders

    def _settle_stations(self, trade: Trade) -> None:
        if self.railway is None:
            return
        for station, handed in trade.handed.items():
            goods = self.railway.station_goods(station)
            for good, amount in handed.items():
                goods.release(good, amount)
        for station, supplied in trade.supplied.items():
            goods = self.railway.station_goods(station)
            for good, amount in supplied.items():
                goods.stockpile[good] = max(0.0, goods.stockpile.get(good, 0.0) - amount)
        for station, gathered in trade.gathered.items():
            goods = self.railway.station_goods(station)
            for good, amount in gathered.items():
                goods.stockpile[good] = goods.stockpile.get(good, 0.0) + amount
        capacity = {}
        for station in self.railway.stations():
            ring = self._inner_ring(station)
            travellers = self._travellers_in(ring)
            goods = self.railway.station_goods(station)
            for good, amount in travellers.items():
                goods.board(good, amount)
            capacity[station] = capacity_of(self._needs_in(ring) | self._as_needs(travellers))
        self.railway.year = self.year
        self.railway.turn_month(capacity)
        for station, handed_for in trade.handed_for.items():
            for (need, good), amount in handed_for.items():
                self.railway.pay_for(station, need, good, amount)
        for station, supplied_for in trade.supplied_for.items():
            for (need, good), amount in supplied_for.items():
                self.railway.pay_for_passing(station, need, good, amount)

    def measure_demand(self, station: Cell) -> None:
        """Sets a new station's demand capacity from the buildings around it now, so cargo pays
        before its first month turns over."""
        if self.railway is not None:
            ring = self._inner_ring(station)
            needs = self._needs_in(ring) | self._as_needs(self._travellers_in(ring))
            self.railway.station_demand(station).capacity = capacity_of(needs)

    def _inner_ring(self, station: Cell) -> npt.NDArray[np.intp]:
        """The cells a station hands goods out to, as flat indices."""
        radius = DELIVERY_RADIUS if self.railway is None else self.railway.reach(station)[0]
        reached = self._grid.within(station, radius)
        return np.array([np.ravel_multi_index(c, self._grid.shape) for c in reached])

    def _travellers_in(self, cells: npt.NDArray[np.intp]) -> dict[Good, float]:
        """The passengers and mail the buildings in these cells send, and want, in a month."""
        passengers = 0
        for kind, count in self._counts.items():
            points, value = self.rule.points[kind]
            passengers += (
                PASSENGERS_PER_POINT.get(points, 0) * value * int(count.ravel()[cells].sum())
            )
        return {
            Good.PASSENGERS: float(passengers),
            Good.MAIL: float(passengers // MAIL_PER_PASSENGERS),
        }

    @staticmethod
    def _as_needs(travellers: dict[Good, float]) -> dict[Need, float]:
        return {Need(good.value): amount for good, amount in travellers.items()}

    def _needs_in(self, cells: npt.NDArray[np.intp]) -> dict[Need, float]:
        """What the buildings in these cells need in a month."""
        needs: dict[Need, float] = {}
        for kind in NEEDS:
            if kind not in self._counts:
                continue
            count = float(self._counts[kind].ravel()[cells].sum())
            for need, amount in needs_of(kind, self.year).items():
                needs[need] = needs.get(need, 0.0) + count * amount
        return needs

    def _reckon(self, place: _Place, cells: npt.NDArray[np.intp]) -> None:
        """Works out the ladder's chances in these cells afresh."""
        raw = np.full(cells.size, place.ladder.base_chance)
        for points, per_doubling in place.ladder.chance_per_doubling.items():
            nearby = place.ladder.pull(self._nearby[points].ravel()[cells])
            raw += per_doubling * (nearby + SERVED_POINT_WORTH * self._served[points][cells])
        place.chances[cells] = raw * place.land[cells]

    def satisfaction(self) -> npt.NDArray[np.int8]:
        """How satisfied each cell's buildings are on the whole, or -1 where none trade."""
        if self.trade is None:
            return np.full(self._grid.shape, -1, dtype=np.int8)
        levels = np.full(self._grid.rows * self._grid.cols, -1, dtype=np.int8)
        everything = tuple(self.trade.got)
        trading = self.trade.wanted(everything) > 0
        levels[self.trade.cells[trading]] = self.trade.levels()[trading]
        if self.assumed_shares is not None:
            cells = self.trade.cells[trading]
            assumed = self.assumed_shares[cells]
            given = ~np.isnan(assumed)
            levels[cells[given]] = levels_of(assumed[given])
        return levels.reshape(self._grid.shape)

    def owned(self, kind: BuildingKind) -> IntGrid:
        """How many buildings of a kind the player owns in each cell."""
        return self._owned.get(kind, np.zeros(self._grid.shape, dtype=np.int64))

    def owned_kinds(self) -> list[BuildingKind]:
        """The kinds of building the player owns any of."""
        return [kind for kind, grid in self._owned.items() if grid.any()]

    def owner_share(self, kind: BuildingKind) -> float:
        """The player's share of what one building of a kind they own sells, with their tech."""
        return OWNER_SHARE + self._boost(kind, Effect.SHARE)

    def output_factor(self, kind: BuildingKind) -> float:
        """How many buildings' worth one of a kind the player owns makes, with their tech."""
        return 1.0 + self._boost(kind, Effect.PRODUCTION)

    def own(self, cell: Cell, kind: BuildingKind) -> bool:
        """Makes one more of the kind in the cell the player's, if any is left to own."""
        grid = self._owned.setdefault(kind, np.zeros(self._grid.shape, dtype=np.int64))
        if grid[cell] >= self._counts[kind][cell]:
            return False
        grid[cell] += 1
        return True

    def price_to_buy(self, kind: BuildingKind) -> float | None:
        """What one building of a kind costs the player; None for what cannot be bought."""
        makes = makes_of(kind, self.year)
        if not makes or kind not in self._counts:
            return None
        month = sum(amount * price_of(good) for good, amount in makes.items())
        discount = 0.0 if self.railway is None else self.railway.research.bonus(Effect.DISCOUNT)
        return 12 * PAYBACK_YEARS * OWNER_SHARE * month * max(MOST_DISCOUNT, 1.0 - discount)

    def buy(self, cell: Cell, kind: BuildingKind) -> str | None:
        """Buys one building of a kind in a cell; why not, if it cannot be bought."""
        price = self.price_to_buy(kind)
        if price is None:
            return f"A {kind.value} cannot be bought"
        if self.owned(kind)[cell] >= self._counts[kind][cell]:
            return f"You already own every {kind.value} here"
        if self.railway is None or not self.railway.finances.can_afford(price):
            return "Not enough cash"
        self.railway.finances.spend(price, "buildings bought")
        self.own(cell, kind)
        return None

    def _lose(self, cell: Cell, kind: BuildingKind, gone: int) -> int:
        """Counts `gone` buildings of a kind leaving a cell, the player's last; returns how
        many of them the player owned."""
        grid = self._owned.get(kind)
        if grid is None or not grid[cell]:
            return 0
        unowned = int(self._counts[kind][cell] - grid[cell])
        owned = max(0, gone - unowned)
        grid[cell] -= owned
        return owned

    def _relocate(self, building: Building) -> set[Cell]:
        """Puts up again, as the player's, a building of theirs that growth pulled down: in the
        nearest cell with room and the land its ladder grows on."""
        ladder = next((ld for ld in self.rule.ladders if building.kind in ld.kinds), None)
        reached = self._grid.within(building.cell, RELOCATE_STEPS)
        for near, _ in sorted(reached.items(), key=lambda item: item[1]):
            if near == building.cell:
                continue
            moved = Building(near, building.kind)
            if self.constructions.why_not(moved) is not None:
                continue
            if ladder is not None:
                flat = np.array([np.ravel_multi_index(near, self._grid.shape)])
                land = ladder.land.weights(self.constructions.tile_map, flat, self._near)
                if land[0] <= 0:
                    continue
            self.build(moved)
            self.own(near, building.kind)
            return {near}
        return set()

    def _station_reach(self) -> set[Cell]:
        if self.railway is None:
            return set()
        reached: set[Cell] = set()
        for station in self.railway.stations():
            reached |= set(self._grid.within(station, self.railway.reach(station)[0]))
        return reached

    def stars(self, kind: BuildingKind) -> int:
        """How high a kind stands on its ladder: 1 for the first kind, 2 for the next..."""
        if kind == BuildingKind.SKYSCRAPER:
            return SKYSCRAPER_STARS
        ladder = next((ld for ld in self.rule.ladders if kind in ld.kinds), None)
        return 1 if ladder is None else ladder.kinds.index(kind) + 1

    def _load_worth(self, kind: BuildingKind) -> dict[Tree, float]:
        """What one unit of a kind's goods is worth to each tree: what it makes, or for those
        that make nothing, the mix of what they need."""
        makes = makes_of(kind, self.year)
        if makes:
            goods = {good: amount for good, amount in makes.items()}
        else:
            goods = {FITS[n][0]: amount for n, amount in needs_of(kind, self.year).items()}
        total = sum(goods.values())
        worth: dict[Tree, float] = {}
        for good, amount in goods.items():
            tree = GOOD_TREES[good]
            worth[tree] = worth.get(tree, 0.0) + amount / total * price_of(good)
        return worth

    def _reward_growth(self, cell: Cell, kind: BuildingKind) -> None:
        """Gives the player experience when a building in their stations' reach rises: as much
        as a train full of its goods would earn, for every star it has."""
        if self.railway is None or cell not in self.in_reach:
            return
        stars = self.stars(kind)
        if stars < 2:
            return
        gained = 0.0
        for tree, worth in self._load_worth(kind).items():
            xp = stars * TRAIN_LOAD * worth
            self.railway.research.earn(tree, xp)
            gained += xp
        self.growth_rewards.append((cell, kind, gained))

    def _boost(self, kind: BuildingKind, effect: Effect) -> float:
        """What the player's tech adds, of one effect, to the buildings of a kind they own."""
        tree = tree_of(kind)
        if self.railway is None or tree is None:
            return 0.0
        return self.railway.research.bonus(effect, tree)

    def _working_counts(self) -> Mapping[BuildingKind, npt.NDArray[np.int64 | np.float64]]:
        """The buildings as they trade: each the player owns counts as more than one, by what
        their production tech adds."""
        if not self._owned:
            return self._counts
        counts: dict[BuildingKind, npt.NDArray[np.int64 | np.float64]] = dict(self._counts)
        for kind, grid in self._owned.items():
            boost = self._boost(kind, Effect.PRODUCTION)
            if boost and grid.any():
                counts[kind] = self._counts[kind] + boost * grid
        return counts

    def _pay_owners(self, trade: Trade) -> None:
        """Pays the player their share of what the buildings they own sold this month, which
        is also experience in their trees, and counts what shops around the player's stations
        took as commerce experience."""
        self.owner_income = {}
        if self.railway is None:
            return
        research = self.railway.research
        for (kind, good), (given, _) in trade.goods_gave.items():
            grid = self._owned.get(kind)
            if grid is None:
                continue
            owned = grid.ravel()[trade.cells].astype(np.float64)
            if not owned.any():
                continue
            boost = self._boost(kind, Effect.PRODUCTION)
            count = self._counts[kind].ravel()[trade.cells] + boost * owned
            theirs = owned * (1.0 + boost) / np.maximum(count, 1.0)
            share = OWNER_SHARE + self._boost(kind, Effect.SHARE)
            earned = float((given * theirs).sum()) * price_of(good) * share
            self.owner_income[kind] = self.owner_income.get(kind, 0.0) + earned
        for kind, earned in self.owner_income.items():
            tree = tree_of(kind)
            if tree is not None:
                research.earn(tree, earned)
        total = sum(self.owner_income.values())
        if total > 0:
            self.railway.finances.earn(total, "owned buildings")
        for (kind, need), (got, _) in trade.goods_got.items():
            if kind in SHOPS:
                research.earn(Tree.COMMERCE, float(got.sum()) * price_of(FITS[need][0]))

    def build(self, building: Building) -> None:
        """Puts up a building no ladder grew, counted like any other."""
        self.constructions.build(building)
        self._add(building.cell, building.kind, 1)

    def count(self, kind: BuildingKind) -> IntGrid:
        """How many grown buildings of a kind stand in each cell."""
        return self._counts[kind]

    def points(self, points: Points = Points.POPULATION) -> IntGrid:
        """Each cell's points of one kind."""
        total = np.zeros(self._grid.shape, dtype=np.int64)
        for kind, count in self._counts.items():
            given, value = self.rule.points[kind]
            if given == points:
                total += value * count
        return total

    def population(self) -> int:
        """The population points of the whole map."""
        return sum(
            value * int(self._counts[kind].sum())
            for kind, (given, value) in self.rule.points.items()
            if given == Points.POPULATION
        )

    def nearby(self, points: Points = Points.POPULATION) -> IntGrid:
        """The points of one kind in each cell's neighbourhood."""
        return self._nearby[points]

    def chances(self, ladder: Ladder = TOWNS) -> FloatGrid:
        """Each cell's chance of a turn on the ladder this month."""
        place = next(place for place in self._places if place.ladder == ladder)
        cells = np.arange(place.chances.size)
        return np.where(self._meets_needs(place, cells), place.chances, 0.0).reshape(
            self._grid.shape
        )

    def month(self, rng: np.random.Generator) -> set[Cell]:
        """Grows one month; returns the cells whose buildings changed."""
        self._months += 1
        if self._months % 12 == 0:
            stations = 0 if self.railway is None else len(self.railway.stations())
            self.history.close_year(self.year - 1, stations, self.population())
        changed: set[Cell] = set()
        self.forests_changed = set()
        if self.rule.trades and self._months % self.rule.trade_every_months == 0:
            changed = self.trade_goods()
        if self.novelties is not None and self._months % 12 == 0:
            changed |= self.novelties.year(self, rng)
        every = self.rule.forests_every_months
        if self.rule.forests is not None and self._months % every == 0:
            changed |= self._change_forests(self.rule.forests, every, rng)
        turns: list[tuple[int, Ladder]] = []
        growing = [
            place
            for place in self._places
            if place.ladder.from_year <= self.year and place.cells.size and place.highest > 0
        ]
        thin = [place for place in growing if place.highest <= THINNING_UP_TO]
        tries = dict(
            zip(
                map(id, thin),
                rng.binomial([p.cells.size for p in thin], [p.highest for p in thin]).tolist(),
                strict=True,
            )
        )
        for place in growing:
            drawn = tries.get(id(place))
            if drawn == 0:
                continue
            hits = self._hits(place, rng, drawn)
            turns += [(index, place.ladder) for index in hits.tolist()]
        for turn in rng.permutation(len(turns)).tolist():
            index, ladder = turns[turn]
            changed |= self._turn(divmod(index, self._grid.cols), ladder, rng)
        return changed

    def _change_forests(self, rule: ForestRule, months: int, rng: np.random.Generator) -> set[Cell]:
        """Clears and spreads forests over `months`, closes the lumber mills left without
        forest, and weighs the land again around them; returns the cells that changed."""
        tile_map = self.constructions.tile_map
        people = (
            self._nearby[Points.POPULATION] / POPULATION_PER_CLEARING_POINT
            + self._nearby[Points.INDUSTRY]
        ).ravel()
        mills = self._nearby[Points.TIMBER].ravel()
        cleared, grown = forest_turn(tile_map, self._near, people, mills, rule, months, rng)
        cols = self._grid.cols
        grown = np.array(
            [index for index in grown.tolist() if not self.constructions.at(divmod(index, cols))],
            dtype=np.intp,
        )
        if not cleared.size and not grown.size:
            return set()
        tile_map.set_vegetation(cleared, Vegetation.GRASS)
        tile_map.set_vegetation(grown, Vegetation.FOREST)
        changed_cells = np.concatenate([cleared, grown])
        self.forests_changed = {divmod(index, cols) for index in changed_cells.tolist()}
        changed = set(self.forests_changed)
        if BuildingKind.LUMBER_MILL in self._counts:
            count = self._counts[BuildingKind.LUMBER_MILL]
            mills = np.flatnonzero(count)
            forests = forests_nearby(tile_map, self._near, mills)
            closing = mills[forests < rule.mills_close_below]
            for row, col in (divmod(int(index), cols) for index in closing):
                for _ in range(int(count[row, col])):
                    mill = Building((row, col), BuildingKind.LUMBER_MILL)
                    if self._take_down(mill):
                        changed |= self._relocate(mill)
                changed.add((row, col))
        around = self._near[:, changed_cells].ravel()
        beside = np.unique(np.concatenate([changed_cells, around[around < self._near.shape[1]]]))
        own = np.unique(changed_cells)
        sharing: dict[int, list[_Place]] = {}
        for place in self._places:
            if place.ladder.land.on_vegetation:
                sharing.setdefault(id(place.land), []).append(place)
        for places in sharing.values():
            first = places[0]
            touched = beside if first.ladder.land.on_forest_nearby else own
            land = first.ladder.land.weights(tile_map, touched, self._near)
            differs = land != first.land[touched]
            if not differs.any():
                continue
            cells, land = touched[differs], land[differs]
            was = first.land[cells] > 0
            first.land[cells] = land
            allowed = _allowed(first.cells, cells[(land > 0) & ~was], first.land)
            for place in places:
                place.cells = allowed
                self._reckon(place, cells)
                place.highest = max(place.highest, float(place.chances[cells].max()))
        return changed

    def run(self, months: int, rng: np.random.Generator) -> None:
        for _ in range(months):
            self.month(rng)

    def _hits(
        self, place: _Place, rng: np.random.Generator, tries: int | None = None
    ) -> npt.NDArray[np.intp]:
        """The cells that take a turn on the ladder this month. With small chances, cells are
        drawn as if all had the highest chance and each kept by how its own compares, so a month
        costs what grows rather than the size of the map: `tries` of them, if already drawn
        for every ladder at once."""
        cells = place.cells
        if not cells.size:
            return cells
        highest = place.highest
        if highest <= 0:
            return cells[:0]
        if highest > THINNING_UP_TO:
            drawn = cells[rng.random(cells.size, dtype=np.float32) < place.chances[cells]]
        else:
            if tries is None:
                tries = int(rng.binomial(cells.size, highest))
            if not tries:
                return cells[:0]
            tried = np.unique(cells[rng.integers(0, cells.size, tries)])
            drawn = tried[rng.random(tried.size) * highest < place.chances[tried]]
        if not drawn.size or not (place.ladder.needs or place.ladder.crowded_at is not None):
            return drawn
        return drawn[self._meets_needs(place, drawn)]

    def _meets_needs(self, place: _Place, cells: npt.NDArray[np.intp]) -> npt.NDArray[np.bool_]:
        meets = np.ones(cells.size, dtype=bool)
        for points, needed in place.ladder.needs.items():
            meets &= self._nearby[points].ravel()[cells] >= needed
        if place.ladder.crowded_at is not None:
            meets &= self._nearby[Points.POPULATION].ravel()[cells] < place.ladder.crowded_at
        return meets

    def _turn(self, cell: Cell, ladder: Ladder, rng: np.random.Generator) -> set[Cell]:
        neighbourhood = [cell, *self._grid.neighbours(*cell)]
        if ladder.skyscrapers and (built := self._skyscraper(cell, neighbourhood, rng)):
            return built
        for small, big in reversed(list(pairwise(ladder.kinds))):
            merge = ladder.merges.get(small)
            if merge is None or self._around(small, neighbourhood) < merge.needed:
                continue
            merged = self._replace(cell, neighbourhood, [(small, merge.pulled_down)], big)
            if merged is not None:
                return merged
        first = ladder.kinds[0]
        if (
            ladder.land_cells_each is not None
            and self._counts[first].sum() >= self._land_cells // ladder.land_cells_each
        ):
            return set()
        if not self._ladder_allows(ladder, cell) or self._not_open(ladder, cell, rng):
            return set()
        new = Building(cell, first)
        if self.constructions.site_problem(new) is not None:
            return set()
        if new.space > self.constructions.space_left(cell):
            return self._spill(cell, ladder, rng)
        self.constructions.build(new)
        self._add(cell, first, 1)
        return {cell}

    def _not_open(self, ladder: Ladder, cell: Cell, rng: np.random.Generator) -> bool:
        """Whether a turn on flat grassland that is not empty falls through, as all but one in
        the land's open field weight do."""
        if ladder.land.open_field == 1.0 or not self.constructions.at(cell):
            return False
        tile = self.constructions.tile_map.tile(cell)
        if tile.relief != Relief.FLAT or tile.vegetation != Vegetation.GRASS:
            return False
        return bool(rng.random() * ladder.land.open_field >= 1.0)

    def _ladder_allows(self, ladder: Ladder, cell: Cell) -> bool:
        """Whether the ladder's own caps let one more of its first kind go up in the cell."""
        first = ladder.kinds[0]
        if (
            ladder.crowded_at is not None
            and self._nearby[Points.POPULATION][cell] >= ladder.crowded_at
        ):
            return False
        if ladder.most_per_cell is not None and self._counts[first][cell] >= ladder.most_per_cell:
            return False
        if ladder.most_nearby is not None:
            neighbourhood = [cell, *self._grid.neighbours(*cell)]
            if self._around(first, neighbourhood) >= ladder.most_nearby:
                return False
        return ladder.apart is None or not self._any_within(ladder, cell)

    def _spill(self, cell: Cell, ladder: Ladder, rng: np.random.Generator) -> set[Cell]:
        """Puts up the ladder's first kind in the nearest cell that will take it, picked among
        cells equally near at random by how well their land suits it; returns the cell it went up
        in, if it found one."""
        reached = [(near, steps) for near, steps in self._grid.within(cell, SPILL_STEPS).items()]
        steps = np.array([step for _, step in reached])
        rows, cols = zip(*(near for near, _ in reached), strict=True)
        flat = np.ravel_multi_index((rows, cols), self._grid.shape)
        land = ladder.land.weights(self.constructions.tile_map, flat, self._near)
        first = ladder.kinds[0]
        suits = rng.random(steps.size) ** (1.0 / np.maximum(land, 1e-12))
        for index in np.lexsort((-suits, steps)).tolist():
            near = reached[index][0]
            new = Building(near, first)
            if (
                near != cell
                and land[index] > 0
                and self.constructions.why_not(new) is None
                and self._ladder_allows(ladder, near)
            ):
                self.constructions.build(new)
                self._add(near, first, 1)
                return {near}
        return set()

    def _any_within(self, ladder: Ladder, cell: Cell) -> bool:
        """Whether any building on the ladder stands within its `apart` steps of the cell."""
        reached = self._grid.within(cell, ladder.apart or 0)
        return any(int(self._counts[kind][near]) for kind in ladder.kinds for near in reached)

    def _around(self, kind: BuildingKind, neighbourhood: list[Cell]) -> int:
        count = self._counts[kind]
        return sum(int(count[near]) for near in neighbourhood)

    def _skyscraper(
        self, cell: Cell, neighbourhood: list[Cell], rng: np.random.Generator
    ) -> set[Cell] | None:
        rule = self.rule
        if self._nearby[Points.POPULATION][cell] < rule.skyscraper_people:
            return None
        if self._around(BuildingKind.FLATS, neighbourhood) < SKYSCRAPER_FLATS:
            return None
        if rng.random() >= rule.skyscraper_chance:
            return None
        takes = [(BuildingKind.FLATS, SKYSCRAPER_FLATS)]
        built = self._replace(cell, neighbourhood, takes, BuildingKind.SKYSCRAPER)
        if built is None:
            return None
        return built | self._house(cell, SKYSCRAPER_HOMES)

    def _house(self, cell: Cell, homes: int) -> set[Cell]:
        """Puts up as many as `homes` houses in the nearest cells around this one with room and
        land for them; returns the cells they went up in."""
        changed: set[Cell] = set()
        reached = sorted(self._grid.within(cell, HOMES_STEPS).items(), key=lambda item: item[1])
        for near, _ in reached:
            house = Building(near, BuildingKind.HOUSE)
            if (
                TOWN_LAND.weights(self.constructions.tile_map, self._flat([near]), self._near)[0]
                <= 0
            ):
                continue
            while homes and self.constructions.why_not(house) is None:
                self.constructions.build(house)
                self._add(near, BuildingKind.HOUSE, 1)
                changed.add(near)
                homes -= 1
            if not homes:
                break
        return changed

    def _flat(self, cells: list[Cell]) -> npt.NDArray[np.intp]:
        rows, cols = zip(*cells, strict=True)
        return np.ravel_multi_index((rows, cols), self._grid.shape)

    def _replace(
        self,
        cell: Cell,
        neighbourhood: list[Cell],
        takes: list[tuple[BuildingKind, int]],
        new_kind: BuildingKind,
    ) -> set[Cell] | None:
        """Pulls down so many of each kind in the neighbourhood, the cell's own first and then
        from the fullest cells, and puts up a building of the new kind in the cell; None, and
        nothing pulled down, if there are too few or it would not fit."""
        new = Building(cell, new_kind)
        if self.constructions.site_problem(new) is not None:
            return None
        plan: list[tuple[Cell, BuildingKind, int]] = []
        for kind, wanted in takes:
            count = self._counts[kind]
            if self._around(kind, neighbourhood) < wanted:
                return None
            others = sorted(neighbourhood[1:], key=lambda near, count=count: -int(count[near]))
            for near in [cell, *others]:
                take = min(wanted, int(count[near]))
                if take:
                    plan.append((near, kind, take))
                wanted -= take
                if not wanted:
                    break
        freed = sum(take * Building(cell, kind).space for near, kind, take in plan if near == cell)
        if new.space > self.constructions.space_left(cell) + freed:
            return None
        owned = 0
        for near, kind, take in plan:
            owned += self._lose(near, kind, take)
            for _ in range(take):
                self.constructions.remove(Building(near, kind))
            self._add(near, kind, -take)
        self.constructions.build(new)
        self._add(cell, new_kind, 1)
        if owned:
            self.own(cell, new_kind)
        self._reward_growth(cell, new_kind)
        return {cell} | {near for near, _, _ in plan}

    def _add(self, cell: Cell, kind: BuildingKind, how_many: int) -> None:
        self._counts[kind][cell] += how_many
        self._occupied[cell] += how_many
        points, value = self.rule.points[kind]
        neighbourhood = [cell, *self._grid.neighbours(*cell)]
        rows, cols = zip(*neighbourhood, strict=True)
        before = self._nearby[points][rows, cols]
        self._nearby[points][rows, cols] += how_many * value
        after = self._nearby[points][rows, cols]
        flat = np.ravel_multi_index((rows, cols), self._grid.shape)
        pulls: dict[int, FloatGrid] = {}
        for place, per_doubling in self._by_points[points]:
            ladder = place.ladder
            added = pulls.get(ladder.draws_from)
            if added is None:
                added = pulls[ladder.draws_from] = ladder.pull(after) - ladder.pull(before)
            chances = place.chances[flat] + per_doubling * added * place.land[flat]
            place.chances[flat] = chances
            if how_many > 0:
                place.highest = max(place.highest, float(chances.max()))

    @staticmethod
    def _neighbourhood_sum(values: IntGrid, rows: IntGrid, cols: IntGrid) -> IntGrid:
        own_rows, own_cols = np.indices(values.shape)
        off_map = (rows == own_rows) & (cols == own_cols)
        return values + np.where(off_map, 0, values[rows, cols]).sum(axis=0)
