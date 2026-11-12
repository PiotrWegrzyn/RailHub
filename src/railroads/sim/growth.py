"""The map grows by itself, a month at a time, like a cellular automaton, by the ladders of
railroads.sim.ladders: towns, farms, orchards, mines, lumber mills, ports and every kind of works.

A building's points count as served MOST_SERVED times over at 100% satisfaction, and by its
satisfaction above or below that (railroads.sim.economy: what it demands and makes that stations
handled, and its bonus goods on top), each at SERVED_POINT_WORTH of a doubling, so only what
stations supply compounds. A building that finds no room in its cell spills over into a cell nearby
(railroads.sim.room).

On town turns, a neighbourhood with SKYSCRAPER_PEOPLE population and SKYSCRAPER_FLATS blocks of
flats may put up a skyscraper, SKYSCRAPER_CHANCE of the time: a business centre that pulls down the
flats and gives corporate points, whose people move into SKYSCRAPER_HOMES houses in the nearest
cells with room for them.

Every TRADE_EVERY_MONTHS the stations hand out their goods and buildings sell to each other
(railroads.sim.economy, settled with the railway by railroads.sim.station_trade), which leaves
each kind of building in each cell unsatisfied, content or thriving, and pays the player for the
buildings they own (railroads.sim.ownership). Each building in a cell counts on its own: of ten
houses with food for two, two are fed and eight starve. Buildings that starve for DIES_AFTER_MONTHS
on end are pulled down, as many as starved all that while, the player's too; those that make a
bonus good never are. Houses go up only as well as the buildings around them are fed: a turn on a ladder of homes
goes ahead with the share of the food the neighbourhood needs that it got (FED_LADDERS), and
always where nobody lives yet.

Growth keeps the tally the other parts share: how many of each kind stand in each cell (counts),
the points around each cell (nearby) and every ladder's chances, all kept up to date by add. Every
change to the counts goes through add, so it can also log them (log_changes): a month's summary
(railroads.sim.summaries) is worked out from the log instead of comparing the counts before and
after.

Forests come and go as well (railroads.sim.forests): people, industry and above all lumber mills
clear them, they spread onto open grassland beside them, and the land weighs again wherever they
changed.

A new map runs this for MONTHS_BEFORE_PLAY before play starts; in play it carries on, one month
every few minutes.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import START_YEAR
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import (
    BONUS_PER_GOOD,
    Economy,
    FloatCells,
    Trade,
    levels_of,
    served_multipliers,
)
from railroads.sim.forests import change_forests
from railroads.sim.goods import BONUS_NEEDS, FITS, MAKES, Need
from railroads.sim.grids import Cell, Grid
from railroads.sim.ladders import (
    FARMS_CROWDED_OUT,
    GROWTH,
    TOWNS,
    Cells,
    FloatGrid,
    GrowthRule,
    IntGrid,
    Ladder,
    Land,
    Points,
)
from railroads.sim.ownership import Owners
from railroads.sim.railway import Railway
from railroads.sim.room import Room
from railroads.sim.station_trade import StationTrade
from railroads.sim.stats import History
from railroads.sim.steps import Steps, finish
from railroads.sim.tech import Effect
from railroads.sim.tilemap import Relief, Terrain, Vegetation

if TYPE_CHECKING:
    from railroads.sim.novelties import Novelties

MONTHS_BEFORE_PLAY = 1000 * 12
SKYSCRAPER_FLATS = 3
SKYSCRAPER_HOMES = 10
# Above this chance, a ladder draws for every cell instead of thinning.
THINNING_UP_TO = 0.05
# What one served point adds to a ladder's chance, as a share of what a doubling of the points
# nearby adds.
SERVED_POINT_WORTH = 2.0
SKYSCRAPER_STARS = 4
# How long the buildings of a kind in a cell last with nothing of what they need or make going.
DIES_AFTER_MONTHS = 30 * 12
# Ladders of homes, which grow only as well as the neighbourhood is fed.
FED_LADDERS = frozenset({BuildingKind.HOUSE})
# Kinds that make a bonus good, which never die out.
UNDYING = frozenset(
    kind
    for kind, goods in MAKES.items()
    if any(good in FITS[need] for need in BONUS_NEEDS for good in goods)
)


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


class Growth:
    """The grown buildings on one map, counted per cell, and the rule that grows them."""

    # The net change in each cell's count of each kind while changes are logged (log_changes),
    # set only for that while, so it is never saved with the game.
    _log: dict[tuple[Cell, BuildingKind], int] | None = None
    # Per kind and cell, how many months some of its buildings have starved on end, and the fewest
    # that starved in any of those months. None in saves from before buildings died out, which
    # start counting afresh.
    _starving: dict[BuildingKind, npt.NDArray[np.int32]] | None = None
    # How many buildings have died out since the game began.
    died_out = 0
    # The food every cell's buildings got and needed last month, one past the last cell standing
    # for off the map. None in saves from before homes waited for food, where they don't.
    _food: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None = None

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
        # The buildings the player owns, the stations growth trades with, and the room it makes
        # (railroads.sim.ownership, railroads.sim.station_trade, railroads.sim.room).
        self.owners = Owners(self)
        self.station_trade = StationTrade(self)
        self.room = Room(self)
        self._months = 0
        self._land_cells = int((tile_map.terrain == Terrain.LAND).sum())
        if rule.trades:
            self.trade_goods()

    def trade_goods(self) -> set[Cell]:
        """Pulls down farms that towns have crowded out, lets stations hand out a month of goods,
        raises buildings with enough people around them and counts what the served buildings add
        to every ladder's chances; returns the cells whose buildings changed."""
        return finish(self._trade_steps())

    def _trade_steps(self) -> Steps[set[Cell]]:
        """trade_goods, pausing once the economy has traded."""
        changed = self._crowd_out_farms()
        self.economy.year = self.year
        self.economy.bonus_per_good = BONUS_PER_GOOD + (
            0.0 if self.railway is None else self.railway.research.bonus(Effect.BONUS_GOODS)
        )
        self.trade = yield from self.economy.trade_steps(
            self.owners.working_counts(),
            self.station_trade.orders(),
            np.flatnonzero(self._occupied),
        )
        yield
        self.station_trade.settle(self.trade)
        self.owners.pay(self.trade)
        self.station_trade.update_reach()
        changed |= self._die_out(self.trade)
        self._count_food(self.trade)
        changed |= self._raise_buildings()
        self._count_served(self.trade)
        for place in self._places:
            place.highest = float(place.chances.max())
        return changed

    def _count_food(self, trade: Trade) -> None:
        size = self._occupied.size + 1
        got, needed = np.zeros(size, dtype=np.float32), np.zeros(size, dtype=np.float32)
        for (_, need), (taken, wanted) in trade.goods_got.items():
            if need == Need.FOOD:
                got[trade.cells] += taken
                needed[trade.cells] += wanted
        self._food = (got, needed)

    def fed(self, cells: npt.NDArray[np.intp]) -> FloatCells:
        """The share of the food they need that the buildings in and around each cell got last
        month; 1 where none need any."""
        if self._food is None:
            return np.ones(cells.size)
        got, needed = self._food
        around = np.vstack([cells, self._near[:, cells]])
        wanted = needed[around].sum(axis=0)
        return np.where(wanted > 0, got[around].sum(axis=0) / np.where(wanted > 0, wanted, 1), 1.0)

    def starving(self, trade: Trade, kind: BuildingKind) -> npt.NDArray[np.int64]:
        """How many buildings of a kind starve in each cell traded on: those left over once the
        share of what they all want that they got is handed to as few of them as it can."""
        count = self._counts[kind].ravel()[trade.cells]
        wanted = trade.needed[kind] + trade.could_make[kind]
        share = np.clip(trade.share((kind,)), 0.0, 1.0)
        served = np.ceil(share * count - 1e-9).astype(np.int64)
        return np.where(wanted > 0, np.maximum(count - served, 0), 0)

    def _die_out(self, trade: Trade) -> set[Cell]:
        """Counts the months buildings of each kind have starved in each cell, and pulls down as
        many as starved all through DIES_AFTER_MONTHS; returns the cells they stood in."""
        if self._starving is None:
            self._starving = {}
        months = self.rule.trade_every_months
        dead: list[tuple[int, BuildingKind, int]] = []
        for kind in trade.got:
            if kind in UNDYING:
                continue
            starving = self.starving(trade, kind)
            counted = self._starving.get(kind)
            if counted is None:
                if not starving.any():
                    continue
                counted = self._starving[kind] = np.zeros((2, self._occupied.size), dtype=np.int32)
            streak, fewest = counted[:, trade.cells]
            fewest = np.where(streak > 0, np.minimum(fewest, starving), starving)
            streak = np.where(starving > 0, streak + months, 0)
            dying = streak >= DIES_AFTER_MONTHS
            dead += [
                (int(index), kind, int(gone))
                for index, gone in zip(trade.cells[dying], fewest[dying], strict=True)
            ]
            streak[dying] = 0
            counted[0, trade.cells] = streak
            counted[1, trade.cells] = np.where(streak > 0, fewest, 0)
        changed: set[Cell] = set()
        for index, kind, gone in dead:
            cell = divmod(index, self._grid.cols)
            self.owners.lose(cell, kind, gone)
            for _ in range(gone):
                self.constructions.remove(Building(cell, kind))
            self.add(cell, kind, -gone)
            self.died_out += gone
            changed.add(cell)
        return changed

    def _count_served(self, trade: Trade) -> None:
        """Works out each cell's served points of every kind, and what they add to the chances:
        in the cells traded on, and the cells traded on last month that have no buildings now."""
        served = {points: np.zeros(trade.cells.size) for points in Points}
        assumed = None if self.assumed_shares is None else self.assumed_shares[trade.cells]
        for kind in trade.got:
            bonus = trade.bonus.get(kind)
            got_any = trade.done((kind,)).any() or (bonus is not None and bonus.any())
            if not got_any and assumed is None:
                continue
            share = trade.share((kind,), empty=0.0)
            if assumed is not None:
                share = np.where(np.isnan(assumed), share, assumed)
            points, value = self.rule.points[kind]
            count = self._counts[kind].ravel()[trade.cells].astype(np.float64)
            boost = self.owners.boost(kind, Effect.GROWTH)
            if boost:
                count += boost * self.owners.owned(kind).ravel()[trade.cells]
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

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def counts(self) -> Mapping[BuildingKind, IntGrid]:
        """How many grown buildings of each kind stand in each cell."""
        return self._counts

    @property
    def near(self) -> Cells:
        """Every cell's neighbours as flat indices (railroads.sim.economy.Neighbourhoods)."""
        return self._near

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

    def take_down(self, building: Building) -> bool:
        """Pulls a building down, the player's too; returns whether it was one they owned."""
        owned = self.owners.lose(building.cell, building.kind, 1) > 0
        self.constructions.remove(building)
        if building.kind in self._counts:
            self.add(building.cell, building.kind, -1)
        return owned

    def _crowd_out_farms(self) -> set[Cell]:
        farms = self._counts.get(BuildingKind.FARM)
        if farms is None:
            return set()
        crowded = np.argwhere((farms > 0) & (self._nearby[Points.POPULATION] >= FARMS_CROWDED_OUT))
        changed: set[Cell] = set()
        for row, col in crowded.tolist():
            gone = int(farms[row, col])
            owned = self.owners.lose((row, col), BuildingKind.FARM, gone)
            for _ in range(gone):
                self.constructions.remove(Building((row, col), BuildingKind.FARM))
            self.add((row, col), BuildingKind.FARM, -gone)
            changed.add((row, col))
            for _ in range(owned):
                changed |= self.room.relocate(Building((row, col), BuildingKind.FARM))
        return changed

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

    def stars(self, kind: BuildingKind) -> int:
        """How high a kind stands on its ladder: 1 for the first kind, 2 for the next..."""
        if kind == BuildingKind.SKYSCRAPER:
            return SKYSCRAPER_STARS
        ladder = next((ld for ld in self.rule.ladders if kind in ld.kinds), None)
        return 1 if ladder is None else ladder.kinds.index(kind) + 1

    def build(self, building: Building) -> None:
        """Puts up a building no ladder grew, counted like any other."""
        self.constructions.build(building)
        self.add(building.cell, building.kind, 1)

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
        return finish(self.month_steps(rng))

    def month_steps(self, rng: np.random.Generator) -> Steps[set[Cell]]:
        """Grows one month in a few steps, which a caller can spread over frames; nothing else may
        change the game between them. Returns the cells whose buildings changed."""
        self._months += 1
        if self._months % 12 == 0:
            stations = 0 if self.railway is None else len(self.railway.stations())
            self.history.close_year(self.year - 1, stations, self.population())
        changed: set[Cell] = set()
        self.forests_changed = set()
        if self.rule.trades and self._months % self.rule.trade_every_months == 0:
            changed = yield from self._trade_steps()
            yield
        if self.novelties is not None and self._months % 12 == 0:
            changed |= self.novelties.year(self, rng)
        every = self.rule.forests_every_months
        if self.rule.forests is not None and self._months % every == 0:
            changed |= change_forests(self, self.rule.forests, every, rng)
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

    def reweigh_land(self, changed_cells: Cells) -> None:
        """Weighs the land again wherever forests came or went, and the cells beside them for
        ladders that care about forest nearby."""
        tile_map = self.constructions.tile_map
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
        if drawn.size and place.ladder.kinds[0] in FED_LADDERS:
            drawn = drawn[rng.random(drawn.size) < self.fed(drawn)]
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
        if not self.ladder_allows(ladder, cell) or self._not_open(ladder, cell, rng):
            return set()
        new = Building(cell, first)
        if self.constructions.site_problem(new) is not None:
            return set()
        if new.space > self.constructions.space_left(cell):
            return self.room.spill(cell, ladder, rng)
        self.constructions.build(new)
        self.add(cell, first, 1)
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

    def ladder_allows(self, ladder: Ladder, cell: Cell) -> bool:
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
        return built | self.room.house(cell, SKYSCRAPER_HOMES)

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
            owned += self.owners.lose(near, kind, take)
            for _ in range(take):
                self.constructions.remove(Building(near, kind))
            self.add(near, kind, -take)
        self.constructions.build(new)
        self.add(cell, new_kind, 1)
        if owned:
            self.owners.own(cell, new_kind)
        self.station_trade.reward(cell, new_kind)
        return {cell} | {near for near, _, _ in plan}

    @contextmanager
    def log_changes(self) -> Iterator[Mapping[tuple[Cell, BuildingKind], int]]:
        """Logs the net change in each cell's count of each kind while the block runs."""
        log: dict[tuple[Cell, BuildingKind], int] = {}
        self._log = log
        try:
            yield log
        finally:
            del self._log

    def add(self, cell: Cell, kind: BuildingKind, how_many: int) -> None:
        """Counts `how_many` buildings of a kind more (or fewer) in the cell, with the points
        they give its neighbourhood and every ladder's chances there."""
        if self._log is not None:
            self._log[(cell, kind)] = self._log.get((cell, kind), 0) + how_many
        self._counts[kind][cell] += how_many
        self._occupied[cell] += how_many
        starving = self._starving
        if starving is not None and kind in starving and not self._counts[kind][cell]:
            starving[kind][:, cell[0] * self._grid.cols + cell[1]] = 0
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
