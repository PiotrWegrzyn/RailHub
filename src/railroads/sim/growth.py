"""The map grows by itself, a month at a time, like a cellular automaton, by the ladders of
railroads.sim.ladders: towns, farms, orchards, mines, logging camps, ports and every kind of works.

Buildings breed (breeding_of): a cell's chance of another of a ladder's kinds is

    base + (base + pull x served) x log2(1 + drawing)

where base is the ladder's base chance (in railroads.sim.ladders.CHANCE_UNIT), pull
BREEDING_TIMES x base unless the ladder gives its own, as homes do so that few come up alone
while towns still grow; served the points the buildings of its own kind of points give in the
cell times their satisfaction (railroads.sim.economy), and drawing the points around the cell
that draw it, each times its weight: six times the people and twice the industry for houses,
food and people for food, people and industry for industry, mines, mills and ports, and twice
the people for shops, inns and churches. Every building also gives its points to its own
ladder, and a ladder drawn by industry is drawn by its own ladder's points OWN_KIND_TIMES as
strongly instead, by industry only CATEGORY_SHARE as strongly, and breeds by its own served
points the same way, so a served sawmill breeds sawmills and hardly any other works. So the base
chance holds anywhere the land allows, good surroundings raise it even where none of the kind
stands yet, and what is satisfied breeds far faster. A building that comes up in a cell
goes up in it or a random cell next to it, and one that finds no room there spills over further
(railroads.sim.room).

On town turns, a neighbourhood with SKYSCRAPER_PEOPLE population and SKYSCRAPER_FLATS blocks of
flats may put up a skyscraper, SKYSCRAPER_CHANCE of the time: a business centre that pulls down the
flats and gives corporate points, whose people move into SKYSCRAPER_HOMES houses in the nearest
cells with room for them.

Every TRADE_EVERY_MONTHS the stations hand out their goods and buildings sell to each other
(railroads.sim.economy, settled with the railway by railroads.sim.station_trade), which leaves each
kind of building in each cell unsatisfied, content or thriving, and pays the player for the
buildings they own (railroads.sim.ownership). What a cell's buildings of a kind get is shared
equally among them: of ten houses with food for two, each gets a fifth of what it wants, and they
starve only when nothing arrives and, for processors, nothing is left in their stockpile, or, for
homes, no food (railroads.sim.economy). Buildings that starve on end are pulled down, the player's too: homes after HOMES_DIE_AFTER_MONTHS,
processors and the rest after DIE_AFTER_MONTHS, or the years their ladder gives (starving_years). Producers of what the land gives and soldiers
never starve to death (UNDYING); fields go only when a town grows over them (crowded_at).

Growth keeps the tally the other parts share: how many of each kind stand in each cell (counts),
the points around each cell (nearby) and every ladder's chances, all kept up to date by add. Every
change to the counts goes through add, so it can also log them (log_changes): a month's summary
(railroads.sim.summaries) is worked out from the log instead of comparing the counts before and
after.

Once a year every ladder that has opened and stands below its least on the map (least_of) gets
one more of its first kind, in its likeliest cell: at least one of everything that opens in some
year, and of the key buildings (KEY_BUILDINGS) at least KEY_LEAST on a map of REFERENCE_LAND land
cells, more or fewer by the land, put on any land they could stand on where none is likely.

History happens too (railroads.sim.happenings): wars, crises and new things, a month at a time; in a war the
military grows military_growth times as fast (railroads.sim.events).

Forests come and go as well (railroads.sim.forests): people, industry and above all logging camps
clear them, they spread onto open grassland beside them, and the land weighs again wherever they
changed.

A new map runs this for MONTHS_BEFORE_PLAY before play starts; in play it carries on, one month
every few minutes.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import numpy.ma  # np.unique loads it on first use, which would stall the first month.
import numpy.typing as npt

from railroads.sim.breeding import (
    LIVELY_GROWTH,
    VENUES,
    Breeding,
    Channel,
    breeding_of,
    kind_channels,
)
from railroads.sim.clock import HISTORY_START, MONTHS_PER_YEAR, START_YEAR
from railroads.sim.constructions import (
    GREW_FROM,
    ONE_OFFS,
    Building,
    BuildingKind,
    Constructions,
)
from railroads.sim.dying import UNDYING, dies_after
from railroads.sim.economy import Economy
from railroads.sim.events import military_growth
from railroads.sim.forests import change_forests
from railroads.sim.goods import FITS, MAKES, RECIPES, Good
from railroads.sim.grids import Cell, FlatIndices, FloatArray, Grid, IntArray
from railroads.sim.ladder_types import Join, Ladder
from railroads.sim.ladders import (
    GROWTH,
    LADDERS,
    MILITARY,
    PASSENGERS_PER_POINT,
    TOWNS,
    GrowthRule,
    ladder_of,
)
from railroads.sim.ledger import Ledger
from railroads.sim.ownership import Owners
from railroads.sim.points import Points
from railroads.sim.power_plants import build_power_plants
from railroads.sim.railway import Railway
from railroads.sim.room import Room
from railroads.sim.station_trade import StationTrade
from railroads.sim.stats import History
from railroads.sim.steps import Steps, finish
from railroads.sim.surroundings import Land, Surroundings
from railroads.sim.tilemap import REFERENCE_LAND, Relief, Terrain, TileMap, Vegetation
from railroads.sim.tiles import era_come, kind_era
from railroads.sim.towns import Towns
from railroads.sim.trade import BONUS_PER_GOOD, Trade, levels_of
from railroads.sim.year_caches import year_cached
from railroads.sim.yearly import demands_of, makes_of

if TYPE_CHECKING:
    from railroads.sim.happenings import Happenings
    from railroads.sim.novelties import Novelties

MONTHS_BEFORE_PLAY = (START_YEAR - HISTORY_START) * MONTHS_PER_YEAR
SKYSCRAPER_FLATS = 3
SKYSCRAPER_HOMES = 10
# Above this chance, a ladder draws for every cell instead of thinning.
THINNING_UP_TO = 0.05
SKYSCRAPER_STARS = 4


# A processor or venue comes up only once something on the map makes each of its main inputs
# (inputs_of): a winery needs a vineyard, a sushi hut a fishery and a rice paddy. Only where the
# land lets their makers grow, then, do they ever open.


def ladder_open(ladder: Ladder, year: int) -> bool:
    """Whether a ladder has opened by a year: its year and the era of its first kind have come
    (railroads.sim.tiles.kind_era)."""
    return ladder.from_year <= year and era_come(kind_era(ladder.kinds[0]), year)


@year_cached
def makers_of(good: Good, year: int) -> frozenset[BuildingKind]:
    """Every kind of building that makes a good in a given year."""
    return frozenset(kind for kind in MAKES if good in makes_of(kind, year))


@year_cached
def inputs_of(kind: BuildingKind, year: int) -> tuple[frozenset[BuildingKind], ...]:
    """For each main input a processor or a venue must have in a given year, the kinds of
    building that make it, any one of which will do; nothing for other kinds. A processor's
    boosts are no main input, and an input another can stand in for is met by either."""
    recipe = RECIPES.get(kind)
    if recipe is not None:
        swaps = recipe.swaps_in(year)
        needs = [
            (need, *(swaps[need][:1] if need in swaps else ())) for need in recipe.inputs(year)
        ]
    elif kind in VENUES:
        needs = [(need,) for need in demands_of(kind, year)]
    else:
        return ()
    return tuple(
        frozenset(maker for need in fits for good in FITS[need] for maker in makers_of(good, year))
        for fits in needs
    )


# Kinds every map keeps at least KEY_LEAST of on REFERENCE_LAND land cells, in proportion to its
# land, once open: the only makers of what whole chains run on.
KEY_BUILDINGS = frozenset({BuildingKind.OIL_WELL, BuildingKind.REFINERY})
KEY_LEAST = 3
# How many of a ladder's likeliest cells a top-up tries, and how many cells anywhere when none is
# likely at all.
LIKELIEST_TRIED = 20
ANYWHERE_TRIED = 200


def _unlocked(tile_map: TileMap, cells: FlatIndices, weights: FloatArray) -> FloatArray:
    """A land's weights of these flat cells, none on land the player has not bought
    (railroads.sim.lots), where nothing grows."""
    if tile_map.locked is None:
        return weights
    return np.where(tile_map.locked.ravel()[cells], 0.0, weights)


def _allowed(cells: FlatIndices, opened: FlatIndices, land: FloatArray) -> FlatIndices:
    """The cells a ladder may grow in once these have opened to it, in order. Cells that closed
    stay in until they are a tenth of the rest, harmless with no chance to grow there."""
    if opened.size and not cells.size:
        cells = opened
    elif opened.size:
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
    cells: FlatIndices
    land: FloatArray
    chances: FloatArray
    # No cell's chance is above this; a month draws its cells against it.
    highest: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class _Touched:
    """The places whose chances one kind of points changes, laid out to work out all their
    chances in a few cells at once (Growth._reckon_touched), the same sums in the same order as
    Growth._reckon: their rows in the stacked chances and lands, the points they draw and shun
    as rows of the stacked points with their weights (0 for a slot a place doesn't use), the
    served points they breed by with their weights, base chance and what each served point
    adds."""

    places: tuple[_Place, ...]
    rows: npt.NDArray[np.intp]
    lands: npt.NDArray[np.intp]
    served_by: npt.NDArray[np.intp]
    served_weights: FloatArray
    spawn: FloatArray
    per_doubling: FloatArray
    drawn_by: npt.NDArray[np.intp]
    drawn_weights: FloatArray
    shunning: npt.NDArray[np.intp]
    shunned_by: npt.NDArray[np.intp]
    shunned_weights: FloatArray


# The arrays Growth._stack lays out, left out of saves and laid out again as a game loads.
_STACKED = (
    "_chances",
    "_lands",
    "_nearby_all",
    "_served_all",
    "_points_row",
    "_touched",
    "_land_of",
)


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
        self.towns = Towns(grid, tile_map.terrain == Terrain.RIVER)
        self.towns.settle(self.sending())
        # The cells last month's trade was worked out on.
        self._traded = np.zeros(0, dtype=np.intp)
        rows, cols = grid.neighbour_index_grids()
        # The kinds on ladders that follow their own ladder's points, each to its ladder's first.
        self._kind_channel = kind_channels(rule)
        channels = [*Points, *dict.fromkeys(self._kind_channel.values())]
        self._nearby = {
            channel: self._neighbourhood_sum(self.points(channel), rows, cols)
            for channel in channels
        }
        self.economy = Economy(tile_map)
        self._near = self.economy.neighbourhoods.index
        # Each cell's points of every kind counted as served, multiplied by how satisfied.
        self._served = {channel: np.zeros(grid.rows * grid.cols) for channel in self._nearby}
        self._places = []
        everywhere = np.arange(grid.rows * grid.cols)
        # Ladders growing on the same land share its weights, and the cells it allows.
        lands: dict[int, tuple[Land, FloatArray, FlatIndices]] = {}
        surroundings = Surroundings(tile_map, everywhere, self._near)
        for ladder in rule.ladders:
            shared = lands.get(id(ladder.land))
            if shared is None or shared[0] is not ladder.land:
                weights = _unlocked(tile_map, everywhere, ladder.land.weights_in(surroundings))
                shared = lands[id(ladder.land)] = (
                    ladder.land,
                    weights,
                    np.flatnonzero(weights > 0),
                )
            _, land, cells = shared
            self._places.append(_Place(ladder, cells, land, np.empty(0)))
        self._stack()
        self._reckon_everywhere()
        for place in self._places:
            place.highest = float(place.chances.max())
        # The cells whose forest came or went in the last month.
        self.forests_changed: set[Cell] = set()
        self.last_trade: Trade | None = None
        # The share of its needs each cell's buildings get in a simulated game, standing in for
        # trains (railroads.sim.autoplay); NaN where trains decide.
        self.assumed_shares: FloatArray | None = None
        # Trains and their stations, once the player has any (railroads.sim.railway).
        self.railway: Railway | None = None
        # The one-off buildings this game may get (railroads.sim.novelties), rolled once a year.
        self.novelties: Novelties | None = None
        # What history does to this game (railroads.sim.happenings); None while the map grows its
        # history before play.
        self.happenings: Happenings | None = None
        # The buildings the player owns, the stations growth trades with, and the room it makes
        # (railroads.sim.ownership, railroads.sim.station_trade, railroads.sim.room).
        self.owners = Owners(self)
        self.ledger = Ledger()
        self.station_trade = StationTrade(self)
        self.room = Room(self)
        self._months = 0
        # The kinds standing on the map this month, worked out as asked for (_standing).
        self._stood: dict[BuildingKind, bool] = {}
        self._stood_month = -1
        self._land_cells = int((tile_map.terrain == Terrain.LAND).sum())
        if rule.trades:
            self.trade_goods()

    def __getstate__(self) -> dict[str, object]:
        """What a saved game keeps of the growth: everything but each ladder's chances, which
        are worked out afresh from the rest as it loads (__setstate__) and would make over a
        quarter of a save."""
        state = dict(self.__dict__)
        for stacked in _STACKED:
            del state[stacked]
        state["_places"] = [replace(place, chances=np.empty(0)) for place in self._places]
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._stack()
        self._reckon_everywhere()

    def _stack(self) -> None:
        """Lays every ladder's chances, every land's weights and every kind of points nearby and
        served out as rows of one array each, the places and the dicts keeping views of their
        rows, so a building's points can change the chances of every ladder they touch at once
        (_reckon_touched)."""
        size = self._grid.rows * self._grid.cols
        self._chances = np.zeros((len(self._places), size))
        land_rows: dict[int, int] = {}
        lands: list[FloatArray] = []
        for place in self._places:
            if id(place.land) not in land_rows:
                land_rows[id(place.land)] = len(lands)
                lands.append(place.land)
        place_lands = [land_rows[id(place.land)] for place in self._places]
        self._lands = np.stack(lands) if lands else np.zeros((0, size))
        views = list(self._lands)
        for row, place in enumerate(self._places):
            place.land = views[place_lands[row]]
            place.chances = self._chances[row]
        kept: dict[int, tuple[Land, FloatArray]] = {}
        for place in self._places:
            kept.setdefault(id(place.ladder.land), (place.ladder.land, place.land))
        self._land_of = tuple(kept.values())
        kinds = list(self._nearby)
        self._nearby_all = np.stack([self._nearby[points] for points in kinds]).reshape(
            len(kinds), size
        )
        self._served_all = np.stack([self._served[points] for points in kinds])
        self._points_row = {points: at for at, points in enumerate(kinds)}
        for at, points in enumerate(kinds):
            self._nearby[points] = self._nearby_all[at].reshape(self._grid.shape)
            self._served[points] = self._served_all[at]
        breedings = [self._breeding(place) for place in self._places]
        drawn_slots = max((len(b.drawn_by) for b in breedings), default=1)
        served_slots = max((len(self._served_by(b)) for b in breedings), default=1)
        shunned_slots = max((len(b.shunned_by) for b in breedings), default=1) or 1
        self._touched = {}
        for points in kinds:
            picked = [
                row
                for row, b in enumerate(breedings)
                if points in b.drawn_by or points in b.shunned_by
            ]
            drawn_by = np.zeros((len(picked), drawn_slots), dtype=np.intp)
            drawn_weights = np.zeros((len(picked), drawn_slots))
            shunned_by = np.zeros((len(picked), shunned_slots), dtype=np.intp)
            shunned_weights = np.zeros((len(picked), shunned_slots))
            served_by = np.zeros((len(picked), served_slots), dtype=np.intp)
            served_weights = np.zeros((len(picked), served_slots))
            for at, row in enumerate(picked):
                b = breedings[row]
                for slot, (served, weight) in enumerate(self._served_by(b).items()):
                    served_by[at, slot] = kinds.index(served)
                    served_weights[at, slot] = weight
                for slot, (drawn, weight) in enumerate(b.drawn_by.items()):
                    drawn_by[at, slot] = kinds.index(drawn)
                    drawn_weights[at, slot] = weight
                for slot, (shunned, weight) in enumerate(b.shunned_by.items()):
                    shunned_by[at, slot] = kinds.index(shunned)
                    shunned_weights[at, slot] = weight
            self._touched[points] = _Touched(
                places=tuple(self._places[row] for row in picked),
                rows=np.array(picked, dtype=np.intp),
                lands=np.array([place_lands[row] for row in picked], dtype=np.intp),
                served_by=served_by,
                served_weights=served_weights,
                spawn=np.array([[breedings[row].spawn] for row in picked]).reshape(-1, 1),
                per_doubling=np.array([[breedings[row].per_doubling] for row in picked]).reshape(
                    -1, 1
                ),
                drawn_by=drawn_by,
                drawn_weights=drawn_weights,
                shunning=np.array(
                    [at for at, row in enumerate(picked) if breedings[row].shunned_by],
                    dtype=np.intp,
                ),
                shunned_by=shunned_by,
                shunned_weights=shunned_weights,
            )

    def big_parts(self) -> list[object]:
        """What of the growth takes longest to pickle (railroads.sim.game.Game.big_parts)."""
        return [
            self._counts,
            self.economy,
            self.last_trade,
            self._starving,
            [place.land for place in self._places],
            [place.cells for place in self._places],
        ]

    def trade_goods(self) -> set[Cell]:
        """Pulls down farms that towns have crowded out, lets stations hand out a month of goods,
        raises buildings with enough people around them and counts what the served buildings add
        to every ladder's chances; returns the cells whose buildings changed."""
        return finish(self._trade_steps())

    def _trade_steps(self) -> Steps[set[Cell]]:
        """trade_goods in steps: before and as the economy trades, and between the parts of
        settling what it traded."""
        changed = self._crowd_out()
        self.economy.year = self.year
        self.economy.bonus_per_good = BONUS_PER_GOOD
        if self.railway is not None:
            self.railway.station_trade = self.station_trade
        yield
        self.last_trade = yield from self.economy.trade_steps(
            self.counts,
            self.station_trade.orders(),
            np.flatnonzero(self._occupied),
            self.owners.first_for_stations(),
        )
        self.ledger.record(self.last_trade, max(0, self._months - 1))
        yield
        yield from self.station_trade.settle_steps(self.last_trade)
        self.station_trade.update_reach()
        yield
        changed |= self._die_out(self.last_trade)
        changed |= self._raise_buildings()
        yield
        self._count_served(self.last_trade)
        for place, highest in zip(self._places, self._chances.max(axis=1).tolist(), strict=True):
            place.highest = highest
        return changed

    def starving(self, trade: Trade, kind: BuildingKind) -> npt.NDArray[np.int64]:
        """How many buildings of a kind starve in each cell traded on: what arrives is shared
        equally among them, so they starve all together, and only when none of it arrives and,
        for processors, their stockpile is empty too, so one that lacks one of its inputs lives
        on what it holds of the others. In a simulated game, cells the stand-in serves have the
        share it assumes."""
        count = self._counts[kind].ravel()[trade.cells]
        wanted = trade.needed[kind] + trade.could_make[kind]
        share = trade.share((kind,))
        if self.assumed_shares is not None:
            assumed = self.assumed_shares[trade.cells]
            share = np.where(np.isnan(assumed), share, assumed)
        starving = (wanted > 0) & (share <= 0.0) & ~self._stocked(kind, trade.cells)
        return np.where(starving, count, 0)

    def _stocked(self, kind: BuildingKind, cells: npt.NDArray[np.intp]) -> npt.NDArray[np.bool_]:
        """Whether the processors of a kind in each of these cells hold any of their inputs."""
        held = np.zeros(cells.size, dtype=np.bool_)
        for (stocked, _), stock in self.economy.stock.items():
            if stocked == kind:
                held |= stock[cells] > 0
        return held

    def _die_out(self, trade: Trade) -> set[Cell]:
        """Counts the months buildings of each kind have starved in each cell, and pulls down as
        many as starved all through the months they last (dies_after); returns the cells they
        stood in."""
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
            dying = streak >= dies_after(kind)
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
            self._pull_down(cell, kind, gone)
            self.died_out += gone
            changed.add(cell)
        return changed

    def _count_served(self, trade: Trade) -> None:
        """Works out each cell's served points of every kind, and what they add to the chances:
        in the cells traded on, and the cells traded on last month that have no buildings now."""
        served = {channel: np.zeros(trade.cells.size) for channel in self._served}
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
            given = np.maximum(share, 0.0) * value * count
            served[points] += given
            channel = self._kind_channel.get(kind)
            if channel is not None:
                served[channel] += given
        served[Points.POPULATION] *= 1.0 + LIVELY_GROWTH * self._lively(trade)
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
            old[cells] = now[moved]
            for place in self._places:
                if points in self._served_by(self._breeding(place)):
                    self._reckon(place, cells)

    def _lively(self, trade: Trade) -> FloatArray:
        """How lively each traded cell is: the best share of what they need that the venues in
        its neighbourhood got."""
        size = self._grid.rows * self._grid.cols
        poured = np.zeros(size + 1)
        for kind in VENUES:
            if kind in trade.got:
                share = np.minimum(trade.share((kind,), empty=0.0), 1.0)
                here = poured[trade.cells]
                poured[trade.cells] = np.maximum(here, share)
        near = self.economy.neighbourhoods.index[:, trade.cells]
        return np.maximum(poured[trade.cells], poured[near].max(axis=0))

    @property
    def year(self) -> int:
        return self.start_year + self._months // MONTHS_PER_YEAR

    @property
    def months(self) -> int:
        """How many months have passed since start_year began."""
        return self._months

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def counts(self) -> Mapping[BuildingKind, IntArray]:
        """How many grown buildings of each kind stand in each cell."""
        return self._counts

    @property
    def near(self) -> FlatIndices:
        """Every cell's neighbours as flat indices (railroads.sim.reach.Neighbourhoods)."""
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

    def take_down(self, building: Building) -> int:
        """Pulls a building down, the player's too; returns how many shares of it they owned."""
        owned = self.owners.lose(building.cell, building.kind, 1)
        self.constructions.remove(building)
        if building.kind in self._counts:
            self.add(building.cell, building.kind, -1)
        return owned

    def _pull_down(self, cell: Cell, kind: BuildingKind, many: int) -> int:
        """Pulls down this many buildings of a kind in a cell, the player's last; returns how
        many shares of them the player owned."""
        owned = self.owners.lose(cell, kind, many)
        for _ in range(many):
            self.constructions.remove(Building(cell, kind))
        self.add(cell, kind, -many)
        return owned

    def _crowd_out(self) -> set[Cell]:
        """Pulls down the fields towns have grown over: the first kind of every ladder, of the
        game's or this growth's own, that grows only where fewer people live around than its
        crowded_at."""
        changed: set[Cell] = set()
        people = self._nearby[Points.POPULATION]
        crowded_at = {
            ladder.kinds[0]: ladder.crowded_at
            for ladder in (*LADDERS, *self.rule.ladders)
            if ladder.crowded_at is not None
        }
        for kind, most in crowded_at.items():
            fields = self._counts.get(kind)
            if fields is None:
                continue
            for row, col in np.argwhere((fields > 0) & (people >= most)).tolist():
                self._pull_down((row, col), kind, int(fields[row, col]))
                changed.add((row, col))
        return changed

    def _breeding(self, place: _Place) -> Breeding:
        return breeding_of(place.ladder, self.rule.points)

    @staticmethod
    def _served_by(breeding: Breeding) -> Mapping[Channel, float]:
        return breeding.served_by or {breeding.own: 1.0}

    def _served_in(self, breeding: Breeding, cells: npt.NDArray[np.intp] | slice) -> FloatArray:
        """The served points a ladder breeds by in these cells, each times its weight."""
        return sum(
            weight * self._served[channel][cells]
            for channel, weight in self._served_by(breeding).items()
        )

    def _reckon(self, place: _Place, cells: npt.NDArray[np.intp]) -> None:
        """Works out the ladder's chances in these cells afresh."""
        breeding = self._breeding(place)
        drawing = sum(
            weight * self._nearby[points].ravel()[cells]
            for points, weight in breeding.drawn_by.items()
        )
        served = self._served_in(breeding, cells)
        doublings = np.log2(1 + drawing)
        raw = breeding.spawn + (breeding.spawn + breeding.per_doubling * served) * doublings
        if breeding.shunned_by:
            raw = raw / (
                1.0
                + sum(
                    weight * self._nearby[points].ravel()[cells]
                    for points, weight in breeding.shunned_by.items()
                )
            )
        place.chances[cells] = raw * place.land[cells]

    def _reckon_everywhere(self) -> None:
        """Works out every ladder's chances in every cell afresh, as _reckon does, the doublings
        of what draws them once for all the ladders drawn alike."""
        doublings: dict[tuple[tuple[Channel, float], ...], FloatArray] = {}
        for place in self._places:
            breeding = self._breeding(place)
            drawn_by = tuple(breeding.drawn_by.items())
            if drawn_by not in doublings:
                drawing = sum(weight * self._nearby[points].ravel() for points, weight in drawn_by)
                doublings[drawn_by] = np.log2(1 + drawing)
            served = self._served_in(breeding, slice(None))
            raw = (
                breeding.spawn
                + (breeding.spawn + breeding.per_doubling * served) * doublings[drawn_by]
            )
            if breeding.shunned_by:
                raw = raw / (
                    1.0
                    + sum(
                        weight * self._nearby[points].ravel()
                        for points, weight in breeding.shunned_by.items()
                    )
                )
            place.chances[:] = raw * place.land

    def land(self, land: Land, cells: npt.NDArray[np.intp]) -> FloatArray:
        """The weights of these flat cells for a land (railroads.sim.surroundings.Land.weights), as
        kept up to date for the ladders growing on it."""
        for kept, weights in self._land_of:
            if kept is land:
                return weights[cells]
        return land.weights(self.constructions.tile_map, cells, self._near)

    def satisfaction(self) -> npt.NDArray[np.int8]:
        """How satisfied each cell's buildings are on the whole, or -1 where none trade."""
        if self.last_trade is None:
            return np.full(self._grid.shape, -1, dtype=np.int8)
        levels = np.full(self._grid.rows * self._grid.cols, -1, dtype=np.int8)
        everything = tuple(self.last_trade.got)
        trading = self.last_trade.wanted(everything) > 0
        levels[self.last_trade.cells[trading]] = self.last_trade.levels()[trading]
        if self.assumed_shares is not None:
            cells = self.last_trade.cells[trading]
            assumed = self.assumed_shares[cells]
            given = ~np.isnan(assumed)
            levels[cells[given]] = levels_of(assumed[given])
        return levels.reshape(self._grid.shape)

    def stars(self, kind: BuildingKind) -> int:
        """How high a kind stands on its ladder: 1 for the first kind, 2 for the next..."""
        if kind == BuildingKind.SKYSCRAPER:
            return SKYSCRAPER_STARS
        ladder = ladder_of(kind, self.rule.ladders)
        return 1 if ladder is None else ladder.kinds.index(kind) + 1

    def build(self, building: Building) -> None:
        """Puts up a building no ladder grew, counted like any other."""
        self.constructions.build(building)
        self.add(building.cell, building.kind, 1)

    def count(self, kind: BuildingKind) -> IntArray:
        """How many grown buildings of a kind stand in each cell."""
        return self._counts[kind]

    def points(self, points: Channel = Points.POPULATION) -> IntArray:
        """Each cell's points of one kind, or the points the buildings of one ladder give."""
        total = np.zeros(self._grid.shape, dtype=np.int64)
        for kind, count in self._counts.items():
            given, value = self.rule.points[kind]
            if given == points or self._kind_channel.get(kind) == points:
                total += value * count
        return total

    def sending(self) -> FloatArray:
        """The passengers each cell's buildings send a month."""
        total = np.zeros(self._grid.shape, dtype=np.float64)
        for kind, count in self._counts.items():
            given, value = self.rule.points[kind]
            per = PASSENGERS_PER_POINT.get(given, 0) * value
            if per:
                total += per * count
        return total

    def population(self) -> int:
        """The population points of the whole map."""
        return sum(
            value * int(self._counts[kind].sum())
            for kind, (given, value) in self.rule.points.items()
            if given == Points.POPULATION
        )

    def nearby(self, points: Channel = Points.POPULATION) -> IntArray:
        """The points of one kind, or of one ladder, in each cell's neighbourhood."""
        return self._nearby[points]

    def chances(self, ladder: Ladder = TOWNS) -> FloatArray:
        """Each cell's chance of a turn on the ladder this month."""
        place = next(place for place in self._places if place.ladder == ladder)
        cells = np.arange(place.chances.size)
        return np.where(self._meets_needs(place, cells), place.chances, 0.0).reshape(
            self._grid.shape
        )

    def month(self, rng: np.random.Generator) -> set[Cell]:
        """Grows one month; returns the cells whose buildings changed."""
        return finish(self.month_steps(rng))

    def month_steps(self, rng: np.random.Generator, months: int = 1) -> Steps[set[Cell]]:
        """Grows one month, or this many months at once, in a few steps, which a caller can
        spread over frames; nothing else may change the game between them. Over many months each
        ladder's chances count that many times over, and a cell may take several turns. Returns
        the cells whose buildings changed."""
        before = self._months
        self._months += months

        def reached(every: int) -> bool:
            return self._months // every > before // every

        stations = 0 if self.railway is None else len(self.railway.stations())
        year = MONTHS_PER_YEAR
        for month in range((before // year + 1) * year, self._months + 1, year):
            self.history.close_year(
                self.start_year + month // year - 1, stations, self.population()
            )
            if self.railway is not None:
                self.railway.new_year(self.start_year + month // year)
        changed: set[Cell] = set()
        self.forests_changed = set()
        if self.rule.trades and reached(self.rule.trade_every_months):
            changed = yield from self._trade_steps()
            yield
        if self.novelties is not None and reached(MONTHS_PER_YEAR):
            changed |= self.novelties.year(self, rng)
        if reached(MONTHS_PER_YEAR):
            changed |= build_power_plants(self)
            changed |= self._top_up(rng)
        if self.happenings is not None:
            changed |= self.happenings.months(self, before + 1, self._months, rng)
        every = self.rule.forests_every_months
        if self.rule.forests is not None and reached(every):
            changed |= change_forests(self, self.rule.forests, every, rng)
        turns: list[tuple[int, Ladder]] = []
        growing = [
            place
            for place in self._places
            if ladder_open(place.ladder, self.year) and place.cells.size and place.highest > 0
        ]
        thin = [place for place in growing if place.highest <= THINNING_UP_TO and months == 1]
        tries = dict(
            zip(
                map(id, thin),
                rng.binomial([p.cells.size for p in thin], [p.highest for p in thin]).tolist(),
                strict=True,
            )
        )
        war = military_growth(self.year)
        for place in growing:
            if war > 1 and place.ladder is MILITARY:
                hits = self._hits(place, rng, None, months * war)
                turns += [(index, place.ladder) for index in hits.tolist()]
                continue
            drawn = tries.get(id(place))
            if drawn == 0:
                continue
            hits = self._hits(place, rng, drawn, months)
            turns += [(index, place.ladder) for index in hits.tolist()]
        for turn in rng.permutation(len(turns)).tolist():
            index, ladder = turns[turn]
            changed |= self._turn(divmod(index, self._grid.cols), ladder, rng)
        return changed

    def reweigh_land(self, changed_cells: FlatIndices) -> None:
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
        surroundings: dict[bool, Surroundings] = {}
        for places in sharing.values():
            first = places[0]
            nearby = first.ladder.land.on_forest_nearby
            touched = beside if nearby else own
            if nearby not in surroundings:
                surroundings[nearby] = Surroundings(tile_map, touched, self._near)
            land = _unlocked(tile_map, touched, first.ladder.land.weights_in(surroundings[nearby]))
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

    def reweigh_deposits(self, cells: FlatIndices) -> None:
        """Weighs the land again in these flat cells, sorted, for the ladders that grow only over a
        deposit, once the deposits there changed (railroads.sim.happenings)."""
        tile_map = self.constructions.tile_map
        surroundings = Surroundings(tile_map, cells, self._near)
        weighed: set[int] = set()
        for place in self._places:
            land = place.ladder.land
            if land.deposit is None:
                continue
            if id(place.land) not in weighed:
                weighed.add(id(place.land))
                place.land[cells] = _unlocked(tile_map, cells, land.weights_in(surroundings))
            place.cells = _allowed(place.cells, cells[place.land[cells] > 0], place.land)
            self._reckon(place, cells)
            place.highest = max(place.highest, float(place.chances[cells].max()))

    def run(self, months: int, rng: np.random.Generator, step: int = 1) -> None:
        """Grows these many months, `step` months at a time and what is left at the end."""
        for _ in range(months // step):
            finish(self.month_steps(rng, step))
        if months % step:
            finish(self.month_steps(rng, months % step))

    def _hits(
        self,
        place: _Place,
        rng: np.random.Generator,
        tries: int | None = None,
        months: int = 1,
    ) -> npt.NDArray[np.intp]:
        """The cells that take a turn on the ladder this month. With small chances, cells are
        drawn as if all had the highest chance and each kept by how its own compares, so a month
        costs what grows rather than the size of the map: `tries` of them, if already drawn
        for every ladder at once. Over many months, a cell is drawn as many times as it takes
        turns, on average its chance times the months."""
        cells = place.cells
        if not cells.size:
            return cells
        highest = place.highest
        if highest <= 0:
            return cells[:0]
        if months > 1:
            tried = cells[rng.integers(0, cells.size, rng.poisson(cells.size * highest * months))]
            drawn = tried[rng.random(tried.size) * highest < place.chances[tried]]
        elif highest > THINNING_UP_TO:
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
        if ladder.joins is not None:
            joined = self._join(cell, neighbourhood, ladder.kinds[0], ladder.joins)
            if joined is not None:
                return joined
        if (
            ladder.land_cells_each is not None
            and self._as_many_as_first(ladder) >= self._land_cells // ladder.land_cells_each
        ):
            return set()
        if not self.ladder_allows(ladder, cell) or self._not_open(ladder, cell, rng):
            return set()
        if not self._inputs_on_map(ladder):
            return set()
        return self.room.scatter(cell, ladder, rng)

    def _join(
        self, cell: Cell, neighbourhood: list[Cell], kind: BuildingKind, join: Join
    ) -> set[Cell] | None:
        """Pulls down join.needed of a kind in the neighbourhood and puts up what they join into
        in the cell, if the neighbourhood is busy enough and none of that stands near; None, and
        nothing changed, otherwise."""
        if self._around(kind, neighbourhood) < join.needed:
            return None
        built = self._counts[join.into]
        if any(built[near] for near in self._grid.within(cell, join.apart)):
            return None
        tile_map = self.constructions.tile_map
        land = sum(
            tile_map.terrain[near] == Terrain.LAND and tile_map.relief[near] != Relief.MOUNTAINS
            for near in neighbourhood
        )
        if not land or self._nearby[Points.POPULATION][cell] < join.people_per_cell * land:
            return None
        return self._replace(cell, neighbourhood, [(kind, join.needed)], join.into)

    def least_of(self, ladder: Ladder) -> int:
        """The fewest of a ladder's first kind the map keeps once it has opened: KEY_LEAST for
        the land of a key building, one of anything that opens in some year, else none."""
        kind = ladder.kinds[0]
        if kind in KEY_BUILDINGS:
            return max(1, round(KEY_LEAST * self._land_cells / REFERENCE_LAND))
        opens_later = ladder.from_year > 0 or kind_era(kind) > 1
        return 1 if opens_later and kind not in ONE_OFFS else 0

    def _top_up(self, rng: np.random.Generator) -> set[Cell]:
        """Puts up one more of the first kind of every opened ladder below its least, in its
        likeliest cell; returns the cells that changed."""
        changed: set[Cell] = set()
        for place in self._places:
            ladder = place.ladder
            if not ladder_open(ladder, self.year):
                continue
            if self.rule.top_ups is not None and ladder.kinds[0] not in self.rule.top_ups:
                continue
            if self._as_many_as_first(ladder) >= self.least_of(ladder):
                continue
            if not self._inputs_on_map(ladder):
                continue
            cell = self._likeliest(place, rng)
            if cell is None:
                continue
            kind = ladder.kinds[0]
            self.constructions.build(Building(cell, kind))
            self.add(cell, kind, 1)
            changed.add(cell)
        return changed

    def _likeliest(self, place: _Place, rng: np.random.Generator) -> Cell | None:
        """The cell a ladder's next building is likeliest in that it can go up in; for a key
        building where none is likely, any cell it could stand on, picked at random."""
        ladder = place.ladder
        cells = place.cells[place.chances[place.cells] > 0]
        if cells.size and (ladder.needs or ladder.crowded_at is not None):
            cells = cells[self._meets_needs(place, cells)]
        best = cells[np.argsort(place.chances[cells])[::-1][:LIKELIEST_TRIED]]
        for index in best.tolist():
            cell = divmod(index, self._grid.cols)
            if self._fits(ladder, cell):
                return cell
        if ladder.kinds[0] not in KEY_BUILDINGS:
            return None
        anywhere = place.cells if place.cells.size else self._open_cells()
        for index in rng.choice(anywhere, min(ANYWHERE_TRIED, anywhere.size), replace=False):
            cell = divmod(int(index), self._grid.cols)
            if self.constructions.tile_map.terrain[cell] == Terrain.LAND and self._fits(
                ladder, cell
            ):
                return cell
        return None

    def _open_cells(self) -> FlatIndices:
        """Every cell not locked (railroads.sim.lots)."""
        locked = self.constructions.tile_map.locked
        if locked is None:
            return np.arange(self._grid.rows * self._grid.cols)
        return np.flatnonzero(~locked)

    def _fits(self, ladder: Ladder, cell: Cell) -> bool:
        """Whether one more of a ladder's first kind can go up in a cell as it stands."""
        new = Building(cell, ladder.kinds[0])
        return (
            self.ladder_allows(ladder, cell)
            and self.constructions.site_problem(new) is None
            and new.space <= self.constructions.space_left(cell)
        )

    def _as_many_as_first(self, ladder: Ladder) -> int:
        """How many of a ladder's first kind its buildings on the map stand for, counting each
        merged one as all those merged into it."""
        stands_for = 1
        total = 0
        for kind in ladder.kinds:
            total += stands_for * int(self._counts[kind].sum())
            merge = ladder.merges.get(kind)
            stands_for *= merge.pulled_down if merge is not None else 1
        return total

    def _inputs_on_map(self, ladder: Ladder) -> bool:
        """Whether something on the map makes each main input of the ladder's first kind
        (inputs_of)."""
        inputs = inputs_of(ladder.kinds[0], self.year)
        return all(any(self._standing(maker) for maker in makers) for makers in inputs)

    def _standing(self, kind: BuildingKind) -> bool:
        """Whether any building of a kind stands on the map, as it was when first asked this
        month."""
        if self._stood_month != self._months:
            self._stood, self._stood_month = {}, self._months
        stood = self._stood.get(kind)
        if stood is None:
            counts = self._counts.get(kind)
            stood = self._stood[kind] = counts is not None and bool(counts.any())
        return stood

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
        nothing pulled down, if there are too few or it would not fit. A big or mega works moves
        the cell's other buildings out of its way (railroads.sim.room) if it must."""
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
        if new_kind in GREW_FROM:
            freed = sum(building.space for building in self.constructions.buildings_at(cell))
        else:
            freed = sum(
                take * Building(cell, kind).space for near, kind, take in plan if near == cell
            )
        if new.space > self.constructions.space_left(cell) + freed:
            return None
        owned = 0
        for near, kind, take in plan:
            owned += self._pull_down(near, kind, take)
        moved = self.room.clear(cell, new.space)
        self.constructions.build(new)
        self.add(cell, new_kind, 1)
        if owned:
            taken = sum(take for _, _, take in plan)
            self.owners.own(cell, new_kind, round(owned / max(taken, 1)))
        return {cell} | {near for near, _, _ in plan} | moved

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
        sends = PASSENGERS_PER_POINT.get(points, 0) * value * how_many
        if sends:
            self.towns.add(cell, float(sends))
        here = cell[0] * self._grid.cols + cell[1]
        around = self._near[:, here]
        flat = np.concatenate(([here], around[around < self._near.shape[1]]))
        self._nearby_all[self._points_row[points], flat] += how_many * value
        channel = self._kind_channel.get(kind)
        if channel is not None:
            self._nearby_all[self._points_row[channel], flat] += how_many * value
        for changed in (points,) if channel is None else (points, channel):
            touched = self._touched[changed]
            if not touched.places:
                continue
            chances = self._reckon_touched(touched, flat)
            if how_many > 0:
                for place, most in zip(touched.places, chances.max(axis=1).tolist(), strict=True):
                    place.highest = max(place.highest, most)

    def _reckon_touched(self, touched: _Touched, cells: npt.NDArray[np.intp]) -> FloatArray:
        """Works out the chances of all the touched places in these cells afresh, as _reckon
        does each, and returns them, a row for each place."""
        nearby = self._nearby_all[:, cells]
        weights = touched.drawn_weights
        drawing = weights[:, :1] * nearby[touched.drawn_by[:, 0]]
        for slot in range(1, weights.shape[1]):
            drawing = drawing + weights[:, slot : slot + 1] * nearby[touched.drawn_by[:, slot]]
        served_weights = touched.served_weights
        served = served_weights[:, :1] * self._served_all[touched.served_by[:, :1], cells]
        for slot in range(1, served_weights.shape[1]):
            served = (
                served
                + served_weights[:, slot : slot + 1]
                * self._served_all[touched.served_by[:, slot : slot + 1], cells]
            )
        doublings = np.log2(1 + drawing)
        raw = touched.spawn + (touched.spawn + touched.per_doubling * served) * doublings
        if touched.shunning.size:
            at = touched.shunning
            weights = touched.shunned_weights[at]
            shunned_by = touched.shunned_by[at]
            shunned = weights[:, :1] * nearby[shunned_by[:, 0]]
            for slot in range(1, weights.shape[1]):
                shunned = shunned + weights[:, slot : slot + 1] * nearby[shunned_by[:, slot]]
            raw[at] = raw[at] / (1.0 + shunned)
        chances = raw * self._lands[touched.lands[:, None], cells]
        self._chances[touched.rows[:, None], cells] = chances
        return chances

    @staticmethod
    def _neighbourhood_sum(values: IntArray, rows: IntArray, cols: IntArray) -> IntArray:
        own_rows, own_cols = np.indices(values.shape)
        off_map = (rows == own_rows) & (cols == own_cols)
        return values + np.where(off_map, 0, values[rows, cols]).sum(axis=0)
