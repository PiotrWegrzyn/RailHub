"""The map grows by itself, a month at a time, like a cellular automaton: towns, farms, orchards,
mines, lumber mills, ports and several kinds of factory.

Every building that grows gives points of one kind (POINTS): houses give population, farms and
orchards food, mines ore, lumber mills timber, ports trade, factories industry and skyscrapers
corporate points. Each cell's neighbourhood (the cell and the cells next to it) adds up the
points of every kind around it.

What grows comes in ladders (LADDERS). Every month each cell may take a turn on each ladder, by
chance: the ladder's base chance plus its chance per nearby point of each kind, but nothing until
the neighbourhood holds the points the ladder needs. The land multiplies the chance (Land): its
vegetation, relief and temperature, and the best water next to the cell; a weight of 0 means
the ladder never grows there.

On its turn a ladder puts up its first kind of building in the cell, unless it can merge: when
the neighbourhood has enough of one kind on the ladder (Merge), some of them are pulled down,
the cell's own first, and one of the next kind goes up in the cell instead, the highest merge
that fits first. Ladders can cap how many of their first kind stand in one cell or one
neighbourhood, so farms and ports keep their distance.

On town and factory turns, a neighbourhood with SKYSCRAPER_PEOPLE population and
SKYSCRAPER_INDUSTRY industry points puts up a skyscraper: it pulls down three blocks of flats and
three points of factories (three factories, or else a big factory) and gives a corporate point.

Every TRADE_EVERY_MONTHS the buildings share out their goods (railroads.sim.economy), which
leaves each kind unsatisfied, content or satisfied. A ladder's chances in a cell are scaled by
how its kinds of building fare across the cell's neighbourhood: x0.5 when unsatisfied, x1 when
content, x1.5 when satisfied, and x1 where there are none yet.

Forests come and go as well (railroads.sim.forests): people, industry and above all lumber mills
clear them, they spread onto open grassland beside them, and the land weighs again wherever they
changed.

A new map runs this for MONTHS_BEFORE_PLAY before play starts; in play it carries on, one month
every few minutes.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import CELL_SPACE, Building, BuildingKind, Constructions
from railroads.sim.economy import (
    Economy,
    Neighbourhoods,
    StationOrder,
    Trade,
    free_food,
    growth_factors,
    levels_of,
)
from railroads.sim.forests import FORESTS, ForestRule, forest_turn, forests_nearby
from railroads.sim.grids import Cell
from railroads.sim.railway import STOCKPILE, Railway
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

IntGrid = npt.NDArray[np.int64]
FloatGrid = npt.NDArray[np.float64]
Cells = npt.NDArray[np.intp]


class Points(Enum):
    POPULATION = "population"
    FOOD = "food"
    ORE = "ore"
    TIMBER = "timber"
    TRADE = "trade"
    INDUSTRY = "industry"
    CORPORATE = "corporate"


MONTHS_BEFORE_PLAY = 300 * 12
# Farms with more population points than this around them are pulled down.
FARMS_CROWDED_OUT = 10
TRADE_EVERY_MONTHS = 1
SKYSCRAPER_PEOPLE = 100
SKYSCRAPER_INDUSTRY = 10
SKYSCRAPER_FLATS = 3
# What a skyscraper may pull down for its three points of factories, in order of preference.
SKYSCRAPER_FACTORIES = ((BuildingKind.FACTORY, 3), (BuildingKind.BIG_FACTORY, 1))
# Above this chance, a ladder draws for every cell instead of thinning.
THINNING_UP_TO = 0.05

POINTS: Mapping[BuildingKind, tuple[Points, int]] = MappingProxyType(
    {
        BuildingKind.HOUSE: (Points.POPULATION, 1),
        BuildingKind.TENEMENT: (Points.POPULATION, 2),
        BuildingKind.FLATS: (Points.POPULATION, 5),
        BuildingKind.FARM: (Points.FOOD, 1),
        BuildingKind.ORCHARD: (Points.FOOD, 1),
        BuildingKind.MINE: (Points.ORE, 1),
        BuildingKind.LUMBER_MILL: (Points.TIMBER, 1),
        BuildingKind.PORT: (Points.TRADE, 1),
        BuildingKind.FACTORY: (Points.INDUSTRY, 1),
        BuildingKind.BIG_FACTORY: (Points.INDUSTRY, 3),
        BuildingKind.MEGA_FACTORY: (Points.INDUSTRY, 9),
        BuildingKind.FOOD_PLANT: (Points.INDUSTRY, 2),
        BuildingKind.STEEL_MILL: (Points.INDUSTRY, 3),
        BuildingKind.SHIPYARD: (Points.INDUSTRY, 3),
        BuildingKind.SKYSCRAPER: (Points.CORPORATE, 1),
    }
)


@dataclass(frozen=True, slots=True)
class Land:
    """How the land changes a ladder's chances: each weight multiplies them. Water weights go by
    the best water next to the cell, LAND standing for none; anything left out weighs 1."""

    vegetation: Mapping[Vegetation, float] = field(default_factory=dict)
    relief: Mapping[Relief, float] = field(default_factory=dict)
    temperature: Mapping[Temperature, float] = field(default_factory=dict)
    water: Mapping[Terrain, float] = field(default_factory=dict)
    # Weight for a cell with mountains next to it.
    beside_mountains: float = 1.0
    # Only where at least one cell next to it is flat open land: not forest, not water.
    forest_edge: bool = False
    # Only where the neighbourhood holds at least this many forest cells.
    least_forest_nearby: int = 0

    def weights(
        self,
        tile_map: TileMap,
        cells: npt.NDArray[np.intp] | None = None,
        near: Cells | None = None,
    ) -> FloatGrid:
        """The weight of every cell as a grid, or of the given flat cells alone. `near` holds
        every cell's neighbours as flat indices, one past the last cell for off the map."""
        size = tile_map.grid.rows * tile_map.grid.cols
        if near is None:
            near = Neighbourhoods(tile_map.grid).index
        chosen = np.arange(size) if cells is None else cells
        terrain = tile_map.terrain.ravel()
        vegetation = tile_map.vegetation.ravel()
        relief = tile_map.relief.ravel()
        around = near[:, chosen]
        real = around < size
        around = np.where(real, around, 0)
        vegetation_weights = np.array([self.vegetation.get(v, 1.0) for v in Vegetation])
        relief_weights = np.array([self.relief.get(r, 1.0) for r in Relief])
        temperature_weights = np.array([self.temperature.get(t, 1.0) for t in Temperature])
        water = np.array([self.water.get(t, 1.0) for t in Terrain])
        terrain_around = terrain[around]
        is_water = real & (terrain_around != Terrain.LAND)
        best = np.where(is_water, water[terrain_around], 0.0).max(axis=0)
        by_water = np.where(is_water.any(axis=0), best, water[Terrain.LAND])
        weight = (
            vegetation_weights[vegetation[chosen]]
            * relief_weights[relief[chosen]]
            * temperature_weights[tile_map.temperature.ravel()[chosen]]
            * by_water
        )
        beside = real & (relief[around] == Relief.MOUNTAINS)
        weight = weight * np.where(beside.any(axis=0), self.beside_mountains, 1.0)
        forest_around = real & (vegetation[around] == Vegetation.FOREST) & ~is_water
        if self.forest_edge:
            open_flat = real & ~is_water & ~forest_around & (relief[around] == Relief.FLAT)
            weight = weight * open_flat.any(axis=0)
        if self.least_forest_nearby:
            forests = forest_around.sum(axis=0) + (vegetation[chosen] == Vegetation.FOREST)
            weight = weight * (forests >= self.least_forest_nearby)
        weight = np.where(terrain[chosen] == Terrain.LAND, weight, 0.0)
        return weight.reshape(tile_map.grid.shape) if cells is None else weight


@dataclass(frozen=True, slots=True)
class Merge:
    """How many of a kind a neighbourhood needs before they merge into the next kind, and how
    many of them the merge pulls down."""

    needed: int
    pulled_down: int


@dataclass(frozen=True, slots=True)
class Ladder:
    """A line of buildings that grows: a turn puts up the first kind, and merges climb it."""

    kinds: tuple[BuildingKind, ...]
    land: Land
    base_chance: float = 0.0
    chance_per_point: Mapping[Points, float] = field(default_factory=dict)
    # Points the neighbourhood must hold before the ladder has any chance there.
    needs: Mapping[Points, int] = field(default_factory=dict)
    merges: Mapping[BuildingKind, Merge] = field(default_factory=dict)
    # The most of the first kind one cell, and one neighbourhood, may hold.
    most_per_cell: int = CELL_SPACE
    most_nearby: int | None = None
    skyscrapers: bool = False

    @property
    def name(self) -> str:
        return self.kinds[0].value


# Built land: towns and the factories that follow them.
# Food decides where houses do well (railroads.sim.economy), so water weighs nothing more here;
# few people clear a forest to live in it.
TOWN_LAND = Land(
    vegetation={Vegetation.BARREN: 0.5, Vegetation.FOREST: 0.3},
    relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.05},
    temperature={Temperature.COLD: 0.6, Temperature.HOT: 0.8},
)
FACTORY_LAND = Land(
    vegetation={Vegetation.FOREST: 0.2},
    relief={Relief.HILLS: 0.5, Relief.MOUNTAINS: 0.0},
    temperature={Temperature.COLD: 0.7},
    water={Terrain.RIVER: 1.5, Terrain.SEA: 1.3, Terrain.LAKE: 1.2},
)

TOWNS = Ladder(
    kinds=(BuildingKind.HOUSE, BuildingKind.TENEMENT, BuildingKind.FLATS),
    land=TOWN_LAND,
    base_chance=0.0000085,
    chance_per_point={Points.POPULATION: 0.0002125},
    merges={
        BuildingKind.HOUSE: Merge(needed=6, pulled_down=3),
        BuildingKind.TENEMENT: Merge(needed=6, pulled_down=3),
    },
    skyscrapers=True,
)
# Farms want open, flat, mild grassland by water, towns nearby to feed and other farms and orchards
# around them; several can share a cell. A town that grows over them (FARMS_CROWDED_OUT) pulls
# them down.
FARMS = Ladder(
    kinds=(BuildingKind.FARM,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.1, Vegetation.FOREST: 0.02},
        relief={Relief.HILLS: 0.5, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.2, Temperature.HOT: 0.6},
        water={Terrain.RIVER: 5.0, Terrain.LAKE: 5.0, Terrain.SEA: 2.0},
    ),
    base_chance=0.000008,
    chance_per_point={Points.POPULATION: 0.00006, Points.FOOD: 0.00004},
)
# Orchards like sunny slopes and water nearby, and stay off the cold.
ORCHARDS = Ladder(
    kinds=(BuildingKind.ORCHARD,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.2, Vegetation.FOREST: 0.1},
        relief={Relief.FLAT: 0.5, Relief.HILLS: 1.0, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.05, Temperature.HOT: 1.2},
        water={Terrain.RIVER: 3.0, Terrain.LAKE: 3.0, Terrain.SEA: 2.0},
    ),
    base_chance=0.000002,
    chance_per_point={Points.POPULATION: 0.000004, Points.FOOD: 0.00002},
)
# Mines are dug in and beside mountains and in hills, where finding ore draws more digging nearby.
MINES = Ladder(
    kinds=(BuildingKind.MINE,),
    land=Land(
        vegetation={Vegetation.FOREST: 0.6, Vegetation.GRASS: 0.8},
        relief={Relief.FLAT: 0.01, Relief.HILLS: 0.4, Relief.MOUNTAINS: 1.0},
        temperature={Temperature.COLD: 0.8},
        beside_mountains=3.0,
    ),
    base_chance=0.000008,
    chance_per_point={Points.ORE: 0.0001},
    most_per_cell=2,
    most_nearby=5,
)
# Lumber mills stand in forests with at least four forest cells around them and flat open land
# beside them, best by a river to float the logs and near towns to buy them. They clear the forest
# around them (railroads.sim.forests) and close once it is gone.
LUMBER_MILLS = Ladder(
    kinds=(BuildingKind.LUMBER_MILL,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.0, Vegetation.GRASS: 0.0},
        relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.HOT: 0.6},
        water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.2, Terrain.SEA: 1.0},
        forest_edge=True,
        least_forest_nearby=4,
    ),
    base_chance=0.00008,
    chance_per_point={Points.POPULATION: 0.0002},
    most_per_cell=1,
    most_nearby=2,
)
# Ports open on the sea, or less often a lake, where a town has grown to use them.
PORTS = Ladder(
    kinds=(BuildingKind.PORT,),
    land=Land(
        relief={Relief.HILLS: 0.5, Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.0, Terrain.RIVER: 0.0, Terrain.SEA: 1.0, Terrain.LAKE: 0.3},
    ),
    base_chance=0.000004,
    chance_per_point={Points.POPULATION: 0.00004},
    needs={Points.POPULATION: 3},
    most_per_cell=1,
    most_nearby=1,
)
# Factories make goods for the people around them, from lumber mills nearby, and draw more
# factories.
FACTORIES = Ladder(
    kinds=(BuildingKind.FACTORY, BuildingKind.BIG_FACTORY, BuildingKind.MEGA_FACTORY),
    land=FACTORY_LAND,
    chance_per_point={Points.POPULATION: 0.00001, Points.INDUSTRY: 0.00005, Points.TIMBER: 0.0001},
    merges={
        BuildingKind.FACTORY: Merge(needed=3, pulled_down=3),
        BuildingKind.BIG_FACTORY: Merge(needed=3, pulled_down=3),
    },
    skyscrapers=True,
)
# Food plants turn the food of nearby farms and orchards into goods for a town.
FOOD_PLANTS = Ladder(
    kinds=(BuildingKind.FOOD_PLANT,),
    land=FACTORY_LAND,
    chance_per_point={Points.FOOD: 0.0002, Points.POPULATION: 0.00001},
    needs={Points.FOOD: 1, Points.POPULATION: 3},
    most_per_cell=1,
    most_nearby=2,
)
# Steel mills need ore from mines nearby and workers, and grow among other industry.
STEEL_MILLS = Ladder(
    kinds=(BuildingKind.STEEL_MILL,),
    land=Land(
        vegetation={Vegetation.FOREST: 0.2},
        relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
        water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.2, Terrain.SEA: 1.2},
    ),
    chance_per_point={Points.ORE: 0.0002, Points.INDUSTRY: 0.00002},
    needs={Points.ORE: 1, Points.POPULATION: 2},
    most_per_cell=1,
    most_nearby=1,
)
# Shipyards build on the water beside a port, among industry that feeds them.
SHIPYARDS = Ladder(
    kinds=(BuildingKind.SHIPYARD,),
    land=Land(
        relief={Relief.HILLS: 0.3, Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.0, Terrain.SEA: 1.0, Terrain.LAKE: 0.4, Terrain.RIVER: 0.3},
    ),
    chance_per_point={Points.TRADE: 0.0001, Points.INDUSTRY: 0.00002},
    needs={Points.TRADE: 1},
    most_per_cell=1,
    most_nearby=1,
)
LADDERS = (
    TOWNS,
    FARMS,
    ORCHARDS,
    MINES,
    LUMBER_MILLS,
    PORTS,
    FACTORIES,
    FOOD_PLANTS,
    STEEL_MILLS,
    SHIPYARDS,
)


@dataclass(frozen=True, slots=True)
class GrowthRule:
    # Whether goods are shared out and satisfaction scales the chances, and how often.
    trades: bool = True
    trade_every_months: int = TRADE_EVERY_MONTHS
    ladders: tuple[Ladder, ...] = LADDERS
    points: Mapping[BuildingKind, tuple[Points, int]] = field(default=POINTS)
    skyscraper_people: int = SKYSCRAPER_PEOPLE
    skyscraper_industry: int = SKYSCRAPER_INDUSTRY
    # How forests are cleared and spread, if they change at all, and how often.
    forests: ForestRule | None = FORESTS
    forests_every_months: int = 1


GROWTH = GrowthRule()


@dataclass(slots=True)
class _Place:
    """Where a ladder can grow: its cells as flat indices, every cell's weight for its land
    alone, its satisfaction factor and the two together, and every cell's chance before the
    ladder's needs, kept up to date as buildings and forests come and go."""

    ladder: Ladder
    cells: Cells
    land: FloatGrid
    factors: FloatGrid
    weights: FloatGrid
    chances: FloatGrid


class Growth:
    """The grown buildings on one map, counted per cell, and the rule that grows them."""

    def __init__(self, constructions: Constructions, rule: GrowthRule = GROWTH) -> None:
        self.constructions = constructions
        self.rule = rule
        tile_map = constructions.tile_map
        grid = tile_map.grid
        self._grid = grid
        self._counts = {kind: np.zeros(grid.shape, dtype=np.int64) for kind in rule.points}
        for built in constructions:
            if isinstance(built, Building) and built.kind in self._counts:
                self._counts[built.kind][built.cell] += 1
        rows, cols = grid.neighbour_index_grids()
        self._nearby = {
            points: self._neighbourhood_sum(self.points(points), rows, cols) for points in Points
        }
        self.economy = Economy(tile_map)
        self._near = self.economy.neighbourhoods.index
        self._places = []
        everywhere = np.arange(grid.rows * grid.cols)
        for ladder in rule.ladders:
            land = ladder.land.weights(tile_map, everywhere, self._near)
            chances = np.full(land.size, ladder.base_chance)
            for points, per_point in ladder.chance_per_point.items():
                chances += per_point * self._nearby[points].ravel()
            self._places.append(
                _Place(
                    ladder,
                    np.flatnonzero(land > 0),
                    land,
                    np.ones(land.size),
                    land.copy(),
                    chances * land,
                )
            )
        # The cells whose forest came or went in the last month.
        self.forests_changed: set[Cell] = set()
        self.trade: Trade | None = None
        # Trains and their stations, once the player has any (railroads.sim.railway).
        self.railway: Railway | None = None
        self._months = 0
        if rule.trades:
            self.trade_goods()

    def trade_goods(self) -> set[Cell]:
        """Pulls down farms that towns have crowded out, shares out a month of goods, and scales
        every ladder's chances by how satisfied its buildings are around each cell; returns the
        cells whose buildings changed."""
        changed = self._crowd_out_farms()
        last = self.economy.active.copy()
        self.trade = self.economy.trade(self._counts, self._station_orders())
        self._settle_stations(self.trade)
        cells = self.trade.cells
        stale = np.flatnonzero(last & ~self.economy.active)
        neighbourhoods = self.economy.neighbourhoods
        for place in self._places:
            self._reweigh(place, stale, np.ones(stale.size))
            kinds = place.ladder.kinds
            wanted = self.trade.wanted(kinds)
            if not wanted.any():
                self._reweigh(place, cells, np.ones(cells.size))
                continue
            done = neighbourhoods.including(self.trade.done(kinds))
            wanted = neighbourhoods.including(wanted)
            share = done / np.where(wanted > 0, wanted, 1.0)
            factors = np.where(wanted > 0, growth_factors(levels_of(share)), 1.0)
            self._reweigh(place, cells, factors)
        return changed

    def clear_room(self, cell: Cell, units: int, avoid: frozenset[Cell] = frozenset()) -> set[Cell]:
        """Makes `units` of room in the cell for track or a station: buildings move to a cell
        next to it with room for them (never into `avoid`) while any can, and when none can,
        the lowest of them comes down. Returns the cells whose buildings changed."""
        changed: set[Cell] = set()
        while self.constructions.space_left(cell) < units:
            here = sorted(
                (b for b in self.constructions.at(cell) if isinstance(b, Building)),
                key=lambda b: (
                    self.rule.points.get(b.kind, (Points.POPULATION, 0))[1],
                    b.kind.value,
                ),
            )
            if not here:
                break
            moved = next(
                (
                    (building, target)
                    for building in here
                    if (target := self._room_beside(building, avoid | {cell})) is not None
                ),
                None,
            )
            victim, target = moved if moved is not None else (here[0], None)
            self._take_down(victim)
            changed.add(cell)
            if target is not None:
                self.constructions.build(Building(target, victim.kind))
                if victim.kind in self._counts:
                    self._add(target, victim.kind, 1)
                changed.add(target)
        return changed

    def _room_beside(self, building: Building, avoid: frozenset[Cell]) -> Cell | None:
        """The cell next to the building's with the most room that would take it, if any."""
        options = [
            near
            for near in self._grid.neighbours(*building.cell)
            if near not in avoid
            and self.constructions.why_not(Building(near, building.kind)) is None
        ]
        return max(options, key=self.constructions.space_left, default=None)

    def _take_down(self, building: Building) -> None:
        self.constructions.remove(building)
        if building.kind in self._counts:
            self._add(building.cell, building.kind, -1)

    def _crowd_out_farms(self) -> set[Cell]:
        farms = self._counts.get(BuildingKind.FARM)
        if farms is None:
            return set()
        crowded = np.argwhere((farms > 0) & (self._nearby[Points.POPULATION] > FARMS_CROWDED_OUT))
        changed: set[Cell] = set()
        for row, col in crowded.tolist():
            for _ in range(int(farms[row, col])):
                self.constructions.remove(Building((row, col), BuildingKind.FARM))
            self._add((row, col), BuildingKind.FARM, -int(farms[row, col]))
            changed.add((row, col))
        return changed

    def _station_orders(self) -> dict[Cell, StationOrder]:
        if self.railway is None:
            return {}
        orders = {}
        for station in self.railway.stations():
            goods = self.railway.station_goods(station)
            orders[station] = StationOrder(dict(goods.brought), max(0.0, STOCKPILE - goods.stocked))
        return orders

    def _settle_stations(self, trade: Trade) -> None:
        if self.railway is None:
            return
        for station, handed in trade.handed.items():
            goods = self.railway.station_goods(station)
            for good, amount in handed.items():
                goods.brought[good] = max(0.0, goods.brought.get(good, 0.0) - amount)
        for station, gathered in trade.gathered.items():
            goods = self.railway.station_goods(station)
            for good, amount in gathered.items():
                goods.stockpile[good] = goods.stockpile.get(good, 0.0) + amount

    def _reweigh(self, place: _Place, cells: npt.NDArray[np.intp], factors: FloatGrid) -> None:
        """Scales the ladder's land weights in these cells by their satisfaction factors."""
        if not cells.size:
            return
        place.factors[cells] = factors
        weights = place.land[cells] * factors
        raw = np.full(cells.size, place.ladder.base_chance)
        for points, per_point in place.ladder.chance_per_point.items():
            raw += per_point * self._nearby[points].ravel()[cells]
        place.weights[cells] = weights
        place.chances[cells] = raw * weights

    def satisfaction(self) -> npt.NDArray[np.int8]:
        """How satisfied each cell's buildings are on the whole, or -1 where none trade."""
        if self.trade is None:
            return np.full(self._grid.shape, -1, dtype=np.int8)
        levels = np.full(self._grid.rows * self._grid.cols, -1, dtype=np.int8)
        everything = tuple(self.trade.got)
        trading = self.trade.wanted(everything) > 0
        levels[self.trade.cells[trading]] = self.trade.levels()[trading]
        return levels.reshape(self._grid.shape)

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
        changed: set[Cell] = set()
        self.forests_changed = set()
        if self.rule.trades and self._months % self.rule.trade_every_months == 0:
            changed = self.trade_goods()
        every = self.rule.forests_every_months
        if self.rule.forests is not None and self._months % every == 0:
            changed |= self._change_forests(self.rule.forests, every, rng)
        turns: list[tuple[int, Ladder]] = []
        for place in self._places:
            turns += [(index, place.ladder) for index in self._hits(place, rng).tolist()]
        for turn in rng.permutation(len(turns)).tolist():
            index, ladder = turns[turn]
            changed |= self._turn(divmod(index, self._grid.cols), ladder)
        return changed

    def _change_forests(self, rule: ForestRule, months: int, rng: np.random.Generator) -> set[Cell]:
        """Clears and spreads forests over `months`, closes the lumber mills left without
        forest, and weighs the land again around them; returns the cells that changed."""
        tile_map = self.constructions.tile_map
        people = (self._nearby[Points.POPULATION] + self._nearby[Points.INDUSTRY]).ravel()
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
            forests = forests_nearby(tile_map, self._near).reshape(self._grid.shape)
            for row, col in np.argwhere((count > 0) & (forests < rule.mills_close_below)).tolist():
                for _ in range(int(count[row, col])):
                    self._take_down(Building((row, col), BuildingKind.LUMBER_MILL))
                changed.add((row, col))
        self.economy.free_food = free_food(tile_map)
        around = self._near[:, changed_cells].ravel()
        touched = np.unique(np.concatenate([changed_cells, around[around < self._near.shape[1]]]))
        for place in self._places:
            land = place.ladder.land.weights(tile_map, touched, self._near)
            opened = (land > 0) != (place.land[touched] > 0)
            place.land[touched] = land
            if opened.any():
                place.cells = np.flatnonzero(place.land > 0)
            self._reweigh(place, touched, place.factors[touched])
        return changed

    def run(self, months: int, rng: np.random.Generator) -> None:
        for _ in range(months):
            self.month(rng)

    def _hits(self, place: _Place, rng: np.random.Generator) -> npt.NDArray[np.intp]:
        """The cells that take a turn on the ladder this month. With small chances, cells are
        drawn as if all had the highest chance and each kept by how its own compares, so a month
        costs what grows rather than the size of the map."""
        cells = place.cells
        if not cells.size:
            return cells
        highest = float(place.chances.max())
        if highest <= 0:
            return cells[:0]
        if highest > THINNING_UP_TO:
            drawn = cells[rng.random(cells.size, dtype=np.float32) < place.chances[cells]]
        else:
            tried = np.unique(cells[rng.integers(0, cells.size, rng.binomial(cells.size, highest))])
            drawn = tried[rng.random(tried.size) * highest < place.chances[tried]]
        return drawn[self._meets_needs(place, drawn)]

    def _meets_needs(self, place: _Place, cells: npt.NDArray[np.intp]) -> npt.NDArray[np.bool_]:
        meets = np.ones(cells.size, dtype=bool)
        for points, needed in place.ladder.needs.items():
            meets &= self._nearby[points].ravel()[cells] >= needed
        return meets

    def _turn(self, cell: Cell, ladder: Ladder) -> set[Cell]:
        neighbourhood = [cell, *self._grid.neighbours(*cell)]
        if ladder.skyscrapers and (built := self._skyscraper(cell, neighbourhood)) is not None:
            return built
        for small, big in reversed(list(pairwise(ladder.kinds))):
            merge = ladder.merges.get(small)
            if merge is None or self._around(small, neighbourhood) < merge.needed:
                continue
            merged = self._replace(cell, neighbourhood, [(small, merge.pulled_down)], big)
            if merged is not None:
                return merged
        first = ladder.kinds[0]
        if self._counts[first][cell] >= ladder.most_per_cell:
            return set()
        if (
            ladder.most_nearby is not None
            and self._around(first, neighbourhood) >= ladder.most_nearby
        ):
            return set()
        new = Building(cell, first)
        if self.constructions.why_not(new) is not None:
            return set()
        self.constructions.build(new)
        self._add(cell, first, 1)
        return {cell}

    def _around(self, kind: BuildingKind, neighbourhood: list[Cell]) -> int:
        count = self._counts[kind]
        return sum(int(count[near]) for near in neighbourhood)

    def _skyscraper(self, cell: Cell, neighbourhood: list[Cell]) -> set[Cell] | None:
        rule = self.rule
        if self._nearby[Points.POPULATION][cell] < rule.skyscraper_people:
            return None
        if self._nearby[Points.INDUSTRY][cell] < rule.skyscraper_industry:
            return None
        for factories in SKYSCRAPER_FACTORIES:
            takes = [(BuildingKind.FLATS, SKYSCRAPER_FLATS), factories]
            built = self._replace(cell, neighbourhood, takes, BuildingKind.SKYSCRAPER)
            if built is not None:
                return built
        return None

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
        for near, kind, take in plan:
            for _ in range(take):
                self.constructions.remove(Building(near, kind))
            self._add(near, kind, -take)
        self.constructions.build(new)
        self._add(cell, new_kind, 1)
        return {cell} | {near for near, _, _ in plan}

    def _add(self, cell: Cell, kind: BuildingKind, how_many: int) -> None:
        self._counts[kind][cell] += how_many
        points, value = self.rule.points[kind]
        neighbourhood = [cell, *self._grid.neighbours(*cell)]
        rows, cols = zip(*neighbourhood, strict=True)
        self._nearby[points][rows, cols] += how_many * value
        flat = np.ravel_multi_index((rows, cols), self._grid.shape)
        for place in self._places:
            per_point = place.ladder.chance_per_point.get(points)
            if per_point:
                place.chances[flat] += per_point * how_many * value * place.weights[flat]

    @staticmethod
    def _neighbourhood_sum(values: IntGrid, rows: IntGrid, cols: IntGrid) -> IntGrid:
        own_rows, own_cols = np.indices(values.shape)
        off_map = (rows == own_rows) & (cols == own_cols)
        return values + np.where(off_map, 0, values[rows, cols]).sum(axis=0)
