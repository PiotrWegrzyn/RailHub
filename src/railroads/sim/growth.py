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
from railroads.sim.grids import Cell
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

IntGrid = npt.NDArray[np.int64]
FloatGrid = npt.NDArray[np.float64]


class Points(Enum):
    POPULATION = "population"
    FOOD = "food"
    ORE = "ore"
    TIMBER = "timber"
    TRADE = "trade"
    INDUSTRY = "industry"
    CORPORATE = "corporate"


MONTHS_BEFORE_PLAY = 300 * 12
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
    the best water next to the cell, LAND standing for none; anything left out weighs 1, or 0
    for water."""

    vegetation: Mapping[Vegetation, float] = field(default_factory=dict)
    relief: Mapping[Relief, float] = field(default_factory=dict)
    temperature: Mapping[Temperature, float] = field(default_factory=dict)
    water: Mapping[Terrain, float] = field(default_factory=dict)

    def weights(self, tile_map: TileMap) -> FloatGrid:
        vegetation = np.array([self.vegetation.get(v, 1.0) for v in Vegetation])
        relief = np.array([self.relief.get(r, 1.0) for r in Relief])
        temperature = np.array([self.temperature.get(t, 1.0) for t in Temperature])
        water = np.array([self.water.get(t, 1.0 if t == Terrain.LAND else 0.0) for t in Terrain])
        rows, cols = tile_map.grid.neighbour_index_grids()
        around = tile_map.terrain[rows, cols]
        is_water = around != Terrain.LAND
        best = np.where(is_water, water[around], 0.0).max(axis=0)
        by_water = np.where(is_water.any(axis=0), best, water[Terrain.LAND])
        weight = (
            vegetation[tile_map.vegetation]
            * relief[tile_map.relief]
            * temperature[tile_map.temperature]
            * by_water
        )
        return np.where(tile_map.terrain == Terrain.LAND, weight, 0.0)


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
TOWN_LAND = Land(
    vegetation={Vegetation.BARREN: 0.5, Vegetation.FOREST: 0.05},
    relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.05},
    temperature={Temperature.COLD: 0.6, Temperature.HOT: 0.8},
    water={Terrain.RIVER: 1.7, Terrain.SEA: 1.7, Terrain.LAKE: 1.0},
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
    base_chance=0.00001,
    chance_per_point={Points.POPULATION: 0.00025},
    merges={
        BuildingKind.HOUSE: Merge(needed=6, pulled_down=3),
        BuildingKind.TENEMENT: Merge(needed=6, pulled_down=3),
    },
    skyscrapers=True,
)
# Farms want open, flat, mild grassland, and towns nearby to feed; each fills its cell with fields.
FARMS = Ladder(
    kinds=(BuildingKind.FARM,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.1, Vegetation.FOREST: 0.02},
        relief={Relief.HILLS: 0.5, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.2, Temperature.HOT: 0.6},
        water={Terrain.RIVER: 1.3, Terrain.LAKE: 1.2},
    ),
    base_chance=0.000002,
    chance_per_point={Points.POPULATION: 0.000002},
    most_per_cell=1,
    most_nearby=2,
)
# Orchards like sunny slopes and water nearby, and stay off the cold.
ORCHARDS = Ladder(
    kinds=(BuildingKind.ORCHARD,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.2, Vegetation.FOREST: 0.1},
        relief={Relief.FLAT: 0.5, Relief.HILLS: 1.0, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.05, Temperature.HOT: 1.2},
        water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.5, Terrain.SEA: 1.2},
    ),
    base_chance=0.000001,
    chance_per_point={Points.POPULATION: 0.000001},
    most_per_cell=1,
    most_nearby=2,
)
# Mines are dug in mountains and hills, where finding ore draws more digging nearby.
MINES = Ladder(
    kinds=(BuildingKind.MINE,),
    land=Land(
        vegetation={Vegetation.FOREST: 0.6, Vegetation.GRASS: 0.8},
        relief={Relief.FLAT: 0.01, Relief.HILLS: 0.4, Relief.MOUNTAINS: 1.0},
        temperature={Temperature.COLD: 0.8},
    ),
    base_chance=0.000004,
    chance_per_point={Points.ORE: 0.00005},
    most_per_cell=2,
    most_nearby=5,
)
# Lumber mills stand in forests, best by a river to float the logs and near towns to buy them.
LUMBER_MILLS = Ladder(
    kinds=(BuildingKind.LUMBER_MILL,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.0, Vegetation.GRASS: 0.0},
        relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.HOT: 0.6},
        water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.2, Terrain.SEA: 1.0},
    ),
    base_chance=0.0000015,
    chance_per_point={Points.POPULATION: 0.000002},
    most_per_cell=1,
    most_nearby=1,
)
# Ports open on the sea, or less often a lake, where a town has grown to use them.
PORTS = Ladder(
    kinds=(BuildingKind.PORT,),
    land=Land(
        relief={Relief.HILLS: 0.5, Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.0, Terrain.SEA: 1.0, Terrain.LAKE: 0.3},
    ),
    base_chance=0.000002,
    chance_per_point={Points.POPULATION: 0.00002},
    needs={Points.POPULATION: 3},
    most_per_cell=1,
    most_nearby=1,
)
# Factories make goods for the people around them and draw more factories.
FACTORIES = Ladder(
    kinds=(BuildingKind.FACTORY, BuildingKind.BIG_FACTORY, BuildingKind.MEGA_FACTORY),
    land=FACTORY_LAND,
    chance_per_point={Points.POPULATION: 0.000001, Points.INDUSTRY: 0.00001},
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
    chance_per_point={Points.FOOD: 0.00002, Points.POPULATION: 0.000001},
    needs={Points.FOOD: 1, Points.POPULATION: 6},
    most_per_cell=1,
    most_nearby=1,
)
# Steel mills need ore from mines nearby and workers, and grow among other industry.
STEEL_MILLS = Ladder(
    kinds=(BuildingKind.STEEL_MILL,),
    land=Land(
        vegetation={Vegetation.FOREST: 0.2},
        relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
        water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.2, Terrain.SEA: 1.2},
    ),
    chance_per_point={Points.ORE: 0.0001, Points.INDUSTRY: 0.00001},
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
    chance_per_point={Points.TRADE: 0.00005, Points.INDUSTRY: 0.00001},
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
    ladders: tuple[Ladder, ...] = LADDERS
    points: Mapping[BuildingKind, tuple[Points, int]] = field(default=POINTS)
    skyscraper_people: int = SKYSCRAPER_PEOPLE
    skyscraper_industry: int = SKYSCRAPER_INDUSTRY


GROWTH = GrowthRule()


@dataclass(frozen=True, slots=True)
class _Place:
    """Where a ladder can grow: its cells as flat indices, every cell's land weight, and every
    cell's chance before the ladder's needs, kept up to date as buildings come and go."""

    ladder: Ladder
    cells: npt.NDArray[np.intp]
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
        self._places = []
        for ladder in rule.ladders:
            weights = ladder.land.weights(tile_map).ravel()
            chances = np.full(weights.size, ladder.base_chance)
            for points, per_point in ladder.chance_per_point.items():
                chances += per_point * self._nearby[points].ravel()
            place = _Place(ladder, np.flatnonzero(weights > 0), weights, chances * weights)
            self._places.append(place)

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
        turns: list[tuple[int, Ladder]] = []
        for place in self._places:
            turns += [(index, place.ladder) for index in self._hits(place, rng).tolist()]
        changed: set[Cell] = set()
        for turn in rng.permutation(len(turns)).tolist():
            index, ladder = turns[turn]
            changed |= self._turn(divmod(index, self._grid.cols), ladder)
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
