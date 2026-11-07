"""Goods: what every building makes and needs, how they are shared out between neighbours each
month, and how satisfied that leaves each kind of building in each cell.

Buildings make and need goods (MAKES, NEEDS). The land makes food by itself (FREE_FOOD): every
water cell feeds two, every flat or forest cell one. Goods go no further than a neighbourhood,
the cell and the cells next to it: each cell first serves its own needs, then shares what is
left with the cells next to it, the hungriest first (unsatisfied cells weigh more than content
ones, content more than satisfied). Factories make only as much as their inputs allow, so goods
are shared out in ORDER, raw goods before what is made from them.

A building is satisfied when it gets all it needs and gives away all it makes, content when it
manages at least half, and unsatisfied below that. Satisfaction scales the chances of more
buildings of its kind growing nearby (GROWTH_FACTORS).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind
from railroads.sim.grids import Grid
from railroads.sim.tilemap import Relief, Terrain, TileMap, Vegetation

FloatCells = npt.NDArray[np.float64]


class Good(Enum):
    FOOD = "food"
    LUMBER = "lumber"
    COAL = "coal"
    STEEL = "steel"
    GOODS = "goods"


class Satisfaction(IntEnum):
    UNSATISFIED = 0
    CONTENT = 1
    SATISFIED = 2


ORDER = (Good.FOOD, Good.LUMBER, Good.COAL, Good.STEEL, Good.GOODS)
CONTENT_SHARE = 0.5
GROWTH_FACTORS = {
    Satisfaction.UNSATISFIED: 0.5,
    Satisfaction.CONTENT: 1.0,
    Satisfaction.SATISFIED: 1.5,
}
# How much more an unsatisfied or content cell weighs than a satisfied one when goods are shared.
SHARING_WEIGHTS = {
    Satisfaction.UNSATISFIED: 9.0,
    Satisfaction.CONTENT: 3.0,
    Satisfaction.SATISFIED: 1.0,
}
FREE_FOOD = MappingProxyType({"water": 2.0, "flat": 1.0, "forest": 1.0})

MAKES: Mapping[BuildingKind, Mapping[Good, float]] = MappingProxyType(
    {
        BuildingKind.FARM: {Good.FOOD: 6},
        BuildingKind.ORCHARD: {Good.FOOD: 3},
        BuildingKind.MINE: {Good.COAL: 2},
        BuildingKind.LUMBER_MILL: {Good.LUMBER: 2},
        BuildingKind.FACTORY: {Good.GOODS: 3},
        BuildingKind.BIG_FACTORY: {Good.GOODS: 9},
        BuildingKind.MEGA_FACTORY: {Good.GOODS: 27},
        BuildingKind.FOOD_PLANT: {Good.GOODS: 3},
        BuildingKind.STEEL_MILL: {Good.STEEL: 2},
        BuildingKind.SHIPYARD: {Good.GOODS: 3},
        BuildingKind.PORT: {Good.GOODS: 2},
    }
)
NEEDS: Mapping[BuildingKind, Mapping[Good, float]] = MappingProxyType(
    {
        BuildingKind.HOUSE: {Good.FOOD: 1},
        BuildingKind.TENEMENT: {Good.FOOD: 2, Good.GOODS: 1},
        BuildingKind.FLATS: {Good.FOOD: 5, Good.GOODS: 2},
        BuildingKind.FACTORY: {Good.LUMBER: 2},
        BuildingKind.BIG_FACTORY: {Good.LUMBER: 6},
        BuildingKind.MEGA_FACTORY: {Good.LUMBER: 18},
        BuildingKind.FOOD_PLANT: {Good.FOOD: 3},
        BuildingKind.STEEL_MILL: {Good.COAL: 3},
        BuildingKind.SHIPYARD: {Good.STEEL: 2, Good.LUMBER: 1},
        BuildingKind.SKYSCRAPER: {Good.GOODS: 3},
    }
)


def free_food(tile_map: TileMap) -> FloatCells:
    """The food each cell's land makes by itself."""
    water = tile_map.terrain != Terrain.LAND
    forest = ~water & (tile_map.vegetation == Vegetation.FOREST)
    flat = ~water & ~forest & (tile_map.relief == Relief.FLAT)
    food = np.zeros(tile_map.grid.shape)
    food[water] = FREE_FOOD["water"]
    food[flat] = FREE_FOOD["flat"]
    food[forest] = FREE_FOOD["forest"]
    return food.ravel()


def levels_of(share: FloatCells) -> npt.NDArray[np.int8]:
    """Satisfaction for a share of needs met or goods given away."""
    levels = np.full(share.shape, Satisfaction.UNSATISFIED, dtype=np.int8)
    levels[share >= CONTENT_SHARE] = Satisfaction.CONTENT
    levels[share >= 1.0 - 1e-9] = Satisfaction.SATISFIED
    return levels


def growth_factors(levels: npt.NDArray[np.int8]) -> FloatCells:
    table = np.array([GROWTH_FACTORS[level] for level in Satisfaction])
    return table[levels]


class Neighbourhoods:
    """Sums over every cell's neighbourhood. They work on the whole map, or on a chosen set of
    cells (`on`), where neighbours outside the set count as holding nothing."""

    def __init__(self, grid: Grid) -> None:
        rows, cols = grid.neighbour_index_grids()
        own = np.indices(grid.shape)
        real = ((rows != own[0]) | (cols != own[1])).reshape(len(rows), -1)
        index = np.ravel_multi_index((rows, cols), grid.shape).reshape(len(rows), -1)
        self.size = grid.rows * grid.cols
        # Off the map, a neighbour points at one extra cell that always holds 0.
        self.index = np.where(real, index, self.size)
        self._on: npt.NDArray[np.intp] | None = None
        self._local = self.index

    def on(self, cells: npt.NDArray[np.intp]) -> None:
        """Works on just these cells (flat indices) from now on."""
        where = np.full(self.size + 1, cells.size, dtype=np.intp)
        where[cells] = np.arange(cells.size)
        self._on = cells
        self._local = where[self.index[:, cells]]

    def grow(self, cells: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        """These cells and every cell next to them, as a mask over the whole map."""
        grown = np.zeros(self.size + 1, dtype=bool)
        grown[: self.size] = cells
        grown[self.index[:, cells]] = True
        return grown[: self.size]

    def around(self, values: FloatCells) -> FloatCells:
        """The sum of the values in the cells next to each cell, leaving the cell out."""
        padded = np.append(values, 0.0)
        total = padded[self._local[0]]
        for direction in self._local[1:]:
            total += padded[direction]
        return total

    def including(self, values: FloatCells) -> FloatCells:
        """The sum over each cell's whole neighbourhood, the cell included."""
        return values + self.around(values)


@dataclass(frozen=True)
class Trade:
    """One month of sharing on the cells near buildings (`cells`, flat indices): per kind of
    building and cell, how much of its needs it got and how much of what it can make it gave
    away, both summed over goods."""

    cells: npt.NDArray[np.intp]
    got: Mapping[BuildingKind, FloatCells]
    needed: Mapping[BuildingKind, FloatCells]
    gave: Mapping[BuildingKind, FloatCells]
    could_make: Mapping[BuildingKind, FloatCells]

    def done(self, kinds: tuple[BuildingKind, ...]) -> FloatCells:
        return sum(
            (self.got[k] + self.gave[k] for k in kinds if k in self.got),
            start=np.zeros(self.cells.size),
        )

    def wanted(self, kinds: tuple[BuildingKind, ...]) -> FloatCells:
        return sum(
            (self.needed[k] + self.could_make[k] for k in kinds if k in self.got),
            start=np.zeros(self.cells.size),
        )

    def levels(self, kinds: tuple[BuildingKind, ...] | None = None) -> npt.NDArray[np.int8]:
        """Satisfaction of each cell's buildings of these kinds, or of all of them; cells with
        none count as satisfied."""
        chosen = kinds if kinds is not None else tuple(self.needed)
        done, wanted = self.done(chosen), self.wanted(chosen)
        return levels_of(np.where(wanted > 0, done / np.where(wanted > 0, wanted, 1.0), 1.0))


class Economy:
    """Shares goods out on one map each month, on the cells near buildings that trade."""

    def __init__(self, tile_map: TileMap) -> None:
        self.tile_map = tile_map
        self.cells = tile_map.grid.rows * tile_map.grid.cols
        self.neighbourhoods = Neighbourhoods(tile_map.grid)
        self.free_food = free_food(tile_map)
        self.kinds = tuple(dict.fromkeys([*MAKES, *NEEDS]))
        # Each cell's satisfaction last month, which decides who is served first.
        self.levels = np.full(self.cells, Satisfaction.UNSATISFIED, dtype=np.int8)
        # The cells near buildings that trade, as a mask.
        self.active = np.zeros(self.cells, dtype=bool)

    def trade(self, counts: Mapping[BuildingKind, npt.NDArray[np.int64]]) -> Trade:
        """Shares out one month of goods between the buildings counted in each cell."""
        flat = {k: counts[k].ravel() for k in self.kinds if k in counts}
        built = np.zeros(self.cells, dtype=bool)
        for n in flat.values():
            built |= n > 0
        self.active = self.neighbourhoods.grow(built)
        cells = np.flatnonzero(self.active)
        self.neighbourhoods.on(cells)
        count = {k: n[cells].astype(np.float64) for k, n in flat.items() if n[cells].any()}
        zero = np.zeros(cells.size)
        table = np.array([SHARING_WEIGHTS[level] for level in Satisfaction])
        weights = table[self.levels[cells]]
        got = {k: zero.copy() for k in count}
        needed = {k: zero.copy() for k in count}
        gave = {k: zero.copy() for k in count}
        could_make = {k: zero.copy() for k in count}
        input_share = {k: np.ones(cells.size) for k in count}
        for good in ORDER:
            want = {k: n * NEEDS[k][good] for k, n in count.items() if good in NEEDS.get(k, {})}
            make = {
                k: n * MAKES[k][good] * input_share[k]
                for k, n in count.items()
                if good in MAKES.get(k, {})
            }
            supply = sum(make.values(), start=zero.copy())
            if good == Good.FOOD:
                supply = supply + self.free_food[cells]
            demand = sum(want.values(), start=zero.copy())
            received, sold = self._share(supply, demand, weights)
            met = received / np.where(demand > 0, demand, 1.0)
            given = sold / np.where(supply > 0, supply, 1.0)
            for kind, amount in want.items():
                got[kind] += met * amount
                needed[kind] += amount
                if kind in MAKES:
                    input_share[kind] = np.minimum(
                        input_share[kind], np.where(amount > 0, met, 1.0)
                    )
            for kind, amount in make.items():
                gave[kind] += given * amount
                could_make[kind] += count[kind] * MAKES[kind][good]
        trade = Trade(cells, got, needed, gave, could_make)
        self.levels[:] = Satisfaction.UNSATISFIED
        self.levels[cells] = trade.levels()
        return trade

    def _share(
        self, supply: FloatCells, demand: FloatCells, weights: FloatCells
    ) -> tuple[FloatCells, FloatCells]:
        """How much of each cell's demand is met, and how much of its supply is given away:
        first within the cell, then across to the cells next to it, split by weighted demand."""
        if not demand.any() or not supply.any():
            return np.zeros_like(demand), np.zeros_like(supply)
        local = np.minimum(supply, demand)
        spare, short = supply - local, demand - local
        pull = weights * short
        offers = self.neighbourhoods.around(pull)
        per_pull = np.where(offers > 0, spare / np.where(offers > 0, offers, 1.0), 0.0)
        wanted = pull * self.neighbourhoods.around(per_pull)
        met = np.minimum(wanted, short)
        kept = np.where(wanted > 0, met / np.where(wanted > 0, wanted, 1.0), 0.0)
        taken = per_pull * self.neighbourhoods.around(pull * kept)
        return local + met, local + np.minimum(taken, spare)
