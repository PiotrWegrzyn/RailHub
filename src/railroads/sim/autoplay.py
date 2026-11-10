"""A game played by a stand-in player, to see how the map grows when towns are served: from the
start of play, stations open beside the towns with the most people, a few at first and more each
decade, each raised a level so it reaches further. No track or trains are laid; instead each cell a
station reaches is drawn thriving, content or unsatisfied, and its buildings grow as if trains
brought them that share of what they need."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind, Station
from railroads.sim.goods import makes_of, price_of
from railroads.sim.grids import Cell, Grid
from railroads.sim.growth import Growth
from railroads.sim.ladders import Points
from railroads.sim.railway import Railway
from railroads.sim.tech import TREES, tree_of

STATIONS_AT_START = 5
STATIONS_PER_DECADE = 2
# A station raised one level reaches a step further than a new one: three cells out.
STATION_LEVEL = 2
THRIVING_PART = 1 / 3
CONTENT_PART = 1 / 2
CONTENT_SHARE = 0.5
BUYS_PER_YEAR = 3


@dataclass(frozen=True, slots=True)
class PlayerRule:
    stations_at_start: int = STATIONS_AT_START
    stations_per_decade: int = STATIONS_PER_DECADE
    station_level: int = STATION_LEVEL
    thriving_part: float = THRIVING_PART
    content_part: float = CONTENT_PART
    content_share: float = CONTENT_SHARE
    buys_per_year: int = BUYS_PER_YEAR


PLAYER = PlayerRule()


class StandInPlayer:
    """Builds stations where they reach the most people no station serves yet and keeps the
    growth's assumed shares for the cells they reach."""

    def __init__(
        self,
        growth: Growth,
        railway: Railway,
        rng: np.random.Generator,
        rule: PlayerRule = PLAYER,
    ) -> None:
        self.growth = growth
        self.railway = railway
        self.rule = rule
        self.stations: list[Cell] = []
        self.months = 0
        self._rng = rng
        self._grid = growth.constructions.tile_map.grid
        growth.assumed_shares = np.full(self._grid.rows * self._grid.cols, np.nan)
        self.open_stations(rule.stations_at_start)

    @property
    def reach(self) -> int:
        return self.railway.reach_at(self.rule.station_level)[0]

    def open_stations(self, count: int) -> set[Cell]:
        """Builds up to `count` stations, each on the free cell whose reach holds the most
        people no station serves yet; returns where they were built."""
        opened: set[Cell] = set()
        for _ in range(count):
            station = self._best_free_cell()
            if station is None:
                break
            self.growth.constructions.build(Station(station))
            self.railway.levels[station] = self.rule.station_level
            self.railway.name_of(station)
            self._serve(station)
            self.stations.append(station)
            opened.add(station)
        return opened

    def _best_free_cell(self) -> Cell | None:
        shares = self.growth.assumed_shares
        assert shares is not None
        unserved = np.isnan(shares).reshape(self._grid.shape)
        people = np.where(unserved, self.growth.points(Points.POPULATION), 0)
        reached = reach_sums(self._grid, people, self.reach)
        constructions = self.growth.constructions
        for flat in np.argsort(-reached, axis=None, kind="stable").tolist():
            row, col = divmod(flat, self._grid.cols)
            if reached[row, col] <= 0:
                return None
            if constructions.why_not(Station((row, col))) is None:
                return row, col
        return None

    def _serve(self, station: Cell) -> None:
        """Draws the share of each cell the station reaches that no other station reaches."""
        rule = self.rule
        shares = self.growth.assumed_shares
        assert shares is not None
        cells = np.array(list(self._grid.within(station, self.reach)))
        reached = np.ravel_multi_index((cells[:, 0], cells[:, 1]), self._grid.shape)
        reached = reached[np.isnan(shares[reached])]
        drawn = self._rng.random(reached.size)
        shares[reached] = np.where(
            drawn < rule.thriving_part,
            1.0,
            np.where(drawn < rule.thriving_part + rule.content_part, rule.content_share, 0.0),
        )

    def month(self, rng: np.random.Generator) -> set[Cell]:
        """Grows one month, earning what the player's buildings would at the share of service
        their cells get; each year buys buildings and spends its points, and each decade
        builds new stations. Returns the cells whose buildings or stations changed."""
        changed = self.growth.month(rng)
        self.months += 1
        self._earn()
        if self.months % 12 == 0:
            self._buy()
            self._learn()
        if self.months % 120 == 0:
            changed |= self.open_stations(self.rule.stations_per_decade)
        return changed

    def _served(self) -> npt.NDArray[np.float64]:
        shares = self.growth.assumed_shares
        assert shares is not None
        return np.nan_to_num(shares).reshape(self._grid.shape)

    def _monthly_worth(self, kind: BuildingKind) -> float:
        makes = makes_of(kind, self.growth.year)
        return sum(amount * price_of(good) for good, amount in makes.items())

    def _earn(self) -> None:
        """Pays the share of what the owned buildings would sell to trains serving their
        cells, as experience too, since no trains run."""
        growth = self.growth
        served = self._served()
        for kind in growth.owned_kinds():
            tree = tree_of(kind)
            sold = float((growth.owned(kind) * served).sum()) * self._monthly_worth(kind)
            earned = sold * growth.owner_share(kind) * growth.output_factor(kind)
            if earned <= 0:
                continue
            self.railway.finances.earn(earned, "owned buildings")
            if tree is not None:
                self.railway.research.earn(tree, earned)

    def _buy(self) -> None:
        """Buys the buildings in its stations' reach that would pay back soonest."""
        growth = self.growth
        served = self._served()
        offers = []
        for kind in growth.rule.points:
            price = growth.price_to_buy(kind)
            if not price:
                continue
            left = growth.count(kind) - growth.owned(kind)
            for row, col in np.argwhere(left > 0).tolist():
                if (row, col) in growth.in_reach and served[row, col] > 0:
                    worth = self._monthly_worth(kind) * served[row, col]
                    offers.append((worth / price, (row, col), kind, price))
        offers.sort(key=lambda offer: -offer[0])
        for _, cell, kind, price in offers[: self.rule.buys_per_year]:
            self.railway.finances.spend(price, "buildings bought")
            growth.own(cell, kind)

    def _learn(self) -> None:
        """Spends every point on the first node of its tree that can be taken."""
        research = self.railway.research
        for tree, nodes in TREES.items():
            while research.points(tree) > 0:
                key = next((node.key for node in nodes if research.can_take(node.key)), None)
                if key is None:
                    break
                research.take(key)


def reach_sums(grid: Grid, values: npt.NDArray[np.int64], steps: int) -> npt.NDArray[np.int64]:
    """For every cell, the sum of `values` over the cells at most `steps` steps from it."""
    sums = np.zeros_like(values)
    padded = np.pad(values, steps)
    middle_row, col = grid.rows // 2, grid.cols // 2
    for parity in (0, 1):
        row = middle_row - middle_row % 2 + parity
        rows = slice(parity, None, 2)
        for d_row, d_col in ((r - row, c - col) for r, c in grid.within((row, col), steps)):
            top, left = steps + d_row, steps + d_col
            shifted = padded[top : top + grid.rows, left : left + grid.cols]
            sums[rows] += shifted[rows]
    return sums


def counts_of(growth: Growth) -> dict[BuildingKind, int]:
    """How many of every kind stand on the map, leaving out kinds with none."""
    counts = {kind: int(growth.count(kind).sum()) for kind in growth.rule.points}
    return {kind: n for kind, n in counts.items() if n}


def play(
    growth: Growth,
    railway: Railway,
    years: int,
    rng: np.random.Generator,
    report: Callable[[str], None] = print,
    rule: PlayerRule = PLAYER,
) -> StandInPlayer:
    """Plays `years` with a stand-in player without showing the map, reporting it every ten
    years."""
    player = StandInPlayer(growth, railway, rng, rule)
    report(describe(player))
    for _ in range(years * 12):
        player.month(rng)
        if player.months % 120 == 0:
            report(describe(player))
    return player


def describe(player: StandInPlayer) -> str:
    growth = player.growth
    near = growth.nearby(Points.POPULATION)
    counts = ", ".join(f"{kind.value} {n}" for kind, n in counts_of(growth).items())
    return (
        f"{growth.year}: {len(player.stations)} stations, "
        f"{int(growth.points(Points.POPULATION).sum())} people, "
        f"biggest town {int(near.max())} nearby\n  {counts}"
    )
