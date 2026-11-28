"""A game played by a stand-in player, to see how the map grows when towns are served: from the
start of play, stations open beside the towns with the most people, a few at first and more each
decade, at the first level, every one raised to the second in 1900 and the third in 1950 so they
reach further. No track or trains are laid; instead every cell a station reaches is served in full,
and its buildings grow as if trains brought them all they need."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.clock import MONTHS_PER_YEAR
from railroads.sim.constructions import BuildingKind, Station
from railroads.sim.goods import price_of
from railroads.sim.grids import Cell, Grid, parity_steps
from railroads.sim.growth import Growth
from railroads.sim.move import Move
from railroads.sim.points import Points
from railroads.sim.railway import Railway
from railroads.sim.steps import Steps, finish
from railroads.sim.tiles import of_year
from railroads.sim.yearly import makes_of

STATIONS_AT_START = 5
STATIONS_PER_DECADE = 2
DECADE = 10 * MONTHS_PER_YEAR
# The years from which every station stands at a level: the first until 1900.
RAISED = ((1900, 2), (1950, 3))
BUYS_PER_YEAR = 3


@dataclass(frozen=True, slots=True)
class PlayerRule:
    stations_at_start: int = STATIONS_AT_START
    stations_per_decade: int = STATIONS_PER_DECADE
    raised: tuple[tuple[int, int], ...] = RAISED
    buys_per_year: int = BUYS_PER_YEAR

    def level(self, year: int) -> int:
        """The level every station stands at in a year."""
        return max((level for since, level in self.raised if year >= since), default=1)


PLAYER = PlayerRule()


class StandInPlayer:
    """Builds stations where they reach the most people no station serves yet, raises them as
    the years come, and serves in full every cell they reach."""

    def __init__(
        self,
        growth: Growth,
        railway: Railway,
        rule: PlayerRule = PLAYER,
    ) -> None:
        self.growth = growth
        self.railway = railway
        self.rule = rule
        self.stations: list[Cell] = []
        self.months = 0
        self._grid = growth.constructions.tile_map.grid
        growth.assumed_shares = np.full(self._grid.rows * self._grid.cols, np.nan)
        self.level = rule.level(growth.year)
        self.open_stations(rule.stations_at_start)

    @property
    def reach(self) -> int:
        return self.railway.reach_at(self.level)[0]

    def open_stations(self, count: int) -> set[Cell]:
        """Builds up to `count` stations, each on the free cell whose reach holds the most
        people no station serves yet; returns where they were built."""
        opened: set[Cell] = set()
        for _ in range(count):
            station = self._best_free_cell()
            if station is None:
                break
            self.growth.constructions.build(Station(station))
            self.railway.levels[station] = self.level
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
        """Serves in full every cell the station reaches."""
        shares = self.growth.assumed_shares
        assert shares is not None
        cells = np.array(list(self._grid.within(station, self.reach)))
        shares[np.ravel_multi_index((cells[:, 0], cells[:, 1]), self._grid.shape)] = 1.0

    def _raise(self) -> set[Cell]:
        """Raises every station to the level of the year if it has come, serving the cells
        they now reach; returns the stations raised."""
        level = self.rule.level(self.growth.year)
        if level == self.level:
            return set()
        self.level = level
        for station in self.stations:
            self.railway.levels[station] = level
            self._serve(station)
        return set(self.stations)

    def month(self, rng: np.random.Generator) -> set[Cell]:
        """Grows one month, earning what the player's buildings would at the share of service
        their cells get; each year buys buildings, and each decade
        builds new stations. Returns the cells whose buildings or stations changed."""
        return finish(self.month_steps(rng))

    def month_steps(self, rng: np.random.Generator) -> Steps[set[Cell]]:
        """month in the steps of the growth's month, which a caller can spread over frames."""
        changed = yield from self.growth.month_steps(rng)
        self.months += 1
        self._earn()
        if self.months % MONTHS_PER_YEAR == 0:
            self._buy()
            changed |= self._raise()
        if self.months % DECADE == 0:
            changed |= self.open_stations(self.rule.stations_per_decade)
        return changed

    def _served(self) -> npt.NDArray[np.float64]:
        shares = self.growth.assumed_shares
        assert shares is not None
        return np.nan_to_num(shares).reshape(self._grid.shape)

    def _monthly_worth(self, kind: BuildingKind) -> float:
        makes = makes_of(kind, self.growth.year)
        boards = of_year(self.growth.year)
        return sum(amount * price_of(good) * boards.price(good) for good, amount in makes.items())

    def _earn(self) -> None:
        """Pays the share of what the owned buildings would sell to trains serving their
        cells, since no trains run."""
        growth = self.growth
        served = self._served()
        for kind in growth.owners.kinds():
            sold = float((growth.owners.owned(kind) * served).sum()) * self._monthly_worth(kind)
            earned = sold
            if earned <= 0:
                continue
            self.railway.finances.earn(earned, "cargo")

    def _buy(self) -> None:
        """Buys the buildings in its stations' reach that would pay back soonest."""
        growth = self.growth
        served = self._served()
        offers = []
        for kind in growth.rule.points:
            price = growth.owners.price_to_buy(kind)
            if not price:
                continue
            left = growth.count(kind) - growth.owners.owned(kind)
            for row, col in np.argwhere(left > 0).tolist():
                if (row, col) in growth.station_trade.in_reach and served[row, col] > 0:
                    worth = self._monthly_worth(kind) * served[row, col]
                    offers.append((worth / price, (row, col), kind, price))
        offers.sort(key=lambda offer: -offer[0])
        for _, cell, kind, price in offers[: self.rule.buys_per_year]:
            self.railway.finances.spend(price, "buildings bought")
            growth.owners.own(cell, kind)

    def move_onto(self, growth: Growth, move: Move) -> None:
        """Plays on in the moved game, its stations by their moved cells."""
        self.growth = growth
        self.stations = [move.cell(station) for station in self.stations]
        self._grid = growth.constructions.tile_map.grid


def reach_sums(grid: Grid, values: npt.NDArray[np.int64], steps: int) -> npt.NDArray[np.int64]:
    """For every cell, the sum of `values` over the cells at most `steps` steps from it."""
    sums = np.zeros_like(values)
    padded = np.pad(values, steps)
    for parity, offsets in enumerate(parity_steps(grid, steps)):
        rows = slice(parity, None, 2)
        for d_row, d_col, _ in offsets:
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
    player = StandInPlayer(growth, railway, rule)
    report(describe(player))
    for _ in range(years * MONTHS_PER_YEAR):
        player.month(rng)
        if player.months % DECADE == 0:
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
