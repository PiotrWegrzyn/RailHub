"""One game: the map's buildings, the towns growing on it, the player's railway and money, and
the history of every year, with the month run in one place and the player's actions checked and
paid for here.

Each action returns a Done: what to tell the player, the cells that changed and whether it went
ahead. Nothing is paid for unless it does, and the price told is what is paid. A station costs
STATION_PRICE and clearing its site (railroads.sim.demolition), track what route_price says, the
bulldozer what clearing the cell does, and moving a station what a new one there would.
"""

from dataclasses import dataclass, field

import numpy as np

from railroads.places import Places, StationNames
from railroads.sim import routes
from railroads.sim.autoplay import StandInPlayer, play
from railroads.sim.clock import START_YEAR
from railroads.sim.constructions import Constructions, TrackWidth
from railroads.sim.demolition import bulldoze, bulldoze_price, station_site_price
from railroads.sim.finance import STATION_PRICE, Finances, dollars
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.railway import Railway
from railroads.sim.stats import History
from railroads.sim.summaries import MonthSummary, Reach, snapshot, summarize
from railroads.sim.tilemap import TileMap


@dataclass(frozen=True)
class Done:
    message: str
    changed: set[Cell] = field(default_factory=set)
    ok: bool = False


@dataclass(frozen=True)
class Month:
    """What a month changed: the cells whose buildings did, and what happened within reach of
    the stations, if anything."""

    changed: set[Cell]
    summary: MonthSummary | None


def place_names(tile_map: TileMap) -> StationNames:
    """Station names taken from the real places on the map."""
    return StationNames(tile_map, Places.load(tile_map.bounds))


class Game:
    def __init__(
        self,
        constructions: Constructions,
        start_year: int = START_YEAR,
        history: History | None = None,
        names: StationNames | None = None,
    ) -> None:
        self.constructions = constructions
        self.tile_map = constructions.tile_map
        self.growth = Growth(constructions, start_year=start_year, history=history)
        self.railway = Railway(constructions, names=names)
        self.railway.year = start_year
        self.railway.history = self.growth.history
        self.growth.railway = self.railway
        # Plays in the player's place in a simulated game (railroads.sim.autoplay).
        self.stand_in: StandInPlayer | None = None
        self._reach = Reach(self.tile_map.grid)

    @property
    def finances(self) -> Finances:
        return self.railway.finances

    def time_passed(self, months: float) -> None:
        """Moves the trains on by this much of a month."""
        self.railway.run(months)

    def month(self, rng: np.random.Generator, date: str = "") -> Month:
        """Grows the towns by a month, or lets the stand-in player play it, and sums up what
        changed within reach of the stations."""
        railway = self.railway
        reach = self._reach.of(railway.stations(), lambda s: railway.reach(s)[1])
        before = snapshot(self.growth, reach)
        if self.stand_in is None:
            changed = self.growth.month(rng)
        else:
            changed = self.stand_in.month(rng)
        return Month(changed, summarize(date, self.growth, reach, before))

    def simulate(self, years: int, rng: np.random.Generator) -> None:
        """Lets a stand-in player build stations and serve the towns for `years`, printing the
        map every ten years, and goes on playing for the player from there."""
        self.stand_in = play(self.growth, self.railway, years, rng)

    def build_station(self, cell: Cell, ends: tuple[int, int] | None = None) -> Done:
        finances = self.finances
        site = station_site_price(self.constructions, cell)
        if site is None:
            return Done("A station can't go there.")
        if not finances.can_afford(STATION_PRICE + site):
            return Done(
                f"This station costs {dollars(STATION_PRICE + site)}; "
                f"you have {dollars(finances.cash)}."
            )
        changed = routes.build_station(self.growth, cell, ends)
        if not changed:
            return Done("A station can't go there.")
        self._pay_for_station(site)
        self.growth.station_trade.measure_demand(cell)
        return Done(f"Station built for {dollars(STATION_PRICE + site)}.", changed, ok=True)

    def lay_track(
        self, start: Cell, end: Cell, width: TrackWidth = TrackWidth.SINGLE, electric: bool = False
    ) -> Done:
        route = routes.plan_route(self.constructions, start, end, width, electric)
        if route is None:
            return Done("No way for track to get there.")
        laid = len(route.changes(self.constructions))
        price = routes.route_price(self.constructions, route)
        finances = self.finances
        if not finances.can_afford(price):
            return Done(f"This track costs {dollars(price)}; you have {dollars(finances.cash)}.")
        changed = routes.lay_route(self.growth, route)
        finances.spend(price, "track")
        message = (
            f"Laid {laid} pieces of track for {dollars(price)}."
            if laid
            else "That track is laid already."
        )
        return Done(message, changed, ok=True)

    def bulldoze(self, cell: Cell) -> Done:
        finances = self.finances
        price = bulldoze_price(self.constructions, cell)
        if price is None:
            return Done("Nothing here can be cleared.")
        if not finances.can_afford(price):
            return Done(f"Clearing this costs {dollars(price)}; you have {dollars(finances.cash)}.")
        changed = bulldoze(self.growth, cell)
        finances.spend(price, "demolition")
        return Done(f"Cleared for {dollars(price)}.", changed, ok=True)

    def move_station(self, old: Cell, new: Cell, ends: tuple[int, int] | None = None) -> Done:
        """Moves a station, its track running between `ends` in its new cell, keeping its
        trains, name and goods but not its upgrades."""
        finances = self.finances
        site = station_site_price(self.constructions, new)
        if site is None:
            return Done("The station can't go there.")
        price = STATION_PRICE + site
        if not finances.can_afford(price):
            return Done(
                f"Moving it there costs {dollars(price)}; you have {dollars(finances.cash)}."
            )
        changed = routes.move_station(self.growth, self.railway, old, new, ends)
        if changed is None:
            return Done("The station can't go there.")
        self._pay_for_station(site)
        self.growth.station_trade.measure_demand(new)
        return Done(f"Station moved for {dollars(price)}.", changed, ok=True)

    def _pay_for_station(self, site: float) -> None:
        self.finances.spend(STATION_PRICE, "stations")
        if site:
            self.finances.spend(site, "demolition")
