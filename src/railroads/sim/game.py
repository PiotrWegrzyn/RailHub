"""One game: the map's buildings, the towns growing on it, the player's railway and money, and
the history of every year, with the month run in one place and the player's actions checked and
paid for here.

Each action returns an Outcome: what to tell the player, the cells that changed and whether it went
ahead. Nothing is paid for unless it does, and the price told is what is paid. A station costs
STATION_PRICE and clearing its site (railroads.sim.demolition), track what route_price says, the
bulldozer what clearing the cell does, and moving a station what a new one there would.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from railroads.places import Places, StationNames
from railroads.sim import routes
from railroads.sim.autoplay import StandInPlayer, play
from railroads.sim.clock import MONTHS_PER_YEAR, START_YEAR
from railroads.sim.company import Goal, News, company_value
from railroads.sim.constructions import Building, Constructions, TrackWidth
from railroads.sim.demolition import bulldoze, bulldoze_price, station_site_price
from railroads.sim.finance import STATION_PRICE, Finances
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.happenings import Happenings, Headline
from railroads.sim.lots import Lots
from railroads.sim.move import Move
from railroads.sim.railway import MAX_LEVEL, PLATFORM_LEVEL, PLATFORM_REACH, QUADRUPLE_ERA, Railway
from railroads.sim.stats import History
from railroads.sim.steps import Steps, finish
from railroads.sim.summaries import MonthSummary, Reach, summarize
from railroads.sim.tilemap import TileMap
from railroads.sim.tiles import Tiles, activate
from railroads.words import dollars, either


@dataclass(frozen=True)
class Quote:
    """What building or clearing in a cell would cost, all told, and how much of that is
    clearing the cell; or, with no price, why it can't be done there."""

    price: float | None
    site: float = 0.0
    problem: str = ""


@dataclass(frozen=True)
class Outcome:
    message: str
    changed: set[Cell] = field(default_factory=set)
    ok: bool = False


@dataclass(frozen=True)
class Month:
    """What a month changed: the cells whose buildings did, and what happened within reach of
    the stations, if anything."""

    changed: set[Cell]
    summary: MonthSummary | None
    # The news of history it brought (railroads.sim.happenings).
    headlines: tuple[Headline, ...] = ()


def place_names(tile_map: TileMap) -> StationNames:
    """Station names taken from the real places on the map."""
    return StationNames(tile_map, Places.near(tile_map.bounds))


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
        self.railway.first_year = start_year
        self.railway.tiles = Tiles.until(start_year)
        activate(self.railway.tiles)
        self.railway.history = self.growth.history
        self.growth.railway = self.railway
        self.railway.station_trade = self.growth.station_trade
        self.railway.owners = self.growth.owners
        # Plays in the player's place in a simulated game (railroads.sim.autoplay).
        self.stand_in: StandInPlayer | None = None
        # The land the player owns and can buy around the map (railroads.sim.lots); None for a
        # game on a map of its own, with nothing around it.
        self.lots: Lots | None = None
        # The company's worth each month, its target and going bankrupt (railroads.sim.company).
        self.goal = Goal()
        self._reach = Reach(self.tile_map.grid)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        activate(self.railway.tiles)

    def begin_history(self) -> None:
        """Lets history happen to the game from now on (railroads.sim.happenings)."""
        self.growth.happenings = self.railway.happenings = Happenings()

    @property
    def finances(self) -> Finances:
        return self.railway.finances

    def big_parts(self) -> list[object]:
        """What of the game takes longest to pickle, in parts a save pickles one at a time before
        the rest (railroads.saves.snapshot_steps)."""
        return [self.constructions, *self.growth.big_parts()]

    def time_passed(self, months: float) -> None:
        """Moves the trains on by this much of a month."""
        self.railway.run(months)

    def month(self, rng: np.random.Generator, date: str = "") -> Month:
        """Grows the towns by a month, or lets the stand-in player play it, notes what the company
        is worth at the start of the new month, and sums up what changed within reach of the
        stations."""
        return finish(self.month_steps(rng, date))

    def month_steps(self, rng: np.random.Generator, date: str = "") -> Steps[Month]:
        """The month in a few steps, which a caller can spread over frames; nothing else may
        change the game between them."""
        railway = self.railway
        reach = self._reach.of(
            railway.stations(), lambda s: railway.reach(s)[1], self.constructions.platforms
        )
        with self.growth.log_changes() as changes:
            if self.stand_in is None:
                changed = yield from self.growth.month_steps(rng)
            else:
                changed = yield from self.stand_in.month_steps(rng)
        growth = self.growth
        self.goal.begin_month(growth.year, growth.months % MONTHS_PER_YEAR, company_value(railway))
        headlines = () if growth.happenings is None else tuple(growth.happenings.take())
        self.goal.news += [
            News(h.title, "\n".join((h.story, "", *h.changes)), bad=True)
            for h in headlines
            if h.big
        ]
        self.goal.news += [News("A new era", news) for news in railway.tile_news]
        railway.tile_news.clear()
        summary = summarize(date, self.growth.rule.points, reach, changes)
        return Month(changed, summary, headlines)

    def simulate(self, years: int, rng: np.random.Generator) -> None:
        """Lets a stand-in player build stations and serve the towns for `years`, printing the
        map every ten years, and goes on playing for the player from there."""
        self.stand_in = play(self.growth, self.railway, years, rng)

    def station_quote(self, cell: Cell) -> Quote:
        """What a station in the cell costs, with clearing the cell."""
        problem = routes.station_problem(self.constructions, cell)
        if problem is not None:
            return Quote(None, problem=problem)
        site = station_site_price(self.constructions, cell)
        if site is None:
            return Quote(None, problem="a building here can't be taken down")
        return Quote(STATION_PRICE + site, site)

    def platform_quote(self, cell: Cell) -> Quote:
        """What clearing the cell for a platform costs."""
        site = station_site_price(self.constructions, cell)
        return Quote(None) if site is None else Quote(site, site)

    def clearing_quote(self, cell: Cell) -> Quote:
        """What clearing the cell with the bulldozer costs."""
        return Quote(bulldoze_price(self.constructions, cell))

    def build_station(self, cell: Cell, ends: tuple[int, int] | None = None) -> Outcome:
        quote = self.station_quote(cell)
        if quote.price is None:
            return Outcome(f"A station can't go there: {quote.problem}.")
        if short := self.finances.short_of(quote.price, "This station"):
            return Outcome(short)
        changed = routes.build_station(self.growth, cell, ends)
        if not changed:
            return Outcome("A station can't go there.")
        self._pay_for_station(quote.site)
        self.growth.station_trade.measure_demand(cell)
        return Outcome(f"Station built for {dollars(quote.price)}.", changed, ok=True)

    def lay_track(
        self,
        start: Cell,
        end: Cell,
        width: TrackWidth = TrackWidth.SINGLE,
        electric: bool = False,
        maglev: bool = False,
        planned: routes.Route | None = None,
    ) -> Outcome:
        """Lays track from `start` to `end`: the route `planned` for it already, if it was
        planned with these choices since anything was last built, else the cheapest."""
        if maglev and not self.railway.maglev_open:
            return Outcome("Maglev track needs a maglev engine on sale.")
        if electric and not self.railway.electric_open:
            return Outcome("Electric track needs an electric engine on sale.")
        if width == TrackWidth.QUADRUPLE and not self.railway.quadruple_open:
            return Outcome(f"Quadruple track comes with era {QUADRUPLE_ERA}.")
        route = (
            planned
            if planned is not None
            else routes.plan_route(self.constructions, start, end, width, electric, maglev)
        )
        if route is None:
            return Outcome("No way for track to get there.")
        laid = len(route.changes(self.constructions))
        price = routes.route_price(self.constructions, route)
        finances = self.finances
        if short := finances.short_of(price, "This track"):
            return Outcome(short)
        changed = routes.lay_route(self.growth, route)
        finances.spend(price, "track")
        message = (
            f"Laid {laid} pieces of track for {dollars(price)}."
            if laid
            else "That track is laid already."
        )
        return Outcome(message, changed, ok=True)

    def bulldoze(self, cell: Cell) -> Outcome:
        finances = self.finances
        price = self.clearing_quote(cell).price
        if price is None:
            return Outcome("Nothing here can be cleared.")
        if short := finances.short_of(price, "Clearing this"):
            return Outcome(short)
        stations = set(self.railway.stations())
        changed = bulldoze(self.growth, cell)
        self.railway.drop_stops(stations - set(self.railway.stations()))
        finances.spend(price, "demolition")
        return Outcome(f"Cleared for {dollars(price)}.", changed, ok=True)

    def add_platform(
        self, station: Cell, cell: Cell, ends: tuple[int, int] | None = None
    ) -> Outcome:
        """Adds a platform to a station raised high enough, in a cell beside it, for what
        clearing the cell costs."""
        if self.railway.platforms_left(station) <= 0:
            levels = either(range(PLATFORM_LEVEL, MAX_LEVEL + 1))
            return Outcome(f"Raise the station to level {levels} for another platform.")
        site = self.platform_quote(cell).price
        if site is None:
            return Outcome("A platform can't go there.")
        if short := self.finances.short_of(site, "Clearing it"):
            return Outcome(short)
        changed = routes.add_platform(self.growth, station, cell, ends)
        if not changed:
            return Outcome(
                f"A platform goes in a free cell up to {PLATFORM_REACH} cells from the station."
            )
        if site:
            self.finances.spend(site, "demolition")
        price = f" for {dollars(site)}" if site else ""
        return Outcome(f"Platform added{price}.", changed, ok=True)

    def move_station(self, old: Cell, new: Cell, ends: tuple[int, int] | None = None) -> Outcome:
        """Moves a station, its track running between `ends` in its new cell, keeping its
        trains, name and goods but not its upgrades."""
        quote = self.station_quote(new)
        if quote.price is None:
            return Outcome(f"The station can't go there: {quote.problem}.")
        if short := self.finances.short_of(quote.price, "Moving it there"):
            return Outcome(short)
        changed = routes.move_station(self.growth, self.railway, old, new, ends)
        if changed is None:
            return Outcome("The station can't go there.")
        self._pay_for_station(quote.site)
        self.growth.station_trade.measure_demand(new)
        return Outcome(f"Station moved for {dollars(quote.price)}.", changed, ok=True)

    def _pay_for_station(self, site: float) -> None:
        self.finances.spend(STATION_PRICE, "stations")
        if site:
            self.finances.spend(site, "demolition")

    def move_onto(self, tile_map: TileMap, move: Move, buildings: Iterable[Building] = ()) -> None:
        """Moves the game onto `tile_map` as `move` moves its cells, and puts up `buildings`
        there, already on the new map's cells: those that can stand there as the map now is."""
        self.constructions.move_onto(tile_map, move)
        for building in buildings:
            if self.constructions.why_not(building) is None:
                self.constructions.build(building)
        self.tile_map = tile_map
        self.growth = self.growth.moved(move)
        self.railway.move_onto(tile_map, self.growth, move)
        self._reach = Reach(tile_map.grid)
        if self.stand_in is not None:
            self.stand_in.move_onto(self.growth, move)
