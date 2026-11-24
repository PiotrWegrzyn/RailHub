"""Trains: they run along laid track from stop to stop, starting at the first. Each stop says
which cars, up to MAX_CARS, the train pulls on to the next one; each car carries one good, up to
its car_size of it.

At each stop a train unloads its cars into the station, freight to be handed out or carried on
by another train and passengers and mail to go on their way, swaps them for that stop's cars and
fills those from the station's stockpile and from freight the buildings around it have no use
for. Freight remembers where it came from: it earns money only when buildings at another station
take it, and back at its own station it just returns to the stockpile. Stations deal with the buildings around them once a month, when goods
are shared out (railroads.sim.economy): they hand what trains brought to buildings within
DELIVERY_RADIUS cells, and take every spare good made within COLLECTION_RADIUS cells into their
stockpile, up to STOCKPILE of each good. Each of the MAX_LEVEL levels a station is raised to
widens both by a cell and adds LEVEL_STORAGE_STEPS storage upgrades' worth of room, and each
storage upgrade adds STORAGE_STEP of each good to its stockpile and WAITING_STEP to its waiting
passengers and mail. Every month each station costs its level's upkeep, the same for every
station, rising every year.

Cargo earns money by the demand around the station that hands it out (railroads.sim.demand). Trains
cost their engine's price, every car car_trip_price by its size each time a train leaves a stop with it, and every
month stations and track cost their upkeep. Each train buys the fuel it burns on the way to its
next stop as it sets off (railroads.sim.fuel_market), and can't set off for a stop further than
its fuel takes it
(railroads.sim.finance).

Trains set off from a stand and pick up speed over their engine's pickup time. They keep out of
each other's way by block signals (railroads.sim.signals): a train runs on past a signal only once
the way beyond it is set for it, and the way is set for the trains in the order they asked for it;
a train that had to stop sets off again from a stand.

The engines, cars, stops and trains are in railroads.sim.trains, what stations hold in
railroads.sim.station_goods, the way along the track in railroads.sim.track_network, and what a
train does calling at a station in railroads.sim.calls; Railway owns them all and runs the trains.
"""

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from railroads.places import StationNames
from railroads.sim import tiles as boards
from railroads.sim.calls import StationCalls
from railroads.sim.clock import SECONDS_PER_MONTH, START_YEAR
from railroads.sim.constructions import Constructions
from railroads.sim.demand import StationDemand
from railroads.sim.events import price_factor as history_price
from railroads.sim.facilities import (
    ENGINE_SHED_SPEED,
    PLACED,
    SAND_TOWER_PICKUP,
    WAREHOUSE_ROOM,
    Facility,
    monthly_cost,
    pay_factor,
    supply,
)
from railroads.sim.facilities import LEVEL as FACILITY_LEVEL
from railroads.sim.facilities import USES as FACILITY_USES
from railroads.sim.finance import (
    LEVEL_PRICES,
    STATION_UPKEEP,
    STATION_UPKEEP_RISE,
    TRAIN_RESALE,
    Finances,
    car_trip_price,
    storage_price,
    track_upkeep,
)
from railroads.sim.fuel_market import (
    GOOD_OF,
    PLANT_FUEL,
    RETAIL_MARKUP,
    Fuel,
    FuelMarket,
    fuel_of,
    signals_of,
)
from railroads.sim.goods import (
    GENERAL_GOODS,
    TRAVELLERS,
    General,
    Good,
    Need,
    freight_of,
    price_of,
)
from railroads.sim.grids import Cell
from railroads.sim.move import Move
from railroads.sim.signals import Signals
from railroads.sim.station_goods import StationGoods
from railroads.sim.stats import History
from railroads.sim.tilemap import TileMap
from railroads.sim.tiles import Tiles
from railroads.sim.track_network import TrackNetwork
from railroads.sim.trade import COLLECTION_RADIUS, DELIVERY_RADIUS, Trade
from railroads.sim.train_names import train_titles
from railroads.sim.trains import (
    BURN_ALONE,
    ENGINES,
    EXPRESS_FARE,
    FIRST_ENGINE,
    GENERATIONS,
    MAGLEV_FARE,
    OPENED_BY,
    SLOWDOWN,
    Cargo,
    Point,
    Power,
    Stop,
    Train,
    car_size,
    full_tonnes,
    general_car_size,
    tare_of,
)
from railroads.sim.year_caches import forget

if TYPE_CHECKING:
    from railroads.sim.growth import Growth
    from railroads.sim.happenings import Happenings
    from railroads.sim.ownership import Owners
    from railroads.sim.station_trade import StationTrade

# What new trains are called, by what drives them, before the number they were bought as.
TRAIN_NAMES = {
    Power.STEAM: "Steam engine",
    Power.DIESEL: "Diesel",
    Power.ELECTRIC: "Electric",
    Power.MAGLEV: "Maglev",
}
# How far a train runs in a month, in cells, on flat ground.
CELLS_PER_MONTH = 48.0
MAX_LEVEL = 1 + len(LEVEL_PRICES)
# How many storage upgrades' worth of room each level a station is raised adds.
LEVEL_STORAGE_STEPS = 2
# From this level up, each level lets a station have one more platform beside it, with its own way
# through, so lines come in from more sides.
PLATFORM_LEVEL = 4
# The level from which a station reaches no further: each level above adds a platform, up to
# PLATFORM_REACH cells away, which reaches as far round it.
REACH_LEVEL = 3
PLATFORM_REACH = 2
# Within how many cells of the end of its path, about its last few blocks, the way set for a
# train must come before it looks again for the quickest platform of the station it is bound for, as it also does every
# PLATFORM_WAIT months it waits at a signal; and how many cells longer a way counts for each cell
# of it another train holds.
PLATFORM_CHOICE = 10.0
PLATFORM_WAIT = 0.02
HELD_WAY_COST = 4.0
# The era of the boards from which quadruple track can be laid: the first lines of four tracks
# came in the 1870s to 1890s.
QUADRUPLE_ERA = 3
# General cars, taking whichever good of their kind pays best, come with this era (railroads.sim.tiles),
# the one of the third generation of wagons.
GENERAL_CARS_ERA = 3
# What a train waits with when no track it can run on joins where it stands to its next stop.
NO_TRACK_ON = "No track to its next stop"
NO_ELECTRIFIED_TRACK_ON = "No electrified track to its next stop"
NO_MAGLEV_TRACK_ON = "No maglev track to its next stop"
NO_WAY_ON = frozenset({NO_TRACK_ON, NO_ELECTRIFIED_TRACK_ON, NO_MAGLEV_TRACK_ON})
# Goods that pay this many times their price when a train delivers them: cattle are hard to haul.
TRANSIT_PAY: Mapping[Good, float] = MappingProxyType({Good.CATTLE: 2.0})


@dataclass(frozen=True, slots=True)
class Ride:
    """What a train's ride to its next stop earns and costs: its cargo's worth there now, the fuel
    for the way, and what its loaded and its empty cars cost to take along."""

    cells: float
    cargo: float
    fuel: float
    loaded_cars: float
    empty_cars: float
    loaded: int
    empty: int

    @property
    def net(self) -> float:
        return self.cargo - self.fuel - self.loaded_cars - self.empty_cars


class Railway:
    """The trains on one map, and the goods held at its stations."""

    # Where goods reaching a station are handed out at once to the buildings around it
    # (railroads.sim.station_trade); set by growth, None until then and in older saves.
    station_trade: StationTrade | None = None
    # The facilities put up in each station itself, taking no room on the map; those standing
    # beside it are in its constructions (railroads.sim.facilities).
    tucked: dict[Cell, set[Facility]] | None = None
    # How many times a train has had to queue for the way on at a signal: the next to queue
    # takes the next number (railroads.sim.signals).
    asks: int = 0
    # The buildings the player owns (railroads.sim.ownership), which make running engines a
    # little cheaper; set by the game, None until then.
    owners: Owners | None = None

    def __init__(
        self,
        constructions: Constructions,
        finances: Finances | None = None,
        names: StationNames | None = None,
    ) -> None:
        self.constructions = constructions
        self.finances = Finances() if finances is None else finances
        # Each station's name (railroads.places).
        self.name_of = StationNames(constructions.tile_map, None) if names is None else names
        for station in self.stations():
            self.name_of(station)
        self.trains: list[Train] = []
        self.goods: dict[Cell, StationGoods] = {}
        self.demand: dict[Cell, StationDemand] = {}
        self.levels: dict[Cell, int] = {}
        self.year = START_YEAR
        # The year play began, from which station upkeep rises.
        self.first_year = START_YEAR
        self._bought = 0
        # What each year ended with and the cars delivered in it (railroads.sim.stats).
        self.history = History()
        self.fuel = FuelMarket()
        # The supply and demand boards, which set what every good earns (railroads.sim.tiles): a
        # game's from the year it begins, flat until then.
        self.tiles = Tiles()
        # What the boards brought at the turn of the year, for the news.
        self.tile_news: list[str] = []
        # The way along the track, and what trains do calling at stations
        # (railroads.sim.track_network, railroads.sim.calls).
        self.network = TrackNetwork(constructions)
        self.calls = StationCalls(self)
        # Each station's town (railroads.sim.towns) and the year it was worked out in.
        self._towns: dict[Cell, tuple[int, int]] = {}
        # What history does to the game, which pays pilgrims more (railroads.sim.happenings).
        self.happenings: Happenings | None = None

    def stations(self) -> list[Cell]:
        return self.constructions.stations()

    def titles(self) -> dict[str, str]:
        """What the player sees each train called, by its own name (train_titles)."""
        return train_titles(self.trains, self.name_of)

    def town_of(self, station: Cell) -> int:
        """The town a station belongs to, 0 for none: worked out the first time it is asked
        for after the station is built, and again each year."""
        kept = self._towns.get(station)
        if kept is None or kept[1] != self.year:
            town = 0 if self.station_trade is None else self.station_trade.town_of(station)
            kept = self._towns[station] = (town, self.year)
        return kept[0]

    def same_town(self, station: Cell, other: Cell) -> bool:
        """Whether two stations belong to the same town, so passengers and mail don't travel
        between them."""
        town = self.town_of(station)
        return town != 0 and town == self.town_of(other)

    @property
    def maglev_open(self) -> bool:
        """Whether the player can lay maglev track: once a maglev engine is open."""
        return any(ENGINES[name].maglev and self.engine_open(name) for name in ENGINES)

    @property
    def quadruple_open(self) -> bool:
        """Whether the player can lay quadruple track: from the era QUADRUPLE_ERA on."""
        return boards.of_year(self.year).wagons >= QUADRUPLE_ERA

    @property
    def electric_open(self) -> bool:
        """Whether the player can lay electric track: once an electric engine is open."""
        return any(ENGINES[name].electric and self.engine_open(name) for name in ENGINES)

    def engines(self) -> list[str]:
        """The engines the player can buy now: those whose supply tile has reached its level
        (engine_open), but not those a newer one of their line has replaced (Engine.replaces)."""
        opened = [name for name in ENGINES if self.engine_open(name)]
        return [
            name
            for name in opened
            if not any(ENGINES[newer].replaces(ENGINES[name]) for newer in opened)
        ]

    def engine_open(self, name: str) -> bool:
        """Whether an engine is on sale: once its supply tile has reached its level in the boards
        in play (railroads.sim.trains.OPENED_BY)."""
        tile, level = OPENED_BY[name]
        return boards.of_year(self.year).supply.get(tile, 0) >= level

    def can_carry(self, train: Train, cargo: Cargo) -> bool:
        """Whether a train's engine takes cars of a good: an express takes only passengers, general
        cars come with the era of GENERAL_CARS_ERA, and vehicles only once its cars are of a
        generation that carries them (railroads.sim.trains.VEHICLES)."""
        if isinstance(cargo, General) and boards.of_year(self.year).era < GENERAL_CARS_ERA:
            return False
        if isinstance(cargo, Good) and not self.car_capacity(cargo, train.engine):
            return False
        return not ENGINES[train.engine].express or cargo == Good.PASSENGERS

    def buy(self, engine: str = FIRST_ENGINE, route_of: Train | None = None) -> Train | None:
        """Buys a train with an engine the player can buy, calling at the stops of `route_of` if
        given, as that one does there and with its cars as far as the engine takes them; with no
        cars otherwise. None without the cash for it."""
        if engine not in self.engines():
            return None
        price = ENGINES[engine].price
        if not self.finances.pay(price, "trains"):
            return None
        self._bought += 1
        train = Train(f"{TRAIN_NAMES[ENGINES[engine].power]} {self._bought}", engine=engine)
        if route_of is not None:
            for old in route_of.stops:
                stop = Stop(old.cell, wait_for_load=old.wait_for_load, refuel_only=old.refuel_only)
                train.stops.append(stop)
                for cargo, generation in old.asked():
                    if self.can_carry(train, cargo) and not self.add_car(
                        train, stop, cargo, generation
                    ):
                        self.add_car(train, stop, cargo)
        self.trains.append(train)
        return train

    def let_go(self) -> None:
        """Lets every held train set off (Train.held)."""
        for train in self.trains:
            train.held = False

    def engine_upgrades(self, train: Train) -> list[str]:
        """The engines a train's engine can be swapped for: those on sale now of its line, newer
        than it."""
        engine = ENGINES[train.engine]
        return [
            name
            for name in self.engines()
            if ENGINES[name].line == engine.line and ENGINES[name].year > engine.year
        ]

    @staticmethod
    def swap_price(train: Train, engine: str) -> float:
        """What swapping a train's engine for another costs: the new one's price, less what the
        old one sells for."""
        return ENGINES[engine].price - ENGINES[train.engine].price * TRAIN_RESALE

    def swap_engine(self, train: Train, engine: str) -> bool:
        """Swaps a train's engine for a newer one of its line, keeping its name, stops and cars,
        and at each stop adding more of its last car while the new engine has room for them and
        can pull them; False if it can't be swapped for that one or there isn't the cash."""
        if engine not in self.engine_upgrades(train):
            return False
        price = self.swap_price(train, engine)
        if not self.finances.pay(price, "trains"):
            return False
        train.engine = engine
        most = self.most_cars(train)
        for stop in train.stops:
            stop.keep_cars(most)
            if stop.cars and not stop.refuel_only:
                last = len(stop.cars) - 1
                cargo, generation = stop.cars[last], stop.generation_at(last)
                while self.add_car(train, stop, cargo, generation):
                    pass
        return True

    def trains_to_upgrade(self, engine: str) -> list[Train]:
        """The trains whose engine can be swapped for `engine`: those of its line with an older
        one."""
        return [train for train in self.trains if engine in self.engine_upgrades(train)]

    def swap_all_engines(self, engine: str) -> int:
        """Swaps the engine of every train of `engine`'s line with an older one for it, in the
        order they were bought, while the cash lasts; returns how many were swapped."""
        return sum(self.swap_engine(train, engine) for train in self.trains_to_upgrade(engine))

    def drop_stops(self, gone: Iterable[Cell]) -> None:
        """Takes stations pulled down out of every train's stops, sending each train that
        called at one back to its first stop left."""
        cells = set(gone)
        for cell in cells:
            (self.tucked or {}).pop(cell, None)
        for train in self.trains:
            kept = [stop for stop in train.stops if stop.cell not in cells]
            if len(kept) != len(train.stops):
                train.stops = kept
                train.restart()

    def sell(self, train: Train) -> None:
        self.trains.remove(train)
        self.finances.earn(ENGINES[train.engine].price * TRAIN_RESALE, "trains")

    def most_cars(self, train: Train) -> int:
        """How many cars a train's engine pulls."""
        return ENGINES[train.engine].cars

    def max_pull(self, engine: str) -> float:
        """The most an engine pulls, in tonnes, whatever generation of cars it pulls."""
        return ENGINES[engine].tonnes

    def pull(self, train: Train) -> float:
        """How much of its engine's top speed a train makes with its cars behind it: all of it
        with none, losing the engine's SLOWDOWN as their weight nears the most it pulls."""
        return 1.0 - SLOWDOWN[ENGINES[train.engine].kind] * self.load_share(train)

    def load_share(self, train: Train) -> float:
        """How much of the most its engine pulls a train's cars weigh, at most all of it."""
        return min(1.0, train.tonnes / self.max_pull(train.engine))

    def burn(self, train: Train) -> float:
        """How much fuel a train burns a cell with its cars behind it: more the more they weigh
        (BURN_ALONE)."""
        return ENGINES[train.engine].burn * (BURN_ALONE + self.load_share(train))

    def car_tonnes(self, cargo: Cargo, engine: str, asked: int = 0) -> float:
        """What a full car for some cargo weighs behind an engine, in tonnes, of the generation
        asked (0 for the newest); a general car as much as the heaviest of its goods would."""
        goods = GENERAL_GOODS[cargo] if isinstance(cargo, General) else (cargo,)
        general = cargo if isinstance(cargo, General) else None
        return max(
            full_tonnes(
                good,
                self.car_capacity(good, engine, general, asked),
                self.car_generation(good, engine, asked),
            )
            for good in goods
        )

    def stop_tonnes(self, train: Train, stop: Stop) -> float:
        """What a train would weigh leaving a stop with every one of its cars full, in tonnes,
        an express's dining car included."""
        express = ENGINES[train.engine].express
        dining = self.car_tare(Good.PASSENGERS, train.engine) if express else 0.0
        cars = stop.asked()
        return dining + sum(self.car_tonnes(cargo, train.engine, asked) for cargo, asked in cars)

    def add_car(self, train: Train, stop: Stop, cargo: Cargo, asked: int = 0) -> bool:
        """Adds a car for some cargo, of the generation asked (0 for the newest), to a stop's
        if the engine has room for one more and can still pull them all full, and the car
        carries it; False if not."""
        if isinstance(cargo, Good) and not self.car_capacity(cargo, train.engine, asked=asked):
            return False
        heavier = self.stop_tonnes(train, stop) + self.car_tonnes(cargo, train.engine, asked)
        if heavier > self.max_pull(train.engine) + 1e-9:
            return False
        return stop.add_car(cargo, self.most_cars(train), asked)

    def fuel_markup(self) -> float:
        """How much more fuel costs bought anywhere but out of a station's stockpile."""
        return RETAIL_MARKUP

    def fuel_cost(self, engine: str, cells: float = 1.0) -> float:
        """What an engine's fuel for `cells` costs bought at retail now, the dearest it comes,
        pulling half the most it can."""
        return ENGINES[engine].burn * cells * self._fuel_price(engine)

    def _fuel_price(self, engine: str) -> float:
        """What a unit of an engine's fuel costs bought at retail now."""
        fuel = fuel_of(ENGINES[engine])
        markup = 0.0 if fuel == Fuel.ELECTRICITY else self.fuel_markup()
        return self.fuel.price(fuel) * (1.0 + markup)

    def _take_fuel(self, train: Train, station: Cell, cells: float) -> None:
        """Buys what a train setting off burns on the way to its next stop: coal, wood (logs) or
        diesel fuel out of the station's stockpile first, at the market price, the rest at retail."""
        engine = ENGINES[train.engine]
        fuel = fuel_of(engine)
        amount = self.burn(train) * cells
        if amount <= 0:
            return
        good = GOOD_OF.get(fuel)
        stocked = 0.0
        if good is not None:
            stockpile = self.station_goods(station).stockpile
            stocked = min(amount, stockpile.get(good, 0.0))
            if stocked > 0:
                stockpile[good] -= stocked
        cost = self.fuel.buy(fuel, amount, stocked, self.fuel_markup())
        self.finances.spend(cost, fuel.value)

    def fuel_month(self, trade: Trade, points: float = 0.0) -> None:
        """Closes the fuel market's month with what this month's trade said about them,
        and the points of the map's homes and factories, which want electricity."""
        owned = (
            None if self.owners is None else {kind: self.owners.owned(kind) for kind in PLANT_FUEL}
        )
        self.fuel.month(self.year, signals_of(trade, owned, points, self.year))

    def car_capacity(
        self,
        good: Good,
        engine: str | None = None,
        general: General | None = None,
        asked: int = 0,
    ) -> float:
        """How much of a good one car of its car_generation carries, a general car of a kind
        (general_car_size) or one for the good alone;
        nothing in a generation before the good's first."""
        generation = self.car_generation(good, engine, asked)
        if general is None:
            size = car_size(good, generation)
        else:
            size = general_car_size(general, good, generation)
        return size

    def car_generation(self, good: Good, engine: str | None = None, asked: int = 0) -> int:
        """The generation of a good's cars: the one asked for if older, or else the newest
        (newest_generation)."""
        newest = self.newest_generation(good, engine)
        return min(asked, newest) if asked else newest

    def newest_generation(self, good: Good, engine: str | None = None) -> int:
        """The newest generation of a good's cars behind an engine: the era's, up to the newest the
        engine pulls."""
        generation = self.wagon_era()
        if engine is None:
            return generation
        return min(generation, ENGINES[engine].top_generation)

    def wagon_era(self) -> int:
        """The newest generation of wagons the era of the boards in play has brought."""
        return min(boards.of_year(self.year).wagons, GENERATIONS)

    def car_tare(self, good: Good, engine: str | None = None, asked: int = 0) -> float:
        """What an empty car for a good of the generation asked weighs behind an engine, in
        tonnes."""
        return tare_of(good, self.car_generation(good, engine, asked))

    def moved(self, old: Cell, new: Cell) -> None:
        """Carries a station's name, goods, demand and trains over to where it now stands; its
        level, storage and facilities start again from nothing. Trains calling there start their rounds
        again."""
        self.name_of.move(old, new)
        goods = self.goods.pop(old, None)
        if goods is not None:
            goods.storage = 0
            self.goods[new] = goods
        demand = self.demand.pop(old, None)
        if demand is not None:
            self.demand[new] = demand
        self.levels.pop(old, None)
        (self.tucked or {}).pop(old, None)
        for train in self.trains:
            calling = [stop for stop in train.stops if stop.cell == old]
            for stop in calling:
                stop.cell = new
            if train.loaded_at == old:
                train.loaded_at = new
            if calling:
                train.restart()

    def station_goods(self, station: Cell) -> StationGoods:
        goods = self.goods.setdefault(station, StationGoods())
        goods.extra_room = (
            WAREHOUSE_ROOM if Facility.WAREHOUSE in self.facilities_at(station) else 0.0
        )
        return goods

    def station_demand(self, station: Cell) -> StationDemand:
        return self.demand.setdefault(station, StationDemand())

    def price_factor(self, good: Good) -> float:
        """How much more a good earns carried by train: its TRANSIT_PAY, what history makes of it (railroads.sim.events) and its price on the boards
        (railroads.sim.tiles)."""
        return (
            TRANSIT_PAY.get(good, 1.0)
            * history_price(good, self.year)
            * self.tiles.of_year(self.year).price(good)
        )

    def destination(self, train: Train) -> Cell | None:
        """The stop the cars a train pulls now are taken to: the one it is heading for, or the
        next one from the stop it stands at, past any it only refuels at."""
        if len(train.stops) < 2:
            return None
        heading = train.heading % len(train.stops)
        if train.stops[heading].refuel_only:
            heading = train.onward(heading)
        return train.stops[heading].cell

    def long_legs(self, train: Train) -> list[tuple[int, float]]:
        """The stops from which the way to the next is longer than the train's engine carries
        fuel for, with how long that way is in cells; the train stands at such a stop."""
        tank = ENGINES[train.engine].tank
        if not tank:
            return []
        legs = enumerate(self.leg_lengths(train.stops, train.engine))
        return [(index, leg) for index, leg in legs if leg is not None and round(leg) > tank]

    def leg_lengths(self, stops: list[Stop], engine: str) -> list[float | None]:
        """How long the way from each stop to the next, the last to the first, is for an engine,
        in cells, from the platform it pulls into there coming from the stop before
        (arrival_platform); None where no track it runs on joins them. Two stops running at the
        same station are no way apart: the train goes on from the platform it stands at."""
        if len(stops) < 2:
            return []
        electric, maglev = ENGINES[engine].electric, ENGINES[engine].maglev
        legs = zip(stops[-1:] + stops[:-1], stops, stops[1:] + stops[:1], strict=True)
        return [
            0.0
            if here.cell == there.cell
            else self.network.route_length(
                self.arrival_platform(before.cell, here.cell, there.cell, engine),
                there.cell,
                electric,
                maglev,
            )
            for before, here, there in legs
        ]

    def arrival_platform(self, before: Cell, station: Cell, after: Cell, engine: str) -> Cell:
        """The cell of a station a train coming from `before` and going on to `after` pulls into:
        of its platforms, the one with the shortest way in and on (onward_cost)."""
        cells = self.constructions.station_cells(
            self.constructions.main_station(station) or station
        )
        if len(cells) < 2:
            return station
        electric, maglev = ENGINES[engine].electric, ENGINES[engine].maglev

        def cost(cell: Cell) -> float:
            way_in = self.network.route_length(before, cell, electric, maglev)
            return (math.inf if way_in is None else way_in) + self.onward_cost(cell, after, engine)

        return min(cells, key=cost)

    def onward_cost(self, platform: Cell, after: Cell, engine: str) -> float:
        """How far a train standing at a platform goes on to its next stop, in cells: nothing if
        that is at the same station, and endless where no track it runs on leads there."""
        main = self.constructions.main_station
        if (main(platform) or platform) == (main(after) or after):
            return 0.0
        electric, maglev = ENGINES[engine].electric, ENGINES[engine].maglev
        onward = self.network.route_length(platform, after, electric, maglev)
        return math.inf if onward is None else onward

    def cargo_worth(self, train: Train) -> list[float]:
        """What each of a train's cars would earn if it were unloaded at its destination now
        (worth_at), passengers on an express paying its fare too; cargo that came from there
        earns nothing, as it only goes back to that station's stockpile."""
        destination = self.destination(train)
        if destination is None:
            return [0.0 for _ in train.cars]
        cargo = [(car.good, car.load - car.origins.get(destination, 0.0)) for car in train.cars]
        earned = self.worth_at(destination, cargo)
        return [
            amount * (self.fare_factor(train) if car.good in TRAVELLERS else 1.0)
            for amount, car in zip(earned, train.cars, strict=True)
        ]

    def worth_at(self, station: Cell, cargo: list[tuple[Good, float]]) -> list[float]:
        """What each load of cargo would earn if a train unloaded it at a station now, one after
        another, as it would be paid, with the station's facilities:
        passengers and mail their freight at the share of demand left as they get off; goods
        their freight for each unit the buildings around would take at once, times what it is
        worth to the building taking it (railroads.sim.station_trade.StationTrade.would_take)."""
        travellers = [(good, amount) for good, amount in cargo if good in TRAVELLERS]
        fares = iter(self.station_demand(station).worth(travellers))
        trade = self.station_trade
        taken = iter([0.0 for _ in cargo] if trade is None else trade.would_take(station, cargo))
        earned = []
        for good, _ in cargo:
            weighed = next(taken)
            base = next(fares) if good in TRAVELLERS else freight_of(good) * weighed
            earned.append(base * self.price_factor(good) * self.facility_pay(station, good))
        return earned

    def ride(self, train: Train, worth: list[float] | None = None) -> Ride | None:
        """What the train's ride to its next stop earns and costs: the one it is on, or the one it
        sets off on next. Fuel is priced at retail, the dearest it comes; `worth` is its cars'
        cargo_worth if already known."""
        destination = self.destination(train)
        if destination is None:
            return None
        engine = ENGINES[train.engine]
        if train.path:
            cells = self.network.length(train)
            burnt = self.network.path_fuel_length(train.path)
        else:
            here = train.stops[(train.heading - 1) % len(train.stops)].cell
            if train.platform in self.constructions.platforms(here):
                here = train.platform
            there = train.stops[train.heading % len(train.stops)].cell
            leg = self.network.route_length(here, there, engine.electric, engine.maglev)
            burnt = self.network.fuel_length(here, there, engine.electric, engine.maglev)
            if leg is None or burnt is None:
                return None
            cells = leg
        loaded = [car for car in train.cars if not car.dining and car.load >= 0.5]
        empty = [car for car in train.cars if car.dining or car.load < 0.5]
        return Ride(
            cells=cells,
            cargo=sum(self.cargo_worth(train) if worth is None else worth),
            fuel=self.burn(train) * burnt * self._fuel_price(train.engine),
            loaded_cars=sum(car_trip_price(car.capacity) for car in loaded),
            empty_cars=sum(car_trip_price(0.0 if car.dining else car.capacity) for car in empty),
            loaded=len(loaded),
            empty=len(empty),
        )

    def network_upkeep(self) -> tuple[float, float]:
        """What all the player's stations and all their track cost a month now."""
        stations = sum(self.station_upkeep(station) for station in self.stations())
        return stations, track_upkeep(self.constructions.track())

    @staticmethod
    def fare_factor(train: Train) -> float:
        """How much more passengers on this train pay: MAGLEV_FARE on a maglev, EXPRESS_FARE on
        another express."""
        engine = ENGINES[train.engine]
        return MAGLEV_FARE if engine.maglev else EXPRESS_FARE if engine.express else 1.0

    def hand_out_now(self, station: Cell, good: Good) -> None:
        """Lets the buildings around a station take a good that has just reached it."""
        if self.station_trade is not None:
            self.station_trade.hand_out_now(station, good)

    def pay_for(
        self, station: Cell, need: Need, good: Good, amount: float, worth: float | None = None
    ) -> None:
        """Pays the freight for cargo trains brought that the buildings around a station took: its
        freight (freight_of) for each unit,
        times what it was worth to the building that took it (`worth`, the units counted by
        railroads.sim.trade.worth_to; the units themselves if not given), with the
        station's facilities. Uses up that much of the station's demand."""
        self.station_demand(station).use(need, amount)
        weighed = amount if worth is None else worth
        earned = freight_of(good) * weighed * self.price_factor(good)
        earned *= self.facility_pay(station, good)
        self.finances.earn(earned, "cargo")

    def sell_own(self, good: Good, amount: float) -> None:
        """Pays the player the market price of what their own buildings brought into a station, as
        the boards price it now: their company sold it, and the railway now carries it."""
        self.finances.earn(self.market_price(good) * amount, "owned buildings")

    def new_year(self, year: int) -> None:
        """Turns the boards to a new year: the tiles picked go up, and what that changes is news."""
        self.tile_news += self.tiles.turn(year)
        forget()

    def facility_pay(self, station: Cell, good: Good) -> float:
        """How many times as much a good delivered at a station pays with its facilities, and for
        passengers while pilgrims crowd its town (railroads.sim.happenings)."""
        pay = pay_factor(self.facilities_at(station), good)
        happenings = self.happenings
        pilgrims = 0 if happenings is None or good != Good.PASSENGERS else happenings.pilgrim_town
        if pilgrims and self.town_of(station) == pilgrims:
            pay *= happenings.pilgrim_pay
        return pay

    def connected(self) -> set[Cell]:
        """The stations that track joins to at least one other station."""
        return self.network.connected()

    def joined(self, station: Cell) -> bool:
        """Whether track joins the station to another, without which it gathers and hands out
        nothing."""
        return station in self.network.connected()

    def level(self, station: Cell) -> int:
        return self.levels.get(station, 1)

    def reach(self, station: Cell) -> tuple[int, int]:
        """How many cells out a station hands goods out to, and gathers them from, round it and
        each of its platforms: as far as at its level, but no further than at REACH_LEVEL, as the
        levels above add platforms instead."""
        return self.reach_at(min(self.level(station), REACH_LEVEL))

    @staticmethod
    def reach_at(level: int) -> tuple[int, int]:
        """How many cells out a station of this level hands goods out to, a cell at the first
        level, and gathers them from; each level a cell further."""
        return DELIVERY_RADIUS + level - 1, COLLECTION_RADIUS + level - 1

    def upgrade_price(self, station: Cell) -> float | None:
        """What raising the station a level costs; None at the top level."""
        level = self.level(station)
        return LEVEL_PRICES[level - 1] if level < MAX_LEVEL else None

    def upgrade(self, station: Cell) -> bool:
        """Raises the station a level, if it can go higher and there is the cash for it."""
        price = self.upgrade_price(station)
        if price is None or not self.finances.pay(price, "station upgrades"):
            return False
        self.levels[station] = self.level(station) + 1
        self.station_goods(station).level_steps += LEVEL_STORAGE_STEPS
        return True

    def platforms_left(self, station: Cell) -> int:
        """How many more platforms the station can have added: one for each level from
        PLATFORM_LEVEL up, less those it has."""
        allowed = max(0, self.level(station) - PLATFORM_LEVEL + 1)
        return max(0, allowed - len(self.constructions.platforms(station)))

    def station_upkeep(self, station: Cell) -> float:
        """What a station costs to keep a month this year, by its level."""
        rise = (1.0 + STATION_UPKEEP_RISE) ** max(0, self.year - self.first_year)
        return STATION_UPKEEP[self.level(station) - 1] * rise

    def facilities_at(self, station: Cell) -> set[Facility]:
        """The facilities put up at a station, in it or beside it (railroads.sim.facilities)."""
        return set(self.constructions.annexes(station)) | (self.tucked or {}).get(station, set())

    def put_up(self, station: Cell, facility: Facility) -> bool:
        """Puts up a facility that stands in the station itself, if it is high enough for it
        and has none of the kind; it costs nothing until its month is paid for."""
        if (
            facility in PLACED
            or self.level(station) < FACILITY_LEVEL[facility]
            or facility in self.facilities_at(station)
        ):
            return False
        if self.tucked is None:
            self.tucked = {}
        self.tucked.setdefault(station, set()).add(facility)
        return True

    def close_facility(self, station: Cell, facility: Facility) -> None:
        """Takes down a facility standing in the station itself, which stops its costs."""
        (self.tucked or {}).get(station, set()).discard(facility)

    def market_price(self, good: Good) -> float:
        """What a unit of a good fetches on the market now, as the boards price it: what the
        player's own buildings sell it for (sell_own) and their facilities pay for it."""
        return price_of(good) * self.tiles.of_year(self.year).price(good)

    def facility_cost(self, station: Cell, facility: Facility) -> float:
        """What a month of a facility beside a station costs with what the station holds now."""
        good, _ = FACILITY_USES[facility]
        held = self.station_goods(station).held(good)
        return monthly_cost(facility, held, self.market_price(good))

    def _supply_facilities(self) -> None:
        """Lets every facility take a month of what it uses from its station and pays for it."""
        cost = sum(
            supply(self.station_goods(station), facility, self.market_price)
            for station in self.stations()
            for facility in sorted(self.facilities_at(station), key=lambda f: f.value)
        )
        if cost > 0:
            self.finances.spend(cost, "station facilities")

    def storage_price(self, station: Cell) -> float:
        """What the station's next storage upgrade costs."""
        return storage_price(self.station_goods(station).storage)

    def add_storage(self, station: Cell) -> bool:
        """Adds STORAGE_STEP of each good to the station's stockpile and WAITING_STEP to its waiting
        room, if there is the cash."""
        price = self.storage_price(station)
        if not self.finances.pay(price, "station upgrades"):
            return False
        self.station_goods(station).storage += 1
        return True

    def turn_month(
        self,
        capacity: Mapping[Cell, Mapping[Good, float]],
    ) -> None:
        """Starts a new month: each station's demand for it, from its capacity, and the upkeep
        of stations and track."""
        for station in self.stations():
            self.station_demand(station).turn_month(capacity.get(station, {}))
        self.finances.close_month(self.year)
        self._supply_facilities()
        track = self.constructions.track()
        stations = sum(self.station_upkeep(station) for station in self.stations())
        self.finances.spend(stations, "station upkeep")
        self.finances.spend(track_upkeep(track), "track upkeep")

    def run(self, months: float) -> None:
        """Moves every train on by this many months, in order of the right of way, each keeping
        on the way its signals set for it, the trains that asked first let on first
        (railroads.sim.signals)."""
        signals = Signals(self.network, self.trains)
        queue = sorted(
            enumerate(self.trains),
            key=lambda pair: (not pair[1].first, not pair[1].asked, pair[1].asked, pair[0]),
        )
        for _, train in queue:
            self._run(train, months, signals)
            signals.leave(train)

    def route(
        self, start: Cell, end: Cell, electric: bool = False, maglev: bool = False
    ) -> tuple[list[Point], list[float]] | None:
        """The way along laid track from one station to another (TrackNetwork.route)."""
        return self.network.route(start, end, electric, maglev)

    def _run(self, train: Train, months: float, signals: Signals) -> None:
        if len(train.stops) < 2:
            train.status = "Needs two stops"
            train.path = []
            return
        if train.held and not train.path:
            train.status = "Sets off once the Buy a train window is closed"
            return
        if not train.path:
            leaving = train.stops[(train.heading - 1) % len(train.stops)]
            if not train.called:
                self.calls.arrive(train, leaving)
            if not leaving.wait_for_load:
                train.waiting = 0.0
            if train.waiting > 0:
                self.calls.load(train, leaving)
                if self.calls.loaded(train, leaving):
                    train.waiting = 0.0
                else:
                    train.waiting -= months
                    train.status = f"Loading at {self.name_of(leaving.cell)}"
                    return
            here = leaving.cell
            if train.platform in self.constructions.platforms(leaving.cell):
                here = train.platform
            there = train.stops[train.heading % len(train.stops)].cell
            engine = ENGINES[train.engine]
            way = self.route(here, there, engine.electric, engine.maglev)
            if way is None:
                train.status = (
                    NO_ELECTRIFIED_TRACK_ON
                    if engine.electric
                    else NO_MAGLEV_TRACK_ON
                    if engine.maglev
                    else NO_TRACK_ON
                )
                return
            leg = self.network.route_length(here, there, engine.electric, engine.maglev)
            assert leg is not None
            if engine.tank and round(leg) > engine.tank:
                train.status = (
                    f"Next stop {leg:.0f} cells away; it carries fuel for {engine.tank:.0f}"
                )
                return
            if months <= 0:
                train.status = "Sets off once time runs"
                return
            burnt = self.network.fuel_length(here, there, engine.electric, engine.maglev)
            assert burnt is not None
            self._take_fuel(train, here, burnt)
            train.path, train.speeds = way
            train.along = 0.0
            train.pace = 0.0
            train.status = "Running"
            if train.cars:
                places = (0.0 if car.dining else car.capacity for car in train.cars)
                trips = sum(map(car_trip_price, places))
                self.finances.spend(trips, "car trips")
        engine = ENGINES[train.engine]
        pickup = engine.pickup / SECONDS_PER_MONTH * (SAND_TOWER_PICKUP if train.sanded else 1.0)
        pace = train.speed_up(months, pickup)
        speed = engine.speed * pace
        if train.fuelled:
            speed *= ENGINE_SHED_SPEED
        wanted = train.along + (
            months * CELLS_PER_MONTH * self.network.speed_at(train) * self.pull(train) * speed
        )
        self._choose_platform(train, signals)
        ask = self.asks + 1
        holder = signals.let_on(train, ask, wanted)
        if train.asked == ask:
            self.asks = ask
        limit, ahead = signals.limit(train)
        allowed = min(wanted, limit)
        train.giving_way = ""
        train.status = "Running"
        if allowed <= train.along and ahead is not None and ahead.pace > 0:
            train.status = f"Following {ahead.name}"
            train.pace = min(train.pace, ahead.pace)
            return
        if allowed <= train.along:
            blocker = ahead or holder
            train.giving_way = blocker.name if blocker is not None else ""
            train.status = (
                f"Following {blocker.name}"
                if ahead
                else f"Waiting at a signal for {blocker.name}"
                if blocker
                else "Waiting at a signal"
            )
            train.pace = 0.0
            train.waited += months
            return
        if allowed < wanted:
            train.pace = min(train.pace, pace * (allowed - train.along) / (wanted - train.along))
            if ahead is not None:
                train.status = f"Following {ahead.name}"
        train.along = allowed
        if train.along >= self.network.length(train):
            train.platform = self.network.cell_at(train.path[-1])
            self.calls.arrive(train, train.stops[train.heading % len(train.stops)])
            train.arrivals += 1
            train.heading = (train.heading + 1) % len(train.stops)
            train.path = []
            train.pace = 0.0
            train.waited = 0.0
            train.first = False

    def _choose_platform(self, train: Train, signals: Signals) -> None:
        """Takes the quickest way on to the station the train is bound for and from it to the stop
        after (onward_cost), if it has more than one platform: looked for again as the way set for it comes within PLATFORM_CHOICE cells
        of the end of its path, and each PLATFORM_WAIT months it waits at a signal, from the end
        of the way set for it, a way counting HELD_WAY_COST cells longer for each cell of it
        another train holds."""
        blocked = bool(train.giving_way)
        length = self.network.length(train)
        if length - train.cleared > PLATFORM_CHOICE and not blocked:
            return
        check = (round(train.cleared, 6), int(train.waited / PLATFORM_WAIT) if blocked else -1)
        if train.platform_check == check:
            return
        train.platform_check = check
        there = train.stops[train.heading % len(train.stops)].cell
        cells = self.constructions.station_cells(self.constructions.main_station(there) or there)
        if len(cells) < 2:
            return
        path = train.path
        ends = self.network.ends(train)
        at = next(
            (
                k
                for k in range(len(path) - 1)
                if ends[k] >= train.cleared - 1e-6
                and not self.network.passing(_middle(path[k], path[k + 1]))[0]
            ),
            None,
        )
        if at is None or ends[at] > train.cleared + 1e-6 and train.cleared > 0:
            return
        a = path[at]
        cell = self.network.cell_at(_middle(a, path[at + 1]))
        entry = self.network.entry_point(cell, a)
        engine = ENGINES[train.engine]
        after = train.stops[(train.heading + 1) % len(train.stops)].cell
        best: tuple[float, Cell, tuple[list[Point], list[float]]] | None = None
        for target in cells:
            way = self.network.way_on(cell, entry, target, engine.electric, engine.maglev)
            if way is None:
                continue
            points = way[0]
            crossed = {
                self.network.cell_at(_middle(p, q)) for p, q in itertools.pairwise(points)
            } - set(cells)
            cost = sum(math.dist(p, q) for p, q in itertools.pairwise(points))
            cost += HELD_WAY_COST * signals.held_by_others(train, crossed)
            cost += self.onward_cost(target, after, train.engine)
            if best is None or cost < best[0]:
                best = (cost, target, way)
        if best is None or best[1] == self.network.cell_at(path[-1]):
            return
        points, speeds = best[2]
        train.path = path[: at + 1] + points
        train.speeds = train.speeds[:at] + [self.network.speed_in(cell)] + speeds
        train.claims_for = train.path

    def position(self, train: Train, back: float = 0.0) -> tuple[Point, Point] | None:
        """Where the train is on the map, and which way it faces (TrackNetwork.position)."""
        return self.network.position(train, back)

    def move_onto(self, tile_map: TileMap, growth: Growth, move: Move) -> None:
        """Moves the stations' goods, demand, levels and names and every train onto
        the moved map, trading with `growth` from now on."""
        growth.railway = self
        self.station_trade = growth.station_trade
        self.owners = growth.owners
        self.goods = move.keys(self.goods)
        for goods in self.goods.values():
            goods.origins = {good: move.parcels(parcels) for good, parcels in goods.origins.items()}
        self.demand = move.keys(self.demand)
        self.levels = move.keys(self.levels)
        if self.tucked is not None:
            self.tucked = move.keys(self.tucked)
        self._towns = {}
        self.name_of.move_onto(tile_map, move)
        for train in self.trains:
            train.move(move)
        self.network = TrackNetwork(self.constructions)
        growth.station_trade.update_reach()


def _middle(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
