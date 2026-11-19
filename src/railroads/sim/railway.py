"""Trains: they run along laid track from stop to stop, starting at the first. Each stop says
which cars, up to MAX_CARS, the train pulls on to the next one; each car carries one good, up to
its car_size of it.

At each stop a train unloads its cars into the station, freight to be handed out or carried on
by another train and passengers and mail to go on their way, swaps them for that stop's cars and
fills those from the station's stockpile and from freight the buildings around it have no use
for. Freight remembers where it came from: it earns money only when buildings at another station
take it, and back at its own station it just returns to the stockpile. Stations deal with the buildings around them once a month, when goods
are shared out (railroads.sim.economy): they hand what trains brought to buildings within
FIRST_REACH cells, and take every spare good made within COLLECTION_RADIUS cells into their
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

Trains set off from a stand and pick up speed over their engine's pickup time. Trains that meet
on the line, ahead of or against each other or where lines cross, can't run through each other
for free: one of them stops and lets the other by, then sets off again from a stand. The train
that has stood longest letting others by since it left its last stop goes first, so none waits for
ever; then the better engine, then the train bought first. Within stations, and on double track
for trains going opposite ways, trains pass each other without stopping.

The engines, cars, stops and trains are in railroads.sim.trains, what stations hold in
railroads.sim.station_goods, the way along the track in railroads.sim.track_network, and what a
train does calling at a station in railroads.sim.calls; Railway owns them all and runs the trains.
"""

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.places import StationNames
from railroads.sim.calls import StationCalls
from railroads.sim.clock import SECONDS_PER_MONTH, START_YEAR
from railroads.sim.constructions import Constructions
from railroads.sim.demand import StationDemand
from railroads.sim.economy import COLLECTION_RADIUS, Trade
from railroads.sim.facilities import (
    ENGINE_SHED_SPEED,
    SAND_TOWER_PICKUP,
    WAREHOUSE_ROOM,
    Facility,
    pay_factor,
    supply,
)
from railroads.sim.facilities import PRICES as FACILITY_PRICES
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
from railroads.sim.goods import FOODS, TRAVELLERS, General, Good, Need, freight_of, price_of
from railroads.sim.grids import Cell
from railroads.sim.station_goods import StationGoods
from railroads.sim.stats import History
from railroads.sim.tech import GENERAL_CARS, GOOD_TREES, Effect, Research, Tree
from railroads.sim.track_network import TrackNetwork
from railroads.sim.trains import (
    ENGINES,
    EXPRESS_FARE,
    FIRST_ENGINE,
    MAGLEV_FARE,
    Cargo,
    Point,
    Power,
    Stop,
    Train,
    bigger,
    car_size,
    general_car_size,
)

if TYPE_CHECKING:
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
CELLS_PER_MONTH = 40.0
# Trains meet where points along them, MEET_STEP cells apart, come within MEET_DISTANCE cells.
MEET_STEP = 0.2
MEET_DISTANCE = 0.2
MAX_LEVEL = 1 + len(LEVEL_PRICES)
# How many cells out a new station hands goods out to; and how many storage upgrades' worth of
# room each level a station is raised adds.
FIRST_REACH = 1
LEVEL_STORAGE_STEPS = 2
# From this level up, each level lets a station have one more platform beside it, with its own way
# through, so lines come in from more sides.
PLATFORM_LEVEL = 4
# Upkeep never falls below this share of its full cost, however much tech cuts it.
LEAST_UPKEEP = 0.5
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
    # The facilities put up beside each station, and those that got what they use this month
    # (railroads.sim.facilities). None in older saves, which have none.
    facilities: dict[Cell, set[Facility]] | None = None
    supplied: dict[Cell, set[Facility]] | None = None
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
        # What the player has learned (railroads.sim.tech).
        self.research = Research()
        # What each year ended with and the cars delivered in it (railroads.sim.stats).
        self.history = History()
        self.fuel = FuelMarket()
        # The way along the track, and what trains do calling at stations
        # (railroads.sim.track_network, railroads.sim.calls).
        self.network = TrackNetwork(constructions)
        self.calls = StationCalls(self)
        # Each station's town (railroads.sim.towns) and the year it was worked out in.
        self._towns: dict[Cell, tuple[int, int]] = {}

    def stations(self) -> list[Cell]:
        return self.constructions.stations()

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
        """Whether the player can lay maglev track: once a maglev engine's year has come and
        they have its tech."""
        taken = self.research.taken
        return any(
            engine.maglev and engine.year <= self.year and all(n in taken for n in engine.needs)
            for engine in ENGINES.values()
        )

    @property
    def electric_open(self) -> bool:
        """Whether the player can lay electric track: once an electric engine's year has come
        and they have its tech."""
        taken = self.research.taken
        return any(
            engine.electric and engine.year <= self.year and all(n in taken for n in engine.needs)
            for engine in ENGINES.values()
        )

    def engines(self) -> list[str]:
        """The engines the player can buy now: those whose year has come and whose tech they
        have, but not those a newer one of their line has replaced (Engine.replaces)."""
        taken = self.research.taken
        opened = [
            name
            for name, engine in ENGINES.items()
            if engine.year <= self.year and all(need in taken for need in engine.needs)
        ]
        return [
            name
            for name in opened
            if not any(ENGINES[newer].replaces(ENGINES[name]) for newer in opened)
        ]

    def can_carry(self, train: Train, cargo: Cargo) -> bool:
        """Whether a train's engine takes cars of a good: an express takes only passengers;
        and a general car only once its tech is taken."""
        if isinstance(cargo, General) and not self.general_open(cargo):
            return False
        return not ENGINES[train.engine].express or cargo == Good.PASSENGERS

    def general_open(self, general: General) -> bool:
        """Whether the player has the tech for general cars of a kind."""
        return GENERAL_CARS[general] in self.research.taken

    def buy(self, engine: str = FIRST_ENGINE, route_of: Train | None = None) -> Train | None:
        """Buys a train with an engine the player can buy, calling at the stops of `route_of` if
        given, with no cars yet; None without the cash for it."""
        if engine not in self.engines():
            return None
        price = ENGINES[engine].price
        if not self.finances.pay(price, "trains"):
            return None
        self._bought += 1
        train = Train(f"{TRAIN_NAMES[ENGINES[engine].power]} {self._bought}", engine=engine)
        if route_of is not None:
            train.stops = [Stop(stop.cell) for stop in route_of.stops]
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
        """Swaps a train's engine for a newer one of its line, keeping its name, stops and cars;
        False if it can't be swapped for that one or there isn't the cash."""
        if engine not in self.engine_upgrades(train):
            return False
        price = self.swap_price(train, engine)
        if not self.finances.pay(price, "trains"):
            return False
        train.engine = engine
        most = self.most_cars(train)
        for stop in train.stops:
            del stop.cars[most:]
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
        for train in self.trains:
            kept = [stop for stop in train.stops if stop.cell not in cells]
            if len(kept) != len(train.stops):
                train.stops = kept
                train.restart()

    def sell(self, train: Train) -> None:
        self.trains.remove(train)
        self.finances.earn(ENGINES[train.engine].price * TRAIN_RESALE, "trains")

    def most_cars(self, train: Train) -> int:
        """How many cars a train's engine pulls, with what tech adds."""
        return ENGINES[train.engine].cars + int(self.research.bonus(Effect.PULL))

    def fuel_markup(self) -> float:
        """How much more fuel costs bought anywhere but out of a station's stockpile, with the
        player's tech."""
        return max(0.0, RETAIL_MARKUP - self.research.bonus(Effect.FUEL_MARKUP))

    def fuel_cost(self, engine: str, cells: float = 1.0) -> float:
        """What an engine's fuel for `cells` costs bought at retail now, the dearest it comes."""
        burnt = ENGINES[engine]
        fuel = fuel_of(burnt)
        markup = 0.0 if fuel == Fuel.ELECTRICITY else self.fuel_markup()
        price = self.fuel.price(fuel) * (1.0 + markup)
        return burnt.burn * cells * price * self._upkeep()

    def _take_fuel(self, train: Train, station: Cell, cells: float) -> None:
        """Buys what a train setting off burns on the way to its next stop: coal, wood (logs) or
        diesel fuel out of the station's stockpile first, at the market price, the rest at retail."""
        engine = ENGINES[train.engine]
        fuel = fuel_of(engine)
        amount = engine.burn * cells
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
        self.finances.spend(cost * self._upkeep() * self._food_subsidy(train), fuel.value)

    def _food_subsidy(self, train: Train) -> float:
        """What share of its fuel a train pays with the player's tech: Effect.FOOD_FUEL less
        for a train of nothing but food, in proportion to its cars carrying food."""
        subsidy = self.research.bonus(Effect.FOOD_FUEL)
        if not subsidy or not train.cars:
            return 1.0
        food = sum(
            1
            for car in train.cars
            if car.load > 0
            and (car.good in FOODS or car.general in (General.FOOD, General.PROCESSED_FOOD))
        )
        return 1.0 - subsidy * food / len(train.cars)

    def fuel_month(self, trade: Trade, points: float = 0.0) -> None:
        """Closes the fuel market's month with what this month's trade said about it, and the
        points of the map's homes and factories, which want electricity."""
        owned = (
            None if self.owners is None else {kind: self.owners.owned(kind) for kind in PLANT_FUEL}
        )
        self.fuel.month(self.year, signals_of(trade, owned, points, self.year))

    def car_capacity(
        self, good: Good, engine: str | None = None, general: General | None = None
    ) -> float:
        """How much of a good one car carries, a general car of a kind (general_car_size) or
        one for the good alone: sizes up with its tree's tech (Effect.CAR_SIZE) and behind newer engines
        (Engine.car_levels), and coaches and mail vans more with tech."""
        base = car_size(good) if general is None else general_car_size(general, good)
        tree = GOOD_TREES.get(good)
        levels = int(self.research.bonus(Effect.CAR_SIZE, tree)) if tree is not None else 0
        if engine is not None:
            levels += ENGINES[engine].car_levels
        size = bigger(base, levels)
        if good in TRAVELLERS:
            return size * (1.0 + self.research.bonus(Effect.COACHES))
        return size

    def _upkeep(self) -> float:
        return max(LEAST_UPKEEP, 1.0 - self.research.bonus(Effect.UPKEEP))

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
        if self.facilities is not None:
            self.facilities.pop(old, None)
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
        goods.extra_room = self.research.bonus(Effect.STORAGE)
        if Facility.WAREHOUSE in self.working_at(station):
            goods.extra_room += WAREHOUSE_ROOM
        goods.extra_waiting = self.research.bonus(Effect.WAITING)
        return goods

    def station_demand(self, station: Cell) -> StationDemand:
        return self.demand.setdefault(station, StationDemand())

    def price_factor(self, good: Good) -> float:
        """How much more a good earns carried by train: its TRANSIT_PAY, with the player's tech."""
        return TRANSIT_PAY.get(good, 1.0) * (
            1.0 + self.research.bonus(Effect.PRICE, GOOD_TREES[good])
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
        another, as it would be paid, with the player's tech and the station's facilities:
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
        else:
            here = train.stops[(train.heading - 1) % len(train.stops)].cell
            if train.platform in self.constructions.platforms(here):
                here = train.platform
            there = train.stops[train.heading % len(train.stops)].cell
            leg = self.network.route_length(here, there, engine.electric, engine.maglev)
            if leg is None:
                return None
            cells = leg
        upkeep = self._upkeep()
        loaded = [car for car in train.cars if not car.dining and car.load >= 0.5]
        empty = [car for car in train.cars if car.dining or car.load < 0.5]
        return Ride(
            cells=cells,
            cargo=sum(self.cargo_worth(train) if worth is None else worth),
            fuel=self.fuel_cost(train.engine, cells),
            loaded_cars=sum(car_trip_price(car.capacity) for car in loaded) * upkeep,
            empty_cars=sum(car_trip_price(0.0 if car.dining else car.capacity) for car in empty)
            * upkeep,
            loaded=len(loaded),
            empty=len(empty),
        )

    def network_upkeep(self) -> tuple[float, float]:
        """What all the player's stations and all their track cost a month now."""
        upkeep = self._upkeep()
        stations = sum(self.station_upkeep(station) for station in self.stations())
        return stations * upkeep, track_upkeep(self.constructions.track()) * upkeep

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
        """Pays the freight for cargo trains brought that the buildings around a station took,
        which is also experience in the good's tree: its freight (freight_of) for each unit,
        times what it was worth to the building that took it (`worth`, the units counted by
        railroads.sim.economy.worth_to; the units themselves if not given), with tech and the
        station's facilities. Uses up that much of the station's demand."""
        self.station_demand(station).use(need, amount)
        weighed = amount if worth is None else worth
        earned = freight_of(good) * weighed * self.price_factor(good)
        earned *= self.facility_pay(station, good)
        self.finances.earn(earned, "cargo")
        self.learn(GOOD_TREES[good], earned)

    def sell_own(self, good: Good, amount: float) -> None:
        """Pays the player the market price of what their own buildings brought into a station:
        their company sold it, and the railway now carries it."""
        self.finances.earn(price_of(good) * amount, "owned buildings")

    def learn(self, tree: Tree, dollars: float) -> None:
        """Counts what a delivery earned as experience in a tree, more for every building of
        that tree the player owns shares of (railroads.sim.ownership.Owners.experience_factor)."""
        factor = 1.0 if self.owners is None else self.owners.experience_factor(tree)
        self.research.earn(tree, dollars * factor)

    def facility_pay(self, station: Cell, good: Good) -> float:
        """How many times as much a good delivered at a station pays with its facilities."""
        return pay_factor(self.working_at(station), good)

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
        """How many cells out a station hands goods out to, and gathers them from."""
        return self.reach_at(self.level(station))

    @staticmethod
    def reach_at(level: int) -> tuple[int, int]:
        """How many cells out a station of this level hands goods out to, a cell at the first
        level, and gathers them from; each level a cell further."""
        return FIRST_REACH + level - 1, COLLECTION_RADIUS + level - 1

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
        """What a station costs to keep a month this year, by its level, before tech cuts it."""
        rise = (1.0 + STATION_UPKEEP_RISE) ** max(0, self.year - self.first_year)
        return STATION_UPKEEP[self.level(station) - 1] * rise

    def facilities_at(self, station: Cell) -> set[Facility]:
        return (self.facilities or {}).get(station, set())

    def working_at(self, station: Cell) -> set[Facility]:
        """The facilities at a station that got what they use this month."""
        return (self.supplied or {}).get(station, set())

    def build_facility(self, station: Cell, facility: Facility) -> bool:
        """Puts up a facility beside a station, if it has none of the kind and there is the cash;
        it works from the month after, given what it uses."""
        price = FACILITY_PRICES[facility]
        if facility in self.facilities_at(station) or not self.finances.pay(
            price, "station upgrades"
        ):
            return False
        if self.facilities is None:
            self.facilities = {}
        self.facilities.setdefault(station, set()).add(facility)
        return True

    def _supply_facilities(self) -> None:
        """Lets every facility take a month of what it uses from its station."""
        self.supplied = {
            station: {f for f in facilities if supply(self.station_goods(station), f)}
            for station, facilities in (self.facilities or {}).items()
            if station in self.goods
        }

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
        self._supply_facilities()
        self.finances.close_month(self.year)
        track = self.constructions.track()
        upkeep = self._upkeep()
        stations = sum(self.station_upkeep(station) for station in self.stations())
        kept = 1.0 - self.research.bonus(Effect.STATION_UPKEEP)
        self.finances.spend(stations * upkeep * kept, "station upkeep")
        self.finances.spend(track_upkeep(track) * upkeep, "track upkeep")

    def run(self, months: float) -> None:
        """Moves every train on by this many months, then has trains that met give way."""
        for train in self.trains:
            self._run(train, months)
        self._meet()

    def route(
        self, start: Cell, end: Cell, electric: bool = False, maglev: bool = False
    ) -> tuple[list[Point], list[float]] | None:
        """The way along laid track from one station to another (TrackNetwork.route)."""
        return self.network.route(start, end, electric, maglev)

    def _run(self, train: Train, months: float) -> None:
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
                    train.status = f"Loading at {leaving.cell[0]}, {leaving.cell[1]}"
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
            self._take_fuel(train, here, leg)
            train.path, train.speeds = way
            train.along = 0.0
            train.pace = 0.0
            train.status = "Running"
            if train.cars:
                places = (0.0 if car.dining else car.capacity for car in train.cars)
                trips = sum(map(car_trip_price, places)) * self._upkeep()
                self.finances.spend(trips, "car trips")
        if train.giving_way:
            train.status = f"Giving way to {train.giving_way}"
            train.waited += months
            return
        engine = ENGINES[train.engine]
        pickup = engine.pickup / SECONDS_PER_MONTH * (SAND_TOWER_PICKUP if train.sanded else 1.0)
        pace = train.speed_up(months, pickup)
        speed = engine.speed * (1.0 + self.research.bonus(Effect.SPEED)) * pace
        if train.fuelled:
            speed *= ENGINE_SHED_SPEED
        train.along += months * CELLS_PER_MONTH * self.network.speed_at(train) * train.pull * speed
        train.status = "Running"
        if train.along >= self.network.length(train):
            train.platform = self.network.cell_at(train.path[-1])
            self.calls.arrive(train, train.stops[train.heading % len(train.stops)])
            train.arrivals += 1
            train.heading = (train.heading + 1) % len(train.stops)
            train.path = []
            train.pace = 0.0
            train.waited = 0.0

    def _meet(self) -> None:
        """Has trains that met give way: of each two running into each other, the one that
        doesn't go first stops where it is; a train that stopped sets off again once nothing
        with the right of way is in its way any more."""
        for train in self.trains:
            if len(train.path) < 2:
                train.giving_way = ""
        running = [train for train in self.trains if len(train.path) >= 2]
        if not running:
            return
        bought = {id(train): index for index, train in enumerate(self.trains)}
        rank = {id(train): _right_of_way(train, bought[id(train)]) for train in running}
        running.sort(key=lambda train: rank[id(train)])
        bodies = [self.network.body(train, MEET_STEP) for train in running]
        met = self._met(bodies)
        for i, train in enumerate(running):
            if train.giving_way:
                continue
            for j in met[i]:
                other = running[j]
                if j > i and not other.giving_way:
                    other.giving_way = train.name
                    other.pace = 0.0
        for i, train in enumerate(running):
            if not train.giving_way:
                continue
            ahead = [running[j] for j in met[i] if j < i or not running[j].giving_way]
            train.giving_way = ahead[0].name if ahead else ""

    def _met(self, bodies: list[list[tuple[Point, Point, Point]]]) -> list[list[int]]:
        """For each running train, as points along it (TrackNetwork.body), the others it has
        met somewhere they can't pass each other."""
        met: list[list[int]] = [[] for _ in bodies]
        boxes = []
        points = []
        for body in bodies:
            xs = [at[0] for at, _, _ in body]
            ys = [at[1] for at, _, _ in body]
            boxes.append((min(xs), min(ys), max(xs), max(ys)) if body else None)
            points.append(np.array([at for at, _, _ in body]))
        reach = MEET_DISTANCE
        for i, j in itertools.combinations(range(len(bodies)), 2):
            one, other = boxes[i], boxes[j]
            if one is None or other is None:
                continue
            if (
                one[0] - reach > other[2]
                or other[0] - reach > one[2]
                or one[1] - reach > other[3]
                or other[1] - reach > one[3]
            ):
                continue
            if self._in_the_way(bodies[i], bodies[j], points[i], points[j]):
                met[i].append(j)
                met[j].append(i)
        return met

    def _in_the_way(
        self,
        one: list[tuple[Point, Point, Point]],
        other: list[tuple[Point, Point, Point]],
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
    ) -> bool:
        """Whether two running trains, as points along them (TrackNetwork.body) and those
        points as arrays, have met somewhere they can't pass each other."""
        close = np.argwhere(
            np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1)) < MEET_DISTANCE
        )
        return any(not self._can_pass(one[i], other[j]) for i, j in close.tolist())

    def _can_pass(self, one: tuple[Point, Point, Point], other: tuple[Point, Point, Point]) -> bool:
        """Whether two trains can pass each other here: within a station, or on double track
        going opposite ways."""
        (_, way, middle), (_, other_way, other_middle) = one, other
        in_station, double = self.network.passing(middle)
        other_in_station, other_double = self.network.passing(other_middle)
        if in_station and other_in_station:
            return True
        opposite = way[0] * other_way[0] + way[1] * other_way[1] < 0
        return opposite and double and other_double

    def position(self, train: Train, back: float = 0.0) -> tuple[Point, Point] | None:
        """Where the train is on the map, and which way it faces (TrackNetwork.position)."""
        return self.network.position(train, back)


def _right_of_way(train: Train, bought: int) -> tuple[float, float, int]:
    """What decides which of two trains that meet goes first, the smaller first: having stood
    longest letting others by on the way, then the better engine, then being bought first
    (`bought` is its place among the trains)."""
    return -train.waited, -ENGINES[train.engine].speed, bought
