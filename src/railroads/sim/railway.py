"""Trains: they run along laid track from stop to stop, starting at the first. Each stop says
which cars, up to MAX_CARS, the train pulls on to the next one; each car carries one good, up to
CAR_CAPACITY of it.

At each stop a train unloads its cars into the station, freight to be handed out or carried on
by another train and passengers and mail to go on their way, swaps them for that stop's cars and
fills those from the station's stockpile and from freight the buildings around it have no use
for. Freight remembers where it came from: it earns money only when buildings at another station
take it, and back at its own station it just returns to the stockpile. Stations deal with the buildings around them once a month, when goods
are shared out (railroads.sim.economy): they hand what trains brought to buildings within
DELIVERY_RADIUS cells, and take every spare good made within COLLECTION_RADIUS cells into their
stockpile, up to STOCKPILE goods in all. Each level a station is raised to widens both by a cell,
and each storage upgrade adds STORAGE_STEP to its stockpile and to its waiting passengers and
mail.

Cargo earns money by the demand around the station that hands it out (railroads.sim.demand). Trains
cost TRAIN_PRICE, every car CAR_TRIP_PRICE each time a train leaves a stop with it, and every
month stations and track cost their upkeep (railroads.sim.finance).
"""

import heapq
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from railroads.places import StationNames
from railroads.sim.clock import START_YEAR
from railroads.sim.constructions import Constructions, Station, Track
from railroads.sim.demand import PASSED_ALONG_PAY, StationDemand
from railroads.sim.economy import COLLECTION_RADIUS, DELIVERY_RADIUS
from railroads.sim.finance import (
    CAR_TRIP_PRICE,
    LEVEL_PRICES,
    STATION_UPKEEP,
    STORAGE_PRICE,
    TRAIN_PRICE,
    TRAIN_RESALE,
    Finances,
    track_upkeep,
)
from railroads.sim.goods import TRAVELLERS, Good, Need
from railroads.sim.grids import Cell
from railroads.sim.stats import History
from railroads.sim.tech import GOOD_TREES, Effect, Research, Tree
from railroads.sim.track_rules import track_rules_for

TRAIN_COST_NAME = "Steam engine"
# How far a train runs in a month, in cells, on flat ground.
CELLS_PER_MONTH = 60.0
MAX_CARS = 5
# A full car weighs this much more than an empty one, which counts 1; each car's weight slows the
# train by DRAG_PER_CAR of the engine's speed alone, so six full cars still make about half of it.
FULL_LOAD = 1.0
DRAG_PER_CAR = 0.08
CAR_CAPACITY = 50.0
# Longest a train waits at a stop for its cars to fill, in months.
WAIT_MONTHS = 1.0
# Most freight a station's stockpile holds, of all kinds together, and most passengers, and mail,
# waiting there.
STOCKPILE = 1000.0
WAITING = 100.0
STORAGE_STEP = 500
MAX_LEVEL = 1 + len(LEVEL_PRICES)
# Upkeep never falls below this share of its full cost, however much tech cuts it.
LEAST_UPKEEP = 0.5


@dataclass(frozen=True, slots=True)
class Engine:
    """An engine model: from what year it can be bought, how fast it runs against the first,
    how many cars it pulls, what it costs, and whether it runs only on electrified track. All but
    the first also need a tech node (railroads.sim.tech) to open."""

    title: str
    year: int
    speed: float
    cars: int
    price: float
    electric: bool = False


ENGINES: dict[str, Engine] = {
    "planet": Engine("Planet steam engine", 1850, 1.0, MAX_CARS, TRAIN_PRICE),
    "mogul": Engine("Mogul freight engine", 1880, 1.0, 8, 12_000.0),
    "pacific": Engine("Pacific express engine", 1910, 1.5, 8, 25_000.0),
    "boxcab": Engine("Boxcab electric", 1925, 1.4, 10, 40_000.0, electric=True),
    "express": Engine("Express electric", 1965, 2.2, 12, 90_000.0, electric=True),
}
FIRST_ENGINE = "planet"

type Point = tuple[float, float]
# How much of some cargo came from each station; None where that is not known.
type Parcels = dict[Cell | None, float]


def _take(parcels: Parcels, amount: float, total: float) -> Parcels:
    """Takes `amount` evenly out of parcels of cargo holding `total` in all."""
    if total <= 0 or amount <= 0:
        return {}
    share = min(1.0, amount / total)
    taken = {origin: held * share for origin, held in parcels.items() if held > 0}
    for origin, part in taken.items():
        parcels[origin] -= part
    unknown = amount - sum(taken.values())
    if unknown > 1e-9:
        taken[None] = taken.get(None, 0.0) + unknown
    return taken


@dataclass
class Car:
    """A car for one good, how much of it it carries, and where that came from."""

    good: Good
    load: float = 0.0
    origins: Parcels = field(default_factory=dict)
    capacity: float = CAR_CAPACITY

    def fill(self, parcels: Parcels) -> None:
        for origin, amount in parcels.items():
            self.origins[origin] = self.origins.get(origin, 0.0) + amount
            self.load += amount

    def empty(self, amount: float) -> Parcels:
        taken = _take(self.origins, amount, self.load)
        self.load -= amount
        return taken


@dataclass
class Stop:
    """A station a train calls at, and the goods of the cars it pulls when it leaves for the
    next stop."""

    cell: Cell
    cars: list[Good] = field(default_factory=list)
    # Whether the train waits here, up to WAIT_MONTHS, for its cars to fill.
    wait_for_load: bool = True

    def add_car(self, good: Good, most: int = MAX_CARS) -> bool:
        if len(self.cars) >= most:
            return False
        self.cars.append(good)
        return True


@dataclass
class Train:
    name: str
    stops: list[Stop] = field(default_factory=list)
    # The cars it pulls now.
    cars: list[Car] = field(default_factory=list)
    # Which stop it is heading for, the way there as points on the map, and how far along it is.
    heading: int = 1
    # Whether it has called at the stop it is leaving, taking on that stop's cars, and how many
    # stops it has reached since it last set off from its first.
    called: bool = False
    arrivals: int = 0
    # How much longer it will wait at the stop it is leaving for its cars to fill, in months.
    waiting: float = 0.0
    # The station whose cars it pulls, where they were filled.
    loaded_at: Cell | None = None
    path: list[Point] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    along: float = 0.0
    status: str = "Waiting for stops"
    engine: str = FIRST_ENGINE

    @property
    def load(self) -> dict[Good, float]:
        """What the train carries, by good."""
        carried: dict[Good, float] = {}
        for car in self.cars:
            if car.load > 0:
                carried[car.good] = carried.get(car.good, 0.0) + car.load
        return carried

    @property
    def weight(self) -> float:
        """How heavy the cars are, in empty cars: an empty car weighs 1, a full one 1 + FULL_LOAD."""
        return sum(1.0 + FULL_LOAD * car.load / car.capacity for car in self.cars)

    @property
    def pull(self) -> float:
        """How much of the engine's top speed the train makes with its cars behind it."""
        return 1.0 / (1.0 + DRAG_PER_CAR * self.weight)

    @property
    def full(self) -> bool:
        return all(car.load >= car.capacity - 1e-9 for car in self.cars)

    def restart(self) -> None:
        """Sends the train back to its first stop after its stops change."""
        self.heading = 1
        self.called = False
        self.arrivals = 0
        self.waiting = 0.0
        self.path = []

    def cars_changed(self) -> None:
        """Takes in new cars at a stop. A train standing at a stop calls there again to swap
        its cars, and one still on its first trip goes back to its first stop for them, so the
        cars set up after the stops still carry goods that trip; later on, changes show at the
        next call."""
        if not self.path:
            self.called = False
            self.waiting = 0.0
        elif self.arrivals == 0:
            self.restart()


@dataclass
class StationGoods:
    """What a station holds: goods trains brought, waiting to be handed out, and its stockpile of
    spare goods from around it, waiting for trains."""

    brought: dict[Good, float] = field(default_factory=dict)
    stockpile: dict[Good, float] = field(default_factory=dict)
    # How many storage upgrades the station has had.
    storage: int = 0
    # Where each good trains brought came from.
    origins: dict[Good, Parcels] = field(default_factory=dict)
    # Room tech adds to every station's stockpile and waiting room (railroads.sim.tech).
    extra_room: float = 0.0
    extra_waiting: float = 0.0

    @property
    def limit(self) -> float:
        """The most freight the stockpile holds."""
        return STOCKPILE + STORAGE_STEP * self.storage + self.extra_room

    @property
    def waiting_limit(self) -> float:
        """The most passengers, and the most mail, that wait here."""
        return WAITING + STORAGE_STEP * self.storage + self.extra_waiting

    @property
    def room(self) -> float:
        return max(0.0, self.limit - self.stocked)

    @property
    def stocked(self) -> float:
        """How much freight the station holds, gathered and brought together; passengers and
        mail wait apart."""
        held = [*self.stockpile.items(), *self.brought.items()]
        return sum(amount for good, amount in held if good not in TRAVELLERS)

    def bring(self, good: Good, parcels: Parcels) -> None:
        """Takes in cargo a train unloaded."""
        known = self.origins.setdefault(good, {})
        for origin, amount in parcels.items():
            known[origin] = known.get(origin, 0.0) + amount
            self.brought[good] = self.brought.get(good, 0.0) + amount

    def release(self, good: Good, amount: float) -> Parcels:
        """Lets go of cargo trains brought, handed out or loaded onto another train."""
        held = self.brought.get(good, 0.0)
        amount = min(amount, held)
        self.brought[good] = held - amount
        return _take(self.origins.setdefault(good, {}), amount, held)

    def board(self, good: Good, amount: float) -> None:
        """Lets passengers or mail come to the station to wait, up to its waiting limit."""
        self.stockpile[good] = min(self.waiting_limit, self.stockpile.get(good, 0.0) + amount)


class Railway:
    """The trains on one map, and the goods held at its stations."""

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
        self._names = itertools.count(1)
        # What the player has learned (railroads.sim.tech).
        self.research = Research()
        # What each year ended with and the cars delivered in it (railroads.sim.stats).
        self.history = History()
        self._connected: tuple[int, set[Cell]] | None = None

    def stations(self) -> list[Cell]:
        return self.constructions.stations()

    def engines(self) -> list[str]:
        """The engines the player can buy now: those whose year has come and whose tech
        they have."""
        opened = self.research.engines() | {FIRST_ENGINE}
        return [name for name, e in ENGINES.items() if e.year <= self.year and name in opened]

    def buy(self, engine: str = FIRST_ENGINE) -> Train | None:
        """Buys a train with an engine the player can buy; None without the cash for it."""
        if engine not in self.engines():
            return None
        price = ENGINES[engine].price
        if not self.finances.can_afford(price):
            return None
        self.finances.spend(price, "trains")
        train = Train(f"{TRAIN_COST_NAME} {next(self._names)}", engine=engine)
        self.trains.append(train)
        return train

    def sell(self, train: Train) -> None:
        self.trains.remove(train)
        self.finances.earn(ENGINES[train.engine].price * TRAIN_RESALE, "trains")

    def most_cars(self, train: Train) -> int:
        """How many cars a train's engine pulls, with what tech adds."""
        return ENGINES[train.engine].cars + int(self.research.bonus(Effect.PULL))

    def car_capacity(self, good: Good) -> float:
        """How much of a good one car carries: coaches and mail vans more with tech."""
        if good in TRAVELLERS:
            return CAR_CAPACITY * (1.0 + self.research.bonus(Effect.COACHES))
        return CAR_CAPACITY

    def _upkeep(self) -> float:
        return max(LEAST_UPKEEP, 1.0 - self.research.bonus(Effect.UPKEEP))

    def moved(self, old: Cell, new: Cell) -> None:
        """Carries a station's name, goods, demand and trains over to where it now stands; its
        level and storage start again from nothing. Trains calling there start their rounds
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
        goods.extra_waiting = self.research.bonus(Effect.WAITING)
        return goods

    def station_demand(self, station: Cell) -> StationDemand:
        return self.demand.setdefault(station, StationDemand())

    def price_factor(self, good: Good) -> float:
        """How much more a good earns with the player's tech."""
        return 1.0 + self.research.bonus(Effect.PRICE, GOOD_TREES[good])

    def pay_for(self, station: Cell, need: Need, good: Good, amount: float) -> None:
        """Pays for cargo trains brought that the buildings around a station took."""
        earned = self.station_demand(station).pay(need, good, amount) * self.price_factor(good)
        self.finances.earn(earned, "cargo")

    def pay_for_passing(self, station: Cell, need: Need, good: Good, amount: float) -> None:
        """Pays for goods a station gathered itself and passed along to the buildings around it."""
        earned = self.station_demand(station).pay(need, good, amount, PASSED_ALONG_PAY)
        self.finances.earn(earned, "goods passed along")

    def connected(self) -> set[Cell]:
        """The stations that track joins to at least one other station, worked out again only
        when track or stations have changed."""
        version = self.constructions.railway_version
        if self._connected is None or self._connected[0] != version:
            self._connected = (version, self._join_stations())
        return set(self._connected[1])

    def _join_stations(self) -> set[Cell]:
        stations = set(self.stations())
        joined: set[Cell] = set()
        for station in stations:
            if station not in joined and (other := self._next_station(station, stations)):
                joined |= {station, other}
        return joined

    def _next_station(self, start: Cell, stations: set[Cell]) -> Cell | None:
        """Another station track leads to from this one, if there is any."""
        grid = self.constructions.tile_map.grid
        frontier: list[tuple[Cell, int | None]] = [(start, None)]
        seen = set(frontier)
        while frontier:
            cell, entry = frontier.pop()
            for exit_point in self._exits(cell, entry):
                nxt, nxt_entry = grid.track_exit(cell, exit_point)
                if not grid.contains(*nxt) or not self._joins(nxt, nxt_entry):
                    continue
                if nxt in stations and nxt != start:
                    return nxt
                if (nxt, nxt_entry) not in seen:
                    seen.add((nxt, nxt_entry))
                    frontier.append((nxt, nxt_entry))
        return None

    def level(self, station: Cell) -> int:
        return self.levels.get(station, 1)

    def reach(self, station: Cell) -> tuple[int, int]:
        """How many cells out a station hands goods out to, and gathers them from."""
        return self.reach_at(self.level(station))

    @staticmethod
    def reach_at(level: int) -> tuple[int, int]:
        """How many cells out a station of this level hands goods out to, and gathers them from."""
        return DELIVERY_RADIUS + level - 1, COLLECTION_RADIUS + level - 1

    def upgrade_price(self, station: Cell) -> float | None:
        """What raising the station a level costs; None at the top level."""
        level = self.level(station)
        return LEVEL_PRICES[level - 1] if level < MAX_LEVEL else None

    def upgrade(self, station: Cell) -> bool:
        """Raises the station a level, if it can go higher and there is the cash for it."""
        price = self.upgrade_price(station)
        if price is None or not self.finances.can_afford(price):
            return False
        self.finances.spend(price, "station upgrades")
        self.levels[station] = self.level(station) + 1
        return True

    def add_storage(self, station: Cell) -> bool:
        """Adds STORAGE_STEP to the station's stockpile and waiting room, if there is the cash."""
        if not self.finances.can_afford(STORAGE_PRICE):
            return False
        self.finances.spend(STORAGE_PRICE, "station upgrades")
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
        self.finances.close_month()
        track = self.constructions.track()
        upkeep = self._upkeep()
        self.finances.spend(STATION_UPKEEP * len(self.stations()) * upkeep, "station upkeep")
        self.finances.spend(track_upkeep(track) * upkeep, "track upkeep")

    def run(self, months: float) -> None:
        """Moves every train on by this many months."""
        for train in self.trains:
            self._run(train, months)

    def route(
        self, start: Cell, end: Cell, electric: bool = False
    ) -> tuple[list[Point], list[float]] | None:
        """The way along laid track, only electrified track if `electric`, from one station to
        another, as points on the map and the speed on each stretch between them; None without
        track joining them. Trains pass through stations on the way, which join track at any of
        their sides, and may turn round in them."""
        grid = self.constructions.tile_map.grid
        steps = grid.track_steps()
        counter = itertools.count()
        frontier: list[tuple[float, int, Cell, int | None, tuple]] = [
            (0.0, next(counter), start, None, ())
        ]
        seen: set[tuple[Cell, int | None]] = set()
        while frontier:
            cost, _, cell, entry, trail = heapq.heappop(frontier)
            if cell == end and trail:
                return self._way(trail, (cell, entry or 0))
            if (cell, entry) in seen:
                continue
            seen.add((cell, entry))
            for exit_point in self._exits(cell, entry, electric):
                nxt, nxt_entry = grid.track_exit(cell, exit_point)
                if not grid.contains(*nxt) or not (nxt == end or self._joins(nxt, nxt_entry)):
                    continue
                length = float(math.hypot(*steps[exit_point]))
                heapq.heappush(
                    frontier,
                    (cost + length, next(counter), nxt, nxt_entry, (*trail, (cell, exit_point))),
                )
        return None

    def _exits(self, cell: Cell, entry: int | None, electric: bool = False) -> list[int]:
        """The points a train that came in by `entry` can leave the cell by: any side of a
        station, the one it came in by too, turning round there; else the other end of each
        piece of track, electrified if `electric`, that starts where it came in."""
        built = self.constructions.at(cell)
        if any(isinstance(b, Station) for b in built):
            return list(range(len(self.constructions.tile_map.grid.track_points())))
        return [
            piece.ends[1] if piece.ends[0] == entry else piece.ends[0]
            for piece in built
            if isinstance(piece, Track) and entry in piece.ends and (not electric or piece.electric)
        ]

    def _joins(self, cell: Cell, entry: int) -> bool:
        """Whether a train coming in by `entry` can go on from this cell."""
        return any(
            isinstance(b, Station) or (isinstance(b, Track) and entry in b.ends)
            for b in self.constructions.at(cell)
        )

    def _way(self, trail: tuple, arrival: tuple[Cell, int]) -> tuple[list[Point], list[float]]:
        grid = self.constructions.tile_map.grid
        steps = grid.track_steps()
        start_cell = trail[0][0]
        points: list[Point] = [grid.center(*start_cell)]
        speeds: list[float] = []
        for cell, exit_point in trail:
            cx, cy = grid.center(*cell)
            sx, sy = steps[exit_point] / 2
            if cell != start_cell:
                points.append((float(cx), float(cy)))
                speeds.append(self._speed(cell))
            points.append((float(cx + sx), float(cy + sy)))
            speeds.append(self._speed(cell))
        points.append(tuple(float(v) for v in grid.center(*arrival[0])))  # type: ignore[arg-type]
        speeds.append(1.0)
        return points, speeds

    def _speed(self, cell: Cell) -> float:
        tile = self.constructions.tile_map.tile(cell)
        pieces = [b for b in self.constructions.at(cell) if isinstance(b, Track)]
        if not pieces:
            return 1.0
        return max(0.2, track_rules_for(tile, pieces[0].bed).speed_factor)

    def _run(self, train: Train, months: float) -> None:
        if len(train.stops) < 2:
            train.status = "Needs two stops"
            train.path = []
            return
        if not train.path:
            leaving = train.stops[(train.heading - 1) % len(train.stops)]
            if not train.called:
                self._arrive(train, leaving)
            if train.waiting > 0:
                self._load(train, leaving)
                if train.full:
                    train.waiting = 0.0
                else:
                    train.waiting -= months
                    train.status = f"Loading at {leaving.cell[0]}, {leaving.cell[1]}"
                    return
            here, there = leaving.cell, train.stops[train.heading % len(train.stops)].cell
            electric = ENGINES[train.engine].electric
            way = self.route(here, there, electric)
            if way is None:
                train.status = (
                    "No electrified track to its next stop"
                    if electric
                    else "No track to its next stop"
                )
                return
            train.path, train.speeds = way
            train.along = 0.0
            train.status = "Running"
            if train.cars:
                trips = CAR_TRIP_PRICE * len(train.cars) * self._upkeep()
                self.finances.spend(trips, "car trips")
        speed = ENGINES[train.engine].speed * (1.0 + self.research.bonus(Effect.SPEED))
        train.along += months * CELLS_PER_MONTH * self._speed_at(train) * train.pull * speed
        if train.along >= self._length(train.path):
            self._arrive(train, train.stops[train.heading % len(train.stops)])
            train.arrivals += 1
            train.heading = (train.heading + 1) % len(train.stops)
            train.path = []

    def _arrive(self, train: Train, stop: Stop) -> None:
        """Calls at a stop: unloads every car, takes on the stop's cars and fills them."""
        train.called = True
        if not any(isinstance(b, Station) for b in self.constructions.at(stop.cell)):
            train.status = "Its stop has no station"
            return
        goods = self.station_goods(stop.cell)
        if train.loaded_at == stop.cell:
            train.cars = self._swap_cars(train.cars, stop.cars, goods, stop.cell)
        else:
            demand = self.station_demand(stop.cell)
            for car in train.cars:
                if car.load <= 0:
                    continue
                if car.good in TRAVELLERS:
                    earned = demand.deliver(car.good, car.load) * self.price_factor(car.good)
                    self.finances.earn(earned, car.good.value)
                    self.research.earn(Tree.PASSENGERS, earned)
                    car.empty(car.load)
                    self.history.deliver(self.year, car.good)
                    continue
                unloaded = min(car.load, goods.room)
                if unloaded > 0:
                    self._store(goods, stop.cell, car.good, car.empty(unloaded))
                    self.history.deliver(self.year, car.good)
            train.cars = self._next_cars(train.cars, stop.cars)
        train.loaded_at = stop.cell
        self._load(train, stop)
        if stop.wait_for_load and train.cars and not train.full:
            train.waiting = WAIT_MONTHS

    def _next_cars(self, cars: list[Car], wanted: list[Good]) -> list[Car]:
        """The cars for the next leg: each keeps what the fullest car of its good still holds,
        freight that did not fit into the station; anything else left aboard is lost."""
        left = sorted(cars, key=lambda car: -car.load)
        kept = []
        for good in wanted:
            car = next((car for car in left if car.good == good), None)
            if car is None:
                kept.append(Car(good, capacity=self.car_capacity(good)))
            else:
                left.remove(car)
                kept.append(car)
        return kept

    def _swap_cars(
        self, cars: list[Car], wanted: list[Good], goods: StationGoods, station: Cell
    ) -> list[Car]:
        """Swaps a train's cars for the wanted ones at the station that filled them: each wanted
        car keeps the fullest car of its good, and what dropped cars held goes back into the
        stockpile."""
        left = sorted(cars, key=lambda car: -car.load)
        kept = []
        for good in wanted:
            car = next((car for car in left if car.good == good), None)
            if car is None:
                kept.append(Car(good, capacity=self.car_capacity(good)))
            else:
                left.remove(car)
                kept.append(car)
        for car in left:
            if car.load > 0:
                Railway._store(goods, station, car.good, car.empty(car.load))
        return kept

    @staticmethod
    def _store(goods: StationGoods, station: Cell, good: Good, parcels: Parcels) -> None:
        """Puts unloaded cargo in the station: what came from here goes back in its stockpile
        for nothing, the rest waits as goods brought."""
        home = parcels.pop(station, 0.0)
        if home > 0:
            goods.stockpile[good] = goods.stockpile.get(good, 0.0) + home
        if parcels:
            goods.bring(good, parcels)

    def _load(self, train: Train, stop: Stop) -> None:
        """Tops each car up from the stop's stockpile, then with goods trains brought that the
        buildings around it have no use for this month."""
        goods = self.station_goods(stop.cell)
        for car in train.cars:
            take = min(car.capacity - car.load, goods.stockpile.get(car.good, 0.0))
            if take > 0:
                car.fill({stop.cell: take})
                goods.stockpile[car.good] -= take
            spare = goods.brought.get(car.good, 0.0)
            spare -= self.station_demand(stop.cell).monthly_want(car.good)
            transfer = min(car.capacity - car.load, spare)
            if transfer > 0:
                car.fill(goods.release(car.good, transfer))

    def position(self, train: Train, back: float = 0.0) -> tuple[Point, Point] | None:
        """Where the train is on the map, or a point `back` cells behind its front, and which way
        it faces there, while it runs; None for a point still inside the station it left."""
        if len(train.path) < 2 or train.along - back < 0:
            return None
        left = train.along - back
        for a, b in itertools.pairwise(train.path):
            length = math.dist(a, b)
            if left <= length or b == train.path[-1]:
                share = min(1.0, left / length) if length else 0.0
                at = (a[0] + (b[0] - a[0]) * share, a[1] + (b[1] - a[1]) * share)
                return at, (b[0] - a[0], b[1] - a[1])
            left -= length
        return None

    def _speed_at(self, train: Train) -> float:
        left = train.along
        for (a, b), speed in zip(itertools.pairwise(train.path), train.speeds, strict=False):
            length = math.dist(a, b)
            if left <= length:
                return speed
            left -= length
        return 1.0

    @staticmethod
    def _length(path: list[Point]) -> float:
        return sum(math.dist(a, b) for a, b in itertools.pairwise(path))
