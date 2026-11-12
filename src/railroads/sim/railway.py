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
stockpile, up to STOCKPILE of each good. Each level a station is raised to widens both by a cell,
and each storage upgrade adds STORAGE_STEP of each good to its stockpile and WAITING_STEP to its
waiting passengers and mail.

Cargo earns money by the demand around the station that hands it out (railroads.sim.demand). Trains
cost TRAIN_PRICE, every car CAR_TRIP_PRICE each time a train leaves a stop with it, and every
month stations and track cost their upkeep (railroads.sim.finance).

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
from collections.abc import Mapping

import numpy as np

from railroads.places import StationNames
from railroads.sim.calls import StationCalls
from railroads.sim.clock import SECONDS_PER_MONTH, START_YEAR
from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.demand import PASSED_ALONG_PAY, StationDemand
from railroads.sim.economy import COLLECTION_RADIUS, DELIVERY_RADIUS
from railroads.sim.finance import (
    CAR_TRIP_PRICE,
    LEVEL_PRICES,
    STATION_UPKEEP,
    STORAGE_PRICE,
    TRAIN_RESALE,
    Finances,
    track_upkeep,
)
from railroads.sim.goods import TRAVELLERS, Good, Need
from railroads.sim.grids import Cell
from railroads.sim.station_goods import StationGoods
from railroads.sim.stats import History
from railroads.sim.tech import GOOD_TREES, Effect, Research
from railroads.sim.track_network import TrackNetwork
from railroads.sim.trains import CAR_CAPACITY, ENGINES, FIRST_ENGINE, Point, Train

TRAIN_COST_NAME = "Steam engine"
# How far a train runs in a month, in cells, on flat ground.
CELLS_PER_MONTH = 60.0
# Trains meet where points along them, MEET_STEP cells apart, come within MEET_DISTANCE cells.
MEET_STEP = 0.2
MEET_DISTANCE = 0.2
MAX_LEVEL = 1 + len(LEVEL_PRICES)
# Upkeep never falls below this share of its full cost, however much tech cuts it.
LEAST_UPKEEP = 0.5


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
        self._bought = 0
        # What the player has learned (railroads.sim.tech).
        self.research = Research()
        # What each year ended with and the cars delivered in it (railroads.sim.stats).
        self.history = History()
        # The way along the track, and what trains do calling at stations
        # (railroads.sim.track_network, railroads.sim.calls).
        self.network = TrackNetwork(constructions)
        self.calls = StationCalls(self)

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
        self._bought += 1
        train = Train(f"{TRAIN_COST_NAME} {self._bought}", engine=engine)
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

    def destination(self, train: Train) -> Cell | None:
        """The stop the cars a train pulls now are taken to: the one it is heading for, or the
        next one from the stop it stands at."""
        if len(train.stops) < 2:
            return None
        return train.stops[train.heading % len(train.stops)].cell

    def cargo_worth(self, train: Train) -> list[float]:
        """What each of a train's cars would earn if it were unloaded at its destination now,
        at the demand left there and the player's tech; cargo that came from there earns
        nothing, as it only goes back to that station's stockpile."""
        destination = self.destination(train)
        if destination is None:
            return [0.0 for _ in train.cars]
        cargo = [(car.good, car.load - car.origins.get(destination, 0.0)) for car in train.cars]
        earned = self.station_demand(destination).worth(cargo)
        return [
            amount * self.price_factor(car.good)
            for amount, car in zip(earned, train.cars, strict=True)
        ]

    def pay_for(self, station: Cell, need: Need, good: Good, amount: float) -> None:
        """Pays for cargo trains brought that the buildings around a station took."""
        earned = self.station_demand(station).pay(need, good, amount) * self.price_factor(good)
        self.finances.earn(earned, "cargo")

    def pay_for_passing(self, station: Cell, need: Need, good: Good, amount: float) -> None:
        """Pays for goods a station gathered itself and passed along to the buildings around it."""
        earned = self.station_demand(station).pay(need, good, amount, PASSED_ALONG_PAY)
        self.finances.earn(earned, "goods passed along")

    def connected(self) -> set[Cell]:
        """The stations that track joins to at least one other station."""
        return self.network.connected()

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
        """Adds STORAGE_STEP of each good to the station's stockpile and WAITING_STEP to its waiting
        room, if there is the cash."""
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
        """Moves every train on by this many months, then has trains that met give way."""
        for train in self.trains:
            self._run(train, months)
        self._meet()

    def route(
        self, start: Cell, end: Cell, electric: bool = False
    ) -> tuple[list[Point], list[float]] | None:
        """The way along laid track from one station to another (TrackNetwork.route)."""
        return self.network.route(start, end, electric)

    def _run(self, train: Train, months: float) -> None:
        if len(train.stops) < 2:
            train.status = "Needs two stops"
            train.path = []
            return
        if not train.path:
            leaving = train.stops[(train.heading - 1) % len(train.stops)]
            if not train.called:
                self.calls.arrive(train, leaving)
            if not leaving.wait_for_load:
                train.waiting = 0.0
            if train.waiting > 0:
                self.calls.load(train, leaving)
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
            train.pace = 0.0
            train.status = "Running"
            if train.cars:
                trips = CAR_TRIP_PRICE * len(train.cars) * self._upkeep()
                self.finances.spend(trips, "car trips")
        if train.giving_way:
            train.status = f"Giving way to {train.giving_way}"
            train.waited += months
            return
        engine = ENGINES[train.engine]
        pace = train.speed_up(months, engine.pickup / SECONDS_PER_MONTH)
        speed = engine.speed * (1.0 + self.research.bonus(Effect.SPEED)) * pace
        train.along += months * CELLS_PER_MONTH * self.network.speed_at(train) * train.pull * speed
        train.status = "Running"
        if train.along >= self.network.length(train.path):
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
        rank = {id(train): self._right_of_way(train) for train in running}
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
        for body in bodies:
            xs = [at[0] for at, _, _ in body]
            ys = [at[1] for at, _, _ in body]
            boxes.append((min(xs), min(ys), max(xs), max(ys)) if body else None)
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
            if self._in_the_way(bodies[i], bodies[j]):
                met[i].append(j)
                met[j].append(i)
        return met

    def _right_of_way(self, train: Train) -> tuple[float, float, int]:
        """What decides which of two trains that meet goes first, the smaller first: having
        stood longest letting others by on the way, then the better engine, then being bought
        first."""
        return -train.waited, -ENGINES[train.engine].speed, self.trains.index(train)

    def _in_the_way(
        self,
        one: list[tuple[Point, Point, Point]],
        other: list[tuple[Point, Point, Point]],
    ) -> bool:
        """Whether two running trains, as points along them (TrackNetwork.body), have met
        somewhere they can't pass each other."""
        a = np.array([at for at, _, _ in one])
        b = np.array([at for at, _, _ in other])
        close = np.argwhere(
            np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1)) < MEET_DISTANCE
        )
        return any(not self._can_pass(one[i], other[j]) for i, j in close.tolist())

    def _can_pass(self, one: tuple[Point, Point, Point], other: tuple[Point, Point, Point]) -> bool:
        """Whether two trains can pass each other here: within a station, or on double track
        going opposite ways."""
        (_, way, middle), (_, other_way, other_middle) = one, other
        cells = [self._cell_at(middle), self._cell_at(other_middle)]
        if all(self.constructions.station_at(cell) is not None for cell in cells):
            return True
        opposite = way[0] * other_way[0] + way[1] * other_way[1] < 0
        return opposite and all(
            any(
                isinstance(piece, Track) and piece.width == TrackWidth.DOUBLE
                for piece in self.constructions.at(cell)
            )
            for cell in cells
        )

    def _cell_at(self, point: Point) -> Cell:
        return self.network.cell_at(point)

    def position(self, train: Train, back: float = 0.0) -> tuple[Point, Point] | None:
        """Where the train is on the map, and which way it faces (TrackNetwork.position)."""
        return self.network.position(train, back)
