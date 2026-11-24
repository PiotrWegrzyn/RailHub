"""Schedules: the round a group of trains runs, set once for all of them. A schedule holds the
stops the player gives it, each with the cars taken on there, and every train belongs to one. Each
train keeps its own copy of the stops (Train.stops), with as many of each stop's cars, from the
left, as its engine has room for and can pull. It takes up a schedule's changed stops once it is
back at its first stop, or at once while it is not out on a round.

Engines burning coal or diesel fuel run only as far as their fuel lasts (Engine.tank). Unless the
player set them by hand (Schedule.manual), the game puts in a schedule's refuelling stops: between
each two of the player's stops, the fewest stations, then the shortest way, with no leg longer
than the tank of the weakest train that can make the whole round. Trains with smaller tanks wait
where their fuel would not last. They are planned again whenever the player's stops, the tanks of
its trains or the track change.

A train moved to another schedule, or whose schedule now starts somewhere else, finishes its round,
then leaves the map and comes in at the new first stop after as long as the way there takes at its
top speed (Railway.way_months), so a train is never anywhere sooner than it could have run there."""

import heapq
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from railroads.sim.goods import Good
from railroads.sim.grids import Cell
from railroads.sim.trains import ENGINES, FIRST_ENGINE, Cargo, Stop, Train

if TYPE_CHECKING:
    from railroads.sim.railway import Railway

# Between the first and last station of a schedule's name.
DASH = " – "
# The most cars a schedule's stop holds: as many as the engine pulling the most in the game, so a
# schedule need not change as bigger engines come on sale. Each train takes as many of them as it
# can (Schedules.fitted).
MOST_CARS = max(engine.cars for engine in ENGINES.values())
# How many times a refuelling plan is made again with legs that much shorter, when the platform a
# train pulls into makes one of them longer than the plan reckoned with.
TRIES = 3


@dataclass
class Schedule:
    # What the simulation knows it by; the player sees it by its title, or by `label` once they
    # name it.
    name: str
    stops: list[Stop] = field(default_factory=list)
    label: str = ""
    # Whether the player set its refuelling stops by hand, so the game leaves them be.
    manual: bool = False
    # Bumped whenever its stops' stations or which of them only refuel change, so that its
    # trains know their copies are out of date.
    version: int = 0
    # Why some of its trains can't make the round, if they can't.
    problem: str = ""
    # The tank its refuelling stops were planned for, in cells; 0 for none.
    tank: float = 0.0
    # What its stops were when last settled: each one's station and whether it only refuels
    # there; and each one's cars and whether it waits for a full load.
    shape: tuple = ()
    loading: tuple = ()
    # The railway version and the player's stops its refuelling stops were planned for.
    planned_for: tuple = ()

    def own_stops(self) -> list[Stop]:
        """The stops the player gave it, without the refuelling stops the game put in."""
        return [stop for stop in self.stops if not stop.auto]


def shape_of(stops: list[Stop]) -> tuple:
    return tuple((stop.cell, stop.refuel_only) for stop in stops)


def loading_of(stops: list[Stop]) -> tuple:
    return tuple((tuple(stop.asked()), stop.wait_for_load) for stop in stops)


class Schedules:
    """Every schedule, and keeping their trains and refuelling stops in step with them."""

    def __init__(self, railway: Railway) -> None:
        self._railway = railway
        self.all: list[Schedule] = []
        self._made = 0

    def new(self) -> Schedule:
        self._made += 1
        schedule = Schedule(f"Schedule {self._made}")
        self.all.append(schedule)
        return schedule

    def named(self, name: str) -> Schedule | None:
        return next((s for s in self.all if s.name == name), None)

    def of(self, train: Train) -> Schedule | None:
        return self.named(train.schedule)

    def trains(self, schedule: Schedule) -> list[Train]:
        return [train for train in self._railway.trains if train.schedule == schedule.name]

    def title(self, schedule: Schedule) -> str:
        """What the player sees a schedule called: the name they gave it, or the first and last
        station of their stops."""
        if schedule.label:
            return schedule.label
        own = [stop.cell for stop in schedule.own_stops()]
        if not own:
            return schedule.name
        name_of = self._railway.name_of
        last = next((cell for cell in reversed(own) if cell != own[0]), None)
        return name_of(own[0]) if last is None else f"{name_of(own[0])}{DASH}{name_of(last)}"

    def remove(self, schedule: Schedule) -> bool:
        """Removes a schedule no train runs; False if one does."""
        if self.trains(schedule):
            return False
        self.all.remove(schedule)
        return True

    def join(self, train: Train, schedule: Schedule) -> None:
        """Moves a train to another schedule, whose stops it takes up after its round."""
        old = self.of(train)
        train.schedule = schedule.name
        if old is not None and old is not schedule:
            self.settle(old, replan=True)
        self.settle(schedule, replan=True)

    def set_manual(self, schedule: Schedule, manual: bool) -> None:
        """Lets the player set the refuelling stops by hand, the game's becoming theirs, or gives
        them back to the game, which takes out every refuelling stop and plans them again."""
        schedule.manual = manual
        if manual:
            for stop in schedule.stops:
                stop.auto = False
        else:
            schedule.stops[:] = [stop for stop in schedule.stops if not stop.refuel_only]
        self.settle(schedule, replan=True)

    def settle(self, schedule: Schedule, replan: bool = False) -> None:
        """Brings a schedule in step after any change: plans its refuelling stops again if the
        player's stops or the track changed, or if `replan` (its trains or their engines did),
        then hands changed cars to its trains at once and changed stops to those not out on a
        round."""
        if replan or schedule.planned_for != self._planned_key(schedule):
            self.plan(schedule)
        shape = shape_of(schedule.stops)
        changed = shape != schedule.shape
        if changed:
            schedule.shape = shape
            schedule.version += 1
        loading = loading_of(schedule.stops)
        loaded = loading != schedule.loading
        schedule.loading = loading
        if not changed and not loaded:
            return
        for train in self.trains(schedule):
            if train.taken != (schedule.name, schedule.version):
                if not train.path and between_rounds(train):
                    self.take_up(train, schedule)
            elif loaded:
                self.refit(train, schedule)

    def due(self, train: Train) -> Schedule | None:
        """The schedule whose stops a train has yet to take up, if its copy is out of date."""
        schedule = self.of(train)
        if schedule is None or train.taken == (schedule.name, schedule.version):
            return None
        return schedule

    def take_up(self, train: Train, schedule: Schedule) -> None:
        """Gives a train standing at a station its schedule's stops. If they start at another
        station, it unloads where it stands and is off the map for as long as the way there
        takes."""
        railway = self._railway
        stops = self.fitted(train, schedule)
        here = train.stops[(train.heading - 1) % len(train.stops)].cell if train.stops else None
        train.taken = (schedule.name, schedule.version)
        if here is not None and stops and not self._same_station(here, stops[0].cell):
            if train.away <= 0:
                railway.calls.arrive(train, Stop(here))
            train.away += railway.way_months(train, here, stops[0].cell)
        train.stops = stops
        train.restart()

    def fitted(self, train: Train, schedule: Schedule) -> list[Stop]:
        """A schedule's stops as a train calls at them: with each stop's cars from the left
        while its engine has room for them and can pull them, leaving out those it can't carry."""
        stops = []
        for template in schedule.stops:
            stop = Stop(
                template.cell,
                wait_for_load=template.wait_for_load,
                refuel_only=template.refuel_only,
                auto=template.auto,
            )
            if not template.refuel_only:
                self._fill(train, stop, template)
            stops.append(stop)
        return stops

    def taken(self, train: Train, template: Stop) -> int:
        """How many of a stop's cars a train takes on there: from the left, while its engine has
        room for them and can pull them, leaving out those it can't carry."""
        stop = Stop(template.cell)
        if not template.refuel_only:
            self._fill(train, stop, template)
        return len(stop.cars)

    def add_car(self, train: Train, stop: Stop, cargo: Cargo, asked: int = 0) -> bool:
        """Adds a car to a schedule's stop, up to MOST_CARS whatever they weigh, if `train`'s
        engine (standing in for the schedule's) has cars for that good yet."""
        railway = self._railway
        if isinstance(cargo, Good) and not railway.car_capacity(cargo, train.engine, asked=asked):
            return False
        return stop.add_car(cargo, MOST_CARS, asked)

    def refit(self, train: Train, schedule: Schedule) -> None:
        """Gives a train whose stops are its schedule's that schedule's cars and loading again,
        taking new cars at once at the stop it stands at, otherwise the next time it calls."""
        for index, (own, template) in enumerate(zip(train.stops, schedule.stops, strict=False)):
            fresh = Stop(template.cell, refuel_only=template.refuel_only)
            if not template.refuel_only:
                self._fill(train, fresh, template)
            own.wait_for_load = template.wait_for_load
            if own.asked() != fresh.asked():
                own.cars, own.generations = fresh.cars, fresh.generations
                train.cars_changed(index)

    def grow(self, schedule: Schedule, train: Train) -> None:
        """After a train's engine was swapped for a bigger one: adds more of each stop's last car
        while that engine has room for them and can pull them."""
        railway = self._railway
        for stop in schedule.stops:
            if stop.cars and not stop.refuel_only:
                last = len(stop.cars) - 1
                cargo, generation = stop.cars[last], stop.generation_at(last)
                while railway.add_car(train, stop, cargo, generation):
                    pass

    def drop(self, gone: set[Cell]) -> None:
        """Takes stations pulled down out of every schedule."""
        for schedule in self.all:
            if any(stop.cell in gone for stop in schedule.stops):
                schedule.stops[:] = [stop for stop in schedule.stops if stop.cell not in gone]
                self.settle(schedule)

    def moved(self, old: Cell, new: Cell) -> None:
        for schedule in self.all:
            for stop in schedule.stops:
                if stop.cell == old:
                    stop.cell = new

    def plan(self, schedule: Schedule) -> None:
        """Puts in the refuelling stops for the weakest of its trains that can make the round,
        unless the player sets them, and says why the others can't."""
        schedule.planned_for = self._planned_key(schedule)
        tanks = self._tanks(schedule)
        if schedule.manual:
            schedule.tank = 0.0
            schedule.problem = self._manual_problem(schedule, tanks)
            return
        own = schedule.own_stops()
        schedule.stops[:] = own
        schedule.tank = 0.0
        schedule.problem = ""
        if len(own) < 2 or not tanks:
            schedule.problem = self._missing_track(own)
            return
        first_failure = ""
        for tank, engine in tanks:
            planned, failure = self._plan_for(own, tank, engine)
            if planned is not None:
                schedule.stops[:] = planned
                schedule.tank = tank
                if first_failure and self.trains(schedule):
                    schedule.problem = (
                        f"Trains carrying fuel for less than {tank:.0f} cells wait: {first_failure}"
                    )
                return
            first_failure = first_failure or failure
        schedule.problem = first_failure

    def _plan_for(self, own: list[Stop], tank: float, engine: str) -> tuple[list[Stop] | None, str]:
        """The player's stops with refuelling stops between them for an engine with this tank,
        or None and why there are none."""
        railway = self._railway
        reach = tank
        for _ in range(TRIES):
            planned: list[Stop] = []
            for index, stop in enumerate(own):
                there = own[(index + 1) % len(own)].cell
                planned.append(stop)
                if self._same_station(stop.cell, there):
                    continue
                if railway.network.route_length(stop.cell, there) is None:
                    return None, f"No track from {railway.name_of(stop.cell)} to " + (
                        railway.name_of(there)
                    )
                chain = refuel_chain(railway, stop.cell, there, reach)
                if chain is None:
                    return None, (
                        f"From {railway.name_of(stop.cell)} to {railway.name_of(there)} no "
                        f"station is within the {tank:.0f} cells the {ENGINES[engine].title} "
                        "carries fuel for"
                    )
                planned += [Stop(cell, refuel_only=True, auto=True) for cell in chain]
            legs = railway.leg_lengths(planned, engine)
            longest = max((leg for leg in legs if leg is not None), default=0.0)
            if round(longest) <= tank:
                return planned, ""
            reach -= max(1.0, math.ceil(longest - tank))
        return None, "The platforms trains pull into make the legs too long to plan"

    def cannot_run(self, schedule: Schedule, engine: str) -> str:
        """Why a train behind this engine would stand rather than run the schedule; "" if it
        would run, as a train carrying fuel for more than the schedule's refuelling is planned
        for, or for which refuelling can be planned, or a train with no fuel to carry."""
        tank = ENGINES[engine].tank
        if len(schedule.stops) < 2:
            return ""
        if schedule.manual:
            return self._manual_problem(schedule, [(tank, engine)] if tank else [])
        if not tank or (schedule.tank and tank >= schedule.tank):
            return self._missing_track(schedule.stops)
        return self._plan_for(schedule.own_stops(), tank, engine)[1]

    def biggest_engine(self, schedule: Schedule) -> str:
        """Of its trains' engines and those on sale but expresses, the one with room for the most
        cars and pulling the most: its stops hold cars for the biggest train it could get, and
        while it has no trains its refuelling stops are planned for that engine."""
        engines = {train.engine for train in self.trains(schedule)}
        railway = self._railway
        engines |= {name for name in railway.engines() if not ENGINES[name].express}
        if not engines:
            return FIRST_ENGINE
        return max(engines, key=lambda name: (ENGINES[name].cars, railway.max_pull(name), name))

    def _tanks(self, schedule: Schedule) -> list[tuple[float, str]]:
        """Each tank of its trains that burn fuel, smallest first, with an engine that has it;
        while it has none, the tank of its biggest engine (biggest_engine), then the others on
        sale smallest first, should that one not make the round."""
        trains = self.trains(schedule)
        if not trains:
            biggest = self.biggest_engine(schedule)
            if not ENGINES[biggest].tank:
                return []
            others = sorted(self._railway.engines(), key=lambda name: ENGINES[name].tank)
            engines = [biggest, *others]
        else:
            engines = sorted((train.engine for train in trains), key=lambda e: ENGINES[e].tank)
        tanks: dict[float, str] = {}
        for engine in engines:
            tank = ENGINES[engine].tank
            if tank:
                tanks.setdefault(tank, engine)
        return list(tanks.items())

    def _manual_problem(self, schedule: Schedule, tanks: list[tuple[float, str]]) -> str:
        if not tanks or len(schedule.stops) < 2:
            return self._missing_track(schedule.stops)
        tank, engine = tanks[0]
        stops = schedule.stops
        for index, leg in enumerate(self._railway.leg_lengths(stops, engine)):
            there = stops[(index + 1) % len(stops)].cell
            names = (self._railway.name_of(stops[index].cell), self._railway.name_of(there))
            if leg is None:
                return f"No track from {names[0]} to {names[1]}"
            if round(leg) > tank:
                return (
                    f"From {names[0]} to {names[1]} is {leg:.0f} cells; the "
                    f"{ENGINES[engine].title} carries fuel for {tank:.0f}"
                )
        return ""

    def _missing_track(self, stops: list[Stop]) -> str:
        railway = self._railway
        for index, stop in enumerate(stops if len(stops) > 1 else []):
            there = stops[(index + 1) % len(stops)].cell
            if self._same_station(stop.cell, there):
                continue
            if railway.network.route_length(stop.cell, there) is None:
                return f"No track from {railway.name_of(stop.cell)} to {railway.name_of(there)}"
        return ""

    def _planned_key(self, schedule: Schedule) -> tuple:
        version = self._railway.constructions.railway_version
        return version, shape_of(schedule.own_stops())

    def _fill(self, train: Train, stop: Stop, template: Stop) -> None:
        railway = self._railway
        for cargo, generation in template.asked():
            if not railway.can_carry(train, cargo):
                continue
            if not railway.add_car(train, stop, cargo, generation):
                break

    def _same_station(self, one: Cell, other: Cell) -> bool:
        main = self._railway.constructions.main_station
        return (main(one) or one) == (main(other) or other)


def between_rounds(train: Train) -> bool:
    """Whether a train standing at a station is not out on a round: it has no round yet, has
    reached no stop since it took up its stops, or is back at its first."""
    stops = len(train.stops)
    return stops < 2 or train.arrivals == 0 or (train.heading - 1) % stops == 0 or train.away > 0


def refuel_chain(railway: Railway, start: Cell, end: Cell, reach: float) -> list[Cell] | None:
    """The stations to refuel at on the way from one station to another, so that no leg is
    longer than `reach` cells: the fewest, then those with the shortest way; none if the way is
    short enough, None if no stations will do."""
    network = railway.network
    direct = network.route_length(start, end)
    if direct is None:
        return None
    if round(direct) <= reach:
        return []
    grid = railway.constructions.tile_map.grid
    stations = [cell for cell in railway.stations() if cell != start]
    if end not in stations:
        stations.append(end)
    centres = {cell: grid.center(*cell) for cell in [start, *stations]}
    best: dict[Cell, tuple[int, float]] = {start: (0, 0.0)}
    before: dict[Cell, Cell] = {}
    frontier: list[tuple[int, float, Cell]] = [(0, 0.0, start)]
    while frontier:
        hops, length, here = heapq.heappop(frontier)
        if here == end:
            chain = []
            while here in before:
                here = before[here]
                chain.append(here)
            return chain[::-1][1:]
        if (hops, length) > best[here]:
            continue
        for other in stations:
            if other == here or math.dist(centres[here], centres[other]) > reach + 1:
                continue
            leg = network.route_length(here, other)
            if leg is None or round(leg) > reach:
                continue
            cost = (hops + 1, length + leg)
            if other not in best or cost < best[other]:
                best[other] = cost
                before[other] = here
                heapq.heappush(frontier, (*cost, other))
    return None
