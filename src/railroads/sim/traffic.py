"""How running trains keep out of each other's way (railroads.sim.railway runs them through it).

Trains pass each other within stations, and on double track when going opposite ways; the
track between such places is single line. A train takes the stretch of single line ahead of it as
it enters, and holds it from its tail to where the stretch ends: no train going the other way
enters any part of it until the train is out, but waits where the stretch begins. Trains going
the same way may follow each other onto it, until a train waiting at the other end has waited
LINE_PATIENCE months: from then on it keeps trains that have waited less from entering ahead of
it the other way, so none waits for ever on a busy line.

Wherever else a train would run into another where they can't pass, it runs only up to
FOLLOW_GAP cells short of it: it slows to the pace of a slower train ahead and follows it until
that one turns off, and stops before a crossing until a train going across has cleared it. Two
trains that stand in each other's way, as when track is changed under them, let the one with the
right of way through the other, as nothing else would move them.
"""

import bisect
import itertools
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import Cell
from railroads.sim.track_network import TrackNetwork
from railroads.sim.trains import Point, Train

# Trains meet where points along them, MEET_STEP cells apart, come within MEET_DISTANCE cells.
MEET_STEP = 0.2
MEET_DISTANCE = 0.2
# How many times the point where a train runs into another is halved between the points
# MEET_STEP apart that it lies between, so a train following another keeps an even gap.
REFINE_STEPS = 6
# How far short of a train in its way, in cells, a train stops or follows it.
FOLLOW_GAP = 0.3
# How long a train waits for single line, in months, before trains may no longer follow each
# other onto it the other way: about 15 seconds at normal speed.
LINE_PATIENCE = 0.1

# A train crossing from one cell into the next.
type Step = tuple[Cell, Cell]
# A point along a train, which way it faces there and the middle of the stretch of track it is on.
type Spot = tuple[Point, Point, Point]


@dataclass
class Stretches:
    """Where a train's path crosses from cell to cell, and where along it single line begins
    and ends."""

    # How far along the path each crossing lies, and the cells it crosses between.
    alongs: list[float]
    steps: list[Step]
    # For each crossing into single line from where trains pass: how far along it lies, and the
    # crossings from it to where that single line ends.
    entries: list[tuple[float, list[Step]]]
    # How far along the path each stretch of single line ends, by where it begins.
    ends: list[tuple[float, float]]

    def held(self, tail: float, front: float) -> list[Step]:
        """The crossings a train holds with its tail and front so far along: from its tail to its
        front, or on to where the single line its front is on ends."""
        low = bisect.bisect_left(self.alongs, tail)
        high = bisect.bisect_left(self.alongs, front)
        for start, stop in self.ends:
            if start < front < stop:
                high = bisect.bisect_right(self.alongs, stop)
                break
        return self.steps[low:high]


def stretches(network: TrackNetwork, train: Train) -> Stretches:
    """The train's path's Stretches, worked out once for each path and state of the track."""
    version = network.constructions.railway_version
    kept = train.path_stretches
    if kept is not None and kept[0] is train.path and kept[1] == version:
        return kept[2]
    path = train.path
    places = network.ends(train)
    cells = []
    single = []
    for a, b in itertools.pairwise(path):
        middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        cells.append(network.cell_at(middle))
        in_station, double = network.passing(middle)
        single.append(not in_station and not double)
    alongs: list[float] = []
    steps: list[Step] = []
    entries: list[tuple[float, list[Step]]] = []
    ends: list[tuple[float, float]] = []
    begun: tuple[float, int] | None = None
    for index in range(1, len(cells)):
        if cells[index] == cells[index - 1]:
            continue
        alongs.append(places[index])
        steps.append((cells[index - 1], cells[index]))
        if single[index] and not single[index - 1]:
            begun = (places[index], len(steps) - 1)
        elif single[index - 1] and not single[index] and begun is not None:
            entries.append((begun[0], steps[begun[1] :]))
            ends.append((begun[0], places[index]))
            begun = None
    if begun is not None:
        entries.append((begun[0], steps[begun[1] :]))
        ends.append((begun[0], places[-1]))
    found = Stretches(alongs, steps, entries, ends)
    train.path_stretches = (path, version, found)
    return found


@dataclass
class _Body:
    spots: list[Spot]
    points: npt.NDArray[np.float64]
    box: tuple[float, float, float, float]


@dataclass
class Traffic:
    """The trains running this moment: what each holds and where it is, kept as they move one
    by one, the one with the right of way first."""

    network: TrackNetwork
    # Each train's place in the right of way, the smaller first.
    rank: dict[int, tuple[float, float, int]]
    _held: dict[Step, set[int]] = field(default_factory=dict)
    _holding: dict[int, list[Step]] = field(default_factory=dict)
    _asked: dict[Step, int] = field(default_factory=dict)
    _bodies: dict[int, tuple[Train, _Body]] = field(default_factory=dict)
    _names: dict[int, str] = field(default_factory=dict)

    def add(self, trains: list[Train]) -> None:
        for train in trains:
            self.moved(train)

    def moved(self, train: Train) -> None:
        """Takes note of where a train is now, and what it holds."""
        key = id(train)
        self._names[key] = train.name
        for step in self._holding.pop(key, []):
            self._held[step].discard(key)
        self._bodies.pop(key, None)
        if len(train.path) < 2 or train.along <= 0:
            return
        held = stretches(self.network, train).held(
            max(0.0, train.along - train.length), train.along
        )
        self._holding[key] = held
        for step in held:
            self._held.setdefault(step, set()).add(key)
        spots = self.network.body(train, MEET_STEP)
        if spots:
            points = np.array([at for at, _, _ in spots])
            box = (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
            self._bodies[key] = (train, _Body(spots, points, box))

    def room(self, train: Train, wanted: float) -> tuple[float, str, bool]:
        """How far along its path the train may run towards `wanted`; with the train that keeps
        it from going further, if one does, and whether that one holds the single line ahead."""
        along = train.along
        if wanted <= along or len(train.path) < 2:
            return wanted, "", False
        found = stretches(self.network, train)
        limit = wanted
        for start, steps in found.entries:
            if start < along or start >= wanted:
                continue
            holder = self._against(train, steps)
            if holder:
                if train.waited >= LINE_PATIENCE:
                    for step in steps:
                        self._asked.setdefault(step, id(train))
                return start, holder, True
        spot, blocker = self._first_in_the_way(train, along, limit + FOLLOW_GAP)
        if blocker:
            return max(along, spot - FOLLOW_GAP), blocker, False
        return limit, "", False

    def _against(self, train: Train, steps: list[Step]) -> str:
        """The train holding, or asking first for, any of these crossings the other way."""
        key = id(train)
        for start, end in steps:
            back = (end, start)
            others = [other for other in self._held.get(back, ()) if other != key]
            if others:
                return self._names[others[0]]
            asked = self._asked.get(back)
            if asked is not None and asked != key:
                return self._names[asked]
        return ""

    def _first_in_the_way(self, train: Train, along: float, ahead: float) -> tuple[float, str]:
        """How far along its path, up to `ahead`, the train first runs into another where they
        can't pass, and which; `ahead` and no train if it runs into none."""
        alongs = list(np.arange(along + MEET_STEP, ahead, MEET_STEP)) + [ahead]
        spots = [self.network.spot(train, at) for at in alongs]
        points = np.array([at for at, _, _ in spots])
        reach = MEET_DISTANCE
        low_x, low_y = points[:, 0].min() - reach, points[:, 1].min() - reach
        high_x, high_y = points[:, 0].max() + reach, points[:, 1].max() + reach
        key = id(train)
        first = len(alongs)
        blocker: Train | None = None
        for other_key, (other, body) in self._bodies.items():
            if other_key == key:
                continue
            box = body.box
            if box[0] > high_x or box[2] < low_x or box[1] > high_y or box[3] < low_y:
                continue
            if other.giving_way == train.name and self.rank[key] < self.rank[other_key]:
                continue
            gaps = np.hypot(*(points[:, None, :] - body.points[None, :, :]).transpose(2, 0, 1))
            for i, j in np.argwhere(gaps < MEET_DISTANCE).tolist():
                if i < first and not self.network.can_pass(spots[i], body.spots[j]):
                    first = i
                    blocker = other
        if blocker is None:
            return ahead, ""
        low = float(alongs[first - 1]) if first else along
        high = float(alongs[first])
        body = self._bodies[id(blocker)][1]
        for _ in range(REFINE_STEPS):
            middle = (low + high) / 2
            if self._runs_into(self.network.spot(train, middle), body):
                high = middle
            else:
                low = middle
        return high, blocker.name

    def _runs_into(self, spot: Spot, body: _Body) -> bool:
        """Whether a train at this spot has run into a train with this body."""
        gaps = np.hypot(*(body.points - np.array(spot[0])).T)
        return any(
            not self.network.can_pass(spot, body.spots[j])
            for j in np.flatnonzero(gaps < MEET_DISTANCE).tolist()
        )
