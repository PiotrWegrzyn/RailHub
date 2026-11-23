"""Block signalling: how running trains keep out of each other's way (railroads.sim.railway runs
them through it).

The track outside stations is cut into blocks: each crossing or junction, with the junctions
beside it, is one; plain line between them is another, cut where single line meets double. A
signal stands wherever track runs into a block from another, or out of
a station. A train runs only on track set for it: before it runs on past a signal it asks for the
way from there on, block by block, up to the first signal it could stand at without being in
anyone's way but trains following it, with all of it in a station or on double track. The way is
set for it only once no other train holds any of it, and then all at once; until then it stands
SIGNAL_GAP short of the signal, red for it. Trains are let on in the order they asked, so as one
train frees track the train that has waited longest for it gets it first. A train gives up track
cell by cell as its tail leaves it.

On double track trains going opposite ways keep to their own tracks and never hold each other up.
A train may follow another going its way onto track that one holds, FOLLOW_GAP behind it, as long
as wherever all the trains ahead stopped it would stand clear too, and never past where theirs is
set; anywhere else two trains never hold track in the same cell at once. Stations are no blocks: any
number of trains may be in one, passing each other. A train standing where it could not have been
let on, as when track is laid under it, takes the track under it without holding up the trains
already on it, so nothing is left stuck.
"""

import itertools
import weakref
from collections.abc import Iterable
from dataclasses import dataclass

from railroads.sim.constructions import Constructions, TrackWidth
from railroads.sim.grids import Cell
from railroads.sim.track_network import TrackNetwork
from railroads.sim.trains import Claim, Point, Train

# How far short of a red signal a train stands, and of a train it follows, in cells.
SIGNAL_GAP = 0.15
FOLLOW_GAP = 0.3
# How far ahead of the end of the way set for it a train asks for more, in cells, so it finds
# the next signal clear without slowing for it.
SIGHT = 2.0
# Distances along a path closer than this, in cells, are one place.
SAME = 1e-6


@dataclass(frozen=True, slots=True)
class Pass:
    """A train's way across one cell outside stations: how far along its path it enters and
    leaves the cell, and the points on the cell's edge it enters and leaves by. Track through a
    corner runs along the edge between the two cells beside it, crossing the way of any train
    crossing that edge: a train going through a corner touches both those cells (`touch`), there
    and then, coming from the middle of one cell and going on to the middle of the other."""

    cell: Cell
    start: float
    end: float
    entry: Point
    exit: Point
    touch: bool = False


@dataclass(frozen=True, slots=True)
class SignalSpot:
    """A signal: the point on a cell's edge where track runs into a block, and which way a train
    passing it runs."""

    at: Point
    heading: Point
    cell: Cell


def rounded(point: Point) -> Point:
    """A point on the map as signalling compares them."""
    return (round(point[0], 4), round(point[1], 4))


class Blocks:
    """The blocks the track outside stations is cut into, and its signals, worked out once for
    each state of the track."""

    def __init__(self, constructions: Constructions) -> None:
        self.version = constructions.railway_version
        grid = constructions.tile_map.grid
        steps = grid.track_steps()
        stations = set(constructions.every_station_cell())
        cells = sorted({piece.cell for piece in constructions.track()} - stations)
        pieces = {cell: constructions.track_at(cell) for cell in cells}

        def point(cell: Cell, end: int) -> Point:
            x, y = grid.center(*cell)
            return rounded((float(x) + steps[end][0] / 2, float(y) + steps[end][1] / 2))

        def kind(cell: Cell) -> str:
            here = pieces[cell]
            if len(here) > 1:
                return "junction"
            return "double" if here[0].width == TrackWidth.DOUBLE else "single"

        # Whether each cell is plain double track, and the pairs of points its double track
        # joins, where trains going opposite ways keep to their own tracks.
        self.double = {cell: kind(cell) == "double" for cell in cells}
        self.lanes: dict[Cell, set[frozenset[Point]]] = {
            cell: {
                frozenset(point(cell, end) for end in piece.ends)
                for piece in pieces[cell]
                if piece.width == TrackWidth.DOUBLE
            }
            for cell in cells
        }
        exits = {
            (cell, end): grid.track_exit(cell, end)
            for cell in cells
            for piece in pieces[cell]
            for end in piece.ends
        }
        for cell in cells:
            for piece in pieces[cell]:
                if piece.width != TrackWidth.DOUBLE:
                    continue
                for end in piece.ends:
                    beyond, _ = exits[(cell, end)]
                    ends = frozenset(rounded(grid.center(*c)) for c in (cell, beyond))
                    for flank in grid.track_flanks(cell, end):
                        self.lanes.setdefault(flank, set()).add(ends)
        links: dict[Cell, list[Cell]] = {cell: [] for cell in cells}
        for (cell, _), (beside, entry) in exits.items():
            if beside not in pieces or kind(beside) != kind(cell):
                continue
            if any(entry in other.ends for other in pieces[beside]):
                links[cell].append(beside)
        # The block of each cell outside stations.
        self.block: dict[Cell, tuple[str, int]] = {}
        for number, cell in enumerate(cells):
            if cell in self.block:
                continue
            for member in _joined(cell, links):
                self.block[member] = (kind(cell), number)
        self.spots: list[SignalSpot] = []
        for (cell, end), (beside, _) in exits.items():
            if beside in stations or self.block.get(beside, self.block[cell]) != self.block[cell]:
                at = point(cell, end)
                x, y = grid.center(*cell)
                self.spots.append(SignalSpot(at, (float(x) - at[0], float(y) - at[1]), cell))

    def apart(self, cell: Cell, one: tuple[Point, Point], other: tuple[Point, Point]) -> bool:
        """Whether two trains crossing a cell, entering and leaving by these points, keep apart:
        on double track going opposite ways."""
        return (
            one[0] == other[1]
            and one[1] == other[0]
            and frozenset(one) in self.lanes.get(cell, set())
        )


def _joined(start: Cell, links: dict[Cell, list[Cell]]) -> set[Cell]:
    """The cells joined to `start`, one link after another."""
    group = {start}
    stack = [start]
    while stack:
        for beside in links[stack.pop()]:
            if beside not in group:
                group.add(beside)
                stack.append(beside)
    return group


_BLOCKS: weakref.WeakKeyDictionary[Constructions, Blocks] = weakref.WeakKeyDictionary()


def blocks_of(constructions: Constructions) -> Blocks:
    """The track's Blocks, worked out again once it changes."""
    kept = _BLOCKS.get(constructions)
    if kept is None or kept.version != constructions.railway_version:
        kept = _BLOCKS[constructions] = Blocks(constructions)
    return kept


def passes(network: TrackNetwork, train: Train) -> list[Pass]:
    """The cells outside stations the train's path crosses, in order, worked out once for each
    path."""
    kept = train.path_passes
    if kept is not None and kept[0] is train.path:
        return kept[1]
    path = train.path
    places = network.ends(train)
    grid = network.constructions.tile_map.grid
    found: list[Pass] = []
    runs = itertools.groupby(
        range(len(path) - 1),
        key=lambda k: network.cell_at(
            ((path[k][0] + path[k + 1][0]) / 2, (path[k][1] + path[k + 1][1]) / 2)
        ),
    )
    before: Cell | None = None
    for cell, steps in runs:
        indices = list(steps)
        first, last = indices[0], indices[-1] + 1
        if before is not None and cell not in grid.neighbours(*before):
            beside = set(grid.neighbours(*before)) & set(grid.neighbours(*cell))
            way = (rounded(path[first - 1]), rounded(path[first + 1]))
            found += [Pass(flank, places[first], places[first], *way, True) for flank in beside]
        before = cell
        middle = (
            (path[first][0] + path[first + 1][0]) / 2,
            (path[first][1] + path[first + 1][1]) / 2,
        )
        if network.passing(middle)[0]:
            continue
        found.append(
            Pass(cell, places[first], places[last], rounded(path[first]), rounded(path[last]))
        )
    train.path_passes = (path, found)
    return found


class Signals:
    """The track each running train holds this moment, kept as trains are let on and move."""

    def __init__(self, network: TrackNetwork, trains: Iterable[Train]) -> None:
        self.network = network
        self.blocks = blocks_of(network.constructions)
        # Who holds track in each cell: the train, the points its way enters and leaves the cell
        # by, and how far along its path it enters it.
        self._held: dict[Cell, list[tuple[Train, Point, Point, float]]] = {}
        # The trains that were not let on this moment, in the order they asked, with the way
        # each waits for, which trains that asked later must not take from them.
        self._waiting: list[tuple[Train, list[Pass]]] = []
        trains = list(trains)
        self._named = {train.name: train for train in trains}
        loose = []
        for train in trains:
            if train.claims_for is not train.path:
                train.claims, train.cleared, train.asked = [], 0.0, 0
                train.claims_for = train.path
                if train.path and train.along > 0:
                    loose.append(train)
                    continue
            for claim in train.claims:
                self._hold(train, claim)
        for train in loose:
            self._take_ground(train)

    def _hold(self, train: Train, claim: Claim) -> None:
        cell, entry, exit_, start, _, soft = claim
        if not soft:
            self._held.setdefault(cell, []).append((train, entry, exit_, start))

    def _holder(self, train: Train, crossing: Pass, follow: bool = False) -> Train | None:
        """A train other than this one holding track in the cell that this way across it can't
        share; with `follow`, a train ahead of it going the same way across it can share it."""
        way = (crossing.entry, crossing.exit)
        for other, entry, exit_, start in self._held.get(crossing.cell, ()):
            if other is train or self.blocks.apart(crossing.cell, way, (entry, exit_)):
                continue
            if follow and way == (entry, exit_) and _ahead(train, crossing, other, start):
                continue
            return other
        return None

    def _going_my_way(self, train: Train, crossing: Pass) -> dict[int, tuple[Train, float]]:
        """The other trains ahead of this one holding the cell on the same way across it, with
        how much further along this train's path than along theirs the cell lies."""
        way = (crossing.entry, crossing.exit)
        return {
            id(other): (other, crossing.start - start)
            for other, entry, exit_, start in self._held.get(crossing.cell, ())
            if other is not train
            and (entry, exit_) == way
            and _ahead(train, crossing, other, start)
        }

    def _clear_behind(
        self,
        train: Train,
        crossings: list[Pass],
        stop: float,
        leaders: dict[int, tuple[Train, float]],
    ) -> bool:
        """Whether a train following these trains would stand clear of all but trains following
        it wherever they all stopped: each as far as the way set for it lets it, or close
        behind the one before; none behind a train that reaches its last stop, which leaves the
        track there."""
        stand = stop
        before: Train | None = None
        for other, offset in sorted(leaders.values(), key=lambda pair: -(pair[0].along + pair[1])):
            if other.cleared >= self.network.length(other) - SAME:
                continue
            if before is not None:
                stand -= before.length + FOLLOW_GAP
            stand = min(stand, other.cleared + offset - SIGNAL_GAP)
            before = other
        if before is not None:
            stand -= before.length + FOLLOW_GAP
        if stand >= stop - SAME:
            return True
        return stand > train.along and self._on_double(crossings, stand - train.length, stand)

    def _take_ground(self, train: Train) -> None:
        """Has a train standing where it was not let on hold the track under it and the rest of
        the cell its front is in, without holding up the trains already holding any of it."""
        tail = train.along - train.length
        under = [p for p in passes(self.network, train) if p.end > tail and p.start < train.along]
        for crossing in under:
            claim = (
                crossing.cell,
                crossing.entry,
                crossing.exit,
                crossing.start,
                crossing.end,
                self._holder(train, crossing) is not None,
            )
            train.claims.append(claim)
            self._hold(train, claim)
        train.cleared = max([train.along, *(p.end for p in under)])

    def let_on(self, train: Train, ask: int, reach: float) -> Train | None:
        """Sets the way on for a train, a stretch at a time, while the end of the way set for it
        lies less than SIGHT past `reach`, how far along its path it would run this moment: each
        stretch up to where it could stand clear of all but trains following it (see the
        module). Returns the train holding track it needs, if one keeps a stretch from being
        set; `ask` is the place in the queue the train takes if it has to wait."""
        length = self.network.length(train)
        while train.cleared < length - SAME and train.cleared - max(train.along, reach) <= SIGHT:
            holder = self._let_on_stretch(train, ask, length)
            if holder is not None:
                return holder
        return None

    def _let_on_stretch(self, train: Train, ask: int, length: float) -> Train | None:
        crossings = passes(self.network, train)
        stop = self._stand(train, crossings, length)
        wanted = [p for p in crossings if p.start >= train.cleared - SAME and p.start < stop - SAME]
        # The trains going its way that it would follow onto track they hold, by how much further
        # along this train's path than along theirs each place lies: it may not be set a way past
        # where theirs is set.
        leaders: dict[int, tuple[Train, float]] = {}
        for held in train.claims:
            leaders |= self._going_my_way(train, Pass(held[0], held[3], held[4], held[1], held[2]))
        holder = None
        for crossing in wanted:
            holder = self._holder(train, crossing, follow=True)
            while holder is not None and train.first and self._give_way(holder, crossing):
                holder = self._holder(train, crossing, follow=True)
            alongside = self._going_my_way(train, crossing)
            if holder is None and not leaders.keys() <= alongside.keys():
                holder = next(iter(leaders.values()))[0]
            if holder is not None:
                break
            leaders |= alongside
        if holder is None and leaders and not self._clear_behind(train, crossings, stop, leaders):
            holder = max(leaders.values(), key=lambda pair: pair[0].along + pair[1])[0]
        if holder is None and not train.first:
            holder = self._queued_before(train, wanted)
        if holder is not None:
            if not train.asked:
                train.asked = ask
            self._waiting.append((train, wanted))
            return holder
        for crossing in wanted:
            claim = (
                crossing.cell,
                crossing.entry,
                crossing.exit,
                crossing.start,
                crossing.end,
                False,
            )
            train.claims.append(claim)
            self._hold(train, claim)
        train.cleared = stop
        train.asked = 0
        return None

    def _give_way(self, other: Train, crossing: Pass) -> bool:
        """Takes back from another train the way set for it from the cell a train put first wants
        on, if it can be: if it has not come to any of it yet, could stop short of it, and would
        stand clear there, with no train following it on any of it. Whether it did."""
        cell = crossing.cell
        taken = [claim for claim in other.claims if claim[0] == cell and not claim[5]]
        if not taken:
            return False
        back = min(claim[3] for claim in taken)
        if other.along > back - SIGNAL_GAP - SAME:
            return False
        if not self._on_double(passes(self.network, other), back - other.length, back):
            return False
        given = [claim for claim in other.claims if claim[3] >= back - SAME]
        for cell_given, entry, exit_, _, _, _ in given:
            for holder, *way in self._held.get(cell_given, ()):
                if holder is not other and tuple(way[:2]) == (entry, exit_):
                    return False
        other.claims = [claim for claim in other.claims if claim[3] < back - SAME]
        for cell_given, entry, exit_, _, _, soft in given:
            if soft:
                continue
            held = self._held.get(cell_given, [])
            self._held[cell_given] = [
                pair for pair in held if not (pair[0] is other and pair[1:3] == (entry, exit_))
            ]
        other.cleared = back
        return True

    def _queued_before(self, train: Train, wanted: list[Pass]) -> Train | None:
        """A train that asked before this one and still waits for track this one wants, which it
        must leave to it: unless this one holds track that train waits for, or that train
        waits, through the trains each waits for in turn, on this one."""
        mine = {crossing.cell: (crossing.entry, crossing.exit) for crossing in wanted}
        holding = {claim[0] for claim in train.claims}
        for other, theirs in self._waiting:
            if any(crossing.cell in holding for crossing in theirs) or self._waits_on(other, train):
                continue
            for crossing in theirs:
                way = mine.get(crossing.cell)
                if way is not None and not self.blocks.apart(
                    crossing.cell, way, (crossing.entry, crossing.exit)
                ):
                    return other
        return None

    def _waits_on(self, other: Train, train: Train) -> bool:
        """Whether a train waits on another, through the trains each waits for in turn."""
        seen = {other.name}
        name = other.giving_way
        while name and name not in seen:
            if name == train.name:
                return True
            seen.add(name)
            found = self._named.get(name)
            name = found.giving_way if found is not None else ""
        return False

    def _on_double(self, crossings: list[Pass], start: float, stop: float) -> bool:
        """Whether the train's way between these places along its path runs only in stations and
        on plain double track, where a train standing blocks none but trains following it."""
        double = self.blocks.double
        return all(
            double.get(other.cell, False)
            for other in crossings
            if not other.touch and other.end > start + SAME and other.start < stop - SAME
        )

    def _stand(self, train: Train, crossings: list[Pass], length: float) -> float:
        """How far along its path the first place past the way set for the train lies where it
        could stand clear of all but trains following it: before a signal, with all of it in a
        station or on plain double track; the end of its path if there is none."""
        blocks = self.blocks
        before = None
        for crossing in crossings:
            if crossing.touch:
                continue
            block = blocks.block.get(crossing.cell)
            signal = before is None or block != before[1] or before[0] < crossing.start - SAME
            before = (crossing.end, block)
            if not signal or crossing.start <= train.cleared + SAME:
                continue
            if self._on_double(crossings, crossing.start - train.length, crossing.start):
                return crossing.start
        return length

    def leave(self, train: Train) -> None:
        """Gives up the track the train's tail has left."""
        tail = train.along - train.length
        if not train.path:
            gone, train.claims = train.claims, []
            train.cleared, train.asked = 0.0, 0
        else:
            gone = [claim for claim in train.claims if claim[4] <= tail + SAME]
            if not gone:
                return
            train.claims = [claim for claim in train.claims if claim[4] > tail + SAME]
        for cell, entry, exit_, _, _, soft in gone:
            if soft:
                continue
            held = self._held.get(cell, [])
            for index, (other, other_entry, other_exit, _) in enumerate(held):
                if other is train and (other_entry, other_exit) == (entry, exit_):
                    del held[index]
                    break

    def limit(self, train: Train) -> tuple[float, Train | None]:
        """How far along its path the train may run on the way set for it, FOLLOW_GAP short of
        the tail of any train going its way ahead of it on track they share; with that train."""
        length = self.network.length(train)
        limit = length if train.cleared >= length - SAME else train.cleared - SIGNAL_GAP
        ahead = None
        for cell, entry, exit_, start, _, _ in train.claims:
            for other, other_entry, other_exit, other_start in self._held.get(cell, ()):
                if other is train or (other_entry, other_exit) != (entry, exit_):
                    continue
                front = start + other.along - other_start
                if front <= train.along:
                    continue
                tail = start + max(0.0, other.along - other.length - other_start) - FOLLOW_GAP
                if tail < limit:
                    limit, ahead = tail, other
        return max(train.along, limit), ahead


def _ahead(train: Train, crossing: Pass, other: Train, start: float) -> bool:
    """Whether another train crossing a cell the same way as this train's `crossing`, which it
    comes into `start` along its path, is ahead of this one."""
    return crossing.start + other.along - start > train.along


def cleared_signals(trains: Iterable[Train]) -> set[tuple[Cell, Point]]:
    """The signals showing clear: those into a cell, by the point it is entered by, on the way
    set for a train that has yet to pass them."""
    return {
        (cell, entry)
        for train in trains
        for cell, entry, _, start, _, soft in train.claims
        if not soft and start > train.along
    }
