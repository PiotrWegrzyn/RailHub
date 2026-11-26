"""Block signalling: how running trains keep out of each other's way (railroads.sim.railway runs
them through it).

The track outside stations is cut into blocks: each crossing or junction, with the junctions
beside it and any short single line between them, is one; plain line between them is another, cut where
single line meets double. A signal stands wherever track runs into a block from another, or out of
a station. A train runs only on track set for it: before it runs on past a signal it asks for the
way from there on, block by block, up to the first signal it could stand at without being in
anyone's way but trains following it, with all of it in a station or on double track (none of it
beside track through a corner), or on the
single line it came along, holding no more than that line, when no other train holds any of the
way on from there to a station or double track (`_line_end`), which no train may stand on
meanwhile. The way is
set for it only once no other train holds any of it, and then all at once; until then it stands
SIGNAL_GAP short of the signal, red for it. Trains are let on in the order they asked, so as one
train frees track the train that has waited longest for it gets it first; but as a dispatcher
would, track is not kept for a train that could not go anyway, as a third train holds some of its
way, while another that would not follow that third train onto its track can use it now. A train gives up track cell by cell as its tail leaves it.

On double track trains going opposite ways keep to their own tracks and never hold each other up;
on quadruple track trains going the same way keep to its two tracks that way, taking one no train
ahead runs on, leaving one they followed onto it to run beside it, and through its junctions too,
where two trains coming together or parting run side by side, the one on the side those tracks lie
on taking the outer track (Blocks.apart). Its two-track stretches are blocks of their own, apart
from its ends where the tracks come together, so a train stands on its track short of an end until
the way through it is set, and one beside it can pass; into a station they run on up to it, so
two trains each way come in or leave at once.
A train may follow another going its way onto track that one holds, FOLLOW_GAP behind it, as long
as wherever all the trains ahead stopped it would stand clear too, and never past where theirs is
set unless it runs beside it on quadruple track; anywhere else two trains never hold track in the same cell at once. Stations are no blocks:
any number of trains may be in one, passing each other, but never a train they follow. Of two
trains level on one way, the one whose way is set further, else the one bought first, is ahead.
A train standing where it could not have been
let on, as when track is laid under it, takes the track under it without holding up the trains
already on it, so nothing is left stuck.
"""

import itertools
import weakref
from collections.abc import Iterable
from dataclasses import dataclass, replace

from railroads.sim.constructions import Constructions, TrackWidth
from railroads.sim.grids import Cell, Grid
from railroads.sim.track_network import TrackNetwork
from railroads.sim.trains import ENGINES, Claim, Point, Train

# The tracks of quadruple track each way: the inner, fast track, which express engines and
# railcars take first, and the outer, slow one, which all others take first.
FAST_LANE = 0
SLOW_LANE = 1
# How far short of a red signal a train stands, and of a train it follows, in cells.
SIGNAL_GAP = 0.15
FOLLOW_GAP = 0.5
# How far ahead of the end of the way set for it a train asks for more, in cells, so it finds
# the next signal clear without slowing for it.
SIGHT = 2.0
# Plain single line shorter than this, in cells, between junctions on both ends belongs with them,
# as no train stands in so little: one block, with signals only on the ways into it. Double track
# keeps its signals however short, as a train stands on it only where all of it fits.
SHORT_LINE = 3
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
    # Which track of quadruple track it takes (railroads.sim.signals.FAST_LANE): the same on
    # all other track.
    lane: int = 0


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


# Each track point of each cell as signalling compares them, by grid (Blocks), kept as the
# blocks are cut again whenever track or stations change.
_ENDS: dict[Grid, dict[tuple[Cell, int], Point]] = {}


def track_point(grid: Grid, cell: Cell, end: int) -> Point:
    """A track point of a cell as signalling compares them."""
    known = _ENDS.setdefault(grid, {})
    found = known.get((cell, end))
    if found is None:
        steps = grid.track_steps()
        x, y = grid.center(*cell)
        found = rounded((float(x) + steps[end][0] / 2, float(y) + steps[end][1] / 2))
        known[cell, end] = found
    return found


class Blocks:
    """The blocks the track outside stations is cut into, and its signals, worked out once for
    each state of the track."""

    def __init__(self, constructions: Constructions) -> None:
        self.version = constructions.railway_version
        grid = constructions.tile_map.grid
        stations = set(constructions.every_station_cell())
        cells = sorted({piece.cell for piece in constructions.track()} - stations)
        pieces = {cell: constructions.track_at(cell) for cell in cells}

        def point(cell: Cell, end: int) -> Point:
            return track_point(grid, cell, end)

        kinds = {
            cell: "junction" if len(here) > 1 else here[0].width.name.lower()
            for cell, here in pieces.items()
        }
        kind = kinds.__getitem__

        # Whether each cell is plain double track, and the pairs of points its double track
        # joins, where trains going opposite ways keep to their own tracks.
        # Track through a corner runs between the two cells beside it, so a train standing in
        # either would be in the way of trains on that track: no train stands there.
        flanking = {
            flank
            for cell in cells
            for piece in pieces[cell]
            for end in piece.ends
            for flank in grid.track_flanks(cell, end)
        }
        self.double = {
            cell: kind(cell) in ("double", "quadruple") and cell not in flanking for cell in cells
        }
        self.lanes: dict[Cell, set[frozenset[Point]]] = {
            cell: {
                frozenset(point(cell, end) for end in piece.ends)
                for piece in pieces[cell]
                if piece.width >= TrackWidth.DOUBLE
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
                if piece.width < TrackWidth.DOUBLE:
                    continue
                for end in piece.ends:
                    beyond, _ = exits[(cell, end)]
                    ends = frozenset(rounded(grid.center(*c)) for c in (cell, beyond))
                    for flank in grid.track_flanks(cell, end):
                        self.lanes.setdefault(flank, set()).add(ends)

        def runs_on_four(cell: Cell, end: int) -> bool:
            beyond, entry = exits[(cell, end)]
            return beyond in stations or (
                beyond in pieces
                and any(
                    piece.width == TrackWidth.QUADRUPLE and entry in piece.ends
                    for piece in pieces[beyond]
                )
            )

        # The cells of quadruple track that runs on as quadruple track at both ends, straight on
        # or through a junction, or into a station, where trains going the same way keep to two
        # tracks: two trains a way can come into a station or leave it at once.
        self.four_lanes = {
            cell
            for cell in cells
            if kind(cell) == "quadruple"
            and all(runs_on_four(cell, end) for end in pieces[cell][0].ends)
        }

        def block_kind(cell: Cell) -> str:
            """What a cell is for cutting blocks: the two-track stretches of quadruple track are
            blocks of their own, apart from its ends where its tracks come together, so a train
            stands on its own track short of them and a train beside it can pass."""
            return "lanes" if cell in self.four_lanes else kind(cell)

        links: dict[Cell, list[Cell]] = {cell: [] for cell in cells}
        for (cell, _), (beside, entry) in exits.items():
            if beside not in pieces or block_kind(beside) != block_kind(cell):
                continue
            if any(entry in other.ends for other in pieces[beside]):
                links[cell].append(beside)
        # The ways across junction cells along quadruple track, where trains going the same way
        # keep to its two tracks too, and the middle of each of those cells.
        self.laned_ways: dict[Cell, set[frozenset[Point]]] = {}
        for cell in cells:
            ways = {
                frozenset(point(cell, end) for end in piece.ends)
                for piece in pieces[cell]
                if piece.width == TrackWidth.QUADRUPLE
            }
            if kind(cell) == "junction" and ways:
                self.laned_ways[cell] = ways
        self.middles = {cell: rounded(grid.center(*cell)) for cell in self.laned_ways}
        # The cells where trains going the same way keep to the two tracks of quadruple track.
        self.lane_cells = self.four_lanes | set(self.laned_ways)
        # The block of each cell outside stations.
        self.block: dict[Cell, tuple[str, int]] = {}
        groups: dict[tuple[str, int], set[Cell]] = {}
        for number, cell in enumerate(cells):
            if cell in self.block:
                continue
            group = groups[(block_kind(cell), number)] = _joined(cell, links)
            for member in group:
                self.block[member] = (block_kind(cell), number)
        parent = {block: block for block in groups}

        def root(block: tuple[str, int]) -> tuple[str, int]:
            while parent[block] != block:
                block = parent[block]
            return block

        for block, group in groups.items():
            if block[0] != "single" or len(group) >= SHORT_LINE:
                continue
            beside = {
                self.block.get(exits[(cell, end)][0])
                for cell in group
                for piece in pieces[cell]
                for end in piece.ends
            } - {block}
            if beside and all(other is not None and other[0] == "junction" for other in beside):
                for other in beside:
                    parent[root(other)] = root(block)
        merged = {root(block) for block in parent if root(block) != block}
        for cell, block in self.block.items():
            found = root(block)
            self.block[cell] = ("junction", found[1]) if found in merged else block
        # Track of a junction through a corner runs between the two cells beside it, so a train
        # standing on plain line in either would be in the way of trains through the junction:
        # those cells belong with it, and a train stands short of them.
        for cell in cells:
            if self.block[cell][0] != "junction":
                continue
            for piece in pieces[cell]:
                for end in piece.ends:
                    for flank in grid.track_flanks(cell, end):
                        if flank in self.block and self.block[flank][0] != "junction":
                            self.block[flank] = self.block[cell]
        self.spots: list[SignalSpot] = []
        for (cell, end), (beside, _) in exits.items():
            if beside in stations or self.block.get(beside, self.block[cell]) != self.block[cell]:
                at = point(cell, end)
                x, y = grid.center(*cell)
                self.spots.append(SignalSpot(at, (float(x) - at[0], float(y) - at[1]), cell))

    def apart(
        self,
        cell: Cell,
        one: tuple[Point, Point],
        other: tuple[Point, Point],
        lanes: tuple[int, int] = (0, 0),
    ) -> bool:
        """Whether two trains crossing a cell, entering and leaving by these points, on these
        tracks of quadruple track, keep apart: on double or quadruple track going opposite ways,
        or on quadruple track going the same way on different tracks; through a junction of
        quadruple track also where they come together or part, the one on the side its tracks
        that way lie on taking the outer one (_branches_apart)."""
        if one == other:
            return lanes[0] != lanes[1] and (cell in self.four_lanes or self._laned(cell, one))
        if one[0] == other[1] and one[1] == other[0]:
            return frozenset(one) in self.lanes.get(cell, set())
        if lanes[0] == lanes[1] or not (self._laned(cell, one) and self._laned(cell, other)):
            return False
        return self._branches_apart(cell, one, other, lanes)

    def takes_lane(self, cell: Cell, entry: Point, exit_: Point) -> bool:
        """Whether a way across a cell runs on one of the two tracks of quadruple track going
        its way."""
        return cell in self.four_lanes or self._laned(cell, (entry, exit_))

    def natural_lane(self, cell: Cell, way: tuple[Point, Point]) -> int | None:
        """The track of quadruple track a way across a junction keeps apart on from every other
        way there going the same way that it comes together with or parts from
        (_branches_apart), so a train coming the other way in can run alongside; None where
        there is no such way, or no one track does."""
        entry, exit_ = way
        others = []
        for other in self.laned_ways.get(cell, ()):
            if other == frozenset(way):
                continue
            if exit_ in other and entry not in other:
                (start,) = other - {exit_}
                others.append((start, exit_))
            elif entry in other and exit_ not in other:
                (end,) = other - {entry}
                others.append((entry, end))
        if not others:
            return None
        for lane in (SLOW_LANE, FAST_LANE):
            if all(self._branches_apart(cell, way, other, (lane, 1 - lane)) for other in others):
                return lane
        return None

    def _laned(self, cell: Cell, way: tuple[Point, Point]) -> bool:
        return frozenset(way) in self.laned_ways.get(cell, ())

    def _branches_apart(
        self,
        cell: Cell,
        one: tuple[Point, Point],
        other: tuple[Point, Point],
        lanes: tuple[int, int],
    ) -> bool:
        """Whether two ways going the same way across a junction of quadruple track, coming
        together at one end or parting at the other, keep apart on these tracks: trains keep to
        the tracks to the left of the way they go, the slow one outermost, so the one whose
        other end lies further to the left must take the slow track and the other the fast."""
        middle = self.middles[cell]
        if one[1] == other[1] and one[0] != other[0]:
            shared, ends, inward = one[1], (one[0], other[0]), False
        elif one[0] == other[0] and one[1] != other[1]:
            shared, ends, inward = one[0], (one[1], other[1]), True
        else:
            return False
        way = (
            (middle[0] - shared[0], middle[1] - shared[1])
            if inward
            else (shared[0] - middle[0], shared[1] - middle[1])
        )
        left = [way[0] * (end[1] - middle[1]) - way[1] * (end[0] - middle[0]) for end in ends]
        if abs(left[0] - left[1]) < SAME:
            return False
        outer = 0 if left[0] > left[1] else 1
        return lanes[outer] == SLOW_LANE and lanes[1 - outer] == FAST_LANE


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
        # by, how far along its path it enters it, and the track of quadruple track it takes.
        self._held: dict[Cell, list[tuple[Train, Point, Point, float, int]]] = {}
        # The trains that were not let on this moment, in the order they asked, with the way
        # each waits for, which trains that asked later must not take from them.
        self._waiting: list[tuple[Train, list[Pass]]] = []
        trains = list(trains)
        self._named = {train.name: train for train in trains}
        # Of two trains level on one way with their ways set as far, the one earlier here counts
        # as ahead of the other.
        self._order = {id(train): index for index, train in enumerate(trains)}
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
        # The trains set a way to the end of a single line, with the way on from there each
        # keeps for itself (_way_on).
        self._kept: list[tuple[Train, list[Pass]]] = []
        for train in trains:
            if train.path and train.claims and train.cleared < network.length(train) - SAME:
                way = self._way_on(train, passes(network, train), train.cleared)
                if way:
                    self._kept.append((train, way))

    def blocked(self, train: Train, cell: Cell, way: tuple[Point, Point]) -> bool:
        """Whether other trains hold track in the cell this way across it can't share on any
        track it could take there: on double track trains going the other way keep to their own
        track, and on quadruple track a train can pass another going its way on the other
        track (Blocks.apart)."""
        holds = self._held.get(cell)
        if not holds:
            return False
        lanes = (FAST_LANE, SLOW_LANE) if self.blocks.takes_lane(cell, *way) else (FAST_LANE,)
        return all(
            any(
                other is not train
                and not self.blocks.apart(cell, way, (entry, exit_), (lane, held))
                for other, entry, exit_, _, held in holds
            )
            for lane in lanes
        )

    def held_by_others(self, train: Train, cells: Iterable[Cell]) -> int:
        """How many of these cells another train holds track in."""
        return sum(
            any(other is not train for other, *_ in self._held.get(cell, ())) for cell in cells
        )

    def _hold(self, train: Train, claim: Claim) -> None:
        cell, entry, exit_, start, _, soft, lane = claim
        if not soft:
            self._held.setdefault(cell, []).append((train, entry, exit_, start, lane))

    def _holder(self, train: Train, crossing: Pass, follow: bool = False) -> Train | None:
        """A train other than this one holding track in the cell that this way across it can't
        share; with `follow`, a train ahead of it going the same way across it can share it."""
        way = (crossing.entry, crossing.exit)
        for other, entry, exit_, start, lane in self._held.get(crossing.cell, ()):
            lanes = (crossing.lane, lane)
            if other is train or self.blocks.apart(crossing.cell, way, (entry, exit_), lanes):
                continue
            same = way == (entry, exit_) and crossing.lane == lane
            if follow and same and self._ahead(train, crossing, other, start):
                continue
            return other
        return None

    def _going_my_way(self, train: Train, crossing: Pass) -> dict[int, tuple[Train, float]]:
        """The other trains ahead of this one holding the cell on the same way and track across
        it, with how much further along this train's path than along theirs the cell lies."""
        way = (crossing.entry, crossing.exit)
        return {
            id(other): (other, crossing.start - start)
            for other, entry, exit_, start, lane in self._held.get(crossing.cell, ())
            if other is not train
            and (entry, exit_) == way
            and lane == crossing.lane
            and self._ahead(train, crossing, other, start)
        }

    def _ahead(self, train: Train, crossing: Pass, other: Train, start: float) -> bool:
        """Whether another train crossing a cell the same way as this train's `crossing`, which it
        comes into `start` along its path, is ahead of this one (_leads)."""
        offset = crossing.start - start
        return self._leads(other, train, offset + other.along - train.along, offset)

    def _leads(self, other: Train, train: Train, lead: float, offset: float) -> bool:
        """Whether a train `lead` further along than another, the place it is at lying `offset`
        further along the other's path than along its own, is ahead of it: where they are level,
        the one whose way is set further, or if neither's is, the one earlier among the
        trains."""
        if abs(lead) > SAME:
            return lead > 0
        further = offset + other.cleared - train.cleared
        if abs(further) > SAME:
            return further > 0
        return self._order.get(id(other), 0) < self._order.get(id(train), 0)

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
            claim = _claim(crossing, soft=self._holder(train, crossing) is not None)
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
        holder = None
        for lane in self._lanes_for(train, wanted):
            on_lane = [
                replace(p, lane=lane) if self.blocks.takes_lane(p.cell, p.entry, p.exit) else p
                for p in wanted
            ]
            found = self._kept_off(train, crossings, stop, on_lane)
            if found is None:
                for crossing in on_lane:
                    claim = _claim(crossing)
                    train.claims.append(claim)
                    self._hold(train, claim)
                train.cleared = stop
                train.asked = 0
                if any(self.blocks.takes_lane(p.cell, p.entry, p.exit) for p in wanted):
                    train.lane = lane
                way = self._way_on(train, crossings, stop) if stop < length - SAME else None
                if way:
                    self._kept.append((train, way))
                return None
            holder = holder or found
        if not train.asked:
            train.asked = ask
        self._waiting.append((train, wanted))
        return holder

    def _lanes_for(self, train: Train, wanted: list[Pass]) -> tuple[int, ...]:
        """The tracks of quadruple track a train tries in turn for the way it wants: one no train
        going its way runs on before one it would follow, so a train held up by one ahead passes
        it on the other track; of two alike the one a junction on the way leaves room beside
        (Blocks.natural_lane), then the fast track first for an express engine or a railcar,
        and the slow track first for any other; only one where the way crosses no quadruple
        track."""
        if not any(self.blocks.takes_lane(p.cell, p.entry, p.exit) for p in wanted):
            return (FAST_LANE,)
        engine = ENGINES[train.engine]
        fast = engine.express or engine.railcar
        lanes = (FAST_LANE, SLOW_LANE) if fast else (SLOW_LANE, FAST_LANE)
        natural = next(
            (
                lane
                for p in wanted
                if (lane := self.blocks.natural_lane(p.cell, (p.entry, p.exit))) is not None
            ),
            None,
        )
        return tuple(
            sorted(
                lanes,
                key=lambda lane: (self._taken(train, wanted, lane), natural not in (None, lane)),
            )
        )

    def _taken(self, train: Train, wanted: list[Pass], lane: int) -> bool:
        """Whether another train holds quadruple track on the way the train wants, going its way
        on this track of it."""
        return any(
            other is not train and (entry, exit_, held) == (crossing.entry, crossing.exit, lane)
            for crossing in wanted
            if self.blocks.takes_lane(crossing.cell, crossing.entry, crossing.exit)
            for other, entry, exit_, _, held in self._held.get(crossing.cell, ())
        )

    def _kept_off(
        self, train: Train, crossings: list[Pass], stop: float, wanted: list[Pass]
    ) -> Train | None:
        """The train that keeps this one from being set the way it wants, if one does."""
        # The trains going its way that it would follow onto track they hold, by how much further
        # along this train's path than along theirs each place lies: it may not be set a way past
        # where theirs is set.
        # A train it ran on beside on the other track of quadruple track it no longer follows.
        leaders: dict[int, tuple[Train, float]] = {}
        for held in train.claims:
            crossing = _crossing(held)
            leaders = self._still_ahead(leaders, crossing)
            leaders |= self._going_my_way(train, crossing)
        for crossing in wanted:
            leaders = self._still_ahead(leaders, crossing)
            holder = self._holder(train, crossing, follow=True)
            while holder is not None and train.first and self._give_way(holder, crossing):
                holder = self._holder(train, crossing, follow=True)
            alongside = self._going_my_way(train, crossing)
            if holder is None and not leaders.keys() <= alongside.keys():
                holder = next(iter(leaders.values()))[0]
            if holder is not None:
                return holder
            leaders |= alongside
        if leaders and not self._clear_behind(train, crossings, stop, leaders):
            return max(leaders.values(), key=lambda pair: pair[0].along + pair[1])[0]
        for leader, _ in leaders.values():
            if self._needs_later(leader, wanted):
                return leader
        if self._kept:
            standing = wanted
            if not leaders:
                standing = [
                    crossing
                    for crossing in crossings
                    if crossing.end > stop - train.length + SAME and crossing.start < stop - SAME
                ]
            arriving = stop >= self.network.length(train) - SAME
            for other, way in self._kept:
                if other is train or not self._crosses(standing, way):
                    continue
                if not (arriving and not leaders and self._ends_with(other, way)):
                    return other
        if not train.first:
            return self._queued_before(train, wanted)
        return None

    def _still_ahead(
        self, leaders: dict[int, tuple[Train, float]], crossing: Pass
    ) -> dict[int, tuple[Train, float]]:
        """The trains a train follows but those that are set the way across this cell of
        quadruple track on its other track going the same way: from there they run side by
        side, so it may pass them."""
        if not leaders or not self.blocks.takes_lane(crossing.cell, crossing.entry, crossing.exit):
            return leaders
        way = (crossing.entry, crossing.exit)
        beside = {
            id(other)
            for other, entry, exit_, _, lane in self._held.get(crossing.cell, ())
            if (entry, exit_) == way and lane != crossing.lane
        }
        return {key: pair for key, pair in leaders.items() if key not in beside}

    def _ends_with(self, train: Train, way: list[Pass]) -> bool:
        """Whether a train's path ends where this way on of it does: a train coming to the end
        of its path there gives up all its track as it gets in, so it may stand on that way
        as it does, but not where the train keeping it turns back to come out again."""
        return not any(
            crossing.start > way[-1].start + SAME for crossing in passes(self.network, train)
        )

    def _crosses(self, wanted: list[Pass], way: list[Pass]) -> bool:
        """Whether a way a train wants crosses another way, on track that can't be shared."""
        theirs = [(crossing.cell, (crossing.entry, crossing.exit)) for crossing in way]
        return any(
            cell == crossing.cell
            and not self.blocks.apart(cell, (crossing.entry, crossing.exit), other)
            for crossing in wanted
            for cell, other in theirs
        )

    def _needs_later(self, leader: Train, wanted: list[Pass]) -> bool:
        """Whether a train ahead has yet to be set a way across track that this way crosses, as
        when it turns back in a station too short for it to stand in: a train following it
        there must leave that track to it, or neither could go on."""
        mine = [(crossing.cell, (crossing.entry, crossing.exit)) for crossing in wanted]
        return any(
            cell == later.cell and not self.blocks.apart(cell, way, (later.entry, later.exit))
            for later in passes(self.network, leader)
            if later.start >= leader.cleared - SAME
            for cell, way in mine
        )

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
        for cell_given, entry, exit_, _, _, _, lane in given:
            for holder, *way in self._held.get(cell_given, ()):
                if holder is not other and (way[0], way[1], way[3]) == (entry, exit_, lane):
                    return False
        other.claims = [claim for claim in other.claims if claim[3] < back - SAME]
        for claim in given:
            self._let_go(other, claim)
        other.cleared = back
        return True

    def _queued_before(self, train: Train, wanted: list[Pass]) -> Train | None:
        """A train that asked before this one and still waits for track this one wants, which it
        must leave to it: unless this one holds track that train waits for, or that train
        waits, through the trains each waits for in turn, on this one, or a third train this one
        would not follow holds some of its way, so it could not go now anyway (_held_up)."""
        mine = {crossing.cell: (crossing.entry, crossing.exit) for crossing in wanted}
        holding = {claim[0] for claim in train.claims}
        for other, theirs in self._waiting:
            if any(crossing.cell in holding for crossing in theirs) or self._waits_on(other, train):
                continue
            if self._held_up(other, theirs, train, mine.keys()):
                continue
            for crossing in theirs:
                way = mine.get(crossing.cell)
                if way is not None and not self.blocks.apart(
                    crossing.cell, way, (crossing.entry, crossing.exit)
                ):
                    return other
        return None

    def _held_up(
        self, other: Train, theirs: list[Pass], train: Train, wanted: Iterable[Cell]
    ) -> bool:
        """Whether a train waiting for this way could not be let on now anyway, as a train other
        than `train` holds track of it, on none of the cells `train` wants: as a dispatcher
        would, the track goes to a train that can go instead of being kept for one that can't,
        but not to one that would follow the train in the way onto its track, keeping the
        waiting one out longer, as a line of trains one way would on a single line."""
        cells = set(wanted)
        for crossing in theirs:
            holder = self._holder(other, crossing)
            if holder is None or holder is train:
                continue
            if not any(held is holder for cell in cells for held, *_ in self._held.get(cell, ())):
                return True
        return False

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
            if self._line_end(train, crossings, crossing.start):
                return crossing.start
        return length

    def _line_end(self, train: Train, crossings: list[Pass], signal: float) -> bool:
        """Whether the train could stand at this signal at the end of a single line (`_way_on`):
        only while no other train holds any of the way on from there, nor keeps it for itself
        so, so that none of the trains it waits for could wait for it; none stands on it later
        (`_kept_off`)."""
        way = self._way_on(train, crossings, signal)
        if way is None or any(self._holder(train, crossing) is not None for crossing in way):
            return False
        return not any(
            other is not train and self._crosses(way, kept) for other, kept in self._kept
        )

    def _way_on(self, train: Train, crossings: list[Pass], signal: float) -> list[Pass] | None:
        """The way on from a signal at the end of a single line, with all of the train on that
        line, where as only trains wanting that line could want the track it holds, it is in no
        one's way while that way is free: up to where the train could stand on double track or
        in a station, across any junctions and single line between, as long as it never comes
        back onto that line."""
        blocks = self.blocks
        tail, ahead = signal - train.length + SAME, signal - SAME
        behind = {
            blocks.block.get(crossing.cell)
            for crossing in crossings
            if not crossing.touch and crossing.end > tail and crossing.start < ahead
        }
        if len(behind) != 1:
            return None
        (line,) = behind
        if line is None or line[0] != "single":
            return None
        way: list[Pass] = []
        before = None
        for crossing in crossings:
            if crossing.start < signal - SAME:
                continue
            if crossing.touch:
                way.append(crossing)
                continue
            block = blocks.block.get(crossing.cell)
            if block == line:
                return None
            changes = before is None or block != before[1] or before[0] < crossing.start - SAME
            before = (crossing.end, block)
            if (
                changes
                and crossing.start > signal + SAME
                and self._on_double(crossings, crossing.start - train.length, crossing.start)
            ):
                return way
            way.append(crossing)
        return way

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
        for claim in gone:
            self._let_go(train, claim)

    def _let_go(self, train: Train, claim: Claim) -> None:
        """Stops the train holding the track of one of its claims."""
        cell, entry, exit_, _, _, soft, lane = claim
        if soft:
            return
        held = self._held.get(cell, [])
        for index, (other, other_entry, other_exit, _, other_lane) in enumerate(held):
            if other is train and (other_entry, other_exit, other_lane) == (entry, exit_, lane):
                del held[index]
                return

    def limit(self, train: Train) -> tuple[float, Train | None]:
        """How far along its path the train may run on the way set for it, FOLLOW_GAP short of
        the tail of any train going its way ahead of it on track they share, and never past its
        front, as in a station where it could otherwise pass the train it was let on behind;
        with that train."""
        length = self.network.length(train)
        limit = length if train.cleared >= length - SAME else train.cleared - SIGNAL_GAP
        ahead = None
        for cell, entry, exit_, start, _, _, lane in train.claims:
            for other, other_entry, other_exit, other_start, other_lane in self._held.get(cell, ()):
                if other is train or (other_entry, other_exit, other_lane) != (entry, exit_, lane):
                    continue
                offset = start - other_start
                front = offset + other.along
                if not self._leads(other, train, front - train.along, offset):
                    continue
                tail = start + max(0.0, other.along - other.length - other_start) - FOLLOW_GAP
                stand = min(tail, front)
                if stand < limit:
                    limit, ahead = stand, other
        return max(train.along, limit), ahead


def _claim(crossing: Pass, soft: bool = False) -> Claim:
    """A train's claim on the track of one crossing of its path."""
    return (
        crossing.cell,
        crossing.entry,
        crossing.exit,
        crossing.start,
        crossing.end,
        soft,
        crossing.lane,
    )


def _crossing(claim: Claim) -> Pass:
    """The crossing of a train's path a claim is on."""
    cell, entry, exit_, start, end, _, lane = claim
    return Pass(cell, start, end, entry, exit_, lane=lane)


def cleared_signals(trains: Iterable[Train]) -> set[tuple[Cell, Point]]:
    """The signals showing clear: those into a cell, by the point it is entered by, on the way
    set for a train that has yet to pass them."""
    return {
        (cell, entry)
        for train in trains
        for cell, entry, _, start, _, soft, _ in train.claims
        if not soft and start > train.along
    }
