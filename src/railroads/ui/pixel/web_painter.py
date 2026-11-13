"""Paints the map's chunks in web workers, each running a Python of its own with a copy of the
pixel world (railroads.ui.pixel.copies), so in a browser, where Python can't start threads, no
frame stops to paint a chunk and several chunks are painted at once.

The page starts the workers (web/painter.js) while it loads and hands them over as
window.railroadsPainters (web/index.html), which posts messages to them and keeps what they send
back until it is taken. The workers paint for the map shown last: its painter sends them its
world, then whatever changed in it before any chunk asked for after the change, and hands them the
chunks the frames want, the most wanted first and a couple to each at a time, so a worker starts
the next the moment one is done. Finer redraws of raised ground (jobs that are not urgent, see
railroads.ui.pixel.layer) wait until every chunk wanted urgently is painted, and some workers are
kept free of them for the chunks the view needs next.

WorkerPainter has the interface of railroads.ui.pixel.painter.Painter. Until a worker is ready,
and if every worker fails, chunks are painted on the frames the way Painter does there; a chunk a
worker could not paint is painted on a frame too."""

import itertools
import sys
import weakref
from collections.abc import Hashable
from functools import cache
from typing import Any, Protocol

import numpy as np

from railroads.ui.pixel.copies import ChunkJob, Seen, update_message, world_message
from railroads.ui.pixel.painter import _Job, _Queue
from railroads.ui.pixel.world import PixelWorld

if sys.platform == "emscripten":
    import js
    from pyodide.ffi import to_js

# Jobs a worker holds at once: the one it paints and the one it starts as soon as that is done.
JOBS_PER_WORKER = 2

type Message = dict[str, Any]
type Done = tuple[Hashable, object, BaseException | None]


class Page(Protocol):
    """The workers as the page holds them. Messages are dicts of plain values and bytes: posted
    to one worker by its number, and taken back from all of them, each naming its `worker` and
    what `kind` of message it is."""

    @property
    def count(self) -> int: ...

    def post(self, worker: int, message: Message) -> None: ...

    def take(self) -> list[Message]: ...


class Workers:
    """The page's workers, which paint for one painter at a time: the painter of the map shown
    last, which is handed everything they send."""

    def __init__(self, page: Page) -> None:
        self.page = page
        # Workers that have started and not failed, in the order they started, and those that
        # failed.
        self.ready: list[int] = []
        self.dead: set[int] = set()
        self._painter: weakref.ref[WorkerPainter] | None = None
        self._worlds = itertools.count(1)

    @property
    def painter(self) -> WorkerPainter | None:
        return None if self._painter is None else self._painter()

    @property
    def starting(self) -> int:
        """How many workers are still starting."""
        return self.page.count - len(self.ready) - len(self.dead)

    def serve(self, painter: WorkerPainter) -> int:
        """Has the workers paint for `painter` from now on, and no longer for the one before;
        returns the number `painter`'s world goes by."""
        previous = self.painter
        self._painter = weakref.ref(painter)
        if previous is not None:
            previous.dropped()
        return next(self._worlds)

    def poll(self) -> None:
        """Takes what the workers sent and hands it to the painter they paint for."""
        for message in self.page.take():
            worker, kind = message["worker"], message["kind"]
            if kind == "ready":
                self.ready.append(worker)
            elif kind == "dead":
                self.dead.add(worker)
                if worker in self.ready:
                    self.ready.remove(worker)
                print(f"A painting worker failed: {message.get('error')}", file=sys.stderr)
            painter = self.painter
            if painter is not None:
                painter.heard(message)


class PaintFailed(Exception):
    """A chunk a worker could not paint."""


class WorkerPainter:
    """Paints chunks (railroads.ui.pixel.copies.ChunkJob) in the page's workers, with the
    interface of railroads.ui.pixel.painter.Painter."""

    def __init__(self, world: PixelWorld, workers: Workers) -> None:
        self._world = world
        self._workers = workers
        self._queue = _Queue()
        # The workers holding this world and what they have seen of its changes, and the world
        # as last sent, kept for workers still starting.
        self._holding: set[int] = set()
        self._seen = Seen.now(world)
        self._sent_world: tuple[Seen, bytes] | None = None
        # Jobs given to workers by their numbers, each with its key and its worker.
        self._given: dict[int, tuple[Hashable, _Job, int]] = {}
        self._numbers = itertools.count()
        # Jobs a worker could not paint, to paint on the frames instead.
        self._failed: list[tuple[Hashable, _Job]] = []
        self._number = workers.serve(self)
        self._join(list(workers.ready))

    @property
    def threaded(self) -> bool:
        """Whether workers paint, so the frames need not."""
        return bool(self._holding)

    def want(self, key: Hashable, work: ChunkJob, urgent: bool, now: float) -> None:
        """Asks for a chunk, or asks again for one already waiting; one being painted, or
        painted and not yet taken, is left alone."""
        queue = self._queue
        queue.now = max(queue.now, now)
        if self._taken(key):
            return
        job = queue.jobs.get(key)
        if job is None:
            queue.jobs[key] = _Job(work, urgent, now)
        else:
            job.wanted = now
            job.urgent = job.urgent or urgent
        self._give_out()

    def waiting(self) -> int:
        queue = self._queue
        return len(queue.jobs) + len(self._given) + len(self._failed) + (queue.busy is not None)

    def finished(self) -> list[Done]:
        """The chunks painted since last asked, each with its error if it failed."""
        if self._workers.painter is self:
            self._workers.poll()
            self._give_out()
        if self._failed:
            self._queue.run(*self._failed.pop(0))
        done, self._queue.done = self._queue.done, []
        return done

    def work_once(self, now: float) -> None:
        """Paints the most wanted chunk on the calling frame; for while no worker paints."""
        queue = self._queue
        queue.now = max(queue.now, now)
        taken = queue.next()
        if taken is not None:
            queue.run(*taken)

    def heard(self, message: Message) -> None:
        """Takes in a message from a worker (Workers.poll)."""
        worker, kind = message["worker"], message["kind"]
        if kind == "ready":
            self._join([worker])
        elif kind == "dead":
            self._lose(worker)
        elif message.get("world") == self._number and message.get("job") in self._given:
            key, job, _ = self._given.pop(message["job"])
            if kind == "painted":
                pixels = np.frombuffer(message["pixels"], dtype=np.uint8)
                self._queue.done.append((key, pixels.reshape(message["shape"]), None))
            else:
                error = message.get("error")
                print(f"A worker could not paint {job.work}: {error}", file=sys.stderr)
                self._failed.append((key, job))
            self._give_out()

    def dropped(self) -> None:
        """Gives up the workers to another painter; jobs given to them wait to be painted on
        the frames."""
        self._holding.clear()
        self._sent_world = None
        for key, job, _ in self._given.values():
            self._queue.jobs.setdefault(key, job)
        self._given.clear()

    def _taken(self, key: Hashable) -> bool:
        """Whether a chunk is being painted, or painted and not yet taken."""
        queue = self._queue
        return (
            key == queue.busy
            or any(key == done[0] for done in queue.done)
            or any(key == given[0] for given in self._given.values())
            or any(key == failed[0] for failed in self._failed)
        )

    def _join(self, workers: list[int]) -> None:
        """Sends the world as it is now to these workers, which then paint it."""
        joining = [worker for worker in workers if worker not in self._holding]
        if not joining or self._workers.painter is not self:
            return
        self._catch_up()
        seen = Seen.now(self._world)
        if self._sent_world is None or self._sent_world[0] != seen:
            try:
                self._sent_world = (seen, world_message(self._world))
            except Exception as error:  # noqa: BLE001 - the frames paint instead
                print(f"The workers can't be sent the map: {error}", file=sys.stderr)
                return
        message = {"kind": "world", "world": self._number, "data": self._sent_world[1]}
        for worker in joining:
            self._workers.page.post(worker, message)
            self._holding.add(worker)
        if not self._workers.starting:
            self._sent_world = None
        self._give_out()

    def _lose(self, worker: int) -> None:
        """Takes back the jobs of a worker that failed, for others or the frames to paint."""
        self._holding.discard(worker)
        for number, (key, job, holder) in list(self._given.items()):
            if holder == worker:
                del self._given[number]
                self._queue.jobs.setdefault(key, job)

    def _catch_up(self) -> None:
        """Sends the workers holding the world whatever changed in it since they last heard;
        a worker that gets the world later gets it as it is then."""
        seen, self._seen = self._seen, Seen.now(self._world)
        if not self._holding:
            return
        message = update_message(self._world, seen)
        if message is None:
            return
        update = {"kind": "update", "world": self._number, "data": message}
        for worker in sorted(self._holding):
            self._workers.page.post(worker, update)

    def _give_out(self) -> None:
        """Gives the most wanted jobs to workers with room for more. Jobs that are not urgent
        wait while any urgent one is waiting or being painted, and go only to workers with
        nothing else to do, at most to half of the workers at once."""
        queue = self._queue
        if not self._holding or not queue.jobs or self._workers.painter is not self:
            return
        load = dict.fromkeys(sorted(self._holding), 0)
        for _, _, worker in self._given.values():
            load[worker] += 1
        if min(load.values()) >= JOBS_PER_WORKER:
            return
        self._catch_up()
        while True:
            worker = min(load, key=load.__getitem__)
            if load[worker] >= JOBS_PER_WORKER:
                return
            taken = queue.most_wanted()
            if taken is None:
                return
            key, job = taken
            if not job.urgent and not self._free_for_finer_work(load[worker]):
                queue.jobs[key] = job
                return
            number = next(self._numbers)
            self._given[number] = (key, job, worker)
            load[worker] += 1
            chunk: ChunkJob = job.work
            self._workers.page.post(
                worker,
                {"kind": "paint", "world": self._number, "job": number, "chunk": [*chunk.drawing]},
            )

    def _free_for_finer_work(self, worker_load: int) -> bool:
        given = [job for _, job, _ in self._given.values()]
        if worker_load or any(job.urgent for job in given):
            return False
        return len(given) < max(1, len(self._holding) // 2)


class _PageObject:
    """The page's own object holding its workers (window.railroadsPainters), with messages
    turned from Python and back."""

    def __init__(self, page: Any) -> None:
        self._page = page

    @property
    def count(self) -> int:
        return int(self._page.count)

    def post(self, worker: int, message: Message) -> None:
        self._page.post(worker, to_js(message, dict_converter=js.Object.fromEntries))

    def take(self) -> list[Message]:
        return [message.to_py() for message in self._page.take()]


@cache
def page_workers() -> Workers | None:
    """The workers the page started, if this runs in a page that started any."""
    if sys.platform != "emscripten":
        return None
    page = getattr(js, "railroadsPainters", None)
    if page is None or not int(page.count):
        return None
    return Workers(_PageObject(page))
