"""Draws chunks away from the frames. Each frame asks again for the drawings it still lacks; the
painter does urgent ones first and, among them, the one asked for most recently, so the part of
the map in view now fills in before parts already panned past, and forgets drawings nobody has
asked for in a while.

Where threads can start, one thread of its own does the drawing. Where they can't, as in a
browser, the frame calls `work_once` to do one drawing when it can spare the time."""

import threading
import weakref
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

from railroads.ui.workers import threads_start

# How long a drawing waits to be asked for again before it is forgotten.
FORGET_S = 1.0


@dataclass
class _Job:
    work: Callable[[], object]
    urgent: bool
    wanted: float


@dataclass
class _Queue:
    """What the painter and its thread share, under one lock."""

    lock: threading.Condition = field(default_factory=threading.Condition)
    jobs: dict[Hashable, _Job] = field(default_factory=dict)
    done: list[tuple[Hashable, object, BaseException | None]] = field(default_factory=list)
    busy: Hashable | None = None
    now: float = 0.0
    stopped: bool = False

    def most_wanted(self) -> tuple[Hashable, _Job] | None:
        """Takes the most wanted job, forgetting those asked for too long ago."""
        stale = [key for key, job in self.jobs.items() if job.wanted < self.now - FORGET_S]
        for key in stale:
            del self.jobs[key]
        if not self.jobs:
            return None
        key = max(self.jobs, key=lambda k: (self.jobs[k].urgent, self.jobs[k].wanted))
        return key, self.jobs.pop(key)

    def next(self) -> tuple[Hashable, _Job] | None:
        """Takes the most wanted job to do it on this thread."""
        taken = self.most_wanted()
        if taken is not None:
            self.busy = taken[0]
        return taken

    def run(self, key: Hashable, job: _Job) -> None:
        try:
            result, error = job.work(), None
        except Exception as raised:  # noqa: BLE001 - the asker reads it and asks again
            result, error = None, raised
        with self.lock:
            self.done.append((key, result, error))
            self.busy = None


def _paint(queue: _Queue) -> None:
    while True:
        with queue.lock:
            while not queue.stopped and not queue.jobs:
                queue.lock.wait()
            if queue.stopped:
                return
            taken = queue.next()
        if taken is not None:
            queue.run(*taken)


class Painter:
    def __init__(self, threads: bool | None = None) -> None:
        self._queue = _Queue()
        self.threaded = threads_start() if threads is None else threads
        if self.threaded:
            threading.Thread(
                target=_paint, args=(self._queue,), name="painter", daemon=True
            ).start()
            weakref.finalize(self, _stop, self._queue)

    def want(self, key: Hashable, work: Callable[[], object], urgent: bool, now: float) -> None:
        """Asks for a drawing, or asks again for one already waiting; one being drawn or drawn
        and not yet taken is left alone."""
        queue = self._queue
        with queue.lock:
            queue.now = max(queue.now, now)
            if key == queue.busy or any(key == done[0] for done in queue.done):
                return
            job = queue.jobs.get(key)
            if job is None:
                queue.jobs[key] = _Job(work, urgent, now)
                queue.lock.notify()
            else:
                job.wanted = now
                job.urgent = job.urgent or urgent

    def waiting(self) -> int:
        with self._queue.lock:
            return len(self._queue.jobs) + (self._queue.busy is not None)

    def finished(self) -> list[tuple[Hashable, object, BaseException | None]]:
        """The drawings done since last asked, each with its error if it failed."""
        with self._queue.lock:
            done, self._queue.done = self._queue.done, []
        return done

    def work_once(self, now: float) -> None:
        """Does the most wanted drawing on the calling thread; only where there is no thread."""
        queue = self._queue
        with queue.lock:
            queue.now = max(queue.now, now)
            taken = queue.next()
        if taken is not None:
            queue.run(*taken)


def _stop(queue: _Queue) -> None:
    with queue.lock:
        queue.stopped = True
        queue.lock.notify_all()
