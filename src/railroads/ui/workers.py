"""Work done away from the frames: on a thread of its own where threads can start, otherwise, as
in a browser, a job at a time on frames the caller picks. A FrameWorker also takes work done in
steps (railroads.sim.steps) and does as many of them on a frame as fit in the time it is given
there, so a long job such as writing a save or growing a new game's towns never holds up one
frame for long."""

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor

from railroads.sim.steps import Steps, finish

# About how long a frame gives to work done in steps.
FRAME_SECONDS = 0.03


def threads_start() -> bool:
    try:
        thread = threading.Thread(target=lambda: None)
        thread.start()
    except RuntimeError:
        return False
    thread.join()
    return True


class FrameWorker(Executor):
    """Does the submitted work on the calling thread when asked to: a plain job whole, a job done
    in steps a few steps at a time."""

    def __init__(self) -> None:
        self._jobs: deque[tuple[Future, Callable[[], object] | Steps[object]]] = deque()

    def submit[T](self, fn: Callable[..., T], /, *args: object, **kwargs: object) -> Future[T]:
        job: Future[T] = Future()
        self._jobs.append((job, lambda: fn(*args, **kwargs)))
        return job

    def submit_steps[T](self, steps: Steps[T]) -> Future[T]:
        job: Future[T] = Future()
        self._jobs.append((job, steps))
        return job

    def work_once(self, seconds: float = math.inf) -> None:
        """Works on the first job: does a plain one whole, and steps of one done in steps until
        `seconds` have passed, at least one step and all of them by default."""
        if not self._jobs:
            return
        job, work = self._jobs[0]
        try:
            if callable(work):
                result = work()
            else:
                done, result = _step_for(work, seconds)
                if not done:
                    return
        except Exception as error:  # noqa: BLE001 - the job's caller reads it from the future
            self._jobs.popleft()
            job.set_exception(error)
            return
        self._jobs.popleft()
        job.set_result(result)


def _step_for[T](steps: Steps[T], seconds: float) -> tuple[bool, T | None]:
    """Does steps until `seconds` have passed, at least one; returns whether they are all done,
    and their result if so."""
    until = time.perf_counter() + seconds
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return True, done.value
        if time.perf_counter() >= until:
            return False, None


def submit_steps[T](executor: Executor, steps: Callable[[], Steps[T]]) -> Future[T]:
    """Has an executor do work done in steps: a FrameWorker a few steps a frame, a thread all of
    them at once."""
    if isinstance(executor, FrameWorker):
        return executor.submit_steps(steps())
    return executor.submit(lambda: finish(steps()))


def worker(name: str, threads: bool | None = None) -> Executor:
    """One thread for jobs taken in turn, or a FrameWorker where threads cannot start."""
    if threads is None:
        threads = threads_start()
    if threads:
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
    return FrameWorker()
