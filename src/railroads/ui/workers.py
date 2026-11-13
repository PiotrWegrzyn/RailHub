"""Work done away from the frames: on a thread of its own where threads can start, otherwise, as
in a browser, a job at a time on frames the caller picks."""

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
    """Does the submitted work on the calling thread each time it is asked to: a job at once, or
    a job in steps for about FRAME_SECONDS at a time."""

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

    def work_once(self) -> None:
        if not self._jobs:
            return
        job, work = self._jobs[0]
        try:
            if callable(work):
                result = work()
            else:
                started = time.perf_counter()
                while time.perf_counter() - started < FRAME_SECONDS:
                    next(work)
                return
        except StopIteration as done:
            result = done.value
        except Exception as error:  # noqa: BLE001 - the job's caller reads it from the future
            self._jobs.popleft()
            job.set_exception(error)
            return
        self._jobs.popleft()
        job.set_result(result)


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
