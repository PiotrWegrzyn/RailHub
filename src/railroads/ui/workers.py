"""Work done away from the frames: on a thread of its own where threads can start, otherwise, as
in a browser, a job at a time on frames the caller picks."""

import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor


def threads_start() -> bool:
    try:
        thread = threading.Thread(target=lambda: None)
        thread.start()
    except RuntimeError:
        return False
    thread.join()
    return True


class FrameWorker(Executor):
    """Does the submitted work on the calling thread, one job each time it is asked to."""

    def __init__(self) -> None:
        self._jobs: deque[tuple[Future, Callable[[], object]]] = deque()

    def submit[T](self, fn: Callable[..., T], /, *args: object, **kwargs: object) -> Future[T]:
        job: Future[T] = Future()
        self._jobs.append((job, lambda: fn(*args, **kwargs)))
        return job

    def work_once(self) -> None:
        if not self._jobs:
            return
        job, work = self._jobs.popleft()
        try:
            job.set_result(work())
        except Exception as error:  # noqa: BLE001 - the job's caller reads it from the future
            job.set_exception(error)


def worker(name: str, threads: bool | None = None) -> Executor:
    """One thread for jobs taken in turn, or a FrameWorker where threads cannot start."""
    if threads is None:
        threads = threads_start()
    if threads:
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
    return FrameWorker()
