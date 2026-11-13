import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from railroads.sim.steps import Steps
from railroads.ui.workers import FrameWorker, submit_steps


def test_a_frame_worker_does_one_job_each_time_it_is_asked() -> None:
    worker = FrameWorker()
    first = worker.submit(lambda: 1)
    second = worker.submit(lambda: 2)

    worker.work_once()

    assert first.result() == 1
    assert not second.done()


def test_a_frame_worker_hands_a_failed_job_its_error() -> None:
    worker = FrameWorker()
    job = worker.submit(int, "not a number")

    worker.work_once()

    with pytest.raises(ValueError, match="not a number"):
        job.result()


def counting(steps: int, slow: float = 0.0) -> Steps[int]:
    for _ in range(steps):
        time.sleep(slow)
        yield
    return steps


def test_a_frame_worker_does_a_job_in_steps_over_several_frames() -> None:
    frames = FrameWorker()
    job = frames.submit_steps(counting(10, slow=0.01))

    frames.work_once()
    done_after_one = job.done()
    for _ in range(10):
        frames.work_once()

    assert not done_after_one
    assert job.result() == 10


def test_a_thread_does_a_job_in_steps_all_at_once() -> None:
    with ThreadPoolExecutor(max_workers=1) as thread:
        job = submit_steps(thread, lambda: counting(5))

        assert job.result() == 5
