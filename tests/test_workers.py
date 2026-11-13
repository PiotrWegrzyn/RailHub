import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from railroads.sim.steps import Steps
from railroads.ui.workers import FRAME_SECONDS, FrameWorker, submit_steps


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

    frames.work_once(FRAME_SECONDS)
    done_after_one = job.done()
    for _ in range(10):
        frames.work_once(FRAME_SECONDS)

    assert not done_after_one
    assert job.result() == 10


def test_a_thread_does_a_job_in_steps_all_at_once() -> None:
    with ThreadPoolExecutor(max_workers=1) as thread:
        job = submit_steps(thread, lambda: counting(5))

        assert job.result() == 5


def counted(steps: int, done: list[int]) -> Steps[str]:
    for step in range(steps):
        done.append(step)
        yield
    return "written"


def test_a_frame_worker_does_a_job_in_steps_a_step_at_a_time_when_given_no_time() -> None:
    worker = FrameWorker()
    done: list[int] = []
    job = worker.submit_steps(counted(3, done))

    worker.work_once(0.0)
    after_one = list(done)
    worker.work_once(0.0)
    worker.work_once(0.0)
    worker.work_once(0.0)

    assert after_one == [0]
    assert done == [0, 1, 2]
    assert job.result() == "written"


def test_a_frame_worker_finishes_a_job_in_steps_when_given_all_the_time_it_needs() -> None:
    worker = FrameWorker()
    done: list[int] = []
    job = worker.submit_steps(counted(50, done))
    later = worker.submit(lambda: "next")

    worker.work_once()

    assert job.result() == "written"
    assert len(done) == 50
    assert not later.done()


def test_a_frame_worker_hands_a_job_in_steps_its_error() -> None:
    worker = FrameWorker()

    def failing() -> Steps[None]:
        yield
        raise OSError("disk full")

    job = worker.submit_steps(failing())
    later = worker.submit(lambda: "next")

    worker.work_once(0.0)
    worker.work_once(0.0)
    worker.work_once()

    with pytest.raises(OSError, match="disk full"):
        job.result()
    assert later.result() == "next"
