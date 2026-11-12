import pytest

from railroads.ui.workers import FrameWorker


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
