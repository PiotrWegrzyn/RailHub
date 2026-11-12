import threading
import time

from railroads.ui.pixel.painter import FORGET_S, Painter


def test_urgent_drawings_come_first_and_the_latest_asked_for_among_them() -> None:
    painter = Painter(threads=False)
    painter.want("old", lambda: "old", True, 1.0)
    painter.want("new", lambda: "new", True, 1.2)
    painter.want("fine", lambda: "fine", False, 1.3)

    for _ in range(3):
        painter.work_once(1.3)
    order = [key for key, _, _ in painter.finished()]

    assert order == ["new", "old", "fine"]


def test_a_drawing_nobody_asked_for_again_is_forgotten() -> None:
    painter = Painter(threads=False)
    painter.want("gone", lambda: "gone", True, 1.0)
    painter.want("here", lambda: "here", True, 1.0 + FORGET_S * 2)

    painter.work_once(1.0 + FORGET_S * 2)
    painter.work_once(1.0 + FORGET_S * 2)
    done = painter.finished()

    assert [key for key, _, _ in done] == ["here"]
    assert painter.waiting() == 0


def test_asking_again_keeps_one_drawing() -> None:
    painter = Painter(threads=False)
    calls = []
    painter.want("chunk", lambda: calls.append(1), True, 1.0)
    painter.want("chunk", lambda: calls.append(2), True, 1.1)

    painter.work_once(1.1)
    painter.work_once(1.1)

    assert calls == [1]


def test_a_failed_drawing_hands_back_its_error() -> None:
    painter = Painter(threads=False)

    def broken() -> None:
        raise KeyError("moved")

    painter.want("chunk", broken, True, 1.0)
    painter.work_once(1.0)
    [(key, result, error)] = painter.finished()

    assert key == "chunk"
    assert result is None
    assert isinstance(error, KeyError)


def test_with_a_thread_drawings_are_done_without_the_frame() -> None:
    painter = Painter(threads=True)
    done = threading.Event()

    def work() -> str:
        done.set()
        return "drawn"

    painter.want("chunk", work, True, time.perf_counter())
    done.wait(5)
    deadline = time.perf_counter() + 5
    finished: list = []
    while not finished and time.perf_counter() < deadline:
        time.sleep(0.005)
        finished = painter.finished()

    assert finished == [("chunk", "drawn", None)]
