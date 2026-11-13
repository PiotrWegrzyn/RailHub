import ctypes
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_elevation import world_over_mountains

from railroads.sim.constructions import Building, BuildingKind
from railroads.ui.pixel import copies, web_painter
from railroads.ui.pixel import layer as layer_module
from railroads.ui.pixel.copies import ChunkJob, WorldCopy
from railroads.ui.pixel.layer import PixelLayer, painter_for
from railroads.ui.pixel.painter import Painter
from railroads.ui.pixel.web_painter import (
    JOBS_PER_WORKER,
    WorkerPainter,
    Workers,
    _PageObject,
    page_workers,
)
from railroads.ui.pixel.world import PixelWorld


class FakePage:
    """The page's workers, each taking its messages in turn into a copy of the world when told
    to work, as web/painter.js does."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.copies = [WorldCopy() for _ in range(count)]
        self.queues: list[list[dict[str, Any]]] = [[] for _ in range(count)]
        self.inbox: list[dict[str, Any]] = []
        self.refusing: set[int] = set()

    def post(self, worker: int, message: dict[str, Any]) -> None:
        if worker in self.refusing:
            raise RuntimeError("Array buffer allocation failed")
        self.queues[worker].append(message)

    def take(self) -> list[dict[str, Any]]:
        taken, self.inbox = self.inbox, []
        return taken

    def start(self, *workers: int) -> None:
        self.inbox += [{"worker": worker, "kind": "ready"} for worker in workers]

    def fail(self, worker: int) -> None:
        self.queues[worker] = []
        self.inbox.append({"worker": worker, "kind": "dead", "error": "out of memory"})

    def kinds(self, worker: int) -> list[str]:
        return [message["kind"] for message in self.queues[worker]]

    def work(self, *workers: int) -> None:
        for worker in workers or range(self.count):
            queue, self.queues[worker] = self.queues[worker], []
            copy = self.copies[worker]
            for message in queue:
                kind, number = message["kind"], message["world"]
                if kind == "world":
                    copy.start(number, message["data"])
                elif kind == "update":
                    copy.update(number, message["data"])
                else:
                    pixels = copy.paint(number, message["chunk"])
                    answer = {"worker": worker, "world": number, "job": message["job"]}
                    if pixels is None:
                        self.inbox.append(answer | {"kind": "failed", "error": "old world"})
                    else:
                        shape = list(pixels.shape)
                        self.inbox.append(
                            answer | {"kind": "painted", "pixels": pixels.tobytes(), "shape": shape}
                        )


def job(world: PixelWorld, chunk: tuple[int, int]) -> ChunkJob:
    return ChunkJob(world, *chunk, 2)


def painted(painter: WorkerPainter) -> dict[Any, np.ndarray]:
    return {key: pixels for key, pixels, _ in painter.finished()}


def test_until_a_worker_is_ready_chunks_are_painted_on_the_frames() -> None:
    world, mountains, _ = world_over_mountains()
    page = FakePage(2)
    painter = WorkerPainter(world, Workers(page))
    painter.want("chunk", job(world, mountains), True, 1.0)

    painter.work_once(1.0)
    done = painted(painter)

    assert not painter.threaded
    np.testing.assert_array_equal(done["chunk"], world.render(*mountains, 2))


def test_a_ready_worker_paints_the_chunks_and_one_still_starting_only_holds_the_world() -> None:
    world, mountains, plains = world_over_mountains()
    page = FakePage(2)
    painter = WorkerPainter(world, Workers(page))
    page.start(0)
    painter.finished()
    painter.want("mountains", job(world, mountains), True, 1.0)
    painter.want("plains", job(world, plains), True, 1.0)
    starting = page.kinds(1)

    page.work()
    done = painted(painter)

    assert painter.threaded
    assert starting == ["world"]
    np.testing.assert_array_equal(done["mountains"], world.render(*mountains, 2))
    np.testing.assert_array_equal(done["plains"], world.render(*plains, 2))
    assert painter.waiting() == 0


def test_workers_ready_before_a_map_is_made_get_its_world_as_it_is_made() -> None:
    world, _, _ = world_over_mountains()
    page = FakePage(2)
    page.start(0, 1)

    painter = WorkerPainter(world, Workers(page))

    assert painter.threaded
    assert page.kinds(0) == page.kinds(1) == ["world"]


def test_the_most_wanted_jobs_go_out_first_a_couple_to_each_worker() -> None:
    world, mountains, plains = world_over_mountains()
    page = FakePage(1)
    painter = WorkerPainter(world, Workers(page))
    chunks = [mountains, plains, (mountains[0], mountains[1] + 1)]
    for when, chunk in enumerate(chunks):
        painter.want(chunk, job(world, chunk), True, 1.0 + when / 10)

    page.start(0)
    painter.finished()
    given = [tuple(message["chunk"][:2]) for message in page.queues[0] if "chunk" in message]

    assert given == chunks[::-1][:JOBS_PER_WORKER]
    assert painter.waiting() == len(chunks)


def test_changes_reach_a_worker_before_any_chunk_asked_for_after_them() -> None:
    world, _, plains = world_over_mountains()
    page = FakePage(1)
    page.start(0)
    painter = WorkerPainter(world, Workers(page))
    painter.finished()
    page.work()
    world.constructions.build(Building((10, 4), BuildingKind.HOUSE))
    world.buildings_changed([(10, 4)])

    painter.want("plains", job(world, plains), True, 1.0)
    kinds = page.kinds(0)
    page.work()
    done = painted(painter)

    assert kinds == ["update", "paint"]
    np.testing.assert_array_equal(done["plains"], world.render(*plains, 2))


def test_a_layer_catching_up_sends_a_change_so_the_chunks_asked_for_next_go_out_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, _, plains = world_over_mountains()
    page = FakePage(1)
    page.start(0)
    monkeypatch.setattr(layer_module, "threads_start", lambda: False)
    monkeypatch.setattr(layer_module, "page_workers", lambda: Workers(page))
    layer = PixelLayer(world)
    layer.take_drawn()
    page.work()
    world.constructions.build(Building((10, 4), BuildingKind.HOUSE))
    layer.refresh(world.buildings_changed([(10, 4)]))

    layer.catch_up()
    caught_up = page.kinds(0)
    layer._ask((*plains, 2), 1.0)

    assert caught_up == ["update"]
    assert page.kinds(0) == ["update", "paint"]


def test_a_failed_workers_chunks_go_to_another() -> None:
    world, mountains, plains = world_over_mountains()
    page = FakePage(2)
    page.start(0, 1)
    painter = WorkerPainter(world, Workers(page))
    painter.finished()
    page.work()
    for chunk in (mountains, plains):
        painter.want(chunk, job(world, chunk), True, 1.0)
    lost = [tuple(message["chunk"][:2]) for message in page.queues[0]]

    page.fail(0)
    painter.finished()
    page.work()
    done = painted(painter)

    assert lost
    assert set(done) == {mountains, plains}
    np.testing.assert_array_equal(done[lost[0]], world.render(*lost[0], 2))


def test_once_every_worker_failed_the_frames_paint_again() -> None:
    world, mountains, _ = world_over_mountains()
    page = FakePage(1)
    page.start(0)
    painter = WorkerPainter(world, Workers(page))
    painter.finished()
    painter.want("chunk", job(world, mountains), True, 1.0)

    page.fail(0)
    painter.finished()
    threaded = painter.threaded
    painter.work_once(1.0)
    done = painted(painter)

    assert not threaded
    np.testing.assert_array_equal(done["chunk"], world.render(*mountains, 2))


def test_a_chunk_a_worker_could_not_paint_is_painted_on_a_frame(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world, mountains, _ = world_over_mountains()
    page = FakePage(1)
    page.start(0)
    painter = WorkerPainter(world, Workers(page))
    painter.finished()
    page.work()
    page.copies[0].world = None
    painter.want("chunk", job(world, mountains), True, 1.0)

    page.work()
    done = painted(painter)

    np.testing.assert_array_equal(done["chunk"], world.render(*mountains, 2))
    assert "could not paint" in capsys.readouterr().err


def test_pixels_that_dont_fill_their_shape_count_as_a_chunk_not_painted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world, mountains, _ = world_over_mountains()
    page = FakePage(1)
    page.start(0)
    painter = WorkerPainter(world, Workers(page))
    painter.finished()
    page.work()
    painter.want("chunk", job(world, mountains), True, 1.0)
    page.work()
    page.inbox[-1]["pixels"] = page.inbox[-1]["pixels"][:100]

    done = painted(painter)

    np.testing.assert_array_equal(done["chunk"], world.render(*mountains, 2))
    assert "wrong size" in capsys.readouterr().err


def test_a_new_map_takes_the_workers_over_and_the_old_one_paints_on_its_frames() -> None:
    old, mountains, _ = world_over_mountains()
    new, _, plains = world_over_mountains()
    new.constructions.build(Building((10, 4), BuildingKind.HOUSE))
    new.buildings_changed([(10, 4)])
    page = FakePage(1)
    page.start(0)
    workers = Workers(page)
    first = WorkerPainter(old, workers)
    first.finished()
    first.want("old", job(old, mountains), True, 1.0)

    second = WorkerPainter(new, workers)
    second.want("new", job(new, plains), True, 1.0)
    page.work()
    done = painted(second)

    assert not first.threaded
    assert first.waiting() == 1
    assert set(done) == {"new"}
    np.testing.assert_array_equal(done["new"], new.render(*plains, 2))


def test_workers_still_starting_get_the_world_as_the_map_is_made_and_never_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, _, _ = world_over_mountains()
    page = FakePage(2)
    made: list[PixelWorld] = []
    monkeypatch.setattr(
        web_painter, "world_message", lambda w: made.append(w) or copies.world_message(w)
    )
    painter = WorkerPainter(world, Workers(page))
    sent = [page.kinds(0), page.kinds(1)]

    page.start(0, 1)
    painter.finished()

    assert sent == [["world"], ["world"]]
    assert painter.threaded
    assert page.kinds(0) == page.kinds(1) == ["world"]
    assert made == [world]


def test_a_worker_ready_later_paints_the_world_as_it_is_then() -> None:
    world, mountains, plains = world_over_mountains()
    page = FakePage(2)
    page.start(0)
    painter = WorkerPainter(world, Workers(page))
    painter.finished()
    page.work()
    world.constructions.build(Building((10, 4), BuildingKind.HOUSE))
    world.buildings_changed([(10, 4)])

    page.start(1)
    painter.finished()
    painter.want("mountains", job(world, mountains), True, 1.0)
    painter.want("plains", job(world, plains), True, 1.0)
    painting = [page.kinds(worker).count("paint") for worker in (0, 1)]
    page.work()
    done = painted(painter)

    assert painting == [1, 1]
    np.testing.assert_array_equal(done["mountains"], world.render(*mountains, 2))
    np.testing.assert_array_equal(done["plains"], world.render(*plains, 2))


def test_a_worker_that_cant_be_posted_to_paints_no_more_and_another_takes_its_chunk(
    capsys: pytest.CaptureFixture[str],
) -> None:
    world, mountains, _ = world_over_mountains()
    page = FakePage(2)
    page.start(0, 1)
    painter = WorkerPainter(world, Workers(page))
    page.work()
    page.refusing.add(0)

    painter.want("chunk", job(world, mountains), True, 1.0)
    page.work()
    done = painted(painter)

    np.testing.assert_array_equal(done["chunk"], world.render(*mountains, 2))
    assert painter.threaded
    assert painter.waiting() == 0
    assert "can't be sent" in capsys.readouterr().err


def test_without_a_page_chunks_are_painted_by_a_thread_or_the_frames() -> None:
    world, _, _ = world_over_mountains()

    painters = [painter_for(world), painter_for(world, threads=False)]

    assert page_workers() is None
    assert all(isinstance(painter, Painter) for painter in painters)


def test_with_workers_the_layer_never_waits_and_draws_raised_chunks_as_the_frames_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, mountains, _ = world_over_mountains()
    page = FakePage(1)
    page.start(0)
    monkeypatch.setattr(layer_module, "threads_start", lambda: False)
    monkeypatch.setattr(layer_module, "page_workers", lambda: Workers(page))
    layer = PixelLayer(world)
    camera = SimpleNamespace(center_x=4.0, center_y=6.0, pixel_scale=2)
    layer._ask((*mountains, 2), 1.0)
    page.work()
    taken = layer.take_drawn()

    layer._refine((*mountains, 2), 1.0)

    assert not layer._moved(camera, 1.0)
    assert taken == {mountains}
    assert layer._painter.waiting() == 0
    np.testing.assert_array_equal(layer._chunks[(*mountains, 2)], world.render(*mountains, 2))


class FakeJsMessage:
    """A message from the page as Pyodide hands it over: its fields read one at a time, or the
    whole of it turned into a dict."""

    def __init__(self, **fields: Any) -> None:
        vars(self).update(fields)

    def to_py(self) -> dict[str, Any]:
        return dict(vars(self))


class FakeJsPage:
    """The page's object holding the workers, which copies bytes from and into Python's memory
    by their address in it."""

    count = 1

    def __init__(self, *messages: FakeJsMessage) -> None:
        self.messages = list(messages)
        self.posted: list[tuple[int, dict[str, Any], bytes | None]] = []

    def post(self, worker: int, message: dict[str, Any], address: int = 0, length: int = 0) -> None:
        data = ctypes.string_at(address, length) if length else None
        self.posted.append((worker, message, data))

    def take(self) -> list[FakeJsMessage]:
        taken, self.messages = self.messages, []
        return taken

    def copy(self, buffer: SimpleNamespace, address: int) -> None:
        if buffer.data is None:
            raise RuntimeError("RangeError: offset is out of bounds")
        ctypes.memmove(address, buffer.data, buffer.byteLength)


@pytest.fixture
def plain_js(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_painter, "to_js", lambda message, dict_converter: message, raising=False
    )
    js = SimpleNamespace(Object=SimpleNamespace(fromEntries=dict))
    monkeypatch.setattr(web_painter, "js", js, raising=False)


def pixels_message(job: int, data: bytes | None, size: int) -> FakeJsMessage:
    buffer = SimpleNamespace(data=data, byteLength=size)
    return FakeJsMessage(worker=0, kind="painted", world=1, job=job, pixels=buffer, shape=[size])


@pytest.mark.usefixtures("plain_js")
def test_bytes_go_to_the_page_by_their_address_in_pythons_memory() -> None:
    page = FakeJsPage()
    workers = _PageObject(page)
    data = bytes(range(256)) * 3

    workers.post(2, {"kind": "world", "world": 1, "data": data})
    workers.post(3, {"kind": "paint", "world": 1, "job": 7, "chunk": [1, 2, 2]})

    assert page.posted == [
        (2, {"kind": "world", "world": 1}, data),
        (3, {"kind": "paint", "world": 1, "job": 7, "chunk": [1, 2, 2]}, None),
    ]


def test_pixels_a_worker_sent_are_copied_into_an_array_by_its_address() -> None:
    data = bytes(range(200))
    page = FakeJsPage(FakeJsMessage(worker=1, kind="ready"), pixels_message(5, data, len(data)))

    taken = _PageObject(page).take()

    assert taken[0] == {"worker": 1, "kind": "ready"}
    assert {key: value for key, value in taken[1].items() if key != "pixels"} == {
        "worker": 0,
        "kind": "painted",
        "world": 1,
        "job": 5,
        "shape": [200],
    }
    assert taken[1]["pixels"].tobytes() == data


def test_a_message_whose_pixels_cant_be_copied_fails_and_the_others_still_come() -> None:
    page = FakeJsPage(pixels_message(5, None, 64), FakeJsMessage(worker=1, kind="dead"))

    taken = _PageObject(page).take()

    assert taken[0]["kind"] == "failed"
    assert taken[0]["job"] == 5
    assert "out of bounds" in taken[0]["error"]
    assert taken[1] == {"worker": 1, "kind": "dead"}
