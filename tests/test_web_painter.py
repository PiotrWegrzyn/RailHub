from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_elevation import world_over_mountains

from railroads.sim.constructions import Building, BuildingKind
from railroads.ui.pixel import layer as layer_module
from railroads.ui.pixel.copies import ChunkJob, WorldCopy
from railroads.ui.pixel.layer import PixelLayer, painter_for
from railroads.ui.pixel.painter import Painter
from railroads.ui.pixel.web_painter import (
    JOBS_PER_WORKER,
    WorkerPainter,
    Workers,
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

    def post(self, worker: int, message: dict[str, Any]) -> None:
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


def job(world: PixelWorld, chunk: tuple[int, int], fine: bool = False) -> ChunkJob:
    return ChunkJob(world, *chunk, 2, fine_relief=fine)


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


def test_a_ready_worker_gets_the_world_and_paints_its_chunks() -> None:
    world, mountains, plains = world_over_mountains()
    page = FakePage(2)
    painter = WorkerPainter(world, Workers(page))
    page.start(0)
    painter.finished()
    painter.want("mountains", job(world, mountains), True, 1.0)
    painter.want("plains", job(world, plains), True, 1.0)

    page.work()
    done = painted(painter)

    assert painter.threaded
    assert page.kinds(1) == []
    np.testing.assert_array_equal(done["mountains"], world.render(*mountains, 2))
    np.testing.assert_array_equal(done["plains"], world.render(*plains, 2))
    assert painter.waiting() == 0


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


def test_finer_redraws_wait_for_urgent_chunks_and_leave_workers_free() -> None:
    world, mountains, plains = world_over_mountains()
    page = FakePage(4)
    page.start(0, 1, 2, 3)
    painter = WorkerPainter(world, Workers(page))
    painter.finished()
    page.work()
    fine = [(mountains[0] + x, mountains[1]) for x in range(-1, 3)]
    painter.want("plains", job(world, plains), True, 1.0)

    for chunk in fine:
        painter.want(chunk, job(world, chunk, fine=True), False, 1.0)
    while_urgent = sum(kinds.count("paint") for kinds in map(page.kinds, range(4)))
    page.work()
    painter.finished()
    finer = sum(kinds.count("paint") for kinds in map(page.kinds, range(4)))

    assert while_urgent == 1
    assert finer == 2
    assert max(len(page.queues[worker]) for worker in range(4)) == 1


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


def test_a_worker_ready_later_gets_the_world_as_it_is_then() -> None:
    world, _, plains = world_over_mountains()
    page = FakePage(2)
    page.start(0)
    painter = WorkerPainter(world, Workers(page))
    painter.finished()
    page.work()
    world.constructions.build(Building((10, 4), BuildingKind.HOUSE))
    world.buildings_changed([(10, 4)])

    page.start(1)
    painter.finished()
    page.work()

    for copy in page.copies:
        np.testing.assert_array_equal(
            copy.paint(1, (*plains, 2, False, False)), world.render(*plains, 2)
        )


def test_without_a_page_chunks_are_painted_by_a_thread_or_the_frames() -> None:
    world, _, _ = world_over_mountains()

    painters = [painter_for(world), painter_for(world, threads=False)]

    assert page_workers() is None
    assert all(isinstance(painter, Painter) for painter in painters)


def test_with_workers_the_layer_never_waits_and_draws_raised_chunks_again_finely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, mountains, _ = world_over_mountains()
    page = FakePage(1)
    page.start(0)
    monkeypatch.setattr(layer_module, "threads_start", lambda: False)
    monkeypatch.setattr(layer_module, "page_workers", lambda: Workers(page))
    layer = PixelLayer(world)
    layer.take_drawn()
    camera = SimpleNamespace(center_x=4.0, center_y=6.0, pixel_scale=2)
    layer._ask((*mountains, 2), 1.0)
    page.work()
    layer.take_drawn()

    layer._refine((*mountains, 2), 1.0)
    page.work()
    taken = layer.take_drawn()

    assert not layer._moved(camera, 1.0)
    assert taken == {mountains}
    np.testing.assert_array_equal(layer._chunks[(*mountains, 2)], world.render(*mountains, 2, True))
