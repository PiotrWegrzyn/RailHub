from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import hello_imgui
from test_elevation import world_over_mountains

from railroads.ui.pixel.layer import SETTLE_S, PixelLayer


class FakeTexture:
    def texture_id(self) -> int:
        return 1


@pytest.fixture(autouse=True)
def no_graphics_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture())


def test_raised_chunks_are_drawn_again_finely_in_the_background() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world)
    layer._upload(mountains, 2)
    layer._refining[(*mountains, 2)][1].result()

    taken = layer.take_refined()

    assert taken == {mountains}
    np.testing.assert_array_equal(layer._chunks[(*mountains, 2)], world.render(*mountains, 2, True))


def test_a_fine_drawing_started_before_a_change_is_thrown_away() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world)
    layer._upload(mountains, 2)
    layer._refining[(*mountains, 2)][1].result()

    layer.refresh({mountains})
    taken = layer.take_refined()

    assert taken == set()
    assert (*mountains, 2) not in layer._chunks


def test_without_threads_a_raised_chunk_waits_for_a_frame_to_be_drawn_finely() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=False)
    layer._upload(mountains, 2)
    waiting = layer.take_refined()

    layer._worker.work_once()
    taken = layer.take_refined()

    assert waiting == set()
    assert taken == {mountains}
    np.testing.assert_array_equal(layer._chunks[(*mountains, 2)], world.render(*mountains, 2, True))


def test_without_threads_new_chunks_wait_until_the_view_has_stood_still() -> None:
    world, _, _ = world_over_mountains()
    layer = PixelLayer(world, threads=False)
    camera = SimpleNamespace(center_x=4.0, center_y=6.0, pixel_scale=2)
    layer._moved(camera, 10.0)

    moving = layer._moved(camera, 10.0 + SETTLE_S / 2)
    settled = layer._moved(camera, 10.0 + SETTLE_S * 2)
    camera.center_x += 1
    moved_again = layer._moved(camera, 10.0 + SETTLE_S * 2)

    assert moving
    assert not settled
    assert moved_again


def test_with_threads_new_chunks_never_wait_for_the_view_to_stand_still() -> None:
    world, _, _ = world_over_mountains()
    layer = PixelLayer(world, threads=True)
    camera = SimpleNamespace(center_x=4.0, center_y=6.0, pixel_scale=2)

    moving = layer._moved(camera, 10.0)

    assert not moving
