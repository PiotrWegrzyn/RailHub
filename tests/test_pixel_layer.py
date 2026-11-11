import numpy as np
import pytest
from imgui_bundle import hello_imgui
from test_elevation import world_over_mountains

from railroads.ui.pixel.layer import PixelLayer


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
