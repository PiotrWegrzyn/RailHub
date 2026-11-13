import time
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import hello_imgui
from test_elevation import world_over_mountains

from railroads.sim.grids import SquareGrid
from railroads.ui.pixel.layer import SETTLE_S, PixelLayer, blown_up, middle_out
from railroads.ui.pixel.projection import CHUNK, ChunkPlacement, PixelProjection


class FakeTexture:
    def texture_id(self) -> int:
        return 1


@pytest.fixture(autouse=True)
def no_graphics_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture())


def drawn_now(layer: PixelLayer, drawn: tuple[int, int, int]) -> None:
    layer._ask(drawn, 0.0)
    layer._painter.work_once(0.0)
    layer.take_drawn()


def test_a_chunk_asked_for_is_drawn_and_kept_once_taken() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=False)
    layer._ask((*mountains, 2), 0.0)
    layer._painter.work_once(0.0)

    taken = layer.take_drawn()

    assert taken == {mountains}
    np.testing.assert_array_equal(layer._chunks[(*mountains, 2)], world.render(*mountains, 2))


def taken_within(layer: PixelLayer, seconds: float = 30) -> set:
    deadline = time.perf_counter() + seconds
    taken: set = set()
    while not taken and time.perf_counter() < deadline:
        time.sleep(0.01)
        taken = layer.take_drawn()
    return taken


def test_raised_chunks_are_drawn_again_finely() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=True)
    layer._ask((*mountains, 2), time.perf_counter())
    taken_within(layer)
    layer._refine((*mountains, 2), time.perf_counter())

    taken = taken_within(layer)

    assert taken == {mountains}
    np.testing.assert_array_equal(layer._chunks[(*mountains, 2)], world.render(*mountains, 2, True))


def test_without_threads_raised_chunks_are_not_drawn_again_finely() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=False)
    drawn_now(layer, (*mountains, 2))

    layer._refine((*mountains, 2), 0.0)

    assert layer._painter.waiting() == 0


def test_without_threads_chunks_wait_while_the_mouse_moves() -> None:
    world, _, _ = world_over_mountains()
    layer = PixelLayer(world, threads=False)
    camera = SimpleNamespace(center_x=4.0, center_y=6.0, pixel_scale=2)
    layer._moved(camera, 10.0, (100.0, 200.0))

    settled = layer._moved(camera, 10.0 + SETTLE_S * 2, (100.0, 200.0))
    pointed = layer._moved(camera, 10.0 + SETTLE_S * 2, (101.0, 200.0))

    assert not settled
    assert pointed


def test_a_drawing_asked_for_before_a_change_is_thrown_away() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=False)
    layer._ask((*mountains, 2), 0.0)

    layer.refresh({mountains})
    layer._painter.work_once(0.0)
    taken = layer.take_drawn()

    assert taken == set()
    assert (*mountains, 2) not in layer._chunks


def test_with_threads_a_chunk_asked_for_is_drawn_in_the_background() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=True)
    layer._ask((*mountains, 2), time.perf_counter())

    taken = taken_within(layer)

    assert taken == {mountains}


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


def test_chunks_nearest_the_middle_of_the_view_come_first() -> None:
    projection = PixelProjection.of(SquareGrid(rows=200, cols=200, km_per_cell=2.0), 0.0, 1.0)
    clip = ((0.0, 0.0), (1280.0, 800.0))
    placement = ChunkPlacement.of(projection, (3000.0, 2000.0), 1, (640.0, 400.0), clip)

    order = middle_out(placement, (640.0, 400.0))

    rects = [placement.screen_rect(chunk) for chunk in order]
    off = [
        ((left + right) / 2 - 640) ** 2 + ((top + bottom) / 2 - 400) ** 2
        for (left, top), (right, bottom) in rects
    ]
    assert sorted(order) == sorted(placement.chunks)
    assert order[0] == (3000 // CHUNK, 2000 // CHUNK)
    assert off == sorted(off)


def test_textures_repeat_pixels_into_squares_and_skip_the_copy_at_screen_scale() -> None:
    pixels = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)

    doubled, same = blown_up(pixels, 2), blown_up(pixels, 1)

    assert doubled.shape == (4, 6, 4)
    np.testing.assert_array_equal(doubled[::2, ::2], pixels)
    np.testing.assert_array_equal(doubled[1::2, 1::2], pixels)
    assert same is pixels


def test_where_the_graphics_card_blows_textures_up_only_its_own_pixels_go_up() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=False)
    drawn_now(layer, (*mountains, 2))
    asked: list[tuple[np.ndarray, int]] = []
    layer.blow_up = lambda pixels, times: asked.append((pixels, times)) or FakeTexture()

    layer._upload(mountains, 4)
    layer._upload(mountains, 2)

    [(pixels, times)] = asked
    assert times == 2
    assert pixels is layer._chunks[(*mountains, 2)]
    assert set(layer._textures) == {(*mountains, 4), (*mountains, 2)}
