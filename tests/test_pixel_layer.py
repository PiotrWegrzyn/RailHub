import time
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui
from screens import FakeTexture, world_over_mountains
from test_pixel_world import chunk_of, track_chunk

from railroads.sim.constructions import Track
from railroads.sim.grids import SquareGrid
from railroads.ui.pixel import layer as layer_module
from railroads.ui.pixel.layer import SETTLE_S, PixelLayer, blown_up, middle_out
from railroads.ui.pixel.projection import CHUNK, ChunkPlacement, PixelProjection
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.track_art import no_track


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
    assert np.array_equal(layer._chunks[(*mountains, 2)], world.render(*mountains, 2))


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
    assert np.array_equal(layer._chunks[(*mountains, 2)], world.render(*mountains, 2, True))


def test_a_chunk_drawn_finely_and_let_go_is_drawn_finely_again_when_it_comes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, mountains, plains = world_over_mountains()
    layer = PixelLayer(world, threads=True)
    drawn = (*mountains, 2)
    layer._ask(drawn, time.perf_counter())
    taken_within(layer)
    layer._refine(drawn, time.perf_counter())
    taken_within(layer)
    monkeypatch.setattr(layer_module, "CHUNK_BUDGET", layer._chunks[drawn].nbytes)
    layer._ask((*plains, 2), time.perf_counter())
    taken_within(layer)
    layer._ask(drawn, time.perf_counter())
    taken_within(layer)

    layer._refine(drawn, time.perf_counter())
    taken = taken_within(layer)

    assert taken == {mountains}
    assert np.array_equal(layer._chunks[drawn], world.render(*drawn, True))


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
    assert np.array_equal(doubled[::2, ::2], pixels)
    assert np.array_equal(doubled[1::2, 1::2], pixels)
    assert same is pixels


class Standing:
    def __init__(self) -> None:
        self.asked: list[tuple] = []

    def __call__(self, chunk, scale, k, scenery, track, ground=None) -> FakeTexture:
        self.asked.append((chunk, scale, k, scenery, track, ground))
        return FakeTexture()


def test_where_the_graphics_card_stands_chunks_one_without_track_near_is_made_at_once() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=False)
    layer.stand = stand = Standing()

    layer._ask((*mountains, 8), 0.0)
    layer._upload(mountains, 8, 4)

    [(chunk, scale, k, scenery, track, ground)] = stand.asked
    assert (chunk, scale, k, ground) == (mountains, 8, 4, None)
    assert scenery is world.scenery
    np.testing.assert_array_equal(track, no_track())
    assert not layer._painter.waiting()


def test_where_the_graphics_card_stands_chunks_their_track_is_laid_out_away_from_the_frames() -> (
    None
):
    world, _ = track_chunk(SpriteSheet.load(), [Track((10, 3), (0, 6))])
    chunk = chunk_of(world, (10, 3))
    layer = PixelLayer(world, threads=False)
    layer.stand = stand = Standing()

    drawn_now(layer, (*chunk, 6))
    layer._upload(chunk, 6, 2)

    [(_, _, _, _, track, _)] = stand.asked
    np.testing.assert_array_equal(track, world.track_shapes(*chunk, 6))


def test_where_the_graphics_card_stands_chunks_raised_ground_drawn_finely_goes_under_them() -> None:
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=True)
    layer.stand = stand = Standing()
    layer._ask((*mountains, 4), time.perf_counter())
    layer._refine((*mountains, 2), time.perf_counter(), (*mountains, 4))

    taken_within(layer)
    layer._upload(mountains, 4, 2)

    ground = stand.asked[-1][5]
    np.testing.assert_array_equal(
        ground, world.render(*mountains, 2, fine_relief=True, standing=False)
    )


def test_zooming_in_shows_a_chunks_last_picture_until_it_is_drawn_for_the_new_zoom() -> None:
    gui = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    world, mountains, _ = world_over_mountains()
    layer = PixelLayer(world, threads=False)
    center = world.projection.to_plane(*((np.array(mountains) + 0.5) * CHUNK))
    camera = SimpleNamespace(center_x=float(center[0]), center_y=float(center[1]), pixel_scale=1)
    clip = ((0.0, 0.0), (1280.0, 800.0))
    drawn_now(layer, (*mountains, 1))
    imgui.new_frame()
    layer.draw(imgui.get_foreground_draw_list(), camera, (640.0, 400.0), clip)
    imgui.end_frame()
    camera.pixel_scale = 4

    imgui.new_frame()
    complete = layer.draw(imgui.get_foreground_draw_list(), camera, (640.0, 400.0), clip)
    imgui.end_frame()

    imgui.destroy_context(gui)
    assert not complete
    assert layer._shown_before(mountains) is not None
