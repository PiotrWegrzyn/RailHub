from collections.abc import Iterator

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui  # noqa: TID251

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.game import Game
from railroads.sim.grids import HexGrid
from railroads.sim.trains import Stop, Train
from railroads.ui import map_view
from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS
from railroads.ui.map_view import MapView
from railroads.ui.pixel import train_layer
from railroads.ui.pixel.relief import Elevation
from railroads.ui.pixel.train_layer import TrainLayer
from railroads.ui.pixel.trains import DETAIL, HEADINGS


class FakeTexture:
    def texture_id(self) -> int:
        return 1


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture())
    gui = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    yield
    imgui.destroy_context(gui)


def railway_seen_in_pixel_art() -> MapView:
    game = Game(Constructions(flat_map(grid=HexGrid(rows=20, cols=24, km_per_cell=2.0))))
    game.finances.cash = 1_000_000.0
    for cell in ((10, 6), (10, 16)):
        game.build_station(cell)
    game.lay_track((10, 6), (10, 16))
    view = MapView(game)
    view.camera.zoom_level = 4
    view.look_at((10, 10))
    return view


def frame(view: MapView) -> None:
    imgui.get_io().delta_time = 1 / 60
    imgui.new_frame()
    imgui.set_next_window_pos((0, 0))
    imgui.set_next_window_size((1280, 800))
    imgui.begin("Railroads")
    view.gui()
    imgui.end()
    imgui.render()


def test_without_trains_the_pixel_map_makes_no_train_sheet_and_draws_no_flat_track(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = railway_seen_in_pixel_art()
    layers: list[object] = []
    flat_track: list[object] = []
    monkeypatch.setattr(
        map_view, "TrainLayer", lambda world: layers.append(world) or TrainLayer(world)
    )
    monkeypatch.setattr(map_view, "draw_track", lambda *args: flat_track.append(args))

    frame(view)

    assert layers == []
    assert flat_track == []


def test_the_pixel_map_draws_its_trains_from_the_train_sheet(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = railway_seen_in_pixel_art()
    train = view.railway.buy()
    assert train is not None
    train.stops = [Stop((10, 6)), Stop((10, 16))]
    drawn: list[object] = []
    monkeypatch.setattr(TrainLayer, "draw", lambda layer, _, railway, *__: drawn.append(railway))

    frame(view)

    assert drawn == [view.railway]


def test_the_ground_under_every_trains_smoke_is_looked_up_in_one_go(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = railway_seen_in_pixel_art()
    for stops in ([(10, 6), (10, 16)], [(10, 16), (10, 6)]):
        train = view.railway.buy()
        assert train is not None
        train.stops = [Stop(stop) for stop in stops]
    view.time_passed(0.1)
    layer = TrainLayer(view._pixel_layer().world)
    looked: list[int] = []
    plain = Elevation.lift

    def counted(elevation: Elevation, x: object, y: object) -> object:
        looked.append(np.size(x))
        return plain(elevation, x, y)

    monkeypatch.setattr(Elevation, "lift", counted)
    imgui.new_frame()

    layer.draw(imgui.get_foreground_draw_list(), view.railway, view.camera, (640.0, 400.0))

    imgui.end_frame()
    assert looked == [4, 2]


def test_no_zoom_on_any_screen_makes_a_train_texture_bigger_than_every_graphics_card_takes(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = railway_seen_in_pixel_art()
    train = view.railway.buy()
    assert train is not None
    train.stops = [Stop((10, 6)), Stop((10, 16))]
    view.time_passed(0.1)
    layer = TrainLayer(view._pixel_layer().world)
    sizes: list[int] = []
    monkeypatch.setattr(
        hello_imgui,
        "create_texture_gpu_from_rgba_data",
        lambda pixels: sizes.append(max(pixels.shape[:2])) or FakeTexture(),
    )
    io = imgui.get_io()

    for density in (1, 2):
        io.display_framebuffer_scale = imgui.ImVec2(density, density)
        for zoom in range(len(ZOOM_CELL_WIDTHS)):
            view.camera.zoom_level = zoom
            imgui.new_frame()
            layer.draw(imgui.get_foreground_draw_list(), view.railway, view.camera, (640.0, 400.0))
            imgui.end_frame()

    assert sizes
    assert max(sizes) <= 4096


def test_up_close_only_the_looks_on_screen_are_blown_up(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = railway_seen_in_pixel_art()
    train = view.railway.buy()
    assert train is not None
    train.stops = [Stop((10, 6)), Stop((10, 16))]
    view.time_passed(0.1)
    view.camera.zoom_level = len(ZOOM_CELL_WIDTHS) - 1
    imgui.get_io().display_framebuffer_scale = imgui.ImVec2(2, 2)
    layer = TrainLayer(view._pixel_layer().world)
    monkeypatch.setattr(train_layer, "PAGES_TIME", 10.0)
    made: list[int] = []
    monkeypatch.setattr(
        hello_imgui,
        "create_texture_gpu_from_rgba_data",
        lambda pixels: made.append(pixels.nbytes) or FakeTexture(),
    )
    imgui.new_frame()

    layer.draw(imgui.get_foreground_draw_list(), view.railway, view.camera, (640.0, 400.0))

    imgui.end_frame()
    looks = {look for look, _ in layer._parts(train)}
    assert len(made) == len(looks)
    assert sum(made) < 64 * 1024 * 1024


def test_up_close_a_train_far_off_the_screen_is_not_drawn(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    game = Game(Constructions(flat_map(grid=HexGrid(rows=20, cols=44, km_per_cell=2.0))))
    game.finances.cash = 1_000_000.0
    for cell in ((10, 4), (10, 38)):
        game.build_station(cell)
    game.lay_track((10, 4), (10, 38))
    view = MapView(game)
    near, far = view.railway.buy(), view.railway.buy()
    assert near is not None and far is not None
    near.engine = far.engine = "mixed_diesel"
    near.stops = [Stop((10, 4)), Stop((10, 38))]
    far.stops = [Stop((10, 38)), Stop((10, 4))]
    view.time_passed(0.1)
    view.camera.zoom_level = len(ZOOM_CELL_WIDTHS) - 1
    view.look_at((10, 5))
    layer = TrainLayer(view._pixel_layer().world)
    parted: list[str] = []
    parts = TrainLayer._parts
    monkeypatch.setattr(
        TrainLayer, "_parts", staticmethod(lambda train: parted.append(train.name) or parts(train))
    )
    imgui.new_frame()

    layer.draw(imgui.get_foreground_draw_list(), view.railway, view.camera, (640.0, 400.0))

    imgui.end_frame()
    assert parted == [near.name]


def train_up_close() -> tuple[MapView, Train]:
    view = railway_seen_in_pixel_art()
    train = view.railway.buy()
    assert train is not None
    train.stops = [Stop((10, 6)), Stop((10, 16))]
    view.time_passed(0.1)
    view.camera.zoom_level = len(ZOOM_CELL_WIDTHS) - 1
    imgui.get_io().display_framebuffer_scale = imgui.ImVec2(2, 2)
    return view, train


def test_up_close_the_pictures_are_drawn_as_many_times_as_fine_as_the_sheet(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view, train = train_up_close()
    layer = TrainLayer(view._pixel_layer().world)
    monkeypatch.setattr(train_layer, "PAGES_TIME", 10.0)
    imgui.new_frame()

    layer.draw(imgui.get_foreground_draw_list(), view.railway, view.camera, (640.0, 400.0))

    imgui.end_frame()
    look, _ = layer._parts(train)[0]
    _, _, width, _, _, _ = layer.sprites.places[(look, 0)]
    _, _, fine_width, _, _, _ = layer._pages[look].spots[0]
    times = view.camera.pixel_scale * 2 // DETAIL
    assert (times - 0.5) * width < fine_width < (times + 0.5) * width


def test_up_close_looks_not_drawn_fine_in_a_frames_time_show_from_the_sheet_till_later(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view, train = train_up_close()
    layer = TrainLayer(view._pixel_layer().world)
    monkeypatch.setattr(train_layer, "PAGES_TIME", -1.0)
    imgui.new_frame()
    layer.draw(imgui.get_foreground_draw_list(), view.railway, view.camera, (640.0, 400.0))
    imgui.end_frame()
    pages_in_no_time = len(layer._pages)
    monkeypatch.setattr(train_layer, "PAGES_TIME", 1.0)
    imgui.new_frame()

    layer.draw(imgui.get_foreground_draw_list(), view.railway, view.camera, (640.0, 400.0))

    imgui.end_frame()
    assert pages_in_no_time == 0
    assert layer._sheet is not None
    assert len(layer._pages) == len({look for look, _ in layer._parts(train)})


class Clock:
    def __init__(self, step: float) -> None:
        self.now, self.step = 0.0, step

    def perf_counter(self) -> float:
        self.now += self.step
        return self.now


def test_up_close_a_page_is_drawn_a_few_pictures_a_frame(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view, train = train_up_close()
    layer = TrainLayer(view._pixel_layer().world)
    monkeypatch.setattr(train_layer, "time", Clock(train_layer.PAGES_TIME / 3))
    looks = {look for look, _ in layer._parts(train)}

    def frame() -> None:
        imgui.new_frame()
        layer.draw(imgui.get_foreground_draw_list(), view.railway, view.camera, (640.0, 400.0))
        imgui.end_frame()

    frame()
    after_one = len(layer._pages), sum(len(done) for _, done in layer._painting.values())
    for _ in range(HEADINGS * len(looks)):
        frame()

    assert after_one[0] == 0
    assert 0 < after_one[1] < HEADINGS
    assert set(layer._pages) == looks
    assert not layer._painting
