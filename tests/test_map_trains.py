from collections.abc import Iterator

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui  # noqa: TID251

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.game import Game
from railroads.sim.grids import HexGrid
from railroads.sim.trains import Stop
from railroads.ui import map_view
from railroads.ui.map_view import MapView
from railroads.ui.pixel.relief import Elevation
from railroads.ui.pixel.train_layer import TrainLayer


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
