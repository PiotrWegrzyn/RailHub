from collections.abc import Iterator

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui, implot

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.game import Game
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.railway import Stop
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.ui.app import App
from railroads.ui.build_menu import Tool
from railroads.ui.iso_camera import ZOOM_CELL_WIDTHS
from railroads.ui.map_view import MapView


class FakeTexture:
    def texture_id(self) -> int:
        return 1


def make_map() -> TileMap:
    grid = HexGrid(rows=20, cols=24, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=terrain,
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture())
    gui = imgui.create_context()
    plots = implot.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    yield
    implot.destroy_context(plots)
    imgui.destroy_context(gui)


def made_app() -> App:
    constructions = Constructions(make_map())
    game = Game(constructions)
    game.finances.cash = 1_000_000.0
    for _ in range(4):
        game.growth.build(Building((10, 10), BuildingKind.HOUSE))
    for cell in ((10, 6), (10, 16)):
        game.build_station(cell)
    game.lay_track((10, 6), (10, 16))
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop((10, 6)), Stop((10, 16))]
    app = App(MapView(game))
    app.help.open = app.tech.open = app.reports.open = True
    app.view.selected = (10, 10)
    app.view.station_window.station = (10, 6)
    app.view.build.choose(Tool.TRACK)
    app.view.build.track_from = (10, 6)
    return app


def frame(app: App) -> None:
    imgui.get_io().delta_time = 1 / 60
    imgui.new_frame()
    imgui.set_next_window_pos((0, 0))
    imgui.set_next_window_size((1280, 800))
    imgui.begin("Railroads")
    app.gui()
    imgui.end()
    imgui.render()


@pytest.mark.parametrize("zoom", range(len(ZOOM_CELL_WIDTHS)))
def test_every_window_draws_at_every_zoom(context: None, zoom: int) -> None:
    app = made_app()
    app.view.camera.zoom_level = zoom
    app.view.look_at((10, 10))

    for _ in range(3):
        frame(app)

    assert app.view.camera.zoom_level == zoom
