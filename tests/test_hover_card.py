from collections.abc import Iterator

import numpy as np
import pytest
from imgui_bundle import imgui

from railroads.sim.constructions import Building, BuildingKind, Constructions, Station
from railroads.sim.game import Game
from railroads.sim.geo import Bounds
from railroads.sim.goods import Good, Need
from railroads.sim.grids import HexGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.ui import map_overlays
from railroads.ui.map_view import MapView


def make_map() -> TileMap:
    grid = HexGrid(rows=12, cols=12, km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )


@pytest.fixture
def frame() -> Iterator[None]:
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    imgui.new_frame()
    imgui.begin("map")
    yield
    imgui.end()
    imgui.render()
    imgui.destroy_context(context)


def test_the_card_lists_each_building_with_what_it_makes_and_needs(
    frame: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructions = Constructions(make_map())
    constructions.build(Building((5, 5), BuildingKind.HOUSE))
    constructions.build(Building((5, 5), BuildingKind.BAKERY))
    growth = Game(constructions).growth
    drawn: list[Good | Need] = []
    monkeypatch.setattr(map_overlays, "good_icon", lambda good, **_: drawn.append(good))

    map_overlays.draw_hovered_buildings(growth, (5, 5), (10.0, 40.0))

    assert Good.BREAD in drawn
    assert Need.WHEAT in drawn
    assert Need.FOOD in drawn


def test_a_left_click_on_a_station_opens_it() -> None:
    constructions = Constructions(make_map())
    constructions.build(Station((5, 5)))
    view = MapView(Game(constructions))

    view.tools.click((5, 5))

    assert view.station_window.station == (5, 5)
