from collections.abc import Iterator

import pytest
from imgui_bundle import imgui

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions, Station
from railroads.sim.game import Game
from railroads.sim.goods import Good, Need
from railroads.sim.tilemap import TileMap
from railroads.ui import map_overlays
from railroads.ui.map_view import MapView


def make_map() -> TileMap:
    return flat_map(12, 12)


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


def test_a_status_too_long_for_the_map_runs_on_to_more_lines_short_of_the_compass(
    frame: None,
) -> None:
    draw_list = imgui.get_window_draw_list()
    short = map_overlays.draw_status(draw_list, (0, 0), (1000, 700), "Zoom 5/8.")

    below = map_overlays.draw_status(
        draw_list, (0, 0), (1000, 700), f"Zoom 5/8. River. {map_overlays.MOODS_LEGEND}" * 2
    )

    assert below[1] > short[1]
