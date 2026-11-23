from collections.abc import Iterator

import pytest
from imgui_bundle import hello_imgui, imgui
from screens import FakeTexture, gui_context

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions, Station
from railroads.sim.game import Game
from railroads.sim.goods import Good, Need
from railroads.sim.tilemap import TileMap
from railroads.ui import map_overlays
from railroads.ui.map_view import MapView
from railroads.ui.station_preview import draw_station_preview, preview_lines


def make_map() -> TileMap:
    return flat_map(12, 12)


@pytest.fixture
def frame() -> Iterator[None]:
    with gui_context():
        imgui.new_frame()
        imgui.begin("map")
        yield
        imgui.end()
        imgui.render()


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


def test_a_station_preview_names_each_kind_of_building_with_the_points_it_gives(
    frame: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture())
    growth = Game(Constructions(make_map())).growth
    for _ in range(3):
        growth.build(Building((5, 6), BuildingKind.HOUSE))
    preview = growth.station_trade.preview((5, 5))
    points, value = growth.rule.points[BuildingKind.HOUSE]

    lines = preview_lines(preview)
    draw_station_preview(preview, (0, 0))

    assert lines == [f"house x3: {3 * value} {points.value} points"]


def test_a_station_preview_says_point_for_a_single_one() -> None:
    growth = Game(Constructions(make_map())).growth
    growth.build(Building((5, 6), BuildingKind.HOUSE))
    preview = growth.station_trade.preview((5, 5))
    points, value = growth.rule.points[BuildingKind.HOUSE]

    lines = preview_lines(preview)

    assert lines == [f"house: {value} {points.value} point{'' if value == 1 else 's'}"]
