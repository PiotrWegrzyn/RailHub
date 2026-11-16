from collections.abc import Iterator

import pytest
from imgui_bundle import imgui  # noqa: TID251

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.game import Game
from railroads.ui import map_tools
from railroads.ui.build_menu import Tool
from railroads.ui.map_view import MapView


@pytest.fixture
def context() -> Iterator[None]:
    gui = imgui.create_context()
    imgui.get_io().display_size = imgui.ImVec2(1280, 800)
    yield
    imgui.destroy_context(gui)


def laying_track() -> MapView:
    game = Game(Constructions(flat_map(rows=16, cols=30)))
    game.finances.cash = 1_000_000.0
    game.build_station((8, 4))
    view = MapView(game)
    view.build.choose(Tool.TRACK)
    view.tools.click((8, 4))
    return view


def plans_counted(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    planned: list[object] = []
    plain = map_tools.plan_route
    monkeypatch.setattr(map_tools, "PREVIEW_REST_S", 0.0)
    monkeypatch.setattr(map_tools, "plan_route", lambda *args: planned.append(args) or plain(*args))
    monkeypatch.setattr("railroads.sim.routes.plan_route", map_tools.plan_route)
    return planned


def test_clicking_where_the_previewed_track_ends_lays_it_without_planning_again(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = laying_track()
    planned = plans_counted(monkeypatch)
    view.tools.update_preview((8, 20))
    previewed = view.build.preview

    view.tools.click((8, 20))

    assert previewed is not None
    assert len(planned) == 1
    assert all(piece in view.constructions.at(piece.cell) for piece in previewed.pieces)


def test_track_is_planned_again_if_anything_was_built_since_the_preview(
    context: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = laying_track()
    planned = plans_counted(monkeypatch)
    view.tools.update_preview((8, 20))
    view.growth.build(Building((2, 2), BuildingKind.HOUSE))

    view.tools.click((8, 20))

    assert len(planned) == 2
