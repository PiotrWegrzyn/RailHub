from collections.abc import Iterator

import pytest
from imgui_bundle import imgui

from railroads.sim.tiles import Board, Tile, Tiles
from railroads.ui.tiles_window import TilesWindow, era_line, next_unlock, rows


def test_the_window_lists_a_boards_tiles_the_lowest_first_with_what_a_pick_does() -> None:
    tiles = Tiles.until(1850)

    shown = rows(tiles, Board.SUPPLY)

    assert shown[0].level == str(min(tiles.supply.values()))
    assert all(row.can_pick for row in shown[:1])


def test_the_window_says_what_the_next_era_waits_for() -> None:
    tiles = Tiles.until(1850)

    line = era_line(tiles)

    assert line.startswith("Era 2: wagons of generation 2.")
    assert "level 14" in line


@pytest.fixture
def frame() -> Iterator[None]:
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    imgui.new_frame()
    yield
    imgui.render()
    imgui.destroy_context(context)


def test_the_window_opens_by_itself_once_a_year_and_draws_both_boards(frame: None) -> None:
    window = TilesWindow()
    window.new_year(1850)
    window.open = False
    window.new_year(1850)
    window.new_year(1851)

    window.draw(Tiles.until(1851))

    assert window.open


def test_a_demand_tile_says_which_buyer_its_next_level_opens() -> None:
    opens = next_unlock(Tile.ORE, 1), next_unlock(Tile.ORE, 3)

    assert opens == ("level 3: farms take limestone", "")
