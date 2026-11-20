from collections.abc import Iterator

import pytest
from imgui_bundle import imgui

from railroads.sim.tiles import Board, Tile, Tiles
from railroads.ui.tiles_window import (
    TilesWindow,
    all_unlocks,
    can_confirm,
    cards,
    era_line,
    next_unlock,
)


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


def test_every_tile_on_a_board_is_a_card_with_what_its_next_level_adds() -> None:
    tiles = Tiles.until(1850)

    shown = cards(tiles, Board.SUPPLY)

    assert [card.tile for card in shown] == [tile for tile in Tile if tile in tiles.supply]
    wood = next(card for card in shown if card.tile == Tile.WOOD)
    assert wood.adds == "next level +30% made"


def test_a_demand_card_says_who_its_next_level_lets_take_what() -> None:
    tiles = Tiles()

    wood = next(card for card in cards(tiles, Board.DEMAND) if card.tile == Tile.WOOD)

    assert (wood.opens, wood.opens_next) == ("Mines take planks", True)


def test_a_demand_card_names_a_later_buyer_by_its_level() -> None:
    opens = next_unlock(Tile.ORE, 1), next_unlock(Tile.ORE, 3)

    assert opens == ("level 3: Farms take limestone", "")


def test_every_buyer_a_tile_opens_is_listed_by_level() -> None:
    lines = all_unlocks(Tile.WOOD)

    assert lines == ["level 2: Mines take planks", "level 4: Breweries take barrels"]


def test_the_screen_says_what_the_next_era_waits_for() -> None:
    tiles = Tiles.until(1850)

    line = era_line(tiles)

    assert line.startswith("Era 2: wagons of generation 2.")
    assert "level 14" in line


def test_a_board_can_be_confirmed_only_once_its_picks_are_made() -> None:
    tiles = Tiles.until(1850)
    before = can_confirm(tiles, Board.SUPPLY)
    tiles.pick(Board.SUPPLY, Tile.WOOD)
    tiles.pick(Board.SUPPLY, Tile.FARMING)

    after = can_confirm(tiles, Board.SUPPLY)

    assert (before, after) == (False, True)


def test_the_game_waits_on_a_years_picks_until_both_boards_are_confirmed() -> None:
    window = TilesWindow()
    waits = window.must_choose(1851)

    window.confirm()
    on_demand = window.board, window.choosing
    window.confirm()

    assert waits
    assert on_demand == (Board.DEMAND, 1851)
    assert window.choosing == 0
    assert not window.must_choose(1851)
    assert window.must_choose(1852)


@pytest.mark.parametrize("choosing", [True, False])
def test_the_screen_draws_both_boards_to_pick_from_or_to_look_at(
    frame: None, choosing: bool
) -> None:
    window = TilesWindow()
    if choosing:
        window.must_choose(1851)
    else:
        window.open = True

    window.draw(Tiles.until(1850))

    assert window.open or window.choosing
