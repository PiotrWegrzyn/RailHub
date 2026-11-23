from collections.abc import Iterator

import pytest
from imgui_bundle import hello_imgui, imgui
from screens import FakeTexture, gui_context

from railroads.sim.goods import Good, freight_of
from railroads.sim.tiles import Board, Tile, Tiles
from railroads.ui.tiles_window import (
    CARD_GAP,
    CARD_TALL,
    TilesWindow,
    all_unlocks,
    can_confirm,
    card_layout,
    cards,
    era_line,
    freight_line,
    next_unlock,
)


@pytest.fixture
def frame(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with gui_context():
        monkeypatch.setattr(
            hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture()
        )
        imgui.new_frame()
        yield
        imgui.render()


def test_every_tile_on_a_board_is_a_card_with_what_its_next_level_adds() -> None:
    tiles = Tiles.until(1850)

    shown = cards(tiles, Board.SUPPLY)
    wood = next(card for card in shown if card.tile == Tile.WOOD)

    assert [card.tile for card in shown] == [tile for tile in Tile if tile in tiles.supply]
    assert wood.adds == "next level +11% made"


def test_a_demand_card_says_who_its_next_level_lets_take_what() -> None:
    tiles = Tiles()

    wood = next(card for card in cards(tiles, Board.DEMAND) if card.tile == Tile.WOOD)

    assert (wood.opens, wood.opens_next) == ("Mines take planks", True)


def test_a_demand_card_names_a_later_buyer_by_its_level() -> None:
    opens = next_unlock(Tile.ORE, 1), next_unlock(Tile.ORE, 3)

    assert opens == ("level 3: Farms take limestone", "")


def test_a_demand_card_tells_of_a_buyer_only_two_levels_ahead() -> None:
    far, near = next_unlock(Tile.OIL, 5), next_unlock(Tile.OIL, 9)

    assert (far, near) == ("", "level 11: Clay pits, copper mines and 13 more take fuel")


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


def test_a_picked_card_shows_what_its_goods_will_earn_once_the_picks_go_up() -> None:
    tiles = Tiles.until(1850)
    tiles.pick(Board.SUPPLY, Tile.WOOD)

    wood = next(card for card in cards(tiles, Board.SUPPLY) if card.tile == Tile.WOOD)

    assert wood.after == f"picked {tiles.planned_price(Tile.WOOD):.0%}"
    assert tiles.planned_price(Tile.WOOD) < tiles.of_year(1850).price(Good.LOGS)


@pytest.mark.parametrize(("width", "height", "columns"), [(1200.0, 700.0, 6), (700.0, 900.0, 4)])
def test_the_cards_grow_as_large_as_the_screen_lets_every_one_fit(
    width: float, height: float, columns: int
) -> None:
    across, size = card_layout(17, width, height)

    rows = -(-17 // across)
    assert across == columns
    assert across * size + (across - 1) * CARD_GAP <= width + 1e-6
    assert rows * size * CARD_TALL + (rows - 1) * CARD_GAP <= height + 1e-6


def test_a_card_names_its_main_good_with_its_freight_now_and_once_picked() -> None:
    tiles = Tiles.until(1850)
    coal = next(card for card in cards(tiles, Board.SUPPLY) if card.tile == Tile.COAL)

    line = freight_line(coal)

    base = freight_of(Good.COAL)
    assert line == f"Coal ${base * coal.now:,.2f} > ${base * coal.then:,.2f}"


def test_every_good_of_a_tile_earns_the_same_share_more_or_less_once_picked() -> None:
    tiles = Tiles.until(1850)
    farming = next(card for card in cards(tiles, Board.SUPPLY) if card.tile == Tile.FARMING)

    line = freight_line(farming, Good.COFFEE)

    base = freight_of(Good.COFFEE)
    assert line == f"Coffee ${base * farming.now:,.2f} > ${base * farming.then:,.2f}"


def test_a_demand_card_starts_from_what_its_goods_earn_after_the_supply_picks_so_far() -> None:
    tiles = Tiles.until(1850)
    tiles.pick(Board.SUPPLY, Tile.COAL)

    coal = next(card for card in cards(tiles, Board.DEMAND) if card.tile == Tile.COAL)

    assert coal.now == tiles.planned_price(Tile.COAL)
    assert coal.then is not None and coal.then > coal.now
