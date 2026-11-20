"""The supply and demand screen: both boards (railroads.sim.tiles) as cards filling the screen,
production (supply) then consumption (demand). Every card shows a tile, its level, what its next
level adds, what a unit of its goods earns now and once up, and on demand the buyer its next
level opens; hovering a demand card lists every buyer it opens under it.

When a year is about to turn the game waits on it: the player picks the supply cards and
confirms, the supply board slides away to the left for the demand board, and once they pick there
and confirm the picks go up on 1 January and the game goes on. Opened from the top bar or with
Y, it shows the same two boards to look at, a slide apart."""

import time
from dataclasses import dataclass

from imgui_bundle import hello_imgui, imgui

from railroads.sim.tiles import (
    CLOSES,
    DEMAND_STEPS,
    ERAS,
    MOST_APART,
    TILE_GOODS,
    Board,
    Tile,
    Tiles,
    supply_step,
    unlocks_of,
)
from railroads.ui.theme import BRASS, GROWN, TEXT, TEXT_DIM, WINDOW_TOP, Rgba, color_u32

CARD_SIZE = (176.0, 132.0)
CARD_GAP = 12.0
SLIDE_SECONDS = 0.35
BUTTON = (160.0, 32.0)
SUPPLY_EDGE: Rgba = (0.8, 0.27, 0.38, 1.0)
DEMAND_EDGE: Rgba = (0.36, 0.72, 0.36, 1.0)
CARD_FILL: Rgba = (0.12, 0.093, 0.07, 1.0)
PICKED_FILL: Rgba = (0.24, 0.18, 0.1, 1.0)
SCREEN_FILL: Rgba = (0.06, 0.05, 0.04, 0.97)
EDGES = {Board.SUPPLY: SUPPLY_EDGE, Board.DEMAND: DEMAND_EDGE}
TITLES = {Board.SUPPLY: "Production", Board.DEMAND: "Consumption"}
HINTS = {
    Board.SUPPLY: "More supply makes more of a tile's goods, and each unit earns less.",
    Board.DEMAND: "More demand makes buyers take more of a tile's goods, and each unit earns more.",
}


@dataclass(frozen=True)
class Card:
    tile: Tile
    level: str
    adds: str
    price: str
    after: str
    opens: str
    # Whether its next level is the one that opens it.
    opens_next: bool
    picked: int
    can_pick: bool


def _kinds(names: list[str]) -> str:
    """Kinds of building in words, big and mega ones counted with the plain one."""
    plain = sorted({name.removeprefix("big ").removeprefix("mega ") for name in names})
    named = [f"{name[:-1]}ies" if name.endswith("y") else f"{name}s" for name in plain]
    if len(named) <= 2:
        return " and ".join(named)
    return f"{named[0]}, {named[1]} and {len(named) - 2} more"


def opened_at(tile: Tile, at: int) -> str:
    """Who starts taking what at a demand tile's level, in words; "" for no one."""
    opened = [(kind, need) for lv, kind, need in unlocks_of(tile) if lv == at]
    if not opened:
        return ""
    needs = sorted({need.value for _, need in opened})
    who = _kinds([kind.value for kind, _ in opened])
    return f"{who[0].upper()}{who[1:]} take {', '.join(needs)}"


def next_unlock(tile: Tile, level: int) -> str:
    """The next buyer a demand tile opens above a level: the level and who starts taking what."""
    coming = sorted(at for at, _, _ in unlocks_of(tile) if at > level)
    return f"level {coming[0]}: {opened_at(tile, coming[0])}" if coming else ""


def all_unlocks(tile: Tile) -> list[str]:
    """Every buyer a demand tile opens, a line for each level."""
    levels = sorted({at for at, _, _ in unlocks_of(tile)})
    return [next_unlock(tile, at - 1) for at in levels]


def cards(tiles: Tiles, board: Board) -> list[Card]:
    """Every tile on a board in the order of the tiles: its level with the picks so far, what
    its next level adds, what a unit of its goods earns now and would once picked, what it
    opens next, how many times it is picked and whether it can be."""
    levels = tiles.levels(board)
    now = tiles.of_year(tiles.year)
    shown = []
    for tile in (t for t in Tile if t in levels):
        picked = tiles.picks[board].count(tile)
        level = levels[tile] + picked
        closed = board == Board.SUPPLY and levels[tile] >= CLOSES.get(tile, levels[tile] + 1)
        step = supply_step(tile, level) if board == Board.SUPPLY else DEMAND_STEPS[tile]
        can = tiles.can_pick(board, tile)
        shown.append(
            Card(
                tile,
                "closed" if closed else f"level {levels[tile]}" + (f" +{picked}" if picked else ""),
                ""
                if closed
                else f"next level +{step:.0%} {'made' if board == Board.SUPPLY else 'taken'}",
                f"{now.price(TILE_GOODS[tile][0]):.0%}",
                f"{tiles.price_after(board, tile):.0%}" if can else "",
                (
                    (opened_at(tile, level + 1) or next_unlock(tile, level))
                    if board == Board.DEMAND
                    else ""
                ),
                board == Board.DEMAND and bool(opened_at(tile, level + 1)),
                picked,
                can,
            )
        )
    return shown


def era_line(tiles: Tiles) -> str:
    """The era, its wagons and what the next one waits for."""
    wagons = ERAS[tiles.era - 1].wagons
    line = f"Era {tiles.era}: wagons of generation {wagons}."
    if tiles.era < len(ERAS):
        coming = ERAS[tiles.era]
        low = min(tiles.supply[tile] for tile in tiles.open_tiles(Board.SUPPLY))
        line += (
            f" The next era comes once every supply tile is at level {coming.level} "
            f"(the lowest is at {low})."
        )
    return line


def can_confirm(tiles: Tiles, board: Board) -> bool:
    """Whether a board's picks are done: all made, or no card left that can take one."""
    return tiles.picks_left(board) == 0 or not any(
        tiles.can_pick(board, tile) for tile in tiles.levels(board)
    )


def _ease(share: float) -> float:
    share = max(0.0, min(1.0, share))
    return share * share * (3 - 2 * share)


def _faded(colour: Rgba, alpha: float) -> int:
    return color_u32((colour[0], colour[1], colour[2], alpha))


class TilesWindow:
    def __init__(self) -> None:
        # Shown to look at, opened by the player.
        self.open = False
        # The year whose picks the game waits on, 0 when it waits on none.
        self.choosing = 0
        # The last year whose picks were confirmed.
        self.confirmed = 0
        self.board = Board.SUPPLY
        # When the boards began to slide to the one now shown, 0 for not sliding.
        self._slid_at = 0.0

    def must_choose(self, year: int) -> bool:
        """Whether the game must wait on the picks for a year; begins waiting on them if so."""
        if year <= self.confirmed:
            return False
        if not self.choosing:
            self.choosing = year
            self.board = Board.SUPPLY
            self._slid_at = 0.0
        return True

    def show(self, board: Board) -> None:
        """Slides to a board."""
        if board != self.board:
            self.board = board
            self._slid_at = time.monotonic()

    def confirm(self) -> None:
        """Confirms the picks on the board shown: supply slides on to demand, demand ends the
        wait."""
        if self.board == Board.SUPPLY:
            self.show(Board.DEMAND)
            return
        self.confirmed = self.choosing
        self.choosing = 0
        self.board = Board.SUPPLY
        self._slid_at = 0.0

    def offset(self, width: float) -> float:
        """How far left of where the supply board sits the boards stand now, on their way to the
        one shown."""
        target = 0.0 if self.board == Board.SUPPLY else width
        if not self._slid_at:
            return target
        share = _ease((time.monotonic() - self._slid_at) / SLIDE_SECONDS)
        start = width - target
        return start + (target - start) * share

    def draw(self, tiles: Tiles) -> None:
        if not (self.open or self.choosing):
            return
        io = imgui.get_io()
        imgui.set_next_window_pos((0.0, WINDOW_TOP))
        imgui.set_next_window_size((io.display_size.x, io.display_size.y - WINDOW_TOP))
        imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(*SCREEN_FILL))
        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_collapse.value
            | imgui.WindowFlags_.no_saved_settings.value
        )
        if imgui.begin("##supply_and_demand", None, flags)[0]:
            self._draw_screen(tiles)
        imgui.end()
        imgui.pop_style_color()
        if self._slid_at and time.monotonic() - self._slid_at < SLIDE_SECONDS:
            hello_imgui.request_refresh()

    def _draw_screen(self, tiles: Tiles) -> None:
        width = imgui.get_content_region_avail().x
        imgui.text_colored(imgui.ImVec4(*BRASS), "SUPPLY AND DEMAND")
        imgui.same_line(0, 16)
        imgui.text_colored(imgui.ImVec4(*TEXT_DIM), era_line(tiles))
        if self.choosing:
            imgui.text_colored(
                imgui.ImVec4(*TEXT_DIM),
                f"The picks go up on 1 January {self.choosing}. No tile can be more than "
                f"{MOST_APART} levels above the lowest on its board.",
            )
        imgui.spacing()
        body = imgui.get_cursor_pos()
        apart = width + imgui.get_style().window_padding.x * 2
        offset = self.offset(apart)
        for index, board in enumerate(Board):
            x = body.x + index * apart - offset
            if x >= apart or x + width <= 0:
                continue
            imgui.set_cursor_pos((x, body.y))
            imgui.begin_group()
            self._draw_board(tiles, board, width)
            imgui.end_group()
        self._draw_buttons(tiles)

    def _draw_board(self, tiles: Tiles, board: Board, width: float) -> None:
        edge = EDGES[board]
        title = TITLES[board]
        if self.choosing:
            left = tiles.picks_left(board)
            title += f": pick {left} tile{'s' if left != 1 else ''}" if left else ": confirm"
        imgui.text_colored(imgui.ImVec4(*edge), title.upper())
        imgui.text_colored(imgui.ImVec4(*TEXT_DIM), HINTS[board])
        imgui.spacing()
        columns = max(1, int((width + CARD_GAP) // (CARD_SIZE[0] + CARD_GAP)))
        picking = bool(self.choosing) and self.board == board
        for index, card in enumerate(cards(tiles, board)):
            if index % columns:
                imgui.same_line(0, CARD_GAP)
            self._draw_card(tiles, board, card, edge, picking)

    def _draw_card(self, tiles: Tiles, board: Board, card: Card, edge: Rgba, picking: bool) -> None:
        low = imgui.get_cursor_screen_pos()
        high = imgui.ImVec2(low.x + CARD_SIZE[0], low.y + CARD_SIZE[1])
        clicked = imgui.invisible_button(f"##{board.name}{card.tile.name}", CARD_SIZE)
        hovered = imgui.is_item_hovered()
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(
            low, high, color_u32(PICKED_FILL if card.picked else CARD_FILL), 6.0
        )
        alpha = 1.0 if card.picked or not picking or card.can_pick else 0.35
        thickness = 3.0 if card.picked else 2.0 if hovered and picking and card.can_pick else 1.2
        draw_list.add_rect(low, high, _faded(edge, alpha), 6.0, thickness)
        x, y = low.x + 10, low.y + 8
        line = imgui.get_text_line_height_with_spacing()
        draw_list.push_clip_rect(low, high, True)
        draw_list.add_text((x, y), _faded(edge, alpha), card.tile.value.capitalize())
        draw_list.add_text((x, y + line), _faded(BRASS, alpha), card.level)
        draw_list.add_text((x, y + 2 * line), _faded(TEXT, alpha), card.adds)
        draw_list.add_text((x, y + 3.3 * line), _faded(TEXT_DIM, alpha), f"earns {card.price}")
        if card.after:
            draw_list.add_text(
                (x, y + 4.3 * line), _faded(TEXT_DIM, alpha), f"once up {card.after}"
            )
        if card.opens:
            colour = GROWN if card.opens_next else TEXT_DIM
            draw_list.add_text((x, high.y - line - 4), _faded(colour, alpha), card.opens)
        draw_list.pop_clip_rect()
        if hovered:
            self._tip(card, board, low.x, high.y)
        if clicked and picking:
            if card.picked:
                tiles.unpick(board, card.tile)
            elif card.can_pick:
                tiles.pick(board, card.tile)

    @staticmethod
    def _tip(card: Card, board: Board, left: float, bottom: float) -> None:
        """What a card stands for, under it: its goods and, on demand, every buyer it opens."""
        imgui.set_next_window_pos((left, bottom + 2))
        imgui.begin_tooltip()
        goods = ", ".join(good.value for good in TILE_GOODS[card.tile])
        imgui.text_colored(imgui.ImVec4(*TEXT_DIM), goods)
        if board == Board.DEMAND:
            for line in all_unlocks(card.tile):
                imgui.text_colored(imgui.ImVec4(*GROWN), line)
        imgui.end_tooltip()

    def _draw_buttons(self, tiles: Tiles) -> None:
        width = imgui.get_content_region_avail().x
        imgui.set_cursor_pos(
            (imgui.get_style().window_padding.x, imgui.get_window_height() - BUTTON[1] - 16.0)
        )
        if self.choosing:
            ready = can_confirm(tiles, self.board)
            imgui.begin_disabled(not ready)
            if imgui.button("Confirm", BUTTON) or (ready and imgui.is_key_pressed(imgui.Key.enter)):
                self.confirm()
            imgui.end_disabled()
            return
        other = Board.DEMAND if self.board == Board.SUPPLY else Board.SUPPLY
        key = imgui.Key.right_arrow if other == Board.DEMAND else imgui.Key.left_arrow
        if imgui.button(TITLES[other], BUTTON) or imgui.is_key_pressed(key):
            self.show(other)
        imgui.same_line(0, max(8.0, width - 2 * BUTTON[0]))
        if imgui.button("Close", BUTTON) or imgui.is_key_pressed(imgui.Key.escape):
            self.open = False
