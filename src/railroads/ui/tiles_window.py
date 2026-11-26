"""The supply and demand screen: both boards (railroads.sim.tiles) as cards filling the screen,
production (supply) then consumption (demand), drawn as the start menu is: dark panels framed in
glowing brass with corner brackets and titles in spaced capitals between fading rules. Every card
shows a tile, its level, what its next level adds, the icons of its goods, what a unit of them
earns now and once picked, and on demand the buyer it opens next; hovering a demand card lists
every buyer it opens under it. Above every card a row of marks stands for its levels, reached and
ahead, from where the lowest on its board stood when the era came: a mark for a level that brings
more than its share (railroads.sim.level_perks) shows what, and hovering one tells of it. The
cards grow as large as the screen lets them.

When a year is about to turn the game waits on it: the player picks the supply cards and
confirms, the supply board slides away to the left for the demand board, and once they pick there
and confirm the picks go up on 1 January and the game goes on. Opened from the top bar or with
Y, it shows the same two boards to look at, a slide apart."""

import math
import time
from dataclasses import dataclass

from imgui_bundle import hello_imgui, imgui

from railroads.sim.goods import Good, Need, freight_of
from railroads.sim.level_perks import Perk, opened_at, perks
from railroads.sim.tiles import (
    ERAS,
    MOST_APART,
    TILE_GOODS,
    Board,
    Tile,
    Tiles,
    demand_step,
    supply_step,
    unlocks_of,
)
from railroads.ui.ornate import (
    DEEP,
    EDGE,
    GLOW,
    bar_buttons,
    frame_rect,
    rule,
    spaced,
    spaced_text,
)
from railroads.ui.theme import (
    BRASS,
    GROWN,
    TEXT_DIM,
    WINDOW_TOP,
    Rgba,
    color_u32,
    colored,
    draw_good_icon,
)
from railroads.words import dollars

# A card's width at its usual size, its height for every unit of width, and the most it grows.
CARD_WIDTH = 190.0
# How many levels ahead a demand card tells of the next buyer it opens.
UNLOCK_AHEAD = 2
CARD_TALL = 1.12
CARD_MOST = 230.0
CARD_GAP = 14.0
SLIDE_SECONDS = 0.35
BUTTON = (180.0, 34.0)
BOARD_TITLE_SCALE = 1.3
# Room under the boards for the buttons.
FOOT = 64.0
SUPPLY_EDGE: Rgba = (0.86, 0.36, 0.42, 1.0)
DEMAND_EDGE: Rgba = (0.46, 0.8, 0.42, 1.0)
SHADE: Rgba = (0.0, 0.0, 0.0, 0.62)
EDGES = {Board.SUPPLY: SUPPLY_EDGE, Board.DEMAND: DEMAND_EDGE}
TITLES = {Board.SUPPLY: "Production", Board.DEMAND: "Consumption"}
WORDS = {Board.SUPPLY: "supply", Board.DEMAND: "demand"}
HINTS = {
    Board.SUPPLY: "More supply makes more of a tile's goods, and each unit earns less.",
    Board.DEMAND: "More demand makes buyers take more of a tile's goods, and each unit earns more.",
}
ICONS_SHOWN = 6
# How many levels above the highest tile on a board the marks over its cards reach: twice as many
# on supply, which goes up two tiles a year, as on demand, with the most marks a row holds.
MARKS_AHEAD = {Board.SUPPLY: 6, Board.DEMAND: 3}
MARKS_MOST = {Board.SUPPLY: 14, Board.DEMAND: 10}
# The room above a card for its marks, for every unit of its width: as tall as a mark is wide
# when the fewest there can be span the card.
MARK_TALL = 1 / 10
MARK_REACHED: Rgba = (0.98, 0.84, 0.42, 1.0)
MARK_AHEAD: Rgba = (0.34, 0.33, 0.32, 1.0)
MARK_INK: Rgba = (0.16, 0.11, 0.06, 1.0)


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
    # Its level with the picks so far and what its next level adds.
    number: int = 0
    step: float = 0.0
    # What a unit of its goods earns before it is picked (now if it is, else after the picks
    # so far on both boards), and once picked or after the picks so far.
    now: float = 1.0
    then: float | None = None
    # The need of the buyer it opens next.
    need: Need | None = None


def next_unlock(tile: Tile, level: int) -> str:
    """The next buyer a demand tile opens above a level: the level and who starts taking what;
    "" for none within UNLOCK_AHEAD levels."""
    coming = sorted(at for at, _, _ in unlocks_of(tile) if level < at <= level + UNLOCK_AHEAD)
    return f"level {coming[0]}: {opened_at(tile, coming[0])}" if coming else ""


def _next_need(tile: Tile, level: int) -> Need | None:
    coming = [(at, need) for at, _, need in unlocks_of(tile) if level < at <= level + UNLOCK_AHEAD]
    return min(coming, key=lambda each: each[0])[1] if coming else None


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
        step = supply_step(tile, level) if board == Board.SUPPLY else demand_step(tile, level)
        can = tiles.can_pick(board, tile)
        price = now.price(TILE_GOODS[tile][0])
        then = (
            tiles.planned_price(tile) if picked else tiles.price_after(board, tile) if can else None
        )
        start = price if picked else tiles.planned_price(tile)
        demand = board == Board.DEMAND
        shown.append(
            Card(
                tile,
                f"level {levels[tile]}" + (f" +{picked}" if picked else ""),
                f"next level +{step:.0%} {'made' if board == Board.SUPPLY else 'taken'}",
                f"{price:.0%}",
                f"picked {then:.0%}"
                if picked and then is not None
                else f"once picked {then:.0%}"
                if then is not None
                else "",
                (opened_at(tile, level + 1) or next_unlock(tile, level)) if demand else "",
                demand and bool(opened_at(tile, level + 1)),
                picked,
                can,
                level,
                step,
                start,
                then,
                _next_need(tile, level) if demand else None,
            )
        )
    return shown


def freight_line(card: Card, good: Good | None = None) -> str:
    """What carrying a unit of a good of a card's tile earns now, and once picked if it can be:
    of its main good unless another is given. Every good of a tile earns the same share more or
    less."""
    good = TILE_GOODS[card.tile][0] if good is None else good
    base = freight_of(good)
    line = f"{good.value.capitalize()} {dollars(base * card.now, cents=True)}"
    if card.then is not None:
        line += f" > {dollars(base * card.then, cents=True)}"
    return line


def era_start(tiles: Tiles, board: Board) -> int:
    """The lowest level on a board when the era in play came: 1 in the first era."""
    year = tiles.era_years.get(tiles.era)
    came = tiles.years.get(year) if year is not None else None
    if came is None:
        return 1 if tiles.era == 1 else min(tiles.levels(board).values())
    return min((came.supply if board == Board.SUPPLY else came.demand).values())


def mark_levels(tiles: Tiles, board: Board) -> range:
    """The levels the marks above a board's cards stand for: from where the lowest stood when the
    era came, so they move left by as much as each era lasted, to MARKS_AHEAD above the highest
    with the picks so far, no more than MARKS_MOST."""
    end = max(tiles.planned(board).values()) + MARKS_AHEAD[board]
    return range(max(era_start(tiles, board), end - MARKS_MOST[board] + 1), end + 1)


def mark_tip(board: Board, tile: Tile, level: int) -> list[str]:
    """What a mark tells of its level: what it adds and what else it brings."""
    step = supply_step(tile, level - 1) if board == Board.SUPPLY else demand_step(tile, level - 1)
    return [f"Level {level}: +{step:.0%} {WORDS[board]}"] + [
        perk.text for perk in perks(board, tile, level)
    ]


def era_line(tiles: Tiles) -> str:
    """The era, its wagons and what the next one waits for."""
    wagons = ERAS[tiles.era - 1].wagons
    line = f"Era {tiles.era}: wagons of generation {wagons}."
    if tiles.era < len(ERAS):
        coming = ERAS[tiles.era]
        low = min(tiles.supply.values())
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


def card_layout(count: int, width: float, height: float) -> tuple[int, float]:
    """How many cards a row holds and how wide each is, for so many cards with their marks to
    fill an area as large as they can without passing it."""
    best = (1, 0.0)
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        across = (width - (columns - 1) * CARD_GAP) / columns
        down = (height - (rows - 1) * CARD_GAP) / rows / (CARD_TALL + MARK_TALL)
        size = min(across, down, CARD_MOST)
        if size > best[1]:
            best = (columns, size)
    return best


def _ease(share: float) -> float:
    share = max(0.0, min(1.0, share))
    return share * share * (3 - 2 * share)


def _faded(colour: Rgba, alpha: float) -> int:
    return color_u32((colour[0], colour[1], colour[2], alpha))


def _text_width(text: str, size: float) -> float:
    return imgui.calc_text_size(text).x * size / imgui.get_font_size()


def _centred(
    draw_list: imgui.ImDrawList, middle: float, y: float, text: str, colour: int, size: float
) -> None:
    left = middle - _text_width(text, size) / 2
    draw_list.add_text(imgui.get_font(), size, imgui.ImVec2(left, y), colour, text)


def _titled(
    draw_list: imgui.ImDrawList, left: float, right: float, y: float, text: str, scale: float
) -> None:
    """Spaced capitals centred between two points with rules fading out either side, as the
    start menu's title is."""
    wide = spaced(text, scale)
    while wide > right - left - 24 and scale > 0.55:
        scale -= 0.05
        wide = spaced(text, scale)
    start = (left + right - wide) / 2
    spaced_text(draw_list, (start, y), text, BRASS, scale)
    middle = y + imgui.get_font_size() * scale / 2 + 1
    rule(draw_list, left + 6, start - 6, middle, fade_left=True)
    rule(draw_list, start + wide + 6, right - 6, middle, fade_left=False)


def _quiet_frame(draw_list: imgui.ImDrawList, low: imgui.ImVec2, high: imgui.ImVec2) -> None:
    """A card's plain edge: thin brass with short brackets at the corners."""
    draw_list.add_rect(low, high, color_u32(EDGE), 6.0, 1.0)
    dim = _faded(GLOW, 0.5)
    inset, bracket = 4.0, 8.0
    for x, y, dx, dy in (
        (low.x + inset, low.y + inset, 1, 1),
        (high.x - inset, low.y + inset, -1, 1),
        (low.x + inset, high.y - inset, 1, -1),
        (high.x - inset, high.y - inset, -1, -1),
    ):
        draw_list.add_line((x, y), (x + dx * bracket, y), dim, 1.5)
        draw_list.add_line((x, y), (x, y + dy * bracket), dim, 1.5)


def _star(middle: imgui.ImVec2, radius: float) -> list[imgui.ImVec2]:
    return [
        imgui.ImVec2(
            middle.x + math.sin(i * math.pi / 5) * radius * (1.0 if i % 2 == 0 else 0.45),
            middle.y - math.cos(i * math.pi / 5) * radius * (1.0 if i % 2 == 0 else 0.45),
        )
        for i in range(10)
    ]


def _draw_sign(
    draw_list: imgui.ImDrawList, perk: Perk, middle: imgui.ImVec2, radius: float, ink: int
) -> None:
    """A level's sign inside its mark: a star for new buyers, an arched bridge for a new kind
    of bridge, an engine for a new one for sale, a wagon for an era."""
    x, y, r = middle.x, middle.y, radius
    if perk == Perk.BUYERS:
        draw_list.add_concave_poly_filled(_star(middle, r * 1.15), ink)
        return
    if perk == Perk.BRIDGE:
        thick = max(1.0, r * 0.3)
        draw_list.add_rect_filled((x - r, y - r * 0.55), (x + r, y - r * 0.55 + thick), ink)
        for px in (x - r * 0.85, x + r * 0.85 - thick):
            draw_list.add_rect_filled((px, y - r * 0.55), (px + thick, y + r * 0.8), ink)
        draw_list.path_arc_to((x, y + r * 0.8), r * 0.55, math.pi, 2 * math.pi, 12)
        draw_list.path_stroke(ink, 0, thick)
        return
    wheel = max(1.0, r * 0.28)
    if perk == Perk.ENGINE:
        draw_list.add_rect_filled((x - r, y - r * 0.35), (x + r * 0.35, y + r * 0.45), ink)
        draw_list.add_rect_filled((x + r * 0.1, y - r * 0.85), (x + r, y + r * 0.45), ink)
        draw_list.add_rect_filled((x - r * 0.8, y - r * 0.85), (x - r * 0.45, y - r * 0.35), ink)
    else:
        draw_list.add_rect_filled((x - r, y - r * 0.7), (x + r, y + r * 0.45), ink)
    for wx in (x - r * 0.55, x + r * 0.55):
        draw_list.add_circle_filled((wx, y + r * 0.6), wheel, ink)


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
        # Whether the screen was drawn on the frame before, to bring it to the front as it opens.
        self._drawn = False

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
            self._drawn = False
            return
        if not self._drawn:
            imgui.set_next_window_focus()
            self._drawn = True
        io = imgui.get_io()
        imgui.set_next_window_pos((0.0, WINDOW_TOP))
        imgui.set_next_window_size((io.display_size.x, io.display_size.y - WINDOW_TOP))
        imgui.push_style_color(imgui.Col_.window_bg, imgui.ImVec4(*DEEP))
        imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(0, 0, 0, 0))
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(24, 18))
        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_collapse.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_scroll_with_mouse.value
        )
        if imgui.begin("##supply_and_demand", None, flags)[0]:
            low, size = imgui.get_window_pos(), imgui.get_window_size()
            frame_rect(
                imgui.get_window_draw_list(),
                imgui.ImVec2(low.x + 6, low.y + 6),
                imgui.ImVec2(low.x + size.x - 6, low.y + size.y - 6),
                4.0,
            )
            self._draw_screen(tiles)
        imgui.end()
        imgui.pop_style_var()
        imgui.pop_style_color(2)
        if self._slid_at and time.monotonic() - self._slid_at < SLIDE_SECONDS:
            hello_imgui.request_refresh()

    def _draw_screen(self, tiles: Tiles) -> None:
        width = imgui.get_content_region_avail().x
        draw_list = imgui.get_window_draw_list()
        at = imgui.get_cursor_screen_pos()
        _titled(draw_list, at.x, at.x + width, at.y, "SUPPLY AND DEMAND", BOARD_TITLE_SCALE)
        imgui.dummy((width, imgui.get_font_size() * BOARD_TITLE_SCALE + 6))
        notes = [era_line(tiles)]
        if self.choosing:
            notes.append(
                f"The picks go up on 1 January {self.choosing}. No tile can be more than "
                f"{MOST_APART} levels above the lowest on its board."
            )
        for note in notes:
            line = imgui.get_cursor_screen_pos()
            _centred(
                draw_list,
                line.x + width / 2,
                line.y,
                note,
                color_u32(TEXT_DIM),
                imgui.get_font_size(),
            )
            imgui.dummy((width, imgui.get_text_line_height_with_spacing()))
        imgui.spacing()
        body = imgui.get_cursor_pos()
        height = imgui.get_window_height() - body.y - FOOT
        apart = width + imgui.get_style().window_padding.x * 2
        offset = self.offset(apart)
        draw_list.push_clip_rect(
            imgui.ImVec2(imgui.get_window_pos().x + 8, imgui.get_window_pos().y),
            imgui.ImVec2(
                imgui.get_window_pos().x + imgui.get_window_size().x - 8,
                imgui.get_window_pos().y + imgui.get_window_size().y,
            ),
            True,
        )
        for index, board in enumerate(Board):
            x = body.x + index * apart - offset
            if x >= apart or x + width <= 0:
                continue
            imgui.set_cursor_pos((x, body.y))
            imgui.begin_group()
            self._draw_board(tiles, board, width, height)
            imgui.end_group()
        draw_list.pop_clip_rect()
        self._draw_buttons(tiles)

    def _draw_board(self, tiles: Tiles, board: Board, width: float, height: float) -> None:
        draw_list = imgui.get_window_draw_list()
        edge = EDGES[board]
        title = TITLES[board].upper()
        if self.choosing:
            left = tiles.picks_left(board)
            title += f"  ·  PICK {left}" if left else "  ·  CONFIRM"
        at = imgui.get_cursor_screen_pos()
        wide = spaced(title, 1.0)
        start = at.x + (width - wide) / 2
        spaced_text(draw_list, (start, at.y), title, edge, 1.0)
        imgui.dummy((width, imgui.get_font_size() + 4))
        hint = imgui.get_cursor_screen_pos()
        _centred(
            draw_list,
            hint.x + width / 2,
            hint.y,
            HINTS[board],
            color_u32(TEXT_DIM),
            imgui.get_font_size(),
        )
        imgui.dummy((width, imgui.get_text_line_height_with_spacing() + 6))
        shown = cards(tiles, board)
        room = height - 2 * imgui.get_text_line_height_with_spacing() - 16
        columns, size = card_layout(len(shown), width, room)
        row = columns * size + (columns - 1) * CARD_GAP
        indent = (width - row) / 2
        picking = bool(self.choosing) and self.board == board
        marks = mark_levels(tiles, board)
        for index, card in enumerate(shown):
            if index % columns:
                imgui.same_line(0, CARD_GAP)
            else:
                imgui.set_cursor_pos((imgui.get_cursor_pos().x + indent, imgui.get_cursor_pos().y))
            imgui.begin_group()
            self._draw_marks(tiles, board, card, marks, size)
            self._draw_card(tiles, board, card, edge, picking, size)
            imgui.end_group()
            if index % columns == columns - 1:
                imgui.dummy((0, CARD_GAP - imgui.get_style().item_spacing.y))

    def _draw_card(
        self, tiles: Tiles, board: Board, card: Card, edge: Rgba, picking: bool, size: float
    ) -> None:
        scale = size / CARD_WIDTH
        low = imgui.get_cursor_screen_pos()
        high = imgui.ImVec2(low.x + size, low.y + size * CARD_TALL)
        clicked = imgui.invisible_button(
            f"##{board.name}{card.tile.name}", (size, size * CARD_TALL)
        )
        hovered = imgui.is_item_hovered()
        pickable = picking and (card.can_pick or card.picked)
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(low, high, color_u32(DEEP), 6.0)
        draw_list.add_rect_filled_multi_color(
            imgui.ImVec2(low.x + 1, low.y + 1),
            imgui.ImVec2(high.x - 1, low.y + 40 * scale),
            _faded(edge, 0.22),
            _faded(edge, 0.22),
            _faded(edge, 0.0),
            _faded(edge, 0.0),
        )
        if card.picked or (hovered and pickable):
            frame_rect(draw_list, low, high, 6.0, 10.0 * scale)
        else:
            _quiet_frame(draw_list, low, high)
        font = imgui.get_font_size()
        small = font * max(0.8, min(1.15, scale))
        middle = (low.x + high.x) / 2
        y = low.y + 10 * scale
        _titled(draw_list, low.x + 4, high.x - 4, y, card.tile.value.upper(), 0.8 * max(0.8, scale))
        y += font * 1.3
        big = font * 1.6 * scale
        body = small * 1.6 + big * 1.25 + 26 * scale + small * 1.2
        band = small * 3.4 + 8 if card.opens else 0.0
        floor = high.y - 6 - band
        y += max(0.0, (floor - y - body) / 2)
        level = f"Level {card.number}"
        _centred(
            draw_list, middle, y, level, color_u32(GROWN if card.picked else BRASS), small * 1.1
        )
        y += small * 1.6
        _centred(draw_list, middle, y, f"+{card.step:.0%} {WORDS[board]}", color_u32(edge), big)
        y += big * 1.25
        self._draw_icons(draw_list, card, middle, y, scale)
        y += 18 * scale + 8 * scale
        self._draw_price(draw_list, card, middle, y, small, size - 16)
        if card.opens:
            self._draw_unlock(draw_list, card, low, high, max(y + small * 1.5, floor), small)
        if picking and not pickable:
            draw_list.add_rect_filled(low, high, color_u32(SHADE), 6.0)
        if hovered:
            self._tip(card, board, low, high)
        if clicked and picking:
            if card.picked:
                tiles.unpick(board, card.tile)
            elif card.can_pick:
                tiles.pick(board, card.tile)

    @staticmethod
    def _draw_marks(tiles: Tiles, board: Board, card: Card, marks: range, size: float) -> None:
        """A card's levels in a row of touching rounds above it: filled for the ones reached,
        green for the ones picked, grey for the ones ahead, and a sign on the ones that bring
        more than their share. Hovering one tells of its level."""
        at = imgui.get_cursor_screen_pos()
        room = size * MARK_TALL
        imgui.dummy((size, room - imgui.get_style().item_spacing.y))
        draw_list = imgui.get_window_draw_list()
        across = min(room, size / max(len(marks), 1))
        left = at.x + (size - across * len(marks)) / 2
        middle_y = at.y + room - across / 2
        reached = tiles.levels(board)[card.tile]
        for index, level in enumerate(marks):
            middle = imgui.ImVec2(left + (index + 0.5) * across, middle_y)
            radius = across / 2 - 0.5
            fill = (
                MARK_REACHED if level <= reached else GROWN if level <= card.number else MARK_AHEAD
            )
            ahead = level > card.number
            draw_list.add_circle_filled(middle, radius, color_u32(fill))
            draw_list.add_circle(
                middle, radius, color_u32(BRASS if not ahead else TEXT_DIM), 0, 1.0
            )
            found = perks(board, card.tile, level)
            if found:
                ink = color_u32(BRASS if ahead else MARK_INK)
                _draw_sign(draw_list, found[0].kind, middle, radius * 0.72, ink)
            low = imgui.ImVec2(middle.x - across / 2, middle.y - across / 2)
            high = imgui.ImVec2(middle.x + across / 2, middle.y + across / 2)
            if imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(low, high):
                imgui.begin_tooltip()
                lines = mark_tip(board, card.tile, level)
                colored(BRASS, lines[0])
                for line in lines[1:]:
                    imgui.push_text_wrap_pos(imgui.get_font_size() * 24)
                    colored(GROWN, line)
                    imgui.pop_text_wrap_pos()
                imgui.end_tooltip()

    @staticmethod
    def _draw_icons(
        draw_list: imgui.ImDrawList, card: Card, middle: float, top: float, scale: float
    ) -> None:
        """The icons of the first goods of a card's tile, side by side."""
        goods = TILE_GOODS[card.tile][:ICONS_SHOWN]
        icon = 18.0 * scale
        gap = 4.0 * scale
        left = middle - (len(goods) * icon + (len(goods) - 1) * gap) / 2
        for index, good in enumerate(goods):
            draw_good_icon(draw_list, good, (left + index * (icon + gap), top), icon)

    @staticmethod
    def _draw_price(
        draw_list: imgui.ImDrawList,
        card: Card,
        middle: float,
        top: float,
        size: float,
        room: float,
    ) -> None:
        """What carrying a unit of the tile's main good earns now and once picked, in grey, as
        small as it must be to fit the card."""
        line = freight_line(card)
        size = min(size, size * room / max(_text_width(line, size), 1.0))
        _centred(draw_list, middle, top, line, color_u32(TEXT_DIM), size)

    @staticmethod
    def _draw_unlock(
        draw_list: imgui.ImDrawList,
        card: Card,
        low: imgui.ImVec2,
        high: imgui.ImVec2,
        top: float,
        size: float,
    ) -> None:
        """The buyer a demand card opens next, on a band at its foot: green for its next level."""
        band_low = imgui.ImVec2(low.x + 6, top)
        band_high = imgui.ImVec2(high.x - 6, high.y - 6)
        if band_high.y - band_low.y < size:
            return
        tint = GROWN if card.opens_next else TEXT_DIM
        draw_list.add_rect_filled(band_low, band_high, _faded(tint, 0.12), 4.0)
        draw_list.add_rect(band_low, band_high, _faded(tint, 0.45), 4.0, 1.0)
        x = band_low.x + 6
        if card.need is not None:
            icon = min(size * 1.2, band_high.y - band_low.y - 8)
            draw_good_icon(draw_list, card.need, (x, band_low.y + 4), icon)
            x += icon + 6
        draw_list.push_clip_rect(band_low, band_high, True)
        draw_list.add_text(
            imgui.get_font(),
            size,
            imgui.ImVec2(x, band_low.y + 4),
            color_u32(tint),
            card.opens,
            band_high.x - x - 4,
        )
        draw_list.pop_clip_rect()

    @staticmethod
    def _tip(card: Card, board: Board, low: imgui.ImVec2, high: imgui.ImVec2) -> None:
        """What a card stands for, beside it on the side with more room, so it stays on the
        screen and leaves the cards in the next row to be seen: what carrying each of its goods
        earns now and once picked, and on demand every buyer it opens."""
        screen = imgui.get_io().display_size
        right = (low.x + high.x) / 2 < screen.x / 2
        lower = (low.y + high.y) / 2 > screen.y / 2
        imgui.set_next_window_pos(
            (high.x + 4 if right else low.x - 4, high.y if lower else low.y),
            pivot=(0.0 if right else 1.0, 1.0 if lower else 0.0),
        )
        imgui.begin_tooltip()
        colored(BRASS, "Freight a unit")
        for good in TILE_GOODS[card.tile]:
            colored(TEXT_DIM, freight_line(card, good))
        if board == Board.DEMAND:
            for line in all_unlocks(card.tile):
                colored(GROWN, line)
        imgui.end_tooltip()

    def _draw_buttons(self, tiles: Tiles) -> None:
        width = imgui.get_content_region_avail().x
        pad = imgui.get_style().window_padding.x
        imgui.set_cursor_pos((pad, imgui.get_window_height() - BUTTON[1] - 22.0))
        with bar_buttons():
            if self.choosing:
                ready = can_confirm(tiles, self.board)
                imgui.set_cursor_pos((pad + (width - BUTTON[0]) / 2, imgui.get_cursor_pos().y))
                imgui.begin_disabled(not ready)
                if imgui.button("Confirm", BUTTON) or (
                    ready and imgui.is_key_pressed(imgui.Key.enter)
                ):
                    self.confirm()
                imgui.end_disabled()
                return
            other = Board.DEMAND if self.board == Board.SUPPLY else Board.SUPPLY
            key = imgui.Key.right_arrow if other == Board.DEMAND else imgui.Key.left_arrow
            if imgui.button(TITLES[other], BUTTON) or imgui.is_key_pressed(key):
                self.show(other)
            imgui.same_line(0, max(8.0, width - 2 * BUTTON[0]))
            if imgui.button("Close", BUTTON):
                self.open = False
