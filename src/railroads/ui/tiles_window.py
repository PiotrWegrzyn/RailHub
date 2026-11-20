"""The supply and demand window: both boards (railroads.sim.tiles), each tile with its level, what
a unit of its goods earns now and what it would earn once picked, and a button to pick it to go
up on 1 January, with the picks left and the era."""

from dataclasses import dataclass

from imgui_bundle import imgui

from railroads.sim.tiles import CLOSES, ERAS, MOST_APART, TILE_GOODS, UNLOCKS, Board, Tile, Tiles
from railroads.ui.ornate import bar_buttons, ornate_window, section
from railroads.ui.theme import BRASS, GROWN, TEXT_DIM, WINDOW_TOP

WINDOW_SIZE = (620.0, 640.0)
BOARD_HINTS = {
    Board.SUPPLY: "More supply makes more of a tile's goods, and each unit earns less.",
    Board.DEMAND: "More demand makes buyers want more of a tile's goods, and each unit earns more.",
}


@dataclass(frozen=True)
class TileRow:
    tile: Tile
    level: str
    price: str
    after: str
    picked: int
    can_pick: bool
    # The next buyer the tile opens on the demand board, in a few words.
    opens: str = ""


def rows(tiles: Tiles, board: Board) -> list[TileRow]:
    """Every tile on a board, the lowest first: its level with the picks so far, what a unit of
    its goods earns now and would earn once it goes up a level more, how many times it is
    picked and whether it can be."""
    levels = tiles.levels(board)
    now = tiles.of_year(tiles.year)
    shown = []
    for tile in sorted(levels, key=lambda t: (levels[t], list(Tile).index(t))):
        picked = tiles.picks[board].count(tile)
        closed = board == Board.SUPPLY and levels[tile] >= CLOSES.get(tile, levels[tile] + 1)
        level = (
            f"{levels[tile]}" + (f" +{picked}" if picked else "") + (" closed" if closed else "")
        )
        can = tiles.can_pick(board, tile)
        shown.append(
            TileRow(
                tile,
                level,
                f"{now.price(TILE_GOODS[tile][0]):.0%}",
                f"{tiles.price_after(board, tile):.0%}" if can else "-",
                picked,
                can,
                next_unlock(tile, levels[tile]) if board == Board.DEMAND else "",
            )
        )
    return shown


def next_unlock(tile: Tile, level: int) -> str:
    """The next buyer a demand tile opens above a level: the level and who starts taking what."""
    coming = [
        (at, kind, need)
        for (kind, need), (opening, at) in UNLOCKS.items()
        if opening == tile and at > level
    ]
    if not coming:
        return ""
    at = min(lv for lv, _, _ in coming)
    kinds = sorted({kind.value for lv, kind, _ in coming if lv == at})
    needs = sorted({need.value for lv, _, need in coming if lv == at})
    who = kinds[0] + "s" if len(kinds) == 1 else f"{len(kinds)} kinds of building"
    return f"level {at}: {who} take {', '.join(needs)}"


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


class TilesWindow:
    def __init__(self) -> None:
        self.open = False
        # The last year the window opened by itself in.
        self.shown_year = 0

    def new_year(self, year: int) -> None:
        """Opens the window once a year, as the picks for the next one begin."""
        if year > self.shown_year:
            self.shown_year = year
            self.open = True

    def draw(self, tiles: Tiles) -> None:
        if not self.open:
            return
        imgui.set_next_window_pos((80.0, WINDOW_TOP), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        with ornate_window("Supply and demand", "tiles") as (expanded, still_open):
            if expanded:
                self._draw_boards(tiles)
        self.open = still_open

    @staticmethod
    def _draw_boards(tiles: Tiles) -> None:
        imgui.text_wrapped(era_line(tiles))
        imgui.text_colored(
            imgui.ImVec4(*TEXT_DIM),
            f"Picks go up on 1 January {tiles.year + 1}; those not made go to the lowest tiles. "
            f"No tile can be more than {MOST_APART} levels above the lowest on its board.",
        )
        for board in Board:
            left = tiles.picks_left(board)
            section(f"{board.value}: {left} pick{'s' if left != 1 else ''} left")
            imgui.text_colored(imgui.ImVec4(*TEXT_DIM), BOARD_HINTS[board])
            flags = imgui.TableFlags_.row_bg.value
            if not imgui.begin_table(f"board_{board.name}", 6, flags):
                continue
            for heading in ("Tile", "Level", "Earns", "Once up", "Opens", ""):
                imgui.table_setup_column(heading)
            imgui.table_headers_row()
            for row in rows(tiles, board):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(row.tile.value.capitalize())
                imgui.table_next_column()
                colour = GROWN if row.picked else BRASS
                imgui.text_colored(imgui.ImVec4(*colour), row.level)
                imgui.table_next_column()
                imgui.text(row.price)
                imgui.table_next_column()
                imgui.text(row.after)
                imgui.table_next_column()
                imgui.text_colored(imgui.ImVec4(*TEXT_DIM), row.opens)
                imgui.table_next_column()
                with bar_buttons():
                    if row.can_pick and imgui.small_button(f"Pick##{board.name}{row.tile.name}"):
                        tiles.pick(board, row.tile)
                    if row.picked:
                        imgui.same_line(0, 4)
                        if imgui.small_button(f"Undo##{board.name}{row.tile.name}"):
                            tiles.unpick(board, row.tile)
            imgui.end_table()
