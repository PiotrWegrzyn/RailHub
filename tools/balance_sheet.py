"""Writes every number growth and the supply and demand boards run on into a spreadsheet,
docs/balance.xlsx, read straight from the game's rules, to look over and to try numbers in:

- Buildings: every ladder, from its base chance and land weights to its needs, caps, merges and
  the year it opens, with what draws and keeps it away, the points it gives and what one makes
  and wants a month;
- Growth rules: the numbers every ladder shares, what draws each kind of points, and forests;
- Tiles: every tile's goods, base levels and steps;
- Supply, Demand and Prices: every tile's level and factor year by year, as the world grows by
  itself (the lowest tiles go up each year), and what a unit of its goods earns.

The game does not read the spreadsheet back: a number changed there is changed in the game by
changing it in the code it names.

    uv run --with openpyxl python tools/balance_sheet.py
"""

from collections.abc import Iterable, Mapping
from dataclasses import fields
from enum import Enum
from itertools import pairwise
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from railroads.sim import breeding, forests, growth, ladders, tiles
from railroads.sim.breeding import breeding_of
from railroads.sim.clock import END_YEAR
from railroads.sim.goods import MAKES, NEEDS
from railroads.sim.ladder_types import Ladder
from railroads.sim.surroundings import Land

OUT = Path(__file__).resolve().parents[1] / "docs" / "balance.xlsx"
HEADER = PatternFill("solid", fgColor="3B2F22")
HEADER_FONT = Font(bold=True, color="F0E0C0")
LAND_RULES = [rule.name for rule in fields(Land)]


def name(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value) if isinstance(value.value, str) else value.name.lower()
    return str(value)


def mapping(values: Mapping[object, object] | None) -> str:
    """A mapping as "key: value; key: value", empty for none."""
    if not values:
        return ""
    return "; ".join(f"{name(key)}: {name(value)}" for key, value in values.items())


def land_cell(land: Land, rule: str) -> object:
    value = getattr(land, rule)
    if isinstance(value, Mapping):
        return mapping(value)
    if isinstance(value, Enum):
        return name(value)
    if isinstance(value, tuple):
        return ", ".join(f"{part:g}" for part in value)
    return value


def ladder_row(ladder: Ladder) -> list[object]:
    first = ladder.kinds[0]
    points, value = ladder_points(first)
    bred = breeding_of(ladder, ladders.POINTS)
    pull = ladder.per_doubling
    merges = "; ".join(
        f"{merge.needed} {small.value} -> {big.value} (pulls down {merge.pulled_down})"
        for small, big in pairwise(ladder.kinds)
        if (merge := ladder.merges.get(small)) is not None
    )
    rises = "; ".join(
        f"{small.value} -> {big.value} at {rise.people} people"
        + (" (first only)" if rise.only_one else "")
        for small, big in pairwise(ladder.kinds)
        if (rise := ladder.rises.get(small)) is not None
    )
    joins = "" if ladder.joins is None else f"{ladder.joins.needed} -> {ladder.joins.into.value}"
    return [
        first.value,
        ", ".join(kind.value for kind in ladder.kinds[1:]),
        name(points),
        value,
        ladder.base,
        pull if pull is not None else breeding.BREEDING_TIMES * ladder.base,
        mapping(bred.drawn_by),
        mapping(bred.shunned_by),
        ladder.draws_from,
        mapping(ladder.needs),
        ladder.crowded_at,
        ladder.most_per_cell,
        ladder.most_nearby,
        ladder.apart,
        ladder.land_cells_each,
        merges,
        rises,
        joins,
        ladder.from_year or "",
        tiles.kind_era(first),
        ladder.starving_years,
        mapping(MAKES.get(first)),
        mapping(NEEDS.get(first)),
        *(land_cell(ladder.land, rule) for rule in LAND_RULES),
    ]


def ladder_points(kind: object) -> tuple[object, int]:
    return ladders.POINTS[kind]  # type: ignore[index]


BUILDING_COLUMNS = [
    ("kind", "The ladder's first kind: what a turn puts up."),
    ("higher kinds", "What it climbs to by merging or rising."),
    ("points", "The kind of points its buildings give."),
    ("points each", "How many of them one gives."),
    ("base", "Chance a month in a cell before anything around, in CHANCE_UNITs (1e-7)."),
    ("per doubling", "What each served point adds per doubling of what draws it, in CHANCE_UNITs."),
    ("drawn by", "Points around that draw it, each times its weight (breeding.DRAWN_BY)."),
    ("shunned by", "Points around its chance is divided by, one plus each times its weight."),
    ("draws from", "Points of a kind draw more only from this many."),
    ("needs", "Points the neighbourhood must hold before it has any chance."),
    ("crowded at", "Grows only where fewer population points are around."),
    ("most per cell", "The most of its first kind in one cell."),
    ("most nearby", "The most of its first kind in one neighbourhood."),
    ("apart", "New ones keep this many cells from any on the ladder."),
    ("land cells each", "At most one on the map for this many land cells."),
    ("merges", "How many in a neighbourhood merge into the next kind."),
    ("rises", "People around that turn one into the next kind."),
    ("joins", "How many of its first kind join into another kind."),
    ("from year", "The first year it grows."),
    ("era", "The era it comes with."),
    ("starving years", "Years it lasts starving, if not the usual."),
    ("makes a month", "What one makes a month."),
    ("wants a month", "What one wants a month."),
    *((f"land: {rule}", "Land weight (railroads.sim.surroundings.Land).") for rule in LAND_RULES),
]


def write_table(
    sheet: object, columns: list[tuple[str, str]], rows: Iterable[list[object]]
) -> None:
    """A table with a header row, its notes as the headers' comments, frozen and filtered."""
    from openpyxl.comments import Comment

    for col, (title, note) in enumerate(columns, start=1):
        cell = sheet.cell(row=1, column=col, value=title)  # type: ignore[attr-defined]
        cell.fill, cell.font = HEADER, HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if note:
            cell.comment = Comment(note, "RailHub")
    for row, values in enumerate(rows, start=2):
        for col, value in enumerate(values, start=1):
            sheet.cell(row=row, column=col, value=value)  # type: ignore[attr-defined]
    for col, (title, _) in enumerate(columns, start=1):
        width = max(12, min(40, len(title) + 4))
        sheet.column_dimensions[get_column_letter(col)].width = width  # type: ignore[attr-defined]
    sheet.freeze_panes = "B2"  # type: ignore[attr-defined]
    sheet.auto_filter.ref = sheet.dimensions  # type: ignore[attr-defined]


def rules_rows() -> list[list[object]]:
    rows: list[list[object]] = [
        ["CHANCE_UNIT", ladders.CHANCE_UNIT, "ladders.py", "What a base of 1 is, a month."],
        ["BREEDING_TIMES", breeding.BREEDING_TIMES, "breeding.py", "Per doubling, x base."],
        ["OWN_KIND_TIMES", breeding.OWN_KIND_TIMES, "breeding.py", "Own kind draws this much."],
        ["CATEGORY_SHARE", breeding.CATEGORY_SHARE, "breeding.py", "Rest of its points."],
        ["ANIMALS_DRAW", breeding.ANIMALS_DRAW, "breeding.py", "Animal farms draw butchers."],
        ["LIVELY_GROWTH", breeding.LIVELY_GROWTH, "breeding.py", "Homes by served venues."],
        ["OPEN_FIELD", ladders.OPEN_FIELD, "ladders.py", "Fields on open flat grass."],
        ["FARMS_CROWDED_OUT", ladders.FARMS_CROWDED_OUT, "ladders.py", "Fields go here."],
        ["SKYSCRAPER_PEOPLE", ladders.SKYSCRAPER_PEOPLE, "ladders.py", ""],
        ["SKYSCRAPER_CHANCE", ladders.SKYSCRAPER_CHANCE, "ladders.py", ""],
        ["KEY_LEAST", growth.KEY_LEAST, "growth.py", "Oil wells, refineries kept."],
        ["CLEARING_PER_POINT", forests.CLEARING_PER_POINT, "forests.py", "A month a cell."],
        ["CLEARING_PER_MILL", forests.CLEARING_PER_MILL, "forests.py", "A month a cell."],
        ["SPREAD_PER_FOREST", forests.SPREAD_PER_FOREST, "forests.py", "A month a cell."],
        ["MILLS_CLOSE_BELOW", forests.MILLS_CLOSE_BELOW, "forests.py", "Forest cells around."],
        ["PRICE_POWER", tiles.PRICE_POWER, "tiles.py", "(demand / supply) ** this."],
        ["MOST_APART", tiles.MOST_APART, "tiles.py", "Levels above the lowest."],
        ["SUPPLY_PICKS", tiles.SUPPLY_PICKS, "tiles.py", "A year."],
        ["DEMAND_PICKS", tiles.DEMAND_PICKS, "tiles.py", "A year."],
    ]
    rows.append([])
    rows.append(["DRAWN_BY (points of a ladder -> what draws it)", "", "breeding.py", ""])
    rows += [[name(points), mapping(drawn), "", ""] for points, drawn in breeding.DRAWN_BY.items()]
    rows.append([])
    rows.append(["DRAWN_BY_KIND (kind -> what draws it)", "", "breeding.py", ""])
    rows += [[kind.value, mapping(drawn), "", ""] for kind, drawn in breeding.DRAWN_BY_KIND.items()]
    return rows


def tile_rows() -> list[list[object]]:
    return [
        [
            tile.value,
            ", ".join(good.value for good in tiles.TILE_GOODS.get(tile, ())),
            next((era.level for era in tiles.ERAS if tile in era.tiles), ""),
            *tiles.BASE_LEVELS.get(tile, ("", "")),
            ", ".join(f"{step:g}" for step in tiles.SUPPLY_STEPS.get(tile, ())),
            tiles.DEMAND_STEPS.get(tile, ""),
        ]
        for tile in tiles.Tile
    ]


def by_year(board: str) -> tuple[list[tuple[str, str]], list[list[object]]]:
    """Every tile's level and factor (or its price) each year, the world growing by itself."""
    grown = tiles.Tiles.until(END_YEAR)
    every = list(tiles.Tile)
    if board == "price":
        columns = [("year", ""), ("era", "")] + [(tile.value, "") for tile in every]
    else:
        columns = [("year", ""), ("era", "")] + [
            column
            for tile in every
            for column in ((f"{tile.value} level", ""), (f"{tile.value} x", ""))
        ]
    rows = []
    for year in sorted(grown.years):
        boards = grown.of_year(year)
        row: list[object] = [year, boards.era]
        for tile in every:
            if board == "price":
                made, wanted = boards.made.get(tile), boards.wanted.get(tile)
                row.append(None if made is None else round(tiles.price(made, wanted or 1.0), 3))
                continue
            levels = boards.supply if board == "supply" else boards.demand
            factors = boards.made if board == "supply" else boards.wanted
            level = levels.get(tile)
            factor = factors.get(tile)
            row += [level, None if factor is None else round(factor, 3)]
        rows.append(row)
    return columns, rows


def main() -> None:
    book = Workbook()
    sheet = book.active
    sheet.title = "Buildings"
    write_table(sheet, BUILDING_COLUMNS, (ladder_row(ladder) for ladder in ladders.LADDERS))
    write_table(
        book.create_sheet("Growth rules"),
        [("name", ""), ("value", ""), ("file", "Under src/railroads/sim."), ("what", "")],
        rules_rows(),
    )
    write_table(
        book.create_sheet("Tiles"),
        [
            ("tile", ""),
            ("goods", ""),
            ("era level", "The level every open supply tile must reach for its era."),
            ("base supply level", "Its level in the benchmark year, BENCHMARK_YEAR."),
            ("base demand level", ""),
            ("supply steps", "The share of base each level adds, by period (PERIODS)."),
            ("demand step", "The share of base each demand level adds."),
        ],
        tile_rows(),
    )
    for board, title in (("supply", "Supply"), ("demand", "Demand"), ("price", "Prices")):
        columns, rows = by_year(board)
        write_table(book.create_sheet(title), columns, rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    book.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
