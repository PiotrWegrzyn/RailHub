"""Writes README.md's tables of buildings, freight and engines from the game's own figures, so the
README never shows a number the game doesn't have. Each table sits between a `<!-- table: name -->`
line and an `<!-- end table -->` line; run this after changing what buildings make or want, the
prices or the engines:

    uv run python tools/readme_tables.py

tests/test_readme_tables.py fails while the README is out of date."""

import re
from collections.abc import Callable
from pathlib import Path

from railroads.sim.clock import END_YEAR, START_YEAR
from railroads.sim.finance import dollars
from railroads.sim.goods import (
    BONUS_PART,
    TRAVELLERS,
    Good,
    Need,
    bonus_of,
    boosts_of,
    freight_of,
)
from railroads.ui.buy_train import fuel_use
from railroads.ui.help_window import BuildingHelp, EngineHelp, building_help, engine_help

README = Path(__file__).resolve().parent.parent / "README.md"
# The freight bands of the freight table: each one's name and the most freight in it.
FREIGHT_BANDS = (
    ("$10 or less", 10.0),
    ("$11-20", 20.0),
    ("$21-40", 40.0),
    ("$41-200", 200.0),
    ("over $200", float("inf")),
)


def _amounts(amounts: list[tuple[Need | Good, float, str]]) -> str:
    """Goods or needs with their amounts, and from when each is wanted where it isn't always."""
    return ", ".join(
        f"{amount:,.4g} {what.value}" + (f" (from {since})" if since else "")
        for what, amount, since in amounts
    )


def building_row(row: BuildingHelp) -> str:
    """A building's line: what it makes a month at most, what it must get, and what it takes as
    bonus goods or boosts, before history changes any of it (railroads.sim.events); a need is a
    bonus good or a boost if it is one in any decade it is wanted in."""
    kind = row.kind
    extras = {
        need
        for year in range(START_YEAR, END_YEAR + 1, 10)
        for need in (*bonus_of(kind, year), *boosts_of(kind, year))
    }
    must: list[tuple[Need | Good, float, str]] = []
    bonus: list[tuple[Need | Good, float, str]] = []
    for need, amount, since in row.wants:
        spare = BONUS_PART.get((kind, need), 0.0)
        if need in extras:
            bonus.append((need, amount, since))
            continue
        if amount - spare > 0:
            must.append((need, amount - spare, since))
        if spare:
            bonus.append((need, spare, since))
    makes = [(good, amount, "") for good, amount in row.makes]
    name = kind.value[0].upper() + kind.value[1:]
    if row.from_year:
        name += f" ({row.from_year})"
    return f"| {name} | {_amounts(makes)} | {_amounts(must)} | {_amounts(bonus)} |"


def buildings_table() -> list[str]:
    lines = [
        "| Building | Makes a month at most | Must get a month | Bonus goods and boosts a month |",
        "|---|---|---|---|",
    ]
    return lines + [building_row(row) for row in building_help()]


def freight_table() -> list[str]:
    """Every good but passengers and mail by its freight, cheapest first, in FREIGHT_BANDS."""
    goods = sorted(
        (good for good in Good if good not in TRAVELLERS), key=lambda g: (freight_of(g), g.value)
    )
    lines = ["| Freight a unit | Goods |", "|---|---|"]
    low = 0.0
    for name, high in FREIGHT_BANDS:
        band = [good for good in goods if low < freight_of(good) <= high]
        low = high
        if band:
            named = ", ".join(f"{good.value} {freight_of(good):,.4g}" for good in band)
            lines.append(f"| {name} | {named} |")
    return lines


def _on_sale(row: EngineHelp) -> str:
    if row.sold_until is None:
        return f"{row.engine.year} on"
    year, title = row.sold_until
    return f"{row.engine.year}–{year} (then the {title})"


def _tech(row: EngineHelp) -> str:
    return "; ".join(f"{tree.value}: {' → '.join(nodes)}" for tree, nodes in row.tech)


def _origins(row: EngineHelp) -> str:
    links = ", ".join(f"[{name}]({url})" for name, url in row.origins)
    return links or "our own design"


def engine_row(row: EngineHelp) -> str:
    engine = row.engine
    cells = [
        engine.title,
        engine.kind.value,
        _on_sale(row),
        _tech(row),
        engine.power.value,
        f"{engine.kmh:.0f} km/h",
        f"{engine.pickup:g} s",
        f"{engine.cars} + dining car" if engine.express else str(engine.cars),
        dollars(engine.price),
        fuel_use(engine).removesuffix(" a cell"),
        f"{engine.tank:.0f} cells" if engine.tank else "-",
        _origins(row),
    ]
    return "| " + " | ".join(cells) + " |"


def engines_table() -> list[str]:
    head = (
        "| Engine | Class | On sale | Tech to take | Power | Speed | To full speed | Cars | Price |"
        " Burns a cell | Range | Drawn after |"
    )
    return [head, "|" + "---|" * 12] + [engine_row(row) for row in engine_help()]


TABLES: dict[str, Callable[[], list[str]]] = {
    "buildings": buildings_table,
    "freight": freight_table,
    "engines": engines_table,
}


def with_tables(text: str) -> str:
    """The README's text with every table written anew between its markers."""
    for name, table in TABLES.items():
        pattern = re.compile(
            rf"(<!-- table: {name} -->\n).*?(\n<!-- end table -->)", flags=re.DOTALL
        )
        if not pattern.search(text):
            raise ValueError(f"README.md has no table called {name}")
        lines = "\n".join(table())
        text = pattern.sub(lambda match, lines=lines: match[1] + lines + match[2], text)
    return text


def main() -> None:
    README.write_text(with_tables(README.read_text()))


if __name__ == "__main__":
    main()
