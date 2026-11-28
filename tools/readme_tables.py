"""Writes README.md's tables of buildings, growth chances, freight and engines from the game's own
figures, so the README never shows a number the game doesn't have. Each table sits between a
`<!-- table: name -->` line and an `<!-- end table -->` line; run this after changing what
buildings make or want, how they grow, the prices or the engines:

    uv run python tools/readme_tables.py

tests/test_readme_tables.py fails while the README is out of date."""

import re
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path

import numpy as np

from railroads.catalogue import BuildingHelp, EngineHelp, building_help, engine_help, fuel_use
from railroads.sim.breeding import breeding_of
from railroads.sim.clock import END_YEAR, START_YEAR
from railroads.sim.goods import BONUS_PART, TRAVELLERS, Good, Need, freight_of
from railroads.sim.ladder_types import Ladder
from railroads.sim.ladders import LADDERS, POINTS
from railroads.sim.ports import TIER_GOODS, TIER_VALUES, monthly
from railroads.sim.surroundings import Land
from railroads.sim.tilemap import Temperature, Terrain
from railroads.sim.yearly import bonus_of, boosts_of
from railroads.words import dollars

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
    if row.since:
        name += f" ({row.since})"
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


def _opened_by(row: EngineHelp) -> str:
    return row.opened_by


def _origins(row: EngineHelp) -> str:
    links = ", ".join(f"[{name}]({url})" for name, url in row.origins)
    return links or "our own design"


def engine_row(row: EngineHelp) -> str:
    engine = row.engine
    cells = [
        engine.title,
        engine.kind.value,
        _on_sale(row),
        _opened_by(row),
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
        "| Engine | Class | On sale | For sale from | Power | Speed | To full speed | Cars | Price |"
        " Burns a cell | Range | Drawn after |"
    )
    return [head, "|" + "---|" * 12] + [engine_row(row) for row in engine_help()]


def ports_table() -> list[str]:
    """Every tier of goods ports draw, its value a month and how many units of each good that
    makes at base prices."""
    lines = ["| Tier | Worth a month | Goods, units a month |", "|---|---|---|"]
    for tier, goods in TIER_GOODS.items():
        named = ", ".join(f"{good.value} {monthly(good, tier):,.4g}" for good in goods)
        lines.append(f"| {tier.value.title()} | {dollars(TIER_VALUES[tier])} | {named} |")
    return lines


def _named(value: object) -> str:
    if value is Temperature.NORMAL:
        return "mild"
    if isinstance(value, Enum):
        return str(value.value) if isinstance(value.value, str) else value.name.lower()
    return str(value)


def _chance(chance: float) -> str:
    """A chance written out in full, to two figures."""
    return np.format_float_positional(chance, precision=2, unique=False, fractional=False, trim="-")


def _weights(weights: Mapping[object, float]) -> str:
    return ", ".join(
        _named(what) if weight == 1 else f"{_named(what)} x{weight:g}"
        for what, weight in weights.items()
    )


def _ladder_chance(ladder: Ladder) -> str:
    """A ladder's chance a month: its base, and what it gains per doubling of what draws it."""
    bred = breeding_of(ladder, POINTS)
    served = bred.served_by or {bred.own: 1.0}
    by = "" if list(served.values()) == [1.0] else f" ({_weights(served)})"
    return (
        f"{_chance(bred.spawn)} + ({_chance(bred.spawn)} + {_chance(bred.per_doubling)}"
        f" per satisfied point{by})"
    )


def _drawn_by(ladder: Ladder) -> str:
    bred = breeding_of(ladder, POINTS)
    drawn = _weights(bred.drawn_by)
    if ladder.draws_from > 1:
        drawn += f", each only from {ladder.draws_from} nearby"
    if bred.shunned_by:
        drawn += f"; kept off by {_weights(bred.shunned_by)}"
    return drawn


def _needs(ladder: Ladder) -> str:
    needs = [f"{amount} {_named(points)}" for points, amount in ladder.needs.items()]
    if ladder.crowded_at is not None:
        needs.append(f"fewer than {ladder.crowded_at} population")
    return ", ".join(needs)


def _limits(ladder: Ladder) -> str:
    limits = []
    if ladder.from_year:
        limits.append(f"from {ladder.from_year}")
    if ladder.most_per_cell is not None:
        limits.append(f"{ladder.most_per_cell} a cell")
    if ladder.most_nearby is not None:
        limits.append(f"{ladder.most_nearby} a neighbourhood")
    if ladder.apart is not None:
        limits.append(f"{ladder.apart} cells apart")
    if ladder.land_cells_each is not None:
        limits.append(f"one per {ladder.land_cells_each:,} land cells")
    return ", ".join(limits)


def _land_weights(weights: Mapping[Enum, float], none: str = "none on") -> list[str]:
    return [
        f"{none} {_named(what)}" if weight == 0 else f"{_named(what)} x{weight:g}"
        for what, weight in weights.items()
    ]


def _degrees(low: float, high: float, plus: str, minus: str) -> str:
    def one(value: float) -> str:
        return f"{abs(value):g}°{plus if value >= 0 else minus}"

    return f"{one(low)} to {one(high)}"


def _land(land: Land) -> str:
    """Every rule of a ladder's land in words."""
    rules = [
        *_land_weights(land.vegetation),
        *_land_weights(land.relief, "none in"),
        *_land_weights(land.temperature, "none in the"),
        *(
            ("never " if weight == 0 else "")
            + ("away from water" if water == Terrain.LAND else f"by {_named(water)}")
            + ("" if weight == 0 else f" x{weight:g}")
            for water, weight in land.water.items()
        ),
    ]
    if not land.minor_rivers:
        rules.append("minor rivers count for nothing")
    if land.beside_mountains != 1.0:
        rules.append(f"beside mountains x{land.beside_mountains:g}")
    if land.forest_edge:
        rules.append("only where forest meets open flat land")
    if land.least_forest_nearby:
        rules.append(f"at least {land.least_forest_nearby} forest cells around")
    if land.forest_share:
        rules.append("by the share of forest around")
    if land.deposit is not None:
        rules.append(f"only over {_named(land.deposit)} deposits")
    if land.most_latitude is not None:
        rules.append(f"only within {land.most_latitude:g}° of the equator")
    if land.region is not None:
        west, east, south, north = land.region
        rules.append(
            f"only between {_degrees(west, east, 'E', 'W')} and {_degrees(south, north, 'N', 'S')}"
        )
    if land.open_field != 1.0:
        rules.append(f"empty flat grass x{land.open_field:g}")
    return ", ".join(rules)


def growth_table() -> list[str]:
    """Every ladder's chance a month, what draws it, what it needs, its limits and its land."""
    head = (
        "| Ladder | Chance per month: base + (base + per satisfied point) per doubling of what draws"
        " it | Drawn by | Needs nearby | Limits | Land |"
    )
    lines = [head, "|---|---|---|---|---|---|"]
    for ladder in LADDERS:
        name = ", ".join(kind.value for kind in ladder.kinds)
        lines.append(
            f"| {name[0].upper()}{name[1:]} | {_ladder_chance(ladder)} | {_drawn_by(ladder)}"
            f" | {_needs(ladder)} | {_limits(ladder)} | {_land(ladder.land)} |"
        )
    return lines


TABLES: dict[str, Callable[[], list[str]]] = {
    "buildings": buildings_table,
    "growth": growth_table,
    "ports": ports_table,
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
