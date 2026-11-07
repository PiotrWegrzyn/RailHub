"""The panel that opens when a cell is clicked: its land, its points, and every building on it with
what it makes and needs and how last month's trade went for it."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from imgui_bundle import imgui

from railroads.sim.constructions import Building, Construction, Station, Track
from railroads.sim.economy import MAKES, NEEDS, Good, Satisfaction
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth, Points
from railroads.ui.biomes import biome_of


@dataclass(frozen=True)
class BuildingRow:
    name: str
    count: int
    makes: str
    needs: str
    got: str
    gave: str
    satisfaction: str


@dataclass(frozen=True)
class HexDetails:
    title: str
    land: str
    free_food: float
    satisfaction: str
    # Each kind of points: in the cell and in its neighbourhood.
    points: list[tuple[str, int, int]]
    buildings: list[BuildingRow]
    others: list[str]


def describe(growth: Growth, cell: Cell) -> HexDetails:
    """Everything the panel shows about one cell."""
    constructions = growth.constructions
    tile_map = constructions.tile_map
    tile = tile_map.tile(cell)
    flat = int(np.ravel_multi_index(cell, tile_map.grid.shape))
    level = int(growth.satisfaction()[cell])
    points = [
        (kind.value, int(growth.points(kind)[cell]), int(growth.nearby(kind)[cell]))
        for kind in Points
        if growth.nearby(kind)[cell]
    ]
    built = constructions.at(cell)
    kinds = Counter(b.kind for b in built if isinstance(b, Building))
    trade = growth.trade
    position = None
    if trade is not None:
        found = int(np.searchsorted(trade.cells, flat))
        if found < trade.cells.size and trade.cells[found] == flat:
            position = found
    rows = []
    for kind, count in kinds.items():
        got = gave = mood = "-"
        if trade is not None and position is not None and kind in trade.got:
            got = _out_of(trade.got[kind][position], trade.needed[kind][position])
            gave = _out_of(trade.gave[kind][position], trade.could_make[kind][position])
            mood = Satisfaction(int(trade.levels((kind,))[position])).name.lower()
        rows.append(
            BuildingRow(
                kind.value,
                count,
                _goods(MAKES.get(kind, {}), count),
                _goods(NEEDS.get(kind, {}), count),
                got,
                gave,
                mood,
            )
        )
    return HexDetails(
        title=f"Cell {cell[0]}, {cell[1]}: {biome_of(tile).name.title()}",
        land=(
            f"{tile.terrain.name.lower()}, {tile.relief.name.lower()}, "
            f"{tile.vegetation.name.lower()}, {tile.temperature.name.lower()}, {tile.height_m} m"
        ),
        free_food=float(growth.economy.free_food[flat]),
        satisfaction=Satisfaction(level).name.lower() if level >= 0 else "nothing trades here",
        points=points,
        buildings=rows,
        others=[_other(c) for c in built if not isinstance(c, Building)],
    )


def draw_hex_details(details: HexDetails) -> None:
    """Draws the details inside the current imgui window."""
    imgui.text(details.title)
    imgui.text_disabled(details.land)
    imgui.text(f"Free food: {details.free_food:g}   Satisfaction: {details.satisfaction}")
    imgui.separator()
    if details.points:
        imgui.text("Points (here / neighbourhood)")
        for name, here, nearby in details.points:
            imgui.bullet_text(f"{name}: {here} / {nearby}")
    else:
        imgui.text_disabled("No points nearby")
    imgui.separator()
    if not details.buildings and not details.others:
        imgui.text_disabled("Nothing built here")
        return
    if details.buildings:
        flags = imgui.TableFlags_.borders.value | imgui.TableFlags_.row_bg.value
        if imgui.begin_table("buildings", 7, flags):
            for heading in ("Building", "#", "Makes", "Needs", "Got", "Gave", "Mood"):
                imgui.table_setup_column(heading)
            imgui.table_headers_row()
            for row in details.buildings:
                imgui.table_next_row()
                for value in (
                    row.name,
                    str(row.count),
                    row.makes,
                    row.needs,
                    row.got,
                    row.gave,
                    row.satisfaction,
                ):
                    imgui.table_next_column()
                    imgui.text(value)
            imgui.end_table()
    for other in details.others:
        imgui.bullet_text(other)


def _goods(per_building: Mapping[Good, float], count: int) -> str:
    if not per_building:
        return "-"
    return ", ".join(f"{amount * count:g} {good.value}" for good, amount in per_building.items())


def _out_of(done: float, wanted: float) -> str:
    return "-" if wanted <= 0 else f"{done:.1f} of {wanted:g}"


def _other(construction: Construction) -> str:
    match construction:
        case Station():
            return "station"
        case Track():
            return f"track ({construction.width.name.lower()}, {construction.bed.name.lower()})"
    return str(construction)
