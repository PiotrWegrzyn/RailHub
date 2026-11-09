"""The panel that opens when a cell is clicked: its land, its points, and every building on it with
what it makes and needs and how last month's trade went for it."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatCells = npt.NDArray[np.float64]
from imgui_bundle import imgui

from railroads.sim.constructions import Building, BuildingKind, Construction, Station, Track
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
            got = _per_good(trade.goods_got, kind, position)
            gave = _per_good(trade.goods_gave, kind, position, MAKES.get(kind, {}), count)
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
        if imgui.begin_table("buildings", 5, flags):
            for heading in ("Building", "#", "Needs: got of needed", "Makes: gave of made", "Mood"):
                imgui.table_setup_column(heading)
            imgui.table_headers_row()
            for row in details.buildings:
                imgui.table_next_row()
                for value in (row.name, str(row.count), row.got, row.gave, row.satisfaction):
                    imgui.table_next_column()
                    imgui.text(value)
            imgui.end_table()
    for other in details.others:
        imgui.bullet_text(other)


def _goods(per_building: Mapping[Good, float], count: int) -> str:
    if not per_building:
        return "-"
    return ", ".join(f"{amount * count:g} {good.value}" for good, amount in per_building.items())


def _per_good(
    goods: Mapping[tuple[BuildingKind, Good], tuple[FloatCells, FloatCells]],
    kind: BuildingKind,
    position: int,
    most: Mapping[Good, float] | None = None,
    count: int = 1,
) -> str:
    """How much of each good the buildings got or gave, out of what they needed or made, one
    good to a line; for goods made, also the most they could make with all their inputs."""
    lines = []
    for (of, good), (done, wanted) in goods.items():
        if of != kind or (wanted[position] <= 0 and not most):
            continue
        line = f"{good.value} {done[position]:.1f} of {wanted[position]:.1f}"
        if most and most.get(good, 0) * count > wanted[position] + 1e-9:
            line += f" (up to {most[good] * count:g})"
        lines.append(line)
    return "\n".join(lines) or "-"


def _other(construction: Construction) -> str:
    match construction:
        case Station():
            return "station"
        case Track():
            return f"track ({construction.width.name.lower()}, {construction.bed.name.lower()})"
    return str(construction)
