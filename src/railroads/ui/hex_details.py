"""The panel that opens when a cell is clicked: its land, its points, and every building on it with
what it makes, demands and takes as bonus goods, and how last month's trade went for it."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatCells = npt.NDArray[np.float64]
from imgui_bundle import imgui

from railroads.sim.constructions import Building, BuildingKind, Construction, Station, Track
from railroads.sim.economy import Satisfaction
from railroads.sim.finance import dollars
from railroads.sim.goods import RECIPES, Good, Need, bonus_of, demands_of, makes_of
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.ladders import Points
from railroads.sim.tilemap import Deposit, TileMap
from railroads.ui.biomes import biome_of
from railroads.ui.theme import good_amount


@dataclass(frozen=True)
class Flow:
    """How much of a good buildings got or gave out of what they wanted to; for goods made, also
    the most they could make with all their inputs, where that is more."""

    good: Good | Need
    done: float
    wanted: float
    up_to: float | None = None

    @property
    def text(self) -> str:
        line = f"{self.done:.1f} of {self.wanted:.1f}"
        return line if self.up_to is None else f"{line} (up to {self.up_to:g})"


@dataclass(frozen=True)
class BuildingRow:
    name: str
    count: int
    makes: list[tuple[Good, float]]
    # What it must get to be satisfied, and the bonus goods that raise it above that.
    needs: list[tuple[Need, float]]
    bonus: list[tuple[Need, float]]
    got: list[Flow]
    gave: list[Flow]
    # What processors hold of each input, waiting to be used.
    stock: list[tuple[Need, float]]
    satisfaction: str
    kind: BuildingKind | None = None
    # How many of them the player owns, and what one more costs; None for what cannot be bought.
    owned: int = 0
    price: float | None = None
    # Its satisfaction last month, past 1 with bonus goods; None where it does not trade.
    percent: float | None = None


@dataclass(frozen=True)
class HexDetails:
    title: str
    land: str
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
        got: list[Flow] = []
        gave: list[Flow] = []
        stock: list[tuple[Need, float]] = []
        mood = "-"
        percent = None
        if trade is not None and position is not None and kind in trade.got:
            got = _per_good(trade.goods_got, kind, position)
            if kind in RECIPES:
                stock = _stock(growth, kind, flat)
            gave = _per_good(trade.goods_gave, kind, position, makes_of(kind, growth.year), count)
            mood = Satisfaction(int(trade.levels((kind,))[position])).name.lower()
            percent = float(trade.share((kind,))[position])
        rows.append(
            BuildingRow(
                kind.value,
                count,
                [(good, amount * count) for good, amount in makes_of(kind, growth.year).items()],
                [(need, amount * count) for need, amount in demands_of(kind, growth.year).items()],
                [(need, amount * count) for need, amount in bonus_of(kind, growth.year).items()],
                got,
                gave,
                stock,
                mood,
                kind,
                int(growth.owners.owned(kind)[cell]),
                growth.owners.price_to_buy(kind),
                percent,
            )
        )
    return HexDetails(
        title=f"Cell {cell[0]}, {cell[1]}: {biome_of(tile).name.title()}",
        land=(
            f"{tile.terrain.name.lower()}, {tile.relief.name.lower()}, "
            f"{tile.vegetation.name.lower()}, {tile.temperature.name.lower()}, {tile.height_m} m"
            + _deposit_note(tile_map, cell)
        ),
        satisfaction=Satisfaction(level).name.lower() if level >= 0 else "nothing trades here",
        points=points,
        buildings=rows,
        others=[_other(c, growth) for c in built if not isinstance(c, Building)],
    )


def draw_hex_details(details: HexDetails) -> BuildingKind | None:
    """Draws the details inside the current imgui window; returns the kind of building whose Buy
    button was pressed, if any."""
    bought = None
    imgui.text(details.title)
    imgui.text_disabled(details.land)
    imgui.text(f"Satisfaction: {details.satisfaction}")
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
        return None
    if details.buildings:
        flags = (
            imgui.TableFlags_.borders.value
            | imgui.TableFlags_.row_bg.value
            | imgui.TableFlags_.sizing_fixed_fit.value
        )
        if imgui.begin_table("buildings", 6, flags):
            headings = (
                "Building",
                "#",
                "Needs: got of needed",
                "Makes: gave of made",
                "Mood",
                "Yours",
            )
            for heading in headings:
                imgui.table_setup_column(heading)
            imgui.table_headers_row()
            for row in details.buildings:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(row.name)
                imgui.table_next_column()
                imgui.text(str(row.count))
                imgui.table_next_column()
                _draw_flows(row.got)
                if row.stock:
                    imgui.text_disabled("in stock")
                    for need, amount in row.stock:
                        imgui.same_line(0, 8)
                        good_amount(need, f"{amount:.0f}")
                imgui.table_next_column()
                _draw_flows(row.gave)
                imgui.table_next_column()
                imgui.text(mood_text(row))
                imgui.table_next_column()
                if row.price is None:
                    imgui.text_disabled("-")
                    continue
                imgui.text(f"{row.owned} of {row.count}")
                if row.owned < row.count and imgui.small_button(
                    f"Buy {dollars(row.price)}##{row.name}"
                ):
                    bought = row.kind
            imgui.end_table()
    for other in details.others:
        imgui.bullet_text(other)
    return bought


def mood_text(row: BuildingRow) -> str:
    """How satisfied a building is, with its satisfaction as a percentage where it trades."""
    if row.percent is None:
        return row.satisfaction
    return f"{row.satisfaction} {row.percent:.0%}"


def _draw_flows(flows: list[Flow]) -> None:
    """Each good's icon and how it went, one good to a line."""
    if not flows:
        imgui.text_disabled("-")
    for flow in flows:
        good_amount(flow.good, flow.text)


def _deposit_note(tile_map: TileMap, cell: Cell) -> str:
    """Names the deposit under a cell, if there is one."""
    if tile_map.deposits is None or tile_map.deposits[cell] in (Deposit.NONE, Deposit.GARRISON):
        return ""
    return f"; {Deposit(int(tile_map.deposits[cell])).name.lower()} deposit"


def _stock(growth: Growth, kind: BuildingKind, flat: int) -> list[tuple[Need, float]]:
    """What a kind of processor holds of each input, waiting to be used."""
    return [
        (need, float(growth.economy.stock[(kind, need)][flat])) for need in RECIPES[kind].per_unit
    ]


def _per_good(
    goods: Mapping[tuple[BuildingKind, Good], tuple[FloatCells, FloatCells]]
    | Mapping[tuple[BuildingKind, Need], tuple[FloatCells, FloatCells]],
    kind: BuildingKind,
    position: int,
    most: Mapping[Good, float] | None = None,
    count: int = 1,
) -> list[Flow]:
    """How much of each good the buildings got or gave, out of what they needed or made; for
    goods made, also the most they could make with all their inputs."""
    flows = []
    for (of, good), (done, wanted) in goods.items():
        if of != kind or (wanted[position] <= 0 and not most):
            continue
        limit = most.get(good, 0) * count if most else 0
        flows.append(
            Flow(
                good,
                float(done[position]),
                float(wanted[position]),
                limit if limit > wanted[position] + 1e-9 else None,
            )
        )
    return flows


def _other(construction: Construction, growth: Growth) -> str:
    match construction:
        case Station():
            if growth.railway is None:
                return "station"
            return f"station: {growth.railway.name_of(construction.cell)}"
        case Track():
            electric = ", electric" if construction.electric else ""
            return f"track ({construction.width.name.lower()}, {construction.bed.name.lower()}{electric})"
    return str(construction)
