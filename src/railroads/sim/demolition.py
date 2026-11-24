"""What clearing things away costs: buildings moved aside for track, a station or a facility, and
everything the bulldozer takes down.

A building costs DEMOLITION_PER_POINT times the points it gives (railroads.sim.ladders): a house or
a field $5,000, mines, factories and works dearer, and a skyscraper a million. Moving one aside
costs that times the surcharge people's anger with the company sets (railroads.sim.anger), and so
does taking one down with the bulldozer; either raises the anger. The player's own buildings, all
of whose shares they hold, cost the plain price and raise no anger, and leave a cell after the
others of their kind, as the player's shares do (railroads.sim.ownership). One-off buildings (UNIQUE) cannot be moved or taken down at all, and
nothing is cleared where one would have to go. Making room moves the smallest buildings first, as
Room.move_aside does, so a price worked out beforehand is what is paid. Track costs TRACK_REMOVAL_SHARE of what it cost to lay, a station
STATION_REMOVAL, taking its platforms and facilities with it, and a facility FACILITY_REMOVAL. Track or a station bulldozed takes with it, for nothing, any branch beside it
that led only into it off a junction; the end of a plain line stays to be laid on from.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from railroads.sim.anger import surcharge
from railroads.sim.constructions import (
    CELL_SPACE,
    ONE_OFFS,
    Annex,
    Building,
    BuildingKind,
    Constructions,
    Station,
)
from railroads.sim.finance import track_price
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.ladders import POINTS
from railroads.sim.ownership import SHARES, Owners
from railroads.sim.points import Points
from railroads.sim.room import leaving_order

DEMOLITION_PER_POINT: Mapping[Points, float] = MappingProxyType(
    {
        Points.POPULATION: 5_000.0,
        Points.FOOD: 5_000.0,
        Points.TIMBER: 15_000.0,
        Points.INDUSTRY: 15_000.0,
        Points.ORE: 20_000.0,
        Points.TRADE: 30_000.0,
        Points.CORPORATE: 20_000.0,
        Points.WORSHIP: 2_000.0,
        Points.MILITARY: 50_000.0,
        Points.AIR: 200_000.0,
    }
)
SKYSCRAPER_DEMOLITION = 1_000_000.0
UNIQUE = ONE_OFFS
TRACK_REMOVAL_SHARE = 0.25
STATION_REMOVAL = 5_000.0
FACILITY_REMOVAL = 2_000.0


def demolition_price(kind: BuildingKind) -> float | None:
    """What taking down one building of a kind costs; None for those that cannot be."""
    if kind in UNIQUE:
        return None
    if kind == BuildingKind.SKYSCRAPER:
        return SKYSCRAPER_DEMOLITION
    points, value = POINTS[kind]
    return DEMOLITION_PER_POINT[points] * value


def moving_price(kind: BuildingKind, anger: int = 0) -> float | None:
    """What moving one building of a kind aside costs while the company's anger is `anger`; None
    for those that cannot be."""
    price = demolition_price(kind)
    return None if price is None else price * surcharge(anger)


def strangers(buildings: Iterable[Building], owners: Owners | None) -> list[Building]:
    """Those of the buildings leaving their cells, in the order they leave, that are not the
    player's own: the player's own buildings of a kind in a cell, all of whose shares they hold,
    leave after the others."""
    leaving = list(buildings)
    if owners is None:
        return leaving
    others: dict[tuple[Cell, BuildingKind], int] = {}
    for cell, kind in Counter((b.cell, b.kind) for b in leaving):
        own = int(owners.shares(kind)[cell]) // SHARES
        others[cell, kind] = owners.total(cell, kind) // SHARES - own
    picked = []
    for building in leaving:
        key = (building.cell, building.kind)
        if others[key] > 0:
            others[key] -= 1
            picked.append(building)
    return picked


def leaving_price(
    buildings: Iterable[Building], anger: int = 0, owners: Owners | None = None
) -> float:
    """What moving the buildings aside or taking them down costs while the company's anger is
    `anger`: the player's own at the plain price, the rest at the surcharge."""
    leaving = list(buildings)
    plain = sum(demolition_price(b.kind) or 0.0 for b in leaving)
    angering = sum(demolition_price(b.kind) or 0.0 for b in strangers(leaving, owners))
    return plain + angering * (surcharge(anger) - 1.0)


def clearing(constructions: Constructions, cell: Cell, units: int) -> list[Building] | None:
    """The buildings that must leave the cell for `units` of room, in the order they go; None if
    one of them cannot be taken down."""
    room = constructions.space_left(cell)
    here = constructions.buildings_at(cell)
    leaving = []
    for building in leaving_order(here):
        if room >= units:
            break
        if building.kind in UNIQUE:
            return None
        leaving.append(building)
        room += building.space
    return leaving


def clearing_price(
    constructions: Constructions,
    cell: Cell,
    units: int,
    anger: int = 0,
    owners: Owners | None = None,
) -> float | None:
    """What making `units` of room in the cell costs while the company's anger is `anger`; None
    where it cannot be made."""
    leaving = clearing(constructions, cell, units)
    if leaving is None:
        return None
    return leaving_price(leaving, anger, owners)


def station_site(constructions: Constructions, cell: Cell) -> list[Building] | None:
    """The buildings that must move aside for a station in the cell: every one there, the track
    giving way for free; None where one cannot be moved."""
    room = constructions.space_left(cell) + constructions.track_room(cell)
    here = constructions.buildings_at(cell)
    if room >= CELL_SPACE:
        return []
    if any(b.kind in UNIQUE for b in here):
        return None
    return list(here)


def station_site_price(
    constructions: Constructions, cell: Cell, anger: int = 0, owners: Owners | None = None
) -> float | None:
    """What clearing a cell for a station costs while the company's anger is `anger`; None where
    a building cannot be moved."""
    site = station_site(constructions, cell)
    if site is None:
        return None
    return leaving_price(site, anger, owners)


def bulldozed_buildings(constructions: Constructions, cell: Cell) -> list[Building]:
    """The buildings bulldozing a cell takes down: all but the one-offs."""
    return [b for b in constructions.buildings_at(cell) if b.kind not in UNIQUE]


def bulldoze_price(
    constructions: Constructions, cell: Cell, anger: int = 0, owners: Owners | None = None
) -> float | None:
    """What bulldozing a cell costs while the company's anger is `anger`: every building at
    what moving it aside would cost, every piece of track and station there, one-off buildings
    left standing; None when there is nothing it can take down."""
    tile = constructions.tile_map.tile(cell)
    taken = bulldozed_buildings(constructions, cell)
    total, found = leaving_price(taken, anger, owners), bool(taken)
    for built in constructions.at(cell):
        if isinstance(built, Building):
            continue
        if isinstance(built, Station):
            total += STATION_REMOVAL
        elif isinstance(built, Annex):
            total += FACILITY_REMOVAL
        else:
            total += track_price(built, tile) * TRACK_REMOVAL_SHARE
        found = True
    return total if found else None


def bulldoze(growth: Growth, cell: Cell) -> set[Cell]:
    """Takes down everything in the cell that can be, buildings the player owns too; returns
    the cells that changed."""
    constructions = growth.constructions
    taken: set[Cell] = set()
    railway = any(not isinstance(built, (Building, Annex)) for built in constructions.at(cell))
    for built in constructions.at(cell):
        if isinstance(built, Building):
            if built.kind in UNIQUE:
                continue
            growth.take_down(built)
        else:
            if isinstance(built, Station) and built.main is None:
                taken |= remove_added(constructions, cell)
            constructions.remove(built)
        taken.add(cell)
    if railway:
        taken |= remove_dead_branches(constructions, cell)
    return taken


def remove_dead_branches(constructions: Constructions, cell: Cell) -> set[Cell]:
    """Takes up the pieces of track beside an emptied cell that led into it off a junction: their
    other end joins another piece in their cell. Returns their cells."""
    grid = constructions.tile_map.grid
    taken: set[Cell] = set()
    for point in range(len(grid.track_points())):
        beside, entry = grid.track_exit(cell, point)
        for piece in constructions.track_at(beside):
            if entry not in piece.ends:
                continue
            far = piece.other_end(entry)
            others = constructions.track_at(beside)
            if any(far in other.ends for other in others if other is not piece):
                constructions.remove(piece)
                taken.add(beside)
    return taken


def remove_added(constructions: Constructions, station: Cell) -> set[Cell]:
    """Takes down every platform and facility added to a station; returns their cells."""
    cells = set(constructions.platforms(station))
    for cell in cells:
        platform = constructions.station_at(cell)
        if platform is not None:
            constructions.remove(platform)
    for cell in constructions.annexes(station).values():
        annex = constructions.annex_at(cell)
        if annex is not None:
            constructions.remove(annex)
            cells.add(cell)
    return cells
