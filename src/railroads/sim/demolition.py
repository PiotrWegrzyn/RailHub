"""What clearing things away costs: buildings a station or track makes room for, and everything the
bulldozer takes down.

A building costs DEMOLITION_PER_POINT times the points it gives (railroads.sim.ladders): homes and
fields are cheap, mines, factories and works dear, and a skyscraper costs a million. One-off
buildings (UNIQUE) cannot be taken down at all, and nothing is cleared where one would have to go.
Making room takes the smallest buildings first, as Growth.clear_room does, so a price worked out
beforehand is what is paid. Track costs TRACK_REMOVAL_SHARE of what it cost to lay, and a station
STATION_REMOVAL.
"""

from collections.abc import Mapping
from types import MappingProxyType

from railroads.sim.constructions import CELL_SPACE, Building, BuildingKind, Constructions, Station
from railroads.sim.finance import track_price
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.ladders import POINTS, Points
from railroads.sim.room import leaving_order

DEMOLITION_PER_POINT: Mapping[Points, float] = MappingProxyType(
    {
        Points.POPULATION: 500.0,
        Points.FOOD: 1_000.0,
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
UNIQUE = frozenset(
    {
        BuildingKind.SPACE_CENTER,
        BuildingKind.PEAR_HQ,
        BuildingKind.ROYAL_PALACE,
        BuildingKind.SALT_CATHEDRAL,
        BuildingKind.SPA_RESORT,
        BuildingKind.WORLDS_FAIR,
        BuildingKind.FILM_STUDIO,
        BuildingKind.AMUSEMENT_PARK,
        BuildingKind.OLYMPIC_STADIUM,
        BuildingKind.ST_PAULS,
    }
)
TRACK_REMOVAL_SHARE = 0.25
STATION_REMOVAL = 5_000.0


def demolition_price(kind: BuildingKind) -> float | None:
    """What taking down one building of a kind costs; None for those that cannot be."""
    if kind in UNIQUE:
        return None
    if kind == BuildingKind.SKYSCRAPER:
        return SKYSCRAPER_DEMOLITION
    points, value = POINTS[kind]
    return DEMOLITION_PER_POINT[points] * value


def clearing(constructions: Constructions, cell: Cell, units: int) -> list[Building] | None:
    """The buildings that must leave the cell for `units` of room, in the order they go; None if
    one of them cannot be taken down."""
    room = constructions.space_left(cell)
    here = [b for b in constructions.at(cell) if isinstance(b, Building)]
    leaving = []
    for building in leaving_order(here):
        if room >= units:
            break
        if building.kind in UNIQUE:
            return None
        leaving.append(building)
        room += building.space
    return leaving


def clearing_price(constructions: Constructions, cell: Cell, units: int) -> float | None:
    """What making `units` of room in the cell costs; None where it cannot be made."""
    leaving = clearing(constructions, cell, units)
    if leaving is None:
        return None
    return sum(demolition_price(b.kind) or 0.0 for b in leaving)


def station_site_price(constructions: Constructions, cell: Cell) -> float | None:
    """What clearing a cell for a station costs: every building there, the track giving way for
    free; None where one cannot be taken down."""
    track = sum(piece.space for piece in constructions.track_at(cell))
    room = constructions.space_left(cell) + track
    here = [b for b in constructions.at(cell) if isinstance(b, Building)]
    if room >= CELL_SPACE:
        return 0.0
    if any(b.kind in UNIQUE for b in here):
        return None
    return sum(demolition_price(b.kind) or 0.0 for b in here)


def bulldoze_price(constructions: Constructions, cell: Cell) -> float | None:
    """What bulldozing a cell costs: every building, piece of track and station there, one-off
    buildings left standing; None when there is nothing it can take down."""
    tile = constructions.tile_map.tile(cell)
    total, found = 0.0, False
    for built in constructions.at(cell):
        if isinstance(built, Building):
            price = demolition_price(built.kind)
            if price is None:
                continue
            total += price
        elif isinstance(built, Station):
            total += STATION_REMOVAL
        else:
            total += track_price(built, tile) * TRACK_REMOVAL_SHARE
        found = True
    return total if found else None


def bulldoze(growth: Growth, cell: Cell) -> set[Cell]:
    """Takes down everything in the cell that can be, buildings the player owns too; returns
    the cells that changed."""
    constructions = growth.constructions
    taken = False
    for built in constructions.at(cell):
        if isinstance(built, Building):
            if built.kind in UNIQUE:
                continue
            growth.take_down(built)
        else:
            constructions.remove(built)
        taken = True
    return {cell} if taken else set()
