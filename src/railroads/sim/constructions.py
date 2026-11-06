"""Things the player can build: track, stations and buildings.

Every cell has room for CELL_SPACE units. A single track takes one unit and a double track two,
a station fills its whole cell and each building takes the room its kind needs, so a cell with a
double track through it still fits four houses.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, IntEnum

from railroads.sim.grids import Cell
from railroads.sim.tilemap import Relief, Terrain, Tile, TileMap
from railroads.sim.track_rules import TrackBed, track_rules_for

CELL_SPACE = 6
MAX_TURN_DEG = 60.0


class TrackWidth(IntEnum):
    """How many tracks run side by side, which is also how many units of room they take."""

    SINGLE = 1
    DOUBLE = 2


class BuildingKind(Enum):
    FARM = "farm"
    HOUSE = "house"
    MINE = "mine"
    FACTORY = "factory"
    LUMBER_MILL = "lumber mill"
    PORT = "port"


@dataclass(frozen=True, slots=True)
class BuildingRules:
    space: int
    relief: frozenset[Relief]
    by_water: bool = False


_OFF_MOUNTAINS = frozenset({Relief.FLAT, Relief.HILLS})

BUILDING_RULES: dict[BuildingKind, BuildingRules] = {
    BuildingKind.FARM: BuildingRules(space=3, relief=_OFF_MOUNTAINS),
    BuildingKind.HOUSE: BuildingRules(space=1, relief=_OFF_MOUNTAINS),
    BuildingKind.MINE: BuildingRules(space=2, relief=frozenset(Relief)),
    BuildingKind.FACTORY: BuildingRules(space=3, relief=_OFF_MOUNTAINS),
    BuildingKind.LUMBER_MILL: BuildingRules(space=2, relief=_OFF_MOUNTAINS),
    BuildingKind.PORT: BuildingRules(space=3, relief=_OFF_MOUNTAINS, by_water=True),
}
STATION_RELIEF = _OFF_MOUNTAINS


@dataclass(frozen=True, slots=True)
class Track:
    """A piece of track inside one cell, between two of its track points; it joins the pieces
    in the cells those points lead to (see `Grid.track_exit`)."""

    cell: Cell
    ends: tuple[int, int]
    width: TrackWidth = TrackWidth.SINGLE
    bed: TrackBed = TrackBed.GROUND

    def __post_init__(self) -> None:
        start, end = sorted(self.ends)
        object.__setattr__(self, "ends", (start, end))

    @property
    def space(self) -> int:
        return int(self.width)


@dataclass(frozen=True, slots=True)
class Station:
    """Fills its whole cell; track joins it at any of the cell's track points."""

    cell: Cell

    @property
    def space(self) -> int:
        return CELL_SPACE


@dataclass(frozen=True, slots=True)
class Building:
    cell: Cell
    kind: BuildingKind

    @property
    def space(self) -> int:
        return BUILDING_RULES[self.kind].space


type Construction = Track | Station | Building


class CannotBuild(ValueError):
    pass


class Constructions:
    """Everything built on one map, kept per cell, and the rules for building more."""

    def __init__(self, tile_map: TileMap) -> None:
        self.tile_map = tile_map
        self._cells: dict[Cell, list[Construction]] = {}

    def __iter__(self) -> Iterator[Construction]:
        for built in self._cells.values():
            yield from built

    def at(self, cell: Cell) -> tuple[Construction, ...]:
        return tuple(self._cells.get(cell, ()))

    def space_left(self, cell: Cell) -> int:
        return CELL_SPACE - sum(built.space for built in self._cells.get(cell, ()))

    def why_not(self, construction: Construction) -> str | None:
        """Why this cannot be built, or None when it can."""
        cell = construction.cell
        if not self.tile_map.grid.contains(*cell):
            return "That is off the map"
        match construction:
            case Track():
                problem = self._track_problem(construction)
            case Station():
                problem = _site_problem(self.tile_map.tile(cell), STATION_RELIEF, "station")
            case Building():
                problem = self._building_problem(construction)
        if problem is None and construction.space > self.space_left(cell):
            problem = f"Not enough room: needs {construction.space}, {self.space_left(cell)} left"
        return problem

    def build(self, construction: Construction) -> None:
        if (problem := self.why_not(construction)) is not None:
            raise CannotBuild(problem)
        self._cells.setdefault(construction.cell, []).append(construction)

    def remove(self, construction: Construction) -> None:
        built = self._cells.get(construction.cell, [])
        if construction not in built:
            raise ValueError(f"{construction} is not built")
        built.remove(construction)
        if not built:
            del self._cells[construction.cell]

    def _track_problem(self, track: Track) -> str | None:
        grid = self.tile_map.grid
        count = len(grid.track_points())
        start, end = track.ends
        if not 0 <= start < end < count:
            return f"Track joins two different track points, numbered 0 to {count - 1}"
        if grid.track_turn(start, end) > MAX_TURN_DEG + 1e-6:
            return f"Track can turn at most {MAX_TURN_DEG:g} degrees in one cell"
        if (problem := self._bed_problem(track.cell, track.bed)) is not None:
            return problem
        for point in track.ends:
            if not grid.contains(*grid.track_exit(track.cell, point)[0]):
                return "Track cannot run off the map"
            for side in grid.track_flanks(track.cell, point):
                if self._bed_problem(side, track.bed) is not None:
                    return "Track through a corner needs the cells on both sides to take it"
        laid = (built for built in self.at(track.cell) if isinstance(built, Track))
        if any(other.ends == track.ends for other in laid):
            return "There is track between those points already"
        return None

    def _bed_problem(self, cell: Cell, bed: TrackBed) -> str | None:
        if not self.tile_map.grid.contains(*cell):
            return "That is off the map"
        tile = self.tile_map.tile(cell)
        if track_rules_for(tile, bed).can_build:
            return None
        if bed == TrackBed.BRIDGE:
            return "Bridges go over rivers and lakes"
        if bed == TrackBed.TUNNEL:
            return "Tunnels go through mountains"
        if tile.terrain == Terrain.SEA:
            return "Track cannot be laid in the sea"
        if tile.terrain != Terrain.LAND:
            return "Track over water needs a bridge"
        return "Track through mountains needs a tunnel"

    def _building_problem(self, building: Building) -> str | None:
        rules = BUILDING_RULES[building.kind]
        name = building.kind.value
        if problem := _site_problem(self.tile_map.tile(building.cell), rules.relief, name):
            return problem
        if rules.by_water and not self._by_water(building.cell):
            return f"A {name} needs water next to it"
        return None

    def _by_water(self, cell: Cell) -> bool:
        terrain = self.tile_map.terrain
        return any(terrain[other] != Terrain.LAND for other in self.tile_map.grid.neighbours(*cell))


def _site_problem(tile: Tile, relief: frozenset[Relief], name: str) -> str | None:
    if tile.terrain != Terrain.LAND:
        return f"A {name} cannot be built on water"
    if tile.relief not in relief:
        return f"A {name} cannot be built on {tile.relief.name.lower()}"
    return None
