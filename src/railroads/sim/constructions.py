"""Things the player can build: track, stations and buildings.

Every cell has room for CELL_SPACE units. A single track takes one unit and a double track two,
and a station fills its whole cell. Most buildings take one unit, so a cell with a double track
through it still fits four houses. Farms, orchards and other fields fill a whole cell, as do the
biggest works and the novelties, and factories, canneries and mills take three.
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
    TENEMENT = "tenement"
    FLATS = "block of flats"
    MINE = "mine"
    FACTORY = "factory"
    BIG_FACTORY = "big factory"
    MEGA_FACTORY = "mega factory"
    SKYSCRAPER = "skyscraper"
    ORCHARD = "orchard"
    DATE_PALMS = "date palms"
    ORANGE_FARM = "orange farm"
    BANANAS = "banana plantation"
    SUGAR = "sugar field"
    FOOD_PLANT = "cannery"
    BAKERY = "bakery"
    FISHERY = "fishery"
    SHEEP_FARM = "sheep farm"
    COTTON_FIELD = "cotton field"
    VINEYARD = "vineyard"
    COFFEE_PLANTATION = "coffee plantation"
    RUBBER_PLANTATION = "rubber plantation"
    IRON_MINE = "iron mine"
    COPPER_MINE = "copper mine"
    GOLD_MINE = "gold mine"
    DIAMOND_MINE = "diamond mine"
    OIL_WELL = "oil well"
    SALT_WORKS = "salt works"
    TEXTILE_MILL = "textile mill"
    FURNITURE_MAKER = "furniture maker"
    TOOL_WORKS = "tool works"
    WINERY = "winery"
    BARREL_MAKER = "barrel maker"
    BEEKEEPER = "beekeeper"
    INCENSE_GROVE = "incense grove"
    CHURCH = "church"
    BASILICA = "basilica"
    CATHEDRAL = "cathedral"
    ST_PAULS = "St. Paul cathedral"
    REFINERY = "refinery"
    CHEMICAL_WORKS = "chemical works"
    ELECTRONICS_PLANT = "electronics plant"
    CAR_FACTORY = "car factory"
    CAR_DEALER = "car dealer"
    JEWELLER = "jeweller"
    OUTPOST = "military outpost"
    BASE = "military base"
    HEADQUARTERS = "headquarters"
    GUN_FACTORY = "gun factory"
    TANK_FACTORY = "tank factory"
    PLANE_FACTORY = "aircraft factory"
    AIRPORT = "airport"
    STEEL_MILL = "steel mill"
    SHIPYARD = "shipyard"
    LUMBER_MILL = "lumber mill"
    PORT = "port"
    CLAY_PIT = "clay pit"
    LIMESTONE_QUARRY = "limestone quarry"
    SAND_PIT = "sand pit"
    POTATO_FARM = "potato farm"
    CATTLE_RANCH = "cattle ranch"
    PIG_FARM = "pig farm"
    CHICKEN_FARM = "chicken farm"
    FLAX_FIELD = "flax field"
    HOP_GARDEN = "hop garden"
    MACHINE_WORKS = "machine works"
    BRICKWORKS = "brickworks"
    CEMENT_WORKS = "cement works"
    GLASSWORKS = "glassworks"
    BREWERY = "brewery"
    DISTILLERY = "distillery"
    TANNERY = "tannery"
    SHOE_FACTORY = "shoe factory"
    PLASTICS_WORKS = "plastics works"
    APPLIANCE_FACTORY = "appliance factory"
    PAPER_MILL = "paper mill"
    PRINTING_WORKS = "printing works"
    SUGAR_REFINERY = "sugar refinery"
    PORCELAIN_WORKS = "porcelain works"
    MEAT_FACTORY = "butcher"
    POWER_PLANT = "power plant"
    SPACE_CENTER = "space center"
    PEAR_HQ = "Pear Company headquarters"
    ROYAL_PALACE = "royal palace"
    SALT_CATHEDRAL = "salt mine cathedral"
    SPA_RESORT = "spa resort"
    WORLDS_FAIR = "World's Fair hall"
    FILM_STUDIO = "film studio"
    AMUSEMENT_PARK = "amusement park"
    OLYMPIC_STADIUM = "Olympic stadium"
    TAVERN = "tavern"
    DEPARTMENT_STORE = "department store"
    HOTEL = "hotel"

    @classmethod
    def _missing_(cls, value: object) -> BuildingKind | None:
        """Kinds by the names they had before, so older saves still load."""
        return {"meat factory": cls.MEAT_FACTORY}.get(str(value))


@dataclass(frozen=True, slots=True)
class BuildingRules:
    relief: frozenset[Relief]
    by_water: bool = False
    space: int = 1


_OFF_MOUNTAINS = frozenset({Relief.FLAT, Relief.HILLS})

BUILDING_RULES: dict[BuildingKind, BuildingRules] = {
    BuildingKind.FARM: BuildingRules(relief=_OFF_MOUNTAINS, space=CELL_SPACE),
    BuildingKind.HOUSE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.TENEMENT: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.FLATS: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.MINE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.FACTORY: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.BIG_FACTORY: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.MEGA_FACTORY: BuildingRules(relief=_OFF_MOUNTAINS, space=CELL_SPACE),
    BuildingKind.SKYSCRAPER: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.LUMBER_MILL: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.PORT: BuildingRules(relief=_OFF_MOUNTAINS, by_water=True),
    **{
        kind: BuildingRules(relief=_OFF_MOUNTAINS, space=CELL_SPACE)
        for kind in (
            BuildingKind.ORCHARD,
            BuildingKind.DATE_PALMS,
            BuildingKind.BANANAS,
            BuildingKind.SUGAR,
            BuildingKind.ORANGE_FARM,
            BuildingKind.SHEEP_FARM,
            BuildingKind.COTTON_FIELD,
            BuildingKind.VINEYARD,
            BuildingKind.COFFEE_PLANTATION,
            BuildingKind.RUBBER_PLANTATION,
            BuildingKind.POTATO_FARM,
            BuildingKind.CATTLE_RANCH,
            BuildingKind.PIG_FARM,
            BuildingKind.CHICKEN_FARM,
            BuildingKind.FLAX_FIELD,
            BuildingKind.HOP_GARDEN,
        )
    },
    BuildingKind.FOOD_PLANT: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.BAKERY: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.TAVERN: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.DEPARTMENT_STORE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.HOTEL: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.LIMESTONE_QUARRY: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.STEEL_MILL: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    **{
        kind: BuildingRules(relief=_OFF_MOUNTAINS, space=CELL_SPACE)
        for kind in (
            BuildingKind.POWER_PLANT,
            BuildingKind.REFINERY,
            BuildingKind.CAR_FACTORY,
            BuildingKind.TANK_FACTORY,
            BuildingKind.PLANE_FACTORY,
            BuildingKind.SPACE_CENTER,
            BuildingKind.PEAR_HQ,
            BuildingKind.ROYAL_PALACE,
            BuildingKind.SALT_CATHEDRAL,
            BuildingKind.SPA_RESORT,
            BuildingKind.WORLDS_FAIR,
            BuildingKind.FILM_STUDIO,
            BuildingKind.AMUSEMENT_PARK,
            BuildingKind.OLYMPIC_STADIUM,
        )
    },
    BuildingKind.TEXTILE_MILL: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.SHIPYARD: BuildingRules(relief=_OFF_MOUNTAINS, by_water=True, space=CELL_SPACE),
    BuildingKind.IRON_MINE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.COPPER_MINE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.GOLD_MINE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.DIAMOND_MINE: BuildingRules(relief=frozenset(Relief)),
    **{
        kind: BuildingRules(relief=_OFF_MOUNTAINS)
        for kind in (
            BuildingKind.FISHERY,
            BuildingKind.OIL_WELL,
            BuildingKind.SALT_WORKS,
            BuildingKind.FURNITURE_MAKER,
            BuildingKind.TOOL_WORKS,
            BuildingKind.WINERY,
            BuildingKind.CHEMICAL_WORKS,
            BuildingKind.ELECTRONICS_PLANT,
            BuildingKind.CAR_DEALER,
            BuildingKind.JEWELLER,
            BuildingKind.OUTPOST,
            BuildingKind.BASE,
            BuildingKind.HEADQUARTERS,
            BuildingKind.GUN_FACTORY,
            BuildingKind.AIRPORT,
            BuildingKind.BARREL_MAKER,
            BuildingKind.BEEKEEPER,
            BuildingKind.INCENSE_GROVE,
            BuildingKind.CHURCH,
            BuildingKind.BASILICA,
            BuildingKind.CATHEDRAL,
            BuildingKind.ST_PAULS,
            BuildingKind.CLAY_PIT,
            BuildingKind.SAND_PIT,
            BuildingKind.MACHINE_WORKS,
            BuildingKind.BRICKWORKS,
            BuildingKind.CEMENT_WORKS,
            BuildingKind.GLASSWORKS,
            BuildingKind.BREWERY,
            BuildingKind.DISTILLERY,
            BuildingKind.TANNERY,
            BuildingKind.SHOE_FACTORY,
            BuildingKind.PLASTICS_WORKS,
            BuildingKind.APPLIANCE_FACTORY,
            BuildingKind.PAPER_MILL,
            BuildingKind.PRINTING_WORKS,
            BuildingKind.SUGAR_REFINERY,
            BuildingKind.PORCELAIN_WORKS,
            BuildingKind.MEAT_FACTORY,
        )
    },
}
STATION_RELIEF = _OFF_MOUNTAINS


@dataclass(frozen=True, slots=True)
class Track:
    """A piece of track inside one cell, between two of its track points; it joins the pieces
    in the cells those points lead to (see `Grid.track_exit`). Electric track has overhead
    wires along it, which electric engines need."""

    cell: Cell
    ends: tuple[int, int]
    width: TrackWidth = TrackWidth.SINGLE
    bed: TrackBed = TrackBed.GROUND
    electric: bool = False

    def __post_init__(self) -> None:
        start, end = sorted(self.ends)
        object.__setattr__(self, "ends", (start, end))

    @property
    def space(self) -> int:
        return int(self.width)


@dataclass(frozen=True, slots=True)
class Station:
    """Fills its whole cell. Its own track runs straight through the cell between `ends`, two
    opposite track points, and other track joins it only there: its way in and its way out. It
    keeps that way however track is laid round it later; with `ends` None its track runs
    across, from the first track point to the one opposite (Constructions.station_ends)."""

    cell: Cell
    ends: tuple[int, int] | None = None

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
        # Where the stations stand and track runs, kept apart so the railway never looks through
        # every building for them, and a count that goes up whenever either changes.
        self._stations: set[Cell] = set()
        self._track: dict[Cell, int] = {}
        self.railway_version = 0

    def __iter__(self) -> Iterator[Construction]:
        for built in self._cells.values():
            yield from built

    def cells(self) -> Iterator[tuple[Cell, list[Construction]]]:
        """Each cell with something built on it, and what is built there."""
        return iter(self._cells.items())

    def stations(self) -> list[Cell]:
        """Every station's cell, in order."""
        return sorted(self._stations)

    def station_at(self, cell: Cell) -> Station | None:
        """The station in the cell, if there is one."""
        return next((b for b in self._cells.get(cell, ()) if isinstance(b, Station)), None)

    def station_ends(self, cell: Cell) -> tuple[int, int]:
        """The two track points the station in the cell joins track by, in order."""
        station = self.station_at(cell)
        if station is None:
            raise ValueError(f"There is no station at {cell}")
        if station.ends is not None:
            return station.ends
        return 0, len(self.tile_map.grid.track_points()) // 2

    def _way_problem(self, ends: tuple[int, int]) -> str | None:
        grid = self.tile_map.grid
        count = len(grid.track_points())
        start, end = ends
        if start not in grid.track_ends() or end not in grid.track_ends():
            return "A station's track runs across edges, or straight up and down through corners"
        if (start + count // 2) % count != end:
            return "A station's track runs straight through it, between opposite sides"
        return None

    def track(self) -> list[Track]:
        """Every piece of track."""
        return [piece for cell in self._track for piece in self.track_at(cell)]

    def at(self, cell: Cell) -> tuple[Construction, ...]:
        return tuple(self._cells.get(cell, ()))

    def track_at(self, cell: Cell) -> tuple[Track, ...]:
        return tuple(built for built in self._cells.get(cell, ()) if isinstance(built, Track))

    def track_between(self, cell: Cell, ends: tuple[int, int]) -> Track | None:
        """The piece of track in the cell between these two track points, if there is one."""
        key = tuple(sorted(ends))
        return next((piece for piece in self.track_at(cell) if piece.ends == key), None)

    def electrified(self, cell: Cell, ends: tuple[int, int] | None = None) -> bool:
        """Whether track in the cell is electric: the piece between `ends`, or with no ends
        given, every piece there. False where there is no such track."""
        if ends is not None:
            piece = self.track_between(cell, ends)
            return piece is not None and piece.electric
        pieces = self.track_at(cell)
        return bool(pieces) and all(piece.electric for piece in pieces)

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
                if problem is None and construction.ends is not None:
                    problem = self._way_problem(construction.ends)
            case Building():
                problem = self._building_problem(construction)
        if problem is None and construction.space > self.space_left(cell):
            problem = f"Not enough room: needs {construction.space}, {self.space_left(cell)} left"
        return problem

    def build(self, construction: Construction) -> None:
        if (problem := self.why_not(construction)) is not None:
            raise CannotBuild(problem)
        self._cells.setdefault(construction.cell, []).append(construction)
        self._index(construction, 1)

    def remove(self, construction: Construction) -> None:
        built = self._cells.get(construction.cell, [])
        if construction not in built:
            raise ValueError(f"{construction} is not built")
        built.remove(construction)
        if not built:
            del self._cells[construction.cell]
        self._index(construction, -1)

    def _index(self, construction: Construction, change: int) -> None:
        match construction:
            case Station():
                if change > 0:
                    self._stations.add(construction.cell)
                else:
                    self._stations.discard(construction.cell)
            case Track():
                left = self._track.get(construction.cell, 0) + change
                if left > 0:
                    self._track[construction.cell] = left
                else:
                    self._track.pop(construction.cell, None)
            case _:
                return
        self.railway_version += 1

    def track_problem(self, track: Track) -> str | None:
        """Why this track cannot be laid, leaving aside the room already taken."""
        return self._track_problem(track)

    def site_problem(self, building: Building) -> str | None:
        """Why the building cannot stand in its cell, leaving aside the room already taken."""
        return self._building_problem(building)

    def _track_problem(self, track: Track) -> str | None:
        grid = self.tile_map.grid
        count = len(grid.track_points())
        start, end = track.ends
        if not 0 <= start < end < count:
            return f"Track joins two different track points, numbered 0 to {count - 1}"
        if start not in (ends := grid.track_ends()) or end not in ends:
            return "Track runs across edges, or straight up and down through corners"
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
        if self.track_between(track.cell, track.ends) is not None:
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
