"""Things the player can build: track, stations, the facilities beside stations and buildings.

Every cell has room for CELL_SPACE units. A single track takes one unit and a double track two,
and a station or an engine shed fills its whole cell, and cattle pens take ANNEX_SPACE of it. Most buildings take one unit, so a cell with a double track
through it still fits four houses. Farms, orchards and other fields fill a whole cell, as do the
biggest works and the novelties, and factories, canneries and mills take three.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import IntEnum

from railroads.sim.grids import Cell
from railroads.sim.keyed import Keyed
from railroads.sim.move import Move
from railroads.sim.tilemap import Relief, Terrain, Tile, TileMap
from railroads.sim.track_rules import BRIDGES, Bridge, TrackBed, carries, track_rules_for

CELL_SPACE = 6
MAX_TURN_DEG = 60.0
# How far past MAX_TURN_DEG a turn worked out in floats may go and still count as within it.
TURN_SLACK_DEG = 1e-6


# How many of the railway's last changes Constructions keeps the cells of.
RAILWAY_LOG = 512


class TrackWidth(IntEnum):
    """How many tracks run side by side, which is also how many units of room they take."""

    SINGLE = 1
    DOUBLE = 2
    QUADRUPLE = 4


class BuildingKind(Keyed):
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
    ST_PETERS = "St. Peter's Basilica"
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
    LOGGING_CAMP = "logging camp"
    SAWMILL = "sawmill"
    PORT = "port"
    BIG_PORT = "big port"
    MEGA_PORT = "mega port"
    CLAY_PIT = "clay pit"
    LIMESTONE_QUARRY = "limestone quarry"
    SAND_PIT = "sand pit"
    POTATO_FARM = "potato farm"
    RICE_PADDY = "rice paddy"
    SOY_FIELD = "soy field"
    TEA_PLANTATION = "tea plantation"
    SOY_SAUCE_BREWERY = "soy sauce brewery"
    TOFU_MAKER = "tofu maker"
    MISO_MAKER = "miso maker"
    FISH_DRYER = "fish drying shed"
    CATTLE_RANCH = "cattle ranch"
    PIG_FARM = "pig farm"
    CHICKEN_FARM = "chicken farm"
    GASWORKS = "gasworks"
    POTTER = "potter"
    TILE_WORKS = "tile works"
    FOUNDRY = "foundry"
    DAIRY = "dairy"
    SAUSAGE_MAKER = "sausage maker"
    CARPET_WEAVER = "carpet weaver"
    CLOCK_MAKER = "clock maker"
    MIRROR_MAKER = "mirror maker"
    MANOR = "manor"
    THEATRE = "theatre"
    HOSPITAL = "hospital"
    RAILWAY_WORKS = "railway works"
    FLAX_FIELD = "flax field"
    HOP_GARDEN = "hop garden"
    MACHINE_WORKS = "machine works"
    BRICKWORKS = "brickworks"
    CEMENT_WORKS = "cement works"
    GLASSWORKS = "glassworks"
    BREWERY = "brewery"
    DISTILLERY = "distillery"
    SAKE_BREWERY = "sake brewery"
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
    DIESEL_POWER_PLANT = "diesel power plant"
    GAS_POWER_PLANT = "gas power plant"
    SPACE_CENTER = "space center"
    PEAR_HQ = "Pear Company headquarters"
    GIGAFACTORY = "battery gigafactory"
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
    CLUB = "club"
    SUSHI_HUT = "sushi hut"
    BUILDERS_YARD = "builder's yard"
    FILLING_STATION = "filling station"
    JUICE_WORKS = "juice works"
    BIG_JUICE_WORKS = "big juice works"
    MEGA_JUICE_WORKS = "mega juice works"
    BIG_STEEL_MILL = "big steel mill"
    MEGA_STEEL_MILL = "mega steel mill"
    BIG_POTTER = "big potter"
    MEGA_POTTER = "mega potter"
    BIG_TILE_WORKS = "big tile works"
    MEGA_TILE_WORKS = "mega tile works"
    BIG_FOUNDRY = "big foundry"
    MEGA_FOUNDRY = "mega foundry"
    BIG_DAIRY = "big dairy"
    MEGA_DAIRY = "mega dairy"
    BIG_SAUSAGE_MAKER = "big sausage maker"
    MEGA_SAUSAGE_MAKER = "mega sausage maker"
    BIG_CARPET_WEAVER = "big carpet weaver"
    MEGA_CARPET_WEAVER = "mega carpet weaver"
    BIG_CLOCK_MAKER = "big clock maker"
    MEGA_CLOCK_MAKER = "mega clock maker"
    BIG_MIRROR_MAKER = "big mirror maker"
    MEGA_MIRROR_MAKER = "mega mirror maker"
    BIG_GUN_FACTORY = "big gun factory"
    MEGA_GUN_FACTORY = "mega gun factory"
    BIG_TANK_FACTORY = "big tank factory"
    MEGA_TANK_FACTORY = "mega tank factory"
    BIG_PLANE_FACTORY = "big aircraft factory"
    MEGA_PLANE_FACTORY = "mega aircraft factory"
    BIG_FOOD_PLANT = "big cannery"
    MEGA_FOOD_PLANT = "mega cannery"
    BIG_BAKERY = "big bakery"
    MEGA_BAKERY = "mega bakery"
    BIG_TEXTILE_MILL = "big textile mill"
    MEGA_TEXTILE_MILL = "mega textile mill"
    BIG_FURNITURE_MAKER = "big furniture maker"
    MEGA_FURNITURE_MAKER = "mega furniture maker"
    BIG_TOOL_WORKS = "big tool works"
    MEGA_TOOL_WORKS = "mega tool works"
    BIG_BARREL_MAKER = "big barrel maker"
    MEGA_BARREL_MAKER = "mega barrel maker"
    BIG_WINERY = "big winery"
    MEGA_WINERY = "mega winery"
    BIG_REFINERY = "big refinery"
    MEGA_REFINERY = "mega refinery"
    BIG_CHEMICAL_WORKS = "big chemical works"
    MEGA_CHEMICAL_WORKS = "mega chemical works"
    BIG_ELECTRONICS_PLANT = "big electronics plant"
    MEGA_ELECTRONICS_PLANT = "mega electronics plant"
    BIG_CAR_FACTORY = "big car factory"
    MEGA_CAR_FACTORY = "mega car factory"
    BIG_MACHINE_WORKS = "big machine works"
    MEGA_MACHINE_WORKS = "mega machine works"
    BIG_BRICKWORKS = "big brickworks"
    MEGA_BRICKWORKS = "mega brickworks"
    BIG_CEMENT_WORKS = "big cement works"
    MEGA_CEMENT_WORKS = "mega cement works"
    BIG_GLASSWORKS = "big glassworks"
    MEGA_GLASSWORKS = "mega glassworks"
    BIG_BREWERY = "big brewery"
    MEGA_BREWERY = "mega brewery"
    BIG_DISTILLERY = "big distillery"
    MEGA_DISTILLERY = "mega distillery"
    BIG_SAKE_BREWERY = "big sake brewery"
    MEGA_SAKE_BREWERY = "mega sake brewery"
    BIG_SOY_SAUCE_BREWERY = "big soy sauce brewery"
    MEGA_SOY_SAUCE_BREWERY = "mega soy sauce brewery"
    BIG_TOFU_MAKER = "big tofu maker"
    MEGA_TOFU_MAKER = "mega tofu maker"
    BIG_MISO_MAKER = "big miso maker"
    MEGA_MISO_MAKER = "mega miso maker"
    BIG_FISH_DRYER = "big fish drying shed"
    MEGA_FISH_DRYER = "mega fish drying shed"
    BIG_TANNERY = "big tannery"
    MEGA_TANNERY = "mega tannery"
    BIG_MEAT_FACTORY = "big butcher"
    MEGA_MEAT_FACTORY = "mega butcher"
    BIG_SHOE_FACTORY = "big shoe factory"
    MEGA_SHOE_FACTORY = "mega shoe factory"
    BIG_PLASTICS_WORKS = "big plastics works"
    MEGA_PLASTICS_WORKS = "mega plastics works"
    BIG_APPLIANCE_FACTORY = "big appliance factory"
    MEGA_APPLIANCE_FACTORY = "mega appliance factory"
    BIG_PAPER_MILL = "big paper mill"
    MEGA_PAPER_MILL = "mega paper mill"
    BIG_PRINTING_WORKS = "big printing works"
    MEGA_PRINTING_WORKS = "mega printing works"
    BIG_SUGAR_REFINERY = "big sugar refinery"
    MEGA_SUGAR_REFINERY = "mega sugar refinery"
    BIG_GASWORKS = "big gasworks"
    MEGA_GASWORKS = "mega gasworks"
    BIG_PORCELAIN_WORKS = "big porcelain works"
    MEGA_PORCELAIN_WORKS = "mega porcelain works"

    @classmethod
    def _missing_(cls, value: object) -> BuildingKind | None:
        """Kinds by the names they had before, so older saves still load."""
        return {"meat factory": cls.MEAT_FACTORY}.get(str(value))


# The buildings a map gets only one of, or a few: novelties, the Olympic stadiums, the railway
# works and St Peter's.
ONE_OFFS = frozenset(
    {
        BuildingKind.SPACE_CENTER,
        BuildingKind.PEAR_HQ,
        BuildingKind.GIGAFACTORY,
        BuildingKind.ROYAL_PALACE,
        BuildingKind.SALT_CATHEDRAL,
        BuildingKind.SPA_RESORT,
        BuildingKind.WORLDS_FAIR,
        BuildingKind.FILM_STUDIO,
        BuildingKind.AMUSEMENT_PARK,
        BuildingKind.OLYMPIC_STADIUM,
        BuildingKind.RAILWAY_WORKS,
        BuildingKind.ST_PETERS,
    }
)


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
    BuildingKind.LOGGING_CAMP: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.SAWMILL: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.GASWORKS: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.TILE_WORKS: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.FOUNDRY: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.MANOR: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.THEATRE: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.HOSPITAL: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    **{
        kind: BuildingRules(relief=_OFF_MOUNTAINS)
        for kind in (
            BuildingKind.POTTER,
            BuildingKind.DAIRY,
            BuildingKind.SAUSAGE_MAKER,
            BuildingKind.TOFU_MAKER,
            BuildingKind.MISO_MAKER,
            BuildingKind.FISH_DRYER,
            BuildingKind.CARPET_WEAVER,
            BuildingKind.CLOCK_MAKER,
            BuildingKind.MIRROR_MAKER,
        )
    },
    # Ports fill their cell with their yard and quay (railroads.ui.pixel.art.ports).
    BuildingKind.PORT: BuildingRules(relief=_OFF_MOUNTAINS, by_water=True, space=CELL_SPACE),
    BuildingKind.BIG_PORT: BuildingRules(relief=_OFF_MOUNTAINS, by_water=True, space=CELL_SPACE),
    BuildingKind.MEGA_PORT: BuildingRules(relief=_OFF_MOUNTAINS, by_water=True, space=CELL_SPACE),
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
            BuildingKind.RICE_PADDY,
            BuildingKind.SOY_FIELD,
            BuildingKind.TEA_PLANTATION,
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
    BuildingKind.CLUB: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.SUSHI_HUT: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.BUILDERS_YARD: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.FILLING_STATION: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.DEPARTMENT_STORE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.HOTEL: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.LIMESTONE_QUARRY: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.STEEL_MILL: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    **{
        kind: BuildingRules(relief=_OFF_MOUNTAINS, space=CELL_SPACE)
        for kind in (
            BuildingKind.POWER_PLANT,
            BuildingKind.DIESEL_POWER_PLANT,
            BuildingKind.GAS_POWER_PLANT,
            BuildingKind.REFINERY,
            BuildingKind.CAR_FACTORY,
            BuildingKind.TANK_FACTORY,
            BuildingKind.PLANE_FACTORY,
            BuildingKind.SPACE_CENTER,
            BuildingKind.PEAR_HQ,
            BuildingKind.GIGAFACTORY,
            BuildingKind.ROYAL_PALACE,
            BuildingKind.SALT_CATHEDRAL,
            BuildingKind.SPA_RESORT,
            BuildingKind.WORLDS_FAIR,
            BuildingKind.FILM_STUDIO,
            BuildingKind.AMUSEMENT_PARK,
            BuildingKind.OLYMPIC_STADIUM,
            BuildingKind.RAILWAY_WORKS,
        )
    },
    BuildingKind.TEXTILE_MILL: BuildingRules(relief=_OFF_MOUNTAINS, space=3),
    BuildingKind.SHIPYARD: BuildingRules(relief=_OFF_MOUNTAINS, by_water=True, space=CELL_SPACE),
    BuildingKind.IRON_MINE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.COPPER_MINE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.GOLD_MINE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.DIAMOND_MINE: BuildingRules(relief=frozenset(Relief)),
    BuildingKind.SALT_WORKS: BuildingRules(relief=frozenset(Relief)),
    **{
        kind: BuildingRules(relief=_OFF_MOUNTAINS)
        for kind in (
            BuildingKind.FISHERY,
            BuildingKind.OIL_WELL,
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
            BuildingKind.ST_PETERS,
            BuildingKind.CLAY_PIT,
            BuildingKind.SAND_PIT,
            BuildingKind.MACHINE_WORKS,
            BuildingKind.BRICKWORKS,
            BuildingKind.CEMENT_WORKS,
            BuildingKind.GLASSWORKS,
            BuildingKind.BREWERY,
            BuildingKind.DISTILLERY,
            BuildingKind.SAKE_BREWERY,
            BuildingKind.SOY_SAUCE_BREWERY,
            BuildingKind.TANNERY,
            BuildingKind.SHOE_FACTORY,
            BuildingKind.PLASTICS_WORKS,
            BuildingKind.APPLIANCE_FACTORY,
            BuildingKind.PAPER_MILL,
            BuildingKind.PRINTING_WORKS,
            BuildingKind.SUGAR_REFINERY,
            BuildingKind.PORCELAIN_WORKS,
            BuildingKind.MEAT_FACTORY,
            BuildingKind.JUICE_WORKS,
        )
    },
}

# Works that merge as they gather: a few of a kind in a neighbourhood become its big one, and as
# many big ones its mega one (railroads.sim.goods.merging, railroads.sim.ladders). The factory line merges its own way.
MERGED: dict[BuildingKind, tuple[BuildingKind, BuildingKind]] = {
    BuildingKind[kind.name.removeprefix("BIG_")]: (kind, BuildingKind["MEGA_" + kind.name[4:]])
    for kind in BuildingKind
    if kind.name.startswith("BIG_")
    and kind not in (BuildingKind.BIG_FACTORY, BuildingKind.BIG_PORT)
}

# Ports by size: two of a size in a neighbourhood merge into one of the next (railroads.sim.ports).
PORT_KINDS = (BuildingKind.PORT, BuildingKind.BIG_PORT, BuildingKind.MEGA_PORT)

# What each big or mega works grew from.
GREW_FROM: dict[BuildingKind, BuildingKind] = {
    grown: kind for kind, kinds in MERGED.items() for grown in kinds
}

# How much of a cell big and mega works take at least: half of it, and two thirds.
BIG_SPACE = CELL_SPACE // 2
MEGA_SPACE = CELL_SPACE * 2 // 3

for _grown, _kind in GREW_FROM.items():
    _least = BIG_SPACE if _grown.name.startswith("BIG_") else MEGA_SPACE
    BUILDING_RULES[_grown] = replace(
        BUILDING_RULES[_kind], space=max(_least, BUILDING_RULES[_kind].space)
    )

STATION_RELIEF = _OFF_MOUNTAINS


@dataclass(frozen=True, slots=True)
class Track:
    """A piece of track inside one cell, between two of its track points; it joins the pieces
    in the cells those points lead to (see `Grid.track_exit`). Electric track has overhead
    wires along it, which electric engines need. Maglev track is a concrete guideway instead of
    rails: only maglev engines run on it, and they run on nothing else. Track on a bridge is
    carried by a bridge of its `bridge` kind; a bridge on land is a flyover over the track in its
    cell, which it never joins."""

    cell: Cell
    ends: tuple[int, int]
    width: TrackWidth = TrackWidth.SINGLE
    bed: TrackBed = TrackBed.GROUND
    electric: bool = False
    maglev: bool = False
    bridge: Bridge = Bridge.WOOD

    def __post_init__(self) -> None:
        start, end = sorted(self.ends)
        object.__setattr__(self, "ends", (start, end))

    @property
    def space(self) -> int:
        return int(self.width)

    def other_end(self, point: int) -> int:
        """The end of the piece away from `point`, one of its ends."""
        return self.ends[1] if self.ends[0] == point else self.ends[0]


@dataclass(frozen=True, slots=True)
class Station:
    """Fills its whole cell. Its own track runs straight through the cell between `ends`, two
    opposite track points, and other track joins it only there: its way in and its way out. The
    first track laid to it turns it (railroads.sim.routes); after that it keeps its way however
    track is laid round it; with `ends` None its track runs
    across, from the first track point to the one opposite (Constructions.station_ends). A station
    raised high enough gets more platforms in cells beside it, each a Station of its own with
    `main` the cell of the station it belongs to, and its own way through, so lines come in from
    more sides."""

    cell: Cell
    ends: tuple[int, int] | None = None
    main: Cell | None = None

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


class Facility(Keyed):
    """What the player can put up beside a station (railroads.sim.facilities)."""

    ENGINE_SHED = "engine shed"
    SAND_TOWER = "sand tower"
    POST_OFFICE = "post office"
    REFRESHMENT_ROOM = "refreshment room"
    MARKET_HALL = "market hall"
    CATTLE_PENS = "cattle pens"
    WAREHOUSE = "warehouse"


# The facilities that stand in a cell of their own beside their station, and how much room each
# takes there; the others stand in the station itself and take none.
ANNEX_SPACE = {Facility.ENGINE_SHED: CELL_SPACE}


@dataclass(frozen=True, slots=True)
class Annex:
    """A facility put up in a cell beside the station in `station`. An engine shed has track of
    its own, running into it between `ends` as a station's does (Constructions.station_ends):
    track laid to it joins it there, and trains waiting for a load wait in it."""

    cell: Cell
    facility: Facility
    station: Cell
    ends: tuple[int, int] | None = None
    # An engine shed's level, which gives its roundhouse more stalls (railroads.sim.sheds), and
    # whether a sand tower stands by its lead (railroads.sim.facilities).
    level: int = 1
    sand_tower: bool = False

    def __setstate__(self, state: tuple[object, ...]) -> None:
        """Takes up a pickled facility, one put up before engine sheds had track running across
        it, levels and sand towers too."""
        names = ("cell", "facility", "station", "ends", "level", "sand_tower")
        defaults = (None, 1, False)
        values = (*state, *defaults[len(state) - 3 :])
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)

    @property
    def space(self) -> int:
        return ANNEX_SPACE.get(self.facility, 0)


type Construction = Track | Station | Annex | Building


class CannotBuild(ValueError):
    pass


# Why nothing can be built on land the player has not bought (railroads.sim.lots).
LOCKED = "That land isn't yours: buy it first"


class Constructions:
    """Everything built on one map, kept per cell, and the rules for building more."""

    def __init__(self, tile_map: TileMap) -> None:
        self.tile_map = tile_map
        self._cells: dict[Cell, list[Construction]] = {}
        # Where the stations stand and track runs, kept apart so the railway never looks through
        # every building for them, and a count that goes up whenever either changes.
        self._stations: set[Cell] = set()
        # The platforms added to each station, by the station's cell.
        self._platforms: dict[Cell, set[Cell]] = {}
        # The facilities put up beside each station, by the station's cell, and where each stands.
        self._annexes: dict[Cell, dict[Facility, Cell]] = {}
        self._track: dict[Cell, int] = {}
        self.railway_version = 0
        # The cell each of the last few changes to the railway was in, with the railway_version
        # it brought, so what is kept about the track can be worked out again only round them.
        self._railway_log: list[tuple[int, Cell]] = []
        # A count that goes up whenever anything is built or removed.
        self.version = 0

    def __getstate__(self) -> dict[str, object]:
        """What a saved game keeps of what is built: each building as its kind alone, in its
        cell's list, which pickles many times faster than the buildings themselves."""
        state = dict(self.__dict__)
        state["_cells"] = {
            cell: [built.kind if type(built) is Building else built for built in here]
            for cell, here in self._cells.items()
        }
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._cells = {
            cell: [
                Building(cell, built) if type(built) is BuildingKind else built for built in here
            ]
            for cell, here in self._cells.items()
        }

    def __iter__(self) -> Iterator[Construction]:
        for built in self._cells.values():
            yield from built

    def cells(self) -> Iterator[tuple[Cell, list[Construction]]]:
        """Each cell with something built on it, and what is built there."""
        return iter(self._cells.items())

    def stations(self) -> list[Cell]:
        """Every station's cell, in order."""
        return sorted(self._stations)

    def platforms(self, station: Cell) -> list[Cell]:
        """The platforms added to a station, in order, without its own cell."""
        return sorted(self._platforms.get(station, ()))

    def station_cells(self, station: Cell) -> list[Cell]:
        """Every cell of a station: its own, then its added platforms."""
        return [station, *self.platforms(station)]

    def main_station(self, cell: Cell) -> Cell | None:
        """The station a cell belongs to, as the station itself or one of its platforms."""
        station = self.station_at(cell)
        if station is None:
            return None
        return station.cell if station.main is None else station.main

    def annexes(self, station: Cell) -> dict[Facility, Cell]:
        """The facilities put up beside a station, and the cell each stands in."""
        return dict(self._annexes.get(station, {}))

    def annex_at(self, cell: Cell) -> Annex | None:
        """The facility standing in the cell, if there is one."""
        return next((b for b in self._cells.get(cell, ()) if isinstance(b, Annex)), None)

    def every_annex(self) -> list[Annex]:
        """Every facility put up beside a station, in order of their cells."""
        found = (self.annex_at(cell) for kinds in self._annexes.values() for cell in kinds.values())
        return sorted((a for a in found if a is not None), key=lambda a: a.cell)

    def shed_at(self, cell: Cell) -> Annex | None:
        """The engine shed standing in the cell, if there is one."""
        annex = self.annex_at(cell)
        return annex if annex is not None and annex.facility == Facility.ENGINE_SHED else None

    def shed_of(self, station: Cell) -> Cell | None:
        """Where a station's engine shed stands, if it has one."""
        return self._annexes.get(station, {}).get(Facility.ENGINE_SHED)

    def every_shed_cell(self) -> list[Cell]:
        """Every cell an engine shed stands in."""
        return sorted(
            cell
            for kinds in self._annexes.values()
            for facility, cell in kinds.items()
            if facility == Facility.ENGINE_SHED
        )

    def is_stop(self, cell: Cell) -> bool:
        """Whether trains stop in the cell, where track joins only by its two ends: a station, a
        platform or an engine shed."""
        return self.station_at(cell) is not None or self.shed_at(cell) is not None

    def every_station_cell(self) -> list[Cell]:
        """Every cell a station or a platform of one fills."""
        return sorted({*self._stations, *(c for cells in self._platforms.values() for c in cells)})

    def station_at(self, cell: Cell) -> Station | None:
        """The station in the cell, if there is one."""
        return next((b for b in self._cells.get(cell, ()) if isinstance(b, Station)), None)

    def station_ends(self, cell: Cell) -> tuple[int, int]:
        """The two track points the station, platform or engine shed in the cell joins track by,
        in order."""
        stop: Station | Annex | None = self.station_at(cell) or self.shed_at(cell)
        if stop is None:
            raise ValueError(f"There is no station at {cell}")
        if stop.ends is not None:
            return stop.ends
        return 0, len(self.tile_map.grid.track_points()) // 2

    def way_problem(self, ends: tuple[int, int]) -> str | None:
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

    def buildings_at(self, cell: Cell) -> list[Building]:
        return [built for built in self._cells.get(cell, ()) if isinstance(built, Building)]

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
        """How much room is left in the cell: all its track, which shares its bed at junctions
        and crossings, takes as much as its widest piece."""
        built = self._cells.get(cell, ())
        others = sum(one.space for one in built if not isinstance(one, Track))
        return CELL_SPACE - others - self.track_room(cell)

    def track_room(self, cell: Cell) -> int:
        """How much room the track in a cell takes: as much as its widest piece."""
        return max((int(one.width) for one in self.track_at(cell)), default=0)

    def room_needed(self, construction: Construction) -> int:
        """How much more room building this takes in its cell: track only as much as it is
        wider than the widest track there."""
        if isinstance(construction, Track):
            return max(0, int(construction.width) - self.track_room(construction.cell))
        return construction.space

    def nearest_free(self, kind: BuildingKind, centre: Cell, steps: int) -> Cell | None:
        """The nearest cell to `centre`, at most `steps` away, a building of the kind can go up
        in."""
        reached = self.tile_map.grid.within(centre, steps)
        for cell, _ in sorted(reached.items(), key=lambda item: item[1]):
            if self.why_not(Building(cell, kind)) is None:
                return cell
        return None

    def why_not(self, construction: Construction) -> str | None:
        """Why this cannot be built, or None when it can."""
        cell = construction.cell
        if not self.tile_map.grid.contains(*cell):
            return "That is off the map"
        if self.tile_map.is_locked(cell):
            return LOCKED
        match construction:
            case Track():
                problem = self._track_problem(construction)
            case Station():
                problem = _site_problem(self.tile_map.tile(cell), STATION_RELIEF, "station")
                if problem is None and construction.ends is not None:
                    problem = self.way_problem(construction.ends)
            case Annex():
                problem = _site_problem(self.tile_map.tile(cell), STATION_RELIEF, "facility")
            case Building():
                problem = self._building_problem(construction)
        needed = self.room_needed(construction)
        if problem is None and needed > self.space_left(cell):
            problem = f"Not enough room: needs {needed}, {self.space_left(cell)} left"
        return problem

    def build(self, construction: Construction) -> None:
        if (problem := self.why_not(construction)) is not None:
            raise CannotBuild(problem)
        self._cells.setdefault(construction.cell, []).append(construction)
        self.version += 1
        self._index(construction, 1)

    def remove(self, construction: Construction) -> None:
        built = self._cells.get(construction.cell, [])
        if construction not in built:
            raise ValueError(f"{construction} is not built")
        built.remove(construction)
        if not built:
            del self._cells[construction.cell]
        self.version += 1
        self._index(construction, -1)

    def _index(self, construction: Construction, change: int) -> None:
        match construction:
            case Station(main=None):
                if change > 0:
                    self._stations.add(construction.cell)
                else:
                    self._stations.discard(construction.cell)
            case Station(main=main) if main is not None:
                if change > 0:
                    self._platforms.setdefault(main, set()).add(construction.cell)
                else:
                    self._platforms.get(main, set()).discard(construction.cell)
                    if not self._platforms.get(main):
                        self._platforms.pop(main, None)
            case Annex():
                kinds = self._annexes.setdefault(construction.station, {})
                if change > 0:
                    kinds[construction.facility] = construction.cell
                else:
                    kinds.pop(construction.facility, None)
                    if not kinds:
                        self._annexes.pop(construction.station, None)
                if construction.facility != Facility.ENGINE_SHED:
                    return
            case Track():
                left = self._track.get(construction.cell, 0) + change
                if left > 0:
                    self._track[construction.cell] = left
                else:
                    self._track.pop(construction.cell, None)
            case _:
                return
        self.railway_version += 1
        log = self.__dict__.setdefault("_railway_log", [])
        log.append((self.railway_version, construction.cell))
        if len(log) > RAILWAY_LOG:
            del log[: len(log) - RAILWAY_LOG]

    def railway_changed_since(self, version: int) -> set[Cell] | None:
        """The cells the railway changed in since it was at `version`; None if that is further
        back than the last RAILWAY_LOG changes, or the railway moved onto another map since."""
        if version == self.railway_version:
            return set()
        log = self.__dict__.get("_railway_log", [])
        if version < 0 or not log or log[0][0] > version + 1 or log[-1][0] != self.railway_version:
            return None
        return {cell for at, cell in log if at > version}

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
        if grid.track_turn(start, end) > MAX_TURN_DEG + TURN_SLACK_DEG:
            return f"Track can turn at most {MAX_TURN_DEG:g} degrees in one cell"
        if (problem := self._bed_problem(track.cell, track.bed, track.bridge)) is not None:
            return problem
        if track.bed == TrackBed.BRIDGE and (problem := self._bridge_problem(track)) is not None:
            return problem
        flyovers = [piece for piece in self.track_at(track.cell) if self.is_flyover(piece)]
        if any(not set(piece.ends).isdisjoint(track.ends) for piece in flyovers):
            return "Track cannot join a flyover inside its cell"
        for point in track.ends:
            beyond = grid.track_exit(track.cell, point)[0]
            if not grid.contains(*beyond):
                return "Track cannot run off the map"
            if self.tile_map.is_locked(beyond):
                return LOCKED
            for side in grid.track_flanks(track.cell, point):
                if self._bed_problem(side, track.bed, track.bridge) is not None:
                    return "Track through a corner needs the cells on both sides to take it"
        if self.track_between(track.cell, track.ends) is not None:
            return "There is track between those points already"
        return None

    def is_flyover(self, track: Track) -> bool:
        """Whether a piece of track is a flyover: on a bridge over land."""
        return (
            track.bed == TrackBed.BRIDGE and self.tile_map.tile(track.cell).terrain == Terrain.LAND
        )

    def _bridge_problem(self, track: Track) -> str | None:
        """Why a piece of track can't go on its bridge: one that can't carry it, or a flyover with
        no track under it to cross, or joining it."""
        kind = BRIDGES[track.bridge]
        if not carries(track.bridge, int(track.width), track.electric, track.maglev):
            if int(track.width) > kind.tracks:
                most = "one track" if kind.tracks == 1 else f"{kind.tracks} tracks"
                return f"A {kind.name} bridge carries at most {most}"
            needs = "maglev guideway" if track.maglev else "overhead wires"
            return f"A {kind.name} bridge can't take a {needs}"
        if not self.is_flyover(track):
            return None
        under = self.track_at(track.cell)
        if not under or self.is_stop(track.cell):
            return "A flyover goes over track"
        if any(not set(piece.ends).isdisjoint(track.ends) for piece in under):
            return "A flyover crosses the track under it without joining it"
        if any(self.is_flyover(piece) for piece in under):
            return "There is a flyover here already"
        return None

    def _bed_problem(self, cell: Cell, bed: TrackBed, bridge: Bridge = Bridge.WOOD) -> str | None:
        if not self.tile_map.grid.contains(*cell):
            return "That is off the map"
        if self.tile_map.is_locked(cell):
            return LOCKED
        tile = self.tile_map.tile(cell)
        if track_rules_for(tile, bed, bridge).can_build:
            return None
        if bed == TrackBed.BRIDGE:
            if tile.terrain == Terrain.LAND and not BRIDGES[bridge].over_track:
                return f"A {BRIDGES[bridge].name} bridge goes only over rivers and lakes"
            return "Bridges go over rivers, lakes and track"
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

    def move_onto(self, tile_map: TileMap, move: Move) -> None:
        """Moves everything built onto a bigger map, every cell as `move` moves it."""

        def moved(built: Construction) -> Construction:
            if isinstance(built, Station) and built.main is not None:
                return replace(built, cell=move.cell(built.cell), main=move.cell(built.main))
            if isinstance(built, Annex):
                return replace(built, cell=move.cell(built.cell), station=move.cell(built.station))
            return replace(built, cell=move.cell(built.cell))

        self.tile_map = tile_map
        self._cells = {
            move.cell(cell): [moved(built) for built in here] for cell, here in self._cells.items()
        }
        self._stations = move.cells(self._stations)
        self._platforms = {
            move.cell(main): move.cells(added) for main, added in self._platforms.items()
        }
        self._annexes = {
            move.cell(station): {kind: move.cell(cell) for kind, cell in kinds.items()}
            for station, kinds in self._annexes.items()
        }
        self._track = move.keys(self._track)
        self.version += 1
        self.railway_version += 1


def _site_problem(tile: Tile, relief: frozenset[Relief], name: str) -> str | None:
    if tile.terrain != Terrain.LAND:
        return f"A {name} cannot be built on water"
    if tile.relief not in relief:
        return f"A {name} cannot be built on {tile.relief.name.lower()}"
    return None
