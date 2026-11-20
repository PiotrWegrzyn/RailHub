"""What grows on the map (railroads.sim.growth runs it): the points every building gives, the
land each kind grows on, and the ladders it climbs.

Every building that grows gives points of one kind (POINTS): houses give population, farms and
orchards food, mines ore, logging camps timber, ports trade, factories industry and skyscrapers
corporate points. Each cell's neighbourhood (the cell and the cells next to it) adds up the
points of every kind around it.

What grows comes in ladders (LADDERS). Every month each cell may take a turn on each ladder, by
chance: the ladder's base chance, plus what it adds for every doubling of the points nearby, plus
the worth of the served buildings in the cell. Points nearby count by their logarithm, so growth
from neighbours alone stays steady instead of compounding. Nothing grows until the neighbourhood
holds the points the ladder needs. The land multiplies the chance (Land): its vegetation, relief
and temperature, and the best water next to the cell; a weight of 0 means the ladder never grows
there.

On its turn a ladder puts up its first kind of building in the cell, unless it can merge: when
the neighbourhood has enough of one kind on the ladder (Merge), some of them are pulled down,
the cell's own first, and one of the next kind goes up in the cell instead, the highest merge
that fits first. Ladders can cap how many of their first kind stand in one cell or one
neighbourhood, so farms and ports keep their distance.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import MERGED, BuildingKind
from railroads.sim.economy import Neighbourhoods
from railroads.sim.forests import FORESTS, ForestRule
from railroads.sim.goods import RECIPES, merging
from railroads.sim.points import Points
from railroads.sim.tilemap import Deposit, Relief, Temperature, Terrain, TileMap, Vegetation

IntGrid = npt.NDArray[np.int64]

FloatGrid = npt.NDArray[np.float64]

Cells = npt.NDArray[np.intp]


# Farms and every other grower of food (not the works that make food from it) with this many
# population points around them or more are pulled down, and none grow there.
FARMS_CROWDED_OUT = 5

# How much likelier farms, orchards and other fields are on open flat grassland with nothing
# built on it.
OPEN_FIELD = 10.0

TRADE_EVERY_MONTHS = 1

SKYSCRAPER_PEOPLE = 150

SKYSCRAPER_CHANCE = 0.005

# The base chance of a town craft coming up: breweries, brickworks, mills and the like.
CRAFT_BASE = 0.000002

# Factories gather into districts: industry nearby draws more of them only from this many points,
# more than a lone steel mill, and then strongly, so that they meet and merge.
FACTORIES_DRAW_FROM = 5

FACTORIES_PER_DOUBLING = 0.000065

# A map has only a few printing works, one per this many land cells, each printing for many towns.
PRINTING_CELLS = 40_000

# How many cells apart jewellers keep.
JEWELLERS_APART = 10

# Passengers each point of a building in a station's inner ring sends, and wants to receive, every
# month.
PASSENGERS_PER_POINT: Mapping[Points, int] = MappingProxyType(
    {
        Points.POPULATION: 1,
        Points.FOOD: 5,
        Points.TIMBER: 5,
        Points.ORE: 10,
        Points.INDUSTRY: 10,
        Points.TRADE: 10,
        Points.CORPORATE: 20,
        Points.MILITARY: 20,
        Points.AIR: 200,
        Points.WORSHIP: 1,
    }
)

_POINTS: dict[BuildingKind, tuple[Points, int]] = {
    BuildingKind.HOUSE: (Points.POPULATION, 1),
    BuildingKind.TENEMENT: (Points.POPULATION, 6),
    BuildingKind.FLATS: (Points.POPULATION, 18),
    BuildingKind.FARM: (Points.FOOD, 1),
    BuildingKind.ORCHARD: (Points.FOOD, 1),
    BuildingKind.DATE_PALMS: (Points.FOOD, 1),
    BuildingKind.ORANGE_FARM: (Points.FOOD, 1),
    BuildingKind.BANANAS: (Points.FOOD, 1),
    BuildingKind.SUGAR: (Points.FOOD, 1),
    BuildingKind.MINE: (Points.ORE, 1),
    BuildingKind.LOGGING_CAMP: (Points.TIMBER, 1),
    BuildingKind.PORT: (Points.TRADE, 1),
    BuildingKind.FACTORY: (Points.INDUSTRY, 1),
    BuildingKind.BIG_FACTORY: (Points.INDUSTRY, 3),
    BuildingKind.MEGA_FACTORY: (Points.INDUSTRY, 9),
    BuildingKind.FOOD_PLANT: (Points.INDUSTRY, 2),
    BuildingKind.BAKERY: (Points.FOOD, 1),
    BuildingKind.STEEL_MILL: (Points.INDUSTRY, 3),
    BuildingKind.SHIPYARD: (Points.INDUSTRY, 3),
    BuildingKind.FISHERY: (Points.FOOD, 1),
    BuildingKind.SHEEP_FARM: (Points.FOOD, 1),
    BuildingKind.COTTON_FIELD: (Points.FOOD, 1),
    BuildingKind.VINEYARD: (Points.FOOD, 1),
    BuildingKind.COFFEE_PLANTATION: (Points.FOOD, 1),
    BuildingKind.RUBBER_PLANTATION: (Points.FOOD, 1),
    BuildingKind.IRON_MINE: (Points.ORE, 1),
    BuildingKind.COPPER_MINE: (Points.ORE, 1),
    BuildingKind.GOLD_MINE: (Points.ORE, 1),
    BuildingKind.DIAMOND_MINE: (Points.ORE, 1),
    BuildingKind.OIL_WELL: (Points.ORE, 1),
    BuildingKind.SALT_WORKS: (Points.ORE, 1),
    BuildingKind.TEXTILE_MILL: (Points.INDUSTRY, 2),
    BuildingKind.FURNITURE_MAKER: (Points.INDUSTRY, 2),
    BuildingKind.TOOL_WORKS: (Points.INDUSTRY, 2),
    BuildingKind.WINERY: (Points.INDUSTRY, 2),
    BuildingKind.REFINERY: (Points.INDUSTRY, 3),
    BuildingKind.CHEMICAL_WORKS: (Points.INDUSTRY, 3),
    BuildingKind.ELECTRONICS_PLANT: (Points.INDUSTRY, 3),
    BuildingKind.CAR_FACTORY: (Points.INDUSTRY, 5),
    BuildingKind.GUN_FACTORY: (Points.INDUSTRY, 3),
    BuildingKind.TANK_FACTORY: (Points.INDUSTRY, 5),
    BuildingKind.PLANE_FACTORY: (Points.INDUSTRY, 5),
    BuildingKind.OUTPOST: (Points.MILITARY, 3),
    BuildingKind.BASE: (Points.MILITARY, 9),
    BuildingKind.HEADQUARTERS: (Points.MILITARY, 27),
    BuildingKind.AIRPORT: (Points.AIR, 1),
    BuildingKind.BARREL_MAKER: (Points.INDUSTRY, 2),
    BuildingKind.BEEKEEPER: (Points.FOOD, 1),
    BuildingKind.INCENSE_GROVE: (Points.FOOD, 1),
    BuildingKind.CHURCH: (Points.WORSHIP, 5),
    BuildingKind.BASILICA: (Points.WORSHIP, 20),
    BuildingKind.CATHEDRAL: (Points.WORSHIP, 50),
    BuildingKind.ST_PAULS: (Points.WORSHIP, 2137),
    BuildingKind.CAR_DEALER: (Points.CORPORATE, 1),
    BuildingKind.JEWELLER: (Points.CORPORATE, 1),
    BuildingKind.SKYSCRAPER: (Points.CORPORATE, 10),
    BuildingKind.CLAY_PIT: (Points.ORE, 1),
    BuildingKind.LIMESTONE_QUARRY: (Points.ORE, 1),
    BuildingKind.SAND_PIT: (Points.ORE, 1),
    BuildingKind.POTATO_FARM: (Points.FOOD, 1),
    BuildingKind.RICE_PADDY: (Points.FOOD, 1),
    BuildingKind.SOY_FIELD: (Points.FOOD, 1),
    BuildingKind.TEA_PLANTATION: (Points.FOOD, 1),
    BuildingKind.TOFU_MAKER: (Points.FOOD, 1),
    BuildingKind.MISO_MAKER: (Points.FOOD, 1),
    BuildingKind.FISH_DRYER: (Points.FOOD, 1),
    BuildingKind.CATTLE_RANCH: (Points.FOOD, 1),
    BuildingKind.PIG_FARM: (Points.FOOD, 1),
    BuildingKind.CHICKEN_FARM: (Points.FOOD, 1),
    BuildingKind.GASWORKS: (Points.INDUSTRY, 1),
    BuildingKind.COKE_PLANT: (Points.INDUSTRY, 2),
    BuildingKind.POTTER: (Points.INDUSTRY, 1),
    BuildingKind.TILE_WORKS: (Points.INDUSTRY, 1),
    BuildingKind.FOUNDRY: (Points.INDUSTRY, 2),
    BuildingKind.DAIRY: (Points.INDUSTRY, 1),
    BuildingKind.SAUSAGE_MAKER: (Points.FOOD, 1),
    BuildingKind.CARPET_WEAVER: (Points.CORPORATE, 2),
    BuildingKind.CLOCK_MAKER: (Points.CORPORATE, 2),
    BuildingKind.MIRROR_MAKER: (Points.CORPORATE, 2),
    BuildingKind.MANOR: (Points.CORPORATE, 8),
    BuildingKind.THEATRE: (Points.CORPORATE, 5),
    BuildingKind.HOSPITAL: (Points.CORPORATE, 5),
    BuildingKind.RAILWAY_WORKS: (Points.INDUSTRY, 20),
    BuildingKind.FLAX_FIELD: (Points.FOOD, 1),
    BuildingKind.HOP_GARDEN: (Points.FOOD, 1),
    BuildingKind.MACHINE_WORKS: (Points.INDUSTRY, 3),
    BuildingKind.BRICKWORKS: (Points.INDUSTRY, 2),
    BuildingKind.CEMENT_WORKS: (Points.INDUSTRY, 2),
    BuildingKind.GLASSWORKS: (Points.INDUSTRY, 2),
    BuildingKind.BREWERY: (Points.INDUSTRY, 1),
    BuildingKind.DISTILLERY: (Points.INDUSTRY, 1),
    BuildingKind.SAKE_BREWERY: (Points.INDUSTRY, 1),
    BuildingKind.SOY_SAUCE_BREWERY: (Points.INDUSTRY, 1),
    BuildingKind.TANNERY: (Points.INDUSTRY, 2),
    BuildingKind.SHOE_FACTORY: (Points.INDUSTRY, 2),
    BuildingKind.PLASTICS_WORKS: (Points.INDUSTRY, 3),
    BuildingKind.APPLIANCE_FACTORY: (Points.INDUSTRY, 3),
    BuildingKind.PAPER_MILL: (Points.INDUSTRY, 2),
    BuildingKind.SAWMILL: (Points.INDUSTRY, 1),
    BuildingKind.PRINTING_WORKS: (Points.INDUSTRY, 2),
    BuildingKind.SUGAR_REFINERY: (Points.INDUSTRY, 2),
    BuildingKind.PORCELAIN_WORKS: (Points.INDUSTRY, 2),
    BuildingKind.MEAT_FACTORY: (Points.INDUSTRY, 1),
    BuildingKind.POWER_PLANT: (Points.INDUSTRY, 5),
    BuildingKind.DIESEL_POWER_PLANT: (Points.INDUSTRY, 4),
    BuildingKind.GAS_POWER_PLANT: (Points.INDUSTRY, 4),
    BuildingKind.SPACE_CENTER: (Points.CORPORATE, 25),
    BuildingKind.PEAR_HQ: (Points.CORPORATE, 50),
    BuildingKind.GIGAFACTORY: (Points.INDUSTRY, 40),
    BuildingKind.ROYAL_PALACE: (Points.CORPORATE, 15),
    BuildingKind.SALT_CATHEDRAL: (Points.WORSHIP, 200),
    BuildingKind.SPA_RESORT: (Points.CORPORATE, 15),
    BuildingKind.WORLDS_FAIR: (Points.CORPORATE, 50),
    BuildingKind.FILM_STUDIO: (Points.CORPORATE, 10),
    BuildingKind.AMUSEMENT_PARK: (Points.CORPORATE, 40),
    BuildingKind.OLYMPIC_STADIUM: (Points.CORPORATE, 100),
    BuildingKind.TAVERN: (Points.CORPORATE, 1),
    BuildingKind.CLUB: (Points.CORPORATE, 3),
    BuildingKind.SUSHI_HUT: (Points.CORPORATE, 1),
    BuildingKind.FILLING_STATION: (Points.CORPORATE, 1),
    BuildingKind.DEPARTMENT_STORE: (Points.CORPORATE, 3),
    BuildingKind.HOTEL: (Points.CORPORATE, 5),
    BuildingKind.JUICE_WORKS: (Points.INDUSTRY, 1),
}

# A big works gives the points of the works merged into it, and a mega one of the big ones.
POINTS: Mapping[BuildingKind, tuple[Points, int]] = MappingProxyType(
    {
        **_POINTS,
        **{
            grown: (_POINTS[kind][0], _POINTS[kind][1] * merging(RECIPES[kind]) ** step)
            for kind, sizes in MERGED.items()
            for step, grown in enumerate(sizes, start=1)
        },
    }
)

# Each land's weight tables (Land._tables), kept with the land they belong to.
_LAND_TABLES: dict[int, tuple] = {}


@dataclass(frozen=True, slots=True)
class Land:
    """How the land changes a ladder's chances: each weight multiplies them. Water weights go by
    the best water next to the cell, LAND standing for none; a minor river through the cell itself
    counts as a river (Surroundings.on_stream). The sea is salty, so unless a ladder names it, it
    does no more than dry land; anything else left out weighs 1."""

    vegetation: Mapping[Vegetation, float] = field(default_factory=dict)
    relief: Mapping[Relief, float] = field(default_factory=dict)
    temperature: Mapping[Temperature, float] = field(default_factory=dict)
    water: Mapping[Terrain, float] = field(default_factory=dict)
    # Weight for a cell with mountains next to it.
    beside_mountains: float = 1.0
    # Only where at least one cell next to it is flat open land: not forest, not water.
    forest_edge: bool = False
    # Only where the neighbourhood holds at least this many forest cells.
    least_forest_nearby: int = 0
    # Only on cells over this deposit (railroads.mapgen.deposits).
    deposit: Deposit | None = None
    # Only within this many degrees of the equator, north or south.
    most_latitude: float | None = None
    # Only inside this box of longitude and latitude: west, east, south, north.
    region: tuple[float, float, float, float] | None = None
    # Flat grassland weighs this much more, but where anything stands already only one turn in
    # this many goes ahead.
    open_field: float = 1.0

    @property
    def on_vegetation(self) -> bool:
        """Whether its weights change as forests come and go."""
        return bool(self.vegetation) or self.open_field != 1.0 or self.on_forest_nearby

    @property
    def on_forest_nearby(self) -> bool:
        """Whether a cell's weight changes with the forests around it, not just its own."""
        return self.forest_edge or self.least_forest_nearby > 0

    @property
    def _dry(self) -> float:
        return self.water.get(Terrain.LAND, 1.0)

    def _tables(self) -> tuple[FloatGrid, FloatGrid, FloatGrid, FloatGrid]:
        """The weight of every vegetation, relief, temperature and water, as arrays indexed by
        their values, worked out once for each land."""
        tables = _LAND_TABLES.get(id(self))
        if tables is None or tables[0] is not self:
            tables = (
                self,
                np.array([self.vegetation.get(v, 1.0) for v in Vegetation]),
                np.array([self.relief.get(r, 1.0) for r in Relief]),
                np.array([self.temperature.get(t, 1.0) for t in Temperature]),
                np.array(
                    [
                        self.water.get(t, self._dry) if t == Terrain.SEA else self.water.get(t, 1.0)
                        for t in Terrain
                    ]
                ),
            )
            _LAND_TABLES[id(self)] = tables
        return tables[1:]

    def weights(
        self,
        tile_map: TileMap,
        cells: npt.NDArray[np.intp] | None = None,
        near: Cells | None = None,
    ) -> FloatGrid:
        """The weight of every cell as a grid, or of the given flat cells alone. `near` holds
        every cell's neighbours as flat indices, one past the last cell for off the map."""
        weight = self.weights_in(Surroundings(tile_map, cells, near))
        return weight.reshape(tile_map.grid.shape) if cells is None else weight

    def weights_in(self, around: Surroundings) -> FloatGrid:
        """The weight of the cells whose surroundings these are, in their order."""
        vegetation_weights, relief_weights, temperature_weights, water = self._tables()
        wet = around.is_water
        best = np.where(wet, water[around.terrain_around], 0.0).max(axis=0)
        own = around.on_stream
        best = np.where(own, np.maximum(best, water[Terrain.RIVER]), best)
        by_water = np.where(wet.any(axis=0) | own, best, water[Terrain.LAND])
        weight = (
            vegetation_weights[around.vegetation]
            * relief_weights[around.relief]
            * temperature_weights[around.temperature]
            * by_water
        )
        weight = weight * np.where(around.beside_mountains, self.beside_mountains, 1.0)
        if self.forest_edge:
            weight = weight * around.beside_open_flat
        if self.least_forest_nearby:
            weight = weight * (around.forests >= self.least_forest_nearby)
        if self.deposit is not None:
            deposits = around.deposits
            on_it = deposits is not None and deposits == self.deposit
            weight = weight * on_it
        if self.most_latitude is not None:
            weight = weight * (np.abs(around.latitude) < self.most_latitude)
        if self.region is not None:
            west, east, south, north = self.region
            lon, lat = around.longitude, around.latitude
            weight = weight * ((lon >= west) & (lon <= east) & (lat >= south) & (lat <= north))
        weight = weight * np.where(around.open_flat_grass, self.open_field, 1.0)
        return np.where(around.land, weight, 0.0)


class Surroundings:
    """What lands weigh (Land.weights_in) in some flat cells of a map, or all of them, and around
    them: worked out once, as asked for, for any number of lands. `near` holds every cell's
    neighbours as flat indices, one past the last cell for off the map."""

    def __init__(
        self,
        tile_map: TileMap,
        cells: npt.NDArray[np.intp] | None = None,
        near: Cells | None = None,
    ) -> None:
        size = tile_map.grid.rows * tile_map.grid.cols
        if near is None:
            near = Neighbourhoods(tile_map.grid).index
        self.tile_map = tile_map
        self.chosen = np.arange(size) if cells is None else cells
        self._terrain = tile_map.terrain.ravel()
        self._vegetation = tile_map.vegetation.ravel()
        self._relief = tile_map.relief.ravel()
        around = near[:, self.chosen]
        self.real = around < size
        self.around = np.where(self.real, around, 0)

    @cached_property
    def terrain_around(self) -> npt.NDArray[np.uint8]:
        return self._terrain[self.around]

    @cached_property
    def is_water(self) -> npt.NDArray[np.bool_]:
        return self.real & (self.terrain_around != Terrain.LAND)

    @cached_property
    def on_stream(self) -> npt.NDArray[np.bool_]:
        """Whether a minor river flows through the cell itself: fresh water like a river there,
        though not beside it, so the streams of a delta don't make a whole plain wet."""
        return self.tile_map.stream.ravel()[self.chosen].astype(bool) & self.land

    @cached_property
    def vegetation(self) -> npt.NDArray[np.uint8]:
        return self._vegetation[self.chosen]

    @cached_property
    def relief(self) -> npt.NDArray[np.uint8]:
        return self._relief[self.chosen]

    @cached_property
    def temperature(self) -> npt.NDArray[np.uint8]:
        return self.tile_map.temperature.ravel()[self.chosen]

    @cached_property
    def beside_mountains(self) -> npt.NDArray[np.bool_]:
        beside = self.real & (self._relief[self.around] == Relief.MOUNTAINS)
        return beside.any(axis=0)

    @cached_property
    def forest_around(self) -> npt.NDArray[np.bool_]:
        return self.real & (self._vegetation[self.around] == Vegetation.FOREST) & ~self.is_water

    @cached_property
    def beside_open_flat(self) -> npt.NDArray[np.bool_]:
        open_flat = (
            self.real
            & ~self.is_water
            & ~self.forest_around
            & (self._relief[self.around] == Relief.FLAT)
        )
        return open_flat.any(axis=0)

    @cached_property
    def forests(self) -> npt.NDArray[np.int64]:
        return self.forest_around.sum(axis=0) + (self.vegetation == Vegetation.FOREST)

    @cached_property
    def deposits(self) -> npt.NDArray[np.uint8] | None:
        deposits = self.tile_map.deposits
        return None if deposits is None else deposits.ravel()[self.chosen]

    @cached_property
    def open_flat_grass(self) -> npt.NDArray[np.bool_]:
        return (self.relief == Relief.FLAT) & (self.vegetation == Vegetation.GRASS)

    @cached_property
    def _lonlat(self) -> tuple[FloatGrid, FloatGrid]:
        rows, cols = np.unravel_index(self.chosen, self.tile_map.grid.shape)
        x, y = self.tile_map.grid.center(rows, cols)
        km = self.tile_map.grid.km_per_cell
        lon, lat = self.tile_map.bounds.to_lonlat(np.asarray(x) * km, np.asarray(y) * km)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    @cached_property
    def longitude(self) -> FloatGrid:
        return self._lonlat[0]

    @cached_property
    def latitude(self) -> FloatGrid:
        return self._lonlat[1]

    @cached_property
    def land(self) -> npt.NDArray[np.bool_]:
        return self._terrain[self.chosen] == Terrain.LAND


@dataclass(frozen=True, slots=True)
class Merge:
    """How many of a kind a neighbourhood needs before they merge into the next kind, and how
    many of them the merge pulls down."""

    needed: int
    pulled_down: int


@dataclass(frozen=True, slots=True)
class Rise:
    """The population a neighbourhood must hold for a building to become the next kind, and
    whether the map only ever gets one of that kind, for the first to get there."""

    people: int
    only_one: bool = False


@dataclass(frozen=True, slots=True)
class Ladder:
    """A line of buildings that grows: a turn puts up the first kind, and merges climb it."""

    kinds: tuple[BuildingKind, ...]
    land: Land
    base_chance: float = 0.0
    # What each doubling of the points of a kind nearby adds to the chance.
    chance_per_doubling: Mapping[Points, float] = field(default_factory=dict)
    # Points the neighbourhood must hold before the ladder has any chance there.
    needs: Mapping[Points, int] = field(default_factory=dict)
    merges: Mapping[BuildingKind, Merge] = field(default_factory=dict)
    # The most of the first kind one cell, and one neighbourhood, may hold, beyond what fits.
    most_per_cell: int | None = None
    most_nearby: int | None = None
    skyscrapers: bool = False
    # The first year it grows, and if it is rare, one on the map for this many land cells at most.
    from_year: int = 0
    land_cells_each: int | None = None
    # New ones keep at least this many cells past any building on the ladder.
    apart: int | None = None
    # Kinds that climb the ladder by the people around them rather than by merging.
    rises: Mapping[BuildingKind, Rise] = field(default_factory=dict)
    # Grows only where the neighbourhood holds fewer population points than this.
    crowded_at: int | None = None
    # Points of a kind nearby draw more only once there are at least this many of them.
    draws_from: int = 1

    def pull(self, nearby: npt.NDArray[np.int64]) -> FloatGrid:
        """What the points nearby add to the chance, in doublings."""
        return np.where(nearby >= self.draws_from, np.log2(1 + nearby), 0.0)

    @property
    def name(self) -> str:
        return self.kinds[0].value


def _merges(ladder: Ladder) -> Ladder:
    """A works ladder whose works merge as they gather (railroads.sim.constructions.MERGED): so
    many of a kind in a neighbourhood become its big one, and as many big ones its mega one."""
    kind = ladder.kinds[0]
    sizes = MERGED.get(kind)
    if sizes is None:
        return ladder
    needed = merging(RECIPES[kind])
    big, _ = sizes
    most_nearby = ladder.most_nearby
    return replace(
        ladder,
        kinds=(kind, *sizes),
        merges={kind: Merge(needed, needed), big: Merge(needed, needed)},
        most_nearby=None if most_nearby is None else max(most_nearby, needed),
    )


# Built land: towns and the factories that follow them.
# People settle by water, and few clear a forest to live in it. A house on its own draws no
# more; two or more nearby start a town.
TOWN_LAND = Land(
    vegetation={Vegetation.BARREN: 0.5, Vegetation.FOREST: 0.3},
    relief={Relief.HILLS: 1.0, Relief.MOUNTAINS: 0.2},
    temperature={Temperature.COLD: 0.6, Temperature.HOT: 0.8},
    water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.3},
)

FACTORY_LAND = Land(
    vegetation={Vegetation.FOREST: 0.2},
    relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
    temperature={Temperature.COLD: 0.7},
    water={Terrain.RIVER: 1.5, Terrain.SEA: 1.3, Terrain.LAKE: 1.2},
)

TOWNS = Ladder(
    kinds=(BuildingKind.HOUSE, BuildingKind.TENEMENT, BuildingKind.FLATS),
    land=TOWN_LAND,
    base_chance=0.0000013,
    chance_per_doubling={Points.POPULATION: 0.000084},
    merges={
        BuildingKind.HOUSE: Merge(needed=6, pulled_down=6),
        BuildingKind.TENEMENT: Merge(needed=5, pulled_down=3),
    },
    skyscrapers=True,
    draws_from=2,
)

# Farms want open, empty, flat, mild grassland by water, on the edge of towns to feed and among
# other farms; several can share a cell. They never grow inside a town, and a town that grows over
# them (FARMS_CROWDED_OUT) pulls them down.
FARMS = Ladder(
    kinds=(BuildingKind.FARM,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.1, Vegetation.FOREST: 0.02},
        relief={Relief.HILLS: 0.6, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.2, Temperature.HOT: 0.6},
        water={Terrain.RIVER: 4.0, Terrain.LAKE: 4.0},
        open_field=OPEN_FIELD,
    ),
    base_chance=9.698e-08,
    chance_per_doubling={Points.POPULATION: 1.7472e-05, Points.FOOD: 7.28e-07},
    crowded_at=FARMS_CROWDED_OUT,
)

# Orchards like sunny slopes and water nearby, and stay off the cold.
ORCHARDS = Ladder(
    kinds=(BuildingKind.ORCHARD,),
    crowded_at=FARMS_CROWDED_OUT,
    land=Land(
        vegetation={Vegetation.BARREN: 0.2, Vegetation.FOREST: 0.1},
        relief={Relief.FLAT: 0.4, Relief.HILLS: 1.5, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.05, Temperature.HOT: 1.2},
        water={Terrain.RIVER: 3.0, Terrain.LAKE: 3.0},
        open_field=OPEN_FIELD,
    ),
    base_chance=1.12e-07,
    chance_per_doubling={Points.POPULATION: 2.16e-07, Points.FOOD: 4.32e-06},
)

# Date palms grow in hot desert, best at an oasis by water.
DATE_PALMS = Ladder(
    kinds=(BuildingKind.DATE_PALMS,),
    crowded_at=FARMS_CROWDED_OUT,
    land=Land(
        vegetation={Vegetation.GRASS: 0.2, Vegetation.FOREST: 0.0},
        relief={Relief.HILLS: 0.7, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.0, Temperature.NORMAL: 0.0},
        water={Terrain.RIVER: 5.0, Terrain.LAKE: 5.0},
        open_field=OPEN_FIELD,
    ),
    base_chance=3.36e-07,
    chance_per_doubling={Points.POPULATION: 3.36e-06, Points.FOOD: 2.52e-06},
)

# Bananas grow in the jungle.
BANANA_PLANTATIONS = Ladder(
    kinds=(BuildingKind.BANANAS,),
    crowded_at=FARMS_CROWDED_OUT,
    land=Land(
        vegetation={Vegetation.BARREN: 0.0, Vegetation.GRASS: 0.0},
        relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.0, Temperature.NORMAL: 0.0},
        water={Terrain.RIVER: 2.0, Terrain.LAKE: 2.0},
        open_field=OPEN_FIELD,
    ),
    base_chance=3.36e-07,
    chance_per_doubling={Points.POPULATION: 3.36e-06, Points.FOOD: 2.52e-06},
)

# Oranges want hot, green land: not the desert, not the jungle.
ORANGE_FARMS = Ladder(
    kinds=(BuildingKind.ORANGE_FARM,),
    crowded_at=FARMS_CROWDED_OUT,
    land=Land(
        vegetation={Vegetation.BARREN: 0.0, Vegetation.FOREST: 0.0},
        relief={Relief.HILLS: 1.0, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.0, Temperature.NORMAL: 0.0},
        water={Terrain.RIVER: 2.0, Terrain.LAKE: 2.0},
        open_field=OPEN_FIELD,
    ),
    base_chance=3.36e-07,
    chance_per_doubling={Points.POPULATION: 3.36e-06, Points.FOOD: 2.52e-06},
)

# Sugar grows on open, flat land wherever it is not cold: beet in mild lands, cane in hot ones.
SUGAR_FIELDS = Ladder(
    kinds=(BuildingKind.SUGAR,),
    crowded_at=FARMS_CROWDED_OUT,
    land=Land(
        vegetation={Vegetation.BARREN: 0.05, Vegetation.FOREST: 0.02},
        relief={Relief.HILLS: 0.6, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.0, Temperature.NORMAL: 0.6, Temperature.HOT: 1.0},
        water={Terrain.RIVER: 3.0, Terrain.LAKE: 2.0},
        open_field=OPEN_FIELD,
    ),
    base_chance=2.6e-08,
    chance_per_doubling={Points.POPULATION: 8.4e-08, Points.FOOD: 3.36e-07},
)

# Mines are dug in and beside mountains and in hills, where finding ore draws more digging nearby,
# so they gather in mining districts.
MINES = Ladder(
    kinds=(BuildingKind.MINE,),
    land=Land(
        vegetation={Vegetation.FOREST: 0.6, Vegetation.GRASS: 0.8},
        relief={Relief.FLAT: 0.01, Relief.HILLS: 0.4, Relief.MOUNTAINS: 1.0},
        temperature={Temperature.COLD: 0.8},
        beside_mountains=3.0,
    ),
    base_chance=0.0000036,
    chance_per_doubling={Points.ORE: 0.0000014},
    most_per_cell=2,
    most_nearby=8,
    from_year=1700,
)

# Logging camps stand in forests with at least four forest cells around them and flat open land
# beside them, best by a river to float the logs and near towns to buy them. They clear the forest
# around them (railroads.sim.forests) and close once it is gone.
LOGGING_CAMPS = Ladder(
    kinds=(BuildingKind.LOGGING_CAMP,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.0, Vegetation.GRASS: 0.0},
        relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.HOT: 0.6},
        water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.2, Terrain.SEA: 1.0},
        forest_edge=True,
        least_forest_nearby=4,
    ),
    base_chance=0.000075,
    chance_per_doubling={Points.POPULATION: 0.0000015},
    most_per_cell=1,
    most_nearby=3,
)

# Ports open on the sea, or less often a lake, where a town has grown to use them.
PORTS = Ladder(
    kinds=(BuildingKind.PORT,),
    land=Land(
        relief={Relief.HILLS: 0.5, Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.0, Terrain.RIVER: 0.0, Terrain.SEA: 1.0, Terrain.LAKE: 0.3},
    ),
    base_chance=0.0000008,
    chance_per_doubling={Points.POPULATION: 0.000002},
    needs={Points.POPULATION: 4},
    most_per_cell=1,
    most_nearby=1,
)

# Factories make goods from steel, copper and machinery brought to them and draw more factories,
# so they gather in industrial districts more than they follow people.
FACTORIES = Ladder(
    kinds=(BuildingKind.FACTORY, BuildingKind.BIG_FACTORY, BuildingKind.MEGA_FACTORY),
    land=FACTORY_LAND,
    base_chance=0.00000004,
    chance_per_doubling={
        Points.POPULATION: 0.000000006075,
        Points.INDUSTRY: FACTORIES_PER_DOUBLING,
        Points.TIMBER: 0.00000081,
    },
    merges={
        BuildingKind.FACTORY: Merge(needed=3, pulled_down=3),
        BuildingKind.BIG_FACTORY: Merge(needed=3, pulled_down=3),
    },
    draws_from=FACTORIES_DRAW_FROM,
    from_year=1780,
)

# Canneries can the fruit of nearby orchards, palms, plantations and sugar fields with steel, and
# gather among other industry.
FOOD_PLANTS = _merges(
    Ladder(
        kinds=(BuildingKind.FOOD_PLANT,),
        land=FACTORY_LAND,
        base_chance=0.0000002,
        chance_per_doubling={
            Points.FOOD: 0.00000026,
            Points.POPULATION: 0.000000003,
            Points.INDUSTRY: 0.00000076,
        },
        needs={Points.FOOD: 1},
        most_per_cell=1,
        most_nearby=6,
        from_year=1810,
    )
)

# Steel mills make steel from iron ore and coal, near mines and among other industry.
STEEL_MILLS = _merges(
    Ladder(
        kinds=(BuildingKind.STEEL_MILL,),
        land=Land(
            vegetation={Vegetation.FOREST: 0.2},
            relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
            water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.2, Terrain.SEA: 1.2},
        ),
        base_chance=0.0000004,
        chance_per_doubling={Points.ORE: 0.000007, Points.INDUSTRY: 0.00000042},
        most_per_cell=1,
        most_nearby=2,
        from_year=1855,
    )
)

# Shipyards build on the water beside a port, among industry that feeds them.
SHIPYARDS = Ladder(
    kinds=(BuildingKind.SHIPYARD,),
    land=Land(
        relief={Relief.HILLS: 0.3, Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.0, Terrain.SEA: 1.0, Terrain.LAKE: 0.4, Terrain.RIVER: 0.3},
    ),
    base_chance=0.000005,
    chance_per_doubling={Points.TRADE: 0.000002, Points.INDUSTRY: 0.0000004},
    needs={Points.TRADE: 1},
    most_per_cell=1,
    most_nearby=1,
)


def _resource(
    kind: BuildingKind,
    deposit: Deposit,
    from_year: int = 0,
    chance: float = 0.0004,
    crowded_at: int | None = None,
) -> Ladder:
    """Something dug, drilled or grown on its own deposit, and nowhere else; what is grown there
    for food gives way to towns, as fields do (`crowded_at`)."""
    return Ladder(
        kinds=(kind,),
        land=Land(deposit=deposit),
        base_chance=chance,
        most_per_cell=1,
        most_nearby=1,
        from_year=from_year,
        crowded_at=crowded_at,
    )


def _works(
    kind: BuildingKind,
    chances: Mapping[Points, float],
    needs: Mapping[Points, int],
    from_year: int = 0,
    land_cells_each: int | None = None,
    base: float = 0.0,
) -> Ladder:
    """A works that opens anywhere by chance, a little likelier where people and industry are,
    whether or not its inputs are near."""
    return _merges(
        Ladder(
            kinds=(kind,),
            land=FACTORY_LAND,
            base_chance=base,
            chance_per_doubling=chances,
            needs=needs,
            most_per_cell=1,
            most_nearby=2,
            from_year=from_year,
            land_cells_each=land_cells_each,
        )
    )


# Rare resources, each on the deposits a map was given.
RESOURCES = (
    _resource(BuildingKind.IRON_MINE, Deposit.IRON),
    _resource(BuildingKind.COPPER_MINE, Deposit.COPPER, 1880, 0.003),
    _resource(BuildingKind.GOLD_MINE, Deposit.GOLD),
    _resource(BuildingKind.DIAMOND_MINE, Deposit.DIAMONDS),
    _resource(BuildingKind.OIL_WELL, Deposit.OIL, 1855, 0.003),
    _resource(BuildingKind.SALT_WORKS, Deposit.SALT),
    _resource(BuildingKind.FISHERY, Deposit.FISH),
    _resource(BuildingKind.SHEEP_FARM, Deposit.WOOL, crowded_at=FARMS_CROWDED_OUT),
    _resource(BuildingKind.VINEYARD, Deposit.GRAPES, crowded_at=FARMS_CROWDED_OUT),
    _resource(BuildingKind.COTTON_FIELD, Deposit.COTTON),
    _resource(BuildingKind.COFFEE_PLANTATION, Deposit.COFFEE, 1700, crowded_at=FARMS_CROWDED_OUT),
    _resource(BuildingKind.RUBBER_PLANTATION, Deposit.RUBBER, 1900, 0.003),
    _resource(BuildingKind.INCENSE_GROVE, Deposit.INCENSE),
)

POP, IND = Points.POPULATION, Points.INDUSTRY

# Soldiers: only on the few patches of military ground a map was given, each of which starts play
# with one base and two outposts (railroads.mapgen.seeding.muster_garrisons) and grows slowly.
MILITARY = Ladder(
    kinds=(BuildingKind.OUTPOST, BuildingKind.BASE, BuildingKind.HEADQUARTERS),
    land=Land(deposit=Deposit.GARRISON),
    base_chance=0.00015,
    chance_per_doubling={Points.MILITARY: 0.00002},
    merges={
        BuildingKind.OUTPOST: Merge(needed=3, pulled_down=3),
        BuildingKind.BASE: Merge(needed=3, pulled_down=3),
    },
    from_year=1800,
)

# Parish churches go up in towns, spaced out, and grow with the people around them; the first
# neighbourhood to reach 400 gets the map's one St. Paul cathedral.
CHURCHES = Ladder(
    kinds=(
        BuildingKind.CHURCH,
        BuildingKind.BASILICA,
        BuildingKind.CATHEDRAL,
        BuildingKind.ST_PAULS,
    ),
    land=Land(relief={Relief.MOUNTAINS: 0.0}),
    chance_per_doubling={POP: 0.000005},
    needs={POP: 10},
    most_per_cell=1,
    apart=4,
    rises={
        BuildingKind.CHURCH: Rise(120),
        BuildingKind.BASILICA: Rise(240),
        BuildingKind.CATHEDRAL: Rise(400, only_one=True),
    },
)

# A few hives out on green land among the farms.
BEEKEEPERS = Ladder(
    kinds=(BuildingKind.BEEKEEPER,),
    crowded_at=FARMS_CROWDED_OUT,
    land=Land(
        vegetation={Vegetation.BARREN: 0.0, Vegetation.FOREST: 0.4},
        relief={Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.1},
    ),
    base_chance=1.04e-08,
    chance_per_doubling={Points.FOOD: 1.04e-07},
    most_per_cell=1,
    most_nearby=1,
)

# Gasworks make town gas out of coal from 1820, one to a town, the likelier the more industry is
# around.
GASWORKS = _merges(
    Ladder(
        kinds=(BuildingKind.GASWORKS,),
        land=FACTORY_LAND,
        base_chance=0.000001,
        chance_per_doubling={Points.POPULATION: 0.000001},
        most_per_cell=1,
        most_nearby=1,
        from_year=1820,
    )
)

# Taverns, filling stations, brewers, distillers and potters open once FOOD_TOWN people live in
# the neighbourhood. Bakeries and the other makers of food are what towns grow on: they open
# wherever people live, the likelier the more of them.
FOOD_TOWN = 3

_BAKERY = Ladder(
    kinds=(BuildingKind.BAKERY,),
    land=TOWN_LAND,
    base_chance=0.0000005,
    chance_per_doubling={Points.POPULATION: 0.000001},
    most_per_cell=1,
    most_nearby=1,
)

BAKERIES = _merges(_BAKERY)

# Butchers open in towns from 1850, the likelier the more people live around, like bakeries.
BUTCHERS = _merges(
    replace(_BAKERY, kinds=(BuildingKind.MEAT_FACTORY,), base_chance=0.000001, from_year=1850)
)

# Town crafts that come up, like bakeries, the likelier the more people live around: potters and
# dairies as often as other crafts; tile works, foundries and sausage makers, which need two
# inputs and make goods worth more, half as often; tofu makers, miso makers and fish drying
# sheds a quarter as often, and only in towns of FOOD_TOWN people; the makers of luxury goods
# only in towns of LUXURY_TOWN people or more, half as often as potters.
LUXURY_TOWN = 20
# Potters, brewers and distillers open only in towns of FOOD_TOWN people, where they are this many
# times likelier than before, so the map keeps about as many of them.
IN_TOWNS_ONLY = 6.0


def _town_craft(
    kind: BuildingKind, share: float, from_year: int = 0, least_people: int = 0
) -> Ladder:
    return _merges(
        Ladder(
            kinds=(kind,),
            land=TOWN_LAND,
            base_chance=CRAFT_BASE * share,
            chance_per_doubling={Points.POPULATION: 0.000001},
            needs={Points.POPULATION: least_people} if least_people else {},
            most_per_cell=1,
            most_nearby=1,
            from_year=from_year,
        )
    )


TOWN_CRAFTS = (
    _town_craft(BuildingKind.POTTER, IN_TOWNS_ONLY / 3, least_people=FOOD_TOWN),
    _town_craft(BuildingKind.DAIRY, 0.25),
    _town_craft(BuildingKind.TILE_WORKS, 0.5),
    _town_craft(BuildingKind.FOUNDRY, 0.5, 1800),
    _town_craft(BuildingKind.SAUSAGE_MAKER, 0.625, 1850),
    _town_craft(BuildingKind.TOFU_MAKER, 0.25, least_people=FOOD_TOWN),
    _town_craft(BuildingKind.MISO_MAKER, 0.25, least_people=FOOD_TOWN),
    _town_craft(BuildingKind.FISH_DRYER, 0.25, least_people=FOOD_TOWN),
    _town_craft(BuildingKind.JUICE_WORKS, 0.5, 1870),
    _town_craft(BuildingKind.CARPET_WEAVER, 0.5, 1700, LUXURY_TOWN),
    _town_craft(BuildingKind.CLOCK_MAKER, 0.5, 1880, LUXURY_TOWN),
    _town_craft(BuildingKind.MIRROR_MAKER, 0.5, 1700, LUXURY_TOWN),
)

# Coke plants bake coal into coke for the steel mills from 1850, drawn to industry.
COKE_PLANTS = _merges(
    Ladder(
        kinds=(BuildingKind.COKE_PLANT,),
        land=FACTORY_LAND,
        base_chance=0.0000002,
        chance_per_doubling={Points.INDUSTRY: 0.0000004},
        most_per_cell=1,
        most_nearby=2,
        from_year=1850,
    )
)

# Buildings that only buy: manors in towns of MANOR_TOWN people, theatres in towns of
# THEATRE_TOWN and hospitals, from 1850, in towns of HOSPITAL_TOWN; each one to a neighbourhood
# and the likelier the more people live around.
MANOR_TOWN = 40
THEATRE_TOWN = 40
HOSPITAL_TOWN = 30


def _town_sight(kind: BuildingKind, least_people: int, from_year: int = 0) -> Ladder:
    return Ladder(
        kinds=(kind,),
        land=TOWN_LAND,
        base_chance=CRAFT_BASE * 0.2,
        chance_per_doubling={Points.POPULATION: 0.000001},
        needs={Points.POPULATION: least_people},
        most_per_cell=1,
        most_nearby=1,
        from_year=from_year,
    )


TOWN_SIGHTS = (
    _town_sight(BuildingKind.MANOR, MANOR_TOWN),
    _town_sight(BuildingKind.THEATRE, THEATRE_TOWN),
    _town_sight(BuildingKind.HOSPITAL, HOSPITAL_TOWN, 1850),
)

# Potatoes like cool, open plains; they feed houses and the distilleries.
POTATO_FARMS = Ladder(
    kinds=(BuildingKind.POTATO_FARM,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.05, Vegetation.FOREST: 0.02},
        relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.NORMAL: 0.8, Temperature.HOT: 0.05},
        water={Terrain.RIVER: 2.0, Terrain.LAKE: 2.0},
        open_field=OPEN_FIELD,
    ),
    base_chance=1.248e-07,
    chance_per_doubling={Points.POPULATION: 1.3e-06, Points.FOOD: 5.2e-08},
    crowded_at=FARMS_CROWDED_OUT,
    from_year=1600,
)

# Rice grows only in warm lands, within RICE_LATITUDE degrees of the equator, on flat land and in
# terraces on hills, never in mountains or forest. It wants fresh water: rivers and lakes flood
# its paddies, but salty sea water does it no good.
RICE_LATITUDE = 41.0

RICE_PADDIES = Ladder(
    kinds=(BuildingKind.RICE_PADDY,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.05, Vegetation.FOREST: 0.0},
        relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.0, Temperature.HOT: 1.2},
        water={Terrain.LAND: 1.0, Terrain.SEA: 1.0, Terrain.RIVER: 4.0, Terrain.LAKE: 4.0},
        open_field=OPEN_FIELD,
        most_latitude=RICE_LATITUDE,
    ),
    base_chance=9.698e-08,
    chance_per_doubling={Points.POPULATION: 1.7472e-05, Points.FOOD: 7.28e-07},
    crowded_at=FARMS_CROWDED_OUT,
)

# Soy is grown only in East Asia (SOY_REGION: China, Korea, Japan, Manchuria and the north of
# Vietnam), on open plains and gentle hills that are not cold, best by rivers and lakes.
SOY_REGION = (95.0, 150.0, 18.0, 50.0)

SOY_FIELDS = Ladder(
    kinds=(BuildingKind.SOY_FIELD,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.05, Vegetation.FOREST: 0.02},
        relief={Relief.HILLS: 0.6, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.0},
        water={Terrain.RIVER: 2.0, Terrain.LAKE: 2.0},
        open_field=OPEN_FIELD,
        region=SOY_REGION,
    ),
    base_chance=3.12e-08,
    chance_per_doubling={Points.POPULATION: 3.25e-07, Points.FOOD: 1.3e-08},
    crowded_at=FARMS_CROWDED_OUT,
)

# Tea grows in the warm, wet south and east of Asia (TEA_REGION: India, Ceylon, southern China,
# Japan, Java), best on hillsides, never in desert, the cold or mountains.
TEA_REGION = (65.0, 150.0, -10.0, 36.0)

TEA_PLANTATIONS = Ladder(
    kinds=(BuildingKind.TEA_PLANTATION,),
    crowded_at=FARMS_CROWDED_OUT,
    land=Land(
        vegetation={Vegetation.BARREN: 0.0, Vegetation.FOREST: 0.2},
        relief={Relief.FLAT: 0.3, Relief.HILLS: 1.5, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.0},
        water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.5},
        region=TEA_REGION,
    ),
    base_chance=1.87e-08,
    chance_per_doubling={Points.POPULATION: 3.6e-08, Points.FOOD: 3.6e-07},
)

# Cattle graze open grassland and hills, away from the heat.
CATTLE_RANCHES = Ladder(
    kinds=(BuildingKind.CATTLE_RANCH,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.1, Vegetation.FOREST: 0.05},
        relief={Relief.HILLS: 1.0, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.HOT: 0.3},
        open_field=OPEN_FIELD,
    ),
    base_chance=4.29e-09,
    chance_per_doubling={Points.POPULATION: 2.34e-07, Points.FOOD: 1.17e-08},
    crowded_at=FARMS_CROWDED_OUT,
)

# Pig farms and chicken farms keep to the same land as cattle; pigs are a little rarer.
PIG_FARMS = replace(CATTLE_RANCHES, kinds=(BuildingKind.PIG_FARM,), base_chance=3.25e-09)
CHICKEN_FARMS = replace(CATTLE_RANCHES, kinds=(BuildingKind.CHICKEN_FARM,))

# Flax wants cool, wet land by rivers and lakes.
FLAX_FIELDS = Ladder(
    kinds=(BuildingKind.FLAX_FIELD,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.0, Vegetation.FOREST: 0.02},
        relief={Relief.HILLS: 0.6, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.NORMAL: 0.6, Temperature.HOT: 0.0},
        water={Terrain.LAND: 0.3, Terrain.RIVER: 3.0, Terrain.LAKE: 2.0},
        open_field=OPEN_FIELD,
    ),
    base_chance=0.000000003,
    chance_per_doubling={Points.POPULATION: 0.0000003, Points.FOOD: 0.00000002},
    crowded_at=FARMS_CROWDED_OUT,
)

# Hop gardens stand near towns, whose brewers buy them.
HOP_GARDENS = Ladder(
    kinds=(BuildingKind.HOP_GARDEN,),
    land=Land(
        vegetation={Vegetation.BARREN: 0.0, Vegetation.FOREST: 0.05},
        relief={Relief.HILLS: 0.9, Relief.MOUNTAINS: 0.0},
        temperature={Temperature.COLD: 0.3, Temperature.HOT: 0.2},
        open_field=OPEN_FIELD,
    ),
    base_chance=0.0000000016,
    chance_per_doubling={Points.POPULATION: 0.00000012},
    crowded_at=FARMS_CROWDED_OUT,
)

# Clay is dug in river lowlands, limestone cut from hills, sand dug by the sea and rivers.
CLAY_PITS = Ladder(
    kinds=(BuildingKind.CLAY_PIT,),
    land=Land(
        vegetation={Vegetation.FOREST: 0.3},
        relief={Relief.HILLS: 0.3, Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.1, Terrain.RIVER: 4.0, Terrain.LAKE: 2.0},
    ),
    base_chance=0.00000006,
    chance_per_doubling={Points.POPULATION: 0.0000001, Points.ORE: 0.0000002},
    most_per_cell=1,
    most_nearby=2,
)

LIMESTONE_QUARRIES = Ladder(
    kinds=(BuildingKind.LIMESTONE_QUARRY,),
    land=Land(
        vegetation={Vegetation.FOREST: 0.5},
        relief={Relief.FLAT: 0.02, Relief.HILLS: 1.0, Relief.MOUNTAINS: 0.4},
    ),
    base_chance=0.00000015,
    chance_per_doubling={Points.ORE: 0.00000015},
    most_per_cell=1,
    most_nearby=2,
)

SAND_PITS = Ladder(
    kinds=(BuildingKind.SAND_PIT,),
    land=Land(
        vegetation={Vegetation.BARREN: 3.0, Vegetation.FOREST: 0.3},
        relief={Relief.HILLS: 0.3, Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.1, Terrain.SEA: 4.0, Terrain.RIVER: 3.0, Terrain.LAKE: 2.0},
    ),
    base_chance=0.000000075,
    chance_per_doubling={Points.ORE: 0.00000015},
    most_per_cell=1,
    most_nearby=2,
)

# Taverns pour a lot of beer and vodka, one to a neighbourhood, and come up as readily as
# bakeries; from 1900 clubs pour far more in towns of CLUB_TOWN people. Either one, getting its
# drink, makes the homes around it grow faster (railroads.sim.growth.LIVELY_GROWTH).
CLUB_TOWN = 30

TAVERNS = Ladder(
    kinds=(BuildingKind.TAVERN,),
    land=TOWN_LAND,
    base_chance=0.00001,
    chance_per_doubling={Points.POPULATION: 0.000001},
    needs={Points.POPULATION: FOOD_TOWN},
    most_per_cell=1,
    most_nearby=1,
)

CLUBS = replace(
    TAVERNS,
    kinds=(BuildingKind.CLUB,),
    base_chance=0.00005,
    needs={Points.POPULATION: CLUB_TOWN},
    from_year=1900,
)

# Sushi huts are a rare treat of grown towns: they serve a lot of fish and rice, only in towns of
# SUSHI_TOWN people, a tenth as readily as taverns, SUSHI_APART cells apart and one per
# SUSHI_LAND_CELLS land cells at most, and only once fish and rice are got on the map; fed, they
# liven the homes around them as taverns do.
SUSHI_TOWN = 20
SUSHI_APART = 6
SUSHI_LAND_CELLS = 4_000

SUSHI_HUTS = replace(
    TAVERNS,
    kinds=(BuildingKind.SUSHI_HUT,),
    base_chance=TAVERNS.base_chance / 10,
    chance_per_doubling={Points.POPULATION: TAVERNS.chance_per_doubling[Points.POPULATION] / 10},
    needs={Points.POPULATION: SUSHI_TOWN},
    apart=SUSHI_APART,
    land_cells_each=SUSHI_LAND_CELLS,
)

# Filling stations open in towns once motor cars come, as often as taverns.
FILLING_STATION_YEAR = 1920
FILLING_STATIONS = replace(
    TAVERNS, kinds=(BuildingKind.FILLING_STATION,), from_year=FILLING_STATION_YEAR
)

# Department stores and hotels open in big towns, spaced out, a few to a map.
DEPARTMENT_STORES = Ladder(
    kinds=(BuildingKind.DEPARTMENT_STORE,),
    land=TOWN_LAND,
    chance_per_doubling={Points.POPULATION: 0.00005},
    needs={Points.POPULATION: 60},
    most_per_cell=1,
    apart=6,
    from_year=1880,
    land_cells_each=10_000,
)

HOTELS = replace(
    DEPARTMENT_STORES,
    kinds=(BuildingKind.HOTEL,),
    apart=5,
    from_year=1900,
    land_cells_each=8_000,
)

# Sake breweries open in towns as distilleries do, once rice grows on the map.
SAKE_BREWERIES = _works(
    BuildingKind.SAKE_BREWERY,
    {POP: 8e-10, Points.FOOD: 3e-9},
    {POP: FOOD_TOWN},
    base=CRAFT_BASE * IN_TOWNS_ONLY / 2,
)

# Soy sauce breweries open in towns as distilleries do, once soy grows on the map.
SOY_SAUCE_BREWERIES = _works(
    BuildingKind.SOY_SAUCE_BREWERY,
    {POP: 8e-10, Points.FOOD: 3e-9},
    {POP: FOOD_TOWN},
    base=CRAFT_BASE * IN_TOWNS_ONLY / 2,
)

# Works that open by people and industry alone: their inputs must be brought to them.
WORKS = (
    _works(
        BuildingKind.SAWMILL,
        {POP: 4e-10, IND: 6e-9, Points.TIMBER: 3e-6},
        {},
        base=CRAFT_BASE / 4,
    ),
    _works(BuildingKind.TEXTILE_MILL, {POP: 4e-10, IND: 1.2e-8}, {}, 1780, base=CRAFT_BASE),
    _works(
        BuildingKind.FURNITURE_MAKER,
        {POP: 4e-10, IND: 6e-9, Points.TIMBER: 6e-8},
        {},
        base=CRAFT_BASE,
    ),
    _works(BuildingKind.WINERY, {POP: 8e-10, Points.FOOD: 6e-9}, {}, base=CRAFT_BASE),
    _works(
        BuildingKind.BARREL_MAKER,
        {POP: 4e-10, IND: 6e-9, Points.TIMBER: 6e-8},
        {},
        base=5e-8,
    ),
    _works(BuildingKind.TOOL_WORKS, {POP: 1.5e-9, IND: 1e-7}, {IND: 2}, 1870, base=2e-6),
    _works(
        BuildingKind.CHEMICAL_WORKS, {IND: 4e-7, Points.ORE: 6e-7}, {IND: 3}, 1880, 20_000, 4e-7
    ),
    _works(BuildingKind.REFINERY, {IND: 4e-7, Points.ORE: 1e-6}, {}, 1856, 10_000, 1e-6),
    _works(BuildingKind.CAR_FACTORY, {IND: 6e-7}, {IND: 6}, 1910, 40_000, 4e-6),
    _works(BuildingKind.ELECTRONICS_PLANT, {POP: 2.5e-8, IND: 4e-7}, {IND: 3}, 1960, 30_000, 4e-7),
    _works(BuildingKind.PLANE_FACTORY, {POP: 2.5e-8, IND: 6e-7}, {IND: 6}, 1935, 40_000, 1e-5),
    _works(BuildingKind.TANK_FACTORY, {IND: 6e-7}, {IND: 6}, 1940, 40_000, 1e-5),
    _works(BuildingKind.BRICKWORKS, {POP: 1e-9, Points.ORE: 1.2e-8}, {}, base=CRAFT_BASE),
    _works(BuildingKind.GLASSWORKS, {POP: 4e-10, IND: 6e-9, Points.ORE: 6e-9}, {}, base=CRAFT_BASE),
    _works(
        BuildingKind.BREWERY,
        {POP: 1.5e-9, Points.FOOD: 3e-9},
        {POP: FOOD_TOWN},
        base=CRAFT_BASE * IN_TOWNS_ONLY / 2,
    ),
    _works(
        BuildingKind.DISTILLERY,
        {POP: 8e-10, Points.FOOD: 3e-9},
        {POP: FOOD_TOWN},
        1600,
        base=CRAFT_BASE * IN_TOWNS_ONLY / 2,
    ),
    SAKE_BREWERIES,
    SOY_SAUCE_BREWERIES,
    _works(BuildingKind.TANNERY, {POP: 4e-10, Points.FOOD: 3e-9}, {}, base=CRAFT_BASE),
    _works(BuildingKind.SHOE_FACTORY, {POP: 8e-10, IND: 6e-9}, {POP: 3}, 1850, base=5e-8),
    _works(BuildingKind.PAPER_MILL, {IND: 6e-9, Points.TIMBER: 6e-8}, {}, 1150, base=2e-7),
    _works(BuildingKind.PRINTING_WORKS, {POP: 1.5e-9, IND: 3e-9}, {}, 1600, PRINTING_CELLS, 2e-7),
    _works(BuildingKind.SUGAR_REFINERY, {IND: 6e-9, Points.FOOD: 3e-9}, {}, 1800, base=CRAFT_BASE),
    _works(BuildingKind.PORCELAIN_WORKS, {POP: 4e-10, Points.ORE: 6e-9}, {}, 1710, 30_000, 2e-7),
    _works(BuildingKind.MACHINE_WORKS, {POP: 1.5e-9, IND: 1e-7}, {IND: 2}, 1870, base=2e-6),
    _works(BuildingKind.CEMENT_WORKS, {IND: 4e-7, Points.ORE: 6e-7}, {IND: 3}, 1880, 20_000, 4e-7),
    _works(BuildingKind.PLASTICS_WORKS, {IND: 4e-7}, {IND: 3}, 1950, 30_000, 4e-7),
    _works(BuildingKind.APPLIANCE_FACTORY, {POP: 2.5e-8, IND: 4e-7}, {IND: 3}, 1950, 30_000, 4e-7),
)

# The only source of guns when play begins.
GUN_FACTORIES = _works(BuildingKind.GUN_FACTORY, {IND: 4e-7}, {IND: 3}, 1800, 26_000, base=1e-6)

AIRPORTS = _works(BuildingKind.AIRPORT, {POP: 0.000001}, {POP: 80}, 1920, 20_000)

# Buyers found only in big cities, a handful to a map, each far from the others.
JEWELLERS = replace(
    _works(BuildingKind.JEWELLER, {POP: 0.00005}, {POP: 25}, land_cells_each=26_000),
    apart=JEWELLERS_APART,
)

CAR_DEALERS = _works(BuildingKind.CAR_DEALER, {POP: 0.000001}, {POP: 120}, 1910, 16_000)

LADDERS = (
    TOWNS,
    FARMS,
    ORCHARDS,
    DATE_PALMS,
    BANANA_PLANTATIONS,
    ORANGE_FARMS,
    SUGAR_FIELDS,
    MINES,
    LOGGING_CAMPS,
    PORTS,
    FACTORIES,
    FOOD_PLANTS,
    STEEL_MILLS,
    SHIPYARDS,
    *RESOURCES,
    *WORKS,
    JEWELLERS,
    CAR_DEALERS,
    MILITARY,
    GUN_FACTORIES,
    AIRPORTS,
    CHURCHES,
    BEEKEEPERS,
    BAKERIES,
    BUTCHERS,
    GASWORKS,
    COKE_PLANTS,
    *TOWN_CRAFTS,
    *TOWN_SIGHTS,
    POTATO_FARMS,
    RICE_PADDIES,
    SOY_FIELDS,
    TEA_PLANTATIONS,
    CATTLE_RANCHES,
    PIG_FARMS,
    CHICKEN_FARMS,
    FLAX_FIELDS,
    HOP_GARDENS,
    CLAY_PITS,
    LIMESTONE_QUARRIES,
    SAND_PITS,
    TAVERNS,
    CLUBS,
    SUSHI_HUTS,
    FILLING_STATIONS,
    DEPARTMENT_STORES,
    HOTELS,
)


@dataclass(frozen=True, slots=True)
class GrowthRule:
    # Whether goods are shared out and satisfaction scales the chances, and how often.
    trades: bool = True
    trade_every_months: int = TRADE_EVERY_MONTHS
    ladders: tuple[Ladder, ...] = LADDERS
    points: Mapping[BuildingKind, tuple[Points, int]] = field(default=POINTS)
    skyscraper_people: int = SKYSCRAPER_PEOPLE
    skyscraper_chance: float = SKYSCRAPER_CHANCE
    # How forests are cleared and spread, if they change at all, and how often.
    forests: ForestRule | None = FORESTS
    forests_every_months: int = 1
    # The first kinds of the ladders the map keeps at least a few of (Growth.least_of), putting
    # one up where too few grew; None for every ladder. Land bought keeps a quarter of them
    # (railroads.land).
    top_ups: frozenset[BuildingKind] | None = None


GROWTH = GrowthRule()
