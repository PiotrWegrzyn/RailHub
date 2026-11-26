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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from railroads.sim.constructions import MERGED, BuildingKind
from railroads.sim.forests import FORESTS, ForestRule
from railroads.sim.goods import RECIPES, merging
from railroads.sim.ladder_types import Join, Ladder, Merge, Rise
from railroads.sim.points import Points
from railroads.sim.surroundings import Land
from railroads.sim.tilemap import Deposit, Relief, Temperature, Terrain, Vegetation

# Farms and every other grower of food (not the works that make food from it) with this many
# population points around them or more are pulled down, and none grow there.
FARMS_CROWDED_OUT = 5

# How much likelier farms, orchards and other fields are on open flat grassland with nothing
# built on it.
OPEN_FIELD = 10.0

TRADE_EVERY_MONTHS = 1

SKYSCRAPER_PEOPLE = 150

SKYSCRAPER_CHANCE = 0.005

# Factories gather into districts: industry nearby draws more of them only from this many points,
# more than a lone steel mill, and then strongly, so that they meet and merge.
FACTORIES_DRAW_FROM = 5

# Growth chances are written in these, so that they read without a row of zeros.
CHANCE_UNIT = 1e-7

# A map has only a few printing works, one per this many land cells, each printing for many towns.
PRINTING_CELLS = 20_000

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
    BuildingKind.ST_PETERS: (Points.WORSHIP, 2137),
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
    base=16,
    per_doubling=840,
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
    base=3.5,
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
    base=4.2,
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
    base=12.6,
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
    base=12.6,
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
    base=12.6,
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
    base=0.975,
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
    base=56,
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
    base=117,
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
    base=8,
    needs={Points.POPULATION: 2},
    most_per_cell=1,
    most_nearby=1,
)

# Three fisheries in a neighbourhood join into a port, however few live there, unless a port
# stands within two cells.
FISHERIES_PER_PORT = 3
PORTS_APART = 2

# Fisheries grow on the shore like farms on fields, mostly by the sea and far less by lakes and
# rivers, and more where people live (railroads.sim.breeding.DRAWN_BY_KIND).
FISHERIES = Ladder(
    kinds=(BuildingKind.FISHERY,),
    land=Land(
        relief={Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.0, Terrain.SEA: 1.0, Terrain.LAKE: 0.15, Terrain.RIVER: 0.1},
    ),
    base=100,
    joins=Join(BuildingKind.PORT, FISHERIES_PER_PORT, PORTS_APART),
)

# Factories make goods from steel, copper and machinery brought to them and draw more factories,
# so they gather in industrial districts more than they follow people.
FACTORIES = Ladder(
    kinds=(BuildingKind.FACTORY, BuildingKind.BIG_FACTORY, BuildingKind.MEGA_FACTORY),
    land=FACTORY_LAND,
    base=0.4,
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
        base=2,
        needs={Points.FOOD: 1},
        from_year=1810,
    )
)

# Steel mills make steel from iron ore and coal, near mines and among other industry, five times
# as often as they once did, as nearly everything wants steel.
STEEL_MILLS = _merges(
    Ladder(
        kinds=(BuildingKind.STEEL_MILL,),
        land=Land(
            vegetation={Vegetation.FOREST: 0.2},
            relief={Relief.HILLS: 0.8, Relief.MOUNTAINS: 0.0},
            water={Terrain.RIVER: 1.5, Terrain.LAKE: 1.2, Terrain.SEA: 1.2},
        ),
        base=20,
    )
)

# Shipyards build on the water beside a port, among industry that feeds them.
SHIPYARDS = Ladder(
    kinds=(BuildingKind.SHIPYARD,),
    land=Land(
        relief={Relief.HILLS: 0.3, Relief.MOUNTAINS: 0.0},
        water={Terrain.LAND: 0.0, Terrain.SEA: 1.0, Terrain.LAKE: 0.4, Terrain.RIVER: 0.3},
    ),
    base=50,
    needs={Points.TRADE: 1},
)


def _resource(
    kind: BuildingKind,
    deposit: Deposit,
    from_year: int = 0,
    chance: float = 4000,
    crowded_at: int | None = None,
    most_nearby: int = 1,
) -> Ladder:
    """Something dug, drilled or grown on its own deposit, and nowhere else; what is grown there
    for food gives way to towns, as fields do (`crowded_at`)."""
    return Ladder(
        kinds=(kind,),
        land=Land(deposit=deposit),
        base=chance,
        most_per_cell=1,
        most_nearby=most_nearby,
        from_year=from_year,
        crowded_at=crowded_at,
    )


def _works(
    kind: BuildingKind,
    needs: Mapping[Points, int],
    from_year: int = 0,
    land_cells_each: int | None = None,
    base: float = 0.0,
    starving_years: int | None = None,
) -> Ladder:
    """A works that opens anywhere by chance, a little likelier where people and industry are,
    whether or not its inputs are near."""
    return _merges(
        Ladder(
            kinds=(kind,),
            land=FACTORY_LAND,
            base=base,
            needs=needs,
            from_year=from_year,
            land_cells_each=land_cells_each,
            starving_years=starving_years,
        )
    )


# Rare resources, each on the deposits a map was given.
RESOURCES = (
    _resource(BuildingKind.IRON_MINE, Deposit.IRON),
    _resource(BuildingKind.COPPER_MINE, Deposit.COPPER, 0, 30_000),
    _resource(BuildingKind.GOLD_MINE, Deposit.GOLD),
    _resource(BuildingKind.DIAMOND_MINE, Deposit.DIAMONDS),
    _resource(BuildingKind.OIL_WELL, Deposit.OIL, 0, 30_000),
    _resource(BuildingKind.SALT_WORKS, Deposit.SALT),
    _resource(BuildingKind.SHEEP_FARM, Deposit.WOOL, crowded_at=FARMS_CROWDED_OUT),
    _resource(BuildingKind.VINEYARD, Deposit.GRAPES, crowded_at=FARMS_CROWDED_OUT),
    _resource(BuildingKind.COTTON_FIELD, Deposit.COTTON),
    _resource(BuildingKind.COFFEE_PLANTATION, Deposit.COFFEE, 1700, crowded_at=FARMS_CROWDED_OUT),
    _resource(BuildingKind.RUBBER_PLANTATION, Deposit.RUBBER, 0, 30_000),
    _resource(BuildingKind.INCENSE_GROVE, Deposit.INCENSE),
)

POP, IND = Points.POPULATION, Points.INDUSTRY

# Soldiers: only on the few patches of military ground a map was given, each of which starts play
# with one base and two outposts (railroads.mapgen.seeding.muster_garrisons) and grows slowly.
MILITARY = Ladder(
    kinds=(BuildingKind.OUTPOST, BuildingKind.BASE, BuildingKind.HEADQUARTERS),
    land=Land(deposit=Deposit.GARRISON),
    base=15,
    merges={
        BuildingKind.OUTPOST: Merge(needed=3, pulled_down=3),
        BuildingKind.BASE: Merge(needed=3, pulled_down=3),
    },
    from_year=1800,
)

# Parish churches go up in towns, spaced out, and grow with the people around them; the first
# neighbourhood to reach 400 gets the map's one St. Peter's Basilica.
CHURCHES = Ladder(
    kinds=(
        BuildingKind.CHURCH,
        BuildingKind.BASILICA,
        BuildingKind.CATHEDRAL,
        BuildingKind.ST_PETERS,
    ),
    land=Land(relief={Relief.MOUNTAINS: 0.0}),
    base=50,
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
    base=0.39,
    most_per_cell=1,
    most_nearby=1,
)

# Gasworks make town gas out of coal from 1820, the likelier the more industry is
# around.
GASWORKS = _merges(
    Ladder(
        kinds=(BuildingKind.GASWORKS,),
        land=FACTORY_LAND,
        base=3,
        from_year=1820,
        starving_years=250,
    )
)

# Taverns, filling stations, brewers, distillers and potters open once FOOD_TOWN people live in
# the neighbourhood. Bakeries and the other makers of food are what towns grow on: they open
# wherever people live, the likelier the more of them.
FOOD_TOWN = 3

_BAKERY = Ladder(
    kinds=(BuildingKind.BAKERY,),
    land=TOWN_LAND,
    base=2.85,
)

BAKERIES = _merges(_BAKERY)

# Butchers open in towns from 1600, the likelier the more people live around, like bakeries but
# a little more often, so most towns have one.
BUTCHERS = _merges(replace(_BAKERY, kinds=(BuildingKind.MEAT_FACTORY,), base=3.75, from_year=1600))

# Town crafts that come up, like bakeries, the likelier the more people live around: potters and
# dairies as often as other crafts; tile works, foundries and sausage makers, which need two
# inputs and make goods worth more, half as often; tofu makers, miso makers and fish drying
# sheds a quarter as often, and only in towns of FOOD_TOWN people; the makers of luxury goods
# only in towns of LUXURY_TOWN people or more, half as often as potters.
LUXURY_TOWN = 20


def _town_craft(
    kind: BuildingKind,
    base: float,
    from_year: int = 0,
    least_people: int = 0,
    starving_years: int | None = None,
) -> Ladder:
    return _merges(
        Ladder(
            kinds=(kind,),
            land=TOWN_LAND,
            base=base,
            needs={Points.POPULATION: least_people} if least_people else {},
            from_year=from_year,
            starving_years=starving_years,
        )
    )


TOWN_CRAFTS = (
    _town_craft(BuildingKind.POTTER, 8.9, least_people=FOOD_TOWN, starving_years=250),
    _town_craft(BuildingKind.DAIRY, 5),
    _town_craft(BuildingKind.TILE_WORKS, 0.36, starving_years=250),
    _town_craft(BuildingKind.FOUNDRY, 2, 1800, starving_years=50),
    _town_craft(BuildingKind.SAUSAGE_MAKER, 12.5, 1850, starving_years=50),
    _town_craft(BuildingKind.TOFU_MAKER, 5, least_people=FOOD_TOWN),
    _town_craft(BuildingKind.MISO_MAKER, 5, least_people=FOOD_TOWN),
    _town_craft(BuildingKind.FISH_DRYER, 5, least_people=FOOD_TOWN),
    _town_craft(BuildingKind.JUICE_WORKS, 1, 1870, starving_years=50),
    _town_craft(BuildingKind.CARPET_WEAVER, 2, 1700, LUXURY_TOWN, 250),
    _town_craft(BuildingKind.CLOCK_MAKER, 10, 1880, LUXURY_TOWN, 50),
    _town_craft(BuildingKind.MIRROR_MAKER, 10, 1700, LUXURY_TOWN, 50),
)

# Buildings that only buy: manors in towns of MANOR_TOWN people, theatres in towns of
# THEATRE_TOWN and hospitals, from 1850, in towns of HOSPITAL_TOWN; each one to a neighbourhood
# and the likelier the more people live around.
MANOR_TOWN = 40
THEATRE_TOWN = 40
HOSPITAL_TOWN = 30


def _town_sight(kind: BuildingKind, base: float, least_people: int, from_year: int = 0) -> Ladder:
    return Ladder(
        kinds=(kind,),
        land=TOWN_LAND,
        base=base,
        needs={Points.POPULATION: least_people},
        most_per_cell=1,
        most_nearby=1,
        from_year=from_year,
    )


TOWN_SIGHTS = (
    _town_sight(BuildingKind.MANOR, 4, MANOR_TOWN),
    _town_sight(BuildingKind.THEATRE, 4, THEATRE_TOWN),
    _town_sight(BuildingKind.HOSPITAL, 4, HOSPITAL_TOWN, 1850),
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
    base=4.68,
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
    base=3.5,
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
    base=1.17,
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
    base=0.70125,
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
    base=0.29,
    crowded_at=FARMS_CROWDED_OUT,
)

# Pig farms and chicken farms keep to the same land as cattle; pigs are a little rarer.
PIG_FARMS = replace(CATTLE_RANCHES, kinds=(BuildingKind.PIG_FARM,), base=0.15)
CHICKEN_FARMS = replace(CATTLE_RANCHES, kinds=(BuildingKind.CHICKEN_FARM,), base=0.285)

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
    base=0.1125,
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
    base=0.06,
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
    base=0.6,
    most_per_cell=1,
    most_nearby=2,
)

LIMESTONE_QUARRIES = Ladder(
    kinds=(BuildingKind.LIMESTONE_QUARRY,),
    land=Land(
        vegetation={Vegetation.FOREST: 0.5},
        relief={Relief.FLAT: 0.02, Relief.HILLS: 1.0, Relief.MOUNTAINS: 0.4},
    ),
    base=4.5,
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
    base=0.75,
    most_per_cell=1,
    most_nearby=2,
)

# Taverns pour a lot of beer and vodka, one to a neighbourhood, and come up as readily as
# bakeries; from 1900 clubs pour far more in towns of CLUB_TOWN people. Either one, getting its
# drink, makes the homes around it grow faster (railroads.sim.breeding.LIVELY_GROWTH).
CLUB_TOWN = 30

TAVERNS = Ladder(
    kinds=(BuildingKind.TAVERN,),
    land=TOWN_LAND,
    base=60,
    needs={Points.POPULATION: FOOD_TOWN},
    most_per_cell=1,
    most_nearby=1,
)

CLUBS = replace(
    TAVERNS,
    kinds=(BuildingKind.CLUB,),
    base=150,
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
    base=6,
    needs={Points.POPULATION: SUSHI_TOWN},
    apart=SUSHI_APART,
    land_cells_each=SUSHI_LAND_CELLS,
)

# Filling stations open in towns once motor cars come, less often than taverns.
FILLING_STATION_YEAR = 1920
FILLING_STATIONS = replace(
    TAVERNS, kinds=(BuildingKind.FILLING_STATION,), base=20, from_year=FILLING_STATION_YEAR
)

# Department stores and hotels open in big towns, spaced out, a few to a map.
DEPARTMENT_STORES = Ladder(
    kinds=(BuildingKind.DEPARTMENT_STORE,),
    land=TOWN_LAND,
    base=500,
    needs={Points.POPULATION: 40},
    most_per_cell=1,
    apart=6,
    from_year=1950,
    land_cells_each=10_000,
)

HOTELS = replace(
    DEPARTMENT_STORES,
    kinds=(BuildingKind.HOTEL,),
    needs={Points.POPULATION: 30},
    apart=5,
    from_year=1900,
    land_cells_each=8_000,
)

# Sake breweries open in towns as distilleries do, once rice grows on the map.
SAKE_BREWERIES = _works(
    BuildingKind.SAKE_BREWERY,
    {POP: FOOD_TOWN},
    base=30,
)

# Soy sauce breweries open in towns as distilleries do, once soy grows on the map.
SOY_SAUCE_BREWERIES = _works(
    BuildingKind.SOY_SAUCE_BREWERY,
    {POP: FOOD_TOWN},
    base=30,
)

# Works that open by people and industry alone: their inputs must be brought to them.
# Works whose inputs seldom come together before the railway last starving_years long enough
# that the land holds about as many in 1850 as it gains by 2040 when well served.
WORKS = (
    _works(BuildingKind.SAWMILL, {}, base=2),
    # Textile mills open three times as often as they once did.
    _works(BuildingKind.TEXTILE_MILL, {}, 1780, base=1.8, starving_years=250),
    _works(BuildingKind.FURNITURE_MAKER, {}, base=1.1),
    _works(BuildingKind.WINERY, {}, base=4),
    _works(BuildingKind.BARREL_MAKER, {}, base=0.25, starving_years=250),
    _works(BuildingKind.TOOL_WORKS, {IND: 2}, 0, base=20),
    _works(BuildingKind.CHEMICAL_WORKS, {IND: 3}, 0, 20_000, base=4),
    _works(BuildingKind.REFINERY, {}, 0, 10_000, base=3.5),
    _works(BuildingKind.CAR_FACTORY, {IND: 6}, 0, 40_000, base=40),
    _works(BuildingKind.ELECTRONICS_PLANT, {IND: 3}, 0, 30_000, base=4),
    _works(BuildingKind.PLANE_FACTORY, {IND: 6}, 0, 40_000, base=100),
    _works(BuildingKind.TANK_FACTORY, {IND: 6}, 0, 40_000, base=100),
    _works(BuildingKind.BRICKWORKS, {}, base=0.8, starving_years=250),
    _works(BuildingKind.GLASSWORKS, {}, base=0.36, starving_years=250),
    # Breweries open five times as often as they once did, so most towns have one.
    _works(BuildingKind.BREWERY, {POP: FOOD_TOWN}, base=31.5, starving_years=250),
    _works(BuildingKind.DISTILLERY, {POP: FOOD_TOWN}, 1600, base=6.35, starving_years=250),
    SAKE_BREWERIES,
    SOY_SAUCE_BREWERIES,
    _works(BuildingKind.TANNERY, {}, base=0.5),
    _works(BuildingKind.SHOE_FACTORY, {POP: 3}, 1850, base=0.5),
    _works(BuildingKind.PAPER_MILL, {}, 1150, base=1.5),
    _works(BuildingKind.PRINTING_WORKS, {}, 1600, PRINTING_CELLS, base=4),
    _works(BuildingKind.SUGAR_REFINERY, {}, 1800, base=0.9),
    _works(BuildingKind.PORCELAIN_WORKS, {}, 1710, 20_000, base=2),
    _works(BuildingKind.MACHINE_WORKS, {IND: 2}, 0, base=20),
    _works(BuildingKind.CEMENT_WORKS, {}, 1840, base=2),
    _works(BuildingKind.PLASTICS_WORKS, {IND: 3}, 0, 30_000, base=4),
    _works(BuildingKind.APPLIANCE_FACTORY, {IND: 3}, 0, 30_000, base=4),
)

# The only source of guns when play begins.
GUN_FACTORIES = _works(BuildingKind.GUN_FACTORY, {}, 1800, 26_000, base=3, starving_years=250)

AIRPORTS = _works(BuildingKind.AIRPORT, {POP: 80}, 1920, 20_000, base=10)

# Buyers found only in big cities, a handful to a map, each far from the others.
JEWELLERS = replace(
    _works(BuildingKind.JEWELLER, {POP: 25}, land_cells_each=26_000, base=500),
    apart=JEWELLERS_APART,
)

CAR_DEALERS = _works(BuildingKind.CAR_DEALER, {POP: 120}, 0, 16_000, base=10)

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
    FISHERIES,
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


def ladder_of(kind: BuildingKind, ladders: Iterable[Ladder] = LADDERS) -> Ladder | None:
    """The ladder a kind of building stands on, if any."""
    return next((ladder for ladder in ladders if kind in ladder.kinds), None)
