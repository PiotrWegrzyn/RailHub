"""The trains on the railway (railroads.sim.railway runs them): the engines that can be bought, the
cars, each carrying one good up to its car_size, the stops a train calls at with the cars it pulls
from each, and the train itself, slowed by what its cars and their goods weigh against the most
its engine pulls, in tonnes. Every engine costs a share of its price to run each month, by what drives
it (RUNNING_SHARE). A train sets off from a stand and picks up speed evenly until it runs at full
speed after its engine's pickup time, the Planet's twelve seconds at normal speed, a light
engine's a few, an express engine's about as long as a general one of its day, and a pioneer's
or a heavy freight engine's far more. An express always leads its coaches with a dining car,
which carries nothing but weighs and costs like any other car.

Freight remembers where it came from as parcels: how much from each station."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

from railroads.sim.finance import TRAIN_PRICE
from railroads.sim.goods import General, Good
from railroads.sim.move import Move

# What a stop asks for a car of: one good, or a whole kind of goods.
type Cargo = Good | General
from railroads.sim.grids import Cell
from railroads.sim.tiles import Tile

if TYPE_CHECKING:
    from railroads.sim.signals import Pass

# Track a train holds in a cell (railroads.sim.signals): the cell, the points on its edge the
# train's way enters and leaves by, how far along its path it enters and leaves the cell,
# whether it took it standing there without holding up the trains already holding it, and which
# track of quadruple track it takes.
type Claim = tuple[Cell, Point, Point, float, float, bool, int]

MAX_CARS = 5


# Cars come in sizes up the Fibonacci numbers, each holding as much as the two before it.
CAR_SIZES = (
    1.0,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    13.0,
    21.0,
    34.0,
    55.0,
    89.0,
    144.0,
    233.0,
    377.0,
    610.0,
    987.0,
    1597.0,
    2584.0,
)
# How much bigger, and so heavier, each size up CAR_SIZES makes a car.
CAR_GROWTH = CAR_SIZES[-1] / CAR_SIZES[-2]

# Cars come in generations, each a size up CAR_SIZES from the one before: the pioneers' small
# wooden wagons (1), the wagons of the Planet's day (2), which the sizes below are for, and the
# steel and bogie wagons after them, up to today's hoppers of about 100 t (GENERATIONS). Each
# comes with an era of the tiles (railroads.sim.tiles), and an engine pulls those up to its
# top_generation, the one after the generation of the era its year falls in (GENERATION_YEARS).
BASE_GENERATION = 2
GENERATIONS = 7
GENERATION_YEARS = (1843, 1873, 1901, 1931, 1955, 1972)


def generation_in(year: int) -> int:
    """The generation of wagons of the era a year falls in at the world's own pace."""
    return 1 + sum(year >= start for start in GENERATION_YEARS)


# Vehicles, carried only from the generation of the years they are first made in: so many to a
# car of it and a Fibonacci size up for each generation after. Cars, from about 1910, from the
# fourth; tanks and planes, from the 1930s, from the fifth, from the first size.
VEHICLES: Mapping[Good, tuple[int, float]] = MappingProxyType(
    {Good.CARS: (4, 5.0), Good.PLANES: (5, 1.0), Good.TANKS: (5, 1.0)}
)

SMALL_CAR = 8.0
MEDIUM_CAR = 21.0
CAR_CAPACITY = 55.0
BULK_CAR = 144.0


def _sized(size: float, *goods: Good) -> dict[Good, float]:
    return dict.fromkeys(goods, size)


# How many units of each good a car of BASE_GENERATION holds, a size up
# from what fits in a wagon of about 10 cubic metres bearing about 8 tonnes (UNIT_KG); goods not
# listed go CAR_CAPACITY to a car; VEHICLES have their own sizes.
CAR_UNITS: Mapping[Good, float] = MappingProxyType(
    _sized(
        BULK_CAR,
        Good.COAL,
        Good.IRON,
        Good.SAND,
        Good.CLAY,
        Good.LIMESTONE,
        Good.SALT,
        Good.WHEAT,
        Good.RICE,
        Good.SOYBEANS,
        Good.BREAD,
        Good.COFFEE,
        Good.REFINED_SUGAR,
        Good.FERTILISER,
    )
    | _sized(
        89.0,
        Good.POTATOES,
        Good.SUGAR,
        Good.RUBBER,
        Good.PLANKS,
        Good.TEA,
        Good.OIL,
        Good.FUEL,
        Good.GAS,
        Good.PLASTICS,
        Good.GUNS,
        Good.CHICKENS,
    )
    | _sized(
        34.0,
        Good.COTTON,
        Good.FLAX,
        Good.HOPS,
        Good.HONEY,
        Good.WAX,
        Good.GLASS,
        Good.PORCELAIN,
        Good.POTS,
        Good.STOVE_TILES,
        Good.PIGS,
        Good.ELECTRONICS,
        Good.MAIL,
    )
    | _sized(
        MEDIUM_CAR,
        Good.PAPER,
        Good.STEEL,
        Good.COPPER,
        Good.CASTINGS,
        Good.BRICKS,
        Good.CEMENT,
        Good.APPLIANCES,
    )
    | _sized(13.0, Good.FURNITURE, Good.CATTLE)
    | _sized(SMALL_CAR, Good.MACHINERY, Good.GOLD, Good.DIAMONDS)
)


def _weighing(kg: float, *goods: Good) -> dict[Good, float]:
    return dict.fromkeys(goods, kg)


# What a unit of each good is and weighs, in kilograms: a quintal (sack) of coal, ore, grain or
# fruit in crates, a 60 kg sack of coffee, a crate of clothes or shoes, a 200 litre barrel of a
# drink or oil, a bundle of steel, a cow or a pig, a car; goods not listed weigh UNIT_KG, a crate.
UNIT_KG = 60.0
GOOD_KG: Mapping[Good, float] = MappingProxyType(
    _weighing(
        100.0,
        Good.COAL,
        Good.IRON,
        Good.SAND,
        Good.CLAY,
        Good.LIMESTONE,
        Good.SALT,
        Good.WHEAT,
        Good.RICE,
        Good.SOYBEANS,
        Good.POTATOES,
        Good.SUGAR,
        Good.REFINED_SUGAR,
        Good.FERTILISER,
        Good.PLANKS,
        Good.WOOD_CHIPS,
        Good.APPLES,
        Good.ORANGES,
        Good.BANANAS,
        Good.DATES,
        Good.GRAPES,
        Good.MEAT,
        Good.LEATHER,
        Good.FLAX,
        Good.HOPS,
        Good.PLASTICS,
        Good.PIGS,
        Good.PORCELAIN,
        Good.FURNITURE,
        Good.ELECTRONICS,
    )
    | _weighing(
        200.0,
        Good.CANS,
        Good.MILK,
        Good.BEER,
        Good.WINE,
        Good.JUICE,
        Good.SAKE,
        Good.MISO,
        Good.NEWSPAPERS,
    )
    | _weighing(150.0, Good.DAIRY, Good.SAUSAGES, Good.TOFU, Good.WOOL, Good.TOOLS)
    | _weighing(600.0, Good.STEEL, Good.COPPER, Good.CASTINGS, Good.BRICKS, Good.CEMENT)
    | _weighing(300.0, Good.HONEY, Good.WAX, Good.STOVE_TILES)
    | _weighing(50.0, Good.BARRELS, Good.SHEEP, Good.LUXURIES, Good.INCENSE)
    | _weighing(30.0, Good.BREAD, Good.MAIL)
    | {
        Good.FISH: 180.0,
        Good.EGGS: 120.0,
        Good.SHOES: 70.0,
        Good.COTTON: 220.0,
        Good.RUBBER: 110.0,
        Good.TEA: 45.0,
        Good.OIL: 170.0,
        Good.FUEL: 160.0,
        Good.GAS: 100.0,
        Good.CHEMICALS: 220.0,
        Good.VODKA: 190.0,
        Good.SOY_SAUCE: 230.0,
        Good.PAPER: 500.0,
        Good.GLASS: 400.0,
        Good.POTS: 80.0,
        Good.APPLIANCES: 80.0,
        Good.MACHINERY: 1_000.0,
        Good.CATTLE: 500.0,
        Good.CHICKENS: 25.0,
        Good.GOLD: 250.0,
        Good.DIAMONDS: 10.0,
        Good.LOGS: 200.0,
        Good.CARS: 1_200.0,
        Good.PLANES: 15_000.0,
        Good.TANKS: 45_000.0,
        Good.PASSENGERS: 80.0,
    }
)


def car_size(good: Good, generation: int = BASE_GENERATION) -> float:
    """How much of a good one car of a generation carries;
    no VEHICLES before the generation they are first carried in."""
    vehicle = VEHICLES.get(good)
    if vehicle is not None:
        first, size = vehicle
        return bigger(size, generation - first) if generation >= first else 0.0
    return bigger(CAR_UNITS.get(good, CAR_CAPACITY), generation - BASE_GENERATION)


# The most a general car of a kind carries, whatever it holds: processed food rides in medium cars.
GENERAL_CAR_MOST: Mapping[General, float] = MappingProxyType({General.PROCESSED_FOOD: MEDIUM_CAR})


def general_car_size(general: General, good: Good, generation: int = BASE_GENERATION) -> float:
    """How much of a good one general car of a kind and generation carries."""
    most = GENERAL_CAR_MOST.get(general)
    size = car_size(good, generation)
    return size if most is None else min(size, bigger(most, generation - BASE_GENERATION))


def bigger(size: float, levels: int) -> float:
    """A car size `levels` sizes up CAR_SIZES, or down for fewer than none."""
    index = min(max(CAR_SIZES.index(size) + levels, 0), len(CAR_SIZES) - 1)
    return CAR_SIZES[index]


# What an empty freight car of each generation weighs, in tonnes, about as much as real wagons of
# its day, whatever it holds: from 1.5 t for a pioneer's wooden wagon to 24 t for a 100 t hopper.
# Coaches and mail vans weigh TARE times as much.
GENERATION_TARE = (1.5, 4.0, 6.0, 9.0, 14.0, 18.0, 24.0)
FREIGHT_TARE = GENERATION_TARE[BASE_GENERATION - 1]
TARE: Mapping[Good, float] = MappingProxyType({Good.PASSENGERS: 1.75, Good.MAIL: 1.25})


def tare_of(good: Good, generation: int = BASE_GENERATION) -> float:
    """What an empty car of a generation for a good weighs, in tonnes."""
    return GENERATION_TARE[generation - 1] * TARE.get(good, 1.0)


def _counting(singular: str, plural: str, *goods: Good) -> dict[Good, tuple[str, str]]:
    return dict.fromkeys(goods, (singular, plural))


# What a unit of each good is called, one and many: a sack of coal or grain, a crate of
# clothes, a barrel of oil, a head of cattle; goods not listed come in crates.
UNIT_NAMES: Mapping[Good, tuple[str, str]] = MappingProxyType(
    _counting(
        "sack",
        "sacks",
        Good.COAL,
        Good.IRON,
        Good.SAND,
        Good.CLAY,
        Good.LIMESTONE,
        Good.SALT,
        Good.WHEAT,
        Good.RICE,
        Good.SOYBEANS,
        Good.POTATOES,
        Good.SUGAR,
        Good.REFINED_SUGAR,
        Good.PLASTICS,
        Good.COFFEE,
        Good.FERTILISER,
        Good.WOOD_CHIPS,
    )
    | _counting(
        "barrel",
        "barrels",
        Good.OIL,
        Good.FUEL,
        Good.GAS,
        Good.CHEMICALS,
        Good.MILK,
        Good.BEER,
        Good.WINE,
        Good.VODKA,
        Good.JUICE,
        Good.SAKE,
        Good.SOY_SAUCE,
        Good.MISO,
        Good.HONEY,
        Good.WAX,
        Good.BARRELS,
    )
    | _counting(
        "bale", "bales", Good.WOOL, Good.COTTON, Good.FLAX, Good.HOPS, Good.RUBBER, Good.LEATHER
    )
    | _counting("bundle", "bundles", Good.STEEL, Good.COPPER, Good.CASTINGS, Good.NEWSPAPERS)
    | _counting("pallet", "pallets", Good.BRICKS, Good.CEMENT)
    | _counting("head", "head", Good.CATTLE, Good.PIGS, Good.SHEEP)
    | _counting("chest", "chests", Good.TEA, Good.GOLD)
    | _counting("piece", "pieces", Good.FURNITURE, Good.APPLIANCES)
    | {
        Good.LOGS: ("log", "logs"),
        Good.PLANKS: ("bundle", "bundles"),
        Good.MEAT: ("side", "sides"),
        Good.PAPER: ("roll", "rolls"),
        Good.MAIL: ("bag", "bags"),
        Good.CHICKENS: ("cage", "cages"),
        Good.MACHINERY: ("machine", "machines"),
        Good.DIAMONDS: ("casket", "caskets"),
        Good.PASSENGERS: ("passenger", "passengers"),
        Good.CARS: ("car", "cars"),
        Good.TANKS: ("tank", "tanks"),
        Good.PLANES: ("plane", "planes"),
    }
)
# Goods whose unit is the good itself, named without "of".
SELF_COUNTED = frozenset({Good.PASSENGERS, Good.CARS, Good.TANKS, Good.PLANES})


def unit_name(good: Good, amount: float = 2.0) -> str:
    """What a unit of a good is called: one, or many."""
    singular, plural = UNIT_NAMES.get(good, ("crate", "crates"))
    return singular if round(amount) == 1 else plural


def amount_of(good: Good, amount: float) -> str:
    """So much of a good in its units: 89 sacks of coal, 8 head of cattle, 34 passengers."""
    counted = f"{amount:.0f} {unit_name(good, amount)}"
    return counted if good in SELF_COUNTED else f"{counted} of {good.value}"


def unit_label(good: Good) -> str:
    """What a unit of a good is and weighs: coal: 1 sack = 100 kg."""
    return f"{good.value}: 1 {unit_name(good, 1)} = {GOOD_KG.get(good, UNIT_KG):,.0f} kg"


def unit_tonnes(good: Good) -> float:
    """What a unit of a good weighs, in tonnes."""
    return GOOD_KG.get(good, UNIT_KG) / 1_000.0


def full_tonnes(good: Good, capacity: float, generation: int = BASE_GENERATION) -> float:
    """What a car of a generation holding `capacity` of a good weighs full, in tonnes."""
    return tare_of(good, generation) + capacity * unit_tonnes(good)


# About how long an engine with its tender, and each car, are on the map, in cells.
ENGINE_LENGTH = 0.8
CAR_LENGTH = 0.42


class Kind(Enum):
    """What an engine is for. Pioneers come before the Planet and are slow to get going. Light
    engines are cheap to buy and run and are up to speed at once, but pull three or four cars,
    slowly, and a heavy load slows them most: for short lines with many stops. General ones are
    the Planet and its heirs. Heavy freight engines pull twice as many cars or more and are hardly
    slowed by a full load, but run slowly, take longest of all to get going and cost the most to
    run after express engines. Express engines carry only passengers, much faster, with a dining
    car, but cost the most, so they pay on long lines."""

    PIONEER = "pioneer"
    LIGHT = "light"
    GENERAL = "general"
    HEAVY = "heavy freight"
    EXPRESS = "express"


class Power(Enum):
    STEAM = "steam"
    DIESEL = "diesel"
    ELECTRIC = "electric"
    MAGLEV = "maglev"


# Speed 1 stands for this many km/h.
KMH_PER_SPEED = 60.0
# How much an engine of a kind pulls at most, as a share of the train it is built for, full, of
# its top_generation's cars: coal for most, as real trains were loaded up to what their engine
# could pull, so a full train of coal runs its slowest; full coaches for expresses and railcars,
# with a little room to spare; light engines about half that.
PULL_SHARE = {
    Kind.PIONEER: 1.0,
    Kind.LIGHT: 0.6,
    Kind.GENERAL: 1.0,
    Kind.HEAVY: 1.0,
    Kind.EXPRESS: 1.25,
}
# How much of its top speed an engine of a kind loses pulling as much as it can, and a share of
# that for a share of it: half for most, less for heavy freight engines, more for light ones.
SLOWDOWN = {
    Kind.PIONEER: 0.5,
    Kind.LIGHT: 0.6,
    Kind.GENERAL: 0.5,
    Kind.HEAVY: 0.3,
    Kind.EXPRESS: 0.5,
}
# Passengers on an express pay this much more, and on a maglev MAGLEV_FARE times as much.
EXPRESS_FARE = 1.2
MAGLEV_FARE = 11.0
# How much fuel (railroads.sim.fuel_market) an engine burns a cell for each car it is built to
# pull, by what drives it: coal, diesel fuel or electricity. Light engines burn LIGHT_BURN of that.
BURN_PER_CAR = {
    Power.STEAM: 0.25,
    Power.DIESEL: 0.1,
    Power.ELECTRIC: 0.15,
    Power.MAGLEV: 0.4,
}
LIGHT_BURN = 0.5
# What an engine burns grows with what its cars weigh: BURN_ALONE of its burn with no cars, its
# burn when they weigh half the most it pulls, and BURN_ALONE + 1 of it when they weigh the most.
BURN_ALONE = 0.5


@dataclass(frozen=True, slots=True)
class Engine:
    """An engine model: the year it was built in history, how fast it runs against the Planet,
    how many cars it pulls (an express's coaches, without its dining car), what it costs, what it
    is for and what drives it; it can be bought once a supply tile reaches a level (OPENED_BY).
    Electric engines run only on electrified track. Once a newer engine of its line can be bought, it is
    no longer sold (railroads.sim.railway.Railway.engines)."""

    title: str
    year: int
    speed: float
    cars: int
    price: float
    kind: Kind = Kind.GENERAL
    power: Power = Power.STEAM
    # How many seconds of play at normal speed it takes to get from a stand to full speed.
    pickup: float = 10.0
    # How many cells it can run on the coal or diesel fuel it takes on setting off, so it can
    # only call at stops that near each other; 0 for electric and maglev engines, which run as
    # far as the wires or guideway go.
    tank: float = 0.0
    # Whether it burns wood (logs) instead of coal, as American engines did while forests
    # were near and coal far.
    wood: bool = False
    # Its PULL_SHARE where that is not its kind's: the Rocket was built for speed, not loads.
    pull_share: float = 0.0
    # Whether it pulls as much as its coaches full rather than full coal cars: railcars.
    railcar: bool = False

    @property
    def electric(self) -> bool:
        return self.power == Power.ELECTRIC

    @property
    def maglev(self) -> bool:
        return self.power == Power.MAGLEV

    @property
    def express(self) -> bool:
        return self.kind == Kind.EXPRESS

    @property
    def burn(self) -> float:
        """How much fuel it burns a cell pulling half the most it can."""
        light = LIGHT_BURN if self.kind == Kind.LIGHT else 1.0
        return BURN_PER_CAR[self.power] * self.hauls * light

    @property
    def top_generation(self) -> int:
        """The newest generation of cars it pulls: pioneers only the first, every other engine
        the one after the generation of its year, built as it was for the cars to come."""
        if self.kind == Kind.PIONEER:
            return 1
        return min(generation_in(self.year) + 1, GENERATIONS)

    @property
    def kmh(self) -> float:
        return self.speed * KMH_PER_SPEED

    @property
    def line(self) -> Kind:
        """The line of engines it belongs to: pioneers lead up to the Planet's."""
        return Kind.GENERAL if self.kind == Kind.PIONEER else self.kind

    def replaces(self, older: Engine) -> bool:
        """Whether it takes an older engine's place on sale: a newer one of the same line that
        runs on the same track, so an electric never ends the sale of a steam or diesel engine,
        nor a maglev that of anything on rails."""
        return (
            self.line == older.line
            and self.electric == older.electric
            and self.maglev == older.maglev
            and self.year > older.year
        )

    @property
    def hauls(self) -> int:
        """How many cars it is built to pull, the dining car included."""
        return self.cars + (1 if self.express else 0)

    @property
    def tonnes(self) -> float:
        """The most it pulls, in tonnes, behind it whatever generation of cars: PULL_SHARE of its
        train full of its top_generation's cars."""
        coaches = self.express or self.railcar
        good = Good.PASSENGERS if coaches else Good.COAL
        generation = self.top_generation
        car = full_tonnes(good, car_size(good, generation), generation)
        share = self.pull_share or PULL_SHARE[Kind.EXPRESS if coaches else self.kind]
        return share * self.hauls * car


P, L, G, H, X = Kind.PIONEER, Kind.LIGHT, Kind.GENERAL, Kind.HEAVY, Kind.EXPRESS
DIESEL, ELECTRIC, MAGLEV = Power.DIESEL, Power.ELECTRIC, Power.MAGLEV
ENGINES: dict[str, Engine] = {
    "penydarren": Engine(
        "Penydarren steam wagon", 1804, 0.25, 1, 1_000.0, P, pickup=45.0, tank=4.0
    ),
    "catch_me": Engine("Catch Me Who Can", 1808, 0.2, 1, 1_500.0, P, pickup=43.0, tank=5.0),
    "salamanca": Engine("Salamanca", 1812, 0.2, 1, 2_000.0, P, pickup=42.0, tank=6.0),
    "puffing_billy": Engine("Puffing Billy", 1815, 0.25, 1, 2_500.0, P, pickup=40.0, tank=7.0),
    "locomotion": Engine("Locomotion", 1820, 0.3, 2, 4_000.0, P, pickup=38.0, tank=8.0),
    "rocket": Engine("Rocket", 1826, 0.4, 2, 6_000.0, P, pickup=36.0, tank=10.0, pull_share=0.55),
    "lion": Engine("Lion", 1834, 0.45, 2, 8_000.0, P, pickup=35.0, tank=12.0),
    "planet": Engine("Planet steam engine", 1840, 0.5, 3, TRAIN_PRICE, G, pickup=12.0, tank=21.0),
    "tank": Engine("Tank engine", 1840, 0.4, 2, 3_000.0, L, pickup=3.0, tank=13.0),
    "goods": Engine("Goods engine", 1853, 0.35, 4, 15_000.0, H, pickup=50.0, tank=20.0),
    "american": Engine(
        "American 4-4-0", 1856, 0.7, 4, 30_000.0, G, pickup=11.0, tank=25.0, wood=True
    ),
    "single": Engine(
        "Single-driver express",
        1858,
        1.0,
        3,
        40_000.0,
        X,
        pickup=12.0,
        tank=30.0,
    ),
    "light_tank": Engine("Light tank engine", 1862, 0.55, 2, 9_000.0, L, pickup=3.0, tank=16.0),
    "mogul": Engine("Mogul freight engine", 1865, 0.45, 5, 45_000.0, H, pickup=45.0, tank=30.0),
    "atlantic": Engine(
        "Atlantic express",
        1900,
        1.5,
        4,
        120_000.0,
        X,
        pickup=10.0,
        tank=50.0,
    ),
    "mallet": Engine(
        "Mallet articulated",
        1905,
        0.6,
        6,
        135_000.0,
        H,
        pickup=40.0,
        tank=50.0,
    ),
    "pacific": Engine("Pacific engine", 1910, 1.0, 5, 90_000.0, G, pickup=9.0, tank=50.0),
    "boxcab": Engine("Boxcab electric", 1925, 1.6, 5, 270_000.0, G, ELECTRIC, pickup=6.0),
    "railbus": Engine(
        "Railbus",
        1935,
        1.0,
        3,
        27_000.0,
        L,
        DIESEL,
        pickup=2.0,
        tank=60.0,
        railcar=True,
    ),
    "streamliner": Engine(
        "Streamliner",
        1935,
        2.3,
        5,
        360_000.0,
        X,
        pickup=8.0,
        tank=60.0,
    ),
    "big_boy": Engine(
        "Big articulated",
        1940,
        0.7,
        7,
        405_000.0,
        H,
        pickup=35.0,
        tank=60.0,
    ),
    "mixed_diesel": Engine(
        "Mixed-traffic diesel",
        1950,
        1.3,
        6,
        800_000.0,
        G,
        DIESEL,
        pickup=7.0,
        tank=100.0,
    ),
    "heavy_diesel": Engine(
        "Heavy diesel",
        1960,
        0.9,
        8,
        1_200_000.0,
        H,
        DIESEL,
        pickup=30.0,
        tank=100.0,
    ),
    "bullet": Engine(
        "Bullet train",
        1964,
        3.5,
        6,
        1_100_000.0,
        X,
        ELECTRIC,
        pickup=7.0,
    ),
    "express": Engine(
        "Universal electric",
        1965,
        2.4,
        7,
        2_400_000.0,
        G,
        ELECTRIC,
        pickup=5.0,
    ),
    "sprinter": Engine(
        "Sprinter unit",
        1985,
        1.6,
        3,
        81_000.0,
        L,
        DIESEL,
        pickup=2.0,
        tank=100.0,
        railcar=True,
    ),
    "high_speed": Engine(
        "High-speed electric",
        1990,
        5.0,
        8,
        3_200_000.0,
        X,
        ELECTRIC,
        pickup=6.0,
    ),
    "twin_diesel": Engine(
        "Twin-unit diesel",
        1995,
        1.0,
        10,
        3_600_000.0,
        H,
        DIESEL,
        pickup=28.0,
        tank=150.0,
    ),
    "modern_electric": Engine(
        "Modern electric", 2000, 3.2, 8, 7_200_000.0, G, ELECTRIC, pickup=4.0
    ),
    "super_diesel": Engine(
        "Super-heavy diesel",
        2040,
        1.1,
        12,
        10_800_000.0,
        H,
        DIESEL,
        pickup=25.0,
        tank=200.0,
    ),
    "maglev": Engine(
        "Maglev",
        2030,
        500.0 / KMH_PER_SPEED,
        10,
        10_000_000.0,
        X,
        MAGLEV,
        pickup=5.0,
    ),
}


# The supply tile and level that open each engine for sale (railroads.sim.tiles): the level its tile
# first reaches nearest the engine's year when the world goes at its own pace, one level apart
# within a line. Steam engines come with coal,
# heavy freight engines with steel, the wood-burning American with wood, expresses with
# passengers, diesels with oil and electrics with copper.
OPENED_BY: Mapping[str, tuple[Tile, int]] = MappingProxyType(
    {
        "penydarren": (Tile.COAL, 1),
        "catch_me": (Tile.COAL, 3),
        "salamanca": (Tile.COAL, 4),
        "puffing_billy": (Tile.COAL, 5),
        "locomotion": (Tile.COAL, 6),
        "rocket": (Tile.COAL, 7),
        "lion": (Tile.COAL, 8),
        "planet": (Tile.COAL, 9),
        "tank": (Tile.COAL, 9),
        "goods": (Tile.STEEL, 11),
        "american": (Tile.WOOD, 12),
        "single": (Tile.PASSENGERS, 12),
        "light_tank": (Tile.COAL, 13),
        "mogul": (Tile.STEEL, 13),
        "atlantic": (Tile.PASSENGERS, 18),
        "mallet": (Tile.STEEL, 19),
        "pacific": (Tile.COAL, 20),
        "boxcab": (Tile.COPPER, 21),
        "railbus": (Tile.OIL, 22),
        "streamliner": (Tile.PASSENGERS, 23),
        "big_boy": (Tile.STEEL, 23),
        "mixed_diesel": (Tile.OIL, 24),
        "heavy_diesel": (Tile.OIL, 26),
        "bullet": (Tile.PASSENGERS, 26),
        "express": (Tile.COPPER, 26),
        "sprinter": (Tile.OIL, 29),
        "high_speed": (Tile.PASSENGERS, 29),
        "twin_diesel": (Tile.OIL, 30),
        "modern_electric": (Tile.COPPER, 30),
        "maglev": (Tile.PASSENGERS, 34),
        "super_diesel": (Tile.OIL, 35),
    }
)


FIRST_ENGINE = "planet"


def successor(name: str) -> str | None:
    """The first newer engine that ends an engine's sale (Engine.replaces) once it can be
    bought; None for the last of its line."""
    newer = [other for other in ENGINES if ENGINES[other].replaces(ENGINES[name])]
    return min(newer, key=lambda other: ENGINES[other].year, default=None)


type Point = tuple[float, float]


# How much of some cargo came from each station; None where that is not known.
type Parcels = dict[Cell | None, float]


def take_parcels(parcels: Parcels, amount: float, total: float) -> Parcels:
    """Takes `amount` evenly out of parcels of cargo holding `total` in all."""
    if total <= 0 or amount <= 0:
        return {}
    share = min(1.0, amount / total)
    taken = {origin: held * share for origin, held in parcels.items() if held > 0}
    for origin, part in taken.items():
        parcels[origin] -= part
    unknown = amount - sum(taken.values())
    if unknown > 1e-9:
        taken[None] = taken.get(None, 0.0) + unknown
    return taken


@dataclass
class Car:
    """A car for one good, how much of it it carries, and where that came from. A general car
    takes any good of its kind, one at a time: `good` is the one it carries, or carried last."""

    good: Good
    load: float = 0.0
    origins: Parcels = field(default_factory=dict)
    capacity: float = 0.0
    # An express's dining car, which carries nothing.
    dining: bool = False
    general: General | None = None
    # What it weighs empty, in tonnes: its good's tare_of at its car size unless given.
    tare: float = 0.0
    # The generation of cars the stop asked for it in; 0 for the newest its engine pulls.
    generation: int = 0

    def __post_init__(self) -> None:
        if not self.capacity:
            self.capacity = car_size(self.good)
        if not self.tare:
            self.tare = tare_of(self.good)

    @property
    def cargo(self) -> Cargo:
        """What a stop asks for to get this car: its kind of goods if it is general."""
        return self.general or self.good

    @property
    def tonnes(self) -> float:
        """What it weighs now, in tonnes."""
        return self.tare + self.load * unit_tonnes(self.good)

    def fill(self, parcels: Parcels) -> None:
        for origin, amount in parcels.items():
            self.origins[origin] = self.origins.get(origin, 0.0) + amount
            self.load += amount

    def empty(self, amount: float) -> Parcels:
        taken = take_parcels(self.origins, amount, self.load)
        self.load -= amount
        return taken


@dataclass
class Stop:
    """A station a train calls at, and the goods of the cars it pulls when it leaves for the
    next stop."""

    cell: Cell
    cars: list[Cargo] = field(default_factory=list)
    # Whether the train waits here until its cars are full.
    wait_for_load: bool = False
    # Whether the train only takes on fuel here, for a leg longer than it carries fuel for: it
    # unloads and loads nothing and goes on with the cars it came with to the next stop that
    # isn't one, where its passengers and mail are bound.
    refuel_only: bool = False
    # The generation of cars asked for each car, in step with `cars`; 0, or none given, for the
    # newest its engine pulls.
    generations: list[int] = field(default_factory=list)
    # Whether the game put this refuelling stop in its schedule, rather than the player
    # (railroads.sim.schedules).
    auto: bool = False

    def add_car(self, cargo: Cargo, most: int = MAX_CARS, generation: int = 0) -> bool:
        if len(self.cars) >= most:
            return False
        del self.generations[len(self.cars) :]
        self.generations += [0] * (len(self.cars) - len(self.generations))
        self.cars.append(cargo)
        self.generations.append(generation)
        return True

    def generation_at(self, index: int) -> int:
        """The generation asked for the car at `index`: 0 for the newest."""
        return self.generations[index] if index < len(self.generations) else 0

    def asked(self) -> list[tuple[Cargo, int]]:
        """Each car's cargo with the generation asked for it."""
        return [(cargo, self.generation_at(index)) for index, cargo in enumerate(self.cars)]

    def remove_car(self, index: int) -> None:
        self.cars.pop(index)
        if index < len(self.generations):
            self.generations.pop(index)

    def clear_cars(self) -> None:
        self.cars.clear()
        self.generations.clear()

    def keep_cars(self, most: int) -> None:
        """Drops the cars past the first `most`."""
        del self.cars[most:]
        del self.generations[most:]


# What a train waiting for a load does with its station's engine shed (Train.shed).
TO_SHED, IN_SHED, FROM_SHED = "to shed", "in shed", "from shed"


@dataclass
class Train:
    # What the simulation knows it by, as bought; the player sees it by its title
    # (railroads.sim.train_names), or by `label` once they name it.
    name: str
    stops: list[Stop] = field(default_factory=list)
    # The cars it pulls now.
    cars: list[Car] = field(default_factory=list)
    # Which stop it is heading for, the way there as points on the map, and how far along it is.
    heading: int = 1
    # Whether it has called at the stop it is leaving, taking on that stop's cars, and how many
    # stops it has reached since it last set off from its first.
    called: bool = False
    arrivals: int = 0
    # How much longer it will wait at the stop it is leaving for its cars to fill, in months.
    waiting: float = 0.0
    # The station whose cars it pulls, where they were filled.
    loaded_at: Cell | None = None
    path: list[Point] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    # How far along the path each of its points lies, with the path it was worked out for
    # (TrackNetwork.ends).
    path_ends: tuple[list[Point], list[float]] | None = None
    # The cells outside stations the path crosses, with the path they were worked out for
    # (railroads.sim.signals.passes).
    path_passes: tuple[list[Point], list[Pass]] | None = None
    along: float = 0.0
    status: str = "Waiting for stops"
    engine: str = FIRST_ENGINE
    # How fast it runs as a share of its full speed, which it picks up after setting off.
    pace: float = 0.0
    # The train holding the track it waits for at a signal, if any, and how long it has stood at
    # signals since it last set off from a stop, in months.
    giving_way: str = ""
    waited: float = 0.0
    # The track it holds (railroads.sim.signals), with the path it holds it on; how far along
    # that path the way is set for it; and its place in the queue for the way on, 0 when it
    # waits for none.
    claims: list[Claim] = field(default_factory=list)
    claims_for: list[Point] | None = None
    cleared: float = 0.0
    asked: int = 0
    # The track of quadruple track it last took, which it keeps to through junctions: the slow
    # one (railroads.sim.signals.SLOW_LANE) until it takes one.
    lane: int = 1
    # The version of the railway (Constructions.railway_version) its way was last found to run
    # on laid track in.
    laid_for: int = -1
    # Whether track on its way was taken up with none left leading on, so it stands where it is.
    stranded: bool = False
    # Whether the player put it first in every queue for the way on until it reaches its next
    # stop.
    first: bool = False
    # How far the way set for it reached, and how many spells it had waited at a signal, when it
    # last looked for the quickest platform of the station it is bound for
    # (railroads.sim.railway.Railway._choose_platform).
    platform_check: tuple[float, int] = (-1.0, -1)
    # Whether it set off from a station whose engine shed had its coal, and whose sand tower had
    # its sand, this month (railroads.sim.facilities).
    fuelled: bool = False
    sanded: bool = False
    # The platform of its station the train stands at, which it sets off from: the station's own
    # cell, or one of the platforms added to it; None for the station's own cell.
    platform: Cell | None = None
    # Where it is with waiting for a load in its station's engine shed: on its way there
    # (TO_SHED), standing in it (IN_SHED), back on its way to load (FROM_SHED), or "" for none.
    shed: str = ""
    # The name the player gave it, if any.
    label: str = ""
    # The schedule it runs (railroads.sim.schedules), and the schedule and version of it its
    # stops were copied from.
    schedule: str = ""
    taken: tuple[str, int] = ("", -1)
    # How many months longer it is off the map on its way to the first stop of a new schedule.
    away: float = 0.0
    # What its round so far earned: its cargo's worth where it was unloaded (as
    # railroads.sim.railway.Railway.cargo_worth reckons it) less its fuel and car trips; and what
    # its last whole round on its schedule earned, None before it finished one.
    takings: float = 0.0
    last_round: float | None = None
    # What the ride to its next stop has earned so far, reckoned the same way.
    leg: float = 0.0

    @property
    def load(self) -> dict[Good, float]:
        """What the train carries, by good."""
        carried: dict[Good, float] = {}
        for car in self.cars:
            if car.load > 0:
                carried[car.good] = carried.get(car.good, 0.0) + car.load
        return carried

    @property
    def tonnes(self) -> float:
        """What its cars weigh, in tonnes."""
        return sum(car.tonnes for car in self.cars)

    @property
    def full(self) -> bool:
        return all(car.dining or car.load >= car.capacity - 1e-9 for car in self.cars)

    @property
    def length(self) -> float:
        """How long the train is on the map, engine and cars, in cells."""
        return ENGINE_LENGTH + CAR_LENGTH * len(self.cars)

    def speed_up(self, months: float, pickup: float) -> float:
        """Picks up speed for `months`, evenly from a stand to full speed in `pickup` months;
        returns its average pace meanwhile."""
        start = self.pace
        if start >= 1.0 or months <= 0:
            return start
        rise = months / pickup
        if start + rise <= 1.0:
            self.pace = start + rise
            return start + rise / 2
        rising = (1.0 - start) * pickup
        self.pace = 1.0
        return (rising * (start + 1.0) / 2 + months - rising) / months

    def onward(self, index: int) -> int:
        """Where among its stops the train is next bound after the stop at `index`: the next
        one that isn't only for refuelling, or simply the next if every other one is."""
        count = len(self.stops)
        for step in range(1, count + 1):
            following = (index + step) % count
            if not self.stops[following].refuel_only:
                return following
        return (index + 1) % count

    def before(self, index: int) -> int:
        """The stop the cars that reach the stop at `index` were taken on at: the last one
        before it that isn't only for refuelling, or simply the one before if every other one
        is."""
        count = len(self.stops)
        for step in range(1, count + 1):
            earlier = (index - step) % count
            if not self.stops[earlier].refuel_only:
                return earlier
        return (index - 1) % count

    def restart(self) -> None:
        """Sends the train back to its first stop after its stops change."""
        self.platform = None
        self.heading = 1
        self.called = False
        self.arrivals = 0
        self.waiting = 0.0
        self.path = []
        self.pace = 0.0
        self.giving_way = ""
        self.waited = 0.0
        self.first = False

    def cars_changed(self, index: int) -> None:
        """Takes in new cars for the stop at `index`. A train standing at that stop calls there
        again to swap its cars; otherwise it keeps going and takes them the next time it calls
        there."""
        if self.path or not self.stops:
            return
        if (self.heading - 1) % len(self.stops) == index % len(self.stops):
            self.called = False
            self.waiting = 0.0

    def move(self, move: Move) -> None:
        """Moves the train, its stops and where its goods came from onto the moved map."""
        for stop in self.stops:
            stop.cell = move.cell(stop.cell)
        if self.loaded_at is not None:
            self.loaded_at = move.cell(self.loaded_at)
        if self.platform is not None:
            self.platform = move.cell(self.platform)
        for car in self.cars:
            car.origins = move.parcels(car.origins)
        self.path = [move.point(point) for point in self.path]
        if self.path_ends is not None:
            points, lengths = self.path_ends
            self.path_ends = ([move.point(point) for point in points], lengths)
