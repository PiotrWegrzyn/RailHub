"""The goods of the game (railroads.sim.economy trades them): what every building makes and
needs, what each good earns, and the order a month's trade goes in.

Buildings need needs (NEEDS), each met by one or more goods (FITS): any food good feeds a house,
any fruit fills a cannery, wool, cotton or flax a textile mill. Some needs start only in later
years (NEED_YEARS), a few only for one kind of building (KIND_NEED_YEARS). Farms, orchards, date
palms, banana plantations, sugar fields, potato farms and rice paddies grow food (MAKES); mines dig coal, lumber
mills saw lumber, pits and quarries dig clay, limestone and sand, ports trade goods.

Processors turn their inputs into something else by a recipe (RECIPES): steel mills coal and
iron ore into steel, canneries fruit and steel into cans, factories lumber and steel into goods,
big factories steel and copper, mega factories steel, copper and machinery. An input whose need
starts later is left out of a recipe until its year.

What a building needs is split in two. Its demands (demands_of) are what it must get to be
satisfied; a home's are food, common goods and heating (HOMES). Its bonus goods (bonus_of, from BONUS_NEEDS) it can do without, but each one it gets
raises its satisfaction above what its demands give (railroads.sim.economy). Processors have no
bonus goods: their inputs make what they make. A building that makes nothing whose every need is
a bonus good keeps them all as demands, since getting them is what it is there for. A building
that makes goods has boosts instead (boosts_of): what it can do without but makes up to BOOST_MOST
more with, as barrels for a brewery, wheat for animals and lumber for mines.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cache
from types import MappingProxyType

from railroads.sim.constructions import MERGED, BuildingKind


class Good(Enum):
    WHEAT = "wheat"
    APPLES = "apples"
    BANANAS = "bananas"
    DATES = "dates"
    ORANGES = "oranges"
    SUGAR = "sugar"
    HONEY = "honey"
    FISH = "fish"
    CANS = "cans"
    BREAD = "bread"
    LUMBER = "lumber"
    COAL = "coal"
    IRON = "iron ore"
    COPPER = "copper"
    GOLD = "gold"
    DIAMONDS = "diamonds"
    OIL = "oil"
    SALT = "salt"
    WOOL = "wool"
    COTTON = "cotton"
    GRAPES = "grapes"
    COFFEE = "coffee"
    RUBBER = "rubber"
    WAX = "wax"
    INCENSE = "incense"
    STEEL = "steel"
    TOOLS = "tools"
    FUEL = "fuel"
    GAS = "gas"
    CHEMICALS = "chemicals"
    FURNITURE = "furniture"
    CLOTHES = "clothes"
    BARRELS = "barrels"
    WINE = "wine"
    ELECTRONICS = "electronics"
    CARS = "cars"
    GUNS = "guns"
    TANKS = "tanks"
    PLANES = "planes"
    GOODS = "goods"
    CLAY = "clay"
    LIMESTONE = "limestone"
    SAND = "sand"
    POTATOES = "potatoes"
    RICE = "rice"
    SOYBEANS = "soybeans"
    SOY_SAUCE = "soy sauce"
    TOFU = "tofu"
    MISO = "miso"
    TEA = "tea"
    DRIED_FISH = "dried fish"
    CATTLE = "cattle"
    PIGS = "pigs"
    CHICKENS = "chickens"
    EGGS = "eggs"
    SHEEP = "sheep"
    FLAX = "flax"
    HOPS = "hops"
    MACHINERY = "machinery"
    BRICKS = "bricks"
    CEMENT = "cement"
    GLASS = "glass"
    BEER = "beer"
    VODKA = "vodka"
    SAKE = "sake"
    LEATHER = "leather"
    SHOES = "shoes"
    PLASTICS = "plastics"
    APPLIANCES = "appliances"
    PAPER = "paper"
    NEWSPAPERS = "newspapers"
    REFINED_SUGAR = "refined sugar"
    PORCELAIN = "porcelain"
    MEAT = "meat"
    COKE = "coke"
    MILK = "milk"
    DAIRY = "dairy products"
    POTS = "pots"
    STOVE_TILES = "stove tiles"
    CASTINGS = "castings"
    SAUSAGES = "sausages"
    LUXURIES = "luxury goods"
    JUICE = "juice"
    PASSENGERS = "passengers"
    MAIL = "mail"


class Need(Enum):
    FOOD = "food"
    WHEAT = "wheat"
    FRUIT = "fruit"
    FIBRE = "fibre"
    LUMBER = "lumber"
    COAL = "coal"
    IRON = "iron ore"
    COPPER = "copper"
    GOLD = "gold"
    DIAMONDS = "diamonds"
    OIL = "oil"
    SALT = "salt"
    GRAPES = "grapes"
    COFFEE = "coffee"
    RUBBER = "rubber"
    WAX = "wax"
    INCENSE = "incense"
    STEEL = "steel"
    TOOLS = "tools"
    FUEL = "fuel"
    CHEMICALS = "chemicals"
    FURNITURE = "furniture"
    CLOTHES = "clothes"
    BARRELS = "barrels"
    WINE = "wine"
    ELECTRONICS = "electronics"
    CARS = "cars"
    GUNS = "guns"
    TANKS = "tanks"
    PLANES = "planes"
    GOODS = "goods"
    SUGAR = "sugar"
    POTATOES = "potatoes"
    RICE = "rice"
    SOYBEANS = "soybeans"
    SOY_SAUCE = "soy sauce"
    HOT_DRINKS = "hot drinks"
    FISH = "fish"
    CLAY = "clay"
    LIMESTONE = "limestone"
    SAND = "sand"
    LIVESTOCK = "livestock"
    ANIMALS = "animals"
    HOPS = "hops"
    MACHINERY = "machinery"
    BRICKS = "bricks"
    CEMENT = "cement"
    GLASS = "glass"
    BEER = "beer"
    VODKA = "vodka"
    SAKE = "sake"
    LEATHER = "leather"
    SHOES = "shoes"
    PLASTICS = "plastics"
    APPLIANCES = "appliances"
    PAPER = "paper"
    NEWSPAPERS = "newspapers"
    REFINED_SUGAR = "refined sugar"
    PORCELAIN = "porcelain"
    GAS = "gas"
    HEATING = "heating"
    BOILER_FUEL = "boiler fuel"
    COKE = "coke"
    MILK = "milk"
    MEAT = "meat"
    POTS = "pots"
    STOVE_TILES = "stove tiles"
    CASTINGS = "castings"
    LUXURIES = "luxury goods"
    COMMON = "common goods"
    ALCOHOL = "alcohol"
    PASSENGERS = "passengers"
    MAIL = "mail"


FRUITS = (Good.APPLES, Good.BANANAS, Good.DATES, Good.ORANGES, Good.SUGAR)

FOODS = (
    Good.WHEAT,
    *FRUITS,
    Good.HONEY,
    Good.FISH,
    Good.CANS,
    Good.BREAD,
    Good.POTATOES,
    Good.RICE,
    Good.TOFU,
    Good.MISO,
    Good.DRIED_FISH,
    Good.MEAT,
    Good.EGGS,
    Good.MILK,
    Good.DAIRY,
    Good.SAUSAGES,
    Good.JUICE,
)

# Everyday things for the home: any of them meets a home's need for common goods.
COMMON_GOODS = (
    Good.CLOTHES,
    Good.SHOES,
    Good.FURNITURE,
    Good.POTS,
    Good.STOVE_TILES,
    Good.CASTINGS,
)


# Food made from other goods; the rest of the food is grown, caught or gathered.
PROCESSED_FOODS = (
    Good.CANS,
    Good.BREAD,
    Good.MEAT,
    Good.DAIRY,
    Good.SAUSAGES,
    Good.JUICE,
    Good.TOFU,
    Good.MISO,
    Good.DRIED_FISH,
)


class General(Enum):
    """A car for a whole kind of goods (railroads.sim.trains), which takes whichever of them pays
    best where the train goes next."""

    FOOD = "general food"
    PROCESSED_FOOD = "general processed food"
    COMMON = "general common goods"


GENERAL_GOODS: Mapping[General, tuple[Good, ...]] = MappingProxyType(
    {
        General.FOOD: tuple(good for good in FOODS if good not in PROCESSED_FOODS),
        General.PROCESSED_FOOD: PROCESSED_FOODS,
        General.COMMON: COMMON_GOODS,
    }
)

# The goods that meet each need.
FITS: Mapping[Need, tuple[Good, ...]] = MappingProxyType(
    {
        Need.FOOD: FOODS,
        Need.COMMON: COMMON_GOODS,
        Need.FRUIT: FRUITS,
        Need.ALCOHOL: (Good.BEER, Good.WINE, Good.VODKA, Good.SAKE),
        Need.HOT_DRINKS: (Good.COFFEE, Good.TEA),
        Need.FIBRE: (Good.WOOL, Good.COTTON, Good.FLAX),
        Need.LIVESTOCK: (Good.CATTLE, Good.SHEEP),
        Need.ANIMALS: (Good.CATTLE, Good.PIGS, Good.CHICKENS, Good.SHEEP),
        Need.HEATING: (Good.LUMBER, Good.COAL, Good.GAS, Good.FUEL),
        Need.BOILER_FUEL: (Good.COAL, Good.OIL, Good.GAS),
        **{need: (Good(need.value),) for need in Need if need.value in Good._value2member_map_},
    }
)

# The year needs start to be felt, where that is after the game begins.
NEED_YEARS: Mapping[Need, int] = MappingProxyType(
    {
        Need.PAPER: 1150,
        Need.VODKA: 1600,
        Need.NEWSPAPERS: 1600,
        Need.COAL: 1700,
        Need.LUXURIES: 1700,
        Need.CASTINGS: 1800,
        Need.COKE: 1850,
        Need.COFFEE: 1700,
        Need.HOT_DRINKS: 1700,
        Need.CLOTHES: 1780,
        Need.REFINED_SUGAR: 1800,
        Need.SHOES: 1850,
        Need.STEEL: 1855,
        Need.TOOLS: 1870,
        Need.MACHINERY: 1870,
        Need.COPPER: 1880,
        Need.CHEMICALS: 1880,
        Need.CEMENT: 1880,
        Need.PLASTICS: 1950,
        Need.APPLIANCES: 1950,
        Need.FUEL: 1910,
        Need.CARS: 1930,
        Need.PLANES: 1935,
        Need.TANKS: 1940,
        Need.ELECTRONICS: 1960,
    }
)

# Needs a kind of building feels only from a later year than everyone else.
# Diesel fuel lets what makes goods from the land make up to DIESEL_BOOST more: farms from
# TRACTOR_YEAR, mines, quarries and lumber mills from MACHINE_YEAR, fisheries and ports from
# MOTOR_BOAT_YEAR (DIESEL_USERS).
DIESEL_BOOST = 0.5
TRACTOR_YEAR = 1920
MACHINE_YEAR = 1920
MOTOR_BOAT_YEAR = 1910
DIESEL_USERS: Mapping[BuildingKind, tuple[float, int]] = MappingProxyType(
    {
        BuildingKind.FARM: (2.0, TRACTOR_YEAR),
        BuildingKind.POTATO_FARM: (2.0, TRACTOR_YEAR),
        BuildingKind.RICE_PADDY: (2.0, TRACTOR_YEAR),
        BuildingKind.SOY_FIELD: (2.0, TRACTOR_YEAR),
        BuildingKind.MINE: (10.0, MACHINE_YEAR),
        BuildingKind.IRON_MINE: (10.0, MACHINE_YEAR),
        BuildingKind.COPPER_MINE: (5.0, MACHINE_YEAR),
        BuildingKind.GOLD_MINE: (2.0, MACHINE_YEAR),
        BuildingKind.DIAMOND_MINE: (2.0, MACHINE_YEAR),
        BuildingKind.SALT_WORKS: (5.0, MACHINE_YEAR),
        BuildingKind.CLAY_PIT: (5.0, MACHINE_YEAR),
        BuildingKind.LIMESTONE_QUARRY: (5.0, MACHINE_YEAR),
        BuildingKind.SAND_PIT: (5.0, MACHINE_YEAR),
        BuildingKind.LUMBER_MILL: (3.0, MACHINE_YEAR),
        BuildingKind.FISHERY: (2.0, MOTOR_BOAT_YEAR),
        BuildingKind.PORT: (10.0, MOTOR_BOAT_YEAR),
    }
)
# Boosts of what makes goods from the land that make up to so much more each, whatever else
# boosts it; its other boosts share BOOST_MOST between them.
LAND_BOOST_MOST: Mapping[Need, float] = MappingProxyType({Need.FUEL: DIESEL_BOOST})
# Boosts only the buildings the player owns take, once their tech opens them
# (railroads.sim.tech.Effect.TAKES), and the most more each makes them make.
OWNED_BOOSTS: Mapping[tuple[BuildingKind, Need], float] = MappingProxyType(
    {
        (BuildingKind.FARM, Need.LIMESTONE): 0.5,
        (BuildingKind.FARM, Need.MACHINERY): 2.0,
        (BuildingKind.SHEEP_FARM, Need.MACHINERY): 1.0,
    }
)

KIND_NEED_YEARS: Mapping[tuple[BuildingKind, Need], int] = MappingProxyType(
    {
        **{(kind, Need.FUEL): year for kind, (_, year) in DIESEL_USERS.items()},
    }
)

# Goods a kind of building makes only from a later year than everyone else: ports bring in natural
# gas long after gasworks make town gas.
KIND_GOOD_YEARS: Mapping[tuple[BuildingKind, Good], int] = MappingProxyType(
    {(BuildingKind.PORT, Good.GAS): 1950}
)

# The year goods first appear, where that is after the game begins.
GOOD_YEARS: Mapping[Good, int] = MappingProxyType(
    {
        Good.TOOLS: 1870,
        Good.MACHINERY: 1870,
        Good.CHEMICALS: 1880,
        Good.CEMENT: 1880,
        Good.COPPER: 1880,
        Good.OIL: 1855,
        Good.RUBBER: 1900,
        Good.FUEL: 1856,
        Good.CARS: 1930,
        Good.PLANES: 1935,
        Good.TANKS: 1940,
        Good.GAS: 1820,
        Good.PLASTICS: 1950,
        Good.APPLIANCES: 1950,
        Good.ELECTRONICS: 1960,
    }
)

# What carrying a unit of each good to a building that takes it earns, at full satisfaction
# (freight_of); goods left out earn CARGO_PRICE. Their market price is MARKET_TIMES as much, and
# freight FREIGHT_SHARE of that.
CARGO_PRICE = 20.0
MARKET_TIMES = 5.0
FREIGHT_SHARE = 0.2

PRICES: Mapping[Good, float] = MappingProxyType(
    {
        Good.WHEAT: 6,
        Good.APPLES: 8,
        Good.BANANAS: 8,
        Good.DATES: 8,
        Good.ORANGES: 8,
        Good.SUGAR: 8,
        Good.FISH: 8,
        Good.CANS: 20,
        Good.BREAD: 6,
        Good.CLAY: 10,
        Good.LIMESTONE: 10,
        Good.SAND: 8,
        Good.POTATOES: 6,
        Good.RICE: 6,
        Good.SOYBEANS: 8,
        Good.SOY_SAUCE: 20,
        Good.TOFU: 8,
        Good.MISO: 10,
        Good.TEA: 40,
        Good.DRIED_FISH: 12,
        Good.CATTLE: 20,
        Good.PIGS: 14,
        Good.CHICKENS: 6,
        Good.EGGS: 4,
        Good.COKE: 14,
        Good.MILK: 6,
        Good.DAIRY: 9,
        Good.POTS: 10,
        Good.STOVE_TILES: 25,
        Good.CASTINGS: 30,
        Good.SAUSAGES: 14,
        Good.JUICE: 20,
        Good.LUXURIES: 300,
        Good.SHEEP: 16,
        Good.MEAT: 8,
        Good.FLAX: 24,
        Good.HOPS: 20,
        Good.MACHINERY: 180,
        Good.BRICKS: 8,
        Good.CEMENT: 12,
        Good.GLASS: 20,
        Good.BEER: 60,
        Good.VODKA: 120,
        Good.SAKE: 100,
        Good.LEATHER: 25,
        Good.SHOES: 100,
        Good.PLASTICS: 40,
        Good.APPLIANCES: 300,
        Good.PAPER: 20,
        Good.NEWSPAPERS: 10,
        Good.REFINED_SUGAR: 14,
        Good.PORCELAIN: 60,
        Good.COAL: 10,
        Good.LUMBER: 10,
        Good.IRON: 18,
        Good.SALT: 16,
        Good.WOOL: 20,
        Good.GRAPES: 20,
        Good.OIL: 24,
        Good.COTTON: 24,
        Good.COPPER: 28,
        Good.RUBBER: 40,
        Good.COFFEE: 50,
        Good.STEEL: 80,
        Good.FUEL: 20,
        Good.GAS: 10,
        Good.CHEMICALS: 25,
        Good.FURNITURE: 60,
        Good.CLOTHES: 70,
        Good.TOOLS: 160,
        Good.GOODS: 120,
        Good.WINE: 160,
        Good.HONEY: 12,
        Good.BARRELS: 15,
        Good.WAX: 30,
        Good.INCENSE: 300,
        Good.ELECTRONICS: 120,
        Good.PASSENGERS: 10,
        Good.MAIL: 25,
        Good.GOLD: 800,
        Good.DIAMONDS: 2000,
        Good.CARS: 5000,
        Good.GUNS: 150,
        Good.TANKS: 8000,
        Good.PLANES: 12000,
    }
)


# How much of its share of the market price carrying a unit of a good earns: how many units a
# car of the Planet's day held before units were things of the real world, over how many it holds
# now (railroads.sim.trains.CAR_UNITS).
FREIGHT_RATE: Mapping[Good, float] = MappingProxyType(
    dict.fromkeys((Good.POTATOES, Good.SUGAR, Good.LUMBER), 89 / 55)
    | dict.fromkeys((Good.APPLES, Good.ORANGES, Good.BANANAS, Good.DATES, Good.GRAPES), 89 / 34)
    | dict.fromkeys((Good.COFFEE, Good.REFINED_SUGAR, Good.RUBBER, Good.MAIL), 34 / 89)
    | dict.fromkeys((Good.TEA, Good.OIL, Good.FUEL, Good.GAS, Good.PLASTICS), 34 / 55)
    | dict.fromkeys((Good.COKE, Good.GUNS, Good.CHICKENS), 13 / 55)
    | dict.fromkeys((Good.COTTON, Good.FLAX, Good.HOPS, Good.HONEY), 34 / 21)
    | dict.fromkeys((Good.GLASS, Good.PORCELAIN, Good.POTS, Good.STOVE_TILES, Good.PIGS), 13 / 21)
    | dict.fromkeys((Good.TOOLS, Good.BARRELS, Good.SHEEP), 13 / 34)
    | dict.fromkeys((Good.LUXURIES, Good.INCENSE), 5 / 34)
    | dict.fromkeys((Good.CARS, Good.TANKS, Good.PLANES), 5 / 1)
    | {
        Good.ELECTRONICS: 5 / 21,
        Good.PAPER: 34 / 13,
        Good.FURNITURE: 13 / 8,
        Good.CATTLE: 21 / 8,
        Good.MACHINERY: 13 / 5,
        Good.PASSENGERS: 55 / 34,
    }
)


def price_of(good: Good) -> float:
    """A unit of a good's market price: what the player's own buildings sell it to their stations
    for, MARKET_TIMES its PRICES."""
    return MARKET_TIMES * PRICES.get(good, CARGO_PRICE)


def freight_of(good: Good) -> float:
    """What carrying a unit of a good to a building that takes it earns: FREIGHT_SHARE of its
    market price, before what the buyer's satisfaction adds (railroads.sim.economy.worth_to). Goods
    whose units fill a car faster or slower than before units were things of the real world
    earn their FREIGHT_RATE of that, so a full car earns about what it did."""
    return FREIGHT_SHARE * price_of(good) * FREIGHT_RATE.get(good, 1.0)


# The goods meeting each need in the order stations hand them out: the most valuable first, so
# meat feeds a house before wheat does.
HANDED_OUT: Mapping[Need, tuple[Good, ...]] = MappingProxyType(
    {need: tuple(sorted(goods, key=price_of, reverse=True)) for need, goods in FITS.items()}
)

# Passengers and mail come and go at stations (railroads.sim.growth) and are never shared out
# between buildings.
TRAVELLERS = (Good.PASSENGERS, Good.MAIL)


# How much more a building makes with all of what boosts it (boosts_of): its share of its boosts
# times this, unless its recipe says otherwise.
BOOST_MOST = 1.0


@dataclass(frozen=True, slots=True)
class Recipe:
    """What a processor makes, the most of it one building makes in a month, and how much of
    each input every unit of it takes; the inputs it can do without that make it make up to
    boost_most more (boosts), and how much of each every unit it makes takes for that; and the
    inputs another can stand in for (swaps), with how many times as much every unit of the
    other inputs makes with it: plastic for glass at a juice works makes twice the juice from
    the same fruit."""

    makes: Good
    most: float
    per_unit: Mapping[Need, float]
    boosts: Mapping[Need, float] = field(default_factory=dict)
    swaps: Mapping[Need, tuple[Need, float]] = field(default_factory=dict)
    boost_most: float = BOOST_MOST

    def inputs(self, year: int) -> Mapping[Need, float]:
        """What every unit takes in a given year: an input whose need starts later is left out
        until then."""
        return _inputs(self, year)

    def swaps_in(self, year: int) -> dict[Need, tuple[Need, float]]:
        """The swaps of inputs there are in a given year: both needs must have started."""
        inputs = self.inputs(year)
        return {
            need: (swap, times)
            for need, (swap, times) in self.swaps.items()
            if need in inputs and NEED_YEARS.get(swap, 0) <= year
        }

    def takes(self, year: int) -> set[Need]:
        """Every input it takes in a given year, those that stand in for others included."""
        return set(self.inputs(year)) | {swap for swap, _ in self.swaps_in(year).values()}

    @property
    def every_need(self) -> set[Need]:
        """Every need it ever takes in: inputs, boosts and the inputs standing in for others."""
        return {*self.per_unit, *self.boosts, *(swap for swap, _ in self.swaps.values())}

    def needs(self) -> dict[Need, float]:
        """What one building takes in a month to make the most it can: of an input another can
        stand in for, as much of either."""
        needs = {
            need: self.most * amount for need, amount in {**self.per_unit, **self.boosts}.items()
        }
        for need, (swap, times) in self.swaps.items():
            needs[swap] = self.most * self.per_unit[need] / times
        return needs


_RECIPE_INPUTS: dict[tuple[int, int], tuple[Recipe, Mapping[Need, float]]] = {}


def _inputs(recipe: Recipe, year: int) -> Mapping[Need, float]:
    """A recipe's inputs in a year, worked out once for each year and kept with the recipe."""
    kept = _RECIPE_INPUTS.get((id(recipe), year))
    if kept is None or kept[0] is not recipe:
        inputs = {
            need: amount
            for need, amount in recipe.per_unit.items()
            if NEED_YEARS.get(need, 0) <= year
        }
        kept = _RECIPE_INPUTS[(id(recipe), year)] = (recipe, MappingProxyType(inputs))
    return kept[1]


# How many works of a kind merge into the next size up (railroads.sim.constructions.MERGED):
# makers of food three, others two; and how much more than they made apart the merged one makes.
MERGING = 2
FOOD_MERGING = 3
MERGE_BONUS = 1.25


def merging(recipe: Recipe) -> int:
    """How many works making by this recipe merge into one the next size up."""
    return FOOD_MERGING if recipe.makes in FOODS else MERGING


def _with_merged(recipes: dict[BuildingKind, Recipe]) -> dict[BuildingKind, Recipe]:
    """The recipes with those of every big and mega works: the same inputs a unit, making as
    much as the works merged into it, and MERGE_BONUS more with each size."""
    merged = dict(recipes)
    for kind, sizes in MERGED.items():
        recipe = recipes[kind]
        for step, grown in enumerate(sizes, start=1):
            most = recipe.most * (merging(recipe) * MERGE_BONUS) ** step
            merged[grown] = replace(recipe, most=most)
    return merged


RECIPES: Mapping[BuildingKind, Recipe] = MappingProxyType(
    _with_merged(
        {
            BuildingKind.STEEL_MILL: Recipe(
                Good.STEEL, 100, {Need.IRON: 2.5, Need.COAL: 2.5}, {Need.COKE: 0.25}
            ),
            BuildingKind.COKE_PLANT: Recipe(Good.COKE, 100, {Need.COAL: 1.0}),
            BuildingKind.POTTER: Recipe(Good.POTS, 40, {Need.CLAY: 0.5}),
            BuildingKind.TILE_WORKS: Recipe(
                Good.STOVE_TILES, 30, {Need.CLAY: 0.5, Need.SAND: 0.25}
            ),
            BuildingKind.FOUNDRY: Recipe(Good.CASTINGS, 40, {Need.IRON: 0.5, Need.SAND: 0.25}),
            BuildingKind.DAIRY: Recipe(Good.DAIRY, 120, {Need.MILK: 0.4}),
            BuildingKind.SAUSAGE_MAKER: Recipe(
                Good.SAUSAGES, 240, {Need.MEAT: 0.4, Need.SALT: 0.2}
            ),
            BuildingKind.CARPET_WEAVER: Recipe(Good.LUXURIES, 5, {Need.FIBRE: 4.0}),
            BuildingKind.CLOCK_MAKER: Recipe(Good.LUXURIES, 5, {Need.COPPER: 2.0, Need.GLASS: 1.0}),
            BuildingKind.MIRROR_MAKER: Recipe(Good.LUXURIES, 5, {Need.GLASS: 2.0, Need.GOLD: 0.1}),
            BuildingKind.GUN_FACTORY: Recipe(Good.GUNS, 100, {Need.STEEL: 0.5, Need.LUMBER: 0.25}),
            BuildingKind.TANK_FACTORY: Recipe(Good.TANKS, 20, {Need.STEEL: 6, Need.FUEL: 2}),
            BuildingKind.PLANE_FACTORY: Recipe(
                Good.PLANES, 20, {Need.STEEL: 3, Need.COPPER: 1, Need.FUEL: 1, Need.ELECTRONICS: 1}
            ),
            BuildingKind.FOOD_PLANT: Recipe(Good.CANS, 300, {Need.FRUIT: 0.3, Need.STEEL: 0.1}),
            BuildingKind.BAKERY: Recipe(
                Good.BREAD, 120, {Need.WHEAT: 0.4}, {Need.REFINED_SUGAR: 0.05}, boost_most=0.5
            ),
            BuildingKind.JUICE_WORKS: Recipe(
                Good.JUICE,
                100,
                {Need.FRUIT: 0.5, Need.GLASS: 0.5},
                {Need.REFINED_SUGAR: 0.1},
                swaps={Need.GLASS: (Need.PLASTICS, 2.0)},
                boost_most=0.5,
            ),
            BuildingKind.FACTORY: Recipe(Good.GOODS, 30, {Need.LUMBER: 1 / 3, Need.STEEL: 1 / 3}),
            BuildingKind.BIG_FACTORY: Recipe(
                Good.GOODS, 90, {Need.STEEL: 1 / 3, Need.COPPER: 1 / 3}
            ),
            BuildingKind.MEGA_FACTORY: Recipe(
                Good.GOODS, 270, {Need.STEEL: 1 / 3, Need.COPPER: 1 / 6, Need.MACHINERY: 1 / 6}
            ),
            BuildingKind.SHIPYARD: Recipe(Good.GOODS, 30, {Need.STEEL: 2 / 3, Need.LUMBER: 1 / 3}),
            BuildingKind.TEXTILE_MILL: Recipe(Good.CLOTHES, 40, {Need.FIBRE: 2.5}),
            BuildingKind.FURNITURE_MAKER: Recipe(Good.FURNITURE, 20, {Need.LUMBER: 4.0}),
            BuildingKind.TOOL_WORKS: Recipe(Good.TOOLS, 80, {Need.STEEL: 1.0}),
            BuildingKind.BARREL_MAKER: Recipe(Good.BARRELS, 40, {Need.LUMBER: 0.5}),
            BuildingKind.WINERY: Recipe(Good.WINE, 20, {Need.GRAPES: 4.0}, {Need.BARRELS: 2.0}),
            BuildingKind.REFINERY: Recipe(Good.FUEL, 300, {Need.OIL: 0.5}),
            BuildingKind.CHEMICAL_WORKS: Recipe(
                Good.CHEMICALS, 100, {Need.SALT: 0.5, Need.BOILER_FUEL: 0.5}
            ),
            BuildingKind.ELECTRONICS_PLANT: Recipe(
                Good.ELECTRONICS, 50, {Need.COPPER: 1.0, Need.CHEMICALS: 1.0}
            ),
            BuildingKind.CAR_FACTORY: Recipe(Good.CARS, 20, {Need.STEEL: 5.0, Need.FUEL: 2.0}),
            BuildingKind.MACHINE_WORKS: Recipe(
                Good.MACHINERY, 40, {Need.STEEL: 1.0, Need.COPPER: 0.5}
            ),
            BuildingKind.BRICKWORKS: Recipe(Good.BRICKS, 100, {Need.CLAY: 0.5}, {Need.COAL: 0.25}),
            BuildingKind.CEMENT_WORKS: Recipe(
                Good.CEMENT, 100, {Need.LIMESTONE: 0.5}, {Need.COAL: 0.25}
            ),
            BuildingKind.GLASSWORKS: Recipe(
                Good.GLASS, 60, {Need.SAND: 0.5}, {Need.COAL: 0.25, Need.LIMESTONE: 0.25}
            ),
            BuildingKind.BREWERY: Recipe(Good.BEER, 40, {Need.HOPS: 1.0}, {Need.BARRELS: 0.4}),
            BuildingKind.DISTILLERY: Recipe(
                Good.VODKA, 10, {Need.POTATOES: 2.0}, {Need.GLASS: 0.4}
            ),
            BuildingKind.SAKE_BREWERY: Recipe(Good.SAKE, 15, {Need.RICE: 2.0}),
            BuildingKind.SOY_SAUCE_BREWERY: Recipe(
                Good.SOY_SAUCE, 20, {Need.SOYBEANS: 0.5, Need.WHEAT: 0.5, Need.SALT: 0.2}
            ),
            BuildingKind.TOFU_MAKER: Recipe(Good.TOFU, 60, {Need.SOYBEANS: 0.4}),
            BuildingKind.MISO_MAKER: Recipe(
                Good.MISO, 60, {Need.SOYBEANS: 0.25, Need.RICE: 0.15, Need.SALT: 0.1}
            ),
            BuildingKind.FISH_DRYER: Recipe(Good.DRIED_FISH, 60, {Need.FISH: 0.4, Need.SALT: 0.1}),
            BuildingKind.TANNERY: Recipe(
                Good.LEATHER, 40, {Need.LIVESTOCK: 0.5}, {Need.LIMESTONE: 0.25}
            ),
            BuildingKind.MEAT_FACTORY: Recipe(Good.MEAT, 120, {Need.ANIMALS: 0.4}),
            BuildingKind.SHOE_FACTORY: Recipe(Good.SHOES, 20, {Need.LEATHER: 2.0}),
            BuildingKind.PLASTICS_WORKS: Recipe(
                Good.PLASTICS, 100, {Need.OIL: 0.5, Need.CHEMICALS: 0.5}
            ),
            BuildingKind.APPLIANCE_FACTORY: Recipe(
                Good.APPLIANCES,
                30,
                {Need.STEEL: 1.0, Need.COPPER: 0.5, Need.PLASTICS: 0.5, Need.ELECTRONICS: 0.25},
            ),
            BuildingKind.PAPER_MILL: Recipe(
                Good.PAPER, 100, {Need.LUMBER: 0.5, Need.CHEMICALS: 0.25}
            ),
            BuildingKind.PRINTING_WORKS: Recipe(Good.NEWSPAPERS, 3000, {Need.PAPER: 0.05}),
            BuildingKind.SUGAR_REFINERY: Recipe(
                Good.REFINED_SUGAR, 60, {Need.SUGAR: 0.5}, {Need.COAL: 0.1, Need.LIMESTONE: 0.1}
            ),
            BuildingKind.GASWORKS: Recipe(Good.GAS, 200, {Need.COAL: 0.5}),
            BuildingKind.PORCELAIN_WORKS: Recipe(
                Good.PORCELAIN, 30, {Need.CLAY: 0.5}, {Need.COAL: 0.25}
            ),
        }
    )
)

# The most each building makes in a month: raw goods from the land, and what processors make
# with all the inputs they need.
MAKES: Mapping[BuildingKind, Mapping[Good, float]] = MappingProxyType(
    {
        BuildingKind.FARM: {Good.WHEAT: 10},
        BuildingKind.ORCHARD: {Good.APPLES: 10},
        BuildingKind.DATE_PALMS: {Good.DATES: 10},
        BuildingKind.ORANGE_FARM: {Good.ORANGES: 10},
        BuildingKind.BANANAS: {Good.BANANAS: 10},
        BuildingKind.SUGAR: {Good.SUGAR: 10},
        BuildingKind.FISHERY: {Good.FISH: 10},
        BuildingKind.MINE: {Good.COAL: 100},
        BuildingKind.LUMBER_MILL: {Good.LUMBER: 20},
        BuildingKind.PORT: {Good.GOODS: 20, Good.OIL: 50, Good.GAS: 50},
        BuildingKind.IRON_MINE: {Good.IRON: 100},
        BuildingKind.COPPER_MINE: {Good.COPPER: 40},
        BuildingKind.GOLD_MINE: {Good.GOLD: 20},
        BuildingKind.DIAMOND_MINE: {Good.DIAMONDS: 10},
        BuildingKind.OIL_WELL: {Good.OIL: 150, Good.GAS: 50},
        BuildingKind.SALT_WORKS: {Good.SALT: 60},
        BuildingKind.SHEEP_FARM: {Good.WOOL: 21, Good.SHEEP: 10},
        BuildingKind.COTTON_FIELD: {Good.COTTON: 80},
        BuildingKind.VINEYARD: {Good.GRAPES: 60},
        BuildingKind.COFFEE_PLANTATION: {Good.COFFEE: 60},
        BuildingKind.RUBBER_PLANTATION: {Good.RUBBER: 60},
        BuildingKind.BEEKEEPER: {Good.HONEY: 2, Good.WAX: 10},
        BuildingKind.INCENSE_GROVE: {Good.INCENSE: 30},
        BuildingKind.CLAY_PIT: {Good.CLAY: 60},
        BuildingKind.LIMESTONE_QUARRY: {Good.LIMESTONE: 60},
        BuildingKind.SAND_PIT: {Good.SAND: 60},
        BuildingKind.POTATO_FARM: {Good.POTATOES: 10},
        BuildingKind.RICE_PADDY: {Good.RICE: 10},
        BuildingKind.SOY_FIELD: {Good.SOYBEANS: 10},
        BuildingKind.TEA_PLANTATION: {Good.TEA: 20},
        BuildingKind.CATTLE_RANCH: {Good.CATTLE: 10, Good.MILK: 20},
        BuildingKind.PIG_FARM: {Good.PIGS: 12},
        BuildingKind.CHICKEN_FARM: {Good.CHICKENS: 20, Good.EGGS: 15},
        BuildingKind.FLAX_FIELD: {Good.FLAX: 60},
        BuildingKind.HOP_GARDEN: {Good.HOPS: 40},
        BuildingKind.SALT_CATHEDRAL: {Good.SALT: 200},
        **{kind: {recipe.makes: recipe.most} for kind, recipe in RECIPES.items()},
    }
)

# What each building needs in a month: consumers what they use, processors what they take in to
# make all they can.
_NEEDS: Mapping[BuildingKind, Mapping[Need, float]] = MappingProxyType(
    {
        BuildingKind.HOUSE: {
            Need.FOOD: 20,
            Need.SOY_SAUCE: 0.5,
            Need.HEATING: 3,
            Need.BEER: 1,
            Need.NEWSPAPERS: 1,
            Need.COMMON: 5,
            Need.VODKA: 0.5,
            Need.REFINED_SUGAR: 0.5,
            Need.BRICKS: 0.5,
            Need.GLASS: 0.2,
            Need.LIMESTONE: 0.5,
        },
        BuildingKind.MANOR: {
            Need.FOOD: 100,
            Need.HEATING: 10,
            Need.WINE: 10,
            Need.FURNITURE: 5,
            Need.LUXURIES: 3,
        },
        BuildingKind.THEATRE: {Need.CLOTHES: 10, Need.GLASS: 5, Need.PAPER: 5},
        BuildingKind.HOSPITAL: {
            Need.FOOD: 50,
            Need.HEATING: 5,
            Need.CHEMICALS: 10,
            Need.GLASS: 5,
        },
        BuildingKind.TENEMENT: {
            Need.FOOD: 120,
            Need.SOY_SAUCE: 5,
            Need.SALT: 10,
            Need.GOODS: 20,
            Need.HOT_DRINKS: 10,
            Need.COMMON: 60,
            Need.TOOLS: 10,
            Need.BEER: 10,
            Need.VODKA: 10,
            Need.BRICKS: 10,
            Need.GLASS: 10,
            Need.CEMENT: 10,
            Need.REFINED_SUGAR: 10,
            Need.NEWSPAPERS: 10,
            Need.HEATING: 20,
            Need.LIMESTONE: 10,
            Need.LUXURIES: 1,
        },
        BuildingKind.FLATS: {
            Need.FOOD: 400,
            Need.SOY_SAUCE: 15,
            Need.SALT: 30,
            Need.GOODS: 40,
            Need.HOT_DRINKS: 30,
            Need.COMMON: 300,
            Need.TOOLS: 10,
            Need.FUEL: 30,
            Need.CARS: 3,
            Need.BEER: 30,
            Need.VODKA: 30,
            Need.BRICKS: 50,
            Need.CEMENT: 50,
            Need.GLASS: 50,
            Need.NEWSPAPERS: 30,
            Need.APPLIANCES: 5,
            Need.REFINED_SUGAR: 30,
            Need.HEATING: 80,
            Need.LUXURIES: 5,
        },
        BuildingKind.SKYSCRAPER: {
            Need.GOODS: 30,
            Need.HOT_DRINKS: 10,
            Need.CLOTHES: 5,
            Need.FURNITURE: 5,
            Need.WINE: 20,
            Need.FUEL: 20,
            Need.ELECTRONICS: 30,
            Need.GLASS: 10,
            Need.CEMENT: 10,
            Need.NEWSPAPERS: 20,
            Need.PAPER: 20,
            Need.APPLIANCES: 5,
            Need.PORCELAIN: 2,
            Need.LUXURIES: 2,
        },
        BuildingKind.FARM: {Need.MACHINERY: 5, Need.LIMESTONE: 10},
        BuildingKind.MINE: {Need.LUMBER: 20},
        BuildingKind.RAILWAY_WORKS: {Need.STEEL: 200},
        BuildingKind.POWER_PLANT: {Need.COAL: 200},
        BuildingKind.DIESEL_POWER_PLANT: {Need.FUEL: 100},
        BuildingKind.GAS_POWER_PLANT: {Need.GAS: 150},
        BuildingKind.SPACE_CENTER: {Need.ELECTRONICS: 300, Need.FUEL: 1000, Need.FOOD: 300},
        BuildingKind.PEAR_HQ: {Need.ELECTRONICS: 500, Need.FOOD: 1000},
        BuildingKind.ROYAL_PALACE: {
            Need.FOOD: 200,
            Need.WINE: 50,
            Need.PORCELAIN: 20,
            Need.FURNITURE: 20,
        },
        BuildingKind.SPA_RESORT: {
            Need.FOOD: 100,
            Need.WINE: 20,
            Need.BEER: 20,
            Need.NEWSPAPERS: 20,
        },
        BuildingKind.WORLDS_FAIR: {Need.GOODS: 200, Need.GLASS: 100, Need.STEEL: 100},
        BuildingKind.FILM_STUDIO: {Need.CHEMICALS: 50, Need.CLOTHES: 50, Need.ELECTRONICS: 30},
        BuildingKind.AMUSEMENT_PARK: {Need.FOOD: 200, Need.BEER: 100, Need.FUEL: 50},
        BuildingKind.OLYMPIC_STADIUM: {Need.FOOD: 500, Need.BEER: 200, Need.NEWSPAPERS: 200},
        BuildingKind.CATTLE_RANCH: {Need.WHEAT: 120},
        BuildingKind.PIG_FARM: {Need.WHEAT: 60},
        BuildingKind.CHICKEN_FARM: {Need.WHEAT: 30},
        BuildingKind.SHEEP_FARM: {Need.WHEAT: 90, Need.MACHINERY: 5},
        BuildingKind.TAVERN: {Need.ALCOHOL: 22, Need.FOOD: 20},
        BuildingKind.CLUB: {Need.ALCOHOL: 88, Need.CHEMICALS: 5},
        BuildingKind.SUSHI_HUT: {Need.FISH: 60, Need.RICE: 60, Need.SOY_SAUCE: 10},
        BuildingKind.DEPARTMENT_STORE: {
            Need.CLOTHES: 20,
            Need.SHOES: 20,
            Need.FURNITURE: 10,
            Need.PORCELAIN: 5,
            Need.APPLIANCES: 10,
        },
        BuildingKind.HOTEL: {Need.FOOD: 80, Need.WINE: 10, Need.BEER: 10, Need.NEWSPAPERS: 10},
        BuildingKind.JEWELLER: {Need.GOLD: 20, Need.DIAMONDS: 10},
        BuildingKind.CAR_DEALER: {Need.CARS: 20, Need.RUBBER: 10},
        BuildingKind.OUTPOST: {Need.GUNS: 60, Need.CARS: 30},
        BuildingKind.BASE: {Need.GUNS: 180, Need.CARS: 90, Need.PLANES: 30},
        BuildingKind.HEADQUARTERS: {
            Need.GUNS: 450,
            Need.CARS: 180,
            Need.PLANES: 90,
            Need.TANKS: 90,
        },
        BuildingKind.AIRPORT: {Need.FUEL: 40},
        BuildingKind.FILLING_STATION: {Need.FUEL: 40},
        BuildingKind.CHURCH: {Need.WAX: 10, Need.INCENSE: 10},
        BuildingKind.BASILICA: {Need.WAX: 30, Need.INCENSE: 20},
        BuildingKind.CATHEDRAL: {Need.WAX: 60, Need.INCENSE: 40},
        BuildingKind.ST_PAULS: {Need.WAX: 200, Need.INCENSE: 120},
        **{kind: recipe.needs() for kind, recipe in RECIPES.items()},
    }
)
NEEDS: Mapping[BuildingKind, Mapping[Need, float]] = MappingProxyType(
    {
        **_NEEDS,
        **{
            kind: {**_NEEDS.get(kind, {}), Need.FUEL: amount}
            for kind, (amount, _) in DIESEL_USERS.items()
        },
    }
)

# The month's trade, a step at a time: a need is shared out, a kind of processor makes what it
# can. Inputs come before what is made from them, and the processors of food before houses.
_PIPELINE: tuple[Need | BuildingKind, ...] = (
    Need.OIL,
    BuildingKind.REFINERY,
    Need.FUEL,
    Need.LUMBER,
    Need.COAL,
    BuildingKind.GASWORKS,
    Need.GAS,
    BuildingKind.COKE_PLANT,
    Need.COKE,
    Need.IRON,
    Need.COPPER,
    Need.SALT,
    Need.GRAPES,
    Need.FIBRE,
    Need.GOLD,
    Need.DIAMONDS,
    Need.COFFEE,
    Need.HOT_DRINKS,
    Need.RUBBER,
    Need.WAX,
    Need.INCENSE,
    Need.CLAY,
    Need.LIMESTONE,
    Need.SAND,
    BuildingKind.POTTER,
    Need.POTS,
    BuildingKind.TILE_WORKS,
    Need.STOVE_TILES,
    BuildingKind.FOUNDRY,
    Need.CASTINGS,
    Need.WHEAT,
    Need.LIVESTOCK,
    Need.HOPS,
    BuildingKind.STEEL_MILL,
    Need.STEEL,
    BuildingKind.TOOL_WORKS,
    Need.TOOLS,
    BuildingKind.MACHINE_WORKS,
    Need.MACHINERY,
    Need.BOILER_FUEL,
    BuildingKind.CHEMICAL_WORKS,
    Need.CHEMICALS,
    BuildingKind.PLASTICS_WORKS,
    Need.PLASTICS,
    BuildingKind.BRICKWORKS,
    Need.BRICKS,
    BuildingKind.CEMENT_WORKS,
    Need.CEMENT,
    BuildingKind.GLASSWORKS,
    Need.GLASS,
    BuildingKind.CARPET_WEAVER,
    BuildingKind.CLOCK_MAKER,
    BuildingKind.MIRROR_MAKER,
    Need.LUXURIES,
    BuildingKind.PORCELAIN_WORKS,
    Need.PORCELAIN,
    BuildingKind.PAPER_MILL,
    Need.PAPER,
    BuildingKind.PRINTING_WORKS,
    Need.NEWSPAPERS,
    BuildingKind.TANNERY,
    Need.LEATHER,
    BuildingKind.SHOE_FACTORY,
    Need.SHOES,
    BuildingKind.FURNITURE_MAKER,
    Need.FURNITURE,
    BuildingKind.TEXTILE_MILL,
    Need.CLOTHES,
    Need.COMMON,
    BuildingKind.BARREL_MAKER,
    Need.BARRELS,
    BuildingKind.WINERY,
    Need.WINE,
    BuildingKind.ELECTRONICS_PLANT,
    Need.ELECTRONICS,
    BuildingKind.APPLIANCE_FACTORY,
    Need.APPLIANCES,
    BuildingKind.CAR_FACTORY,
    Need.CARS,
    BuildingKind.GUN_FACTORY,
    Need.GUNS,
    BuildingKind.TANK_FACTORY,
    Need.TANKS,
    BuildingKind.PLANE_FACTORY,
    Need.PLANES,
    Need.HEATING,
    BuildingKind.BREWERY,
    Need.BEER,
    Need.POTATOES,
    Need.RICE,
    Need.FISH,
    Need.SOYBEANS,
    BuildingKind.SOY_SAUCE_BREWERY,
    Need.SOY_SAUCE,
    BuildingKind.TOFU_MAKER,
    BuildingKind.MISO_MAKER,
    BuildingKind.FISH_DRYER,
    BuildingKind.DISTILLERY,
    Need.VODKA,
    BuildingKind.SAKE_BREWERY,
    Need.SAKE,
    Need.ALCOHOL,
    Need.SUGAR,
    BuildingKind.SUGAR_REFINERY,
    Need.REFINED_SUGAR,
    BuildingKind.BAKERY,
    Need.ANIMALS,
    BuildingKind.MEAT_FACTORY,
    Need.MEAT,
    BuildingKind.SAUSAGE_MAKER,
    Need.MILK,
    BuildingKind.DAIRY,
    Need.FRUIT,
    BuildingKind.FOOD_PLANT,
    BuildingKind.JUICE_WORKS,
    Need.FOOD,
    BuildingKind.FACTORY,
    BuildingKind.BIG_FACTORY,
    BuildingKind.MEGA_FACTORY,
    BuildingKind.SHIPYARD,
    Need.GOODS,
)
PIPELINE: tuple[Need | BuildingKind, ...] = tuple(
    kind
    for step in _PIPELINE
    for kind in (
        (step, *MERGED[step]) if isinstance(step, BuildingKind) and step in MERGED else (step,)
    )
)


@cache
def makes_of(kind: BuildingKind, year: int) -> Mapping[Good, float]:
    """The most a kind of building makes of each good in a month in a given year: nothing of a
    good that does not exist yet."""
    return MappingProxyType(
        {
            good: amount
            for good, amount in MAKES.get(kind, {}).items()
            if KIND_GOOD_YEARS.get((kind, good), GOOD_YEARS.get(good, 0)) <= year
        }
    )


# Needs buildings can do without but that raise their satisfaction: every special resource
# (railroads.mapgen.deposits), beer, newspapers and furniture, the wheat animals are fed and the
# lumber mines prop their shafts with. Wool, cotton and fish meet
# needs that ordinary goods meet too (fibre and food), so those stay demands.
BONUS_NEEDS = frozenset(
    {
        Need.IRON,
        Need.COPPER,
        Need.GOLD,
        Need.DIAMONDS,
        Need.OIL,
        Need.SALT,
        Need.GRAPES,
        Need.COFFEE,
        Need.HOT_DRINKS,
        Need.SOY_SAUCE,
        Need.RUBBER,
        Need.INCENSE,
        Need.BEER,
        Need.NEWSPAPERS,
        Need.FURNITURE,
        Need.WHEAT,
        Need.LUMBER,
        Need.LIMESTONE,
    }
)


def need_year(kind: BuildingKind, need: Need) -> int:
    """The year a kind of building starts to need something: once there is any of it, and its
    own year for that need if it has one."""
    return max(NEED_YEARS.get(need, 0), KIND_NEED_YEARS.get((kind, need), 0))


@cache
def needs_of(kind: BuildingKind, year: int) -> Mapping[Need, float]:
    """What a kind of building needs in a month in a given year."""
    return MappingProxyType(
        {
            need: amount
            for need, amount in NEEDS.get(kind, {}).items()
            if need_year(kind, need) <= year
        }
    )


@cache
def boosts_of(kind: BuildingKind, year: int) -> Mapping[Need, float]:
    """What a kind of building that makes goods can do without but makes up to BOOST_MOST more
    with, and how much of each it needs in a month for all of that: a processor's recipe boosts,
    and the bonus goods of one that makes goods from the land, such as wheat for its animals."""
    recipe = RECIPES.get(kind)
    if recipe is not None:
        boosts = {need: recipe.most * amount for need, amount in recipe.boosts.items()}
        needs = needs_of(kind, year)
        return MappingProxyType({need: amount for need, amount in boosts.items() if need in needs})
    if kind not in MAKES:
        return MappingProxyType({})
    return MappingProxyType(
        {
            need: amount
            for need, amount in needs_of(kind, year).items()
            if need in BONUS_NEEDS or need in LAND_BOOST_MOST or (kind, need) in OWNED_BOOSTS
        }
    )


# Homes need food, common goods and heating to live, more of each the bigger they grow; everything
# else they want is a bonus good.
HOMES = frozenset({BuildingKind.HOUSE, BuildingKind.TENEMENT, BuildingKind.FLATS})
HOME_DEMANDS = frozenset({Need.FOOD, Need.COMMON, Need.HEATING})
# A house must get only food and heating; common goods are a bonus for it.
HOUSE_DEMANDS = frozenset({Need.FOOD, Need.HEATING})
# Needs whose last part is only a bonus: a home must eat half the food it wants, and the other
# half counts as a bonus good (railroads.sim.economy): 10 of a house's 20, 60 of a tenement's 120
# and 200 of a block's 400.
BONUS_PART: Mapping[tuple[BuildingKind, Need], float] = MappingProxyType(
    {
        (BuildingKind.HOUSE, Need.FOOD): 10.0,
        (BuildingKind.TENEMENT, Need.FOOD): 60.0,
        (BuildingKind.FLATS, Need.FOOD): 200.0,
    }
)


# Venues need alcohol of any kind; a tavern takes food as a bonus, and a club a little of
# whatever chemicals it can get. A sushi hut needs fish and rice, and takes soy sauce as a bonus.
VENUE_BONUSES: Mapping[BuildingKind, frozenset[Need]] = MappingProxyType(
    {
        BuildingKind.TAVERN: frozenset({Need.FOOD}),
        BuildingKind.CLUB: frozenset({Need.CHEMICALS}),
        BuildingKind.SUSHI_HUT: frozenset({Need.SOY_SAUCE}),
    }
)


@cache
def bonus_of(kind: BuildingKind, year: int) -> Mapping[Need, float]:
    """The bonus goods among what a kind of building needs in a month in a given year: for homes
    all but food, common goods and heating; those of buildings that make goods boost what they make instead
    (boosts_of)."""
    needs = needs_of(kind, year)
    if kind in VENUE_BONUSES:
        return MappingProxyType(
            {need: amount for need, amount in needs.items() if need in VENUE_BONUSES[kind]}
        )
    if kind in HOMES:
        demands = HOUSE_DEMANDS if kind == BuildingKind.HOUSE else HOME_DEMANDS
        return MappingProxyType(
            {need: amount for need, amount in needs.items() if need not in demands}
        )
    bonus = {need: amount for need, amount in needs.items() if need in BONUS_NEEDS}
    if kind in MAKES or len(bonus) == len(needs):
        return MappingProxyType({})
    return MappingProxyType(bonus)


@cache
def demands_of(kind: BuildingKind, year: int) -> Mapping[Need, float]:
    """What a kind of building must get in a month in a given year to be satisfied: its needs
    that are neither bonus goods nor boosts."""
    left_out = {**bonus_of(kind, year), **boosts_of(kind, year)}
    return MappingProxyType(
        {need: amount for need, amount in needs_of(kind, year).items() if need not in left_out}
    )
