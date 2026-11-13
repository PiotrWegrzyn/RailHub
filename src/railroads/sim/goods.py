"""The goods of the game (railroads.sim.economy trades them): what every building makes and
needs, what each good earns, and the order a month's trade goes in.

Buildings need needs (NEEDS), each met by one or more goods (FITS): any food good feeds a house,
any fruit fills a cannery, wool, cotton or flax a textile mill. Some needs start only in later
years (NEED_YEARS), a few only for one kind of building (KIND_NEED_YEARS). Farms, orchards, date
palms, banana plantations, sugar fields and potato farms grow food (MAKES); mines dig coal, lumber
mills saw lumber, pits and quarries dig clay, limestone and sand, ports trade goods.

Processors turn their inputs into something else by a recipe (RECIPES): steel mills coal and
iron ore into steel, canneries fruit and steel into cans, factories lumber and steel into goods,
big factories steel and copper, mega factories steel, copper and machinery. An input whose need
starts later is left out of a recipe until its year.

What a building needs is split in two. Its demands (demands_of) are what it must get to be
satisfied; a home's are only food and heating (HOMES). Its bonus goods (bonus_of, from BONUS_NEEDS) it can do without, but each one it gets
raises its satisfaction above what its demands give (railroads.sim.economy). Processors have no
bonus goods: their inputs make what they make. A building that makes nothing whose every need is
a bonus good keeps them all as demands, since getting them is what it is there for. A building
that makes goods has boosts instead (boosts_of): what it can do without but makes up to BOOST_MOST
more with, as barrels for a brewery, wheat for animals and lumber for mines.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from types import MappingProxyType

from railroads.sim.constructions import BuildingKind


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
    LEATHER = "leather"
    SHOES = "shoes"
    PLASTICS = "plastics"
    APPLIANCES = "appliances"
    PAPER = "paper"
    NEWSPAPERS = "newspapers"
    REFINED_SUGAR = "refined sugar"
    PORCELAIN = "porcelain"
    MEAT = "meat"
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
    LEATHER = "leather"
    SHOES = "shoes"
    PLASTICS = "plastics"
    APPLIANCES = "appliances"
    PAPER = "paper"
    NEWSPAPERS = "newspapers"
    REFINED_SUGAR = "refined sugar"
    PORCELAIN = "porcelain"
    HEATING = "heating"
    BOILER_FUEL = "boiler fuel"
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
    Good.MEAT,
    Good.EGGS,
)

# The goods that meet each need.
FITS: Mapping[Need, tuple[Good, ...]] = MappingProxyType(
    {
        Need.FOOD: FOODS,
        Need.FRUIT: FRUITS,
        Need.FIBRE: (Good.WOOL, Good.COTTON, Good.FLAX),
        Need.LIVESTOCK: (Good.CATTLE, Good.SHEEP),
        Need.ANIMALS: (Good.CATTLE, Good.PIGS, Good.CHICKENS, Good.SHEEP),
        Need.HEATING: (Good.LUMBER, Good.COAL, Good.GAS),
        Need.BOILER_FUEL: (Good.COAL, Good.OIL, Good.GAS),
        **{need: (Good(need.value),) for need in Need if need.value in Good._value2member_map_},
    }
)

# The year needs start to be felt, where that is after the game begins.
NEED_YEARS: Mapping[Need, int] = MappingProxyType(
    {
        Need.VODKA: 1600,
        Need.NEWSPAPERS: 1600,
        Need.COAL: 1700,
        Need.COFFEE: 1700,
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
KIND_NEED_YEARS: Mapping[tuple[BuildingKind, Need], int] = MappingProxyType(
    {(BuildingKind.FARM, Need.MACHINERY): 1920}
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

# What a unit of each good earns where all the demand for it is left (railroads.sim.demand);
# goods left out earn CARGO_PRICE.
CARGO_PRICE = 20.0

PRICES: Mapping[Good, float] = MappingProxyType(
    {
        Good.WHEAT: 6,
        Good.APPLES: 8,
        Good.BANANAS: 8,
        Good.DATES: 8,
        Good.ORANGES: 8,
        Good.SUGAR: 8,
        Good.FISH: 8,
        Good.CANS: 14,
        Good.BREAD: 8,
        Good.CLAY: 5,
        Good.LIMESTONE: 5,
        Good.SAND: 4,
        Good.POTATOES: 6,
        Good.CATTLE: 20,
        Good.PIGS: 14,
        Good.CHICKENS: 5,
        Good.EGGS: 4,
        Good.SHEEP: 15,
        Good.MEAT: 16,
        Good.FLAX: 12,
        Good.HOPS: 10,
        Good.MACHINERY: 60,
        Good.BRICKS: 8,
        Good.CEMENT: 12,
        Good.GLASS: 20,
        Good.BEER: 15,
        Good.VODKA: 30,
        Good.LEATHER: 25,
        Good.SHOES: 35,
        Good.PLASTICS: 25,
        Good.APPLIANCES: 150,
        Good.PAPER: 15,
        Good.NEWSPAPERS: 10,
        Good.REFINED_SUGAR: 14,
        Good.PORCELAIN: 60,
        Good.COAL: 7,
        Good.LUMBER: 7,
        Good.IRON: 9,
        Good.SALT: 8,
        Good.WOOL: 10,
        Good.GRAPES: 10,
        Good.OIL: 12,
        Good.COTTON: 12,
        Good.COPPER: 14,
        Good.RUBBER: 20,
        Good.COFFEE: 25,
        Good.STEEL: 18,
        Good.FUEL: 20,
        Good.GAS: 10,
        Good.CHEMICALS: 25,
        Good.FURNITURE: 28,
        Good.CLOTHES: 30,
        Good.TOOLS: 30,
        Good.GOODS: 30,
        Good.WINE: 40,
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


def price_of(good: Good) -> float:
    """What a unit of a good earns where all the demand for it is left."""
    return PRICES.get(good, CARGO_PRICE)


# The goods meeting each need in the order stations hand them out: the most valuable first, so
# meat feeds a house before wheat does.
HANDED_OUT: Mapping[Need, tuple[Good, ...]] = MappingProxyType(
    {need: tuple(sorted(goods, key=price_of, reverse=True)) for need, goods in FITS.items()}
)

# Passengers and mail come and go at stations (railroads.sim.growth) and are never shared out
# between buildings.
TRAVELLERS = (Good.PASSENGERS, Good.MAIL)


@dataclass(frozen=True, slots=True)
class Recipe:
    """What a processor makes, the most of it one building makes in a month, and how much of
    each input every unit of it takes; and the inputs it can do without that make it make up to
    BOOST_MOST more (boosts), and how much of each every unit it makes takes for that."""

    makes: Good
    most: float
    per_unit: Mapping[Need, float]
    boosts: Mapping[Need, float] = field(default_factory=dict)

    def inputs(self, year: int) -> Mapping[Need, float]:
        """What every unit takes in a given year: an input whose need starts later is left out
        until then."""
        return _inputs(self, year)


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


# How much more a building makes with all of what boosts it (boosts_of): its share of its boosts
# times this.
BOOST_MOST = 1.0

RECIPES: Mapping[BuildingKind, Recipe] = MappingProxyType(
    {
        BuildingKind.STEEL_MILL: Recipe(Good.STEEL, 200, {Need.IRON: 0.5, Need.COAL: 0.5}),
        BuildingKind.GUN_FACTORY: Recipe(Good.GUNS, 100, {Need.STEEL: 0.5, Need.LUMBER: 0.25}),
        BuildingKind.TANK_FACTORY: Recipe(Good.TANKS, 20, {Need.STEEL: 6, Need.FUEL: 2}),
        BuildingKind.PLANE_FACTORY: Recipe(
            Good.PLANES, 20, {Need.STEEL: 3, Need.COPPER: 1, Need.FUEL: 1, Need.ELECTRONICS: 1}
        ),
        BuildingKind.FOOD_PLANT: Recipe(Good.CANS, 40, {Need.FRUIT: 0.25, Need.STEEL: 0.25}),
        BuildingKind.BAKERY: Recipe(Good.BREAD, 50, {Need.WHEAT: 0.2}),
        BuildingKind.FACTORY: Recipe(Good.GOODS, 30, {Need.LUMBER: 1 / 3, Need.STEEL: 1 / 3}),
        BuildingKind.BIG_FACTORY: Recipe(Good.GOODS, 90, {Need.STEEL: 1 / 3, Need.COPPER: 1 / 3}),
        BuildingKind.MEGA_FACTORY: Recipe(
            Good.GOODS, 270, {Need.STEEL: 1 / 3, Need.COPPER: 1 / 6, Need.MACHINERY: 1 / 6}
        ),
        BuildingKind.SHIPYARD: Recipe(Good.GOODS, 30, {Need.STEEL: 2 / 3, Need.LUMBER: 1 / 3}),
        BuildingKind.TEXTILE_MILL: Recipe(Good.CLOTHES, 100, {Need.FIBRE: 1.0}),
        BuildingKind.FURNITURE_MAKER: Recipe(Good.FURNITURE, 60, {Need.LUMBER: 1.0}),
        BuildingKind.TOOL_WORKS: Recipe(Good.TOOLS, 80, {Need.STEEL: 1.0}),
        BuildingKind.BARREL_MAKER: Recipe(Good.BARRELS, 40, {Need.LUMBER: 0.5}),
        BuildingKind.WINERY: Recipe(Good.WINE, 80, {Need.GRAPES: 1.0, Need.BARRELS: 0.5}),
        BuildingKind.REFINERY: Recipe(Good.FUEL, 300, {Need.OIL: 0.5}),
        BuildingKind.CHEMICAL_WORKS: Recipe(
            Good.CHEMICALS, 100, {Need.SALT: 0.5, Need.BOILER_FUEL: 0.5}
        ),
        BuildingKind.ELECTRONICS_PLANT: Recipe(
            Good.ELECTRONICS, 50, {Need.COPPER: 1.0, Need.CHEMICALS: 1.0}
        ),
        BuildingKind.CAR_FACTORY: Recipe(Good.CARS, 20, {Need.STEEL: 5.0, Need.FUEL: 2.0}),
        BuildingKind.MACHINE_WORKS: Recipe(Good.MACHINERY, 40, {Need.STEEL: 1.0, Need.COPPER: 0.5}),
        BuildingKind.BRICKWORKS: Recipe(Good.BRICKS, 100, {Need.CLAY: 0.5, Need.COAL: 0.25}),
        BuildingKind.CEMENT_WORKS: Recipe(Good.CEMENT, 100, {Need.LIMESTONE: 0.5, Need.COAL: 0.25}),
        BuildingKind.GLASSWORKS: Recipe(Good.GLASS, 60, {Need.SAND: 0.5, Need.COAL: 0.25}),
        BuildingKind.BREWERY: Recipe(Good.BEER, 160, {Need.HOPS: 0.25}, {Need.BARRELS: 0.1}),
        BuildingKind.DISTILLERY: Recipe(Good.VODKA, 40, {Need.POTATOES: 0.5, Need.GLASS: 0.1}),
        BuildingKind.TANNERY: Recipe(Good.LEATHER, 40, {Need.LIVESTOCK: 0.5}),
        BuildingKind.MEAT_FACTORY: Recipe(Good.MEAT, 150, {Need.ANIMALS: 0.2}),
        BuildingKind.SHOE_FACTORY: Recipe(Good.SHOES, 60, {Need.LEATHER: 0.5}),
        BuildingKind.PLASTICS_WORKS: Recipe(
            Good.PLASTICS, 100, {Need.OIL: 0.5, Need.CHEMICALS: 0.5}
        ),
        BuildingKind.APPLIANCE_FACTORY: Recipe(
            Good.APPLIANCES,
            30,
            {Need.STEEL: 1.0, Need.COPPER: 0.5, Need.PLASTICS: 0.5, Need.ELECTRONICS: 0.25},
        ),
        BuildingKind.PAPER_MILL: Recipe(Good.PAPER, 100, {Need.LUMBER: 0.5, Need.CHEMICALS: 0.25}),
        BuildingKind.PRINTING_WORKS: Recipe(Good.NEWSPAPERS, 3000, {Need.PAPER: 0.05}),
        BuildingKind.SUGAR_REFINERY: Recipe(
            Good.REFINED_SUGAR, 60, {Need.SUGAR: 0.5, Need.COAL: 0.1}
        ),
        BuildingKind.GASWORKS: Recipe(Good.GAS, 200, {Need.COAL: 0.5}),
        BuildingKind.PORCELAIN_WORKS: Recipe(Good.PORCELAIN, 30, {Need.CLAY: 0.5, Need.COAL: 0.25}),
    }
)

# The most each building makes in a month: raw goods from the land, and what processors make
# with all the inputs they need.
MAKES: Mapping[BuildingKind, Mapping[Good, float]] = MappingProxyType(
    {
        BuildingKind.FARM: {Good.WHEAT: 10},
        BuildingKind.ORCHARD: {Good.APPLES: 10},
        BuildingKind.DATE_PALMS: {Good.DATES: 12},
        BuildingKind.ORANGE_FARM: {Good.ORANGES: 12},
        BuildingKind.BANANAS: {Good.BANANAS: 18},
        BuildingKind.SUGAR: {Good.SUGAR: 16},
        BuildingKind.FISHERY: {Good.FISH: 16},
        BuildingKind.MINE: {Good.COAL: 100},
        BuildingKind.LUMBER_MILL: {Good.LUMBER: 20},
        BuildingKind.PORT: {Good.GOODS: 20, Good.OIL: 50, Good.GAS: 50},
        BuildingKind.IRON_MINE: {Good.IRON: 250},
        BuildingKind.COPPER_MINE: {Good.COPPER: 40},
        BuildingKind.GOLD_MINE: {Good.GOLD: 20},
        BuildingKind.DIAMOND_MINE: {Good.DIAMONDS: 10},
        BuildingKind.OIL_WELL: {Good.OIL: 150, Good.GAS: 50},
        BuildingKind.SALT_WORKS: {Good.SALT: 60},
        BuildingKind.SHEEP_FARM: {Good.WOOL: 60, Good.SHEEP: 10},
        BuildingKind.COTTON_FIELD: {Good.COTTON: 80},
        BuildingKind.VINEYARD: {Good.GRAPES: 60},
        BuildingKind.COFFEE_PLANTATION: {Good.COFFEE: 60},
        BuildingKind.RUBBER_PLANTATION: {Good.RUBBER: 60},
        BuildingKind.BEEKEEPER: {Good.HONEY: 4, Good.WAX: 10},
        BuildingKind.INCENSE_GROVE: {Good.INCENSE: 30},
        BuildingKind.CLAY_PIT: {Good.CLAY: 60},
        BuildingKind.LIMESTONE_QUARRY: {Good.LIMESTONE: 60},
        BuildingKind.SAND_PIT: {Good.SAND: 60},
        BuildingKind.POTATO_FARM: {Good.POTATOES: 24},
        BuildingKind.CATTLE_RANCH: {Good.CATTLE: 10},
        BuildingKind.PIG_FARM: {Good.PIGS: 12},
        BuildingKind.CHICKEN_FARM: {Good.CHICKENS: 20, Good.EGGS: 40},
        BuildingKind.FLAX_FIELD: {Good.FLAX: 60},
        BuildingKind.HOP_GARDEN: {Good.HOPS: 40},
        BuildingKind.SALT_CATHEDRAL: {Good.SALT: 200},
        **{kind: {recipe.makes: recipe.most} for kind, recipe in RECIPES.items()},
    }
)

# What each building needs in a month: consumers what they use, processors what they take in to
# make all they can.
NEEDS: Mapping[BuildingKind, Mapping[Need, float]] = MappingProxyType(
    {
        BuildingKind.HOUSE: {Need.FOOD: 10, Need.BEER: 1, Need.NEWSPAPERS: 1, Need.HEATING: 1},
        BuildingKind.TENEMENT: {
            Need.FOOD: 60,
            Need.SALT: 2,
            Need.GOODS: 20,
            Need.COFFEE: 2,
            Need.CLOTHES: 3,
            Need.FURNITURE: 2,
            Need.TOOLS: 10,
            Need.BEER: 3,
            Need.VODKA: 1,
            Need.SHOES: 2,
            Need.BRICKS: 5,
            Need.REFINED_SUGAR: 2,
            Need.NEWSPAPERS: 3,
            Need.HEATING: 3,
        },
        BuildingKind.FLATS: {
            Need.FOOD: 180,
            Need.SALT: 4,
            Need.GOODS: 40,
            Need.COFFEE: 4,
            Need.CLOTHES: 30,
            Need.FURNITURE: 20,
            Need.TOOLS: 10,
            Need.FUEL: 5,
            Need.CARS: 1,
            Need.BEER: 4,
            Need.VODKA: 2,
            Need.SHOES: 5,
            Need.BRICKS: 5,
            Need.CEMENT: 5,
            Need.GLASS: 5,
            Need.NEWSPAPERS: 5,
            Need.APPLIANCES: 1,
            Need.REFINED_SUGAR: 4,
            Need.HEATING: 5,
        },
        BuildingKind.SKYSCRAPER: {
            Need.GOODS: 30,
            Need.COFFEE: 10,
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
        },
        BuildingKind.FARM: {Need.MACHINERY: 1},
        BuildingKind.MINE: {Need.LUMBER: 20},
        BuildingKind.RAILWAY_WORKS: {Need.STEEL: 200},
        BuildingKind.POWER_PLANT: {Need.BOILER_FUEL: 200},
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
        BuildingKind.SHEEP_FARM: {Need.WHEAT: 90},
        BuildingKind.TAVERN: {Need.BEER: 20, Need.VODKA: 10},
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
        BuildingKind.CHURCH: {Need.WAX: 10, Need.INCENSE: 10},
        BuildingKind.BASILICA: {Need.WAX: 30, Need.INCENSE: 20},
        BuildingKind.CATHEDRAL: {Need.WAX: 60, Need.INCENSE: 40},
        BuildingKind.ST_PAULS: {Need.WAX: 200, Need.INCENSE: 120},
        **{
            kind: {
                need: recipe.most * amount
                for need, amount in {**recipe.per_unit, **recipe.boosts}.items()
            }
            for kind, recipe in RECIPES.items()
        },
    }
)

# The month's trade, a step at a time: a need is shared out, a kind of processor makes what it
# can. Inputs come before what is made from them, and the processors of food before houses.
PIPELINE: tuple[Need | BuildingKind, ...] = (
    Need.LUMBER,
    Need.COAL,
    BuildingKind.GASWORKS,
    Need.IRON,
    Need.COPPER,
    Need.OIL,
    Need.SALT,
    Need.GRAPES,
    Need.FIBRE,
    Need.GOLD,
    Need.DIAMONDS,
    Need.COFFEE,
    Need.RUBBER,
    Need.WAX,
    Need.INCENSE,
    Need.CLAY,
    Need.LIMESTONE,
    Need.SAND,
    Need.WHEAT,
    Need.LIVESTOCK,
    Need.HOPS,
    BuildingKind.STEEL_MILL,
    Need.STEEL,
    BuildingKind.TOOL_WORKS,
    Need.TOOLS,
    BuildingKind.MACHINE_WORKS,
    Need.MACHINERY,
    BuildingKind.REFINERY,
    Need.FUEL,
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
    BuildingKind.BAKERY,
    BuildingKind.BREWERY,
    Need.BEER,
    Need.POTATOES,
    BuildingKind.DISTILLERY,
    Need.VODKA,
    Need.SUGAR,
    BuildingKind.SUGAR_REFINERY,
    Need.REFINED_SUGAR,
    Need.ANIMALS,
    BuildingKind.MEAT_FACTORY,
    Need.FRUIT,
    BuildingKind.FOOD_PLANT,
    Need.FOOD,
    BuildingKind.FACTORY,
    BuildingKind.BIG_FACTORY,
    BuildingKind.MEGA_FACTORY,
    BuildingKind.SHIPYARD,
    Need.GOODS,
)

# Inputs that are food as well: what is left of them is gathered with the food.
FOOD_INPUTS = (Need.WHEAT, Need.POTATOES, Need.SUGAR, Need.FRUIT)


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
        Need.RUBBER,
        Need.INCENSE,
        Need.BEER,
        Need.NEWSPAPERS,
        Need.FURNITURE,
        Need.WHEAT,
        Need.LUMBER,
    }
)


@cache
def needs_of(kind: BuildingKind, year: int) -> Mapping[Need, float]:
    """What a kind of building needs in a month in a given year."""
    return MappingProxyType(
        {
            need: amount
            for need, amount in NEEDS.get(kind, {}).items()
            if NEED_YEARS.get(need, 0) <= year and KIND_NEED_YEARS.get((kind, need), 0) <= year
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
        {need: amount for need, amount in needs_of(kind, year).items() if need in BONUS_NEEDS}
    )


# Homes need only food and heating to live; everything else they want is a bonus good.
HOMES = frozenset({BuildingKind.HOUSE, BuildingKind.TENEMENT, BuildingKind.FLATS})
HOME_DEMANDS = frozenset({Need.FOOD, Need.HEATING})


@cache
def bonus_of(kind: BuildingKind, year: int) -> Mapping[Need, float]:
    """The bonus goods among what a kind of building needs in a month in a given year: for homes
    all but food and heating; those of buildings that make goods boost what they make instead
    (boosts_of)."""
    needs = needs_of(kind, year)
    if kind in HOMES:
        return MappingProxyType(
            {need: amount for need, amount in needs.items() if need not in HOME_DEMANDS}
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
