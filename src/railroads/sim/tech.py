"""Tech trees: what the player learns by running their business, and the bonuses it brings.

Each tree gains experience in dollars (Research.earn): from what the cargo of its goods earns as
buildings take it out of the player's stations (railroads.sim.railway), passengers and mail
from what they pay as they get off, the business trees also from the share of what the
player's own buildings sell (railroads.sim.ownership), commerce also from what shops take out
of the player's stations, and any tree from buildings growing a rank around a station (growth
rewards). Every level a tree reaches gives
one point to spend on one of its nodes, once the nodes it builds on are taken. A node's effect
adds up with the others of its kind: in its own tree's buildings and goods, or everywhere.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import MAKES, General, Good, Need

# Dollars of experience a tree needs for its first level, kept low so the first node comes early;
# the second level needs LEVEL_STEP more and each next one LEVEL_GROWTH times the step before.
FIRST_LEVEL = 5_000.0
LEVEL_STEP = 80_000.0
LEVEL_GROWTH = 1.6


class Tree(Enum):
    FOOD = "food and drink"
    METALS = "metals and machines"
    FUEL = "fuel and minerals"
    CRAFT = "craft and textiles"
    MODERN = "modern industry"
    COMMERCE = "commerce"
    PASSENGERS = "passengers and mail"


class Effect(Enum):
    # The player's buildings make, and take in, this share more.
    PRODUCTION = "production"
    # The player's share of what their buildings sell rises by this much.
    SHARE = "owner's share"
    # Cargo of the tree's goods earns this share more.
    PRICE = "price"
    # Every station's stockpile holds this many more goods.
    STORAGE = "station storage"
    # Buildings cost this share less to buy.
    DISCOUNT = "purchase discount"
    # Trains run this share faster.
    SPEED = "train speed"
    # Trains pull this many more cars.
    PULL = "cars per train"
    # Stations, track, car trips and the fuel engines burn cost this share less.
    UPKEEP = "upkeep"
    # The player's buildings count this share more towards growth around them.
    GROWTH = "growth"
    # Passenger coaches and mail vans carry this share more.
    COACHES = "coach capacity"
    # Stations hold this many more waiting passengers and pieces of mail.
    WAITING = "waiting room"
    # Each bonus good a building gets all of adds this much more to its satisfaction
    # (railroads.sim.economy).
    BONUS_GOODS = "per bonus good"
    # Fuel bought anywhere but out of a station's stockpile costs this much less over the market
    # price (railroads.sim.fuel_market).
    FUEL_MARKUP = "retail fuel markup"
    # Every car holds this many sizes more (railroads.sim.trains.CAR_SIZES).
    CAR_SIZE = "car size"
    # The buildings of a kind the player owns take a good as a boost (railroads.sim.goods.OWNED_BOOSTS).
    TAKES = "takes"
    # Opens the engines (railroads.sim.trains) that need it, once their year has come, or a
    # general car (GENERAL_CARS).
    UNLOCK = "unlocks"


@dataclass(frozen=True, slots=True)
class Node:
    key: str
    name: str
    effect: Effect
    amount: float = 0.0
    requires: tuple[str, ...] = ()
    # For UNLOCK, the engines it opens, in a few words.
    opens: str = ""
    # Whether the effect reaches every tree's buildings and goods, not just its own.
    everywhere: bool = False
    # For TAKES, the kind of building and what it takes.
    takes: tuple[BuildingKind, Need] | None = None


def _n(key: str, name: str, effect: Effect, amount: float = 0.0, *requires: str, **kw) -> Node:
    return Node(key, name, effect, amount, requires, **kw)


E = Effect
TREES: Mapping[Tree, tuple[Node, ...]] = MappingProxyType(
    {
        Tree.FOOD: (
            _n("seed", "Better seed", E.PRODUCTION, 0.25),
            _n("silos", "Grain silos", E.STORAGE, 200),
            _n("stalls", "Market stalls", E.PRICE, 0.05, "seed"),
            _n("rotation", "Crop rotation", E.PRODUCTION, 0.25, "seed"),
            _n("villages", "Thriving villages", E.GROWTH, 0.5, "rotation"),
            _n("cold", "Cold stores", E.STORAGE, 300, "silos"),
            _n("hoppers", "Grain hoppers", E.CAR_SIZE, 1, "silos"),
            _n("coop", "Farmers' cooperative", E.SHARE, 0.05, "stalls"),
            _n(
                "mixed_vans",
                "Mixed food vans",
                E.UNLOCK,
                0.0,
                "stalls",
                opens="general food cars, taking whichever grown food pays best",
            ),
            _n(
                "provision_vans",
                "Provision vans",
                E.UNLOCK,
                0.0,
                "mixed_vans",
                opens="general processed food cars of 25, taking whichever pays best",
            ),
            _n("bread", "Branded bread", E.PRICE, 0.1, "stalls"),
            _n(
                "liming",
                "Liming the fields",
                E.TAKES,
                0.0,
                "rotation",
                takes=(BuildingKind.FARM, Need.LIMESTONE),
            ),
            _n("fertiliser", "Fertiliser", E.PRODUCTION, 0.25, "rotation"),
            _n("reefers", "Refrigerated vans", E.PRICE, 0.1, "cold", "bread"),
            _n("conglomerate", "Food conglomerate", E.SHARE, 0.05, "coop"),
            _n("tractors", "Mechanised farming", E.PRODUCTION, 0.25, "fertiliser"),
            _n(
                "farm_machinery",
                "Farm machinery",
                E.TAKES,
                0.0,
                "tractors",
                takes=(BuildingKind.FARM, Need.MACHINERY),
            ),
            _n("brand", "National brand", E.PRICE, 0.15, "reefers", "conglomerate"),
        ),
        Tree.METALS: (
            _n("bessemer", "Bessemer converters", E.PRODUCTION, 0.25),
            _n("boilers", "Heavy boilers", E.UNLOCK, opens="heavy freight engines"),
            _n("couplings", "Steel couplings", E.PULL, 1, "bessemer"),
            _n("open_hearth", "Open hearth", E.PRODUCTION, 0.25, "bessemer"),
            _n("bearings", "Roller bearings", E.SPEED, 0.05, "boilers"),
            _n("cartel", "Steel cartel", E.SHARE, 0.05, "open_hearth"),
            _n(
                "superheaters",
                "Superheaters",
                E.UNLOCK,
                0.0,
                "bearings",
                opens="newer steam engines",
            ),
            _n("drawbars", "Heavy drawbars", E.PULL, 1, "couplings"),
            _n("bogie_wagons", "Bogie wagons", E.CAR_SIZE, 1, "drawbars", everywhere=True),
            _n("rolling", "Rolling mills", E.PRODUCTION, 0.25, "open_hearth"),
            _n("streamlining", "Streamlining", E.SPEED, 0.1, "superheaters"),
            _n("works_trust", "Engineering trust", E.SHARE, 0.05, "cartel", "rolling"),
            _n("oxygen", "Basic oxygen steel", E.PRODUCTION, 0.25, "rolling"),
        ),
        Tree.FUEL: (
            _n("screening", "Coal screening", E.PRODUCTION, 0.25),
            _n("firing", "Better firing", E.UPKEEP, 0.05),
            _n("sanding", "Sand boxes", E.PULL, 1, "screening"),
            _n("depots", "Coaling depots", E.UPKEEP, 0.05, "firing"),
            _n("seams", "Deep seams", E.PRODUCTION, 0.25, "screening"),
            _n("kilns", "Ring kilns", E.PRICE, 0.1, "seams"),
            _n("grid", "Power grid", E.UPKEEP, 0.05, "depots"),
            _n("drilling", "Rotary drilling", E.PRODUCTION, 0.25, "seams"),
            _n("pipelines", "Pipelines", E.STORAGE, 300, "drilling"),
            _n("diesel", "Diesel engines", E.UNLOCK, 0.0, "drilling", opens="diesel engines"),
            _n("fuel_trust", "Fuel trust", E.SHARE, 0.1, "kilns", "drilling"),
            _n("efficiency", "Fuel efficiency", E.UPKEEP, 0.1, "grid"),
        ),
        Tree.CRAFT: (
            _n("looms", "Power looms", E.PRODUCTION, 0.25),
            _n("guilds", "Craft guilds", E.PRICE, 0.05),
            _n("patterns", "Pattern books", E.PRICE, 0.1, "guilds"),
            _n("spinning", "Spinning mules", E.PRODUCTION, 0.25, "looms"),
            _n("workshops", "Model workshops", E.GROWTH, 0.5, "spinning"),
            _n("fashion", "Fashion houses", E.PRICE, 0.15, "patterns"),
            _n(
                "parcel_vans",
                "Parcel vans",
                E.UNLOCK,
                0.0,
                "patterns",
                opens="general common goods cars, taking whichever pays best",
            ),
            _n("mills_trust", "Mill owners' trust", E.SHARE, 0.05, "spinning"),
            _n("synthetics", "Synthetic dyes", E.PRODUCTION, 0.25, "spinning"),
            _n("couture", "Couture", E.SHARE, 0.05, "fashion", "mills_trust"),
        ),
        Tree.MODERN: (
            _n("labs", "Research labs", E.PRODUCTION, 0.25),
            _n("telegraph", "Telegraph signals", E.SPEED, 0.05),
            _n(
                "traction",
                "Electric traction",
                E.UNLOCK,
                0.0,
                "telegraph",
                opens="electric engines",
            ),
            _n("assembly", "Assembly lines", E.PRODUCTION, 0.25, "labs"),
            _n("relays", "Relay interlocking", E.SPEED, 0.05, "traction"),
            _n("patents", "Patents", E.PRICE, 0.1, "labs"),
            _n("automation", "Automation", E.PRODUCTION, 0.25, "assembly"),
            _n("thyristors", "Thyristor control", E.UNLOCK, 0.0, "relays", opens="fast electrics"),
            _n("tech_trust", "Technology giant", E.SHARE, 0.1, "patents", "automation"),
            _n("computers", "Computer timetables", E.SPEED, 0.1, "thyristors"),
            _n(
                "maglev",
                "Magnetic levitation",
                E.UNLOCK,
                0.0,
                "thyristors",
                opens="maglev track and trains",
            ),
        ),
        Tree.COMMERCE: (
            _n("books", "Double-entry books", E.DISCOUNT, 0.1),
            _n("investors", "Investors", E.SHARE, 0.05, everywhere=True),
            _n("chains", "Chain stores", E.DISCOUNT, 0.1, "books"),
            _n("regular_customer", "Regular customer", E.FUEL_MARKUP, 0.25, "books"),
            _n("advertising", "Advertising", E.PRICE, 0.05, "investors", everywhere=True),
            _n("holding", "Holding company", E.SHARE, 0.05, "investors", everywhere=True),
            _n("takeovers", "Hostile takeovers", E.DISCOUNT, 0.15, "chains"),
            _n("brands", "Household brands", E.PRICE, 0.05, "advertising", everywhere=True),
            _n("empire", "Business empire", E.SHARE, 0.1, "holding", "takeovers", everywhere=True),
            # TODO: a placeholder until bonus goods get their own, stronger tech interactions,
            # which the user will come back to.
            _n("fine_living", "Fine living", E.BONUS_GOODS, 0.1, "advertising", everywhere=True),
        ),
        Tree.PASSENGERS: (
            _n("timetables", "Timetables", E.PRICE, 0.1),
            _n("coaches", "Bogie coaches", E.COACHES, 0.2),
            _n("waiting", "Waiting rooms", E.WAITING, 100, "timetables"),
            _n("dining", "Dining cars", E.PRICE, 0.1, "timetables"),
            _n("corridors", "Corridor coaches", E.COACHES, 0.2, "coaches"),
            _n("express_mail", "Express mail", E.PRICE, 0.15, "dining"),
            _n(
                "express_trains", "Express trains", E.UNLOCK, 0.0, "dining", opens="express engines"
            ),
            _n("concourse", "Grand concourses", E.WAITING, 200, "waiting"),
            _n("sleepers", "Sleeping cars", E.PRICE, 0.15, "express_mail"),
            _n("double_deck", "Double-deck coaches", E.COACHES, 0.3, "corridors"),
            _n("high_speed", "High-speed lines", E.SPEED, 0.1, "sleepers", "double_deck"),
        ),
    }
)
NODES: Mapping[str, tuple[Tree, Node]] = MappingProxyType(
    {node.key: (tree, node) for tree, nodes in TREES.items() for node in nodes}
)
_GOODS: Mapping[Tree, tuple[Good, ...]] = {
    Tree.FOOD: (
        Good.WHEAT,
        Good.APPLES,
        Good.BANANAS,
        Good.DATES,
        Good.ORANGES,
        Good.SUGAR,
        Good.HONEY,
        Good.FISH,
        Good.CANS,
        Good.BREAD,
        Good.POTATOES,
        Good.CATTLE,
        Good.PIGS,
        Good.CHICKENS,
        Good.EGGS,
        Good.SHEEP,
        Good.HOPS,
        Good.BEER,
        Good.VODKA,
        Good.REFINED_SUGAR,
        Good.MEAT,
        Good.MILK,
        Good.DAIRY,
        Good.SAUSAGES,
        Good.JUICE,
        Good.GRAPES,
        Good.WINE,
        Good.COFFEE,
        Good.SALT,
    ),
    Tree.METALS: (
        Good.IRON,
        Good.COPPER,
        Good.STEEL,
        Good.TOOLS,
        Good.MACHINERY,
        Good.GOODS,
        Good.GUNS,
        Good.TANKS,
    ),
    Tree.FUEL: (
        Good.COAL,
        Good.COKE,
        Good.OIL,
        Good.GAS,
        Good.FUEL,
        Good.SAND,
        Good.CLAY,
        Good.LIMESTONE,
        Good.BRICKS,
        Good.CEMENT,
        Good.GLASS,
        Good.STOVE_TILES,
    ),
    Tree.CRAFT: (
        Good.LUMBER,
        Good.WOOL,
        Good.COTTON,
        Good.FLAX,
        Good.CLOTHES,
        Good.LEATHER,
        Good.SHOES,
        Good.FURNITURE,
        Good.BARRELS,
        Good.PAPER,
        Good.NEWSPAPERS,
        Good.PORCELAIN,
        Good.POTS,
        Good.CASTINGS,
        Good.WAX,
        Good.INCENSE,
    ),
    Tree.MODERN: (
        Good.CHEMICALS,
        Good.ELECTRONICS,
        Good.PLASTICS,
        Good.APPLIANCES,
        Good.CARS,
        Good.PLANES,
        Good.RUBBER,
    ),
    Tree.COMMERCE: (Good.GOLD, Good.DIAMONDS, Good.LUXURIES),
    Tree.PASSENGERS: (Good.PASSENGERS, Good.MAIL),
}
# The tree each good, and each kind of building, belongs to.
GOOD_TREES: Mapping[Good, Tree] = MappingProxyType(
    {good: tree for tree, goods in _GOODS.items() for good in goods}
)
SHOPS = frozenset(
    {
        BuildingKind.TAVERN,
        BuildingKind.CLUB,
        BuildingKind.FILLING_STATION,
        BuildingKind.DEPARTMENT_STORE,
        BuildingKind.HOTEL,
        BuildingKind.JEWELLER,
        BuildingKind.CAR_DEALER,
    }
)


# The node that opens each general car (railroads.sim.trains).
GENERAL_CARS: Mapping[General, str] = MappingProxyType(
    {
        General.FOOD: "mixed_vans",
        General.PROCESSED_FOOD: "provision_vans",
        General.COMMON: "parcel_vans",
    }
)


def tree_of(kind: BuildingKind) -> Tree | None:
    """The tree a kind of building's business belongs to: shops to commerce, the rest by what
    they make; None for those that make nothing."""
    if kind in SHOPS:
        return Tree.COMMERCE
    makes = MAKES.get(kind)
    return GOOD_TREES[next(iter(makes))] if makes else None


def path_to(keys: tuple[str, ...]) -> tuple[str, ...]:
    """Every node to take to have some, each after the nodes it requires."""
    path: list[str] = []

    def visit(key: str) -> None:
        if key in path:
            return
        for need in NODES[key][1].requires:
            visit(need)
        path.append(key)

    for key in keys:
        visit(key)
    return tuple(path)


def xp_for(level: int) -> float:
    """The experience a tree needs in all to reach a level."""
    if level <= 0:
        return 0.0
    return FIRST_LEVEL + LEVEL_STEP * (LEVEL_GROWTH ** (level - 1) - 1) / (LEVEL_GROWTH - 1)


@dataclass
class Research:
    """The player's experience in every tree and the nodes they took."""

    xp: dict[Tree, float] = field(default_factory=lambda: dict.fromkeys(Tree, 0.0))
    taken: set[str] = field(default_factory=set)

    def earn(self, tree: Tree, dollars: float) -> None:
        self.xp[tree] = self.xp.get(tree, 0.0) + max(dollars, 0.0)

    def level(self, tree: Tree) -> int:
        level = 0
        while xp_for(level + 1) <= self.xp.get(tree, 0.0):
            level += 1
        return level

    def points(self, tree: Tree) -> int:
        """Points still to spend in a tree."""
        spent = sum(1 for key in self.taken if NODES[key][0] == tree)
        return self.level(tree) - spent

    def can_take(self, key: str) -> bool:
        tree, node = NODES[key]
        return (
            key not in self.taken
            and self.points(tree) > 0
            and all(need in self.taken for need in node.requires)
        )

    def take(self, key: str) -> bool:
        if not self.can_take(key):
            return False
        self.taken.add(key)
        return True

    def takes(self, kind: BuildingKind) -> frozenset[Need]:
        """The boosts the buildings of a kind the player owns take, by the nodes taken."""
        return frozenset(
            node.takes[1]
            for key in self.taken
            if (node := NODES[key][1]).takes is not None and node.takes[0] == kind
        )

    def bonus(self, effect: Effect, tree: Tree | None = None) -> float:
        """The sum of the taken nodes' effect of a kind: those of one tree and those reaching
        everywhere, or, with no tree, all of them."""
        total = 0.0
        for key in self.taken:
            of, node = NODES[key]
            if node.effect == effect and (tree is None or of == tree or node.everywhere):
                total += node.amount
        return total
