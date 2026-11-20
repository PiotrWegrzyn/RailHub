"""Supply and demand tiles: how the world's making and wanting of goods grows, year by year.

Every good belongs to a tile (TILE_GOODS), and every tile stands on two boards: supply, how much
of its goods buildings make, and demand, how much of them ordinary buyers want. Every year from
FIRST_YEAR SUPPLY_PICKS supply tiles and DEMAND_PICKS demand tiles go up a level: the ones the
player picked, the lowest for the rest. No tile on a board may get more than MOST_APART levels
above the lowest one on it. Each level adds a flat share of the tile's base, not compounded: on
supply the share its level's period gives it (SUPPLY_STEPS, PERIODS), on demand always the same
(DEMAND_STEPS). A tile CLOSES at its level: it goes no higher, can't be picked and no longer holds
the others back.

Once every open supply tile reaches the level of the next era (ERAS), the era comes with the next
generation of wagons and its new tiles, on both boards at the lowest level on that board. A
good, the buildings that make it and those there only to buy it come into the game with its era:
its tile's, or a later one of its own (GOOD_ERAS).

What a unit of a tile's goods earns is its PRICE LIST: (demand ÷ supply) ** PRICE_POWER of its
base, worked out from the levels alone, so it changes only when a tile goes up a level, and the
player sees what a pick does to it before making it.

A game's tiles are on the boards of the year it is in (Tiles); everything else asks for a year's
(of_year), from the game in play (activate) or, with none, as the world grows by itself, every
year the lowest tiles.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import FITS, MAKES, NEEDS, Good, Need
from railroads.sim.year_caches import forget


class Tile(Enum):
    COAL = "coal"
    ORE = "ore and stone"
    WOOD = "wood"
    FARMING = "farming"
    ANIMALS = "animals"
    FOOD = "food and drink"
    HOUSEHOLD = "textiles and household"
    PASSENGERS = "passengers"
    MAIL = "mail and press"
    LUXURIES = "luxuries"
    OIL = "oil and fuel"
    STEEL = "steel and metals"
    MACHINES = "machines and tools"
    CHEMICALS = "chemicals and building"
    COPPER = "copper and rubber"
    VEHICLES = "vehicles"
    MODERN = "modern goods"


class Board(Enum):
    SUPPLY = "supply"
    DEMAND = "demand"


G = Good
TILE_GOODS: Mapping[Tile, tuple[Good, ...]] = MappingProxyType(
    {
        Tile.COAL: (G.COAL,),
        Tile.ORE: (G.IRON, G.CLAY, G.LIMESTONE, G.SAND, G.SALT),
        Tile.WOOD: (G.LOGS, G.PLANKS, G.WOOD_CHIPS, G.BARRELS),
        Tile.FARMING: (
            G.WHEAT,
            G.POTATOES,
            G.RICE,
            G.SOYBEANS,
            G.SUGAR,
            G.HOPS,
            G.FLAX,
            G.APPLES,
            G.ORANGES,
            G.BANANAS,
            G.DATES,
            G.GRAPES,
            G.COFFEE,
            G.TEA,
        ),
        Tile.ANIMALS: (G.CATTLE, G.PIGS, G.SHEEP, G.CHICKENS, G.EGGS, G.MILK, G.FISH),
        Tile.FOOD: (
            G.BREAD,
            G.MEAT,
            G.SAUSAGES,
            G.DAIRY,
            G.CANS,
            G.REFINED_SUGAR,
            G.JUICE,
            G.BEER,
            G.WINE,
            G.VODKA,
            G.SAKE,
            G.MISO,
            G.TOFU,
            G.SOY_SAUCE,
            G.DRIED_FISH,
            G.HONEY,
        ),
        Tile.HOUSEHOLD: (
            G.WOOL,
            G.COTTON,
            G.CLOTHES,
            G.LEATHER,
            G.SHOES,
            G.FURNITURE,
            G.POTS,
            G.STOVE_TILES,
            G.PORCELAIN,
            G.BRICKS,
            G.GLASS,
            G.WAX,
            G.GOODS,
        ),
        Tile.PASSENGERS: (G.PASSENGERS,),
        Tile.MAIL: (G.MAIL, G.PAPER, G.NEWSPAPERS),
        Tile.LUXURIES: (G.GOLD, G.DIAMONDS, G.INCENSE, G.LUXURIES),
        Tile.OIL: (G.OIL, G.FUEL, G.GAS),
        Tile.STEEL: (G.STEEL, G.CASTINGS),
        Tile.MACHINES: (G.TOOLS, G.MACHINERY, G.GUNS),
        Tile.CHEMICALS: (G.CHEMICALS, G.CEMENT, G.FERTILISER),
        Tile.COPPER: (G.COPPER, G.RUBBER),
        Tile.VEHICLES: (G.CARS, G.PLANES, G.TANKS),
        Tile.MODERN: (G.PLASTICS, G.APPLIANCES, G.ELECTRONICS),
    }
)
TILE_OF: Mapping[Good, Tile] = MappingProxyType(
    {good: tile for tile, goods in TILE_GOODS.items() for good in goods}
)

FIRST_YEAR = 1804
SUPPLY_PICKS = 2
DEMAND_PICKS = 1
MOST_APART = 2
PRICE_POWER = 0.8


@dataclass(frozen=True, slots=True)
class Era:
    # The level every open supply tile must reach for it to come.
    level: int
    # The generation of wagons it brings.
    wagons: int
    tiles: tuple[Tile, ...]


T = Tile
ERAS: tuple[Era, ...] = (
    Era(
        1,
        1,
        (
            T.COAL,
            T.ORE,
            T.WOOD,
            T.FARMING,
            T.ANIMALS,
            T.FOOD,
            T.HOUSEHOLD,
            T.PASSENGERS,
            T.MAIL,
            T.LUXURIES,
        ),
    ),
    Era(9, 2, (T.OIL, T.STEEL)),
    Era(14, 3, (T.MACHINES, T.CHEMICALS)),
    Era(18, 4, (T.COPPER, T.VEHICLES)),
    Era(22, 5, ()),
    Era(25, 6, (T.MODERN,)),
    Era(27, 7, ()),
)
CLOSES: Mapping[Tile, int] = MappingProxyType({T.COAL: 30})

# The supply level each period begins at; a level counts in the last period begun below it.
PERIODS = (1, 8, 14, 17, 21, 25, 30)
# The share of its base every supply level adds to a tile, by period.
SUPPLY_STEPS: Mapping[Tile, tuple[float, ...]] = MappingProxyType(
    {
        T.COAL: (0.10, 0.40, 0.80, 0.60, 1.50, 1.00, 0.0),
        T.ORE: (0.10, 0.30, 0.80, 0.60, 1.50, 1.20, 0.40),
        T.WOOD: (0.10, 0.30, 0.40, 0.40, 0.60, 0.60, 0.40),
        T.FARMING: (0.05, 0.10, 0.15, 0.15, 0.30, 0.60, 0.30),
        T.ANIMALS: (0.05, 0.10, 0.15, 0.15, 0.40, 0.60, 0.30),
        T.FOOD: (0.05, 0.20, 0.30, 0.40, 0.60, 0.80, 0.40),
        T.HOUSEHOLD: (0.20, 0.40, 0.50, 0.40, 0.60, 0.80, 0.60),
        T.PASSENGERS: (0.10, 0.40, 0.60, 0.40, 0.30, 0.30, 0.30),
        T.MAIL: (0.05, 0.30, 0.60, 0.40, 0.40, 0.20, 0.10),
        T.LUXURIES: (0.05, 0.10, 0.20, 0.20, 0.30, 0.40, 0.40),
        T.OIL: (0.0, 0.60, 1.00, 1.50, 2.00, 1.50, 0.20),
        T.STEEL: (0.0, 0.60, 1.20, 0.80, 1.20, 0.80, 0.30),
        T.MACHINES: (0.0, 0.0, 0.60, 1.00, 1.50, 1.00, 0.50),
        T.CHEMICALS: (0.0, 0.0, 0.50, 0.80, 1.50, 1.50, 0.60),
        T.COPPER: (0.0, 0.0, 0.0, 0.80, 1.50, 1.50, 0.80),
        T.VEHICLES: (0.0, 0.0, 0.0, 0.80, 1.00, 2.00, 0.80),
        T.MODERN: (0.0, 0.0, 0.0, 0.0, 0.0, 2.00, 1.50),
    }
)
# The share of its base every demand level adds to what ordinary buyers want of a tile.
DEMAND_STEPS: Mapping[Tile, float] = MappingProxyType(
    {
        T.COAL: 0.10,
        T.ORE: 0.10,
        T.WOOD: 0.10,
        T.FARMING: 0.05,
        T.ANIMALS: 0.05,
        T.FOOD: 0.05,
        T.HOUSEHOLD: 0.15,
        T.PASSENGERS: 0.15,
        T.MAIL: 0.10,
        T.LUXURIES: 0.20,
        T.OIL: 0.30,
        T.STEEL: 0.15,
        T.MACHINES: 0.20,
        T.CHEMICALS: 0.20,
        T.COPPER: 0.25,
        T.VEHICLES: 0.30,
        T.MODERN: 0.30,
    }
)

# The buyers a demand tile's levels open: each kind of building and the need it starts to take,
# with the tile and the level that open it.
K = BuildingKind
_DIESEL = (
    K.FARM,
    K.POTATO_FARM,
    K.RICE_PADDY,
    K.SOY_FIELD,
    K.MINE,
    K.IRON_MINE,
    K.COPPER_MINE,
    K.GOLD_MINE,
    K.DIAMOND_MINE,
    K.SALT_WORKS,
    K.CLAY_PIT,
    K.LIMESTONE_QUARRY,
    K.SAND_PIT,
    K.LOGGING_CAMP,
    K.SAWMILL,
)
UNLOCKS: Mapping[tuple[BuildingKind, Need], tuple[Tile, int]] = MappingProxyType(
    {
        (K.MINE, Need.PLANKS): (T.WOOD, 2),
        **{
            (kind, Need.WHEAT): (T.FARMING, 2)
            for kind in (K.CATTLE_RANCH, K.PIG_FARM, K.CHICKEN_FARM, K.SHEEP_FARM)
        },
        (K.FARM, Need.LIMESTONE): (T.ORE, 3),
        **{
            (kind, Need.BARRELS): (T.WOOD, 4) for kind in (K.BREWERY, K.BIG_BREWERY, K.MEGA_BREWERY)
        },
        (K.MINE, Need.MACHINERY): (T.MACHINES, 8),
        (K.IRON_MINE, Need.MACHINERY): (T.MACHINES, 8),
        (K.FARM, Need.MACHINERY): (T.MACHINES, 10),
        (K.SHEEP_FARM, Need.MACHINERY): (T.MACHINES, 10),
        (K.FARM, Need.FERTILISER): (T.CHEMICALS, 10),
        **{(kind, Need.FUEL): (T.OIL, 11) for kind in _DIESEL},
    }
)


def supply_factor(tile: Tile, entry: int, level: int) -> float:
    """How many times its base a tile's goods are made at a supply level, from the level it came
    at: every level above adds its period's share."""
    steps = SUPPLY_STEPS[tile]
    top = min(level, CLOSES.get(tile, level))
    total = 1.0
    for step in range(entry + 1, top + 1):
        period = max(i for i, start in enumerate(PERIODS) if start < step)
        total += steps[period]
    return total


def demand_factor(tile: Tile, entry: int, level: int) -> float:
    """How many times its base ordinary buyers want of a tile's goods at a demand level."""
    return 1.0 + DEMAND_STEPS[tile] * (level - entry)


def price(supply: float, demand: float) -> float:
    """How many times its base price a unit earns with so much made and wanted."""
    return (demand / supply) ** PRICE_POWER


@dataclass(frozen=True, slots=True)
class Year:
    """The boards of one year: every tile there is with its levels, what they make of its goods
    and the era."""

    supply: Mapping[Tile, int]
    demand: Mapping[Tile, int]
    made: Mapping[Tile, float]
    wanted: Mapping[Tile, float]
    era: int

    def make_factor(self, good: Good) -> float:
        return self.made.get(TILE_OF[good], 1.0)

    def want_factor(self, need: Need) -> float:
        """How many times its base ordinary buyers want of a need: the average over the tiles of
        the goods meeting it."""
        tiles = {TILE_OF[good] for good in FITS[need]}
        return sum(self.wanted.get(tile, 1.0) for tile in tiles) / len(tiles) if tiles else 1.0

    def price(self, good: Good) -> float:
        tile = TILE_OF[good]
        return price(self.made.get(tile, 1.0), self.wanted.get(tile, 1.0))

    @property
    def wagons(self) -> int:
        return ERAS[self.era - 1].wagons


BASE = Year(
    MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), 1
)


@dataclass
class Tiles:
    """A game's boards, the picks made for next year and every year's boards so far."""

    supply: dict[Tile, int] = field(default_factory=lambda: dict.fromkeys(ERAS[0].tiles, 1))
    demand: dict[Tile, int] = field(default_factory=lambda: dict.fromkeys(ERAS[0].tiles, 1))
    supply_entry: dict[Tile, int] = field(default_factory=lambda: dict.fromkeys(ERAS[0].tiles, 1))
    demand_entry: dict[Tile, int] = field(default_factory=lambda: dict.fromkeys(ERAS[0].tiles, 1))
    era: int = 1
    # The year each era came in.
    era_years: dict[int, int] = field(default_factory=dict)
    years: dict[int, Year] = field(default_factory=dict)
    # The tiles picked to go up on 1 January, in order.
    picks: dict[Board, list[Tile]] = field(
        default_factory=lambda: {Board.SUPPLY: [], Board.DEMAND: []}
    )

    @classmethod
    def until(cls, year: int) -> Tiles:
        """The boards as the world grows by itself, every year the lowest tiles, up to a year."""
        tiles = cls()
        for each in range(FIRST_YEAR, year + 1):
            tiles.turn(each)
        return tiles

    @property
    def year(self) -> int:
        return max(self.years, default=FIRST_YEAR - 1)

    def of_year(self, year: int) -> Year:
        if year < FIRST_YEAR or not self.years:
            return BASE
        return self.years.get(year) or self.years[min(self.year, max(year, FIRST_YEAR))]

    def levels(self, board: Board) -> dict[Tile, int]:
        return self.supply if board == Board.SUPPLY else self.demand

    def open_tiles(self, board: Board, levels: Mapping[Tile, int] | None = None) -> list[Tile]:
        """The tiles that can still go up on a board: on supply, those not closed."""
        levels = self.levels(board) if levels is None else levels
        if board == Board.DEMAND:
            return list(levels)
        return [tile for tile in levels if levels[tile] < CLOSES.get(tile, levels[tile] + 1)]

    def can_raise(self, board: Board, tile: Tile, levels: Mapping[Tile, int] | None = None) -> bool:
        """Whether a tile can go up a level on a board: it is open, and no higher than
        MOST_APART above the lowest open one once up."""
        levels = self.levels(board) if levels is None else levels
        open_tiles = self.open_tiles(board, levels)
        if tile not in open_tiles:
            return False
        return levels[tile] + 1 - min(levels[t] for t in open_tiles) <= MOST_APART

    def lowest(self, board: Board, levels: Mapping[Tile, int] | None = None) -> Tile:
        levels = self.levels(board) if levels is None else levels
        return min(self.open_tiles(board, levels), key=lambda tile: levels[tile])

    def picks_left(self, board: Board) -> int:
        return (SUPPLY_PICKS if board == Board.SUPPLY else DEMAND_PICKS) - len(self.picks[board])

    def planned(self, board: Board) -> dict[Tile, int]:
        """A board's levels once the tiles picked so far go up."""
        levels = dict(self.levels(board))
        for tile in self.picks[board]:
            levels[tile] += 1
        return levels

    def can_pick(self, board: Board, tile: Tile) -> bool:
        return self.picks_left(board) > 0 and self.can_raise(board, tile, self.planned(board))

    def pick(self, board: Board, tile: Tile) -> bool:
        """Picks a tile to go up on 1 January, if it still can."""
        if not self.can_pick(board, tile):
            return False
        self.picks[board].append(tile)
        return True

    def unpick(self, board: Board, tile: Tile) -> None:
        """Takes a pick back, with any made after it that it allowed."""
        picks = self.picks[board]
        if tile in picks:
            kept = picks[: len(picks) - 1 - picks[::-1].index(tile)]
            self.picks[board] = []
            for each in kept:
                self.pick(board, each)

    def price_after(self, board: Board, tile: Tile) -> float:
        """What a unit of a tile's goods would earn next year if it went up a level on a board
        on top of the picks so far."""
        supply = self.planned(Board.SUPPLY)
        demand = self.planned(Board.DEMAND)
        (supply if board == Board.SUPPLY else demand)[tile] += 1
        return price(
            supply_factor(tile, self.supply_entry[tile], supply[tile]),
            demand_factor(tile, self.demand_entry[tile], demand[tile]),
        )

    def planned_price(self, tile: Tile) -> float:
        """What a unit of a tile's goods will earn next year once the picks so far go up."""
        return price(
            supply_factor(tile, self.supply_entry[tile], self.planned(Board.SUPPLY)[tile]),
            demand_factor(tile, self.demand_entry[tile], self.planned(Board.DEMAND)[tile]),
        )

    def turn(self, year: int) -> list[str]:
        """Begins a year: the tiles picked go up, the lowest for the picks left, and an era
        comes once every open supply tile is at its level. Returns what that brought, in words."""
        news = []
        for board in Board:
            levels = self.levels(board)
            for tile in self.picks[board]:
                if self.can_raise(board, tile):
                    levels[tile] += 1
            for _ in range(self.picks_left(board)):
                levels[self.lowest(board)] += 1
            self.picks[board] = []
        while self.era < len(ERAS):
            supply_low = min(self.supply[tile] for tile in self.open_tiles(Board.SUPPLY))
            coming = ERAS[self.era]
            if supply_low < coming.level:
                break
            self.era += 1
            self.era_years[self.era] = year
            demand_low = min(self.demand.values())
            for tile in coming.tiles:
                self.supply[tile] = self.supply_entry[tile] = supply_low
                self.demand[tile] = self.demand_entry[tile] = demand_low
            news.append(era_news(coming))
        self.years[year] = self._snapshot()
        return news

    def _snapshot(self) -> Year:
        return Year(
            MappingProxyType(dict(self.supply)),
            MappingProxyType(dict(self.demand)),
            MappingProxyType(
                {t: supply_factor(t, self.supply_entry[t], lv) for t, lv in self.supply.items()}
            ),
            MappingProxyType(
                {t: demand_factor(t, self.demand_entry[t], lv) for t, lv in self.demand.items()}
            ),
            self.era,
        )


def era_news(era: Era) -> str:
    number = ERAS.index(era) + 1
    new = ", ".join(tile.value for tile in era.tiles)
    tiles = f" and new tiles: {new}" if new else ""
    goods = [good.value for good in Good if good_era(good) == number]
    made = f" From now on buildings make {', '.join(goods)}." if goods else ""
    return f"A new era: wagons of generation {era.wagons}{tiles}.{made}"


# The world as it grows by itself, far enough for any year a game asks about.
_WORLD = Tiles.until(FIRST_YEAR)
_active: Tiles | None = None


def world() -> Tiles:
    return _active if _active is not None else _WORLD


def of_year(year: int) -> Year:
    """The boards of a year: of the game in play, or as the world grows by itself."""
    tiles = world()
    if year > tiles.year and tiles is _WORLD:
        for each in range(tiles.year + 1, year + 1):
            tiles.turn(each)
    return tiles.of_year(year)


def activate(tiles: Tiles | None) -> None:
    """Makes a game's boards the ones every year is asked of; None for the world growing by
    itself."""
    global _active
    _active = tiles
    forget()


def goods_of(tiles: Iterable[Tile]) -> list[Good]:
    return [good for tile in tiles for good in TILE_GOODS[tile]]


def opens(kind: BuildingKind, need: Need, year: int) -> bool:
    """Whether a kind of building takes a need in a year as far as the demand board goes: always,
    unless a tile opens it (UNLOCKS) and is not at its level yet."""
    unlock = UNLOCKS.get((kind, need))
    if unlock is None:
        return True
    tile, level = unlock
    return of_year(year).demand.get(tile, 0) >= level


def opened_from(kind: BuildingKind, need: Need) -> str:
    """From when a kind of building takes a need, in a few words: the demand tile level that opens
    it, the era of its goods, its year, or "" for always."""
    from railroads.sim.goods import need_year

    unlock = UNLOCKS.get((kind, need))
    if unlock is not None:
        tile, level = unlock
        return f"{tile.value} demand {level}"
    era = need_era(need)
    if era > 1:
        return f"era {era}"
    year = need_year(kind, need)
    return str(year) if year else ""


# The era each tile comes in, and goods that come in a later era than their tile.
TILE_ERAS: Mapping[Tile, int] = MappingProxyType(
    {tile: number for number, era in enumerate(ERAS, 1) for tile in era.tiles}
)
GOOD_ERAS: Mapping[Good, int] = MappingProxyType({G.PLANES: 5, G.TANKS: 5})
# Buildings there only to buy a later era's goods, which come in with them.
ERA_BUYERS = frozenset({K.CAR_DEALER})


def good_era(good: Good) -> int:
    return GOOD_ERAS.get(good, TILE_ERAS[TILE_OF[good]])


def need_era(need: Need) -> int:
    """The era a need comes in: that of the first good meeting it."""
    return min(good_era(good) for good in FITS[need])


def kind_era(kind: BuildingKind) -> int:
    """The era a kind of building comes in: that of the first good it makes, or of the last it
    must buy if it is there only for those; the first era for the rest."""
    made = MAKES.get(kind)
    if made:
        return min(good_era(good) for good in made)
    if kind in ERA_BUYERS:
        return max(need_era(need) for need in NEEDS[kind])
    return 1


def era_come(era: int, year: int) -> bool:
    """Whether an era has come by a year: the first one always has."""
    return era <= 1 or of_year(year).era >= era


def supply_step(tile: Tile, level: int) -> float:
    """The share of its base a tile's next supply level adds over this one; nothing once closed."""
    step = level + 1
    if step > CLOSES.get(tile, step):
        return 0.0
    period = max(i for i, start in enumerate(PERIODS) if start < step)
    return SUPPLY_STEPS[tile][period]


def unlocks_of(tile: Tile) -> list[tuple[int, BuildingKind, Need]]:
    """Every buyer a demand tile opens, by the level that opens it."""
    return sorted(
        ((level, kind, need) for (kind, need), (of, level) in UNLOCKS.items() if of == tile),
        key=lambda unlock: (unlock[0], unlock[1].value, unlock[2].value),
    )
