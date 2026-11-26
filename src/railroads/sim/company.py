"""What the player's company is worth, the target it is set for 1 January of END_YEAR, and going
bankrupt.

The company is worth its cash, in debt below nothing, plus the shares it owns at what they would
fetch today (railroads.sim.ownership), plus BOOK_SHARE of what its stations, track and trains
would cost to build or buy today: a station with its levels, storage and facilities, each piece
of track on its own bed, each train's engine. The game keeps one Goal: the company's worth at
the start of every month, the target its difficulty sets, when that was first reached and
whether it stood on 1 January of END_YEAR. Play goes on after that. The difficulty also sets the
cash the company starts with (START_CASHES) and how many times the boards' prices every good
fetches all game (PRICE_FACTORS). Cash below nothing at the start of BANKRUPT_MONTHS months in a
row ends the game.
"""

import weakref
from dataclasses import dataclass, field

import numpy as np

from railroads.sim.clock import END_YEAR, MONTH_NAMES
from railroads.sim.constructions import Constructions
from railroads.sim.finance import (
    LEVEL_PRICES,
    START_CASH,
    STATION_PRICE,
    TRAIN_RESALE,
    storage_price,
    track_price,
)
from railroads.sim.grids import Cell
from railroads.sim.keyed import Keyed
from railroads.sim.ownership import LOT, Owners
from railroads.sim.railway import Railway
from railroads.sim.trains import ENGINES
from railroads.words import dollars

# What stations, track and trains count for, as a share of what they cost today: what a train
# sells for.
BOOK_SHARE = TRAIN_RESALE
BANKRUPT_MONTHS = 6


class Difficulty(Keyed):
    EASY = "Easy"
    NORMAL = "Normal"
    HARD = "Hard"


# The company's worth each difficulty sets as the target for 1 January of END_YEAR.
TARGETS = {
    Difficulty.EASY: 50_000_000.0,
    Difficulty.NORMAL: 100_000_000.0,
    Difficulty.HARD: 150_000_000.0,
}

# How many times the price the supply and demand boards give every good fetches at each
# difficulty, for the whole game (railroads.sim.tiles.Tiles.markup).
PRICE_FACTORS = {
    Difficulty.EASY: 2.0,
    Difficulty.NORMAL: 1.5,
    Difficulty.HARD: 1.0,
}

# The cash each difficulty starts the company with.
START_CASHES = {
    Difficulty.EASY: 150_000.0,
    Difficulty.NORMAL: START_CASH,
    Difficulty.HARD: 80_000.0,
}


@dataclass(frozen=True)
class CompanyValue:
    cash: float = 0.0
    shares: float = 0.0
    stations: float = 0.0
    track: float = 0.0
    trains: float = 0.0

    @property
    def total(self) -> float:
        return self.cash + self.shares + self.stations + self.track + self.trains

    def parts(self) -> list[tuple[str, float]]:
        """What the worth is made of, by name."""
        return [
            ("Cash", self.cash),
            ("Shares", self.shares),
            ("Stations", self.stations),
            ("Track", self.track),
            ("Trains", self.trains),
        ]


def station_cost(railway: Railway, station: Cell) -> float:
    """What a station as it stands would cost today: the station, its levels and its storage
    upgrades; its facilities cost nothing to put up."""
    levels = sum(LEVEL_PRICES[: railway.level(station) - 1])
    goods = railway.goods.get(station)
    storage = sum(storage_price(n) for n in range(0 if goods is None else goods.storage))
    return STATION_PRICE + levels + storage


def shares_worth(owners: Owners) -> float:
    """What the shares the player owns would fetch today."""
    worth = 0.0
    for kind in owners.kinds():
        grid = owners.shares(kind)
        for row, col in np.argwhere(grid > 0).tolist():
            price = owners.lot_price((row, col), kind)
            if price is not None:
                worth += int(grid[row, col]) / LOT * price
    return worth


# What the laid track of each Constructions cost, with the railway version it was summed for:
# the worth shows every frame and its cash changes nearly every one.
_TRACK_COST: weakref.WeakKeyDictionary[Constructions, tuple[int, float]] = (
    weakref.WeakKeyDictionary()
)


def _track_cost(constructions: Constructions) -> float:
    """What every piece of track laid cost, summed again only once the railway changed."""
    kept = _TRACK_COST.get(constructions)
    if kept is None or kept[0] != constructions.railway_version:
        tile_map = constructions.tile_map
        cost = sum(track_price(piece, tile_map.tile(piece.cell)) for piece in constructions.track())
        kept = _TRACK_COST[constructions] = (constructions.railway_version, cost)
    return kept[1]


def company_value(railway: Railway) -> CompanyValue:
    """What the player's company is worth now."""
    stations = sum(station_cost(railway, station) for station in railway.stations())
    track = _track_cost(railway.constructions)
    trains = sum(ENGINES[train.engine].price for train in railway.trains)
    return CompanyValue(
        cash=railway.finances.cash,
        shares=0.0 if railway.owners is None else shares_worth(railway.owners),
        stations=stations * BOOK_SHARE,
        track=track * BOOK_SHARE,
        trains=trains * BOOK_SHARE,
    )


@dataclass(frozen=True)
class News:
    """What the game has to tell the player about the target or bankruptcy."""

    title: str
    text: str
    # Whether it is bad news, and whether it ends the game.
    bad: bool = False
    ends_game: bool = False


@dataclass
class Goal:
    difficulty: Difficulty = Difficulty.NORMAL
    # The company's worth at the start of each month of play: its year, its month from 0, and
    # the worth.
    values: list[tuple[int, int, float]] = field(default_factory=list)
    # How many months in a row have begun with the cash below nothing.
    months_in_red: int = 0
    # The date the target was first reached, whether it stood on 1 January of END_YEAR (None
    # until then), and the date the company went bankrupt.
    reached: str | None = None
    met: bool | None = None
    bankrupt: str | None = None
    # Whether the player was told the cash went below nothing since a month last began above it.
    warned: bool = False
    # What the player has yet to be told, oldest first.
    news: list[News] = field(default_factory=list)

    @property
    def target(self) -> float:
        return TARGETS[self.difficulty]

    @property
    def value(self) -> float:
        """The worth at the start of this month, nothing before the first."""
        return self.values[-1][2] if self.values else 0.0

    @property
    def months_left(self) -> int:
        """How many more months may begin with the cash below nothing before the company goes
        bankrupt."""
        return BANKRUPT_MONTHS - self.months_in_red

    def begin_month(self, year: int, month: int, value: CompanyValue) -> None:
        """Notes the company's worth and cash at the start of a month, and what that means for
        the target and bankruptcy."""
        if self.bankrupt is not None:
            return
        date = f"{MONTH_NAMES[month]} {year}"
        worth = value.total
        self.values.append((year, month, worth))
        if worth >= self.target and self.reached is None:
            self.reached = date
            self.news.append(
                News(
                    "Target reached",
                    f"Your company is worth {dollars(worth)}, past the {dollars(self.target)} "
                    f"set for 1 January {END_YEAR}. Keep it there until then.",
                )
            )
        if year >= END_YEAR and self.met is None:
            self.met = worth >= self.target
            verb = "met" if self.met else "missed"
            self.news.append(
                News(
                    f"Target {verb}",
                    f"On 1 January {END_YEAR} your company is worth {dollars(worth)}; the "
                    f"target was {dollars(self.target)}. You can play on.",
                    bad=not self.met,
                )
            )
        self.months_in_red = self.months_in_red + 1 if value.cash < 0 else 0
        if value.cash >= 0:
            self.warned = False
        if self.months_left <= 0:
            self.bankrupt = date
            self.news.append(
                News(
                    "Bankrupt",
                    f"Your cash stayed below nothing for {BANKRUPT_MONTHS} months in a row. "
                    f"The company is wound up in {date}.",
                    bad=True,
                    ends_game=True,
                )
            )

    def watch(self, cash: float) -> None:
        """Warns the player as soon as the cash goes below nothing, once until a month begins
        above it again."""
        if cash >= 0 or self.warned or self.bankrupt is not None:
            return
        self.warned = True
        self.news.append(
            News(
                "Cash below nothing",
                f"If {self.months_left} more months begin with your cash below nothing, the "
                "company goes bankrupt and the game ends.",
                bad=True,
            )
        )
